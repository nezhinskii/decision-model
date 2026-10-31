"""CPU-only cache reading and collation; no backbone loading dependencies."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import random

import torch
from torch.utils.data import IterableDataset, get_worker_info
from ..features import FeatureBatch


@dataclass
class TrainingBatch:
    """Head inputs plus supervision; targets follow features.options row order."""

    features: FeatureBatch
    targets: torch.Tensor  # [V], probabilities normalized within each question.
    context_ids: list[str]  # B context IDs.
    categories: list[str]  # Q category labels.
    tasks: list[str]  # Q task labels.
    target_kinds: list[str]  # Q hard/soft labels.
    score_values: list[list[float] | None]  # Q entries, each K_q values or None.

    def to(self, device: str | torch.device) -> "TrainingBatch":
        """Move tensors for head training while keeping IDs and labels as lists."""
        return replace(
            self, features=self.features.to(device), targets=self.targets.to(device)
        )


def collate_features(records: list[dict]) -> TrainingBatch:
    """Combine whole contexts and rebase option-to-question indices."""
    offset = 0
    links, context_links = [], []
    for context, record in enumerate(records):
        count = len(record["question_ids"])
        links.append(record["option_to_question"] + offset)
        context_links.append(torch.full((count,), context, dtype=torch.long))
        offset += count

    def flatten(key):
        return [value for record in records for value in record[key]]

    # Preserve variable token lengths without padding or duplicating context per question.
    contexts = [record.get("context") for record in records]
    if any(context is None for context in contexts):
        if not all(context is None for context in contexts):
            raise ValueError("Cannot mix records with and without context features")
        contexts = None

    return TrainingBatch(
        features=FeatureBatch(
            questions=torch.cat([r["questions"] for r in records]),
            options=torch.cat([r["options"] for r in records]),
            option_to_question=torch.cat(links),
            question_to_context=torch.cat(context_links),
            contexts=contexts,
            question_ids=flatten("question_ids"),
            option_ids=flatten("option_ids"),
            metadata=records[0]["feature_metadata"],
        ),
        targets=torch.cat([r["targets"] for r in records]),
        context_ids=[r["context_id"] for r in records],
        categories=flatten("categories"),
        tasks=flatten("tasks"),
        target_kinds=flatten("target_kinds"),
        score_values=flatten("score_values"),
    )


class FeatureDataset(IterableDataset):
    """Read one shard at a time, optionally shuffling shards and their contexts.

    Call set_epoch before each training epoch. Use DataLoader without shuffle;
    workers get disjoint shards. Batch size counts contexts, not questions.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        shuffle: bool = False,
        seed: int = 42,
        include_context: bool = False,
    ):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if not self.manifest.get("complete", False):
            raise ValueError(
                "Feature cache is incomplete; resume preparation before training"
            )
        self.info = self.manifest["splits"][split]
        self.include_context = include_context
        if (
            include_context
            and not self.manifest["features"]["pooling"]["include_context"]
        ):
            raise ValueError(
                "This cache has no context features; use include_context=False"
            )
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Select a reproducible new ordering without changing stored features."""
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        shards = list(self.info["parts"])
        if self.shuffle:
            rng.shuffle(shards)
        worker = get_worker_info()
        if worker is not None:
            shards = shards[worker.id :: worker.num_workers]
        for part in shards:
            payload = torch.load(
                self.root / part["file"], map_location="cpu", weights_only=True
            )
            # Skipped source contexts are progress metadata, never training rows.
            records = payload["records"] if isinstance(payload, dict) else payload
            for record in records:
                # Contexts share the shard file, so torch.load still reads them.
                # Dropping references here prevents transfer to workers' batches/GPU.
                if not self.include_context:
                    record["context"] = None
                elif record.get("context") is None:
                    raise ValueError(
                        "A cache record is missing requested context features"
                    )
                record["feature_metadata"] = self.manifest["features"]
            if self.shuffle:
                rng.shuffle(records)
            yield from records
