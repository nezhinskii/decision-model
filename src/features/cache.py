"""CPU-only cache serialization, reading and collation; no model loading."""

import json
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from .schema import FeatureBatch, LabeledFeatureBatch


def cache_records(features, examples) -> list[dict]:
    """Detach each shared context into independent CPU tensors and aligned labels."""
    records = []
    q_start = o_start = 0
    for index, example in enumerate(examples):
        q_stop = q_start + len(example.questions)
        o_stop = o_start + sum(len(q.options) for q in example.questions)
        # clone() prevents a small slice from retaining the entire batch storage.
        records.append(
            {
                "context_id": example.id,
                "context": (
                    features.contexts[index].detach().cpu().clone()
                    if features.contexts is not None
                    else None
                ),
                "questions": features.questions[q_start:q_stop].detach().cpu().clone(),
                "options": features.options[o_start:o_stop].detach().cpu().clone(),
                "option_to_question": (
                    features.option_to_question[o_start:o_stop] - q_start
                )
                .cpu()
                .clone(),
                "targets": torch.tensor(
                    [p for q in example.questions for p in q.target.probabilities],
                    dtype=torch.float32,
                ),
                "question_ids": [q.id for q in example.questions],
                "option_ids": [o.id for q in example.questions for o in q.options],
                "categories": [q.category for q in example.questions],
                # Keep storage as plain strings, not pickled enum instances, so
                # torch.load(weights_only=True) needs no custom allowlist.
                "tasks": [str(q.task) for q in example.questions],
                "target_kinds": [str(q.target.kind) for q in example.questions],
                "score_values": [q.score_values for q in example.questions],
            }
        )
        q_start, o_start = q_stop, o_stop
    return records


def collate_features(records: list[dict]) -> LabeledFeatureBatch:
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

    return LabeledFeatureBatch(
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


def make_loader(cache, split, context, batch_size, seed, shuffle=False):
    dataset = FeatureDataset(
        cache, split, shuffle=shuffle, seed=seed, include_context=context
    )
    # Dedicated generator prevents iterator creation/skipping from consuming model
    # RNG on resume. num_workers=0 fixes the reproducible source cursor.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_features,
        num_workers=0,
        generator=generator,
    )
