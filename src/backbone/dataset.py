"""Tokenize prepared decision examples once and reuse them across live-training epochs."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from functools import partial

import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm

from .inputs import InputBuilder, InputTooLongError, PackedBatch
from ..data.dataset import read_examples
from ..utils.files import file_hash


@dataclass
class RejectedExample:
    example: object
    reason: str


@dataclass
class DecisionBatch:
    inputs: PackedBatch  # Remains on CPU; forward_decoder transfers model inputs.
    targets: torch.Tensor
    option_to_question: torch.Tensor
    categories: list[str]
    target_kinds: list[str]

    @property
    def num_questions(self):
        return len(self.categories)

    def to(self, device):
        return replace(
            self,
            targets=self.targets.to(device),
            option_to_question=self.option_to_question.to(device),
        )


def collate_decisions(samples, builder):
    questions = [q for sample in samples for q in sample.example.questions]
    return DecisionBatch(
        builder.pack(samples),
        torch.tensor(
            [p for q in questions for p in q.target.probabilities], dtype=torch.float32
        ),
        torch.tensor(
            [i for i, q in enumerate(questions) for _ in q.options], dtype=torch.long
        ),
        [q.category for q in questions],
        [str(q.target.kind) for q in questions],
    )


class DecisionDataset(IterableDataset):
    """Private temporary disk shards, read one at a time across training epochs.

    Preparation holds at most shard_size contexts or max_shard_tokens tokens,
    plus the current indivisible example and the bounded Parquet reader batch.
    Files are generated locally by this object, never loaded from user input.
    """

    def __init__(
        self, dataset: Path, split: str, builder: InputBuilder, position_limit: int,
        *, seed: int = 42, shuffle: bool = False,
        shard_size: int = 128, max_shard_tokens: int = 65536,
    ):
        super().__init__()
        if shard_size < 1 or max_shard_tokens < 1:
            raise ValueError("Token shard limits must be positive")
        self.builder, self.seed, self.shuffle, self.epoch = builder, seed, shuffle, 0
        manifest = json.loads((dataset / "manifest.json").read_text())
        info = manifest["splits"][split]
        path = dataset / info["file"]
        checksum = file_hash(path)
        if checksum != info["sha256"]:
            raise ValueError(f"Prepared {split} checksum mismatch")
        self._storage = TemporaryDirectory(prefix="decision-tokens-")
        self.root = Path(self._storage.name)
        self.parts = []
        self.summary = {
            "split_sha256": checksum, "source_questions": 0,
            "contexts": 0, "questions": 0, "skipped_contexts": 0,
            "skipped_questions": 0, "skipped": [],
        }
        records, tokens, seen = [], 0, set()

        def flush():
            nonlocal records, tokens
            if records:
                part = self.root / f"part-{len(self.parts):05d}.pt"
                torch.save(records, part)
                self.parts.append(part)
                records, tokens = [], 0

        try:
            for example in tqdm(read_examples(path), desc=f"Tokenize {split}", unit="context"):
                for question in example.questions:
                    if question.id in seen:
                        raise ValueError(f"Duplicate question ID: {question.id}")
                    seen.add(question.id)
                try:
                    encoded = builder.encode(example)
                    if max(encoded.positions) >= position_limit:
                        raise InputTooLongError("Branch positions exceed the backbone position limit")
                    # Store compact tensors rather than pickled lists of Python integers.
                    encoded.input_ids = torch.tensor(encoded.input_ids, dtype=torch.int64)
                    encoded.positions = torch.tensor(encoded.positions, dtype=torch.int64)
                    record = encoded
                    size = len(encoded.input_ids)
                    self.summary["contexts"] += 1
                    self.summary["questions"] += len(example.questions)
                except InputTooLongError as error:
                    record, size = RejectedExample(example, str(error)), 0
                    self.summary["skipped_contexts"] += 1
                    self.summary["skipped_questions"] += len(example.questions)
                    self.summary["skipped"].append({
                        "context_id": example.id,
                        "question_ids": [q.id for q in example.questions],
                        "reason": str(error),
                    })
                if records and (len(records) >= shard_size or tokens + size > max_shard_tokens):
                    flush()
                records.append(record)
                tokens += size
            flush()
            self.summary["source_questions"] = len(seen)
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return self.summary["contexts"]

    def close(self):
        """Release temporary token files; otherwise cleaned up with this dataset."""
        self._storage.cleanup()

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _records(self, *, shuffle=False):
        rng = random.Random(self.seed + self.epoch)
        parts = list(self.parts)
        if shuffle:
            rng.shuffle(parts)
        for path in parts:
            # Trusted private temporary files written above, not external checkpoints.
            records = torch.load(path, map_location="cpu", weights_only=False)
            if shuffle:
                rng.shuffle(records)
            yield from records
            del records

    @property
    def rejected(self):
        # Stream rejected texts too; a fully rejected dataset must not fill RAM.
        return (r for r in self._records() if isinstance(r, RejectedExample))

    def __iter__(self):
        for record in self._records(shuffle=self.shuffle):
            if not isinstance(record, RejectedExample):
                record.input_ids = record.input_ids.tolist()
                record.positions = record.positions.tolist()
                yield record


def make_decision_loader(dataset, batch_size):
    generator = torch.Generator().manual_seed(dataset.seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        generator=generator,
        collate_fn=partial(collate_decisions, builder=dataset.builder),
    )
