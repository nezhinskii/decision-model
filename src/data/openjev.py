"""OpenJevData SFT download, selection metadata and common-format conversion."""

from bisect import bisect_left
import math
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
import pyarrow as pa
import pyarrow.parquet as pq

from .sampling import Candidate
from .schema import Example, Option, Question, Target, TargetKind, Task, context_id

DATASET = "shenjunhao/OpenJevData-140k"
REVISION = "619db5c439cf4f83cbed74eeab90479f6c39f294"


def download(raw_root: Path, revision: str) -> Path:
    """Fetch only SFT and source notices from an immutable dataset revision."""
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Use an exact dataset commit SHA, not a moving branch")
    destination = raw_root / "openjev" / revision
    snapshot_download(
        DATASET,
        repo_type="dataset",
        revision=revision,
        local_dir=destination,
        allow_patterns=[
            "data/SFT.parquet",
            "README.md",
            "LICENSE",
            "SOURCE_NOTICES.md",
        ],
    )
    return destination / "data" / "SFT.parquet"


def read_index(path: Path, batch_size: int) -> list[Candidate]:
    """Scan selection metadata; no normalized full dataset is written."""
    candidates = []
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=batch_size,
        columns=["state", "category", "task", "answer"],
    ):
        for row in batch.to_pylist():
            candidates.append(
                Candidate(
                    row=len(candidates),
                    group=context_id(row["state"]),
                    category=row["category"],
                    task=row["task"],
                    kind=row["answer"]["kind"],
                )
            )
    return candidates


def question_id(index: int, revision: str) -> str:
    """Keep IDs stable across seeds, splits and preparation versions."""
    return f"openjev@{revision}:SFT:{index}"


def convert(row: dict, index: int, revision: str) -> Question:
    """Convert a selected row, checking only training-critical target alignment."""
    probabilities = row["answer"]["probabilities"]
    options = row["options"]
    if not options or len(options) != len(probabilities):
        raise ValueError(f"SFT row {index}: targets must align with nonempty options")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError(f"SFT row {index}: invalid target probabilities")
    if abs(sum(probabilities) - 1) > 1e-6:
        raise ValueError(f"SFT row {index}: target probabilities must sum to one")
    scores = row["score_values"]
    if scores is not None and len(scores) != len(options):
        raise ValueError(f"SFT row {index}: score values must align with options")
    return Question(
        id=question_id(index, revision),
        source_row=index,
        text=row["question"],
        category=row["category"],
        task=Task(row["task"]),
        options=[
            Option(id=o["id"], text=o["name"], criteria=o["criteria"]) for o in options
        ],
        target=Target(
            kind=TargetKind(row["answer"]["kind"]), probabilities=probabilities
        ),
        score_values=scores,
    )


@dataclass
class OpenJevSource:
    """Source adapter: native Parquet fields stop at this boundary."""

    path: Path
    revision: str = REVISION

    def describe(self) -> dict:
        return {
            "dataset": DATASET,
            "revision": self.revision,
            "stage": "SFT",
            "file": str(self.path),
            "limitations": [
                "Source task/template IDs are unavailable; paraphrase and template leakage are not certified absent."
            ],
        }

    def read_index(self, batch_size: int) -> list[Candidate]:
        return read_index(self.path, batch_size)

    def read_examples(self, batch_size: int, rows: set[int] | None = None):
        """Skip irrelevant row groups; decode requested rows from bounded batches.

        Parquet is not a row-addressable store: neighboring rows in relevant
        groups may still be read/decompressed, but never retained as a full table.
        """
        wanted = sorted(rows) if rows is not None else None
        if wanted == []:
            return
        with pq.ParquetFile(self.path, pre_buffer=False) as parquet:
            if wanted is not None and (wanted[0] < 0 or wanted[-1] >= parquet.metadata.num_rows):
                raise ValueError("Requested row lies outside the source Parquet")
            offset = 0
            for group in range(parquet.metadata.num_row_groups):
                end = offset + parquet.metadata.row_group(group).num_rows
                if wanted is not None and bisect_left(wanted, offset) == bisect_left(wanted, end):
                    offset = end
                    continue
                position = offset
                for batch in parquet.iter_batches(batch_size=batch_size, row_groups=[group]):
                    stop = position + batch.num_rows
                    if wanted is None:
                        indices = range(position, stop)
                    else:
                        left, right = bisect_left(wanted, position), bisect_left(wanted, stop)
                        indices = wanted[left:right]
                        # Drop irrelevant rows before allocating Python strings/dicts.
                        batch = batch.take(pa.array([i - position for i in indices], type=pa.int64()))
                    for index, raw in zip(indices, batch.to_pylist()):
                        yield Example(context_id(raw["state"]), raw["state"],
                                      [convert(raw, index, self.revision)])
                    position = stop
                offset = end
