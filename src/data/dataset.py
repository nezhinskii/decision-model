"""Read and incrementally write prepared Parquet, independently of its source."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import Example, Option, Question, Target, TargetKind, Task, arrow_schema


def read_examples(path: Path) -> Iterator[Example]:
    """Stream shared contexts, restoring nested dataclasses from stored records."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            questions = []
            for question in row["questions"]:
                question["options"] = [Option(**option) for option in question["options"]]
                question["target"]["kind"] = TargetKind(question["target"]["kind"])
                question["target"] = Target(**question["target"])
                question["task"] = Task(question["task"])
                questions.append(Question(**question))
            yield Example(id=row["id"], context=row["context"], questions=questions)


class DatasetWriter:
    """Bounded Arrow buffer and running statistics for one prepared split.

    A single context is indivisible. Buffer limits apply between contexts; an
    unusually large context is written alone and must still fit in memory itself.
    """

    def __init__(self, path: Path, batch_size: int = 128, max_batch_bytes: int = 8 * 1024 * 1024):
        self._writer = pq.ParquetWriter(path, arrow_schema(), compression="zstd")
        self._batch_size, self._max_bytes = batch_size, max_batch_bytes
        self._tables, self._bytes = [], 0
        self._contexts = 0
        self._counts = {key: Counter() for key in (
            "categories", "tasks", "targets", "option_counts", "questions_per_context")}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self.flush()
        finally:
            self._writer.close()

    def write(self, example: Example):
        table = pa.Table.from_pylist([asdict(example)], schema=arrow_schema())
        if self._tables and self._bytes + table.nbytes > self._max_bytes:
            self.flush()
        self._tables.append(table)
        self._bytes += table.nbytes
        if len(self._tables) >= self._batch_size or self._bytes >= self._max_bytes:
            self.flush()
        self._contexts += 1
        self._counts["questions_per_context"][len(example.questions)] += 1
        for question in example.questions:
            self._counts["categories"][question.category] += 1
            self._counts["tasks"][question.task] += 1
            self._counts["targets"][question.target.kind] += 1
            self._counts["option_counts"][len(question.options)] += 1

    def flush(self):
        if self._tables:
            self._writer.write_table(pa.concat_tables(self._tables))
            self._tables.clear()
            self._bytes = 0

    def statistics(self):
        return {"contexts": self._contexts,
                "questions": sum(self._counts["categories"].values()),
                **{key: dict(sorted(counts.items())) for key, counts in self._counts.items()}}
