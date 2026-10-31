"""Read prepared Parquet records without source-specific conversion."""

from pathlib import Path
from collections.abc import Iterator

import pyarrow.parquet as pq

from .schema import Example, Option, Question, Target, TargetKind, Task


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
