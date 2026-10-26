"""Plain data containers shared by source adapters and model input code."""

from dataclasses import dataclass
import hashlib
from typing import Literal

import pyarrow as pa


@dataclass
class Option:
    """Candidate text and optional conditions, both available to the backbone."""

    id: str
    text: str
    criteria: str | None = None


@dataclass
class Target:
    """Supervision aligned with option order; soft probabilities stay intact."""

    kind: Literal["hard", "soft"]
    probabilities: list[float]


@dataclass
class Question:
    """An independent decision with its original source-row identity."""

    id: str
    source_row: int
    text: str
    category: str
    task: Literal["choice", "score", "noul"]
    options: list[Option]
    target: Target
    score_values: list[float] | None = None


@dataclass
class Example:
    """One shared context and all questions selected for that context."""

    id: str
    context: str
    questions: list[Question]


def canonical_context(text: str) -> str:
    """Normalize line endings and outer newlines, preserving code indentation."""
    return text.replace("\r\n", "\n").strip("\n")


def context_id(text: str) -> str:
    """Use conservative text equality for grouping, without semantic matching."""
    return hashlib.sha256(canonical_context(text).encode()).hexdigest()


def arrow_schema() -> pa.Schema:
    """Explicit nested storage types avoid ambiguous null columns in small splits."""
    option = pa.struct([("id", pa.string()), ("text", pa.string()), ("criteria", pa.string())])
    target = pa.struct([("kind", pa.string()), ("probabilities", pa.list_(pa.float64()))])
    question = pa.struct([
        ("id", pa.string()), ("source_row", pa.int64()), ("text", pa.string()),
        ("category", pa.string()), ("task", pa.string()),
        ("options", pa.list_(option)), ("target", target),
        ("score_values", pa.list_(pa.float64())),
    ])
    return pa.schema([
        ("id", pa.string()), ("context", pa.string()), ("questions", pa.list_(question)),
    ])
