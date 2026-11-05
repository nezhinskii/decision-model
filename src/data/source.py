"""Dataset source contract and source-independent preparation algorithms."""

from collections import Counter, defaultdict
from collections.abc import Iterator
import random
from typing import Protocol

from tqdm.auto import tqdm

from .sampling import Candidate
from .schema import Example


class DatasetSource(Protocol):
    """Re-readable source yielding the common schema, before tokenization.

    Candidate.row and Question.source_row are stable unique integer addresses
    within this source. Example.id must equal Candidate.group for its questions;
    one group identifies one shared context. IDs must be namespaced when combining
    datasets. read_examples must honor rows and preserve original question order.
    """

    def describe(self) -> dict:
        """Dataset identity/revision and source-specific provenance or limitations."""
        ...

    def read_index(self, batch_size: int) -> list[Candidate]:
        """Compact row/group/category/task/target-kind metadata, sorted by row."""
        ...

    def read_examples(
        self, batch_size: int, rows: set[int] | None = None
    ) -> Iterator[Example]:
        """Common examples containing exactly requested rows, or all rows for None."""
        ...


def iter_indexed_questions(source: DatasetSource, candidates: list[Candidate], batch_size: int):
    """Validate adapter alignment while streaming; no second full dataset copy."""
    requested = {candidate.row: candidate for candidate in candidates}
    if len(requested) != len(candidates):
        raise ValueError("Source index contains duplicate row addresses")
    seen = set()
    for example in source.read_examples(batch_size, rows=set(requested)):
        for question in example.questions:
            row = question.source_row
            if row not in requested or row in seen:
                raise ValueError(
                    f"Source returned an unexpected or duplicate row: {row}"
                )
            candidate = requested[row]
            if (
                example.id,
                question.category,
                str(question.task),
                str(question.target.kind),
            ) != (candidate.group, candidate.category, candidate.task, candidate.kind):
                raise ValueError(
                    f"Source index and question metadata differ at row {row}"
                )
            seen.add(row)
            yield candidate, example.context, question
    if seen != set(requested):
        raise ValueError("Source omitted requested rows")


def iter_materialized(
    source: DatasetSource,
    assignment: dict[str, list[Candidate]],
    batch_size: int,
    context_batch_size: int,
) -> Iterator[tuple[str, Example]]:
    """Read a bounded portion of complete contexts, yield it, then release texts.

    Only candidate metadata spans the whole selection. Re-reading source blocks
    is allowed: an adapter chooses how to retrieve each requested set of rows.
    """
    if min(batch_size, context_batch_size) < 1:
        raise ValueError("Batch sizes must be positive")
    groups = defaultdict(list)
    splits, seen_rows = {}, set()
    for name, candidates in assignment.items():
        for candidate in candidates:
            if candidate.row in seen_rows:
                raise ValueError("A source row was selected more than once")
            seen_rows.add(candidate.row)
            if candidate.group in splits and splits[candidate.group] != name:
                raise RuntimeError("A context was assigned to more than one split")
            splits[candidate.group] = name
            groups[candidate.group].append(candidate)

    order = sorted(groups)
    for start in range(0, len(order), context_batch_size):
        portion = order[start:start + context_batch_size]
        candidates = sorted((c for group in portion for c in groups[group]), key=lambda c: c.row)
        examples = {}
        for candidate, context, question in iter_indexed_questions(source, candidates, batch_size):
            if candidate.group not in examples:
                # Preserve the first selected original rendering of this context.
                examples[candidate.group] = Example(candidate.group, context, [])
            examples[candidate.group].questions.append(question)
        # Global context order also preserves sorted order within each split.
        for group in portion:
            yield splits[group], examples.pop(group)


def filter_candidates(
    source: DatasetSource,
    candidates: list[Candidate],
    builder,
    seed: int,
    batch_size: int,
):
    """Bound each candidate context group before sampling any subset from it.

    Tokenization is segment-based, so question branches have additive costs.
    Limiting the entire eligible group guarantees that any sampled subset fits,
    without splitting a shared context across records or dataset splits.
    """
    groups = defaultdict(list)
    bases = defaultdict(int)
    too_long = Counter()
    budget = builder.config.max_packed_tokens
    with tqdm(total=len(candidates), desc="Token lengths", unit="question") as progress:
        for candidate, context, question in iter_indexed_questions(source, candidates, batch_size):
            encoded = builder.encode(
                Example(candidate.group, context, [question]), check_length=False
            )
            base = sum(
                block.stop - block.start
                for block in encoded.blocks
                if block.role in {"prefix", "context", "suffix"}
            )
            # Reserve the largest original rendering of an equivalent context.
            bases[candidate.group] = max(bases[candidate.group], base)
            if len(encoded.input_ids) > budget:
                too_long[candidate.category] += 1
            else:
                groups[candidate.group].append(
                    (candidate, len(encoded.input_ids) - base)
                )
            progress.update()

    rng = random.Random(seed)
    eligible, grouped_out = [], Counter()
    for group, rows in sorted(groups.items()):
        rng.shuffle(rows)
        used = bases[group]
        for candidate, cost in rows:
            if used + cost <= budget:
                eligible.append(candidate)
                used += cost
            else:
                grouped_out[candidate.category] += 1

    before = Counter(c.category for c in candidates)
    after = Counter(c.category for c in eligible)
    stats = {
        "source_questions": len(candidates),
        "eligible_questions": len(eligible),
        "excluded_individually": sum(too_long.values()),
        "excluded_group_budget": sum(grouped_out.values()),
        "by_category": {
            category: {
                "source": before[category],
                "eligible": after[category],
                "excluded_individually": too_long[category],
                "excluded_group_budget": grouped_out[category],
            }
            for category in sorted(before)
        },
        "policy": "Seeded greedy admission per source context; complete eligible group fits budget",
    }
    return sorted(eligible, key=lambda c: c.row), stats
