"""Filter source questions using the same serializer as backbone execution."""

from collections import Counter, defaultdict
import random

import pyarrow.parquet as pq
from tqdm.auto import tqdm

from . import openjev
from .schema import Example


def filter_candidates(source, candidates, builder, revision, seed, batch_size):
    """Bound each candidate context group before sampling any subset from it.

    Tokenization is segment-based, so question branches have additive costs.
    Limiting the entire eligible group guarantees that any sampled subset fits,
    without splitting a shared context across records or dataset splits.
    """
    groups = defaultdict(list)
    bases = defaultdict(int)
    too_long = Counter()
    budget = builder.config.max_packed_tokens
    index = 0
    with tqdm(total=len(candidates), desc="Token lengths", unit="question") as progress:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=batch_size):
            for raw in batch.to_pylist():
                candidate = candidates[index]
                question = openjev.convert(raw, index, revision)
                encoded = builder.encode(
                    Example(candidate.group, raw["state"], [question]), check_length=False,
                )
                base = sum(b.stop - b.start for b in encoded.blocks
                           if b.role in {"prefix", "context", "suffix"})
                # Equivalent contexts may differ in outer newlines; reserve the
                # largest actual shared segment, whichever selected text is stored.
                bases[candidate.group] = max(bases[candidate.group], base)
                if len(encoded.input_ids) > budget:
                    too_long[candidate.category] += 1
                else:
                    groups[candidate.group].append((candidate, len(encoded.input_ids) - base))
                index += 1
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
                "source": before[category], "eligible": after[category],
                "excluded_individually": too_long[category],
                "excluded_group_budget": grouped_out[category],
            } for category in sorted(before)
        },
        "policy": "Seeded greedy admission per source context; complete eligible group fits budget",
    }
    return sorted(eligible, key=lambda c: c.row), stats
