"""Deterministic category balancing and context-disjoint research splits."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Candidate:
    """Small source-row index used before converting selected records."""

    row: int
    group: str
    category: str
    task: str
    kind: str

    @property
    def stratum(self) -> tuple[str, str, str]:
        """Preserve task and target-kind proportions within each category."""
        return self.category, self.task, self.kind


def proportional_quotas(total: int, capacities: dict) -> dict:
    """Round proportional counts to integers using largest remainders."""
    available = sum(capacities.values())
    if total < 0 or total > available:
        raise ValueError(f"Requested {total} questions, but only {available} are available")
    if available == 0:
        return {key: 0 for key in capacities}
    quotas = {key: total * count // available for key, count in capacities.items()}
    order = sorted(capacities, key=lambda k: (-(total * capacities[k] % available), k))
    for key in order[:total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def balanced_quotas(total: int, capacities: dict[str, int]) -> dict[str, int]:
    """Use equal category quotas, redistributing shortages without resampling."""
    if total > sum(capacities.values()):
        raise ValueError("The requested research set exceeds the source dataset")
    quotas = dict.fromkeys(sorted(capacities), 0)
    remaining = total
    while remaining:
        active = [key for key in quotas if quotas[key] < capacities[key]]
        share = max(1, remaining // len(active))
        for key in active:
            take = min(share, capacities[key] - quotas[key], remaining)
            quotas[key] += take
            remaining -= take
    return quotas


def select(candidates: list[Candidate], total: int, seed: int) -> tuple[list[Candidate], dict]:
    """Select by category, then proportionally by task and target kind."""
    pools = defaultdict(list)
    capacities = Counter()
    for candidate in candidates:
        pools[candidate.stratum].append(candidate)
        capacities[candidate.category] += 1
    categories = balanced_quotas(total, capacities)
    rng = random.Random(seed)
    selected = []
    for category, quota in categories.items():
        counts = {key: len(rows) for key, rows in pools.items() if key[0] == category}
        for stratum, count in proportional_quotas(quota, counts).items():
            pool = pools[stratum].copy()
            rng.shuffle(pool)
            selected.extend(pool[:count])
    return sorted(selected, key=lambda c: c.row), categories


def split_groups(selected: list[Candidate], sizes: dict[str, int], seed: int) -> tuple[dict, dict]:
    """Assign whole context groups; quotas are soft, isolation is strict."""
    remaining = Counter(candidate.stratum for candidate in selected)
    targets = {}
    # Reserve held-out quotas first; the remaining questions belong to train.
    for name in ("validation", "test"):
        targets[name] = proportional_quotas(sizes[name], remaining)
        remaining.subtract(targets[name])
    targets["train"] = dict(remaining)
    groups = defaultdict(list)
    for candidate in selected:
        groups[candidate.group].append(candidate)
    rng = random.Random(seed)
    ordered = list(groups.values())
    rng.shuffle(ordered)
    # Place large groups first while there is still room in every split.
    ordered.sort(key=lambda group: -len(group))
    names = list(sizes)
    rng.shuffle(names)
    assigned = {name: [] for name in sizes}
    used = {name: Counter() for name in sizes}
    for group in ordered:
        counts = Counter(candidate.stratum for candidate in group)
        fits = [name for name in names if len(assigned[name]) + len(group) <= sizes[name]]
        choices = fits or names

        def score(name: str) -> tuple[float, float]:
            matched = sum(min(count, max(0, targets[name].get(key, 0) - used[name][key]))
                          for key, count in counts.items())
            space = sizes[name] - len(assigned[name])
            return matched / len(group), space / max(1, sizes[name])

        name = max(choices, key=score)
        assigned[name].extend(group)
        used[name].update(counts)
    for rows in assigned.values():
        rows.sort(key=lambda candidate: candidate.row)
    return assigned, targets
