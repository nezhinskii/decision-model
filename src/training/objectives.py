"""Question-weighted objectives and metrics for variable option counts."""

from collections import defaultdict

import torch


def question_metrics(logits, batch):
    """Return [Q, 3] CE, accuracy and Brier without per-question GPU loops."""
    links = batch.features.option_to_question  # [V], question for each option.
    targets = batch.targets.float()  # [V]
    scores = logits.float()  # [V], FP32 also under mixed precision.
    questions = len(batch.features.questions)

    # Each reduction groups options by question. Subtracting a detached maximum
    # gives stable log-softmax while preserving its usual gradient.
    maxima = scores.new_full((questions,), -torch.inf)
    maxima.scatter_reduce_(0, links, scores.detach(), reduce="amax")
    shifted = scores - maxima[links]
    denominators = scores.new_zeros(questions).scatter_add(0, links, shifted.exp())
    log_probs = shifted - denominators.log()[links]  # [V]
    loss = scores.new_zeros(questions).scatter_add(0, links, -targets * log_probs)
    brier = scores.new_zeros(questions).scatter_add(
        0, links, (log_probs.exp() - targets).square()
    )

    # Accuracy uses the first option on ties, matching torch.argmax.
    with torch.no_grad():
        target_maxima = targets.new_full((questions,), -torch.inf)
        target_maxima.scatter_reduce_(0, links, targets, reduce="amax")
        indices = torch.arange(scores.numel(), device=scores.device)
        sentinel = scores.numel()
        predicted = links.new_full((questions,), sentinel)
        expected = links.new_full((questions,), sentinel)
        predicted.scatter_reduce_(
            0,
            links,
            torch.where(scores == maxima[links], indices, sentinel),
            reduce="amin",
        )
        expected.scatter_reduce_(
            0,
            links,
            torch.where(targets == target_maxima[links], indices, sentinel),
            reduce="amin",
        )
        accuracy = (predicted == expected).float()  # [Q]
    return torch.stack((loss, accuracy, brier), dim=-1)


class MetricTotals:
    """Accumulate sums, not batch means, so small final batches are weighted correctly."""

    def __init__(self):
        self.groups = defaultdict(lambda: [torch.zeros(3, dtype=torch.float64), 0])

    def add(self, values, batch):
        values = values.detach().cpu().double()
        for q, value in enumerate(values):
            for key in (
                "all",
                f"category/{batch.categories[q]}",
                f"target/{batch.target_kinds[q]}",
            ):
                self.groups[key][0] += value
                self.groups[key][1] += 1

    def result(self):
        return {
            f"{group}/{name}": float(sums[i] / count)
            for group, (sums, count) in self.groups.items()
            for i, name in enumerate(("loss", "accuracy", "brier"))
        }
