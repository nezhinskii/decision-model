"""Question-weighted objectives and metrics for variable option counts."""

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
import math

import torch


class SmoothingScope(StrEnum):
    ce = "ce"
    both = "both"


@dataclass
class LossConfig:
    """Training-only weights and hard-target smoothing; evaluation stays unchanged."""
    ce_weight: float = 1.0
    brier_weight: float = 0.0
    label_smoothing: float = 0.0
    smoothing_scope: SmoothingScope = SmoothingScope.both

    def __post_init__(self):
        if (not all(math.isfinite(w) and w >= 0 for w in (self.ce_weight, self.brier_weight))
                or self.ce_weight + self.brier_weight == 0):
            raise ValueError("Loss weights must be finite, nonnegative and not all zero")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.smoothing_scope = SmoothingScope(self.smoothing_scope)


TRAIN_METRIC_NAMES = (
    "all/loss", "all/accuracy", "all/brier", "objective", "ce_component", "brier_component",
)


def _question_statistics(logits, batch):
    """Compute grouped log probabilities once, alongside original-target metrics."""
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
    return torch.stack((loss, accuracy, brier), dim=-1), log_probs, loss, brier


def question_metrics(logits, batch):
    """Return [Q, 3] original-target CE, accuracy and Brier."""
    return _question_statistics(logits, batch)[0]


def training_values(logits, batch, config: LossConfig):
    """Return [Q, 6] columns named by TRAIN_METRIC_NAMES; objective is column 3.

    Components are before weighting. Smoothing is uniform within each question,
    applies only to declared hard targets and never mutates batch.targets.
    """
    metrics, log_probs, ce, brier = _question_statistics(logits, batch)
    links = batch.features.option_to_question  # [V]
    targets = batch.targets.float()  # [V]
    questions = len(batch.features.questions)
    # Use the individual [Q] tensors: slicing the stacked metrics would keep
    # both loss branches attached through StackBackward even for CE-only runs.
    if config.label_smoothing:
        counts = log_probs.new_zeros(questions).scatter_add(
            0, links, torch.ones_like(log_probs))  # [Q], options per question.
        hard = torch.tensor([kind == "hard" for kind in batch.target_kinds],
                            device=links.device)  # [Q]
        softened = (1 - config.label_smoothing) * targets + config.label_smoothing / counts[links]
        smoothed = torch.where(hard[links], softened, targets)  # [V]
        ce = log_probs.new_zeros(questions).scatter_add(0, links, -smoothed * log_probs)
        if config.smoothing_scope == SmoothingScope.both:
            brier = log_probs.new_zeros(questions).scatter_add(
                0, links, (log_probs.exp() - smoothed).square())
    if config.brier_weight == 0:
        objective = config.ce_weight * ce
    elif config.ce_weight == 0:
        objective = config.brier_weight * brier
    else:
        objective = config.ce_weight * ce + config.brier_weight * brier
    # Only objective participates in backward; all other columns are reports.
    return torch.cat((metrics.detach(), objective[:, None],
                      ce.detach()[:, None], brier.detach()[:, None]), dim=1)


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
