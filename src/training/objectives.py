"""Question-weighted objectives and metrics for variable option counts."""

from dataclasses import dataclass
from enum import StrEnum
import math

import torch

from ..evaluation.metrics import (
    question_log_probs,
    cross_entropy,
    brier_score,
    accuracy,
)


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
        if (
            not all(
                math.isfinite(w) and w >= 0 for w in (self.ce_weight, self.brier_weight)
            )
            or self.ce_weight + self.brier_weight == 0
        ):
            raise ValueError(
                "Loss weights must be finite, nonnegative and not all zero"
            )
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.smoothing_scope = SmoothingScope(self.smoothing_scope)


TRAIN_METRIC_NAMES = (
    "all/loss",
    "all/accuracy",
    "all/brier",
    "objective",
    "ce_component",
    "brier_component",
)


@dataclass
class TrainingValues:
    """Only objective [Q] has gradients; all other fields are detached reports."""

    objective: torch.Tensor
    metrics: torch.Tensor  # [Q, 3]: original-target CE, accuracy, Brier.
    ce_component: torch.Tensor  # [Q], after training-only smoothing.
    brier_component: torch.Tensor  # [Q], after configured smoothing scope.


def training_values(logits, batch, config: LossConfig) -> TrainingValues:
    """Normalize once, build the configured objective and original-target reports.

    Disabled components are still reported, but their autograd graphs are never
    built. Unsmoothed loss values are reused for metrics rather than recomputed.
    """
    links = batch.features.option_to_question
    questions = len(batch.features.questions)
    targets = batch.targets.float()
    log_probs = question_log_probs(logits, links, questions)
    ce_targets = brier_targets = targets
    if config.label_smoothing:
        counts = log_probs.new_zeros(questions).scatter_add(
            0, links, torch.ones_like(log_probs)
        )
        hard = torch.tensor(
            [kind == "hard" for kind in batch.target_kinds], device=links.device
        )
        softened = (
            1 - config.label_smoothing
        ) * targets + config.label_smoothing / counts[links]
        ce_targets = torch.where(hard[links], softened, targets)
        if config.smoothing_scope == SmoothingScope.both:
            brier_targets = ce_targets

    gradients = torch.is_grad_enabled()
    with torch.set_grad_enabled(gradients and config.ce_weight > 0):
        ce = cross_entropy(log_probs, ce_targets, links, questions)
    with torch.set_grad_enabled(gradients and config.brier_weight > 0):
        brier = brier_score(log_probs, brier_targets, links, questions)
    if config.brier_weight == 0:
        objective = config.ce_weight * ce
    elif config.ce_weight == 0:
        objective = config.brier_weight * brier
    else:
        objective = config.ce_weight * ce + config.brier_weight * brier

    with torch.no_grad():
        original_ce = (
            ce
            if ce_targets is targets
            else cross_entropy(log_probs, targets, links, questions)
        )
        original_brier = (
            brier
            if brier_targets is targets
            else brier_score(log_probs, targets, links, questions)
        )
        metrics = torch.stack(
            (
                original_ce,
                accuracy(log_probs, targets, links, questions),
                original_brier,
            ),
            dim=-1,
        )
    return TrainingValues(objective, metrics, ce.detach(), brier.detach())
