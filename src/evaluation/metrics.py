"""Independent grouped probability, loss and reporting calculations.

All option tensors are flat [V]. option_to_question maps each option to a
question in 0..num_questions-1. Loss functions return one value per question [Q].
"""

from collections import defaultdict
import torch


def question_log_probs(logits, option_to_question, num_questions):
    """Normalize logits within each question; return flat FP32 log probabilities."""
    scores = logits.float()
    maxima = scores.new_full((num_questions,), -torch.inf)
    maxima.scatter_reduce_(0, option_to_question, scores.detach(), reduce="amax")
    shifted = scores - maxima[option_to_question]
    denominators = scores.new_zeros(num_questions).scatter_add(
        0, option_to_question, shifted.exp()
    )
    return shifted - denominators.log()[option_to_question]


def cross_entropy(log_probs, targets, option_to_question, num_questions):
    """Per-question CE against hard or soft target probabilities, with gradients."""
    return log_probs.new_zeros(num_questions).scatter_add(
        0, option_to_question, -targets.float() * log_probs
    )


def brier_score(log_probs, targets, option_to_question, num_questions):
    """Per-question sum of squared probability errors, with gradients."""
    return log_probs.new_zeros(num_questions).scatter_add(
        0, option_to_question, (log_probs.exp() - targets.float()).square()
    )


@torch.no_grad()
def accuracy(log_probs, targets, option_to_question, num_questions):
    """Per-question argmax agreement; ties select the first original option."""
    predicted_scores = log_probs.new_full((num_questions,), -torch.inf)
    predicted_scores.scatter_reduce_(0, option_to_question, log_probs, reduce="amax")
    target_scores = targets.new_full((num_questions,), -torch.inf)
    target_scores.scatter_reduce_(0, option_to_question, targets, reduce="amax")
    indices = torch.arange(log_probs.numel(), device=log_probs.device)
    sentinel = log_probs.numel()
    predicted = option_to_question.new_full((num_questions,), sentinel)
    expected = option_to_question.new_full((num_questions,), sentinel)
    predicted.scatter_reduce_(
        0,
        option_to_question,
        torch.where(
            log_probs == predicted_scores[option_to_question], indices, sentinel
        ),
        reduce="amin",
    )
    expected.scatter_reduce_(
        0,
        option_to_question,
        torch.where(targets == target_scores[option_to_question], indices, sentinel),
        reduce="amin",
    )
    return (predicted == expected).float()


@torch.no_grad()
def question_metrics(log_probs, targets, option_to_question, num_questions):
    """Report [Q, 3] columns: original-target CE, accuracy, Brier. No autograd graph."""
    return torch.stack(
        (
            cross_entropy(log_probs, targets, option_to_question, num_questions),
            accuracy(log_probs, targets, option_to_question, num_questions),
            brier_score(log_probs, targets, option_to_question, num_questions),
        ),
        dim=-1,
    )


class MetricTotals:
    """Accumulate sums, not batch means, so small final batches are weighted correctly."""

    def __init__(self):
        self.groups = defaultdict(lambda: [torch.zeros(3, dtype=torch.float64), 0])

    def add(self, values, categories, target_kinds):
        values = values.detach().cpu().double()
        for q, value in enumerate(values):
            for key in (
                "all",
                f"category/{categories[q]}",
                f"target/{target_kinds[q]}",
            ):
                self.groups[key][0] += value
                self.groups[key][1] += 1

    def result(self):
        return {
            f"{group}/{name}": float(sums[i] / count)
            for group, (sums, count) in self.groups.items()
            for i, name in enumerate(("loss", "accuracy", "brier"))
        }
