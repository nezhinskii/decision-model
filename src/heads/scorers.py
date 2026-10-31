"""Scorers produce raw option logits; grouped normalization belongs to the loss."""

import math

import torch
from torch import nn

from .processors import DecisionStates


class PointerScorer(nn.Module):
    """Projected dot products with one shared question state per question."""

    def __init__(self, dim: int, score_dim: int):
        super().__init__()
        self.question = nn.Linear(dim, score_dim, bias=False)
        self.option = nn.Linear(dim, score_dim, bias=False)
        self.scale = math.sqrt(score_dim)

    def forward(self, states: DecisionStates, option_to_question: torch.Tensor) -> torch.Tensor:
        queries = self.question(states.questions)[option_to_question]
        options = self.option(states.options)
        return (queries * options).sum(-1) / self.scale
