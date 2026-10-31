"""Reduce selected cached layers using one set of weights per question."""

from enum import StrEnum

import torch
from torch import nn

from ..features import FeatureBatch


class ReadoutMode(StrEnum):
    last = "last"
    mean = "mean"
    static = "static"
    adaptive = "adaptive"


class LayerReadout(nn.Module):
    """Mix projected layers; the same question weights apply to all its options.

    Adaptive routing scores each layer's question and mean option states using
    the same small MLP for every layer, then normalizes scores across layers.
    Indices refer to the cache layer axis, not decoder depths. Token memory is
    untouched and always comes from the final backbone layer.
    """

    def __init__(
        self, dim, num_layers, mode=ReadoutMode.last, indices=(), router_dim=128
    ):
        super().__init__()
        self.mode = ReadoutMode(mode)
        self.indices = tuple(indices) or tuple(range(num_layers))
        if not self.indices or len({i % num_layers for i in self.indices}) != len(
            self.indices
        ):
            raise ValueError("Readout needs distinct cached layer indices")
        if any(not -num_layers <= i < num_layers for i in self.indices):
            raise ValueError("Readout layer index is outside the cached layer axis")
        self.num_layers = num_layers
        count = len(self.indices)
        self.logits = (
            nn.Parameter(torch.zeros(count))
            if self.mode == ReadoutMode.static
            else None
        )
        self.router = None
        if self.mode == ReadoutMode.adaptive:
            self.router = nn.Sequential(
                nn.Linear(2 * dim, router_dim), nn.GELU(), nn.Linear(router_dim, 1)
            )

    def forward(self, features: FeatureBatch):
        if (
            features.questions.shape[1] != self.num_layers
            or features.options.shape[1] != self.num_layers
        ):
            raise ValueError("Feature layer count differs from head construction")
        q = features.questions[:, self.indices, :]  # [Q, R_selected, d]
        o = features.options[:, self.indices, :]  # [V, R_selected, d]
        if self.mode == ReadoutMode.last:
            return q[:, -1], o[:, -1]
        if self.mode == ReadoutMode.mean:
            return q.mean(1), o.mean(1)
        if self.mode == ReadoutMode.static:
            # One learned distribution shared by every question.
            layer_weights = self.logits.softmax(dim=0)  # [R_selected]
            question_weights = layer_weights.unsqueeze(0)  # [1, R_selected]
            question_weights = question_weights.expand(len(q), -1)  # [Q, R_selected]
        else:
            num_questions = q.shape[0]
            selected_layers = q.shape[1]
            hidden_dim = q.shape[2]
            option_to_question = (
                features.option_to_question
            )  # [V], question index per option

            option_sums = o.new_zeros(
                (num_questions, selected_layers, hidden_dim)
            )  # [Q, R_selected, d]
            # Sum options per question independently at every selected layer.
            option_sums.index_add_(dim=0, index=option_to_question, source=o)
            option_counts = torch.bincount(
                option_to_question,
                minlength=num_questions,
            )  # [Q]
            divisors = option_counts.clamp_min(1).reshape(
                num_questions, 1, 1
            )  # [Q, 1, 1]
            mean_options = option_sums / divisors  # [Q, R_selected, d]

            router_input = torch.cat((q, mean_options), dim=2)  # [Q, R_selected, 2*d]
            # Linear operates on the last axis: one shared MLP scores each
            # question/layer pair. Its parameter count does not depend on R.
            layer_scores = self.router(router_input)  # [Q, R_selected, 1]
            layer_logits = layer_scores.squeeze(2)  # [Q, R_selected]
            question_weights = layer_logits.softmax(dim=1)  # [Q, R_selected]

        # Each scalar weight scales all d coordinates of its layer's vector.
        question_scales = question_weights.unsqueeze(2)  # [Q, R_selected, 1]
        weighted_questions = q * question_scales  # [Q, R_selected, d]
        mixed_questions = weighted_questions.sum(dim=1)  # [Q, d]

        # Every option reuses the distribution computed for its own question.
        option_weights = question_weights.index_select(
            dim=0,
            index=features.option_to_question,
        )  # [V, R_selected]
        option_scales = option_weights.unsqueeze(2)  # [V, R_selected, 1]
        weighted_options = o * option_scales  # [V, R_selected, d]
        mixed_options = weighted_options.sum(dim=1)  # [V, d]
        return mixed_questions, mixed_options
