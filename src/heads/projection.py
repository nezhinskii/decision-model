"""Shared learned projections from backbone features into head space."""

from dataclasses import replace

from torch import nn

from ..features import FeatureBatch
from .norms import NormType, make_norm


class InputProjection(nn.Module):
    """Project types separately, preserving the selected-layer axis.

    Match inputs to the module's dtype; cached BF16 features can feed an FP32
    head. Tensor casts preserve gradients for future live-backbone training.
    """

    def __init__(self, input_dim: int, dim: int, include_context: bool,
                 norm_type: NormType = NormType.layernorm):
        super().__init__()
        self.question = nn.Sequential(nn.Linear(input_dim, dim), make_norm(dim, norm_type))
        self.option = nn.Sequential(nn.Linear(input_dim, dim), make_norm(dim, norm_type))
        self.context = (
            nn.Sequential(nn.Linear(input_dim, dim), make_norm(dim, norm_type))
            if include_context
            else None
        )

    def forward(self, features: FeatureBatch) -> FeatureBatch:
        """Project context once per context, only when the processor needs it."""
        dtype = self.question[0].weight.dtype
        contexts = None
        if self.context is not None:
            if features.contexts is None:
                raise ValueError("This head requires context-token features")
            contexts = [
                self.context(context.to(dtype=dtype)) for context in features.contexts
            ]
        return replace(
            features,
            questions=self.question(features.questions.to(dtype=dtype)),
            options=self.option(features.options.to(dtype=dtype)),
            contexts=contexts,
        )
