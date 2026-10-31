"""Compose projection, processor and scorer without changing the training loop."""

from dataclasses import dataclass
from enum import StrEnum

from torch import nn

from ..features import FeatureBatch
from .projection import InputProjection
from .processors import IdentityProcessor, JointProcessor, SeparateMLPProcessor
from .scorers import PointerScorer
from .readout import LayerReadout, ReadoutMode
from .norms import NormType


class ProcessorType(StrEnum):
    identity = "identity"
    mlp = "mlp"
    joint = "joint"


@dataclass
class HeadConfig:
    """Processor and layer readout settings; indices address the cached layer axis."""

    processor: ProcessorType = ProcessorType.joint
    dim: int = 768
    hidden_dim: int = 3072
    depth: int = 2
    attention_heads: int = 12
    num_slots: int = 64
    use_cross_attention: bool = True
    score_dim: int = 256
    dropout: float = 0.0
    readout: ReadoutMode = ReadoutMode.last
    layer_indices: tuple[int, ...] = ()  # Empty means all R cached layers, in order.
    router_dim: int = 128
    norm_type: NormType = NormType.layernorm


class DecisionHead(nn.Module):
    """Return [V] logits in original option order, with no softmax or loss."""

    def __init__(
        self, projection: InputProjection, processor: nn.Module, scorer: nn.Module
    ):
        super().__init__()
        self.projection = projection
        self.processor = processor
        self.scorer = scorer

    @property
    def requires_context(self) -> bool:
        return self.processor.requires_context

    def forward(self, features: FeatureBatch):
        projected = self.projection(features)
        states = self.processor(projected)
        return self.scorer(states, projected.option_to_question)


def build_head(
    input_dim: int, config: HeadConfig, num_feature_layers: int = 1
) -> DecisionHead:
    """Small explicit factory; dimensions can be read from the first feature batch."""
    if min(input_dim, config.dim, config.hidden_dim, config.score_dim) < 1:
        raise ValueError("Head dimensions must be positive")
    if config.depth < 1 or config.num_slots < 0:
        raise ValueError("depth must be positive and num_slots nonnegative")
    if num_feature_layers < 1 or config.router_dim < 1:
        raise ValueError("Feature layer count and router_dim must be positive")
    readout = LayerReadout(
        config.dim,
        num_feature_layers,
        config.readout,
        config.layer_indices,
        config.router_dim,
    )
    if config.processor == "identity":
        processor = IdentityProcessor(readout)
    elif config.processor == "mlp":
        processor = SeparateMLPProcessor(
            config.dim,
            config.hidden_dim,
            config.depth,
            config.dropout,
            readout,
            config.norm_type,
        )
    elif config.processor == "joint":
        if config.attention_heads < 1 or config.dim % config.attention_heads:
            raise ValueError("dim must be divisible by attention_heads")
        processor = JointProcessor(
            config.dim,
            config.attention_heads,
            config.hidden_dim,
            config.depth,
            config.dropout,
            config.num_slots,
            config.use_cross_attention,
            readout,
            config.norm_type,
        )
    else:
        raise ValueError(f"Unknown processor: {config.processor}")
    return DecisionHead(
        InputProjection(input_dim, config.dim, processor.requires_context, config.norm_type),
        processor,
        PointerScorer(config.dim, config.score_dim),
    )
