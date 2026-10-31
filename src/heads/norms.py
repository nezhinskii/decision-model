"""One normalization choice shared across head projections and processors."""

from enum import StrEnum

from torch import nn


class NormType(StrEnum):
    layernorm = "layernorm"
    rmsnorm = "rmsnorm"


def make_norm(dim: int, norm_type: NormType) -> nn.Module:
    """Use explicit equal epsilon so dtype-dependent defaults do not confound comparisons."""
    if NormType(norm_type) == NormType.layernorm:
        return nn.LayerNorm(dim, eps=1e-5)
    return nn.RMSNorm(dim, eps=1e-5)
