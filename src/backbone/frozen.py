"""Lazy frozen text-backbone loading; no decision head or generation logic."""

import torch

from .loading import BackboneConfig, ModelSource

from .inputs import PackedBatch
from .forward import BackboneOutput, forward_decoder, validate_attention, validate_decoder


class FrozenBackbone(ModelSource):
    """Frozen decoder with selective hidden-state extraction."""

    def load(self):
        """Load AutoModel, optionally quantizing source weights to NF4 in memory."""
        if self._model is None:
            self._model = self.load_text_decoder()
            self._model.requires_grad_(False).eval()
        return self._model

    @classmethod
    def from_model(cls, model) -> "FrozenBackbone":
        """Wrap an in-memory decoder for fully offline notebook experiments."""
        validate_decoder(model)
        parameter = next(model.parameters())
        wrapper = cls(
            BackboneConfig(
                model_id="in-memory",
                device=str(parameter.device),
                dtype=str(parameter.dtype).removeprefix("torch."),
                quantization="none",
                local_files_only=True,
            )
        )
        wrapper._model = model.requires_grad_(False).eval()
        wrapper._hf_config = model.config
        return wrapper

    @torch.no_grad()
    def forward(
        self, batch: PackedBatch, layers: tuple[int, ...] = (-1,)
    ) -> BackboneOutput:
        """Execute the shared decoder path without building an autograd graph."""
        validate_attention(self._load_config(), batch.config.attention)
        return forward_decoder(self.load(), batch, layers, {
            "model_id": self.config.model_id,
            "revision": self._revision,
            "quantization": self.config.quantization,
            "dtype": self.config.dtype,
        })
