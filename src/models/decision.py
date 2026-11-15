"""One trainable module connecting a text decoder to any existing decision head."""

from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
from torch import nn

from ..backbone.forward import forward_decoder, validate_attention, validate_decoder
from ..backbone.inputs import InputConfig
from ..backbone.loading import BackboneConfig, ModelSource
from ..backbone.lora import LoRAConfig, attach_lora, adapter_summary
from ..backbone.pooling import PoolingConfig, extract_features
from ..heads import HeadConfig, build_head
from ..utils.files import file_hash


@dataclass
class DecisionConfig:
    backbone: BackboneConfig
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    inputs: InputConfig = field(default_factory=lambda: InputConfig(attention="causal"))
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    layers: tuple[int, ...] = (-1,)
    head: HeadConfig = field(default_factory=HeadConfig)
    gradient_checkpointing: bool = False


def source_identity(source: ModelSource):
    """Pin Hub identity; fingerprint local base files so resume cannot silently change them."""
    identity = source.describe()
    path = Path(source.config.model_id)
    if path.is_dir():
        files = set()
        for pattern in (
            "*.safetensors",
            "pytorch_model*.bin",
            "*.index.json",
            "*.json",
            "*.model",
            "*.txt",
        ):
            files.update(path.glob(pattern))
        identity["local_files"] = {
            p.name: file_hash(p) for p in sorted(files) if p.is_file()
        }
    return identity


class DecisionModel(nn.Module):
    def __init__(
        self, decoder, config: DecisionConfig, identity: dict, *, training: bool = True
    ):
        super().__init__()
        validate_decoder(decoder)
        validate_attention(decoder.config, config.inputs.attention)
        depth = decoder.config.num_hidden_layers
        resolved = tuple(depth if layer == -1 else layer for layer in config.layers)
        if (
            not resolved
            or len(set(resolved)) != len(resolved)
            or any(not 1 <= layer <= depth for layer in resolved)
        ):
            raise ValueError("Decision layers must be distinct valid decoder depths")
        if config.head.requires_context and depth not in resolved:
            raise ValueError(
                "Cross-attention requires the final decoder layer for context memory"
            )
        self.settings = config
        self.identity = identity
        self.layers = resolved
        self.pooling = replace(
            config.pooling, include_context=config.head.requires_context
        )
        if config.lora.enabled:
            self.decoder = attach_lora(
                decoder,
                config.lora,
                quantized=config.backbone.quantization == "nf4",
                gradient_checkpointing=training and config.gradient_checkpointing,
            )
            self.adapters = adapter_summary(self.decoder)
        else:
            self.decoder = decoder.requires_grad_(False).eval()
            self.adapters = {
                "modules": [],
                "trainable_parameters": 0,
                "total_parameters": sum(p.numel() for p in decoder.parameters()),
            }
        self.head = build_head(decoder.config.hidden_size, config.head, len(resolved))
        # Accelerate does not have to move an already-placed quantized decoder.
        self.head.to(device=decoder.get_input_embeddings().weight.device)
        self.train(training)

    @property
    def text_decoder(self):
        return (
            self.decoder.get_base_model()
            if self.settings.lora.enabled
            else self.decoder
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.settings.lora.enabled:
            self.decoder.eval()
        return self

    def forward(self, batch):
        if batch.config != self.settings.inputs:
            raise ValueError(
                "Packed inputs differ from the decision model configuration"
            )
        output = forward_decoder(
            self.text_decoder,
            batch,
            self.layers,
            {
                "model_id": self.identity["model_id"],
                "revision": self.identity["revision"],
                "quantization": self.settings.backbone.quantization,
                "dtype": self.settings.backbone.dtype,
            },
        )
        return self.head(extract_features(output, batch, self.pooling))


def load_decision(
    config: DecisionConfig, *, training: bool = True, source: ModelSource | None = None
):
    source = source or ModelSource(config.backbone)
    identity = source_identity(source)
    validate_attention(source._load_config(), config.inputs.attention)
    return DecisionModel(
        source.load_text_decoder(), config, identity, training=training
    )
