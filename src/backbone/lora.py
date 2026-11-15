"""PEFT adapters for text decoders; independent of decision heads and training loops."""

from dataclasses import dataclass
import math

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer


@dataclass
class LoRAConfig:
    enabled: bool = True
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    # PEFT supports all-linear or a regex; explicit suffixes override it when supplied.
    target_modules: str = "all-linear"
    module_names: tuple[str, ...] = ()

    def __post_init__(self):
        if self.rank < 1 or self.alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_modules or any(not name for name in self.module_names):
            raise ValueError("LoRA targets must not be empty")


def attach_lora(
    decoder, config: LoRAConfig, *, quantized: bool, gradient_checkpointing: bool
):
    """Freeze base weights and add ordinary LoRA; no native-inference patches."""
    decoder.requires_grad_(False)
    if quantized:
        if not getattr(decoder, "is_loaded_in_4bit", False):
            raise RuntimeError(
                "NF4 was requested but the decoder is not loaded in 4-bit"
            )
        # Enable checkpointing below, once, with non-reentrant autograd semantics.
        decoder = prepare_model_for_kbit_training(
            decoder, use_gradient_checkpointing=False
        )
    decoder.config.use_cache = False
    adapted = get_peft_model(
        decoder,
        LoraConfig(
            r=config.rank,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=(
                list(config.module_names)
                if config.module_names
                else config.target_modules
            ),
            bias="none",
        ),
    )
    if gradient_checkpointing:
        decoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    trainable = [
        name
        for name, parameter in adapted.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError(
            "Expected only LoRA parameters to be trainable in the decoder"
        )
    return adapted


def adapter_summary(decoder):
    """Persist actual targets, not just a requested pattern."""
    return {
        "modules": [
            name
            for name, module in decoder.named_modules()
            if isinstance(module, LoraLayer)
        ],
        "trainable_parameters": sum(
            p.numel() for p in decoder.parameters() if p.requires_grad
        ),
        "total_parameters": sum(p.numel() for p in decoder.parameters()),
    }
