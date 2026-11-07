"""Shared lazy Hugging Face sources and loading settings."""

from dataclasses import dataclass
from pathlib import Path
from enum import StrEnum
import torch
from huggingface_hub import resolve_revision
from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig


class ComputeDtype(StrEnum):
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"


class Quantization(StrEnum):
    none = "none"
    nf4 = "nf4"


class AttentionBackend(StrEnum):
    sdpa = "sdpa"
    eager = "eager"


@dataclass
class BackboneConfig:
    """Shared text-model loading settings; an explicit checkpoint is required."""

    model_id: str
    revision: str | None = None
    cache_dir: Path = Path("artifacts/models/hub")
    local_files_only: bool = False
    device: str = "cuda:0"
    dtype: ComputeDtype = ComputeDtype.bfloat16
    quantization: Quantization = Quantization.nf4
    attention_backend: AttentionBackend = AttentionBackend.sdpa


class ModelSource:
    """Lazy pinned configuration/tokenizer shared by text model wrappers."""

    def __init__(self, config: BackboneConfig):
        self.config = config
        self._hf_config = None
        self._tokenizer = None
        self._model = None
        self._revision = config.revision

    def _hub_kwargs(self) -> dict:
        """Keep every Hub file under the project cache and honor offline mode."""
        return dict(
            revision=self._revision,
            cache_dir=str(self.config.cache_dir),
            local_files_only=self.config.local_files_only,
            trust_remote_code=False,
        )

    def _load_config(self):
        """Resolve a moving reference once, before loading tokenizer or weights."""
        if self._hf_config is None:
            if not Path(self.config.model_id).is_dir():
                # ResolvedRevision is a str whose displayed value may still be
                # 'main'. Persist .resolved, the immutable commit, not str(result).
                resolved = resolve_revision(
                    self.config.model_id,
                    revision=self._revision,
                    cache_dir=str(self.config.cache_dir),
                    local_files_only=self.config.local_files_only,
                )
                self._revision = resolved.resolved
            self._hf_config = AutoConfig.from_pretrained(
                self.config.model_id, **self._hub_kwargs()
            )
        return self._hf_config

    def describe(self) -> dict:
        """Resolve checkpoint identity and text layout without loading weights."""
        config = self._load_config()
        text = getattr(config, "text_config", config)
        return {
            "model_id": self.config.model_id,
            "revision": self._revision,
            "model_type": text.model_type,
            "num_hidden_layers": text.num_hidden_layers,
        }

    @property
    def tokenizer(self):
        """Fetch tokenizer files on first access; never load model weights here."""
        if self._tokenizer is None:
            self._load_config()
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_id, **self._hub_kwargs()
            )
        return self._tokenizer

    @property
    def loaded(self) -> bool:
        """Inspect weight-loading status without triggering I/O."""
        return self._model is not None

    def _model_kwargs(self, config):
        """One quantization policy for feature extraction and native LM inference."""
        settings = self.config
        kwargs = dict(
            config=config,
            dtype=getattr(torch, settings.dtype),
            # Transformers retains this value in config and reads it inside
            # compiled forward. Dynamo cannot trace StrEnum.startswith.
            attn_implementation=str(settings.attention_backend),
            device_map={"": settings.device},
            **self._hub_kwargs(),
        )
        if settings.quantization == "nf4":
            if not settings.device.startswith("cuda"):
                raise ValueError("The NF4 research path requires a CUDA device")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, settings.dtype),
            )
        return kwargs
