"""Lazy frozen text-backbone loading; no decision head or generation logic."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

from .inputs import PackedBatch


@dataclass
class BackboneConfig:
    """Require a checkpoint explicitly; the research backbone is not selected yet."""

    model_id: str
    revision: str | None = None
    cache_dir: Path = Path("artifacts/models/hub")
    local_files_only: bool = False
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    quantization: Literal["none", "nf4"] = "nf4"
    attention_backend: Literal["sdpa", "eager"] = "sdpa"


@dataclass
class BackboneOutput:
    """Selected 1-based depths; the final depth includes final normalization."""

    hidden_states: dict[int, torch.Tensor]
    metadata: dict


class FrozenBackbone:
    """Construction is offline; tokenizer and weight loading happen on demand."""

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
            self._hf_config = AutoConfig.from_pretrained(
                self.config.model_id, **self._hub_kwargs()
            )
            self._revision = (
                getattr(self._hf_config, "_commit_hash", None) or self._revision
            )
        return self._hf_config

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

    def load(self):
        """Load AutoModel, optionally quantizing source weights to NF4 in memory."""
        if self._model is None:
            hf_config = self._load_config()
            # Load only the text decoder of a multimodal Qwen3.5 checkpoint.
            # Transformers handles its language_model checkpoint prefix mapping.
            if hf_config.model_type == "qwen3_5":
                hf_config = hf_config.text_config
            kwargs = dict(
                config=hf_config,
                dtype=getattr(torch, self.config.dtype),
                attn_implementation=self.config.attention_backend,
                device_map={"": self.config.device},
                **self._hub_kwargs(),
            )
            if self.config.quantization == "nf4":
                if not self.config.device.startswith("cuda"):
                    raise ValueError("The NF4 research path requires a CUDA device")
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.dtype),
                )
            self._model = AutoModel.from_pretrained(self.config.model_id, **kwargs)
            self._model.requires_grad_(False).eval()
        return self._model

    @classmethod
    def from_model(cls, model) -> "FrozenBackbone":
        """Wrap an in-memory decoder for fully offline notebook experiments."""
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
        """Capture only requested depths rather than retaining every decoder output."""
        # Check branch support before fetching weights. Native causal execution
        # does not need our custom-mask guarantees or full-attention-only layers.
        hf_config = self._load_config()
        text_config = getattr(hf_config, "text_config", hf_config)
        if batch.config.attention != "causal":
            if text_config.model_type not in {"qwen3", "llama"} or any(
                kind != "full_attention"
                for kind in (getattr(text_config, "layer_types", None) or [])
            ):
                raise NotImplementedError(
                    "Independent branches currently require Qwen3/Llama with full "
                    "attention; use attention='causal' for hybrid decoders such as Qwen3.5"
                )
        model = self.load()
        if not hasattr(model, "layers"):
            raise NotImplementedError(
                f"Layer extraction for {model.config.model_type!r} needs a text "
                "decoder exposing model.layers"
            )
        depth = model.config.num_hidden_layers
        requested = tuple(depth if layer == -1 else layer for layer in layers)
        if (
            not requested
            or len(set(requested)) != len(requested)
            or any(not 1 <= l <= depth for l in requested)
        ):
            raise ValueError(
                f"layers must be distinct 1-based depths in [1, {depth}], or -1 for final"
            )
        if int(batch.position_ids.max()) >= model.config.max_position_embeddings:
            raise ValueError("Branch positions exceed the backbone position limit")
        device = model.get_input_embeddings().weight.device
        dtype = model.get_input_embeddings().weight.dtype
        if batch.config.attention == "causal":
            # Let each layer type build its native causal/recurrent mask. A 4D
            # attention matrix cannot control DeltaNet's recurrent state.
            mask = batch.padding_mask.to(device)
        else:
            if batch.allowed_attention is None:
                raise ValueError("Independent branches require an allowed-attention mask")
            allowed = batch.allowed_attention.to(device)
            if model.config._attn_implementation == "sdpa":
                # SDPA accepts True for allowed edges, avoiding a separate float mask.
                mask = allowed
            else:
                # Eager attention adds the mask directly to attention scores.
                mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
                mask.masked_fill_(~allowed, torch.finfo(dtype).min)
        captured, handles = {}, []
        for layer in requested:
            if layer != depth:

                def capture(_module, _args, output, index=layer):
                    captured[index] = output[0] if isinstance(output, tuple) else output

                handles.append(model.layers[layer - 1].register_forward_hook(capture))
        try:
            result = model(
                input_ids=batch.input_ids.to(device),
                position_ids=batch.position_ids.to(device),
                attention_mask=mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        if depth in requested:
            captured[depth] = result.last_hidden_state
        return BackboneOutput(
            {layer: captured[layer] for layer in requested},
            {
                "model_id": self.config.model_id,
                "revision": self._revision,
                "quantization": self.config.quantization,
                "dtype": self.config.dtype,
                "attention_backend": model.config._attn_implementation,
                "layers": list(requested),
                "num_hidden_layers": depth,
                "final_layer": "after_final_norm",
                "intermediate_layers": "decoder_block_output_before_final_norm",
            },
        )
