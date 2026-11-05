"""Lazy frozen text-backbone loading; no decision head or generation logic."""

from dataclasses import dataclass

import torch
from transformers import AutoModel

from .loading import BackboneConfig, ModelSource

from .inputs import PackedBatch


@dataclass
class BackboneOutput:
    """Selected 1-based depths; the final depth includes final normalization."""

    hidden_states: dict[int, torch.Tensor]  # 1-based depth -> [B, L_max, D].
    metadata: dict


class FrozenBackbone(ModelSource):
    """Frozen decoder with selective hidden-state extraction."""

    def load(self):
        """Load AutoModel, optionally quantizing source weights to NF4 in memory."""
        if self._model is None:
            hf_config = self._load_config()
            # Load only the text decoder of a multimodal Qwen3.5 checkpoint.
            # Transformers handles its language_model checkpoint prefix mapping.
            if hf_config.model_type == "qwen3_5":
                hf_config = hf_config.text_config
            kwargs = self._model_kwargs(hf_config)
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
                raise ValueError(
                    "Independent branches require an allowed-attention mask"
                )
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
