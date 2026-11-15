"""Decoder execution and selected hidden states, independent of freezing or adapters."""

from dataclasses import dataclass

import torch

from .inputs import PackedBatch


@dataclass
class BackboneOutput:
    """Selected 1-based depths; the final depth includes final normalization."""

    hidden_states: dict[int, torch.Tensor]
    metadata: dict


def validate_attention(config, attention):
    """Check the chosen attention mode before loading weights or starting a run."""
    text = getattr(config, "text_config", config)
    if attention != "causal" and (
        text.model_type not in {"qwen3", "llama"}
        or any(kind != "full_attention" for kind in (getattr(text, "layer_types", None) or []))
    ):
        raise NotImplementedError(
            "Independent branches currently require Qwen3/Llama with full "
            "attention; use attention='causal' for hybrid decoders such as Qwen3.5"
        )


def validate_decoder(model):
    """Check the loaded layout once, before accepting it as a text backbone."""
    if not hasattr(model, "layers"):
        raise NotImplementedError(
            f"Layer extraction for {model.config.model_type!r} needs a text "
            "decoder exposing model.layers"
        )
    if len(model.layers) != model.config.num_hidden_layers:
        raise ValueError("Decoder layers differ from its configured depth")


def forward_decoder(
    model, batch: PackedBatch, layers: tuple[int, ...], metadata: dict,
) -> BackboneOutput:
    """Run a loaded decoder and collect layers, preserving autograd and model mode.

    The caller validates architecture and attention at loading/setup. Layer numbers
    remain per-call because callers may change them. PackedBatch owns CPU position
    validation; this function never reads a device scalar. Hooks are temporary.
    """
    depth = model.config.num_hidden_layers
    requested = tuple(depth if layer == -1 else layer for layer in layers)
    if (not requested or len(set(requested)) != len(requested)
            or any(not 1 <= layer <= depth for layer in requested)):
        raise ValueError(f"layers must be distinct 1-based depths in [1, {depth}], or -1 for final")
    if batch.max_position_id >= model.config.max_position_embeddings:
        raise ValueError("Branch positions exceed the backbone position limit")
    embedding = model.get_input_embeddings().weight
    device, dtype = embedding.device, embedding.dtype
    if batch.config.attention == "causal":
        # The decoder builds its own causal/recurrent masks from padding.
        mask = batch.padding_mask.to(device)
    else:
        if batch.allowed_attention is None:
            raise ValueError("Independent branches require an allowed-attention mask")
        allowed = batch.allowed_attention.to(device)
        if model.config._attn_implementation == "sdpa":
            mask = allowed
        else:
            mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
            mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    captured, handles = {}, []
    try:
        for layer in requested:
            if layer == depth:
                continue

            def capture(_module, _args, output, index=layer):
                captured[index] = output[0] if isinstance(output, tuple) else output

            handles.append(model.layers[layer - 1].register_forward_hook(capture))
        result = model(
            input_ids=batch.input_ids.to(device),
            position_ids=batch.position_ids.to(device),
            attention_mask=mask,
            use_cache=False, output_hidden_states=False, return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()
    if depth in requested:
        captured[depth] = result.last_hidden_state
    return BackboneOutput(
        {layer: captured[layer] for layer in requested},
        {
            **metadata, "attention_backend": model.config._attn_implementation,
            "layers": list(requested), "num_hidden_layers": depth,
            "final_layer": "after_final_norm",
            "intermediate_layers": "decoder_block_output_before_final_norm",
        },
    )
