"""Shared decoder execution preserves gradients; freezing remains an explicit facade."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.backbone.forward import forward_decoder, validate_decoder
from src.backbone.frozen import FrozenBackbone
from src.backbone.inputs import InputBuilder, InputConfig
from src.backbone.loading import BackboneConfig, ModelSource
from src.backbone.pooling import PoolingConfig, extract_features
from src.heads import HeadConfig, build_head


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(512, 8)
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Tanh()) for _ in range(2)])
        self.norm = nn.LayerNorm(8)
        self.config = SimpleNamespace(model_type='qwen3', num_hidden_layers=2,
                                      max_position_embeddings=4096, _attn_implementation='sdpa')
        self.fail = False

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, **kwargs):
        assert kwargs['use_cache'] is False
        assert kwargs['output_hidden_states'] is False
        values = self.embedding(input_ids)
        for layer in self.layers:
            values = layer(values)
            if self.fail:
                raise RuntimeError('interrupted decoder')
        return SimpleNamespace(last_hidden_state=self.norm(values))


def metadata():
    return dict(model_id='tiny', revision='pinned', quantization='none', dtype='float32')


def test_shared_forward_and_pooling_keep_gradient_graph(tokenizer, example):
    decoder = TinyDecoder().train()
    batch = InputBuilder(tokenizer, InputConfig(attention='causal'))(example)
    output = forward_decoder(decoder, batch, (1, -1), metadata())
    assert decoder.training
    assert list(output.hidden_states) == [1, 2]
    assert all(t.requires_grad for t in output.hidden_states.values())
    # Independent numerical reference, including final normalization.
    values = decoder.embedding(batch.input_ids)
    first = decoder.layers[0](values)
    final = decoder.norm(decoder.layers[1](first))
    torch.testing.assert_close(output.hidden_states[1], first)
    torch.testing.assert_close(output.hidden_states[2], final)
    assert not any(layer._forward_hooks for layer in decoder.layers)
    features = extract_features(output, batch, PoolingConfig())
    head = build_head(8, HeadConfig(processor='identity', dim=8, score_dim=4,
                                   readout='adaptive'), 2)
    head(features).square().sum().backward()
    assert decoder.embedding.weight.grad is not None
    assert all(layer[0].weight.grad is not None for layer in decoder.layers)
    assert head.processor.readout.router[0].weight.grad is not None


def test_frozen_facade_matches_shared_forward_without_gradients(tokenizer, example):
    decoder = TinyDecoder()
    frozen = FrozenBackbone.from_model(deepcopy(decoder))
    batch = InputBuilder(tokenizer, InputConfig(attention='causal'))(example)
    expected = forward_decoder(decoder, batch, (1, 2), metadata())
    actual = frozen.forward(batch, (1, -1))
    for layer in (1, 2):
        torch.testing.assert_close(actual.hidden_states[layer], expected.hidden_states[layer])
        assert not actual.hidden_states[layer].requires_grad
    assert not frozen.load().training
    assert not any(p.requires_grad for p in frozen.load().parameters())


def test_shared_forward_removes_hooks_after_failure(tokenizer, example):
    decoder = TinyDecoder().train()
    decoder.fail = True
    batch = InputBuilder(tokenizer, InputConfig(attention='causal'))(example)
    with pytest.raises(RuntimeError, match='interrupted decoder'):
        forward_decoder(decoder, batch, (1, -1), metadata())
    assert decoder.training
    assert not any(layer._forward_hooks for layer in decoder.layers)
    decoder.fail = False
    assert forward_decoder(decoder, batch, (1, -1), metadata()).hidden_states[1].requires_grad


@pytest.mark.parametrize('layers', [(), (0,), (3,), (2, -1), (1, 1)])
def test_invalid_layer_selection_installs_no_hooks(tokenizer, example, layers):
    decoder = TinyDecoder()
    batch = InputBuilder(tokenizer, InputConfig(attention='causal'))(example)
    with pytest.raises(ValueError, match='distinct 1-based'):
        forward_decoder(decoder, batch, layers, metadata())
    assert not any(layer._forward_hooks for layer in decoder.layers)


@pytest.mark.parametrize('multimodal', [False, True])
def test_text_loader_selects_config_without_freezing_or_changing_mode(monkeypatch, multimodal):
    from src.backbone import loading
    text = SimpleNamespace(model_type='qwen3_5_text' if multimodal else 'llama')
    config = SimpleNamespace(model_type='qwen3_5', text_config=text) if multimodal else text
    source = ModelSource(BackboneConfig(model_id='local-test', quantization='none'))
    source._hf_config = config
    decoder = TinyDecoder().train()
    calls = []

    def load(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return decoder

    monkeypatch.setattr(loading.AutoModel, 'from_pretrained', load)
    monkeypatch.setattr(source, '_model_kwargs', lambda selected: {'config': selected})
    assert source.load_text_decoder() is decoder
    assert calls == [('local-test', {'config': text})]
    assert decoder.training and all(p.requires_grad for p in decoder.parameters())


def test_frozen_rejects_unsupported_attention_before_loading(monkeypatch):
    backbone = FrozenBackbone(BackboneConfig(model_id='unused'))
    monkeypatch.setattr(backbone, '_load_config', lambda: SimpleNamespace(model_type='qwen3_5_text'))
    monkeypatch.setattr(backbone, 'load', lambda: pytest.fail('Weights must not load'))
    with pytest.raises(NotImplementedError, match='causal'):
        backbone.forward(SimpleNamespace(config=SimpleNamespace(attention='independent_options')))


def test_frozen_supports_changed_layers_without_persistent_execution_state(tokenizer, example):
    frozen = FrozenBackbone.from_model(TinyDecoder())
    batch = InputBuilder(tokenizer, InputConfig(attention='causal'))(example)
    first = frozen.forward(batch, (1, -1))
    second = frozen.forward(batch, (1, -1))
    for layer in first.hidden_states:
        torch.testing.assert_close(first.hidden_states[layer], second.hidden_states[layer])
    assert list(frozen.forward(batch, (1,)).hidden_states) == [1]
    assert not any(layer._forward_hooks for layer in frozen.load().layers)


def test_forward_uses_cpu_prepared_position_bound_without_tensor_reduction(monkeypatch, tokenizer, example):
    from dataclasses import replace

    decoder = TinyDecoder()
    batch = InputBuilder(tokenizer, InputConfig(attention='causal'))(example)
    expected = int(batch.position_ids.max())
    assert batch.max_position_id == expected

    def forbidden(*args, **kwargs):
        raise AssertionError('Forward must not reduce position_ids or read a device scalar')

    monkeypatch.setattr(batch.position_ids, 'max', forbidden)
    forward_decoder(decoder, batch, (1, -1), metadata())
    # A new batch recomputes its CPU bound; equality with the model limit is invalid.
    changed = replace(batch, position_ids=torch.full_like(batch.position_ids, decoder.config.max_position_embeddings))
    with pytest.raises(ValueError, match='position limit'):
        forward_decoder(decoder, changed, (1, -1), metadata())
    assert not any(layer._forward_hooks for layer in decoder.layers)


def test_forward_keeps_metadata_independent(tokenizer, example):
    decoder = TinyDecoder()
    batch = InputBuilder(tokenizer, InputConfig(attention='causal'))(example)
    source_metadata = metadata()
    output = forward_decoder(decoder, batch, (1, -1), source_metadata)
    output.metadata['layers'].append(99)
    output.metadata['revision'] = 'modified-by-caller'
    second = forward_decoder(decoder, batch, (1, -1), source_metadata)
    assert second.metadata['layers'] == [1, 2]
    assert second.metadata['revision'] == 'pinned'
    assert source_metadata == metadata()


def test_loaded_decoder_layout_is_checked_explicitly():
    model = TinyDecoder()
    model.config.num_hidden_layers = 3
    with pytest.raises(ValueError, match='configured depth'):
        validate_decoder(model)
    del model.layers
    with pytest.raises(NotImplementedError, match='exposing model.layers'):
        validate_decoder(model)


def test_batch_rejects_non_cpu_positions_without_reading_them(tokenizer, example):
    from dataclasses import replace

    batch = InputBuilder(tokenizer, InputConfig(attention='causal'))(example)
    # Meta tensors exercise this guard on CPU-only hosts, without CUDA access.
    with pytest.raises(ValueError, match='positions on CPU'):
        replace(batch, position_ids=torch.empty_like(batch.position_ids, device='meta'))
