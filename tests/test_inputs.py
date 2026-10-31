"""Semantic input checks against explicitly expected visibility and positions."""

from dataclasses import replace

import pytest
import torch

from src.backbone.inputs import AttentionMode, InputBuilder, InputConfig, InputTooLongError, build_mask
from src.backbone.features import PoolingConfig, extract_features
from src.backbone.model import BackboneOutput, FrozenBackbone
from conftest import make_example


@pytest.mark.parametrize('mode', list(AttentionMode))
@pytest.mark.parametrize('prompt', ['plain', 'chat'])
def test_branch_visibility_positions_and_padding(tokenizer, example, mode, prompt):
    config = InputConfig(attention=mode, prompt_format=prompt, query_source='decision')
    batch = InputBuilder(tokenizer, config)([example, make_example('short', 'x', 1)])
    sample = batch.samples[0]
    blocks = sample.blocks
    find = lambda role, q=None, o=None: next(
        b for b in blocks if b.role == role and b.question == q and b.option == o
    )
    context = find('context')
    q0, q1 = find('question', 0), find('question', 1)
    a, b = find('option', 0, 0), find('option', 0, 1)
    decision = find('decision', 0)
    mask = build_mask(blocks, len(sample.input_ids), mode)
    assert not mask.triu(1).any()
    assert mask[b.start, context.start]
    assert mask[b.start, q0.start]
    assert bool(mask[b.start, a.start]) == (mode != AttentionMode.independent_options)
    assert bool(mask[q1.start, b.start]) == (mode == AttentionMode.causal)
    assert mask[decision.content_start, a.start] and mask[decision.content_start, b.start]
    positions = sample.positions
    if mode == AttentionMode.causal:
        assert positions == list(range(len(positions)))
        assert batch.allowed_attention is None
    else:
        assert positions[q0.start] == positions[q1.start] == positions[context.stop - 1] + 1
        expected_b = positions[a.start] if mode == AttentionMode.independent_options else positions[a.stop - 1] + 1
        assert positions[b.start] == expected_b
        assert positions[decision.start] == max(positions[a.stop - 1], positions[b.stop - 1]) + 1
        short_len = batch.lengths[1]
        assert not batch.allowed_attention[1, 0, :short_len, short_len:].any()
        assert batch.allowed_attention[1, 0].any(dim=1).all()
    assert not batch.padding_mask[1, batch.lengths[1]:].any()


def test_spans_chat_cache_and_length_boundary(tokenizer, example):
    builder = InputBuilder(tokenizer, InputConfig(prompt_format='chat', query_source='decision'))
    encoded = builder.encode(example)
    builder.encode(example)
    assert tokenizer.template_calls == 1
    decode = lambda block: ''.join(chr(i - 2) for i in encoded.input_ids[block.content_start:block.content_stop])
    question = next(b for b in encoded.blocks if b.role == 'question')
    option = next(b for b in encoded.blocks if b.role == 'option')
    decision = next(b for b in encoded.blocks if b.role == 'decision')
    assert decode(question) == 'Q0?'
    assert decode(option) == 'yes\nCriteria:\nrule'
    assert decode(decision) == ':'
    length = len(encoded.input_ids)
    exact = InputBuilder(tokenizer, replace(builder.config, max_packed_tokens=length))
    assert len(exact.encode(example).input_ids) == length
    limited = InputBuilder(tokenizer, replace(builder.config, max_packed_tokens=length - 1))
    with pytest.raises(InputTooLongError):
        limited.encode(example)
    assert len(limited.encode(example, check_length=False).input_ids) == length
    empty = replace(example, questions=[replace(example.questions[0], text='')])
    with pytest.raises(ValueError, match='empty tokenized question'):
        builder.encode(empty)


@pytest.mark.parametrize('query', ['question', 'decision'])
@pytest.mark.parametrize('pooling', ['last', 'mean'])
def test_extraction_matches_token_span_reference(tokenizer, example, query, pooling):
    batch = InputBuilder(tokenizer, InputConfig(query_source=query))([example, make_example('b', 'x', 1)])
    b, length = batch.input_ids.shape
    # Token indices as values reveal accidental use of headers, separators or padding.
    states = torch.arange(b * length * 3, dtype=torch.float32).reshape(b, length, 3)
    output = BackboneOutput({1: states, 2: states + 1000}, {'num_hidden_layers': 2})
    features = extract_features(output, batch, PoolingConfig(pooling, pooling, True))
    expected_q, expected_o = [], []
    for i, sample in enumerate(batch.samples):
        for block in sample.blocks:
            if block.role not in {query, 'option'}:
                continue
            values = states[i, block.content_start:block.content_stop]
            value = values[-1] if block.role == 'decision' or pooling == 'last' else values.mean(0)
            (expected_q if block.role == query else expected_o).append(value)
        context = next(x for x in sample.blocks if x.role == 'context')
        torch.testing.assert_close(features.contexts[i], states[i, context.content_start:context.content_stop] + 1000)
    torch.testing.assert_close(features.questions[:, 0], torch.stack(expected_q))
    torch.testing.assert_close(features.options[:, 0], torch.stack(expected_o))
    assert features.option_to_question.tolist() == [0, 0, 1, 1, 2, 2]
    assert features.question_to_context.tolist() == [0, 0, 1]


@pytest.mark.parametrize('mode', list(AttentionMode))
def test_real_tiny_decoder_packed_matches_visible_causal_paths(tokenizer, example, mode):
    from transformers import Qwen3Config, Qwen3Model

    config = Qwen3Config(vocab_size=256, hidden_size=16, intermediate_size=24,
                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                        head_dim=8, max_position_embeddings=512, attention_dropout=0.0)
    config._attn_implementation = 'sdpa'
    backbone = FrozenBackbone.from_model(Qwen3Model(config))
    batch = InputBuilder(tokenizer, InputConfig(attention=mode))(example)
    result = backbone.forward(batch, layers=(1, -1))
    mask = build_mask(batch.samples[0].blocks, batch.lengths[0], mode)
    with torch.no_grad():
        for block in batch.samples[0].blocks:
            if block.role not in {'question', 'option'}:
                continue
            endpoint = block.content_stop - 1
            keep = mask[endpoint].nonzero().flatten()
            separate = backbone.load()(input_ids=batch.input_ids[:, keep],
                position_ids=torch.arange(len(keep))[None], use_cache=False).last_hidden_state
            torch.testing.assert_close(result.hidden_states[2][0, endpoint], separate[0, -1], atol=2e-6, rtol=2e-5)
    assert not any(layer._forward_hooks for layer in backbone.load().layers)
