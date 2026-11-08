"""Execution contract using real tiny CPU decoders; no downloads/CUDA needed."""
from dataclasses import replace
from types import SimpleNamespace
import inspect
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.backbone.loading import BackboneConfig
from src.native_lm.model import NativeLM, NativeLMConfig
from src.native_lm import execution


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    def encode(self, text, **kwargs):
        return [ord(c) for c in text.replace("</think>", "\x01").replace("<think>", "\x02")]
    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids)).replace('\x01', '</think>').replace('\x02', '<think>')
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content'] + ('<think>\n' if kwargs['enable_thinking'] else '</think>\n\n')


def wrapper(model=None):
    torch.manual_seed(17)
    if model is None:
        model = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, head_dim=8, max_position_embeddings=8192))
    model = model.eval().requires_grad_(False)
    model.generation_config.eos_token_id = 1
    model.generation_config.pad_token_id = 0
    model.generation_config.forced_eos_token_id = 1
    lm = NativeLM(BackboneConfig('offline', device='cpu', quantization='none'),
                  NativeLMConfig(mode='thinking', execution='compiled_decode',
                                 max_input_tokens=4096, max_reasoning_tokens=8))
    lm._model, lm._tokenizer = model, Tokenizer()
    return lm


@pytest.fixture
def cpu_compiler(monkeypatch):
    real_compile = torch.compile
    created, calls = [], []
    def compile(function, **kwargs):
        created.append(kwargs)
        kwargs.pop('mode', None)
        compiled = real_compile(function, backend='eager', **kwargs)
        def forward(*args, **kw):
            calls.append(kw['input_ids'].shape[1])
            return compiled(*args, **kw)
        return forward
    monkeypatch.setattr(execution, '_require_cuda', lambda model: None)
    monkeypatch.setattr(execution.torch, 'compile', compile)
    yield created, calls
    torch._dynamo.reset()


def test_reuses_compiler_and_clears_static_cache(example, cpu_compiler):
    lm = wrapper()
    first = lm.predict(example.context, example.questions[0])
    cache = lm._execution.cache
    second = lm.predict(example.context + ' extra context', example.questions[0])
    assert first['status'] == second['status'] == 'ok'
    assert lm._execution.cache is cache
    assert not lm._execution.active
    created, calls = cpu_compiler
    assert len(created) == 1
    assert calls and set(calls) == {1}
    assert len(calls) == first['reasoning_tokens'] + second['reasoning_tokens'] - 2
    fresh = wrapper()
    repeated = fresh.predict(example.context + ' extra context', example.questions[0])
    torch.testing.assert_close(torch.tensor(second['scores']), torch.tensor(repeated['scores']), rtol=0, atol=0)


def test_hybrid_scoring_branches_after_compiled_reasoning(example, cpu_compiler, monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    for name in ('causal_conv1d_fn', 'causal_conv1d_update',
                 'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'):
        monkeypatch.setattr(qwen, name, inspect.unwrap(getattr(qwen, name)))
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, layer_types=['linear_attention', 'full_attention'],
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        max_position_embeddings=8192,
        rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                         'partial_rotary_factor': 1., 'mrope_section': [2, 3, 3]}))
    lm = wrapper(model)
    q = replace(example.questions[0], options=example.questions[0].options * 27)
    result = lm.predict(example.context, q)
    assert result['status'] == 'ok' and len(result['scores']) == 54
    assert all(torch.isfinite(torch.tensor(result['scores'])))
    # Scoring is eager: even one-token label branches do not enter compilation.
    assert len(cpu_compiler[1]) == result['reasoning_tokens'] - 1
    again = lm.predict('different context', q)
    assert again['status'] == 'ok' and len(again['scores']) == 54


def test_compile_failure_is_explicit_and_restores_eager_routing(example, monkeypatch):
    lm = wrapper()
    monkeypatch.setattr(execution, '_require_cuda', lambda model: None)
    def compiler(*args, **kwargs):
        def fail(*args, **kwargs):
            raise RuntimeError('deliberate backend failure')
        return fail
    monkeypatch.setattr(execution.torch, 'compile', compiler)
    with pytest.raises(RuntimeError, match='no eager fallback'):
        lm.predict(example.context, example.questions[0])
    assert not lm._execution.active
    lm.inference = NativeLMConfig(mode='direct')
    assert lm.predict(example.context, example.questions[0])['status'] == 'ok'


def test_capabilities_not_checkpoint_name():
    lm = wrapper()
    with pytest.raises(ValueError, match='CUDA'):
        execution.DecodeExecution(lm.load())
    with pytest.raises(ValueError, match='thinking'):
        NativeLMConfig(mode='direct', execution='compiled_decode')


def test_noncompileable_cache_is_rejected_without_attaching(monkeypatch):
    lm = wrapper()
    model = lm.load()
    monkeypatch.setattr(execution, '_require_cuda', lambda model: None)
    monkeypatch.setattr(execution, 'StaticCache', lambda **kwargs: SimpleNamespace(is_compileable=False))
    controller = execution.DecodeExecution(model)
    original = model.forward
    with pytest.raises(ValueError, match='not compileable'):
        with controller.generation(128):
            pytest.fail('must reject cache')
    assert model.forward == original and not controller.active


@pytest.mark.parametrize("limit_delta", [0, -1])
def test_static_capacity_includes_label_branch_at_position_limit(example, cpu_compiler, limit_delta):
    lm = wrapper()
    q = replace(example.questions[0], options=example.questions[0].options * 27)
    _, tokens, labels = lm._prepare_input(example.context, q)
    lm.inference.max_input_tokens = len(tokens)
    reserve = max(map(len, labels)) - 1
    required = len(tokens) + lm.inference.max_reasoning_tokens + 2 + reserve
    lm._model.config.max_position_embeddings = required + limit_delta
    result = lm.predict(example.context, q)
    assert result['status'] == 'ok' and len(result['scores']) == 54
    assert lm._execution.capacity == required + limit_delta
    assert int(lm._execution.cache.get_seq_length()) <= lm._execution.capacity


def test_generate_cannot_silently_replace_static_cache(example, cpu_compiler, monkeypatch):
    import copy
    lm = wrapper()
    generate = lm._model.generate
    def replaced(**kwargs):
        output = generate(**kwargs)
        output.past_key_values = copy.deepcopy(output.past_key_values)
        return output
    monkeypatch.setattr(lm._model, 'generate', replaced)
    with pytest.raises(RuntimeError, match='retain the supplied static cache'):
        lm.predict(example.context, example.questions[0])
    assert not lm._execution.active
