"""Native scoring protocol without pretrained downloads or a GPU."""

from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from torch import nn

from src.backbone.loading import BackboneConfig
from src.native_lm.inputs import build_prompt, candidate_sequences, option_labels, render_question
from src.native_lm.model import NativeLM, NativeLMConfig, StopAtTokens


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        end = "<think>\n" if kwargs["enable_thinking"] else "<think>\n\n</think>\n\n"
        return messages[0]["content"] + "\n<assistant>\n" + end


class Cache:
    def __init__(self, tokens):
        self.tokens = tokens.clone()

    def get_seq_length(self):
        return self.tokens.shape[1]


class Model(nn.Module):
    def __init__(self, tokenizer, close=True):
        super().__init__()
        self.embedding = nn.Embedding(256, 2)
        self.config = SimpleNamespace(max_position_embeddings=8192)
        self.tokenizer = tokenizer
        self.close = close
        self.forward_calls = 0

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, input_ids, stopping_criteria, **kwargs):
        text = "evidence</think>" if self.close else "unfinished"
        tail = torch.tensor([self.tokenizer.encode(text)])
        result = torch.cat((input_ids, tail), dim=1)
        assert bool(stopping_criteria(result, None)[0]) == self.close
        assert kwargs["use_cache"] and kwargs["return_dict_in_generate"]
        return SimpleNamespace(sequences=result, past_key_values=Cache(result[:, :-1]))

    def forward(self, input_ids, logits_to_keep, **kwargs):
        assert logits_to_keep == 1
        cache = kwargs.get("past_key_values")
        tokens = torch.cat((cache.tokens, input_ids), dim=1) if cache is not None else input_ids
        assert self.tokenizer.decode(tokens[0].tolist()).endswith("</think>\n\n")
        self.last_forward_tokens = input_ids.shape[1]
        self.forward_calls += 1
        logits = torch.zeros(1, 1, 256)
        logits[0, 0, ord("A")] = 1
        logits[0, 0, ord("B")] = 3
        return SimpleNamespace(logits=logits, past_key_values=Cache(tokens))


def wrapper(mode="direct", close=True, **kwargs):
    lm = NativeLM(BackboneConfig("fake", device="cpu", quantization="none"),
                  NativeLMConfig(mode=mode, **kwargs))
    lm._tokenizer = Tokenizer()
    lm._model = Model(lm._tokenizer, close)
    return lm


@pytest.mark.parametrize("mode", ["direct", "thinking"])
def test_scores_address_original_options_and_reasoning_boundary(example, mode):
    lm = wrapper(mode)
    result = lm.predict(example.context, example.questions[0])
    assert result["status"] == "ok"
    assert result["scores"] == [1, 3]
    assert 0 < result["label_probability_mass"] < 1
    assert result["reasoning_tokens"] > 0 if mode == "thinking" else result["reasoning_tokens"] == 0
    assert "reasoning" not in result
    if mode == "thinking":
        assert result["cache_reused_tokens"] > result["input_tokens"]
        assert result["scoring_forward_tokens"] == 3
        assert lm._model.last_forward_tokens == 3


def test_reasoning_limit_is_not_silently_scored(example):
    lm = wrapper("thinking", close=False, save_reasoning=True)
    result = lm.predict(example.context, example.questions[0])
    assert result["status"] == "reasoning_limit"
    assert result["scores"] is None
    assert result["reasoning"] == "unfinished"
    assert lm._model.forward_calls == 0


def test_long_input_skips_before_loading_weights(example):
    lm = wrapper(max_input_tokens=1)
    lm.load = lambda: pytest.fail("must not load weights")
    assert lm.predict(example.context, example.questions[0])["status"] == "input_too_long"


def test_prompt_omits_targets_and_contains_criteria(example):
    q = example.questions[0]
    changed = replace(q, target=replace(q.target, probabilities=[.9, .1]))
    assert render_question(example.context, q) == render_question(example.context, changed)
    assert "Criteria: rule" in render_question(example.context, q)
    assert "A: yes" in render_question(example.context, q)
    assert "01: yes" in render_question(example.context, replace(q, options=q.options * 27))
    with pytest.raises(ValueError, match="option"):
        render_question(example.context, replace(q, options=[]))


def test_bad_continuation_tokenization_fails():
    tokenizer = Tokenizer()
    assert candidate_sequences(tokenizer, "prefix\n\n", 2) == [(65,), (66,)]
    with pytest.raises(ValueError, match="expected boundary"):
        candidate_sequences(tokenizer, "prefix", 2, [1])
    tokenizer.encode = lambda text, **kwargs: [0] if text.endswith("A") else [0, 1]
    with pytest.raises(ValueError, match="continuation"):
        candidate_sequences(tokenizer, "prefix", 2)


def test_unsupported_template_fails(example):
    tokenizer = Tokenizer()
    tokenizer.apply_chat_template = lambda *args, **kwargs: "no thinking boundary"
    with pytest.raises(ValueError, match="template"):
        build_prompt(tokenizer, example.context, example.questions[0], False)


def test_stop_matches_complete_closing_sequence_only():
    stop = StopAtTokens([2, 3])
    assert not stop(torch.tensor([[1, 2]]), None)
    assert stop(torch.tensor([[1, 2, 3]]), None)


def test_text_lm_loads_native_weights_from_multimodal_checkpoint(tmp_path):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    config = Qwen3_5Config(
        text_config=dict(vocab_size=256, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16,
                         layer_types=["full_attention"],
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                                          "partial_rotary_factor": 1., "mrope_section": [2, 3, 3]}),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
                           num_heads=4, out_hidden_size=32, num_position_embeddings=16))
    source = Qwen3_5ForConditionalGeneration(config).eval()
    source.save_pretrained(tmp_path)
    lm = NativeLM(BackboneConfig(str(tmp_path), device="cpu", dtype="float32",
                                quantization="none", local_files_only=True), NativeLMConfig())
    loaded = lm.load()
    assert not hasattr(loaded.model, "visual")
    torch.testing.assert_close(loaded.lm_head.weight, source.lm_head.weight)
    torch.testing.assert_close(loaded.model.embed_tokens.weight,
                               source.model.language_model.embed_tokens.weight)
    tokens = torch.tensor([[10, 11, 12]])
    with torch.no_grad():
        expected = source(input_ids=tokens, logits_to_keep=1).logits
        actual = loaded(input_ids=tokens, logits_to_keep=1).logits
    torch.testing.assert_close(actual, expected)


def test_large_candidate_set_uses_complete_sequence_probabilities():
    lm = wrapper()
    class SequenceModel(Model):
        def forward(self, input_ids, **kwargs):
            self.forward_calls += 1
            logits = torch.zeros(1, 1, 256)
            cache = kwargs.get("past_key_values")
            tokens = torch.cat((cache.tokens, input_ids), dim=1) if cache is not None else input_ids
            root = self.tokenizer.decode(tokens[0].tolist()).endswith("</think>\n\n")
            assert root or tokens.shape[1] == len("</think>\n\n") + 1
            logits[0, 0, ord("5") if root else ord("4")] = 4 if root else 3
            return SimpleNamespace(logits=logits, past_key_values=Cache(tokens))
    lm._model = SequenceModel(lm.tokenizer)
    prefix = "</think>\n\n"
    inputs = torch.tensor([lm.tokenizer.encode(prefix)])
    result = lm._score_options(lm._model, inputs, prefix, 54)
    root, following = torch.zeros(256), torch.zeros(256)
    root[ord("5")], following[ord("4")] = 4, 3
    root, following = root.log_softmax(0), following.log_softmax(0)
    expected = [float(root[ord(label[0])] + following[ord(label[1])])
                for label in option_labels(54)]
    torch.testing.assert_close(torch.tensor(result["scores"]), torch.tensor(expected))
    assert int(torch.tensor(result["scores"]).argmax()) == 53
    assert result["scoring_forward_tokens"] == len(prefix) + 6
    assert lm._model.forward_calls == 7  # Root plus six distinct first digits.
    assert result["label_probability_mass"] == pytest.approx(float(torch.tensor(expected).exp().sum()))


@torch.inference_mode()
def test_hybrid_cache_branches_match_full_prefix_scoring(monkeypatch):
    import inspect
    from transformers.models.qwen3_5 import modeling_qwen3_5 as modeling
    # Installed FLA kernels require CUDA; this correctness test stays on CPU.
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                 "causal_conv1d_fn", "causal_conv1d_update"):
        monkeypatch.setattr(modeling, name, inspect.unwrap(getattr(modeling, name)))
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    torch.manual_seed(13)
    config = Qwen3_5TextConfig(
        vocab_size=256, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 1., "mrope_section": [2, 3, 3]})
    model = Qwen3_5ForCausalLM(config).eval()
    lm = wrapper()
    prefix = "</think>\n\n"
    inputs = torch.tensor([lm.tokenizer.encode(prefix)])
    actual = lm._score_options(model, inputs, prefix, 54)
    expected = []
    for label in option_labels(54):
        tokens = torch.tensor([lm.tokenizer.encode(prefix + label[:-1])])
        logits = model(tokens, use_cache=False).logits[0].float().log_softmax(-1)
        expected.append(sum(logits[len(prefix) - 1 + i, ord(c)] for i, c in enumerate(label)))
    torch.testing.assert_close(torch.tensor(actual["scores"]), torch.stack(expected), atol=2e-5, rtol=2e-5)
    cached = model(inputs[:, :-3], use_cache=True).past_key_values
    reused = lm._score_options(model, inputs, prefix, 54, cached)
    torch.testing.assert_close(torch.tensor(reused["scores"]), torch.stack(expected), atol=2e-5, rtol=2e-5)
    assert reused["scoring_forward_tokens"] == 3 + 6

    original_generate = model.generate
    forced = lm.tokenizer.encode("x</think>")
    def generate(**kwargs):
        return original_generate(**kwargs, prefix_allowed_tokens_fn=lambda _, ids: [
            forced[len(ids) - inputs.shape[1]]])
    monkeypatch.setattr(model, "generate", generate)
    answer, cache, timing = lm._generate_reasoning(model, inputs, len(forced))
    answer_prefix = lm.tokenizer.decode(answer[0].tolist())
    reused = lm._score_options(model, answer, answer_prefix, 54, cache)
    full = lm._score_options(model, answer, answer_prefix, 54)
    torch.testing.assert_close(torch.tensor(reused["scores"]), torch.tensor(full["scores"]),
                               atol=2e-5, rtol=2e-5)
    assert reused["cache_reused_tokens"] == answer.shape[1] - 3
    assert reused["scoring_forward_tokens"] == 3 + 6
    assert timing["reasoning_tokens"] == len(forced)
    assert timing["prefill_seconds"] > 0


def test_requested_nf4_must_actually_be_loaded():
    lm = wrapper()
    lm.config.quantization = "nf4"
    with pytest.raises(RuntimeError, match="no 4-bit"):
        lm.load()


def test_noncanonical_reasoning_does_not_require_tokenizer_round_trip(example):
    lm = wrapper("thinking")
    decode = lm.tokenizer.decode
    # The original generated token IDs still contain "evidence". Decoding them
    # has a different canonical spelling, as can happen with real tokenizers.
    lm.tokenizer.decode = lambda ids, **kwargs: decode(ids, **kwargs).replace("evidence", "canonical")
    result = lm.predict(example.context, example.questions[0])
    assert result["status"] == "ok"
    assert result["scores"] == [1, 3]
    assert result["scoring_forward_tokens"] == 3
    assert result["cache_reused_tokens"] > result["input_tokens"]
