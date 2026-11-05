"""Native scoring protocol without pretrained downloads or a GPU."""

from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from torch import nn

from src.backbone.loading import BackboneConfig
from src.native_lm.inputs import build_prompt, candidate_tokens, render_question
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
        return result

    def forward(self, input_ids, logits_to_keep, **kwargs):
        assert logits_to_keep == 1
        assert self.tokenizer.decode(input_ids[0].tolist()).endswith("</think>\n\n")
        self.forward_calls += 1
        logits = torch.zeros(1, 1, 256)
        logits[0, 0, ord("A")] = 1
        logits[0, 0, ord("B")] = 3
        return SimpleNamespace(logits=logits)


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
    with pytest.raises(ValueError, match="options"):
        render_question(example.context, replace(q, options=q.options * 27))


def test_bad_continuation_tokenization_fails():
    tokenizer = Tokenizer()
    assert candidate_tokens(tokenizer, "prefix\n\n", 2) == [65, 66]
    with pytest.raises(ValueError, match="round-trip"):
        candidate_tokens(tokenizer, "prefix", 2, [1])
    tokenizer.encode = lambda text, **kwargs: [0] if text.endswith("A") else [0, 1]
    with pytest.raises(ValueError, match="continuation"):
        candidate_tokens(tokenizer, "prefix", 2)


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
