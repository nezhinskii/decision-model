"""Text decoder plus its pretrained output projection, with optional reasoning."""

from dataclasses import dataclass
from enum import StrEnum
import time
import torch
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

from ..backbone.loading import BackboneConfig, ModelSource
from .inputs import ANSWER_SUFFIX, build_prompt, candidate_tokens


class InferenceMode(StrEnum):
    direct = "direct"
    thinking = "thinking"


@dataclass
class NativeLMConfig:
    mode: InferenceMode = InferenceMode.direct
    max_input_tokens: int = 2048
    max_reasoning_tokens: int = 2048
    save_reasoning: bool = False

    def __post_init__(self):
        self.mode = InferenceMode(self.mode)
        if min(self.max_input_tokens, self.max_reasoning_tokens) < 1:
            raise ValueError("Token budgets must be positive")


class StopAtTokens(StoppingCriteria):
    def __init__(self, tokens):
        self.tokens = tokens

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[1] >= len(self.tokens) and input_ids[0, -len(self.tokens):].tolist() == self.tokens


class NativeLM(ModelSource):
    def __init__(self, backbone: BackboneConfig, inference: NativeLMConfig):
        super().__init__(backbone)
        self.inference = inference

    def load(self):
        if self._model is None:
            config = self._load_config()
            # Qwen3.5 exports a text CausalLM with native checkpoint prefix
            # conversion; use it to retain lm_head without loading vision weights.
            if config.model_type == "qwen3_5":
                config = config.text_config
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id, **self._model_kwargs(config))
            self._model.requires_grad_(False).eval()
        return self._model

    def _prepare_input(self, context, question):
        """Build the prompt and token IDs, validating labels before loading weights."""
        thinking = self.inference.mode == InferenceMode.thinking
        prompt = build_prompt(self.tokenizer, context, question, thinking)
        answer_prefix = prompt + ("</think>" + ANSWER_SUFFIX if thinking else "")
        candidate_tokens(self.tokenizer, answer_prefix, len(question.options))
        return prompt, self.tokenizer.encode(prompt, add_special_tokens=False)

    def _generate_reasoning(self, model, inputs, budget):
        """Return the completed answer prefix and reasoning metadata, or None on limit."""
        tokenizer = self.tokenizer
        closing = tokenizer.encode("</think>", add_special_tokens=False)
        if not closing:
            raise ValueError("Empty reasoning boundary")
        # Greedy generation needs no resumed RNG state. Retaining per-token
        # vocabulary scores would unnecessarily increase memory consumption.
        generated = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            do_sample=False,
            max_new_tokens=budget,
            stopping_criteria=StoppingCriteriaList([StopAtTokens(closing)]),
            return_dict_in_generate=False,
            output_scores=False,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                          else tokenizer.eos_token_id),
        )
        reasoning = generated[0, inputs.shape[1]:].tolist()
        metadata = {"reasoning_tokens": len(reasoning)}
        if self.inference.save_reasoning:
            metadata["reasoning"] = tokenizer.decode(
                reasoning, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if reasoning[-len(closing):] != closing:
            return None, {**metadata, "status": "reasoning_limit"}
        # Continue the same assistant message. Re-rendering a chat template could
        # strip or rewrite the generated reasoning and alter the scoring position.
        suffix = tokenizer.encode(ANSWER_SUFFIX, add_special_tokens=False)
        answer_inputs = torch.cat(
            (generated, torch.tensor([suffix], device=inputs.device)), dim=1)
        return answer_inputs, metadata

    def _score_options(self, model, inputs, answer_prefix, num_options):
        """Read native logits at one answer position, preserving original option order."""
        label_ids = candidate_tokens(
            self.tokenizer, answer_prefix, num_options, inputs[0].tolist())
        # Avoid allocating vocabulary logits for every prompt/reasoning token.
        output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                       use_cache=False, logits_to_keep=1)
        logits = output.logits[0, -1].float()
        selected = logits[label_ids]
        mass = (selected.logsumexp(0) - logits.logsumexp(0)).exp().item()
        return {"scores": selected.cpu().tolist(), "label_probability_mass": mass}

    @torch.inference_mode()
    def predict(self, context, question):
        """Prepare input, optionally reason, then score the candidate labels."""
        prompt, tokens = self._prepare_input(context, question)
        result = {"status": "ok", "scores": None, "input_tokens": len(tokens),
                  "reasoning_tokens": 0}
        if len(tokens) > self.inference.max_input_tokens:
            return {**result, "status": "input_too_long"}

        model = self.load()
        device = model.get_input_embeddings().weight.device
        inputs = torch.tensor([tokens], device=device)
        started = time.perf_counter()
        thinking = self.inference.mode == InferenceMode.thinking
        text_config = getattr(model.config, "text_config", model.config)
        limit = getattr(text_config, "max_position_embeddings", None)
        budget = self.inference.max_reasoning_tokens
        if limit is not None:
            # Reserve space for the separator after the generated reasoning.
            separator_size = len(self.tokenizer.encode(ANSWER_SUFFIX, add_special_tokens=False))
            budget = min(budget, limit - len(tokens) - separator_size)
            if len(tokens) > limit or (thinking and budget < 1):
                return {**result, "status": "context_limit"}

        answer_prefix = prompt
        if thinking:
            inputs, metadata = self._generate_reasoning(model, inputs, budget)
            result.update(metadata)
            if inputs is None:
                return {**result, "seconds": time.perf_counter() - started}
            answer_prefix = self.tokenizer.decode(
                inputs[0].tolist(), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if limit is not None and inputs.shape[1] > limit:
            return {**result, "status": "context_limit", "seconds": time.perf_counter() - started}

        result.update(self._score_options(model, inputs, answer_prefix, len(question.options)))
        result["seconds"] = time.perf_counter() - started
        return result
