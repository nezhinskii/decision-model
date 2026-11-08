"""Text decoder plus its pretrained output projection, with optional reasoning."""

from dataclasses import dataclass
from enum import StrEnum
import copy
import time
import torch
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

from ..backbone.loading import BackboneConfig, ModelSource
from .inputs import ANSWER_SUFFIX, build_prompt, candidate_sequences
from .kernels import install_qwen_convolutions


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
        self.tokens = torch.tensor(tokens, dtype=torch.long)

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < self.tokens.numel():
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if self.tokens.device != input_ids.device:
            self.tokens = self.tokens.to(input_ids.device)
        return (input_ids[:, -self.tokens.numel():] == self.tokens).all(dim=-1)


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
            # Runtime compatibility, not diagnostics; see kernels.py.
            install_qwen_convolutions(self._model)
            if self.config.quantization == "nf4" and not (
                getattr(self._model, "is_loaded_in_4bit", False)
                or any(type(m).__name__ == "Linear4bit" for m in self._model.modules())
            ):
                raise RuntimeError("NF4 was requested but the loaded model has no 4-bit linear layers")
        return self._model

    def _prepare_input(self, context, question):
        """Build the prompt and token IDs, validating labels before loading weights."""
        thinking = self.inference.mode == InferenceMode.thinking
        prompt = build_prompt(self.tokenizer, context, question, thinking)
        answer_prefix = prompt + ("</think>" + ANSWER_SUFFIX if thinking else "")
        sequences = candidate_sequences(self.tokenizer, answer_prefix, len(question.options))
        return prompt, self.tokenizer.encode(prompt, add_special_tokens=False), sequences

    @staticmethod
    def _synchronize(device):
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _generate_reasoning(self, model, inputs, budget):
        """Return answer tokens, generation cache and optional reasoning text."""
        tokenizer = self.tokenizer
        closing = tokenizer.encode("</think>", add_special_tokens=False)
        if not closing:
            raise ValueError("Empty reasoning boundary")
        device = inputs.device
        generated = model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs),
            do_sample=False, use_cache=True, max_new_tokens=budget,
            stopping_criteria=StoppingCriteriaList([StopAtTokens(closing)]),
            return_dict_in_generate=True, output_scores=False, output_logits=False,
            pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                          else tokenizer.eos_token_id),
        )
        sequences = generated.sequences
        reasoning = sequences[0, inputs.shape[1]:].tolist()
        metadata = dict(reasoning_tokens=len(reasoning))
        if self.inference.save_reasoning:
            metadata["reasoning"] = tokenizer.decode(
                reasoning, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if reasoning[-len(closing):] != closing:
            return None, None, {**metadata, "status": "reasoning_limit"}
        cache = generated.past_key_values
        if cache is None:
            raise RuntimeError("Generation returned no cache despite use_cache=True")
        cached = int(cache.get_seq_length())
        if not inputs.shape[1] <= cached <= sequences.shape[1]:
            raise RuntimeError(f"Unexpected generation cache length: {cached}")
        # generate normally leaves the last sampled token unprocessed. The caller
        # feeds only that uncached tail plus the answer separator into this cache.
        suffix = tokenizer.encode(ANSWER_SUFFIX, add_special_tokens=False)
        answer_inputs = torch.cat((sequences, torch.tensor([suffix], device=device)), dim=1)
        return answer_inputs, cache, metadata

    def _cached_forward(self, model, tokens, cache=None):
        """Append only new tokens to KV/recurrent state and return next-token logits."""
        cached = int(cache.get_seq_length()) if cache is not None else 0
        output = model(input_ids=tokens, past_key_values=cache, use_cache=True,
                       attention_mask=torch.ones((1, cached + tokens.shape[1]),
                                                 dtype=torch.long, device=tokens.device),
                       logits_to_keep=1, return_dict=True)
        if output.past_key_values is None:
            raise RuntimeError("Scoring forward did not return the requested cache")
        return output.logits[0, -1].float(), output.past_key_values

    def _score_options(self, model, inputs, answer_prefix, num_options, cache=None):
        """Score full labels with shared cached prefixes and isolated branch state."""
        sequences = candidate_sequences(self.tokenizer, answer_prefix, num_options, inputs[0].tolist())
        cached = int(cache.get_seq_length()) if cache is not None else 0
        if not 0 <= cached < inputs.shape[1]:
            raise RuntimeError("Scoring requires an uncached answer suffix")
        tail = inputs[:, cached:]
        logits, root_cache = self._cached_forward(model, tail, cache)
        if all(len(sequence) == 1 for sequence in sequences):
            selected = logits[[sequence[0] for sequence in sequences]]
            mass = (selected.logsumexp(0) - logits.logsumexp(0)).exp().item()
            return dict(scores=selected.cpu().tolist(), label_probability_mass=mass)

        continuations = {}
        for sequence in sequences:
            for i, token in enumerate(sequence):
                continuations.setdefault(sequence[:i], set()).add(token)
        edges = {}

        def visit(prefix, prefix_logits, prefix_cache):
            children = sorted(continuations[prefix])
            values = prefix_logits.log_softmax(0)[children].cpu().tolist()
            edges[prefix] = dict(zip(children, values))
            branches = [token for token in children if prefix + (token,) in continuations]
            for i, token in enumerate(branches):
                # Hybrid recurrent states mutate in place. Fork before siblings;
                # consume the parent only for its last branch. DFS bounds memory
                # by label depth, rather than retaining a cache for every option.
                branch_cache = prefix_cache
                if i < len(branches) - 1:
                    branch_cache = copy.deepcopy(prefix_cache)
                next_logits, branch_cache = self._cached_forward(
                    model, torch.tensor([[token]], device=inputs.device), branch_cache)
                visit(prefix + (token,), next_logits, branch_cache)

        visit((), logits, root_cache)
        scores = [sum(edges[sequence[:i]][token] for i, token in enumerate(sequence))
                  for sequence in sequences]
        mass = torch.tensor(scores, dtype=torch.float64).logsumexp(0).exp().item()
        return dict(scores=scores, label_probability_mass=mass)

    @torch.inference_mode()
    def predict(self, context, question):
        prompt, tokens, sequences = self._prepare_input(context, question)
        label_reserve = max(len(sequence) for sequence in sequences) - 1
        result = dict(status="ok", scores=None, input_tokens=len(tokens), reasoning_tokens=0)
        if len(tokens) > self.inference.max_input_tokens:
            return {**result, "status": "input_too_long"}
        model = self.load()
        device = model.get_input_embeddings().weight.device
        inputs = torch.tensor([tokens], device=device)
        self._synchronize(device)
        started = time.perf_counter()
        thinking = self.inference.mode == InferenceMode.thinking
        text_config = getattr(model.config, "text_config", model.config)
        limit = getattr(text_config, "max_position_embeddings", None)
        budget = self.inference.max_reasoning_tokens
        if limit is not None:
            separator_size = len(self.tokenizer.encode(ANSWER_SUFFIX, add_special_tokens=False))
            budget = min(budget, limit - len(tokens) - separator_size - label_reserve)
            if len(tokens) + label_reserve > limit or (thinking and budget < 1):
                return {**result, "status": "context_limit"}
        answer_prefix, cache = prompt, None
        if thinking:
            inputs, cache, metadata = self._generate_reasoning(model, inputs, budget)
            result.update(metadata)
            if inputs is None:
                result["seconds"] = time.perf_counter() - started
                return result
            # Only the stable special-token boundary is needed for label
            # tokenization; generated reasoning IDs and cache remain untouched.
            answer_prefix = "</think>" + ANSWER_SUFFIX
        if limit is not None and inputs.shape[1] + label_reserve > limit:
            return {**result, "status": "context_limit", "seconds": time.perf_counter() - started}
        self._synchronize(device)
        result.update(self._score_options(model, inputs, answer_prefix, len(question.options), cache))
        self._synchronize(device)
        result["seconds"] = time.perf_counter() - started
        return result
