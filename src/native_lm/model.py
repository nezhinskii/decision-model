"""Text decoder plus its pretrained output projection, with optional reasoning."""

from dataclasses import dataclass
from enum import StrEnum
import copy
import importlib
import inspect
import time
import torch
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from tqdm.auto import tqdm

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
        self.runtime_info = None
        self._profile_count = 0

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
        if self.runtime_info is None:
            # Use installed FLA convolutions when Dao kernels are unavailable;
            # see kernels.py for the Windows rationale and monkey-patch scope.
            install_qwen_convolutions(self._model)
            self._report_runtime(self._model)
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

    def _report_runtime(self, model):
        """Report actual loaded modules and resolved kernels, not just requested options."""
        device = model.get_input_embeddings().weight.device
        quantized_modules = [module for module in model.modules() if type(module).__name__ == "Linear4bit"]
        quantized = len(quantized_modules)
        loaded_4bit = bool(getattr(model, "is_loaded_in_4bit", False) or quantized)
        if self.config.quantization == "nf4" and not loaded_4bit:
            raise RuntimeError("NF4 was requested but the loaded model has no 4-bit linear layers")
        footprint = model.get_memory_footprint() / 1024**3 if hasattr(model, "get_memory_footprint") else None
        providers = {}
        module = importlib.import_module(type(model).__module__)
        for name in ("causal_conv1d_fn", "causal_conv1d_update",
                     "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            function = getattr(module, name, None)
            if getattr(function, "kernel_provider", None):
                providers[name] = function.kernel_provider
                continue
            visited = set()
            while callable(function) and id(function) not in visited:
                visited.add(id(function))
                try:
                    bindings = inspect.getclosurevars(function).nonlocals
                except TypeError:
                    break
                if "implementation" in bindings:
                    implementation = bindings["implementation"]
                    provider = f"{implementation.__module__}.{implementation.__name__}"
                    if not bindings.get("is_new_implementation", True):
                        provider = "TORCH_REFERENCE (slow): " + provider
                    providers[name] = provider
                    break
                function = getattr(function, "__wrapped__", None)
        self.runtime_info = dict(
            device=str(device), loaded_in_4bit=loaded_4bit, linear4bit_modules=quantized,
            quantized_compute_dtypes=sorted({str(module.compute_dtype) for module in quantized_modules}),
            gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            embedding_dtype=str(model.get_input_embeddings().weight.dtype),
            weights_gib=footprint, kernels=providers,
            attention_backend=getattr(model.config, "_attn_implementation", "unknown"),
        )
        tqdm.write(f"Native LM runtime: {self.runtime_info}")
        tqdm.write("Cache policy: reuse generation cache; clone state at candidate branches; no full-prefix replay.")
        if any("TORCH_REFERENCE" in value for value in providers.values()):
            tqdm.write("PERFORMANCE: reference PyTorch kernels are active. Cache reuse does not remove this kernel bottleneck.")

    def _generate_reasoning(self, model, inputs, budget):
        """Return answer tokens, generation cache and phase timings (no scoring replay)."""
        tokenizer = self.tokenizer
        closing = tokenizer.encode("</think>", add_special_tokens=False)
        if not closing:
            raise ValueError("Empty reasoning boundary")
        device = inputs.device
        first = {}

        def prefill_start(module, args):
            if not first:
                self._synchronize(device)
                first["started"] = time.perf_counter()

        def prefill_end(module, args, output):
            if "seconds" not in first:
                self._synchronize(device)
                first["seconds"] = time.perf_counter() - first["started"]

        # Time the first generation forward separately from subsequent decode.
        hooks = [model.register_forward_pre_hook(prefill_start), model.register_forward_hook(prefill_end)]
        self._synchronize(device)
        started = time.perf_counter()
        try:
            generated = model.generate(
                input_ids=inputs, attention_mask=torch.ones_like(inputs),
                do_sample=False, use_cache=True, max_new_tokens=budget,
                stopping_criteria=StoppingCriteriaList([StopAtTokens(closing)]),
                return_dict_in_generate=True, output_scores=False, output_logits=False,
                pad_token_id=(tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                              else tokenizer.eos_token_id),
            )
            self._synchronize(device)
        finally:
            for hook in hooks:
                hook.remove()
        elapsed = time.perf_counter() - started
        sequences = generated.sequences
        reasoning = sequences[0, inputs.shape[1]:].tolist()
        prefill_seconds = first.get("seconds", 0.)
        metadata = dict(reasoning_tokens=len(reasoning), generation_seconds=elapsed,
                        prefill_seconds=prefill_seconds,
                        decode_seconds=max(0., elapsed - prefill_seconds),
                        decode_tokens=max(0, len(reasoning) - 1))
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
        profile = dict(cache_reused_tokens=cached, scoring_forward_tokens=tail.shape[1],
                       scoring_forwards=1, cache_branch_copies=0)
        if all(len(sequence) == 1 for sequence in sequences):
            selected = logits[[sequence[0] for sequence in sequences]]
            mass = (selected.logsumexp(0) - logits.logsumexp(0)).exp().item()
            return dict(scores=selected.cpu().tolist(), label_probability_mass=mass, **profile)

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
                    profile["cache_branch_copies"] += 1
                next_logits, branch_cache = self._cached_forward(
                    model, torch.tensor([[token]], device=inputs.device), branch_cache)
                profile["scoring_forward_tokens"] += 1
                profile["scoring_forwards"] += 1
                visit(prefix + (token,), next_logits, branch_cache)

        visit((), logits, root_cache)
        scores = [sum(edges[sequence[:i]][token] for i, token in enumerate(sequence))
                  for sequence in sequences]
        mass = torch.tensor(scores, dtype=torch.float64).logsumexp(0).exp().item()
        return dict(scores=scores, label_probability_mass=mass, **profile)

    def _log_profile(self, result):
        self._profile_count += 1
        if self._profile_count <= 3 or self._profile_count % 25 == 0:
            decode_seconds = result.get("decode_seconds", 0)
            decode_rate = result.get("decode_tokens", 0) / decode_seconds if decode_seconds > 0 else 0
            tqdm.write(
                f"Inference #{self._profile_count}: generation={result.get('generation_seconds', 0):.3f}s "
                f"prefill={result.get('prefill_seconds', 0):.3f}s "
                f"decode={decode_seconds:.3f}s ({decode_rate:.1f} tok/s) "
                f"scoring={result.get('scoring_seconds', 0):.3f}s "
                f"reasoning_tokens={result['reasoning_tokens']} "
                f"reused_cache_tokens={result.get('cache_reused_tokens', 0)} "
                f"scoring_forward_tokens={result.get('scoring_forward_tokens', 0)} "
                f"scoring_forwards={result.get('scoring_forwards', 0)}")

    @torch.inference_mode()
    def predict(self, context, question):
        prompt, tokens, sequences = self._prepare_input(context, question)
        label_reserve = max(len(sequence) for sequence in sequences) - 1
        result = dict(status="ok", scores=None, input_tokens=len(tokens), reasoning_tokens=0,
                      generation_seconds=0., prefill_seconds=0., decode_seconds=0.,
                      decode_tokens=0, scoring_seconds=0.)
        if len(tokens) > self.inference.max_input_tokens:
            return {**result, "status": "input_too_long"}
        model = self.load()
        device = model.get_input_embeddings().weight.device
        inputs = torch.tensor([tokens], device=device)
        self._synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
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
                self._log_profile(result)
                return result
            # Only the stable special-token boundary is needed for label
            # tokenization; generated reasoning IDs and cache remain untouched.
            answer_prefix = "</think>" + ANSWER_SUFFIX
        if limit is not None and inputs.shape[1] + label_reserve > limit:
            return {**result, "status": "context_limit", "seconds": time.perf_counter() - started}
        self._synchronize(device)
        scoring_started = time.perf_counter()
        result.update(self._score_options(model, inputs, answer_prefix, len(question.options), cache))
        self._synchronize(device)
        result["scoring_seconds"] = time.perf_counter() - scoring_started
        result["seconds"] = time.perf_counter() - started
        if device.type == "cuda":
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
            result["reserved_gib"] = torch.cuda.memory_reserved(device) / 1024**3
        self._log_profile(result)
        return result
