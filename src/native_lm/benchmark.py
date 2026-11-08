"""Complete native-LM benchmark, diagnostics and offline self-tests in ONE file.

Default: python -m src.native_lm.benchmark (one real smoke call)
Full matrix: python -m src.native_lm.benchmark --full
Other checkpoint: --model-id Qwen/Qwen3.5-27B --revision COMMIT
CPU verification of this tool: --self-test

Matrix: none/NF4 x BF16/FP16/FP32, plus torch-reference FP32; each x dynamic/static x eager/compile_decode/compile_all.
With --full, every row checks prefill, teacher-forced decode, direct and thinking answer
scoring (4 and 54 labels), then measures warmed complete pipelines. Compiled
paths do not silently fall back. Errors/OOM are data; other rows continue.
Only report.json is retained. Workers and tensor baselines are temporary.
This imports production loading/prompt/scoring code, never evaluation/tracking.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import gc
import functools
import hashlib
import importlib.metadata
import importlib
import inspect
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import weakref

import torch
import tyro
from transformers import DynamicCache, StaticCache

from src.backbone.loading import BackboneConfig
from src.data.schema import Option, Question, Target, TargetKind, Task
from src.native_lm.inputs import ANSWER_SUFFIX, build_prompt, candidate_sequences
from src.native_lm.model import NativeLM, NativeLMConfig
from src.native_lm.execution import ExecutionMode


# Benchmark-only opaque boundaries for FLA public dispatchers. Their Python
# wrappers are compiler-disabled even though the contained GPU kernels can run
# inside a captured model graph. Production inference is not patched globally
# outside this benchmark's isolated worker process.
_BENCH_IMPLEMENTATIONS = {}


@torch.library.custom_op("decision_benchmark::conv_prefill", mutates_args=())
def _conv_prefill(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None,
                  activation: str | None) -> torch.Tensor:
    return _BENCH_IMPLEMENTATIONS["conv"](x, weight, bias, activation=activation).clone(memory_format=torch.contiguous_format)


@_conv_prefill.register_fake
def _fake_conv_prefill(x, weight, bias, activation):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


@torch.library.custom_op("decision_benchmark::delta_chunk", mutates_args=())
def _delta_chunk(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                 g: torch.Tensor, beta: torch.Tensor, initial: torch.Tensor | None,
                 normalize: bool, chunk_size: int, cu_seqlens: torch.Tensor | None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    output, state = _BENCH_IMPLEMENTATIONS["chunk"](
        q, k, v, g=g, beta=beta, initial_state=initial,
        output_final_state=True, use_qk_l2norm_in_kernel=normalize,
        chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    if state is None:
        raise RuntimeError("Chunk kernel did not return requested state")
    return output.clone(memory_format=torch.contiguous_format), state.clone(memory_format=torch.contiguous_format)


@_delta_chunk.register_fake
def _fake_delta_chunk(q, k, v, g, beta, initial, normalize, chunk_size, cu_seqlens):
    batch = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    return (torch.empty_like(v, memory_format=torch.contiguous_format),
            torch.empty((batch, v.shape[2], k.shape[-1], v.shape[-1]),
                        device=q.device, dtype=torch.float32))


def install_benchmark_boundaries(model=None, module=None):
    if module is None:
        module = importlib.import_module(type(model).__module__)
    if "qwen3_5" not in module.__name__:
        return []
    installed = []
    original_conv = module.causal_conv1d_fn
    if not getattr(original_conv, "benchmark_boundary", False):
        _BENCH_IMPLEMENTATIONS["conv"] = original_conv
        def conv(x, weight, bias=None, activation=None, **kwargs):
            if torch.compiler.is_compiling():
                # Both the project's convolution bridge and Qwen's torch
                # fallback ignore these decoder-level kwargs (e.g. use_cache).
                return _conv_prefill(x, weight, bias, activation)
            return original_conv(x, weight, bias, activation=activation, **kwargs)
        conv.benchmark_boundary = True
        conv.kernel_provider = "benchmark opaque boundary; eager uses original convolution"
        module.causal_conv1d_fn = conv
        installed.append("causal_conv1d_fn")
    original_chunk = module.torch_chunk_gated_delta_rule
    if not getattr(original_chunk, "benchmark_boundary", False):
        _BENCH_IMPLEMENTATIONS["chunk"] = original_chunk
        def chunk(q, k, v, g, beta, chunk_size=64, initial_state=None,
                  output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
            if torch.compiler.is_compiling():
                cu_seqlens = kwargs.pop("cu_seqlens", None)
                for name in ("use_cache", "output_attentions", "output_hidden_states"):
                    kwargs.pop(name, None)
                if any(value is not None for value in kwargs.values()):
                    raise NotImplementedError("Nonstandard DeltaNet metadata in compiled benchmark")
                output, state = _delta_chunk(q, k, v, g, beta, initial_state,
                    use_qk_l2norm_in_kernel, chunk_size, cu_seqlens)
                return output, state if output_final_state else None
            return original_chunk(q, k, v, g=g, beta=beta, chunk_size=chunk_size,
                initial_state=initial_state, output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)
        chunk.benchmark_boundary = True
        chunk.kernel_provider = "benchmark opaque boundary; eager uses original chunk DeltaNet"
        module.torch_chunk_gated_delta_rule = chunk
        installed.append("torch_chunk_gated_delta_rule")
    return installed


def brief(error):
    return str(error).splitlines()[0][:240]


@dataclass
class Config:
    model_id: str = "Qwen/Qwen3.5-4B"
    revision: str | None = None  # Resolve once and reuse the commit across weight workers.
    cache_dir: Path = Path("artifacts/models/hub")
    device: str = "cuda:0"
    output: Path | None = None  # Default report path follows smoke/full mode.
    weights: tuple[str, ...] = ("bf16", "nf4", "fp16", "nf4_fp16", "nf4_fp32", "fp32", "fp32_reference")
    caches: tuple[str, ...] = ("dynamic", "static")
    executions: tuple[str, ...] = ("eager", "compile_decode", "compile_all")
    validation_tokens: int = 128
    timing_tokens: int = 256
    repeats: int = 3
    option_counts: tuple[int, ...] = (4, 54)
    recompile_limit: int = 64  # Finite budget for deliberately varied prompt/label shapes.
    profile: bool = False  # Operator tables embedded in the same JSON, no traces.
    full: bool = False  # Explicit opt-in to the full, expensive diagnostic matrix.
    smoke_tokens: int = 2048
    self_test: bool = False
    worker_spec: Path | None = None  # Internal subprocess protocol.

    @property
    def native_smoke(self):
        return not self.full


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            # Windows readers can briefly hold a handle while the parent
            # mirrors a worker snapshot. Retry replacement, never truncate.
            if attempt == 19:
                raise
            time.sleep(0.05)


def difference(actual, expected):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} != {expected.shape}")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError("Nonfinite logits/cache")
    delta = actual - expected
    rms = delta.square().mean().sqrt().item()
    reference = expected.square().mean().sqrt().item()
    return dict(max_abs=delta.abs().max().item(), rms=rms, reference_rms=reference,
                relative_rms=(rms / reference if reference else None))


def compare_logits(actual, expected):
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    result = difference(a, b)
    pa, pb = a.softmax(-1), b.softmax(-1)
    top = b.topk(2).values
    result.update(top1_equal=bool((a.argmax(-1) == b.argmax(-1)).all()),
                  reference_margin=(top[..., 0] - top[..., 1]).item(),
                  probability_tv=(pa - pb).abs().sum().item() / 2,
                  kl_reference_to_actual=max(0., (pb * (b.log_softmax(-1) - a.log_softmax(-1))).sum().item()))
    return result


def snapshot(cache):
    length = int(cache.get_seq_length())
    result = {}
    for i, layer in enumerate(cache.layers):
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            value = getattr(layer, name, None)
            items = value.items() if isinstance(value, dict) else [(0, value)]
            for index, tensor in items:
                if isinstance(tensor, torch.Tensor):
                    if name in ("keys", "values"):
                        tensor = tensor[..., :length, :]
                    result[f"{i}/{name}/{index}"] = tensor.detach().float().cpu().clone()
    if not result:
        raise AssertionError("No cache tensors found")
    return result


def compare_states(actual, expected):
    if actual.keys() != expected.keys():
        raise AssertionError("Cache fields differ")
    # Keep every layer's statistics, not tensors, in the final JSON.
    return {key: difference(value, expected[key]) for key, value in actual.items()}


def error_record(error):
    message = str(error).lower()
    missing_cpp = ("compiler: cl is not found" in message
                   or ("cl.exe" in message and ("not found" in message or "no such file" in message)))
    status = "oom" if isinstance(error, torch.OutOfMemoryError) else "unsupported" if missing_cpp else "error"
    result = dict(status=status, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    if missing_cpp:
        result["reason"] = "requires_unavailable_cpp_compiler"
    return result


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class Runner:
    """Explicitly route *every* forward, including prefill and label branches.

    The production scorer is reused with an instance-local _cached_forward.
    Compilation is requested directly; Transformers auto-compile restrictions
    do not silently exclude dynamic cache or prefill from this experiment.
    """
    def __init__(self, lm, cache_kind, execution, backend="inductor"):
        self.lm, self.model = lm, lm.load()
        self.cache_kind, self.execution = cache_kind, execution
        self.device = self.model.get_input_embeddings().weight.device
        self.capacity = 0
        self.lengths = weakref.WeakKeyDictionary()
        self.calls = {stage: dict(eager=0, compiled=0) for stage in ("prefill", "decode", "scoring")}
        self.measure_events = None
        self.stage_host = None
        self.compiled = None
        if execution != "eager":
            options = dict(fullgraph=True, dynamic=(cache_kind == "dynamic"), backend=backend)
            if backend == "inductor":
                options["mode"] = "reduce-overhead"
            self.compiled = torch.compile(self.model.forward, **options)

    def cache(self, capacity):
        self.capacity = capacity
        value = (DynamicCache(config=self.model.config) if self.cache_kind == "dynamic"
                 else StaticCache(config=self.model.config, max_cache_len=capacity))
        self.lengths[value] = 0
        return value

    def cache_length(self, cache):
        if cache not in self.lengths:
            # A branch clone is inspected once; normal decode uses a Python
            # cursor, avoiding an extra GPU->CPU sync for static cache each token.
            self.lengths[cache] = int(cache.get_seq_length())
        return self.lengths[cache]

    def forward(self, tokens, cache, stage):
        compiled = self.execution == "compile_all" or (self.execution == "compile_decode" and stage == "decode")
        self.calls[stage]["compiled" if compiled else "eager"] += 1
        length = self.cache_length(cache)
        # Fixed mask/storage for static; variable shapes intentionally tested
        # for dynamic. No per-token CPU extraction inside the compiled graph.
        mask_length = self.capacity if self.cache_kind == "static" else length + tokens.shape[1]
        positions = torch.arange(length, length + tokens.shape[1], device=self.device)[None]
        call = self.compiled if compiled else self.model
        event = None
        started = time.perf_counter()
        if self.measure_events is not None and self.device.type == "cuda":
            event = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            event[0].record()
        output = call(input_ids=tokens, past_key_values=cache,
            attention_mask=torch.ones((1, mask_length), device=self.device, dtype=torch.long),
            position_ids=positions, use_cache=True, logits_to_keep=1, return_dict=True)
        self.lengths[output.past_key_values] = length + tokens.shape[1]
        if event is not None:
            event[1].record()
            self.measure_events.append((stage, event))
        if self.stage_host is not None:
            self.stage_host[stage] = self.stage_host.get(stage, 0.) + time.perf_counter() - started
        # Clone immediately: compiled/CUDA graph output storage may be reused.
        return output.logits[:, -1].detach().clone(), output.past_key_values

    def score(self, inputs, prefix, count, cache):
        original = self.lm._cached_forward
        def cached_forward(model, tokens, cache=None):
            if cache is None:
                cache = self.cache(self.capacity)
            stage = "prefill" if self.cache_length(cache) == 0 else "scoring"
            logits, updated = self.forward(tokens, cache, stage)
            return logits[0].float(), updated
        self.lm._cached_forward = cached_forward
        try:
            return self.lm._score_options(self.model, inputs, prefix, count, cache)
        finally:
            self.lm._cached_forward = original


def question(count):
    return Question(id="benchmark", source_row=0,
        text="Which option follows from the context?", category="diagnostic", task=Task.choice,
        options=[Option(str(i), ("Ice floats on liquid water." if i == 0 else
                                 f"Alternative statement number {i}.")) for i in range(count)],
        target=Target(TargetKind.hard, [1.] + [0.] * (count - 1)))


def make_inputs(lm, case, thinking, count):
    contexts = ["Water becomes less dense when it freezes. Ice floats on liquid water.",
                "Compute 17 times 23 and compare it with 391.",
                "Water becomes less dense when it freezes. Ice floats on liquid water. " * 32]
    prompt = build_prompt(lm.tokenizer, contexts[case], question(count), thinking)
    ids = lm.tokenizer.encode(prompt, add_special_tokens=False)
    device = lm.load().get_input_embeddings().weight.device
    return prompt, torch.tensor([ids], device=device)


def suffix(lm, device):
    return torch.tensor([lm.tokenizer.encode("</think>" + ANSWER_SUFFIX, add_special_tokens=False)], device=device)


def next_token(logits, lm):
    # Fixed-work test: suppress EOS/closing boundary until the budget ends.
    # Natural stopping is exercised separately, never disguised as this test.
    scores = logits.clone()
    eos = lm.load().generation_config.eos_token_id
    excluded = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    closing = lm.tokenizer.encode("</think>", add_special_tokens=False)
    if len(closing) == 1:
        excluded += closing
    if excluded:
        scores[:, excluded] = -float("inf")
    return scores.argmax(-1)[:, None]


def paired_check(reference, candidate, inputs, prompt, config, teacher=None, capture=None):
    """Bounded memory: compare two live caches step by step; discard snapshots."""
    tail = suffix(reference.lm, inputs.device)
    capacity = inputs.shape[1] + config.validation_tokens + tail.shape[1] + 32
    ca, cb = reference.cache(capacity), candidate.cache(capacity)
    current = inputs
    history, steps = [], []
    for step in range(config.validation_tokens):
        stage = "prefill" if step == 0 else "decode"
        a, ca = reference.forward(current, ca, stage)
        sa = snapshot(ca)
        a_cpu = a.float().cpu().clone()
        b, cb = candidate.forward(current, cb, stage)
        sb = snapshot(cb)
        steps.append(dict(step=step, stage=stage, logits=compare_logits(b, a_cpu), cache=compare_states(sb, sa)))
        if capture is not None:
            capture(step, a_cpu, b.float().cpu().clone())
        current = (next_token(a, reference.lm) if teacher is None
                   else torch.tensor([[teacher[step]]], device=inputs.device))
        history.append(int(current.item()))
    answer_inputs = torch.cat((inputs, torch.tensor([history], device=inputs.device), tail), dim=1)
    answers = []
    for count in config.option_counts:
        try:
            last = count == config.option_counts[-1]
            ac = ca if last else copy.deepcopy(ca)
            bc = cb if last else copy.deepcopy(cb)
            a = reference.score(answer_inputs, "</think>" + ANSWER_SUFFIX, count, ac)
            sa = snapshot(ac)
            b = candidate.score(answer_inputs, "</think>" + ANSWER_SUFFIX, count, bc)
            answers.append(dict(status="complete", options=count, reference=a, candidate=b,
                logits=compare_logits(torch.tensor([b["scores"]]), torch.tensor([a["scores"]])),
                cache=compare_states(snapshot(bc), sa)))
        except Exception as error:
            answers.append(dict(options=count, **error_record(error)))
    return dict(status="complete" if all(a["status"] == "complete" for a in answers) else "partial",
                input_tokens=inputs.shape[1], teacher_tokens=history, steps=steps,
                answer_checks=answers)


def pipeline(runner, config, case, count, mode, forced=None):
    """Complete prompt -> optional reasoning -> original label scorer.

    thinking_fixed closes reasoning explicitly after fixed work. thinking_natural
    stops only at the actual closing boundary and reports reasoning_limit when
    no boundary appears; no score is fabricated for that case.
    """
    thinking = mode != "direct"
    prompt, inputs = make_inputs(runner.lm, case, thinking, count)
    tail = suffix(runner.lm, inputs.device)
    label_prefix = "</think>" + ANSWER_SUFFIX if thinking else prompt
    label_reserve = max(map(len, candidate_sequences(runner.lm.tokenizer, label_prefix, count)))
    capacity = inputs.shape[1] + (config.timing_tokens if thinking else 0) + tail.shape[1] + label_reserve
    cache = runner.cache(capacity)
    history = []
    if thinking:
        current = inputs
        closing = runner.lm.tokenizer.encode("</think>", add_special_tokens=False)
        for step in range(config.timing_tokens):
            logits, cache = runner.forward(current, cache, "prefill" if step == 0 else "decode")
            current = (torch.tensor([[forced[step]]], device=inputs.device) if forced is not None
                       else next_token(logits, runner.lm) if mode == "thinking_fixed"
                       else logits.argmax(-1)[:, None])
            history.append(int(current.item()))
            if mode == "thinking_natural" and history[-len(closing):] == closing:
                break
            eos = runner.model.generation_config.eos_token_id
            eos = [] if eos is None else ([eos] if isinstance(eos, int) else eos)
            if mode == "thinking_natural" and history[-1] in eos:
                return dict(status="eos_without_reasoning_boundary", reasoning_tokens=len(history),
                            tokens=history, input_tokens=inputs.shape[1])
        if mode == "thinking_natural":
            if history[-len(closing):] != closing:
                return dict(status="reasoning_limit", reasoning_tokens=len(history), tokens=history, input_tokens=inputs.shape[1])
            # Closing tokens already in history. Append only native separator.
            end = torch.tensor([runner.lm.tokenizer.encode(ANSWER_SUFFIX, add_special_tokens=False)], device=inputs.device)
        else:
            end = tail
        answer_inputs = torch.cat((inputs, torch.tensor([history], device=inputs.device), end), dim=1)
        prefix = "</think>" + ANSWER_SUFFIX
    else:
        answer_inputs, prefix = inputs, prompt
    sequences = candidate_sequences(runner.lm.tokenizer, prefix, count, answer_inputs[0].tolist())
    result = runner.score(answer_inputs, prefix, count, cache)
    if not all(math.isfinite(v) for v in result["scores"]):
        raise AssertionError("Nonfinite answer scores")
    return dict(status="complete", **result, tokens=history, reasoning_tokens=len(history),
                input_tokens=inputs.shape[1], max_label_tokens=max(map(len, sequences)),
                forced_reasoning_boundary=(mode == "thinking_fixed"))


def native_predict(runner, config, count, thinking):
    """Exercise unmodified NativeLM.predict + HF generate with routed forward.

    Hooks exist only in this benchmark process and are restored in finally.
    Static generation reserves room for the real answer suffix/label branches.
    """
    lm, model = runner.lm, runner.model
    production = ((runner.cache_kind == "static" and runner.execution == "compile_decode")
                  or (runner.cache_kind == "dynamic" and runner.execution == "eager"))
    if production:
        # Validated paths call the same execution component as evaluation.
        # Experimental cache/compile-all paths below remain diagnostic only.
        original_inference = lm.inference
        try:
            mode = (ExecutionMode.compiled_decode if thinking and runner.execution == "compile_decode"
                    else ExecutionMode.eager)
            lm.inference = NativeLMConfig(mode="thinking" if thinking else "direct",
                max_input_tokens=8192, max_reasoning_tokens=config.timing_tokens, execution=mode)
            result = lm.predict("Water becomes less dense when it freezes. Ice floats on liquid water.", question(count))
            result["execution"] = lm.describe_execution()
            result["execution_implementation"] = "src.native_lm.execution"
            if lm._execution is not None and mode == ExecutionMode.compiled_decode:
                result["static_cache_capacity"] = lm._execution.capacity
            return result
        finally:
            lm.inference = original_inference
    original_forward, original_generate = model.forward, model.generate
    original_score, original_inference = lm._score_options, lm.inference
    original_generation = copy.deepcopy(model.generation_config)
    phase = "decode"

    @functools.wraps(original_forward)
    def forward(*args, **kwargs):
        tokens = kwargs.get("input_ids")
        if tokens is None and args:
            tokens = args[0]
        cache = kwargs.get("past_key_values")
        if cache is None:
            cache = runner.cache(tokens.shape[1] + config.timing_tokens + 64)
            kwargs["past_key_values"] = cache
        stage = "prefill" if int(cache.get_seq_length()) == 0 else phase
        compiled = runner.execution == "compile_all" or (runner.execution == "compile_decode" and stage == "decode")
        runner.calls[stage]["compiled" if compiled else "eager"] += 1
        call = runner.compiled if compiled else original_forward
        return call(*args, **kwargs)

    @functools.wraps(original_generate)
    def generate(*args, **kwargs):
        tokens = kwargs.get("input_ids")
        if tokens is None:
            tokens = args[0]
        kwargs["past_key_values"] = runner.cache(tokens.shape[1] + config.timing_tokens + 64)
        return original_generate(*args, **kwargs)

    def score(*args, **kwargs):
        nonlocal phase
        phase = "scoring"
        try:
            return original_score(*args, **kwargs)
        finally:
            phase = "decode"

    try:
        model.forward, model.generate = forward, generate
        # Compilation is explicitly routed above, not silently delegated to HF.
        model.generation_config.disable_compile = True
        model.generation_config.compile_config = None
        model.generation_config.cache_implementation = None
        lm._score_options = score
        lm.inference = NativeLMConfig(mode="thinking" if thinking else "direct",
            max_input_tokens=8192, max_reasoning_tokens=config.timing_tokens)
        value = lm.predict("Water becomes less dense when it freezes. Ice floats on liquid water.", question(count))
        if value.get("scores") is not None and not all(math.isfinite(v) for v in value["scores"]):
            raise AssertionError("Nonfinite NativeLM.predict scores")
        return value
    finally:
        model.forward, model.generate = original_forward, original_generate
        model.generation_config = original_generation
        lm._score_options, lm.inference = original_score, original_inference


def measured(action, device, runner=None):
    sync(device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if runner is not None:
        runner.measure_events, runner.stage_host = [], {}
    started = time.perf_counter()
    value = action()
    sync(device)
    elapsed = time.perf_counter() - started
    stages = {}
    if runner is not None:
        for stage, events in runner.measure_events:
            stages[stage] = stages.get(stage, 0.) + events[0].elapsed_time(events[1])
        host = dict(runner.stage_host)
        runner.measure_events, runner.stage_host = None, None
    else:
        host = {}
    return dict(seconds=elapsed, forward_stage_gpu_timeline_ms=stages,
                forward_stage_host_seconds=host,
                peak_allocated_gib=(torch.cuda.max_memory_allocated(device) / 1024**3
                                    if torch.device(device).type == "cuda" else None),
                result=value)


def bridge_check(model):
    from fla.modules.causal_conv1d import causal_conv1d_update
    from src.native_lm.kernels import _compiled_convolution_update
    conv = next(m for name, m in model.named_modules() if name.endswith("linear_attn.conv1d"))
    weight, bias = conv.weight.squeeze(1).contiguous(), conv.bias
    g = torch.Generator(device=weight.device).manual_seed(19)
    initial = torch.randn((1, weight.shape[0], weight.shape[1]), device=weight.device, dtype=weight.dtype, generator=g)
    a, b = initial.clone(), initial.clone()
    call = torch.compile(_compiled_convolution_update, fullgraph=True, dynamic=False, mode="reduce-overhead")
    x = torch.randn((1, 1, weight.shape[0]), device=weight.device, dtype=weight.dtype, generator=g)
    call(x, initial.clone(), weight, bias, "silu")
    steps = []
    for step in range(16):
        x = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)
        expected, _ = causal_conv1d_update(x, cache=a, weight=weight, bias=bias, activation="silu")
        expected = expected.detach().clone()
        actual = call(x, b, weight, bias, "silu").detach().clone()
        steps.append(dict(step=step, output=difference(actual, expected), cache=difference(b, a)))
    checks = {"update": steps}
    if "conv" in _BENCH_IMPLEMENTATIONS:
        sequence = torch.randn((1, weight.shape[0], 17), device=weight.device, dtype=weight.dtype, generator=g)
        expected = _BENCH_IMPLEMENTATIONS["conv"](sequence, weight, bias, activation="silu").clone()
        compiled_conv = torch.compile(_conv_prefill, fullgraph=True, dynamic=False, mode="reduce-overhead")
        actual = compiled_conv(sequence, weight, bias, "silu").clone()
        checks["prefill"] = difference(actual, expected)
    if "chunk" in _BENCH_IMPLEMENTATIONS:
        layer = next(m for name, m in model.named_modules() if name.endswith("linear_attn"))
        heads, key_dim, value_dim = layer.num_v_heads, layer.head_k_dim, layer.head_v_dim
        q = torch.randn((1, 17, heads, key_dim), device=weight.device, dtype=weight.dtype, generator=g)
        k = torch.randn(q.shape, device=q.device, dtype=q.dtype, generator=g)
        v = torch.randn((1, 17, heads, value_dim), device=q.device, dtype=q.dtype, generator=g)
        decay = -torch.rand((1, 17, heads), device=q.device, generator=g)
        beta = torch.rand((1, 17, heads), device=q.device, dtype=q.dtype, generator=g)
        initial_state = torch.zeros((1, heads, key_dim, value_dim), device=q.device, dtype=torch.float32)
        before = initial_state.clone()
        expected, expected_state = _BENCH_IMPLEMENTATIONS["chunk"](q, k, v, g=decay, beta=beta,
            initial_state=initial_state, output_final_state=True, use_qk_l2norm_in_kernel=True)
        expected, expected_state = expected.clone(), expected_state.clone()
        compiled_chunk = torch.compile(_delta_chunk, fullgraph=True, dynamic=False, mode="reduce-overhead")
        actual, actual_state = compiled_chunk(q, k, v, decay, beta, initial_state, True, 64, None)
        checks["chunk"] = dict(output=difference(actual, expected), state=difference(actual_state, expected_state),
                               input_state=difference(initial_state, before))
    exact = all(s["output"]["max_abs"] == 0 and s["cache"]["max_abs"] == 0 for s in steps)
    exact &= checks.get("prefill", {"max_abs": 0})["max_abs"] == 0
    exact &= all(value["max_abs"] == 0 for value in checks.get("chunk", {}).values())
    return dict(status="complete", exact=exact, checks=checks)


def summary(check):
    steps = check.get("steps", [])
    if not steps:
        return {}
    relative = [s["logits"]["relative_rms"] for s in steps if s["logits"]["relative_rms"] is not None]
    state = [v["relative_rms"] for s in steps for v in s["cache"].values() if v["relative_rms"] is not None]
    return dict(top1_changes=sum(not s["logits"]["top1_equal"] for s in steps),
                logits_relative_rms_mean=statistics.mean(relative), logits_relative_rms_max=max(relative),
                cache_relative_rms_max=max(state), probability_tv_max=max(s["logits"]["probability_tv"] for s in steps),
                answers=[dict(options=a["options"], status=a["status"],
                    probability_tv=a.get("logits", {}).get("probability_tv"),
                    top1_equal=a.get("logits", {}).get("top1_equal"))
                    for a in check.get("answer_checks", [])])


def cleanup():
    torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def worker(spec):
    config = Config(**spec["config"])
    config.cache_dir = Path(config.cache_dir)
    path = Path(spec["result"])
    result = dict(weight=spec["weight"], status="running", rows=[])
    def save():
        write_json(path, result)
    save()
    try:
        weight = spec["weight"]
        torch._dynamo.config.recompile_limit = config.recompile_limit
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if weight == "fp32_reference":
            # Capture torch implementations before NativeLM can install FLA
            # convolution bridges. This worker is isolated from native kernels.
            from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
            for name in ("causal_conv1d_fn", "causal_conv1d_update",
                         "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
                setattr(qwen, name, inspect.unwrap(getattr(qwen, name)))
        lm = NativeLM(BackboneConfig(config.model_id, revision=config.revision,
            device=config.device, cache_dir=config.cache_dir,
            dtype=("float32" if weight in ("fp32", "fp32_reference", "nf4_fp32")
                   else "float16" if weight in ("fp16", "nf4_fp16") else "bfloat16"),
            quantization="nf4" if weight.startswith("nf4") else "none"), NativeLMConfig())
        started = time.perf_counter()
        model = lm.load()
        sync(config.device)
        result["loading_seconds"] = time.perf_counter() - started
        result["model"] = lm.describe()
        result["benchmark_boundaries"] = (install_benchmark_boundaries(model)
            if weight != "fp32_reference" else [])
        module = importlib.import_module(type(model).__module__)
        providers = {}
        for name in ("causal_conv1d_fn", "causal_conv1d_update",
                     "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            function = getattr(module, name, None)
            provider = getattr(function, "kernel_provider", None)
            if provider is None and callable(function):
                try:
                    bindings = inspect.getclosurevars(function).nonlocals
                    function = bindings.get("implementation", inspect.unwrap(function))
                except TypeError:
                    pass
                provider = f"{getattr(function, '__module__', '?')}.{getattr(function, '__qualname__', '?')}"
            providers[name] = provider
        result["runtime"] = dict(kernels=providers, embedding_dtype=str(model.get_input_embeddings().weight.dtype),
            weights_gib=model.get_memory_footprint() / 1024**3,
            linear4bit_modules=sum(type(m).__name__ == "Linear4bit" for m in model.modules()),
            quantized_compute_dtypes=sorted({str(m.compute_dtype) for m in model.modules() if type(m).__name__ == "Linear4bit"}),
            matmul_precision=torch.get_float32_matmul_precision(),
            pytorch_tf32=torch.backends.cuda.matmul.allow_tf32,
            fp32_caveat="PyTorch TF32 disabled; third-party Triton kernels may select their own internal precision.")
        if config.native_smoke:
            runner = Runner(lm, config.caches[0], config.executions[0])
            print(f"[native smoke] ONE real thinking call, budget={config.timing_tokens}; compilation may occur on first use", flush=True)
            value = measured(lambda: native_predict(runner, config, 4, True), config.device)
            prediction = value["result"]
            scores = prediction.get("scores")
            passed = (prediction.get("status") == "ok" and prediction.get("reasoning_tokens", 0) > 0
                      and scores is not None and len(scores) == 4 and all(math.isfinite(v) for v in scores))
            result.update(status="complete" if passed else "incomplete", native_smoke=value,
                          execution=value["result"].get("execution"), passed=passed)
            print(f"[native smoke] {'PASS' if passed else 'NOT PASSED'}: status={prediction.get('status')}, reasoning_tokens={prediction.get('reasoning_tokens')}, scores={scores}, seconds={value['seconds']:.2f}", flush=True)
            return
        try:
            result["bridge"] = (dict(status="not_applicable", reason="torch reference kernels do not use the FLA custom-op bridge")
                                if weight == "fp32_reference" else bridge_check(model))
        except Exception as error:
            result["bridge"] = error_record(error)
        save()
        shared_path = Path(spec["shared"])
        shared = torch.load(shared_path, map_location="cpu", weights_only=True) if shared_path.exists() else {}
        reference = Runner(lm, "dynamic", "eager")
        own_baselines = {}
        fp32_path = path.parent / "fp32_reference_logits.pt"
        fp32_baseline = (torch.load(fp32_path, map_location="cpu", weights_only=True)
                         if weight == "fp32_reference" and fp32_path.exists() else {})
        for cache_kind in config.caches:
            for execution in config.executions:
                label = f"{weight}/{cache_kind}/{execution}"
                row = dict(cache=cache_kind, execution=execution, status="running", checks=[], workloads=[])
                result["rows"].append(row)
                save()
                print(f"\n[{label}] start", flush=True)
                candidate = None
                try:
                    cleanup()
                    candidate = Runner(lm, cache_kind, execution)
                    for case in range(3):
                        try:
                            prompt, inputs = make_inputs(lm, case, True, 4)
                            key = str(case)
                            existing = shared.get(key)
                            if existing is not None and existing["inputs"] != inputs.cpu().tolist():
                                raise AssertionError("Prompt/tokenizer mismatch across weight loads")
                            captured = []
                            cross = []
                            def capture(step, a, b):
                                if key not in own_baselines:
                                    captured.append(a)
                                if existing is not None:
                                    cross.append(dict(step=step,
                                        dynamic=compare_logits(a, existing["logits"][step]),
                                        candidate=compare_logits(b, existing["logits"][step])))
                            print(f"[{label}] prefill/decode/cache/scorer case {case+1}/3", flush=True)
                            value = paired_check(reference, candidate, inputs, prompt, config,
                                teacher=existing["tokens"] if existing else None, capture=capture)
                            value["summary"] = summary(value)
                            value["cross_weight_reference"] = dict(weight=shared.get("weight", weight), steps=cross)
                            row["checks"].append(value)
                            if key not in own_baselines:
                                own_baselines[key] = dict(inputs=inputs.cpu().tolist(), tokens=value["teacher_tokens"], logits=captured)
                                if weight == "fp32":
                                    torch.save(own_baselines, fp32_path)
                            if fp32_baseline.get(key) is not None:
                                baseline_logits = fp32_baseline[key]["logits"]
                                value["native_fp32_vs_torch_reference"] = [compare_logits(a, b) for a, b in zip(own_baselines[key]["logits"], baseline_logits)]
                            if existing is None:
                                shared[key] = dict(inputs=inputs.cpu().tolist(), tokens=value["teacher_tokens"], logits=captured)
                                shared.setdefault("weight", weight)
                                torch.save(shared, shared_path)
                            save()
                        except Exception as error:
                            row["checks"].append(dict(case=case, **error_record(error)))
                            print(f"[{label}] validation case {case+1} FAILED: {type(error).__name__}: {brief(error)} (full error in JSON)", flush=True)
                            cleanup()
                            save()
                    # Real direct and reasoning pipelines, including multi-token
                    # labels and branch-cache cloning. Each workload independent.
                    for case in (0, 2):
                        for count in config.option_counts:
                            for mode in ("direct", "thinking_fixed", "thinking_natural"):
                                # Long decode need not be repeated for every
                                # label-count/context combination: teacher-forced
                                # checks cover all contexts; scoring covers both
                                # label types. Measure full reasoning on case 0.
                                if mode != "direct" and (case != 0 or count != config.option_counts[0]):
                                    continue
                                work = dict(case=case, options=count, mode=mode, status="running")
                                row["workloads"].append(work)
                                try:
                                    # Compilation/warmup is excluded from steady samples.
                                    warmup = measured(lambda: pipeline(candidate, config, case, count, mode), config.device, runner=candidate)
                                    work["warmup_seconds"] = warmup["seconds"]
                                    if mode == "thinking_natural":
                                        samples = [warmup]  # functional stopping check, not steady throughput
                                    else:
                                        pipeline(candidate, config, case, count, mode)
                                        samples = [measured(lambda: pipeline(candidate, config, case, count, mode), config.device, runner=candidate)
                                                   for _ in range(config.repeats)]
                                    work.update(status="complete", includes_compilation=(mode == "thinking_natural"), samples=samples,
                                        median_seconds=statistics.median(s["seconds"] for s in samples))
                                    if mode != "direct":
                                        work["median_tokens_per_second"] = statistics.median(
                                            s["result"]["reasoning_tokens"] / s["seconds"] for s in samples)
                                    # Exact end-to-end same-history label-scoring
                                    # comparison separately from free-generation drift.
                                    chosen = samples[0]["result"]
                                    forced = chosen.get("tokens") if mode == "thinking_fixed" else None
                                    expected = pipeline(reference, config, case, count, mode, forced=forced)
                                    work["reference"] = expected
                                    work["generation_identical"] = chosen.get("tokens") == expected.get("tokens")
                                    if chosen.get("scores") is not None and expected.get("scores") is not None:
                                        work["answer_difference"] = compare_logits(
                                            torch.tensor([chosen["scores"]]), torch.tensor([expected["scores"]]))
                                    print(f"[{label}] {mode}, context {case}, {count} labels: {work['median_seconds']:.3f}s", flush=True)
                                except Exception as error:
                                    work.update(error_record(error))
                                    print(f"[{label}] workload FAILED: {brief(work['error'])} (full error in JSON)", flush=True)
                                    cleanup()
                                save()
                    row["native_eval"] = []
                    for count in config.option_counts:
                        for thinking in (False, True):
                            integration = dict(mode="thinking" if thinking else "direct", options=count)
                            try:
                                # Functional integration latency can include new
                                # compilation shapes; not a steady speed sample.
                                value = measured(lambda: native_predict(candidate, config, count, thinking), config.device)
                                integration.update(status="complete", first_call=value)
                            except Exception as error:
                                integration.update(error_record(error))
                                cleanup()
                            row["native_eval"].append(integration)
                            save()
                    if config.profile:
                        try:
                            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                  torch.profiler.ProfilerActivity.CUDA]) as profiler:
                                pipeline(candidate, config, 0, 4, "direct")
                                pipeline(candidate, config, 0, 4, "thinking_fixed")
                                pipeline(candidate, config, 0, 54, "direct")
                            work = profiler.key_averages()
                            row["profile"] = dict(cpu=work.table(sort_by="self_cpu_time_total", row_limit=30),
                                                  cuda=work.table(sort_by="self_device_time_total", row_limit=30))
                        except Exception as error:
                            row["profile"] = error_record(error)
                    row["status"] = "complete" if (
                        all(w["status"] == "complete" for w in row["workloads"])
                        and all(c["status"] == "complete" for c in row["checks"])
                        and all(c["status"] == "complete" for c in row["native_eval"])) else "partial"
                except Exception as error:
                    row.update(error_record(error))
                    print(f"[{label}] FAILED: {brief(row['error'])} (full error in JSON)", flush=True)
                finally:
                    if candidate is not None:
                        row["forward_attempts"] = candidate.calls
                    del candidate
                    cleanup()
                    save()
        result["status"] = "complete" if all(r["status"] == "complete" for r in result["rows"]) else "partial"
    except Exception as error:
        result.update(error_record(error))
        print(f"[{spec['weight']}] load/worker FAILED: {brief(result['error'])} (full error in JSON)", flush=True)
    finally:
        save()


def report_summary(report):
    lines = ["Per-row evidence; no automatic BF16/NF4 approval from a numerical tolerance."]
    for weight in report.get("results", []):
        lines.append(f"{weight['weight']}: {weight['status']}")
        if "native_smoke" in weight:
            prediction = weight["native_smoke"]["result"]
            lines.append(f"NATIVE SMOKE {'PASS' if weight.get('passed') else 'NOT PASSED'}: status={prediction.get('status')}, reasoning_tokens={prediction.get('reasoning_tokens')}, scores={prediction.get('scores')}")
            continue
        if weight.get("error"):
            lines.append(brief(weight["error"]))
        lines.append(f"  isolated FLA custom-op exact: {weight.get('bridge', {}).get('exact', 'not verified')}")
        for row in weight.get("rows", []):
            label = f"{weight['weight']}/{row['cache']}/{row['execution']}"
            lines.append(f"{label}: {row['status']}")
            if row.get("error"):
                lines.append("  " + brief(row["error"]))
            for i, check in enumerate(row.get("checks", [])):
                if check.get("status") != "complete":
                    lines.append(f"  validation case {i+1}: {check.get('status')}: {brief(check.get('error', 'unknown error'))}")
                    continue
                s = check["summary"]
                lines.append(f"  case {i+1}: logits RMS max {100*s['logits_relative_rms_max']:.6f}%, cache RMS max {100*s['cache_relative_rms_max']:.6f}%, TV max {100*s['probability_tv_max']:.6f}%, changed tokens {s['top1_changes']}")
                for answer in s["answers"]:
                    lines.append(f"    {answer['options']} labels: {answer['status']}, TV={answer['probability_tv']}, top1_equal={answer['top1_equal']}")
            for integration in row.get("native_eval", []):
                result = integration.get("first_call", {}).get("result", {})
                lines.append(f"  NativeLM.predict {integration['mode']}/{integration['options']} labels: {integration['status']}, inference_status={result.get('status')}, seconds={integration.get('first_call', {}).get('seconds')}")
            for work in row.get("workloads", []):
                lines.append(f"  {work['mode']}/case{work['case']}/{work['options']} labels: {work['status']}, seconds={work.get('median_seconds')}, tok/s={work.get('median_tokens_per_second')}, generation_identical={work.get('generation_identical')}")
    return lines


def validate_config(config):
    if not config.weights or set(config.weights) - {"bf16", "nf4", "fp16", "nf4_fp16", "nf4_fp32", "fp32", "fp32_reference"}:
        raise ValueError("weights: bf16, fp16, fp32, nf4, nf4_fp16, nf4_fp32, fp32_reference")
    if not config.caches or set(config.caches) - {"dynamic", "static"}:
        raise ValueError("caches: dynamic, static")
    if not config.executions or set(config.executions) - {"eager", "compile_decode", "compile_all"}:
        raise ValueError("executions: eager, compile_decode, compile_all")
    if config.native_smoke and config.smoke_tokens < 1:
        raise ValueError("Positive smoke_tokens required")
    if config.recompile_limit < 1:
        raise ValueError("Positive recompile_limit required")
    if min(config.validation_tokens, config.timing_tokens, config.repeats) < 1 or not config.option_counts or min(config.option_counts) < 2:
        raise ValueError("Positive token budgets/repeats and at least two options required")
    for values in (config.weights, config.caches, config.executions, config.option_counts):
        if len(set(values)) != len(values):
            raise ValueError("Duplicate matrix entries")


def prepare_config(config):
    """Safe default: one selected real inference call; matrix is explicit."""
    if config.native_smoke:
        if tuple(config.weights) == ("bf16", "nf4", "fp16", "nf4_fp16", "nf4_fp32", "fp32", "fp32_reference"):
            config.weights = ("nf4_fp16",)
        if tuple(config.caches) == ("dynamic", "static"):
            config.caches = ("static",)
        if tuple(config.executions) == ("eager", "compile_decode", "compile_all"):
            config.executions = ("compile_decode",)
        if any(len(v) != 1 for v in (config.weights, config.caches, config.executions)):
            raise ValueError("Smoke accepts one weight/cache/execution; use --full for a matrix")
        config.timing_tokens, config.option_counts = config.smoke_tokens, (4,)
    if config.output is None:
        config.output = Path("artifacts/diagnostics") / ("native_benchmark.json" if config.full else "native_smoke.json")
    return config


def run(config):
    config = prepare_config(config)
    validate_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("Full benchmark requires CUDA; --self-test works offline on CPU")
    report = dict(schema_version=2, config=asdict(config), status="running", results=[],
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(config.device),
        gpu_total_gib=torch.cuda.get_device_properties(config.device).total_memory / 1024**3,
        platform=platform.platform(), cpu=platform.processor(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), packages={},
        matrix_scope="none/NF4 x BF16/FP16/FP32 plus torch-reference FP32; not every third-party quantization format",
        numerical_scope="All rows compared against same-weight eager dynamic; cross-weight rows share teacher histories. FP32 does not imply IEEE-only FLA internals.")
    for name in ("transformers", "flash-linear-attention", "triton-windows", "triton", "bitsandbytes"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    write_json(config.output, report)
    resolved_revision = config.revision
    with tempfile.TemporaryDirectory(prefix="native-benchmark-") as temporary:
        root = Path(temporary)
        for weight in config.weights:
            spec = dict(config=asdict(config), weight=weight, shared=str(root / "baseline.pt"),
                        result=str(root / f"{weight}.json"))
            spec["config"]["revision"] = resolved_revision
            path = root / "worker.json"
            write_json(path, spec)
            process = subprocess.Popen([sys.executable, "-m", "src.native_lm.benchmark", "--worker-spec", str(path)])
            try:
                last_snapshot = None
                while process.poll() is None:
                    # Mirror partial worker results into the ONE retained report.
                    try:
                        worker_path = Path(spec["result"])
                        stat = worker_path.stat()
                        signature = (stat.st_mtime_ns, stat.st_size)
                        if signature != last_snapshot:
                            current = json.loads(worker_path.read_text(encoding="utf-8"))
                            report["current_worker"] = current
                            write_json(config.output, report)
                            last_snapshot = signature
                    except FileNotFoundError:
                        pass
                    time.sleep(1)
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                report["status"] = "interrupted"
                write_json(config.output, report)
                raise
            result_path = Path(spec["result"])
            value = (json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists()
                     else dict(weight=weight, status="crashed", error=f"Worker exited {process.returncode}"))
            if process.returncode != 0:
                value["worker_exit_code"] = process.returncode
                if value.get("status") == "running":
                    value["status"] = "crashed"
            loaded_revision = value.get("model", {}).get("revision")
            if resolved_revision is None and loaded_revision is not None:
                resolved_revision = loaded_revision
                report["resolved_revision"] = loaded_revision
            elif loaded_revision is not None and loaded_revision != resolved_revision:
                value["status"] = "error"
                value["error"] = "Checkpoint revision changed across workers"
            report["results"].append(value)
            report.pop("current_worker", None)
            write_json(config.output, report)
    report["status"] = "complete" if all(r["status"] == "complete" for r in report["results"]) else "partial"
    report["summary"] = report_summary(report)
    write_json(config.output, report)
    print("\n" + "\n".join(report["summary"]), flush=True)
    print(f"\nONE report: {config.output.resolve()}", flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


@torch.inference_mode()
def _self_test():
    """All benchmark-specific regression checks live here, not in other files."""
    from transformers import LlamaConfig, LlamaForCausalLM
    class Tokenizer:
        eos_token_id = 0
        pad_token_id = 0
        def encode(self, text, add_special_tokens=False):
            text = text.replace("</think>", "\x01").replace("<think>", "\x02")
            return [ord(c) for c in text]
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"] + ("<think>\n" if kwargs["enable_thinking"] else "</think>" + ANSWER_SUFFIX)
    source_class = NativeLM
    def tiny():
        torch.manual_seed(7)
        model = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8)).eval()
        lm = source_class(BackboneConfig("in-memory", device="cpu", dtype="float32", quantization="none"), NativeLMConfig())
        model.generation_config.eos_token_id = 1
        model.generation_config.pad_token_id = 0
        model.generation_config.forced_eos_token_id = 1  # force a real closing token at the budget boundary
        lm._model, lm._tokenizer = model, Tokenizer()
        return lm
    config = Config(device="cpu", validation_tokens=3, timing_tokens=3, repeats=1)
    validate_config(config)
    smoke = prepare_config(Config())
    assert smoke.native_smoke and smoke.weights == ("nf4_fp16",) and smoke.caches == ("static",)
    assert smoke.executions == ("compile_decode",) and smoke.timing_tokens == 2048
    assert smoke.output.name == "native_smoke.json"
    full = prepare_config(Config(full=True))
    assert len(full.weights) == 7 and len(full.caches) == 2 and len(full.executions) == 3
    assert full.output.name == "native_benchmark.json"
    selected = prepare_config(Config(weights=("bf16",), output=Path("custom.json")))
    assert selected.weights == ("bf16",) and selected.output == Path("custom.json")
    torch._dynamo.config.recompile_limit = config.recompile_limit
    lm = tiny()
    reference = Runner(lm, "dynamic", "eager")
    for cache_kind in ("dynamic", "static"):
        for execution in ("eager", "compile_decode", "compile_all"):
            candidate = Runner(lm, cache_kind, execution, backend="eager")
            prompt, inputs = make_inputs(lm, 0, True, 4)
            check = paired_check(reference, candidate, inputs, prompt, config, teacher=[6, 7, 8])
            assert check["teacher_tokens"] == [6, 7, 8]
            assert check["status"] == "complete" and summary(check)["top1_changes"] == 0
            assert len(check["answer_checks"]) == 2
            for mode in ("direct", "thinking_fixed", "thinking_natural"):
                output = pipeline(candidate, config, 0, 4, mode)
                assert output["status"] in ("complete", "reasoning_limit")
            # Explicitly exercise multi-token labels and branch cache cloning.
            output = pipeline(candidate, config, 0, 54, "direct")
            assert len(output["scores"]) == 54 and output["max_label_tokens"] == 2
            integrated = native_predict(candidate, config, 4, thinking=False)
            assert integrated["status"] == "ok" and len(integrated["scores"]) == 4
            integrated = native_predict(candidate, config, 4, thinking=True)
            assert integrated["status"] == "ok" and len(integrated["scores"]) == 4
            assert integrated["reasoning_tokens"] > 0
            if execution == "compile_all":
                assert candidate.calls["prefill"]["compiled"] and candidate.calls["scoring"]["compiled"]
            cleanup()
    a = torch.tensor([[1., 3., 2.]])
    assert compare_logits(a, a)["probability_tv"] == 0
    assert compare_states({"state": torch.ones(3)}, {"state": torch.zeros(3)})["state"]["max_abs"] == 1
    for bad in (torch.tensor([float("nan")]), torch.zeros(2)):
        try:
            difference(bad, torch.zeros(1))
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid state not detected")
    def broken(*args, **kwargs):
        raise RuntimeError("deliberate compiler failure")
    candidate = Runner(lm, "static", "compile_all", backend="eager")
    candidate.compiled = broken
    try:
        pipeline(candidate, config, 0, 4, "direct")
    except RuntimeError as error:
        assert "deliberate" in str(error)
    else:
        raise AssertionError("Compiler failure silently ignored")
    # Exercise the actual hybrid Qwen cache schema on CPU torch references.
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    names = ("causal_conv1d_fn", "causal_conv1d_update",
             "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")
    saved = {name: getattr(qwen, name) for name in names}
    try:
        for name, function in saved.items():
            setattr(qwen, name, inspect.unwrap(function))
        hybrid = Qwen3_5ForCausalLM(Qwen3_5TextConfig(
            vocab_size=256, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, layer_types=["linear_attention", "full_attention"],
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=2, linear_num_value_heads=4,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                "partial_rotary_factor": 1., "mrope_section": [2, 3, 3]})).eval()
        lm._model = hybrid
        eager, static = Runner(lm, "dynamic", "eager"), Runner(lm, "static", "eager")
        prompt, inputs = make_inputs(lm, 0, True, 4)
        checked = paired_check(eager, static, inputs, prompt, config, teacher=[6, 7, 8])
        assert checked["status"] == "complete"
        fields = checked["steps"][0]["cache"]
        assert all(any(name in key for key in fields) for name in ("keys", "values", "conv_states", "recurrent_states"))
        assert max(v["max_abs"] for v in fields.values()) < 1e-5
        # Reproduce the reported failure without CUDA: deliberately mark the
        # reference convolution AND chunk functions as compiler-disabled.
        before_impl = dict(_BENCH_IMPLEMENTATIONS)
        qwen.causal_conv1d_fn = torch.compiler.disable(qwen.causal_conv1d_fn)
        qwen.torch_chunk_gated_delta_rule = torch.compiler.disable(qwen.torch_chunk_gated_delta_rule)
        try:
            install_benchmark_boundaries(module=qwen)
            compiled = Runner(lm, "static", "compile_all", backend="eager")
            checked = paired_check(eager, compiled, inputs, prompt, config, teacher=[6, 7, 8])
            assert checked["status"] == "complete"
            assert summary(checked)["top1_changes"] == 0
            assert max(s["logits"]["max_abs"] for s in checked["steps"]) < 1e-5
        finally:
            _BENCH_IMPLEMENTATIONS.clear()
            _BENCH_IMPLEMENTATIONS.update(before_impl)
    finally:
        for name, function in saved.items():
            setattr(qwen, name, function)
    unsupported = error_record(RuntimeError("Compiler: cl is not found."))
    assert unsupported["status"] == "unsupported"
    assert unsupported["reason"] == "requires_unavailable_cpp_compiler"
    assert error_record(RuntimeError("unrelated failure"))["status"] == "error"
    # Fault injection validates the actual worker loop: failed validation must
    # not hide direct/scoring or later cases; only one report is emitted.
    from unittest.mock import patch
    original_runner = Runner
    def broken_runner(lm, cache_kind, execution, **kwargs):
        value = original_runner(lm, cache_kind, execution, **kwargs)
        if cache_kind == "static":
            value.forward = broken
        return value
    def fake_source(*args, **kwargs):
        value = tiny()
        value.describe = lambda: {"model_id": "offline-test", "revision": None}
        return value
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        small = Config(full=True, device="cpu", weights=("bf16",), caches=("static",),
                       executions=("eager",), validation_tokens=2, timing_tokens=2,
                       repeats=1, option_counts=(4,))
        spec = dict(config=asdict(small), weight="bf16", shared=str(root / "baseline.pt"),
                    result=str(root / "report.json"))
        with patch.object(sys.modules[__name__], "NativeLM", fake_source), \
             patch.object(sys.modules[__name__], "Runner", broken_runner):
            worker(spec)
        partial = json.loads((root / "report.json").read_text())
        assert partial["status"] == "partial"
        assert len(partial["rows"][0]["checks"]) == 3
        assert partial["rows"][0]["workloads"]
        assert all(w["status"] == "error" for w in partial["rows"][0]["workloads"])
        assert any("deliberate" in line for line in report_summary({"results": [partial]}))
    print("SELF-TEST PASS: 6 cache/execution combinations, all pipeline stages, teacher forcing, direct/thinking, multi-token branches, hybrid Qwen cache, corruption detection and worker error isolation.")


def self_test():
    """Exercise production execution on CPU with an explicit test-only backend."""
    from unittest.mock import patch
    from . import execution as execution_module
    real_compile = torch.compile
    def cpu_compile(function, **kwargs):
        kwargs.pop("mode", None)
        kwargs["backend"] = "eager"
        return real_compile(function, **kwargs)
    with patch.object(execution_module, "_require_cuda", lambda model: None), \
         patch.object(execution_module.torch, "compile", cpu_compile):
        _self_test()


if __name__ == "__main__":
    arguments = tyro.cli(Config)
    if arguments.worker_spec is not None:
        worker(json.loads(arguments.worker_spec.read_text(encoding="utf-8")))
    elif arguments.self_test:
        self_test()
    else:
        run(arguments)
