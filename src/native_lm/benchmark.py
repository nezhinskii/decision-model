"""Compare generation overhead and NF4/BF16 on one fixed prompt, without tracking.

Run: uv run python -m src.native_lm.benchmark
This measures generation only, not answer scoring or dataset throughput.
"""

from dataclasses import dataclass
import gc
import importlib
import inspect
import json
import os
import platform
from pathlib import Path
import statistics
import time

import torch
import tyro

from ..backbone.loading import BackboneConfig, Quantization
from ..utils.files import write_json
from .model import NativeLM, NativeLMConfig


@dataclass
class BenchmarkConfig:
    model_id: str = "Qwen/Qwen3.5-4B"
    revision: str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    tokens: int = 256
    repeats: int = 3
    profile: bool = True
    profile_only: bool = False
    compile_decode: bool = False  # Add a BF16 static-cache/compiled comparison.
    quantizations: tuple[Quantization, ...] = (Quantization.nf4, Quantization.none)
    profile_tokens: int = 32
    device: str = "cuda:0"
    cache_dir: Path = Path("artifacts/models/hub")
    output: Path = Path("artifacts/diagnostics/native_generation.json")


def _runtime_info(lm):
    """Inspect actual loaded modules/backends only in this diagnostic command."""
    model = lm.load()
    device = model.get_input_embeddings().weight.device
    quantized_modules = [module for module in model.modules() if type(module).__name__ == "Linear4bit"]
    quantized = len(quantized_modules)
    loaded_4bit = bool(getattr(model, "is_loaded_in_4bit", False) or quantized)
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
    info = dict(
        device=str(device), loaded_in_4bit=loaded_4bit, linear4bit_modules=quantized,
        quantized_compute_dtypes=sorted({str(module.compute_dtype) for module in quantized_modules}),
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        embedding_dtype=str(model.get_input_embeddings().weight.dtype),
        weights_gib=footprint, kernels=providers,
        attention_backend=getattr(model.config, "_attn_implementation", "unknown"),
    )
    return info


def _measure(lm, inputs, tokens, project):
    model = lm.load()
    device = inputs.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    if project:
        _, _, metadata = lm._generate_reasoning(model, inputs, tokens)
        count = metadata["reasoning_tokens"]
    else:
        output = model.generate(
            input_ids=inputs, attention_mask=torch.ones_like(inputs),
            do_sample=False, use_cache=True, max_new_tokens=tokens,
            return_dict_in_generate=True, output_scores=False, output_logits=False,
            pad_token_id=(lm.tokenizer.pad_token_id if lm.tokenizer.pad_token_id is not None
                          else lm.tokenizer.eos_token_id),
        )
        count = output.sequences.shape[1] - inputs.shape[1]
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    if count != tokens:
        raise RuntimeError(f"Expected {tokens} generated tokens, got {count}")
    return dict(seconds=seconds, tokens=count, tokens_per_second=count / seconds,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3)


def _profile(lm, inputs, config, name):
    """Record warmed decode, excluding prefill and four extra decode warmup steps.

    The post-forward hook advances profiler steps. It exists only during this
    diagnostic call; production generation never installs profiling hooks.
    CPU dispatch gaps are visible in the exported timeline, not the tables.
    """
    model = lm.load()
    stem = config.output.parent / f"{config.output.stem}_{name}"
    result = {}
    budget = 1 + 4 + config.profile_tokens + 1
    previous_minimum = model.generation_config.min_new_tokens
    model.generation_config.min_new_tokens = budget

    def save(profiler):
        trace = stem.with_suffix(".trace.json")
        table = stem.with_suffix(".operators.txt")
        profiler.export_chrome_trace(str(trace))
        averages = profiler.key_averages()
        table.write_text(
            "CPU operators (profiled timings include profiler overhead):\n"
            + averages.table(sort_by="self_cpu_time_total", row_limit=30)
            + "\nCUDA operators:\n"
            + averages.table(sort_by="self_device_time_total", row_limit=30), encoding="utf-8")
        cuda_events = sum(event.device_type.name == "CUDA" for event in profiler.events())
        result.update(trace=str(trace), operators=str(table), cuda_events=cuda_events)
        if not cuda_events:
            print("WARNING: no CUDA events captured (check CUPTI availability); CPU trace alone cannot identify GPU bottlenecks.", flush=True)
        print(f"Profile saved: {trace}\nOperator tables: {table}", flush=True)

    print(f"\n{name}: profiling {config.profile_tokens} warmed decode steps (not a speed measurement)", flush=True)
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=4, active=config.profile_tokens, repeat=1),
            on_trace_ready=save, record_shapes=False, with_stack=False,
        ) as profiler:
            hook = model.register_forward_hook(lambda *args: profiler.step())
            try:
                _measure(lm, inputs, budget, project=False)
            finally:
                hook.remove()
    finally:
        model.generation_config.min_new_tokens = previous_minimum
    return result


@torch.inference_mode()
def benchmark(config: BenchmarkConfig):
    if config.tokens < 2 or config.repeats < 1 or config.profile_tokens < 1:
        raise ValueError("Use at least two tokens and one repeat")
    if not config.quantizations or len(set(config.quantizations)) != len(config.quantizations):
        raise ValueError("Choose distinct quantizations")
    if config.compile_decode and Quantization.none not in config.quantizations:
        raise ValueError("Compiled decode comparison requires --quantizations none")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    report = dict(config=vars(config), torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(config.device),
                  cpu=platform.processor(), logical_cpus=os.cpu_count(),
                  torch_threads=torch.get_num_threads(),
                  torch_interop_threads=torch.get_num_interop_threads(), modes={})
    config.output.parent.mkdir(parents=True, exist_ok=True)
    reference_ids = None
    for quantization in config.quantizations:
        lm = NativeLM(BackboneConfig(config.model_id, revision=config.revision,
                                    quantization=quantization, device=config.device,
                                    cache_dir=config.cache_dir),
                      NativeLMConfig(mode="thinking", max_reasoning_tokens=config.tokens))
        prompt = lm.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Find all integer solutions to x^2 + y^2 = 2025. Explain your derivation carefully."}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = lm.tokenizer.encode(prompt, add_special_tokens=False)
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise RuntimeError("Input token IDs differ across model loads")
        model = lm.load()
        inputs = torch.tensor([ids], device=config.device)
        closing = lm.tokenizer.encode("</think>", add_special_tokens=False)
        if len(closing) != 1:
            raise RuntimeError("Fixed-length benchmark expects a single Qwen </think> special token")
        # Suppress both normal EOS and </think> until the fixed token budget.
        # BOTH paths use this same generation config, ensuring identical lengths
        # without changing our generation helper or removing its stop checks.
        eos = model.generation_config.eos_token_id
        eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        model.generation_config.eos_token_id = list(dict.fromkeys(eos + closing))
        model.generation_config.min_new_tokens = config.tokens
        model.generation_config.do_sample = False
        runtime = _runtime_info(lm)
        print("Native LM runtime: " + json.dumps(runtime, default=str), flush=True)
        paths = [(False, False)]
        if quantization == Quantization.nf4 and not config.profile_only:
            paths.append((True, False))
        if quantization == Quantization.none and config.compile_decode:
            paths.append((False, True))
        for project, compiled in paths:
            name = f"{quantization}_{'project' if project else 'minimal'}"
            if compiled:
                name += "_compiled"
                from transformers import CompileConfig, StaticCache

                # Fullgraph makes graph breaks explicit instead of reporting a
                # partially eager run as a successful compiled comparison.
                model.generation_config.cache_implementation = "static"
                model.generation_config.compile_config = CompileConfig(
                    fullgraph=True, dynamic=False, mode="reduce-overhead")
                model.generation_config.disable_compile = False
                cache = StaticCache(config=model.config, max_cache_len=len(ids) + config.tokens)
                if not model._valid_auto_compile_criteria(
                    {"past_key_values": cache}, model.generation_config
                ):
                    raise RuntimeError("This runtime cannot compile Qwen decode with a static cache")
                del cache
                print("Compiled BF16 decode: first warmup includes compilation; this may take minutes.", flush=True)
            else:
                model.generation_config.cache_implementation = None
                model.generation_config.compile_config = None
                model.generation_config.disable_compile = True
            print(f"\n{name}: warmup ({config.tokens} tokens)", flush=True)
            _measure(lm, inputs, config.tokens, project)
            if compiled:
                # Give CUDA graph recording its own untimed iteration as well.
                _measure(lm, inputs, config.tokens, project)
            samples = []
            for i in range(0 if config.profile_only else config.repeats):
                sample = _measure(lm, inputs, config.tokens, project)
                samples.append(sample)
                print(f"  {i + 1}/{config.repeats}: {sample['tokens_per_second']:.2f} tok/s, "
                      f"{sample['seconds']:.3f}s, peak {sample['peak_allocated_gib']:.2f} GiB", flush=True)
            report["modes"][name] = dict(
                input_tokens=len(ids), runtime=runtime, samples=samples,
                execution=dict(compiled=compiled, cache="static" if compiled else "dynamic"))
            if samples:
                report["modes"][name]["median_tokens_per_second"] = statistics.median(
                    s["tokens_per_second"] for s in samples)
            if not project and (config.profile or config.profile_only):
                report["modes"][name]["profile"] = _profile(lm, inputs, config, name)
            # Save completed measurements even if the next mode fails or OOMs.
            write_json(config.output, report)
        del model, lm, inputs
        gc.collect()
        torch.cuda.empty_cache()
    print("\nMedian generation speed (includes prefill):", flush=True)
    for name, values in report["modes"].items():
        if "median_tokens_per_second" in values:
            print(f"  {name}: {values['median_tokens_per_second']:.2f} tok/s")
    print(f"Results: {config.output.resolve()}")


if __name__ == "__main__":
    benchmark(tyro.cli(BenchmarkConfig))
