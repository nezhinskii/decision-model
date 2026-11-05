"""Compare generation overhead and NF4/BF16 on one fixed prompt, without tracking.

Run: uv run python -m src.native_lm.benchmark
This measures generation only, not answer scoring or dataset throughput.
"""

from dataclasses import dataclass
import gc
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
    device: str = "cuda:0"
    cache_dir: Path = Path("artifacts/models/hub")
    output: Path = Path("artifacts/diagnostics/native_generation.json")


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


@torch.inference_mode()
def benchmark(config: BenchmarkConfig):
    if config.tokens < 2 or config.repeats < 1:
        raise ValueError("Use at least two tokens and one repeat")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    report = dict(config=vars(config), torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(config.device), modes={})
    config.output.parent.mkdir(parents=True, exist_ok=True)
    reference_ids = None
    for quantization in (Quantization.nf4, Quantization.none):
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
        paths = (False, True) if quantization == Quantization.nf4 else (False,)
        for project in paths:
            name = f"{quantization}_{'project' if project else 'minimal'}"
            print(f"\n{name}: warmup ({config.tokens} tokens)", flush=True)
            _measure(lm, inputs, config.tokens, project)
            samples = []
            for i in range(config.repeats):
                sample = _measure(lm, inputs, config.tokens, project)
                samples.append(sample)
                print(f"  {i + 1}/{config.repeats}: {sample['tokens_per_second']:.2f} tok/s, "
                      f"{sample['seconds']:.3f}s, peak {sample['peak_allocated_gib']:.2f} GiB", flush=True)
            report["modes"][name] = dict(
                input_tokens=len(ids), runtime=lm.runtime_info, samples=samples,
                median_tokens_per_second=statistics.median(s["tokens_per_second"] for s in samples))
            # Save completed measurements even if the next mode fails or OOMs.
            write_json(config.output, report)
        del model, lm, inputs
        gc.collect()
        torch.cuda.empty_cache()
    print("\nMedian generation speed (includes prefill):", flush=True)
    for name, values in report["modes"].items():
        print(f"  {name}: {values['median_tokens_per_second']:.2f} tok/s")
    print(f"Results: {config.output.resolve()}")


if __name__ == "__main__":
    benchmark(tyro.cli(BenchmarkConfig))
