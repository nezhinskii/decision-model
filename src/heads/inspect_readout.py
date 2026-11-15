"""Standalone adaptive-readout diagnostics; never modifies a run or its weights.

Example:
    python -m src.heads.inspect_readout --run artifacts/runs/RUN_NAME

Writes summary.json and questions.jsonl under artifacts/diagnostics/RUN_NAME_readout.
The second pass replaces per-question mixtures with their validation mean. This
is an exploratory intervention in the same trained model, not a retrained static
baseline or an independently calibrated test-set comparison.
"""

from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path

from accelerate import Accelerator
from pydantic import TypeAdapter
import torch
from tqdm.auto import tqdm
import tyro

from . import HeadConfig, build_head
from .checkpoints import load_model
from .readout import LayerReadout, ReadoutMode
from ..evaluation.metrics import MetricTotals, question_log_probs, question_metrics
from ..experiments.provenance import check_cache, manifest_hash
from ..features.cache import make_loader
from ..utils.files import file_hash, write_json


@dataclass
class InspectConfig:
    run: Path
    split: str = "validation"
    batch_size: int = 64  # Contexts, as in training.
    cpu: bool = False
    cache: Path | None = None  # Override a machine-specific saved cache path.
    output: Path | None = None  # Diagnostic directory, separate from the run/cache.


def weight_summary(weights):
    """Question-weighted statistics; dominant-layer ties select the first layer."""
    weights = weights.double()
    count, layers = weights.shape
    entropy = -(weights * weights.clamp_min(1e-30).log()).sum(1)
    if layers > 1:
        entropy /= math.log(layers)
    else:
        entropy.zero_()
    return {
        "questions": count,
        "mean": weights.mean(0).tolist(),
        "std": weights.std(0, unbiased=False).tolist(),
        "min": weights.amin(0).tolist(),
        "max": weights.amax(0).tolist(),
        "dominant_fraction": (
            torch.bincount(weights.argmax(1), minlength=layers).double() / count
        ).tolist(),
        "mean_normalized_entropy": entropy.mean().item(),
    }


def fixed_mixture_hook(weights):
    """Override only the temporary readout output, using its projected inputs."""
    def replace_output(module, args, output):
        features = args[0]
        questions = features.questions[:, module.indices, :]
        options = features.options[:, module.indices, :]
        mixture = weights.to(device=questions.device, dtype=questions.dtype)
        return ((questions * mixture[None, :, None]).sum(1),
                (options * mixture[None, :, None]).sum(1))
    return replace_output


@torch.inference_mode()
def inspect(config: InspectConfig):
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if config.split != "validation":
        raise ValueError("This exploratory diagnostic uses validation only; keep test sealed")
    run = config.run.resolve()
    saved = json.loads((run / "config.json").read_text())
    head = TypeAdapter(HeadConfig).validate_python(saved["head"])
    if head.readout != ReadoutMode.adaptive:
        raise ValueError("Expected an adaptive readout checkpoint")
    cache = (config.cache or Path(saved["cache"])).resolve()
    output = (config.output or Path("artifacts/diagnostics") / f"{run.name}_readout").resolve()
    # Diagnostic outputs must never overwrite training data or model artifacts.
    if output == run or output.is_relative_to(run) or run.is_relative_to(output):
        raise ValueError("Output directory must be separate from the training run")
    if output == cache or output.is_relative_to(cache) or cache.is_relative_to(output):
        raise ValueError("Output directory must be separate from the feature cache")
    loader = make_loader(cache, config.split, head.requires_context,
                         config.batch_size, saved.get("seed", 42))
    manifest = loader.dataset.manifest
    check_cache(run, manifest)
    dimensions = json.loads((run / "provenance.json").read_text())
    layers = manifest["features"]["layers"]
    if len(layers) != dimensions["num_feature_layers"]:
        raise ValueError("Cached layers differ from saved model dimensions")
    best = json.loads((run / "best.json").read_text())
    checkpoint = (run / best["weights"]).resolve()
    if file_hash(checkpoint) != best["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    accelerator = Accelerator(cpu=config.cpu, mixed_precision=saved.get("mixed_precision", "bf16"))
    if accelerator.num_processes != 1:
        raise ValueError("Run diagnostics in one process")
    model = build_head(dimensions["input_dim"], head, dimensions["num_feature_layers"])
    load_model(checkpoint, model)
    model = accelerator.prepare(model).eval()
    readouts = [m for m in model.modules() if isinstance(m, LayerReadout)]
    if len(readouts) != 1 or readouts[0].router is None:
        raise ValueError("Expected exactly one adaptive LayerReadout with a router")
    readout = readouts[0]
    selected_layers = [layers[i] for i in readout.indices]
    captured = []

    def capture_router(module, args, result):
        # Preserve the actual forward softmax precision before copying to CPU.
        captured.append(result.squeeze(2).softmax(dim=1).detach().float().cpu())

    rows, all_weights, seen = [], [], set()
    adaptive_totals = MetricTotals()
    handle = readout.router.register_forward_hook(capture_router)
    try:
        for batch in tqdm(loader, desc="Adaptive weights", unit="batch"):
            captured.clear()
            with accelerator.autocast():
                logits = model(batch.features.to(accelerator.device)).float().cpu()
            count = len(batch.features.question_ids)
            if len(captured) != 1 or captured[0].shape != (count, len(selected_layers)):
                raise RuntimeError("Router hook did not capture one aligned weight matrix")
            weights = captured[0]
            if not torch.isfinite(weights).all():
                raise ValueError("Nonfinite router weights")
            values = question_metrics(
                question_log_probs(logits, batch.features.option_to_question, count),
                batch.targets, batch.features.option_to_question, count,
            )
            adaptive_totals.add(values, batch.categories, batch.target_kinds)
            all_weights.append(weights)
            for q, question_id in enumerate(batch.features.question_ids):
                if question_id in seen:
                    raise ValueError(f"Duplicate question ID: {question_id}")
                seen.add(question_id)
                mask = batch.features.option_to_question == q
                rows.append({
                    "context_id": batch.context_ids[int(batch.features.question_to_context[q])],
                    "question_id": question_id,
                    "category": batch.categories[q], "target_kind": str(batch.target_kinds[q]),
                    "option_count": int(mask.sum()), "weights": weights[q].tolist(),
                    "predicted_option": int(logits[mask].argmax()),
                    "target_option": int(batch.targets[mask].argmax()),
                    "correct": bool(values[q, 1]),
                    "ce": float(values[q, 0]), "brier": float(values[q, 2]),
                })
    finally:
        handle.remove()
    if not rows or len(rows) != manifest["splits"][config.split]["questions"]:
        raise ValueError("Diagnostic population differs from cache manifest")
    weights = torch.cat(all_weights)
    mean_weights = weights.double().mean(0).float()
    fixed_totals = MetricTotals()
    cursor = 0
    handle = readout.register_forward_hook(fixed_mixture_hook(mean_weights))
    try:
        for batch in tqdm(loader, desc="Fixed mean intervention", unit="batch"):
            with accelerator.autocast():
                logits = model(batch.features.to(accelerator.device)).float().cpu()
            count = len(batch.features.question_ids)
            values = question_metrics(
                question_log_probs(logits, batch.features.option_to_question, count),
                batch.targets, batch.features.option_to_question, count,
            )
            fixed_totals.add(values, batch.categories, batch.target_kinds)
            for q, question_id in enumerate(batch.features.question_ids):
                row = rows[cursor]
                if row["question_id"] != question_id:
                    raise RuntimeError("Cache order changed between diagnostic passes")
                mask = batch.features.option_to_question == q
                row.update(fixed_mean_correct=bool(values[q, 1]),
                           fixed_mean_predicted_option=int(logits[mask].argmax()),
                           fixed_mean_ce=float(values[q, 0]), fixed_mean_brier=float(values[q, 2]))
                cursor += 1
    finally:
        handle.remove()
    if cursor != len(rows):
        raise RuntimeError("Cache population changed between diagnostic passes")
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[f"category/{row['category']}"].append(i)
        groups[f"target/{row['target_kind']}"].append(i)
        groups["correct" if row["correct"] else "incorrect"].append(i)
    adaptive, fixed = adaptive_totals.result(), fixed_totals.result()
    report = {
        "run": str(run), "checkpoint": str(checkpoint), "checkpoint_sha256": best["sha256"],
        "training_config_sha256": file_hash(run / "config.json"),
        "cache": str(cache), "cache_manifest_sha256": manifest_hash(manifest),
        "split": config.split, "device": str(accelerator.device),
        "mixed_precision": saved.get("mixed_precision", "bf16"), "batch_size": config.batch_size,
        "layers": selected_layers, "overall": weight_summary(weights),
        "groups": {name: weight_summary(weights[indices]) for name, indices in sorted(groups.items())},
        "saved_best_metrics": best["metrics"], "adaptive_metrics": adaptive,
        "fixed_mean_metrics": fixed,
        "fixed_minus_adaptive": {key: fixed[key] - value for key, value in adaptive.items()},
        "prediction_changes": sum(r["predicted_option"] != r["fixed_mean_predicted_option"] for r in rows),
        "correct_to_wrong": sum(r["correct"] and not r["fixed_mean_correct"] for r in rows),
        "wrong_to_correct": sum(not r["correct"] and r["fixed_mean_correct"] for r in rows),
        "notes": [
            "Layer arrays follow the layers field; option indices are zero-based within each question.",
            "std is population standard deviation; entropy is normalized by log(number of selected layers).",
            "Mean weights are computed on this validation set; this intervention is exploratory, not an independent test.",
            "Weights alone do not establish causal importance. Fixed-mean intervention does not retrain the head.",
            "Compare recomputed adaptive_metrics with saved_best_metrics before interpreting small differences.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "questions.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output / "questions.jsonl")
    write_json(output / "summary.json", report)
    print("Layers:", selected_layers)
    print("Mean weights:", [round(v, 4) for v in report["overall"]["mean"]])
    print("Weight std:", [round(v, 4) for v in report["overall"]["std"]])
    for label, metrics in (("Adaptive", adaptive), ("Fixed mean", fixed)):
        print(f"{label}: accuracy={metrics['all/accuracy']:.2%}, CE={metrics['all/loss']:.6f}, Brier={metrics['all/brier']:.6f}")
    print(f"Reports: {output}")
    return report


if __name__ == "__main__":
    inspect(tyro.cli(InspectConfig))
