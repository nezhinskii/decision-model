"""Evaluate a saved local best checkpoint as a separate linked MLflow run."""

from dataclasses import dataclass
import json
from pathlib import Path
import uuid
import subprocess
import platform
import torch

from accelerate import Accelerator
from pydantic import TypeAdapter
import tyro

from ..heads import build_head
from .checkpoints import load_model, weight_hash
from .loop import evaluate
from .train_head import TrainConfig, make_loader
from .tracking import Tracker, try_sync, write_json
from .provenance import check_cache


@dataclass
class EvaluateConfig:
    run: Path
    split: str = "test"
    batch_size: int = 8
    cpu: bool = False
    remote_uri: str | None = None


def evaluate_checkpoint(config: EvaluateConfig):
    train = TypeAdapter(TrainConfig).validate_json(
        (config.run / "config.json").read_text()
    )
    best = json.loads((config.run / "best.json").read_text())
    path = config.run / best["weights"]
    if weight_hash(path) != best["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    context = train.head.processor == "joint" and train.head.use_cross_attention
    loader = make_loader(
        train.cache, config.split, context, config.batch_size, train.seed
    )
    check_cache(config.run, train.cache, loader.dataset.manifest)
    dimensions = json.loads((config.run / "provenance.json").read_text())
    model = build_head(
        dimensions["input_dim"], train.head, dimensions["num_feature_layers"]
    )
    load_model(path, model)
    accelerator = Accelerator(cpu=config.cpu, mixed_precision=train.mixed_precision)
    model = accelerator.prepare(model)
    root = config.run / "evaluations" / uuid.uuid4().hex
    root.mkdir(parents=True)
    source_tracking = json.loads((config.run / "tracking.json").read_text())
    tracker = Tracker(
        root,
        source_tracking["local_uri"],
        train.experiment,
        {
            "task": "evaluate",
            "source_train_run_id": source_tracking["local_run_id"],
            "checkpoint_sha256": best["sha256"],
            "split": config.split,
        },
    )
    try:
        metrics = evaluate(model, loader, accelerator)
        tracker.log(metrics, best["step"])
        write_json(
            root / "evaluation.json",
            {
                "metrics": metrics,
                "split": config.split,
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": best["sha256"],
                "source_train_run_id": source_tracking["local_run_id"],
                "python": platform.python_version(),
                "torch": torch.__version__,
                "git_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"], capture_output=True, text=True
                ).stdout.strip(),
                "git_status": subprocess.run(
                    ["git", "status", "--porcelain"], capture_output=True, text=True
                ).stdout.strip(),
                "batch_size": config.batch_size,
                "device": str(accelerator.device),
                "cache": str(train.cache.resolve()),
            },
        )
        tracker.artifact(root / "evaluation.json")
        tracker.status("FINISHED")
    except BaseException:
        tracker.status("FAILED")
        raise
    finally:
        try_sync(root, config.remote_uri)
    return metrics


if __name__ == "__main__":
    print(evaluate_checkpoint(tyro.cli(EvaluateConfig)))
