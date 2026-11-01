"""Train a head using a completed feature cache; no backbone is instantiated."""

from dataclasses import asdict, dataclass, field
from itertools import islice
import json
from pathlib import Path
import subprocess
import time
import platform
from importlib.metadata import version
from enum import StrEnum

from accelerate import Accelerator
from accelerate.utils import set_seed
from pydantic import TypeAdapter
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import tyro

from ..feature_cache.dataset import FeatureDataset, collate_features
from ..heads import HeadConfig, build_head
from .checkpoints import load_latest, save_latest, save_model, weight_hash
from .loop import evaluate, train_step
from .tracking import Tracker, BackgroundSync, write_json
from .provenance import cache_summary, data_summary, check_cache


class MixedPrecision(StrEnum):
    no = "no"
    bf16 = "bf16"
    fp16 = "fp16"


@dataclass
class TrainConfig:
    cache: Path
    name: str
    head: HeadConfig = field(default_factory=HeadConfig)
    output: Path = Path("artifacts/runs")
    tracking_db: Path = Path("artifacts/mlflow/tracking.db")
    experiment: str = "decision-heads"
    remote_uri: str | None = None
    epochs: int = 5
    batch_size: int = 16
    accumulation_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    seed: int = 42
    cpu: bool = False
    mixed_precision: MixedPrecision = MixedPrecision.bf16
    # Intervals count optimizer updates, including gradient accumulation.
    log_every: int = 25
    validate_every: int = 500
    save_every: int = 500
    resume: bool = True
    max_steps: int | None = None


def make_loader(cache, split, context, batch_size, seed, shuffle=False):
    dataset = FeatureDataset(
        cache, split, shuffle=shuffle, seed=seed, include_context=context
    )
    # Dedicated generator prevents iterator creation/skipping from consuming model
    # RNG on resume. num_workers=0 fixes the reproducible source cursor.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_features,
        num_workers=0,
        generator=generator,
    )


def train(config: TrainConfig):
    if (
        min(
            config.epochs,
            config.batch_size,
            config.accumulation_steps,
            config.validate_every,
            config.save_every,
            config.log_every,
        )
        < 1
    ):
        raise ValueError("Epochs, batch sizes and intervals must be positive")
    if Path(config.name).name != config.name or config.name in (".", ".."):
        raise ValueError("Run name must be one directory name")
    root = config.output / config.name
    if root.exists() and not config.resume:
        raise ValueError("Run exists; use --resume or a new --name")
    resuming = (root / "latest.pt").exists() and config.resume
    if root.exists() and any(root.iterdir()) and not resuming:
        raise ValueError(
            "Run directory has files but no resumable latest.pt; use a new name"
        )
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if resuming:
        old = TypeAdapter(TrainConfig).validate_json(config_path.read_text())
        ignored = {
            "resume",
            "epochs",
            "max_steps",
            "remote_uri",
            "save_every",
            "validate_every",
            "log_every",
        }
        if {k: v for k, v in asdict(old).items() if k not in ignored} != {
            k: v for k, v in asdict(config).items() if k not in ignored
        }:
            raise ValueError("Resume configuration changed the model/data/optimizer")
    accelerator = Accelerator(cpu=config.cpu, mixed_precision=config.mixed_precision)
    if accelerator.num_processes != 1:
        raise ValueError("This initial training loop supports one process only")
    set_seed(config.seed)
    context = config.head.processor == "joint" and config.head.use_cross_attention
    train_loader = make_loader(
        config.cache, "train", context, config.batch_size, config.seed, True
    )
    val_loader = make_loader(
        config.cache, "validation", context, config.batch_size, config.seed
    )
    # Reject an incompatible cache before reopening the tracking run as RUNNING.
    if resuming:
        check_cache(root, config.cache, train_loader.dataset.manifest)
    first = next(iter(train_loader))
    dimensions = {
        "input_dim": first.features.questions.shape[-1],
        "num_feature_layers": first.features.questions.shape[1],
    }
    del first
    model = build_head(config=config.head, **dimensions)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    config.tracking_db.parent.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(
        root,
        f"sqlite:///{config.tracking_db.resolve()}",
        config.experiment,
        {
            "mlflow.runName": config.name,
            "task": "train",
            "processor": str(config.head.processor),
        },
    )
    if not resuming:
        write_json(config_path, asdict(config))
        write_json(
            root / "cache_summary.json",
            cache_summary(config.cache, train_loader.dataset.manifest),
        )
        prepared_path = train_loader.dataset.manifest.get("config", {}).get("dataset")
        if prepared_path and (Path(prepared_path) / "manifest.json").exists():
            write_json(
                root / "data_summary.json",
                data_summary(Path(prepared_path)),
            )

        def git(*args):
            return subprocess.run(
                ["git", *args], capture_output=True, text=True
            ).stdout.strip()

        write_json(
            root / "provenance.json",
            {
                **dimensions,
                "git_commit": git("rev-parse", "HEAD"),
                "git_status": git("status", "--porcelain"),
                "cache": str(config.cache.resolve()),
                "parameters": sum(p.numel() for p in model.parameters()),
                "torch": torch.__version__,
            },
        )
        provenance = json.loads((root / "provenance.json").read_text())
        provenance.update(
            {
                "python": platform.python_version(),
                "device": str(accelerator.device),
                "packages": {
                    name: version(name)
                    for name in ("accelerate", "safetensors", "mlflow", "pyarrow")
                },
                "learning_rate_schedule": "constant",
            }
        )
        write_json(root / "provenance.json", provenance)
    for name in ("config.json", "cache_summary.json", "provenance.json"):
        if not (root / name).exists():
            continue
        tracker.artifact(root / name)
    if (root / "data_summary.json").exists():
        tracker.artifact(root / "data_summary.json")
    progress = {"epoch": 0, "batch": 0, "step": 0, "best_loss": float("inf")}
    if resuming:
        progress = load_latest(root / "latest.pt", model, optimizer, accelerator)
        write_json(root / "resume_config.json", asdict(config))
        tracker.artifact(root / "resume_config.json")
        # A best export may be newer than the last periodic training snapshot.
        # Keep it rather than replacing it with a worse post-resume validation.
        if (root / "best.json").exists():
            best = json.loads((root / "best.json").read_text())
            progress["best_loss"] = min(
                progress["best_loss"], best["metrics"]["all/loss"]
            )
    else:
        save_latest(root / "latest.pt", model, optimizer, accelerator, progress)
    model.train()
    sender = BackgroundSync(root, config.remote_uri)

    def validate():
        metrics = evaluate(model, val_loader, accelerator)
        tracker.log(
            {f"validation/{k}": v for k, v in metrics.items()}, progress["step"]
        )
        if metrics["all/loss"] < progress["best_loss"]:
            progress["best_loss"] = metrics["all/loss"]
            old = (
                json.loads((root / "best.json").read_text())
                if (root / "best.json").exists()
                else None
            )
            weights = save_model(root / "checkpoints", accelerator.unwrap_model(model))
            write_json(
                root / "best.json",
                {
                    "weights": str(weights.relative_to(root)),
                    "sha256": weight_hash(weights),
                    "step": progress["step"],
                    "metrics": metrics,
                },
            )
            tracker.artifact(root / "best.json")
            # Remove a previous UUID-named export from older runs, but keep the
            # fixed best.safetensors path that has just been replaced.
            if old and root / old["weights"] != weights:
                (root / old["weights"]).unlink(missing_ok=True)
        sender.request()

    try:
        for epoch in range(progress["epoch"], config.epochs):
            train_loader.dataset.set_epoch(epoch)
            iterator = iter(train_loader)
            for _ in range(progress["batch"]):
                next(iterator)
            interval_sums = None  # [3]: CE, accuracy, Brier sums on device.
            interval_questions = interval_steps = 0
            started = time.perf_counter()
            if accelerator.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            def log_interval():
                nonlocal interval_sums, interval_questions, interval_steps, started
                if not interval_steps:
                    return
                # One device-to-host transfer per interval completes these metrics.
                # Wall time includes loading/CPU work; no explicit CUDA synchronize.
                averages = (interval_sums / interval_questions).cpu().tolist()
                elapsed = time.perf_counter() - started
                metrics = {
                    f"train/all/{key}": value
                    for key, value in zip(("loss", "accuracy", "brier"), averages)
                }
                metrics.update(
                    {
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/step_seconds": elapsed / interval_steps,
                        "train/questions_per_second": interval_questions / elapsed,
                    }
                )
                if accelerator.device.type == "cuda":
                    metrics["train/peak_allocated_gib"] = (
                        torch.cuda.max_memory_allocated() / 2**30
                    )
                    torch.cuda.reset_peak_memory_stats()
                tracker.log(metrics, progress["step"])
                bar.set_postfix(loss=averages[0])
                interval_sums = None
                interval_questions = interval_steps = 0
                started = time.perf_counter()

            with tqdm(desc=f"epoch {epoch + 1}", unit="step") as bar:
                while batches := list(islice(iterator, config.accumulation_steps)):
                    if (
                        config.max_steps is not None
                        and progress["step"] >= config.max_steps
                    ):
                        break
                    sums, questions = train_step(
                        model, batches, optimizer, accelerator, config.clip_grad
                    )
                    interval_sums = (
                        sums if interval_sums is None else interval_sums + sums
                    )
                    interval_questions += questions
                    interval_steps += 1
                    progress["step"] += 1
                    progress["batch"] += len(batches)
                    bar.update()
                    if progress["step"] % config.log_every == 0:
                        log_interval()
                    if progress["step"] % config.validate_every == 0:
                        log_interval()
                        validate()
                        if accelerator.device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter()
                    if progress["step"] % config.save_every == 0:
                        log_interval()
                        save_latest(
                            root / "latest.pt", model, optimizer, accelerator, progress
                        )
                        started = time.perf_counter()
                else:
                    progress["epoch"], progress["batch"] = epoch + 1, 0
                log_interval()
            validate()
            save_latest(root / "latest.pt", model, optimizer, accelerator, progress)
            if config.max_steps is not None and progress["step"] >= config.max_steps:
                break
        tracker.status("FINISHED")
    except BaseException:
        # Never serialize partially accumulated gradients as a resumable boundary.
        # The previous periodic latest checkpoint remains authoritative.
        import sys

        tracker.status("KILLED" if sys.exc_info()[0] is KeyboardInterrupt else "FAILED")
        raise
    finally:
        sender.close()
    return root


if __name__ == "__main__":
    train(tyro.cli(TrainConfig))
