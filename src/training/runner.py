"""Training lifecycle shared by task assemblers; no cache or backbone knowledge.

An assembler receives the accelerator and resume identity, validates its data,
and returns TrainingSetup. Batches expose inputs, targets, option_to_question,
num_questions, categories, target_kinds and to(device). The runner never inspects
feature tensors, token layouts, head configuration or adapter modules.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from itertools import islice
import json
from pathlib import Path
import time

from accelerate import Accelerator
from accelerate.utils import set_seed
from pydantic import TypeAdapter
import torch
from tqdm.auto import tqdm

from .checkpoints import FullModelState, ModelState, load_latest, save_latest
from .config import TrainingConfig
from .early_stopping import EarlyStopping
from .loop import train_step, validate
from .objectives import TRAIN_METRIC_NAMES
from .schedules import ScheduleType, CosineSchedule
from ..experiments.tracking import Tracker, BackgroundSync
from ..experiments.provenance import environment_info
from ..utils.files import file_hash, write_json


@dataclass
class TrainingSetup:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: object
    val_loader: object
    updates_per_epoch: int
    save_best: Callable[[Path, torch.nn.Module], Path]
    model_state: ModelState = field(default_factory=FullModelState)
    device_placement: bool = True
    artifacts: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    tags: dict = field(default_factory=dict)


def run_training(config: TrainingConfig, prepare: Callable) -> Path:
    """Own progress, scheduling, validation, persistence and tracking."""
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
        old = TypeAdapter(type(config)).validate_json(config_path.read_text())
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
    setup = prepare(accelerator, root, resuming)
    model, optimizer = setup.model, setup.optimizer
    train_loader, val_loader = setup.train_loader, setup.val_loader
    updates_per_epoch = setup.updates_per_epoch
    if updates_per_epoch < 1:
        raise ValueError(
            "Training setup must contain at least one optimizer update per epoch"
        )
    total_schedule_steps = config.schedule_steps or updates_per_epoch * config.epochs
    if config.schedule_steps is not None and config.schedule_steps < 1:
        raise ValueError("schedule_steps must be positive")
    scheduler = (
        CosineSchedule(optimizer, total_schedule_steps, config.min_lr_ratio)
        if config.schedule == ScheduleType.cosine
        else None
    )
    if setup.device_placement:
        model, optimizer = accelerator.prepare(model, optimizer)
    else:
        # Live decoders (including NF4) are already placed by their loader.
        model = accelerator.prepare_model(model, device_placement=False)
        optimizer = accelerator.prepare_optimizer(optimizer)
    config.tracking_db.parent.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(
        root,
        f"sqlite:///{config.tracking_db.resolve()}",
        config.experiment,
        {
            "mlflow.runName": config.name,
            "task": "train",
            **setup.tags,
        },
    )
    if not resuming:
        write_json(config_path, asdict(config))
        for name, document in setup.artifacts.items():
            write_json(root / name, document)
        write_json(
            root / "provenance.json",
            {
                **environment_info(),
                **setup.provenance,
                "device": str(accelerator.device),
                "learning_rate_schedule": str(config.schedule),
                "schedule_steps": (
                    total_schedule_steps if scheduler is not None else None
                ),
            },
        )
    for name in ("config.json", "provenance.json", *setup.artifacts):
        if (root / name).exists():
            tracker.artifact(root / name)
    progress = {"epoch": 0, "batch": 0, "step": 0, "best_loss": float("inf")}
    if resuming:
        progress = load_latest(
            root / "latest.pt",
            model,
            optimizer,
            accelerator,
            scheduler,
            model_state=setup.model_state,
        )
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
        save_latest(
            root / "latest.pt",
            model,
            optimizer,
            accelerator,
            progress,
            scheduler,
            model_state=setup.model_state,
        )
    model.train()
    stopper = EarlyStopping(config.early_stopping, progress.get("early_stopping"))
    progress["early_stopping"] = stopper.state
    sender = BackgroundSync(root, config.remote_uri)

    def record_validation():
        if progress["step"] <= stopper.state["last_step"]:
            return
        metrics = validate(model, val_loader, accelerator)
        tracker.log(
            {f"validation/{k}": v for k, v in metrics.items()}, progress["step"]
        )
        stopper.update(metrics["all/loss"], progress["step"])
        tracker.log(
            {
                "early_stopping/bad_checks": stopper.state["bad_checks"],
                "early_stopping/stopped": int(stopper.stopped),
            },
            progress["step"],
        )
        if metrics["all/loss"] < progress["best_loss"]:
            progress["best_loss"] = metrics["all/loss"]
            old = (
                json.loads((root / "best.json").read_text())
                if (root / "best.json").exists()
                else None
            )
            weights = setup.save_best(
                root / "checkpoints", accelerator.unwrap_model(model)
            )
            write_json(
                root / "best.json",
                {
                    "weights": str(weights.relative_to(root)),
                    "sha256": file_hash(weights),
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
            if stopper.stopped:
                break
            train_loader.dataset.set_epoch(epoch)
            iterator = iter(train_loader)
            for _ in range(progress["batch"]):
                next(iterator)
            interval_sums = None  # [6]: TRAIN_METRIC_NAMES sums on device.
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
                raw = interval_sums.cpu().tolist()
                metric_count = len(TRAIN_METRIC_NAMES)
                averages = [value / interval_questions for value in raw[:metric_count]]
                elapsed = time.perf_counter() - started
                metrics = {
                    f"train/{key}": value
                    for key, value in zip(TRAIN_METRIC_NAMES, averages)
                }
                if len(raw) > metric_count:
                    options, balance_sum, *assignments = raw[metric_count:]
                    metrics["train/routing/balance_loss"] = balance_sum / options
                    total = sum(assignments)
                    for i, count in enumerate(assignments):
                        metrics[f"train/routing/expert_{i}_fraction"] = count / total
                metrics.update(
                    {
                        "train/learning_rate": used_lr,
                        **used_group_lrs,
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
                    used_lr = optimizer.param_groups[0]["lr"]
                    used_group_lrs = {
                        f"train/{group['name']}_learning_rate": group["lr"]
                        for group in optimizer.param_groups if "name" in group
                    }
                    sums, questions = train_step(
                        model,
                        batches,
                        optimizer,
                        accelerator,
                        config.clip_grad,
                        config.loss,
                    )
                    if (
                        scheduler is not None
                        and not accelerator.optimizer_step_was_skipped
                    ):
                        scheduler.step()
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
                        record_validation()
                        if stopper.stopped:
                            break
                        if accelerator.device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats()
                        started = time.perf_counter()
                    if progress["step"] % config.save_every == 0:
                        log_interval()
                        save_latest(
                            root / "latest.pt",
                            model,
                            optimizer,
                            accelerator,
                            progress,
                            scheduler,
                            model_state=setup.model_state,
                        )
                        started = time.perf_counter()
                else:
                    progress["epoch"], progress["batch"] = epoch + 1, 0
                log_interval()
            record_validation()
            save_latest(
                root / "latest.pt",
                model,
                optimizer,
                accelerator,
                progress,
                scheduler,
                model_state=setup.model_state,
            )
            if stopper.stopped:
                tqdm.write(
                    f"Early stopping at step {progress['step']}: "
                    f"{stopper.state['bad_checks']} checks without CE improvement "
                    f"> {config.early_stopping.min_delta}. Best weights are saved."
                )
                break
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
