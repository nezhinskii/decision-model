"""Model-independent training and run settings; CLI fields remain flat."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from .early_stopping import EarlyStoppingConfig
from .objectives import LossConfig
from .schedules import ScheduleType


class MixedPrecision(StrEnum):
    no = "no"
    bf16 = "bf16"
    fp16 = "fp16"


@dataclass(kw_only=True)
class TrainingConfig:
    name: str
    loss: LossConfig = field(default_factory=LossConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    output: Path = Path("artifacts/runs")
    tracking_db: Path = Path("artifacts/mlflow/tracking.db")
    experiment: str = "decision-heads"
    remote_uri: str | None = None
    epochs: int = 5
    batch_size: int = 16
    accumulation_steps: int = 1
    learning_rate: float = 1e-4
    schedule: ScheduleType = ScheduleType.constant
    min_lr_ratio: float = 0.1
    schedule_steps: int | None = None  # None uses all requested epochs, not max_steps.
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
