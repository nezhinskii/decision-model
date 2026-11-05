"""Run options shared by the two standalone evaluation commands."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(kw_only=True)
class EvaluationConfig:
    name: str
    split: str = "validation"
    output: Path = Path("artifacts/runs")
    tracking_db: Path = Path("artifacts/mlflow/tracking.db")
    experiment: str = "decision-heads"
    remote_uri: str | None = None
    reference_predictions: Path | None = None
    log_every: int = 25
