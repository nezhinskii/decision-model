"""Evaluate a trained head against a feature cache."""

from dataclasses import dataclass
from pathlib import Path
import tyro

from .cached_head import CachedHeadSource
from .config import EvaluationConfig
from .run import run_evaluation


@dataclass(kw_only=True)
class HeadEvaluationConfig(EvaluationConfig):
    run: Path
    batch_size: int = 8
    cpu: bool = False


def evaluate_head(config: HeadEvaluationConfig):
    if config.batch_size < 1:
        raise ValueError("Batch size must be positive")
    return run_evaluation(config, kind="head",
                          source_factory=lambda previous: CachedHeadSource(config))


if __name__ == "__main__":
    print(evaluate_head(tyro.cli(HeadEvaluationConfig)))
