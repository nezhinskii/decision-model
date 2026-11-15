"""Evaluate a saved backbone + LoRA + head on prepared texts, without a feature cache."""

from dataclasses import dataclass
from pathlib import Path
import tyro

from .config import EvaluationConfig
from .decision import DecisionSource
from .run import run_evaluation


@dataclass(kw_only=True)
class DecisionEvaluationConfig(EvaluationConfig):
    run: Path
    dataset: Path | None = None  # Default: the saved training dataset.
    device: str = "cuda:0"
    batch_size: int = 1


def evaluate_decision(config: DecisionEvaluationConfig):
    if config.batch_size < 1:
        raise ValueError("Batch size must be positive")
    return run_evaluation(config, kind="decision", source_factory=lambda previous: DecisionSource(config))


if __name__ == "__main__":
    print(evaluate_decision(tyro.cli(DecisionEvaluationConfig)))
