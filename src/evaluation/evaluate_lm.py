"""Evaluate a native language model on prepared question texts."""

from dataclasses import dataclass, field
from pathlib import Path
import tyro

from ..backbone.loading import BackboneConfig
from ..native_lm.model import NativeLMConfig
from .config import EvaluationConfig
from .native_lm import NativeLMSource
from .run import run_evaluation


@dataclass(kw_only=True)
class LMEvaluationConfig(EvaluationConfig):
    dataset: Path
    backbone: BackboneConfig
    inference: NativeLMConfig = field(default_factory=NativeLMConfig)


def evaluate_lm(config: LMEvaluationConfig):
    return run_evaluation(config, kind="lm",
                          source_factory=lambda previous: NativeLMSource(config, previous))


if __name__ == "__main__":
    print(evaluate_lm(tyro.cli(LMEvaluationConfig)))
