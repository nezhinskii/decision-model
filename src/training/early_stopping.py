"""Validation-based stopping, independent of checkpoint selection."""

from dataclasses import dataclass
import math


@dataclass
class EarlyStoppingConfig:
    patience: int = 8  # Distinct validation steps; zero disables stopping.
    min_delta: float = 1e-4  # Absolute CE improvement required to reset patience.

    def __post_init__(self):
        if self.patience < 0 or not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("Early stopping needs nonnegative patience and finite min_delta")


class EarlyStopping:
    """Persist only a small state dictionary in the training checkpoint."""

    def __init__(self, config, state=None):
        self.config = config
        self.state = state if state is not None else {
            'best': float('inf'), 'bad_checks': 0, 'last_step': -1, 'stopped': False,
        }

    def update(self, loss, step):
        if step <= self.state['last_step']:
            return  # A periodic validation and epoch end can share one step.
        self.state['last_step'] = step
        if not math.isfinite(loss):
            raise FloatingPointError('Nonfinite validation CE')
        if loss < self.state['best'] - self.config.min_delta:
            self.state['best'] = loss
            self.state['bad_checks'] = 0
        else:
            self.state['bad_checks'] += 1
        self.state['stopped'] = (self.config.patience > 0
                                 and self.state['bad_checks'] >= self.config.patience)

    @property
    def stopped(self):
        return self.state['stopped']
