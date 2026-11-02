"""Optimizer-update schedules independent of microbatch accumulation."""

from enum import StrEnum
import math

import torch


class ScheduleType(StrEnum):
    constant = "constant"
    cosine = "cosine"


class CosineSchedule(torch.optim.lr_scheduler.LRScheduler):
    """Decay to a fixed fraction, then hold it rather than starting another cycle."""

    def __init__(self, optimizer, total_steps, min_lr_ratio):
        if total_steps < 1 or not 0 <= min_lr_ratio <= 1:
            raise ValueError("Schedule needs positive steps and min_lr_ratio in [0, 1]")
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer)

    def get_lr(self):
        fraction = min(self.last_epoch / self.total_steps, 1.0)
        factor = (
            self.min_lr_ratio
            + (1 - self.min_lr_ratio) * (1 + math.cos(math.pi * fraction)) / 2
        )
        return [base_lr * factor for base_lr in self.base_lrs]

    def load_state_dict(self, state):
        if (
            state["total_steps"] != self.total_steps
            or state["min_lr_ratio"] != self.min_lr_ratio
        ):
            raise ValueError("Resume changed the cosine schedule horizon or minimum LR")
        super().load_state_dict(state)
