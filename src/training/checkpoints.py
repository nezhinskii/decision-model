"""Trusted training-state snapshots: optimizer, RNG, model and data progress."""

import os
from pathlib import Path
import random
from typing import Protocol

import numpy as np
import torch


class ModelState(Protocol):
    """Model-specific weights only; the training runner owns optimizer/RNG state.

    A future adapter codec can use PEFT without teaching this module about PEFT.
    Its format must identify a stable representation, not an individual run.
    """

    format: str

    def dump(self, model) -> dict: ...

    def restore(self, model, state: dict) -> None: ...


class FullModelState:
    """Existing head checkpoints: all parameters and persistent buffers."""

    format = "full"

    def dump(self, model):
        return model.state_dict()

    def restore(self, model, state):
        model.load_state_dict(state, strict=True)


def save_latest(
    path,
    model,
    optimizer,
    accelerator,
    progress,
    scheduler=None,
    *,
    model_state: ModelState | None = None,
):
    """Save only after an optimizer step; gradients are then empty by construction."""
    codec = model_state if model_state is not None else FullModelState()
    state = {
        "model": codec.dump(accelerator.unwrap_model(model)),
        "model_state_format": codec.format,
        "optimizer": optimizer.state_dict(),
        "progress": progress,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "scaler": (
            accelerator.scaler.state_dict() if accelerator.scaler is not None else None
        ),
    }
    temporary = Path(path).with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_latest(
    path,
    model,
    optimizer,
    accelerator,
    scheduler=None,
    *,
    model_state: ModelState | None = None,
):
    """Training pickle is trusted local input, never a downloaded arbitrary artifact."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    codec = model_state if model_state is not None else FullModelState()
    if state.get("model_state_format", "full") != codec.format:
        raise ValueError(
            "Checkpoint model-state format differs from the requested model"
        )
    saved_schedule = state.get("scheduler")
    if (scheduler is None) != (saved_schedule is None):
        raise ValueError("Checkpoint scheduler differs from the requested schedule")
    if scheduler is not None:
        scheduler.load_state_dict(saved_schedule)
    codec.restore(accelerator.unwrap_model(model), state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if accelerator.scaler is not None and state["scaler"] is not None:
        accelerator.scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["progress"]
