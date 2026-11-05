"""Trusted training-state snapshots: optimizer, RNG, model and data progress."""

import os
from pathlib import Path
import random

import numpy as np
import torch


def save_latest(path, model, optimizer, accelerator, progress, scheduler=None):
    """Save only after an optimizer step; gradients are then empty by construction."""
    state = {
        "model": accelerator.unwrap_model(model).state_dict(),
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


def load_latest(path, model, optimizer, accelerator, scheduler=None):
    """Training pickle is trusted local input, never a downloaded arbitrary artifact."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    saved_schedule = state.get('scheduler')
    if (scheduler is None) != (saved_schedule is None):
        raise ValueError("Checkpoint scheduler differs from the requested schedule")
    if scheduler is not None:
        scheduler.load_state_dict(saved_schedule)
    accelerator.unwrap_model(model).load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    if accelerator.scaler is not None and state["scaler"] is not None:
        accelerator.scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["progress"]
