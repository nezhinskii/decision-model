"""Atomic local model exports and trusted training-state snapshots."""

import hashlib
import os
from pathlib import Path
import random

import numpy as np
from safetensors.torch import load_file, save_file
import torch


def weight_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_model(root, model):
    """Replace best weights only after the temporary file is fully written."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "best.safetensors"
    temporary = path.with_suffix(".tmp")
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
        temporary,
    )
    os.replace(temporary, path)
    return path


def save_latest(path, model, optimizer, accelerator, progress):
    """Save only after an optimizer step; gradients are then empty by construction."""
    state = {
        "model": accelerator.unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "progress": progress,
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


def load_latest(path, model, optimizer, accelerator):
    """Training pickle is trusted local input, never a downloaded arbitrary artifact."""
    state = torch.load(path, map_location="cpu", weights_only=False)
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


def load_model(path, model):
    model.load_state_dict(load_file(str(path)), strict=True)
