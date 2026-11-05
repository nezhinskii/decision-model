"""Portable head weights, independent of optimizer state and training orchestration."""

import os
from pathlib import Path
from safetensors.torch import load_file, save_file


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


def load_model(path, model):
    model.load_state_dict(load_file(str(path)), strict=True)
