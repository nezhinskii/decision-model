"""One atomic portable file for adapter + head; never serializes frozen base weights."""

from dataclasses import asdict
import json
from pathlib import Path

from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ..utils.files import file_hash


def model_signature(model):
    settings = asdict(model.settings)
    # Preserve signatures of pre-MoE models; routing settings only affect MoE.
    if settings["head"]["processor"] != "question_option_moe":
        settings["head"].pop("num_experts")
        settings["head"].pop("experts_per_option")
    # Placement and download location may change when evaluating a saved model.
    for key in ("device", "cache_dir", "local_files_only"):
        settings["backbone"].pop(key, None)
    return json.loads(
        json.dumps({"model": settings, "base": model.identity}, default=str)
    )


class DecisionState:
    format = "decision-adapter-head-v1"

    def dump(self, model):
        return {
            "signature": model_signature(model),
            "adapter": (
                get_peft_model_state_dict(model.decoder, save_embedding_layers=False)
                if model.settings.lora.enabled
                else {}
            ),
            "head": model.head.state_dict(),
        }

    def restore(self, model, state):
        if state["signature"] != model_signature(model):
            raise ValueError("Decision checkpoint model/base configuration differs")
        expected = (
            get_peft_model_state_dict(model.decoder, save_embedding_layers=False)
            if model.settings.lora.enabled
            else {}
        )
        for name, actual, required in (
            ("adapter", state["adapter"], expected),
            ("head", state["head"], model.head.state_dict()),
        ):
            if set(actual) != set(required) or any(
                actual[k].shape != required[k].shape for k in required
            ):
                raise ValueError(f"Decision checkpoint {name} keys/shapes differ")
        # Check every part before mutating either part.
        if model.settings.lora.enabled:
            result = set_peft_model_state_dict(model.decoder, state["adapter"])
            if result.unexpected_keys:
                raise ValueError(
                    f"Unexpected adapter weights: {result.unexpected_keys}"
                )
        model.head.load_state_dict(state["head"], strict=True)


def save_best(root, model):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    state = DecisionState().dump(model)
    tensors = {
        f"{part}.{key}": tensor.detach().cpu().contiguous()
        for part in ("adapter", "head")
        for key, tensor in state[part].items()
    }
    path = root / "best.safetensors"
    temporary = path.with_suffix(".tmp")
    save_file(
        tensors,
        temporary,
        metadata={
            "format": DecisionState.format,
            "signature": json.dumps(state["signature"], sort_keys=True),
        },
    )
    temporary.replace(path)
    return path


def load_best(run, model):
    run = Path(run)
    best = json.loads((run / "best.json").read_text())
    path = run / best["weights"]
    if file_hash(path) != best["sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    with safe_open(path, framework="pt", device="cpu") as stream:
        metadata = stream.metadata() or {}
    if metadata.get("format") != DecisionState.format:
        raise ValueError("Not a decision adapter/head checkpoint")
    tensors = load_file(str(path))
    if any(not key.startswith(("adapter.", "head.")) for key in tensors):
        raise ValueError("Unexpected decision checkpoint namespace")
    state = {
        part: {
            key[len(part) + 1 :]: value
            for key, value in tensors.items()
            if key.startswith(part + ".")
        }
        for part in ("adapter", "head")
    }
    state["signature"] = json.loads(metadata["signature"])
    DecisionState().restore(model, state)
    return best
