"""Train LoRA/QLoRA and any existing decision head directly on prepared texts."""

from dataclasses import dataclass, replace
from importlib.metadata import version
import json
import math
from pathlib import Path

import torch
import tyro

from ..backbone.dataset import DecisionDataset, make_decision_loader
from ..backbone.inputs import InputBuilder
from ..backbone.forward import validate_attention
from ..backbone.loading import ModelSource
from ..experiments.provenance import data_summary
from ..models.decision import DecisionConfig, DecisionModel, source_identity
from ..models.checkpoints import DecisionState, save_best
from .config import TrainingConfig
from .runner import TrainingSetup, run_training


@dataclass(kw_only=True)
class DecisionTrainingConfig(TrainingConfig):
    dataset: Path
    model: DecisionConfig
    batch_size: int = 1
    accumulation_steps: int = 8
    adapter_learning_rate: float = 2e-4


def prepare_decision(config, accelerator, root, resuming, source, identity):
    device = torch.device(config.model.backbone.device)
    if device.type != accelerator.device.type or (
        device.type == "cuda" and (device.index or 0) != (accelerator.device.index or 0)
    ):
        raise ValueError(
            "Backbone device must match the single-process Accelerator device"
        )
    if (
        not math.isfinite(config.adapter_learning_rate)
        or config.adapter_learning_rate <= 0
    ):
        raise ValueError("adapter_learning_rate must be finite and positive")
    hf_config = source._load_config()
    text = getattr(hf_config, "text_config", hf_config)
    validate_attention(text, config.model.inputs.attention)
    builder = InputBuilder(source.tokenizer, config.model.inputs)
    datasets = {
        split: DecisionDataset(
            config.dataset,
            split,
            builder,
            text.max_position_embeddings,
            seed=config.seed,
            shuffle=split == "train",
        )
        for split in ("train", "validation")
    }
    if any(not len(data) for data in datasets.values()):
        raise ValueError("No eligible train/validation contexts; review token budgets")
    selection = {split: data.summary for split, data in datasets.items()}
    # Identity checks happen before loading weights or reopening tracking.
    summary = data_summary(config.dataset)
    model_source = {"base": identity, "peft": version("peft")}
    if resuming:
        for name, value in (
            ("data_selection.json", selection),
            ("data_summary.json", summary),
            ("model_source.json", model_source),
        ):
            saved = json.loads((root / name).read_text())
            if saved != json.loads(json.dumps(value, default=str)):
                raise ValueError(f"Live training source changed: {name}")
    model = DecisionModel(source.load_text_decoder(), config.model, identity)
    groups = [
        {
            "params": list(model.head.parameters()),
            "lr": config.learning_rate,
            "name": "head",
        }
    ]
    adapters = [p for p in model.decoder.parameters() if p.requires_grad]
    if adapters:
        groups.append(
            {"params": adapters, "lr": config.adapter_learning_rate, "name": "adapter"}
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    updates = math.ceil(
        math.ceil(len(datasets["train"]) / config.batch_size)
        / config.accumulation_steps
    )
    return TrainingSetup(
        model=model,
        optimizer=optimizer,
        train_loader=make_decision_loader(datasets["train"], config.batch_size),
        val_loader=make_decision_loader(datasets["validation"], config.batch_size),
        updates_per_epoch=updates,
        save_best=save_best,
        model_state=DecisionState(),
        device_placement=False,
        artifacts={
            "data_summary.json": summary,
            "data_selection.json": selection,
            "model_source.json": model_source,
            "adapter_summary.json": model.adapters,
        },
        provenance={
            "input_dim": text.hidden_size,
            "num_feature_layers": len(model.layers),
            "layers": list(model.layers),
            "base": identity,
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "head_parameters": sum(p.numel() for p in model.head.parameters()),
            "adapter_parameters": model.adapters["trainable_parameters"],
        },
        tags={
            "model_family": "decision",
            "processor": str(config.model.head.processor),
            "adaptation": (
                "qlora"
                if adapters and config.model.backbone.quantization == "nf4"
                else "lora" if adapters else "frozen"
            ),
        },
    )


def train(config: DecisionTrainingConfig):
    if config.cpu and config.model.backbone.quantization == "nf4":
        raise ValueError(
            "NF4 training requires CUDA; use quantization=none for CPU tests"
        )
    backbone = config.model.backbone
    previous = config.output / config.name / "config.json"
    if previous.exists() and config.resume and backbone.revision is None:
        saved = json.loads(previous.read_text())
        backbone = replace(backbone, revision=saved["model"]["backbone"]["revision"])
    source = ModelSource(backbone)
    identity = source_identity(source)
    backbone = replace(backbone, revision=identity["revision"])
    source.config = backbone
    config = replace(config, model=replace(config.model, backbone=backbone))
    return run_training(
        config,
        lambda accelerator, root, resuming: prepare_decision(
            config, accelerator, root, resuming, source, identity
        ),
    )


if __name__ == "__main__":
    train(tyro.cli(DecisionTrainingConfig))
