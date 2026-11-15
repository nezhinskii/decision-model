"""Assemble cached-head training; the shared runner owns the run lifecycle."""

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path

import torch
import tyro

from ..features.cache import make_loader
from ..heads import HeadConfig, build_head
from ..heads.checkpoints import save_model
from ..experiments.provenance import cache_summary, data_summary, check_cache
from .config import TrainingConfig, MixedPrecision
from .runner import TrainingSetup, run_training


@dataclass
class TrainConfig(TrainingConfig):
    cache: Path
    head: HeadConfig = field(default_factory=HeadConfig)


def prepare_head(config, accelerator, root, resuming):
    """Resolve cache, construct head/optimizer and describe this specific task."""
    context = config.head.requires_context
    train_loader = make_loader(
        config.cache, "train", context, config.batch_size, config.seed, True, head_config=config.head
    )
    val_loader = make_loader(
        config.cache, "validation", context, config.batch_size, config.seed, head_config=config.head
    )
    manifest = train_loader.dataset.manifest
    if resuming:
        check_cache(root, manifest)
    first = next(iter(train_loader))
    dimensions = {
        "input_dim": first.features.questions.shape[-1],
        "num_feature_layers": first.features.questions.shape[1],
    }
    del first
    model = build_head(config=config.head, **dimensions)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    artifacts = {"cache_summary.json": cache_summary(config.cache, manifest)}
    prepared = manifest.get("config", {}).get("dataset")
    if prepared and (Path(prepared) / "manifest.json").exists():
        artifacts["data_summary.json"] = data_summary(Path(prepared))
    updates = ceil(
        ceil(train_loader.dataset.info["contexts"] / config.batch_size)
        / config.accumulation_steps
    )
    return TrainingSetup(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        updates_per_epoch=updates,
        save_best=save_model,
        artifacts=artifacts,
        provenance={
            **dimensions,
            "cache": str(config.cache.resolve()),
            "context_layer": train_loader.dataset.context_layer,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        tags={"processor": str(config.head.processor)},
    )


def train(config: TrainConfig):
    return run_training(
        config,
        lambda accelerator, root, resuming: prepare_head(
            config, accelerator, root, resuming
        ),
    )


if __name__ == "__main__":
    train(tyro.cli(TrainConfig))
