"""Saved head and feature cache adapter; no tracking or evaluation lifecycle."""

from dataclasses import dataclass
import json
from pathlib import Path

from accelerate import Accelerator
from pydantic import TypeAdapter
import torch

from ..experiments.provenance import cache_summary, check_cache
from ..features.cache import make_loader
from ..utils.files import file_hash
from ..heads import HeadConfig, build_head
from ..heads.checkpoints import load_model


@dataclass
class SavedHeadConfig:
    """Only inference-relevant fields of the saved training configuration."""

    cache: Path
    head: HeadConfig
    seed: int = 42
    mixed_precision: str = "bf16"


class CachedHeadSource:
    def __init__(self, config):
        self.config = config
        self.saved = TypeAdapter(SavedHeadConfig).validate_json(
            (config.run / "config.json").read_text()
        )
        best = json.loads((config.run / "best.json").read_text())
        self.weights = config.run / best["weights"]
        if file_hash(self.weights) != best["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        context = self.saved.head.requires_context
        self.loader = make_loader(
            self.saved.cache, config.split, context, config.batch_size, self.saved.seed
        )
        manifest = self.loader.dataset.manifest
        check_cache(config.run, manifest)
        self.dimensions = json.loads((config.run / "provenance.json").read_text())
        tracking = json.loads((config.run / "tracking.json").read_text())
        self.tags = {
            "model_family": "cached_head",
            "source_train_run_id": tracking["local_run_id"],
            "checkpoint_sha256": best["sha256"],
        }
        self.artifacts = {
            "cache_summary.json": cache_summary(self.saved.cache, manifest)
        }
        data = config.run / "data_summary.json"
        if data.exists():
            self.artifacts["data_summary.json"] = json.loads(data.read_text())
        self.signature = {
            "checkpoint_sha256": best["sha256"],
            "training_config_sha256": file_hash(config.run / "config.json"),
            "dimensions": {
                key: self.dimensions[key] for key in ("input_dim", "num_feature_layers")
            },
            "cache_manifest_sha256": self.artifacts["cache_summary.json"][
                "manifest_sha256"
            ],
            "source_train_run_id": tracking["local_run_id"],
        }
        self.provenance = {
            "checkpoint": str(self.weights.resolve()),
            "cache": str(self.saved.cache.resolve()),
            **self.signature["dimensions"],
            **self.signature,
        }
        self.model = self.accelerator = None

    def batches(self):
        return iter(self.loader)

    def rows(self, batch):
        """Translate cache metadata here; shared evaluation never sees features."""
        features = batch.features
        rows = []
        for q, question_id in enumerate(features.question_ids):
            mask = features.option_to_question == q
            options = mask.nonzero().flatten().tolist()
            rows.append(
                dict(
                    context_id=batch.context_ids[int(features.question_to_context[q])],
                    question_id=question_id,
                    option_ids=[features.option_ids[i] for i in options],
                    targets=batch.targets[mask].float().tolist(),
                    category=batch.categories[q],
                    task=str(batch.tasks[q]),
                    target_kind=str(batch.target_kinds[q]),
                )
            )
        return rows

    @torch.inference_mode()
    def predict(self, batch):
        if self.model is None:
            self.accelerator = Accelerator(
                cpu=self.config.cpu, mixed_precision=self.saved.mixed_precision
            )
            if self.accelerator.num_processes != 1:
                raise ValueError("Evaluation supports one process only")
            self.model = build_head(
                self.dimensions["input_dim"],
                self.saved.head,
                self.dimensions["num_feature_layers"],
            )
            load_model(self.weights, self.model)
            self.model = self.accelerator.prepare(self.model).eval()
        with self.accelerator.autocast():
            logits = (
                self.model(batch.features.to(self.accelerator.device)).float().cpu()
            )
        return [
            dict(
                status="ok",
                scores=logits[batch.features.option_to_question == q].tolist(),
            )
            for q in range(len(batch.features.question_ids))
        ]
