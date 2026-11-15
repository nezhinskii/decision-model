"""Saved live decision-model adapter for the shared evaluation lifecycle."""

from dataclasses import replace
import json
from pathlib import Path
import time

from accelerate import Accelerator
from pydantic import TypeAdapter
import torch

from ..backbone.dataset import DecisionDataset, RejectedExample, make_decision_loader
from ..backbone.inputs import InputBuilder
from ..backbone.loading import ModelSource
from ..backbone.forward import validate_attention
from ..models.decision import DecisionConfig, DecisionModel, source_identity
from ..models.checkpoints import load_best
from ..experiments.provenance import data_summary
from ..utils.files import file_hash
from .predictions import prediction


class DecisionSource:
    def __init__(self, config):
        self.config = config
        saved = json.loads((config.run / "config.json").read_text())
        self.settings = TypeAdapter(DecisionConfig).validate_python(saved["model"])
        if (
            self.settings.backbone.quantization == "nf4"
            and not config.device.startswith("cuda")
        ):
            raise ValueError("Saved NF4 decision models require a CUDA device")
        self.settings = replace(
            self.settings,
            backbone=replace(self.settings.backbone, device=config.device),
        )
        self.source = ModelSource(self.settings.backbone)
        self.identity = source_identity(self.source)
        hf_config = self.source._load_config()
        text = getattr(hf_config, "text_config", hf_config)
        validate_attention(text, self.settings.inputs.attention)
        dataset = config.dataset or Path(saved["dataset"])
        self.dataset = DecisionDataset(
            dataset,
            config.split,
            InputBuilder(self.source.tokenizer, self.settings.inputs),
            text.max_position_embeddings,
            seed=saved.get("seed", 42),
        )
        self.loader = make_decision_loader(self.dataset, config.batch_size)
        self.precision = saved.get("mixed_precision", "bf16")
        best = json.loads((config.run / "best.json").read_text())
        self.weights = config.run / best["weights"]
        if file_hash(self.weights) != best["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
        tracking = json.loads((config.run / "tracking.json").read_text())
        self.signature = {
            "source_train_run_id": tracking["local_run_id"],
            "checkpoint_sha256": best["sha256"],
            "training_config_sha256": file_hash(config.run / "config.json"),
            "split_sha256": self.dataset.summary["split_sha256"],
            "selection": self.dataset.summary,
            "model": self.identity,
        }
        self.provenance = {**self.signature, "checkpoint": str(self.weights.resolve())}
        self.artifacts = {
            "data_summary.json": data_summary(dataset),
            "data_selection.json": {config.split: self.dataset.summary},
        }
        self.tags = {
            "model_family": "decision",
            "source_train_run_id": tracking["local_run_id"],
            "checkpoint_sha256": best["sha256"],
        }
        self.model = self.accelerator = None

    def batches(self):
        yield from self.loader
        # Preserve the entire evaluation population, including over-budget contexts.
        yield from self.dataset.rejected

    def rows(self, batch):
        examples = (
            [batch.example]
            if isinstance(batch, RejectedExample)
            else [s.example for s in batch.inputs.samples]
        )
        return [
            prediction(example.id, question, {})
            for example in examples
            for question in example.questions
        ]

    @torch.inference_mode()
    def predict(self, batch):
        if isinstance(batch, RejectedExample):
            return [
                dict(status="context_limit", scores=None, reason=batch.reason)
                for _ in batch.example.questions
            ]
        if self.model is None:
            self.accelerator = Accelerator(
                cpu=self.config.device == "cpu", mixed_precision=self.precision
            )
            requested = torch.device(self.config.device)
            actual = self.accelerator.device
            if (
                self.accelerator.num_processes != 1
                or requested.type != actual.type
                or (
                    requested.type == "cuda"
                    and (requested.index or 0) != (actual.index or 0)
                )
            ):
                raise ValueError(
                    "Decision evaluation requires one process on the requested device"
                )
            self.model = DecisionModel(
                self.source.load_text_decoder(),
                self.settings,
                self.identity,
                training=False,
            )
            load_best(self.config.run, self.model)
            self.model = self.accelerator.prepare_model(
                self.model, device_placement=False
            ).eval()
        started = time.perf_counter()
        with self.accelerator.autocast():
            logits = self.model(batch.inputs).float().cpu()
        seconds = (time.perf_counter() - started) / batch.num_questions
        return [
            dict(
                status="ok",
                scores=logits[batch.option_to_question == q].tolist(),
                seconds=seconds,
            )
            for q in range(batch.num_questions)
        ]
