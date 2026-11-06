"""Prepared text and native LM adapter; tracking and resume live in run.py."""

from dataclasses import replace

from ..data.dataset import read_examples
from ..experiments.provenance import data_summary
from ..utils.files import file_hash
from ..native_lm.inputs import PROMPT_VERSION
from ..native_lm.model import NativeLM
from .predictions import prediction


class NativeLMSource:
    def __init__(self, config, previous=None):
        self.config = config
        self.path = config.dataset / f"{config.split}.parquet"
        # Restore the resolved revision through configuration, without mutating
        # the loader's internal state or changing the user's requested config.
        backbone = (
            replace(config.backbone, revision=previous["model"]["revision"])
            if previous is not None
            else config.backbone
        )
        self.model = NativeLM(backbone, config.inference)
        identity = self.model.describe()
        _ = self.model.tokenizer
        self.signature = {
            "model": identity,
            "split_sha256": file_hash(self.path),
            "prompt_version": PROMPT_VERSION,
        }
        self.provenance = dict(self.signature)
        self.tags = {"model_family": "native_lm", "mode": str(config.inference.mode)}
        self.artifacts = {"data_summary.json": data_summary(config.dataset)}

    def batches(self):
        for example in read_examples(self.path):
            for question in example.questions:
                yield example, question

    def rows(self, batch):
        example, question = batch
        return [prediction(example.id, question, {})]

    def predict(self, batch):
        example, question = batch
        return [self.model.predict(example.context, question)]
