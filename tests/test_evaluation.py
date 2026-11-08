"""Durable predictions, failure-aware metrics and native evaluation lifecycle."""

from dataclasses import asdict, replace
import json
import pytest
import torch
import pyarrow as pa
import pyarrow.parquet as pq
from mlflow.tracking import MlflowClient

from src.data.schema import arrow_schema
from src.evaluation.predictions import PredictionJournal, PredictionSummary, prediction
from src.evaluation.evaluate_lm import LMEvaluationConfig, evaluate_lm
from src.evaluation.evaluate_head import HeadEvaluationConfig, evaluate_head
from src.backbone.loading import BackboneConfig
from src.experiments.tracking import sync
from src.experiments.export import ExportConfig, export
from src.training.train_head import train
from test_training_state import run_setup
from test_cache import cache_setup
from test_training import single_thread_accelerate


def test_journal_discards_only_uncommitted_tail(tmp_path, example):
    path = tmp_path / "predictions.jsonl"
    journal = PredictionJournal(path)
    row = prediction(example.id, example.questions[0], dict(status="ok", scores=[0., 1.]))
    journal.append(row)
    with path.open("ab") as stream:
        stream.write(b'{"partial":')
    restored = PredictionJournal(path)
    assert restored.completed == journal.completed
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(ValueError, match="already"):
        restored.append(row)
    path.write_text("bad committed row\n")
    with pytest.raises(json.JSONDecodeError):
        PredictionJournal(path)


def test_variable_options_random_baseline_and_failure_denominators(example):
    rows = [prediction(example.id, q, dict(status="ok", scores=[0., 1.])) for q in example.questions]
    rows[1].update(status="reasoning_limit", scores=None)
    report = PredictionSummary(rows).result()
    m = report["metrics"]
    assert m["all/accuracy"] == .5
    assert m["all/completed_accuracy"] == 1
    assert m["all/coverage"] == .5
    assert m["all/random_accuracy"] == .5
    logp = torch.tensor([0., 1.]).log_softmax(0)
    assert m["all/loss"] == pytest.approx(float(-(torch.tensor([.25, .75]) * logp).sum()))
    rows[1]["targets"] = [.2, .3, .5]
    report = PredictionSummary(rows).result()
    assert report["metrics"]["all/random_accuracy"] == pytest.approx((.5 + 1/3) / 2)
    assert report["option_counts"] == {2: 1, 3: 1}


def test_native_resume_metrics_local_remote_and_cohort_validation(tmp_path, example, monkeypatch):
    import src.evaluation.native_lm as module

    dataset = tmp_path / "data"
    dataset.mkdir()
    pq.write_table(pa.Table.from_pylist([asdict(example)], schema=arrow_schema()), dataset / "validation.parquet")
    (dataset / "manifest.json").write_text(json.dumps({"splits": {}}))
    calls = []

    class FakeLM:
        tokenizer = object()
        def __init__(self, *args): pass
        def describe(self): return {"model_id": "fake", "revision": "fixed"}
        def predict(self, context, question):
            calls.append(question.id)
            if len(calls) == 2:
                raise RuntimeError("interrupted")
            return dict(status="ok", scores=[0., 1.])

    monkeypatch.setattr(module, "NativeLM", FakeLM)
    config = LMEvaluationConfig(dataset=dataset, name="native", backbone=BackboneConfig("fake"),
                              output=tmp_path / "runs", tracking_db=tmp_path / "tracking.db")
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate_lm(config)
    root = config.output / config.name
    assert len(PredictionJournal(root / "predictions.jsonl").completed) == 1
    # Legacy eager config/source predating the execution field must resume.
    saved = json.loads((root / "config.json").read_text())
    saved["config"]["inference"].pop("execution")
    saved["source"].pop("execution")
    (root / "config.json").write_text(json.dumps(saved))
    assert evaluate_lm(config)["all/accuracy"] == 1
    assert calls.count(example.questions[0].id) == 1
    before = list(calls)
    evaluate_lm(config)
    assert calls == before
    with pytest.raises(ValueError, match="configuration"):
        evaluate_lm(replace(config, inference=replace(config.inference, max_reasoning_tokens=99)))
    with pytest.raises(ValueError, match="configuration"):
        evaluate_lm(replace(config, inference=replace(config.inference, mode="thinking", execution="compiled_decode")))
    info = json.loads((root / "tracking.json").read_text())
    local = MlflowClient(tracking_uri=info["local_uri"])
    assert local.get_run(info["local_run_id"]).info.status == "FINISHED"
    remote_uri = f"sqlite:///{tmp_path / 'remote.db'}"
    sync(root, remote_uri)
    info = json.loads((root / "tracking.json").read_text())
    remote = MlflowClient(tracking_uri=remote_uri)
    assert remote.get_run(info["remote_run_id"]).data.metrics["all/accuracy"] == 1
    assert "predictions.jsonl" in {a.path for a in remote.list_artifacts(info["remote_run_id"])}
    exported = tmp_path / "export.json"
    export(ExportConfig(run_id=info["local_run_id"], local_db=config.tracking_db,
                        output=exported))
    assert json.loads(exported.read_text())["evaluation"]["statuses"] == {"ok": 2}
    reference = tmp_path / "reference.jsonl"
    row = prediction(example.id, example.questions[0], dict(status="ok", scores=[1., 0.]))
    reference.write_text(json.dumps(row) + "\n")
    cohort = replace(config, name="cohort", reference_predictions=reference)
    assert evaluate_lm(cohort)["evaluation/questions"] == 1
    row["option_ids"] = list(reversed(row["option_ids"]))
    reference.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="metadata"):
        evaluate_lm(replace(cohort, name="bad-cohort"))


def test_head_resume_uses_common_journal_and_skips_completed_batches(run_setup, monkeypatch):
    from src.evaluation.cached_head import CachedHeadSource
    import src.evaluation.cached_head as module

    training, runs = run_setup
    trained = train(training)
    config = HeadEvaluationConfig(run=trained, name="head_resume", split="test",
        output=trained.parent, tracking_db=training.tracking_db, cpu=True,
        batch_size=1, log_every=1)
    original = CachedHeadSource.predict
    calls = []
    def interrupt(self, batch):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("interrupted")
        return original(self, batch)
    monkeypatch.setattr(CachedHeadSource, "predict", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        evaluate_head(config)
    root = config.output / config.name
    journal = PredictionJournal(root / "predictions.jsonl")
    assert len(journal.completed) == 1
    first = {row["question_id"]: row for row in journal.iter_rows()}
    monkeypatch.setattr(CachedHeadSource, "predict", original)
    metrics = evaluate_head(config)
    journal = PredictionJournal(root / "predictions.jsonl")
    assert metrics["evaluation/questions"] == len(journal.completed) == 3
    restored = {row["question_id"]: row for row in journal.iter_rows()}
    assert all(restored[key] == value for key, value in first.items())
    assert runs[str(root)]["statuses"][-1] == "FINISHED"
    monkeypatch.setattr(module, "load_model", lambda *args: pytest.fail("completed run must not load weights"))
    assert evaluate_head(config) == metrics


def test_reporting_does_not_recompute_existing_predictions(example, monkeypatch):
    import src.evaluation.predictions as module
    calls = []
    original = module.question_metrics
    def count(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, "question_metrics", count)
    summary = PredictionSummary()
    for q in example.questions:
        summary.add(prediction(example.id, q, dict(status="ok", scores=[0., 1.])))
        summary.result()
    summary.result()
    assert len(calls) == len(example.questions)
