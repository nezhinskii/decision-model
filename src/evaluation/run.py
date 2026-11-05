"""One evaluation lifecycle for cached heads and native language models."""

from dataclasses import asdict
import json
from pathlib import Path

from tqdm.auto import tqdm

from ..experiments.provenance import environment_info
from ..experiments.tracking import Tracker, BackgroundSync
from ..utils.files import file_hash, write_json
from .config import EvaluationConfig
from .predictions import PredictionJournal, PredictionSummary, load_reference, check_question


def run_evaluation(config: EvaluationConfig, *, kind: str, source_factory):
    if not config.name or config.name in {".", ".."} or Path(config.name).name != config.name:
        raise ValueError("name must be a single directory name")
    if config.split not in {"train", "validation", "test"}:
        raise ValueError("Unknown split")
    if config.log_every < 1:
        raise ValueError("Log interval must be positive")
    root = config.output / config.name
    previous = json.loads((root / "config.json").read_text()) if (root / "config.json").exists() else None
    requested = json.loads(json.dumps(asdict(config), default=str))
    requested.pop("remote_uri")  # A failed remote transfer may be retried elsewhere.
    if previous is not None and (previous.get("config") != requested or previous.get("kind") != kind):
        raise ValueError("Evaluation configuration changed; choose a new name")
    # The command supplies its model-specific source factory; this runner does
    # not import either model path. Sources expose aligned predictions and never
    # create runs, write journals, calculate reports or manage remote tracking.
    source = source_factory(previous["source"] if previous is not None else None)
    reference = load_reference(config.reference_predictions)
    signature = dict(kind=kind, config=requested, source=source.signature,
                     reference_sha256=file_hash(config.reference_predictions) if reference is not None else None)
    root.mkdir(parents=True, exist_ok=True)
    if previous is not None:
        if signature != previous:
            raise ValueError("Evaluation configuration or source changed; choose a new name")
    elif any(root.iterdir()):
        raise ValueError("Destination is not an evaluation run")
    else:
        for name, value in {
            "config.json": signature,
            "provenance.json": {**environment_info(), **source.provenance},
            **source.artifacts,
        }.items():
            write_json(root / name, value)
    journal = PredictionJournal(root / "predictions.jsonl")
    # Rebuild once on resume, then update metrics once per new prediction.
    summary = PredictionSummary(journal.iter_rows())
    config.tracking_db.parent.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(root, f"sqlite:///{config.tracking_db.resolve()}", config.experiment,
                      {"task": "evaluate", "split": config.split,
                       "mlflow.runName": config.name, **source.tags})
    sender = BackgroundSync(root, config.remote_uri)
    try:
        for name in ("config.json", "provenance.json", *source.artifacts):
            tracker.artifact(root / name)
        seen = set()
        for batch in tqdm(source.batches(), desc=f"Evaluate {kind}", unit="batch"):
            rows = source.rows(batch)
            selected = []
            for index, row in enumerate(rows):
                identity = row["question_id"]
                if reference is not None and identity not in reference:
                    continue
                if identity in seen:
                    raise ValueError("Duplicate dataset question ID")
                seen.add(identity)
                check_question(row, reference.get(identity) if reference is not None else None)
                check_question(row, journal.completed.get(identity))
                if identity not in journal.completed:
                    selected.append(index)
            if not selected:
                continue
            outputs = source.predict(batch)
            if len(outputs) != len(rows):
                raise ValueError("Source returned a different number of predictions")
            for index in selected:
                row = {**rows[index], **outputs[index]}
                journal.append(row)
                summary.add(row)
                if len(journal.completed) % config.log_every == 0:
                    tracker.log(summary.result()["metrics"], len(journal.completed))
                    sender.request()
        if reference is not None and set(reference) != seen:
            raise ValueError("Some reference questions are missing from the dataset")
        if set(journal.completed) != seen or not seen:
            raise ValueError("Prediction journal does not match the selected dataset")
        report = summary.result()
        report.update(split=config.split, kind=kind, **source.provenance)
        write_json(root / "evaluation.json", report)
        tracker.log(report["metrics"], len(journal.completed))
        for name in ("predictions.jsonl", "evaluation.json"):
            tracker.artifact(root / name)
        tracker.status("FINISHED")
    except BaseException:
        tracker.status("FAILED")
        if journal.path.exists():
            tracker.artifact(journal.path)
        raise
    finally:
        sender.close()
    return report["metrics"]
