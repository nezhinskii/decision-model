"""Import one MLflow run locally and write a compact, disposable analysis report."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import tyro

from .tracking import REPORT_ARTIFACTS, client, load_tracking_env


@dataclass
class ExportConfig:
    run_id: str
    source_uri: str | None = None  # None reads the local tracking database.
    local_db: Path = Path("artifacts/mlflow/tracking.db")
    output: Path | None = None  # Defaults to artifacts/exports/<run_id>.json.
    import_local: bool = True
    max_points: int | None = None  # None keeps complete overall curves.


def latest_per_step(history):
    """A replayed step uses its latest timestamp in the report, not in MLflow."""
    steps = {}
    for metric in sorted(history, key=lambda m: (m.step, m.timestamp)):
        steps[metric.step] = metric
    return [steps[step] for step in sorted(steps)]


def curve_summary(history, limit):
    """Keep full curves by default; optional thinning preserves endpoints/extrema."""
    points = latest_per_step(history)
    finite = [i for i, m in enumerate(points) if math.isfinite(m.value)]
    if not points:
        return {}
    if limit is None:
        limit = len(points)
    selected = {0, len(points) - 1}
    if finite:
        selected.update((min(finite, key=lambda i: points[i].value),
                         max(finite, key=lambda i: points[i].value)))
    if len(points) <= limit:
        selected.update(range(len(points)))
    else:
        # Fill the remaining budget with approximately evenly spaced samples.
        for i in range(limit):
            if len(selected) >= limit:
                break
            selected.add(round(i * (len(points) - 1) / (limit - 1)))
    return {
        "logged_points": len(history), "unique_steps": len(points),
        "nonfinite_points": sum(not math.isfinite(m.value) for m in history),
        "first": [points[0].step, points[0].value],
        "last": [points[-1].step, points[-1].value],
        "min": ([points[min(finite, key=lambda i: points[i].value)].step,
                 min(points[i].value for i in finite)] if finite else None),
        "max": ([points[max(finite, key=lambda i: points[i].value)].step,
                 max(points[i].value for i in finite)] if finite else None),
        "curve": [[points[i].step, points[i].value] for i in sorted(selected)],
    }


def import_run(source, run, histories, artifacts, local_uri, source_uri):
    """Copy full histories to an identifiable local run; retries add only new points.

    The imported run is a viewing copy, not a resumable training directory.
    Existing training runs are never overwritten. Run one importer at a time.
    """
    experiment = source.get_experiment(run.info.experiment_id).name
    local, experiment_id = client(local_uri, experiment)
    identity = hashlib.sha256(f"{source_uri}\n{run.info.run_id}".encode()).hexdigest()
    matches = local.search_runs([experiment_id],
                               filter_string=f"tags.import_identity = '{identity}'")
    target = matches[0] if matches else local.create_run(
        experiment_id, start_time=run.info.start_time,
        tags={"import_identity": identity},
    )
    target_id = target.info.run_id
    for key, value in run.data.tags.items():
        # Parent IDs and filesystem paths belong to the source machine/store.
        if key not in ("mlflow.parentRunId", "local_directory"):
            local.set_tag(target_id, key, value)
    for key, value in {"import_identity": identity, "import_source_uri": source_uri,
                       "import_source_run_id": run.info.run_id,
                       "imported_view_only": "true"}.items():
        local.set_tag(target_id, key, value)
    for key, value in run.data.params.items():
        local.log_param(target_id, key, value)
    for key, history in histories.items():
        known = {(m.step, m.timestamp, repr(m.value))
                 for m in local.get_metric_history(target_id, key)}
        missing = [m for m in history if (m.step, m.timestamp, repr(m.value)) not in known]
        for start in range(0, len(missing), 500):
            local.log_batch(target_id, metrics=missing[start:start + 500])
    for path in artifacts.values():
        local.log_artifact(target_id, str(path))
    if run.info.status in ("RUNNING", "SCHEDULED"):
        local.update_run(target_id, status=run.info.status)
    else:
        local.set_terminated(target_id, run.info.status, end_time=run.info.end_time)
    return target_id


def compact(value):
    """Six significant digits suffice for reading; MLflow retains exact values."""
    if isinstance(value, float):
        return float(f"{value:.6g}") if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def normalized_uri(uri):
    """Treat relative/absolute paths to the same SQLite store as one source."""
    if uri.startswith("sqlite:///"):
        return f"sqlite:///{Path(uri.removeprefix('sqlite:///')).resolve()}"
    return uri.rstrip("/")


def export(config: ExportConfig):
    """Read a single run; optionally import full data, always export a small report."""
    if config.max_points is not None and config.max_points < 4:
        raise ValueError("max_points must be at least 4")
    load_tracking_env()
    from mlflow.tracking import MlflowClient

    local_uri = f"sqlite:///{config.local_db.resolve()}"
    source_uri = normalized_uri(config.source_uri or local_uri)
    source = MlflowClient(tracking_uri=source_uri)
    run = source.get_run(config.run_id)
    histories = {key: source.get_metric_history(config.run_id, key)
                 for key in sorted(run.data.metrics)}
    output = config.output or Path("artifacts/exports") / f"{config.run_id}.json"
    with TemporaryDirectory() as temporary:
        listed = {item.path for item in source.list_artifacts(config.run_id) if not item.is_dir}
        artifacts = {name: Path(source.download_artifacts(config.run_id, name, temporary))
                     for name in REPORT_ARTIFACTS if name in listed}
        documents = {Path(name).stem: json.loads(path.read_text(encoding="utf-8"))
                     for name, path in artifacts.items()}
        local_id = None
        if config.import_local and source_uri != local_uri:
            config.local_db.parent.mkdir(parents=True, exist_ok=True)
            local_id = import_run(source, run, histories, artifacts, local_uri, source_uri)

    best_step = documents.get("best", {}).get("step")
    overall, breakdowns = {}, {}
    for key, history in histories.items():
        if "/category/" in key or "/target/" in key:
            points = latest_per_step(history)
            if points:
                breakdowns[key] = {"last": [points[-1].step, points[-1].value]}
                at_best = next((m for m in points if m.step == best_step), None)
                if at_best:
                    breakdowns[key]["at_best"] = at_best.value
        else:
            overall[key] = curve_summary(history, config.max_points)
    # Keep essential provenance and avoid repeating metrics/config in artifacts.
    provenance = documents.get("provenance", {})
    lineage = {name: documents[name] for name in ("data_summary", "cache_summary")
               if name in documents}
    best = documents.get("best", {})
    report = {
        "run": {"id": run.info.run_id, "name": run.data.tags.get("mlflow.runName"),
                "status": run.info.status, "source_uri": source_uri,
                "local_copy_id": local_id, "start_time_ms": run.info.start_time,
                "end_time_ms": run.info.end_time,
                "source_train_run_id": run.data.tags.get("source_train_run_id")},
        "config": documents.get("config", run.data.params),
        "resume_config": documents.get("resume_config"),
        "environment": {key: provenance[key] for key in (
            "git_commit", "parameters", "input_dim", "num_feature_layers", "device",
            "torch", "python", "packages", "learning_rate_schedule") if key in provenance},
        "code_dirty": bool(provenance.get("git_status")), "lineage": lineage,
        "best": {"step": best_step, "sha256": best.get("sha256"),
                 "metrics": {k: v for k, v in best.get("metrics", {}).items() if k.startswith("all/")}},
        "metrics": overall, "breakdowns": breakdowns,
        "notes": {"curve_columns": ["step", "value"], "max_curve_points": config.max_points,
                  "replayed_steps": "latest timestamp wins in report; MLflow keeps full history",
                  "scope": "one run only; linked test evaluations are separate runs",
                  "missing_artifacts": [name for name in REPORT_ARTIFACTS if name not in artifacts]},
    }
    # Evaluation runs have metrics but usually no config/provenance artifacts.
    if "evaluation" in documents:
        report["evaluation"] = {k: v for k, v in documents["evaluation"].items() if k != "metrics"}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(compact(report), ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    return {"report": str(output), "local_run_id": local_id}


if __name__ == "__main__":
    print(export(tyro.cli(ExportConfig)))
