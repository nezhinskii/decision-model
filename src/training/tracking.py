"""Local MLflow is authoritative; remote failures never abort training."""

import json
from pathlib import Path
import time
import warnings

import tyro
from dotenv import load_dotenv


def load_tracking_env():
    """Read repository-local credentials; explicit shell variables take precedence."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def client(uri, experiment):
    load_tracking_env()
    from mlflow.tracking import MlflowClient

    result = MlflowClient(tracking_uri=uri)
    existing = result.get_experiment_by_name(experiment)
    artifacts = None
    if uri.startswith("sqlite:///"):
        artifacts = (
            Path(uri.removeprefix("sqlite:///")).resolve().parent / "artifacts"
        ).as_uri()
    return result, (
        existing.experiment_id
        if existing
        else result.create_experiment(experiment, artifact_location=artifacts)
    )


class Tracker:
    """Persist metric history locally; remote synchronization can be retried later."""

    def __init__(self, root, uri, experiment, tags):
        self.root = Path(root)
        self.local, experiment_id = client(uri, experiment)
        path = self.root / "tracking.json"
        if path.exists():
            self.info = json.loads(path.read_text())
        else:
            run = self.local.create_run(experiment_id, tags=tags)
            self.info = {
                "local_uri": uri,
                "experiment": experiment,
                "local_run_id": run.info.run_id,
                "remote_run_id": None,
            }
            write_json(path, self.info)
        self.run_id = self.info["local_run_id"]
        self.local.update_run(self.run_id, status="RUNNING")
        self.local.set_tag(self.run_id, "local_directory", str(self.root.resolve()))

    def log(self, metrics, step):
        from mlflow.entities import Metric

        timestamp = int(time.time() * 1000)
        self.local.log_batch(
            self.run_id,
            metrics=[
                Metric(key, float(value), timestamp, step)
                for key, value in metrics.items()
            ],
        )

    def artifact(self, path):
        # Explicit allowlist at call sites: no checkpoints/cache directories here.
        self.local.log_artifact(self.run_id, str(path))
        if Path(path).name == "config.json":

            def flatten(value, prefix=""):
                for key, item in value.items():
                    name = f"{prefix}.{key}" if prefix else key
                    if isinstance(item, dict):
                        yield from flatten(item, name)
                    else:
                        yield name, json.dumps(item)

            for key, value in flatten(json.loads(Path(path).read_text())):
                self.local.log_param(self.run_id, key, value)

    def status(self, status):
        self.local.set_terminated(self.run_id, status)


def sync(run: Path, remote_uri: str):
    """Mirror local histories, deduplicating by (step, timestamp, value).

    Remote auth uses MLFLOW_TRACKING_USERNAME/PASSWORD environment variables.
    We never copy model files; artifacts are a fixed small allowlist.
    """
    from mlflow.tracking import MlflowClient

    run = Path(run)
    info_path = run / "tracking.json"
    info = json.loads(info_path.read_text())
    local = MlflowClient(tracking_uri=info["local_uri"])
    remote, experiment = client(remote_uri, info["experiment"])
    identity = info["local_run_id"]
    matches = remote.search_runs(
        [experiment], filter_string=f"tags.source_local_run_id = '{identity}'"
    )
    remote_id = (
        matches[0].info.run_id
        if matches
        else remote.create_run(
            experiment,
            tags={"source_local_run_id": identity},
        ).info.run_id
    )
    source = local.get_run(identity)
    for key, value in source.data.tags.items():
        remote.set_tag(remote_id, key, value)
    for key, value in source.data.params.items():
        remote.log_param(remote_id, key, value)
    for key in source.data.metrics:
        known = {
            (m.step, m.timestamp, m.value)
            for m in remote.get_metric_history(remote_id, key)
        }
        missing = [
            m
            for m in local.get_metric_history(identity, key)
            if (m.step, m.timestamp, m.value) not in known
        ]
        for start in range(0, len(missing), 500):
            remote.log_batch(remote_id, metrics=missing[start : start + 500])
    for name in (
        "config.json",
        "resume_config.json",
        "provenance.json",
        "cache_summary.json",
        "data_summary.json",
        "best.json",
        "evaluation.json",
    ):
        if (run / name).exists():
            remote.log_artifact(remote_id, str(run / name))
    if source.info.status == "RUNNING":
        remote.update_run(remote_id, status="RUNNING")
    else:
        remote.set_terminated(
            remote_id, source.info.status, end_time=source.info.end_time
        )
    info["remote_run_id"] = remote_id
    info["remote_uri"] = remote_uri
    write_json(info_path, info)


def try_sync(run, remote_uri):
    if remote_uri:
        try:
            sync(run, remote_uri)
        except Exception as error:
            warnings.warn(f"Remote metrics pending; local data is intact: {error}")


if __name__ == "__main__":
    tyro.cli(sync)
