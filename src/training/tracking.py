"""Local MLflow is authoritative; remote failures never abort training."""

import json
from pathlib import Path
import time
import warnings
import threading
import hashlib
from tempfile import TemporaryDirectory

import tyro
from dotenv import load_dotenv
from tqdm.auto import tqdm

REPORT_ARTIFACTS = (
    "config.json",
    "resume_config.json",
    "provenance.json",
    "cache_summary.json",
    "data_summary.json",
    "best.json",
    "evaluation.json",
)


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


def history_digest(history):
    """Compact acknowledgment of an exact prefix, including replayed steps."""
    digest = hashlib.sha256()
    for point in history:
        digest.update(f"{point.step}:{point.timestamp}:{point.value!r}\n".encode())
    return digest.hexdigest()


def sync(run: Path, remote_uri: str):
    """Mirror histories in batches; acknowledge prefixes to avoid remote reads.

    Before writing we mark the transfer incomplete. If it fails mid-request,
    the next attempt reconciles actual remote histories instead of trusting
    cursors. Only one sender may operate on a run at a time.
    """
    from mlflow.tracking import MlflowClient
    from mlflow.entities import Param, RunTag

    run = Path(run)
    info_path = run / "tracking.json"
    info = json.loads(info_path.read_text())
    local = MlflowClient(tracking_uri=info["local_uri"])
    identity = info["local_run_id"]
    source = local.get_run(identity)
    # Progress describes work items, not bytes: one artifact may require several
    # HTTP requests. Keep the current operation visible during a slow request.
    with tqdm(desc="DagsHub sync: connecting", total=1, unit="item") as bar:
        remote, experiment = client(remote_uri, info["experiment"])
        if info.get("remote_uri") == remote_uri and info.get("remote_run_id"):
            target = remote.get_run(info["remote_run_id"])
            if target.data.tags.get("source_local_run_id") != identity:
                raise ValueError("Remote run does not match local run identity")
        else:
            matches = remote.search_runs(
                [experiment], filter_string=f"tags.source_local_run_id = '{identity}'"
            )
            target = (
                matches[0]
                if matches
                else remote.create_run(
                    experiment, tags={"source_local_run_id": identity}
                )
            )
        remote_id = target.info.run_id
        # Persist the association early, so a failed upload does not lose it.
        info.update(remote_run_id=remote_id, remote_uri=remote_uri)
        write_json(info_path, info)
        files = [name for name in REPORT_ARTIFACTS if (run / name).exists()]
        bar.reset(total=len(source.data.metrics) + len(files) + 4)
        interrupted = target.data.tags.get("decision_sync.in_progress") == "true"

        bar.set_description("DagsHub sync: parameters/tags")
        remote.set_tag(remote_id, "decision_sync.in_progress", "true")
        params = [
            Param(k, v)
            for k, v in source.data.params.items()
            if target.data.params.get(k) != v
        ]
        tags = [
            RunTag(k, v)
            for k, v in source.data.tags.items()
            if not k.startswith("decision_sync.") and target.data.tags.get(k) != v
        ]
        for start in range(0, max(len(params), len(tags)), 100):
            remote.log_batch(
                remote_id,
                params=params[start : start + 100],
                tags=tags[start : start + 100],
            )
        bar.update()

        missing, acknowledgments = [], []
        for key in source.data.metrics:
            bar.set_description(f"DagsHub sync: history {key}")
            history = sorted(
                local.get_metric_history(identity, key),
                key=lambda m: (m.step, m.timestamp, repr(m.value)),
            )
            cursor_key = f"decision_sync.metric.{key}"
            saved = target.data.tags.get(cursor_key)
            cursor = json.loads(saved) if saved else None
            # Normal case: remote already has the first N exact points. No
            # download of that metric history is needed, even after restarting.
            if (
                not interrupted
                and cursor
                and cursor[0] <= len(history)
                and history_digest(history[: cursor[0]]) == cursor[1]
            ):
                pending = history[cursor[0] :]
            elif key not in target.data.metrics:
                pending = history
            else:
                # Old runs, replay inserted into an earlier step, or a previous
                # interrupted upload need full reconciliation once.
                known = {
                    (m.step, m.timestamp, repr(m.value))
                    for m in remote.get_metric_history(remote_id, key)
                }
                pending = [
                    m
                    for m in history
                    if (m.step, m.timestamp, repr(m.value)) not in known
                ]
            missing.extend(pending)
            acknowledgments.append(
                RunTag(
                    cursor_key,
                    json.dumps(
                        [len(history), history_digest(history)], separators=(",", ":")
                    ),
                )
            )
            bar.update()

        bar.set_description(f"DagsHub sync: uploading {len(missing)} metric points")
        for start in range(0, len(missing), 500):
            remote.log_batch(remote_id, metrics=missing[start : start + 500])
            bar.set_postfix(points=f"{min(start + 500, len(missing))}/{len(missing)}")
        for start in range(0, len(acknowledgments), 100):
            remote.log_batch(remote_id, tags=acknowledgments[start : start + 100])
        bar.update()

        # Snapshot bytes before hashing/uploading: best.json may change while
        # training continues. The acknowledgment must refer to the uploaded bytes.
        with TemporaryDirectory() as temporary:
            for name in files:
                data = (run / name).read_bytes()
                checksum = hashlib.sha256(data).hexdigest()
                tag = f"decision_sync.artifact.{name}"
                if target.data.tags.get(tag) != checksum:
                    bar.set_description(
                        f"DagsHub sync: uploading {name} ({len(data)} B)"
                    )
                    path = Path(temporary) / name
                    path.write_bytes(data)
                    remote.log_artifact(remote_id, str(path))
                    remote.set_tag(remote_id, tag, checksum)
                bar.update()
        bar.set_description("DagsHub sync: final status")
        if source.info.status == "RUNNING":
            remote.update_run(remote_id, status="RUNNING")
        else:
            remote.set_terminated(
                remote_id, source.info.status, end_time=source.info.end_time
            )
        bar.update()
        remote.set_tag(remote_id, "decision_sync.in_progress", "false")
        bar.update()
        bar.set_description(f"DagsHub sync: complete ({source.info.status})")


def try_sync(run, remote_uri):
    if remote_uri:
        try:
            sync(run, remote_uri)
        except Exception as error:
            warnings.warn(f"Remote metrics pending; local data is intact: {error}")


class BackgroundSync:
    """One sender per run; requests coalesce while a transfer is in progress.

    Local MLflow remains the durable queue. Failed transfers retry in the
    background; normal shutdown waits for the final transfer to finish.
    """

    def __init__(self, run, remote_uri, retry_seconds=60):
        self.run = run
        self.remote_uri = remote_uri
        self.retry_seconds = retry_seconds
        self.condition = threading.Condition()
        self.pending = False
        self.closing = False
        self.thread = None
        if remote_uri:
            self.thread = threading.Thread(
                target=self._work, daemon=True, name="mlflow-sync"
            )
            self.thread.start()

    def request(self):
        """Signal new local data without waiting for network activity."""
        if self.thread is not None:
            with self.condition:
                self.pending = True
                self.condition.notify()

    def _work(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending or self.closing)
                if not self.pending:
                    return
                self.pending = False
            try:
                sync(self.run, self.remote_uri)
            except Exception as error:
                warnings.warn(f"Remote metrics pending; local data is intact: {error}")
                with self.condition:
                    if self.closing:
                        return
                    self.pending = True
                    # Condition.wait releases the lock: training never waits on
                    # either the transfer or this retry delay.
                    self.condition.wait(timeout=self.retry_seconds)

    def close(self, timeout=None):
        """Wait for the final snapshot by default, including an in-flight transfer.

        An explicit timeout is available to callers that deliberately want a
        bounded wait; training/evaluation use the default complete wait.
        """
        if self.thread is None:
            return
        tqdm.write("Local results saved. Waiting for final DagsHub synchronization...")
        with self.condition:
            self.pending = True
            self.closing = True
            self.condition.notify()
        # A heartbeat remains visible even if one HTTP request is waiting on
        # MLflow's internal retries. This does not impose a transfer deadline.
        started = time.monotonic()
        while self.thread.is_alive():
            remaining = (
                None if timeout is None else timeout - (time.monotonic() - started)
            )
            if remaining is not None and remaining <= 0:
                break
            self.thread.join(
                timeout=min(15, remaining) if remaining is not None else 15
            )
            if self.thread.is_alive():
                tqdm.write(
                    f"DagsHub sync still running ({time.monotonic() - started:.0f}s); "
                    "local results are safe."
                )
        if self.thread.is_alive():
            warnings.warn(
                "Remote sync still pending; retry with src.training.tracking. "
                "All results are saved locally."
            )


if __name__ == "__main__":
    tyro.cli(sync)
