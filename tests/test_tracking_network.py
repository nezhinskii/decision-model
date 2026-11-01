"""Count remote requests and simulate a lost response using real temporary MLflow."""

from collections import Counter
import hashlib
import json

from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
import pytest

from src.training import tracking


def test_sync_batches_skips_unchanged_data_and_recovers_lost_response(tmp_path, monkeypatch, capsys):
    root = tmp_path / 'run'
    root.mkdir()
    local_uri = f"sqlite:///{tmp_path / 'local.db'}"
    remote_uri = f"sqlite:///{tmp_path / 'remote.db'}"
    tracker = tracking.Tracker(root, local_uri, 'research', {'mlflow.runName': 'example'})
    config = root / 'config.json'
    tracking.write_json(config, {f'parameter{i}': i for i in range(40)})
    tracker.artifact(config)
    best = root / 'best.json'
    tracking.write_json(best, {'step': 10})
    points = [Metric(f'metric/{key}', float(step), step * 100, step)
              for key in range(20) for step in range(30)]
    tracker.local.log_batch(tracker.run_id, metrics=points)
    counts = Counter()
    real_client = tracking.client

    class Remote:
        lose_response = False

        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __getattr__(self, name):
            function = getattr(self.wrapped, name)

            def counted(*args, **kwargs):
                counts[name] += 1
                result = function(*args, **kwargs)
                if name == 'log_batch' and kwargs.get('metrics'):
                    counts['metric_batches'] += 1
                    assert len(kwargs['metrics']) <= 500
                    if Remote.lose_response:
                        Remote.lose_response = False
                        raise ConnectionError('server stored metrics but response was lost')
                return result

            return counted

    def client(uri, experiment):
        result, experiment_id = real_client(uri, experiment)
        return (Remote(result) if uri == remote_uri else result), experiment_id

    monkeypatch.setattr(tracking, 'client', client)
    tracking.sync(root, remote_uri)
    assert counts['metric_batches'] == 2  # 600 points across 20 metrics.
    assert counts['get_metric_history'] == 0
    assert counts['log_param'] == 0  # Forty parameters sent in one batch.
    assert counts['log_artifact'] == 2
    remote_id = json.loads((root / 'tracking.json').read_text())['remote_run_id']
    remote = MlflowClient(tracking_uri=remote_uri)

    counts.clear()
    tracking.sync(root, remote_uri)
    assert counts['get_metric_history'] == counts['metric_batches'] == 0
    assert counts['log_artifact'] == 0
    assert counts['search_runs'] == 0  # Persisted association avoids searching.

    tracker.local.log_metric(tracker.run_id, 'metric/0', 30., timestamp=3000, step=30)
    counts.clear()
    Remote.lose_response = True
    with pytest.raises(ConnectionError, match='response was lost'):
        tracking.sync(root, remote_uri)
    assert remote.get_run(remote_id).data.tags['decision_sync.in_progress'] == 'true'
    tracking.sync(root, remote_uri)
    assert len(remote.get_metric_history(remote_id, 'metric/0')) == 31
    assert remote.get_run(remote_id).data.tags['decision_sync.in_progress'] == 'false'
    assert counts['get_metric_history'] > 0  # Reconcile ambiguous failed writes.

    # Replayed earlier steps invalidate the prefix rather than losing the point.
    tracker.local.log_metric(tracker.run_id, 'metric/0', 99., timestamp=4000, step=5)
    counts.clear()
    tracking.write_json(best, {'step': 30})
    tracker.status('FINISHED')
    tracking.sync(root, remote_uri)
    assert counts['get_metric_history'] == 1
    assert counts['log_artifact'] == 1  # Only best.json changed.
    assert len(remote.get_metric_history(remote_id, 'metric/0')) == 32
    copied = remote.get_run(remote_id)
    assert copied.info.status == 'FINISHED'
    assert copied.data.tags['decision_sync.artifact.best.json'] == hashlib.sha256(best.read_bytes()).hexdigest()
    assert 'DagsHub sync:' in capsys.readouterr().err
