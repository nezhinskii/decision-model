"""CPU-only train/resume/evaluate and MLflow mirroring using temporary tiny caches."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch
from mlflow.tracking import MlflowClient

from src.heads import HeadConfig
from src.training.objectives import question_metrics
from src.training.train_head import TrainConfig, train
from src.training.evaluate import EvaluateConfig, evaluate_checkpoint
from src.training.tracking import sync
from src.training import tracking
from src.feature_cache.prepare import prepare
from src.feature_cache.dataset import FeatureDataset, collate_features
from test_cache import cache_setup


def test_grouped_metrics_match_reference_values_and_gradients():
    # Unequal option counts, hard/soft targets, ties and large score offsets.
    links = torch.tensor([0, 0, 1, 1, 1, 2])
    targets = torch.tensor([0.5, 0.5, 0., 0.25, 0.75, 1.])
    batch = SimpleNamespace(targets=targets, features=SimpleNamespace(
        option_to_question=links, questions=torch.empty(3, 1, 1)))
    scores = torch.tensor([1000., 1000., -1000., -999., -998., 5.], requires_grad=True)
    actual = question_metrics(scores, batch)
    reference = []
    for question in range(3):
        log_probs = scores[links == question].log_softmax(0)
        target = targets[links == question]
        reference.append(torch.stack((-(target * log_probs).sum(),
            (log_probs.argmax() == target.argmax()).float(),
            (log_probs.exp() - target).square().sum())))
    reference = torch.stack(reference)
    torch.testing.assert_close(actual, reference)
    actual_grad = torch.autograd.grad(actual[:, 0].sum(), scores, retain_graph=True)[0]
    reference_grad = torch.autograd.grad(reference[:, 0].sum(), scores)[0]
    torch.testing.assert_close(actual_grad, reference_grad)


@pytest.fixture(autouse=True)
def single_thread_accelerate(monkeypatch):
    # Accelerate otherwise chooses the host's physical core count on CPU.
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '1')


def test_loss_normalizes_per_question_and_matches_soft_targets(cache_setup):
    config, _, _ = cache_setup
    root = prepare(config)
    batch = collate_features(list(FeatureDataset(root, 'train')))
    logits = torch.tensor([0., 1., 100., 101., -100., -99.], requires_grad=True)
    values = question_metrics(logits, batch)
    torch.testing.assert_close(values[:, 0], values[0, 0].expand(3))
    expected = -(torch.tensor([0.25, 0.75]) * torch.tensor([0., 1.]).log_softmax(0)).sum()
    torch.testing.assert_close(values[:, 0], expected.expand(3))
    values[:, 0].mean().backward()
    assert torch.isfinite(logits.grad).all()


def test_interrupted_training_matches_uninterrupted_and_sync_is_idempotent(cache_setup):
    config, _, _ = cache_setup
    cache = prepare(config)
    cfg = TrainConfig(
        cache=cache, name='resumed', output=config.output.parent / 'runs',
        tracking_db=config.output.parent / 'tracking.db', cpu=True,
        mixed_precision='no',
        head=HeadConfig(processor='mlp', dim=4, hidden_dim=8, score_dim=2, dropout=0.1),
        batch_size=1, accumulation_steps=2, epochs=1, max_steps=1,
    )
    root = train(cfg)
    train(replace(cfg, max_steps=None))  # Existing checkpoint resumes automatically.
    full = train(replace(cfg, name='full', max_steps=None))
    resumed = torch.load(root / 'latest.pt', weights_only=False)
    expected = torch.load(full / 'latest.pt', weights_only=False)
    assert resumed['progress']['step'] == expected['progress']['step'] == 2
    for key, value in expected['model'].items():
        torch.testing.assert_close(resumed['model'][key], value, atol=0, rtol=0)
    result = evaluate_checkpoint(EvaluateConfig(run=root, cpu=True))
    assert (root / 'cache_summary.json').exists()
    assert not (root / 'cache_manifest.json').exists()
    summary = json.loads((root / 'cache_summary.json').read_text())
    assert all('parts' not in split for split in summary['splits'].values())
    assert result['all/loss'] >= 0
    assert 0 <= result['all/accuracy'] <= 1
    remote_uri = f"sqlite:///{config.output.parent / 'remote.db'}"
    sync(root, remote_uri)
    info = json.loads((root / 'tracking.json').read_text())
    remote = MlflowClient(tracking_uri=remote_uri)
    run_id = info['remote_run_id']
    before = len(remote.get_metric_history(run_id, 'train/all/loss'))
    sync(root, remote_uri)
    assert len(remote.get_metric_history(run_id, 'train/all/loss')) == before
    artifacts = {a.path for a in remote.list_artifacts(run_id)}
    assert 'config.json' in artifacts
    assert not any(name.endswith(('.pt', '.safetensors')) for name in artifacts)
    # A remote added after training receives the complete original history.
    local_info = json.loads((root / 'tracking.json').read_text())
    local = MlflowClient(tracking_uri=local_info['local_uri'])
    local_id = local_info['local_run_id']
    local_history = local.get_metric_history(local_id, 'train/all/loss')
    remote_history = remote.get_metric_history(run_id, 'train/all/loss')
    assert [(m.step, m.timestamp, m.value) for m in remote_history] == [
        (m.step, m.timestamp, m.value) for m in local_history]
    assert remote.get_run(run_id).data.params == local.get_run(local_id).data.params
    assert remote.get_run(run_id).info.status == 'FINISHED'
    # A later point and status update go to the same remote run.
    local.update_run(local_id, status='RUNNING')
    local.log_metric(local_id, 'train/new_metric', 7., step=3)
    sync(root, remote_uri)
    assert json.loads((root / 'tracking.json').read_text())['remote_run_id'] == run_id
    assert remote.get_run(run_id).info.status == 'RUNNING'
    assert remote.get_run(run_id).data.metrics['train/new_metric'] == 7.
    # Remote-only runs stay remote: sync neither imports nor removes them.
    experiment_id = remote.get_run(run_id).info.experiment_id
    extra = remote.create_run(experiment_id)
    remote.log_metric(extra.info.run_id, 'remote_only', 1.)
    sync(root, remote_uri)
    assert remote.get_run(extra.info.run_id).data.metrics['remote_only'] == 1.
    assert all('remote_only' not in r.data.metrics for r in
               local.search_runs([local.get_run(local_id).info.experiment_id]))


def test_remote_failure_only_warns(monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise ConnectionError('offline')
    monkeypatch.setattr(tracking, 'sync', unavailable)
    with pytest.warns(UserWarning, match='local data is intact'):
        tracking.try_sync(tmp_path, 'https://example.invalid/project.mlflow')
