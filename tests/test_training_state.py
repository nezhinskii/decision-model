"""Fast CPU checks of optimizer math, persistence, restart and run lifecycle."""

from copy import deepcopy
from dataclasses import replace
import importlib
import json
import random
from types import SimpleNamespace

from accelerate import Accelerator
import numpy as np
import pytest
import torch

from src.feature_cache.dataset import collate_features, FeatureDataset
from src.feature_cache.prepare import prepare
from src.heads import HeadConfig, build_head
from src.training import checkpoints
from src.training.loop import evaluate, train_step
from src.training.objectives import MetricTotals, question_metrics
from src.training.provenance import cache_summary, check_cache, data_summary
from src.training.tracking import write_json
from src.training.train_head import TrainConfig, train
from src.training.evaluate import EvaluateConfig, evaluate_checkpoint
from test_cache import cache_setup
from test_training import single_thread_accelerate

training = importlib.import_module('src.training.train_head')
evaluation = importlib.import_module('src.training.evaluate')


@pytest.fixture
def run_setup(cache_setup, monkeypatch):
    """Exercise the real trainer; only replace the external tracking boundary."""
    source, _, _ = cache_setup
    cache = prepare(source)
    runs = {}

    class MemoryTracker:
        def __init__(self, root, uri, experiment, tags):
            self.state = runs.setdefault(str(root), {'logs': [], 'statuses': [], 'artifacts': []})
            self.state['statuses'].append('RUNNING')
            self.run_id = str(root)
            write_json(root / 'tracking.json', {'local_uri': uri, 'local_run_id': self.run_id})

        def log(self, metrics, step):
            self.state['logs'].append((step, dict(metrics)))

        def artifact(self, path):
            self.state['artifacts'].append(path.name)

        def status(self, status):
            self.state['statuses'].append(status)

    monkeypatch.setattr(training, 'Tracker', MemoryTracker)
    monkeypatch.setattr(evaluation, 'Tracker', MemoryTracker)
    config = TrainConfig(cache=cache, name='tiny', output=source.output.parent / 'runs',
                         tracking_db=source.output.parent / 'tracking.db', cpu=True,
                         mixed_precision='no', batch_size=1, epochs=1,
                         head=HeadConfig(processor='mlp', dim=4, hidden_dim=8,
                                         score_dim=2, dropout=0.1))
    return config, runs


def test_accumulation_matches_large_batch_with_unequal_question_counts(run_setup):
    config, _ = run_setup
    records = list(FeatureDataset(config.cache, 'train'))
    # The fake backbone repeats each scalar across D; use nondegenerate vectors
    # here so near-zero LayerNorm gradients do not dominate Adam rounding.
    for record in records:
        record['questions'] = torch.randn_like(record['questions'])
        record['options'] = torch.randn_like(record['options'])
    # Unequal microbatches ensure normalization is by questions, not batches.
    microbatches = [collate_features(records[:1]), collate_features(records[1:])]
    whole = collate_features(records)
    head = build_head(4, replace(config.head, dropout=0.), 1)
    other = deepcopy(head)
    accelerator = Accelerator(cpu=True, mixed_precision='no')
    # SGD isolates accumulation/clipping math. Adam can amplify rounding in
    # theoretically zero gradients (e.g. shared option-score shifts).
    opt = torch.optim.SGD(head.parameters(), lr=.01)
    other_opt = torch.optim.SGD(other.parameters(), lr=.01)
    # Also cover accumulation over more than one optimizer update and clipping.
    for _ in range(2):
        sums, count = train_step(head, microbatches, opt, accelerator, .1)
        expected, expected_count = train_step(other, [whole], other_opt, accelerator, .1)
        assert count == expected_count == 3
        torch.testing.assert_close(sums, expected)
        for actual, reference in zip(head.parameters(), other.parameters()):
            torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
            assert actual.grad is None


def test_validation_is_batch_size_independent_and_does_not_update_weights(run_setup):
    config, _ = run_setup
    records = list(FeatureDataset(config.cache, 'validation'))
    head = build_head(4, config.head, 1)
    before = deepcopy(head.state_dict())
    accelerator = Accelerator(cpu=True, mixed_precision='no')
    small = evaluate(head, [collate_features([r]) for r in records], accelerator)
    large = evaluate(head, [collate_features(records)], accelerator)
    assert small == pytest.approx(large, abs=1e-6)
    assert head.training
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert all(p.grad is None for p in head.parameters())


def test_metric_totals_weight_questions_and_group_labels():
    totals = MetricTotals()
    totals.add(torch.tensor([[1., 1., .1], [3., 0., .3]]),
               SimpleNamespace(categories=['a', 'b'], target_kinds=['hard', 'soft']))
    totals.add(torch.tensor([[8., 1., .8]]),
               SimpleNamespace(categories=['a'], target_kinds=['soft']))
    result = totals.result()
    assert result['all/loss'] == 4.
    assert result['all/accuracy'] == pytest.approx(2 / 3)
    assert result['category/a/loss'] == 4.5
    assert result['target/soft/loss'] == 5.5


def test_checkpoint_roundtrip_restores_optimizer_rng_and_progress(tmp_path):
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 3)).sum().backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    accelerator = Accelerator(cpu=True, mixed_precision='no')
    progress = {'epoch': 2, 'batch': 3, 'step': 7, 'best_loss': .4}
    path = tmp_path / 'latest.pt'
    before = deepcopy(model.state_dict())
    checkpoints.save_latest(path, model, opt, accelerator, progress)
    expected_random = (random.random(), np.random.random(), torch.rand(3))
    new_model = torch.nn.Linear(3, 2)
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=.5)
    assert checkpoints.load_latest(path, new_model, new_opt, accelerator) == progress
    assert random.random() == expected_random[0]
    assert np.random.random() == expected_random[1]
    torch.testing.assert_close(torch.rand(3), expected_random[2], atol=0, rtol=0)
    assert new_opt.param_groups[0]['lr'] == opt.param_groups[0]['lr']
    for key in before:
        torch.testing.assert_close(new_model.state_dict()[key], before[key])
    # Equal next updates prove the optimizer moments were restored too.
    for current, optimizer in ((model, opt), (new_model, new_opt)):
        current(torch.ones(1, 3)).square().sum().backward()
        optimizer.step()
    for a, b in zip(model.parameters(), new_model.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('kind', ['best', 'latest'])
def test_checkpoint_replace_and_failed_write_keeps_previous(tmp_path, monkeypatch, kind):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    accelerator = Accelerator(cpu=True, mixed_precision='no')
    path = tmp_path / ('best.safetensors' if kind == 'best' else 'latest.pt')

    def save():
        if kind == 'best':
            return checkpoints.save_model(tmp_path, model)
        checkpoints.save_latest(path, model, optimizer, accelerator, {'step': 1})
        return path

    assert save() == path
    first_hash = checkpoints.weight_hash(path)
    with torch.no_grad():
        model.weight.add_(1)
    save()
    assert checkpoints.weight_hash(path) != first_hash
    assert list(tmp_path.iterdir()) == [path]
    previous = path.read_bytes()

    def interrupted(*args, **kwargs):
        raise OSError('simulated disk failure')

    if kind == 'best':
        monkeypatch.setattr(checkpoints, 'save_file', interrupted)
    else:
        monkeypatch.setattr(checkpoints.torch, 'save', interrupted)
    with pytest.raises(OSError, match='disk failure'):
        save()
    assert path.read_bytes() == previous


@pytest.mark.parametrize('processor', ['identity', 'mlp', 'joint'])
def test_each_head_can_train_and_evaluate_from_cache(run_setup, processor):
    config, runs = run_setup
    head = replace(config.head, processor=processor, attention_heads=1, num_slots=2)
    root = train(replace(config, head=head))
    result = evaluate_checkpoint(EvaluateConfig(run=root, cpu=True))
    assert np.isfinite(result['all/loss'])
    assert list((root / 'checkpoints').glob('*.safetensors')) == [root / 'checkpoints/best.safetensors']
    assert runs[str(root)]['statuses'][-1] == 'FINISHED'
    report = next((root / 'evaluations').glob('*/evaluation.json'))
    assert json.loads(report.read_text())['source_train_run_id'] == str(root)


def test_resume_across_epoch_boundary_matches_continuous_training(run_setup):
    config, _ = run_setup
    root = train(replace(config, epochs=1))
    train(replace(config, epochs=3))
    full = train(replace(config, name='full', epochs=3))
    resumed = torch.load(root / 'latest.pt', weights_only=False)
    expected = torch.load(full / 'latest.pt', weights_only=False)
    assert resumed['progress'] == expected['progress']
    for key in expected['model']:
        torch.testing.assert_close(resumed['model'][key], expected['model'][key], atol=0, rtol=0)


@pytest.mark.parametrize('error', [RuntimeError('failed step'), KeyboardInterrupt()])
def test_interruption_keeps_safe_checkpoint_and_sets_status(run_setup, monkeypatch, error):
    config, runs = run_setup
    original = training.train_step
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            # Interrupt after consuming dropout RNG and accumulating gradients.
            model, batches = args[:2]
            question_metrics(model(batches[0].features), batches[0])[:, 0].mean().backward()
            raise error
        return original(*args, **kwargs)

    monkeypatch.setattr(training, 'train_step', interrupt)
    with pytest.raises(type(error)):
        train(replace(config, save_every=1))
    root = config.output / config.name
    assert torch.load(root / 'latest.pt', weights_only=False)['progress']['step'] == 1
    assert runs[str(root)]['statuses'][-1] == ('KILLED' if isinstance(error, KeyboardInterrupt) else 'FAILED')
    monkeypatch.setattr(training, 'train_step', original)
    train(replace(config, save_every=1))
    full = train(replace(config, name='full', save_every=1))
    restored = torch.load(root / 'latest.pt', weights_only=False)
    expected = torch.load(full / 'latest.pt', weights_only=False)
    for key in expected['model']:
        torch.testing.assert_close(restored['model'][key], expected['model'][key], atol=0, rtol=0)


def test_logging_intervals_partial_flush_and_best_selection(run_setup, monkeypatch):
    config, runs = run_setup
    # Three steps: validation after each, with an improvement then regression.
    losses = iter([3., 2., 4., 5.])
    monkeypatch.setattr(training, 'evaluate', lambda *args: {'all/loss': next(losses)})
    root = train(replace(config, log_every=2, validate_every=1))
    best = json.loads((root / 'best.json').read_text())
    assert best['step'] == 2 and best['metrics']['all/loss'] == 2.
    assert checkpoints.weight_hash(root / best['weights']) == best['sha256']
    assert len(list((root / 'checkpoints').iterdir())) == 1
    steps = [step for step, metrics in runs[str(root)]['logs'] if 'train/all/loss' in metrics]
    assert steps == [1, 2, 3]  # Validation flushes even short intervals.


def test_train_logging_uses_interval_question_weighted_sums(run_setup, monkeypatch):
    config, runs = run_setup
    sums = iter([(torch.tensor([2., 1., .2]), 1),
                 (torch.tensor([12., 1., .6]), 3),
                 (torch.tensor([5., 0., .4]), 1)])
    monkeypatch.setattr(training, 'train_step', lambda *args: next(sums))
    root = train(replace(config, log_every=2))
    points = [(step, metrics) for step, metrics in runs[str(root)]['logs']
              if 'train/all/loss' in metrics]
    assert [step for step, _ in points] == [2, 3]
    assert points[0][1]['train/all/loss'] == 3.5
    assert points[0][1]['train/all/accuracy'] == .5
    assert points[1][1]['train/all/loss'] == 5.


def test_resume_guards_and_evaluation_reject_changed_files(run_setup):
    config, runs = run_setup
    root = train(config)
    with pytest.raises(ValueError, match='Run exists'):
        train(replace(config, resume=False))
    with pytest.raises(ValueError, match='configuration changed'):
        train(replace(config, learning_rate=.5))
    manifest_path = config.cache / 'manifest.json'
    original = manifest_path.read_text()
    manifest = json.loads(original)
    manifest['model_revision'] = 'changed'
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match='Cache manifest changed'):
        train(config)
    assert runs[str(root)]['statuses'][-1] == 'FINISHED'
    with pytest.raises(ValueError, match='Cache manifest changed'):
        evaluate_checkpoint(EvaluateConfig(run=root, cpu=True))
    manifest_path.write_text(original)
    best = json.loads((root / 'best.json').read_text())
    with (root / best['weights']).open('ab') as stream:
        stream.write(b'corrupted')
    with pytest.raises(ValueError, match='checksum mismatch'):
        evaluate_checkpoint(EvaluateConfig(run=root, cpu=True))


def test_nonempty_run_without_checkpoint_is_not_overwritten(run_setup):
    config, _ = run_setup
    root = config.output / config.name
    root.mkdir(parents=True)
    (root / 'important.json').write_text('{}')
    with pytest.raises(ValueError, match='no resumable'):
        train(config)
    assert (root / 'important.json').read_text() == '{}'


@pytest.mark.parametrize('field', ['epochs', 'batch_size', 'accumulation_steps',
                                  'log_every', 'validate_every', 'save_every'])
def test_zero_training_intervals_are_rejected(tmp_path, field):
    config = TrainConfig(cache=tmp_path, name='run', output=tmp_path / 'runs')
    with pytest.raises(ValueError, match='positive'):
        train(replace(config, **{field: 0}))
    assert not config.output.exists()


def test_old_best_filename_is_removed_after_improvement(run_setup, monkeypatch):
    config, _ = run_setup
    root = train(config)
    best = json.loads((root / 'best.json').read_text())
    original = root / best['weights']
    legacy = original.with_name('legacy-uuid.safetensors')
    original.rename(legacy)
    best['weights'] = str(legacy.relative_to(root))
    write_json(root / 'best.json', best)
    monkeypatch.setattr(training, 'evaluate', lambda *args: {'all/loss': -1.})
    train(replace(config, epochs=2))
    assert not legacy.exists()
    best = json.loads((root / 'best.json').read_text())
    assert best['weights'] == 'checkpoints/best.safetensors'
    assert checkpoints.weight_hash(root / best['weights']) == best['sha256']


def test_weight_loading_rejects_incompatible_architecture(tmp_path):
    path = checkpoints.save_model(tmp_path, torch.nn.Linear(2, 1))
    with pytest.raises(RuntimeError, match='size mismatch'):
        checkpoints.load_model(path, torch.nn.Linear(3, 1))


def test_nonfinite_gradient_does_not_update_parameters(run_setup):
    config, _ = run_setup
    batch = collate_features(list(FeatureDataset(config.cache, 'train')))
    head = build_head(4, config.head, 1)
    before = deepcopy(head.state_dict())
    handle = next(head.parameters()).register_hook(lambda grad: grad * float('nan'))
    with pytest.raises(RuntimeError, match='non-finite'):
        train_step(head, [batch], torch.optim.AdamW(head.parameters()),
                   Accelerator(cpu=True, mixed_precision='no'), 1.)
    handle.remove()
    for key, value in head.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_gradient_clipping_is_applied_before_optimizer_step(run_setup):
    config, _ = run_setup
    batch = collate_features(list(FeatureDataset(config.cache, 'train')))
    head = build_head(4, config.head, 1)
    # Force a large gradient regardless of the synthetic feature scale.
    handle = next(head.parameters()).register_hook(lambda grad: grad + 1000.)

    class InspectSGD(torch.optim.SGD):
        def step(self, closure=None):
            norm = torch.stack([p.grad.norm() for group in self.param_groups
                                for p in group['params'] if p.grad is not None]).norm()
            assert .049 < norm.item() <= .050001
            return super().step(closure)

    train_step(head, [batch], InspectSGD(head.parameters(), lr=.1),
               Accelerator(cpu=True, mixed_precision='no'), .05)
    handle.remove()


def test_evaluation_failure_marks_its_own_run_failed(run_setup, monkeypatch):
    config, runs = run_setup
    root = train(config)

    def fail(*args):
        raise RuntimeError('evaluation failed')

    monkeypatch.setattr(evaluation, 'evaluate', fail)
    with pytest.raises(RuntimeError, match='evaluation failed'):
        evaluate_checkpoint(EvaluateConfig(run=root, cpu=True))
    assert runs[str(root)]['statuses'][-1] == 'FINISHED'
    evaluation_runs = [state for path, state in runs.items() if '/evaluations/' in path]
    assert len(evaluation_runs) == 1
    assert evaluation_runs[0]['statuses'][-1] == 'FAILED'


def test_provenance_keeps_identity_and_counts_but_omits_inventories(tmp_path):
    manifest = {'features': {'layers': [2]}, 'source_files': {'train': {'sha256': 'abc'}},
                'splits': {'train': {'contexts': 4, 'questions': 7, 'parts': ['large inventory']}}}
    write_json(tmp_path / 'cache_summary.json', cache_summary(tmp_path, manifest))
    check_cache(tmp_path, tmp_path, manifest)
    assert 'parts' not in cache_summary(tmp_path, manifest)['splits']['train']
    # Old full-manifest runs remain readable.
    (tmp_path / 'cache_summary.json').unlink()
    write_json(tmp_path / 'cache_manifest.json', manifest)
    check_cache(tmp_path, tmp_path, manifest)
    with pytest.raises(ValueError, match='changed'):
        check_cache(tmp_path, tmp_path, {**manifest, 'features': {'layers': [1]}})
    source = {'source': {'dataset': 'demo', 'revision': 'fixed'}, 'seed': 3,
              'length_filter': {'max_tokens': 2048, 'by_category': {'large': []}},
              'splits': {'train': {'sha256': 'abc', 'statistics': {
                  'questions': 7, 'categories': {'a': 7}, 'questions_per_context': [1, 2]}}}}
    write_json(tmp_path / 'manifest.json', source)
    summary = data_summary(tmp_path)
    assert summary['source'] == source['source']
    assert summary['splits']['train']['questions'] == 7
    assert summary['splits']['train']['categories'] == {'a': 7}
    assert 'by_category' not in summary['length_filter']
    assert 'questions_per_context' not in summary['splits']['train']
