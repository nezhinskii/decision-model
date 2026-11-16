"""Independent cached context depths and consistent train/evaluation selection."""

from dataclasses import replace
import json

import pytest
import torch

from src.backbone.forward import BackboneOutput
from src.features.prepare import prepare
from src.features.cache import FeatureDataset, make_loader
from src.heads import HeadConfig
from test_cache import cache_setup
from test_training import single_thread_accelerate


@pytest.fixture
def multi_cache(cache_setup, monkeypatch):
    config, calls, backbone = cache_setup
    monkeypatch.setattr(backbone, 'describe', lambda self: {
        'revision': 'a' * 40, 'num_hidden_layers': 2,
    })
    def forward(self, batch, layers):
        calls.extend(sample.example.id for sample in batch.samples)
        values = batch.input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        return BackboneOutput({layer: values + 1000 * layer for layer in layers}, {
            'layers': list(layers), 'num_hidden_layers': 2, 'revision': 'a' * 40,
        })
    monkeypatch.setattr(backbone, 'forward', forward)
    # Pooled features need layer 1 only. Layer 2 is exclusively context memory.
    config = replace(config, layers=(1,), context_layers=(1, 2))
    return config, prepare(config)


def test_context_storage_and_selection_are_independent_of_pooled_layers(multi_cache):
    config, root = multi_cache
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['format_version'] == 5
    assert manifest['features']['layers'] == [1]
    assert manifest['features']['context_layers'] == [1, 2]
    raw = torch.load(root / 'train/part-00000.pt', weights_only=True)['records'][0]
    assert raw['context'] is None
    payload = torch.load(root / 'train/part-00000.pt', weights_only=True)
    assert set(payload['context_files']) == {1, 2}
    assert raw['questions'].shape[1] == 1
    for layer in (1, 2):
        records = list(FeatureDataset(root, 'train', include_context=True, context_layer=layer))
        stored = torch.load(root / 'train' / payload['context_files'][layer], weights_only=True)
        torch.testing.assert_close(records[0]['context'], stored['contexts'][0])
        assert 'contexts_by_layer' not in records[0]
        assert records[0]['feature_metadata']['context_layer'] == layer
    without = list(FeatureDataset(root, 'train'))
    assert all(r['context'] is None and 'contexts_by_layer' not in r for r in without)
    assert prepare(config) == root  # Completed cache is reusable.
    with pytest.raises(ValueError, match='different settings'):
        prepare(replace(config, context_layers=(1,)))


def test_automatic_selection_and_explicit_final_override(multi_cache):
    _, root = multi_cache
    auto = HeadConfig(processor='joint')
    first = make_loader(root, 'train', True, 2, 42, head_config=auto)
    assert first.dataset.context_layer == 1
    batch = next(iter(first))
    assert len(batch.features.contexts) == 2
    assert batch.features.metadata['context_layer'] == 1
    final = make_loader(root, 'train', True, 2, 42,
                        head_config=replace(auto, context_layer=-1))
    assert final.dataset.context_layer == 2
    a = batch.features.contexts[0]
    b = next(iter(final)).features.contexts[0]
    torch.testing.assert_close(b - a, torch.full_like(a, 1000))
    with pytest.raises(ValueError, match='not cached'):
        FeatureDataset(root, 'train', include_context=True, context_layer=3)


def test_missing_per_record_context_is_rejected(multi_cache):
    _, root = multi_cache
    path = root / 'train/part-00000.pt'
    payload = torch.load(path, weights_only=True)
    context_path = path.parent / payload['context_files'][1]
    contexts = torch.load(context_path, weights_only=True)
    contexts['contexts'][0] = None
    torch.save(contexts, context_path)
    with pytest.raises(ValueError, match='missing context layer'):
        list(FeatureDataset(root, 'train', include_context=True, context_layer=1))


def test_old_single_context_cache_remains_readable(cache_setup):
    config, _, _ = cache_setup
    root = prepare(config)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    # Original format predates any context_layer metadata.
    manifest['features'].pop('context_layer', None)
    manifest['features']['pooling'].pop('context_layer', None)
    manifest_path.write_text(json.dumps(manifest))
    data = FeatureDataset(root, 'train', include_context=True)
    assert data.context_layer == 2
    assert list(data)[0]['context'] is not None
    with pytest.raises(ValueError, match='not cached'):
        FeatureDataset(root, 'train', include_context=True, context_layer=1)


@pytest.mark.parametrize('mode,indices,expected', [
    ('mean', (0, 1), 24), ('static', (-3, -2), 24),
    ('adaptive', (0, 1), 24), ('last', (1, 0), 16),
])
def test_shared_context_policy_uses_actual_readout(mode, indices, expected):
    head = HeadConfig(readout=mode, layer_indices=indices)
    assert head.context_depth((16, 24, 32), 32) == expected
    assert replace(head, context_layer=32).context_depth((16, 24, 32), 32) == 32
    assert replace(head, context_layer=-1).context_depth((16, 24, 32), 32) == 32


def test_context_layer_persists_through_head_training_and_evaluation(multi_cache, tmp_path):
    from src.training.train_head import TrainConfig, train
    from src.evaluation.evaluate_head import HeadEvaluationConfig, evaluate_head
    from src.evaluation.cached_head import CachedHeadSource

    _, cache = multi_cache
    config = TrainConfig(cache=cache, name='context2', output=tmp_path / 'runs',
        tracking_db=tmp_path / 'tracking.db', cpu=True, mixed_precision='no',
        epochs=1, batch_size=1, log_every=1, validate_every=2, save_every=2,
        head=HeadConfig(processor='joint', dim=8, hidden_dim=16,
            attention_heads=2, score_dim=4, depth=1, num_slots=0, context_layer=2))
    run = train(config)
    saved = json.loads((run / 'config.json').read_text())
    assert saved['head']['context_layer'] == 2
    assert json.loads((run / 'provenance.json').read_text())['context_layer'] == 2
    evaluation = HeadEvaluationConfig(run=run, name='context2_val', split='validation',
        output=config.output, tracking_db=config.tracking_db, cpu=True)
    source = CachedHeadSource(evaluation)
    assert source.loader.dataset.context_layer == 2
    assert source.signature['context_layer'] == 2
    report = evaluate_head(evaluation)
    best = json.loads((run / 'best.json').read_text())
    assert report['all/loss'] == pytest.approx(best['metrics']['all/loss'], abs=1e-6)
    assert evaluate_head(evaluation)['all/accuracy'] == report['all/accuracy']
