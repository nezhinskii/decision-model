"""Storage conversion preserves identity/order and avoids unwanted context I/O."""

from dataclasses import replace
import json

import pytest
import torch

from src.features.cache import FeatureDataset
from src.features.split_contexts import SplitContextConfig, split_contexts
from src.features import storage
from test_cache import cache_setup
from test_context_cache import multi_cache
from test_training import single_thread_accelerate


def monolithic(root):
    """Rebuild the previous storage layout for migration tests, without changing identity."""
    manifest = json.loads((root / 'manifest.json').read_text())
    for split in manifest['splits'].values():
        for part in split['parts']:
            path = root / part['file']
            payload = storage.load_shard(path)
            files = payload.pop('context_files')
            for record in payload['records']:
                record['contexts_by_layer'] = {}
            for layer, name in files.items():
                contexts = storage.load_shard(path.parent / name)['contexts']
                for record, tensor in zip(payload['records'], contexts):
                    record['contexts_by_layer'][layer] = tensor
            torch.save(payload, path)


def snapshot(root, layer=None):
    return list(FeatureDataset(root, 'train', include_context=layer is not None,
                              context_layer=layer, seed=17, shuffle=True))


def assert_records_equal(a, b):
    assert len(a) == len(b)
    for before, after in zip(a, b):
        assert before.keys() == after.keys()
        for key in before:
            if isinstance(before[key], torch.Tensor):
                torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
            else:
                assert before[key] == after[key]


def test_conversion_preserves_manifest_values_shuffle_and_is_idempotent(multi_cache):
    _, root = multi_cache
    monolithic(root)
    manifest = (root / 'manifest.json').read_bytes()
    before = {layer: snapshot(root, layer) for layer in (None, 1, 2)}
    split_contexts(SplitContextConfig(root))
    assert (root / 'manifest.json').read_bytes() == manifest
    for layer in before:
        assert_records_equal(before[layer], snapshot(root, layer))
    path = root / 'train/part-00000.pt'
    payload = storage.load_shard(path)
    assert set(payload['context_files']) == {1, 2}
    assert all(r['context'] is None and 'contexts_by_layer' not in r
               for r in payload['records'])
    primary = path.read_bytes()
    split_contexts(SplitContextConfig(root))
    assert path.read_bytes() == primary


def test_loader_reads_no_context_for_identity_and_one_depth_for_cross(multi_cache, monkeypatch):
    _, root = multi_cache
    original, paths = torch.load, []
    def tracked(path, **kwargs):
        paths.append(str(path))
        return original(path, **kwargs)
    monkeypatch.setattr(torch, 'load', tracked)
    rows = snapshot(root)
    assert rows and not any('.context-' in p for p in paths)
    paths.clear()
    rows = snapshot(root, 1)
    context_paths = [p for p in paths if '.context-' in p]
    assert context_paths and all('.context-1.pt' in p for p in context_paths)
    assert not any('.context-2.pt' in p for p in paths)


def test_failed_publish_keeps_original_and_conversion_can_resume(multi_cache, monkeypatch):
    _, root = multi_cache
    monolithic(root)
    path = root / 'train/part-00000.pt'
    before = path.read_bytes()
    save = storage._save_atomic
    def interrupted(destination, payload):
        if destination == path:
            raise RuntimeError('interrupted before feature publish')
        return save(destination, payload)
    monkeypatch.setattr(storage, '_save_atomic', interrupted)
    with pytest.raises(RuntimeError, match='interrupted'):
        split_contexts(SplitContextConfig(root))
    assert path.read_bytes() == before
    assert snapshot(root, 1)
    monkeypatch.setattr(storage, '_save_atomic', save)
    split_contexts(SplitContextConfig(root))
    assert 'context_files' in storage.load_shard(path)
    assert snapshot(root, 2)


def test_context_file_order_is_validated(multi_cache):
    _, root = multi_cache
    path = root / 'train/part-00000.pt'
    payload = storage.load_shard(path)
    context_path = path.parent / payload['context_files'][1]
    contexts = storage.load_shard(context_path)
    contexts['context_ids'][0] = 'wrong-context'
    torch.save(contexts, context_path)
    with pytest.raises(ValueError, match='not aligned'):
        snapshot(root, 1)
    assert snapshot(root)  # A head without context never opens that file.


def test_empty_skip_only_shards_do_not_require_context_files(multi_cache):
    _, root = multi_cache
    path = root / 'train/part-00000.pt'
    payload = storage.load_shard(path)
    payload['records'] = []
    payload['context_files'] = {}
    torch.save(payload, path)
    # Remaining shards still contain valid contexts; the empty shard is progress only.
    assert snapshot(root, 1)


def test_training_resume_after_conversion_matches_continuous(multi_cache, tmp_path):
    from src.heads import HeadConfig
    from src.training.train_head import TrainConfig, train

    _, root = multi_cache
    monolithic(root)
    config = TrainConfig(cache=root, name='resumed', output=tmp_path / 'runs',
        tracking_db=tmp_path / 'tracking.db', cpu=True, mixed_precision='no',
        epochs=2, batch_size=1, log_every=1, validate_every=2, save_every=1,
        head=HeadConfig(processor='joint', dim=8, hidden_dim=16,
            attention_heads=2, score_dim=4, depth=1, num_slots=0, context_layer=1))
    # Stop on a scheduled validation boundary so both executions perform the
    # same checks. Stopping at step 1 adds a final validation absent from the
    # uninterrupted run and can legitimately change early-stopping history.
    resumed = train(replace(config, max_steps=2))
    split_contexts(SplitContextConfig(root))
    train(config)
    continuous = train(replace(config, name='continuous'))
    a = torch.load(resumed / 'latest.pt', weights_only=False)
    b = torch.load(continuous / 'latest.pt', weights_only=False)
    assert a['progress'] == b['progress']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
