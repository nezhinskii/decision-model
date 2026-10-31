"""Cache lifecycle checks with actual Parquet/.pt files and a tiny fake backbone."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import src.feature_cache.prepare as preparation
import src.feature_cache.dataset as reading
from src.backbone.features import PoolingConfig
from src.backbone.inputs import InputConfig
from src.backbone.model import BackboneConfig, BackboneOutput
from src.data.schema import arrow_schema
from conftest import make_example


@pytest.fixture
def cache_setup(tmp_path, monkeypatch, tokenizer):
    source = tmp_path / 'data'
    source.mkdir()
    source_manifest = {'splits': {}}
    for split in ('train', 'validation', 'test'):
        examples = [make_example(f'{split}/{i}', 'x' * (500 if i == 1 else i + 1), 1) for i in range(4)]
        path = source / f'{split}.parquet'
        pq.write_table(pa.Table.from_pylist([asdict(e) for e in examples], schema=arrow_schema()), path)
        source_manifest['splits'][split] = {'file': path.name, 'sha256': preparation.file_hash(path)}
    (source / 'manifest.json').write_text(json.dumps(source_manifest))
    calls = []

    class FakeBackbone:
        def __init__(self, config):
            self.tokenizer = tokenizer

        def describe(self):
            return {'revision': 'a' * 40}

        def forward(self, batch, layers):
            calls.extend(sample.example.id for sample in batch.samples)
            states = batch.input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
            return BackboneOutput({2: states}, {'num_hidden_layers': 2, 'revision': 'a' * 40})

    monkeypatch.setattr(preparation, 'FrozenBackbone', FakeBackbone)
    config = preparation.PrepareConfig(
        name='tiny', backbone=BackboneConfig(model_id='fake'), dataset=source,
        output=tmp_path / 'features', inputs=InputConfig(max_packed_tokens=128),
        pooling=PoolingConfig(include_context=True), shard_size=2,
    )
    return config, calls, FakeBackbone


@pytest.mark.parametrize('interruption', ['forward', 'after_shard_rename'])
def test_resume_keeps_shards_and_counts_skipped_source_rows(cache_setup, monkeypatch, interruption):
    config, calls, backbone = cache_setup
    original_forward = backbone.forward
    original_save = preparation.save_manifest
    if interruption == 'forward':
        def interrupted_forward(self, batch, layers):
            if batch.samples[0].example.id == 'train/2':
                raise RuntimeError('simulated interruption')
            return original_forward(self, batch, layers)
        monkeypatch.setattr(backbone, 'forward', interrupted_forward)
    else:
        def interrupted_save(path, manifest):
            if manifest['splits']['train']['parts']:
                raise RuntimeError('simulated interruption')
            original_save(path, manifest)
        monkeypatch.setattr(preparation, 'save_manifest', interrupted_save)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        preparation.prepare(config)
    root = config.output / config.name
    first_part = root / 'train/part-00000.pt'
    first_bytes = first_part.read_bytes()
    with pytest.raises(ValueError, match='incomplete'):
        reading.FeatureDataset(root, 'train')
    monkeypatch.setattr(backbone, 'forward', original_forward)
    monkeypatch.setattr(preparation, 'save_manifest', original_save)
    # Changing batch/shard sizes must not shift the source restart cursor.
    preparation.prepare(replace(config, batch_size=2, shard_size=3))
    assert first_part.read_bytes() == first_bytes
    assert calls == [f'{split}/{i}' for split in ('train', 'validation', 'test') for i in (0, 2, 3)]
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['complete']
    for info in manifest['splits'].values():
        assert info['processed_contexts'] == 4
        assert info['contexts'] == info['questions'] == 3
        assert info['skipped_contexts'] == info['skipped_questions'] == 1
    payload = torch.load(first_part, weights_only=True)
    assert payload['skipped'][0]['context_id'] == 'train/1'
    assert payload['skipped'][0]['question_ids'] == ['train/1/q0']
    assert 'exceed' in payload['skipped'][0]['reason']
    # A completed cache must be reusable without even constructing a backbone.
    def forbidden(*args, **kwargs):
        pytest.fail('Completed cache attempted to load a backbone')
    monkeypatch.setattr(preparation, 'FrozenBackbone', forbidden)
    assert preparation.prepare(config) == root
    with pytest.raises(ValueError, match='different settings'):
        preparation.prepare(replace(config, layers=(1,)))


@pytest.mark.parametrize('include_context', [False, True])
def test_reader_roundtrip_labels_contexts_and_worker_partition(cache_setup, monkeypatch, include_context):
    config, _, _ = cache_setup
    root = preparation.prepare(config)
    dataset = reading.FeatureDataset(root, 'train', include_context=include_context)
    records = list(dataset)
    assert [r['context_id'] for r in records] == ['train/0', 'train/2', 'train/3']
    raw = torch.load(root / 'train/part-00000.pt', weights_only=True)['records'][0]
    torch.testing.assert_close(records[0]['questions'], raw['questions'])
    torch.testing.assert_close(records[0]['options'], raw['options'])
    batch = reading.collate_features(records).to('cpu')
    torch.testing.assert_close(batch.targets, torch.tensor([0.25, 0.75] * 3))
    assert batch.features.option_to_question.tolist() == [0, 0, 1, 1, 2, 2]
    assert batch.categories == ['demo'] * 3
    assert batch.target_kinds == ['soft'] * 3
    assert (batch.features.contexts is not None) == include_context
    if include_context:
        assert [len(c) for c in batch.features.contexts] == [1, 3, 4]
    # Worker assignment without spawning subprocesses: every shard exactly once.
    partitions = []
    for worker in range(2):
        monkeypatch.setattr(reading, 'get_worker_info', lambda worker=worker: SimpleNamespace(id=worker, num_workers=2))
        partitions.append([r['context_id'] for r in dataset])
    assert not set(partitions[0]) & set(partitions[1])
    assert sorted(partitions[0] + partitions[1]) == sorted(r['context_id'] for r in records)


def test_nonlength_errors_are_not_silently_skipped(cache_setup, monkeypatch):
    config, _, backbone = cache_setup
    def fail(self, batch, layers):
        raise RuntimeError('model failure')
    monkeypatch.setattr(backbone, 'forward', fail)
    with pytest.raises(RuntimeError, match='model failure'):
        preparation.prepare(config)
    manifest = json.loads((config.output / config.name / 'manifest.json').read_text())
    assert not manifest['complete']
    assert not manifest['splits']['train']['parts']
