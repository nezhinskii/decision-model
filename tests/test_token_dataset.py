"""Disk token iteration is bounded, deterministic and does not retokenize epochs."""

from dataclasses import asdict
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from src.backbone.dataset import DecisionDataset, RejectedExample, make_decision_loader
from src.backbone.inputs import InputBuilder, InputConfig
from src.features.cache import FeatureDataset
from src.utils.files import file_hash
from src.data.schema import arrow_schema
from conftest import make_example


@pytest.fixture
def token_source(tmp_path, tokenizer):
    examples = [make_example(str(i), 'Facts ' * (i + 1), 1) for i in range(9)]
    path = tmp_path / 'train.parquet'
    pq.write_table(pa.Table.from_pylist([asdict(e) for e in examples], schema=arrow_schema()), path)
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'splits': {'train': {'file': path.name, 'sha256': file_hash(path)}}
    }))
    return tmp_path, InputBuilder(tokenizer, InputConfig(attention='causal')), examples


def test_disk_roundtrip_is_lazy_and_does_not_retokenize(token_source, monkeypatch):
    root, builder, examples = token_source
    expected = [builder.encode(e) for e in examples]
    data = DecisionDataset(root, 'train', builder, 10000, shard_size=2)
    assert len(data) == 9 and len(data.parts) == 5
    assert not hasattr(data, 'samples')
    monkeypatch.setattr(builder, 'encode', lambda *a, **kw: pytest.fail('Retokenized epoch'))
    load = torch.load
    calls = []
    def tracked(path, **kwargs):
        calls.append(path)
        return load(path, **kwargs)
    monkeypatch.setattr(torch, 'load', tracked)
    iterator = iter(data)
    first = next(iterator)
    assert len(calls) == 1  # Later shards have not been loaded.
    actual = [first, *iterator]
    assert len(calls) == 5
    for before, after in zip(expected, actual):
        assert before == after  # Includes original texts, labels, blocks and positions.
    assert sum(b.num_questions for b in make_decision_loader(data, 2)) == 9
    storage = data.root
    data.close()
    assert not storage.exists()


def test_token_limit_and_context_limit_bound_shards(token_source):
    root, builder, _ = token_source
    data = DecisionDataset(root, 'train', builder, 10000,
                           shard_size=3, max_shard_tokens=300)
    assert len(data.parts) > 1
    for path in data.parts:
        records = torch.load(path, weights_only=False)
        assert len(records) <= 3
        assert sum(len(r.input_ids) for r in records) <= 300 or len(records) == 1
        assert all(isinstance(r.input_ids, torch.Tensor) for r in records)
        assert all(isinstance(r.positions, torch.Tensor) for r in records)


def test_shuffle_replay_resume_and_rng_isolation(token_source):
    root, builder, _ = token_source
    data = DecisionDataset(root, 'train', builder, 10000,
                           shard_size=2, seed=17, shuffle=True)
    def ids():
        return [s.example.id for s in data]
    rng = torch.get_rng_state().clone()
    data.set_epoch(0)
    first = ids()
    assert first == ids()
    data.set_epoch(1)
    second = ids()
    assert set(first) == set(second) and first != second
    iterator = iter(data)
    next(iterator)
    next(iterator)
    assert [s.example.id for s in iterator] == second[2:]
    assert torch.equal(rng, torch.get_rng_state())


def test_rejected_texts_also_stream_from_disk(token_source):
    root, builder, examples = token_source
    builder.config.max_packed_tokens = 1
    data = DecisionDataset(root, 'train', builder, 10000, shard_size=2)
    assert len(data) == 0 and list(data) == []
    assert data.summary['skipped_contexts'] == len(examples)
    assert [r.example for r in data.rejected] == examples
    assert all(isinstance(r, RejectedExample) for r in data.rejected)


def test_source_checksum_and_limits_are_checked(token_source):
    root, builder, _ = token_source
    with pytest.raises(ValueError, match='limits'):
        DecisionDataset(root, 'train', builder, 10000, shard_size=0)
    with (root / 'train.parquet').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        DecisionDataset(root, 'train', builder, 10000)


def test_cached_head_loader_already_reads_one_shard_at_a_time(tmp_path, monkeypatch):
    parts = []
    for i in range(3):
        path = tmp_path / f'part-{i}.pt'
        torch.save({'records': [{'context': None, 'question_ids': [str(i)]}]}, path)
        parts.append({'file': path.name})
    (tmp_path / 'manifest.json').write_text(json.dumps({
        'complete': True, 'features': {'pooling': {'include_context': False}},
        'splits': {'train': {'parts': parts}},
    }))
    data = FeatureDataset(tmp_path, 'train')
    load, calls = torch.load, []
    def tracked(path, **kwargs):
        calls.append(path)
        return load(path, **kwargs)
    monkeypatch.setattr(torch, 'load', tracked)
    iterator = iter(data)
    assert next(iterator)['question_ids'] == ['0']
    assert len(calls) == 1
    assert [r['question_ids'] for r in iterator] == [['1'], ['2']]
    assert len(calls) == 3
