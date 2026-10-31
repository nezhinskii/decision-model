"""Small scientific invariants of selection, grouping and token filtering."""

from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

from src.backbone.inputs import InputBuilder, InputConfig
from src.data import openjev
from src.data.lengths import filter_candidates
from src.data.sampling import Candidate, select, split_groups
from src.data.schema import Example, context_id


def test_balancing_is_deterministic_and_context_splits_are_disjoint():
    candidates = [Candidate(i, str(i // 2), 'small' if i < 2 else 'large', 'choice', 'hard') for i in range(14)]
    selected, quotas = select(candidates, 10, seed=42)
    assert quotas == {'large': 8, 'small': 2}
    assert selected == select(candidates, 10, seed=42)[0]
    assert len({c.row for c in selected}) == 10
    assignment, _ = split_groups(selected, {'train': 6, 'validation': 2, 'test': 2}, seed=42)
    seen = set()
    for split in assignment.values():
        groups = {c.group for c in split}
        assert not groups & seen
        seen.update(groups)
    assert sorted(c.row for rows in assignment.values() for c in rows) == sorted(c.row for c in selected)


def test_length_filter_caps_group_before_sampling(tmp_path, tokenizer):
    rows = []
    for state in ['shared', 'shared', 'shared', 'x' * 300, 'other']:
        rows.append({
            'state': state, 'question': 'Choose?', 'category': 'demo', 'task': 'choice',
            'options': [{'id': 'a', 'name': 'yes', 'criteria': None},
                        {'id': 'b', 'name': 'no', 'criteria': None}],
            'answer': {'kind': 'soft', 'probabilities': [0.25, 0.75]}, 'score_values': None,
        })
    path = tmp_path / 'source.parquet'
    pq.write_table(pa.Table.from_pylist(rows), path)
    candidates = openjev.read_index(path, 2)
    builder = InputBuilder(tokenizer, InputConfig(max_packed_tokens=90))
    eligible, stats = filter_candidates(path, candidates, builder, openjev.REVISION, 42, 2)
    assert stats['excluded_individually'] == 1
    assert stats['excluded_group_budget'] > 0
    assert Counter(c.group for c in eligible)[context_id('shared')] < 3
    for group in {c.group for c in eligible}:
        picked = [c for c in eligible if c.group == group]
        example = Example(group, rows[picked[0].row]['state'], [
            openjev.convert(rows[c.row], c.row, openjev.REVISION) for c in picked
        ])
        assert len(builder.encode(example).input_ids) <= 90
