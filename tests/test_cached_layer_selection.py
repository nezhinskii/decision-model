"""CPU layer selection preserves projection/readout order and checkpoint shapes."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from src.features.cache import collate_features
from src.heads import HeadConfig, build_head
from test_cache import cache_setup
from test_context_cache import multi_cache
from test_heads import features


def records_from(data):
    records = []
    for context in range(2):
        qs = (data.question_to_context == context).nonzero().flatten()
        vs = torch.isin(data.option_to_question, qs).nonzero().flatten()
        records.append({
            'context_id': str(context), 'context': data.contexts[context],
            'questions': data.questions[qs], 'options': data.options[vs],
            'option_to_question': data.option_to_question[vs] - qs[0],
            'question_ids': [data.question_ids[i] for i in qs.tolist()],
            'option_ids': [data.option_ids[i] for i in vs.tolist()],
            'targets': torch.ones(len(vs)),
            'categories': ['demo'] * len(qs), 'tasks': ['choice'] * len(qs),
            'target_kinds': ['hard'] * len(qs), 'score_values': [None] * len(qs),
            'feature_metadata': {'layers': [12, 16, 24]},
        })
    return records


@pytest.mark.parametrize('processor', ['identity', 'mlp', 'joint'])
@pytest.mark.parametrize('readout,indices', [
    ('last', (2, 0)), ('mean', (2, 0)), ('static', (-1, 0)), ('adaptive', (1, 2)),
])
@pytest.mark.parametrize('norm', ['layernorm', 'rmsnorm'])
def test_cpu_selection_preserves_logits_gradients_and_existing_state(processor, readout, indices, norm):
    data = features()
    config = HeadConfig(processor=processor, dim=8, hidden_dim=16,
                        score_dim=4, depth=1, attention_heads=2, num_slots=2,
                        readout=readout, layer_indices=indices, norm_type=norm)
    records = records_from(data)
    full = collate_features(records).features
    selected = collate_features(records, head_config=config).features
    expected = config.readout_indices(3)
    assert selected.questions.device.type == 'cpu'
    assert selected.questions.shape[1] == len(expected)
    assert selected.layer_indices == expected and selected.source_num_layers == 3
    assert selected.metadata == full.metadata  # Original cache identity remains intact.
    torch.testing.assert_close(selected.questions, full.questions[:, expected, :])
    assert selected.to('cpu').layer_indices == expected
    assert selected.to('cpu').source_num_layers == 3

    original = build_head(8, config, num_feature_layers=3)
    optimized = deepcopy(original)
    optimized.load_state_dict(original.state_dict(), strict=True)
    seen = []
    hook = optimized.projection.question[0].register_forward_pre_hook(
        lambda module, inputs: seen.append(inputs[0].shape[1]))
    a, b = original(full), optimized(selected)
    hook.remove()
    assert seen == [len(expected)]  # No unused layers reach the projection.
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    a.square().sum().backward()
    b.square().sum().backward()
    for (name, p), (_, q) in zip(original.named_parameters(), optimized.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=5e-6, rtol=1e-4)


def test_readout_rejects_wrong_selection_or_original_axis_size():
    config = HeadConfig(processor='identity', dim=8, score_dim=4,
                        readout='static', layer_indices=(2, 0))
    batch = collate_features(records_from(features()), head_config=config).features
    head = build_head(8, config, 3)
    with pytest.raises(ValueError, match='Preselected'):
        head(replace(batch, layer_indices=(0, 2)))
    with pytest.raises(ValueError, match='Preselected'):
        head(replace(batch, source_num_layers=4))


def test_loader_wires_selection_without_changing_head_construction(multi_cache):
    from src.features.cache import make_loader
    from src.features.prepare import prepare
    config, _ = multi_cache
    root = prepare(replace(config, name='two_pooled_layers', layers=(1, 2)))
    settings = HeadConfig(processor='joint', dim=8, hidden_dim=16, attention_heads=2,
                          score_dim=4, num_slots=0, readout='static', layer_indices=(0,))
    loader = make_loader(root, 'train', True, 2, 42, head_config=settings)
    batch = next(iter(loader)).features
    assert batch.questions.shape[1] == 1 and batch.source_num_layers == 2
    assert loader.dataset.context_layer == 1
    head = build_head(4, settings, batch.source_num_layers)
    assert head(batch).shape == (len(batch.option_ids),)

