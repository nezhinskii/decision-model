"""Training objectives checked against an independent per-question reference."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from src.training.objectives import LossConfig, training_values
from src.evaluation.metrics import question_log_probs, question_metrics
from src.training.train_head import train
from test_cache import cache_setup
from test_training import single_thread_accelerate
from test_training_state import run_setup


@pytest.mark.parametrize('weights,expected', [((1., 0.), {'ce'}),
                                            ((0., 1.), {'brier'}),
                                            ((1., .5), {'ce', 'brier'})])
def test_disabled_loss_branch_is_not_visited_in_backward(monkeypatch, weights, expected):
    import src.training.objectives as objectives
    visited, graphs = set(), set()

    def instrument(function, name):
        def wrapped(*args):
            result = function(*args)
            if result.requires_grad:
                graphs.add(name)
                result.register_hook(lambda gradient: visited.add(name))
            return result
        return wrapped

    monkeypatch.setattr(objectives, 'cross_entropy', instrument(objectives.cross_entropy, 'ce'))
    monkeypatch.setattr(objectives, 'brier_score', instrument(objectives.brier_score, 'brier'))
    batch = SimpleNamespace(targets=torch.tensor([1., 0.]),
                            features=SimpleNamespace(option_to_question=torch.tensor([0, 0]),
                                                     questions=torch.empty(1, 1, 1)))
    logits = torch.tensor([.1, .2], requires_grad=True)
    values = training_values(logits, batch, LossConfig(*weights))
    values.objective.sum().backward()
    assert visited == graphs == expected
    assert not values.metrics.requires_grad
    assert not values.ce_component.requires_grad
    assert not values.brier_component.requires_grad


@pytest.mark.parametrize('weights', [(1., 0.), (0., 1.), (1., .5)])
@pytest.mark.parametrize('scope', ['ce', 'both'])
@pytest.mark.parametrize('epsilon', [0., .1])
def test_objective_values_gradients_and_original_metrics(weights, scope, epsilon):
    links = torch.tensor([0, 0, 1, 1, 1, 2])
    targets = torch.tensor([1., 0., .2, .3, .5, 1.])
    batch = SimpleNamespace(targets=targets, target_kinds=['hard', 'soft', 'hard'],
                            features=SimpleNamespace(option_to_question=links, questions=torch.empty(3, 1, 1)))
    logits = torch.tensor([.3, -.2, -1., 2., .5, .4], requires_grad=True)
    config = LossConfig(*weights, epsilon, scope)
    before = targets.clone()
    actual = training_values(logits, batch, config)
    references = []
    for q in range(3):
        selected = links == q
        log_probs = logits[selected].log_softmax(0)
        original = targets[selected]
        smoothed = ((1 - epsilon) * original + epsilon / len(original)
                    if batch.target_kinds[q] == 'hard' else original)
        ce = -(smoothed * log_probs).sum()
        brier = (log_probs.exp() - (smoothed if scope == 'both' else original)).square().sum()
        references.append(torch.stack((weights[0] * ce + weights[1] * brier, ce, brier)))
    reference = torch.stack(references)
    torch.testing.assert_close(torch.stack((actual.objective, actual.ce_component, actual.brier_component), dim=-1), reference)
    torch.testing.assert_close(actual.metrics, question_metrics(question_log_probs(logits, links, 3), targets, links, 3))
    torch.testing.assert_close(targets, before, atol=0, rtol=0)
    a = torch.autograd.grad(actual.objective.sum(), logits, retain_graph=True)[0]
    b = torch.autograd.grad(reference[:, 0].sum(), logits)[0]
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize('kwargs', [dict(ce_weight=0, brier_weight=0),
    dict(ce_weight=-1), dict(brier_weight=float('nan')), dict(label_smoothing=-.1),
    dict(label_smoothing=1), dict(smoothing_scope='unknown')])
def test_invalid_loss_settings(kwargs):
    with pytest.raises(ValueError):
        LossConfig(**kwargs)


def test_mixed_objective_resume_and_configuration_guard(run_setup):
    config, runs = run_setup
    part = config.cache / 'train/part-00000.pt'
    payload = torch.load(part, weights_only=True)
    payload['records'][0]['targets'] = torch.tensor([1., 0.])
    payload['records'][0]['target_kinds'] = ['hard']
    torch.save(payload, part)
    config = replace(config, loss=LossConfig(1., .5, .05),
                     accumulation_steps=2, epochs=2, log_every=1, schedule='cosine')
    root = train(replace(config, max_steps=1))
    train(config)
    full = train(replace(config, name='full'))
    a = torch.load(root / 'latest.pt', weights_only=False)
    b = torch.load(full / 'latest.pt', weights_only=False)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
    for _, values in runs[str(root)]['logs']:
        if 'train/objective' in values:
            assert values['train/objective'] == pytest.approx(
                values['train/ce_component'] + .5 * values['train/brier_component'], rel=1e-6)
    with pytest.raises(ValueError, match='configuration changed'):
        train(replace(config, loss=LossConfig()))
