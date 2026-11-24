"""Small CPU references for sparse dispatch and optimizer-step routing history."""

from copy import deepcopy

from accelerate import Accelerator
import pytest
import torch

from src.features.schema import FeatureBatch, LabeledFeatureBatch
from src.heads import HeadConfig, build_head
from src.heads.processors import DecisionStates
from src.training.loop import train_step, validate
from src.training.objectives import LossConfig


def batch(sizes, questions, options):
    ids = [str(i) for i in range(len(sizes))]
    links = torch.repeat_interleave(torch.arange(len(sizes)), torch.tensor(sizes))
    features = FeatureBatch(questions, options, links, torch.arange(len(sizes)),
                            None, ids, list(map(str, range(len(options)))), {})
    targets = torch.cat([torch.eye(1, size).flatten() for size in sizes])
    return LabeledFeatureBatch(features, targets, ids, ["qa"] * len(ids),
                               ["choice"] * len(ids), ["hard"] * len(ids), [None] * len(ids))


def make_head(k):
    return build_head(6, HeadConfig(processor="question_option_moe", dim=6,
                                    hidden_dim=9, score_dim=4,
                                    num_experts=4, experts_per_option=k))


@pytest.mark.parametrize("k", [1, 2, 4])
def test_sparse_dispatch_matches_dense_reference(k):
    torch.manual_seed(7)
    head = make_head(k)
    data = batch([2, 3], torch.randn(2, 1, 6), torch.randn(5, 1, 6)).inputs
    logits, probabilities, counts = head(data, return_routing=True)
    proc = head.processor
    q, o = proc.readout(head.projection(data))
    pairs = torch.cat([q[data.option_to_question], o], -1)
    inputs = proc.combine(pairs)
    probs = proc.router(pairs).softmax(-1)
    weights, indices = probs.topk(k)
    if k > 1:
        weights = weights / weights.sum(-1, keepdim=True)
    dense_weights = torch.zeros_like(probs).scatter(1, indices, weights)
    update = sum(dense_weights[:, i, None] * expert(inputs)
                 for i, expert in enumerate(proc.experts))
    expected = head.scorer(DecisionStates(q, proc.output_norm(o + update)), data.option_to_question)
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(probabilities, probs.sum(0))
    torch.testing.assert_close(counts, torch.bincount(indices.flatten(), minlength=4))
    logits.square().sum().backward()
    assert proc.router.weight.grad.abs().sum() > 0
    for expert, count in zip(proc.experts, counts):
        assert (next(expert.parameters()).grad is not None) == bool(count)


def test_accumulation_uses_option_counts_and_updates_history_once():
    torch.manual_seed(11)
    q, o = torch.randn(2, 1, 6), torch.randn(7, 1, 6)
    combined = batch([2, 5], q, o)
    split = [batch([2], q[:1], o[:2]), batch([5], q[1:], o[2:])]
    model = make_head(2)
    history = torch.tensor([.1, .2, .3, .4])
    model.processor.load_fraction.copy_(history)
    reference = deepcopy(model)
    accelerator = Accelerator(cpu=True, mixed_precision="no")
    with torch.no_grad():
        _, probabilities, counts = reference(combined.inputs, return_routing=True)
        expected_balance = 4 * probabilities.dot(history)
    for batches, current in [([combined], reference), (split, model)]:
        opt = torch.optim.SGD(current.parameters(), lr=.02)
        sums, questions = train_step(current, batches, opt, accelerator, 100,
                                     LossConfig(balance_weight=.1))
        assert questions == 2 and sums[6] == 7
        torch.testing.assert_close(sums[7], expected_balance)
        torch.testing.assert_close(sums[8:], counts.float())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[key])
    expected_load = history.lerp(counts.float() / 14, .1)
    metrics = validate(model, split, accelerator)
    assert sum(metrics[f"routing/expert_{i}_fraction"] for i in range(4)) == pytest.approx(1.)
    torch.testing.assert_close(model.processor.load_fraction, expected_load)
    restored = make_head(2)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored.processor.load_fraction, expected_load)
