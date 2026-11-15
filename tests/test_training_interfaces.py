"""Training math accepts ordinary tensor inputs without a feature-cache batch."""

from copy import deepcopy
from dataclasses import dataclass, replace

from accelerate import Accelerator
import torch
from torch import nn

from src.evaluation.metrics import question_log_probs, question_metrics
from src.training.loop import train_step, validate
from src.training.objectives import LossConfig, training_values


@dataclass
class TensorBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    option_to_question: torch.Tensor
    num_questions: int
    categories: list[str]
    target_kinds: list[str]

    def to(self, device):
        return replace(self, inputs=self.inputs.to(device), targets=self.targets.to(device),
                       option_to_question=self.option_to_question.to(device))


def make_batch():
    return TensorBatch(torch.tensor([[1., 2.], [2., -1.], [0., 3.], [1., 1.], [2., 2.]]),
                       torch.tensor([1., 0., .2, .3, .5]), torch.tensor([0, 0, 1, 1, 1]),
                       2, ['rules', 'logic'], ['hard', 'soft'])


def test_training_step_and_validation_do_not_require_features():
    batch = make_batch()
    model = nn.Sequential(nn.Linear(2, 1), nn.Flatten(0))
    reference = deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
    accelerator = Accelerator(cpu=True, mixed_precision='no')
    config = LossConfig(1., .5, .05)
    logits = reference(batch.inputs)
    values = training_values(logits, batch.targets, batch.option_to_question,
                             batch.num_questions, batch.target_kinds, config)
    values.objective.mean().backward()
    reference_optimizer.step()
    sums, count = train_step(model, [batch], optimizer, accelerator, 100., config)
    assert count == 2 and sums.shape == (6,)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected)
    model.train()
    metrics = validate(model, [batch], accelerator)
    assert model.training
    with torch.no_grad():
        expected = question_metrics(question_log_probs(model(batch.inputs), batch.option_to_question, 2),
                                    batch.targets, batch.option_to_question, 2).double().mean(0)
    for i, key in enumerate(('loss', 'accuracy', 'brier')):
        assert abs(metrics[f'all/{key}'] - expected[i].item()) < 1e-7
