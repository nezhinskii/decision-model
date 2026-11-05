"""Validation during training and one question-weighted optimizer step."""

import torch

from ..evaluation.metrics import MetricTotals, question_log_probs, question_metrics
from .objectives import LossConfig, training_values


@torch.no_grad()
def validate(model, loader, accelerator):
    was_training = model.training
    model.eval()
    totals = MetricTotals()
    try:
        for batch in loader:
            batch = batch.to(accelerator.device)
            with accelerator.autocast():
                logits = model(batch.features)
                log_probs = question_log_probs(logits, batch.features.option_to_question,
                                               len(batch.features.questions))
                values = question_metrics(log_probs, batch.targets,
                                          batch.features.option_to_question, len(batch.features.questions))
            totals.add(values, batch.categories, batch.target_kinds)
        return totals.result()
    finally:
        model.train(was_training)


def train_step(model, batches, optimizer, accelerator, clip_grad, loss_config=None):
    """Accumulate microbatches weighted by questions, not by microbatch size."""
    questions = sum(len(batch.features.questions) for batch in batches)
    # Keep detached metric sums on device; transfer only at logging intervals.
    sums = None
    loss_config = loss_config if loss_config is not None else LossConfig()
    optimizer.zero_grad(set_to_none=True)
    for batch in batches:
        batch = batch.to(accelerator.device)
        with accelerator.autocast():
            values = training_values(model(batch.features), batch, loss_config)
            loss = values.objective.sum() / questions
        accelerator.backward(loss)
        # Pack only at the logging boundary, in TRAIN_METRIC_NAMES order.
        batch_sums = torch.cat((values.metrics.sum(0),
                               values.objective.detach().sum().reshape(1),
                               values.ce_component.sum().reshape(1),
                               values.brier_component.sum().reshape(1)))
        sums = batch_sums if sums is None else sums + batch_sums
    accelerator.unscale_gradients(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), clip_grad, error_if_nonfinite=True
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return sums, questions
