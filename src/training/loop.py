"""Shared evaluation and one question-weighted optimizer step."""

import torch

from .objectives import MetricTotals, question_metrics


@torch.no_grad()
def evaluate(model, loader, accelerator):
    model.eval()
    totals = MetricTotals()
    for batch in loader:
        batch = batch.to(accelerator.device)
        with accelerator.autocast():
            values = question_metrics(model(batch.features), batch)
        totals.add(values, batch)
    model.train()
    return totals.result()


def train_step(model, batches, optimizer, accelerator, clip_grad):
    """Accumulate microbatches weighted by questions, not by microbatch size."""
    questions = sum(len(batch.features.questions) for batch in batches)
    # Keep detached metric sums on device; transfer only at logging intervals.
    sums = None
    optimizer.zero_grad(set_to_none=True)
    for batch in batches:
        batch = batch.to(accelerator.device)
        with accelerator.autocast():
            values = question_metrics(model(batch.features), batch)
            loss = values[:, 0].sum() / questions
        accelerator.backward(loss)
        batch_sums = values.detach().sum(0)
        sums = batch_sums if sums is None else sums + batch_sums
    accelerator.unscale_gradients(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), clip_grad, error_if_nonfinite=True
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return sums, questions
