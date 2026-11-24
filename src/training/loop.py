"""Validation and optimizer math for any supervised decision batch.

Batches expose inputs, targets, option_to_question, num_questions, categories,
target_kinds and to(device). Neither function reads hidden states or token layout.
"""

import torch

from ..evaluation.metrics import MetricTotals, question_log_probs, question_metrics
from .objectives import LossConfig, TRAIN_METRIC_NAMES, routing_balance_sum, training_values


@torch.no_grad()
def validate(model, loader, accelerator):
    was_training = model.training
    model.eval()
    totals = MetricTotals()
    routing = getattr(model, "routing_processor", None)
    counts = None
    try:
        for batch in loader:
            batch = batch.to(accelerator.device)
            with accelerator.autocast():
                if routing is None:
                    logits = model(batch.inputs)
                else:
                    logits, _, assignments = model(batch.inputs, return_routing=True)
                    counts = assignments if counts is None else counts + assignments
                log_probs = question_log_probs(
                    logits, batch.option_to_question, batch.num_questions
                )
                values = question_metrics(
                    log_probs,
                    batch.targets,
                    batch.option_to_question,
                    batch.num_questions,
                )
            totals.add(values, batch.categories, batch.target_kinds)
        result = totals.result()
        if counts is not None:
            for i, fraction in enumerate((counts / counts.sum()).tolist()):
                result[f"routing/expert_{i}_fraction"] = fraction
        return result
    finally:
        model.train(was_training)


def train_step(model, batches, optimizer, accelerator, clip_grad, loss_config=None):
    """Accumulate microbatches weighted by questions, not by microbatch size."""
    questions = sum(batch.num_questions for batch in batches)
    # Keep detached metric sums on device; transfer only at logging intervals.
    sums = None
    loss_config = loss_config if loss_config is not None else LossConfig()
    routing = getattr(model, "routing_processor", None)
    if loss_config.balance_weight and routing is None:
        raise ValueError("balance_weight requires a routed head")
    if routing is not None:
        options = sum(batch.targets.numel() for batch in batches)
        counts = torch.zeros_like(routing.load_fraction, dtype=torch.long)
        balance_sum = counts.new_zeros((), dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    for batch in batches:
        batch = batch.to(accelerator.device)
        with accelerator.autocast():
            if routing is None:
                logits = model(batch.inputs)
            else:
                logits, probability_sums, assignments = model(batch.inputs, return_routing=True)
                balance = routing_balance_sum(probability_sums, routing.load_fraction)
                balance_sum += balance.detach()
                counts += assignments
            values = training_values(
                logits,
                batch.targets,
                batch.option_to_question,
                batch.num_questions,
                batch.target_kinds,
                loss_config,
            )
            loss = values.objective.sum() / questions
            if loss_config.balance_weight:
                loss = loss + loss_config.balance_weight * balance / options
        accelerator.backward(loss)
        # Pack only at the logging boundary, in TRAIN_METRIC_NAMES order.
        batch_sums = torch.cat(
            (
                values.metrics.sum(0),
                values.objective.detach().sum().reshape(1),
                values.ce_component.sum().reshape(1),
                values.brier_component.sum().reshape(1),
            )
        )
        sums = batch_sums if sums is None else sums + batch_sums
    accelerator.unscale_gradients(optimizer)
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), clip_grad, error_if_nonfinite=True
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if routing is not None:
        if not accelerator.optimizer_step_was_skipped:
            routing.record_assignments(counts)
        # The leading metrics use question counts; the routing suffix has its
        # own option/assignment denominators. Keep raw sums until logging.
        objective_index = TRAIN_METRIC_NAMES.index("objective")
        sums[objective_index] += loss_config.balance_weight * balance_sum * (questions / options)
        sums = torch.cat((sums, sums.new_tensor([options]), balance_sum.reshape(1), counts))
    return sums, questions
