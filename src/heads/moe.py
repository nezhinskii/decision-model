"""Dense and sparsely routed question–option updates for the shared pointer."""

import torch
from torch import nn

from .norms import make_norm
from .processors import DecisionStates, feed_forward


class QuestionOptionProcessor(nn.Module):
    requires_context = False

    def __init__(self, readout, dim, hidden_dim, dropout, norm_type, *,
                 num_experts=1, experts_per_option=1, routed=False):
        super().__init__()
        if not 1 <= experts_per_option <= num_experts:
            raise ValueError("experts_per_option must be between 1 and num_experts")
        self.readout = readout
        self.combine = nn.Sequential(nn.Linear(2 * dim, dim), make_norm(dim, norm_type))
        self.output_norm = make_norm(dim, norm_type)
        self.experts = nn.ModuleList(
            feed_forward(dim, hidden_dim, dropout) for _ in range(num_experts)
        )
        self.router = nn.Linear(2 * dim, num_experts) if routed else None
        self.k = experts_per_option
        if routed:
            self.register_buffer("load_fraction", torch.full((num_experts,), 1 / num_experts))

    def forward(self, features, *, return_routing=False):
        questions, options = self.readout(features)
        pairs = torch.cat((questions[features.option_to_question], options), dim=-1)
        inputs = self.combine(pairs)
        if self.router is None:
            if return_routing:
                raise ValueError("Routing statistics require a routed processor")
            update = self.experts[0](inputs)
        else:
            scores = self.router(pairs).float()
            selected, indices = scores.topk(self.k, dim=-1)
            full_probs = scores.softmax(-1) if return_routing or self.k == 1 else None
            if self.k == 1:
                # A one-element softmax is constant and cannot train the router.
                weights = full_probs.gather(1, indices)
            else:
                weights = selected.softmax(-1)
            # Group assignments once. Reading group sizes costs one device sync,
            # instead of a separate nonzero()/sync for every expert.
            assignments = indices.flatten()
            counts = assignments.new_zeros(len(self.experts))
            counts.scatter_add_(0, assignments, torch.ones_like(assignments))
            order = assignments.argsort()
            rows = order // self.k
            grouped_inputs = inputs[rows].split(counts.tolist())
            outputs = [
                expert(group) for expert, group in zip(self.experts, grouped_inputs)
                if group.shape[0]
            ]
            deltas = torch.cat(outputs) * weights.flatten()[order, None]
            update = torch.zeros_like(inputs).index_add(0, rows, deltas.to(inputs.dtype))
        states = DecisionStates(questions, self.output_norm(options + update))
        if return_routing:
            probabilities = full_probs.sum(0)
            return states, probabilities, counts
        return states

    @torch.no_grad()
    def record_assignments(self, counts):
        # Once per successful optimizer update, independent of microbatch packing.
        self.load_fraction.lerp_((counts / counts.sum()).to(self.load_fraction), 0.1)
