"""Shared head inputs, independent of feature extraction or on-disk storage."""

from dataclasses import dataclass, replace

import torch


@dataclass
class FeatureBatch:
    """Real rows with local indices; no supervision or padding.

    Options are contiguous by question, questions contiguous by context, both in
    ascending index order. Extractors/collators preserve this grouping contract.
    """

    questions: torch.Tensor  # [questions, layers, hidden_size]
    options: torch.Tensor  # [options, layers, hidden_size]
    option_to_question: torch.Tensor  # [V], integer indices into questions.
    question_to_context: torch.Tensor  # [Q], integer indices into contexts.
    contexts: list[torch.Tensor] | None  # B tensors [T_b, D], variable token lengths.
    question_ids: list[str]  # Q IDs, aligned with questions.
    option_ids: list[str]  # V IDs, aligned with options; not globally unique alone.
    metadata: dict

    def to(self, device: str | torch.device) -> "FeatureBatch":
        """Move head inputs, preserving dtype and autograd connections."""
        return replace(
            self,
            questions=self.questions.to(device),
            options=self.options.to(device),
            option_to_question=self.option_to_question.to(device),
            question_to_context=self.question_to_context.to(device),
            contexts=(
                [context.to(device) for context in self.contexts]
                if self.contexts is not None
                else None
            ),
        )
