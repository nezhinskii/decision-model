"""Shared head inputs, independent of feature extraction or on-disk storage."""

from dataclasses import dataclass, replace

import torch


@dataclass
class FeatureBatch:
    """Real question/option rows with local indices; no supervision or padding."""

    questions: torch.Tensor  # [questions, layers, hidden_size]
    options: torch.Tensor  # [options, layers, hidden_size]
    option_to_question: torch.Tensor
    question_to_context: torch.Tensor
    contexts: list[torch.Tensor] | None
    question_ids: list[str]
    option_ids: list[str]
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
