"""Pool selected hidden states into flat question and option features."""

from dataclasses import asdict, dataclass
from enum import StrEnum

import torch

from .inputs import Block, PackedBatch
from .model import BackboneOutput
from ..features import FeatureBatch


class PoolingMode(StrEnum):
    last = "last"
    mean = "mean"


@dataclass
class PoolingConfig:
    """Pooling is independent of the decoder depths requested from the backbone."""

    question: PoolingMode = PoolingMode.last
    option: PoolingMode = PoolingMode.last
    include_context: bool = False


def pool(states: torch.Tensor, block: Block, mode: str) -> torch.Tensor:
    """Exclude block headers, trailing separators and padding from pooling."""
    if block.content_stop <= block.content_start:
        raise ValueError(f"Cannot pool empty {block.role} content")
    if mode == "last":
        return states[block.content_stop - 1]
    if mode == "mean":
        return (
            states[block.content_start : block.content_stop]
            .float()
            .mean(0)
            .to(states.dtype)
        )
    raise ValueError(f"Unknown pooling mode: {mode}")


def extract_features(
    output: BackboneOutput, batch: PackedBatch, config: PoolingConfig | None = None
) -> FeatureBatch:
    """Use question or Decision spans consistently at every requested depth."""
    config = config or PoolingConfig()
    layers = tuple(output.hidden_states)
    queries, options, q_to_context, o_to_q, question_ids, option_ids = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    contexts = [] if config.include_context else None
    for b, sample in enumerate(batch.samples):
        for q, question in enumerate(sample.example.questions):
            role = batch.config.query_source
            query = next(
                block
                for block in sample.blocks
                if block.role == role and block.question == q
            )
            query_pooling = "last" if role == "decision" else config.question
            queries.append(
                torch.stack(
                    [
                        pool(output.hidden_states[layer][b], query, query_pooling)
                        for layer in layers
                    ]
                )
            )
            q_to_context.append(b)
            question_ids.append(question.id)
            for o, option in enumerate(question.options):
                block = next(
                    block
                    for block in sample.blocks
                    if block.role == "option"
                    and block.question == q
                    and block.option == o
                )
                options.append(
                    torch.stack(
                        [
                            pool(output.hidden_states[layer][b], block, config.option)
                            for layer in layers
                        ]
                    )
                )
                o_to_q.append(len(queries) - 1)
                option_ids.append(option.id)
        if contexts is not None:
            # Cross-attention memory is always from the final normalized layer.
            final = output.metadata["num_hidden_layers"]
            if final not in layers:
                raise ValueError(
                    "include_context requires requesting the final decoder layer"
                )
            block = next(block for block in sample.blocks if block.role == "context")
            contexts.append(
                output.hidden_states[final][b, block.content_start : block.content_stop]
            )
    if not queries or not options:
        raise ValueError(
            "Feature extraction requires questions with nonempty option sets"
        )
    device = queries[0].device
    return FeatureBatch(
        torch.stack(queries),
        torch.stack(options),
        torch.tensor(o_to_q, device=device),
        torch.tensor(q_to_context, device=device),
        contexts,
        question_ids,
        option_ids,
        {
            **output.metadata,
            "input": asdict(batch.config),
            "pooling": asdict(config),
            "effective_query_pooling": (
                "last" if batch.config.query_source == "decision" else config.question
            ),
        },
    )
