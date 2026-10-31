"""Processors return one question state and one state per original option."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from ..features import FeatureBatch
from .readout import LayerReadout
from .norms import NormType, make_norm


@dataclass
class DecisionStates:
    """Flat states in original order, with the layer axis reduced by a processor."""

    questions: torch.Tensor  # [Q, d]
    options: torch.Tensor  # [V, d]


class IdentityProcessor(nn.Module):
    """Apply layer readout and leave the resulting states unchanged."""

    requires_context = False

    def __init__(self, readout: LayerReadout):
        super().__init__()
        self.readout = readout

    def forward(self, features: FeatureBatch) -> DecisionStates:
        return DecisionStates(*self.readout(features))


def feed_forward(dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """Shared FFN structure for MLP and Joint blocks, with independent weights."""
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class ResidualMLP(nn.Module):
    """Pre-normalized pointwise update; no interaction between input rows."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float,
        norm_type: NormType = NormType.layernorm,
    ):
        super().__init__()
        self.norm = make_norm(dim, norm_type)
        self.ffn = feed_forward(dim, hidden_dim, dropout)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return states + self.ffn(self.norm(states))


class SeparateMLPProcessor(nn.Module):
    """Different question/option MLPs; one option MLP is shared across all options."""

    requires_context = False

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        readout: LayerReadout,
        norm_type: NormType = NormType.layernorm,
    ):
        super().__init__()
        self.readout = readout
        self.questions = nn.Sequential(
            *[ResidualMLP(dim, hidden_dim, dropout, norm_type) for _ in range(depth)]
        )
        self.options = nn.Sequential(
            *[ResidualMLP(dim, hidden_dim, dropout, norm_type) for _ in range(depth)]
        )

    def forward(self, features: FeatureBatch) -> DecisionStates:
        questions, options = self.readout(features)
        return DecisionStates(self.questions(questions), self.options(options))


def group_positions(groups: torch.Tensor, count: int):
    """Local positions for rows already ordered by ascending group ID."""
    sizes = torch.bincount(groups, minlength=count)  # [count]
    starts = sizes.cumsum(dim=0) - sizes  # [count], first row of each group
    row_indices = torch.arange(len(groups), device=groups.device)  # [rows]
    positions = row_indices - starts[groups]  # [rows], position within its group
    return positions, int(sizes.max())


def pack_rows(rows: torch.Tensor, groups: torch.Tensor, count: int):
    """Pack contiguous groups into [count, max_group_size, d].

    Groups must be in ascending order, as produced by our feature pipeline.
    Padding is local to attention and never changes the public flat format.
    """
    positions, max_size = group_positions(groups, count)
    packed = rows.new_zeros((count, max_size, rows.shape[-1]))
    valid = torch.zeros(packed.shape[:2], dtype=torch.bool, device=rows.device)
    packed[groups, positions] = rows
    valid[groups, positions] = True
    return packed, valid, positions


@dataclass
class ContextAttention:
    """Fixed memory and query addresses reused by all Joint blocks in a forward."""

    memory: torch.Tensor  # [B, T_max, d]
    padding_mask: torch.Tensor  # [B, T_max], True means ignore this token
    context_ids: torch.Tensor  # [S], context ID per real workspace state
    positions: torch.Tensor  # [S], query position inside its context group
    max_queries: int  # Maximum real workspace states sharing one context


def prepare_context_attention(
    features: FeatureBatch, valid: torch.Tensor
) -> ContextAttention:
    """Prepare context padding and grouping once; values change only inside blocks."""
    if features.contexts is None or any(len(c) == 0 for c in features.contexts):
        raise ValueError("Cross-attention requires nonempty context-token features")
    memory = pad_sequence(features.contexts, batch_first=True)  # [B, T_max, d]
    lengths = torch.tensor(
        [len(c) for c in features.contexts], device=memory.device
    )  # [B]
    token_positions = torch.arange(memory.shape[1], device=memory.device)  # [T_max]
    padding_mask = token_positions[None, :] >= lengths[:, None]  # [B, T_max]

    # Assign every real question/slot/option state its question's context ID.
    per_state_context = features.question_to_context[:, None].expand_as(valid)  # [Q, N]
    context_ids = per_state_context[valid]  # [S], grouped by context already
    positions, max_queries = group_positions(context_ids, len(memory))
    return ContextAttention(memory, padding_mask, context_ids, positions, max_queries)


class JointBlock(nn.Module):
    """Cross -> Self -> FFN, each with pre-norm and residual connections."""

    def __init__(
        self,
        dim: int,
        heads: int,
        hidden_dim: int,
        dropout: float,
        use_cross_attention: bool,
        norm_type: NormType = NormType.layernorm,
    ):
        super().__init__()
        self.cross = None
        if use_cross_attention:
            self.cross_norm = make_norm(dim, norm_type)
            self.cross = nn.MultiheadAttention(
                dim, heads, dropout=dropout, batch_first=True
            )
        self.self_norm = make_norm(dim, norm_type)
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.ffn = ResidualMLP(dim, hidden_dim, dropout, norm_type)
        self.dropout = nn.Dropout(dropout)

    def forward(self, states, valid, context: ContextAttention | None):
        if self.cross is not None:
            # Group all real queries sharing a context. Cross-attention never
            # mixes queries, so this shares K/V computation without mixing questions.
            real_queries = self.cross_norm(states)[valid]  # [S, d]
            queries = states.new_zeros(
                (len(context.memory), context.max_queries, states.shape[-1])
            )
            queries[context.context_ids, context.positions] = (
                real_queries  # [B, S_max, d]
            )
            read, _ = self.cross(
                queries,
                context.memory,
                context.memory,
                key_padding_mask=context.padding_mask,
                need_weights=False,
            )
            update = torch.zeros_like(states)
            update[valid] = read[context.context_ids, context.positions]
            states = states + self.dropout(update)

        # Each question is a separate batch row: attention is bidirectional
        # within a question, with no exchange between different questions.
        normalized = self.self_norm(states)
        update, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=False,
        )
        states = self.ffn(states + self.dropout(update))
        return states.masked_fill(~valid[..., None], 0)


class JointProcessor(nn.Module):
    """Joint question/option workspace with optional slots and context access."""

    def __init__(
        self,
        dim: int,
        heads: int,
        hidden_dim: int,
        depth: int,
        dropout: float,
        num_slots: int,
        use_cross_attention: bool,
        readout: LayerReadout,
        norm_type: NormType = NormType.layernorm,
    ):
        super().__init__()
        self.readout = readout
        self.requires_context = use_cross_attention
        self.num_slots = num_slots
        # Zero slots means no slot parameter at all, not masked unused parameters.
        self.slots = nn.Parameter(torch.empty(num_slots, dim)) if num_slots else None
        if self.slots is not None:
            nn.init.normal_(self.slots, std=0.02)
        self.blocks = nn.ModuleList(
            [
                JointBlock(
                    dim, heads, hidden_dim, dropout, use_cross_attention, norm_type
                )
                for _ in range(depth)
            ]
        )

    def forward(self, features: FeatureBatch) -> DecisionStates:
        questions, flat_options = self.readout(features)  # [Q, d], [V, d]
        q_count = len(questions)
        # Build one padded workspace per question: [question, slots, options].
        options, option_valid, positions = pack_rows(
            flat_options,
            features.option_to_question,
            q_count,
        )
        prefix = questions[:, None]
        if self.slots is not None:
            prefix = torch.cat(
                (prefix, self.slots[None].expand(q_count, -1, -1)), dim=1
            )
        states = torch.cat((prefix, options), dim=1)
        valid = torch.cat(
            (
                torch.ones(prefix.shape[:2], dtype=torch.bool, device=states.device),
                option_valid,
            ),
            dim=1,
        )

        # Workspace membership is fixed across blocks: compute its layout once.
        context = (
            prepare_context_attention(features, valid)
            if self.requires_context
            else None
        )
        for block in self.blocks:
            states = block(states, valid, context)
        # Discard slots/padding and restore flat options in their original order.
        return DecisionStates(
            states[:, 0],
            states[features.option_to_question, 1 + self.num_slots + positions],
        )
