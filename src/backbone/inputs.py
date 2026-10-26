"""Serialize decision examples and construct matching attention masks and positions."""

from dataclasses import dataclass
from typing import Literal

import torch

from ..data.schema import Example

AttentionMode = Literal["causal", "independent_questions", "independent_options"]


class InputTooLongError(ValueError):
    """An oversized serialized example; other input errors must not be skipped."""


@dataclass
class InputConfig:
    """Input settings that must be included in any future feature-cache key."""

    attention: AttentionMode = "independent_options"
    prompt_format: Literal["plain", "chat"] = "plain"
    query_source: Literal["question", "decision"] = "question"
    max_packed_tokens: int = 2048


@dataclass
class Block:
    """Half-open token spans for attention and representation extraction.

    Content spans exclude headers and separators. For Decision, the extraction
    span instead selects the last token of the marker itself.
    """

    role: str
    start: int
    stop: int
    content_start: int
    content_stop: int
    question: int | None = None
    option: int | None = None


@dataclass
class EncodedExample:
    """Unpadded tokens and block boundaries for inspection and extraction."""

    example: Example
    input_ids: list[int]
    blocks: list[Block]
    positions: list[int]


@dataclass
class PackedBatch:
    """CPU tensors; causal batches omit the dense allowed_attention mask.

    For independent branches, True permits a query to read a key.
    """

    samples: list[EncodedExample]
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    padding_mask: torch.Tensor
    allowed_attention: torch.Tensor | None
    config: InputConfig

    @property
    def lengths(self) -> list[int]:
        """Physical lengths include branches, markers and chat wrappers."""
        return [len(sample.input_ids) for sample in self.samples]


def can_read(query: Block, key: Block, mode: AttentionMode) -> bool:
    """Describe block ancestry; token-level causal order is applied separately."""
    if mode == "causal" or query.role == "suffix":
        return True
    if key.role in {"prefix", "context"}:
        return True
    if query.question is None or query.question != key.question:
        return False
    if query.role == "decision":
        return key.role in {"question", "option", "decision"}
    if key.role == "question":
        return True
    return query.role == key.role == "option" and (
        mode == "independent_questions" or query.option == key.option
    )


def build_mask(blocks: list[Block], length: int, mode: AttentionMode) -> torch.Tensor:
    """Build a dense reference mask; optimized paths must preserve its meaning."""
    allowed = torch.zeros((length, length), dtype=torch.bool)
    for query in blocks:
        for key in blocks:
            if key.start > query.start:
                break
            if query is key:
                allowed[query.start : query.stop, key.start : key.stop] = torch.ones(
                    (query.stop - query.start, key.stop - key.start),
                    dtype=torch.bool,
                ).tril()
            elif can_read(query, key, mode):
                allowed[query.start : query.stop, key.start : key.stop] = True
    return allowed


def build_positions(blocks: list[Block], length: int, mode: AttentionMode) -> list[int]:
    """Reset positions at the same branches isolated by the attention mask."""
    if mode == "causal":
        return list(range(length))
    positions = [0] * length
    shared_end, maximum = 0, 0
    question_end, branch_end = {}, {}
    for block in blocks:
        size = block.stop - block.start
        if block.role in {"prefix", "context"}:
            start = shared_end
            shared_end += size
        elif block.role == "question":
            start = shared_end
            question_end[block.question] = branch_end[block.question] = start + size
        elif block.role == "option":
            start = (
                question_end[block.question]
                if mode == "independent_options"
                else branch_end[block.question]
            )
            branch_end[block.question] = max(branch_end[block.question], start + size)
        elif block.role == "decision":
            # Decision follows the longest option path, not the packed length.
            start = branch_end[block.question]
            branch_end[block.question] = start + size
        else:
            start = maximum
        positions[block.start : block.stop] = range(start, start + size)
        maximum = max(maximum, start + size)
    return positions


class InputBuilder:
    """Tokenize one example or a batch without loading model weights."""

    def __init__(self, tokenizer, config: InputConfig | None = None):
        self.tokenizer = tokenizer
        self.config = config or InputConfig()
        if self.config.attention not in {
            "causal",
            "independent_questions",
            "independent_options",
        }:
            raise ValueError(f"Unknown attention mode: {self.config.attention}")
        if self.config.prompt_format not in {"plain", "chat"}:
            raise ValueError(f"Unknown prompt format: {self.config.prompt_format}")
        if self.config.query_source not in {"question", "decision"}:
            raise ValueError(f"Unknown query source: {self.config.query_source}")

        # The chat wrapper depends on the tokenizer, so resolve it once per builder.
        self._prefix, self._suffix = "", ""
        if self.config.prompt_format == "chat":
            marker = "__DECISION_CONTENT_PLACEHOLDER__"
            template = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": marker}],
                tokenize=False,
                add_generation_prompt=False,
            )
            if template.count(marker) != 1:
                raise ValueError(
                    "Chat template must preserve user content exactly once"
                )
            self._prefix, self._suffix = template.split(marker)

        # Reuse constant segments; example content is tokenized in encode().
        self._header_ids = {
            text: self.tokenizer.encode(text, add_special_tokens=False)
            for text in {
                self._prefix, self._suffix, "Context:\n", "Question:\n",
                "Option:\n", "Decision:",
            }
        }
        self._separator_ids = self.tokenizer.encode("\n\n", add_special_tokens=False)

    def encode(self, example: Example, *, check_length: bool = True) -> EncodedExample:
        """Tokenize segments; disable the budget check only for length inspection."""
        ids, blocks = [], []

        def append(role, header, content="", question=None, option=None):
            start = len(ids)
            # Segment tokenization is deliberate and part of the input convention.
            ids.extend(self._header_ids[header])
            content_start = len(ids)
            ids.extend(self.tokenizer.encode(content, add_special_tokens=False))
            content_stop = len(ids)
            if role in {"question", "option"} and content_start == content_stop:
                raise ValueError(
                    f"Example {example.id}: empty tokenized {role} content "
                    f"(question={question}, option={option}; indices are zero-based)"
                )
            # Decision pooling selects the last marker token, before the separator.
            if role == "decision":
                content_start, content_stop = len(ids) - 1, len(ids)
            if role in {"context", "question", "option", "decision"}:
                ids.extend(self._separator_ids)
            if len(ids) > start:
                blocks.append(
                    Block(
                        role,
                        start,
                        len(ids),
                        content_start,
                        content_stop,
                        question,
                        option,
                    )
                )

        append("prefix", self._prefix)
        append("context", "Context:\n", example.context)
        for q, question in enumerate(example.questions):
            append("question", "Question:\n", question.text, q)
            for o, option in enumerate(question.options):
                text = option.text
                if option.criteria is not None:
                    text += "\nCriteria:\n" + option.criteria
                append("option", "Option:\n", text, q, o)
            if self.config.query_source == "decision":
                append("decision", "Decision:", question=q)
        append("suffix", self._suffix)
        if check_length and len(ids) > self.config.max_packed_tokens:
            raise InputTooLongError(
                f"Example {example.id}: {len(ids)} packed tokens exceed "
                f"max_packed_tokens={self.config.max_packed_tokens}; input was not truncated"
            )
        return EncodedExample(
            example,
            ids,
            blocks,
            build_positions(blocks, len(ids), self.config.attention),
        )

    def __call__(self, examples: Example | list[Example]) -> PackedBatch:
        """Right-pad inputs; real queries cannot attend to padding."""
        samples = [
            self.encode(example)
            for example in ([examples] if isinstance(examples, Example) else examples)
        ]
        if not samples:
            raise ValueError("Cannot build an empty batch")
        return self.pack(samples)

    def pack(self, samples: list[EncodedExample]) -> PackedBatch:
        """Pad already encoded examples, allowing callers to filter before batching."""
        length = max(len(sample.input_ids) for sample in samples)
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError("Tokenizer needs a padding or EOS token")
        ids = torch.full((len(samples), length), pad, dtype=torch.long)
        positions = torch.zeros_like(ids)
        padding = torch.zeros_like(ids, dtype=torch.bool)
        allowed = (
            None if self.config.attention == "causal"
            else torch.zeros((len(samples), 1, length, length), dtype=torch.bool)
        )
        for b, sample in enumerate(samples):
            n = len(sample.input_ids)
            ids[b, :n] = torch.tensor(sample.input_ids)
            positions[b, :n] = torch.tensor(sample.positions)
            padding[b, :n] = True
            if allowed is not None:
                allowed[b, 0, :n, :n] = build_mask(sample.blocks, n, self.config.attention)
                # Self-edges for discarded pad queries avoid fully masked rows and NaNs.
                for p in range(n, length):
                    allowed[b, 0, p, p] = True
        return PackedBatch(samples, ids, positions, padding, allowed, self.config)
