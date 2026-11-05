"""Versioned text-only multiple-choice prompts and candidate token addresses."""

from string import ascii_uppercase, ascii_lowercase

PROMPT_VERSION = "native_choice_v1"
ANSWER_SUFFIX = "\n\n"
LABELS = ascii_uppercase + ascii_lowercase


def render_question(context, question):
    if not 1 <= len(question.options) <= len(LABELS):
        raise ValueError(f"Expected 1..{len(LABELS)} options: {question.id}")
    labels = LABELS[:len(question.options)]
    options = []
    for label, option in zip(labels, question.options):
        text = f"{label}: {option.text}"
        if option.criteria is not None:
            text += f"\nCriteria: {option.criteria}"
        options.append(text)
    return (f"Context:\n{context}\n\nQuestion:\n{question.text}\n\nOptions:\n"
            + "\n\n".join(options)
            + "\n\nSelect the best option according to the question and any criteria. "
              "Your final answer must contain only the option label, with no explanation.")


def candidate_tokens(tokenizer, prefix, count, prefix_ids=None):
    """Validate continuation tokenization, including the answer boundary."""
    before = tokenizer.encode(prefix, add_special_tokens=False)
    if prefix_ids is not None and before != prefix_ids:
        raise ValueError("Answer prefix does not round-trip through tokenizer")
    ids = []
    for label in LABELS[:count]:
        after = tokenizer.encode(prefix + label, add_special_tokens=False)
        if after[:-1] != before or len(after) != len(before) + 1:
            raise ValueError(f"Label {label!r} is not a single continuation token")
        ids.append(after[-1])
    if len(set(ids)) != count:
        raise ValueError("Candidate labels must have distinct token IDs")
    return ids


def build_prompt(tokenizer, context, question, thinking):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": render_question(context, question)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    # This runner deliberately supports the explicit Qwen thinking boundary.
    # Fail rather than silently treating an unsupported template as another mode.
    expected = "<think>\n" if thinking else "</think>" + ANSWER_SUFFIX
    if not text.endswith(expected):
        raise ValueError("Chat template does not expose the expected thinking boundary")
    return text
