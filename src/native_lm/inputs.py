"""Versioned text-only multiple-choice prompts and candidate token addresses."""

from string import ascii_uppercase, ascii_lowercase

PROMPT_VERSION = "native_choice_v1"
ANSWER_SUFFIX = "\n\n"
LABELS = ascii_uppercase + ascii_lowercase


def option_labels(count):
    """Keep existing letter prompts; use fixed-width numbers for larger sets."""
    if count < 1:
        raise ValueError("Questions must have at least one option")
    if count <= len(LABELS):
        return list(LABELS[:count])
    width = len(str(count))
    return [str(i).zfill(width) for i in range(1, count + 1)]


def render_question(context, question):
    labels = option_labels(len(question.options))
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


def candidate_sequences(tokenizer, prefix, count, prefix_ids=None):
    """Tokenize labels at the answer boundary without re-encoding generated text."""
    if prefix_ids is not None:
        # Generated token sequences need not survive decode -> encode unchanged.
        # The special </think> boundary isolates label tokenization from arbitrary
        # reasoning. Verify that exact boundary in the ORIGINAL IDs instead of
        # requiring the entire generated reasoning to have a canonical encoding.
        prefix = "</think>" + ANSWER_SUFFIX
        boundary = tokenizer.encode(prefix, add_special_tokens=False)
        if not boundary or prefix_ids[-len(boundary):] != boundary:
            raise ValueError("Answer token prefix does not end with the expected boundary")
    before = tokenizer.encode(prefix, add_special_tokens=False)
    sequences = []
    for label in option_labels(count):
        after = tokenizer.encode(prefix + label, add_special_tokens=False)
        if after[:len(before)] != before or len(after) <= len(before):
            raise ValueError(f"Label {label!r} is not a valid continuation")
        sequences.append(tuple(after[len(before):]))
    if len(set(sequences)) != count:
        raise ValueError("Candidate labels must have distinct token sequences")
    # Labels of equal character width have no textual prefix ambiguity. Verify
    # that the tokenizer preserves that property before interpreting their mass.
    ordered = sorted(sequences)
    if any(right[:len(left)] == left for left, right in zip(ordered, ordered[1:])):
        raise ValueError("One candidate token sequence is a prefix of another")
    return sequences


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
