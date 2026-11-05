# Native language-model inference

`model.py` loads the text decoder and its original vocabulary projection, and
produces option scores in direct or thinking mode. `inputs.py` owns the versioned
question prompt and label mapping. Shared revision/NF4/dtype loading settings live
in `backbone/loading.py`. Native inference has no dependency on training or MLflow.

Use the [unified evaluation command](../evaluation/README.md) for datasets,
resume, reports, question matching and tracking. Its `native_lm.py` adapter calls
this package. Defaults are NF4, BF16, SDPA, input budget 2048, greedy reasoning
budget 2048 and one question per model call. Context is repeated across questions;
option criteria are included, while targets are never prompt inputs.

The runner requires a Qwen-style template exposing `<think>`/`</think>` and a
text decoder supporting `logits_to_keep`. Unsupported templates fail explicitly.
Qwen3.5 loads its text CausalLM without vision weights. GPU/NF4 memory and throughput
still need checking on the experiment machine.

## Answer position

Up to 52 options retain the original case-sensitive `A..Z,a..z` labels.
Larger sets use fixed-width numbers (`01..54`, `001..120`, etc.). No option-count
cap is imposed by the label alphabet. Labels are tokenized at the actual answer
boundary and must have distinct, non-prefix token sequences.

Single-token labels use one scoring forward as before. Multi-token labels use
summed autoregressive log probabilities of the complete label, with one forward
per shared token prefix. They are not scored using only their first token and
are not length-normalized. This fallback can be slower for very large sets.
Prompts and scores for previously supported questions remain unchanged, so an
interrupted v1 run can resume without recomputing its completed answers.

Direct uses `enable_thinking=False`: the template already provides a closed empty
reasoning block. Thinking generates until the complete `</think>` token sequence;
the runner then appends exactly two newlines. Both modes score the next token
after `</think>\n\n`. No `Answer:` prefix is injected. After reasoning, tokenization
is checked against the actual generated token sequence before scoring.

Candidate vocabulary logits are returned in the original option order. Metrics
use softmax across valid labels at temperature 1. `label_probability_mass` records
their combined probability before renormalization; this is constrained-choice
scoring, not an evaluation of freely generated answer formatting.

Reasoning uses model generation with its native cache. Scoring performs a fresh
forward over prompt + reasoning and retains vocabulary logits at only the last
position. This trades one additional forward for simple, reliable handling of
hybrid recurrent caches. Per-token generation logits are not retained.

Reasoning text is saved only with `--inference.save-reasoning`; this text then
becomes part of the prediction artifact uploaded to MLflow. Timing includes
reasoning and scoring, excludes initial weight loading, and is summed across
resumed attempts. It is not the complete evaluation wall time.

Internally, `NativeLM.predict` orchestrates three operations in the same class:
`_prepare_input` builds and validates the prompt, `_generate_reasoning` optionally
runs generation up to the closing boundary, and `_score_options` reads candidate
logits. Dataset iteration, output files, reports and tracking remain in evaluation.
