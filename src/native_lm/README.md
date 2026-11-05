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

Reasoning returns its native cache to scoring. Only the unprocessed final token
and answer separator are appended; the complete prompt/reasoning is not replayed.
Multi-token labels share cached prefixes. At branches both attention KV and
mutable recurrent state are cloned; depth-first traversal bounds retained copies.
Per-token generation logits are not retained.

Startup logs report actual 4-bit modules, dtype, attention backend and resolved
hybrid kernel providers, warning when reference PyTorch kernels remain active.
The first three predictions and every 25th print generation/prefill/decode/scoring
times and reused-cache/forward token counts. Phase timings, cache counters and CUDA
memory peaks are saved with predictions and aggregated in evaluation metrics.
Decode throughput excludes the first prefill forward; generation throughput includes
it. Old resumed predictions remain valid but do not enter the new phase rates;
`evaluation/profiled_questions` identifies their coverage. Use a new run name for
an entirely fresh speed measurement.

Reasoning text is saved only with `--inference.save-reasoning`; this text then
becomes part of the prediction artifact uploaded to MLflow. Timing includes
reasoning and scoring, excludes initial weight loading, and is summed across
resumed attempts. It is not the complete evaluation wall time.

Internally, `NativeLM.predict` orchestrates three operations in the same class:
`_prepare_input` builds and validates the prompt, `_generate_reasoning` optionally
runs generation up to the closing boundary, and `_score_options` reads candidate
logits. Dataset iteration, output files, reports and tracking remain in evaluation.

`kernels.py` contains a compatibility monkey patch for Qwen3.5 convolutions:
Transformers normally discovers Dao kernels, not FLA convolutions. On CUDA,
reference convolution calls are bridged to installed FLA Triton kernels; existing
optimized Dao kernels and CPU behavior are preserved. The bridge changes layout
and argument conventions, retaining the same mutable convolution-state buffer.
Its module-level scope and removal conditions are documented in the code.
Runtime logs identify it explicitly as `[triton bridge]`. Numerical agreement is
validated within tolerances, rather than requiring bit-identical output.
