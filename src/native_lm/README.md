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
after `</think>\n\n`. No `Answer:` prefix is injected. After reasoning, label tokenization
is checked at the exact closing boundary in the original token IDs; arbitrary
generated reasoning is never decoded and re-encoded for scoring.

Candidate vocabulary logits are returned in the original option order. Metrics
use softmax across valid labels at temperature 1. `label_probability_mass` records
their combined probability before renormalization; this is constrained-choice
scoring, not an evaluation of freely generated answer formatting.

Reasoning returns its native cache to scoring. Only the unprocessed final token
and answer separator are appended; the complete prompt/reasoning is not replayed.
Multi-token labels share cached prefixes. At branches both attention KV and
mutable recurrent state are cloned; depth-first traversal bounds retained copies.
Per-token generation logits are not retained.

Detailed runtime/kernel inspection, memory measurements, timing comparisons and
CPU/CUDA profiling live only in `benchmark.py`. Normal inference does not install
profiling hooks or emit per-question performance diagnostics.

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


For an isolated speed comparison, stop other GPU jobs and run:

```bash
uv run python -m src.native_lm.benchmark
```

This sequentially measures minimal NF4 generation, the project's reasoning helper
with NF4, and minimal BF16 generation. The same pinned student, prompt IDs, FLA
bridge and greedy settings are used. EOS and the closing reasoning token are
suppressed until the fixed 256-token budget in all paths. Each mode gets one
full-length warmup and three measurements; reported speed includes prefill.
There is no scoring, dataset access or MLflow. By default the command also captures
CPU/CUDA profiles of minimal NF4 and BF16 decode after warmup. Results are saved incrementally to
`artifacts/diagnostics/native_generation.json`; model loading is not timed.


To capture only profiles without repeating the speed comparisons:

```bash
uv run python -m src.native_lm.benchmark --profile-only
```

Each minimal mode gets a warmup, then profiles 32 decode steps (prefill and four
extra decode warmup steps are excluded). The profiler is deliberately separate
from throughput measurements because it adds overhead. Alongside the JSON report,
`*_nf4_minimal.operators.txt` and `*_none_minimal.operators.txt` contain CPU/CUDA
tables, while `*.trace.json` files contain timelines for Perfetto/Chrome tracing.
The timeline shows CPU dispatch gaps that operator totals alone cannot reveal.
If CUDA profiling is unavailable (e.g. CUPTI missing), the command explicitly
warns that the trace contains no CUDA events. Use `--no-profile` for speed only.


### Windows: compare compiled BF16 decode

On a 24 GiB GPU, compare BF16 eager decode against full-graph compilation with
static hybrid cache. This targets the many small CUDA launches visible in the
diagnostic traces. It is an opt-in benchmark, not a verified production speedup:

```bash
uv run python -m src.native_lm.benchmark --quantizations none --compile-decode --no-profile --output artifacts/diagnostics/native_generation_compiled.json
```

The first compiled warmup can take several minutes. Compilation and an additional
CUDA graph warmup are excluded from the three measured runs. The report retains
the eager BF16 baseline and adds `none_minimal_compiled`. Full-graph compilation
fails explicitly on graph breaks; unsupported compilation criteria also raise
instead of silently benchmarking eager execution. Recent Transformers with
static linear-attention cache support, working triton-windows and a working
TorchInductor compiler toolchain are required. The actual CUDA graph capture and
speedup depend on the runtime and must be checked on the experiment machine.
This command leaves normal evaluation and its dynamic generation cache unchanged.

The FLA convolution-update dispatcher is marked `torch.compiler.disable`.
During compiled decode, our bridge calls it through a registered PyTorch custom
operation with explicit `cache` mutation and a fake-tensor implementation. Dynamo
can retain that operation inside a full graph without tracing the dispatcher.
Eager execution still calls FLA directly. This removes that specific graph break;
CUDA graph eligibility and GPU performance still require a hardware measurement.
