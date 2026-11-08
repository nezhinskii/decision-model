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
Qwen3.5 loads its text CausalLM without vision weights. The tested runtime and
its limits are recorded below.

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

Runtime diagnostics and benchmarks live in
[`benchmark.py`](benchmark.py).

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


## Execution modes

`--inference.execution eager` is the default. `compiled_decode` is an explicit
alternative for thinking inference on one CUDA device. Quantization and compute
dtype remain independent loading settings; NF4/FP16 is the tested fast setup.
Direct mode uses eager execution and rejects `compiled_decode`, which would not
accelerate its one-shot answer scoring.

```powershell
python -m src.evaluation.evaluate_lm --dataset artifacts/data/prepared/research_50_2_2_2048 --name qwen35_nf4_compiled_validation --backbone.model-id Qwen/Qwen3.5-4B --backbone.revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --backbone.quantization nf4 --backbone.dtype float16 --inference.mode thinking --inference.execution compiled_decode
```

`execution.py` owns the per-model execution controller. It attaches a scoped
forward router once, explicitly compiling single-token reasoning decode with
fullgraph/reduce-overhead. HF generate retains its logits processors and natural
stopping rules. Prefill and answer scoring remain eager; no process-wide compiler
settings or automatic fallback are used. Explicit routing avoids HF quantizer
flags silently skipping the requested NF4 compilation.

One static cache is reset and reused between questions. Its capacity covers the
configured maximum input/reasoning lengths, answer separator and full label
continuation, capped at the model position limit. It grows only if a larger
capacity is requested. This bounds shape changes but can allocate more memory
than a short prompt's dynamic cache. First-call compilation and possible
recompilation after a capacity change count toward inference latency.

Support is checked through model/cache capabilities, not a checkpoint-name
allowlist: decoder-only generate, a compileable StaticCache, one CUDA device and
no CPU/disk offload. Static-cache construction and compiler errors are explicit.
New models/library versions still need a smoke check; compatible interfaces do
not establish numerical equivalence. Qwen-specific convolution compatibility
remains isolated in `kernels.py`. Neither this mode nor its configuration alters
backbone extraction or decision-head training/evaluation.

Execution mode/scope is recorded in native evaluation provenance, source identity
and MLflow tags. Resume cannot switch execution mode. Older runs with no execution
field are treated as eager without rewriting their saved configuration. To compare
modes, use separate run names and an identical question cohort.

## Developer diagnostics

Run from the repository root using the project environment. All diagnostic
logic and offline self-checks live in `benchmark.py`, alongside the native LM
implementation. The default is **one real thinking-to-score smoke call**:

```powershell
python -m src.native_lm.benchmark
```

Default smoke settings: NF4, FP16 compute, static cache, compiled decode,
four options and a 2048-token reasoning budget. Success requires natural
reasoning termination and four finite scores from the original NativeLM scorer.
A reasoning limit is a failed smoke, not a successful scoring check. Weight
loading/first compilation contribute to this call's latency; it is not a warmed
speed measurement. Output: `artifacts/diagnostics/native_smoke.json`.
Use `--weights bf16` (or another single weight mode), `--model-id`, `--revision`,
`--smoke-tokens` and `--output` for deliberate checkpoint-specific checks.

The expensive full matrix requires an explicit flag:

```powershell
python -m src.native_lm.benchmark --full
```

It tests none/NF4 quantization with BF16/FP16/FP32 compute, plus native-vs-torch
reference FP32; dynamic/static cache; eager, compiled decode and compiled all
forwards. Checks cover prefill, identical teacher histories, per-layer cache,
logits/probabilities, original 4/54-label scoring, direct/thinking pipelines and
NativeLM.predict/HF generate integration. Fixed-work reasoning is labeled
separately from natural stopping. Fullgraph errors/OOM/unsupported combinations
are retained while independent cases continue. Missing `cl.exe` marks only the
affected compilation path unsupported; the tool does not configure MSVC or alter
the Windows build environment. Optional `--profile` adds operator tables.
Output: `artifacts/diagnostics/native_benchmark.json`, including its summary.
Temporary baselines are removed on exit. No tensor sidecars or separate summary
files are retained. Full runs can take hours; they are intended for new runtime
versions/checkpoints, not routine experiment execution. Selectors `--weights`,
`--caches`, `--executions` narrow explicit full runs.

Offline checks of the tool itself, including CPU fullgraph routing, hybrid Qwen
cache, multi-token scoring and error isolation:

```powershell
python -m src.native_lm.benchmark --self-test
```

### Validated configuration and limits

Local reference artifacts: `artifacts/diagnostics/native_benchmark.json` (matrix
snapshot) and `native_smoke.json` (completed real inference call). These are
ignored research artifacts, not package dependencies.

- Checkpoint: `Qwen/Qwen3.5-4B`, commit
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Windows, RTX 3090, PyTorch `2.11.0+cu128`, Transformers `5.20.1`,
  FLA `0.6.0`, triton-windows `3.6.0.post26`, bitsandbytes `0.50.2`.
- NF4/FP16 + static cache + compiled decode: approximately 99 tokens/s in the
  warmed fixed-work pipeline. This is a 4B measurement, not a prediction for 27B.
- Real smoke: 483 reasoning tokens, natural closure, then four finite scores;
  3.27 GiB peak allocation. Its 104-second first call included possible compilation.
- Numerical equivalence is approximate. The NF4/FP16 matrix changed one of 384
  teacher-forced next-token decisions; six tested answer decisions matched, with
  answer-distribution TV up to about 0.33%. These are synthetic consistency checks,
  not an accuracy benchmark or proof of equivalence on arbitrary inputs.
- Dynamic-cache compilation and compiled-all paths in that environment required
  unavailable `cl.exe`. Working compiled-decode evidence does not validate them.
- Evaluation now exposes the alternative as `--inference.execution compiled_decode`.
  The shared execution component has offline contract coverage; its current
  integration should be smoke-checked on the experiment GPU after runtime changes.

The smoke and validated eager/dynamic or compiled-decode/static integration paths
call the same `execution.py` component used by evaluation. Experimental cache and
compiled-prefill/all paths still require diagnostic routing and benchmark-local
FLA boundaries. Those experiments remain isolated from normal inference.
