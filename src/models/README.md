# Live decision models

`decision.py` composes a registered text decoder, existing pooling and any existing
`HeadConfig`. `checkpoints.py` saves adapters and the head together, without frozen
base weights. Model code does not read datasets, run optimizers or manage MLflow.

The live path uses Transformers + bitsandbytes + PEFT. NF4 is the backbone default
(QLoRA); `quantization=none` selects ordinary LoRA. The decision head is trained in
full, while base decoder parameters remain frozen. No LM vocabulary projection is
loaded. No native-generation convolution patches or compilation modes are installed.

All head processors and last/mean/static/adaptive readouts are available. `model.layers`
selects 1-based backbone depths (`-1` means final); `model.head.layer-indices` selects
indices in that extracted axis. Context memory is enabled automatically for heads
requiring it. By default it comes from the deepest depth actually used by the
readout (`last` uses only its last selected index). For example, extracted depths
16/24/32 with readout indices 0/1 use context from depth 24. Set
`--model.head.context-layer 32` (or `-1` for final) to override this; the requested
context depth must be present in `model.layers`. Intermediate states remain decoder
block outputs; only the true final depth includes final normalization. This changes
feature selection, not decoder execution: all blocks still run, with no truncation.
Pooling and input conventions match frozen feature extraction.
The same `HeadConfig.context_depth` rule is used by cached-head loaders.
`model.pooling.context_layer` remains accepted for compatibility; contradictory
explicit head/pooling depths are rejected.

## Train

Example Qwen3.5-4B QLoRA with an adaptive identity readout:

```powershell
uv run python -m src.training.train_decision `
  --dataset artifacts/data/prepared/research_50_2_2_2048 `
  --name qlora4b_adaptive_s42 `
  --model.backbone.model-id Qwen/Qwen3.5-4B `
  --model.backbone.revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a `
  --model.backbone.device cuda:0 `
  --model.backbone.quantization nf4 `
  --model.backbone.dtype bfloat16 `
  --model.layers 8 16 24 32 `
  --model.head.processor identity `
  --model.head.dim 384 --model.head.score-dim 128 `
  --model.head.readout adaptive `
  --model.lora.rank 16 --model.lora.alpha 32 `
  --batch-size 1 --accumulation-steps 8 `
  --learning-rate 0.00003 --adapter-learning-rate 0.0002 `
  --epochs 3 --schedule cosine
```

`learning-rate` controls the head; `adapter-learning-rate` controls LoRA. Both groups
use the shared scheduler, and both rates are logged. These are initial settings,
not a tuned recipe or a measured VRAM/throughput guarantee.

Use `--model.backbone.quantization none` for BF16/FP16 LoRA. Use
`--model.lora.no-enabled` for a live frozen-backbone control. For a cross head select
`--model.head.processor joint --model.head.use-cross-attention`; slots and all other
head settings keep their existing meaning. `--model.lora.module-names q_proj v_proj`
selects explicit module suffixes; otherwise `target-modules` defaults to PEFT's
`all-linear`. Actual matched modules and parameter counts go to adapter_summary.json.

Gradient checkpointing is disabled by default for throughput; enable non-reentrant
checkpointing with `--model.gradient-checkpointing` if training runs out of VRAM. Intermediate-layer and context
features keep gradients. An early-layer-only readout naturally leaves later adapters
without task gradients; installed/trainable parameter counts are not proof that every
adapter receives gradients on every batch. Real hybrid Qwen/NF4 backward on the user's
GPU remains a separate validation step; no GPU checks are implied by code support.

Only one process/device is supported. The loaded device must match Accelerator.
CPU tests require `--cpu --mixed-precision no --model.backbone.device cpu
--model.backbone.quantization none --model.backbone.dtype float32`.

## Data and resume

Prepared train/validation examples are tokenized once per process into private temporary
files on the system temporary disk. Restarting the process rebuilds these files; they
are not a persistent reusable cache. Files are removed when the dataset is released
(or explicitly closed); abrupt termination may leave temporary files behind.
Token IDs and positions are stored as compact tensors. Preparation buffers at most
128 contexts or 65,536 tokens per shard (an indivisible oversized context is stored
alone). Iteration reads one shard at a time; only the active batch goes to the GPU.
Original texts and rejected examples also live in these shards, not a full RAM list.
Small shard indexes, question IDs for duplicate detection and selection reports still
scale with dataset size. Shuffle uses a separate seeded RNG for shard and within-shard
order. Hidden states are recomputed on every forward. Cached-head training already
reads its feature cache one shard at a time and keeps that existing implementation.

Hub revisions are pinned before config.json is saved and reused on resume. Dataset
checksums, selected/skipped population, tokenizer/input settings, base identity and
PEFT version are checked. Local model files are fingerprinted. Old runs/checkpoints
and feature caches are not modified. Training data and model configuration must remain
unchanged when resuming. Repeat the original command to continue from latest.pt.

Best weights are in one `checkpoints/best.safetensors` with `adapter.*` and `head.*`
namespaces plus configuration/base identity metadata. latest.pt adds optimizer,
scheduler, scaler, RNG and progress using the shared runner's model-state codec.
These are project decision-model bundles, not a standalone Hugging Face adapter repo;
load with this project's evaluation command. The base checkpoint is still required.

## Evaluate

```powershell
uv run python -m src.evaluation.evaluate_decision `
  --run artifacts/runs/qlora4b_adaptive_s42 `
  --name val_qlora4b_adaptive_s42 --split validation --device cuda:0
```

The saved dataset is the default; `--dataset` explicitly selects another prepared
set. Evaluation uses the same journal, metrics, reference cohort and tracking as
other evaluators. Over-budget questions produce context_limit failures and stay in
the accuracy/coverage denominator. Evaluation reloads the immutable base plus both
trained parts; it does not use a frozen feature cache. The training dataset is not
modified. Use validation while selecting methods; keep test sealed until selection.


## User-run checks

The implementation is not accompanied by a claim of a successful GPU run. CPU
contract/integration tests construct local tiny random Qwen decoders and never
fetch pretrained files. After installing the locked dependencies, run:

```powershell
uv sync --locked
uv run --no-sync python -m pytest tests -q
```

An additional real NF4/backward smoke is opt-in. Set an existing local HF model
directory explicitly; the test never downloads a checkpoint:

```powershell
$env:DECISION_LORA_CHECKPOINT = "D:\path\to\local\checkpoint"
uv run --no-sync python -m pytest tests/test_lora_cuda.py -q
```

That smoke uses a small joint/adaptive head and a tiny input but loads the actual
base model. It checks finite loss/gradients and adapter/head updates with gradient
checkpointing. It is not a training-speed benchmark or a full numerical parity study.
