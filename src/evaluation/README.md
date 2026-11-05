# Decision evaluation

Two explicit commands evaluate custom heads and native language models:

```bash
uv run python -m src.evaluation.evaluate_head \
  --run artifacts/runs/pointer_seed42 \
  --name pointer_seed42_test --split test

uv run python -m src.evaluation.evaluate_lm \
  --dataset artifacts/data/prepared/research_50_2_2_2048 \
  --name qwen35_direct_validation --split validation \
  --backbone.model-id Qwen/Qwen3.5-4B \
  --backbone.revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --inference.mode direct
```

For reasoning, use a new name and `--inference.mode thinking
--inference.max-reasoning-tokens 2048`. Use `evaluate_head --help` or `evaluate_lm --help` for
source-specific arguments. Both commands default to validation; freeze the
protocol there before running test. Heads use their saved mixed-precision config;
pass `--cpu` for CPU evaluation. Native LM loading options are under `--backbone`.

## Responsibilities

- `evaluate_head.py`: head-specific configuration and command.
- `evaluate_lm.py`: native LM configuration and command.
- `config.py`: shared run options only.
- `run.py`: one run lifecycle, reference matching, resume, tracking and reports.
- `cached_head.py`: read saved head settings, verify best weights/cache, batch
  cached features and return option scores. No training CLI is imported.
- `native_lm.py`: read prepared questions and call the native LM implementation.
- `predictions.py`: durable question journal and incremental report accumulator.
- `metrics.py`: independent tensor math reused by training losses and validation.

Both commands use the same `run.py` lifecycle; it does not select or import models.
The source adapters implement `batches()`, `rows(batch)` and `predict(batch)`.
`rows` describes question IDs/options/targets without loading model weights;
`predict` returns aligned scores/statuses. The shared runner chooses which rows
need work. The adapters do not own MLflow, output files or metric accumulation.
No base class or plugin registry is required.

During head training, `training/loop.py:validate` computes validation metrics for
checkpoint selection. Standalone evaluation uses the shared runner instead: each
new prediction updates the report exactly once. Reading a completed report does
not rerun the model or replay all predictions.

## Outputs, resume and tracking

Both commands write independent evaluation runs under:

```text
artifacts/runs/<name>/
  config.json
  provenance.json
  cache_summary.json     # cached head
  data_summary.json      # native LM; head if available from training
  tracking.json
  predictions.jsonl
  evaluation.json
```

Head evaluations link `source_train_run_id` and the exact checkpoint checksum.
The original training directory is unchanged. Existing training checkpoints,
configs and feature caches require no conversion. Old evaluation files remain
readable, but old evaluation CLI paths have been replaced by this command; use a
new name for runs made with the earlier evaluator configuration format.

The prediction journal and reference files are read line by line. RAM retains
only question IDs and alignment metadata (options, targets, categories); scores
and reasoning stay on disk. Resume reconstructs metrics by streaming saved rows.

Repeat a command to resume its journal. A newline commits one complete question;
only an interrupted final line is discarded. Source identity and configuration
must match, including split/reference checksums or the head/cache identity.
Moving Hub revisions are pinned on first creation and reused on resume. Local
checkpoint directories must remain immutable. Only one process may write a name.
A fully completed native run does not load weights; a head verifies its weight
checksum but does not load the model. A partially completed head batch may be
computed again, while already recorded answers remain unchanged.

`--log-every 25` logs cumulative metrics every 25 new question attempts and at
completion. Metric accumulation is incremental. Original-target CE, Brier,
accuracy and category/hard/soft breakdowns use the same formulas as training.
Accuracy includes failures as incorrect; CE/Brier and `completed_accuracy` cover
successful answers. Coverage and status counts are explicit. `random_accuracy`
is the mean of `1/K`, and the report includes the option-count histogram.
No inputs are silently truncated. A recorded budget failure is a completed
attempt: change the budget under a new run name to retry it.

Add `--remote-uri URL` for existing background MLflow synchronization. Config,
provenance, predictions and final reports are artifacts locally and remotely;
weights and caches are not uploaded. Credentials, standalone retry, local UI and
exports are described in [experiments](../experiments/README.md).

## Compare the same questions

To match a native evaluation to exactly the questions retained in a head cache:

```bash
--reference-predictions artifacts/runs/pointer_seed42_test/predictions.jsonl
```

This option works for both sources. The runner checks context/question IDs,
option order, targets and category/task metadata; missing or mismatched IDs fail.
It evaluates only referenced questions. Formatting can still make a native input
exceed its input budget, which is reported through coverage and failure counts.

Native prompts and reasoning details are documented in
[native LM](../native_lm/README.md). A native-inference comparison changes input
formatting as well as the output head, so it does not isolate only the projection.
