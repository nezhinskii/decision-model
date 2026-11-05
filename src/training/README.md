# Head training

- `train_head.py`: typed CLI, cached datasets, head/optimizer construction and run lifecycle.
- `loop.py`: question-weighted optimization and validation shared across head types.
- `objectives.py`: CE/Brier objective, hard-target smoothing and detached training reports.
- `checkpoints.py`: atomic `latest.pt` snapshots with optimizer/RNG/data progress.
- `heads/checkpoints.py`: portable best-model safetensors weights.

## Train

```bash
uv run python -m src.training.train_head \
  --cache artifacts/features/qwen35_4b_causal_2048 \
  --name pointer_seed42 \
  --head.processor identity \
  --batch-size 16 --epochs 5
```

For Joint controls use `--head.processor joint --head.num-slots 0`; the slot
variant uses `--head.num-slots 64`. Disable cross-attention with
`--head.no-use-cross-attention`. Cache context memory is loaded only when needed.
BF16 mixed precision is the default. Use `--cpu --mixed-precision no` for
CPU FP32 runs, or `--mixed-precision no` for a GPU FP32 reference.
No LLM is instantiated during head training.

Optimization uses AdamW with a constant learning rate, question-weighted CE,
gradient clipping and optional `--accumulation-steps`. A partial final accumulation
group is normalized by its actual question count. Soft-target accuracy compares
argmax labels (ties use the first index); CE/Brier evaluate full distributions.
Brier sums squared errors across options, then averages across questions.

`--log-every` (25), `--validate-every` (500) and `--save-every` (500) count optimizer steps. Validation also runs
at epoch/limited-run end. `--max-steps` limits total steps, including resumed ones.
Best model selection uses validation loss; test data is not read during training.

Early stopping uses original-target validation CE. Defaults are
`--early-stopping.patience 8 --early-stopping.min-delta 0.0001`: eight distinct
validation steps without an absolute improvement greater than 1e-4. Small changes
accumulate relative to the last significant best value. Periodic and end-of-epoch
validation at the same step count once. With 777 updates/epoch and validation every
400 updates plus epoch end, eight checks give roughly 2–3 epochs of patience.
Set `--early-stopping.patience 0` to disable. Patience/minimum/last step and stop
state persist in latest.pt. Stopping saves the latest state and finishes normally;
best.safetensors remains the model with the smallest raw validation CE, even if
its improvement was below min_delta. Repeating an already early-stopped run does
not start updating weights again. Early-stopping configuration must match on
resume; use a new run for a different stopping policy. Cosine retains its original
horizon when early stopping ends a run before the requested epoch limit.

### Training objective

`--loss.ce-weight 1 --loss.brier-weight 0` preserves the original CE training.
Weights are nonnegative and not normalized; at least one must be positive.
For a mixture use `--loss.ce-weight 1 --loss.brier-weight 0.5`, or use weights
`0` and `1` for Brier-only training.

`--loss.label-smoothing 0.05` replaces hard targets with
`(1-epsilon)*target + epsilon/K`, separately for each question's K options.
Soft targets are preserved. `--loss.smoothing-scope both` (default) applies the
smoothed target to both components. `--loss.smoothing-scope ce` smooths CE only,
keeping Brier against the original target. With CE disabled, scope=ce makes
smoothing irrelevant to the optimized objective.

Train logs include `objective` (weighted training loss), `ce_component` and
`brier_component` (before weighting). The existing `all/loss`, `all/brier`, and
accuracy remain computed against original targets, as do all validation metrics.
Best weights are still selected by original-target validation CE. Objective sums
are weighted by question count across accumulation/logging intervals. Loss config
is saved with the run and must match on resume; changing it requires a new name.

## Local outputs and resume

```text
artifacts/runs/<name>/
  config.json                 # full initial typed config
  resume_config.json          # most recent continuation config, if any
  provenance.json             # dimensions, code/environment, parameter count
  cache_summary.json          # extraction settings, source hashes and split totals
  data_summary.json           # source, selection, category/task totals and token budget
  tracking.json               # local and optional remote MLflow IDs
  best.json                   # selected weights path/hash, step and validation metrics
  checkpoints/best.safetensors
  latest.pt                   # model, optimizer, scaler, RNG and data cursor
```

Repeat the original training command to resume automatically; epochs, max steps and
logging/checkpoint intervals may change. Model/data/optimizer settings must match.
`latest.pt` restores shuffle epoch, consumed microbatch cursor and RNG states.
The single-process loader uses zero workers and a separate RNG for iterator
creation, so skipping consumed batches does not consume model/dropout RNG.
Snapshots are taken only at completed optimizer-step boundaries. Interruption
inside a step leaves the preceding periodic snapshot intact; unfinished work is
replayed. CPU resume with dropout and accumulation is checked against uninterrupted
training; exact CUDA reproducibility still depends on the backend.

Best weights and latest state are local only. Each uses a fixed filename;
a temporary file replaces the previous version after writing finishes.
No checkpoint history is retained. Previous UUID-named best exports from older
runs are removed on the next best-model update.
The trainer state uses trusted-local PyTorch pickle; never load arbitrary external
`latest.pt` files. Constant LR remains compatible with old checkpoints.

For a new cosine run use `--schedule cosine --min-lr-ratio 0.1`. LR starts at
`--learning-rate`, declines smoothly to 10% and stays there. No warmup/restarts
are applied. Scheduler state is saved in `latest.pt`, and advances only after
successful optimizer updates. Logged LR is the rate used for the last completed
update in the interval. The horizon defaults to epochs times updates per epoch,
including partial batches/accumulation groups. `--max-steps` pauses training
without shortening that horizon. Set `--schedule-steps N` for an explicit horizon.
Resuming must preserve it and the minimum ratio; changing epochs with an automatic
cosine horizon raises an error. Use new run names when changing schedules.
Training metrics are question-weighted averages over logging intervals; category
breakdowns are computed during validation. Speed is wall time over each interval,
including data loading but excluding validation/checkpoint pauses. CUDA peak
allocated memory covers that interval. Metrics transfer to CPU only when logged;
there are no explicit per-step CUDA synchronizations for timing. Small logging
overhead remains. Partial intervals are flushed before validation, saving and at
epoch end. Full manifests remain with their sources; their hashes detect changes.
Existing runs with full cache manifests remain readable.

For 9,985 training contexts, batch size 8 gives 1,249 steps per epoch and batch
size 16 gives 625 (without accumulation). These defaults yield roughly 25–50
training log points per epoch and validation every 500 steps plus epoch end.
An existing nonempty run directory without `latest.pt` raises an error; use a new
name rather than mixing histories. `--no-resume` rejects an existing run.

## Evaluation and experiment tracking

Standalone evaluation has explicit head/LM commands with a shared run lifecycle:
see [evaluation](../evaluation/README.md). Training validation remains an internal
`loop.validate` call for early stopping and best-checkpoint selection; it creates
no separate evaluation run or prediction journal.

MLflow UI, DagsHub synchronization, retry and run exports are documented in
[experiments](../experiments/README.md).

### Metric and loss calculation

`evaluation/metrics.py` has explicit tensor inputs, without training batches:
`question_log_probs(logits, option_to_question, num_questions)` normalizes once;
`cross_entropy`, `brier_score` and `accuracy` each compute one quantity from those
log probabilities and targets. `question_metrics` explicitly requests all three
as a detached `[Q, 3]` report (CE, accuracy, Brier).

`training/objectives.py:training_values` returns named fields: `objective [Q]`,
`metrics [Q, 3]`, `ce_component [Q]`, `brier_component [Q]`. Only `objective` retains
a gradient graph. Disabled loss components are still logged, but computed without
autograd. With no smoothing their values are reused for the original-target report;
with smoothing only the quantities whose targets differ are recomputed. Training
loss and reporting share the same CE/Brier functions. The six-column packing exists
only when summing values for existing training logs, in `TRAIN_METRIC_NAMES` order.
