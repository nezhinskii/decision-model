# Head training and experiment tracking

- `train_head.py`: typed CLI, cached datasets, head/optimizer construction and run lifecycle.
- `loop.py`: question-weighted optimization and validation shared across head types.
- `objectives.py`: soft-target CE, argmax accuracy and multiclass Brier; category/hard/soft breakdowns.
- `checkpoints.py`: atomic local `latest.pt` and best-model safetensors exports.
- `tracking.py`: local MLflow first, optional DagsHub mirroring and retry CLI.
- `evaluate.py`: separate linked evaluation runs loading best weights strictly.

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
  evaluations/<id>/           # separately tracked evaluation reports
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
`latest.pt` files. There is no scheduler state because the initial schedule is constant.
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

## Local UI

```bash
uv run mlflow server --backend-store-uri sqlite:///artifacts/mlflow/tracking.db \
  --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`. Metrics, parameters and small reports are in local
MLflow; model/cache files are not uploaded as artifacts. `local_directory` and
`best.json` identify local weights. Local paths shown remotely are references,
not downloadable model files. Keep the database and its adjacent artifacts folder
together when moving the tracking store.

## DagsHub

Create a public repository and use its MLflow endpoint:
`https://dagshub.com/OWNER/REPOSITORY.mlflow`. Set credentials outside the repository:
`MLFLOW_TRACKING_USERNAME` is your username and `MLFLOW_TRACKING_PASSWORD` your token.
Do not put credentials into the URI or saved CLI config.

Pass `--remote-uri URL` to training for synchronization after validation/end,
or retry independently:

```bash
uv run python -m src.training.tracking \
  --run artifacts/runs/pointer_seed42 \
  --remote-uri https://dagshub.com/OWNER/REPOSITORY.mlflow
```

Sync mirrors local parameters, tags, full metric histories and an explicit
allowlist of small JSON files. A source-local-run tag links distinct server IDs.
Existing metric points are deduplicated on retry; only one sync process per run
should be active. Network failures warn without discarding local results. Sync is
synchronous and may wait on network timeouts; it is not a background streaming
service. Remote progress updates at validation intervals, not every batch.
Model weights and caches are never in the upload allowlist.

Resume can replay steps newer than the last snapshot; their earlier metrics remain
in MLflow history rather than being silently deleted. The checkpoint cursor,
not the last logged metric, determines which training state is restored.

## Evaluate

```bash
uv run python -m src.training.evaluate \
  --run artifacts/runs/pointer_seed42 --split test
```

Evaluation checks the loaded weight hash/cache identity, logs current code/device
metadata and links `source_train_run_id`. Add `--remote-uri URL` to mirror this
evaluation run too. Dedicated latency benchmarking and live LoRA training remain
separate future entry points.
