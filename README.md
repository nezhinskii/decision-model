# Decision readout research

Research on direct choice over candidate answers using language-model hidden states.
Research notes and the technical specification are in [docs/](docs/) (Russian).

## Setup

Install `uv` and an NVIDIA driver for GPU workloads, then run:

```bash
uv sync --locked
```

The same installation includes GPU kernels, notebook tools and pytest. For
notebooks, select the project `.venv` kernel; no dependency groups are required.

Dependencies are selected automatically for Linux and Windows (x86-64).
Linux uses FLA's CUDA extra and standard Triton; Windows uses FLA with
`triton-windows` 3.6, matching PyTorch 2.11. Use the same `uv sync --locked`
command on either platform. Windows kernel execution still depends on the GPU,
driver and the Windows Triton runtime; resolving dependencies does not validate
kernel execution.

## Checks

Fast offline CPU checks:

```bash
uv run python -m pytest tests -q
```

They use tiny synthetic data and temporary files, without downloading checkpoints
or reading research artifacts. GPU performance is measured separately.

## Data preparation

```bash
uv run python -m src.data.prepare
```

One command selects a balanced research set from OpenJevData **SFT**, splits it by
shared context, and converts selected rows to the common format. Default budgets
are **10,000 / 1,000 / 1,000 questions** for train / validation / test. There are no
training subsets. See [src/data/README.md](src/data/README.md) for the selection
policy, storage format and manifest contents.

## Feature preparation

The feature extraction command writes pooled backbone outputs and targets in
shards for head training without loading the backbone again. See
[src/feature_cache/README.md](src/feature_cache/README.md) for checkpoint settings, commands
and a cache-loading example.

## Repository layout

```text
src/
  features.py            # shared feature tensors and links consumed by heads
  data/                  # common schema, source adapter, sampling and preparation
  backbone/              # packed inputs, lazy frozen loading and feature pooling
  feature_cache/         # feature cache preparation and training-batch loading
  heads/                 # shared projections, processors and Pointer scoring
notebooks/
  backbone.ipynb          # offline mask/shape checks; optional pretrained usage
docs/                    # research plans and technical agreements
pyproject.toml
uv.lock
```

See [the head guide](src/heads/README.md) for processors, ablations and cache usage.
Training, resume, evaluation and local/remote MLflow commands are described in
[the training guide](src/training/README.md).
See [the backbone guide](src/backbone/README.md) for input conventions and lazy
loading. The notebook runs without downloads by default; pretrained loading is
explicitly opt-in, and the main research checkpoint has not been selected yet.

Generated files stay outside Git. Directories are created when used:

```text
artifacts/
  models/hub/                  # shared Hugging Face config/tokenizer/weight cache
  data/
    raw/openjev/<revision>/     # source SFT, description and license notices
    prepared/<dataset_name>/   # train/validation/test.parquet and manifest.json
  features/<cache_id>/         # feature manifest and shards
  runs/<run_id>/               # resolved config, metrics, reports, checkpoints/
  mlflow/                     # tracking.db and small tracking artifacts
  backups/<snapshot_id>/      # tracking backup snapshots
  notebooks/                 # local executed notebooks and exploration outputs
```
