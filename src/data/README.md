# Data preparation

- `schema.py`: plain dataclasses for contexts, questions, options and targets;
  explicit Arrow types for nested Parquet storage. No model-level validators.
- `openjev.py`: pinned SFT download and `OpenJevSource` adapter. It owns native
  field conversion; RL is not used. Target alignment is checked on conversion.
- `source.py`: `DatasetSource` contract, shared token-budget filtering and split
  materialization. These algorithms consume common examples, not native rows.
- `sampling.py`: deterministic category quotas, stratified selection and assignment
  of whole context groups to research splits.
- `prepare.py`: CLI orchestration, grouping, Parquet output and manifest generation.
- `dataset.py`: streaming Parquet reader and `DatasetWriter` with bounded Arrow
  buffers and running split statistics.

## Run

```bash
uv run python -m src.data.prepare
uv run python -m src.data.prepare --help
```

Defaults: seed `42`, dataset name `research_v1`, question budgets `10000/1000/1000`.
Use `--train-size`, `--validation-size`, `--test-size` and `--seed` to change the
recipe; use a new `--name` to keep a changed dataset separate from an existing one.

## Token budget

Provide a tokenizer to build a length-limited dataset. Only configuration and
tokenizer files are loaded, never model weights:

```bash
uv run python -m src.data.prepare \
  --name research_10_1_1_2048 \
  --tokenizer-id Qwen/Qwen3-4B-Instruct-2507 \
  --max-tokens 2048 \
  --token-reserve 64
```

Pass `--tokenizer-revision COMMIT_SHA` to pin the tokenizer explicitly, or
`--local-files-only` to load it from the local cache only. Hub files use
`artifacts/models/hub`. Without `--tokenizer-id`, length filtering is disabled.

Lengths use plain text with headers, criteria and separators, without chat or
Decision markers. The selection budget is `max_tokens - token_reserve` (1984
with the settings above). The reserve leaves room for later prompt formatting;
it is not a guarantee for arbitrary templates. Preparation has no attention,
position or chat settings. Nothing is truncated. Oversized questions are excluded before balancing. For
shared contexts, a seeded greedy pass admits only branches whose combined input
fits the budget. This conservatively bounds the entire eligible group, even if
only some of its questions are later sampled. Questions rejected by that group
budget are counted separately. Contexts remain isolated across dataset splits.

The requested question budgets are then sampled from the eligible pool. Small
categories may still be capped; insufficient total capacity raises an error.
All final grouped records are tokenized again to verify the limit. The manifest
records the resolved tokenizer revision, budget/reserve, before/after counts by
category, individual/group exclusions and maximum selected packed length.
Feature extraction checks the actual prompt and skips oversized context records,
logging their IDs and reasons, if the reserve is insufficient.

## Selection and grouping

- Category quotas are equal where possible. Small categories are capped at their
  available counts, and the remaining budget is redistributed without resampling.
- Within each category, task and hard/soft target proportions follow the eligible
  source pool (after length filtering, when enabled).
- Selected questions sharing a context stay in one split and one output record.
  Only CRLF line endings and outer newlines are normalized for the grouping key;
  indentation and other whitespace are preserved. The first selected context text
  is stored unchanged, together with all selected questions for that context.
- A deterministic greedy assignment aims at split sizes and category/task/target
  quotas. Group isolation takes precedence; actual counts are recorded.
- Source questions not selected are not automatically included with a context.
- No training subsets, full normalized intermediate dataset, stored token IDs or
  text truncation are produced. Template/paraphrase grouping needs a separate policy;
  source task/template IDs are unavailable in this release.

## Files and record format

```text
artifacts/data/
  raw/openjev/<revision>/data/SFT.parquet
  prepared/<dataset_name>/
    train.parquet
    validation.parquet
    test.parquet
    manifest.json
```

Each Parquet row is one shared context:

```text
Example: id (context hash), context, questions[]
  Question: id, source_row, text, category, task, options[], target, score_values
    Option: id, text, criteria (optional input text)
    Target: kind (hard/soft), probabilities (aligned with options)
```

Question IDs contain the source revision, SFT stage and zero-based source row.
Dataset-level provenance lives once in the manifest. `score_values` maps options
to numeric ratings; it is separate from target probabilities. No target is intended
as backbone input. Other datasets can use the same schema through their own adapter.

The manifest is a compact dataset description: source revision, seed, length
filter settings/statistics, requested split sizes and actual counts by category,
task and target kind. IDs and source rows live in Parquet, not duplicated lists
in the manifest. Split checksums identify the data for feature caches.
There is no code fingerprint or automatic recipe reuse: an existing destination
requires a new `--name`. Existing datasets are never overwritten.

## Add a dataset source

Implement the three methods of `source.DatasetSource` (a structural `Protocol`;
inheritance is optional):

- `describe()` returns source identity/revision and source-specific limitations.
- `read_index(batch_size)` returns lightweight `Candidate` records sorted by row.
- `read_examples(batch_size, rows=None)` streams common `Example` records and
  converts only requested questions when `rows` is given.

Each question has a stable unique integer `source_row` matching `Candidate.row`.
Its context ID equals `Candidate.group`; category, task and target kind must agree.
One group represents one shared context. Preserve original question order on every
read, and namespace IDs when combining sources. Source adapters own target
validation and native-file access. The shared pipeline checks index alignment,
missing/duplicate rows, group isolation and token budgets.

Pass an adapter to the existing preparation pipeline:

```python
from src.data.prepare import PrepareConfig, prepare

prepare(PrepareConfig(name="other_research_v1"), source=my_source)
```

Sampling, splits, optional length filtering, Parquet and manifests are shared.
The current CLI defaults to OpenJev; its `--revision` is used only for that default
adapter. A supplied adapter owns its revision and records it through `describe()`.
There is no plugin registry. A second CLI dataset choice can be added explicitly
when its adapter is introduced. Token-length filtering assumes the current
segment-based serializer with additive question-branch costs.

## Memory and source reads

Only selection metadata and numeric length information span the full dataset in
RAM. Texts are read in input batches (`--batch-size`, default 1024). After global
selection/splitting, `source.iter_materialized` requests complete context portions
(`--context-batch-size`, default 128), yields them in sorted context order, and
releases them before reading another portion. No full list of selected examples
or full-split Arrow table is built.

`DatasetWriter` buffers at most the configured context count or roughly 8 MiB of
Arrow data per split, flushing sooner when either threshold is reached. One
context remains indivisible; the largest individual context must fit in RAM.
Length checks and category/task/target/option statistics are updated as examples
are written. Written question totals are checked against the selected index.

OpenJev maps requested row numbers to Parquet row groups, skips irrelevant groups,
and streams relevant groups in batches. Only requested rows become Python objects.
Neighboring rows may still be decoded, and the same row group can be read again for
a later context portion. This trades repeated I/O for bounded text memory, without
SQLite or a temporary copy of the normalized dataset. The staging directory holds
only the final split files until atomic publication. Existing datasets are not
rewritten; new streaming output can have different Parquet checksums/row groups.
