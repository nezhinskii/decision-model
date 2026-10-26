# Frozen feature caches

- `prepare.py`: extract pooled question/option features from all prepared splits,
  transfer them to CPU and write bounded shards. Model loading is lazy until run.
- `dataset.py`: read shards without Transformers or a backbone instance and collate
  whole contexts into flat training batches with rebased indices.

## Prepare

Choose a checkpoint and pass its commit SHA for reproducible extraction:

```bash
uv run python -m src.feature_cache.prepare \
  --name baseline_v1 \
  --backbone.model-id ORGANIZATION/CHECKPOINT \
  --backbone.revision COMMIT_SHA \
  --inputs.attention causal
```

The checkpoint above is a placeholder. Qwen3.5 requires `causal`; independent
branches currently require full-attention Qwen3/Llama. Default extraction uses
NF4 on CUDA, plain input, question queries, last-token pooling, the final layer,
one context per forward and 256 contexts per shard. Use `--help` for nested
backbone/input/pooling settings, `--layers`, `--batch-size` and `--shard-size`.
`batch_size` counts contexts, which can contain several questions.

Each split has a tqdm progress bar with context count, rate, elapsed time and ETA,
starting from the consumed source count on resume (including skips). `skipped`
counts skipped contexts and `saved_shards` counts durable
parts; progress within the current unsaved part is recomputed after interruption.
Use the opt-in tokenizer-only length analysis in `notebooks/backbone.ipynb` to
choose a packed-token budget before extraction. Batch and shard defaults remain
provisional until GPU throughput and actual shard sizes are measured.

Add `--pooling.include-context` to store context-token features for cross-attention.
The final layer must be requested (for example `--layers -1`). Each context is
stored once as `[context_tokens, hidden_size]` from the final normalized layer,
excluding headers, separators and padding. This can substantially increase cache
size and CPU memory per shard; reduce `--shard-size` as needed.

Inputs exceeding the actual packed-token budget are skipped as whole context
records. IDs, categories and the reason are saved in the shard's `skipped` list;
the manifest reports skipped context/question counts. Other errors (including
malformed inputs and OOM) still stop extraction. No text is truncated. Extraction preserves
the prepared split order and does not yet bucket forward batches by length.

```text
artifacts/features/<name>/
  manifest.json
  train/part-00000.pt
  validation/part-00000.pt
  test/part-00000.pt
```

Each shard contains `records` (one dictionary per retained context) and `skipped`
(rejected context IDs and reasons). Each feature record
contains question features `[Q, R, D]`, option features `[V, R, D]`, local
option-to-question indices, flat float32 target probabilities `[V]`, context ID,
question/option IDs, categories, task types, hard/soft kinds and score values.
Features retain the backbone output dtype. The optional `context` field contains
token features, or `None` when disabled. No raw text or padded rows are stored.

The manifest records extraction settings, source file hashes, resolved model
revision, layer conventions and saved shard counts. Package versions are recorded
for reference only. There is no code fingerprint or cache-shard checksum scan.
Changing comments or reader code does not invalidate a cache. Changes that affect
the meaning of features require a new cache name; they are not detected from code.

Repeat the same command to resume. Completed shards are kept in the named cache
directory and skipped in source order. Only the unfinished shard is recomputed.
Progress counts both retained and rejected source contexts, so skipping does not
shift the restart cursor. Even an all-skipped shard is saved for progress recovery.
Each shard is first written to `.pt.tmp` and then renamed; the manifest is also
replaced atomically after each saved shard. A renamed shard missing from progress
after an interruption is recovered without running the backbone again.

Batch size, new shard size, device and cache paths can change on resume. Model,
dtype/quantization, attention backend, input settings, pooling, layers and source
data must match. The resolved Hub revision is pinned on first creation and reused
when continuing, even if the original revision was omitted or a moving branch.
Local checkpoint files must remain unchanged between runs.

`complete=false` marks an unfinished cache; the training reader rejects it.
After all splits finish, `complete` becomes true. Repeating a completed run does
not load the backbone or access the Hub. Older cache formats need a new name.
Use only one preparation process per named cache at a time.

## Read for head training

```python
from pathlib import Path
from torch.utils.data import DataLoader
from src.feature_cache.dataset import FeatureDataset, collate_features

dataset = FeatureDataset(
    Path("artifacts/features/baseline_v1"), "train", shuffle=True, seed=42,
)
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_features)

for epoch in range(10):
    dataset.set_epoch(epoch)
    for batch in loader:
        batch = batch.to("cuda:0")
        # batch.questions: [Q, R, D]; batch.options: [V, R, D]
        # batch.option_to_question: [V]; batch.targets: [V]
        # A head must normalize scores separately within each question.
```

Use `shuffle=False` for validation/test. Shuffling occurs at shard and context
levels, with one shard resident per worker; it is not a global question shuffle.
Do not pass `shuffle=True` to DataLoader for this iterable dataset. With multiple
workers, leave `persistent_workers=False` so set_epoch reaches new worker copies.
Targets remain distributions per question, not one distribution over the batch.
`question_to_context` indexes the batch's context IDs. No model is instantiated
or downloaded when reading the cache; only the head needs GPU memory.

`batch.contexts` is a list of variable-length context tensors, or `None`.
`batch.to(device)` also moves these tensors. A cross-attention head can select the
memory for question `q` using `contexts[question_to_context[q]]`; padding and its
mask, if required by the head, are not added by this collator.
