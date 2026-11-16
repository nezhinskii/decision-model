"""Atomic cache shards with context tensors in separate, depth-specific files.

The semantic manifest and record order are unchanged by storage conversion.
"""

from pathlib import Path

import torch


def load_shard(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _save_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _context_depths(metadata):
    if not metadata.get("pooling", {}).get("include_context", False):
        return ()
    depths = metadata.get("context_layers")
    if depths is None:
        depth = metadata.get("context_layer", metadata.get("num_hidden_layers"))
        if depth is None:
            raise ValueError(
                "Cannot split a context cache without its stored decoder depth"
            )
        depths = [depth]
    return tuple(depths)


def write_shard(path, payload, metadata):
    """Write context files first; atomically publish their references with features.

    Replacing a monolithic shard is safe to retry if interrupted: before the final
    rename its original data is intact. Finished side files may be overwritten on
    retry. The source shard is replaced only after all its contexts are saved.
    """
    path = Path(path)
    records = payload["records"] if isinstance(payload, dict) else payload
    context_files = {}
    for depth in _context_depths(metadata):
        contexts = []
        for record in records:
            by_layer = record.get("contexts_by_layer")
            context = (
                by_layer.get(depth) if by_layer is not None else record.get("context")
            )
            if not isinstance(context, torch.Tensor):
                raise ValueError(f"A cache record is missing context layer {depth}")
            contexts.append(context)
        context_path = path.with_name(f"{path.stem}.context-{depth}.pt")
        _save_atomic(
            context_path,
            {
                "context_ids": [r["context_id"] for r in records],
                "contexts": contexts,
            },
        )
        context_files[depth] = context_path.name
        del contexts
    features = [
        {
            **{k: v for k, v in r.items() if k not in {"context", "contexts_by_layer"}},
            "context": None,
        }
        for r in records
    ]
    result = dict(payload) if isinstance(payload, dict) else {}
    result.update(records=features, context_files=context_files)
    _save_atomic(path, result)


def load_contexts(shard_path, context_files, depth, records):
    """Read one selected depth and verify alignment before batching/shuffling."""
    if depth not in context_files:
        raise ValueError(f"A cache shard is missing context layer {depth}")
    name = context_files[depth]
    # Context references always name siblings of the feature shard.
    if Path(name).name != name:
        raise ValueError("Invalid context shard reference")
    payload = load_shard(Path(shard_path).parent / name)
    contexts = payload["contexts"]
    if payload["context_ids"] != [r["context_id"] for r in records] or len(
        contexts
    ) != len(records):
        raise ValueError("Context shard records are not aligned with feature records")
    if any(not isinstance(context, torch.Tensor) for context in contexts):
        raise ValueError(f"A cache record is missing context layer {depth}")
    return contexts
