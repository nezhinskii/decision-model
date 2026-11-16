"""Split an existing cache in place without model loading or feature recomputation."""

from dataclasses import dataclass
import json
from pathlib import Path

import tyro
from tqdm.auto import tqdm

from .storage import load_shard, write_shard


@dataclass
class SplitContextConfig:
    cache: Path


def split_contexts(config: SplitContextConfig):
    root = config.cache
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Complete feature preparation before splitting context storage")
    parts = [part for split in manifest["splits"].values() for part in split["parts"]]
    converted = 0
    for part in tqdm(parts, desc="Split context storage", unit="shard"):
        path = root / part["file"]
        payload = load_shard(path)
        if isinstance(payload, dict) and "context_files" in payload:
            for name in payload["context_files"].values():
                if Path(name).name != name or not (path.parent / name).is_file():
                    raise ValueError(f"Missing or invalid context file referenced by {path}")
        else:
            write_shard(path, payload, manifest["features"])
            converted += 1
        del payload
    print(f"Converted {converted} shards. Cache: {root}")
    print("Manifest, feature values and record order are unchanged; existing runs can resume.")


if __name__ == "__main__":
    split_contexts(tyro.cli(SplitContextConfig))
