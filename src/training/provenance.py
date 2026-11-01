"""Compact source descriptions; full manifests stay beside their datasets."""

import hashlib
import json
from pathlib import Path


def manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def cache_summary(path, manifest):
    """Keep extraction settings and split totals, omitting per-shard inventories."""
    return {
        "path": str(Path(path).resolve()),
        "manifest_sha256": manifest_hash(manifest),
        "features": manifest.get("features"),
        "settings": manifest.get("settings"),
        "model_revision": manifest.get("model_revision"),
        "source_files": manifest.get("source_files"),
        "splits": {
            name: {k: v for k, v in split.items() if k != "parts"}
            for name, split in manifest.get("splits", {}).items()
        },
    }


def data_summary(path):
    """Preserve source identity, preparation seed, length budget and split sizes."""
    manifest = json.loads((path / "manifest.json").read_text())
    length = manifest.get("length_filter", {})
    return {
        "path": str(path.resolve()),
        "manifest_sha256": manifest_hash(manifest),
        "source": manifest.get("source"),
        "seed": manifest.get("seed"),
        "selection": manifest.get("selection"),
        "length_filter": {
            k: v
            for k, v in length.items()
            if k not in ("by_category", "selected_max_packed_tokens", "policy")
        },
        "splits": {
            name: {
                "sha256": split.get("sha256"),
                **{
                    k: v
                    for k, v in split.get("statistics", split).items()
                    if k
                    in (
                        "contexts",
                        "questions",
                        "options",
                        "categories",
                        "tasks",
                        "targets",
                    )
                },
            }
            for name, split in manifest.get("splits", {}).items()
        },
    }


def check_cache(run, path, manifest):
    """Also accept existing runs that saved the original full manifest."""
    summary = run / "cache_summary.json"
    if summary.exists():
        matches = json.loads(summary.read_text())["manifest_sha256"] == manifest_hash(
            manifest
        )
    else:
        matches = json.loads((run / "cache_manifest.json").read_text()) == manifest
    if not matches:
        raise ValueError("Cache manifest changed since training started")
