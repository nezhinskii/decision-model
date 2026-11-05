"""Small file primitives shared by data preparation, models and experiment runs."""

import hashlib
import json
from pathlib import Path


def file_hash(path):
    """Stream a file checksum without retaining its contents in memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Atomically replace a JSON document after writing the complete temporary file."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)
