"""Materialize frozen features with restart progress at completed shard boundaries."""

from dataclasses import asdict, dataclass, field, replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from itertools import islice

import torch
import tyro
from tqdm.auto import tqdm
import pyarrow.parquet as pq

from ..backbone.features import PoolingConfig, extract_features
from ..backbone.inputs import InputBuilder, InputConfig, InputTooLongError
from ..backbone.model import BackboneConfig, FrozenBackbone
from ..data.dataset import read_examples


@dataclass
class PrepareConfig:
    """One named cache with explicit checkpoint and immutable extraction settings."""

    name: str
    backbone: BackboneConfig
    dataset: Path = Path("artifacts/data/prepared/research_10_1_1")
    output: Path = Path("artifacts/features")
    inputs: InputConfig = field(default_factory=InputConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    layers: tuple[int, ...] = (-1,)
    batch_size: int = 1
    shard_size: int = 256


def file_hash(path: Path) -> str:
    """Hash source files and completed shards without loading them into RAM."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache_records(features, examples) -> list[dict]:
    """Detach each shared context into independent CPU tensors and aligned labels."""
    records = []
    q_start = o_start = 0
    for index, example in enumerate(examples):
        q_stop = q_start + len(example.questions)
        o_stop = o_start + sum(len(q.options) for q in example.questions)
        # clone() prevents a small slice from retaining the entire batch storage.
        records.append(
            {
                "context_id": example.id,
                "context": (
                    features.contexts[index].detach().cpu().clone()
                    if features.contexts is not None
                    else None
                ),
                "questions": features.questions[q_start:q_stop].detach().cpu().clone(),
                "options": features.options[o_start:o_stop].detach().cpu().clone(),
                "option_to_question": (
                    features.option_to_question[o_start:o_stop] - q_start
                )
                .cpu()
                .clone(),
                "targets": torch.tensor(
                    [p for q in example.questions for p in q.target.probabilities],
                    dtype=torch.float32,
                ),
                "question_ids": [q.id for q in example.questions],
                "option_ids": [o.id for q in example.questions for o in q.options],
                "categories": [q.category for q in example.questions],
                "tasks": [q.task for q in example.questions],
                "target_kinds": [q.target.kind for q in example.questions],
                "score_values": [q.score_values for q in example.questions],
            }
        )
        q_start, o_start = q_stop, o_stop
    return records


def save_manifest(path: Path, manifest: dict) -> None:
    """Replace progress atomically so interruption leaves the previous JSON intact."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def part_info(path: Path, root: Path, payload: dict) -> dict:
    """Progress counts consumed source records, including length skips."""
    records, skipped = payload["records"], payload["skipped"]
    return {
        "file": str(path.relative_to(root)),
        "contexts": len(records),
        "questions": sum(len(r["question_ids"]) for r in records),
        "options": sum(len(r["option_ids"]) for r in records),
        "processed_contexts": len(records) + len(skipped),
        "skipped_contexts": len(skipped),
        "skipped_questions": sum(len(r["question_ids"]) for r in skipped),
    }


def prepare(config: PrepareConfig) -> Path:
    """Continue saved shards in source order; recompute only the unsaved remainder."""
    if (
        not config.name
        or config.name in {".", ".."}
        or Path(config.name).name != config.name
    ):
        raise ValueError("Cache name must be one directory name")
    if min(config.batch_size, config.shard_size) < 1:
        raise ValueError("batch_size and shard_size must be positive")
    source_manifest = json.loads((config.dataset / "manifest.json").read_text())
    source_files = {}
    for split in ("train", "validation", "test"):
        info = source_manifest["splits"][split]
        path = config.dataset / info["file"]
        checksum = file_hash(path)
        if checksum != info["sha256"]:
            raise ValueError(f"Prepared {split} checksum mismatch")
        source_files[split] = {"file": info["file"], "sha256": checksum}

    resolved = json.loads(json.dumps(asdict(config), default=str))
    # Execution details (batch/shard sizes, device, paths) may change on resume.
    settings = {
        "backbone": {
            key: resolved["backbone"][key]
            for key in (
                "model_id",
                "revision",
                "dtype",
                "quantization",
                "attention_backend",
            )
        },
        "inputs": resolved["inputs"],
        "pooling": resolved["pooling"],
        "layers": resolved["layers"],
    }
    destination = config.output / config.name
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        previous_settings = manifest.get("settings")
        # An old unpinned cache may be resumed with an explicitly supplied SHA.
        # This is a user assertion of its origin; never infer it from today's main.
        if (
            not manifest.get("model_revision")
            and config.backbone.revision
            and previous_settings is not None
        ):
            previous_settings = {
                **previous_settings,
                "backbone": {
                    **previous_settings["backbone"],
                    "revision": config.backbone.revision,
                },
            }
        if (
            manifest.get("format_version") != 4
            or previous_settings != settings
            or manifest.get("source_files") != source_files
        ):
            raise ValueError(
                f"{destination} has different settings/data/format; choose a new --name"
            )
        for info in manifest["splits"].values():
            for part in info["parts"]:
                if not (destination / part["file"]).is_file():
                    raise ValueError(f"Saved shard is missing: {part['file']}")
        if manifest["complete"]:
            print(f"Already prepared: {destination}", flush=True)
            return destination
        if (
            not manifest.get("model_revision")
            and not Path(config.backbone.model_id).is_dir()
            and (
                config.backbone.revision is None
                or len(config.backbone.revision) != 40
                or any(c not in "0123456789abcdef" for c in config.backbone.revision)
            )
        ):
            raise ValueError(
                "This unfinished cache has no pinned revision. Supply the original "
                "checkpoint commit with --backbone.revision, or choose a new cache name."
            )
        backbone = FrozenBackbone(
            replace(
                config.backbone,
                revision=manifest.get("model_revision") or config.backbone.revision,
            )
        )
        if not manifest.get("model_revision"):
            manifest["model_revision"] = backbone.describe()["revision"]
            manifest["settings"] = settings
            save_manifest(manifest_path, manifest)
    else:
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(
                f"{destination} has files but no manifest; choose a new --name"
            )
        backbone = FrozenBackbone(config.backbone)
        # Pin a moving Hub revision before any shard is written or resumed.
        checkpoint = backbone.describe()
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {
            "format_version": 4,
            "settings": settings,
            "config": resolved,
            "source_files": source_files,
            "model_revision": checkpoint["revision"],
            "packages": {
                name: version(name)
                for name in ("torch", "transformers", "bitsandbytes", "pyarrow")
            },
            "features": None,
            "complete": False,
            "splits": {
                split: {
                    "contexts": 0,
                    "questions": 0,
                    "options": 0,
                    "processed_contexts": 0,
                    "skipped_contexts": 0,
                    "skipped_questions": 0,
                    "complete": False,
                    "parts": [],
                }
                for split in source_files
            },
        }
        save_manifest(manifest_path, manifest)

    builder = None
    for split, source in source_files.items():
        info = manifest["splits"][split]
        if info["complete"]:
            continue
        (destination / split).mkdir(exist_ok=True)

        def register_part(path, payload):
            part = part_info(path, destination, payload)
            info["parts"].append(part)
            for key in (
                "contexts",
                "questions",
                "options",
                "processed_contexts",
                "skipped_contexts",
                "skipped_questions",
            ):
                info[key] += part[key]
            save_manifest(manifest_path, manifest)
            tqdm.write(
                f"{split}: saved {path.name}, {info['contexts']} kept, "
                f"{info['skipped_contexts']} skipped contexts total"
            )

        # Recover a shard renamed successfully just before a manifest interruption.
        path = destination / split / f"part-{len(info['parts']):05d}.pt"
        while path.exists():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            register_part(path, payload)
            del payload
            path = destination / split / f"part-{len(info['parts']):05d}.pt"

        # A skipped record still advances the source cursor; using only kept
        # counts here would duplicate later questions after restarting.
        consumed = info["processed_contexts"]
        print(f"{split}: continuing after {consumed} processed contexts", flush=True)
        remaining = islice(
            read_examples(config.dataset / source["file"]), consumed, None
        )
        total = pq.ParquetFile(config.dataset / source["file"]).metadata.num_rows
        with tqdm(
            total=total, initial=consumed, desc=split, unit="context"
        ) as progress:
            progress.set_postfix(
                saved_shards=len(info["parts"]), skipped=info["skipped_contexts"]
            )
            while examples := list(islice(remaining, config.shard_size)):
                if builder is None:
                    builder = InputBuilder(backbone.tokenizer, config.inputs)
                records, skipped = [], []
                for start in range(0, len(examples), config.batch_size):
                    chunk = examples[start : start + config.batch_size]
                    encoded = []
                    for example in chunk:
                        try:
                            encoded.append(builder.encode(example))
                        except InputTooLongError as error:
                            # Only length rejection is recoverable. Malformed
                            # inputs, OOM and model errors must remain visible.
                            skipped.append(
                                {
                                    "context_id": example.id,
                                    "question_ids": [q.id for q in example.questions],
                                    "categories": [
                                        q.category for q in example.questions
                                    ],
                                    "reason": str(error),
                                }
                            )
                    if encoded:
                        batch = builder.pack(encoded)
                        output = backbone.forward(batch, layers=config.layers)
                        features = extract_features(output, batch, config.pooling)
                        if manifest["features"] is None:
                            manifest["features"] = features.metadata
                            save_manifest(manifest_path, manifest)
                        records.extend(
                            cache_records(
                                features, [sample.example for sample in encoded]
                            )
                        )
                        del output, features
                    progress.update(len(chunk))
                path = destination / split / f"part-{len(info['parts']):05d}.pt"
                temporary = path.with_suffix(".pt.tmp")
                # Store skips in the same atomic file as features, including an
                # all-skipped shard, so recovery preserves the exact source cursor.
                payload = {"records": records, "skipped": skipped}
                torch.save(payload, temporary)
                temporary.replace(path)
                register_part(path, payload)
                del payload
                progress.set_postfix(
                    saved_shards=len(info["parts"]), skipped=info["skipped_contexts"]
                )
        if not info["contexts"]:
            raise ValueError(f"Prepared split is empty: {split}")
        info["complete"] = True
        save_manifest(manifest_path, manifest)
    manifest["complete"] = True
    save_manifest(manifest_path, manifest)
    print(f"Prepared features: {destination}", flush=True)
    return destination


def main() -> None:
    """Expose extraction and backbone settings through the CLI."""
    prepare(tyro.cli(PrepareConfig))


if __name__ == "__main__":
    main()
