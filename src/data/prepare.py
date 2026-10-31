"""Build a fixed research dataset directly from source SFT in one command."""

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import tyro
from transformers import AutoTokenizer
from huggingface_hub import resolve_revision

from ..backbone.inputs import InputBuilder, InputConfig
from . import openjev
from .sampling import select, split_groups
from .schema import Example, arrow_schema, context_id


@dataclass
class PrepareConfig:
    """Question budgets and reproducibility settings for one immutable dataset."""

    artifacts: Path = Path("artifacts")
    name: str = "research_10_1_1"
    revision: str = openjev.REVISION
    seed: int = 42
    train_size: int = 10_000
    validation_size: int = 1_000
    test_size: int = 1_000
    batch_size: int = 1024
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    local_files_only: bool = False
    max_tokens: int = 2048
    token_reserve: int = 64


def file_hash(path: Path) -> str:
    """Hash file contents for source identity and prepared-file integrity."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(examples: list[Example]) -> dict:
    """Count questions separately from shared contexts in every manifest section."""
    categories, tasks, targets, options, per_context = (Counter() for _ in range(5))
    for example in examples:
        per_context[len(example.questions)] += 1
        for question in example.questions:
            categories[question.category] += 1
            tasks[question.task] += 1
            targets[question.target.kind] += 1
            options[len(question.options)] += 1
    return {
        "contexts": len(examples),
        "questions": sum(categories.values()),
        "categories": dict(sorted(categories.items())),
        "tasks": dict(sorted(tasks.items())),
        "targets": dict(sorted(targets.items())),
        "option_counts": dict(sorted(options.items())),
        "questions_per_context": dict(sorted(per_context.items())),
    }


def materialize(
    source: Path, assignment: dict, config: PrepareConfig
) -> dict[str, list[Example]]:
    """Convert only selected rows and merge their questions under shared contexts."""
    lookup = {
        candidate.row: name for name, rows in assignment.items() for candidate in rows
    }
    grouped = {name: {} for name in assignment}
    index = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=config.batch_size):
        for raw in batch.to_pylist():
            if index in lookup:
                name = lookup[index]
                key = context_id(raw["state"])
                if key not in grouped[name]:
                    # Store the first selected source text; only the grouping key is normalized.
                    grouped[name][key] = Example(
                        id=key, context=raw["state"], questions=[]
                    )
                grouped[name][key].questions.append(
                    openjev.convert(raw, index, config.revision)
                )
            index += 1
    return {
        name: sorted(groups.values(), key=lambda example: example.id)
        for name, groups in grouped.items()
    }


def prepare(config: PrepareConfig) -> Path:
    """Select, split, convert and publish without a full normalized intermediate copy."""
    sizes = {
        "train": config.train_size,
        "validation": config.validation_size,
        "test": config.test_size,
    }
    if min(sizes.values()) < 1 or config.batch_size < 1:
        raise ValueError("Split sizes and batch_size must be positive")
    if (
        not config.name
        or config.name in {".", ".."}
        or Path(config.name).name != config.name
    ):
        raise ValueError("Dataset name must be one directory name")
    root = config.artifacts / "data" / "prepared"
    destination = root / config.name
    if destination.exists():
        raise ValueError(f"{destination} already exists; choose a new --name")
    if config.tokenizer_id is None and config.tokenizer_revision is not None:
        raise ValueError("tokenizer_revision requires tokenizer_id")
    if not 0 <= config.token_reserve < config.max_tokens:
        raise ValueError("Require 0 <= token_reserve < max_tokens")
    source = openjev.download(config.artifacts / "data" / "raw", config.revision)
    candidates = openjev.read_index(source, config.batch_size)
    source_count = len(candidates)
    builder, length_filter = None, None
    if config.tokenizer_id is not None:
        from .lengths import filter_candidates

        hub = dict(
            revision=config.tokenizer_revision,
            cache_dir=str(config.artifacts / "models" / "hub"),
            local_files_only=config.local_files_only,
            trust_remote_code=False,
        )
        if not Path(config.tokenizer_id).is_dir():
            hub["revision"] = resolve_revision(
                config.tokenizer_id, revision=config.tokenizer_revision,
                cache_dir=hub["cache_dir"], local_files_only=config.local_files_only,
            ).resolved
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id, **hub)
        # Data selection uses plain text only. Reserve room for later chat
        # wrappers/Decision markers without binding the dataset to an experiment.
        builder = InputBuilder(
            tokenizer,
            InputConfig(
                max_packed_tokens=config.max_tokens - config.token_reserve,
            ),
        )
        candidates, stats = filter_candidates(
            source,
            candidates,
            builder,
            config.revision,
            config.seed,
            config.batch_size,
        )
        length_filter = {
            "tokenizer_id": config.tokenizer_id,
            "revision": hub["revision"],
            "max_tokens": config.max_tokens,
            "token_reserve": config.token_reserve,
            **stats,
        }
        print(
            f"Length filter: {len(candidates)}/{source_count} eligible questions",
            flush=True,
        )
    selected, _ = select(candidates, sum(sizes.values()), config.seed)
    assignment, _ = split_groups(selected, sizes, config.seed)
    examples = materialize(source, assignment, config)
    if builder is not None:
        # Verify the final grouped text, not just individual source rows.
        packed_lengths = [
            len(builder.encode(record).input_ids)
            for records in examples.values()
            for record in records
        ]
        length_filter["selected_max_packed_tokens"] = max(packed_lengths)
    # Group isolation is a scientific invariant, independent of approximate quota matching.
    seen = set()
    for split in examples.values():
        ids = {example.id for example in split}
        if seen & ids:
            raise RuntimeError("A context was assigned to more than one split")
        seen.update(ids)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=root) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        split_manifest = {}
        for name, records in examples.items():
            path = staging / f"{name}.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [asdict(record) for record in records], schema=arrow_schema()
                ),
                path,
                compression="zstd",
            )
            stats = summarize(records)
            split_manifest[name] = {
                "file": path.name,
                "sha256": file_hash(path),
                "statistics": stats,
                "requested_questions": sizes[name],
            }
            print(
                f"{name}: {stats['questions']} questions, {stats['contexts']} contexts",
                flush=True,
            )
        manifest = {
            "seed": config.seed,
            "source": {
                "dataset": openjev.DATASET,
                "revision": config.revision,
                "stage": "SFT",
                "file": str(source),
                "questions": source_count,
            },
            "selection": {
                "category_balance": "equal, capped by source availability; shortages redistributed",
                "within_category": "proportional by task and target kind in the eligible source pool",
            },
            "grouping": "SHA256 of context with CRLF normalized and outer newlines removed; no other whitespace changes",
            "splits": split_manifest,
            "length_filter": length_filter,
            "limitations": [
                "Only selected questions are merged; unselected rows are not added to context groups.",
                "Source task/template IDs are unavailable; paraphrase and template leakage are not certified absent.",
                (
                    "Plain input is limited with a token reserve; actual cache inputs may still be skipped if oversized."
                    if builder is not None
                    else "No token-length filtering was applied."
                ),
            ],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Interrupted writes never publish a partial named dataset.
        staging.rename(destination)
    print(f"Prepared dataset: {destination}", flush=True)
    return destination


def main() -> None:
    """Expose PrepareConfig as a typed command-line interface."""
    prepare(tyro.cli(PrepareConfig))


if __name__ == "__main__":
    main()
