"""Build a fixed research dataset directly from source SFT in one command."""

from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile

import tyro

from ..utils.files import file_hash

from . import openjev
from .sampling import select, split_groups
from .source import DatasetSource, filter_candidates, iter_materialized
from .dataset import DatasetWriter


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
    batch_size: int = 1024  # Maximum input rows decoded per batch.
    context_batch_size: int = 512  # Complete selected contexts held per portion.
    tokenizer_id: str | None = None
    tokenizer_revision: str | None = None
    local_files_only: bool = False
    max_tokens: int = 2048
    token_reserve: int = 64


def prepare(config: PrepareConfig, source: DatasetSource | None = None) -> Path:
    """Select, split, convert and publish without a full normalized intermediate copy."""
    sizes = {
        "train": config.train_size,
        "validation": config.validation_size,
        "test": config.test_size,
    }
    if min(*sizes.values(), config.batch_size, config.context_batch_size) < 1:
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
    if source is None:
        path = openjev.download(config.artifacts / "data" / "raw", config.revision)
        source = openjev.OpenJevSource(path, config.revision)
    candidates = source.read_index(config.batch_size)
    source_count = len(candidates)
    builder, length_filter = None, None
    if config.tokenizer_id is not None:
        # Tokenization is optional; ordinary preparation does not import the
        # model stack. Reuse the actual serializer when enforcing token budgets.
        from transformers import AutoTokenizer
        from huggingface_hub import resolve_revision
        from ..backbone.inputs import InputBuilder, InputConfig

        hub = dict(
            revision=config.tokenizer_revision,
            cache_dir=str(config.artifacts / "models" / "hub"),
            local_files_only=config.local_files_only,
            trust_remote_code=False,
        )
        if not Path(config.tokenizer_id).is_dir():
            hub["revision"] = resolve_revision(
                config.tokenizer_id,
                revision=config.tokenizer_revision,
                cache_dir=hub["cache_dir"],
                local_files_only=config.local_files_only,
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
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=root) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        # Only the final outputs are staged for atomic publication. There is no
        # temporary dataset copy or disk-backed grouping store.
        with ExitStack() as stack:
            writers = {
                name: stack.enter_context(
                    DatasetWriter(
                        staging / f"{name}.parquet",
                        batch_size=config.context_batch_size,
                    )
                )
                for name in sizes
            }
            maximum_length = 0
            for name, example in iter_materialized(
                source, assignment, config.batch_size, config.context_batch_size
            ):
                if builder is not None:
                    # Check each complete context before writing, without a
                    # second in-memory collection of all selected examples.
                    maximum_length = max(
                        maximum_length, len(builder.encode(example).input_ids)
                    )
                writers[name].write(example)
        if length_filter is not None:
            length_filter["selected_max_packed_tokens"] = maximum_length
        split_manifest = {}
        for name, writer in writers.items():
            path = staging / f"{name}.parquet"
            stats = writer.statistics()
            if stats["questions"] != len(assignment[name]):
                raise RuntimeError(
                    f"Written question count differs from selection for {name}"
                )
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
            "source": {**source.describe(), "questions": source_count},
            "selection": {
                "category_balance": "equal, capped by source availability; shortages redistributed",
                "within_category": "proportional by task and target kind in the eligible source pool",
            },
            "grouping": "Source-provided context IDs; complete groups stay within one split",
            "preparation": {
                "source_batch_size": config.batch_size,
                "context_batch_size": config.context_batch_size,
                "storage": "streamed_parquet_v1",
            },
            "splits": split_manifest,
            "length_filter": length_filter,
            "limitations": [
                "Only selected questions are merged; unselected rows are not added to context groups.",
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
