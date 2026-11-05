"""Explicit upload/export commands; library modules have no standalone CLI."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Union
import tyro

from .export import ExportConfig, export
from .tracking import sync


@dataclass
class UploadConfig:
    """Send an existing local run to remote MLflow, including missing history."""
    run: Path
    remote_uri: str


if __name__ == "__main__":
    config = tyro.cli(Union[
        Annotated[UploadConfig, tyro.conf.subcommand(name="upload")],
        Annotated[ExportConfig, tyro.conf.subcommand(name="export")],
    ])
    if isinstance(config, UploadConfig):
        sync(config.run, config.remote_uri)
    else:
        result = export(config)
        print(result)
        if result.get("failed"):
            raise SystemExit(1)
