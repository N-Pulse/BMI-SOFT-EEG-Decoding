# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Any
from pathlib import Path
from dataclasses import dataclass

from .exporters import Writer
from ..domain import DataActor


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Exporter:
    def export(
        self,
        actors: list[DataActor],
        writers: list[Writer[Any]],
        dest_dir: Path,
    ) -> None:
        for writer in writers:
            matching = [actor for actor in actors if actor.name == writer.actor]
            if not matching:
                raise ValueError(
                    f"Writer {writer.name!r} found no {writer.actor!r} actors"
                )
            for actor in matching:
                writer.write(actor, dest_dir)
