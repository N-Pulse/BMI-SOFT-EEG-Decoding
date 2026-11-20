# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar

from ..actors import XdfData
from .prune_step import PruneStep
from .step_registry import StepRegistry


# ================================================================
# 1. Section: Functions
# ================================================================
@StepRegistry.register("remove_session")
@dataclass
class RemoveSessionStep(PruneStep[XdfData, XdfData]):
    name: ClassVar[str] = "remove_session"
    files_to_remove: list[str]

    def apply(self, actors: list[XdfData]) -> list[XdfData]:
        filtered_actors = [p for p in actors if p.file_name not in self.files_to_remove]
        return filtered_actors
