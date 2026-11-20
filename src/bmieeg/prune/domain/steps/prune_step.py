# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Any, ClassVar, Self
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..actors import DataActor


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class PruneStep[InT: DataActor, OutT: DataActor](ABC):
    name: ClassVar[str]

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> Self:
        """Build the step from its parsed config section.

        Override when a parameter needs converting from plain YAML types.
        """
        return cls(**data)

    @abstractmethod
    def apply(self, actors: list[InT]) -> list[OutT]:
        pass
