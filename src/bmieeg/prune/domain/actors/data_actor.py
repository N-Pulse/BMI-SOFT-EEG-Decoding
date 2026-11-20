# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC
from typing import Any, ClassVar, Self
from dataclasses import dataclass, replace


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass
class DataActor(ABC):
    name: ClassVar[str]

    def copy_with(self, **changes: Any) -> Self:
        return replace(self, **changes)
