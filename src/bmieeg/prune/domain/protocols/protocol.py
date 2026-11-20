# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Any
from dataclasses import dataclass
from collections.abc import Sequence

from ..steps import PruneStep


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Protocol:
    name: str
    steps: Sequence[PruneStep[Any, Any]]
