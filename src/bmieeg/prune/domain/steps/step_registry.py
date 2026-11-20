# ================================================================
# 0. Section: IMPORTS
# ================================================================
from bmieeg.utils import Registry

from .prune_step import PruneStep


# ================================================================
# 1. Section: Class definition
# ================================================================
class StepRegistry(Registry[PruneStep]): ...
