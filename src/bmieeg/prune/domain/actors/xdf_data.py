# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass

from .metadata_actor import MetadataActor


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass(kw_only=True)
class XdfData(MetadataActor):
    name: ClassVar[str] = "xdf_data"
    streams: list[dict]
