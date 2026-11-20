# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pyxdf

from pathlib import Path

from .file_loader import FileLoader
from ...domain import DataActor, XdfData
from .file_loader_registry import FileLoaderRegistry


# ================================================================
# 1. Section: Class definition
# ================================================================
@FileLoaderRegistry.register(".xdf")
class XdfLoader(FileLoader):
    def load_actors(self, path: Path) -> list[DataActor]:
        streams, file_header = pyxdf.load_xdf(path)

        return [
            XdfData(
                streams=streams,
                file_header=file_header,
                file_name=path.stem,
                file_path=path,
            )
        ]
