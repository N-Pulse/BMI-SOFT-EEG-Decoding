# ================================================================
# 0. Section: IMPORTS
# ================================================================
from bmieeg.utils import Registry

from .file_loader import FileLoader


# ================================================================
# 1. Section: Class definition
# ================================================================
class FileLoaderRegistry(Registry[FileLoader]): ...
