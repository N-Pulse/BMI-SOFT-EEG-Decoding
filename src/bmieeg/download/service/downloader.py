# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass, field

from ..domain import Registry, DownloadSpec



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Downloader:
    _registry: Registry = field(default_factory=Registry)

    def run(self, spec: DownloadSpec) -> None:
        # 1. Select strategy based on source
        source = spec.source
        strategy = self._registry.get(source)

        # 2. Fetch data using selected strategy
        strategy.fetch(spec)
