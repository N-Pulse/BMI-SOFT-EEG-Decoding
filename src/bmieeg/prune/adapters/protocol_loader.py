# ================================================================
# 0. Section: IMPORTS
# ================================================================
import yaml

from pathlib import Path
from typing import Any
from dataclasses import dataclass

from ..domain import Protocol, StepRegistry
from .exporters import Writer, WriterRegistry


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ProtocolLoader:
    config_dir: Path

    def load(self, protocol_name: str) -> Protocol:
        config = self._read(protocol_name)
        steps = [
            StepRegistry.get_type(key).from_config(params or {})
            for key, params in _entries(config["steps"])
        ]
        return Protocol(name=config["protocol_name"], steps=steps)

    def load_writers(self, protocol_name: str, timestamp: str) -> list[Writer[Any]]:
        config = self._read(protocol_name)
        return [
            WriterRegistry.get_type(key).from_config(params or {}, timestamp)
            for key, params in _entries(config.get("export", []))
        ]

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _read(self, protocol_name: str) -> dict[str, Any]:
        path = self.config_dir / f"{protocol_name}.yaml"
        with path.open() as file:
            return yaml.safe_load(file)


def _entries(section: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    # Each list item is a one-key mapping {name: params}; order is run order.
    return [(key, params) for entry in section for key, params in entry.items()]
