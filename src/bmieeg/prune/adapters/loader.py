# ================================================================
# 0. Section: IMPORTS
# ================================================================
from tqdm import tqdm

from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

from .source import Source
from ..domain import DataActor
from .file_loaders import FileLoaderRegistry


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Loader:
    source: Source

    def load(self, source_names: list[str]) -> list[DataActor]:
        known_suffixes = FileLoaderRegistry.known_keys()

        actors: list[DataActor] = []
        for source_str in source_names:
            source_path = Path(source_str)
            full_path = self.source.raw_data_dir(source_path)

            files_present = [p for p in full_path.rglob("*") if p.is_file()]

            files_by_type: dict[str, list[Path]] = defaultdict(list)
            for p in files_present:
                file_suffix = p.suffix.lower()
                if file_suffix not in known_suffixes:
                    continue
                files_by_type[file_suffix].append(p)

            # files_by_type[".xdf"] = files_by_type[".xdf"][1:2]

            _print_readable_summary(source_str, files_present)

            for file_suffix, files in files_by_type.items():
                loader = FileLoaderRegistry.get(file_suffix)
                for file in tqdm(files, desc="Loading files"):
                    actors.extend(loader.load_actors(file))

        return actors


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _print_readable_summary(source_str: str, files_present: list[Path]) -> None:
    known = FileLoaderRegistry.known_keys()
    types = {p.suffix.lower() for p in files_present}
    n_readable = sum(p.suffix.lower() in known for p in files_present)
    pct = 100 * n_readable / len(files_present) if files_present else 0.0

    print(f"[{source_str}] readable: {n_readable}/{len(files_present)} ({pct:.1f}%)")
    print(f"[{source_str}] not read: {sorted(types - known)}")
