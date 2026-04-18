"""Data preprocessing pipeline: EEG processing and label simplification."""

from src.data.simplify_labels import MAPS, simplify_npz

__all__ = ["MAPS", "simplify_npz"]
