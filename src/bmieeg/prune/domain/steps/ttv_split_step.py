# ================================================================
# 0. Section: IMPORTS
# ================================================================
import math

import numpy as np

from numpy.typing import NDArray
from dataclasses import dataclass
from typing import ClassVar, Literal

from .prune_step import PruneStep
from .step_registry import StepRegistry
from ..actors import Dataset, DatasetPartition, TrialData


# ================================================================
# 1. Section: Functions
# ================================================================
@StepRegistry.register("ttv_split")
@dataclass
class TTVSplitStep(PruneStep[TrialData, Dataset]):
    name: ClassVar[str] = "ttv_split"
    train_ratio: float
    val_ratio: float
    test_ratio: float
    user_balance: bool = False
    movement_balance: bool = False
    seed: int = 0
    group_by: Literal["subject"] | None = None

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(total, 1.0):
            raise ValueError(f"train/val/test ratios must sum to 1.0, got {total}")
        if self.group_by not in (None, "subject"):
            raise ValueError(
                f"group_by must be None or 'subject', got {self.group_by!r}"
            )
        if self.group_by is not None and (self.user_balance or self.movement_balance):
            raise ValueError("group_by cannot be combined with user/movement balance")

    def apply(self, actors: list[TrialData]) -> list[Dataset]:
        rng = np.random.default_rng(self.seed)

        if self.group_by is not None:
            train, val, test = self._split_groups(actors, rng)
        else:
            train, val, test = self._split_strata(actors, rng)

        test_unlabelled = [_hide_markers(trial) for trial in test]
        client = [
            DatasetPartition(partition_name="train", trials=train),
            DatasetPartition(partition_name="val", trials=val),
            DatasetPartition(partition_name="test", trials=test_unlabelled),
        ]
        server = [DatasetPartition(partition_name="test", trials=test)]

        return [Dataset(client=client, server=server)]

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _split_strata(
        self, actors: list[TrialData], rng: np.random.Generator
    ) -> tuple[list[TrialData], list[TrialData], list[TrialData]]:
        train: list[TrialData] = []
        val: list[TrialData] = []
        test: list[TrialData] = []

        # Split each stratum on its own so the requested balances hold.
        for stratum in self._strata(actors).values():
            t, v, s = self._split_stratum(stratum, rng)
            train += t
            val += v
            test += s
        return train, val, test

    def _strata(self, actors: list[TrialData]) -> dict[tuple, list[TrialData]]:
        strata: dict[tuple, list[TrialData]] = {}
        for actor in actors:
            strata.setdefault(self._stratum_key(actor), []).append(actor)
        return strata

    def _stratum_key(self, actor: TrialData) -> tuple:
        key: list[object] = []
        if self.user_balance:
            key.append(actor.subject_number)
        if self.movement_balance:
            key.append(_dominant_movement(actor.markers))
        return tuple(key)

    def _split_groups(
        self, actors: list[TrialData], rng: np.random.Generator
    ) -> tuple[list[TrialData], list[TrialData], list[TrialData]]:
        groups: dict[int, list[TrialData]] = {}
        for actor in actors:
            subject = actor.subject_number
            if subject is None:
                raise ValueError(
                    f"group_by=subject needs a subject for {actor.file_name}"
                )
            groups.setdefault(subject, []).append(actor)

        keys = sorted(groups)
        shuffled = [keys[i] for i in rng.permutation(len(keys))]
        n_train, n_val, _ = _largest_remainder(
            len(shuffled), (self.train_ratio, self.val_ratio, self.test_ratio)
        )

        def collect(selected: list[int]) -> list[TrialData]:
            return [trial for key in selected for trial in groups[key]]

        return (
            collect(shuffled[:n_train]),
            collect(shuffled[n_train : n_train + n_val]),
            collect(shuffled[n_train + n_val :]),
        )

    def _split_stratum(
        self, actors: list[TrialData], rng: np.random.Generator
    ) -> tuple[list[TrialData], list[TrialData], list[TrialData]]:
        shuffled = [actors[i] for i in rng.permutation(len(actors))]

        n = len(shuffled)
        n_train = min(round(n * self.train_ratio), n)
        n_val = min(round(n * self.val_ratio), n - n_train)

        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]
        return train, val, test


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Module Helpers
# ──────────────────────────────────────────────────────
def _hide_markers(trial: TrialData) -> TrialData:
    return trial.copy_with(markers=np.zeros_like(trial.markers))


def _largest_remainder(total: int, ratios: tuple[float, ...]) -> list[int]:
    # Whole groups can't be rounded per partition without losing or adding one.
    exact = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in exact]
    order = sorted(range(len(ratios)), key=lambda i: (counts[i] - exact[i], i))
    for i in order[: total - sum(counts)]:
        counts[i] += 1
    return counts


def _dominant_movement(markers: NDArray) -> int:
    values = markers[markers != 0]
    movements = values % 100
    movements = movements[movements != 0]
    if movements.size == 0:
        return 0
    codes, counts = np.unique(movements, return_counts=True)
    return int(codes[np.argmax(counts)])
