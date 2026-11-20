# ================================================================
# 0. Section: IMPORTS
# ================================================================
import statistics
import numpy as np

from collections import Counter
from dataclasses import dataclass, field
from typing import ClassVar

from ..actors import TrialData, TrialWindow
from .prune_step import PruneStep
from .step_registry import StepRegistry


# ================================================================
# 1. Section: Functions
# ================================================================
@StepRegistry.register("find_trials")
@dataclass
class FindTrialsStep(PruneStep[TrialData, TrialData]):
    """Lists the trials of a whole recording from its markers (after `marker_unification`).

    A trial starts with the prep signal and ends when a normal iti is over. A trial that was cut
    short (the recording stopped) is left out. A recording with fewer than `min_trials` trials is a
    test recording and is dropped. Recordings that already have their trials (MOVING) pass through.
    """

    name: ClassVar[str] = "find_trials"
    movements: dict[str, int]  # name of the movement -> code DD of the markers
    phases: dict[str, int] = field(default_factory=lambda: {"prep": 2, "move": 3, "return": 4, "iti": 5})
    min_trials: int = 12

    def apply(self, actors: list[TrialData]) -> list[TrialData]:
        found = []
        for actor in actors:
            if actor.windows:
                found.append(actor)
                continue
            windows, unknown = self._find_windows(actor)
            if len(windows) < self.min_trials:
                print(f"skipped {actor.file_name}: {len(windows)} trials, a test recording")
                continue
            if unknown:
                raise ValueError(f"{actor.file_name}: no name for the movement codes {sorted(unknown)}")
            found.append(actor.copy_with(windows=windows))
        return found

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _find_windows(self, actor: TrialData) -> tuple[list[TrialWindow], set[int]]:
        codes = actor.markers[0]
        names = {code: name for name, code in self.movements.items()}

        # The markers are held until the next one, so every run of one code is one phase
        change = np.flatnonzero(np.diff(codes)) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [codes.size]])
        values = codes[starts]
        phase, movement = values // 100, values % 100
        phase_name = {code: name for name, code in self.phases.items()}

        # How long the iti normally lasts: the middle value, the last run has no marker after it
        iti_runs = [i for i in range(len(values)) if phase_name.get(int(phase[i])) == "iti"]
        lengths = [(ends[i] - starts[i]) / actor.sfreq for i in iti_runs[:-1]]
        iti_seconds = statistics.median(lengths) if lengths else 0.0

        # Every prep starts a trial, the phases up to the next prep belong to it
        preps = [i for i in range(len(values)) if phase_name.get(int(phase[i])) == "prep"]
        windows, repetitions, unknown = [], Counter(), set()
        for n, first in enumerate(preps):
            last = preps[n + 1] if n + 1 < len(preps) else len(values)
            found: dict[str, int] = {}
            for i in range(first, last):
                found.setdefault(phase_name.get(int(phase[i]), "other"), i)
            if any(name not in found for name in self.phases):
                continue

            movement_name = names.get(int(movement[first]))
            if movement_name in (None, "undefined"):
                unknown.add(int(movement[first]))
                continue
            windows.append(TrialWindow(
                start=float(starts[first] / actor.sfreq),
                end=float(starts[found["iti"]] / actor.sfreq + iti_seconds),
                movement=movement_name,
                repetition=repetitions[movement_name],
                orientation="n/a",
            ))
            repetitions[movement_name] += 1
        return windows, unknown
