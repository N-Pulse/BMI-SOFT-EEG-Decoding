# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass


# ================================================================
# 1. Section: Target structure (what markers become)
# ================================================================
@dataclass
class CanonicalMarkerMap:
    phases: dict[str, int]  # phase name -> XX
    movements: dict[str, int]  # movement name -> DD

    def encode(self, phase: str, movement: str) -> int:
        return self.phases[phase] * 100 + self.movements[movement]


# ================================================================
# 2. Section: Source structure (what markers currently are)
# ================================================================
@dataclass
class SourceMarkerSchema:
    phase_pos: int
    movement_slice: tuple[int, int]
    phase_map: dict[int, str]  # raw phase digit -> canonical phase name
    movement_groups: dict[str, list[int]]  # canonical movement name -> raw codes
    special_triggers: dict[int, str]  # whole raw code -> canonical phase name
    code_length: int = 5

    def decode(self, raw: int) -> tuple[str, str]:
        if raw in self.special_triggers:
            return self.special_triggers[raw], "undefined"

        digits = str(raw).zfill(self.code_length)
        phase_digit = int(digits[self.phase_pos])
        movement_code = int(digits[slice(*self.movement_slice)])

        phase = self.phase_map.get(phase_digit, "undefined")
        return phase, self._movement_name(movement_code)

    def _movement_name(self, movement_code: int) -> str:
        for name, codes in self.movement_groups.items():
            if movement_code in codes:
                return name
        return "undefined"
