"""
Driving systems (hammer cushion and helmet) as a held reference prints them.

GEC-12 Vol 2 (FHWA-NHI-16-009) has no table of cushion and helmet data by
hammer: Appendix C (pp. 519-541) lists each hammer's ram weight, stroke
and rated energy only, and p. 225 sends the analyst to GRLWEAP's own data
file ("Manufacturer's Recommended Driving System", p. 226), which is not
held. "Ideally, the contractor would provide the above drive system data
for his actual hammer system" (p. 226). The only complete driving systems
printed for a named hammer on a steel pile, with nothing but the hammer
cushion between the hammer and the helmet (the module's one cushion
spring), are Example 6's two, Table 12-2 (p. 183):

============================  ===========  ===========
                              Vulcan 014   ICE 42S
============================  ===========  ===========
Helmet weight, kips           1.67         2.05
Hammer cushion material       Nycast       Blue Nylon
Hammer cushion E, ksi         208          175
Hammer cushion area, in2      234          398
Hammer cushion thickness, in  6.0          2.0
Hammer cushion COR            0.91         0.92
============================  ===========  ===========

The stiffness is E A / t (p. 155): 8,112 kips/in (1.42 million kN/m) and
34,825 kips/in (6.10 million kN/m).

The other examples print part of a system or a concrete pile's (which
also needs a pile cushion): Example 1, Delmag D12-42, 1 in aluminum and
1 in Conbest, 227 in2, 530 ksi averaged, helmet 1.7 kips, no COR (p. 163);
Example 3, Conmaco C 160, Nylon, helmet 4.07 kips, 10 in plywood pile
cushion (p. 167); Example 4, Junttan HHK 3A on a concrete pile, 8 in,
360 ksi, COR 0.90, 250 in2, helmet 1.0 kip (p. 174); Example 5, Vulcan 08
on a concrete pile, 8.5 in Hammortex, 148 in2, helmet 2.6 kips (p. 179).
They are listed in the error a run without a cushion gets, not offered by
name.

All units SI: kN, m, kPa.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from wave_equation.cushion import Cushion

_KIP = 4.4482216152605      # kN
_IN = 0.0254                # m
_KSI = 6894.757293168       # kPa


@dataclass(frozen=True)
class DrivingSystem:
    """A driving system as printed: the hammer cushion and the helmet."""
    hammer: str
    helmet_kips: float
    material: str
    E_ksi: float
    area_in2: float
    thickness_in: float
    cor: float
    source: str

    @property
    def helmet_weight(self) -> float:
        """Helmet weight (kN)."""
        return self.helmet_kips * _KIP

    @property
    def stiffness(self) -> float:
        """Hammer cushion stiffness E A / t (kN/m), GEC-12 Vol 2 p. 155."""
        return (self.E_ksi * _KSI) * (self.area_in2 * _IN ** 2) \
            / (self.thickness_in * _IN)

    def cushion(self) -> Cushion:
        """The hammer cushion as the module's one cushion spring."""
        return Cushion(stiffness=self.stiffness, cor=self.cor,
                       thickness=self.thickness_in * _IN,
                       material=self.material)

    def describe(self) -> str:
        """The system in the print's terms, with its source."""
        return (f"{self.hammer}: {self.material} hammer cushion, "
                f"{self.E_ksi:g} ksi, {self.area_in2:g} in2, "
                f"{self.thickness_in:g} in (k = {self.stiffness:,.0f} kN/m), "
                f"COR {self.cor:g}; helmet {self.helmet_kips:g} kips "
                f"({self.helmet_weight:.2f} kN); {self.source}")


_TABLE_12_2 = "GEC-12 Vol 2 Example 6, Table 12-2 (p. 183)"

#: The driving systems printed complete for a named hammer, by hammer.
PRINTED_DRIVING_SYSTEMS: Dict[str, DrivingSystem] = {
    "Vulcan 014": DrivingSystem("Vulcan 014", 1.67, "Nycast", 208.0, 234.0,
                                6.0, 0.91, _TABLE_12_2),
    "ICE 42S": DrivingSystem("ICE 42S", 2.05, "Blue Nylon", 175.0, 398.0,
                             2.0, 0.92, _TABLE_12_2),
}

#: What a held reference prints for a cushion or a helmet, for an error
#: that asks for one (GM-299).
PRINTED_OPTIONS = (
    "Printed options (GEC-12 Vol 2, FHWA-NHI-16-009): the contractor's "
    "submittal for the actual hammer (p. 226); driving_system 'Vulcan 014' "
    "or 'ICE 42S', Example 6's systems as Table 12-2 (p. 183) prints them "
    "(Vulcan 014: Nycast, 208 ksi, 234 in2, 6.0 in, COR 0.91, helmet 1.67 "
    "kips; ICE 42S: Blue Nylon, 175 ksi, 398 in2, 2.0 in, COR 0.92, helmet "
    "2.05 kips); stiffness = E A / t (p. 155), two materials in series "
    "1/k = 1/k1 + 1/k2 (p. 209). Hammer cushion COR in the examples 0.90 "
    "to 0.92 (pp. 174, 183); wood 0.50, a badly worn softwood cushion 0.25 "
    "(p. 156). Helmets in the examples 1.0 to 4.07 kips (4.4 to 18.1 kN): "
    "Delmag D12-42 1.7 (p. 163), Conmaco C 160 4.07 (p. 167), Junttan "
    "HHK 3A 1.0 (p. 174), Vulcan 08 2.6 (p. 179), Vulcan 014 1.67 and "
    "ICE 42S 2.05 (p. 183). GEC-12 prints no cushion or helmet by hammer "
    "for other hammers (Appendix C lists ram weight, stroke and energy "
    "only; GRLWEAP's data file, p. 226, is not held).")


def get_driving_system(name: str) -> DrivingSystem:
    """The printed driving system of ``name`` (a key of
    :data:`PRINTED_DRIVING_SYSTEMS`, matched without case or spaces)."""
    key = _key(name)
    for hammer, system in PRINTED_DRIVING_SYSTEMS.items():
        if _key(hammer) == key:
            return system
    raise ValueError(
        f"No printed driving system {name!r}; the printed ones are "
        f"{sorted(PRINTED_DRIVING_SYSTEMS)} (GEC-12 Vol 2 Table 12-2, "
        "p. 183).")


def driving_system_for_hammer(hammer_name: str) -> Optional[DrivingSystem]:
    """The printed driving system of the hammer named ``hammer_name``,
    None when no held reference prints one."""
    key = _key(hammer_name)
    for hammer, system in PRINTED_DRIVING_SYSTEMS.items():
        if _key(hammer) == key:
            return system
    return None


def _key(name) -> str:
    return "".join(str(name).lower().split()).replace("-", "")


def require_helmet_weight(helmet_weight) -> float:
    """``helmet_weight`` (kN), refused when left out: no held reference
    prints a helmet weight for every hammer (GM-299)."""
    if helmet_weight is None:
        raise ValueError(
            "helmet_weight is required (kN): the combined weight of the "
            "helmet, hammer cushion, striker plate and inserts (GEC-12 "
            "Vol 2 p. 226), or 0 with no helmet mass. " + PRINTED_OPTIONS)
    if helmet_weight < 0:
        raise ValueError(f"helmet_weight must be >= 0, got {helmet_weight}")
    return float(helmet_weight)
