"""
Hammer cushion and pile cushion properties.

Cushions are modeled as linear springs with optional coefficient
of restitution (COR) to account for energy losses on unloading.

All units are SI: kN, m, seconds.

References:
    WEAP87 Manual (FHWA, Goble & Rausche)
    FHWA GEC-12, Chapter 12
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Cushion:
    """Cushion (spring) between hammer and pile.

    Parameters
    ----------
    stiffness : float
        Cushion stiffness (kN/m).
    cor : float
        Coefficient of restitution (0 to 1). 1.0 = perfectly elastic.
        Typical: 0.80 for hammer cushion, 0.50 for pile cushion.
    thickness : float, optional
        Cushion thickness (m). Informational only.
    material : str, optional
        Material description (e.g. 'plywood', 'micarta', 'aluminum').
    """
    stiffness: float
    cor: float = 0.80
    thickness: Optional[float] = None
    material: str = ""

    def __post_init__(self):
        if self.stiffness <= 0:
            raise ValueError(f"Stiffness must be positive, got {self.stiffness}")
        if not 0 <= self.cor <= 1:
            raise ValueError(f"COR must be 0-1, got {self.cor}")


def make_cushion_from_properties(
    area: float,
    thickness: float,
    elastic_modulus: float,
    cor: float = 0.80,
    material: str = "",
) -> Cushion:
    """Create a cushion from material properties.

    Stiffness = E * A / t

    Parameters
    ----------
    area : float
        Cross-sectional area (m^2).
    thickness : float
        Thickness (m).
    elastic_modulus : float
        Elastic modulus of cushion material (kPa).
    cor : float
        Coefficient of restitution.
    material : str
        Material description.

    Returns
    -------
    Cushion
    """
    if thickness <= 0:
        raise ValueError(f"Thickness must be positive, got {thickness}")
    k = elastic_modulus * area / thickness
    return Cushion(stiffness=k, cor=cor, thickness=thickness, material=material)
