"""
Hammer cushion and pile cushion properties.

Cushions are modeled as linear springs with optional coefficient
of restitution (COR) to account for energy losses on unloading.

All units are SI: kN, m, seconds.

The COR: 1.0 for a perfectly elastic collision, 0.0 for a perfectly
plastic one (GEC-12 Vol 2 Sec. 12.3, p. 156). The unloading path that
gives an energy return of COR squared is not printed in a held reference.

References:
    FHWA GEC-12 Vol 2, Sec. 12.3 (p. 156) and 12.6.6 (pp. 209-210)
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
        GEC-12 Vol 2: wood 0.50 by default, a badly worn softwood
        cushion 0.25 (p. 156); the hammer cushions of its examples 0.90
        to 0.92 (pp. 174, 183). The default 0.80 is not printed in a
        held reference.
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

    Stiffness = E * A / t (GEC-12 Vol 2 p. 155). Two materials in series:
    1/k = 1/k1 + 1/k2 (p. 209). Plywood pile cushions: E = 30 ksi
    (207 MPa) new and 75 ksi (517 MPa) used; oak boards 60 and 90 ksi;
    always with the nominal (uncompressed) thickness (p. 209).

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
