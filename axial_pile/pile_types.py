"""
Pile type definitions and section property database.

Defines driven pile types (H-pile, pipe, concrete, timber) with
cross-section properties for axial capacity calculations.

All units are SI: meters, kN, kPa.

References:
    AISC Steel Construction Manual — HP shapes
    FHWA GEC-12, Chapter 3 (Pile Types)
"""

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class PileSection:
    """Cross-section properties for a driven pile.

    Parameters
    ----------
    name : str
        Section designation (e.g., "HP14x117", "24in Pipe").
    pile_type : str
        Pile type: "h_pile", "pipe_open", "pipe_closed", "concrete", "timber".
    area : float
        Cross-sectional area (m²) for structural capacity.
    perimeter : float
        Perimeter for skin friction computation (m).
    tip_area : float
        Tip area for end bearing (m²). For H-piles, this is the
        box area (flange_width × depth), not the steel area.
    width : float
        Width or diameter of pile (m). Used for Nordlund V_displaced.
    depth : float, optional
        Depth of section (m). For H-piles, this is the nominal depth.
        For round piles, equals width.
    E : float, optional
        Young's modulus (kPa). Default: 200e6 for steel, 25e6 for concrete.
    """
    name: str
    pile_type: str
    area: float
    perimeter: float
    tip_area: float
    width: float
    depth: Optional[float] = None
    E: float = 200e6  # kPa, default steel

    def __post_init__(self):
        if self.depth is None:
            self.depth = self.width
        valid = ("h_pile", "pipe_open", "pipe_closed", "concrete", "timber")
        if self.pile_type not in valid:
            raise ValueError(f"pile_type must be one of {valid}, got '{self.pile_type}'")


def make_pipe_pile(diameter: float, thickness: float,
                   closed_end: bool = True,
                   E: float = 200e6) -> PileSection:
    """Create a pipe pile section from diameter and wall thickness.

    Parameters
    ----------
    diameter : float
        Outside diameter (m).
    thickness : float
        Wall thickness (m).
    closed_end : bool, optional
        If True, pile tip is closed (full tip area). Default True.
    E : float, optional
        Young's modulus (kPa). Default 200e6 (steel).

    Returns
    -------
    PileSection
    """
    r_outer = diameter / 2
    r_inner = r_outer - thickness
    steel_area = math.pi * (r_outer**2 - r_inner**2)
    perimeter = math.pi * diameter
    if closed_end:
        tip_area = math.pi * r_outer**2
        ptype = "pipe_closed"
    else:
        tip_area = steel_area  # annular ring for open-ended
        ptype = "pipe_open"

    d_in = diameter / 0.0254  # convert to inches for name
    t_in = thickness / 0.0254
    name = f"{d_in:.0f}in Pipe ({t_in:.2f}in wall)"

    return PileSection(
        name=name, pile_type=ptype,
        area=steel_area, perimeter=perimeter,
        tip_area=tip_area, width=diameter, E=E,
    )


def make_concrete_pile(width: float, shape: str = "square",
                       E: float = 25e6) -> PileSection:
    """Create a precast concrete pile section.

    Parameters
    ----------
    width : float
        Width (square) or diameter (circular) in meters.
    shape : str, optional
        "square" or "circular". Default "square".
    E : float, optional
        Young's modulus (kPa). Default 25e6 (concrete).

    Returns
    -------
    PileSection
    """
    if shape == "square":
        area = width**2
        perimeter = 4 * width
        tip_area = width**2
    elif shape == "circular":
        area = math.pi / 4 * width**2
        perimeter = math.pi * width
        tip_area = area
    else:
        raise ValueError(f"Shape must be 'square' or 'circular', got '{shape}'")

    w_in = width / 0.0254
    name = f"{w_in:.0f}in {shape.capitalize()} Concrete"

    return PileSection(
        name=name, pile_type="concrete",
        area=area, perimeter=perimeter,
        tip_area=tip_area, width=width, E=E,
    )


# ── Built-in HP Shape Database ─────────────────────────────────────────
# Common AISC HP shapes used in driven pile applications.
# Values: (area_m2, depth_m, flange_width_m, perimeter_m)
# Perimeter for H-piles = 2*(depth + flange_width) approximately (box perimeter)
# Tip area for H-piles = depth * flange_width (box area)

_HP_SHAPES: Dict[str, Dict] = {
    "HP10x42": {"area": 0.00797, "depth": 0.2464, "bf": 0.2565, "tw": 0.00886},
    "HP10x57": {"area": 0.01081, "depth": 0.2540, "bf": 0.2606, "tw": 0.01194},
    "HP12x53": {"area": 0.01006, "depth": 0.2997, "bf": 0.3048, "tw": 0.00876},
    "HP12x63": {"area": 0.01194, "depth": 0.3048, "bf": 0.3061, "tw": 0.01067},
    "HP12x74": {"area": 0.01406, "depth": 0.3099, "bf": 0.3086, "tw": 0.01245},
    "HP12x84": {"area": 0.01594, "depth": 0.3150, "bf": 0.3112, "tw": 0.01435},
    "HP14x73": {"area": 0.01387, "depth": 0.3505, "bf": 0.3658, "tw": 0.01029},
    "HP14x89": {"area": 0.01690, "depth": 0.3581, "bf": 0.3683, "tw": 0.01270},
    "HP14x102": {"area": 0.01935, "depth": 0.3632, "bf": 0.3721, "tw": 0.01448},
    "HP14x117": {"area": 0.02219, "depth": 0.3721, "bf": 0.3759, "tw": 0.01689},
}


def make_h_pile(designation: str, E: float = 200e6) -> PileSection:
    """Create an HP-shape pile from the built-in database.

    Parameters
    ----------
    designation : str
        AISC HP shape designation (e.g., "HP14x117").
    E : float, optional
        Young's modulus (kPa). Default 200e6 (steel).

    Returns
    -------
    PileSection

    Raises
    ------
    ValueError
        If designation not found in database.
    """
    key = designation.replace(" ", "")
    if key not in _HP_SHAPES:
        available = ", ".join(sorted(_HP_SHAPES.keys()))
        raise ValueError(f"Unknown HP shape '{designation}'. Available: {available}")

    props = _HP_SHAPES[key]
    d = props["depth"]
    bf = props["bf"]
    perimeter = 2 * (d + bf)  # box perimeter
    tip_area = d * bf  # box area

    return PileSection(
        name=designation,
        pile_type="h_pile",
        area=props["area"],
        perimeter=perimeter,
        tip_area=tip_area,
        width=bf,
        depth=d,
        E=E,
    )
