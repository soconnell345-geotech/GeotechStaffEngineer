"""
MSE wall reinforcement properties.

Defines reinforcement types and built-in common products.

All units are SI: kN/m, m, kPa.

References:
    FHWA GEC-11, Chapter 3
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Reinforcement:
    """MSE wall reinforcement properties.

    Parameters
    ----------
    name : str
        Product/designation name.
    type : str
        "metallic_strip", "metallic_grid", or "geosynthetic".
    Tallowable : float
        Long-term allowable tensile strength per unit width (kN/m).
    width : float, optional
        Strip width (m). For strips only. Default 0.05 (50mm).
    Fy : float, optional
        Yield strength of steel (kPa). Default 0 (not metallic).
    thickness : float, optional
        Strip thickness (m) for strips; transverse-bar diameter (m) for steel
        bar-mat / welded-grid reinforcement (the ``t`` in the F* = 20·t/St
        bearing-resistance ratio). Default 0.
    coverage_ratio : float, optional
        Coverage ratio Rc = b/Sh (strip width / horizontal spacing).
        For metallic strips, typically 0.10-0.15.
        For continuous geogrids, Rc = 1.0 (default).
    transverse_spacing : float, optional
        Transverse-bar (grid) spacing St (m) for steel bar-mat / welded-grid
        reinforcement (the ``St`` in F* = 20·t/St). Only used for
        ``metallic_grid``. Default 0 (then the per-call ``t_over_St`` /
        F* default applies).
    """
    name: str
    type: str
    Tallowable: float
    width: float = 0.05
    Fy: float = 0.0
    thickness: float = 0.0
    coverage_ratio: float = 1.0
    transverse_spacing: float = 0.0

    def __post_init__(self):
        valid_types = ("metallic_strip", "metallic_grid", "geosynthetic")
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got '{self.type}'")
        if self.Tallowable <= 0:
            raise ValueError(f"Tallowable must be positive, got {self.Tallowable}")

    @property
    def is_metallic(self) -> bool:
        return self.type.startswith("metallic")

    @property
    def is_grid(self) -> bool:
        """True for steel bar-mat / welded-grid reinforcement (uses the
        bar-mat Kr/Ka and F* = 20·t/St curves, not the ribbed-strip curves)."""
        return self.type == "metallic_grid"

    @property
    def t_over_St(self):
        """Transverse-bar diameter / spacing (t/St), unit-free, for the
        steel-grid F* curve. Returns None if the grid geometry is not set."""
        if self.is_grid and self.thickness > 0 and self.transverse_spacing > 0:
            return self.thickness / self.transverse_spacing
        return None


# Built-in reinforcement types
# NOTE: For metallic strips, coverage_ratio Rc = b/Sh must be set by
# the user based on horizontal spacing. Typical values: 0.10-0.15.
# Default Rc=1.0 is conservative (overestimates pullout resistance).
RIBBED_STEEL_STRIP_75x4 = Reinforcement(
    name="Ribbed Steel Strip 75x4mm",
    type="metallic_strip",
    Tallowable=43.1,  # kN/m (Grade 65 steel, 75mm wide, 4mm thick, after corrosion)
    width=0.075,
    Fy=450000.0,
    thickness=0.004,
    coverage_ratio=1.0,  # User should set Rc = b/Sh for actual design
)

WELDED_WIRE_GRID_W11 = Reinforcement(
    name="Welded Wire Grid W11xW11",
    type="metallic_grid",
    Tallowable=32.0,  # kN/m
    Fy=450000.0,
    thickness=0.0095,           # W11 transverse-bar diameter t = 0.374 in
    transverse_spacing=0.1524,  # St = 6 in (densest level) -> t/St = 0.0623
)

GEOGRID_UX1600 = Reinforcement(
    name="Geogrid UX1600 (HDPE)",
    type="geosynthetic",
    Tallowable=21.9,  # kN/m (LTDS after creep, damage, durability)
)

GEOGRID_UX1700 = Reinforcement(
    name="Geogrid UX1700 (HDPE)",
    type="geosynthetic",
    Tallowable=29.2,  # kN/m
)
