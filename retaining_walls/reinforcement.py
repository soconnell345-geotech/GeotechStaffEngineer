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
        Strip/grid thickness (m). Default 0.
    """
    name: str
    type: str
    Tallowable: float
    width: float = 0.05
    Fy: float = 0.0
    thickness: float = 0.0

    def __post_init__(self):
        valid_types = ("metallic_strip", "metallic_grid", "geosynthetic")
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got '{self.type}'")
        if self.Tallowable <= 0:
            raise ValueError(f"Tallowable must be positive, got {self.Tallowable}")

    @property
    def is_metallic(self) -> bool:
        return self.type.startswith("metallic")


# Built-in reinforcement types
RIBBED_STEEL_STRIP_75x4 = Reinforcement(
    name="Ribbed Steel Strip 75x4mm",
    type="metallic_strip",
    Tallowable=43.1,  # kN/m (Grade 65 steel, 75mm wide, 4mm thick, after corrosion)
    width=0.075,
    Fy=450000.0,
    thickness=0.004,
)

WELDED_WIRE_GRID_W11 = Reinforcement(
    name="Welded Wire Grid W11xW11",
    type="metallic_grid",
    Tallowable=32.0,  # kN/m
    Fy=450000.0,
    thickness=0.009,
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
