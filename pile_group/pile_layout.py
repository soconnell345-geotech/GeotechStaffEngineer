"""
Pile layout definitions for pile group analysis.

Defines individual pile positions, batter angles, and stiffnesses
within a pile group.

All units are SI: meters, kN, radians.

References:
    USACE EM 1110-2-2906, Chapter 4
    FHWA GEC-12, Chapter 9
    CPGA User's Guide (ITL-89-4)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class GroupPile:
    """A single pile within a pile group.

    Parameters
    ----------
    x : float
        X-coordinate of pile head relative to cap centroid (m).
    y : float
        Y-coordinate of pile head relative to cap centroid (m).
    batter_x : float, optional
        Batter angle in X-direction (degrees from vertical).
        Positive = pile leans in +X direction. Default 0 (vertical).
    batter_y : float, optional
        Batter angle in Y-direction (degrees from vertical).
        Positive = pile leans in +Y direction. Default 0 (vertical).
    axial_stiffness : float, optional
        Axial stiffness ka (kN/m). If None, must be set before analysis.
    lateral_stiffness : float, optional
        Lateral stiffness kl (kN/m). If None, assumed rigid laterally.
    axial_capacity_compression : float, optional
        Ultimate axial compression capacity (kN).
    axial_capacity_tension : float, optional
        Ultimate axial tension (uplift) capacity (kN).
    label : str, optional
        Pile label for identification.
    """
    x: float = 0.0
    y: float = 0.0
    batter_x: float = 0.0
    batter_y: float = 0.0
    axial_stiffness: Optional[float] = None
    lateral_stiffness: Optional[float] = None
    axial_capacity_compression: Optional[float] = None
    axial_capacity_tension: Optional[float] = None
    label: str = ""

    @property
    def is_vertical(self) -> bool:
        """True if pile is vertical (no batter)."""
        return abs(self.batter_x) < 0.01 and abs(self.batter_y) < 0.01

    @property
    def batter_x_rad(self) -> float:
        return math.radians(self.batter_x)

    @property
    def batter_y_rad(self) -> float:
        return math.radians(self.batter_y)

    def direction_cosines(self) -> Tuple[float, float, float]:
        """Unit vector along the pile axis (from head toward tip).

        Returns
        -------
        lx, ly, lz : float
            Direction cosines. lz is the vertical component (≈1 for vertical).
        """
        bx = self.batter_x_rad
        by = self.batter_y_rad
        lx = math.sin(bx)
        ly = math.sin(by)
        lz = math.sqrt(max(1.0 - lx**2 - ly**2, 0.0))
        return lx, ly, lz


def create_rectangular_layout(
    n_rows: int, n_cols: int,
    spacing_x: float, spacing_y: float,
    axial_stiffness: Optional[float] = None,
    lateral_stiffness: Optional[float] = None,
) -> List[GroupPile]:
    """Create a regular rectangular pile layout centered on (0,0).

    Parameters
    ----------
    n_rows : int
        Number of rows (along Y).
    n_cols : int
        Number of columns (along X).
    spacing_x : float
        Pile spacing in X-direction (m).
    spacing_y : float
        Pile spacing in Y-direction (m).
    axial_stiffness : float, optional
        Axial stiffness for all piles (kN/m).
    lateral_stiffness : float, optional
        Lateral stiffness for all piles (kN/m).

    Returns
    -------
    list of GroupPile
    """
    piles = []
    x_offset = (n_cols - 1) * spacing_x / 2
    y_offset = (n_rows - 1) * spacing_y / 2

    for row in range(n_rows):
        for col in range(n_cols):
            x = col * spacing_x - x_offset
            y = row * spacing_y - y_offset
            piles.append(GroupPile(
                x=x, y=y,
                axial_stiffness=axial_stiffness,
                lateral_stiffness=lateral_stiffness,
                label=f"P{row+1}-{col+1}",
            ))
    return piles
