"""
Drilled shaft geometry definition.

All units are SI: meters, kPa, kN.

References:
    FHWA GEC-10, Chapter 12
"""

import math
import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass
class DrillShaft:
    """Drilled shaft cross-section and geometry.

    Parameters
    ----------
    diameter : float
        Shaft diameter (m).
    length : float
        Total shaft length / embedment depth (m).
    socket_diameter : float, optional
        Rock socket diameter if different from shaft diameter (m).
        If None, same as diameter.
    socket_length : float, optional
        Rock socket length (m). Default 0 (no socket).
    bell_diameter : float, optional
        Bell (underream) diameter at base (m). If None, no bell.
    casing_depth : float, optional
        Depth of permanent casing (m). Excludes cased zone from
        side resistance. Default 0.
    concrete_fc : float, optional
        Concrete compressive strength f'c (kPa). Default 28000 (28 MPa).
    rebar_area : float, optional
        Total longitudinal reinforcement area (m²). Default 0.

    Properties
    ----------
    area : float
        Cross-sectional area of shaft (m²).
    perimeter : float
        Shaft perimeter for skin friction (m).
    tip_area : float
        Tip area, accounts for bell if present (m²).
    """
    diameter: float
    length: float
    socket_diameter: Optional[float] = None
    socket_length: float = 0.0
    bell_diameter: Optional[float] = None
    casing_depth: float = 0.0
    concrete_fc: float = 28000.0
    rebar_area: float = 0.0

    def __post_init__(self):
        if self.diameter <= 0:
            raise ValueError(f"Diameter must be positive, got {self.diameter}")
        if self.length <= 0:
            raise ValueError(f"Length must be positive, got {self.length}")
        if self.socket_length < 0:
            raise ValueError(f"Socket length must be non-negative, got {self.socket_length}")
        if self.casing_depth < 0:
            raise ValueError(f"Casing depth must be non-negative, got {self.casing_depth}")
        if self.bell_diameter is not None and self.bell_diameter < self.diameter:
            raise ValueError(
                f"Bell diameter ({self.bell_diameter}) must be >= shaft diameter ({self.diameter})"
            )
        if self.casing_depth > self.length:
            warnings.warn(
                f"Casing depth ({self.casing_depth}m) exceeds shaft length ({self.length}m)"
            )

    @property
    def area(self) -> float:
        """Shaft cross-sectional area (m²)."""
        return math.pi * self.diameter**2 / 4

    @property
    def perimeter(self) -> float:
        """Shaft perimeter for skin friction (m)."""
        return math.pi * self.diameter

    @property
    def socket_perimeter(self) -> float:
        """Rock socket perimeter (m). Uses socket_diameter if specified."""
        d = self.socket_diameter if self.socket_diameter is not None else self.diameter
        return math.pi * d

    @property
    def tip_area(self) -> float:
        """Tip area (m²). Uses bell diameter if belled."""
        d = self.bell_diameter if self.bell_diameter is not None else self.diameter
        return math.pi * d**2 / 4
