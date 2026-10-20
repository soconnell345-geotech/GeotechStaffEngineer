"""
Drilled shaft geometry definition.

All units are SI: meters, kPa, kN.

References:
    FHWA GEC-10 (FHWA-NHI-18-024), Sec. 10.3.4, Select Trial Lengths and Diameters
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
        If None, same as diameter. One diameter for every rock layer:
        GEC-10 sizes a rock socket by one diameter B (Eq. 10-21, Example
        10-4, p. 10-27) and gives no socket diameter per rock layer. It
        is the diameter of the rock side resistance and, with the tip in
        rock, of the base: GEC-10 Example 10-4 (p. 10-28) takes R_BN =
        pi/4 B^2 q_BN with B the 6 ft socket below 6.5 ft casing (GM-221).
    socket_length : float, optional
        Rock socket length (m). Default 0 (no socket).
    bell_diameter : float, optional
        Bell (underream) diameter at base (m). If None, no bell.
    casing_depth : float, optional
        Depth of permanent casing (m). Excludes cased zone from
        side resistance. Default 0.
    concrete_fc : float, optional
        Concrete compressive strength f'c (kPa). Default 28000 (28 MPa).
        Carried for the caller's own structural checks; the GEC-10
        capacity in this module does not use it (GM-160).
    rebar_area : float, optional
        Total longitudinal reinforcement area (m²). Default 0. Likewise
        not used by the capacity.
    fractured_rock_thickness : float, optional
        Thickness (m) of decomposed or fractured rock below the top of
        rock (the top of the profile's first rock layer) that is "not
        considered part of the socket for design purposes" (GEC-10 p.
        8-15): no rock side resistance over it, while the soil above
        keeps its own (``casing_depth`` takes both away). Default 0: all
        rock below the top 1.5 m and the casing is socket (GM-220).

    Properties
    ----------
    area : float
        Cross-sectional area of shaft (m²).
    perimeter : float
        Shaft perimeter for skin friction (m).
    tip_area : float
        Tip area, accounts for bell if present (m²). The base area the
        capacity uses is :meth:`base_area`, the socket's with the tip in
        rock.
    """
    diameter: float
    length: float
    socket_diameter: Optional[float] = None
    socket_length: float = 0.0
    bell_diameter: Optional[float] = None
    casing_depth: float = 0.0
    concrete_fc: float = 28000.0
    rebar_area: float = 0.0
    fractured_rock_thickness: float = 0.0

    def __post_init__(self):
        if self.diameter <= 0:
            raise ValueError(f"Diameter must be positive, got {self.diameter}")
        if self.length <= 0:
            raise ValueError(f"Length must be positive, got {self.length}")
        if self.socket_diameter is not None and self.socket_diameter <= 0:
            raise ValueError(
                f"Socket diameter must be positive, got {self.socket_diameter}")
        if self.socket_length < 0:
            raise ValueError(f"Socket length must be non-negative, got {self.socket_length}")
        if self.casing_depth < 0:
            raise ValueError(f"Casing depth must be non-negative, got {self.casing_depth}")
        if self.fractured_rock_thickness < 0:
            raise ValueError(
                "fractured_rock_thickness must be non-negative, got "
                f"{self.fractured_rock_thickness}")
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

    def base_diameter(self, tip_in_rock: bool) -> float:
        """Diameter (m) of the base the end bearing acts on: the bell's if
        belled; with the tip in rock, the socket's (GEC-10 Example 10-4,
        p. 10-28: R_BN = pi/4 B^2 q_BN, B the socket diameter, GM-221);
        else the shaft's."""
        if self.bell_diameter is not None:
            return self.bell_diameter
        if tip_in_rock and self.socket_diameter is not None:
            return self.socket_diameter
        return self.diameter

    def base_area(self, tip_in_rock: bool) -> float:
        """Base area (m²), pi/4 times :meth:`base_diameter` squared."""
        return math.pi * self.base_diameter(tip_in_rock) ** 2 / 4
