"""
Footing geometry definitions for bearing capacity analysis.

Supports strip, rectangular, square, and circular footings with optional
eccentric loading (effective area method per Meyerhof, 1953).

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).

References:
    FHWA GEC-6, Chapter 6 (FHWA-IF-02-054)
    Meyerhof, G.G. (1953) — Effective area concept for eccentric loads.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass
class Footing:
    """Shallow foundation geometry.

    Parameters
    ----------
    width : float
        Footing width B (m). For circular footings, this is the diameter.
    length : float, optional
        Footing length L (m). If None:
        - For shape="strip", L is treated as infinite.
        - For shape="circular", L = width.
        - For shape="square", L = width.
    depth : float
        Embedment depth Df (m) from ground surface to footing base.
    shape : str
        Footing shape: "strip", "rectangular", "square", or "circular".
    base_tilt : float, optional
        Base inclination angle alpha (degrees) from horizontal. Default 0.
    eccentricity_B : float, optional
        Load eccentricity in the B direction, eB (m). Default 0.
    eccentricity_L : float, optional
        Load eccentricity in the L direction, eL (m). Default 0.

    Attributes
    ----------
    B_eff : float
        Effective width after eccentricity adjustment (m).
    L_eff : float
        Effective length after eccentricity adjustment (m).
    A_eff : float
        Effective bearing area (m²).
    """
    width: float
    length: Optional[float] = None
    depth: float = 0.0
    shape: str = "strip"
    base_tilt: float = 0.0
    eccentricity_B: float = 0.0
    eccentricity_L: float = 0.0

    def __post_init__(self):
        if self.width <= 0:
            raise ValueError(f"Footing width must be positive, got {self.width}")
        if self.depth < 0:
            raise ValueError(f"Embedment depth must be non-negative, got {self.depth}")

        self.shape = self.shape.lower()
        valid_shapes = ("strip", "rectangular", "square", "circular")
        if self.shape not in valid_shapes:
            raise ValueError(f"Shape must be one of {valid_shapes}, got '{self.shape}'")

        # Set length based on shape
        if self.shape == "strip":
            if self.length is None:
                self.length = 1e6  # effectively infinite
        elif self.shape == "square":
            self.length = self.width
        elif self.shape == "circular":
            self.length = self.width  # diameter = B = L
        elif self.shape == "rectangular":
            if self.length is None:
                raise ValueError("Length must be provided for rectangular footings")
            if self.length < self.width:
                # Convention: L >= B. Swap if needed.
                self.width, self.length = self.length, self.width

        # Eccentricity checks
        if abs(self.eccentricity_B) >= self.width / 2:
            raise ValueError(
                f"Eccentricity eB={self.eccentricity_B} m exceeds B/2={self.width/2} m; "
                "load is outside the footing"
            )
        if self.length is not None and abs(self.eccentricity_L) >= self.length / 2:
            if self.shape != "strip":
                raise ValueError(
                    f"Eccentricity eL={self.eccentricity_L} m exceeds L/2={self.length/2} m; "
                    "load is outside the footing"
                )

        if abs(self.base_tilt) > 30:
            warnings.warn(f"Base tilt of {self.base_tilt}° is large; results may be unreliable")

    @property
    def B_eff(self) -> float:
        """Effective width B' = B - 2*eB (m) per Meyerhof effective area method."""
        return self.width - 2.0 * abs(self.eccentricity_B)

    @property
    def L_eff(self) -> float:
        """Effective length L' = L - 2*eL (m) per Meyerhof effective area method."""
        return self.length - 2.0 * abs(self.eccentricity_L)

    @property
    def A_eff(self) -> float:
        """Effective bearing area (m²).

        For circular footings with eccentricity, this uses the effective
        rectangular approximation B' * L'.
        """
        if self.shape == "circular" and (self.eccentricity_B == 0 and self.eccentricity_L == 0):
            return math.pi / 4.0 * self.width ** 2
        return self.B_eff * self.L_eff

    @property
    def B_for_factors(self) -> float:
        """The B dimension to use in bearing capacity factors.

        This is always the shorter effective dimension (B' <= L').
        """
        b = self.B_eff
        l = self.L_eff
        return min(b, l)

    @property
    def L_for_factors(self) -> float:
        """The L dimension to use in bearing capacity factors.

        This is always the longer effective dimension (L' >= B').
        """
        b = self.B_eff
        l = self.L_eff
        return max(b, l)
