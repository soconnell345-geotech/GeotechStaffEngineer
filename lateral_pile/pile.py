"""
Pile definition module.

Defines the Pile class for representing single piles or drilled shafts
with geometry, material properties, and flexural rigidity.

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np


@dataclass
class PileSection:
    """A section of pile with constant properties over a depth range.

    Parameters
    ----------
    top : float
        Depth to top of section (m), measured from pile head.
    bottom : float
        Depth to bottom of section (m), measured from pile head.
    EI : float
        Flexural rigidity of this section (kN-m^2).
    """
    top: float
    bottom: float
    EI: float

    def __post_init__(self):
        if self.bottom <= self.top:
            raise ValueError(f"Section bottom ({self.bottom}) must be greater than top ({self.top})")
        if self.EI <= 0:
            raise ValueError(f"EI must be positive, got {self.EI}")


@dataclass
class Pile:
    """Single pile or drilled shaft definition.

    Supports solid circular, hollow circular (pipe), and arbitrary cross-sections.
    For pipe piles, provide diameter and thickness. For solid piles, provide
    diameter only. For arbitrary sections, provide moment_of_inertia directly.

    Parameters
    ----------
    length : float
        Embedded pile length (m).
    diameter : float
        Outer diameter of the pile (m).
    E : float
        Young's modulus of pile material (kPa). Steel ~ 200e6 kPa, concrete ~ 25e6 kPa.
    thickness : float, optional
        Wall thickness for pipe piles (m). If None, pile is treated as solid.
    moment_of_inertia : float, optional
        Moment of inertia (m^4). If None, computed from diameter and thickness.
    sections : list of PileSection, optional
        Variable EI sections along the pile. If provided, overrides uniform E*I.

    Examples
    --------
    Steel pipe pile:
    >>> pile = Pile(length=20.0, diameter=0.6, thickness=0.012, E=200e6)

    Solid concrete pile:
    >>> pile = Pile(length=15.0, diameter=0.45, E=25e6)

    Custom moment of inertia:
    >>> pile = Pile(length=20.0, diameter=0.6, E=200e6, moment_of_inertia=1.5e-4)
    """
    length: float
    diameter: float
    E: float
    thickness: Optional[float] = None
    moment_of_inertia: Optional[float] = None
    sections: Optional[List[PileSection]] = None

    def __post_init__(self):
        if self.length <= 0:
            raise ValueError(f"Pile length must be positive, got {self.length}")
        if self.diameter <= 0:
            raise ValueError(f"Pile diameter must be positive, got {self.diameter}")
        if self.E <= 0:
            raise ValueError(f"Young's modulus must be positive, got {self.E}")
        if self.thickness is not None:
            if self.thickness <= 0:
                raise ValueError(f"Wall thickness must be positive, got {self.thickness}")
            if self.thickness >= self.diameter / 2:
                raise ValueError(f"Wall thickness ({self.thickness}) must be less than radius ({self.diameter/2})")

        # Compute moment of inertia if not provided
        if self.moment_of_inertia is None:
            self.moment_of_inertia = self._compute_moment_of_inertia()

        # Parameter range warnings
        if self.diameter < 0.1:
            warnings.warn(f"Pile diameter {self.diameter} m is unusually small")
        if self.diameter > 5.0:
            warnings.warn(f"Pile diameter {self.diameter} m is unusually large")
        if self.length / self.diameter < 3:
            warnings.warn(f"L/D ratio {self.length/self.diameter:.1f} is very low; "
                          "method may not be applicable")

    def _compute_moment_of_inertia(self) -> float:
        """Compute moment of inertia from geometry.

        Returns
        -------
        float
            Moment of inertia (m^4).
        """
        r_outer = self.diameter / 2.0
        if self.thickness is not None:
            # Hollow circular (pipe pile)
            r_inner = r_outer - self.thickness
            I = math.pi / 4.0 * (r_outer**4 - r_inner**4)
        else:
            # Solid circular
            I = math.pi / 4.0 * r_outer**4
        return I

    @property
    def EI(self) -> float:
        """Flexural rigidity (kN-m^2) for uniform pile."""
        return self.E * self.moment_of_inertia

    @property
    def area(self) -> float:
        """Cross-sectional area (m^2)."""
        r_outer = self.diameter / 2.0
        if self.thickness is not None:
            r_inner = r_outer - self.thickness
            return math.pi * (r_outer**2 - r_inner**2)
        else:
            return math.pi * r_outer**2

    def get_EI_at_depth(self, z: float) -> float:
        """Get flexural rigidity at a given depth.

        Parameters
        ----------
        z : float
            Depth from pile head (m).

        Returns
        -------
        float
            EI at depth z (kN-m^2).
        """
        if self.sections is not None:
            for section in self.sections:
                if section.top <= z <= section.bottom:
                    return section.EI
            # If depth is outside all sections, use the nearest section
            warnings.warn(f"Depth {z} is outside defined sections, using uniform EI")
        return self.EI

    def get_EI_profile(self, depths: np.ndarray) -> np.ndarray:
        """Get EI values at an array of depths.

        Parameters
        ----------
        depths : numpy.ndarray
            Array of depths from pile head (m).

        Returns
        -------
        numpy.ndarray
            Array of EI values (kN-m^2).
        """
        if self.sections is None:
            return np.full_like(depths, self.EI)
        return np.array([self.get_EI_at_depth(z) for z in depths])
