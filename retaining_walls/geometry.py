"""
Wall geometry definitions for cantilever and MSE retaining walls.

All units are SI: meters, degrees, kPa.

References:
    Das, B.M., Principles of Foundation Engineering, Ch 13
    FHWA GEC-11, Chapter 4
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class CantileverWallGeometry:
    """Cantilever retaining wall geometry.

    Parameters
    ----------
    wall_height : float
        Total retained height H (m), measured from base of footing
        to top of stem.
    base_width : float, optional
        Base slab width B (m). If None, auto-sized to ~0.6*H.
    toe_length : float, optional
        Toe projection from front face of stem (m). If None, ~0.1*base_width.
    stem_thickness_top : float, optional
        Stem thickness at top (m). Default 0.30.
    stem_thickness_base : float, optional
        Stem thickness at base of stem (m). If None, auto-sized.
    base_thickness : float, optional
        Base slab thickness (m). Default 0.6.
    has_shear_key : bool, optional
        Whether wall has a shear key. Default False.
    key_depth : float, optional
        Shear key depth below base slab (m). Default 0.
    backfill_slope : float, optional
        Slope of backfill behind wall (degrees). Default 0.
    surcharge : float, optional
        Uniform surcharge on backfill (kPa). Default 0.
    """
    wall_height: float
    base_width: Optional[float] = None
    toe_length: Optional[float] = None
    stem_thickness_top: float = 0.30
    stem_thickness_base: Optional[float] = None
    base_thickness: float = 0.60
    has_shear_key: bool = False
    key_depth: float = 0.0
    backfill_slope: float = 0.0
    surcharge: float = 0.0

    def __post_init__(self):
        if self.wall_height <= 0:
            raise ValueError(f"Wall height must be positive, got {self.wall_height}")
        if self.base_thickness <= 0:
            raise ValueError(f"Base thickness must be positive, got {self.base_thickness}")

        # Auto-size if not specified
        if self.base_width is None:
            self.base_width = round(0.6 * self.wall_height, 2)
        if self.toe_length is None:
            self.toe_length = round(0.1 * self.base_width, 2)
        if self.stem_thickness_base is None:
            self.stem_thickness_base = round(
                max(0.08 * self.wall_height + 0.2, self.stem_thickness_top), 2
            )

    @property
    def stem_height(self) -> float:
        """Height of stem above base slab (m)."""
        return self.wall_height - self.base_thickness

    @property
    def heel_length(self) -> float:
        """Heel projection behind back face of stem (m)."""
        return self.base_width - self.toe_length - self.stem_thickness_base

    @property
    def H_active(self) -> float:
        """Height for active pressure computation (m).
        Includes backfill slope contribution.
        """
        if self.backfill_slope > 0:
            i_rad = math.radians(self.backfill_slope)
            return self.wall_height + self.heel_length * math.tan(i_rad)
        return self.wall_height


@dataclass
class MSEWallGeometry:
    """MSE (Mechanically Stabilized Earth) wall geometry.

    Parameters
    ----------
    wall_height : float
        Total wall height H (m).
    reinforcement_length : float, optional
        Length of reinforcement strips/grids (m).
        If None, auto-sized to max(0.7*H, 2.5).
    reinforcement_spacing : float, optional
        Vertical spacing between reinforcement levels Sv (m).
        Default 0.60.
    backfill_slope : float, optional
        Slope of backfill above wall (degrees). Default 0.
    surcharge : float, optional
        Uniform surcharge (kPa). Default 0.
    """
    wall_height: float
    reinforcement_length: Optional[float] = None
    reinforcement_spacing: float = 0.60
    backfill_slope: float = 0.0
    surcharge: float = 0.0

    def __post_init__(self):
        if self.wall_height <= 0:
            raise ValueError(f"Wall height must be positive, got {self.wall_height}")
        if self.reinforcement_spacing <= 0:
            raise ValueError(f"Spacing must be positive, got {self.reinforcement_spacing}")

        if self.reinforcement_length is None:
            self.reinforcement_length = round(max(0.7 * self.wall_height, 2.5), 2)

    @property
    def n_reinforcement_levels(self) -> int:
        """Number of reinforcement levels."""
        return max(1, int(self.wall_height / self.reinforcement_spacing))

    @property
    def reinforcement_depths(self) -> List[float]:
        """Depth of each reinforcement level from top of wall (m)."""
        n = self.n_reinforcement_levels
        return [
            round(self.reinforcement_spacing * (i + 0.5), 3)
            for i in range(n)
        ]
