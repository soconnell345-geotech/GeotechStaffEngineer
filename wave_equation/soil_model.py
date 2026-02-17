"""
Smith soil model for wave equation analysis.

Implements the classic Smith (1960) soil resistance model with
quake and damping parameters. Distributes total soil resistance
to individual pile segments for skin friction and end bearing.

All units are SI: kN, m, seconds.

References:
    Smith, E.A.L. (1960) "Bearing Capacity of Piles"
    WEAP87 Manual, Chapter 4
    FHWA GEC-12, Table 12-3 (typical parameters)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SmithSoilModel:
    """Smith soil resistance model for a single pile segment.

    The total resistance at a segment is:
        R = R_static + J * R_ultimate * v

    where R_static is the elasto-plastic static resistance,
    J is the Smith damping factor, R_ultimate is the ultimate
    static capacity, and v is the pile segment velocity.

    The damping force (J * R_ultimate * v) is proportional to
    the ultimate resistance, not the currently mobilized static
    resistance. This is the standard GRLWEAP/GEC-12 formulation.

    Parameters
    ----------
    R_ultimate : float
        Ultimate static resistance at this segment (kN).
    quake : float
        Elastic limit displacement (m). Typical 0.0025 m (0.1 in).
    damping : float
        Smith damping factor (s/m). Typical: 0.16 s/m (skin, sand),
        0.65 s/m (skin, clay), 0.50 s/m (toe).
    """
    R_ultimate: float
    quake: float = 0.0025
    damping: float = 0.16

    def static_resistance(self, displacement: float) -> float:
        """Compute static soil resistance.

        Elasto-plastic: linear up to quake, then constant.

        Parameters
        ----------
        displacement : float
            Pile segment displacement (m). Positive downward.

        Returns
        -------
        float
            Static resistance force (kN). Positive opposes downward motion.
        """
        if self.R_ultimate == 0:
            return 0.0
        if abs(displacement) <= self.quake:
            return self.R_ultimate * displacement / self.quake
        else:
            return self.R_ultimate * (1.0 if displacement > 0 else -1.0)

    def total_resistance(self, displacement: float, velocity: float) -> float:
        """Compute total (static + dynamic) soil resistance.

        R = R_static + J * R_ultimate * v

        Smith damping: the velocity-dependent damping force is proportional
        to the *ultimate* static resistance (R_ultimate), not the currently
        mobilized static resistance. This is the standard GRLWEAP/GEC-12
        formulation with damping constants from FHWA GEC-12 Table 12-3.

        Damping is only applied during loading (velocity in the direction
        that compresses the soil spring). During unloading/rebound, only
        the static resistance acts.

        Parameters
        ----------
        displacement : float
            Pile segment displacement (m).
        velocity : float
            Pile segment velocity (m/s). Positive downward.

        Returns
        -------
        float
            Total resistance force (kN).
        """
        R_s = self.static_resistance(displacement)
        # Smith damping: R_d = J * R_u * v (proportional to ultimate, not mobilized)
        # Applied only during loading (velocity same direction as displacement)
        if velocity > 0 and displacement > 0:
            R_d = self.damping * self.R_ultimate * velocity
            return R_s + R_d
        elif velocity < 0 and displacement < 0:
            R_d = self.damping * self.R_ultimate * abs(velocity)
            return R_s - R_d  # R_s is negative here; R_d adds to magnitude
        else:
            return R_s


@dataclass
class SoilSetup:
    """Complete soil setup for wave equation analysis.

    Distributes ultimate resistance among pile segments as skin
    friction and end bearing.

    Parameters
    ----------
    R_ultimate : float
        Total ultimate static resistance (kN).
    skin_fraction : float
        Fraction of Rult carried by skin friction (0 to 1).
    quake_side : float
        Side quake (m). Default 0.0025 m.
    quake_toe : float
        Toe quake (m). Default 0.0025 m.
    damping_side : float
        Side Smith damping (s/m). Default 0.16 (sand).
    damping_toe : float
        Toe Smith damping (s/m). Default 0.50 (sand).
    """
    R_ultimate: float
    skin_fraction: float = 0.5
    quake_side: float = 0.0025
    quake_toe: float = 0.0025
    damping_side: float = 0.16
    damping_toe: float = 0.50

    def __post_init__(self):
        if not 0 <= self.skin_fraction <= 1:
            raise ValueError(
                f"skin_fraction must be 0-1, got {self.skin_fraction}"
            )

    @property
    def R_skin(self) -> float:
        """Total skin friction resistance (kN)."""
        return self.R_ultimate * self.skin_fraction

    @property
    def R_toe(self) -> float:
        """Toe (end bearing) resistance (kN)."""
        return self.R_ultimate * (1.0 - self.skin_fraction)

    def create_segment_models(
        self, n_segments: int
    ) -> tuple:
        """Create soil spring models for each pile segment.

        Distributes skin friction uniformly along the pile.
        Toe resistance is applied to the last segment only.

        Parameters
        ----------
        n_segments : int
            Number of pile segments.

        Returns
        -------
        side_models : list of SmithSoilModel
            Skin friction springs for each segment.
        toe_model : SmithSoilModel
            End bearing spring at pile toe.
        """
        # Uniform skin friction distribution
        R_per_segment = self.R_skin / n_segments if n_segments > 0 else 0

        side_models = [
            SmithSoilModel(R_per_segment, self.quake_side, self.damping_side)
            for _ in range(n_segments)
        ]

        toe_model = SmithSoilModel(
            self.R_toe, self.quake_toe, self.damping_toe
        )

        return side_models, toe_model
