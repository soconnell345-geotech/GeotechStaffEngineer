"""
Drivability analysis for pile driving.

Performs wave equation analysis at multiple pile penetration depths
to assess whether a given hammer can drive the pile to the required
depth and capacity.

All units are SI: kN, m, seconds.

References:
    FHWA GEC-12, Section 12.6
    WEAP87 Manual, Chapter 8
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from wave_equation.hammer import Hammer
from wave_equation.cushion import Cushion
from wave_equation.pile_model import discretize_pile, PileModel
from wave_equation.soil_model import SoilSetup
from wave_equation.time_integration import simulate_blow, BlowResult


@dataclass
class DrivabilityPoint:
    """Results at one penetration depth.

    Attributes
    ----------
    depth : float
        Pile penetration depth (m).
    R_ultimate : float
        Ultimate resistance at this depth (kN).
    permanent_set : float
        Permanent set per blow (m).
    blow_count : float
        Blow count (blows/m).
    max_comp_stress : float
        Max compression stress (kPa).
    max_tens_stress : float
        Max tension stress (kPa).
    """
    depth: float
    R_ultimate: float
    permanent_set: float = 0.0
    blow_count: float = 0.0
    max_comp_stress: float = 0.0
    max_tens_stress: float = 0.0


@dataclass
class DrivabilityResult:
    """Complete drivability study results.

    Attributes
    ----------
    points : list of DrivabilityPoint
        Results at each depth.
    can_drive : bool
        True if pile can be driven to all specified depths.
    refusal_depth : float
        Depth (m) at which refusal occurs, if any. 0 if no refusal.
    """
    points: List[DrivabilityPoint] = field(default_factory=list)
    can_drive: bool = True
    refusal_depth: float = 0.0

    def summary(self) -> str:
        """Text summary of drivability study."""
        lines = [
            "=" * 70,
            "  DRIVABILITY ANALYSIS",
            "=" * 70,
            "",
            f"  {'Depth (m)':>10} {'Rult (kN)':>10} {'Set (mm)':>10} "
            f"{'Blows/m':>10} {'Comp (kPa)':>12} {'Tens (kPa)':>12}",
            "-" * 70,
        ]
        for pt in self.points:
            set_mm = pt.permanent_set * 1000
            lines.append(
                f"  {pt.depth:>10.1f} {pt.R_ultimate:>10.0f} {set_mm:>10.2f} "
                f"{pt.blow_count:>10.0f} {pt.max_comp_stress:>12,.0f} "
                f"{pt.max_tens_stress:>12,.0f}"
            )
        lines.append("")
        if self.can_drive:
            lines.append("  Result: PILE CAN BE DRIVEN to all specified depths.")
        else:
            lines.append(
                f"  Result: REFUSAL at depth {self.refusal_depth:.1f} m"
            )
        lines.extend(["", "=" * 70])
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "can_drive": self.can_drive,
            "refusal_depth_m": self.refusal_depth,
            "points": [
                {
                    "depth_m": p.depth,
                    "R_ultimate_kN": p.R_ultimate,
                    "permanent_set_m": p.permanent_set,
                    "blow_count_per_m": p.blow_count,
                    "max_comp_stress_kPa": p.max_comp_stress,
                    "max_tens_stress_kPa": p.max_tens_stress,
                }
                for p in self.points
            ],
        }


def drivability_study(
    hammer: Hammer,
    cushion: Cushion,
    pile_area: float,
    pile_E: float,
    pile_unit_weight: float,
    depths: list,
    R_at_depth: list,
    skin_fractions: Optional[list] = None,
    segment_length: float = 1.0,
    quake_side: float = 0.0025,
    quake_toe: float = 0.0025,
    damping_side: float = 0.16,
    damping_toe: float = 0.50,
    helmet_weight: float = 5.0,
    refusal_blow_count: float = 3000.0,
) -> DrivabilityResult:
    """Run drivability analysis at multiple depths.

    Parameters
    ----------
    hammer : Hammer
        Hammer model.
    cushion : Cushion
        Pile cushion.
    pile_area : float
        Pile cross-sectional area (m^2).
    pile_E : float
        Pile elastic modulus (kPa).
    pile_unit_weight : float
        Pile material unit weight (kN/m^3).
    depths : list of float
        Penetration depths to analyze (m).
    R_at_depth : list of float
        Ultimate resistance at each depth (kN).
    skin_fractions : list of float, optional
        Skin friction fraction at each depth. Default 0.5 for all.
    segment_length : float
        Pile segment length (m).
    quake_side, quake_toe : float
        Quake values (m).
    damping_side, damping_toe : float
        Smith damping values (s/m).
    helmet_weight : float
        Helmet weight (kN).
    refusal_blow_count : float
        Blow count (blows/m) considered as refusal.

    Returns
    -------
    DrivabilityResult
    """
    if len(depths) != len(R_at_depth):
        raise ValueError("depths and R_at_depth must have same length")

    if skin_fractions is None:
        skin_fractions = [0.5] * len(depths)

    points = []
    can_drive = True
    refusal_depth = 0.0

    for i, (depth, R_ult) in enumerate(zip(depths, R_at_depth)):
        pile = discretize_pile(
            depth, pile_area, pile_E, segment_length, pile_unit_weight
        )

        soil = SoilSetup(
            R_ultimate=R_ult,
            skin_fraction=skin_fractions[i],
            quake_side=quake_side,
            quake_toe=quake_toe,
            damping_side=damping_side,
            damping_toe=damping_toe,
        )

        result = simulate_blow(
            hammer, cushion, pile, soil,
            helmet_weight=helmet_weight,
        )

        if result.permanent_set > 1e-6:
            bc = 1.0 / result.permanent_set
        else:
            bc = 1e6

        pt = DrivabilityPoint(
            depth=depth,
            R_ultimate=R_ult,
            permanent_set=result.permanent_set,
            blow_count=bc,
            max_comp_stress=result.max_compression_stress,
            max_tens_stress=result.max_tension_stress,
        )
        points.append(pt)

        if bc > refusal_blow_count and can_drive:
            can_drive = False
            refusal_depth = depth

    return DrivabilityResult(
        points=points,
        can_drive=can_drive,
        refusal_depth=refusal_depth,
    )
