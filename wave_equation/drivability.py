"""
Drivability analysis for pile driving.

Performs wave equation analysis at multiple pile penetration depths
to assess whether a given hammer can drive the pile to the required
depth and capacity.

All units are SI: kN, m, seconds.

Refusal: a blow count above ``refusal_blow_count``. The default is 120
blows/ft (393.7 blows/m): GEC-12 Vol 2 Sec. 12.4, p. 161, "Hammer size,
energy setting, and cushion materials should always be chosen such that
the maximum expected penetration resistance is less than 120 blows/ft",
and the drivability study of Vol 3 (p. 43) calls 120 blows/ft "practical
refusal". The one exception p. 161 prints, piles through soft soils
seated on hard rock at 5 blows per 1/4 inch, is 240 blows/ft (787.4
blows/m), the refusal of the Vol 2 examples (pp. 171, 177, 188).

The pile analysed at each depth: the pile in the leads at every depth,
embedded to the depth, the shaft resistance only below grade (GM-301):
GEC-12 Vol 2 p. 224, "Length is the total pile length in the leads",
"Penetration refers to ... final penetration for Drivability Analyses",
and p. 236, the depths to be analyzed with the pile's length. That pile is
``pile_length``; left out, the deepest depth analysed, the shortest pile
that reaches it (p. 227: the penetration "cannot be greater than the pile
length"), stated in the result's choices. Before the GM-300 review the
left-out pile was as long as each depth (no stick-up), which no page
prints; at shallow depths it read far less tension. The shaft resistance
is spread over the embedded length by ``shaft_distribution`` ("uniform" by
default, or "triangular"; see wave_equation.soil_model).

References:
    FHWA GEC-12 Vol 2, Sec. 12.4 (p. 161) and 12.5.3 (pp. 167-172)
    FHWA GEC-12 Vol 3, Sec. D.10.3 (pp. 37-43)
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from wave_equation.hammer import Hammer, diesel_warning
from wave_equation.cushion import Cushion
from wave_equation.pile_model import discretize_pile, PileModel
from wave_equation.soil_model import SoilSetup, quake_weighting_choice
from wave_equation.time_integration import simulate_blow, BlowResult

#: 120 blows/ft in blows per metre, 120 / 0.3048 (GEC-12 Vol 2 p. 161;
#: Vol 3 p. 43).
REFUSAL_BLOWS_PER_M = 393.7


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
    refusal_blow_count : float
        The blow count (blows/m) above which the study called refusal.
    choices : list of dict
        The weights of the quakes in the permanent set, with the other
        weightings' blow counts (``quake_weighting_choice``; none with
        equal quakes), and the pile length analysed
        (``pile_length_choice``).
    warnings : list of str
        A diesel hammer's (``diesel_warning``).
    pile_length : float, optional
        The pile's length in the leads at every depth (m) (GM-301): the
        given ``pile_length``, else the deepest depth analysed.
    pile_length_given : bool
        Whether ``pile_length`` was given.
    shaft_distribution : str
        How the shaft resistance was spread over the embedded length.
    """
    points: List[DrivabilityPoint] = field(default_factory=list)
    can_drive: bool = True
    refusal_depth: float = 0.0
    refusal_blow_count: float = REFUSAL_BLOWS_PER_M
    choices: list = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    pile_length: Optional[float] = None
    pile_length_given: bool = False
    shaft_distribution: str = "uniform"

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
        if self.pile_length is not None:
            lines.append(f"  Pile length {self.pile_length:.2f} m at every "
                         "depth, embedded to the depth"
                         + ("" if self.pile_length_given else
                            " (the deepest depth; pile_length left out)"))
        lines.append(f"  Refusal: more than {self.refusal_blow_count:.1f} "
                     "blows/m")
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
        out = {
            "can_drive": self.can_drive,
            "refusal_depth_m": self.refusal_depth,
            "refusal_blow_count_per_m": self.refusal_blow_count,
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
        if self.pile_length is not None:
            out["pile_length_m"] = self.pile_length
        if self.shaft_distribution != "uniform":
            out["shaft_distribution"] = self.shaft_distribution
        if self.choices:
            out["choices"] = list(self.choices)
        if self.warnings:
            out["warnings"] = list(self.warnings)
        return out


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
    damping_toe: float = 0.49,
    helmet_weight: Optional[float] = None,
    refusal_blow_count: float = REFUSAL_BLOWS_PER_M,
    damping_model: str = "smith",
    pile_length: Optional[float] = None,
    shaft_distribution: str = "uniform",
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
        Quake values (m); GEC-12 Vol 2 p. 211 (see SoilSetup).
    damping_side, damping_toe : float
        Smith damping values (s/m); GEC-12 Vol 2 p. 211 (see SoilSetup).
    helmet_weight : float
        Helmet weight (kN). Required (GM-299; see simulate_blow).
    refusal_blow_count : float
        Blow count (blows/m) above which the pile has refused. Default
        393.7 (120 blows/ft, GEC-12 Vol 2 p. 161); 787.4 (240 blows/ft)
        for a pile seated on hard rock (p. 161). See the module docstring.
    damping_model : str
        "smith" (default): damping proportional to the mobilized static
        resistance, R_d = J*R_static*v. "smith_viscous": damping
        proportional to R_ultimate (UFC 3-220-20 pp. 487-488). See
        wave_equation.soil_model.
    pile_length : float, optional
        The pile's total length in the leads (m), the same at every depth,
        embedded to the depth (GM-301; GEC-12 Vol 2 pp. 224, 236). No depth
        may pass it (p. 227). None (default): the deepest depth analysed,
        the shortest pile that reaches it (p. 227), stated in the choices.
    shaft_distribution : str
        "uniform" (default) or "triangular" over the embedded length
        (GM-301; GEC-12 Vol 2 pp. 162, 179, 182). See SoilSetup.

    Returns
    -------
    DrivabilityResult
    """
    if len(depths) != len(R_at_depth):
        raise ValueError("depths and R_at_depth must have same length")
    if pile_length is not None:
        deeper = [d for d in depths if d > pile_length * (1.0 + 1e-9)]
        if deeper:
            raise ValueError(
                f"depths {deeper} are deeper than the pile_length "
                f"{pile_length:g} m (GEC-12 Vol 2 p. 227: the penetration "
                f"'cannot be greater than the pile length')")

    length_given = pile_length is not None
    if not length_given and len(depths):
        # The pile in the leads, the same at every depth (p. 224), no
        # shorter than the deepest penetration (p. 227).
        pile_length = float(max(depths))

    if skin_fractions is None:
        skin_fractions = [0.5] * len(depths)

    points = []
    toe_peaks = []
    can_drive = True
    refusal_depth = 0.0

    for i, (depth, R_ult) in enumerate(zip(depths, R_at_depth)):
        pile = discretize_pile(
            pile_length, pile_area, pile_E, segment_length, pile_unit_weight
        )

        soil = SoilSetup(
            R_ultimate=R_ult,
            skin_fraction=skin_fractions[i],
            quake_side=quake_side,
            quake_toe=quake_toe,
            damping_side=damping_side,
            damping_toe=damping_toe,
            damping_model=damping_model,
            shaft_distribution=shaft_distribution,
            embedded_length=depth,
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
        toe_peaks.append((skin_fractions[i], result.max_toe_displacement))

        if bc > refusal_blow_count and can_drive:
            can_drive = False
            refusal_depth = depth

    quakes = quake_weighting_choice(
        quake_side, quake_toe, toe_peaks,
        " at depths " + " / ".join(f"{d:g}" for d in depths) + " m"
        if len(depths) > 1 else "")
    choices = [quakes] if quakes else []
    if pile_length is not None:
        choices.append(pile_length_choice(
            pile_length, float(max(depths)), length_given))
    warning = diesel_warning(hammer)
    return DrivabilityResult(
        points=points,
        can_drive=can_drive,
        refusal_depth=refusal_depth,
        refusal_blow_count=refusal_blow_count,
        choices=choices,
        warnings=[warning] if warning else [],
        pile_length=pile_length,
        pile_length_given=length_given,
        shaft_distribution=shaft_distribution,
    )


def pile_length_choice(pile_length: float, deepest: float,
                       given: bool) -> dict:
    """The pile length a drivability study analysed at every depth, as a
    result's ``choices`` entry (GM-300 review): ``pile_length`` when
    ``given``, else the deepest depth analysed. GEC-12 Vol 2 prints no
    per-depth length but a Temporary Length before splicing (p. 236),
    which is not offered."""
    from geotech_common.choices import choice, how_given

    basis = ("GEC-12 Vol 2 p. 224: \"Length is the total pile length in "
             "the leads\", and a drivability analysis takes the final "
             "penetration; p. 236 analyses each depth with that length")
    if abs(pile_length - deepest) <= 1e-9 * max(deepest, 1.0):
        chosen = (f"{pile_length:g} m, the deepest depth analysed, the same "
                  f"pile at every depth, embedded to each depth")
        basis += ("; p. 227: the penetration \"cannot be greater than the "
                  "pile length\", so the deepest depth is the shortest pile "
                  "that reaches it")
    else:
        chosen = (f"{pile_length:g} m, the same pile at every depth, "
                  f"embedded to each depth")
    return choice(
        "pile length in the leads (pile_length)", chosen,
        how_given(pile_length if given else None, None, name="pile_length",
                  took=pile_length),
        basis,
        ["a longer pile in the leads (give pile_length): p. 224, \"if plans "
         "require a pile of 50 feet in length but the contractor is driving "
         "60 long piles, the proper analysis length would be the full 60 "
         "feet\""])
