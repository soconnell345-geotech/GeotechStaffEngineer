"""
Bearing graph generation for wave equation analysis.

Runs multiple wave equation simulations at different assumed ultimate
resistances to produce a bearing graph (blow count vs capacity).

All units are SI: kN, m, seconds.

References:
    FHWA GEC-12, Section 12.5
    WEAP87 Manual, Chapter 7
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from wave_equation.hammer import Hammer
from wave_equation.cushion import Cushion
from wave_equation.pile_model import PileModel
from wave_equation.soil_model import SoilSetup
from wave_equation.time_integration import BlowResult, simulate_blow


@dataclass
class BearingGraphResult:
    """Bearing graph output.

    Attributes
    ----------
    R_values : np.ndarray
        Ultimate resistance values analyzed (kN).
    blow_counts : np.ndarray
        Blow count (blows/m) for each R_ultimate.
    permanent_sets : np.ndarray
        Permanent set per blow (m) for each R_ultimate.
    max_comp_stresses : np.ndarray
        Maximum compression stress (kPa) for each R_ultimate.
    max_tens_stresses : np.ndarray
        Maximum tension stress (kPa) for each R_ultimate.
    max_comp_forces : np.ndarray
        Maximum compression force (kN) for each R_ultimate.
    blow_results : list of BlowResult
        Full results for each R_ultimate analyzed.
    """
    R_values: np.ndarray = field(default_factory=lambda: np.array([]))
    blow_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    permanent_sets: np.ndarray = field(default_factory=lambda: np.array([]))
    max_comp_stresses: np.ndarray = field(default_factory=lambda: np.array([]))
    max_tens_stresses: np.ndarray = field(default_factory=lambda: np.array([]))
    max_comp_forces: np.ndarray = field(default_factory=lambda: np.array([]))
    blow_results: List[BlowResult] = field(default_factory=list)

    def capacity_at_blow_count(self, target_blow_count: float) -> float:
        """Interpolate capacity for a given blow count.

        Parameters
        ----------
        target_blow_count : float
            Desired blow count (blows/m).

        Returns
        -------
        float
            Interpolated ultimate capacity (kN). Returns 0 if
            blow count is outside the range.
        """
        valid = self.blow_counts > 0
        if not np.any(valid):
            return 0.0
        bc = self.blow_counts[valid]
        rv = self.R_values[valid]
        if target_blow_count < bc.min() or target_blow_count > bc.max():
            return float(np.interp(target_blow_count, bc, rv))
        return float(np.interp(target_blow_count, bc, rv))

    def summary(self) -> str:
        """Text summary of the bearing graph."""
        lines = [
            "=" * 60,
            "  WAVE EQUATION BEARING GRAPH",
            "=" * 60,
            "",
            f"  {'Rult (kN)':>12} {'Set (mm)':>10} {'Blows/m':>10} "
            f"{'Comp (kPa)':>12} {'Tens (kPa)':>12}",
            "-" * 60,
        ]
        for i in range(len(self.R_values)):
            set_mm = self.permanent_sets[i] * 1000
            bc = self.blow_counts[i]
            lines.append(
                f"  {self.R_values[i]:>12,.0f} {set_mm:>10.2f} {bc:>10.0f} "
                f"{self.max_comp_stresses[i]:>12,.0f} "
                f"{self.max_tens_stresses[i]:>12,.0f}"
            )
        lines.extend(["", "=" * 60])
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "R_values_kN": self.R_values.tolist(),
            "blow_counts_per_m": self.blow_counts.tolist(),
            "permanent_sets_m": self.permanent_sets.tolist(),
            "max_comp_stresses_kPa": self.max_comp_stresses.tolist(),
            "max_tens_stresses_kPa": self.max_tens_stresses.tolist(),
        }


def generate_bearing_graph(
    hammer: Hammer,
    cushion: Cushion,
    pile: PileModel,
    skin_fraction: float = 0.5,
    quake_side: float = 0.0025,
    quake_toe: float = 0.0025,
    damping_side: float = 0.16,
    damping_toe: float = 0.50,
    R_min: float = 200.0,
    R_max: float = 2000.0,
    R_step: float = 200.0,
    helmet_weight: float = 5.0,
    max_time: float = 0.10,
) -> BearingGraphResult:
    """Generate a bearing graph by running wave equation at multiple Rult.

    Parameters
    ----------
    hammer : Hammer
        Hammer model.
    cushion : Cushion
        Pile cushion.
    pile : PileModel
        Discretized pile.
    skin_fraction : float
        Fraction of Rult as skin friction.
    quake_side, quake_toe : float
        Quake values (m).
    damping_side, damping_toe : float
        Smith damping values (s/m).
    R_min, R_max, R_step : float
        Range of ultimate resistances to analyze (kN).
    helmet_weight : float
        Helmet weight (kN).
    max_time : float
        Maximum simulation time per blow (s).

    Returns
    -------
    BearingGraphResult
    """
    R_values = np.arange(R_min, R_max + R_step / 2, R_step)
    n_runs = len(R_values)

    blow_counts = np.zeros(n_runs)
    perm_sets = np.zeros(n_runs)
    comp_stresses = np.zeros(n_runs)
    tens_stresses = np.zeros(n_runs)
    comp_forces = np.zeros(n_runs)
    blow_results = []

    for i, R_ult in enumerate(R_values):
        soil = SoilSetup(
            R_ultimate=R_ult,
            skin_fraction=skin_fraction,
            quake_side=quake_side,
            quake_toe=quake_toe,
            damping_side=damping_side,
            damping_toe=damping_toe,
        )

        result = simulate_blow(
            hammer, cushion, pile, soil,
            helmet_weight=helmet_weight,
            max_time=max_time,
        )

        perm_sets[i] = result.permanent_set
        comp_stresses[i] = result.max_compression_stress
        tens_stresses[i] = result.max_tension_stress
        comp_forces[i] = result.max_pile_force

        # Blow count = 1/set (blows per meter)
        if result.permanent_set > 1e-6:
            blow_counts[i] = 1.0 / result.permanent_set
        else:
            blow_counts[i] = 1e6  # Essentially refusal

        blow_results.append(result)

    return BearingGraphResult(
        R_values=R_values,
        blow_counts=blow_counts,
        permanent_sets=perm_sets,
        max_comp_stresses=comp_stresses,
        max_tens_stresses=tens_stresses,
        max_comp_forces=comp_forces,
        blow_results=blow_results,
    )
