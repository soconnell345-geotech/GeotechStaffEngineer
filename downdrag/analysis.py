"""
Downdrag (negative skin friction) analysis using the neutral plane method.

Implements the Fellenius unified method (2004/2006) for pile downdrag:
1. Force equilibrium to find the neutral plane depth.
2. Settlement compatibility (soil settlement = pile settlement at NP).
3. Structural and geotechnical limit state checks.

The neutral plane basis is a choice (``neutral_plane_method``): the cited
bases (force equilibrium at 100 / 50 / 0 % toe mobilization, settlement
compatibility, the bearing-layer rule for end-bearing piles, the pile toe,
Endo) are all evaluated on the same inputs and returned side by side, with a
``judgment`` block when they differ materially. See DESIGN.md
"Neutral-plane methods" for the sources.

Supports fill placement and groundwater drawdown as settlement triggers.

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).

References
----------
- Fellenius, B.H. (2006). "Results of static loading tests on driven piles."
- Fellenius, B.H. (2004). "Unified design of piled foundations with emphasis
  on settlement analysis." ASCE GSP 125.
- AASHTO LRFD Bridge Design Specifications, Section 10.7.3.7.
- UFC 3-220-20, 16 Jan 2025, Chapter 6, Eqs 6-51 through 6-53, 6-80.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from downdrag.soil import DowndragSoilProfile, DowndragSoilLayer
from downdrag.results import DowndragResult
from downdrag.cgpr56 import ENDO_NEUTRAL_PLANE_RATIOS


#: Accepted values of ``DowndragAnalysis.neutral_plane_method``.
NEUTRAL_PLANE_METHODS = (
    "auto", "force_equilibrium", "settlement_compatibility",
    "bearing_layer_top", "pile_toe", "endo",
)

#: Toe mobilizations always reported (UFC 3-220-20 §6-5.8.4.2 and GEC-12
#: §7.3.6.1 step 4: 0 %, 50 % and 100 % of the nominal base resistance).
COMPARISON_TOE_MOBILIZATIONS = (1.0, 0.5, 0.0)

#: When the applicable bases differ by more than this, the result carries a
#: ``judgment`` block: neutral planes more than max(0.5 m, 5 % of the pile
#: length) apart, or drag loads more than max(10 kN, 10 % of the largest)
#: apart.
NP_SPREAD_ABS_M = 0.5
NP_SPREAD_REL_LENGTH = 0.05
DRAG_SPREAD_ABS_KN = 10.0
DRAG_SPREAD_REL = 0.10

#: Where each basis is stated (printed page numbers). UFC 3-220-20 =
#: 16 Jan 2025 (DM 7.2); GEC-12 = FHWA-NHI-16-009 Vol I; CGPR #56 =
#: Greenfield & Filz (2009), sections only (not public-domain, not in the
#: repo).
_SOURCE = {
    "fe_full": ("UFC 3-220-20 §6-7.4 steps 3-5, Fig 6-26 (pp. 476-478); "
                "GEC-12 §7.3.6.1 steps 4-5, Fig 7-60 point A (pp. 343-346)"),
    "fe_partial": ("UFC 3-220-20 §6-5.8.4.2 (p. 447) and App. B-5.5 Table "
                   "B-28 (p. 640); GEC-12 §7.3.6.1 steps 2, 4-5, Figs "
                   "7-59/7-60 (pp. 342-346); Siegel et al. (2013)"),
    "sc": ("UFC 3-220-20 §6-5.8.4.3, Fig 6-19 (pp. 449-450); GEC-12 "
           "§7.3.5.7 (p. 334); Fellenius (2004, 2021)"),
    "blt": "UFC 3-220-20 §6-7.4 step 5 (p. 477) and §6-5.7 (p. 438)",
    "toe": ("GEC-12 §7.3.5.7 (p. 336, after Goudreault & Fellenius 1994); "
            "CGPR #56 Table 3.1 'end bearing'"),
    "endo": "CGPR #56 §3.2.1, Table 3.1 (Endo et al. 1969; Little 1994)",
}


def _fe_name(mobilization: float) -> str:
    """Comparison-row name of force equilibrium at a toe mobilization."""
    return f"force_equilibrium, toe {round(mobilization * 100):d}%"


@dataclass
class DowndragAnalysis:
    """Downdrag analysis using the Fellenius unified neutral plane method.

    Parameters
    ----------
    soil : DowndragSoilProfile
        Soil profile with strength and consolidation parameters.
    pile_length : float
        Embedded pile length (m).
    pile_diameter : float
        Pile diameter (m). Used to compute perimeter if not provided.
    pile_perimeter : float, optional
        Pile perimeter (m). If None, computed as pi*D.
    pile_area : float, optional
        Pile cross-sectional area (m^2). If None, computed as pi/4*D^2.
    pile_E : float
        Pile Young's modulus (kPa). Default 200e6 (steel).
    pile_unit_weight : float
        Pile material unit weight (kN/m^3). Default 24.0 (concrete).
        For steel pipe piles, use ~78.5 * (area_ratio).
    Q_dead : float
        Dead load at pile head (kN). Only dead load causes downdrag.
    structural_capacity : float, optional
        Factored structural resistance of pile (kN). For limit state check.
    allowable_settlement : float, optional
        Allowable pile settlement (m). For serviceability check.
    fill_thickness : float
        Thickness of new fill placed at ground surface (m). Default 0.
    fill_unit_weight : float
        Unit weight of fill material (kN/m^3). Default 19.0.
    gw_drawdown : float
        Groundwater drawdown from original GWT (m). Default 0.
    Nt : float, optional
        Toe bearing capacity factor. If None, estimated from tip layer phi.
    n_sublayers : int
        Number of sublayers per soil layer for discretization. Default 10.
    neutral_plane_method : str
        How the neutral plane is located (DESIGN.md "Neutral-plane
        methods" gives the sources):

        - ``"auto"`` (default): UFC 3-220-20's procedure. Force
          equilibrium (§6-7.4 steps 3-5) with the toe at
          ``toe_mobilization`` of its resistance; when the curves do not
          cross, settlement compatibility (Fig 6-19); no plane when the
          dead load exceeds the nominal resistance.
        - ``"force_equilibrium"``: force equilibrium only (raises when the
          curves do not cross).
        - ``"settlement_compatibility"``: pile and soil settle equally,
          with an elastic-plastic toe (Fellenius unified method).
        - ``"bearing_layer_top"``: at the base of the settling soil, for
          an end-bearing pile (UFC §6-7.4 step 5).
        - ``"pile_toe"``: at the toe, the whole shaft as drag (the upper
          bound; GEC-12 §7.3.5.7 uses it for group settlement).
        - ``"endo"``: at a fraction of the embedment (CGPR #56 Table 3.1);
          needs ``endo_bearing_condition``.

        Whatever is chosen, every basis is evaluated and returned in
        ``neutral_plane_comparison``.
    toe_mobilization : float
        Fraction (0-1) of the toe resistance mobilized for force
        equilibrium. Default 1.0 (UFC §6-7.4 step 3, full mobilization,
        conservative for drag). 0.5 and 0.0 are the lower brackets of UFC
        §6-5.8.4.2 / GEC-12 §7.3.6.1 (always reported in the comparison).
    endo_bearing_condition : str, optional
        CGPR #56 Table 3.1 condition for the Endo basis: ``"floating"``,
        ``"stiff_flexible"`` or ``"end_bearing"``.
    skin_friction_stress : str
        Effective stress used for beta-method unit skin friction and the
        cohesionless toe: ``"initial"`` (default, sigma'v0) or ``"final"``
        (sigma'v0 + the fill / drawdown stress change, as UFC 3-220-20
        §6-7.4 step 2 asks for the drag-force evaluation).

    Examples
    --------
    >>> from downdrag import DowndragAnalysis, DowndragSoilProfile, DowndragSoilLayer
    >>> layers = [
    ...     DowndragSoilLayer(thickness=3.0, soil_type="cohesionless",
    ...         unit_weight=19.0, phi=30.0, description="Fill"),
    ...     DowndragSoilLayer(thickness=10.0, soil_type="cohesive",
    ...         unit_weight=17.0, cu=30.0, settling=True,
    ...         Cc=0.3, Cr=0.05, e0=1.0, description="Soft clay"),
    ...     DowndragSoilLayer(thickness=7.0, soil_type="cohesionless",
    ...         unit_weight=20.0, phi=35.0, description="Dense sand"),
    ... ]
    >>> soil = DowndragSoilProfile(layers=layers, gwt_depth=2.0)
    >>> analysis = DowndragAnalysis(
    ...     soil=soil, pile_length=18.0, pile_diameter=0.3,
    ...     Q_dead=500.0, fill_thickness=3.0, fill_unit_weight=19.0)
    >>> result = analysis.compute()
    >>> print(result.summary())
    """
    soil: DowndragSoilProfile
    pile_length: float
    pile_diameter: float
    pile_perimeter: Optional[float] = None
    pile_area: Optional[float] = None
    pile_E: float = 200e6
    pile_unit_weight: float = 24.0
    Q_dead: float = 0.0
    structural_capacity: Optional[float] = None
    allowable_settlement: Optional[float] = None
    fill_thickness: float = 0.0
    fill_unit_weight: float = 19.0
    gw_drawdown: float = 0.0
    Nt: Optional[float] = None
    n_sublayers: int = 10
    neutral_plane_method: str = "auto"
    toe_mobilization: float = 1.0
    endo_bearing_condition: Optional[str] = None
    skin_friction_stress: str = "initial"

    def __post_init__(self):
        if self.pile_length <= 0:
            raise ValueError(
                f"Pile length must be positive, got {self.pile_length}"
            )
        if self.pile_diameter <= 0:
            raise ValueError(
                f"Pile diameter must be positive, got {self.pile_diameter}"
            )
        if self.pile_perimeter is None:
            self.pile_perimeter = math.pi * self.pile_diameter
        if self.pile_area is None:
            self.pile_area = math.pi / 4.0 * self.pile_diameter**2
        if self.Q_dead < 0:
            raise ValueError(
                f"Dead load must be non-negative, got {self.Q_dead}"
            )
        if self.fill_thickness < 0:
            raise ValueError(
                f"fill_thickness must be non-negative, got {self.fill_thickness}"
            )
        if self.gw_drawdown < 0:
            raise ValueError(
                f"gw_drawdown must be non-negative, got {self.gw_drawdown}"
            )
        if self.neutral_plane_method not in NEUTRAL_PLANE_METHODS:
            raise ValueError(
                f"neutral_plane_method must be one of "
                f"{list(NEUTRAL_PLANE_METHODS)}, got "
                f"'{self.neutral_plane_method}'")
        if not 0.0 <= self.toe_mobilization <= 1.0:
            raise ValueError(
                f"toe_mobilization must be between 0 and 1 (a fraction of "
                f"the toe resistance), got {self.toe_mobilization}")
        if (self.endo_bearing_condition is not None
                and self.endo_bearing_condition
                not in ENDO_NEUTRAL_PLANE_RATIOS):
            raise ValueError(
                f"endo_bearing_condition must be one of "
                f"{sorted(ENDO_NEUTRAL_PLANE_RATIOS)} (CGPR #56 Table 3.1), "
                f"got '{self.endo_bearing_condition}'")
        if (self.neutral_plane_method == "endo"
                and self.endo_bearing_condition is None):
            raise ValueError(
                "neutral_plane_method='endo' needs endo_bearing_condition: "
                "'floating', 'stiff_flexible' or 'end_bearing' (CGPR #56 "
                "Table 3.1)")
        if self.skin_friction_stress not in ("initial", "final"):
            raise ValueError(
                f"skin_friction_stress must be 'initial' or 'final', got "
                f"'{self.skin_friction_stress}'")

    def compute(self) -> DowndragResult:
        """Run the downdrag analysis.

        The neutral plane is located by ``neutral_plane_method`` (default
        ``"auto"``: UFC 3-220-20's procedure). Every cited basis is also
        evaluated on the same inputs and returned in
        ``neutral_plane_comparison``; when the applicable bases differ
        materially the result carries a ``judgment`` block saying which
        one was used and why.

        Returns
        -------
        DowndragResult
            Analysis results including neutral plane depth, dragload,
            settlement, and limit state checks.
        """
        S = self._state()
        z_nodes, dz, fs = S["z"], S["dz"], S["fs"]
        R = S["R_toe"]
        L = self.pile_length
        m = self.toe_mobilization
        req = self.neutral_plane_method
        rows = self._comparison_rows(S)
        by_name = {r["name"]: r for r in rows}
        warning_list: List[str] = []

        if S["overloaded"]:
            # The dead load alone exceeds the nominal geotechnical
            # resistance: UFC 3-220-20 §6-7.4 step 5 limiting case, "there
            # is no neutral plane". The pile plunges, every part of the
            # shaft moves down relative to the soil, so no drag develops.
            used_name = None
            np_method = "none"
            z_np = 0.0
            toe_force = R
            capacity = float(S["resist_full"][0])
            np_basis = (
                "No neutral plane: the dead load exceeds the pile's nominal "
                "geotechnical resistance (UFC 3-220-20 §6-7.4 step 5, the "
                "limiting case). Reported at the pile head with zero drag.")
            warning_list.append(
                f"NO NEUTRAL PLANE: Q_dead = {self.Q_dead:.1f} kN exceeds "
                f"the total geotechnical resistance (toe + full shaft) = "
                f"{capacity:.1f} kN. The pile is overloaded; a drag force "
                f"is not meaningful. Increase the pile length or size.")
        else:
            if req == "auto":
                used_name = (_fe_name(m) if by_name[_fe_name(m)]["applies"]
                             else "settlement_compatibility")
            elif req == "force_equilibrium":
                used_name = _fe_name(m)
            elif req == "endo":
                used_name = f"endo ({self.endo_bearing_condition})"
            else:
                used_name = req
            row = by_name[used_name]
            if not row["applies"]:
                raise ValueError(
                    f"neutral_plane_method='{req}' does not locate the "
                    f"neutral plane for these inputs: {row['reason']} Use "
                    f"neutral_plane_method='auto' or another basis.")
            np_method = row["method"]
            z_np = float(row["neutral_plane_depth_m"])
            toe_force = float(row["toe_force_kN"])
            np_basis = self._basis_text(S, row)
            self._basis_warnings(S, np_method, z_np, toe_force, warning_list)
        if self.skin_friction_stress == "final":
            np_basis += (
                " Unit skin friction (beta layers) and the cohesionless toe "
                "use the final effective stress, sigma'v0 + the fill / "
                "drawdown change (UFC 3-220-20 §6-7.4 step 2).")

        # Compute dragload and positive resistance
        dragload = self._compute_dragload(z_nodes, dz, fs, z_np)
        pile_weight_to_np = self.pile_unit_weight * self.pile_area * z_np
        max_pile_load = self.Q_dead + dragload + pile_weight_to_np

        positive_skin = self._compute_positive_resistance(z_nodes, dz, fs, z_np)
        total_resistance = positive_skin + R

        # Compute axial load distribution along pile
        if np_method == "force_equilibrium":
            axial_load = self._compute_axial_load_distribution(
                z_nodes, dz, fs, m * R)
        elif np_method == "none":
            axial_load = self._compute_axial_load_distribution(
                z_nodes, dz, fs, R)
        else:
            axial_load = self._axial_load_with_mobilized_toe(
                z_nodes, S["drag_from_top"], S["resist_full"], R, z_np)

        # Compute pile settlement at neutral plane
        elastic_short = self._compute_elastic_shortening(z_nodes, dz,
                                                          axial_load, z_np)
        toe_settle = self._compute_toe_settlement(
            S["sigma_v0"][-1], S["delta_sigma"][-1], toe_force)
        pile_settlement = elastic_short + toe_settle

        # Settlement at the neutral plane from the soil profile
        soil_settle_at_np = float(np.interp(z_np, z_nodes,
                                            S["soil_settlement"]))

        # Use the larger of pile settlement and soil settlement at NP
        # as the controlling settlement (they should be close if compatible)
        settlement = max(pile_settlement, soil_settle_at_np)

        judgment = self._judgment(rows, used_name, warning_list)

        # Limit state checks

        # Structural: UFC Eq 6-80 LRFD factored demand
        #   1.25*Q_dead + 1.10*(Q_np - Q_dead) <= P_r
        # where Q_np = max_pile_load (total load at neutral plane)
        structural_ok = None
        structural_demand = None
        if self.structural_capacity is not None:
            drag_force = max_pile_load - self.Q_dead  # dragload + pile weight
            structural_demand = 1.25 * self.Q_dead + 1.10 * drag_force
            structural_ok = bool(structural_demand <= self.structural_capacity)

        geotechnical_ok = None
        if total_resistance > 0:
            # Per Fellenius/AASHTO/UFC: dragload is NOT included in
            # geotechnical check — it cancels at the neutral plane
            geotechnical_ok = bool(self.Q_dead <= total_resistance)

        settlement_ok = None
        if self.allowable_settlement is not None:
            settlement_ok = bool(settlement <= self.allowable_settlement)

        return DowndragResult(
            neutral_plane_depth=z_np,
            dragload=dragload,
            max_pile_load=max_pile_load,
            Q_dead=self.Q_dead,
            pile_weight_to_np=pile_weight_to_np,
            positive_skin_friction=positive_skin,
            toe_resistance=R,
            total_resistance=total_resistance,
            pile_settlement=pile_settlement,
            elastic_shortening=elastic_short,
            toe_settlement=toe_settle,
            soil_settlement_at_np=soil_settle_at_np,
            z=z_nodes,
            axial_load=axial_load,
            soil_settlement_profile=S["soil_settlement"],
            unit_skin_friction=fs,
            structural_ok=structural_ok,
            structural_demand=structural_demand,
            geotechnical_ok=geotechnical_ok,
            settlement_ok=settlement_ok,
            pile_length=L,
            pile_diameter=self.pile_diameter,
            neutral_plane_method=np_method,
            neutral_plane_basis=np_basis,
            toe_force_mobilized=toe_force,
            warnings=warning_list,
            neutral_plane_requested=req,
            toe_mobilization=m,
            skin_friction_stress=self.skin_friction_stress,
            neutral_plane_comparison=rows,
            judgment=judgment,
        )

    # ── Neutral-plane bases ───────────────────────────────────────────────

    def _state(self) -> dict:
        """Everything the neutral-plane bases share: the depth grid, unit
        skin friction, the soil settlement profile, the load and resistance
        curves (toe fully mobilized), the settlement mismatch, and the
        planes force equilibrium and settlement compatibility give."""
        z_nodes, dz = self._discretize()
        sigma_v0 = np.array([
            self.soil.effective_stress_at_depth(z) for z in z_nodes
        ])
        # Stress change at each node (from fill and/or GW drawdown)
        delta_sigma = self._compute_stress_change(z_nodes)
        sigma_fs = (sigma_v0 + delta_sigma
                    if self.skin_friction_stress == "final" else sigma_v0)
        fs = self._compute_skin_friction(z_nodes, sigma_fs)
        # Soil settlement profile, measured from the toe level
        soil_settlement = self._compute_soil_settlement(
            z_nodes, dz, sigma_v0, delta_sigma)
        R = self._compute_toe_resistance(float(sigma_fs[-1]))

        # Load and resistance curves with the toe fully mobilized
        z_fe_full, drag_from_top, resist_full = self._find_neutral_plane(
            z_nodes, dz, fs, R)
        overloaded = bool(drag_from_top[0] - resist_full[0] > 0)
        m = self.toe_mobilization
        z_fe_m = (z_fe_full if m == 1.0 else
                  self._find_neutral_plane(z_nodes, dz, fs, m * R)[0])

        # Settlement compatibility (UFC 3-220-20 Fig 6-19: pile and soil
        # settle equally at the NP), evaluated at every node.
        g, toe_force_at = self._settlement_mismatch(
            z_nodes, dz, drag_from_top, resist_full, R,
            soil_settlement, sigma_v0[-1], delta_sigma[-1])
        z_sc, sc_toe_yields = _compatible_plane(z_nodes, g, z_fe_full)
        return {
            "z": z_nodes, "dz": dz, "fs": fs,
            "sigma_v0": sigma_v0, "delta_sigma": delta_sigma,
            "soil_settlement": soil_settlement,
            "R_toe": R,
            "drag_from_top": drag_from_top, "resist_full": resist_full,
            "f_below": resist_full - R,
            "overloaded": overloaded,
            "z_fe_full": z_fe_full, "z_fe_m": z_fe_m,
            "g": g, "toe_force_at": toe_force_at,
            # where the soil settlement falls to the (elastic-toe) pile's
            "z_se": _deepest_positive_crossing(z_nodes, g),
            "z_sc": z_sc, "sc_toe_yields": sc_toe_yields,
            "any_soil_settles": bool(np.any(soil_settlement > 0)),
            "s_fill_only": self._compute_toe_settlement(
                sigma_v0[-1], delta_sigma[-1], 0.0),
        }

    def _pile_settlement_at(self, S: dict, z: float,
                            toe_force: float) -> float:
        """Pile settlement at depth ``z`` with the neutral plane there,
        measured like the soil profile (from the soil at the toe level):
        toe penetration (bearing-stratum compression under the toe force,
        less what the fill / drawdown alone causes) + elastic compression
        of the pile between ``z`` and the toe (UFC 3-220-20 Fig 6-19)."""
        zn, fb = S["z"], S["f_below"]
        q_np = float(np.interp(z, zn, S["drag_from_top"]))
        f_np = float(np.interp(z, zn, fb))
        toe_raw = q_np - f_np
        # Positive friction below is only partly mobilized when it would
        # exceed the load at the plane (toe force then zero).
        k = 1.0 if (toe_raw >= 0 or f_np <= 0) else q_np / f_np
        below = zn > z
        zz = np.concatenate(([z], zn[below]))
        ff = np.concatenate(([f_np], fb[below]))
        q = q_np - k * (f_np - ff)
        AE = self.pile_area * self.pile_E
        compression = (float(np.trapezoid(q, zz)) / AE
                       if AE > 0 and len(zz) > 1 else 0.0)
        penetration = max(0.0, self._compute_toe_settlement(
            S["sigma_v0"][-1], S["delta_sigma"][-1], toe_force)
            - S["s_fill_only"])
        return penetration + compression

    def _plane_numbers(self, S: dict, z: float, toe_force: float) -> dict:
        """Drag load, load at the plane and the soil / pile settlement at
        the plane for a neutral plane at ``z`` carrying ``toe_force``."""
        drag = self._compute_dragload(S["z"], S["dz"], S["fs"], z)
        q_np = self.Q_dead + drag + self.pile_unit_weight * self.pile_area * z
        s_soil = float(np.interp(z, S["z"], S["soil_settlement"]))
        s_pile = self._pile_settlement_at(S, z, toe_force)
        R = S["R_toe"]
        if R > 0 and toe_force >= R * (1.0 - 1e-9) and s_soil > s_pile:
            # Toe at its full resistance: it yields and the pile goes down
            # with the soil (elastic-plastic toe).
            s_pile = s_soil
        return {
            "neutral_plane_depth_m": float(z),
            "dragload_kN": float(drag),
            "max_pile_load_kN": float(q_np),
            "toe_force_kN": float(toe_force),
            "soil_settlement_at_np_mm": s_soil * 1000.0,
            "pile_settlement_at_np_mm": s_pile * 1000.0,
        }

    def _settling_base_depth(self, S: dict) -> Optional[float]:
        """Depth to the base of the deepest settling layer that actually
        settles along the pile, or None when no soil settles."""
        sub = S["soil_settlement"][:-1] - S["soil_settlement"][1:]
        z_mid = 0.5 * (S["z"][:-1] + S["z"][1:])
        base = None
        top = 0.0
        for layer in self.soil.layers:
            bot = top + layer.thickness
            if layer.settling:
                inside = (z_mid > top) & (z_mid < bot)
                if np.any(sub[inside] > 0):
                    base = bot
            top = bot
        return base

    def _comparison_rows(self, S: dict) -> list:
        """Every cited neutral-plane basis on the same inputs.

        Each row: name, method, toe_mobilization, role ('basis' = a design
        basis; 'check' = settlement compatibility when no soil settlement
        is computed; 'bound' = the pile-toe upper bound), source,
        assumptions, applies, reason, and (when it applies) the plane
        depth, drag load, load at the plane, toe force, and the soil and
        pile settlement at the plane (both from the soil at the toe level).
        """
        L = self.pile_length
        R = S["R_toe"]
        rows = []

        def row(name, method, role, source, assumptions, mob=None):
            r = {"name": name, "method": method, "toe_mobilization": mob,
                 "role": role, "source": source,
                 "assumptions": assumptions, "applies": False,
                 "reason": "", "neutral_plane_depth_m": None,
                 "dragload_kN": None, "max_pile_load_kN": None,
                 "toe_force_kN": None, "soil_settlement_at_np_mm": None,
                 "pile_settlement_at_np_mm": None}
            rows.append(r)
            return r

        def fill(r, z, toe):
            r.update(self._plane_numbers(S, z, toe))
            r["applies"] = True

        def imposed(r, z, what):
            """A plane set by a rule: check force equilibrium can hold."""
            q_np = float(np.interp(z, S["z"], S["drag_from_top"]))
            f_np = float(np.interp(z, S["z"], S["f_below"]))
            toe_raw = q_np - f_np
            if toe_raw > R * (1.0 + 1e-6) + 1e-9:
                r["reason"] = (
                    f"with the plane {what} ({z:.2f} m) the toe would have "
                    f"to carry {toe_raw:.0f} kN, more than its resistance "
                    f"({R:.0f} kN): force equilibrium puts the plane "
                    f"higher.")
                return
            fill(r, z, float(min(max(toe_raw, 0.0), R)))

        none_reason = ("no neutral plane: the dead load exceeds the nominal "
                       "geotechnical resistance (UFC 3-220-20 §6-7.4 step "
                       "5).")
        mobs = list(COMPARISON_TOE_MOBILIZATIONS)
        if not any(abs(self.toe_mobilization - x) < 1e-9 for x in mobs):
            mobs.append(self.toe_mobilization)
            mobs.sort(reverse=True)
        f_total = float(S["f_below"][0])
        for mob in mobs:
            pct = round(mob * 100)
            if mob == 1.0:
                r = row(_fe_name(mob), "force_equilibrium", "basis",
                        _SOURCE["fe_full"],
                        "Side and base fully mobilized. Conservative for the "
                        "drag force (structural limit state); not "
                        "conservative for settlement (UFC §6-5.8.4.2).",
                        mob)
            else:
                r = row(_fe_name(mob), "force_equilibrium", "basis",
                        _SOURCE["fe_partial"],
                        f"Side fully mobilized, base at {pct}% of its "
                        f"resistance: the 0/50/100% bracket for settlement "
                        f"(Siegel et al. 2013).", mob)
            if S["overloaded"]:
                r["reason"] = none_reason
                continue
            z = (S["z_fe_full"] if mob == 1.0 else self._find_neutral_plane(
                S["z"], S["dz"], S["fs"], mob * R)[0])
            if z is None:
                q0 = float(S["drag_from_top"][0])
                if q0 > mob * R + f_total:
                    r["reason"] = (
                        f"the curves do not cross: Q_dead ({q0:.0f} kN) "
                        f"exceeds {pct}% of the toe resistance + the whole "
                        f"shaft ({mob * R + f_total:.0f} kN).")
                else:
                    r["reason"] = (
                        f"the curves do not cross: {pct}% of the toe "
                        f"resistance ({mob * R:.0f} kN) exceeds all the pile "
                        f"can deliver to its toe "
                        f"({float(S['drag_from_top'][-1]):.0f} kN, whole "
                        f"shaft as drag).")
                continue
            fill(r, z, mob * R)

        # Settlement compatibility (Fellenius unified method)
        settles = S["any_soil_settles"]
        r = row("settlement_compatibility", "settlement_compatibility",
                "basis" if settles else "check", _SOURCE["sc"],
                "Pile and soil settle equally at the plane (Fig 6-19). Toe "
                "elastic-plastic: penetration from bearing-stratum "
                "compression under the toe force (UFC Eqs 6-51 to 6-54), "
                "capped at the toe resistance; side fully mobilized.")
        if S["overloaded"]:
            r["reason"] = none_reason
        else:
            z = S["z_sc"]
            toe = (R if S["sc_toe_yields"]
                   else float(np.interp(z, S["z"], S["toe_force_at"])))
            fill(r, z, toe)
            if not settles:
                r["reason"] = (
                    "no soil settlement is computed (no loaded settling "
                    "layer), so compatibility finds no drag; UFC §6-5.7 / "
                    "FHWA (2016): the pile-soil stiffness contrast alone "
                    "gives some drag force.")

        # Bearing-layer rule for end-bearing piles
        r = row("bearing_layer_top", "bearing_layer_top", "basis",
                _SOURCE["blt"],
                "End-bearing pile in a stratum much stiffer than the "
                "compressible soil: plane at the base of the settling soil.")
        z_b = self._settling_base_depth(S)
        if S["overloaded"]:
            r["reason"] = none_reason
        elif z_b is None:
            r["reason"] = ("no soil settles (no settling layer loaded by fill "
                           "or drawdown).")
        elif z_b > L + 1e-9:
            r["reason"] = (
                f"the toe ({L:.2f} m) is inside the settling soil (its base "
                f"is at {z_b:.2f} m): a floating pile, not end-bearing.")
        else:
            imposed(r, z_b, "at the base of the settling soil")

        # Pile toe: the upper bound
        r = row("pile_toe", "pile_toe", "bound", _SOURCE["toe"],
                "Upper bound: the whole shaft as drag. GEC-12 uses it for "
                "group settlement, not for a single pile's drag.")
        if S["overloaded"]:
            r["reason"] = none_reason
        else:
            imposed(r, L, "at the toe")

        # Endo (CGPR #56 Table 3.1)
        cond = self.endo_bearing_condition
        if cond is None:
            r = row("endo", "endo", "basis", _SOURCE["endo"],
                    "Plane assumed at a fraction of the embedment; an "
                    "estimate / check.")
            r["reason"] = ("needs endo_bearing_condition ('floating', "
                           "'stiff_flexible' or 'end_bearing', CGPR #56 "
                           "Table 3.1).")
        else:
            ratio = ENDO_NEUTRAL_PLANE_RATIOS[cond]
            r = row(f"endo ({cond})", "endo", "basis", _SOURCE["endo"],
                    f"Plane assumed at {ratio:g} x the embedment "
                    f"('{cond}'); an estimate / check, not computed.")
            if S["overloaded"]:
                r["reason"] = none_reason
            else:
                imposed(r, ratio * L, f"at {ratio:g} L")
        return rows

    def _basis_text(self, S: dict, row: dict) -> str:
        """One-paragraph statement of the basis actually used."""
        L = self.pile_length
        method = row["method"]
        z_np = row["neutral_plane_depth_m"]
        m = row["toe_mobilization"]
        if method == "force_equilibrium":
            if m == 1.0:
                return (
                    "Force equilibrium: the load curve (Q_dead + pile weight "
                    "+ negative skin friction) meets the resistance curve "
                    "(toe resistance + positive skin friction) at this "
                    "depth, with side and base resistance fully mobilized "
                    "(UFC 3-220-20 §6-7.4 steps 3-5; Fellenius 2004).")
            return (
                f"Force equilibrium with the toe at {round(m * 100)}% of its "
                f"resistance (toe_mobilization = {m:g}): the load curve "
                f"meets the resistance curve (mobilized toe + positive skin "
                f"friction) at this depth (UFC 3-220-20 §6-5.8.4.2 and "
                f"§6-7.4 steps 3-5; GEC-12 §7.3.6.1; Siegel et al. 2013).")
        if method == "settlement_compatibility":
            if not S["any_soil_settles"]:
                return (
                    "Settlement compatibility: no soil settles (no settling "
                    "layer, or no fill / drawdown to load it), so the pile "
                    "settles at least as much as the soil everywhere and no "
                    "negative skin friction develops. Neutral plane at the "
                    "pile head, zero drag.")
            if S["sc_toe_yields"]:
                return (
                    "Settlement compatibility: down to the force-equilibrium "
                    "crossing the soil settles more than the pile would on "
                    "an elastic toe, so the toe yields at its full "
                    "resistance and the plane is the force-equilibrium plane "
                    "(UFC 3-220-20 §6-5.8.4.3 and Fig 6-19; Fellenius "
                    "2021).")
            if z_np >= L - 1e-9:
                return (
                    "Settlement compatibility, end-bearing case: the soil "
                    "settles more than the pile all the way down to the toe, "
                    "so the neutral plane is at the pile toe (UFC 3-220-20 "
                    "§6-7.4 step 5; Fellenius 2004).")
            return (
                "Settlement compatibility: the depth where the computed "
                "soil settlement equals the pile settlement (UFC 3-220-20 "
                "Fig 6-19). For an end-bearing pile in a stratum stiffer "
                "than the compressible soil this falls near the top of "
                "the bearing layer (UFC 3-220-20 §6-7.4 step 5).")
        if method == "bearing_layer_top":
            return (
                "Bearing-layer rule: neutral plane at the base of the "
                "settling soil (the top of the bearing stratum), for an "
                "end-bearing pile in a stratum much stiffer than the "
                "compressible soil (UFC 3-220-20 §6-7.4 step 5). Not a "
                "computed equilibrium.")
        if method == "pile_toe":
            return (
                "Neutral plane at the pile toe, the whole shaft as drag: the "
                "upper bound on drag (chosen by the caller). GEC-12 §7.3.5.7 "
                "places the plane at the toe for GROUP settlement "
                "(Goudreault & Fellenius 1994).")
        cond = self.endo_bearing_condition
        return (
            f"Endo method: neutral plane assumed at "
            f"{ENDO_NEUTRAL_PLANE_RATIOS[cond]:g} x the embedment for a "
            f"'{cond}' pile (CGPR #56 §3.2.1, Table 3.1). An estimate / "
            f"check, not a computed plane.")

    def _basis_warnings(self, S: dict, np_method: str, z_np: float,
                        toe_force: float, warning_list: list) -> None:
        """The warnings that go with the basis used."""
        z_se = S["z_se"]
        R = S["R_toe"]
        m = self.toe_mobilization
        if np_method == "force_equilibrium":
            if S["any_soil_settles"] and z_np > z_se + 1e-6:
                warning_list.append(
                    f"The force-equilibrium neutral plane ({z_np:.2f} m) "
                    f"lies below {z_se:.2f} m, where the computed soil "
                    f"settlement falls to the pile's; the friction between "
                    f"them is counted as drag although that soil settles "
                    f"less than the pile. With full mobilization that is "
                    f"UFC 3-220-20 §6-7.4's conservative assumption for "
                    f"drag force. For settlement, full base mobilization is "
                    f"not conservative (§6-5.8.4.2): see the 50% and 0% "
                    f"toe rows of the comparison (toe_mobilization).")
        elif (np_method == "settlement_compatibility"
              and S["z_fe_m"] is None):
            q0 = float(S["drag_from_top"][0])
            f_total = float(S["f_below"][0])
            delivered = float(S["drag_from_top"][-1])
            L = self.pile_length
            upper = self._compute_dragload(S["z"], S["dz"], S["fs"], L)
            if q0 > m * R + f_total:
                cause = (
                    f"with toe_mobilization = {m:g} the dead load "
                    f"({q0:.1f} kN) exceeds the mobilized resistance "
                    f"({m * R + f_total:.1f} kN: {m:g} x toe + whole "
                    f"shaft), so the toe must carry more than that fraction")
            else:
                cause = (
                    f"the toe resistance ({m * R:.1f} kN, ultimate unless "
                    f"Nt was given as a mobilized value) exceeds everything "
                    f"the pile can carry to its toe ({delivered:.1f} kN "
                    f"with the whole shaft in drag), so full base "
                    f"mobilization is impossible")
            warning_list.append(
                f"The load and resistance curves do not intersect: {cause}, "
                f"and force equilibrium does not locate the neutral plane. "
                f"It is placed by settlement compatibility at {z_np:.2f} m, "
                f"with a toe force of {toe_force:.1f} kN. Upper bound if the "
                f"soil settled relative to the pile down to the toe: neutral "
                f"plane at the toe, drag {upper:.1f} kN. To use force "
                f"equilibrium instead, give toe_mobilization or Nt for the "
                f"MOBILIZED toe resistance (UFC 3-220-20 §6-5.8.4.2: try "
                f"0%, 50%, 100%).")

    def _judgment(self, rows: list, used_name: Optional[str],
                  warning_list: list) -> Optional[dict]:
        """A ``judgment`` block when the applicable bases differ materially
        (the design bases plus the one used; the pile-toe bound and a
        compatibility 'check' row do not count), else None."""
        if used_name is None:
            return None
        considered = [r for r in rows if r["applies"]
                      and (r["role"] == "basis" or r["name"] == used_name)]
        if len(considered) < 2:
            return None
        nps = [r["neutral_plane_depth_m"] for r in considered]
        drags = [r["dragload_kN"] for r in considered]
        np_tol = max(NP_SPREAD_ABS_M, NP_SPREAD_REL_LENGTH * self.pile_length)
        drag_tol = max(DRAG_SPREAD_ABS_KN, DRAG_SPREAD_REL * max(drags))
        if (max(nps) - min(nps) <= np_tol
                and max(drags) - min(drags) <= drag_tol):
            return None
        req = self.neutral_plane_method
        if req == "auto":
            why = (
                "Default: UFC 3-220-20, the governing reference for DoD "
                "work. Its §6-7.4 procedure locates the plane by force "
                "equilibrium (side and base fully mobilized unless "
                "toe_mobilization is given; conservative for drag, not for "
                "settlement) and, when the curves do not cross, by "
                "settlement compatibility (Fig 6-19). The bases differ "
                "materially here: choose one (neutral_plane_method, "
                "toe_mobilization).")
        else:
            why = (f"Chosen by the caller (neutral_plane_method='{req}'). "
                   f"The bases differ materially here.")
        options = []
        for r in rows:
            if r["applies"]:
                result = (
                    f"NP {r['neutral_plane_depth_m']:.2f} m, drag "
                    f"{r['dragload_kN']:.0f} kN (Q_np "
                    f"{r['max_pile_load_kN']:.0f} kN); settlement at NP soil "
                    f"{r['soil_settlement_at_np_mm']:.1f} / pile "
                    f"{r['pile_settlement_at_np_mm']:.1f} mm")
                if r["role"] != "basis":
                    result += f" ({r['role']})"
            else:
                result = "not applicable: " + r["reason"]
            options.append({"name": r["name"], "source": r["source"],
                            "assumptions": r["assumptions"],
                            "applies": bool(r["applies"]), "result": result})
        warning_list.append(
            f"The neutral-plane bases differ materially for these inputs "
            f"(plane {min(nps):.2f}-{max(nps):.2f} m, drag "
            f"{min(drags):.0f}-{max(drags):.0f} kN): the basis is the "
            f"engineer's choice (see 'judgment' and the comparison). UFC "
            f"3-220-20 §6-5.8.4.2: if the conclusion depends on the toe "
            f"mobilization, refine with t-z / q-z curves.")
        return {
            "question": "Which neutral-plane basis governs for this pile?",
            "options": options,
            "used": used_name,
            "why": why,
        }

    # ── Private helper methods ────────────────────────────────────────────

    def _discretize(self):
        """Create depth nodes along the pile.

        Returns
        -------
        z_nodes : numpy.ndarray
            Depth array from 0 to pile_length.
        dz : float
            Sublayer thickness.
        """
        total_nodes = max(
            int(self.pile_length / 0.25),  # ~0.25 m spacing
            self.n_sublayers * len(self.soil.layers),
            50,
        )
        z_nodes = np.linspace(0, self.pile_length, total_nodes + 1)
        dz = z_nodes[1] - z_nodes[0]
        return z_nodes, dz

    def _compute_skin_friction(self, z_nodes: np.ndarray,
                                sigma_v: np.ndarray) -> np.ndarray:
        """Compute unit skin friction (kPa) at each depth.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        sigma_v : numpy.ndarray
            Effective vertical stress at each node.

        Returns
        -------
        numpy.ndarray
            Unit skin friction fs (kPa) at each node.
        """
        fs = np.zeros(len(z_nodes))
        for i, z in enumerate(z_nodes):
            try:
                layer = self.soil.layer_at_depth(z)
            except ValueError:
                continue

            if layer.soil_type == "cohesionless":
                beta = layer.beta if layer.beta is not None else 0.3
                fs[i] = beta * sigma_v[i]
            else:
                alpha = layer.alpha if layer.alpha is not None else 1.0
                fs[i] = alpha * layer.cu

        return fs

    def _compute_stress_change(self, z_nodes: np.ndarray) -> np.ndarray:
        """Compute stress change at each depth from fill and/or GW drawdown.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.

        Returns
        -------
        numpy.ndarray
            Stress change delta_sigma (kPa) at each node.
        """
        delta_sigma = np.zeros(len(z_nodes))

        # Fill placement: uniform 1-D stress increase
        if self.fill_thickness > 0:
            delta_sigma += self.fill_thickness * self.fill_unit_weight

        # Groundwater drawdown: increase in effective stress
        if self.gw_drawdown > 0:
            original_gwt = self.soil.gwt_depth
            new_gwt = original_gwt + self.gw_drawdown
            for i, z in enumerate(z_nodes):
                if z > original_gwt and z <= new_gwt:
                    # This zone was below GWT, now above: full drawdown effect
                    delta_sigma[i] += self.soil.gamma_w * (z - original_gwt)
                elif z > new_gwt:
                    # Below new GWT: constant effect = full drawdown
                    delta_sigma[i] += self.soil.gamma_w * self.gw_drawdown

        return delta_sigma

    def _compute_soil_settlement(self, z_nodes: np.ndarray, dz: float,
                                  sigma_v: np.ndarray,
                                  delta_sigma: np.ndarray) -> np.ndarray:
        """Compute cumulative soil settlement profile.

        Settlement is accumulated from the bottom of the settling zone
        upward: S(z) = sum of sublayer settlements from z downward to
        the bottom of the settling zone.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        dz : float
            Sublayer thickness.
        sigma_v : numpy.ndarray
            Initial effective stress at each node.
        delta_sigma : numpy.ndarray
            Stress change at each node.

        Returns
        -------
        numpy.ndarray
            Soil settlement (m) at each depth, measured from the pile-toe
            level (zero at the toe). Settlement at the surface is the total
            settlement of the soil along the pile; it decreases with depth.

        Notes
        -----
        Each sublayer [z_k, z_k+1] is evaluated at its MIDPOINT (layer,
        initial effective stress and stress change), so a sublayer is
        never attributed to the layer above a boundary and the top
        sublayer (where sigma'v0 = 0 at the surface node) still counts.
        The nodal ``sigma_v`` / ``delta_sigma`` arguments are kept for
        signature compatibility.
        """
        n = len(z_nodes)
        z_mid = 0.5 * (z_nodes[:-1] + z_nodes[1:])
        sv_mid = np.array([self.soil.effective_stress_at_depth(z)
                           for z in z_mid])
        ds_mid = self._compute_stress_change(z_mid)
        sublayer_settlement = np.zeros(n - 1)

        for k in range(n - 1):
            h = z_nodes[k + 1] - z_nodes[k]
            try:
                layer = self.soil.layer_at_depth(z_mid[k])
            except ValueError:
                continue

            if not layer.settling or ds_mid[k] <= 0 or sv_mid[k] <= 0:
                continue

            if layer.soil_type == "cohesive":
                # Clay settlement: Eq 6-53 using modified compression indices
                sigma_p = (layer.sigma_p
                           if layer.sigma_p is not None else sv_mid[k])
                sublayer_settlement[k] = _settlement_clay(
                    H=h, C_ec=layer.C_ec, C_er=layer.C_er,
                    sigma_v0=sv_mid[k], sigma_p=sigma_p,
                    delta_sigma=ds_mid[k],
                )
            else:
                # Coarse-grained elastic settlement: Eq 6-54
                if layer.E_s is not None and layer.E_s > 0:
                    sublayer_settlement[k] = _settlement_sand_elastic(
                        H=h, nu_s=layer.nu_s, E_s=layer.E_s,
                        delta_sigma=ds_mid[k],
                    )

        # Cumulate from the toe upward: settlement at depth z_i is the sum
        # of the sublayer settlements between z_i and the toe.
        cumulative = np.zeros(n)
        cumulative[:-1] = np.cumsum(sublayer_settlement[::-1])[::-1]

        return cumulative

    def _find_neutral_plane(self, z_nodes: np.ndarray, dz: float,
                             fs: np.ndarray,
                             toe_resistance: float):
        """Find neutral plane depth by force equilibrium.

        The neutral plane is where the cumulative load from the top
        (dead load + pile weight + dragload) equals the cumulative
        resistance from the bottom (toe + positive friction).

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        dz : float
            Sublayer thickness.
        fs : numpy.ndarray
            Unit skin friction at each node (kPa).
        toe_resistance : float
            Toe bearing capacity (kN).

        Returns
        -------
        z_np : float or None
            Neutral plane depth (m), or None when the two curves never
            cross (either the toe resistance exceeds everything the pile
            can deliver, or the dead load exceeds the total resistance).
            The caller decides what that means; it is never silently the
            pile toe.
        drag_from_top : numpy.ndarray
            Cumulative load from top at each node.
        resist_from_tip : numpy.ndarray
            Cumulative resistance from tip at each node.
        """
        n = len(z_nodes)
        perimeter = self.pile_perimeter
        pile_weight_per_m = self.pile_unit_weight * self.pile_area

        # Cumulative load from the top (dead load + pile weight + negative friction)
        drag_from_top = np.zeros(n)
        drag_from_top[0] = self.Q_dead
        for i in range(1, n):
            drag_from_top[i] = (drag_from_top[i - 1]
                                + pile_weight_per_m * dz
                                + fs[i] * perimeter * dz)

        # Cumulative resistance from the tip (toe + positive friction upward)
        resist_from_tip = np.zeros(n)
        resist_from_tip[-1] = toe_resistance
        for i in range(n - 2, -1, -1):
            resist_from_tip[i] = (resist_from_tip[i + 1]
                                  + fs[i + 1] * perimeter * dz)

        # Find crossing point: where drag_from_top = resist_from_tip
        diff = drag_from_top - resist_from_tip
        z_np = None  # no crossing unless found below

        for i in range(n - 1):
            if diff[i] <= 0 and diff[i + 1] > 0:
                # Linear interpolation for crossing
                frac = abs(diff[i]) / (abs(diff[i]) + abs(diff[i + 1]))
                z_np = z_nodes[i] + frac * dz
                break
        if z_np is None and diff[-1] == 0.0:
            z_np = float(z_nodes[-1])  # curves meet exactly at the toe

        return z_np, drag_from_top, resist_from_tip

    def _settlement_mismatch(self, z_nodes: np.ndarray, dz: float,
                             drag_from_top: np.ndarray,
                             resist_from_tip: np.ndarray,
                             toe_resistance: float,
                             soil_settlement: np.ndarray,
                             sigma_v_tip: float, delta_sigma_tip: float):
        """Soil minus pile settlement with the neutral plane at each node.

        For a trial neutral plane at node j the pile carries
        Q_np = drag_from_top[j]; below it positive friction sheds load and
        the toe takes what is left, never more than ``toe_resistance`` and
        never less than zero (when the full positive friction below would
        exceed Q_np, it is only partly mobilized). The pile settlement
        relative to the soil at the toe level is the toe penetration (the
        bearing-stratum settlement under the toe force, minus what the
        fill / drawdown alone causes there) plus the elastic compression
        of the pile between the trial NP and the toe. The soil settlement
        profile is already measured from the toe level (it accumulates
        from the bottom up), so the two compare directly (UFC 3-220-20
        Fig 6-19 and §6-5.8.4.3: equal settlement at the neutral plane).

        Returns
        -------
        g : numpy.ndarray
            soil settlement - pile settlement at each node (m); positive
            where the soil settles more than the pile (drag above).
        toe_force : numpy.ndarray
            Toe force (kN) carried with the neutral plane at each node.
        """
        n = len(z_nodes)
        AE = self.pile_area * self.pile_E
        f_below = resist_from_tip - toe_resistance   # positive friction j->L
        q_toe_raw = drag_from_top - f_below
        toe_force = np.clip(q_toe_raw, 0.0, toe_resistance)

        s_fill_only = self._compute_toe_settlement(
            sigma_v_tip, delta_sigma_tip, 0.0)
        g = np.zeros(n)
        for j in range(n):
            q_np = drag_from_top[j]
            if q_toe_raw[j] >= 0 or f_below[j] <= 0:
                k = 1.0
            else:
                k = q_np / f_below[j]
            q = q_np - k * (f_below[j] - f_below[j:])
            compression = (float(np.sum(0.5 * (q[:-1] + q[1:]))) * dz / AE
                           if AE > 0 and n - j > 1 else 0.0)
            penetration = max(0.0, self._compute_toe_settlement(
                sigma_v_tip, delta_sigma_tip, float(toe_force[j]))
                - s_fill_only)
            g[j] = soil_settlement[j] - (penetration + compression)
        return g, toe_force

    def _axial_load_with_mobilized_toe(self, z_nodes: np.ndarray,
                                       drag_from_top: np.ndarray,
                                       resist_from_tip: np.ndarray,
                                       toe_resistance: float,
                                       z_np: float) -> np.ndarray:
        """Axial load when the toe is not fully mobilized.

        Above the neutral plane the load is the load-from-top curve; below
        it positive friction sheds load down to the toe force the pile
        delivers. When the full positive friction below the NP exceeds the
        load at the NP, the friction is scaled so the toe force is zero.
        """
        f_below = resist_from_tip - toe_resistance
        q_np = float(np.interp(z_np, z_nodes, drag_from_top))
        f_np = float(np.interp(z_np, z_nodes, f_below))
        k = 1.0 if (q_np >= f_np or f_np <= 0) else q_np / f_np
        below = q_np - k * (f_np - f_below)
        q = np.where(z_nodes <= z_np, drag_from_top, below)
        return np.maximum(q, 0.0)

    def _compute_dragload(self, z_nodes: np.ndarray, dz: float,
                           fs: np.ndarray, z_np: float) -> float:
        """Compute negative skin friction (dragload) above neutral plane.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        dz : float
            Sublayer thickness.
        fs : numpy.ndarray
            Unit skin friction at each node (kPa).
        z_np : float
            Neutral plane depth (m).

        Returns
        -------
        float
            Dragload (kN), positive value.
        """
        dragload = 0.0
        for i in range(len(z_nodes)):
            if z_nodes[i] >= z_np:
                break
            # Partial sublayer at NP boundary
            z_top = z_nodes[i]
            z_bot = min(z_nodes[i] + dz, z_np)
            thickness = z_bot - z_top
            if thickness > 0:
                dragload += fs[i] * self.pile_perimeter * thickness
        return dragload

    def _compute_positive_resistance(self, z_nodes: np.ndarray, dz: float,
                                      fs: np.ndarray, z_np: float) -> float:
        """Compute positive skin friction below neutral plane.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        dz : float
            Sublayer thickness.
        fs : numpy.ndarray
            Unit skin friction at each node (kPa).
        z_np : float
            Neutral plane depth (m).

        Returns
        -------
        float
            Positive shaft resistance (kN).
        """
        positive = 0.0
        for i in range(len(z_nodes)):
            if z_nodes[i] < z_np:
                continue
            z_top = max(z_nodes[i], z_np)
            z_bot = z_nodes[i] + dz
            if z_bot > self.pile_length:
                z_bot = self.pile_length
            thickness = z_bot - z_top
            if thickness > 0:
                positive += fs[i] * self.pile_perimeter * thickness
        return positive

    def _compute_toe_resistance(self, sigma_v_tip: float) -> float:
        """Compute toe bearing resistance.

        Parameters
        ----------
        sigma_v_tip : float
            Effective vertical stress at pile tip (kPa).

        Returns
        -------
        float
            Toe resistance (kN).
        """
        tip_area = self.pile_area

        # Get tip layer
        try:
            tip_layer = self.soil.layer_at_depth(self.pile_length)
        except ValueError:
            return 0.0

        if self.Nt is not None:
            Nt = self.Nt
        elif tip_layer.phi > 0:
            Nt = _Nt_from_phi(tip_layer.phi)
        elif tip_layer.cu > 0:
            Nt = 9.0  # Nc for deep clay
        else:
            Nt = 0.0

        if tip_layer.soil_type == "cohesive":
            return Nt * tip_layer.cu * tip_area
        else:
            return Nt * sigma_v_tip * tip_area

    def _compute_axial_load_distribution(self, z_nodes: np.ndarray,
                                          dz: float, fs: np.ndarray,
                                          toe_resistance: float) -> np.ndarray:
        """Compute axial load distribution along the pile.

        Above NP: load increases (dead load + weight + dragload).
        Below NP: load decreases (positive friction removes load).
        This uses the force-from-top approach, which naturally produces
        the correct distribution.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        dz : float
            Sublayer thickness.
        fs : numpy.ndarray
            Unit skin friction (kPa).
        toe_resistance : float
            Toe resistance (kN).

        Returns
        -------
        numpy.ndarray
            Axial load Q(z) at each node (kN).
        """
        n = len(z_nodes)
        Q = np.zeros(n)
        pile_weight_per_m = self.pile_unit_weight * self.pile_area

        # We build the load distribution from equilibrium:
        # Above NP: friction adds load (negative skin friction)
        # Below NP: friction removes load (positive resistance)
        # The crossing is the neutral plane (max load).
        # Simple approach: use the from-top accumulation (drag_from_top)
        Q[0] = self.Q_dead
        for i in range(1, n):
            Q[i] = Q[i - 1] + pile_weight_per_m * dz + fs[i] * self.pile_perimeter * dz

        # However, below the NP the friction should be subtracting.
        # The _find_neutral_plane method already found the NP.
        # Re-do: build from both ends and use the minimum envelope.
        Q_from_top = np.zeros(n)
        Q_from_top[0] = self.Q_dead
        for i in range(1, n):
            Q_from_top[i] = (Q_from_top[i - 1]
                             + pile_weight_per_m * dz
                             + fs[i] * self.pile_perimeter * dz)

        Q_from_bot = np.zeros(n)
        Q_from_bot[-1] = toe_resistance
        for i in range(n - 2, -1, -1):
            Q_from_bot[i] = (Q_from_bot[i + 1]
                             + fs[i + 1] * self.pile_perimeter * dz)

        # The actual axial load is the minimum of both curves at each depth
        # (above NP: from_top governs; below NP: from_bot governs)
        Q = np.minimum(Q_from_top, Q_from_bot)
        return Q

    def _compute_elastic_shortening(self, z_nodes: np.ndarray, dz: float,
                                     axial_load: np.ndarray,
                                     z_np: float) -> float:
        """Compute elastic shortening of the pile above the neutral plane.

        Parameters
        ----------
        z_nodes : numpy.ndarray
            Depth array.
        dz : float
            Sublayer thickness.
        axial_load : numpy.ndarray
            Axial load distribution Q(z) (kN).
        z_np : float
            Neutral plane depth (m).

        Returns
        -------
        float
            Elastic shortening (m).
        """
        AE = self.pile_area * self.pile_E
        if AE <= 0:
            return 0.0

        shortening = 0.0
        for i in range(len(z_nodes) - 1):
            if z_nodes[i] >= z_np:
                break
            # Average load in this segment
            Q_avg = 0.5 * (axial_load[i] + axial_load[i + 1])
            seg_len = min(z_nodes[i + 1], z_np) - z_nodes[i]
            if seg_len > 0:
                shortening += Q_avg * seg_len / AE

        return shortening

    def _compute_toe_settlement(self, sigma_v_tip: float,
                                 delta_sigma_tip: float,
                                 toe_resistance: float) -> float:
        """Estimate settlement of the bearing stratum below the pile tip.

        Uses the equivalent footing concept (UFC Eqs 6-49/6-50) with
        2V:1H stress distribution (Eq 6-51) into the bearing stratum.
        The equivalent footing width B' = pile_diameter (single pile).
        Settlement is computed for sublayers within an influence zone
        of 3*B' below the pile tip.

        Parameters
        ----------
        sigma_v_tip : float
            Effective stress at pile tip (kPa).
        delta_sigma_tip : float
            Stress change at pile tip from fill/GW (kPa).
        toe_resistance : float
            Toe bearing resistance (kN) for stress distribution below tip.

        Returns
        -------
        float
            Estimated toe settlement (m).
        """
        # Equivalent footing dimensions (single pile: B' = L' = diameter)
        B_prime = self.pile_diameter
        L_prime = self.pile_diameter

        # Influence zone: 3*B' below pile tip
        influence_depth = 3.0 * B_prime
        n_sub = max(int(influence_depth / 0.25), 10)
        dz_sub = influence_depth / n_sub

        total_settle = 0.0
        for j in range(n_sub):
            z_below_tip = (j + 0.5) * dz_sub  # midpoint depth below tip
            z_abs = self.pile_length + z_below_tip

            # Get the layer at this depth
            try:
                layer = self.soil.layer_at_depth(z_abs)
            except ValueError:
                break

            # Stress change from pile load using 2V:1H (Eq 6-51)
            denom = (B_prime + z_below_tip) * (L_prime + z_below_tip)
            delta_sigma_pile = toe_resistance / denom if denom > 0 else 0.0

            # Additional stress from fill/GW at this depth
            delta_sigma_total = delta_sigma_pile + delta_sigma_tip

            if delta_sigma_total <= 0:
                continue

            # Effective stress at this depth
            sigma_v0 = self.soil.effective_stress_at_depth(z_abs)
            if sigma_v0 <= 0:
                continue

            if layer.soil_type == "cohesive" and layer.C_ec is not None:
                sigma_p = (layer.sigma_p
                           if layer.sigma_p is not None else sigma_v0)
                total_settle += _settlement_clay(
                    H=dz_sub, C_ec=layer.C_ec, C_er=layer.C_er,
                    sigma_v0=sigma_v0, sigma_p=sigma_p,
                    delta_sigma=delta_sigma_total,
                )
            elif (layer.soil_type == "cohesionless"
                  and layer.E_s is not None and layer.E_s > 0):
                total_settle += _settlement_sand_elastic(
                    H=dz_sub, nu_s=layer.nu_s, E_s=layer.E_s,
                    delta_sigma=delta_sigma_total,
                )

        return total_settle


# ── Module-level helper functions ─────────────────────────────────────────

def _deepest_positive_crossing(z_nodes: np.ndarray, g: np.ndarray) -> float:
    """Deepest depth at which ``g`` (soil minus pile settlement) is > 0.

    Interpolated between the deepest node with g > 0 and the node below
    it. Returns the toe depth when the soil settles more than the pile
    down to the toe, and 0.0 when it nowhere does.
    """
    pos = np.nonzero(g > 0)[0]
    if len(pos) == 0:
        return 0.0
    j = int(pos[-1])
    if j >= len(z_nodes) - 1:
        return float(z_nodes[-1])
    g0, g1 = float(g[j]), float(g[j + 1])
    frac = g0 / (g0 - g1) if g0 != g1 else 0.0
    return float(z_nodes[j] + frac * (z_nodes[j + 1] - z_nodes[j]))


def _compatible_plane(z_nodes: np.ndarray, g: np.ndarray,
                      z_fe: Optional[float]):
    """Settlement-compatible neutral plane with an elastic-plastic toe.

    ``g`` is soil minus pile settlement for a trial plane at each node,
    with the toe elastic (``_settlement_mismatch``). Below the
    force-equilibrium crossing ``z_fe`` (toe fully mobilized) equilibrium
    is impossible, so the plane is never deeper than ``z_fe``; if the soil
    still settles more than the elastic pile there, the toe yields at its
    full resistance and the plane IS ``z_fe`` (Fellenius 2021, the
    q-z iteration UFC 3-220-20 §6-5.8.4.3 describes).

    Returns
    -------
    (z, toe_yields) : (float, bool)
    """
    if z_fe is None:
        return _deepest_positive_crossing(z_nodes, g), False
    g_fe = float(np.interp(z_fe, z_nodes, g))
    if g_fe > 0:
        return float(z_fe), True
    above = z_nodes < z_fe
    zz = np.append(z_nodes[above], z_fe)
    gg = np.append(g[above], g_fe)
    return _deepest_positive_crossing(zz, gg), False


def _settlement_clay(H: float, C_ec: float, C_er: float,
                     sigma_v0: float, sigma_p: float,
                     delta_sigma: float) -> float:
    """Settlement of clay using modified compression indices (UFC Eq 6-53).

    Uses modified compression indices C_ec and C_er (= Cc/(1+e0) and
    Cr/(1+e0) respectively). Three cases:

    1. NC (sigma_v0 >= sigma_p): Sc = C_ec * H * log10(sigma_final/sigma_v0)
    2. OC stays OC (sigma_final <= sigma_p): Sc = C_er * H * log10(...)
    3. OC → NC: recompression to sigma_p, then virgin compression beyond

    NC tolerance band (DD-1): the soil is treated as normally consolidated
    whenever |sigma_p - sigma_v0| / sigma_v0 < 0.05, i.e. a preconsolidation
    pressure within 5% of the in-situ stress counts as NC and the full
    increment uses the virgin index C_ec. This avoids spurious
    recompression-only behavior from small sigma_p measurement noise, but
    creates a small settlement step versus a soil just outside the band
    (slightly OC, case 3). The same convention is used by the settlement
    module's consolidation routine, so the two modules agree.

    Parameters
    ----------
    H : float
        Sublayer thickness (m).
    C_ec : float
        Modified compression index = Cc/(1+e0).
    C_er : float
        Modified recompression index = Cr/(1+e0).
    sigma_v0 : float
        Initial effective vertical stress (kPa).
    sigma_p : float
        Preconsolidation pressure (kPa).
    delta_sigma : float
        Stress change (kPa).

    Returns
    -------
    float
        Settlement (m).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-53.
    """
    if delta_sigma <= 0 or sigma_v0 <= 0:
        return 0.0

    sigma_final = sigma_v0 + delta_sigma

    # Treat as NC if sigma_v0 is within 5% of sigma_p
    is_NC = abs(sigma_p - sigma_v0) / sigma_v0 < 0.05

    if is_NC:
        return C_ec * H * math.log10(sigma_final / sigma_v0)
    elif sigma_final <= sigma_p:
        return C_er * H * math.log10(sigma_final / sigma_v0)
    else:
        Sc_oc = C_er * H * math.log10(sigma_p / sigma_v0)
        Sc_nc = C_ec * H * math.log10(sigma_final / sigma_p)
        return Sc_oc + Sc_nc


def _settlement_sand_elastic(H: float, nu_s: float, E_s: float,
                             delta_sigma: float) -> float:
    """Elastic settlement of coarse-grained soil (UFC Eq 6-54).

    .. math::
        delta_s = H * (1+nu_s)*(1-2*nu_s) / ((1-nu_s)*E_s) * delta_sigma

    Parameters
    ----------
    H : float
        Sublayer thickness (m).
    nu_s : float
        Poisson's ratio (dimensionless).
    E_s : float
        Young's modulus of soil (kPa).
    delta_sigma : float
        Stress change (kPa).

    Returns
    -------
    float
        Settlement (m).

    References
    ----------
    UFC 3-220-20, 16 Jan 2025, Chapter 6, Equation 6-54.
    """
    if E_s <= 0 or delta_sigma <= 0:
        return 0.0
    return H * (1.0 + nu_s) * (1.0 - 2.0 * nu_s) / ((1.0 - nu_s) * E_s) * delta_sigma


def _consolidation_settlement(H: float, e0: float, Cc: float, Cr: float,
                               sigma_v0: float, sigma_p: float,
                               delta_sigma: float) -> float:
    """Legacy wrapper: convert traditional Cc/Cr/e0 to modified indices.

    Kept for backward compatibility with tests using the traditional API.
    Delegates to _settlement_clay().
    """
    C_ec = Cc / (1.0 + e0) if e0 > 0 else 0.0
    C_er = Cr / (1.0 + e0) if e0 > 0 else 0.0
    return _settlement_clay(H, C_ec, C_er, sigma_v0, sigma_p, delta_sigma)


def _Nt_from_phi(phi_deg: float) -> float:
    """Estimate toe bearing capacity factor Nt from friction angle.

    Parameters
    ----------
    phi_deg : float
        Friction angle (degrees).

    Returns
    -------
    float
        Bearing capacity factor Nt.

    References
    ----------
    Fellenius (1991), FHWA GEC-12 Table 7-9.
    """
    if phi_deg <= 0:
        return 3.0
    elif phi_deg <= 20:
        return 3.0 + (phi_deg / 20) * 7
    elif phi_deg <= 28:
        return 10 + (phi_deg - 20) * 2.5
    elif phi_deg <= 33:
        return 30 + (phi_deg - 28) * 8
    elif phi_deg <= 38:
        return 70 + (phi_deg - 33) * 16
    else:
        return 150 + (phi_deg - 38) * 20
