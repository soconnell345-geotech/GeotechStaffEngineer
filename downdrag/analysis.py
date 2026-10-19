"""
Downdrag (negative skin friction) analysis using the neutral plane method.

Every method the references give for locating the neutral plane is
evaluated on the same inputs and returned side by side (the
``neutral_plane_comparison`` rows): the neutral plane at 100 / 50 / 0 % toe
mobilization (UFC 3-220-20 §6-7.4 and §6-5.8.4.2; FHWA GEC-12 §7.3.6.1),
the mobilized toe resistance made compatible with the toe penetration (UFC
§6-5.8.4.3, Fig 6-19), the neutral plane at the bearing layer for an
end-bearing pile (UFC §6-7.4 step 5), at the pile toe (GEC-12 §7.3.5.7) and
Endo et al. (1969) (CGPR #56 §3.2.1). One of them gives the headline
numbers: by default UFC 3-220-20's §6-7.4 procedure. See DESIGN.md.

Supports fill placement and groundwater drawdown as settlement triggers.

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).

References
----------
- Fellenius, B.H. (2006). "Results of static loading tests on driven piles."
- Fellenius, B.H. (2004). "Unified design of piled foundations with emphasis
  on settlement analysis." ASCE GSP 125.
- AASHTO LRFD Bridge Design Specifications, Section 10.7.3.7.
- UFC 3-220-20, 16 Jan 2025, Chapter 6, Eqs 6-49 through 6-54, 6-80;
  §6-5.8.4.3 (how deep below the toe the toe load is counted).
- FHWA GEC-12 (FHWA-NHI-16-009) Vol I, §7.3.5.3 step 1c (the same, by the
  effective stress).
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

#: Where each method is stated (printed page numbers). UFC 3-220-20 =
#: 16 Jan 2025 (DM 7.2); GEC-12 = FHWA-NHI-16-009 Vol I; CGPR #56 =
#: Greenfield & Filz (2009), sections only (not public-domain, not in the
#: repo).
_SOURCE = {
    "fe_full": ("UFC 3-220-20 §6-7.4 steps 3-5, Fig 6-26 (pp. 476-478); "
                "FHWA GEC-12 §7.3.6.1 step 5, Fig 7-60 point A (p. 346)"),
    "fe_partial": ("UFC 3-220-20 §6-5.8.4.2 (p. 447), App. B-5.5 Table "
                   "B-28 (p. 640); FHWA GEC-12 §7.3.6.1 steps 2-5, Figs "
                   "7-59 and 7-60 (pp. 342-346); Siegel et al. (2013)"),
    "sc": ("UFC 3-220-20 §6-5.8.4.3, Fig 6-19 (pp. 449-450); "
           "Fellenius (2021)"),
    "blt": "UFC 3-220-20 §6-7.4 step 5 (p. 477); §6-5.7 (p. 438)",
    "toe": ("FHWA GEC-12 §7.3.5.7 (p. 336; Goudreault and Fellenius "
            "1994); CGPR #56 Table 3.1 'end bearing'"),
    "endo": "CGPR #56 §3.2.1, Table 3.1 (Endo et al. 1969)",
}

#: The headline method when ``neutral_plane_method='auto'``.
UFC_PROCEDURE = "UFC 3-220-20 §6-7.4 procedure"

#: Accepted values of ``DowndragAnalysis.influence_depth_criterion``: how
#: deep below the toe the toe load is counted (DESIGN.md "Settlement below
#: the toe").
INFLUENCE_DEPTH_CRITERIA = ("bearing_stress", "effective_stress")

#: Both references stop counting the toe load where its stress increase
#: falls below 10 % of a stress: UFC 3-220-20 §6-5.8.4.3 (p. 449) "of the
#: bearing stress of the equivalent footing", FHWA GEC-12 §7.3.5.3 step 1c
#: (p. 324) "of existing vertical effective stress at that depth".
INFLUENCE_STRESS_FRACTION = 0.10

#: Each criterion in the reference's words, and where it is stated.
_INFLUENCE = {
    "bearing_stress": ("10% of the bearing stress of the equivalent footing",
                       "UFC 3-220-20 §6-5.8.4.3, p. 449"),
    "effective_stress": ("10% of the existing vertical effective stress",
                         "FHWA GEC-12 §7.3.5.3 step 1c, p. 324"),
}

#: Thickest sublayer below the toe (m), as along the shaft.
_SUBLAYER = 0.25


def _fe_name(mobilization: float) -> str:
    """Row name of the neutral plane at a toe mobilization (GEC-12 Vol III
    Figs D-32 / D-33: "Neutral plane at 50% / 100% toe mobilization")."""
    return f"Neutral plane at {round(mobilization * 100):d}% toe mobilization"


_SC_NAME = "Mobilized toe resistance compatible with the toe penetration"
_BLT_NAME = "Neutral plane at the bearing layer (end-bearing pile)"
_TOE_NAME = "Neutral plane at the pile toe"


def _endo_name(cond: Optional[str]) -> str:
    if cond is None:
        return "Endo et al. (1969)"
    return (f"Endo et al. (1969), '{cond}' "
            f"({ENDO_NEUTRAL_PLANE_RATIOS[cond]:g} L)")


@dataclass
class DowndragAnalysis:
    """Downdrag analysis using the neutral plane method, every method side
    by side.

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
        Pile material unit weight (kN/m^3), times ``pile_area`` for the
        pile's own weight per metre. Default 78.5 (steel, as ``pile_E``;
        concrete is about 24). For an H or pipe pile give ``pile_area`` as
        the steel area. Used only when ``include_pile_weight`` is True.
    include_pile_weight : bool
        Whether the load curve carries the pile's own weight above each
        depth. Default False: left out, as UFC 3-220-20 Fig 6-26 and FHWA
        GEC-12 §7.3.6.1 (and both their worked examples) do (owner,
        2026-10-15). True adds it to Q_np and the drag force.
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
        Which method gives the headline numbers (DESIGN.md "Neutral-plane
        methods" gives the sources). Every method is evaluated and
        returned in ``neutral_plane_comparison`` whatever is chosen.

        - ``"auto"`` (default): UFC 3-220-20's §6-7.4 procedure. The
          neutral plane is where the load and resistance curves intersect
          (steps 3-5), with the toe at ``toe_mobilization`` of its nominal
          resistance; when they do not intersect, where the mobilized toe
          resistance is compatible with the toe penetration (§6-5.8.4.3,
          Fig 6-19); no neutral plane when the dead load exceeds the
          nominal resistance (step 5).
        - ``"force_equilibrium"``: where the load and resistance curves
          intersect only (raises when they do not).
        - ``"settlement_compatibility"``: where the pile and the soil
          settle equally, the toe force compatible with the toe
          penetration (UFC §6-5.8.4.3, Fig 6-19; Fellenius 2021).
        - ``"bearing_layer_top"``: at the top of the bearing layer (the
          base of the settling soil), for an end-bearing pile (UFC §6-7.4
          step 5).
        - ``"pile_toe"``: at the toe, the whole shaft in negative skin
          friction (the upper bound on drag; GEC-12 §7.3.5.7 uses it for
          group settlement).
        - ``"endo"``: at a fraction of the embedment (CGPR #56 Table 3.1);
          needs ``endo_bearing_condition``.
    toe_mobilization : float
        Fraction (0-1) of the nominal toe resistance mobilized where the
        load and resistance curves are intersected. Default 1.0 (UFC
        §6-7.4 step 3: "a conservative approach for evaluating the drag
        force"). 0.5 and 0.0 are the other two cases of UFC §6-5.8.4.2 /
        GEC-12 §7.3.6.1 (always reported in the comparison).
    endo_bearing_condition : str, optional
        CGPR #56 Table 3.1 condition for Endo et al. (1969):
        ``"floating"``, ``"stiff_flexible"`` or ``"end_bearing"``.
    skin_friction_stress : str
        Effective stress for beta-method unit skin friction and the
        cohesionless toe: ``"final"`` (default: sigma'v0 + the fill /
        drawdown stress change) or ``"initial"`` (sigma'v0). UFC 3-220-20
        §6-7.4 step 2: "Nominal side and base resistances should be
        evaluated considering the vertical effective stress profile that
        includes stress change(s) producing settlement." FHWA GEC-12
        §7.3.6.1 states no rule in those words (DESIGN.md quotes what it
        says).
    influence_depth_criterion : str
        How deep below the toe the toe load is counted in the settlement
        (the toe force spread 2V:1H from an equivalent footing as wide as
        the pile, UFC 3-220-20 Eq 6-52): down to where its stress increase
        falls below

        - ``"bearing_stress"`` (default): "10% of the bearing stress of
          the equivalent footing" (UFC 3-220-20 §6-5.8.4.3, p. 449); for a
          single pile (sqrt(10) - 1) = 2.16 pile widths below the toe;
        - ``"effective_stress"``: "10% of existing vertical effective
          stress at that depth" (FHWA GEC-12 §7.3.5.3 step 1c, p. 324);
          deeper for a loaded toe in soft soil.

        Either way the fill / drawdown stress increase, which does not
        decay with depth, compresses the settling layers below the toe down
        to the bottom of the soil profile. The result's ``below_toe`` says
        how deep each was counted, why it stopped, and what the other
        criterion gives.

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
    pile_unit_weight: float = 78.5
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
    skin_friction_stress: str = "final"
    include_pile_weight: bool = False
    influence_depth_criterion: str = "bearing_stress"

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
        if not isinstance(self.include_pile_weight, (bool, np.bool_)):
            raise ValueError(
                f"include_pile_weight must be True or False, got "
                f"{self.include_pile_weight!r}")
        self.include_pile_weight = bool(self.include_pile_weight)
        if self.influence_depth_criterion not in INFLUENCE_DEPTH_CRITERIA:
            raise ValueError(
                f"influence_depth_criterion must be one of "
                f"{list(INFLUENCE_DEPTH_CRITERIA)} ('bearing_stress': UFC "
                f"3-220-20 §6-5.8.4.3; 'effective_stress': FHWA GEC-12 "
                f"§7.3.5.3), got '{self.influence_depth_criterion}'")

    def _weight_per_m(self) -> float:
        """The pile's own weight per metre in the load curve (kN/m): zero
        unless ``include_pile_weight``."""
        if not self.include_pile_weight:
            return 0.0
        return self.pile_unit_weight * self.pile_area

    def _pile_weight_basis(self) -> str:
        """One plain sentence: whether the load curve carries the pile's own
        weight, and what the references do."""
        if not self.include_pile_weight:
            return ("The pile's own weight is left out of the load curve, as "
                    "UFC 3-220-20 Fig 6-26 and FHWA GEC-12 §7.3.6.1 do "
                    "(include_pile_weight=False).")
        return (f"The pile's own weight is included in the load curve "
                f"(include_pile_weight=True: {self.pile_unit_weight:g} kN/m³ "
                f"x {self.pile_area:.4g} m² = {self._weight_per_m():.3g} "
                f"kN/m), so Q_np and the drag force carry the weight of the "
                f"pile above the neutral plane; UFC 3-220-20 Fig 6-26 and "
                f"FHWA GEC-12 §7.3.6.1 leave it out.")

    def compute(self) -> DowndragResult:
        """Run the downdrag analysis.

        Every method in ``neutral_plane_comparison`` is evaluated on the
        same inputs; the one ``neutral_plane_method`` names (default
        ``"auto"``: UFC 3-220-20's §6-7.4 procedure) gives the headline
        numbers, and ``methods_range`` states the spread of the others.

        Returns
        -------
        DowndragResult
            Analysis results including neutral plane depth, dragload,
            settlement, and limit state checks.
        """
        S = self._state()
        z_nodes, dz, fs = S["z"], S["dz"], S["fs"]
        R = S["R_toe"]
        m = self.toe_mobilization
        req = self.neutral_plane_method
        rows = self._comparison_rows(S)
        warning_list: List[str] = []

        if S["overloaded"]:
            # The dead load alone exceeds the nominal geotechnical
            # resistance: UFC 3-220-20 §6-7.4 step 5 limiting case, "there
            # is no neutral plane". The pile plunges, every part of the
            # shaft moves down relative to the soil, so no drag develops.
            np_method = "none"
            z_np = 0.0
            toe_force = R
            capacity = float(S["resist_full"][0])
            np_basis = (
                "No neutral plane: the dead load exceeds the pile's nominal "
                "geotechnical resistance, the limiting case of UFC 3-220-20 "
                "§6-7.4 step 5. Reported at the pile head with zero drag.")
            warning_list.append(
                f"NO NEUTRAL PLANE: Q_dead = {self.Q_dead:.1f} kN exceeds "
                f"the total geotechnical resistance (toe + full shaft) = "
                f"{capacity:.1f} kN. The pile is overloaded; a drag force "
                f"is not meaningful. Increase the pile length or size.")
            used = None
        else:
            used = self._headline_row(rows)
            if not used["applies"]:
                raise ValueError(
                    f"neutral_plane_method='{req}' does not locate the "
                    f"neutral plane for these inputs ({used['name']}): "
                    f"{used['reason']} Use neutral_plane_method='auto' or "
                    f"another method.")
            used["used"] = True
            np_method = used["method"]
            z_np = float(used["neutral_plane_depth_m"])
            toe_force = float(used["toe_force_kN"])
            np_basis = self._basis_text(S, used)
            self._basis_warnings(S, np_method, z_np, toe_force, warning_list)
        if self.skin_friction_stress == "final":
            np_basis += (
                " Unit skin friction (beta layers) and the toe in sand use "
                "the final effective stress, sigma'v0 plus the fill / "
                "drawdown change (UFC 3-220-20 §6-7.4 step 2).")
        else:
            np_basis += (
                " Unit skin friction (beta layers) and the toe in sand use "
                "the initial effective stress sigma'v0 "
                "(skin_friction_stress='initial'); UFC 3-220-20 §6-7.4 "
                "step 2 uses the final stress.")
        np_basis += " " + self._pile_weight_basis()

        # Drag load and positive friction: the same integral of the same
        # unit skin friction profile as the load and resistance curves
        # (GM-01), so at an intersection Q_np = toe + positive friction.
        F_L = float(S["F"][-1])
        dragload = S["shaft_to"](z_np)
        pile_weight_to_np = self._weight_per_m() * z_np
        max_pile_load = self.Q_dead + dragload + pile_weight_to_np

        positive_skin = F_L - dragload
        total_resistance = positive_skin + R

        # Compute axial load distribution along pile
        if np_method == "force_equilibrium":
            axial_load = self._compute_axial_load_distribution(
                z_nodes, dz, fs, m * R)
        elif np_method == "none":
            axial_load = self._compute_axial_load_distribution(
                z_nodes, dz, fs, R)
        else:
            axial_load = self._axial_load_with_mobilized_toe(S, z_np)

        toe_load_s = self._toe_load_compression(toe_force)
        toe_settle = S["s_fill_only"] + toe_load_s
        below_toe = self._below_toe_info(toe_force, S["s_fill_only"],
                                         toe_load_s)
        unrated = self._unrated_below_toe(toe_force)
        if unrated:
            # GM-51: with no compressibility given, the soil below the toe
            # adds nothing, and the pile settlement reads as if it were rigid
            spans = "; ".join(
                f"{a:.2f}-{b:.2f} m ({kind})" if b is not None
                else f"below {a:.2f} m ({kind})" for a, b, kind in unrated)
            text = (
                "The soil below the pile toe has no compressibility in this "
                f"model ({spans}), so its settlement counts as zero (here "
                f"{toe_settle * 1000:.1f} mm below the toe in all).")
            z_q = self._toe_load_depth(toe_force) if toe_force > 0 else 0.0
            if z_q > 0:
                what, src = _INFLUENCE[self.influence_depth_criterion]
                text += (
                    f" The toe load counts down to "
                    f"{self.pile_length + z_q:.2f} m, where its stress "
                    f"increase falls below {what} ({src}): extend the "
                    f"profile at least that deep.")
            if self.fill_thickness > 0 or self.gw_drawdown > 0:
                text += (
                    " The fill / drawdown stress increase does not decay "
                    "with depth: give the profile down to the base of the "
                    "compressible soil.")
            warning_list.append(
                text + " Give Cc and e0 (or C_ec) for clay, or E_s for "
                "sand, to include it, and mark a layer settling if the "
                "fill / drawdown compresses it (UFC 3-220-20 Eqs 6-49 to "
                "6-54, the equivalent footing at the toe).")
        if used is None:
            elastic_short = 0.0
            pile_settlement = toe_settle
            soil_settle_at_np = (float(S["soil_settlement"][0])
                                 + S["s_fill_only"])
        else:
            # UFC 3-220-20 Fig 6-19 / §6-5.8 item 6: the pile head settles
            # as much as the neutral plane plus the elastic compression of
            # the pile above it.
            elastic_short = used["elastic_compression_above_np_mm"] / 1000.0
            pile_settlement = used["pile_settlement_mm"] / 1000.0
            soil_settle_at_np = used["soil_settlement_at_np_mm"] / 1000.0
        settlement = max(pile_settlement, soil_settle_at_np)

        # Limit state checks

        # Structural: UFC Eq 6-80 LRFD factored demand
        #   1.25*Q_dead + 1.10*(Q_np - Q_dead) <= P_r
        # where Q_np = max_pile_load (total load at neutral plane)
        structural_ok = None
        structural_demand = None
        if self.structural_capacity is not None:
            # dragload (+ pile weight when include_pile_weight)
            drag_force = max_pile_load - self.Q_dead
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
            pile_length=self.pile_length,
            pile_diameter=self.pile_diameter,
            neutral_plane_method=np_method,
            neutral_plane_basis=np_basis,
            toe_force_mobilized=toe_force,
            warnings=warning_list,
            neutral_plane_requested=req,
            toe_mobilization=m,
            skin_friction_stress=self.skin_friction_stress,
            include_pile_weight=self.include_pile_weight,
            neutral_plane_comparison=rows,
            methods_range=_methods_range(rows),
            headline_method=(used["name"] if used is not None
                             else "no neutral plane"),
            influence_depth_criterion=self.influence_depth_criterion,
            below_toe=below_toe,
        )

    # ── Neutral-plane methods ─────────────────────────────────────────────

    def _headline_row(self, rows: list) -> dict:
        """The comparison row that gives the headline numbers."""
        req = self.neutral_plane_method
        m = self.toe_mobilization

        def fe_row():
            return next(r for r in rows
                        if r["method"] == "force_equilibrium"
                        and abs(r["toe_mobilization"] - m) < 1e-9)

        if req == "auto":
            row = fe_row()
            if row["applies"]:
                return row
            return next(r for r in rows
                        if r["method"] == "settlement_compatibility")
        if req == "force_equilibrium":
            return fe_row()
        return next(r for r in rows if r["method"] == req)

    def _state(self) -> dict:
        """Everything the methods share: the depth grid, unit skin
        friction and its running integral, the soil settlement profile,
        the load and resistance curves (toe fully mobilized), the
        settlement mismatch, and the planes the curves' intersection and
        the toe-penetration compatibility give."""
        z_nodes, dz = self._discretize()
        sigma_v0 = np.array([
            self.soil.effective_stress_at_depth(z) for z in z_nodes
        ])
        # Stress change at each node (from fill and/or GW drawdown)
        delta_sigma = self._compute_stress_change(z_nodes)
        sigma_fs = (sigma_v0 + delta_sigma
                    if self.skin_friction_stress == "final" else sigma_v0)
        fs = self._compute_skin_friction(z_nodes, sigma_fs)
        # Shaft resistance from the head to each node (kN): one integral of
        # the reported unit skin friction profile for every use (GM-01).
        F = _shaft_cumulative(z_nodes, fs, self.pile_perimeter)

        def shaft_to(z: float) -> float:
            return _shaft_to(z, z_nodes, fs, F, self.pile_perimeter)

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

        # Pile and soil settle equally at the NP (UFC 3-220-20 Fig 6-19),
        # evaluated at every node.
        g, toe_force_at = self._settlement_mismatch(
            z_nodes, dz, drag_from_top, resist_full, R, soil_settlement)
        z_sc, sc_toe_yields = _compatible_plane(z_nodes, g, z_fe_full)
        return {
            "z": z_nodes, "dz": dz, "fs": fs, "F": F, "shaft_to": shaft_to,
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
            "s_fill_only": self._fill_settlement_below_toe(),
        }

    def _load_at(self, S: dict, z: float) -> float:
        """Load curve at depth ``z``: Q_dead (+ pile weight when
        ``include_pile_weight``) + negative skin friction from the head
        (kN)."""
        return self.Q_dead + self._weight_per_m() * z + S["shaft_to"](z)

    def _pile_settlement_at(self, S: dict, z: float,
                            toe_force: float) -> float:
        """Settlement of the pile at depth ``z`` with the neutral plane
        there: settlement of the soil under the toe (bearing-stratum
        compression under the toe force and the fill / drawdown) + elastic
        compression of the pile between ``z`` and the toe (UFC 3-220-20
        Fig 6-19)."""
        zn, fb = S["z"], S["f_below"]
        q_np = self._load_at(S, z)
        f_np = float(S["F"][-1]) - S["shaft_to"](z)
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
        under_toe = S["s_fill_only"] + self._toe_load_compression(toe_force)
        return under_toe + compression

    def _compression_above(self, S: dict, z: float) -> float:
        """Elastic compression of the pile from the head to depth ``z``
        under the load curve (UFC 3-220-20 Eq 6-45, segment by segment)."""
        AE = self.pile_area * self.pile_E
        zn = S["z"]
        above = zn < z
        zz = np.append(zn[above], z)
        if AE <= 0 or len(zz) < 2:
            return 0.0
        qq = np.append(S["drag_from_top"][above], self._load_at(S, z))
        return float(np.trapezoid(qq, zz)) / AE

    def _plane_numbers(self, S: dict, z: float, toe_force: float) -> dict:
        """Drag load, drag force, load at the plane, the soil and pile
        settlement at the plane and the pile head settlement for a neutral
        plane at ``z`` carrying ``toe_force``. Settlements are measured from
        the original ground (the soil at the toe level settles by the
        bearing-stratum compression under the fill / drawdown alone)."""
        drag = S["shaft_to"](z)
        q_np = self._load_at(S, z)
        s_soil = (float(np.interp(z, S["z"], S["soil_settlement"]))
                  + S["s_fill_only"])
        s_pile = self._pile_settlement_at(S, z, toe_force)
        R = S["R_toe"]
        if R > 0 and toe_force >= R * (1.0 - 1e-9) and s_soil > s_pile:
            # Toe at its full resistance: it yields and the pile goes down
            # with the soil (elastic-plastic toe).
            s_pile = s_soil
        above = self._compression_above(S, z)
        return {
            "neutral_plane_depth_m": float(z),
            "max_pile_load_kN": float(q_np),
            "dragload_kN": float(drag),
            "drag_force_kN": float(q_np - self.Q_dead),
            "toe_force_kN": float(toe_force),
            "soil_settlement_at_np_mm": s_soil * 1000.0,
            "pile_settlement_at_np_mm": s_pile * 1000.0,
            "elastic_compression_above_np_mm": above * 1000.0,
            "pile_settlement_mm": (max(s_soil, s_pile) + above) * 1000.0,
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
        """Every method on the same inputs, one row each.

        Each row: name (as the cited reference words it), method (the
        ``neutral_plane_method`` value), toe_mobilization, source,
        assumptions, applies, used (True on the row that gives the
        headline), reason (why it does not apply, or a note), and when it
        applies: the neutral plane depth, the maximum load Q_np, the drag
        load (negative skin friction above the plane), the drag force
        Q_np - Q_dead (UFC Eq 6-80), the toe force, the soil and pile
        settlement at the plane, the elastic compression above it and the
        pile head settlement.
        """
        L = self.pile_length
        R = S["R_toe"]
        F_L = float(S["F"][-1])
        rows = []

        def row(name, method, source, assumptions, mob=None):
            r = {"name": name, "method": method, "toe_mobilization": mob,
                 "source": source, "assumptions": assumptions,
                 "applies": False, "used": False, "reason": "",
                 "neutral_plane_depth_m": None, "max_pile_load_kN": None,
                 "dragload_kN": None, "drag_force_kN": None,
                 "toe_force_kN": None, "soil_settlement_at_np_mm": None,
                 "pile_settlement_at_np_mm": None,
                 "elastic_compression_above_np_mm": None,
                 "pile_settlement_mm": None}
            rows.append(r)
            return r

        def fill(r, z, toe):
            r.update(self._plane_numbers(S, z, toe))
            r["applies"] = True

        def imposed(r, z, what):
            """A plane set by a statement, not by equilibrium: check the
            toe could carry what equilibrium then asks of it."""
            q_np = self._load_at(S, z)
            f_np = F_L - S["shaft_to"](z)
            toe_raw = q_np - f_np
            if toe_raw > R * (1.0 + 1e-6) + 1e-9:
                r["reason"] = (
                    f"with the neutral plane {what} ({z:.2f} m) the toe "
                    f"would have to carry {toe_raw:.0f} kN, more than its "
                    f"resistance ({R:.0f} kN): the load and resistance "
                    f"curves put the plane higher.")
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
                r = row(_fe_name(mob), "force_equilibrium",
                        _SOURCE["fe_full"],
                        "Intersection of the load and resistance curves with "
                        "side and base resistance fully mobilized: UFC "
                        "§6-7.4 step 3 calls it 'a conservative approach for "
                        "evaluating the drag force'; not conservative for "
                        "settlement (§6-5.8.4.2).", mob)
            else:
                r = row(_fe_name(mob), "force_equilibrium",
                        _SOURCE["fe_partial"],
                        f"Intersection of the load and resistance curves, "
                        f"side fully mobilized, toe at {pct}% of its nominal "
                        f"resistance: UFC §6-5.8.4.2 and GEC-12 §7.3.6.1 "
                        f"analyse 0%, 50% and 100% (Siegel et al. 2013).",
                        mob)
            if S["overloaded"]:
                r["reason"] = none_reason
                continue
            z = (S["z_fe_full"] if mob == 1.0 else self._find_neutral_plane(
                S["z"], S["dz"], S["fs"], mob * R)[0])
            if z is None:
                q0 = float(S["drag_from_top"][0])
                if q0 > mob * R + f_total:
                    r["reason"] = (
                        f"the curves do not intersect: Q_dead ({q0:.0f} kN) "
                        f"exceeds {pct}% of the toe resistance + the whole "
                        f"shaft ({mob * R + f_total:.0f} kN).")
                else:
                    r["reason"] = (
                        f"the curves do not intersect: {pct}% of the toe "
                        f"resistance ({mob * R:.0f} kN) exceeds all the pile "
                        f"can deliver to its toe "
                        f"({float(S['drag_from_top'][-1]):.0f} kN with the "
                        f"whole shaft in negative skin friction).")
                continue
            fill(r, z, mob * R)

        # Pile and soil settle equally; toe force compatible with the toe
        # penetration (UFC §6-5.8.4.3, Fig 6-19; Fellenius 2021)
        settles = S["any_soil_settles"]
        r = row(_SC_NAME, "settlement_compatibility", _SOURCE["sc"],
                "Pile and soil settle equally at the neutral plane (Fig "
                "6-19). The toe force follows the toe penetration: "
                "bearing-stratum compression under the toe force (UFC Eqs "
                "6-51 to 6-54) standing in for a q-z curve, capped at the "
                "toe resistance; side fully mobilized.")
        if S["overloaded"]:
            r["reason"] = none_reason
        else:
            z = S["z_sc"]
            toe = (R if S["sc_toe_yields"]
                   else float(np.interp(z, S["z"], S["toe_force_at"])))
            fill(r, z, toe)
            if not settles:
                r["reason"] = (
                    "no soil settles here (no settling layer loaded by fill "
                    "or drawdown), so this gives no drag; UFC §6-5.7 (FHWA "
                    "2016): the pile-soil stiffness contrast alone gives "
                    "some drag force.")

        # End-bearing pile: plane near the top of the bearing layer
        r = row(_BLT_NAME, "bearing_layer_top", _SOURCE["blt"],
                "UFC §6-7.4 step 5: 'the neutral plane will be located near "
                "the interface of the column and the bearing layer for an "
                "end-bearing column bearing in a stratum that is much "
                "stiffer than the compressible soil'. Plane at the base of "
                "the settling soil; set by that statement, not computed.")
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

        # Neutral plane at the pile toe: the upper bound on drag
        r = row(_TOE_NAME, "pile_toe", _SOURCE["toe"],
                "The whole shaft in negative skin friction: the upper bound "
                "on the drag force for a single pile. GEC-12 §7.3.5.7 uses "
                "the plane at the toe for group settlement.")
        if S["overloaded"]:
            r["reason"] = none_reason
        else:
            imposed(r, L, "at the toe")

        # Endo et al. (1969), CGPR #56 Table 3.1
        cond = self.endo_bearing_condition
        if cond is None:
            r = row(_endo_name(None), "endo", _SOURCE["endo"],
                    "Neutral plane assumed at a fraction of the embedment; "
                    "an estimate.")
            r["reason"] = ("needs endo_bearing_condition ('floating', "
                           "'stiff_flexible' or 'end_bearing', CGPR #56 "
                           "Table 3.1).")
        else:
            ratio = ENDO_NEUTRAL_PLANE_RATIOS[cond]
            r = row(_endo_name(cond), "endo", _SOURCE["endo"],
                    f"Neutral plane assumed at {ratio:g} x the embedment "
                    f"('{cond}', Table 3.1); an estimate, not computed.")
            if S["overloaded"]:
                r["reason"] = none_reason
            else:
                imposed(r, ratio * L, f"at {ratio:g} L")
        return rows

    def _basis_text(self, S: dict, row: dict) -> str:
        """How the headline neutral plane was located, in plain words."""
        L = self.pile_length
        method = row["method"]
        z_np = row["neutral_plane_depth_m"]
        m = row["toe_mobilization"]
        if self.neutral_plane_method == "auto":
            lead = f"{UFC_PROCEDURE}: "
        else:
            lead = (f"{row['name']} (neutral_plane_method="
                    f"'{self.neutral_plane_method}', {row['source']}): ")
        if method == "force_equilibrium":
            if m == 1.0:
                weight = (" + pile weight" if self.include_pile_weight
                          else "")
                return lead + (
                    f"the neutral plane is where the load curve (Q_dead"
                    f"{weight} + negative skin friction) meets the "
                    f"resistance curve (toe resistance + positive skin "
                    f"friction), with side and base resistance fully "
                    f"mobilized (UFC 3-220-20 §6-7.4 steps 3-5, Fig 6-26).")
            return lead + (
                f"the neutral plane is where the load curve meets the "
                f"resistance curve with the toe at {round(m * 100)}% of its "
                f"nominal resistance (toe_mobilization = {m:g}; UFC 3-220-20 "
                f"§6-5.8.4.2; GEC-12 §7.3.6.1; Siegel et al. 2013).")
        if method == "settlement_compatibility":
            if self.neutral_plane_method == "auto":
                lead += (
                    f"the load and resistance curves do not intersect with "
                    f"the toe at {round(self.toe_mobilization * 100)}% of "
                    f"its resistance, so ")
            if not S["any_soil_settles"]:
                return lead + (
                    "the neutral plane is where the pile and the soil settle "
                    "equally (UFC 3-220-20 §6-5.8.4.3, Fig 6-19). Here no "
                    "soil settles (no settling layer, or no fill or drawdown "
                    "to load it): the pile settles at least as much as the "
                    "soil everywhere and no negative skin friction develops. "
                    "Neutral plane at the pile head, zero drag.")
            if S["sc_toe_yields"]:
                return lead + (
                    "the neutral plane is where the pile and the soil settle "
                    "equally (UFC 3-220-20 §6-5.8.4.3, Fig 6-19; Fellenius "
                    "2021). Down to the intersection of the curves with the "
                    "toe fully mobilized the soil settles more than the pile "
                    "would on an elastic toe, so the toe is pushed to its "
                    "full resistance and the neutral plane is that "
                    "intersection.")
            if z_np >= L - 1e-9:
                return lead + (
                    "the neutral plane is where the pile and the soil settle "
                    "equally (UFC 3-220-20 §6-5.8.4.3, Fig 6-19). The soil "
                    "settles more than the pile all the way down to the toe, "
                    "so the neutral plane is at the pile toe (an end-bearing "
                    "pile, §6-7.4 step 5).")
            return lead + (
                "the neutral plane is where the pile and the soil settle "
                "equally, with the toe force the pile delivers there "
                "compatible with the toe penetration it causes (UFC "
                "3-220-20 §6-5.8.4.3, Fig 6-19; Fellenius 2021). For an "
                "end-bearing pile in a stratum stiffer than the "
                "compressible soil this is near the top of the bearing "
                "layer (§6-7.4 step 5).")
        if method == "bearing_layer_top":
            return lead + (
                "neutral plane at the base of the settling soil (the top of "
                "the bearing stratum), where UFC 3-220-20 §6-7.4 step 5 puts "
                "it for an end-bearing pile in a stratum much stiffer than "
                "the compressible soil. Set by that statement, not computed "
                "from the load and resistance curves.")
        if method == "pile_toe":
            return lead + (
                "neutral plane at the pile toe, the whole shaft in negative "
                "skin friction: the upper bound on the drag force. GEC-12 "
                "§7.3.5.7 places the neutral plane at the toe for group "
                "settlement (Goudreault and Fellenius 1994).")
        cond = self.endo_bearing_condition
        return lead + (
            f"neutral plane assumed at {ENDO_NEUTRAL_PLANE_RATIOS[cond]:g} x "
            f"the embedment for a '{cond}' pile (CGPR #56 §3.2.1, Table 3.1, "
            f"Endo et al. 1969). An estimate, not computed.")

    def _basis_warnings(self, S: dict, np_method: str, z_np: float,
                        toe_force: float, warning_list: list) -> None:
        """The warnings that go with the headline method."""
        z_se = S["z_se"]
        R = S["R_toe"]
        m = self.toe_mobilization
        if np_method == "force_equilibrium":
            if S["any_soil_settles"] and z_np > z_se + 1e-6:
                warning_list.append(
                    f"The neutral plane where the load and resistance curves "
                    f"intersect ({z_np:.2f} m) lies below {z_se:.2f} m, "
                    f"where the computed soil settlement falls to the "
                    f"pile's; the friction between them is counted as "
                    f"negative although that soil settles less than the "
                    f"pile. With full mobilization that is UFC 3-220-20 "
                    f"§6-7.4's conservative assumption for the drag force. "
                    f"For settlement, full base mobilization is not "
                    f"conservative (§6-5.8.4.2): see the 50% and 0% toe "
                    f"mobilization rows (toe_mobilization).")
        elif (np_method == "settlement_compatibility"
              and S["z_fe_m"] is None):
            q0 = float(S["drag_from_top"][0])
            f_total = float(S["f_below"][0])
            delivered = float(S["drag_from_top"][-1])
            L = self.pile_length
            upper = S["shaft_to"](L)
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
                    f"with the whole shaft in negative skin friction), so "
                    f"full base mobilization is impossible")
            warning_list.append(
                f"The load and resistance curves do not intersect: {cause}. "
                f"The neutral plane is placed where the pile and the soil "
                f"settle equally, with the toe force compatible with the toe "
                f"penetration (UFC 3-220-20 §6-5.8.4.3, Fig 6-19): "
                f"{z_np:.2f} m, toe force {toe_force:.1f} kN. Upper bound "
                f"if the soil settled relative to the pile down to the toe: "
                f"neutral plane at the toe, drag load {upper:.1f} kN. To "
                f"intersect the curves instead, give toe_mobilization or Nt "
                f"for the MOBILIZED toe resistance (UFC 3-220-20 "
                f"§6-5.8.4.2: try 0%, 50%, 100%).")

    # ── Private helper methods ────────────────────────────────────────────

    def _discretize(self):
        """Create depth nodes along the pile.

        Returns
        -------
        z_nodes : numpy.ndarray
            Depth array from 0 to pile_length. Each layer boundary inside
            the pile appears twice (the unit skin friction jumps there: the
            first node carries the value above, the second the value below,
            as a repeated depth encodes a jump in a CGPR #56 profile), and
            each water-table depth once (the effective stress has a kink
            there), so the trapezoid integral of the piecewise-linear
            profile is exact for alpha and beta layers (GM-01).
        dz : float
            Nominal node spacing (the grid is uniform between those
            depths).
        """
        L = self.pile_length
        total_nodes = max(
            int(L / 0.25),  # ~0.25 m spacing
            self.n_sublayers * len(self.soil.layers),
            50,
        )
        z_nodes = np.linspace(0, L, total_nodes + 1)
        dz = z_nodes[1] - z_nodes[0]
        tol = 1e-9 * max(L, 1.0)
        bounds, top = [], 0.0
        for layer in self.soil.layers:
            top += layer.thickness
            if tol < top < L - tol:
                bounds.append(top)
        for zb in self._extra_boundaries():
            if tol < zb < L - tol and all(abs(zb - b) > tol for b in bounds):
                bounds.append(zb)
        kinks = []
        for zw in (self.soil.gwt_depth,
                   self.soil.gwt_depth + self.gw_drawdown):
            if (tol < zw < L - tol
                    and all(abs(zw - b) > tol for b in bounds + kinks)):
                kinks.append(zw)
        special = bounds + kinks
        if special:
            keep = [z for z in z_nodes
                    if all(abs(z - s) > tol for s in special)]
            z_nodes = np.array(sorted(keep + kinks + bounds + bounds))
        return z_nodes, dz

    def _extra_boundaries(self) -> tuple:
        """Depths, besides the layer boundaries, where the unit skin
        friction jumps (carried twice in the grid, as a layer boundary is).
        None here; the post-liquefaction analysis adds the liquefied
        zones' top and bottom (``downdrag.liquefaction``)."""
        return ()

    @staticmethod
    def _lookup_depths(z_nodes: np.ndarray) -> np.ndarray:
        """Depth at which each node picks its layer: at a repeated depth (a
        layer boundary) the first node takes the layer above, the second
        the layer below."""
        look = np.array(z_nodes, dtype=float)
        eps = 1e-9 * max(float(z_nodes[-1]), 1.0)
        for i in range(len(z_nodes) - 1):
            if z_nodes[i + 1] == z_nodes[i]:
                look[i] -= eps
                look[i + 1] += eps
        return look

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
        for i, z in enumerate(self._lookup_depths(z_nodes)):
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
        (dead load + dragload, + the pile weight when
        ``include_pile_weight``) equals the cumulative resistance from the
        bottom (toe + positive friction).

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

        Notes
        -----
        Both curves integrate the piecewise-linear unit skin friction
        profile ``fs`` exactly (``_shaft_cumulative``), and the crossing
        inside a segment is the exact root of that integral, so the drag
        load and positive friction reported at the plane balance the curves
        to rounding (GM-01; before 2026-10-14 the curves summed each
        segment with its lower node's friction and the drag load with its
        upper node's, ~4 % apart on DD-1 at 0 % toe).
        """
        perimeter = self.pile_perimeter
        w = self._weight_per_m()
        F = _shaft_cumulative(z_nodes, fs, perimeter)

        # Load from the top (dead load + pile weight when included +
        # negative friction) and resistance from the tip (toe + positive
        # friction upward)
        drag_from_top = self.Q_dead + w * z_nodes + F
        resist_from_tip = toe_resistance + F[-1] - F

        diff = drag_from_top - resist_from_tip
        z_np = None  # no crossing unless found below
        for i in range(len(z_nodes) - 1):
            if diff[i] <= 0 and diff[i + 1] > 0:
                # diff(z_i + t) = diff_i + w t + 2 P (fs_i t + s t^2 / 2):
                # quadratic in t inside the segment; take its root.
                h = z_nodes[i + 1] - z_nodes[i]
                a = perimeter * (fs[i + 1] - fs[i]) / h
                b = w + 2.0 * perimeter * fs[i]
                c = float(diff[i])
                disc = max(b * b - 4.0 * a * c, 0.0)
                den = b + math.sqrt(disc)
                if c == 0.0:
                    t = 0.0
                elif den > 0.0:
                    t = -2.0 * c / den
                else:
                    t = -c / (diff[i + 1] - c) * h
                z_np = float(z_nodes[i] + min(max(t, 0.0), h))
                break
        if z_np is None and diff[-1] == 0.0:
            z_np = float(z_nodes[-1])  # curves meet exactly at the toe

        return z_np, drag_from_top, resist_from_tip

    def _settlement_mismatch(self, z_nodes: np.ndarray, dz: float,
                             drag_from_top: np.ndarray,
                             resist_from_tip: np.ndarray,
                             toe_resistance: float,
                             soil_settlement: np.ndarray):
        """Soil minus pile settlement with the neutral plane at each node.

        For a trial neutral plane at node j the pile carries
        Q_np = drag_from_top[j]; below it positive friction sheds load and
        the toe takes what is left, never more than ``toe_resistance`` and
        never less than zero (when the full positive friction below would
        exceed Q_np, it is only partly mobilized). The pile settlement
        relative to the soil at the toe level is the toe penetration (the
        compression of the soil below the toe under the toe force, on top
        of what the fill / drawdown alone causes there:
        ``_toe_load_compression``) plus the elastic compression of the
        pile between the trial NP and the toe. The soil settlement
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

        g = np.zeros(n)
        for j in range(n):
            q_np = drag_from_top[j]
            if q_toe_raw[j] >= 0 or f_below[j] <= 0:
                k = 1.0
            else:
                k = q_np / f_below[j]
            q = q_np - k * (f_below[j] - f_below[j:])
            compression = (float(np.trapezoid(q, z_nodes[j:])) / AE
                           if AE > 0 and n - j > 1 else 0.0)
            penetration = self._toe_load_compression(float(toe_force[j]))
            g[j] = soil_settlement[j] - (penetration + compression)
        return g, toe_force

    def _axial_load_with_mobilized_toe(self, S: dict,
                                       z_np: float) -> np.ndarray:
        """Axial load when the toe is not fully mobilized.

        Above the neutral plane the load is the load-from-top curve; below
        it positive friction sheds load down to the toe force the pile
        delivers. When the full positive friction below the NP exceeds the
        load at the NP, the friction is scaled so the toe force is zero.
        """
        z_nodes, f_below = S["z"], S["f_below"]
        q_np = self._load_at(S, z_np)
        f_np = float(S["F"][-1]) - S["shaft_to"](z_np)
        k = 1.0 if (q_np >= f_np or f_np <= 0) else q_np / f_np
        below = q_np - k * (f_np - f_below)
        q = np.where(z_nodes <= z_np, S["drag_from_top"], below)
        return np.maximum(q, 0.0)

    def _compute_dragload(self, z_nodes: np.ndarray, dz: float,
                           fs: np.ndarray, z_np: float) -> float:
        """Negative skin friction above the neutral plane (kN): the
        integral of the piecewise-linear ``fs`` profile from the head to
        ``z_np`` times the perimeter (the same integral as the load
        curve)."""
        F = _shaft_cumulative(z_nodes, fs, self.pile_perimeter)
        return _shaft_to(z_np, z_nodes, fs, F, self.pile_perimeter)

    def _compute_positive_resistance(self, z_nodes: np.ndarray, dz: float,
                                      fs: np.ndarray, z_np: float) -> float:
        """Positive shaft resistance below the neutral plane (kN): the same
        integral from ``z_np`` to the toe."""
        F = _shaft_cumulative(z_nodes, fs, self.pile_perimeter)
        return float(F[-1]) - _shaft_to(z_np, z_nodes, fs, F,
                                        self.pile_perimeter)

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

        Above NP: load increases (dead load + dragload, + weight when
        ``include_pile_weight``).
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
        # Above the NP friction adds load (negative skin friction), below it
        # friction removes load: the axial load is the lower of the load
        # curve from the head and the resistance curve from the toe (the
        # same curves, and the same integral, as the NP search).
        _, Q_from_top, Q_from_bot = self._find_neutral_plane(
            z_nodes, dz, fs, toe_resistance)
        return np.minimum(Q_from_top, Q_from_bot)

    # ── Settlement below the toe (GM-49) ──────────────────────────────────

    def _toe_load_depth(self, toe_force: float,
                        criterion: Optional[str] = None) -> float:
        """Depth below the toe (m) down to which the toe load is counted:
        where its stress increase, the toe force spread 2V:1H from an
        equivalent footing B' = L' = the pile width at the toe (UFC
        3-220-20 Eq 6-52), falls below 10 % of

        - ``"bearing_stress"``: the bearing stress of the equivalent
          footing (UFC 3-220-20 §6-5.8.4.3, p. 449). It does not depend on
          the load: (B' + z')(L' + z') = 10 B'L', so z' = (sqrt(10) - 1) B'
          for a square footing;
        - ``"effective_stress"``: the existing vertical effective stress at
          that depth (FHWA GEC-12 §7.3.5.3 step 1c, p. 324); zero when the
          toe carries no load.

        Below the bottom of the profile sigma'v0 is continued with the
        deepest layer's unit weight, only to say how far a profile that
        stops short should go.
        """
        crit = criterion or self.influence_depth_criterion
        B = L = self.pile_diameter
        f = INFLUENCE_STRESS_FRACTION
        if crit == "bearing_stress":
            s = B + L
            return 0.5 * (-s + math.sqrt(s * s + 4.0 * B * L * (1.0 / f - 1.0)))
        if toe_force <= 0:
            return 0.0
        z0 = self.pile_length

        def excess(dz):
            return (toe_force / ((B + dz) * (L + dz))
                    - f * self._sigma_v0_below(z0 + dz))

        if excess(0.0) <= 0:
            return 0.0
        lo, hi = 0.0, B
        while excess(hi) > 0 and hi < 1e4:
            lo, hi = hi, 2.0 * hi
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if excess(mid) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _sigma_v0_below(self, z: float) -> float:
        """Initial effective vertical stress (kPa) at depth ``z``; below the
        bottom of the profile, continued with the deepest layer's unit
        weight (submerged below the water table)."""
        bottom = self.soil.total_depth
        if z <= bottom:
            return self.soil.effective_stress_at_depth(z)
        gamma = self.soil.layers[-1].unit_weight
        dry = max(0.0, min(z, self.soil.gwt_depth) - bottom)
        wet = (z - bottom) - dry
        return (self.soil.effective_stress_at_depth(bottom) + gamma * dry
                + (gamma - self.soil.gamma_w) * wet)

    def _below_toe_sublayers(self, z_bot: float, graded: bool) -> list:
        """``(thickness, layer, sigma'v0, fill / drawdown stress change,
        depth below the toe)`` at the midpoint of each sublayer from the
        toe to ``z_bot`` (within the profile). Sublayers break at every
        layer boundary and water-table depth and are at most 0.25 m thick;
        ``graded`` makes them 0.1 x (B' + z') thick near the toe, where the
        toe load's stress falls fast."""
        z_top = self.pile_length
        z_bot = min(z_bot, self.soil.total_depth)
        if z_bot <= z_top + 1e-9:
            return []
        breaks = {z_top, z_bot}
        top = 0.0
        for layer in self.soil.layers:
            top += layer.thickness
            if z_top < top < z_bot:
                breaks.add(top)
        for zw in (self.soil.gwt_depth,
                   self.soil.gwt_depth + self.gw_drawdown):
            if z_top < zw < z_bot:
                breaks.add(zw)
        breaks = sorted(breaks)
        edges = [breaks[0]]
        for a, b in zip(breaks[:-1], breaks[1:]):
            if not graded:
                n = max(int(math.ceil((b - a) / _SUBLAYER - 1e-9)), 1)
                edges += list(np.linspace(a, b, n + 1)[1:])
                continue
            z = a
            while b - z > 1e-9:
                h = min(_SUBLAYER, 0.1 * (self.pile_diameter + z - z_top))
                z = b if b - (z + h) < 0.5 * h else z + h
                edges.append(z)
        edges = np.array(edges)
        z_mid = 0.5 * (edges[:-1] + edges[1:])
        ds_other = self._compute_stress_change(z_mid)
        out = []
        for zm, h, dso in zip(z_mid, np.diff(edges), ds_other):
            layer = self.soil.layer_at_depth(zm)
            out.append((float(h), layer,
                        self.soil.effective_stress_at_depth(zm),
                        float(dso), float(zm - z_top)))
        return out

    @staticmethod
    def _compression(layer, H: float, sigma_v0: float,
                     delta_sigma: float) -> float:
        """Compression of a sublayer (m): UFC Eq 6-53 for clay with C_ec,
        Eq 6-54 for sand with E_s, zero for soil with no compressibility."""
        if delta_sigma <= 0 or sigma_v0 <= 0:
            return 0.0
        if layer.soil_type == "cohesive":
            if layer.C_ec is None:
                return 0.0
            sigma_p = (layer.sigma_p if layer.sigma_p is not None
                       else sigma_v0)
            return _settlement_clay(H=H, C_ec=layer.C_ec, C_er=layer.C_er,
                                    sigma_v0=sigma_v0, sigma_p=sigma_p,
                                    delta_sigma=delta_sigma)
        if layer.E_s is not None and layer.E_s > 0:
            return _settlement_sand_elastic(H=H, nu_s=layer.nu_s,
                                            E_s=layer.E_s,
                                            delta_sigma=delta_sigma)
        return 0.0

    def _fill_settlement_below_toe(self) -> float:
        """Settlement of the soil below the toe under the fill / drawdown
        alone (m): the settling layers from the toe to the bottom of the
        soil profile. A wide fill's stress increase (and a drawdown's,
        below the lowered water table) does not decay with depth (UFC
        3-220-20 Eq 6-52, the Δσz,other term; the 10 % rules of UFC
        §6-5.8.4.3 and GEC-12 §7.3.5.3 are for the foundation load), so no
        depth criterion cuts it off. A layer not marked ``settling`` does
        not settle under the fill here, as along the shaft."""
        total = 0.0
        for h, layer, sv0, dso, _ in self._below_toe_sublayers(
                self.soil.total_depth, graded=False):
            if layer.settling:
                total += self._compression(layer, h, sv0, dso)
        return total

    def _toe_load_compression(self, toe_force: float,
                              criterion: Optional[str] = None) -> float:
        """Compression of the soil below the toe under the toe load (m),
        the toe penetration relative to the soil at the toe level: the toe
        force spread 2V:1H from an equivalent footing B' = L' = the pile
        width at the toe (UFC 3-220-20 Eqs 6-49 to 6-52; "for a single
        element, the dimensions of the footing are equal to the width of
        the element", §6-5.8.4.2), counted down to ``_toe_load_depth`` or
        the bottom of the profile. Each sublayer adds its compression under
        the toe load + the fill / drawdown less that under the fill /
        drawdown alone: the toe load acts on the stress the fill leaves."""
        if toe_force <= 0:
            return 0.0
        z_end = self.pile_length + self._toe_load_depth(toe_force, criterion)
        B = self.pile_diameter
        total = 0.0
        for h, layer, sv0, dso, dz in self._below_toe_sublayers(
                z_end, graded=True):
            dq = toe_force / ((B + dz) * (B + dz))
            total += (self._compression(layer, h, sv0, dq + dso)
                      - self._compression(layer, h, sv0, dso))
        return max(total, 0.0)

    def _below_toe_info(self, toe_force: float, s_fill: float,
                        s_toe: float) -> dict:
        """How deep the settlement below the toe was counted and why it
        stopped (the result's ``below_toe``), with what the other criterion
        gives for the same toe force."""
        L = float(self.pile_length)
        bottom = float(self.soil.total_depth)
        B = float(self.pile_diameter)
        crit = self.influence_depth_criterion
        what, src = _INFLUENCE[crit]
        z_q = self._toe_load_depth(toe_force) if toe_force > 0 else 0.0
        if toe_force <= 0:
            stop, toe_to = "no_toe_load", L
        elif bottom < L + z_q - 1e-9:
            stop, toe_to = "profile_end", max(bottom, L)
        else:
            stop, toe_to = "criterion", L + z_q
        stress_change = self.fill_thickness > 0 or self.gw_drawdown > 0
        settling_below = any(
            layer.settling for _, layer, _, _, _ in
            self._below_toe_sublayers(bottom, graded=False))
        fill_to = max(bottom, L) if stress_change else None

        if stop == "no_toe_load":
            toe_text = (f"Below the toe ({L:.2f} m) the toe carries no load, "
                        f"so only the fill / drawdown can compress the soil "
                        f"there.")
        else:
            head = (f"Below the toe ({L:.2f} m) the toe force "
                    f"({toe_force:.1f} kN) is spread 2V:1H from an "
                    f"equivalent footing {B:.2f} m square at the toe (UFC "
                    f"3-220-20 Eqs 6-49 to 6-52)")
            if stop == "criterion":
                toe_text = (f"{head} and counted {z_q:.2f} m below the toe, "
                            f"to {toe_to:.2f} m, where its stress increase "
                            f"falls below {what} ({src}).")
            elif bottom <= L + 1e-9:
                toe_text = (f"{head}, but the soil profile ends at the toe, "
                            f"above {L + z_q:.2f} m, where its stress "
                            f"increase would fall below {what} ({src}): "
                            f"nothing below the toe is counted.")
            else:
                toe_text = (f"{head} and counted to the bottom of the soil "
                            f"profile, {bottom:.2f} m, above {L + z_q:.2f} m, "
                            f"where its stress increase would fall below "
                            f"{what} ({src}).")
        if not stress_change:
            fill_text = ("There is no fill or drawdown, so nothing else "
                         "compresses the soil below the toe.")
        elif bottom <= L + 1e-9:
            fill_text = ("The fill / drawdown stress increase does not "
                         "decay with depth, but the soil profile ends at "
                         "the toe, so nothing below it is counted.")
        elif not settling_below:
            fill_text = (f"The fill / drawdown stress increase does not "
                         f"decay with depth, but no layer between the toe "
                         f"and the bottom of the soil profile ({bottom:.2f} "
                         f"m) is marked settling, so it adds no settlement "
                         f"there.")
        else:
            fill_text = (f"The fill / drawdown stress increase does not "
                         f"decay with depth, so the settling layers below "
                         f"the toe are counted down to the bottom of the "
                         f"soil profile, {bottom:.2f} m: "
                         f"{s_fill * 1000:.1f} mm under the fill / drawdown "
                         f"alone.")
        other = next(c for c in INFLUENCE_DEPTH_CRITERIA if c != crit)
        o_what, o_src = _INFLUENCE[other]
        o_depth = (self._toe_load_depth(toe_force, other)
                   if toe_force > 0 else 0.0)
        o_toe = self._toe_load_compression(toe_force, other)
        other_text = ""
        if toe_force > 0:
            other_text = (
                f" With {o_what} instead ({o_src}; influence_depth_criterion"
                f"='{other}') the toe force is counted {o_depth:.2f} m below "
                f"the toe: ")
            if abs(o_toe - s_toe) < 5e-5:
                other_text += (f"the same settlement below the toe, "
                               f"{(s_fill + s_toe) * 1000:.1f} mm.")
            else:
                other_text += (f"settlement below the toe "
                               f"{(s_fill + o_toe) * 1000:.1f} mm against "
                               f"{(s_fill + s_toe) * 1000:.1f} mm.")
        return {
            "influence_depth_criterion": crit,
            "criterion": f"{what} ({src})",
            "equivalent_footing_width_m": B,
            "toe_force_kN": float(toe_force),
            "bearing_stress_kPa": float(toe_force) / (B * B),
            "toe_load_depth_below_toe_m": z_q,
            "toe_load_counted_to_m": toe_to,
            "toe_load_stopped_by": stop,
            "fill_counted_to_m": fill_to,
            "fill_settlement_mm": s_fill * 1000.0,
            "toe_load_compression_mm": s_toe * 1000.0,
            "settlement_below_toe_mm": (s_fill + s_toe) * 1000.0,
            "depth_m": max(toe_to, fill_to if fill_to is not None else L),
            "other_criterion": {
                "influence_depth_criterion": other,
                "criterion": f"{o_what} ({o_src})",
                "toe_load_depth_below_toe_m": o_depth,
                "settlement_below_toe_mm": (s_fill + o_toe) * 1000.0,
            },
            "basis": f"{toe_text} {fill_text}{other_text}",
        }

    def _unrated_below_toe(self, toe_force: float) -> List[tuple]:
        """``(top, bottom, soil type)`` of the soil below the toe that has
        no compressibility (clay without C_ec or Cc and e0, sand without
        E_s) where a load acts on it: the toe load's zone (from the toe to
        ``_toe_load_depth``, as :meth:`_toe_load_compression`) when the toe
        carries load, and with a fill or drawdown everything down to the
        bottom of the profile (its stress increase does not decay with
        depth). ``bottom`` is None when the profile ends at the toe and only
        the fill / drawdown reaches below it."""
        z0 = float(self.pile_length)
        z_toe = (z0 + self._toe_load_depth(toe_force) if toe_force > 0
                 else z0)
        fill = self.fill_thickness > 0 or self.gw_drawdown > 0
        z1 = max(z_toe, self.soil.total_depth if fill else z0)
        out = []
        top = 0.0
        for layer in self.soil.layers:
            bottom = top + layer.thickness
            a, b = max(top, z0), min(bottom, z1)
            if b > a:
                if layer.soil_type == "cohesive":
                    rated = layer.C_ec is not None and layer.C_ec > 0
                else:
                    rated = layer.E_s is not None and layer.E_s > 0
                if not rated:
                    out.append((a, b, layer.soil_type))
            top = bottom
        if top < z1 - 1e-9:
            # the profile stops above the bottom of the toe load's zone
            out.append((max(top, z0), z1, "not in the soil profile"))
        elif fill and top <= z0 + 1e-9:
            # nothing below the toe at all, and the fill / drawdown reaches
            # whatever is there
            out.append((z0, None, "not in the soil profile"))
        return out


# ── Module-level helper functions ─────────────────────────────────────────

def _shaft_cumulative(z_nodes: np.ndarray, fs: np.ndarray,
                      perimeter: float) -> np.ndarray:
    """Shaft resistance from the head to each node (kN): the exact integral
    of the piecewise-linear unit skin friction profile through the nodal
    values (the trapezoid rule), times the perimeter. The drag load, the
    positive friction and both curves of the neutral plane search all use
    this one integral."""
    seg = 0.5 * (fs[:-1] + fs[1:]) * np.diff(z_nodes) * perimeter
    return np.concatenate(([0.0], np.cumsum(seg)))


def _shaft_to(z: float, z_nodes: np.ndarray, fs: np.ndarray,
              F: np.ndarray, perimeter: float) -> float:
    """Shaft resistance from the head to depth ``z`` (kN), exact for the
    piecewise-linear profile, consistent with ``_shaft_cumulative``."""
    if z <= z_nodes[0]:
        return 0.0
    if z >= z_nodes[-1]:
        return float(F[-1])
    i = int(np.searchsorted(z_nodes, z, side="right")) - 1
    h = z_nodes[i + 1] - z_nodes[i]
    t = z - z_nodes[i]
    f_z = fs[i] + (fs[i + 1] - fs[i]) * t / h
    return float(F[i] + 0.5 * (fs[i] + f_z) * t * perimeter)


def _methods_range(rows: list) -> str:
    """One plain sentence: the spread of the neutral plane and the drag
    force over every method that applies."""
    ok = [r for r in rows if r["applies"]]
    if not ok:
        return ("No method gives a neutral plane: the dead load exceeds the "
                "nominal geotechnical resistance.")
    nps = [r["neutral_plane_depth_m"] for r in ok]
    dfs = [r["drag_force_kN"] for r in ok]
    if len(ok) == 1:
        return (f"Only one of the {len(rows)} methods applies: neutral plane "
                f"{nps[0]:.2f} m, drag force {dfs[0]:.0f} kN.")
    return (f"The methods give neutral planes from {min(nps):.2f} m to "
            f"{max(nps):.2f} m and drag forces from {min(dfs):.0f} kN to "
            f"{max(dfs):.0f} kN ({len(ok)} of {len(rows)} methods apply).")


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
    """Neutral plane where pile and soil settle equally, with an
    elastic-plastic toe (UFC 3-220-20 Fig 6-19).

    ``g`` is soil minus pile settlement for a trial plane at each node,
    with the toe elastic (``_settlement_mismatch``). Below the
    intersection of the curves ``z_fe`` (toe fully mobilized) equilibrium
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
    if C_er is None:
        raise ValueError(
            f"sigma_p ({sigma_p:g} kPa) makes this clay overconsolidated at "
            f"sigma'v0 = {sigma_v0:g} kPa, and UFC Eq 6-53 then needs the "
            f"recompression index: give C_er (= Cr/(1+e0)), or Cc, Cr and "
            f"e0")
    if sigma_final <= sigma_p:
        return C_er * H * math.log10(sigma_final / sigma_v0)
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
