"""
Combined axial pile capacity analysis.

Auto-selects the appropriate method for each soil layer:
- Nordlund for cohesionless layers (sand/gravel)
- the alpha-method (Tomlinson's adhesion, GEC-12 Figure 7-17) for cohesive
  layers (clay/silt)

Also supports the beta (effective stress) method for any soil type.

Each method follows the GEC-12 step-by-step procedure, with its limits:
sigma'p at the toe limited to 3 ksf for the Nordlund toe (Step 8); the
alpha method's Ca read off Figure 7-17 as drawn (Step 1), and past the
chart's curves API RP 2A's alpha where the clay is highly overconsolidated
(p. 247), by an su / sigma'vo the engineer gives; an
open-end pipe pile as the lesser of the plugged and unplugged cases (Eqs.
7-88/7-89), with the interior side resistance a fraction of the exterior,
the soil plug weight subtracted and Tomlinson's 105 ksf limiting unit toe
resistance in cohesionless soil, or Tomlinson's Eq. 7-90 for a pipe under
24 inches in predominantly cohesive soil (Sec. 7.10.7); uplift as the whole
shaft resistance (Sec. 7.2.3.1). Where GEC-12 leaves a judgment (a range or
"local experience"), it is an input with a cited default and the result's
``choices`` say what was used. Options beside the defaults (GM-144): Figure
7-18's stratigraphy cases or the smaller of Figures 7-17 and 7-18 (p. 246),
a shrink-swell zone (p. 247), Table 7-9's beta and Nt by soil type, and the
toe phi averaged from the toe to 3b below (Step 7c).

All units are SI: kN, m, kPa.

References:
    FHWA GEC-12 (FHWA-NHI-16-009), Chapter 7
"""

import warnings
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import numpy as np

from axial_pile.pile_types import PileSection
from axial_pile.soil_profile import AxialSoilProfile, AxialSoilLayer
from axial_pile.tomlinson import (
    BEYOND_SOURCES, CHART_END_KSF, CURVES, FIG_7_18_CASES,
    SAND_OVER_STIFF_CLAY, STIFF_CLAY, adhesion, adhesion_fig_7_18,
    curve_for_pile, end_bearing_cohesive, fig_7_18_range_ksf,
)
from axial_pile.nordlund import (
    FIG_7_9_CURVE, KSF, SIGMA_P_LIMIT, delta_phi_ratio_fig_7_9,
    displaced_volume, end_bearing_cohesionless, nordlund_side_factors,
)
from axial_pile.beta_method import (
    TABLE_7_9, Nt_from_phi, beta_from_phi, end_bearing_beta, table_7_9,
)
from axial_pile.results import AxialPileResult

#: Side-resistance method names in a result's ``choices`` (plain text: the
#: agent reads them as JSON).
_METHOD_TEXT = {"nordlund": "Nordlund", "tomlinson": "Tomlinson alpha",
                "beta": "beta method"}

#: GEC-12 Sec. 7.2.1.3.1 (Step 8): the vertical effective stress at the
#: toe used for the Nordlund toe resistance is limited to 3 ksf (kPa).
_TOE_STRESS_LIMIT = SIGMA_P_LIMIT

#: GEC-12 Sec. 7.10.7 (p. 407): Tomlinson's limiting unit toe resistance of
#: an open-end pipe pile in cohesionless soil, 105 ksf (kPa).
OPEN_PIPE_TOE_LIMIT = 105.0 * KSF

#: GEC-12 p. 407: an interior unit shaft resistance "typically on the order
#: of 1/3 to 1/2 the exterior"; the default is the low end, since the lesser
#: of the plugged and unplugged cases governs.
INSIDE_FRICTION_RATIO = 1.0 / 3.0

#: GEC-12 p. 252 / p. 256: an Nt of 3 in clay is about the undrained toe.
COHESIVE_NT = 3.0

#: GEC-12 Sec. 7.2.3.1 (Eq. 7-38): uplift is the whole shaft resistance.
UPLIFT_SKIN_FRACTION = 1.0

_INCH = 0.0254
#: GEC-12 p. 407: Tomlinson's Eq. 7-90 for open-end pipes under 24 inches.
_EQ_7_90_MAX_OD = 24 * _INCH
#: GEC-12 p. 232: piles larger than 18 inches were not in the correlation
#: database of the Nordlund method or the alpha-method; the Nordlund method
#: tends to over predict for pile widths larger than 24 inches.
_DATABASE_MAX_WIDTH = 18 * _INCH
_NORDLUND_OVERPREDICT_WIDTH = 24 * _INCH

_OPEN_PIPE_METHODS = ("auto", "paikowsky_whitman", "tomlinson")
_ADHESION_CURVES = ("auto",) + tuple(CURVES)
#: The alpha method where su is past the charts' curves: API alpha where
#: the clay is highly overconsolidated (GEC-12 p. 247 and Step 1, p. 250),
#: by the su / sigma'vo the engineer gives, Ca held at the curve's end
#: elsewhere; or one of the two in every such layer.
API_IF_HIGHLY_OC = "api_if_highly_overconsolidated"
_BEYOND_CHART = (API_IF_HIGHLY_OC, "chart_end", "api")
_H_TOE_AREAS = ("box", "steel")
#: GEC-12 p. 246: Figure 7-17 generally, Figure 7-18 for its stratigraphy
#: cases, and the smaller of the two for the inexperienced user.
_ADHESION_CHARTS = ("fig_7_17", "fig_7_18", "smaller")
_BETA_COEFFICIENTS = ("from_phi", "table_7_9")
_TOE_PHI_ZONES = ("toe_layer", "average_3b")


class JudgmentNeeded(ValueError):
    """A step of the manual leaves a call to the engineer, the run needs
    it and no input gives it: the analysis is refused, saying which call
    and what to give."""


@dataclass
class AxialPileAnalysis:
    """Axial pile capacity analysis for a single driven pile.

    Parameters
    ----------
    pile : PileSection
        Pile cross-section properties.
    soil : AxialSoilProfile
        Layered soil profile, depths below the ground surface.
    pile_length : float
        Embedded pile length (m), head to toe.
    method : str, optional
        Analysis method:
        - "auto" (default): Nordlund for sand, the alpha-method for clay
        - "beta": Effective stress (beta) method for all layers
    factor_of_safety : float, optional
        Factor of safety. Default 2.5.
    include_uplift : bool, optional
        If True, also compute uplift (tension) capacity. Default False.
    cohesive_phi : float, optional
        FALLBACK effective friction angle (degrees) for COHESIVE layers when
        ``method="beta"`` (used for the side resistance). A per-layer
        ``friction_angle`` supplied on a cohesive layer WINS over this
        global; ``cohesive_phi`` applies only to cohesive layers without
        one. Default 25.0 — within the clay range 25-30 deg of GEC-12
        Table 7-9. Ignored for cohesionless layers and for "auto".
    uplift_skin_fraction : float, optional
        Fraction of the OUTSIDE side resistance taken as the uplift
        resistance. Default 1.0: GEC-12 Sec. 7.2.3.1 (Eq. 7-38) takes the
        nominal shaft resistance, with the reduction for uplift in the
        uplift resistance factor; De Nicola and Randolph's customary 70 % in
        free-draining soils (cited there) is the alternative.
    pile_weight : float, optional
        Pile self-weight (kN) added to the uplift capacity. Default None
        (self-weight NOT credited — conservative). Supply the buoyant
        weight below the water table if you want it included.
    head_depth : float, optional
        Depth of the pile head below the ground surface (m). Default 0.
    toe_stress_limit : float or None, optional
        Limit on sigma'p at the toe for the Nordlund toe resistance (kPa).
        Default 3 ksf (143.6 kPa), GEC-12 Step 8. None applies no limit (as
        GEC-12's own worked example, Vol 3 Table D-6, did).
    inside_friction_ratio : float, optional
        Interior over exterior unit side resistance of an open-end pipe pile
        in the unplugged case. Default 1/3, the low end of GEC-12's
        "typically on the order of 1/3 to 1/2" (Sec. 7.10.7, p. 407).
    open_pipe_toe_limit : float or None, optional
        Limiting unit toe resistance of an open-end pipe pile in
        cohesionless soil (kPa). Default 105 ksf (5,027 kPa), Tomlinson
        (1994) per GEC-12 p. 407. None applies none.
    open_pipe_method : str, optional
        "auto" (default): Tomlinson's Eq. 7-90 for a pipe under 24 inches
        in predominantly cohesive soil (GEC-12 p. 407; taken here as the toe
        in clay and more than half the embedded length in clay layers),
        else the lesser of Eqs. 7-88 and 7-89 (Paikowsky and Whitman).
        "paikowsky_whitman" or "tomlinson" force one.
    cohesive_Nt : float, optional
        Nt for a cohesive toe layer in the beta method. Default 3 (GEC-12
        p. 256: a relatively low Nt in clays unless local correlations
        suggest higher). A layer's own ``Nt`` wins.
    h_pile_toe_area : str, optional
        "box" (default; GEC-12 Sec. 7.10.7, p. 408) or "steel": the toe area
        of an H-pile. GEC-12 Step 9 leaves it to past experience and local
        correlations.
    adhesion_curve : str, optional
        Figure 7-17 curves for the alpha-method: "auto" (default; smooth
        steel for pipe piles, the concrete, timber and corrugated steel
        curve for H-piles per GEC-12 p. 247, concrete and timber piles),
        "concrete_timber_corrugated" or "smooth_steel".
    adhesion_beyond_chart : str, optional
        The alpha method where a clay layer's su is past the end of the
        chart's curves (Figure 7-17, or 7-18 where it is read). GEC-12 p.
        247 and Step 1 (p. 250) recommend API RP 2A's alpha there for
        highly overconsolidated clays. "api_if_highly_overconsolidated"
        (default): API alpha where the layer's su / sigma'vo is at least
        ``highly_overconsolidated_su_ratio``, Ca held at the curve's end
        where it is less. "chart_end" holds Ca at the curve's end in every
        such layer (GEC-12's worked example, Vol 3 Table D-100, did); "api"
        takes API alpha in every such layer. Within the curves Ca is read
        as drawn.
    highly_overconsolidated_su_ratio : float or None, optional
        The su / sigma'vo at and above which a clay is highly
        overconsolidated, for ``adhesion_beyond_chart``
        "api_if_highly_overconsolidated". GEC-12 gives no value: it is the
        engineer's judgment. Default None: an analysis with a clay layer
        past the curves is refused until it is given (sigma'vo at the
        middle of the layer's length along the pile).
    displaced_volume : float or None, optional
        Displaced volume V per unit length (m3/m) for the Nordlund charts.
        Default None: the steel area of an H-pile, the gross area of other
        piles (an open pipe included).
    adhesion_chart : str, optional
        The alpha-method's chart (GEC-12 p. 246): "fig_7_17" (default;
        "It is recommended that Figure 7-17 generally be used"),
        "fig_7_18" (Figure 7-18's alpha for each clay layer's stratigraphy
        case; a layer whose case is "none" keeps Figure 7-17) or "smaller"
        (the smaller Ca of the two in each layer: "the inexperienced user
        should select and use the smaller value obtained from either
        figure"). A clay layer's case is its ``fig_7_18_case``, else from
        the profile: contiguous cohesive layers are one clay stratum,
        "sand_over_stiff_clay" under a cohesionless layer and "stiff_clay"
        from the ground surface; "soft_clay_over_stiff_clay" only where a
        layer gives it (GEC-12 gives no su that separates them). D for
        Figure 7-18 runs from the top of the clay stratum to the bottom of
        the layer, or to the toe if less.
    shrink_swell_depth : float or None, optional
        Depth (m below the ground surface) of a shrink-swell zone in clay:
        no shaft resistance in cohesive soil above it (GEC-12 p. 247, the
        alpha-method; also applied to the beta method in cohesive soil).
        Default None: no zone.
    beta_coefficients : str, optional
        The beta method's beta and Nt where a layer gives none (Steps 2b,
        5b): "from_phi" (default; beta = (1 - sin phi') tan phi', Fellenius
        1991, and Nt within Table 7-9's ranges by phi') or "table_7_9"
        (Table 7-9 by the layer's soil type, linear in phi' across the
        type's phi' range). A clay toe keeps ``cohesive_Nt`` either way.
    toe_phi_zone : str, optional
        The phi near the toe for the Nordlund toe (Steps 7 and 9):
        "toe_layer" (default; the toe layer's toe_friction_angle or
        friction_angle) or "average_3b" (the thickness-weighted average of
        the cohesionless layers' toe phi from the toe to 3b below, the zone
        of Step 7c).

    Notes
    -----
    Side resistance is integrated layer-by-layer using the midpoint rule
    (unit side resistance at the mid-depth of each layer segment within the
    pile, times segment length). Segments are split at the groundwater
    table when it falls inside a layer, so the kink in sigma_v' there does
    not bias the integral.

    Uplift (when ``include_uplift=True``) is
    ``uplift_skin_fraction * Q_skin_outside`` (+ ``pile_weight`` if given).
    Side resistance inside an open-ended pipe is left out of uplift.

    Examples
    --------
    >>> pile = make_pipe_pile(0.3239, 0.00953, closed_end=True)
    >>> soil = AxialSoilProfile(layers=[
    ...     AxialSoilLayer(10, 'cohesionless', 18.0, friction_angle=30),
    ...     AxialSoilLayer(5, 'cohesive', 17.0, cohesion=50),
    ... ])
    >>> analysis = AxialPileAnalysis(pile=pile, soil=soil, pile_length=15)
    >>> result = analysis.compute()
    """
    pile: PileSection = None
    soil: AxialSoilProfile = None
    pile_length: float = 0.0
    method: str = "auto"
    factor_of_safety: float = 2.5
    include_uplift: bool = False
    cohesive_phi: float = 25.0
    uplift_skin_fraction: float = UPLIFT_SKIN_FRACTION
    pile_weight: Optional[float] = None
    head_depth: float = 0.0
    toe_stress_limit: Optional[float] = _TOE_STRESS_LIMIT
    inside_friction_ratio: float = INSIDE_FRICTION_RATIO
    open_pipe_toe_limit: Optional[float] = OPEN_PIPE_TOE_LIMIT
    open_pipe_method: str = "auto"
    cohesive_Nt: float = COHESIVE_NT
    h_pile_toe_area: str = "box"
    adhesion_curve: str = "auto"
    adhesion_beyond_chart: str = API_IF_HIGHLY_OC
    highly_overconsolidated_su_ratio: Optional[float] = None
    displaced_volume: Optional[float] = None
    adhesion_chart: str = "fig_7_17"
    shrink_swell_depth: Optional[float] = None
    beta_coefficients: str = "from_phi"
    toe_phi_zone: str = "toe_layer"

    def __post_init__(self):
        if self.pile is None:
            raise ValueError("Pile section must be provided")
        if self.soil is None:
            raise ValueError("Soil profile must be provided")
        if self.pile_length <= 0:
            raise ValueError(f"Pile length must be positive, got {self.pile_length}")
        if self.head_depth < 0:
            raise ValueError(
                f"head_depth must be >= 0 (depth of the pile head below the "
                f"ground surface), got {self.head_depth}"
            )
        for name, allowed in (("open_pipe_method", _OPEN_PIPE_METHODS),
                              ("adhesion_curve", _ADHESION_CURVES),
                              ("adhesion_beyond_chart", _BEYOND_CHART),
                              ("h_pile_toe_area", _H_TOE_AREAS),
                              ("adhesion_chart", _ADHESION_CHARTS),
                              ("beta_coefficients", _BETA_COEFFICIENTS),
                              ("toe_phi_zone", _TOE_PHI_ZONES)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {list(allowed)}, "
                                 f"got '{getattr(self, name)}'")
        if not 0.0 <= self.inside_friction_ratio <= 1.0:
            raise ValueError("inside_friction_ratio must be between 0 and 1, "
                             f"got {self.inside_friction_ratio}")
        for name in ("toe_stress_limit", "open_pipe_toe_limit",
                     "displaced_volume", "shrink_swell_depth",
                     "highly_overconsolidated_su_ratio"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 or None, got {value}")
        if self.cohesive_Nt <= 0:
            raise ValueError(f"cohesive_Nt must be > 0, got {self.cohesive_Nt}")
        if self.uplift_skin_fraction < 0:
            raise ValueError("uplift_skin_fraction must be >= 0, got "
                             f"{self.uplift_skin_fraction}")
        if self.head_depth + self.pile_length > self.soil.total_thickness:
            warnings.warn(
                f"Pile tip (head_depth {self.head_depth}m + pile_length "
                f"{self.pile_length}m) exceeds soil profile "
                f"({self.soil.total_thickness}m); using full profile depth"
            )

    @property
    def _tip_depth(self) -> float:
        """Absolute depth of the pile tip below the ground surface (m).

        With the default ``head_depth = 0`` this equals ``pile_length``.
        """
        return self.head_depth + self.pile_length

    @property
    def _open_pipe(self) -> bool:
        return (not self.pile.closed_end
                and self.pile.tip_area_plugged is not None)

    def _displaced_area(self) -> Tuple[float, str]:
        """(area in m2, what it is): the displacing area for V.

        H-piles: the steel area (GEC-12 Fig. 7-9 curve f; Vol 3 Table D-6
        uses it). An open pipe: the gross area, as a closed pipe. GEC-12
        does not say which V an open pipe takes; its steel area is far
        below Fig. 7-9 curve a (from 0.3 ft3/ft), so it stays an input."""
        if self.displaced_volume is not None:
            return self.displaced_volume, "given"
        if self.pile.pile_type == "h_pile":
            return self.pile.area, "steel area of the H-pile"
        if self.pile.pile_type == "pipe_open":
            return self.pile.tip_area_plugged, ("gross area of the open "
                                                "pipe, as a closed pipe")
        return self.pile.tip_area, "gross area of the pile"

    def _adhesion_curve(self) -> str:
        if self.adhesion_curve == "auto":
            return curve_for_pile(self.pile.pile_type)
        return self.adhesion_curve

    def _toe_area(self) -> float:
        if self.pile.pile_type == "h_pile" and self.h_pile_toe_area == "steel":
            return self.pile.area
        return self.pile.tip_area

    def compute(self) -> AxialPileResult:
        """Run the axial capacity analysis.

        Returns
        -------
        AxialPileResult
            Complete results with side and toe resistance, and the per-layer
            breakdown.
        """
        # ``notes`` collects what the user must see (a value read off a
        # chart's edge, a pile outside GEC-12's correlation database); they
        # ride on the result as its warnings.
        notes: List[str] = []
        pile = self.pile
        tip_depth = self._tip_depth

        area_V, _ = self._displaced_area()
        V = displaced_volume(area_V)
        layer_results, shaft_layers, total_Qs = self._shaft(V, notes)
        shaft_notes = len(notes)

        # Outside side resistance (before any inside/plug friction)
        Qs_outside = total_Qs

        tip_layer = self.soil.layer_at_depth(tip_depth - 0.01)
        sigma_v_tip = self.soil.effective_stress_at_depth(tip_depth)
        beta = self.method == "beta"
        open_pipe = self._open_pipe
        q_cap = (self.open_pipe_toe_limit
                 if open_pipe and tip_layer.soil_type == "cohesionless"
                 else None)
        nordlund_toe = not beta and tip_layer.soil_type == "cohesionless"
        toe_phi, toe_zone = (self._toe_phi(tip_layer, notes) if nordlund_toe
                             else (tip_layer.toe_phi, None))

        def toe(area: float, details: dict) -> float:
            """Toe resistance (kN) on ``area`` by the toe layer's method."""
            if beta:
                Nt, source = self._toe_Nt(tip_layer, notes)
                q_p = Nt * sigma_v_tip
                governs = "Eq. 7-20"
                if q_cap is not None and q_cap < q_p:
                    q_p, governs = q_cap, "open-end pipe limit"
                details.update({"N_t": Nt, "N_t_source": source,
                                "sigma_v_tip_kPa": sigma_v_tip,
                                "q_p_kPa": q_p, "governs": governs,
                                "toe_area_m2": area})
                return q_p * area
            if tip_layer.soil_type == "cohesionless":
                details["toe_phi_deg"] = toe_phi
                if toe_zone:
                    details["toe_zone"] = toe_zone
                return end_bearing_cohesionless(
                    toe_phi, sigma_v_tip, area, self.pile_length,
                    pile.width, notes=notes,
                    sigma_p_limit=self.toe_stress_limit, q_limit=q_cap,
                    details=details)
            details.update({"s_u_kPa": tip_layer.cohesion,
                            "q_p_kPa": 9.0 * tip_layer.cohesion,
                            "governs": "Eq. 7-11", "toe_area_m2": area})
            return end_bearing_cohesive(tip_layer.cohesion, area)

        toe_details: dict = {}
        plug_case = None
        open_pipe_info = None
        plug_weight = 0.0
        if open_pipe:
            # GEC-12 Sec. 7.10.7: the lesser of the plugged (Eq. 7-88) and
            # unplugged (Eq. 7-89) cases, or Tomlinson's Eq. 7-90.
            pw_method, reading = self._open_pipe_method(tip_layer,
                                                        layer_results)
            plugged_details: dict = {}
            Qt_plugged = toe(pile.tip_area_plugged, plugged_details)
            Q_plugged = Qs_outside + Qt_plugged
            Qs_inside = (self.inside_friction_ratio
                         * self._compute_inside_skin_friction(V, notes))
            unplugged_details: dict = {}
            Qt_unplugged = toe(pile.tip_area, unplugged_details)
            W_p = self._plug_weight()
            Q_unplugged = Qs_outside + Qs_inside + Qt_unplugged - W_p
            open_pipe_info = {"Q_plugged": Q_plugged,
                              "Q_unplugged": Q_unplugged,
                              "Qs_inside": Qs_inside, "W_p": W_p,
                              "method": pw_method,
                              "predominantly_cohesive": reading}
            if pw_method == "tomlinson":
                # Eq. 7-90: Rn = 0.8 Ca As + 4.5 su Ap (Ap of the plugged
                # pile), Ca from Fig. 7-17 as p. 407 prints it whatever
                # adhesion_chart is (the shaft again on Fig. 7-17, its notes
                # in place of the first pass's); a cohesionless layer keeps
                # its own side resistance.
                if self.adhesion_chart != "fig_7_17":
                    again: List[str] = []
                    layer_results, _, _ = self._shaft(V, again, "fig_7_17")
                    notes[:shaft_notes] = again
                for lb in layer_results:
                    if lb["method"] == "tomlinson":
                        lb["skin_friction_kN"] *= 0.8
                        lb["unit_side_resistance_kPa"] *= 0.8
                        lb["eq_7_90_factor"] = 0.8
                total_Qs = sum(lb["skin_friction_kN"] for lb in layer_results)
                Qt = 4.5 * tip_layer.cohesion * pile.tip_area_plugged
                toe_details = {"s_u_kPa": tip_layer.cohesion,
                               "q_p_kPa": 4.5 * tip_layer.cohesion,
                               "governs": "Eq. 7-90",
                               "toe_area_m2": pile.tip_area_plugged}
                plug_case = "eq_7_90"
                open_pipe_info["Q_eq_7_90"] = total_Qs + Qt
            elif Q_plugged <= Q_unplugged:
                Qt, toe_details, plug_case = (Qt_plugged, plugged_details,
                                              "plugged")
            else:
                Qt, toe_details, plug_case = (Qt_unplugged,
                                              unplugged_details, "unplugged")
                total_Qs = Qs_outside + Qs_inside
                plug_weight = W_p
            if pile.width >= _EQ_7_90_MAX_OD:
                notes.append(
                    "GEC-12 Sec. 7.10.7 (p. 407): for moderate (24 to 36 "
                    "inch O.D.) and large (36+ inch) diameter open ended pipe "
                    "piles, plugging is less certain under static conditions "
                    "and should be carefully analyzed.")
        else:
            Qt = toe(self._toe_area(), toe_details)

        # The alpha method past the chart's curves, where the engineer's
        # highly overconsolidated line would change the answer and is not
        # given: refused, naming every such layer (GM-229).
        wanted = [lb.pop("line_needed") for lb in layer_results
                  if "line_needed" in lb]
        if wanted:
            raise JudgmentNeeded(self._line_needed(wanted))

        Q_ultimate = total_Qs + Qt - plug_weight
        Q_allowable = Q_ultimate / self.factor_of_safety
        self._width_notes(layer_results, notes)

        # Uplift: the OUTSIDE side resistance (GEC-12 Eq. 7-38 takes the
        # nominal shaft resistance; under Eq. 7-90 that is 0.8 Ca As), plus
        # optional pile self-weight.
        Q_uplift = None
        if self.include_uplift:
            Q_uplift = self.uplift_skin_fraction * (
                total_Qs if plug_case == "eq_7_90" else Qs_outside)
            if self.pile_weight is not None:
                Q_uplift += self.pile_weight

        return AxialPileResult(
            Q_ultimate=Q_ultimate,
            Q_skin=total_Qs,
            Q_tip=Qt,
            Q_allowable=Q_allowable,
            Q_uplift=Q_uplift,
            factor_of_safety=self.factor_of_safety,
            pile_length=self.pile_length,
            pile_name=pile.name,
            method=self.method,
            layer_breakdown=layer_results,
            sigma_v_tip=sigma_v_tip,
            warnings=notes,
            choices=self._choices(layer_results, shaft_layers, tip_layer,
                                  plug_case, sigma_v_tip, toe_details, V,
                                  open_pipe_info),
            plug_weight=plug_weight,
            toe_details=toe_details,
        )

    def _shaft(self, V: float, notes: List[str],
               chart: Optional[str] = None) -> Tuple[list, list, float]:
        """(layer rows, shaft layers, outside side resistance in kN) of the
        layers along the shaft, the alpha-method on ``chart`` (default
        ``adhesion_chart``)."""
        pile = self.pile
        tip_depth = self._tip_depth
        total_Qs = 0.0
        layer_results = []
        shaft_layers = []
        current_depth = 0.0

        for layer, stratum in zip(self.soil.layers, self._fig_7_18_strata()):
            layer_top = current_depth
            layer_bottom = current_depth + layer.thickness
            current_depth = layer_bottom

            if layer_top >= tip_depth:
                break

            # Portion of this layer that intersects the pile shaft. Layers
            # above the pile head (head_depth) contribute no skin friction;
            # with the default head_depth = 0 this clips at [0, pile_length].
            z_top = max(layer_top, self.head_depth)
            z_bottom = min(layer_bottom, tip_depth)
            if z_bottom - z_top <= 0:
                continue

            Qs_layer, sigma_v, method_used, info = self._layer_skin_friction(
                layer, layer_bottom, z_top, z_bottom, pile.perimeter, V,
                notes, stratum, chart)

            shaft_layers.append(layer)
            total_Qs += Qs_layer
            layer_results.append({
                "depth_top_m": round(z_top, 2),
                "depth_bottom_m": round(z_bottom, 2),
                "soil_type": layer.soil_type,
                "description": layer.description,
                "method": method_used,
                # Unrounded, so the layers sum to Q_s exactly (the
                # calculation tables, GM-39); every display formats them.
                "skin_friction_kN": Qs_layer,
                "sigma_v_kPa": sigma_v,
                # Q_s,i / (perimeter x segment length): the unit side
                # resistance averaged over the segment (exact for one
                # segment; across a GWT split it is the average).
                "unit_side_resistance_kPa": (
                    Qs_layer / (pile.perimeter * (z_bottom - z_top))
                    if pile.perimeter > 0 else 0.0),
                **info,
            })
        return layer_results, shaft_layers, total_Qs

    def _toe_Nt(self, tip_layer: AxialSoilLayer,
                notes: Optional[List[str]] = None) -> Tuple[float, str]:
        """(Nt, its source) of the toe layer for the beta method (GEC-12
        Step 5): the layer's own (local experience, Step 5a), else, in clay,
        ``cohesive_Nt`` (p. 256), else Table 7-9 by soil type with
        ``beta_coefficients`` 'table_7_9', else estimated from phi'."""
        if tip_layer.Nt is not None:
            return tip_layer.Nt, "given"
        soil = tip_layer.table_7_9_soil
        if self.beta_coefficients == "table_7_9" and soil != "clay":
            return (table_7_9(soil, self._beta_phi(tip_layer, toe=True),
                              notes)[1], f"Table 7-9 ({soil})")
        if tip_layer.soil_type != "cohesionless":
            return self.cohesive_Nt, "cohesive_Nt"
        return Nt_from_phi(tip_layer.toe_phi), "from phi'"

    def _beta_phi(self, layer: AxialSoilLayer, toe: bool = False) -> float:
        """phi' of a layer for the beta method: a cohesionless layer's own
        (its toe phi at the toe), a cohesive layer's friction_angle, else
        ``cohesive_phi``."""
        if layer.soil_type == "cohesionless":
            return layer.toe_phi if toe else layer.friction_angle
        return ((layer.toe_friction_angle if toe else None)
                or layer.friction_angle or self.cohesive_phi)

    def _fig_7_18_strata(self) -> List[Optional[Tuple[str, float, str]]]:
        """Each layer's GEC-12 Figure 7-18 stratigraphy case: None for a
        cohesionless layer, else (case, depth of the top of its clay
        stratum, "given" or "profile"). Contiguous cohesive layers are one
        clay stratum; a layer that gives a ``fig_7_18_case`` starts one at
        its top. A stratum's case, where no layer gives it, is the top
        graph's under a cohesionless layer and the bottom graph's from the
        ground surface (GEC-12 p. 246-247); the middle graph (soft clay over
        stiff clay) only where a layer gives it."""
        out: List[Optional[Tuple[str, float, str]]] = []
        top, current, above = 0.0, None, None
        for layer in self.soil.layers:
            if layer.soil_type != "cohesive":
                current = None
            elif layer.fig_7_18_case is not None:
                current = (layer.fig_7_18_case, top, "given")
            elif current is None:
                current = (SAND_OVER_STIFF_CLAY if above is not None
                           else STIFF_CLAY, top, "profile")
            out.append(current)
            above = layer
            top += layer.thickness
        return out

    def _toe_phi(self, tip_layer: AxialSoilLayer,
                 notes: List[str]) -> Tuple[float, Optional[dict]]:
        """(phi near the toe, the zone it was averaged over) for the
        Nordlund toe (Steps 7 and 9). "toe_layer": the toe layer's toe phi.
        "average_3b": the thickness-weighted average of the cohesionless
        layers' toe phi from the toe to 3b below (Step 7c's zone; the
        profile's last layer continues below it)."""
        if self.toe_phi_zone == "toe_layer":
            return tip_layer.toe_phi, None
        tip = self._tip_depth
        depth = 3.0 * self.pile.width
        bottom = tip + depth
        layers = self.soil.layers
        top, weighted, length = 0.0, 0.0, 0.0
        parts, clay = [], []
        for i, layer in enumerate(layers):
            lt, lb = top, top + layer.thickness
            top = lb
            if i == len(layers) - 1:
                lb = max(lb, bottom)
            a, b = max(lt, tip), min(lb, bottom)
            if b <= a:
                continue
            if layer.soil_type == "cohesionless":
                weighted += layer.toe_phi * (b - a)
                length += b - a
                parts.append({"depth_top_m": a, "depth_bottom_m": b,
                              "toe_phi_deg": layer.toe_phi})
            else:
                clay.append((a, b))
        if clay:
            where = ", ".join(f"{a:.2f}-{b:.2f} m" for a, b in clay)
            notes.append(
                f"The zone from the toe to 3b below ({tip:.2f}-{bottom:.2f} "
                f"m) takes in cohesive soil ({where}); the toe phi is "
                f"averaged over the cohesionless soil in it only. A weaker "
                f"layer below the toe lowers the toe resistance; check it.")
        if length <= 0:
            notes.append(
                "The zone from the toe to 3b below is all cohesive soil; the "
                "toe layer's own toe phi was used.")
            return tip_layer.toe_phi, None
        return weighted / length, {"from_m": tip, "to_m": bottom,
                                   "layers": parts}

    def _open_pipe_method(self, tip_layer, layer_results
                          ) -> Tuple[str, Optional[dict]]:
        """('tomlinson' (Eq. 7-90) or 'paikowsky_whitman' (Eqs. 7-88/89),
        the reading of "predominantly cohesive" when 'auto' made one)."""
        clay_toe = (tip_layer.soil_type == "cohesive"
                    and self.method != "beta")
        if self.open_pipe_method == "tomlinson":
            if not clay_toe:
                raise ValueError(
                    "open_pipe_method 'tomlinson' (GEC-12 Eq. 7-90) needs the "
                    "toe in a cohesive layer and the alpha-method (method "
                    "'auto')")
            return "tomlinson", None
        if (self.open_pipe_method == "paikowsky_whitman" or not clay_toe
                or self.pile.width >= _EQ_7_90_MAX_OD):
            return "paikowsky_whitman", None
        clay = sum(lb["depth_bottom_m"] - lb["depth_top_m"]
                   for lb in layer_results if lb["soil_type"] == "cohesive")
        reading = {"clay_m": clay, "length_m": self.pile_length,
                   "predominantly": clay > 0.5 * self.pile_length}
        return ("tomlinson" if reading["predominantly"]
                else "paikowsky_whitman"), reading

    def _plug_weight(self) -> float:
        """Wp of Eq. 7-89 (kN): the effective weight of the soil inside the
        pipe from the head to the toe."""
        inner_area = self.pile.tip_area_plugged - self.pile.tip_area
        stress = (self.soil.effective_stress_at_depth(self._tip_depth)
                  - self.soil.effective_stress_at_depth(self.head_depth))
        return max(inner_area, 0.0) * stress

    def _width_notes(self, layer_results, notes: List[str]) -> None:
        """GEC-12 p. 232 on piles wider than the methods' database."""
        if self.method == "beta":
            return
        width_in = self.pile.width / _INCH
        if self.pile.width > _DATABASE_MAX_WIDTH:
            notes.append(
                f"The pile is {width_in:.0f} inches wide: GEC-12 Sec. "
                "7.2.1.3 (p. 232) notes that piles larger than 18 inches were "
                "not in the correlation database for either the Nordlund "
                "Method or the alpha-method.")
        if (self.pile.width > _NORDLUND_OVERPREDICT_WIDTH
                and any(lb["method"] == "nordlund" for lb in layer_results)):
            notes.append(
                "GEC-12 Sec. 7.2.1.3.1 (p. 232): the Nordlund Method tends to "
                "over predict the nominal resistance for pile widths larger "
                "than 24 inches; alternative static analysis methods should "
                "be evaluated.")

    def _choices(self, layer_results, shaft_layers, tip_layer, plug_case,
                 sigma_v_tip, toe_details, V, open_pipe_info) -> List[dict]:
        """What this analysis chose (GM-115, GM-121): the side and toe
        resistance methods, the GEC-12 charts and limits each used and the
        judgments GEC-12 leaves to the engineer, with their basis."""
        from geotech_common.choices import MODULE, choice, how_given
        # Citations are to the printed GEC-12 (FHWA-NHI-16-009, 2016).
        out = []
        beta = self.method == "beta"
        # the beta method as the call named it, or as a calc package's call
        # says it is assumed (GM-180)
        method_how = (how_given(self.method, "auto", name="method")
                      if beta else None)
        auto_text = "Nordlund / alpha-method by soil type (method 'auto')"
        beta_text = ("the beta method in every layer, GEC-12 Sec. 7.2.1.3.4 "
                     "(method 'beta')")
        if beta:
            out.append(choice(
                "side resistance method", "beta method, every layer",
                method_how, "fs = beta sigma'v (GEC-12 Eq. 7-19); beta from "
                "local experience where a layer gives one (Step 2a), else as "
                "'beta and Nt (beta method)' says (Step 2b)", [auto_text]))
            c = self._beta_choice(layer_results, tip_layer, toe_details)
            if c:
                out.append(c)
        else:
            # Contiguous runs of one method, top to bottom.
            runs = []
            for lb in layer_results:
                name = _METHOD_TEXT.get(lb["method"], lb["method"])
                if runs and runs[-1][0] == name:
                    runs[-1][2] = lb["depth_bottom_m"]
                else:
                    runs.append([name, lb["depth_top_m"],
                                 lb["depth_bottom_m"]])
            out.append(choice(
                "side resistance method",
                "; ".join(f"{n} {a:g}-{b:g} m" for n, a, b in runs), MODULE,
                "'auto' by soil type: Nordlund in cohesionless soil (GEC-12 "
                "Sec. 7.2.1.3.1, Eq. 7-8: K_delta from Tables 7-6/7-7, C_F "
                "from Fig. 7-14, no limit on sigma'd or the unit side "
                "resistance), the alpha-method with Tomlinson's adhesion "
                "(Fig. 7-17) in cohesive soil (Sec. 7.2.1.3.2)",
                [beta_text]))
        if beta:
            toe, basis, how = ("Nt sigma'p (beta method)",
                               "GEC-12 Eq. 7-20; Nt from local experience "
                               "where a layer gives one (Step 5a), else as "
                               "'beta and Nt (beta method)' says, or "
                               "cohesive_Nt in clay; no limiting value "
                               "(p. 256)", method_how)
            alts = [auto_text]
        elif tip_layer.soil_type == "cohesionless":
            toe, basis, how = (
                "alpha_t N'q sigma'p, at most q_L",
                "cohesionless toe layer: GEC-12 Eq. 7-9 (Steps 7-9), alpha_t "
                "from Fig. 7-16(a) by phi and D/b (D/b below 20 or above 45 "
                "takes the nearest curve), N'q from Fig. 7-16(b), q_L from "
                "Fig. 7-15", MODULE)
            alts = ["Nt sigma'v (method 'beta')"]
        else:
            toe, basis, how = ("9 su", "cohesive toe layer: GEC-12 Eq. 7-11 "
                               "(Nc = 9)", MODULE)
            alts = ["Nt sigma'v (method 'beta')"]
        out.append(choice("toe resistance method", toe, how, basis, alts))

        if (not beta and tip_layer.soil_type == "cohesionless"
                and plug_case != "eq_7_90"):
            limit = self.toe_stress_limit
            if limit is None:
                chosen = f"not limited (sigma'v at the toe {sigma_v_tip:.1f} kPa)"
            else:
                chosen = (f"{min(sigma_v_tip, limit):.1f} kPa (limit "
                          f"{limit:.1f} kPa; sigma'v at the toe "
                          f"{sigma_v_tip:.1f} kPa)")
            out.append(choice(
                "sigma'p at the toe (Nordlund, Step 8)", chosen,
                how_given(limit, _TOE_STRESS_LIMIT, name="toe_stress_limit"),
                "Nordlund method, Step 8 (GEC-12 Sec. 7.2.1.3.1, p. 244): "
                "'The limiting value of sigma'p is 3 ksf' (144 kPa), as "
                "Nordlund recommended (p. 234); GEC-12's worked example (Vol "
                "3, Table D-6) did not apply it",
                ["sigma'p limited to 3 ksf (toe_stress_limit 143.6)"
                 if limit is None else "no limit (toe_stress_limit None)"]))
        if "toe_phi_deg" in toe_details and plug_case != "eq_7_90":
            phi = toe_details["toe_phi_deg"]
            zone = toe_details.get("toe_zone")
            if zone:
                chosen = (f"{phi:.1f} deg, averaged from the toe to 3b below "
                          f"({zone['from_m']:.2f}-{zone['to_m']:.2f} m): "
                          + ", ".join(f"{z['toe_phi_deg']:g} deg over "
                                      f"{z['depth_top_m']:.2f}-"
                                      f"{z['depth_bottom_m']:.2f} m"
                                      for z in zone["layers"]))
            else:
                chosen = (f"{phi:g} deg, the toe layer's "
                          + ("toe_friction_angle"
                             if tip_layer.toe_friction_angle is not None
                             else "friction_angle"))
            out.append(choice(
                "toe friction angle (Nordlund)", chosen,
                how_given(self.toe_phi_zone, "toe_layer",
                          name="toe_phi_zone"),
                "GEC-12 Steps 7a, 7b and 9 (pp. 244-245) enter Figs. 7-16 "
                "and 7-15 with the 'phi angle near the pile toe'; Step 7c: "
                "where phi is estimated from SPT data, average (N1)60 'over "
                "the zone from the pile toe to 3 diameters below the pile "
                "toe' and estimate phi from that; the module takes phi, so "
                "'average_3b' averages the cohesionless layers' toe phi over "
                "that zone (the same where phi is linear in N); GEC-12's "
                "worked examples (Vol 3, Tables D-6 and D-51, DrivenPiles) "
                "took the toe layer's phi",
                ["the toe layer's phi (toe_phi_zone 'toe_layer')",
                 "averaged from the toe to 3b below (toe_phi_zone "
                 "'average_3b')"]))

        sand = [l for l in shaft_layers if l.soil_type == "cohesionless"]
        if sand and not beta:
            area, what = self._displaced_area()
            alts = []
            basis = ("GEC-12 Step 2a: the volume of soil displaced per unit "
                     "length of pile; it enters delta/phi (Fig. 7-9) and "
                     "K_delta (Tables 7-6/7-7, log-linear in V, Step 3)")
            if self.pile.pile_type == "pipe_open":
                alts = ["the steel area, if the pipe drives unplugged "
                        "(displaced_volume; below Fig. 7-9 curve a, so "
                        "delta/phi is held at the curve's end)"]
                basis += ("; GEC-12 does not give V for an open pipe, and "
                          "its Nordlund database had no open end pipes "
                          "(p. 232)")
            out.append(choice(
                "displaced volume V (Nordlund)",
                f"{V:.3g} ft3/ft ({what})",
                how_given(self.displaced_volume, None,
                          name="displaced_volume")
                if self.displaced_volume is not None else MODULE,
                basis, alts))
            given = sorted({l.delta_phi_ratio for l in sand
                            if l.delta_phi_ratio is not None})
            from_chart = any(l.delta_phi_ratio is None for l in sand)
            parts = [f"{r:g} (given)" if from_chart else f"{r:g}"
                     for r in given]
            if from_chart:
                key, label = FIG_7_9_CURVE[self.pile.pile_type]
                ratio = delta_phi_ratio_fig_7_9(self.pile.pile_type, V, [])
                parts.append(f"{ratio:.2f} (Fig. 7-9 curve {key}, {label})")
            out.append(choice(
                "Nordlund delta/phi", ", ".join(parts),
                MODULE if from_chart else how_given(
                    given[0], None, within="layers"),
                "GEC-12 Step 2b: Fig. 7-9 with V for the pile type; a "
                "layer's delta_phi_ratio wins",
                ["delta/phi given for a layer (delta_phi_ratio)"]))

        clay = [lb for lb in layer_results if lb["method"] == "tomlinson"]
        if clay:
            out.extend(self._adhesion_choices(clay, plug_case == "eq_7_90"))
        if self.shrink_swell_depth is not None:
            left = sum(lb.get("shrink_swell_m", 0.0) for lb in layer_results)
            out.append(choice(
                "shrink-swell zone (cohesive soil)",
                f"no shaft resistance in cohesive soil above "
                f"{self.shrink_swell_depth:g} m ({left:.2f} m of the pile in "
                f"cohesive soil left out)",
                how_given(self.shrink_swell_depth, None,
                          name="shrink_swell_depth"),
                "GEC-12 p. 247 (alpha-method): 'In clays with large "
                "shrink-swell potential, static resistance calculations "
                "should ignore the shaft resistance from the adhesion in the "
                "shrink-swell zone'; the module leaves out the beta method's "
                "shaft resistance in cohesive soil there too, since the gap "
                "p. 247 describes is the same",
                ["no shrink-swell zone (shrink_swell_depth None)"]))

        if self.pile.pile_type == "h_pile":
            chosen = ("box area, depth x flange width"
                      if self.h_pile_toe_area == "box" else "steel area")
            out.append(choice(
                "toe area (H-pile)", chosen,
                how_given(self.h_pile_toe_area, "box",
                          name="h_pile_toe_area"),
                "GEC-12 Step 9 (p. 245): for steel H piles the toe area "
                "(steel only, or steel and enclosed soil) 'should be based on "
                "past experience and local correlations with static load "
                "test results'; Sec. 7.10.7 (p. 408) uses the box area in "
                "cohesionless and cohesive soils, the steel area on rock",
                ["box area (h_pile_toe_area 'box')",
                 "steel area (h_pile_toe_area 'steel')"]))

        if plug_case:
            info = open_pipe_info
            cases = {
                "plugged": "plugged (Eq. 7-88)",
                "unplugged": "unplugged (Eq. 7-89)",
                "eq_7_90": "Tomlinson's Eq. 7-90",
            }
            numbers = (f"plugged {info['Q_plugged']:.1f} kN, unplugged "
                       f"{info['Q_unplugged']:.1f} kN")
            if plug_case == "eq_7_90":
                basis = (
                    "GEC-12 Sec. 7.10.7 (p. 407): for open end pipe piles "
                    "less than 24 inches in diameter in predominantly "
                    "cohesive soils, Rn = 0.8 Ca As + 4.5 su Ap (Eq. 7-90) "
                    "should be used (taken here as the toe in clay and more "
                    "than half the embedded length in clay layers; Tomlinson "
                    "applies it when field measurements confirm a plug); a "
                    f"cohesionless layer keeps its side resistance; Ca from "
                    f"Fig. 7-17 as Eq. 7-90 prints it (p. 407), whatever "
                    f"adhesion chart is chosen for the side resistance; "
                    f"Eqs. 7-88/7-89 give {numbers}")
            else:
                basis = ("the lesser of Eqs. 7-88 and 7-89 governs "
                         "(Paikowsky and Whitman 1990, GEC-12 Sec. 7.10.7, "
                         f"p. 406): {numbers}")
            out.append(choice(
                "open-ended pipe", cases[plug_case],
                MODULE if self.open_pipe_method == "auto" else how_given(
                    self.open_pipe_method, "auto", name="open_pipe_method"),
                basis,
                [cases["plugged"] + " / " + cases["unplugged"]
                 + " (open_pipe_method 'paikowsky_whitman')",
                 cases["eq_7_90"] + " (open_pipe_method 'tomlinson')"]
                if plug_case == "eq_7_90" else
                [c for k, c in cases.items()
                 if k not in (plug_case, "eq_7_90")]
                + [cases["eq_7_90"] + " (open_pipe_method 'tomlinson')"]))
            reading = info.get("predominantly_cohesive")
            if reading:
                share = 100.0 * reading["clay_m"] / reading["length_m"]
                out.append(choice(
                    "predominantly cohesive soil (Eq. 7-90)",
                    f"{'yes' if reading['predominantly'] else 'no'}: "
                    f"{reading['clay_m']:.2f} of {reading['length_m']:.2f} m "
                    f"of embedded length in cohesive layers ({share:.0f} %)",
                    MODULE,
                    "GEC-12 Sec. 7.10.7 (p. 407) takes Eq. 7-90 for open end "
                    "pipe piles less than 24 inches in diameter 'in "
                    "predominantly cohesive soils' and gives no measure of "
                    "it; the module reads it as the toe in a cohesive layer "
                    "and more than half the embedded length in cohesive "
                    "layers",
                    ["Eq. 7-90 whatever the share (open_pipe_method "
                     "'tomlinson')",
                     "Eqs. 7-88/7-89 whatever the share (open_pipe_method "
                     "'paikowsky_whitman')"]))
            out.append(choice(
                "interior unit side resistance (open pipe)",
                f"{self.inside_friction_ratio:.3g} x the exterior "
                f"({info['Qs_inside']:.1f} kN in the unplugged case)",
                how_given(self.inside_friction_ratio, INSIDE_FRICTION_RATIO,
                          name="inside_friction_ratio"),
                "GEC-12 Sec. 7.10.7 (p. 407): 'typically on the order of 1/3 "
                "to 1/2 the exterior unit shaft resistance'; the module "
                "takes the low end",
                ["1/2 of the exterior (inside_friction_ratio 0.5)"]))
            out.append(choice(
                "soil plug weight Wp (open pipe)",
                f"{info['W_p']:.1f} kN, subtracted in the unplugged case",
                MODULE,
                "GEC-12 Eq. 7-89; the effective weight of the soil inside the "
                "pipe from the head to the toe (soil assumed to fill the "
                "pipe)"))
            if tip_layer.soil_type == "cohesionless":
                limit = self.open_pipe_toe_limit
                governed = toe_details.get("governs") == "open-end pipe limit"
                out.append(choice(
                    "unit toe resistance limit (open pipe)",
                    ("none" if limit is None else
                     f"{limit:,.0f} kPa (105 ksf)"
                     + ("; it governed" if governed else "")),
                    how_given(limit, OPEN_PIPE_TOE_LIMIT,
                              name="open_pipe_toe_limit"),
                    "Tomlinson (1994), GEC-12 Sec. 7.10.7 (p. 407): a "
                    "limiting unit toe resistance of 105 ksf for open end "
                    "pipe piles in cohesionless soils, regardless of pile "
                    "size or soil density, once q_L of Fig. 7-15 exceeds it",
                    ["no limit (open_pipe_toe_limit None)"]))

        if (beta and tip_layer.soil_type != "cohesionless"
                and toe_details.get("N_t_source") in ("given", "cohesive_Nt")):
            given = toe_details["N_t_source"] == "given"
            out.append(choice(
                "Nt of the cohesive toe layer (beta method)",
                f"{toe_details['N_t']:g}",
                how_given(toe_details["N_t"], None, within="layers")
                if given else how_given(self.cohesive_Nt, COHESIVE_NT,
                                        name="cohesive_Nt"),
                "GEC-12 p. 256: an Nt of 3 in clay gives about the toe "
                "resistance of an undrained (su) analysis; 'the use of a "
                "relatively low Nt coefficient in clays is recommended unless "
                "local correlations suggest higher values' (Table 7-9: clay "
                "Nt 3-30)", ["a layer's own Nt (local experience)"]))
        clay_no_phi = beta and any(
            l.soil_type != "cohesionless" and not l.friction_angle
            for l in list(shaft_layers))
        if clay_no_phi:
            out.append(choice(
                "phi' of cohesive layers given no friction angle (beta "
                "method)",
                f"{self.cohesive_phi:g} deg",
                how_given(self.cohesive_phi, 25.0, name="cohesive_phi"),
                "within the clay range 25-30 deg of GEC-12 Table 7-9"))
        if self.include_uplift:
            out.append(choice(
                "uplift capacity",
                f"{self.uplift_skin_fraction:g} x outside side resistance"
                + (" + pile weight" if self.pile_weight is not None else ""),
                how_given(self.uplift_skin_fraction, UPLIFT_SKIN_FRACTION,
                          name="uplift_skin_fraction"),
                "GEC-12 Sec. 7.2.3.1 (Eq. 7-38, pp. 288-289): the nominal "
                "shaft resistance with an uplift resistance factor (AASHTO "
                "C10.7.3.10 takes it as 80 % of the compression factor), so "
                "'the shaft resistance does not need to be further reduced "
                "for the uplift case'; side resistance inside an open pipe "
                "is left out",
                ["0.7 x, the customary uplift share in free-draining soils "
                 "De Nicola and Randolph (1993) report "
                 "(uplift_skin_fraction 0.7)"]))
        return out

    def _adhesion_choices(self, clay: List[dict],
                          eq_7_90: bool = False) -> List[dict]:
        """The alpha-method's choices over the clay layers (``clay``, their
        breakdown rows): the chart (GEC-12 p. 246), Figure 7-17's curves
        and limit, each layer's Figure 7-18 case, and su past a chart.
        Under Eq. 7-90 (``eq_7_90``) the side resistance is on Figure 7-17
        whatever the chart, so no Figure 7-18 case is read."""
        from geotech_common.choices import MODULE, choice, how_given
        out = []
        charts = {"fig_7_17": "Fig. 7-17 (Tomlinson 1979)",
                  "fig_7_18": "Fig. 7-18 (Tomlinson 1980) by stratigraphy "
                              "case",
                  "smaller": "the smaller Ca of Figs. 7-17 and 7-18 in each "
                             "layer"}
        chart = self.adhesion_chart
        chosen = charts[chart]
        used = sorted({lb.get("adhesion_chart") for lb in clay} - {None})
        if chart != "fig_7_17":
            chosen += f" (used: {', '.join(used)}"
            chosen += (", Eq. 7-90 takes Ca from Fig. 7-17, p. 407)"
                       if eq_7_90 else ")")
        out.append(choice(
            "adhesion chart (alpha-method)", chosen,
            how_given(chart, "fig_7_17", name="adhesion_chart"),
            "GEC-12 p. 246: 'It is recommended that Figure 7-17 generally be "
            "used for adhesion calculations, unless one of the special soil "
            "stratigraphy cases identified in Figure 7-18 is present at a "
            "site. In cases where either Figures 7-17 or 7-18 could be "
            "used, the inexperienced user should select and use the smaller "
            "value obtained from either figure'",
            [f"{text} (adhesion_chart '{key}')"
             for key, text in charts.items() if key != chart]))

        read_17 = [lb for lb in clay if "fig_7_17_source" in lb]
        if read_17:
            curve = self._adhesion_curve()
            chosen = f"Fig. 7-17, the curves for {CURVES[curve]}"
            if self.adhesion_curve == "auto":
                chosen += " (by pile type)"
            h_pile = self.pile.pile_type == "h_pile"
            if h_pile:
                chosen += ", on the box perimeter"
            out.append(choice(
                "adhesion (alpha-method)", chosen,
                how_given(self.adhesion_curve, "auto",
                          name="adhesion_curve"),
                "alpha method, Step 1 (GEC-12 p. 250): Ca from Fig. 7-17 "
                "(p. 248) by su, pile type and D/b, linear between the D = "
                "10b and 40b curves"
                + ("; an H-pile takes the corrugated pile curve times the "
                   "box area (p. 247); GEC-12's worked example (Vol 3, Table "
                   "D-100) used the smooth steel curves for its H-pile"
                   if h_pile else ""),
                [f"the curves for {CURVES[c]} (adhesion_curve '{c}')"
                 for c in CURVES if c != curve]))
            ratios = [lb["D_over_b"] for lb in read_17]
            out.append(choice(
                "D for Fig. 7-17 (alpha method)",
                "from the ground surface to the bottom of each clay layer, "
                "or to the toe if less (D/b "
                + (f"{min(ratios):.1f}" if len(set(ratios)) == 1 else
                   f"{min(ratios):.1f}-{max(ratios):.1f}") + ")",
                MODULE,
                "alpha method: p. 246 takes Fig. 7-17's embedded length D "
                "as 'the minimum value of the length from the ground surface "
                "to the bottom of the clay layer, or the length from the "
                "ground surface to the pile toe', as the figure's legend (p. "
                "248) has it; Step 1's 'embedded pile length in clay, D' "
                "(p. 250) is read the same way"))
            above = [lb["alpha"] for lb in read_17
                     if lb.get("adhesion_chart") == "Fig. 7-17"
                     and lb.get("fig_7_17_source") == "Fig. 7-17"
                     and lb.get("alpha", 0.0) > 1.0 + 1e-9]
            out.append(choice(
                "Ca above su (alpha method)",
                "Fig. 7-17 as drawn, no limit on Ca"
                + (f"; Ca above su in {len(above)} layer(s), alpha up to "
                   f"{max(above):.2f}" if above else ""),
                MODULE,
                "alpha method, Steps 1 and 2 (p. 250): Ca from Fig. 7-17 and "
                "fs = Ca (Eq. 7-10), with no limit; the figure's D = 40b "
                "concrete, timber and corrugated curve passes above Ca = su "
                "below su of about 1.5 ksf (alpha about 1.1 at su = 1.0 "
                "ksf). The limit alpha <= 1.0 is API RP 2A's, part of the "
                "API method (Sec. 7.2.1.3.3; p. 252 recommends alpha 1.0 for "
                "unconsolidated clays), and applies where API alpha is "
                "taken"))

        read_18 = [lb for lb in clay if "fig_7_18_case" in lb]
        if chart != "fig_7_17" and not eq_7_90:
            strata = {}
            for lb in clay:
                case = lb.get("fig_7_18_case", "none")
                text = (FIG_7_18_CASES[case] if case != "none"
                        else "none (Fig. 7-17)")
                where = f"{lb['depth_top_m']:g}-{lb['depth_bottom_m']:g} m"
                if case != "none":
                    where += f", D = {lb['D_over_b_fig_7_18']:.1f}b"
                strata.setdefault(text, []).append(where)
            given = [l.fig_7_18_case for l in self.soil.layers
                     if l.fig_7_18_case is not None]
            out.append(choice(
                "Fig. 7-18 stratigraphy case",
                "; ".join(f"{t}: {', '.join(w)}" for t, w in strata.items()),
                how_given(given[0], None, within="layers") if given
                else MODULE,
                "GEC-12 pp. 246-247: the top graph for piles driven through "
                "a sand or sandy gravel layer into an underlying stiff clay, "
                "the middle graph through a soft clay layer into an "
                "underlying stiff clay, the bottom graph in stiff clays "
                "without any different overlying strata; D is the "
                "penetration into the stiff clay (the figure's sketches), "
                "taken from the top of the clay stratum (the ground surface "
                "in the bottom graph) to the bottom of the layer or to the "
                "toe if less; the module takes contiguous cohesive layers as "
                "one stratum, the top graph under a cohesionless layer and "
                "the bottom graph from the ground surface, and the middle "
                "graph only where a layer gives it (GEC-12 gives no su that "
                "separates a soft clay from a stiff one); a layer's "
                "fig_7_18_case wins",
                ["a layer's fig_7_18_case: 'sand_over_stiff_clay', "
                 "'soft_clay_over_stiff_clay', 'stiff_clay' or 'none'"]))
        if any("Ca_fig_7_17_kPa" in lb for lb in clay):
            parts = [f"{lb['depth_top_m']:g}-{lb['depth_bottom_m']:g} m: "
                     f"Fig. 7-17 {lb['Ca_fig_7_17_kPa']:.1f} kPa, Fig. 7-18 "
                     f"{lb['Ca_fig_7_18_kPa']:.1f} kPa"
                     for lb in clay if "Ca_fig_7_17_kPa" in lb]
            out.append(choice(
                "Ca of the two charts (alpha-method)", "; ".join(parts),
                MODULE, "GEC-12 p. 246: the smaller value governs"))

        modes = {
            API_IF_HIGHLY_OC: "API RP 2A's alpha where the clay is highly "
                              "overconsolidated, Ca held at the curve's end "
                              "elsewhere",
            "chart_end": "Ca held at the curve's end in every such layer",
            "api": "API RP 2A's alpha (Eqs. 7-13/7-14) in every such layer"}
        for which, rows, text in (
                ("Fig. 7-17", read_17, "fig_7_17_source"),
                ("Fig. 7-18", read_18, "fig_7_18_source")):
            if not any(lb.get(text) in BEYOND_SOURCES for lb in rows):
                continue
            basis = ("alpha method, Step 1 (p. 250) and p. 247: where su "
                     "exceeds the charts in highly overconsolidated clays, "
                     "alpha by API RP 2A (Sec. 7.2.1.3.3, Eqs. 7-13 to "
                     "7-15, p. 251); GEC-12's worked example (Vol 3, Table "
                     "D-100) held Ca at the chart's end")
            if which == "Fig. 7-17":
                basis += ("; the concrete, timber and corrugated curves end "
                          "at su = 3.45 ksf and the smooth steel curves at "
                          "3.05 ksf, and API alpha there can make Ca jump")
            else:
                basis += ("; Fig. 7-18's curves end at su = 5.0 ksf (top "
                          "and bottom graphs) and about 4.6-4.8 ksf (middle "
                          "graph)")
            mode = self.adhesion_beyond_chart
            chosen = modes[mode]
            if mode == API_IF_HIGHLY_OC \
                    and self.highly_overconsolidated_su_ratio is None:
                # the run was not refused: the line changes nothing here
                chosen = ("Ca held at the curve's end: no highly "
                          "overconsolidated line was given, and either side "
                          "of one gives the same answer here (the other "
                          "chart governs, or the layer is in the "
                          "shrink-swell zone)")
            out.append(choice(
                f"su past {which} (alpha method)", chosen,
                how_given(mode, API_IF_HIGHLY_OC,
                          name="adhesion_beyond_chart"),
                basis,
                [f"{t} (adhesion_beyond_chart '{key}')"
                 for key, t in modes.items() if key != mode]))
        sides = [lb for lb in clay if "highly_overconsolidated" in lb]
        if sides:
            out.append(choice(
                "highly overconsolidated clay (alpha method)",
                f"su / sigma'vo at or above "
                f"{self.highly_overconsolidated_su_ratio:g}: "
                + "; ".join(self._sides(sides)),
                how_given(self.highly_overconsolidated_su_ratio, None,
                          name="highly_overconsolidated_su_ratio"),
                "alpha method: p. 247 and Step 1 (p. 250) recommend API RP "
                "2A's alpha for highly overconsolidated clays whose su "
                "exceeds the charts, and GEC-12 gives no su / sigma'vo at "
                "which a clay becomes highly overconsolidated: the line is "
                "the engineer's judgment; sigma'vo at the middle of each "
                "layer's length along the pile. The 1.0 in Eqs. 7-13 and "
                "7-14 (p. 251) chooses between API's two expressions for "
                "alpha; it does not define a highly overconsolidated clay"))
        return out

    @staticmethod
    def _sides(rows: List[dict]) -> List[str]:
        """Each run of contiguous clay layers past the chart on one side of
        the highly overconsolidated line: depths, su / sigma'vo and what the
        alpha method took there."""
        runs: List[list] = []
        for lb in rows:
            high = lb["highly_overconsolidated"]
            ratio = lb["su_over_sigma_v"]
            if runs and runs[-1][0] == high \
                    and abs(runs[-1][2] - lb["depth_top_m"]) < 1e-6:
                runs[-1][2] = lb["depth_bottom_m"]
                runs[-1][3].append(ratio)
            else:
                runs.append([high, lb["depth_top_m"], lb["depth_bottom_m"],
                             [ratio]])
        out = []
        for high, top, bottom, ratios in runs:
            lo, hi = min(ratios), max(ratios)
            span = f"{lo:.2f}" if f"{lo:.2f}" == f"{hi:.2f}" else \
                f"{lo:.2f}-{hi:.2f}"
            out.append(f"{top:g}-{bottom:g} m, su / sigma'vo {span}: "
                       + ("highly overconsolidated, API alpha" if high else
                          "below the line, Ca held at the curve's end"))
        return out

    def _beta_choice(self, layer_results, tip_layer,
                     toe_details) -> Optional[dict]:
        """The beta method's beta and Nt where a layer gives none (Steps 2b
        and 5b), or None when every layer gave its own."""
        from geotech_common.choices import choice, how_given
        sources = {lb.get("beta_source") for lb in layer_results}
        nt = toe_details.get("N_t_source")
        if sources <= {"given"} and nt in ("given", "cohesive_Nt"):
            return None
        if self.beta_coefficients == "table_7_9":
            soils = []
            for s in list(sources) + [nt]:
                if s and s.startswith("Table 7-9 ("):
                    soil = s[len("Table 7-9 ("):-1]
                    if soil not in soils:
                        soils.append(soil)
            soils.sort(key=list(TABLE_7_9).index)
            chosen = ("Table 7-9 by soil type, linear in phi' across the "
                      "type's phi' range: " + "; ".join(
                          f"{s} beta {TABLE_7_9[s][1][0]:g}-"
                          f"{TABLE_7_9[s][1][1]:g}, Nt {TABLE_7_9[s][2][0]:g}-"
                          f"{TABLE_7_9[s][2][1]:g} over phi' "
                          f"{TABLE_7_9[s][0][0]:g}-{TABLE_7_9[s][0][1]:g} deg"
                          for s in soils))
        else:
            chosen = ("from phi': beta = (1 - sin phi') tan phi' (Fellenius "
                      "1991), Nt within Table 7-9's ranges by phi'")
        return choice(
            "beta and Nt (beta method)", chosen,
            how_given(self.beta_coefficients, "from_phi",
                      name="beta_coefficients"),
            "GEC-12 Steps 2b and 5b (pp. 257-258): in the absence of local "
            "experience, use Table 7-9 (after Fellenius 2014) to estimate "
            "beta and Nt from phi'; the table gives ranges by soil type "
            "(clay, silt, sand, gravel), which 'table_7_9' reads linearly in "
            "phi' (a layer's soil_class, else clay or sand); (1 - sin phi') "
            "tan phi' is about 0.27-0.30 for phi' 25-40 deg: inside Table "
            "7-9's clay and silt ranges, the low end of its sand range "
            "(0.30-0.90) and below its gravel range (0.35-0.80); a clay toe "
            "takes cohesive_Nt either way (p. 256)",
            ["from phi' (beta_coefficients 'from_phi')"
             if self.beta_coefficients == "table_7_9" else
             "Table 7-9 by soil type (beta_coefficients 'table_7_9')"])

    def _layer_skin_friction(self, layer: AxialSoilLayer, layer_bottom: float,
                             z_top: float, z_bottom: float, perimeter: float,
                             V: float, notes: Optional[List[str]],
                             stratum: Optional[Tuple[str, float, str]] = None,
                             chart: Optional[str] = None
                             ) -> Tuple[float, float, str, dict]:
        """Side resistance of one layer segment of the pile.

        Integrates with the midpoint rule, splitting the segment at the
        groundwater table when it falls strictly inside: sigma_v' is
        piecewise linear with a kink at the GWT, so midpoint x thickness
        on each side of the kink is exact for the Nordlund and beta methods
        (unit side resistance linear in sigma_v'). The segment is also split
        at ``shrink_swell_depth``; cohesive soil above it gives no shaft
        resistance (GEC-12 p. 247). ``stratum`` is the layer's Figure 7-18
        case (:meth:`_fig_7_18_strata`); ``chart`` the alpha-method's chart
        (default ``adhesion_chart``).

        Returns
        -------
        (Qs, sigma_v_mid, method_used, info) : tuple
            Segment side resistance (kN), effective stress at the full
            segment midpoint (kPa, for reporting), the method label and the
            method's factors (delta/phi, K_delta, C_F; D/b, the adhesion
            source and the Figure 7-18 case; beta).
        """
        cuts = [z_top, z_bottom]
        for z in (self.soil.gwt_depth, self.shrink_swell_depth):
            if z is not None and z_top < z < z_bottom:
                cuts.append(z)
        cuts.sort()
        zone = (self.shrink_swell_depth
                if layer.soil_type == "cohesive" else None)
        sigma_v_mid = self.soil.effective_stress_at_depth((z_top + z_bottom) / 2)

        Qs = 0.0
        counted = 0.0
        method_used = ""
        info: dict = {}
        line_needed = None
        for za, zb in zip(cuts[:-1], cuts[1:]):
            seg_thickness = zb - za
            sigma_v = self.soil.effective_stress_at_depth((za + zb) / 2)
            # The shrink-swell zone: the segment's resistance is worked out
            # (so the layer reports its chart readings) and left out.
            share = 0.0 if zone is not None and zb <= zone else 1.0
            counted += share * seg_thickness

            if self.method == "beta":
                # Cohesive layers: an explicitly supplied per-layer
                # friction_angle WINS; the global cohesive_phi is only the
                # fallback for layers without one.
                if layer.beta is not None:
                    beta, source = layer.beta, "given"
                elif self.beta_coefficients == "table_7_9":
                    soil = layer.table_7_9_soil
                    beta = table_7_9(soil, self._beta_phi(layer), notes)[0]
                    source = f"Table 7-9 ({soil})"
                else:
                    beta = beta_from_phi(self._beta_phi(layer))
                    source = "from phi'"
                Qs += share * beta * sigma_v * perimeter * seg_thickness
                method_used = "beta"
                info = {"beta": beta, "beta_source": source}
            elif layer.soil_type == "cohesionless":
                ratio = layer.delta_phi_ratio
                if ratio is None:
                    ratio = delta_phi_ratio_fig_7_9(self.pile.pile_type, V,
                                                    notes)
                f = nordlund_side_factors(layer.friction_angle, ratio, V,
                                          notes)
                Qs += f["fs_ratio"] * sigma_v * perimeter * seg_thickness
                method_used = "nordlund"
                info = {"delta_phi_ratio": ratio, "K_delta": f["K_delta"],
                        "C_F": f["C_F"]}
            else:
                ca, info = self._adhesion(layer, layer_bottom, sigma_v,
                                          stratum, notes, chart,
                                          sigma_v_layer=sigma_v_mid,
                                          where=(z_top, z_bottom),
                                          counts=share > 0)
                # any segment the highly overconsolidated line would change
                line_needed = info.pop("line_needed", line_needed)
                Qs += share * ca * perimeter * seg_thickness
                method_used = "tomlinson"

        if line_needed is not None:
            info["line_needed"] = line_needed
        if zone is not None and counted < z_bottom - z_top:
            info["shrink_swell_m"] = (z_bottom - z_top) - counted
        if method_used == "tomlinson" and z_bottom > z_top and perimeter > 0:
            # alpha = Ca / su over the length that counts (all of it, but
            # for a shrink-swell zone)
            info["alpha"] = (Qs / (perimeter * counted * layer.cohesion)
                             if counted > 0 else ca / layer.cohesion)
        return Qs, sigma_v_mid, method_used, info

    def _beyond_chart(self, su: float, sigma_v: float
                      ) -> Tuple[Optional[str], dict]:
        """(how the alpha method takes Ca where this clay layer's su is past
        the chart's curves, "api" or "chart_end", or None where the
        default needs ``highly_overconsolidated_su_ratio`` and it is not
        given; the layer's su / sigma'vo and the side of that line it is
        on, for the default)."""
        if self.adhesion_beyond_chart != API_IF_HIGHLY_OC:
            return self.adhesion_beyond_chart, {}
        ratio = su / sigma_v if sigma_v > 0 else float("inf")
        reading = {"su_over_sigma_v": ratio}
        line = self.highly_overconsolidated_su_ratio
        if line is None:
            return None, reading
        reading["highly_overconsolidated"] = ratio >= line
        return ("api" if ratio >= line else "chart_end"), reading

    def _adhesion(self, layer: AxialSoilLayer, layer_bottom: float,
                  sigma_v: float, stratum: Optional[Tuple[str, float, str]],
                  notes: Optional[List[str]], chart: Optional[str] = None,
                  sigma_v_layer: Optional[float] = None,
                  where: Tuple[float, float] = (0.0, 0.0),
                  counts: bool = True
                  ) -> Tuple[float, dict]:
        """(Ca in kPa, its readings) of a clay layer by ``chart`` (default
        ``adhesion_chart``):
        Figure 7-17, Figure 7-18 for the layer's stratigraphy case, or the
        smaller of the two (GEC-12 p. 246). ``sigma_v`` is the segment's
        sigma'vo (API alpha's), ``sigma_v_layer`` the layer's at the middle
        of its length along the pile (``where``, m), which decides whether
        it is highly overconsolidated. Where the layer is past the chart's
        curves, no line is given and the line would change the Ca that
        counts (``counts``: the segment is below any shrink-swell zone),
        the readings carry ``line_needed`` and :meth:`compute` refuses the
        run, naming every such layer. The notes of a chart not used are
        left out."""
        bottom = min(layer_bottom, self._tip_depth)
        # Fig. 7-17's D: ground surface to the bottom of the clay layer, or
        # to the toe if less (GEC-12 p. 246).
        D_over_b = bottom / self.pile.width
        info: dict = {"D_over_b": D_over_b}
        case = stratum[0] if stratum else "none"
        chart = chart or self.adhesion_chart
        beyond, reading = self._beyond_chart(
            layer.cohesion, sigma_v if sigma_v_layer is None
            else sigma_v_layer)
        # Fig. 7-18's D: the penetration into the clay stratum, from its
        # top (the ground surface in the bottom graph) to the bottom of
        # the layer, or to the toe if less.
        D18 = ((bottom - stratum[1]) / self.pile.width
               if case != "none" else None)

        def read(mode: str) -> Tuple[dict, dict]:
            """(chart -> (Ca, source, notes), chart -> curves' end in ksf)
            with ``mode`` past the curves."""
            readings, ends = {}, {}
            if chart in ("fig_7_17", "smaller") or case == "none":
                own: List[str] = []
                curve = self._adhesion_curve()
                readings["Fig. 7-17"] = (*adhesion(
                    layer.cohesion, D_over_b, curve, sigma_v=sigma_v,
                    beyond_chart=mode, notes=own), own)
                ends["Fig. 7-17"] = CHART_END_KSF[curve]
            if chart in ("fig_7_18", "smaller") and case != "none":
                own = []
                readings["Fig. 7-18"] = (*adhesion_fig_7_18(
                    layer.cohesion, D18, case, sigma_v=sigma_v,
                    beyond_chart=mode, notes=own), own)
                ends["Fig. 7-18"] = fig_7_18_range_ksf(case, D18)[1]
            return readings, ends

        readings, ends = read(beyond or "chart_end")
        if "Fig. 7-17" in readings:
            info["fig_7_17_source"] = readings["Fig. 7-17"][1]
        if "Fig. 7-18" in readings:
            info.update({"fig_7_18_case": case, "D_over_b_fig_7_18": D18,
                         "fig_7_18_source": readings["Fig. 7-18"][1]})
        past = [k for k, r in readings.items() if r[1] in BEYOND_SOURCES]
        if past and beyond is None and counts:
            # Would the line change the answer? Not where the other chart
            # governs either way (adhesion_chart 'smaller').
            api = min(r[0] for r in read("api")[0].values())
            if abs(api - min(r[0] for r in readings.values())) > 1e-9:
                info["line_needed"] = {
                    "where": where, "su": layer.cohesion, "chart": past[0],
                    "end_ksf": ends[past[0]],
                    "ratio": reading["su_over_sigma_v"]}
        if len(readings) == 2:
            info["Ca_fig_7_17_kPa"] = readings["Fig. 7-17"][0]
            info["Ca_fig_7_18_kPa"] = readings["Fig. 7-18"][0]
        used = min(readings, key=lambda k: (readings[k][0], k))
        ca, source, own = readings[used]
        if past and "highly_overconsolidated" in reading:
            info.update(reading)
            own = own + [self._line_note()]
        if notes is not None:
            notes.extend(n for n in own if n not in notes)
        else:
            for n in own:
                warnings.warn(n, stacklevel=3)
        info["adhesion_source"] = source
        info["adhesion_chart"] = used
        return ca, info

    @staticmethod
    def _line_needed(rows: List[dict]) -> str:
        """Why an analysis is refused for want of
        ``highly_overconsolidated_su_ratio``: every clay layer past the
        chart's curves whose answer the line would change (``rows``, their
        ``line_needed`` readings), and what to give."""
        layers = "; ".join(
            f"{r['where'][0]:.2f}-{r['where'][1]:.2f} m: su = "
            f"{r['su'] / KSF:.2f} ksf ({r['su']:.1f} kPa), where "
            f"{r['chart']}'s curves end at {r['end_ksf']:.2f} ksf; su / "
            f"sigma'vo = {r['ratio']:.2f}" for r in rows)
        return (
            f"alpha method (GEC-12 Sec. 7.2.1.3.2), Step 1 (p. 250): these "
            f"clay layers have su past the end of the chart's curves: "
            f"{layers}. For highly overconsolidated "
            f"clays whose su exceeds the charts, Step 1 and p. 247 recommend "
            f"the adhesion factor of the API method (API RP 2A, Sec. "
            f"7.2.1.3.3, Eqs. 7-13 to 7-15, p. 251). GEC-12 prints no su / "
            f"sigma'vo (or other number) at which a clay becomes highly "
            f"overconsolidated: that is the engineer's judgment. Give "
            f"highly_overconsolidated_su_ratio (API alpha in a layer whose "
            f"su / sigma'vo, at the middle of its length along the pile, is "
            f"at or above it; Ca held at the curve's end below it), or "
            f"adhesion_beyond_chart 'chart_end' (Ca held at the curve's end "
            f"in every such layer) or 'api' (API alpha in every such "
            f"layer).")

    def _line_note(self) -> str:
        """The judgment note of a run that took the line given."""
        return (
            f"alpha method (GEC-12 p. 247; Step 1, p. 250): past the chart's "
            f"curves, API RP 2A's alpha was taken in clay layers with su / "
            f"sigma'vo at or above {self.highly_overconsolidated_su_ratio:g} "
            f"(highly overconsolidated, by the line given) and Ca was held "
            f"at the curve's end below it. Where a clay becomes highly "
            f"overconsolidated is the engineer's judgment; GEC-12 gives no "
            f"value.")

    def _compute_inside_skin_friction(self, V: float,
                                      notes: Optional[List[str]] = None
                                      ) -> float:
        """Side resistance (kN) inside an open-ended pipe at the EXTERIOR
        unit side resistance, on the inner perimeter over the embedded
        length; :meth:`compute` scales it by ``inside_friction_ratio``."""
        inner_perim = self.pile.inner_perimeter
        if inner_perim is None or inner_perim <= 0:
            return 0.0

        Qs_inside = 0.0
        current_depth = 0.0
        tip_depth = self._tip_depth

        for layer, stratum in zip(self.soil.layers, self._fig_7_18_strata()):
            layer_top = current_depth
            layer_bottom = current_depth + layer.thickness
            current_depth = layer_bottom

            if layer_top >= tip_depth:
                break

            z_top = max(layer_top, self.head_depth)
            z_bottom = min(layer_bottom, tip_depth)
            if z_bottom - z_top <= 0:
                continue

            Qs_layer, _, _, _ = self._layer_skin_friction(
                layer, layer_bottom, z_top, z_bottom, inner_perim, V, notes,
                stratum)
            Qs_inside += Qs_layer

        return Qs_inside

    def capacity_vs_depth(self, depth_min: float = 3.0,
                          depth_max: Optional[float] = None,
                          n_points: int = 20) -> List[dict]:
        """Compute capacity at multiple pile lengths for optimization.

        Parameters
        ----------
        depth_min : float, optional
            Minimum pile length to evaluate (m). Default 3.0.
        depth_max : float, optional
            Maximum pile length. Default: soil profile depth.
        n_points : int, optional
            Number of evaluation points. Default 20.

        Returns
        -------
        list of dict
            Each dict has 'depth_m', 'Q_ultimate_kN', 'Q_skin_kN', 'Q_tip_kN'.
        """
        if depth_max is None:
            depth_max = min(self.pile_length, self.soil.total_thickness)

        depths = np.linspace(depth_min, depth_max, n_points)
        results = []
        # the deepest refusal names every layer down to it (GM-229)
        wanted: Optional[JudgmentNeeded] = None

        # Use a copy per trial depth rather than temporarily mutating
        # self.pile_length (re-entrant / thread-safe).
        for d in depths:
            try:
                trial = replace(self, pile_length=float(d))
                r = trial.compute()
                results.append({
                    "depth_m": round(d, 2),
                    "Q_ultimate_kN": round(r.Q_ultimate, 1),
                    "Q_skin_kN": round(r.Q_skin, 1),
                    "Q_tip_kN": round(r.Q_tip, 1),
                    **({"warnings": r.warnings} if r.warnings else {}),
                })
            except JudgmentNeeded as e:
                wanted = e
            except Exception:
                pass
        if wanted is not None:
            raise wanted
        return results
