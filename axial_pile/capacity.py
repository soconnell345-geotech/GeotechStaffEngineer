"""
Combined axial pile capacity analysis.

Auto-selects the appropriate method for each soil layer:
- Nordlund for cohesionless layers (sand/gravel)
- the alpha-method (Tomlinson's adhesion, GEC-12 Figure 7-17) for cohesive
  layers (clay/silt)

Also supports the beta (effective stress) method for any soil type.

Each method follows the GEC-12 step-by-step procedure, with its limits:
sigma'p at the toe limited to 3 ksf for the Nordlund toe (Step 8); an
open-end pipe pile as the lesser of the plugged and unplugged cases (Eqs.
7-88/7-89), with the interior side resistance a fraction of the exterior,
the soil plug weight subtracted and Tomlinson's 105 ksf limiting unit toe
resistance in cohesionless soil, or Tomlinson's Eq. 7-90 for a pipe under
24 inches in predominantly cohesive soil (Sec. 7.10.7); uplift as the whole
shaft resistance (Sec. 7.2.3.1). Where GEC-12 leaves a judgment (a range or
"local experience"), it is an input with a cited default and the result's
``choices`` say what was used.

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
    AT_MOST_SU, CURVES, adhesion, curve_for_pile, end_bearing_cohesive,
)
from axial_pile.nordlund import (
    FIG_7_9_CURVE, KSF, SIGMA_P_LIMIT, delta_phi_ratio_fig_7_9,
    displaced_volume, end_bearing_cohesionless, nordlund_side_factors,
)
from axial_pile.beta_method import beta_from_phi, end_bearing_beta, Nt_from_phi
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
_BEYOND_CHART = ("api", "chart_end")
_H_TOE_AREAS = ("box", "steel")


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
        Where su exceeds Figure 7-17: "chart_end" (default; Ca held at the
        curve's end, as GEC-12's worked example, Vol 3 Table D-100, did) or
        "api" (alpha from API RP 2A, which GEC-12 p. 247 recommends for
        highly overconsolidated clays). Within the chart Ca is at most su.
    displaced_volume : float or None, optional
        Displaced volume V per unit length (m3/m) for the Nordlund charts.
        Default None: the steel area of an H-pile, the gross area of other
        piles (an open pipe included).

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
    adhesion_beyond_chart: str = "chart_end"
    displaced_volume: Optional[float] = None

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
                              ("h_pile_toe_area", _H_TOE_AREAS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {list(allowed)}, "
                                 f"got '{getattr(self, name)}'")
        if not 0.0 <= self.inside_friction_ratio <= 1.0:
            raise ValueError("inside_friction_ratio must be between 0 and 1, "
                             f"got {self.inside_friction_ratio}")
        for name in ("toe_stress_limit", "open_pipe_toe_limit",
                     "displaced_volume"):
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
        total_Qs = 0.0
        layer_results = []
        shaft_layers = []
        current_depth = 0.0
        tip_depth = self._tip_depth

        area_V, _ = self._displaced_area()
        V = displaced_volume(area_V)

        for layer in self.soil.layers:
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
                notes)

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

        # Outside side resistance (before any inside/plug friction)
        Qs_outside = total_Qs

        tip_layer = self.soil.layer_at_depth(tip_depth - 0.01)
        sigma_v_tip = self.soil.effective_stress_at_depth(tip_depth)
        beta = self.method == "beta"
        open_pipe = self._open_pipe
        q_cap = (self.open_pipe_toe_limit
                 if open_pipe and tip_layer.soil_type == "cohesionless"
                 else None)

        def toe(area: float, details: dict) -> float:
            """Toe resistance (kN) on ``area`` by the toe layer's method."""
            if beta:
                Nt = self._toe_Nt(tip_layer)
                q_p = Nt * sigma_v_tip
                governs = "Eq. 7-20"
                if q_cap is not None and q_cap < q_p:
                    q_p, governs = q_cap, "open-end pipe limit"
                details.update({"N_t": Nt, "sigma_v_tip_kPa": sigma_v_tip,
                                "q_p_kPa": q_p, "governs": governs,
                                "toe_area_m2": area})
                return q_p * area
            if tip_layer.soil_type == "cohesionless":
                return end_bearing_cohesionless(
                    tip_layer.toe_phi, sigma_v_tip, area, self.pile_length,
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
            pw_method = self._open_pipe_method(tip_layer, layer_results)
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
                              "method": pw_method}
            if pw_method == "tomlinson":
                # Eq. 7-90: Rn = 0.8 Ca As + 4.5 su Ap (Ap of the plugged
                # pile); a cohesionless layer keeps its own side resistance.
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

    def _toe_Nt(self, tip_layer: AxialSoilLayer) -> float:
        """Nt of the toe layer for the beta method (GEC-12 Step 5): the
        layer's own (local experience), else ``cohesive_Nt`` in clay, else
        estimated from phi'."""
        if tip_layer.Nt is not None:
            return tip_layer.Nt
        if tip_layer.soil_type != "cohesionless":
            return self.cohesive_Nt
        return Nt_from_phi(tip_layer.toe_phi)

    def _open_pipe_method(self, tip_layer, layer_results) -> str:
        """'tomlinson' (Eq. 7-90) or 'paikowsky_whitman' (Eqs. 7-88/89)."""
        clay_toe = (tip_layer.soil_type == "cohesive"
                    and self.method != "beta")
        if self.open_pipe_method == "tomlinson":
            if not clay_toe:
                raise ValueError(
                    "open_pipe_method 'tomlinson' (GEC-12 Eq. 7-90) needs the "
                    "toe in a cohesive layer and the alpha-method (method "
                    "'auto')")
            return "tomlinson"
        if self.open_pipe_method == "paikowsky_whitman" or not clay_toe:
            return "paikowsky_whitman"
        clay = sum(lb["depth_bottom_m"] - lb["depth_top_m"]
                   for lb in layer_results if lb["soil_type"] == "cohesive")
        if (self.pile.width < _EQ_7_90_MAX_OD
                and clay > 0.5 * self.pile_length):
            return "tomlinson"
        return "paikowsky_whitman"

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
        from geotech_common.choices import (
            GIVEN, MODULE, choice, how_given,
        )
        # Citations are to the printed GEC-12 (FHWA-NHI-16-009, 2016).
        out = []
        beta = self.method == "beta"
        auto_text = "Nordlund / alpha-method by soil type (method 'auto')"
        beta_text = ("the beta method in every layer, GEC-12 Sec. 7.2.1.3.4 "
                     "(method 'beta')")
        if beta:
            out.append(choice(
                "side resistance method", "beta method, every layer",
                GIVEN, "fs = beta sigma'v (GEC-12 Eq. 7-19); beta from local "
                "experience where a layer gives one (Step 2a), else from "
                "phi' as (1 - sin phi') tan phi' (Fellenius 1991), about "
                "0.27-0.30 for phi' 25-40 deg: inside Table 7-9's clay and "
                "silt ranges, the low end of its sand range (0.30-0.90) and "
                "below its gravel range (0.35-0.80)", [auto_text]))
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
                               "where a layer gives one (Step 5a), else from "
                               "phi' within Table 7-9's ranges, or "
                               "cohesive_Nt in clay; no limiting value "
                               "(p. 256)", GIVEN)
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
                "sigma'p at the toe", chosen,
                how_given(limit, _TOE_STRESS_LIMIT, name="toe_stress_limit"),
                "GEC-12 Sec. 7.2.1.3.1 Step 8 (p. 244): 'The limiting value "
                "of sigma'p is 3 ksf' (144 kPa), Nordlund's recommendation "
                "(p. 234); GEC-12's worked example (Vol 3, Table D-6, "
                "DrivenPiles) did not apply it",
                ["sigma'p limited to 3 ksf (toe_stress_limit 143.6)"
                 if limit is None else "no limit (toe_stress_limit None)"]))

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
                GIVEN if self.displaced_volume is not None else MODULE,
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
                MODULE if from_chart else GIVEN,
                "GEC-12 Step 2b: Fig. 7-9 with V for the pile type; a "
                "layer's delta_phi_ratio wins",
                ["delta/phi given for a layer (delta_phi_ratio)"]))

        clay = [lb for lb in layer_results if lb["method"] == "tomlinson"]
        if clay:
            curve = self._adhesion_curve()
            chosen = f"Fig. 7-17, {CURVES[curve]} curves"
            if self.adhesion_curve == "auto":
                chosen += " (by pile type)"
            if self.pile.pile_type == "h_pile":
                chosen += ", on the box perimeter"
            out.append(choice(
                "adhesion (alpha-method)", chosen,
                how_given(self.adhesion_curve, "auto",
                          name="adhesion_curve"),
                "GEC-12 Step 1 (p. 250): Ca from Fig. 7-17 by su, pile type "
                "and D/b, D from the ground surface to the bottom of the "
                "clay layer or to the toe if less (p. 246), linear between "
                "the D = 10b and 40b curves; H-piles: the corrugated pile "
                "curve times the box area (p. 247); GEC-12's worked example "
                "(Vol 3, Table D-100, DrivenPiles) used the smooth steel "
                "curves for an H-pile",
                [f"{CURVES[c]} curves (adhesion_curve '{c}')"
                 for c in CURVES if c != curve]
                + ["Fig. 7-18's alpha for the three stratigraphy cases (sand "
                   "or sandy gravel over stiff clay, soft clay over stiff "
                   "clay, stiff clay alone; p. 246-247), not computed by the "
                   "module: where either figure could be used, p. 246 "
                   "advises the inexperienced user to take the smaller "
                   "value"]))
            capped = sum(lb.get("adhesion_source") == AT_MOST_SU
                         for lb in clay)
            out.append(choice(
                "adhesion limit (alpha-method)",
                "Ca at most su (alpha <= 1)"
                + (f"; it governed in {capped} layer(s)" if capped else ""),
                MODULE,
                "API RP 2A takes alpha at most 1.0 (GEC-12 p. 252: 'An alpha "
                "value of 1.0 is recommended for unconsolidated clays'), and "
                "p. 247 takes su on an H-pile's soil-to-soil faces; Fig. "
                "7-17 as drawn passes above Ca = su at low su (the D = 40b "
                "concrete, timber and corrugated curve up to about 1.5 ksf)",
                ["uncapped, Fig. 7-17 as drawn (not offered)"]))
            beyond = sorted({lb.get("adhesion_source") for lb in clay}
                            - {"Fig. 7-17", AT_MOST_SU, None})
            if beyond:
                chosen = ("alpha from API RP 2A (Eqs. 7-13/7-14)"
                          if self.adhesion_beyond_chart == "api"
                          else "Ca held at the curve's end")
                out.append(choice(
                    "su beyond Fig. 7-17", chosen,
                    how_given(self.adhesion_beyond_chart, "chart_end",
                              name="adhesion_beyond_chart"),
                    "GEC-12's worked example (Vol 3, Table D-100) held Ca at "
                    "the chart's end; p. 247 (and Step 1) recommends alpha by "
                    "API RP 2A (Sec. 7.2.1.3.3) for highly overconsolidated "
                    "clays whose su exceeds the charts; at the chart's end "
                    "Ca jumps (the concrete, timber and corrugated curves "
                    "end at su = 3.45 ksf with Ca 0.83-1.37 ksf for D = "
                    "10b-40b; API alpha there at sigma'v 4 ksf gives 1.86 "
                    "ksf)",
                    ["Ca held at the curve's end (adhesion_beyond_chart "
                     "'chart_end')"
                     if self.adhesion_beyond_chart == "api" else
                     "API alpha (adhesion_beyond_chart 'api')"]))

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
                    f"cohesionless layer keeps its side resistance; "
                    f"Eqs. 7-88/7-89 give {numbers}")
            else:
                basis = ("the lesser of Eqs. 7-88 and 7-89 governs "
                         "(Paikowsky and Whitman 1990, GEC-12 Sec. 7.10.7, "
                         f"p. 406): {numbers}")
            out.append(choice(
                "open-ended pipe", cases[plug_case],
                MODULE if self.open_pipe_method == "auto" else GIVEN, basis,
                [cases["plugged"] + " / " + cases["unplugged"]
                 + " (open_pipe_method 'paikowsky_whitman')",
                 cases["eq_7_90"] + " (open_pipe_method 'tomlinson')"]
                if plug_case == "eq_7_90" else
                [c for k, c in cases.items()
                 if k not in (plug_case, "eq_7_90")]
                + [cases["eq_7_90"] + " (open_pipe_method 'tomlinson')"]))
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

        if beta and tip_layer.soil_type != "cohesionless":
            given = tip_layer.Nt is not None
            out.append(choice(
                "Nt of the cohesive toe layer (beta method)",
                f"{self._toe_Nt(tip_layer):g}",
                GIVEN if given else how_given(self.cohesive_Nt, COHESIVE_NT,
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

    def _layer_skin_friction(self, layer: AxialSoilLayer, layer_bottom: float,
                             z_top: float, z_bottom: float, perimeter: float,
                             V: float, notes: Optional[List[str]]
                             ) -> Tuple[float, float, str, dict]:
        """Side resistance of one layer segment of the pile.

        Integrates with the midpoint rule, splitting the segment at the
        groundwater table when it falls strictly inside: sigma_v' is
        piecewise linear with a kink at the GWT, so midpoint x thickness
        on each side of the kink is exact for the Nordlund and beta methods
        (unit side resistance linear in sigma_v').

        Returns
        -------
        (Qs, sigma_v_mid, method_used, info) : tuple
            Segment side resistance (kN), effective stress at the full
            segment midpoint (kPa, for reporting), the method label and the
            method's factors (delta/phi, K_delta, C_F; D/b and the adhesion
            source; beta).
        """
        gwt = self.soil.gwt_depth
        if gwt is not None and z_top < gwt < z_bottom:
            cuts = [z_top, gwt, z_bottom]
        else:
            cuts = [z_top, z_bottom]

        Qs = 0.0
        method_used = ""
        info: dict = {}
        for za, zb in zip(cuts[:-1], cuts[1:]):
            seg_thickness = zb - za
            sigma_v = self.soil.effective_stress_at_depth((za + zb) / 2)

            if self.method == "beta":
                # Cohesive layers: an explicitly supplied per-layer
                # friction_angle WINS; the global cohesive_phi is only the
                # fallback for layers without one.
                if layer.beta is not None:
                    beta = layer.beta
                else:
                    phi = (layer.friction_angle
                           if layer.soil_type == "cohesionless"
                           else (layer.friction_angle or self.cohesive_phi))
                    beta = beta_from_phi(phi)
                Qs += beta * sigma_v * perimeter * seg_thickness
                method_used = "beta"
                info = {"beta": beta}
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
                # Fig. 7-17's D: ground surface to the bottom of the clay
                # layer, or to the toe if less (GEC-12 p. 246).
                D_over_b = (min(layer_bottom, self._tip_depth)
                            / self.pile.width)
                ca, source = adhesion(
                    layer.cohesion, D_over_b, self._adhesion_curve(),
                    sigma_v=sigma_v, beyond_chart=self.adhesion_beyond_chart,
                    notes=notes)
                Qs += ca * perimeter * seg_thickness
                method_used = "tomlinson"
                info = {"D_over_b": D_over_b, "adhesion_source": source}

        if method_used == "tomlinson" and z_bottom > z_top and perimeter > 0:
            info["alpha"] = Qs / (perimeter * (z_bottom - z_top)
                                  * layer.cohesion)
        sigma_v_mid = self.soil.effective_stress_at_depth((z_top + z_bottom) / 2)
        return Qs, sigma_v_mid, method_used, info

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

        for layer in self.soil.layers:
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
                layer, layer_bottom, z_top, z_bottom, inner_perim, V, notes)
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
            except Exception:
                pass
        return results
