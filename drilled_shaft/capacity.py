"""
Combined drilled shaft capacity analysis.

Auto-selects the appropriate method for each soil layer:
- Alpha method for cohesive layers (clay/silt)
- Beta method for cohesionless layers (sand/gravel)
- Rock socket method for rock layers

Applies GEC-10 exclusion zones:
- Top 1.5m excluded from side resistance
- Bottom 1 diameter excluded from side resistance (cohesive only)
- Permanently cased zone excluded

All units are SI: kN, m, kPa.

References:
    FHWA GEC-10 (FHWA-NHI-18-024), Chapter 10, Sec. 10.3.5
"""

import math
import warnings
from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np

from drilled_shaft.shaft import DrillShaft
from drilled_shaft.soil_profile import ShaftSoilProfile, ShaftSoilLayer
from drilled_shaft.side_resistance import (
    alpha_cohesive, side_resistance_cohesive,
    beta_cohesionless, side_resistance_cohesionless,
    side_resistance_rock, PA,
    alpha_cohesive_rational, su_to_ciuc,
    beta_gec10, phi_prime_from_N1_60, preconsolidation_stress_gec10,
    SIGMA_P_CORRELATIONS,
)
from drilled_shaft.end_bearing import (
    end_bearing_cohesive, end_bearing_cohesionless, end_bearing_rock,
)
from drilled_shaft.results import DrillShaftResult


def require_tip_N60(tip_layer: ShaftSoilLayer, depth: float) -> float:
    """The N60 a cohesionless tip layer's end bearing uses (GM-142).

    End bearing in cohesionless soil is q_b = 57.5 N60 kPa (N60 capped
    at 50; GEC-10 (2018) Eq. 10-14, O'Neill & Reese 1999), so it needs the tip layer's
    N60. Until GM-142 a layer with no N60 took N60 = 15 without saying
    so; now it is refused.
    """
    if tip_layer.N60 > 0:
        return tip_layer.N60
    name = f" ({tip_layer.description})" if tip_layer.description else ""
    raise ValueError(
        f"the shaft tip at {depth:g} m is in a cohesionless layer{name} "
        "with no N60: end bearing q_b = 57.5 N60 kPa (N60 capped at 50; "
        "GEC-10 (2018) Eq. 10-14, O'Neill & Reese 1999) needs that layer's N60 (the SPT "
        "blow count corrected to 60 % energy at the tip)")


def skipped_depths_note(skipped, n_points: int) -> str:
    """One sentence naming the trial lengths a capacity-against-length
    sweep left out and why (GM-203)."""
    shown = "; ".join(f"L = {m['depth_m']:g} m ({m['reason']})"
                      for m in skipped[:3])
    more = (f"; and {len(skipped) - 3} more" if len(skipped) > 3 else "")
    return (f"The analysis refused {len(skipped)} of the {n_points} trial "
            f"shaft lengths, which the capacity against length leaves out: "
            f"{shown}{more}")


def side_n60_choice(spans) -> dict:
    """The choice made for cohesionless side layers given no N60 under the
    depth beta (GM-148): beta unreduced, as for N60 >= 15. ``spans`` are
    the layers' (top, bottom) depths in m."""
    from geotech_common.choices import MODULE, choice
    where = ", ".join(f"{a:g}-{b:g} m" for a, b in spans)
    return choice(
        "beta of cohesionless side layers given no N60",
        f"depth beta unreduced, as for N60 >= 15 ({where})", MODULE,
        basis=("FHWA NHI-06-089 Eq. 9-42 (p. 9-132) prints beta = 1.5 - "
               "0.135 sqrt(z, ft), 0.25 <= beta <= 1.2, with no N60 term; "
               "the reduction beta x N60/15 for N60 < 15 is O'Neill & Reese "
               "(1999), applied here only where the layer's N60 is given. "
               "GEC-10 (2018) prints the rational beta instead (Sec. "
               "10.3.5.1, Eqs. 10-6 to 10-13, pp. 10-10 to 10-12)."),
        alternatives=["give each layer's N60 (beta x N60/15 when N60 < 15)",
                      "beta_method 'rational', the default (GEC-10 Sec. "
                      "10.3.5.1, Eq. 10-13; needs each layer's N60 or "
                      "OCR)"])


#: Side resistance left out over the top of the shaft (m) in cohesive soil
#: and rock when top_exclusion is not given: GEC-10's 'top 5 ft' (p.
#: 10-17), printed for cohesive soils. Cohesionless soil then takes side
#: resistance from the ground surface: Sec. 10.3.5.1 prints no such zone,
#: and Examples 10-1 and 10-5 take the sand's from the surface.
TOP_EXCLUSION = 1.5

#: GEC-10 (FHWA-NHI-18-024) on the top of the shaft, as printed.
TOP_EXCLUSION_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) p. 10-17, in Sec. 10.3.5.2, Cohesive Soils: "
    "'It is common design practice to neglect side resistance over the top "
    "5 ft of a drilled shaft, or to the depth of seasonal moisture "
    "change', for wetting and drying, freezing and thawing, 'gapping' "
    "caused by cyclic lateral loading, or any process near the ground "
    "surface that softens the soil or ends its contact with the shaft; "
    "'This recommendation is subject to modification based on local "
    "experience and judgment of the designer.' Sec. 10.3.5.1, "
    "Cohesionless Soils, prints no such zone, and Examples 10-1 and 10-5 "
    "(pp. 10-12 to 10-14, 10-29) take the sand's side resistance from the "
    "ground surface. With top_exclusion left out, the top 1.5 m carries "
    "none in cohesive soil and rock and cohesionless soil carries side "
    "resistance from the surface; a top_exclusion given applies in every "
    "layer. The permanent casing, where deeper, takes its place")

#: The depth (m) at whose sigma'v GEC-10 limits beta near the surface:
#: 'the value corresponding to a depth of 7.5 ft' (pp. 10-11, 10-12).
SHALLOW_BETA_DEPTH = 7.5 * 0.3048

#: The depth beta's limit on f_s (kPa).
DEPTH_BETA_FS_MAX = 200.0

#: FHWA NHI-06-089's 4 ksf limit on the depth beta's f_s, in kPa.
NHI_FS_MAX = 4000 * 0.04788

#: GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.1, pp. 10-10 to 10-12, as printed.
GEC10_BETA_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.1, Step 5 of the design for axial "
    "load (Sec. 10.3), pp. 10-10 to 10-12: R_SN = pi B dz f_SN with f_SN = "
    "beta sigma'v (Eqs. 10-5, 10-7), sigma'v the 'average vertical "
    "effective stress over the depth interval' dz, taken here over each "
    "layer (Table 10-1, p. 10-6; Example 10-1 takes it at mid-layer). "
    "beta = K tan(delta) (Eq. 10-6) with delta = phi'; K = Ko, 'a rational "
    "first-order approximation' (K/Ko = 1, p. 10-11); Ko = (1 - sin phi') "
    "OCR^(sin phi') <= Kp = tan^2(45 + phi'/2) (Eqs. 10-8, 10-10; Mayne "
    "and Kulhawy 1982); OCR = sigma'p / sigma'v (Eq. 10-9) with sigma'p "
    "from N60 by Eq. 10-11 or 10-12; together Eq. 10-13. 'The value of "
    "beta at shallow depths should be limited to the value corresponding "
    "to a depth of 7.5 ft' (pp. 10-11, 10-12). phi' is the layer's, or "
    "27.5 + 9.2 log (N1)60 where (N1)60 is given (Appendix A, Eq. A-2, p. "
    "A-8). GEC-10 prints no upper limit on f_SN in cohesionless soil. "
    "Resistance factor 0.55 (p. 10-12)")

#: The depth beta, as its sources print it.
DEPTH_BETA_BASIS = (
    "O'Neill and Reese (1999) as printed in FHWA NHI-06-089 Eqs. 9-41 to "
    "9-43 (pp. 9-132, 9-133): beta = 1.5 - 0.135 sqrt(z, ft) = 1.5 - 0.245 "
    "sqrt(z, m), 0.25 <= beta <= 1.2, z the depth to the layer's centre; "
    "f_s = beta sigma'v, limited here to 200 kPa (NHI-06-089 prints 4 ksf, "
    "191 kPa). Where a layer's N60 is under 15, beta x N60/15, from "
    "O'Neill and Reese (1999), which NHI-06-089 does not print. GEC-10 "
    "(2018) does not print this method: its Sec. 10.3.5.1 gives Eq. 10-13")

#: GEC-10 (FHWA-NHI-18-024) pp. 10-11 and 10-12 on K/Ko, as printed.
K_OVER_KO_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) p. 10-11: 'A rational first-order "
    "approximation is that K/Ko = 1, assuming there is no stress change "
    "induced by construction'; p. 10-12: valid 'for dry, slurry (wet-hole), "
    "and casing methods of construction with minimal sidewall disturbance, "
    "proper handling of slurry and casing, and prompt placement of "
    "concrete'; 'when these aspects of construction quality are not "
    "controlled properly, the coefficient K can be reduced to 2/3 of its "
    "initial in-situ value (Ko), or lower if soil caving is allowed to "
    "occur. Judgment and understanding of the influence of construction "
    "activities are therefore needed'")

#: GEC-10 (FHWA-NHI-18-024) p. 10-11 on sigma'p from N60, as printed.
SIGMA_P_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) p. 10-11: Eq. 10-11, sigma'p/pa = 0.47 "
    "(N60)^m, 'where m = 0.6 for clean quartzitic sands and m = 0.8 for "
    "silty sands to sandy silts (e.g., Piedmont residual soils)' (Mayne "
    "2007); Eq. 10-12, sigma'p/pa = 0.15 N60, which 'Kulhawy and Chen "
    "(2007) suggest ... provides a good fit for gravelly soils'; "
    "'sigma'p is estimated by Equation 10-11 for sandy soils and Equation "
    "10-12 for gravelly soils'. The soil's description decides which; "
    "Example 10-1's silty sands take m = 0.8")


def _side_values(Qs_sand: float, Q_ultimate: float) -> str:
    """A run's side resistance in cohesionless soil and its ultimate
    capacity, in words, for a choice's alternatives (GM-241)."""
    return (f"Q_s in cohesionless soil {Qs_sand:,.0f} kN, Q_ult "
            f"{Q_ultimate:,.0f} kN")


#: GEC-10 (FHWA-NHI-18-024) p. 8-15, as printed.
FRACTURED_ROCK_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) p. 8-15: the amount of rock to be removed "
    "should be defined, 'including depth of rock sockets as well as the "
    "decomposed or fractured rock that may be removed but not considered "
    "part of the socket for design purposes'. No rock side resistance is "
    "taken over it; the soil above keeps its own, and the base is the "
    "rock at the tip")


#: The rock side resistance coefficient C the module takes when none is
#: given: GEC-10 Eq. 10-22's 0.65 alpha_E with alpha_E = 1.0.
ROCK_SIDE_C = 0.65

#: GEC-10 (FHWA-NHI-18-024) pp. 10-20 to 10-22, as printed (GM-221).
ROCK_SIDE_C_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.3: f_SN = C pa sqrt(qu / pa), C "
    "'a regression coefficient used to analyze load test results' (Eq. "
    "10-21, p. 10-20). In summary, 'for routine design of rock sockets, "
    "Equation 10-21 with C = 1.0 is recommended' (p. 10-21; Example 10-4, "
    "p. 10-28). "
    "Where caving or instability of the socket is anticipated, Eq. 10-22 "
    "(O'Neill and Reese 1999): f_SN = 0.65 alpha_E pa sqrt(qu / pa), "
    "alpha_E from Table 10-3 by RQD and joint condition. For roughened "
    "sockets a mean C = 1.9 (Kulhawy and Prakoso 2007, p. 10-22); 'load "
    "tests or local experience' should verify C greater than 1.0. The "
    "resistance factor is 0.55 with either equation (p. 10-21)")

#: The values of C GEC-10 names, each with where it applies.
_ROCK_SIDE_C_NAMED = {
    0.65: "Eq. 10-22 (where caving is anticipated) with alpha_E = 1.0",
    1.0: "Eq. 10-21, a 'normal' socket, recommended for routine design "
         "(p. 10-21)",
    1.9: "roughened sockets, verified by load tests or local experience "
         "(p. 10-22)",
}


def rock_side_choice(C: float) -> dict:
    """The choice made for the rock side resistance coefficient C (GM-221):
    the module's 0.65 is GEC-10 Eq. 10-22 with alpha_E = 1.0, where GEC-10
    recommends Eq. 10-21 with C = 1.0 for a normal socket."""
    from geotech_common.choices import choice, how_given
    named = _ROCK_SIDE_C_NAMED.get(C)
    alternatives = [f"{c!r}: {why}" for c, why in _ROCK_SIDE_C_NAMED.items()
                    if c != C and c != ROCK_SIDE_C]
    alternatives.append("0.65 alpha_E: Eq. 10-22 with alpha_E from Table "
                        "10-3 by RQD and joint condition, where caving is "
                        "anticipated (p. 10-21)")
    return choice("rock side resistance coefficient C",
                  f"{float(C)!r}: {named}" if named else f"{C:g}",
                  how_given(C, ROCK_SIDE_C, name="rock_side_C"),
                  basis=ROCK_SIDE_C_BASIS, alternatives=alternatives)


def top_of_rock(soil: ShaftSoilProfile) -> Optional[float]:
    """Depth (m) of the top of the profile's first rock layer; None with
    no rock layer."""
    depth = 0.0
    for layer in soil.layers:
        if layer.soil_type == "rock":
            return depth
        depth += layer.thickness
    return None


def fractured_rock_zone(shaft: DrillShaft, soil: ShaftSoilProfile):
    """(top, bottom) depths (m) of the rock not considered part of the
    socket (GEC-10 p. 8-15; ``shaft.fractured_rock_thickness`` below the
    top of rock), or None when it is 0. Refused with no rock layer."""
    t = shaft.fractured_rock_thickness
    if not t:
        return None
    z = top_of_rock(soil)
    if z is None:
        raise ValueError(
            f"fractured_rock_thickness = {t:g} m is rock below the top of "
            "rock that is not part of the socket (GEC-10 p. 8-15), but the "
            "profile has no rock layer (soil_type 'rock' with qu)")
    return z, z + t


def fractured_rock_choice(shaft: DrillShaft, soil: ShaftSoilProfile) -> dict:
    """The choice made for the top of the rock (GM-220): how much of it is
    not considered part of the socket."""
    from geotech_common.choices import choice, how_given
    t = shaft.fractured_rock_thickness
    how = how_given(t, 0.0, name="fractured_rock_thickness")
    if t:
        z0, z1 = fractured_rock_zone(shaft, soil)
        L = shaft.length
        # where the tip is against the zone: below it the input changes
        # nothing; inside it the base is on that rock (GM-220 review)
        where, base_alt = "", []
        if z0 >= L:
            where = f"; below the shaft tip ({L:g} m), so it changes nothing"
        elif L < z1:
            where = (f"; the tip ({L:g} m) is within it: the base takes "
                     "the tip rock layer's qu and RQD as given")
            base_alt = ["the fractured rock as a rock layer of its own, "
                        "with its qu and RQD, for the base on it"]
        return choice(
            "rock not considered part of the socket",
            f"{t:g} m below the top of rock ({z0:g}-{z1:g} m){where}", how,
            basis=FRACTURED_ROCK_BASIS,
            alternatives=[
                "fractured_rock_thickness 0: all rock below the top 1.5 m "
                "and the casing is socket",
                "casing_depth to the top of competent rock: no side "
                "resistance, soil or rock, above it (GEC-10 Example 10-4, "
                "p. 10-27)", *base_alt])
    return choice(
        "rock not considered part of the socket", "none", how,
        basis=("all rock in the shaft below the top 1.5 m and the casing "
               "is socket. " + FRACTURED_ROCK_BASIS.split(": ", 1)[0]
               + " allows decomposed or fractured rock 'that may be "
               "removed but not considered part of the socket for design "
               "purposes'"),
        alternatives=["fractured_rock_thickness: that thickness below the "
                      "top of rock gives no side resistance"])


@dataclass
class DrillShaftAnalysis:
    """Drilled shaft axial capacity analysis.

    Parameters
    ----------
    shaft : DrillShaft
        Shaft geometry.
    soil : ShaftSoilProfile
        Layered soil profile.
    factor_of_safety : float, optional
        Factor of safety. Default 2.5.
    beta_method : str, optional
        Cohesionless side-resistance method: "rational" (default, GEC-10
        (2018) Sec. 10.3.5.1, Eq. 10-13: beta = Ko tan(phi'), Ko from OCR =
        sigma'p/sigma'v, sigma'p from each layer's N60 (or its OCR), on the
        layer's average sigma'v; see :data:`GEC10_BETA_BASIS`) or "depth"
        (O'Neill & Reese 1999, beta = 1.5 - 0.135*sqrt(z, ft) = 1.5 -
        0.245*sqrt(z, m), 0.25 to 1.2; FHWA NHI-06-089 Eqs. 9-41 to 9-43),
        the default until GM-241.
    K_over_Ko : float, optional
        K/Ko in the GEC-10 beta: 1 (default), 'a rational first-order
        approximation' (GEC-10 p. 10-11); 2/3 or lower where construction
        quality is not controlled (p. 10-12). 0 < K/Ko <= 1.
    top_exclusion : float, optional
        Side resistance left out over the top of the shaft (m), in every
        layer. Left out (None): 1.5 m in cohesive soil and rock, GEC-10's
        'top 5 ft' (p. 10-17, printed for cohesive soils), and none in
        cohesionless soil (Sec. 10.3.5.1 prints none; Examples 10-1 and
        10-5 take the sand's side resistance from the ground surface). The
        permanent casing, where deeper, takes its place.
    alpha_method : str, optional
        Cohesive side-resistance method: "aashto" (default, piecewise 0.55) or
        "rational" (GEC-10 Chen-2011 alpha = 0.30+0.17/(su_CIUC/pa) with the
        UU/UC->CIUC transform). Default "aashto" preserves prior results.
    su_test_type : str, optional
        For alpha_method="rational": what lab test the layer ``cu`` represents,
        for the CIUC transform — "ciuc" (default, no transform), "uc", or "uu".
    pa : float, optional
        Atmospheric pressure (kPa) for the rational chains. Default 101.325.
    rock_side_C : float, optional
        C in the rock side resistance f_s = C pa sqrt(qu / pa) (GEC-10 Eq.
        10-21). Default 0.65, Eq. 10-22 with alpha_E = 1.0; GEC-10
        recommends 1.0 for a normal socket (p. 10-21). See
        :data:`ROCK_SIDE_C_BASIS` (GM-221).

    Examples
    --------
    >>> shaft = DrillShaft(diameter=1.0, length=20.0)
    >>> soil = ShaftSoilProfile(layers=[
    ...     ShaftSoilLayer(5, 'cohesionless', 18.0, phi=32, N60=20),
    ...     ShaftSoilLayer(10, 'cohesive', 17.0, cu=75),
    ...     ShaftSoilLayer(5, 'rock', 22.0, qu=5000),
    ... ])
    >>> analysis = DrillShaftAnalysis(shaft=shaft, soil=soil)
    >>> result = analysis.compute()
    """
    shaft: DrillShaft = None
    soil: ShaftSoilProfile = None
    factor_of_safety: float = 2.5
    beta_method: str = "rational"
    alpha_method: str = "aashto"
    su_test_type: str = "ciuc"
    pa: float = PA
    rock_side_C: float = ROCK_SIDE_C
    K_over_Ko: float = 1.0
    top_exclusion: Optional[float] = None

    def __post_init__(self):
        if self.shaft is None:
            raise ValueError("Shaft must be provided")
        if not self.rock_side_C > 0:
            raise ValueError(
                f"rock_side_C must be positive, got {self.rock_side_C}")
        if not 0 < self.K_over_Ko <= 1:
            raise ValueError(
                f"K_over_Ko must be more than 0 and at most 1, got "
                f"{self.K_over_Ko}: GEC-10 takes K/Ko = 1 (p. 10-11), and 2/3 "
                "or lower where construction quality is not controlled "
                "(p. 10-12)")
        if self.top_exclusion is not None and not self.top_exclusion >= 0:
            raise ValueError(
                f"top_exclusion must be 0 or more (m), got "
                f"{self.top_exclusion}")
        if self.soil is None:
            raise ValueError("Soil profile must be provided")
        if self.beta_method not in ("depth", "rational"):
            raise ValueError(
                f"beta_method must be 'depth' or 'rational', got '{self.beta_method}'"
            )
        if self.alpha_method not in ("aashto", "rational"):
            raise ValueError(
                f"alpha_method must be 'aashto' or 'rational', got '{self.alpha_method}'"
            )
        if self.su_test_type not in ("ciuc", "uc", "uu"):
            raise ValueError(
                f"su_test_type must be 'ciuc', 'uc', or 'uu', got '{self.su_test_type}'"
            )
        if self.shaft.length > self.soil.total_thickness:
            warnings.warn(
                f"Shaft length ({self.shaft.length}m) exceeds soil profile "
                f"({self.soil.total_thickness}m); using full profile depth"
            )

    def _gec10_beta(self, layer, sigma_v: float, where: str) -> dict:
        """GEC-10 (2018) Sec. 10.3.5.1 beta of a cohesionless layer, each
        step's value (Eqs. 10-8 to 10-13). ``sigma_v`` is the layer's
        average sigma'v (Eq. 10-5); ``where`` names the layer in a refusal.

        phi' is the layer's, or from (N1)60 where it is given (Appendix A
        Eq. A-2); OCR is the layer's, or sigma'p (Eq. 10-11 or 10-12, from
        N60) over sigma_v_ref or the layer's sigma'v, that sigma'v not less
        than the sigma'v at 7.5 ft (pp. 10-11, 10-12); delta = phi'.
        """
        if layer.N1_60 > 0:
            steps = {"phi_deg": phi_prime_from_N1_60(layer.N1_60),
                     "phi_from": "N1_60"}
        else:
            steps = {"phi_deg": layer.phi, "phi_from": "phi"}
        if layer.OCR > 0:
            OCR = layer.OCR
            steps["OCR_from"] = "OCR"
        else:
            if layer.N60 <= 0:
                raise ValueError(
                    "beta_method 'rational' (GEC-10 Eq. 10-13) needs N60 "
                    "(for sigma'p, Eq. 10-11 or 10-12) or a direct OCR on "
                    f"each cohesionless layer; the layer at {where} has "
                    "neither. Give its N60 or OCR, or name beta_method "
                    "'depth' (O'Neill and Reese 1999)")
            sigma_p = preconsolidation_stress_gec10(
                layer.N60, layer.sigma_p_correlation, self.pa)
            sv = layer.sigma_v_ref if layer.sigma_v_ref > 0 else sigma_v
            sv_shallow = self.soil.effective_stress_at_depth(
                SHALLOW_BETA_DEPTH)
            # a layer whose average is the stress at 7.5 ft (Example 10-1's
            # Layer 1, 0-15 ft) is not limited: 1e-6 kPa past the rounding
            limited = sv < sv_shallow - 1e-6
            if limited:
                sv = sv_shallow
            OCR = sigma_p / sv
            steps.update(
                OCR_from=("sigma_v_ref" if layer.sigma_v_ref > 0
                          else "sigma_v"),
                N60=layer.N60, sigma_p_correlation=layer.sigma_p_correlation,
                sigma_p_kPa=sigma_p, sigma_v_for_OCR_kPa=sv,
                shallow_limit=limited)
        b = beta_gec10(steps["phi_deg"], OCR, self.K_over_Ko)
        steps.update(OCR=OCR, Ko=b["Ko"], Kp=b["Kp"],
                     K_over_Ko=self.K_over_Ko, K=b["K"], beta=b["beta"])
        return steps

    def _gec10_side(self, layer, z1: float, z2: float, where: str,
                    sublayer: Optional[float] = None):
        """(Q_s kN, f_SN kPa, sigma'v kPa, steps) of a cohesionless layer's
        side resistance from z1 to z2 (m) by GEC-10 (2018) Eqs. 10-5 to
        10-13, on the layer's average sigma'v. ``sublayer`` (m) splits it
        into sub-layers that thick or less, each on its own sigma'v."""
        perimeter = self.shaft.perimeter
        n = max(1, math.ceil((z2 - z1) / sublayer - 1e-9)) if sublayer else 1
        dz = (z2 - z1) / n
        Qs = 0.0
        for i in range(n):
            sv = self.soil.average_effective_stress(z1 + i * dz,
                                                    z1 + (i + 1) * dz)
            steps = self._gec10_beta(layer, sv, where)
            Qs += steps["beta"] * sv * perimeter * dz
        sigma_v = self.soil.average_effective_stress(z1, z2)
        return Qs, Qs / (perimeter * (z2 - z1)), sigma_v, steps

    def _top_zone(self, soil_type: str) -> float:
        """The top zone (m) with no side resistance in a layer of
        ``soil_type``, before the casing: ``top_exclusion`` where given,
        else 1.5 m in cohesive soil and rock (GEC-10 p. 10-17, printed for
        cohesive soils) and none in cohesionless soil (Sec. 10.3.5.1;
        Examples 10-1 and 10-5)."""
        if self.top_exclusion is not None:
            return self.top_exclusion
        return 0.0 if soil_type == "cohesionless" else TOP_EXCLUSION

    def _top_of_side(self, soil_type: str) -> float:
        """Depth (m) above which a layer of ``soil_type`` takes no side
        resistance: its top zone or the permanent casing, whichever is
        deeper."""
        return max(self._top_zone(soil_type), self.shaft.casing_depth)

    def compute(self) -> DrillShaftResult:
        """Run the drilled shaft capacity analysis.

        Returns
        -------
        DrillShaftResult
            Complete results with side resistance, end bearing,
            and per-layer breakdown.
        """
        return self._compute()

    def _compute(self, sublayer: Optional[float] = None,
                 report: bool = True) -> DrillShaftResult:
        """:meth:`compute`; ``sublayer`` (m) splits each cohesionless layer
        into sub-layers for the GEC-10 beta, and ``report=False`` leaves
        out the choices (an alternative's run, or a trial length)."""
        shaft = self.shaft
        L = shaft.length
        D = shaft.diameter

        # A cohesionless tip with no N60 is refused by its end bearing
        # (GM-142) before the GEC-10 beta asks the same layer's N60
        tip_layer = self.soil.layer_at_depth(L - 0.01)
        if tip_layer.soil_type == "cohesionless":
            require_tip_N60(tip_layer, L)

        # Exclusion zones: the top zone by soil type (_top_of_side)
        # Bottom exclusion: 1 diameter for cohesive only (applied per-layer)
        # Rock not considered part of the socket (GEC-10 p. 8-15, GM-220)
        not_socket = fractured_rock_zone(shaft, self.soil)
        in_rock = False

        total_Qs = 0.0
        no_n60 = []          # cohesionless side layers with no N60 (GM-148)
        Qs_clay = 0.0
        Qs_sand = 0.0
        Qs_rock = 0.0
        layer_results = []

        current_depth = 0.0
        for layer in self.soil.layers:
            layer_top = current_depth
            layer_bottom = current_depth + layer.thickness
            current_depth = layer_bottom

            if layer_top >= L:
                break

            # Portion of this layer within shaft length
            z_top = layer_top
            z_bottom = min(layer_bottom, L)

            # Apply top exclusion zone
            z_top_eff = max(z_top, self._top_of_side(layer.soil_type))

            # Apply bottom exclusion for cohesive only (bottom 1*D)
            z_bottom_eff = z_bottom
            if layer.soil_type == "cohesive":
                bottom_excl_start = L - D
                z_bottom_eff = min(z_bottom, bottom_excl_start)

            if layer.soil_type == "rock":
                in_rock = True
                # the zone starts at the top of the first rock layer, so
                # it can only take off the top of a rock layer
                if not_socket is not None and z_top < not_socket[1]:
                    z_top_eff = max(z_top_eff, not_socket[1])

            effective_thickness = z_bottom_eff - z_top_eff
            if effective_thickness <= 0:
                layer_results.append({
                    "depth_top_m": round(z_top, 2),
                    "depth_bottom_m": round(z_bottom, 2),
                    "soil_type": layer.soil_type,
                    "description": layer.description,
                    "method": "excluded",
                    "side_resistance_kN": 0.0,
                    "fs_kPa": 0.0,
                })
                continue

            z_center = (z_top_eff + z_bottom_eff) / 2
            sigma_v = self.soil.effective_stress_at_depth(z_center)
            steps = None

            if layer.soil_type == "cohesive":
                if self.alpha_method == "rational":
                    # GEC-10 Chen-2011 alpha on the CIUC-equivalent strength.
                    su_ciuc = su_to_ciuc(layer.cu, sigma_v, self.su_test_type)
                    alpha = alpha_cohesive_rational(su_ciuc, self.pa)
                    Qs_layer = side_resistance_cohesive(
                        su_ciuc, shaft.perimeter, effective_thickness, alpha
                    )
                    fs = alpha * su_ciuc
                    method_used = f"alpha={alpha:.3f} (rational)"
                else:
                    alpha = alpha_cohesive(layer.cu)
                    Qs_layer = side_resistance_cohesive(
                        layer.cu, shaft.perimeter, effective_thickness, alpha
                    )
                    fs = alpha * layer.cu
                    method_used = f"alpha={alpha:.3f}"
                Qs_clay += Qs_layer

            elif layer.soil_type == "cohesionless":
                if self.beta_method == "rational":
                    # GEC-10 (2018) Eq. 10-13 on the layer's average
                    # sigma'v (Eq. 10-5), no limit on f_SN (GM-241)
                    where = (f"{z_top:g}-{z_bottom:g} m"
                             + (f" ({layer.description})"
                                if layer.description else ""))
                    Qs_layer, fs, sigma_v, steps = self._gec10_side(
                        layer, z_top_eff, z_bottom_eff, where, sublayer)
                    method_used = f"beta={steps['beta']:.3f} (rational)"
                else:
                    # N60 < 15 reduces beta (O'Neill-Reese); N60 = 0 means
                    # "not measured" -> no reduction (assumes N60 >= 15).
                    N60_side = layer.N60 if layer.N60 > 0 else None
                    if N60_side is None:
                        no_n60.append((z_top, z_bottom))
                    beta = beta_cohesionless(z_center, N60=N60_side)
                    method_used = f"beta={beta:.3f}"
                    Qs_layer = side_resistance_cohesionless(
                        sigma_v, beta, shaft.perimeter, effective_thickness
                    )
                    fs = min(beta * sigma_v, DEPTH_BETA_FS_MAX)
                Qs_sand += Qs_layer

            else:  # rock
                perimeter = shaft.socket_perimeter
                Qs_layer = side_resistance_rock(
                    layer.qu, perimeter, effective_thickness,
                    C=self.rock_side_C,
                )
                fs = self.rock_side_C * (layer.qu * PA) ** 0.5
                Qs_rock += Qs_layer
                method_used = "rock socket"

            total_Qs += Qs_layer
            row = {
                "depth_top_m": round(z_top, 2),
                "depth_bottom_m": round(z_bottom, 2),
                "effective_top_m": round(z_top_eff, 2),
                "effective_bottom_m": round(z_bottom_eff, 2),
                "soil_type": layer.soil_type,
                "description": layer.description,
                "method": method_used,
                "side_resistance_kN": round(Qs_layer, 1),
                "fs_kPa": round(fs, 1),
                "sigma_v_kPa": round(sigma_v, 1),
            }
            if steps is not None:
                row["beta_steps"] = {
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in steps.items()}
            layer_results.append(row)

        # End bearing. Large-base reductions are governed by the bearing
        # surface, so use the bell diameter when the shaft is belled; in
        # rock the base is the socket's (GEC-10 Example 10-4, GM-221).
        tip_layer = self.soil.layer_at_depth(L - 0.01)
        sigma_v_tip = self.soil.effective_stress_at_depth(L)
        L_over_D = L / D
        tip_in_rock = tip_layer.soil_type == "rock"
        base_D = shaft.base_diameter(tip_in_rock)
        base_area = shaft.base_area(tip_in_rock)

        if tip_layer.soil_type == "cohesive":
            Qt = end_bearing_cohesive(
                tip_layer.cu, base_area, L_over_D,
                base_diameter=base_D, shaft_length=L,
            )
        elif tip_layer.soil_type == "cohesionless":
            N60 = require_tip_N60(tip_layer, L)
            Qt = end_bearing_cohesionless(N60, base_area, base_D)
        else:  # rock
            Qt = end_bearing_rock(tip_layer.qu, base_area, tip_layer.RQD)

        Q_ultimate = total_Qs + Qt
        Q_allowable = Q_ultimate / self.factor_of_safety

        choices = []
        if report:
            choices = (
                self._side_choices(layer_results, Qs_sand, Q_ultimate)
                + ([side_n60_choice(no_n60)] if no_n60 else [])
                + ([fractured_rock_choice(shaft, self.soil)]
                   if in_rock or shaft.fractured_rock_thickness else [])
                + ([rock_side_choice(self.rock_side_C)] if in_rock else []))

        return DrillShaftResult(
            Q_ultimate=Q_ultimate,
            Q_skin=total_Qs,
            Q_tip=Qt,
            Q_allowable=Q_allowable,
            Q_side_clay=Qs_clay,
            Q_side_sand=Qs_sand,
            Q_side_rock=Qs_rock,
            factor_of_safety=self.factor_of_safety,
            shaft_diameter=D,
            shaft_length=L,
            base_diameter=base_D,
            base_area=base_area,
            method="GEC-10",
            layer_breakdown=layer_results,
            sigma_v_tip=sigma_v_tip,
            choices=choices,
        )

    def _alternative(self, sublayer: Optional[float] = None,
                     correlation: Optional[str] = None, **changes) -> str:
        """The side resistance in cohesionless soil and the ultimate
        capacity with ``changes`` to this analysis, in words; or why it
        is refused. ``correlation`` gives every cohesionless layer that
        takes sigma'p from N60 that ``sigma_p_correlation``."""
        soil = self.soil
        if correlation is not None:
            soil = replace(soil, layers=[
                replace(lay, sigma_p_correlation=correlation)
                if lay.soil_type == "cohesionless" and not lay.OCR > 0
                else lay for lay in soil.layers])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = replace(self, soil=soil, **changes)._compute(
                    sublayer=sublayer, report=False)
        except ValueError as exc:
            return f"refused: {exc}"
        return _side_values(r.Q_side_sand, r.Q_ultimate)

    def _top_zone_choice(self, layer_results) -> Optional[dict]:
        """The choice made for side resistance near the ground surface
        (GM-241), or None where it changes nothing: with top_exclusion
        left out, cohesionless soil in the top 1.5 m below the casing
        keeps its side resistance; given, the top zone takes the layers it
        reaches below the casing."""
        from geotech_common.choices import choice, how_given
        casing = self.shaft.casing_depth

        def reached(lb, bottom):
            """Whether the layer has a part below the casing and above
            ``bottom``."""
            return (max(lb["depth_top_m"], casing)
                    < min(lb["depth_bottom_m"], bottom))

        def spans(rows, bottom):
            return ", ".join(f"{max(lb['depth_top_m'], casing):g}-"
                             f"{min(lb['depth_bottom_m'], bottom):g} m"
                             for lb in rows)

        if self.top_exclusion is None:
            sand = [lb for lb in layer_results
                    if lb["soil_type"] == "cohesionless"
                    and reached(lb, TOP_EXCLUSION)]
            if not sand:
                return None
            others = [lb for lb in layer_results
                      if lb["soil_type"] != "cohesionless"
                      and reached(lb, TOP_EXCLUSION)]
            chosen = (f"side resistance taken in cohesionless soil within "
                      f"the top {TOP_EXCLUSION:g} m ({spans(sand, TOP_EXCLUSION)})")
            if others:
                chosen += (f"; none in cohesive soil or rock there "
                           f"({spans(others, TOP_EXCLUSION)})")
            alternatives = [
                f"top_exclusion {TOP_EXCLUSION:g}, the top 5 ft of p. 10-17 "
                "left out in every layer: "
                + self._alternative(top_exclusion=TOP_EXCLUSION),
                "top_exclusion the depth of seasonal moisture change or of "
                "frost, where deeper (p. 10-17)"]
        else:
            cut = [lb for lb in layer_results
                   if reached(lb, self.top_exclusion)]
            if not cut:
                return None
            chosen = (f"none over the top {self.top_exclusion:g} m, in "
                      f"every layer ({spans(cut, self.top_exclusion)})")
            left_out = self._alternative(top_exclusion=None)
            alternatives = [
                f"top_exclusion left out: none over the top "
                f"{TOP_EXCLUSION:g} m in cohesive soil and rock, side "
                "resistance from the ground surface in cohesionless soil: "
                + left_out]
            zero = self._alternative(top_exclusion=0.0)
            if self.top_exclusion != 0 and zero != left_out:
                alternatives.append(
                    "top_exclusion 0, side resistance from the ground "
                    "surface in every layer: " + zero)
        return choice(
            "side resistance near the ground surface", chosen,
            how_given(self.top_exclusion, None, name="top_exclusion"),
            basis=TOP_EXCLUSION_BASIS, alternatives=alternatives)

    def _side_choices(self, layer_results, Qs_sand: float,
                      Q_ultimate: float) -> List[dict]:
        """The choices made for side resistance in cohesionless soil
        (GM-241): the method, and under GEC-10 (2018) K/Ko, sigma'p from
        N60, the limit on beta near the surface and on f_SN; and the top
        zone (:meth:`_top_zone_choice`). Each lists its alternatives with
        their values."""
        from geotech_common.choices import MODULE, choice, how_given
        sand = [lb for lb in layer_results
                if lb["soil_type"] == "cohesionless"
                and lb["method"] != "excluded"]
        out = []
        top = self._top_zone_choice(layer_results)
        if top:
            out.append(top)
        if not sand:
            return out
        here = _side_values(Qs_sand, Q_ultimate)
        how = how_given(self.beta_method, "rational", name="beta_method")
        if self.beta_method != "rational":
            out.insert(0, choice(
                "side resistance in cohesionless soil",
                "depth beta of O'Neill and Reese (1999): " + here, how,
                basis=DEPTH_BETA_BASIS,
                alternatives=[
                    "beta_method 'rational', GEC-10 (2018) Eq. 10-13: "
                    + self._alternative(beta_method="rational")]))
            return out
        out.insert(0, choice(
            "side resistance in cohesionless soil",
            "GEC-10 (2018) beta, Eq. 10-13, on each layer's average "
            "sigma'v: " + here, how,
            basis=GEC10_BETA_BASIS,
            alternatives=[
                "each layer split into sub-layers of 0.1 m, each on its own "
                "sigma'v (GEC-10 p. 10-7 splits a layer where its N-values "
                "vary by more than a COV of 45 %): "
                + self._alternative(sublayer=0.1),
                "beta_method 'depth', O'Neill and Reese (1999) beta = 1.5 "
                "- 0.135 sqrt(z, ft), 0.25 to 1.2 (FHWA NHI-06-089 Eqs. "
                "9-41 to 9-43), which GEC-10 (2018) does not print: "
                + self._alternative(beta_method="depth")]))
        k = self.K_over_Ko
        k_alts = []
        if abs(k - 2 / 3) > 0.005:
            k_alts.append("2/3, where construction quality is not "
                          "controlled properly (p. 10-12): "
                          + self._alternative(K_over_Ko=2 / 3))
        if k != 1:
            k_alts.append("1, no change in K from construction (p. 10-11): "
                          + self._alternative(K_over_Ko=1.0))
        out.append(choice(
            "K/Ko in cohesionless soil (construction's effect on the "
            "horizontal stress)", f"{k:.3g}",
            how_given(k, 1.0, name="K_over_Ko"),
            basis=K_OVER_KO_BASIS, alternatives=k_alts))
        steps = [lb["beta_steps"] for lb in sand]
        from_n60 = [(lb, s) for lb, s in zip(sand, steps)
                    if s["OCR_from"] != "OCR"]
        if from_n60:
            used = [s["sigma_p_correlation"] for _, s in from_n60]
            chosen = "; ".join(
                SIGMA_P_CORRELATIONS[c] + ": " + ", ".join(
                    f"{lb['depth_top_m']:g}-{lb['depth_bottom_m']:g} m"
                    for lb, s in from_n60 if s["sigma_p_correlation"] == c)
                for c in dict.fromkeys(used))
            out.append(choice(
                "preconsolidation stress sigma'p of cohesionless soil from "
                "N60", chosen,
                how_given(tuple(used), ("clean_sand",) * len(used),
                          within="layers"),
                basis=SIGMA_P_BASIS,
                alternatives=[
                    f"sigma_p_correlation '{c}' on these layers, "
                    f"{SIGMA_P_CORRELATIONS[c]}: "
                    + self._alternative(correlation=c)
                    for c in SIGMA_P_CORRELATIONS if set(used) != {c}]))
        shallow = [lb for lb, s in zip(sand, steps)
                   if s.get("shallow_limit")]
        if shallow:
            sv = self.soil.effective_stress_at_depth(SHALLOW_BETA_DEPTH)
            out.append(choice(
                "beta near the ground surface",
                f"limited to its value at 7.5 ft (2.29 m), sigma'v = "
                f"{sv:.1f} kPa, at " + ", ".join(
                    f"{lb['depth_top_m']:g}-{lb['depth_bottom_m']:g} m"
                    for lb in shallow), MODULE,
                basis=("GEC-10 (FHWA-NHI-18-024) pp. 10-11, 10-12: 'The "
                       "value of beta at shallow depths should be limited "
                       "to the value corresponding to a depth of 7.5 ft, "
                       "which corresponds to a vertical effective stress "
                       "of approximately 900 psf. At lower confining "
                       "stress, the correlations for effective stress "
                       "friction angle and preconsolidation stress have "
                       "not been validated'. The sigma'v at 7.5 ft is this "
                       "profile's"),
                alternatives=[]))
        high = [lb for lb in sand if lb["fs_kPa"] > NHI_FS_MAX]
        if high:
            capped = Qs_sand - sum(
                (lb["fs_kPa"] - NHI_FS_MAX) * self.shaft.perimeter
                * (lb["effective_bottom_m"] - lb["effective_top_m"])
                for lb in high)
            out.append(choice(
                "upper limit on f_SN in cohesionless soil",
                "none: f_SN up to "
                f"{max(lb['fs_kPa'] for lb in high):.0f} kPa", MODULE,
                basis=("GEC-10 (2018) Sec. 10.3.5.1 prints no upper limit "
                       "on f_SN in cohesionless soil; FHWA NHI-06-089 "
                       "limits the depth beta's f_s to 4 ksf (191 kPa, p. "
                       "9-133)"),
                alternatives=[
                    "f_SN limited to 191 kPa, as NHI-06-089 does for the "
                    f"depth beta: Q_s in cohesionless soil about "
                    f"{capped:,.0f} kN, Q_ult about "
                    f"{Q_ultimate - Qs_sand + capped:,.0f} kN"]))
        return out

    def capacity_vs_depth(self, depth_min: float = 3.0,
                          depth_max: float = None,
                          n_points: int = 20,
                          skipped: Optional[list] = None) -> List[dict]:
        """Compute capacity at multiple shaft lengths for optimization.

        Parameters
        ----------
        depth_min : float, optional
            Minimum shaft length (m). Default 3.0.
        depth_max : float, optional
            Maximum shaft length (m). Default: soil profile depth.
        n_points : int, optional
            Number of evaluation points. Default 20.
        skipped : list, optional
            Receives a ``{'depth_m', 'reason'}`` dict for each trial length
            the analysis refused, which the curve leaves out. Not given, a
            warning names them (GM-203: they were dropped without a word).
            With every length refused the call is refused.

        Returns
        -------
        list of dict
            Each dict has 'depth_m', 'Q_ultimate_kN', 'Q_skin_kN', 'Q_tip_kN'
            and 'Q_allowable_kN' (Q_ultimate / factor_of_safety).
        """
        if depth_max is None:
            depth_max = min(self.shaft.length, self.soil.total_thickness)

        depths = np.linspace(depth_min, depth_max, n_points)
        results = []

        # A cohesionless tip with no N60 is refused (GM-142), not left out
        # of the curve by the guard below.
        for d in depths:
            tip = self.soil.layer_at_depth(float(d) - 0.01)
            if tip.soil_type == "cohesionless":
                require_tip_N60(tip, float(d))

        # Use a copy of the shaft per trial depth rather than temporarily
        # mutating the shared shaft.length (re-entrant / thread-safe).
        missed = []
        for d in depths:
            try:
                trial_shaft = replace(self.shaft, length=float(d))
                trial = replace(self, shaft=trial_shaft)
                r = trial._compute(report=False)
                results.append({
                    "depth_m": round(d, 2),
                    "Q_ultimate_kN": round(r.Q_ultimate, 1),
                    "Q_skin_kN": round(r.Q_skin, 1),
                    "Q_tip_kN": round(r.Q_tip, 1),
                    "Q_allowable_kN": round(r.Q_allowable, 1),
                })
            except Exception as exc:
                missed.append({"depth_m": round(float(d), 2),
                               "reason": f"{type(exc).__name__}: {exc}"})
        if missed and not results:
            raise ValueError(skipped_depths_note(missed, len(depths)))
        if missed:
            if skipped is not None:
                skipped.extend(missed)
            else:
                warnings.warn(skipped_depths_note(missed, len(depths)))
        return results
