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
    FHWA GEC-10 (FHWA-NHI-10-016), Chapters 13-14
"""

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
    beta_cohesionless_rational, phi_prime_from_N1_60, preconsolidation_stress,
)
from drilled_shaft.end_bearing import (
    end_bearing_cohesive, end_bearing_cohesionless, end_bearing_rock,
)
from drilled_shaft.results import DrillShaftResult


def require_tip_N60(tip_layer: ShaftSoilLayer, depth: float) -> float:
    """The N60 a cohesionless tip layer's end bearing uses (GM-142).

    End bearing in cohesionless soil is q_b = 57.5 N60 kPa (N60 capped
    at 50; GEC-10, O'Neill & Reese 1999), so it needs the tip layer's
    N60. Until GM-142 a layer with no N60 took N60 = 15 without saying
    so; now it is refused.
    """
    if tip_layer.N60 > 0:
        return tip_layer.N60
    name = f" ({tip_layer.description})" if tip_layer.description else ""
    raise ValueError(
        f"the shaft tip at {depth:g} m is in a cohesionless layer{name} "
        "with no N60: end bearing q_b = 57.5 N60 kPa (N60 capped at 50; "
        "GEC-10, O'Neill & Reese 1999) needs that layer's N60 (the SPT "
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
                      "beta_method 'rational' (GEC-10 Sec. 10.3.5.1; "
                      "needs N60 or N1_60)"])


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
        Cohesionless side-resistance method: "depth" (default, O'Neill & Reese
        1.5-0.245*sqrt(z_ft)) or "rational" (GEC-10 Appendix A OCR/Ko chain, from
        per-layer N60/N1_60/sigma_v_ref). Default "depth" preserves prior results.
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
    ...     ShaftSoilLayer(5, 'cohesionless', 18.0, phi=32),
    ...     ShaftSoilLayer(10, 'cohesive', 17.0, cu=75),
    ...     ShaftSoilLayer(5, 'rock', 22.0, qu=5000),
    ... ])
    >>> analysis = DrillShaftAnalysis(shaft=shaft, soil=soil)
    >>> result = analysis.compute()
    """
    shaft: DrillShaft = None
    soil: ShaftSoilProfile = None
    factor_of_safety: float = 2.5
    beta_method: str = "depth"
    alpha_method: str = "aashto"
    su_test_type: str = "ciuc"
    pa: float = PA
    rock_side_C: float = ROCK_SIDE_C

    def __post_init__(self):
        if self.shaft is None:
            raise ValueError("Shaft must be provided")
        if not self.rock_side_C > 0:
            raise ValueError(
                f"rock_side_C must be positive, got {self.rock_side_C}")
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

    def _rational_beta(self, layer, sigma_v: float) -> float:
        """GEC-10 Appendix A rational beta for a cohesionless layer.

        phi' from (N1)60 (or layer.phi); OCR from sigma'p(N60)/sigma_v_ref
        (or a direct layer.OCR); delta = phi'.
        """
        phi = phi_prime_from_N1_60(layer.N1_60) if layer.N1_60 > 0 else layer.phi
        if layer.OCR > 0:
            OCR = layer.OCR
        else:
            if layer.N60 <= 0:
                raise ValueError(
                    "beta_method='rational' needs N60 (for sigma'p) or a direct "
                    "OCR on each cohesionless layer"
                )
            sigma_p = preconsolidation_stress(layer.N60, self.pa)
            sigma_ref = layer.sigma_v_ref if layer.sigma_v_ref > 0 else sigma_v
            OCR = sigma_p / sigma_ref
        return beta_cohesionless_rational(phi, OCR)

    def compute(self) -> DrillShaftResult:
        """Run the drilled shaft capacity analysis.

        Returns
        -------
        DrillShaftResult
            Complete results with side resistance, end bearing,
            and per-layer breakdown.
        """
        shaft = self.shaft
        L = shaft.length
        D = shaft.diameter

        # Exclusion zones
        top_exclusion = max(1.5, shaft.casing_depth)
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
            z_top_eff = max(z_top, top_exclusion)

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
                    beta = self._rational_beta(layer, sigma_v)
                    method_used = f"beta={beta:.3f} (rational)"
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
                fs = min(beta * sigma_v, 200.0)
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
            layer_results.append({
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
            })

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
            choices=([side_n60_choice(no_n60)] if no_n60 else [])
            + ([fractured_rock_choice(shaft, self.soil)]
               if in_rock or shaft.fractured_rock_thickness else [])
            + ([rock_side_choice(self.rock_side_C)] if in_rock else []),
        )

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
                trial = DrillShaftAnalysis(
                    shaft=trial_shaft, soil=self.soil,
                    factor_of_safety=self.factor_of_safety,
                    beta_method=self.beta_method, alpha_method=self.alpha_method,
                    su_test_type=self.su_test_type, pa=self.pa,
                    rock_side_C=self.rock_side_C,
                )
                r = trial.compute()
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
