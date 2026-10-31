"""
Combined drilled shaft capacity analysis.

Auto-selects the appropriate method for each soil layer:
- Alpha method for cohesive layers (clay/silt)
- Beta method for cohesionless layers (sand/gravel)
- Rock socket method for rock layers

Side resistance is left out over the top 1.5 m in cohesive soil (GEC-10's
'top 5 ft', p. 10-17) unless top_exclusion is given, and over the
permanently cased zone; over the bottom of the shaft in cohesive soil only
where bottom_exclusion is given (GEC-10 (2018) prints no such zone).

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
    SIGMA_P_CORRELATIONS, DEPTH_BETA_FS_MAX,
    alpha_E_table_10_3, CAVING_C, ROCK_JOINTS, TABLE_10_3,
)
from drilled_shaft.end_bearing import (
    CLAY_BASE_METHODS, LARGE_BASE_DIAMETER, LARGE_BASE_DIAMETER_SAND,
    ROCK_BASE_METHODS, hoek_brown_base, nc_below_table_10_2,
    rock_base_factor, unit_end_bearing_cohesionless,
    unit_end_bearing_cohesive,
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
#: when top_exclusion is not given: GEC-10's 'top 5 ft' (p. 10-17), printed
#: for cohesive soils. Cohesionless soil and rock then take side resistance
#: from the top: Secs. 10.3.5.1 and 10.3.5.3 print no such zone (Examples
#: 10-1 and 10-5 take the sand's from the surface, Example 10-4 the
#: socket's over its whole length below the casing).
TOP_EXCLUSION = 1.5

#: GEC-10 (FHWA-NHI-18-024) on the top of the shaft, as printed.
_TOP_COHESIVE = (
    "GEC-10 (FHWA-NHI-18-024) p. 10-17, in Sec. 10.3.5.2, Cohesive Soils: "
    "'It is common design practice to neglect side resistance over the top "
    "5 ft of a drilled shaft, or to the depth of seasonal moisture "
    "change', for wetting and drying, freezing and thawing, 'gapping' "
    "caused by cyclic lateral loading, or any process near the ground "
    "surface that softens the soil or ends its contact with the shaft; "
    "'This recommendation is subject to modification based on local "
    "experience and judgment of the designer.' ")
_TOP_SAND = (
    "Sec. 10.3.5.1, Cohesionless Soils, prints no such zone, and Examples "
    "10-1 and 10-5 (pp. 10-12 to 10-14, 10-29) take the sand's side "
    "resistance from the ground surface. ")
_TOP_ROCK = (
    "Sec. 10.3.5.3, Rock (pp. 10-20 to 10-26), prints none at the top of a "
    "socket, and Example 10-4 (pp. 10-27, 10-28) takes the socket's side "
    "resistance over its whole 10 ft below the casing. ")
_TOP_RULE = (
    "With top_exclusion left out, the top 1.5 m carries none in cohesive "
    "soil, and cohesionless soil and rock carry side resistance from the "
    "top; a top_exclusion given applies in every layer. The permanent "
    "casing, where deeper, takes its place")


def top_exclusion_basis(soil_types) -> str:
    """The top zone's basis, citing GEC-10 on the kinds of ground the
    profile has (``soil_types``), not on the others (GM-230)."""
    return (_TOP_COHESIVE
            + (_TOP_SAND if "cohesionless" in soil_types else "")
            + (_TOP_ROCK if "rock" in soil_types else "") + _TOP_RULE)

#: The depth (m) at whose sigma'v GEC-10 limits beta near the surface:
#: 'the value corresponding to a depth of 7.5 ft' (pp. 10-11, 10-12).
SHALLOW_BETA_DEPTH = 7.5 * 0.3048

#: FHWA NHI-06-089's 4 ksf limit on the depth beta's f_s, in kPa
#: (p. 9-133; 4 x 47.88 = 191.5 kPa): :data:`DEPTH_BETA_FS_MAX`.
NHI_FS_MAX = DEPTH_BETA_FS_MAX

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
    "to a depth of 7.5 ft' (pp. 10-11, 10-12). phi' is the layer's where "
    "given, else 27.5 + 9.2 log (N1)60 (Example 10-1, p. 10-13; Appendix "
    "A, Eq. A-2, p. A-8). GEC-10 prints no upper limit on f_SN in "
    "cohesionless soil. Resistance factor 0.55 (p. 10-12)")

#: The depth beta, as its sources print it.
DEPTH_BETA_BASIS = (
    "O'Neill and Reese (1999) as printed in FHWA NHI-06-089 Eqs. 9-41 to "
    "9-43 (pp. 9-132, 9-133): beta = 1.5 - 0.135 sqrt(z, ft) = 1.5 - 0.245 "
    "sqrt(z, m), 0.25 <= beta <= 1.2, z the depth to the layer's centre; "
    "f_s = beta sigma'v <= 4 ksf, 'The limiting value of fsi for shafts in "
    "cohesionless soils is 4 ksf (191 kPa)' (p. 9-133), taken as 4 x "
    "47.88 = 191.5 kPa. Where a layer's N60 is under 15, beta x N60/15, "
    "from "
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


#: The parts of the capacity a choice's alternatives give (GM-244): the
#: words and the result's attribute.
_PARTS = {"sand": ("Q_s in cohesionless soil", "Q_side_sand"),
          "clay": ("Q_s in cohesive soil", "Q_side_clay"),
          "rock": ("Q_s in rock", "Q_side_rock"),
          "side": ("Q_s", "Q_skin"),
          "base": ("Q_b", "Q_tip")}


def _rounded(d: dict, sig: tuple = ()) -> dict:
    """``d`` with its floats rounded to 3 places, for a result; those named
    in ``sig`` to 4 significant figures (the Hoek-Brown s, GM-251, is
    0.0004 at GSI 30)."""
    return {k: ((float(f"{v:.4g}") if k in sig else round(v, 3))
                if isinstance(v, float) else v)
            for k, v in d.items()}


def _zone_steps(zone: dict) -> dict:
    """The result's record of the mean su over 2B below the base
    (GM-252): the tip layer's su, the zone, each layer's part of it with
    its su (None where it has none) and, where the profile ends inside
    the zone, its last depth."""
    out = {"su_tip_kPa": zone["su_tip_kPa"],
           "su_zone_m": [round(zone["top_m"], 3), round(zone["bottom_m"], 3)],
           "su_layers": [
               {"top_m": round(r["top_m"], 3),
                "bottom_m": round(r["bottom_m"], 3),
                "soil_type": r["soil_type"],
                "su_kPa": (None if r["su_kPa"] is None
                           else round(r["su_kPa"], 3))}
               for r in zone["rows"]]}
    if zone["profile_bottom_m"] < zone["bottom_m"] - 1e-9:
        out["su_profile_ends_m"] = zone["profile_bottom_m"]
    return out


def _n60_zone_steps(zone: dict) -> dict:
    """The result's record of the mean N60 over 2B below the base in
    cohesionless soil (GM-284): the mean, the tip layer's, the zone, each
    layer's part of it with its N60 (None where it gives none) and, where
    the profile ends inside the zone, its last depth."""
    out = {"N60_mean": round(zone["N60"], 3),
           "N60_tip": zone["N60_tip"],
           "N60_zone_m": [round(zone["top_m"], 3), round(zone["bottom_m"], 3)],
           "N60_layers": [
               {"top_m": round(r["top_m"], 3),
                "bottom_m": round(r["bottom_m"], 3),
                "soil_type": r["soil_type"], "N60": r["N60"]}
               for r in zone["rows"]]}
    if zone["profile_bottom_m"] < zone["bottom_m"] - 1e-9:
        out["N60_profile_ends_m"] = zone["profile_bottom_m"]
    return out


def _n60_zone_words(steps: dict) -> str:
    """The N60 of the sand base in words (GM-284), from the base steps of
    :func:`_n60_zone_steps`."""
    z1, z2 = steps["N60_zone_m"]
    rows = steps["N60_layers"]
    used = [r for r in rows if r["N60"] is not None]
    other = [r for r in rows if r["N60"] is None]
    end = steps.get("N60_profile_ends_m", z2)
    if not used:
        words = (f"the tip layer's N60 {steps['N60_tip']:g}, as the profile "
                 + (f"ends at {end:g} m" if not rows else
                    f"gives no cohesionless N60 within 2B below the base "
                    f"({z1:g}-{z2:g} m)"))
    elif len(used) == 1 and used[0]["N60"] == steps["N60_tip"]:
        words = (f"the mean over 2B = {z2 - z1:g} m below the base "
                 f"({z1:g}-{z2:g} m), all at the tip layer's "
                 f"{steps['N60_tip']:g}")
    else:
        words = (f"the mean over 2B = {z2 - z1:g} m below the base "
                 f"({z1:g}-{z2:g} m), by thickness: " + "; ".join(
                     f"{r['top_m']:g}-{r['bottom_m']:g} m N60 {r['N60']:g}"
                     for r in used))
    if other:
        words += "; " + ", ".join(
            f"{r['soil_type']} at {r['top_m']:g}-{r['bottom_m']:g} m"
            for r in other) + " left out (no N60)"
    if rows and end < z2 - 1e-9:
        words += (f"; the profile ends at {end:g} m, {z2 - end:g} m short "
                  "of 2B below the base: the mean is over the soil given")
    return words


def _su_zone_words(steps: dict, tip_words: str) -> str:
    """The su of the clay base in words (GM-252), from the base steps of
    :func:`_zone_steps`: the mean over 2B below the base, the layers it
    covers and what it leaves out."""
    z1, z2 = steps["su_zone_m"]
    head = f"the mean over 2B = {z2 - z1:g} m below the base ({z1:g}-{z2:g} m)"
    rows = steps["su_layers"]
    clay = [r for r in rows if r["su_kPa"] is not None]
    other = [r for r in rows if r["su_kPa"] is None]
    end = steps.get("su_profile_ends_m", z2)
    if not rows:
        return (tip_words + f", as the profile ends at {end:g} m and gives "
                "nothing below the base")
    if not clay:
        words = (tip_words + ", as the profile gives no cohesive soil "
                 f"within 2B below the base ({z1:g}-{z2:g} m)")
    elif len(clay) == 1 and abs(clay[0]["su_kPa"] - steps["su_tip_kPa"]) \
            < 1e-3:
        words = f"{head}, all in the tip layer: {tip_words}"
    else:
        words = (f"{head}, by thickness, of the cohesive soil: " + "; ".join(
            f"{r['top_m']:g}-{r['bottom_m']:g} m {r['su_kPa']:.1f} kPa"
            for r in clay))
    if other:
        words += "; " + ", ".join(
            f"{r['soil_type']} at {r['top_m']:g}-{r['bottom_m']:g} m"
            for r in other) + " left out (no su)"
    if end < z2 - 1e-9:
        words += (f"; the profile ends at {end:g} m, {z2 - end:g} m short "
                  "of 2B below the base: the mean is over the soil given")
    return words


def _spans(rows, key_top="depth_top_m", key_bottom="depth_bottom_m") -> str:
    """The layers' depth ranges, in words."""
    return ", ".join(f"{lb[key_top]:g}-{lb[key_bottom]:g} m" for lb in rows)


def _values(r: DrillShaftResult, part: str = "sand") -> str:
    """A run's ``part`` of the capacity (:data:`_PARTS`) and its ultimate
    capacity, in words, for a choice's alternatives."""
    words, attr = _PARTS[part]
    return f"{words} {getattr(r, attr):,.0f} kN, Q_ult {r.Q_ultimate:,.0f} kN"


#: GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.2 on side resistance in cohesive
#: soil, as printed (GM-244).
ALPHA_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.2, Step 5 of the design for "
    "axial load, pp. 10-15 to 10-18: R_SN = pi B dz alpha su (Eq. 10-15); "
    "alpha = 0.30 + 0.17 / (su(CIUC) / pa) (Eq. 10-18, p. 10-16; Chen et "
    "al. 2011, Fig. 10-6, p. 10-17), 'recommended as a revision to the "
    "expressions for alpha currently given in the AASHTO LRFD Bridge Design "
    "Specifications (2017a)'; f_SN = alpha su(CIUC) (Example 10-2, p. "
    "10-18). su is each layer's, with sigma'vo for a transform at the "
    "middle of the length that carries side resistance. Resistance factor "
    "0.45 (Table 8-4, p. 8-14)")

#: The older alpha, as FHWA NHI-06-089 prints it (GM-244).
AASHTO_ALPHA_BASIS = (
    "O'Neill and Reese (1999) as FHWA NHI-06-089 prints it, Eqs. 9-37a and "
    "9-37b (p. 9-129): alpha = 0.55 for su/pa <= 1.5, 0.55 - 0.1 (su/pa - "
    "1.5) for 1.5 < su/pa <= 2.5, on su as measured; NHI-06-089 prints "
    "nothing above su/pa = 2.5, where this tool goes on along the line to "
    "0.35 and holds it. GEC-10 (2018) recommends Eq. 10-18 'as a revision' "
    "to it (p. 10-16)")

#: GEC-10 (FHWA-NHI-18-024) p. 10-16 on the strength in Eq. 10-18.
SU_TEST_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) p. 10-16: Eq. 10-18 is fitted to su(CIUC) "
    "(Fig. 10-6); 'For values of undrained shear strength determined by UC "
    "or UU lab tests, the value of su should first be transformed to its "
    "equivalent CIUC value' by Eq. 10-16 (UC) or Eq. 10-17 (UU), Chen and "
    "Kulhawy (1993), on sigma'vo at the layer's middle (Example 10-2, p. "
    "10-18). Which test each cu came from is the engineer's to state: left "
    "out, cu is taken as su(CIUC), untransformed")

#: GEC-10 (FHWA-NHI-18-024) and NHI-06-089 on side resistance near the
#: base in cohesive soil (GM-244).
BOTTOM_ZONE_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) prints no zone above the base that carries "
    "no side resistance: Sec. 10.3.5.2 prints only the top 5 ft (p. "
    "10-17), and Example 10-2 (p. 10-18) takes the clay's side resistance "
    "over the whole 27 ft of Layer 3, down to the shaft's base at 60 ft. "
    "FHWA NHI-06-089 p. 9-129 (O'Neill and Reese 1999): 'the top and bottom "
    "5-ft of the shaft should not be included in the development of the "
    "ultimate skin resistance', its Fig. 9-49 (FHWA 1999) marking the "
    "bottom one diameter as noncontributing")

#: GEC-10 (FHWA-NHI-18-024) p. 10-19 on base resistance in cohesive soil.
CLAY_BASE_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.2, p. 10-19: q_BN = N*c su (Eq. "
    "10-19, O'Neill and Reese 1999), su the 'mean undrained shear strength "
    "of the cohesive soil over a depth of 2B below the base'; 'For cases "
    "where the shaft depth is at least 3 times the diameter and the mean "
    "undrained shear strength is at least 2,000 psf, the bearing capacity "
    "factor can be taken as 9.0. For smaller values of undrained shear "
    "strength, N*c can be approximated as a function of undrained shear "
    "strength as given in TABLE 10-2' (6.5 at 500 psf, 8.0 at 1,000 psf, "
    "9.0 at 2,000 psf or more; 'Linear interpolation can be used'); for a "
    "depth D less than 3B, q_BN = (2/3)[1 + D/(6B)] N*c su (Eq. 10-20). "
    "Example 10-3 (p. 10-20): 9.0 x 2,571 psf = 23.1 ksf. No upper limit "
    "on q_BN and no reduction for a large base is printed. 'it is unusual "
    "to locate the base of a drilled shaft in cohesive soil with su less "
    "than 2,000 psf when compression loads are supported' (p. 10-19). "
    "Resistance factor 0.40 (Table 8-4, p. 8-14). su is taken as the mean "
    "of the cohesive soil between the base and 2B below it, by thickness, "
    "B the base's diameter (GM-252): the layer the base is in at its su as "
    "its side resistance took it (Example 10-3 applies N*c to Example "
    "10-2's su), each cohesive layer below on its own cu, transformed by "
    "Eq. 10-16 or 10-17 at the middle of its part within 2B where cu is a "
    "UC or UU strength. A cohesionless or rock layer within 2B has no su "
    "and is left out of the mean; where the profile ends less than 2B "
    "below the base, the mean is over the soil the profile gives")

#: GEC-10 (FHWA-NHI-18-024) pp. 10-14, 10-15 on base resistance in
#: cohesionless soil, as printed (GM-284).
SAND_BASE_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.1, pp. 10-14, 10-15: q_BN (tsf) = "
    "0.60 N60 < 30 tsf (Eq. 10-14, Reese and O'Neill 1989, 'recommended "
    "for routine design'), taken as 57.5 kPa a blow (0.60 tsf = 57.46 "
    "kPa), 'N60 is the average "
    "value between the base and two diameters beneath the base'; 'the "
    "N-value to be used in Equation 10-14 is the field value, with no "
    "correction for overburden stress, although it is recommended that "
    "N-values correspond to 60 percent hammer efficiency' (so q_BN does "
    "not depend on sigma'v); 'limited to cohesionless soils with SPT "
    "N-values of 50 blows per foot or less. In cohesionless soils with "
    "N-values exceeding 50, load testing is recommended; otherwise unit "
    "base resistance is limited to the upper-bound value' of 30 tsf. It is "
    "the base resistance 'at settlements equal to five percent of the base "
    "diameter'; bearing capacity theory is an alternative GEC-10 does not "
    "recommend for routine design. Resistance factor 0.50 (p. 10-15; Table "
    "8-4, p. 8-14). N60 is taken as the mean, by thickness, of the "
    "cohesionless layers' N60 between the base and 2B below it, B the "
    "base's diameter, and limited to 50; a layer within 2B with no N60, "
    "or of cohesive soil or rock, is left out of the mean; where the "
    "profile ends less than 2B below the base, the mean is over what it "
    "gives")

#: The older base in clay, as FHWA NHI-06-089 prints it (GM-244).
ONEILL_REESE_CLAY_BASE_BASIS = (
    "O'Neill and Reese (1999) as FHWA NHI-06-089 prints it (p. 9-131): q_t "
    "= Nc su_t (Eq. 9-39), Nc = 6.0 [1 + 0.2 (z/D)] <= 9 (Eq. 9-40); 'The "
    "limiting value of unit end bearing (qt = Ncsut) is 80 ksf', 'a limit "
    "based on the largest measured values'; 'If the soil within 2.0 "
    "diameters of the tip has sut < 0.5 ksf, the value of Nc should be "
    "multiplied by 0.67' (taken here on the tip layer's cu). GEC-10 (2018) "
    "gives Eqs. 10-19, 10-20 with Table 10-2 instead")

#: GEC-10 (FHWA-NHI-18-024) pp. 10-22 to 10-24 on base resistance in rock.
_EQ_10_23_CONDITIONS = (
    "GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.3, p. 10-22: q_BN = N*cr qu "
    "(Eq. 10-23); 'a value of Ncr* = 2.5 is recommended for design when qu "
    "is the sole parameter used for establishing qBN and the following "
    "conditions are met: 1. The drilled shaft base is bearing on rock which "
    "is either massive or tightly jointed (no compressible seams or joints) "
    "to a depth of at least one diameter beneath the base, 2. It can be "
    "verified that no solution cavities or voids exist beneath the base, "
    "and 3. A clean base can be achieved and verified using conventional "
    "clean-out equipment'. Example 10-4 (pp. 10-27, 10-28) takes it with "
    "RQD 37 to 100 in the zone of the socket because 'Rock coring and local "
    "experience indicate the rock mass below the socket tip is competent': "
    "2.5 x 360 ksf = 900 ksf. ")
_CONDITIONS_JUDGED = ("Whether the three conditions hold is the engineer's "
                      "judgment (rock_base_conditions_met)")
ROCK_BASE_BASIS = (
    _EQ_10_23_CONDITIONS + "For a fractured rock mass, the Hoek-Brown "
    "base (Eqs. 10-24 to 10-29, pp. 10-23, 10-24), for which 2.5 'can be "
    "used as an upper-bound limit'. Resistance factor 0.50 (Table 8-4, p. "
    "8-14; p. 10-22). " + _CONDITIONS_JUDGED)
#: The conditions choice's basis: the conditions alone, not the base's
#: method and resistance factor, which the base choice gives (GM-230).
ROCK_CONDITIONS_BASIS = _EQ_10_23_CONDITIONS + _CONDITIONS_JUDGED

#: GEC-10 (FHWA-NHI-18-024) pp. 10-23 to 10-25 on the base of a fractured
#: rock mass, as printed (GM-251).
HOEK_BROWN_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.3, pp. 10-23, 10-24: 'For "
    "fractured rock mass, bearing capacity theory in combination with the "
    "Hoek-Brown failure criterion provides a framework for evaluating "
    "drilled shaft base resistance (Turner and Ramey, 2010)'. Eq. 10-24 "
    "(p. 10-23), sigma'1 = sigma'3 + qu [m_b (sigma'3 / qu) + s]^a (Hoek et "
    "al. 2002), qu the uniaxial compressive strength of the intact rock; "
    "Eqs. 10-25 to 10-27 (p. 10-24), m_b / m_i = exp[(GSI - 100) / 28], s = "
    "exp[(GSI - 100) / 9], a = 1/2 + (1/6)[exp(-GSI / 15) - exp(-20 / 3)]; "
    "Eq. 10-28, A = sigma'vb + qu [m_b (sigma'vb / qu) + s]^a, sigma'vb the "
    "'vertical effective stress at the socket bearing elevation (tip "
    "elevation)'; Eq. 10-29, q_BN = A + qu [m_b (A / qu) + s]^a, which "
    "'can be normalized by the rock uniaxial compressive strength to "
    "provide a bearing capacity factor N*cr as defined by Equation 10-23. "
    "A value of N*cr equal to 2.5 can be used as an upper-bound limit to "
    "base resistance calculated by Equation 10-23, unless local experience "
    "or load tests can be used to validate higher values' (p. 10-24); this "
    "tool limits it to 2.5 and gives the value past it in the choices. "
    "Fig. 10-8 (p. 10-25) draws N*cr = "
    "q_BN / qu against GSI for m_i 4 to 33 at sigma'vb = 0, up to that "
    "limit. GSI comes from the rock core descriptions (p. 10-23 points to "
    "Marinos and Hoek 2000, Hoek et al. 2013 and GEC 5) and m_i from the "
    "rock type (p. 10-24 points to GEC 5 Table 9-10): both are the "
    "engineer's. GEC-10 gives the resistance factor 0.50 (p. 10-24; Table "
    "8-4, p. 8-14, 'Eq. 10-23 or Eq. 10-29')")

#: Why the default rock base is refused where the conditions of Eq. 10-23
#: are in doubt and the Hoek-Brown base's inputs are not given (GM-244
#: review, GM-251).
ROCK_BASE_REFUSAL = (
    "GEC-10 (FHWA-NHI-18-024) p. 10-22 recommends N*cr = 2.5 (Eq. 10-23) "
    "only where the rock below the base is massive or tightly jointed (no "
    "compressible seams or joints) to at least one diameter below it, no "
    "solution cavities or voids exist beneath it, and a clean base can be "
    "achieved and verified; for a fractured rock mass it gives the "
    "Hoek-Brown base, Eqs. 10-24 to 10-29 (pp. 10-23, 10-24), from the "
    "rock's qu, GSI and m_i and the vertical effective stress at the base. "
    "Give the tip rock layer's GSI (Geological Strength Index, 0 to 100, "
    "from the core descriptions) and m_i (the Hoek-Brown constant of the "
    "intact rock, by rock type: GEC 5 Table 9-10) for that base; or "
    "rock_base_conditions_met true where the engineer judges the three "
    "conditions met (Example 10-4 takes 2.5 with RQD 37 to 100 because "
    "coring and local experience showed the rock below the tip competent); "
    "or rock_base_method 'rqd' (this tool's older reduction of q_b/q_u by "
    "RQD, not checked against a printed page)")


def hoek_brown_missing(layer: ShaftSoilLayer) -> List[str]:
    """The Hoek-Brown base's inputs a rock layer does not give (GM-251)."""
    return [n for n in ("GSI", "m_i") if getattr(layer, n) is None]

#: The earlier rock base of this tool (GM-244).
RQD_BASE_BASIS = (
    "q_b/q_u by RQD: 2.5 at RQD >= 70 %, then linear to 1.5 at 50 %, 0.8 at "
    "25 % and 0.4 at 0 %; this tool's earlier rock base, cited to GEC-10 "
    "(2010 edition) Sec. 13.3.4.4 and the Canadian Foundation Engineering "
    "Manual, neither in the reference library, so not checked against a "
    "printed page. GEC-10 (2018) gives Eq. 10-23 with N*cr = 2.5 instead")

#: The large-base reductions (GM-244).
LARGE_BASE_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) prints no reduction of base resistance for a "
    "large base: Eq. 10-14 (p. 10-14) in cohesionless soil, Eqs. 10-19 and "
    "10-20 (p. 10-19) in cohesive soil, and neither does FHWA NHI-06-089 "
    "(Eqs. 9-39, 9-40, 9-45). O'Neill and Reese (1999), not in the "
    "reference library, reduce it by 1.27/B for a base wider than 1.27 m "
    "in cohesionless soil and by F_r = 2.5 / (a B + 2.5 b) <= 1 (B in "
    "inches) for a base wider than 1.90 m in cohesive soil")

#: GEC-10 (FHWA-NHI-18-024) on phi' of cohesionless soil (GM-244).
PHI_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) Table 10-1 (p. 10-6) lists phi' among the "
    "properties to evaluate for each layer; Example 10-1 (p. 10-13) takes "
    "it from (N1)60 by phi' = 27.5 + 9.2 log (N1)60 (Chen and Kulhawy "
    "2002; Appendix A, Eq. A-2, p. A-8). A phi' given for a layer is used; "
    "the correlation only where a layer gives (N1)60 and no phi'")


#: GEC-10 (FHWA-NHI-18-024) p. 8-15, as printed.
FRACTURED_ROCK_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) p. 8-15: the amount of rock to be removed "
    "should be defined, 'including depth of rock sockets as well as the "
    "decomposed or fractured rock that may be removed but not considered "
    "part of the socket for design purposes'. No rock side resistance is "
    "taken over it; the soil above keeps its own, and the base is the "
    "rock at the tip")


#: The rock side resistance coefficient C the module takes when none is
#: given: GEC-10 Eq. 10-21 with C = 1.0, 'recommended for design of
#: "normal" rock sockets' (p. 10-20; GM-244, was 0.65, Eq. 10-22's 0.65
#: alpha_E with alpha_E = 1.0).
ROCK_SIDE_C = 1.0

#: GEC-10 (FHWA-NHI-18-024) pp. 10-20 to 10-22, as printed (GM-221,
#: GM-244).
ROCK_SIDE_C_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.3: f_SN = C pa sqrt(qu / pa), C "
    "'a regression coefficient used to analyze load test results' (Eq. "
    "10-21, p. 10-20); 'Equation 10-21 with C = 1.0 is recommended for "
    "design of \"normal\" rock sockets' (p. 10-20), a normal socket being "
    "one 'constructed with conventional equipment and resulting in "
    "nominally clean sidewalls without resorting to special procedures or "
    "artificial roughening'; rock prone to smearing or deterioration, and "
    "rock that cannot be excavated without caving, are outside it (p. "
    "10-21). In summary, 'for routine design of rock sockets, Equation "
    "10-21 with C = 1.0 is recommended' (p. 10-21); Example 10-4 (pp. "
    "10-27, 10-28) takes C = 1.0 for a socket that 'satisfies the criteria "
    "for a \"normal\" socket'. 'For highly fractured or weathered rock that "
    "cannot be drilled without some type of artificial support ... or where "
    "caving is anticipated', Eq. 10-22 (O'Neill and Reese 1999): f_SN = "
    "0.65 alpha_E pa sqrt(qu / pa), alpha_E from Table 10-3 by RQD and "
    "joint condition (1.0 at RQD 100 with closed joints, down to 0.45 at "
    "RQD 20; p. 10-21): rock_side_method 'caving'. For roughened sockets "
    "a mean C = 1.9 (Kulhawy and Prakoso 2007, p. 10-22); 'load tests or "
    "local experience' should verify C greater than 1.0. Whether a socket "
    "is normal is the engineer's judgment. Resistance factor: Table 8-4 (p. "
    "8-14) prints 0.50 for side resistance in rock by Eq. 10-21 (0.40 in "
    "uplift), which lrfd_capacity takes")

#: The values of C GEC-10 names, each with where it applies.
_ROCK_SIDE_C_NAMED = {
    1.0: "Eq. 10-21, a 'normal' socket, recommended for routine design "
         "(pp. 10-20, 10-21)",
    0.65: "the value of Eq. 10-22 (O'Neill and Reese 1999, where caving or "
          "instability of the socket is anticipated) with alpha_E = 1.0, "
          "the most Table 10-3 gives (RQD 100, closed joints); "
          "rock_side_method 'caving' takes alpha_E from Table 10-3 by each "
          "rock layer's RQD (p. 10-21)",
    1.9: "roughened sockets, verified by load tests or local experience "
         "(p. 10-22)",
}

#: The rock side resistance methods: GEC-10 Eq. 10-21 for a 'normal'
#: socket (the default) and Eq. 10-22 with Table 10-3 where caving is
#: anticipated (GM-284).
ROCK_SIDE_METHODS = ("normal", "caving")

#: GEC-10 (FHWA-NHI-18-024) p. 10-21 on Eq. 10-22, as printed (GM-284).
CAVING_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) p. 10-21: 'Rock mass that does not allow "
    "excavation of an unsupported socket without caving is also outside "
    "the \"normal\" category and may exhibit lower side resistance than "
    "given by Equation 10-21 with C = 1.0'; 'For calculating side "
    "resistance in cases where caving or instability of the socket "
    "excavation is anticipated', Eq. 10-22 (O'Neill and Reese 1999): f_SN "
    "/ pa = 0.65 alpha_E sqrt(qu / pa), 'where the coefficient alpha_E is "
    "estimated from the RQD and condition of the discontinuity surfaces "
    "of the rock mass', Table 10-3: closed joints 1.00, 0.85, 0.60, 0.50, "
    "0.45 and open or gouge-filled joints 0.85, 0.55, 0.55, 0.50, 0.45 at "
    "RQD 100, 70, 50, 30, 20 %. 'For highly fractured or weathered rock "
    "that cannot be drilled without some type of artificial support, such "
    "as casing or by the \"plug ahead\" method, or where caving is "
    "anticipated, Equation 10-22 with the reduction factors given in "
    "Table 10-3 based on RQD are recommended.' GEC-10 says nothing of "
    "values between the printed rows: taken linear between them, and held "
    "at the RQD 20 row below it. The joint condition is the engineer's, "
    "from the cores (rock_joints). Resistance factor: Table 8-4 (p. 8-14) "
    "prints 0.50 for side resistance in rock by Eq. 10-21 and no row for "
    "Eq. 10-22; p. 10-21's text gives 0.55 for both; lrfd_capacity takes "
    "the table's 0.50 for both")

#: GEC-10 (FHWA-NHI-18-024) p. 10-21 on qu and the concrete strength, as
#: printed (GM-284).
CONCRETE_FC_BASIS = (
    "GEC-10 (FHWA-NHI-18-024) p. 10-21: 'For design, the value of qu used "
    "in Equation 10-21 should not exceed the compressive strength of the "
    "drilled shaft concrete. If, however, load testing can be used to "
    "demonstrate side resistances greater than would be calculated using "
    "the concrete design strength, the load test-validated values should "
    "be allowed'. Example 10-4 (pp. 10-27, 10-28) gives both: qu 360 ksf "
    "and f'c 4,000 psi = 576 ksf. FHWA NHI-06-089 Eq. 9-47 (p. 9-140) "
    "limits Eq. 10-22's form the same way, by 0.65 pa (f'c / pa)^0.5. "
    "Taken in Eq. 10-22 as in Eq. 10-21")


def rock_side_choice(C: float, values=None) -> dict:
    """The choice made for the rock side resistance coefficient C (GM-221):
    1.0, GEC-10 Eq. 10-21 for a normal socket, by default (GM-244); the
    other values GEC-10 names are the alternatives.
    ``values``, where given, maps a C to the run's values with it, in
    words."""
    from geotech_common.choices import choice, how_given
    named = _ROCK_SIDE_C_NAMED.get(C)
    alternatives = [
        f"{c!r}: {why}" + (f": {values(c)}" if values else "")
        for c, why in _ROCK_SIDE_C_NAMED.items() if c != C]
    return choice("rock side resistance coefficient C",
                  f"{float(C)!r}: {named}" if named else f"{C:g}",
                  how_given(C, ROCK_SIDE_C, name="rock_side_C"),
                  basis=ROCK_SIDE_C_BASIS,
                  alternatives=alternatives)


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
                "fractured_rock_thickness 0: all rock below the casing is "
                "socket",
                "casing_depth to the top of competent rock: no side "
                "resistance, soil or rock, above it (GEC-10 Example 10-4, "
                "p. 10-27)", *base_alt])
    return choice(
        "rock not considered part of the socket", "none", how,
        basis=("all rock in the shaft below the casing (and below "
               "top_exclusion, where given) is socket. "
               + FRACTURED_ROCK_BASIS.split(": ", 1)[0]
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
        layer. Left out (None): 1.5 m in cohesive soil, GEC-10's 'top 5 ft'
        (p. 10-17, printed for cohesive soils), and none in cohesionless
        soil or rock (Secs. 10.3.5.1 and 10.3.5.3 print none; Examples
        10-1, 10-4 and 10-5). The permanent casing, where deeper, takes its
        place.
    bottom_exclusion : float, optional
        Side resistance left out over the bottom of the shaft (m) in
        cohesive soil. Default 0: GEC-10 (2018) prints no such zone
        (Example 10-2 takes the clay's to the base). One diameter is the
        bottom zone of FHWA NHI-06-089 Fig. 9-49 (FHWA 1999); its text
        says the bottom 5 ft (p. 9-129). See :data:`BOTTOM_ZONE_BASIS`.
    alpha_method : str, optional
        Cohesive side-resistance method: "rational" (default, GEC-10 (2018)
        Eq. 10-18, alpha = 0.30 + 0.17/(su(CIUC)/pa), on su(CIUC); see
        :data:`ALPHA_BASIS`) or "aashto" (O'Neill & Reese 1999 as FHWA
        NHI-06-089 Eqs. 9-37a, 9-37b print it, 0.55 for su/pa <= 1.5), the
        default until GM-244.
    su_test_type : str, optional
        For alpha_method="rational": what lab test the layer ``cu`` represents,
        for the CIUC transform — "ciuc" (default, no transform), "uc"
        (GEC-10 Eq. 10-16), or "uu" (Eq. 10-17).
    clay_base_method : str, optional
        Base resistance with the tip in cohesive soil: "gec10" (default,
        GEC-10 (2018) Eqs. 10-19, 10-20 with Table 10-2, su the mean of
        the cohesive soil over 2B below the base (GM-252); see
        :data:`CLAY_BASE_BASIS`) or "oneill_reese" (FHWA NHI-06-089 Eqs.
        9-39, 9-40: Nc = 6[1 + 0.2 z/D] <= 9, x 0.67 where su < 0.5 ksf,
        q <= 80 ksf, on the tip layer's cu).
    rock_base_method : str, optional
        Base resistance with the tip in rock: "gec10" (default, GEC-10
        (2018): Eq. 10-23, N*cr = 2.5, where the conditions of p. 10-22
        hold (:data:`ROCK_BASE_BASIS`), else the Hoek-Brown base of a
        fractured rock mass, Eqs. 10-24 to 10-29 on the tip rock layer's
        GSI and m_i (:data:`HOEK_BROWN_BASIS`, GM-251)) or "rqd" (q_b/q_u
        reduced by the tip layer's RQD; see :data:`RQD_BASE_BASIS`).
    rock_base_conditions_met : bool, optional
        For ``rock_base_method`` "gec10": whether the rock below the base
        meets the three conditions GEC-10 p. 10-22 sets on N*cr = 2.5
        (massive or tightly jointed to one diameter below the base, no
        cavities or voids, a clean base). True: 2.5 qu by Eq. 10-23, the
        engineer's statement. False: the Hoek-Brown base (Eqs. 10-24 to
        10-29), refused where the tip rock layer gives no GSI or m_i. Left
        out (None): 2.5 qu with the conditions taken as met, stated in the
        choices, where the layers within one base diameter below the base
        show no RQD below 100 and no soil (the Hoek-Brown base beside it
        where the tip rock layer gives GSI and m_i); where they show
        either, the Hoek-Brown base on the tip rock layer's GSI and m_i,
        refused where it gives none (:data:`ROCK_BASE_REFUSAL`).
    large_base_reduction : bool, optional
        O'Neill and Reese's (1999) reduction of base resistance for a large
        base: 1.27/B in cohesionless soil (B > 1.27 m), F_r in cohesive
        soil (B > 1.90 m). Default False: GEC-10 (2018) prints neither.
    pa : float, optional
        Atmospheric pressure (kPa) for the rational chains. Default 101.325.
    rock_side_C : float, optional
        C in the rock side resistance f_s = C pa sqrt(qu / pa) (GEC-10 Eq.
        10-21). Default 1.0, recommended for a normal socket (pp. 10-20,
        10-21; GM-244); 0.65 is Eq. 10-22 with alpha_E = 1.0, where caving
        is anticipated (the default until GM-244). See
        :data:`ROCK_SIDE_C_BASIS` (GM-221). For rock_side_method "normal".
    rock_side_method : str, optional
        "normal" (default): GEC-10 Eq. 10-21 with ``rock_side_C``, for a
        'normal' socket. "caving": Eq. 10-22, f_s = 0.65 alpha_E pa
        sqrt(qu / pa), alpha_E from Table 10-3 by each rock layer's RQD and
        ``rock_joints``, 'where caving or instability of the socket
        excavation is anticipated' (p. 10-21; :data:`CAVING_BASIS`,
        GM-284).
    rock_joints : str, optional
        For rock_side_method "caving": the joint condition of Table 10-3,
        "closed" (default) or "open" (open or gouge-filled joints).

    The rock side resistance takes qu no higher than the shaft's
    ``concrete_fc`` where it is given (GEC-10 p. 10-21,
    :data:`CONCRETE_FC_BASIS`).

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
    alpha_method: str = "rational"
    su_test_type: str = "ciuc"
    pa: float = PA
    rock_side_C: float = ROCK_SIDE_C
    K_over_Ko: float = 1.0
    top_exclusion: Optional[float] = None
    bottom_exclusion: float = 0.0
    clay_base_method: str = "gec10"
    rock_base_method: str = "gec10"
    rock_base_conditions_met: Optional[bool] = None
    large_base_reduction: bool = False
    rock_side_method: str = "normal"
    rock_joints: str = "closed"

    def __post_init__(self):
        if self.shaft is None:
            raise ValueError("Shaft must be provided")
        if not self.rock_side_C > 0:
            raise ValueError(
                f"rock_side_C must be positive, got {self.rock_side_C}")
        if self.rock_side_method not in ROCK_SIDE_METHODS:
            raise ValueError(
                f"rock_side_method must be one of {list(ROCK_SIDE_METHODS)}, "
                f"got {self.rock_side_method!r}")
        if self.rock_joints not in ROCK_JOINTS:
            raise ValueError(
                f"rock_joints must be one of {list(ROCK_JOINTS)}, got "
                f"{self.rock_joints!r}")
        if self.rock_side_method == "caving" and \
                self.rock_side_C != ROCK_SIDE_C:
            raise ValueError(
                f"rock_side_C = {self.rock_side_C:g} is the C of GEC-10 Eq. "
                "10-21; rock_side_method 'caving' is Eq. 10-22, f_s = 0.65 "
                "alpha_E pa sqrt(qu / pa) with alpha_E from Table 10-3 (p. "
                "10-21), which takes no C. Leave rock_side_C out, or name "
                "rock_side_method 'normal'")
        if self.rock_side_method == "normal" and self.rock_joints != "closed":
            raise ValueError(
                f"rock_joints = {self.rock_joints!r} is the joint condition of "
                "GEC-10 Table 10-3, which gives alpha_E in Eq. 10-22 "
                "(rock_side_method 'caving', p. 10-21); rock_side_method "
                "'normal' is Eq. 10-21, which takes no joint condition. Leave "
                "rock_joints out, or name rock_side_method 'caving'")
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
        if not self.bottom_exclusion >= 0:
            raise ValueError(
                f"bottom_exclusion must be 0 or more (m), got "
                f"{self.bottom_exclusion}")
        if self.clay_base_method not in CLAY_BASE_METHODS:
            raise ValueError(
                f"clay_base_method must be one of {list(CLAY_BASE_METHODS)}, "
                f"got {self.clay_base_method!r}")
        if self.rock_base_method not in ROCK_BASE_METHODS:
            raise ValueError(
                f"rock_base_method must be one of {list(ROCK_BASE_METHODS)}, "
                f"got {self.rock_base_method!r}")
        if self.rock_base_conditions_met not in (None, True, False):
            raise ValueError(
                "rock_base_conditions_met must be true, false or left out, "
                f"got {self.rock_base_conditions_met!r}")
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

        phi' is the layer's where given, else from (N1)60 (Example 10-1;
        Appendix A Eq. A-2); OCR is the layer's, or sigma'p (Eq. 10-11 or 10-12, from
        N60) over sigma_v_ref or the layer's sigma'v, that sigma'v not less
        than the sigma'v at 7.5 ft (pp. 10-11, 10-12); delta = phi'.
        """
        # a phi' given wins; (N1)60 only where it is not (GM-244)
        if layer.phi > 0:
            steps = {"phi_deg": layer.phi, "phi_from": "phi"}
            if layer.N1_60 > 0:
                steps["phi_from_N1_60_deg"] = phi_prime_from_N1_60(
                    layer.N1_60)
        else:
            steps = {"phi_deg": phi_prime_from_N1_60(layer.N1_60),
                     "phi_from": "N1_60"}
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
        else 1.5 m in cohesive soil (GEC-10 p. 10-17, printed for cohesive
        soils) and none in cohesionless soil or rock (Secs. 10.3.5.1 and
        10.3.5.3; Examples 10-1, 10-4 and 10-5)."""
        if self.top_exclusion is not None:
            return self.top_exclusion
        return TOP_EXCLUSION if soil_type == "cohesive" else 0.0

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
        self._check_hoek_brown_layers([self.shaft.length], below=True)
        return self._compute()

    def _check_hoek_brown_layers(self, lengths, below: bool) -> None:
        """Refuse GSI or m_i on a rock layer no base at ``lengths`` takes
        them from (GM-251 review; as GM-179 and GM-249, an input a layer
        would not use is refused, not dropped). The Hoek-Brown base takes
        the GSI and m_i of the rock layer the base is in; the single-length
        result (``below``) also gives the base on each rock layer within
        one base diameter below it, among the choices' alternatives."""
        layers = self.soil.layers
        used, tips = set(), set()
        for L in lengths:
            i = self._tip_index(L)
            tips.add(i)
            if layers[i].soil_type != "rock":
                continue
            used.add(i)
            if below:
                B = self.shaft.base_diameter(True)
                used.update(j for j, lay, _, _ in
                            self._layers_below_base(L, B)
                            if lay.soil_type == "rock")
        if self.rock_base_conditions_met is not None and not any(
                layers[i].soil_type == "rock" for i in tips):
            if len(lengths) > 1:
                where = (f"no trial base from {min(lengths):g} to "
                         f"{max(lengths):g} m is in rock")
            else:
                where = (f"the base at {lengths[0]:g} m is in "
                         f"{layers[min(tips)].soil_type} soil")
            raise ValueError(
                "The shaft does not take rock_base_conditions_met: it is "
                "the engineer's judgment on the three conditions GEC-10 "
                "(2018) p. 10-22 sets on a base in rock (q_BN = 2.5 qu, "
                f"Eq. 10-23), and {where}. Leave it out, or give the shaft "
                "length that puts the base in rock.")
        top = 0.0
        for i, layer in enumerate(layers):
            given = [n for n in ("GSI", "m_i")
                     if getattr(layer, n) is not None]
            span = f"{top:g}-{top + layer.thickness:g} m"
            top += layer.thickness
            if not given or i in used:
                continue
            if len(lengths) > 1:
                where = (f"no trial base from {min(lengths):g} to "
                         f"{max(lengths):g} m is in it")
            else:
                L = lengths[0]
                tip = layers[min(tips)]
                if tip.soil_type != "rock":
                    where = (f"the base at {L:g} m is in {tip.soil_type} "
                             "soil")
                elif i < min(tips):
                    where = f"it is above the base at {L:g} m"
                else:
                    where = (f"it is more than one base diameter "
                             f"({self.shaft.base_diameter(True):g} m) below "
                             f"the base at {L:g} m")
            raise ValueError(
                f"The rock layer at {span} does not take "
                f"{' and '.join(given)}: they give GEC-10's base for a "
                "fractured rock mass, the Hoek-Brown base (Eqs. 10-24 to "
                "10-29), which takes the GSI and m_i of the rock layer the "
                "base is in and shows the rock within one base diameter "
                f"below the base beside it, and {where}. Leave them out of "
                "that layer, or give the shaft length that puts the base in "
                "it.")

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

        # Exclusion zones: the top zone by soil type (_top_of_side); over
        # the bottom in cohesive soil only where bottom_exclusion is given
        # (GM-244); rock not considered part of the socket (GEC-10 p.
        # 8-15, GM-220)
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

            # The bottom zone, cohesive soil only, where given (GM-244)
            z_bottom_eff = z_bottom
            if layer.soil_type == "cohesive" and self.bottom_exclusion > 0:
                z_bottom_eff = min(z_bottom, L - self.bottom_exclusion)

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
                    # GEC-10 (2018) Eq. 10-18 on su(CIUC), a UC or UU
                    # strength transformed by Eq. 10-16 or 10-17 (p. 10-16)
                    su = su_to_ciuc(layer.cu, sigma_v, self.su_test_type)
                    alpha = alpha_cohesive_rational(su, self.pa)
                    method_used = f"alpha={alpha:.3f} (rational)"
                else:
                    su = layer.cu
                    alpha = alpha_cohesive(layer.cu)
                    method_used = f"alpha={alpha:.3f}"
                Qs_layer = side_resistance_cohesive(
                    su, shaft.perimeter, effective_thickness, alpha)
                fs = alpha * su
                alpha_steps = {"alpha_method": self.alpha_method,
                               "su_kPa": layer.cu, "su_used_kPa": su,
                               "su_over_pa": su / self.pa, "alpha": alpha}
                if self.alpha_method == "rational":
                    alpha_steps["su_test_type"] = self.su_test_type
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
                # Eq. 10-21 or 10-22, qu no higher than f'c (GM-284)
                rock_steps = self._rock_side_steps(layer)
                Qs_layer = side_resistance_rock(
                    rock_steps["qu_used_kPa"], shaft.socket_perimeter,
                    effective_thickness, C=rock_steps["C"],
                    alpha_E=rock_steps["alpha_E"])
                fs = rock_steps["C"] * rock_steps["alpha_E"] * (
                    rock_steps["qu_used_kPa"] * PA) ** 0.5
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
                row["beta_steps"] = _rounded(steps)
            if layer.soil_type == "cohesive":
                row["alpha_steps"] = _rounded(alpha_steps)
            elif layer.soil_type == "rock":
                row["rock_steps"] = _rounded(rock_steps)
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

        # Each step of the unit base resistance, for the result and the
        # calc package (GM-148, GM-244)
        if tip_layer.soil_type == "cohesive":
            su_b, su_from, zone = self._base_su(tip_layer, layer_results,
                                                base_D)
            u = unit_end_bearing_cohesive(
                su_b, L_over_D, base_diameter=base_D,
                shaft_length=L, method=self.clay_base_method,
                large_base_reduction=self.large_base_reduction)
        elif tip_layer.soil_type == "cohesionless":
            # N60 the mean over 2B below the base (p. 10-15, GM-284)
            n_zone = self._n60_zone(tip_layer, base_D)
            u = unit_end_bearing_cohesionless(n_zone["N60"], base_D,
                                              self.large_base_reduction)
        else:  # rock
            u = self._rock_base(tip_layer, L, base_D)
        base_steps = {"soil_type": tip_layer.soil_type,
                      "base_diameter_m": base_D, **u}
        if tip_layer.soil_type == "cohesive":
            base_steps.update(su_kPa=su_b, su_from=su_from)
            if zone is not None:
                base_steps.update(_zone_steps(zone))
        elif tip_layer.soil_type == "cohesionless":
            base_steps.update(_n60_zone_steps(n_zone))
        Qt = u["qb"] * base_area

        Q_ultimate = total_Qs + Qt
        Q_allowable = Q_ultimate / self.factor_of_safety

        choices = []
        if report:
            choices = (
                self._side_choices(layer_results, Qs_sand, Q_ultimate)
                + self._clay_side_choices(layer_results, Qs_clay,
                                          Q_ultimate)
                + ([side_n60_choice(no_n60)] if no_n60 else [])
                + ([fractured_rock_choice(shaft, self.soil)]
                   if in_rock or shaft.fractured_rock_thickness else [])
                + (self._rock_side_choices(layer_results, Qs_rock, Q_ultimate)
                   if in_rock else [])
                + self._base_choices(tip_layer, base_steps, Qt, Q_ultimate))

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
            base_steps=_rounded(base_steps, sig=("s",)),
            choices=choices,
        )

    def _tip_index(self, L: float) -> int:
        """Index of the layer the base is in (``layer_at_depth(L - 0.01)``,
        the last layer below the profile)."""
        top = 0.0
        for i, layer in enumerate(self.soil.layers):
            if top <= L - 0.01 < top + layer.thickness:
                return i
            top += layer.thickness
        return len(self.soil.layers) - 1

    def _layers_below_base(self, L: float, depth: float):
        """(index, layer, top, bottom) of each layer within ``depth`` below
        the base at ``L``, its part there in m. The last layer is taken to
        go on below the profile."""
        out = []
        top = 0.0
        layers = self.soil.layers
        for i, layer in enumerate(layers):
            bottom = top + layer.thickness
            if i == len(layers) - 1:
                bottom = max(bottom, L + depth)
            if bottom > L + 1e-9 and top < L + depth - 1e-9:
                out.append((i, layer, max(top, L), min(bottom, L + depth)))
            top += layer.thickness
        return out

    def _rock_below_base(self, L: float, B: float) -> List[str]:
        """What the layers within one base diameter ``B`` below the base at
        ``L`` show against the first condition of Eq. 10-23 (GEC-10 p.
        10-22, 'massive or tightly jointed ... to a depth of at least one
        diameter beneath the base'): each rock layer there with an RQD
        below 100, and each soil layer, in words. The last layer is taken
        to go on below the profile."""
        out = []
        for _, layer, a, b in self._layers_below_base(L, B):
            where = f"{a:g}-{b:g} m"
            if layer.soil_type != "rock":
                out.append(f"{layer.soil_type} soil at {where}")
            elif layer.RQD < 100:
                out.append(f"rock with RQD {layer.RQD:g} % at {where}")
        return out

    def _rock_base_equation(self, tip_layer, L: float, B: float) -> str:
        """Which of GEC-10's rock bases applies under ``rock_base_method``
        "gec10": "10-23" (N*cr = 2.5) where the call states the conditions
        of p. 10-22 met, or leaves them out and nothing within one base
        diameter ``B`` below the base is against them, GSI and m_i given or
        not (p. 10-22 recommends 2.5 for rock that meets them; the
        Hoek-Brown base is 'For fractured rock mass', p. 10-23); "10-29"
        (the Hoek-Brown base, GM-251) where the call states them not met,
        or leaves them out and the layers there show an RQD below 100 or
        soil. Refused where the Hoek-Brown base is needed and the tip rock
        layer does not give its inputs."""
        missing = hoek_brown_missing(tip_layer)
        if self.rock_base_conditions_met:
            return "10-23"
        if self.rock_base_conditions_met is False:
            if missing:
                raise ValueError(
                    "rock_base_conditions_met is false, and the base GEC-10 "
                    "gives for a fractured rock mass, the Hoek-Brown base "
                    "(Eqs. 10-24 to 10-29, pp. 10-23, 10-24), needs the tip "
                    f"rock layer's {' and '.join(missing)}, which it does "
                    "not give. " + ROCK_BASE_REFUSAL)
            return "10-29"
        below = self._rock_below_base(L, B)
        if below and not missing:
            return "10-29"
        if below:
            raise ValueError(
                f"the base at {L:g} m bears on rock, and within one "
                f"base diameter ({B:g} m) below it the layers give "
                + "; ".join(below) + ". Whether that rock meets the "
                "conditions of Eq. 10-23 is the engineer's call, and "
                "the call does not say; the tip rock layer gives no "
                + " and no ".join(missing) + " for the Hoek-Brown base. "
                + ROCK_BASE_REFUSAL)
        return "10-23"

    def _rock_base(self, tip_layer, L: float, B: float) -> dict:
        """The unit base resistance in rock and each step to it: by RQD
        under ``rock_base_method`` "rqd"; under "gec10" Eq. 10-23 or the
        Hoek-Brown base, Eqs. 10-24 to 10-29, on sigma'vb at the base
        (:meth:`_rock_base_equation`, GM-251)."""
        out = {"method": self.rock_base_method, "qu_kPa": tip_layer.qu,
               "RQD": tip_layer.RQD}
        if self.rock_base_method == "rqd":
            factor = rock_base_factor(tip_layer.RQD, "rqd")
            return {**out, "factor": factor, "qb": factor * tip_layer.qu}
        eq = self._rock_base_equation(tip_layer, L, B)
        if eq == "10-23":
            return {**out, "equation": eq, "factor": 2.5,
                    "qb": 2.5 * tip_layer.qu}
        return {**out, "equation": eq, **hoek_brown_base(
            tip_layer.qu, tip_layer.GSI, tip_layer.m_i,
            self.soil.effective_stress_at_depth(L))}

    def _rock_side_steps(self, layer) -> dict:
        """Each step of a rock layer's unit side resistance (GM-284): qu as
        given and as used, no higher than the shaft's concrete_fc where it
        is given (GEC-10 p. 10-21); the equation, C and alpha_E: Eq. 10-21
        with rock_side_C and alpha_E 1, or Eq. 10-22 with 0.65 and alpha_E
        from Table 10-3 at the layer's RQD and rock_joints."""
        fc = self.shaft.concrete_fc
        qu = layer.qu if fc is None else min(layer.qu, fc)
        out = {"qu_kPa": layer.qu, "qu_used_kPa": qu,
               "qu_limited": qu < layer.qu}
        if self.rock_side_method == "caving":
            out.update(equation="10-22", C=CAVING_C, RQD=layer.RQD,
                       rock_joints=self.rock_joints,
                       alpha_E=alpha_E_table_10_3(layer.RQD,
                                                  self.rock_joints),
                       below_table=layer.RQD < TABLE_10_3[-1][0])
        else:
            out.update(equation="10-21", C=self.rock_side_C, alpha_E=1.0)
        return out

    def _rock_side_choices(self, layer_results, Qs_rock: float,
                           Q_ultimate: float) -> List[dict]:
        """The choices made for side resistance in rock (GM-221, GM-284):
        the method (Eq. 10-21 for a normal socket, or Eq. 10-22 where
        caving is anticipated), then Eq. 10-21's C or Eq. 10-22's joint
        condition, then qu against the concrete strength."""
        from geotech_common.choices import choice, how_given
        rows = [lb for lb in layer_results if "rock_steps" in lb]
        here = f"Q_s in rock {Qs_rock:,.0f} kN, Q_ult {Q_ultimate:,.0f} kN"
        out = []
        how = how_given(self.rock_side_method, "normal",
                        name="rock_side_method")
        if self.rock_side_method == "normal":
            out.append(choice(
                "rock side resistance method",
                "Eq. 10-21, f_s = C pa sqrt(qu / pa), for a 'normal' socket: "
                + here, how,
                basis=ROCK_SIDE_C_BASIS,
                alternatives=[
                    "rock_side_method 'caving', Eq. 10-22, f_s = 0.65 "
                    "alpha_E pa sqrt(qu / pa), alpha_E from Table 10-3 by "
                    "each rock layer's RQD, closed joints, where caving or "
                    "instability of the socket is anticipated: "
                    + self._alternative(rock_side_method="caving",
                                        rock_side_C=ROCK_SIDE_C,
                                        part="rock")]))
            out.append(rock_side_choice(
                self.rock_side_C,
                lambda c: self._alternative(rock_side_C=c, part="rock")))
        else:
            spans = "; ".join(
                f"{lb['depth_top_m']:g}-{lb['depth_bottom_m']:g} m RQD "
                f"{lb['rock_steps']['RQD']:g} %: alpha_E "
                f"{lb['rock_steps']['alpha_E']:.3f}" for lb in rows)
            basis = CAVING_BASIS
            low = [lb for lb in rows if lb["rock_steps"]["below_table"]]
            if low:
                basis += (". RQD below 20 % (" + _spans(low) + ") is below "
                          "Table 10-3: alpha_E is held at 0.45, the RQD 20 "
                          "row; GEC-10 gives no value there, the engineer's "
                          "judgment")
            out.append(choice(
                "rock side resistance method",
                "Eq. 10-22, f_s = 0.65 alpha_E pa sqrt(qu / pa), where "
                "caving or instability of the socket is anticipated, "
                f"alpha_E from Table 10-3 with {self.rock_joints} joints"
                + (f" ({spans})" if spans else "") + ": " + here, how,
                basis=basis,
                alternatives=[
                    "rock_side_method 'normal', Eq. 10-21 with C = 1.0, a "
                    "'normal' socket, recommended for routine design (pp. "
                    "10-20, 10-21): "
                    + self._alternative(rock_side_method="normal",
                                        part="rock")]))
            other = "open" if self.rock_joints == "closed" else "closed"
            words = {"closed": "closed joints",
                     "open": "open or gouge-filled joints"}
            out.append(choice(
                "joint condition in Table 10-3", words[self.rock_joints],
                how_given(self.rock_joints, "closed", name="rock_joints"),
                basis=("GEC-10 (FHWA-NHI-18-024) Table 10-3 (p. 10-21) "
                       "gives alpha_E by RQD for closed joints and for open "
                       "or gouge-filled joints; 'the coefficient alpha_E is "
                       "estimated from the RQD and condition of the "
                       "discontinuity surfaces of the rock mass'. Which "
                       "applies is the engineer's, from the cores"),
                alternatives=[
                    f"rock_joints '{other}', {words[other]}: "
                    + self._alternative(rock_joints=other, part="rock")]))
        out.append(self._concrete_fc_choice(rows, here))
        return out

    def _concrete_fc_choice(self, rows, here: str) -> dict:
        """The choice made for qu in the rock side resistance against the
        concrete strength (GEC-10 p. 10-21, GM-284); ``rows`` the rock
        layers' results."""
        from geotech_common.choices import choice, how_given
        fc = self.shaft.concrete_fc
        how = how_given(fc, None, name="concrete_fc")
        top = max((lb["rock_steps"]["qu_kPa"] for lb in rows), default=0.0)
        if fc is None:
            return choice(
                "qu in the rock side resistance, against the concrete "
                "strength", "not checked, as the call gives no concrete_fc: "
                f"qu as given (up to {top:,.0f} kPa)", how,
                basis=CONCRETE_FC_BASIS,
                alternatives=["concrete_fc, the shaft concrete's f'c (kPa): "
                              "qu no higher than it in the rock side "
                              "resistance"])
        limited = [lb for lb in rows if lb["rock_steps"]["qu_limited"]]
        if limited:
            chosen = (f"qu limited to f'c = {fc:,.0f} kPa in "
                      + "; ".join(
                          f"{lb['depth_top_m']:g}-{lb['depth_bottom_m']:g} m "
                          f"(qu {lb['rock_steps']['qu_kPa']:,.0f} kPa)"
                          for lb in limited) + ": " + here)
            alternatives = [
                "qu as given, a load test showing the higher side "
                "resistance (p. 10-21): "
                + self._alternative(
                    shaft=replace(self.shaft, concrete_fc=None),
                    part="rock")]
        else:
            chosen = (f"no limit binds: f'c = {fc:,.0f} kPa is not below "
                      f"the rock's qu (up to {top:,.0f} kPa)")
            alternatives = []
        return choice("qu in the rock side resistance, against the "
                      "concrete strength", chosen, how,
                      basis=CONCRETE_FC_BASIS, alternatives=alternatives)

    def _rock_conditions_choice(self, B: float, equation: str) -> dict:
        """The choice made for the conditions of Eq. 10-23 (GEC-10 p.
        10-22) under the default rock base: stated by the call, or, where
        it does not say, taken as met where the layers within one base
        diameter ``B`` below the base show nothing against them (GM-244
        review) and not taken as met where they show an RQD below 100 or
        soil (the Hoek-Brown base, GM-251)."""
        from geotech_common.choices import choice, how_given
        below = self._rock_below_base(self.shaft.length, B)
        below_words = ("; the layers within one base diameter below the "
                       "base give " + "; ".join(below)) if below else ""
        met = self.rock_base_conditions_met
        hb = "the Hoek-Brown base of a fractured rock mass (Eqs. 10-24 to 10-29)"
        tip = self.soil.layers[self._tip_index(self.shaft.length)]
        if met:
            chosen = ("met, as the call states: the rock below the base is "
                      "massive or tightly jointed to one diameter below "
                      "it, with no cavities or voids, and a clean base"
                      + below_words)
        elif met is False:
            chosen = ("not met, as the call states: the base is " + hb
                      + below_words)
        elif equation == "10-29":
            chosen = ("not taken as met, as the call does not say and the "
                      f"layers within one base diameter ({B:g} m) below "
                      "the base give " + "; ".join(below) + ": the base is "
                      + hb + " on the tip rock layer's GSI and m_i, which "
                      "2.5 qu bounds above (p. 10-24)")
        else:
            chosen = ("taken as met, as the call does not say: the layers "
                      f"within one base diameter ({B:g} m) below the base "
                      "give no RQD below 100 (a rock layer's RQD is 100 "
                      "where none is given) and no soil; no cavities or "
                      "voids and a clean base are assumed")
            if not hoek_brown_missing(tip):
                chosen += (
                    f"; the tip rock layer's GSI {tip.GSI:g} and m_i "
                    f"{tip.m_i:g} give {hb}, GEC-10's base for rock that "
                    "does not meet the conditions (p. 10-23), beside it as "
                    "the alternative")
        alternatives = []
        if equation == "10-29":
            alternatives.append(
                "rock_base_conditions_met true, the engineer judging the "
                "three conditions met: 2.5 qu by Eq. 10-23: "
                + self._alternative(rock_base_conditions_met=True,
                                    part="base"))
        else:
            alternatives.append(
                "rock_base_conditions_met false, the rock below the base "
                "fractured: " + self._hoek_brown_alternative())
        alternatives.append(
            "side resistance alone, the base neglected: AASHTO Article "
            "C.10.8.3.5.4a as GEC-10 quotes it (p. 10-25), for 'cases in "
            "which the base of the drilled hole cannot be cleaned and "
            "inspected'; the result's Q_skin is that resistance")
        return choice(
            "rock below the base (the conditions of Eq. 10-23)", chosen,
            how_given(met, None, name="rock_base_conditions_met"),
            basis=ROCK_CONDITIONS_BASIS, alternatives=alternatives)

    def _hoek_brown_alternative(self) -> str:
        """The Hoek-Brown base as an alternative, in words: its values on
        the tip rock layer's GSI and m_i, or the inputs it needs."""
        tip = self.soil.layers[self._tip_index(self.shaft.length)]
        missing = hoek_brown_missing(tip)
        words = ("the Hoek-Brown base, Eqs. 10-24 to 10-29 (pp. 10-23, "
                 "10-24)")
        if missing:
            return (words + ", which needs the tip rock layer's "
                    + " and ".join(missing) + " (not given)")
        return (f"{words}, on the tip rock layer's GSI {tip.GSI:g} and m_i "
                f"{tip.m_i:g}: "
                + self._alternative(rock_base_method="gec10",
                                    rock_base_conditions_met=False,
                                    part="base"))

    def _hoek_brown_below(self, B: float, Q_side: float, area: float,
                          skip: Optional[tuple] = None):
        """(below, alternatives): each layer within one base diameter ``B``
        below the base other than the one the base is in, in words, and
        the Hoek-Brown base (GEC-10 Eqs. 10-24 to 10-29) on each rock layer
        there that gives GSI and m_i, with its Q_b and Q_ult (``Q_side``
        plus that Q_b on the base ``area``); a layer whose (GSI, m_i, qu)
        is ``skip``, the rock the base already takes, is passed over
        (GM-251)."""
        L = self.shaft.length
        tip_i = self._tip_index(L)
        sigma_vb = self.soil.effective_stress_at_depth(L)
        below, alternatives = [], []
        for i, layer, a, b in self._layers_below_base(L, B):
            if i == tip_i:
                continue
            where = f"{a:g}-{b:g} m"
            missing = hoek_brown_missing(layer)
            if layer.soil_type != "rock":
                below.append(f"{layer.soil_type} soil at {where}")
            elif missing:
                below.append(f"rock at {where} with no "
                             + " and no ".join(missing))
            elif (layer.GSI, layer.m_i, layer.qu) != skip:
                rock = (f"rock at {where} with GSI {layer.GSI:g}, m_i "
                        f"{layer.m_i:g}, qu {layer.qu:,.0f} kPa")
                below.append(rock)
                hb = hoek_brown_base(layer.qu, layer.GSI, layer.m_i,
                                     sigma_vb)
                Qb = hb["qb"] * area
                alternatives.append(
                    f"the Hoek-Brown base on the {rock}: N*cr "
                    f"{hb['factor']:.3f}, Q_b {Qb:,.0f} kN, Q_ult "
                    f"{Q_side + Qb:,.0f} kN")
        return below, alternatives

    def _hoek_brown_below_alternatives(self, tip_layer, B: float, Qt: float,
                                       Q_ultimate: float,
                                       qb: float) -> List[str]:
        """Where the base takes Eq. 10-23 or the 'rqd' base: the Hoek-Brown
        base on each other rock layer within one base diameter ``B`` below
        the base that gives GSI and m_i (:meth:`_hoek_brown_below`), so a
        GSI and m_i given there are shown, not dropped (GM-251 review)."""
        skip = (None if hoek_brown_missing(tip_layer)
                else (tip_layer.GSI, tip_layer.m_i, tip_layer.qu))
        area = Qt / qb if qb > 0 else 0.0
        _, others = self._hoek_brown_below(B, Q_ultimate - Qt, area,
                                           skip=skip)
        L = self.shaft.length
        tip_i = self._tip_index(L)
        for i, layer, a, b in self._layers_below_base(L, B):
            missing = hoek_brown_missing(layer)
            if i != tip_i and len(missing) == 1:
                others.append(
                    f"the Hoek-Brown base on the rock at {a:g}-{b:g} m, "
                    f"which needs its {missing[0]} (not given)")
        return ["the base of a fractured rock mass, with the rock below the "
                "base governing (GEC-10 gives no depth for it): " + a
                for a in others]

    def _clay_base_with(self, su: float) -> float:
        """Q_b (kN) of the base in cohesive soil by this analysis's method
        at ``su`` (kPa), for a choice's alternatives (GM-252)."""
        L = self.shaft.length
        u = unit_end_bearing_cohesive(
            su, L / self.shaft.diameter,
            base_diameter=self.shaft.base_diameter(False), shaft_length=L,
            method=self.clay_base_method,
            large_base_reduction=self.large_base_reduction)
        return u["qb"] * self.shaft.base_area(False)

    def _hoek_brown_choice(self, steps: dict, Qt: float, Q_ultimate: float,
                           here: str):
        """(chosen, basis, alternatives) of the choice made for the base
        in rock where it is the Hoek-Brown base of a fractured rock mass,
        GEC-10 Eqs. 10-24 to 10-29 (GM-251). The alternatives: Eq. 10-23
        with the conditions stated met; N*cr past 2.5 where it is limited;
        the rock within one base diameter below the base on its own GSI and
        m_i, where a layer there gives them."""
        B = steps["base_diameter_m"]
        qu = steps["qu_kPa"]
        area = Qt / steps["qb"] if steps["qb"] > 0 else 0.0
        chosen = (
            "GEC-10 (2018) Eqs. 10-24 to 10-29, the Hoek-Brown base of a "
            f"fractured rock mass (Turner and Ramey 2010): GSI "
            f"{steps['GSI']:g}, m_i {steps['m_i']:g}, qu {qu:,.0f} kPa, "
            f"sigma'vb {steps['sigma_vb_kPa']:.1f} kPa; m_b "
            f"{steps['m_b']:.3f}, s {steps['s']:.3g}, a {steps['a']:.3f} "
            f"(Eqs. 10-25 to 10-27), A {steps['A_kPa']:,.0f} kPa (Eq. "
            f"10-28), q_BN {steps['qb_uncapped']:,.0f} kPa (Eq. 10-29), N*cr "
            f"= q_BN / qu = {steps['Ncr_uncapped']:.3f}"
            + (", limited to 2.5 (p. 10-24)" if steps["capped"] else "")
            + ": " + here)
        basis = HOEK_BROWN_BASIS
        alternatives = [
            "rock_base_conditions_met true, Eq. 10-23, q_BN = 2.5 qu, where "
            "the engineer judges the rock below the base massive or tightly "
            "jointed to one diameter below it, with no cavities or voids "
            "and a clean base (p. 10-22): "
            + self._alternative(rock_base_conditions_met=True, part="base")]
        if steps["capped"]:
            Qb = steps["qb_uncapped"] * area
            alternatives.append(
                f"N*cr {steps['Ncr_uncapped']:.2f}, not limited to 2.5, where "
                "'local experience or load tests' validate it (p. 10-24): "
                f"Q_b {Qb:,.0f} kN, Q_ult {Q_ultimate - Qt + Qb:,.0f} kN")
        # the other layers within one base diameter below the base: the
        # depth of the first condition of Eq. 10-23 (p. 10-22)
        below, others = self._hoek_brown_below(
            B, Q_ultimate - Qt, area, skip=(steps["GSI"], steps["m_i"], qu))
        alternatives += others
        if below:
            basis += (
                ". GSI, m_i and qu are the tip layer's, the rock the base is "
                "in; within one base diameter below the base the profile "
                "gives " + "; ".join(below) + ". GEC-10 gives no depth "
                "below the base over which the rock mass is taken: which "
                "rock governs is the engineer's judgment")
        return chosen, basis, alternatives

    def _alternative(self, sublayer: Optional[float] = None,
                     correlation: Optional[str] = None,
                     part: str = "sand", soil=None, **changes) -> str:
        """The ``part`` of the capacity (:data:`_PARTS`, side resistance
        in cohesionless soil by default) and the ultimate capacity with
        ``changes`` to this analysis, in words; or why it is refused.
        ``correlation`` gives every cohesionless layer that takes sigma'p
        from N60 that ``sigma_p_correlation``; ``soil`` replaces the
        profile."""
        soil = self.soil if soil is None else soil
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
        return _values(r, part)

    def _top_zone_choice(self, layer_results) -> Optional[dict]:
        """The choice made for side resistance near the ground surface
        (GM-241), or None where it changes nothing: with top_exclusion
        left out, cohesionless soil and rock in the top 1.5 m below the
        casing keep their side resistance (GM-244); given, the top zone
        takes the layers it reaches below the casing."""
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
            kept = [lb for lb in layer_results
                    if lb["soil_type"] != "cohesive"
                    and reached(lb, TOP_EXCLUSION)]
            if not kept:
                return None
            others = [lb for lb in layer_results
                      if lb["soil_type"] == "cohesive"
                      and reached(lb, TOP_EXCLUSION)]
            kinds = " and ".join(dict.fromkeys(
                "rock" if lb["soil_type"] == "rock" else "cohesionless soil"
                for lb in kept))
            chosen = (f"side resistance taken in {kinds} within the top "
                      f"{TOP_EXCLUSION:g} m ({spans(kept, TOP_EXCLUSION)})")
            if others:
                chosen += (f"; none in cohesive soil there "
                           f"({spans(others, TOP_EXCLUSION)})")
            alternatives = [
                f"top_exclusion {TOP_EXCLUSION:g}, the top 5 ft of p. 10-17 "
                "left out in every layer: "
                + self._alternative(top_exclusion=TOP_EXCLUSION,
                                    part="side"),
                "top_exclusion the depth of seasonal moisture change or of "
                "frost, where deeper (p. 10-17)"]
        else:
            cut = [lb for lb in layer_results
                   if reached(lb, self.top_exclusion)]
            if not cut:
                return None
            chosen = (f"none over the top {self.top_exclusion:g} m, in "
                      f"every layer ({spans(cut, self.top_exclusion)})")
            left_out = self._alternative(top_exclusion=None, part="side")
            alternatives = [
                f"top_exclusion left out: none over the top "
                f"{TOP_EXCLUSION:g} m in cohesive soil, side resistance "
                "from the top in cohesionless soil and rock: " + left_out]
            zero = self._alternative(top_exclusion=0.0, part="side")
            if self.top_exclusion != 0 and zero != left_out:
                alternatives.append(
                    "top_exclusion 0, side resistance from the ground "
                    "surface in every layer: " + zero)
        return choice(
            "side resistance near the ground surface", chosen,
            how_given(self.top_exclusion, None, name="top_exclusion"),
            basis=top_exclusion_basis(
                {lb["soil_type"] for lb in layer_results}),
            alternatives=alternatives)

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
        phi = self._phi_choice(sand, steps)
        if phi:
            out.append(phi)
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

    def _base_su(self, tip_layer, layer_results, B: float):
        """(su kPa, where the tip layer's came from, the zone) for the base
        in cohesive soil. Under GEC-10's base, su is the mean of the
        cohesive soil over 2B below the base (p. 10-19; GM-252,
        :meth:`_su_zone`); under O'Neill and Reese's, the tip layer's
        (zone None)."""
        su_tip, su_from = self._tip_su(tip_layer, layer_results)
        if self.clay_base_method != "gec10":
            return su_tip, su_from, None
        zone = self._su_zone(B, su_tip, su_from == "su_ciuc")
        return zone["su_kPa"], su_from, zone

    def _su_zone(self, B: float, su_tip: float, transform: bool) -> dict:
        """The mean su (kPa) of the cohesive soil between the base and 2B
        below it, by thickness (GEC-10 p. 10-19: 'su = mean undrained shear
        strength of the cohesive soil over a depth of 2B below the base';
        GM-252). The layer the base is in counts at ``su_tip``, as its side
        resistance took it; each cohesive layer below at its cu, made
        su(CIUC) by Eq. 10-16 or 10-17 on sigma'vo at the middle of its
        part within 2B where ``transform``. A cohesionless or rock layer
        has no su and is left out; the profile is not carried on below its
        last layer, so where it ends less than 2B below the base the mean
        is over what it gives, and where nothing cohesive is given there
        su is ``su_tip``."""
        L = self.shaft.length
        z2 = L + 2 * B
        tip_i = self._tip_index(L)
        rows, top = [], 0.0
        for i, layer in enumerate(self.soil.layers):
            bottom = top + layer.thickness
            a, b = max(top, L), min(bottom, z2)
            top = bottom
            if b <= a + 1e-9:
                continue
            if layer.soil_type != "cohesive":
                su = None
            elif i == tip_i:
                su = su_tip
            elif transform:
                su = su_to_ciuc(
                    layer.cu, self.soil.effective_stress_at_depth((a + b) / 2),
                    self.su_test_type)
            else:
                su = layer.cu
            rows.append({"top_m": a, "bottom_m": b,
                         "soil_type": layer.soil_type, "su_kPa": su})
        clay = [r for r in rows if r["su_kPa"] is not None]
        t = sum(r["bottom_m"] - r["top_m"] for r in clay)
        su = (sum(r["su_kPa"] * (r["bottom_m"] - r["top_m"]) for r in clay)
              / t if t > 0 else su_tip)
        return {"top_m": L, "bottom_m": z2,
                "profile_bottom_m": self.soil.total_thickness,
                "rows": rows, "su_kPa": su, "su_tip_kPa": su_tip}

    def _n60_zone(self, tip_layer, B: float) -> dict:
        """N60 of Eq. 10-14 with the base in cohesionless soil (GM-284):
        'the average value between the base and two diameters beneath the
        base' (GEC-10 p. 10-15), the mean by thickness of the cohesionless
        layers' N60 between the base and 2B below it. A layer there that
        gives no N60, or is cohesive soil or rock, is left out; the
        profile is not carried on below its last layer, so where it ends
        less than 2B below the base the mean is over what it gives, and
        where it gives nothing there N60 is the tip layer's. Eq. 10-14
        limits the mean to 50 (:func:`unit_end_bearing_cohesionless`)."""
        L = self.shaft.length
        z2 = L + 2 * B
        rows, top = [], 0.0
        for layer in self.soil.layers:
            bottom = top + layer.thickness
            a, b = max(top, L), min(bottom, z2)
            top = bottom
            if b <= a + 1e-9:
                continue
            n = (layer.N60 if layer.soil_type == "cohesionless"
                 and layer.N60 > 0 else None)
            rows.append({"top_m": a, "bottom_m": b,
                         "soil_type": layer.soil_type, "N60": n})
        used = [r for r in rows if r["N60"] is not None]
        t = sum(r["bottom_m"] - r["top_m"] for r in used)
        N60 = (sum(r["N60"] * (r["bottom_m"] - r["top_m"]) for r in used)
               / t if t > 0 else tip_layer.N60)
        return {"top_m": L, "bottom_m": z2,
                "profile_bottom_m": self.soil.total_thickness,
                "rows": rows, "N60": N60, "N60_tip": tip_layer.N60}

    def _tip_su(self, tip_layer, layer_results):
        """su (kPa) of the layer the base is in and where it came from:
        under GEC-10's base and alpha with cu a UC or UU strength, its
        su(CIUC) as its side resistance took it ("su_ciuc"; Example 10-3,
        p. 10-20, applies N*c to Example 10-2's su(CIUC)); else its cu as
        given ("cu")."""
        if (self.clay_base_method != "gec10"
                or self.alpha_method != "rational"
                or self.su_test_type == "ciuc"):
            return tip_layer.cu, "cu"
        row = layer_results[-1]
        if row["method"] != "excluded":
            z = (row["effective_top_m"] + row["effective_bottom_m"]) / 2
        else:
            z = (row["depth_top_m"] + row["depth_bottom_m"]) / 2
        sigma_v = self.soil.effective_stress_at_depth(z)
        return su_to_ciuc(tip_layer.cu, sigma_v, self.su_test_type), \
            "su_ciuc"

    def _phi_choice(self, sand, steps) -> Optional[dict]:
        """The choice made for phi' of the cohesionless layers under the
        GEC-10 beta (GM-244), where a layer gives (N1)60: a phi' given
        wins, the correlation only where none is. None where no layer
        gives (N1)60."""
        from geotech_common.choices import choice, how_given
        both = [(lb, st) for lb, st in zip(sand, steps)
                if "phi_from_N1_60_deg" in st]
        corr = [(lb, st) for lb, st in zip(sand, steps)
                if st["phi_from"] == "N1_60"]
        if not both and not corr:
            return None
        given = [(lb, st) for lb, st in zip(sand, steps)
                 if st["phi_from"] == "phi"]
        parts = []
        if given:
            parts.append("the layer's phi' as given (" + ", ".join(
                f"{lb['depth_top_m']:g}-{lb['depth_bottom_m']:g} m: "
                f"{st['phi_deg']:g} deg" for lb, st in given) + ")")
        if corr:
            parts.append("27.5 + 9.2 log (N1)60 where no phi' is given ("
                         + ", ".join(
                             f"{lb['depth_top_m']:g}-{lb['depth_bottom_m']:g}"
                             f" m: {st['phi_deg']:.1f} deg"
                             for lb, st in corr) + ")")
        alternatives = []
        if both:
            soil = replace(self.soil, layers=[
                replace(lay, phi=0.0)
                if lay.soil_type == "cohesionless" and lay.phi > 0
                and lay.N1_60 > 0 else lay for lay in self.soil.layers])
            alternatives.append(
                "phi' from (N1)60 in the layers that give both ("
                + ", ".join(f"{lb['depth_top_m']:g}-{lb['depth_bottom_m']:g}"
                            f" m: {st['phi_from_N1_60_deg']:.1f} deg"
                            for lb, st in both)
                + "): " + self._alternative(soil=soil))
        phis = tuple(st["phi_deg"] if st["phi_from"] == "phi" else 0.0
                     for st in steps)
        return choice("phi' of cohesionless soil", "; ".join(parts),
                      how_given(phis, (0.0,) * len(phis), within="layers"),
                      basis=PHI_BASIS, alternatives=alternatives)

    def _clay_side_choices(self, layer_results, Qs_clay: float,
                           Q_ultimate: float) -> List[dict]:
        """The choices made for side resistance in cohesive soil (GM-244):
        the alpha method, under GEC-10's the test each cu stands for, and
        the zone above the base. Each lists its alternatives with their
        values."""
        from geotech_common.choices import choice, how_given
        out = []
        bottom = self._bottom_zone_choice(layer_results)
        clay = [lb for lb in layer_results
                if lb["soil_type"] == "cohesive"
                and lb["method"] != "excluded"]
        if not clay:
            return [bottom] if bottom else []
        here = (f"Q_s in cohesive soil {Qs_clay:,.0f} kN, Q_ult "
                f"{Q_ultimate:,.0f} kN")
        how = how_given(self.alpha_method, "rational", name="alpha_method")
        each = "; ".join(
            f"{lb['effective_top_m']:g}-{lb['effective_bottom_m']:g} m: "
            f"su/pa {lb['alpha_steps']['su_over_pa']:.3g}, alpha "
            f"{lb['alpha_steps']['alpha']:.3f}" for lb in clay)
        if self.alpha_method == "rational":
            basis = ALPHA_BASIS
            outside = [lb for lb in clay
                       if not 0.25 <= lb["alpha_steps"]["su_over_pa"] <= 5]
            if outside:
                basis += (
                    ". Fig. 10-6 draws Eq. 10-18 over su(CIUC)/pa from about "
                    "0.25 (alpha 1.0) to 5; at " + _spans(outside)
                    + " su(CIUC)/pa is outside it, and alpha is the "
                    "equation's beyond the data: the engineer's judgment")
            out.append(choice(
                "side resistance in cohesive soil",
                f"GEC-10 (2018) alpha, Eq. 10-18, on su(CIUC) ({each}): "
                + here, how, basis=basis,
                alternatives=[
                    "alpha_method 'aashto', O'Neill and Reese (1999) as FHWA "
                    "NHI-06-089 prints it (Eqs. 9-37a, 9-37b, p. 9-129), "
                    "alpha = 0.55 for su/pa <= 1.5: "
                    + self._alternative(alpha_method="aashto", part="clay")]))
            out.append(self._su_test_choice(clay))
        else:
            basis = AASHTO_ALPHA_BASIS
            past = [lb for lb in clay
                    if lb["alpha_steps"]["su_over_pa"] > 2.5]
            if past:
                basis += (". At " + _spans(past) + " su/pa is above 2.5, "
                          "where NHI-06-089 prints no alpha")
            out.append(choice(
                "side resistance in cohesive soil",
                "alpha of O'Neill and Reese (1999), FHWA NHI-06-089 Eqs. "
                f"9-37a, 9-37b ({each}): " + here, how, basis=basis,
                alternatives=[
                    "alpha_method 'rational', GEC-10 (2018) Eq. 10-18 on "
                    "su(CIUC) (su_test_type for a UC or UU strength): "
                    + self._alternative(alpha_method="rational",
                                        part="clay")]))
        if bottom:
            out.append(bottom)
        return out

    def _su_test_choice(self, clay) -> dict:
        """The choice made for the strength in GEC-10's Eq. 10-18: which
        test each cu stands for (GM-244)."""
        from geotech_common.choices import choice, how_given
        named = {"ciuc": "su(CIUC), untransformed",
                 "uc": "UC, transformed to su(CIUC) by Eq. 10-16",
                 "uu": "UU, transformed to su(CIUC) by Eq. 10-17"}
        tt = self.su_test_type
        chosen = "each layer's cu taken as " + named[tt]
        if tt != "ciuc":
            chosen += " (" + "; ".join(
                f"{lb['effective_top_m']:g}-{lb['effective_bottom_m']:g} m: "
                f"{lb['alpha_steps']['su_kPa']:.1f} to "
                f"{lb['alpha_steps']['su_used_kPa']:.1f} kPa"
                for lb in clay) + ")"
        return choice(
            "strength su in Eq. 10-18 (the test each cu came from)", chosen,
            how_given(tt, "ciuc", name="su_test_type"),
            basis=SU_TEST_BASIS,
            alternatives=[
                f"su_test_type '{t}', {named[t]}: "
                + self._alternative(su_test_type=t, part="clay")
                for t in named if t != tt])

    def _bottom_zone_choice(self, layer_results) -> Optional[dict]:
        """The choice made for side resistance near the base in cohesive
        soil (GM-244), or None where no cohesive layer reaches within one
        diameter of the base (or within bottom_exclusion, where given)."""
        from geotech_common.choices import choice, how_given
        L, D = self.shaft.length, self.shaft.diameter
        z = self.bottom_exclusion
        reach = z if z > 0 else D
        rows = [lb for lb in layer_results
                if lb["soil_type"] == "cohesive"
                and lb["depth_bottom_m"] > L - reach + 1e-9]
        if not rows:
            return None
        where = ", ".join(f"{max(lb['depth_top_m'], L - reach):g}-"
                          f"{lb['depth_bottom_m']:g} m" for lb in rows)
        one_d = (f"bottom_exclusion {D:g}, the bottom one diameter of FHWA "
                 "NHI-06-089 Fig. 9-49 (p. 9-129; FHWA 1999): "
                 + self._alternative(bottom_exclusion=D, part="clay"))
        five_ft = ("bottom_exclusion 1.524, the bottom 5 ft of NHI-06-089's "
                   "text (p. 9-129): "
                   + self._alternative(bottom_exclusion=1.524, part="clay"))
        if z == 0:
            chosen = (f"taken to the base: the bottom {D:g} m (one "
                      f"diameter) carries side resistance ({where})")
            alternatives = [one_d] + ([five_ft] if abs(D - 1.524) > 1e-3
                                      else [])
        else:
            chosen = f"none over the bottom {z:g} m ({where})"
            alternatives = [
                "bottom_exclusion 0, side resistance to the base as GEC-10 "
                "(2018) takes it (Example 10-2, p. 10-18): "
                + self._alternative(bottom_exclusion=0.0, part="clay")]
            if abs(z - D) > 1e-9:
                alternatives.append(one_d)
            if abs(z - 1.524) > 1e-3:
                alternatives.append(five_ft)
        return choice("side resistance near the base in cohesive soil",
                      chosen, how_given(z, 0.0, name="bottom_exclusion"),
                      basis=BOTTOM_ZONE_BASIS, alternatives=alternatives)

    def _base_choices(self, tip_layer, base_steps: dict, Qt: float,
                      Q_ultimate: float) -> List[dict]:
        """The choices made for the base resistance (GM-244): the method
        in cohesive soil or rock, and the reduction for a large base. Each
        lists its alternatives with their values."""
        from geotech_common.choices import choice, how_given
        out = []
        here = f"Q_b {Qt:,.0f} kN, Q_ult {Q_ultimate:,.0f} kN"
        kind = base_steps["soil_type"]
        B = base_steps["base_diameter_m"]
        if kind == "cohesive":
            how = how_given(self.clay_base_method, "gec10",
                            name="clay_base_method")
            Nc = base_steps["Nc"]
            if self.clay_base_method == "gec10":
                eq = base_steps["equation"]
                factor = ("" if eq == "10-19"
                          else f"{base_steps['depth_factor']:.3f} x ")
                tip_words = (
                    "the tip layer's su(CIUC), as its side resistance took "
                    "it" if base_steps["su_from"] == "su_ciuc"
                    else "the tip layer's cu")
                su_words = _su_zone_words(base_steps, tip_words)
                chosen = (f"GEC-10 (2018) Eq. {eq}, q_BN = {factor}N*c su, "
                          f"su = {base_steps['su_kPa']:.1f} kPa ({su_words}), "
                          f"N*c = {Nc:.2f} (Table 10-2 at "
                          f"{base_steps['su_psf']:,.0f} psf): " + here)
                basis = CLAY_BASE_BASIS
                alternatives = [
                    "clay_base_method 'oneill_reese', O'Neill and Reese "
                    "(1999) as FHWA NHI-06-089 prints it (Eqs. 9-39, 9-40, "
                    "p. 9-131): Nc = 6[1 + 0.2 z/D] <= 9, x 0.67 where su < "
                    "0.5 ksf, q_t <= 80 ksf, on the tip layer's cu: "
                    + self._alternative(clay_base_method="oneill_reese",
                                        part="base")]
                su_tip = base_steps["su_tip_kPa"]
                if abs(su_tip - base_steps["su_kPa"]) > 1e-6:
                    Qb = self._clay_base_with(su_tip)
                    alternatives.append(
                        f"su = the tip layer's alone, {su_tip:.1f} kPa (the "
                        "layer the base is in; this tool's su until "
                        f"GM-252): Q_b {Qb:,.0f} kN, Q_ult "
                        f"{Q_ultimate - Qt + Qb:,.0f} kN")
                if base_steps["below_table"]:
                    line = nc_below_table_10_2(base_steps["su_kPa"])
                    Qb = Qt * line / Nc
                    basis += (
                        f". Table 10-2 starts at 500 psf; su = "
                        f"{base_steps['su_psf']:,.0f} psf is below it, and "
                        "N*c is held at 6.5, the table's first row. Below "
                        "500 psf N*c is the engineer's judgment: GEC-10 "
                        "gives no value")
                    alternatives.append(
                        f"N*c = {line:.2f} on the line through Table 10-2's "
                        "first two rows, which GEC-10 does not print: Q_b "
                        f"{Qb:,.0f} kN, Q_ult {Q_ultimate - Qt + Qb:,.0f} kN")
                if base_steps["su_from"] == "su_ciuc":
                    basis += (
                        ". Example 10-3 (p. 10-20) applies N*c to the "
                        "su(CIUC) of Example 10-2, transformed from the UU "
                        "strength; here the tip layer's, on sigma'vo at the "
                        "middle of the length that carries side resistance")
                    plain = self._su_zone(B, tip_layer.cu, False)["su_kPa"]
                    Qb = self._clay_base_with(plain)
                    alternatives.append(
                        f"su = cu as given, {plain:.1f} kPa, "
                        f"untransformed: Q_b {Qb:,.0f} kN, Q_ult "
                        f"{Q_ultimate - Qt + Qb:,.0f} kN")
                if eq == "10-20" and self.shaft.bell_diameter is not None:
                    basis += (". Eq. 10-20 is taken with B the bell's "
                              "diameter, on which the base bears")
            else:
                notes = ""
                if base_steps["soft_clay"]:
                    notes += " (x 0.67, su < 0.5 ksf)"
                if base_steps["capped"]:
                    notes += ", q_t limited to 80 ksf"
                chosen = ("O'Neill and Reese (1999), FHWA NHI-06-089 Eqs. "
                          f"9-39, 9-40: Nc = {Nc:.2f}{notes}: " + here)
                basis = ONEILL_REESE_CLAY_BASE_BASIS
                alternatives = [
                    "clay_base_method 'gec10', GEC-10 (2018) Eqs. 10-19, "
                    "10-20 with Table 10-2: "
                    + self._alternative(clay_base_method="gec10",
                                        part="base")]
            out.append(choice("base resistance in cohesive soil", chosen,
                              how, basis=basis, alternatives=alternatives))
        elif kind == "rock":
            how = how_given(self.rock_base_method, "gec10",
                            name="rock_base_method")
            RQD = base_steps["RQD"]
            conditions = None
            rqd = (f"rock_base_method 'rqd', q_b/q_u = "
                   f"{rock_base_factor(RQD, 'rqd'):.2f} at RQD {RQD:g} %: "
                   + self._alternative(rock_base_method="rqd", part="base"))
            if self.rock_base_method == "gec10" \
                    and base_steps["equation"] == "10-29":
                chosen, basis, alternatives = self._hoek_brown_choice(
                    base_steps, Qt, Q_ultimate, here)
                alternatives.insert(1, rqd)
                conditions = self._rock_conditions_choice(B, "10-29")
            elif self.rock_base_method == "gec10":
                chosen = "GEC-10 (2018) Eq. 10-23, q_BN = 2.5 qu: " + here
                basis = ROCK_BASE_BASIS
                conditions = self._rock_conditions_choice(B, "10-23")
                alternatives = [rqd, "the base of a fractured rock mass: "
                                + self._hoek_brown_alternative()]
                alternatives += self._hoek_brown_below_alternatives(
                    tip_layer, B, Qt, Q_ultimate, base_steps["qb"])
            else:
                chosen = (f"q_b/q_u = {base_steps['factor']:.2f} at RQD "
                          f"{RQD:g} %: " + here)
                basis = RQD_BASE_BASIS
                alternatives = [
                    "rock_base_method 'gec10', GEC-10 (2018) Eq. 10-23, "
                    "N*cr = 2.5, where the rock below the base meets the "
                    "conditions of p. 10-22 (rock_base_conditions_met "
                    "true): "
                    + self._alternative(rock_base_method="gec10",
                                        rock_base_conditions_met=True,
                                        part="base"),
                    "rock_base_method 'gec10' with rock_base_conditions_met "
                    "false, the base of a fractured rock mass: "
                    + self._hoek_brown_alternative()]
                alternatives += self._hoek_brown_below_alternatives(
                    tip_layer, B, Qt, Q_ultimate, base_steps["qb"])
            out.append(choice("base resistance in rock", chosen, how,
                              basis=basis, alternatives=alternatives))
            if conditions:
                out.append(conditions)
        elif kind == "cohesionless":
            out.append(self._sand_base_choice(base_steps, Qt, Q_ultimate,
                                              here))
        limit = {"cohesive": LARGE_BASE_DIAMETER,
                 "cohesionless": LARGE_BASE_DIAMETER_SAND}.get(kind)
        if limit is not None and B > limit:
            name = "1.27/B" if kind == "cohesionless" else "F_r"
            how = how_given(self.large_base_reduction, False,
                            name="large_base_reduction")
            if self.large_base_reduction:
                factor = (base_steps["reduction"] if kind == "cohesionless"
                          else base_steps["Fr"])
                chosen = (f"{name} = {factor:.3f} on the {B:g} m base "
                          "(O'Neill and Reese 1999): " + here)
                alternatives = [
                    "large_base_reduction false, none, as GEC-10 (2018) "
                    "prints it: "
                    + self._alternative(large_base_reduction=False,
                                        part="base")]
            else:
                chosen = (f"none on the {B:g} m base, as GEC-10 (2018) "
                          "prints it: " + here)
                alternatives = [
                    f"large_base_reduction true, O'Neill and Reese's (1999) "
                    f"{name} for a base wider than {limit:g} m: "
                    + self._alternative(large_base_reduction=True,
                                        part="base")]
            out.append(choice("reduction of base resistance for a large "
                              "base", chosen, how, basis=LARGE_BASE_BASIS,
                              alternatives=alternatives))
        return out

    def _sand_base_choice(self, base_steps: dict, Qt: float,
                          Q_ultimate: float, here: str) -> dict:
        """The choice made for the base in cohesionless soil (GM-284): Eq.
        10-14 on N60 the mean over 2B below the base (GEC-10 p. 10-15),
        limited to 50; the tip layer's N60 alone the alternative where it
        differs."""
        from geotech_common.choices import MODULE, choice
        mean, tip = base_steps["N60_mean"], base_steps["N60_tip"]
        capped = mean > 50.0
        chosen = (f"GEC-10 (2018) Eq. 10-14, q_BN = 57.5 N60 kPa (0.60 N60 "
                  f"tsf), N60 = {mean:.1f} ({_n60_zone_words(base_steps)})"
                  + (", limited to 50 (30 tsf)" if capped else "")
                  + ": " + here)
        basis = SAND_BASE_BASIS
        if capped:
            basis += (f". The mean N60 {mean:.1f} is above 50: GEC-10 "
                      "recommends load testing, 'otherwise unit base "
                      "resistance is limited to the upper-bound value' (p. "
                      "10-15)")
        alternatives = []
        if abs(min(mean, 50.0) - min(tip, 50.0)) > 1e-6:
            Qb = Qt * min(tip, 50.0) / min(mean, 50.0)
            alternatives.append(
                f"N60 = the tip layer's alone, {tip:g}, the layer the base "
                f"is in: Q_b {Qb:,.0f} kN, Q_ult "
                f"{Q_ultimate - Qt + Qb:,.0f} kN")
        return choice("base resistance in cohesionless soil", chosen, MODULE,
                      basis=basis, alternatives=alternatives)

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
        self._check_hoek_brown_layers([float(d) for d in depths],
                                      below=False)

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
