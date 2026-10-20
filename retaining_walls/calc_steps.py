"""
Calculation package steps for retaining wall analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation.

Handles both cantilever retaining walls and MSE walls.

The `analysis` parameter should be a dict with:
  Common keys:
    - "wall_type": "cantilever" or "mse"
    - "geom": CantileverWallGeometry or MSEWallGeometry
    - "gamma_backfill": float (kN/m3)
    - "phi_backfill": float (degrees)
    - "c_backfill": float (kPa, default 0)
    - "pressure_method": "rankine" or "coulomb" (cantilever only, default "rankine")
  Cantilever-specific:
    - "gamma_concrete": float (kN/m3, default 24)
    - "phi_foundation": float (degrees, optional)
    - "c_foundation": float (kPa, default 0)
    - "q_allowable": float (kPa, optional)
    - "FOS_sliding": float (default 1.5)
    - "FOS_overturning": float (default 2.0)
  MSE-specific:
    - "reinforcement": Reinforcement dataclass
    - "gamma_foundation": float (kN/m3, optional)
    - "phi_foundation": float (degrees, optional)
    - "c_foundation": float (kPa, default 0)
    - "q_allowable": float (kPa, optional)

References:
    FHWA GEC-11 (FHWA-NHI-10-024) — MSE Walls and Reinforced Slopes
    AASHTO LRFD Bridge Design Specifications, Section 11
    Das, B.M., Principles of Foundation Engineering, Ch 13
    Coulomb (1776), Rankine (1857) — Earth pressure theories
"""

import math
from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Retaining Wall Stability Analysis"

REFERENCES = [
    'FHWA GEC-11 (FHWA-NHI-10-024): Design of Mechanically Stabilized '
    'Earth Walls and Reinforced Slopes. FHWA, 2009.',
    'AASHTO LRFD Bridge Design Specifications, Sections 11.6 and 11.10.',
    'Das, B.M. Principles of Foundation Engineering, 9th Ed., Ch. 13.',
    'Coulomb, C.A. (1776). "Essai sur une application des regles de '
    'maximis et minimis a quelques problemes de statique." Memoires de '
    'Mathematique et de Physique, Academie Royale des Sciences.',
    'Rankine, W.J.M. (1857). "On the Stability of Loose Earth." '
    'Philosophical Transactions of the Royal Society, 147.',
]


# ── Helper to detect wall type ──────────────────────────────────────
def _wall_type(result, analysis) -> str:
    """Return 'cantilever' or 'mse' based on analysis dict or result class."""
    if isinstance(analysis, dict) and "wall_type" in analysis:
        return analysis["wall_type"]
    # Fall back to result class inspection
    from retaining_walls.results import CantileverWallResult, MSEWallResult
    if isinstance(result, MSEWallResult):
        return "mse"
    return "cantilever"


def _get(analysis, key, default=None):
    """Safely get a value from analysis dict."""
    if isinstance(analysis, dict):
        return analysis.get(key, default)
    return getattr(analysis, key, default)


def _used(analysis, key):
    """An input the cantilever run used: the analysis dict's, else the
    default of ``analyze_cantilever_wall``, which the run then took
    (GM-141: the steps had defaults of their own)."""
    if isinstance(analysis, dict) and key in analysis:
        return analysis[key]
    import inspect
    from retaining_walls.cantilever import analyze_cantilever_wall
    default = inspect.signature(
        analyze_cantilever_wall).parameters[key].default
    if default is inspect.Parameter.empty:
        raise ValueError(
            f"retaining_walls calc steps: the analysis gives no {key}, "
            f"which analyze_cantilever_wall requires; pass the inputs the "
            f"run was given")
    return default


# ════════════════════════════════════════════════════════════════════
#  INPUT SUMMARY
# ════════════════════════════════════════════════════════════════════

def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for retaining wall calc package.

    Parameters
    ----------
    result : CantileverWallResult or MSEWallResult
        Computed results.
    analysis : dict
        Analysis parameters (see module docstring).

    Returns
    -------
    list of InputItem
    """
    wtype = _wall_type(result, analysis)

    if wtype == "mse":
        return _input_summary_mse(result, analysis)
    else:
        return _input_summary_cantilever(result, analysis)


def _input_summary_cantilever(result, analysis) -> List[InputItem]:
    """Input summary for cantilever walls."""
    # the inputs the run used (the cantilever_wall tool's, GM-141)
    geom = _get(analysis, "geom")
    gamma = _used(analysis, "gamma_backfill")
    phi = _used(analysis, "phi_backfill")
    c = _used(analysis, "c_backfill")
    method = _used(analysis, "pressure_method")
    gamma_c = _used(analysis, "gamma_concrete")
    phi_f = _used(analysis, "phi_foundation")
    c_f = _used(analysis, "c_foundation")
    q_all = _used(analysis, "q_allowable")
    FOS_s = result.FOS_sliding_required
    FOS_o = result.FOS_overturning_required

    items = [
        InputItem("Type", "Wall type", "Cantilever", ""),
        InputItem("H", "Wall height (underside of base to top of stem)",
                  f"{geom.wall_height:.2f}", "m"),
        InputItem("h_stem", "Stem height (H - base thickness)",
                  f"{geom.stem_height:.2f}", "m"),
        InputItem("B", "Base width", f"{geom.base_width:.2f}", "m"),
        InputItem("t_toe", "Toe length", f"{geom.toe_length:.2f}", "m"),
        InputItem("t_heel", "Heel length", f"{geom.heel_length:.2f}", "m"),
        InputItem("t_stem_top", "Stem thickness (top)", f"{geom.stem_thickness_top:.2f}", "m"),
        InputItem("t_stem_base", "Stem thickness (base)", f"{geom.stem_thickness_base:.2f}", "m"),
        InputItem("t_base", "Base slab thickness", f"{geom.base_thickness:.2f}", "m"),
    ]

    if geom.backfill_slope > 0:
        items.append(InputItem("\u03b2", "Backfill slope", f"{geom.backfill_slope:.1f}", "deg"))
    # The live and the dead-load surcharge (GM-157)
    if geom.surcharge > 0:
        items.append(InputItem("q_s", "Live-load surcharge", f"{geom.surcharge:.1f}", "kPa"))
    if getattr(geom, "dead_load_surcharge", 0.0) > 0:
        items.append(InputItem("q_D", "Dead-load surcharge",
                               f"{geom.dead_load_surcharge:.1f}", "kPa"))

    items.extend([
        InputItem("\u03b3_backfill", "Backfill unit weight", f"{gamma:.1f}", "kN/m\u00b3"),
        InputItem("\u03c6_backfill", "Backfill friction angle", f"{phi:.1f}", "deg"),
    ])
    if c > 0:
        items.append(InputItem("c_backfill", "Backfill cohesion", f"{c:.1f}", "kPa"))
    items.append(InputItem("\u03b3_concrete", "Concrete unit weight", f"{gamma_c:.1f}", "kN/m\u00b3"))

    if phi_f is not None and phi_f != phi:
        items.append(InputItem("\u03c6_foundation", "Foundation friction angle",
                               f"{phi_f:.1f}", "deg"))
    if c_f > 0:
        items.append(InputItem("c_foundation", "Foundation cohesion", f"{c_f:.1f}", "kPa"))
    if q_all is not None:
        items.append(InputItem("q_all", "Allowable bearing pressure", f"{q_all:.1f}", "kPa"))
    else:
        items.append(InputItem("q_all", "Allowable bearing pressure",
                               "not given: bearing NOT CHECKED", ""))

    items.extend([
        InputItem("Method", "Earth pressure theory", method.capitalize(), ""),
        InputItem("FOS_s_req", "Required FOS (sliding)", f"{FOS_s:.1f}", ""),
        InputItem("FOS_ot_req", "Required FOS (overturning)", f"{FOS_o:.1f}", ""),
    ])

    return items


def _input_summary_mse(result, analysis) -> List[InputItem]:
    """Input summary for MSE walls."""
    geom = _get(analysis, "geom")
    gamma = _get(analysis, "gamma_backfill", 18.0)
    phi = _get(analysis, "phi_backfill", 34.0)
    reinforcement = _get(analysis, "reinforcement")
    phi_f = _get(analysis, "phi_foundation")
    c_f = _get(analysis, "c_foundation", 0.0)
    q_all = _get(analysis, "q_allowable")

    items = [
        InputItem("Type", "Wall type", "MSE (Mechanically Stabilized Earth)", ""),
        InputItem("H", "Wall height", f"{geom.wall_height:.2f}", "m"),
        InputItem("L", "Reinforcement length", f"{geom.reinforcement_length:.2f}", "m"),
        InputItem("S_v", "Reinforcement spacing", f"{geom.reinforcement_spacing:.2f}", "m"),
        InputItem("N_levels", "Number of reinforcement levels", geom.n_reinforcement_levels, ""),
    ]

    if geom.backfill_slope > 0:
        items.append(InputItem("\u03b2", "Backfill slope", f"{geom.backfill_slope:.1f}", "deg"))
    if geom.surcharge > 0:
        items.append(InputItem("q_s", "Live-load surcharge", f"{geom.surcharge:.1f}", "kPa"))

    items.extend([
        InputItem("\u03b3_backfill", "Backfill unit weight", f"{gamma:.1f}", "kN/m\u00b3"),
        InputItem("\u03c6_backfill", "Backfill friction angle", f"{phi:.1f}", "deg"),
    ])

    if reinforcement is not None:
        items.extend([
            InputItem("Reinf.", "Reinforcement product", reinforcement.name, ""),
            InputItem("T_al", "Allowable tensile strength", f"{reinforcement.Tallowable:.1f}",
                      "kN/m"),
            InputItem("Type_r", "Reinforcement type",
                      "Metallic" if reinforcement.is_metallic else "Geosynthetic", ""),
        ])

    if phi_f is not None:
        items.append(InputItem("\u03c6_foundation", "Foundation friction angle",
                               f"{phi_f:.1f}", "deg"))
    if c_f > 0:
        items.append(InputItem("c_foundation", "Foundation cohesion", f"{c_f:.1f}", "kPa"))
    if q_all is not None:
        items.append(InputItem("q_all", "Allowable bearing pressure", f"{q_all:.1f}", "kPa"))

    return items


# ════════════════════════════════════════════════════════════════════
#  CALCULATION STEPS
# ════════════════════════════════════════════════════════════════════

def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections.

    Parameters
    ----------
    result : CantileverWallResult or MSEWallResult
        Computed results.
    analysis : dict
        Analysis parameters (see module docstring).

    Returns
    -------
    list of CalcSection
    """
    wtype = _wall_type(result, analysis)

    if wtype == "mse":
        return _calc_steps_mse(result, analysis)
    else:
        return _calc_steps_cantilever(result, analysis)


def _calc_steps_cantilever(result, analysis) -> List[CalcSection]:
    """Calculation steps for cantilever retaining wall."""
    # the inputs the run used (the cantilever_wall tool's, GM-141)
    geom = _get(analysis, "geom")
    gamma = _used(analysis, "gamma_backfill")
    phi = _used(analysis, "phi_backfill")
    c = _used(analysis, "c_backfill")
    method = _used(analysis, "pressure_method")
    phi_f = _used(analysis, "phi_foundation")
    q_all = _used(analysis, "q_allowable")
    FOS_s_req = result.FOS_sliding_required
    FOS_o_req = result.FOS_overturning_required

    if phi_f is None:
        phi_f = phi

    sections = []

    # ── Section 1: Earth Pressure Coefficient ──────────────────
    ep_items = []

    # The terms the analysis used (GM-143): the result's own, so the
    # printout cannot drift from the run.
    ep, forces, sl, ot, br = _cantilever_terms(result, analysis)

    Ka_val = ep["Ka"]
    if method == "coulomb":
        delta = ep["delta_deg"]
        beta = geom.backfill_slope

        ep_items.append(CalcStep(
            title="Coulomb Active Earth Pressure Coefficient",
            equation=(
                "K_a = sin\u00b2(\u03b1 + \u03c6) / "
                "[sin\u00b2(\u03b1) \u00d7 sin(\u03b1 - \u03b4) \u00d7 "
                "(1 + \u221a(sin(\u03c6+\u03b4)\u00d7sin(\u03c6-\u03b2) / "
                "(sin(\u03b1-\u03b4)\u00d7sin(\u03b1+\u03b2))))\u00b2]"
            ),
            substitution=(
                f"\u03c6 = {phi:.1f}\u00b0, \u03b4 = {delta:.1f}\u00b0, "
                f"\u03b2 = {beta:.1f}\u00b0"
            ),
            result_name="K_a",
            result_value=f"{Ka_val:.4f}",
            reference="Coulomb (1776)",
        ))
    else:
        if geom.backfill_slope > 0:
            ep_items.append(CalcStep(
                title="Rankine Active Earth Pressure Coefficient (sloped backfill)",
                equation=(
                    "K_a = cos\u03b2 \u00d7 (cos\u03b2 - \u221a(cos\u00b2\u03b2 - "
                    "cos\u00b2\u03c6)) / (cos\u03b2 + \u221a(cos\u00b2\u03b2 - cos\u00b2\u03c6))"
                ),
                substitution=(
                    f"\u03c6 = {phi:.1f}\u00b0, \u03b2 = {geom.backfill_slope:.1f}\u00b0"
                ),
                result_name="K_a",
                result_value=f"{Ka_val:.4f}",
                reference="Rankine (1857), sloped backfill",
            ))
        else:
            ep_items.append(CalcStep(
                title="Rankine Active Earth Pressure Coefficient",
                equation="K_a = tan\u00b2(45\u00b0 - \u03c6/2)",
                substitution=f"K_a = tan\u00b2(45\u00b0 - {phi:.1f}\u00b0/2)",
                result_name="K_a",
                result_value=f"{Ka_val:.4f}",
                reference="Rankine (1857)",
            ))

    # Active height
    H = ep["H_active"]
    ep_items.append(CalcStep(
        title="Active Pressure Height",
        equation="H_active = H + heel \u00d7 tan(\u03b2)" if geom.backfill_slope > 0
        else "H_active = H (no backfill slope)",
        substitution=f"H_active = {geom.wall_height:.2f}" + (
            f" + {geom.heel_length:.2f} \u00d7 tan({geom.backfill_slope:.1f}\u00b0)"
            if geom.backfill_slope > 0 else ""
        ),
        result_name="H_active",
        result_value=f"{H:.3f}",
        result_unit="m",
    ))

    # Active force, as the analysis summed it; q is the live and the
    # dead-load surcharge together (GM-157)
    Pa_total = ep["Pa"]
    q_live = ep.get("q_live", geom.surcharge)
    q_dead = ep.get("q_dead", getattr(geom, "dead_load_surcharge", 0.0))
    q = q_live + q_dead
    ep_items.append(CalcStep(
        title="Total Active Thrust",
        equation="P_a = 0.5 \u00d7 K_a \u00d7 \u03b3 \u00d7 H\u00b2"
                 + (" + K_a \u00d7 q \u00d7 H" if q > 0 else "")
                 + (" - 2c\u221aK_a \u00d7 H" if c > 0 else ""),
        substitution=(
            f"P_a = 0.5 \u00d7 {Ka_val:.4f} \u00d7 {gamma:.1f} \u00d7 {H:.3f}\u00b2"
            + (f" + {Ka_val:.4f} \u00d7 {q:.1f} \u00d7 {H:.3f}"
               if q > 0 else "")
            + (f" - 2 \u00d7 {c:.1f} \u00d7 \u221a{Ka_val:.4f} \u00d7 {H:.3f}"
               if c > 0 else "")
        ),
        result_name="P_a",
        result_value=f"{Pa_total:.1f}",
        result_unit="kN/m",
        reference="Das Ch. 13",
        notes=(f"q = q_live + q_dead = {q_live:.1f} + {q_dead:.1f} kPa: "
               "both surcharges' thrust drives every check."
               if q_live > 0 and q_dead > 0 else ""),
    ))

    # Thrust inclination & components (cantilever._active_thrust, RW-1):
    # Coulomb -> delta = 2/3*phi; sloped Rankine -> delta = beta; else horizontal.
    delta_incl = ep["delta_deg"]
    Ph = ep["Ph"]
    Pv = ep["Pv"]

    if delta_incl > 0:
        # The live load's share of P_v the analysis left out of the
        # resisting side (GM-157), from the forces it summed
        Pv_left_out = sum(f["W"] for f in forces
                          if f.get("kind") == "thrust"
                          and not f.get("resists", True))
        ep_items.append(CalcStep(
            title="Thrust Components (inclined thrust)",
            equation="P_h = P_a \u00d7 cos(\u03b4),  P_v = P_a \u00d7 sin(\u03b4)",
            substitution=(
                f"\u03b4 = {delta_incl:.2f}\u00b0; "
                f"P_h = {Pa_total:.1f} \u00d7 cos({delta_incl:.2f}\u00b0), "
                f"P_v = {Pa_total:.1f} \u00d7 sin({delta_incl:.2f}\u00b0)"
            ),
            result_name="P_h, P_v",
            result_value=f"{Ph:.1f}, {Pv:.1f}",
            result_unit="kN/m",
            notes=(
                "P_h drives sliding/overturning; P_v acts downward at the "
                "pressure plane (back of heel) and is stabilizing."
                + (f" The live load's share, {Pv_left_out:.1f} kN/m, is "
                   "left out of sliding, overturning and the eccentricity "
                   "check." if Pv_left_out > 0 else "")
            ),
            reference="Coulomb wall friction \u03b4 = 2/3\u03c6" if method == "coulomb"
            else "Rankine thrust parallel to backfill slope",
        ))

    sections.append(CalcSection(title="Earth Pressure Computation", items=ep_items))

    # ── Section 2: Wall Weight Components ──────────────────────
    weight_items = []
    B = geom.base_width

    # Every vertical force the analysis summed, with its arm (GM-143)
    weight_rows = [[f["component"], f"{f['W']:.1f}", f"{f['x']:.3f}",
                    f"{f['M']:.1f}"] for f in forces]

    # Totals, as the analysis summed them. A surcharge over the heel that
    # is not a resisting force (GM-157) makes two sets of loads: the one
    # sliding, overturning and the eccentricity check count, and the one
    # the bearing pressures count. Each check's own total is printed.
    total_W = br["V"]
    M_stab = br["M_stab"]
    split = _loads_differ(ot, br)
    headers = ["Component", "Weight (kN/m)", "Arm from Toe (m)",
               "Moment (kN\u00b7m/m)"]
    notes = "Moments computed about toe for overturning check."
    if split or any(not f.get("resists", True) or not f.get("bearing", True)
                    for f in forces):
        headers.append("Counted in")
        for row, f in zip(weight_rows, forces):
            row.append(_counted_in(f))
        notes = ("Counted in: the checks that sum the force. A surcharge "
                 "over the heel that is not a resisting force is left out "
                 "of sliding, overturning and the eccentricity check "
                 "(see Choices).")
    if split:
        weight_rows.append(["TOTAL, sliding / overturning / eccentricity",
                            f"{ot['V']:.1f}", "\u2014",
                            f"{ot['M_stabilizing']:.1f}", ""])
        weight_rows.append(["TOTAL, bearing", f"{total_W:.1f}", "\u2014",
                            f"{M_stab:.1f}", ""])
    else:
        weight_rows.append(["TOTAL", f"{total_W:.1f}", "\u2014",
                            f"{M_stab:.1f}"]
                           + ([""] if len(headers) == 5 else []))

    weight_items.append(TableData(
        title="Wall Component Weights and Moments about Toe",
        headers=headers,
        rows=weight_rows,
        notes=notes,
    ))

    sections.append(CalcSection(title="Wall Weights & Stabilizing Moments", items=weight_items))

    # ── Section 3: Sliding Check ───────────────────────────────
    slide_items = []

    # The base interface and passive resistance the analysis USED (GM-103:
    # the package printed (2/3)phi and no passive even when delta_base,
    # base_adhesion or include_passive set them otherwise).
    delta_b = sl["delta_b"]
    ca = sl["ca"]
    Pp = sl["Pp"]
    # GM-212: the printed rule for the foundation cohesion
    rule = sl.get("sliding_cohesion", "drained")
    width = sl.get("adhesion_width")

    if delta_b is None:
        pass
    elif sl["delta_b_given"]:
        slide_items.append(CalcStep(
            title="Base Friction Angle",
            equation="\u03b4_b given directly (no 2/3 factor)",
            substitution=f"\u03b4_b = {delta_b:.2f}\u00b0",
            result_name="\u03b4_b",
            result_value=f"{delta_b:.2f}",
            result_unit="deg",
            reference="input delta_base",
        ))
    else:
        slide_items.append(CalcStep(
            title="Base Friction Angle",
            equation="\u03b4_b = (2/3) \u00d7 \u03c6_foundation",
            substitution=f"\u03b4_b = (2/3) \u00d7 {phi_f:.1f}\u00b0",
            result_name="\u03b4_b",
            result_value=f"{delta_b:.2f}",
            result_unit="deg",
            reference=("EN 1997-1 6.5.3(10), 2/3 \u03c6 for a smooth "
                       "precast base; UFC 3-220-20 Table 4-1"),
        ))

    if rule == "friction_plus_adhesion":
        if sl["ca_given"]:
            slide_items.append(CalcStep(
                title="Base Adhesion",
                equation="C_a given directly",
                substitution=f"C_a = {ca:.1f} kPa",
                result_name="C_a", result_value=f"{ca:.1f}",
                result_unit="kPa", reference="input base_adhesion"))
        else:
            slide_items.append(CalcStep(
                title="Base Adhesion",
                equation=("C_a from the undrained shear strength c_f, "
                          "read linearly within its row"),
                substitution=f"c_f = {sl['c_foundation']:.1f} kPa",
                result_name="C_a", result_value=f"{ca:.1f}",
                result_unit="kPa", reference="UFC 3-220-20 Table 4-1"))
        e_s = sl["eccentricity"]
        slide_items.append(CalcStep(
            title="Effective Base Width",
            equation="B_e = B \u2212 2|e|",
            substitution=f"B_e = {B:.2f} \u2212 2 \u00d7 {abs(e_s):.3f}",
            result_name="B_e", result_value=f"{width:.3f}", result_unit="m",
            reference="UFC 3-220-20 Sec. 4-5.1.3, Fig. 4-27",
            notes=("e from the loads the sliding check counts, the uniform "
                   "bearing pressure method; NHI-06-089 Fig. 10-17 takes "
                   "the full B.")))

    R_sliding = sl["resisting"]
    if rule == "undrained":
        equation = "R = c_a \u00d7 B"
        substitution = f"R = {ca:.1f} \u00d7 {width:.2f}"
        reference = "GEC-6 Eq. 5-39"
        rule_note = ("Undrained: adhesion alone, c_a = "
                     + ("base_adhesion given directly"
                        if sl["ca_given"] else
                        "c_foundation as the undrained shear strength")
                     + ", no base friction (GEC-6: Eqs. 5-38 and 5-39 "
                     "'should not be combined'). ")
    elif rule == "friction_plus_adhesion":
        equation = "R = V \u00d7 tan(\u03b4_b) + C_a \u00d7 B_e"
        substitution = (f"R = {sl['V']:.1f} \u00d7 tan({delta_b:.2f}\u00b0)"
                        f" + {ca:.1f} \u00d7 {width:.2f}")
        reference = "UFC 3-220-20 Fig. 4-27"
        rule_note = ("Friction and adhesion added (UFC 3-220-20 Sec. "
                     "4-5.1.3, which adds that the minimum FS may need to "
                     "be increased when base adhesion is included). ")
    else:
        equation = "R = V \u00d7 tan(\u03b4_b)"
        substitution = f"R = {sl['V']:.1f} \u00d7 tan({delta_b:.2f}\u00b0)"
        reference = "GEC-6 Eq. 5-38"
        rule_note = (f"Drained: c_foundation = {sl['c_foundation']:g} kPa "
                     "not counted (GEC-6: Eqs. 5-38 and 5-39 'should not "
                     "be combined'; EN 1997-1 6.5.3(10)). "
                     if sl.get("c_foundation") else "")
    slide_items.append(CalcStep(
        title="Resisting Force (Sliding)",
        equation=equation + (" + P_p" if Pp > 0 else ""),
        substitution=substitution + (f" + {Pp:.1f}" if Pp > 0 else ""),
        result_name="R",
        result_value=f"{R_sliding:.1f}",
        result_unit="kN/m",
        reference=reference,
        notes=rule_note
              + (("P_p: Rankine passive resistance over the depth of the "
                  "shear key only, below the toe from the underside of the "
                  f"base ({sl['D_passive_top']:.2f} m) to the level of the "
                  f"bottom of the key ({sl['D_passive']:.2f} m), "
                  "UFC 3-220-20 Fig. 4-29. "
                  if sl.get("D_passive_top") else
                  "P_p: Rankine passive resistance in front of the base. ")
                 if Pp > 0 else "")
              + (f"V leaves out {sl['V_left_out']:.1f} kN/m of surcharge "
                 "over the heel, not a resisting force (see Choices)."
                 if sl.get("V_left_out", 0.0) > 0 else ""),
    ))

    if result.FOS_sliding is None:
        # No driving force (GM-100): say so, never a placeholder FOS.
        slide_items.append(
            "Sliding: no driving force — the active thrust P_h is zero, "
            "so nothing pushes the wall. The check passes; there is no "
            "finite factor of safety to report.")
    else:
        slide_items.append(CalcStep(
            title="Factor of Safety Against Sliding",
            equation="FOS_sliding = R / P_h",
            substitution=f"FOS_sliding = {R_sliding:.1f} / {sl['driving']:.1f}",
            result_name="FOS_sliding",
            result_value=f"{result.FOS_sliding:.3f}",
            reference="AASHTO 11.6.3 (min 1.5)",
            notes=("P_h = P_a × cos(δ) — horizontal thrust component."
                   if delta_incl > 0 else ""),
        ))

        slide_items.append(CheckItem(
            description="Sliding stability",
            demand=FOS_s_req,
            demand_label="FOS_required",
            capacity=result.FOS_sliding,
            capacity_label="FOS_sliding",
            passes=result.passes_sliding,
        ))

    sections.append(CalcSection(title="Sliding Stability Check", items=slide_items))

    # ── Section 4: Overturning Check ───────────────────────────
    ot_items = []

    z_Pa = ot["z_Pa"]
    M_over = ot["M_overturning"]

    ot_items.append(CalcStep(
        title="Overturning Moment about Toe",
        equation="M_ot = P_h \u00d7 z_Pa",
        substitution=f"M_ot = {ot['Ph']:.1f} \u00d7 {z_Pa:.3f}",
        result_name="M_ot",
        result_value=f"{M_over:.1f}",
        result_unit="kN\u00b7m/m",
        notes=("P_h = P_a \u00d7 cos(\u03b4); the vertical component P_v "
               "appears as a stabilizing entry in the weight table."
               if delta_incl > 0 else ""),
    ))

    ot_items.append(CalcStep(
        title="Stabilizing Moment about Toe",
        equation="M_stab = \u03a3(W_i \u00d7 x_i)",
        substitution=("(see weight table above: TOTAL, sliding / "
                      "overturning / eccentricity)" if split
                      else "(see weight table above)"),
        result_name="M_stab",
        result_value=f"{ot['M_stabilizing']:.1f}",
        result_unit="kN\u00b7m/m",
    ))

    if result.FOS_overturning is None:
        ot_items.append(
            "Overturning: no driving force — the overturning moment "
            "M_ot is zero. The check passes; there is no finite factor of "
            "safety to report.")
    else:
        ot_items.append(CalcStep(
            title="Factor of Safety Against Overturning",
            equation="FOS_ot = M_stab / M_ot",
            substitution=(f"FOS_ot = {ot['M_stabilizing']:.1f} / "
                          f"{M_over:.1f}"),
            result_name="FOS_ot",
            result_value=f"{result.FOS_overturning:.3f}",
            reference="AASHTO 11.6.3 (min 2.0)",
        ))

        ot_items.append(CheckItem(
            description="Overturning stability",
            demand=FOS_o_req,
            demand_label="FOS_required",
            capacity=result.FOS_overturning,
            capacity_label="FOS_overturning",
            passes=result.passes_overturning,
        ))

    sections.append(CalcSection(title="Overturning Stability Check", items=ot_items))

    # ── Section 5: Bearing Pressure & Eccentricity ─────────────
    brg_items = []

    # The eccentricity check, on the loads the overturning check counts,
    # as the analysis worked it out (GM-157)
    x_R_check = ot["x_resultant"]
    brg_items.append(CalcStep(
        title="Resultant Location from Toe"
              + (" (sliding / overturning loads)" if split else ""),
        equation="x_R = (M_stab - M_ot) / V",
        substitution=(f"x_R = ({ot['M_stabilizing']:.1f} - "
                      f"{ot['M_overturning']:.1f}) / {ot['V']:.1f}"),
        result_name="x_R",
        result_value=f"{x_R_check:.3f}",
        result_unit="m",
    ))

    brg_items.append(CalcStep(
        title="Eccentricity of Resultant",
        equation="e = B/2 - x_R",
        substitution=f"e = {B:.2f}/2 - {x_R_check:.3f}",
        result_name="e",
        result_value=f"{result.eccentricity:.3f}",
        result_unit="m",
        notes=f"B/6 = {B / 6:.3f} m. "
              f"{'Resultant within middle third' if result.in_middle_third else 'OUTSIDE middle third'}.",
    ))

    # The bearing pressures, on the bearing check's loads
    x_R = br["x_resultant"]
    e = br["eccentricity"]
    if split:
        brg_items.append(CalcStep(
            title="Resultant for Bearing (bearing loads)",
            equation="x_R,b = (M_stab,b - M_ot) / V_b,  e_b = B/2 - x_R,b",
            substitution=(f"x_R,b = ({br['M_stab']:.1f} - "
                          f"{br['M_over']:.1f}) / {total_W:.1f}; "
                          f"e_b = {B:.2f}/2 - {x_R:.3f}"),
            result_name="e_b",
            result_value=f"{e:.3f}",
            result_unit="m",
            notes=("The eccentricity for bearing is not the eccentricity "
                   "check's (GEC-11 Vol I p. 4-26): the bearing pressures "
                   "count the surcharge over the heel as the Choices say."),
        ))

    # Bearing pressures, by the distribution the analysis used
    if br["distribution"] == "trapezoidal":
        brg_items.append(CalcStep(
            title="Toe Bearing Pressure (Trapezoidal Distribution)",
            equation="q_toe = (V/B)(1 + 6e/B)",
            substitution=f"q_toe = ({total_W:.1f}/{B:.2f})(1 + 6\u00d7{e:.3f}/{B:.2f})",
            result_name="q_toe",
            result_value=f"{result.q_toe:.1f}",
            result_unit="kPa",
        ))
        brg_items.append(CalcStep(
            title="Heel Bearing Pressure",
            equation="q_heel = (V/B)(1 - 6e/B)",
            substitution=f"q_heel = ({total_W:.1f}/{B:.2f})(1 - 6\u00d7{e:.3f}/{B:.2f})",
            result_name="q_heel",
            result_value=f"{result.q_heel:.1f}",
            result_unit="kPa",
        ))
    elif not (br["B_eff"] or 0) > 0:
        # x_R <= 0 (or >= B): the resultant falls outside the base
        brg_items.append(
            "Bearing pressure: the resultant falls outside the base "
            f"(x_R = {x_R:.3f} m from the toe, B = {B:.2f} m), so no "
            "bearing pressure distribution balances it.")
    elif br["distribution"] == "triangular_toe":
        brg_items.append(CalcStep(
            title="Toe Bearing Pressure (Triangular Distribution)",
            equation="q_toe = 2V / (3 \u00d7 x_R)  [resultant outside middle third]",
            substitution=f"q_toe = 2 \u00d7 {total_W:.1f} / (3 \u00d7 {x_R:.3f})",
            result_name="q_toe",
            result_value=f"{result.q_toe:.1f}",
            result_unit="kPa",
            notes="q_heel = 0 kPa (resultant outside middle third).",
        ))
    else:
        brg_items.append(CalcStep(
            title="Heel Bearing Pressure (Triangular Distribution)",
            equation=("q_heel = 2V / (3 \u00d7 (B - x_R))  [resultant outside "
                      "middle third, heel side]"),
            substitution=(f"q_heel = 2 \u00d7 {total_W:.1f} / (3 \u00d7 "
                          f"({B:.2f} - {x_R:.3f}))"),
            result_name="q_heel",
            result_value=f"{result.q_heel:.1f}",
            result_unit="kPa",
            notes="q_toe = 0 kPa (resultant outside middle third).",
        ))

    # Bearing check (GM-100): run only against a given q_allowable, on the
    # larger edge pressure; otherwise say plainly that it was not run.
    q_max = max(result.q_toe, result.q_heel)
    if q_all is None or result.FOS_bearing is None:
        brg_items.append(
            "Bearing capacity: NOT CHECKED — no allowable bearing "
            "pressure (q_allowable) was given. The bearing pressures q_toe "
            "and q_heel and the middle-third check above are reported; "
            "compare the larger of q_toe and q_heel with the foundation's "
            "allowable bearing pressure.")
    elif q_max > 0:
        brg_items.append(CalcStep(
            title="Factor of Safety Against Bearing Failure",
            equation="FOS_bearing = q_allowable / q_max,  "
                     "q_max = max(q_toe, q_heel)",
            substitution=f"FOS_bearing = {q_all:.1f} / {q_max:.1f}",
            result_name="FOS_bearing",
            result_value=f"{result.FOS_bearing:.3f}",
            reference="AASHTO 11.6.3",
        ))
        brg_items.append(CheckItem(
            description="Bearing capacity adequacy",
            demand=q_max,
            demand_label="q_max",
            capacity=q_all,
            capacity_label="q_allowable",
            unit="kPa",
            passes=bool(result.passes_bearing),
        ))
    else:
        brg_items.append(
            "Bearing capacity: FAILS — the resultant falls outside the "
            "base, so no bearing pressure can balance it (the wall tips "
            "over the toe).")

    sections.append(CalcSection(
        title="Bearing Pressure & Eccentricity", items=brg_items
    ))

    # ── Section 6: Summary ─────────────────────────────────────
    summary_items = []

    def _fos_row(name, fos, required, passes):
        if fos is None:
            return [name, "\u2014", required, "OK (no driving force)"]
        return [name, f"{fos:.3f}", required, "OK" if passes else "FAIL"]

    if result.FOS_bearing is None:
        bearing_row = ["Bearing", "\u2014", "N/A", "NOT CHECKED"]
    else:
        bearing_row = _fos_row("Bearing", result.FOS_bearing, "1.0",
                               result.passes_bearing)
    summary_table = TableData(
        title="Stability Check Summary",
        headers=["Check", "FOS Computed", "FOS Required", "Status"],
        rows=[
            _fos_row("Sliding", result.FOS_sliding, f"{FOS_s_req:.1f}",
                     result.passes_sliding),
            _fos_row("Overturning", result.FOS_overturning,
                     f"{FOS_o_req:.1f}", result.passes_overturning),
            bearing_row,
            ["Eccentricity (middle third)", "\u2014", "e \u2264 B/6",
             "OK" if result.in_middle_third else "FAIL"],
        ],
    )
    summary_items.append(summary_table)

    eccentricity_status = "Within middle third" if result.in_middle_third else "OUTSIDE middle third"
    summary_items.append(
        f"Eccentricity: e = {result.eccentricity:.3f} m "
        f"(B/6 = {B / 6:.3f} m) \u2014 {eccentricity_status}."
    )
    # The verdict in plain words: failed checks first, then not checked.
    summary_items.append(f"Result: {result.verdict}")

    sections.append(CalcSection(title="Stability Summary", items=summary_items))

    return sections


def _calc_steps_mse(result, analysis) -> List[CalcSection]:
    """Calculation steps for MSE wall. The external steps print the terms
    the ASD analysis used (GM-143, GM-164; :func:`_mse_terms`)."""
    geom = _get(analysis, "geom")
    reinforcement = _get(analysis, "reinforcement")
    q_all = _get(analysis, "q_allowable")

    if geom.backfill_slope > 0:
        # The steps below print a level backfill (Rankine Ka); printing
        # them beside a sloping-backfill result would contradict it (GM-112).
        raise ValueError(
            "MSE calculation steps cover a level backfill only; the analysis "
            "result itself (mse_wall / analyze_mse_wall) handles a sloping "
            "backfill per GEC-11 Section 4.4.4 and reports its basis in "
            "'backslope'.")

    t = _mse_terms(result, analysis)
    phi, phi_f = t["phi_backfill"], t["phi_foundation"]

    sections = []
    H = geom.wall_height
    L = geom.reinforcement_length
    q = geom.surcharge
    forces = t["vertical_forces"]
    thrusts = t["horizontal_forces"]

    # ── Section 1: Earth Pressure ──────────────────────────────
    ep_items = []

    Ka = t["Ka"]
    phi_b, gamma_b = t["phi_retained"], t["gamma_retained"]
    ep_items.append(CalcStep(
        title="Rankine Active Earth Pressure Coefficient (retained fill)",
        equation="K_a = tan²(45° - φ_b/2)",
        substitution=f"K_a = tan²(45° - {phi_b:.1f}°/2)",
        result_name="K_a",
        result_value=f"{Ka:.4f}",
        reference="Rankine (1857); GEC-11 uses Rankine for MSE walls",
    ))

    # Active force: the thrusts the analysis summed
    Pa_total = t["Pa"]
    ep_items.append(CalcStep(
        title="Total Active Force on Reinforced Zone",
        equation="P_a = 0.5 × K_a × γ_b × H²"
                 + (" + K_a × q × H" if q > 0 else ""),
        substitution=(
            f"P_a = 0.5 × {Ka:.4f} × {gamma_b:.1f} × {H:.2f}²"
            + (f" + {Ka:.4f} × {q:.1f} × {H:.2f}" if q > 0 else "")
        ),
        result_name="P_a",
        result_value=f"{Pa_total:.1f}",
        result_unit="kN/m",
        reference="GEC-11, Ch. 4",
        notes=("The live-load surcharge's thrust (K_a q H) drives every "
               "check." if q > 0 else ""),
    ))

    sections.append(CalcSection(title="Earth Pressure Computation", items=ep_items))

    # ── Section 2: External Stability ──────────────────────────
    ext_items = []

    # Every vertical force the analysis summed and the checks that count it
    # (GM-164: a live load over the reinforced zone is left out of sliding
    # and overturning and counted in bearing, GEC-11 Fig. 4-2)
    V, M_stab = t["V"], t["M_stabilizing"]
    V_b, M_b = t["V_bearing"], t["M_bearing"]
    split = abs(V - V_b) > 1e-9 or abs(M_stab - M_b) > 1e-9
    marked = split or any(not f["resists"] or not f["bearing"]
                          for f in forces)
    rows = []
    for f in forces:
        row = [f["component"], f"{f['W']:.1f}", f"{f['x']:.3f}",
               f"{f['M']:.1f}"]
        if marked:
            row.append(_counted_in(f, "sliding, overturning"))
        rows.append(row)
    if split:
        rows.append(["TOTAL, sliding / overturning", f"{V:.1f}", "—",
                     f"{M_stab:.1f}", ""])
        rows.append(["TOTAL, bearing", f"{V_b:.1f}", "—", f"{M_b:.1f}",
                     ""])
    else:
        rows.append(["TOTAL", f"{V:.1f}", "—", f"{M_stab:.1f}"]
                    + ([""] if marked else []))
    headers = ["Component", "Weight (kN/m)", "Arm from Toe (m)",
               "Moment (kN·m/m)"] + (["Counted in"] if marked else [])
    left_out = sum(f["W"] for f in forces
                   if f["kind"] == "surcharge" and not f["resists"])
    notes = "Moments about the toe of the reinforced zone."
    if marked:
        notes += (" Counted in: the checks that sum the force. The live load "
                  "over the reinforced zone is not a stabilizing force in "
                  "sliding or overturning (GEC-11 Vol I pp. 4-19, 4-21 "
                  "Eq. 4-12, 4-22) and is applied for bearing (p. 4-26; "
                  "Fig. 4-2); see Choices."
                  if left_out > 0 else
                  " Counted in: the checks that sum the force (see Choices).")
    ext_items.append(TableData(
        title="Vertical Forces and Moments about the Toe",
        headers=headers,
        rows=rows,
        notes=notes,
    ))

    delta_b = t["delta_b"]
    ca = t["ca"]
    R_slide = t["resisting"]
    ext_items.append(CalcStep(
        title="Sliding Resistance",
        equation=("R = V × tan(δ_b) + c_a × L,  "
                  "δ_b = min(φ_r, φ_f)"),
        substitution=(
            f"δ_b = min({phi:.1f}°, {phi_f:.1f}°); "
            f"R = {V:.1f} × tan({delta_b:.2f}°)"
            + (f" + {ca:.1f} × {L:.2f}" if ca > 0 else "")
        ),
        result_name="R",
        result_value=f"{R_slide:.1f}",
        result_unit="kN/m",
        reference="GEC-11 Vol I Eq. 4-12 (soil on soil)",
        notes=(f"V leaves out the live load over the reinforced zone, "
               f"{left_out:.1f} kN/m (GEC-11 p. 4-19)."
               if left_out > 0 else ""),
    ))

    ext_items.append(CalcStep(
        title="Factor of Safety Against Sliding",
        equation="FOS_sliding = R / P_a",
        substitution=f"FOS_sliding = {R_slide:.1f} / {Pa_total:.1f}",
        result_name="FOS_sliding",
        result_value=f"{result.FOS_sliding:.3f}",
        reference="GEC-11 (min 1.5)",
    ))

    # Overturning: the thrusts' moments as the analysis summed them
    M_over = t["M_overturning"]
    ext_items.append(CalcStep(
        title="Overturning Moment about Toe",
        equation="M_ot = Σ P × z",
        substitution="M_ot = " + " + ".join(
            f"{f['P']:.1f} × {f['z']:.3f}" for f in thrusts),
        result_name="M_ot",
        result_value=f"{M_over:.1f}",
        result_unit="kN·m/m",
    ))

    ext_items.append(CalcStep(
        title="Stabilizing Moment",
        equation="M_stab = Σ W × x (the forces counted in overturning)",
        substitution="M_stab = " + " + ".join(
            f"{f['W']:.1f} × {f['x']:.3f}" for f in forces
            if f["resists"]),
        result_name="M_stab",
        result_value=f"{M_stab:.1f}",
        result_unit="kN·m/m",
    ))

    ext_items.append(CalcStep(
        title="Factor of Safety Against Overturning",
        equation="FOS_ot = M_stab / M_ot",
        substitution=f"FOS_ot = {M_stab:.1f} / {M_over:.1f}",
        result_name="FOS_ot",
        result_value=f"{result.FOS_overturning:.3f}",
        reference="GEC-11 (min 2.0)",
    ))

    # Bearing — Meyerhof uniform pressure over effective width (RW-3), on
    # the bearing loads: e_B is not the overturning check's (GEC-11 p. 4-26)
    e_B = t["eccentricity"]
    sigma_v = t["sigma_v"]
    ext_items.append(CalcStep(
        title="Eccentricity & Bearing Pressure (Meyerhof effective width)",
        equation=("e_B = L/2 - (M_b - M_ot)/V_b;  "
                  "σ_v = V_b / (L - 2e_B)"),
        substitution=(
            f"e_B = {L:.2f}/2 - ({M_b:.1f} - {M_over:.1f})/{V_b:.1f} "
            f"= {e_B:.3f} m;  σ_v = {V_b:.1f} / ({L:.2f} - "
            f"2×{max(e_B, 0.0):.3f})"
        ),
        result_name="σ_v",
        result_value=f"{sigma_v:.1f}",
        result_unit="kPa",
        reference="AASHTO 11.10.5.4; GEC-11 Eqs. 4-19, 4-20",
        notes=("V_b and M_b count the live load over the reinforced zone "
               "(GEC-11 p. 4-26), so e_B is not the eccentricity of the "
               "sliding and overturning loads."
               if split else ""),
    ))

    bearing_checked = q_all is not None and result.FOS_bearing is not None
    if bearing_checked:
        ext_items.append(CalcStep(
            title="Factor of Safety Against Bearing",
            equation="FOS_bearing = q_allowable / σ_v",
            substitution=f"FOS_bearing = {q_all:.1f} / {sigma_v:.1f}",
            result_name="FOS_bearing",
            result_value=f"{result.FOS_bearing:.3f}",
        ))
    else:
        # GM-100: a bearing check not run is said, never counted as a pass.
        ext_items.append(
            "Bearing capacity: NOT CHECKED — no allowable bearing "
            "pressure (q_allowable) was given. The bearing pressure σ_v "
            "above is reported; compare it with the foundation's allowable "
            "bearing pressure.")

    # External stability summary: only the checks that were run
    fos_s_req = result.FOS_sliding_required
    fos_o_req = result.FOS_overturning_required
    pass_s = result.FOS_sliding >= fos_s_req
    pass_o = result.FOS_overturning >= fos_o_req
    run_fos = [result.FOS_sliding, result.FOS_overturning]
    run_pass = [pass_s, pass_o]
    if bearing_checked:
        run_fos.append(result.FOS_bearing)
        run_pass.append(result.FOS_bearing >= 1.0)
    ext_items.append(CheckItem(
        description=("External stability (sliding, overturning, bearing)"
                     if bearing_checked else
                     "External stability (sliding, overturning; bearing "
                     "NOT CHECKED)"),
        demand=1.0,
        demand_label="Required",
        capacity=min(run_fos),
        capacity_label="Min FOS",
        passes=all(run_pass),
    ))

    ext_summary = TableData(
        title="External Stability Summary",
        headers=["Check", "FOS", "Required", "Status"],
        rows=[
            ["Sliding", f"{result.FOS_sliding:.3f}", f"{fos_s_req:.1f}",
             "OK" if pass_s else "FAIL"],
            ["Overturning", f"{result.FOS_overturning:.3f}",
             f"{fos_o_req:.1f}", "OK" if pass_o else "FAIL"],
            (["Bearing", f"{result.FOS_bearing:.3f}", "1.0",
              "OK" if result.FOS_bearing >= 1.0 else "FAIL"]
             if bearing_checked else
             ["Bearing", "—", "N/A", "NOT CHECKED"]),
        ],
    )
    ext_items.append(ext_summary)

    sections.append(CalcSection(title="External Stability", items=ext_items))

    # ── Section 3: Internal Stability ──────────────────────────
    int_items = []

    r_type = "Metallic" if (reinforcement and reinforcement.is_metallic) else "Geosynthetic"

    int_items.append(CalcStep(
        title="Lateral Earth Pressure Ratio (Kr/Ka)",
        equation="Kr/Ka varies with depth per GEC-11 Fig 4-10"
                 if r_type == "Metallic"
                 else "Kr/Ka = 1.0 for geosynthetic reinforcement",
        substitution=(
            "Metallic: Kr/Ka = 1.7 at z=0 to 1.2 at z\u226520 ft (6.096 m)"
            if r_type == "Metallic"
            else "Geosynthetic: Kr/Ka = 1.0 (constant)"
        ),
        result_name="Kr/Ka",
        result_value="variable" if r_type == "Metallic" else "1.0",
        reference="GEC-11, Figure 4-10",
    ))

    int_items.append(CalcStep(
        title="Maximum Reinforcement Tension",
        equation="T_max = (Kr/Ka) \u00d7 K_a \u00d7 \u03c3_v \u00d7 S_v",
        substitution="\u03c3_v = \u03b3\u00b7z + q",
        result_name="T_max",
        result_value="(per level)",
        result_unit="kN/m",
        reference="GEC-11, Eq. 4-10",
    ))

    int_items.append(CalcStep(
        title="Pullout Resistance",
        equation="P_r = F* \u00d7 \u03b1 \u00d7 \u03c3_v' \u00d7 L_e \u00d7 C",
        substitution=(
            "L_e = L - L_a; L_a per bilinear coherent-gravity surface "
            "(0.3H over upper H/2, tapering to toe) for inextensible "
            "(metallic) reinforcement; C = 2"
            if r_type == "Metallic" else
            "L_e = L - L_a; L_a from Rankine 45+\u03c6/2 plane "
            "(extensible/geosynthetic reinforcement); C = 2"
        ),
        result_name="P_r",
        result_value="(per level)",
        result_unit="kN/m",
        reference="GEC-11, Eq. 4-18; AASHTO Fig. 11.10.6.3.1-1",
    ))

    # Per-level results table
    if result.internal_results:
        int_rows = []
        for r in result.internal_results:
            status = "OK" if r.get("passes", False) else "FAIL"
            int_rows.append([
                f"{r['depth_m']:.2f}",
                f"{r.get('Kr_Ka', 0):.3f}",
                f"{r['Tmax_kN_per_m']:.2f}",
                f"{r.get('Tallowable_kN_per_m', 0):.1f}",
                f"{r.get('Le_m', 0):.2f}",
                f"{r.get('F_star', 0):.3f}",
                f"{r['Pr_kN_per_m']:.2f}",
                f"{r['FOS_pullout']:.2f}",
                f"{r.get('FOS_rupture', 0):.2f}",
                status,
            ])

        int_items.append(TableData(
            title="Internal Stability per Reinforcement Level",
            headers=["z (m)", "Kr/Ka", "T_max (kN/m)", "T_al (kN/m)",
                     "L_e (m)", "F*", "P_r (kN/m)",
                     "FOS_po", "FOS_ru", "Status"],
            rows=int_rows,
            notes=(
                f"FOS_pullout required: 1.5. FOS_rupture required: 1.0. "
                f"Reinforcement: {reinforcement.name if reinforcement else 'N/A'}"
            ),
        ))

    int_items.append(CheckItem(
        description="Internal stability (all levels)",
        demand=1.0,
        demand_label="Required",
        capacity=1.0 if result.all_pass_internal else 0.0,
        capacity_label="All pass",
        passes=result.all_pass_internal,
    ))

    sections.append(CalcSection(title="Internal Stability (GEC-11)", items=int_items))

    # ── Section 4: Overall Summary ─────────────────────────────
    summary_items = []

    def _status(passes):
        if passes is None:
            return "NOT ALL CHECKED (bearing not checked)"
        return "OK" if passes else "FAIL"

    if result.passes_external is False or not result.all_pass_internal:
        overall_pass = False
    elif result.passes_external is None:
        overall_pass = None
    else:
        overall_pass = True
    summary_items.append(TableData(
        title="Overall MSE Wall Assessment",
        headers=["Category", "Status"],
        rows=[
            ["External Stability", _status(result.passes_external)],
            ["Internal Stability", _status(result.all_pass_internal)],
            ["Overall", _status(overall_pass)],
        ],
    ))
    summary_items.append(f"Result: {result.verdict}")

    sections.append(CalcSection(title="Overall Assessment", items=summary_items))

    return sections


# ════════════════════════════════════════════════════════════════════
#  FIGURES
# ════════════════════════════════════════════════════════════════════

def get_figures(result, analysis) -> List[FigureData]:
    """Generate figures for the retaining wall calc package.

    Parameters
    ----------
    result : CantileverWallResult or MSEWallResult
        Computed results.
    analysis : dict
        Analysis parameters.

    Returns
    -------
    list of FigureData
    """
    wtype = _wall_type(result, analysis)
    figures = []

    if wtype == "mse":
        # MSE wall cross-section
        try:
            fig = _plot_mse_section(result, analysis)
            b64 = figure_to_base64(fig, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig)
            figures.append(FigureData(
                title="MSE Wall Cross-Section",
                image_base64=b64,
                caption=(
                    f"Figure 1: MSE wall cross-section. "
                    f"H = {result.wall_height:.2f} m, "
                    f"L = {result.reinforcement_length:.2f} m, "
                    f"{result.n_levels} reinforcement levels."
                ),
                width_percent=80,
            ))
        except ImportError:
            pass

        # Internal stability profile
        if result.internal_results:
            try:
                fig2 = _plot_mse_internal(result, analysis)
                b64_2 = figure_to_base64(fig2, dpi=150)
                import matplotlib.pyplot as plt
                plt.close(fig2)
                figures.append(FigureData(
                    title="Internal Stability Profile",
                    image_base64=b64_2,
                    caption=(
                        "Figure 2: T_max and pullout resistance vs. depth. "
                        "All levels must have FOS_pullout >= 1.5."
                    ),
                    width_percent=75,
                ))
            except ImportError:
                pass
    else:
        # Cantilever wall cross-section
        try:
            fig = _plot_cantilever_section(result, analysis)
            b64 = figure_to_base64(fig, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig)
            figures.append(FigureData(
                title="Cantilever Wall Cross-Section",
                image_base64=b64,
                caption=(
                    f"Figure 1: Cantilever wall cross-section. "
                    f"H = {result.wall_height:.2f} m, "
                    f"B = {result.base_width:.2f} m."
                ),
                width_percent=80,
            ))
        except ImportError:
            pass

        # Bearing pressure diagram
        try:
            fig2 = _plot_bearing_pressure(result, analysis)
            b64_2 = figure_to_base64(fig2, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig2)
            figures.append(FigureData(
                title="Base Bearing Pressure Distribution",
                image_base64=b64_2,
                caption=(
                    f"Figure 2: Bearing pressure distribution along base. "
                    f"q_toe = {result.q_toe:.1f} kPa, "
                    f"q_heel = {result.q_heel:.1f} kPa."
                ),
                width_percent=70,
            ))
        except ImportError:
            pass

    return figures


# ═══════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════════════

def _cantilever_terms(result, analysis):
    """(earth pressure, vertical forces, sliding, overturning, bearing):
    the terms the cantilever analysis used (GM-143). The result's own;
    a result built by hand carries none, and then the module's own term
    functions work them out from the inputs the run used."""
    if getattr(result, "bearing_terms", None) is not None:
        return (result.earth_pressure, result.vertical_forces,
                result.sliding_terms, result.overturning_terms,
                result.bearing_terms)
    from retaining_walls import cantilever as cw
    geom = _get(analysis, "geom")
    gamma = _used(analysis, "gamma_backfill")
    phi = _used(analysis, "phi_backfill")
    c = _used(analysis, "c_backfill")
    method = _used(analysis, "pressure_method")
    gamma_c = _used(analysis, "gamma_concrete")
    ep = cw._active_thrust_terms(geom, gamma, phi, c, method)
    live = bool(_used(analysis, "live_load_resists"))
    dead = bool(_used(analysis, "dead_load_resists"))
    in_bearing = bool(_used(analysis, "surcharge_in_bearing"))
    return (
        ep, cw._vertical_force_items(geom, gamma, gamma_c, ep, live, dead,
                                     in_bearing),
        cw._sliding_terms(
            geom, gamma, phi, c, _used(analysis, "phi_foundation"),
            _used(analysis, "c_foundation"), gamma_c, method,
            bool(_used(analysis, "include_passive")),
            _used(analysis, "gamma_foundation"),
            delta_base=_used(analysis, "delta_base"),
            base_adhesion=_used(analysis, "base_adhesion"),
            live_load_resists=live, dead_load_resists=dead,
            sliding_cohesion=_used(analysis, "sliding_cohesion")),
        cw._overturning_terms(geom, gamma, phi, c, gamma_c, method,
                              live_load_resists=live,
                              dead_load_resists=dead),
        cw._bearing_terms(geom, gamma, phi, c,
                          _used(analysis, "q_allowable"), gamma_c, method,
                          surcharge_in_bearing=in_bearing))


def _loads_differ(ot, br) -> bool:
    """Whether the sliding / overturning / eccentricity loads and the
    bearing loads differ (GM-157: a surcharge over the heel counted in
    one and not the other)."""
    return (abs(ot["V"] - br["V"]) > 1e-9
            or abs(ot["M_stabilizing"] - br["M_stab"]) > 1e-9)


def _counted_in(force, resisting="sliding, overturning, eccentricity") -> str:
    """The checks a vertical force is counted in (GM-157); ``resisting``
    names the checks its ``resists`` flag stands for."""
    resists, bearing = force.get("resists", True), force.get("bearing", True)
    if resists and bearing:
        return "all checks"
    if bearing:
        return "bearing only"
    if resists:
        return resisting
    return "none"


def _mse_used(analysis, key):
    """An input the MSE run used: the analysis dict's, else the default of
    ``analyze_mse_wall``, which the run then took."""
    if isinstance(analysis, dict) and key in analysis:
        return analysis[key]
    if not isinstance(analysis, dict) and hasattr(analysis, key):
        return getattr(analysis, key)
    import inspect
    from retaining_walls.mse import analyze_mse_wall
    default = inspect.signature(
        analyze_mse_wall).parameters[key].default
    if default is inspect.Parameter.empty:
        raise ValueError(
            f"retaining_walls calc steps: the analysis gives no {key}, "
            f"which analyze_mse_wall requires; pass the inputs the run "
            f"was given")
    return default


def _mse_terms(result, analysis):
    """The ASD external terms the MSE analysis used (GM-143, GM-164): the
    result's own; a result built by hand carries none, and then the
    module's term function works them out from the inputs the run used."""
    if getattr(result, "external_terms", None) is not None:
        return result.external_terms
    from retaining_walls import mse
    phi = _mse_used(analysis, "phi_backfill")
    phi_f = _mse_used(analysis, "phi_foundation")
    return mse._external_terms(
        _get(analysis, "geom"), _mse_used(analysis, "gamma_backfill"), phi,
        phi if phi_f is None else phi_f,
        _mse_used(analysis, "c_foundation"),
        phi_retained=_mse_used(analysis, "phi_retained"),
        gamma_retained=_mse_used(analysis, "gamma_retained"),
        traffic_vertical_resists=bool(
            _mse_used(analysis, "traffic_vertical_resists")),
        live_load_resists=bool(_mse_used(analysis, "live_load_resists")),
        surcharge_in_bearing=bool(
            _mse_used(analysis, "surcharge_in_bearing")))


def _Ka_value(phi, method, geom):
    """Compute Ka exactly as the cantilever analysis does, so the rendered calc
    package matches the analysis.

    Mirrors retaining_walls.cantilever: Coulomb uses wall friction delta = 2/3*phi
    with the backfill slope; Rankine uses the sloped coefficient when the
    backfill slopes and the level coefficient otherwise. (Previously this always
    returned the level Rankine value, so the package disagreed with the analysis
    for sloped backfills — CS-1.)
    """
    from retaining_walls.earth_pressure import (
        rankine_Ka, rankine_Ka_sloped, coulomb_Ka,
    )
    if method == "coulomb":
        delta = 2.0 / 3.0 * phi  # matches analysis delta_wall
        return coulomb_Ka(phi, delta, beta_deg=geom.backfill_slope)
    if geom.backfill_slope > 0:
        return rankine_Ka_sloped(phi, geom.backfill_slope)
    return rankine_Ka(phi)


def _plot_cantilever_section(result, analysis):
    """Create a schematic cross-section of a cantilever retaining wall."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    geom = _get(analysis, "geom")
    gamma = _used(analysis, "gamma_backfill")
    phi = _used(analysis, "phi_backfill")
    method = _used(analysis, "pressure_method")

    H = geom.wall_height
    B = geom.base_width
    t_toe = geom.toe_length
    t_stem_b = geom.stem_thickness_base
    t_stem_t = geom.stem_thickness_top
    h_stem = geom.stem_height
    t_base = geom.base_thickness
    heel = geom.heel_length

    fig, ax = plt.subplots(figsize=(8, 6))

    # Coordinate system: x from left, y from bottom of base (y=0)

    # Base slab
    base = patches.Rectangle(
        (0, 0), B, t_base,
        facecolor='#94a3b8', edgecolor='#1a1a1a', linewidth=1.5,
    )
    ax.add_patch(base)

    # Stem (trapezoid: wider at base, narrower at top)
    stem_x = [
        t_toe,                              # bottom-left (front face)
        t_toe,                              # top-left (front face)
        t_toe + t_stem_t,                   # top-right
        t_toe + t_stem_b,                   # bottom-right
    ]
    stem_y = [
        t_base,                             # bottom-left
        t_base + h_stem,                    # top-left
        t_base + h_stem,                    # top-right
        t_base,                             # bottom-right
    ]
    ax.fill(stem_x, stem_y, facecolor='#94a3b8', edgecolor='#1a1a1a', linewidth=1.5)

    # The earth pressure as the analysis takes it (GM-143), and its height:
    # the vertical plane through the back of the heel, H_active high
    ep = _cantilever_terms(result, analysis)[0]
    Ka_val = ep["Ka"]
    H_a = ep["H_active"]
    q = (ep.get("q_live") or 0.0) + (ep.get("q_dead") or 0.0)
    p_q = Ka_val * q                       # the surcharge's, kPa
    p_soil = Ka_val * gamma * H_a          # the earth's, at the base
    # the diagram's widest, a set share of B, and the room right of it
    p_scale = B * 0.5 / max(p_q + p_soil, 1e-9)
    w_diagram = (p_q + p_soil) * p_scale

    # The ground on each side, as the analysis has it (GM-200: the toe
    # side's ground was drawn at H, level with the backfill). Behind the
    # wall the backfill meets the stem at its top and rises at the
    # backfill slope from the back of the stem at the heel, where the
    # analysis's wedge over the heel starts, so it meets the plane through
    # the back of the heel at H_active (GM-216); in front, the ground is
    # at the top of the base: H is measured from the underside of the
    # base, the wall retains H less the base thickness above the top of
    # the footing, and passive resistance is taken on the base's
    # embedment (``cantilever._sliding_terms``).
    x_left, x_right = -B * 0.4, B + w_diagram + B * 0.3
    slope = math.tan(math.radians(geom.backfill_slope or 0.0))
    x_back_top = t_toe + t_stem_t
    x_heel = t_toe + t_stem_b

    def _backfill_surface(x):
        return H + max(x - x_heel, 0.0) * slope

    backfill_x = [x_heel, B, B, x_right, x_right, x_heel, x_back_top]
    backfill_y = [t_base, t_base, 0.0, 0.0, _backfill_surface(x_right), H,
                  H]
    ax.fill(backfill_x, backfill_y, facecolor='#f5e6c8', edgecolor='none',
            alpha=0.7, zorder=0)

    # A shear key: where it is given, else under the stem, as UFC 3-220-20
    # Fig. 4-29 and GEC-6 Fig. 5-22 draw it, and the figure says so
    # (GM-216); the analysis has no place for it
    key_depth = geom.key_depth if geom.has_shear_key else 0.0
    key_lines = []
    if key_depth > 0:
        key_x = _get(analysis, "key_distance_from_toe")
        key_w = _get(analysis, "key_width")
        if key_x is None or key_w is None:
            missing = ("position or width" if key_x is None
                       and key_w is None else
                       "position" if key_x is None else "width")
            key_lines += [("Shear key drawn under the stem, as UFC 3-220-20 "
                           "Fig. 4-29 draws it:"),
                          f"no key {missing} given."]
        key_x = t_toe if key_x is None else float(key_x)
        key_w = t_stem_b if key_w is None else float(key_w)
        ax.add_patch(patches.Rectangle(
            (key_x, -key_depth), key_w, key_depth, facecolor='#94a3b8',
            edgecolor='#1a1a1a', linewidth=1.5))
        if not _used(analysis, "include_passive"):
            key_lines.append("No passive resistance in the sliding check: "
                             "the key is not counted.")

    # Foundation soil under the base and the soil in front of the toe
    y_bottom = -B * 0.45 - key_depth
    ax.fill([x_left, x_right, x_right, 0.0, 0.0, x_left],
            [y_bottom, y_bottom, 0.0, 0.0, t_base, t_base],
            facecolor='#e8dcc8', edgecolor='none', alpha=0.5, zorder=0)

    # The pressure diagram on the plane through the back of the heel, from
    # the underside of the base to H_active, where the analysis applies it
    # (it was drawn on the stem's face, over the stem only; GM-216): the
    # surcharge's K_a q next to the plane, the earth's K_a gamma z beyond
    press = dict(linewidth=1.2, alpha=0.5, zorder=1)
    if p_q > 0:
        ax.add_patch(patches.Rectangle(
            (B, 0.0), p_q * p_scale, H_a, facecolor='#fdba74',
            edgecolor='#c2410c', hatch='..', **press))
    x_q = B + p_q * p_scale
    ax.fill([x_q, x_q + p_soil * p_scale, x_q], [0.0, 0.0, H_a],
            facecolor='#fca5a5', edgecolor='#dc2626', **press)
    # The thrust, at its height above the base and inclination as the
    # analysis has them, named in the key (written at its tail it ran
    # past the frame at the printed width)
    delta = math.radians(ep.get("delta_deg") or 0.0)
    z_pa = ep.get("z_Pa", H_a / 3.0)
    reach = w_diagram + 0.12 * B
    tail = (B + reach * math.cos(delta), z_pa + reach * math.sin(delta))
    ax.annotate('', xy=(B, z_pa), xytext=tail,
                arrowprops=dict(arrowstyle='-|>', color='#991b1b', lw=1.6,
                                mutation_scale=12), zorder=4)
    # The surcharge on the backfill it stands for
    if q > 0:
        for i in range(7):
            x = x_heel + 0.1 * B + i * (x_right - 0.18 * B - x_heel) / 6
            ax.annotate('', xy=(x, _backfill_surface(x)), xytext=(0, 12),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='#c2410c',
                                        lw=0.8))

    # Dimension annotations
    # Wall height
    ax.annotate('', xy=(-B * 0.15, 0), xytext=(-B * 0.15, H),
                arrowprops=dict(arrowstyle='<->', color='#1a1a1a', lw=1.2))
    ax.text(-B * 0.22, H / 2, f'H = {H:.2f} m', ha='right', va='center',
            fontsize=9, fontweight='bold', rotation=90)

    # The dimensions under the base, set off from it by points (by metres
    # they ran into each other and their values onto their lines at the
    # printed width; GM-194): toe and heel 4 pt under it, B under those;
    # under the shear key where there is one (GM-216).
    from matplotlib.transforms import offset_copy

    def _under(pts):
        return offset_copy(ax.transData, fig=fig, y=-pts, units='points')

    def _dimension(x0, x1, pts, text, **kw):
        line = dict(color=kw.pop('color', '#1a1a1a'), lw=kw.pop('lw', 1.2))
        ax.annotate('', xy=(x0, -key_depth), xytext=(x1, -key_depth),
                    xycoords=_under(pts), textcoords=_under(pts),
                    arrowprops=dict(arrowstyle='<->', **line))
        ax.text((x0 + x1) / 2, -key_depth, text, transform=_under(pts + 4),
                ha='center', va='top', color=line['color'], **kw)

    if t_toe > 0:
        _dimension(0, t_toe, 4, f'toe={t_toe:.2f}', fontsize=7,
                   color='#666', lw=0.8)
    if heel > 0:
        _dimension(t_toe + t_stem_b, B, 4, f'heel={heel:.2f}', fontsize=7,
                   color='#666', lw=0.8)
    _dimension(0, B, 20, f'B = {B:.2f} m', fontsize=9, fontweight='bold')

    # Ground surface lines: the backfill behind the stem, the ground in
    # front at the top of the base
    ax.plot([x_back_top, x_heel, x_right],
            [H, H, _backfill_surface(x_right)], 'g-', linewidth=2)
    ax.plot([x_left, 0.0], [t_base, t_base], 'g-', linewidth=2)

    # Resultant location marker, its point on the base's underside, and
    # its label beside it inside the base (under the base it ran into B)
    x_R_scaled = (result.base_width / 2.0 - result.eccentricity)
    ax.plot(x_R_scaled, 0, 'rv', markersize=10,
            transform=offset_copy(ax.transData, fig=fig, y=4,
                                  units='points'))
    ax.annotate(f'R (e={result.eccentricity:.3f}m)', (x_R_scaled, 0),
                xytext=(7, 5), textcoords='offset points', ha='left',
                va='center', fontsize=7, color='#dc2626')

    # The soil and the factors of safety under the drawing: in boxes in
    # the axes they covered the stem and ran over the frame at the
    # printed width (GM-194). A check with no FOS says why (GM-100).
    def _fos_text(fos, word):
        return word if fos is None else f"{fos:.2f}"

    lines = [
        f"Active pressure on the vertical plane through the back of the "
        f"heel, {H_a:.2f} m high",
        f'\u03b3 = {gamma:.1f} kN/m\u00b3, \u03c6 = {phi:.1f}\u00b0, '
        f'Ka = {Ka_val:.3f}']
    if ep.get("Pa_cohesion"):
        lines.append(f"The backfill's cohesion takes "
                     f"{-ep['Pa_cohesion']:.0f} kN/m off P_a (not drawn).")
    lines.append(
        f"FOS_slide = {_fos_text(result.FOS_sliding, 'no driving force')}, "
        f"FOS_OT = {_fos_text(result.FOS_overturning, 'no driving force')}, "
        f"FOS_brg = {_fos_text(result.FOS_bearing, 'not checked')}")
    lines += key_lines
    from matplotlib.lines import Line2D
    handles = [Line2D(
        [], [], color='#991b1b', lw=1.6, marker='<', markersize=6,
        label=(f'P_a = {ep["Pa"]:.0f} kN/m, {z_pa:.2f} m above the base'
               + (f', at {ep["delta_deg"]:.0f}° to horizontal'
                  if ep.get("delta_deg") else '')))]
    handles += [patches.Patch(
        facecolor='#fca5a5', edgecolor='#dc2626', alpha=0.5,
        label=f'Earth, K_a\u03b3z: {p_soil:.1f} kPa at the base')]
    if p_q > 0:
        handles.append(patches.Patch(
            facecolor='#fdba74', edgecolor='#c2410c', hatch='..', alpha=0.5,
            label=f'Surcharge, K_a q: {p_q:.1f} kPa (q = {q:g} kPa)'))

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, max(H * 1.15, H_a + 0.05 * H,
                              _backfill_surface(x_right)
                              + (0.12 if q > 0 else 0.05) * H))
    # the wall's own frame, said as such: the inputs give no elevation
    # (the axis read "Elevation" with 0 at the underside of the base)
    ax.set_xlabel('Distance from the toe (m)', fontsize=10)
    ax.set_ylabel('Height above the underside of the base (m)', fontsize=10)
    ax.set_title('Cantilever Retaining Wall Cross-Section', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    from geotech_common.plotting import key_under_axes
    key_under_axes(fig, ax, handles=handles, note="\n".join(lines), ncol=1)
    return fig


def _plot_bearing_pressure(result, analysis):
    """Create a bearing pressure distribution diagram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    geom = _get(analysis, "geom")
    B = geom.base_width
    q_toe = result.q_toe
    q_heel = result.q_heel

    fig, ax = plt.subplots(figsize=(7, 3.5))

    # x along base from toe (0) to heel (B); outside the middle third the
    # pressure is a triangle over the width the analysis used (GM-143)
    terms = getattr(result, "bearing_terms", None) or {}
    B_eff = terms.get("B_eff")
    if terms.get("distribution") == "triangular_toe" and B_eff:
        x, q = [0, min(B_eff, B)], [q_toe, 0.0]
    elif terms.get("distribution") == "triangular_heel" and B_eff:
        x, q = [max(B - B_eff, 0.0), B], [0.0, q_heel]
    else:
        x = [0, B]
        q = [q_toe, q_heel]

    # Fill the pressure distribution
    ax.fill_between(x, 0, q, color='#fca5a5', alpha=0.5, edgecolor='#dc2626', linewidth=1.5)
    ax.plot(x, q, 'r-', linewidth=2)

    # Base line
    ax.axhline(y=0, color='k', linewidth=1)

    # Middle-third zone
    b6 = B / 6.0
    ax.axvspan(B / 2 - b6, B / 2 + b6, alpha=0.1, color='#16a34a',
               label=f'Middle third (B/6 = {b6:.2f} m)')

    # Eccentricity marker: the resultant of the loads drawn, the bearing
    # check's (GM-157: it can differ from the eccentricity check's)
    e = terms.get("eccentricity", result.eccentricity)
    x_R = B / 2.0 - e
    differs = getattr(result, "eccentricity_bearing", None) is not None
    ax.axvline(x=x_R, color='#dc2626', linestyle='--', linewidth=1.2,
               label=(f'Resultant for bearing (e_b = {e:.3f} m)' if differs
                      else f'Resultant (e = {e:.3f} m)'))
    ax.axvline(x=B / 2.0, color='#666', linestyle=':', linewidth=0.8,
               label='Base center')

    q_all = _used(analysis, "q_allowable")
    if q_all is not None:
        ax.axhline(y=q_all, color='#16a34a', linestyle='--', linewidth=1.2,
                   label=f'q_allowable = {q_all:.0f} kPa')

    ax.set_xlabel('Distance from Toe (m)', fontsize=10)
    ax.set_ylabel('Bearing Pressure (kPa)', fontsize=10)
    ax.set_title('Base Bearing Pressure Distribution', fontsize=11, fontweight='bold')
    ax.set_xlim(-B * 0.05, B * 1.05)
    ax.set_ylim(0, max(q_toe, q_heel, q_all or 0) * 1.3 + 5)
    ax.grid(True, alpha=0.2)
    # the key under the axes, headed by the edge pressures: inside them it
    # covered the q_toe note once the figure was drawn for the width it
    # prints at (GM-186), and the notes over the diagram lay on the
    # resultant's line and the pressure line (GM-194)
    from geotech_common.plotting import key_under_axes
    key_under_axes(fig, ax, fontsize=7,
                   note=(f'q_toe = {q_toe:.1f} kPa, '
                         f'q_heel = {q_heel:.1f} kPa'))
    return fig


def _plot_mse_section(result, analysis):
    """Create a schematic cross-section of an MSE wall."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    geom = _get(analysis, "geom")
    reinforcement = _get(analysis, "reinforcement")
    gamma = _get(analysis, "gamma_backfill", 18.0)
    phi = _get(analysis, "phi_backfill", 34.0)

    H = geom.wall_height
    L = geom.reinforcement_length
    Sv = geom.reinforcement_spacing

    fig, ax = plt.subplots(figsize=(9, 6))

    # Reinforced soil block
    ax.add_patch(patches.Rectangle(
        (0, 0), L, H,
        facecolor='#f5e6c8', edgecolor='#1a1a1a', linewidth=2,
    ))

    # Wall face (thicker line on the left)
    ax.plot([0, 0], [0, H], 'k-', linewidth=3)

    # Reinforcement layers
    r_color = '#2563eb' if (reinforcement and reinforcement.is_metallic) else '#16a34a'
    for z in geom.reinforcement_depths:
        y = H - z  # convert depth to elevation
        ax.plot([0, L], [y, y], color=r_color, linewidth=1.5, alpha=0.7)
        ax.plot(L, y, '|', color=r_color, markersize=8, markeredgewidth=2)

    # Failure surface: bilinear coherent-gravity surface for inextensible
    # (metallic) reinforcement, Rankine 45+phi/2 plane for extensible (RW-2)
    if reinforcement is not None and reinforcement.is_metallic:
        x_fail_top = 0.3 * H
        ax.plot([0, 0.3 * H, 0.3 * H], [0, H / 2.0, H], 'r--',
                linewidth=2, alpha=0.7,
                label='Coherent-gravity surface (0.3H, inextensible)')
    else:
        failure_angle = math.radians(45 + phi / 2)
        # Line from toe going up at the failure angle
        x_fail_top = H / math.tan(failure_angle)
        ax.plot([0, x_fail_top], [0, H], 'r--', linewidth=2, alpha=0.7,
                label=f'Failure plane (45+\u03c6/2 = {45 + phi / 2:.0f}\u00b0)')

    # Retained soil behind reinforced zone
    ax.add_patch(patches.Rectangle(
        (L, 0), L * 0.4, H,
        facecolor='#e8dcc8', edgecolor='none', alpha=0.5,
    ))

    # Earth pressure arrows on back face of reinforced zone
    Ka = math.tan(math.radians(45 - phi / 2)) ** 2
    n_arrows = 5
    for i in range(1, n_arrows + 1):
        y_arr = H * (1 - i / n_arrows)
        z_arr = H - y_arr  # depth
        sigma = Ka * gamma * z_arr
        arrow_len = sigma / (Ka * gamma * H) * L * 0.3
        ax.annotate('', xy=(L, y_arr), xytext=(L + arrow_len, y_arr),
                    arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5))

    ax.text(L + L * 0.15, H * 0.1, 'P_a', fontsize=10, color='#dc2626', fontweight='bold')

    # Dimension annotations
    # Height
    ax.annotate('', xy=(-L * 0.08, 0), xytext=(-L * 0.08, H),
                arrowprops=dict(arrowstyle='<->', color='#1a1a1a', lw=1.2))
    ax.text(-L * 0.15, H / 2, f'H = {H:.2f} m', ha='right', va='center',
            fontsize=9, fontweight='bold', rotation=90)

    # Reinforcement length
    ax.annotate('', xy=(0, -H * 0.06), xytext=(L, -H * 0.06),
                arrowprops=dict(arrowstyle='<->', color='#1a1a1a', lw=1.2))
    ax.annotate(f'L = {L:.2f} m', (L / 2, -H * 0.06), xytext=(0, -2),
                textcoords='offset points', ha='center', va='top',
                fontsize=9, fontweight='bold')

    # Spacing annotation
    if len(geom.reinforcement_depths) >= 2:
        z1 = geom.reinforcement_depths[0]
        z2 = geom.reinforcement_depths[1]
        y1 = H - z1
        y2 = H - z2
        ax.annotate('', xy=(L * 1.05, y1), xytext=(L * 1.05, y2),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8))
        ax.text(L * 1.08, (y1 + y2) / 2, f'Sv = {Sv:.2f} m',
                fontsize=7, va='center', color='#666')

    # Labels: the zone's name between the failure surface and the back of
    # the zone, between two reinforcement levels, and ringed so a level
    # stops short of it (centred in the zone the surface ran through it;
    # GM-194). Measured from the surface where it crosses the upper of the
    # two levels: from the plane's top, it sat across the zone's back.
    from geotech_common.plotting import halo
    levels = sorted(H - z for z in geom.reinforcement_depths)
    bands = list(zip(levels, levels[1:])) or [(0.4 * H, 0.6 * H)]
    lo, hi = min(bands, key=lambda b: abs((b[0] + b[1]) / 2 - H / 2))
    if reinforcement is not None and reinforcement.is_metallic:
        x_surface = x_fail_top * min(1.0, hi / (H / 2.0))
    else:
        x_surface = x_fail_top * hi / H
    halo([ax.text((min(x_surface, L) + L) / 2, (lo + hi) / 2,
                  'Reinforced\nSoil Zone', ha='center', va='center',
                  fontsize=10, fontweight='bold', color='#6b6358')],
         colour='#f5e6c8')
    ax.text(L * 1.2, H / 2, 'Retained\nSoil', ha='center', va='center',
            fontsize=9, alpha=0.5)

    # Reinforcement legend
    r_name = reinforcement.name if reinforcement else "Reinforcement"
    ax.plot([], [], color=r_color, linewidth=2, label=r_name)

    # The factors of safety head the key, under the drawing: the box in
    # the axes covered the zone's top corner and the key covered L at the
    # printed width (GM-194). Red for a check that fails, green when every
    # check passes, dark grey when one was not made (bearing with no
    # allowable pressure): in red, "all pass" read as a failure.
    verdicts = (result.passes_external, result.all_pass_internal)
    if any(v is False for v in verdicts):
        status_color = '#dc2626'
    elif all(v is True for v in verdicts):
        status_color = '#16a34a'
    else:
        status_color = '#333333'
    fos_brg = getattr(result, 'FOS_bearing', None)
    fos_text = (
        f"External: slide={result.FOS_sliding:.2f}, OT={result.FOS_overturning:.2f}, "
        f"bearing={'not checked' if fos_brg is None else f'{fos_brg:.2f}'}; "
        f"Internal: {'all pass' if result.all_pass_internal else 'FAIL'}"
    )

    ax.set_xlim(-L * 0.25, L * 1.5)
    ax.set_ylim(-H * 0.15, H * 1.1)
    # the wall's own frame, said as such: the inputs give no elevation
    # (the axis read "Elevation" with 0 at the base of the reinforced
    # zone; GM-218, as GM-200 did for the cantilever section)
    ax.set_xlabel('Distance from the wall face (m)', fontsize=10)
    ax.set_ylabel('Height above the base of the reinforced zone (m)',
                  fontsize=10)
    ax.set_title('MSE Wall Cross-Section', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    from geotech_common.plotting import key_under_axes
    key = key_under_axes(fig, ax, note=fos_text)
    key.get_title().set_color(status_color)
    return fig


def _plot_mse_internal(result, analysis):
    """Create a Tmax vs pullout resistance depth profile."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = result.internal_results
    if not data:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, 'No internal stability data', transform=ax.transAxes,
                ha='center', fontsize=12)
        return fig

    depths = [r['depth_m'] for r in data]
    Tmax = [r['Tmax_kN_per_m'] for r in data]
    Pr = [r['Pr_kN_per_m'] for r in data]
    passes = [r.get('passes', False) for r in data]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot Tmax and Pr vs depth (depth increasing downward)
    ax.plot(Tmax, depths, 'ro-', linewidth=2, markersize=6, label='T_max (demand)')
    ax.plot(Pr, depths, 'bs-', linewidth=2, markersize=6, label='P_r (capacity)')

    # Mark failures
    for i, p in enumerate(passes):
        if not p:
            ax.plot(Tmax[i], depths[i], 'rx', markersize=12, markeredgewidth=3)

    ax.invert_yaxis()  # depth increases downward
    ax.set_xlabel('Force (kN/m)', fontsize=10)
    ax.set_ylabel('Depth below the top of the wall, z (m)', fontsize=10)
    ax.set_title('Internal Stability: T_max vs Pullout Resistance', fontsize=11,
                 fontweight='bold')
    ax.grid(True, alpha=0.3)
    # the key under the axes: inside them the deepest P_r ran through it
    # (GM-194)
    from geotech_common.plotting import key_under_axes
    key_under_axes(fig, ax, fontsize=9)
    return fig
