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
    geom = _get(analysis, "geom")
    gamma = _get(analysis, "gamma_backfill", 18.0)
    phi = _get(analysis, "phi_backfill", 30.0)
    c = _get(analysis, "c_backfill", 0.0)
    method = _get(analysis, "pressure_method", "rankine")
    gamma_c = _get(analysis, "gamma_concrete", 24.0)
    phi_f = _get(analysis, "phi_foundation")
    c_f = _get(analysis, "c_foundation", 0.0)
    q_all = _get(analysis, "q_allowable")
    FOS_s = _get(analysis, "FOS_sliding", 1.5)
    FOS_o = _get(analysis, "FOS_overturning", 2.0)

    items = [
        InputItem("Type", "Wall type", "Cantilever", ""),
        InputItem("H", "Wall height", f"{geom.wall_height:.2f}", "m"),
        InputItem("B", "Base width", f"{geom.base_width:.2f}", "m"),
        InputItem("t_toe", "Toe length", f"{geom.toe_length:.2f}", "m"),
        InputItem("t_heel", "Heel length", f"{geom.heel_length:.2f}", "m"),
        InputItem("t_stem_top", "Stem thickness (top)", f"{geom.stem_thickness_top:.2f}", "m"),
        InputItem("t_stem_base", "Stem thickness (base)", f"{geom.stem_thickness_base:.2f}", "m"),
        InputItem("t_base", "Base slab thickness", f"{geom.base_thickness:.2f}", "m"),
    ]

    if geom.backfill_slope > 0:
        items.append(InputItem("\u03b2", "Backfill slope", f"{geom.backfill_slope:.1f}", "deg"))
    if geom.surcharge > 0:
        items.append(InputItem("q_s", "Surcharge", f"{geom.surcharge:.1f}", "kPa"))

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
        items.append(InputItem("q_s", "Surcharge", f"{geom.surcharge:.1f}", "kPa"))

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
    geom = _get(analysis, "geom")
    gamma = _get(analysis, "gamma_backfill", 18.0)
    phi = _get(analysis, "phi_backfill", 30.0)
    c = _get(analysis, "c_backfill", 0.0)
    method = _get(analysis, "pressure_method", "rankine")
    gamma_c = _get(analysis, "gamma_concrete", 24.0)
    phi_f = _get(analysis, "phi_foundation", phi)
    c_f = _get(analysis, "c_foundation", 0.0)
    q_all = _get(analysis, "q_allowable")
    FOS_s_req = _get(analysis, "FOS_sliding", 1.5)
    FOS_o_req = _get(analysis, "FOS_overturning", 2.0)

    if phi_f is None:
        phi_f = phi

    sections = []

    # ── Section 1: Earth Pressure Coefficient ──────────────────
    ep_items = []

    phi_rad = math.radians(phi)

    if method == "coulomb":
        delta = 2.0 / 3.0 * phi
        alpha = 90.0  # vertical back face
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
            result_value=f"{_Ka_value(phi, method, geom):.4f}",
            reference="Coulomb (1776)",
        ))
    else:
        Ka = math.tan(math.radians(45 - phi / 2)) ** 2
        ep_items.append(CalcStep(
            title="Rankine Active Earth Pressure Coefficient",
            equation="K_a = tan\u00b2(45\u00b0 - \u03c6/2)",
            substitution=f"K_a = tan\u00b2(45\u00b0 - {phi:.1f}\u00b0/2)",
            result_name="K_a",
            result_value=f"{Ka:.4f}",
            reference="Rankine (1857)",
        ))

    # Active height
    H = geom.H_active
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

    # Active force
    Ka_val = _Ka_value(phi, method, geom)
    Pa_earth = 0.5 * Ka_val * gamma * H ** 2
    Pa_surcharge = Ka_val * geom.surcharge * H if geom.surcharge > 0 else 0
    Pa_cohesion = -2.0 * c * math.sqrt(Ka_val) * H if c > 0 else 0
    Pa_total = max(Pa_earth + Pa_surcharge + Pa_cohesion, 0.0)

    ep_items.append(CalcStep(
        title="Total Active Horizontal Force",
        equation="P_a = 0.5 \u00d7 K_a \u00d7 \u03b3 \u00d7 H\u00b2"
                 + (" + K_a \u00d7 q \u00d7 H" if geom.surcharge > 0 else "")
                 + (" - 2c\u221aK_a \u00d7 H" if c > 0 else ""),
        substitution=(
            f"P_a = 0.5 \u00d7 {Ka_val:.4f} \u00d7 {gamma:.1f} \u00d7 {H:.3f}\u00b2"
            + (f" + {Ka_val:.4f} \u00d7 {geom.surcharge:.1f} \u00d7 {H:.3f}"
               if geom.surcharge > 0 else "")
            + (f" - 2 \u00d7 {c:.1f} \u00d7 \u221a{Ka_val:.4f} \u00d7 {H:.3f}"
               if c > 0 else "")
        ),
        result_name="P_a",
        result_value=f"{Pa_total:.1f}",
        result_unit="kN/m",
        reference="Das Ch. 13",
    ))

    sections.append(CalcSection(title="Earth Pressure Computation", items=ep_items))

    # ── Section 2: Wall Weight Components ──────────────────────
    weight_items = []

    # Build the weight table
    B = geom.base_width
    t_toe = geom.toe_length
    t_stem_b = geom.stem_thickness_base
    t_stem_t = geom.stem_thickness_top
    h_stem = geom.stem_height
    t_base = geom.base_thickness
    heel = geom.heel_length

    weight_rows = []

    # Base slab
    W_base = gamma_c * B * t_base
    x_base = B / 2.0
    weight_rows.append(["Base slab", f"{W_base:.1f}", f"{x_base:.3f}",
                         f"{W_base * x_base:.1f}"])

    # Stem (rectangular part)
    W_stem_rect = gamma_c * t_stem_t * h_stem
    x_stem_rect = t_toe + t_stem_b / 2.0
    weight_rows.append(["Stem (rect)", f"{W_stem_rect:.1f}", f"{x_stem_rect:.3f}",
                         f"{W_stem_rect * x_stem_rect:.1f}"])

    # Stem taper
    if t_stem_b > t_stem_t:
        t_taper = t_stem_b - t_stem_t
        W_stem_tri = gamma_c * 0.5 * t_taper * h_stem
        x_stem_tri = t_toe + t_stem_t + t_taper / 3.0
        weight_rows.append(["Stem (taper)", f"{W_stem_tri:.1f}", f"{x_stem_tri:.3f}",
                             f"{W_stem_tri * x_stem_tri:.1f}"])

    # Soil on heel
    if heel > 0:
        W_soil = gamma * heel * h_stem
        x_soil = t_toe + t_stem_b + heel / 2.0
        weight_rows.append(["Soil on heel", f"{W_soil:.1f}", f"{x_soil:.3f}",
                             f"{W_soil * x_soil:.1f}"])

    # Surcharge on heel
    if heel > 0 and geom.surcharge > 0:
        W_q = geom.surcharge * heel
        x_q = t_toe + t_stem_b + heel / 2.0
        weight_rows.append(["Surcharge on heel", f"{W_q:.1f}", f"{x_q:.3f}",
                             f"{W_q * x_q:.1f}"])

    # Totals
    total_W = sum(float(r[1]) for r in weight_rows)
    total_M = sum(float(r[3]) for r in weight_rows)
    weight_rows.append(["TOTAL", f"{total_W:.1f}", "\u2014", f"{total_M:.1f}"])

    weight_items.append(TableData(
        title="Wall Component Weights and Moments about Toe",
        headers=["Component", "Weight (kN/m)", "Arm from Toe (m)",
                 "Moment (kN\u00b7m/m)"],
        rows=weight_rows,
        notes="Moments computed about toe for overturning check.",
    ))

    sections.append(CalcSection(title="Wall Weights & Stabilizing Moments", items=weight_items))

    # ── Section 3: Sliding Check ───────────────────────────────
    slide_items = []

    delta_b = 2.0 / 3.0 * phi_f
    ca = 2.0 / 3.0 * c_f

    slide_items.append(CalcStep(
        title="Base Friction Angle",
        equation="\u03b4_b = (2/3) \u00d7 \u03c6_foundation",
        substitution=f"\u03b4_b = (2/3) \u00d7 {phi_f:.1f}\u00b0",
        result_name="\u03b4_b",
        result_value=f"{delta_b:.2f}",
        result_unit="deg",
        reference="Das Ch. 13; typically 1/2 to 2/3 of \u03c6",
    ))

    R_sliding = total_W * math.tan(math.radians(delta_b)) + ca * B
    slide_items.append(CalcStep(
        title="Resisting Force (Sliding)",
        equation="R = V \u00d7 tan(\u03b4_b) + c_a \u00d7 B",
        substitution=(
            f"R = {total_W:.1f} \u00d7 tan({delta_b:.2f}\u00b0)"
            + (f" + {ca:.1f} \u00d7 {B:.2f}" if ca > 0 else "")
        ),
        result_name="R",
        result_value=f"{R_sliding:.1f}",
        result_unit="kN/m",
    ))

    slide_items.append(CalcStep(
        title="Factor of Safety Against Sliding",
        equation="FOS_sliding = R / P_a",
        substitution=f"FOS_sliding = {R_sliding:.1f} / {Pa_total:.1f}",
        result_name="FOS_sliding",
        result_value=f"{result.FOS_sliding:.3f}",
        reference="AASHTO 11.6.3 (min 1.5)",
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

    M_stab = total_M
    # Location of resultant of Pa
    z_Pa = H / 3.0  # simplified; with surcharge the actual code accounts for it
    if geom.surcharge > 0 and Pa_total > 0:
        z_earth = H / 3.0
        z_surcharge = H / 2.0
        z_Pa = (Pa_earth * z_earth + Pa_surcharge * z_surcharge) / Pa_total
        z_Pa = max(z_Pa, 0.0)
    M_over = Pa_total * z_Pa

    ot_items.append(CalcStep(
        title="Overturning Moment about Toe",
        equation="M_ot = P_a \u00d7 z_Pa",
        substitution=f"M_ot = {Pa_total:.1f} \u00d7 {z_Pa:.3f}",
        result_name="M_ot",
        result_value=f"{M_over:.1f}",
        result_unit="kN\u00b7m/m",
    ))

    ot_items.append(CalcStep(
        title="Stabilizing Moment about Toe",
        equation="M_stab = \u03a3(W_i \u00d7 x_i)",
        substitution="(see weight table above)",
        result_name="M_stab",
        result_value=f"{M_stab:.1f}",
        result_unit="kN\u00b7m/m",
    ))

    ot_items.append(CalcStep(
        title="Factor of Safety Against Overturning",
        equation="FOS_ot = M_stab / M_ot",
        substitution=f"FOS_ot = {M_stab:.1f} / {M_over:.1f}",
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

    # Eccentricity
    x_R = (M_stab - M_over) / total_W if total_W > 0 else B / 2.0
    e = B / 2.0 - x_R

    brg_items.append(CalcStep(
        title="Resultant Location from Toe",
        equation="x_R = (M_stab - M_ot) / V",
        substitution=f"x_R = ({M_stab:.1f} - {M_over:.1f}) / {total_W:.1f}",
        result_name="x_R",
        result_value=f"{x_R:.3f}",
        result_unit="m",
    ))

    brg_items.append(CalcStep(
        title="Eccentricity of Resultant",
        equation="e = B/2 - x_R",
        substitution=f"e = {B:.2f}/2 - {x_R:.3f}",
        result_name="e",
        result_value=f"{result.eccentricity:.3f}",
        result_unit="m",
        notes=f"B/6 = {B / 6:.3f} m. "
              f"{'Resultant within middle third' if result.in_middle_third else 'OUTSIDE middle third'}.",
    ))

    # Bearing pressures
    if result.in_middle_third:
        brg_items.append(CalcStep(
            title="Toe Bearing Pressure (Trapezoidal Distribution)",
            equation="q_toe = (V/B)(1 + 6e/B)",
            substitution=f"q_toe = ({total_W:.1f}/{B:.2f})(1 + 6\u00d7{abs(result.eccentricity):.3f}/{B:.2f})",
            result_name="q_toe",
            result_value=f"{result.q_toe:.1f}",
            result_unit="kPa",
        ))
        brg_items.append(CalcStep(
            title="Heel Bearing Pressure",
            equation="q_heel = (V/B)(1 - 6e/B)",
            substitution=f"q_heel = ({total_W:.1f}/{B:.2f})(1 - 6\u00d7{abs(result.eccentricity):.3f}/{B:.2f})",
            result_name="q_heel",
            result_value=f"{result.q_heel:.1f}",
            result_unit="kPa",
        ))
    else:
        B_eff = 3.0 * x_R if x_R > 0 else B
        brg_items.append(CalcStep(
            title="Toe Bearing Pressure (Triangular Distribution)",
            equation="q_toe = 2V / (3 \u00d7 x_R)  [resultant outside middle third]",
            substitution=f"q_toe = 2 \u00d7 {total_W:.1f} / (3 \u00d7 {x_R:.3f})",
            result_name="q_toe",
            result_value=f"{result.q_toe:.1f}",
            result_unit="kPa",
            notes="q_heel = 0 kPa (resultant outside middle third).",
        ))

    if q_all is not None:
        brg_items.append(CalcStep(
            title="Factor of Safety Against Bearing Failure",
            equation="FOS_bearing = q_allowable / q_toe",
            substitution=f"FOS_bearing = {q_all:.1f} / {result.q_toe:.1f}",
            result_name="FOS_bearing",
            result_value=f"{result.FOS_bearing:.3f}",
            reference="AASHTO 11.6.3",
        ))
        brg_items.append(CheckItem(
            description="Bearing capacity adequacy",
            demand=result.q_toe,
            demand_label="q_toe",
            capacity=q_all,
            capacity_label="q_allowable",
            unit="kPa",
            passes=result.passes_bearing,
        ))

    sections.append(CalcSection(
        title="Bearing Pressure & Eccentricity", items=brg_items
    ))

    # ── Section 6: Summary ─────────────────────────────────────
    summary_items = []

    summary_table = TableData(
        title="Stability Check Summary",
        headers=["Check", "FOS Computed", "FOS Required", "Status"],
        rows=[
            ["Sliding", f"{result.FOS_sliding:.3f}", f"{FOS_s_req:.1f}",
             "OK" if result.passes_sliding else "FAIL"],
            ["Overturning", f"{result.FOS_overturning:.3f}", f"{FOS_o_req:.1f}",
             "OK" if result.passes_overturning else "FAIL"],
            ["Bearing", f"{result.FOS_bearing:.3f}",
             "1.0" if q_all else "N/A",
             "OK" if result.passes_bearing else "FAIL"],
        ],
    )
    summary_items.append(summary_table)

    eccentricity_status = "Within middle third" if result.in_middle_third else "OUTSIDE middle third"
    summary_items.append(
        f"Eccentricity: e = {result.eccentricity:.3f} m "
        f"(B/6 = {B / 6:.3f} m) \u2014 {eccentricity_status}."
    )

    sections.append(CalcSection(title="Stability Summary", items=summary_items))

    return sections


def _calc_steps_mse(result, analysis) -> List[CalcSection]:
    """Calculation steps for MSE wall."""
    geom = _get(analysis, "geom")
    gamma = _get(analysis, "gamma_backfill", 18.0)
    phi = _get(analysis, "phi_backfill", 34.0)
    reinforcement = _get(analysis, "reinforcement")
    phi_f = _get(analysis, "phi_foundation", phi)
    c_f = _get(analysis, "c_foundation", 0.0)
    q_all = _get(analysis, "q_allowable")

    if phi_f is None:
        phi_f = phi

    sections = []
    H = geom.wall_height
    L = geom.reinforcement_length

    # ── Section 1: Earth Pressure ──────────────────────────────
    ep_items = []

    Ka = math.tan(math.radians(45 - phi / 2)) ** 2
    ep_items.append(CalcStep(
        title="Rankine Active Earth Pressure Coefficient",
        equation="K_a = tan\u00b2(45\u00b0 - \u03c6/2)",
        substitution=f"K_a = tan\u00b2(45\u00b0 - {phi:.1f}\u00b0/2)",
        result_name="K_a",
        result_value=f"{Ka:.4f}",
        reference="Rankine (1857); GEC-11 uses Rankine for MSE walls",
    ))

    # Active force
    Pa = 0.5 * Ka * gamma * H ** 2
    Pa_q = Ka * geom.surcharge * H if geom.surcharge > 0 else 0
    Pa_total = Pa + Pa_q

    ep_items.append(CalcStep(
        title="Total Active Force on Reinforced Zone",
        equation="P_a = 0.5 \u00d7 K_a \u00d7 \u03b3 \u00d7 H\u00b2"
                 + (" + K_a \u00d7 q \u00d7 H" if geom.surcharge > 0 else ""),
        substitution=(
            f"P_a = 0.5 \u00d7 {Ka:.4f} \u00d7 {gamma:.1f} \u00d7 {H:.2f}\u00b2"
            + (f" + {Ka:.4f} \u00d7 {geom.surcharge:.1f} \u00d7 {H:.2f}"
               if geom.surcharge > 0 else "")
        ),
        result_name="P_a",
        result_value=f"{Pa_total:.1f}",
        result_unit="kN/m",
        reference="GEC-11, Ch. 4",
    ))

    sections.append(CalcSection(title="Earth Pressure Computation", items=ep_items))

    # ── Section 2: External Sliding ────────────────────────────
    ext_items = []

    W = gamma * H * L + geom.surcharge * L
    ext_items.append(CalcStep(
        title="Weight of Reinforced Soil Block",
        equation="W = \u03b3 \u00d7 H \u00d7 L + q \u00d7 L",
        substitution=(
            f"W = {gamma:.1f} \u00d7 {H:.2f} \u00d7 {L:.2f}"
            + (f" + {geom.surcharge:.1f} \u00d7 {L:.2f}" if geom.surcharge > 0 else "")
        ),
        result_name="W",
        result_value=f"{W:.1f}",
        result_unit="kN/m",
    ))

    delta_b = 2.0 / 3.0 * phi_f
    ca = 2.0 / 3.0 * c_f
    R_slide = W * math.tan(math.radians(delta_b)) + ca * L

    ext_items.append(CalcStep(
        title="Sliding Resistance",
        equation="R = W \u00d7 tan(\u03b4_b) + c_a \u00d7 L,  \u03b4_b = 2/3 \u00d7 \u03c6_f",
        substitution=(
            f"R = {W:.1f} \u00d7 tan({delta_b:.2f}\u00b0)"
            + (f" + {ca:.1f} \u00d7 {L:.2f}" if ca > 0 else "")
        ),
        result_name="R",
        result_value=f"{R_slide:.1f}",
        result_unit="kN/m",
    ))

    ext_items.append(CalcStep(
        title="Factor of Safety Against Sliding",
        equation="FOS_sliding = R / P_a",
        substitution=f"FOS_sliding = {R_slide:.1f} / {Pa_total:.1f}",
        result_name="FOS_sliding",
        result_value=f"{result.FOS_sliding:.3f}",
        reference="GEC-11 (min 1.5)",
    ))

    # Overturning
    M_stab = W * L / 2.0
    z_Pa = H / 3.0
    if geom.surcharge > 0 and Pa_total > 0:
        z_Pa = (Pa * H / 3.0 + Pa_q * H / 2.0) / Pa_total
    M_over = Pa_total * z_Pa

    ext_items.append(CalcStep(
        title="Overturning Moment about Toe",
        equation="M_ot = P_a \u00d7 z_Pa",
        substitution=f"M_ot = {Pa_total:.1f} \u00d7 {z_Pa:.3f}",
        result_name="M_ot",
        result_value=f"{M_over:.1f}",
        result_unit="kN\u00b7m/m",
    ))

    ext_items.append(CalcStep(
        title="Stabilizing Moment",
        equation="M_stab = W \u00d7 L/2",
        substitution=f"M_stab = {W:.1f} \u00d7 {L:.2f}/2",
        result_name="M_stab",
        result_value=f"{M_stab:.1f}",
        result_unit="kN\u00b7m/m",
    ))

    ext_items.append(CalcStep(
        title="Factor of Safety Against Overturning",
        equation="FOS_ot = M_stab / M_ot",
        substitution=f"FOS_ot = {M_stab:.1f} / {M_over:.1f}",
        result_name="FOS_ot",
        result_value=f"{result.FOS_overturning:.3f}",
        reference="GEC-11 (min 2.0)",
    ))

    # Bearing
    x_R = (M_stab - M_over) / W if W > 0 else L / 2.0
    e_ext = L / 2.0 - x_R
    if abs(e_ext) <= L / 6.0:
        q_toe = W / L * (1.0 + 6.0 * e_ext / L)
    else:
        B_eff = 3.0 * x_R
        q_toe = 2.0 * W / B_eff if B_eff > 0 else 0.0

    ext_items.append(CalcStep(
        title="Eccentricity & Toe Bearing Pressure",
        equation="e = L/2 - (M_stab - M_ot)/W;  q_toe = (V/L)(1 + 6e/L)",
        substitution=f"e = {L:.2f}/2 - ({M_stab:.1f} - {M_over:.1f})/{W:.1f}",
        result_name="q_toe",
        result_value=f"{q_toe:.1f}",
        result_unit="kPa",
    ))

    if q_all is not None:
        ext_items.append(CalcStep(
            title="Factor of Safety Against Bearing",
            equation="FOS_bearing = q_allowable / q_toe",
            substitution=f"FOS_bearing = {q_all:.1f} / {q_toe:.1f}",
            result_name="FOS_bearing",
            result_value=f"{result.FOS_bearing:.3f}",
        ))

    # External stability summary
    ext_items.append(CheckItem(
        description="External stability (sliding, overturning, bearing)",
        demand=1.0,
        demand_label="Required",
        capacity=min(result.FOS_sliding, result.FOS_overturning,
                     result.FOS_bearing if q_all else 99.9),
        capacity_label="Min FOS",
        passes=result.passes_external,
    ))

    ext_summary = TableData(
        title="External Stability Summary",
        headers=["Check", "FOS", "Required", "Status"],
        rows=[
            ["Sliding", f"{result.FOS_sliding:.3f}", "1.5",
             "OK" if result.FOS_sliding >= 1.5 else "FAIL"],
            ["Overturning", f"{result.FOS_overturning:.3f}", "2.0",
             "OK" if result.FOS_overturning >= 2.0 else "FAIL"],
            ["Bearing", f"{result.FOS_bearing:.3f}",
             "1.0" if q_all else "N/A",
             "OK" if (result.FOS_bearing >= 1.0 if q_all else True) else "FAIL"],
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
            "Metallic: Kr/Ka = 1.7 at z=0 to 1.2 at z\u22656m"
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
            f"L_e = L - L_a (L_a from 45+\u03c6/2 failure plane); C = 2"
        ),
        result_name="P_r",
        result_value="(per level)",
        result_unit="kN/m",
        reference="GEC-11, Eq. 4-18",
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

    overall_pass = result.passes_external and result.all_pass_internal
    summary_items.append(TableData(
        title="Overall MSE Wall Assessment",
        headers=["Category", "Status"],
        rows=[
            ["External Stability", "OK" if result.passes_external else "FAIL"],
            ["Internal Stability", "OK" if result.all_pass_internal else "FAIL"],
            ["Overall", "OK" if overall_pass else "FAIL"],
        ],
    ))

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

def _Ka_value(phi, method, geom):
    """Compute Ka using the appropriate method."""
    if method == "coulomb":
        from retaining_walls.earth_pressure import coulomb_Ka
        delta = 2.0 / 3.0 * phi
        return coulomb_Ka(phi, delta, beta_deg=geom.backfill_slope)
    else:
        return math.tan(math.radians(45 - phi / 2)) ** 2


def _plot_cantilever_section(result, analysis):
    """Create a schematic cross-section of a cantilever retaining wall."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    geom = _get(analysis, "geom")
    gamma = _get(analysis, "gamma_backfill", 18.0)
    phi = _get(analysis, "phi_backfill", 30.0)
    method = _get(analysis, "pressure_method", "rankine")

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

    # Backfill behind stem (to the right of stem, above base)
    backfill_x = [
        t_toe + t_stem_b, B, B, t_toe + t_stem_b,
    ]
    backfill_y = [
        t_base, t_base, t_base + h_stem, t_base + h_stem,
    ]
    ax.fill(backfill_x, backfill_y, facecolor='#f5e6c8', edgecolor='none', alpha=0.7)

    # Soil below base
    ax.add_patch(patches.Rectangle(
        (-B * 0.3, -B * 0.3), B * 1.6, B * 0.3,
        facecolor='#e8dcc8', edgecolor='none', alpha=0.5,
    ))

    # Earth pressure diagram (triangular, on back of wall)
    Ka_val = _Ka_value(phi, method, geom)
    sigma_base = Ka_val * gamma * geom.H_active  # max pressure at base

    # Scale the pressure arrow to fit in the figure
    p_scale = B * 0.5 / max(sigma_base, 1)

    # Draw triangular pressure diagram behind the stem
    press_x_base = t_toe + t_stem_b
    ax.fill(
        [press_x_base, press_x_base + sigma_base * p_scale,
         press_x_base],
        [t_base, t_base, H],
        facecolor='#fca5a5', edgecolor='#dc2626', linewidth=1.5, alpha=0.5,
    )
    ax.annotate(
        f'P_a = {_compute_Pa(gamma, geom.H_active, Ka_val, geom.surcharge):.0f} kN/m',
        xy=(press_x_base + sigma_base * p_scale * 0.5, t_base + geom.H_active / 3.0),
        fontsize=8, ha='left', color='#dc2626', fontweight='bold',
    )

    # Dimension annotations
    # Wall height
    ax.annotate('', xy=(-B * 0.15, 0), xytext=(-B * 0.15, H),
                arrowprops=dict(arrowstyle='<->', color='#1a1a1a', lw=1.2))
    ax.text(-B * 0.22, H / 2, f'H = {H:.2f} m', ha='right', va='center',
            fontsize=9, fontweight='bold', rotation=90)

    # Base width
    ax.annotate('', xy=(0, -B * 0.08), xytext=(B, -B * 0.08),
                arrowprops=dict(arrowstyle='<->', color='#1a1a1a', lw=1.2))
    ax.text(B / 2, -B * 0.14, f'B = {B:.2f} m', ha='center', fontsize=9, fontweight='bold')

    # Toe dimension
    if t_toe > 0:
        ax.annotate('', xy=(0, -B * 0.02), xytext=(t_toe, -B * 0.02),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8))
        ax.text(t_toe / 2, -B * 0.05, f'toe={t_toe:.2f}', ha='center', fontsize=7, color='#666')

    # Heel dimension
    if heel > 0:
        heel_start = t_toe + t_stem_b
        ax.annotate('', xy=(heel_start, -B * 0.02), xytext=(B, -B * 0.02),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8))
        ax.text((heel_start + B) / 2, -B * 0.05, f'heel={heel:.2f}',
                ha='center', fontsize=7, color='#666')

    # Ground surface line
    ax.plot([-B * 0.3, t_toe], [H, H], 'g-', linewidth=2)
    ax.plot([t_toe, t_toe], [H, t_base + h_stem], 'g-', linewidth=1)

    # Resultant location marker
    x_R_scaled = (result.base_width / 2.0 - result.eccentricity)
    ax.plot(x_R_scaled, 0, 'rv', markersize=10)
    ax.text(x_R_scaled, -B * 0.2, f'R\n(e={result.eccentricity:.3f}m)',
            ha='center', fontsize=7, color='#dc2626')

    # Soil property label
    ax.text(B * 1.1, H * 0.5,
            f'\u03b3 = {gamma:.1f} kN/m\u00b3\n\u03c6 = {phi:.1f}\u00b0\n'
            f'Ka = {Ka_val:.3f}',
            fontsize=9, va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffde7', alpha=0.9))

    # FOS annotation box
    status_items = [
        f"FOS_slide = {result.FOS_sliding:.2f}",
        f"FOS_OT    = {result.FOS_overturning:.2f}",
        f"FOS_brg   = {result.FOS_bearing:.2f}",
    ]
    ax.text(0.02, 0.98, '\n'.join(status_items),
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='#333', alpha=0.9))

    ax.set_xlim(-B * 0.4, B * 1.4)
    ax.set_ylim(-B * 0.35, H * 1.15)
    ax.set_xlabel('Distance (m)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)
    ax.set_title('Cantilever Retaining Wall Cross-Section', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
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

    # x along base from toe (0) to heel (B)
    x = [0, B]
    q = [q_toe, q_heel]

    # Fill the pressure distribution
    ax.fill_between(x, 0, q, color='#fca5a5', alpha=0.5, edgecolor='#dc2626', linewidth=1.5)
    ax.plot(x, q, 'r-', linewidth=2)

    # Annotations
    ax.annotate(f'q_toe = {q_toe:.1f} kPa', xy=(0, q_toe),
                xytext=(B * 0.15, q_toe * 1.1 + 5), fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#dc2626'))
    if q_heel > 0:
        ax.annotate(f'q_heel = {q_heel:.1f} kPa', xy=(B, q_heel),
                    xytext=(B * 0.7, q_heel * 1.1 + 5), fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2563eb'))

    # Base line
    ax.axhline(y=0, color='k', linewidth=1)

    # Middle-third zone
    b6 = B / 6.0
    ax.axvspan(B / 2 - b6, B / 2 + b6, alpha=0.1, color='#16a34a',
               label=f'Middle third (B/6 = {b6:.2f} m)')

    # Eccentricity marker
    e = result.eccentricity
    x_R = B / 2.0 - e
    ax.axvline(x=x_R, color='#dc2626', linestyle='--', linewidth=1.2,
               label=f'Resultant (e = {e:.3f} m)')
    ax.axvline(x=B / 2.0, color='#666', linestyle=':', linewidth=0.8,
               label='Base center')

    q_all = _get(analysis, "q_allowable")
    if q_all is not None:
        ax.axhline(y=q_all, color='#16a34a', linestyle='--', linewidth=1.2,
                   label=f'q_allowable = {q_all:.0f} kPa')

    ax.set_xlabel('Distance from Toe (m)', fontsize=10)
    ax.set_ylabel('Bearing Pressure (kPa)', fontsize=10)
    ax.set_title('Base Bearing Pressure Distribution', fontsize=11, fontweight='bold')
    ax.set_xlim(-B * 0.05, B * 1.05)
    ax.set_ylim(0, max(q_toe, q_heel, q_all or 0) * 1.3 + 5)
    ax.legend(loc='upper center', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
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

    # Failure surface (Rankine: 45 + phi/2 from horizontal)
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
    ax.text(L / 2, -H * 0.1, f'L = {L:.2f} m', ha='center',
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

    # Labels
    ax.text(L / 2, H / 2, 'Reinforced\nSoil Zone', ha='center', va='center',
            fontsize=10, fontweight='bold', alpha=0.6)
    ax.text(L * 1.2, H / 2, 'Retained\nSoil', ha='center', va='center',
            fontsize=9, alpha=0.5)

    # Reinforcement legend
    r_name = reinforcement.name if reinforcement else "Reinforcement"
    ax.plot([], [], color=r_color, linewidth=2, label=r_name)

    # FOS box
    status = "OK" if result.passes_external and result.all_pass_internal else "CHECK"
    status_color = '#16a34a' if status == "OK" else '#dc2626'
    fos_text = (
        f"External: slide={result.FOS_sliding:.2f}, OT={result.FOS_overturning:.2f}\n"
        f"Internal: {'all pass' if result.all_pass_internal else 'FAIL'}"
    )
    ax.text(0.02, 0.98, fos_text,
            transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor=status_color, alpha=0.9))

    ax.set_xlim(-L * 0.25, L * 1.5)
    ax.set_ylim(-H * 0.15, H * 1.1)
    ax.set_xlabel('Distance (m)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)
    ax.set_title('MSE Wall Cross-Section', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
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
    ax.set_ylabel('Depth z (m)', fontsize=10)
    ax.set_title('Internal Stability: T_max vs Pullout Resistance', fontsize=11,
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _compute_Pa(gamma, H, Ka, surcharge=0):
    """Quick total active force computation for annotation."""
    Pa = 0.5 * Ka * gamma * H ** 2 + Ka * surcharge * H
    return max(Pa, 0.0)
