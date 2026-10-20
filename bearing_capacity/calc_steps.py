"""
Calculation package steps for bearing capacity analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation.

References:
    Vesic (1973) — Bearing capacity factors and Ngamma
    Meyerhof (1963) — Shape, depth, inclination factors
    FHWA GEC-6 (FHWA-IF-02-054), Chapter 6
"""

import math
from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Shallow Foundation Bearing Capacity Analysis"

REFERENCES = [
    'Vesic, A.S. (1973). "Analysis of Ultimate Loads of Shallow Foundations." '
    'JSMFD, ASCE, Vol. 99, No. SM1, pp. 45-73.',
    'Meyerhof, G.G. (1963). "Some Recent Research on the Bearing Capacity '
    'of Foundations." Canadian Geotechnical Journal, Vol. 1, No. 1, pp. 16-26.',
    "FHWA GEC-6 (FHWA-IF-02-054): Shallow Foundations, Chapter 6.",
    'Prandtl, L. (1921). "Uber die Eindringungsfestigkeit plastischer '
    'Baustoffe." Zeitschrift fur Angewandte Mathematik und Mechanik, 1(1).',
    'Hansen, J.B. (1970). "A Revised and Extended Formula for Bearing '
    'Capacity." Danish Geotechnical Institute Bulletin 28.',
    "Terzaghi, K. (1943). Theoretical Soil Mechanics. Wiley, New York.",
    'Kumbhojkar, A.S. (1993). "Numerical Evaluation of Terzaghi\'s Nγ." '
    "J. Geotech. Eng., ASCE, 119(3), 598-607 (values as tabulated in Das, "
    "Principles of Foundation Engineering).",
    "UFC 3-220-20 (2025). Foundations. Ch. 5, Tables 5-2 to 5-5 (Terzaghi "
    "factors with the Coduto et al. 2016 Nγ approximation, shape "
    "factors, local shear).",
]

_TERZAGHI_NG_LABEL = {
    "kumbhojkar": "Kumbhojkar (1993), as tabulated in Das",
    "coduto": "Coduto et al. (2016); UFC 3-220-20 Table 5-2",
}


def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for bearing capacity calc package.

    Parameters
    ----------
    result : BearingCapacityResult
        Computed results.
    analysis : BearingCapacityAnalysis
        Analysis object holding all inputs.

    Returns
    -------
    list of InputItem
    """
    f = analysis.footing
    s = analysis.soil
    layer = s.layer1

    items = [
        InputItem("B", "Footing width", f.width, "m"),
    ]
    if f.shape == "rectangular":
        items.append(InputItem("L", "Footing length", f.length, "m"))
    items.extend([
        InputItem("Shape", "Footing shape", f.shape, ""),
        InputItem("D_f", "Embedment depth", f.depth, "m"),
    ])

    if f.eccentricity_B != 0:
        items.append(InputItem("e_B", "Eccentricity (B direction)", f.eccentricity_B, "m"))
    if f.eccentricity_L != 0:
        items.append(InputItem("e_L", "Eccentricity (L direction)", f.eccentricity_L, "m"))

    items.extend([
        InputItem("c", "Cohesion", layer.cohesion, "kPa"),
        InputItem("\u03c6", "Friction angle", layer.friction_angle, "deg"),
        InputItem("\u03b3", "Unit weight", layer.unit_weight, "kN/m\u00b3"),
    ])

    if s.gwt_depth is not None:
        items.append(InputItem("GWT", "Groundwater depth", s.gwt_depth, "m"))

    items.append(InputItem("FS", "Factor of safety", analysis.factor_of_safety, ""))
    if analysis.factor_method == "terzaghi":
        items.extend([
            InputItem("Method", "Bearing capacity method",
                      f"Terzaghi (1943), {analysis.failure_mode} shear", ""),
            InputItem("N\u03b3", "N\u03b3 source",
                      _TERZAGHI_NG_LABEL[analysis.ngamma_method], ""),
        ])
    else:
        items.append(InputItem("Method", "N\u03b3 method",
                               analysis.ngamma_method.capitalize(), ""))

    if analysis.load_inclination != 0:
        items.append(InputItem("\u03b2", "Load inclination", analysis.load_inclination, "deg"))
    if f.base_tilt != 0:
        items.append(InputItem("\u03b1", "Base tilt", f.base_tilt, "deg"))
    if analysis.ground_slope != 0:
        items.append(InputItem("\u03c9", "Ground slope", analysis.ground_slope, "deg"))

    return items


def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections.

    Parameters
    ----------
    result : BearingCapacityResult
        Computed results.
    analysis : BearingCapacityAnalysis
        Analysis object holding all inputs.

    Returns
    -------
    list of CalcSection
    """
    r = result
    f = analysis.footing
    phi_deg = analysis.soil.layer1.friction_angle
    sections = []

    # ── Section: Effective Dimensions ───────────────────────────
    eff_steps = []
    if f.eccentricity_B != 0 or f.eccentricity_L != 0:
        eff_steps.append(CalcStep(
            title="Effective Footing Width",
            equation="B' = B - 2 \u00d7 |e_B|",
            substitution=f"B' = {f.width:.3f} - 2 \u00d7 {abs(f.eccentricity_B):.3f}",
            result_name="B'",
            result_value=f"{r.B_eff:.3f}",
            result_unit="m",
            reference="Meyerhof (1953) effective area method",
        ))
        eff_steps.append(CalcStep(
            title="Effective Footing Length",
            equation="L' = L - 2 \u00d7 |e_L|",
            substitution=f"L' = {f.length:.3f} - 2 \u00d7 {abs(f.eccentricity_L):.3f}",
            result_name="L'",
            result_value=f"{r.L_eff:.3f}",
            result_unit="m",
        ))
    else:
        eff_steps.append(CalcStep(
            title="Effective Footing Dimensions (no eccentricity)",
            equation="B' = B, L' = L (no load eccentricity)",
            substitution="",
            result_name="B', L'",
            result_value=f"{r.B_eff:.3f} m, {r.L_eff:.3f} m",
        ))

    # Overburden
    eff_steps.append(CalcStep(
        title="Overburden Pressure at Footing Base",
        equation="q = \u03b3 \u00d7 D_f",
        substitution=f"q = {analysis.soil.layer1.unit_weight:.2f} \u00d7 {f.depth:.3f}",
        result_name="q",
        result_value=f"{r.q_overburden:.2f}",
        result_unit="kPa",
        notes="Adjusted for GWT if applicable" if analysis.soil.gwt_depth is not None else "",
    ))

    # Effective unit weight
    eff_steps.append(CalcStep(
        title="Effective Unit Weight Below Footing",
        equation="\u03b3' = \u03b3 - \u03b3_w (if below GWT)" if (
            analysis.soil.gwt_depth is not None and analysis.soil.gwt_depth <= f.depth
        ) else "\u03b3' = \u03b3 (GWT below footing)",
        substitution="",
        result_name="\u03b3'",
        result_value=f"{r.gamma_eff:.2f}",
        result_unit="kN/m\u00b3",
    ))

    sections.append(CalcSection(title="Effective Dimensions & Overburden", items=eff_steps))

    if analysis.factor_method == "terzaghi":
        sections.extend(_terzaghi_sections(r, analysis))
        return sections

    # ── Section: Bearing Capacity Factors ──────────────────────
    bc_steps = []

    # Nq
    nq_eq = "N_q = exp(\u03c0 \u00d7 tan(\u03c6)) \u00d7 tan\u00b2(45 + \u03c6/2)"
    nq_sub = (
        f"N_q = exp(\u03c0 \u00d7 tan({phi_deg:.1f}\u00b0)) "
        f"\u00d7 tan\u00b2(45 + {phi_deg/2:.1f}\u00b0)"
    )
    bc_steps.append(CalcStep(
        title="Bearing Capacity Factor N_q",
        equation=nq_eq,
        substitution=nq_sub,
        result_name="N_q",
        result_value=f"{r.Nq:.2f}",
        reference="Reissner (1924); FHWA GEC-6 Eq. 5-2",
    ))

    # Nc
    if phi_deg == 0:
        nc_eq = "N_c = 5.14 (Prandtl exact solution for \u03c6 = 0)"
        nc_sub = ""
    else:
        nc_eq = "N_c = (N_q - 1) / tan(\u03c6)"
        nc_sub = f"N_c = ({r.Nq:.2f} - 1) / tan({phi_deg:.1f}\u00b0)"
    bc_steps.append(CalcStep(
        title="Bearing Capacity Factor N_c",
        equation=nc_eq,
        substitution=nc_sub,
        result_name="N_c",
        result_value=f"{r.Nc:.2f}",
        reference="Prandtl (1921); FHWA GEC-6 Eqs. 5-3, 5-4",
    ))

    # Ngamma
    method = analysis.ngamma_method.lower()
    if method == "vesic":
        ng_eq = "N_\u03b3 = 2 \u00d7 (N_q + 1) \u00d7 tan(\u03c6)"
        ng_sub = f"N_\u03b3 = 2 \u00d7 ({r.Nq:.2f} + 1) \u00d7 tan({phi_deg:.1f}\u00b0)"
        ng_ref = "Vesic (1973); FHWA GEC-6 Eq. 5-8, Table 5-1"
    elif method == "meyerhof":
        ng_eq = "N_\u03b3 = (N_q - 1) \u00d7 tan(1.4\u03c6)"
        ng_sub = f"N_\u03b3 = ({r.Nq:.2f} - 1) \u00d7 tan(1.4 \u00d7 {phi_deg:.1f}\u00b0)"
        ng_ref = "Meyerhof (1963); UFC 3-220-20 Table 5-6"
    else:
        ng_eq = "N_\u03b3 = 1.5 \u00d7 (N_q - 1) \u00d7 tan(\u03c6)"
        ng_sub = f"N_\u03b3 = 1.5 \u00d7 ({r.Nq:.2f} - 1) \u00d7 tan({phi_deg:.1f}\u00b0)"
        ng_ref = "Hansen (1970)"
    bc_steps.append(CalcStep(
        title="Bearing Capacity Factor N_\u03b3",
        equation=ng_eq,
        substitution=ng_sub,
        result_name="N_\u03b3",
        result_value=f"{r.Ngamma:.2f}",
        reference=ng_ref,
    ))

    sections.append(CalcSection(title="Bearing Capacity Factors", items=bc_steps))

    # ── Section: Correction Factors ────────────────────────────
    factor_method = analysis.factor_method.capitalize()
    meyerhof = analysis.factor_method == "meyerhof"
    cf_items = []

    # Shape factors; Meyerhof's are 1 under an inclined load (GM-188)
    not_combined = meyerhof and analysis.load_inclination != 0
    cf_items.append(CalcStep(
        title="Shape Factors",
        equation=_shape_factor_equation(analysis.factor_method, phi_deg),
        substitution=("s_c = s_q = s_\u03b3 = 1: shape and inclined load "
                      "not combined (UFC 3-220-20 Table 5-6 note A)"
                      if not_combined else ""),
        result_name="s_c, s_q, s_\u03b3",
        result_value=f"{r.sc:.4f}, {r.sq:.4f}, {r.sgamma:.4f}",
        reference=("Meyerhof; UFC 3-220-20 Table 5-6" if meyerhof
                   else f"{factor_method}; FHWA GEC-6 Table 5-2"),
    ))

    # Depth factors
    cf_items.append(CalcStep(
        title="Depth Factors",
        equation=_depth_factor_equation(analysis.factor_method, phi_deg,
                                        analysis.depth_factor_method),
        substitution="",
        result_name="d_c, d_q, d_\u03b3",
        result_value=f"{r.dc:.4f}, {r.dq:.4f}, {r.dgamma:.4f}",
        reference=_depth_factor_reference(analysis),
    ))

    # Only show inclination, base, and ground factors if non-unity
    if analysis.load_inclination != 0:
        cf_items.append(CalcStep(
            title="Inclination Factors",
            equation=_inclination_factor_equation(analysis, phi_deg),
            substitution=f"\u03b2 = {analysis.load_inclination:.1f}\u00b0",
            result_name="i_c, i_q, i_\u03b3",
            result_value=f"{r.ic:.4f}, {r.iq:.4f}, {r.igamma:.4f}",
            reference=("Meyerhof; UFC 3-220-20 Table 5-6" if meyerhof
                       else "Vesic (1975)" if phi_deg == 0
                       else "Vesic (1975); EN 1997-1:2004 Annex D.4"),
        ))

    if f.base_tilt != 0:
        cf_items.append(CalcStep(
            title="Base Inclination Factors",
            equation="(tilted footing base)",
            substitution=f"\u03b1 = {f.base_tilt:.1f}\u00b0",
            result_name="b_c, b_q, b_\u03b3",
            result_value=f"{r.bc:.4f}, {r.bq:.4f}, {r.bgamma:.4f}",
            reference="Hansen (1970); FHWA GEC-6 Table 5-5",
        ))

    if analysis.ground_slope != 0:
        cf_items.append(CalcStep(
            title="Ground Inclination Factors",
            equation="(sloping ground surface)",
            substitution=f"\u03c9 = {analysis.ground_slope:.1f}\u00b0",
            result_name="g_c, g_q, g_\u03b3",
            result_value=f"{r.gc:.4f}, {r.gq:.4f}, {r.ggamma:.4f}",
            reference="Vesic (1975)",
        ))

    # Summary table of all factors
    factor_table = TableData(
        title="Correction Factor Summary",
        headers=["Factor Type", "c-term", "q-term", "\u03b3-term"],
        rows=[
            ["Shape (s)", f"{r.sc:.4f}", f"{r.sq:.4f}", f"{r.sgamma:.4f}"],
            ["Depth (d)", f"{r.dc:.4f}", f"{r.dq:.4f}", f"{r.dgamma:.4f}"],
            ["Inclination (i)", f"{r.ic:.4f}", f"{r.iq:.4f}", f"{r.igamma:.4f}"],
            ["Base tilt (b)", f"{r.bc:.4f}", f"{r.bq:.4f}", f"{r.bgamma:.4f}"],
            ["Ground (g)", f"{r.gc:.4f}", f"{r.gq:.4f}", f"{r.ggamma:.4f}"],
        ],
    )
    cf_items.append(factor_table)
    sections.append(CalcSection(title="Correction Factors", items=cf_items))

    # ── Section: General Bearing Capacity Equation ──────────────
    gbc_items = []

    # The general equation
    gbc_items.append(CalcStep(
        title="General Bearing Capacity Equation",
        equation=(
            "q_ult = c \u00d7 N_c \u00d7 s_c \u00d7 d_c \u00d7 i_c \u00d7 b_c \u00d7 g_c\n"
            "      + q \u00d7 N_q \u00d7 s_q \u00d7 d_q \u00d7 i_q \u00d7 b_q \u00d7 g_q\n"
            "      + 0.5 \u00d7 \u03b3' \u00d7 B' \u00d7 N_\u03b3 \u00d7 s_\u03b3 \u00d7 d_\u03b3 \u00d7 i_\u03b3 \u00d7 b_\u03b3 \u00d7 g_\u03b3"
        ),
        substitution="",
        result_name="q_ult",
        result_value=f"{r.q_ultimate:,.1f}",
        result_unit="kPa",
        reference="Vesic (1975); cf. FHWA GEC-6 Eq. 5-14",
    ))

    # Term breakdown
    total = r.q_ultimate
    pct_c = 100 * r.term_cohesion / total if total > 0 else 0
    pct_q = 100 * r.term_overburden / total if total > 0 else 0
    pct_g = 100 * r.term_selfweight / total if total > 0 else 0

    gbc_items.append(TableData(
        title="Term Breakdown",
        headers=["Term", "Value (kPa)", "Contribution (%)"],
        rows=[
            ["Cohesion", f"{r.term_cohesion:,.1f}", f"{pct_c:.0f}%"],
            ["Overburden", f"{r.term_overburden:,.1f}", f"{pct_q:.0f}%"],
            ["Self-weight", f"{r.term_selfweight:,.1f}", f"{pct_g:.0f}%"],
            ["Total q_ult", f"{r.q_ultimate:,.1f}", "100%"],
        ],
    ))

    # Allowable
    gbc_items.append(CalcStep(
        title="Allowable Bearing Capacity",
        equation="q_all = q_ult / FS",
        substitution=f"q_all = {r.q_ultimate:,.1f} / {r.factor_of_safety:.1f}",
        result_name="q_all",
        result_value=f"{r.q_allowable:,.1f}",
        result_unit="kPa",
    ))
    gbc_items.extend(_allowable_load_items(r))

    # Two-layer info
    if r.is_two_layer:
        gbc_items.append(
            "Two-layer analysis (load-spread method; NAVFAC DM-7.01 / Bowles):"
        )
        gbc_items.append(TableData(
            title="Two-Layer Capacities",
            headers=["Layer", "q_ult (kPa)"],
            rows=[
                ["Upper layer (if semi-infinite)", f"{r.q_upper_layer:,.1f}"],
                ["Lower layer (if semi-infinite)", f"{r.q_lower_layer:,.1f}"],
                ["Combined (load-spread)", f"{r.q_ultimate:,.1f}"],
            ],
        ))

    sections.append(CalcSection(
        title="General Bearing Capacity Equation", items=gbc_items
    ))

    return sections


def get_figures(result, analysis) -> List[FigureData]:
    """Generate figures for the bearing capacity calc package.

    Parameters
    ----------
    result : BearingCapacityResult
        Computed results.
    analysis : BearingCapacityAnalysis
        Analysis object.

    Returns
    -------
    list of FigureData
    """
    figures = []

    # Figure 1: Term breakdown bar chart
    try:
        fig = _plot_term_breakdown(result)
        b64 = figure_to_base64(fig, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        figures.append(FigureData(
            title="Bearing Capacity Term Breakdown",
            image_base64=b64,
            caption=(
                f"Figure 1: Contribution of cohesion, overburden, and self-weight "
                f"terms to ultimate bearing capacity "
                f"(q_ult = {result.q_ultimate:,.1f} kPa)."
            ),
            width_percent=70,
        ))
    except ImportError:
        pass  # matplotlib not available

    # Figure 2: Footing cross-section schematic
    try:
        fig = _plot_footing_section(result, analysis)
        b64 = figure_to_base64(fig, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        figures.append(FigureData(
            title="Footing Cross-Section",
            image_base64=b64,
            caption=(
                f"Figure 2: Footing geometry — "
                f"B = {analysis.footing.width:.2f} m, "
                f"D_f = {analysis.footing.depth:.2f} m."
            ),
            width_percent=70,
        ))
    except ImportError:
        pass

    return figures


# ── Helper functions ──────────────────────────────────────────────

def _terzaghi_sections(r, analysis) -> List[CalcSection]:
    """Factor, shape and equation sections for Terzaghi (1943)."""
    phi0 = analysis.soil.layer1.friction_angle
    c0 = analysis.soil.layer1.cohesion
    phi = r.phi_used if r.phi_used is not None else phi0
    c = r.c_used if r.c_used is not None else c0
    sections = []

    bc_steps = []
    if analysis.failure_mode == "local":
        bc_steps.append(CalcStep(
            title="Local Shear Strengths",
            equation="c* = (2/3) c,  φ* = atan((2/3) tan φ)",
            substitution=(f"c* = (2/3) × {c0:.2f},  "
                          f"φ* = atan((2/3) tan {phi0:.1f}°)"),
            result_name="c*, φ*",
            result_value=f"{c:.2f} kPa, {phi:.2f}°",
            reference="Terzaghi (1943); UFC 3-220-20 Table 5-4",
        ))
    bc_steps.append(CalcStep(
        title="Bearing Capacity Factor N_q",
        equation=("N_q = a² / (2 cos²(45 + φ/2)),  "
                  "a = exp((3π/4 - φ/2) tan φ)"),
        substitution=f"φ = {phi:.2f}°",
        result_name="N_q",
        result_value=f"{r.Nq:.2f}",
        reference="Terzaghi (1943); UFC 3-220-20 Table 5-2",
    ))
    bc_steps.append(CalcStep(
        title="Bearing Capacity Factor N_c",
        equation=("N_c = 1.5π + 1 = 5.7 (φ = 0)" if phi == 0
                  else "N_c = (N_q - 1) cot φ"),
        substitution="" if phi == 0 else
        f"N_c = ({r.Nq:.2f} - 1) / tan({phi:.2f}°)",
        result_name="N_c",
        result_value=f"{r.Nc:.2f}",
        reference="Terzaghi (1943); UFC 3-220-20 Table 5-2",
    ))
    if analysis.ngamma_method == "coduto":
        ng_eq = "N_γ = 2 (N_q + 1) tan φ / (1 + 0.4 sin 4φ)"
        ng_sub = f"φ = {phi:.2f}°, N_q = {r.Nq:.2f}"
    else:
        ng_eq = "N_γ from the tabulated values (log-linear between whole degrees)"
        ng_sub = f"φ = {phi:.2f}°"
    alt = "; ".join(f"{_TERZAGHI_NG_LABEL[k]}: {v:.2f}"
                    for k, v in r.Ngamma_alternatives.items())
    bc_steps.append(CalcStep(
        title="Bearing Capacity Factor N_γ",
        equation=ng_eq,
        substitution=ng_sub,
        result_name="N_γ",
        result_value=f"{r.Ngamma:.2f}",
        reference=_TERZAGHI_NG_LABEL[analysis.ngamma_method],
        notes=(f"Terzaghi gave Nγ only as a chart; other cited value: "
               f"{alt}" if alt else ""),
    ))
    sections.append(CalcSection(title="Bearing Capacity Factors (Terzaghi 1943)",
                                items=bc_steps))

    shape_items = [
        CalcStep(
            title="Shape Factors",
            equation=("strip: s_c = 1, s_γ = 1;  square: s_c = 1.3, "
                      "s_γ = 0.8;  circular: s_c = 1.3, s_γ = 0.6;  "
                      "s_q = 1"),
            substitution=f"shape = {analysis.footing.shape}",
            result_name="s_c, s_q, s_γ",
            result_value=f"{r.sc:.4f}, {r.sq:.4f}, {r.sgamma:.4f}",
            reference="Terzaghi (1943); UFC 3-220-20 Table 5-5",
        ),
        "Terzaghi's method has no depth, load-inclination, base-tilt or "
        "ground-slope factors (UFC 3-220-20 Sec 5-3.3.1; Table 5-3: "
        "D_f/B ≤ 1, no inclined loads, slopes or tilted bases).",
    ]
    sections.append(CalcSection(title="Shape Factors", items=shape_items))

    eq_items = [
        CalcStep(
            title="Terzaghi Bearing Capacity Equation",
            equation=("q_ult = s_c × c × N_c + q × N_q "
                      "+ 0.5 × s_γ × γ' × B' × N_γ"),
            substitution=(
                f"q_ult = {r.sc:.3f} × {c:.2f} × {r.Nc:.2f} + "
                f"{r.q_overburden:.2f} × {r.Nq:.2f} + 0.5 × "
                f"{r.sgamma:.3f} × {r.gamma_eff:.2f} × {r.B_eff:.3f}"
                f" × {r.Ngamma:.2f}"),
            result_name="q_ult",
            result_value=f"{r.q_ultimate:,.1f}",
            result_unit="kPa",
            reference="Terzaghi (1943); UFC 3-220-20 Eq. 5-15",
        ),
        TableData(
            title="Term Breakdown",
            headers=["Term", "Value (kPa)"],
            rows=[
                ["Cohesion", f"{r.term_cohesion:,.1f}"],
                ["Overburden", f"{r.term_overburden:,.1f}"],
                ["Self-weight", f"{r.term_selfweight:,.1f}"],
                ["Total q_ult", f"{r.q_ultimate:,.1f}"],
            ],
        ),
        CalcStep(
            title="Allowable Bearing Capacity",
            equation="q_all = q_ult / FS",
            substitution=f"q_all = {r.q_ultimate:,.1f} / {r.factor_of_safety:.1f}",
            result_name="q_all",
            result_value=f"{r.q_allowable:,.1f}",
            result_unit="kPa",
        ),
        *_allowable_load_items(r),
    ]
    if r.is_two_layer:
        eq_items.append(TableData(
            title="Two-Layer Capacities (load-spread method; NAVFAC DM-7.01 / Bowles)",
            headers=["Layer", "q_ult (kPa)"],
            rows=[
                ["Upper layer (if semi-infinite)", f"{r.q_upper_layer:,.1f}"],
                ["Lower layer (if semi-infinite)", f"{r.q_lower_layer:,.1f}"],
                ["Combined (load-spread)", f"{r.q_ultimate:,.1f}"],
            ],
        ))
    eq_items.extend(r.notes)
    sections.append(CalcSection(title="Bearing Capacity Equation", items=eq_items))
    return sections


def _allowable_load_items(r) -> list:
    """The net allowable bearing pressure and the allowable structural load
    on the footing, after q_all (GM-170; UFC 3-220-20 Sec. 5-2.2 and
    5-2.4)."""
    items = [CalcStep(
        title="Net Allowable Bearing Pressure",
        equation="q_all,net = q_ult / FS - q",
        substitution=(f"q_all,net = {r.q_allowable:,.1f} - "
                      f"{r.q_overburden:,.1f}"),
        result_name="q_all,net",
        result_value=f"{r.q_allowable_net:,.1f}",
        result_unit="kPa",
        reference="UFC 3-220-20 Eq. 5-14",
    )]
    loads = r.allowable_loads()
    if not loads:
        return items
    unit = "kN_per_m" if r.is_strip else "kN"
    area = "B'" if r.is_strip else "A'"
    if r.area_full_m2 in (None, r.area_m2):
        equation = f"Q_DL+LL = q_all,net × {area}"
        substitution = (f"Q_DL+LL = {r.q_allowable_net:,.1f} × "
                        f"{r.area_m2:,.3f}")
    else:
        # an eccentric load bears on A'; the foundation's and the soil's
        # weight on the whole footing (Eq. 5-10)
        whole = "B" if r.is_strip else "A"
        equation = f"Q_DL+LL = q_all × {area} - q × {whole}"
        substitution = (f"Q_DL+LL = {r.q_allowable:,.1f} × "
                        f"{r.area_m2:,.3f} - {r.q_overburden:,.1f} × "
                        f"{r.area_full_m2:,.3f}")
    items.append(CalcStep(
        title="Allowable Structural Load",
        equation=equation,
        substitution=substitution,
        result_name="Q_DL+LL",
        result_value=f"{loads['structural_load_' + unit]:,.0f}",
        result_unit="kN/m" if r.is_strip else "kN",
        reference="UFC 3-220-20 Eq. 5-3",
    ))
    items.append(
        "The foundation and the soil over it are taken at the soil's unit "
        "weight (UFC 3-220-20 Sec. 5-2.2). q_all "
        f"× {area} = {loads['gross_load_' + unit]:,.0f} "
        f"{'kN/m' if r.is_strip else 'kN'} is the gross load on the bearing "
        "surface, the structural load plus the foundation's and the soil's "
        "weight (Eq. 5-1); it is not the structural load.")
    return items


def _shape_factor_equation(method: str, phi_deg: float) -> str:
    """Return the shape factor equation as text.

    Mirrors :func:`bearing_capacity.factors.shape_factors`, including the
    phi = 0 special forms, so the displayed equation matches the computed
    values (BC-9).
    """
    if method.lower() == "vesic":
        if phi_deg == 0:
            return (
                "s_c = 1 + 0.2(B/L)  [\u03c6 = 0],  "
                "s_q = 1.0,  "
                "s_\u03b3 = 1 - 0.4(B/L)"
            )
        return (
            "s_c = 1 + (B/L)(N_q/N_c),  "
            "s_q = 1 + (B/L)tan(\u03c6),  "
            "s_\u03b3 = 1 - 0.4(B/L)"
        )
    # Meyerhof
    if phi_deg <= 10:
        return (
            "s_c = 1 + 0.2\u00d7Kp(B/L),  "
            "s_q = s_\u03b3 = 1.0  [\u03c6 \u2264 10\u00b0]"
        )
    return (
        "s_c = 1 + 0.2\u00d7Kp(B/L),  "
        "s_q = s_\u03b3 = 1 + 0.1\u00d7Kp(B/L)"
    )


def _depth_factor_reference(analysis) -> str:
    """Where the depth factors used are printed (GM-197)."""
    dm = analysis.depth_factor_method
    if dm == "none":
        return "None used; FHWA GEC-6 Sec. 5.2.3.3, Table 5-4 note"
    if dm == "gec6":
        return "FHWA GEC-6 Eq. 5-14, Table 5-4 (Brinch Hansen 1970)"
    if analysis.factor_method == "meyerhof":
        return ("Meyerhof; UFC 3-220-20 Table 5-6; k beyond D_f/B = 1, "
                "Brinch Hansen (1970), Table 5-8 note B")
    return ("Brinch Hansen (1970); UFC 3-220-20 Tables 5-7, 5-8; "
            "FHWA GEC-6 Table 5-4")


def _inclination_factor_equation(analysis, phi_deg: float) -> str:
    """The load inclination factors as text, as
    :func:`bearing_capacity.factors.inclination_factors` works them out."""
    if analysis.factor_method == "meyerhof":
        return ("i_c = i_q = (1 - \u03b2/90)\u00b2,  "
                "i_\u03b3 = (1 - \u03b2/\u03c6)\u00b2")
    m = "m = (2 + B/L)/(1 + B/L)"
    if phi_deg == 0:
        return f"i_c = 1 - m\u00d7H/(A'\u00d7c\u00d7N_c),  H = V\u00d7tan(\u03b2),  {m}"
    if analysis.soil.layer1.cohesion == 0:
        return ("i_q = (1 - tan(\u03b2))^m,  i_\u03b3 = (1 - tan(\u03b2))^(m+1)"
                "  [c = 0: H/V = tan(\u03b2)],  "
                f"i_c = i_q - (1-i_q)/(N_c\u00d7tan(\u03c6)),  {m}")
    return ("i_q = [1 - H/(V + A'\u00d7c\u00d7cot(\u03c6))]^m,  "
            "i_\u03b3 = [...]^(m+1),  i_c = i_q - (1-i_q)/(N_c\u00d7tan(\u03c6)),"
            f"  H = V\u00d7tan(\u03b2),  {m}")


def _depth_factor_equation(method: str, phi_deg: float,
                           depth_method: str = "hansen") -> str:
    """Return the depth factor equation as text.

    Mirrors :func:`bearing_capacity.factors.depth_factors`, including the
    d_c formula and the phi = 0 special forms, so the displayed equation
    matches the computed values (BC-9).
    """
    if depth_method == "none":
        return "d_c = d_q = d_\u03b3 = 1.0  (no depth factors)"
    if depth_method == "gec6":
        return (
            "d_q = 1 + 2\u00d7tan(\u03c6)\u00d7(1-sin(\u03c6))\u00b2\u00d7"
            "arctan(D_f/B) for D_f/B \u2265 1, 1.0 below,  "
            "d_c = d_\u03b3 = 1.0"
        )
    if method.lower() == "vesic":
        if phi_deg == 0:
            return (
                "d_c = 1 + 0.4\u00d7k  [\u03c6 = 0],  "
                "d_q = 1.0,  d_\u03b3 = 1.0,  "
                "k = D_f/B if D_f/B < 1 else arctan(D_f/B)"
            )
        return (
            "d_q = 1 + 2\u00d7tan(\u03c6)\u00d7(1-sin(\u03c6))\u00b2\u00d7k,  "
            "d_c = d_q - (1-d_q)/(N_c\u00d7tan(\u03c6)),  "
            "d_\u03b3 = 1.0,  "
            "k = D_f/B if D_f/B < 1 else arctan(D_f/B)"
        )
    # Meyerhof
    if phi_deg <= 10:
        return (
            "d_c = 1 + 0.2\u00d7\u221aKp\u00d7k,  "
            "d_q = d_\u03b3 = 1.0  [\u03c6 \u2264 10\u00b0],  "
            "k = D_f/B if D_f/B \u2264 1 else arctan(D_f/B)"
        )
    return (
        "d_c = 1 + 0.2\u00d7\u221aKp\u00d7k,  "
        "d_q = d_\u03b3 = 1 + 0.1\u00d7\u221aKp\u00d7k,  "
        "k = D_f/B if D_f/B \u2264 1 else arctan(D_f/B)"
    )


def _plot_term_breakdown(result):
    """Create a horizontal bar chart of bearing capacity term contributions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    terms = ['Cohesion', 'Overburden', 'Self-weight']
    values = [result.term_cohesion, result.term_overburden, result.term_selfweight]
    colors = ['#2563eb', '#16a34a', '#d97706']

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(terms, values, color=colors, edgecolor='#333', height=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        if val > 0:
            pct = 100 * val / result.q_ultimate if result.q_ultimate > 0 else 0
            ax.text(bar.get_width() + max(values) * 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{val:,.1f} kPa ({pct:.0f}%)',
                    va='center', fontsize=9)

    ax.set_xlabel('Contribution to q_ult (kPa)', fontsize=10)
    ax.set_title(f'Bearing Capacity Term Breakdown  (q_ult = {result.q_ultimate:,.1f} kPa)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(0, max(values) * 1.35 if max(values) > 0 else 1)
    ax.grid(True, axis='x', alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def _plot_footing_section(result, analysis):
    """Create a schematic cross-section of the footing and soil."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    f = analysis.footing
    s = analysis.soil
    B = f.width
    Df = f.depth

    fig, ax = plt.subplots(figsize=(7, 5))

    # Soil background
    soil_depth = Df + B * 1.5
    ax.add_patch(patches.Rectangle(
        (-B * 1.5, 0), B * 4, soil_depth,
        facecolor='#f5e6c8', edgecolor='none',
    ))

    # GWT line
    if s.gwt_depth is not None and s.gwt_depth < soil_depth:
        ax.axhline(y=s.gwt_depth, color='#2563eb', linestyle='--',
                   linewidth=1.5, label=f'GWT = {s.gwt_depth:.1f} m')
        # Water-colored zone below GWT
        ax.add_patch(patches.Rectangle(
            (-B * 1.5, s.gwt_depth), B * 4, soil_depth - s.gwt_depth,
            facecolor='#dbeafe', edgecolor='none', alpha=0.5,
        ))

    # Ground surface
    ax.axhline(y=0, color='#16a34a', linewidth=2, label='Ground surface')

    # Footing
    footing_width_display = B
    footing_thickness = max(Df * 0.3, 0.15)
    ax.add_patch(patches.Rectangle(
        (-footing_width_display / 2, Df - footing_thickness),
        footing_width_display, footing_thickness,
        facecolor='#94a3b8', edgecolor='#1a1a1a', linewidth=1.5,
    ))

    # Dimension annotations
    # B dimension
    ax.annotate('', xy=(-B / 2, Df + B * 0.15), xytext=(B / 2, Df + B * 0.15),
                arrowprops=dict(arrowstyle='<->', color='#1a1a1a', lw=1.2))
    # each dimension's value clear of its line, by points (placed by
    # metres the line ran through it at the printed width; GM-194)
    ax.annotate(f'B = {B:.2f} m', (0, Df + B * 0.15), xytext=(0, -3),
                textcoords='offset points', ha='center', va='top',
                fontsize=9, fontweight='bold')

    # Df dimension (a footing on the surface has none: its value over the
    # ground line)
    if Df > 0:
        ax.annotate('', xy=(B / 2 + B * 0.3, 0), xytext=(B / 2 + B * 0.3, Df),
                    arrowprops=dict(arrowstyle='<->', color='#1a1a1a', lw=1.2))
        ax.text(B / 2 + B * 0.4, Df / 2, f'D_f = {Df:.2f} m',
                ha='left', va='center', fontsize=9, fontweight='bold')
    else:
        ax.annotate(f'D_f = {Df:.2f} m', (B / 2 + B * 0.3, 0),
                    xytext=(0, 3), textcoords='offset points', ha='left',
                    va='bottom', fontsize=9, fontweight='bold')

    # Load arrow
    arrow_top = max(-B * 0.3, Df - footing_thickness - B * 0.6)
    ax.annotate('', xy=(0, Df - footing_thickness),
                xytext=(0, arrow_top),
                arrowprops=dict(arrowstyle='->', color='#dc2626', lw=2))
    ax.text(0.05, arrow_top, 'q', fontsize=11, fontweight='bold', color='#dc2626')

    # Pressure bulb (approximate)
    bulb_depth = Df + B * 1.2
    from matplotlib.patches import FancyBboxPatch
    bulb = patches.FancyBboxPatch(
        (-B * 0.6, Df), B * 1.2, B * 1.2,
        boxstyle="round,pad=0.1",
        facecolor='none', edgecolor='#dc2626',
        linestyle=':', linewidth=1, alpha=0.6,
    )
    ax.add_patch(bulb)

    # Soil label
    layer = s.layer1
    soil_desc = []
    if layer.cohesion > 0:
        soil_desc.append(f"c = {layer.cohesion:.0f} kPa")
    if layer.friction_angle > 0:
        soil_desc.append(f"\u03c6 = {layer.friction_angle:.0f}\u00b0")
    soil_desc.append(f"\u03b3 = {layer.unit_weight:.1f} kN/m\u00b3")
    ax.text(-B * 1.2, Df + B * 0.5, '\n'.join(soil_desc),
            fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlim(-B * 1.8, B * 1.8)
    ax.set_ylim(soil_depth, min(-B * 0.5, arrow_top - B * 0.2))
    ax.set_xlabel('Distance (m)', fontsize=10)
    ax.set_ylabel('Depth (m)', fontsize=10)
    ax.set_title('Footing Cross-Section', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig
