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
]


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

    items.extend([
        InputItem("FS", "Factor of safety", analysis.factor_of_safety, ""),
        InputItem("Method", "N\u03b3 method", analysis.ngamma_method.capitalize(), ""),
    ])

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
    phi = f.width  # just for variable naming clarity below
    phi_deg = analysis.soil.layer1.friction_angle
    phi_rad = math.radians(phi_deg)
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
        reference="Reissner (1924); FHWA GEC-6 Eq. 6-2",
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
        reference="Prandtl (1921); FHWA GEC-6 Eq. 6-1",
    ))

    # Ngamma
    method = analysis.ngamma_method.lower()
    if method == "vesic":
        ng_eq = "N_\u03b3 = 2 \u00d7 (N_q + 1) \u00d7 tan(\u03c6)"
        ng_sub = f"N_\u03b3 = 2 \u00d7 ({r.Nq:.2f} + 1) \u00d7 tan({phi_deg:.1f}\u00b0)"
        ng_ref = "Vesic (1973); FHWA GEC-6 Table 6-1"
    elif method == "meyerhof":
        ng_eq = "N_\u03b3 = (N_q - 1) \u00d7 tan(1.4\u03c6)"
        ng_sub = f"N_\u03b3 = ({r.Nq:.2f} - 1) \u00d7 tan(1.4 \u00d7 {phi_deg:.1f}\u00b0)"
        ng_ref = "Meyerhof (1963)"
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
    cf_items = []

    # Shape factors
    cf_items.append(CalcStep(
        title="Shape Factors",
        equation=_shape_factor_equation(analysis.factor_method, phi_deg),
        substitution="",
        result_name="s_c, s_q, s_\u03b3",
        result_value=f"{r.sc:.4f}, {r.sq:.4f}, {r.sgamma:.4f}",
        reference=f"{factor_method}; FHWA GEC-6 Table 6-2",
    ))

    # Depth factors
    cf_items.append(CalcStep(
        title="Depth Factors",
        equation=_depth_factor_equation(analysis.factor_method, phi_deg),
        substitution="",
        result_name="d_c, d_q, d_\u03b3",
        result_value=f"{r.dc:.4f}, {r.dq:.4f}, {r.dgamma:.4f}",
        reference=f"{factor_method}; FHWA GEC-6 Table 6-3",
    ))

    # Only show inclination, base, and ground factors if non-unity
    if analysis.load_inclination != 0:
        cf_items.append(CalcStep(
            title="Inclination Factors",
            equation="(load inclined from vertical)",
            substitution=f"\u03b2 = {analysis.load_inclination:.1f}\u00b0",
            result_name="i_c, i_q, i_\u03b3",
            result_value=f"{r.ic:.4f}, {r.iq:.4f}, {r.igamma:.4f}",
            reference=f"{factor_method}; FHWA GEC-6 Table 6-4",
        ))

    if f.base_tilt != 0:
        cf_items.append(CalcStep(
            title="Base Inclination Factors",
            equation="(tilted footing base)",
            substitution=f"\u03b1 = {f.base_tilt:.1f}\u00b0",
            result_name="b_c, b_q, b_\u03b3",
            result_value=f"{r.bc:.4f}, {r.bq:.4f}, {r.bgamma:.4f}",
            reference="Hansen (1970); FHWA GEC-6 Table 6-5",
        ))

    if analysis.ground_slope != 0:
        cf_items.append(CalcStep(
            title="Ground Inclination Factors",
            equation="(sloping ground surface)",
            substitution=f"\u03c9 = {analysis.ground_slope:.1f}\u00b0",
            result_name="g_c, g_q, g_\u03b3",
            result_value=f"{r.gc:.4f}, {r.gq:.4f}, {r.ggamma:.4f}",
            reference="Hansen (1970); FHWA GEC-6 Table 6-6",
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
        reference="FHWA GEC-6, Eq. 6-1",
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

    # Two-layer info
    if r.is_two_layer:
        gbc_items.append(f"Two-layer analysis (Meyerhof & Hanna, 1978):")
        gbc_items.append(TableData(
            title="Two-Layer Capacities",
            headers=["Layer", "q_ult (kPa)"],
            rows=[
                ["Upper layer (if semi-infinite)", f"{r.q_upper_layer:,.1f}"],
                ["Lower layer (if semi-infinite)", f"{r.q_lower_layer:,.1f}"],
                ["Combined (interpolated)", f"{r.q_ultimate:,.1f}"],
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

def _shape_factor_equation(method: str, phi_deg: float) -> str:
    """Return the shape factor equation as text."""
    if method.lower() == "vesic":
        return (
            "s_c = 1 + (B/L)(N_q/N_c),  "
            "s_q = 1 + (B/L)tan(\u03c6),  "
            "s_\u03b3 = 1 - 0.4(B/L)"
        )
    return (
        "s_c = 1 + 0.2\u00d7Kp(B/L),  "
        "s_q = s_\u03b3 = 1 + 0.1\u00d7Kp(B/L)"
    )


def _depth_factor_equation(method: str, phi_deg: float) -> str:
    """Return the depth factor equation as text."""
    if method.lower() == "vesic":
        return (
            "d_q = 1 + 2\u00d7tan(\u03c6)\u00d7(1-sin(\u03c6))\u00b2\u00d7k,  "
            "d_\u03b3 = 1.0,  "
            "k = D_f/B if D_f/B \u2264 1 else arctan(D_f/B)"
        )
    return (
        "d_c = 1 + 0.2\u00d7\u221aKp\u00d7(D_f/B),  "
        "d_q = d_\u03b3 = 1 + 0.1\u00d7\u221aKp\u00d7(D_f/B)"
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
    ax.text(0, Df + B * 0.2, f'B = {B:.2f} m', ha='center', fontsize=9, fontweight='bold')

    # Df dimension
    ax.annotate('', xy=(B / 2 + B * 0.3, 0), xytext=(B / 2 + B * 0.3, Df),
                arrowprops=dict(arrowstyle='<->', color='#1a1a1a', lw=1.2))
    ax.text(B / 2 + B * 0.4, Df / 2, f'D_f = {Df:.2f} m',
            ha='left', va='center', fontsize=9, fontweight='bold')

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
