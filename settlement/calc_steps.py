"""
Calculation package steps for settlement analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation.

References:
    FHWA GEC-6 (FHWA-IF-02-054), Chapter 8
    Terzaghi, K. (1925) — Theory of 1-D consolidation
    Schmertmann, J.H. et al. (1978) — Improved strain influence factors
    Timoshenko & Goodier — Elastic settlement theory
    Mesri, G. (1973) — Secondary compression
"""

import math
from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Settlement Analysis"

REFERENCES = [
    "FHWA GEC-6 (FHWA-IF-02-054): Shallow Foundations, Chapter 8.",
    'Terzaghi, K. (1925). "Erdbaumechanik auf Bodenphysikalischer Grundlage." '
    "Deuticke, Vienna.",
    'Schmertmann, J.H. et al. (1978). "Improved Strain Influence Factor '
    'Diagrams." JGED, ASCE, Vol. 104, No. GT8, pp. 1131-1135.',
    "Timoshenko, S.P. & Goodier, J.N. (1970). Theory of Elasticity, 3rd Ed. "
    "McGraw-Hill.",
    'Mesri, G. (1973). "Coefficient of Secondary Compression." '
    "JSMFE, ASCE, Vol. 99, No. SM1, pp. 123-137.",
    "USACE EM 1110-1-1904: Settlement Analysis.",
]


def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for settlement calc package.

    Parameters
    ----------
    result : SettlementResult
        Computed results.
    analysis : SettlementAnalysis
        Analysis object holding all inputs.

    Returns
    -------
    list of InputItem
    """
    items = [
        InputItem("q_app", "Applied bearing pressure", f"{analysis.q_applied:.1f}", "kPa"),
        InputItem("q_0", "Overburden pressure", f"{analysis.q_overburden:.1f}", "kPa"),
        InputItem("q_net", "Net applied pressure", f"{analysis.q_net:.1f}", "kPa"),
        InputItem("B", "Footing width", f"{analysis.B:.2f}", "m"),
        InputItem("L", "Footing length", f"{analysis.L:.2f}", "m"),
        InputItem("Shape", "Footing shape", analysis.footing_shape, ""),
        InputItem("Stress", "Stress distribution method", analysis.stress_method, ""),
    ]

    # Immediate settlement method parameters
    items.append(InputItem("Imm. Method", "Immediate settlement method",
                           analysis.immediate_method, ""))
    if analysis.immediate_method == "elastic":
        if analysis.Es_immediate is not None:
            items.append(InputItem("E_s", "Elastic modulus (immediate)",
                                   f"{analysis.Es_immediate:.0f}", "kPa"))
        items.append(InputItem("\u03bd", "Poisson's ratio", f"{analysis.nu:.2f}", ""))
    elif analysis.immediate_method == "schmertmann":
        if analysis.schmertmann_layers:
            items.append(InputItem("N_layers", "Schmertmann sublayers",
                                   str(len(analysis.schmertmann_layers)), ""))
        if analysis.time_years_schmertmann > 0:
            items.append(InputItem("t_creep", "Time for creep correction",
                                   f"{analysis.time_years_schmertmann:.1f}", "years"))

    # Consolidation parameters
    if analysis.consolidation_layers:
        items.append(InputItem("N_consol", "Consolidation sublayers",
                               str(len(analysis.consolidation_layers)), ""))

    # Time-rate parameters
    if analysis.cv is not None:
        items.append(InputItem("c_v", "Coefficient of consolidation",
                               f"{analysis.cv:.4f}", "m\u00b2/yr"))
        items.append(InputItem("Drainage", "Drainage condition",
                               analysis.drainage, ""))

    # Secondary compression parameters
    if analysis.C_alpha is not None:
        items.append(InputItem("C_\u03b1", "Secondary compression index",
                               f"{analysis.C_alpha:.4f}", ""))
        items.append(InputItem("e_0(sec)", "Void ratio (secondary)",
                               f"{analysis.e0_secondary:.2f}", ""))
        if analysis.t_secondary > 0:
            items.append(InputItem("t_sec", "Secondary time",
                                   f"{analysis.t_secondary:.1f}", "years"))

    return items


def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections.

    Parameters
    ----------
    result : SettlementResult
        Computed results.
    analysis : SettlementAnalysis
        Analysis object holding all inputs.

    Returns
    -------
    list of CalcSection
    """
    sections = []

    # -- Section: Net Applied Pressure -----------------------------------
    pressure_items = []
    pressure_items.append(CalcStep(
        title="Net Applied Pressure",
        equation="q_net = q_applied - q_overburden",
        substitution=f"q_net = {analysis.q_applied:.1f} - {analysis.q_overburden:.1f}",
        result_name="q_net",
        result_value=f"{analysis.q_net:.1f}",
        result_unit="kPa",
        reference="FHWA GEC-6, Section 8.2",
    ))
    sections.append(CalcSection(title="Applied Loading", items=pressure_items))

    # -- Section: Immediate Settlement -----------------------------------
    imm_items = []

    if analysis.immediate_method == "elastic":
        imm_items.append(CalcStep(
            title="Elastic (Immediate) Settlement",
            equation="S_e = q_net \u00d7 B \u00d7 (1 - \u03bd\u00b2) / E_s \u00d7 I_w",
            substitution=_elastic_substitution(analysis),
            result_name="S_e",
            result_value=f"{result.immediate * 1000:.2f}",
            result_unit="mm",
            reference="Timoshenko & Goodier; FHWA GEC-6, Eq. 8-1",
            notes="I_w = 1.0 (flexible footing on surface)" if result.immediate > 0 else "",
        ))
    elif analysis.immediate_method == "schmertmann":
        imm_items.append(CalcStep(
            title="Schmertmann's Improved Method (1978)",
            equation="S_e = C_1 \u00d7 C_2 \u00d7 C_3 \u00d7 q_net \u00d7 \u03a3(I_z / E_s \u00d7 \u0394z)",
            substitution="",
            result_name="S_e",
            result_value=f"{result.immediate * 1000:.2f}",
            result_unit="mm",
            reference="Schmertmann et al. (1978); FHWA GEC-6, Section 8.3.2",
        ))

        # Schmertmann correction factors
        if analysis.schmertmann_layers and analysis.q_net > 0:
            C1 = 1.0 - 0.5 * (analysis.q_overburden / analysis.q_net)
            C1 = max(C1, 0.5)
            imm_items.append(CalcStep(
                title="Depth Correction Factor C_1",
                equation="C_1 = 1 - 0.5 \u00d7 (q_0 / q_net) \u2265 0.5",
                substitution=(
                    f"C_1 = 1 - 0.5 \u00d7 ({analysis.q_overburden:.1f} / "
                    f"{analysis.q_net:.1f})"
                ),
                result_name="C_1",
                result_value=f"{C1:.3f}",
                reference="Schmertmann et al. (1978), Eq. 2",
            ))

            if analysis.time_years_schmertmann > 0.1:
                C2 = 1.0 + 0.2 * math.log10(analysis.time_years_schmertmann / 0.1)
            else:
                C2 = 1.0
            imm_items.append(CalcStep(
                title="Creep Correction Factor C_2",
                equation="C_2 = 1 + 0.2 \u00d7 log\u2081\u2080(t / 0.1)" if C2 > 1.0
                else "C_2 = 1.0 (t \u2264 0.1 years, no creep correction)",
                substitution=(
                    f"C_2 = 1 + 0.2 \u00d7 log\u2081\u2080"
                    f"({analysis.time_years_schmertmann:.1f} / 0.1)"
                ) if C2 > 1.0 else "",
                result_name="C_2",
                result_value=f"{C2:.3f}",
                reference="Schmertmann et al. (1978), Eq. 3",
            ))

        # Schmertmann layer table
        if analysis.schmertmann_layers:
            layer_rows = []
            for i, layer in enumerate(analysis.schmertmann_layers, 1):
                layer_rows.append([
                    str(i),
                    f"{layer.depth_top:.2f}",
                    f"{layer.depth_bottom:.2f}",
                    f"{layer.thickness:.2f}",
                    f"{layer.Es:.0f}",
                ])
            imm_items.append(TableData(
                title="Schmertmann Sublayers",
                headers=["#", "z_top (m)", "z_bot (m)", "\u0394z (m)", "E_s (kPa)"],
                rows=layer_rows,
            ))

    sections.append(CalcSection(title="Immediate Settlement", items=imm_items))

    # -- Section: Primary Consolidation ----------------------------------
    if analysis.consolidation_layers:
        consol_items = []

        # Consolidation equation
        consol_items.append(CalcStep(
            title="Primary Consolidation Settlement (e-log(p) Method)",
            equation=_consolidation_equation_text(),
            substitution="",
            result_name="S_c",
            result_value=f"{result.consolidation * 1000:.2f}",
            result_unit="mm",
            reference="Terzaghi (1925); FHWA GEC-6, Eqs. 8-5 through 8-7",
        ))

        # Stress distribution note
        consol_items.append(CalcStep(
            title="Stress Distribution Method",
            equation=_stress_method_equation(analysis.stress_method),
            substitution="",
            result_name="Method",
            result_value=analysis.stress_method,
            reference=_stress_method_reference(analysis.stress_method),
        ))

        # Per-layer consolidation table
        if result.consolidation_layers:
            layer_rows = []
            for i, layer_data in enumerate(result.consolidation_layers, 1):
                desc = layer_data.get("description", "")
                label = desc if desc else f"Layer {i}"
                layer_rows.append([
                    str(i),
                    label,
                    f"{layer_data['depth_m']:.2f}",
                    f"{layer_data['thickness_m']:.2f}",
                    f"{layer_data['delta_sigma_kPa']:.1f}",
                    f"{layer_data['OCR']:.2f}",
                    f"{layer_data['settlement_mm']:.2f}",
                ])

            consol_items.append(TableData(
                title="Consolidation Settlement by Layer",
                headers=["#", "Description", "Depth (m)", "H (m)",
                         "\u0394\u03c3 (kPa)", "OCR", "S_c (mm)"],
                rows=layer_rows,
                notes=f"Total consolidation: {result.consolidation * 1000:.2f} mm",
            ))

        # Show detailed per-layer equations for first layer (representative)
        if analysis.consolidation_layers:
            first_layer = analysis.consolidation_layers[0]
            consol_items.append(_consolidation_layer_step(first_layer, analysis))

        sections.append(CalcSection(
            title="Primary Consolidation Settlement", items=consol_items
        ))

    # -- Section: Secondary Compression ----------------------------------
    if analysis.C_alpha is not None and analysis.t_secondary > 0:
        sec_items = []

        sec_items.append(CalcStep(
            title="Secondary Compression Settlement",
            equation="S_s = [C_\u03b1 / (1 + e_0)] \u00d7 H \u00d7 log\u2081\u2080(t_2 / t_1)",
            substitution=_secondary_substitution(analysis, result),
            result_name="S_s",
            result_value=f"{result.secondary * 1000:.2f}",
            result_unit="mm",
            reference="Mesri (1973); FHWA GEC-6, Eq. 8-20",
            notes="t_1 = time at end of primary consolidation, "
                  "t_2 = t_1 + t_secondary",
        ))

        # Modified secondary compression index
        C_alpha_eps = analysis.C_alpha / (1.0 + analysis.e0_secondary)
        sec_items.append(CalcStep(
            title="Modified Secondary Compression Index",
            equation="C_\u03b1\u03b5 = C_\u03b1 / (1 + e_0)",
            substitution=(
                f"C_\u03b1\u03b5 = {analysis.C_alpha:.4f} / "
                f"(1 + {analysis.e0_secondary:.2f})"
            ),
            result_name="C_\u03b1\u03b5",
            result_value=f"{C_alpha_eps:.5f}",
            reference="Mesri (1973)",
        ))

        sections.append(CalcSection(
            title="Secondary Compression Settlement", items=sec_items
        ))

    # -- Section: Total Settlement Summary --------------------------------
    summary_items = []

    # Total equation
    summary_items.append(CalcStep(
        title="Total Settlement",
        equation="S_total = S_immediate + S_consolidation + S_secondary",
        substitution=(
            f"S_total = {result.immediate * 1000:.2f} + "
            f"{result.consolidation * 1000:.2f} + "
            f"{result.secondary * 1000:.2f}"
        ),
        result_name="S_total",
        result_value=f"{result.total * 1000:.2f}",
        result_unit="mm",
    ))

    # Breakdown table
    total = result.total
    pct_imm = 100 * result.immediate / total if total > 0 else 0
    pct_con = 100 * result.consolidation / total if total > 0 else 0
    pct_sec = 100 * result.secondary / total if total > 0 else 0

    summary_items.append(TableData(
        title="Settlement Component Breakdown",
        headers=["Component", "Settlement (mm)", "Contribution (%)"],
        rows=[
            ["Immediate", f"{result.immediate * 1000:.2f}", f"{pct_imm:.0f}%"],
            ["Consolidation", f"{result.consolidation * 1000:.2f}", f"{pct_con:.0f}%"],
            ["Secondary", f"{result.secondary * 1000:.2f}", f"{pct_sec:.0f}%"],
            ["Total", f"{result.total * 1000:.2f}", "100%"],
        ],
    ))

    sections.append(CalcSection(title="Total Settlement Summary", items=summary_items))

    return sections


def get_figures(result, analysis) -> List[FigureData]:
    """Generate figures for the settlement calc package.

    Parameters
    ----------
    result : SettlementResult
        Computed results.
    analysis : SettlementAnalysis
        Analysis object.

    Returns
    -------
    list of FigureData
    """
    figures = []

    # Figure 1: Settlement breakdown bar chart
    try:
        fig = _plot_settlement_breakdown(result)
        b64 = figure_to_base64(fig, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        figures.append(FigureData(
            title="Settlement Component Breakdown",
            image_base64=b64,
            caption=(
                f"Figure 1: Contribution of immediate, consolidation, and secondary "
                f"settlement components to total settlement "
                f"(S_total = {result.total * 1000:.1f} mm)."
            ),
            width_percent=70,
        ))
    except ImportError:
        pass

    # Figure 2: Time-settlement curve (if available)
    if result.time_settlement_curve is not None:
        try:
            fig = _plot_time_settlement(result)
            b64 = figure_to_base64(fig, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig)
            figures.append(FigureData(
                title="Settlement vs Time",
                image_base64=b64,
                caption=(
                    f"Figure 2: Settlement vs time curve. Immediate settlement = "
                    f"{result.immediate * 1000:.1f} mm occurs at t = 0. "
                    f"Primary consolidation settlement = "
                    f"{result.consolidation * 1000:.1f} mm develops over time."
                ),
                width_percent=75,
            ))
        except ImportError:
            pass

    # Figure 3: Consolidation layer profile (if layer data available)
    if result.consolidation_layers:
        try:
            fig = _plot_consolidation_layers(result)
            b64 = figure_to_base64(fig, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig)
            fig_num = 3 if result.time_settlement_curve is not None else 2
            figures.append(FigureData(
                title="Consolidation Settlement by Layer",
                image_base64=b64,
                caption=(
                    f"Figure {fig_num}: Per-layer consolidation settlement and "
                    f"stress increase distribution with depth."
                ),
                width_percent=80,
            ))
        except ImportError:
            pass

    # DM7 Reference Figure: Terzaghi consolidation curve (when time-rate used)
    if result.time_settlement_curve is not None and analysis.cv is not None:
        try:
            import sys
            sys.path.insert(0, "DM7Eqs")
            from geotech.dm7_1.chapter5 import plot_figure_5_16
            from calc_package.dm7_figures import dm7_figure

            # Compute Tv at 50% consolidation as a representative query point
            from settlement.time_rate import time_factor as compute_Tv
            Hdr = analysis._get_Hdr()
            # Use time at 50% consolidation as the query point
            from settlement.time_rate import time_for_consolidation
            t50 = time_for_consolidation(50.0, analysis.cv, Hdr)
            Tv50 = compute_Tv(analysis.cv, t50, Hdr)

            figures.append(dm7_figure(
                plot_figure_5_16,
                Tv=Tv50,
                caption=(
                    f"UFC Figure 5-16: Average degree of consolidation vs time "
                    f"factor. Tv = {Tv50:.3f} at U = 50% "
                    f"(cv = {analysis.cv:.4f} m\u00b2/yr, "
                    f"Hdr = {Hdr:.2f} m, t\u2085\u2080 = {t50:.1f} yr)."
                ),
            ))
        except (ImportError, Exception):
            pass

    return figures


# -- Helper functions -------------------------------------------------------

def _elastic_substitution(analysis) -> str:
    """Build substitution string for elastic settlement equation."""
    if analysis.Es_immediate is None or analysis.Es_immediate <= 0:
        return "E_s not provided or non-positive; S_e = 0"
    if analysis.q_net <= 0:
        return "q_net \u2264 0; S_e = 0"
    return (
        f"S_e = {analysis.q_net:.1f} \u00d7 {analysis.B:.2f} \u00d7 "
        f"(1 - {analysis.nu:.2f}\u00b2) / {analysis.Es_immediate:.0f} \u00d7 1.0"
    )


def _consolidation_equation_text() -> str:
    """Return the general consolidation equation text."""
    return (
        "Case 1 (NC): S_c = [C_c / (1+e_0)] \u00d7 H \u00d7 "
        "log\u2081\u2080[(\u03c3'_v0 + \u0394\u03c3) / \u03c3'_v0]\n"
        "Case 2 (OC, stays OC): S_c = [C_r / (1+e_0)] \u00d7 H \u00d7 "
        "log\u2081\u2080[(\u03c3'_v0 + \u0394\u03c3) / \u03c3'_v0]\n"
        "Case 3 (OC, exceeds \u03c3'_p): S_c = [C_r / (1+e_0)] \u00d7 H \u00d7 "
        "log\u2081\u2080[\u03c3'_p / \u03c3'_v0] + "
        "[C_c / (1+e_0)] \u00d7 H \u00d7 "
        "log\u2081\u2080[(\u03c3'_v0 + \u0394\u03c3) / \u03c3'_p]"
    )


def _stress_method_equation(method: str) -> str:
    """Return the stress distribution equation text."""
    m = method.lower().replace(" ", "")
    if m == "2:1" or m == "2to1":
        return "\u0394\u03c3_z = q_net \u00d7 B \u00d7 L / [(B + z)(L + z)]"
    elif m == "boussinesq":
        return (
            "\u0394\u03c3_z = Boussinesq elastic solution "
            "(Newmark integration under center of rectangle)"
        )
    elif m == "westergaard":
        return (
            "\u0394\u03c3_z = Westergaard layered solution "
            "(alternating rigid/elastic layers)"
        )
    return f"\u0394\u03c3_z computed using '{method}' method"


def _stress_method_reference(method: str) -> str:
    """Return the reference for a stress distribution method."""
    m = method.lower().replace(" ", "")
    if m == "2:1" or m == "2to1":
        return "FHWA Soils & Foundations Reference Manual, Vol II, Section 8.3"
    elif m == "boussinesq":
        return "Boussinesq (1885); Newmark (1935)"
    elif m == "westergaard":
        return "Westergaard (1938)"
    return ""


def _consolidation_layer_step(layer, analysis) -> CalcStep:
    """Build a representative CalcStep for the first consolidation layer."""
    from settlement.stress_distribution import stress_at_depth

    delta_sigma = stress_at_depth(
        analysis.q_net, analysis.B, analysis.L,
        layer.depth_to_center,
        method=analysis.stress_method
    )

    sigma_final = layer.sigma_v0 + delta_sigma

    if layer.is_normally_consolidated:
        case = "Case 1 (NC)"
        eq = (
            f"S_c = [{layer.Cc:.3f} / (1 + {layer.e0:.2f})] \u00d7 "
            f"{layer.thickness:.2f} \u00d7 "
            f"log\u2081\u2080[({layer.sigma_v0:.1f} + {delta_sigma:.1f}) / "
            f"{layer.sigma_v0:.1f}]"
        )
    elif sigma_final <= layer.sigma_p:
        case = "Case 2 (OC, stays in OC range)"
        eq = (
            f"S_c = [{layer.Cr:.4f} / (1 + {layer.e0:.2f})] \u00d7 "
            f"{layer.thickness:.2f} \u00d7 "
            f"log\u2081\u2080[({layer.sigma_v0:.1f} + {delta_sigma:.1f}) / "
            f"{layer.sigma_v0:.1f}]"
        )
    else:
        case = "Case 3 (OC, exceeds preconsolidation)"
        eq = (
            f"S_c = [{layer.Cr:.4f} / (1 + {layer.e0:.2f})] \u00d7 "
            f"{layer.thickness:.2f} \u00d7 "
            f"log\u2081\u2080[{layer.sigma_p:.1f} / {layer.sigma_v0:.1f}] + "
            f"[{layer.Cc:.3f} / (1 + {layer.e0:.2f})] \u00d7 "
            f"{layer.thickness:.2f} \u00d7 "
            f"log\u2081\u2080[{sigma_final:.1f} / {layer.sigma_p:.1f}]"
        )

    from settlement.consolidation import consolidation_settlement_layer
    Sc = consolidation_settlement_layer(layer, delta_sigma)

    desc = layer.description if layer.description else "Layer 1"
    return CalcStep(
        title=f"Consolidation — {desc} ({case})",
        equation=eq,
        substitution=(
            f"\u0394\u03c3 = {delta_sigma:.1f} kPa at depth {layer.depth_to_center:.2f} m, "
            f"OCR = {layer.OCR:.2f}"
        ),
        result_name="S_c",
        result_value=f"{Sc * 1000:.2f}",
        result_unit="mm",
        reference="FHWA GEC-6, Eqs. 8-5 through 8-7",
    )


def _secondary_substitution(analysis, result) -> str:
    """Build substitution string for secondary settlement equation."""
    # Estimate t1 (end of primary consolidation)
    if analysis.cv is not None and analysis.consolidation_layers:
        Hdr = analysis._get_Hdr()
        from settlement.time_rate import time_for_consolidation
        t1 = time_for_consolidation(95.0, analysis.cv, Hdr)
    else:
        t1 = 1.0

    t2 = t1 + analysis.t_secondary
    total_H = sum(layer.thickness for layer in analysis.consolidation_layers) \
        if analysis.consolidation_layers else 0

    return (
        f"S_s = [{analysis.C_alpha:.4f} / (1 + {analysis.e0_secondary:.2f})] "
        f"\u00d7 {total_H:.2f} \u00d7 "
        f"log\u2081\u2080({t2:.2f} / {t1:.2f})"
    )


def _plot_settlement_breakdown(result):
    """Create a horizontal bar chart of settlement components."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    components = ['Immediate', 'Consolidation', 'Secondary']
    values = [
        result.immediate * 1000,
        result.consolidation * 1000,
        result.secondary * 1000,
    ]
    colors = ['#2563eb', '#16a34a', '#d97706']

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(components, values, color=colors, edgecolor='#333', height=0.5)

    # Add value labels on bars
    max_val = max(values) if max(values) > 0 else 1
    for bar, val in zip(bars, values):
        if val > 0:
            pct = 100 * val / (result.total * 1000) if result.total > 0 else 0
            ax.text(
                bar.get_width() + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f} mm ({pct:.0f}%)',
                va='center', fontsize=9,
            )

    ax.set_xlabel('Settlement (mm)', fontsize=10)
    ax.set_title(
        f'Settlement Component Breakdown  (S_total = {result.total * 1000:.1f} mm)',
        fontsize=11, fontweight='bold',
    )
    ax.set_xlim(0, max_val * 1.4 if max_val > 0 else 1)
    ax.grid(True, axis='x', alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def _plot_time_settlement(result):
    """Create a settlement vs time curve plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    times = [t for t, s in result.time_settlement_curve]
    settlements = [s * 1000 for t, s in result.time_settlement_curve]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, settlements, 'b-', linewidth=2, label='Total settlement')

    # Mark the immediate settlement as a horizontal line
    if result.immediate > 0:
        ax.axhline(y=result.immediate * 1000, color='#2563eb', linestyle='--',
                   linewidth=1, alpha=0.6,
                   label=f'Immediate = {result.immediate * 1000:.1f} mm')

    # Mark the final (imm + consolidation) as a horizontal line
    final_primary = (result.immediate + result.consolidation) * 1000
    if result.consolidation > 0:
        ax.axhline(y=final_primary, color='#16a34a', linestyle='--',
                   linewidth=1, alpha=0.6,
                   label=f'Imm + Consol = {final_primary:.1f} mm')

    ax.invert_yaxis()
    ax.set_xlabel('Time (years)', fontsize=10)
    ax.set_ylabel('Settlement (mm)', fontsize=10)
    ax.set_title('Settlement vs Time', fontsize=11, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3, which='both')
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def _plot_consolidation_layers(result):
    """Create a dual-axis plot of per-layer consolidation and stress increase."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers = result.consolidation_layers
    depths = [layer['depth_m'] for layer in layers]
    delta_sigmas = [layer['delta_sigma_kPa'] for layer in layers]
    settlements = [layer['settlement_mm'] for layer in layers]
    thicknesses = [layer['thickness_m'] for layer in layers]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Settlement bars
    color1 = '#2563eb'
    bar_width = [t * 0.6 for t in thicknesses]
    ax1.barh(depths, settlements, height=bar_width,
             color=color1, alpha=0.6, edgecolor='#333',
             label='Layer settlement')
    ax1.set_xlabel('Settlement (mm)', fontsize=10, color=color1)
    ax1.set_ylabel('Depth below footing base (m)', fontsize=10)
    ax1.tick_params(axis='x', labelcolor=color1)
    ax1.invert_yaxis()

    # Stress increase on secondary axis
    ax2 = ax1.twiny()
    color2 = '#dc2626'
    ax2.plot(delta_sigmas, depths, 'o-', color=color2, markersize=5,
             linewidth=1.5, label='\u0394\u03c3 stress increase')
    ax2.set_xlabel('\u0394\u03c3 (kPa)', fontsize=10, color=color2)
    ax2.tick_params(axis='x', labelcolor=color2)

    ax1.set_title('Consolidation Settlement & Stress Distribution by Layer',
                   fontsize=11, fontweight='bold', pad=25)
    ax1.grid(True, alpha=0.2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=8)

    plt.tight_layout()
    return fig
