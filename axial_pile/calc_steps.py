"""
Calculation package steps for axial pile capacity analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation.

References:
    FHWA GEC-12 (FHWA-NHI-16-009), Chapters 7-8
    Nordlund, R.D. (1963) — Bearing Capacity of Piles in Cohesionless Soils
    Tomlinson, M.J. (1971) — Alpha method for cohesive soils
    Burland, J.B. (1973) — Beta (effective stress) method
"""

from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Axial Pile Capacity Analysis"

REFERENCES = [
    'FHWA GEC-12 (FHWA-NHI-16-009): Design and Construction of Driven '
    'Pile Foundations, Volumes I & II.',
    'Nordlund, R.D. (1963). "Bearing Capacity of Piles in Cohesionless '
    'Soils." JSMFD, ASCE, Vol. 89, No. SM3, pp. 1-35.',
    'Tomlinson, M.J. (1971). "Some Effects of Pile Driving on Skin '
    'Friction." Behaviour of Piles, ICE, London, pp. 107-114.',
    'Burland, J.B. (1973). "Shaft Friction of Piles in Clay — A Simple '
    'Fundamental Approach." Ground Engineering, Vol. 6, No. 3, pp. 30-42.',
    'Meyerhof, G.G. (1976). "Bearing Capacity and Settlement of Pile '
    'Foundations." JGED, ASCE, Vol. 102, No. GT3, pp. 197-228.',
    'API RP 2GEO (2014). Geotechnical and Foundation Design Considerations.',
]


def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for axial pile calc package.

    Parameters
    ----------
    result : AxialPileResult
        Computed results.
    analysis : AxialPileAnalysis
        Analysis object holding pile, soil, and method inputs.

    Returns
    -------
    list of InputItem
    """
    pile = analysis.pile
    soil = analysis.soil

    items = [
        InputItem("Pile", "Pile section", pile.name, ""),
        InputItem("Type", "Pile type", pile.pile_type.replace("_", " ").title(), ""),
        InputItem("L", "Embedded pile length", f"{analysis.pile_length:.2f}", "m"),
        InputItem("D", "Pile width/diameter", f"{pile.width:.4f}", "m"),
    ]

    if pile.depth is not None and pile.depth != pile.width:
        items.append(InputItem("d", "Pile depth (section)", f"{pile.depth:.4f}", "m"))

    items.extend([
        InputItem("P", "Pile perimeter", f"{pile.perimeter:.4f}", "m"),
        InputItem("A_tip", "Tip area", f"{pile.tip_area:.6f}", "m\u00b2"),
        InputItem("A_s", "Cross-section area", f"{pile.area:.6f}", "m\u00b2"),
    ])

    # Soil layers
    depth_acc = 0.0
    for i, layer in enumerate(soil.layers, 1):
        top = depth_acc
        bot = depth_acc + layer.thickness
        desc = layer.description or layer.soil_type.capitalize()
        items.append(InputItem(
            f"Layer {i}",
            f"{desc} ({top:.1f}-{bot:.1f} m)",
            _layer_params_str(layer),
            "",
        ))
        depth_acc = bot

    if soil.gwt_depth is not None:
        items.append(InputItem("GWT", "Groundwater depth", f"{soil.gwt_depth:.1f}", "m"))

    # Method and FS
    method_display = _method_display_name(analysis.method)
    items.extend([
        InputItem("Method", "Analysis method", method_display, ""),
        InputItem("FS", "Factor of safety", analysis.factor_of_safety, ""),
    ])

    return items


def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections.

    Parameters
    ----------
    result : AxialPileResult
        Computed results.
    analysis : AxialPileAnalysis
        Analysis object.

    Returns
    -------
    list of CalcSection
    """
    sections = []

    # == Section 1: Soil Profile Summary ==
    soil_items = []
    layer_rows = []
    depth_acc = 0.0
    for i, layer in enumerate(analysis.soil.layers, 1):
        top = depth_acc
        bot = depth_acc + layer.thickness
        desc = layer.description or layer.soil_type.capitalize()
        params = _layer_params_str(layer)
        layer_rows.append([
            str(i),
            f"{top:.1f}",
            f"{bot:.1f}",
            desc,
            layer.soil_type.capitalize(),
            params,
        ])
        depth_acc = bot

    soil_items.append(TableData(
        title="Soil Layer Definition",
        headers=["#", "Top (m)", "Bottom (m)", "Description",
                 "Type", "Key Parameters"],
        rows=layer_rows,
    ))

    if analysis.soil.gwt_depth is not None:
        soil_items.append(CalcStep(
            title="Groundwater Table",
            equation="GWT below ground surface",
            substitution="",
            result_name="GWT",
            result_value=f"{analysis.soil.gwt_depth:.1f}",
            result_unit="m",
            notes="Effective stress computed using buoyant unit weight below GWT.",
        ))

    sections.append(CalcSection(title="Soil Profile", items=soil_items))

    # == Section 2: Pile Section Properties ==
    pile = analysis.pile
    pile_items = []

    pile_items.append(CalcStep(
        title="Pile Section",
        equation=f"{pile.name} ({pile.pile_type.replace('_', ' ').title()})",
        substitution="",
        result_name="Perimeter, Tip Area",
        result_value=f"P = {pile.perimeter:.4f} m, A_tip = {pile.tip_area:.6f} m\u00b2",
    ))

    pile_items.append(CalcStep(
        title="Pile Embedment",
        equation="L = embedded pile length",
        substitution="",
        result_name="L",
        result_value=f"{analysis.pile_length:.2f}",
        result_unit="m",
    ))

    sections.append(CalcSection(title="Pile Section Properties", items=pile_items))

    # == Section 3: Skin Friction (per-layer breakdown) ==
    skin_items = []

    method = result.method
    method_display = _method_display_name(method)

    # Method equation
    if method == "beta":
        skin_items.append(CalcStep(
            title="Beta (Effective Stress) Method",
            equation="f_s = \u03b2 \u00d7 \u03c3'_v",
            substitution="Q_s = \u03a3 (f_s \u00d7 P \u00d7 \u0394L)",
            result_name="Method",
            result_value=method_display,
            reference="Burland (1973); FHWA GEC-12 Ch. 8",
            notes="\u03b2 = K \u00d7 tan(\u03b4), where K depends on soil friction angle.",
        ))
    else:
        # Auto method: Nordlund for cohesionless, Tomlinson for cohesive
        skin_items.append(CalcStep(
            title="Combined Method (Nordlund + Tomlinson)",
            equation=(
                "Cohesionless: f_s = K_\u03b4 \u00d7 C_F \u00d7 \u03c3'_v \u00d7 "
                "sin(\u03b4 + \u03c9) / cos(\u03c9)  (Nordlund)\n"
                "Cohesive: f_s = \u03b1 \u00d7 c_u  (Tomlinson)"
            ),
            substitution="Q_s = \u03a3 (f_s \u00d7 P \u00d7 \u0394L)",
            result_name="Method",
            result_value="Auto (Nordlund/Tomlinson)",
            reference="Nordlund (1963); Tomlinson (1971); FHWA GEC-12 Ch. 7-8",
        ))

    # Per-layer breakdown table
    if result.layer_breakdown:
        breakdown_rows = []
        for layer_info in result.layer_breakdown:
            desc = layer_info.get("description", "") or layer_info["soil_type"].capitalize()
            method_used = layer_info["method"].capitalize()
            qs = layer_info["skin_friction_kN"]
            sigma_v = layer_info.get("sigma_v_kPa", "")
            pct = 100 * qs / result.Q_skin if result.Q_skin > 0 else 0
            breakdown_rows.append([
                f"{layer_info['depth_top_m']:.1f}-{layer_info['depth_bottom_m']:.1f}",
                desc,
                method_used,
                f"{sigma_v:.1f}" if isinstance(sigma_v, (int, float)) else str(sigma_v),
                f"{qs:,.1f}",
                f"{pct:.0f}%",
            ])

        skin_items.append(TableData(
            title="Per-Layer Skin Friction Breakdown",
            headers=["Depth (m)", "Description", "Method",
                     "\u03c3'_v (kPa)", "Q_s (kN)", "% of Total"],
            rows=breakdown_rows,
        ))

    skin_items.append(CalcStep(
        title="Total Skin Friction",
        equation="Q_s = \u03a3 Q_s,i (sum over all layers)",
        substitution=_skin_sum_substitution(result),
        result_name="Q_s",
        result_value=f"{result.Q_skin:,.1f}",
        result_unit="kN",
    ))

    sections.append(CalcSection(title="Skin Friction", items=skin_items))

    # == Section 4: End Bearing ==
    tip_items = []

    tip_layer_info = _get_tip_layer_info(analysis)

    if method == "beta":
        tip_items.append(CalcStep(
            title="End Bearing (Beta Method)",
            equation="Q_t = \u03c3'_v,tip \u00d7 N_t \u00d7 A_tip",
            substitution=(
                f"Q_t = {result.sigma_v_tip:.1f} \u00d7 N_t \u00d7 "
                f"{analysis.pile.tip_area:.6f}"
            ),
            result_name="Q_t",
            result_value=f"{result.Q_tip:,.1f}",
            result_unit="kN",
            reference="FHWA GEC-12 Ch. 8",
            notes=f"Tip in {tip_layer_info}. \u03c3'_v,tip = {result.sigma_v_tip:.1f} kPa.",
        ))
    elif tip_layer_info.startswith("cohesionless") or tip_layer_info.startswith("Cohesionless"):
        tip_items.append(CalcStep(
            title="End Bearing (Nordlund — Cohesionless)",
            equation="Q_t = \u03c3'_v,tip \u00d7 N_q \u00d7 A_tip (limited by N_q)",
            substitution=(
                f"Q_t = {result.sigma_v_tip:.1f} \u00d7 N_q \u00d7 "
                f"{analysis.pile.tip_area:.6f}"
            ),
            result_name="Q_t",
            result_value=f"{result.Q_tip:,.1f}",
            result_unit="kN",
            reference="Meyerhof (1976); FHWA GEC-12 Ch. 7",
            notes=f"\u03c3'_v at pile tip = {result.sigma_v_tip:.1f} kPa.",
        ))
    else:
        tip_items.append(CalcStep(
            title="End Bearing (Tomlinson — Cohesive)",
            equation="Q_t = N_c \u00d7 c_u \u00d7 A_tip",
            substitution=(
                f"Q_t = 9.0 \u00d7 c_u \u00d7 {analysis.pile.tip_area:.6f}"
            ),
            result_name="Q_t",
            result_value=f"{result.Q_tip:,.1f}",
            result_unit="kN",
            reference="Tomlinson (1971); FHWA GEC-12 Ch. 7",
            notes=f"N_c = 9.0 for driven piles. \u03c3'_v at pile tip = {result.sigma_v_tip:.1f} kPa.",
        ))

    sections.append(CalcSection(title="End Bearing", items=tip_items))

    # == Section 5: Ultimate & Allowable Capacity ==
    cap_items = []

    pct_skin = 100 * result.Q_skin / result.Q_ultimate if result.Q_ultimate > 0 else 0
    pct_tip = 100 * result.Q_tip / result.Q_ultimate if result.Q_ultimate > 0 else 0

    cap_items.append(CalcStep(
        title="Ultimate Axial Capacity",
        equation="Q_ult = Q_s + Q_t",
        substitution=f"Q_ult = {result.Q_skin:,.1f} + {result.Q_tip:,.1f}",
        result_name="Q_ult",
        result_value=f"{result.Q_ultimate:,.1f}",
        result_unit="kN",
        reference="FHWA GEC-12, Eq. 7-1",
    ))

    cap_items.append(TableData(
        title="Capacity Breakdown",
        headers=["Component", "Value (kN)", "Contribution (%)"],
        rows=[
            ["Skin friction (Q_s)", f"{result.Q_skin:,.1f}", f"{pct_skin:.0f}%"],
            ["End bearing (Q_t)", f"{result.Q_tip:,.1f}", f"{pct_tip:.0f}%"],
            ["Ultimate (Q_ult)", f"{result.Q_ultimate:,.1f}", "100%"],
        ],
    ))

    cap_items.append(CalcStep(
        title="Allowable Axial Capacity",
        equation="Q_all = Q_ult / FS",
        substitution=f"Q_all = {result.Q_ultimate:,.1f} / {result.factor_of_safety:.1f}",
        result_name="Q_all",
        result_value=f"{result.Q_allowable:,.1f}",
        result_unit="kN",
    ))

    if result.Q_uplift is not None:
        cap_items.append(CalcStep(
            title="Uplift (Tension) Capacity",
            equation="Q_uplift = 0.75 \u00d7 Q_s (conservative estimate)",
            substitution=f"Q_uplift = 0.75 \u00d7 {result.Q_skin:,.1f}",
            result_name="Q_uplift",
            result_value=f"{result.Q_uplift:,.1f}",
            result_unit="kN",
            reference="FHWA GEC-12 Ch. 7",
            notes="Uplift capacity reduced from compression skin friction.",
        ))

    sections.append(CalcSection(
        title="Ultimate & Allowable Capacity", items=cap_items
    ))

    return sections


def get_figures(result, analysis) -> List[FigureData]:
    """Generate figures for the axial pile calc package.

    Parameters
    ----------
    result : AxialPileResult
        Computed results.
    analysis : AxialPileAnalysis
        Analysis object.

    Returns
    -------
    list of FigureData
    """
    figures = []

    # Figure 1: Capacity breakdown bar chart
    try:
        fig = _plot_capacity_breakdown(result)
        b64 = figure_to_base64(fig, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        figures.append(FigureData(
            title="Axial Capacity Breakdown",
            image_base64=b64,
            caption=(
                f"Figure 1: Skin friction vs end bearing contribution to "
                f"ultimate axial capacity "
                f"(Q_ult = {result.Q_ultimate:,.1f} kN)."
            ),
            width_percent=70,
        ))
    except ImportError:
        pass

    # Figure 2: Skin friction profile vs depth
    if result.layer_breakdown:
        try:
            fig2 = _plot_skin_friction_profile(result)
            b64_2 = figure_to_base64(fig2, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig2)
            figures.append(FigureData(
                title="Skin Friction Profile vs Depth",
                image_base64=b64_2,
                caption=(
                    f"Figure 2: Per-layer skin friction distribution along the pile "
                    f"(L = {result.pile_length:.1f} m, {result.pile_name})."
                ),
                width_percent=70,
            ))
        except ImportError:
            pass

    return figures


# -- Helper functions --------------------------------------------------------

def _method_display_name(method: str) -> str:
    """Convert method code to display name."""
    mapping = {
        "auto": "Auto (Nordlund/Tomlinson)",
        "nordlund": "Nordlund (Cohesionless)",
        "tomlinson": "Tomlinson Alpha (Cohesive)",
        "beta": "Beta (Effective Stress)",
    }
    return mapping.get(method.lower(), method.capitalize())


def _layer_params_str(layer) -> str:
    """Build a concise parameter string for a soil layer."""
    parts = []
    parts.append(f"\u03b3 = {layer.unit_weight:.1f} kN/m\u00b3")
    if layer.soil_type == "cohesionless":
        parts.append(f"\u03c6 = {layer.friction_angle:.0f}\u00b0")
        if layer.delta_phi_ratio is not None:
            parts.append(f"\u03b4/\u03c6 = {layer.delta_phi_ratio:.2f}")
    else:
        parts.append(f"c_u = {layer.cohesion:.0f} kPa")
    return ", ".join(parts)


def _skin_sum_substitution(result) -> str:
    """Build the substitution string for the total skin friction sum."""
    if not result.layer_breakdown:
        return ""
    parts = [f"{lb['skin_friction_kN']:.1f}" for lb in result.layer_breakdown]
    if len(parts) <= 6:
        return "Q_s = " + " + ".join(parts)
    # Truncate for very many layers
    return "Q_s = " + " + ".join(parts[:5]) + " + ..."


def _get_tip_layer_info(analysis) -> str:
    """Get a description of the soil layer at the pile tip."""
    tip_layer = analysis.soil.layer_at_depth(analysis.pile_length - 0.01)
    if tip_layer is None:
        return "unknown"
    desc = tip_layer.description or tip_layer.soil_type.capitalize()
    return f"{tip_layer.soil_type} ({desc})"


def _plot_capacity_breakdown(result):
    """Create a horizontal bar chart of Q_skin vs Q_tip contributions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    components = ['Skin Friction\n(Q_s)', 'End Bearing\n(Q_t)']
    values = [result.Q_skin, result.Q_tip]
    colors = ['#2563eb', '#d97706']

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(components, values, color=colors, edgecolor='#333', height=0.5)

    # Value labels on bars
    max_val = max(values) if max(values) > 0 else 1
    for bar, val in zip(bars, values):
        pct = 100 * val / result.Q_ultimate if result.Q_ultimate > 0 else 0
        label_x = bar.get_width() + max_val * 0.02
        ax.text(label_x, bar.get_y() + bar.get_height() / 2,
                f'{val:,.1f} kN ({pct:.0f}%)',
                va='center', fontsize=9)

    ax.set_xlabel('Capacity (kN)', fontsize=10)
    ax.set_title(
        f'Axial Capacity Breakdown  (Q_ult = {result.Q_ultimate:,.1f} kN)',
        fontsize=11, fontweight='bold',
    )
    ax.set_xlim(0, max_val * 1.4 if max_val > 0 else 1)
    ax.grid(True, axis='x', alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def _plot_skin_friction_profile(result):
    """Create a depth-profile plot of per-layer skin friction."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    layers = result.layer_breakdown
    if not layers:
        fig, ax = plt.subplots(figsize=(6, 7))
        ax.text(0.5, 0.5, 'No layer data available', transform=ax.transAxes,
                ha='center', va='center')
        return fig

    fig, ax = plt.subplots(figsize=(6, 7))

    # Color by method
    method_colors = {
        'nordlund': '#2563eb',
        'tomlinson': '#16a34a',
        'beta': '#d97706',
    }

    for layer_info in layers:
        z_top = layer_info['depth_top_m']
        z_bot = layer_info['depth_bottom_m']
        qs = layer_info['skin_friction_kN']
        thickness = z_bot - z_top
        z_mid = (z_top + z_bot) / 2
        method_used = layer_info['method']
        color = method_colors.get(method_used, '#666666')

        bar = ax.barh(
            z_mid, qs, height=thickness * 0.85,
            color=color, edgecolor='#333', linewidth=0.8, alpha=0.8,
        )

        # Label on bar
        if qs > 0:
            label_x = qs + max(lb['skin_friction_kN'] for lb in layers) * 0.03
            ax.text(label_x, z_mid, f'{qs:.0f} kN',
                    va='center', fontsize=8)

    # Soil type labels on the left
    for layer_info in layers:
        z_mid = (layer_info['depth_top_m'] + layer_info['depth_bottom_m']) / 2
        desc = layer_info.get('description', '') or layer_info['soil_type'].capitalize()
        ax.text(-max(lb['skin_friction_kN'] for lb in layers) * 0.02, z_mid,
                f'{desc} ',
                va='center', ha='right', fontsize=7, fontstyle='italic')

    # Pile tip line
    ax.axhline(y=result.pile_length, color='#dc2626', linestyle='--',
               linewidth=1.5, label=f'Pile tip ({result.pile_length:.1f} m)')

    # Build legend from unique methods
    used_methods = set(lb['method'] for lb in layers)
    import matplotlib.patches as mpatches
    legend_handles = []
    for m in sorted(used_methods):
        color = method_colors.get(m, '#666666')
        legend_handles.append(mpatches.Patch(
            color=color, alpha=0.8, label=m.capitalize(),
        ))
    legend_handles.append(plt.Line2D([0], [0], color='#dc2626', linestyle='--',
                                     linewidth=1.5, label='Pile tip'))
    ax.legend(handles=legend_handles, loc='lower right', fontsize=8)

    max_qs = max(lb['skin_friction_kN'] for lb in layers)
    ax.set_xlim(-max_qs * 0.15 if max_qs > 0 else -1,
                max_qs * 1.35 if max_qs > 0 else 1)
    ax.set_ylim(result.pile_length * 1.08, -result.pile_length * 0.03)
    ax.set_xlabel('Skin Friction per Layer (kN)', fontsize=10)
    ax.set_ylabel('Depth (m)', fontsize=10)
    ax.set_title(
        f'Skin Friction Profile  ({result.pile_name})',
        fontsize=11, fontweight='bold',
    )
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig
