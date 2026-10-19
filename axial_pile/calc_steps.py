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
            reference="FHWA GEC-12 Eq. 7-19 (Sec. 7.2.1.3.4)",
            notes=_beta_notes(result, analysis),
        ))
    else:
        # Auto method: Nordlund for cohesionless, Tomlinson for cohesive
        skin_items.append(CalcStep(
            title="Combined Method (Nordlund + Tomlinson)",
            equation=(
                "Cohesionless: f_s = K_\u03b4 \u00d7 C_F \u00d7 \u03c3'_d \u00d7 "
                "sin(\u03b4)  (Nordlund, Eq. 7-8)\n"
                "Cohesive: f_s = C_a = \u03b1 \u00d7 s_u  (\u03b1-method, Eq. 7-10)"
            ),
            substitution="Q_s = \u03a3 (f_s \u00d7 P \u00d7 \u0394L)",
            result_name="Method",
            result_value="Auto (Nordlund/Tomlinson)",
            reference="FHWA GEC-12 Sec. 7.2.1.3.1 and 7.2.1.3.2",
            notes=_auto_method_notes(result, analysis),
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
            equation="Q_t = N_t \u00d7 \u03c3'_p \u00d7 A_p  (Eq. 7-20)",
            substitution=(
                f"Q_t = {result.toe_details.get('N_t', 0):g} \u00d7 "
                f"{result.sigma_v_tip:.1f} \u00d7 "
                f"{result.toe_details.get('toe_area_m2', 0):.6f}"
                + (" (open-end pipe limit governs)"
                   if result.toe_details.get("governs") ==
                   "open-end pipe limit" else "")
            ),
            result_name="Q_t",
            result_value=f"{result.Q_tip:,.1f}",
            result_unit="kN",
            reference="FHWA GEC-12 Sec. 7.2.1.3.4",
            notes=f"Tip in {tip_layer_info}. \u03c3'_v,tip = {result.sigma_v_tip:.1f} kPa.",
        ))
    elif "alpha_t" in result.toe_details:
        t = result.toe_details
        tip_items.append(CalcStep(
            title="End Bearing (Nordlund — Cohesionless)",
            equation=("Q_t = \u03b1_t \u00d7 N'_q \u00d7 \u03c3'_p \u00d7 A_p "
                      "\u2264 q_L \u00d7 A_p  (Eq. 7-9)"),
            substitution=(
                f"Q_t = min({t['alpha_t']:.3f} \u00d7 {t['N_q_prime']:.1f} "
                f"\u00d7 {t['sigma_p_kPa']:.1f}, {t['q_L_kPa']:,.0f}"
                + (f", {t['q_p_kPa']:,.0f}" if t["governs"] ==
                   "open-end pipe limit" else "")
                + f") \u00d7 {t['toe_area_m2']:.6f}"
            ),
            result_name="Q_t",
            result_value=f"{result.Q_tip:,.1f}",
            result_unit="kN",
            reference="FHWA GEC-12 Steps 7-9, Figs. 7-15 and 7-16",
            notes=((f"\u03c6 near the toe = {t['toe_phi_deg']:.1f}\u00b0"
                    + (" (averaged from the toe to 3b below, Step 7c)"
                       if t.get("toe_zone") else "") + "; ")
                   if "toe_phi_deg" in t else "")
                   + (f"\u03c3'_v at the toe = {t['sigma_v_tip_kPa']:.1f} kPa; "
                   f"\u03c3'_p = {t['sigma_p_kPa']:.1f} kPa; D/b = "
                   f"{t['D_over_b']:.1f}; governed by {t['governs']}."
                   + "".join(f" WARNING: {w}" for w in result.warnings)),
        ))
    else:
        tip_items.append(CalcStep(
            title="End Bearing (Tomlinson — Cohesive)",
            equation=("Q_t = 4.5 \u00d7 s_u \u00d7 A_p  (Eq. 7-90)"
                      if result.toe_details.get("governs") == "Eq. 7-90"
                      else "Q_t = N_c \u00d7 s_u \u00d7 A_p  (Eq. 7-11)"),
            substitution=(
                f"Q_t = {result.toe_details.get('q_p_kPa', 0):,.1f} \u00d7 "
                f"{result.toe_details.get('toe_area_m2', 0):.6f}"
            ),
            result_name="Q_t",
            result_value=f"{result.Q_tip:,.1f}",
            result_unit="kN",
            reference="FHWA GEC-12 Sec. 7.2.1.3.2 and 7.10.7",
            notes=(f"N_c = 9.0 for driven piles (4.5 in Tomlinson's Eq. 7-90 "
                   f"for a small open pipe in clay). \u03c3'_v at the toe = "
                   f"{result.sigma_v_tip:.1f} kPa."),
        ))

    sections.append(CalcSection(title="End Bearing", items=tip_items))

    # == Section 5: Ultimate & Allowable Capacity ==
    cap_items = []

    pct_skin = 100 * result.Q_skin / result.Q_ultimate if result.Q_ultimate > 0 else 0
    pct_tip = 100 * result.Q_tip / result.Q_ultimate if result.Q_ultimate > 0 else 0

    if result.plug_weight:
        ult_equation = "Q_ult = Q_s + Q_t - W_p  (Eq. 7-89, unplugged)"
        ult_sub = (f"Q_ult = {result.Q_skin:,.1f} + {result.Q_tip:,.1f} - "
                   f"{result.plug_weight:,.1f}")
    else:
        ult_equation = "Q_ult = Q_s + Q_t"
        ult_sub = f"Q_ult = {result.Q_skin:,.1f} + {result.Q_tip:,.1f}"
    cap_items.append(CalcStep(
        title="Ultimate Axial Capacity",
        equation=ult_equation,
        substitution=ult_sub,
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
        frac = analysis.uplift_skin_fraction
        weight = analysis.pile_weight
        outside = (result.Q_uplift - (weight or 0.0)) / frac if frac else 0.0
        cap_items.append(CalcStep(
            title="Uplift (Tension) Capacity",
            equation=("Q_uplift = f \u00d7 Q_s,outside"
                      + (" + W_pile" if weight is not None else "")),
            substitution=(f"Q_uplift = {frac:g} \u00d7 {outside:,.1f}"
                          + (f" + {weight:,.1f}" if weight is not None
                             else "")),
            result_name="Q_uplift",
            result_value=f"{result.Q_uplift:,.1f}",
            result_unit="kN",
            reference="FHWA GEC-12 Sec. 7.2.3.1, Eq. 7-38",
            notes=("GEC-12 takes the nominal shaft resistance in uplift, "
                   "with the uplift resistance factor carrying the "
                   "reduction; side resistance inside an open pipe is "
                   "left out."),
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


#: Pile lengths in the capacity-against-length table.
CAPACITY_TABLE_POINTS = 20


def get_tables(result, analysis) -> list:
    """APILE-style tables: the result's own (side resistance per layer,
    load transfer at the ultimate load, capacity summary), the ultimate
    capacity against pile length (``analysis.capacity_vs_depth``: the same
    analysis re-run at each length), and the soil layers as input (CSV
    only; the package prints them in its Soil Profile section)."""
    from geotech_common.tables import CalcTable

    tables = list(result.get_tables())

    profile_depth = analysis.soil.total_thickness - analysis.head_depth
    top = max(analysis.pile_length, profile_depth)
    low = min(3.0, analysis.pile_length)
    if top > low:
        rows, warned = [], False
        for r in analysis.capacity_vs_depth(
                depth_min=low, depth_max=top,
                n_points=CAPACITY_TABLE_POINTS):
            rows.append([r["depth_m"], r["Q_skin_kN"], r["Q_tip_kN"],
                         r["Q_ultimate_kN"]])
            warned = warned or bool(r.get("warnings"))
        if rows:
            note = ("The same analysis re-run with the pile length set to "
                    "each value (APILE's capacity against depth); FS is "
                    "not applied")
            if warned:
                note += ("; some lengths carry the toe warnings the "
                         "design length's result lists")
            tables.append(CalcTable(
                title="Ultimate capacity against pile length",
                columns=[("Pile length L", "m", ".2f"),
                         ("Side resistance Q_s", "kN", ".1f"),
                         ("Toe resistance Q_t", "kN", ".1f"),
                         ("Ultimate capacity Q_ult", "kN", ".1f")],
                rows=rows, note=note, name="capacity_vs_length",
            ))

    layer_rows, depth = [], 0.0
    for i, layer in enumerate(analysis.soil.layers, 1):
        layer_rows.append([
            i, depth, depth + layer.thickness, layer.soil_type,
            layer.description, layer.unit_weight, layer.friction_angle,
            layer.cohesion, layer.delta_phi_ratio])
        depth += layer.thickness
    if layer_rows:
        tables.append(CalcTable(
            title="Soil layers",
            columns=["Layer", ("Top", "m"), ("Bottom", "m"), "Soil type",
                     "Description", ("Unit weight γ", "kN/m³"),
                     ("Friction angle φ", "deg"),
                     ("Cohesion c_u", "kPa"), "δ/φ"],
            rows=layer_rows,
            note=("The soil layers as input, depths below the ground "
                  f"surface; groundwater at "
                  f"{analysis.soil.gwt_depth if analysis.soil.gwt_depth is not None else 'none given'}"
                  + (" m" if analysis.soil.gwt_depth is not None else "")),
            name="soil_layers", printed=False,
        ))
    return tables


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
    if len(parts) > 6:
        # Truncate for very many layers
        parts = parts[:5] + ["..."]
    # An unplugged open pipe adds its interior side resistance (Eq. 7-89)
    inside = result.Q_skin - sum(lb["skin_friction_kN"]
                                 for lb in result.layer_breakdown)
    if abs(inside) > 1e-6 * max(abs(result.Q_skin), 1.0):
        parts.append(f"{inside:.1f} (inside the pipe)")
    return "Q_s = " + " + ".join(parts)


#: The alpha-method's chart, as the method step says it (GM-144).
_ADHESION_CHART_TEXT = {
    "fig_7_17": "C_a from Fig. 7-17 by s_u, pile type and D/b",
    "fig_7_18": ("\u03b1 from Fig. 7-18 by s_u, stratigraphy case and the "
                 "penetration D/b into the clay"),
    "smaller": ("C_a the smaller of Fig. 7-17 (by s_u, pile type and D/b) "
                "and Fig. 7-18 (by s_u, stratigraphy case and D/b into the "
                "clay)"),
}


def _auto_method_notes(result, analysis=None) -> str:
    """The chart readings of each layer, for the method step."""
    parts = []
    for lb in result.layer_breakdown or []:
        where = f"{lb['depth_top_m']:.1f}-{lb['depth_bottom_m']:.1f} m"
        if lb["method"] == "nordlund":
            parts.append(f"{where}: \u03b4/\u03c6 = {lb['delta_phi_ratio']:.2f}, "
                         f"K_\u03b4 = {lb['K_delta']:.2f}, C_F = "
                         f"{lb['C_F']:.2f}")
        elif lb["method"] == "tomlinson":
            reading = (f"D/b = {lb['D_over_b']:.0f}"
                       if lb.get("adhesion_chart", "Fig. 7-17") == "Fig. 7-17"
                       else f"{lb['fig_7_18_case'].replace('_', ' ')}, D/b "
                            f"into the clay = {lb['D_over_b_fig_7_18']:.1f}")
            parts.append(f"{where}: {reading}, \u03b1 = "
                         f"{lb['alpha']:.2f} ({lb['adhesion_source']})"
                         + (", \u00d7 0.8 (Eq. 7-90)"
                            if "eq_7_90_factor" in lb else "")
                         + (f", none in the shrink-swell zone "
                            f"({lb['shrink_swell_m']:.2f} m)"
                            if lb.get("shrink_swell_m") else ""))
    chart = getattr(analysis, "adhesion_chart", "fig_7_17")
    if (result.toe_details or {}).get("governs") == "Eq. 7-90":
        chart = "fig_7_17"  # Eq. 7-90 takes Ca from Fig. 7-17 (p. 407)
    text = ("K_\u03b4 from Tables 7-6/7-7 (log-linear in the displaced volume "
            "V), \u03b4/\u03c6 from Fig. 7-9, C_F from Fig. 7-14; "
            f"{_ADHESION_CHART_TEXT.get(chart, _ADHESION_CHART_TEXT['fig_7_17'])}"
            ".")
    return text + (" " + "; ".join(parts) + "." if parts else "")


def _beta_notes(result, analysis=None) -> str:
    """Where the beta method's beta came from, for the method step."""
    if getattr(analysis, "beta_coefficients", "from_phi") == "table_7_9":
        text = ("\u03b2 from local experience where a layer gives one "
                "(Step 2a), else GEC-12 Table 7-9 by soil type, linear in "
                "\u03c6' across the type's \u03c6' range (Step 2b).")
    else:
        text = ("\u03b2 from local experience where a layer gives one "
                "(Step 2a), else (1 - sin \u03c6') tan \u03c6' "
                "(Fellenius 1991).")
    parts = [f"{lb['depth_top_m']:.1f}-{lb['depth_bottom_m']:.1f} m: "
             f"\u03b2 = {lb['beta']:.3f} ({lb['beta_source']})"
             for lb in result.layer_breakdown or [] if "beta_source" in lb]
    return text + (" " + "; ".join(parts) + "." if parts else "")


def _get_tip_layer_info(analysis) -> str:
    """Get a description of the soil layer at the pile tip."""
    tip_layer = analysis.soil.layer_at_depth(analysis._tip_depth - 0.01)
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

    # Soil type labels on a depth axis of their own at the right (written
    # left of the bars they ran over the frame and the depth numbers at
    # the printed width; GM-194)
    side = ax.secondary_yaxis('right')
    side.set_yticks(
        [(lb['depth_top_m'] + lb['depth_bottom_m']) / 2 for lb in layers],
        [lb.get('description', '') or lb['soil_type'].capitalize()
         for lb in layers], fontsize=7, fontstyle='italic')
    side.tick_params(length=0)

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

    max_qs = max(lb['skin_friction_kN'] for lb in layers)
    ax.set_xlim(0 if max_qs > 0 else -1,
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
    # the key under the axes: inside them it covered the pile tip (GM-194)
    from geotech_common.plotting import key_under_axes
    key_under_axes(fig, ax, handles=legend_handles)
    return fig
