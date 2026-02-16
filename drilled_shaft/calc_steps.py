"""
Calculation package steps for drilled shaft capacity analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation.

References:
    FHWA GEC-10 (FHWA-NHI-10-016), Brown, Turner & Castelli (2010)
    O'Neill & Reese (1999), FHWA-RD-99-049
    Horvath & Kenney (1979) — rock socket side resistance
"""

import math
from typing import List

import matplotlib
matplotlib.use('Agg')

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Drilled Shaft Capacity Analysis (GEC-10)"

REFERENCES = [
    'Brown, D.A., Turner, J.P. & Castelli, R.J. (2010). "Drilled Shafts: '
    'Construction Procedures and LRFD Design Methods." '
    'FHWA-NHI-10-016, GEC-10.',
    'O\'Neill, M.W. & Reese, L.C. (1999). "Drilled Shafts: Construction '
    'Procedures and Design Methods." FHWA-RD-99-049.',
    'Horvath, R.G. & Kenney, T.C. (1979). "Shaft Resistance of Rock-Socketed '
    'Drilled Piers." Symposium on Deep Foundations, ASCE, pp. 182-214.',
    "FHWA GEC-10, Chapters 12-14: Drilled Shaft Design and Construction.",
]


# ── Atmospheric pressure constant (kPa) for alpha method ─────────
_PA = 101.325


def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for drilled shaft calc package.

    Parameters
    ----------
    result : DrillShaftResult
        Computed results.
    analysis : DrillShaftAnalysis
        Analysis object holding all inputs.

    Returns
    -------
    list of InputItem
    """
    shaft = analysis.shaft
    soil = analysis.soil

    items = [
        InputItem("D", "Shaft diameter", shaft.diameter, "m"),
        InputItem("L", "Shaft length", shaft.length, "m"),
        InputItem("P", "Shaft perimeter", f"{shaft.perimeter:.3f}", "m"),
        InputItem("A_tip", "Tip area", f"{shaft.tip_area:.4f}", "m\u00b2"),
    ]

    # Socket geometry if present
    if shaft.socket_diameter is not None:
        items.append(InputItem(
            "D_sock", "Rock socket diameter", shaft.socket_diameter, "m",
        ))
    if shaft.socket_length > 0:
        items.append(InputItem(
            "L_sock", "Rock socket length", shaft.socket_length, "m",
        ))

    # Bell
    if shaft.bell_diameter is not None:
        items.append(InputItem(
            "D_bell", "Bell diameter", shaft.bell_diameter, "m",
        ))

    # Casing
    if shaft.casing_depth > 0:
        items.append(InputItem(
            "L_case", "Permanent casing depth", shaft.casing_depth, "m",
        ))

    # Concrete
    items.append(InputItem(
        "f'c", "Concrete strength", shaft.concrete_fc / 1000, "MPa",
    ))

    # Soil layers
    items.append(InputItem(
        "Layers", "Number of soil layers", len(soil.layers), "",
    ))

    current_depth = 0.0
    for i, layer in enumerate(soil.layers, 1):
        depth_top = current_depth
        depth_bot = current_depth + layer.thickness
        desc = layer.description or layer.soil_type.capitalize()
        items.append(InputItem(
            f"Layer {i}",
            f"{desc} ({depth_top:.1f}-{depth_bot:.1f} m)",
            _layer_properties_str(layer),
            "",
        ))
        current_depth = depth_bot

    # GWT
    if soil.gwt_depth is not None:
        items.append(InputItem("GWT", "Groundwater depth", soil.gwt_depth, "m"))

    # Factor of safety
    items.append(InputItem("FS", "Factor of safety", analysis.factor_of_safety, ""))
    items.append(InputItem("Method", "Analysis method", result.method, ""))

    return items


def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections for drilled shaft analysis.

    Parameters
    ----------
    result : DrillShaftResult
        Computed results.
    analysis : DrillShaftAnalysis
        Analysis object holding all inputs.

    Returns
    -------
    list of CalcSection
    """
    r = result
    shaft = analysis.shaft
    soil = analysis.soil
    sections = []

    # ── Section 1: Shaft Geometry ────────────────────────────────
    geom_steps = []
    geom_steps.append(CalcStep(
        title="Shaft Perimeter",
        equation="P = \u03c0 \u00d7 D",
        substitution=f"P = \u03c0 \u00d7 {shaft.diameter:.3f}",
        result_name="P",
        result_value=f"{shaft.perimeter:.3f}",
        result_unit="m",
    ))
    geom_steps.append(CalcStep(
        title="Shaft Tip Area",
        equation="A_tip = \u03c0 \u00d7 D\u00b2 / 4"
        if shaft.bell_diameter is None
        else "A_tip = \u03c0 \u00d7 D_bell\u00b2 / 4",
        substitution=(
            f"A_tip = \u03c0 \u00d7 {shaft.diameter:.3f}\u00b2 / 4"
            if shaft.bell_diameter is None
            else f"A_tip = \u03c0 \u00d7 {shaft.bell_diameter:.3f}\u00b2 / 4"
        ),
        result_name="A_tip",
        result_value=f"{shaft.tip_area:.4f}",
        result_unit="m\u00b2",
        notes="Bell diameter used for tip area" if shaft.bell_diameter else "",
    ))
    geom_steps.append(CalcStep(
        title="Length-to-Diameter Ratio",
        equation="L/D = L / D",
        substitution=f"L/D = {shaft.length:.2f} / {shaft.diameter:.3f}",
        result_name="L/D",
        result_value=f"{shaft.length / shaft.diameter:.1f}",
    ))

    # Exclusion zones
    top_excl = max(1.5, shaft.casing_depth)
    geom_steps.append(CalcStep(
        title="Top Exclusion Zone",
        equation="z_excl_top = max(1.5 m, casing depth)",
        substitution=f"z_excl_top = max(1.5, {shaft.casing_depth:.2f})",
        result_name="z_excl_top",
        result_value=f"{top_excl:.2f}",
        result_unit="m",
        reference="GEC-10 Section 13.3.3: top 1.5 m excluded from side resistance",
    ))
    geom_steps.append(CalcStep(
        title="Bottom Exclusion Zone (cohesive only)",
        equation="z_excl_bot = L - D (bottom 1\u00d7D excluded for cohesive layers)",
        substitution=f"z_excl_bot = {shaft.length:.2f} - {shaft.diameter:.3f}",
        result_name="z_excl_bot",
        result_value=f"{shaft.length - shaft.diameter:.2f}",
        result_unit="m",
        reference="GEC-10 Section 13.3.3: bottom 1D excluded for cohesive soils",
    ))

    sections.append(CalcSection(title="Shaft Geometry & Exclusion Zones", items=geom_steps))

    # ── Section 2: Side Resistance Methods ───────────────────────
    method_steps = []

    # Alpha method explanation
    has_clay = any(
        lb["soil_type"] == "cohesive" for lb in (r.layer_breakdown or [])
    )
    has_sand = any(
        lb["soil_type"] == "cohesionless" for lb in (r.layer_breakdown or [])
    )
    has_rock = any(
        lb["soil_type"] == "rock" for lb in (r.layer_breakdown or [])
    )

    if has_clay:
        method_steps.append(CalcStep(
            title="Alpha Method (Cohesive Soils)",
            equation=(
                "\u03b1 = 0.55 for c_u/p_a \u2264 1.5\n"
                "\u03b1 = 0.55 - 0.1\u00d7(c_u/p_a - 1.5) for c_u/p_a > 1.5\n"
                "    (minimum \u03b1 = 0.35)\n"
                "f_s = \u03b1 \u00d7 c_u\n"
                "Q_s = f_s \u00d7 P \u00d7 \u0394z"
            ),
            substitution="",
            result_name="",
            result_value="",
            reference="GEC-10 Section 13.3.3.2; O'Neill & Reese (1999)",
        ))

    if has_sand:
        method_steps.append(CalcStep(
            title="Beta Method (Cohesionless Soils)",
            equation=(
                "\u03b2 = 1.5 - 0.245\u00d7\u221az, clamped to [0.25, 1.2]\n"
                "f_s = \u03b2 \u00d7 \u03c3'_v, capped at 200 kPa\n"
                "Q_s = f_s \u00d7 P \u00d7 \u0394z"
            ),
            substitution="",
            result_name="",
            result_value="",
            reference="GEC-10 Section 13.3.3.3; Brown et al. (2010)",
        ))

    if has_rock:
        method_steps.append(CalcStep(
            title="Rock Socket Method",
            equation=(
                "f_s = C \u00d7 \u03b1_E \u00d7 \u221aq_u\n"
                "Q_s = f_s \u00d7 P_socket \u00d7 \u0394z\n"
                "where C = roughness factor, \u03b1_E = rock mass reduction"
            ),
            substitution="",
            result_name="",
            result_value="",
            reference="Horvath & Kenney (1979); GEC-10 Section 13.3.3.4",
        ))

    if method_steps:
        sections.append(CalcSection(
            title="Side Resistance Methods", items=method_steps,
        ))

    # ── Section 3: Per-Layer Side Resistance ─────────────────────
    layer_items = []

    if r.layer_breakdown:
        # Build per-layer calculation detail
        for i, lb in enumerate(r.layer_breakdown, 1):
            z_top = lb["depth_top_m"]
            z_bot = lb["depth_bottom_m"]
            soil_type = lb["soil_type"]
            method_used = lb["method"]
            Qs_layer = lb["side_resistance_kN"]
            desc = lb.get("description", "") or soil_type.capitalize()

            if method_used == "excluded":
                layer_items.append(CalcStep(
                    title=f"Layer {i}: {desc} ({z_top:.1f}-{z_bot:.1f} m)",
                    equation="Excluded from side resistance (within exclusion zone)",
                    substitution="",
                    result_name="Q_s",
                    result_value="0.0",
                    result_unit="kN",
                    notes="Layer within top or bottom exclusion zone",
                ))
                continue

            eff_top = lb.get("effective_top_m", z_top)
            eff_bot = lb.get("effective_bottom_m", z_bot)
            eff_thickness = eff_bot - eff_top
            fs = lb.get("fs_kPa", 0.0)
            sigma_v = lb.get("sigma_v_kPa", 0.0)

            if soil_type == "cohesive":
                layer_items.append(CalcStep(
                    title=f"Layer {i}: {desc} ({z_top:.1f}-{z_bot:.1f} m) -- Alpha Method",
                    equation=(
                        f"f_s = \u03b1 \u00d7 c_u = {method_used}\n"
                        f"Q_s = f_s \u00d7 P \u00d7 \u0394z"
                    ),
                    substitution=(
                        f"f_s = {fs:.1f} kPa\n"
                        f"Q_s = {fs:.1f} \u00d7 {shaft.perimeter:.3f} "
                        f"\u00d7 {eff_thickness:.2f}"
                    ),
                    result_name="Q_s",
                    result_value=f"{Qs_layer:,.1f}",
                    result_unit="kN",
                    notes=(
                        f"Effective zone: {eff_top:.1f}-{eff_bot:.1f} m, "
                        f"\u03c3'_v = {sigma_v:.1f} kPa"
                    ),
                ))

            elif soil_type == "cohesionless":
                layer_items.append(CalcStep(
                    title=f"Layer {i}: {desc} ({z_top:.1f}-{z_bot:.1f} m) -- Beta Method",
                    equation=(
                        f"f_s = \u03b2 \u00d7 \u03c3'_v, {method_used}\n"
                        f"Q_s = f_s \u00d7 P \u00d7 \u0394z"
                    ),
                    substitution=(
                        f"f_s = {fs:.1f} kPa "
                        f"(\u03c3'_v = {sigma_v:.1f} kPa)\n"
                        f"Q_s = {fs:.1f} \u00d7 {shaft.perimeter:.3f} "
                        f"\u00d7 {eff_thickness:.2f}"
                    ),
                    result_name="Q_s",
                    result_value=f"{Qs_layer:,.1f}",
                    result_unit="kN",
                    notes=f"Effective zone: {eff_top:.1f}-{eff_bot:.1f} m",
                ))

            else:  # rock
                socket_perim = shaft.socket_perimeter
                layer_items.append(CalcStep(
                    title=f"Layer {i}: {desc} ({z_top:.1f}-{z_bot:.1f} m) -- Rock Socket",
                    equation=(
                        f"f_s = C \u00d7 \u03b1_E \u00d7 \u221aq_u\n"
                        f"Q_s = f_s \u00d7 P_socket \u00d7 \u0394z"
                    ),
                    substitution=(
                        f"f_s = {fs:.1f} kPa\n"
                        f"Q_s = {fs:.1f} \u00d7 {socket_perim:.3f} "
                        f"\u00d7 {eff_thickness:.2f}"
                    ),
                    result_name="Q_s",
                    result_value=f"{Qs_layer:,.1f}",
                    result_unit="kN",
                    notes=f"Effective zone: {eff_top:.1f}-{eff_bot:.1f} m",
                ))

        # Summary table for all layers
        table_rows = []
        for i, lb in enumerate(r.layer_breakdown, 1):
            desc = lb.get("description", "") or lb["soil_type"].capitalize()
            table_rows.append([
                str(i),
                f"{lb['depth_top_m']:.1f}-{lb['depth_bottom_m']:.1f}",
                desc,
                lb["soil_type"],
                lb["method"],
                f"{lb.get('fs_kPa', 0.0):.1f}",
                f"{lb['side_resistance_kN']:,.1f}",
            ])

        layer_items.append(TableData(
            title="Per-Layer Side Resistance Summary",
            headers=[
                "Layer", "Depth (m)", "Description", "Soil Type",
                "Method", "f_s (kPa)", "Q_s (kN)",
            ],
            rows=table_rows,
        ))

    # Total side resistance
    layer_items.append(CalcStep(
        title="Total Side Resistance",
        equation="Q_skin = \u03a3 Q_s,i",
        substitution=_sum_layers_substitution(r.layer_breakdown),
        result_name="Q_skin",
        result_value=f"{r.Q_skin:,.1f}",
        result_unit="kN",
    ))

    # Side resistance by soil type
    type_rows = []
    if r.Q_side_clay > 0:
        type_rows.append(["Cohesive (clay/silt)", f"{r.Q_side_clay:,.1f}",
                          _pct(r.Q_side_clay, r.Q_skin)])
    if r.Q_side_sand > 0:
        type_rows.append(["Cohesionless (sand/gravel)", f"{r.Q_side_sand:,.1f}",
                          _pct(r.Q_side_sand, r.Q_skin)])
    if r.Q_side_rock > 0:
        type_rows.append(["Rock socket", f"{r.Q_side_rock:,.1f}",
                          _pct(r.Q_side_rock, r.Q_skin)])
    type_rows.append(["Total Q_skin", f"{r.Q_skin:,.1f}", "100%"])

    layer_items.append(TableData(
        title="Side Resistance by Soil Type",
        headers=["Soil Type", "Q_s (kN)", "% of Q_skin"],
        rows=type_rows,
    ))

    sections.append(CalcSection(
        title="Per-Layer Side Resistance", items=layer_items,
    ))

    # ── Section 4: End Bearing ───────────────────────────────────
    eb_items = []

    # Determine tip layer
    tip_layer = soil.layer_at_depth(shaft.length - 0.01)
    L_over_D = shaft.length / shaft.diameter

    if tip_layer.soil_type == "cohesive":
        Nc = min(6.0 + L_over_D, 9.0)
        qb = Nc * tip_layer.cu
        eb_items.append(CalcStep(
            title="Bearing Capacity Factor N_c (Cohesive Tip)",
            equation="N_c = min(6.0 + L/D, 9.0)",
            substitution=f"N_c = min(6.0 + {L_over_D:.1f}, 9.0)",
            result_name="N_c",
            result_value=f"{Nc:.2f}",
            reference="GEC-10 Section 13.3.4.2; O'Neill & Reese (1999)",
        ))
        eb_items.append(CalcStep(
            title="Unit End Bearing",
            equation="q_b = N_c \u00d7 c_u",
            substitution=f"q_b = {Nc:.2f} \u00d7 {tip_layer.cu:.1f}",
            result_name="q_b",
            result_value=f"{qb:,.1f}",
            result_unit="kPa",
        ))
        eb_items.append(CalcStep(
            title="End Bearing Capacity",
            equation="Q_tip = q_b \u00d7 A_tip",
            substitution=f"Q_tip = {qb:,.1f} \u00d7 {shaft.tip_area:.4f}",
            result_name="Q_tip",
            result_value=f"{r.Q_tip:,.1f}",
            result_unit="kN",
        ))

    elif tip_layer.soil_type == "cohesionless":
        N60 = tip_layer.N60 if tip_layer.N60 > 0 else 15.0
        N60_capped = min(N60, 50.0)
        qb = 57.5 * N60_capped
        large_dia_note = ""
        if shaft.diameter > 1.27:
            reduction = 1.27 / shaft.diameter
            qb_reduced = qb * reduction
            large_dia_note = (
                f"Large-diameter reduction: q_b = {qb:,.1f} \u00d7 "
                f"(1.27/{shaft.diameter:.3f}) = {qb_reduced:,.1f} kPa"
            )
            qb = qb_reduced

        eb_items.append(CalcStep(
            title="Unit End Bearing (Cohesionless Tip)",
            equation="q_b = 57.5 \u00d7 N_60 (N_60 capped at 50)",
            substitution=f"q_b = 57.5 \u00d7 {N60_capped:.0f}",
            result_name="q_b",
            result_value=f"{qb:,.1f}",
            result_unit="kPa",
            reference="GEC-10 Section 13.3.4.3; O'Neill & Reese (1999)",
            notes=large_dia_note,
        ))
        eb_items.append(CalcStep(
            title="End Bearing Capacity",
            equation="Q_tip = q_b \u00d7 A_tip",
            substitution=f"Q_tip = {qb:,.1f} \u00d7 {shaft.tip_area:.4f}",
            result_name="Q_tip",
            result_value=f"{r.Q_tip:,.1f}",
            result_unit="kN",
        ))

    else:  # rock
        RQD = tip_layer.RQD
        qu = tip_layer.qu
        # Replicate RQD-based multiplier logic from end_bearing.py
        if RQD >= 70:
            mult = 2.5
        elif RQD >= 50:
            mult = 1.5 + (RQD - 50) / 20 * 1.0
        elif RQD >= 25:
            mult = 0.8 + (RQD - 25) / 25 * 0.7
        else:
            mult = 0.4 + RQD / 25 * 0.4
        qb = mult * qu

        eb_items.append(CalcStep(
            title="Unit End Bearing (Rock Tip)",
            equation=f"q_b = {mult:.2f} \u00d7 q_u (RQD = {RQD:.0f}%)",
            substitution=f"q_b = {mult:.2f} \u00d7 {qu:,.1f}",
            result_name="q_b",
            result_value=f"{qb:,.1f}",
            result_unit="kPa",
            reference="GEC-10 Section 13.3.4.4",
            notes=_rqd_note(RQD),
        ))
        eb_items.append(CalcStep(
            title="End Bearing Capacity",
            equation="Q_tip = q_b \u00d7 A_tip",
            substitution=f"Q_tip = {qb:,.1f} \u00d7 {shaft.tip_area:.4f}",
            result_name="Q_tip",
            result_value=f"{r.Q_tip:,.1f}",
            result_unit="kN",
        ))

    # Effective stress at tip
    eb_items.append(CalcStep(
        title="Effective Stress at Shaft Tip",
        equation="\u03c3'_v,tip = \u03a3(\u03b3'_i \u00d7 \u0394z_i)",
        substitution="",
        result_name="\u03c3'_v,tip",
        result_value=f"{r.sigma_v_tip:,.1f}",
        result_unit="kPa",
    ))

    sections.append(CalcSection(title="End Bearing", items=eb_items))

    # ── Section 5: Ultimate & Allowable Capacity ─────────────────
    cap_items = []

    cap_items.append(CalcStep(
        title="Ultimate Axial Capacity",
        equation="Q_ult = Q_skin + Q_tip",
        substitution=f"Q_ult = {r.Q_skin:,.1f} + {r.Q_tip:,.1f}",
        result_name="Q_ult",
        result_value=f"{r.Q_ultimate:,.1f}",
        result_unit="kN",
        reference="GEC-10 Eq. 13-1",
    ))

    # Capacity breakdown
    pct_skin = _pct(r.Q_skin, r.Q_ultimate)
    pct_tip = _pct(r.Q_tip, r.Q_ultimate)

    cap_items.append(TableData(
        title="Capacity Breakdown",
        headers=["Component", "Value (kN)", "% of Q_ult"],
        rows=[
            ["Side resistance (Q_skin)", f"{r.Q_skin:,.1f}", pct_skin],
            ["  Clay", f"{r.Q_side_clay:,.1f}",
             _pct(r.Q_side_clay, r.Q_ultimate)],
            ["  Sand", f"{r.Q_side_sand:,.1f}",
             _pct(r.Q_side_sand, r.Q_ultimate)],
            ["  Rock", f"{r.Q_side_rock:,.1f}",
             _pct(r.Q_side_rock, r.Q_ultimate)],
            ["End bearing (Q_tip)", f"{r.Q_tip:,.1f}", pct_tip],
            ["Total Q_ult", f"{r.Q_ultimate:,.1f}", "100%"],
        ],
    ))

    cap_items.append(CalcStep(
        title="Allowable Capacity",
        equation="Q_all = Q_ult / FS",
        substitution=f"Q_all = {r.Q_ultimate:,.1f} / {r.factor_of_safety:.1f}",
        result_name="Q_all",
        result_value=f"{r.Q_allowable:,.1f}",
        result_unit="kN",
    ))

    sections.append(CalcSection(
        title="Ultimate & Allowable Capacity", items=cap_items,
    ))

    return sections


def get_figures(result, analysis) -> List[FigureData]:
    """Generate figures for the drilled shaft calc package.

    Parameters
    ----------
    result : DrillShaftResult
        Computed results.
    analysis : DrillShaftAnalysis
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
            title="Drilled Shaft Capacity Breakdown",
            image_base64=b64,
            caption=(
                f"Figure 1: Capacity components -- "
                f"Q_skin = {result.Q_skin:,.1f} kN, "
                f"Q_tip = {result.Q_tip:,.1f} kN, "
                f"Q_ult = {result.Q_ultimate:,.1f} kN."
            ),
            width_percent=70,
        ))
    except ImportError:
        pass

    # Figure 2: Per-layer side resistance depth profile
    try:
        if result.layer_breakdown:
            fig = _plot_layer_profile(result, analysis)
            b64 = figure_to_base64(fig, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig)
            figures.append(FigureData(
                title="Side Resistance vs. Depth",
                image_base64=b64,
                caption=(
                    f"Figure 2: Per-layer side resistance profile -- "
                    f"total Q_skin = {result.Q_skin:,.1f} kN "
                    f"across {len(result.layer_breakdown)} layer(s)."
                ),
                width_percent=70,
            ))
    except ImportError:
        pass

    return figures


# ── Helper functions ──────────────────────────────────────────────

def _layer_properties_str(layer) -> str:
    """Format key layer properties as a compact string."""
    parts = [f"\u03b3 = {layer.unit_weight:.1f} kN/m\u00b3"]
    if layer.soil_type == "cohesive":
        parts.append(f"c_u = {layer.cu:.0f} kPa")
    elif layer.soil_type == "cohesionless":
        parts.append(f"\u03c6 = {layer.phi:.0f}\u00b0")
        if layer.N60 > 0:
            parts.append(f"N_60 = {layer.N60:.0f}")
    else:  # rock
        parts.append(f"q_u = {layer.qu:,.0f} kPa")
        if layer.RQD < 100:
            parts.append(f"RQD = {layer.RQD:.0f}%")
    return ", ".join(parts)


def _sum_layers_substitution(layer_breakdown) -> str:
    """Build a summation substitution string from layer breakdown."""
    if not layer_breakdown:
        return "No layers"
    contributing = [
        lb for lb in layer_breakdown if lb["side_resistance_kN"] > 0
    ]
    if not contributing:
        return "All layers excluded"
    if len(contributing) <= 6:
        parts = [f"{lb['side_resistance_kN']:,.1f}" for lb in contributing]
        return "Q_skin = " + " + ".join(parts)
    # Truncate if many layers
    parts = [f"{lb['side_resistance_kN']:,.1f}" for lb in contributing[:5]]
    return "Q_skin = " + " + ".join(parts) + " + ..."


def _pct(component: float, total: float) -> str:
    """Format a percentage string."""
    if total > 0:
        return f"{100 * component / total:.0f}%"
    return "0%"


def _rqd_note(RQD: float) -> str:
    """Return a descriptive note for RQD-based bearing factor."""
    if RQD >= 70:
        return "Intact rock (RQD >= 70%): multiplier = 2.5"
    elif RQD >= 50:
        return f"Moderately fractured (RQD = {RQD:.0f}%): reduced multiplier"
    elif RQD >= 25:
        return f"Fractured rock (RQD = {RQD:.0f}%): reduced multiplier"
    else:
        return f"Very fractured (RQD = {RQD:.0f}%): significantly reduced multiplier"


def _plot_capacity_breakdown(result):
    """Create a stacked/grouped bar chart of capacity components."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Bars: side resistance components stacked, then end bearing
    # Left group: side resistance breakdown; Right: end bearing
    components = []
    values = []
    colors = []

    if result.Q_side_clay > 0:
        components.append("Q_s (Clay)")
        values.append(result.Q_side_clay)
        colors.append("#2563eb")

    if result.Q_side_sand > 0:
        components.append("Q_s (Sand)")
        values.append(result.Q_side_sand)
        colors.append("#16a34a")

    if result.Q_side_rock > 0:
        components.append("Q_s (Rock)")
        values.append(result.Q_side_rock)
        colors.append("#7c3aed")

    components.append("Q_tip")
    values.append(result.Q_tip)
    colors.append("#d97706")

    bars = ax.barh(components, values, color=colors, edgecolor="#333",
                   height=0.5)

    # Value labels on bars
    max_val = max(values) if values else 1.0
    for bar, val in zip(bars, values):
        if val > 0:
            pct = 100 * val / result.Q_ultimate if result.Q_ultimate > 0 else 0
            ax.text(
                bar.get_width() + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f} kN ({pct:.0f}%)",
                va="center", fontsize=9,
            )

    # Ultimate capacity line
    ax.axvline(x=result.Q_ultimate, color="#dc2626", linestyle="--",
               linewidth=1.5, label=f"Q_ult = {result.Q_ultimate:,.0f} kN")

    ax.set_xlabel("Capacity (kN)", fontsize=10)
    ax.set_title(
        f"Drilled Shaft Capacity Breakdown\n"
        f"D = {result.shaft_diameter:.2f} m, L = {result.shaft_length:.1f} m",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(0, result.Q_ultimate * 1.25 if result.Q_ultimate > 0 else 1)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def _plot_layer_profile(result, analysis):
    """Create a horizontal bar chart of side resistance per layer vs depth."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(7, 5.5))

    soil_colors = {
        "cohesive": "#2563eb",
        "cohesionless": "#16a34a",
        "rock": "#7c3aed",
    }

    layer_data = result.layer_breakdown or []
    max_qs = max(
        (lb["side_resistance_kN"] for lb in layer_data), default=1.0,
    )
    if max_qs <= 0:
        max_qs = 1.0

    for lb in layer_data:
        z_top = lb["depth_top_m"]
        z_bot = lb["depth_bottom_m"]
        z_mid = (z_top + z_bot) / 2
        thickness = z_bot - z_top
        Qs = lb["side_resistance_kN"]
        soil_type = lb["soil_type"]
        color = soil_colors.get(soil_type, "#6b7280")

        bar = ax.barh(
            z_mid, Qs, height=thickness * 0.85,
            color=color, edgecolor="#333", alpha=0.85,
        )

        # Label on bar if enough space
        if Qs > max_qs * 0.05:
            ax.text(
                Qs + max_qs * 0.02, z_mid,
                f"{Qs:,.0f} kN",
                va="center", fontsize=8,
            )

    # Shaft length indicator
    ax.axhline(y=analysis.shaft.length, color="#dc2626", linestyle="--",
               linewidth=1.5, label=f"Shaft tip (L = {analysis.shaft.length:.1f} m)")

    # Exclusion zone shading (top)
    top_excl = max(1.5, analysis.shaft.casing_depth)
    ax.axhspan(0, top_excl, alpha=0.1, color="#ef4444",
               label=f"Top exclusion ({top_excl:.1f} m)")

    # Invert y-axis so depth increases downward
    ax.invert_yaxis()

    ax.set_xlabel("Side Resistance, Q_s (kN)", fontsize=10)
    ax.set_ylabel("Depth (m)", fontsize=10)
    ax.set_title(
        f"Per-Layer Side Resistance Profile\n"
        f"Total Q_skin = {result.Q_skin:,.0f} kN",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(0, max_qs * 1.3)
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(labelsize=9)

    # Legend for soil types present
    legend_handles = []
    seen_types = set()
    for lb in layer_data:
        st = lb["soil_type"]
        if st not in seen_types:
            seen_types.add(st)
            legend_handles.append(mpatches.Patch(
                color=soil_colors.get(st, "#6b7280"),
                label=st.capitalize(),
            ))
    # Add shaft tip and exclusion zone to legend
    legend_handles.append(plt.Line2D(
        [0], [0], color="#dc2626", linestyle="--", linewidth=1.5,
        label=f"Shaft tip ({analysis.shaft.length:.1f} m)",
    ))
    legend_handles.append(mpatches.Patch(
        color="#ef4444", alpha=0.15,
        label=f"Top exclusion ({top_excl:.1f} m)",
    ))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    return fig
