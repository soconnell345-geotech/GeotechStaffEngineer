"""
Calculation package steps for downdrag (negative skin friction) analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation using the neutral plane method,
with every method the references give tabulated side by side.

References:
    UFC 3-220-20 (2025) — Chapter 6, §6-5.8.4, §6-7.4, Eqs 6-45 to 6-54, 6-80
    FHWA GEC-12 (2016) — §7.3.5.7, §7.3.6.1
    Fellenius, B.H. (2004/2006) — Unified design of piled foundations
    AASHTO LRFD Bridge Design Specifications, Section 10.7.3.7
"""

import math
from typing import List

import numpy as np

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Downdrag Analysis (Neutral Plane Method)"

REFERENCES = [
    'Fellenius, B.H. (2006). "Results of static loading tests on driven '
    'piles." Geotechnical News Magazine.',
    'Fellenius, B.H. (2004). "Unified design of piled foundations with '
    'emphasis on settlement analysis." ASCE GSP 125, pp. 253-275.',
    'UFC 3-220-20, 16 Jan 2025. "Geotechnical Engineering." '
    'Chapter 6: Deep Foundations.',
    'AASHTO LRFD Bridge Design Specifications, 9th Ed. (2020). '
    'Section 10.7.3.7: Downdrag.',
    'Fellenius, B.H. (1991). "Pile foundations." Chapter 13 in '
    'Foundation Engineering Handbook, 2nd Ed. Van Nostrand Reinhold.',
    'Hannigan, P.J. et al. (2016). "Design and Construction of Driven Pile '
    'Foundations." FHWA-NHI-16-009 (GEC-12), Vol I, Sections 7.3.5.7 and '
    '7.3.6.1.',
    'Siegel, T.C. et al. (2013). Neutral plane method for drag force and '
    'downdrag within LRFD (as cited by UFC 3-220-20 and GEC-12).',
    'Greenfield, M.L. and Filz, G.M. (2009). "Downdrag and Drag Load on '
    'Piles." Virginia Tech CGPR #56, Section 3.2.1 (Endo method).',
]


def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for downdrag calc package.

    Parameters
    ----------
    result : DowndragResult
        Computed results.
    analysis : DowndragAnalysis
        Analysis object holding all inputs.

    Returns
    -------
    list of InputItem
    """
    items = [
        InputItem("L", "Pile length", f"{analysis.pile_length:.2f}", "m"),
        InputItem("D", "Pile diameter", f"{analysis.pile_diameter:.3f}", "m"),
        InputItem("P", "Pile perimeter", f"{analysis.pile_perimeter:.3f}", "m"),
        InputItem("A_p", "Pile cross-sectional area",
                  f"{analysis.pile_area:.6f}", "m\u00b2"),
        InputItem("E", "Pile Young's modulus", f"{analysis.pile_E:.0f}", "kPa"),
        InputItem("\u03b3_pile", "Pile unit weight",
                  f"{analysis.pile_unit_weight:.1f}", "kN/m\u00b3"),
        InputItem("Q_dead", "Dead load at pile head",
                  f"{analysis.Q_dead:.1f}", "kN"),
    ]

    # Settlement source
    if analysis.fill_thickness > 0:
        items.append(InputItem("H_fill", "Fill thickness",
                               f"{analysis.fill_thickness:.2f}", "m"))
        items.append(InputItem("\u03b3_fill", "Fill unit weight",
                               f"{analysis.fill_unit_weight:.1f}", "kN/m\u00b3"))
    if analysis.gw_drawdown > 0:
        items.append(InputItem("\u0394GW", "Groundwater drawdown",
                               f"{analysis.gw_drawdown:.2f}", "m"))

    settlement_source = _settlement_source_description(analysis)
    items.append(InputItem("Source", "Settlement trigger", settlement_source, ""))

    if analysis.structural_capacity is not None:
        items.append(InputItem("P_r", "Structural capacity",
                               f"{analysis.structural_capacity:.1f}", "kN"))
    if analysis.allowable_settlement is not None:
        items.append(InputItem("S_allow", "Allowable settlement",
                               f"{analysis.allowable_settlement * 1000:.1f}", "mm"))

    # GWT
    items.append(InputItem("GWT", "Groundwater depth",
                           f"{analysis.soil.gwt_depth:.2f}", "m"))

    if analysis.Nt is not None:
        items.append(InputItem("N_t", "Toe bearing factor (user)",
                               f"{analysis.Nt:.1f}", ""))

    # Which method gives the headline numbers: say which was asked for.
    method = getattr(analysis, "neutral_plane_method", "auto")
    items.append(InputItem(
        "NP method", "Neutral-plane method for the headline numbers",
        "auto (UFC 3-220-20 §6-7.4 procedure)" if method == "auto"
        else method, ""))
    mob = getattr(analysis, "toe_mobilization", 1.0)
    items.append(InputItem("m_toe", "Toe mobilization where the load and "
                           "resistance curves intersect",
                           f"{mob * 100:.0f}", "%"))
    if getattr(analysis, "endo_bearing_condition", None):
        items.append(InputItem("Endo", "Endo bearing condition (CGPR #56 "
                               "Table 3.1)",
                               analysis.endo_bearing_condition, ""))
    if getattr(analysis, "skin_friction_stress", "final") == "final":
        stress = ("final (σ'v0 + Δσ), UFC 3-220-20 §6-7.4 "
                  "step 2")
    else:
        stress = "initial (σ'v0)"
    items.append(InputItem("σ'_fs", "Effective stress for skin "
                           "friction and the toe in sand", stress, ""))

    return items


def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections.

    Parameters
    ----------
    result : DowndragResult
        Computed results.
    analysis : DowndragAnalysis
        Analysis object holding all inputs.

    Returns
    -------
    list of CalcSection
    """
    r = result
    sections = []

    # ── Section 1: Soil Profile Summary ────────────────────────────
    layer_rows = []
    depth_top = 0.0
    for i, layer in enumerate(analysis.soil.layers, 1):
        depth_bot = depth_top + layer.thickness
        settling = "Yes" if layer.settling else "No"

        if layer.soil_type == "cohesive":
            strength = f"c_u = {layer.cu:.1f} kPa"
            if layer.alpha is not None:
                strength += f", \u03b1 = {layer.alpha:.2f}"
        else:
            strength = f"\u03c6 = {layer.phi:.1f}\u00b0"
            if layer.beta is not None:
                strength += f", \u03b2 = {layer.beta:.2f}"

        desc = layer.description if layer.description else layer.soil_type.capitalize()

        layer_rows.append([
            str(i),
            f"{depth_top:.1f}",
            f"{depth_bot:.1f}",
            desc,
            layer.soil_type.capitalize(),
            f"{layer.unit_weight:.1f}",
            strength,
            settling,
        ])
        depth_top = depth_bot

    layer_table = TableData(
        title="Soil Layer Definition",
        headers=["#", "Top (m)", "Bottom (m)", "Description", "Type",
                 "\u03b3 (kN/m\u00b3)", "Strength", "Settling"],
        rows=layer_rows,
    )
    sections.append(CalcSection(title="Soil Profile", items=[layer_table]))

    # ── Section 2: Neutral Plane Location ──────────────────────────
    np_items = []

    np_method = getattr(r, "neutral_plane_method", "force_equilibrium")
    np_basis = getattr(r, "neutral_plane_basis", "")
    np_warnings = list(getattr(r, "warnings", []) or [])
    if np_method == "settlement_compatibility":
        np_title = "Neutral Plane: Pile and Soil Settle Equally"
        np_equation = (
            "Load and resistance curves do not intersect "
            "(R_toe > Q_dead + W_pile + the whole shaft in negative skin "
            "friction).\n"
            "At the neutral plane depth z_np: s_soil(z_np) = s_pile(z_np), "
            "the toe force compatible with the toe penetration"
        )
        np_substitution = (
            "Soil settlement profile = pile settlement (settlement under "
            "the toe force delivered + pile compression below z_np)"
        )
        np_reference = ("UFC 3-220-20 \u00a76-5.8.4.3, Fig 6-19 (Fellenius "
                        "2021); \u00a76-7.4 step 5")
    elif np_method == "none":
        np_title = "Neutral Plane: None (pile overloaded)"
        np_equation = "Q_dead > R_toe + R_shaft(0\u2192L)"
        np_substitution = "Dead load exceeds the nominal geotechnical resistance"
        np_reference = "UFC 3-220-20 \u00a76-7.4 step 5 (limiting case)"
    elif np_method == "bearing_layer_top":
        np_title = "Neutral Plane at the Bearing Layer (end-bearing pile)"
        np_equation = "z_np = depth to the base of the settling soil"
        np_substitution = ("End-bearing pile in a stratum much stiffer than "
                           "the compressible soil")
        np_reference = "UFC 3-220-20 \u00a76-7.4 step 5"
    elif np_method == "pile_toe":
        np_title = "Neutral Plane at the Pile Toe (upper bound on drag)"
        np_equation = ("z_np = L (the whole shaft in negative skin "
                       "friction)")
        np_substitution = "Upper bound on the drag force"
        np_reference = ("GEC-12 \u00a77.3.5.7 (group settlement; "
                        "Goudreault and Fellenius 1994)")
    elif np_method == "endo":
        np_title = "Neutral Plane: Endo et al. (1969), assumed"
        np_equation = "z_np = ratio \u00d7 L  (CGPR #56 Table 3.1)"
        np_substitution = (f"Bearing condition: "
                           f"{getattr(analysis, 'endo_bearing_condition', '')}")
        np_reference = "CGPR #56 \u00a73.2.1 (Endo et al. 1969)"
    else:
        np_title = "Neutral Plane: Intersection of the Load and Resistance Curves"
        np_equation = (
            "At the neutral plane depth z_np:\n"
            "Q_dead + W_pile(0\u2192z_np) + Q_n(0\u2192z_np) "
            "= R_toe + R_s(z_np\u2192L)"
        )
        np_substitution = (
            f"Load curve at NP = resistance curve at NP\n"
            f"Q_dead + pile weight + negative skin friction = toe + "
            f"positive skin friction"
        )
        np_reference = "UFC 3-220-20 \u00a76-7.4 steps 3-5, Fig 6-26"
        mob = getattr(r, "toe_mobilization", 1.0)
        if mob < 1.0:
            np_substitution += (
                f"\nToe at {mob * 100:.0f}% of R_toe "
                f"({mob * r.toe_resistance:.1f} kN)")
            np_reference += ("; \u00a76-5.8.4.2 and GEC-12 \u00a77.3.6.1 "
                             "(toe resistance partly mobilized)")
    np_notes = (
        f"NP at {r.neutral_plane_depth:.2f} m "
        f"({r.neutral_plane_depth / r.pile_length * 100:.0f}% of pile length)"
    )
    if np_basis:
        np_notes += f". {np_basis}"
    if np_warnings:
        np_notes += (f"\n{len(np_warnings)} warning(s): see 'Neutral Plane: "
                     f"How It Was Located, and Warnings'.")
    np_items.append(CalcStep(
        title=np_title,
        equation=np_equation,
        substitution=np_substitution,
        result_name="z_np",
        result_value=f"{r.neutral_plane_depth:.2f}",
        result_unit="m",
        reference=np_reference,
        notes=np_notes,
    ))

    # Every method on the same inputs, as a first-class table.
    comparison = comparison_table(r)
    if comparison is not None:
        np_items.append(comparison)

    # Load components at NP
    np_items.append(TableData(
        title="Force Components at the Neutral Plane",
        headers=["Component", "Value (kN)", "Description"],
        rows=[
            ["Q_dead", f"{r.Q_dead:.1f}", "Applied dead load at pile head"],
            ["Pile weight to NP", f"{r.pile_weight_to_np:.1f}",
             f"\u03b3_pile \u00d7 A_p \u00d7 z_np = "
             f"{analysis.pile_unit_weight:.1f} \u00d7 "
             f"{analysis.pile_area:.6f} \u00d7 "
             f"{r.neutral_plane_depth:.2f}"],
            ["Drag load", f"{r.dragload:.1f}",
             "Negative skin friction above NP (UFC §6-7.4 step 4)"],
            ["Max pile load", f"{r.max_pile_load:.1f}",
             "Q_dead + pile weight + drag load"],
            ["Drag force", f"{r.max_pile_load - r.Q_dead:.1f}",
             "Q_np - Q_dead (UFC Eq 6-80; GEC-12 Eq 7-70)"],
        ],
    ))

    np_items.append(CalcStep(
        title="Maximum Axial Load at Neutral Plane",
        equation="Q_np = Q_dead + W_pile(0\u2192z_np) + Q_nf",
        substitution=(
            f"Q_np = {r.Q_dead:.1f} + {r.pile_weight_to_np:.1f} "
            f"+ {r.dragload:.1f}"
        ),
        result_name="Q_np",
        result_value=f"{r.max_pile_load:.1f}",
        result_unit="kN",
        reference="UFC 3-220-20 §6-7.4 step 5, Eq. 6-80",
    ))

    sections.append(CalcSection(
        title="Neutral Plane Location and Drag Force", items=np_items
    ))

    # ── How the plane was located and the warnings, stated in full ──
    # (live smoke G8: the result states how its neutral plane was located
    # and its warnings; the package must carry them as prominently as the
    # numbers, not only inside one step's notes.)
    basis_items = basis_and_warnings(r)
    if basis_items:
        sections.append(CalcSection(
            title="Neutral Plane: How It Was Located, and Warnings",
            items=basis_items
        ))

    # ── Section 3: Resistance Below NP ─────────────────────────────
    res_items = []

    res_items.append(CalcStep(
        title="Positive Shaft Resistance Below NP",
        equation="R_s = \u03a3 [f_s(z) \u00d7 P \u00d7 \u0394z]  for z > z_np",
        substitution=f"Sum of positive skin friction from z_np to pile tip",
        result_name="R_s",
        result_value=f"{r.positive_skin_friction:.1f}",
        result_unit="kN",
    ))

    res_items.append(CalcStep(
        title="Toe Bearing Resistance",
        equation="R_t = N_t \u00d7 q_t \u00d7 A_tip  (cohesionless) or  N_c \u00d7 c_u \u00d7 A_tip  (cohesive)",
        substitution="",
        result_name="R_t",
        result_value=f"{r.toe_resistance:.1f}",
        result_unit="kN",
        reference="Fellenius (1991); UFC 3-220-20",
    ))

    res_items.append(CalcStep(
        title="Total Resistance Below NP",
        equation="R_total = R_s + R_t",
        substitution=f"R_total = {r.positive_skin_friction:.1f} + {r.toe_resistance:.1f}",
        result_name="R_total",
        result_value=f"{r.total_resistance:.1f}",
        result_unit="kN",
    ))

    sections.append(CalcSection(
        title="Resistance Below Neutral Plane", items=res_items
    ))

    # ── Section 4: Settlement Breakdown ────────────────────────────
    settle_items = []

    settle_items.append(CalcStep(
        title="Elastic Shortening of Pile (above NP)",
        equation="\u03b4_e = \u03a3 [Q_avg \u00d7 \u0394z / (A_p \u00d7 E)]  for z = 0 to z_np",
        substitution=(
            f"AE = {analysis.pile_area:.6f} \u00d7 {analysis.pile_E:.0f} "
            f"= {analysis.pile_area * analysis.pile_E:,.0f} kN"
        ),
        result_name="\u03b4_e",
        result_value=f"{r.elastic_shortening * 1000:.2f}",
        result_unit="mm",
    ))

    settle_items.append(CalcStep(
        title="Toe Settlement (bearing stratum compression)",
        equation=(
            "Equivalent footing B' = D, influence depth = 3\u00d7D.\n"
            "2V:1H stress distribution below pile tip (UFC Eq 6-51)."
        ),
        substitution=f"B' = {analysis.pile_diameter:.3f} m",
        result_name="\u03b4_toe",
        result_value=f"{r.toe_settlement * 1000:.2f}",
        result_unit="mm",
        reference="UFC 3-220-20, Eqs 6-49 through 6-51",
    ))

    settle_items.append(CalcStep(
        title="Soil Settlement at Neutral Plane",
        equation=(
            "S_soil(z_np) = 1-D consolidation of the settling layers between "
            "z_np and the toe\n+ settlement of the soil below the toe under "
            "the fill / drawdown."
        ),
        substitution="Interpolated from soil settlement profile at z_np",
        result_name="S_soil(z_np)",
        result_value=f"{r.soil_settlement_at_np * 1000:.2f}",
        result_unit="mm",
        reference="UFC 3-220-20, Eq. 6-53 (clay), Eq. 6-54 (sand)",
    ))

    used = next((row for row in
                 getattr(r, "neutral_plane_comparison", []) or []
                 if row.get("used")), None)
    s_pile_np = (used["pile_settlement_at_np_mm"] if used is not None
                 else r.toe_settlement * 1000)
    s_np = max(s_pile_np, r.soil_settlement_at_np * 1000)
    settle_items.append(CalcStep(
        title="Pile Head Settlement",
        equation=(
            "\u03b4_head = s(z_np) + \u03b4_e,  s(z_np) = larger of "
            "S_soil(z_np) and the pile settlement at z_np\n"
            "(\u03b4_toe + elastic compression of the pile below z_np)"
        ),
        substitution=(
            f"\u03b4_head = max({r.soil_settlement_at_np * 1000:.2f}, "
            f"{s_pile_np:.2f}) + {r.elastic_shortening * 1000:.2f}"
        ),
        result_name="\u03b4_head",
        result_value=f"{r.pile_settlement * 1000:.2f}",
        result_unit="mm",
        reference="UFC 3-220-20 Fig 6-19 and \u00a76-5.8 item 6",
    ))

    settle_items.append(TableData(
        title="Settlement Summary",
        headers=["Component", "Value (mm)", "Description"],
        rows=[
            ["Elastic shortening", f"{r.elastic_shortening * 1000:.2f}",
             "Pile compression above NP (\u03b4_e)"],
            ["Toe settlement", f"{r.toe_settlement * 1000:.2f}",
             "Bearing stratum compression below tip (\u03b4_toe)"],
            ["Soil settlement at NP", f"{r.soil_settlement_at_np * 1000:.2f}",
             "Settling layers below NP + soil below the toe"],
            ["Settlement at NP", f"{s_np:.2f}",
             "Larger of the soil and pile settlement at NP"],
            ["Pile head settlement", f"{r.pile_settlement * 1000:.2f}",
             "Settlement at NP + \u03b4_e"],
        ],
        notes=(
            "UFC 3-220-20 Fig 6-19: the pile and the soil settle equally at "
            "the neutral plane; the pile head settles that much plus the "
            "elastic compression of the pile above it."
        ),
    ))

    sections.append(CalcSection(
        title="Settlement at the Neutral Plane", items=settle_items
    ))

    # ── Section 5: Limit State Checks ──────────────────────────────
    check_items = []

    # Structural check (UFC Eq 6-80)
    if r.structural_ok is not None:
        check_items.append(CalcStep(
            title="Structural Limit State (UFC Eq 6-80)",
            equation=(
                "LRFD Demand = 1.25 \u00d7 Q_dead + 1.10 \u00d7 (Q_np - Q_dead)\n"
                "Demand \u2264 P_r (factored structural resistance)"
            ),
            substitution=(
                f"Demand = 1.25 \u00d7 {r.Q_dead:.1f} + 1.10 \u00d7 "
                f"({r.max_pile_load:.1f} - {r.Q_dead:.1f})"
            ),
            result_name="Demand",
            result_value=f"{r.structural_demand:.1f}",
            result_unit="kN",
            reference="UFC 3-220-20, Eq. 6-80",
        ))

        check_items.append(CheckItem(
            description="Structural capacity check (UFC Eq 6-80)",
            demand=r.structural_demand,
            demand_label="LRFD demand",
            capacity=analysis.structural_capacity,
            capacity_label="P_r",
            unit="kN",
            passes=r.structural_ok,
        ))

    # Geotechnical check
    if r.geotechnical_ok is not None:
        check_items.append(CalcStep(
            title="Geotechnical Limit State",
            equation=(
                "Q_dead \u2264 R_total  (positive friction + toe)\n"
                "Dragload is NOT included — it cancels at the neutral plane."
            ),
            substitution=(
                f"Q_dead = {r.Q_dead:.1f} kN vs "
                f"R_total = {r.total_resistance:.1f} kN"
            ),
            result_name="Q_dead / R_total",
            result_value=f"{r.Q_dead / r.total_resistance:.3f}" if r.total_resistance > 0 else "N/A",
            reference="Fellenius (2004); AASHTO 10.7.3.7",
            notes="Dragload cancels in geotechnical equilibrium per Fellenius unified method.",
        ))

        check_items.append(CheckItem(
            description="Geotechnical capacity check (Q_dead vs R_total)",
            demand=r.Q_dead,
            demand_label="Q_dead",
            capacity=r.total_resistance,
            capacity_label="R_total",
            unit="kN",
            passes=r.geotechnical_ok,
        ))

    # Settlement check
    if r.settlement_ok is not None:
        controlling_settlement = max(r.pile_settlement, r.soil_settlement_at_np)
        check_items.append(CheckItem(
            description="Settlement serviceability check",
            demand=controlling_settlement * 1000,
            demand_label="Settlement",
            capacity=analysis.allowable_settlement * 1000,
            capacity_label="S_allow",
            unit="mm",
            passes=r.settlement_ok,
        ))

    if check_items:
        sections.append(CalcSection(
            title="Limit State Checks", items=check_items
        ))

    return sections


def comparison_table(result):
    """TableData of every method on the same inputs, or None when the
    result carries none (built before 2026-10-09)."""
    rows = list(getattr(result, "neutral_plane_comparison", []) or [])
    if not rows:
        return None
    table_rows = []
    for row in rows:
        name = row["name"] + (" (headline)" if row.get("used") else "")
        if row["applies"]:
            table_rows.append([
                name, row["source"],
                f"{row['neutral_plane_depth_m']:.2f}",
                f"{row['max_pile_load_kN']:.1f}",
                f"{row['dragload_kN']:.1f}",
                f"{row['drag_force_kN']:.1f}",
                f"{row['pile_settlement_mm']:.1f}",
                row.get("reason") or row["assumptions"],
            ])
        else:
            table_rows.append([name, row["source"], "-", "-", "-", "-",
                               "-", "Not applicable: " + row["reason"]])
    spread = getattr(result, "methods_range", "") or ""
    return TableData(
        title="Neutral Plane by Each Method (same inputs)",
        headers=["Method", "Reference", "z_np (m)", "Q_np (kN)",
                 "Drag load (kN)", "Drag force (kN)",
                 "Pile settlement (mm)", "Assumptions / notes"],
        rows=table_rows,
        notes=(
            (spread + " " if spread else "")
            + "Drag load = negative skin friction above the neutral plane "
            "(UFC 3-220-20 §6-7.4 step 4). Drag force = Q_np - Q_dead "
            "(UFC Eq 6-80; GEC-12 Eq 7-70); here it includes the pile weight "
            "above the plane. Pile settlement = settlement at the neutral "
            "plane + elastic compression of the pile above it (UFC Fig "
            "6-19)."),
    )


def basis_and_warnings(result) -> list:
    """Plain-text paragraphs stating how the neutral plane was located and
    every warning the result carries. Empty when the result states neither
    (a result built before 2026-10-09)."""
    method = getattr(result, "neutral_plane_method", "") or ""
    basis = getattr(result, "neutral_plane_basis", "") or ""
    warnings = list(getattr(result, "warnings", []) or [])
    if not basis and not warnings:
        return []
    items = []
    requested = getattr(result, "neutral_plane_requested", None)
    if requested:
        what = (" (UFC 3-220-20 §6-7.4 procedure)"
                if requested == "auto" else "")
        items.append(f"Neutral-plane method requested: {requested}{what}.")
    headline = getattr(result, "headline_method", "") or ""
    if headline:
        items.append(f"Headline numbers from: {headline}.")
    if basis:
        items.append("How it was located: " + basis)
    spread = getattr(result, "methods_range", "") or ""
    if spread:
        items.append(spread + " See 'Neutral Plane by Each Method'.")
    toe = getattr(result, "toe_force_mobilized", None)
    if toe is not None and method == "settlement_compatibility":
        items.append(
            f"Toe force mobilized at the neutral plane: {toe:.1f} kN "
            f"(toe resistance {result.toe_resistance:.1f} kN).")
    for w in warnings:
        items.append("WARNING: " + w)
    if not warnings:
        items.append("Warnings: none.")
    return items


def get_figures(result, analysis) -> List[FigureData]:
    """Generate downdrag analysis figures for the calc package.

    Parameters
    ----------
    result : DowndragResult
        Computed results.
    analysis : DowndragAnalysis
        Analysis object.

    Returns
    -------
    list of FigureData
    """
    figures = []

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure 1: Axial load profile Q(z) vs depth
        fig1 = _plot_axial_load(result, analysis)
        b64_1 = figure_to_base64(fig1, dpi=150)
        plt.close(fig1)
        figures.append(FigureData(
            title="Axial Load Profile Q(z)",
            image_base64=b64_1,
            caption=(
                f"Figure 1: Axial load distribution along the pile. "
                f"Neutral plane at z = {result.neutral_plane_depth:.2f} m "
                f"(Q_max = {result.max_pile_load:.1f} kN)."
            ),
            width_percent=65,
        ))

        # Figure 2: Unit skin friction profile fs(z) vs depth
        fig2 = _plot_skin_friction(result, analysis)
        b64_2 = figure_to_base64(fig2, dpi=150)
        plt.close(fig2)
        figures.append(FigureData(
            title="Unit Skin Friction Profile f_s(z)",
            image_base64=b64_2,
            caption=(
                f"Figure 2: Unit skin friction along the pile. "
                f"Negative above NP ({result.neutral_plane_depth:.2f} m), "
                f"positive below."
            ),
            width_percent=65,
        ))

        # Figure 3: Soil settlement profile vs depth
        fig3 = _plot_settlement_profile(result, analysis)
        b64_3 = figure_to_base64(fig3, dpi=150)
        plt.close(fig3)
        figures.append(FigureData(
            title="Soil Settlement Profile",
            image_base64=b64_3,
            caption=(
                f"Figure 3: Soil settlement profile along the pile depth. "
                f"Soil settlement at NP = {result.soil_settlement_at_np * 1000:.2f} mm, "
                f"pile settlement = {result.pile_settlement * 1000:.2f} mm."
            ),
            width_percent=65,
        ))

    except ImportError:
        pass  # matplotlib not available

    return figures


# ── Private helper functions ──────────────────────────────────────────────

def _settlement_source_description(analysis) -> str:
    """Build a human-readable description of the settlement trigger."""
    sources = []
    if analysis.fill_thickness > 0:
        sources.append(
            f"Fill placement ({analysis.fill_thickness:.1f} m "
            f"@ {analysis.fill_unit_weight:.0f} kN/m\u00b3)"
        )
    if analysis.gw_drawdown > 0:
        sources.append(f"GW drawdown ({analysis.gw_drawdown:.1f} m)")
    if not sources:
        return "None specified"
    return " + ".join(sources)


def _plot_axial_load(result, analysis):
    """Create axial load profile Q(z) vs depth with neutral plane annotation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z = result.z
    Q = result.axial_load
    z_np = result.neutral_plane_depth

    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot axial load curve
    ax.plot(Q, z, '-', color='#2563eb', linewidth=2, label='Q(z)')

    # Fill regions: negative friction above NP (red), positive below (green)
    z_above = z[z <= z_np]
    Q_above = Q[:len(z_above)]
    z_below = z[z >= z_np]
    Q_below = Q[len(z) - len(z_below):]

    if len(z_above) > 1:
        ax.fill_betweenx(z_above, 0, Q_above, alpha=0.12, color='#dc2626',
                         label='Dragload zone')
    if len(z_below) > 1:
        ax.fill_betweenx(z_below, 0, Q_below, alpha=0.12, color='#16a34a',
                         label='Resistance zone')

    # Neutral plane horizontal line
    Q_at_np = float(np.interp(z_np, z, Q))
    ax.axhline(y=z_np, color='#dc2626', linestyle='--', linewidth=1.5,
               alpha=0.8)
    ax.plot(Q_at_np, z_np, 'o', color='#dc2626', markersize=8, zorder=5)

    # NP annotation
    ax.annotate(
        f'Neutral Plane\nz = {z_np:.2f} m\nQ = {Q_at_np:.0f} kN',
        xy=(Q_at_np, z_np),
        xytext=(Q_at_np * 0.5, z_np - result.pile_length * 0.08),
        fontsize=8, fontweight='bold', color='#dc2626',
        arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#dc2626', alpha=0.9),
    )

    # Dead load annotation at top
    ax.plot(result.Q_dead, 0, 's', color='#2563eb', markersize=8, zorder=5)
    ax.annotate(f'Q_dead = {result.Q_dead:.0f} kN',
                xy=(result.Q_dead, 0),
                xytext=(result.Q_dead + max(Q) * 0.05, -result.pile_length * 0.03),
                fontsize=8, color='#2563eb')

    # Pile tip
    ax.axhline(y=result.pile_length, color='#666', linestyle=':',
               linewidth=1, alpha=0.6)
    ax.text(max(Q) * 0.02, result.pile_length + result.pile_length * 0.01,
            f'Pile tip (L = {result.pile_length:.1f} m)',
            fontsize=7, color='#666')

    ax.set_xlabel('Axial Load Q(z) (kN)', fontsize=10)
    ax.set_ylabel('Depth (m)', fontsize=10)
    ax.set_title('Axial Load Profile Along Pile', fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=result.pile_length * 1.05, top=-result.pile_length * 0.03)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def _plot_skin_friction(result, analysis):
    """Create unit skin friction profile fs(z) vs depth."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z = result.z
    fs = result.unit_skin_friction
    z_np = result.neutral_plane_depth

    fig, ax = plt.subplots(figsize=(6, 8))

    # Above NP: negative skin friction (plotted as negative values)
    # Below NP: positive skin friction (plotted as positive values)
    fs_signed = np.copy(fs)
    for i, zi in enumerate(z):
        if zi < z_np:
            fs_signed[i] = -fs[i]  # negative (dragload direction)

    # Plot the signed skin friction profile
    ax.plot(fs_signed, z, '-', color='#7c3aed', linewidth=2, label='f_s(z)')

    # Fill negative zone (red) and positive zone (green)
    ax.fill_betweenx(z, 0, fs_signed, where=(fs_signed < 0),
                     alpha=0.15, color='#dc2626',
                     label='Negative skin friction')
    ax.fill_betweenx(z, 0, fs_signed, where=(fs_signed >= 0),
                     alpha=0.15, color='#16a34a',
                     label='Positive skin friction')

    # Neutral plane line
    ax.axhline(y=z_np, color='#dc2626', linestyle='--', linewidth=1.5,
               alpha=0.8)
    ax.annotate(
        f'NP = {z_np:.2f} m',
        xy=(0, z_np),
        xytext=(max(abs(fs)) * 0.3, z_np - result.pile_length * 0.04),
        fontsize=8, fontweight='bold', color='#dc2626',
        arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.0),
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='#dc2626', alpha=0.9),
    )

    # Zero line
    ax.axvline(x=0, color='#333', linewidth=0.8)

    # Pile tip
    ax.axhline(y=result.pile_length, color='#666', linestyle=':',
               linewidth=1, alpha=0.6)

    ax.set_xlabel('Unit Skin Friction f_s (kPa)', fontsize=10)
    ax.set_ylabel('Depth (m)', fontsize=10)
    ax.set_title('Unit Skin Friction Profile', fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.set_ylim(bottom=result.pile_length * 1.05, top=-result.pile_length * 0.03)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def _plot_settlement_profile(result, analysis):
    """Create soil settlement profile vs depth with NP annotation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    z = result.z
    s_soil = result.soil_settlement_profile * 1000  # convert to mm
    z_np = result.neutral_plane_depth
    s_at_np = result.soil_settlement_at_np * 1000  # mm
    s_pile = result.pile_settlement * 1000  # mm

    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot soil settlement profile
    ax.plot(s_soil, z, '-', color='#d97706', linewidth=2,
            label='Soil settlement')

    # Mark settlement at NP
    ax.plot(s_at_np, z_np, 'o', color='#d97706', markersize=10, zorder=5)

    # Horizontal line at NP
    ax.axhline(y=z_np, color='#dc2626', linestyle='--', linewidth=1.5,
               alpha=0.8)

    # Pile settlement marker at NP
    ax.plot(s_pile, z_np, 's', color='#2563eb', markersize=10, zorder=5,
            label=f'Pile settlement = {s_pile:.2f} mm')

    # Annotations
    ax.annotate(
        f'NP: z = {z_np:.2f} m\n'
        f'S_soil = {s_at_np:.2f} mm\n'
        f'S_pile = {s_pile:.2f} mm',
        xy=(s_at_np, z_np),
        xytext=(max(s_soil) * 0.5 if max(s_soil) > 0 else 1.0,
                z_np - result.pile_length * 0.10),
        fontsize=8, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='#333', lw=1.0),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#333', alpha=0.9),
    )

    # Pile tip
    ax.axhline(y=result.pile_length, color='#666', linestyle=':',
               linewidth=1, alpha=0.6)
    ax.text(max(s_soil) * 0.02 if max(s_soil) > 0 else 0.1,
            result.pile_length + result.pile_length * 0.01,
            f'Pile tip (L = {result.pile_length:.1f} m)',
            fontsize=7, color='#666')

    # Surface settlement annotation
    if s_soil[0] > 0:
        ax.annotate(
            f'Surface settlement = {s_soil[0]:.1f} mm',
            xy=(s_soil[0], 0),
            xytext=(s_soil[0] + max(s_soil) * 0.1, -result.pile_length * 0.03),
            fontsize=8, color='#d97706',
        )

    ax.set_xlabel('Settlement (mm)', fontsize=10)
    ax.set_ylabel('Depth (m)', fontsize=10)
    ax.set_title('Soil Settlement Profile with Neutral Plane',
                 fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=result.pile_length * 1.05, top=-result.pile_length * 0.03)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig
