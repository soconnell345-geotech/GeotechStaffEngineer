"""
Calculation package steps for wave equation analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation (WEAP-style output).

Handles two result types:
    - BearingGraphResult: capacity vs blow count at fixed pile tip depth
    - DrivabilityResult:  blow count vs depth profile for pile installation

The `analysis` parameter should be a dict with:
    - "hammer": Hammer
    - "cushion": Cushion
    - "pile": PileModel (for bearing graph) or pile params dict (for drivability)
    - "soil": SoilSetup (for bearing graph) or omitted (for drivability)
    - "helmet_weight": float (kN)
    - "type": "bearing_graph" or "drivability"

References:
    Smith (1960) — original wave equation formulation
    Goble & Rausche (1976) — WEAP methodology
    FHWA GEC-12 — Design and Construction of Driven Pile Foundations
"""

import math
from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64
from wave_equation.hammer import diesel_warning

DISPLAY_NAME = "Wave Equation Analysis (Smith 1-D Model)"

REFERENCES = [
    'Smith, E.A.L. (1960). "Bearing Capacity of Piles Determined by Means '
    'of a Pile Driving Analysis by the Wave Equation." ASCE, Vol. 86, No. EM4.',
    'Goble, G.G. & Rausche, F. (1976). "Wave Equation Analysis of Pile '
    'Driving — WEAP Program." FHWA IP-76-14.',
    'FHWA GEC-12 (FHWA-NHI-16-009/010): Design and Construction of Driven '
    'Pile Foundations, Vol 2 Chapter 12 — Wave Equation Analysis; Vol 1 '
    'Chapter 8 — driving stress limits.',
    'UFC 3-220-20 (2025): Foundations and Earth Structures, Sec. 6-9.2 — '
    'Wave Mechanics Basics (quake, Smith and Smith viscous damping).',
]


# ─── Helpers ─────────────────────────────────────────────────────────

def _is_bearing_graph(result) -> bool:
    """Return True if result is a BearingGraphResult."""
    return hasattr(result, 'R_values') and hasattr(result, 'blow_counts')


def _is_drivability(result) -> bool:
    """Return True if result is a DrivabilityResult."""
    return hasattr(result, 'points') and hasattr(result, 'can_drive')


# ─── Public API ──────────────────────────────────────────────────────

def _used(analysis, key):
    """An input the bearing graph run used: the analysis dict's, else the
    default of ``generate_bearing_graph``, which the run then took
    (GM-141: the steps had a default of their own)."""
    if key in analysis:
        return analysis[key]
    import inspect
    from wave_equation.bearing_graph import generate_bearing_graph
    return inspect.signature(generate_bearing_graph).parameters[key].default


def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for wave equation calc package.

    Parameters
    ----------
    result : BearingGraphResult or DrivabilityResult
        Computed results.
    analysis : dict
        Analysis setup — see module docstring for expected keys.

    Returns
    -------
    list of InputItem
    """
    items = []

    # ── Hammer ──
    hammer = analysis.get("hammer")
    if hammer is not None:
        items.extend([
            InputItem("Hammer", "Hammer model", hammer.name, ""),
            InputItem("W_ram", "Ram weight", f"{hammer.ram_weight:.1f}", "kN"),
            InputItem("h", "Stroke", f"{hammer.stroke:.3f}", "m"),
            InputItem("E_rated", "Rated energy", f"{hammer.energy:.1f}", "kN-m"),
            InputItem("\u03b7", "Hammer efficiency", f"{hammer.efficiency:.2f}", ""),
            InputItem("v_impact", "Impact velocity",
                      f"{hammer.impact_velocity:.2f}", "m/s"),
            InputItem("Type", "Hammer type", hammer.hammer_type.replace("_", " ").title(), ""),
        ])

    # ── Cushion ──
    cushion = analysis.get("cushion")
    if cushion is not None:
        items.extend([
            InputItem("k_cushion", "Cushion stiffness",
                      f"{cushion.stiffness:,.0f}", "kN/m"),
            InputItem("COR", "Coefficient of restitution",
                      f"{cushion.cor:.2f}", ""),
        ])
        if cushion.material:
            items.append(InputItem("Material", "Cushion material",
                                   cushion.material, ""))
        if cushion.thickness is not None:
            items.append(InputItem("t_cushion", "Cushion thickness",
                                   f"{cushion.thickness * 1000:.0f}", "mm"))

    # ── Helmet ──
    helmet_wt = _used(analysis, "helmet_weight")
    items.append(InputItem("W_helmet", "Helmet weight",
                           f"{helmet_wt:.1f}", "kN"))

    # ── Pile ──
    pile = analysis.get("pile")
    if pile is not None:
        # E as the run took it (GM-141); worked back from the segment's
        # wave speed only when the caller did not say
        E = analysis.get("pile_E")
        if E is None and pile.n_segments > 0:
            E = (pile.wave_speeds[0]**2 * pile.masses[0]
                 / (pile.segment_areas[0] * pile.segment_lengths[0] * 1000))
        items.extend([
            InputItem("L_pile", "Pile length", f"{pile.total_length:.2f}", "m"),
            InputItem("A_pile", "Pile area",
                      f"{pile.segment_areas[0]:.6f}", "m\u00b2"),
            InputItem("E_pile", "Pile elastic modulus",
                      f"{E:,.0f}" if E is not None else "N/A", "kPa"),
            InputItem("N_seg", "Number of segments",
                      str(pile.n_segments), ""),
            InputItem("Z", "Pile impedance", f"{pile.impedance:.1f}", "kN-s/m"),
        ])
        if analysis.get("pile_unit_weight") is not None:
            items.append(InputItem("\u03b3_pile", "Pile unit weight",
                                   f"{analysis['pile_unit_weight']:.1f}",
                                   "kN/m\u00b3"))

    # ── Pile params for drivability (no PileModel yet) ──
    pile_params = analysis.get("pile_params")
    if pile_params is not None:
        items.extend([
            InputItem("A_pile", "Pile area",
                      f"{pile_params.get('area', 0):.6f}", "m\u00b2"),
            InputItem("E_pile", "Pile elastic modulus",
                      f"{pile_params.get('E', 0):,.0f}", "kPa"),
            InputItem("\u03b3_pile", "Pile unit weight",
                      f"{pile_params.get('unit_weight', 77.3):.1f}", "kN/m\u00b3"),
        ])

    # ── Soil ──
    soil = analysis.get("soil")
    if soil is None and analysis.get("quake_side") is not None:
        # a bearing graph run: the soil model's inputs as the run took them
        items.extend([
            InputItem("f_skin", "Skin friction fraction",
                      f"{_used(analysis, 'skin_fraction'):.2f}", ""),
            InputItem("Q_side", "Side quake",
                      f"{_used(analysis, 'quake_side') * 1000:.1f}", "mm"),
            InputItem("Q_toe", "Toe quake",
                      f"{_used(analysis, 'quake_toe') * 1000:.1f}", "mm"),
            InputItem("J_side", "Side Smith damping",
                      f"{_used(analysis, 'damping_side'):.2f}", "s/m"),
            InputItem("J_toe", "Toe Smith damping",
                      f"{_used(analysis, 'damping_toe'):.2f}", "s/m"),
            InputItem("Damping model", "Smith damping variant",
                      _used(analysis, 'damping_model'), ""),
            InputItem("R_ult", "Ultimate resistances run",
                      f"{_used(analysis, 'R_min'):,.0f} to "
                      f"{_used(analysis, 'R_max'):,.0f} "
                      f"by {_used(analysis, 'R_step'):,.0f}", "kN"),
        ])
    if soil is not None:
        items.extend([
            InputItem("R_ult", "Ultimate resistance (total)",
                      f"{soil.R_ultimate:,.0f}", "kN"),
            InputItem("f_skin", "Skin friction fraction",
                      f"{soil.skin_fraction:.2f}", ""),
            InputItem("Q_side", "Side quake",
                      f"{soil.quake_side * 1000:.1f}", "mm"),
            InputItem("Q_toe", "Toe quake",
                      f"{soil.quake_toe * 1000:.1f}", "mm"),
            InputItem("J_side", "Side Smith damping",
                      f"{soil.damping_side:.2f}", "s/m"),
            InputItem("J_toe", "Toe Smith damping",
                      f"{soil.damping_toe:.2f}", "s/m"),
            InputItem("Damping model", "Smith damping variant",
                      getattr(soil, "damping_model", "smith"), ""),
        ])

    # ── Analysis type label ──
    if _is_bearing_graph(result):
        items.insert(0, InputItem("Analysis", "Analysis type",
                                  "Bearing Graph", ""))
    elif _is_drivability(result):
        items.insert(0, InputItem("Analysis", "Analysis type",
                                  "Drivability Study", ""))

    return items


def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections.

    Parameters
    ----------
    result : BearingGraphResult or DrivabilityResult
        Computed results.
    analysis : dict
        Analysis setup.

    Returns
    -------
    list of CalcSection
    """
    sections = []

    # ── Smith Model Overview ─────────────────────────────────────
    model_items = []
    model_items.append(CalcStep(
        title="Smith 1-D Wave Equation Model",
        equation=(
            "The pile is modeled as a series of lumped masses connected "
            "by linear elastic springs. Soil resistance at each segment "
            "is an elasto-plastic spring with velocity-dependent damping."
        ),
        substitution="",
        result_name="Model",
        result_value="Smith (1960) — explicit time integration",
        reference="Smith (1960); FHWA GEC-12 Section 12.3",
    ))

    # Impact velocity
    hammer = analysis.get("hammer")
    if hammer is not None:
        if hammer.rated_energy is not None:
            v_eq = "v = \u221a(2 \u00d7 E_rated \u00d7 \u03b7 / m_ram)"
            v_sub = (
                f"v = \u221a(2 \u00d7 {hammer.energy:.1f} \u00d7 "
                f"{hammer.efficiency:.2f} / "
                f"{hammer.ram_mass:.1f})"
            )
        else:
            v_eq = "v = \u221a(2 \u00d7 g \u00d7 h \u00d7 \u03b7)"
            v_sub = (
                f"v = \u221a(2 \u00d7 9.81 \u00d7 {hammer.stroke:.3f} \u00d7 "
                f"{hammer.efficiency:.2f})"
            )
        model_items.append(CalcStep(
            title="Ram Impact Velocity",
            equation=v_eq,
            substitution=v_sub,
            result_name="v_impact",
            result_value=f"{hammer.impact_velocity:.3f}",
            result_unit="m/s",
            reference=("FHWA GEC-12 Vol 2 Eq. 15-1 (p. 358), with the "
                       "stroke times the hammer efficiency (Sec. 12.6.2, "
                       "p. 205)"),
            # A diesel run as a drop ram at its rated energy (GM-294 review)
            notes=diesel_warning(hammer) or "",
        ))

    # Courant condition
    pile = analysis.get("pile")
    if pile is not None and pile.n_segments > 0:
        c_wave = pile.wave_speeds[0]
        seg_len = pile.segment_lengths[0]
        model_items.append(CalcStep(
            title="Stress Wave Speed",
            equation="c = \u221a(E / \u03c1)",
            substitution="",
            result_name="c",
            result_value=f"{c_wave:,.0f}",
            result_unit="m/s",
            reference="Smith (1960)",
        ))
        model_items.append(CalcStep(
            title="Courant Stability Condition",
            equation=("\u0394t \u2264 \u0394L / c, divided by Smith's safety "
                      "factor against numerical instability, 1.6"),
            substitution=f"\u0394t \u2264 {seg_len:.3f} / {c_wave:,.0f}",
            result_name="\u0394t_Courant",
            result_value=f"{seg_len / c_wave * 1000:.4f}",
            result_unit="ms",
            reference=("FHWA GEC-12 Vol 2 Sec. 12.6.8 (p. 213): the factor "
                       "is normally 1.6 in GRLWEAP"),
        ))

    sections.append(CalcSection(
        title="Smith Wave Equation Model", items=model_items
    ))

    # ── Soil Resistance Model ────────────────────────────────────
    soil = analysis.get("soil")
    soil_items = []

    soil_items.append(CalcStep(
        title="Static Soil Resistance (Elasto-Plastic)",
        equation=(
            "Loading:   R_static = (R_ult / Q) \u00d7 (d - d_plastic) \u2264 R_ult\n"
            "Unloading: elastic slope R_ult / Q from the peak \u2014 a plastic\n"
            "offset d_plastic remains"
        ),
        substitution="Q = quake (elastic limit displacement)",
        result_name="Model",
        result_value="Smith elasto-plastic spring (kinematic, with memory)",
        reference="Smith (1960); FHWA GEC-12 Vol 2 Sec. 12.3 (p. 156)",
    ))

    damping_model = (getattr(soil, "damping_model", "smith") if soil is not None
                     else _used(analysis, "damping_model"))
    if damping_model == "smith_viscous":
        damping_step = CalcStep(
            title="Dynamic Soil Resistance (Smith-Viscous Damping)",
            equation="R_total = R_static + J \u00d7 R_ultimate \u00d7 v",
            substitution="J = Smith damping factor (s/m), v = pile segment velocity",
            result_name="Model",
            result_value="Damping proportional to the fully mobilized static resistance (Smith viscous)",
            reference="UFC 3-220-20 (2025) Sec. 6-9.2 (pp. 487-488)",
        )
    else:
        damping_step = CalcStep(
            title="Dynamic Soil Resistance (Smith Damping)",
            equation="R_total = R_static + J \u00d7 |R_static| \u00d7 v",
            substitution="J = Smith damping factor (s/m), v = pile segment velocity",
            result_name="Model",
            result_value="Damping proportional to the mobilized static resistance (Smith damping)",
            reference=("Smith (1960); FHWA GEC-12 Vol 2 Sec. 12.3 (p. 156); "
                       "UFC 3-220-20 (2025) p. 487"),
        )
    soil_items.append(damping_step)

    # The permanent set, as GEC-12 prints it (the soil's quakes as run)
    if soil is not None:
        f_s, q_s, q_t = soil.skin_fraction, soil.quake_side, soil.quake_toe
    elif analysis.get("quake_side") is not None:
        f_s, q_s, q_t = (_used(analysis, "skin_fraction"),
                         _used(analysis, "quake_side"),
                         _used(analysis, "quake_toe"))
    else:
        f_s = None
    soil_items.append(CalcStep(
        title="Permanent Set per Blow",
        equation=("s = D_toe,max - Q_avg,  Q_avg = f_skin \u00d7 Q_side + "
                  "(1 - f_skin) \u00d7 Q_toe"),
        substitution=("" if f_s is None else
                      f"Q_avg = {f_s:.2f} \u00d7 {q_s * 1000:.2f} + "
                      f"{1 - f_s:.2f} \u00d7 {q_t * 1000:.2f}"),
        result_name="Q_avg",
        result_value=("" if f_s is None else
                      f"{(f_s * q_s + (1 - f_s) * q_t) * 1000:.2f}"),
        result_unit="" if f_s is None else "mm",
        reference=("FHWA GEC-12 Vol 2 Sec. 12.3 (p. 158); the weights are "
                   "the module's assumption"),
        notes=("D_toe,max = maximum pile toe displacement. GEC-12 prints "
               "'a weighted average of the shaft and toe quakes' and names "
               "no weights; the shares of R_ult on the shaft and at the toe "
               "are this module's assumption, not printed. Other weightings: "
               "a plain average"
               + ("" if f_s is None else
                  f", Q_avg = {(q_s + q_t) / 2 * 1000:.2f} mm")
               + ", or the toe quake alone"
               + ("" if f_s is None else f", Q_avg = {q_t * 1000:.2f} mm")
               + "; where the quakes differ, the choices list the blow "
                 "counts each gives."),
    ))

    if soil is not None:
        skin_R = soil.R_ultimate * soil.skin_fraction
        toe_R = soil.R_ultimate * (1.0 - soil.skin_fraction)
        soil_items.append(CalcStep(
            title="Resistance Distribution",
            equation="R_skin = R_ult \u00d7 f_skin,  R_toe = R_ult \u00d7 (1 - f_skin)",
            substitution=(
                f"R_skin = {soil.R_ultimate:,.0f} \u00d7 {soil.skin_fraction:.2f},  "
                f"R_toe = {soil.R_ultimate:,.0f} \u00d7 {1 - soil.skin_fraction:.2f}"
            ),
            result_name="R_skin, R_toe",
            result_value=f"{skin_R:,.0f} kN, {toe_R:,.0f} kN",
            reference=("Input; the skin spread evenly over the pile. FHWA "
                       "GEC-12 Vol 2 p. 212: stresses, tension above all, "
                       "are sensitive to the resistance distribution and "
                       "the percentage at the toe"),
        ))

        soil_items.append(TableData(
            title="Smith Soil Parameters",
            headers=["Parameter", "Side", "Toe", "Unit"],
            rows=[
                ["Quake (Q)",
                 f"{soil.quake_side * 1000:.1f}",
                 f"{soil.quake_toe * 1000:.1f}", "mm"],
                ["Damping (J)",
                 f"{soil.damping_side:.2f}",
                 f"{soil.damping_toe:.2f}", "s/m"],
            ],
            notes=("FHWA GEC-12 Vol 2 Sec. 12.6.7 (pp. 211-212): shaft "
                   "quake 2.5 mm (0.10 in); toe quake 2.5 mm for a "
                   "non-displacement pile (p. 171), D/120 to D/60 for a "
                   "displacement pile, 1 mm (0.04 in) on hard rock. "
                   "J_side 0.16 s/m non-cohesive soils, 0.33 silty and "
                   "clayey sands and sandy silts, 0.49 cohesive silts and "
                   "sandy clays, 0.66 cohesive soils; J_toe about 0.49 "
                   "s/m for most soils."),
        ))

    sections.append(CalcSection(
        title="Soil Resistance Model", items=soil_items
    ))

    # ── Bearing Graph Results ────────────────────────────────────
    if _is_bearing_graph(result):
        sections.extend(_bearing_graph_steps(result, analysis))

    # ── Drivability Results ──────────────────────────────────────
    if _is_drivability(result):
        sections.extend(_drivability_steps(result, analysis))

    return sections


def get_figures(result, analysis) -> List[FigureData]:
    """Generate WEAP-style figures for the wave equation calc package.

    Parameters
    ----------
    result : BearingGraphResult or DrivabilityResult
    analysis : dict

    Returns
    -------
    list of FigureData
    """
    figures = []

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return figures

    if _is_bearing_graph(result):
        figures.extend(_bearing_graph_figures(result, analysis))

    if _is_drivability(result):
        figures.extend(_drivability_figures(result, analysis))

    return figures


# ─── Bearing Graph Sections ──────────────────────────────────────────

def _bearing_graph_steps(result, analysis) -> List[CalcSection]:
    """Build calc sections specific to bearing graph results."""
    sections = []

    import numpy as np

    # ── Bearing Graph Data Table ──
    bg_items = []

    bg_items.append(CalcStep(
        title="Bearing Graph",
        equation=(
            "For each assumed R_ult, simulate a single blow and compute:\n"
            "  Set = permanent pile penetration per blow\n"
            "  Blow count = 1 / Set  (blows per meter)"
        ),
        substitution="",
        result_name="N_analyses",
        result_value=str(len(result.R_values)),
        reference="FHWA GEC-12 Vol 2 Sec. 12.3 (p. 158) and 12.5.1",
    ))

    valid = result.blow_counts < 1e5
    bg_rows = []
    for i in range(len(result.R_values)):
        bc_str = (f"{result.blow_counts[i]:,.0f}"
                  if result.blow_counts[i] < 1e5 else "Refusal")
        bg_rows.append([
            f"{result.R_values[i]:,.0f}",
            f"{result.permanent_sets[i] * 1000:.2f}",
            bc_str,
            f"{result.max_comp_stresses[i]:,.0f}",
            f"{result.max_tens_stresses[i]:,.0f}",
        ])

    bg_items.append(TableData(
        title="Bearing Graph Data",
        headers=["R_ult (kN)", "Set (mm)", "Blows/m",
                 "Max Comp. (kPa)", "Max Tens. (kPa)"],
        rows=bg_rows,
    ))

    sections.append(CalcSection(
        title="Bearing Graph Results", items=bg_items
    ))

    # ── Driving Stress Check ──
    stress_items = []

    # Find peak stresses
    max_comp = float(np.max(result.max_comp_stresses))
    max_tens = float(np.max(result.max_tens_stresses))

    stress_items.append(CalcStep(
        title="Maximum Compressive Driving Stress",
        equation="Max compressive stress over all R_ult values",
        substitution="",
        result_name="\u03c3_comp,max",
        result_value=f"{max_comp:,.0f}",
        result_unit="kPa",
        notes=f"= {max_comp / 1000:,.1f} MPa",
    ))

    stress_items.append(CalcStep(
        title="Maximum Tensile Driving Stress",
        equation="Max tensile stress over all R_ult values",
        substitution="",
        result_name="\u03c3_tens,max",
        result_value=f"{max_tens:,.0f}",
        result_unit="kPa",
        notes=f"= {max_tens / 1000:,.1f} MPa",
    ))

    # The steel driving stress limit, as the yield stress this stress needs
    # (the pile's grade is not an input; GEC-12 Vol 1 Eq. 8-33)
    stress_items.append(CalcStep(
        title="Steel Driving Stress Limit",
        equation=("\u03c3_dr = \u03c6_da \u00d7 0.9 \u00d7 F_y \u2265 "
                  "\u03c3_comp,max with \u03c6_da = 1.0, so "
                  "F_y \u2265 \u03c3_comp,max / 0.9"),
        substitution=f"F_y \u2265 {max_comp / 1000:,.1f} / 0.9",
        result_name="F_y,needed",
        result_value=f"{max_comp / 1000 / 0.9:,.0f}",
        result_unit="MPa",
        reference="FHWA GEC-12 Vol 1 Eq. 8-33 (p. 445)",
        notes=("For a steel pile. New steel H-piles have a minimum yield "
               "stress of 50 ksi (345 MPa) (GEC-12 Vol 3 p. 38). A "
               "concrete pile's limits are 0.85 f'c - f_pe in compression "
               "and 0.095 \u221af'c + f_pe in tension, ksi (Vol 1 Eqs. 8-62 "
               "and 8-63, p. 459)."),
    ))

    sections.append(CalcSection(
        title="Driving Stress Summary", items=stress_items
    ))

    # ── Blow-level detail for one representative R_ult ──
    if result.blow_results:
        # Pick the middle analysis
        mid_idx = len(result.blow_results) // 2
        blow = result.blow_results[mid_idx]
        r_ult_val = result.R_values[mid_idx]

        detail_items = []
        detail_items.append(CalcStep(
            title=f"Single Blow Detail (R_ult = {r_ult_val:,.0f} kN)",
            equation="Explicit central-difference time integration",
            substitution="",
            result_name="n_steps",
            result_value=str(blow.n_steps),
            notes=f"Permanent set = {blow.permanent_set * 1000:.2f} mm",
        ))

        detail_items.append(TableData(
            title="Blow Simulation Summary",
            headers=["Quantity", "Value", "Unit"],
            rows=[
                ["Ultimate resistance", f"{blow.R_ultimate:,.0f}", "kN"],
                ["Permanent set", f"{blow.permanent_set * 1000:.2f}", "mm"],
                ["Max compression stress",
                 f"{blow.max_compression_stress:,.0f}", "kPa"],
                ["Max tension stress",
                 f"{blow.max_tension_stress:,.0f}", "kPa"],
                ["Max compressive force",
                 f"{blow.max_pile_force:,.0f}", "kN"],
                ["Time steps computed", str(blow.n_steps), ""],
            ],
        ))

        sections.append(CalcSection(
            title="Representative Blow Detail", items=detail_items
        ))

    return sections


# ─── Drivability Sections ────────────────────────────────────────────

def _drivability_steps(result, analysis) -> List[CalcSection]:
    """Build calc sections specific to drivability results."""
    sections = []

    drv_items = []

    drv_items.append(CalcStep(
        title="Drivability Study",
        equation=(
            "Wave equation analysis at each penetration depth using\n"
            "the anticipated soil resistance profile vs depth."
        ),
        substitution="",
        result_name="N_depths",
        result_value=str(len(result.points)),
        reference="FHWA GEC-12 Vol 2 Sec. 12.5.3 (pp. 167-172)",
    ))

    # Results table
    drv_rows = []
    for pt in result.points:
        bc_str = (f"{pt.blow_count:,.0f}"
                  if pt.blow_count < 1e5 else "Refusal")
        drv_rows.append([
            f"{pt.depth:.1f}",
            f"{pt.R_ultimate:,.0f}",
            f"{pt.permanent_set * 1000:.2f}",
            bc_str,
            f"{pt.max_comp_stress:,.0f}",
            f"{pt.max_tens_stress:,.0f}",
        ])

    drv_items.append(TableData(
        title="Drivability Results",
        headers=["Depth (m)", "R_ult (kN)", "Set (mm)", "Blows/m",
                 "Comp. (kPa)", "Tens. (kPa)"],
        rows=drv_rows,
    ))

    sections.append(CalcSection(
        title="Drivability Analysis Results", items=drv_items
    ))

    # ── Drivability Assessment ──
    assess_items = []
    limit = getattr(result, "refusal_blow_count", None)
    limit_eq = ("Pile can be driven if blow count ≤ refusal limit at all "
                "depths" + ("" if limit is None else
                            f" ({limit:,.1f} blows/m)"))
    limit_ref = ("FHWA GEC-12 Vol 2 Sec. 12.4 (p. 161): 120 blows/ft "
                 "(393.7 blows/m); 240 blows/ft (787.4 blows/m) for piles "
                 "seated on hard rock")
    if result.can_drive:
        assess_items.append(CalcStep(
            title="Drivability Assessment",
            equation=limit_eq,
            substitution="",
            result_name="Result",
            result_value="PILE CAN BE DRIVEN to all specified depths",
            reference=limit_ref,
            notes="No refusal encountered at any depth.",
        ))
    else:
        assess_items.append(CalcStep(
            title="Drivability Assessment",
            equation=limit_eq,
            substitution="",
            result_name="Result",
            result_value=f"REFUSAL at depth {result.refusal_depth:.1f} m",
            reference=limit_ref,
            notes="Consider a larger hammer, pre-drilling, or jetting.",
        ))

        assess_items.append(CheckItem(
            description="Pile drivability to target depth",
            demand=result.refusal_depth,
            demand_label="Refusal depth",
            capacity=result.points[-1].depth if result.points else 0,
            capacity_label="Target depth",
            unit="m",
            passes=False,
        ))

    # Peak stresses across all depths
    if result.points:
        max_comp = max(pt.max_comp_stress for pt in result.points)
        max_tens = max(pt.max_tens_stress for pt in result.points)
        assess_items.append(TableData(
            title="Peak Driving Stresses (All Depths)",
            headers=["Stress Type", "Value (kPa)", "Value (MPa)"],
            rows=[
                ["Max compression", f"{max_comp:,.0f}", f"{max_comp / 1000:.1f}"],
                ["Max tension", f"{max_tens:,.0f}", f"{max_tens / 1000:.1f}"],
            ],
            notes=("Check against the driving stress limits: steel "
                   "0.9 F_y (FHWA GEC-12 Vol 1 Eq. 8-33); concrete Eqs. "
                   "8-62 and 8-63."),
        ))

    sections.append(CalcSection(
        title="Drivability Assessment", items=assess_items
    ))

    return sections


# ─── Bearing Graph Figures ───────────────────────────────────────────

def _bearing_graph_figures(result, analysis) -> List[FigureData]:
    """Generate figures for bearing graph results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    figures = []
    fig_num = 1

    # ── Figure 1: Bearing Graph (Capacity vs Blow Count) ──
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        valid = result.blow_counts < 1e5
        if np.any(valid):
            ax.plot(result.blow_counts[valid], result.R_values[valid],
                    'bo-', linewidth=2, markersize=6)
        ax.set_xlabel('Blow Count (blows/m)', fontsize=10)
        ax.set_ylabel('Ultimate Resistance, R_ult (kN)', fontsize=10)
        ax.set_title('Wave Equation Bearing Graph', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Annotate min/max
        if np.any(valid):
            r_min = result.R_values[valid][0]
            r_max = result.R_values[valid][-1]
            bc_min = result.blow_counts[valid][0]
            bc_max = result.blow_counts[valid][-1]
            # both inside the frame: right of the first point, left of
            # the last (beyond them they ran past the frame at the
            # printed width; GM-186)
            ax.annotate(f'{r_min:,.0f} kN',
                        xy=(bc_min, r_min), fontsize=8,
                        xytext=(8, 0), textcoords='offset points',
                        va='center')
            ax.annotate(f'{r_max:,.0f} kN',
                        xy=(bc_max, r_max), fontsize=8,
                        xytext=(-8, 3), textcoords='offset points',
                        ha='right', va='center')

        plt.tight_layout()
        b64 = figure_to_base64(fig, dpi=150)
        plt.close(fig)
        figures.append(FigureData(
            title="Bearing Graph",
            image_base64=b64,
            caption=(
                f"Figure {fig_num}: Wave equation bearing graph — ultimate "
                f"resistance vs blow count. Range: "
                f"{result.R_values[0]:,.0f} to {result.R_values[-1]:,.0f} kN."
            ),
            width_percent=80,
        ))
        fig_num += 1
    except Exception:
        pass

    # ── Figure 2: Bearing Graph with Driving Stresses ──
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        valid = result.blow_counts < 1e5
        twins = [ax]
        if np.any(valid):
            ax.plot(result.blow_counts[valid], result.R_values[valid],
                    'bo-', linewidth=2, markersize=6, label='R_ult')
            ax.set_xlabel('Blow Count (blows/m)', fontsize=10)
            ax.set_ylabel('Ultimate Resistance (kN)', fontsize=10, color='b')
            ax.tick_params(axis='y', labelcolor='b')

            ax2 = ax.twinx()
            ax2.plot(result.blow_counts[valid],
                     result.max_comp_stresses[valid] / 1000,
                     'r--s', linewidth=1.5, markersize=4,
                     label='Comp. Stress')
            ax2.plot(result.blow_counts[valid],
                     result.max_tens_stresses[valid] / 1000,
                     'g--^', linewidth=1.5, markersize=4,
                     label='Tens. Stress')
            ax2.set_ylabel('Driving Stress (MPa)', fontsize=10)

            twins.append(ax2)

        ax.set_title('Bearing Graph with Driving Stresses',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # one key for both axes, under them: inside, the stress lines on
        # the twin axis ran through it (GM-194)
        from geotech_common.plotting import key_under_axes
        key_under_axes(fig, twins, ncol=3)
        b64 = figure_to_base64(fig, dpi=150)
        plt.close(fig)
        figures.append(FigureData(
            title="Bearing Graph with Driving Stresses",
            image_base64=b64,
            caption=(
                f"Figure {fig_num}: Bearing graph with overlaid maximum "
                f"compressive and tensile driving stresses."
            ),
            width_percent=85,
        ))
        fig_num += 1
    except Exception:
        pass

    # ── Figure 3: Force/Velocity at Pile Head (representative blow) ──
    if result.blow_results:
        try:
            mid_idx = len(result.blow_results) // 2
            blow = result.blow_results[mid_idx]
            r_ult_val = result.R_values[mid_idx]

            if len(blow.time) > 1:
                fig, ax = plt.subplots(figsize=(9, 5))
                t_ms = blow.time * 1000  # s -> ms

                color_f = '#2563eb'
                color_v = '#dc2626'

                ax.plot(t_ms, blow.pile_head_force, '-',
                        color=color_f, linewidth=1.5, label='Force (kN)')
                ax.set_xlabel('Time (ms)', fontsize=10)
                ax.set_ylabel('Pile Head Force (kN)', fontsize=10, color=color_f)
                ax.tick_params(axis='y', labelcolor=color_f)

                ax2 = ax.twinx()
                ax2.plot(t_ms, blow.pile_head_velocity, '--',
                         color=color_v, linewidth=1.5, label='Velocity (m/s)')
                ax2.set_ylabel('Pile Head Velocity (m/s)', fontsize=10,
                               color=color_v)
                ax2.tick_params(axis='y', labelcolor=color_v)

                ax.set_title(
                    f'Pile Head Response (R_ult = {r_ult_val:,.0f} kN)',
                    fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)

                # Combined legend
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2,
                          loc='upper right', fontsize=8)

                plt.tight_layout()
                b64 = figure_to_base64(fig, dpi=150)
                plt.close(fig)
                figures.append(FigureData(
                    title="Pile Head Force & Velocity vs Time",
                    image_base64=b64,
                    caption=(
                        f"Figure {fig_num}: Force and velocity time histories "
                        f"at the pile head for R_ult = {r_ult_val:,.0f} kN. "
                        f"Permanent set = {blow.permanent_set * 1000:.2f} mm."
                    ),
                    width_percent=85,
                ))
                fig_num += 1
        except Exception:
            pass

    # ── Figure 4: Max Comp/Tens Stress vs R_ult ──
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(result.R_values, result.max_comp_stresses / 1000,
                'rs-', linewidth=1.5, markersize=5, label='Compression')
        ax.plot(result.R_values, result.max_tens_stresses / 1000,
                'g^-', linewidth=1.5, markersize=5, label='Tension')
        ax.set_xlabel('Ultimate Resistance, R_ult (kN)', fontsize=10)
        ax.set_ylabel('Driving Stress (MPa)', fontsize=10)
        ax.set_title('Maximum Driving Stresses vs Ultimate Resistance',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        b64 = figure_to_base64(fig, dpi=150)
        plt.close(fig)
        figures.append(FigureData(
            title="Driving Stresses vs Ultimate Resistance",
            image_base64=b64,
            caption=(
                f"Figure {fig_num}: Maximum compressive and tensile driving "
                f"stresses as a function of ultimate resistance."
            ),
            width_percent=75,
        ))
        fig_num += 1
    except Exception:
        pass

    return figures


# ─── Drivability Figures ─────────────────────────────────────────────

def _drivability_figures(result, analysis) -> List[FigureData]:
    """Generate figures for drivability results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures = []
    fig_num = 1

    if not result.points:
        return figures

    depths = [pt.depth for pt in result.points]
    blow_counts = [pt.blow_count for pt in result.points]
    r_values = [pt.R_ultimate for pt in result.points]
    comp_stresses = [pt.max_comp_stress for pt in result.points]
    tens_stresses = [pt.max_tens_stress for pt in result.points]

    # Cap blow counts for display (refusal shown at plot edge)
    bc_display = [min(bc, 5000) for bc in blow_counts]

    # ── Figure 1: Blow Count vs Depth ──
    try:
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.plot(bc_display, depths, 'bo-', linewidth=2, markersize=5)
        ax.set_xlabel('Blow Count (blows/m)', fontsize=10)
        ax.set_ylabel('Depth (m)', fontsize=10)
        ax.set_title('Drivability — Blow Count vs Depth',
                     fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        # Mark refusal if applicable
        if not result.can_drive:
            ax.axhline(y=result.refusal_depth, color='#dc2626',
                       linestyle='--', linewidth=1.5,
                       label=f'Refusal at {result.refusal_depth:.1f} m')
            ax.legend(fontsize=9)

        plt.tight_layout()
        b64 = figure_to_base64(fig, dpi=150)
        plt.close(fig)
        figures.append(FigureData(
            title="Blow Count vs Depth",
            image_base64=b64,
            caption=(
                f"Figure {fig_num}: Predicted blow count profile with depth. "
                + ("Refusal encountered at "
                   f"{result.refusal_depth:.1f} m." if not result.can_drive
                   else "No refusal — pile can be driven to all depths.")
            ),
            width_percent=60,
        ))
        fig_num += 1
    except Exception:
        pass

    # ── Figure 2: Resistance vs Depth ──
    try:
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.plot(r_values, depths, 'ko-', linewidth=2, markersize=5)
        ax.set_xlabel('Ultimate Resistance, R_ult (kN)', fontsize=10)
        ax.set_ylabel('Depth (m)', fontsize=10)
        ax.set_title('Soil Resistance Profile',
                     fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        b64 = figure_to_base64(fig, dpi=150)
        plt.close(fig)
        figures.append(FigureData(
            title="Resistance Profile vs Depth",
            image_base64=b64,
            caption=(
                f"Figure {fig_num}: Ultimate soil resistance profile with "
                f"depth used in the drivability study."
            ),
            width_percent=60,
        ))
        fig_num += 1
    except Exception:
        pass

    # ── Figure 3: Driving Stresses vs Depth ──
    try:
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.plot([s / 1000 for s in comp_stresses], depths,
                'rs-', linewidth=1.5, markersize=5, label='Compression')
        ax.plot([s / 1000 for s in tens_stresses], depths,
                'g^-', linewidth=1.5, markersize=5, label='Tension')
        ax.set_xlabel('Driving Stress (MPa)', fontsize=10)
        ax.set_ylabel('Depth (m)', fontsize=10)
        ax.set_title('Driving Stresses vs Depth',
                     fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        b64 = figure_to_base64(fig, dpi=150)
        plt.close(fig)
        figures.append(FigureData(
            title="Driving Stresses vs Depth",
            image_base64=b64,
            caption=(
                f"Figure {fig_num}: Maximum compressive and tensile "
                f"driving stresses at each penetration depth."
            ),
            width_percent=60,
        ))
        fig_num += 1
    except Exception:
        pass

    # ── Figure 4: Combined Blow Count + Resistance ──
    try:
        fig, ax1 = plt.subplots(figsize=(7, 8))

        color1 = '#2563eb'
        color2 = '#d97706'

        ax1.plot(bc_display, depths, 'o-', color=color1,
                 linewidth=2, markersize=5, label='Blow Count')
        ax1.set_xlabel('Blow Count (blows/m)', fontsize=10, color=color1)
        ax1.set_ylabel('Depth (m)', fontsize=10)
        ax1.tick_params(axis='x', labelcolor=color1)
        ax1.invert_yaxis()

        ax2 = ax1.twiny()
        ax2.plot(r_values, depths, 's-', color=color2,
                 linewidth=1.5, markersize=4, label='R_ult')
        ax2.set_xlabel('Ultimate Resistance (kN)', fontsize=10, color=color2)
        ax2.tick_params(axis='x', labelcolor=color2)

        ax1.set_title('Drivability — Blow Count & Resistance vs Depth',
                      fontsize=12, fontweight='bold', pad=30)
        ax1.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2,
                   loc='lower right', fontsize=8)

        plt.tight_layout()
        b64 = figure_to_base64(fig, dpi=150)
        plt.close(fig)
        figures.append(FigureData(
            title="Combined Drivability Profile",
            image_base64=b64,
            caption=(
                f"Figure {fig_num}: Combined blow count and resistance "
                f"profiles with depth."
            ),
            width_percent=65,
        ))
        fig_num += 1
    except Exception:
        pass

    return figures
