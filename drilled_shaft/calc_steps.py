"""
Calculation package steps for drilled shaft capacity analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation.

References:
    FHWA GEC-10 (FHWA-NHI-18-024), Brown, Turner, Castelli & Loehr (2018)
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
    'Brown, D.A., Turner, J.P., Castelli, R.J. & Loehr, E.J. (2018). '
    '"Drilled Shafts: Construction Procedures and Design Methods." '
    'FHWA-NHI-18-024, GEC-10.',
    'Samtani, N.C. & Nowatzki, E.A. (2006). "Soils and Foundations '
    'Reference Manual, Volume II." FHWA NHI-06-089.',
    'Brown, D.A., Turner, J.P. & Castelli, R.J. (2010). "Drilled Shafts: '
    'Construction Procedures and LRFD Design Methods." '
    'FHWA-NHI-10-016, GEC-10.',
    'O\'Neill, M.W. & Reese, L.C. (1999). "Drilled Shafts: Construction '
    'Procedures and Design Methods." FHWA-RD-99-049.',
    'Horvath, R.G. & Kenney, T.C. (1979). "Shaft Resistance of Rock-Socketed '
    'Drilled Piers." Symposium on Deep Foundations, ASCE, pp. 182-214.',
    "FHWA GEC-10 (FHWA-NHI-18-024), Chapter 10, Sec. 10.3.5: nominal side and "
    "base resistances; Table 8-4: resistance factors.",
]


# ── Atmospheric pressure constant (kPa) for alpha method ─────────
from .side_resistance import PA as _PA


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
    _, _, base_area = _base(shaft, soil)

    items = [
        InputItem("D", "Shaft diameter", shaft.diameter, "m"),
        InputItem("L", "Shaft length", shaft.length, "m"),
        InputItem("P", "Shaft perimeter", f"{shaft.perimeter:.3f}", "m"),
        InputItem("A_tip", "Tip area", f"{base_area:.4f}", "m\u00b2"),
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
    if shaft.fractured_rock_thickness > 0:
        items.append(InputItem(
            "t_fr", "Rock not part of the socket, below the top of rock",
            shaft.fractured_rock_thickness, "m",
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

    # Side resistance settings: the top zone, and in sand the beta (GM-241)
    from drilled_shaft.capacity import TOP_EXCLUSION
    items.append(InputItem(
        "z_top", "Top zone with no side resistance (or the casing)",
        (analysis.top_exclusion if analysis.top_exclusion is not None
         else f"{TOP_EXCLUSION:g} in cohesive soil, 0 in cohesionless "
              "soil and rock"), "m"))
    if any(lay.soil_type == "cohesive" for lay in soil.layers):
        items.append(InputItem(
            "Alpha", "Side resistance in cohesive soil",
            "GEC-10 (2018) Eq. 10-18 on su(CIUC), cu taken as "
            + {"ciuc": "su(CIUC)", "uc": "UC (Eq. 10-16)",
               "uu": "UU (Eq. 10-17)"}[analysis.su_test_type]
            if analysis.alpha_method == "rational"
            else "alpha of O'Neill & Reese (1999), NHI-06-089 Eq. 9-37",
            ""))
        items.append(InputItem(
            "z_bot", "Bottom zone with no side resistance (cohesive soil)",
            analysis.bottom_exclusion, "m"))
    if any(lay.soil_type == "cohesionless" for lay in soil.layers):
        gec10 = analysis.beta_method == "rational"
        items.append(InputItem(
            "Beta", "Side resistance in cohesionless soil",
            "GEC-10 (2018) Eq. 10-13" if gec10
            else "depth beta, O'Neill & Reese (1999)", ""))
        if gec10:
            items.append(InputItem(
                "K/K_o", "K/K_o, construction's effect on the horizontal "
                "stress", analysis.K_over_Ko, ""))

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
    base_name, base_D, base_area = _base(shaft, soil)
    geom_steps.append(CalcStep(
        title="Shaft Tip Area",
        equation=f"A_tip = \u03c0 \u00d7 {base_name}\u00b2 / 4",
        substitution=f"A_tip = \u03c0 \u00d7 {base_D:.3f}\u00b2 / 4",
        result_name="A_tip",
        result_value=f"{base_area:.4f}",
        result_unit="m\u00b2",
        notes=("Bell diameter used for tip area" if base_name == "D_bell"
               else "Rock socket diameter used for the base in rock"
               if base_name == "D_sock" else ""),
        reference=("GEC-10 (FHWA-NHI-18-024) Example 10-4, p. 10-28: "
                   "R_BN = \u03c0/4 B\u00b2 q_BN, B the socket diameter"
                   if base_name == "D_sock" else ""),
    ))
    geom_steps.append(CalcStep(
        title="Length-to-Diameter Ratio",
        equation="L/D = L / D",
        substitution=f"L/D = {shaft.length:.2f} / {shaft.diameter:.3f}",
        result_name="L/D",
        result_value=f"{shaft.length / shaft.diameter:.1f}",
    ))

    # Exclusion zones: the top zone by soil type unless one is given
    c = shaft.casing_depth
    if analysis.top_exclusion is not None:
        top_excl = analysis._top_of_side("cohesive")
        top_equation = "z_excl_top = max(top zone, casing depth), every layer"
        top_sub = (f"z_excl_top = max({analysis.top_exclusion:.2f}, "
                   f"{c:.2f})")
        top_value = f"{top_excl:.2f}"
        top_notes = ("Given; GEC-10 prints the top 5 ft for cohesive soils "
                     "and 'subject to modification based on local "
                     "experience and judgment of the designer'")
    else:
        clay = analysis._top_of_side("cohesive")
        sand = analysis._top_of_side("cohesionless")
        top_equation = ("z_excl_top = max(1.5 m, casing depth) in cohesive "
                        "soil; max(0, casing depth) in cohesionless soil "
                        "and rock")
        top_sub = (f"cohesive soil: max(1.50, {c:.2f}) = {clay:.2f} m; "
                   f"cohesionless soil and rock: max(0, {c:.2f}) = "
                   f"{sand:.2f} m")
        top_value = f"{clay:.2f} / {sand:.2f}"
        top_notes = ("The top 5 ft is printed in Sec. 10.3.5.2, Cohesive "
                     "Soils, 'subject to modification based on local "
                     "experience and judgment of the designer'; Secs. "
                     "10.3.5.1 and 10.3.5.3 print no such zone in "
                     "cohesionless soil or rock: Examples 10-1 and 10-5 "
                     "take the sand's side resistance from the ground "
                     "surface, Example 10-4 the socket's over its whole "
                     "length below the casing")
    geom_steps.append(CalcStep(
        title="Top Exclusion Zone",
        equation=top_equation,
        substitution=top_sub,
        result_name="z_excl_top",
        result_value=top_value,
        result_unit="m",
        reference=("GEC-10 (FHWA-NHI-18-024) p. 10-17: 'common design "
                   "practice to neglect side resistance over the top 5 ft "
                   "of a drilled shaft, or to the depth of seasonal "
                   "moisture change'"),
        notes=top_notes,
    ))
    z_bot = analysis.bottom_exclusion
    if any(lay.soil_type == "cohesive" for lay in soil.layers) and z_bot > 0:
        geom_steps.append(CalcStep(
            title="Bottom Exclusion Zone (cohesive soil)",
            equation="z_excl_bot = L - z_bot (given)",
            substitution=f"z_excl_bot = {shaft.length:.2f} - {z_bot:.3f}",
            result_name="z_excl_bot",
            result_value=f"{shaft.length - z_bot:.2f}",
            result_unit="m",
            reference=("FHWA NHI-06-089 p. 9-129 (O'Neill & Reese 1999): "
                       "the bottom 5 ft, its Fig. 9-49 the bottom one "
                       "diameter; GEC-10 (2018) prints no such zone"),
        ))
    elif any(lay.soil_type == "cohesive" for lay in soil.layers):
        geom_steps.append(CalcStep(
            title="Bottom Zone (cohesive soil)",
            equation="none: side resistance taken to the base",
            substitution="",
            result_name="z_excl_bot",
            result_value=f"{shaft.length:.2f}",
            result_unit="m",
            reference=("GEC-10 (FHWA-NHI-18-024) Example 10-2 (p. 10-18) "
                       "takes the clay's side resistance to the base; "
                       "Sec. 10.3.5.2 prints only the top 5 ft"),
            notes=("FHWA NHI-06-089 (O'Neill & Reese 1999) leaves out the "
                   "bottom 5 ft (p. 9-129) or one diameter (Fig. 9-49); "
                   "the choices give both with their values"),
        ))

    # Rock not considered part of the socket (GEC-10 p. 8-15, GM-220)
    from drilled_shaft.capacity import fractured_rock_zone
    not_socket = fractured_rock_zone(shaft, soil)
    if not_socket is not None:
        geom_steps.append(CalcStep(
            title="Rock Not Considered Part of the Socket",
            equation="z_rock to z_rock + t_fr: no rock side resistance",
            substitution=(
                f"{not_socket[0]:.2f} to {not_socket[0]:.2f} + "
                f"{shaft.fractured_rock_thickness:.2f}"),
            result_name="z_socket_top",
            result_value=f"{not_socket[1]:.2f}",
            result_unit="m",
            reference=("GEC-10 (FHWA-NHI-18-024) p. 8-15: decomposed or "
                       "fractured rock 'not considered part of the socket "
                       "for design purposes'"),
            notes="The soil above the rock keeps its side resistance.",
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

    if has_clay and analysis.alpha_method == "rational":
        method_steps.append(CalcStep(
            title="Alpha Method (Cohesive Soils), GEC-10 (2018) Eq. 10-18",
            equation=(
                "s_u(CIUC) = c_u where c_u is a CIUC strength; from UC: "
                "s_u(UC)/s_u(CIUC) = 0.893 + 0.513 log (s_u(UC)/\u03c3'_vo) "
                "(Eq. 10-16); from UU: s_u(UU)/s_u(CIUC) = 0.911 + 0.499 "
                "log (s_u(UU)/\u03c3'_vo) (Eq. 10-17)\n"
                "\u03b1 = 0.30 + 0.17 / (s_u(CIUC) / p_a) (Eq. 10-18)\n"
                "f_SN = \u03b1 \u00d7 s_u(CIUC)\n"
                "R_SN = \u03c0 B \u0394z f_SN (Eq. 10-15)"),
            substitution="",
            result_name="",
            result_value="",
            reference=("GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.2, Eqs. 10-15 "
                       "to 10-18 (pp. 10-15, 10-16), Fig. 10-6 (p. 10-17), "
                       "Example 10-2 (p. 10-18)"),
            notes=(f"c_u taken as "
                   + {"ciuc": "s_u(CIUC), untransformed",
                      "uc": "a UC strength (Eq. 10-16)",
                      "uu": "a UU strength (Eq. 10-17)"}[
                          analysis.su_test_type]
                   + f" (su_test_type '{analysis.su_test_type}'); p_a = "
                   f"{analysis.pa:g} kPa. Eq. 10-18 is recommended 'as a "
                   "revision' to the AASHTO alpha (p. 10-16)"),
        ))
    elif has_clay:
        method_steps.append(CalcStep(
            title="Alpha Method (Cohesive Soils), O'Neill & Reese (1999)",
            equation=(
                "\u03b1 = 0.55 for c_u/p_a \u2264 1.5\n"
                "\u03b1 = 0.55 - 0.1\u00d7(c_u/p_a - 1.5) for 1.5 < "
                "c_u/p_a \u2264 2.5\n"
                "f_s = \u03b1 \u00d7 c_u\n"
                "Q_s = f_s \u00d7 P \u00d7 \u0394z"
            ),
            substitution="",
            result_name="",
            result_value="",
            reference=("FHWA NHI-06-089 Eqs. 9-37a, 9-37b (p. 9-129), "
                       "O'Neill & Reese (1999); GEC-10 (2018) recommends "
                       "Eq. 10-18 instead (p. 10-16)"),
            notes=("NHI-06-089 prints nothing above c_u/p_a = 2.5; this "
                   "tool goes on along the line to 0.35 and holds it"),
        ))

    if has_sand and analysis.beta_method == "rational":
        method_steps.append(CalcStep(
            title="Beta Method (Cohesionless Soils), GEC-10 (2018) Eq. 10-13",
            equation=(
                "\u03c6' = the layer's \u03c6', or 27.5 + 9.2 log (N_1)_60\n"
                "\u03c3'_p / p_a = 0.47 N_60^m, m = 0.6 clean quartzitic "
                "sands, 0.8 silty sands to sandy silts (Eq. 10-11); "
                "\u03c3'_p / p_a = 0.15 N_60 gravelly soils (Eq. 10-12)\n"
                "OCR = \u03c3'_p / \u03c3'_v, \u03c3'_v the layer's average "
                "and not less than \u03c3'_v at 7.5 ft (2.29 m) (Eq. 10-9)\n"
                "K_o = (1 - sin \u03c6') OCR^(sin \u03c6') \u2264 K_p = "
                "tan\u00b2(45\u00b0 + \u03c6'/2) (Eqs. 10-8, 10-10)\n"
                "\u03b2 = (K/K_o) K_o tan \u03b4, \u03b4 = \u03c6' "
                "(Eqs. 10-6, 10-13)\n"
                "f_SN = \u03b2 \u03c3'_v (kPa) (Eq. 10-7)\n"
                "R_SN = \u03c0 B \u0394z f_SN (Eq. 10-5)"),
            substitution="",
            result_name="",
            result_value="",
            reference=("GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.1, Step 5 of "
                       "the design for axial load, Eqs. 10-5 to 10-13 (pp. "
                       "10-10 to 10-12); Appendix A Eq. A-2 (p. A-8) for "
                       "\u03c6' from (N_1)_60"),
            notes=(f"K/K_o = {analysis.K_over_Ko:.3g}: 1 is 'a rational "
                   "first-order approximation' (p. 10-11), 2/3 or lower "
                   "where construction quality is not controlled (p. "
                   "10-12). No upper limit on f_SN is printed. The "
                   "choices list each judgment with its alternatives."),
        ))
    elif has_sand:
        method_steps.append(CalcStep(
            title="Beta Method (Cohesionless Soils)",
            equation=(
                "\u03b2 = 1.5 - 0.135\u00d7\u221a(z, ft) = 1.5 - 0.245\u00d7\u221a(z, m), "
                "0.25 \u2264 \u03b2 \u2264 1.2, z the depth to the layer's centre\n"
                "\u03b2 \u00d7 N_60/15 where N_60 < 15\n"
                "f_s = \u03b2 \u00d7 \u03c3'_v \u2264 4 ksf = 191.5 kPa\n"
                "Q_s = f_s \u00d7 P \u00d7 \u0394z"
            ),
            substitution="",
            result_name="",
            result_value="",
            reference=("FHWA NHI-06-089 Eqs. 9-41 to 9-43 (pp. 9-132, 9-133), "
                       "the 4 ksf (191 kPa) limit p. 9-133; O'Neill & Reese "
                       "(1999) for the N_60/15 reduction"),
        ))

    if has_rock:
        method_steps.append(CalcStep(
            title="Rock Socket Method",
            equation=(
                "f_s = C \u00d7 p_a \u00d7 \u221a(q_u / p_a) = "
                "C \u00d7 \u221a(q_u \u00d7 p_a)\n"
                "Q_s = f_s \u00d7 P_socket \u00d7 \u0394z\n"
                "where p_a = 101.325 kPa and C = "
                f"{analysis.rock_side_C:g} (rock_side_C)"
            ),
            substitution="",
            result_name="",
            result_value="",
            reference=("GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.3, Eq. 10-21 "
                       "(p. 10-20) and Eq. 10-22 (p. 10-21)"),
            notes=("C = 1.0 is Eq. 10-21 as GEC-10 recommends it for "
                   "routine design of a 'normal' socket (pp. 10-20, "
                   "10-21); C = 0.65 is Eq. 10-22 (O'Neill & Reese 1999, "
                   "where caving is anticipated) with \u03b1_E = 1.0"
                   if analysis.rock_side_C == 1.0 else
                   "C = 0.65 is Eq. 10-22 (O'Neill & Reese 1999, where "
                   "caving is anticipated) with \u03b1_E = 1.0; GEC-10 "
                   "recommends Eq. 10-21 with C = 1.0 for routine design "
                   "of a 'normal' socket (pp. 10-20, 10-21)"),
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

            in_not_socket = (soil_type == "rock" and not_socket is not None
                             and z_top < not_socket[1])
            if method_used == "excluded":
                layer_items.append(CalcStep(
                    title=f"Layer {i}: {desc} ({z_top:.1f}-{z_bot:.1f} m)",
                    equation="Excluded from side resistance (within exclusion zone)",
                    substitution="",
                    result_name="Q_s",
                    result_value="0.0",
                    result_unit="kN",
                    notes=("Rock not considered part of the socket "
                           "(GEC-10 p. 8-15)" if in_not_socket
                           else "Layer within top or bottom exclusion zone"),
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
                        f"f_s = \u03b1 \u00d7 s_u, {method_used}\n"
                        f"Q_s = f_s \u00d7 P \u00d7 \u0394z"
                    ),
                    substitution=_alpha_lines(lb, analysis.pa) + (
                        f"\nQ_s = {fs:.1f} \u00d7 {shaft.perimeter:.3f} "
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

            elif soil_type == "cohesionless" and lb.get("beta_steps"):
                layer_items.append(_gec10_layer_step(
                    i, desc, lb, shaft.perimeter, analysis.pa))

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
                qu = soil.layer_at_depth(0.5 * (z_top + z_bot)).qu
                layer_items.append(CalcStep(
                    title=f"Layer {i}: {desc} ({z_top:.1f}-{z_bot:.1f} m) -- Rock Socket",
                    equation=(
                        f"f_s = C \u00d7 \u221a(q_u \u00d7 p_a)\n"
                        f"Q_s = f_s \u00d7 P_socket \u00d7 \u0394z"
                    ),
                    substitution=(
                        f"f_s = {analysis.rock_side_C:g} \u00d7 "
                        f"\u221a({qu:,.0f} \u00d7 {_PA:g}) = {fs:.1f} kPa\n"
                        f"Q_s = {fs:.1f} \u00d7 {socket_perim:.3f} "
                        f"\u00d7 {eff_thickness:.2f}"
                    ),
                    result_name="Q_s",
                    result_value=f"{Qs_layer:,.1f}",
                    result_unit="kN",
                    notes=f"Effective zone: {eff_top:.1f}-{eff_bot:.1f} m"
                    + ("; the rock above it is not considered part of the "
                       "socket (GEC-10 p. 8-15)" if in_not_socket else ""),
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

    # The unit end bearing the analysis used, each step from the same code
    # (GM-148): the reductions are governed by the base, so a belled
    # shaft's bell diameter, as DrillShaftAnalysis.compute does.
    from drilled_shaft.end_bearing import (
        QB_MAX_COHESIVE, rock_base_factor, unit_end_bearing_cohesionless,
        unit_end_bearing_cohesive)
    base_name, base_D, base_area = _base(shaft, soil)

    if tip_layer.soil_type == "cohesive":
        # the su the analysis took for the base (GM-244)
        cu = (r.base_steps or {}).get("su_kPa", tip_layer.cu)
        u = unit_end_bearing_cohesive(
            cu, L_over_D, base_diameter=base_D,
            shaft_length=shaft.length, method=analysis.clay_base_method,
            large_base_reduction=analysis.large_base_reduction)
        Nc = u["Nc"]
        if u["method"] == "gec10":
            # s_u as the mean over 2B below the base (GM-252)
            from drilled_shaft.capacity import _su_zone_words
            st = r.base_steps
            tip_words = ("the tip layer's s_u(CIUC), as for its side "
                         "resistance (Example 10-3, p. 10-20)"
                         if st.get("su_from") == "su_ciuc"
                         else "the tip layer's c_u")
            clay = [c for c in st["su_layers"] if c["su_kPa"] is not None]
            thick = sum(c["bottom_m"] - c["top_m"] for c in clay)
            eb_items.append(CalcStep(
                title="Base s_u, the Mean over 2B below the Base",
                equation=("s_u = Σ(s_u,i × Δz_i) / Σ Δz_i over the cohesive "
                          "soil from the base to 2B below it"),
                substitution=(
                    "(" + " + ".join(
                        f"{c['su_kPa']:.1f} × {c['bottom_m'] - c['top_m']:.2f}"
                        for c in clay) + f") / {thick:.2f}"
                    if len(clay) > 1 else f"s_u = {cu:.1f}"),
                result_name="s_u",
                result_value=f"{cu:.1f}",
                result_unit="kPa",
                reference=("GEC-10 (FHWA-NHI-18-024) p. 10-19: 'mean "
                           "undrained shear strength of the cohesive soil "
                           "over a depth of 2B below the base'"),
                notes=("s_u is " + _su_zone_words(st, tip_words)),
            ))
            eb_items.append(CalcStep(
                title="Bearing Capacity Factor N*c (Cohesive Base), Table 10-2",
                equation=("N*c = 6.5 at s_u = 500 psf, 8.0 at 1,000 psf, 9.0 "
                          "at 2,000 psf or more; linear between"),
                substitution=(f"s_u = {cu:.1f} kPa = {u['su_psf']:,.0f} "
                              "psf"),
                result_name="N*c",
                result_value=f"{Nc:.2f}",
                reference=("GEC-10 (FHWA-NHI-18-024) Table 10-2 and text, "
                           "p. 10-19"),
                notes=("s_u is below the table's first row (500 psf); N*c "
                       "is held at 6.5, the engineer's judgment (the "
                       "choices give the alternative)" if u["below_table"]
                       else ""),
            ))
            if u["equation"] == "10-19":
                equation = "q_BN = N*c × s_u (Eq. 10-19), D ≥ 3B"
                sub = (f"D/B = {u['D_over_B']:.2f}; q_BN = {Nc:.2f} × "
                       f"{cu:.1f}")
            else:
                equation = ("q_BN = (2/3)[1 + D/(6B)] N*c × s_u (Eq. 10-20), "
                            "D < 3B")
                sub = (f"D/B = {u['D_over_B']:.2f}; q_BN = (2/3)[1 + "
                       f"{u['D_over_B']:.2f}/6] × {Nc:.2f} × {cu:.1f}")
            eb_items.append(CalcStep(
                title="Unit End Bearing",
                equation=equation,
                substitution=sub,
                result_name="q_b",
                result_value=f"{u['qb_unreduced']:,.1f}",
                result_unit="kPa",
                reference=("GEC-10 (FHWA-NHI-18-024) Eqs. 10-19, 10-20 "
                           "(p. 10-19), Example 10-3 (p. 10-20)"),
                notes="GEC-10 (2018) prints no upper limit on q_BN",
            ))
        else:
            soft = (" × 0.67 (s_u < 0.5 ksf within 2 diameters of the tip)"
                    if u["soft_clay"] else "")
            eb_items.append(CalcStep(
                title="Bearing Capacity Factor N_c (Cohesive Tip), "
                      "O'Neill & Reese (1999)",
                equation="N_c = min(6.0 × (1 + 0.2 × L/D), 9.0)" + soft,
                substitution=(f"N_c = min(6.0 × (1 + 0.2 × {L_over_D:.1f}), "
                              f"9.0){soft}"),
                result_name="N_c",
                result_value=f"{Nc:.2f}",
                reference=("FHWA NHI-06-089 Eqs. 9-39, 9-40 (p. 9-131), "
                           "O'Neill & Reese (1999); GEC-10 (2018) takes N*c "
                           "from Table 10-2 (p. 10-19)"),
            ))
            cap_note = (
                f"N_c × c_u = {u['qb_uncapped']:,.1f} kPa exceeds the "
                f"limiting {QB_MAX_COHESIVE:,.0f} kPa (80 ksf); the limit is "
                f"used" if u["capped"] else "")
            eb_items.append(CalcStep(
                title="Unit End Bearing",
                equation=("q_b = N_c × c_u ≤ "
                          f"{QB_MAX_COHESIVE:,.0f} kPa (80 ksf)"),
                substitution=f"q_b = {Nc:.2f} × {cu:.1f}",
                result_name="q_b",
                result_value=f"{u['qb_unreduced']:,.1f}",
                result_unit="kPa",
                reference="FHWA NHI-06-089 Eq. 9-39 and p. 9-131",
                notes=cap_note,
            ))
        qb_cap = u["qb_unreduced"]
        if u["Fr"] is not None:
            eb_items.append(CalcStep(
                title="Large-Base Reduction F_r",
                equation=("F_r = 2.5 / (a × B_b + 2.5 b) ≤ 1.0, "
                          "a = 0.0071 + 0.0021 L/B_b ≤ 0.015, "
                          "b = 0.45 √(c_u, ksf) in [0.5, 1.5], "
                          "B_b in inches"),
                substitution=(
                    f"a = {u['a']:.4f}, b = {u['b']:.3f}, "
                    f"{base_name} = {base_D:.3f} m "
                    f"({base_D * 39.3701:.1f} in), L = {u['L']:.2f} m; "
                    f"q_b = {qb_cap:,.1f} × {u['Fr']:.3f}"),
                result_name="q_b",
                result_value=f"{u['qb']:,.1f}",
                result_unit="kPa",
                reference=("O'Neill & Reese (1999), large_base_reduction; "
                           "not printed in GEC-10 (2018)"),
                notes=f"Applies to a base wider than 1.90 m (75 in); "
                      f"{base_name} = {base_D:.3f} m",
            ))
        eb_items.append(CalcStep(
            title="End Bearing Capacity",
            equation="Q_tip = q_b × A_tip",
            substitution=f"Q_tip = {u['qb']:,.1f} × {base_area:.4f}",
            result_name="Q_tip",
            result_value=f"{r.Q_tip:,.1f}",
            result_unit="kN",
        ))

    elif tip_layer.soil_type == "cohesionless":
        from drilled_shaft.capacity import require_tip_N60
        N60 = require_tip_N60(tip_layer, shaft.length)
        u = unit_end_bearing_cohesionless(N60, base_D,
                                          analysis.large_base_reduction)
        large_dia_note = ""
        if u["reduction"] is not None:
            large_dia_note = (
                f"Large-diameter reduction: q_b = {u['qb_unreduced']:,.1f} "
                f"× (1.27/{base_D:.3f}) = {u['qb']:,.1f} kPa "
                f"({base_name}, the base diameter)"
            )

        eb_items.append(CalcStep(
            title="Unit End Bearing (Cohesionless Tip)",
            equation="q_b = 57.5 × N_60 (N_60 capped at 50)",
            substitution=f"q_b = 57.5 × {u['N60_capped']:.0f}",
            result_name="q_b",
            result_value=f"{u['qb']:,.1f}",
            result_unit="kPa",
            reference=("GEC-10 (FHWA-NHI-18-024) Eq. 10-14 (p. 10-14), "
                       "N60 limited to 50 (p. 10-15); the 1.27/B reduction "
                       "(large_base_reduction) is O'Neill & Reese (1999), "
                       "not printed in the 2018 edition"),
            notes=large_dia_note,
        ))
        eb_items.append(CalcStep(
            title="End Bearing Capacity",
            equation="Q_tip = q_b × A_tip",
            substitution=f"Q_tip = {u['qb']:,.1f} × {base_area:.4f}",
            result_name="Q_tip",
            result_value=f"{r.Q_tip:,.1f}",
            result_unit="kN",
        ))

    else:  # rock
        RQD = tip_layer.RQD
        qu = tip_layer.qu
        hb = (r.base_steps or {}).get("equation") == "10-29"
        if hb:
            # the same code as the analysis, unrounded (GM-148)
            from drilled_shaft.end_bearing import hoek_brown_base
            st = hoek_brown_base(qu, tip_layer.GSI, tip_layer.m_i,
                                 soil.effective_stress_at_depth(shaft.length))
            mult, qb = st["factor"], st["qb"]
        else:
            mult = rock_base_factor(RQD, analysis.rock_base_method)
            qb = mult * qu

        if hb:
            # the Hoek-Brown base of a fractured rock mass (GM-251)
            ref = "GEC-10 (FHWA-NHI-18-024) Eqs. 10-25 to 10-27 (p. 10-24)"
            eb_items.append(CalcStep(
                title="Hoek-Brown Parameters of the Rock Mass",
                equation=("m_b = m_i exp[(GSI − 100)/28]; s = exp[(GSI − "
                          "100)/9]; a = 1/2 + (1/6)[exp(−GSI/15) − "
                          "exp(−20/3)]"),
                substitution=f"GSI = {st['GSI']:g}, m_i = {st['m_i']:g}",
                result_name="m_b, s, a",
                result_value=(f"{st['m_b']:.3f}, {st['s']:.3g}, "
                              f"{st['a']:.3f}"),
                reference=ref,
                notes=("GSI and m_i are the tip rock layer's, the "
                       "engineer's (m_i by rock type, GEC 5 Table 9-10)"),
            ))
            eb_items.append(CalcStep(
                title="Hoek-Brown Base, Eq. 10-28",
                equation="A = σ'vb + q_u [m_b (σ'vb / q_u) + s]^a",
                substitution=(f"A = {st['sigma_vb_kPa']:,.1f} + {qu:,.1f} × "
                              f"[{st['m_b']:.3f} × ({st['sigma_vb_kPa']:,.1f} "
                              f"/ {qu:,.1f}) + {st['s']:.3g}]^{st['a']:.3f}"),
                result_name="A",
                result_value=f"{st['A_kPa']:,.1f}",
                result_unit="kPa",
                reference="GEC-10 (FHWA-NHI-18-024) Eq. 10-28 (p. 10-24)",
                notes="σ'vb is the vertical effective stress at the base",
            ))
            eb_items.append(CalcStep(
                title="Unit End Bearing (Fractured Rock Mass), GEC-10 Eq. 10-29",
                equation=("q_BN = A + q_u [m_b (A / q_u) + s]^a, q_BN / q_u "
                          "≤ 2.5"),
                substitution=(f"q_BN = {st['A_kPa']:,.1f} + {qu:,.1f} × "
                              f"[{st['m_b']:.3f} × ({st['A_kPa']:,.1f} / "
                              f"{qu:,.1f}) + {st['s']:.3g}]^{st['a']:.3f} = "
                              f"{st['qb_uncapped']:,.1f}"
                              + (f"; limited to 2.5 × {qu:,.1f}"
                                 if st["capped"] else "")),
                result_name="q_b",
                result_value=f"{qb:,.1f}",
                result_unit="kPa",
                reference=("GEC-10 (FHWA-NHI-18-024) Eqs. 10-24 to 10-29 "
                           "(pp. 10-23, 10-24), Fig. 10-8 (p. 10-25)"),
                notes=(f"N*cr = q_BN / q_u = {st['Ncr_uncapped']:.3f}"
                       + ("; 2.5 is the upper-bound limit 'unless local "
                          "experience or load tests can be used to validate "
                          "higher values' (p. 10-24)" if st["capped"] else "")
                       + ("; the call does not state the conditions of Eq. "
                          "10-23 (p. 10-22) met, and the layers within one "
                          "base diameter below the base show an RQD below "
                          "100 or soil"
                          if analysis.rock_base_conditions_met is None
                          else "; the rock below the base stated not to "
                          "meet the conditions of Eq. 10-23 (p. 10-22)")),
            ))
        elif analysis.rock_base_method == "gec10":
            eb_items.append(CalcStep(
                title="Unit End Bearing (Rock Base), GEC-10 Eq. 10-23",
                equation="q_BN = N*cr \u00d7 q_u, N*cr = 2.5",
                substitution=f"q_BN = 2.5 \u00d7 {qu:,.1f}",
                result_name="q_b",
                result_value=f"{qb:,.1f}",
                result_unit="kPa",
                reference=("GEC-10 (FHWA-NHI-18-024) Eq. 10-23 (p. 10-22), "
                           "Example 10-4 (p. 10-28)"),
                notes=("For rock massive or tightly jointed to one diameter "
                       "below the base, with no cavities or voids beneath and "
                       "a clean base (p. 10-22): "
                       + ("as the engineer states (rock_base_conditions_met)"
                          if analysis.rock_base_conditions_met else
                          "taken as met, not stated")
                       + "; a fractured rock mass takes the Hoek-Brown base, "
                       "Eqs. 10-24 to 10-29 (the choices give it, or the "
                       "inputs it needs)"
                       + (f"; RQD given as {RQD:.0f}%" if RQD < 100 else "")),
            ))
        else:
            eb_items.append(CalcStep(
                title="Unit End Bearing (Rock Tip), q_b/q_u by RQD",
                equation=f"q_b = {mult:.2f} \u00d7 q_u (RQD = {RQD:.0f}%)",
                substitution=f"q_b = {mult:.2f} \u00d7 {qu:,.1f}",
                result_name="q_b",
                result_value=f"{qb:,.1f}",
                result_unit="kPa",
                reference=("rock_base_method 'rqd': this tool's earlier rock "
                           "base, cited to GEC-10 (2010 edition) Sec. "
                           "13.3.4.4, not in the reference library; GEC-10 "
                           "(2018) uses N*cr = 2.5 (Eq. 10-23, p. 10-22)"),
                notes=_rqd_note(RQD),
            ))
        eb_items.append(CalcStep(
            title="End Bearing Capacity",
            equation="Q_tip = q_b \u00d7 A_tip",
            substitution=f"Q_tip = {qb:,.1f} \u00d7 {base_area:.4f}",
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
        reference="GEC-10 Eq. 10-2 (p. 10-9), resistances unfactored",
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


#: Shaft lengths in the capacity-against-length table.
CAPACITY_TABLE_POINTS = 20


def get_tables(result, analysis) -> list:
    """SHAFT-style tables: the result's own (side resistance per layer,
    load transfer at the ultimate load, capacity summary), the ultimate
    capacity against shaft length (``analysis.capacity_vs_depth``: the same
    analysis re-run at each length), and the soil layers as input (CSV
    only)."""
    from geotech_common.tables import CalcTable

    tables = list(result.get_tables())

    length = analysis.shaft.length
    top = max(length, analysis.soil.total_thickness)
    low = min(3.0, length)
    if top > low:
        skipped = []
        rows = [[r["depth_m"], r["Q_skin_kN"], r["Q_tip_kN"],
                 r["Q_ultimate_kN"]]
                for r in analysis.capacity_vs_depth(
                    depth_min=low, depth_max=top,
                    n_points=CAPACITY_TABLE_POINTS, skipped=skipped)]
        note = ("The same analysis re-run with the shaft length set to "
                "each value (SHAFT's capacity against depth); FS is not "
                "applied")
        if skipped:
            from drilled_shaft.capacity import skipped_depths_note
            note += ". " + skipped_depths_note(skipped,
                                               CAPACITY_TABLE_POINTS)
        if rows:
            tables.append(CalcTable(
                title="Ultimate capacity against shaft length",
                columns=[("Shaft length L", "m", ".2f"),
                         ("Side resistance Q_s", "kN", ".1f"),
                         ("Base resistance Q_b", "kN", ".1f"),
                         ("Ultimate capacity Q_ult", "kN", ".1f")],
                rows=rows,
                note=note,
                name="capacity_vs_length",
            ))

    layer_rows, depth = [], 0.0
    for i, layer in enumerate(analysis.soil.layers, 1):
        layer_rows.append([
            i, depth, depth + layer.thickness, layer.soil_type,
            layer.description, layer.unit_weight, layer.cu, layer.phi,
            layer.N60, layer.qu, layer.RQD])
        depth += layer.thickness
    if layer_rows:
        gwt = analysis.soil.gwt_depth
        tables.append(CalcTable(
            title="Soil layers",
            columns=["Layer", ("Top", "m"), ("Bottom", "m"), "Soil type",
                     "Description", ("Unit weight γ", "kN/m³"),
                     ("Undrained shear strength c_u", "kPa"),
                     ("Friction angle φ", "deg"), "N60",
                     ("Unconfined compressive strength q_u", "kPa"),
                     ("RQD", "%")],
            rows=layer_rows,
            note=("The soil layers as input, depths below the ground "
                  "surface; groundwater at "
                  + (f"{gwt:g} m" if gwt is not None else "none given")),
            name="soil_layers", printed=False,
        ))
    return tables


# ── Helper functions ──────────────────────────────────────────────

def _base(shaft, soil):
    """(name, diameter m, area m\u00b2) of the base the end bearing acts on,
    as DrillShaftAnalysis.compute takes it: the bell's, the rock socket's
    with the tip in rock (GEC-10 Example 10-4, GM-221), else the shaft's."""
    in_rock = soil.layer_at_depth(shaft.length - 0.01).soil_type == "rock"
    name = ("D_bell" if shaft.bell_diameter is not None
            else "D_sock" if in_rock and shaft.socket_diameter is not None
            else "D")
    return name, shaft.base_diameter(in_rock), shaft.base_area(in_rock)


def _gec10_layer_step(i, desc, lb, perimeter, pa) -> CalcStep:
    """A cohesionless layer's side resistance by GEC-10 (2018) Eqs. 10-5
    to 10-13, each step with the layer's own numbers (GM-241)."""
    st = lb["beta_steps"]
    phi = st["phi_deg"]
    z_top, z_bot = lb["depth_top_m"], lb["depth_bottom_m"]
    eff_top = lb.get("effective_top_m", z_top)
    eff_bot = lb.get("effective_bottom_m", z_bot)
    sv = lb.get("sigma_v_kPa", 0.0)
    if st["phi_from"] == "N1_60":
        lines = [f"\u03c6' = 27.5 + 9.2 log (N_1)_60 = {phi:.1f}\u00b0"]
    elif "phi_from_N1_60_deg" in st:
        lines = [f"\u03c6' = {phi:.1f}\u00b0 (given; 27.5 + 9.2 log "
                 f"(N_1)_60 would give {st['phi_from_N1_60_deg']:.1f}\u00b0)"]
    else:
        lines = [f"\u03c6' = {phi:.1f}\u00b0"]
    if st["OCR_from"] == "OCR":
        lines.append(f"OCR = {st['OCR']:.3g} (given)")
    else:
        corr = st["sigma_p_correlation"]
        if corr == "gravel":
            lines.append(f"\u03c3'_p = {pa:g} \u00d7 0.15 \u00d7 "
                         f"{st['N60']:g} = {st['sigma_p_kPa']:.1f} kPa")
        else:
            m = 0.8 if corr == "silty_sand" else 0.6
            lines.append(f"\u03c3'_p = {pa:g} \u00d7 0.47 \u00d7 "
                         f"{st['N60']:g}^{m:g} = {st['sigma_p_kPa']:.1f} kPa")
        lines.append(f"OCR = {st['sigma_p_kPa']:.1f} / "
                     f"{st['sigma_v_for_OCR_kPa']:.1f} = {st['OCR']:.3f}")
    lines += [
        f"K_o = (1 - sin {phi:.1f}\u00b0) \u00d7 {st['OCR']:.3f}^(sin "
        f"{phi:.1f}\u00b0) = {st['Ko']:.3f} (K_p = {st['Kp']:.3f})",
        f"\u03b2 = {st['K_over_Ko']:.3g} \u00d7 "
        f"{min(st['Ko'], st['Kp']):.3f} \u00d7 tan {phi:.1f}\u00b0 = "
        f"{st['beta']:.3f}",
        f"f_SN = {st['beta']:.3f} \u00d7 {sv:.1f} = {lb['fs_kPa']:.1f} kPa",
        f"R_SN = {lb['fs_kPa']:.1f} \u00d7 {perimeter:.3f} \u00d7 "
        f"{eff_bot - eff_top:.2f}"]
    notes = [f"Effective zone: {eff_top:.1f}-{eff_bot:.1f} m, \u03c3'_v the "
             "layer's average"]
    if st["OCR_from"] == "sigma_v_ref":
        notes.append("OCR on the given reference (pre-scour) \u03c3'_v")
    if st.get("shallow_limit"):
        notes.append("\u03c3'_v for the OCR raised to its value at 7.5 ft "
                     "(GEC-10 pp. 10-11, 10-12)")
    return CalcStep(
        title=(f"Layer {i}: {desc} ({z_top:.1f}-{z_bot:.1f} m) -- Beta "
               "Method, GEC-10 (2018) Eq. 10-13"),
        equation=("\u03b2 = (K/K_o)(1 - sin \u03c6')(\u03c3'_p/\u03c3'_v)"
                  "^(sin \u03c6') tan \u03c6'\n"
                  "f_SN = \u03b2 \u03c3'_v,  R_SN = f_SN \u00d7 P \u00d7 "
                  "\u0394z"),
        substitution="\n".join(lines),
        result_name="Q_s",
        result_value=f"{lb['side_resistance_kN']:,.1f}",
        result_unit="kN",
        notes="; ".join(notes),
        reference="GEC-10 (FHWA-NHI-18-024) Eqs. 10-5 to 10-13",
    )


def _alpha_lines(lb, pa) -> str:
    """A cohesive layer's alpha with its own numbers (GM-244)."""
    st = lb.get("alpha_steps")
    fs = lb.get("fs_kPa", 0.0)
    if not st:
        return f"f_s = {fs:.1f} kPa"
    su, used, a = st["su_kPa"], st["su_used_kPa"], st["alpha"]
    if st["alpha_method"] != "rational":
        return (f"c_u/p_a = {su:.1f} / {pa:g} = {st['su_over_pa']:.3f}; "
                f"\u03b1 = {a:.3f}\nf_s = {a:.3f} \u00d7 {su:.1f} = "
                f"{fs:.1f} kPa")
    lines = []
    if st.get("su_test_type", "ciuc") != "ciuc":
        eq = "10-16" if st["su_test_type"] == "uc" else "10-17"
        lines.append(f"s_u(CIUC) = {used:.1f} kPa (Eq. {eq} on c_u = "
                     f"{su:.1f} kPa, \u03c3'_vo = "
                     f"{lb.get('sigma_v_kPa', 0.0):.1f} kPa)")
    else:
        lines.append(f"s_u(CIUC) = c_u = {used:.1f} kPa")
    lines += [f"\u03b1 = 0.30 + 0.17 / ({used:.1f} / {pa:g}) = {a:.3f}",
              f"f_SN = {a:.3f} \u00d7 {used:.1f} = {fs:.1f} kPa"]
    return "\n".join(lines)


def _layer_properties_str(layer) -> str:
    """Format key layer properties as a compact string."""
    parts = [f"\u03b3 = {layer.unit_weight:.1f} kN/m\u00b3"]
    if layer.soil_type == "cohesive":
        parts.append(f"c_u = {layer.cu:.0f} kPa")
    elif layer.soil_type == "cohesionless":
        if layer.phi > 0:
            parts.append(f"\u03c6 = {layer.phi:.0f}\u00b0")
        if layer.N60 > 0:
            parts.append(f"N_60 = {layer.N60:.0f}")
        if layer.N1_60 > 0:
            parts.append(f"(N_1)_60 = {layer.N1_60:.0f}")
        if layer.OCR > 0:
            parts.append(f"OCR = {layer.OCR:g}")
        if layer.sigma_v_ref > 0:
            parts.append(f"\u03c3'_v,ref = {layer.sigma_v_ref:.1f} kPa")
        if layer.sigma_p_correlation != "clean_sand":
            parts.append(f"\u03c3'_p by {layer.sigma_p_correlation}")
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

    # Value labels on bars, set off by points and ringed, so the Q_ult
    # line stops short of a value that reaches it (a component near the
    # whole capacity ends at that line; GM-194)
    from geotech_common.plotting import halo
    notes = []
    for bar, val in zip(bars, values):
        if val > 0:
            pct = 100 * val / result.Q_ultimate if result.Q_ultimate > 0 else 0
            notes.append(ax.annotate(
                f"{val:,.0f} kN ({pct:.0f}%)",
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=9,
            ))
    halo(notes)

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
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(labelsize=9)
    # the key under the axes: inside them the Q_ult line ran through it
    # (GM-194)
    from geotech_common.plotting import key_under_axes
    key_under_axes(fig, ax, fontsize=9)
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

    # Exclusion zone shading (top), over each layer its zone reaches
    # (none in cohesionless soil unless top_exclusion is given, GM-241)
    top_spans = []
    depth = 0.0
    for layer in analysis.soil.layers:
        top, depth = depth, depth + layer.thickness
        bottom = min(depth, analysis._top_of_side(layer.soil_type),
                     analysis.shaft.length)
        if bottom > top:
            ax.axhspan(top, bottom, alpha=0.1, color="#ef4444")
            top_spans.append((top, bottom))
    from drilled_shaft.capacity import fractured_rock_zone
    not_socket = fractured_rock_zone(analysis.shaft, analysis.soil)
    if not_socket is not None:
        ax.axhspan(*not_socket, alpha=0.1, color="#ef4444", hatch="//")

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
    if top_spans:
        legend_handles.append(mpatches.Patch(
            color="#ef4444", alpha=0.15,
            label="Top exclusion (" + ", ".join(
                f"{a:.1f}-{b:.1f} m" for a, b in top_spans) + ")",
        ))
    if not_socket is not None:
        legend_handles.append(mpatches.Patch(
            facecolor="#ef4444", alpha=0.15, hatch="//",
            label=(f"Rock not part of the socket "
                   f"({not_socket[0]:.1f}-{not_socket[1]:.1f} m)"),
        ))
    # the key under the axes: inside them it covered the bars and the
    # shaft tip (GM-194)
    from geotech_common.plotting import key_under_axes
    key_under_axes(fig, ax, handles=legend_handles)
    return fig
