"""
Calculation package steps for ground improvement analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation.  Supports all four
ground improvement methods plus the feasibility screening:

- Aggregate piers (Barksdale & Bachus, Priebe)
- Wick drains / PVD (Barron/Hansbo radial consolidation)
- Surcharge preloading (with/without drains)
- Vibro-compaction feasibility (GEC-13)
- Feasibility evaluation (decision support)

The ``analysis`` parameter is a dict whose ``"method"`` key selects
the branch, and the remaining keys are the input parameters echoed
in the calc package.

References:
    FHWA GEC-13: Ground Modification Methods Reference Manual
    Barron (1948) / Hansbo (1981) — radial consolidation
    Barksdale & Bachus (1983) — aggregate pier / stone column design
    Priebe (1995) — vibro-replacement improvement factors
"""

import math
from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Ground Improvement Analysis"

REFERENCES = [
    "FHWA GEC-13 (FHWA-NHI-16-027): Ground Modification Methods "
    "Reference Manual, 2017.",
    "FHWA NHI-06-019/020: Ground Improvement Methods, Vols. I & II.",
    'Barksdale, R.D. & Bachus, R.C. (1983). "Design and Construction of '
    'Stone Columns." FHWA/RD-83/026.',
    'Priebe, H.J. (1995). "The Design of Vibro Replacement." '
    "Ground Engineering, Dec 1995, pp. 31-37.",
    'Barron, R.A. (1948). "Consolidation of Fine-Grained Soils by Drain '
    'Wells." Trans. ASCE, Vol. 113, pp. 718-742.',
    'Hansbo, S. (1981). "Consolidation of Fine-Grained Soils by '
    'Prefabricated Drains." Proc. 10th ICSMFE, Stockholm, Vol. 3.',
]


# =====================================================================
# Public API
# =====================================================================

def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for the ground improvement calc package.

    Parameters
    ----------
    result : AggregatePierResult | WickDrainResult | SurchargeResult |
             VibroResult | FeasibilityResult
        Computed results from any of the five analysis functions.
    analysis : dict
        Must contain ``"method"`` key (one of ``"aggregate_piers"``,
        ``"wick_drains"``, ``"surcharge"``, ``"vibro"``, ``"feasibility"``).
        Remaining keys mirror the analyse_*() call arguments.

    Returns
    -------
    list of InputItem
    """
    method = analysis.get("method", "")
    dispatch = {
        "aggregate_piers": _inputs_aggregate_piers,
        "wick_drains": _inputs_wick_drains,
        "surcharge": _inputs_surcharge,
        "vibro": _inputs_vibro,
        "feasibility": _inputs_feasibility,
    }
    fn = dispatch.get(method)
    if fn is None:
        return [InputItem("Method", "Analysis method", method, "")]
    return fn(result, analysis)


def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections.

    Parameters
    ----------
    result : AggregatePierResult | WickDrainResult | SurchargeResult |
             VibroResult | FeasibilityResult
        Computed results.
    analysis : dict
        See ``get_input_summary`` for required keys.

    Returns
    -------
    list of CalcSection
    """
    method = analysis.get("method", "")
    dispatch = {
        "aggregate_piers": _steps_aggregate_piers,
        "wick_drains": _steps_wick_drains,
        "surcharge": _steps_surcharge,
        "vibro": _steps_vibro,
        "feasibility": _steps_feasibility,
    }
    fn = dispatch.get(method)
    if fn is None:
        return []
    return fn(result, analysis)


def get_figures(result, analysis) -> List[FigureData]:
    """Generate figures for the ground improvement calc package.

    Parameters
    ----------
    result : AggregatePierResult | WickDrainResult | SurchargeResult |
             VibroResult | FeasibilityResult
        Computed results.
    analysis : dict
        See ``get_input_summary`` for required keys.

    Returns
    -------
    list of FigureData
    """
    method = analysis.get("method", "")
    dispatch = {
        "aggregate_piers": _figures_aggregate_piers,
        "wick_drains": _figures_wick_drains,
        "surcharge": _figures_surcharge,
        "vibro": _figures_vibro,
        "feasibility": _figures_feasibility,
    }
    fn = dispatch.get(method)
    if fn is None:
        return []
    return fn(result, analysis)


# =====================================================================
# AGGREGATE PIERS
# =====================================================================

def _inputs_aggregate_piers(result, analysis) -> List[InputItem]:
    r = result
    items = [
        InputItem("Method", "Analysis method", "Aggregate Piers", ""),
        InputItem("d_c", "Column diameter", r.column_diameter_m, "m"),
        InputItem("s", "Column spacing (c-c)", r.column_spacing_m, "m"),
        InputItem("Pattern", "Layout pattern", r.pattern.capitalize(), ""),
        InputItem("E_c", "Column modulus",
                  analysis.get("E_column", 80000.0), "kPa"),
        InputItem("E_s", "Soil modulus",
                  analysis.get("E_soil", 5000.0), "kPa"),
        InputItem("n", "Stress concentration ratio",
                  r.stress_concentration_ratio, ""),
    ]
    if r.unreinforced_bearing_kPa > 0:
        items.append(InputItem("q_unr", "Unreinforced bearing capacity",
                               r.unreinforced_bearing_kPa, "kPa"))
    if r.settlement_unreinforced_mm > 0:
        items.append(InputItem("S_unr", "Unreinforced settlement",
                               r.settlement_unreinforced_mm, "mm"))
    return items


def _steps_aggregate_piers(result, analysis) -> List[CalcSection]:
    r = result
    sections = []

    # -- Section 1: Area Replacement Ratio ----------------------------
    geo_items = []

    dc = r.column_diameter_m
    s = r.column_spacing_m
    pat = r.pattern

    Ac_val = math.pi / 4.0 * dc ** 2
    if pat == "triangular":
        A_trib_val = math.sqrt(3) / 2.0 * s ** 2
        trib_eq = "A_trib = (\u221a3 / 2) \u00d7 s\u00b2"
        trib_sub = (f"A_trib = (\u221a3 / 2) \u00d7 {s:.2f}\u00b2 "
                    f"= {A_trib_val:.4f}")
    else:
        A_trib_val = s ** 2
        trib_eq = "A_trib = s\u00b2"
        trib_sub = f"A_trib = {s:.2f}\u00b2 = {A_trib_val:.4f}"

    geo_items.append(CalcStep(
        title="Column Cross-Sectional Area",
        equation="A_c = \u03c0 / 4 \u00d7 d_c\u00b2",
        substitution=f"A_c = \u03c0 / 4 \u00d7 {dc:.3f}\u00b2",
        result_name="A_c",
        result_value=f"{Ac_val:.4f}",
        result_unit="m\u00b2",
    ))

    geo_items.append(CalcStep(
        title=f"Tributary Area ({pat} pattern)",
        equation=trib_eq,
        substitution=trib_sub,
        result_name="A_trib",
        result_value=f"{A_trib_val:.4f}",
        result_unit="m\u00b2",
        reference="Barksdale & Bachus (1983)",
    ))

    geo_items.append(CalcStep(
        title="Area Replacement Ratio",
        equation="a_s = A_c / A_trib",
        substitution=f"a_s = {Ac_val:.4f} / {A_trib_val:.4f}",
        result_name="a_s",
        result_value=f"{r.area_replacement_ratio:.4f}",
        reference="GEC-13, Section 7.3",
    ))

    sections.append(CalcSection(
        title="Geometry & Area Replacement", items=geo_items
    ))

    # -- Section 2: Settlement Reduction Factor -----------------------
    srf_items = []

    n = r.stress_concentration_ratio
    a_s = r.area_replacement_ratio

    srf_items.append(CalcStep(
        title="Settlement Reduction Factor (SRF)",
        equation="SRF = 1 / [1 + a_s \u00d7 (n - 1)]",
        substitution=f"SRF = 1 / [1 + {a_s:.4f} \u00d7 ({n:.1f} - 1)]",
        result_name="SRF",
        result_value=f"{r.settlement_reduction_factor:.4f}",
        reference="Priebe (1995); GEC-13 Eq. 7-2",
        notes="SRF < 1 indicates settlement improvement.",
    ))

    # -- Section 3: Composite Modulus ---------------------------------
    E_c = analysis.get("E_column", 80000.0)
    E_s = analysis.get("E_soil", 5000.0)

    srf_items.append(CalcStep(
        title="Composite Modulus of Improved Ground",
        equation="E_comp = a_s \u00d7 E_c + (1 - a_s) \u00d7 E_s",
        substitution=(f"E_comp = {a_s:.4f} \u00d7 {E_c:.0f} "
                      f"+ (1 - {a_s:.4f}) \u00d7 {E_s:.0f}"),
        result_name="E_comp",
        result_value=f"{r.composite_modulus_kPa:.0f}",
        result_unit="kPa",
        reference="GEC-13, Eq. 7-3",
    ))

    sections.append(CalcSection(
        title="Settlement Reduction & Composite Modulus", items=srf_items
    ))

    # -- Section 4: Bearing Capacity Improvement ----------------------
    if r.unreinforced_bearing_kPa > 0:
        bc_items = []
        bc_items.append(CalcStep(
            title="Improved Bearing Capacity",
            equation="q_improved = q_unr \u00d7 [1 + a_s \u00d7 (n - 1)]",
            substitution=(
                f"q_improved = {r.unreinforced_bearing_kPa:.1f} "
                f"\u00d7 [1 + {a_s:.4f} \u00d7 ({n:.1f} - 1)]"
            ),
            result_name="q_improved",
            result_value=f"{r.improved_bearing_kPa:.1f}",
            result_unit="kPa",
            reference="Priebe (1995) improvement factor",
        ))

        bc_items.append(CheckItem(
            description="Bearing capacity improvement",
            demand=r.unreinforced_bearing_kPa,
            demand_label="q_unreinforced",
            capacity=r.improved_bearing_kPa,
            capacity_label="q_improved",
            unit="kPa",
            passes=r.improved_bearing_kPa >= r.unreinforced_bearing_kPa,
        ))

        sections.append(CalcSection(
            title="Bearing Capacity Improvement", items=bc_items
        ))

    # -- Section 5: Settlement Improvement ----------------------------
    if r.settlement_unreinforced_mm > 0:
        settle_items = []
        settle_items.append(CalcStep(
            title="Improved Settlement",
            equation="S_improved = SRF \u00d7 S_unreinforced",
            substitution=(
                f"S_improved = {r.settlement_reduction_factor:.4f} "
                f"\u00d7 {r.settlement_unreinforced_mm:.1f}"
            ),
            result_name="S_improved",
            result_value=f"{r.settlement_improved_mm:.1f}",
            result_unit="mm",
            reference="Priebe (1995)",
        ))

        settle_items.append(TableData(
            title="Settlement Comparison",
            headers=["Condition", "Settlement (mm)"],
            rows=[
                ["Unreinforced", f"{r.settlement_unreinforced_mm:.1f}"],
                ["With aggregate piers", f"{r.settlement_improved_mm:.1f}"],
                ["Reduction",
                 f"{r.settlement_unreinforced_mm - r.settlement_improved_mm:.1f} "
                 f"({(1 - r.settlement_reduction_factor) * 100:.0f}%)"],
            ],
        ))

        sections.append(CalcSection(
            title="Settlement Improvement", items=settle_items
        ))

    return sections


def _figures_aggregate_piers(result, analysis) -> List[FigureData]:
    figures = []
    r = result

    # Only generate figures when there is enough data
    if r.settlement_unreinforced_mm <= 0 and r.unreinforced_bearing_kPa <= 0:
        return figures

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Figure 1: SRF sensitivity to stress concentration ratio n
        fig1 = _plot_srf_vs_n(r)
        b64_1 = figure_to_base64(fig1, dpi=150)
        plt.close(fig1)
        figures.append(FigureData(
            title="Settlement Reduction Factor vs Stress Concentration Ratio",
            image_base64=b64_1,
            caption=(
                f"Figure 1: SRF vs stress concentration ratio n for "
                f"a_s = {r.area_replacement_ratio:.4f}. "
                f"Design point: n = {r.stress_concentration_ratio:.1f}, "
                f"SRF = {r.settlement_reduction_factor:.3f}."
            ),
            width_percent=70,
        ))

        # Figure 2: Before/after bar chart
        if (r.settlement_unreinforced_mm > 0 or
                r.unreinforced_bearing_kPa > 0):
            fig2 = _plot_pier_comparison(r)
            b64_2 = figure_to_base64(fig2, dpi=150)
            plt.close(fig2)
            figures.append(FigureData(
                title="Before / After Improvement Comparison",
                image_base64=b64_2,
                caption=(
                    "Figure 2: Comparison of unreinforced vs improved "
                    "conditions with aggregate piers."
                ),
                width_percent=70,
            ))

    except ImportError:
        pass

    return figures


# =====================================================================
# WICK DRAINS
# =====================================================================

def _inputs_wick_drains(result, analysis) -> List[InputItem]:
    r = result
    items = [
        InputItem("Method", "Analysis method", "Wick Drains (PVD)", ""),
        InputItem("s", "Drain spacing (c-c)", r.drain_spacing_m, "m"),
        InputItem("Pattern", "Layout pattern", r.pattern.capitalize(), ""),
        InputItem("d_w", "Equivalent drain diameter",
                  r.equivalent_drain_diameter_m, "m"),
        InputItem("d_e", "Influence zone diameter",
                  r.influence_diameter_m, "m"),
        InputItem("c_h", "Horizontal coeff. of consolidation",
                  r.ch_m2_per_year, "m\u00b2/yr"),
        InputItem("c_v", "Vertical coeff. of consolidation",
                  r.cv_m2_per_year, "m\u00b2/yr"),
        InputItem("H_dr", "Vertical drainage path",
                  analysis.get("Hdr", ""), "m"),
        InputItem("t", "Analysis time", r.time_years, "years"),
    ]
    smear = analysis.get("smear_ratio", 2.0)
    kh_ks = analysis.get("kh_ks_ratio", 2.0)
    if smear > 1.0:
        items.append(InputItem("s_r", "Smear zone ratio (d_s/d_w)",
                               smear, ""))
        items.append(InputItem("k_h/k_s", "Permeability ratio",
                               kh_ks, ""))
    return items


def _steps_wick_drains(result, analysis) -> List[CalcSection]:
    r = result
    sections = []

    # -- Section 1: Drain Geometry ------------------------------------
    geo_items = []

    geo_items.append(CalcStep(
        title="Influence Zone Diameter",
        equation=("d_e = 1.05 \u00d7 s  (triangular)\n"
                  "d_e = 1.13 \u00d7 s  (square)"),
        substitution=(
            f"d_e = {'1.05' if r.pattern == 'triangular' else '1.13'} "
            f"\u00d7 {r.drain_spacing_m:.2f}"
        ),
        result_name="d_e",
        result_value=f"{r.influence_diameter_m:.3f}",
        result_unit="m",
        reference="Barron (1948); Hansbo (1981)",
    ))

    geo_items.append(CalcStep(
        title="Spacing Ratio",
        equation="n = d_e / d_w",
        substitution=f"n = {r.influence_diameter_m:.3f} / {r.equivalent_drain_diameter_m:.4f}",
        result_name="n",
        result_value=f"{r.spacing_ratio_n:.1f}",
    ))

    sections.append(CalcSection(
        title="Drain Geometry", items=geo_items
    ))

    # -- Section 2: Drain Function F(n) -------------------------------
    fn_items = []

    smear = analysis.get("smear_ratio", 2.0)
    kh_ks = analysis.get("kh_ks_ratio", 2.0)

    if smear <= 1.0:
        fn_eq = "F(n) = ln(n) - 0.75"
        fn_sub = f"F(n) = ln({r.spacing_ratio_n:.1f}) - 0.75"
    else:
        fn_eq = "F(n) = ln(n/s_r) + (k_h/k_s) \u00d7 ln(s_r) - 0.75"
        fn_sub = (f"F(n) = ln({r.spacing_ratio_n:.1f}/{smear:.1f}) + "
                  f"{kh_ks:.1f} \u00d7 ln({smear:.1f}) - 0.75")

    fn_items.append(CalcStep(
        title="Barron/Hansbo Drain Function",
        equation=fn_eq,
        substitution=fn_sub,
        result_name="F(n)",
        result_value=f"{r.F_n:.3f}",
        reference="Hansbo (1981)",
        notes="Includes smear zone correction." if smear > 1.0 else "",
    ))

    sections.append(CalcSection(
        title="Drain Function F(n)", items=fn_items
    ))

    # -- Section 3: Degree of Consolidation ---------------------------
    consol_items = []

    Hdr = analysis.get("Hdr", 0.0)
    t = r.time_years

    # Radial time factor
    Tr_val = r.ch_m2_per_year * t / r.influence_diameter_m ** 2 if r.influence_diameter_m > 0 else 0.0
    consol_items.append(CalcStep(
        title="Radial Time Factor",
        equation="T_r = c_h \u00d7 t / d_e\u00b2",
        substitution=(f"T_r = {r.ch_m2_per_year:.2f} \u00d7 {t:.3f} "
                      f"/ {r.influence_diameter_m:.3f}\u00b2"),
        result_name="T_r",
        result_value=f"{Tr_val:.4f}",
    ))

    # Radial consolidation
    consol_items.append(CalcStep(
        title="Radial Degree of Consolidation",
        equation="U_r = 1 - exp(-8 \u00d7 T_r / F(n))",
        substitution=f"U_r = 1 - exp(-8 \u00d7 {Tr_val:.4f} / {r.F_n:.3f})",
        result_name="U_r",
        result_value=f"{r.Ur_percent:.1f}",
        result_unit="%",
        reference="Barron (1948); Hansbo (1981)",
    ))

    # Vertical consolidation
    if Hdr > 0:
        Tv_val = r.cv_m2_per_year * t / Hdr ** 2 if Hdr > 0 else 0.0
        consol_items.append(CalcStep(
            title="Vertical Time Factor",
            equation="T_v = c_v \u00d7 t / H_dr\u00b2",
            substitution=(f"T_v = {r.cv_m2_per_year:.2f} \u00d7 {t:.3f} "
                          f"/ {Hdr:.2f}\u00b2"),
            result_name="T_v",
            result_value=f"{Tv_val:.4f}",
        ))

    consol_items.append(CalcStep(
        title="Vertical Degree of Consolidation",
        equation="U_v from Terzaghi 1-D theory",
        substitution="",
        result_name="U_v",
        result_value=f"{r.Uv_percent:.1f}",
        result_unit="%",
    ))

    # Combined
    consol_items.append(CalcStep(
        title="Combined Degree of Consolidation",
        equation="U_total = 1 - (1 - U_v/100) \u00d7 (1 - U_r/100)",
        substitution=(f"U_total = 1 - (1 - {r.Uv_percent:.1f}/100) "
                      f"\u00d7 (1 - {r.Ur_percent:.1f}/100)"),
        result_name="U_total",
        result_value=f"{r.U_total_percent:.1f}",
        result_unit="%",
        reference="Carrillo (1942) combined consolidation",
    ))

    sections.append(CalcSection(
        title="Degree of Consolidation", items=consol_items
    ))

    # -- Section 4: Summary Table -------------------------------------
    summary_items = []
    summary_items.append(TableData(
        title="Consolidation Summary",
        headers=["Parameter", "Value"],
        rows=[
            ["Drain spacing", f"{r.drain_spacing_m:.2f} m ({r.pattern})"],
            ["Spacing ratio n", f"{r.spacing_ratio_n:.1f}"],
            ["F(n)", f"{r.F_n:.3f}"],
            ["Radial consolidation U_r", f"{r.Ur_percent:.1f}%"],
            ["Vertical consolidation U_v", f"{r.Uv_percent:.1f}%"],
            ["Combined consolidation U_total", f"{r.U_total_percent:.1f}%"],
            ["Analysis time", f"{r.time_years:.2f} years"],
        ],
    ))
    sections.append(CalcSection(
        title="Results Summary", items=summary_items
    ))

    return sections


def _figures_wick_drains(result, analysis) -> List[FigureData]:
    figures = []
    r = result

    if r.time_settlement_curve is None or len(r.time_settlement_curve) < 2:
        return figures

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = _plot_consolidation_vs_time(r)
        b64 = figure_to_base64(fig, dpi=150)
        plt.close(fig)
        figures.append(FigureData(
            title="Degree of Consolidation vs Time",
            image_base64=b64,
            caption=(
                f"Figure 1: Combined degree of consolidation vs time "
                f"for s = {r.drain_spacing_m:.2f} m ({r.pattern}). "
                f"At t = {r.time_years:.2f} yr, U_total = {r.U_total_percent:.1f}%."
            ),
            width_percent=75,
        ))

    except ImportError:
        pass

    return figures


# =====================================================================
# SURCHARGE PRELOADING
# =====================================================================

def _inputs_surcharge(result, analysis) -> List[InputItem]:
    r = result
    items = [
        InputItem("Method", "Analysis method", "Surcharge Preloading", ""),
        InputItem("q_s", "Surcharge pressure", r.surcharge_kPa, "kPa"),
        InputItem("S_ult", "Ultimate settlement",
                  r.settlement_ultimate_mm, "mm"),
        InputItem("U_target", "Target consolidation",
                  r.target_U_percent, "%"),
        InputItem("c_v", "Vertical coeff. of consolidation",
                  analysis.get("cv", ""), "m\u00b2/yr"),
        InputItem("H_dr", "Vertical drainage path",
                  analysis.get("Hdr", ""), "m"),
    ]
    if r.uses_wick_drains:
        items.append(InputItem("Drains", "Uses wick drains", "Yes", ""))
        items.append(InputItem("c_h", "Horizontal coeff. of consolidation",
                               analysis.get("ch", ""), "m\u00b2/yr"))
        items.append(InputItem("s_drain", "Drain spacing",
                               analysis.get("drain_spacing", ""), "m"))
    else:
        items.append(InputItem("Drains", "Uses wick drains", "No", ""))

    if r.equivalent_sigma_p_kPa > 0:
        items.append(InputItem("\u03c3_v0", "Current effective stress",
                               analysis.get("sigma_v0", ""), "kPa"))
    return items


def _steps_surcharge(result, analysis) -> List[CalcSection]:
    r = result
    sections = []

    # -- Section 1: Surcharge Design ----------------------------------
    design_items = []

    design_items.append(CalcStep(
        title="Ultimate Settlement Under Surcharge",
        equation="S_ult (from consolidation analysis)",
        substitution="",
        result_name="S_ult",
        result_value=f"{r.settlement_ultimate_mm:.1f}",
        result_unit="mm",
        notes="Input from separate consolidation settlement analysis.",
    ))

    if r.equivalent_sigma_p_kPa > 0:
        sigma_v0 = analysis.get("sigma_v0", 0.0)
        design_items.append(CalcStep(
            title="Equivalent Preconsolidation Pressure",
            equation="\u03c3'_p = \u03c3'_v0 + q_surcharge",
            substitution=(f"\u03c3'_p = {sigma_v0:.1f} + "
                          f"{r.surcharge_kPa:.1f}"),
            result_name="\u03c3'_p",
            result_value=f"{r.equivalent_sigma_p_kPa:.1f}",
            result_unit="kPa",
            reference="GEC-13, Section 3",
        ))

    sections.append(CalcSection(
        title="Surcharge Design Parameters", items=design_items
    ))

    # -- Section 2: Time to Target Consolidation ----------------------
    time_items = []

    if r.uses_wick_drains:
        time_items.append(CalcStep(
            title="Time to Target Consolidation (with wick drains)",
            equation=(
                "Bisection on U_total(t) = 1 - (1-U_v)(1-U_r)\n"
                "where U_v = Terzaghi, U_r = Barron/Hansbo"
            ),
            substitution=f"Find t such that U_total = {r.target_U_percent:.0f}%",
            result_name="t_target",
            result_value=f"{r.time_to_target_years:.3f}",
            result_unit="years",
            reference="Carrillo (1942) combined consolidation",
        ))
    else:
        time_items.append(CalcStep(
            title="Time to Target Consolidation (vertical only)",
            equation=(
                "T_v = -(\u03c0/4) \u00d7 [1 - U/100]\u00b2  (approx.)\n"
                "t = T_v \u00d7 H_dr\u00b2 / c_v"
            ),
            substitution=f"Find t such that U_v = {r.target_U_percent:.0f}%",
            result_name="t_target",
            result_value=f"{r.time_to_target_years:.3f}",
            result_unit="years",
            reference="Terzaghi (1925) 1-D consolidation theory",
        ))

    time_items.append(CalcStep(
        title="Settlement at Target Consolidation",
        equation="S_target = (U_target / 100) \u00d7 S_ult",
        substitution=(f"S_target = ({r.target_U_percent:.0f} / 100) "
                      f"\u00d7 {r.settlement_ultimate_mm:.1f}"),
        result_name="S_target",
        result_value=f"{r.settlement_at_target_mm:.1f}",
        result_unit="mm",
    ))

    sections.append(CalcSection(
        title="Time to Target Consolidation", items=time_items
    ))

    # -- Section 3: Wick Drain Sub-Result -----------------------------
    if r.uses_wick_drains and r.wick_drain_result is not None:
        wd = r.wick_drain_result
        wd_items = []
        wd_items.append(TableData(
            title="Wick Drain Details at Target Time",
            headers=["Parameter", "Value"],
            rows=[
                ["Drain spacing", f"{wd.drain_spacing_m:.2f} m ({wd.pattern})"],
                ["Drain diameter d_w", f"{wd.equivalent_drain_diameter_m:.4f} m"],
                ["Influence diameter d_e", f"{wd.influence_diameter_m:.3f} m"],
                ["Spacing ratio n", f"{wd.spacing_ratio_n:.1f}"],
                ["F(n)", f"{wd.F_n:.3f}"],
                ["Radial U_r", f"{wd.Ur_percent:.1f}%"],
                ["Vertical U_v", f"{wd.Uv_percent:.1f}%"],
                ["Combined U_total", f"{wd.U_total_percent:.1f}%"],
            ],
        ))
        sections.append(CalcSection(
            title="Wick Drain Sub-Analysis", items=wd_items
        ))

    # -- Section 4: Summary -------------------------------------------
    summary_items = []
    summary_items.append(TableData(
        title="Surcharge Preloading Summary",
        headers=["Parameter", "Value"],
        rows=[
            ["Surcharge pressure", f"{r.surcharge_kPa:.1f} kPa"],
            ["Ultimate settlement", f"{r.settlement_ultimate_mm:.1f} mm"],
            ["Target consolidation", f"{r.target_U_percent:.0f}%"],
            ["Time to target", f"{r.time_to_target_years:.3f} years "
                               f"({r.time_to_target_years * 12:.1f} months)"],
            ["Settlement at target", f"{r.settlement_at_target_mm:.1f} mm"],
            ["Uses wick drains", "Yes" if r.uses_wick_drains else "No"],
        ],
    ))

    if r.equivalent_sigma_p_kPa > 0:
        summary_items[0].rows.append(
            ["Equiv. preconsolidation", f"{r.equivalent_sigma_p_kPa:.1f} kPa"]
        )

    sections.append(CalcSection(
        title="Results Summary", items=summary_items
    ))

    return sections


def _figures_surcharge(result, analysis) -> List[FigureData]:
    figures = []
    r = result

    if r.time_settlement_curve is None or len(r.time_settlement_curve) < 2:
        return figures

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = _plot_surcharge_settlement_vs_time(r)
        b64 = figure_to_base64(fig, dpi=150)
        plt.close(fig)
        figures.append(FigureData(
            title="Settlement vs Time Under Surcharge",
            image_base64=b64,
            caption=(
                f"Figure 1: Settlement vs time under {r.surcharge_kPa:.0f} kPa "
                f"surcharge"
                f"{' with wick drains' if r.uses_wick_drains else ''}. "
                f"Target {r.target_U_percent:.0f}% consolidation reached at "
                f"t = {r.time_to_target_years:.2f} yr "
                f"(S = {r.settlement_at_target_mm:.1f} mm)."
            ),
            width_percent=75,
        ))

    except ImportError:
        pass

    return figures


# =====================================================================
# VIBRO-COMPACTION
# =====================================================================

def _inputs_vibro(result, analysis) -> List[InputItem]:
    r = result
    items = [
        InputItem("Method", "Analysis method",
                  "Vibro-Compaction Feasibility", ""),
        InputItem("FC", "Fines content", r.fines_content_percent, "%"),
        InputItem("N_initial", "Initial SPT N", r.initial_N_spt, ""),
        InputItem("N_target", "Target SPT N", r.target_N_spt, ""),
    ]
    D50 = analysis.get("D50")
    if D50 is not None:
        items.append(InputItem("D_50", "Median grain size", D50, "mm"))
    if r.is_feasible and r.recommended_spacing_m > 0:
        items.append(InputItem("Pattern", "Probe pattern",
                               r.probe_pattern.capitalize(), ""))
    return items


def _steps_vibro(result, analysis) -> List[CalcSection]:
    r = result
    sections = []

    # -- Section 1: Feasibility Screening -----------------------------
    screen_items = []

    # Fines content check
    if r.fines_content_percent <= 10:
        fc_status = "Favorable (< 10%)"
    elif r.fines_content_percent <= 15:
        fc_status = "Acceptable (10-15%)"
    elif r.fines_content_percent <= 20:
        fc_status = "Marginal (15-20%)"
    else:
        fc_status = "Not feasible (> 20%)"

    screen_items.append(CalcStep(
        title="Fines Content Screening",
        equation="FC < 10%: favorable; 10-15%: acceptable; "
                 "15-20%: marginal; > 20%: not feasible",
        substitution=f"FC = {r.fines_content_percent:.1f}%",
        result_name="FC status",
        result_value=fc_status,
        reference="GEC-13, Table 5-1; Brown (1977)",
    ))

    # D50 check
    D50 = analysis.get("D50")
    if D50 is not None:
        if D50 >= 0.2:
            d50_status = "Suitable"
        elif D50 >= 0.1:
            d50_status = "Marginal"
        else:
            d50_status = "Too fine"

        screen_items.append(CalcStep(
            title="Grain Size Screening",
            equation="D_50 > 0.2 mm: suitable; 0.1-0.2 mm: marginal; "
                     "< 0.1 mm: not feasible",
            substitution=f"D_50 = {D50:.2f} mm",
            result_name="D_50 status",
            result_value=d50_status,
        ))

    # SPT density check
    if r.initial_N_spt > 0:
        if r.initial_N_spt <= 20:
            spt_status = "Good improvement potential"
        elif r.initial_N_spt <= 25:
            spt_status = "Moderate improvement potential"
        else:
            spt_status = "Already dense (limited potential)"

        screen_items.append(CalcStep(
            title="Density Screening (SPT N)",
            equation="N < 20: good potential; 20-25: moderate; > 25: limited",
            substitution=f"N_initial = {r.initial_N_spt:.0f}",
            result_name="SPT status",
            result_value=spt_status,
        ))

    # Overall feasibility
    screen_items.append(CalcStep(
        title="Overall Feasibility Assessment",
        equation="Vibro-compaction is feasible if all criteria are met",
        substitution="",
        result_name="Feasible",
        result_value="YES" if r.is_feasible else "NO",
        reference="FHWA GEC-13",
    ))

    sections.append(CalcSection(
        title="Feasibility Screening", items=screen_items
    ))

    # -- Section 2: Probe Spacing (if feasible) -----------------------
    if r.is_feasible and r.recommended_spacing_m > 0:
        spacing_items = []

        improvement_ratio = r.target_N_spt / r.initial_N_spt if r.initial_N_spt > 0 else 0
        spacing_items.append(CalcStep(
            title="Improvement Ratio",
            equation="IR = N_target / N_initial",
            substitution=f"IR = {r.target_N_spt:.0f} / {r.initial_N_spt:.0f}",
            result_name="IR",
            result_value=f"{improvement_ratio:.2f}",
            reference="GEC-13, empirical correlation",
        ))

        spacing_items.append(CalcStep(
            title="Recommended Probe Spacing",
            equation="Based on empirical correlation from GEC-13\n"
                     "IR 1.5 \u2192 3.0 m;  IR 4.0 \u2192 1.5 m "
                     "(linear interpolation, adjusted for fines)",
            substitution=f"IR = {improvement_ratio:.2f}, "
                         f"FC = {r.fines_content_percent:.1f}%",
            result_name="s_probe",
            result_value=f"{r.recommended_spacing_m:.2f}",
            result_unit="m",
            reference="FHWA GEC-13, Section 5.4",
        ))

        sections.append(CalcSection(
            title="Probe Spacing Estimate", items=spacing_items
        ))

    # -- Section 3: Notes & Reasons -----------------------------------
    if r.reasons:
        note_items = []
        for i, reason in enumerate(r.reasons, 1):
            note_items.append(f"{i}. {reason}")
        sections.append(CalcSection(
            title="Assessment Notes", items=note_items
        ))

    return sections


def _figures_vibro(result, analysis) -> List[FigureData]:
    """Vibro-compaction has no continuous data to plot."""
    return []


# =====================================================================
# FEASIBILITY EVALUATION
# =====================================================================

def _inputs_feasibility(result, analysis) -> List[InputItem]:
    r = result
    items = [
        InputItem("Method", "Analysis method",
                  "Ground Improvement Feasibility Evaluation", ""),
        InputItem("Soil", "Soil description", r.soil_description, ""),
        InputItem("Problem", "Design problem", r.design_problem, ""),
    ]
    # Echo key input parameters when available
    soil_type = analysis.get("soil_type", "")
    if soil_type:
        items.append(InputItem("Soil type", "Classification", soil_type, ""))
    fc = analysis.get("fines_content")
    if fc is not None:
        items.append(InputItem("FC", "Fines content", fc, "%"))
    N = analysis.get("N_spt")
    if N is not None:
        items.append(InputItem("N_spt", "SPT blow count", N, ""))
    cu = analysis.get("cu_kPa")
    if cu is not None:
        items.append(InputItem("c_u", "Undrained shear strength", cu, "kPa"))
    thick = analysis.get("thickness_m", 0.0)
    if thick > 0:
        items.append(InputItem("H", "Treatment zone thickness", thick, "m"))
    pred = analysis.get("predicted_settlement_mm", 0.0)
    if pred > 0:
        items.append(InputItem("S_pred", "Predicted settlement", pred, "mm"))
    allow = analysis.get("allowable_settlement_mm", 50.0)
    items.append(InputItem("S_allow", "Allowable settlement", allow, "mm"))
    time_c = analysis.get("time_constraint_months", 0.0)
    if time_c > 0:
        items.append(InputItem("t_avail", "Time available", time_c, "months"))
    return items


def _steps_feasibility(result, analysis) -> List[CalcSection]:
    r = result
    sections = []

    # -- Section 1: Applicable Methods --------------------------------
    if r.applicable_methods:
        app_items = []
        rows = [[m, "Applicable"] for m in r.applicable_methods]
        for item in r.not_applicable:
            rows.append([item["method"], f"Not applicable: {item['reason']}"])
        app_items.append(TableData(
            title="Method Screening Results",
            headers=["Method", "Assessment"],
            rows=rows,
        ))
        sections.append(CalcSection(
            title="Method Feasibility Screening", items=app_items
        ))
    else:
        sections.append(CalcSection(
            title="Method Feasibility Screening",
            items=["No standard ground improvement methods appear feasible. "
                   "Consider deep foundations."],
        ))

    # -- Section 2: Preliminary Sizing --------------------------------
    if r.preliminary_sizing:
        sizing_items = []
        for method_key, sizing in r.preliminary_sizing.items():
            if isinstance(sizing, dict):
                rows = [[k, str(v)] for k, v in sizing.items()]
                sizing_items.append(TableData(
                    title=f"Preliminary Sizing: {method_key.replace('_', ' ').title()}",
                    headers=["Parameter", "Value"],
                    rows=rows,
                ))
        sections.append(CalcSection(
            title="Preliminary Sizing Estimates", items=sizing_items
        ))

    # -- Section 3: Recommendations -----------------------------------
    if r.recommendations:
        rec_items = []
        for i, rec in enumerate(r.recommendations, 1):
            rec_items.append(f"{i}. {rec}")
        sections.append(CalcSection(
            title="Recommendations", items=rec_items
        ))

    return sections


def _figures_feasibility(result, analysis) -> List[FigureData]:
    """Feasibility screening has no numerical curves to plot.

    Could add a method comparison bar chart in the future.
    """
    return []


# =====================================================================
# Private plotting helpers
# =====================================================================

def _plot_srf_vs_n(result):
    """Plot SRF vs stress concentration ratio n for the given a_s."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    a_s = result.area_replacement_ratio
    n_vals = [i * 0.5 for i in range(2, 41)]  # n from 1.0 to 20.0
    srf_vals = [1.0 / (1.0 + a_s * (n - 1.0)) for n in n_vals]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(n_vals, srf_vals, '-', color='#2563eb', linewidth=2,
            label=f'a_s = {a_s:.4f}')

    # Mark the design point
    n_design = result.stress_concentration_ratio
    srf_design = result.settlement_reduction_factor
    ax.plot(n_design, srf_design, 'o', color='#dc2626', markersize=10,
            zorder=5, label=f'Design: n={n_design:.1f}, SRF={srf_design:.3f}')
    ax.axhline(y=srf_design, color='#dc2626', linestyle=':', linewidth=1,
               alpha=0.5)
    ax.axvline(x=n_design, color='#dc2626', linestyle=':', linewidth=1,
               alpha=0.5)

    ax.set_xlabel('Stress Concentration Ratio, n', fontsize=10)
    ax.set_ylabel('Settlement Reduction Factor, SRF', fontsize=10)
    ax.set_title(
        f'SRF vs n  (a_s = {a_s:.4f}, d_c = {result.column_diameter_m:.3f} m, '
        f's = {result.column_spacing_m:.2f} m)',
        fontsize=11, fontweight='bold',
    )
    ax.set_xlim(1, max(n_vals))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def _plot_pier_comparison(result):
    """Bar chart comparing unreinforced vs improved conditions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r = result
    categories = []
    unr_vals = []
    imp_vals = []

    if r.settlement_unreinforced_mm > 0:
        categories.append('Settlement (mm)')
        unr_vals.append(r.settlement_unreinforced_mm)
        imp_vals.append(r.settlement_improved_mm)

    if r.unreinforced_bearing_kPa > 0:
        categories.append('Bearing Capacity (kPa)')
        unr_vals.append(r.unreinforced_bearing_kPa)
        imp_vals.append(r.improved_bearing_kPa)

    if not categories:
        # No data to plot
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, 'No comparison data', ha='center', va='center')
        return fig

    n_cats = len(categories)
    x = list(range(n_cats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar([xi - width / 2 for xi in x], unr_vals, width,
                   label='Unreinforced', color='#ef4444', edgecolor='#333')
    bars2 = ax.bar([xi + width / 2 for xi in x], imp_vals, width,
                   label='With Aggregate Piers', color='#22c55e',
                   edgecolor='#333')

    # Value labels
    for bar, val in zip(bars1, unr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, imp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title('Aggregate Pier Improvement', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def _plot_consolidation_vs_time(result):
    """Plot combined degree of consolidation vs time for wick drains."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    curve = result.time_settlement_curve
    times = [pt[0] for pt in curve]
    U_vals = [pt[1] for pt in curve]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(times, U_vals, '-', color='#2563eb', linewidth=2,
            label='U_total(t)')

    # Mark the analysis time point
    ax.plot(result.time_years, result.U_total_percent, 'o',
            color='#dc2626', markersize=10, zorder=5,
            label=(f't = {result.time_years:.2f} yr, '
                   f'U = {result.U_total_percent:.1f}%'))
    ax.axhline(y=result.U_total_percent, color='#dc2626', linestyle=':',
               linewidth=1, alpha=0.5)
    ax.axvline(x=result.time_years, color='#dc2626', linestyle=':',
               linewidth=1, alpha=0.5)

    ax.set_xlabel('Time (years)', fontsize=10)
    ax.set_ylabel('Degree of Consolidation U_total (%)', fontsize=10)
    ax.set_title(
        f'Consolidation vs Time  (s = {result.drain_spacing_m:.2f} m, '
        f'{result.pattern})',
        fontsize=11, fontweight='bold',
    )
    ax.set_ylim(0, 105)
    ax.set_xlim(left=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig


def _plot_surcharge_settlement_vs_time(result):
    """Plot settlement vs time under surcharge preloading."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    curve = result.time_settlement_curve
    times = [pt[0] for pt in curve]
    S_vals = [pt[1] for pt in curve]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(times, S_vals, '-', color='#2563eb', linewidth=2,
            label='S(t)')

    # Mark target point
    ax.plot(result.time_to_target_years, result.settlement_at_target_mm,
            'o', color='#dc2626', markersize=10, zorder=5,
            label=(f'Target: t = {result.time_to_target_years:.2f} yr, '
                   f'S = {result.settlement_at_target_mm:.1f} mm'))
    ax.axhline(y=result.settlement_at_target_mm, color='#dc2626',
               linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=result.time_to_target_years, color='#dc2626',
               linestyle=':', linewidth=1, alpha=0.5)

    # Mark ultimate settlement
    ax.axhline(y=result.settlement_ultimate_mm, color='#94a3b8',
               linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'S_ult = {result.settlement_ultimate_mm:.1f} mm')

    # Invert y-axis so settlement increases downward
    ax.invert_yaxis()

    ax.set_xlabel('Time (years)', fontsize=10)
    ax.set_ylabel('Settlement (mm)', fontsize=10)
    drain_str = " with wick drains" if result.uses_wick_drains else ""
    ax.set_title(
        f'Settlement vs Time Under {result.surcharge_kPa:.0f} kPa Surcharge'
        f'{drain_str}',
        fontsize=11, fontweight='bold',
    )
    ax.set_xlim(left=0)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    plt.tight_layout()
    return fig
