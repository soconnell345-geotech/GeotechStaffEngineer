"""
Calculation package steps for ground improvement analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation.  Supports all four
ground improvement methods plus the feasibility screening:

- Aggregate piers (Barksdale & Bachus; GEC-13's stress reduction method and
  Fig. 5-21 settlement ratios)
- Wick drains / PVD (GEC-13 Eq. 2-1 radial consolidation)
- Surcharge preloading (with/without drains)
- Vibro-compaction feasibility (GEC-13)
- Feasibility evaluation (decision support)

The ``analysis`` parameter is a dict whose ``"method"`` key selects
the branch, and the remaining keys are the input parameters echoed
in the calc package.

References:
    FHWA GEC-13: Ground Modification Methods Reference Manual
    UFC 3-220-10 (2026) — smear factor, combined consolidation
    Barksdale & Bachus (1983) — aggregate pier / stone column design
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
    "UFC 3-220-10: Soil Mechanics, 2026.",
    'Barksdale, R.D. & Bachus, R.C. (1983). "Design and Construction of '
    'Stone Columns." FHWA/RD-83/026.',
]


def _used(analysis, key, analyze):
    """An input the run used: the analysis dict's, else the default of the
    module function ``analyze`` (``analyze_aggregate_piers`` etc.), which
    the run then took (GM-141: the steps had defaults of their own); None
    where that function has no default."""
    if isinstance(analysis, dict) and key in analysis:
        return analysis[key]
    import inspect
    default = inspect.signature(analyze).parameters[key].default
    return None if default is inspect.Parameter.empty else default


def _piers(analysis, key):
    from ground_improvement import analyze_aggregate_piers
    return _used(analysis, key, analyze_aggregate_piers)


def _wicks(analysis, key):
    from ground_improvement import analyze_wick_drains
    return _used(analysis, key, analyze_wick_drains)


def _surcharge(analysis, key):
    from ground_improvement import analyze_surcharge_preloading
    return _used(analysis, key, analyze_surcharge_preloading)


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
        # E_column is not printed: it feeds only a warning, n sets the
        # stress split (GM-142; the choices table says so)
        InputItem("E_s", "Soil modulus",
                  _piers(analysis, "E_soil"), "kPa"),
        InputItem("n", "Stress concentration ratio",
                  r.stress_concentration_ratio, ""),
    ]
    if r.settlement_method != "stress_ratio":
        # GEC-13 Fig. 5-21's methods take E_c / E_s (GM-220)
        items[4:4] = [InputItem("E_c", "Column modulus",
                                _piers(analysis, "E_column"), "kPa")]
        items.append(InputItem("Settlement", "Settlement method",
                               _FIG_5_21_NAMES[r.settlement_method], ""))
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
        reference="FHWA GEC-13 (2017), Ch. 5, Sec. 4.1.2.2",
    ))

    sections.append(CalcSection(
        title="Geometry & Area Replacement", items=geo_items
    ))

    # -- Section 2: Settlement Reduction Factor -----------------------
    srf_items = []

    n = r.stress_concentration_ratio
    a_s = r.area_replacement_ratio

    if r.settlement_method != "stress_ratio":
        sections.append(CalcSection(
            title="Settlement Ratio by E_c / E_s (GEC-13 Fig. 5-21)",
            items=_fig_5_21_steps(r, _piers(analysis, "E_column"),
                                  _piers(analysis, "E_soil")),
        ))
    else:
        _stress_ratio_steps(r, analysis, srf_items, n, a_s)
        sections.append(CalcSection(
            title="Settlement Reduction & Composite Modulus",
            items=srf_items
        ))
    _bearing_and_settlement_steps(r, sections, n, a_s)
    return sections


#: GEC-13 Fig. 5-21's settlement methods by name (GM-220).
_FIG_5_21_NAMES = {
    "weighted_modulus": "Weighted modulus (GEC-13 Fig. 5-21 curve 2)",
    "balaam_rigid": "Balaam, rigid (GEC-13 Fig. 5-21 curves 5b, 5c)",
    "balaam_flexible": "Balaam, flexible (GEC-13 Fig. 5-21 curve 5a)",
}


def _fig_5_21_steps(r, E_c, E_s):
    """The settlement ratio by E_c / E_s off GEC-13 Fig. 5-21 (p. 5-36
    prints it as "n" in quotes; not Eq. 5-2's stress ratio n, GM-279)."""
    from ground_improvement.settlement_ratio import BASIS
    a_s = r.area_replacement_ratio
    ratio = r.modular_ratio
    n_p = r.settlement_ratio
    items = [
        CalcStep(
            title="Modular Ratio",
            equation="E_c / E_s",
            substitution=f"E_c / E_s = {E_c:.0f} / {E_s:.0f}",
            result_name="E_c/E_s",
            result_value=f"{ratio:.3g}",
        ),
        CalcStep(
            title="Area Ratio",
            equation="A / A_c = 1 / a_s",
            substitution=f"A / A_c = 1 / {a_s:.4f}",
            result_name="A/A_c",
            result_value=f"{1.0 / a_s:.2f}",
        ),
    ]
    if r.settlement_method == "weighted_modulus":
        items.append(CalcStep(
            title="Settlement Ratio, Weighted Modulus",
            equation="settlement ratio = 1 + a_s \u00d7 (E_c / E_s - 1)",
            substitution=(f"settlement ratio = 1 + {a_s:.4f} \u00d7 "
                          f"({ratio:.3g} - 1)"),
            result_name="settlement ratio",
            result_value=f"{n_p:.3f}",
            reference=BASIS["weighted_modulus"],
        ))
    else:
        curve = ("(5a), E_c/E_s = 10, flexible"
                 if r.settlement_method == "balaam_flexible"
                 else "(5b) E_c/E_s = 10 and (5c) E_c/E_s = 5, rigid")
        items.append(CalcStep(
            title="Settlement Ratio, Balaam",
            equation=(f"settlement ratio read off GEC-13 Fig. 5-21, "
                      f"curve {curve}"),
            substitution=(f"A / A_c = {1.0 / a_s:.2f}, "
                          f"E_c / E_s = {ratio:.3g}"),
            result_name="settlement ratio",
            result_value=f"{n_p:.3f}",
            reference=BASIS[r.settlement_method],
            notes=("Linear in a_s between the tabulated reads of the "
                   "printed curve"
                   + ("; linear in E_c/E_s between (5c) and (5b)"
                      if r.settlement_method == "balaam_rigid"
                      and not any(math.isclose(ratio, v, rel_tol=0.01)
                                  for v in (5.0, 10.0)) else "")),
        ))
    items.append(CalcStep(
        title="Settlement Reduction Factor (SRF)",
        equation="SRF = 1 / settlement ratio",
        substitution=f"SRF = 1 / {n_p:.3f}",
        result_name="SRF",
        result_value=f"{r.settlement_reduction_factor:.4f}",
        notes="The settlement ratio is the settlement without columns "
              "over that with them; GEC-13 p. 5-36 prints it as \"n\", "
              "which is not the stress ratio n of Eq. 5-2.",
    ))
    items.append(CalcStep(
        title="Composite Modulus of Improved Ground",
        equation="E_comp = E_s \u00d7 settlement ratio",
        substitution=f"E_comp = {E_s:.0f} \u00d7 {n_p:.3f}",
        result_name="E_comp",
        result_value=f"{r.composite_modulus_kPa:.0f}",
        result_unit="kPa",
        notes="Consistent with SRF: E_comp = E_s / SRF.",
    ))
    return items


def _stress_ratio_steps(r, analysis, srf_items, n, a_s):
    srf_items.append(CalcStep(
        title="Settlement Reduction Factor (SRF)",
        equation="SRF = 1 / [1 + a_s \u00d7 (n - 1)]",
        substitution=f"SRF = 1 / [1 + {a_s:.4f} \u00d7 ({n:.1f} - 1)]",
        result_name="SRF",
        result_value=f"{r.settlement_reduction_factor:.4f}",
        reference="FHWA GEC-13 (2017), Ch. 5, Eq. 5-4 and Sec. 4.1.2.6 "
                  "(stress reduction method), pp. 5-32, 5-36",
        notes="SRF < 1 indicates settlement improvement. SRF is the "
              "stress on the soil over the average stress, "
              "sigma_soil / q (Eq. 5-4), under equal strain.",
    ))

    # -- Section 3: Composite Modulus ---------------------------------
    E_s = _piers(analysis, "E_soil")

    srf_items.append(CalcStep(
        title="Composite Modulus of Improved Ground",
        equation="E_comp = E_s \u00d7 [1 + a_s \u00d7 (n - 1)]",
        substitution=(f"E_comp = {E_s:.0f} \u00d7 "
                      f"[1 + {a_s:.4f} \u00d7 ({n:.1f} - 1)]"),
        result_name="E_comp",
        result_value=f"{r.composite_modulus_kPa:.0f}",
        result_unit="kPa",
        reference=("GEC-13 (FHWA-NHI-16-027) Vol. I: the unit cell (Sec. "
                   "4.1.2.1, pp. 5-28 to 5-30) under equal strain (Sec. "
                   "4.1.2.4, p. 5-31), the soil stress by Eq. 5-4 (p. 5-32)"),
        notes=("Consistent with SRF: E_comp = E_s / SRF. GEC-13 prints no "
               "composite modulus; this is the modulus that settles the "
               "unit cell as the stress reduction method does."),
    ))



def _bearing_and_settlement_steps(r, sections, n, a_s):
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
            reference="Stress-concentration estimate (Barksdale & Bachus "
                      "1983; GEC-13)",
            notes="First-order estimate using the equal-strain "
                  "stress-concentration factor. GEC-13's Priebe curve "
                  "(Fig. 5-21 curve 4) is a settlement ratio, not a "
                  "bearing factor.",
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
            reference=("GEC-13 (FHWA-NHI-16-027) Vol. I, Sec. 4.1.2.6, "
                       "p. 5-36: the stress reduction method, the soil "
                       "stress by Eq. 5-4 (p. 5-32)"
                       if r.settlement_method == "stress_ratio" else
                       "GEC-13 Fig. 5-21: S = S_unreinforced / the "
                       "settlement ratio"),
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

        if r.settlement_method != "stress_ratio":
            # Figure 1: the GEC-13 Fig. 5-21 curve the run read (GM-220)
            fig1 = _plot_settlement_ratio(r)
            b64_1 = figure_to_base64(fig1, dpi=150)
            plt.close(fig1)
            figures.append(FigureData(
                title="Settlement Ratio vs Area Ratio (GEC-13 Fig. 5-21)",
                image_base64=b64_1,
                caption=(
                    f"Figure 1: settlement ratio (settlement without "
                    f"columns / with them) vs A/A_c, "
                    f"{_FIG_5_21_NAMES[r.settlement_method]}, E_c/E_s = "
                    f"{r.modular_ratio:.3g}. Design point: A/A_c = "
                    f"{1.0 / r.area_replacement_ratio:.2f}, settlement "
                    f"ratio {r.settlement_ratio:.2f}."
                ),
                width_percent=70,
            ))
        else:
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
    smear = _wicks(analysis, "smear_ratio")
    kh_ks = _wicks(analysis, "kh_ks_ratio")
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
        reference="FHWA GEC-13 (2017), Ch. 2, p. 2-33 (d_c)",
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

    smear = _wicks(analysis, "smear_ratio")
    kh_ks = _wicks(analysis, "kh_ks_ratio")

    well = ("Well resistance not modelled (F_r = 0): GEC-13 p. 2-36, "
            "'Well resistance is rarely significant, and, therefore, "
            "ignored except for extremely deep drains or where a "
            "combination of high loads and very permeable soils is "
            "present.'")
    if smear <= 1.0:
        fn_eq = "F = F(n) = ln(n) - 0.75"
        fn_sub = f"F = ln({r.spacing_ratio_n:.1f}) - 0.75"
        fn_ref = "FHWA GEC-13 (2017), Ch. 2, Eqs. 2-1 and 2-2, p. 2-32"
        fn_notes = ("Smear ignored (F_s = 0): GEC-13 p. 2-35, 'The effects "
                    "of both soil disturbance and well resistance are often "
                    "ignored for typical projects.' " + well)
    else:
        fn_eq = ("F = F(n) + F_s = ln(n) - 0.75 + "
                 "(k_h/k_s) \u00d7 ln(s_r)")
        fn_sub = (f"F = ln({r.spacing_ratio_n:.1f}) - 0.75 + "
                  f"{kh_ks:.1f} \u00d7 ln({smear:.1f})")
        fn_ref = ("FHWA GEC-13 (2017), Ch. 2, Eqs. 2-1 and 2-2, p. 2-32; "
                  "F_s by UFC 3-220-10 (2026) Eq. 5-25, p. 267")
        fn_notes = ("GEC-13 prints no formula for F_s; p. 2-35: 'a sample "
                    "disturbance factor F_s \u2248 2 may be considered' in "
                    "highly plastic, sensitive soils whose c_h is "
                    "accurately determined. " + well)

    fn_items.append(CalcStep(
        title="Drain Function",
        equation=fn_eq,
        substitution=fn_sub,
        result_name="F",
        result_value=f"{r.F_n:.3f}",
        reference=fn_ref,
        notes=fn_notes,
    ))

    sections.append(CalcSection(
        title="Drain Function", items=fn_items
    ))

    # -- Section 3: Degree of Consolidation ---------------------------
    consol_items = []

    Hdr = _wicks(analysis, "Hdr")       # required: None if not given
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
        equation="U_r = 1 - exp(-8 \u00d7 T_r / F)",
        substitution=f"U_r = 1 - exp(-8 \u00d7 {Tr_val:.4f} / {r.F_n:.3f})",
        result_name="U_r",
        result_value=f"{r.Ur_percent:.1f}",
        result_unit="%",
        reference=("FHWA GEC-13 (2017), Ch. 2, Eq. 2-1 (Rixner et al. "
                   "1986a), p. 2-32, solved for U_h"),
    ))

    # Vertical consolidation
    if Hdr:
        Tv_val = r.cv_m2_per_year * t / Hdr ** 2
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
        reference=("UFC 3-220-10 (2026) Eq. 5-27 (Carrillo 1942), p. 268"),
        notes=("GEC-13's Eq. 2-1 is radial only. p. 2-33: 'the contribution "
               "of vertical consolidation may be significant and should be "
               "considered for major/complex projects'; p. 2-34: 'for a "
               "conservative approach, no beneficial effects may be "
               "considered for vertical drainage in preliminary feasibility "
               "evaluations' (U_r above)."),
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
            ["Drain function F", f"{r.F_n:.3f}"],
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
        sigma_v0 = _surcharge(analysis, "sigma_v0")
        design_items.append(CalcStep(
            title="Equivalent Preconsolidation Pressure",
            equation="\u03c3'_p = \u03c3'_v0 + q_surcharge",
            substitution=(f"\u03c3'_p = {sigma_v0:.1f} + "
                          f"{r.surcharge_kPa:.1f}"),
            result_name="\u03c3'_p",
            result_value=f"{r.equivalent_sigma_p_kPa:.1f}",
            result_unit="kPa",
            reference="FHWA GEC-13 (2017), Ch. 2, Sec. 4.2 (surcharge to "
                      "overconsolidate the soil for the design load, "
                      "p. 2-36); preconsolidation stress as defined in "
                      "Sec. 1.4, p. 2-7",
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
                "where U_v = Terzaghi, U_r = GEC-13 Eq. 2-1"
            ),
            substitution=f"Find t such that U_total = {r.target_U_percent:.0f}%",
            result_name="t_target",
            result_value=f"{r.time_to_target_years:.3f}",
            result_unit="years",
            reference=("UFC 3-220-10 (2026) Eq. 5-27 (Carrillo 1942), p. "
                       "268; U_r by GEC-13 (2017) Ch. 2, Eq. 2-1"),
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
                ["Drain function F", f"{wd.F_n:.3f}"],
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
    ]
    if r.target_relative_density_percent is not None:
        items.append(InputItem("D_r,target", "Target relative density",
                               r.target_relative_density_percent, "%"))
    if r.is_feasible and r.recommended_spacing_m > 0:
        items.append(InputItem("Pattern", "Probe pattern",
                               r.probe_pattern.capitalize(), ""))
    return items


def _steps_vibro(result, analysis) -> List[CalcSection]:
    """The screen and the spacing as ``ground_improvement.vibro`` works
    them from GEC-13 Ch. 4 (GM-291)."""
    from ground_improvement.vibro import (
        FIG_4_34, FINES_IDEAL_PERCENT, FINES_LIMIT_PERCENT, density_from_N)
    r = result
    sections = []
    lo, hi = FINES_LIMIT_PERCENT

    # -- Section 1: Feasibility Screening -----------------------------
    screen_items = []

    fc = r.fines_content_percent
    if fc <= FINES_IDEAL_PERCENT:
        fc_status = "Ideally suited (zone B, below 10%)"
    elif fc <= lo:
        fc_status = "Feasible (above zone B's 10%, below 12%)"
    elif fc <= hi:
        fc_status = "Marginal (within 12 to 15%)"
    else:
        fc_status = "Not feasible (above 15%)"

    screen_items.append(CalcStep(
        title="Fines Content Screening",
        equation="FC below 10%: ideally suited (Fig. 4-27 zone B); "
                 "above 12 to 15%: densification generally cannot be "
                 "achieved",
        substitution=f"FC = {fc:.1f}%",
        result_name="FC status",
        result_value=fc_status,
        reference="FHWA GEC-13 (2017), Ch. 4, Sec. 3.1.3.1, Table 4-5 "
                  "and Figure 4-27, pp. 4-65 to 4-67",
        notes="GEC-13 p. 4-65: 'proper densification generally cannot be "
              "achieved when the granular soil contains more than 12 to 15 "
              "percent silt OR more than 2 percent clay'; p. 4-67: zone B "
              "is 'ideally suited for vibro-compaction with fines content "
              "below 10 percent'. The module reads the printed 12 to 15 % "
              "as marginal and above 15 % as not feasible. The clay "
              "fraction is not screened (the tool takes the fines content); "
              "Table 4-5 inhibits a silty sand (SM) above 8 % silt and a "
              "clayey sand (SC) above 2 % clay.",
    ))

    # SPT density: GEC-13 Table 4-8
    if r.initial_N_spt > 0:
        name, (dr_lo, dr_hi) = density_from_N(r.initial_N_spt)
        screen_items.append(CalcStep(
            title="Density from SPT N",
            equation="GEC-13 Table 4-8: N < 4 very loose (D_r < 15%); "
                     "4-10 loose (15-35%); 10-30 medium dense (35-65%); "
                     "30-50 dense (65-85%); > 50 very dense (85-100%)",
            substitution=f"N_initial = {r.initial_N_spt:.0f}",
            result_name="Density",
            result_value=f"{name}, D_r {dr_lo:.0f} to {dr_hi:.0f}%",
            reference="FHWA GEC-13 (2017), Ch. 4, Table 4-8, p. 4-74 "
                      "(normally consolidated sand)",
            notes=("GEC-13 p. 4-75: 'It is unlikely, even for heavy "
                   "loading, that it will be necessary to achieve a "
                   "relative density above 85 percent.'"
                   if dr_lo >= 85.0 else ""),
        ))

    # Overall feasibility
    screen_items.append(CalcStep(
        title="Overall Feasibility Assessment",
        equation="Vibro-compaction is feasible if all criteria are met",
        substitution="",
        result_name="Feasible",
        result_value="YES" if r.is_feasible else "NO",
        reference="FHWA GEC-13 (2017), Ch. 4, pp. 4-65 to 4-75",
    ))

    sections.append(CalcSection(
        title="Feasibility Screening", items=screen_items
    ))

    # -- Section 2: Probe Spacing (if feasible) -----------------------
    if r.is_feasible and r.recommended_spacing_m > 0:
        dr = r.target_relative_density_percent
        nodes = "; ".join(f"{k:g}% \u2192 {v:.2f} m"
                          for k, v in sorted(FIG_4_34.items()))
        sections.append(CalcSection(title="Probe Spacing", items=[CalcStep(
            title="Probe Spacing for the Target Relative Density",
            equation=("s read off GEC-13 Fig. 4-34, relative density "
                      "versus probe spacing for silty sands"),
            substitution=f"D_r,target = {dr:g}%",
            result_name="s_probe",
            result_value=f"{r.recommended_spacing_m:.2f}",
            result_unit="m",
            reference="FHWA GEC-13 (2017), Ch. 4, Fig. 4-34, p. 4-77",
            notes=("Read off the printed figure at " + nodes + ", linear "
                   "between them. The chart 'is based on the lower bound "
                   "soil gradation (silty sand) indicated in Figure 4-33' "
                   "(p. 4-76). GEC-13's criteria (p. 4-77): 60% for floor "
                   "slabs, flat bottom tanks, embankments; 70 to 75% for "
                   "column and bridge footings; 80% for machinery and mat "
                   "foundations."),
        )]))

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


def _plot_settlement_ratio(result):
    """The settlement ratio against A/A_c on the GEC-13 Fig. 5-21 curve
    the run read, over
    the curve's drawn range, with the design point (GM-220)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ground_improvement.settlement_ratio import (
        BALAAM_FLEXIBLE_10, BALAAM_RIGID_5, settlement_ratio)

    r = result
    table = (BALAAM_FLEXIBLE_10 if r.settlement_method == "balaam_flexible"
             else BALAAM_RIGID_5)
    x0, x1 = ((1.5, 9.0) if r.settlement_method == "weighted_modulus"
              else (min(table), max(table)))
    xs = [x0 + (x1 - x0) * i / 100 for i in range(101)]
    ys = [settlement_ratio(r.settlement_method, 1.0 / x, r.modular_ratio)
          for x in xs]
    x_d, y_d = 1.0 / r.area_replacement_ratio, r.settlement_ratio

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, ys, '-', color='#2563eb', linewidth=2)
    # the curve named near its steep end, clear of the design point
    ax.annotate(f"E_c/E_s = {r.modular_ratio:.3g}", (xs[12], ys[12]),
                xytext=(10, 4), textcoords="offset points", fontsize=9,
                color='#2563eb')
    ax.plot(x_d, y_d, 'o', color='#dc2626', markersize=8, zorder=5)
    ax.annotate(f"A/A_c = {x_d:.2f}, settlement ratio {y_d:.2f}",
                (x_d, y_d),
                xytext=(8, 8), textcoords="offset points", fontsize=9,
                color='#dc2626')
    ax.set_xlabel("Area ratio, A / A_c", fontsize=10)
    ax.set_ylabel("Settlement ratio", fontsize=10)
    ax.set_title(_FIG_5_21_NAMES[r.settlement_method], fontsize=11,
                 fontweight='bold')
    ax.set_xlim(1, 10)
    ax.set_ylim(1, max(ys) * 1.05)
    ax.tick_params(labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
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
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)
    # the key under the axes: in the corner the curve and the target's
    # lines ran through it (GM-194)
    from geotech_common.plotting import key_under_axes
    key_under_axes(fig, ax, fontsize=9, ncol=1)
    return fig
