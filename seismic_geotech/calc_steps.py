"""
Calculation package steps for seismic geotechnical analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation.

Handles three result types:
    - SiteClassResult: AASHTO site classification and coefficients
    - SeismicEarthPressureResult: Mononobe-Okabe seismic earth pressures
    - LiquefactionResult: SPT-based liquefaction triggering evaluation

The `analysis` parameter should be a dict with:
    - "analysis_type": one of "site_classification", "seismic_earth_pressure",
      "liquefaction"
    - Plus type-specific keys echoing the analysis inputs.

References:
    FHWA GEC-12 Vol I (FHWA-NHI-16-009) Sec. 7.4.2.1, after AASHTO (2014)
    UFC 3-220-20 (2025) Sec. 4-4.4.1, Eqs. 4-15 to 4-17
    FHWA GEC-13 Vol I (FHWA-NHI-16-027) Eq. 5-9
    Mononobe & Matsuo (1929); Okabe (1926)
    Youd et al. (2001), ASCE JGGE, Vol 127, No 10
    Seed & Idriss (1971), ASCE JSMFED
    Seed & Harder (1990), H. Bolton Seed Memorial Symposium
"""

import math
from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64
from seismic_geotech.results import (
    SiteClassResult, SeismicEarthPressureResult, LiquefactionResult,
)

DISPLAY_NAME = "Seismic Geotechnical Analysis"

REFERENCES = [
    'FHWA (2016). Design and Construction of Driven Pile Foundations, '
    'Vol. I. GEC-12, FHWA-NHI-16-009. Sec. 7.4.2.1 (site class, site '
    'factors and design response spectrum, after AASHTO 2014).',
    'UFC 3-220-20 (16 January 2025). Sec. 4-4.4.1 (Mononobe-Okabe, Eqs. '
    '4-15 to 4-17).',
    'FHWA (2017). Ground Modification Methods Reference Manual, Vol. I. '
    'GEC-13, FHWA-NHI-16-027. Eq. 5-9 and its r_d (pp. 5-40 and 5-41).',
    'AASHTO LRFD Bridge Design Specifications, 9th Ed., Section 3.10.3.',
    'Mononobe, N. & Matsuo, H. (1929). "On the Determination of Earth '
    'Pressures During Earthquakes." Proc. World Engineering Congress, Tokyo.',
    'Okabe, S. (1926). "General Theory of Earth Pressure." '
    'J. Japan Society of Civil Engineers, 12(1).',
    'Seed, H.B. & Whitman, R.V. (1970). "Design of Earth Retaining '
    'Structures for Dynamic Loads." ASCE Specialty Conference.',
    'Youd, T.L. et al. (2001). "Liquefaction Resistance of Soils: '
    'Summary Report from the 1996 NCEER and 1998 NCEER/NSF Workshops." '
    'ASCE JGGE, Vol 127, No 10, pp. 817-833.',
    'Seed, H.B. & Idriss, I.M. (1971). "Simplified Procedure for '
    'Evaluating Soil Liquefaction Potential." ASCE JSMFED, Vol 97, SM9.',
    'Seed, R.B. & Harder, L.F. (1990). "SPT-Based Analysis of Cyclic '
    'Pore Pressure Generation and Undrained Residual Strength." '
    'H. Bolton Seed Memorial Symposium.',
    'Liao, S.S.C. & Whitman, R.V. (1986). "Overburden Correction Factors '
    'for SPT in Sand." ASCE JGGE, Vol 112, No 3.',
]


# ═══════════════════════════════════════════════════════════════════
# Dispatch helpers
# ═══════════════════════════════════════════════════════════════════

def _detect_result_type(result):
    """Determine the analysis type from the result object."""
    if isinstance(result, SiteClassResult):
        return "site_classification"
    elif isinstance(result, SeismicEarthPressureResult):
        return "seismic_earth_pressure"
    elif isinstance(result, LiquefactionResult):
        return "liquefaction"
    return "unknown"


# ═══════════════════════════════════════════════════════════════════
# get_input_summary
# ═══════════════════════════════════════════════════════════════════

def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for seismic geotechnical calc package.

    Parameters
    ----------
    result : SiteClassResult | SeismicEarthPressureResult | LiquefactionResult
        Computed results.
    analysis : dict
        Dict with analysis inputs.

    Returns
    -------
    list of InputItem
    """
    rtype = _detect_result_type(result)
    if rtype == "site_classification":
        return _inputs_site_class(result, analysis)
    elif rtype == "seismic_earth_pressure":
        return _inputs_earth_pressure(result, analysis)
    elif rtype == "liquefaction":
        return _inputs_liquefaction(result, analysis)
    return []


def _inputs_site_class(result, analysis) -> List[InputItem]:
    """Input summary for site classification analysis."""
    items = []
    items.append(InputItem("Analysis", "Analysis type", "Site Classification", ""))

    if result.vs30 is not None:
        items.append(InputItem("Vs30", "Average shear wave velocity (top 30 m)", f"{result.vs30:.0f}", "m/s"))
    if result.n_bar is not None:
        items.append(InputItem("N-bar", "Average SPT N (top 30 m)", f"{result.n_bar:.0f}", ""))
    if result.su_bar is not None:
        items.append(InputItem("su-bar", "Average undrained strength (top 30 m)", f"{result.su_bar:.0f}", "kPa"))

    if result.Ss is not None:
        items.append(InputItem("Ss", "Spectral acceleration at 0.2 s", f"{result.Ss:.3f}", "g"))
        items.append(InputItem("S1", "Spectral acceleration at 1.0 s", f"{result.S1:.3f}", "g"))
    else:
        items.append(InputItem("Ss, S1", "Spectral accelerations",
                               "not given: site class only", ""))

    # Echo layer data if provided in analysis dict
    a = analysis or {}
    if a.get("pga") is not None and result.Ss is not None:
        items.append(InputItem("PGA", "Peak ground acceleration on rock",
                               f"{a['pga']:.3f}", "g"))
    if "layer_thicknesses" in a:
        items.append(InputItem("Layers", "Number of soil layers", len(a["layer_thicknesses"]), ""))

    return items


def _inputs_earth_pressure(result, analysis) -> List[InputItem]:
    """Input summary for Mononobe-Okabe analysis."""
    items = [
        InputItem("Analysis", "Analysis type", "Mononobe-Okabe Seismic Earth Pressure", ""),
        InputItem("\u03c6", "Soil friction angle", result.phi, "deg"),
        InputItem("\u03b4", "Wall-soil interface friction", result.delta, "deg"),
        InputItem("k_h", "Horizontal seismic coefficient", result.kh, ""),
        InputItem("k_v", "Vertical seismic coefficient", result.kv, ""),
    ]

    a = analysis or {}
    if a.get("beta") is not None:
        items.append(InputItem("\u03b2", "Wall batter from vertical", a["beta"], "deg"))
    if a.get("i") is not None:
        items.append(InputItem("i", "Backfill slope angle", a["i"], "deg"))
    if a.get("gamma") is not None:
        items.append(InputItem("\u03b3", "Backfill unit weight", a["gamma"], "kN/m\u00b3"))
    if a.get("H") is not None:
        items.append(InputItem("H", "Wall height", a["H"], "m"))
    if a.get("KA") is not None:
        items.append(InputItem(
            "K_A", "Static active coefficient behind PA",
            f"{a['KA']:.4f}",
            "given" if a.get("KA_given") else "Rankine, from \u03c6"))

    return items


def _inputs_liquefaction(result, analysis) -> List[InputItem]:
    """Input summary for liquefaction evaluation."""
    items = [
        InputItem("Analysis", "Analysis type", "Liquefaction Triggering Evaluation", ""),
        InputItem("PGA", "Peak ground acceleration", f"{result.amax_g:.3f}", "g"),
        InputItem("M_w", "Earthquake magnitude", f"{result.magnitude:.1f}", ""),
        InputItem("GWT", "Groundwater depth", f"{result.gwt_depth:.1f}", "m"),
        InputItem("N_layers", "Number of evaluation layers", len(result.layer_results), ""),
    ]
    return items


# ═══════════════════════════════════════════════════════════════════
# get_calc_steps
# ═══════════════════════════════════════════════════════════════════

def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections.

    Parameters
    ----------
    result : SiteClassResult | SeismicEarthPressureResult | LiquefactionResult
    analysis : dict

    Returns
    -------
    list of CalcSection
    """
    rtype = _detect_result_type(result)
    if rtype == "site_classification":
        return _steps_site_class(result, analysis)
    elif rtype == "seismic_earth_pressure":
        return _steps_earth_pressure(result, analysis)
    elif rtype == "liquefaction":
        return _steps_liquefaction(result, analysis)
    return []


# ── Site Classification Calc Steps ──────────────────────────────

def _steps_site_class(result, analysis) -> List[CalcSection]:
    """Calc steps for site classification."""
    sections = []
    a = analysis or {}

    # Section: Vs30 / N-bar / su-bar computation
    param_items = []
    if result.vs30 is not None:
        param_items.append(CalcStep(
            title="Average Shear Wave Velocity (Vs30)",
            equation="Vs30 = \u03a3d_i / \u03a3(d_i / Vs_i)  (harmonic mean over top 30 m)",
            substitution="",
            result_name="Vs30",
            result_value=f"{result.vs30:.0f}",
            result_unit="m/s",
            reference="FHWA GEC-12 Vol I Table 7-26 (p. 379); GEC-12 p. 378 (upper 100 ft)",
            notes="Weighted harmonic average of shear wave velocity in top 30 m",
        ))
    if result.n_bar is not None:
        param_items.append(CalcStep(
            title="Average SPT N (N-bar)",
            equation="N-bar = \u03a3d_i / \u03a3(d_i / N_i)  (harmonic mean over top 30 m)",
            substitution="",
            result_name="N-bar",
            result_value=f"{result.n_bar:.0f}",
            result_unit="blows/ft",
            reference="FHWA GEC-12 Vol I Table 7-26 (p. 379); GEC-7 p. 93 (field N, 100 bpf at most)",
        ))
    if result.su_bar is not None:
        param_items.append(CalcStep(
            title="Average Undrained Strength (su-bar)",
            equation="su-bar = \u03a3d_i / \u03a3(d_i / su_i)  (harmonic mean)",
            substitution="",
            result_name="su-bar",
            result_value=f"{result.su_bar:.0f}",
            result_unit="kPa",
            reference="FHWA GEC-12 Vol I Table 7-26 (p. 379)",
        ))

    if param_items:
        sections.append(CalcSection(
            title="Site Parameter Computation", items=param_items
        ))

    # Section: Site Classification
    class_items = []

    # Classification criteria table
    class_table = TableData(
        title="AASHTO Site Classification Criteria (GEC-12 Table 7-26)",
        headers=["Site Class", "Vs30 (m/s)", "N-bar", "su-bar (kPa)"],
        rows=[
            ["A", "> 1524", "\u2014", "\u2014"],
            ["B", "762 \u2013 1524", "\u2014", "\u2014"],
            ["C", "365.76 \u2013 762", "> 50", "> 95.76"],
            ["D", "182.88 \u2013 365.76", "15 \u2013 50",
             "47.88 \u2013 95.76"],
            ["E", "< 182.88", "< 15", "< 47.88"],
            ["F", "Site-specific", "Site-specific", "Site-specific"],
        ],
        notes=("Printed in ft/s and ksf (5,000, 2,500, 1,200 and 600 ft/s; "
               "2.0 and 1.0 ksf), converted exactly. Priority here: "
               "Vs30 > N-bar > su-bar. Also Site Class E: any profile with "
               "more than 10 ft of soft clay (PI > 20, w > 40 %, su < 0.5 "
               "ksf); Site Class F: peat, very high plasticity or very "
               "thick soft clay. Neither is checked by the tool."),
    )
    class_items.append(class_table)

    class_items.append(CalcStep(
        title="Site Classification Result",
        equation="Classify based on Vs30, N-bar, or su-bar per GEC-12 Table 7-26",
        substitution=_site_class_substitution(result),
        result_name="Site Class",
        result_value=result.site_class,
        reference="FHWA GEC-12 Vol I Table 7-26 (p. 379)",
    ))
    sections.append(CalcSection(title="Site Classification", items=class_items))

    # Without Ss and S1 the run gives the site class alone.
    if result.Ss is None:
        return sections

    # Section: Site Coefficients
    coeff_items = []

    # Fpga is read against PGA only; without it there is no Fpga and no As
    # (site_coefficients, GM-293).
    pga = result.pga if result.pga is not None else a.get("pga")
    if result.Fpga is not None:
        coeff_items.append(CalcStep(
            title="Site Coefficient Fpga (PGA amplification)",
            equation="Fpga = f(Site Class, PGA) from GEC-12 Table 7-27",
            substitution=f"Site Class {result.site_class}, PGA = {pga:.3f} g",
            result_name="Fpga",
            result_value=f"{result.Fpga:.3f}",
            reference="FHWA GEC-12 Vol I Table 7-27 (p. 380)",
            notes="Straight-line interpolation between the columns (p. 379)",
        ))

    coeff_items.append(CalcStep(
        title="Site Coefficient Fa (short-period amplification)",
        equation="Fa = f(Site Class, Ss) from GEC-12 Table 7-28",
        substitution=f"Site Class {result.site_class}, Ss = {result.Ss:.3f} g",
        result_name="Fa",
        result_value=f"{result.Fa:.3f}",
        reference="FHWA GEC-12 Vol I Table 7-28 (p. 380)",
    ))

    coeff_items.append(CalcStep(
        title="Site Coefficient Fv (long-period amplification)",
        equation="Fv = f(Site Class, S1) from GEC-12 Table 7-29",
        substitution=f"Site Class {result.site_class}, S1 = {result.S1:.3f} g",
        result_name="Fv",
        result_value=f"{result.Fv:.3f}",
        reference="FHWA GEC-12 Vol I Table 7-29 (p. 380)",
    ))

    sections.append(CalcSection(title="Site Coefficients", items=coeff_items))

    # Section: Design Spectral Accelerations
    SDS = result.Fa * result.Ss
    SD1 = result.Fv * result.S1

    design_items = []
    design_items.append(CalcStep(
        title="Design Short-Period Spectral Acceleration",
        equation="SDS = Fa \u00d7 Ss",
        substitution=f"SDS = {result.Fa:.3f} \u00d7 {result.Ss:.3f}",
        result_name="SDS",
        result_value=f"{SDS:.4f}",
        result_unit="g",
        reference="FHWA GEC-12 Vol I Eq. 7-79 (p. 381)",
    ))

    design_items.append(CalcStep(
        title="Design Long-Period Spectral Acceleration",
        equation="SD1 = Fv \u00d7 S1",
        substitution=f"SD1 = {result.Fv:.3f} \u00d7 {result.S1:.3f}",
        result_name="SD1",
        result_value=f"{SD1:.4f}",
        result_unit="g",
        reference="FHWA GEC-12 Vol I p. 382 (SD1 = FvS1)",
    ))

    if result.As is not None:
        design_items.append(CalcStep(
            title="Peak Ground Acceleration Modified by Fpga",
            equation="As = Fpga \u00d7 PGA",
            substitution=f"As = {result.Fpga:.3f} \u00d7 {pga:.3f}",
            result_name="As",
            result_value=f"{result.As:.4f}",
            result_unit="g",
            reference="FHWA GEC-12 Vol I Eq. 7-78 (p. 381)",
        ))

    # Summary table
    rows = [["Site Class", result.site_class, ""]]
    if result.Fpga is not None:
        rows.append(["Fpga", f"{result.Fpga:.3f}", ""])
    rows += [
        ["Fa", f"{result.Fa:.3f}", ""],
        ["Fv", f"{result.Fv:.3f}", ""],
        ["SDS = Fa \u00d7 Ss", f"{SDS:.4f}", "g"],
        ["SD1 = Fv \u00d7 S1", f"{SD1:.4f}", "g"],
    ]
    if result.As is not None:
        rows.append(["As = Fpga \u00d7 PGA", f"{result.As:.4f}", "g"])
    design_items.append(TableData(
        title="Design Spectral Parameters Summary",
        headers=["Parameter", "Value", "Unit"],
        rows=rows,
        notes=("" if result.As is not None else
               "PGA not given: no Fpga and no As (GEC-12 Table 7-27 reads "
               "Fpga against the PGA)"),
    ))

    sections.append(CalcSection(
        title="Design Spectral Accelerations", items=design_items
    ))

    return sections


def _site_class_substitution(result) -> str:
    """Build substitution string showing which parameter was used."""
    if result.vs30 is not None:
        return f"Vs30 = {result.vs30:.0f} m/s \u2192 Site Class {result.site_class}"
    if result.n_bar is not None:
        return f"N-bar = {result.n_bar:.0f} \u2192 Site Class {result.site_class}"
    if result.su_bar is not None:
        return f"su-bar = {result.su_bar:.0f} kPa \u2192 Site Class {result.site_class}"
    return f"Site Class {result.site_class}"


# ── Mononobe-Okabe Calc Steps ──────────────────────────────────

def _steps_earth_pressure(result, analysis) -> List[CalcSection]:
    """Calc steps for Mononobe-Okabe seismic earth pressure."""
    r = result
    a = analysis or {}
    sections = []

    # Section: Seismic Inertia Angle
    theta_items = []

    theta_rad = math.atan(r.kh / (1.0 - r.kv)) if abs(1.0 - r.kv) > 1e-10 else 0.0
    theta_deg = math.degrees(theta_rad)

    theta_items.append(CalcStep(
        title="Seismic Inertia Angle",
        equation="\u03b8 = arctan(k_h / (1 - k_v))",
        substitution=f"\u03b8 = arctan({r.kh:.3f} / (1 - {r.kv:.3f}))",
        result_name="\u03b8",
        result_value=f"{theta_deg:.2f}",
        result_unit="deg",
        reference="AASHTO LRFD Eq. 11.6.5.2",
    ))

    sections.append(CalcSection(title="Seismic Inertia Angle", items=theta_items))

    # Section: Active Earth Pressure Coefficient (KAE)
    kae_items = []

    beta_deg = a.get("beta", 0.0)
    i_deg = a.get("i", 0.0)

    kae_items.append(CalcStep(
        title="Mononobe-Okabe Active Coefficient (KAE)",
        equation=(
            # Matches mononobe_okabe_KAE (SG-3 sign convention: beta
            # positive = back face leaning toward the backfill).
            "KAE = cos\u00b2(\u03c6 + \u03b2 - \u03b8) / "
            "[cos\u03b8 \u00d7 cos\u00b2\u03b2 \u00d7 cos(\u03b4 + \u03b8 - \u03b2) \u00d7 "
            "(1 + \u221a(sin(\u03c6+\u03b4)\u00d7sin(\u03c6-\u03b8-i) / "
            "(cos(\u03b4+\u03b8-\u03b2)\u00d7cos(i+\u03b2))))\u00b2]"
        ),
        substitution=(
            f"\u03c6 = {r.phi:.1f}\u00b0, \u03b4 = {r.delta:.1f}\u00b0, "
            f"k_h = {r.kh:.3f}, k_v = {r.kv:.3f}, "
            f"\u03b2 = {beta_deg:.1f}\u00b0, i = {i_deg:.1f}\u00b0"
        ),
        result_name="KAE",
        result_value=f"{r.KAE:.4f}",
        reference="AASHTO LRFD Eq. 11.6.5.2-1; Mononobe-Okabe (1929)",
    ))

    # Static Ka: with no KA given it is the KA the run used behind PA and
    # delta-PAE (GM-120); beside a KA given it is for comparison.
    # The KA behind PA and delta-PAE, printed where they are worked out.
    Ka_used, ka_worked_out = _ka_used(a, r.phi)
    Ka_static = (Ka_used if ka_worked_out else
                 math.tan(math.pi / 4 - math.radians(r.phi) / 2) ** 2)
    ratio = f"KAE/Ka = {r.KAE/Ka_static:.2f}" if Ka_static > 0 else ""
    kae_items.append(CalcStep(
        title=("Static Active Coefficient (Rankine Ka)" if ka_worked_out
               else "Static Active Coefficient (Rankine Ka, for comparison)"),
        equation="Ka = tan\u00b2(45\u00b0 - \u03c6/2)",
        substitution=f"Ka = tan\u00b2(45\u00b0 - {r.phi:.1f}\u00b0/2)",
        result_name="Ka",
        result_value=f"{Ka_static:.4f}",
        reference="Rankine (1857)",
        notes=(f"KA not given: this Ka gives PA and \u0394PAE. {ratio}"
               if ka_worked_out else ratio),
    ))

    sections.append(CalcSection(
        title="Seismic Active Earth Pressure Coefficient", items=kae_items
    ))

    # Section: Passive Coefficient (if available)
    if r.KPE > 0:
        kpe_items = []
        kpe_items.append(CalcStep(
            title="Mononobe-Okabe Passive Coefficient (KPE)",
            equation=(
                # Matches mononobe_okabe_KPE after the SG-3 sign fix
                # (v5.1): numerator cos\u00b2(\u03c6\u2212\u03b8\u2212\u03b2), cos(\u03b4+\u03b8+\u03b2) terms,
                # cos(i+\u03b2).
                "KPE = cos\u00b2(\u03c6 - \u03b8 - \u03b2) / "
                "[cos\u03b8 \u00d7 cos\u00b2\u03b2 \u00d7 cos(\u03b4 + \u03b8 + \u03b2) \u00d7 "
                "(1 - \u221a(sin(\u03c6+\u03b4)\u00d7sin(\u03c6-\u03b8+i) / "
                "(cos(\u03b4+\u03b8+\u03b2)\u00d7cos(i+\u03b2))))\u00b2]"
            ),
            substitution=(
                f"\u03c6 = {r.phi:.1f}\u00b0, \u03b4 = {r.delta_passive if r.delta_passive is not None else r.delta:.1f}\u00b0, "
                f"k_h = {r.kh:.3f}, k_v = {r.kv:.3f}, "
                f"\u03b2 = {beta_deg:.1f}\u00b0, i = {i_deg:.1f}\u00b0"
            ),
            result_name="KPE",
            result_value=f"{r.KPE:.4f}",
            reference="FHWA GEC-4 Eq. 44 (p. 114); Mononobe-Okabe (1929)",
        ))
        sections.append(CalcSection(
            title="Seismic Passive Earth Pressure Coefficient", items=kpe_items
        ))

    # Section: Pressure Resultants (if H and gamma available)
    if r.PAE_total > 0:
        # as the run used them; the symbol where the analysis lacks one
        gamma, H = a.get("gamma"), a.get("H")
        g_txt = f"{gamma:.1f}" if gamma is not None else "\u03b3"
        H_txt = f"{H:.2f}" if H is not None else "H"
        # the kv in PAE: the run's, except where M-O has no solution and
        # the Seed & Whitman KAE leaves it out (GM-149)
        from seismic_geotech.mononobe_okabe import mo_no_solution
        kv_force = (0.0 if mo_no_solution(r.phi, r.kh, r.kv,
                                          a.get("i", 0.0) or 0.0)
                    else r.kv)

        force_items = []
        force_items.append(CalcStep(
            title="Total Seismic Active Force",
            equation=("PAE = 0.5 \u00d7 \u03b3 \u00d7 H\u00b2 \u00d7 "
                      "(1 - k_v) \u00d7 KAE"),
            substitution=(f"PAE = 0.5 \u00d7 {g_txt} \u00d7 {H_txt}\u00b2 "
                          f"\u00d7 (1 - {kv_force:.3f}) \u00d7 {r.KAE:.4f}"),
            result_name="PAE",
            result_value=f"{r.PAE_total:.2f}",
            result_unit="kN/m",
            reference="UFC 3-220-20 Eq. 4-16 (p. 215); Seed & Whitman (1970)",
            notes=("M-O has no solution here: the Seed & Whitman KAE "
                   "leaves k_v out, and so does PAE" if kv_force != r.kv
                   else ""),
        ))

        force_items.append(CalcStep(
            title="Static Active Force",
            equation="PA = 0.5 \u00d7 \u03b3 \u00d7 H\u00b2 \u00d7 Ka",
            substitution=f"PA = 0.5 \u00d7 {g_txt} \u00d7 {H_txt}\u00b2 \u00d7 {Ka_used:.4f}",
            result_name="PA",
            result_value=f"{r.PA_static:.2f}",
            result_unit="kN/m",
        ))

        force_items.append(CalcStep(
            title="Seismic Increment",
            equation="\u0394PAE = PAE - PA",
            substitution=f"\u0394PAE = {r.PAE_total:.2f} - {r.PA_static:.2f}",
            result_name="\u0394PAE",
            result_value=f"{r.delta_PAE:.2f}",
            result_unit="kN/m",
            reference="Seed & Whitman (1970)",
            notes="Applied at 0.6H above base per Seed-Whitman recommendation",
        ))

        force_items.append(CalcStep(
            title="Height of Application of Seismic Increment",
            equation="h_app = 0.6 \u00d7 H",
            substitution=f"h_app = 0.6 \u00d7 {H_txt}",
            result_name="h_app",
            result_value=f"{r.height_of_application:.2f}",
            result_unit="m",
            reference="Seed & Whitman (1970)",
        ))

        # Summary table
        force_items.append(TableData(
            title="Seismic Earth Pressure Summary",
            headers=["Quantity", "Value", "Unit"],
            rows=[
                ["KAE (seismic active)", f"{r.KAE:.4f}", ""],
                ["Ka (static active)", f"{Ka_used:.4f}", ""],
                ["PAE (total seismic)", f"{r.PAE_total:.2f}", "kN/m"],
                ["PA (static)", f"{r.PA_static:.2f}", "kN/m"],
                ["\u0394PAE (seismic increment)", f"{r.delta_PAE:.2f}", "kN/m"],
                ["Application height", f"{r.height_of_application:.2f}", "m above base"],
            ],
        ))

        sections.append(CalcSection(
            title="Seismic Earth Pressure Resultants", items=force_items
        ))

    return sections


# ── Liquefaction Calc Steps ─────────────────────────────────────

def _steps_liquefaction(result, analysis) -> List[CalcSection]:
    """Calc steps for liquefaction triggering evaluation."""
    r = result
    sections = []

    # Section: Method Overview
    method_items = []
    method_items.append(CalcStep(
        title="Cyclic Stress Ratio (CSR)",
        equation="CSR = 0.65 \u00d7 (a_max / g) \u00d7 (\u03c3_v / \u03c3'_v) \u00d7 r_d",
        substitution=f"a_max = {r.amax_g:.3f}g, M_w = {r.magnitude:.1f}",
        result_name="CSR",
        result_value="(computed per layer)",
        reference="FHWA GEC-13 Vol I Eq. 5-9 (p. 5-40); Seed & Idriss (1971)",
        notes="CSR adjusted to M=7.5 using magnitude scaling factor (MSF)",
    ))

    # MSF
    if abs(r.magnitude - 7.5) > 0.01:
        MSF = 10 ** 2.24 / (r.magnitude ** 2.56)
        method_items.append(CalcStep(
            title="Magnitude Scaling Factor",
            equation="MSF = 10^2.24 / M_w^2.56",
            substitution=f"MSF = 10^2.24 / {r.magnitude:.1f}^2.56",
            result_name="MSF",
            result_value=f"{MSF:.3f}",
            reference="Youd et al. (2001) Eq. 20",
            notes="CSR_M7.5 = CSR / MSF",
        ))

    method_items.append(CalcStep(
        title="Stress Reduction Factor (r_d)",
        equation=(
            "r_d = 1.0 - 0.00765\u00d7z  (z \u2264 9.144 m, 30 ft)\n"
            "r_d = 1.174 - 0.0267\u00d7z  (9.144 < z \u2264 22.86 m, 75 ft)\n"
            "r_d = 0.744 - 0.008\u00d7z  (22.86 < z \u2264 30.48 m, 100 ft)\n"
            "r_d = 0.5  (z > 30.48 m)"
        ),
        substitution="z in m (the coefficients are metric)",
        result_name="r_d",
        result_value="(depth dependent)",
        reference=("FHWA GEC-13 Vol I pp. 5-40 and 5-41; Liao & Whitman "
                   "(1986)"),
    ))

    method_items.append(CalcStep(
        title="Fines Content Correction for (N1)60",
        equation=(
            "(N1)60cs = \u03b1 + \u03b2 \u00d7 (N1)60\n"
            "\u03b1 = 0, \u03b2 = 1.0  for FC \u2264 5%\n"
            "\u03b1 = exp(1.76 - 190/FC\u00b2), \u03b2 = 0.99 + FC^1.5/1000  for 5 < FC < 35%\n"
            "\u03b1 = 5.0, \u03b2 = 1.2  for FC \u2265 35%"
        ),
        substitution="",
        result_name="(N1)60cs",
        result_value="(computed per layer)",
        reference="Youd et al. (2001) Eqs. 5-6",
    ))

    method_items.append(CalcStep(
        title="Cyclic Resistance Ratio (CRR) from (N1)60cs",
        equation=(
            "CRR_7.5 = 1/(34 - N160cs) + N160cs/135 "
            "+ 50/(10\u00d7N160cs + 45)\u00b2 - 1/200"
        ),
        substitution="CRR = 2.0 for (N1)60cs \u2265 30 (too dense to liquefy)",
        result_name="CRR",
        result_value="(computed per layer)",
        reference="Youd et al. (2001) Eq. 2 (NCEER deterministic curve)",
    ))

    method_items.append(CalcStep(
        title="Factor of Safety Against Liquefaction",
        equation="FOS_liq = CRR / CSR",
        substitution="Liquefiable if FOS < 1.0",
        result_name="FOS_liq",
        result_value="(computed per layer)",
        reference="Youd et al. (2001)",
    ))

    sections.append(CalcSection(
        title="Liquefaction Evaluation Method", items=method_items
    ))

    # Section: Layer-by-Layer Results
    if r.layer_results:
        layer_rows = []
        for lr in r.layer_results:
            liq_str = "YES" if lr["liquefiable"] else "no"
            layer_rows.append([
                f"{lr['depth_m']:.1f}",
                f"{lr['N160']:.0f}",
                f"{lr.get('FC_pct', 0):.0f}",
                f"{lr['N160cs']:.1f}",
                f"{lr.get('sigma_v_kPa', 0):.1f}",
                f"{lr.get('sigma_v_eff_kPa', 0):.1f}",
                f"{lr['CSR']:.4f}",
                f"{lr['CRR']:.4f}",
                f"{lr['FOS_liq']:.3f}",
                liq_str,
            ])

        layer_table = TableData(
            title="Layer-by-Layer Liquefaction Results",
            headers=[
                "Depth (m)", "(N1)60", "FC (%)", "(N1)60cs",
                "\u03c3_v (kPa)", "\u03c3'_v (kPa)",
                "CSR", "CRR", "FOS", "Liq?"
            ],
            rows=layer_rows,
        )

        sections.append(CalcSection(
            title="Layer-by-Layer Results", items=[layer_table]
        ))

    # Section: Summary & Checks
    summary_items = []

    summary_items.append(TableData(
        title="Liquefaction Evaluation Summary",
        headers=["Parameter", "Value"],
        rows=[
            ["Total layers evaluated", f"{len(r.layer_results)}"],
            ["Liquefiable layers", f"{r.n_liquefiable}"],
            ["Minimum FOS", f"{r.min_FOS:.3f}"],
            ["Critical depth", f"{r.critical_depth:.1f} m" if r.critical_depth is not None else "N/A"],
        ],
    ))

    summary_items.append(CheckItem(
        description="Liquefaction triggering (FOS \u2265 1.0 at all depths)",
        demand=1.0,
        demand_label="FOS_required",
        capacity=r.min_FOS,
        capacity_label="FOS_min",
        unit="",
        passes=r.min_FOS >= 1.0,
    ))

    sections.append(CalcSection(
        title="Liquefaction Summary & Check", items=summary_items
    ))

    return sections


# ═══════════════════════════════════════════════════════════════════
# get_figures
# ═══════════════════════════════════════════════════════════════════

def get_figures(result, analysis) -> List[FigureData]:
    """Generate figures for seismic geotechnical calc package.

    Parameters
    ----------
    result : SiteClassResult | SeismicEarthPressureResult | LiquefactionResult
    analysis : dict

    Returns
    -------
    list of FigureData
    """
    rtype = _detect_result_type(result)
    if rtype == "site_classification":
        return _figures_site_class(result, analysis)
    elif rtype == "seismic_earth_pressure":
        return _figures_earth_pressure(result, analysis)
    elif rtype == "liquefaction":
        return _figures_liquefaction(result, analysis)
    return []


# ── Site Classification Figures ─────────────────────────────────

def _figures_site_class(result, analysis) -> List[FigureData]:
    """Figures for site classification: design response spectrum."""
    figures = []
    if result.Ss is None:
        return figures
    try:
        fig = _plot_design_spectrum(result)
        b64 = figure_to_base64(fig, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        figures.append(FigureData(
            title="Design Response Spectrum",
            image_base64=b64,
            caption=(
                f"Figure 1: Design response spectrum (GEC-12 Fig. 7-77, "
                f"after AASHTO 2014) for Site Class {result.site_class}. "
                f"SDS = {result.Fa * result.Ss:.4f} g, "
                f"SD1 = {result.Fv * result.S1:.4f} g"
                + (f", As = {result.As:.4f} g." if result.As is not None
                   else "; no PGA given, so no As and the line below T0 "
                   "is not drawn.")
            ),
            width_percent=80,
        ))
    except ImportError:
        pass

    return figures


def _plot_design_spectrum(result):
    """The design response spectrum of GEC-12 Vol I Fig. 7-77 (p. 381,
    after AASHTO 2014): Csm = As + (SDS - As)(Tm / T0) below T0 (Eq. 7-77,
    As = Fpga PGA, Eq. 7-78), SDS from T0 to Ts, SD1 / Tm beyond, with
    T0 = 0.2 Ts and Ts = SD1 / SDS.

    Until GM-293 the line below T0 was the ASCE 7 form SDS (0.4 + 0.6 T /
    T0), starting from 0.4 Fpga Ss rather than As. Without the PGA there
    is no As, and the line below T0 is not drawn."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    SDS = result.Fa * result.Ss
    SD1 = result.Fv * result.S1
    As = result.As

    # GEC-12 Fig. 7-77
    T0 = 0.2 * SD1 / SDS if SDS > 0 else 0.1
    Ts = SD1 / SDS if SDS > 0 else 0.5

    # Period array; from T0 when there is no As to start from
    periods = np.concatenate([
        np.array([0.0]) if As is not None else np.array([]),
        np.linspace(T0, Ts, 20),
        np.linspace(Ts, 4.0, 100),
    ])

    Sa = np.zeros_like(periods)
    for i, T in enumerate(periods):
        if T < T0:
            Sa[i] = As + (SDS - As) * T / T0      # Eq. 7-77
        elif T <= Ts:
            Sa[i] = SDS
        else:
            Sa[i] = SD1 / T

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(periods, Sa, 'b-', linewidth=2, label='Design spectrum')

    # Mark key points
    if As is not None:
        ax.plot(0, As, 'ro', markersize=6, label=f'As = {As:.3f} g')
    ax.plot(T0, SDS, 'go', markersize=8, label=f'T0 = {T0:.2f} s')
    ax.plot(Ts, SDS, 'ms', markersize=8, label=f'Ts = {Ts:.2f} s')

    # Horizontal/vertical reference lines
    ax.axhline(y=SDS, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=T0, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=Ts, color='gray', linestyle=':', alpha=0.3)

    # Annotations: SDS over its own line at the right, where the spectrum
    # has fallen away (over the plateau it ran across the T0 line at the
    # printed width; GM-194)
    ax.annotate(f'SDS = {SDS:.3f} g',
                xy=(1, SDS), xycoords=('axes fraction', 'data'),
                xytext=(-4, 3), textcoords='offset points',
                fontsize=9, ha='right', va='bottom', fontweight='bold')
    if Ts < 3.5:
        ax.annotate(f'SD1 = {SD1:.3f} g',
                    xy=(1.0, SD1 / 1.0 if 1.0 > Ts else SDS),
                    xytext=(15, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Period T (s)', fontsize=11)
    ax.set_ylabel('Spectral Acceleration Sa (g)', fontsize=11)
    ax.set_title(
        f'Design Response Spectrum (GEC-12 Fig. 7-77) \u2014 Site Class '
        f'{result.site_class}',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, max(SDS * 1.3, 0.1))
    ax.grid(True, alpha=0.3)

    # Info box
    info_text = (
        f"Site Class: {result.site_class}\n"
        f"Ss = {result.Ss:.3f} g\n"
        f"S1 = {result.S1:.3f} g\n"
        f"Fa = {result.Fa:.3f}\n"
        f"Fv = {result.Fv:.3f}"
    )
    ax.text(0.98, 0.55, info_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                     edgecolor='gray', alpha=0.9))

    # the key under the axes, clear of the SDS note
    from geotech_common.plotting import key_under_axes
    key_under_axes(fig, ax, fontsize=9, ncol=3)
    return fig


# ── Mononobe-Okabe Figures ──────────────────────────────────────

def _ka_used(a, phi):
    """(the static KA behind PA and delta-PAE, whether it was worked out):
    the seismic_earth_pressure run's KA (it passes KA and KA_given), else a
    KA given in the analysis, else Rankine Ka from phi, as the tool works
    it out (GM-120)."""
    KA = a.get("KA")
    if KA is None:
        return math.tan(math.pi / 4 - math.radians(phi) / 2) ** 2, True
    return float(KA), a.get("KA_given") is False


def _figures_earth_pressure(result, analysis) -> List[FigureData]:
    """Figures for Mononobe-Okabe: pressure diagram."""
    r = result
    a = analysis or {}
    figures = []

    if r.PAE_total > 0 and a.get("H") is not None:
        try:
            fig = _plot_pressure_diagram(r, a)
            b64 = figure_to_base64(fig, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig)
            figures.append(FigureData(
                title="Seismic Earth Pressure Diagram",
                image_base64=b64,
                caption=(
                    f"Figure 1: Static active earth pressure (kPa) and the "
                    f"seismic increment \u0394PAE = {r.delta_PAE:.1f} kN/m "
                    f"applied at {r.height_of_application:.2f} m above the "
                    f"base (0.6H, Seed & Whitman 1970); "
                    f"PAE = {r.PAE_total:.1f} kN/m."
                ),
                width_percent=75,
            ))
        except ImportError:
            pass

    return figures


def _plot_pressure_diagram(result, analysis):
    """The static active earth pressure on the wall in kPa, PA at H/3 and
    the seismic increment \u0394PAE at 0.6H.

    Pressures in kPa, the owner's convention (GM-200; the axis read
    "Pressure (normalized)"). The static pressure is the triangle whose
    resultant is the run's PA: K_A \u03b3 z, 2 PA / H at the base, so it is
    drawn from PA and H whether or not \u03b3 was given. \u0394PAE is drawn as the
    force the method gives at its point of action, 0.6H above the base
    (Seed & Whitman 1970; DM7.2 Ch. 4, "assumed to act at 0.6H above the
    wall base"), not as a pressure: the method gives no distribution for
    it, and PAE drawn as one triangle puts its resultant at H/3."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    r = result
    H = analysis["H"]          # the caller draws this only with H known
    # The KA the run used behind PA: as given, else Rankine Ka from phi
    # (GM-120).
    Ka_static, _ = _ka_used(analysis, r.phi)
    p_base = 2.0 * r.PA_static / H          # K_A \u03b3 H (kPa)
    # the drawing's pressure scale: the static pressure at the base, or
    # PAE's equivalent where PA is nil
    P = p_base if p_base > 0 else max(2.0 * r.PAE_total / H, 1e-6)

    # drawn at the width it prints at (5 in), so what is placed here in
    # points is where it prints (GM-186)
    fig, ax = plt.subplots(figsize=(5.0, 4.2))

    # Wall: its back face at zero pressure, the soil to the right
    ax.fill_betweenx([0, H], -0.06 * P, 0, color='#666', alpha=0.6,
                     linewidth=0)
    ax.plot([0, 0], [0, H], 'k-', linewidth=2.5)

    # Static active pressure, zero at the top to K_A \u03b3 H at the base
    ax.fill_betweenx([0, H], 0, [p_base, 0], color='#2563eb', alpha=0.2,
                     linewidth=0,
                     label=f'Static active pressure (K_A = {Ka_static:.3f})')
    ax.plot([p_base, 0], [0, H], color='#2563eb', linewidth=1.5)
    ax.annotate(f'K_A\u03b3H = {p_base:.1f} kPa', (p_base, 0),
                xytext=(4, 3), textcoords='offset points', ha='left',
                va='bottom', fontsize=7.5, color='#2563eb')

    # The forces, as arrows onto the wall at their heights (their length
    # is not to the pressure scale), each named at its tail
    tail = 0.95 * P
    for y, colour, text in (
            (H / 3.0, '#2563eb',
             f'PA = {r.PA_static:.1f} kN/m\nat H/3 = {H / 3.0:.2f} m'),
            (r.height_of_application, '#dc2626',
             f'\u0394PAE = {r.delta_PAE:.1f} kN/m\n'
             f'at 0.6H = {r.height_of_application:.2f} m')):
        ax.annotate('', xy=(0, y), xytext=(tail, y),
                    arrowprops=dict(arrowstyle='->', color=colour, lw=1.8))
        ax.annotate(text, (tail, y), xytext=(4, 0),
                    textcoords='offset points', ha='left', va='center',
                    fontsize=7.5, color=colour)

    # H dimension, left of the wall
    ax.annotate('', xy=(-0.17 * P, 0), xytext=(-0.17 * P, H),
                arrowprops=dict(arrowstyle='<->', color='k', lw=1.0))
    ax.annotate(f'H = {H:.2f} m', (-0.17 * P, H / 2.0), xytext=(-4, 0),
                textcoords='offset points', ha='right', va='center',
                fontsize=8, fontweight='bold', rotation=90)

    ax.set_xlim(-0.32 * P, 1.6 * P)
    ax.set_ylim(-0.04 * H, H * 1.06)
    # pressures are read off the soil side only
    ax.set_xticks([t for t in MaxNLocator(5).tick_values(0, 1.6 * P)
                   if -1e-9 <= t <= 1.6 * P])
    ax.set_xlabel('Lateral earth pressure (kPa)', fontsize=9)
    ax.set_ylabel('Height above the wall base (m)', fontsize=9)
    ax.set_title('Seismic Earth Pressure (Mononobe-Okabe)', fontsize=10,
                 fontweight='bold')
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)

    # The inputs and PAE head the key, under the drawing: in a box in the
    # axes they covered the H dimension at the printed width (GM-194)
    info = (
        f"\u03c6 = {r.phi:.0f}\u00b0, \u03b4 = {r.delta:.0f}\u00b0, "
        f"k_h = {r.kh:.3f}, k_v = {r.kv:.3f}"
    )
    if analysis.get("gamma") is not None:
        info += f", \u03b3 = {analysis['gamma']:.1f} kN/m\u00b3"
    info += (f"\nK_AE = {r.KAE:.3f}: PAE = {r.PAE_total:.1f} kN/m "
             f"= PA + \u0394PAE")
    from geotech_common.plotting import key_under_axes
    key_under_axes(fig, ax, note=info)
    return fig


# ── Liquefaction Figures ────────────────────────────────────────

def _figures_liquefaction(result, analysis) -> List[FigureData]:
    """Figures for liquefaction: FOS profile and CSR/CRR comparison."""
    r = result
    figures = []

    if not r.layer_results:
        return figures

    # Figure 1: FOS vs depth profile
    try:
        fig1 = _plot_fos_profile(r)
        b64 = figure_to_base64(fig1, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig1)
        figures.append(FigureData(
            title="Liquefaction Factor of Safety vs. Depth",
            image_base64=b64,
            caption=(
                f"Figure 1: Factor of safety against liquefaction vs. depth. "
                f"PGA = {r.amax_g:.3f}g, M_w = {r.magnitude:.1f}. "
                f"Minimum FOS = {r.min_FOS:.3f} at depth "
                f"{r.critical_depth:.1f} m."
                if r.critical_depth is not None else
                f"Figure 1: Factor of safety against liquefaction vs. depth. "
                f"PGA = {r.amax_g:.3f}g, M_w = {r.magnitude:.1f}."
            ),
            width_percent=75,
        ))
    except ImportError:
        pass

    # Figure 2: CSR/CRR comparison
    try:
        fig2 = _plot_csr_crr(r)
        b64 = figure_to_base64(fig2, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig2)
        figures.append(FigureData(
            title="CSR and CRR vs. Depth",
            image_base64=b64,
            caption=(
                f"Figure 2: Cyclic stress ratio (CSR) and cyclic resistance "
                f"ratio (CRR) vs. depth. Layers where CSR > CRR are liquefiable. "
                f"{r.n_liquefiable} of {len(r.layer_results)} layers liquefiable."
            ),
            width_percent=75,
        ))
    except ImportError:
        pass

    return figures


def _plot_fos_profile(result):
    """Plot FOS against liquefaction vs. depth."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    depths = [lr["depth_m"] for lr in result.layer_results]
    fos_vals = [lr["FOS_liq"] for lr in result.layer_results]
    liquefiable = [lr["liquefiable"] for lr in result.layer_results]

    fig, ax = plt.subplots(figsize=(6, 7))

    # Cap display FOS for readability
    fos_display = [min(f, 3.0) for f in fos_vals]

    # Color by status
    colors = ['#dc2626' if liq else '#16a34a' for liq in liquefiable]

    ax.barh(depths, fos_display, height=0.4, color=colors,
            edgecolor='#333', linewidth=0.5, alpha=0.8)

    # FOS = 1.0 line
    ax.axvline(x=1.0, color='#dc2626', linestyle='--', linewidth=2,
               label='FOS = 1.0')

    # Value labels, set off from the bar by points and ringed, so the
    # FOS = 1.0 and water-table lines stop short of a value that sits
    # on them (set off by 0.05 in FOS, a value near 1.0 lay on its line;
    # GM-194)
    from geotech_common.plotting import halo
    values = []
    for d, f, f_display in zip(depths, fos_vals, fos_display):
        label = f"{f:.2f}" if f <= 3.0 else f"{f:.1f}+"
        values.append(ax.annotate(label, (f_display, d), xytext=(3, 0),
                                  textcoords='offset points', va='center',
                                  fontsize=8))
    halo(values)

    # GWT line
    ax.axhline(y=result.gwt_depth, color='#2563eb', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f'GWT = {result.gwt_depth:.1f} m')

    ax.set_xlabel('Factor of Safety (FOS_liq)', fontsize=11)
    ax.set_ylabel('Depth (m)', fontsize=11)
    ax.set_title(
        f'Liquefaction FOS Profile \u2014 PGA = {result.amax_g:.3f}g, '
        f'M_w = {result.magnitude:.1f}',
        fontsize=11, fontweight='bold'
    )
    ax.invert_yaxis()
    ax.set_xlim(0, max(fos_display) * 1.3)
    ax.grid(True, alpha=0.3, axis='x')

    # The status heads the key, under the axes: the box and the key in
    # the corner covered each other and the deepest layer's value (GM-194)
    status = "LIQUEFIABLE" if result.n_liquefiable > 0 else "NON-LIQUEFIABLE"
    status_color = '#dc2626' if result.n_liquefiable > 0 else '#16a34a'
    from geotech_common.plotting import key_under_axes
    key = key_under_axes(
        fig, ax, fontsize=9,
        note=(f"{status}: {result.n_liquefiable}/"
              f"{len(result.layer_results)} layers"))
    key.get_title().set_color(status_color)
    key.get_title().set_fontweight('bold')
    return fig


def _plot_csr_crr(result):
    """Plot CSR and CRR profiles vs depth."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    depths = [lr["depth_m"] for lr in result.layer_results]
    csr_vals = [lr["CSR"] for lr in result.layer_results]
    crr_vals = [lr["CRR"] for lr in result.layer_results]
    liquefiable = [lr["liquefiable"] for lr in result.layer_results]

    fig, ax = plt.subplots(figsize=(6, 7))

    ax.plot(csr_vals, depths, 'rs-', linewidth=2, markersize=7,
            label='CSR (demand)', markerfacecolor='#dc2626')
    ax.plot(crr_vals, depths, 'go-', linewidth=2, markersize=7,
            label='CRR (capacity)', markerfacecolor='#16a34a')

    # Shade liquefiable layers
    for i, (d, csr, crr, liq) in enumerate(zip(depths, csr_vals, crr_vals, liquefiable)):
        if liq:
            ax.fill_betweenx([d - 0.3, d + 0.3], 0, max(csr, crr) * 1.1,
                              color='#dc2626', alpha=0.08)

    # GWT line
    ax.axhline(y=result.gwt_depth, color='#2563eb', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f'GWT = {result.gwt_depth:.1f} m')

    ax.set_xlabel('Cyclic Stress / Resistance Ratio', fontsize=11)
    ax.set_ylabel('Depth (m)', fontsize=11)
    ax.set_title(
        f'CSR vs CRR Profile \u2014 PGA = {result.amax_g:.3f}g, '
        f'M_w = {result.magnitude:.1f}',
        fontsize=11, fontweight='bold'
    )
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(csr_vals), max(crr_vals)) * 1.3)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
