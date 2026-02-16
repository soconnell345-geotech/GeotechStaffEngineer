"""
Calculation package steps for seismic geotechnical analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation.

Handles three result types:
    - SiteClassResult: AASHTO/NEHRP site classification and coefficients
    - SeismicEarthPressureResult: Mononobe-Okabe seismic earth pressures
    - LiquefactionResult: SPT-based liquefaction triggering evaluation

The `analysis` parameter should be a dict with:
    - "analysis_type": one of "site_classification", "seismic_earth_pressure",
      "liquefaction"
    - Plus type-specific keys echoing the analysis inputs.

References:
    AASHTO LRFD Bridge Design Specifications, 9th Ed., Section 3.10.3
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
    'AASHTO LRFD Bridge Design Specifications, 9th Ed., Section 3.10.3.',
    'NEHRP Recommended Seismic Provisions (FEMA P-1050).',
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

    items.append(InputItem("Ss", "Spectral acceleration at 0.2 s", f"{result.Ss:.3f}", "g"))
    items.append(InputItem("S1", "Spectral acceleration at 1.0 s", f"{result.S1:.3f}", "g"))

    # Echo layer data if provided in analysis dict
    a = analysis or {}
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
    if "beta" in a:
        items.append(InputItem("\u03b2", "Wall batter from vertical", a["beta"], "deg"))
    if "i" in a:
        items.append(InputItem("i", "Backfill slope angle", a["i"], "deg"))
    if "gamma" in a:
        items.append(InputItem("\u03b3", "Backfill unit weight", a["gamma"], "kN/m\u00b3"))
    if "H" in a:
        items.append(InputItem("H", "Wall height", a["H"], "m"))

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
            reference="AASHTO LRFD Table 3.10.3.1-1",
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
            reference="AASHTO LRFD Table 3.10.3.1-1",
        ))
    if result.su_bar is not None:
        param_items.append(CalcStep(
            title="Average Undrained Strength (su-bar)",
            equation="su-bar = \u03a3d_i / \u03a3(d_i / su_i)  (harmonic mean)",
            substitution="",
            result_name="su-bar",
            result_value=f"{result.su_bar:.0f}",
            result_unit="kPa",
            reference="AASHTO LRFD Table 3.10.3.1-1",
        ))

    if param_items:
        sections.append(CalcSection(
            title="Site Parameter Computation", items=param_items
        ))

    # Section: Site Classification
    class_items = []

    # Classification criteria table
    class_table = TableData(
        title="AASHTO/NEHRP Site Classification Criteria",
        headers=["Site Class", "Vs30 (m/s)", "N-bar", "su-bar (kPa)"],
        rows=[
            ["A", "> 1500", "\u2014", "\u2014"],
            ["B", "760 \u2013 1500", "\u2014", "\u2014"],
            ["C", "360 \u2013 760", "> 50", "> 100"],
            ["D", "180 \u2013 360", "15 \u2013 50", "50 \u2013 100"],
            ["E", "< 180", "< 15", "< 50"],
            ["F", "Site-specific", "Site-specific", "Site-specific"],
        ],
        notes="Priority: Vs30 > N-bar > su-bar per AASHTO LRFD 3.10.3.1",
    )
    class_items.append(class_table)

    class_items.append(CalcStep(
        title="Site Classification Result",
        equation="Classify based on Vs30, N-bar, or su-bar per AASHTO Table 3.10.3.1-1",
        substitution=_site_class_substitution(result),
        result_name="Site Class",
        result_value=result.site_class,
        reference="AASHTO LRFD Table 3.10.3.1-1",
    ))
    sections.append(CalcSection(title="Site Classification", items=class_items))

    # Section: Site Coefficients
    coeff_items = []

    coeff_items.append(CalcStep(
        title="Site Coefficient Fpga (PGA amplification)",
        equation="Fpga = f(Site Class, Ss) from AASHTO Table 3.10.3.2-1",
        substitution=f"Site Class {result.site_class}, Ss = {result.Ss:.3f} g",
        result_name="Fpga",
        result_value=f"{result.Fpga:.3f}",
        reference="AASHTO LRFD Table 3.10.3.2-1",
    ))

    coeff_items.append(CalcStep(
        title="Site Coefficient Fa (short-period amplification)",
        equation="Fa = f(Site Class, Ss) from AASHTO Table 3.10.3.2-1",
        substitution=f"Site Class {result.site_class}, Ss = {result.Ss:.3f} g",
        result_name="Fa",
        result_value=f"{result.Fa:.3f}",
        reference="AASHTO LRFD Table 3.10.3.2-1",
    ))

    coeff_items.append(CalcStep(
        title="Site Coefficient Fv (long-period amplification)",
        equation="Fv = f(Site Class, S1) from AASHTO Table 3.10.3.2-3",
        substitution=f"Site Class {result.site_class}, S1 = {result.S1:.3f} g",
        result_name="Fv",
        result_value=f"{result.Fv:.3f}",
        reference="AASHTO LRFD Table 3.10.3.2-3",
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
        reference="AASHTO LRFD Eq. 3.10.4.2-1",
    ))

    design_items.append(CalcStep(
        title="Design Long-Period Spectral Acceleration",
        equation="SD1 = Fv \u00d7 S1",
        substitution=f"SD1 = {result.Fv:.3f} \u00d7 {result.S1:.3f}",
        result_name="SD1",
        result_value=f"{SD1:.4f}",
        result_unit="g",
        reference="AASHTO LRFD Eq. 3.10.4.2-2",
    ))

    # Summary table
    design_items.append(TableData(
        title="Design Spectral Parameters Summary",
        headers=["Parameter", "Value", "Unit"],
        rows=[
            ["Site Class", result.site_class, ""],
            ["Fpga", f"{result.Fpga:.3f}", ""],
            ["Fa", f"{result.Fa:.3f}", ""],
            ["Fv", f"{result.Fv:.3f}", ""],
            ["SDS = Fa \u00d7 Ss", f"{SDS:.4f}", "g"],
            ["SD1 = Fv \u00d7 S1", f"{SD1:.4f}", "g"],
        ],
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
            "KAE = cos\u00b2(\u03c6 - \u03b8 - \u03b2) / "
            "[cos\u03b8 \u00d7 cos\u00b2\u03b2 \u00d7 cos(\u03b4 + \u03b2 + \u03b8) \u00d7 "
            "(1 + \u221a(sin(\u03c6+\u03b4)\u00d7sin(\u03c6-\u03b8-i) / "
            "(cos(\u03b4+\u03b2+\u03b8)\u00d7cos(i-\u03b2))))\u00b2]"
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

    # Static Ka for comparison
    Ka_static = math.tan(math.pi / 4 - math.radians(r.phi) / 2) ** 2
    kae_items.append(CalcStep(
        title="Static Active Coefficient (Rankine Ka, for comparison)",
        equation="Ka = tan\u00b2(45\u00b0 - \u03c6/2)",
        substitution=f"Ka = tan\u00b2(45\u00b0 - {r.phi:.1f}\u00b0/2)",
        result_name="Ka",
        result_value=f"{Ka_static:.4f}",
        reference="Rankine (1857)",
        notes=f"KAE/Ka = {r.KAE/Ka_static:.2f}" if Ka_static > 0 else "",
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
                "KPE = cos\u00b2(\u03c6 - \u03b8 + \u03b2) / "
                "[cos\u03b8 \u00d7 cos\u00b2\u03b2 \u00d7 cos(\u03b4 - \u03b2 + \u03b8) \u00d7 "
                "(1 - \u221a(sin(\u03c6+\u03b4)\u00d7sin(\u03c6-\u03b8+i) / "
                "(cos(\u03b4-\u03b2+\u03b8)\u00d7cos(i-\u03b2))))\u00b2]"
            ),
            substitution=(
                f"\u03c6 = {r.phi:.1f}\u00b0, \u03b4 = {r.delta:.1f}\u00b0, "
                f"k_h = {r.kh:.3f}, k_v = {r.kv:.3f}"
            ),
            result_name="KPE",
            result_value=f"{r.KPE:.4f}",
            reference="AASHTO LRFD Eq. 11.6.5.2; Mononobe-Okabe (1929)",
        ))
        sections.append(CalcSection(
            title="Seismic Passive Earth Pressure Coefficient", items=kpe_items
        ))

    # Section: Pressure Resultants (if H and gamma available)
    if r.PAE_total > 0:
        gamma = a.get("gamma", 0.0)
        H = a.get("H", 0.0)

        force_items = []
        force_items.append(CalcStep(
            title="Total Seismic Active Force",
            equation="PAE = 0.5 \u00d7 \u03b3 \u00d7 H\u00b2 \u00d7 KAE",
            substitution=f"PAE = 0.5 \u00d7 {gamma:.1f} \u00d7 {H:.2f}\u00b2 \u00d7 {r.KAE:.4f}",
            result_name="PAE",
            result_value=f"{r.PAE_total:.2f}",
            result_unit="kN/m",
            reference="Seed & Whitman (1970); FHWA GEC-3",
        ))

        force_items.append(CalcStep(
            title="Static Active Force",
            equation="PA = 0.5 \u00d7 \u03b3 \u00d7 H\u00b2 \u00d7 Ka",
            substitution=f"PA = 0.5 \u00d7 {gamma:.1f} \u00d7 {H:.2f}\u00b2 \u00d7 {Ka_static:.4f}",
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
            substitution=f"h_app = 0.6 \u00d7 {H:.2f}",
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
                ["Ka (static active)", f"{Ka_static:.4f}", ""],
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
        reference="Seed & Idriss (1971); Youd et al. (2001) Eq. 1",
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
            "r_d = 1.0 - 0.00765\u00d7z  (z \u2264 9.15 m)\n"
            "r_d = 1.174 - 0.0267\u00d7z  (9.15 < z \u2264 23 m)"
        ),
        substitution="",
        result_name="r_d",
        result_value="(depth dependent)",
        reference="Liao & Whitman (1986); Youd et al. (2001) Eq. 4",
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
    try:
        fig = _plot_design_spectrum(result)
        b64 = figure_to_base64(fig, dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)
        figures.append(FigureData(
            title="Design Response Spectrum",
            image_base64=b64,
            caption=(
                f"Figure 1: AASHTO design response spectrum for Site Class "
                f"{result.site_class}. SDS = {result.Fa * result.Ss:.4f} g, "
                f"SD1 = {result.Fv * result.S1:.4f} g."
            ),
            width_percent=80,
        ))
    except ImportError:
        pass

    return figures


def _plot_design_spectrum(result):
    """Plot AASHTO/ASCE design response spectrum."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    SDS = result.Fa * result.Ss
    SD1 = result.Fv * result.S1

    # Build spectrum per AASHTO
    T0 = 0.2 * SD1 / SDS if SDS > 0 else 0.1
    Ts = SD1 / SDS if SDS > 0 else 0.5

    # Period array
    periods = np.concatenate([
        np.array([0.0, T0]),
        np.linspace(T0, Ts, 20),
        np.linspace(Ts, 4.0, 100),
    ])

    Sa = np.zeros_like(periods)
    for i, T in enumerate(periods):
        if T <= 0:
            Sa[i] = result.Fpga * result.Ss * 0.4 + 0.6 * SDS * T / T0 if T0 > 0 else SDS
        elif T <= T0:
            Sa[i] = SDS * (0.4 + 0.6 * T / T0) if T0 > 0 else SDS
        elif T <= Ts:
            Sa[i] = SDS
        else:
            Sa[i] = SD1 / T

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(periods, Sa, 'b-', linewidth=2, label='Design spectrum')

    # Mark key points
    ax.plot(0, Sa[0], 'ro', markersize=6)
    ax.plot(T0, SDS, 'go', markersize=8, label=f'T0 = {T0:.2f} s')
    ax.plot(Ts, SDS, 'ms', markersize=8, label=f'Ts = {Ts:.2f} s')

    # Horizontal/vertical reference lines
    ax.axhline(y=SDS, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=T0, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(x=Ts, color='gray', linestyle=':', alpha=0.3)

    # Annotations
    ax.annotate(f'SDS = {SDS:.3f} g',
                xy=(T0 + (Ts - T0) / 2, SDS),
                xytext=(0, 15), textcoords='offset points',
                fontsize=9, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='gray'))
    if Ts < 3.5:
        ax.annotate(f'SD1 = {SD1:.3f} g',
                    xy=(1.0, SD1 / 1.0 if 1.0 > Ts else SDS),
                    xytext=(15, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Period T (s)', fontsize=11)
    ax.set_ylabel('Spectral Acceleration Sa (g)', fontsize=11)
    ax.set_title(
        f'AASHTO Design Response Spectrum \u2014 Site Class {result.site_class}',
        fontsize=12, fontweight='bold'
    )
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, max(SDS * 1.3, 0.1))
    ax.legend(loc='upper right', fontsize=9)
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

    plt.tight_layout()
    return fig


# ── Mononobe-Okabe Figures ──────────────────────────────────────

def _figures_earth_pressure(result, analysis) -> List[FigureData]:
    """Figures for Mononobe-Okabe: pressure diagram."""
    r = result
    a = analysis or {}
    figures = []

    if r.PAE_total > 0 and "H" in a:
        try:
            fig = _plot_pressure_diagram(r, a)
            b64 = figure_to_base64(fig, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig)
            figures.append(FigureData(
                title="Seismic Earth Pressure Diagram",
                image_base64=b64,
                caption=(
                    f"Figure 1: Seismic earth pressure distribution. "
                    f"PAE = {r.PAE_total:.1f} kN/m, "
                    f"\u0394PAE = {r.delta_PAE:.1f} kN/m applied at "
                    f"{r.height_of_application:.1f} m above base."
                ),
                width_percent=75,
            ))
        except ImportError:
            pass

    return figures


def _plot_pressure_diagram(result, analysis):
    """Plot seismic earth pressure distribution on wall."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    r = result
    H = analysis.get("H", 6.0)
    gamma = analysis.get("gamma", 18.0)
    Ka_static = math.tan(math.pi / 4 - math.radians(r.phi) / 2) ** 2

    fig, ax = plt.subplots(figsize=(8, 6))

    # Wall
    ax.plot([0, 0], [0, H], 'k-', linewidth=4)
    ax.fill_betweenx([0, H], -0.15, 0, color='#666', alpha=0.6)

    # Static pressure triangle
    z_vals = np.linspace(0, H, 50)
    p_static = Ka_static * gamma * (H - z_vals)  # pressure at elevation z from base
    ax.fill_betweenx(z_vals, 0, p_static / max(p_static) * H * 0.4,
                      color='#2563eb', alpha=0.3, label=f'Static PA (Ka={Ka_static:.3f})')
    ax.plot(p_static / max(p_static) * H * 0.4, z_vals, 'b-', linewidth=1.5)

    # Seismic pressure (M-O)
    p_seismic = r.KAE * gamma * (H - z_vals)
    ax.fill_betweenx(z_vals, 0, p_seismic / max(p_seismic) * H * 0.55,
                      color='#dc2626', alpha=0.2, label=f'Seismic PAE (KAE={r.KAE:.3f})')
    ax.plot(p_seismic / max(p_seismic) * H * 0.55, z_vals, 'r-', linewidth=1.5)

    # Force arrows
    # Static PA at H/3
    pa_y = H / 3
    ax.annotate('', xy=(0, pa_y), xytext=(H * 0.35, pa_y),
                arrowprops=dict(arrowstyle='->', color='#2563eb', lw=2))
    ax.text(H * 0.36, pa_y, f'PA = {r.PA_static:.1f} kN/m\nat H/3 = {pa_y:.1f} m',
            fontsize=8, color='#2563eb', va='center')

    # Seismic increment at 0.6H
    dPAE_y = r.height_of_application
    ax.annotate('', xy=(0, dPAE_y), xytext=(H * 0.45, dPAE_y),
                arrowprops=dict(arrowstyle='->', color='#dc2626', lw=2))
    ax.text(H * 0.46, dPAE_y,
            f'\u0394PAE = {r.delta_PAE:.1f} kN/m\nat 0.6H = {dPAE_y:.1f} m',
            fontsize=8, color='#dc2626', va='center')

    # Soil fill behind wall
    ax.add_patch(patches.Rectangle(
        (0, 0), H * 0.6, H,
        facecolor='#f5e6c8', edgecolor='none', alpha=0.3, zorder=0,
    ))

    # H dimension
    ax.annotate('', xy=(-0.3, 0), xytext=(-0.3, H),
                arrowprops=dict(arrowstyle='<->', color='k', lw=1.2))
    ax.text(-0.5, H / 2, f'H = {H:.1f} m',
            ha='right', va='center', fontsize=9, fontweight='bold', rotation=90)

    ax.set_xlabel('Pressure (normalized)', fontsize=10)
    ax.set_ylabel('Elevation (m)', fontsize=10)
    ax.set_title('Seismic Earth Pressure Distribution (Mononobe-Okabe)',
                 fontsize=11, fontweight='bold')
    ax.set_xlim(-0.8, H * 0.7)
    ax.set_ylim(-0.3, H * 1.1)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.2)

    # Info box
    info = (
        f"\u03c6 = {r.phi:.0f}\u00b0, \u03b4 = {r.delta:.0f}\u00b0\n"
        f"k_h = {r.kh:.3f}, k_v = {r.kv:.3f}\n"
        f"\u03b3 = {gamma:.1f} kN/m\u00b3"
    )
    ax.text(0.02, 0.98, info, transform=ax.transAxes,
            fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    plt.tight_layout()
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

    # Value labels
    for d, f, f_display in zip(depths, fos_vals, fos_display):
        label = f"{f:.2f}" if f <= 3.0 else f"{f:.1f}+"
        ax.text(f_display + 0.05, d, label, va='center', fontsize=8)

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
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    # Status box
    status = "LIQUEFIABLE" if result.n_liquefiable > 0 else "NON-LIQUEFIABLE"
    status_color = '#dc2626' if result.n_liquefiable > 0 else '#16a34a'
    ax.text(0.98, 0.02,
            f"{status}\n{result.n_liquefiable}/{len(result.layer_results)} layers",
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            ha='right', va='bottom', color=status_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor=status_color, alpha=0.9))

    plt.tight_layout()
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
