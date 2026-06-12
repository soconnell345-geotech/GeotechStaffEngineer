"""
Calculation package steps for slope stability analysis.

Renders the modern LE engine (rigorous GLE/Spencer/M-P interslice
output, Janbu correction, reinforcement, probabilistic FOS, critical
surface search) into Mathcad-style calc package sections with
SLOPE/W-style figures (slope_stability.plotting).

The `analysis` parameter for this module is a dict:
    Required:
        "geom" : SlopeGeometry
    Optional:
        "FOS_required" : float (default 1.5)
        "search"       : SearchResult — adds a search-summary section
                         and the trial-surface map figure.
        "mc"           : MonteCarloResult — adds the probabilistic
                         section and FOS-histogram figure.
        "fosm"         : FOSMResult — adds FOSM reliability steps and
                         the variance-contribution figure.
        "variables"    : probabilistic variable spec dict (echoed in
                         the probabilistic inputs table; supports an
                         extra per-variable "source" note).
        "f_interslice" : str — interslice function name used for
                         M-P/GLE (display only; default "half_sine").

CS-3: every value shown here is a stored result field — nothing is
re-derived.

References:
    Duncan, Wright & Brandon (2014) — Soil Strength and Slope Stability
    Fellenius (1927); Bishop (1955); Janbu (1973); Spencer (1967);
    Morgenstern & Price (1965); Duncan (2000) reliability.
"""

import math
from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Slope Stability Analysis"

REFERENCES = [
    'Duncan, J.M., Wright, S.G. & Brandon, T.L. (2014). Soil Strength '
    'and Slope Stability, 2nd Ed. Wiley.',
    'Fellenius, W. (1927). "Erdstatische Berechnungen." W. Ernst & Sohn.',
    'Bishop, A.W. (1955). "The Use of the Slip Circle in the Stability '
    'Analysis of Slopes." Geotechnique, 5(1), 7-17.',
    'Janbu, N. (1973). "Slope Stability Computations." Embankment-Dam '
    'Engineering, Casagrande Volume, Wiley, 47-86.',
    'Spencer, E. (1967). "A Method of Analysis of the Stability of '
    'Embankments Assuming Parallel Interslice Forces." Geotechnique, '
    '17(1), 11-26.',
    'Morgenstern, N.R. & Price, V.E. (1965). "The Analysis of the '
    'Stability of General Slip Surfaces." Geotechnique, 15(1), 79-93.',
    'Duncan, J.M. (2000). "Factors of Safety and Reliability in '
    'Geotechnical Engineering." J. Geotech. Geoenviron. Eng., 126(4), '
    '307-316.',
    'Abramson, L.W. et al. (2002). Slope Stability and Stabilization '
    'Methods, 2nd Ed. Wiley.',
]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for slope stability calc package.

    Parameters
    ----------
    result : SlopeStabilityResult
    analysis : dict — see module docstring.

    Returns
    -------
    list of InputItem
    """
    geom = analysis["geom"]
    FOS_required = analysis.get("FOS_required", 1.5)

    items = [InputItem("Method", "Analysis method", result.method, "")]

    if result.method in ("Morgenstern-Price", "GLE"):
        items.append(InputItem(
            "f(x)", "Interslice force function",
            analysis.get("f_interslice", "half_sine"), ""))

    if result.is_circular:
        items.extend([
            InputItem("x_c", "Circle center x", f"{result.xc:.2f}", "m"),
            InputItem("y_c", "Circle center y (elev.)",
                      f"{result.yc:.2f}", "m"),
            InputItem("R", "Circle radius", f"{result.radius:.2f}", "m"),
        ])
    else:
        items.append(InputItem(
            "Surface", "Slip surface type",
            f"Noncircular ({len(result.slip_points or [])} vertices)", ""))

    items.extend([
        InputItem("N", "Number of slices", result.n_slices, ""),
        InputItem("FOS_req", "Required FOS", FOS_required, ""),
    ])

    if result.has_seismic:
        items.append(InputItem("k_h", "Seismic coefficient", result.kh, ""))

    if geom.surcharge > 0:
        items.append(InputItem("q_s", "Surcharge", geom.surcharge, "kPa"))

    if result.tension_crack_depth > 0:
        items.append(InputItem("z_t", "Tension crack depth",
                               f"{result.tension_crack_depth:.2f}", "m"))
        if result.tension_crack_water_depth > 0:
            items.append(InputItem(
                "z_w", "Water depth in tension crack",
                f"{result.tension_crack_water_depth:.2f}", "m"))

    n_nails = len(geom.nails or [])
    n_anchors = len(geom.anchors or [])
    n_geo = len(geom.geosynthetics or [])
    if n_nails:
        items.append(InputItem("n_nail", "Soil nails", n_nails, ""))
    if n_anchors:
        items.append(InputItem("n_anch", "Anchors / tiebacks",
                               n_anchors, ""))
    if n_geo:
        items.append(InputItem("n_geo", "Geosynthetic layers", n_geo, ""))

    return items


# ---------------------------------------------------------------------------
# Calculation sections
# ---------------------------------------------------------------------------

def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections.

    Parameters
    ----------
    result : SlopeStabilityResult
    analysis : dict

    Returns
    -------
    list of CalcSection
    """
    geom = analysis["geom"]
    sections = []

    sections.append(_geometry_section(geom))
    sections.append(_slip_surface_section(result))
    sections.append(_method_section(result, analysis))

    reinf = _reinforcement_section(result, geom)
    if reinf is not None:
        sections.append(reinf)

    comparison = _comparison_section(result, analysis)
    if comparison is not None:
        sections.append(comparison)

    if result.slice_data:
        sections.append(_slice_table_section(result))

    search = analysis.get("search")
    if search is not None:
        sections.append(_search_section(search))

    prob = _probabilistic_section(analysis)
    if prob is not None:
        sections.append(prob)

    # Stability check
    FOS_required = analysis.get("FOS_required", 1.5)
    sections.append(CalcSection(title="Stability Check", items=[
        CheckItem(
            description="Slope stability adequacy",
            demand=FOS_required,
            demand_label="FOS_required",
            capacity=result.FOS,
            capacity_label="FOS_computed",
            unit="",
            passes=result.FOS >= FOS_required,
        ),
    ]))

    return sections


def _geometry_section(geom) -> CalcSection:
    items = []

    surface_rows = [[f"{x:.1f}", f"{z:.1f}"] for x, z in geom.surface_points]
    items.append(TableData(
        title="Ground Surface Profile",
        headers=["x (m)", "Elevation (m)"],
        rows=surface_rows,
    ))

    layer_rows = []
    any_special = any(L.strength_model != "mohr_coulomb"
                      for L in geom.soil_layers)
    for layer in geom.soil_layers:
        c, phi = layer.shear_strength_params
        row = [
            layer.name,
            f"{layer.top_elevation:.1f}",
            f"{layer.bottom_elevation:.1f}",
            f"{layer.gamma:.1f}",
            f"{c:.1f}",
            f"{phi:.1f}",
            layer.analysis_mode,
        ]
        if any_special:
            if layer.strength_model == "shansep":
                row.append(f"SHANSEP (S={layer.shansep_S:.2f}, "
                           f"m={layer.shansep_m:.2f}, OCR={layer.ocr:.1f})")
            elif layer.strength_model == "hoek_brown":
                row.append(f"Hoek-Brown (GSI={layer.hb_gsi:.0f}, "
                           f"mi={layer.hb_mi:.0f}, "
                           f"sig_ci={layer.hb_sigci:.0f} kPa)")
            else:
                row.append("Mohr-Coulomb")
        layer_rows.append(row)
    headers = ["Layer", "Top (m)", "Bottom (m)", "γ (kN/m³)",
               "c/c_u (kPa)", "φ (deg)", "Mode"]
    if any_special:
        headers.append("Strength model")
    items.append(TableData(
        title="Soil Layer Properties",
        headers=headers,
        rows=layer_rows,
    ))

    if geom.gwt_points is not None:
        gwt_rows = [[f"{x:.1f}", f"{z:.1f}"] for x, z in geom.gwt_points]
        items.append(TableData(
            title="Groundwater Table",
            headers=["x (m)", "GWT Elev. (m)"],
            rows=gwt_rows,
            notes="GWT above the ground surface is treated as ponded "
                  "water (external hydrostatic load).",
        ))

    return CalcSection(title="Slope Geometry & Soil Properties",
                       items=items)


def _slip_surface_section(result) -> CalcSection:
    items = []
    if result.is_circular:
        items.append(CalcStep(
            title="Slip Circle Definition",
            equation="Circle: (x - x_c)² + (z - y_c)² = R²",
            substitution=(
                f"(x - {result.xc:.2f})² + (z - {result.yc:.2f})² "
                f"= {result.radius:.2f}²"
            ),
            result_name="R",
            result_value=f"{result.radius:.2f}",
            result_unit="m",
        ))
    else:
        pts = result.slip_points or []
        items.append(TableData(
            title="Noncircular Slip Surface Vertices",
            headers=["x (m)", "z (m)"],
            rows=[[f"{x:.2f}", f"{z:.2f}"] for x, z in pts],
        ))

    items.append(CalcStep(
        title="Slip Surface Intersection",
        equation="Entry and exit points where the slip surface "
                 "intersects the ground surface",
        substitution="",
        result_name="x_entry, x_exit",
        result_value=f"{result.x_entry:.2f} m, {result.x_exit:.2f} m",
    ))

    return CalcSection(title="Slip Surface", items=items)


def _method_section(result, analysis) -> CalcSection:
    items = []
    m = result.method

    if m == "Fellenius":
        items.append(CalcStep(
            title="Ordinary Method of Slices (Fellenius)",
            equation=(
                "FOS = Σ[c_i × b_i / cos(α_i) + "
                "max(W_i × cos(α_i) - u_i × b_i / cos(α_i), 0) "
                "× tan(φ_i)] / Σ[W_i × sin(α_i)]"
            ),
            substitution="",
            result_name="FOS",
            result_value=f"{result.FOS:.3f}",
            reference="Fellenius (1927)",
            notes="Satisfies moment equilibrium only. Effective normal "
                  "clamped at zero (no negative friction under high "
                  "pore pressure).",
        ))

    elif m == "Bishop":
        items.append(CalcStep(
            title="Bishop's Simplified Method",
            equation=(
                "FOS = Σ[(c_i·b_i + max(W_i - u_i·b_i, 0)·tan(φ_i)) "
                "/ m_α_i] / Σ[W_i × sin(α_i)]"
            ),
            substitution="m_α = cos(α) + sin(α)·tan(φ)/FOS",
            result_name="FOS",
            result_value=f"{result.FOS:.3f}",
            reference="Bishop (1955)",
            notes="Iterative — satisfies moment equilibrium; assumes "
                  "zero interslice shear. Frictional term clamped at "
                  "zero where u·b > W (SS-2).",
        ))

    elif m == "Janbu":
        items.append(CalcStep(
            title="Janbu's Simplified Method (uncorrected)",
            equation=(
                "FOS₀ = Σ[(c_i·b_i + (W_i - u_i·b_i)·tan(φ_i)) "
                "/ (m_α·cos(α_i))] / Σ[W_i × tan(α_i)]"
            ),
            substitution="",
            result_name="FOS₀",
            result_value=(f"{result.FOS_janbu_uncorrected:.3f}"
                          if result.FOS_janbu_uncorrected is not None
                          else "-"),
            reference="Janbu (1973)",
            notes="Horizontal force equilibrium; zero interslice shear.",
        ))
        if result.janbu_f0 is not None:
            items.append(CalcStep(
                title="Janbu Correction Factor",
                equation="f₀ = 1 + b₁·[d/L - 1.4·(d/L)²] "
                         "(depth-to-length ratio of the sliding mass)",
                substitution="FOS = f₀ × FOS₀ = "
                             f"{result.janbu_f0:.3f} × "
                             f"{result.FOS_janbu_uncorrected:.3f}",
                result_name="FOS",
                result_value=f"{result.FOS:.3f}",
                reference="Janbu (1973)",
                notes="Correction compensates for the neglected "
                      "interslice shear forces.",
            ))

    elif m == "Spencer":
        items.append(CalcStep(
            title="Spencer's Method (rigorous GLE, constant f(x))",
            equation=(
                "FOS from simultaneous force AND moment equilibrium; "
                "interslice resultant at constant inclination θ: "
                "X = E·tan(θ)"
            ),
            substitution="",
            result_name="FOS",
            result_value=f"{result.FOS:.3f}",
            reference="Spencer (1967)",
            notes="Solved with the rigorous GLE engine (interslice "
                  "force distribution and line of thrust reported "
                  "below).",
        ))
        if result.theta_spencer is not None:
            items.append(CalcStep(
                title="Interslice Force Angle",
                equation="θ = angle of the interslice resultant "
                         "from horizontal (tan θ = λ)",
                substitution="",
                result_name="θ",
                result_value=f"{result.theta_spencer:.2f}",
                result_unit="deg",
            ))

    elif m in ("Morgenstern-Price", "GLE"):
        f_name = analysis.get("f_interslice", "half_sine")
        items.append(CalcStep(
            title=f"{m} Method (rigorous GLE)",
            equation=(
                "X = λ·f(x)·E with interslice function "
                f"f(x) = {f_name}; FOS at the F_moment = F_force "
                "crossing"
            ),
            substitution="",
            result_name="FOS",
            result_value=f"{result.FOS:.3f}",
            reference="Morgenstern & Price (1965)",
            notes="Complete equilibrium: both force and moment "
                  "equations satisfied at the reported λ.",
        ))
        if result.lambda_mp is not None:
            items.append(CalcStep(
                title="Interslice Force Scaling",
                equation="λ = interslice force scaling factor "
                         "(X = λ·f(x)·E)",
                substitution="",
                result_name="λ",
                result_value=f"{result.lambda_mp:.3f}",
            ))

    # Interslice / thrust-line summary (rigorous engine output)
    if result.slice_data and result.slice_data[0].E_left_kN is not None:
        E_all = [s.E_left_kN for s in result.slice_data] + \
                [result.slice_data[-1].E_right_kN]
        X_all = [s.X_left_kN for s in result.slice_data] + \
                [result.slice_data[-1].X_right_kN]
        items.append(CalcStep(
            title="Interslice Force Extremes",
            equation="Max interslice normal E and shear X over all "
                     "slice boundaries (distribution plotted in the "
                     "Figures section)",
            substitution="",
            result_name="E_max, X_max",
            result_value=(f"{max(E_all):.1f} kN/m, "
                          f"{max(X_all):.1f} kN/m"),
        ))
    if result.thrust_line:
        z_thrust = [z for _, z in result.thrust_line]
        items.append(CalcStep(
            title="Line of Thrust",
            equation="Elevation of the point of application of E at "
                     "each slice boundary (acceptability check: "
                     "thrust line within the sliding mass, typically "
                     "near the lower third)",
            substitution="",
            result_name="z_thrust range",
            result_value=(f"{min(z_thrust):.2f} to "
                          f"{max(z_thrust):.2f} m"),
        ))

    if result.has_seismic:
        items.append(CalcStep(
            title="Pseudo-static Seismic Load",
            equation="Horizontal force k_h × W applied at each "
                     "slice centroid",
            substitution=f"k_h = {result.kh:.3f}",
            result_name="k_h",
            result_value=f"{result.kh:.3f}",
        ))

    return CalcSection(title="Analysis Method & Factor of Safety",
                       items=items)


def _reinforcement_section(result, geom):
    """Reinforcement layout + mobilized forces (None if unreinforced)."""
    has_layout = any([geom.nails, geom.anchors, geom.geosynthetics])
    if not has_layout and not result.reinforcements:
        return None

    items = []

    layout_rows = []
    for i, n in enumerate(geom.nails or []):
        layout_rows.append([
            "Nail", str(i + 1),
            f"({n.x_head:.1f}, {n.z_head:.1f})",
            f"{n.length:.1f}", f"{n.inclination:.0f}",
            f"bar {n.bar_diameter:.0f} mm, bond "
            f"{n.bond_stress:.0f} kPa @ {n.spacing_h:.1f} m c/c",
        ])
    for i, a in enumerate(geom.anchors or []):
        layout_rows.append([
            "Anchor", str(i + 1),
            f"({a.x_head:.1f}, {a.z_head:.1f})",
            f"{a.length:.1f}", f"{a.inclination:.0f}",
            f"T_allow = {a.T_allow:.0f} kN/m",
        ])
    for i, g in enumerate(geom.geosynthetics or []):
        extent = ("full width" if g.x_start is None and g.x_end is None
                  else f"x = {g.x_start} to {g.x_end} m")
        layout_rows.append([
            "Geosynthetic", str(i + 1),
            f"elev. {g.elevation:.1f} m", "-", "0",
            f"T_allow = {g.T_allow:.0f} kN/m ({extent})",
        ])
    if layout_rows:
        items.append(TableData(
            title="Reinforcement Layout",
            headers=["Type", "#", "Head / level", "Length (m)",
                     "Incl. (deg)", "Capacity"],
            rows=layout_rows,
        ))

    if result.reinforcements:
        mob_rows = [[
            r["kind"].capitalize(), str(r["index"] + 1),
            f"({r['x_m']:.1f}, {r['z_m']:.1f})",
            f"{r['T_kN_per_m']:.1f}", r["controlled_by"],
        ] for r in result.reinforcements]
        total_T = sum(r["T_kN_per_m"] for r in result.reinforcements)
        items.append(TableData(
            title="Mobilized Reinforcement Forces at Slip-Surface "
                  "Crossings",
            headers=["Type", "#", "Crossing (x, z) (m)", "T (kN/m)",
                     "Controlled by"],
            rows=mob_rows,
            notes=f"Total stabilizing reinforcement force = "
                  f"{total_T:.1f} kN/m of slope run. Elements not "
                  f"listed do not cross this slip surface.",
        ))
    elif has_layout:
        items.append("No reinforcement element crosses this slip "
                     "surface — the layout contributes no stabilizing "
                     "force to the reported FOS.")

    return CalcSection(title="Reinforcement", items=items)


def _comparison_section(result, analysis):
    rows = []
    if result.FOS_fellenius is not None:
        rows.append(["Fellenius (OMS)", f"{result.FOS_fellenius:.3f}",
                     "moment only"])
    if result.FOS_bishop is not None:
        rows.append(["Bishop simplified", f"{result.FOS_bishop:.3f}",
                     "moment only"])
    if result.FOS_janbu_uncorrected is not None:
        rows.append(["Janbu simplified (uncorrected)",
                     f"{result.FOS_janbu_uncorrected:.3f}",
                     "force only"])
    if result.FOS_janbu is not None:
        f0 = (f"f₀ = {result.janbu_f0:.3f}"
              if result.janbu_f0 else "")
        rows.append(["Janbu simplified (corrected)",
                     f"{result.FOS_janbu:.3f}", f0])
    if result.FOS_spencer is not None:
        det = (f"θ = {result.theta_spencer:.2f} deg"
               if result.theta_spencer is not None else "")
        rows.append(["Spencer", f"{result.FOS_spencer:.3f}", det])
    if result.FOS_morgenstern_price is not None:
        det = (f"λ = {result.lambda_mp:.3f}"
               if result.lambda_mp is not None else "")
        f_name = analysis.get("f_interslice", "half_sine")
        rows.append([f"Morgenstern-Price ({f_name})",
                     f"{result.FOS_morgenstern_price:.3f}", det])

    if len(rows) < 2:
        return None

    return CalcSection(
        title="Method Comparison",
        items=[TableData(
            title="FOS by Method (Fredlund & Krahn style — one "
                  "surface, all methods)",
            headers=["Method", "FOS", "Detail"],
            rows=rows,
            notes="All methods use the same slip surface and slice "
                  "geometry. Methods satisfying complete equilibrium "
                  "(Spencer, Morgenstern-Price) are preferred for "
                  "design.",
        )],
    )


def _slice_table_section(result) -> CalcSection:
    sd = result.slice_data
    n = len(sd)
    has_interslice = sd[0].E_left_kN is not None

    if n > 40:
        step = max(1, n // 30)
        indices = list(range(0, n, step))
        if (n - 1) not in indices:
            indices.append(n - 1)
        note = f"Showing {len(indices)} of {n} slices."
    else:
        indices = list(range(n))
        note = f"All {n} slices."

    headers = ["#", "x_mid (m)", "b (m)", "W (kN/m)", "α (deg)",
               "c (kPa)", "φ (deg)", "u (kPa)", "N' (kN/m)",
               "S_mob (kN/m)", "u·l (kN/m)"]
    if has_interslice:
        headers += ["E_L (kN/m)", "E_R (kN/m)", "X_L (kN/m)",
                    "X_R (kN/m)"]

    rows = []
    for i in indices:
        s = sd[i]
        row = [
            str(i + 1), f"{s.x_mid:.2f}", f"{s.width:.2f}",
            f"{s.weight:.1f}", f"{s.alpha_deg:.1f}", f"{s.c:.1f}",
            f"{s.phi:.1f}", f"{s.pore_pressure:.1f}",
            f"{s.N_eff_kN:.1f}", f"{s.S_mob_kN:.1f}",
            f"{s.U_base_kN:.1f}",
        ]
        if has_interslice:
            row += [f"{s.E_left_kN:.1f}", f"{s.E_right_kN:.1f}",
                    f"{s.X_left_kN:.1f}", f"{s.X_right_kN:.1f}"]
        if s.in_tension_crack:
            row[0] += " (tc)"
        rows.append(row)

    if has_interslice:
        note += (" N', S_mob, E and X from the converged rigorous GLE "
                 "state. (tc) = slice in tension crack zone.")
    else:
        note += (" N' and S_mob from the Fellenius decomposition "
                 "(method does not resolve interslice forces). "
                 "(tc) = slice in tension crack zone.")

    return CalcSection(
        title="Slice Force Table",
        items=[TableData(
            title="Per-Slice Forces",
            headers=headers,
            rows=rows,
            notes=note,
        )],
    )


def _search_section(search) -> CalcSection:
    items = []
    crit = search.critical
    rows = [
        ["Surfaces evaluated", str(search.n_surfaces_evaluated)],
    ]
    if crit is not None:
        rows.append(["Method", crit.method])
        rows.append(["Minimum FOS", f"{crit.FOS:.3f}"])
        if crit.is_circular:
            rows.append(["Critical circle center",
                         f"({crit.xc:.2f}, {crit.yc:.2f}) m"])
            rows.append(["Critical circle radius",
                         f"{crit.radius:.2f} m"])
        else:
            rows.append(["Critical surface",
                         f"Noncircular, "
                         f"{len(crit.slip_points or [])} vertices"])
        rows.append(["Entry / exit",
                     f"x = {crit.x_entry:.2f} m / "
                     f"x = {crit.x_exit:.2f} m"])
    items.append(TableData(
        title="Critical Surface Search Summary",
        headers=["Item", "Value"],
        rows=rows,
        notes="The trial-surface map in the Figures section shows "
              "every evaluated surface colored by FOS.",
    ))

    if crit is not None and not crit.is_circular and crit.slip_points:
        items.append(TableData(
            title="Critical Surface Vertices",
            headers=["x (m)", "z (m)"],
            rows=[[f"{x:.2f}", f"{z:.2f}"] for x, z in crit.slip_points],
        ))

    return CalcSection(title="Critical Surface Search", items=items)


def _probabilistic_section(analysis):
    mc = analysis.get("mc")
    fosm = analysis.get("fosm")
    variables = analysis.get("variables")
    if mc is None and fosm is None:
        return None

    items = []

    if variables:
        var_rows = []
        for key, spec in variables.items():
            mean = spec.get("mean", "(layer value)")
            if "cov" in spec:
                disp = f"COV = {spec['cov']:.2f}"
            elif "std" in spec:
                disp = f"std = {spec['std']:.2f}"
            else:
                disp = "-"
            var_rows.append([
                key,
                f"{mean}",
                disp,
                spec.get("dist", "normal"),
                spec.get("source", "-"),
            ])
        items.append(TableData(
            title="Random Variables",
            headers=["Parameter", "Mean", "Variability", "Distribution",
                     "Source"],
            rows=var_rows,
            notes="Parameter:layer keys are scoped to a single layer; "
                  "unscoped keys apply to all layers carrying the "
                  "parameter.",
        ))

    if fosm is not None:
        items.append(CalcStep(
            title="FOSM (Taylor Series) Standard Deviation of FOS",
            equation="σ_F² = Σ (ΔF_i / 2)² — "
                     "central differences at ±σ per variable",
            substitution="",
            result_name="σ_F",
            result_value=f"{fosm.sigma_f:.4f}",
            reference="Duncan (2000)",
        ))
        items.append(CalcStep(
            title="Reliability Index (normal)",
            equation="β = (F_MLV − 1) / σ_F",
            substitution=f"β = ({fosm.fos_mlv:.3f} − 1) / "
                         f"{fosm.sigma_f:.4f}",
            result_name="β",
            result_value=f"{fosm.beta_normal:.2f}",
            notes=f"pf = Φ(−β) = {fosm.pf_normal:.2%}",
        ))
        items.append(CalcStep(
            title="Reliability Index (lognormal)",
            equation="β_LN = ln(F_MLV / √(1 + COV_F²)) / "
                     "√(ln(1 + COV_F²))",
            substitution=f"F_MLV = {fosm.fos_mlv:.3f}, "
                         f"COV_F = {fosm.cov_f:.3f}",
            result_name="β_LN",
            result_value=f"{fosm.beta_lognormal:.2f}",
            reference="Duncan (2000), Eq. 8",
            notes=f"pf = Φ(−β_LN) = {fosm.pf_lognormal:.2%}",
        ))
        pct_rows = [[k, f"{fosm.variable_deltas.get(k, 0.0):+.4f}",
                     f"{v:.1f}%"]
                    for k, v in sorted(fosm.variable_variance_pct.items(),
                                       key=lambda kv: -kv[1])]
        items.append(TableData(
            title="FOSM Variance Contributions",
            headers=["Variable", "ΔF (±σ)", "% of Var(FOS)"],
            rows=pct_rows,
        ))

    if mc is not None:
        mc_rows = [
            ["Realizations", str(mc.n)
             + (" (surface re-searched)" if mc.research_surface
                else " (fixed critical surface)")],
            ["FOS mean / median",
             f"{mc.fos_mean:.3f} / {mc.fos_median:.3f}"],
            ["FOS std (COV)", f"{mc.fos_std:.3f} ({mc.fos_cov:.3f})"],
            ["FOS range", f"{mc.fos_min:.3f} – {mc.fos_max:.3f}"],
            ["P(FOS < 1) — count", f"{mc.pf:.2%}  "
             f"({mc.n_failed}/{mc.n})"],
            ["β_LN (lognormal fit)", f"{mc.beta_lognormal:.2f}"],
            ["pf (lognormal fit)", f"{mc.pf_lognormal:.2%}"],
        ]
        if mc.seed is not None:
            mc_rows.append(["Random seed", str(mc.seed)])
        items.append(TableData(
            title="Monte Carlo FOS Distribution",
            headers=["Item", "Value"],
            rows=mc_rows,
            notes="Histogram with the fitted lognormal distribution is "
                  "shown in the Figures section.",
        ))

    return CalcSection(title="Probabilistic Analysis (Reliability)",
                       items=items)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def get_figures(result, analysis) -> List[FigureData]:
    """Generate SLOPE/W-style figures for the slope calc package.

    Figures degrade gracefully: each is skipped when matplotlib is
    unavailable or its data is absent.

    Parameters
    ----------
    result : SlopeStabilityResult
    analysis : dict — see module docstring.

    Returns
    -------
    list of FigureData
    """
    geom = analysis["geom"]
    figures = []
    fig_no = [0]

    def _add(builder, title, caption, width=90):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        try:
            fig = builder()
        except ValueError:
            return
        b64 = figure_to_base64(fig, dpi=150)
        plt.close(fig)
        fig_no[0] += 1
        figures.append(FigureData(
            title=title,
            image_base64=b64,
            caption=f"Figure {fig_no[0]}: {caption}",
            width_percent=width,
        ))

    from slope_stability import plotting as _plots

    fos_req = analysis.get("FOS_required", 1.5)

    _add(lambda: _plots.plot_cross_section(result, geom,
                                           fos_required=fos_req),
         "Slope Cross-Section",
         f"Slope cross-section with analyzed slip surface. "
         f"FOS = {result.FOS:.3f} ({result.method} method). "
         f"Entry x = {result.x_entry:.1f} m, "
         f"exit x = {result.x_exit:.1f} m.")

    search = analysis.get("search")
    if search is not None:
        _add(lambda: _plots.plot_trial_surface_map(search, geom),
             "Trial Surface Map",
             f"Critical surface search: "
             f"{search.n_surfaces_evaluated} trial surfaces colored "
             f"by FOS; critical surface highlighted.")

    if result.slice_data:
        _add(lambda: _plots.plot_slice_forces(result),
             "Slice Force Distribution",
             f"Per-slice weight, effective base normal, mobilized "
             f"shear and base water force "
             f"({result.n_slices} slices).", 85)

        if result.slice_data[0].E_left_kN is not None:
            _add(lambda: _plots.plot_interslice_forces(result, geom),
                 "Interslice Forces & Line of Thrust",
                 "Interslice normal (E) and shear (X) distributions "
                 "with the line of thrust from the rigorous GLE "
                 "solution.", 85)

    mc = analysis.get("mc")
    if mc is not None:
        _add(lambda: _plots.plot_mc_histogram(mc),
             "Monte Carlo FOS Distribution",
             f"Monte Carlo FOS histogram (n = {mc.n}) with "
             f"moment-matched lognormal fit. "
             f"beta_LN = {mc.beta_lognormal:.2f}, "
             f"pf = {mc.pf_lognormal:.2%}.", 75)

    fosm = analysis.get("fosm")
    if fosm is not None:
        _add(lambda: _plots.plot_fosm_tornado(fosm),
             "FOSM Variance Contributions",
             f"FOSM (Taylor series) variance contributions. "
             f"beta_LN = {fosm.beta_lognormal:.2f}, "
             f"pf = {fosm.pf_lognormal:.2%}.", 75)

    return figures
