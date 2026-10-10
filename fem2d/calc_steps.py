"""
Calculation package steps for 2D finite element analysis (fem2d).

Renders FEMResult / SeepageResult / ConsolidationResult /
StagedConstructionResult into PLAXIS-report-style calc package
sections with figures from fem2d.plotting.

The `analysis` parameter for this module is an optional dict:
    "soil_layers"       : list of dict — the material list passed to
                          the analyze_* function (rendered as the
                          materials table: name, gamma, E, nu, c,
                          phi, psi, model).
    "model_description" : str — free-text model narrative.
    "loads_description" : str — free-text load narrative.
    "material_ids"      : (n_elem,) array — element material ids for
                          mesh coloring (e.g. from
                          assign_layers_by_elevation).
    "material_names"    : list of str — legend names per id.
    "bc_nodes"          : dict from detect_boundary_nodes() — BC
                          symbols on the mesh figure.
    "FOS_required"      : float — SRM stability check (default 1.5).
    "gwt_description"   : str — groundwater statement.

CS-3: sections RENDER stored result fields only.

References:
    Griffiths & Lane (1999); Smith & Griffiths (2004);
    Clausen et al. (2006); Schanz et al. (1999); Biot (1941).
"""

from typing import List

import numpy as np

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "2D Finite Element Analysis"

REFERENCES = [
    'Griffiths, D.V. & Lane, P.A. (1999). "Slope Stability Analysis by '
    'Finite Elements." Geotechnique, 49(3), 387-403.',
    'Smith, I.M. & Griffiths, D.V. (2004). Programming the Finite '
    'Element Method, 4th Ed. Wiley.',
    'Clausen, J., Damkilde, L. & Andersen, L. (2006). "Efficient Return '
    'Algorithms for Associated Plasticity with Multiple Yield Planes." '
    'Int. J. Numer. Meth. Engng, 66(6), 1036-1059.',
    'Schanz, T., Vermeer, P.A. & Bonnier, P.G. (1999). "The Hardening '
    'Soil Model: Formulation and Verification." Beyond 2000 in '
    'Computational Geotechnics, Balkema.',
    'Biot, M.A. (1941). "General Theory of Three-Dimensional '
    'Consolidation." J. Appl. Phys., 12(2), 155-164.',
    'de Souza Neto, E.A., Peric, D. & Owen, D.R.J. (2008). '
    'Computational Methods for Plasticity. Wiley.',
]

_ANALYSIS_LABEL = {
    'elastic': 'Linear elastic',
    'elastoplastic': 'Elastoplastic (Mohr-Coulomb)',
    'srm': 'Strength Reduction Method (SRM) slope stability',
    'excavation': 'Braced excavation (beam-coupled)',
}


def _element_type_label(result):
    elements = getattr(result, 'elements', None)
    if elements is None:
        return "-"
    n = np.asarray(elements).shape[1]
    return {3: "CST (3-node triangle)",
            4: "Q4 (4-node quadrilateral)",
            6: "T6 (6-node quadratic triangle)"}.get(n, f"{n}-node")


def _is_seepage(result):
    return type(result).__name__ == "SeepageResult"


def _is_consolidation(result):
    return type(result).__name__ == "ConsolidationResult"


def _is_staged(result):
    return type(result).__name__ == "StagedConstructionResult"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def get_input_summary(result, analysis) -> List[InputItem]:
    """Build the input parameter table for the fem2d calc package.

    Parameters
    ----------
    result : FEMResult | SeepageResult | ConsolidationResult |
        StagedConstructionResult
    analysis : dict or None — see module docstring.

    Returns
    -------
    list of InputItem
    """
    analysis = analysis or {}
    items = []

    if _is_seepage(result):
        items.append(InputItem("Type", "Analysis type",
                               "Steady-state seepage (Laplace)", ""))
    elif _is_consolidation(result):
        items.append(InputItem("Type", "Analysis type",
                               "Coupled Biot consolidation", ""))
    elif _is_staged(result):
        items.append(InputItem("Type", "Analysis type",
                               "Staged construction "
                               f"({result.n_phases} phases)", ""))
    else:
        items.append(InputItem(
            "Type", "Analysis type",
            _ANALYSIS_LABEL.get(result.analysis_type,
                                result.analysis_type), ""))

    items.append(InputItem("n_nodes", "Mesh nodes", result.n_nodes, ""))
    items.append(InputItem("n_elem", "Mesh elements",
                           result.n_elements, ""))
    if getattr(result, 'elements', None) is not None:
        items.append(InputItem("Element", "Element type",
                               _element_type_label(result), ""))

    if getattr(result, 'FOS', None) is not None:
        items.append(InputItem("FOS_req", "Required FOS",
                               analysis.get("FOS_required", 1.5), ""))

    if _is_consolidation(result):
        items.append(InputItem("n_t", "Time steps",
                               result.n_time_steps, ""))

    if analysis.get("gwt_description"):
        items.append(InputItem("GWT", "Groundwater",
                               analysis["gwt_description"], ""))

    return items


# ---------------------------------------------------------------------------
# Calculation sections
# ---------------------------------------------------------------------------

def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build calculation sections for the fem2d calc package.

    Parameters
    ----------
    result : FEMResult | SeepageResult | ConsolidationResult |
        StagedConstructionResult
    analysis : dict or None

    Returns
    -------
    list of CalcSection
    """
    analysis = analysis or {}
    sections = [_model_section(result, analysis)]

    if _is_seepage(result):
        sections.append(_seepage_section(result))
    elif _is_consolidation(result):
        sections.append(_consolidation_section(result))
    elif _is_staged(result):
        sections.extend(_staged_sections(result))
    else:
        if result.analysis_type == "srm":
            sections.append(_srm_section(result, analysis))
        sections.append(_results_section(result))
        if result.beam_forces:
            sections.append(_beam_section(result))

    sections.append(_checks_section(result, analysis))
    return sections


def _model_section(result, analysis) -> CalcSection:
    items = []

    desc = analysis.get("model_description")
    if desc:
        items.append(desc)

    mesh_rows = [
        ["Nodes", str(result.n_nodes)],
        ["Elements", str(result.n_elements)],
    ]
    if getattr(result, 'elements', None) is not None:
        mesh_rows.append(["Element type", _element_type_label(result)])
        nodes = np.asarray(result.nodes, dtype=float)
        mesh_rows.append([
            "Domain extents",
            f"x: {nodes[:, 0].min():.1f} to {nodes[:, 0].max():.1f} m, "
            f"y: {nodes[:, 1].min():.1f} to {nodes[:, 1].max():.1f} m"])
    items.append(TableData(
        title="Mesh Summary",
        headers=["Item", "Value"],
        rows=mesh_rows,
    ))

    soil_layers = analysis.get("soil_layers")
    if soil_layers:
        mat_rows = []
        any_hs = any(sl.get('model') == 'hs' for sl in soil_layers)
        for i, sl in enumerate(soil_layers):
            row = [
                sl.get('name', f"Material {i + 1}"),
                f"{sl.get('gamma', 18):.1f}",
                f"{sl.get('E', 30000):.0f}",
                f"{sl.get('nu', 0.3):.2f}",
                f"{sl.get('c', 0):.1f}",
                f"{sl.get('phi', 0):.1f}",
                f"{sl.get('psi', 0):.1f}",
            ]
            if any_hs:
                if sl.get('model') == 'hs':
                    row.append(
                        f"HS (E50_ref={sl.get('E50_ref', 0):.0f}, "
                        f"Eur_ref={sl.get('Eur_ref', 0):.0f}, "
                        f"m={sl.get('m', 0.5):.2f})")
                else:
                    row.append("Mohr-Coulomb")
            mat_rows.append(row)
        headers = ["Material", "γ (kN/m³)", "E (kPa)", "ν",
                   "c (kPa)", "φ (deg)", "ψ (deg)"]
        if any_hs:
            headers.append("Constitutive model")
        items.append(TableData(
            title="Material Properties",
            headers=headers,
            rows=mat_rows,
            notes="Plane-strain idealization; elastic-perfectly-"
                  "plastic Mohr-Coulomb unless noted.",
        ))

    items.append(CalcStep(
        title="Boundary Conditions",
        equation="Base: fixed (u = v = 0). Lateral boundaries: "
                 "rollers (u = 0, v free)",
        substitution="",
        result_name="BC scheme",
        result_value="standard geotechnical box",
        notes=analysis.get("loads_description", ""),
    ))

    return CalcSection(title="Model Summary", items=items)


def _srm_section(result, analysis) -> CalcSection:
    items = []
    items.append(CalcStep(
        title="Strength Reduction Scheme",
        equation="c_red = c / SRF;  φ_red = arctan(tan(φ) / SRF);  "
                 "ψ_red = min(ψ, φ_red)",
        substitution="SRF increased until no stress distribution "
                     "satisfies both the Mohr-Coulomb criterion and "
                     "global equilibrium",
        result_name="FOS",
        result_value=(f"{result.FOS:.3f}" if result.FOS is not None
                      else "-"),
        reference="Griffiths & Lane (1999)",
        notes=f"Failure basis: "
              f"{getattr(result, 'fos_basis', None) or 'n/a'}. "
              f"{result.n_srf_trials} SRF trials.",
    ))

    history = getattr(result, 'srf_history', None)
    if history:
        rows = []
        for h in history:
            rows.append([
                f"{h['srf']:.3f}",
                f"{h['max_disp_m']:.4g}",
                f"{h['dimensionless_disp']:.3g}",
                "yes" if h['converged'] else "NO",
                "failed" if h['failed'] else "stable",
                str(h['n_iter']),
            ])
        items.append(TableData(
            title="SRF Trial History",
            headers=["SRF", "δ_max (m)", "E·δ/(γH²)",
                     "Converged", "State", "Iterations"],
            rows=rows,
            notes="Dimensionless displacement per Griffiths & Lane "
                  "(1999); the knee of the SRF-displacement curve "
                  "marks failure.",
        ))

    pp = getattr(result, 'plastic_points', None)
    if pp:
        frac = pp['n_plastic'] / max(pp['n_gp_total'], 1)
        items.append(CalcStep(
            title="Plastic Zone Extent (last stable state)",
            equation="Gauss points on the reduced-strength "
                     "Mohr-Coulomb yield surface",
            substitution=f"at SRF = {pp['srf']:.3f}",
            result_name="n_plastic",
            result_value=f"{pp['n_plastic']} of {pp['n_gp_total']} "
                         f"({frac:.0%})",
            notes="Plastic-point map shown in the Figures section.",
        ))

    return CalcSection(
        title="Strength Reduction Method (Slope Stability)",
        items=items)


def _results_section(result) -> CalcSection:
    items = []
    disp_rows = [
        ["Max displacement |u|", f"{result.max_displacement_m:.4g} m"],
        ["Max horizontal u_x", f"{result.max_displacement_x_m:.4g} m"],
        ["Max vertical u_y", f"{result.max_displacement_y_m:.4g} m"],
    ]
    # locations from the stored arrays
    if result.nodes is not None and result.displacements is not None:
        nodes = np.asarray(result.nodes, dtype=float)
        u = np.asarray(result.displacements,
                       dtype=float)[:2 * len(nodes)]
        ux, uy = u[0::2], u[1::2]
        mag = np.sqrt(ux**2 + uy**2)
        i_m = int(mag.argmax())
        disp_rows[0][1] += (f"  at ({nodes[i_m, 0]:.1f}, "
                            f"{nodes[i_m, 1]:.1f})")
        i_x = int(np.abs(ux).argmax())
        disp_rows[1][1] += (f"  at ({nodes[i_x, 0]:.1f}, "
                            f"{nodes[i_x, 1]:.1f})")
        i_y = int(np.abs(uy).argmax())
        disp_rows[2][1] += (f"  at ({nodes[i_y, 0]:.1f}, "
                            f"{nodes[i_y, 1]:.1f})")
    items.append(TableData(
        title="Displacement Extremes",
        headers=["Item", "Value"],
        rows=disp_rows,
    ))

    items.append(TableData(
        title="Stress Extremes (element averages)",
        headers=["Component", "Value (kPa)"],
        rows=[
            ["max |σ_xx|", f"{result.max_sigma_xx_kPa:.1f}"],
            ["σ_yy range",
             f"{result.min_sigma_yy_kPa:.1f} to "
             f"{result.max_sigma_yy_kPa:.1f}"],
            ["max |τ_xy|", f"{result.max_tau_xy_kPa:.1f}"],
        ],
        notes="Tension-positive convention (compression negative).",
    ))

    if result.strut_forces:
        items.append(TableData(
            title="Strut Forces",
            headers=["Depth (m)", "Stiffness (kN/m/m)", "Force (kN/m)"],
            rows=[[f"{sf['depth_m']:.1f}",
                   f"{sf['stiffness_kN_per_m']:.0f}",
                   f"{sf['force_kN_per_m']:.1f}"]
                  for sf in result.strut_forces],
        ))

    if result.warnings:
        items.append(TableData(
            title="Solver Warnings",
            headers=["Warning"],
            rows=[[w] for w in result.warnings],
        ))

    return CalcSection(title="Results", items=items)


def _beam_section(result) -> CalcSection:
    rows = []
    for bf in result.beam_forces:
        rows.append([
            str(bf.element_index + 1),
            f"{bf.length:.2f}",
            f"{bf.axial_i:.1f} / {bf.axial_j:.1f}",
            f"{bf.shear_i:.1f} / {bf.shear_j:.1f}",
            f"{bf.moment_i:.1f} / {bf.moment_j:.1f}",
        ])
    return CalcSection(
        title="Structural (Beam) Results",
        items=[
            CalcStep(
                title="Wall Force Extremes",
                equation="Euler-Bernoulli beam elements coupled to "
                         "the soil mesh",
                substitution="",
                result_name="M_max, V_max",
                result_value=(
                    f"{result.max_beam_moment_kNm_per_m:.1f} kN·m/m, "
                    f"{result.max_beam_shear_kN_per_m:.1f} kN/m"),
            ),
            TableData(
                title="Beam Element Forces (node i / node j)",
                headers=["#", "L (m)", "Axial (kN/m)", "Shear (kN/m)",
                         "Moment (kN·m/m)"],
                rows=rows,
            ),
        ])


def _seepage_section(result) -> CalcSection:
    return CalcSection(title="Seepage Results", items=[
        CalcStep(
            title="Steady-State Flow Solution",
            equation="∇·(k ∇h) = 0 (Laplace) solved for total head "
                     "h on CST flow elements",
            substitution="",
            result_name="Q",
            result_value=f"{result.total_flow_m3_per_s_per_m:.3e}",
            result_unit="m³/s per m",
            reference="Smith & Griffiths (2004)",
        ),
        TableData(
            title="Seepage Result Summary",
            headers=["Item", "Value"],
            rows=[
                ["Total head range",
                 f"{result.min_head_m:.3f} to "
                 f"{result.max_head_m:.3f} m"],
                ["Max pore pressure",
                 f"{result.max_pore_pressure_kPa:.2f} kPa"],
                ["Max Darcy velocity",
                 f"{result.max_velocity_m_per_s:.3e} m/s"],
                ["Total flow",
                 f"{result.total_flow_m3_per_s_per_m:.3e} m³/s/m"],
            ],
        ),
    ])


def _mm(v):
    return f"{float(v) * 1000:.2f}"


def _movement_rows(label, m):
    """Table rows for one movement summary (fem2d.results.movement_summary)."""
    def _at(xy):
        return f" at ({xy[0]:.1f}, {xy[1]:.1f})" if xy else ""
    return [
        [f"{label}: settlement",
         f"{_mm(m['settlement_m'])} mm{_at(m.get('settlement_at_xy'))}"],
        [f"{label}: heave",
         f"{_mm(m['heave_m'])} mm{_at(m.get('heave_at_xy'))}"],
        [f"{label}: horizontal",
         f"{_mm(m['horizontal_m'])} mm{_at(m.get('horizontal_at_xy'))}"],
        [f"{label}: max |u|",
         f"{_mm(m['total_m'])} mm{_at(m.get('total_at_xy'))}"],
    ]


def _judgment_item(judgment):
    """The reference-stage judgment as a table (option, basis, result)."""
    return TableData(
        title="Judgment: " + judgment["question"],
        headers=["Option", "Assumes", "Result", "Used"],
        rows=[[o["name"], o["assumptions"], o["result"],
               "yes" if o["name"] == judgment["used"] else ""]
              for o in judgment["options"]],
        notes=judgment["why"],
    )


def _consolidation_section(result) -> CalcSection:
    ref = result.displacement_reference or {}
    rows = [
        ["Time steps", str(result.n_time_steps)],
        ["Max settlement", f"{result.max_settlement_m * 1000:.2f} mm"
         + (f" (from the {ref['measured_from']})" if ref else "")],
        ["Max excess pore pressure",
         f"{result.max_excess_pore_pressure_kPa:.2f} kPa"],
        ["Degree of consolidation (final)",
         f"{result.degree_of_consolidation:.3f}"],
        ["Converged", "yes" if result.converged else "NO"],
    ]
    if result.times is not None and len(result.times):
        t = np.asarray(result.times, dtype=float)
        rows.insert(1, ["Time range",
                        f"{t.min():.3g} to {t.max():.3g} s"])
    if result.scheme == "monolithic":
        eq = ("Monolithic u-p (Taylor-Hood T6/T3): load applied undrained, "
              "then equilibrium + flow continuity solved together in time")
    else:
        eq = ("Staggered u-p scheme: equilibrium with effective stress + "
              "transient flow continuity (drained at every step)")
    items = [
        CalcStep(
            title="Coupled Biot Consolidation",
            equation=eq,
            substitution="",
            result_name="U_final",
            result_value=f"{result.degree_of_consolidation:.3f}",
            reference="Biot (1941)",
        ),
        TableData(
            title="Consolidation Result Summary",
            headers=["Item", "Value"],
            rows=rows,
        ),
    ]
    if result.stages:
        items.append(TableData(
            title="Construction Stages",
            headers=["Stage", "What happens", "Settlement this stage (mm)",
                     "Cumulative settlement (mm)"],
            rows=[[f"{st['stage']} {st['name']}", st['description'],
                   _mm(st['surface_settlement_m']),
                   _mm(st['cumulative_surface_settlement_m'])]
                  for st in result.stages],
            notes=("Surface settlement, negative = down. Reported "
                   f"settlements are measured from the "
                   f"{ref.get('measured_from', 'start of analysis')} "
                   f"(chosen by: {ref.get('chosen_by', '-')})."),
        ))
        load = result.stages[-1]
        if load.get("time_s"):
            U = load.get("degree_of_consolidation_by_time") or \
                [None] * len(load["time_s"])
            items.append(TableData(
                title="Load Stage by Time",
                headers=["t (s)", "U", "Settlement this stage (mm)",
                         "Cumulative settlement (mm)"],
                rows=[[f"{tt:.3g}", "-" if u is None else f"{u:.3f}",
                       _mm(s), _mm(c)]
                      for tt, u, s, c in zip(
                          load["time_s"], U,
                          load["surface_settlement_m_by_time"],
                          load["cumulative_surface_settlement_m_by_time"])],
            ))
    if result.judgment:
        items.append(_judgment_item(result.judgment))
    return CalcSection(title="Consolidation Results", items=items)


def _staged_sections(result) -> list:
    sections = []
    ref = result.displacement_reference or {}

    def _s(m):
        return _mm(m['settlement_m']) if m else "-"

    overview_rows = [[
        str(p.phase_index + 1), p.phase_name,
        str(p.n_active_elements), str(p.n_active_beams),
        "yes" if p.converged else "NO",
        _s(p.delta_displacement), _s(p.displacement_since_reference),
        _s(p.cumulative_displacement),
    ] for p in result.phases]
    items = [TableData(
        title="Phase Overview",
        headers=["#", "Phase", "Active elements", "Active beams",
                 "Converged", "Settlement this stage (mm)",
                 "Settlement since reference (mm)",
                 "Cumulative settlement (mm)"],
        rows=overview_rows,
        notes=("Settlement = most negative vertical movement (negative = "
               "down) over the stage's active nodes. Stresses and plastic "
               "state carry forward between phases; an element activated "
               "in a stage is placed stress-free. Displacements reset after: "
               f"{ref.get('measured_from', 'start of analysis')} (chosen by: "
               f"{ref.get('chosen_by', '-')}); stage numbers here are 1-based, "
               f"stage indices in that statement are 0-based."),
    )]
    if result.judgment:
        items.append(_judgment_item(result.judgment))
    sections.append(CalcSection(title="Construction Sequence", items=items))

    for p in result.phases:
        items = []
        rows = [
            ["Converged", "yes" if p.converged else "NO"],
            ["Active elements / beams",
             f"{p.n_active_elements} / {p.n_active_beams}"],
        ]
        if p.delta_displacement is not None:
            rows += _movement_rows("This stage", p.delta_displacement)
            # Skip bases identical to the delta (stage 0 starts from rest;
            # a stage right after the reference is measured from its start).
            if (p.reference_stage_index is not None
                    and p.reference_stage_index < p.phase_index - 1):
                rows += _movement_rows(f"Since {p.reference_stage}",
                                       p.displacement_since_reference)
            if p.phase_index > 0:
                rows += _movement_rows("Cumulative",
                                       p.cumulative_displacement)
        else:
            rows += [
                ["Max displacement |u| (cumulative)",
                 f"{p.max_displacement_m:.4g} m"],
                ["Max u_x / u_y (cumulative)",
                 f"{p.max_displacement_x_m:.4g} / "
                 f"{p.max_displacement_y_m:.4g} m"],
            ]
        rows += [
            ["σ_yy range (active)",
             f"{p.min_sigma_yy_kPa:.1f} to "
             f"{p.max_sigma_yy_kPa:.1f} kPa"],
            ["max |τ_xy| (active)", f"{p.max_tau_xy_kPa:.1f} kPa"],
        ]
        if p.n_beam_elements > 0:
            rows.append(["Max beam moment / shear",
                         f"{p.max_beam_moment_kNm_per_m:.1f} kN·m/m / "
                         f"{p.max_beam_shear_kN_per_m:.1f} kN/m"])
        items.append(TableData(
            title=f"Phase {p.phase_index + 1} Results",
            headers=["Item", "Value"],
            rows=rows,
        ))
        sections.append(CalcSection(
            title=f"Phase {p.phase_index + 1}: {p.phase_name}",
            items=items,
        ))

    return sections


def _checks_section(result, analysis) -> CalcSection:
    items = []
    if getattr(result, 'FOS', None) is not None:
        req = (analysis or {}).get("FOS_required", 1.5)
        items.append(CheckItem(
            description="Slope stability adequacy (SRM)",
            demand=req,
            demand_label="FOS_required",
            capacity=result.FOS,
            capacity_label="FOS_computed",
            unit="",
            passes=result.FOS >= req,
        ))
    converged = getattr(result, 'converged', True)
    items.append(CheckItem(
        description="Solution convergence",
        demand=1.0,
        demand_label="required",
        capacity=1.0 if converged else 0.0,
        capacity_label="achieved",
        unit="",
        passes=bool(converged),
    ))
    return CalcSection(title="Checks", items=items)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def get_figures(result, analysis) -> List[FigureData]:
    """PLAXIS-style figures for the fem2d calc package.

    Figures degrade gracefully (skipped without matplotlib or when
    the required arrays are absent).

    Parameters
    ----------
    result : FEMResult | SeepageResult | ConsolidationResult |
        StagedConstructionResult
    analysis : dict or None

    Returns
    -------
    list of FigureData
    """
    analysis = analysis or {}
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
        except (ValueError, AttributeError, TypeError):
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

    from fem2d import plotting as _plots

    # ---- seepage --------------------------------------------------------
    if _is_seepage(result):
        _add(lambda: _plots.plot_seepage(result),
             "Head Contours & Flow Vectors",
             "Total head contours with element Darcy velocity "
             "vectors.")
        return figures

    # ---- consolidation --------------------------------------------------
    if _is_consolidation(result):
        ref = result.displacement_reference or {}
        _add(lambda: _plots.plot_consolidation_history(result),
             "Consolidation Time History",
             f"Settlement and excess pore pressure dissipation; "
             f"final degree of consolidation "
             f"U = {result.degree_of_consolidation:.2f}"
             + (f"; settlement measured from the {ref['measured_from']}"
                if ref.get('measured_from') else "") + ".", 75)
        return figures

    # ---- staged construction -------------------------------------------
    if _is_staged(result):
        import types
        for p in result.phases:
            if p.displacements is None or p.n_active_elements == 0:
                continue
            own = p.delta_displacements is not None
            shim = types.SimpleNamespace(
                nodes=result.nodes, elements=result.elements,
                displacements=(p.delta_displacements if own
                               else p.displacements),
                stresses=p.stresses)
            _add(lambda s=shim, name=p.phase_name, own=own:
                 _plots.plot_contour(
                     s, field='u_mag',
                     title=(f'|u| this stage — {name}' if own
                            else f'|u| — {name}')),
                 f"Displacement |u| — {p.phase_name}",
                 (f"Displacement magnitude of phase '{p.phase_name}' "
                  f"alone (this stage's own movement; the cumulative and "
                  f"since-reference values are in the phase table)."
                  if own else
                  f"Cumulative displacement magnitude at the end of "
                  f"phase '{p.phase_name}'."))
        return figures

    # ---- FEMResult ------------------------------------------------------
    if result.nodes is not None and result.elements is not None:
        _add(lambda: _plots.plot_mesh(
                 result.nodes, result.elements,
                 material_ids=analysis.get("material_ids"),
                 material_names=analysis.get("material_names"),
                 bc_nodes=analysis.get("bc_nodes")),
             "Finite Element Mesh",
             f"Mesh: {result.n_nodes} nodes, {result.n_elements} "
             f"{_element_type_label(result)} elements.")

        _add(lambda: _plots.plot_deformed_mesh(result),
             "Deformed Mesh",
             f"Deformed mesh (auto-scaled) over the undeformed "
             f"outline; u_max = {result.max_displacement_m:.4g} m.")

        if result.analysis_type == "srm":
            _add(lambda: _plots.plot_failure_mechanism(result),
                 "Failure Mechanism",
                 f"Displacement magnitude at the last stable SRF — "
                 f"the localized band marks the critical mechanism. "
                 f"FOS = {result.FOS:.3f}."
                 if result.FOS is not None else
                 "Displacement magnitude at the last stable SRF.")
            _add(lambda: _plots.plot_plastic_points(result),
                 "Plastic Point Map",
                 "Gauss points on the reduced-strength Mohr-Coulomb "
                 "yield surface at the last stable state.")
            _add(lambda: _plots.plot_srf_curve(result),
                 "SRF vs Displacement",
                 "Strength reduction trials: dimensionless "
                 "displacement vs SRF with the bracketed FOS.", 75)
        else:
            _add(lambda: _plots.plot_contour(result, field='u_mag'),
                 "Displacement Contours",
                 "Displacement magnitude |u|.")

        if result.stresses is not None:
            _add(lambda: _plots.plot_contour(result, field='sigma_yy'),
                 "Vertical Stress Contours",
                 "Vertical stress σ_yy (tension positive).")
            _add(lambda: _plots.plot_contour(result, field='tau_max'),
                 "Maximum Shear Stress Contours",
                 "Maximum in-plane shear stress τ_max "
                 "(Mohr circle radius).")

    return figures
