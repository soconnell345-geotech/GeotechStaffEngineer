"""Calc package adapter — flat dict → analysis → generate_calc_package → summary dict.

Generates Mathcad-style HTML/LaTeX/PDF calculation packages for 13 analysis
modules.  Each handler builds domain objects, runs the analysis, and calls
generate_calc_package().  The response includes a file path and key results
(never the full HTML, which would exceed the 8000-char ReAct truncation limit).

``slope_report_package`` is the one-call slope REPORT for a critical-surface
SEARCH run (F7): it runs the search (no fixed circle), re-analyzes the critical
surface with method comparison + slice data, and optionally runs FOSM/Monte
Carlo, then renders the full report (search story with rejection diagnostics,
FOS-by-method table, interslice/thrust figures, reinforcement + probabilistic
annexes) via slope_stability.calc_steps.

Ported from foundry/calc_package_agent_foundry.py — same object construction
logic, adapted to the funhouse adapter pattern (METHOD_REGISTRY + METHOD_INFO).
"""

import os
from datetime import datetime

from funhouse_agent.adapters import (require_keys, require_params,
                                     reject_unknown_params)
from funhouse_agent._fileio import (
    default_output_dir, rescue_write, workspace_write_hint,
    written_file_problem,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_metadata(params: dict) -> dict:
    """Extract common calc-package header fields from params."""
    return {
        "project_name": params.get("project_name", "Project"),
        "project_number": params.get("project_number", ""),
        "engineer": params.get("engineer", ""),
        "checker": params.get("checker", ""),
        "company": params.get("company", ""),
        "date": params.get("date", ""),
    }


def _default_output_path(module_name: str, fmt: str = "html") -> str:
    """Generate a timestamped default output filename.

    On Databricks (or whenever the CWD is a /Workspace FUSE path) the file
    goes to the system temp dir — the notebook CWD is a workspace path where
    plain file writes are not durably stored.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(default_output_dir(), f"{module_name}_calc_{ts}.{fmt}")


def _build_response(module: str, result, analysis, params: dict,
                    analysis_type: str, extra: dict = None) -> dict:
    """Run generate_calc_package and return a summary response dict.

    Always saves to disk.  Auto-generates output_path if not provided.
    """
    from calc_package import generate_calc_package

    meta = _extract_metadata(params)
    fmt = params.get("format", "html")
    output_path = params.get("output_path") or _default_output_path(module, fmt)

    render_report = {}
    content = generate_calc_package(
        module=module,
        result=result,
        analysis=analysis,
        output_path=output_path,
        format=fmt,
        render_report=render_report,
        **meta,
    )

    # Verify the write on the REAL filesystem so callers (LLM agents) never
    # need a separate ls/read to trust the save. Agent-side filesystem tools
    # may be sandboxed/virtual and cannot see this file. Content (not just
    # existence) is checked: Databricks /Workspace FUSE writes can "succeed"
    # while the workspace stores only a literal PLACEHOLDER file.
    abs_path = os.path.abspath(output_path)
    if fmt == "html" and isinstance(content, str):
        expected = content.encode("utf-8", errors="replace")
    else:
        # pdf: generate_calc_package returns the path, not content.
        # latex: the saved file is rendered with figure paths and differs
        # from the returned string, so only existence/size can be checked.
        expected = None
    problem = written_file_problem(abs_path, expected)
    file_exists = os.path.isfile(abs_path)
    response = {
        "status": "success",
        "analysis_type": analysis_type,
        "output_path": abs_path,
        "file_exists": file_exists and problem is None,
        "file_size_bytes": os.path.getsize(abs_path) if file_exists else 0,
        "format": fmt,
        "html_length": len(content) if isinstance(content, str) else None,
    }
    # For PDFs, surface which engine produced the file (pdflatex vs the
    # pure-Python PyMuPDF Story fallback) so environment differences are
    # diagnosable, plus any renderer warnings (e.g. "pdflatex not found").
    if fmt == "pdf" and render_report.get("renderer"):
        response["renderer"] = render_report["renderer"]
        if render_report.get("warnings"):
            response["renderer_warnings"] = render_report["warnings"]
    if problem:
        rescue = rescue_write(abs_path, expected) if expected is not None else None
        response["status"] = "error"
        response["error"] = (
            f"Calc package was generated but {problem}."
            + workspace_write_hint(abs_path)
        )
        if rescue:
            response["rescue_path"] = rescue
            response["error"] += (
                f" A verified copy of the calc package was saved to "
                f"'{rescue}' — report THAT path to the user."
            )
    if extra:
        response.update(extra)
    return response


# ---------------------------------------------------------------------------
# 1. Bearing Capacity
# ---------------------------------------------------------------------------

def _generate_bearing_capacity_package(params: dict) -> dict:
    from bearing_capacity.footing import Footing
    from bearing_capacity.soil_profile import SoilLayer, BearingSoilProfile
    from bearing_capacity.capacity import BearingCapacityAnalysis

    require_params(params, ["width", "unit_weight"],
                   method="bearing_capacity_package")

    footing = Footing(
        width=params["width"],
        length=params.get("length"),
        depth=params.get("depth", 0.0),
        shape=params.get("shape", "strip"),
        base_tilt=params.get("base_tilt", 0.0),
        eccentricity_B=params.get("eccentricity_B", 0.0),
        eccentricity_L=params.get("eccentricity_L", 0.0),
    )
    layer1 = SoilLayer(
        cohesion=params.get("cohesion", 0.0),
        friction_angle=params.get("friction_angle", 0.0),
        unit_weight=params["unit_weight"],
    )
    soil = BearingSoilProfile(layer1=layer1, gwt_depth=params.get("gwt_depth"))
    analysis = BearingCapacityAnalysis(
        footing=footing, soil=soil,
        load_inclination=params.get("load_inclination", 0.0),
        ground_slope=params.get("ground_slope", 0.0),
        vertical_load=params.get("vertical_load", 0.0),
        factor_of_safety=params.get("factor_of_safety", 3.0),
        ngamma_method=params.get("ngamma_method", "vesic"),
        factor_method=params.get("factor_method", "vesic"),
    )
    result = analysis.compute()

    return _build_response(
        "bearing_capacity", result, analysis, params,
        analysis_type="Bearing Capacity",
        extra={
            "q_ultimate_kPa": round(result.q_ultimate, 1),
            "q_allowable_kPa": round(result.q_allowable, 1),
            "factor_of_safety": result.factor_of_safety,
        },
    )


# ---------------------------------------------------------------------------
# 2. Lateral Pile
# ---------------------------------------------------------------------------

def _generate_lateral_pile_package(params: dict) -> dict:
    from lateral_pile import Pile, LateralPileAnalysis
    from lateral_pile.soil import SoilLayer
    # Same p-y construction/validation as lateral_pile.lateral_pile_analysis,
    # so both tools accept identical layer dicts and fail with the same
    # actionable messages (model names, required keys) instead of raw
    # KeyError/TypeError.
    from funhouse_agent.adapters.lateral_pile import _build_py_model

    require_params(
        params, ["pile_length", "pile_diameter", "pile_E", "soil_layers"],
        method="lateral_pile_package",
        valid=["pile_length", "pile_diameter", "pile_E", "pile_thickness",
               "moment_of_inertia", "soil_layers", "Vt", "Mt", "Q",
               "head_condition", "output_path", "format", "project_name",
               "project_number", "engineer", "checker", "company", "date"],
    )

    pile = Pile(
        length=params["pile_length"],
        diameter=params["pile_diameter"],
        thickness=params.get("pile_thickness"),
        E=params["pile_E"],
        moment_of_inertia=params.get("moment_of_inertia"),
    )

    layers = []
    for ld in params["soil_layers"]:
        d = dict(ld)
        # The analysis tool calls this key 'model'; accept both spellings.
        if d.get("model") is None and d.get("py_model") is not None:
            d["model"] = d.pop("py_model")
        if d.get("model") is None:
            raise ValueError(
                "lateral_pile_package: each soil_layers[] dict requires "
                "'py_model' (p-y model name; 'model' also accepted) "
                f"(got keys {sorted(ld.keys())})."
            )
        require_keys(d, ["top", "bottom"], method="lateral_pile_package",
                     item_label="soil_layers[]")
        layers.append(SoilLayer(
            top=d["top"], bottom=d["bottom"],
            py_model=_build_py_model(d), description=d.get("description"),
        ))

    analysis = LateralPileAnalysis(pile, layers)
    result = analysis.solve(
        Vt=params.get("Vt", 0.0),
        Mt=params.get("Mt", 0.0),
        Q=params.get("Q", 0.0),
        head_condition=params.get("head_condition", "free"),
    )

    return _build_response(
        "lateral_pile", result, analysis, params,
        analysis_type="Lateral Pile",
        extra={
            "y_top_mm": round(result.y_top * 1000, 2),
            "max_moment_kNm": round(result.max_moment, 1),
            "converged": result.converged,
        },
    )


# ---------------------------------------------------------------------------
# 3. Slope Stability
# ---------------------------------------------------------------------------

def _build_slope_geom(params: dict, method: str):
    """Build a SlopeGeometry from flat params. Shared by the slope packages."""
    from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer

    soil_layers = []
    for ld in params["soil_layers"]:
        require_keys(ld, ["name", "top_elevation", "bottom_elevation", "gamma"],
                     method=method, item_label="soil_layers[]")
        soil_layers.append(SlopeSoilLayer(
            name=ld["name"],
            top_elevation=ld["top_elevation"],
            bottom_elevation=ld["bottom_elevation"],
            gamma=ld["gamma"],
            gamma_sat=ld.get("gamma_sat"),
            phi=ld.get("phi", 0.0),
            c_prime=ld.get("c_prime", 0.0),
            cu=ld.get("cu", 0.0),
            analysis_mode=ld.get("analysis_mode", "drained"),
            bottom_boundary_points=(
                [tuple(p) for p in ld["bottom_boundary_points"]]
                if ld.get("bottom_boundary_points") else None),
        ))

    surface_points = [tuple(p) for p in params["surface_points"]]
    gwt_points = None
    if params.get("gwt_points"):
        gwt_points = [tuple(p) for p in params["gwt_points"]]

    return SlopeGeometry(
        surface_points=surface_points,
        soil_layers=soil_layers,
        gwt_points=gwt_points,
        surcharge=params.get("surcharge", 0.0),
        kh=params.get("kh", 0.0),
    )


def _generate_slope_stability_package(params: dict) -> dict:
    from slope_stability.analysis import analyze_slope

    require_params(params, ["surface_points", "soil_layers", "xc", "yc", "radius"],
                   method="slope_stability_package")

    geom = _build_slope_geom(params, "slope_stability_package")

    result = analyze_slope(
        geom,
        xc=params["xc"], yc=params["yc"], radius=params["radius"],
        method=params.get("method", "bishop"),
        n_slices=params.get("n_slices", 30),
        include_slice_data=True,
        compare_methods=params.get("compare_methods", False),
    )

    # slope_stability calc_steps expects a dict with "geom" + optional metadata
    analysis_dict = {
        "geom": geom,
        "FOS_required": params.get("FOS_required", 1.5),
    }
    return _build_response(
        "slope_stability", result, analysis_dict, params,
        analysis_type="Slope Stability",
        extra={
            "FOS": round(result.FOS, 3),
            "method": result.method,
            "is_stable": result.FOS >= analysis_dict.get("FOS_required", 1.5),
        },
    )


def _generate_slope_report_package(params: dict) -> dict:
    """One-call slope calc REPORT for a critical-surface SEARCH run.

    Runs a critical-surface search (not a single fixed circle), then re-analyzes
    the critical surface with method comparison + slice data so the package
    carries: input echo, the search story (surfaces tried + rejection
    diagnostics + critical surface/FOS), a FOS-by-method table, the slice-force
    and (for rigorous methods) interslice/thrust-line figures, a reinforcement
    summary when present, and a probabilistic annex when a ``variables`` spec is
    supplied (FOSM by default, Monte Carlo when ``monte_carlo=True``). The rich
    rendering lives in slope_stability.calc_steps — this handler just feeds it a
    search + probabilistic results in the ``analysis`` dict.
    """
    from slope_stability.analysis import analyze_slope, search_critical_surface

    require_params(params, ["surface_points", "soil_layers"],
                   method="slope_report_package")
    reject_unknown_params(
        params,
        valid=[
            # geometry
            "surface_points", "soil_layers", "gwt_points", "surcharge", "kh",
            # search
            "method", "surface_type", "n_slices", "tol", "x_range", "y_range",
            "nx", "ny", "x_entry_range", "x_exit_range", "n_trials", "n_points",
            "seed", "f_interslice", "FOS_required",
            # probabilistic annex
            "variables", "fosm", "monte_carlo", "n", "research_surface",
            # output + header metadata
            "output_path", "format", "project_name", "project_number",
            "engineer", "checker", "company", "date",
        ],
        method="slope_report_package",
    )

    geom = _build_slope_geom(params, "slope_report_package")

    method = params.get("method", "spencer")
    surface_type = params.get("surface_type", "circular")
    n_slices = params.get("n_slices", 30)
    tol = params.get("tol", 1e-4)

    def _range(key):
        return tuple(params[key]) if params.get(key) else None

    search = search_critical_surface(
        geom=geom,
        x_range=_range("x_range"), y_range=_range("y_range"),
        nx=params.get("nx", 10), ny=params.get("ny", 10),
        method=method, n_slices=n_slices, tol=tol,
        surface_type=surface_type,
        x_entry_range=_range("x_entry_range"),
        x_exit_range=_range("x_exit_range"),
        n_trials=params.get("n_trials", 500),
        n_points=params.get("n_points", 5),
        seed=params.get("seed"),
    )

    crit = search.critical
    if crit is None:
        return {
            "status": "error",
            "analysis_type": "Slope Stability Report",
            "error": (
                "Critical-surface search found no admissible surface "
                f"({search.n_surfaces_evaluated} evaluated). Widen the search "
                "grid (x_range/y_range for circular, or x_entry_range/"
                "x_exit_range) or check the geometry."
            ),
        }

    # Re-analyze the critical surface with method comparison + slice data so the
    # report carries the FOS-by-method table and, for a rigorous method, the
    # interslice-force / line-of-thrust figure. Uses the SAME method the search
    # minimized, so the reported critical FOS stays consistent with the search.
    if crit.is_circular:
        surf_kwargs = dict(xc=crit.xc, yc=crit.yc, radius=crit.radius)
    else:
        from slope_stability.slip_surface import PolylineSlipSurface
        surf_kwargs = dict(slip_surface=PolylineSlipSurface(crit.slip_points))
    result = analyze_slope(
        geom, method=method, n_slices=n_slices, tol=tol,
        f_interslice=params.get("f_interslice", "half_sine"),
        include_slice_data=True, compare_methods=True, **surf_kwargs,
    )

    analysis_dict = {
        "geom": geom,
        "FOS_required": params.get("FOS_required", 1.5),
        "search": search,
        "f_interslice": params.get("f_interslice", "half_sine"),
    }

    # Optional probabilistic annex (reliability) on the critical surface.
    variables = params.get("variables")
    if variables:
        analysis_dict["variables"] = variables
        prob_surf = (dict(xc=crit.xc, yc=crit.yc, radius=crit.radius)
                     if crit.is_circular
                     else dict(slip_surface=PolylineSlipSurface(crit.slip_points)))
        if params.get("fosm", True):
            from slope_stability.probabilistic import fosm_fos
            analysis_dict["fosm"] = fosm_fos(
                geom, variables, method=method, n_slices=n_slices, tol=tol,
                **prob_surf)
        if params.get("monte_carlo", False):
            from slope_stability.probabilistic import monte_carlo_fos
            analysis_dict["mc"] = monte_carlo_fos(
                geom, variables, method=method, n=params.get("n", 1000),
                seed=params.get("seed"), n_slices=n_slices, tol=tol,
                research_surface=params.get("research_surface", False),
                **prob_surf)

    return _build_response(
        "slope_stability", result, analysis_dict, params,
        analysis_type="Slope Stability Report",
        extra={
            "FOS": round(result.FOS, 3),
            "method": result.method,
            "n_surfaces_evaluated": search.n_surfaces_evaluated,
            "is_stable": result.FOS >= analysis_dict.get("FOS_required", 1.5),
        },
    )


# ---------------------------------------------------------------------------
# 4. Settlement
# ---------------------------------------------------------------------------

def _generate_settlement_package(params: dict) -> dict:
    from settlement import SettlementAnalysis, ConsolidationLayer, SchmertmannLayer

    consol_layers = None
    if params.get("consolidation_layers"):
        consol_layers = []
        for ld in params["consolidation_layers"]:
            require_keys(ld, ["thickness", "depth_to_center", "e0", "Cc", "Cr",
                              "sigma_v0"],
                         method="settlement_package",
                         item_label="consolidation_layers[]")
            consol_layers.append(ConsolidationLayer(
                thickness=ld["thickness"],
                depth_to_center=ld["depth_to_center"],
                e0=ld["e0"],
                Cc=ld["Cc"],
                Cr=ld["Cr"],
                sigma_v0=ld["sigma_v0"],
                sigma_p=ld.get("sigma_p"),
                description=ld.get("description", ""),
            ))

    schm_layers = None
    if params.get("schmertmann_layers"):
        schm_layers = []
        for ld in params["schmertmann_layers"]:
            require_keys(ld, ["depth_top", "depth_bottom", "Es"],
                         method="settlement_package",
                         item_label="schmertmann_layers[]")
            schm_layers.append(SchmertmannLayer(
                depth_top=ld["depth_top"],
                depth_bottom=ld["depth_bottom"],
                Es=ld["Es"],
            ))

    analysis = SettlementAnalysis(
        q_applied=params.get("q_applied", 0.0),
        q_overburden=params.get("q_overburden", 0.0),
        B=params.get("B", 1.0),
        L=params.get("L", 1.0),
        footing_shape=params.get("footing_shape", "square"),
        stress_method=params.get("stress_method", "2:1"),
        immediate_method=params.get("immediate_method", "elastic"),
        Es_immediate=params.get("Es_immediate"),
        nu=params.get("nu", 0.3),
        schmertmann_layers=schm_layers,
        time_years_schmertmann=params.get("time_years_schmertmann", 0.0),
        consolidation_layers=consol_layers,
        cv=params.get("cv"),
        Hdr=params.get("Hdr"),
        drainage=params.get("drainage", "double"),
        C_alpha=params.get("C_alpha"),
        e0_secondary=params.get("e0_secondary", 1.0),
        t_secondary=params.get("t_secondary", 0.0),
    )
    result = analysis.compute()

    return _build_response(
        "settlement", result, analysis, params,
        analysis_type="Settlement",
        extra={
            "total_settlement_mm": round(result.total * 1000, 2),
            "immediate_mm": round(result.immediate * 1000, 2),
            "consolidation_mm": round(result.consolidation * 1000, 2),
            "secondary_mm": round(result.secondary * 1000, 2),
        },
    )


# ---------------------------------------------------------------------------
# 5. Axial Pile
# ---------------------------------------------------------------------------

def _generate_axial_pile_package(params: dict) -> dict:
    from axial_pile import AxialPileAnalysis, AxialSoilLayer, AxialSoilProfile, PileSection

    require_params(params, ["pile_area", "pile_perimeter", "pile_tip_area",
                            "pile_width", "pile_length", "soil_layers"],
                   method="axial_pile_package")

    pile = PileSection(
        name=params.get("pile_name", "Custom Pile"),
        pile_type=params.get("pile_type", "h_pile"),
        area=params["pile_area"],
        perimeter=params["pile_perimeter"],
        tip_area=params["pile_tip_area"],
        width=params["pile_width"],
        depth=params.get("pile_depth"),
    )

    layers = []
    for ld in params["soil_layers"]:
        require_keys(ld, ["soil_type", "thickness", "unit_weight"],
                     method="axial_pile_package", item_label="soil_layers[]")
        layers.append(AxialSoilLayer(
            soil_type=ld["soil_type"],
            thickness=ld["thickness"],
            unit_weight=ld["unit_weight"],
            friction_angle=ld.get("friction_angle", 0.0),
            cohesion=ld.get("cohesion", 0.0),
            delta_phi_ratio=ld.get("delta_phi_ratio"),
            description=ld.get("description", ""),
        ))
    soil = AxialSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth"))

    analysis = AxialPileAnalysis(
        pile=pile, soil=soil,
        pile_length=params["pile_length"],
        factor_of_safety=params.get("factor_of_safety", 2.5),
        method=params.get("method", "auto"),
    )
    result = analysis.compute()

    return _build_response(
        "axial_pile", result, analysis, params,
        analysis_type="Axial Pile Capacity",
        extra={
            "Q_ultimate_kN": round(result.Q_ultimate, 1),
            "Q_allowable_kN": round(result.Q_allowable, 1),
            "Q_skin_kN": round(result.Q_skin, 1),
            "Q_tip_kN": round(result.Q_tip, 1),
        },
    )


# ---------------------------------------------------------------------------
# 6. Drilled Shaft
# ---------------------------------------------------------------------------

def _generate_drilled_shaft_package(params: dict) -> dict:
    from drilled_shaft import DrillShaft, ShaftSoilLayer, ShaftSoilProfile, DrillShaftAnalysis

    require_params(params, ["diameter", "length", "soil_layers"],
                   method="drilled_shaft_package")

    shaft = DrillShaft(
        diameter=params["diameter"],
        length=params["length"],
        socket_diameter=params.get("socket_diameter"),
        socket_length=params.get("socket_length", 0.0),
        bell_diameter=params.get("bell_diameter"),
        casing_depth=params.get("casing_depth", 0.0),
        concrete_fc=params.get("concrete_fc", 28000.0),
    )

    layers = []
    for ld in params["soil_layers"]:
        require_keys(ld, ["soil_type", "thickness", "unit_weight"],
                     method="drilled_shaft_package", item_label="soil_layers[]")
        layers.append(ShaftSoilLayer(
            soil_type=ld["soil_type"],
            thickness=ld["thickness"],
            unit_weight=ld["unit_weight"],
            cu=ld.get("cu", 0.0),
            phi=ld.get("phi", 0.0),
            N60=ld.get("N60", 0),
            qu=ld.get("qu", 0.0),
            RQD=ld.get("RQD", 100.0),
            description=ld.get("description", ""),
        ))
    soil = ShaftSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth"))

    analysis = DrillShaftAnalysis(
        shaft=shaft, soil=soil,
        factor_of_safety=params.get("factor_of_safety", 2.5),
    )
    result = analysis.compute()

    return _build_response(
        "drilled_shaft", result, analysis, params,
        analysis_type="Drilled Shaft Capacity",
        extra={
            "Q_ultimate_kN": round(result.Q_ultimate, 1),
            "Q_allowable_kN": round(result.Q_allowable, 1),
            "Q_skin_kN": round(result.Q_skin, 1),
            "Q_tip_kN": round(result.Q_tip, 1),
        },
    )


# ---------------------------------------------------------------------------
# 7. Downdrag
# ---------------------------------------------------------------------------

def _generate_downdrag_package(params: dict) -> dict:
    from downdrag import DowndragAnalysis, DowndragSoilLayer, DowndragSoilProfile

    require_params(params, ["pile_length", "pile_diameter", "pile_E", "Q_dead",
                            "soil_layers"],
                   method="downdrag_package")

    layers = []
    for ld in params["soil_layers"]:
        require_keys(ld, ["soil_type", "thickness", "unit_weight"],
                     method="downdrag_package", item_label="soil_layers[]")
        layers.append(DowndragSoilLayer(
            soil_type=ld["soil_type"],
            thickness=ld["thickness"],
            unit_weight=ld["unit_weight"],
            cu=ld.get("cu", 0.0),
            phi=ld.get("phi", 0.0),
            settling=ld.get("settling", False),
            alpha=ld.get("alpha"),
            beta=ld.get("beta"),
            Cc=ld.get("Cc"),
            e0=ld.get("e0"),
            sigma_p=ld.get("sigma_p"),
            description=ld.get("description", ""),
        ))
    soil = DowndragSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth", 0.0))

    analysis = DowndragAnalysis(
        pile_length=params["pile_length"],
        pile_diameter=params["pile_diameter"],
        pile_E=params["pile_E"],
        pile_unit_weight=params.get("pile_unit_weight", 78.5),
        Q_dead=params["Q_dead"],
        soil=soil,
        fill_thickness=params.get("fill_thickness", 0.0),
        fill_unit_weight=params.get("fill_unit_weight", 20.0),
        gw_drawdown=params.get("gw_drawdown", 0.0),
        structural_capacity=params.get("structural_capacity"),
        allowable_settlement=params.get("allowable_settlement"),
        Nt=params.get("Nt"),
    )
    result = analysis.compute()

    return _build_response(
        "downdrag", result, analysis, params,
        analysis_type="Downdrag",
        extra={
            "neutral_plane_m": round(result.neutral_plane_depth, 2),
            "dragload_kN": round(result.dragload, 1),
            "max_pile_load_kN": round(result.max_pile_load, 1),
            "pile_settlement_mm": round(result.pile_settlement * 1000, 2),
        },
    )


# ---------------------------------------------------------------------------
# 8. Seismic Geotechnical
# ---------------------------------------------------------------------------

def _generate_seismic_package(params: dict) -> dict:
    from seismic_geotech import (
        classify_site, site_coefficients, mononobe_okabe_KAE,
        seismic_earth_pressure, evaluate_liquefaction,
    )
    from seismic_geotech.results import (
        SiteClassResult, SeismicEarthPressureResult, LiquefactionResult,
    )

    analysis_type = params.get("analysis_type", "site_classification")
    analysis_dict = dict(params)

    if analysis_type == "site_classification":
        require_params(params, ["Ss", "S1"],
                       method="seismic_package (site_classification)")
        if all(params.get(k) is None for k in ("vs30", "n_bar", "su_bar")):
            raise ValueError(
                "seismic_package (site_classification): provide one of "
                "'vs30' (m/s), 'n_bar' (avg SPT N), or 'su_bar' (kPa)."
            )
        site_class = classify_site(
            vs30=params.get("vs30"),
            n_bar=params.get("n_bar"),
            su_bar=params.get("su_bar"),
        )
        result = site_coefficients(site_class, Ss=params["Ss"], S1=params["S1"])

    elif analysis_type == "seismic_earth_pressure":
        require_params(params, ["phi", "kh", "gamma", "H"],
                       method="seismic_package (seismic_earth_pressure)")
        KAE = mononobe_okabe_KAE(
            phi_deg=params["phi"],
            delta_deg=params.get("delta", params["phi"] * 2 / 3),
            kh=params["kh"], kv=params.get("kv", 0.0),
        )
        result_dict = seismic_earth_pressure(
            gamma=params["gamma"], H=params["H"], KAE=KAE,
            KA=params.get("KA", 0.333),
        )
        result = SeismicEarthPressureResult(
            KAE=KAE, KPE=0.0,
            PAE_total=result_dict["PAE_total"],
            PA_static=result_dict["PA_static"],
            delta_PAE=result_dict["delta_PAE"],
            height_of_application=result_dict["height_of_application"],
            phi=params["phi"],
            delta=params.get("delta", params["phi"] * 2 / 3),
            kh=params["kh"], kv=params.get("kv", 0.0),
        )

    elif analysis_type == "liquefaction":
        require_params(params, ["layer_depths", "layer_N160", "layer_FC",
                                "layer_gamma", "amax_g", "gwt_depth"],
                       method="seismic_package (liquefaction)")
        layer_results = evaluate_liquefaction(
            layer_depths=params["layer_depths"],
            layer_N160=params["layer_N160"],
            layer_FC=params["layer_FC"],
            layer_gamma=params["layer_gamma"],
            amax_g=params["amax_g"],
            gwt_depth=params["gwt_depth"],
            M=params.get("M", 7.5),
        )
        result = LiquefactionResult(
            layer_results=layer_results,
            amax_g=params["amax_g"],
            magnitude=params.get("M", 7.5),
            gwt_depth=params["gwt_depth"],
        )
    else:
        raise ValueError(
            f"Unknown analysis_type '{analysis_type}'. Allowed: "
            "['site_classification', 'seismic_earth_pressure', 'liquefaction']."
        )

    return _build_response(
        "seismic_geotech", result, analysis_dict, params,
        analysis_type="Seismic Geotechnical",
        extra={"seismic_analysis_type": analysis_type},
    )


# ---------------------------------------------------------------------------
# 9. Retaining Wall
# ---------------------------------------------------------------------------

def _generate_retaining_wall_package(params: dict) -> dict:
    from retaining_walls import (
        CantileverWallGeometry, analyze_cantilever_wall,
    )

    wall_type = params.get("wall_type", "cantilever")
    analysis_dict = dict(params)

    if wall_type == "cantilever":
        require_params(params, ["wall_height", "gamma_backfill", "phi_backfill"],
                       method="retaining_wall_package")
        geom = CantileverWallGeometry(
            wall_height=params["wall_height"],
            base_width=params.get("base_width"),
            toe_length=params.get("toe_length"),
            stem_thickness_top=params.get("stem_thickness_top", 0.3),
            stem_thickness_base=params.get("stem_thickness_base"),
            base_thickness=params.get("base_thickness", 0.6),
            backfill_slope=params.get("backfill_slope", 0.0),
            surcharge=params.get("surcharge", 0.0),
        )
        result = analyze_cantilever_wall(
            geom=geom,
            gamma_backfill=params["gamma_backfill"],
            phi_backfill=params["phi_backfill"],
            c_backfill=params.get("c_backfill", 0.0),
            phi_foundation=params.get("phi_foundation"),
            c_foundation=params.get("c_foundation", 0.0),
            q_allowable=params.get("q_allowable"),
            gamma_concrete=params.get("gamma_concrete", 24.0),
            FOS_sliding=params.get("FOS_sliding", 1.5),
            FOS_overturning=params.get("FOS_overturning", 2.0),
            pressure_method=params.get("pressure_method", "rankine"),
        )
        analysis_dict["geom"] = geom
        extra = {
            "FOS_sliding": round(result.FOS_sliding, 2),
            "FOS_overturning": round(result.FOS_overturning, 2),
        }
    else:
        raise ValueError("MSE wall calc package not yet implemented")

    return _build_response(
        "retaining_walls", result, analysis_dict, params,
        analysis_type="Retaining Wall",
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 10. Ground Improvement
# ---------------------------------------------------------------------------

def _generate_ground_improvement_package(params: dict) -> dict:
    from ground_improvement import (
        analyze_wick_drains, analyze_aggregate_piers,
        analyze_surcharge_preloading, analyze_vibro_compaction,
    )

    method = params.get("method", "wick_drains")
    analysis_dict = dict(params)

    if method == "wick_drains":
        require_params(params, ["spacing", "ch", "cv", "Hdr", "time"],
                       method="ground_improvement_package (wick_drains)")
        result = analyze_wick_drains(
            spacing=params["spacing"],
            ch=params["ch"], cv=params["cv"],
            Hdr=params["Hdr"], time=params["time"],
            dw=params.get("dw", 0.066),
            pattern=params.get("pattern", "triangular"),
            smear_ratio=params.get("smear_ratio", 2.0),
            kh_ks_ratio=params.get("kh_ks_ratio", 2.0),
        )
    elif method == "aggregate_piers":
        require_params(params, ["diameter", "spacing", "length"],
                       method="ground_improvement_package (aggregate_piers)")
        result = analyze_aggregate_piers(
            diameter=params["diameter"],
            spacing=params["spacing"],
            length=params["length"],
            E_pier=params.get("E_pier", 50000.0),
            E_soil=params.get("E_soil", 5000.0),
            phi_pier=params.get("phi_pier", 45.0),
            applied_stress=params.get("applied_stress", 100.0),
        )
    elif method == "surcharge":
        require_params(params, ["H_clay", "cv", "surcharge", "sigma_v0", "Cc", "e0"],
                       method="ground_improvement_package (surcharge)")
        result = analyze_surcharge_preloading(
            H_clay=params["H_clay"],
            cv=params["cv"],
            surcharge=params["surcharge"],
            sigma_v0=params["sigma_v0"],
            Cc=params["Cc"], e0=params["e0"],
            sigma_p=params.get("sigma_p"),
            target_consolidation=params.get("target_consolidation", 90.0),
        )
    elif method == "vibro":
        require_params(params, ["spacing", "N1_before", "depth"],
                       method="ground_improvement_package (vibro)")
        result = analyze_vibro_compaction(
            spacing=params["spacing"],
            N1_before=params["N1_before"],
            depth=params["depth"],
        )
    else:
        raise ValueError(
            f"Unknown ground improvement method '{method}'. Allowed: "
            "['wick_drains', 'aggregate_piers', 'surcharge', 'vibro']."
        )

    return _build_response(
        "ground_improvement", result, analysis_dict, params,
        analysis_type="Ground Improvement",
        extra={"method": method},
    )


# ---------------------------------------------------------------------------
# 11. Wave Equation
# ---------------------------------------------------------------------------

def _generate_wave_equation_package(params: dict) -> dict:
    from wave_equation import (
        get_hammer, make_cushion_from_properties, Cushion,
        discretize_pile, generate_bearing_graph,
    )

    require_params(params, ["hammer_name", "pile_length", "pile_area", "pile_E"],
                   method="wave_equation_package")
    if params.get("cushion_stiffness") is None and any(
            params.get(k) is None
            for k in ("cushion_area", "cushion_thickness", "cushion_E")):
        raise ValueError(
            "wave_equation_package: provide either 'cushion_stiffness' (kN/m) "
            "or all of 'cushion_area' (m2), 'cushion_thickness' (m), "
            "'cushion_E' (kPa)."
        )

    hammer = get_hammer(params["hammer_name"])
    if "cushion_stiffness" in params:
        cushion = Cushion(stiffness=params["cushion_stiffness"],
                          cor=params.get("cushion_cor", 0.8))
    else:
        cushion = make_cushion_from_properties(
            area=params["cushion_area"],
            thickness=params["cushion_thickness"],
            elastic_modulus=params["cushion_E"],
            cor=params.get("cushion_cor", 0.8),
        )
    pile = discretize_pile(
        length=params["pile_length"],
        area=params["pile_area"],
        elastic_modulus=params["pile_E"],
        segment_length=params.get("segment_length", 1.0),
        unit_weight_material=params.get("pile_unit_weight", 78.5),
    )
    result = generate_bearing_graph(
        hammer=hammer, cushion=cushion, pile=pile,
        skin_fraction=params.get("skin_fraction", 0.5),
        quake_side=params.get("quake_side", 0.0025),
        quake_toe=params.get("quake_toe", 0.0025),
        damping_side=params.get("damping_side", 0.16),
        damping_toe=params.get("damping_toe", 0.5),
        R_min=params.get("R_min", 200.0),
        R_max=params.get("R_max", 2000.0),
        R_step=params.get("R_step", 200.0),
    )

    analysis_dict = {"hammer": hammer, "cushion": cushion, "pile": pile}
    return _build_response(
        "wave_equation", result, analysis_dict, params,
        analysis_type="Wave Equation",
        extra={"n_points": len(result.blow_counts)},
    )


# ---------------------------------------------------------------------------
# 12. Pile Group
# ---------------------------------------------------------------------------

def _generate_pile_group_package(params: dict) -> dict:
    from pile_group import (
        GroupPile, create_rectangular_layout, GroupLoad,
        analyze_vertical_group_simple, analyze_group_6dof,
    )

    if params.get("layout") == "rectangular":
        require_params(params, ["n_rows", "n_cols", "spacing_x", "spacing_y"],
                       method="pile_group_package (layout='rectangular')")
        piles = create_rectangular_layout(
            n_rows=params["n_rows"], n_cols=params["n_cols"],
            spacing_x=params["spacing_x"], spacing_y=params["spacing_y"],
        )
    else:
        if not params.get("piles"):
            raise ValueError(
                "pile_group_package: provide either layout='rectangular' with "
                "n_rows/n_cols/spacing_x/spacing_y, or a 'piles' list of "
                "{x, y} dicts."
            )
        piles = []
        for pd in params["piles"]:
            require_keys(pd, ["x", "y"], method="pile_group_package",
                         item_label="piles[]")
            piles.append(GroupPile(
                x=pd["x"], y=pd["y"],
                axial_capacity_compression=pd.get("axial_capacity_compression"),
                axial_capacity_tension=pd.get("axial_capacity_tension"),
                label=pd.get("label", ""),
            ))

    load = GroupLoad(
        Vx=params.get("Vx", 0.0), Vy=params.get("Vy", 0.0),
        Vz=params.get("Vz", 0.0),
        Mx=params.get("Mx", 0.0), My=params.get("My", 0.0),
        Mz=params.get("Mz", 0.0),
    )

    if params.get("method_6dof", False):
        result = analyze_group_6dof(piles=piles, load=load)
    else:
        result = analyze_vertical_group_simple(piles=piles, load=load)

    analysis_dict = {
        "piles": piles, "load": load,
        "pile_diameter": params.get("pile_diameter", 0.3),
        "pile_spacing": params.get("pile_spacing", 1.5),
    }
    return _build_response(
        "pile_group", result, analysis_dict, params,
        analysis_type="Pile Group",
        extra={
            "n_piles": result.n_piles,
            "group_efficiency": round(result.group_efficiency, 3)
            if hasattr(result, "group_efficiency") else None,
        },
    )


# ---------------------------------------------------------------------------
# 13. Sheet Pile
# ---------------------------------------------------------------------------

def _generate_sheet_pile_package(params: dict) -> dict:
    from sheet_pile import WallSoilLayer, analyze_cantilever, analyze_anchored

    require_params(params, ["excavation_depth", "soil_layers"],
                   method="sheet_pile_package")
    if params.get("wall_type") == "anchored":
        require_params(params, ["anchor_depth"],
                       method="sheet_pile_package (wall_type='anchored')")

    layers = []
    for ld in params["soil_layers"]:
        require_keys(ld, ["thickness", "unit_weight"],
                     method="sheet_pile_package", item_label="soil_layers[]")
        layers.append(WallSoilLayer(
            thickness=ld["thickness"],
            unit_weight=ld["unit_weight"],
            friction_angle=ld.get("friction_angle", 30.0),
            cohesion=ld.get("cohesion", 0.0),
            description=ld.get("description", ""),
        ))

    wall_type = params.get("wall_type", "cantilever")
    analysis_dict = dict(params)
    analysis_dict["soil_layers"] = layers

    if wall_type == "cantilever":
        result = analyze_cantilever(
            soil_layers=layers,
            excavation_depth=params["excavation_depth"],
            gwt_depth_active=params.get("gwt_depth_active"),
            gwt_depth_passive=params.get("gwt_depth_passive"),
            surcharge=params.get("surcharge", 0.0),
            FOS_passive=params.get("FOS_passive", 1.5),
            pressure_method=params.get("pressure_method", "rankine"),
        )
        extra = {
            "embedment_m": round(result.embedment_depth, 2),
            "max_moment_kNm_per_m": round(result.max_moment, 1),
        }
    else:
        result = analyze_anchored(
            soil_layers=layers,
            excavation_depth=params["excavation_depth"],
            anchor_depth=params["anchor_depth"],
            gwt_depth_active=params.get("gwt_depth_active"),
            gwt_depth_passive=params.get("gwt_depth_passive"),
            surcharge=params.get("surcharge", 0.0),
            FOS_passive=params.get("FOS_passive", 1.5),
            pressure_method=params.get("pressure_method", "rankine"),
        )
        extra = {
            "embedment_m": round(result.embedment_depth, 2),
            "anchor_force_kN_per_m": round(result.anchor_force, 1),
            "max_moment_kNm_per_m": round(result.max_moment, 1),
        }

    return _build_response(
        "sheet_pile", result, analysis_dict, params,
        analysis_type="Sheet Pile Wall",
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "bearing_capacity_package": _generate_bearing_capacity_package,
    "lateral_pile_package": _generate_lateral_pile_package,
    "slope_stability_package": _generate_slope_stability_package,
    "slope_report_package": _generate_slope_report_package,
    "settlement_package": _generate_settlement_package,
    "axial_pile_package": _generate_axial_pile_package,
    "drilled_shaft_package": _generate_drilled_shaft_package,
    "downdrag_package": _generate_downdrag_package,
    "seismic_package": _generate_seismic_package,
    "retaining_wall_package": _generate_retaining_wall_package,
    "ground_improvement_package": _generate_ground_improvement_package,
    "wave_equation_package": _generate_wave_equation_package,
    "pile_group_package": _generate_pile_group_package,
    "sheet_pile_package": _generate_sheet_pile_package,
}


# ---------------------------------------------------------------------------
# Method info — shared metadata fragments then per-method entries
# ---------------------------------------------------------------------------

_COMMON_PARAMS = {
    "project_name": {"type": "str", "required": False, "default": "Project",
                     "description": "Project name for header."},
    "project_number": {"type": "str", "required": False, "default": "",
                       "description": "Project number for header."},
    "engineer": {"type": "str", "required": False, "default": "",
                 "description": "Engineer name for header."},
    "checker": {"type": "str", "required": False, "default": "",
                "description": "Checker name for header."},
    "company": {"type": "str", "required": False, "default": "",
                "description": "Company name for header."},
    "date": {"type": "str", "required": False, "default": "",
             "description": "Date string (auto-filled if empty)."},
    "output_path": {"type": "str", "required": False,
                    "description": "File path to save output. Auto-generated if omitted."},
    "format": {"type": "str", "required": False, "default": "html",
               "allowed_values": ["html", "latex", "pdf"],
               "description": "Output format: 'html' (recommended, self-contained with embedded CSS), "
                              "'latex' (raw .tex source), or 'pdf' (requires pdflatex installed). "
                              "HTML is always available; PDF requires a LaTeX compiler on the system."},
}

_COMMON_RETURNS = {
    "status": "success or error.",
    "analysis_type": "Type of analysis performed.",
    "output_path": "Absolute path of the saved calc package file.",
    "file_exists": "True if the file was verified on disk after writing. "
                   "Trust this field — do NOT try to verify the file with "
                   "agent-side filesystem tools (they may be sandboxed).",
    "file_size_bytes": "Size of the saved file (0 if file_exists is false).",
    "rescue_path": "Only on write failure: a verified copy of the package in "
                   "the local temp dir — report this path to the user.",
    "format": "Output format used.",
    "html_length": "Length of generated content.",
}

METHOD_INFO = {
    "bearing_capacity_package": {
        "category": "Calculation Package",
        "brief": "Run bearing capacity analysis and generate Mathcad-style calc package.",
        "parameters": {
            "width": {"type": "float", "required": True, "description": "Footing width B (m)."},
            "unit_weight": {"type": "float", "required": True, "description": "Soil unit weight (kN/m3)."},
            "friction_angle": {"type": "float", "required": False, "default": 0.0, "description": "Friction angle (deg)."},
            "cohesion": {"type": "float", "required": False, "default": 0.0, "description": "Cohesion (kPa)."},
            "depth": {"type": "float", "required": False, "default": 0.0, "description": "Embedment depth Df (m)."},
            "shape": {"type": "str", "required": False, "default": "strip", "allowed_values": ["strip", "square", "rectangular", "circular"], "description": "Footing shape."},
            "length": {"type": "float", "required": False, "description": "Footing length L (m). Required for rectangular."},
            "factor_of_safety": {"type": "float", "required": False, "default": 3.0, "description": "Factor of safety."},
            "ngamma_method": {"type": "str", "required": False, "default": "vesic", "allowed_values": ["vesic", "meyerhof", "hansen"], "description": "Ngamma factor method."},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "q_ultimate_kPa": "Ultimate bearing capacity.", "q_allowable_kPa": "Allowable bearing capacity."},
    },
    "lateral_pile_package": {
        "category": "Calculation Package",
        "brief": "Run lateral pile analysis and generate LPILE-style calc package.",
        "parameters": {
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_diameter": {"type": "float", "required": True, "description": "Pile diameter (m)."},
            "pile_E": {"type": "float", "required": True, "description": "Pile Young's modulus (kPa). Steel ~200e6; concrete drilled shafts ~25e6 gross (use a reduced value, e.g. 0.3x, for cracked sections)."},
            "pile_thickness": {"type": "float", "required": False, "description": "Wall thickness (m). Solid if omitted."},
            "moment_of_inertia": {"type": "float", "required": False, "description": "Moment of inertia (m4). Overrides the value computed from diameter/thickness."},
            "soil_layers": {"type": "array", "required": True, "description": "List of dicts: top, bottom, py_model (alias: model), plus model params. py_model one of SoftClayMatlock/StiffClayAboveWT {c, gamma, eps50}; StiffClayBelowWT {c, gamma, eps50, ks}; SoftClayJeanjean {su, gamma, Gmax}; SandReese/SandAPI {phi, gamma, k} where k = initial subgrade modulus (kN/m3); WeakRock {qu, Er}. Same layer schema as lateral_pile.lateral_pile_analysis."},
            "Vt": {"type": "float", "required": False, "default": 0.0, "description": "Lateral load at head (kN)."},
            "Mt": {"type": "float", "required": False, "default": 0.0, "description": "Moment at head (kN-m)."},
            "Q": {"type": "float", "required": False, "default": 0.0, "description": "Axial load at pile top (kN), for P-delta."},
            "head_condition": {"type": "str", "required": False, "default": "free", "allowed_values": ["free", "fixed"], "description": "Pile head condition."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "y_top_mm": "Top deflection (mm).", "max_moment_kNm": "Max moment (kN-m)."},
    },
    "slope_stability_package": {
        "category": "Calculation Package",
        "brief": "Run slope stability analysis and generate calc package with cross-section plot.",
        "parameters": {
            "surface_points": {"type": "array", "required": True, "description": "List of [x, z] ground surface coordinates."},
            "soil_layers": {"type": "array", "required": True, "description": "List of dicts: name, top_elevation, bottom_elevation, gamma, phi, c_prime (or cu + analysis_mode)."},
            "xc": {"type": "float", "required": True, "description": "Circle center x (m)."},
            "yc": {"type": "float", "required": True, "description": "Circle center y (m)."},
            "radius": {"type": "float", "required": True, "description": "Circle radius (m)."},
            "method": {"type": "str", "required": False, "default": "bishop", "allowed_values": ["bishop", "fellenius", "spencer"], "description": "Limit-equilibrium method."},
            "n_slices": {"type": "int", "required": False, "default": 30, "description": "Number of slices."},
            "gwt_points": {"type": "array", "required": False, "description": "List of [x, z] GWT coordinates."},
            "kh": {"type": "float", "required": False, "default": 0.0, "description": "Seismic coefficient."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "FOS": "Factor of safety.", "method": "Method used.", "is_stable": "Whether FOS >= required."},
    },
    "slope_report_package": {
        "category": "Calculation Package",
        "brief": "One-call slope stability REPORT for a critical-surface SEARCH: input echo, search story (surfaces tried + rejection diagnostics + critical surface/FOS), FOS-by-method table, slice-force + interslice/thrust-line figures, reinforcement summary, and a probabilistic annex (FOSM/Monte Carlo) when 'variables' is supplied. No fixed circle needed — the search finds the critical surface.",
        "parameters": {
            "surface_points": {"type": "array", "required": True, "description": "List of [x, z] ground surface coordinates."},
            "soil_layers": {"type": "array", "required": True, "description": "List of dicts: name, top_elevation, bottom_elevation, gamma, phi, c_prime (or cu + analysis_mode). Optional 'bottom_boundary_points' ([[x,z],...]) defines a non-horizontal layer base (e.g. a weak seam)."},
            "method": {"type": "str", "required": False, "default": "spencer", "allowed_values": ["spencer", "morgenstern_price", "gle", "bishop", "janbu", "fellenius"], "description": "Limit-equilibrium method the search minimizes and the report highlights. Default 'spencer' (rigorous — populates the interslice-force / line-of-thrust figure)."},
            "surface_type": {"type": "str", "required": False, "default": "circular", "allowed_values": ["circular", "entry_exit", "noncircular", "noncircular_de", "pso", "weak_layer"], "description": "Trial-surface family for the search. 'circular' uses the x_range/y_range grid; the others use x_entry_range/x_exit_range. 'weak_layer' biases trials toward a defined weak seam."},
            "x_range": {"type": "array", "required": False, "description": "[min, max] circle-center x for the search grid (circular)."},
            "y_range": {"type": "array", "required": False, "description": "[min, max] circle-center y for the search grid (circular)."},
            "nx": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in x (circular search)."},
            "ny": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in y (circular search)."},
            "x_entry_range": {"type": "array", "required": False, "description": "[min, max] entry x for entry-exit / noncircular searches."},
            "x_exit_range": {"type": "array", "required": False, "description": "[min, max] exit x for entry-exit / noncircular searches."},
            "n_trials": {"type": "int", "required": False, "default": 500, "description": "Number of random trials (noncircular searches)."},
            "n_points": {"type": "int", "required": False, "default": 5, "description": "Vertices per noncircular trial surface."},
            "n_slices": {"type": "int", "required": False, "default": 30, "description": "Number of slices."},
            "gwt_points": {"type": "array", "required": False, "description": "List of [x, z] GWT coordinates."},
            "kh": {"type": "float", "required": False, "default": 0.0, "description": "Seismic coefficient (pseudo-static)."},
            "seed": {"type": "int", "required": False, "description": "Random seed for the search / Monte Carlo (reproducibility)."},
            "variables": {"type": "dict", "required": False, "description": "Probabilistic variable spec {param[:layer]: {mean?, cov|std, dist?, source?}}. When given, adds a reliability annex (FOSM by default) on the critical surface."},
            "monte_carlo": {"type": "bool", "required": False, "default": False, "description": "If True (and 'variables' given), also run a Monte Carlo simulation on the critical surface for the annex."},
            "n": {"type": "int", "required": False, "default": 1000, "description": "Monte Carlo realizations (when monte_carlo=True)."},
            "research_surface": {"type": "bool", "required": False, "default": False, "description": "Monte Carlo: re-search the critical surface each realization (slower) instead of holding it fixed."},
            "FOS_required": {"type": "float", "required": False, "default": 1.5, "description": "Required factor of safety for the stability check."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "FOS": "Critical factor of safety.", "method": "Method used.", "n_surfaces_evaluated": "Trial surfaces evaluated by the search.", "is_stable": "Whether FOS >= required."},
    },
    "settlement_package": {
        "category": "Calculation Package",
        "brief": "Run settlement analysis and generate calc package.",
        "parameters": {
            "q_applied": {"type": "float", "required": True, "description": "Applied bearing pressure (kPa)."},
            "B": {"type": "float", "required": False, "default": 1.0, "description": "Footing width (m)."},
            "L": {"type": "float", "required": False, "default": 1.0, "description": "Footing length (m)."},
            "Es_immediate": {"type": "float", "required": False, "description": "Elastic modulus for immediate settlement (kPa)."},
            "consolidation_layers": {"type": "array", "required": False, "description": "List of dicts: thickness, depth_to_center, e0, Cc, Cr, sigma_v0, sigma_p."},
            "schmertmann_layers": {"type": "array", "required": False, "description": "List of dicts: depth_top, depth_bottom, Es."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "total_settlement_mm": "Total settlement.", "immediate_mm": "Immediate.", "consolidation_mm": "Consolidation."},
    },
    "axial_pile_package": {
        "category": "Calculation Package",
        "brief": "Run axial pile capacity analysis and generate calc package.",
        "parameters": {
            "pile_area": {"type": "float", "required": True, "description": "Pile cross-section area (m2)."},
            "pile_perimeter": {"type": "float", "required": True, "description": "Pile perimeter (m)."},
            "pile_tip_area": {"type": "float", "required": True, "description": "Pile tip area (m2)."},
            "pile_width": {"type": "float", "required": True, "description": "Pile width/diameter (m)."},
            "pile_length": {"type": "float", "required": True, "description": "Embedded pile length (m)."},
            "soil_layers": {"type": "array", "required": True, "description": "List of dicts: soil_type, thickness, unit_weight, friction_angle, cohesion."},
            "factor_of_safety": {"type": "float", "required": False, "default": 2.5, "description": "Factor of safety."},
            "method": {"type": "str", "required": False, "default": "auto", "allowed_values": ["auto", "beta"], "description": "auto = Nordlund (sand) / Tomlinson (clay); beta = effective-stress method."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "Q_ultimate_kN": "Ultimate capacity.", "Q_allowable_kN": "Allowable capacity."},
    },
    "drilled_shaft_package": {
        "category": "Calculation Package",
        "brief": "Run drilled shaft capacity analysis and generate calc package.",
        "parameters": {
            "diameter": {"type": "float", "required": True, "description": "Shaft diameter (m)."},
            "length": {"type": "float", "required": True, "description": "Shaft length (m)."},
            "soil_layers": {"type": "array", "required": True, "description": "List of dicts: soil_type, thickness, unit_weight, cu, phi, N60, qu, RQD."},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
            "factor_of_safety": {"type": "float", "required": False, "default": 2.5, "description": "Factor of safety."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "Q_ultimate_kN": "Ultimate capacity.", "Q_allowable_kN": "Allowable capacity."},
    },
    "downdrag_package": {
        "category": "Calculation Package",
        "brief": "Run downdrag analysis and generate calc package.",
        "parameters": {
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_diameter": {"type": "float", "required": True, "description": "Pile diameter (m)."},
            "pile_E": {"type": "float", "required": True, "description": "Pile Young's modulus (kPa)."},
            "Q_dead": {"type": "float", "required": True, "description": "Dead load at pile head (kN)."},
            "soil_layers": {"type": "array", "required": True, "description": "List of dicts: soil_type, thickness, unit_weight, cu, phi, settling, alpha, beta, Cc, e0."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "neutral_plane_m": "Neutral plane depth.", "dragload_kN": "Dragload."},
    },
    "seismic_package": {
        "category": "Calculation Package",
        "brief": "Run seismic geotechnical analysis and generate calc package.",
        "parameters": {
            "analysis_type": {"type": "str", "required": True, "allowed_values": ["site_classification", "seismic_earth_pressure", "liquefaction"], "description": "Seismic sub-analysis to run."},
            "vs30": {"type": "float", "required": False, "description": "Avg shear wave velocity (m/s) — site classification."},
            "Ss": {"type": "float", "required": False, "description": "Short-period spectral accel (g) — site classification."},
            "S1": {"type": "float", "required": False, "description": "1-sec spectral accel (g) — site classification."},
            "phi": {"type": "float", "required": False, "description": "Friction angle (deg) — earth pressure."},
            "kh": {"type": "float", "required": False, "description": "Horizontal seismic coefficient — earth pressure."},
            "gamma": {"type": "float", "required": False, "description": "Unit weight (kN/m3) — earth pressure."},
            "H": {"type": "float", "required": False, "description": "Wall height (m) — earth pressure."},
            "amax_g": {"type": "float", "required": False, "description": "Peak ground acceleration (g) — liquefaction."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "seismic_analysis_type": "Which sub-analysis was performed."},
    },
    "retaining_wall_package": {
        "category": "Calculation Package",
        "brief": "Run retaining wall analysis and generate calc package.",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "Wall height H (m)."},
            "gamma_backfill": {"type": "float", "required": True, "description": "Backfill unit weight (kN/m3)."},
            "phi_backfill": {"type": "float", "required": True, "description": "Backfill friction angle (deg)."},
            "wall_type": {"type": "str", "required": False, "default": "cantilever", "allowed_values": ["cantilever"], "description": "Wall type (MSE not yet supported)."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surcharge (kPa)."},
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb"], "description": "Earth pressure theory."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "FOS_sliding": "Sliding FOS.", "FOS_overturning": "Overturning FOS."},
    },
    "ground_improvement_package": {
        "category": "Calculation Package",
        "brief": "Run ground improvement analysis and generate calc package.",
        "parameters": {
            "method": {"type": "str", "required": True, "allowed_values": ["wick_drains", "aggregate_piers", "surcharge", "vibro"], "description": "Ground improvement method."},
            "spacing": {"type": "float", "required": False, "description": "Element spacing (m) — wick drains, agg piers, vibro."},
            "ch": {"type": "float", "required": False, "description": "Horizontal cv (m2/yr) — wick drains."},
            "cv": {"type": "float", "required": False, "description": "Vertical cv (m2/yr) — wick drains, surcharge."},
            "Hdr": {"type": "float", "required": False, "description": "Drainage path (m) — wick drains."},
            "time": {"type": "float", "required": False, "description": "Design time (years) — wick drains."},
            "diameter": {"type": "float", "required": False, "description": "Pier diameter (m) — aggregate piers."},
            "length": {"type": "float", "required": False, "description": "Pier length (m) — aggregate piers."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "method": "Ground improvement method used."},
    },
    "wave_equation_package": {
        "category": "Calculation Package",
        "brief": "Run wave equation analysis and generate bearing graph calc package.",
        "parameters": {
            "hammer_name": {"type": "str", "required": True, "description": "Hammer designation string."},
            "cushion_area": {"type": "float", "required": True, "description": "Cushion area (m2)."},
            "cushion_thickness": {"type": "float", "required": True, "description": "Cushion thickness (m)."},
            "cushion_E": {"type": "float", "required": True, "description": "Cushion elastic modulus (Pa)."},
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_area": {"type": "float", "required": True, "description": "Pile cross-section area (m2)."},
            "pile_E": {"type": "float", "required": True, "description": "Pile elastic modulus (Pa)."},
            "skin_fraction": {"type": "float", "required": False, "default": 0.5, "description": "Fraction of resistance from skin."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "n_points": "Number of bearing graph points."},
    },
    "pile_group_package": {
        "category": "Calculation Package",
        "brief": "Run pile group analysis and generate calc package.",
        "parameters": {
            "layout": {"type": "str", "required": False, "allowed_values": ["rectangular"], "description": "'rectangular' for auto-layout, or provide 'piles' list instead."},
            "n_rows": {"type": "int", "required": False, "description": "Rows — rectangular layout."},
            "n_cols": {"type": "int", "required": False, "description": "Columns — rectangular layout."},
            "spacing_x": {"type": "float", "required": False, "description": "X spacing (m) — rectangular."},
            "spacing_y": {"type": "float", "required": False, "description": "Y spacing (m) — rectangular."},
            "piles": {"type": "array", "required": False, "description": "List of dicts: x, y, label — custom layout."},
            "Vz": {"type": "float", "required": False, "default": 0.0, "description": "Vertical load (kN)."},
            "Mx": {"type": "float", "required": False, "default": 0.0, "description": "Moment about x (kN-m)."},
            "My": {"type": "float", "required": False, "default": 0.0, "description": "Moment about y (kN-m)."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "n_piles": "Number of piles.", "group_efficiency": "Group efficiency."},
    },
    "sheet_pile_package": {
        "category": "Calculation Package",
        "brief": "Run sheet pile wall analysis and generate calc package.",
        "parameters": {
            "excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."},
            "soil_layers": {"type": "array", "required": True, "description": "List of dicts: thickness, unit_weight, friction_angle, cohesion."},
            "wall_type": {"type": "str", "required": False, "default": "cantilever", "allowed_values": ["cantilever", "anchored"], "description": "Wall type."},
            "anchor_depth": {"type": "float", "required": False, "description": "Anchor depth (m) — anchored walls."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge (kPa)."},
            "FOS_passive": {"type": "float", "required": False, "default": 1.5, "description": "FOS on passive resistance."},
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb"], "description": "Earth pressure theory."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "embedment_m": "Required embedment.", "max_moment_kNm_per_m": "Max moment."},
    },
}
