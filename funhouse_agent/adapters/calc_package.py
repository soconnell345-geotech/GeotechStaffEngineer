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

import base64
import contextvars
import os
import re
from datetime import datetime

from funhouse_agent.adapters import (apply_aliases, check_nested_list,
                                     declare_aliases, declare_applies_when,
                                     require_keys, require_params,
                                     reject_unknown_params)
from funhouse_agent._fileio import (
    default_output_dir, find_in_working_folder, rescue_note, rescue_write,
    resolve_output_path, save_verified, workspace_write_hint,
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


#: Figure-only mode for the CURRENT ``render_figures`` call: ``None`` normally,
#: else ``{"output_dir": ..., "name_prefix": ...}``. It rides a context var
#: rather than the params dict because a package handler validates its params
#: strictly (``reject_unknown_params``) and forwards them as ``**kwargs`` into
#: the module function — a router-injected key would be rejected as unknown
#: (slope_report_package) or arrive as an unexpected kwarg (pavement_design_
#: package). Keeping it out of band means every package, present and future,
#: renders figures without knowing this mode exists.
_FIGURE_MODE = contextvars.ContextVar("calc_package_figure_mode", default=None)

#: Packages that cannot be rendered figures-only (no analysis behind them).
_NO_FIGURES = {"html_to_pdf", "render_figures"}


def _slug(text: str, n: int = 40) -> str:
    import re
    t = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_").lower()
    return (t or "figure")[:n].strip("_") or "figure"


#: Longest full path a generated figure is given. Windows refuses a path over
#: 260 characters unless long paths are enabled, and a conversation folder is
#: already deep; a 266-character figure path failed to save in live smoke
#: wave 1 (A14).
_MAX_FIGURE_PATH = 220


def _figure_file_name(out_dir: str, stem: str, index: int, title: str,
                      ts: str) -> str:
    """``<stem>_f<i>_<title slug>_<HHMMSS>.png``, short: the title slug is
    cut, then dropped, to keep the full path under :data:`_MAX_FIGURE_PATH`."""
    room = _MAX_FIGURE_PATH - len(os.path.abspath(out_dir)) - 1
    base = f"{stem}_f{index}"
    tail = f"_{ts}.png"
    slug = _slug(title, 24)
    for name in (f"{base}_{slug}{tail}", f"{base}{tail}"):
        if len(name) <= room:
            return name
    return f"f{index}{tail}"


def _figures_response(module: str, result, analysis, opts: dict,
                      analysis_type: str, extra: dict = None) -> dict:
    """Save the module's calc-package figures as standalone PNGs.

    ``opts`` is the ``_FIGURE_MODE`` payload (``output_dir``/``name_prefix``),
    not the package's own params.

    Owner feedback 2026-09-11: ~25 ready matplotlib figures (slope sections,
    trial-surface maps, p-y curves, settlement plots, pavement charts, FEM
    contours) were reachable ONLY inside a whole canned package. A bespoke
    ``html_to_pdf`` report — SOE, a multi-analysis narrative, anything with
    no canned template — could not carry them. Now any package's figures can
    be rendered on their own and pasted in via ``html_img_tag``.
    """
    import base64

    from calc_package import _ensure_registered

    reg = _ensure_registered(module)
    figures = reg["get_figures"](result, analysis) or []
    out_dir = opts.get("output_dir")
    out_dir = (resolve_output_path(str(out_dir)) if out_dir
               else default_output_dir())
    os.makedirs(out_dir, exist_ok=True)
    stem = _slug(opts.get("name_prefix") or module, 24)
    ts = datetime.now().strftime("%H%M%S")

    saved_figs = []
    problems = []
    for i, fig in enumerate(figures, start=1):
        title = getattr(fig, "title", "") or f"Figure {i}"
        b64 = getattr(fig, "image_base64", "") or ""
        try:
            png = base64.b64decode(b64)
        except Exception as exc:                       # noqa: BLE001
            problems.append(f"figure {i} ('{title}'): undecodable image "
                            f"({type(exc).__name__})")
            continue
        path = os.path.join(out_dir, _figure_file_name(out_dir, stem, i,
                                                       title, ts))
        saved = save_verified(path, png)
        abs_path = saved.get("saved", os.path.abspath(path))
        entry = {
            "index": i,
            "title": title,
            "caption": getattr(fig, "caption", "") or "",
            "output_path": abs_path,
            "file_exists": bool(saved.get("file_exists")),
            "file_size_bytes": saved.get("file_size_bytes", 0),
            "html_img_tag": (
                f'<img src="{abs_path}" alt="{title}" '
                f'style="width:100%;max-width:640px;">'),
        }
        for key in ("error", "rescue_path", "workspace_api_note"):
            if saved.get(key):
                entry[key] = saved[key]
        if not entry["file_exists"]:
            problems.append(f"figure {i} ('{title}'): {saved.get('error')}")
        saved_figs.append(entry)

    response = {
        "status": "success" if not problems else "error",
        "analysis_type": f"{analysis_type} — figures only",
        "module": module,
        "output_dir": os.path.abspath(out_dir),
        "n_figures": len(saved_figs),
        "figures": saved_figs,
        "embed_note": (
            "Each figure is a saved PNG. The web app shows it under your reply "
            "(files written outside the conversation folder are copied in at "
            "the end of the turn) -- do not put markdown image links to local "
            "paths in the reply. html_to_pdf embeds the real local path when "
            "you paste its html_img_tag into report HTML. Use the caption text "
            "under it."),
    }
    if not figures:
        response["note"] = (f"{reg.get('display_name', module)} produces no "
                            "figures for this analysis.")
    if problems:
        response["error"] = "; ".join(problems)
    if extra:
        response.update(extra)
    return response


def _render_figures(params: dict) -> dict:
    """``render_figures``: a canned package's figures as standalone PNGs.

    ``params`` = ``{"package": "<name>_package", ...that package's inputs,
    optional "output_dir", "name_prefix"}``. The analysis runs exactly as the
    package would run it; only the figures are written.
    """
    p = dict(params)
    package = p.pop("package", None) or p.pop("method", None)
    if not package:
        return {"status": "error",
                "error": "Provide 'package' — the *_package method whose "
                         "figures you want, e.g. 'slope_stability_package', "
                         "plus that package's own inputs. Options: "
                         + ", ".join(sorted(k for k in METHOD_REGISTRY
                                            if k not in _NO_FIGURES))}
    if package in _NO_FIGURES or package not in METHOD_REGISTRY:
        return {"status": "error",
                "error": f"'{package}' is not a renderable package. Options: "
                         + ", ".join(sorted(k for k in METHOD_REGISTRY
                                            if k not in _NO_FIGURES))}
    p.pop("format", None)          # meaningless without a package
    p.pop("output_path", None)     # figures go to output_dir, one PNG each
    mode = {"output_dir": p.pop("output_dir", None),
            "name_prefix": p.pop("name_prefix", None)}
    token = _FIGURE_MODE.set(mode)
    try:
        return METHOD_REGISTRY[package](p)
    finally:
        _FIGURE_MODE.reset(token)


def _build_response(module: str, result, analysis, params: dict,
                    analysis_type: str, extra: dict = None) -> dict:
    """Run generate_calc_package and return a summary response dict.

    Always saves to disk.  Auto-generates output_path if not provided.

    Inside a ``render_figures`` call (``_FIGURE_MODE`` set), the package is NOT
    generated: the module's ``get_figures`` runs on the same result and each
    figure is saved as a standalone PNG instead.
    """
    from calc_package import generate_calc_package

    figure_mode = _FIGURE_MODE.get()
    if figure_mode is not None:
        return _figures_response(module, result, analysis, figure_mode,
                                 analysis_type=analysis_type, extra=extra)

    meta = _extract_metadata(params)
    fmt = params.get("format", "html")
    output_path = params.get("output_path") or _default_output_path(module, fmt)

    # Calculation tables (GM-39): built here so each CSV is written with a
    # verified save, like the package itself. A table that fails to build
    # costs the tables, never the package.
    from calc_package import package_tables
    from calc_package.tables import csv_file_names
    table_dir = os.path.dirname(os.path.abspath(output_path))
    try:
        calc_tables = package_tables(module, result, analysis)
        table_error = None
    except Exception as exc:                               # noqa: BLE001
        calc_tables = []
        table_error = (f"the calculation tables could not be built "
                       f"({type(exc).__name__}: {exc}); the package has none")
    table_names = csv_file_names(
        calc_tables, os.path.splitext(os.path.basename(output_path))[0],
        table_dir)

    render_report = {}
    content = generate_calc_package(
        module=module,
        result=result,
        analysis=analysis,
        output_path=output_path,
        format=fmt,
        render_report=render_report,
        tables=calc_tables,
        table_files=table_names,
        **meta,
    )
    tables_out = _save_tables(calc_tables, table_names, table_dir)

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
            response["error"] += rescue_note(rescue)
    if tables_out:
        response["tables"] = tables_out
        response["tables_note"] = _TABLES_NOTE
        failed = [t["title"] for t in tables_out if not t["file_exists"]]
        if failed:
            table_error = ("these table CSVs did not save: "
                           + "; ".join(failed))
    if table_error:
        response["tables_error"] = table_error
    if extra:
        response.update(extra)
    return response


#: Said once beside the table list of every package response.
_TABLES_NOTE = (
    "Each calculation table is saved as a CSV beside the package (the input "
    "echo is the *_inputs.csv) and printed in the package's Calculation "
    "Tables section. They show as downloads under your reply; name them to "
    "the user.")


def _save_tables(tables, names, out_dir) -> list:
    """Write each table's CSV with a verified save; one entry per table:
    ``{title, output_path, rows, file_exists[, error, rescue_path]}``."""
    from calc_package.tables import csv_bytes

    out = []
    for t, name in zip(tables, names):
        path = os.path.join(out_dir, name)
        saved = save_verified(path, csv_bytes(t))
        entry = {
            "title": t.title,
            "output_path": saved.get("saved", os.path.abspath(path)),
            "rows": t.n_rows,
            "file_exists": bool(saved.get("file_exists")),
        }
        for key in ("error", "rescue_path"):
            if saved.get(key):
                entry[key] = saved[key]
        out.append(entry)
    return out


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
        ngamma_method=params.get("ngamma_method"),
        factor_method=params.get("factor_method", "vesic"),
        failure_mode=params.get("failure_mode", "general"),
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
    from funhouse_agent.adapters.lateral_pile import (
        _build_py_model, check_py_layers,
    )

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

    layers, dicts = [], []
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
        dicts.append(d)
    check_py_layers(dicts, method="lateral_pile_package",
                    label="soil_layers")
    for d in dicts:
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
    """Build a SlopeGeometry from flat params. Shared by the slope packages.

    Each layer is read by the slope adapter's `build_soil_layer` (one source
    of truth for the layer keys): an unknown key is refused with a
    did-you-mean instead of being dropped (a layer given 'c' ran with
    c' = 0), and ru, strength_model and every SHANSEP / Hoek-Brown /
    anisotropic / R-envelope field pass through."""
    from slope_stability.geometry import SlopeGeometry
    from funhouse_agent.adapters.slope_stability import build_soil_layer

    soil_layers = [
        build_soil_layer(ld, i, method=method,
                         required=("name", "top_elevation", "bottom_elevation",
                                   "gamma"))
        for i, ld in enumerate(params["soil_layers"])]

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
            "seed", "f_interslice", "FOS_required", "time_budget_s",
            # probabilistic annex
            "variables", "fosm", "monte_carlo", "n", "research_surface",
            # output + header metadata
            "output_path", "format", "project_name", "project_number",
            "engineer", "checker", "company", "date",
        ],
        method="slope_report_package",
    )

    geom = _build_slope_geom(params, "slope_report_package")
    from funhouse_agent.adapters.slope_stability import (
        budget_is_default, check_variable_specs, _time_budget,
    )
    check_variable_specs(params, method="slope_report_package")
    # Every search has the agent's time budget (GM-123), per search.
    time_budget_s = _time_budget(params, "slope_report_package")

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
        time_budget_s=time_budget_s,
    )
    budget_is_default(search.choices, params)

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
                search_kwargs=({"time_budget_s": time_budget_s}
                               if params.get("research_surface") else None),
                **prob_surf)

    extra = {
        "FOS": round(result.FOS, 3),
        "method": result.method,
        "n_surfaces_evaluated": search.n_surfaces_evaluated,
        "is_stable": result.FOS >= analysis_dict.get("FOS_required", 1.5),
    }
    # Which kind of surface governs and what the search could not be sure
    # of -- also printed in the package's search section.
    if search.governing_surface:
        extra["governing_surface"] = search.governing_surface
        extra["candidates"] = dict(search.candidates)
    if search.warnings:
        extra["warnings"] = list(search.warnings)
    return _build_response(
        "slope_stability", result, analysis_dict, params,
        analysis_type="Slope Stability Report",
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 4. Settlement
# ---------------------------------------------------------------------------

def _generate_settlement_package(params: dict) -> dict:
    from settlement import SettlementAnalysis, ConsolidationLayer, SchmertmannLayer

    from funhouse_agent.adapters.settlement import (
        _CONSOLIDATION_KEY_HINTS, _CONSOLIDATION_KEYS,
        _SCHMERTMANN_KEY_HINTS, _SCHMERTMANN_KEYS,
    )
    check_nested_list(params.get("consolidation_layers"),
                      _CONSOLIDATION_KEYS, method="settlement_package",
                      label="consolidation_layers",
                      hints=_CONSOLIDATION_KEY_HINTS)
    check_nested_list(params.get("schmertmann_layers"), _SCHMERTMANN_KEYS,
                      method="settlement_package", label="schmertmann_layers",
                      hints=_SCHMERTMANN_KEY_HINTS)

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

    from funhouse_agent.adapters.axial_pile import (
        _LAYER_KEY_HINTS, _LAYER_KEYS,
    )
    # The analysis tool's layer keys, all read below; anything else was
    # silently left out (GM-58).
    check_nested_list(params["soil_layers"], _LAYER_KEYS,
                      method="axial_pile_package", label="soil_layers",
                      hints=_LAYER_KEY_HINTS)
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
            # the end-bearing phi, as the analysis tool reads it ('toe_phi'
            # its alias); unset, the toe uses friction_angle
            toe_friction_angle=ld.get("toe_friction_angle", ld.get("toe_phi")),
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

    from funhouse_agent.adapters.drilled_shaft import _LAYER_KEY_HINTS
    # The keys read below; anything else was silently left out (GM-58). The
    # package runs the default methods only, so the rational-beta inputs
    # are refused with a note saying where they are read, never offered as
    # N60 (a different quantity).
    rational = ("N1_60", "N160", "sigma_v_ref", "sigma_v_preload", "OCR")
    note = ("(This package runs the default beta method, beta_method="
            "'depth'; N1_60 / N160, sigma_v_ref / sigma_v_preload and OCR "
            "are read by drilled_shaft.drilled_shaft_capacity with "
            "beta_method='rational'.)")
    check_nested_list(params["soil_layers"],
                      ("soil_type", "thickness", "unit_weight", "cu", "phi",
                       "N60", "qu", "RQD", "description"),
                      method="drilled_shaft_package", label="soil_layers",
                      hints={**_LAYER_KEY_HINTS, "n160": (),
                             "sigmavref": (), "sigmavpreload": (), "ocr": ()},
                      notes={(k,): note for k in rational})
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
    # The analysis tool's layer keys (DowndragSoilLayer's fields); anything
    # else was silently left out (GM-58).
    from funhouse_agent.adapters.downdrag import (
        _LAYER_KEY_HINTS, _LAYER_KEY_NOTES, _LAYER_KEYS,
    )
    check_nested_list(params["soil_layers"], _LAYER_KEYS,
                      method="downdrag_package", label="soil_layers",
                      hints=_LAYER_KEY_HINTS, notes=_LAYER_KEY_NOTES)

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
            Cc=ld.get("Cc", 0.0),
            Cr=ld.get("Cr", 0.0),
            e0=ld.get("e0", 0.0),
            C_ec=ld.get("C_ec"),
            C_er=ld.get("C_er"),
            sigma_p=ld.get("sigma_p"),
            E_s=ld.get("E_s"),
            nu_s=ld.get("nu_s", 0.3),
            description=ld.get("description", ""),
        ))
    soil = DowndragSoilProfile(layers=layers, gwt_depth=params.get("gwt_depth", 0.0))

    analysis = DowndragAnalysis(
        pile_length=params["pile_length"],
        pile_diameter=params["pile_diameter"],
        # an H or pipe pile's own steel area and perimeter (GM-99); the
        # solid circle of pile_diameter when not given
        pile_perimeter=params.get("pile_perimeter"),
        pile_area=params.get("pile_area"),
        pile_E=params["pile_E"],
        pile_unit_weight=params.get("pile_unit_weight", 78.5),
        include_pile_weight=params.get("include_pile_weight", False),
        Q_dead=params["Q_dead"],
        soil=soil,
        fill_thickness=params.get("fill_thickness", 0.0),
        fill_unit_weight=params.get("fill_unit_weight", 20.0),
        gw_drawdown=params.get("gw_drawdown", 0.0),
        structural_capacity=params.get("structural_capacity"),
        allowable_settlement=params.get("allowable_settlement"),
        Nt=params.get("Nt"),
        neutral_plane_method=params.get("neutral_plane_method", "auto"),
        toe_mobilization=params.get("toe_mobilization", 1.0),
        endo_bearing_condition=params.get("endo_bearing_condition"),
        skin_friction_stress=params.get("skin_friction_stress", "final"),
        influence_depth_criterion=params.get("influence_depth_criterion",
                                             "bearing_stress"),
    )
    result = analysis.compute()

    extra = {
        "neutral_plane_m": round(result.neutral_plane_depth, 2),
        "neutral_plane_method": result.neutral_plane_method,
        "headline_method": result.headline_method,
        "neutral_plane_basis": result.neutral_plane_basis,
        "methods_range": result.methods_range,
        "skin_friction_stress": result.skin_friction_stress,
        "include_pile_weight": result.include_pile_weight,
        "pile_weight_basis": result.pile_weight_basis,
        "below_toe_basis": result.below_toe.get("basis", ""),
    }
    return _build_response(
        "downdrag", result, analysis, params,
        analysis_type="Downdrag",
        extra={
            **extra,
            "warnings": list(result.warnings),
            "neutral_plane_comparison": result.comparison_rows(),
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
        from seismic_geotech.mononobe_okabe import (
            earth_pressure_choices, mo_no_solution_warning,
        )
        no_solution = mo_no_solution_warning(params["phi"], params["kh"],
                                             params.get("kv", 0.0))
        result = SeismicEarthPressureResult(
            KAE=KAE, KPE=0.0,
            PAE_total=result_dict["PAE_total_kN_per_m"],
            PA_static=result_dict["PA_static_kN_per_m"],
            delta_PAE=result_dict["delta_PAE_kN_per_m"],
            height_of_application=result_dict["height_of_application_m"],
            phi=params["phi"],
            delta=params.get("delta", params["phi"] * 2 / 3),
            kh=params["kh"], kv=params.get("kv", 0.0),
            choices=earth_pressure_choices(
                params["phi"], params["kh"], kv=params.get("kv"),
                delta_deg=params.get("delta"),
                ka=params.get("KA", 0.333),
                ka_how="given" if "KA" in params else "default",
                forces=True),
            warnings=[no_solution] if no_solution else [],
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
        from seismic_geotech.liquefaction import liquefaction_choices
        result = LiquefactionResult(
            layer_results=layer_results,
            amax_g=params["amax_g"],
            magnitude=params.get("M", 7.5),
            gwt_depth=params["gwt_depth"],
            choices=liquefaction_choices(layer_results, params.get("M", 7.5)),
        )
    else:
        raise ValueError(
            f"Unknown analysis_type '{analysis_type}'. Allowed: "
            "['site_classification', 'seismic_earth_pressure', 'liquefaction']."
        )

    extra = {"seismic_analysis_type": analysis_type}
    if getattr(result, "warnings", None):
        extra["warnings"] = list(result.warnings)
    return _build_response(
        "seismic_geotech", result, analysis_dict, params,
        analysis_type="Seismic Geotechnical",
        extra=extra,
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
            has_shear_key=params.get("has_shear_key", False),
            key_depth=params.get("key_depth", 0.0),
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
            include_passive=params.get("include_passive", False),
            gamma_foundation=params.get("gamma_foundation"),
            delta_base=params.get("delta_base"),
            base_adhesion=params.get("base_adhesion"),
        )
        analysis_dict["geom"] = geom

        def _r2(x):
            return None if x is None else round(x, 2)

        # The verdict travels with the package (GM-100): a failed check or a
        # check not run must reach the user, not only the file.
        extra = {
            "verdict": result.verdict,
            "checks_failed": result.checks_failed,
            "checks_not_run": result.checks_not_run,
            "FOS_sliding": _r2(result.FOS_sliding),
            "FOS_overturning": _r2(result.FOS_overturning),
            "FOS_bearing": _r2(result.FOS_bearing),
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
        # Aliases from the pre-GEC-13 parameterization this package once used.
        p = apply_aliases(params, {"diameter": "column_diameter",
                                   "E_pier": "E_column"})
        require_params(p, ["column_diameter", "spacing"],
                       method="ground_improvement_package (aggregate_piers)")
        result = analyze_aggregate_piers(
            column_diameter=p["column_diameter"],
            spacing=p["spacing"],
            pattern=p.get("pattern", "triangular"),
            E_column=p.get("E_column", 80000.0),
            E_soil=p.get("E_soil", 5000.0),
            n=p.get("n", 5.0),
            q_unreinforced=p.get("q_unreinforced", 0.0),
            S_unreinforced=p.get("S_unreinforced", 0.0),
        )
    elif method == "surcharge":
        p = apply_aliases(params, {"surcharge": "surcharge_kPa",
                                   "target_consolidation": "target_U"})
        require_params(p, ["S_ultimate", "surcharge_kPa", "cv", "Hdr"],
                       method="ground_improvement_package (surcharge)")
        result = analyze_surcharge_preloading(
            S_ultimate=p["S_ultimate"],
            surcharge_kPa=p["surcharge_kPa"],
            cv=p["cv"], Hdr=p["Hdr"],
            target_U=p.get("target_U", 90.0),
            ch=p.get("ch"),
            drain_spacing=p.get("drain_spacing"),
            dw=p.get("dw", 0.066),
            pattern=p.get("pattern", "triangular"),
            smear_ratio=p.get("smear_ratio", 2.0),
            kh_ks_ratio=p.get("kh_ks_ratio", 2.0),
            sigma_v0=p.get("sigma_v0", 0.0),
        )
    elif method == "vibro":
        p = apply_aliases(params, {"N1_before": "initial_N_spt",
                                   "N1_target": "target_N_spt"})
        require_params(p, ["fines_content", "initial_N_spt"],
                       method="ground_improvement_package (vibro)")
        result = analyze_vibro_compaction(
            fines_content=p["fines_content"],
            initial_N_spt=p["initial_N_spt"],
            target_N_spt=p.get("target_N_spt", 25.0),
            D50=p.get("D50"),
            pattern=p.get("pattern", "triangular"),
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
    from pile_group import analyze_vertical_group_simple, analyze_group_6dof
    # The pile_group module's own layout and load readers, so the package
    # takes the batter, stiffnesses and capacities its methods take (until
    # 2026-10-14 it dropped the stiffnesses and method_6dof always failed).
    from funhouse_agent.adapters.pile_group import _build_load, _build_piles

    if params.get("layout") == "rectangular":
        require_params(params, ["n_rows", "n_cols", "spacing_x", "spacing_y"],
                       method="pile_group_package (layout='rectangular')")
    elif not params.get("piles"):
        raise ValueError(
            "pile_group_package: provide either layout='rectangular' with "
            "n_rows/n_cols/spacing_x/spacing_y, or a 'piles' list of "
            "{x, y} dicts."
        )
    # The simplified vertical distribution carries no lateral load or
    # torsion; refused rather than dropped with only a Python warning
    # (GM-110).
    lateral = [k for k in ("Vx", "Vy", "Mz") if params.get(k)]
    if lateral and not params.get("method_6dof", False):
        raise ValueError(
            f"pile_group_package: {lateral} given without method_6dof. The "
            "simplified vertical distribution carries only Vz, Mx and My; "
            "set method_6dof=true (with axial_stiffness, and "
            "lateral_stiffness for Vx/Vy) to carry them.")
    piles = _build_piles(params)
    load = _build_load(params)

    if params.get("method_6dof", False):
        missing = [p.label or f"#{i}" for i, p in enumerate(piles)
                   if p.axial_stiffness is None]
        if missing:
            raise ValueError(
                "pile_group_package (method_6dof): every pile needs an "
                f"axial_stiffness (kN/m); missing on {missing[:6]}"
                f"{' ...' if len(missing) > 6 else ''}. Give "
                "'axial_stiffness' at the top level with layout='rectangular' "
                "or on each piles[] entry, and 'lateral_stiffness' too when "
                "vertical piles carry Vx/Vy.")
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
    from sheet_pile import analyze_cantilever, analyze_anchored

    require_params(params, ["excavation_depth", "soil_layers"],
                   method="sheet_pile_package")
    if params.get("wall_type") == "anchored":
        require_params(params, ["anchor_depth"],
                       method="sheet_pile_package (wall_type='anchored')")

    # The analysis tool's own layer reader: the same keys and aliases
    # (phi, gamma, c, delta, ...), unknown keys refused (GM-58), and the
    # wall friction passed through for pressure_method='coulomb'.
    from funhouse_agent.adapters.sheet_pile import _build_soil_layers
    layers = _build_soil_layers(params, method="sheet_pile_package",
                                key="soil_layers")

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
# Generic HTML -> PDF report render (owner wall session 2026-07-14: no canned
# package matched an atypical wall, and the agent had no way to turn its own
# composed report into a PDF)
# ---------------------------------------------------------------------------

# --- figure embedding -------------------------------------------------------
# Field feedback 2026-09-04 (a downdrag session): the model shipped a report PDF
# containing a literal "[image]" placeholder, because the Story renderer drops
# anything it cannot embed WITHOUT complaining. So: inline what we can (a real
# local PNG/JPEG path), and refuse loudly for the rest.

_IMG_TAG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_SRC_RE = re.compile(
    r"""\bsrc\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE)
_SVG_RE = re.compile(r"<svg\b[^>]*>", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(
    r"\[\s*(?:insert\s+|add\s+|see\s+)?"
    r"(?:images?|figures?|fig|charts?|plots?|diagrams?|graphics?|photos?|"
    r"screenshots?)"
    r"(?:\s+(?:placeholder|here|omitted|goes\s+here|to\s+be\s+inserted|"
    r"not\s+available|tbd|missing))?\s*\]",
    re.IGNORECASE)
_EMBEDDABLE_URI_RE = re.compile(r"\s*data:image/(?:png|jpe?g);base64,",
                                re.IGNORECASE)
_DATA_URI_RE = re.compile(r"\s*data:", re.IGNORECASE)
_REMOTE_RE = re.compile(r"\s*(?:https?|ftp|s3|gs)://", re.IGNORECASE)
_EMBEDDABLE_EXT = (".png", ".jpg", ".jpeg")
#: Refuse to inline a single image bigger than this (bytes).
_MAX_INLINE_BYTES = 12 * 1024 * 1024

_EMBED_FIX = (
    "Fix: render each figure to a PNG file, then reference it either as "
    '<img src="fig.png"> (the name a figure tool returned: a PNG/JPEG in the '
    'working folder, or a real local path, is read and embedded for you) or '
    'as a base64 data URI (<img '
    'src="data:image/png;base64,...">). For a layered subsurface profile call '
    "call_agent('profile_figure', 'subsurface_profile', {...}) — it saves the "
    "PNG and hands back a ready-to-paste html_img_tag. Every other analysis "
    "plot tool takes an output_path. Pass allow_unembeddable=true to render "
    "anyway, accepting that those figures will be MISSING from the PDF.")


def _resolve_image_path(src: str, base_dir: str = None) -> str:
    """Local filesystem path for an <img> src (file: URI or plain path).

    A relative path is tried next to the source HTML file first (``base_dir``,
    when the caller passed ``html_path``), then against the working folder.
    """
    path = src.strip()
    if path.lower().startswith("file:"):
        # file:///tmp/x.png -> /tmp/x.png; file:///C:/x.png -> C:/x.png. Until
        # 2026-09-15 the whole "file:///" was sliced off, which made a Linux
        # path RELATIVE ("tmp/x.png") and sent it to the working folder, so
        # every figure referenced that way was refused (field feedback N7).
        from urllib.parse import unquote
        path = unquote(path[5:])
        if path.startswith("//"):
            path = path[2:]
            if path.lower().startswith("localhost/"):
                path = path[len("localhost"):]
        if (len(path) >= 3 and path[0] == "/" and path[1].isalpha()
                and path[2] == ":"):
            path = path[1:]
    if base_dir and not os.path.isabs(path):
        beside = os.path.abspath(os.path.join(base_dir, path))
        if os.path.isfile(beside):
            return beside
    return os.path.abspath(resolve_output_path(path))


def _embed_local_images(html: str, base_dir: str = None):
    """Inline real local PNG/JPEG ``<img>`` files as base64 data URIs.

    Returns ``(html, inlined, problems)`` — ``inlined`` lists what was embedded,
    ``problems`` describes every ``<img>`` that could NOT be embedded.
    """
    inlined, problems, out, cursor = [], [], [], 0

    for tag_match in _IMG_TAG_RE.finditer(html):
        tag = tag_match.group(0)
        snippet = tag if len(tag) <= 120 else tag[:117] + "..."
        src_match = _SRC_RE.search(tag)
        src = None
        if src_match:
            src = next(g for g in src_match.groups() if g is not None)

        if not src:
            problems.append(f'{snippet} — <img> with no src attribute')
            continue
        if _EMBEDDABLE_URI_RE.match(src):
            continue  # already an embeddable base64 PNG/JPEG
        if _DATA_URI_RE.match(src):
            kind = src.split(";", 1)[0].split(":", 1)[-1] or "unknown"
            problems.append(
                f'{snippet} — data URI of type "{kind}"; the PDF engine embeds '
                "only base64 PNG or JPEG")
            continue
        if _REMOTE_RE.match(src):
            problems.append(
                f'{snippet} — remote image URL; the PDF engine does not fetch '
                "anything over the network")
            continue

        path = _resolve_image_path(src, base_dir)
        ext = os.path.splitext(path)[1].lower()
        if not os.path.isfile(path):
            problems.append(
                f'{snippet} — no such file on the REAL filesystem '
                f'("{path}")')
            continue
        if ext not in _EMBEDDABLE_EXT:
            problems.append(
                f'{snippet} — "{ext or "no extension"}" is not an embeddable '
                "image format (PNG or JPEG only)")
            continue
        size = os.path.getsize(path)
        if size > _MAX_INLINE_BYTES:
            problems.append(
                f"{snippet} — image is {size / 1e6:.1f} MB, over the "
                f"{_MAX_INLINE_BYTES / 1e6:.0f} MB embed limit; re-render it "
                "smaller (e.g. dpi=150)")
            continue

        with open(path, "rb") as fh:
            raw = fh.read()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        data_uri = (f"data:{mime};base64,"
                    + base64.b64encode(raw).decode("ascii"))
        new_tag = _SRC_RE.sub(lambda _m: f'src="{data_uri}"', tag, count=1)

        out.append(html[cursor:tag_match.start()])
        out.append(new_tag)
        cursor = tag_match.end()
        inlined.append({"src": src, "path": path, "bytes": size})

    out.append(html[cursor:])
    return "".join(out), inlined, problems


def _unembeddable_findings(html: str, img_problems) -> list:
    """Everything in the HTML that would silently vanish from the PDF."""
    findings = list(img_problems)
    for match in _SVG_RE.finditer(html):
        tag = match.group(0)
        snippet = tag if len(tag) <= 120 else tag[:117] + "..."
        findings.append(
            f"{snippet} — inline <svg> is NOT rendered by the PDF engine")
    for match in _PLACEHOLDER_RE.finditer(html):
        findings.append(
            f'literal placeholder text "{match.group(0)}" at character '
            f"{match.start()} — a placeholder is not a figure")
    return findings


def _unembeddable_error(findings: list) -> str:
    shown = findings[:12]
    lines = "\n".join(f"  {i}. {f}" for i, f in enumerate(shown, 1))
    more = ("\n  ... and "
            f"{len(findings) - len(shown)} more" if len(findings) > len(shown)
            else "")
    return (f"html_to_pdf: {len(findings)} figure reference(s) in this HTML "
            "cannot be embedded, and the PDF engine drops them WITHOUT "
            "warning — the report would show blank space or a literal "
            "placeholder where a figure belongs:\n" + lines + more + "\n"
            + _EMBED_FIX)


def _pdf_layout(path: str, max_pages: int = 20) -> dict:
    """``{"pages", "page_contents"}`` read back from a written PDF: per page,
    its first line of text, how many characters and images it holds, and a
    note on a page with nothing on it. ``{}`` when it cannot be read."""
    try:
        import fitz                                        # PyMuPDF
        doc = fitz.open(path)
    except Exception:                                      # noqa: BLE001
        return {}
    try:
        rows = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text = page.get_text("text") or ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            n_img = len(page.get_images(full=True))
            row = {"page": i + 1, "first_line": (lines[0][:80] if lines
                                                 else ""),
                   "chars": len(text.strip()), "images": n_img}
            if not lines and not n_img:
                row["note"] = "nothing on this page"
            rows.append(row)
        out = {"pages": doc.page_count, "page_contents": rows}
        if doc.page_count > max_pages:
            out["page_contents_note"] = (f"first {max_pages} of "
                                         f"{doc.page_count} pages")
        return out
    except Exception:                                      # noqa: BLE001
        return {}
    finally:
        doc.close()


def _generate_html_to_pdf(params: dict) -> dict:
    from calc_package.latex_renderer import save_html_pdf_story

    reject_unknown_params(
        params, {"html", "html_path", "output_path", "allow_unembeddable"},
        method="html_to_pdf")
    html = params.get("html")
    html_path = params.get("html_path")
    if not html and not html_path:
        return {"status": "error",
                "error": "Provide 'html' (self-contained HTML content) or "
                         "'html_path' (an existing HTML file on the REAL "
                         "filesystem, e.g. a save_file or calc-package path)."}
    base_dir = None
    if not html:
        src = os.path.abspath(html_path)
        if not os.path.isfile(src):
            # a bare name is the file of that name in the working folder
            # (results name files that way since 2026-10-09)
            src = find_in_working_folder(html_path) or src
        if not os.path.isfile(src):
            return {"status": "error",
                    "error": f"html_path not found on the real filesystem: "
                             f"'{src}'. Agent-side scratch filesystems are "
                             "sandboxed and not on the real disk — pass the "
                             "HTML content directly via 'html' instead."}
        with open(src, "r", encoding="utf-8", errors="replace") as f:
            html = f.read()
        base_dir = os.path.dirname(src)  # relative <img> paths sit beside it

    # Embed what can be embedded (real local PNG/JPEG paths), then refuse
    # loudly if anything the engine would silently drop is left.
    html, inlined, img_problems = _embed_local_images(html, base_dir)
    findings = _unembeddable_findings(html, img_problems)
    allow = bool(params.get("allow_unembeddable", False))
    if findings and not allow:
        return {"status": "error",
                "analysis_type": "HTML report to PDF",
                "error": _unembeddable_error(findings),
                "unembeddable": findings[:12],
                "images_embedded": len(inlined)}

    output_path = params.get("output_path") or _default_output_path(
        "report", "pdf")
    if not output_path.lower().endswith(".pdf"):
        output_path += ".pdf"
    abs_path = os.path.abspath(output_path)
    try:
        save_html_pdf_story(html, abs_path)
    except Exception as exc:
        return {"status": "error",
                "error": f"PDF render failed ({type(exc).__name__}: {exc}). "
                         "The Story engine handles headings/tables/colours and "
                         "base64 PNG/JPEG <img> figures, but NOT inline <svg> "
                         "or advanced CSS — simplify the HTML and retry."}

    problem = written_file_problem(abs_path, None)
    file_exists = os.path.isfile(abs_path)
    response = {
        "status": "success",
        "analysis_type": "HTML report to PDF",
        "output_path": abs_path,
        "file_exists": file_exists and problem is None,
        "file_size_bytes": os.path.getsize(abs_path) if file_exists else 0,
        "format": "pdf",
        "renderer": "story",
    }
    if inlined:
        response["images_embedded"] = len(inlined)
        response["embedded_images"] = [item["path"] for item in inlined]
    if file_exists and problem is None:
        # What landed on each page, read back from the PDF (GM-101: the
        # calc helper could not see the PDF it built, and spent three
        # delegations finding out what was in it).
        layout = _pdf_layout(abs_path)
        if layout:
            response.update(layout)
    if findings:  # only reachable with allow_unembeddable=true
        response["unembeddable"] = findings[:12]
        response["renderer_warnings"] = [
            f"{len(findings)} figure reference(s) were NOT embedded and are "
            "missing from the PDF (allow_unembeddable=true). Tell the user "
            "which figures are absent — do not describe the PDF as complete."]
    if problem:
        response["status"] = "error"
        response["error"] = (f"PDF was generated but {problem}."
                             + workspace_write_hint(abs_path))
    return response


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
# Pavement design (AASHTO 1993)
# ---------------------------------------------------------------------------

def _generate_pavement_design_package(params: dict) -> dict:
    """AASHTO 1993 flexible or rigid pavement design calc package.

    ``design_type`` selects the procedure ('flexible' default | 'rigid');
    the remaining parameters are those of the pavement_design adapter
    methods (US customary units). Note: for rigid designs,
    ``pavement_type`` keeps its Table 2.6 meaning (JCP/JRCP vs CRCP).
    """
    from funhouse_agent.adapters.pavement_design_adapter import (
        _FLEX_ALIASES, _FLEX_VALID, _RIGID_ALIASES, _RIGID_VALID,
        check_items)
    from pavement_design import (design_flexible_pavement,
                                 design_rigid_pavement)

    from funhouse_agent.adapters.pavement_design_adapter import (
        _COMPARE_VALID, _UFC_ALIASES, _UFC_FLEX_VALID, _UFC_RIGID_VALID)
    from pavement_design import (compare_flexible_pavement_methods,
                                 design_flexible_pavement_ufc,
                                 design_rigid_pavement_ufc)

    design_type = str(params.get("design_type", "flexible")).strip().lower()
    meta_keys = set(_COMMON_PARAMS) | {"design_type"}
    design_params = {k: v for k, v in params.items() if k not in meta_keys}

    if design_type in ("flexible_ufc", "ufc_flexible", "ufc"):
        p = apply_aliases(design_params, _UFC_ALIASES)
        reject_unknown_params(p, _UFC_FLEX_VALID,
                              method="pavement_design_package (flexible_ufc)",
                              aliases=_UFC_ALIASES)
        require_params(p, ["passes_18kip", "cbr_subgrade"],
                       method="pavement_design_package",
                       valid=_UFC_FLEX_VALID)
        result = design_flexible_pavement_ufc(**p)
        extra = {
            "required_total_thickness_in":
                result["section"]["required_total_thickness_in"],
            "provided_total_thickness_in":
                result["section"]["provided_total_thickness_in"],
            "frost_governs": result.get("frost_governs"),
        }
        return _build_response("pavement_design", result, None, params,
                               analysis_type="UFC 3-250-01 Flexible "
                                             "Pavement Design", extra=extra)
    if design_type in ("rigid_ufc", "ufc_rigid"):
        p = apply_aliases(design_params, _UFC_ALIASES)
        reject_unknown_params(p, _UFC_RIGID_VALID,
                              method="pavement_design_package (rigid_ufc)",
                              aliases=_UFC_ALIASES)
        require_params(p, ["passes_18kip", "flexural_strength_psi"],
                       method="pavement_design_package",
                       valid=_UFC_RIGID_VALID)
        result = design_rigid_pavement_ufc(**p)
        extra = {"hd_required_in": result["hd_required_in"],
                 "slab_provided_in": result["slab_provided_in"]}
        return _build_response("pavement_design", result, None, params,
                               analysis_type="UFC 3-250-01 Rigid Pavement "
                                             "Design", extra=extra)
    if design_type in ("compare", "comparison"):
        p = apply_aliases(design_params, _UFC_ALIASES)
        reject_unknown_params(p, _COMPARE_VALID,
                              method="pavement_design_package (compare)",
                              aliases=_UFC_ALIASES)
        require_params(p, ["passes_18kip"],
                       method="pavement_design_package",
                       valid=_COMPARE_VALID)
        result = compare_flexible_pavement_methods(**p)
        extra = {
            "aashto_total_in": result["aashto_1993"]["total_thickness_in"],
            "ufc_total_in": result["ufc_3_250_01"]["total_thickness_in"],
            "delta_total_thickness_in": result["delta_total_thickness_in"],
        }
        return _build_response("pavement_design", result, None, params,
                               analysis_type="Pavement Method Comparison "
                                             "(AASHTO 1993 vs UFC 3-250-01)",
                               extra=extra)

    if design_type == "flexible":
        p = apply_aliases(design_params, _FLEX_ALIASES)
        reject_unknown_params(p, _FLEX_VALID,
                              method="pavement_design_package (flexible)",
                              aliases=_FLEX_ALIASES)
        require_params(p, ["w18", "layers"],
                       method="pavement_design_package", valid=_FLEX_VALID)
        check_items(p, method="pavement_design_package (flexible)")
        result = design_flexible_pavement(**p)
        extra = {
            "sn_required": result.sn_required,
            "sn_provided": result.sn_provided,
            "layer_thicknesses_in": {
                lay["layer_type"]: lay.get("thickness_in")
                for lay in result.layers},
            "adequate": result.adequate,
        }
        label = "AASHTO 1993 Flexible Pavement Design"
    elif design_type == "rigid":
        p = apply_aliases(design_params, _RIGID_ALIASES)
        reject_unknown_params(p, _RIGID_VALID,
                              method="pavement_design_package (rigid)",
                              aliases=_RIGID_ALIASES)
        require_params(p, ["w18", "sc_psi", "ec_psi"],
                       method="pavement_design_package", valid=_RIGID_VALID)
        check_items(p, method="pavement_design_package (rigid)")
        result = design_rigid_pavement(**p)
        extra = {
            "d_required_in": result.d_required_in,
            "d_provided_in": result.d_provided_in,
            "k_pci": result.k_pci,
            "adequate": result.adequate,
        }
        label = "AASHTO 1993 Rigid Pavement Design"
    else:
        raise ValueError(
            f"design_type must be 'flexible' or 'rigid', got '{design_type}'"
        )

    return _build_response("pavement_design", result, None, params,
                           analysis_type=label, extra=extra)


# ---------------------------------------------------------------------------

METHOD_REGISTRY = {
    "html_to_pdf": _generate_html_to_pdf,
    "render_figures": _render_figures,
    "pavement_design_package": _generate_pavement_design_package,
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
    "output_path": ("The saved calc package file: its name in the working "
                    "folder (an absolute path only outside one)."),
    "file_exists": "True if the file was verified on disk after writing. "
                   "Trust this field — do NOT try to verify the file with "
                   "agent-side filesystem tools (they may be sandboxed).",
    "file_size_bytes": "Size of the saved file (0 if file_exists is false).",
    "rescue_path": "Only on write failure: a verified copy of the package, in "
                   "the working folder (the temp dir outside the app); the "
                   "error names it.",
    "format": "Output format used.",
    "html_length": "Length of generated content.",
    "tables": ("[{title, output_path, rows, file_exists}]: one CSV per "
               "calculation table (per-node, per-layer, per-pile values; "
               "the input echo is *_inputs.csv), saved beside the package, "
               "which prints the same tables. Name them to the user."),
    "tables_error": "Only when a table could not be built or saved.",
}

METHOD_INFO = {
    "pavement_design_package": {
        "category": "Calculation Package",
        "brief": ("AASHTO 1993 pavement design calc package: flexible "
                  "(required SN + layer split) or rigid (slab D) with "
                  "reliability, drainage, and forward checks. US customary "
                  "units. Same parameters as the pavement_design module "
                  "methods, plus design_type."),
        "parameters": {
            "design_type": {"type": "str", "required": False,
                            "default": "flexible",
                            "allowed_values": ["flexible", "rigid"],
                            "description": "Which AASHTO 1993 procedure to "
                                           "run and report."},
            "w18": {"type": "number", "required": True,
                    "description": "Design-lane 18-kip ESALs."},
            "layers": {"type": "array", "required": False,
                       "description": "Flexible only -- layer dicts, "
                                      "top-down (see pavement_design."
                                      "flexible_pavement_design)."},
            "sc_psi": {"type": "number", "required": False,
                       "description": "Rigid only -- PCC modulus of "
                                      "rupture, psi."},
            "ec_psi": {"type": "number", "required": False,
                       "description": "Rigid only -- PCC elastic modulus, "
                                      "psi."},
            "reliability_pct": {"type": "number", "required": False,
                                "description": "Design reliability R %."},
            "mr_psi": {"type": "number", "required": False,
                       "description": "Roadbed resilient modulus, psi "
                                      "(flexible), or simplified-k source "
                                      "(rigid)."},
            "k_pci": {"type": "number", "required": False,
                      "description": "Rigid: effective k directly, pci."},
            "composite_k": {"type": "object", "required": False,
                            "description": "Rigid: Section 3.2 worksheet "
                                           "spec (see pavement_design."
                                           "rigid_pavement_design)."},
            "slab_thickness_in": {"type": "number", "required": False,
                                  "description": "Rigid: check an existing "
                                                 "slab instead."},
            "**other": {"type": "any", "required": False,
                        "description": "All other pavement_design method "
                                       "parameters pass through (so, pt, "
                                       "po, delta_psi, j, cd, "
                                       "drainage_quality, ...)."},
            **_COMMON_PARAMS,
        },
        # The handler checks the parameters of the design_type it runs, so
        # dispatch leaves them to it (GM-110).
        "open_parameters": True,
        "returns": {**_COMMON_RETURNS,
                    "extra": "sn_required/sn_provided + layer thicknesses "
                             "(flexible) or d_required_in/d_provided_in + "
                             "k_pci (rigid), and adequate."},
    },
    "render_figures": {
        "category": "Calculation Package",
        "brief": ("The FIGURES of any canned *_package — slope section + "
                  "trial-surface map, p-y curves, settlement/consolidation "
                  "plots, wall pressure diagrams, pavement charts, FEM "
                  "mesh/contours — saved as standalone PNGs WITHOUT building "
                  "the package. For a bespoke html_to_pdf report (SOE, a "
                  "multi-analysis narrative, anything with no canned "
                  "template): run the analysis here, paste each figure's "
                  "html_img_tag into your HTML. Same inputs as the package."),
        "parameters": {
            "package": {"type": "str", "required": True,
                        "description": ("Which package's figures: e.g. "
                                        "'slope_stability_package', "
                                        "'axial_pile_package', "
                                        "'settlement_package'. Any *_package "
                                        "method of this module except "
                                        "html_to_pdf. PLUS every input the "
                                        "named package takes, beside it — "
                                        "describe_method('calc_package', "
                                        "'<package>') lists them. The "
                                        "analysis runs exactly as the "
                                        "package would.")},
            "output_dir": {"type": "str", "required": False,
                           "description": ("Folder for the PNGs (one per "
                                           "figure). Defaults to the working "
                                           "folder.")},
            "name_prefix": {"type": "str", "required": False,
                            "description": "File-name prefix (default: module)."},
        },
        # Dispatch also accepts the named package's parameters (GM-110).
        "parameters_of": "package",
        "parameter_aliases": {"method": "package"},
        "returns": {
            "status": "success or error.",
            "n_figures": "How many figures were saved.",
            "figures": ("[{index, title, caption, output_path, file_exists, "
                        "file_size_bytes, html_img_tag}] — paste html_img_tag "
                        "into report HTML; html_to_pdf embeds the PNG."),
            "output_dir": "Where the PNGs went.",
            "note": "Present when the analysis produces no figures.",
            "...": ("The package's own key results (e.g. q_ultimate_kPa, "
                    "fos) are echoed too."),
        },
    },
    "html_to_pdf": {
        "category": "Calculation Package",
        "brief": "Render self-contained HTML (e.g. a custom report YOU composed) to a "
                 "paginated PDF on the REAL filesystem. Use when no canned *_package "
                 "method matches the analysis — compose the report HTML yourself "
                 "(inline CSS; figures as a real local PNG path or a base64 PNG/JPEG "
                 "data URI) and pass it here. Figures it cannot embed are a hard ERROR, "
                 "never a silent blank. Pure-Python (PyMuPDF Story engine).",
        "parameters": {
            "html": {"type": "str", "required": False,
                     "description": "Self-contained HTML content to render. Inline all "
                                    "CSS. FIGURES: use <img src=\"fig.png\"> "
                                    "(a PNG/JPEG in the working folder, named as the "
                                    "figure tool returned it, or a real local path, "
                                    "is read and embedded for you — this is the easy path, e.g. the "
                                    "html_img_tag returned by profile_figure) or a "
                                    "base64 PNG/JPEG data URI. Inline <svg>, remote "
                                    "image URLs and literal '[image]' placeholders are "
                                    "REJECTED with a list of what to fix. "
                                    "Provide this or 'html_path'."},
            "html_path": {"type": "str", "required": False,
                          "description": "Path to an existing HTML file on the REAL "
                                         "filesystem (a save_file or calc-package output "
                                         "path — NOT a sandboxed scratch-filesystem path). "
                                         "Relative <img> paths inside it resolve against "
                                         "the working folder."},
            "output_path": {"type": "str", "required": False,
                            "description": "PDF output path. Auto-generated if omitted."},
            "allow_unembeddable": {
                "type": "bool", "required": False, "default": False,
                "description": "Escape hatch: render anyway when the HTML references "
                               "figures that cannot be embedded, instead of erroring. "
                               "Those figures WILL BE MISSING from the PDF — the "
                               "response then carries 'unembeddable' and "
                               "'renderer_warnings', and you must tell the user which "
                               "figures are absent. Prefer fixing the figures."},
        },
        "returns": {
            "status": "success or error.",
            "output_path": "Absolute path of the saved PDF.",
            "file_exists": "True if the file was verified on disk after writing. "
                           "Trust this field — do NOT try to verify the file with "
                           "agent-side filesystem tools (they may be sandboxed).",
            "file_size_bytes": "Size of the saved PDF.",
            "renderer": "'story' (pure-Python PyMuPDF engine).",
            "images_embedded": "How many local image files were inlined into the PDF.",
            "pages": "How many pages the PDF has.",
            "page_contents": "What landed on each page, read back from the PDF: "
                             "its first line of text, characters and images "
                             "(a page with nothing on it says so). Check a "
                             "figure landed where the report needs it from "
                             "this, without opening the PDF.",
            "unembeddable": "On error (or with allow_unembeddable=true): each figure "
                            "reference that cannot be embedded, and why.",
        },
    },
    "bearing_capacity_package": {
        "category": "Calculation Package",
        "brief": "Run bearing capacity analysis and generate Mathcad-style calc package, with a table (CSV + printed) of the factors N, s, d, i, b, g and the value of each term.",
        "parameters": {
            "width": {"type": "float", "required": True, "description": "Footing width B (m)."},
            "unit_weight": {"type": "float", "required": True, "description": "Soil unit weight (kN/m3)."},
            "friction_angle": {"type": "float", "required": False, "default": 0.0, "description": "Friction angle (deg)."},
            "cohesion": {"type": "float", "required": False, "default": 0.0, "description": "Cohesion (kPa)."},
            "depth": {"type": "float", "required": False, "default": 0.0, "description": "Embedment depth Df (m)."},
            "shape": {"type": "str", "required": False, "default": "strip", "allowed_values": ["strip", "square", "rectangular", "circular"], "description": "Footing shape."},
            "length": {"type": "float", "required": False, "description": "Footing length L (m). Required for rectangular."},
            "factor_of_safety": {"type": "float", "required": False, "default": 3.0, "description": "Factor of safety."},
            "ngamma_method": {"type": "str", "required": False, "default": "vesic", "allowed_values": ["vesic", "meyerhof", "hansen", "kumbhojkar", "coduto"], "description": "Ngamma factor method; with factor_method='terzaghi', 'kumbhojkar' (default) or 'coduto'."},
            "factor_method": {"type": "str", "required": False, "default": "vesic", "allowed_values": ["vesic", "meyerhof", "terzaghi"], "description": "Correction-factor method; 'terzaghi' = Terzaghi (1943) factors and shape coefficients."},
            "failure_mode": {"type": "str", "required": False, "default": "general", "allowed_values": ["general", "local"], "description": "factor_method='terzaghi' only: 'local' = Terzaghi's local-shear reduction."},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
            "base_tilt": {"type": "float", "required": False, "default": 0.0, "description": "Base tilt angle (degrees)."},
            "eccentricity_B": {"type": "float", "required": False, "default": 0.0, "description": "Load eccentricity along B (m)."},
            "eccentricity_L": {"type": "float", "required": False, "default": 0.0, "description": "Load eccentricity along L (m)."},
            "load_inclination": {"type": "float", "required": False, "default": 0.0, "description": "Load inclination from vertical (degrees)."},
            "ground_slope": {"type": "float", "required": False, "default": 0.0, "description": "Ground slope angle (degrees)."},
            "vertical_load": {"type": "float", "required": False, "default": 0.0, "description": "Applied vertical load (kN) for inclination factors."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "q_ultimate_kPa": "Ultimate bearing capacity.", "q_allowable_kPa": "Allowable bearing capacity."},
    },
    "lateral_pile_package": {
        "category": "Calculation Package",
        "brief": "Run lateral pile analysis and generate LPILE-style calc package, with tables (CSV + printed) of y, slope, M, V, p and Es at every node and each layer's p-y curve.",
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
        "brief": "Run slope stability analysis and generate calc package with cross-section plot, and tables (CSV) of the stresses on the slip surface, the forces on every slice, FOS by method and the line of thrust.",
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
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge (kPa)."},
            "compare_methods": {"type": "bool", "required": False, "default": False, "description": "Also compute the FOS by the other methods."},
            "FOS_required": {"type": "float", "required": False, "default": 1.5, "description": "Required factor of safety for the stability check."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "FOS": "Factor of safety.", "method": "Method used.", "is_stable": "Whether FOS >= required."},
    },
    "slope_report_package": {
        "category": "Calculation Package",
        "brief": "One-call slope stability REPORT for a critical-surface SEARCH: input echo, search story (surfaces tried + rejection diagnostics + critical surface/FOS), FOS-by-method table, slice-force + interslice/thrust-line figures, reinforcement summary, a probabilistic annex (FOSM/Monte Carlo) when 'variables' is supplied, and CSV tables (every slice, every trial surface). No fixed circle needed — the search finds the critical surface.",
        "parameters": {
            "surface_points": {"type": "array", "required": True, "description": "List of [x, z] ground surface coordinates."},
            "soil_layers": {"type": "array", "required": True, "description": "List of dicts: name, top_elevation, bottom_elevation, gamma, phi, c_prime (or cu + analysis_mode). Optional 'bottom_boundary_points' ([[x,z],...]) defines a non-horizontal layer base (e.g. a weak seam)."},
            "method": {"type": "str", "required": False, "default": "spencer", "allowed_values": ["spencer", "morgenstern_price", "gle", "bishop", "janbu", "fellenius"], "description": "Limit-equilibrium method the search minimizes and the report highlights. Default 'spencer' (rigorous — populates the interslice-force / line-of-thrust figure)."},
            "surface_type": {"type": "str", "required": False, "default": "circular", "allowed_values": ["circular", "entry_exit", "noncircular", "noncircular_de", "pso", "weak_layer"], "description": "Trial-surface family for the search. 'circular' grids circle centres over x_range/y_range and keeps only circles whose ends fall in x_entry_range/x_exit_range; the others draw their surfaces between x_entry_range and x_exit_range. Ranges not given are set from the slope (toe, crest, height, depth to the base). 'weak_layer' biases trials toward a defined weak seam."},
            "x_range": {"type": "array", "required": False, "description": "[min, max] circle-center x for the search grid (circular)."},
            "y_range": {"type": "array", "required": False, "description": "[min, max] circle-center y for the search grid (circular)."},
            "nx": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in x (circular search)."},
            "ny": {"type": "int", "required": False, "default": 10, "description": "Grid divisions in y (circular search)."},
            "x_entry_range": {"type": "array", "required": False, "description": "[min, max] x where a trial surface's low-x end may lie (every search, circular included)."},
            "x_exit_range": {"type": "array", "required": False, "description": "[min, max] x where a trial surface's high-x end may lie (every search, circular included)."},
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
            "time_budget_s": {"type": "float", "required": False, "default": 180.0, "description": "Time budget of EACH critical-surface search in seconds (at most 600), as slope_stability search_critical_surface: the report's search, and with research_surface each Monte Carlo realization's (per search, not for the run). A search cut short reports the lowest surface in what it covered and says so in its warnings."},
            "fosm": {"type": "bool", "required": False, "default": True, "description": "With 'variables': run the FOSM annex."},
            "FOS_required": {"type": "float", "required": False, "default": 1.5, "description": "Required factor of safety for the stability check."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge (kPa)."},
            "tol": {"type": "float", "required": False, "default": 1e-4, "description": "Convergence tolerance for the iterative methods."},
            "f_interslice": {"type": "str", "required": False, "default": "half_sine", "description": "Interslice force function for the rigorous re-analysis of the critical surface."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "FOS": "Critical factor of safety.", "method": "Method used.", "n_surfaces_evaluated": "Trial surfaces evaluated by the search.", "is_stable": "Whether FOS >= required.", "governing_surface": "Kind of surface that governs: 'circular' / 'noncircular' (the main search), 'toe_circle', 'circle_above_toe' (lower end on the face above the toe), or 'infinite_slope' (shallow planar sliding where c'=0 soil forms the face).", "candidates": "Minimum FOS of each surface family tried.", "warnings": "What the search could not be sure of (also printed in the package). Report every one."},
    },
    "settlement_package": {
        "category": "Calculation Package",
        "brief": "Run settlement analysis and generate calc package, with tables (CSV + printed) of consolidation settlement per layer (e0, Cc, Cr, stresses) and settlement against time.",
        "parameters": {
            "q_applied": {"type": "float", "required": True, "description": "Applied bearing pressure (kPa)."},
            "B": {"type": "float", "required": False, "default": 1.0, "description": "Footing width (m)."},
            "L": {"type": "float", "required": False, "default": 1.0, "description": "Footing length (m)."},
            "Es_immediate": {"type": "float", "required": False, "description": "Elastic modulus for immediate settlement (kPa)."},
            "consolidation_layers": {"type": "array", "required": False, "description": "List of dicts: thickness, depth_to_center, e0, Cc, Cr, sigma_v0, sigma_p."},
            "schmertmann_layers": {"type": "array", "required": False, "description": "List of dicts: depth_top, depth_bottom, Es."},
            "q_overburden": {"type": "float", "required": False, "default": 0.0, "description": "Overburden pressure at footing base (kPa)."},
            "footing_shape": {"type": "str", "required": False, "default": "square", "allowed_values": ["square", "rectangular", "strip"], "description": "Footing shape."},
            "stress_method": {"type": "str", "required": False, "default": "2:1", "allowed_values": ["2:1", "boussinesq", "westergaard"], "description": "Stress distribution method."},
            "immediate_method": {"type": "str", "required": False, "default": "elastic", "allowed_values": ["elastic", "schmertmann"], "description": "Immediate settlement method."},
            "nu": {"type": "float", "required": False, "default": 0.3, "description": "Poisson's ratio for immediate settlement."},
            "time_years_schmertmann": {"type": "float", "required": False, "default": 0.0, "description": "Creep time for Schmertmann immediate method (years)."},
            "cv": {"type": "float", "required": False, "description": "Coefficient of consolidation (m2/yr), for the time curve."},
            "Hdr": {"type": "float", "required": False, "description": "Drainage path length (m) for the time curve."},
            "drainage": {"type": "str", "required": False, "default": "double", "allowed_values": ["single", "double"], "description": "Drainage condition for time rate."},
            "C_alpha": {"type": "float", "required": False, "description": "Secondary compression index. Enables secondary settlement."},
            "e0_secondary": {"type": "float", "required": False, "default": 1.0, "description": "Void ratio for secondary settlement."},
            "t_secondary": {"type": "float", "required": False, "default": 0.0, "description": "Time for secondary settlement (years)."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "total_settlement_mm": "Total settlement.", "immediate_mm": "Immediate.", "consolidation_mm": "Consolidation."},
    },
    "axial_pile_package": {
        "category": "Calculation Package",
        "brief": "Run axial pile capacity analysis and generate APILE-style calc package, with tables (CSV + printed) of side resistance per layer, load transfer at the ultimate load and capacity against pile length.",
        "parameters": {
            "pile_area": {"type": "float", "required": True, "description": "Pile cross-section area (m2)."},
            "pile_perimeter": {"type": "float", "required": True, "description": "Pile perimeter (m)."},
            "pile_tip_area": {"type": "float", "required": True, "description": "Pile tip area (m2)."},
            "pile_width": {"type": "float", "required": True, "description": "Pile width/diameter (m)."},
            "pile_length": {"type": "float", "required": True, "description": "Embedded pile length (m)."},
            "soil_layers": {"type": "array", "required": True, "description": "List of dicts: soil_type, thickness, unit_weight, friction_angle, cohesion; optional delta_phi_ratio, toe_friction_angle (alias toe_phi; the end-bearing phi), description. Same layer keys as axial_pile.axial_pile_capacity."},
            "factor_of_safety": {"type": "float", "required": False, "default": 2.5, "description": "Factor of safety."},
            "method": {"type": "str", "required": False, "default": "auto", "allowed_values": ["auto", "beta"], "description": "auto = Nordlund (sand) / Tomlinson (clay); beta = effective-stress method."},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
            "pile_name": {"type": "str", "required": False, "default": "Custom Pile", "description": "Section designation for the report."},
            "pile_type": {"type": "str", "required": False, "default": "h_pile", "allowed_values": ["h_pile", "pipe_open", "pipe_closed", "concrete", "timber"], "description": "Pile type."},
            "pile_depth": {"type": "float", "required": False, "description": "Section depth (m): an H-pile's nominal depth; a round pile's equals pile_width."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "Q_ultimate_kN": "Ultimate capacity.", "Q_allowable_kN": "Allowable capacity."},
    },
    "drilled_shaft_package": {
        "category": "Calculation Package",
        "brief": "Run drilled shaft capacity analysis and generate SHAFT-style calc package, with tables (CSV + printed) of side resistance per layer, load transfer at the ultimate load and capacity against shaft length.",
        "parameters": {
            "diameter": {"type": "float", "required": True, "description": "Shaft diameter (m)."},
            "length": {"type": "float", "required": True, "description": "Shaft length (m)."},
            "soil_layers": {"type": "array", "required": True, "description": "List of dicts: soil_type, thickness, unit_weight, cu, phi, N60, qu, RQD."},
            "gwt_depth": {"type": "float", "required": False, "description": "Groundwater depth (m)."},
            "factor_of_safety": {"type": "float", "required": False, "default": 2.5, "description": "Factor of safety."},
            "socket_diameter": {"type": "float", "required": False, "description": "Rock socket diameter if different from the shaft's (m)."},
            "socket_length": {"type": "float", "required": False, "default": 0.0, "description": "Rock socket length (m)."},
            "bell_diameter": {"type": "float", "required": False, "description": "Bell (underream) diameter at the base (m)."},
            "casing_depth": {"type": "float", "required": False, "default": 0.0, "description": "Depth of permanent casing (m); no side resistance over it."},
            "concrete_fc": {"type": "float", "required": False, "default": 28000.0, "description": "Concrete f'c (kPa)."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "Q_ultimate_kN": "Ultimate capacity.", "Q_allowable_kN": "Allowable capacity."},
    },
    "downdrag_package": {
        "category": "Calculation Package",
        "brief": "Run downdrag analysis and generate calc package, with tables (CSV + printed) of axial load, unit skin friction and soil settlement along the pile, and every neutral-plane method side by side (CSV).",
        "parameters": {
            "pile_length": {"type": "float", "required": True, "description": "Pile length (m)."},
            "pile_diameter": {"type": "float", "required": True, "description": "Pile diameter (m)."},
            "pile_E": {"type": "float", "required": True, "description": "Pile Young's modulus (kPa)."},
            "Q_dead": {"type": "float", "required": True, "description": "Dead load at pile head (kN)."},
            "soil_layers": {"type": "array", "required": True, "description": "List of dicts: soil_type ('cohesionless' or 'cohesive'), thickness (m), unit_weight (kN/m3), cu (kPa) or phi (deg), alpha / beta (optional), settling (bool). A settling layer needs its compressibility: cohesive -> Cc and e0 (Cr, sigma_p optional) or C_ec (C_er optional); cohesionless -> E_s (kPa, nu_s optional)."},
            "gwt_depth": {"type": "float", "required": False, "default": 0.0, "description": "Groundwater depth (m)."},
            "fill_thickness": {"type": "float", "required": False, "default": 0.0, "description": "Fill thickness causing downdrag (m)."},
            "fill_unit_weight": {"type": "float", "required": False, "default": 20.0, "description": "Fill unit weight (kN/m3)."},
            "gw_drawdown": {"type": "float", "required": False, "default": 0.0, "description": "Groundwater drawdown (m)."},
            "pile_perimeter": {"type": "float", "required": False, "description": "Pile perimeter for skin friction (m). Default: the circle of pile_diameter (pi x D); give it for an H pile or any non-circular section."},
            "pile_area": {"type": "float", "required": False, "description": "Pile cross-section area (m2) for the pile's elastic compression (A x E) and, when include_pile_weight is true, its own weight. Default: the solid circle of pile_diameter; for a steel H or pipe pile give the steel area."},
            "pile_unit_weight": {"type": "float", "required": False, "default": 78.5, "description": "Pile material unit weight (kN/m3), times pile_area for the pile's own weight; used only when include_pile_weight is true. Default 78.5 = steel, as in downdrag.downdrag_analysis (concrete is about 24)."},
            "include_pile_weight": {"type": "bool", "required": False, "default": False, "description": "Whether the load curve carries the pile's own weight. Default false: left out, as UFC 3-220-20 Fig 6-26 and FHWA GEC-12 7.3.6.1 do. True adds the weight of the pile above the neutral plane to Q_np and the drag force."},
            "structural_capacity": {"type": "float", "required": False, "description": "Factored structural resistance P_r (kN) for the UFC Eq 6-80 check."},
            "allowable_settlement": {"type": "float", "required": False, "description": "Allowable settlement (m) for the serviceability check."},
            "Nt": {"type": "float", "required": False, "description": "Toe bearing capacity factor: R_toe = Nt x sigma'v(toe) x A_toe in cohesionless soil, Nt x cu x A_toe in cohesive. Give the MOBILIZED value -- the toe resistance actually developed at the pile's settlement -- not the ultimate. Omitted, Nt is the ULTIMATE value from the toe layer's phi (100-150 in dense sand; 9 in clay); then the load and resistance curves often do not cross, and the neutral plane comes instead from the mobilized toe resistance compatible with the toe penetration (UFC 3-220-20 6-5.8.4.3, Fig 6-19; see neutral_plane_method). UFC 3-220-20 6-5.8.4.2: check 0 %, 50 % and 100 % toe mobilization."},
            "neutral_plane_method": {"type": "string", "required": False, "default": "auto", "description": "How the headline neutral plane is located, as in downdrag.downdrag_analysis: 'auto' (the UFC 3-220-20 6-7.4 procedure), 'force_equilibrium' (where the load curve meets the resistance curve at the chosen toe mobilization), 'settlement_compatibility' (the mobilized toe resistance compatible with the toe penetration, UFC Fig 6-19), 'bearing_layer_top', 'pile_toe', 'endo' (CGPR #56). The package tabulates every method whatever is chosen.", "allowed_values": ["auto", "force_equilibrium", "settlement_compatibility", "bearing_layer_top", "pile_toe", "endo"]},
            "toe_mobilization": {"type": "float", "required": False, "default": 1.0, "description": "Fraction (0-1) of the toe resistance for force equilibrium (UFC 6-5.8.4.2: 0, 0.5, 1)."},
            "endo_bearing_condition": {"type": "string", "required": False, "description": "CGPR #56 Table 3.1 condition for the Endo et al. (1969) method.", "allowed_values": ["floating", "stiff_flexible", "end_bearing"]},
            "skin_friction_stress": {"type": "string", "required": False, "default": "final", "description": "Effective stress for beta friction and the toe in sand: 'final' (default; UFC 3-220-20 6-7.4 step 2, the profile that includes the stress change producing settlement; GEC-12 is silent) or 'initial' (before the fill or drawdown).", "allowed_values": ["final", "initial"]},
            "influence_depth_criterion": {"type": "string", "required": False, "default": "bearing_stress", "description": "How deep below the toe the toe load is counted, as in downdrag.downdrag_analysis: to where its stress increase falls below 'bearing_stress' (default) = 10% of the bearing stress of the equivalent footing (UFC 3-220-20 6-5.8.4.3) or 'effective_stress' = 10% of the existing vertical effective stress (FHWA GEC-12 7.3.5.3). The fill / drawdown compresses the settling layers below the toe down to the bottom of the soil profile either way.", "allowed_values": ["bearing_stress", "effective_stress"]},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS,
                    "neutral_plane_m": "Neutral plane depth (m).",
                    "neutral_plane_method": "'force_equilibrium', 'settlement_compatibility', 'bearing_layer_top', 'pile_toe', 'endo' or 'none' (see downdrag.downdrag_analysis).",
                    "headline_method": "The row of neutral_plane_comparison the headline numbers come from, named as its reference names it.",
                    "neutral_plane_basis": "One sentence saying how the headline neutral plane was located, with its source; also printed in the package.",
                    "methods_range": "One sentence giving the spread of neutral planes and drag forces over the methods that apply. Report it; the engineer chooses.",
                    "skin_friction_stress": "'final' or 'initial': the effective stress the friction and toe resistance used.",
                    "include_pile_weight": "Whether the load curve carried the pile's own weight (false by default).",
                    "pile_weight_basis": "One sentence saying whether the pile's own weight is in the load curve, naming the references; also printed in the package.",
                    "below_toe_basis": "How deep the settlement below the toe was counted and why it stopped (the toe load per influence_depth_criterion, the fill / drawdown to the bottom of the soil profile); also printed in the package.",
                    "neutral_plane_comparison": "Every method on the same inputs, each with its source (also tabulated in the package).",
                    "warnings": "What the engineer must know about the result; also printed in the package. Report every one.",
                    "dragload_kN": "Dragload (kN).",
                    "max_pile_load_kN": "Maximum pile load at the neutral plane (kN).",
                    "pile_settlement_mm": "Pile settlement (mm)."},
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
            "n_bar": {"type": "float", "required": False, "description": "Average SPT N — site classification (instead of vs30)."},
            "su_bar": {"type": "float", "required": False, "description": "Average undrained shear strength (kPa) — site classification (instead of vs30)."},
            "delta": {"type": "float", "required": False, "description": "Wall friction angle (deg) — earth pressure. Default 2/3 phi."},
            "kv": {"type": "float", "required": False, "default": 0.0, "description": "Vertical seismic coefficient — earth pressure."},
            "KA": {"type": "float", "required": False, "default": 0.333, "description": "Static active coefficient — earth pressure."},
            "amax_g": {"type": "float", "required": False, "description": "Peak ground acceleration (g) — liquefaction."},
            "layer_depths": {"type": "array", "required": False, "description": "Layer mid-depths (m) — liquefaction."},
            "layer_N160": {"type": "array", "required": False, "description": "(N1)60 per layer — liquefaction."},
            "layer_FC": {"type": "array", "required": False, "description": "Fines content (%) per layer — liquefaction."},
            "layer_gamma": {"type": "array", "required": False, "description": "Total unit weight (kN/m3) per layer — liquefaction."},
            "gwt_depth": {"type": "float", "required": False, "description": "Depth to the water table during the design earthquake (m) — liquefaction, where it is required: no default, it is a site input."},
            "M": {"type": "float", "required": False, "default": 7.5, "description": "Earthquake magnitude — liquefaction."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "seismic_analysis_type": "Which sub-analysis was performed."},
    },
    "retaining_wall_package": {
        "category": "Calculation Package",
        "brief": "Run retaining wall analysis and generate calc package.",
        "parameters": {
            "wall_height": {"type": "float", "required": True, "description": "TOTAL wall height H (m), from the UNDERSIDE of the base slab to the top of the stem (= base_thickness + stem height); add base_thickness to a height retained above the top of the footing."},
            "gamma_backfill": {"type": "float", "required": True, "description": "Backfill unit weight (kN/m3)."},
            "phi_backfill": {"type": "float", "required": True, "description": "Backfill friction angle (deg)."},
            "wall_type": {"type": "str", "required": False, "default": "cantilever", "allowed_values": ["cantilever"], "description": "Wall type (MSE not yet supported)."},
            "base_width": {"type": "float", "required": False, "description": "Base slab width B (m). Auto-sized ~0.6H if omitted — pass the REAL geometry for a representative package."},
            "toe_length": {"type": "float", "required": False, "description": "Toe projection (m). Auto-sized ~0.1B if omitted. Heel is inferred: B - toe - stem_thickness_base."},
            "stem_thickness_top": {"type": "float", "required": False, "default": 0.3, "description": "Stem thickness at top (m)."},
            "stem_thickness_base": {"type": "float", "required": False, "description": "Stem thickness at base of stem (m). Auto-sized if omitted."},
            "base_thickness": {"type": "float", "required": False, "default": 0.6, "description": "Base slab (footing) thickness (m)."},
            "has_shear_key": {"type": "bool", "required": False, "default": False, "description": "Include a shear key (with key_depth)."},
            "key_depth": {"type": "float", "required": False, "default": 0.0, "description": "Shear key depth below base (m)."},
            "backfill_slope": {"type": "float", "required": False, "default": 0.0, "description": "Backfill slope angle (deg)."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surcharge (kPa)."},
            "c_backfill": {"type": "float", "required": False, "default": 0.0, "description": "Backfill cohesion (kPa)."},
            "phi_foundation": {"type": "float", "required": False, "description": "Foundation soil friction angle (deg), FULL value — sliding applies delta_b = (2/3)*phi_foundation INTERNALLY; do NOT pre-reduce. Use delta_base to set the interface angle directly. Defaults to phi_backfill."},
            "c_foundation": {"type": "float", "required": False, "default": 0.0, "description": "Foundation soil cohesion (kPa), FULL value — sliding applies ca = (2/3)*c_foundation INTERNALLY; do NOT pre-reduce. Use base_adhesion to set adhesion directly."},
            "delta_base": {"type": "float", "required": False, "description": "Base interface friction angle delta_b (deg), used DIRECTLY (no 2/3 factor)."},
            "base_adhesion": {"type": "float", "required": False, "description": "Base adhesion ca (kPa), used DIRECTLY (no 2/3 factor; 0 = no adhesion)."},
            "include_passive": {"type": "bool", "required": False, "default": False, "description": "Include passive resistance in the sliding check."},
            "gamma_foundation": {"type": "float", "required": False, "description": "Foundation unit weight (kN/m3) for passive. Defaults to gamma_backfill."},
            "q_allowable": {"type": "float", "required": False, "description": "Allowable bearing pressure (kPa) for the bearing check. Without it the bearing check is NOT RUN and the package says 'not checked'."},
            "gamma_concrete": {"type": "float", "required": False, "default": 24.0, "description": "Concrete unit weight (kN/m3)."},
            "FOS_sliding": {"type": "float", "required": False, "default": 1.5, "description": "Required sliding FOS."},
            "FOS_overturning": {"type": "float", "required": False, "default": 2.0, "description": "Required overturning FOS."},
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb"], "description": "Earth pressure theory."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "verdict": "The result in plain words: failed checks first, then not checked. Report it to the user.", "checks_failed": "Checks that failed (sliding, overturning, bearing, eccentricity). Report every one.", "checks_not_run": "Checks not run ('bearing' without q_allowable).", "FOS_sliding": "Sliding FOS (null: no driving force).", "FOS_overturning": "Overturning FOS (null: no driving force).", "FOS_bearing": "Bearing FOS (null: not checked)."},
    },
    "ground_improvement_package": {
        "category": "Calculation Package",
        "brief": "Run ground improvement analysis and generate calc package.",
        "parameters": {
            "method": {"type": "str", "required": True, "allowed_values": ["wick_drains", "aggregate_piers", "surcharge", "vibro"], "description": "Ground improvement method."},
            "spacing": {"type": "float", "required": False, "description": "Element spacing (m) — wick drains, aggregate piers."},
            "ch": {"type": "float", "required": False, "description": "Horizontal cv (m2/yr) — wick drains, surcharge (optional drains)."},
            "cv": {"type": "float", "required": False, "description": "Vertical cv (m2/yr) — wick drains, surcharge."},
            "Hdr": {"type": "float", "required": False, "description": "Drainage path (m) — wick drains, surcharge."},
            "time": {"type": "float", "required": False, "description": "Design time (years) — wick drains."},
            "column_diameter": {"type": "float", "required": False, "description": "Stone-column/pier diameter (m) — aggregate piers (alias: diameter)."},
            "E_column": {"type": "float", "required": False, "default": 80000.0, "description": "Column modulus (kPa) — aggregate piers (alias: E_pier)."},
            "E_soil": {"type": "float", "required": False, "default": 5000.0, "description": "Matrix soil modulus (kPa) — aggregate piers."},
            "n": {"type": "float", "required": False, "default": 5.0, "description": "Stress concentration ratio — aggregate piers."},
            "S_unreinforced": {"type": "float", "required": False, "description": "Unreinforced settlement (m) — aggregate piers (enables the improvement figures)."},
            "q_unreinforced": {"type": "float", "required": False, "description": "Unreinforced bearing (kPa) — aggregate piers."},
            "S_ultimate": {"type": "float", "required": False, "description": "Ultimate primary settlement (m) — surcharge."},
            "surcharge_kPa": {"type": "float", "required": False, "description": "Surcharge fill stress (kPa) — surcharge (alias: surcharge)."},
            "target_U": {"type": "float", "required": False, "default": 90.0, "description": "Target consolidation (%) — surcharge (alias: target_consolidation)."},
            "drain_spacing": {"type": "float", "required": False, "description": "Optional wick-drain spacing (m) under the surcharge."},
            "fines_content": {"type": "float", "required": False, "description": "Fines content (%) — vibro feasibility."},
            "initial_N_spt": {"type": "float", "required": False, "description": "Pre-improvement (N1)60 — vibro (alias: N1_before)."},
            "target_N_spt": {"type": "float", "required": False, "default": 25.0, "description": "Target (N1)60 — vibro."},
            "D50": {"type": "float", "required": False, "description": "Median grain size (mm) — vibro."},
            "pattern": {"type": "str", "required": False, "default": "triangular", "allowed_values": ["triangular", "square"], "description": "Element pattern — piers/drains/vibro."},
            "dw": {"type": "float", "required": False, "default": 0.066, "description": "Equivalent drain diameter (m) — wick drains, surcharge with drains."},
            "smear_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Smear zone ratio ds/dw — wick drains, surcharge with drains."},
            "kh_ks_ratio": {"type": "float", "required": False, "default": 2.0, "description": "Permeability ratio kh/ks — wick drains, surcharge with drains."},
            "sigma_v0": {"type": "float", "required": False, "default": 0.0, "description": "Current effective stress at layer center (kPa) — surcharge."},
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
            "cushion_stiffness": {"type": "float", "required": False, "description": "Cushion stiffness (kN/m), instead of cushion_area/thickness/E."},
            "cushion_cor": {"type": "float", "required": False, "default": 0.8, "description": "Cushion coefficient of restitution."},
            "segment_length": {"type": "float", "required": False, "default": 1.0, "description": "Pile segment length (m)."},
            "pile_unit_weight": {"type": "float", "required": False, "default": 78.5, "description": "Pile material unit weight (kN/m3)."},
            "quake_side": {"type": "float", "required": False, "default": 0.0025, "description": "Side quake (m)."},
            "quake_toe": {"type": "float", "required": False, "default": 0.0025, "description": "Toe quake (m)."},
            "damping_side": {"type": "float", "required": False, "default": 0.16, "description": "Side Smith damping (s/m)."},
            "damping_toe": {"type": "float", "required": False, "default": 0.5, "description": "Toe Smith damping (s/m)."},
            "R_min": {"type": "float", "required": False, "default": 200.0, "description": "Min resistance (kN)."},
            "R_max": {"type": "float", "required": False, "default": 2000.0, "description": "Max resistance (kN)."},
            "R_step": {"type": "float", "required": False, "default": 200.0, "description": "Resistance increment (kN)."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "n_points": "Number of bearing graph points."},
    },
    "pile_group_package": {
        "category": "Calculation Package",
        "brief": "Run rigid-cap pile group analysis (CPGA stiffness method) and generate calc package; with method_6dof, tables (CSV + printed) of the force and displacement at every pile head.",
        "parameters": {
            "layout": {"type": "str", "required": False, "allowed_values": ["rectangular"], "description": "'rectangular' for auto-layout, or provide 'piles' list instead."},
            "n_rows": {"type": "int", "required": False, "description": "Rows — rectangular layout."},
            "n_cols": {"type": "int", "required": False, "description": "Columns — rectangular layout."},
            "spacing_x": {"type": "float", "required": False, "description": "X spacing (m) — rectangular."},
            "spacing_y": {"type": "float", "required": False, "description": "Y spacing (m) — rectangular."},
            "pile_diameter": {"type": "float", "required": False, "description": "Pile diameter/width (m), printed with the layout."},
            "axial_capacity_compression": {"type": "float", "required": False, "description": "Per-pile compression capacity (kN) — rectangular layout."},
            "axial_capacity_tension": {"type": "float", "required": False, "description": "Per-pile tension capacity (kN) — rectangular layout."},
            "piles": {"type": "array", "required": False, "description": "List of dicts: x, y, label, plus optional batter_x/batter_y (deg), axial_stiffness/lateral_stiffness (kN/m), axial_capacity_compression/tension (kN) — custom layout."},
            "method_6dof": {"type": "bool", "required": False, "default": False, "description": "True: 6-DOF rigid-cap analysis (batter, lateral load; needs axial_stiffness), with per-pile head forces and displacements. False: simplified vertical distribution."},
            "axial_stiffness": {"type": "float", "required": False, "description": "Pile axial stiffness (kN/m) for every pile of a rectangular layout (6-DOF)."},
            "lateral_stiffness": {"type": "float", "required": False, "description": "Pile lateral stiffness (kN/m), rectangular layout (6-DOF with Vx/Vy)."},
            "Vx": {"type": "float", "required": False, "default": 0.0, "description": "Lateral load x (kN), 6-DOF only: refused without method_6dof."},
            "Vy": {"type": "float", "required": False, "default": 0.0, "description": "Lateral load y (kN), 6-DOF only: refused without method_6dof."},
            "Vz": {"type": "float", "required": False, "default": 0.0, "description": "Vertical load (kN)."},
            "Mx": {"type": "float", "required": False, "default": 0.0, "description": "Moment about x (kN-m)."},
            "My": {"type": "float", "required": False, "default": 0.0, "description": "Moment about y (kN-m)."},
            "Mz": {"type": "float", "required": False, "default": 0.0, "description": "Torsion (kN-m), 6-DOF only: refused without method_6dof."},
            **_COMMON_PARAMS,
        },
        # A rectangular layout needs spacing_x and spacing_y (GM-110).
        "refused_parameters": {"spacing": (
            "with layout='rectangular' the package takes spacing_x and "
            "spacing_y (the same value twice for a square grid)")},
        "returns": {**_COMMON_RETURNS, "n_piles": "Number of piles.", "group_efficiency": "Group efficiency."},
    },
    "sheet_pile_package": {
        "category": "Calculation Package",
        "brief": "Run sheet pile wall analysis and generate calc package, with a table (CSV + printed) of pressure, shear and moment along the wall.",
        "parameters": {
            "excavation_depth": {"type": "float", "required": True, "description": "Excavation depth (m)."},
            "soil_layers": {"type": "array", "required": True, "description": "List of dicts: thickness, unit_weight, friction_angle, cohesion; optional wall_friction_deg (used with pressure_method='coulomb'), description. Same layer dicts and aliases (gamma, phi, c, cu, delta, wall_friction) as sheet_pile.cantilever_wall."},
            "wall_type": {"type": "str", "required": False, "default": "cantilever", "allowed_values": ["cantilever", "anchored"], "description": "Wall type."},
            "anchor_depth": {"type": "float", "required": False, "description": "Anchor depth (m) — anchored walls."},
            "surcharge": {"type": "float", "required": False, "default": 0.0, "description": "Surface surcharge (kPa)."},
            "FOS_passive": {"type": "float", "required": False, "default": 1.5, "description": "FOS on passive resistance."},
            "pressure_method": {"type": "str", "required": False, "default": "rankine", "allowed_values": ["rankine", "coulomb"], "description": "Earth pressure theory."},
            "gwt_depth_active": {"type": "float", "required": False, "description": "Groundwater depth on the retained side (m)."},
            "gwt_depth_passive": {"type": "float", "required": False, "description": "Groundwater depth on the excavation side (m)."},
            **_COMMON_PARAMS,
        },
        "returns": {**_COMMON_RETURNS, "embedment_m": "Required embedment.", "max_moment_kNm_per_m": "Max moment."},
    },
}

# The older names the ground improvement package still reads (GM-110).
declare_aliases(METHOD_INFO, {"ground_improvement_package": {
    "diameter": "column_diameter", "E_pier": "E_column",
    "surcharge": "surcharge_kPa", "target_consolidation": "target_U",
    "N1_before": "initial_N_spt", "N1_target": "target_N_spt"}})

# When each default applies (GM-119), as in the analysis tools: the seismic
# package's settings to its own sub-analysis, the settlement package's to
# its method and inputs, the slope report's to its kind of search and its
# probabilistic annex.
_CIRCLES_GRID = ["circular", "entry_exit", "pso"]
_RANDOM_POLYLINES = ["noncircular", "pso", "weak_layer"]
declare_applies_when(METHOD_INFO, {
    "seismic_package": {
        "kv": {"analysis_type": ["seismic_earth_pressure"]},
        "KA": {"analysis_type": ["seismic_earth_pressure"]},
        "M": {"analysis_type": ["liquefaction"]},
    },
    "settlement_package": {
        "nu": {"immediate_method": ["elastic"], "Es_immediate": "set"},
        "time_years_schmertmann": {"immediate_method": ["schmertmann"]},
        "stress_method": {"consolidation_layers": "set"},
        "drainage": {"cv": "set", "consolidation_layers": "set",
                     "Hdr": "unset"},
        "e0_secondary": {"C_alpha": "set", "t_secondary": "set"},
        "t_secondary": {"C_alpha": "set"},
    },
    "slope_report_package": {
        "nx": {"surface_type": _CIRCLES_GRID},
        "ny": {"surface_type": _CIRCLES_GRID},
        "n_trials": {"surface_type": _RANDOM_POLYLINES},
        "n_points": {"surface_type": _RANDOM_POLYLINES + ["noncircular_de"]},
        "fosm": {"variables": "set"},
        "monte_carlo": {"variables": "set"},
        "n": {"variables": "set", "monte_carlo": [True]},
        "research_surface": {"variables": "set", "monte_carlo": [True]},
    },
})
