"""Calc package adapter — flat dict → analysis → generate_calc_package → summary dict.

Generates Mathcad-style HTML/LaTeX/PDF calculation packages for 13 analysis
modules.  A package runs its analysis tool: it takes the tool's parameters
and runs the tool's own code path (``_run_tool``, GM-141), then calls
generate_calc_package() with the result and the inputs that run used.  The
response includes a file path and key results (never the full HTML, which
would exceed the 8000-char ReAct truncation limit).

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
import math
import os
import re
from datetime import datetime

from funhouse_agent.adapters import (declare_applies_when, declare_when_absent,
                                     require_params,
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


def _figure_file_name(out_dir: str, prefix: str, index: int, title: str,
                      taken=()) -> str:
    """``[<prefix>_]<title slug>.png``: a short name from the figure's
    title (GM-171: figures were named ``f1_<time>.png`` when the folder was
    deep), unique among ``taken`` (this package's names) and the files
    already in ``out_dir`` (``_2``, ``_3`` ...), with the full path under
    :data:`_MAX_FIGURE_PATH`: the slug is cut to fit at a word, and is
    ``fig<i>`` when under 8 characters would be left."""
    lead = f"{prefix}_" if prefix else ""
    # what the slug may take: the path cap less the folder, a separator,
    # a "_99" suffix and ".png"
    room = (_MAX_FIGURE_PATH - len(os.path.abspath(out_dir)) - 1 - 3 - 4
            - len(lead))
    full = slug = _slug(title, 32)
    if len(slug) == 32 and "_" in slug:            # end on a whole word
        slug = slug.rsplit("_", 1)[0]
    if len(slug) > room:
        room = max(room, 0)
        cut = slug[:room].rstrip("_")
        whole = slug[room] == "_" or "_" not in cut     # ends on a word
        word = cut if whole else cut.rsplit("_", 1)[0]
        slug = word if len(word) >= 8 else cut
    base = lead + (slug if len(slug) >= 8 or slug == full
                   else f"fig{index}")
    taken = {t.lower() for t in taken}
    name, n = f"{base}.png", 1
    while (name.lower() in taken
           or os.path.exists(os.path.join(out_dir, name))):
        n += 1
        name = f"{base}_{n}.png"
    return name


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
    prefix = (_slug(opts["name_prefix"], 24) if opts.get("name_prefix")
              else "")
    names = []

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
        name = _figure_file_name(
            out_dir, prefix, i,
            getattr(fig, "title", "") or getattr(fig, "caption", "")
            or f"{module} figure {i}", names)
        names.append(name)
        path = os.path.join(out_dir, name)
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
                    analysis_type: str, extra: dict = None,
                    run: "_ToolRun" = None, package: str = None) -> dict:
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
    if not _statements(params, package)[1]:
        # the call names no input as the user's, so no choice the module
        # calls given is the user's either (GM-167)
        from calc_package.tables import package_choices
        for c in package_choices(result, analysis):
            if isinstance(c, dict) and c.get("how") == "given":
                c["how"] = "assumed"

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
        # given and assumed kept apart in the package (GM-155)
        given_inputs=_given_inputs(params, run, package),
        assumed_inputs=_assumed_inputs(params, run, result, analysis,
                                       package),
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
        # the package's own figures are drawn to print at 7 pt or more
        # (GM-186): one that does not is a regression, and says so, as the
        # module's figure (reported, not the caller's to redraw)
        figure_warnings = _figure_warnings(render_report.get("figures"),
                                           package=True)
        if figure_warnings:
            response["figure_warnings"] = figure_warnings
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
    if run is not None and run.defaults_used:
        # the defaults the tool's run left in force (GM-141)
        response["defaults_used"] = dict(run.defaults_used)
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
# A package runs its analysis tool (GM-141)
# ---------------------------------------------------------------------------
# Each package below takes its inputs as its analysis tool takes them (the
# tool's METHOD_INFO: names, aliases, required inputs, defaults and units),
# runs the tool's own code path (``run_<tool>`` in the tool's adapter) and
# prints the result with the inputs that run used. Until GM-141 each
# package re-implemented its tool and drifted from it: units documented in
# Pa where the tool takes kPa, a pile layout dropped, other defaults.
#
# Per package: the tool adapter; the tool run per value of the selector
# input (no selector: one tool); the names the package took before that its
# tool takes under another name, kept (per selector value); keys of list
# items likewise; the package's own inputs beside the tool's; and the
# older names it no longer takes, with the reason.

_PACKAGE_TOOLS = {
    "bearing_capacity_package": {
        "adapter": "bearing_capacity",
        "tools": {None: "bearing_capacity_analysis"},
    },
    "lateral_pile_package": {
        "adapter": "lateral_pile",
        "tools": {None: "lateral_pile_analysis"},
        "older": {"soil_layers": "layers"},
        "older_item_keys": {"layers": {"py_model": "model"}},
    },
    "settlement_package": {
        "adapter": "settlement",
        "tools": {None: "combined_settlement_analysis"},
        "older": {"Es_immediate": "Es", "footing_shape": "shape"},
    },
    "axial_pile_package": {
        "adapter": "axial_pile",
        "tools": {None: "axial_pile_capacity"},
        "older": {"soil_layers": "layers"},
    },
    "drilled_shaft_package": {
        "adapter": "drilled_shaft",
        "tools": {None: "drilled_shaft_capacity"},
        "older": {"soil_layers": "layers"},
    },
    "downdrag_package": {
        "adapter": "downdrag",
        "tools": {None: "downdrag_analysis"},
        "older": {"soil_layers": "layers"},
    },
    "seismic_package": {
        "adapter": "seismic_geotech",
        "selector": ("analysis_type", "site_classification",
                     "Which analysis to run and report: the "
                     "site_classification, seismic_earth_pressure or "
                     "liquefaction_evaluation tool."),
        "tools": {"site_classification": "site_classification",
                  "seismic_earth_pressure": "seismic_earth_pressure",
                  "liquefaction": "liquefaction_evaluation"},
        "selector_values": {"liquefaction_evaluation": "liquefaction"},
        "older": {"liquefaction": {
            "layer_depths": "depths", "layer_N160": "N160",
            "layer_FC": "FC", "layer_gamma": "gamma", "M": "magnitude"}},
    },
    "retaining_wall_package": {
        "adapter": "retaining_walls",
        "selector": ("wall_type", "cantilever",
                     "The wall to analyse and report: 'cantilever' (the "
                     "cantilever_wall tool). An MSE wall has no package; "
                     "run the mse_wall tool."),
        "tools": {"cantilever": "cantilever_wall"},
        "selector_values": {"cantilever_wall": "cantilever"},
        # Where the shear key is drawn (GM-216): the sliding check takes
        # its passive on the plane below the toe over the key's depth,
        # wherever the key is, so its place is the figure's only
        "own": {"key_distance_from_toe": {
            "type": "float", "required": False,
            "description": "Distance from the toe to the front face of the "
                           "shear key (m), for the wall section figure "
                           "only: the sliding check takes the passive "
                           "resistance on the plane below the toe over the "
                           "key's depth, wherever the key is. Not given: "
                           "the key is drawn under the stem, as UFC "
                           "3-220-20 Fig. 4-29 and GEC-6 Fig. 5-22 draw "
                           "it, and the figure says so.",
            "applies_when": {"has_shear_key": [True], "key_depth": "set"},
            "when_absent": {
                "case": "under the stem",
                "basis": "no key position given: the wall section draws "
                         "the key's front face under the stem's, as UFC "
                         "3-220-20 Fig. 4-29 and GEC-6 Fig. 5-22 draw it, "
                         "and says so"}},
            "key_width": {
                "type": "float", "required": False,
                "description": "Width of the shear key (m), for the wall "
                               "section figure only. Not given: the "
                               "stem's thickness at its base, as UFC "
                               "3-220-20 Fig. 4-29 draws it.",
                "applies_when": {"has_shear_key": [True],
                                 "key_depth": "set"},
                "when_absent": {
                    "case": "the stem's thickness at its base",
                    "basis": "no key width given: the wall section draws "
                             "the key as wide as the stem, as UFC "
                             "3-220-20 Fig. 4-29 draws it"}}},
        "not_packaged": {"mse": (
            "the MSE steps work the earth pressure out again from "
            "phi_backfill, level backfill only, where the mse_wall tool "
            "takes the retained fill's own phi, a sloping backfill and "
            "traffic; run the mse_wall tool")},
        "older": {"cantilever": {
            "FOS_sliding": "FOS_sliding_required",
            "FOS_overturning": "FOS_overturning_required"}},
    },
    "ground_improvement_package": {
        "adapter": "ground_improvement",
        "selector": ("method", "wick_drains",
                     "Which method to analyse and report: the wick_drains, "
                     "aggregate_piers, surcharge_preloading ('surcharge') "
                     "or vibro_compaction ('vibro') tool."),
        "tools": {"wick_drains": "wick_drains",
                  "aggregate_piers": "aggregate_piers",
                  "surcharge": "surcharge_preloading",
                  "vibro": "vibro_compaction"},
        "selector_values": {"surcharge_preloading": "surcharge",
                            "vibro_compaction": "vibro"},
        "older": {"aggregate_piers": {"E_pier": "E_column"},
                  "surcharge": {"target_consolidation": "target_U"},
                  "vibro": {"N1_before": "initial_N_spt",
                            "N1_target": "target_N_spt"}},
    },
    "wave_equation_package": {
        "adapter": "wave_equation",
        "tools": {None: "bearing_graph"},
    },
    "pile_group_package": {
        "adapter": "pile_group",
        "selector": ("method_6dof", False,
                     "False: the pile_group_simple tool (vertical load and "
                     "moments on a rigid cap); True: the pile_group_6dof "
                     "tool (all six load components; needs the pile "
                     "stiffnesses)."),
        "tools": {False: "pile_group_simple", True: "pile_group_6dof"},
        "own": {"pile_diameter": {
            "type": "float", "required": False,
            "description": "Pile diameter or width (m). With n_rows, n_cols "
                           "and spacing, the package adds the "
                           "Converse-Labarre group efficiency, as the "
                           "group_efficiency tool works it out. The plan "
                           "view draws the piles to scale at it.",
            "when_absent": {
                "case": "no diameter",
                "basis": "the plan view draws the piles as symbols, not "
                         "to scale, and says so; no group efficiency"}},
                "north_deg": {
            "type": "float", "required": False,
            "description": "Direction of north on the plan view, degrees "
                           "clockwise from the cap's +y axis (up on the "
                           "plan), -360 to 360, as the plan the layout "
                           "came from shows it. Not given: north is drawn "
                           "up and the figure says so.",
            "when_absent": {
                "case": "north drawn up",
                "basis": "no north given: the plan view draws north "
                         "straight up and says so"}},
                "cap_outline": {
            "type": "array", "required": False,
            "description": "The pile cap's outline on the plan view: its "
                           "corners [[x, y], ...] in the cap's x, y axes "
                           "(m, the axes the pile coordinates use), at "
                           "least three, drawn closed. For the figure "
                           "only. Not given: no cap is drawn and the "
                           "figure says so.",
            "when_absent": {
                "case": "not drawn",
                "basis": "no cap outline given: the plan view shows the "
                         "piles and the cap's axes, and says so"}},
                "property_line": {
            "type": "array", "required": False,
            "description": "The property line on the plan view: its points "
                           "[[x, y], ...] in the cap's x, y axes (m), at "
                           "least two, drawn dash-dot as given (repeat the "
                           "first point last to close it). For the figure "
                           "only. Not given: none is drawn and the figure "
                           "says so.",
            "when_absent": {
                "case": "not drawn",
                "basis": "no property line given: the plan view draws "
                         "none, and says so"}}},
        "dropped": {"layout": {
            "reason": "the layout is rectangular when n_rows and n_cols "
                      "are given, else the piles list",
            "accepts": ["rectangular"]}},
    },
    "sheet_pile_package": {
        "adapter": "sheet_pile",
        "selector": ("wall_type", "cantilever",
                     "Which wall to analyse and report: the cantilever_wall "
                     "or anchored_wall tool."),
        "tools": {"cantilever": "cantilever_wall",
                  "anchored": "anchored_wall"},
        "selector_values": {"cantilever_wall": "cantilever",
                            "anchored_wall": "anchored"},
        "older": {"cantilever": {"soil_layers": "layers"},
                  "anchored": {"soil_layers": "layers"}},
    },
    "soe_package": {
        "adapter": "soe",
        "selector": ("wall_type", "cantilever",
                     "The excavation wall to analyse and report: "
                     "'cantilever' (the cantilever_excavation tool), "
                     "'braced' (braced_excavation: struts, rakers or "
                     "anchors at given support levels) or 'fhwa_anchored' "
                     "(fhwa_apparent_pressure_anchored_wall)."),
        "tools": {"cantilever": "cantilever_excavation",
                  "braced": "braced_excavation",
                  "fhwa_anchored": "fhwa_apparent_pressure_anchored_wall"},
        "selector_values": {"cantilever_excavation": "cantilever",
                            "braced_excavation": "braced",
                            "fhwa": "fhwa_anchored",
                            "fhwa_apparent_pressure_anchored_wall":
                                "fhwa_anchored"},
    },
    "reliability_package": {
        "adapter": "reliability_adapter",
        "selector": ("engine", "fosm",
                     "Which reliability engine to run and report: 'fosm' "
                     "(Taylor series), 'pem' (Rosenblueth point estimates), "
                     "'monte_carlo' or 'form' (the tool of that name)."),
        "tools": {"fosm": "fosm", "pem": "pem",
                  "monte_carlo": "monte_carlo", "form": "form"},
        "selector_values": {"mc": "monte_carlo", "monte carlo": "monte_carlo",
                            "taylor": "fosm"},
    },
    "pavement_design_package": {
        "adapter": "pavement_design_adapter",
        "selector": ("design_type", "flexible",
                     "Which procedure to run and report: AASHTO 1993 "
                     "flexible or rigid (flexible_pavement_design, "
                     "rigid_pavement_design), UFC 3-250-01 flexible_ufc or "
                     "rigid_ufc, or compare (compare_pavement_methods)."),
        "tools": {"flexible": "flexible_pavement_design",
                  "rigid": "rigid_pavement_design",
                  "flexible_ufc": "flexible_pavement_design_ufc",
                  "rigid_ufc": "rigid_pavement_design_ufc",
                  "compare": "compare_pavement_methods"},
        "selector_values": {"ufc": "flexible_ufc",
                            "ufc_flexible": "flexible_ufc",
                            "ufc_rigid": "rigid_ufc",
                            "comparison": "compare"},
    },
}


def _tool_adapter(package: str):
    """The adapter module of ``package``'s analysis tool."""
    import importlib
    return importlib.import_module(
        "funhouse_agent.adapters." + _PACKAGE_TOOLS[package]["adapter"])


def _older_names(spec: dict, value) -> dict:
    """The older names a package takes for the tool run at ``value``."""
    older = spec.get("older", {})
    return older.get(value, {}) if "selector" in spec else older


class _ToolRun:
    """A package's run of its tool: the selector value, the tool, the
    parameters it was given, what its runner returned, the package's own
    inputs, the defaults the run left in force (the tool's, and the
    selector's when it was left out) and the tool's declared parameters."""

    def __init__(self, value, tool, params, out, own, defaults_used,
                 declared=None, package=None):
        self.value, self.tool, self.params = value, tool, params
        self.out, self.own = out, own
        self.defaults_used = defaults_used
        self.declared = declared or {}
        self.package = package


def _stated_assumptions(value, package: str = None) -> dict:
    """A package's ``assumptions`` (GM-155): {input name or statement:
    basis}; a list of names is taken with no basis. Refused in any other
    shape. With ``package`` (GM-167): an input given by another spelling
    the package reads is put under its own name; a key written as an
    input's name (``water_table``) that is not one, a word that is a
    misspelt input (``cohesoin``) and a header field are refused, only
    those, with a did-you-mean; any other key is a statement in words
    ("groundwater", "drained", "c = 0 for sand"), printed as its own
    row."""
    if value is None or value == "" or value == {} or value == []:
        return {}
    if isinstance(value, dict):
        out = {str(k): "" if v is None else str(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)) and all(isinstance(v, str)
                                                  for v in value):
        out = {v: "" for v in value}
    else:
        raise ValueError("assumptions: give {input name: its basis} for the "
                         "inputs you pass that the user did not give.")
    if package is None:
        return out
    from funhouse_agent.adapters import suggest_params
    inputs, aliases = _package_inputs(package), _package_aliases(package)
    valid = [n for n in inputs if n not in _COMMON_PARAMS]
    resolved, bad = {}, {}
    for key, basis in out.items():
        name = aliases.get(key, key)
        if name in _COMMON_PARAMS:
            bad[key] = None
        elif name in inputs:
            resolved[name] = basis
        else:
            near = suggest_params(key, valid)
            if near or _NAME.fullmatch(key):
                bad[key] = near
            else:
                resolved[key] = basis
    if bad:
        _refuse(package, "assumptions", bad, [k for k in out if k not in bad],
                words=True)
    return resolved


#: A key written as an input's name: a word joined to another by "_".
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+")


def _named_given(value, package: str = None, passed=None) -> dict:
    """A package's ``given`` (GM-167): the inputs the user gave, as
    ``{name: the user's words, or ""}``. A list of names, one name, or
    ``{name: the user's words}``; with ``package``, a name given by
    another spelling is put under its own, and a name that is not an input
    is refused with a did-you-mean. With ``passed`` (the inputs the call
    passes, by their own names), a name the call does not pass is refused
    too, so the user's value never drops out of Given Inputs to print as a
    default, and each refusal lists the inputs passed."""
    if value is None or value == "" or value == [] or value == {}:
        return {}
    if isinstance(value, str):
        value = [value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        words = {k: "" if w is None else str(w).strip()
                 for k, w in value.items()}
    elif isinstance(value, (list, tuple, set)) \
            and all(isinstance(v, str) for v in value):
        words = dict.fromkeys(value, "")
    else:
        raise ValueError("given: give the names of the inputs you pass that "
                         "the user gave, e.g. [\"width\", \"friction_angle\"]"
                         ", or {input name: the user's own words}.")
    if package is None:
        return words
    from funhouse_agent.adapters import suggest_params
    inputs, aliases = _package_inputs(package), _package_aliases(package)
    valid = [n for n in inputs if n not in _COMMON_PARAMS]
    passing = ("" if passed is None else " The inputs this call passes: "
               + ", ".join(sorted(passed)) + ".")
    names, bad = {}, {}
    for key, said in words.items():
        name = aliases.get(key, key)
        if name in inputs and name not in _COMMON_PARAMS:
            names[name] = said
        else:
            bad[key] = (None if name in _COMMON_PARAMS
                        else suggest_params(key, valid))
    if bad:
        _refuse(package, "given", bad, [k for k in words if k not in bad],
                extra=passing)
    missing = sorted(k for k in words if passed is not None
                     and aliases.get(k, k) not in passed)
    if missing:
        raise ValueError(
            f"{package} given: {missing} named but not passed. Pass the "
            "value the user gave for each as well, so Given Inputs prints "
            "it." + passing)
    return names


def _refuse(package: str, field: str, bad: dict, fine: list,
            words: bool = False, extra: str = ""):
    """Refuse the keys of ``field`` in ``bad`` ({key: its near inputs, or
    None for a header field}) in the words of an unknown input (GM-110),
    naming the keys that were fine so the next call sends them again;
    ``extra`` is added at the end."""
    header = sorted(k for k, near in bad.items() if near is None)
    unknown = sorted(k for k in bad if k not in header)
    parts = []
    if unknown:
        hints = [f"{k} -> {' or '.join(bad[k])}" for k in unknown if bad[k]]
        parts.append(f"unknown parameter(s) {unknown}."
                     + (f" Did you mean: {'; '.join(hints)}?" if hints
                        else ""))
    if header:
        parts.append(f"{header}: header fields, not inputs.")
    msg = f"{package} {field}: " + " ".join(parts)
    if words:
        msg += (" An assumption that is not an input is written in words, "
                "e.g. {\"drained\": \"long-term condition\"}.")
    if fine:
        msg += (" The others (" + ", ".join(map(repr, fine)) + ") are "
                "fine: send them again with these corrected.")
    raise ValueError(msg + extra)


def _checked_statements(params: dict, package: str):
    """``(assumptions, given)`` of a package call, checked (GM-155,
    GM-167): ``given`` names only inputs the call passes, and an input
    both stated as an assumption and named as the user's is refused."""
    aliases = _package_aliases(package)
    passed = {aliases.get(k, k) for k, v in params.items()
              if k not in _COMMON_PARAMS and v is not None}
    stated = _stated_assumptions(params.get("assumptions"), package)
    named = _named_given(params.get("given"), package, passed)
    both = sorted(set(stated) & set(named))
    if both:
        raise ValueError(f"{package}: {both} in both assumptions and "
                         "given; an input is either the user's or "
                         "assumed.")
    return stated, named


def _package_inputs(package: str) -> dict:
    """``package``'s declared inputs, {name: spec}."""
    return dict((METHOD_INFO.get(package) or {}).get("parameters") or {})


def _package_aliases(package: str) -> dict:
    """The other spellings of ``package``'s inputs it reads."""
    return dict((METHOD_INFO.get(package) or {}).get("parameter_aliases")
                or {})


def _input_words(name: str, declared: dict) -> str:
    """An input as the package prints it: the first sentence of its
    description (plain words, as the Method Choices table prints them),
    else its name."""
    desc = str((declared.get(name) or {}).get("description") or "").strip()
    if not desc:
        return name
    first = re.match(r"(.+?[^A-Z]\.)\s+(?=[A-Z])", desc)
    text = (first.group(1) if first else desc).rstrip(".").strip()
    text = re.split(r" — |; |,? e\.g\.|,? \(e\.g\.", text)[0].strip()
    if len(text) > 80:
        text = text[:77].rsplit(" ", 1)[0] + "..."
    return text


def _package_declared(package: str = None, run: "_ToolRun" = None) -> dict:
    """The inputs a package's rows are worded from: the package's own,
    then its tool's."""
    package = package or getattr(run, "package", None)
    declared = _package_inputs(package) if package else {}
    if run is not None:
        declared.update(run.declared)
    return declared


def _statements(params: dict, package: str = None):
    """``(assumptions, given)`` of a call, checked when the package is
    known."""
    if package:
        return _checked_statements(params, package)
    return (_stated_assumptions(params.get("assumptions")),
            _named_given(params.get("given")))


def _passed_inputs(params: dict, run: "_ToolRun" = None,
                   package: str = None, declared: dict = None) -> dict:
    """What a package call passed for the analysis, in the order passed:
    the selector, the package's own inputs and the tool's, without the
    header fields. ``declared`` gets the selector's words."""
    if run is None:
        return {k: v for k, v in params.items() if k not in _COMMON_PARAMS}
    selector = (_PACKAGE_TOOLS.get(package, {}).get("selector") or [None])[0]
    passed = {}
    if selector and selector in params:
        passed[selector] = params[selector]
        if declared is not None:
            # "The wall to analyse and report: 'cantilever' (...)": the
            # words before the list of tools, where they make a phrase
            words = _input_words(selector, declared)
            head = words.split(": ", 1)[0]
            declared[selector] = {"description": head if head != words
                                  and len(head.split()) >= 3 else selector}
    passed.update(run.own)
    passed.update(run.params)
    return passed


def _given_inputs(params: dict, run: "_ToolRun" = None,
                  package: str = None) -> list:
    """The inputs the user gave, for the package's Given Inputs section
    (GM-155, GM-167): the inputs the call passed and named in ``given``,
    each ``{input, value, words}`` (``words``: the user's own, when the
    call gave them). Any other input passed is assumed: an input the call
    does not say is the user's is never printed as given."""
    package = package or getattr(run, "package", None)
    named = _statements(params, package)[1]
    declared = _package_declared(package, run)
    passed = _passed_inputs(params, run, package, declared)
    return [{"input": _input_words(name, declared), "value": value,
             "words": named[name]}
            for name, value in passed.items() if name in named]


#: The basis of an input the call passed and did not name as the user's
#: (GM-167).
_PASSED_BASIS = "passed, not stated as given"


def _assumed_inputs(params: dict, run: "_ToolRun" = None, result=None,
                    analysis=None, package: str = None) -> list:
    """The inputs that were not given, for the package's Assumed Inputs
    section (GM-155): the inputs and statements the caller stated as
    assumptions (``assumptions``), every other input it passed and did not
    name as the user's in ``given`` (GM-167; not knowing whose an input is,
    the package takes it as assumed), then the defaults the tool's run
    left in force (``defaults_used``): a left-out input with no default as
    the case it stands for, or as the other input's value it takes
    (GM-165); each ``{input, value, basis}``."""
    from calc_package.tables import package_choices
    package = package or getattr(run, "package", None)
    stated, named = _statements(params, package)
    used = dict(run.defaults_used) if run is not None else {}
    declared = _package_declared(package, run)
    given = _passed_inputs(params, run, package, declared)
    try:
        chosen = " ".join(str(c.get("chosen", ""))
                          for c in package_choices(result, analysis)
                          if isinstance(c, dict))
    except Exception:                                      # noqa: BLE001
        chosen = ""

    def value_of(name):
        if name in given:
            return given[name]
        if name in params:
            return params[name]
        case = (declared.get(name) or {}).get("when_absent")
        return case["case"] if case else used.get(name)

    out = []
    for name, basis in stated.items():
        if name in declared:
            out.append({"input": _input_words(name, declared),
                        "value": value_of(name),
                        "basis": basis or "stated as an assumption"})
        else:   # a statement that is not an input ("drained")
            out.append({"input": name, "value": "",
                        "basis": basis or "stated as an assumption"})
    for name, value in given.items():
        if name not in stated and name not in named:
            out.append({"input": _input_words(name, declared),
                        "value": value, "basis": _PASSED_BASIS})
    for name, value in used.items():
        if name in stated:
            continue
        spec = declared.get(name) or {}
        case, source = spec.get("when_absent"), spec.get("default_from")
        if case:
            out.append({"input": _input_words(name, declared),
                        "value": case["case"],
                        "basis": "not given: " + case["basis"]})
            continue
        if source:
            basis = "not given: taken equal to the " + _lower_first(
                re.sub(r"\s*\([^()]*\)$", "",     # its unit is this one's
                       _input_words(source, declared)))
            from funhouse_agent.dispatch import absent_case, case_of
            if value == absent_case(spec, declared):   # left out too
                case = case_of(spec, declared)
                value = case["case"]
                basis += ", itself not given: " + case["basis"]
            out.append({"input": _input_words(name, declared),
                        "value": value, "basis": basis})
            continue
        in_choices = re.search(rf"\b{re.escape(name)}\b", chosen)
        out.append({"input": _input_words(name, declared), "value": value,
                    "basis": "the analysis's default" + (
                        "; its basis is under Method Choices"
                        if in_choices else "")})
    return out


def _lower_first(text: str) -> str:
    """``text`` with its first letter lower case, unless it starts an
    abbreviation or a symbol ("Backfill friction angle" -> "backfill
    friction angle"; "FOS" kept)."""
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def _run_tool(package: str, params: dict) -> _ToolRun:
    """Run ``package``'s analysis tool on the package's parameters.

    The header fields and the selector are taken off, a name the package
    took before is put under the tool's, and the tool's own check of the
    names runs (BACKLOG GM-110), so the package refuses what its tool
    refuses, with the tool's did-you-mean; an input the tool would use is
    never dropped. Then ``run_<tool>``, the tool's own code path, runs."""
    from funhouse_agent.adapters import refused_reason, without_unused
    from funhouse_agent.dispatch import _check_parameter_names

    spec = _PACKAGE_TOOLS[package]
    # checked before a run (GM-155, GM-167)
    named = _checked_statements(params, package)[1]
    p = {k: v for k, v in params.items() if k not in _COMMON_PARAMS}
    value = None
    if "selector" in spec:
        name, default, _ = spec["selector"]
        given = p.pop(name, default)
        value = given
        if isinstance(given, str):
            value = given.strip().lower()
            value = spec.get("selector_values", {}).get(value, value)
        why = spec.get("not_packaged", {}).get(value)
        if why:
            raise ValueError(f"{package}: no package for {name}={given!r}: "
                             f"{why}.")
        if value not in spec["tools"]:
            raise ValueError(
                f"{package}: {name} must be one of {list(spec['tools'])}, "
                f"got {given!r}.")
    tool = spec["tools"][value]
    own = {k: p.pop(k) for k in list(p) if k in spec.get("own", {})}
    dropped = spec.get("dropped", {})
    p = without_unused(p, dropped)
    hits = sorted(k for k in p if k in dropped)
    if hits:
        why = {}
        for k in hits:
            why.setdefault(refused_reason(dropped[k]), []).append(k)
        raise ValueError(
            f"{package} does not take {hits}: "
            + "; ".join(r if len(why) == 1 else f"{', '.join(ks)}: {r}"
                        for r, ks in why.items()) + ".")
    renamed = {}
    for old, new in _older_names(spec, value).items():
        if old in p:
            if new in p:
                raise ValueError(f"{package}: give '{new}' or its older "
                                 f"name '{old}', not both.")
            p[new] = p.pop(old)
            renamed[new] = old

    item_renamed = {}
    for key, renames in spec.get("older_item_keys", {}).items():
        if isinstance(p.get(key), list):
            for item in p[key]:
                if isinstance(item, dict):
                    item_renamed.update(
                        {renames[k]: k for k in item if k in renames
                         and renames[k] not in item})
            p[key] = [
                {(renames[k] if k in renames and renames[k] not in item
                  else k): v for k, v in item.items()}
                if isinstance(item, dict) else item for item in p[key]]

    def as_given(msg):
        # an input given by its older name is named so in a message, and
        # another tool of the package by the selector value that runs it
        for new, old in renamed.items():
            msg = re.sub(rf"\b{re.escape(new)}\[", f"{old}[", msg)
        for new, old in item_renamed.items():
            msg = msg.replace(f"'{new}'", f"'{old}'")
        if "selector" in spec:
            for v, t in spec["tools"].items():
                if t != tool:
                    msg = re.sub(rf"\b{re.escape(t)}\b",
                                 f"{spec['selector'][0]}={v!r} ({t})", msg)
        return msg
    label = f"{package} ({tool})"
    mod = _tool_adapter(package)
    try:
        _check_parameter_names(mod, tool, p)
    except ValueError as exc:
        msg = as_given(str(exc))
        if msg.startswith(tool):
            msg = msg[len(tool):].lstrip(": ")
        if "selector" in spec:
            # which input picks the tool that would take it
            name = spec["selector"][0]
            msg += (f" ({package} runs {tool} for {name}={value!r}; "
                    + "; ".join(f"{name}={v!r} runs {t}"
                                for v, t in spec["tools"].items()
                                if t != tool) + ".)")
        raise ValueError(f"{label}: {msg}") from None
    # the run in the tool's own call context, so its choices and its
    # defaults_used are the tool's (dispatch does the same for the tool)
    from funhouse_agent.dispatch import (
        _declared, _defaults_used, _given_names, _left_out)
    from geotech_common.choices import call_inputs
    given = set(p)
    # the inputs passed that the call does not name as the user's: the
    # module's choices say "assumed" for them, not "given" (GM-167)
    declared = _declared(mod, tool)[0]
    tool_name = {old: new for new, old in renamed.items()}
    assumed = given - {tool_name.get(k, k) for k in named}
    with call_inputs(_given_names(mod, tool, given),
                     _left_out(mod, tool, given, p),
                     assumed=assumed & given) as inputs:
        try:
            out = getattr(mod, f"run_{tool}")(p, method=label)
        except ValueError as exc:
            msg = as_given(str(exc))
            if msg == str(exc):
                raise
            raise ValueError(msg) from None
    used = {}
    if "selector" in spec and spec["selector"][0] not in params:
        used[spec["selector"][0]] = spec["selector"][1]
    used.update(_defaults_used(mod, tool, given, p, inputs.reported))
    used.update(_own_defaults_used(spec.get("own", {}), own, p, declared))
    return _ToolRun(value, tool, p, out, own, used,
                    declared=declared, package=package)


def _own_defaults_used(own_specs: dict, own: dict, params: dict,
                       declared: dict) -> dict:
    """The package's own inputs (``own``: not its tool's, such as the plan
    view's ``north_deg``) that the call left out, as ``defaults_used``
    gives the tool's (GM-218): name -> its default, or the case a left-out
    input with no default stands for (``when_absent``), where its
    ``applies_when`` holds for the call (the tool's inputs in force, then
    the package's own)."""
    from funhouse_agent.dispatch import (
        _values_in_force, absent_case, applies)
    every = {**declared, **own_specs}
    values = _values_in_force(every, {}, {**params, **own})
    out = {}
    for name, spec in own_specs.items():
        if name in own:
            continue
        value = spec.get("default")
        if value is None:
            value = absent_case(spec, every) or None
        if value not in (None, "") and applies(spec.get("applies_when"),
                                               values):
            out[name] = value
    return out


def _when(condition, selector: str, value):
    """``condition`` (an ``applies_when``) and the selector at ``value``."""
    if condition is False:
        return False
    if condition is None:
        return {selector: [value]}
    if isinstance(condition, list):
        return [{**c, selector: [value]} for c in condition]
    return {**condition, selector: [value]}


def _union_spec(selector: str, entries, n_tools: int) -> dict:
    """One parameter spec of a package that runs several tools, from each
    tool's spec of it (``entries``: (selector value, spec)): its default
    applies only where a tool that takes it runs (GM-119)."""
    import copy
    specs = [s for _, s in entries]
    out = copy.deepcopy(specs[0])
    descriptions = []
    for value, s in entries:
        d = s.get("description", "")
        if d not in [x for _, x in descriptions]:
            descriptions.append((value, d))
    if len(descriptions) > 1:
        out["description"] = " | ".join(f"{selector}={v}: {d}"
                                        for v, d in descriptions)
    types = []
    for s in specs:
        for t in str(s.get("type", "")).split("|"):
            if t and t not in types:
                types.append(t)
    if types:
        out["type"] = "|".join(types)
    every = len(entries) == n_tools
    required = [v for v, s in entries if s.get("required")]
    out["required"] = every and len(required) == n_tools
    if required and not out["required"]:
        out["description"] = (f"Required when {selector} is "
                              + " or ".join(map(str, required)) + ". "
                              + out.get("description", ""))
    defaults = [repr(s.get("default")) for s in specs]
    if len(set(defaults)) > 1:
        # each tool's own default, said; the run lists the one it used
        out.pop("default", None)
        out["description"] = (
            "Default: " + "; ".join(
                f"{v!s} {s['default']!r}" for v, s in entries
                if s.get("default") is not None)
            + f" ({selector}). " + out.get("description", ""))
    for key in ("when_absent", "default_from"):
        # what a left-out input stands for (GM-165), where every tool that
        # takes it says the same; else the run's own list says it
        if len({repr(s.get(key)) for s in specs}) > 1:
            out.pop(key, None)
    allowed = [s.get("allowed_values") for s in specs]
    if any(a is not None for a in allowed) and len(
            {repr(a) for a in allowed}) > 1:
        out["allowed_values"] = sorted({v for a in allowed if a
                                        for v in a}, key=str)
    if every and len({repr(s.get("applies_when")) for s in specs}) == 1:
        # every tool takes it on the same condition: the selector adds
        # nothing
        out.pop("applies_when", None)
        if specs[0].get("applies_when") is not None:
            out["applies_when"] = copy.deepcopy(specs[0]["applies_when"])
    else:
        conditions = []
        for value, s in entries:
            c = _when(s.get("applies_when"), selector, value)
            if c is False:
                continue
            conditions.extend(c if isinstance(c, list) else [c])
        out["applies_when"] = (conditions[0] if len(conditions) == 1
                               else conditions or False)
    return out


def _package_parameters(package: str) -> dict:
    """A package's ``parameters``, ``parameter_aliases`` and
    ``refused_parameters``: its tool's (GM-141), one source, with the
    header fields, the selector, the package's own inputs and the older
    names it still takes. A package that runs several tools takes the
    union of their parameters (:func:`_union_spec`). A name one tool
    refuses and another takes is left to the run's own check."""
    import copy
    spec = _PACKAGE_TOOLS[package]
    info = _tool_adapter(package).METHOD_INFO
    tools = spec["tools"]
    params = {}
    selector = spec.get("selector")
    if selector:
        name, default, doc = selector
        params[name] = {
            "type": "bool" if isinstance(default, bool) else "str",
            "required": False, "default": default,
            "allowed_values": list(tools), "description": doc}
    entries = {}
    for value, tool in tools.items():
        for pname, pspec in info[tool]["parameters"].items():
            entries.setdefault(pname, []).append((value, pspec))
    for pname, found in entries.items():
        params[pname] = (_union_spec(selector[0], found, len(tools))
                         if selector else copy.deepcopy(found[0][1]))
    params.update(copy.deepcopy(spec.get("own", {})))
    params.update(_COMMON_PARAMS)
    aliases, refused = {}, {}
    for value, tool in tools.items():
        aliases.update(info[tool].get("parameter_aliases") or {})
        refused.update(info[tool].get("refused_parameters") or {})
        aliases.update(_older_names(spec, value))
    out = {"parameters": params}
    aliases = {a: c for a, c in aliases.items() if a not in params}
    if aliases:
        out["parameter_aliases"] = aliases
    refused = {k: v for k, v in refused.items() if k not in params}
    refused.update(spec.get("dropped", {}))
    if refused:
        out["refused_parameters"] = refused
    return out


# ---------------------------------------------------------------------------
# 1. Bearing Capacity
# ---------------------------------------------------------------------------

def _generate_bearing_capacity_package(params: dict) -> dict:
    run = _run_tool("bearing_capacity_package", params)
    result, analysis = run.out
    return _build_response(
        "bearing_capacity", result, analysis, params,
        run=run,
        analysis_type="Bearing Capacity",
        extra={
            "q_ultimate_kPa": round(result.q_ultimate, 1),
            "q_allowable_kPa": round(result.q_allowable, 1),
            "q_allowable_net_kPa": round(result.q_allowable_net, 1),
            "allowable_load": result.allowable_loads(),
            "factor_of_safety": result.factor_of_safety,
        },
    )


# ---------------------------------------------------------------------------
# 2. Lateral Pile
# ---------------------------------------------------------------------------

def _generate_lateral_pile_package(params: dict) -> dict:
    run = _run_tool("lateral_pile_package", params)
    result, analysis = run.out
    return _build_response(
        "lateral_pile", result, analysis, params,
        run=run,
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
    _checked_statements(params, "slope_stability_package")

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
        analysis_type="Slope Stability", package="slope_stability_package",
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
            # output + header metadata; the caller's assumptions and the
            # inputs it says the user gave (GM-155, GM-167)
            "output_path", "format", "project_name", "project_number",
            "engineer", "checker", "company", "date", "assumptions",
            "given",
        ],
        method="slope_report_package",
    )
    _checked_statements(params, "slope_report_package")

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
        extra=extra, package="slope_report_package",
    )


# ---------------------------------------------------------------------------
# 4. Settlement
# ---------------------------------------------------------------------------

def _generate_settlement_package(params: dict) -> dict:
    run = _run_tool("settlement_package", params)
    result, analysis = run.out
    return _build_response(
        "settlement", result, analysis, params,
        run=run,
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
    run = _run_tool("axial_pile_package", params)
    result, analysis = run.out
    return _build_response(
        "axial_pile", result, analysis, params,
        run=run,
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
    run = _run_tool("drilled_shaft_package", params)
    result, analysis = run.out
    return _build_response(
        "drilled_shaft", result, analysis, params,
        run=run,
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
    run = _run_tool("downdrag_package", params)
    result, analysis = run.out
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
        run=run,
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
    from seismic_geotech.results import SiteClassResult

    run = _run_tool("seismic_package", params)
    # the inputs in force, by the tool's names
    analysis = dict(run.params)
    if run.tool == "site_classification":
        site_class, vs30, n_bar, su_bar, coefficients = run.out
        # Without Ss and S1 the tool gives the site class alone; so does
        # the package (no coefficients, no design spectrum).
        result = coefficients or SiteClassResult(
            site_class=site_class, Fpga=None, Fa=None, Fv=None, Ss=None,
            S1=None)
        result.vs30, result.n_bar, result.su_bar = vs30, n_bar, su_bar
    elif run.tool == "seismic_earth_pressure":
        result, _, analysis = run.out
    else:
        result = run.out
        analysis["magnitude"] = result.magnitude

    extra = {"seismic_analysis_type": run.value}
    if getattr(result, "warnings", None):
        extra["warnings"] = list(result.warnings)
    return _build_response(
        "seismic_geotech", result, analysis, params,
        run=run,
        analysis_type="Seismic Geotechnical",
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 9. Retaining Wall
# ---------------------------------------------------------------------------

def _generate_retaining_wall_package(params: dict) -> dict:
    run = _run_tool("retaining_wall_package", params)
    result, used = run.out
    analysis = {**used, "wall_type": run.value}
    analysis.update(_shear_key_place(run.own, used.get("geom")))

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
    return _build_response(
        "retaining_walls", result, analysis, params,
        run=run,
        analysis_type="Retaining Wall",
        extra=extra,
    )


def _shear_key_place(own: dict, geom) -> dict:
    """The shear key's place for the wall section (GM-216): the given
    ``key_distance_from_toe`` and ``key_width``, each a number, the key
    under the base; refused by name otherwise. Left out, the figure draws
    it under the stem and says so."""
    out = {}
    for name in ("key_distance_from_toe", "key_width"):
        value = own.get(name)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = None
        if isinstance(value, bool) or number is None \
                or not math.isfinite(number) \
                or number < 0 or (name == "key_width" and number == 0):
            raise ValueError(
                f"retaining_wall_package: {name} must be a length in m, "
                f"{'above' if name == 'key_width' else 'at or above'} 0; "
                f"got {value!r}.")
        out[name] = number
    if not out or geom is None:
        return out
    if not (geom.has_shear_key and geom.key_depth > 0):
        names = " and ".join(sorted(out))
        raise ValueError(
            f"retaining_wall_package: the wall has no shear key for "
            f"{names}: give has_shear_key=true and key_depth, or leave "
            f"{names} out.")
    x0 = out.get("key_distance_from_toe", geom.toe_length)
    width = out.get("key_width", geom.stem_thickness_base)
    if x0 + width > geom.base_width + 1e-9:
        raise ValueError(
            f"retaining_wall_package: the shear key ({width:g} m wide, its "
            f"front face {x0:g} m from the toe) runs past the back of the "
            f"base (B = {geom.base_width:g} m); give key_distance_from_toe "
            f"+ key_width up to B.")
    return out


# ---------------------------------------------------------------------------
# 10. Ground Improvement
# ---------------------------------------------------------------------------

def _generate_ground_improvement_package(params: dict) -> dict:
    run = _run_tool("ground_improvement_package", params)
    result, used = run.out
    # the inputs in force by the module's names, and the method by the
    # package's name for it, as the steps read them
    analysis = {**used, "method": run.value}
    return _build_response(
        "ground_improvement", result, analysis, params,
        run=run,
        analysis_type="Ground Improvement",
        extra={"method": run.value},
    )


# ---------------------------------------------------------------------------
# 11. Wave Equation
# ---------------------------------------------------------------------------

def _generate_wave_equation_package(params: dict) -> dict:
    run = _run_tool("wave_equation_package", params)
    result, analysis = run.out
    return _build_response(
        "wave_equation", result, analysis, params,
        run=run,
        analysis_type="Wave Equation",
        extra={"n_points": len(result.blow_counts)},
    )


# ---------------------------------------------------------------------------
# 12. Pile Group
# ---------------------------------------------------------------------------

def _north_deg(value):
    """``north_deg`` as the plan draws it: a number of degrees, -360 to
    360, else refused by name (a text, an infinite or a NaN angle drew no
    north arrow, and 400 drew 40 degrees without a word; GM-200 review)."""
    if value is None:
        return None
    try:
        deg = float(value)
    except (TypeError, ValueError):
        deg = None
    if isinstance(value, bool) or deg is None or not math.isfinite(deg) \
            or abs(deg) > 360.0:
        raise ValueError(
            f"pile_group_package: north_deg must be the direction of north "
            f"on the plan in degrees clockwise from up, -360 to 360; got "
            f"{value!r}.")
    return deg


def _plan_points(value, name: str, least: int) -> list:
    """A line on the pile group's plan (GM-216: ``cap_outline``,
    ``property_line``) as ``[[x, y], ...]`` floats in the cap's axes, at
    least ``least`` points, each a finite pair; else refused by name."""
    points = []
    ok = isinstance(value, (list, tuple)) and len(value) >= least
    for p in (value if ok else []):
        try:
            x, y = (float(v) for v in p)
        except (TypeError, ValueError):
            ok = False
            break
        if any(isinstance(v, bool) for v in p) \
                or not (math.isfinite(x) and math.isfinite(y)):
            ok = False
            break
        points.append([x, y])
    if not ok:
        raise ValueError(
            f"pile_group_package: {name} must be [[x, y], ...], at least "
            f"{least} points in the cap's x, y axes (m); got {value!r}.")
    return points


def _generate_pile_group_package(params: dict) -> dict:
    from funhouse_agent.adapters.pile_group import _run_group_efficiency

    north = _north_deg(params.get("north_deg"))
    drawn = {name: _plan_points(params.get(name), name, least)
             for name, least in (("cap_outline", 3), ("property_line", 2))
             if params.get(name) is not None}
    run = _run_tool("pile_group_package", params)
    result, used = run.out
    analysis = {**used, "method": "6dof" if run.value else "simple",
                **drawn}
    # The Converse-Labarre efficiency, as the group_efficiency tool works
    # it out from the layout run and the diameter given. It takes one
    # spacing: where the rows and columns are spaced differently, the
    # smaller, where the piles are closer and E_g is lower.
    d = run.own.get("pile_diameter")
    sx = used.get("spacing_x", used.get("spacing"))
    sy = used.get("spacing_y", used.get("spacing"))
    spacing = min(sx, sy) if sx is not None and sy is not None else None
    extra = {"n_piles": result.n_piles, "group_efficiency": None}
    if north is not None:
        analysis["north_deg"] = north
    if d is not None:
        analysis["pile_diameter"] = d
        if (used.get("n_rows") and used.get("n_cols")
                and spacing is not None):
            analysis["efficiency_spacing"] = spacing
            analysis["efficiency"] = _run_group_efficiency({
                "n_rows": used["n_rows"], "n_cols": used["n_cols"],
                "pile_diameter": d, "spacing": spacing,
            })["converse_labarre_Eg"]
            extra["group_efficiency"] = analysis["efficiency"]
            if sx != sy:
                extra["group_efficiency_spacing_m"] = spacing
                extra["group_efficiency_note"] = (
                    f"Converse-Labarre takes one spacing; with spacing_x "
                    f"{sx:g} m and spacing_y {sy:g} m the smaller, "
                    f"{spacing:g} m, is used (the closer piles, the lower "
                    f"E_g).")
    return _build_response(
        "pile_group", result, analysis, params,
        run=run,
        analysis_type="Pile Group",
        extra=extra,
    )


# ---------------------------------------------------------------------------
# 13. Sheet Pile
# ---------------------------------------------------------------------------

def _generate_sheet_pile_package(params: dict) -> dict:
    run = _run_tool("sheet_pile_package", params)
    result, used = run.out
    analysis = {**used, "wall_type": run.value,
                "soil_layers": used["layers"]}
    extra = {
        "embedment_m": round(result.embedment_depth, 2),
        "max_moment_kNm_per_m": round(result.max_moment, 1),
    }
    if run.value == "anchored":
        extra["anchor_force_kN_per_m"] = round(result.anchor_force, 1)
    return _build_response(
        "sheet_pile", result, analysis, params,
        run=run,
        analysis_type="Sheet Pile Wall",
        extra=extra,
    )


def _generate_soe_package(params: dict) -> dict:
    run = _run_tool("soe_package", params)
    result, used = run.out
    analysis = {**used, "wall_type": run.value}
    if run.value == "fhwa_anchored":
        extra = {
            "pe_kPa": result["pe_kPa"],
            "anchor_design_loads_kN": [a["design_load_kN"]
                                       for a in result["anchors"]],
            "max_moment_kNm_per_m": result.get("max_moment_kN_m_per_m"),
        }
        if result.get("embedment_D_m") is not None:
            extra["embedment_m"] = result["embedment_D_m"]
    else:
        extra = {
            "embedment_m": result.required_embedment_m,
            "total_wall_length_m": result.total_wall_length_m,
            "max_moment_kNm_per_m": result.max_moment_kNm_per_m,
            "required_Sx_cm3": result.required_Sx_cm3,
            "converged": result.converged,
        }
        if run.value == "braced":
            extra["support_loads_kN_per_m"] = [
                r["load_kN_per_m"] for r in result.support_reactions]
        if result.notes:
            extra["notes"] = list(result.notes)
    if run.value == "fhwa_anchored" and result.get("notes"):
        extra["notes"] = list(result["notes"])
    return _build_response(
        "soe", result, analysis, params,
        run=run,
        analysis_type="Support of Excavation Wall",
        extra=extra,
    )


def _generate_reliability_package(params: dict) -> dict:
    run = _run_tool("reliability_package", params)
    result, used = run.out
    extra = {"engine": run.value}
    if run.value == "form":
        extra.update(beta=round(result.beta, 4), pf=result.pf,
                     converged=result.converged)
    elif run.value == "monte_carlo":
        extra.update(pf=result.pf, n_failed=result.n_failed, n=result.n,
                     beta_normal=round(result.beta_normal, 3))
    else:
        extra.update(beta_normal=round(result.beta_normal, 3),
                     pf_normal=result.pf_normal)
        if result.beta_lognormal is not None:
            extra.update(beta_lognormal=round(result.beta_lognormal, 3),
                         pf_lognormal=result.pf_lognormal)
    return _build_response(
        "reliability", result, used, params,
        run=run,
        analysis_type="Reliability Analysis",
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


def _figure_warnings(figures, package: bool = False) -> list:
    """What a reader of the PDF cannot read, per figure (GM-168): for any
    figure whose PNG records its lettering, lettering that prints under
    7 pt; for a figure the plotting box drew (its PNG says the width its
    lettering check measured it at, so the caller has its code), one
    printed narrower than that width; for any figure, one taller than a
    page scaled far down. The numbers are the plotting box's own. The
    canned package figures are drawn for the width they print at
    (BACKLOG GM-186), so one of theirs warned on here is printed narrower
    than that, or a regression. A picture with no record (a photo, a
    drawing, a logo) is not judged by its width.

    Each warning says whose figure it is and what the caller can do: the
    plotting box's is the caller's to redraw; a module's own figure (no
    checked width) can be given the page width in report HTML, and is
    otherwise reported, not the caller's to redraw; in a canned package
    (``package``) every figure is the module's."""
    from calc_package.latex_renderer import MIN_FIGURE_SCALE
    from geotech_common.layout_ink import MIN_PRINT_PT
    out = []
    for k, f in enumerate(figures or [], start=1):
        checked = f.get("checked_width_in")
        lettering = f.get("lettering_pt")
        why = []
        if lettering is not None and lettering < MIN_PRINT_PT - 0.05:
            why.append(f"its smallest lettering prints {lettering:g} pt, "
                       f"under {MIN_PRINT_PT:g} pt")
        if checked and f["width_in"] < checked - 0.05:
            why.append(f"it prints narrower than the {checked:g} in its "
                       "lettering is checked at")
        scaled = f.get("scaled")
        if scaled is not None and scaled < MIN_FIGURE_SCALE:
            why.append(f"it is taller than a page, so it prints at "
                       f"{scaled:.0%} of its width")
        if not why:
            continue
        if package:
            fix = ("It is the package's own figure, drawn by the module: "
                   "tell the user; it is not yours to redraw.")
        elif checked:
            fix = ("You drew it: draw it for the width it prints at "
                   "(shorter, if taller than a page) or give it the page "
                   "width (width:100%), then look at the page.")
        else:
            fix = ("It is a module's own figure, not yours to redraw: give "
                   "it the page width (width:100%); if it already has it, "
                   "tell the user.")
        out.append(f"Figure {k} (pdf_page {f['pdf_page']}) prints "
                   f"{f['width_in']:g} x {f['height_in']:g} in: "
                   + "; ".join(why) + ". " + fix)
    return out


def _pdf_layout(path: str, figures=None, max_pages: int = 20) -> dict:
    """``{"pages", "page_contents"[, "figure_warnings"]}`` read back from a
    written PDF: per page (``pdf_page``, from 1, as a PDF viewer shows),
    its first line of text, how many characters and images it holds, a
    note on a page with nothing on it, and each figure on it: its printed
    ``width_in`` and ``height_in`` and, where the figure's PNG says,
    ``lettering_pt`` (its smallest lettering as printed). ``figures`` is
    the renderer's (``render_html_pdf_story``); without it the sizes are
    read from the PDF. ``{}`` when it cannot be read."""
    try:
        import fitz                                        # PyMuPDF
        doc = fitz.open(path)
    except Exception:                                      # noqa: BLE001
        return {}
    try:
        if figures is None:
            figures = []
            for page in doc:
                for info in page.get_image_info():
                    r = fitz.Rect(info["bbox"])
                    figures.append({"pdf_page": page.number + 1,
                                    "width_in": round(r.width / 72, 2),
                                    "height_in": round(r.height / 72, 2)})
        rows = []
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text = page.get_text("text") or ""
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            n_img = len(page.get_images(full=True))
            row = {"pdf_page": i + 1, "first_line": (lines[0][:80] if lines
                                                     else ""),
                   "chars": len(text.strip()), "images": n_img}
            on_page = []
            for k, f in enumerate(figures, start=1):
                if f["pdf_page"] != i + 1:
                    continue
                fig = {"figure": k, "width_in": f["width_in"],
                       "height_in": f["height_in"]}
                for key in ("lettering_pt", "moved", "scaled"):
                    if f.get(key):
                        fig[key] = f[key]
                on_page.append(fig)
            if on_page:
                row["figures"] = on_page
            if not lines and not n_img:
                row["note"] = "nothing on this page"
            rows.append(row)
        out = {"pages": doc.page_count, "page_contents": rows}
        if doc.page_count > max_pages:
            out["page_contents_note"] = (f"first {max_pages} of "
                                         f"{doc.page_count} pages")
        warnings = _figure_warnings(figures)
        if warnings:
            out["figure_warnings"] = warnings
        return out
    except Exception:                                      # noqa: BLE001
        return {}
    finally:
        doc.close()


def _generate_html_to_pdf(params: dict) -> dict:
    from calc_package.latex_renderer import (figures_not_printed_warning,
                                           render_html_pdf_story)

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
        rendered = render_html_pdf_story(html, abs_path)
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
        layout = _pdf_layout(abs_path, rendered.get("figures"))
        if layout:
            response.update(layout)
    if findings:  # only reachable with allow_unembeddable=true
        response["unembeddable"] = findings[:12]
        response["renderer_warnings"] = [
            f"{len(findings)} figure reference(s) were NOT embedded and are "
            "missing from the PDF (allow_unembeddable=true). Tell the user "
            "which figures are absent — do not describe the PDF as complete."]
    missing = figures_not_printed_warning(rendered)
    if missing:
        response.setdefault("renderer_warnings", []).append(missing)
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
    """AASHTO 1993 or UFC 3-250-01 pavement design calc package.

    ``design_type`` selects the procedure, each the pavement_design tool of
    that name run on the remaining parameters (US customary units). Note:
    for rigid designs, ``pavement_type`` keeps its Table 2.6 meaning
    (JCP/JRCP vs CRCP).
    """
    run = _run_tool("pavement_design_package", params)
    result = run.out
    if run.value == "flexible_ufc":
        extra = {
            "required_total_thickness_in":
                result["section"]["required_total_thickness_in"],
            "provided_total_thickness_in":
                result["section"]["provided_total_thickness_in"],
            "frost_governs": result.get("frost_governs"),
        }
        label = "UFC 3-250-01 Flexible Pavement Design"
    elif run.value == "rigid_ufc":
        extra = {"hd_required_in": result["hd_required_in"],
                 "slab_provided_in": result["slab_provided_in"]}
        label = "UFC 3-250-01 Rigid Pavement Design"
    elif run.value == "compare":
        extra = {
            "aashto_total_in": result["aashto_1993"]["total_thickness_in"],
            "ufc_total_in": result["ufc_3_250_01"]["total_thickness_in"],
            "delta_total_thickness_in": result["delta_total_thickness_in"],
        }
        label = ("Pavement Method Comparison (AASHTO 1993 vs "
                 "UFC 3-250-01)")
    elif run.value == "flexible":
        extra = {
            "sn_required": result.sn_required,
            "sn_provided": result.sn_provided,
            "layer_thicknesses_in": {
                lay["layer_type"]: lay.get("thickness_in")
                for lay in result.layers},
            "adequate": result.adequate,
        }
        label = "AASHTO 1993 Flexible Pavement Design"
    else:
        extra = {
            "d_required_in": result.d_required_in,
            "d_provided_in": result.d_provided_in,
            "k_pci": result.k_pci,
            "adequate": result.adequate,
        }
        label = "AASHTO 1993 Rigid Pavement Design"

    return _build_response("pavement_design", result, None, params,
                           analysis_type=label, extra=extra, run=run)


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
    "soe_package": _generate_soe_package,
    "reliability_package": _generate_reliability_package,
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
    "assumptions": {"type": "dict", "required": False,
                    "description": (
                        "The inputs you pass that the user did not give, as "
                        "{input name: its basis}, e.g. {\"unit_weight\": "
                        "\"not given; typical of the soil described\"}. An "
                        "assumption that is not an input is its own row: "
                        "{\"drained\": \"long-term condition\"}. The "
                        "package lists them under Assumed Inputs with the "
                        "analysis's own defaults, so no assumption reads as "
                        "given. A misspelt input name is refused with a "
                        "did-you-mean.")},
    "given": {"type": "list|dict", "required": False,
              "description": (
                  "The inputs you pass that the user gave, by the names "
                  "you pass them under, e.g. [\"width\", \"depth\", "
                  "\"friction_angle\"], or with the user's own words, "
                  "{\"friction_angle\": \"phi 32 deg from the boring "
                  "logs\"}. Only these print under Given Inputs. "
                  "Every other input you pass prints under Assumed Inputs "
                  "as passed, not stated as given (with its basis when it "
                  "is in assumptions); left out, no input is the user's. A "
                  "value you read off, worked out or chose is assumed, not "
                  "given.")},
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
    "figure_warnings": ("Only for a PDF with a figure that prints too small "
                        "to read (lettering under 7 pt). The figure is the "
                        "module's own, not yours to redraw: tell the user."),
}

METHOD_INFO = {
    "pavement_design_package": {
        "category": "Calculation Package",
        "brief": ("Pavement design calc package: AASHTO 1993 flexible "
                  "(required SN + layer split) or rigid (slab D) with "
                  "reliability, drainage, and forward checks, UFC 3-250-01 "
                  "flexible or rigid, or the two methods compared. US "
                  "customary units. The parameters of the pavement_design "
                  "tool design_type names."),
        "returns": {**_COMMON_RETURNS,
                    "extra": "sn_required/sn_provided + layer thicknesses "
                             "(flexible) or d_required_in/d_provided_in + "
                             "k_pci (rigid), and adequate; the UFC and "
                             "compare designs give their thicknesses."},
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
                 "never a silent blank. A figure that does not fit what is left of a "
                 "page starts the next page at its full size; one taller than a page is "
                 "scaled to fit a page. Pure-Python (PyMuPDF Story engine).",
        "parameters": {
            "html": {"type": "str", "required": False,
                     "description": "Self-contained HTML content to render: the "
                                    "finished page as ONE string value, written out "
                                    "in full with each figure's <img> tag in place "
                                    "(nothing in the call is run, so no code builds "
                                    "or edits it). Inline all "
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
                             "pdf_page (from 1, as a PDF viewer shows), its "
                             "first line of text, characters and images (a "
                             "page with nothing on it says so), and each "
                             "figure on it with its printed width_in and "
                             "height_in and, where its PNG says, lettering_pt "
                             "(its smallest lettering as printed). Check a "
                             "figure landed where the report needs it, at a "
                             "size that reads, from this, without opening "
                             "the PDF.",
            "figure_warnings": "Present when a figure prints too small to "
                               "read: any figure whose lettering prints "
                               "under 7 pt, one the plotting box drew that "
                               "prints narrower than the width its lettering "
                               "was checked at, or any figure taller than a "
                               "page and scaled below half its width. Each "
                               "says whose figure it is: fix the ones you "
                               "drew before handing the PDF over; a module's "
                               "own figure is reported, not yours to redraw "
                               "(give it the page width, else tell the "
                               "user).",
            "unembeddable": "On error (or with allow_unembeddable=true): each figure "
                            "reference that cannot be embedded, and why.",
        },
    },
    "bearing_capacity_package": {
        "category": "Calculation Package",
        "brief": "Run bearing capacity analysis and generate Mathcad-style calc package, with a table (CSV + printed) of the factors N, s, d, i, b, g and the value of each term.",
        "returns": {**_COMMON_RETURNS, "q_ultimate_kPa": "Ultimate bearing capacity.", "q_allowable_kPa": "Allowable bearing capacity (gross).", "q_allowable_net_kPa": "Net allowable bearing pressure q_ult/FS - q (UFC 3-220-20 Eq. 5-14).", "allowable_load": "The allowable loads on the footing (per metre for a strip): structural_load_kN, the column or structure load Q_DL+LL = q_all,net x A' (Eq. 5-3; under an eccentric load q_all x A' - q x A), and gross_load_kN = q_all x A', the total load on the base with the footing and the soil over it (Eq. 5-1), with their basis. Quote the structural load as the allowable column load, never the gross load."},
    },
    "lateral_pile_package": {
        "category": "Calculation Package",
        "brief": "Run lateral pile analysis and generate LPILE-style calc package, with tables (CSV + printed) of y, slope, M, V, p and Es at every node and each layer's p-y curve.",
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
        "returns": {**_COMMON_RETURNS, "total_settlement_mm": "Total settlement.", "immediate_mm": "Immediate.", "consolidation_mm": "Consolidation."},
    },
    "axial_pile_package": {
        "category": "Calculation Package",
        "brief": "Run axial pile capacity analysis and generate APILE-style calc package, with tables (CSV + printed) of side resistance per layer, load transfer at the ultimate load and capacity against pile length.",
        "returns": {**_COMMON_RETURNS, "Q_ultimate_kN": "Ultimate capacity.", "Q_allowable_kN": "Allowable capacity."},
    },
    "drilled_shaft_package": {
        "category": "Calculation Package",
        "brief": "Run drilled shaft capacity analysis and generate SHAFT-style calc package, with tables (CSV + printed) of side resistance per layer, load transfer at the ultimate load and capacity against shaft length.",
        "returns": {**_COMMON_RETURNS, "Q_ultimate_kN": "Ultimate capacity.", "Q_allowable_kN": "Allowable capacity."},
    },
    "downdrag_package": {
        "category": "Calculation Package",
        "brief": "Run downdrag analysis and generate calc package, with tables (CSV + printed) of axial load, unit skin friction and soil settlement along the pile, and every neutral-plane method side by side (CSV).",
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
        "returns": {**_COMMON_RETURNS, "seismic_analysis_type": "Which sub-analysis was performed."},
    },
    "retaining_wall_package": {
        "category": "Calculation Package",
        "brief": "Run retaining wall analysis and generate calc package.",
        "returns": {**_COMMON_RETURNS, "verdict": "The result in plain words: failed checks first, then not checked. Report it to the user.", "checks_failed": "Checks that failed (sliding, overturning, bearing, eccentricity). Report every one.", "checks_not_run": "Checks not run ('bearing' without q_allowable).", "FOS_sliding": "Sliding FOS (null: no driving force).", "FOS_overturning": "Overturning FOS (null: no driving force).", "FOS_bearing": "Bearing FOS (null: not checked)."},
    },
    "ground_improvement_package": {
        "category": "Calculation Package",
        "brief": "Run ground improvement analysis and generate calc package.",
        "returns": {**_COMMON_RETURNS, "method": "Ground improvement method used."},
    },
    "wave_equation_package": {
        "category": "Calculation Package",
        "brief": "Run wave equation analysis and generate bearing graph calc package.",
        "returns": {**_COMMON_RETURNS, "n_points": "Number of bearing graph points."},
    },
    "pile_group_package": {
        "category": "Calculation Package",
        "brief": "Run rigid-cap pile group analysis (CPGA stiffness method) and generate calc package; with method_6dof, tables (CSV + printed) of the force and displacement at every pile head.",
        "returns": {**_COMMON_RETURNS, "n_piles": "Number of piles.", "group_efficiency": "Converse-Labarre group efficiency (with n_rows, n_cols, spacing and pile_diameter), else null."},
    },
    "sheet_pile_package": {
        "category": "Calculation Package",
        "brief": "Run sheet pile wall analysis and generate calc package, with a table (CSV + printed) of pressure, shear and moment along the wall.",
        "returns": {**_COMMON_RETURNS, "embedment_m": "Required embedment.", "max_moment_kNm_per_m": "Max moment."},
    },
    "soe_package": {
        "category": "Calculation Package",
        "brief": "Run a support-of-excavation wall analysis (cantilever, braced/anchored at support levels, or FHWA apparent-pressure anchored wall) and generate its calc package: earth pressures, embedment, moments and the strut or anchor loads, with tables (CSV + printed) of pressure, shear and moment along the wall.",
        "returns": {**_COMMON_RETURNS, "embedment_m": "Required embedment below the excavation (null: no moment balance; see notes).", "max_moment_kNm_per_m": "Max wall moment.", "support_loads_kN_per_m": "Braced: the load on each support level.", "anchor_design_loads_kN": "FHWA anchored: each anchor's design load.", "notes": "The analysis's notes (also printed in the package). Report every one."},
    },
    "reliability_package": {
        "category": "Calculation Package",
        "brief": "Run a reliability analysis (FOSM, point estimates, Monte Carlo or FORM) on a performance function of random variables and generate its calc package: the variables and their distributions, the moments of g, the reliability index beta and the probability of failure, and which variable matters.",
        "returns": {**_COMMON_RETURNS, "engine": "The engine run.", "beta_normal": "Reliability index, g normal (FOSM, PEM, Monte Carlo moments).", "pf_normal": "Probability of failure, g normal.", "beta_lognormal": "Reliability index, factor of safety lognormal (convention 'fos').", "pf_lognormal": "Probability of failure, lognormal.", "beta": "FORM reliability index.", "pf": "FORM or Monte Carlo (counted) probability of failure."},
    },
}

# A package that runs an analysis tool takes the tool's parameters, aliases
# and refusals, and its defaults apply when the tool's do (GM-141).
for _package in _PACKAGE_TOOLS:
    METHOD_INFO[_package].update(_package_parameters(_package))

# When each default applies (GM-119), as in the analysis tools: the slope
# report's settings to its kind of search and its probabilistic annex.
_CIRCLES_GRID = ["circular", "entry_exit", "pso"]
_RANDOM_POLYLINES = ["noncircular", "pso", "weak_layer"]
declare_applies_when(METHOD_INFO, {
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

# What a left-out input of a slope package stands for (GM-165), as in the
# slope tools: no water table, the search's windows and grid from the
# slope, and an unseeded search or Monte Carlo.
_SLOPE_NO_WATER = ("no water table", "no pore water pressure: effective "
                   "stress equals total stress")
_SLOPE_WINDOW = ("set from the slope's geometry",
                 "entry and exit windows from the slope's toe, crest, height "
                 "and depth to the base; the package lists what the search "
                 "covered")
_SLOPE_GRID = ("set from the slope's geometry",
               "the circle-centre grid over the windows, from the slope's "
               "height and depth to the base")
declare_when_absent(METHOD_INFO, {
    "slope_stability_package": {"gwt_points": _SLOPE_NO_WATER},
    "slope_report_package": {
        "gwt_points": _SLOPE_NO_WATER,
        "x_entry_range": _SLOPE_WINDOW, "x_exit_range": _SLOPE_WINDOW,
        "x_range": _SLOPE_GRID, "y_range": _SLOPE_GRID,
        "seed": ("not seeded", "random trial surfaces and Monte Carlo "
                 "samples can differ from run to run; give seed to repeat "
                 "a run")},
})
_report = METHOD_INFO["slope_report_package"]["parameters"]
for _name in ("x_range", "y_range"):
    _report[_name]["applies_when"] = {"surface_type": _CIRCLES_GRID}
_report["seed"]["applies_when"] = [
    {"surface_type": _RANDOM_POLYLINES + ["noncircular_de"]},
    {"variables": "set", "monte_carlo": [True]}]
