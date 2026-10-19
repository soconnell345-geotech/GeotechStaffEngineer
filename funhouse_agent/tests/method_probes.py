"""The probe table behind the METHOD_INFO guard (BACKLOG GM-158, GM-150).

``method_probes.json`` holds, for every analysis-side method, the smallest
inputs it runs on: ``"agent|method"`` (or ``"agent|method#variant"`` where
one method needs several, e.g. one p-y model each) -> ``{"params": {...}}``,
with ``"before": [[agent, method, params], ...]`` for a method that works on
what an earlier call loaded. The inputs came from the test suite's own
calls, cut down to the parameters each method needs, so every parameter a
probe leaves out runs at its default and the guard can check that
default. A few probes also give a seed (a sampler with its own generator)
or a coarse mesh or search grid (the guard runs each probe twice), so
those defaults are the ones not checked here.

A file reader's probe names a sample in ``funhouse_agent/eval_samples/``
as ``"<eval_samples>/<file>"``, or a test fixture as ``"<repo>/<path>"``,
made an absolute path here (the guard runs each probe in a scratch
folder); a probe whose method needs an optional
backend names it under ``"needs"`` (an importable module), and the guard
skips it where that is not installed.

Each method either has a probe or is in :data:`EXCLUDED` with the reason.
"""

import json
from pathlib import Path

#: The sample files the file readers' probes read (GM-160).
REPO = Path(__file__).resolve().parents[2]
SAMPLES = REPO / "funhouse_agent" / "eval_samples"
_TAGS = {"<eval_samples>/": SAMPLES, "<repo>/": REPO}


def _with_samples(x):
    """``x`` with every ``"<eval_samples>/<file>"`` or ``"<repo>/<path>"``
    made an absolute path."""
    if isinstance(x, str):
        for tag, root in _TAGS.items():
            if x.startswith(tag):
                return str(root / x[len(tag):])
    if isinstance(x, dict):
        return {k: _with_samples(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_with_samples(v) for v in x]
    return x


PROBES = _with_samples(json.loads(
    (Path(__file__).with_name("method_probes.json")).read_text(
        encoding="utf-8")))

_FILE = "reads a file the call names; no file fixture here"
_HANDLE = "works on a handle an earlier file read returned"

#: Methods with no probe, and why.
EXCLUDED = {
    "drawing_ir|digitize_drawing": _FILE,
    "drawing_ir|get_entities": _HANDLE,
    "drawing_ir|query_drawing": _HANDLE,
    "drawing_ir|search_drawing_set": _FILE,
    "drawing_ir|snip_region": _FILE,
    "pdf_import|build_fem_inputs": _HANDLE,
    "pdf_import|build_slope_geometry": _HANDLE,
    "pdf_import|cleanup_geometry": _HANDLE,
    "pdf_import|cross_check": _HANDLE,
    "pdf_import|discover_pdf_content": _FILE,
    "pdf_import|extract_vector_geometry": _FILE,
    "pdf_import|propose_role_mapping": _HANDLE,
    "pdf_import|propose_scale": _HANDLE,
    "opensees|pm4sand_cyclic_dss": "needs OpenSeesPy, an optional backend "
                                   "that is not always present",
    "opensees|site_response_1d": "needs OpenSeesPy, an optional backend "
                                 "that is not always present",
}

#: Declared parameters an adapter reads and removes before its own name
#: check, so the check's list leaves them out: (method, name) -> how.
READ_BEFORE_THE_CHECK = {
    ("ground_improvement|aggregate_piers", "area_replacement_ratio"):
        "turned into a spacing for the column diameter and pattern",
}

#: Probes that take several seconds a run even on a coarse grid.
SLOW = frozenset({"slope_stability|search_rapid_drawdown"})

#: Probes whose run takes several seconds, so giving each listed default
#: other values takes a minute or more (GM-160's check only).
SLOW_TO_VARY = frozenset({
    "calc_package|downdrag_package", "calc_package|lateral_pile_package",
    "calc_package|retaining_wall_package",
    "calc_package|retaining_wall_package#surcharge",
    "calc_package|sheet_pile_package", "calc_package|slope_report_package",
    "calc_package|wave_equation_package", "fem2d|fem2d_excavation",
    "fem2d|fem2d_footing_capacity", "fem2d|fem2d_local_fos",
    "fem2d|fem2d_slope_srm", "gstools|conditional_simulation",
    "slope_stability|monte_carlo_fos",
    "slope_stability|search_critical_surface",
    "slope_stability|yield_acceleration"})

#: Probes GM-160's check does not vary, and why.
NOT_VARIED = {
    k: "each run is a full critical-surface search (minutes), and "
       "slope_stability is frozen by the owner until the integral review"
    for k in ("slope_stability|monte_carlo_fos",
              "slope_stability|search_critical_surface",
              "slope_stability|search_rapid_drawdown")}

#: Result keys that share a parameter's name but report something else,
#: so ``defaults_used`` and the result may differ there: (probe key, name)
#: -> what the result's value is.
ECHO_IS_NOT_THE_INPUT = {
    ("calc_package|seismic_package", "analysis_type"):
        "the package's title",
    ("calc_package|downdrag_package", "neutral_plane_method"):
        "the method 'auto' chose",
    ("downdrag|downdrag_analysis", "neutral_plane_method"):
        "the method 'auto' chose",
    ("gstools|variogram", "n_bins"): "the lag classes that hold pairs",
    ("reliability|pem", "scheme"): "the scheme's full name",
    ("salib|sobol_analysis", "n_samples"): "the model outputs analysed",
}

#: Declared inputs that change nothing in their probe's result by design
#: (GM-160): ``(probe key, name) -> what the input does``. The guard
#: gives every default a probe's run listed (``defaults_used``) another
#: value and fails where the result does not move, past the input's own
#: echo and the result's ``choices``; an input here is skipped. A name
#: whose effect the probe's inputs cannot show (a clamp the probe does not
#: reach, a branch it does not take) goes in :data:`NOT_REACHED`, never
#: here.
LABEL_ONLY = {}

#: Declared inputs that act, but not on their probe's inputs: ``(probe
#: key, name) -> why the probe does not show it``.
NOT_REACHED = {}

#: Mismatches the guard finds and leaves, each with its BACKLOG item.
#: slope_stability is frozen by the owner until the integral review.
KNOWN = {
    ("slope_stability|analyze_slope", "method"):
        "GM-140: 'bishop' is listed, but a noncircular slip_points surface "
        "runs Spencer",
}
