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

Each method either has a probe or is in :data:`EXCLUDED` with the reason.
"""

import json
from pathlib import Path

PROBES = json.loads((Path(__file__).with_name("method_probes.json"))
                    .read_text(encoding="utf-8"))

_FILE = "reads a file the call names; no file fixture here"
_HANDLE = "works on a handle an earlier file read returned"

#: Methods with no probe, and why.
EXCLUDED = {
    "drawing_ir|digitize_drawing": _FILE,
    "drawing_ir|get_entities": _HANDLE,
    "drawing_ir|query_drawing": _HANDLE,
    "drawing_ir|search_drawing_set": _FILE,
    "drawing_ir|snip_region": _FILE,
    "dxf_import|discover_layers": _FILE,
    "dxf_import|parse_geometry": _FILE,
    "pdf_import|build_fem_inputs": _HANDLE,
    "pdf_import|build_slope_geometry": _HANDLE,
    "pdf_import|cleanup_geometry": _HANDLE,
    "pdf_import|cross_check": _HANDLE,
    "pdf_import|discover_pdf_content": _FILE,
    "pdf_import|extract_vector_geometry": _FILE,
    "pdf_import|propose_role_mapping": _HANDLE,
    "pdf_import|propose_scale": _HANDLE,
    "subsurface|parse_bore": _FILE,
    "subsurface|parse_cpt": _FILE,
    "subsurface|parse_diggs": _FILE,
    "subsurface|read_ags4": _FILE,
    "subsurface|validate_ags4": _FILE,
    "subsurface|validate_diggs_dictionary": _FILE,
    "subsurface|validate_diggs_schema": _FILE,
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

#: Mismatches the guard finds and leaves, each with its BACKLOG item.
#: slope_stability is frozen by the owner until the integral review.
KNOWN = {
    ("slope_stability|analyze_slope", "method"):
        "GM-140: 'bishop' is listed, but a noncircular slip_points surface "
        "runs Spencer",
}

#: Inputs a run lists as taken from another input (GM-165) that, given
#: back, add a row the package prints only for a given input, the numbers
#: the same: (probe key, name) -> why. The guard does not give them back.
SHOWN_ONLY_WHEN_GIVEN = {
    ("calc_package|drilled_shaft_package", "socket_diameter"):
        "the calc steps print a rock-socket diameter row only when it is "
        "given; the probe has no rock layer",
}
