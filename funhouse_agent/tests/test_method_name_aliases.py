"""Tests for the curated ``_METHOD_ALIASES`` method-name guess map (dispatch.py).

The agent frequently guesses a method NAME that doesn't exist (e.g. calling
``drilled_shaft/alpha_method`` instead of ``drilled_shaft_capacity``), or guesses
a selector VALUE as if it were a method (``bearing_capacity/vesic``). ``call_agent``
resolves these via ``_METHOD_ALIASES``. These tests assert the curated aliases
route to the intended real method (and a selector injection lands), rather than
returning ``Unknown method``.

Targets/params are sourced from the agent test-suite triage
(``module_work/module_feedback.json`` + ``docs/geotech_test_suite_results.json``).
"""

import pytest

from funhouse_agent.dispatch import (
    _METHOD_ALIASES,
    _load_adapter,
    _resolve_unknown_method,
    call_agent,
)


# (agent, guessed_name) -> expected real method, and an optional selector
# injection that must appear in the resolved params.
ROUTING_CASES = [
    ("bearing_capacity", "vesic", "bearing_capacity_analysis", ("factor_method", "vesic")),
    ("bearing_capacity", "vesic_footing", "bearing_capacity_analysis", ("factor_method", "vesic")),
    ("bearing_capacity", "terzaghi", "bearing_capacity_analysis", None),
    ("bearing_capacity", "two_layer_clay", "bearing_capacity_analysis", None),
    ("settlement", "consolidation", "consolidation_settlement", None),
    ("settlement", "elastic_foundation", "elastic_settlement", None),
    ("drilled_shaft", "alpha_method", "drilled_shaft_capacity", None),
    ("drilled_shaft", "beta_method", "drilled_shaft_capacity", None),
    ("drilled_shaft", "rock_socket_capacity", "drilled_shaft_capacity", None),
    ("drilled_shaft", "single_shaft_capacity", "drilled_shaft_capacity", None),
    ("axial_pile", "beta_method", "axial_pile_capacity", None),
    ("downdrag", "fellenius_neutral_plane", "downdrag_analysis", None),
    ("fem2d", "slope_strength_reduction", "fem2d_slope_srm", None),
    ("liquepy", "cpt_boulanger_idriss_2014", "cpt_liquefaction", None),
    ("salib", "sobol_sensitivity", "sobol_sample", None),
    ("pystrata", "equivalent_linear", "eql_site_response", None),
    ("gstools", "fit_variogram", "variogram", None),
    ("subsurface", "read_and_validate", "read_ags4", None),
    ("dxf_export", "export_cross_section", "export_geometry_to_dxf", None),
    # Curated from the 2026-07-05 71-question eval run.
    ("lateral_pile", "analyze_lateral_pile", "lateral_pile_analysis", None),
    ("retaining_walls", "earth_pressure_analysis", "earth_pressure_coefficient", None),
    ("ground_improvement", "aggregate_pier_design", "aggregate_piers", None),
    ("liquefaction", "cpt_based_triggering", "liquefaction_analysis", None),
    ("gstools", "ordinary_kriging", "kriging", None),
    ("dxf_import", "discover_dxf", "discover_layers", None),
    # Curated from the 2026-07-13 100-question eval run (NMK-2, EPC-3).
    ("slope_stability", "newmark", "newmark_displacement", None),
    ("slope_stability", "newmark_analysis", "newmark_displacement", None),
    ("slope_stability", "sliding_block", "newmark_displacement", None),
    ("retaining_walls", "caquot", "earth_pressure_coefficient", ("theory", "caquot_kerisel")),
    ("retaining_walls", "log_spiral", "earth_pressure_coefficient", ("theory", "caquot_kerisel")),
    ("retaining_walls", "passive_coefficient", "earth_pressure_coefficient", ("state", "passive")),
]


@pytest.mark.parametrize("agent,guess,expected,inject", ROUTING_CASES)
def test_alias_routes_to_real_method(agent, guess, expected, inject):
    """Each curated guess resolves to the intended real method (not Unknown)."""
    mod = _load_adapter(agent)
    resolved = _resolve_unknown_method(mod, agent, guess, {"_probe": 1})
    assert resolved is not None, f"{agent}/{guess} did not resolve"
    real, params = resolved
    assert real == expected
    assert real in mod.METHOD_REGISTRY        # routes to a real, callable method
    assert params["_probe"] == 1              # original params preserved
    if inject is not None:
        key, val = inject
        assert params[key] == val             # selector value injected


def test_case_insensitive_resolution():
    """Guesses resolve regardless of case (keys are lowercased)."""
    mod = _load_adapter("drilled_shaft")
    resolved = _resolve_unknown_method(mod, "drilled_shaft", "ALPHA_Method", {})
    assert resolved is not None
    assert resolved[0] == "drilled_shaft_capacity"


def test_no_alias_returns_none():
    """A genuinely-unknown guess with no curated entry does NOT resolve."""
    mod = _load_adapter("bearing_capacity")
    assert _resolve_unknown_method(
        mod, "bearing_capacity", "totally_made_up_method", {}) is None


def test_alias_end_to_end_bearing_capacity():
    """A full call via an alias produces a real result (not an error)."""
    r = call_agent("bearing_capacity", "vesic", {
        "width": 2.5, "depth": 1.0, "shape": "strip", "cohesion": 0.0,
        "friction_angle": 34.0, "unit_weight": 19.0, "factor_of_safety": 3.0,
    })
    assert "error" not in r
    assert r["q_ultimate_kPa"] > 0


def test_alias_end_to_end_drilled_shaft():
    r = call_agent("drilled_shaft", "alpha_method", {
        "diameter": 1.0, "shaft_length": 12.0,
        "layers": [{"thickness": 12.0, "soil_type": "cohesive",
                    "unit_weight": 18.0, "cu": 80.0, "phi": 0.0,
                    "N60": 0.0, "qu": 0.0, "RQD": 0.0}],
    })
    assert "error" not in r


def test_all_alias_targets_exist():
    """Every alias target must be a real method in its module's registry."""
    for (agent, _guess), entry in _METHOD_ALIASES.items():
        real = entry if isinstance(entry, str) else entry[0]
        mod = _load_adapter(agent)
        assert real in mod.METHOD_REGISTRY, f"{agent}: alias target {real!r} missing"
