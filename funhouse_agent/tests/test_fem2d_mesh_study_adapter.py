"""Adapter tests for fem2d_srm_mesh_refinement_study (BACKLOG GM-280)."""

import pytest

from funhouse_agent.adapters.fem2d_adapter import METHOD_REGISTRY as R, METHOD_INFO

_NAME = "fem2d_srm_mesh_refinement_study"

_SLOPE = {
    "surface_points": [[0, 10], [10, 10], [20, 0], [30, 0]],
    "soil_layers": [{"name": "clay", "bottom_elevation": -5, "E": 20000,
                     "nu": 0.3, "c": 20, "phi": 20, "psi": 0, "gamma": 18}],
}


def test_the_slope_tool_points_at_a_method_that_exists():
    assert _NAME in R and _NAME in METHOD_INFO
    from funhouse_agent.adapters import fem2d_adapter
    import inspect
    assert _NAME in inspect.getsource(fem2d_adapter._run_analyze_slope_srm)


def test_every_declared_parameter_is_one_the_handler_takes():
    from funhouse_agent.tests.method_probes import PROBES
    probe = PROBES[f"fem2d|{_NAME}"]["params"]
    assert set(probe) <= set(METHOD_INFO[_NAME]["parameters"])


def test_returns_the_mesh_table_and_the_study_summary():
    out = R[_NAME](dict(_SLOPE, meshes=[[8, 4], [12, 6]], published=1.2))
    levels = out["levels"]
    assert [(lv["nx"], lv["ny"]) for lv in levels] == [(8, 4), (12, 6)]
    assert levels[0]["rel_change"] is None
    assert levels[1]["n_elements"] > levels[0]["n_elements"]
    assert all(lv["error_vs_published"] is not None for lv in levels)
    assert out["fos_finest"] == levels[-1]["FOS"]
    assert out["fos_richardson"] is None          # two meshes: no estimate
    assert out["fos_estimate"] == out["fos_finest"]
    assert out["conv_tol"] == 0.03 and out["element_type"] == "t6"


def test_fewer_than_two_meshes_refused():
    with pytest.raises(ValueError, match="at least two"):
        R[_NAME](dict(_SLOPE, meshes=[[8, 4]]))


def test_missing_meshes_and_unknown_inputs_refused():
    with pytest.raises(ValueError, match="meshes"):
        R[_NAME](dict(_SLOPE))
    with pytest.raises(ValueError, match="bogus"):
        R[_NAME](dict(_SLOPE, meshes=[[8, 4], [12, 6]], bogus=1))


def test_bad_element_type_refused():
    with pytest.raises(ValueError, match="element_type must be one of"):
        R[_NAME](dict(_SLOPE, meshes=[[8, 4], [12, 6]], element_type="q8"))
