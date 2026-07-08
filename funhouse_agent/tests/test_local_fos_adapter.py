"""Adapter tests for fem2d_local_fos (v5.4 F5)."""

import pytest

from funhouse_agent.adapters.fem2d_adapter import METHOD_REGISTRY as R, METHOD_INFO


_SLOPE = {
    "surface_points": [[0, 10], [12, 10], [32, 0], [44, 0]],
    "soil_layers": [{"name": "s", "bottom_elevation": -10, "E": 1e5,
                     "nu": 0.3, "c": 10.0, "phi": 20.0, "gamma": 20.0}],
    "depth": 0.5, "x_extend": 0.0, "srf_tol": 0.02,
}


@pytest.mark.slow
def test_local_fos_min_tracks_global():
    out = R["fem2d_local_fos"](dict(_SLOPE, nx=24, ny=10))
    assert out["frac_below_1"] == 0.0
    assert 0.85 <= out["min_local_fos"] / out["global_fos"] <= 1.05
    assert out["FOS"] == out["global_fos"]
    assert len(out["min_location_xy"]) == 2
    assert out["n_elements"] > 0


def test_missing_required_raises():
    with pytest.raises(ValueError, match="surface_points|soil_layers"):
        R["fem2d_local_fos"]({"nx": 20})


def test_bad_element_type_raises():
    with pytest.raises(ValueError, match="element_type must be one of"):
        R["fem2d_local_fos"](dict(_SLOPE, element_type="q8"))


def test_unknown_param_rejected():
    with pytest.raises(ValueError, match="bogus"):
        R["fem2d_local_fos"](dict(_SLOPE, bogus=1))


def test_method_info_allowed_values():
    p = METHOD_INFO["fem2d_local_fos"]["parameters"]
    assert p["element_type"]["allowed_values"] == ["t6", "cst"]
    assert p["surface_points"]["required"] is True
