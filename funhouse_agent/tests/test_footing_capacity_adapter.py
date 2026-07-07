"""Adapter tests for fem2d_footing_capacity (v5.4 E11)."""

import pytest

from funhouse_agent.adapters.fem2d_adapter import METHOD_REGISTRY as R, METHOD_INFO


def test_prandtl_via_adapter_coarse():
    """Coarse-mesh Prandtl footing through the adapter returns sane keys and a
    back-figured Nc near 5.14 (loose band for the coarse mesh)."""
    out = R["fem2d_footing_capacity"]({
        "B": 2.0, "c": 100.0, "nx": 24, "ny": 12, "n_load_steps": 30,
        "q_applied": 150.0})
    assert out["collapse_bracketed"] is True
    assert 4.8 <= out["Nc_backfigured"] <= 6.0
    assert out["q_ult_kPa"] > 0
    assert out["bearing_FOS"] == pytest.approx(out["q_ult_kPa"] / 150.0, rel=1e-6)
    assert set(out["bearing_capacity_factors"]) == {"Nc", "Nq", "Ngamma"}


def test_missing_required_raises():
    with pytest.raises(ValueError, match="B|c"):
        R["fem2d_footing_capacity"]({"c": 100.0})


def test_bad_element_type_raises():
    with pytest.raises(ValueError, match="element_type must be one of"):
        R["fem2d_footing_capacity"]({"B": 2.0, "c": 100.0, "element_type": "q8"})


def test_unknown_param_rejected():
    with pytest.raises(ValueError, match="phi_deg"):
        R["fem2d_footing_capacity"]({"B": 2.0, "c": 100.0, "phi_deg": 30})


def test_method_info_allowed_values():
    p = METHOD_INFO["fem2d_footing_capacity"]["parameters"]
    assert p["element_type"]["allowed_values"] == ["t6", "cst"]
    assert p["B"]["required"] is True and p["c"]["required"] is True
