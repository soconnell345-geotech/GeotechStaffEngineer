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
    assert "warnings" not in out


def test_mesh_set_q_ult_warning_reaches_the_agent():
    """GM-304: on a cohesionless soil with no surcharge q_ult is set by the
    mesh, and the warning is in the tool's output, not only on the result."""
    out = R["fem2d_footing_capacity"]({
        "B": 2.0, "c": 0.0, "phi": 30.0, "psi": 30.0, "gamma": 18.0,
        "nx": 20, "ny": 10})
    assert any("depends on the mesh" in w for w in out["warnings"])


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


def test_flexible_reports_its_choice():
    out = R["fem2d_footing_capacity"]({
        "B": 2.0, "c": 100.0, "nx": 20, "ny": 10, "n_load_steps": 10})
    assert out["footing"] == "flexible" and out["base"] is None
    assert [c["chosen"] for c in out["choices"]] == ["flexible", "uniform"]
    assert out["mesh"] == "uniform"
    assert "pressure_settlement" not in out


def test_rigid_via_adapter_coarse():
    """GM-305: the rigid footing through the adapter, coarse mesh."""
    out = R["fem2d_footing_capacity"]({
        "B": 2.0, "c": 100.0, "footing": "rigid", "nx": 20, "ny": 10,
        "n_load_steps": 12})
    assert out["footing"] == "rigid" and out["base"] == "rough"
    assert [c["chosen"] for c in out["choices"]] == ["rigid", "rough",
                                                      "uniform"]
    assert out["q_ult_kPa"] > (2.0 + 3.14159) * 100.0
    assert len(out["pressure_settlement"]) == out["n_steps_converged"]
    assert out["settlement_max_m"] > 0
    assert any("fell towards (2 + pi) c as the mesh was refined" in w for w in out["warnings"])


def test_bad_footing_raises():
    with pytest.raises(ValueError, match="footing must be one of"):
        R["fem2d_footing_capacity"]({"B": 2.0, "c": 100.0,
                                     "footing": "stiff"})


def test_graded_mesh_via_adapter():
    """GM-306: mesh='graded' through the adapter, coarse mesh."""
    out = R["fem2d_footing_capacity"]({
        "B": 2.0, "c": 100.0, "nx": 20, "ny": 10, "n_load_steps": 10,
        "mesh": "graded"})
    assert out["mesh"] == "graded"
    assert out["choices"][-1]["what"] == "footing mesh"
    assert out["choices"][-1]["chosen"] == "graded"
    assert out["q_ult_kPa"] > 0


def test_bad_mesh_raises():
    with pytest.raises(ValueError, match="mesh must be one of"):
        R["fem2d_footing_capacity"]({"B": 2.0, "c": 100.0,
                                     "mesh": "fine"})


def test_method_info_mesh():
    p = METHOD_INFO["fem2d_footing_capacity"]["parameters"]["mesh"]
    assert p["allowed_values"] == ["uniform", "graded"]
    assert p["default"] == "uniform"
    assert "phi = 0" in p["description"]


def test_graded_refused_on_a_frictional_soil():
    """GM-306 review: graded read about a third of uniform on c-phi."""
    with pytest.raises(ValueError, match=r"phi = 0\) only"):
        R["fem2d_footing_capacity"]({"B": 2.0, "c": 20.0, "phi": 25.0,
                                     "gamma": 18.0, "mesh": "graded",
                                     "nx": 20, "ny": 10})


def test_method_info_max_iter_scales_with_the_mesh():
    """GM-306 review: no fixed default; the case left out says the rule."""
    p = METHOD_INFO["fem2d_footing_capacity"]["parameters"]["max_iter"]
    assert "default" not in p
    assert "nx ny / 800" in p["when_absent"]["case"]
    assert "nx*ny/800" in p["description"]
