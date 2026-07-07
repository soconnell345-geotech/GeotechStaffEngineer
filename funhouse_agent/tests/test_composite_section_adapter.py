"""Adapter tests for the lateral_pile composite_section_ei method (v5.4 E5)."""

import math

import pytest

from funhouse_agent.adapters.lateral_pile import METHOD_REGISTRY as R, METHOD_INFO


def test_filled_pipe_ei_matches_module():
    """The adapter returns the same composite EI as the module helper."""
    out = R["composite_section_ei"]({
        "section_type": "filled_pipe",
        "outer_diameter": 0.1969, "wall_thickness": 0.0151,
        "fc": 27600.0, "E_steel": 199947980.0,
    })
    assert out["EI_kNm2"] == pytest.approx(8109.0, abs=3.0)
    assert out["basis"] == "uncracked_transformed_section"
    # transformed inertia at E_ref reproduces the EI
    assert out["E_ref_kPa"] * out["inertia_transformed_m4"] == pytest.approx(
        out["EI_kNm2"], rel=1e-9)


def test_reinforced_concrete_rectangular_bar_layers():
    """bar_layers arrive as JSON [[n, y], ...] and are accepted."""
    out = R["composite_section_ei"]({
        "section_type": "reinforced_concrete",
        "width": 0.4, "height": 0.6, "fc": 30000.0,
        "bar_diameter": 0.0254, "bar_layers": [[3, 0.25], [3, -0.25]],
    })
    assert out["EI_kNm2"] > 0
    assert out["neutral_axis_m"] == pytest.approx(0.0, abs=1e-9)


def test_unknown_param_rejected():
    """An invented parameter is rejected loudly (not silently ignored)."""
    with pytest.raises(ValueError, match="E_GPa"):
        R["composite_section_ei"]({
            "section_type": "filled_pipe", "outer_diameter": 0.4,
            "wall_thickness": 0.02, "fc": 30000.0, "E_GPa": 9.3,
        })


def test_missing_section_type_clear_error():
    with pytest.raises(ValueError, match="section_type"):
        R["composite_section_ei"]({"outer_diameter": 0.4, "wall_thickness": 0.02,
                                   "fc": 30000.0})


def test_method_info_advertises_allowed_values():
    params = METHOD_INFO["composite_section_ei"]["parameters"]
    assert params["section_type"]["allowed_values"] == [
        "filled_pipe", "cased_concrete", "reinforced_concrete"]
    assert params["section_type"]["required"] is True
