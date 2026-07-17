"""Tests for the pavement_design adapter through the dispatch layer."""

import pytest

from funhouse_agent.dispatch import (ANALYSIS_MODULES, REFERENCE_MODULES,
                                     call_agent, describe_method,
                                     list_methods)

FLEX_PARAMS = {
    "w18": 5e6, "reliability_pct": 95, "so": 0.35, "mr_psi": 5000,
    "delta_psi": 1.9,
    "layers": [
        {"layer_type": "asphalt", "modulus_psi": 400000},
        {"layer_type": "granular_base", "modulus_psi": 30000,
         "drainage_quality": "fair", "pct_saturation_time": "1-5%"},
        {"layer_type": "granular_subbase", "modulus_psi": 11000, "m": 1.0},
    ],
}

RIGID_PARAMS = {
    "w18": 5.1e6, "sc_psi": 650, "ec_psi": 5e6, "reliability_pct": 95,
    "so": 0.29, "delta_psi": 1.7, "j": 3.2, "cd": 1.0, "k_pci": 72,
}


class TestRegistration:
    def test_is_analysis_module_not_reference(self):
        assert "pavement_design" in ANALYSIS_MODULES
        assert "pavement_design" not in REFERENCE_MODULES

    def test_list_methods(self):
        res = list_methods("pavement_design")
        names = {name for cat in res.values() for name in cat}
        assert {"flexible_pavement_design", "rigid_pavement_design",
                "design_traffic_esals",
                "effective_subgrade_modulus"} <= names

    def test_describe_method(self):
        res = describe_method("pavement_design", "flexible_pavement_design")
        assert "parameters" in res
        assert "w18" in res["parameters"]


class TestCalls:
    def test_flexible_guide_example(self):
        res = call_agent("pavement_design", "flexible_pavement_design",
                         FLEX_PARAMS)
        assert "error" not in res, res.get("error")
        assert 4.9 <= res["sn_required"] <= 5.05
        assert res["adequate"] is True

    def test_flexible_aliases(self):
        p = dict(FLEX_PARAMS)
        p["esals"] = p.pop("w18")
        p["reliability"] = p.pop("reliability_pct")
        res = call_agent("pavement_design", "flexible_pavement_design", p)
        assert "error" not in res, res.get("error")
        assert 4.9 <= res["sn_required"] <= 5.05

    def test_rigid_guide_example(self):
        res = call_agent("pavement_design", "rigid_pavement_design",
                         RIGID_PARAMS)
        assert "error" not in res, res.get("error")
        assert 9.6 <= res["d_required_in"] <= 10.05
        assert res["adequate"] is True

    def test_rigid_check_mode(self):
        p = dict(RIGID_PARAMS, slab_thickness_in=8.0)
        res = call_agent("pavement_design", "rigid_pavement_design", p)
        assert res["mode"] == "check"
        assert res["adequate"] is False

    def test_traffic(self):
        res = call_agent("pavement_design", "design_traffic_esals", {
            "base_year_w18_two_way": 200000, "growth_rate_pct": 4,
            "design_period_yr": 20, "num_lanes_per_direction": 1,
            "directional_factor": 0.5,
        })
        assert "error" not in res, res.get("error")
        assert res["growth_factor"] == pytest.approx(29.778, abs=0.01)
        assert res["w18_design_lane"] == pytest.approx(
            200000 * 29.778 * 0.5, rel=0.001)

    def test_effective_mr_guide_example(self):
        res = call_agent("pavement_design", "effective_subgrade_modulus", {
            "monthly_mr_psi": [20000, 20000, 2500, 4000, 4000, 7000, 7000,
                               7000, 7000, 7000, 4000, 20000],
        })
        assert res["effective_mr_psi"] == pytest.approx(5000, rel=0.05)

    def test_unknown_param_rejected(self):
        res = call_agent("pavement_design", "flexible_pavement_design",
                         dict(FLEX_PARAMS, bogus_param=1))
        assert "error" in res
        assert "bogus_param" in str(res["error"])

    def test_missing_required_actionable(self):
        res = call_agent("pavement_design", "rigid_pavement_design",
                         {"w18": 1e6})
        assert "error" in res
        assert "sc_psi" in str(res["error"])


class TestEnvironmental:
    def test_flexible_with_swelling(self):
        pytest.importorskip("geotech_references.aashto_1993.environmental")
        p = dict(FLEX_PARAMS, swelling={"vr_in": 2.0, "ps_pct": 60,
                                        "theta": 0.10},
                 design_period_yr=15)
        res = call_agent("pavement_design", "flexible_pavement_design", p)
        assert "error" not in res, res.get("error")
        assert res["environmental"]["delta_psi_sw"] == pytest.approx(
            0.31, abs=0.02)
        assert res["sn_required"] > 4.976  # env loss grows the required SN

    def test_performance_period_method(self):
        pytest.importorskip("geotech_references.aashto_1993.environmental")
        res = call_agent("pavement_design", "performance_period", {
            "pavement_type": "flexible", "delta_psi_design": 1.9,
            "base_year_w18": 300000, "growth_rate_pct": 2.0,
            "swelling": {"vr_in": 2.0, "ps_pct": 60, "theta": 0.10},
            "max_performance_period_yr": 15,
            "sn": 5.0, "mr_psi": 5000, "reliability_pct": 95, "so": 0.35,
        })
        assert "error" not in res, res.get("error")
        assert res["converged"] is True
        assert res["performance_period_yr"] > 0
        assert isinstance(res["rows"], list) and len(res["rows"]) >= 1


class TestCalcPackage:
    def test_flexible_package(self, tmp_path):
        out = str(tmp_path / "flex_pavement.html")
        res = call_agent("calc_package", "pavement_design_package",
                         dict(FLEX_PARAMS, design_type="flexible",
                              output_path=out,
                              project_name="Pavement Test"))
        assert "error" not in res, res.get("error")
        assert res["status"] == "success"
        assert res["file_exists"] is True
        assert res["adequate"] is True
        assert 4.9 <= res["sn_required"] <= 5.05
        html = open(out, encoding="utf-8").read()
        assert "AASHTO 1993" in html
        assert "Structural" in html

    def test_rigid_package(self, tmp_path):
        out = str(tmp_path / "rigid_pavement.html")
        res = call_agent("calc_package", "pavement_design_package",
                         dict(RIGID_PARAMS, design_type="rigid",
                              output_path=out))
        assert "error" not in res, res.get("error")
        assert res["status"] == "success"
        assert res["file_exists"] is True
        assert 9.6 <= res["d_required_in"] <= 10.05
        assert res["adequate"] is True

    def test_bad_design_type(self):
        res = call_agent("calc_package", "pavement_design_package",
                         dict(FLEX_PARAMS, design_type="composite"))
        assert "error" in res

    def test_flexible_ufc_package(self, tmp_path):
        pytest.importorskip("geotech_references.ufc_pavement")
        out = str(tmp_path / "ufc_flex.html")
        res = call_agent("calc_package", "pavement_design_package", {
            "design_type": "flexible_ufc", "passes_18kip": 1e6,
            "cbr_subgrade": 6, "cbr_subbase": 30,
            "frost": {"uscs_class": "CL"}, "output_path": out,
        })
        assert "error" not in res, res.get("error")
        assert res["status"] == "success"
        assert res["frost_governs"] is True
        html = open(out, encoding="utf-8").read()
        assert "UFC 3-250-01" in html
        assert "data:image/png;base64" in html  # E-1 chart embedded

    def test_rigid_ufc_package(self, tmp_path):
        pytest.importorskip("geotech_references.ufc_pavement")
        from geotech_references.ufc_pavement import tables as utb
        if not hasattr(utb, "figure_f1_rigid_thickness"):
            pytest.skip("F-1 not digitized in this refs install")
        out = str(tmp_path / "ufc_rigid.html")
        res = call_agent("calc_package", "pavement_design_package", {
            "design_type": "rigid_ufc", "passes_18kip": 5e6,
            "flexural_strength_psi": 650, "k_pci": 200,
            "output_path": out,
        })
        assert res["status"] == "success"
        assert res["slab_provided_in"] >= res["hd_required_in"] - 0.51

    def test_compare_package(self, tmp_path):
        pytest.importorskip("geotech_references.ufc_pavement")
        out = str(tmp_path / "cmp.html")
        res = call_agent("calc_package", "pavement_design_package", {
            "design_type": "compare", "passes_18kip": 1e6,
            "cbr_subgrade": 6, "output_path": out,
        })
        assert res["status"] == "success"
        assert res["delta_total_thickness_in"] == pytest.approx(
            res["ufc_total_in"] - res["aashto_total_in"], abs=0.02)
        html = open(out, encoding="utf-8").read()
        assert "Method Comparison" in html
