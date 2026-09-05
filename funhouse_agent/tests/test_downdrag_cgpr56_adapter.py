"""Tests for the CGPR #56 downdrag adapter methods through dispatch.

SI inputs converted from the CGPR #56 Section 3.4 worked example
(100 ft = 30.48 m pile, 16 in square, group load 2000 kips / 9 piles);
published answers in report units are cross-checked after conversion.
"""

import json

import pytest

from funhouse_agent.dispatch import call_agent, describe_method, list_methods

FT = 0.3048
KIP = 4.448222
KSF = 47.88026  # kPa

Q_STATIC = 2000.0 / 9.0 * KIP          # kN
PILE_LENGTH = 100.0 * FT               # m
PILE_WIDTH = (16.0 / 12.0) * FT        # m
PERIMETER = 4.0 * PILE_WIDTH
AREA = PILE_WIDTH**2
PILE_E = 500e3 * KSF                   # kPa
QB = 200.0 * KSF                       # kPa
BEARING_E = 8e3 * KSF

FS_PROFILE = [[z * FT, fs * KSF] for z, fs in [
    (0.0, 0.0), (15.0, 0.8), (20.0, 0.9),
    (20.0, 0.6), (90.0, 1.2), (90.0, 2.0), (100.0, 2.3),
]]

SETTLEMENT_PROFILE = [[z * FT, s * FT] for z, s in [
    (0.0, 0.086), (20.0, 0.086), (30.0, 0.067), (40.0, 0.052),
    (50.0, 0.039), (60.0, 0.028), (70.0, 0.017), (80.0, 0.008), (90.0, 0.0),
]]

CONSOLIDATION = {
    "layers": [{"z_top": 20.0 * FT, "z_bot": 90.0 * FT, "C_er": 0.015}],
    "p0_profile": [[0.0, 0.0], [5.0 * FT, 0.550 * KSF],
                   [20.0 * FT, 1.414 * KSF], [90.0 * FT, 4.046 * KSF]],
    "dp_profile": 0.524 * KSF,
    "sublayer_thickness": 5.0 * FT,
}


def _call(method, params):
    result = call_agent("downdrag", method, params)
    if isinstance(result, str):
        result = json.loads(result)
    return result


class TestRegistration:
    def test_methods_listed(self):
        res = list_methods("downdrag")
        names = {name for cat in res.values() for name in cat}
        assert {"downdrag_analysis", "endo_downdrag", "poulos_downdrag",
                "fellenius_cgpr56", "pileneg_downdrag",
                "downdrag_method_comparison", "group_rigid_block",
                "group_drag_reduction"} <= names

    def test_describe_methods(self):
        for m in ("endo_downdrag", "poulos_downdrag", "fellenius_cgpr56",
                  "pileneg_downdrag", "downdrag_method_comparison",
                  "group_rigid_block", "group_drag_reduction"):
            res = describe_method("downdrag", m)
            assert "parameters" in res, m


class TestEndoAdapter:
    def test_worked_example(self):
        res = _call("endo_downdrag", {
            "Q_static": Q_STATIC, "pile_length": PILE_LENGTH,
            "pile_perimeter": PERIMETER,
            "skin_friction_profile": FS_PROFILE,
            "bearing_condition": "stiff_flexible",
        })
        assert res["neutral_plane_depth"] == pytest.approx(75 * FT, rel=1e-6)
        assert res["max_force"] == pytest.approx(522 * KIP, rel=0.005)

    def test_bad_condition_message(self):
        res = _call("endo_downdrag", {
            "Q_static": Q_STATIC, "pile_length": PILE_LENGTH,
            "pile_perimeter": PERIMETER,
            "skin_friction_profile": FS_PROFILE,
            "bearing_condition": "rock",
        })
        assert "error" in res and "bearing_condition" in res["error"]

    def test_bad_profile_message(self):
        res = _call("endo_downdrag", {
            "Q_static": Q_STATIC, "pile_length": PILE_LENGTH,
            "pile_perimeter": PERIMETER,
            "skin_friction_profile": [0.0, 15.0],
            "bearing_condition": "floating",
        })
        assert "error" in res and "pairs" in res["error"]

    def test_unknown_param_rejected(self):
        res = _call("endo_downdrag", {
            "Q_static": Q_STATIC, "pile_length": PILE_LENGTH,
            "pile_perimeter": PERIMETER,
            "skin_friction_profile": FS_PROFILE,
            "bearing_condition": "floating", "pile_diamter": 0.4,
        })
        assert "error" in res and "unknown parameter" in res["error"]


class TestPoulosAdapter:
    def test_worked_example(self):
        res = _call("poulos_downdrag", {
            "Q_static": Q_STATIC, "pile_length": PILE_LENGTH,
            "pile_perimeter": PERIMETER, "pile_area": AREA,
            "pile_E": PILE_E, "depth_to_bearing_layer": 90.0 * FT,
            "toe_bearing_capacity": QB,
            "skin_friction_profile": FS_PROFILE,
            "settlement_profile": SETTLEMENT_PROFILE,
        })
        assert res["neutral_plane_depth"] == pytest.approx(73.6 * FT, abs=0.05)
        assert res["max_force"] == pytest.approx(542 * KIP, rel=0.005)
        assert res["pile_settlement"] == pytest.approx(0.046 * FT, abs=0.0005)


class TestFelleniusCgpr56Adapter:
    def test_worked_example_with_consolidation(self):
        res = _call("fellenius_cgpr56", {
            "Q_static": Q_STATIC, "pile_length": PILE_LENGTH,
            "pile_perimeter": PERIMETER, "pile_area": AREA,
            "pile_E": PILE_E, "toe_bearing_capacity": QB,
            "skin_friction_profile": FS_PROFILE,
            "consolidation": CONSOLIDATION,
            "eq_footing_width": 11.33 * FT,
            "eq_footing_load": 2000.0 * KIP,
        })
        assert res["neutral_plane_depth"] == pytest.approx(78.4 * FT, abs=0.05)
        assert res["max_force"] == pytest.approx(541 * KIP, rel=0.005)
        assert res["pile_settlement"] == pytest.approx(0.118 * FT, abs=0.001)
        assert res["includes_pile_load_transfer"] is True

    def test_consolidation_requires_width(self):
        res = _call("fellenius_cgpr56", {
            "Q_static": Q_STATIC, "pile_length": PILE_LENGTH,
            "pile_perimeter": PERIMETER, "pile_area": AREA,
            "pile_E": PILE_E, "toe_bearing_capacity": QB,
            "skin_friction_profile": FS_PROFILE,
            "consolidation": CONSOLIDATION,
        })
        assert "error" in res and "eq_footing_width" in res["error"]


class TestPilenegAdapter:
    def test_worked_example_report_grid(self):
        res = _call("pileneg_downdrag", {
            "Q_static": Q_STATIC, "pile_length": PILE_LENGTH,
            "pile_perimeter": PERIMETER, "pile_area": AREA,
            "pile_E": PILE_E, "toe_bearing_capacity": QB,
            "pile_width": PILE_WIDTH, "bearing_E": BEARING_E,
            "skin_friction_profile": FS_PROFILE,
            "settlement_profile": SETTLEMENT_PROFILE,
            "trial_depths": [50.0 * FT, 60.0 * FT, 70.0 * FT, 80.0 * FT],
        })
        assert res["neutral_plane_depth"] == pytest.approx(62.4 * FT, abs=0.05)
        assert res["max_force"] == pytest.approx(453 * KIP, rel=0.005)
        assert res["toe_load"] == pytest.approx(180 * KIP, rel=0.01)
        assert res["toe_fully_mobilized"] is False


class TestComparisonAdapter:
    def test_full_run(self):
        res = _call("downdrag_method_comparison", {
            "Q_static": Q_STATIC, "pile_length": PILE_LENGTH,
            "pile_perimeter": PERIMETER, "pile_area": AREA,
            "pile_E": PILE_E, "toe_bearing_capacity": QB,
            "skin_friction_profile": FS_PROFILE,
            "settlement_profile": SETTLEMENT_PROFILE,
            "consolidation": CONSOLIDATION,
            "endo_bearing_condition": "stiff_flexible",
            "depth_to_bearing_layer": 90.0 * FT,
            "pile_width": PILE_WIDTH, "bearing_E": BEARING_E,
            "eq_footing_width": 11.33 * FT, "eq_footing_load": 2000.0 * KIP,
        })
        rows = {r["method"]: r for r in res["comparison_table"]}
        assert not any(r.get("error") for r in res["comparison_table"]), rows
        assert rows["endo"]["max_force"] == pytest.approx(522 * KIP, rel=0.005)
        assert rows["poulos"]["max_force"] == pytest.approx(542 * KIP, rel=0.005)
        assert rows["fellenius_cgpr56"]["max_force"] == pytest.approx(
            541 * KIP, rel=0.005)
        assert rows["pileneg"]["max_force"] == pytest.approx(
            453 * KIP, rel=0.01)
        assert "summary_table" in res
        json.dumps(res)  # JSON-safe

    def test_partial_inputs_skip_rows(self):
        res = _call("downdrag_method_comparison", {
            "Q_static": Q_STATIC, "pile_length": PILE_LENGTH,
            "pile_perimeter": PERIMETER, "pile_area": AREA,
            "pile_E": PILE_E, "toe_bearing_capacity": QB,
            "skin_friction_profile": FS_PROFILE,
        })
        rows = {r["method"]: r for r in res["comparison_table"]}
        assert "error" in rows["endo"]
        assert "error" in rows["pileneg"]
        assert rows["fellenius_cgpr56"]["max_force"] == pytest.approx(
            541 * KIP, rel=0.005)


class TestGroupAdapters:
    def test_rigid_block(self):
        res = _call("group_rigid_block", {
            "Q_static_group": 2000.0 * KIP, "n_piles": 9,
            "spacing": 5.0 * FT, "neutral_plane_depth": 78.4 * FT,
            "cu_average": 0.5 * KSF, "delta_q": 0.524 * KSF,
        })
        expected = (2000.0 / 9.0 + 5.0 * 78.4 * 0.5) * KIP
        assert res["perimeter_pile_max_force"] == pytest.approx(
            expected, rel=1e-5)

    def test_drag_reduction(self):
        res = _call("group_drag_reduction", {
            "F_max_single": 541.0 * KIP, "Q_static_group": 2000.0 * KIP,
            "n_piles": 9, "location": "corner",
            "spacing": 5.0 * FT, "pile_diameter": PILE_WIDTH,
        })
        assert res["s_over_d"] == pytest.approx(3.75)
        assert res["reduction_factor"] == pytest.approx(0.7)
        assert res["max_force_group_pile"] == pytest.approx(
            445.4 * KIP, rel=0.005)

    def test_reduction_location_validated(self):
        res = _call("group_drag_reduction", {
            "F_max_single": 541.0, "Q_static_group": 2000.0,
            "n_piles": 9, "location": "edge", "s_over_d": 3.0,
        })
        assert "error" in res and "location" in res["error"]
