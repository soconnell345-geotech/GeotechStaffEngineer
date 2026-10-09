"""Adapter follow-ups to the analysis-module fixes of e01db59 (live smoke
G6, G8-G10, G17, G19, G20): the agent can reach every new option, and each
METHOD_INFO says what the code takes and returns.

One class per item:
  1. salib one-call sobol_analysis / morris_analysis (expression or model spec)
  2. seismic_signals: no pyrotd gate, returns doc keys, Sa / RotD spectra
  3. target_pga_g through seismic_signals, pystrata, opensees
  4. fem2d consolidation per-time keys documented
  5. settlement combined analysis: total_mm, time curve, t50/t90
  6. gec10 cites FHWA-NHI-18-024
  7. downdrag: mobilized Nt, basis + warnings in the adapter and calc package
  8. limit-state expressions accept scientific notation, stay AST-whitelisted
"""

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from funhouse_agent.dispatch import call_agent, describe_method


def _info(agent, method):
    from funhouse_agent.dispatch import _load_adapter
    return _load_adapter(agent).METHOD_INFO[method]


def _assert_documented(agent, method, result, *, conditional=()):
    """Every documented return key is in the result (except the ones the
    method only returns in some cases), and no documented key is stale."""
    assert "error" not in result, result
    documented = set(_info(agent, method)["returns"])
    missing = documented - set(result) - set(conditional)
    assert not missing, f"{agent}.{method} documents {sorted(missing)} " \
                        f"but returned {sorted(result)}"


# ---------------------------------------------------------------------------
# 1. salib one-call analyses
# ---------------------------------------------------------------------------

def _has_salib():
    from salib_agent import has_salib
    return has_salib()


_BC_SPEC = {
    "agent_name": "bearing_capacity",
    "method": "bearing_capacity_analysis",
    "parameters": {"width": 2.0, "depth": 1.0, "unit_weight": 18.0},
    "output": "q_ultimate_kPa",
    "variable_map": {"phi": "friction_angle", "c": "cohesion"},
}


@pytest.mark.skipif(not _has_salib(), reason="SALib not installed")
class TestSalibOneCall:
    def test_sobol_analysis_expression(self):
        r = call_agent("salib", "sobol_analysis", {
            "var_names": ["a", "b", "c"],
            "bounds": [[0, 1], [0, 1], [0, 1]],
            "expression": "a + 2*b + 1e-3*c",
        })
        assert "error" not in r, r
        assert "sample_matrix" not in r
        # Var(a) : Var(2b) = 1 : 4, c negligible
        assert r["S1"][1] == pytest.approx(0.8, abs=0.05)
        assert r["S1"][0] == pytest.approx(0.2, abs=0.05)
        assert abs(r["ST"][2]) < 0.01
        assert r["model_evaluations"] == 256 * (3 + 2)
        assert r["model"] == {"kind": "expression",
                              "expression": "a + 2*b + 1e-3*c"}
        _assert_documented("salib", "sobol_analysis", r, conditional=("S2",))

    def test_morris_analysis_model_spec(self):
        r = call_agent("salib", "morris_analysis", {
            "var_names": ["phi", "c"], "bounds": [[28, 36], [0, 20]],
            "model": _BC_SPEC,
        })
        assert "error" not in r, r
        assert r["model_evaluations"] == 20 * (2 + 1)
        assert r["model"]["kind"] == "module"
        assert r["model"]["agent_name"] == "bearing_capacity"
        assert all(m > 0 for m in r["mu_star"])
        _assert_documented("salib", "morris_analysis", r)

    def test_model_spec_matches_direct_call(self):
        """The spec runs the module: output_mean lies inside the range the
        module gives at the bound corners."""
        lo = call_agent("bearing_capacity", "bearing_capacity_analysis", {
            "width": 2.0, "depth": 1.0, "unit_weight": 18.0,
            "friction_angle": 28, "cohesion": 0})["q_ultimate_kPa"]
        hi = call_agent("bearing_capacity", "bearing_capacity_analysis", {
            "width": 2.0, "depth": 1.0, "unit_weight": 18.0,
            "friction_angle": 36, "cohesion": 20})["q_ultimate_kPa"]
        r = call_agent("salib", "sobol_analysis", {
            "var_names": ["phi", "c"], "bounds": [[28, 36], [0, 20]],
            "model": _BC_SPEC, "n_samples": 64})
        assert lo < r["output_mean"] < hi

    def test_model_spec_refuses_non_calculation_module(self):
        r = call_agent("salib", "sobol_analysis", {
            "var_names": ["a", "b"], "bounds": [[0, 1], [0, 1]],
            "model": {"agent_name": "calc_package", "method": "html_to_pdf",
                      "output": "x"}})
        assert "not allowed" in r["error"]

    def test_model_spec_refuses_recursion(self):
        r = call_agent("salib", "sobol_analysis", {
            "var_names": ["a", "b"], "bounds": [[0, 1], [0, 1]],
            "model": {"agent_name": "salib", "method": "sobol_analysis",
                      "output": "S1"}})
        assert "not allowed" in r["error"]

    def test_model_spec_refuses_file_parameters(self):
        spec = dict(_BC_SPEC, parameters=dict(_BC_SPEC["parameters"],
                                              output_path="x.html"))
        r = call_agent("salib", "sobol_analysis", {
            "var_names": ["phi", "c"], "bounds": [[28, 36], [0, 20]],
            "model": spec})
        assert "may not name files" in r["error"]

    def test_bad_output_lists_numeric_outputs(self):
        spec = dict(_BC_SPEC, output="q_ult")
        r = call_agent("salib", "sobol_analysis", {
            "var_names": ["phi", "c"], "bounds": [[28, 36], [0, 20]],
            "model": spec})
        assert "trial run" in r["error"]
        assert "q_ultimate_kPa" in r["error"]

    def test_time_budget_refuses_an_expensive_design(self):
        r = call_agent("salib", "sobol_analysis", {
            "var_names": ["phi", "c"], "bounds": [[28, 36], [0, 20]],
            "model": _BC_SPEC, "time_budget_s": 1e-6})
        assert "time budget" in r["error"]

    def test_exactly_one_of_expression_or_model(self):
        neither = call_agent("salib", "sobol_analysis", {
            "var_names": ["a", "b"], "bounds": [[0, 1], [0, 1]]})
        both = call_agent("salib", "sobol_analysis", {
            "var_names": ["a", "b"], "bounds": [[0, 1], [0, 1]],
            "expression": "a+b", "model": _BC_SPEC})
        assert "exactly one" in neither["error"]
        assert "exactly one" in both["error"]

    def test_expression_stays_ast_whitelisted(self):
        for bad in ("__import__('os')", "a.__class__", "[a][0]", "'s'"):
            r = call_agent("salib", "morris_analysis", {
                "var_names": ["a", "b"], "bounds": [[0, 1], [0, 1]],
                "expression": bad})
            assert "error" in r, bad

    def test_method_info_required_flags(self):
        for m in ("sobol_analysis", "morris_analysis"):
            p = _info("salib", m)["parameters"]
            assert p["var_names"]["required"] and p["bounds"]["required"]
            assert not p["expression"]["required"]
            assert not p["model"]["required"]
        assert _info("salib", "morris_analyze")["parameters"]["X"]["required"]


# ---------------------------------------------------------------------------
# 2. seismic_signals: no pyrotd gate, returns doc keys, spectra
# ---------------------------------------------------------------------------

def _has_eqsig():
    from seismic_signals_agent import has_eqsig
    return has_eqsig()


class TestSeismicSignalsReturns:
    def test_rotd_has_no_pyrotd_gate(self):
        from funhouse_agent.adapters import seismic_signals_adapter as ssa
        assert not hasattr(ssa, "_check_pyrotd")

    def test_rotd_returns_every_percentile_spectrum(self):
        r = call_agent("seismic_signals", "rotd_spectrum", {
            "motion_a": "synthetic_pulse", "motion_b": "synthetic_long",
            "periods": [0.1, 0.5, 1.0], "percentiles": [0, 50, 84, 100]})
        assert r["percentiles"] == ["0", "50", "84", "100"]
        for p in r["percentiles"]:
            assert len(r["spectra"][p]) == 3
        assert r["engine"] in ("pyrotd", "numpy")
        _assert_documented("seismic_signals", "rotd_spectrum", r,
                           conditional=("engine_note",))

    @pytest.mark.skipif(not _has_eqsig(), reason="eqsig not installed")
    def test_intensity_measures_doc_keys(self):
        r = call_agent("seismic_signals", "intensity_measures",
                       {"motion": "synthetic_pulse"})
        for k in ("pgv_m_per_s", "pgd_m", "cav_m_per_s"):
            assert k in r
        returns = _info("seismic_signals", "intensity_measures")["returns"]
        for stale in ("pgv_cm_per_s", "pgd_cm", "CAV_m_per_s"):
            assert stale not in returns
        _assert_documented("seismic_signals", "intensity_measures", r)

    @pytest.mark.skipif(not _has_eqsig(), reason="eqsig not installed")
    def test_response_spectrum_returns_sa(self):
        r = call_agent("seismic_signals", "response_spectrum", {
            "motion": "synthetic_pulse", "periods": [0.1, 0.3, 1.0]})
        assert r["periods_s"] == [0.1, 0.3, 1.0]
        assert len(r["Sa_g"]) == 3
        assert r["Sa_max_g"] == max(r["Sa_g"])
        _assert_documented("seismic_signals", "response_spectrum", r)

    @pytest.mark.skipif(not _has_eqsig(), reason="eqsig not installed")
    def test_signal_processing_doc_keys(self):
        r = call_agent("seismic_signals", "signal_processing", {
            "motion": "synthetic_pulse", "baseline_order": 1})
        _assert_documented("seismic_signals", "signal_processing", r)


# ---------------------------------------------------------------------------
# 3. target_pga_g wherever the module takes it
# ---------------------------------------------------------------------------

class TestTargetPga:
    @pytest.mark.parametrize("agent,method", [
        ("seismic_signals", "response_spectrum"),
        ("seismic_signals", "intensity_measures"),
        ("seismic_signals", "signal_processing"),
        ("pystrata", "eql_site_response"),
        ("pystrata", "linear_site_response"),
        ("opensees", "site_response_1d"),
    ])
    def test_declared(self, agent, method):
        assert "target_pga_g" in describe_method(agent, method)["parameters"]

    @pytest.mark.skipif(not _has_eqsig(), reason="eqsig not installed")
    @pytest.mark.parametrize("method,extra,pga_key", [
        ("response_spectrum", {"periods": [0.2]}, "pga_g"),
        ("intensity_measures", {}, "pga_g"),
        ("signal_processing", {"baseline_order": 1}, "pga_original_g"),
    ])
    def test_seismic_signals_scales(self, method, extra, pga_key):
        r = call_agent("seismic_signals", method, dict(
            motion="synthetic_pulse", target_pga_g=0.2, **extra))
        assert r[pga_key] == pytest.approx(0.2, abs=1e-4)
        assert "scaled to PGA 0.2 g" in r["motion_name"]

    def test_pystrata_scales(self):
        from pystrata_agent import has_pystrata
        if not has_pystrata():
            pytest.skip("pystrata not installed")
        layers = [
            {"thickness": 10, "Vs": 200, "unit_wt": 18,
             "soil_model": "linear", "damping": 0.02},
            {"thickness": 0, "Vs": 760, "unit_wt": 22,
             "soil_model": "linear", "damping": 0.01},
        ]
        r = call_agent("pystrata", "linear_site_response", {
            "layers": layers, "motion": "synthetic_pulse",
            "target_pga_g": 0.2})
        assert r["pga_input_g"] == pytest.approx(0.2, abs=1e-4)
        _assert_documented("pystrata", "linear_site_response", r,
                           conditional=("max_shear_strain_pct",))

    def test_opensees_passes_it_through(self):
        fake = MagicMock()
        fake.return_value.to_dict.return_value = {"pga_surface_g": 0.3}
        with patch("opensees_agent.has_opensees", return_value=True), \
                patch("opensees_agent.analyze_site_response", fake):
            r = call_agent("opensees", "site_response_1d", {
                "layers": [{"thickness": 5, "Vs": 200, "density": 1.9,
                            "material_type": "clay", "su": 50}],
                "motion": "synthetic_pulse", "target_pga_g": 0.25})
        assert "error" not in r, r
        assert fake.call_args.kwargs["target_pga_g"] == 0.25


# ---------------------------------------------------------------------------
# 4. fem2d consolidation: U and settlement per requested time
# ---------------------------------------------------------------------------

class TestFem2dConsolidationPerTime:
    def test_per_time_keys_returned_and_documented(self):
        K, G, M = 500000.0, 200000.0, 4e6
        E = 9 * K * G / (3 * K + G)
        nu = (3 * K - 2 * G) / (2 * (3 * K + G))
        r = call_agent("fem2d", "fem2d_consolidation", {
            "width": 2.0, "depth": 20.0,
            "soil_layers": [{"bottom_elevation": -20.0, "E": E, "nu": nu,
                             "gamma": 0.0}],
            "k": 1e-10, "load_q": 100.0, "time_points": [1e3, 1e7, 1e8],
            "consolidation_scheme": "monolithic", "theta": 0.5, "n_w": M,
            "nx": 3, "ny": 20})
        _assert_documented("fem2d", "fem2d_consolidation", r)
        assert r["time_s"] == [0.0, 1e3, 1e7, 1e8]
        for k in ("degree_of_consolidation_by_time",
                  "surface_settlement_m_by_time",
                  "max_excess_pore_pressure_kPa_by_time"):
            assert len(r[k]) == 4, k
        # Terzaghi: U = 0.985 at 1e7 s, 1.000 at 1e8 s (CON-1 key)
        assert r["degree_of_consolidation_by_time"][2] == \
            pytest.approx(0.985, abs=0.02)
        assert r["n_time_steps"] == 4

    def test_time_points_described_as_output_times(self):
        desc = _info("fem2d", "fem2d_consolidation")[
            "parameters"]["time_points"]["description"]
        assert "OUTPUT times" in desc and "t = 0" in desc


# ---------------------------------------------------------------------------
# 5. settlement combined analysis: total_mm, time curve, t50/t90
# ---------------------------------------------------------------------------

class TestSettlementCombinedReturns:
    def test_total_mm_and_time_curve(self):
        r = call_agent("settlement", "combined_settlement_analysis", {
            "q_applied": 100, "B": 2, "Es": 20000, "cv": 2.0,
            "consolidation_layers": [{
                "thickness": 4, "depth_to_center": 4, "e0": 1.0, "Cc": 0.3,
                "Cr": 0.03, "sigma_v0": 60}]})
        _assert_documented("settlement", "combined_settlement_analysis", r)
        returns = _info("settlement", "combined_settlement_analysis")["returns"]
        assert "total_settlement_mm" not in returns
        # Hdr = 4/2 (double drainage); t90 = 0.848 Hdr^2 / cv
        assert r["Hdr_m"] == pytest.approx(2.0)
        assert r["t90_years"] == pytest.approx(0.848 * 4 / 2.0, rel=0.01)
        assert r["t50_years"] == pytest.approx(0.197 * 4 / 2.0, rel=0.01)
        pt = r["time_settlement_curve"][-1]
        assert set(pt) == {"time_years", "settlement_mm", "U_percent"}
        assert r["total_mm"] == pytest.approx(
            r["immediate_mm"] + r["consolidation_mm"] + r["secondary_mm"],
            abs=0.02)


# ---------------------------------------------------------------------------
# 6. gec10 citation
# ---------------------------------------------------------------------------

class TestGec10Citation:
    def test_every_method_cites_2018_edition(self):
        from funhouse_agent.adapters import gec10_adapter
        refs = {info.get("reference")
                for info in gec10_adapter.METHOD_INFO.values()}
        assert refs == {"FHWA-NHI-18-024"}

    def test_describe_method_shows_it(self):
        from funhouse_agent.adapters import gec10_adapter
        m = next(iter(gec10_adapter.METHOD_INFO))
        assert "18-024" in json.dumps(describe_method("gec10", m))
        assert "10-016" not in json.dumps(describe_method("gec10", m))


# ---------------------------------------------------------------------------
# 7. downdrag: mobilized Nt; basis and warnings in adapter and calc package
# ---------------------------------------------------------------------------

_DD1_LAYERS = [
    {"thickness": 8, "soil_type": "cohesive", "unit_weight": 17, "cu": 30,
     "alpha": 0.5, "settling": True, "Cc": 0.25, "e0": 0.9},
    {"thickness": 7, "soil_type": "cohesionless", "unit_weight": 19,
     "phi": 38, "beta": 0.3},
]
_DD1 = {"gwt_depth": 1.0, "pile_length": 15, "pile_diameter": 0.4,
        "fill_thickness": 2, "fill_unit_weight": 20, "Q_dead": 0}


class TestDowndragBasis:
    def test_nt_described_as_mobilized(self):
        for agent, method in (("downdrag", "downdrag_analysis"),
                              ("calc_package", "downdrag_package")):
            desc = describe_method(agent, method)["parameters"]["Nt"][
                "description"]
            assert "MOBILIZED" in desc, (agent, method)

    def test_basis_and_warnings_lead_the_result(self):
        r = call_agent("downdrag", "downdrag_analysis",
                       dict(_DD1, layers=_DD1_LAYERS))
        assert r["neutral_plane_method"] == "settlement_compatibility"
        assert r["neutral_plane_basis"].startswith("Settlement compatibility")
        assert r["warnings"] and "do not intersect" in r["warnings"][0]
        keys = list(r)
        assert keys[:4] == ["neutral_plane_depth_m", "neutral_plane_method",
                            "neutral_plane_basis", "warnings"]
        # Ahead of the depth profiles, so a size-capped result keeps them.
        assert keys.index("warnings") < keys.index("z_m")
        _assert_documented("downdrag", "downdrag_analysis", r,
                           conditional=("structural_ok",))

    def test_mobilized_nt_gives_force_equilibrium(self):
        r = call_agent("downdrag", "downdrag_analysis",
                       dict(_DD1, layers=_DD1_LAYERS, Nt=5.0))
        assert r["neutral_plane_method"] == "force_equilibrium"

    def test_calc_package_carries_basis_and_warnings(self, tmp_path):
        out = tmp_path / "dd.html"
        r = call_agent("calc_package", "downdrag_package", dict(
            _DD1, soil_layers=_DD1_LAYERS, pile_E=200e6,
            output_path=str(out)))
        assert r.get("status") == "success", r
        assert r["neutral_plane_method"] == "settlement_compatibility"
        assert r["neutral_plane_basis"]
        assert r["warnings"]
        html = out.read_text(encoding="utf-8")
        assert "Neutral Plane Basis and Warnings" in html
        assert "WARNING: The load and resistance curves" in html

    def test_calc_package_takes_settling_sand_modulus(self, tmp_path):
        """Layer keys match downdrag_analysis: a settling cohesionless layer
        needs E_s, which the package used to drop."""
        layers = [
            {"thickness": 6, "soil_type": "cohesionless", "unit_weight": 18,
             "phi": 30, "settling": True, "E_s": 8000},
            {"thickness": 9, "soil_type": "cohesionless", "unit_weight": 19,
             "phi": 38},
        ]
        r = call_agent("calc_package", "downdrag_package", dict(
            _DD1, soil_layers=layers, pile_E=200e6,
            output_path=str(tmp_path / "dd2.html")))
        assert r.get("status") == "success", r

    def test_calc_steps_basis_section_empty_without_basis(self):
        from downdrag.calc_steps import basis_and_warnings

        class Old:
            pass
        assert basis_and_warnings(Old()) == []


# ---------------------------------------------------------------------------
# 8. limit-state expressions: scientific notation, still AST-whitelisted
# ---------------------------------------------------------------------------

_REL_VARS = {"R": {"mean": 200, "cov": 0.1, "dist": "normal"},
             "S": {"mean": 100, "cov": 0.2, "dist": "normal"}}


class TestScientificNotation:
    @pytest.mark.parametrize("expr", ["R - 1e0*S", "R - 1.0E0*S + 1e-9",
                                      "R - 1000e-3*S"])
    def test_reliability_accepts(self, expr):
        r = call_agent("reliability", "fosm", {
            "variables": _REL_VARS, "g_expression": expr,
            "convention": "margin"})
        assert "error" not in r, r
        # beta = 100 / sqrt(20^2 + 20^2)
        assert r["beta_normal"] == pytest.approx(100 / math.sqrt(800),
                                                 abs=1e-3)

    @pytest.mark.parametrize("expr,why", [
        ("R - e", "Unknown identifier 'e'"),
        ("R - S.real", "Attribute"),
        ("__import__('os').getcwd()", "Unknown identifier"),
        ("R - (lambda: 1)()", "Lambda|function calls are limited"),
    ])
    def test_reliability_still_refuses(self, expr, why):
        r = call_agent("reliability", "fosm", {
            "variables": _REL_VARS, "g_expression": expr})
        assert "error" in r
        import re
        assert re.search(why, r["error"]), r["error"]

    def test_pystra_utils_compiler(self):
        from pystra_agent.pystra_utils import _compile_limit_state
        f = _compile_limit_state("R - 1e-3*S", ["R", "S"])
        assert f(R=200.0, S=1000.0) == pytest.approx(199.0)
        for bad in ("R - e", "R.__class__", "R[0]", "'x'", "f(R)"):
            with pytest.raises(ValueError):
                _compile_limit_state(bad, ["R", "S"])

    def test_pystra_form_accepts(self):
        from pystra_agent import has_pystra
        if not has_pystra():
            pytest.skip("pystra not installed")
        v = [{"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
             {"name": "S", "dist": "normal", "mean": 100, "stdv": 20}]
        r = call_agent("pystra", "form_analysis",
                       {"variables": v, "limit_state": "R - 1e3*1e-3*S"})
        assert "error" not in r, r
        assert r["beta"] == pytest.approx(100 / math.sqrt(800), abs=1e-3)
