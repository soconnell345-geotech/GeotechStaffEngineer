"""Tests for the unified liquefaction router (consolidation #5).

The single `liquefaction` module auto-routes a triggering request by input type
(CPT vs SPT) and method (B&I-2014 default / NCEER-2001 behind a flag):

    CPT (q_c/f_s)      -> liquepy B&I-2014 CPT (LPI/LSN/LDI).
    SPT (N160), bi2014 -> liquepy B&I-2014 SPT.
    SPT (N160), nceer2001 -> seismic_geotech NCEER/Youd-2001.

Covers: input-type detection, method flags, error paths, discoverability, and
consistency with the underlying per-module outputs.
"""

import numpy as np
import pytest

from liquepy_agent.liquepy_utils import has_liquepy
from funhouse_agent.dispatch import (
    call_agent, list_agents, list_methods,
    ANALYSIS_MODULES, REFERENCE_MODULES,
)
from funhouse_agent.adapters.liquefaction_adapter import (
    METHOD_INFO, METHOD_REGISTRY, _detect_input_type, _normalize_method,
)

requires_liquepy = pytest.mark.skipif(
    not has_liquepy(), reason="liquepy not installed"
)

REQUIRED_INFO_FIELDS = {"category", "brief", "parameters", "returns"}


def _spt_params(method=None):
    p = {
        "depth": [3, 6, 9, 12], "N160": [8, 12, 25, 10],
        "FC": [5, 10, 15, 5], "gamma": [18, 18, 18, 18],
        "amax_g": 0.35, "gwt_depth": 2.0, "m_w": 7.0,
    }
    if method is not None:
        p["method"] = method
    return p


def _cpt_params(method=None):
    depth = np.arange(0.5, 15.0, 0.5)
    p = {
        "depth": depth.tolist(),
        "q_c": (np.ones_like(depth) * 5000).tolist(),
        "f_s": (np.ones_like(depth) * 50).tolist(),
        "gwl": 1.5, "pga": 0.3, "m_w": 7.5,
    }
    if method is not None:
        p["method"] = method
    return p


# ============================================================================
# Adapter metadata + discoverability
# ============================================================================

class TestLiquefactionAdapterMetadata:
    def test_info_registry_keys_match(self):
        assert set(METHOD_INFO) == set(METHOD_REGISTRY)
        for name, info in METHOD_INFO.items():
            for f in REQUIRED_INFO_FIELDS:
                assert f in info, f"{name} missing {f}"

    def test_single_method(self):
        assert list(METHOD_REGISTRY) == ["liquefaction_analysis"]

    def test_discoverable_in_list_agents(self):
        assert "liquefaction" in list_agents()

    def test_is_analysis_not_reference(self):
        assert "liquefaction" in ANALYSIS_MODULES
        assert "liquefaction" not in REFERENCE_MODULES

    def test_list_methods_via_dispatch(self):
        lm = list_methods("liquefaction")
        assert "Liquefaction" in lm
        assert "liquefaction_analysis" in lm["Liquefaction"]

    def test_method_allows_both_flags(self):
        allowed = METHOD_INFO["liquefaction_analysis"]["parameters"]["method"]["allowed_values"]
        assert set(allowed) == {"bi2014", "nceer2001"}


# ============================================================================
# Input-type detection + method normalization (pure, no liquepy)
# ============================================================================

class TestInputTypeDetection:
    def test_detects_cpt(self):
        assert _detect_input_type({"q_c": [1], "f_s": [1]}) == "CPT"

    def test_detects_spt(self):
        assert _detect_input_type({"N160": [10]}) == "SPT"

    def test_ambiguous_raises(self):
        with pytest.raises(ValueError):
            _detect_input_type({"q_c": [1], "N160": [10]})

    def test_none_raises(self):
        with pytest.raises(ValueError):
            _detect_input_type({"depth": [1]})

    def test_explicit_override(self):
        # Both present, but explicit input_type wins.
        assert _detect_input_type({"q_c": [1], "N160": [10], "input_type": "SPT"}) == "SPT"

    def test_default_method_is_bi2014(self):
        assert _normalize_method({}, "SPT") == "bi2014"
        assert _normalize_method({}, "CPT") == "bi2014"

    def test_nceer_aliases(self):
        for name in ["nceer2001", "nceer", "youd2001", "youd", "simplified"]:
            assert _normalize_method({"method": name}, "SPT") == "nceer2001"

    def test_bi_aliases(self):
        for name in ["bi2014", "boulanger_idriss", "b&i"]:
            assert _normalize_method({"method": name}, "SPT") == "bi2014"

    def test_nceer_rejected_for_cpt(self):
        with pytest.raises(ValueError):
            _normalize_method({"method": "nceer2001"}, "CPT")

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            _normalize_method({"method": "bogus"}, "SPT")


# ============================================================================
# SPT routing (NCEER-2001 path works without liquepy)
# ============================================================================

class TestSptNceer2001Routing:
    def test_routes_to_nceer(self):
        out = call_agent("liquefaction", "liquefaction_analysis", _spt_params("nceer2001"))
        assert out.get("method") == "nceer2001"
        assert out.get("input_type") == "SPT"
        assert out["n_layers"] == 4
        assert "layer_results" in out

    def test_matches_underlying_seismic_geotech(self):
        from seismic_geotech.liquefaction import evaluate_liquefaction
        direct = evaluate_liquefaction(
            layer_depths=[3, 6, 9, 12], layer_N160=[8, 12, 25, 10],
            layer_FC=[5, 10, 15, 5], layer_gamma=[18, 18, 18, 18],
            amax_g=0.35, gwt_depth=2.0, M=7.0,
        )
        out = call_agent("liquefaction", "liquefaction_analysis", _spt_params("nceer2001"))
        direct_min = min(r["FOS_liq"] for r in direct)
        assert out["min_fos"] == pytest.approx(direct_min, abs=1e-6)
        assert len(out["layer_results"]) == len(direct)


# ============================================================================
# SPT + CPT B&I-2014 routing (requires liquepy)
# ============================================================================

@requires_liquepy
class TestBi2014Routing:
    def test_spt_default_is_bi2014(self):
        out = call_agent("liquefaction", "liquefaction_analysis", _spt_params())
        assert out.get("method") == "bi2014"
        assert out.get("input_type") == "SPT"

    def test_spt_bi2014_matches_underlying(self):
        from liquepy_agent import analyze_spt_liquefaction
        direct = analyze_spt_liquefaction(
            depth=[3, 6, 9, 12], n1_60=[8, 12, 25, 10],
            fc=[5, 10, 15, 5], gamma=[18, 18, 18, 18],
            amax_g=0.35, gwt_depth=2.0, m_w=7.0,
        ).to_dict()
        out = call_agent("liquefaction", "liquefaction_analysis", _spt_params())
        assert out["min_fos"] == pytest.approx(direct["min_fos"], abs=1e-9)
        assert out["n_liquefiable"] == direct["n_liquefiable"]

    def test_cpt_routes_to_bi2014_with_indices(self):
        out = call_agent("liquefaction", "liquefaction_analysis", _cpt_params())
        assert out.get("method") == "bi2014"
        assert out.get("input_type") == "CPT"
        # CPT path carries the post-triggering indices.
        assert "lpi" in out and "lsn" in out and "ldi_m" in out

    def test_cpt_matches_underlying_liquepy(self):
        from liquepy_agent import analyze_cpt_liquefaction
        depth = np.arange(0.5, 15.0, 0.5)
        direct = analyze_cpt_liquefaction(
            depth=depth, q_c=np.ones_like(depth) * 5000,
            f_s=np.ones_like(depth) * 50, gwl=1.5, pga=0.3, m_w=7.5,
        ).to_dict()
        out = call_agent("liquefaction", "liquefaction_analysis", _cpt_params())
        assert out["lpi"] == pytest.approx(direct["lpi"], abs=1e-6)

    def test_bi2014_and_nceer_differ_but_close(self):
        # Same SPT input, two methods -> both valid, numerically distinct fits.
        bi = call_agent("liquefaction", "liquefaction_analysis", _spt_params("bi2014"))
        nc = call_agent("liquefaction", "liquefaction_analysis", _spt_params("nceer2001"))
        assert bi["method"] == "bi2014"
        assert nc["method"] == "nceer2001"
        # Both should flag a similar number of liquefiable layers here.
        assert bi["n_liquefiable"] == nc["n_liquefiable"]


# ============================================================================
# Error routing
# ============================================================================

class TestErrorRouting:
    def test_cpt_with_nceer_flag_errors(self):
        out = call_agent("liquefaction", "liquefaction_analysis", _cpt_params("nceer2001"))
        assert "error" in out
        assert "nceer2001" in out["error"]

    def test_ambiguous_input_errors(self):
        out = call_agent("liquefaction", "liquefaction_analysis", {
            "depth": [3], "N160": [10], "q_c": [5000], "f_s": [50],
            "FC": [5], "gamma": [18], "amax_g": 0.3, "gwt_depth": 1.0,
        })
        assert "error" in out

    def test_no_data_errors(self):
        out = call_agent("liquefaction", "liquefaction_analysis", {"depth": [3]})
        assert "error" in out


# ============================================================================
# Method-name alias routing (dispatch _METHOD_ALIASES)
# ============================================================================

class TestAliasRouting:
    def test_evaluate_liquefaction_alias(self):
        # Agent guesses 'evaluate_liquefaction' -> routes to the unified method.
        out = call_agent("liquefaction", "evaluate_liquefaction", _spt_params("nceer2001"))
        assert "error" not in out
        assert out.get("method") == "nceer2001"

    def test_nceer2001_name_alias_injects_method(self):
        # Agent guesses 'nceer2001' as a METHOD NAME -> injects method flag.
        params = _spt_params()  # no method key
        out = call_agent("liquefaction", "nceer2001", params)
        assert "error" not in out
        assert out.get("method") == "nceer2001"
