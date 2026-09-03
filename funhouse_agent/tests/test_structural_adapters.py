"""Tests for the structural round-1 adapters (section_props, concrete_props,
pynite) — METHOD_INFO/REGISTRY integrity + dispatch + real calls."""

import pytest

REQUIRED_INFO_FIELDS = ("category", "brief", "parameters", "returns")


# ===================================================================
# section_props adapter
# ===================================================================

class TestSectionPropsMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.section_props_adapter import (
            METHOD_INFO, METHOD_REGISTRY)
        assert set(METHOD_INFO.keys()) == set(METHOD_REGISTRY.keys())

    def test_required_fields(self):
        from funhouse_agent.adapters.section_props_adapter import METHOD_INFO
        for name, info in METHOD_INFO.items():
            for f in REQUIRED_INFO_FIELDS:
                assert f in info, f"{name} missing {f}"


class TestSectionPropsDispatch:
    def test_list_and_describe(self):
        from funhouse_agent.dispatch import list_methods, describe_method
        methods = list_methods("section_props")
        all_methods = [m for cat in methods.values() for m in cat]
        assert set(all_methods) == {"section_properties", "polygon_section"}
        info = describe_method("section_props", "section_properties")
        assert "shape" in info["parameters"]

    def test_call_rectangle(self):
        pytest.importorskip("sectionproperties")
        from funhouse_agent.dispatch import call_agent
        r = call_agent("section_props", "section_properties",
                       {"shape": "rectangle", "d": 200, "b": 100,
                        "warping": False})
        assert "error" not in r
        assert r["ixx_mm4"] == pytest.approx(100 * 200**3 / 12, rel=1e-6)

    def test_unknown_param_rejected(self):
        from funhouse_agent.dispatch import call_agent
        r = call_agent("section_props", "section_properties",
                       {"shape": "rectangle", "d": 200, "b": 100,
                        "bogus": 1})
        assert "error" in r and "bogus" in r["error"]


# ===================================================================
# concrete_props adapter
# ===================================================================

class TestConcretePropsMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.concrete_props_adapter import (
            METHOD_INFO, METHOD_REGISTRY)
        assert set(METHOD_INFO.keys()) == set(METHOD_REGISTRY.keys())

    def test_required_fields(self):
        from funhouse_agent.adapters.concrete_props_adapter import METHOD_INFO
        for name, info in METHOD_INFO.items():
            for f in REQUIRED_INFO_FIELDS:
                assert f in info, f"{name} missing {f}"


class TestConcretePropsDispatch:
    def test_list_and_describe(self):
        from funhouse_agent.dispatch import list_methods, describe_method
        methods = list_methods("concrete_props")
        all_methods = [m for cat in methods.values() for m in cat]
        assert all_methods == ["rc_rectangular_section"]
        info = describe_method("concrete_props", "rc_rectangular_section")
        assert "fc" in info["parameters"]

    def test_call_rc_section(self):
        pytest.importorskip("concreteproperties")
        from funhouse_agent.dispatch import call_agent
        r = call_agent("concrete_props", "rc_rectangular_section",
                       {"b": 300, "h": 550, "fc": 32, "fy": 500,
                        "n_bot": 3, "dia_bot": 28, "cover": 48})
        assert "error" not in r
        # DESIGN.md anchor: hand ACI calc 398.7 kN*m
        assert r["mn_pos_kNm"] == pytest.approx(398.7, rel=0.02)

    def test_missing_required(self):
        from funhouse_agent.dispatch import call_agent
        r = call_agent("concrete_props", "rc_rectangular_section",
                       {"b": 300, "h": 550})
        assert "error" in r


# ===================================================================
# pynite adapter
# ===================================================================

class TestPyniteMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.pynite_adapter import (
            METHOD_INFO, METHOD_REGISTRY)
        assert set(METHOD_INFO.keys()) == set(METHOD_REGISTRY.keys())

    def test_required_fields(self):
        from funhouse_agent.adapters.pynite_adapter import METHOD_INFO
        for name, info in METHOD_INFO.items():
            for f in REQUIRED_INFO_FIELDS:
                assert f in info, f"{name} missing {f}"


class TestPyniteDispatch:
    def test_list_and_describe(self):
        from funhouse_agent.dispatch import list_methods, describe_method
        methods = list_methods("pynite")
        all_methods = [m for cat in methods.values() for m in cat]
        assert set(all_methods) == {"frame_analysis", "continuous_beam"}
        info = describe_method("pynite", "continuous_beam")
        assert "span_lengths" in info["parameters"]

    def test_call_continuous_beam(self):
        pytest.importorskip("Pynite")
        from funhouse_agent.dispatch import call_agent
        r = call_agent("pynite", "continuous_beam",
                       {"span_lengths": [5, 5], "E": 200e6, "I": 1e-4,
                        "udl": 10})
        assert "error" not in r
        # textbook two-span: center reaction 1.25wL = 62.5 kN
        assert r["support_reactions_kN"][1] == pytest.approx(62.5, rel=1e-4)
        assert r["support_moments_kNm"][1] == pytest.approx(-31.25, rel=1e-4)

    def test_call_frame(self):
        pytest.importorskip("Pynite")
        from funhouse_agent.dispatch import call_agent
        r = call_agent("pynite", "frame_analysis", {
            "nodes": [{"name": "N1", "x": 0, "y": 0},
                      {"name": "N2", "x": 6, "y": 0}],
            "members": [{"name": "M1", "i": "N1", "j": "N2",
                         "E": 200e6, "A": 0.01, "Iz": 1e-4}],
            "supports": [{"node": "N1", "type": "pinned"},
                         {"node": "N2", "type": "roller_y"}],
            "member_dist_loads": [{"member": "M1", "direction": "FY",
                                   "w": -10}],
        })
        assert "error" not in r
        assert r["members"][0]["moment_abs_kNm"] == pytest.approx(45.0,
                                                                  rel=1e-5)

    def test_bad_support_type(self):
        pytest.importorskip("Pynite")
        from funhouse_agent.dispatch import call_agent
        r = call_agent("pynite", "frame_analysis", {
            "nodes": [{"name": "N1", "x": 0, "y": 0},
                      {"name": "N2", "x": 6, "y": 0}],
            "members": [{"name": "M1", "i": "N1", "j": "N2",
                         "E": 200e6, "A": 0.01, "Iz": 1e-4}],
            "supports": [{"node": "N1", "type": "clamped"}],
        })
        assert "error" in r and "clamped" in r["error"]


# ===================================================================
# registry integration
# ===================================================================

class TestRegistryIntegration:
    def test_modules_registered(self):
        from funhouse_agent.dispatch import ANALYSIS_MODULES
        for name in ("section_props", "concrete_props", "pynite"):
            assert name in ANALYSIS_MODULES

    def test_catalog_budget(self):
        import json
        from funhouse_agent.dispatch import list_agents
        cat = list_agents()
        s = cat if isinstance(cat, str) else json.dumps(cat)
        assert len(s) < 8000, f"list_agents catalog at {len(s)} chars"
