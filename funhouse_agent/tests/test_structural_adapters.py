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
        pytest.importorskip("Pynite")
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


class TestPyniteOptional:
    """PyNiteFEA cannot install where numpy is held at 1.26 (Foundry's
    transforms repos, INF-53): the module is then left out of the registry,
    so no agent is shown a tool that cannot run."""

    def test_listed_exactly_when_installed(self):
        import importlib.util
        from funhouse_agent.adapters import ABSENT_MODULES, MODULE_REGISTRY
        from funhouse_agent.dispatch import ANALYSIS_MODULES, list_agents
        # The same check the registry makes: present, not imported.
        installed = importlib.util.find_spec("Pynite") is not None
        assert ("pynite" in MODULE_REGISTRY) == installed
        assert ("pynite" in ANALYSIS_MODULES) == installed
        assert ("pynite" in list_agents()) == installed
        assert ABSENT_MODULES == ({} if installed
                                  else {"pynite": "PyNiteFEA"})

    def test_a_module_left_out_is_unknown(self, monkeypatch):
        from funhouse_agent import dispatch
        monkeypatch.delitem(dispatch.MODULE_REGISTRY, "pynite", raising=False)
        r = dispatch.call_agent("pynite", "continuous_beam",
                                {"span_lengths": [5, 5], "E": 200e6,
                                 "I": 1e-4, "udl": 10})
        assert r["error"].startswith("Unknown module 'pynite'")
        assert "pynite" not in dispatch.list_agents()

    def test_a_broken_pynite_fails_only_its_own_calls(self, tmp_path):
        """A Pynite that is installed but raises on import (say an
        AttributeError against an older numpy): the registry still loads,
        pynite's call returns an error result, and the other tools run."""
        import json
        import os
        import subprocess
        import sys
        import textwrap
        fake = tmp_path / "Pynite"
        fake.mkdir()
        (fake / "__init__.py").write_text(
            "raise AttributeError(\"module 'numpy' has no attribute "
            "'broken'\")\n", encoding="utf-8")
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        script = textwrap.dedent(f"""
            import json, sys
            sys.path[:0] = [{str(tmp_path)!r}, {root!r}]
            import funhouse_agent.adapters as adapters
            from funhouse_agent.dispatch import call_agent
            beam = call_agent("pynite", "continuous_beam",
                              {{"span_lengths": [5, 5], "E": 200e6,
                               "I": 1e-4, "udl": 10}})
            other = call_agent("bearing_capacity",
                               "bearing_capacity_analysis",
                               {{"width": 2, "depth": 1, "unit_weight": 18,
                                "friction_angle": 30, "cohesion": 0}})
            print(json.dumps({{"absent": adapters.ABSENT_MODULES,
                              "listed": "pynite" in adapters.MODULE_REGISTRY,
                              "beam": beam, "other": other}}, default=str))
        """)
        run = subprocess.run([sys.executable, "-c", script],
                             capture_output=True, text=True, timeout=300,
                             cwd=root)
        assert run.returncode == 0, run.stderr[-2000:]
        out = json.loads(run.stdout.strip().splitlines()[-1])
        assert out["absent"] == {} and out["listed"]
        assert out["beam"]["error"].startswith("AttributeError"), out["beam"]
        assert "error" not in out["other"], out["other"]
        assert out["other"]["q_ultimate_kPa"] > 0


# ===================================================================
# registry integration
# ===================================================================

class TestRegistryIntegration:
    def test_modules_registered(self):
        from funhouse_agent.adapters import ABSENT_MODULES
        from funhouse_agent.dispatch import ANALYSIS_MODULES
        for name in ("section_props", "concrete_props", "pynite"):
            assert name in ANALYSIS_MODULES or name in ABSENT_MODULES

    def test_catalog_budget(self):
        import json
        from funhouse_agent.dispatch import list_agents
        cat = list_agents()
        s = cat if isinstance(cat, str) else json.dumps(cat)
        assert len(s) < 8000, f"list_agents catalog at {len(s)} chars"
