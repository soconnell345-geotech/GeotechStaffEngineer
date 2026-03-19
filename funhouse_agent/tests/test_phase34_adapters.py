"""Tests for Phase 3 + Phase 4 funhouse_agent adapters.

Covers:
- fem2d_adapter (7 methods, no external dep)
- fdm2d_adapter (2 methods, no external dep)
- gstools_adapter (3 methods, has_gstools guard)
- hvsrpy_adapter (1 method, has_hvsrpy guard)
- swprocess_adapter (1 method, has_swprocess guard)
- subsurface_adapter (5 methods, no external dep)

Each adapter gets:
- TestXxxMethodInfo: METHOD_INFO/REGISTRY key match, required fields
- TestXxxDispatch: list_methods / describe_method via dispatch
- TestXxxCalls: real calls (core modules) or error checks (optional deps)
"""

import pytest
import matplotlib
matplotlib.use("Agg")


# ===================================================================
# FEM2D adapter
# ===================================================================

class TestFem2dMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.fem2d_adapter import METHOD_INFO, METHOD_REGISTRY
        assert set(METHOD_INFO.keys()) == set(METHOD_REGISTRY.keys())

    def test_required_fields(self):
        from funhouse_agent.adapters.fem2d_adapter import METHOD_INFO
        for name, info in METHOD_INFO.items():
            assert "category" in info, f"{name} missing category"
            assert "brief" in info, f"{name} missing brief"
            assert "parameters" in info, f"{name} missing parameters"
            assert "returns" in info, f"{name} missing returns"

    def test_method_count(self):
        from funhouse_agent.adapters.fem2d_adapter import METHOD_INFO
        assert len(METHOD_INFO) == 7


class TestFem2dDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        methods = list_methods("fem2d")
        assert "error" not in methods
        all_methods = []
        for cat_methods in methods.values():
            all_methods.extend(cat_methods.keys())
        assert "fem2d_gravity" in all_methods
        assert "fem2d_foundation" in all_methods
        assert "fem2d_slope_srm" in all_methods
        assert "fem2d_excavation" in all_methods
        assert "fem2d_seepage" in all_methods
        assert "fem2d_consolidation" in all_methods
        assert "fem2d_staged" in all_methods
        assert len(all_methods) == 7

    def test_describe_method_gravity(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("fem2d", "fem2d_gravity")
        assert "parameters" in info
        assert "width" in info["parameters"]
        assert "depth" in info["parameters"]
        assert "gamma" in info["parameters"]
        assert "E" in info["parameters"]
        assert "nu" in info["parameters"]

    def test_describe_method_foundation(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("fem2d", "fem2d_foundation")
        assert "parameters" in info
        assert "B" in info["parameters"]
        assert "q" in info["parameters"]

    def test_describe_unknown_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("fem2d", "nonexistent")
        assert "error" in info


class TestFem2dCalls:
    def test_gravity(self):
        from funhouse_agent.dispatch import call_agent
        result = call_agent("fem2d", "fem2d_gravity", {
            "width": 10, "depth": 5, "gamma": 18,
            "E": 10000, "nu": 0.3, "nx": 8, "ny": 4,
        })
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "max_displacement_m" in result
        assert result["max_displacement_m"] > 0
        assert "max_sigma_yy_kPa" in result

    def test_foundation(self):
        from funhouse_agent.dispatch import call_agent
        result = call_agent("fem2d", "fem2d_foundation", {
            "B": 2.0, "q": 100, "depth": 5,
            "E": 20000, "nu": 0.3, "nx": 10, "ny": 5,
        })
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "max_displacement_m" in result
        assert result["max_displacement_m"] >= 0
        assert "max_sigma_yy_kPa" in result


# ===================================================================
# FDM2D adapter
# ===================================================================

class TestFdm2dMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.fdm2d_adapter import METHOD_INFO, METHOD_REGISTRY
        assert set(METHOD_INFO.keys()) == set(METHOD_REGISTRY.keys())

    def test_required_fields(self):
        from funhouse_agent.adapters.fdm2d_adapter import METHOD_INFO
        for name, info in METHOD_INFO.items():
            assert "category" in info, f"{name} missing category"
            assert "brief" in info, f"{name} missing brief"
            assert "parameters" in info, f"{name} missing parameters"
            assert "returns" in info, f"{name} missing returns"

    def test_method_count(self):
        from funhouse_agent.adapters.fdm2d_adapter import METHOD_INFO
        assert len(METHOD_INFO) == 2


class TestFdm2dDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        methods = list_methods("fdm2d")
        assert "error" not in methods
        all_methods = []
        for cat_methods in methods.values():
            all_methods.extend(cat_methods.keys())
        assert "fdm2d_gravity" in all_methods
        assert "fdm2d_foundation" in all_methods
        assert len(all_methods) == 2

    def test_describe_method_gravity(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("fdm2d", "fdm2d_gravity")
        assert "parameters" in info
        assert "width" in info["parameters"]
        assert "E" in info["parameters"]
        assert "damping" in info["parameters"]

    def test_describe_method_foundation(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("fdm2d", "fdm2d_foundation")
        assert "parameters" in info
        assert "B" in info["parameters"]
        assert "q" in info["parameters"]


class TestFdm2dCalls:
    def test_gravity(self):
        from funhouse_agent.dispatch import call_agent
        result = call_agent("fdm2d", "fdm2d_gravity", {
            "width": 10, "depth": 5, "gamma": 18,
            "E": 10000, "nu": 0.3, "nx": 8, "ny": 4,
        })
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "max_displacement_m" in result
        assert result["max_displacement_m"] > 0
        assert "converged" in result

    def test_foundation(self):
        from funhouse_agent.dispatch import call_agent
        result = call_agent("fdm2d", "fdm2d_foundation", {
            "B": 2.0, "q": 100, "depth": 5,
            "E": 20000, "nu": 0.3, "nx": 10, "ny": 5,
        })
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "max_displacement_m" in result
        assert result["max_displacement_m"] > 0
        assert "converged" in result


# ===================================================================
# GSTools adapter
# ===================================================================

class TestGstoolsMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.gstools_adapter import METHOD_INFO, METHOD_REGISTRY
        assert set(METHOD_INFO.keys()) == set(METHOD_REGISTRY.keys())

    def test_required_fields(self):
        from funhouse_agent.adapters.gstools_adapter import METHOD_INFO
        for name, info in METHOD_INFO.items():
            assert "category" in info, f"{name} missing category"
            assert "brief" in info, f"{name} missing brief"
            assert "parameters" in info, f"{name} missing parameters"
            assert "returns" in info, f"{name} missing returns"

    def test_method_count(self):
        from funhouse_agent.adapters.gstools_adapter import METHOD_INFO
        assert len(METHOD_INFO) == 3


class TestGstoolsDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        methods = list_methods("gstools")
        assert "error" not in methods
        all_methods = []
        for cat_methods in methods.values():
            all_methods.extend(cat_methods.keys())
        assert "kriging" in all_methods
        assert "variogram" in all_methods
        assert "random_field" in all_methods
        assert len(all_methods) == 3

    def test_describe_method_kriging(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("gstools", "kriging")
        assert "parameters" in info
        assert "x" in info["parameters"]
        assert "y" in info["parameters"]
        assert "values" in info["parameters"]


class TestGstoolsCalls:
    def test_kriging_missing_or_unavailable(self):
        """call_agent with empty params returns error (dep missing or param error)."""
        from funhouse_agent.dispatch import call_agent
        result = call_agent("gstools", "kriging", {})
        assert "error" in result

    def test_variogram_missing_or_unavailable(self):
        from funhouse_agent.dispatch import call_agent
        result = call_agent("gstools", "variogram", {})
        assert "error" in result

    def test_random_field_missing_or_unavailable(self):
        from funhouse_agent.dispatch import call_agent
        result = call_agent("gstools", "random_field", {})
        # random_field has all defaults, so if gstools is not installed -> error
        # if gstools IS installed -> success (all params optional)
        # Either way, no crash
        assert isinstance(result, dict)


# ===================================================================
# HVSRPY adapter
# ===================================================================

class TestHvsrpyMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.hvsrpy_adapter import METHOD_INFO, METHOD_REGISTRY
        assert set(METHOD_INFO.keys()) == set(METHOD_REGISTRY.keys())

    def test_required_fields(self):
        from funhouse_agent.adapters.hvsrpy_adapter import METHOD_INFO
        for name, info in METHOD_INFO.items():
            assert "category" in info, f"{name} missing category"
            assert "brief" in info, f"{name} missing brief"
            assert "parameters" in info, f"{name} missing parameters"
            assert "returns" in info, f"{name} missing returns"

    def test_method_count(self):
        from funhouse_agent.adapters.hvsrpy_adapter import METHOD_INFO
        assert len(METHOD_INFO) == 1


class TestHvsrpyDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        methods = list_methods("hvsrpy")
        assert "error" not in methods
        all_methods = []
        for cat_methods in methods.values():
            all_methods.extend(cat_methods.keys())
        assert "hvsr_analysis" in all_methods
        assert len(all_methods) == 1

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("hvsrpy", "hvsr_analysis")
        assert "parameters" in info
        assert "ns" in info["parameters"]
        assert "ew" in info["parameters"]
        assert "vt" in info["parameters"]
        assert "dt" in info["parameters"]


class TestHvsrpyCalls:
    def test_hvsr_missing_or_unavailable(self):
        """call_agent with empty params returns error (dep missing or param error)."""
        from funhouse_agent.dispatch import call_agent
        result = call_agent("hvsrpy", "hvsr_analysis", {})
        assert "error" in result


# ===================================================================
# swprocess adapter
# ===================================================================

class TestSwprocessMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.swprocess_adapter import METHOD_INFO, METHOD_REGISTRY
        assert set(METHOD_INFO.keys()) == set(METHOD_REGISTRY.keys())

    def test_required_fields(self):
        from funhouse_agent.adapters.swprocess_adapter import METHOD_INFO
        for name, info in METHOD_INFO.items():
            assert "category" in info, f"{name} missing category"
            assert "brief" in info, f"{name} missing brief"
            assert "parameters" in info, f"{name} missing parameters"
            assert "returns" in info, f"{name} missing returns"

    def test_method_count(self):
        from funhouse_agent.adapters.swprocess_adapter import METHOD_INFO
        assert len(METHOD_INFO) == 1


class TestSwprocessDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        methods = list_methods("swprocess")
        assert "error" not in methods
        all_methods = []
        for cat_methods in methods.values():
            all_methods.extend(cat_methods.keys())
        assert "masw_dispersion" in all_methods
        assert len(all_methods) == 1

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("swprocess", "masw_dispersion")
        assert "parameters" in info
        assert "traces" in info["parameters"]
        assert "offsets" in info["parameters"]
        assert "dt" in info["parameters"]


class TestSwprocessCalls:
    def test_masw_missing_or_unavailable(self):
        """call_agent with empty params returns error (dep missing or param error)."""
        from funhouse_agent.dispatch import call_agent
        result = call_agent("swprocess", "masw_dispersion", {})
        assert "error" in result


# ===================================================================
# Subsurface adapter
# ===================================================================

class TestSubsurfaceMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.subsurface_adapter import METHOD_INFO, METHOD_REGISTRY
        assert set(METHOD_INFO.keys()) == set(METHOD_REGISTRY.keys())

    def test_required_fields(self):
        from funhouse_agent.adapters.subsurface_adapter import METHOD_INFO
        for name, info in METHOD_INFO.items():
            assert "category" in info, f"{name} missing category"
            assert "brief" in info, f"{name} missing brief"
            assert "parameters" in info, f"{name} missing parameters"
            assert "returns" in info, f"{name} missing returns"

    def test_method_count(self):
        from funhouse_agent.adapters.subsurface_adapter import METHOD_INFO
        assert len(METHOD_INFO) == 8


class TestSubsurfaceDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        methods = list_methods("subsurface")
        assert "error" not in methods
        all_methods = []
        for cat_methods in methods.values():
            all_methods.extend(cat_methods.keys())
        assert "load_site" in all_methods
        assert "parse_diggs" in all_methods
        assert "plot_parameter_vs_depth" in all_methods
        assert "plot_atterberg_limits" in all_methods
        assert "plot_multi_parameter" in all_methods
        assert "plot_plan_view" in all_methods
        assert "plot_cross_section" in all_methods
        assert "compute_trend" in all_methods
        assert len(all_methods) == 8

    def test_describe_method_load_site(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("subsurface", "load_site")
        assert "parameters" in info
        assert "site_data" in info["parameters"]

    def test_describe_method_compute_trend(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("subsurface", "compute_trend")
        assert "parameters" in info
        assert "depths" in info["parameters"]
        assert "values" in info["parameters"]


class TestSubsurfaceCalls:
    SITE_DATA = {
        "project_name": "Test",
        "investigations": [{
            "investigation_id": "BH-1",
            "x": 0,
            "y": 0,
            "measurements": [
                {"depth": 1, "parameter": "N_spt", "value": 10},
            ],
        }],
    }

    def test_load_site(self):
        from funhouse_agent.dispatch import call_agent
        result = call_agent("subsurface", "load_site", {
            "site_data": self.SITE_DATA,
        })
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert result["project_name"] == "Test"
        assert result["n_investigations"] == 1
        assert len(result["investigations"]) == 1
        assert result["investigations"][0]["investigation_id"] == "BH-1"
        assert result["investigations"][0]["n_measurements"] == 1

    def test_compute_trend(self):
        from funhouse_agent.dispatch import call_agent
        result = call_agent("subsurface", "compute_trend", {
            "depths": [1, 2, 3, 4, 5],
            "values": [10, 12, 15, 18, 20],
            "parameter": "N_spt",
        })
        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "slope" in result
        assert "intercept" in result
        assert "r_squared" in result
        assert result["r_squared"] > 0.9  # strong linear trend
        assert result["slope"] > 0  # positive slope (increasing with depth)
