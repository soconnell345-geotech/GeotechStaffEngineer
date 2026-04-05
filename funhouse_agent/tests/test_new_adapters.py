"""Tests for Phase 1 + Phase 2 funhouse_agent adapters.

Covers 12 adapters:
- opensees, pystrata, liquepy, seismic_signals (has_* guards, raise ValueError)
- pyseismosoil, pystra, salib, pygef, ags4, pydiggs (has_* guards, return error dict)
- dxf_import, pdf_import (no guard, external deps mocked)

Each adapter has:
- TestXxxMethodInfo  — METHOD_INFO/REGISTRY key match, required fields
- TestXxxDispatch    — list_methods / describe_method via dispatch layer
- TestXxxCalls       — call_agent: mock has_* to False for "not installed" path,
                       and/or actual calls when the library IS available
"""

import pytest
from unittest.mock import patch, MagicMock


# ============================================================================
# Helper — common assertions
# ============================================================================

REQUIRED_INFO_FIELDS = {"category", "brief", "parameters", "returns"}


def assert_method_info_complete(method_info, method_registry):
    """METHOD_INFO and METHOD_REGISTRY keys match; required fields present."""
    assert set(method_info.keys()) == set(method_registry.keys())
    for name, info in method_info.items():
        for field in REQUIRED_INFO_FIELDS:
            assert field in info, f"{name} missing {field}"


# ============================================================================
# 1. opensees_adapter — 3 methods, has_opensees guard (raises ValueError)
# ============================================================================

class TestOpenseesMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.opensees_adapter import METHOD_INFO, METHOD_REGISTRY
        assert_method_info_complete(METHOD_INFO, METHOD_REGISTRY)

    def test_expected_methods(self):
        from funhouse_agent.adapters.opensees_adapter import METHOD_INFO
        assert set(METHOD_INFO.keys()) == {
            "pm4sand_cyclic_dss", "bnwf_lateral_pile", "site_response_1d",
        }


class TestOpenseesDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("opensees")
        total = sum(len(v) for v in result.values())
        assert total == 3

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("opensees", "pm4sand_cyclic_dss")
        assert "parameters" in info
        assert "Dr" in info["parameters"]


class TestOpenseesCalls:
    def test_pm4sand_not_installed(self):
        """Mock has_opensees to return False, verify error is returned."""
        with patch("opensees_agent.has_opensees", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("opensees", "pm4sand_cyclic_dss", {
                "Dr": 0.55, "G0": 600, "hpo": 0.4, "Den": 1.7,
            })
            assert "error" in result
            assert "not installed" in result["error"].lower() or "opensees" in result["error"].lower()

    def test_bnwf_not_installed(self):
        with patch("opensees_agent.has_opensees", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("opensees", "bnwf_lateral_pile", {
                "pile_length": 15, "pile_diameter": 0.5,
                "wall_thickness": 0.01, "E_pile": 200e6,
                "layers": [{"top": 0, "bottom": 15, "py_model": "SandAPI",
                            "phi": 35, "gamma": 18}],
            })
            assert "error" in result

    def test_site_response_not_installed(self):
        with patch("opensees_agent.has_opensees", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("opensees", "site_response_1d", {
                "layers": [{"thickness": 5, "Vs": 200, "density": 1.8,
                            "material_type": "sand", "phi": 35}],
            })
            assert "error" in result


# ============================================================================
# 2. pystrata_adapter — 2 methods, has_pystrata guard (raises ValueError)
# ============================================================================

class TestPystrataMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.pystrata_adapter import METHOD_INFO, METHOD_REGISTRY
        assert_method_info_complete(METHOD_INFO, METHOD_REGISTRY)

    def test_expected_methods(self):
        from funhouse_agent.adapters.pystrata_adapter import METHOD_INFO
        assert set(METHOD_INFO.keys()) == {
            "eql_site_response", "linear_site_response",
        }


class TestPystrataDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("pystrata")
        total = sum(len(v) for v in result.values())
        assert total == 2

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("pystrata", "eql_site_response")
        assert "parameters" in info
        assert "layers" in info["parameters"]


class TestPystrataCalls:
    def test_eql_not_installed(self):
        with patch("pystrata_agent.has_pystrata", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("pystrata", "eql_site_response", {
                "layers": [{"thickness": 10, "Vs": 200, "unit_wt": 18,
                            "soil_model": "darendeli"}],
            })
            assert "error" in result
            assert "not installed" in result["error"].lower()

    def test_linear_not_installed(self):
        with patch("pystrata_agent.has_pystrata", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("pystrata", "linear_site_response", {
                "layers": [{"thickness": 10, "Vs": 200, "unit_wt": 18,
                            "soil_model": "linear"}],
            })
            assert "error" in result


# ============================================================================
# 3. liquepy_adapter — 2 methods, has_liquepy guard (raises ValueError)
# ============================================================================

class TestLiquepyMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.liquepy_adapter import METHOD_INFO, METHOD_REGISTRY
        assert_method_info_complete(METHOD_INFO, METHOD_REGISTRY)

    def test_expected_methods(self):
        from funhouse_agent.adapters.liquepy_adapter import METHOD_INFO
        assert set(METHOD_INFO.keys()) == {
            "cpt_liquefaction", "field_correlations",
        }


class TestLiquepyDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("liquepy")
        total = sum(len(v) for v in result.values())
        assert total == 2

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("liquepy", "cpt_liquefaction")
        assert "parameters" in info
        assert "depth" in info["parameters"]
        assert "q_c" in info["parameters"]


class TestLiquepyCalls:
    def test_cpt_liquefaction_not_installed(self):
        """Mock has_liquepy to return False, verify ValueError-based error."""
        with patch("liquepy_agent.has_liquepy", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("liquepy", "cpt_liquefaction", {
                "depth": [1, 2, 3], "q_c": [5000, 6000, 7000],
                "f_s": [50, 60, 70],
            })
            assert "error" in result
            assert "not installed" in result["error"].lower()

    def test_field_correlations_not_installed(self):
        with patch("liquepy_agent.has_liquepy", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("liquepy", "field_correlations", {
                "depth": [1, 2, 3], "q_c": [5000, 6000, 7000],
                "f_s": [50, 60, 70],
            })
            assert "error" in result

    def test_cpt_liquefaction_actual(self):
        """Test actual call when liquepy IS available."""
        from funhouse_agent.dispatch import call_agent
        result = call_agent("liquepy", "cpt_liquefaction", {
            "depth": [1, 2, 3, 4, 5],
            "q_c": [5000, 6000, 7000, 8000, 9000],
            "f_s": [50, 60, 70, 80, 90],
            "gwl": 1.0, "pga": 0.25, "m_w": 7.5,
        })
        # If library is installed, we get results; if not, error
        if "error" not in result:
            assert "lpi" in result or "LPI" in result


# ============================================================================
# 4. seismic_signals_adapter — 4 methods, has_eqsig/has_pyrotd guards
# ============================================================================

class TestSeismicSignalsMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.seismic_signals_adapter import METHOD_INFO, METHOD_REGISTRY
        assert_method_info_complete(METHOD_INFO, METHOD_REGISTRY)

    def test_expected_methods(self):
        from funhouse_agent.adapters.seismic_signals_adapter import METHOD_INFO
        assert set(METHOD_INFO.keys()) == {
            "response_spectrum", "intensity_measures",
            "rotd_spectrum", "signal_processing",
        }


class TestSeismicSignalsDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("seismic_signals")
        total = sum(len(v) for v in result.values())
        assert total == 4

    def test_describe_method_response_spectrum(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("seismic_signals", "response_spectrum")
        assert "parameters" in info
        assert "damping" in info["parameters"]

    def test_describe_method_rotd(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("seismic_signals", "rotd_spectrum")
        assert "parameters" in info
        assert "percentiles" in info["parameters"]


class TestSeismicSignalsCalls:
    def test_response_spectrum_not_installed(self):
        with patch("seismic_signals_agent.has_eqsig", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("seismic_signals", "response_spectrum", {
                "accel_history": [0.1, 0.2, -0.1], "dt": 0.01,
            })
            assert "error" in result
            assert "not installed" in result["error"].lower()

    def test_intensity_measures_not_installed(self):
        with patch("seismic_signals_agent.has_eqsig", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("seismic_signals", "intensity_measures", {
                "accel_history": [0.1, 0.2, -0.1], "dt": 0.01,
            })
            assert "error" in result

    def test_rotd_spectrum_not_installed(self):
        with patch("seismic_signals_agent.has_pyrotd", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("seismic_signals", "rotd_spectrum", {
                "accel_history_a": [0.1, 0.2], "accel_history_b": [0.05, 0.1],
                "dt": 0.01,
            })
            assert "error" in result

    def test_signal_processing_not_installed(self):
        with patch("seismic_signals_agent.has_eqsig", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("seismic_signals", "signal_processing", {
                "accel_history": [0.1, 0.2, -0.1], "dt": 0.01,
            })
            assert "error" in result


# ============================================================================
# 5. pyseismosoil_adapter — 2 methods, has_pyseismosoil guard (returns error dict)
# ============================================================================

class TestPyseismosoilMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.pyseismosoil_adapter import METHOD_INFO, METHOD_REGISTRY
        assert_method_info_complete(METHOD_INFO, METHOD_REGISTRY)

    def test_expected_methods(self):
        from funhouse_agent.adapters.pyseismosoil_adapter import METHOD_INFO
        assert set(METHOD_INFO.keys()) == {
            "generate_curves", "analyze_vs_profile",
        }


class TestPyseismosoilDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("pyseismosoil")
        total = sum(len(v) for v in result.values())
        assert total == 2

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("pyseismosoil", "generate_curves")
        assert "parameters" in info
        assert "model" in info["parameters"]
        assert "params" in info["parameters"]


class TestPyseismosoilCalls:
    def test_generate_curves_not_installed(self):
        with patch("pyseismosoil_agent.has_pyseismosoil", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("pyseismosoil", "generate_curves", {
                "model": "MKZ",
                "params": {"gamma_ref": 0.05, "beta": 1.0, "s": 0.919, "Gmax": 50000},
            })
            assert "error" in result
            assert "not installed" in result["error"].lower()

    def test_analyze_vs_profile_not_installed(self):
        with patch("pyseismosoil_agent.has_pyseismosoil", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("pyseismosoil", "analyze_vs_profile", {
                "thicknesses": [5, 10, 0],
                "vs_values": [150, 300, 760],
            })
            assert "error" in result
            assert "not installed" in result["error"].lower()

    def test_generate_curves_actual(self):
        """Test actual call — succeeds if PySeismoSoil installed."""
        from funhouse_agent.dispatch import call_agent
        result = call_agent("pyseismosoil", "generate_curves", {
            "model": "MKZ",
            "params": {"gamma_ref": 0.05, "beta": 1.0, "s": 0.919, "Gmax": 50000},
        })
        if "error" not in result:
            assert result["model"] == "MKZ"
            assert result["n_points"] == 50
            assert len(result["G_Gmax"]) == 50

    def test_analyze_vs_profile_actual(self):
        """Test actual call — succeeds if PySeismoSoil installed."""
        from funhouse_agent.dispatch import call_agent
        result = call_agent("pyseismosoil", "analyze_vs_profile", {
            "thicknesses": [5, 10, 0],
            "vs_values": [150, 300, 760],
        })
        if "error" not in result:
            assert result["n_layers"] == 2
            assert result["vs30"] > 0


# ============================================================================
# 6. pystra_adapter — 3 methods, has_pystra guard (returns error dict)
# ============================================================================

class TestPystraMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.pystra_adapter import METHOD_INFO, METHOD_REGISTRY
        assert_method_info_complete(METHOD_INFO, METHOD_REGISTRY)

    def test_expected_methods(self):
        from funhouse_agent.adapters.pystra_adapter import METHOD_INFO
        assert set(METHOD_INFO.keys()) == {
            "form_analysis", "sorm_analysis", "monte_carlo_analysis",
        }


class TestPystraDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("pystra")
        total = sum(len(v) for v in result.values())
        assert total == 3

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("pystra", "form_analysis")
        assert "parameters" in info
        assert "variables" in info["parameters"]
        assert "limit_state" in info["parameters"]


class TestPystraCalls:
    _VARIABLES = [
        {"name": "R", "dist": "normal", "mean": 200, "stdv": 20},
        {"name": "S", "dist": "normal", "mean": 100, "stdv": 15},
    ]

    def test_form_not_installed(self):
        with patch("pystra_agent.has_pystra", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("pystra", "form_analysis", {
                "variables": self._VARIABLES, "limit_state": "R - S",
            })
            assert "error" in result
            assert "not installed" in result["error"].lower()

    def test_sorm_not_installed(self):
        with patch("pystra_agent.has_pystra", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("pystra", "sorm_analysis", {
                "variables": self._VARIABLES, "limit_state": "R - S",
            })
            assert "error" in result

    def test_monte_carlo_not_installed(self):
        with patch("pystra_agent.has_pystra", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("pystra", "monte_carlo_analysis", {
                "variables": self._VARIABLES, "limit_state": "R - S",
                "n_samples": 1000,
            })
            assert "error" in result

    def test_form_actual(self):
        """Test actual FORM call — succeeds if pystra installed."""
        from funhouse_agent.dispatch import call_agent
        result = call_agent("pystra", "form_analysis", {
            "variables": self._VARIABLES, "limit_state": "R - S",
        })
        if "error" not in result:
            assert result["beta"] > 0
            assert result["converged"] is True


# ============================================================================
# 7. salib_adapter — 4 methods, has_salib guard (returns error dict)
# ============================================================================

class TestSalibMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.salib_adapter import METHOD_INFO, METHOD_REGISTRY
        assert_method_info_complete(METHOD_INFO, METHOD_REGISTRY)

    def test_expected_methods(self):
        from funhouse_agent.adapters.salib_adapter import METHOD_INFO
        assert set(METHOD_INFO.keys()) == {
            "sobol_sample", "sobol_analyze",
            "morris_sample", "morris_analyze",
        }


class TestSalibDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("salib")
        total = sum(len(v) for v in result.values())
        assert total == 4

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("salib", "sobol_sample")
        assert "parameters" in info
        assert "var_names" in info["parameters"]
        assert "bounds" in info["parameters"]


class TestSalibCalls:
    def test_sobol_sample_not_installed(self):
        with patch("salib_agent.has_salib", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("salib", "sobol_sample", {
                "var_names": ["x1", "x2"],
                "bounds": [[0, 1], [0, 1]],
            })
            assert "error" in result
            assert "not installed" in result["error"].lower()

    def test_sobol_analyze_not_installed(self):
        with patch("salib_agent.has_salib", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("salib", "sobol_analyze", {
                "var_names": ["x1", "x2"],
                "bounds": [[0, 1], [0, 1]],
                "Y": [1.0] * 12,
            })
            assert "error" in result

    def test_morris_sample_not_installed(self):
        with patch("salib_agent.has_salib", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("salib", "morris_sample", {
                "var_names": ["x1", "x2"],
                "bounds": [[0, 1], [0, 1]],
            })
            assert "error" in result

    def test_morris_analyze_not_installed(self):
        with patch("salib_agent.has_salib", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("salib", "morris_analyze", {
                "var_names": ["x1", "x2"],
                "bounds": [[0, 1], [0, 1]],
                "X": [[0, 0], [1, 0], [1, 1]],
                "Y": [0.1, 0.2, 0.3],
            })
            assert "error" in result

    def test_sobol_sample_actual(self):
        """Test actual Sobol sampling — succeeds if SALib installed."""
        from funhouse_agent.dispatch import call_agent
        result = call_agent("salib", "sobol_sample", {
            "var_names": ["x1", "x2"],
            "bounds": [[0, 1], [0, 1]],
            "n_samples": 64,
        })
        if "error" not in result:
            assert result["n_vars"] == 2
            assert result["n_rows"] > 0


# ============================================================================
# 8. pygef_adapter — 2 methods, has_pygef guard (returns error dict)
# ============================================================================

class TestPygefMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.pygef_adapter import METHOD_INFO, METHOD_REGISTRY
        assert_method_info_complete(METHOD_INFO, METHOD_REGISTRY)

    def test_expected_methods(self):
        from funhouse_agent.adapters.pygef_adapter import METHOD_INFO
        assert set(METHOD_INFO.keys()) == {"parse_cpt", "parse_bore"}


class TestPygefDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("pygef")
        total = sum(len(v) for v in result.values())
        assert total == 2

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("pygef", "parse_cpt")
        assert "parameters" in info
        assert "file_path" in info["parameters"]


class TestPygefCalls:
    def test_parse_cpt_not_installed(self):
        with patch("pygef_agent.has_pygef", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("pygef", "parse_cpt", {
                "file_path": "test.gef",
            })
            assert "error" in result
            assert "not installed" in result["error"].lower()

    def test_parse_bore_not_installed(self):
        with patch("pygef_agent.has_pygef", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("pygef", "parse_bore", {
                "file_path": "test.gef",
            })
            assert "error" in result
            assert "not installed" in result["error"].lower()


# ============================================================================
# 9. ags4_adapter — 2 methods, has_ags4 guard (returns error dict)
# ============================================================================

class TestAgs4MethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.ags4_adapter import METHOD_INFO, METHOD_REGISTRY
        assert_method_info_complete(METHOD_INFO, METHOD_REGISTRY)

    def test_expected_methods(self):
        from funhouse_agent.adapters.ags4_adapter import METHOD_INFO
        assert set(METHOD_INFO.keys()) == {"read_ags4", "validate_ags4"}


class TestAgs4Dispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("ags4")
        total = sum(len(v) for v in result.values())
        assert total == 2

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("ags4", "read_ags4")
        assert "parameters" in info
        assert "file_path" in info["parameters"]


class TestAgs4Calls:
    def test_read_ags4_not_installed(self):
        with patch("ags4_agent.has_ags4", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("ags4", "read_ags4", {
                "file_path": "test.ags",
            })
            assert "error" in result
            assert "not installed" in result["error"].lower()

    def test_validate_ags4_not_installed(self):
        with patch("ags4_agent.has_ags4", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("ags4", "validate_ags4", {
                "file_path": "test.ags",
            })
            assert "error" in result
            assert "not installed" in result["error"].lower()


# ============================================================================
# 10. pydiggs_adapter — 2 methods, has_pydiggs guard (returns error dict)
# ============================================================================

class TestPydiggsMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.pydiggs_adapter import METHOD_INFO, METHOD_REGISTRY
        assert_method_info_complete(METHOD_INFO, METHOD_REGISTRY)

    def test_expected_methods(self):
        from funhouse_agent.adapters.pydiggs_adapter import METHOD_INFO
        assert set(METHOD_INFO.keys()) == {
            "validate_diggs_schema", "validate_diggs_dictionary",
        }


class TestPydiggsDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("pydiggs")
        total = sum(len(v) for v in result.values())
        assert total == 2

    def test_describe_method(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("pydiggs", "validate_diggs_schema")
        assert "parameters" in info
        assert "file_path" in info["parameters"]


class TestPydiggsCalls:
    def test_validate_schema_not_installed(self):
        with patch("pydiggs_agent.has_pydiggs", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("pydiggs", "validate_diggs_schema", {
                "file_path": "test.xml",
            })
            assert "error" in result
            assert "not installed" in result["error"].lower()

    def test_validate_dictionary_not_installed(self):
        with patch("pydiggs_agent.has_pydiggs", return_value=False):
            from funhouse_agent.dispatch import call_agent
            result = call_agent("pydiggs", "validate_diggs_dictionary", {
                "file_path": "test.xml",
            })
            assert "error" in result
            assert "not installed" in result["error"].lower()


# ============================================================================
# 11. dxf_import_adapter — 4 methods (no external dep guard, uses ezdxf)
# ============================================================================

class TestDxfImportMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.dxf_import_adapter import METHOD_INFO, METHOD_REGISTRY
        assert_method_info_complete(METHOD_INFO, METHOD_REGISTRY)

    def test_expected_methods(self):
        from funhouse_agent.adapters.dxf_import_adapter import METHOD_INFO
        assert set(METHOD_INFO.keys()) == {
            "discover_layers", "parse_geometry",
            "build_slope_geometry", "build_fem_inputs",
        }


class TestDxfImportDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("dxf_import")
        total = sum(len(v) for v in result.values())
        assert total == 4

    def test_describe_method_discover(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("dxf_import", "discover_layers")
        assert "parameters" in info
        assert "file_path" in info["parameters"]

    def test_describe_method_parse(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("dxf_import", "parse_geometry")
        assert "parameters" in info
        assert "layer_mapping" in info["parameters"]


class TestDxfImportCalls:
    def test_discover_layers_mocked(self):
        """Mock dxf_import.discover_layers to test adapter wiring."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "n_layers": 3,
            "n_total_entities": 42,
            "units_hint": "Meters",
            "layers": [
                {"name": "SURFACE", "n_entities": 10, "entity_types": ["LINE", "POLYLINE"]},
            ],
        }
        with patch("dxf_import.discover_layers", return_value=mock_result) as mock_fn:
            from funhouse_agent.dispatch import call_agent
            result = call_agent("dxf_import", "discover_layers", {
                "file_path": "test.dxf",
            })
            mock_fn.assert_called_once_with(filepath="test.dxf")
            assert result["n_layers"] == 3
            assert result["n_total_entities"] == 42

    def test_parse_geometry_mocked(self):
        """Mock dxf_import.parse_geometry to test adapter wiring."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "surface_points": [{"x": 0, "z": 10}, {"x": 20, "z": 5}],
            "boundary_profiles": {},
            "gwt_points": None,
            "nail_lines": [],
            "warnings": [],
        }
        with patch("dxf_import.parse_dxf_geometry", return_value=mock_result):
            with patch("dxf_import.LayerMapping") as mock_lm:
                mock_lm.return_value = MagicMock()
                from funhouse_agent.dispatch import call_agent
                result = call_agent("dxf_import", "parse_geometry", {
                    "file_path": "test.dxf",
                    "layer_mapping": {"surface": "SURFACE", "soil_boundaries": {}},
                })
                assert len(result["surface_points"]) == 2

    def test_build_slope_geometry_mocked(self):
        """Mock dxf_import.build_slope_geometry to test adapter wiring."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "surface_points": [[0, 10], [20, 5]],
            "soil_layers": [{"name": "Clay", "gamma": 18}],
            "gwt_points": None,
        }
        with patch("dxf_import.build_slope_geometry", return_value=mock_result):
            with patch("dxf_import.SoilPropertyAssignment") as mock_spa:
                mock_spa.return_value = MagicMock()
                from funhouse_agent.dispatch import call_agent
                result = call_agent("dxf_import", "build_slope_geometry", {
                    "parse_result": {
                        "surface_points": [[0, 10], [20, 5]],
                        "boundary_profiles": {"Clay": [[0, 5], [20, 0]]},
                    },
                    "soil_properties": [
                        {"name": "Clay", "gamma": 18, "phi": 25, "c_prime": 10},
                    ],
                })
                assert result["soil_layers"][0]["name"] == "Clay"

    def test_build_fem_inputs_mocked(self):
        """Mock dxf_import.build_fem_inputs to test adapter wiring."""
        mock_result = {
            "surface_points": [[0, 10], [20, 5]],
            "soil_layers": [{"name": "Sand", "E": 30000}],
            "gwt": None,
            "boundary_polylines": {},
        }
        with patch("dxf_import.build_fem_inputs", return_value=mock_result):
            with patch("dxf_import.FEMSoilPropertyAssignment") as mock_fspa:
                mock_fspa.return_value = MagicMock()
                from funhouse_agent.dispatch import call_agent
                result = call_agent("dxf_import", "build_fem_inputs", {
                    "parse_result": {
                        "surface_points": [[0, 10], [20, 5]],
                        "boundary_profiles": {},
                    },
                    "soil_properties": [
                        {"name": "Sand", "gamma": 18, "phi": 35, "c": 0, "E": 30000},
                    ],
                })
                assert result["soil_layers"][0]["name"] == "Sand"


# ============================================================================
# 12. pdf_import_adapter — 2 methods (no external dep guard, uses PyMuPDF)
# ============================================================================

class TestPdfImportMethodInfo:
    def test_keys_match(self):
        from funhouse_agent.adapters.pdf_import_adapter import METHOD_INFO, METHOD_REGISTRY
        assert_method_info_complete(METHOD_INFO, METHOD_REGISTRY)

    def test_expected_methods(self):
        from funhouse_agent.adapters.pdf_import_adapter import METHOD_INFO
        assert set(METHOD_INFO.keys()) == {
            "discover_pdf_content", "extract_vector_geometry",
            "build_slope_geometry", "build_fem_inputs",
        }


class TestPdfImportDispatch:
    def test_list_methods(self):
        from funhouse_agent.dispatch import list_methods
        result = list_methods("pdf_import")
        total = sum(len(v) for v in result.values())
        assert total == 4

    def test_describe_method_discover(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("pdf_import", "discover_pdf_content")
        assert "parameters" in info
        assert "file_path" in info["parameters"]

    def test_describe_method_extract(self):
        from funhouse_agent.dispatch import describe_method
        info = describe_method("pdf_import", "extract_vector_geometry")
        assert "parameters" in info
        assert "scale" in info["parameters"]


class TestPdfImportCalls:
    def test_discover_pdf_content_mocked(self):
        """Mock pdf_import.discover_pdf_content to test adapter wiring."""
        mock_result = {
            "page_size": {"width": 612, "height": 792},
            "n_drawings": 15,
            "colors": {"#000000": 10, "#0000ff": 5},
            "text_blocks": [{"text": "Section A-A", "x": 100, "y": 700, "size": 12}],
            "has_images": False,
        }
        with patch("pdf_import.discover_pdf_content", return_value=mock_result) as mock_fn:
            from funhouse_agent.dispatch import call_agent
            result = call_agent("pdf_import", "discover_pdf_content", {
                "file_path": "section.pdf",
                "page": 0,
            })
            mock_fn.assert_called_once_with(filepath="section.pdf", page=0)
            assert result["n_drawings"] == 15
            assert result["page_size"]["width"] == 612

    def test_extract_vector_geometry_mocked(self):
        """Mock pdf_import.extract_vector_geometry to test adapter wiring."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "surface_points": [{"x": 0, "z": 10}, {"x": 30, "z": 5}],
            "boundary_profiles": {},
            "gwt_points": None,
            "text_annotations": [],
            "extraction_method": "vector",
            "confidence": 1.0,
        }
        with patch("pdf_import.extract_vector_geometry", return_value=mock_result) as mock_fn:
            from funhouse_agent.dispatch import call_agent
            result = call_agent("pdf_import", "extract_vector_geometry", {
                "file_path": "section.pdf",
                "page": 0,
                "scale": 0.001,
                "origin": "bottom_left",
            })
            mock_fn.assert_called_once_with(
                filepath="section.pdf", page=0, scale=0.001,
                origin="bottom_left", role_mapping=None,
            )
            assert result["extraction_method"] == "vector"
            assert result["confidence"] == 1.0
            assert len(result["surface_points"]) == 2


# ============================================================================
# Cross-cutting tests
# ============================================================================

class TestAllAdaptersRegistered:
    """Verify all 12 adapters are reachable via dispatch."""

    @pytest.mark.parametrize("agent_name", [
        "opensees", "pystrata", "liquepy", "seismic_signals",
        "pyseismosoil", "pystra", "salib", "pygef",
        "ags4", "pydiggs", "dxf_import", "pdf_import",
    ])
    def test_agent_in_registry(self, agent_name):
        from funhouse_agent.dispatch import list_methods
        result = list_methods(agent_name)
        assert "error" not in result, f"{agent_name} not in registry"
        total = sum(len(v) for v in result.values())
        assert total > 0

    @pytest.mark.parametrize("agent_name,method_name", [
        ("opensees", "unknown_method"),
        ("pystrata", "nonexistent"),
        ("dxf_import", "fake_method"),
    ])
    def test_unknown_method_returns_error(self, agent_name, method_name):
        from funhouse_agent.dispatch import call_agent
        result = call_agent(agent_name, method_name, {})
        assert "error" in result
        assert "Unknown method" in result["error"]

    def test_unknown_agent_returns_error(self):
        from funhouse_agent.dispatch import call_agent
        result = call_agent("nonexistent_agent", "foo", {})
        assert "error" in result
        assert "Unknown module" in result["error"]
