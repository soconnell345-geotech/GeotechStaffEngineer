"""Tests for the pystrata_agent module.

Tier 1: No pystrata required (result dataclass, validation, Foundry metadata)
Tier 2: Requires pystrata (actual analyses, skip if not installed)
"""

import json

import numpy as np
import pytest

from pystrata_agent.pystrata_utils import has_pystrata
from pystrata_agent.results import EQLSiteResponseResult
from pystrata_agent.eql_site_response import _validate_eql_inputs


# Skip marker for Tier 2 tests
requires_pystrata = pytest.mark.skipif(
    not has_pystrata(), reason="pystrata not installed")


# ---------------------------------------------------------------------------
# Helpers: standard layer definitions for testing
# ---------------------------------------------------------------------------

def _darendeli_layer(thickness=10.0, Vs=250, unit_wt=18.0, plas_index=30):
    return {
        "thickness": thickness, "Vs": Vs, "unit_wt": unit_wt,
        "soil_model": "darendeli", "plas_index": plas_index,
    }


def _menq_layer(thickness=8.0, Vs=350, unit_wt=20.0):
    return {
        "thickness": thickness, "Vs": Vs, "unit_wt": unit_wt,
        "soil_model": "menq",
    }


def _linear_layer(thickness=5.0, Vs=400, unit_wt=20.0, damping=0.02):
    return {
        "thickness": thickness, "Vs": Vs, "unit_wt": unit_wt,
        "soil_model": "linear", "damping": damping,
    }


def _custom_layer(thickness=12.0, Vs=180, unit_wt=17.0):
    return {
        "thickness": thickness, "Vs": Vs, "unit_wt": unit_wt,
        "soil_model": "custom",
        "strains": [0.0001, 0.001, 0.01, 0.1],
        "mod_reduc": [1.0, 0.92, 0.55, 0.10],
        "damping_values": [0.01, 0.025, 0.07, 0.20],
    }


def _bedrock_layer(Vs=760, unit_wt=24.0, damping=0.01):
    return {
        "thickness": 0, "Vs": Vs, "unit_wt": unit_wt,
        "soil_model": "linear", "damping": damping,
    }


def _simple_profile():
    """Single Darendeli layer + bedrock."""
    return [_darendeli_layer(), _bedrock_layer()]


# ===========================================================================
# Tier 1: Result Dataclass Tests
# ===========================================================================

class TestEQLSiteResponseResult:
    """Tests for the result dataclass (no pystrata needed)."""

    def test_construction_defaults(self):
        r = EQLSiteResponseResult()
        assert r.analysis_type == "equivalent_linear"
        assert r.total_depth_m == 0.0
        assert r.pga_surface_g == 0.0
        assert r.n_iterations == 0
        assert r.converged is True
        assert len(r.time) == 0

    def test_construction_with_values(self):
        r = EQLSiteResponseResult(
            analysis_type="linear_elastic",
            total_depth_m=30.0,
            n_layers=3,
            pga_input_g=0.2,
            pga_surface_g=0.4,
            amplification_factor=2.0,
        )
        assert r.analysis_type == "linear_elastic"
        assert r.total_depth_m == 30.0
        assert r.amplification_factor == 2.0

    def test_analysis_type_stored(self):
        r = EQLSiteResponseResult(analysis_type="equivalent_linear")
        assert r.analysis_type == "equivalent_linear"


class TestEQLResultSummary:

    def test_summary_eql(self):
        r = EQLSiteResponseResult(
            analysis_type="equivalent_linear",
            total_depth_m=30.0, n_layers=3, motion_name="test",
            pga_input_g=0.2, pga_surface_g=0.4,
            amplification_factor=2.0, n_iterations=5,
        )
        s = r.summary()
        assert "EQUIVALENT-LINEAR" in s
        assert "30.0 m" in s
        assert "0.200 g" in s

    def test_summary_linear(self):
        r = EQLSiteResponseResult(analysis_type="linear_elastic")
        s = r.summary()
        assert "LINEAR ELASTIC" in s

    def test_summary_with_strain(self):
        r = EQLSiteResponseResult(
            max_strain_pct=np.array([0.01, 0.05, 0.02]),
        )
        s = r.summary()
        assert "0.0500" in s


class TestEQLResultToDict:

    def test_to_dict_keys(self):
        r = EQLSiteResponseResult(
            total_depth_m=20.0, pga_input_g=0.15, pga_surface_g=0.3,
        )
        d = r.to_dict()
        assert "analysis_type" in d
        assert "total_depth_m" in d
        assert "pga_input_g" in d
        assert "pga_surface_g" in d
        assert "amplification_factor" in d
        assert "converged" in d

    def test_to_dict_json_serializable(self):
        r = EQLSiteResponseResult(
            total_depth_m=20.0, pga_input_g=0.15, pga_surface_g=0.3,
            max_strain_pct=np.array([0.01, 0.05]),
        )
        d = r.to_dict()
        result = json.dumps(d)
        assert isinstance(result, str)

    def test_to_dict_rounding(self):
        r = EQLSiteResponseResult(
            pga_input_g=0.123456789,
            pga_surface_g=0.987654321,
            amplification_factor=8.01234,
        )
        d = r.to_dict()
        assert d["pga_input_g"] == 0.1235
        assert d["pga_surface_g"] == 0.9877
        assert d["amplification_factor"] == 8.012


class TestEQLResultPlots:

    @pytest.fixture(autouse=True)
    def _setup(self):
        import matplotlib
        matplotlib.use("Agg")

    def _make_result(self):
        n = 100
        return EQLSiteResponseResult(
            time=np.linspace(0, 10, n),
            surface_accel_g=np.sin(np.linspace(0, 20, n)) * 0.3,
            depths=np.linspace(0, 30, 10),
            max_strain_pct=np.linspace(0.001, 0.05, 10),
            max_accel_g=np.linspace(0.3, 0.1, 10),
            initial_Vs=np.linspace(200, 500, 10),
            compatible_Vs=np.linspace(180, 480, 10),
            periods=np.logspace(-2, 1, 50),
            Sa_surface_g=np.random.rand(50) * 0.5,
            Sa_input_g=np.random.rand(50) * 0.3,
        )

    def test_plot_surface_motion(self):
        r = self._make_result()
        ax = r.plot_surface_motion(show=False)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_response_spectra(self):
        r = self._make_result()
        ax = r.plot_response_spectra(show=False)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_profile(self):
        r = self._make_result()
        axes = r.plot_profile(show=False)
        assert len(axes) == 2
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_Vs_profile(self):
        r = self._make_result()
        ax = r.plot_Vs_profile(show=False)
        assert ax is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_all(self):
        r = self._make_result()
        fig, axes = r.plot_all(show=False)
        assert axes.shape == (2, 2)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_with_provided_ax(self):
        import matplotlib.pyplot as plt
        r = self._make_result()
        _, ax = plt.subplots()
        returned = r.plot_surface_motion(ax=ax, show=False)
        assert returned is ax
        plt.close("all")


# ===========================================================================
# Tier 1: Input Validation Tests
# ===========================================================================

class TestEQLInputValidation:
    """Tests for _validate_eql_inputs (no pystrata needed)."""

    def test_valid_darendeli(self):
        _validate_eql_inputs([_darendeli_layer(), _bedrock_layer()])

    def test_valid_menq(self):
        _validate_eql_inputs([_menq_layer(), _bedrock_layer()])

    def test_valid_linear(self):
        _validate_eql_inputs([_linear_layer(), _bedrock_layer()])

    def test_valid_custom(self):
        _validate_eql_inputs([_custom_layer(), _bedrock_layer()])

    def test_valid_multilayer(self):
        layers = [
            _darendeli_layer(), _menq_layer(), _linear_layer(),
            _bedrock_layer(),
        ]
        _validate_eql_inputs(layers)

    def test_empty_layers(self):
        with pytest.raises(ValueError, match="non-empty list"):
            _validate_eql_inputs([])

    def test_layer_not_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _validate_eql_inputs(["not_a_dict", _bedrock_layer()])

    def test_layer_missing_thickness(self):
        layer = {"Vs": 250, "unit_wt": 18, "soil_model": "linear",
                 "damping": 0.02}
        with pytest.raises(ValueError, match="missing required key 'thickness'"):
            _validate_eql_inputs([layer, _bedrock_layer()])

    def test_layer_missing_Vs(self):
        layer = {"thickness": 10, "unit_wt": 18, "soil_model": "linear",
                 "damping": 0.02}
        with pytest.raises(ValueError, match="missing required key 'Vs'"):
            _validate_eql_inputs([layer, _bedrock_layer()])

    def test_layer_missing_unit_wt(self):
        layer = {"thickness": 10, "Vs": 250, "soil_model": "linear",
                 "damping": 0.02}
        with pytest.raises(ValueError, match="missing required key 'unit_wt'"):
            _validate_eql_inputs([layer, _bedrock_layer()])

    def test_layer_missing_soil_model(self):
        layer = {"thickness": 10, "Vs": 250, "unit_wt": 18}
        with pytest.raises(ValueError, match="missing required key 'soil_model'"):
            _validate_eql_inputs([layer, _bedrock_layer()])

    def test_layer_negative_thickness(self):
        layer = _darendeli_layer()
        layer["thickness"] = -5
        with pytest.raises(ValueError, match="must be positive"):
            _validate_eql_inputs([layer, _bedrock_layer()])

    def test_layer_zero_Vs(self):
        layer = _darendeli_layer()
        layer["Vs"] = 0
        with pytest.raises(ValueError, match="must be positive"):
            _validate_eql_inputs([layer, _bedrock_layer()])

    def test_layer_unknown_soil_model(self):
        layer = {"thickness": 10, "Vs": 250, "unit_wt": 18,
                 "soil_model": "unknown_model"}
        with pytest.raises(ValueError, match="not recognized"):
            _validate_eql_inputs([layer, _bedrock_layer()])

    def test_darendeli_missing_plas_index(self):
        layer = {"thickness": 10, "Vs": 250, "unit_wt": 18,
                 "soil_model": "darendeli"}
        with pytest.raises(ValueError, match="plas_index"):
            _validate_eql_inputs([layer, _bedrock_layer()])

    def test_linear_missing_damping(self):
        layer = {"thickness": 10, "Vs": 250, "unit_wt": 18,
                 "soil_model": "linear"}
        with pytest.raises(ValueError, match="damping"):
            _validate_eql_inputs([layer, _bedrock_layer()])

    def test_custom_missing_strains(self):
        layer = {"thickness": 10, "Vs": 250, "unit_wt": 18,
                 "soil_model": "custom", "mod_reduc": [1.0],
                 "damping_values": [0.01]}
        with pytest.raises(ValueError, match="strains"):
            _validate_eql_inputs([layer, _bedrock_layer()])

    def test_custom_array_length_mismatch(self):
        layer = {"thickness": 10, "Vs": 250, "unit_wt": 18,
                 "soil_model": "custom",
                 "strains": [0.001, 0.01],
                 "mod_reduc": [1.0],  # wrong length
                 "damping_values": [0.01, 0.05]}
        with pytest.raises(ValueError, match="must match strains length"):
            _validate_eql_inputs([layer, _bedrock_layer()])

    def test_no_bedrock_layer(self):
        layers = [_darendeli_layer(), _darendeli_layer()]
        with pytest.raises(ValueError, match="bedrock half-space"):
            _validate_eql_inputs(layers)

    def test_strain_ratio_out_of_range(self):
        with pytest.raises(ValueError, match="strain_ratio"):
            _validate_eql_inputs(_simple_profile(), strain_ratio=0.2)

    def test_single_layer_only_bedrock(self):
        with pytest.raises(ValueError, match="At least 2 layers"):
            _validate_eql_inputs([_bedrock_layer()])


# ===========================================================================
# Tier 1: Utility Tests
# ===========================================================================

class TestPystrataUtils:

    def test_has_pystrata_returns_bool(self):
        result = has_pystrata()
        assert isinstance(result, bool)

    def test_import_pystrata_error_message(self):
        if has_pystrata():
            pytest.skip("pystrata is installed; can't test error path")
        from pystrata_agent.pystrata_utils import import_pystrata
        with pytest.raises(ImportError, match="pip install pystrata"):
            import_pystrata()


# ===========================================================================
# Tier 1: Foundry Agent Metadata Tests
# ===========================================================================

class TestFoundryAgentMetadata:

    def test_list_methods_all(self):
        import pystrata_agent_foundry as paf
        result = json.loads(paf.pystrata_list_methods.__wrapped__(""))
        assert "Site Response" in result
        methods = result["Site Response"]
        assert "eql_site_response" in methods
        assert "linear_site_response" in methods

    def test_list_methods_filter(self):
        import pystrata_agent_foundry as paf
        result = json.loads(
            paf.pystrata_list_methods.__wrapped__("Site Response"))
        assert "Site Response" in result

    def test_list_methods_unknown_category(self):
        import pystrata_agent_foundry as paf
        result = json.loads(
            paf.pystrata_list_methods.__wrapped__("Nonexistent"))
        assert "error" in result

    def test_describe_eql_method(self):
        import pystrata_agent_foundry as paf
        result = json.loads(
            paf.pystrata_describe_method.__wrapped__("eql_site_response"))
        assert "category" in result
        assert "parameters" in result
        assert "returns" in result
        assert "layers" in result["parameters"]

    def test_describe_linear_method(self):
        import pystrata_agent_foundry as paf
        result = json.loads(
            paf.pystrata_describe_method.__wrapped__("linear_site_response"))
        assert "category" in result

    def test_describe_unknown_method(self):
        import pystrata_agent_foundry as paf
        result = json.loads(
            paf.pystrata_describe_method.__wrapped__("nonexistent"))
        assert "error" in result

    def test_agent_invalid_json(self):
        import pystrata_agent_foundry as paf
        result = json.loads(
            paf.pystrata_agent.__wrapped__("eql_site_response", "not json"))
        assert "error" in result
        assert "Invalid" in result["error"]

    def test_agent_unknown_method(self):
        import pystrata_agent_foundry as paf
        result = json.loads(
            paf.pystrata_agent.__wrapped__("nonexistent", "{}"))
        assert "error" in result
        assert "Unknown method" in result["error"]

    def test_agent_validation_error(self):
        """Validation errors should be caught before pystrata import check."""
        import pystrata_agent_foundry as paf
        result = json.loads(
            paf.pystrata_agent.__wrapped__(
                "eql_site_response", '{"layers": []}'))
        assert "error" in result

    def test_parameters_have_required_fields(self):
        import pystrata_agent_foundry as paf
        for method_name in ("eql_site_response", "linear_site_response"):
            info = json.loads(
                paf.pystrata_describe_method.__wrapped__(method_name))
            for param_name, param_info in info["parameters"].items():
                assert "type" in param_info, (
                    f"{method_name}.{param_name} missing 'type'")
                assert "description" in param_info, (
                    f"{method_name}.{param_name} missing 'description'")


# ===========================================================================
# Tier 2: Integration Tests (require pystrata)
# ===========================================================================

@requires_pystrata
class TestEQLSiteResponseIntegration:

    def test_darendeli_single_layer(self):
        from pystrata_agent import analyze_eql_site_response
        result = analyze_eql_site_response(
            layers=_simple_profile(),
            motion="synthetic_pulse",
        )
        assert isinstance(result, EQLSiteResponseResult)
        assert result.analysis_type == "equivalent_linear"
        assert result.total_depth_m == 10.0
        assert result.n_layers == 1
        assert result.pga_input_g > 0
        assert result.pga_surface_g > 0
        assert len(result.time) > 0
        assert len(result.surface_accel_g) > 0
        assert len(result.depths) > 0
        assert len(result.max_strain_pct) > 0
        assert len(result.max_accel_g) > 0
        assert len(result.periods) > 0
        assert len(result.Sa_surface_g) > 0
        assert len(result.Sa_input_g) > 0

    def test_menq_single_layer(self):
        from pystrata_agent import analyze_eql_site_response
        result = analyze_eql_site_response(
            layers=[_menq_layer(), _bedrock_layer()],
            motion="synthetic_pulse",
        )
        assert result.pga_surface_g > 0
        assert result.n_layers == 1

    def test_linear_elastic(self):
        from pystrata_agent import analyze_linear_site_response
        result = analyze_linear_site_response(
            layers=_simple_profile(),
            motion="synthetic_pulse",
        )
        assert result.analysis_type == "linear_elastic"
        assert result.n_iterations == 0
        assert result.converged is True
        assert result.pga_surface_g > 0

    def test_custom_curves(self):
        from pystrata_agent import analyze_eql_site_response
        result = analyze_eql_site_response(
            layers=[_custom_layer(), _bedrock_layer()],
            motion="synthetic_pulse",
        )
        assert result.pga_surface_g > 0

    def test_multilayer_mixed(self):
        from pystrata_agent import analyze_eql_site_response
        layers = [
            _darendeli_layer(thickness=5.0),
            _menq_layer(thickness=5.0),
            _linear_layer(thickness=5.0),
            _bedrock_layer(),
        ]
        result = analyze_eql_site_response(
            layers=layers, motion="synthetic_pulse")
        assert result.n_layers == 3
        assert result.total_depth_m == 15.0

    def test_amplification_reasonable(self):
        from pystrata_agent import analyze_eql_site_response
        result = analyze_eql_site_response(
            layers=_simple_profile(), motion="synthetic_pulse")
        assert 0.1 < result.amplification_factor < 10.0

    def test_compatible_Vs_le_initial(self):
        """EQL stiffness degradation: compatible Vs should be <= initial."""
        from pystrata_agent import analyze_eql_site_response
        result = analyze_eql_site_response(
            layers=_simple_profile(), motion="synthetic_pulse")
        if len(result.initial_Vs) > 0 and len(result.compatible_Vs) > 0:
            # At least some layers should show degradation
            assert np.all(result.compatible_Vs <= result.initial_Vs + 0.01)

    def test_eql_vs_linear_differ(self):
        """EQL and linear should give different surface PGAs."""
        from pystrata_agent import (
            analyze_eql_site_response, analyze_linear_site_response)
        profile = _simple_profile()
        r_eql = analyze_eql_site_response(
            layers=profile, motion="synthetic_pulse")
        r_lin = analyze_linear_site_response(
            layers=profile, motion="synthetic_pulse")
        # They should differ (EQL degrades stiffness, changing response)
        assert r_eql.pga_surface_g != pytest.approx(
            r_lin.pga_surface_g, abs=1e-6)

    def test_auto_stress_mean(self):
        """Without explicit stress_mean, it should be auto-calculated."""
        from pystrata_agent import analyze_eql_site_response
        # Same layer with and without explicit stress_mean
        layer_auto = _darendeli_layer(thickness=20.0)
        layer_manual = _darendeli_layer(thickness=20.0)
        # sigma_v_mid = 18.0 * 10.0 = 180 kPa
        # stress_mean = 180 * (1 + 2*0.5) / 3 = 120 kPa
        layer_manual["stress_mean"] = 120.0

        r_auto = analyze_eql_site_response(
            layers=[layer_auto, _bedrock_layer()],
            motion="synthetic_pulse")
        r_manual = analyze_eql_site_response(
            layers=[layer_manual, _bedrock_layer()],
            motion="synthetic_pulse")
        # Should produce very similar results
        assert r_auto.pga_surface_g == pytest.approx(
            r_manual.pga_surface_g, rel=0.01)

    def test_result_json_serializable(self):
        from pystrata_agent import analyze_eql_site_response
        result = analyze_eql_site_response(
            layers=_simple_profile(), motion="synthetic_pulse")
        d = result.to_dict()
        text = json.dumps(d)
        assert isinstance(text, str)

    def test_custom_motion_input(self):
        """Using accel_history + dt instead of built-in motion."""
        from pystrata_agent import analyze_eql_site_response
        n = 2000
        dt = 0.01
        t = np.arange(n) * dt
        accel = 0.2 * np.sin(2 * np.pi * 2.0 * t) * np.exp(-t / 5.0)
        result = analyze_eql_site_response(
            layers=_simple_profile(),
            accel_history=accel.tolist(),
            dt=dt,
        )
        assert result.motion_name == "custom"
        assert result.pga_surface_g > 0

    def test_convergence_flag(self):
        from pystrata_agent import analyze_eql_site_response
        result = analyze_eql_site_response(
            layers=_simple_profile(),
            motion="synthetic_pulse",
            max_iterations=50,  # plenty of room
        )
        assert result.converged is True
