"""
Tests for the opensees_agent module.

Tier 1 tests (no openseespy required):
  - Result dataclasses: construction, summary(), to_dict(), plot_*()
  - Input validation for analyze_pm4sand_dss
  - Ground motions: list, get, validation
  - Utility functions: clean_numpy, compute_response_spectrum, has_opensees
  - Foundry agent: list_methods, describe_method, error handling

Tier 2 tests (require openseespy, skipped if not installed):
  - PM4Sand DSS integration
  - Site response integration
  - Site response benchmarks (GM-77): closed-form response of a uniform
    damped layer on elastic rock (Kramer 1996, Sec. 7.2), pystrata's
    frequency-domain linear and equivalent-linear runs of the same column

Windows Smart App Control blocks the openseespy DLL, so every Tier 2
test skips there; run them in WSL or on CI (BACKLOG INF-13). Every site
response run here goes through the child process with a time limit, so a
stuck solver step fails the test instead of hanging the run.
"""

import json
import math

import numpy as np
import pytest

from opensees_agent.results import (
    PM4SandDSSResult, SiteResponseResult,
)
from opensees_agent.opensees_utils import (
    has_opensees, clean_numpy, compute_response_spectrum,
)
from opensees_agent.ground_motions import (
    get_motion, list_motions, validate_motion_input,
)
from opensees_agent.pm4sand_dss import _validate_pm4sand_inputs
from opensees_agent.site_response import _validate_site_response_inputs

# Skip marker for tests that need openseespy
requires_opensees = pytest.mark.skipif(
    not has_opensees(), reason="openseespy not installed"
)

# Wall-clock limit for every site response run in this file (each takes
# a few seconds); a stuck solver step fails the test instead of hanging
# CI (GM-77, INF-07).
TEST_TIME_LIMIT_S = 180.0


# ===========================================================================
# PM4SandDSSResult
# ===========================================================================

class TestPM4SandDSSResult:
    """Tests for PM4SandDSSResult dataclass."""

    def _make_result(self, liquefied=True):
        """Helper to build a test result with known data."""
        n = 100
        time = np.linspace(0, 10, n)
        strain = np.sin(2 * np.pi * 0.5 * time) * 2.0  # +/- 2%
        stress = np.sin(2 * np.pi * 0.5 * time) * 15.0  # +/- 15 kPa
        sv = np.linspace(100, 5 if liquefied else 60, n)
        ru = 1.0 - sv / 100.0
        return PM4SandDSSResult(
            Dr=0.5,
            sigma_v_kPa=100.0,
            CSR_applied=0.15,
            K0=0.5,
            n_cycles_to_liq=5.5 if liquefied else float('inf'),
            liquefied=liquefied,
            max_ru=float(np.max(ru)),
            max_shear_strain_pct=2.0,
            time=time,
            shear_stress_kPa=stress,
            shear_strain_pct=strain,
            vert_eff_stress_kPa=sv,
            ru=ru,
        )

    def test_construction_defaults(self):
        r = PM4SandDSSResult()
        assert r.Dr == 0.0
        assert r.liquefied is False
        assert len(r.time) == 0

    def test_summary_liquefied(self):
        r = self._make_result(liquefied=True)
        s = r.summary()
        assert "PM4SAND" in s
        assert "YES" in s
        assert "5.5" in s

    def test_summary_no_liquefaction(self):
        r = self._make_result(liquefied=False)
        s = r.summary()
        assert "NO" in s

    def test_to_dict_liquefied(self):
        r = self._make_result(liquefied=True)
        d = r.to_dict()
        assert d["liquefied"] is True
        assert d["n_cycles_to_liq"] == 5.5
        assert d["Dr"] == 0.5
        assert d["CSR_applied"] == 0.15
        assert isinstance(d["max_ru"], float)

    def test_to_dict_no_liq(self):
        r = self._make_result(liquefied=False)
        d = r.to_dict()
        assert d["liquefied"] is False
        assert d["n_cycles_to_liq"] is None

    def test_to_dict_json_serializable(self):
        r = self._make_result()
        d = r.to_dict()
        json.dumps(d)  # should not raise


class TestPM4SandDSSResultPlots:
    """Plot smoke tests for PM4SandDSSResult."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import matplotlib
        matplotlib.use("Agg")

    def _make_result(self):
        n = 50
        return PM4SandDSSResult(
            Dr=0.5, sigma_v_kPa=100.0, CSR_applied=0.15, K0=0.5,
            n_cycles_to_liq=3.0, liquefied=True,
            max_ru=0.98, max_shear_strain_pct=1.5,
            time=np.linspace(0, 5, n),
            shear_stress_kPa=np.sin(np.linspace(0, 6, n)) * 15,
            shear_strain_pct=np.sin(np.linspace(0, 6, n)) * 1.5,
            vert_eff_stress_kPa=np.linspace(100, 2, n),
            ru=np.linspace(0, 0.98, n),
        )

    def test_plot_stress_strain(self):
        import matplotlib
        r = self._make_result()
        ax = r.plot_stress_strain(show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_stress_path(self):
        import matplotlib
        r = self._make_result()
        ax = r.plot_stress_path(show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_pore_pressure(self):
        import matplotlib
        r = self._make_result()
        ax = r.plot_pore_pressure(show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_all(self):
        import matplotlib
        r = self._make_result()
        fig, axes = r.plot_all(show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert axes.shape == (2, 2)
        import matplotlib.pyplot as plt
        plt.close("all")


# ===========================================================================
# SiteResponseResult
# ===========================================================================

class TestSiteResponseResult:
    """Tests for SiteResponseResult dataclass."""

    def _make_result(self):
        n = 200
        return SiteResponseResult(
            total_depth_m=30.0, n_layers=3,
            motion_name="synthetic_pulse",
            pga_input_g=0.3, pga_surface_g=0.42,
            amplification_factor=1.4,
            time=np.linspace(0, 20, n),
            surface_accel_g=np.sin(np.linspace(0, 20, n)) * 0.42,
            depths=np.linspace(0, 30, 10),
            max_strain_pct=np.linspace(0.5, 0.01, 10),
            max_accel_g=np.linspace(0.42, 0.3, 10),
            max_pore_pressure_ratio=np.linspace(0.8, 0.1, 10),
            periods=np.logspace(-2, 1, 50),
            Sa_surface_g=np.ones(50) * 0.5,
            Sa_input_g=np.ones(50) * 0.3,
        )

    def test_construction_defaults(self):
        r = SiteResponseResult()
        assert r.total_depth_m == 0.0
        assert r.amplification_factor == 0.0

    def test_summary(self):
        r = self._make_result()
        s = r.summary()
        assert "SITE RESPONSE" in s
        assert "0.420" in s or "0.42" in s
        assert "1.4" in s or "1.40" in s

    def test_to_dict(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["pga_surface_g"] == 0.42
        assert d["amplification_factor"] == 1.4
        assert d["n_layers"] == 3
        json.dumps(d)  # serializable


class TestSiteResponseResultPlots:
    """Plot smoke tests for SiteResponseResult."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import matplotlib
        matplotlib.use("Agg")

    def _make_result(self):
        n = 100
        return SiteResponseResult(
            total_depth_m=30.0, n_layers=3,
            motion_name="test", pga_input_g=0.3, pga_surface_g=0.4,
            amplification_factor=1.33,
            time=np.linspace(0, 10, n),
            surface_accel_g=np.sin(np.linspace(0, 10, n)) * 0.4,
            depths=np.linspace(0, 30, 8),
            max_strain_pct=np.linspace(0.5, 0.01, 8),
            max_accel_g=np.linspace(0.4, 0.3, 8),
            max_pore_pressure_ratio=np.linspace(0.7, 0.1, 8),
            periods=np.logspace(-2, 1, 30),
            Sa_surface_g=np.ones(30) * 0.5,
            Sa_input_g=np.ones(30) * 0.3,
        )

    def test_plot_surface_motion(self):
        import matplotlib
        ax = self._make_result().plot_surface_motion(show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_response_spectra(self):
        import matplotlib
        ax = self._make_result().plot_response_spectra(show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")


# ===========================================================================
# Input validation (PM4Sand DSS)
# ===========================================================================

class TestPM4SandValidation:
    """Input validation for analyze_pm4sand_dss."""

    def test_Dr_zero(self):
        with pytest.raises(ValueError, match="Dr must be"):
            _validate_pm4sand_inputs(0.0, 476, 0.53, 1.7, 100, 0.15, 0.5, 30)

    def test_Dr_negative(self):
        with pytest.raises(ValueError, match="Dr must be"):
            _validate_pm4sand_inputs(-0.1, 476, 0.53, 1.7, 100, 0.15, 0.5, 30)

    def test_Dr_too_high(self):
        with pytest.raises(ValueError, match="Dr must be"):
            _validate_pm4sand_inputs(1.5, 476, 0.53, 1.7, 100, 0.15, 0.5, 30)

    def test_G0_zero(self):
        with pytest.raises(ValueError, match="G0 must be positive"):
            _validate_pm4sand_inputs(0.5, 0, 0.53, 1.7, 100, 0.15, 0.5, 30)

    def test_hpo_negative(self):
        with pytest.raises(ValueError, match="hpo must be positive"):
            _validate_pm4sand_inputs(0.5, 476, -0.1, 1.7, 100, 0.15, 0.5, 30)

    def test_Den_zero(self):
        with pytest.raises(ValueError, match="Den.*must be positive"):
            _validate_pm4sand_inputs(0.5, 476, 0.53, 0, 100, 0.15, 0.5, 30)

    def test_sigma_v_negative(self):
        with pytest.raises(ValueError, match="sigma_v must be positive"):
            _validate_pm4sand_inputs(0.5, 476, 0.53, 1.7, -10, 0.15, 0.5, 30)

    def test_CSR_zero(self):
        with pytest.raises(ValueError, match="CSR must be positive"):
            _validate_pm4sand_inputs(0.5, 476, 0.53, 1.7, 100, 0.0, 0.5, 30)

    def test_K0_out_of_range(self):
        with pytest.raises(ValueError, match="K0 must be"):
            _validate_pm4sand_inputs(0.5, 476, 0.53, 1.7, 100, 0.15, 2.5, 30)

    def test_n_cycles_zero(self):
        with pytest.raises(ValueError, match="n_cycles must be"):
            _validate_pm4sand_inputs(0.5, 476, 0.53, 1.7, 100, 0.15, 0.5, 0)

    def test_valid_inputs(self):
        # Should not raise
        _validate_pm4sand_inputs(0.5, 476, 0.53, 1.7, 100, 0.15, 0.5, 30)

    def test_edge_Dr_1(self):
        # Dr = 1.0 is valid (very dense)
        _validate_pm4sand_inputs(1.0, 476, 0.53, 1.7, 100, 0.15, 0.5, 30)


# ===========================================================================
# Ground motions
# ===========================================================================

class TestGroundMotions:
    """Tests for ground_motions module."""

    def test_list_motions(self):
        motions = list_motions()
        assert "synthetic_pulse" in motions
        assert "synthetic_long" in motions
        assert "dt" in motions["synthetic_pulse"]
        assert "pga_g" in motions["synthetic_pulse"]

    def test_get_synthetic_pulse(self):
        accel, dt = get_motion("synthetic_pulse")
        assert isinstance(accel, np.ndarray)
        assert dt == 0.01
        assert len(accel) > 100
        assert abs(np.max(np.abs(accel)) - 0.3) < 0.05  # PGA ~ 0.3g

    def test_get_synthetic_long(self):
        accel, dt = get_motion("synthetic_long")
        assert dt == 0.01
        assert len(accel) > len(get_motion("synthetic_pulse")[0])

    def test_get_motion_case_insensitive(self):
        accel, _ = get_motion("Synthetic_Pulse")
        assert len(accel) > 0

    def test_get_motion_unknown(self):
        with pytest.raises(ValueError, match="Unknown ground motion"):
            get_motion("nonexistent_quake")

    def test_validate_motion_by_name(self):
        accel, dt = validate_motion_input(motion="synthetic_pulse")
        assert len(accel) > 0
        assert dt > 0

    def test_validate_motion_custom(self):
        custom = np.sin(np.linspace(0, 10, 500)) * 0.2
        accel, dt = validate_motion_input(accel_history=custom, dt=0.02)
        assert len(accel) == 500
        assert dt == 0.02

    def test_validate_motion_custom_no_dt(self):
        with pytest.raises(ValueError, match="dt must be a positive"):
            validate_motion_input(accel_history=[1, 2, 3] * 10, dt=None)

    def test_validate_motion_custom_short(self):
        with pytest.raises(ValueError, match="at least 10 points"):
            validate_motion_input(accel_history=[1, 2, 3], dt=0.01)

    def test_validate_motion_none(self):
        with pytest.raises(ValueError, match="Must provide"):
            validate_motion_input()


# ===========================================================================
# Utility functions
# ===========================================================================

class TestCleanNumpy:
    """Tests for clean_numpy conversion utility."""

    def test_ndarray_to_list(self):
        assert clean_numpy(np.array([1, 2, 3])) == [1, 2, 3]

    def test_float64(self):
        assert isinstance(clean_numpy(np.float64(3.14)), float)
        assert clean_numpy(np.float64(3.14)) == pytest.approx(3.14)

    def test_int64(self):
        assert isinstance(clean_numpy(np.int64(42)), int)
        assert clean_numpy(np.int64(42)) == 42

    def test_nan_to_none(self):
        assert clean_numpy(float('nan')) is None
        assert clean_numpy(np.float64('nan')) is None

    def test_bool(self):
        assert clean_numpy(np.bool_(True)) is True
        assert clean_numpy(np.bool_(False)) is False

    def test_dict_recursive(self):
        d = {"a": np.float64(1.0), "b": np.array([1, 2])}
        cleaned = clean_numpy(d)
        assert cleaned["a"] == 1.0
        assert cleaned["b"] == [1, 2]
        json.dumps(cleaned)

    def test_list_recursive(self):
        lst = [np.float64(1.0), np.int64(2)]
        cleaned = clean_numpy(lst)
        assert cleaned == [1.0, 2]

    def test_passthrough(self):
        assert clean_numpy("hello") == "hello"
        assert clean_numpy(42) == 42
        assert clean_numpy(None) is None


class TestResponseSpectrum:
    """Tests for compute_response_spectrum."""

    def test_basic(self):
        # Simple sine wave at 1 Hz (resonant period = 1.0s)
        dt = 0.01
        t = np.arange(0, 10, dt)
        accel = 0.1 * np.sin(2 * np.pi * 1.0 * t)
        periods = np.array([0.5, 1.0, 2.0, 4.0])
        T, Sa = compute_response_spectrum(accel, dt, periods)
        assert len(T) == 4
        assert len(Sa) == 4
        assert all(s >= 0 for s in Sa)
        # Peak response near resonance (T=1.0s) > off-resonance (T=0.5s)
        assert Sa[1] > Sa[0]

    def test_default_periods(self):
        accel = np.random.randn(1000) * 0.1
        T, Sa = compute_response_spectrum(accel, 0.01)
        assert len(T) == 100  # default 100 points
        assert T[0] == pytest.approx(0.01, rel=0.01)

    def test_zero_motion(self):
        accel = np.zeros(500)
        _, Sa = compute_response_spectrum(accel, 0.01, [0.5, 1.0])
        assert all(s == 0 for s in Sa)


class TestHasOpensees:
    """Tests for has_opensees utility."""

    def test_returns_bool(self):
        result = has_opensees()
        assert isinstance(result, bool)


# ===========================================================================
# Foundry agent metadata (always runs, no openseespy needed)
# ===========================================================================

class TestFoundryAgentMetadata:
    """Tests for opensees_agent_foundry list/describe functions."""

    def test_list_methods_all(self):
        # Import the function but skip the @function decorator
        import foundry.opensees_agent_foundry as oaf
        result = json.loads(oaf.opensees_list_methods.__wrapped__(""))
        assert "Cyclic Element Tests" in result
        assert "pm4sand_cyclic_dss" in result["Cyclic Element Tests"]

    def test_list_methods_filter(self):
        import foundry.opensees_agent_foundry as oaf
        result = json.loads(oaf.opensees_list_methods.__wrapped__(
            "Cyclic Element Tests"))
        assert "Cyclic Element Tests" in result

    def test_list_methods_unknown_category(self):
        import foundry.opensees_agent_foundry as oaf
        result = json.loads(oaf.opensees_list_methods.__wrapped__("Nonexistent"))
        assert "error" in result

    def test_describe_method(self):
        import foundry.opensees_agent_foundry as oaf
        result = json.loads(oaf.opensees_describe_method.__wrapped__(
            "pm4sand_cyclic_dss"))
        assert "category" in result
        assert "parameters" in result
        assert "Dr" in result["parameters"]
        assert result["parameters"]["Dr"]["required"] is True

    def test_describe_unknown_method(self):
        import foundry.opensees_agent_foundry as oaf
        result = json.loads(oaf.opensees_describe_method.__wrapped__("bogus"))
        assert "error" in result

    def test_agent_invalid_json(self):
        import foundry.opensees_agent_foundry as oaf
        result = json.loads(oaf.opensees_agent.__wrapped__(
            "pm4sand_cyclic_dss", "not json"))
        assert "error" in result
        assert "Invalid parameters_json" in result["error"]

    def test_agent_unknown_method(self):
        import foundry.opensees_agent_foundry as oaf
        result = json.loads(oaf.opensees_agent.__wrapped__(
            "bogus_method", "{}"))
        assert "error" in result
        assert "Unknown method" in result["error"]

    def test_list_methods_includes_site_response(self):
        import foundry.opensees_agent_foundry as oaf
        result = json.loads(oaf.opensees_list_methods.__wrapped__(""))
        assert "Site Response" in result
        assert "site_response_1d" in result["Site Response"]

    def test_describe_site_response(self):
        import foundry.opensees_agent_foundry as oaf
        result = json.loads(oaf.opensees_describe_method.__wrapped__(
            "site_response_1d"))
        assert result["category"] == "Site Response"
        assert "layers" in result["parameters"]
        assert "pga_surface_g" in result["returns"]

    def test_list_methods_filter_site_response(self):
        import foundry.opensees_agent_foundry as oaf
        result = json.loads(oaf.opensees_list_methods.__wrapped__(
            "Site Response"))
        assert "Site Response" in result
        assert "site_response_1d" in result["Site Response"]


# ===========================================================================
# Tier 2: Integration tests (require openseespy)
# ===========================================================================

class TestPM4SandDSSIntegration:
    """Integration tests that run actual OpenSees models."""

    @requires_opensees
    def test_basic_analysis(self):
        """Run a basic PM4Sand DSS analysis and verify result structure."""
        from opensees_agent.pm4sand_dss import analyze_pm4sand_dss

        result = analyze_pm4sand_dss(
            Dr=0.5, G0=476.0, hpo=0.53, Den=1.42,
            sigma_v=100.0, CSR=0.2, K0=0.5,
            n_cycles=5,
        )
        assert isinstance(result, PM4SandDSSResult)
        assert result.Dr == 0.5
        assert result.CSR_applied == 0.2
        assert len(result.time) > 0
        assert len(result.shear_stress_kPa) == len(result.time)
        assert result.max_ru >= 0

    @requires_opensees
    def test_dense_sand_fewer_cycles(self):
        """Dense sand (Dr=0.8) should need more cycles than loose (Dr=0.3)."""
        from opensees_agent.pm4sand_dss import analyze_pm4sand_dss

        # This test verifies relative behavior, not exact numbers
        result_loose = analyze_pm4sand_dss(
            Dr=0.3, G0=300.0, hpo=0.40, Den=1.5,
            sigma_v=100.0, CSR=0.2, n_cycles=30,
        )
        result_dense = analyze_pm4sand_dss(
            Dr=0.8, G0=700.0, hpo=1.0, Den=1.7,
            sigma_v=100.0, CSR=0.2, n_cycles=30,
        )
        # Dense sand should resist liquefaction better
        if result_loose.liquefied and result_dense.liquefied:
            assert result_dense.n_cycles_to_liq >= result_loose.n_cycles_to_liq
        elif result_loose.liquefied and not result_dense.liquefied:
            pass  # expected: dense didn't liquefy
        # Both not liquefying is also acceptable (low CSR)

    @requires_opensees
    def test_result_to_dict_serializable(self):
        """Verify the result can be JSON-serialized."""
        from opensees_agent.pm4sand_dss import analyze_pm4sand_dss

        result = analyze_pm4sand_dss(
            Dr=0.5, G0=476.0, hpo=0.53, Den=1.42,
            sigma_v=100.0, CSR=0.15, n_cycles=3,
        )
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 10


# ===========================================================================
# Input validation (Site Response)
# ===========================================================================

class TestSiteResponseValidation:
    """Input validation for analyze_site_response (Tier 1)."""

    def _good_sand(self):
        return {"thickness": 5.0, "Vs": 200.0, "density": 1.8,
                "material_type": "sand", "phi": 33.0}

    def _good_clay(self):
        return {"thickness": 3.0, "Vs": 120.0, "density": 1.6,
                "material_type": "clay", "su": 50.0}

    def _call(self, **overrides):
        defaults = dict(
            layers=[self._good_sand()],
            gwt_depth=0.0, bedrock_Vs=760.0, bedrock_density=2.4,
            damping=0.02, scale_factor=1.0, n_elem_per_layer=4,
        )
        defaults.update(overrides)
        _validate_site_response_inputs(**defaults)

    def test_valid_sand(self):
        self._call()  # should not raise

    def test_valid_clay(self):
        self._call(layers=[self._good_clay()])

    def test_valid_multilayer(self):
        self._call(layers=[self._good_sand(), self._good_clay()])

    def test_empty_layers(self):
        with pytest.raises(ValueError, match="non-empty list"):
            self._call(layers=[])

    def test_layer_not_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            self._call(layers=["not a dict"])

    def test_layer_missing_thickness(self):
        bad = self._good_sand()
        del bad["thickness"]
        with pytest.raises(ValueError, match="missing required key 'thickness'"):
            self._call(layers=[bad])

    def test_layer_missing_Vs(self):
        bad = self._good_sand()
        del bad["Vs"]
        with pytest.raises(ValueError, match="missing required key 'Vs'"):
            self._call(layers=[bad])

    def test_layer_missing_density(self):
        bad = self._good_sand()
        del bad["density"]
        with pytest.raises(ValueError, match="missing required key 'density'"):
            self._call(layers=[bad])

    def test_layer_missing_material_type(self):
        bad = self._good_sand()
        del bad["material_type"]
        with pytest.raises(ValueError, match="missing required key 'material_type'"):
            self._call(layers=[bad])

    def test_layer_negative_thickness(self):
        bad = self._good_sand()
        bad["thickness"] = -1.0
        with pytest.raises(ValueError, match="thickness.*must be positive"):
            self._call(layers=[bad])

    def test_layer_zero_Vs(self):
        bad = self._good_sand()
        bad["Vs"] = 0
        with pytest.raises(ValueError, match="Vs.*must be positive"):
            self._call(layers=[bad])

    def test_layer_bad_material_type(self):
        bad = self._good_sand()
        bad["material_type"] = "rock"
        with pytest.raises(ValueError, match="must be 'sand' or 'clay'"):
            self._call(layers=[bad])

    def test_sand_missing_phi(self):
        bad = {"thickness": 5, "Vs": 200, "density": 1.8,
               "material_type": "sand"}
        with pytest.raises(ValueError, match="requires 'phi'"):
            self._call(layers=[bad])

    def test_clay_missing_su(self):
        bad = {"thickness": 5, "Vs": 120, "density": 1.6,
               "material_type": "clay"}
        with pytest.raises(ValueError, match="requires 'su'"):
            self._call(layers=[bad])

    def test_gwt_negative(self):
        with pytest.raises(ValueError, match="gwt_depth must be >= 0"):
            self._call(gwt_depth=-1.0)

    def test_bedrock_Vs_negative(self):
        with pytest.raises(ValueError, match="bedrock_Vs must be positive"):
            self._call(bedrock_Vs=-100)

    def test_bedrock_density_zero(self):
        with pytest.raises(ValueError, match="bedrock_density must be positive"):
            self._call(bedrock_density=0)

    def test_damping_out_of_range(self):
        with pytest.raises(ValueError, match="damping must be"):
            self._call(damping=1.5)

    def test_scale_factor_zero(self):
        with pytest.raises(ValueError, match="scale_factor must be positive"):
            self._call(scale_factor=0)

    def test_n_elem_zero(self):
        with pytest.raises(ValueError, match="n_elem_per_layer must be >= 1"):
            self._call(n_elem_per_layer=0)


# ===========================================================================
# Tier 2: Site Response Integration (require openseespy)
# ===========================================================================

class TestSiteResponseIntegration:
    """Integration tests for 1D site response (require openseespy)."""

    @requires_opensees
    def test_uniform_sand_profile(self):
        """Uniform sand layer: basic structure and sanity checks."""
        from opensees_agent.site_response import analyze_site_response

        result = analyze_site_response(
            layers=[{
                "thickness": 20.0, "Vs": 200.0, "density": 1.9,
                "material_type": "sand", "phi": 33.0,
            }],
            motion="synthetic_pulse",
            bedrock_Vs=760.0,
            n_elem_per_layer=4,
            time_limit_s=TEST_TIME_LIMIT_S,
        )
        assert isinstance(result, SiteResponseResult)
        assert result.total_depth_m == 20.0
        assert result.n_layers == 1
        assert result.pga_input_g > 0
        # GM-77: 8.5e-7 g came out of a 0.30 g input when the motion was
        # never applied; "> 0" let that pass.
        assert result.pga_surface_g > 0.1 * result.pga_input_g
        assert result.amplification_factor > 0.1
        # Saturated sand of the Dr = 40 % calibration under 0.30 g for
        # 20 s near the site frequency liquefies.
        assert np.max(result.max_pore_pressure_ratio) > 0.9
        assert result.analysis_type == "nonlinear effective stress"
        assert result.site_period_s == pytest.approx(0.4)
        assert len(result.time) > 100
        assert len(result.surface_accel_g) == len(result.time)
        assert len(result.depths) == 4  # n_elem_per_layer
        assert len(result.max_strain_pct) == 4
        assert len(result.max_accel_g) == 4
        assert len(result.max_pore_pressure_ratio) == 4
        assert len(result.periods) > 0
        assert len(result.Sa_surface_g) == len(result.periods)
        assert len(result.Sa_input_g) == len(result.periods)

        # Sanity
        assert result.pga_surface_g < 5.0
        assert np.all(result.max_strain_pct >= 0)
        assert np.all(result.max_pore_pressure_ratio >= 0)
        # ru = excess pore pressure / sigma'v0 (GM-79), not capped: once
        # the sand has liquefied the pore pressure also carries the
        # column's swings in total vertical stress (measured 1.016 here;
        # the old 1 - sigma'v / sigma'v0 was capped at 1).
        assert np.all(result.max_pore_pressure_ratio <= 1.05)

        d = result.to_dict()
        json.dumps(d)

    @requires_opensees
    def test_uniform_clay_profile(self):
        """Uniform clay layer: runs and returns sensible results."""
        from opensees_agent.site_response import analyze_site_response

        result = analyze_site_response(
            layers=[{
                "thickness": 15.0, "Vs": 150.0, "density": 1.7,
                "material_type": "clay", "su": 60.0,
            }],
            motion="synthetic_pulse",
            n_elem_per_layer=3,
            time_limit_s=TEST_TIME_LIMIT_S,
        )
        assert isinstance(result, SiteResponseResult)
        assert result.n_layers == 1
        assert result.pga_surface_g > 0.1 * result.pga_input_g
        assert len(result.depths) == 3
        # Clay (PIMY, total stress) builds no excess pore pressure
        assert np.max(result.max_pore_pressure_ratio) < 0.1

    @requires_opensees
    def test_multilayer_sand_clay(self):
        """Multi-layer profile (sand over clay over sand)."""
        from opensees_agent.site_response import analyze_site_response

        result = analyze_site_response(
            layers=[
                {"thickness": 5.0, "Vs": 150.0, "density": 1.8,
                 "material_type": "sand", "phi": 30.0},
                {"thickness": 10.0, "Vs": 120.0, "density": 1.6,
                 "material_type": "clay", "su": 40.0},
                {"thickness": 10.0, "Vs": 250.0, "density": 1.9,
                 "material_type": "sand", "phi": 35.0},
            ],
            motion="synthetic_pulse",
            gwt_depth=2.0,
            n_elem_per_layer=2,
            time_limit_s=TEST_TIME_LIMIT_S,
        )
        assert result.total_depth_m == pytest.approx(25.0)
        assert result.n_layers == 3
        assert len(result.depths) == 6  # 3 layers * 2 elements
        assert result.pga_surface_g > 0.1 * result.pga_input_g
        d = result.to_dict()
        json.dumps(d)


# ---------------------------------------------------------------------------
# G19 (live smoke PST-1, 2026-10-08): built-in motions scaled to a target PGA
# ---------------------------------------------------------------------------

class TestTargetPGA:

    def test_builtin_motion_scaled_to_target(self):
        base, dt0 = validate_motion_input(motion="synthetic_pulse")
        scaled, dt1 = validate_motion_input(motion="synthetic_pulse",
                                            target_pga_g=0.2)
        assert dt1 == dt0
        assert np.max(np.abs(base)) == pytest.approx(0.30, rel=1e-3)
        assert np.max(np.abs(scaled)) == pytest.approx(0.20, rel=1e-12)
        # Linear scaling: same shape, every sample by 0.2 / PGA.
        np.testing.assert_allclose(
            scaled, base * (0.2 / np.max(np.abs(base))), rtol=1e-12)
        # The cached built-in record itself is not changed.
        again, _ = validate_motion_input(motion="synthetic_pulse")
        np.testing.assert_array_equal(again, base)

    def test_custom_record_scaled(self):
        acc = 0.05 * np.sin(np.linspace(0, 20, 400))
        out, dt = validate_motion_input(accel_history=acc, dt=0.01,
                                        target_pga_g=0.4)
        assert np.max(np.abs(out)) == pytest.approx(0.4)

    def test_bad_target_refused(self):
        for bad in (0.0, -0.1, float("nan")):
            with pytest.raises(ValueError, match="target_pga_g"):
                validate_motion_input(motion="synthetic_pulse",
                                      target_pga_g=bad)

    def test_label(self):
        from opensees_agent.ground_motions import motion_label
        assert motion_label("synthetic_pulse") == "synthetic_pulse"
        assert motion_label("synthetic_pulse", 0.2) == \
            "synthetic_pulse scaled to PGA 0.2 g"
        assert motion_label(None, None) == "custom"


# ===========================================================================
# GM-77 (2026-10-14): the column's OpenSees commands, checked without
# openseespy (Tier 1, so the Windows gate sees them too)
# ===========================================================================

SAND33 = {"thickness": 20.0, "Vs": 200.0, "density": 1.9,
          "material_type": "sand", "phi": 33.0}
CLAY60 = {"thickness": 15.0, "Vs": 150.0, "density": 1.7,
          "material_type": "clay", "su": 60.0}

PDMY02_ARG_NAMES = (
    "type", "tag", "nd", "rho", "refShearModul", "refBulkModul",
    "frictionAng", "peakShearStra", "refPress", "pressDependCoe", "PTAng",
    "contrac1", "contrac3", "dilat1", "dilat3", "noYieldSurf",
    "contrac2", "dilat2", "liquefac1", "liquefac2", "e",
    "cs1", "cs2", "cs3", "pa")

PIMY_ARG_NAMES = (
    "type", "tag", "nd", "rho", "refShearModul", "refBulkModul", "cohesi",
    "peakShearStra", "frictionAng", "refPress", "pressDependCoe",
    "noYieldSurf")


class _FakeOps:
    """Stands in for openseespy.opensees and records every call.

    ``fail_first`` names algorithms whose FIRST dynamic step attempt
    fails (each once); ``fail_first_full_step`` fails every algorithm on
    the first full dynamic step (dt = 0.01), so it is taken in substeps.
    """

    # Pore pressure at every node free to carry it: PP0 kPa after
    # gravity, PP0 + EXCESS once the shaking starts.
    PP0 = 10.0
    EXCESS = 30.0

    def __init__(self, fail_dynamic=False, fail_gravity=False,
                 fail_first=(), fail_first_full_step=False):
        self.calls = []
        self.time = 0.0
        self.shaking = False
        self.fail_dynamic = fail_dynamic
        self.fail_gravity = fail_gravity
        self.fail_first = set(fail_first)
        self.fail_first_full_step = fail_first_full_step
        self.full_steps_tried = 0
        self.algo = None

    def algorithm(self, *args):
        self.calls.append(("algorithm", args))
        self.algo = args[0]

    def analyze(self, n, dt):
        self.calls.append(("analyze", (n, dt)))
        gravity = dt >= 1.0
        self.shaking = self.shaking or not gravity
        if (gravity and self.fail_gravity) or (
                not gravity and self.fail_dynamic):
            return -3
        if not gravity and self.algo in self.fail_first:
            self.fail_first.discard(self.algo)
            return -3
        if not gravity and self.fail_first_full_step and dt == 0.01:
            self.full_steps_tried += 1
            if self.full_steps_tried <= 3:  # Newton, Krylov, modified
                return -3
        self.time += n * dt
        return 0

    def getTime(self):
        return self.time

    def setTime(self, t):
        self.calls.append(("setTime", (t,)))
        self.time = t

    def eleResponse(self, tag, what):
        # Vertical effective stress halves once the shaking starts
        s = 0.5 if self.shaking else 1.0
        if what == "stress":
            return [-50.0 * s, -100.0 * s, -50.0 * s, 0.0, 0.0]
        return [0.0, 0.0, 0.0]

    def nodeAccel(self, node, dof):
        return 0.0

    def nodeVel(self, node, dof):
        """DOF 3 is the pore pressure (kPa); 0 where it is fixed."""
        if dof != 3:
            return 0.0
        drained = {a[0] for a in self.named("fix")
                   if len(a) == 4 and a[3] == 1}
        if node in drained:
            return 0.0
        return self.PP0 + (self.EXCESS if self.shaking else 0.0)

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, args))
        return record

    def named(self, name):
        return [args for n, args in self.calls if n == name]

    def index(self, name, pred=lambda a: True, last=False):
        idx = [i for i, (n, a) in enumerate(self.calls)
               if n == name and pred(a)]
        return idx[-1] if last else idx[0]


def _build_fake(layers, linear_elastic=False, gwt_depth=0.0, accel=None,
                n_elem_per_layer=2, **fake_kw):
    from opensees_agent.site_response import _run_site_response_model
    ops = _FakeOps(**fake_kw)
    acc = (accel if accel is not None
           else 0.1 * np.sin(np.linspace(0.0, 6.0, 50)))
    result = _run_site_response_model(
        ops, layers=layers, accel_g=acc, dt_motion=0.01,
        gwt_depth=gwt_depth, bedrock_Vs=760.0, bedrock_density=2.4,
        damping=0.02, scale_factor=1.0, n_elem_per_layer=n_elem_per_layer,
        motion_name="test", linear_elastic=linear_elastic)
    return ops, result


class TestSiteResponseModelWiring:

    def _material(self, ops, kind, names):
        (args,) = [a for a in ops.named("nDMaterial") if a[0] == kind]
        assert len(args) == len(names)
        return dict(zip(names, args))

    def test_sand_material_arguments_in_documented_order(self):
        """Defect 2: PTAng was left out, so every later value sat one slot
        off (noYieldSurf read 0.27, e and the critical-state constants 0)."""
        ops, _ = _build_fake([SAND33])
        a = self._material(ops, "PressureDependMultiYield02",
                           PDMY02_ARG_NAMES)
        assert a["nd"] == 2
        assert a["refShearModul"] == pytest.approx(1.9 * 200.0 ** 2)
        assert a["frictionAng"] == 33.0
        assert a["peakShearStra"] == 0.1
        assert a["pressDependCoe"] == 0.5
        # OpenSees wiki table of suggested parameters, Dr = 40 % column
        assert a["PTAng"] == 26.0
        assert (a["contrac1"], a["contrac3"], a["dilat1"], a["dilat3"]) \
            == (0.067, 0.23, 0.06, 0.27)
        assert a["noYieldSurf"] == 20 and isinstance(a["noYieldSurf"], int)
        assert (a["contrac2"], a["dilat2"], a["liquefac1"],
                a["liquefac2"]) == (5.0, 3.0, 1.0, 0.0)
        assert a["e"] == 0.77
        assert (a["cs1"], a["cs2"], a["cs3"], a["pa"]) \
            == (0.9, 0.02, 0.7, 101.0)
        # refPress is the mean effective stress at mid-layer (10 m, water
        # at the surface): sigma'v = (1.9 - 1.0) * 9.81 * 10
        K0 = 1.0 - math.sin(math.radians(33.0))
        assert a["refPress"] == pytest.approx(
            0.9 * 9.81 * 10.0 * (1.0 + 2.0 * K0) / 3.0)

    def test_phase_transformation_angle_capped_at_phi(self):
        loose = dict(SAND33, phi=24.0)
        ops, _ = _build_fake([loose])
        a = self._material(ops, "PressureDependMultiYield02",
                           PDMY02_ARG_NAMES)
        assert a["PTAng"] == 24.0
        ops, _ = _build_fake([dict(SAND33, PTAng=28.0, e_init=0.6,
                                   n_surf=15)])
        a = self._material(ops, "PressureDependMultiYield02",
                           PDMY02_ARG_NAMES)
        assert (a["PTAng"], a["e"], a["noYieldSurf"]) == (28.0, 0.6, 15)

    def test_clay_strength_in_simple_shear_is_su(self):
        """PIMY (phi = 0, von Mises) peaks at 2c/sqrt(3) in simple shear,
        so c = sqrt(3)/2 * su (OpenSees wiki, PressureIndependMultiYield)."""
        ops, _ = _build_fake([CLAY60])
        a = self._material(ops, "PressureIndependMultiYield",
                           PIMY_ARG_NAMES)
        assert a["cohesi"] == pytest.approx(math.sqrt(3.0) / 2.0 * 60.0)
        assert 2.0 * a["cohesi"] / math.sqrt(3.0) == pytest.approx(60.0)
        assert (a["frictionAng"], a["pressDependCoe"], a["noYieldSurf"]) \
            == (0.0, 0.0, 20)

    def test_gravity_is_element_body_force_not_a_time_series(self):
        """The weight was nodal loads on a Linear series (factor = time):
        10,000 times the weight at t = 10,000 s."""
        ops, _ = _build_fake([SAND33])
        elements = [a for a in ops.named("element") if a[0] == "SSPquadUP"]
        assert len(elements) == 2
        for a in elements:
            assert a[-2:] == (0.0, -9.81)
        series = [a[0] for a in ops.named("timeSeries")]
        assert series == ["Path"]
        assert all(a[2] == 0.0 for a in ops.named("load"))  # no y loads

    def test_clock_set_to_zero_before_the_motion(self):
        """Defect 1: the gravity stages left the clock at 10,000 s, past
        the end of the motion's force history, so no motion was applied."""
        ops, _ = _build_fake([SAND33])
        gravity = [a for a in ops.named("analyze") if a[1] >= 1.0]
        assert gravity == [(1, 500.0)] * 50  # 10 elastic + 40 plastic
        last_gravity = ops.index("analyze", lambda a: a[1] >= 1.0,
                                 last=True)
        reset = ops.index("setTime")
        path = ops.index("timeSeries", lambda a: a[0] == "Path")
        first_dynamic = ops.index("analyze", lambda a: a[1] < 1.0)
        assert last_gravity < reset < path < first_dynamic
        assert ops.calls[reset][1] == (0.0,)
        stages = [a[3] for a in ops.named("updateMaterialStage")]
        assert stages == [0, 1]

    def test_input_force_is_dashpot_times_outcrop_velocity(self):
        """F = rho_r * Vs_r * A * v_outcrop (Joyner & Chen 1975): an
        outcrop motion is already twice the incident wave."""
        dt = 0.01
        acc = 0.2 * np.sin(2 * np.pi * 2.0 * np.arange(0, 2.0, dt))
        ops, _ = _build_fake([SAND33], accel=acc)
        (path,) = [a for a in ops.named("timeSeries") if a[0] == "Path"]
        i0 = path.index("-values") + 1
        i1 = path.index("-factor")
        force = np.array(path[i0:i1])
        a = acc * 9.81
        v = np.concatenate(([0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * dt)))
        np.testing.assert_allclose(force, 2.4 * 760.0 * 1.0 * v,
                                   rtol=1e-12, atol=1e-12)
        assert path[path.index("-dt") + 1] == dt

    def test_linear_elastic_stays_in_the_elastic_stage(self):
        ops, r = _build_fake([SAND33], linear_elastic=True)
        stages = [a[3] for a in ops.named("updateMaterialStage")]
        assert stages == [0]
        gravity = [a for a in ops.named("analyze") if a[1] >= 1.0]
        assert len(gravity) == 10
        assert r.analysis_type == "linear elastic"

    def test_water_table_on_a_node_is_drained(self):
        """A node AT the water table has zero pore pressure (it was left
        free, which put the water table one element too shallow)."""
        layer = dict(SAND33, thickness=10.0)  # nodes at 10, 5, 0 m depth
        ops, _ = _build_fake([layer], gwt_depth=5.0)
        drained = {a[0] for a in ops.named("fix") if a[1:] == (0, 0, 1)}
        # left column tags 1, 2, 3 from the base: depth 10, 5, 0
        assert drained == {2, 3, 5, 6}
        ops, _ = _build_fake([layer], gwt_depth=4.0)
        drained = {a[0] for a in ops.named("fix") if a[1:] == (0, 0, 1)}
        assert drained == {3, 6}

    def test_ru_is_excess_pore_pressure_over_initial_effective_stress(self):
        """GM-79: ru = (u - u0) / sigma'v0, u0 the pore pressure after
        gravity, at the element's centre (the mean of its four nodes). In
        the fake, u rises by 30 kPa at every free node and sigma'v0 is
        100 kPa, while sigma'v halves: the old ru, 1 - sigma'v / sigma'v0,
        read 0.5 from that swing in vertical stress."""
        layer = dict(SAND33, thickness=10.0)  # elements 0-5 m, 5-10 m
        # element order is from the base; the surface node is drained
        _, wet = _build_fake([layer], gwt_depth=0.0)
        np.testing.assert_allclose(wet.max_pore_pressure_ratio,
                                   [0.30, 0.15])
        _, dry = _build_fake([layer], gwt_depth=40.0)
        np.testing.assert_allclose(dry.max_pore_pressure_ratio, [0.0, 0.0])
        # water table on the 5 m node: 5-10 m drained at its top only,
        # 0-5 m no pore water
        _, half = _build_fake([layer], gwt_depth=5.0)
        np.testing.assert_allclose(half.max_pore_pressure_ratio, [0.15, 0.0])

    def test_ru_does_not_count_the_pore_pressure_after_gravity(self):
        """u0 (here 10 kPa) is the reference; a fall in pore pressure
        (dilation) gives no ru."""
        layer = dict(SAND33, thickness=10.0)
        ops = _FakeOps()
        ops.EXCESS = -5.0
        from opensees_agent.site_response import _run_site_response_model
        r = _run_site_response_model(
            ops, layers=[layer], accel_g=0.1 * np.sin(np.linspace(0, 6, 50)),
            dt_motion=0.01, gwt_depth=0.0, bedrock_Vs=760.0,
            bedrock_density=2.4, damping=0.02, scale_factor=1.0,
            n_elem_per_layer=2, motion_name="test")
        np.testing.assert_allclose(r.max_pore_pressure_ratio, [0.0, 0.0])

    def test_sigma_v_eff_hand_calculation(self):
        from opensees_agent.site_response import _sigma_v_eff
        layers = [dict(SAND33, thickness=5.0, density=1.8),
                  dict(CLAY60, thickness=10.0, density=1.6)]
        # 5 m of sand at 1.8 + 2.5 m of clay at 1.6, water at 2 m
        expected = 9.81 * (1.8 * 5.0 + 1.6 * 2.5) - 9.81 * (7.5 - 2.0)
        assert _sigma_v_eff(layers, 7.5, 2.0) == pytest.approx(expected)
        assert _sigma_v_eff(layers, 1.0, 2.0) == pytest.approx(
            9.81 * 1.8 * 1.0)

    def test_dynamic_step_failure_is_an_error_not_skipped(self):
        with pytest.raises(RuntimeError, match="did not converge at t ="):
            _build_fake([SAND33], fail_dynamic=True)

    def test_gravity_failure_is_an_error(self):
        with pytest.raises(RuntimeError, match="gravity stage did not"):
            _build_fake([SAND33], fail_gravity=True)

    @pytest.mark.parametrize("linear_elastic", [False, True])
    def test_gravity_at_1_m_per_s_then_each_layers_permeability(
            self, linear_elastic):
        """GM-85: gravity at the shaking permeability failed at the first
        elastoplastic step on finer sand meshes. As the OpenSees effective
        stress example: every element at 1 m/s through gravity, then each
        element's own permeability (sand 1e-4, clay 1e-7 m/s) set through
        element parameters, after the last gravity step and before the
        clock reset."""
        sand = dict(SAND33, thickness=10.0)
        ops, _ = _build_fake([sand, CLAY60], linear_elastic=linear_elastic)
        elements = [a for a in ops.named("element") if a[0] == "SSPquadUP"]
        assert len(elements) == 4
        # SSPquadUP tag n1 n2 n3 n4 mat t fBulk fDen perm1 perm2 ...
        assert all(a[10] == 1.0 and a[11] == 1.0 for a in elements)
        params = {a[0]: a[1:] for a in ops.named("parameter")}
        set_to = {}
        for tag, value in ops.named("updateParameter"):
            kind, ele, name = params[tag]
            assert kind == "element"
            set_to[(ele, name)] = value
        # built from the base: elements 1-2 clay, 3-4 sand
        expected = {1: 1.0e-7, 2: 1.0e-7, 3: 1.0e-4, 4: 1.0e-4}
        assert set_to == {(e, n): k for e, k in expected.items()
                          for n in ("hPerm", "vPerm")}
        last_gravity = ops.index("analyze", lambda a: a[1] >= 1.0,
                                 last=True)
        first_update = ops.index("updateParameter")
        last_update = ops.index("updateParameter", last=True)
        reset = ops.index("setTime")
        assert last_gravity < first_update <= last_update < reset

    def test_dynamic_stage_is_newton_at_the_examples_test(self):
        """GM-84: KrylovNewton at NormDispIncr 1e-3 m accepted unconverged
        steps on small elements; the dynamic stage now starts on Newton
        (McGann & Arduino's total stress example) at the same test."""
        ops, _ = _build_fake([SAND33])
        reset = ops.index("setTime")
        algos = [a for i, (n, a) in enumerate(ops.calls)
                 if n == "algorithm" and i > reset]
        tests = [a for i, (n, a) in enumerate(ops.calls)
                 if n == "test" and i > reset]
        assert algos == [("Newton",)]
        assert tests == [("NormDispIncr", 1.0e-3, 35, 0)]

    def test_a_step_newton_cannot_take_falls_back_then_newton_returns(self):
        """Newton, then KrylovNewton, then modified Newton, the fallbacks
        at a test 100 times tighter (1e-5 m: linear convergence passes a
        loose test early); Newton and its test restored for the next
        step."""
        ops, r = _build_fake([SAND33],
                             fail_first=("Newton", "KrylovNewton"))
        first_dyn = ops.index("analyze", lambda a: a[1] < 1.0)
        seq = [(n, a) for n, a in ops.calls[first_dyn:]
               if n in ("analyze", "algorithm", "test")][:10]
        assert seq == [
            ("analyze", (1, 0.01)),                      # Newton: fails
            ("test", ("NormDispIncr", 1.0e-5, 50, 0)),
            ("algorithm", ("KrylovNewton",)),
            ("analyze", (1, 0.01)),                      # fails
            ("test", ("NormDispIncr", 1.0e-5, 50, 0)),
            ("algorithm", ("ModifiedNewton",)),
            ("analyze", (1, 0.01)),                      # converges
            ("test", ("NormDispIncr", 1.0e-3, 35, 0)),
            ("algorithm", ("Newton",)),
            ("analyze", (1, 0.01)),                      # next step
        ]
        assert len(r.surface_accel_g) == 50

    def test_a_step_no_algorithm_can_take_is_taken_in_substeps(self):
        """Every algorithm fails the whole step: ten substeps of dt/10 to
        the same time, each tried with the algorithms in turn, and the
        record still lines up with the input."""
        ops, r = _build_fake([SAND33], fail_first_full_step=True)
        dyn = [a[1] for a in ops.named("analyze") if a[1] < 1.0]
        assert dyn[:3] == [0.01, 0.01, 0.01]
        subs = [dt for dt in dyn if dt < 0.01]
        assert all(dt <= 0.001 + 1e-15 for dt in subs)
        assert sum(subs) == pytest.approx(0.01, abs=1e-12)
        assert len([dt for dt in dyn if dt == 0.01]) == 3 + 48
        assert len(r.surface_accel_g) == 50

    def test_surface_record_lines_up_with_the_input(self):
        acc = 0.1 * np.sin(np.linspace(0.0, 6.0, 50))
        _, r = _build_fake([SAND33], accel=acc)
        assert len(r.time) == len(r.surface_accel_g) == len(acc)
        assert r.time[0] == 0.0 and r.surface_accel_g[0] == 0.0
        assert r.site_period_s == pytest.approx(4 * 20.0 / 200.0)

    def test_result_dict_names_the_analysis(self):
        _, r = _build_fake([SAND33])
        d = r.to_dict()
        assert d["analysis_type"] == "nonlinear effective stress"
        assert d["site_period_s"] == pytest.approx(0.4)
        json.dumps(d)


class TestSiteResponseTimeLimit:
    """A stuck OpenSees step holds the GIL and cannot be interrupted, so
    the analysis runs in a child process stopped at time_limit_s."""

    def test_bad_time_limit_refused(self):
        from opensees_agent.site_response import analyze_site_response
        for bad in (0, -5.0, float("nan"), "60"):
            with pytest.raises(ValueError, match="time_limit_s"):
                analyze_site_response([SAND33], motion="synthetic_pulse",
                                      time_limit_s=bad)

    def test_child_stopped_at_the_limit(self):
        from opensees_agent.site_response import _run_job_in_child
        with pytest.raises(RuntimeError, match="did not finish within"):
            _run_job_in_child({"layers": []}, time_limit_s=0.001)

    @requires_opensees
    def test_child_error_reaches_the_caller(self):
        from opensees_agent.site_response import _run_job_in_child
        with pytest.raises(RuntimeError, match="unexpected keyword"):
            _run_job_in_child({"bogus": 1}, time_limit_s=TEST_TIME_LIMIT_S)


# ===========================================================================
# GM-77 benchmarks (Tier 2): the column against closed-form and
# independent solutions
# ===========================================================================

def _kramer_transfer(f, H, Vs, rho, Vs_rock, rho_rock, xi):
    """|surface / rock outcrop| of a uniform damped layer on elastic rock.

    Kramer (1996), Geotechnical Earthquake Engineering, Sec. 7.2.1:
    F(w) = 1 / (cos(k* H) + i alpha* sin(k* H)), k* = w / Vs*,
    Vs* = Vs sqrt(1 + 2 i xi), alpha* = rho Vs* / (rho_r Vs_r).
    At the site frequency Vs/4H this is about 1 / (alpha + pi xi / 2).
    """
    Vs_c = Vs * np.sqrt(1.0 + 2.0j * xi)
    k = 2.0 * np.pi * f / Vs_c
    alpha = rho * Vs_c / (rho_rock * Vs_rock)
    return float(np.abs(1.0 / (np.cos(k * H) + 1j * alpha * np.sin(k * H))))


def _rayleigh_xi(f, f1, xi):
    """Damping ratio of the module's Rayleigh damping (f1 and 5 f1)."""
    w1, w2, w = 2 * np.pi * f1, 2 * np.pi * 5 * f1, 2 * np.pi * f
    a0 = 2 * xi * w1 * w2 / (w1 + w2)
    a1 = 2 * xi / (w1 + w2)
    return a0 / (2 * w) + a1 * w / 2


def _ramped_sine(f, amp_g=0.01, duration=12.0, dt=0.005, ramp=2.0):
    t = np.arange(0.0, duration, dt)
    return amp_g * np.sin(2 * np.pi * f * t) * np.minimum(1.0, t / ramp), dt


def _hyperbolic_masing(gamma_ref, xi_min, strains):
    """G/Gmax and Masing damping of a hyperbolic backbone (Hardin &
    Drnevich), the backbone PIMY builds its yield surfaces on."""
    x = strains / gamma_ref
    ggmax = 1.0 / (1.0 + x)
    damping = ((4.0 / np.pi) * (1.0 + 1.0 / x)
               * (1.0 - np.log(1.0 + x) / x) - 2.0 / np.pi)
    return ggmax, damping + xi_min


class TestSiteResponseBenchmarks:
    """The column checked against solutions it does not share code with.

    Linear elastic (``linear_elastic=True``): the closed-form transfer
    function of a uniform damped layer on elastic rock (Kramer 1996,
    Sec. 7.2; the check Kwok et al. 2007 use for nonlinear codes) and
    pystrata's frequency-domain linear-elastic run of the same column.
    These pin the input convention, the clock, the dashpot, the mass and
    the modulus to a few per cent.

    Nonlinear: the same clay column run equivalent-linear in pystrata with
    the backbone PIMY uses (hyperbolic, reference strain su/Gmax, Masing
    damping). The two agree at small strain and part as strain grows: EQL
    applies one strain-compatible modulus and damping to the whole record
    and has no shear strength, while the nonlinear model follows the
    backbone cycle by cycle and cannot pass more shear stress than su, so
    it amplifies less in strong shaking (Kramer 1996, Sec. 7.2-7.3).
    """

    H, VS, RHO = 20.0, 200.0, 1.9
    VS_ROCK, RHO_ROCK, XI = 760.0, 2.4, 0.02

    def _layer(self):
        return {"thickness": self.H, "Vs": self.VS, "density": self.RHO,
                "material_type": "sand", "phi": 33.0}

    @requires_opensees
    def test_rock_column_reproduces_the_outcrop_motion(self):
        """A column with the rock's own impedance and no damping passes
        the outcrop motion to the surface unchanged (|F| = 1). The old
        doubled input force gave 2.0, the old clock 0."""
        from opensees_agent.site_response import analyze_site_response
        r = analyze_site_response(
            [{"thickness": 20.0, "Vs": 760.0, "density": 2.4,
              "material_type": "sand", "phi": 33.0}],
            motion="synthetic_pulse", bedrock_Vs=760.0,
            bedrock_density=2.4, damping=0.0, n_elem_per_layer=4,
            linear_elastic=True, time_limit_s=TEST_TIME_LIMIT_S)
        assert r.amplification_factor == pytest.approx(1.0, abs=0.01)
        # The motion reaches the surface within the travel time H/Vs
        first = r.time[np.argmax(np.abs(r.surface_accel_g) > 1e-4)]
        assert first < 0.1

    @requires_opensees
    @pytest.mark.parametrize("f_ratio", [0.5, 1.0, 2.0, 3.0])
    def test_linear_transfer_function_matches_closed_form(self, f_ratio):
        """Steady-state surface/outcrop amplitude of a sine at f_ratio x
        the site frequency (2.5 Hz) against Kramer's closed form, within
        3 % (measured: -1.1 % at 0.5, +0.2 % at 1, 0.0 % at 2, -1.6 % at
        3 with 10 elements; the third-mode error is the mesh). At
        resonance the closed form is 4.17."""
        from opensees_agent.site_response import analyze_site_response
        f1 = self.VS / (4.0 * self.H)
        f = f_ratio * f1
        acc, dt = _ramped_sine(f)
        r = analyze_site_response(
            [self._layer()], accel_history=acc, dt=dt, damping=self.XI,
            bedrock_Vs=self.VS_ROCK, bedrock_density=self.RHO_ROCK,
            n_elem_per_layer=10, linear_elastic=True,
            time_limit_s=TEST_TIME_LIMIT_S)
        steady = r.time > r.time[-1] - 3.0
        fe = np.max(np.abs(r.surface_accel_g[steady])) / 0.01
        closed = _kramer_transfer(f, self.H, self.VS, self.RHO,
                                  self.VS_ROCK, self.RHO_ROCK,
                                  _rayleigh_xi(f, f1, self.XI))
        assert fe == pytest.approx(closed, rel=0.03)
        if f_ratio == 1.0:
            assert closed == pytest.approx(4.17, abs=0.01)

    @requires_opensees
    def test_linear_matches_pystrata_linear_elastic(self):
        """Time domain (OpenSees, Rayleigh damping) against frequency
        domain (pystrata, constant 2 % damping) for the same column and
        motion: PGA within 5 % (measured 1.3 %), spectral accelerations
        from 0.1 to 1 s within 6 % (measured at most 3.3 %)."""
        pystrata_agent = pytest.importorskip("pystrata_agent")
        if not pystrata_agent.has_pystrata():
            pytest.skip("pystrata not installed")
        from opensees_agent.site_response import analyze_site_response
        os_r = analyze_site_response(
            [self._layer()], motion="synthetic_pulse", damping=self.XI,
            bedrock_Vs=self.VS_ROCK, bedrock_density=self.RHO_ROCK,
            n_elem_per_layer=10, linear_elastic=True,
            time_limit_s=TEST_TIME_LIMIT_S)
        ps_r = pystrata_agent.analyze_linear_site_response(
            [{"thickness": self.H, "Vs": self.VS,
              "unit_wt": self.RHO * 9.81, "soil_model": "linear",
              "damping": self.XI},
             {"thickness": 0, "Vs": self.VS_ROCK,
              "unit_wt": self.RHO_ROCK * 9.81, "soil_model": "linear",
              "damping": 0.0}],
            motion="synthetic_pulse")
        assert os_r.pga_surface_g == pytest.approx(ps_r.pga_surface_g,
                                                   rel=0.05)
        for T in (0.1, 0.2, 0.4, 0.5, 1.0):
            sa_os = np.interp(T, os_r.periods, os_r.Sa_surface_g)
            sa_ps = np.interp(T, ps_r.periods, ps_r.Sa_surface_g)
            assert sa_os == pytest.approx(sa_ps, rel=0.06), T

    @requires_opensees
    def test_gravity_stresses_match_hand_calculation(self):
        """Sand over clay over sand, water at the sand/clay boundary:
        vertical effective stress after the gravity stages within 0.1 % of
        the weight of the layers above less hydrostatic pore pressure
        (measured within 0.0003 % with gravity at 1 m/s, GM-85; 0.04 %
        when gravity ran at the shaking permeability, the pore pressures
        not quite settled). Before GM-77 it was 10,000 times."""
        from opensees_agent.site_response import (
            analyze_site_response, _sigma_v_eff)
        layers = [
            {"thickness": 5.0, "Vs": 150.0, "density": 1.8,
             "material_type": "sand", "phi": 30.0},
            {"thickness": 10.0, "Vs": 120.0, "density": 1.6,
             "material_type": "clay", "su": 40.0},
            {"thickness": 10.0, "Vs": 250.0, "density": 1.9,
             "material_type": "sand", "phi": 35.0},
        ]
        r = analyze_site_response(layers, motion="synthetic_pulse",
                                  target_pga_g=0.05, gwt_depth=5.0,
                                  n_elem_per_layer=2,
                                  time_limit_s=TEST_TIME_LIMIT_S)
        hand = np.array([_sigma_v_eff(layers, d, 5.0) for d in r.depths])
        np.testing.assert_allclose(r.initial_sigma_v_eff_kPa, hand,
                                   rtol=0.001)

    @requires_opensees
    @pytest.mark.parametrize("gwt_depth, n_elem", [(0.0, 40), (2.0, 20)])
    def test_finer_sand_meshes_finish_gravity(self, gwt_depth, n_elem):
        """GM-85: 20 m of sand at 0.5 m elements, and with water 2 m down
        at 1 m elements, failed at the first elastoplastic gravity step
        while gravity ran at the shaking permeability (1e-4 m/s). Gravity
        at 1 m/s, then 1e-4 for the shaking (the OpenSees effective stress
        example's procedure): sigma'v within 0.1 % of the hand value
        (measured within 0.0005 %), and the sand liquefies at 0.30 g as
        the 4-element column does (max ru 1.0 and 0.996; the 4-element
        column left at 1 m/s for the shaking stays drained, max ru 0.22)."""
        from opensees_agent.site_response import (
            analyze_site_response, _sigma_v_eff)
        r = analyze_site_response([SAND33], motion="synthetic_pulse",
                                  gwt_depth=gwt_depth,
                                  n_elem_per_layer=n_elem,
                                  time_limit_s=TEST_TIME_LIMIT_S)
        hand = np.array([_sigma_v_eff([SAND33], d, gwt_depth)
                         for d in r.depths])
        np.testing.assert_allclose(r.initial_sigma_v_eff_kPa, hand,
                                   rtol=0.001)
        assert np.all(np.isfinite(r.surface_accel_g))
        assert r.pga_surface_g > 0.1 * r.pga_input_g
        assert np.max(r.max_pore_pressure_ratio) > 0.9

    def _clay_run(self, pga, linear_elastic=False):
        from opensees_agent.site_response import analyze_site_response
        return analyze_site_response(
            [CLAY60], motion="synthetic_pulse", target_pga_g=pga,
            damping=self.XI, n_elem_per_layer=6,
            linear_elastic=linear_elastic, time_limit_s=TEST_TIME_LIMIT_S)

    @requires_opensees
    def test_nonlinear_small_strain_approaches_linear(self):
        """At 0.01 g (max strain about 0.01 %) the nonlinear clay column
        amplifies within 10 % of the linear elastic one (measured 2.62
        against 2.81: -7 %, the hysteretic damping of the first yield
        surfaces)."""
        lin = self._clay_run(0.01, linear_elastic=True)
        nl = self._clay_run(0.01)
        assert np.max(nl.max_strain_pct) < 0.05
        assert nl.amplification_factor == pytest.approx(
            lin.amplification_factor, rel=0.10)

    @requires_opensees
    def test_nonlinear_against_equivalent_linear(self):
        """Clay H = 15 m, Vs 150 m/s, su 60 kPa, synthetic pulse. Measured
        amplification (nonlinear / EQL with the same hyperbolic backbone):
        0.05 g 2.78 / 3.17 (max strain about 0.1 %), 0.30 g 0.96 / 1.40
        (about 0.5 %). Pinned: within 20 % at 0.05 g; at 0.30 g the
        nonlinear column amplifies less than EQL (its stress is capped at
        su) and less than half what it does at 0.01 g."""
        pystrata_agent = pytest.importorskip("pystrata_agent")
        if not pystrata_agent.has_pystrata():
            pytest.skip("pystrata not installed")
        G = CLAY60["density"] * CLAY60["Vs"] ** 2
        strains = np.logspace(-6, -1, 30)
        ggmax, damping = _hyperbolic_masing(CLAY60["su"] / G, self.XI,
                                            strains)
        eql_layers = [
            {"thickness": CLAY60["thickness"], "Vs": CLAY60["Vs"],
             "unit_wt": CLAY60["density"] * 9.81, "soil_model": "custom",
             "strains": strains.tolist(), "mod_reduc": ggmax.tolist(),
             "damping_values": damping.tolist()},
            {"thickness": 0, "Vs": self.VS_ROCK,
             "unit_wt": self.RHO_ROCK * 9.81, "soil_model": "linear",
             "damping": 0.0},
        ]
        amp = {}
        for pga in (0.01, 0.05, 0.30):
            nl = self._clay_run(pga)
            eql = pystrata_agent.analyze_eql_site_response(
                eql_layers, motion="synthetic_pulse", target_pga_g=pga)
            amp[pga] = (nl.amplification_factor, eql.amplification_factor)
        nl05, eql05 = amp[0.05]
        assert nl05 == pytest.approx(eql05, rel=0.20)
        nl30, eql30 = amp[0.30]
        assert nl30 < eql30
        assert nl30 < 0.5 * amp[0.01][0]

    @requires_opensees
    def test_sand_over_clay_finishes(self):
        """Defect 3: sand over clay hung in the first elastoplastic gravity
        step, with or without water (the shifted sand arguments and the
        10,000-times weight); now about 2 s."""
        from opensees_agent.site_response import analyze_site_response
        layers = [
            {"thickness": 5.0, "Vs": 150.0, "density": 1.8,
             "material_type": "sand", "phi": 30.0},
            {"thickness": 10.0, "Vs": 120.0, "density": 1.6,
             "material_type": "clay", "su": 40.0},
        ]
        for gwt in (2.0, 50.0):
            r = analyze_site_response(layers, motion="synthetic_pulse",
                                      gwt_depth=gwt, n_elem_per_layer=2,
                                      time_limit_s=TEST_TIME_LIMIT_S)
            assert r.pga_surface_g > 0.1 * r.pga_input_g
            assert np.all(np.isfinite(r.surface_accel_g))
        # the dry run (water below the column) reports no pore pressure
        assert np.all(r.max_pore_pressure_ratio == 0.0)


class _PorePressureSpy:
    """Wraps openseespy and reads every node's pore pressure (DOF 3
    "velocity") and height when the clock is set back after gravity."""

    def __init__(self, ops):
        self._ops = ops
        self.after_gravity = None

    def __getattr__(self, name):
        return getattr(self._ops, name)

    def setTime(self, t):
        # the left column's nodes; the last tag is the dashpot's node
        self.after_gravity = [
            (self._ops.nodeCoord(n)[1], self._ops.nodeVel(n, 3))
            for n in sorted(self._ops.getNodeTags())[:-1]
            if self._ops.nodeCoord(n)[0] == 0.0]
        return self._ops.setTime(t)


class TestPorePressureRatio:
    """GM-79: ru is the excess pore pressure over the initial vertical
    effective stress, read off the pore pressure the model carries."""

    @requires_opensees
    def test_the_pore_pressure_read_is_hydrostatic_after_gravity(self):
        """What the module reads as pore pressure (the "velocity" of DOF
        3, as the OpenSees effective stress example records it) is, after
        gravity, 9.81 kPa per metre below the water table and 0 at and
        above it: hydrostatic by definition."""
        from opensees_agent.opensees_utils import fresh_model
        from opensees_agent.site_response import _run_site_response_model
        layer = dict(SAND33, thickness=10.0)
        spy = _PorePressureSpy(fresh_model(ndm=2, ndf=3))
        try:
            _run_site_response_model(
                spy, layers=[layer],
                accel_g=0.01 * np.sin(np.linspace(0.0, 6.0, 20)),
                dt_motion=0.01, gwt_depth=2.5, bedrock_Vs=760.0,
                bedrock_density=2.4, damping=0.02, scale_factor=1.0,
                n_elem_per_layer=4, motion_name="test")
        finally:
            spy.wipe()
        heights, pp = np.array(spy.after_gravity).T
        hydrostatic = 9.81 * np.maximum(0.0, (10.0 - heights) - 2.5)
        assert len(pp) == 5
        np.testing.assert_allclose(pp, hydrostatic, rtol=1e-3, atol=1e-6)
