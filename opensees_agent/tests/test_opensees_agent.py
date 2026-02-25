"""
Tests for the opensees_agent module.

Tier 1 tests (no openseespy required):
  - Result dataclasses: construction, summary(), to_dict(), plot_*()
  - Input validation for analyze_pm4sand_dss and analyze_bnwf_pile
  - Ground motions: list, get, validation
  - Utility functions: clean_numpy, compute_response_spectrum, has_opensees
  - Foundry agent: list_methods, describe_method, error handling
  - BNWF pile: input validation, layer parsing, Foundry metadata

Tier 2 tests (require openseespy, skipped if not installed):
  - PM4Sand DSS integration
  - BNWF lateral pile integration
"""

import json
import math

import numpy as np
import pytest

from opensees_agent.results import (
    PM4SandDSSResult, BNWFPileResult, SiteResponseResult,
)
from opensees_agent.opensees_utils import (
    has_opensees, clean_numpy, compute_response_spectrum,
)
from opensees_agent.ground_motions import (
    get_motion, list_motions, validate_motion_input,
)
from opensees_agent.pm4sand_dss import _validate_pm4sand_inputs
from opensees_agent.bnwf_pile import _validate_bnwf_inputs, _build_py_model
from opensees_agent.site_response import _validate_site_response_inputs

# Skip marker for tests that need openseespy
requires_opensees = pytest.mark.skipif(
    not has_opensees(), reason="openseespy not installed"
)


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
# BNWFPileResult
# ===========================================================================

class TestBNWFPileResult:
    """Tests for BNWFPileResult dataclass."""

    def _make_result(self):
        n = 50
        z = np.linspace(0, 20, n)
        return BNWFPileResult(
            pile_length=20.0, pile_diameter=0.6,
            lateral_force_kN=200.0, moment_kNm=0.0, axial_force_kN=500.0,
            z=z,
            deflection_m=np.exp(-z / 5) * 0.01,
            moment_profile_kNm=np.sin(z * np.pi / 20) * 500,
            shear_profile_kN=np.cos(z * np.pi / 20) * 200,
            soil_reaction_kN_per_m=np.exp(-z / 3) * 50,
            y_top_m=0.01, rotation_top_rad=0.002,
            max_moment_kNm=500.0, max_moment_depth_m=5.0,
            max_deflection_m=0.01, converged=True,
        )

    def test_construction_defaults(self):
        r = BNWFPileResult()
        assert r.pile_length == 0.0
        assert r.converged is True
        assert len(r.z) == 0

    def test_summary(self):
        r = self._make_result()
        s = r.summary()
        assert "BNWF" in s
        assert "20.0" in s
        assert "YES" in s

    def test_to_dict(self):
        r = self._make_result()
        d = r.to_dict()
        assert d["pile_length_m"] == 20.0
        assert d["y_top_mm"] == 10.0
        assert d["converged"] is True
        json.dumps(d)  # serializable


class TestBNWFPileResultPlots:
    """Plot smoke tests for BNWFPileResult."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        import matplotlib
        matplotlib.use("Agg")

    def _make_result(self):
        n = 30
        z = np.linspace(0, 20, n)
        return BNWFPileResult(
            pile_length=20.0, pile_diameter=0.6,
            lateral_force_kN=200.0, moment_kNm=0.0, axial_force_kN=0.0,
            z=z,
            deflection_m=np.exp(-z / 5) * 0.01,
            moment_profile_kNm=np.sin(z * np.pi / 20) * 500,
            shear_profile_kN=np.cos(z * np.pi / 20) * 200,
            soil_reaction_kN_per_m=np.zeros(n),
            y_top_m=0.01, rotation_top_rad=0.002,
            max_moment_kNm=500.0, max_moment_depth_m=5.0,
            max_deflection_m=0.01,
        )

    def test_plot_deflection(self):
        import matplotlib
        ax = self._make_result().plot_deflection(show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_moment(self):
        import matplotlib
        ax = self._make_result().plot_moment(show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_shear(self):
        import matplotlib
        ax = self._make_result().plot_shear(show=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_plot_all(self):
        import matplotlib
        fig, axes = self._make_result().plot_all(show=False)
        assert isinstance(fig, matplotlib.figure.Figure)
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

    def test_list_methods_includes_bnwf(self):
        import foundry.opensees_agent_foundry as oaf
        result = json.loads(oaf.opensees_list_methods.__wrapped__(""))
        assert "Lateral Pile" in result
        assert "bnwf_lateral_pile" in result["Lateral Pile"]

    def test_describe_bnwf(self):
        import foundry.opensees_agent_foundry as oaf
        result = json.loads(oaf.opensees_describe_method.__wrapped__(
            "bnwf_lateral_pile"))
        assert "category" in result
        assert result["category"] == "Lateral Pile"
        assert "parameters" in result
        assert "pile_length" in result["parameters"]
        assert "layers" in result["parameters"]

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
# Input validation (BNWF Pile)
# ===========================================================================

def _good_layer():
    """Return a valid sand layer dict for testing."""
    return {
        "top": 0.0, "bottom": 15.0,
        "py_model": "api_sand",
        "phi": 35.0, "gamma": 10.0, "k": 16000.0,
    }


class TestBNWFPileValidation:
    """Input validation for analyze_bnwf_pile."""

    def _call(self, **overrides):
        defaults = dict(
            pile_length=15.0, pile_diameter=0.6,
            wall_thickness=0.0125, E_pile=200e6,
            layers=[_good_layer()],
            lateral_load=100.0, moment=0.0, axial_load=0.0,
            head_condition='free', pile_above_ground=0.0,
            n_elem_per_meter=5,
        )
        defaults.update(overrides)
        _validate_bnwf_inputs(**defaults)

    def test_valid_inputs(self):
        self._call()  # should not raise

    def test_pile_length_zero(self):
        with pytest.raises(ValueError, match="pile_length must be positive"):
            self._call(pile_length=0)

    def test_pile_diameter_negative(self):
        with pytest.raises(ValueError, match="pile_diameter must be positive"):
            self._call(pile_diameter=-0.5)

    def test_wall_thickness_too_large(self):
        with pytest.raises(ValueError, match="wall_thickness.*must be <"):
            self._call(pile_diameter=0.6, wall_thickness=0.4)

    def test_E_pile_zero(self):
        with pytest.raises(ValueError, match="E_pile must be positive"):
            self._call(E_pile=0)

    def test_empty_layers(self):
        with pytest.raises(ValueError, match="non-empty list"):
            self._call(layers=[])

    def test_no_load(self):
        with pytest.raises(ValueError, match="At least one of lateral_load"):
            self._call(lateral_load=0, moment=0)

    def test_invalid_head_condition(self):
        with pytest.raises(ValueError, match="head_condition must be"):
            self._call(head_condition='pinned')

    def test_layer_missing_top(self):
        bad = {"bottom": 5.0, "py_model": "api_sand", "phi": 35, "gamma": 10, "k": 16000}
        with pytest.raises(ValueError, match="missing required key 'top'"):
            self._call(layers=[bad])

    def test_layer_bad_py_model(self):
        bad = _good_layer()
        bad["py_model"] = "bogus_model"
        with pytest.raises(ValueError, match="not recognized"):
            self._call(layers=[bad])

    def test_layer_bottom_le_top(self):
        bad = _good_layer()
        bad["top"] = 10.0
        bad["bottom"] = 5.0
        with pytest.raises(ValueError, match="must be > 'top'"):
            self._call(layers=[bad])

    def test_pile_above_ground_negative(self):
        with pytest.raises(ValueError, match="pile_above_ground must be >= 0"):
            self._call(pile_above_ground=-1.0)


class TestBNWFBuildPyModel:
    """Tests for _build_py_model layer parsing."""

    def test_api_sand(self):
        layer = _good_layer()
        model, name = _build_py_model(layer)
        assert name == "api_sand"
        from lateral_pile.py_curves import SandAPI
        assert isinstance(model, SandAPI)

    def test_matlock_with_c(self):
        layer = {
            "top": 0, "bottom": 5,
            "py_model": "matlock",
            "c": 50.0, "gamma": 9.0, "eps50": 0.01,
        }
        model, name = _build_py_model(layer)
        assert name == "matlock"

    def test_matlock_with_su_alias(self):
        """'su' should be accepted as alias for 'c' in Matlock model."""
        layer = {
            "top": 0, "bottom": 5,
            "py_model": "matlock",
            "su": 50.0, "gamma": 9.0, "eps50": 0.01,
        }
        model, name = _build_py_model(layer)
        assert name == "matlock"
        assert model.c == 50.0

    def test_jeanjean(self):
        layer = {
            "top": 0, "bottom": 10,
            "py_model": "jeanjean",
            "su": 30.0, "gamma": 8.0, "Gmax": 12000.0,
        }
        model, name = _build_py_model(layer)
        assert name == "jeanjean"

    def test_case_insensitive(self):
        layer = _good_layer()
        layer["py_model"] = "API_Sand"
        model, name = _build_py_model(layer)
        assert name == "api_sand"

    def test_weak_rock(self):
        layer = {
            "top": 0, "bottom": 5,
            "py_model": "weak_rock",
            "qu": 500.0, "Er": 100000.0,
        }
        model, name = _build_py_model(layer)
        assert name == "weak_rock"


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


class TestBNWFPileIntegration:
    """Integration tests for BNWF lateral pile (require openseespy)."""

    @requires_opensees
    def test_basic_sand_pile(self):
        """Free-head pile in uniform sand — basic structure check."""
        from opensees_agent.bnwf_pile import analyze_bnwf_pile

        result = analyze_bnwf_pile(
            pile_length=15.0,
            pile_diameter=0.6,
            wall_thickness=0.0125,
            E_pile=200e6,
            layers=[{
                "top": 0.0, "bottom": 15.0,
                "py_model": "api_sand",
                "phi": 35.0, "gamma": 10.0, "k": 16000.0,
            }],
            lateral_load=100.0,
        )
        assert isinstance(result, BNWFPileResult)
        assert result.converged
        assert result.y_top_m > 0  # pile deflects in load direction
        assert result.max_moment_kNm > 0
        assert len(result.z) > 10
        assert len(result.deflection_m) == len(result.z)

    @requires_opensees
    def test_fixed_head_less_deflection(self):
        """Fixed-head pile should deflect less than free-head."""
        from opensees_agent.bnwf_pile import analyze_bnwf_pile

        common = dict(
            pile_length=15.0, pile_diameter=0.6,
            wall_thickness=0.0125, E_pile=200e6,
            layers=[{
                "top": 0.0, "bottom": 15.0,
                "py_model": "api_sand",
                "phi": 35.0, "gamma": 10.0, "k": 16000.0,
            }],
            lateral_load=100.0,
        )
        free = analyze_bnwf_pile(**common, head_condition='free')
        fixed = analyze_bnwf_pile(**common, head_condition='fixed')
        assert fixed.y_top_m < free.y_top_m

    @requires_opensees
    def test_multilayer_clay_sand(self):
        """Multi-layer profile (clay over sand) runs and converges."""
        from opensees_agent.bnwf_pile import analyze_bnwf_pile

        result = analyze_bnwf_pile(
            pile_length=20.0,
            pile_diameter=0.9,
            wall_thickness=0.0,  # solid
            E_pile=30e6,  # concrete
            layers=[
                {
                    "top": 0.0, "bottom": 8.0,
                    "py_model": "matlock",
                    "c": 50.0, "gamma": 9.0, "eps50": 0.01,
                },
                {
                    "top": 8.0, "bottom": 20.0,
                    "py_model": "api_sand",
                    "phi": 33.0, "gamma": 10.0, "k": 11000.0,
                },
            ],
            lateral_load=200.0,
        )
        assert result.converged
        assert result.y_top_m > 0
        d = result.to_dict()
        json.dumps(d)  # serializable


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
        )
        assert isinstance(result, SiteResponseResult)
        assert result.total_depth_m == 20.0
        assert result.n_layers == 1
        assert result.pga_input_g > 0
        assert result.pga_surface_g > 0
        assert result.amplification_factor > 0
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
        assert np.all(result.max_pore_pressure_ratio <= 1.0)

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
        )
        assert isinstance(result, SiteResponseResult)
        assert result.n_layers == 1
        assert result.pga_surface_g > 0
        assert len(result.depths) == 3

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
        )
        assert result.total_depth_m == pytest.approx(25.0)
        assert result.n_layers == 3
        assert len(result.depths) == 6  # 3 layers * 2 elements
        assert result.pga_surface_g > 0
        d = result.to_dict()
        json.dumps(d)
