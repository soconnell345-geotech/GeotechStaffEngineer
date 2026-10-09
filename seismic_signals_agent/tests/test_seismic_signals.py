"""
Tests for seismic_signals_agent module.

Tier 1: No eqsig/pyrotd required (result dataclasses, validation, metadata).
Tier 2: Requires eqsig or pyrotd (integration tests).
"""

import json
import math

import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from seismic_signals_agent.results import (
    ResponseSpectrumResult,
    IntensityMeasuresResult,
    RotDSpectrumResult,
    SignalProcessingResult,
)
from seismic_signals_agent.signal_utils import has_eqsig, has_pyrotd

requires_eqsig = pytest.mark.skipif(not has_eqsig(), reason="eqsig not installed")
requires_pyrotd = pytest.mark.skipif(not has_pyrotd(), reason="pyrotd not installed")


# ---------------------------------------------------------------------------
# Helper: synthetic data for Tier 1 result tests
# ---------------------------------------------------------------------------

def _make_response_spectrum_result():
    """Create a ResponseSpectrumResult with synthetic data."""
    periods = np.logspace(-2, 1, 50)
    return ResponseSpectrumResult(
        motion_name="test_motion",
        n_points=2000,
        duration_s=20.0,
        dt_s=0.01,
        pga_g=0.30,
        pgv_m_per_s=0.15,
        pgd_m=0.005,
        damping=0.05,
        periods=periods,
        Sa_g=0.3 * np.exp(-0.5 * ((np.log10(periods) - np.log10(0.3)) / 0.5) ** 2),
        time=np.arange(2000) * 0.01,
        accel_g=0.3 * np.sin(2 * np.pi * 2.0 * np.arange(2000) * 0.01),
    )


def _make_intensity_measures_result():
    """Create an IntensityMeasuresResult with synthetic data."""
    time = np.arange(2000) * 0.01
    return IntensityMeasuresResult(
        motion_name="test_motion",
        n_points=2000,
        duration_s=20.0,
        dt_s=0.01,
        pga_g=0.30,
        pgv_m_per_s=0.15,
        pgd_m=0.005,
        arias_intensity_m_per_s=1.5,
        significant_duration_s=8.5,
        sig_dur_start=0.05,
        sig_dur_end=0.95,
        cav_m_per_s=5.0,
        bracketed_duration_s=12.0,
        arias_cumulative=np.linspace(0, 1.5, 2000),
        time=time,
        accel_g=0.3 * np.sin(2 * np.pi * 2.0 * time),
    )


def _make_rotd_spectrum_result():
    """Create a RotDSpectrumResult with synthetic data."""
    periods = np.logspace(-2, 1, 50)
    base_sa = 0.3 * np.exp(-0.5 * ((np.log10(periods) - np.log10(0.3)) / 0.5) ** 2)
    return RotDSpectrumResult(
        motion_a_name="comp_a",
        motion_b_name="comp_b",
        n_points=2000,
        dt_s=0.01,
        pga_a_g=0.30,
        pga_b_g=0.25,
        damping=0.05,
        percentiles=[0, 50, 100],
        periods=periods,
        rotd0=base_sa * 0.7,
        rotd50=base_sa,
        rotd100=base_sa * 1.3,
    )


def _make_signal_processing_result():
    """Create a SignalProcessingResult with synthetic data."""
    time = np.arange(2000) * 0.01
    accel = 0.3 * np.sin(2 * np.pi * 2.0 * time)
    return SignalProcessingResult(
        motion_name="test_motion",
        n_points=2000,
        dt_s=0.01,
        bandpass_hz=[0.5, 10.0],
        baseline_order=1,
        pga_original_g=0.30,
        pga_processed_g=0.28,
        pgv_processed_m_per_s=0.12,
        pgd_processed_m=0.004,
        time=time,
        accel_original_g=accel,
        accel_processed_g=accel * 0.95,
        velocity_m_per_s=0.12 * np.cos(2 * np.pi * 2.0 * time),
        displacement_m=0.004 * np.sin(2 * np.pi * 2.0 * time),
    )


# ===================================================================
# Tier 1: Result Dataclass Tests (no eqsig/pyrotd needed)
# ===================================================================

class TestResponseSpectrumResult:
    def test_defaults(self):
        r = ResponseSpectrumResult()
        assert r.motion_name == ""
        assert r.pga_g == 0.0
        assert len(r.periods) == 0

    def test_with_values(self):
        r = _make_response_spectrum_result()
        assert r.motion_name == "test_motion"
        assert r.n_points == 2000
        assert r.pga_g == 0.30
        assert len(r.periods) == 50

    def test_summary_content(self):
        r = _make_response_spectrum_result()
        s = r.summary()
        assert "RESPONSE SPECTRUM" in s
        assert "test_motion" in s
        assert "0.3000" in s  # PGA

    def test_to_dict_keys(self):
        r = _make_response_spectrum_result()
        d = r.to_dict()
        expected = {"motion_name", "n_points", "duration_s", "dt_s",
                    "pga_g", "pgv_m_per_s", "pgd_m", "damping",
                    "Sa_max_g", "T_peak_s", "periods_s", "Sa_g"}
        assert set(d.keys()) == expected

    def test_to_dict_carries_the_spectrum(self):
        """G6 (live smoke SS2-1): the documented Sa_max_g / T_peak_s /
        periods_s / Sa_g were never returned, so the model had "no spectral
        ordinates to report"."""
        r = _make_response_spectrum_result()
        d = r.to_dict()
        assert len(d["periods_s"]) == len(d["Sa_g"]) == 50
        i = int(np.argmax(r.Sa_g))
        assert d["Sa_max_g"] == pytest.approx(r.Sa_g[i], abs=1e-5)
        assert d["T_peak_s"] == pytest.approx(r.periods[i], abs=1e-5)
        assert d["Sa_max_g"] == pytest.approx(max(d["Sa_g"]))

    def test_to_dict_without_spectrum_has_no_empty_arrays(self):
        d = ResponseSpectrumResult().to_dict()
        assert "Sa_g" not in d and "Sa_max_g" not in d

    def test_to_dict_json_serializable(self):
        r = _make_response_spectrum_result()
        d = r.to_dict()
        s = json.dumps(d)
        assert isinstance(s, str)


class TestIntensityMeasuresResult:
    def test_defaults(self):
        r = IntensityMeasuresResult()
        assert r.arias_intensity_m_per_s == 0.0
        assert r.cav_m_per_s == 0.0

    def test_with_values(self):
        r = _make_intensity_measures_result()
        assert r.arias_intensity_m_per_s == 1.5
        assert r.significant_duration_s == 8.5
        assert r.cav_m_per_s == 5.0

    def test_summary_content(self):
        r = _make_intensity_measures_result()
        s = r.summary()
        assert "INTENSITY MEASURES" in s
        assert "Arias" in s
        assert "D5-95" in s

    def test_to_dict_keys(self):
        r = _make_intensity_measures_result()
        d = r.to_dict()
        expected = {"motion_name", "pga_g", "pgv_m_per_s", "pgd_m",
                    "arias_intensity_m_per_s", "significant_duration_s",
                    "cav_m_per_s", "bracketed_duration_s"}
        assert set(d.keys()) == expected

    def test_to_dict_json_serializable(self):
        r = _make_intensity_measures_result()
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


class TestRotDSpectrumResult:
    def test_defaults(self):
        r = RotDSpectrumResult()
        assert r.motion_a_name == ""
        assert len(r.rotd50) == 0

    def test_with_values(self):
        r = _make_rotd_spectrum_result()
        assert r.motion_a_name == "comp_a"
        assert r.pga_a_g == 0.30
        assert len(r.rotd50) == 50

    def test_summary_content(self):
        r = _make_rotd_spectrum_result()
        s = r.summary()
        assert "RotD" in s
        assert "comp_a" in s
        assert "comp_b" in s

    def test_to_dict_keys(self):
        r = _make_rotd_spectrum_result()
        d = r.to_dict()
        assert "motion_a_name" in d
        assert "peak_rotd50_g" in d
        assert "peak_rotd100_g" in d

    def test_to_dict_carries_the_spectra(self):
        """G6: periods_s / percentiles / spectra were documented, never
        returned."""
        r = _make_rotd_spectrum_result()
        d = r.to_dict()
        assert d["percentiles"] == ["0", "50", "100"]
        assert len(d["periods_s"]) == 50
        assert set(d["spectra"]) == {"0", "50", "100"}
        assert max(d["spectra"]["50"]) == pytest.approx(d["peak_rotd50_g"],
                                                        abs=1e-4)
        assert d["engine"] == "pyrotd"

    def test_to_dict_json_serializable(self):
        r = _make_rotd_spectrum_result()
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


class TestSignalProcessingResult:
    def test_defaults(self):
        r = SignalProcessingResult()
        assert r.pga_original_g == 0.0
        assert r.bandpass_hz == []

    def test_with_values(self):
        r = _make_signal_processing_result()
        assert r.bandpass_hz == [0.5, 10.0]
        assert r.baseline_order == 1
        assert r.pga_processed_g == 0.28

    def test_summary_content(self):
        r = _make_signal_processing_result()
        s = r.summary()
        assert "SIGNAL PROCESSING" in s
        assert "Bandpass" in s
        assert "Baseline" in s

    def test_to_dict_keys(self):
        r = _make_signal_processing_result()
        d = r.to_dict()
        expected = {"motion_name", "n_points", "dt_s", "bandpass_hz",
                    "baseline_order", "pga_original_g", "pga_processed_g",
                    "pgv_processed_m_per_s", "pgd_processed_m",
                    "filter_applied", "baseline_corrected"}
        assert set(d.keys()) == expected
        assert d["filter_applied"] is True and d["baseline_corrected"] is True
        none = SignalProcessingResult().to_dict()
        assert none["filter_applied"] is False
        assert none["baseline_corrected"] is False

    def test_to_dict_json_serializable(self):
        r = _make_signal_processing_result()
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# ===================================================================
# Tier 1: Plot Tests
# ===================================================================

class TestResultPlots:
    def test_response_spectrum_plot(self):
        r = _make_response_spectrum_result()
        ax = r.plot_spectrum(show=False)
        assert ax is not None

    def test_response_spectrum_time_history(self):
        r = _make_response_spectrum_result()
        ax = r.plot_time_history(show=False)
        assert ax is not None

    def test_response_spectrum_plot_all(self):
        r = _make_response_spectrum_result()
        fig, axes = r.plot_all(show=False)
        assert fig is not None

    def test_intensity_measures_plot_arias(self):
        r = _make_intensity_measures_result()
        ax = r.plot_arias(show=False)
        assert ax is not None

    def test_intensity_measures_plot_all(self):
        r = _make_intensity_measures_result()
        fig, axes = r.plot_all(show=False)
        assert fig is not None

    def test_rotd_plot(self):
        r = _make_rotd_spectrum_result()
        ax = r.plot_rotd(show=False)
        assert ax is not None

    def test_rotd_plot_all(self):
        r = _make_rotd_spectrum_result()
        fig, ax = r.plot_all(show=False)
        assert fig is not None

    def test_signal_processing_plot_comparison(self):
        r = _make_signal_processing_result()
        ax = r.plot_comparison(show=False)
        assert ax is not None

    def test_signal_processing_plot_vel_disp(self):
        r = _make_signal_processing_result()
        axes = r.plot_vel_disp(show=False)
        assert axes is not None

    def test_signal_processing_plot_all(self):
        r = _make_signal_processing_result()
        fig, axes = r.plot_all(show=False)
        assert fig is not None


# ===================================================================
# Tier 1: Input Validation Tests
# ===================================================================

class TestInputValidation:
    def test_spectrum_bad_damping_zero(self):
        from seismic_signals_agent.response_spectrum import _validate_spectrum_inputs
        with pytest.raises(ValueError, match="damping"):
            _validate_spectrum_inputs(None, 0.0)

    def test_spectrum_bad_damping_one(self):
        from seismic_signals_agent.response_spectrum import _validate_spectrum_inputs
        with pytest.raises(ValueError, match="damping"):
            _validate_spectrum_inputs(None, 1.0)

    def test_spectrum_negative_periods(self):
        from seismic_signals_agent.response_spectrum import _validate_spectrum_inputs
        with pytest.raises(ValueError, match="positive"):
            _validate_spectrum_inputs([-1.0, 0.5], 0.05)

    def test_spectrum_empty_periods(self):
        from seismic_signals_agent.response_spectrum import _validate_spectrum_inputs
        with pytest.raises(ValueError, match="non-empty"):
            _validate_spectrum_inputs([], 0.05)

    def test_intensity_bad_start(self):
        from seismic_signals_agent.intensity_measures import _validate_intensity_inputs
        with pytest.raises(ValueError, match="sig_dur_start"):
            _validate_intensity_inputs(0.0, 0.95)

    def test_intensity_bad_end(self):
        from seismic_signals_agent.intensity_measures import _validate_intensity_inputs
        with pytest.raises(ValueError, match="sig_dur_end"):
            _validate_intensity_inputs(0.05, 1.0)

    def test_intensity_start_ge_end(self):
        from seismic_signals_agent.intensity_measures import _validate_intensity_inputs
        with pytest.raises(ValueError, match="less than"):
            _validate_intensity_inputs(0.95, 0.05)

    def test_rotd_missing_component_a(self):
        from seismic_signals_agent.rotd_spectrum import _validate_rotd_inputs
        with pytest.raises(ValueError, match="Component A"):
            _validate_rotd_inputs(None, np.array([1, 2, 3]), None, 0.05, None)

    def test_rotd_missing_component_b(self):
        from seismic_signals_agent.rotd_spectrum import _validate_rotd_inputs
        with pytest.raises(ValueError, match="Component B"):
            _validate_rotd_inputs(np.array([1, 2, 3]), None, None, 0.05, None)

    def test_rotd_bad_percentile(self):
        from seismic_signals_agent.rotd_spectrum import _validate_rotd_inputs
        with pytest.raises(ValueError, match="percentile"):
            _validate_rotd_inputs(np.array([1]), np.array([1]), None, 0.05, [150])

    def test_processing_nothing_specified(self):
        from seismic_signals_agent.signal_processing import _validate_processing_inputs
        with pytest.raises(ValueError, match="At least one"):
            _validate_processing_inputs(None, None)

    def test_processing_bad_bandpass_format(self):
        from seismic_signals_agent.signal_processing import _validate_processing_inputs
        with pytest.raises(ValueError, match="bandpass"):
            _validate_processing_inputs([1.0], None)

    def test_processing_bad_bandpass_flow(self):
        from seismic_signals_agent.signal_processing import _validate_processing_inputs
        with pytest.raises(ValueError, match="f_low"):
            _validate_processing_inputs([0.0, 10.0], None)

    def test_processing_bad_bandpass_fhigh(self):
        from seismic_signals_agent.signal_processing import _validate_processing_inputs
        with pytest.raises(ValueError, match="f_high"):
            _validate_processing_inputs([10.0, 5.0], None)

    def test_processing_bad_baseline_order(self):
        from seismic_signals_agent.signal_processing import _validate_processing_inputs
        with pytest.raises(ValueError, match="baseline_order"):
            _validate_processing_inputs(None, -1)


# ===================================================================
# Tier 1: Utility Tests
# ===================================================================

class TestSignalUtils:
    def test_has_eqsig_returns_bool(self):
        assert isinstance(has_eqsig(), bool)

    def test_has_pyrotd_returns_bool(self):
        assert isinstance(has_pyrotd(), bool)

    def test_pyrotd_import_error_is_the_real_one(self, monkeypatch):
        """G10 (live smoke SS2-1): pyrotd 0.6.1 was installed but its import
        failed on pkg_resources, and the app said "not installed"."""
        import builtins
        import importlib.util
        from seismic_signals_agent import signal_utils as su

        real_import = builtins.__import__
        real_find = importlib.util.find_spec

        def fake_import(name, *a, **k):
            if name == "pyrotd":
                raise ModuleNotFoundError("No module named 'pkg_resources'")
            return real_import(name, *a, **k)

        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda n, *a: object() if n == "pyrotd"
                            else real_find(n, *a))
        monkeypatch.setattr(builtins, "__import__", fake_import)
        err = su.pyrotd_import_error()
        assert err.startswith("pyrotd is installed but cannot be imported")
        assert "pkg_resources" in err and "setuptools" in err
        assert su.has_pyrotd() is False
        with pytest.raises(ImportError, match="installed but cannot"):
            su.import_pyrotd()

    def test_pyrotd_not_installed_says_so(self, monkeypatch):
        import importlib.util
        from seismic_signals_agent import signal_utils as su
        real_find = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda n, *a: None if n == "pyrotd"
                            else real_find(n, *a))
        assert su.pyrotd_import_error() == "pyrotd is not installed"
        with pytest.raises(ImportError, match="not installed"):
            su.import_pyrotd()


# ===================================================================
# Tier 1: Foundry Metadata Tests
# ===================================================================

class TestFoundryMetadata:
    def test_list_methods_all(self):
        import foundry.seismic_signals_agent_foundry as saf
        result = json.loads(saf.seismic_signals_list_methods.__wrapped__(""))
        assert "Response Spectrum" in result
        assert "Intensity Measures" in result

    def test_list_methods_filtered(self):
        import foundry.seismic_signals_agent_foundry as saf
        result = json.loads(saf.seismic_signals_list_methods.__wrapped__(
            "Response Spectrum"))
        assert "Response Spectrum" in result
        assert "Intensity Measures" not in result

    def test_list_methods_bad_category(self):
        import foundry.seismic_signals_agent_foundry as saf
        result = json.loads(saf.seismic_signals_list_methods.__wrapped__(
            "Nonexistent"))
        assert "error" in result

    def test_describe_response_spectrum(self):
        import foundry.seismic_signals_agent_foundry as saf
        result = json.loads(saf.seismic_signals_describe_method.__wrapped__(
            "response_spectrum"))
        assert "parameters" in result
        assert "damping" in result["parameters"]

    def test_describe_intensity_measures(self):
        import foundry.seismic_signals_agent_foundry as saf
        result = json.loads(saf.seismic_signals_describe_method.__wrapped__(
            "intensity_measures"))
        assert "Arias" in result["description"]

    def test_describe_rotd_spectrum(self):
        import foundry.seismic_signals_agent_foundry as saf
        result = json.loads(saf.seismic_signals_describe_method.__wrapped__(
            "rotd_spectrum"))
        assert "RotD" in result["description"]

    def test_describe_signal_processing(self):
        import foundry.seismic_signals_agent_foundry as saf
        result = json.loads(saf.seismic_signals_describe_method.__wrapped__(
            "signal_processing"))
        assert "bandpass" in result["parameters"]

    def test_describe_unknown_method(self):
        import foundry.seismic_signals_agent_foundry as saf
        result = json.loads(saf.seismic_signals_describe_method.__wrapped__(
            "nonexistent"))
        assert "error" in result

    def test_agent_invalid_json(self):
        import foundry.seismic_signals_agent_foundry as saf
        result = json.loads(saf.seismic_signals_agent.__wrapped__(
            "response_spectrum", "not valid json"))
        assert "error" in result

    def test_agent_unknown_method(self):
        import foundry.seismic_signals_agent_foundry as saf
        result = json.loads(saf.seismic_signals_agent.__wrapped__(
            "nonexistent", "{}"))
        assert "error" in result


# ===================================================================
# Tier 2: eqsig Integration Tests
# ===================================================================

class TestResponseSpectrumIntegration:
    @requires_eqsig
    def test_requested_periods_come_back_with_sa(self):
        """G6, the live SS2-1 call: periods given, Sa expected back."""
        from seismic_signals_agent import analyze_response_spectrum
        periods = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
        d = analyze_response_spectrum(motion="synthetic_pulse",
                                      damping=0.05,
                                      periods=periods).to_dict()
        assert d["periods_s"] == periods
        assert len(d["Sa_g"]) == len(periods)
        assert d["Sa_max_g"] == max(d["Sa_g"])
        assert d["T_peak_s"] in periods
        # Short-period Sa tends to PGA (rigid oscillator).
        assert d["Sa_g"][0] == pytest.approx(d["pga_g"], rel=0.1)

    @requires_eqsig
    def test_synthetic_pulse(self):
        from seismic_signals_agent import analyze_response_spectrum
        r = analyze_response_spectrum(motion="synthetic_pulse")
        assert r.pga_g > 0
        assert len(r.Sa_g) > 0
        assert len(r.periods) == 200
        # Periods should be sorted ascending
        assert np.all(np.diff(r.periods) > 0)

    @requires_eqsig
    def test_custom_motion(self):
        from seismic_signals_agent import analyze_response_spectrum
        accel = 0.2 * np.sin(2 * np.pi * 3.0 * np.arange(2000) * 0.01)
        r = analyze_response_spectrum(accel_history=accel, dt=0.01)
        assert r.motion_name == "custom"
        assert r.pga_g > 0

    @requires_eqsig
    def test_custom_periods(self):
        from seismic_signals_agent import analyze_response_spectrum
        periods = [0.1, 0.5, 1.0, 2.0]
        r = analyze_response_spectrum(motion="synthetic_pulse", periods=periods)
        assert len(r.periods) == 4

    @requires_eqsig
    def test_pga_matches_short_period(self):
        from seismic_signals_agent import analyze_response_spectrum
        r = analyze_response_spectrum(motion="synthetic_pulse",
                                      periods=[0.01, 0.02, 0.05, 0.1, 1.0])
        # PGA should approximately equal Sa at very short period
        assert abs(r.Sa_g[0] - r.pga_g) / r.pga_g < 0.3  # within 30%

    @requires_eqsig
    def test_result_json_serializable(self):
        from seismic_signals_agent import analyze_response_spectrum
        r = analyze_response_spectrum(motion="synthetic_pulse")
        d = r.to_dict()
        s = json.dumps(d)
        assert isinstance(s, str)


class TestIntensityMeasuresIntegration:
    @requires_eqsig
    def test_synthetic_pulse(self):
        from seismic_signals_agent import analyze_intensity_measures
        r = analyze_intensity_measures(motion="synthetic_pulse")
        assert r.arias_intensity_m_per_s > 0
        assert r.significant_duration_s > 0
        assert r.cav_m_per_s > 0
        assert r.pga_g > 0
        assert r.pgv_m_per_s > 0

    @requires_eqsig
    def test_custom_husid_bounds(self):
        from seismic_signals_agent import analyze_intensity_measures
        r = analyze_intensity_measures(motion="synthetic_pulse",
                                       sig_dur_start=0.10, sig_dur_end=0.90)
        assert r.sig_dur_start == 0.10
        assert r.sig_dur_end == 0.90
        assert r.significant_duration_s > 0

    @requires_eqsig
    def test_arias_cumulative_monotonic(self):
        from seismic_signals_agent import analyze_intensity_measures
        r = analyze_intensity_measures(motion="synthetic_pulse")
        # Cumulative Arias should be monotonically non-decreasing
        assert np.all(np.diff(r.arias_cumulative) >= -1e-10)


class TestSignalProcessingIntegration:
    @requires_eqsig
    def test_bandpass_only(self):
        from seismic_signals_agent import analyze_signal_processing
        r = analyze_signal_processing(motion="synthetic_pulse",
                                       bandpass=[0.5, 10.0])
        assert r.pga_processed_g > 0
        assert r.bandpass_hz == [0.5, 10.0]
        assert r.baseline_order == -1

    @requires_eqsig
    def test_baseline_only(self):
        from seismic_signals_agent import analyze_signal_processing
        r = analyze_signal_processing(motion="synthetic_pulse",
                                       baseline_order=1)
        assert r.pga_processed_g > 0
        assert r.baseline_order == 1

    @requires_eqsig
    def test_both_operations(self):
        from seismic_signals_agent import analyze_signal_processing
        r = analyze_signal_processing(motion="synthetic_pulse",
                                       bandpass=[0.5, 10.0],
                                       baseline_order=1)
        assert r.pga_processed_g > 0
        assert r.bandpass_hz == [0.5, 10.0]
        assert r.baseline_order == 1

    @requires_eqsig
    def test_velocity_displacement_populated(self):
        from seismic_signals_agent import analyze_signal_processing
        r = analyze_signal_processing(motion="synthetic_pulse",
                                       baseline_order=0)
        assert len(r.velocity_m_per_s) > 0
        assert len(r.displacement_m) > 0
        assert r.pgv_processed_m_per_s > 0


class TestNigamJenningsVsNewmarkBeta:
    @requires_eqsig
    def test_comparison(self):
        """Nigam-Jennings and Newmark-beta should give similar spectra."""
        from seismic_signals_agent import analyze_response_spectrum
        from opensees_agent.opensees_utils import compute_response_spectrum
        from opensees_agent.ground_motions import get_motion

        accel_g, dt = get_motion("synthetic_pulse")
        periods = np.logspace(-1, 0.5, 20)

        # eqsig (Nigam-Jennings)
        r = analyze_response_spectrum(motion="synthetic_pulse",
                                      periods=periods, damping=0.05)

        # Newmark-beta
        Sa_nb = compute_response_spectrum(accel_g, dt, periods, damping=0.05)

        # Should be within ~10% for most periods
        ratio = r.Sa_g / Sa_nb
        assert np.median(ratio) > 0.85
        assert np.median(ratio) < 1.15


# ===================================================================
# Tier 2: RotD integration tests — pyrotd when it imports, otherwise the
# numpy implementation of the same procedure (rotd_native), so these run
# everywhere.
# ===================================================================

def _load_pyrotd_bypassing_pkg_resources():
    """pyrotd 0.6 only uses pkg_resources for its __version__; load it with
    a stub so it can serve as the oracle for the numpy implementation."""
    import importlib
    import importlib.metadata as md
    import importlib.util
    import sys
    import types
    if importlib.util.find_spec("pyrotd") is None:
        pytest.skip("pyrotd not installed; no oracle")
    added = "pkg_resources" not in sys.modules
    if added:
        stub = types.ModuleType("pkg_resources")
        stub.get_distribution = lambda n: types.SimpleNamespace(
            version=md.version(n))
        sys.modules["pkg_resources"] = stub
    try:
        pyrotd = importlib.import_module("pyrotd")
    finally:
        if added:
            del sys.modules["pkg_resources"]
    pyrotd.processes = 1   # no worker pool (Windows spawn re-imports)
    return pyrotd


class TestRotDNative:
    def test_matches_pyrotd(self):
        """The fallback reproduces pyrotd 0.6.1 to round-off (measured
        max relative difference ~2e-16 on these records)."""
        from opensees_agent.ground_motions import validate_motion_input
        from seismic_signals_agent.rotd_native import rotated_spectral_accels
        pyrotd = _load_pyrotd_bypassing_pkg_resources()
        a, dt = validate_motion_input("synthetic_pulse", None, None)
        b, _ = validate_motion_input("synthetic_long", None, None)
        n = max(len(a), len(b))
        a = np.pad(a, (0, n - len(a)))
        b = np.pad(b, (0, n - len(b)))
        periods = np.logspace(-2, 1, 30)
        ref = pyrotd.calc_rotated_spec_accels(
            dt, a, b, 1.0 / periods, osc_damping=0.05,
            percentiles=[0, 50, 100])
        ours = rotated_spectral_accels(dt, a, b, periods, 0.05, [0, 50, 100])
        for p in (0, 50, 100):
            expected = np.asarray(ref[ref.percentile == p].spec_accel, float)
            np.testing.assert_allclose(ours[float(p)], expected, rtol=1e-9)

    def test_rotd100_of_one_component_is_its_spectrum(self):
        """Component B = 0: every rotation is a cos-scaled copy of A, so
        RotD100 = A's own pseudo-spectral acceleration (theta = 0)."""
        from seismic_signals_agent.rotd_native import (
            rotated_spectral_accels, _psa_response,
        )
        dt = 0.01
        t = np.arange(1500) * dt
        a = 0.2 * np.sin(2 * np.pi * 1.5 * t) * np.exp(-0.2 * t)
        periods = np.array([0.2, 0.5, 1.0])
        out = rotated_spectral_accels(dt, a, np.zeros_like(a), periods)
        fa = np.fft.rfft(a)
        freqs = np.linspace(0, 1 / (2 * dt), fa.size)
        own = [np.abs(_psa_response(freqs, fa, 0.05, 1 / T, 5.0)).max()
               for T in periods]
        np.testing.assert_allclose(out[100.0], own, rtol=1e-12)

    def test_analysis_falls_back_and_says_why(self, monkeypatch):
        from seismic_signals_agent import rotd_spectrum
        monkeypatch.setattr(rotd_spectrum, "pyrotd_import_error",
                            lambda: "pyrotd is installed but cannot be "
                                    "imported: ModuleNotFoundError: x")
        r = rotd_spectrum.analyze_rotd_spectrum(
            motion_a="synthetic_pulse", motion_b="synthetic_long",
            percentiles=[0, 50, 84, 100])
        d = r.to_dict()
        assert d["engine"] == "numpy"
        assert "cannot be imported" in d["engine_note"]
        assert d["percentiles"] == ["0", "50", "84", "100"]
        assert len(d["spectra"]["84"]) == len(d["periods_s"])


class TestRotDSpectrumIntegration:
    def test_two_synthetic_components(self):
        from seismic_signals_agent import analyze_rotd_spectrum
        r = analyze_rotd_spectrum(
            motion_a="synthetic_pulse",
            motion_b="synthetic_long",
        )
        assert len(r.rotd50) > 0
        assert len(r.rotd100) > 0
        assert r.pga_a_g > 0
        assert r.pga_b_g > 0

    # (runs with pyrotd or the numpy fallback)
    def test_rotd_ordering(self):
        """RotD0 <= RotD50 <= RotD100 at every period."""
        from seismic_signals_agent import analyze_rotd_spectrum
        r = analyze_rotd_spectrum(
            motion_a="synthetic_pulse",
            motion_b="synthetic_long",
        )
        if len(r.rotd0) > 0 and len(r.rotd50) > 0:
            assert np.all(r.rotd0 <= r.rotd50 + 1e-10)
        if len(r.rotd50) > 0 and len(r.rotd100) > 0:
            assert np.all(r.rotd50 <= r.rotd100 + 1e-10)

    # (runs with pyrotd or the numpy fallback)
    def test_identical_components(self):
        """Identical components: RotD50 and RotD100 are close."""
        from seismic_signals_agent import analyze_rotd_spectrum
        r = analyze_rotd_spectrum(
            motion_a="synthetic_pulse",
            motion_b="synthetic_pulse",
        )
        if len(r.rotd50) > 0 and len(r.rotd100) > 0:
            # For identical components, RotD100/RotD50 ratio should be
            # bounded (both driven by same signal content)
            ratio = r.rotd100 / np.maximum(r.rotd50, 1e-10)
            assert np.median(ratio) < 2.0

    # (runs with pyrotd or the numpy fallback)
    def test_result_json_serializable(self):
        from seismic_signals_agent import analyze_rotd_spectrum
        r = analyze_rotd_spectrum(
            motion_a="synthetic_pulse",
            motion_b="synthetic_long",
        )
        d = r.to_dict()
        s = json.dumps(d)
        assert isinstance(s, str)


@requires_eqsig
def test_response_spectrum_target_pga():
    """G19: a built-in motion scaled to a target PGA; Sa scales with it."""
    from seismic_signals_agent import analyze_response_spectrum
    periods = [0.1, 0.5, 1.0]
    a = analyze_response_spectrum(motion="synthetic_pulse", periods=periods)
    b = analyze_response_spectrum(motion="synthetic_pulse", periods=periods,
                                  target_pga_g=0.15)
    assert b.pga_g == pytest.approx(0.15, rel=1e-6)
    np.testing.assert_allclose(b.Sa_g, a.Sa_g * (0.15 / a.pga_g), rtol=1e-6)
    assert "scaled to PGA 0.15 g" in b.to_dict()["motion_name"]
