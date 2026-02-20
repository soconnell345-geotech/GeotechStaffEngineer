"""
Tests for hvsrpy_agent — HVSR site characterization.

Tier 1: No hvsrpy required (result dataclasses, validation, utilities, Foundry metadata)
Tier 2: Requires hvsrpy (integration tests with synthetic seismograms)
"""

import json
import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from hvsrpy_agent.hvsrpy_utils import has_hvsrpy
from hvsrpy_agent.results import HvsrResult

requires_hvsrpy = pytest.mark.skipif(
    not has_hvsrpy(), reason="hvsrpy not installed"
)


# =====================================================================
# Tier 1: Result dataclass defaults
# =====================================================================

class TestHvsrResultDefaults:
    """Test HvsrResult with default values."""

    def test_default_construction(self):
        r = HvsrResult()
        assert r.f0_hz == 0.0
        assert r.A0 == 0.0
        assert r.T0_s == 0.0
        assert r.n_windows == 0

    def test_construction_with_values(self):
        r = HvsrResult(
            f0_hz=2.5, A0=4.2, T0_s=0.4,
            f0_std_hz=0.1, A0_std=0.5,
            n_windows=10, n_valid_windows=8,
            window_length_s=60.0,
        )
        assert r.f0_hz == 2.5
        assert r.A0 == 4.2
        assert r.T0_s == 0.4
        assert r.n_valid_windows == 8

    def test_summary_contains_f0(self):
        r = HvsrResult(f0_hz=3.0, A0=5.0, T0_s=0.333)
        s = r.summary()
        assert "3.000" in s
        assert "5.00" in s

    def test_to_dict_keys(self):
        r = HvsrResult(f0_hz=2.0, A0=3.0, T0_s=0.5)
        d = r.to_dict()
        assert "f0_hz" in d
        assert "A0" in d
        assert "T0_s" in d
        assert "sesame_reliability" in d
        assert "sesame_clarity" in d

    def test_to_dict_json_serializable(self):
        r = HvsrResult(
            f0_hz=2.0, A0=3.0, T0_s=0.5,
            frequency=np.geomspace(0.2, 50, 100),
            mean_curve=np.ones(100),
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_to_dict_includes_curves(self):
        freq = np.geomspace(0.2, 50, 50)
        r = HvsrResult(
            f0_hz=2.0, A0=3.0, T0_s=0.5,
            frequency=freq,
            mean_curve=np.ones(50),
            std_curve=np.ones(50) * 0.1,
        )
        d = r.to_dict()
        assert "frequency_hz" in d
        assert "mean_curve" in d
        assert "std_curve" in d
        assert len(d["frequency_hz"]) == 50

    def test_to_dict_no_curves_when_none(self):
        r = HvsrResult(f0_hz=2.0, A0=3.0, T0_s=0.5)
        d = r.to_dict()
        assert "frequency_hz" not in d
        assert "mean_curve" not in d

    def test_sesame_pass_counts(self):
        r = HvsrResult(
            f0_hz=2.0, A0=3.0, T0_s=0.5,
            sesame_reliability=[1.0, 1.0, 0.0],
            sesame_clarity=[1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        )
        d = r.to_dict()
        assert d["sesame_reliability_pass"] == 2
        assert d["sesame_clarity_pass"] == 5


# =====================================================================
# Tier 1: Plot smoke tests
# =====================================================================

class TestPlotSmoke:
    """Smoke tests for plot methods (no hvsrpy needed)."""

    def test_hvsr_plot(self):
        freq = np.geomspace(0.2, 50, 100)
        curve = np.ones(100) * 1.5
        curve[40:50] = 4.0
        r = HvsrResult(
            f0_hz=2.0, A0=4.0, T0_s=0.5,
            frequency=freq,
            mean_curve=curve,
            upper_curve=curve * 1.2,
            lower_curve=curve * 0.8,
        )
        ax = r.plot_hvsr(show=False)
        assert ax is not None


# =====================================================================
# Tier 1: Input validation
# =====================================================================

class TestInputValidation:
    """Test input validation (no hvsrpy needed)."""

    def test_empty_arrays(self):
        from hvsrpy_agent.hvsr_analysis import _validate_hvsr_inputs
        with pytest.raises(ValueError, match="non-empty"):
            _validate_hvsr_inputs(
                np.array([]), np.array([]), np.array([]),
                0.01, 60.0, 40, "lognormal", "geometric_mean"
            )

    def test_mismatched_lengths(self):
        from hvsrpy_agent.hvsr_analysis import _validate_hvsr_inputs
        with pytest.raises(ValueError, match="same length"):
            _validate_hvsr_inputs(
                np.ones(100), np.ones(200), np.ones(100),
                0.01, 60.0, 40, "lognormal", "geometric_mean"
            )

    def test_bad_dt(self):
        from hvsrpy_agent.hvsr_analysis import _validate_hvsr_inputs
        with pytest.raises(ValueError, match="dt"):
            _validate_hvsr_inputs(
                np.ones(100), np.ones(100), np.ones(100),
                -0.01, 60.0, 40, "lognormal", "geometric_mean"
            )

    def test_window_exceeds_duration(self):
        from hvsrpy_agent.hvsr_analysis import _validate_hvsr_inputs
        with pytest.raises(ValueError, match="exceeds total duration"):
            _validate_hvsr_inputs(
                np.ones(100), np.ones(100), np.ones(100),
                0.01, 999.0, 40, "lognormal", "geometric_mean"
            )

    def test_bad_distribution(self):
        from hvsrpy_agent.hvsr_analysis import _validate_hvsr_inputs
        with pytest.raises(ValueError, match="distribution"):
            _validate_hvsr_inputs(
                np.ones(1000), np.ones(1000), np.ones(1000),
                0.01, 5.0, 40, "uniform", "geometric_mean"
            )

    def test_bad_horizontal_method(self):
        from hvsrpy_agent.hvsr_analysis import _validate_hvsr_inputs
        with pytest.raises(ValueError, match="horizontal_method"):
            _validate_hvsr_inputs(
                np.ones(1000), np.ones(1000), np.ones(1000),
                0.01, 5.0, 40, "lognormal", "invalid_method"
            )

    def test_bad_smoothing_bandwidth(self):
        from hvsrpy_agent.hvsr_analysis import _validate_hvsr_inputs
        with pytest.raises(ValueError, match="smoothing_bandwidth"):
            _validate_hvsr_inputs(
                np.ones(1000), np.ones(1000), np.ones(1000),
                0.01, 5.0, -10, "lognormal", "geometric_mean"
            )


# =====================================================================
# Tier 1: Utility functions
# =====================================================================

class TestUtilities:
    """Test utility functions."""

    def test_has_hvsrpy_returns_bool(self):
        assert isinstance(has_hvsrpy(), bool)


# =====================================================================
# Tier 1: Foundry metadata
# =====================================================================

class TestFoundryMetadata:
    """Test Foundry agent metadata functions (no hvsrpy needed)."""

    def test_list_methods_all(self):
        from hvsrpy_agent_foundry import hvsrpy_list_methods
        result = json.loads(hvsrpy_list_methods(""))
        assert "HVSR Analysis" in result

    def test_list_methods_filtered(self):
        from hvsrpy_agent_foundry import hvsrpy_list_methods
        result = json.loads(hvsrpy_list_methods("HVSR Analysis"))
        assert "analyze_hvsr" in result["HVSR Analysis"]

    def test_list_methods_bad_category(self):
        from hvsrpy_agent_foundry import hvsrpy_list_methods
        result = json.loads(hvsrpy_list_methods("nonexistent"))
        assert "error" in result

    def test_describe_analyze_hvsr(self):
        from hvsrpy_agent_foundry import hvsrpy_describe_method
        result = json.loads(hvsrpy_describe_method("analyze_hvsr"))
        assert "parameters" in result
        assert "ns" in result["parameters"]
        assert "ew" in result["parameters"]
        assert "vt" in result["parameters"]

    def test_describe_unknown_method(self):
        from hvsrpy_agent_foundry import hvsrpy_describe_method
        result = json.loads(hvsrpy_describe_method("nonexistent"))
        assert "error" in result

    def test_agent_invalid_json(self):
        from hvsrpy_agent_foundry import hvsrpy_agent
        result = json.loads(hvsrpy_agent("analyze_hvsr", "not json"))
        assert "error" in result

    def test_agent_unknown_method(self):
        from hvsrpy_agent_foundry import hvsrpy_agent
        result = json.loads(hvsrpy_agent("nonexistent", "{}"))
        assert "error" in result


# =====================================================================
# Tier 2: Integration tests (requires hvsrpy)
# =====================================================================

def _make_synthetic_recording(f0=2.0, dt=0.01, duration_s=300.0,
                              amp_h=0.005, amp_v=0.001):
    """Create synthetic 3-component recording with embedded resonance.

    Default 300s at 100Hz gives 10 windows of 30s each — enough for
    reliable statistics and window rejection.
    """
    np.random.seed(42)
    n = int(duration_s / dt)
    t = np.arange(n) * dt
    signal = amp_h * np.sin(2 * np.pi * f0 * t)
    ns = np.random.randn(n) * amp_v + signal
    ew = np.random.randn(n) * amp_v + signal * 0.8
    vt = np.random.randn(n) * amp_v
    return ns, ew, vt, dt


@requires_hvsrpy
class TestHvsrIntegration:
    """Integration tests for HVSR analysis."""

    def test_basic_hvsr(self):
        ns, ew, vt, dt = _make_synthetic_recording(f0=2.0)
        from hvsrpy_agent import analyze_hvsr
        r = analyze_hvsr(ns, ew, vt, dt, window_length_s=30.0)
        assert 1.5 < r.f0_hz < 2.5
        assert r.A0 > 1.0
        assert 0.4 < r.T0_s < 0.7

    def test_n_windows(self):
        ns, ew, vt, dt = _make_synthetic_recording()
        from hvsrpy_agent import analyze_hvsr
        r = analyze_hvsr(ns, ew, vt, dt, window_length_s=30.0)
        assert r.n_windows == 10
        assert r.n_valid_windows <= r.n_windows

    def test_curves_populated(self):
        ns, ew, vt, dt = _make_synthetic_recording()
        from hvsrpy_agent import analyze_hvsr
        r = analyze_hvsr(ns, ew, vt, dt, window_length_s=30.0)
        assert r.frequency is not None
        assert r.mean_curve is not None
        assert r.std_curve is not None
        assert len(r.frequency) == 200  # default n_freq

    def test_sesame_populated(self):
        ns, ew, vt, dt = _make_synthetic_recording()
        from hvsrpy_agent import analyze_hvsr
        r = analyze_hvsr(ns, ew, vt, dt, window_length_s=30.0)
        assert len(r.sesame_reliability) == 3
        assert len(r.sesame_clarity) == 6

    def test_different_f0(self):
        ns, ew, vt, dt = _make_synthetic_recording(f0=5.0)
        from hvsrpy_agent import analyze_hvsr
        r = analyze_hvsr(ns, ew, vt, dt, window_length_s=30.0,
                         rejection_n_std=0)  # skip rejection for synthetic
        assert 4.0 < r.f0_hz < 6.0

    def test_custom_freq_range(self):
        ns, ew, vt, dt = _make_synthetic_recording()
        from hvsrpy_agent import analyze_hvsr
        r = analyze_hvsr(ns, ew, vt, dt, window_length_s=30.0,
                         freq_min=0.5, freq_max=25.0, n_freq=100)
        assert len(r.frequency) == 100
        assert r.frequency[0] >= 0.49
        assert r.frequency[-1] <= 25.1

    def test_with_bandpass_filter(self):
        ns, ew, vt, dt = _make_synthetic_recording()
        from hvsrpy_agent import analyze_hvsr
        r = analyze_hvsr(ns, ew, vt, dt, window_length_s=30.0,
                         filter_hz=[0.5, 25.0])
        assert r.f0_hz > 0

    def test_no_rejection(self):
        ns, ew, vt, dt = _make_synthetic_recording()
        from hvsrpy_agent import analyze_hvsr
        r = analyze_hvsr(ns, ew, vt, dt, window_length_s=30.0,
                         rejection_n_std=0)
        assert r.n_valid_windows == r.n_windows

    def test_normal_distribution(self):
        ns, ew, vt, dt = _make_synthetic_recording()
        from hvsrpy_agent import analyze_hvsr
        r = analyze_hvsr(ns, ew, vt, dt, window_length_s=30.0,
                         distribution="normal")
        assert r.distribution == "normal"
        assert r.f0_hz > 0

    def test_arithmetic_mean_method(self):
        ns, ew, vt, dt = _make_synthetic_recording()
        from hvsrpy_agent import analyze_hvsr
        r = analyze_hvsr(ns, ew, vt, dt, window_length_s=30.0,
                         horizontal_method="arithmetic_mean")
        assert r.horizontal_method == "arithmetic_mean"
        assert r.f0_hz > 0

    def test_to_dict_json_serializable(self):
        ns, ew, vt, dt = _make_synthetic_recording()
        from hvsrpy_agent import analyze_hvsr
        r = analyze_hvsr(ns, ew, vt, dt, window_length_s=30.0)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_foundry_agent_analyze_hvsr(self):
        ns, ew, vt, dt = _make_synthetic_recording()
        from hvsrpy_agent_foundry import hvsrpy_agent
        params = {
            "ns": ns.tolist(),
            "ew": ew.tolist(),
            "vt": vt.tolist(),
            "dt": dt,
            "window_length_s": 30.0,
        }
        result = json.loads(hvsrpy_agent("analyze_hvsr", json.dumps(params)))
        assert "error" not in result
        assert result["f0_hz"] > 0
        assert result["A0"] > 0

    def test_plot_integration(self):
        ns, ew, vt, dt = _make_synthetic_recording()
        from hvsrpy_agent import analyze_hvsr
        r = analyze_hvsr(ns, ew, vt, dt, window_length_s=30.0)
        ax = r.plot_hvsr(show=False)
        assert ax is not None
