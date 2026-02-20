"""
Tests for swprocess_agent — MASW surface wave processing.

Tier 1: No swprocess required (result dataclasses, validation, utilities, Foundry metadata)
Tier 2: Requires swprocess (integration tests with synthetic data)
"""

import json
import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from swprocess_agent.swprocess_utils import has_swprocess
from swprocess_agent.results import DispersionResult

requires_swprocess = pytest.mark.skipif(
    not has_swprocess(), reason="swprocess not installed"
)


# =====================================================================
# Helpers
# =====================================================================

def _make_synthetic_masw(nsensors=12, nsamples=500, dt=0.002, dx=2.0,
                          phase_vel=150.0, freq=15.0):
    """Create synthetic MASW data with a single surface wave mode."""
    traces = []
    offsets = []
    for i in range(nsensors):
        offset = (i + 1) * dx
        offsets.append(offset)
        t = np.arange(nsamples) * dt
        delay = offset / phase_vel
        # Gaussian-windowed sinusoid
        env = np.exp(-((t - delay - 0.15) / 0.05) ** 2)
        amp = np.sin(2 * np.pi * freq * (t - delay)) * env
        amp[t < delay] = 0
        traces.append(amp)
    return traces, offsets, dt


# =====================================================================
# Tier 1: DispersionResult defaults
# =====================================================================

class TestDispersionResultDefaults:

    def test_default_construction(self):
        r = DispersionResult()
        assert r.n_channels == 0
        assert r.transform == ""
        assert r.power is None

    def test_construction_with_values(self):
        r = DispersionResult(
            n_channels=24, spacing_m=1.0, transform="phase_shift",
            n_freq=50, n_vel=200,
        )
        assert r.n_channels == 24
        assert r.transform == "phase_shift"

    def test_summary_contains_info(self):
        r = DispersionResult(
            n_channels=24, spacing_m=1.0, transform="phase_shift",
            n_freq=50, n_vel=200,
        )
        s = r.summary()
        assert "24" in s
        assert "phase_shift" in s

    def test_to_dict_keys(self):
        r = DispersionResult(
            n_channels=24, spacing_m=1.0, transform="phase_shift",
            n_freq=50, n_vel=200,
        )
        d = r.to_dict()
        assert "n_channels" in d
        assert "transform" in d
        assert "spacing_m" in d

    def test_to_dict_with_disp_curve(self):
        r = DispersionResult(
            n_channels=12, spacing_m=2.0, transform="phase_shift",
            n_freq=20, n_vel=100,
            disp_freq=np.linspace(5, 50, 20),
            disp_vel=np.linspace(100, 300, 20),
        )
        d = r.to_dict()
        assert "disp_freq_hz" in d
        assert "disp_vel_mps" in d
        assert len(d["disp_freq_hz"]) == 20

    def test_to_dict_no_power_grid(self):
        """Power grid should NOT be in JSON (too large)."""
        r = DispersionResult(
            n_channels=12, transform="phase_shift",
            power=np.ones((100, 50)),
        )
        d = r.to_dict()
        assert "power" not in d

    def test_to_dict_json_serializable(self):
        r = DispersionResult(
            n_channels=12, spacing_m=2.0, transform="phase_shift",
            n_freq=10, n_vel=50,
            disp_freq=np.linspace(5, 50, 10),
            disp_vel=np.linspace(100, 300, 10),
            frequencies=np.linspace(5, 50, 10),
            velocities_grid=np.linspace(50, 500, 50),
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 1: Plot smoke test
# =====================================================================

class TestPlotSmoke:

    def test_dispersion_plot(self):
        freq = np.linspace(5, 50, 20)
        vel = np.linspace(100, 300, 20)
        r = DispersionResult(
            n_channels=12, transform="phase_shift",
            disp_freq=freq, disp_vel=vel,
        )
        ax = r.plot_dispersion(show=False)
        assert ax is not None


# =====================================================================
# Tier 1: Input validation
# =====================================================================

class TestValidation:

    def test_too_few_traces(self):
        from swprocess_agent.masw_analysis import _validate_masw_inputs
        with pytest.raises(ValueError, match="at least 3"):
            _validate_masw_inputs(
                [np.ones(100), np.ones(100)], [1.0, 2.0],
                0.001, "phase_shift", 5, 50, 50, 500, 100,
            )

    def test_mismatched_lengths(self):
        from swprocess_agent.masw_analysis import _validate_masw_inputs
        with pytest.raises(ValueError, match="same length"):
            _validate_masw_inputs(
                [np.ones(100)] * 5, [1.0, 2.0],
                0.001, "phase_shift", 5, 50, 50, 500, 100,
            )

    def test_bad_dt(self):
        from swprocess_agent.masw_analysis import _validate_masw_inputs
        with pytest.raises(ValueError, match="dt"):
            _validate_masw_inputs(
                [np.ones(100)] * 5, [1, 2, 3, 4, 5],
                -0.001, "phase_shift", 5, 50, 50, 500, 100,
            )

    def test_bad_transform(self):
        from swprocess_agent.masw_analysis import _validate_masw_inputs
        with pytest.raises(ValueError, match="transform"):
            _validate_masw_inputs(
                [np.ones(100)] * 5, [1, 2, 3, 4, 5],
                0.001, "bad_transform", 5, 50, 50, 500, 100,
            )

    def test_bad_freq_range(self):
        from swprocess_agent.masw_analysis import _validate_masw_inputs
        with pytest.raises(ValueError, match="fmin"):
            _validate_masw_inputs(
                [np.ones(100)] * 5, [1, 2, 3, 4, 5],
                0.001, "phase_shift", 50, 5, 50, 500, 100,
            )

    def test_bad_vel_range(self):
        from swprocess_agent.masw_analysis import _validate_masw_inputs
        with pytest.raises(ValueError, match="vmin"):
            _validate_masw_inputs(
                [np.ones(100)] * 5, [1, 2, 3, 4, 5],
                0.001, "phase_shift", 5, 50, 500, 50, 100,
            )

    def test_mismatched_trace_lengths(self):
        from swprocess_agent.masw_analysis import _validate_masw_inputs
        with pytest.raises(ValueError, match="same length"):
            _validate_masw_inputs(
                [np.ones(100), np.ones(200), np.ones(100)], [1, 2, 3],
                0.001, "phase_shift", 5, 50, 50, 500, 100,
            )


# =====================================================================
# Tier 1: Utilities
# =====================================================================

class TestUtilities:

    def test_has_swprocess_returns_bool(self):
        assert isinstance(has_swprocess(), bool)


# =====================================================================
# Tier 1: Foundry metadata
# =====================================================================

class TestFoundryMetadata:

    def test_list_methods_all(self):
        from swprocess_agent_foundry import swprocess_list_methods
        result = json.loads(swprocess_list_methods(""))
        assert "MASW" in result

    def test_list_methods_filtered(self):
        from swprocess_agent_foundry import swprocess_list_methods
        result = json.loads(swprocess_list_methods("MASW"))
        assert "analyze_masw" in result["MASW"]

    def test_list_methods_bad_category(self):
        from swprocess_agent_foundry import swprocess_list_methods
        result = json.loads(swprocess_list_methods("nonexistent"))
        assert "error" in result

    def test_describe_masw(self):
        from swprocess_agent_foundry import swprocess_describe_method
        result = json.loads(swprocess_describe_method("analyze_masw"))
        assert "parameters" in result
        assert "traces" in result["parameters"]
        assert "offsets" in result["parameters"]

    def test_describe_unknown(self):
        from swprocess_agent_foundry import swprocess_describe_method
        result = json.loads(swprocess_describe_method("nonexistent"))
        assert "error" in result

    def test_agent_invalid_json(self):
        from swprocess_agent_foundry import swprocess_agent
        result = json.loads(swprocess_agent("analyze_masw", "not json"))
        assert "error" in result

    def test_agent_unknown_method(self):
        from swprocess_agent_foundry import swprocess_agent
        result = json.loads(swprocess_agent("nonexistent", "{}"))
        assert "error" in result


# =====================================================================
# Tier 2: Integration tests (requires swprocess)
# =====================================================================

@requires_swprocess
class TestMaswIntegration:

    def test_basic_masw(self):
        traces, offsets, dt = _make_synthetic_masw()
        from swprocess_agent import analyze_masw
        r = analyze_masw(traces, offsets, dt, fmin=5, fmax=50,
                         vmin=50, vmax=500, nvel=50)
        assert r.n_channels == 12
        assert r.transform == "phase_shift"
        assert r.disp_freq is not None
        assert r.disp_vel is not None
        assert len(r.disp_freq) == len(r.disp_vel)

    def test_dispersion_curve_range(self):
        traces, offsets, dt = _make_synthetic_masw(phase_vel=150.0)
        from swprocess_agent import analyze_masw
        r = analyze_masw(traces, offsets, dt, fmin=5, fmax=50,
                         vmin=50, vmax=500, nvel=100)
        # At least some velocities should be near 150 m/s
        assert r.disp_vel.min() >= 50
        assert r.disp_vel.max() <= 500

    def test_custom_freq_range(self):
        traces, offsets, dt = _make_synthetic_masw()
        from swprocess_agent import analyze_masw
        r = analyze_masw(traces, offsets, dt, fmin=10, fmax=40,
                         vmin=50, vmax=500, nvel=50)
        assert r.frequencies[0] >= 10
        assert r.frequencies[-1] <= 40

    def test_power_grid_populated(self):
        traces, offsets, dt = _make_synthetic_masw()
        from swprocess_agent import analyze_masw
        r = analyze_masw(traces, offsets, dt, fmin=5, fmax=50,
                         vmin=50, vmax=500, nvel=50)
        assert r.power is not None
        assert r.power.shape[0] == 50  # nvel

    def test_to_dict_json_serializable(self):
        traces, offsets, dt = _make_synthetic_masw()
        from swprocess_agent import analyze_masw
        r = analyze_masw(traces, offsets, dt, fmin=5, fmax=50,
                         vmin=50, vmax=500, nvel=50)
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)

    def test_plot_integration(self):
        traces, offsets, dt = _make_synthetic_masw()
        from swprocess_agent import analyze_masw
        r = analyze_masw(traces, offsets, dt, fmin=5, fmax=50,
                         vmin=50, vmax=500, nvel=50)
        ax = r.plot_dispersion(show=False)
        assert ax is not None


# =====================================================================
# Tier 2: Foundry integration (requires swprocess)
# =====================================================================

@requires_swprocess
class TestFoundryIntegration:

    def test_foundry_masw(self):
        traces, offsets, dt = _make_synthetic_masw(nsensors=6)
        from swprocess_agent_foundry import swprocess_agent
        params = {
            "traces": [t.tolist() for t in traces],
            "offsets": offsets,
            "dt": dt,
            "fmin": 5, "fmax": 50,
            "vmin": 50, "vmax": 500, "nvel": 30,
        }
        result = json.loads(swprocess_agent("analyze_masw", json.dumps(params)))
        assert "error" not in result
        assert result["n_channels"] == 6
        assert "disp_freq_hz" in result
