"""The shared response spectrum (GM-274): its own contract.

Its accuracy against scipy's lsim, and that the OpenSees and seismic-signals
tools give the same Sa through it, are checked in
``opensees_agent/tests/test_opensees_agent.py`` (TestNigamJenningsSpectrum,
TestOneSpectrumCode).
"""

import numpy as np
import pytest

from geotech_common.response_spectrum import (
    G, POINTS_PER_PERIOD, oscillator_response, peak_between_points,
    pseudo_spectral_accel, substeps,
)


def _record(dt=0.01, duration=5.0):
    t = np.arange(0.0, duration, dt)
    return 0.2 * np.sin(2.0 * np.pi * 2.0 * t) * np.exp(-0.3 * t)


def test_periods_come_back_in_the_order_given():
    a = _record()
    sa = pseudo_spectral_accel(a, 0.01, [1.0, 0.1, 0.5])
    np.testing.assert_array_equal(
        sa[[1, 2, 0]], pseudo_spectral_accel(a, 0.01, [0.1, 0.5, 1.0]))


def test_a_period_of_zero_gives_the_pga():
    a = _record()
    assert pseudo_spectral_accel(a, 0.01, [0.0])[0] == np.max(np.abs(a))


def test_damping_is_a_ratio_below_critical():
    with pytest.raises(ValueError, match="below 1"):
        pseudo_spectral_accel(_record(), 0.01, [0.5], damping=1.0)
    with pytest.raises(ValueError, match="at least 0"):
        pseudo_spectral_accel(_record(), 0.01, [0.5], damping=-0.01)


def test_substeps_lie_on_the_straight_lines():
    p = np.array([0.0, 1.0, -1.0])
    np.testing.assert_allclose(substeps(p, 4), [0.0, 0.25, 0.5, 0.75, 1.0,
                                                0.5, 0.0, -0.5, -1.0])
    assert POINTS_PER_PERIOD == 20


def test_the_response_history_is_the_spectrum_s():
    """The history the rotated spectra use (GM-275) gives the spectrum's
    Sa, value for value, with sub-steps (0.05 s, five record steps) and
    without (1 s)."""
    a = _record()
    for period in (0.05, 1.0):
        u, v, h = oscillator_response(a, 0.01, period)
        assert h == pytest.approx(0.01 / (4 if period == 0.05 else 1))
        assert len(u) == (len(a) - 1) * round(0.01 / h) + 1
        w = 2.0 * np.pi / period
        assert (w ** 2 * peak_between_points(u, v, h) / G
                == pseudo_spectral_accel(a, 0.01, [period])[0])


def test_the_response_history_needs_a_period():
    with pytest.raises(ValueError, match="above 0"):
        oscillator_response(_record(), 0.01, 0.0)
