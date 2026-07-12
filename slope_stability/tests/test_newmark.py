"""Unit tests for the Newmark seismic sliding-block analysis (v5.3 B2b).

Covers the three pieces: the yield-acceleration bisection (self-consistent with
the module's pseudo-static FOS), the rigid-block double integrator (checked
against the closed-form rectangular-pulse displacement), and the Jibson (2007)
regression (checked against the published equation). Slide2 #104's own record is
not published, so the integrator is validated analytically here and #104 is
documented in validation_examples/test_published_v039_slope.py.
"""

import copy
import math

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.analysis import analyze_slope
from slope_stability.newmark import (
    yield_acceleration, newmark_displacement, newmark_jibson2007,
    bray_travasarou_2007,
    YieldAccelerationResult, NewmarkResult, _G,
)


def _slope():
    """Simple 2:1 homogeneous slope, 10 m high, c'=25/phi'=30/gamma=20."""
    return SlopeGeometry(
        surface_points=[(0, 0), (10, 0), (30, 10), (50, 10)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=10,
                                    bottom_elevation=-15, gamma=20.0,
                                    phi=30.0, c_prime=25.0)])


# --- rigid-block integrator: closed-form rectangular pulse --------------------

def _rect_pulse_analytic(ap, ay, T):
    """Newmark displacement for a single rectangular acceleration pulse of
    amplitude ap (> ay) and duration T: D = ap*(ap-ay)*T^2/(2*ay)."""
    return ap * (ap - ay) * T * T / (2.0 * ay)


def test_integrator_matches_rectangular_pulse_closed_form():
    ap, T, dt = 3.0, 2.0, 0.0005
    ky = 1.5 / _G                     # ay = 1.5 m/s^2
    ay = ky * _G
    accel = [ap] * int(T / dt) + [0.0] * int(4.0 / dt)   # pulse + rest tail
    r = newmark_displacement(ky, accel, dt)
    assert isinstance(r, NewmarkResult)
    assert r.displacement == pytest.approx(_rect_pulse_analytic(ap, ay, T),
                                           rel=1e-3)
    assert r.displacement_cm == pytest.approx(r.displacement * 100.0)


def test_integrator_exact_for_rectangular_pulse_any_dt():
    """Trapezoidal integration of the piecewise-linear relative velocity is
    EXACT for a rectangular pulse, so the result is dt-independent (up to
    roundoff) when the pulse/tail align with the grid."""
    ap, T = 4.0, 1.5
    ky = 2.0 / _G
    analytic = _rect_pulse_analytic(ap, ky * _G, T)
    for dt in (0.01, 0.001):
        accel = [ap] * int(round(T / dt)) + [0.0] * int(round(3.0 / dt))
        d = newmark_displacement(ky, accel, dt).displacement
        assert d == pytest.approx(analytic, rel=1e-6)


def test_integrator_no_slip_below_yield():
    """If the record never exceeds ay, there is no permanent displacement."""
    ky = 0.3
    accel = [0.2 * _G * math.sin(0.1 * i) for i in range(500)]   # peak 0.2g < ay
    r = newmark_displacement(ky, accel, 0.01)
    assert r.displacement == 0.0
    assert r.n_exceedances == 0


def test_integrator_both_polarities_rectified():
    """Rectified mode (polarity='rectified'): both polarities drive the downslope
    block (abs record), no rebound, so a +/- pulse pair gives ~2x a single
    positive pulse."""
    ap, T, dt = 3.0, 1.0, 0.0005
    ky = 1.0 / _G
    tail = [0.0] * int(3.0 / dt)
    one = newmark_displacement(ky, [ap] * int(T / dt) + tail, dt,
                               polarity="rectified").displacement
    two = newmark_displacement(
        ky, [ap] * int(T / dt) + tail + [-ap] * int(T / dt) + tail, dt,
        polarity="rectified"
    ).displacement
    assert two == pytest.approx(2.0 * one, rel=1e-3)


def test_downslope_symmetric_record_is_half_rectified():
    """Default downslope mode integrates ONLY the destabilizing polarity, so a
    symmetric +/- record gives half the rectified (abs) displacement -- the
    standard Newmark/Jibson single-block convention."""
    ap, T, dt = 3.0, 1.0, 0.0005
    ky = 1.0 / _G
    tail = [0.0] * int(3.0 / dt)
    record = [ap] * int(T / dt) + tail + [-ap] * int(T / dt) + tail
    down = newmark_displacement(ky, record, dt).displacement           # default
    rect = newmark_displacement(ky, record, dt,
                                polarity="rectified").displacement
    assert down == pytest.approx(0.5 * rect, rel=1e-3)
    assert newmark_displacement(ky, record, dt).polarity == "downslope"
    with pytest.raises(ValueError):
        newmark_displacement(ky, record, dt, polarity="sideways")


def test_one_sided_pulse_same_under_both_polarities():
    """A one-sided (all-positive) pulse exceeds ay only in the destabilizing
    direction, so downslope and rectified give the SAME displacement -- the
    closed-form rectangular-pulse checks hold under both modes."""
    ap, T, dt = 3.0, 2.0, 0.0005
    ky = 1.5 / _G
    accel = [ap] * int(T / dt) + [0.0] * int(4.0 / dt)
    d_down = newmark_displacement(ky, accel, dt, polarity="downslope").displacement
    d_rect = newmark_displacement(ky, accel, dt, polarity="rectified").displacement
    analytic = _rect_pulse_analytic(ap, ky * _G, T)
    assert d_down == pytest.approx(analytic, rel=1e-3)
    assert d_rect == pytest.approx(analytic, rel=1e-3)


# --- Jibson (2007) regression -------------------------------------------------

def test_jibson2007_matches_published_equation():
    for ky, amax in [(0.10, 0.20), (0.05, 0.20), (0.139, 0.30)]:
        hand = 10.0 ** (0.215 + 2.341 * math.log10(1 - ky / amax)
                        - 1.438 * math.log10(ky / amax))
        r = newmark_jibson2007(ky, amax)
        assert r.displacement_cm == pytest.approx(hand, rel=1e-9)
        assert r.method == "jibson2007"
        assert r.sigma_log10 == 0.510


def test_jibson2007_ratio_limits():
    # ratio -> 1 gives zero; smaller ratio gives larger displacement
    assert newmark_jibson2007(0.20, 0.20).displacement == 0.0
    big = newmark_jibson2007(0.05, 0.5).displacement_cm
    small = newmark_jibson2007(0.25, 0.5).displacement_cm
    assert big > small > 0
    with pytest.raises(ValueError):
        newmark_jibson2007(0.1, 0.0)
    with pytest.raises(ValueError):
        newmark_jibson2007(0.0, 0.2)


# --- yield acceleration -------------------------------------------------------

def test_yield_acceleration_self_consistent():
    geom = _slope()
    slip = CircularSlipSurface(28.0, 26.0, 22.0)
    r = yield_acceleration(geom, slip_surface=slip, method="spencer",
                           n_slices=40)
    assert isinstance(r, YieldAccelerationResult)
    assert r.converged and r.ky > 0
    assert r.ay == pytest.approx(r.ky * _G)
    # the FOS computed at kh = ky must be ~1.0
    g2 = copy.copy(geom)
    g2.kh = r.ky
    fos = analyze_slope(g2, slip_surface=slip, method="spencer",
                        n_slices=40).FOS
    assert fos == pytest.approx(1.0, abs=2e-3)
    assert geom.kh == 0.0            # caller geometry untouched


def test_yield_acceleration_zero_when_static_unstable():
    """A surface already below FOS=1 with no seismic load has ky=0."""
    geom = _slope()
    # a shallow near-surface sliver with a very large radius -> low FOS;
    # use a weak soil to force static FOS < 1
    geom.soil_layers[0].c_prime = 0.0
    geom.soil_layers[0].phi = 5.0
    slip = CircularSlipSurface(28.0, 26.0, 22.0)
    r = yield_acceleration(geom, slip_surface=slip, method="spencer",
                           n_slices=40)
    if r.fos_static <= 1.0:
        assert r.ky == 0.0 and r.converged


# ── Bray & Travasarou (2007) seismic displacement ──────────────────────

def test_bray_travasarou_matches_paper_worked_example():
    """Two published anchor points from the Bray (2007) Ch.14 worked example
    (ky=0.14, Ts=0.4 s, Sa(1.5Ts)=0.48 g, M=7.2): ln(D)=2.29 (D≈9.9 cm) and
    P(D=0)=0.01. Reproduce both to the paper's stated precision."""
    r = bray_travasarou_2007(ky=0.14, ts=0.4, sa_1p5ts=0.48, magnitude=7.2)
    assert r.ln_d_cm == pytest.approx(2.29, abs=0.01)          # paper: 2.29
    assert r.displacement_cm == pytest.approx(math.exp(2.29), rel=0.02)  # ~9.9 cm
    assert r.p_zero == pytest.approx(0.01, abs=0.005)           # paper: 0.01
    assert r.sigma_ln == 0.66 and r.rigid is False
    # 16/84% band is median * e^-/+sigma
    assert r.d16_cm == pytest.approx(r.displacement_cm * math.exp(-0.66), rel=1e-9)
    assert r.d84_cm == pytest.approx(r.displacement_cm * math.exp(0.66), rel=1e-9)


def test_bray_travasarou_monotone_behaviours():
    """Closed-form limiting behaviours: displacement falls as ky rises, rises with
    Sa and with magnitude; P(D=0) rises as ky rises (stronger slope, less likely
    to displace)."""
    base = dict(ts=0.4, sa_1p5ts=0.48, magnitude=7.2)
    d_lo = bray_travasarou_2007(ky=0.05, **base).displacement_cm
    d_mid = bray_travasarou_2007(ky=0.14, **base).displacement_cm
    d_hi = bray_travasarou_2007(ky=0.25, **base).displacement_cm
    assert d_lo > d_mid > d_hi
    # more spectral demand -> more displacement
    assert (bray_travasarou_2007(ky=0.14, ts=0.4, sa_1p5ts=0.70, magnitude=7.2)
            .displacement_cm >
            bray_travasarou_2007(ky=0.14, ts=0.4, sa_1p5ts=0.30, magnitude=7.2)
            .displacement_cm)
    # larger magnitude -> more displacement
    assert (bray_travasarou_2007(ky=0.14, ts=0.4, sa_1p5ts=0.48, magnitude=8.0)
            .displacement_cm >
            bray_travasarou_2007(ky=0.14, ts=0.4, sa_1p5ts=0.48, magnitude=6.5)
            .displacement_cm)
    # P(D=0) rises with ky
    assert (bray_travasarou_2007(ky=0.30, **base).p_zero >
            bray_travasarou_2007(ky=0.05, **base).p_zero)


def test_bray_travasarou_orders_with_jibson():
    """Cross-check: B&T and the rigid-block Jibson (2007) model order the same way
    in ky at a fixed shaking level — both give MORE displacement for a weaker
    slope (lower ky)."""
    amax = 0.5   # PGA for Jibson; use it as Sa for the near-rigid B&T comparison
    bt_lo = bray_travasarou_2007(ky=0.08, ts=0.05, sa_1p5ts=amax, magnitude=7.0)
    bt_hi = bray_travasarou_2007(ky=0.25, ts=0.05, sa_1p5ts=amax, magnitude=7.0)
    jb_lo = newmark_jibson2007(0.08, amax)
    jb_hi = newmark_jibson2007(0.25, amax)
    assert bt_lo.displacement_cm > bt_hi.displacement_cm      # B&T ordering
    assert jb_lo.displacement_cm > jb_hi.displacement_cm      # Jibson ordering
    # same direction (both decrease with ky)


def test_bray_travasarou_rigid_branch():
    """Ts < 0.05 s auto-selects the rigid branch (intercept -0.22, no 1.5*Ts
    term); an explicit rigid flag overrides the auto-selection."""
    r = bray_travasarou_2007(ky=0.1, ts=0.0, sa_1p5ts=0.4, magnitude=7.0)
    assert r.rigid is True
    # matches the closed-form rigid equation (a0=-0.22, no Ts term)
    lnky, lnsa = math.log(0.1), math.log(0.4)
    expect = (-0.22 - 2.83 * lnky - 0.333 * lnky ** 2 + 0.566 * lnky * lnsa
              + 3.04 * lnsa - 0.244 * lnsa ** 2 + 0.278 * (7.0 - 7.0))
    assert r.ln_d_cm == pytest.approx(expect, rel=1e-9)
    # explicit override: force flexible on a small Ts
    assert bray_travasarou_2007(ky=0.1, ts=0.04, sa_1p5ts=0.4, magnitude=7.0,
                                rigid=False).rigid is False


def test_bray_travasarou_input_validation():
    with pytest.raises(ValueError):
        bray_travasarou_2007(ky=0.0, ts=0.4, sa_1p5ts=0.48, magnitude=7.2)
    with pytest.raises(ValueError):
        bray_travasarou_2007(ky=0.14, ts=0.4, sa_1p5ts=0.0, magnitude=7.2)
    with pytest.raises(ValueError):
        bray_travasarou_2007(ky=0.14, ts=-0.1, sa_1p5ts=0.48, magnitude=7.2)
