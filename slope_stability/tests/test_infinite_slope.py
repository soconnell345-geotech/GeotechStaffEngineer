"""Unit tests for `infinite_slope_fos` (v5.3 B2c): closed-form planar /
translational factor of safety (dry / seepage-parallel / ru / c-phi), including
the Slide2 #79/#81 cohesionless infinite-slope referees."""

import math

import pytest

from slope_stability.analysis import infinite_slope_fos


def test_dry_cohesionless_is_tanphi_over_tanbeta():
    """Dry cohesionless infinite slope: FOS = tan(phi')/tan(beta), depth-free."""
    for beta in (15.0, 21.8, 26.565, 35.0):
        for phi in (25.0, 30.0, 34.0):
            r = infinite_slope_fos(slope_angle=beta, phi=phi, gamma=19.0,
                                   c=0.0, water_condition="dry")
            expect = math.tan(math.radians(phi)) / math.tan(math.radians(beta))
            assert r.FOS == pytest.approx(expect, rel=1e-9)
    a = infinite_slope_fos(slope_angle=25, phi=30, gamma=19, c=0, depth=1.0)
    b = infinite_slope_fos(slope_angle=25, phi=30, gamma=19, c=0, depth=7.3)
    assert a.FOS == pytest.approx(b.FOS, rel=1e-12)


def test_slide2_79_and_81_referees():
    """Slide2 #79 (2.5:1, phi=30) referee FOS 1.44 and #81 (2:1, phi=30) referee
    1.15 are the dry cohesionless infinite-slope mechanism."""
    PCF = 0.157087
    b79 = math.degrees(math.atan(0.4))   # 2.5H:1V
    r79 = infinite_slope_fos(slope_angle=b79, phi=30.0, gamma=120 * PCF, c=0.0)
    assert r79.FOS == pytest.approx(1.443, abs=0.005)

    b81 = math.degrees(math.atan(0.5))   # 2H:1V
    r81 = infinite_slope_fos(slope_angle=b81, phi=30.0, gamma=124 * PCF, c=0.0)
    assert r81.FOS == pytest.approx(1.155, abs=0.005)


def test_seepage_parallel_submerged():
    """Seepage parallel with the water table at the ground: cohesionless
    FOS = (gamma'/gamma) * tan(phi')/tan(beta)."""
    g, gw, phi, beta = 20.0, 9.81, 30.0, 26.565
    r = infinite_slope_fos(slope_angle=beta, phi=phi, gamma=g, c=0.0,
                           water_condition="seepage_parallel", gamma_w=gw)
    expect = (g - gw) / g * math.tan(math.radians(phi)) / math.tan(math.radians(beta))
    assert r.FOS == pytest.approx(expect, rel=1e-9)
    # a water table below the slip plane recovers the dry value
    wet = infinite_slope_fos(slope_angle=beta, phi=phi, gamma=g, c=5.0,
                             depth=3.0, water_condition="seepage_parallel",
                             water_depth=10.0)
    dry = infinite_slope_fos(slope_angle=beta, phi=phi, gamma=g, c=5.0,
                             depth=3.0, water_condition="dry")
    assert wet.FOS == pytest.approx(dry.FOS, rel=1e-12)


def test_c_phi_and_ru_match_closed_form():
    beta, phi, gamma, c, z, ru = 20.0, 25.0, 19.0, 10.0, 3.0, 0.3
    r = infinite_slope_fos(slope_angle=beta, phi=phi, gamma=gamma, c=c,
                           depth=z, water_condition="ru", ru=ru)
    b = math.radians(beta)
    sigma_n = gamma * z * math.cos(b) ** 2
    tau = gamma * z * math.sin(b) * math.cos(b)
    u = ru * gamma * z
    expect = (c + (sigma_n - u) * math.tan(math.radians(phi))) / tau
    assert r.FOS == pytest.approx(expect, rel=1e-9)
    assert r.normal_stress == pytest.approx(sigma_n, rel=1e-9)
    assert r.pore_pressure == pytest.approx(u, rel=1e-9)
    assert r.shear_stress == pytest.approx(tau, rel=1e-9)


def test_validation_errors():
    with pytest.raises(ValueError):
        infinite_slope_fos(slope_angle=0.0, phi=30, gamma=19)
    with pytest.raises(ValueError):
        infinite_slope_fos(slope_angle=95.0, phi=30, gamma=19)
    with pytest.raises(ValueError):
        infinite_slope_fos(slope_angle=20, phi=30, gamma=-1)
    with pytest.raises(ValueError):
        infinite_slope_fos(slope_angle=20, phi=30, gamma=19, depth=0)
