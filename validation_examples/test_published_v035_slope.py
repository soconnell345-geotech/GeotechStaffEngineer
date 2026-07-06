"""Phase E / v5.3 B2c validation — slope_stability infinite-slope method.

V-035  Slide2 #79 = Duncan & Wright (2005) Fig 14.4, cohesionless embankment,
       INFINITE-SLOPE (shallow, surface-parallel) mechanism (referee 1.44).
V-036  Slide2 #81 = Duncan & Wright (2005) Fig 14.7, cohesionless embankment,
       INFINITE-SLOPE mechanism (referee 1.15).

See validation_examples/INVENTORY.md (V-035, V-036) and RESULTS.md.
"""

import math

import pytest

from slope_stability.analysis import infinite_slope_fos

PCF = 0.157087


def test_v035_slide2_79_infinite_slope():
    """PASS: Slide2 #79 cohesionless infinite-slope FOS. 2.5:1 slope, phi'=30 ->
    tan30/tan(atan 0.4) = 1.443, matching Slide 1.443 / referee 1.44 exactly."""
    beta = math.degrees(math.atan(0.4))     # 2.5H:1V embankment
    r = infinite_slope_fos(slope_angle=beta, phi=30.0, gamma=120 * PCF, c=0.0,
                           water_condition="dry")
    assert r.FOS == pytest.approx(1.443, abs=0.003)   # Slide 1.443
    assert r.FOS == pytest.approx(1.44, abs=0.01)     # referee 1.44


def test_v036_slide2_81_infinite_slope():
    """PASS: Slide2 #81 cohesionless infinite-slope FOS. 2:1 slope, phi'=30 ->
    tan30/tan(atan 0.5) = 1.155, matching Slide 1.155 / referee 1.15 exactly."""
    beta = math.degrees(math.atan(0.5))     # 2H:1V embankment
    r = infinite_slope_fos(slope_angle=beta, phi=30.0, gamma=124 * PCF, c=0.0,
                           water_condition="dry")
    assert r.FOS == pytest.approx(1.155, abs=0.003)   # Slide 1.155
    assert r.FOS == pytest.approx(1.15, abs=0.01)     # referee 1.15


def test_v035_v036_depth_independence():
    """The cohesionless infinite-slope FOS is independent of the slip-plane
    depth (the mechanism the Duncan-Wright 'very shallow' surfaces represent)."""
    beta = math.degrees(math.atan(0.4))
    shallow = infinite_slope_fos(slope_angle=beta, phi=30.0, gamma=18.85,
                                 c=0.0, depth=0.5)
    deep = infinite_slope_fos(slope_angle=beta, phi=30.0, gamma=18.85,
                              c=0.0, depth=9.0)
    assert shallow.FOS == pytest.approx(deep.FOS, rel=1e-12)
