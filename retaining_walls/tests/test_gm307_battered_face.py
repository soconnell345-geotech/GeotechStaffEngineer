"""GM-307: an MSE wall with a battered face, FHWA GEC-11 (FHWA-NHI-10-024)
Vol I as printed. Made-up walls; every number worked by hand from the print.

* External Kab: Eqs. 4-2, 4-3 (p. 4-15) with theta = 90 + omega, delta =
  beta; F_T at delta to the normal of the back of the zone (Fig. 4-5,
  p. 4-16), at h/3.
* Internal Ka: Eq. 4-26 (p. 4-36), the Coulomb Ka with delta = beta = 0.
* Extensible failure surface: Fig. 4-9(b) (p. 4-34) through the toe.
* A face under 10 deg is near vertical (p. 4-15); a battered face under a
  broken backslope and on the LRFD path is refused.
"""

import math

import pytest

from retaining_walls import MSEWallGeometry, analyze_mse_wall
from retaining_walls import mse
from retaining_walls.reinforcement import (
    GEOGRID_GEC11_E1_GG2, RIBBED_STEEL_STRIP_50x4)
import dataclasses

STRIPS = dataclasses.replace(RIBBED_STEEL_STRIP_50x4, horizontal_spacing=0.9)


def _geom(**kw):
    return MSEWallGeometry(wall_height=6.0, reinforcement_length=4.5,
                           reinforcement_spacing=0.6, **kw)


def _wall(geom, reinforcement, **kw):
    return analyze_mse_wall(geom, 19.0, 34.0, reinforcement,
                            phi_foundation=30.0, phi_retained=30.0,
                            gamma_retained=18.0, **kw)


def _sin(d):
    return math.sin(math.radians(d))


def test_external_kab_with_theta_by_hand():
    # Eqs. 4-2, 4-3, theta 102, phi_b 30, delta = beta = 0:
    # Gamma = (1 + sin 30 / sin 102)^2 = 2.28364;
    # Kab = sin^2 132 / (2.28364 sin^2 102 sin 102) = 0.25841
    gamma = (1 + _sin(30) / _sin(102)) ** 2
    kab = _sin(132) ** 2 / (gamma * _sin(102) ** 3)
    assert kab == pytest.approx(0.25841, abs=1e-5)
    ld = mse.backslope_loads(_geom(face_batter=12.0), 19.0, 30.0, 18.0)
    assert ld["Kab"] == pytest.approx(kab, rel=1e-9)
    assert ld["theta_deg"] == 102.0


def _hand_strip_wall():
    """omega 12, L 4.5, H 6, gamma_r 19, gamma_b 18, phi_b 30, phi_f 30."""
    kab = _sin(132) ** 2 / ((1 + _sin(30) / _sin(102)) ** 2 * _sin(102) ** 3)
    FT = 0.5 * kab * 18.0 * 36.0                 # Eq. 4-7, h = H
    FH = FT * math.cos(math.radians(-12.0))     # delta - omega = -12
    FV = FT * _sin(-12.0)                        # points up
    tw = math.tan(math.radians(12.0))
    x_FT = 4.5 + 2.0 * tw                        # back of the zone at h/3
    V1, x_V1 = 19.0 * 6.0 * 4.5, 2.25 + 3.0 * tw
    V = V1 + FV
    M_stab = V1 * x_V1 + FV * x_FT
    M_ot = FH * 2.0
    e = 2.25 - (M_stab - M_ot) / V
    sigma = V / (4.5 - 2.0 * max(e, 0.0))
    return {"FT": FT, "FH": FH, "FV": FV, "x_FT": x_FT,
            "FOS_sliding": V * math.tan(math.radians(30.0)) / FH,
            "FOS_overturning": M_stab / M_ot, "sigma_v": sigma}


def test_the_external_checks_on_a_battered_face_by_hand():
    hand = _hand_strip_wall()
    # by hand: F_T 83.72, F_H 81.89, F_V -17.41 at 4.925 m; FOS_s 3.494,
    # FOS_ot 8.521, sigma_v 110.1 (e < 0, L' = L)
    assert hand["FT"] == pytest.approx(83.72, abs=0.01)
    assert hand["FH"] == pytest.approx(81.89, abs=0.01)
    assert hand["FV"] == pytest.approx(-17.41, abs=0.01)
    assert hand["FOS_sliding"] == pytest.approx(3.494, abs=1e-3)
    assert hand["FOS_overturning"] == pytest.approx(8.521, abs=1e-3)
    assert hand["sigma_v"] == pytest.approx(110.1, abs=0.05)
    r = _wall(_geom(face_batter=12.0), STRIPS)
    assert r.FOS_sliding == pytest.approx(hand["FOS_sliding"], abs=1e-3)
    assert r.FOS_overturning == pytest.approx(hand["FOS_overturning"],
                                              abs=1e-3)
    assert r.sigma_v == pytest.approx(hand["sigma_v"], abs=0.05)
    fb = r.face_batter
    assert fb["theta_deg"] == 102.0
    assert fb["Ka"] == pytest.approx(0.2584, abs=1e-4)
    assert fb["FT_vertical_kN_per_m"] == pytest.approx(hand["FV"], abs=1e-3)
    assert fb["FT_arm_from_toe_m"] == pytest.approx(hand["x_FT"], abs=1e-3)
    assert "Fig. 4-5" in fb["basis"]


def test_internal_ka_eq_4_26_by_hand():
    # p. 4-36, theta 102, phi_r 34: sin^2 136 / [sin^3 102 (1 + sin 34 /
    # sin 102)^2] = 0.20874; as printed (no square) 0.32807; at theta 90
    # it is Eq. 4-25
    assert mse.battered_internal_Ka(34.0, 102.0) == pytest.approx(
        0.20874, abs=1e-5)
    assert mse.battered_internal_Ka_as_printed(34.0, 102.0) == \
        pytest.approx(0.32807, abs=1e-5)
    assert mse.battered_internal_Ka(34.0, 90.0) == pytest.approx(
        math.tan(math.radians(28.0)) ** 2, rel=1e-12)
    # and the printed form is not Eq. 4-25 at a vertical face
    assert mse.battered_internal_Ka_as_printed(34.0, 90.0) == \
        pytest.approx(1 - _sin(34.0), rel=1e-12)


def test_extensible_failure_surface_fig_4_9b_by_hand():
    # Fig. 4-9(b), phi 34, theta 102, beta = delta = 0: A = tan 34, C =
    # cot 46, D = tan(-12); tan(psi - phi) = 0.40403, psi = 56.00 deg
    assert mse.extensible_failure_angle(34.0, 102.0, 0.0) == pytest.approx(
        56.00, abs=0.01)
    # its own check: a vertical face gives psi = 45 + phi/2
    assert mse.extensible_failure_angle(34.0, 90.0, 0.0) == pytest.approx(
        62.0, abs=1e-9)
    r = _wall(_geom(face_batter=12.0), GEOGRID_GEC11_E1_GG2)
    top = r.internal_results[0]
    # La = (6 - 0.3)(cot 56 - tan 12) = 2.633, Le = 4.5 - 2.633 = 1.867
    La = 5.7 * (1 / math.tan(math.radians(56.0))
                - math.tan(math.radians(12.0)))
    assert top["Le_m"] == pytest.approx(round(4.5 - La, 2), abs=0.011)
    # Tmax = Ka gamma z Sv = 0.20874 x 19 x 0.3 x 0.6 = 0.714
    assert top["Tmax_kN_per_m"] == pytest.approx(0.20874 * 19 * 0.3 * 0.6,
                                                 abs=0.006)


def test_steel_keeps_0_3_H1_from_the_face():
    # Fig. 4-9(a) note: on a battered face, 0.3 H1 from the face, parallel
    flat = _wall(_geom(), STRIPS)
    batt = _wall(_geom(face_batter=12.0), STRIPS)
    assert [lv["Le_m"] for lv in batt.internal_results] == \
        [lv["Le_m"] for lv in flat.internal_results]


def test_a_near_vertical_face_is_vertical():
    flat = _wall(_geom(), STRIPS)
    near = _wall(_geom(face_batter=6.0), STRIPS)
    assert near.FOS_sliding == flat.FOS_sliding
    assert near.face_batter is None
    assert any(c["what"] == "face batter" and "vertical" in c["chosen"]
               for c in near.choices)


def test_the_choices_state_the_battered_readings():
    r = _wall(_geom(face_batter=12.0), GEOGRID_GEC11_E1_GG2)
    by = {c["what"]: c for c in r.choices}
    ka = by["internal Ka on a battered face"]
    assert "0.2087" in ka["chosen"]
    assert "0.3281" in ka["alternatives"][0]
    assert "56.00" in by["failure surface on a battered face"]["chosen"]
    assert "theta = 90 + 12" in by[
        "external earth pressure on a battered face"]["chosen"]


def test_a_battered_face_under_a_slope_lowers_the_surface():
    r = _wall(_geom(face_batter=12.0, backfill_slope=15.0),
              GEOGRID_GEC11_E1_GG2)
    assert r.backslope is not None and r.face_batter is not None
    psi = mse.extensible_failure_angle(34.0, 102.0, 15.0)
    assert psi < 56.0
    # Zp: the slope starts at the top of the face, z tan omega behind the
    # face of the level at depth z
    assert all(lv["Zp_m"] >= lv["depth_m"] for lv in r.internal_results)


def test_refused_cases():
    with pytest.raises(ValueError, match="broken backslope"):
        _wall(_geom(face_batter=12.0, backfill_slope=20.0,
                    broken_backslope_height=1.0), STRIPS)
    with pytest.raises(ValueError, match="ASD path only"):
        _wall(_geom(face_batter=12.0), STRIPS, lrfd_external=True)
    with pytest.raises(ValueError, match="outward leaning"):
        _geom(face_batter=-2.0)
