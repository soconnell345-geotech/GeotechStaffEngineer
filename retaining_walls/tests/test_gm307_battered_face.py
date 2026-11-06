"""GM-307: an MSE wall with a battered face, FHWA GEC-11 (FHWA-NHI-10-024)
Vol I as printed. Made-up walls; every number worked by hand from the print.

* External Kab: Eqs. 4-2, 4-3 (p. 4-15) with theta = 90 + omega, delta =
  beta; F_T at delta to the normal of the back of the zone (Fig. 4-5,
  p. 4-16), at h/3; under a slope, h and V2 to where the inclined back
  meets the slope, as Fig. 4-5 draws it (GM-307 review).
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


def test_a_battered_face_under_a_slope_by_hand_fig_4_5():
    # GM-307 review. Fig. 4-5 (p. 4-16) carries the back of the zone,
    # parallel to the face, up to the slope: omega 12, beta 18, L 4.5,
    # H 6: h - H = 4.5 tan 18 / (1 - tan 12 tan 18) = 1.46214 / 0.93094 =
    # 1.5706, h = 7.5706 m (Eq. 4-7's H + L tan 18 = 7.4621, for a
    # vertical wall). Eqs. 4-2, 4-3, theta 102, delta = beta = 18, phi_b
    # 30: Gamma = [1 + sqrt(sin 48 sin 12 / (sin 84 sin 120))]^2 = 2.02649,
    # Kab = sin^2 132 / (2.02649 sin^2 102 sin 84) = 0.28640.
    # F_T = 0.5 x 0.28640 x 18 x 7.5706^2 = 147.73 kN/m (Eq. 4-7 with this
    # h); at delta - omega = 6 deg: F_H = 146.93 at h/3, F_V = 15.44 at
    # 4.5 + (7.5706 / 3) tan 12 = 5.036 m.
    # V1 = 19 x 6 x 4.5 = 513 at 2.25 + 3 tan 12 = 2.888 m; V2 = 19 x 4.5
    # x 1.5706 / 2 = 67.14 (Fig. 4-3's gamma_r L (h - H) / 2) at (9 +
    # 1.5706 tan 12) / 3 + 6 tan 12 = 4.387 m.
    # V = 595.59; FOS_s = 595.59 tan 30 / 146.93 = 2.340; M_stab = 1853.7,
    # M_ot = 146.93 x 2.5235 = 370.77, FOS_ot = 5.000; e < 0, sigma_v =
    # 595.59 / 4.5 = 132.4 kPa
    tw, t = math.tan(math.radians(12.0)), math.tan(math.radians(18.0))
    dh = 4.5 * t / (1 - tw * t)
    assert 6.0 + dh == pytest.approx(7.5706, abs=1e-4)
    gamma = (1 + math.sqrt(_sin(48) * _sin(12) / (_sin(84) * _sin(120)))) ** 2
    kab = _sin(132) ** 2 / (gamma * _sin(102) ** 2 * _sin(84))
    assert kab == pytest.approx(0.28640, abs=1e-5)
    FT = 0.5 * kab * 18.0 * (6.0 + dh) ** 2
    assert FT == pytest.approx(147.73, abs=0.01)
    FH, FV = FT * math.cos(math.radians(6.0)), FT * _sin(6.0)
    x_FT = 4.5 + (6.0 + dh) / 3.0 * tw
    V2, x_V2 = 19.0 * 4.5 * dh / 2.0, (9.0 + dh * tw) / 3.0 + 6.0 * tw
    assert V2 == pytest.approx(67.14, abs=0.01)
    V = 513.0 + V2 + FV
    M_stab = 513.0 * (2.25 + 3.0 * tw) + V2 * x_V2 + FV * x_FT
    M_ot = FH * (6.0 + dh) / 3.0
    assert V * math.tan(math.radians(30.0)) / FH == pytest.approx(2.340,
                                                                  abs=5e-4)
    assert M_stab / M_ot == pytest.approx(5.000, abs=5e-4)
    assert 2.25 - (M_stab - M_ot) / V < 0
    r = _wall(_geom(face_batter=12.0, backfill_slope=18.0), STRIPS)
    ld = r.external_terms["backslope_loads"]
    assert ld["h"] == pytest.approx(6.0 + dh, rel=1e-12)
    assert ld["FT"] == pytest.approx(FT, rel=1e-12)
    assert ld["V2"] == pytest.approx(V2, rel=1e-12)
    assert ld["x_V2"] == pytest.approx(x_V2, rel=1e-12)
    assert ld["x_FT"] == pytest.approx(x_FT, rel=1e-12)
    assert r.FOS_sliding == pytest.approx(2.340, abs=1e-3)
    assert r.FOS_overturning == pytest.approx(5.000, abs=1e-3)
    assert r.sigma_v == pytest.approx(V / 4.5, abs=0.05)
    assert r.backslope["h_m"] == pytest.approx(7.571, abs=1e-3)
    assert r.face_batter["FT_kN_per_m"] == pytest.approx(147.735, abs=1e-3)
    # the choice: Fig. 4-5's h, Eq. 4-7's (a vertical wall's) beside it
    ch = {c["what"]: c for c in r.choices}[
        "height h of the thrust plane on a battered face under a slope"]
    assert "7.571" in ch["chosen"] and "147.7" in ch["chosen"]
    assert "Fig. 4-5" in ch["basis"]
    assert "7.462" in ch["alternatives"][0]
    assert "143.5" in ch["alternatives"][0]
    assert "vertical wall" in ch["alternatives"][0]


def test_fig_4_5_h_leaves_level_and_vertical_walls_as_they_were():
    # h = H on a level battered face; Eq. 4-7 on a vertical one
    level = mse.backslope_geometry(_geom(face_batter=12.0))
    assert level["h"] == 6.0 and level["V2_area"] == 0.0
    vert = mse.backslope_geometry(_geom(backfill_slope=18.0))
    t = math.tan(math.radians(18.0))
    assert vert["h"] == 6.0 + 4.5 * t and vert["h_rule"] == "Eq. 4-7"
    assert vert["V2_area"] == pytest.approx(4.5 * 4.5 * t / 2, rel=1e-12)
    assert vert["V2_arm"] == pytest.approx(3.0, rel=1e-12)
    # a face under 10 deg is vertical: Eq. 4-7
    near = mse.backslope_geometry(_geom(face_batter=6.0, backfill_slope=18.0))
    assert near["h"] == vert["h"]
    r = _wall(_geom(face_batter=12.0), STRIPS)
    assert not any("thrust plane" in c["what"] for c in r.choices)


def test_refused_cases():
    with pytest.raises(ValueError, match="broken backslope"):
        _wall(_geom(face_batter=12.0, backfill_slope=20.0,
                    broken_backslope_height=1.0), STRIPS)
    with pytest.raises(ValueError, match="ASD path only"):
        _wall(_geom(face_batter=12.0), STRIPS, lrfd_external=True)
    with pytest.raises(ValueError, match="outward leaning"):
        _geom(face_batter=-2.0)
