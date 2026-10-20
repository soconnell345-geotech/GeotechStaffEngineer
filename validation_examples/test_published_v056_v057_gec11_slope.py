"""Phase E validation — GEC-11 MSE walls with a sloping backfill (V-056, V-057).

Source: FHWA GEC-11 Vol 2 (FHWA-NHI-10-025), Appendix E, read from the PDF
(geotech-references/docs/GEC 11 VOL 2.pdf):
  - Example E3 (V-056): segmental precast panel MSE wall with a sloping
    backfill surcharge (2H:1V slope that continues), ribbed steel strips.
    PDF pp. 273-292 (printed E3-1 to E3-20): Kaf p. 276 (E3-4); unfactored
    loads Tables E3-4.3/4.4 p. 278 (E3-6); sliding Table E3-6.1 p. 280
    (E3-8); eccentricity Table E3-6.2 p. 281 (E3-9); bearing Table E3-6.3
    p. 283 (E3-11); internal Steps 7.1-7.5 pp. 284-290 (E3-12 to E3-18).
  - Example E1 (V-057): MBW unit faced MSE wall with a broken backslope
    (2H:1V, 9 ft high, crest at the end of the reinforcement) and traffic
    on the level ground behind the crest, geogrid. PDF pp. 243-262: Kab
    p. 246 (E1-4); loads p. 247 (E1-5); sliding p. 249 (E1-7); eccentricity
    pp. 250-251 (E1-8/9); bearing pp. 252-253 (E1-10/11); S_eq p. 255 (E1-13).
Method: GEC-11 Vol 1 (FHWA-NHI-10-024) §4.4.4, §4.4.6, §4.4.7 — see
retaining_walls/DESIGN.md "Sloping backfill (GM-112)".

Everything runs through the high-level module path (MSEWallGeometry with
``backfill_slope`` / ``broken_backslope_height`` into
``check_external_stability_lrfd`` / ``check_external_stability`` /
``check_internal_stability``); no hand-assembled factors.

Units: the examples are US customary; the module is SI.
  1 ft = 0.3048 m, 1 pcf = 0.157087 kN/m3, 1 ksf = 47.88 kPa,
  1 kip/ft = 14.594 kN/m.
"""

import math

import pytest

from retaining_walls.geometry import MSEWallGeometry
from retaining_walls.mse import (
    analyze_mse_wall, check_external_stability,
    check_external_stability_lrfd, check_internal_stability,
)
from retaining_walls.reinforcement import Reinforcement, GEOGRID_UX1600

FT = 0.3048
PCF = 0.157087
KSF = 47.88
KFT = 14.594
GAMMA = 125 * PCF               # every fill in both examples
BETA = math.degrees(math.atan(0.5))   # 2H:1V (E3 writes 26.56 deg)


# ── V-056: Example E3, a slope that continues ──────────────────────────────

def _e3_geom():
    # H = He 28 ft + embedment 2 ft = 30 ft; L = 0.8H = 24 ft; Sv 2.5 ft
    # gives E3's twelve levels at Z = 1.25, 3.75, ..., 28.75 ft.
    return MSEWallGeometry(wall_height=30 * FT, reinforcement_length=24 * FT,
                           reinforcement_spacing=2.5 * FT,
                           backfill_slope=BETA)


def _e3_lrfd():
    # Reinforced phi_r 34; retained phi_f 30; foundation phi_fd 30;
    # bearing resistances 10.50 ksf (strength), 7.50 ksf (service).
    return check_external_stability_lrfd(
        _e3_geom(), GAMMA, 34.0, 30.0, phi_retained=30.0,
        bearing_resistance_strength=10.5 * KSF,
        bearing_resistance_service=7.5 * KSF)


def test_v056_kaf_and_unfactored_loads():
    """E3 p. E3-4: Kaf = 0.537 (Gamma = 1.563). Tables E3-4.3/4.4:
    h = 42.00 ft; V1 = 90.00, V2 = 18.00, FTV = 26.48, FTH = 52.95 k/ft."""
    r = _e3_lrfd()
    b, u = r["backslope"], r["unfactored_kN_m"]
    assert b["Kab"] == pytest.approx(0.537, abs=0.001)
    assert b["h_m"] / FT == pytest.approx(42.00, abs=0.01)
    assert u["V1"] / KFT == pytest.approx(90.00, rel=0.002)
    assert u["V2"] / KFT == pytest.approx(18.00, rel=0.002)
    # E3 carries Kaf = 0.537 rounded; the module's 0.5367 is 0.06 % lower.
    assert u["FT_V"] / KFT == pytest.approx(26.48, rel=0.002)
    assert u["FT_H"] / KFT == pytest.approx(52.95, rel=0.002)


def test_v056_sliding_cdr():
    """Table E3-6.1: CDR 1.35 (Str I max), 1.60 (Str I min), and 1.07 for
    the critical values (EV 1.00 on V1 + V2, EH 1.50 on both components)."""
    s = _e3_lrfd()["sliding"]
    assert s["CDR_strength_max"] == pytest.approx(1.35, abs=0.005)
    assert s["CDR_strength_min"] == pytest.approx(1.60, abs=0.005)
    assert s["CDR_critical"] == pytest.approx(1.07, abs=0.005)
    assert s["CDR_governing"] == pytest.approx(1.07, abs=0.005)


def test_v056_eccentricity():
    """Table E3-6.2: eL 2.90 ft (Str I max), 3.81 ft (critical); L/4 =
    6.00 ft. Str I min (EV 1.00, EH 0.90) 2.35 ft, by giving those factors
    as the maximum ones."""
    e = _e3_lrfd()["eccentricity"]
    assert e["eL_strength_max_m"] / FT == pytest.approx(2.90, abs=0.01)
    assert e["eL_critical_m"] / FT == pytest.approx(3.81, abs=0.01)
    assert e["e_limit_m"] / FT == pytest.approx(6.00, abs=1e-6)
    assert e["passes"] is True
    e_min = check_external_stability_lrfd(
        _e3_geom(), GAMMA, 34.0, 30.0, phi_retained=30.0,
        gamma_EV_max=1.0, gamma_EH_max=0.9)["eccentricity"]
    # E3 prints 2.35; its own columns give 9.65 ft -> 12 - 9.65 = 2.35,
    # the module 2.344 (Kaf 0.5367 against 0.537).
    assert e_min["eL_strength_max_m"] / FT == pytest.approx(2.35, abs=0.01)


def test_v056_bearing():
    """Table E3-6.3: sigma_v 10.19 ksf, B' 18.20 ft, CDR 1.03 (Str I max);
    Service I 7.16 ksf, CDR 1.05; critical values 9.02 ksf, CDR 1.16."""
    b = _e3_lrfd()["bearing"]
    assert b["sigma_v_strength_max_kPa"] / KSF == pytest.approx(10.19, abs=0.01)
    assert b["B_eff_strength_max_m"] / FT == pytest.approx(18.20, abs=0.01)
    assert b["CDR_strength"] == pytest.approx(1.03, abs=0.005)
    assert b["sigma_v_service_kPa"] / KSF == pytest.approx(7.16, abs=0.01)
    assert b["CDR_service"] == pytest.approx(1.05, abs=0.005)
    assert b["sigma_v_critical_kPa"] / KSF == pytest.approx(9.02, abs=0.01)
    assert b["CDR_critical"] == pytest.approx(1.16, abs=0.005)
    # Str I min (EV 1.00, EH 0.90), given as the maximum factors: 6.83 ksf,
    # CDR 1.54
    b_min = check_external_stability_lrfd(
        _e3_geom(), GAMMA, 34.0, 30.0, phi_retained=30.0,
        bearing_resistance_strength=10.5 * KSF,
        gamma_EV_max=1.0, gamma_EH_max=0.9)["bearing"]
    assert b_min["sigma_v_strength_max_kPa"] / KSF == pytest.approx(
        6.83, abs=0.01)
    assert b_min["CDR_strength"] == pytest.approx(1.54, abs=0.005)


def test_v056_asd_path_is_the_service_column():
    """The ASD check is unfactored: its resultant is E3's Service I column
    (Table E3-6.3): sum V 134.48 k/ft, MR 2003.44 and MO 741.35 k-ft/ft,
    eL 2.61 ft, sigma_v 7.16 ksf. Sliding FOS = 134.48 tan 30 / 52.95,
    overturning FOS = 2003.44 / 741.35, by hand from those columns."""
    out = check_external_stability(_e3_geom(), GAMMA, 34.0, GAMMA, 30.0,
                                   phi_retained=30.0)
    assert out["eccentricity_m"] / FT == pytest.approx(2.61, abs=0.01)
    assert out["sigma_v_kPa"] / KSF == pytest.approx(7.16, abs=0.01)
    assert out["FOS_sliding"] == pytest.approx(
        134.48 * math.tan(math.radians(30)) / 52.95, abs=0.005)
    assert out["FOS_overturning"] == pytest.approx(2003.44 / 741.35,
                                                   abs=0.005)


def _e3_internal():
    strip = Reinforcement(name="E3 ribbed strip 50x4 mm",
                          type="metallic_strip", Tallowable=1000.0,
                          width=0.050, Fy=448.0, thickness=0.004)
    return check_internal_stability(_e3_geom(), GAMMA, 34.0, strip)


def test_v056_internal_level4_tension():
    """E3 Step 7.3, Level 4 (Z = 8.75 ft), Strength I max: sigma_H =
    Kr gamma_r (Z + S) gamma_EV = 0.99 ksf with S = 1/2 (0.7H) tan(beta),
    sigma_2 = 0.656 ksf, Ka from phi_r with beta = 0. The module's Tmax is
    unfactored per metre at the level itself; times 1.35 / Sv it is
    sigma_H. Kr = 0.340 + (20 - 8.75)(0.481 - 0.340)/20 = 0.419 (E3's own
    interpolation; the module's strip Kr/Ka reaches 1.2 at 20 ft as E3's
    does)."""
    lv = _e3_internal()[3]
    assert lv["depth_m"] / FT == pytest.approx(8.75, abs=0.01)
    assert lv["Kr_Ka"] * 0.2827 == pytest.approx(0.419, abs=0.001)
    sigma_H = 1.35 * lv["Tmax_kN_per_m"] / (2.5 * FT)
    assert sigma_H / KSF == pytest.approx(0.99, rel=0.01)
    # Tmax = 12.37 k for a 5-ft panel (Atrib 12.50 ft2) -> per ft of wall
    assert 1.35 * lv["Tmax_kN_per_m"] / KFT == pytest.approx(12.37 / 5,
                                                              rel=0.01)


def test_v056_internal_level4_pullout_geometry():
    """E3 Step 7.5, Level 4: H1 = 35.29 ft, La = 0.3 H1 = 10.59 ft, Le =
    13.41 ft, Zp-ave = Z + 1/2 tan(beta) (La + L) = 17.40 ft (sigma_v =
    gamma_r Zp = 2.175 ksf, unfactored), F* = 1.420."""
    lv = _e3_internal()[3]
    assert lv["La_m"] / FT == pytest.approx(10.59, abs=0.05)
    assert lv["Le_m"] / FT == pytest.approx(13.41, abs=0.05)
    assert lv["Zp_m"] / FT == pytest.approx(17.40, abs=0.05)
    assert GAMMA * lv["Zp_m"] / KSF == pytest.approx(2.175, rel=0.005)
    # F*: 2.0 at Z = 0 to tan(34) at 20 ft, in the module as in E3
    assert lv["F_star"] == pytest.approx(1.420, abs=0.001)


def test_v056_analyze_mse_wall_reports_the_basis():
    r = analyze_mse_wall(_e3_geom(), GAMMA, 34.0, GEOGRID_UX1600,
                         phi_foundation=30.0, phi_retained=30.0,
                         lrfd_external=True)
    b = r.to_dict()["backslope"]
    assert b["Kab"] == pytest.approx(0.537, abs=0.001)
    # Figure E3-2b: sigma_2 = 1/2 (0.7 * 30 ft) tan(26.56) (0.125) = 0.656 ksf
    assert b["sigma_2_kPa"] / KSF == pytest.approx(0.656, abs=0.002)
    assert b["H1_m"] / FT == pytest.approx(35.29, abs=0.01)
    assert r.external_lrfd["sliding"]["CDR_critical"] == pytest.approx(
        1.07, abs=0.005)


# ── V-057: Example E1, a broken backslope with traffic behind the crest ─────

def _e1_lrfd(traffic_vertical_resists=True):
    # H = He 18 ft + 2 ft = 20 ft; L = 0.9H = 18 ft; 2H:1V slope 9 ft high,
    # crest at the end of the reinforcement; traffic heq 2 ft, q = 250 psf on
    # the level ground behind the crest; foundation phi 30, retained phi 30.
    # E1 counts the traffic thrust's vertical component FV2 as resistance
    # (pp. E1-7 to E1-9): traffic_vertical_resists=True. The module's
    # default follows Vol I p. 4-22 instead (test_v057_default_basis).
    geom = MSEWallGeometry(wall_height=20 * FT, reinforcement_length=18 * FT,
                           backfill_slope=BETA,
                           broken_backslope_height=9 * FT,
                           surcharge=0.25 * KSF)
    return check_external_stability_lrfd(
        geom, GAMMA, 34.0, 30.0, phi_retained=30.0,
        bearing_resistance_strength=10.5 * KSF,
        traffic_vertical_resists=traffic_vertical_resists)


def test_v057_kab_and_loads():
    """E1 pp. E1-3 to E1-5: I = arctan(9/40) = 12.7 deg, h = 29 ft,
    Kab = 0.360 (Gamma = 2.133); F1 = 18.92, FH1 = 18.46, FV1 = 4.16,
    F2 = 2.61, FH2 = 2.55, FV2 = 0.57, V1 = 45.00, V2 = 10.12 k/ft."""
    r = _e1_lrfd()
    b, u = r["backslope"], r["unfactored_kN_m"]
    assert b["interface_angle_deg"] == pytest.approx(12.7, abs=0.05)
    assert b["h_m"] / FT == pytest.approx(29.0, abs=0.01)
    assert b["Kab"] == pytest.approx(0.360, abs=0.001)
    # E1 carries Kab = 0.360 rounded; the module's 0.3604 is 0.1 % higher.
    for key, pub in (("FT", 18.92), ("FT_H", 18.46), ("FT_V", 4.16),
                     ("F2", 2.61), ("F2_H", 2.55), ("F2_V", 0.57),
                     ("V1", 45.00), ("V2", 10.12)):
        assert u[key] / KFT == pytest.approx(pub, rel=0.002, abs=0.01), key
    assert u["Vs"] == 0.0          # no traffic over the reinforced zone


def test_v057_sliding_cdr():
    """E1 Step 6.1: Pd = 1.50 FH1 + 1.75 FH2 = 32.15 k/ft; Rr = [1.00 (V1 +
    V2) + 1.50 FV1 + 1.75 FV2] tan 30 = 36.0 k/ft; CDR = 1.12."""
    assert _e1_lrfd()["sliding"]["CDR_critical"] == pytest.approx(
        1.12, abs=0.005)


def test_v057_eccentricity():
    """E1 Step 6.2: e = 237.00 / 62.36 = 3.80 ft <= L/4 = 4.50 ft."""
    e = _e1_lrfd()["eccentricity"]
    assert e["eL_critical_m"] / FT == pytest.approx(3.80, abs=0.01)
    assert e["e_limit_m"] / FT == pytest.approx(4.50, abs=1e-6)
    assert e["passes"] is True


def test_v057_bearing():
    """E1 Step 6.3: eB = 226.36 / 81.65 = 2.77 ft; sigma_V-F = 81.65 /
    (18 - 2 x 2.77) = 6.55 ksf; CDR = 10.50 / 6.55 = 1.60."""
    b = _e1_lrfd()["bearing"]
    assert b["eL_strength_max_m"] / FT == pytest.approx(2.77, abs=0.01)
    assert b["sigma_v_strength_max_kPa"] / KSF == pytest.approx(6.55,
                                                                abs=0.01)
    assert b["CDR_strength"] == pytest.approx(1.60, abs=0.005)


def test_v057_e2_nominal_bearing_resistance():
    """GEC-11 Vol II Example E2 (pp. E2-3 to E2-5) works the bearing
    resistance under E1's wall: qn by Eq. 4-22 with Nγ from Table 4-6 for
    φ'f = 30 deg, γf = 125 pcf, L' = 18 ft - 2 (2.77 ft) = 12.46 ft:
    qn = 17.44 ksf; qR = 0.65 (17.44) = 11.34 ksf (Eq. 4-23). GM-179:
    gamma_foundation given in place of bearing_resistance_strength."""
    geom = MSEWallGeometry(wall_height=20 * FT, reinforcement_length=18 * FT,
                           backfill_slope=BETA,
                           broken_backslope_height=9 * FT,
                           surcharge=0.25 * KSF)
    b = check_external_stability_lrfd(
        geom, GAMMA, 34.0, 30.0, phi_retained=30.0, gamma_foundation=GAMMA,
        traffic_vertical_resists=True)["bearing"]
    assert b["B_eff_strength_max_m"] / FT == pytest.approx(12.46, abs=0.01)
    assert b["Ngamma"] == pytest.approx(22.4, abs=0.01)
    assert b["qn_strength_max_kPa"] / KSF == pytest.approx(17.44, abs=0.01)
    assert b["qR_strength_max_kPa"] / KSF == pytest.approx(11.34, abs=0.01)
    # by hand: CDR = 11.34 / 6.55 = 1.73
    assert b["CDR_strength"] == pytest.approx(1.73, abs=0.005)
    assert "Eq. 4-22" in b["resistance_basis"]
    with pytest.raises(ValueError, match="not both"):
        check_external_stability_lrfd(
            geom, GAMMA, 34.0, 30.0, gamma_foundation=GAMMA,
            bearing_resistance_strength=10.5 * KSF)


def test_v057_default_basis():
    """The module's default leaves FV2 (traffic) out of the sliding and
    eccentricity resistance: GEC-11 Vol I p. 4-22 includes loads that
    increase sliding resistance "only if those loads are permanent". On E1,
    by hand from E1's own loads: Rr = [1.00 (45.00 + 10.12) + 1.50 (4.16)]
    tan 30 = 35.43 k/ft over Pd 32.15, CDR 1.10; e = (267.8 + 64.70 - 30.36
    - 56.16) / 61.36 = 4.01 ft (<= 4.50). Bearing counts FV2 either way
    (p. 4-26): sigma_V-F 6.55 ksf, as E1."""
    r = _e1_lrfd(traffic_vertical_resists=False)
    assert r["sliding"]["CDR_critical"] == pytest.approx(1.10, abs=0.005)
    assert r["eccentricity"]["eL_critical_m"] / FT == pytest.approx(
        4.01, abs=0.01)
    assert r["eccentricity"]["passes"] is True
    assert r["bearing"]["sigma_v_strength_max_kPa"] / KSF == pytest.approx(
        6.55, abs=0.01)
    assert r["backslope"]["traffic_vertical_resists"] is False
    assert "p. 4-22" in r["backslope"]["traffic_basis"]
    assert "Example E1" in _e1_lrfd()["backslope"]["traffic_basis"]


def test_v057_internal_equivalent_height():
    """E1 Step 7.3 (p. E1-13): S_eq = 1/2 (0.7 x 20 ft) tan 26.6 = 3.51 ft,
    under the 9 ft slope height; no traffic over the reinforced zone."""
    geom = MSEWallGeometry(wall_height=20 * FT, reinforcement_length=18 * FT,
                           backfill_slope=BETA,
                           broken_backslope_height=9 * FT,
                           surcharge=0.25 * KSF)
    r = analyze_mse_wall(geom, GAMMA, 34.0, GEOGRID_UX1600,
                         phi_foundation=30.0, phi_retained=30.0)
    # E1 rounds beta to 26.6 deg (tan 0.5008); 2H:1V gives 3.50 ft
    assert r.backslope["S_eq_m"] / FT == pytest.approx(3.51, abs=0.02)
