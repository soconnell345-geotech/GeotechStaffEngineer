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
    pp. 250-251 (E1-8/9); bearing pp. 252-253 (E1-10/11); S_eq p. 255 (E1-13);
    TMAX Table E1-7.2 p. 256 (E1-14); Tal and Tr Table E1-7.3 p. 257
    (E1-15); rupture CDRs Table E1-7.4 p. 258 (E1-16) (GM-268).
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
    Kr_Ka_ratio, Tmax_at_level, analyze_mse_wall, backslope_geometry,
    check_external_stability, check_external_stability_lrfd,
    check_internal_stability, pullout_resistance, rankine_Ka,
    rupture_allowable,
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


KIP = 4.4482216
B_STRIP = 1.969 * 0.0254        # E3's 50 mm strip, b = 1.969 in = 0.164 ft
TN_STRIP = 13.00 * KIP          # E3 Step 7.4 (p. E3-16): Tn = 13.00 k/strip


def _e3_internal(Cu_backfill=7.0, strips_per_panel=2):
    # E3 Step 2: reinforced backfill Cu = 7.0 (p. E3-2); Step 7.1 (p. E3-12)
    # F* = 1.2 + log10(7.0) = 2.045 > 2.000, use 2.000 (GM-246). GM-256:
    # E3's strips at their own spacing, Sh = 5 ft panel / Ng strips (Table
    # E3-7.3, p. E3-19: Sh 2.50 ft at Levels 1-8, 1.67 ft at 9-12), and Tal
    # per metre of strip width = Tn / b, so FOS_rupture is Tn Rc / Tmax
    strip = Reinforcement(name="E3 ribbed strip 50x4 mm",
                          type="metallic_strip", Tallowable=TN_STRIP / B_STRIP,
                          width=B_STRIP, Fy=448.0, thickness=0.004,
                          horizontal_spacing=5.0 * FT / strips_per_panel)
    return check_internal_stability(_e3_geom(), GAMMA, 34.0, strip,
                                    Cu_backfill=Cu_backfill)


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
    # F*: 2.0 at Z = 0 (Cu 7.0, at the 2.0 maximum) to tan(34) at 20 ft,
    # in the module as in E3
    assert lv["F_star"] == pytest.approx(1.420, abs=0.001)


def test_v056_internal_strip_F_star_by_level():
    """E3 Table E3-7.3 (p. E3-19): F* 1.917 at Level 1 (Z = 1.25 ft) and
    1.751 at Level 2 (Z = 3.75 ft), from 2.000 at Z = 0 (Cu 7.0) to 0.675
    at 20 ft. Without the backfill's Cu the module takes Cu = 4, F* 1.8 at
    the top, as GEC-11 Vol I p. 3-20 says to when it is unknown (GM-246)."""
    levels = _e3_internal()
    assert levels[0]["F_star"] == pytest.approx(1.917, abs=0.001)
    assert levels[1]["F_star"] == pytest.approx(1.751, abs=0.001)
    unknown = _e3_internal(None)
    top = 1.2 + math.log10(4.0)
    assert unknown[3]["F_star"] == pytest.approx(
        0.675 + (20.0 - 8.75) * (top - 0.675) / 20.0, abs=0.002)
    assert unknown[3]["Pr_kN_per_m"] < levels[3]["Pr_kN_per_m"]


#: GEC-11 Vol II Table E3-7.3 (p. E3-19), Strength I (max): Level: (Tmax
#: k per 5 ft panel, factored; phi_p Pr k/strip; Ng strips per panel).
E3_TABLE_7_3 = {
    1: (6.46, 9.40, 2), 2: (8.63, 10.76, 2), 3: (10.58, 11.70, 2),
    4: (12.35, 12.23, 2), 5: (13.96, 12.36, 2), 6: (15.40, 12.70, 2),
    7: (16.59, 13.05, 2), 8: (17.60, 12.76, 2), 9: (18.98, 13.33, 3),
    10: (20.77, 15.50, 3), 11: (22.56, 17.79, 3), 12: (24.36, 20.24, 3)}


@pytest.mark.parametrize("level", sorted(E3_TABLE_7_3))
def test_v056_internal_pullout_and_tension_by_level(level):
    """GM-256: E3's pullout and tension at every level of Table E3-7.3.
    E3 Step 7.5 (p. E3-18): Pr = (1.0)(1.420)(2)(0.164 ft)(13.41 ft)(2.175
    ksf) = 13.58 k/strip at Level 4, b = 0.164 ft; Step 7.4 (p. E3-16):
    phi_t Tn = 0.75 x 13.00 = 9.75 k/strip; Ng strips per 5 ft panel. The
    module works per metre of wall, Pr and Tal times Rc = b/Sh (GEC-11 Vol
    I Eq. 4-35, p. 4-49; p. 4-47), unfactored: its Pr times Sh
    is E3's Pr per strip, and its FOS times phi / gamma_EV (0.90 / 1.35
    pullout, 0.75 / 1.35 tension) is E3's Ng phi R / Tmax. Before GM-256 it
    took Rc = 1: FOS_pullout at Level 4 was 45.2 (Ng phi Pr / Tmax 30.2)
    where E3 prints 2 x 12.23 / 12.35 = 1.98."""
    Tmax, phi_Pr, Ng = E3_TABLE_7_3[level]
    lv = _e3_internal(strips_per_panel=Ng)[level - 1]
    Sh = 5.0 * FT / Ng
    assert lv["Rc"] == pytest.approx(B_STRIP / Sh, abs=1e-4)
    assert 0.90 * lv["Pr_kN_per_m"] * Sh / KIP == pytest.approx(phi_Pr,
                                                                rel=0.003)
    assert lv["FOS_pullout"] * 0.90 / 1.35 == pytest.approx(
        Ng * phi_Pr / Tmax, rel=0.004)
    assert lv["FOS_rupture"] * 0.75 / 1.35 == pytest.approx(
        Ng * 0.75 * 13.00 / Tmax, rel=0.004)


def test_v056_internal_by_level_in_one_run():
    """GM-261: E3 in ONE run, its spacing per level as Table E3-7.3 (p.
    E3-19) prints it, Sh = wp / Ng: 2.50 ft at Levels 1 to 8 and 1.67 ft at
    Levels 9 to 12. Every level's pullout and tension as the per-Ng runs
    above, and each level the same as the run at its own spacing. One
    spacing for the wall (2.50 ft, the run before GM-261) leaves Levels 9
    to 12 at two thirds of their resistance."""
    Sh = [5.0 * FT / E3_TABLE_7_3[i][2] for i in sorted(E3_TABLE_7_3)]
    strip = Reinforcement(name="E3 ribbed strip 50x4 mm",
                          type="metallic_strip", Tallowable=TN_STRIP / B_STRIP,
                          width=B_STRIP, Fy=448.0, thickness=0.004,
                          horizontal_spacing=Sh)
    levels = check_internal_stability(_e3_geom(), GAMMA, 34.0, strip,
                                      Cu_backfill=7.0)
    assert len(levels) == 12
    one = _e3_internal(strips_per_panel=2)
    for level, (Tmax, phi_Pr, Ng) in E3_TABLE_7_3.items():
        lv = levels[level - 1]
        assert lv == _e3_internal(strips_per_panel=Ng)[level - 1]
        assert lv["Rc"] == pytest.approx(B_STRIP / (5.0 * FT / Ng), abs=1e-4)
        assert 0.90 * lv["Pr_kN_per_m"] * Sh[level - 1] / KIP == \
            pytest.approx(phi_Pr, rel=0.003)
        assert lv["FOS_pullout"] * 0.90 / 1.35 == pytest.approx(
            Ng * phi_Pr / Tmax, rel=0.004)
        assert lv["FOS_rupture"] * 0.75 / 1.35 == pytest.approx(
            Ng * 0.75 * 13.00 / Tmax, rel=0.004)
        assert lv["FOS_rupture"] == pytest.approx(
            one[level - 1]["FOS_rupture"] * Ng / 2, rel=2e-3)
    # Level 12: Ng phi_t Tn / Tmax = 3 x 9.75 / 24.36 = 1.20 (E3); at one
    # spacing of 2.50 ft, 2 x 9.75 / 24.36 = 0.80
    assert levels[-1]["FOS_rupture"] * 0.75 / 1.35 == pytest.approx(
        1.20, abs=0.005)
    assert one[-1]["FOS_rupture"] * 0.75 / 1.35 == pytest.approx(
        0.80, abs=0.005)


def test_v056_e5_level4_pullout_per_strip():
    """GEC-11 Vol II Example E5, Level 4 (p. E5-28): Pr = (1.0)(0.832)(2)
    (0.164 ft)(15.06 ft)(2.20 ksf) = 9.04 k/strip. E5 is a bridge abutment
    whose footing loads the module does not take, so only the per-strip
    pullout is checked: per metre of strip width (Rc = 1) times b."""
    Pr = pullout_resistance(z=17.62 * FT, gamma_backfill=GAMMA,
                            Le=15.06 * FT, F_star=0.832, Rc=1.0)
    assert Pr * B_STRIP / KIP == pytest.approx(9.04, rel=0.003)


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
    # GM-256: E1's own pullout factors, alpha 0.8 for these geogrids
    assert {lv["alpha"] for lv in r.internal_results} == {0.8}


#: GEC-11 Vol II Table E1-7.5 (p. E1-17), per layer: Z (ft), available Le
#: (ft), TMAX (lb/ft), Zp (ft), pullout CDR.
E1_TABLE_7_5 = [
    (0.67, 7.72, 346, 7.74, 14.0), (2.67, 8.78, 590, 9.47, 11.4),
    (4.67, 9.84, 781, 11.21, 11.4), (6.67, 10.91, 972, 12.94, 11.8),
    (8.67, 11.97, 1163, 14.68, 12.2), (10.67, 13.04, 1354, 16.41, 12.8),
    (12.67, 14.10, 1545, 18.14, 13.4), (14.67, 15.16, 1736, 19.88, 14.1),
    (16.67, 16.23, 1927, 21.61, 14.7), (18.67, 17.29, 1366, 23.35, 23.9),
    (19.33, 17.64, 1091, 23.92, 31.1)]


@pytest.mark.parametrize("z, Le, Tmax, Zp, cdr", E1_TABLE_7_5)
def test_v057_pullout_cdr_by_layer(z, Le, Tmax, Zp, cdr):
    """GM-256: E1 Step 7.8 (p. E1-16) takes F* 0.45, 'alpha = Scale
    correction factor, = 0.8 for these geogrids', C = 2 and Rc = 1, and
    Table E1-7.5 (p. E1-17) gives each layer's pullout CDR, phi Pr / TMAX
    with phi 0.90, from its own Zp and Le. The module's built-in geogrid
    takes GEC-11 Vol I Table 3-6's alpha 0.8 for a geogrid (p. 3-17) and
    its own F*, 0.67 tan 34 = 0.452, and reproduces every CDR within 1.2 %
    (F* 0.452 against 0.45 is 0.4 % of it). At the alpha 1.0 every
    geosynthetic took before GM-256, each was 25 % high."""
    geom = MSEWallGeometry(wall_height=20 * FT, reinforcement_length=18 * FT,
                           backfill_slope=BETA,
                           broken_backslope_height=9 * FT,
                           surcharge=0.25 * KSF)
    level = analyze_mse_wall(geom, GAMMA, 34.0, GEOGRID_UX1600,
                             phi_foundation=30.0,
                             phi_retained=30.0).internal_results[0]
    assert level["alpha"] == 0.8
    Pr = pullout_resistance(Zp * FT, GAMMA, Le * FT, level["F_star"],
                            alpha_pullout=level["alpha"])
    assert 0.90 * Pr / (Tmax * KFT / 1000) == pytest.approx(cdr, rel=0.012)


# ── V-057: E1's geogrid rupture CDRs (GM-268) ──────────────────────────────

#: GEC-11 Vol II Table E1-7.3 (p. E1-15): Tal = Tult / (RFCR x RFD x RFID),
#: Eq. 3-12, and Tr = phi Tal with phi 0.90 (Eq. 4-33): grade, Tult, RFID,
#: Tal and Tr (lb/ft).
E1_TABLE_7_3 = {"GG-I": (3000, 1.3, 1085, 976),
                "GG-II": (6000, 1.3, 2169, 1952),
                "GG-III": (9000, 1.2, 3525, 3173)}

#: GEC-11 Vol II Table E1-7.4 (p. E1-16), per layer: Z (ft), Sv (ft), TMAX
#: (lb/ft, factored, Table E1-7.2 p. E1-14), grade and CDR = Tr / TMAX.
#: Layer 1's TMAX is at ZAve 0.835 ft, the middle of its contributory
#: height (p. E1-14).
E1_TABLE_7_4 = [
    (0.835, 1.67, 346, "GG-I", 2.82), (2.67, 2.0, 590, "GG-I", 1.65),
    (4.67, 2.0, 781, "GG-I", 1.25), (6.67, 2.0, 972, "GG-I", 1.00),
    (8.67, 2.0, 1163, "GG-II", 1.68), (10.67, 2.0, 1354, "GG-II", 1.44),
    (12.67, 2.0, 1545, "GG-II", 1.26), (14.67, 2.0, 1736, "GG-II", 1.12),
    (16.67, 2.0, 1927, "GG-II", 1.01), (18.67, 1.33, 1366, "GG-II", 1.43),
    (19.33, 1.0, 1091, "GG-II", 1.79)]


def _e1_geogrid(grade):
    Tult, RFID, _, _ = E1_TABLE_7_3[grade]
    Tal = Tult / (1.85 * 1.15 * RFID) * KFT / 1000          # kN/m
    return Reinforcement(f"E1 {grade}", "geogrid", Tallowable=Tal)


@pytest.mark.parametrize("grade", sorted(E1_TABLE_7_3))
def test_v057_geogrid_tal_and_tr(grade):
    """GM-268: E1 Step 7.6 (p. E1-15) takes Tal from Eq. 3-12 and 'The
    resistance factor for geosynthetic reinforcement is 0.90', Tr = phi Tal.
    The module's allowable is Tr / 1.35, so it checks Tr against the
    factored TMAX at 1.0 by checking 0.90 / 1.35 Tal against the
    unfactored Tmax."""
    _, _, Tal, Tr = E1_TABLE_7_3[grade]
    r = _e1_geogrid(grade)
    assert r.Tallowable * 1000 / KFT == pytest.approx(Tal, abs=0.5)
    assert rupture_allowable(r) * 1.35 * 1000 / KFT == pytest.approx(
        Tr, abs=0.5)


@pytest.mark.parametrize("z, Sv, Tmax, grade, cdr", E1_TABLE_7_4)
def test_v057_rupture_cdr_by_layer(z, Sv, Tmax, grade, cdr):
    """GM-268: Table E1-7.4 (p. E1-16) gives each layer's rupture CDR,
    Tr / TMAX, TMAX factored by gamma_EV-MAX 1.35 (p. E1-14). The module's
    FOS_rupture, 0.90 / 1.35 Tal Rc over its unfactored Tmax (Rc = 1), at
    E1's own Z and Sv, under the slope's S_eq, reproduces every CDR within
    0.6 % (its Tmax is 0.13 to 0.22 % under E1's, which rounds the slope to
    26.6 degrees, and E1 prints the CDR to two places) but layer 10's. There the module's Tmax at Z 18.67 ft, Sv 1.33 ft
    is 1,407 lb/ft factored against the printed 1,366 (3.0 % over; layer
    11 below it agrees within 0.2 %), so its CDR 1.39 is 2.9 % under the
    printed 1.43, on the safe side. Before GM-268 the module took Tal
    itself and every value was 1.5 times E1's (layer 4: 1.51 against
    1.00)."""
    geom = MSEWallGeometry(wall_height=20 * FT, reinforcement_length=18 * FT,
                           backfill_slope=BETA,
                           broken_backslope_height=9 * FT,
                           surcharge=0.25 * KSF)
    S_eq = backslope_geometry(geom)["S_eq"]
    T = Tmax_at_level(z * FT, GAMMA, rankine_Ka(34.0),
                      Kr_Ka_ratio(z * FT, "geosynthetic"), Sv * FT,
                      GAMMA * S_eq)
    assert 1.35 * T * 1000 / KFT == pytest.approx(
        Tmax, rel=0.031 if z == 18.67 else 0.0025)
    fos = rupture_allowable(_e1_geogrid(grade)) / T
    assert fos == pytest.approx(cdr, rel=0.03 if z == 18.67 else 0.0065)


def test_v057_rupture_in_one_run():
    """GM-268: the whole wall through ``analyze_mse_wall`` with E1's GG-I
    at every level and Sv 2 ft (the module takes one spacing and one
    reinforcement; E1's layout varies both): every level's FOS_rupture is
    0.90 / 1.35 Tal / Tmax; the levels at 1, 3 and 5 ft pass and those from
    7 ft down fail (0.97 at 7 ft), GG-I being at its limit at E1's layer 4
    (CDR 1.00 at Z 6.67 ft), and the result says so in its choices."""
    geom = MSEWallGeometry(wall_height=20 * FT, reinforcement_length=18 * FT,
                           reinforcement_spacing=2 * FT, backfill_slope=BETA,
                           broken_backslope_height=9 * FT,
                           surcharge=0.25 * KSF)
    gg1 = _e1_geogrid("GG-I")
    r = analyze_mse_wall(geom, GAMMA, 34.0, gg1, phi_foundation=30.0,
                         phi_retained=30.0)
    for lv in r.internal_results:
        assert lv["FOS_rupture"] == pytest.approx(
            0.90 / 1.35 * gg1.Tallowable / lv["Tmax_kN_per_m"], rel=1e-3)
        assert lv["Ta_Rc_kN_per_m"] == pytest.approx(
            0.90 / 1.35 * gg1.Tallowable, abs=0.01)
        assert (lv["FOS_rupture"] >= 1.0) == (lv["depth_m"] / FT < 6.0)
    (c,) = [c for c in r.choices if c["what"].startswith(
        "Allowable tensile strength Ta of the geosynthetic")]
    assert c["how"] == "module"
    for words in ("Table 4-7, p. 4-48", "Eq. 3-12, p. 3-28",
                  "gamma_P-EV = 1.35", "Table E1-7.4, p. E1-16",
                  "p. E9-2"):
        assert words in c["basis"], words
    assert c["alternatives"][0].startswith("Tal itself at a factor of "
                                           "safety of 1.0")
