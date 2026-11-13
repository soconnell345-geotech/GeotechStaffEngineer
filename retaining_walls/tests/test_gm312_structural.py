"""BACKLOG GM-312: the cantilever wall's structural design on USACE EM
1110-2-2104 (1 November 2023).

The pins are the manual's own worked examples, D-3 (pp. 73-76) and D-5
(pp. 79-87), read from the print. The wall of the hand check is made up.
"""

import math

import pytest

from geotech_common import units as U
from retaining_walls import structural_section as S
from retaining_walls.geometry import CantileverWallGeometry
from retaining_walls.structural import (
    SCHEMA, _base_pressure, _segment_force,
    analyze_cantilever_wall_structure,
)


# ── unit conversions (EM 1110-2-2104 Glossary Section III, p. 129) ─────

def test_the_converters_are_the_manuals_factors():
    # pounds (force) per square inch x 6.894757 = kilopascals
    assert U.psi_to_MPa(1000.0) == pytest.approx(6.894757)
    assert U.MPa_to_psi(6.894757) == pytest.approx(1000.0)
    # 1 kip-ft/ft = 4.448222 kN-m/m; 1 kip/ft = 4.448222 / 0.3048 kN/m
    assert U.kipft_per_ft_to_kNm_per_m(1.0) == pytest.approx(4.448222)
    assert U.kip_per_ft_to_kN_per_m(1.0) == pytest.approx(4.448222 / 0.3048)
    # 1 sq in/ft = 645.16 mm2 / 0.3048 m
    assert U.in2_per_ft_to_mm2_per_m(1.0) == pytest.approx(645.16 / 0.3048)
    for there, back in ((U.kNm_per_m_to_kipft_per_ft,
                         U.kipft_per_ft_to_kNm_per_m),
                        (U.kN_per_m_to_kip_per_ft, U.kip_per_ft_to_kN_per_m),
                        (U.mm2_per_m_to_in2_per_ft,
                         U.in2_per_ft_to_mm2_per_m)):
        assert back(there(123.4)) == pytest.approx(123.4)


# ── D-3, the singly reinforced wall stem (pp. 73-76) ───────────────────

def test_d3_wall_stem_as_printed():
    # Given M = 5 k-ft (usual), f'c = 4 ksi, fy = 60 ksi; Step 1: Mu =
    # 2.2 x 5 = 11 k-ft; Step 2: d = 6 in.; the example's 2 in. cover
    Mu = S.SINGLE_LOAD_FACTOR["usual"] * 5.0
    assert Mu == pytest.approx(11.0)
    r = S.design_section(
        Mu, 0.0, 9.0, 2.0, 4000.0, d_in=6.0,
        service_moments=[{"case": "1", "category": "usual",
                          "M_kipft": 5.0}],
        modular_ratio=29000.0 / 3605.0)
    assert r["Mn_required_kin"] == pytest.approx(147.0, abs=0.5)
    assert r["Ku"] == pytest.approx(0.105, abs=0.0005)
    assert r["As_flexure_in2"] == pytest.approx(0.43, abs=0.005)
    # "Use No. 6 bars at 12-in. on center"
    assert (r["bar_size"], r["spacing_in"]) == ("No. 6", 12.0)
    assert r["As_provided_in2"] == pytest.approx(0.44)
    # Step 4: rho = 0.0061 against 0.25 rho_b = 0.0071, beta1 = 0.85
    assert r["beta1"] == 0.85
    assert r["rho"] == pytest.approx(0.0061, abs=0.00005)
    assert 0.25 * r["rho_b"] == pytest.approx(0.0071, abs=0.00005)
    # Step 5: fs = 24.96 ksi (x = 1.61 in.), s_max = 19 in.
    (svc,) = r["service"]
    assert svc["x_in"] == pytest.approx(1.61, abs=0.005)
    assert svc["fs_psi"] / 1000.0 == pytest.approx(24.96, abs=0.01)
    assert svc["s_max_in"] == pytest.approx(19.0, abs=0.05)
    assert r["passes"]


# ── D-5, the coastal floodwall's stem (pp. 79-87) ──────────────────────

def test_d5_stem_loads_rebuild_from_its_tables():
    """Tables D-2 (p. 80) and D-4 (p. 84) print enough to rebuild the
    stem's factored loads by statics: Vu = 36.9 kips/ft, Mu = 428
    k-ft/ft."""
    top_of_base, gw = -12.33, 0.0642            # ft, kcf
    h_fs = 14.8 - top_of_base                    # 500-year flood side
    h_ls = -1.9 - top_of_base                    # land side low water
    h_soil = -10.0 - top_of_base
    Ko, g_soil = 0.524, 0.115
    fs = 1.4 * 0.5 * gw * h_fs ** 2
    ls = 1.4 * 0.5 * gw * h_ls ** 2
    eh = 0.5 * Ko * (g_soil - gw) * h_soil ** 2  # buoyant, each side
    wave = 1.4 * 6.16                            # 500-year wave load
    assert fs == pytest.approx(33.08, abs=0.01)
    assert ls == pytest.approx(4.89, abs=0.01)
    assert 1.35 * eh == pytest.approx(0.10, abs=0.005)
    assert 0.9 * eh == pytest.approx(0.07, abs=0.006)
    assert wave == pytest.approx(8.62, abs=0.01)
    Vu = fs - ls + 1.35 * eh - 0.9 * eh + wave
    Mu = (fs * h_fs / 3.0 - ls * h_ls / 3.0
          + (1.35 - 0.9) * eh * h_soil / 3.0
          + wave * (4.63 - top_of_base))
    assert Vu == pytest.approx(36.85, abs=0.03)
    assert Mu == pytest.approx(428.0, abs=1.0)


def test_d5_stem_section_as_printed():
    # Step 3: h = 42 in., 4 in. cover, d used 36.5 in., f'c = 5,000 psi
    r = S.design_section(428.0, 36.9, 42.0, 4.0, 5000.0, d_in=36.5)
    # (h) Vc = 61.9, phi Vc = 46.5 kips/ft against Vu = 36.9
    assert r["Vc_kip"] == pytest.approx(61.9, abs=0.05)
    assert r["phi_Vc_kip"] == pytest.approx(46.5, abs=0.05)
    # (i) Ku = 0.088, As = 2.73 sq in/ft; (j)-(k) No. 11 at 6 in.
    assert r["Ku"] == pytest.approx(0.088, abs=0.0005)
    assert r["As_flexure_in2"] == pytest.approx(2.73, abs=0.005)
    assert (r["bar_size"], r["spacing_in"]) == ("No. 11", 6.0)
    assert r["As_provided_in2"] == pytest.approx(3.12)
    # Step 4: beta1 = 0.80, rho = 0.0071, 0.25 rho_b = 0.0083 (printed
    # cut short of 0.00838)
    assert r["beta1"] == pytest.approx(0.80)
    assert r["rho"] == pytest.approx(0.0071, abs=0.00005)
    assert 0.25 * r["rho_b"] == pytest.approx(0.0083, abs=0.0001)
    assert r["passes"]


def test_d5_hook_development_length_as_printed():
    # Step 6 (p. 87): psi_c = 0.93, ldh = 24.1 in. for No. 11 (1.41 in.)
    h = S.hook_development_length(1.41, 5000.0, 60000.0)
    assert h["psi_c"] == pytest.approx(0.93, abs=0.005)
    assert h["ldh_in"] == pytest.approx(24.1, abs=0.05)
    # psi_c is printed "for f'c < 6,000 psi" only
    none = S.hook_development_length(1.41, 6000.0, 60000.0)
    assert none["ldh_in"] is None and "6,000" in none["why"]


def test_d5_service_stress_is_the_arithmetic_not_the_printed_34_7():
    """Review B6. D-5 Step 5(a) (p. 86) prints fs = 34.7 ksi '< 35 ksi'
    for M = 306 k-ft, As = 3.12 sq in, d = 36.5 in., n = 29,000 / 4,031.
    The print's own numbers give 35.5 ksi (35.47 on its x = 9.96 in.,
    35.52 on the 10.09 it then uses), over Table 3-3's 35 ksi. The module
    gives 35.47 and FAILS the check: correct arithmetic. Do not
    "correct" it to 34.7 (module_work/gm312/CHECK.md lists the slip)."""
    r = S.design_section(
        428.0, 36.9, 42.0, 4.0, 5000.0, d_in=36.5, bar_size="No. 11",
        spacing_in=6.0,
        service_moments=[{"case": "FS", "category": "unusual",
                          "M_kipft": 306.0}],
        modular_ratio=29000.0 / 4031.0)
    (svc,) = r["service"]
    assert svc["x_in"] == pytest.approx(9.96, abs=0.005)
    assert svc["fs_psi"] / 1000.0 == pytest.approx(35.47, abs=0.005)
    stress = [c for c in r["checks"]
              if c["name"].startswith("service steel stress")]
    assert len(stress) == 1 and not stress[0]["passes"]


def test_d5_temperature_steel_as_printed():
    # Step 7 (p. 87): monolith over 40 ft (12 m), 0.005 x 42 x 12 / 2 =
    # 1.26, held to No. 9 at 12 in. (1.00 sq in/ft)
    assert S.temperature_shrinkage_ratio(13.7) == 0.005
    assert 0.005 * 42.0 * 12.0 / 2.0 == pytest.approx(1.26)
    assert S.temperature_shrinkage_area(0.005, 42.0) == 1.00
    # with a joint at mid-length, 0.003: "#8 bars"
    As = S.temperature_shrinkage_area(0.003, 42.0)
    assert S.temperature_shrinkage_bar(As)["bar_size"] == "No. 8"


def test_the_printed_tables():
    # Table 2-3 (p. 7), on the metres it prints: "Less than 30 ft (9 m)",
    # "30-40 ft (9-12 m)", "Greater than 40 ft (12 m)" (review A1)
    assert [S.temperature_shrinkage_ratio(x)
            for x in (8.9, 9.0, 12.0, 12.1)] == [0.003, 0.004, 0.004, 0.005]
    # para. 2-9a: not less than No. 4 at 12 in.
    assert S.temperature_shrinkage_area(0.003, 9.0) == 0.20
    # Table 2-1 (p. 5)
    assert S.table_2_1_cover_in(30.0, against_ground=True) == 4.0
    assert S.table_2_1_cover_in(24.0) == 4.0
    assert S.table_2_1_cover_in(23.9) == 3.0
    assert S.table_2_1_cover_in(12.1) == 3.0
    assert S.table_2_1_cover_in(12.0) is None
    # beta1 as the manual prints it at 4,000 psi (pp. 52, 75) and 5,000
    # psi (p. 86)
    assert S.beta1_default(4000.0) == 0.85
    assert S.beta1_default(5000.0) == pytest.approx(0.80)
    # Table 6-2 of GEC-10 (p. 6-3)
    assert S.BARS["No. 4"] == (0.500, 0.20)
    assert S.BARS["No. 11"] == (1.410, 1.56)


def test_eq_5_1_two_printed_forms_differ_by_about_two_percent():
    r = S.design_section(50.0, 5.0, 18.0, 3.0, 4000.0)
    ratio = r["phi_Vc_SI_form_kip"] / r["phi_Vc_kip"]
    # 0.17 sqrt(MPa) against 2 sqrt(psi) = 0.166 sqrt(MPa)
    assert ratio == pytest.approx(0.17 / (2.0 * math.sqrt(6.894757e-3)),
                                  rel=1e-6)
    assert 1.02 < ratio < 1.03


# ── a made-up wall, worked by hand (module_work/gm312/CHECK.md) ────────

GAMMA, PHI, GAMMA_C, Q_LIVE = 19.0, 32.0, 23.6, 12.0


def _wall(**kw):
    geom = dict(wall_height=4.5, base_width=3.0, toe_length=0.6,
                stem_thickness_top=0.305, stem_thickness_base=0.45,
                base_thickness=0.45, embedment_depth=0.9, surcharge=Q_LIVE)
    geom.update({k: kw.pop(k) for k in list(kw) if k in (
        "surcharge", "water_table_depth", "dead_load_surcharge",
        "backfill_slope", "stem_thickness_base", "base_thickness",
        "toe_length", "base_width", "wall_height", "stem_thickness_top",
        "has_shear_key", "key_depth")})
    args = dict(fc_MPa=28.0, joint_spacing=8.0, q_allowable=250.0)
    args.update(kw)
    return analyze_cantilever_wall_structure(
        CantileverWallGeometry(**geom), GAMMA, PHI, **args)


@pytest.fixture(scope="module")
def wall():
    return _wall()


def test_hand_check_stem(wall):
    Ka = math.tan(math.radians(45.0 - PHI / 2.0)) ** 2
    h = 4.5 - 0.45
    V_eh = 0.5 * Ka * GAMMA * h ** 2            # 47.88 kN/m at h/3
    M_eh = V_eh * h / 3.0                       # 64.64 kN-m/m
    V_es = Ka * Q_LIVE * h                      # 14.93 kN/m at h/2
    M_es = V_es * h / 2.0                       # 30.24 kN-m/m
    assert (V_eh, M_eh) == pytest.approx((47.88, 64.64), abs=0.01)
    assert (V_es, M_es) == pytest.approx((14.93, 30.24), abs=0.01)
    cases = {(c["case"], c["column"]): c
             for c in wall.elements["stem"]["load_cases"]}
    # case 1: 2.2 (EH); 1.5 EH. Case 2: 1.6 (EH + ES); 1.5 EH + 1.6 ES
    assert cases["1", "serviceability"]["M_kNm_per_m"] == pytest.approx(
        2.2 * M_eh, abs=0.01)
    assert cases["1", "strength"]["M_kNm_per_m"] == pytest.approx(
        1.5 * M_eh, abs=0.01)
    assert cases["2", "serviceability"]["M_kNm_per_m"] == pytest.approx(
        1.6 * (M_eh + M_es), abs=0.01)
    assert cases["2", "strength"]["M_kNm_per_m"] == pytest.approx(
        1.5 * M_eh + 1.6 * M_es, abs=0.01)
    stem = wall.elements["stem"]
    assert stem["Mu_kNm_per_m"] == pytest.approx(151.80, abs=0.01)
    assert stem["Vu_kN_per_m"] == pytest.approx(2.2 * V_eh, abs=0.01)
    assert stem["governing_moment_case"]["case"] == "2"
    assert stem["governing_shear_case"]["case"] == "1"


def test_hand_check_stem_steel(wall):
    stem = wall.elements["stem"]
    fc = 28.0 * 1000.0 / 6.894757                # 4,061 psi
    Mu = 151.80 / 4.448222                       # 34.13 kip-ft/ft
    h = 450.0 / 25.4                             # 17.72 in.
    d = h - 3.0 - 0.625 / 2.0                    # No. 5, 3 in. cover
    assert stem["d_in"] == pytest.approx(d, abs=0.001)
    Mn = Mu * 12.0 / 0.9
    Ku = 1.0 - math.sqrt(1.0 - Mn / (0.425 * fc / 1000.0 * 12.0 * d ** 2))
    As = 0.85 * fc / 1000.0 * Ku * 12.0 * d / 60.0
    assert As == pytest.approx(0.541, abs=0.001)
    assert stem["As_flexure_in2_per_ft"] == pytest.approx(As, abs=0.001)
    # shrinkage and temperature floor: 0.003 x 17.72 x 12 / 2
    assert stem["As_minimum_in2_per_ft"] == pytest.approx(0.319, abs=0.001)
    # crack control on the 35 ksi limit: 15 x 40/35 - 2.5 x 2.5 = 10.9
    # in., so 12 in. is passed over: No. 5 at 6 in. (0.62 sq in/ft)
    assert (stem["bar"]["size"], stem["bar"]["spacing_in"]) == ("No. 5",
                                                                6.0)
    assert stem["bar"]["spacing_mm"] == 150.0
    assert stem["service"][1]["s_max_in"] == pytest.approx(10.89, abs=0.01)
    a = 0.62 * 60000.0 / (0.85 * fc * 12.0)
    phi_Mn = 0.9 * 0.62 * 60.0 * (d - a / 2.0) / 12.0
    assert stem["phi_Mn_kipft_per_ft"] == pytest.approx(phi_Mn, abs=0.01)
    phi_Vc = 0.75 * 2.0 * math.sqrt(fc) * 12.0 * d / 1000.0
    assert stem["phi_Vc_kip_per_ft"] == pytest.approx(phi_Vc, abs=0.01)
    assert stem["passes"]


def _hand_base(live_on_heel=None):
    """V, the resultant and the edge pressures by hand, unfactored: case 1
    (None), or case 2 with the surcharge over the heel or behind it."""
    Ka = math.tan(math.radians(45.0 - PHI / 2.0)) ** 2
    H, B, toe, tb, tt, t, Df = 4.5, 3.0, 0.6, 0.45, 0.305, 0.45, 0.9
    hs, heel, taper = H - t, B - toe - tb, tb - tt
    items = [
        (GAMMA_C * B * t, B / 2.0),                          # base slab
        (GAMMA_C * tt * hs, toe + tt / 2.0),                 # stem
        (GAMMA_C * 0.5 * taper * hs, toe + tt + taper / 3.0),
        (GAMMA * 0.5 * taper * hs, toe + tt + 2 * taper / 3.0),
        (GAMMA * heel * hs, toe + tb + heel / 2.0),          # soil, heel
        (GAMMA * (Df - t) * toe, toe / 2.0),                 # soil, toe
    ]
    M_over = 0.5 * Ka * GAMMA * H ** 2 * H / 3.0
    if live_on_heel is not None:
        M_over += Ka * Q_LIVE * H * H / 2.0
        if live_on_heel:
            items.append((Q_LIVE * heel, toe + tb + heel / 2.0))
    V = sum(w for w, _ in items)
    x = (sum(w * a for w, a in items) - M_over) / V
    e = B / 2.0 - x
    return V, x, V / B * (1 + 6 * e / B), V / B * (1 - 6 * e / B)


def test_hand_check_toe_and_heel(wall):
    V, x, q_toe, q_heel = _hand_base()
    assert (V, x) == pytest.approx((228.70, 1.3059), abs=0.005)
    assert (q_toe, q_heel) == pytest.approx((105.83, 46.63), abs=0.01)
    B, toe, xb, heel = 3.0, 0.6, 1.05, 1.95
    slope = (q_heel - q_toe) / B
    # toe: the pressure up, the slab and the soil over the toe down
    q_face = q_toe + slope * toe                              # 93.99
    w = GAMMA_C * 0.45 + GAMMA * 0.45                         # 19.17
    M_toe = (q_face * toe ** 2 / 2.0
             + (q_toe - q_face) * toe / 2.0 * (2.0 * toe / 3.0)
             - w * toe ** 2 / 2.0)
    assert M_toe == pytest.approx(14.89, abs=0.01)
    # heel: the slab and the soil down, the pressure up
    q_back = q_toe + slope * xb                               # 85.11
    w_heel = GAMMA_C * 0.45 + GAMMA * 4.05                    # 87.57
    M_heel = (w_heel * heel ** 2 / 2.0 - q_heel * heel ** 2 / 2.0
              - (q_back - q_heel) * heel / 2.0 * (heel / 3.0))
    V_heel = w_heel * heel - 0.5 * (q_back + q_heel) * heel
    assert M_heel == pytest.approx(53.44, abs=0.01)
    assert V_heel == pytest.approx(42.31, abs=0.01)

    toe_el, heel_el = wall.elements["toe"], wall.elements["heel"]
    case = {(c["case"], c["column"]): c for c in toe_el["load_cases"]}
    assert case["1", "serviceability"]["M_kNm_per_m"] == pytest.approx(
        2.2 * M_toe, abs=0.01)
    assert toe_el["Mu_kNm_per_m"] == pytest.approx(32.76, abs=0.01)
    bp = toe_el["governing_moment_case"]["base_pressure"]
    assert (bp["q_toe_kPa"], bp["q_heel_kPa"]) == pytest.approx(
        (q_toe, q_heel), abs=0.01)
    case = {(c["case"], c["column"]): c for c in heel_el["load_cases"]}
    assert case["1", "serviceability"]["M_kNm_per_m"] == pytest.approx(
        2.2 * M_heel, abs=0.01)
    assert case["1", "serviceability"]["V_kN_per_m"] == pytest.approx(
        2.2 * V_heel, abs=0.01)

    # case 2: the surcharge over the heel, or behind it only; the heel's
    # moment is larger with it behind (less pressure under the heel)
    for place, M2 in ((True, None), (False, 80.26)):
        V2, x2, qt2, qh2 = _hand_base(place)
        s2 = (qh2 - qt2) / B
        qb2 = qt2 + s2 * xb
        w2 = w_heel + (Q_LIVE if place else 0.0)
        M = (w2 * heel ** 2 / 2.0 - qh2 * heel ** 2 / 2.0
             - (qb2 - qh2) * heel / 2.0 * (heel / 3.0))
        if M2 is not None:
            assert M == pytest.approx(M2, abs=0.01)
            assert heel_el["Mu_kNm_per_m"] == pytest.approx(1.6 * M,
                                                            abs=0.01)
        else:
            assert M < 80.26
            # the wall module's own bearing pressures are this state's
            assert (qt2, qh2) == pytest.approx(
                (wall.stability.q_toe, wall.stability.q_heel), abs=0.06)
    assert heel_el["governing_moment_case"]["surcharge_over_heel"] is False


def test_toe_shear_is_taken_at_d_from_the_stem(wall):
    toe = wall.elements["toe"]
    V, x, q_toe, q_heel = _hand_base()
    d = toe["d_mm"] / 1000.0                    # 450 - 100 - 6.35 mm
    assert d == pytest.approx(0.45 - 4 * 0.0254 - 0.25 * 0.0254, abs=1e-4)
    xs = 0.6 - d
    q_s = q_toe + (q_heel - q_toe) / 3.0 * xs
    V1 = 0.5 * (q_toe + q_s) * xs - 19.17 * xs
    case = {(c["case"], c["column"]): c for c in toe["load_cases"]}
    assert case["1", "serviceability"]["V_kN_per_m"] == pytest.approx(
        2.2 * V1, abs=0.02)


def test_every_combination_is_in_vertical_equilibrium(wall):
    """The pressure a combination's toe and heel are designed for carries
    that combination's vertical load at its resultant."""
    for name in ("toe", "heel"):
        for row in wall.elements[name]["_rows"]:
            st = row["state"]
            F, Mx = _segment_force(st["pressure"], 0.0, 3.0)
            assert F == pytest.approx(st["V"], rel=1e-9)
            assert Mx / F == pytest.approx(st["x_r"], rel=1e-9)


def test_base_pressure_shapes():
    seg, shape = _base_pressure(100.0, 1.5, 3.0)
    assert shape == "trapezoidal" and seg[1] == pytest.approx(seg[3])
    seg, shape = _base_pressure(100.0, 0.6, 3.0)
    assert shape == "triangular_toe"
    assert seg == pytest.approx((0.0, 2 * 100.0 / 1.8, 1.8, 0.0))
    seg, shape = _base_pressure(100.0, 2.4, 3.0)
    assert shape == "triangular_heel"
    assert _segment_force(seg, 0.0, 3.0)[0] == pytest.approx(100.0)


# ── the result: what it carries and what it refuses ────────────────────

def test_the_stability_result_is_the_same_wall(wall):
    from retaining_walls.cantilever import analyze_cantilever_wall
    g = CantileverWallGeometry(
        wall_height=4.5, base_width=3.0, toe_length=0.6,
        stem_thickness_top=0.305, stem_thickness_base=0.45,
        base_thickness=0.45, embedment_depth=0.9, surcharge=Q_LIVE)
    alone = analyze_cantilever_wall(g, GAMMA, PHI, q_allowable=250.0)
    assert wall.stability.to_dict() == alone.to_dict()
    d = wall.to_dict()
    assert d["stability"]["FOS_sliding"] == alone.to_dict()["FOS_sliding"]
    # the stability checks' failures lead the list
    assert d["checks_failed"][0] == "sliding"
    assert d["verdict"].startswith("Concrete: no check failed.")
    assert "Stability: FAILS: sliding" in d["verdict"]
    assert "Not designed: earthquake load cases" in d["verdict"]
    assert "STEM" in wall.summary()


def test_the_reinforcement_description_keeps_its_contract(wall):
    r = wall.reinforcement
    assert list(r) == ["schema", "title", "geometry", "materials", "bars",
                       "notes", "basis"]
    assert r["schema"] == SCHEMA == "reinforced_wall_section/1"
    assert list(r["geometry"]) == [
        "wall_height", "base_width", "toe_length", "heel_length",
        "stem_thickness_top", "stem_thickness_base", "base_thickness",
        "stem_batter", "embedment_depth", "backfill_slope_deg",
        "has_shear_key", "key_depth", "key_width", "key_distance_from_toe"]
    assert r["geometry"]["key_depth"] == 0.0
    assert r["geometry"]["key_width"] is None
    assert r["materials"] == {"fc_MPa": 28.0, "fy_MPa": 414.0}
    keys = ["mark", "element", "face", "runs", "role", "size",
            "diameter_mm", "spacing_mm", "cover_mm", "end", "callout",
            "As_required_mm2_per_m", "As_provided_mm2_per_m"]
    assert all(list(b) == keys for b in r["bars"])
    by = {b["mark"]: b for b in r["bars"]}
    # the base's top cover (75 mm) and bottom cover (100 mm) differ, so
    # the base's bars along the wall are two sets, F top and G bottom
    # (review C6); the drawing must not assume six marks
    assert list(by) == ["A", "B", "C", "D", "E", "F", "G"]
    assert [(b["element"], b["face"], b["runs"], b["role"])
            for b in r["bars"]] == [
        ("stem", "back", "in_section", "flexure"),
        ("stem", "front", "in_section", "temperature_shrinkage"),
        ("stem", "both", "along_wall", "temperature_shrinkage"),
        ("toe", "bottom", "in_section", "flexure"),
        ("heel", "top", "in_section", "flexure"),
        ("base", "top", "along_wall", "temperature_shrinkage"),
        ("base", "bottom", "along_wall", "temperature_shrinkage")]
    assert (by["F"]["cover_mm"], by["G"]["cover_mm"]) == (75.0, 100.0)
    assert by["F"]["size"] == by["G"]["size"]
    assert by["G"]["callout"].endswith("bottom, along the wall")
    # equal covers: one set, both faces
    same = _wall(base_top_cover_mm=100.0).reinforcement["bars"]
    assert [b["mark"] for b in same] == list("ABCDEF")
    assert (same[-1]["face"], same[-1]["cover_mm"]) == ("both", 100.0)
    assert by["A"]["end"] == {"type": "standard_hook_90", "into": "base",
                              "ldh_mm": 187.0}
    assert by["A"]["callout"] == "No. 5 @ 150 mm (6 in.), back face"
    assert by["A"]["diameter_mm"] == 15.9
    assert by["D"]["cover_mm"] == 100.0 and by["E"]["cover_mm"] == 75.0
    assert all(b["end"] is None for m, b in by.items() if m != "A")
    assert by["C"]["callout"].endswith("each face, along the wall")
    assert any("not designed" in n for n in r["notes"])
    assert any("4,060 psi" in n for n in r["notes"])
    import json
    json.dumps(wall.to_dict())


def test_choices_and_not_designed_name_their_pages(wall):
    by = {c["what"]: c for c in wall.structural_choices}
    for what in ("strength reduction factors", "beta1", "serviceability",
                 "shear strength", "toe and heel moment sections",
                 "stem bars' anchorage", "crack control", "clear cover",
                 "minimum flexural steel", "bar choice", "load cases",
                 "which loads are which", "effective depth"):
        assert what in by, what
        assert "p. " in by[what]["basis"] or "pp. " in by[what]["basis"]
    assert "not printed in a held reference" in by[
        "toe and heel moment sections"]["basis"]
    assert "p. 84" in by["strength reduction factors"]["basis"]
    # review B2: the active state is an assumption, the at-rest basis named
    lf = by["strength load factors"]
    assert "active" in lf["chosen"]
    assert "Table E-2 note b (p. 93)" in lf["basis"]
    assert any("at-rest" in a and "1.35" in a and "p. 22" in a
               and "not built" in a for a in lf["alternatives"])
    # review C2: the live surcharge by para. 3-2f(7)
    assert "para. 3-2f(7) (p. 19)" in by["which loads are which"]["basis"]
    # review B4: the hook's factors are the example's; confirm against ACI
    assert "confirmed against ACI 318-19" in by[
        "stem bars' anchorage"]["basis"]
    assert "ACI 318-19" in wall.development["confirm"]
    assert any("confirm the hook against ACI 318-19" in n
               for n in wall.reinforcement["notes"])
    # review B5 (i): the wall's height, read conservatively, and said
    th = by["minimum wall thickness"]
    assert th["chosen"].startswith("305 mm") and "conservative" in th[
        "basis"]
    # the stability analysis's choices are carried too
    assert len(wall.choices) > len(wall.structural_choices)
    whats = " | ".join(n["what"] for n in wall.not_designed)
    for word in ("3A and 3B", "case 4", "case 5", "lap lengths",
                 "cut-offs", "counterforts"):
        assert word in whats
    assert "shear key" not in whats
    keyed = _wall(has_shear_key=True, key_depth=0.4, include_passive=True)
    assert "the shear key's steel" in [n["what"]
                                       for n in keyed.not_designed]
    assert keyed.reinforcement["geometry"]["key_depth"] == 0.4


def test_without_a_live_surcharge_only_case_1_is_designed():
    r = _wall(surcharge=0.0)
    designed = {c["case"]: c["designed"] for c in r.load_cases}
    assert designed == {"1": True, "2": False, "3A, 3B": False,
                        "4": False, "5": False}
    assert {c["case"] for c in r.elements["stem"]["load_cases"]} == {"1"}
    # usual loads: 12 in. meets the crack control (para. 2-6b(2))
    assert r.elements["stem"]["bar"]["spacing_in"] == 12.0
    assert r.elements["stem"]["bar"]["spacing_mm"] == 300.0
    assert r.elements["stem"]["service"][0]["s_max_in"] is None


def test_another_fy_needs_the_modular_ratio():
    with pytest.raises(ValueError, match="60,000 psi"):
        _wall(fy_MPa=500.0)
    with pytest.raises(ValueError, match="80,000 psi"):
        _wall(fy_MPa=600.0)
    r = _wall(fy_MPa=500.0, modular_ratio=8.0)
    assert "service stress limits of Table 3-3" in r.basis
    # no single-load-factor rows; the strength column governs
    assert {c["column"] for c in r.elements["stem"]["load_cases"]} == {
        "strength"}
    names = [c["name"] for c in r.elements["stem"]["checks"]]
    assert any(n.startswith("service steel stress, usual") for n in names)
    assert any(n.startswith("service steel stress, unusual")
               for n in names)


def test_a_modular_ratio_gives_the_actual_stress():
    r = _wall(modular_ratio=8.0)
    stem = r.elements["stem"]
    usual, unusual = stem["service"]
    assert usual["stress_basis"] == unusual["stress_basis"] == "actual"
    # fs = M / [As (d - x/3)] on the bar chosen
    sec = stem["_section"]
    st = S.service_steel_stress(
        U.kNm_per_m_to_kipft_per_ft(64.636), sec["As_provided_in2"],
        sec["d_in"], 8.0)
    assert usual["fs_ksi"] == pytest.approx(st["fs_psi"] / 1000.0, abs=0.01)
    assert usual["fs_ksi"] <= 25.0 and unusual["fs_ksi"] <= 35.0


def test_a_thin_section_needs_its_cover_from_the_engineer():
    with pytest.raises(ValueError, match="stem_cover_mm is required"):
        _wall(stem_thickness_base=0.30, stem_thickness_top=0.30,
              wall_height=2.5, base_thickness=0.4, base_width=1.8,
              toe_length=0.3)
    with pytest.raises(ValueError, match="base_top_cover_mm is required"):
        _wall(base_thickness=0.30)
    r = _wall(base_thickness=0.30, base_top_cover_mm=50.0)
    assert r.elements["heel"]["cover_mm"] == 50.0
    assert r.elements["heel"]["cover_given"] is True
    assert r.elements["toe"]["cover_mm"] == 100.0


def test_the_engineers_bar_and_spacing_are_taken():
    r = _wall(stem_bar_size="No. 8", stem_bar_spacing_mm=250.0)
    bar = r.elements["stem"]["bar"]
    assert (bar["size"], bar["spacing_mm"]) == ("No. 8", 250.0)
    assert bar["spacing_in"] == pytest.approx(9.84, abs=0.005)
    assert r.reinforcement["bars"][0]["callout"].startswith(
        "No. 8 @ 250 mm (9.84 in.)")
    # a spacing over 12 in. fails the printed limit
    wide = _wall(heel_bar_spacing_mm=400.0)
    assert "heel: bar spacing <= 12 in." in wide.structural_checks_failed
    with pytest.raises(ValueError, match="stem_bar_size"):
        _wall(stem_bar_size="No. 14")


def test_the_minimum_wall_thickness_is_checked():
    # 300 mm is under the 12 in. (30.5 cm) of para. 3-7 for a wall over
    # 10 ft (3 m) high; compared in the metric the print gives, with the
    # value to give named (review B5, C1)
    r = _wall(stem_thickness_top=0.30)
    assert ("wall: stem thickness at its top >= the minimum wall "
            "thickness") in r.structural_checks_failed
    assert r.verdict.startswith(
        "CONCRETE FAILS: wall: stem thickness at its top >= the minimum "
        "wall thickness (needs 305 mm, has 300 mm): the wall is 4.5 m "
        "high")
    assert "give stem_thickness_top 0.305 or more" in r.verdict
    # the WALL's height, base included, against the printed 3 m: a 3.2 m
    # wall with a 0.45 m base has a stem under 3 m and still needs 305 mm
    low = _wall(wall_height=3.2, base_width=2.2, stem_thickness_top=0.30,
                stem_thickness_base=0.40)
    assert low.geometry["stem_height"] < 3.0
    assert any(c.startswith("wall: stem thickness") for c in
               low.structural_checks_failed)
    ok = _wall(wall_height=2.9, base_width=2.0, stem_thickness_top=0.25,
               stem_thickness_base=0.40)
    (thick,) = [c for c in ok.checks if c["name"].startswith("stem thick")]
    assert (thick["demand"], thick["capacity"], thick["passes"]) == (
        200.0, 250.0, True)


def test_temperature_steel_bands_on_the_metres_printed():
    # review A1: Table 2-3 (p. 7) prints 9 m and 12 m; 9.0 m is in the
    # 0.004 band and 12.1 m in the 0.005 band; the stem is 450 mm thick,
    # half the steel in each face
    def along(joints):
        r = _wall(joint_spacing=joints)
        (c,) = [b for b in r.reinforcement["bars"] if b["mark"] == "C"]
        chosen = {x["what"]: x for x in r.structural_choices}[
            "temperature and shrinkage steel"]["chosen"]
        return c["As_required_mm2_per_m"], chosen
    for joints, ratio in ((8.9, 0.003), (9.0, 0.004), (12.0, 0.004),
                          (12.1, 0.005)):
        As, chosen = along(joints)
        assert As == pytest.approx(ratio * 450.0 * 1000.0 / 2.0, rel=2e-3)
        assert chosen.startswith(f"along the wall {ratio:g} of the gross "
                                 f"area for joints {joints:g} m apart")


def test_a_given_cover_is_checked_against_table_2_1():
    # review B1: 50 mm against the foundation, where Table 2-1 (p. 5)
    # prints 4 in. (10 cm); 50 mm on a 450 mm stem, where it prints 3 in.
    # (7.5 cm)
    r = _wall(base_bottom_cover_mm=50.0, stem_cover_mm=50.0)
    failed = r.structural_checks_failed
    assert "wall: clear cover >= Table 2-1 (base bottom)" in failed
    assert "wall: clear cover >= Table 2-1 (stem)" in failed
    assert ("clear cover >= Table 2-1 (base bottom) (needs 100 mm, has 50 "
            "mm): Table 2-1 prints 4 in. (10 cm) for unformed surfaces in "
            "contact with foundation: give base_bottom_cover_mm 100 or "
            "more") in r.verdict
    # the printed metric values pass; a thin section keeps today's rule
    fine = _wall(base_bottom_cover_mm=100.0, stem_cover_mm=75.0)
    assert not any("Table 2-1" in c for c in fine.structural_checks_failed)
    assert fine.elements["stem"]["cover_in"] == 3.0
    thin = _wall(base_thickness=0.30, base_top_cover_mm=50.0)
    assert not any("(base top)" in c["name"] for c in thin.checks)


def test_the_bar_choice_meets_the_cover_and_the_clear_distance():
    """Review B3: H 6.6 m, stem 550 mm (3 in. cover by Table 2-1), no
    surcharge. No. 11 at 12 in. gives the steel but needs 2.5 x 1.41 =
    3.53 in. of cover; the search moves on to 6 in. and No. 8 (1.58 sq
    in/ft) passes every check."""
    geom = CantileverWallGeometry(
        wall_height=6.6, base_width=0.62 * 6.6, toe_length=0.9,
        stem_thickness_top=0.31, stem_thickness_base=0.55,
        base_thickness=0.62)
    r = analyze_cantilever_wall_structure(
        geom, 19.0, 30.0, fc_MPa=28.0, joint_spacing=8.0,
        phi_foundation=32.0, q_allowable=400.0)
    stem = r.elements["stem"]
    assert stem["passes"], [c["name"] for c in stem["checks"]
                            if c["required"] and not c["passes"]]
    assert (stem["bar"]["size"], stem["bar"]["spacing_mm"]) == (
        "No. 8", 150.0)
    # the verdict of a failed cover check reads needed first (review C1)
    forced = _wall(stem_bar_size="No. 11", stem_bar_spacing_mm=300.0)
    assert ("stem: clear cover >= 2.5 bar diameters (needs 3.53 in., has "
            "3.00 in.)") in forced.verdict


def test_a_printed_metric_spacing_is_the_inch_value():
    # review C3: 300 mm is the manual's "12 in. (30 cm)"
    r = _wall(heel_bar_spacing_mm=300.0)
    bar = r.elements["heel"]["bar"]
    assert (bar["spacing_mm"], bar["spacing_in"]) == (300.0, 12.0)
    assert "heel: bar spacing <= 12 in." not in r.structural_checks_failed
    (E,) = [b for b in r.reinforcement["bars"] if b["mark"] == "E"]
    assert "@ 300 mm (12 in.)" in E["callout"]


def test_a_water_table_loads_the_stem_and_lifts_the_base():
    dry, wet = _wall(surcharge=0.0), _wall(surcharge=0.0,
                                           water_table_depth=1.5)
    hw = 4.05 - 1.5
    Pw = 0.5 * 9.81 * hw ** 2
    assert wet.loads.stem["Hs"] == pytest.approx((Pw, Pw * hw / 3.0),
                                                 rel=1e-3)
    assert (wet.elements["stem"]["Mu_kNm_per_m"]
            > dry.elements["stem"]["Mu_kNm_per_m"])
    # the uplift is part of what pushes up on the slab, and the contact
    # pressure with it carries the weights less the uplift
    row = wet.elements["heel"]["_rows"][0]
    st = row["state"]
    assert st["uplift"] is not None
    U_force = _segment_force(st["uplift"], 0.0, 3.0)[0]
    assert U_force == pytest.approx(0.5 * 9.81 * (4.5 - 1.5) * 3.0,
                                    rel=1e-3)
    assert _segment_force(st["pressure"], 0.0, 3.0)[0] == pytest.approx(
        st["V"])


def test_a_sloping_backfill_and_a_dead_surcharge_run():
    r = _wall(backfill_slope=15.0, dead_load_surcharge=8.0)
    parts = r.elements["heel"]["_rows"][0]["state"]["parts"]
    assert parts["wedge"] > 0 and parts["tip"] > 0
    # the dead-load surcharge is an earth load: it is in case 1
    flat = _wall(backfill_slope=15.0)
    c1 = {(c["case"], c["column"]): c
          for c in r.elements["stem"]["load_cases"]}
    f1 = {(c["case"], c["column"]): c
          for c in flat.elements["stem"]["load_cases"]}
    assert (c1["1", "serviceability"]["M_kNm_per_m"]
            > f1["1", "serviceability"]["M_kNm_per_m"])


def test_bad_inputs_are_refused_with_the_reason():
    with pytest.raises(ValueError, match="Table\\s+2-3"):
        _wall(joint_spacing=0.0)
    with pytest.raises(ValueError, match="fc_MPa"):
        _wall(fc_MPa=0.0)
    with pytest.raises(ValueError, match="beta1"):
        _wall(beta1=0.9)
    assert _wall(beta1=0.80).materials["beta1"] == 0.80
    noted = _wall(lap_length_mm=900.0)
    assert any("900 mm" in n and "not checked" in n
               for n in noted.reinforcement["notes"])
