"""V-061: cantilever retaining wall external stability, two printed
examples (GM-288).

UFC 3-220-20 (16 January 2025) App. B-1, the cantilever cut wall (pp. 596-
606), read from ``geotech-references/docs/ufc_3_220_20_2025.pdf``: H = 18 ft
(stem 16 ft on a 2 ft footing), toe 2 ft, stem 1 ft at the top and 1.5 ft
at the base with the FRONT face battered (Fig. B-2), the ground in front
3 ft above the underside of the footing (1 ft of soil over the toe, W_ff),
backfill and foundation 140 pcf, phi' 37 deg with Ka taken as 0.25, base
friction tan delta = 0.55, concrete 150 pcf, a 250 psf traffic surcharge
that drives every check but is not over the heel for overturning and
sliding (B-1.2), and Pv = 0. Table B-1 (p. 599) gives R = 2540 B - 4560
lb/ft; B-1.4 (p. 600) the middle third at B = 9.11 ft; B-1.5 (p. 601)
FSL = (1397 B - 2508)/6795, 1.5 at B = 9.09 ft; B-1.6 (pp. 601-602) the
bearing resultant with the surcharge, R = 2790 B - 5435, and at B = 7.50 ft
x0 = 1.805 ft and theta = 23.686 deg (Table B-2); Table B-4 (p. 604) at
B = 9.5 ft FSL 1.58 and x0/B 0.35; B-1.8 (p. 605) FOT = (1270 B^2 -
4840)/44145, 2.49 at B = 9.5 ft and 1 at B = 6.21 ft.

The example rounds Ka to 0.25 (Ph2 = 0.5 (0.25)(140)(18^2) = 5670 lb/ft);
phi' is set to 36.87 deg here, where Rankine's Ka is 0.25 exactly. Table
B-4's x0/B 0.35 is the resultant without the surcharge, B-1.8's x0 = 3.36
ft (3.354 ft unrounded), not B-1.6's with it (3.58 ft at B = 9.5 ft).

FHWA NHI-06-089 Vol II (2006) Example 10-3 (pp. 10-52 to 10-56), read from
``geotech-references/docs/FHWA NHI-06-089 Soils and Foundations Vol
2.pdf``:
stem 18 ft, base 2.3 ft thick and 13.1 ft wide, toe 2.3 ft, heel 8.5 ft,
stem 1.6 ft with a 0.7 ft FRONT batter, a 10 deg backslope (H = 21.8 ft on
the plane through the heel), phi 30 deg, 115 pcf, foundation phi_b 38 deg
with delta_b = 3/4 phi_b = 28.5 deg, concrete 150 pcf, the soil over the
toe neglected, Coulomb with delta = beta = 10 deg on that plane (p. 10-53;
p. 10-20). Printed: Ka 0.35, Pa 9,564.2, Ph 9,418.9, Pv 1,660.8 lb/ft; W
28,112.6 lb/ft, Mw 211,881.1 ft-lb/ft; FSs 1.72; d 5.55 ft, e 1.0 ft <
B/6; q_max 3,318.1, q_min 1,227.3 psf; FS against bearing 20,000/3,318.1
= 6.03 with the trapezoid (the example divides q_ult by q_max). The
example rounds Ka to 0.35 (Coulomb's is 0.3495), which puts its thrust
0.15 % above the module's.
"""

import math

import pytest

from retaining_walls.cantilever import analyze_cantilever_wall
from retaining_walls.geometry import CantileverWallGeometry

FT, PCF, PSF, PLF = 0.3048, 0.15708746, 0.04788026, 0.01459390
#: phi' at which Rankine's Ka is 0.25 exactly, as UFC B-1 rounds it
PHI_KA_025 = 2.0 * (45.0 - math.degrees(math.atan(0.5)))


def _ufc_b1(B_ft, **kw):
    geom = CantileverWallGeometry(
        wall_height=18 * FT, base_width=B_ft * FT, toe_length=2 * FT,
        stem_thickness_top=1 * FT, stem_thickness_base=1.5 * FT,
        base_thickness=2 * FT, stem_batter="front", embedment_depth=3 * FT,
        surcharge=250 * PSF)
    return analyze_cantilever_wall(
        geom, gamma_backfill=140 * PCF, phi_backfill=PHI_KA_025,
        delta_base=math.degrees(math.atan(0.55)), gamma_concrete=150 * PCF,
        **kw)


def _nhi_10_3(**kw):
    geom = CantileverWallGeometry(
        wall_height=20.3 * FT, base_width=13.1 * FT, toe_length=2.3 * FT,
        stem_thickness_top=1.6 * FT, stem_thickness_base=2.3 * FT,
        base_thickness=2.3 * FT, stem_batter="front", backfill_slope=10)
    return analyze_cantilever_wall(
        geom, gamma_backfill=115 * PCF, phi_backfill=30, phi_foundation=38,
        delta_base=28.5, gamma_concrete=150 * PCF,
        pressure_method="coulomb", q_allowable=20000 * PSF,
        bearing_pressure="linear", **kw)


# ---------------------------------------------------------------- UFC B-1

@pytest.mark.parametrize("B_ft", [6.21, 7.5, 9.09, 9.5])
def test_v061_ufc_b1_table_b1_weights(B_ft):
    """Table B-1: R = 2540 B - 4560 lb/ft (W_f 300 B, W_s1 2400 at 3 ft,
    W_s2 600 at 2.33 ft, W_bf 2240 (B - 3.5), W_ff 280 at 1 ft); Ph1 1125
    and Ph2 5670 lb/ft."""
    r = _ufc_b1(B_ft)
    assert r.overturning_terms["V"] / PLF == pytest.approx(
        2540 * B_ft - 4560, rel=1e-6)
    assert r.sliding_terms["driving"] / PLF == pytest.approx(
        1125 + 5670, rel=1e-6)
    w = {i["component"]: (i["W"] / PLF, i["x"] / FT)
         for i in r.vertical_forces}
    assert w["Stem (rect)"] == pytest.approx((2400, 3.0), rel=1e-6)
    assert w["Stem (taper)"] == pytest.approx((600, 2 + 1 / 3), rel=1e-6)
    assert w["Soil over toe"] == pytest.approx((280, 1.0), rel=1e-6)


def test_v061_ufc_b1_sliding():
    """B-1.5 and Table B-4: FSL = (1397 B - 2508)/6795, 1.5 at B = 9.09 ft
    and 1.58 at B = 9.5 ft."""
    for B_ft in (7.5, 9.09, 9.5):
        assert _ufc_b1(B_ft).sliding_terms["FOS"] == pytest.approx(
            (1397 * B_ft - 2508) / 6795, rel=2e-4)
    assert _ufc_b1(9.09).FOS_sliding == pytest.approx(1.50, abs=0.005)
    assert _ufc_b1(9.5).FOS_sliding == pytest.approx(1.58, abs=0.005)


def test_v061_ufc_b1_overturning_and_middle_third():
    """B-1.8: FOT = (1270 B^2 - 4840)/44145, 2.49 at B = 9.5 ft and 1 at
    B = 6.21 ft; B-1.4: the resultant at the middle third at B = 9.11 ft;
    Table B-4: x0/B 0.35 at B = 9.5 ft (x0 3.36 ft, B-1.8)."""
    for B_ft in (6.21, 9.5):
        assert _ufc_b1(B_ft).overturning_terms["FOS"] == pytest.approx(
            (1270 * B_ft ** 2 - 4840) / 44145, rel=1e-6)
    assert _ufc_b1(9.5).FOS_overturning == pytest.approx(2.49, abs=0.005)
    assert _ufc_b1(6.21).FOS_overturning == pytest.approx(1.0, abs=0.001)
    o = _ufc_b1(9.5).overturning_terms
    assert o["x_resultant"] / (9.5 * FT) == pytest.approx(0.35, abs=0.005)
    assert o["x_resultant"] / FT == pytest.approx(3.36, abs=0.01)
    assert _ufc_b1(9.11).overturning_terms["eccentricity"] / (9.11 * FT) \
        == pytest.approx(1 / 6, abs=2e-4)
    assert _ufc_b1(9.12).within_eccentricity_limit
    assert not _ufc_b1(9.10).within_eccentricity_limit


def test_v061_ufc_b1_bearing_resultant():
    """B-1.6 and Table B-2: with the surcharge over the heel, R = 2790 B -
    5435 lb/ft and at B = 7.50 ft x0 = 1.805 ft, theta = 23.686 deg; the
    uniform pressure R/(2 x0) (Eq. 4-27) is the module's q_eq = V/B'."""
    r = _ufc_b1(7.5, q_allowable=500.0)
    b = r.bearing_terms
    assert b["V"] / PLF == pytest.approx(2790 * 7.5 - 5435, rel=1e-6)
    assert b["x_resultant"] / FT == pytest.approx(1.805, abs=5e-4)
    theta = math.degrees(math.atan(r.sliding_terms["driving"] / b["V"]))
    assert theta == pytest.approx(23.686, abs=5e-4)
    assert b["bearing_pressure"] == "uniform"
    assert b["q_eq"] / PSF == pytest.approx(
        (2790 * 7.5 - 5435) / (2 * 1.805), rel=5e-4)


# ------------------------------------------------------ NHI-06-089 Ex 10-3

def test_v061_nhi_10_3_thrust():
    """Coulomb with delta = beta = 10 deg on the plane through the heel:
    Ka 0.35, H 21.8 ft, Pa 9,564.2, Ph 9,418.9, Pv 1,660.8 lb/ft, Ph at
    7.27 ft (the example's Ka is rounded to 0.35)."""
    e = _nhi_10_3().earth_pressure
    assert e["Ka"] == pytest.approx(0.35, abs=0.001)
    assert e["delta_deg"] == pytest.approx(10.0)
    assert e["H_active"] / FT == pytest.approx(21.8, abs=0.005)
    assert e["Pa"] / PLF == pytest.approx(9564.2, rel=0.002)
    assert e["Ph"] / PLF == pytest.approx(9418.9, rel=0.002)
    assert e["Pv"] / PLF == pytest.approx(1660.8, rel=0.002)
    assert e["z_Pa"] / FT == pytest.approx(7.27, abs=0.005)


def test_v061_nhi_10_3_weights_and_checks():
    """W 28,112.6 lb/ft and Mw 211,881.1 ft-lb/ft about the toe; FSs 1.72;
    d 5.55 ft, e 1.0 ft < B/6 = 2.18 ft; q_max 3,318.1 and q_min 1,227.3
    psf; FS against bearing 6.03 (20,000 psf over q_max)."""
    r = _nhi_10_3()
    weights = [i for i in r.vertical_forces if i["kind"] == "weight"]
    assert sum(i["W"] for i in weights) / PLF == pytest.approx(
        28112.6, rel=1e-4)
    assert sum(i["W"] * i["x"] for i in weights) / PLF / FT \
        == pytest.approx(211881.1, rel=1e-4)
    assert r.FOS_sliding == pytest.approx(1.72, abs=0.005)
    o, b = r.overturning_terms, r.bearing_terms
    assert o["x_resultant"] / FT == pytest.approx(5.55, abs=0.005)
    assert o["eccentricity"] / FT == pytest.approx(1.0, abs=0.005)
    assert r.within_eccentricity_limit
    assert b["q_toe"] / PSF == pytest.approx(3318.1, rel=0.005)
    assert b["q_heel"] / PSF == pytest.approx(1227.3, rel=0.005)
    assert r.FOS_bearing == pytest.approx(6.03, abs=0.01)
