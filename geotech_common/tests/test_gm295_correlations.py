"""GM-295: the soil correlations as the manuals we hold print them.

Each pin is the printed equation or table at made-up inputs (no worked
example in the held manuals exercises these), with the page in the test.
"""

import math
import warnings

import pytest

from geotech_common.soil_profile import (
    GroundwaterCondition, SoilLayer, SoilProfile,
)
from geotech_common.soil_properties import (
    phi_to_K0, spt_n1_60, spt_overburden_correction, spt_to_cu,
    spt_to_phi, spt_to_relative_density,
)


# ------------------------------------------------------------- SPT -> phi'

def test_peck_is_wolff_eq_8_5_on_n1_60():
    # UFC 3-220-10 (2026) Eq 8-5, p. 399: 27.1 + 0.3 N - 0.00054 N^2
    assert spt_to_phi(30) == pytest.approx(27.1 + 9.0 - 0.486)
    assert spt_to_phi(0) == pytest.approx(27.1)
    # Fig 8-4 draws the Peck curve at about 36 deg at N1,60 = 30 and
    # about 43 deg at 60
    assert spt_to_phi(60) == pytest.approx(43.156)


def test_peck_sits_inside_the_aashto_ranges():
    # GEC-12 Table 5-5 (p. 123), (N1)60 against phi' (AASHTO 2014)
    for n, lo, hi in ((4, 27, 32), (10, 30, 35), (30, 35, 40), (50, 38, 43)):
        assert lo <= spt_to_phi(n) <= hi


def test_peck_holds_past_the_chart_and_says_so():
    with pytest.warns(UserWarning, match="Fig 8-4"):
        assert spt_to_phi(80) == pytest.approx(spt_to_phi(60))


def test_hatanaka_uchida_gec5_eq_7_23():
    # GEC-5 Eq 7.23 (p. 7-55), 60 % hammer: sqrt(15.4 (N1)60) + 20
    assert spt_to_phi(20, "hatanaka_uchida") == pytest.approx(
        math.sqrt(308.0) + 20.0)


def test_kulhawy_mayne_gec5_eq_7_22():
    # GEC-5 Eq 7.22 (p. 7-54): tan phi' = [N60/(12.2 + 20.3 s'v/pa)]^0.34
    phi = spt_to_phi(20, "kulhawy_mayne", sigma_v=100.0)
    assert math.tan(math.radians(phi)) == pytest.approx((20 / 32.5) ** 0.34)
    # Fig 7-41 (p. 7-55): N60 = 20 at 100 kPa reads close to the 40 deg line
    assert phi == pytest.approx(40.3, abs=0.1)
    with pytest.raises(ValueError, match="sigma_v"):
        spt_to_phi(20, "kulhawy_mayne")


def test_kulhawy_chen_gec12_eq_5_16():
    # GEC-12 Eq. 5-16 (p. 124): 27.5 + 9.2 log (N1)60
    assert spt_to_phi(10, "kulhawy_chen") == pytest.approx(36.7)
    with pytest.raises(ValueError, match="log"):
        spt_to_phi(0, "kulhawy_chen")


def test_shioi_fukui_ufc_eq_8_2_and_8_3():
    # UFC 3-220-10 p. 397
    assert spt_to_phi(20, "shioi_fukui_roads") == pytest.approx(
        math.sqrt(300.0) + 15.0)
    assert spt_to_phi(20, "shioi_fukui_buildings") == pytest.approx(33.0)


def test_meyerhof_reads_table_7_5_between_its_band_edges():
    # GEC-5 Table 7-5 (p. 7-54): 30/35/40/45 deg at N = 4/10/30/50
    assert spt_to_phi(4, "meyerhof") == pytest.approx(30.0)
    assert spt_to_phi(10, "meyerhof") == pytest.approx(35.0)
    assert spt_to_phi(20, "meyerhof") == pytest.approx(37.5)
    assert spt_to_phi(50, "meyerhof") == pytest.approx(45.0)
    for n in (3, 51):          # the table prints only "< 30" and "> 45"
        with pytest.raises(ValueError, match="Table 7-5"):
            spt_to_phi(n, "meyerhof")


# --------------------------------------------------------------- SPT -> su

def test_terzaghi_peck_matches_nhi_table_4_2():
    # NHI-06-088 Table 4-2 (p. 4-5): qu 25/50/100 kPa at N60 2/4/8, su = qu/2
    for n, qu in ((2, 25.0), (4, 50.0), (8, 100.0)):
        assert spt_to_cu(n) == pytest.approx(qu / 2)


def test_hara_as_ufc_fig_8_32_prints_it():
    # su = 0.297 N^0.72 kgf/cm^2; the figure draws it near 800 kPa at N = 100
    assert spt_to_cu(100, "hara") == pytest.approx(802.2, abs=0.5)
    assert spt_to_cu(10, "hara") == pytest.approx(
        0.297 * 98.0665 * 10 ** 0.72)


def test_stroud_butler_ufc_eq_8_26():
    # su/N = 8910/PI^3 + 4.36 kPa (p. 428)
    assert spt_to_cu(10, "stroud_butler", PI=30) == pytest.approx(
        10 * (8910 / 27000 + 4.36))
    with pytest.raises(ValueError, match="PI"):
        spt_to_cu(10, "stroud_butler")
    with pytest.warns(UserWarning, match="Fig 8-31"):
        spt_to_cu(10, "stroud_butler", PI=10)


# ------------------------------------------------------- relative density

def test_density_index_en_1997_2_table_f1():
    # EN 1997-2 Table F.1 (p. 125): 15/35/65/85/100 % at 3/8/25/42/58
    for n, d in ((3, 15.0), (8, 35.0), (25, 65.0), (42, 85.0), (58, 100.0)):
        assert spt_to_relative_density(n) == pytest.approx(d)
    assert spt_to_relative_density(16.5) == pytest.approx(50.0)
    assert spt_to_relative_density(70) == 100.0


def test_density_index_skempton_table_f2():
    # EN 1997-2 Table F.2: (N1)60/ID^2 = 55 natural, 40 recent fill, 35 lab
    assert spt_to_relative_density(22, "skempton") == pytest.approx(
        100 * math.sqrt(22 / 55))
    assert spt_to_relative_density(10, "skempton", age="recent_fill") == \
        pytest.approx(50.0)
    assert spt_to_relative_density(40, "skempton", age="laboratory") == 100.0


def test_density_index_no_longer_saturates():
    # before GM-295 every N60 above about 0.5 gave 100 %
    assert spt_to_relative_density(10) < 50.0


# ----------------------------------------------------------------- K0, C_N

def test_k0_overconsolidated_nhi_eq_10_3():
    # NHI-06-089 Eq 10-3 (p. 10-11): (1 - sin phi') OCR^sin phi'
    assert phi_to_K0(30.0) == pytest.approx(0.5)
    assert phi_to_K0(30.0, OCR=4.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="OCR"):
        phi_to_K0(30.0, OCR=0.5)


def test_peck_overburden_correction_nhi_eq_3_3():
    # NHI-06-088 Eq 3-3 (p. 3-56): 0.77 log10(20/p_o), p_o in tsf, <= 2.0
    tsf = 95.76052
    assert spt_overburden_correction(10, tsf, "peck")["CN"] == pytest.approx(
        0.77 * math.log10(20.0))
    r = spt_overburden_correction(10, 0.5 * tsf, "peck")
    assert r["CN"] == pytest.approx(0.77 * math.log10(40.0))
    assert r["N1"] == pytest.approx(10 * r["CN"])
    assert spt_overburden_correction(10, 2.0, "peck")["CN"] == 2.0
    with pytest.raises(ValueError, match="20 tsf"):
        spt_overburden_correction(10, 2000.0, "peck")
    chain = spt_n1_60(15, 50.0, overburden_method="peck")
    assert chain["N1_60"] == pytest.approx(
        15 * 0.77 * math.log10(20 * tsf / 50.0))


def test_liao_whitman_default_is_unchanged():
    # GEC-5 Eq 5.10 (p. 5-31) with pa = 100 kPa; cap 1.7 (Youd et al. 2001)
    assert spt_overburden_correction(10, 50.0)["CN"] == pytest.approx(
        math.sqrt(2.0))
    assert spt_overburden_correction(10, 20.0)["CN"] == 1.7


# ------------------------------------------------- soil_profile uses N1,60

def test_profile_estimates_phi_from_n1_60():
    layers = [SoilLayer(0, 4, "Sand", N60=8, uscs="SP", gamma=18.0),
              SoilLayer(4, 10, "Sand", N60=20, uscs="SP", gamma=19.0)]
    p = SoilProfile(layers=layers, groundwater=GroundwaterCondition(depth=20))
    p.fill_missing_from_correlations()
    # mid-depth 7 m: sigma'v = 4*18 + 3*19 = 129 kPa (dry)
    n1 = 20 * math.sqrt(100.0 / 129.0)
    assert p.layers[1].phi == pytest.approx(spt_to_phi(n1))
    assert "N1,60" in p.layers[1].get_estimation_log()["phi"]


def test_profile_without_unit_weights_leaves_phi_unestimated():
    rock = SoilLayer(0, 2, "Rock", is_rock=True)
    sand = SoilLayer(2, 6, "Sand", N60=15, uscs="SP", gamma=19.0)
    p = SoilProfile(layers=[rock, sand],
                    groundwater=GroundwaterCondition(depth=20))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log = p.fill_missing_from_correlations()
    assert p.layers[1].phi is None
    assert any("phi not estimated" in line for line in log)
