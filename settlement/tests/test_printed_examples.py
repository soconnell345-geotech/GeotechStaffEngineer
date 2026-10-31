"""GM-285: the settlement module against the printed worked examples of
GEC-6 Appendix B, FHWA NHI-06-088 / NHI-06-089 and UFS 3-220-10, and
against the printed tables and charts it reads (NHI-06-089 Table 8-11,
UFS Fig. 5-16, UFS Fig. 4-10). GEC-6 Appendix B Example 1 (Hough) is
V-022 in validation_examples. Each example's own units are kept: the
module's formulas hold in any consistent set."""

import math

import pytest

from settlement.consolidation import (
    ConsolidationLayer, consolidation_settlement_layer,
)
from settlement.hough import HoughLayer, hough_settlement
from settlement.immediate import (
    SchmertmannLayer, elastic_settlement, gec6_influence_factor,
    schmertmann_terms,
)
from settlement.secondary import secondary_settlement
from settlement.stress_distribution import (
    boussinesq_rectangular, stress_at_depth, westergaard_center_strip,
    westergaard_influence_factor,
)
from settlement.time_rate import (
    degree_of_consolidation, time_for_consolidation,
)


# -- Schmertmann: NHI-06-089 Example 8-2 and Table 8-11 ----------------------

#: Example 8-2 (pp. 8-48 to 8-52), in ft and psf: a 6 ft x 24 ft footing
#: 3 ft deep, applied stress 2,000 psf, p_o = 345 psf, dp = 1,655 psf. The
#: layers' XE (tsf) from the example's table, x 2,000 psf/tsf.
EX_8_2_LAYERS = [SchmertmannLayer(0.0, 3.0, 142 * 2000.0),
                 SchmertmannLayer(3.0, 4.0, 426 * 2000.0),
                 SchmertmannLayer(4.0, 8.0, 426 * 2000.0),
                 SchmertmannLayer(8.0, 16.0, 1159 * 2000.0)]
#: The example's p_op = 840 psf at the peak depth (4 ft below the base),
#: from its layers' unit weights; one unit weight giving it:
#: (840 - 345) / 4 = 123.75 pcf.
EX_8_2_GAMMA = (840.0 - 345.0) / 4.0


def test_nhi_example_8_2_terms():
    t = schmertmann_terms(1655.0, 345.0, 6.0, EX_8_2_LAYERS, L=24.0,
                          gamma_soil=EX_8_2_GAMMA)
    assert t["Iz_0"] == pytest.approx(0.133, abs=5e-4)     # I_ZB
    assert t["z_max"] == pytest.approx(16.0)               # D_I
    assert t["z_peak"] == pytest.approx(4.0)               # D_IP
    assert t["sigma_vp"] == pytest.approx(840.0)           # p_op
    assert t["Iz_peak"] == pytest.approx(0.64, abs=5e-3)   # I_ZP
    assert t["C1"] == pytest.approx(0.896, abs=5e-4)
    # sum Iz H / XE = 0.1760 in/tsf printed
    assert t["sum_Iz_dz_over_Es"] * 12 * 2000 == pytest.approx(0.1760,
                                                              abs=3e-4)


@pytest.mark.parametrize("years, printed_in", [(0.1, 0.130), (1.0, 0.156)])
def test_nhi_example_8_2_settlement(years, printed_in):
    t = schmertmann_terms(1655.0, 345.0, 6.0, EX_8_2_LAYERS, L=24.0,
                          gamma_soil=EX_8_2_GAMMA, time_years=years)
    assert t["Se"] * 12 == pytest.approx(printed_in, abs=1e-3)


#: NHI-06-089 Table 8-11 (p. 8-53), a sample of its rows: L/B, I_ZB,
#: D_IP / B and D_I / B, printed to three decimals.
TABLE_8_11 = [(1.00, 0.100, 0.500, 2.000), (2.00, 0.111, 0.556, 2.222),
              (3.25, 0.125, 0.625, 2.500), (5.50, 0.150, 0.750, 3.000),
              (7.75, 0.175, 0.875, 3.500), (9.50, 0.194, 0.972, 3.889),
              (10.0, 0.200, 1.000, 4.000), (25.0, 0.200, 1.000, 4.000)]


@pytest.mark.parametrize("lb, izb, dip, di", TABLE_8_11)
def test_nhi_table_8_11_diagram(lb, izb, dip, di):
    B = 2.0
    t = schmertmann_terms(100.0, 20.0, B, [SchmertmannLayer(0, 10 * B, 1e4)],
                          L=lb * B, gamma_soil=18.0)
    assert t["Iz_0"] == pytest.approx(izb, abs=6e-4)
    assert t["z_peak"] / B == pytest.approx(dip, abs=6e-4)
    assert t["z_max"] / B == pytest.approx(di, abs=6e-4)


def test_schmertmann_peak_not_capped():
    """The print gives Izp no upper limit (it was held to 1.0)."""
    t = schmertmann_terms(500.0, 0.0, 1.0, [SchmertmannLayer(0, 2, 1e4)],
                          gamma_soil=10.0)
    assert t["Iz_peak"] == pytest.approx(0.5 + 0.1 * math.sqrt(500 / 5.0))
    assert t["Iz_peak"] > 1.0


# -- Consolidation: NHI-06-089 Example 8-3, UFS Fig. 5-13 --------------------

def test_nhi_example_8_3():
    """10 ft x 10 ft footing, 130 kips (1,300 psf); NC clay 10 ft thick,
    centre 15 ft below the base, p_o = 2,145 psf, e_o 0.75, Cc 0.4: the
    2:1 dp = 208 psf and dH = 0.09 ft (1.1 in), pp. 8-54, 8-55."""
    ds = stress_at_depth(1300.0, 10.0, 10.0, 15.0, method="2:1")
    assert ds == pytest.approx(208.0)
    clay = ConsolidationLayer(thickness=10.0, depth_to_center=15.0, e0=0.75,
                              Cc=0.4, Cr=0.04, sigma_v0=2145.0)
    dH = consolidation_settlement_layer(clay, ds)
    assert dH == pytest.approx(0.09, abs=5e-3)
    assert dH * 12 == pytest.approx(1.1, abs=0.05)


def test_ufs_fig_5_13():
    """sigma'z0 3,000 psf, sigma'p 3,500, dsigma 1,000, H0 10 ft, C_er
    0.015, C_ec 0.2: s_c = 0.13 ft or 1.5 in (p. 242). C_e = C / (1 + e0)
    (UFS Eqs. 5-10, 5-11), here with e0 = 1."""
    clay = ConsolidationLayer(thickness=10.0, depth_to_center=5.0, e0=1.0,
                              Cc=0.4, Cr=0.03, sigma_v0=3000.0,
                              sigma_p=3500.0)
    sc = consolidation_settlement_layer(clay, 1000.0)
    assert sc == pytest.approx(
        (0.015 * math.log10(3500 / 3000) + 0.2 * math.log10(4000 / 3500))
        * 10.0, rel=1e-12)
    assert sc == pytest.approx(0.13, abs=5e-3)
    assert sc * 12 == pytest.approx(1.5, abs=0.05)


def test_slightly_overconsolidated_uses_recompression():
    """GM-285: an OCR within 5 % of 1 is no longer taken as normally
    consolidated; the recompression part (GEC-6 Eq. 5-22) runs into the
    NC form as sigma_p falls to sigma_v0."""
    nc = ConsolidationLayer(2.0, 1.0, 1.0, 0.4, 0.04, 100.0)
    oc = ConsolidationLayer(2.0, 1.0, 1.0, 0.4, 0.04, 100.0, sigma_p=103.0)
    s_nc = consolidation_settlement_layer(nc, 50.0)
    s_oc = consolidation_settlement_layer(oc, 50.0)
    assert s_oc == pytest.approx(
        2.0 / 2.0 * (0.04 * math.log10(1.03) + 0.4 * math.log10(150 / 103)))
    assert s_oc < s_nc


def test_underconsolidated_is_flagged():
    """sigma_p below sigma_v0 reads as underconsolidated (UFS Sec. 5-2.3):
    the analysis says so; the arithmetic is as before."""
    from settlement.analysis import SettlementAnalysis
    uc = ConsolidationLayer(2.0, 1.0, 1.0, 0.4, 0.04, 100.0, sigma_p=90.0)
    assert uc.is_underconsolidated and not uc.is_normally_consolidated
    assert consolidation_settlement_layer(uc, 50.0) == pytest.approx(
        2.0 / 2.0 * (0.04 * math.log10(0.9) + 0.4 * math.log10(150 / 90)))
    r = SettlementAnalysis(q_applied=100, B=2, L=2,
                           consolidation_layers=[uc]).compute()
    assert any("underconsolidated" in w and "5-2.3" in w for w in r.warnings)
    nc = ConsolidationLayer(2.0, 1.0, 1.0, 0.4, 0.04, 100.0)
    r = SettlementAnalysis(q_applied=100, B=2, L=2,
                           consolidation_layers=[nc]).compute()
    assert not any("underconsolidated" in w for w in r.warnings)


# -- Secondary compression: UFS Fig. 5-24 ------------------------------------

def test_ufs_fig_5_24():
    """Organic clay H0 = 10 ft, e0 = 6, C_alpha = 0.16 (C_ae 0.023),
    t_p = 1 year, 50 years: s_s = 0.4 ft (p. 255)."""
    ss = secondary_settlement(0.16, 10.0, 1.0, 50.0, e0=6.0)
    assert ss == pytest.approx(0.16 / 7 * 10 * math.log10(50), rel=1e-12)
    assert ss == pytest.approx(0.4, abs=0.02)


# -- Time rate: UFS Fig. 5-16, NHI-06-088 Table 7-4 and Eq. 7-8 --------------

#: UFS 3-220-10 Fig. 5-16 (p. 245): U and T as tabulated, with NHI-06-088
#: Table 7-4's (p. 7-32) last rows.
FIG_5_16 = [(0.1, 0.008), (0.2, 0.031), (0.3, 0.071), (0.4, 0.126),
            (0.5, 0.197), (0.6, 0.287), (0.7, 0.403), (0.8, 0.567),
            (0.9, 0.848), (0.931, 1.000), (0.98, 1.500), (0.994, 2.000)]


@pytest.mark.parametrize("U, T", FIG_5_16)
def test_degree_of_consolidation_as_tabulated(U, T):
    assert degree_of_consolidation(T) / 100 == pytest.approx(U, abs=1.5e-3)
    if U <= 0.98:   # past it, U printed to 0.1 % no longer fixes T
        assert time_for_consolidation(100 * U, 1.0, 1.0) == pytest.approx(
            T, abs=1.5e-3 + 0.002 * T)


def test_u95_by_the_printed_approximation():
    """The figure tabulates T = 1.163 at U = 0.95; its own approximation,
    which the module uses, gives 1.129 (U(1.163) = 0.954)."""
    T95 = time_for_consolidation(95.0, 1.0, 1.0)
    assert T95 == pytest.approx(-4 / math.pi**2
                                * math.log(math.pi**2 / 8 * 0.05))
    assert T95 == pytest.approx(1.129, abs=1e-3)


def test_switch_at_t_0_2():
    """UFS Fig. 5-16 changes approximation at T = 0.2."""
    assert degree_of_consolidation(0.2) == pytest.approx(
        100 * 2 / math.sqrt(math.pi) * math.sqrt(0.2))
    assert degree_of_consolidation(0.25) == pytest.approx(
        100 * (1 - 8 / math.pi**2 * math.exp(-math.pi**2 * 0.25 / 4)))


def test_nhi_088_t90_example():
    """t90 = 0.848 (10 ft)^2 / (0.2 ft2/day) = 424 days (p. 7-38)."""
    assert time_for_consolidation(90.0, 0.2, 10.0) == pytest.approx(424.0,
                                                                   abs=0.5)


# -- Hough: GEC-6 Appendix B Examples 2, 3 and 4 -----------------------------

def test_gec6_example_2_hough_layers():
    """Example 2 (pp. B-27 to B-29), the structural fill at B = 2 m, L = 25 m,
    q = 100 kPa: Layer 1 11.2 mm, Layer 2 4 mm."""
    layers = [HoughLayer(2.0, 1.0, 20.5, 110), HoughLayer(2.57, 3.28, 67.3, 110)]
    r = hough_settlement(layers, 100.0, 2.0, 25.0)
    assert [lay["delta_sigma_kPa"] / 100 for lay in r.layers] == \
        pytest.approx([0.64, 0.33], abs=6e-3)                    # Table B2-3
    assert r.layers[0]["settlement_mm"] == pytest.approx(11.2, abs=0.1)
    assert r.layers[1]["settlement_mm"] == pytest.approx(4.0, abs=0.5)


EX_3_LAYERS = [HoughLayer(2.0, 1.0, 89.1, 110), HoughLayer(2.57, 3.28, 135.9, 110),
               HoughLayer(3.0, 6.07, 188.2, 30), HoughLayer(3.0, 9.07, 240.1, 30),
               HoughLayer(3.0, 12.07, 280.7, 50), HoughLayer(3.0, 15.07, 310.1, 58)]
#: Table B3-4 (p. B-51): settlement (mm) by q (kPa) for B = 3, 4, 5 m.
TABLE_B3_4 = {100: (19, 21, 24), 200: (33, 38, 42), 300: (46, 53, 59)}


@pytest.mark.parametrize("q", sorted(TABLE_B3_4))
def test_gec6_example_3_table_b3_4(q):
    """Example 3, L = 25 m, six layers (pp. B-46 to B-51); the example
    rounds dsigma/q to two figures and each layer to the mm."""
    for B, printed in zip((3.0, 4.0, 5.0), TABLE_B3_4[q]):
        r = hough_settlement(EX_3_LAYERS, q, B, 25.0)
        assert r.total_mm == pytest.approx(printed, abs=1.0)


def test_gec6_example_3_worked_layers():
    r = hough_settlement(EX_3_LAYERS, 100.0, 3.0, 25.0)
    assert [round(lay["settlement_mm"]) for lay in r.layers] == \
        [5, 3, 6, 3, 1, 1]


EX_4_LAYERS = [HoughLayer(3.0, 1.5, 58.8, 108), HoughLayer(1.5, 3.75, 95.6, 115)]
#: Table B4-3 (p. B-70): settlement (mm) by q (kPa), B = 3.0, 4.3, 5.5,
#: 6.7 m, a strip (2:1, q B / (B + z), Fig. 5-12).
TABLE_B4_3 = {240: (20, 22, 23, 23), 290: (22, 24, 25, 26),
              335: (24, 26, 27, 28)}


@pytest.mark.parametrize("q", sorted(TABLE_B4_3))
def test_gec6_example_4_table_b4_3(q):
    for B, printed in zip((3.0, 4.3, 5.5, 6.7), TABLE_B4_3[q]):
        r = hough_settlement(EX_4_LAYERS, q, B, math.inf)
        assert r.total_mm == pytest.approx(printed, abs=0.6)


def test_gec6_example_4_worked_layers():
    r = hough_settlement(EX_4_LAYERS, 240.0, 3.0, math.inf)
    assert [lay["delta_sigma_kPa"] / 240 for lay in r.layers] == \
        pytest.approx([0.67, 0.44], abs=5e-3)
    assert [round(lay["settlement_mm"]) for lay in r.layers] == [16, 4]


# -- Elastic, Eq. 5-36: GEC-6 Appendix B Examples 4 and 5 (rock) -------------

def test_gec6_example_4_shale():
    """Cd 1.83 (Table 5-12, L/B = 5, read in the average column), dsigma
    159 kPa over B = 10 m, nu 0.14, alpha_E E_IR = 0.53 x 9.8e6 kPa:
    0.5 mm (p. B-70)."""
    cd = gec6_influence_factor("rectangular", point="average", B=1, L=5)[0]
    assert cd == 1.83
    s = elastic_settlement(159.0, 10.0, 0.53 * 9.8e6, 0.14, Iw=cd)
    assert s * 1000 == pytest.approx(0.5, abs=0.05)


def test_gec6_example_5_granite():
    """Cd 1.78 (Table 5-12, L/B = 3, the centre column), 10,200 kPa,
    B = 1 m, nu 0.2, 0.99 x 52e6 kPa: 0.3 mm (p. B-89)."""
    cd = gec6_influence_factor("rectangular", point="center", B=1, L=3)[0]
    assert cd == 1.78
    s = elastic_settlement(10200.0, 1.0, 0.99 * 52e6, 0.2, Iw=cd)
    assert s * 1000 == pytest.approx(0.3, abs=0.05)


# -- Stress: UFS Table 4-2 and Fig. 4-10 -------------------------------------

def test_boussinesq_corner_ufs_table_4_2():
    """x = y = z: (1/2 pi)[atan(1/sqrt 3) + (1/sqrt 3)(1/2 + 1/2)]."""
    expected = (math.atan(1 / math.sqrt(3)) + 1 / math.sqrt(3)) / (2 * math.pi)
    assert boussinesq_rectangular(1.0, 2.0, 2.0, 2.0) == pytest.approx(
        expected, rel=1e-12)
    assert expected == pytest.approx(0.1752, abs=1e-4)


#: Points read off UFS Fig. 4-10 (p. 194): (m, n, I).
FIG_4_10 = [(0.1, 10.0, 0.0223), (0.5, 0.5, 0.0540), (1.0, 1.0, 0.1160),
            (1.0, 10.0, 0.1514), (2.0, 2.0, 0.1741), (3.0, 3.0, 0.1979),
            (0.4, 2.0, 0.0768), (math.inf, 1.0, 0.1517),
            (math.inf, 10.0, 0.2383)]


@pytest.mark.parametrize("m, n, printed", FIG_4_10)
def test_westergaard_fig_4_10(m, n, printed):
    assert westergaard_influence_factor(m, n) == pytest.approx(printed,
                                                               abs=1e-9)
    assert westergaard_influence_factor(n, m) == pytest.approx(printed,
                                                               abs=1e-9)


def test_westergaard_between_and_beyond_the_curves():
    f = westergaard_influence_factor
    # between curves: within the neighbouring nodes, rising with m and n
    assert f(1.0, 1.0) < f(1.1, 1.0) < f(1.2, 1.0)
    assert f(1.0, 1.0) < f(1.0, 1.5) < f(1.0, 2.0)
    # beyond n = 10 toward the m = infinity curve, and 0.25 at both infinite
    assert f(1.0, 10.0) < f(1.0, 40.0) < f(1.0, math.inf)
    assert f(math.inf, math.inf) == 0.25
    assert f(1e9, 1e9) == pytest.approx(0.25, abs=1e-6)
    # below 0.1 in proportion to zero
    assert f(0.05, 2.0) == pytest.approx(0.5 * f(0.1, 2.0))
    # the strip is the limit of a long rectangle
    assert westergaard_center_strip(100.0, 2.0, 1.0) == pytest.approx(
        stress_at_depth(100.0, 2.0, 1e9, 1.0, method="westergaard"))
