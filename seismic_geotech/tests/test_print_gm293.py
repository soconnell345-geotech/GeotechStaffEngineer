"""seismic_geotech against the print held (GM-293): each pin is a printed
table cell, equation, chart reading or limit, cited to the page
(module_work/gm293/CHECK.md); the inputs are made up.
"""

import math

import pytest

from seismic_geotech.site_class import (
    classify_site, compute_n_bar, site_coefficients,
)
from seismic_geotech.liquefaction import (
    compute_CSR, evaluate_liquefaction, stress_reduction_rd,
)
from seismic_geotech.residual_strength import (
    Sr_idriss_boulanger, Sr_ratio_idriss_boulanger,
    post_liquefaction_strength,
)
from seismic_geotech.mononobe_okabe import (
    mononobe_okabe_KAE, mononobe_okabe_KPE, passive_delta_warning,
)
from seismic_geotech.dynamic_properties import (
    gmax_cpt_clay_mayne_rix, gmax_hardin_black, hardin_k,
)


# ── Site class: GEC-12 Vol I Tables 7-26 to 7-29 (pp. 379-380) ──────

# Table 7-27 (Fpga vs PGA), 7-28 (Fa vs Ss), 7-29 (Fv vs S1), as printed
FPGA = {"A": [0.8] * 5, "B": [1.0] * 5, "C": [1.2, 1.2, 1.1, 1.0, 1.0],
        "D": [1.6, 1.4, 1.2, 1.1, 1.0], "E": [2.5, 1.7, 1.2, 0.9, 0.9]}
FA = FPGA
FV = {"A": [0.8] * 5, "B": [1.0] * 5, "C": [1.7, 1.6, 1.5, 1.4, 1.3],
      "D": [2.4, 2.0, 1.8, 1.6, 1.5], "E": [3.5, 3.2, 2.8, 2.4, 2.4]}
PGA_COLS = [0.1, 0.2, 0.3, 0.4, 0.5]
SS_COLS = [0.25, 0.5, 0.75, 1.0, 1.25]
S1_COLS = [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.mark.parametrize("sc", "ABCDE")
@pytest.mark.parametrize("j", range(5))
def test_site_factor_tables_cell_by_cell(sc, j):
    """All 75 cells of GEC-12 Tables 7-27 to 7-29 (p. 380)."""
    r = site_coefficients(sc, Ss=SS_COLS[j], S1=S1_COLS[j], pga=PGA_COLS[j])
    assert r.Fpga == FPGA[sc][j]
    assert r.Fa == FA[sc][j]
    assert r.Fv == FV[sc][j]


def test_fpga_straight_line_between_columns_and_held_beyond():
    """GEC-12 p. 379: straight-line interpolation; "PGA < 0.1" and
    "PGA > 0.5" columns hold beyond the table."""
    assert site_coefficients("E", 1.0, 0.3, pga=0.25).Fpga == \
        pytest.approx((1.7 + 1.2) / 2)
    assert site_coefficients("E", 1.0, 0.3, pga=0.05).Fpga == 2.5
    assert site_coefficients("E", 1.0, 0.3, pga=0.7).Fpga == 0.9


def test_no_fpga_without_pga_and_as_with_it():
    """Fpga is read against PGA only (Table 7-27); As = Fpga PGA (Eq.
    7-78, p. 381)."""
    r = site_coefficients("D", Ss=0.375, S1=0.15)
    assert r.Fpga is None and r.As is None
    assert "Fpga" in r.to_dict() and r.to_dict()["Fpga"] is None
    assert "As_g" not in r.to_dict()
    r = site_coefficients("D", Ss=0.375, S1=0.15, pga=0.15)
    assert r.Fpga == pytest.approx(1.5)
    assert r.As == pytest.approx(1.5 * 0.15)
    assert r.to_dict()["As_g"] == pytest.approx(0.225)


@pytest.mark.parametrize("vs30, cls", [
    (1524.5, "A"), (1524.0, "B"), (762.5, "B"), (762.0, "C"),
    (366.0, "C"), (365.7, "D"), (183.0, "D"), (182.8, "E")])
def test_vs30_boundaries_are_the_printed_ft_per_s(vs30, cls):
    """Table 7-26: 5,000, 2,500, 1,200 and 600 ft/s = 1524, 762, 365.76 and
    182.88 m/s (1 ft = 0.3048 m)."""
    assert classify_site(vs30=vs30) == cls


@pytest.mark.parametrize("su, cls", [
    (95.8, "C"), (95.7, "D"), (47.9, "D"), (47.8, "E")])
def test_su_boundaries_are_the_printed_ksf(su, cls):
    """Table 7-26: 2.0 and 1.0 ksf = 95.76 and 47.88 kPa."""
    assert classify_site(su_bar=su) == cls


def test_n_bar_takes_field_n_at_most_100():
    """GEC-7 p. 93: Ni is field N, limited to 100 bpf."""
    assert compute_n_bar([10.0, 20.0], [20.0, 150.0]) == \
        pytest.approx(30.0 / (10.0 / 20.0 + 20.0 / 100.0))


# ── rd: GEC-13 Vol I pp. 5-40 and 5-41 ───────────────────────────────

@pytest.mark.parametrize("z, rd", [
    (5.0, 1.0 - 0.00765 * 5.0),
    (9.144, 1.0 - 0.00765 * 9.144),
    (15.0, 1.174 - 0.0267 * 15.0),
    (22.86, 1.174 - 0.0267 * 22.86),
    (25.0, 0.744 - 0.008 * 25.0),
    (30.48, 0.744 - 0.008 * 30.48),
    (35.0, 0.5),
    (60.0, 0.5)])
def test_rd_four_lines_as_printed(z, rd):
    """Coefficients metric (z in m), limits printed as 30, 75 and 100 ft;
    0.5 below 100 ft."""
    assert stress_reduction_rd(z) == pytest.approx(rd)


def test_csr_is_eq_5_9():
    """CSR = 0.65 (amax/g)(sigma_v/sigma'_v) rd, GEC-13 Eq. 5-9."""
    assert compute_CSR(0.25, 400.0, 250.0, 35.0) == \
        pytest.approx(0.65 * 0.25 * 400.0 / 250.0 * 0.5)


def test_deep_layer_takes_rd_floor():
    r = evaluate_liquefaction([32.0], [12.0], [5.0], [19.0], 0.3, 2.0)[0]
    assert r["rd"] == 0.5


# ── Residual strength: GEC-12 Vol I Fig. 7-78 (p. 383) ───────────────

@pytest.mark.parametrize("N, ratio", [
    (0.0, 0.031), (5.0, 0.060), (10.0, 0.091), (15.0, 0.131),
    (18.0, 0.161), (26.0, 0.297), (28.0, 0.359), (29.0, 0.397)])
def test_significant_curve_read_off(N, ratio):
    """The "void redistribution effects could be significant" curve, as
    module_work/gm293/trace_fig7_78.py reads it off the figure (to 0.005,
    about 5 px)."""
    assert Sr_ratio_idriss_boulanger(N, "significant") == \
        pytest.approx(ratio, abs=0.005)


@pytest.mark.parametrize("N, ratio", [
    (11.0, 0.109), (13.0, 0.144), (14.0, 0.175), (15.0, 0.217),
    (16.81, 0.367)])
def test_negligible_curve_read_off(N, ratio):
    assert Sr_ratio_idriss_boulanger(N, "negligible") == \
        pytest.approx(ratio, abs=0.005)


def test_curves_agree_below_n_9():
    for N in (0.0, 3.0, 6.5, 9.0):
        assert Sr_ratio_idriss_boulanger(N, "negligible") == \
            Sr_ratio_idriss_boulanger(N, "significant")


@pytest.mark.parametrize("N, curve", [(29.2, "significant"),
                                      (17.2, "negligible")])
def test_refused_where_the_curve_leaves_the_chart(N, curve):
    with pytest.raises(ValueError, match="leaves the chart"):
        Sr_ratio_idriss_boulanger(N, curve)


def test_sr_is_ratio_times_stress():
    assert Sr_idriss_boulanger(10.0, 80.0) == \
        pytest.approx(80.0 * Sr_ratio_idriss_boulanger(10.0))
    assert post_liquefaction_strength(
        14.0, 80.0, "idriss_boulanger", void_redistribution="negligible") \
        == pytest.approx(80.0 * Sr_ratio_idriss_boulanger(14.0, "negligible"))


def test_residual_tool_offers_what_it_runs():
    from funhouse_agent.adapters.seismic_geotech import (
        METHOD_INFO, METHOD_REGISTRY)
    spec = METHOD_INFO["residual_strength"]["parameters"]
    assert spec["method"]["allowed_values"] == ["seed_harder",
                                                "idriss_boulanger"]
    for m in spec["method"]["allowed_values"]:
        METHOD_REGISTRY["residual_strength"](
            {"N160cs": 10.0, "sigma_v_eff": 80.0, "method": m})


# ── Mononobe-Okabe: UFC 3-220-20 Eq. 4-15, GEC-4 Eq. 44 ──────────────

def _ufc_4_15(phi, delta, kh, kv, theta, beta):
    """UFC 3-220-20 Eq. 4-15 (p. 214) as printed: theta the wall face,
    beta the backfill slope, psi = atan(kh / (1 - kv))."""
    r = math.radians
    phi, delta, theta, beta = r(phi), r(delta), r(theta), r(beta)
    psi = math.atan(kh / (1.0 - kv))
    root = math.sqrt(math.sin(phi + delta) * math.sin(phi - beta - psi)
                     / (math.cos(delta + theta + psi) * math.cos(beta - theta)))
    return (math.cos(phi - theta - psi) ** 2
            / (math.cos(psi) * math.cos(theta) ** 2
               * math.cos(delta + theta + psi) * (1.0 + root) ** 2))


@pytest.mark.parametrize("phi, delta, kh, kv, theta, beta", [
    (30, 0, 0.0, 0.0, 10, 0), (35, 20, 0.2, 0.1, 5, 10),
    (32, 21, 0.15, 0.0, -8, 5), (38, 25, 0.3, -0.1, 0, 12)])
def test_kae_is_ufc_eq_4_15_with_theta_minus_beta(phi, delta, kh, kv,
                                                  theta, beta):
    assert mononobe_okabe_KAE(phi, delta, kh, kv, beta_deg=-theta,
                              i_deg=beta) == \
        pytest.approx(_ufc_4_15(phi, delta, kh, kv, theta, beta), rel=1e-12)


def test_ufc_theta_10_gives_coulomb_0_4067():
    """UFC Fig. 4-8's theta = 10 deg (phi 30, delta 0, level): Coulomb
    Ka 0.4067, the module's beta = -10."""
    assert mononobe_okabe_KAE(30, 0, 0.0, 0.0, beta_deg=-10.0) == \
        pytest.approx(0.4067, abs=5e-5)


def _gec4_44(phi, delta, kh, kv, beta, i):
    """FHWA GEC-4 Eq. 44 (p. 114) with its D' bracket "1 -" (printed "1 +",
    which gives the active coefficient at kh = 0)."""
    r = math.radians
    phi, delta, beta, i = r(phi), r(delta), r(beta), r(i)
    th = math.atan(kh / (1.0 - kv))
    root = math.sqrt(math.sin(phi + delta) * math.sin(phi + i - th)
                     / (math.cos(delta - beta + th) * math.cos(i - beta)))
    return (math.cos(phi - th + beta) ** 2
            / (math.cos(th) * math.cos(beta) ** 2
               * math.cos(delta - beta + th) * (1.0 - root) ** 2))


@pytest.mark.parametrize("phi, delta, kh, kv, beta, i", [
    (30, 15, 0.2, 0.0, 0, 0), (34, 12, 0.15, 0.05, 6, 0),
    (36, 10, 0.1, 0.0, -5, 4)])
def test_kpe_is_gec4_eq_44_with_beta_negated(phi, delta, kh, kv, beta, i):
    assert mononobe_okabe_KPE(phi, delta, kh, kv, beta_deg=-beta,
                              i_deg=i) == \
        pytest.approx(_gec4_44(phi, delta, kh, kv, beta, i), rel=1e-12)


def test_passive_delta_default_and_the_04_limit():
    """KPE takes phi/2 unless delta_passive is given (NHI-06-089 p. 10-17)
    and warns above 0.4 phi (UFC 3-220-20 Fig. 4-8)."""
    from funhouse_agent.adapters.seismic_geotech import METHOD_REGISTRY as R
    r = R["seismic_earth_pressure"]({"phi": 34.0, "kh": 0.15,
                                     "include_passive": True})
    assert r["delta_passive_deg"] == 17.0
    assert r["KPE"] == pytest.approx(
        round(mononobe_okabe_KPE(34.0, 17.0, 0.15), 4))
    assert any("0.4" in w for w in r["warnings"])
    assert any(c["what"] == "wall friction delta for KPE"
               for c in r["choices"])
    r = R["seismic_earth_pressure"]({"phi": 34.0, "kh": 0.15,
                                     "include_passive": True,
                                     "delta_passive": 13.6})
    assert r["delta_passive_deg"] == 13.6
    assert not r.get("warnings")
    assert passive_delta_warning(34.0, 13.6) is None


# ── Gmax: GEC-5 Table 8-7 (p. 8-21), Eq. 8.14 and Table 8-4 (p. 8-18) ─

def test_mayne_rix_as_table_8_7():
    """Gmax = 49.4 qt^0.695 e^-1.130, Gmax and qt in MPa."""
    assert gmax_cpt_clay_mayne_rix(1.0, 1.0)["Gmax_kPa"] == \
        pytest.approx(49400.0)
    assert gmax_cpt_clay_mayne_rix(2.0, 1.5)["Gmax_kPa"] == \
        pytest.approx(49.4 * 2.0 ** 0.695 * 1.5 ** -1.130 * 1000.0)


@pytest.mark.parametrize("PI, k", [(0, 0.0), (20, 0.18), (40, 0.30),
                                   (60, 0.41), (80, 0.48), (100, 0.50),
                                   (150, 0.50), (30, 0.24)])
def test_hardin_k_table_8_4(PI, k):
    assert hardin_k(PI) == pytest.approx(k)


def test_hardin_1978_eq_8_14():
    """A = 625, n = 0.5, F(e) = 1/(0.3 + 0.7 e^2), OCR^k, pa = 100 kPa."""
    e, s, ocr = 0.9, 150.0, 3.0
    want = 625.0 / (0.3 + 0.7 * e ** 2) * ocr ** 0.30 * 100.0 ** 0.5 * s ** 0.5
    assert gmax_hardin_black(s, e, OCR=ocr, PI_pct=40.0)["Gmax_kPa"] == \
        pytest.approx(want)
    with pytest.raises(ValueError, match="PI"):
        gmax_hardin_black(s, e, OCR=ocr)
