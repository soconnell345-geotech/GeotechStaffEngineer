"""Tests for small-strain dynamic property correlations.

Numeric pins were cross-checked against an independent implementation
(groundhog 0.15.0, GPL — used only as a numerical oracle before its
removal from the dependency tree, never as source) and against hand
calculation of the published equations.
"""

import math

import pytest

from seismic_geotech.dynamic_properties import (
    gmax_from_vs,
    gmax_cpt_sand_rix_stokoe,
    gmax_cpt_clay_mayne_rix,
    gmax_hardin_black,
    gmax_clay_andersen,
    modulus_reduction_ishibashi_zhang,
)


class TestGmaxFromVs:
    def test_elastic_identity(self):
        # G = rho Vs^2: gamma=19 kN/m3, Vs=200 m/s -> rho=1936.8, G=77.47 MPa
        r = gmax_from_vs(Vs=200.0, unit_weight=19.0)
        assert r["rho_kg_m3"] == pytest.approx(1936.799, rel=1e-4)
        assert r["Gmax_kPa"] == pytest.approx(77471.97, rel=1e-4)

    def test_zero_vs(self):
        assert gmax_from_vs(0.0, 18.0)["Gmax_kPa"] == 0.0

    def test_bad_unit_weight(self):
        with pytest.raises(ValueError):
            gmax_from_vs(200.0, 5.0)


class TestGmaxCpt:
    def test_rix_stokoe_pin(self):
        # qc=10 MPa, sigma_v'=100 kPa: 1634*10000^0.25*100^0.375 = 91886.6
        r = gmax_cpt_sand_rix_stokoe(qc_MPa=10.0, sigma_vo_eff_kPa=100.0)
        assert r["Gmax_kPa"] == pytest.approx(91886.57, rel=1e-4)

    def test_mayne_rix_pins(self):
        assert gmax_cpt_clay_mayne_rix(1.0)["Gmax_kPa"] == \
            pytest.approx(28121.91, rel=1e-4)
        assert gmax_cpt_clay_mayne_rix(5.0)["Gmax_kPa"] == \
            pytest.approx(241084.76, rel=1e-4)

    def test_negative_qc_rejected(self):
        with pytest.raises(ValueError):
            gmax_cpt_sand_rix_stokoe(-1.0, 100.0)


class TestGmaxHardinBlack:
    def test_pin(self):
        # p'=100, e=0.7, B=875: 875*100/(0.3+0.7*0.49)*1 = 136080.9
        r = gmax_hardin_black(sigma_m_eff_kPa=100.0, void_ratio=0.7)
        assert r["Gmax_kPa"] == pytest.approx(136080.87, rel=1e-4)

    def test_stress_scaling_sqrt(self):
        g1 = gmax_hardin_black(100.0, 0.7)["Gmax_kPa"]
        g4 = gmax_hardin_black(400.0, 0.7)["Gmax_kPa"]
        assert g4 / g1 == pytest.approx(2.0, rel=1e-9)


class TestGmaxAndersen:
    def test_pin(self):
        # PI=30, OCR=2, sigma_v'=100: (30+75/0.33)*sqrt(2)*100 = 36383.9
        r = gmax_clay_andersen(PI_pct=30.0, OCR=2.0, sigma_vo_eff_kPa=100.0)
        assert r["sigma_ref_kPa"] == pytest.approx(100.0)
        assert r["Gmax_kPa"] == pytest.approx(36383.86, rel=1e-4)

    def test_ocr_bounds(self):
        with pytest.raises(ValueError):
            gmax_clay_andersen(30.0, 0.5, 100.0)


class TestIshibashiZhang:
    def test_small_strain_full_modulus(self):
        # 1e-4 pct strain, PI=0: G/Gmax clipped at 1.0; damping = 1.2987 pct
        r = modulus_reduction_ishibashi_zhang(
            strain_pct=0.0001, PI_pct=0.0, sigma_m_eff_kPa=100.0)
        assert r["G_over_Gmax"] == pytest.approx(1.0)
        assert r["n"] == 0.0
        assert r["K"] == pytest.approx(0.98955, rel=1e-4)
        assert r["damping_pct"] == pytest.approx(1.2987, rel=1e-3)

    def test_plastic_soil_pin(self):
        # 0.1 pct strain, PI=30, sigma_m'=100 kPa (groundhog 0.15.0 oracle)
        r = modulus_reduction_ishibashi_zhang(
            strain_pct=0.1, PI_pct=30.0, sigma_m_eff_kPa=100.0)
        assert r["G_over_Gmax"] == pytest.approx(0.645706, rel=1e-4)
        assert r["K"] == pytest.approx(0.407161, rel=1e-4)
        assert r["m"] == pytest.approx(0.100134, rel=1e-4)
        assert r["n"] == pytest.approx(5.80617e-4, rel=1e-4)
        assert r["damping_pct"] == pytest.approx(5.30864, rel=1e-3)

    def test_monotone_decreasing_with_strain(self):
        vals = [modulus_reduction_ishibashi_zhang(s, 15.0, 100.0)["G_over_Gmax"]
                for s in (0.001, 0.01, 0.1, 1.0)]
        assert all(a >= b for a, b in zip(vals, vals[1:]))

    def test_higher_pi_less_reduction(self):
        lo = modulus_reduction_ishibashi_zhang(0.1, 0.0, 100.0)["G_over_Gmax"]
        hi = modulus_reduction_ishibashi_zhang(0.1, 50.0, 100.0)["G_over_Gmax"]
        assert hi > lo

    def test_damping_increases_with_strain(self):
        d_lo = modulus_reduction_ishibashi_zhang(0.001, 0.0, 100.0)["damping_pct"]
        d_hi = modulus_reduction_ishibashi_zhang(0.5, 0.0, 100.0)["damping_pct"]
        assert d_hi > d_lo
