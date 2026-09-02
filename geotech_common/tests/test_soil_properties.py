"""Tests for SPT field corrections and Bolton stress-dilatancy.

Overburden-correction pins cross-checked against an independent
implementation (groundhog 0.15.0, numerical oracle only); energy
corrections anchored to Youd et al. (2001) Table 2 values directly.
"""

import pytest

from geotech_common.soil_properties import (
    spt_energy_correction,
    spt_overburden_correction,
    spt_n1_60,
    stress_dilatancy_bolton,
)


class TestEnergyCorrection:
    def test_standard_conditions_identity(self):
        r = spt_energy_correction(20.0)
        assert r["N60"] == pytest.approx(20.0)
        assert (r["CE"], r["CB"], r["CR"], r["CS"]) == (1.0, 1.0, 1.0, 1.0)

    def test_youd_table2_factors(self):
        # Safety hammer ER=45 -> CE=0.75; short rods CR=0.75; 150mm CB=1.05;
        # linerless sampler CS=1.2 (Youd et al. 2001 Table 2)
        r = spt_energy_correction(20.0, energy_ratio_pct=45.0,
                                  rod_length_m=2.5,
                                  borehole_diameter_mm=150.0,
                                  sampler_correction=1.2)
        assert r["CE"] == pytest.approx(0.75)
        assert r["CR"] == pytest.approx(0.75)
        assert r["CB"] == pytest.approx(1.05)
        assert r["CS"] == pytest.approx(1.2)
        assert r["N60"] == pytest.approx(20.0 * 0.75 * 0.75 * 1.05 * 1.2)

    def test_rod_length_bands(self):
        assert spt_energy_correction(10, rod_length_m=3.5)["CR"] == 0.80
        assert spt_energy_correction(10, rod_length_m=5.0)["CR"] == 0.85
        assert spt_energy_correction(10, rod_length_m=8.0)["CR"] == 0.95
        assert spt_energy_correction(10, rod_length_m=15.0)["CR"] == 1.00

    def test_automatic_hammer(self):
        # ER=90 automatic-trip -> CE=1.5
        assert spt_energy_correction(20, energy_ratio_pct=90.0)["CE"] == \
            pytest.approx(1.5)

    def test_bad_inputs(self):
        with pytest.raises(ValueError):
            spt_energy_correction(-1.0)
        with pytest.raises(ValueError):
            spt_energy_correction(10, energy_ratio_pct=10.0)
        with pytest.raises(ValueError):
            spt_energy_correction(10, borehole_diameter_mm=300.0)
        with pytest.raises(ValueError):
            spt_energy_correction(10, sampler_correction=1.5)


class TestOverburdenCorrection:
    def test_liao_whitman_pins(self):
        # groundhog 0.15.0 oracle: N=20 @ 50 kPa -> CN=1.41421, N1=28.284
        r = spt_overburden_correction(20.0, 50.0)
        assert r["CN"] == pytest.approx(1.414214, rel=1e-5)
        assert r["N1"] == pytest.approx(28.28427, rel=1e-5)
        # exact identity at the 100 kPa reference stress
        assert spt_overburden_correction(20.0, 100.0)["CN"] == pytest.approx(1.0)
        # N=30 @ 200 kPa -> CN=0.70711, N1=21.213
        r = spt_overburden_correction(30.0, 200.0)
        assert r["N1"] == pytest.approx(21.21320, rel=1e-5)

    def test_liao_whitman_youd_cap(self):
        # sigma_v'=25 kPa gives raw CN=2.0 -> capped at 1.7 (Youd 2001)
        assert spt_overburden_correction(10.0, 25.0)["CN"] == pytest.approx(1.7)

    def test_iso_pins(self):
        # groundhog oracle: sqrt(98/50)=1.4; sqrt(98/100)=0.98995
        assert spt_overburden_correction(20.0, 50.0, method="iso")["CN"] == \
            pytest.approx(1.4, rel=1e-5)
        assert spt_overburden_correction(20.0, 100.0, method="iso")["CN"] == \
            pytest.approx(0.989949, rel=1e-5)
        assert spt_overburden_correction(30.0, 200.0, method="iso")["N1"] == \
            pytest.approx(21.0, rel=1e-5)

    def test_iso_cap(self):
        # sigma_v'=10 kPa: raw sqrt(9.8)=3.13 -> capped at 2.0
        assert spt_overburden_correction(10.0, 10.0, method="iso")["CN"] == \
            pytest.approx(2.0)

    def test_bad_method(self):
        with pytest.raises(ValueError):
            spt_overburden_correction(10.0, 100.0, method="bogus")


class TestN160Chain:
    def test_chain_composition(self):
        r = spt_n1_60(18.0, sigma_vo_eff_kPa=80.0, energy_ratio_pct=80.0,
                      rod_length_m=6.5)
        # CE=80/60, CR=0.95, CN=sqrt(100/80)
        n60 = 18.0 * (80.0 / 60.0) * 0.95
        assert r["N60"] == pytest.approx(n60)
        assert r["N1_60"] == pytest.approx(n60 * (100.0 / 80.0) ** 0.5)

    def test_reports_all_factors(self):
        r = spt_n1_60(20.0, 100.0)
        assert set(r) == {"CE", "CB", "CR", "CS", "N60", "CN", "N1_60"}


class TestBoltonDilatancy:
    def test_triaxial_pin(self):
        # groundhog 0.15.0 oracle: Dr=0.75, p'=100 kPa
        r = stress_dilatancy_bolton(0.75, 100.0, stress_condition="triaxial")
        assert r["IR"] == pytest.approx(3.04612, rel=1e-4)
        assert r["phi_excess_deg"] == pytest.approx(9.13837, rel=1e-4)
        assert r["max_dilation_rate"] == pytest.approx(0.913837, rel=1e-4)

    def test_plane_strain_pin(self):
        r = stress_dilatancy_bolton(0.75, 100.0,
                                    stress_condition="plane_strain")
        assert r["phi_excess_deg"] == pytest.approx(15.23061, rel=1e-4)
        assert r["dilation_angle_deg"] == pytest.approx(19.03826, rel=1e-4)

    def test_loose_sand_no_dilation(self):
        # loose sand at high stress: IR clipped at 0
        r = stress_dilatancy_bolton(0.2, 1000.0)
        assert r["IR"] == 0.0
        assert r["phi_excess_deg"] == 0.0

    def test_fraction_input_enforced(self):
        with pytest.raises(ValueError):
            stress_dilatancy_bolton(75.0, 100.0)   # percent, not fraction
