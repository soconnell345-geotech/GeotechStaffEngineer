"""Tests for Brinkgreve et al. (2010) HS parameter estimation.

Pin values cross-checked against an independent implementation
(groundhog 0.15.0, used as a numerical oracle only) and against the
published closed-form expressions by hand.
"""

import pytest

from fem2d.hs_correlations import estimate_hs_parameters_sand


class TestHsParametersSand:
    def test_dr60_pins(self):
        p = estimate_hs_parameters_sand(60.0)
        assert p["gamma_unsat_kN_m3"] == pytest.approx(17.4)
        assert p["gamma_sat_kN_m3"] == pytest.approx(19.96)
        assert p["E50_ref_kPa"] == pytest.approx(36000.0)
        assert p["Eoed_ref_kPa"] == pytest.approx(36000.0)
        assert p["Eur_ref_kPa"] == pytest.approx(108000.0)
        assert p["G0_ref_kPa"] == pytest.approx(100800.0)
        assert p["m"] == pytest.approx(0.5125)
        assert p["gamma_07"] == pytest.approx(1.4e-4)
        assert p["phi_eff_deg"] == pytest.approx(35.5)
        assert p["psi_deg"] == pytest.approx(5.5)
        assert p["Rf"] == pytest.approx(0.925)

    def test_dense_sand_dilates(self):
        p = estimate_hs_parameters_sand(100.0)
        assert p["psi_deg"] == pytest.approx(10.5)
        assert p["phi_eff_deg"] == pytest.approx(40.5)

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            estimate_hs_parameters_sand(5.0)
        with pytest.raises(ValueError):
            estimate_hs_parameters_sand(105.0)

    def test_stiffness_ratio_convention(self):
        # Eur = 3 * E50 at every density per the published set
        for dr in (20.0, 50.0, 90.0):
            p = estimate_hs_parameters_sand(dr)
            assert p["Eur_ref_kPa"] / p["E50_ref_kPa"] == pytest.approx(3.0)
