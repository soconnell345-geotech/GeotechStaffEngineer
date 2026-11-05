"""Footing bearing-capacity convenience — tests (v5.4 E11).

Fast tests cover the closed-form bearing-capacity factors and input validation
(no FEM). The slow tests drive the real load-control collapse and pin the
Prandtl anchor q_ult = (2 + pi) c (Nc = 5.14) for T6, and the CST locking that
makes CST unusable for collapse loads (VALIDATION.md §3).
"""

import math

import pytest

from fem2d import analyze_footing_capacity, bearing_capacity_factors


class TestBearingCapacityFactors:
    def test_phi_zero_prandtl(self):
        Nc, Nq, Ng = bearing_capacity_factors(0.0)
        assert Nc == pytest.approx(2.0 + math.pi, rel=1e-12)   # 5.142
        assert Nq == pytest.approx(1.0, rel=1e-12)
        assert Ng == pytest.approx(0.0, abs=1e-12)

    def test_phi_30_vesic_table(self):
        Nc, Nq, Ng = bearing_capacity_factors(30.0)
        # standard published values (Vesic): Nc=30.14, Nq=18.40, Ngamma=22.40
        assert Nc == pytest.approx(30.14, abs=0.1)
        assert Nq == pytest.approx(18.40, abs=0.1)
        assert Ng == pytest.approx(22.40, abs=0.1)


class TestValidation:
    def test_bad_width(self):
        with pytest.raises(ValueError, match="B must be positive"):
            analyze_footing_capacity(B=0.0, c=100.0)

    def test_bad_element_type(self):
        with pytest.raises(ValueError, match="element_type"):
            analyze_footing_capacity(B=2.0, c=100.0, element_type="q8")

    def test_cohesionless_weightless_needs_qmax(self):
        with pytest.raises(ValueError, match="auto-size q_max"):
            analyze_footing_capacity(B=2.0, c=0.0, phi=30.0, gamma=0.0)


@pytest.mark.slow
class TestPrandtlCollapse:
    def test_t6_reproduces_prandtl_nc(self):
        """Weightless phi=0 strip footing: q_ult = (2+pi)c, Nc = 5.14. T6
        load-control collapse lands within ~2% (VALIDATION.md §3 band)."""
        r = analyze_footing_capacity(B=2.0, c=100.0, phi=0.0, gamma=0.0,
                                     nx=40, ny=20, element_type='t6',
                                     q_applied=200.0)
        print(f"\n  Footing T6: Nc={r.Nc_backfigured:.3f} (exact 5.14), "
              f"q_ult={r.q_ult_kPa:.0f} kPa")
        assert r.collapse_bracketed
        assert 4.8 <= r.Nc_backfigured <= 5.5      # house Prandtl band (~2%)
        # FOS against a 200 kPa working pressure
        assert r.bearing_FOS == pytest.approx(r.q_ult_kPa / 200.0, rel=1e-9)

    def test_cst_locks_and_does_not_collapse(self):
        """CST cannot represent the isochoric Prandtl mechanism: it locks and
        carries the full ramp without collapsing (VALIDATION.md §3)."""
        r = analyze_footing_capacity(B=2.0, c=100.0, phi=0.0, gamma=0.0,
                                     nx=20, ny=10, element_type='cst',
                                     n_load_steps=30)
        print(f"\n  Footing CST: Nc~{r.Nc_backfigured:.2f}, "
              f"bracketed={r.collapse_bracketed}")
        assert not r.collapse_bracketed            # never collapsed
        assert r.Nc_backfigured > 7.0              # locked well above 5.14


class TestSurchargeValidation:
    def test_qmax_must_exceed_surcharge(self):
        with pytest.raises(ValueError, match="must exceed the surcharge"):
            analyze_footing_capacity(B=2.0, c=50.0, surcharge=36.0,
                                     q_max=30.0, nx=10, ny=5)


@pytest.mark.slow
class TestSurchargeResists:
    """GM-11: the surcharge beside the footing is applied and resists through
    the Nq mechanism. Gross q_ult = c Nc + q Nq (+ 0.5 gamma B Ngamma), GEC-6
    (FHWA-IF-02-054) Eqs. 5-1, 5-2, 5-4 and 5-7, pp. 51-53; for phi = 0,
    UFC 3-220-20 Eq. 5-17 (q_ult = su Nc + sigma_zD). Made-up values."""

    def test_undrained_weightless_c_nc_plus_q(self):
        # 5.14 x 50 + 36 = 293.1 kPa (Nq = 1)
        r = analyze_footing_capacity(B=2.0, c=50.0, surcharge=36.0)
        assert r.collapse_bracketed
        assert r.q_ult_kPa == pytest.approx(293.1, rel=0.03)
        assert r.surcharge_kPa == 36.0

    def test_undrained_with_weight_unchanged_by_weight(self):
        # phi = 0: Ngamma = 0, so the soil's weight adds nothing (Eq. 5-17)
        r = analyze_footing_capacity(B=2.0, c=50.0, gamma=18.0,
                                     surcharge=36.0, nu=0.49)
        assert r.collapse_bracketed
        assert r.q_ult_kPa == pytest.approx(293.1, rel=0.03)

    def test_weightless_sand_q_nq(self):
        # Associated flow (psi = phi), the Prandtl-Reissner basis of Eq. 5-2:
        # Nq(20) = 6.40, q Nq = 20 x 6.40 = 128.0 kPa. Before GM-11: 0.
        r = analyze_footing_capacity(B=2.0, c=0.0, phi=20.0, psi=20.0,
                                     surcharge=20.0)
        assert r.collapse_bracketed
        assert r.q_ult_kPa == pytest.approx(128.0, rel=0.04)

    def test_no_surcharge_path_unchanged(self):
        r = analyze_footing_capacity(B=2.0, c=50.0)
        assert r.surcharge_kPa == 0.0
        assert r.q_ult_kPa == pytest.approx(411.3 * 28 / 45, rel=1e-3)
