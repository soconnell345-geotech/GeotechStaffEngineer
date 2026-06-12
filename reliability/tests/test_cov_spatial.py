"""Knowledge base (cov_database) + spatial averaging (Vanmarcke) tests."""

import math

import pytest

from reliability.cov_database import (
    ALIASES, CATEGORIES, COV_DATABASE, cov_guidance, list_properties,
)
from reliability.spatial import (
    ANISOTROPY_TYPICAL, SCALE_OF_FLUCTUATION, averaged_cov, averaged_std,
    scale_of_fluctuation_guidance, variance_reduction,
)


class TestDatabaseIntegrity:
    def test_every_entry_has_source(self):
        for e in COV_DATABASE:
            assert e.source, f"{e.property}/{e.label} missing source"

    def test_ranges_sane(self):
        for e in COV_DATABASE:
            assert 0 <= e.cov_min_pct <= e.cov_max_pct, e.label
            if e.cov_mean_pct is not None:
                assert e.cov_min_pct <= e.cov_mean_pct <= e.cov_max_pct, \
                    e.label

    def test_categories_valid(self):
        for e in COV_DATABASE:
            assert e.category in CATEGORIES

    def test_aliases_resolve(self):
        canon = {e.property for e in COV_DATABASE}
        for alias, key in ALIASES.items():
            assert key in canon, f"alias '{alias}' -> unknown '{key}'"


class TestPublishedValues:
    """Spot checks against the published tables (provenance tests)."""

    def test_duncan_unit_weight(self):
        rows = cov_guidance("gamma", category="inherent")
        assert len(rows) == 1
        assert (rows[0].cov_min_pct, rows[0].cov_max_pct) == (3, 7)
        assert "Duncan (2000)" in rows[0].source

    def test_duncan_su(self):
        rows = cov_guidance("su", category="inherent")
        assert (rows[0].cov_min_pct, rows[0].cov_max_pct) == (13, 40)

    def test_duncan_phi(self):
        rows = cov_guidance("phi", category="inherent")
        assert (rows[0].cov_min_pct, rows[0].cov_max_pct) == (2, 13)

    def test_tc304_phi_sand(self):
        rows = cov_guidance("phi", soil_type="sand",
                            category="site_specific")
        assert rows[0].cov_mean_pct == pytest.approx(7.9)
        assert "TC304" in rows[0].source

    def test_tc304_su_clay_mean(self):
        rows = cov_guidance("su", soil_type="clay",
                            category="site_specific")
        assert rows[0].cov_mean_pct == pytest.approx(28.2)
        assert (rows[0].cov_min_pct, rows[0].cov_max_pct) == (6, 56)

    def test_transformation_su_from_cpt(self):
        rows = cov_guidance("su", category="transformation", test="CPT")
        assert (rows[0].cov_min_pct, rows[0].cov_max_pct) == (29, 35)
        assert "UFC 3-220-20" in rows[0].source

    def test_spt_total_test(self):
        rows = cov_guidance("spt", category="total_test")
        assert (rows[0].cov_min_pct, rows[0].cov_max_pct) == (15, 45)


class TestLookups:
    def test_alias_lookup(self):
        assert cov_guidance("friction_angle") == cov_guidance("phi")

    def test_test_filter(self):
        rows = cov_guidance("su", test="VST")
        assert all("VST" in r.test for r in rows)

    def test_unknown_property_raises(self):
        with pytest.raises(ValueError, match="No COV guidance"):
            cov_guidance("modulus_of_silliness")

    def test_bad_category_raises(self):
        with pytest.raises(ValueError, match="category"):
            cov_guidance("su", category="vibes")

    def test_list_properties(self):
        props = list_properties()
        for p in ("su", "phi", "gamma", "N", "qc"):
            assert p in props

    def test_to_dict(self):
        d = cov_guidance("su")[0].to_dict()
        assert {"property", "cov_min_pct", "cov_max_pct", "source"} <= set(d)


class TestVarianceReduction:
    def test_point_limit(self):
        assert variance_reduction(0.0, 2.0) == 1.0
        assert variance_reduction(1e-9, 2.0) == pytest.approx(1.0)

    def test_long_average_limit(self):
        # Gamma^2 -> delta/L for L >> delta
        d, L = 2.0, 2000.0
        assert variance_reduction(L, d) == pytest.approx(d / L, rel=0.01)

    def test_exact_exponential_value(self):
        # x = 2L/delta = 2: Gamma^2 = 2/4*(2-1+e^-2) = 0.5*(1+e^-2)
        val = variance_reduction(2.0, 2.0)
        assert val == pytest.approx(0.5 * (1.0 + math.exp(-2.0)))

    def test_monotone_decreasing_in_L(self):
        vals = [variance_reduction(L, 2.0) for L in (0.5, 1, 2, 5, 10, 50)]
        assert all(a > b for a, b in zip(vals, vals[1:]))

    def test_simple_model(self):
        assert variance_reduction(1.0, 2.0, model="simple") == 1.0
        assert variance_reduction(8.0, 2.0, model="simple") == \
            pytest.approx(0.25)

    def test_bounds(self):
        for L in (0.1, 1.0, 10.0, 100.0):
            g2 = variance_reduction(L, 2.0)
            assert 0.0 < g2 <= 1.0

    def test_errors(self):
        with pytest.raises(ValueError):
            variance_reduction(1.0, 0.0)
        with pytest.raises(ValueError):
            variance_reduction(-1.0, 2.0)
        with pytest.raises(ValueError):
            variance_reduction(1.0, 1.0, model="gauss")


class TestAveragedDispersion:
    def test_averaged_std(self):
        s = averaged_std(10.0, 20.0, 2.0)
        assert s == pytest.approx(
            10.0 * math.sqrt(variance_reduction(20.0, 2.0)))
        assert s < 10.0

    def test_averaged_cov_reduces_inherent_only(self):
        # systematic components survive averaging untouched
        c = averaged_cov(0.30, L=1000.0, delta=1.0,
                         cov_measurement=0.10, cov_transformation=0.05)
        # Gamma^2 ~ 0.001 -> inherent contributes ~nothing
        floor = math.sqrt(0.10 ** 2 + 0.05 ** 2)
        assert c == pytest.approx(floor, rel=0.01)
        assert c > floor  # but strictly above

    def test_no_averaging_recovers_combined(self):
        from reliability.stats import combined_cov
        assert averaged_cov(0.3, L=0.0, delta=1.0, cov_measurement=0.1) == \
            pytest.approx(combined_cov(0.3, 0.1))


class TestFluctuationGuidance:
    def test_all_rows_have_source(self):
        for e in SCALE_OF_FLUCTUATION:
            assert "Cami" in e.source

    def test_clay_lookup(self):
        rows = scale_of_fluctuation_guidance("clay")
        assert any(e.soil_type == "clay" for e in rows)
        clay = [e for e in rows if e.soil_type == "clay"][0]
        assert clay.delta_v_avg == pytest.approx(2.47)
        assert clay.delta_h_avg == pytest.approx(24.43)

    def test_sand_values(self):
        sand = [e for e in scale_of_fluctuation_guidance("sand")
                if e.soil_type == "sand"][0]
        assert (sand.delta_v_min, sand.delta_v_max) == (0.1, 4.0)
        assert sand.delta_v_avg == pytest.approx(1.14)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="scale-of-fluctuation"):
            scale_of_fluctuation_guidance("moon dust")

    def test_anisotropy_constants(self):
        assert ANISOTROPY_TYPICAL == (10.0, 20.0)

    def test_none_returns_all(self):
        assert len(scale_of_fluctuation_guidance()) == \
            len(SCALE_OF_FLUCTUATION)


class TestEndToEndUsage:
    def test_guidance_into_variable_into_engine(self):
        """The intended workflow: KB -> averaged COV -> RV -> engine."""
        from reliability import RandomVariable, fosm

        su_cov_pct = cov_guidance("su", soil_type="clay",
                                  category="site_specific")[0].cov_mean_pct
        delta_v = [e for e in scale_of_fluctuation_guidance("clay")
                   if e.soil_type == "clay"][0].delta_v_avg
        cov_avg = averaged_cov(su_cov_pct / 100.0, L=10.0, delta=delta_v,
                               cov_measurement=0.10)
        su = RandomVariable("su", mean=50.0, cov=cov_avg, dist="lognormal")
        res = fosm(lambda v: v["su"] / 25.0, [su])
        assert res.g_mean == pytest.approx(2.0)
        assert res.beta_lognormal > 0
