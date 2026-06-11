"""
P5 tests: probabilistic FOS (FOSM + Monte Carlo), benchmark B4.

Published anchor (Duncan 2000): F_MLV = 1.5, COV_F = 0.17 ->
beta_LN = 2.32, pf ~ 1%.

Closed-form cross-check: for an undrained (phi=0) slope, FOS is exactly
proportional to cu, so COV_F == COV_cu and the MC distribution of FOS is
exactly the (log)normal of cu scaled — FOSM, MC and hand values must all
agree.
"""

import math

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.probabilistic import (
    fosm_fos, monte_carlo_fos, lognormal_beta, normal_beta, _phi_cdf,
)


def _clay_slope(cu=40.0):
    layer = SlopeSoilLayer(
        name="clay", top_elevation=20.0, bottom_elevation=-15.0,
        gamma=18.0, cu=cu, analysis_mode="undrained",
    )
    geom = SlopeGeometry(
        surface_points=[(0.0, 10.0), (20.0, 10.0), (40.0, 20.0),
                        (70.0, 20.0)],
        soil_layers=[layer],
    )
    return geom


CIRCLE = dict(xc=30.0, yc=32.0, radius=26.0)


class TestDuncanAnchor:
    """B4 — Duncan (2000) published beta/pf anchor."""

    def test_lognormal_beta_anchor(self):
        beta = lognormal_beta(1.5, 0.17)
        assert beta == pytest.approx(2.32, abs=0.02)
        assert _phi_cdf(-beta) == pytest.approx(0.01, abs=0.003)

    def test_normal_beta(self):
        # Duncan 2000 companion value: beta_normal = (1.5-1)/(0.17*1.5)
        assert normal_beta(1.5, 0.255) == pytest.approx(1.96, abs=0.01)


class TestFOSMClosedForm:

    def test_undrained_cov_propagates_exactly(self):
        """phi=0: F ~ cu linearly, so COV_F == COV_cu and
        F_MLV = F(mean cu)."""
        geom = _clay_slope(cu=40.0)
        res = fosm_fos(
            geom, variables={"cu": {"mean": 40.0, "cov": 0.20}},
            method="fellenius", n_slices=40, **CIRCLE)
        assert res.cov_f == pytest.approx(0.20, rel=0.01)
        # beta_LN consistent with the closed form
        assert res.beta_lognormal == pytest.approx(
            lognormal_beta(res.fos_mlv, 0.20), rel=0.01)
        # single variable -> 100% variance contribution
        assert res.variable_variance_pct["cu"] == pytest.approx(100.0)

    def test_two_variables_drained(self):
        layer = SlopeSoilLayer(
            name="soil", top_elevation=20.0, bottom_elevation=-15.0,
            gamma=19.0, phi=25.0, c_prime=8.0,
        )
        geom = SlopeGeometry(
            surface_points=[(0.0, 10.0), (20.0, 10.0), (40.0, 20.0),
                            (70.0, 20.0)],
            soil_layers=[layer],
        )
        res = fosm_fos(
            geom,
            variables={
                "phi": {"mean": 25.0, "cov": 0.10},
                "c_prime": {"mean": 8.0, "cov": 0.30},
            },
            method="bishop", n_slices=40, **CIRCLE)
        assert res.fos_mlv > 1.0
        assert 0.0 < res.cov_f < 0.5
        pct = res.variable_variance_pct
        assert pct["phi"] + pct["c_prime"] == pytest.approx(100.0, abs=0.1)
        assert res.pf_lognormal < 0.5

    def test_bad_variable_errors(self):
        geom = _clay_slope()
        with pytest.raises(ValueError, match="Unknown parameter"):
            fosm_fos(geom, {"su": {"mean": 1, "cov": 0.1}}, **CIRCLE)
        with pytest.raises(ValueError, match="Unknown layer"):
            fosm_fos(geom, {"cu:nope": {"mean": 1, "cov": 0.1}}, **CIRCLE)
        with pytest.raises(ValueError, match="cov"):
            fosm_fos(geom, {"cu": {"mean": 40}}, **CIRCLE)


class TestMonteCarlo:

    def test_mc_matches_fosm_undrained_lognormal(self):
        """Lognormal cu with phi=0: FOS is exactly lognormal, so MC pf
        must agree with the FOSM lognormal pf within sampling error."""
        geom = _clay_slope(cu=40.0)
        variables = {"cu": {"mean": 40.0, "cov": 0.25, "dist": "lognormal"}}
        fosm = fosm_fos(geom, variables, method="fellenius", n_slices=40,
                        **CIRCLE)
        mc = monte_carlo_fos(geom, variables, method="fellenius",
                             n=4000, seed=42, n_slices=40, **CIRCLE)
        assert mc.n == 4000
        # distribution stats
        assert mc.fos_mean == pytest.approx(fosm.fos_mlv, rel=0.03)
        assert mc.fos_cov == pytest.approx(0.25, rel=0.10)
        # pf agreement (within MC error)
        assert mc.pf == pytest.approx(fosm.pf_lognormal, abs=0.015)
        assert mc.beta_lognormal == pytest.approx(fosm.beta_lognormal,
                                                  rel=0.08)

    def test_mc_reproducible_with_seed(self):
        geom = _clay_slope()
        variables = {"cu": {"mean": 40.0, "cov": 0.2}}
        a = monte_carlo_fos(geom, variables, n=200, seed=7, n_slices=30,
                            **CIRCLE)
        b = monte_carlo_fos(geom, variables, n=200, seed=7, n_slices=30,
                            **CIRCLE)
        assert a.fos_mean == b.fos_mean
        assert a.pf == b.pf

    def test_mc_histogram_and_dict(self):
        geom = _clay_slope()
        variables = {"cu": {"mean": 40.0, "cov": 0.2}}
        mc = monte_carlo_fos(geom, variables, n=300, seed=1, n_slices=30,
                             n_bins=15, **CIRCLE)
        assert len(mc.histogram_counts) == 15
        assert len(mc.histogram_bins) == 16
        assert sum(mc.histogram_counts) == mc.n
        d = mc.to_dict()
        assert d["n_realizations"] == mc.n
        assert "pf" in d and "beta_lognormal" in d
        assert "FOS distribution" in mc.summary() or "MONTE CARLO" in mc.summary()

    def test_mc_research_surface_small(self):
        """Re-search per realization (expensive path) works."""
        geom = _clay_slope()
        variables = {"cu": {"mean": 40.0, "cov": 0.15}}
        mc = monte_carlo_fos(
            geom, variables, method="bishop", n=5, seed=3, n_slices=20,
            research_surface=True,
            search_kwargs={"nx": 4, "ny": 4})
        assert mc.n == 5
        assert mc.research_surface
        assert mc.fos_mean > 0

    def test_layer_scoped_variable(self):
        upper = SlopeSoilLayer(
            name="upper", top_elevation=20.0, bottom_elevation=5.0,
            gamma=19.0, phi=30.0, c_prime=5.0,
        )
        lower = SlopeSoilLayer(
            name="lower", top_elevation=5.0, bottom_elevation=-15.0,
            gamma=18.0, cu=50.0, analysis_mode="undrained",
        )
        geom = SlopeGeometry(
            surface_points=[(0.0, 10.0), (20.0, 10.0), (40.0, 20.0),
                            (70.0, 20.0)],
            soil_layers=[upper, lower],
        )
        res = fosm_fos(
            geom, variables={"cu:lower": {"cov": 0.3}},
            method="bishop", n_slices=40, **CIRCLE)
        assert res.fos_mlv > 0
        assert "cu:lower" in res.variable_deltas
