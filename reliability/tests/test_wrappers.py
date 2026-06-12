"""Pre-canned wrapper tests + slope cross-validation (both paths agree)."""

import math

import pytest

from reliability.wrappers import (
    axial_pile_reliability, bearing_capacity_reliability, run_engine,
    slope_reliability,
)

FOOTING = {"width": 2.0, "depth": 1.5, "shape": "square"}
SAND = {"friction_angle": 32.0, "cohesion": 0.0, "unit_weight": 18.0}

PILE = {"name": "P1", "pile_type": "pipe_closed", "area": 0.05,
        "perimeter": 1.5, "tip_area": 0.18, "width": 0.46}
PILE_LAYERS = [
    {"thickness": 10.0, "soil_type": "cohesive", "unit_weight": 18.0,
     "cohesion": 50.0},
    {"thickness": 15.0, "soil_type": "cohesionless", "unit_weight": 19.0,
     "friction_angle": 33.0},
]


class TestBearingCapacityWrapper:
    def test_fosm_matches_deterministic_mean(self):
        from bearing_capacity import (
            BearingCapacityAnalysis, BearingSoilProfile, Footing, SoilLayer,
        )
        det = BearingCapacityAnalysis(
            footing=Footing(**FOOTING),
            soil=BearingSoilProfile(layer1=SoilLayer(
                cohesion=0.0, friction_angle=32.0, unit_weight=18.0)),
        ).compute()
        res = bearing_capacity_reliability(
            FOOTING, SAND, applied_pressure=300.0,
            variables={"friction_angle": {"cov": 0.08}})
        assert res.g_mean == pytest.approx(det.q_ultimate / 300.0, rel=1e-9)
        assert res.convention == "fos"
        assert res.beta_lognormal > 0

    def test_phi_dominates_variance(self):
        res = bearing_capacity_reliability(
            FOOTING, SAND, applied_pressure=300.0,
            variables={"friction_angle": {"cov": 0.08},
                       "unit_weight": {"cov": 0.05}})
        pct = res.variance_contributions_pct
        assert pct["friction_angle"] > 90.0
        assert pct["friction_angle"] + pct["unit_weight"] == \
            pytest.approx(100.0)

    def test_monte_carlo_engine(self):
        res = bearing_capacity_reliability(
            FOOTING, SAND, applied_pressure=300.0,
            variables={"friction_angle": {"cov": 0.08,
                                          "dist": "lognormal"}},
            engine="monte_carlo", n=2000, seed=42)
        assert res.n == 2000
        assert res.g_mean > 1.0

    def test_form_engine(self):
        res = bearing_capacity_reliability(
            FOOTING, SAND, applied_pressure=600.0,
            variables={"friction_angle": {"cov": 0.10,
                                          "dist": "lognormal"}},
            engine="form")
        assert res.converged
        # design point friction angle below the mean (resistance variable)
        assert res.design_point["friction_angle"] < 32.0
        assert res.alphas["friction_angle"] < 0

    def test_load_variable(self):
        res = bearing_capacity_reliability(
            FOOTING, SAND, applied_pressure=300.0,
            variables={"applied_pressure": {"cov": 0.15}})
        assert res.beta_lognormal is not None

    def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="unknown variable"):
            bearing_capacity_reliability(
                FOOTING, SAND, applied_pressure=300.0,
                variables={"su": {"cov": 0.2}})

    def test_bad_pressure_raises(self):
        with pytest.raises(ValueError, match="applied_pressure"):
            bearing_capacity_reliability(
                FOOTING, SAND, applied_pressure=0.0,
                variables={"friction_angle": {"cov": 0.08}})


class TestAxialPileWrapper:
    def test_fosm_matches_deterministic_mean(self):
        from axial_pile import (
            AxialPileAnalysis, AxialSoilLayer, AxialSoilProfile, PileSection,
        )
        det = AxialPileAnalysis(
            pile=PileSection(**PILE),
            soil=AxialSoilProfile(
                layers=[AxialSoilLayer(**d) for d in PILE_LAYERS]),
            pile_length=18.0).compute()
        res = axial_pile_reliability(
            PILE, PILE_LAYERS, pile_length=18.0, applied_load=800.0,
            variables={"cohesion": {"cov": 0.3, "dist": "lognormal"}})
        assert res.g_mean == pytest.approx(det.Q_ultimate / 800.0, rel=1e-9)

    def test_layer_scoped_variable(self):
        res = axial_pile_reliability(
            PILE, PILE_LAYERS, pile_length=18.0, applied_load=800.0,
            variables={"cohesion:1": {"cov": 0.3},
                       "friction_angle:2": {"cov": 0.08}})
        pct = res.variance_contributions_pct
        assert set(pct) == {"cohesion:1", "friction_angle:2"}
        assert sum(pct.values()) == pytest.approx(100.0)

    def test_pem_engine(self):
        res = axial_pile_reliability(
            PILE, PILE_LAYERS, pile_length=18.0, applied_load=800.0,
            variables={"cohesion": {"cov": 0.3},
                       "friction_angle": {"cov": 0.08}},
            engine="pem")
        assert res.n_points == 4
        assert res.g_mean > 1.0

    def test_unknown_variable_raises(self):
        with pytest.raises(ValueError, match="unknown variable"):
            axial_pile_reliability(
                PILE, PILE_LAYERS, pile_length=18.0, applied_load=800.0,
                variables={"cohesion:9": {"cov": 0.3}})


def _clay_slope(cu=40.0):
    from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
    layer = SlopeSoilLayer(
        name="clay", top_elevation=20.0, bottom_elevation=-15.0,
        gamma=18.0, cu=cu, analysis_mode="undrained")
    return SlopeGeometry(
        surface_points=[(0.0, 10.0), (20.0, 10.0), (40.0, 20.0),
                        (70.0, 20.0)],
        soil_layers=[layer])


CIRCLE = dict(xc=30.0, yc=32.0, radius=26.0)


class TestSlopeDelegate:
    def test_delegates_to_slope_stability(self):
        from slope_stability.probabilistic import FOSMResult
        res = slope_reliability(
            _clay_slope(), {"cu": {"mean": 40.0, "cov": 0.20}},
            engine="fosm", method="fellenius", n_slices=40, **CIRCLE)
        assert isinstance(res, FOSMResult)
        assert res.cov_f == pytest.approx(0.20, rel=0.01)

    def test_monte_carlo_delegate(self):
        res = slope_reliability(
            _clay_slope(), {"cu": {"mean": 40.0, "cov": 0.25,
                                   "dist": "lognormal"}},
            engine="monte_carlo", method="fellenius", n_slices=40,
            n=300, seed=11, **CIRCLE)
        assert res.n == 300

    def test_bad_engine_raises(self):
        with pytest.raises(ValueError, match="fosm.*monte_carlo"):
            slope_reliability(_clay_slope(), {"cu": {"cov": 0.2}},
                              engine="form")


class TestSlopeCrossValidation:
    """Same slope problem through BOTH paths must agree.

    Path A: slope_stability.probabilistic.fosm_fos (validated vs Duncan).
    Path B: reliability.fosm driving the slope FOS as a black-box g().
    Both use +/-1-sigma central differences, so beta must match closely.
    """

    def test_fosm_both_paths_agree(self):
        from reliability import fosm
        from slope_stability.search import _compute_fos
        from slope_stability.slip_surface import CircularSlipSurface

        mean_cu, cov_cu = 40.0, 0.20
        geom = _clay_slope(mean_cu)
        slip = CircularSlipSurface(CIRCLE["xc"], CIRCLE["yc"],
                                   CIRCLE["radius"])

        # Path A
        res_a = slope_reliability(
            geom, {"cu": {"mean": mean_cu, "cov": cov_cu}},
            engine="fosm", method="fellenius", n_slices=40, **CIRCLE)

        # Path B
        def g(values):
            gm = _clay_slope(values["cu"])
            return _compute_fos(gm, slip, "fellenius", 40, tol=1e-4)

        res_b = fosm(g, {"cu": {"mean": mean_cu, "cov": cov_cu}},
                     convention="fos")

        assert res_b.g_mean == pytest.approx(res_a.fos_mlv, rel=1e-6)
        assert res_b.g_std == pytest.approx(res_a.sigma_f, rel=1e-6)
        assert res_b.beta_lognormal == pytest.approx(
            res_a.beta_lognormal, rel=1e-6)
        assert res_b.pf_lognormal == pytest.approx(
            res_a.pf_lognormal, rel=1e-4)


class TestRunEngine:
    def test_bad_engine(self):
        with pytest.raises(ValueError, match="engine must be"):
            run_engine(lambda v: 1.0, {"x": {"mean": 1.0, "std": 0.1}},
                       engine="sorm")
