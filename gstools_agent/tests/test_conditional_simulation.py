"""
Tests for gstools_agent.analyze_conditional_simulation and its adapter.

The capability answers "what is the probability that a TOTAL over an area
exceeds a value" (field session 2026-10-15: rock excavation volume under a
footprint from a few borings). The checks are on the numbers, not only the
plumbing: limiting levels, data honoured, convergence to the kriged mean,
polygon areas, common random numbers, reproducibility, bounds, a hand
computation, and the joint covariance that decides the spread of a total.
"""

import json
import math

import numpy as np
import pytest

from gstools_agent.gstools_utils import has_gstools


@pytest.fixture(autouse=True)
def _outputs_in_tmp_path(tmp_path, monkeypatch):
    """INF-55: the adapter writes its CSVs and figures into the working
    folder; point it at this test's tmp_path so nothing lands in the repo."""
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))

pytestmark = pytest.mark.skipif(not has_gstools(),
                                reason="gstools not installed")

# Seven borings, top-of-rock elevations (m), around a 60 x 50 m footprint.
BX = [0, 60, 0, 60, 30, 75, -15]
BY = [0, 0, 50, 50, 25, 20, 30]
BZ = [96.0, 98.5, 95.2, 99.1, 97.4, 99.8, 94.5]
NAMES = ["B-1", "B-2", "B-3", "B-4", "B-5", "B-6", "B-7"]
FOOT = [[5, 5], [65, 5], [65, 55], [5, 55]]
MODEL = dict(model_type="Exponential", variance=2.0, len_scale=40.0)


def _run(regions, **kw):
    from gstools_agent import analyze_conditional_simulation
    args = dict(MODEL, cell_size=2.5, n_realizations=400, seed=11)
    args.update(kw)
    return analyze_conditional_simulation(BX, BY, BZ, regions, **args)


def _shoelace(poly):
    p = np.asarray(poly, dtype=float)
    return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                     - np.dot(p[:, 1], np.roll(p[:, 0], -1)))


# =====================================================================
# Limiting levels
# =====================================================================

class TestLevels:

    def test_level_far_above_every_value_gives_zero(self):
        r = _run([{"name": "b", "polygon": FOOT, "level": 200.0}])
        res = r.results[0]
        assert res["p_any"] == 0.0
        assert res["volume_above"]["mean"] == 0.0
        assert res["volume_above"]["P95"] == 0.0
        assert res["area_above"]["P95"] == 0.0
        assert res["kriged_surface"]["volume_above"] == 0.0
        assert res["exceedance"][0] == {"value": 0.0, "p_exceed": 0.0,
                                        "std_error": 0.0}

    def test_level_far_below_is_mean_field_times_area(self):
        lv = 0.0
        r = _run([{"name": "b", "polygon": FOOT, "level": lv}],
                 n_realizations=1000)
        res = r.results[0]
        m = r.cell_masks[0]
        a = res["cell_area_sum"]
        # Kriged surface: every cell above the level -> (mean - level) x A.
        assert res["kriged_surface"]["volume_above"] == pytest.approx(
            (r.kriged_cells[m].mean() - lv) * a, rel=1e-10)
        # Every realization's area is the whole region.
        assert res["area_above"]["P5"] == pytest.approx(a)
        assert res["p_any"] == 1.0
        # Mean over realizations = (mean field - level) x A within MC error.
        v = res["volume_above"]
        se = v["std"] / math.sqrt(r.n_kept)
        assert v["mean"] == pytest.approx(
            (r.kriged_cells[m].mean() - lv) * a, abs=4 * se)

    def test_quantity_below_mirrors_above_exactly(self):
        """Same seed, same grid = the same realizations, so per realization
        V_above(L) - V_below(L) = sum(z - L) A = V_above(far) +
        (far - L) A."""
        L, far = 97.0, 0.0
        up = _run([{"name": "b", "polygon": FOOT, "level": [L, far]}])
        dn = _run([{"name": "b", "polygon": FOOT, "level": L}],
                  quantity="below")
        a = up.results[0]["cell_area_sum"]
        lhs = up.totals("b", L) - dn.totals("b", L)
        rhs = up.totals("b", far) + (far - L) * a
        np.testing.assert_allclose(lhs, rhs, rtol=0, atol=1e-6)
        assert "volume_below" in dn.results[0]


# =====================================================================
# The data are honoured
# =====================================================================

class TestDataHonoured:

    def test_exact_without_nugget(self):
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}])
        assert r.data_exact
        assert r.data_check["max_abs_deviation"] < 1e-8

    def test_tiny_measurement_error_nearly_honours(self):
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 measurement_error=1e-6)
        assert r.nugget == pytest.approx(1e-6)        # raised to the error
        assert not r.data_exact
        assert r.data_check["max_abs_deviation"] < 0.02

    def test_measurement_error_lets_the_field_leave_the_data(self):
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 measurement_error=0.5)
        assert r.data_check["mean_abs_deviation"] > 0.1
        d = r.to_dict()
        assert d["field_passes_through_data"] is False
        assert d["measurement_error"] == 0.5

    @pytest.mark.parametrize("kw", [
        dict(kriging_type="universal"),
        dict(kriging_type="simple"),
        dict(anis=0.4, angle_deg=30.0),
        dict(model_type="Spherical"),
        dict(model_type="Gaussian", len_scale=25.0),
    ])
    def test_other_kriging_types_models_and_anisotropy(self, kw):
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 n_realizations=300, **kw)
        assert r.data_check["max_abs_deviation"] < 1e-6
        res = r.results[0]
        v = res["volume_above"]
        se = v["std"] / math.sqrt(r.n_kept)
        assert v["mean"] == pytest.approx(
            res["pointwise_expectation"]["volume_above"], abs=4 * se + 1.0)

    def test_per_point_measurement_error(self):
        me = [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0]     # B-5 uncertain
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 measurement_error=me)
        assert r.to_dict()["measurement_error"] == me
        assert r.nugget == pytest.approx(0.3)

    def test_measurement_error_above_the_nugget_is_refused(self):
        with pytest.raises(ValueError, match="cond_err <= nugget"):
            _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 measurement_error=0.5, nugget=0.1)


# =====================================================================
# Convergence to kriging
# =====================================================================

class TestConvergence:

    def test_realization_mean_tends_to_the_kriged_mean(self):
        n = 2000
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 n_realizations=n)
        bound = 5.0 * math.sqrt(MODEL["variance"] / n)
        assert np.abs(r.realization_mean - r.kriged_cells).max() < bound

    def test_mean_volume_above_is_at_least_the_kriged_surface(self):
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 n_realizations=1000)
        res = r.results[0]
        v = res["volume_above"]
        se = v["std"] / math.sqrt(r.n_kept)
        assert v["mean"] > res["kriged_surface"]["volume_above"] - 3 * se

    def test_convergence_table(self):
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 n_realizations=600)
        conv = r.results[0]["convergence"]
        assert [c["n"] for c in conv] == [100, 250, 500, 600]
        assert conv[-1]["mean"] == pytest.approx(
            r.results[0]["volume_above"]["mean"])


# =====================================================================
# Polygons and the lattice
# =====================================================================

L_SHAPE = [[3.3, 2.1], [41.7, 2.1], [41.7, 14.6], [17.2, 14.6],
           [17.2, 38.9], [3.3, 38.9]]
C_SHAPE = [[50.4, 1.3], [79.6, 1.3], [79.6, 10.2], [61.1, 10.2],
           [61.1, 30.7], [79.6, 30.7], [79.6, 39.8], [50.4, 39.8]]


class TestGeometry:

    def test_concave_polygons_cell_area_matches_shoelace(self):
        r = _run([{"name": "L", "polygon": L_SHAPE, "level": 97.0},
                  {"name": "C", "polygon": C_SHAPE, "level": 97.0}],
                 cell_size=0.75, n_realizations=20)
        for res, poly in zip(r.results, (L_SHAPE, C_SHAPE)):
            assert res["polygon_area"] == pytest.approx(_shoelace(poly))
            assert res["cell_area_sum"] == pytest.approx(
                res["polygon_area"], rel=0.03)
        # the L's notch is outside: a centre in it is not counted
        m = r.cell_masks[0]
        notch = (r.cell_x > 20) & (r.cell_x < 40) & (r.cell_y > 20)
        assert not np.any(m & notch)
        # the C's mouth is outside too
        mouth = (r.cell_x > 65) & (r.cell_y > 15) & (r.cell_y < 25)
        assert not np.any(r.cell_masks[1] & mouth)

    def test_fine_lattice_converges_to_the_shoelace_area(self):
        from gstools_agent.regions import (build_lattice, parse_regions,
                                           region_masks)
        regs = parse_regions([{"name": "L", "polygon": L_SHAPE, "level": 0},
                              {"name": "C", "polygon": C_SHAPE, "level": 0}])
        lat = build_lattice(regs, 0.1)
        for reg, m in zip(regs, region_masks(regs, lat["px"], lat["py"],
                                             0.1)):
            assert m.sum() * 0.01 == pytest.approx(
                _shoelace(reg["polygon"]), rel=0.005)

    def test_hand_areas(self):
        assert _shoelace(L_SHAPE) == pytest.approx(
            38.4 * 12.5 + 13.9 * 24.3)

    def test_rectangle_on_round_coordinates_is_exact(self):
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 cell_size=2.0, n_realizations=20)
        assert r.results[0]["cell_area_sum"] == pytest.approx(3000.0)
        assert r.results[0]["n_cells"] == 750

    def test_auto_cell_size_is_round_and_under_the_target(self):
        from gstools_agent.regions import AUTO_TARGET_CELLS
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 cell_size=None, n_realizations=20)
        assert r.cell_size == 1.0
        assert r.n_cells <= AUTO_TARGET_CELLS
        assert any("cell_size not given" in s for s in r.notes)


# =====================================================================
# Common random numbers
# =====================================================================

INNER = [[20, 15], [45, 15], [45, 40], [20, 40]]


class TestCommonRandomNumbers:

    def test_nested_region_never_has_more_volume(self):
        r = _run([{"name": "outer", "polygon": FOOT, "level": 97.0},
                  {"name": "inner", "polygon": INNER, "level": 97.0}])
        vo, vi = r.totals("outer", 97.0), r.totals("inner", 97.0)
        assert np.all(vi <= vo + 1e-9)
        assert np.any(vi < vo)
        ao, ai = (r.totals("outer", 97.0, "area"),
                  r.totals("inner", 97.0, "area"))
        assert np.all(ai <= ao + 1e-9)

    def test_lower_level_never_has_less_volume(self):
        r = _run([{"name": "b", "polygon": FOOT,
                   "level": [96.0, 97.0, 98.0]}])
        v96, v97, v98 = (r.totals("b", lv) for lv in (96.0, 97.0, 98.0))
        assert np.all(v96 >= v97 - 1e-9) and np.all(v97 >= v98 - 1e-9)
        assert "same realizations" in r.to_dict()["common_random_numbers"]


# =====================================================================
# Reproducibility and guards
# =====================================================================

class TestReproducibilityAndGuards:

    def test_same_seed_same_numbers(self):
        reg = [{"name": "b", "polygon": FOOT, "level": 97.0}]
        a = _run(reg, seed=5)
        b = _run(reg, seed=5)
        c = _run(reg, seed=6)
        np.testing.assert_array_equal(a.realization_table,
                                      b.realization_table)
        assert not np.allclose(a.realization_table, c.realization_table)

    def test_cell_cap_refuses_with_a_cell_count_and_a_size(self):
        big = [[0, 0], [400, 0], [400, 300], [0, 300]]
        with pytest.raises(ValueError) as exc:
            _run([{"name": "site", "polygon": big, "level": 97.0}],
                 cell_size=1.0)
        msg = str(exc.value)
        assert "120,000 cells of 1 m" in msg
        assert "cell_size >=" in msg

    def test_region_with_no_cell_is_refused(self):
        tiny = [[10, 10], [10.3, 10], [10.3, 10.3]]
        with pytest.raises(ValueError, match="no cell centre"):
            _run([{"name": "t", "polygon": tiny, "level": 97.0},
                  {"name": "b", "polygon": FOOT, "level": 97.0}])

    @pytest.mark.parametrize("kw, match", [
        (dict(quantity="sideways"), "quantity"),
        (dict(simulation_method="sgs"), "simulation_method"),
        (dict(n_realizations=50000), "n_realizations"),
        (dict(bounds_mode="clip"), "bounds_mode"),
        (dict(model_type="Gauss"), "model_type"),
    ])
    def test_bad_options(self, kw, match):
        with pytest.raises(ValueError, match=match):
            _run([{"name": "b", "polygon": FOOT, "level": 97.0}], **kw)

    def test_bad_region(self):
        with pytest.raises(ValueError, match="at least 3"):
            _run([{"name": "b", "polygon": [[0, 0], [1, 1]], "level": 1}])
        with pytest.raises(ValueError, match="unknown key"):
            _run([{"name": "b", "polygon": FOOT, "level": 1, "elev": 2}])

    def test_warnings_name_the_weak_points(self):
        r = _run([{"name": "b", "polygon": FOOT, "level": 120.0}])
        text = " ".join(r.warnings)
        assert "Only 7 observations" in text
        assert "outside the observed range" in text

    def test_assumed_variance_is_flagged(self):
        from gstools_agent import analyze_conditional_simulation
        r = analyze_conditional_simulation(
            BX, BY, BZ, [{"name": "b", "polygon": FOOT, "level": 97.0}],
            cell_size=2.5, n_realizations=50)
        text = " ".join(r.assumptions)
        assert "variance not given" in text
        assert "len_scale not given" in text


# =====================================================================
# Bounds (inequality data)
# =====================================================================

class TestBounds:

    def test_report_counts_violations(self):
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 bounds=[{"x": 20, "y": 40, "max": 50.0, "name": "low"},
                         {"x": 20, "y": 40, "max": 500.0, "name": "high"},
                         {"x": 40, "y": 10, "min": 0.0}])
        b = {x["name"]: x for x in r.bounds}
        assert b["low"]["violated_fraction"] == 1.0
        assert b["high"]["violated_fraction"] == 0.0
        assert b["bound_3"]["violated_fraction"] == 0.0
        assert r.n_kept == 400
        assert 94 < b["low"]["kriged_value"] < 100
        assert any("violated in 100%" in w for w in r.warnings)

    def test_reject_keeps_only_realizations_inside_the_bound(self):
        # A bound at a cell centre, with max = the region level: in the
        # kept realizations that cell is never above the level.
        cx, cy = 21.25, 31.25          # a centre at cell_size 2.5 from (5, 5)
        rep = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                   bounds=[{"x": cx, "y": cy, "max": 97.0}])
        k = int(np.argmin(np.hypot(rep.cell_x - cx, rep.cell_y - cy)))
        assert math.hypot(rep.cell_x[k] - cx, rep.cell_y[k] - cy) < 1e-9
        p_before = rep.exceed_counts[0][k] / rep.n_kept
        assert 0.05 < p_before < 0.95
        rej = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                   bounds=[{"x": cx, "y": cy, "max": 97.0}],
                   bounds_mode="reject")
        assert rej.n_kept == 400
        assert rej.exceed_counts[0][k] / rej.n_kept <= 0.01
        assert 0.05 < rej.acceptance_rate < 0.95
        d = rej.to_dict()
        assert "rejection sampling" in d["bounds_method"]
        assert d["results"][0]["volume_above"]["mean"] < \
            rep.to_dict()["results"][0]["volume_above"]["mean"]

    def test_reject_with_an_impossible_bound_hits_the_cap(self):
        with pytest.raises(ValueError, match="attempt cap"):
            _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 bounds=[{"x": 30, "y": 30, "max": -1000.0}],
                 bounds_mode="reject", n_realizations=30)

    def test_bad_bounds(self):
        with pytest.raises(ValueError, match="max and/or min"):
            _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 bounds=[{"x": 1, "y": 2}])


# =====================================================================
# Hand computations and the joint covariance
# =====================================================================

class TestHandComputation:
    """Simple kriging far from every observation: the field over a small
    region is unconditioned, N(m, sigma^2), and nearly constant across the
    region (correlation length 1e5 m). Then
        E[V] = A sigma [d Phi(d) + phi(d)], d = (m - L) / sigma,
        P(V > 0) = Phi(d)."""

    XO = [1e7, 1e7 + 10.0, 1e7]
    YO = [0.0, 0.0, 10.0]
    ZO = [99.0, 101.0, 100.0]           # sample mean m = 100
    REG = [{"name": "sq", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "level": 102.0}]            # L = m + sigma, sigma = 2

    def _expected(self):
        sigma, d, a = 2.0, -1.0, 100.0
        phi = math.exp(-0.5 * d * d) / math.sqrt(2 * math.pi)
        Phi = 0.5 * (1 + math.erf(d / math.sqrt(2)))
        return a * sigma * (d * Phi + phi), Phi

    @pytest.mark.parametrize("method, n, tol_p", [
        ("kriging_residual", 4000, 0.02),
        ("condsrf", 1000, 0.04),
    ])
    def test_unconditioned_region_matches_the_normal_formula(self, method,
                                                             n, tol_p):
        from gstools_agent import analyze_conditional_simulation
        r = analyze_conditional_simulation(
            self.XO, self.YO, self.ZO, self.REG, model_type="Exponential",
            variance=4.0, len_scale=1e5, kriging_type="simple",
            cell_size=1.0, n_realizations=n, seed=3,
            simulation_method=method)
        ev, pv = self._expected()
        res = r.results[0]
        v = res["volume_above"]
        se = v["std"] / math.sqrt(r.n_kept)
        assert v["mean"] == pytest.approx(ev, abs=4 * se + 0.02 * ev)
        assert res["p_any"] == pytest.approx(pv, abs=tol_p)
        assert any("more than len_scale" in w or "observation" in w
                   for w in r.warnings)

    def test_kriged_surface_of_constant_data(self):
        """Equal observations: the ordinary-kriged surface is that value
        everywhere, so the volume above 8 over a 10 x 20 m rectangle is
        (10 - 8) x 200 = 400 m3, in both methods' deterministic answer."""
        from gstools_agent import analyze_conditional_simulation, \
            analyze_kriging
        reg = [{"name": "r", "polygon": [[0, 0], [10, 0], [10, 20],
                                         [0, 20]], "level": 8.0}]
        kw = dict(variance=1.0, len_scale=10.0, cell_size=1.0)
        k = analyze_kriging([0, 30, 15], [0, 0, 30], [10.0, 10.0, 10.0],
                            fit_variogram=False, regions=reg, **kw)
        row = k.to_dict()["regions"][0]
        assert row["volume_above"] == pytest.approx(400.0, rel=1e-9)
        assert row["area_above"] == pytest.approx(200.0)
        c = analyze_conditional_simulation(
            [0, 30, 15], [0, 0, 30], [10.0, 10.0, 10.0], reg,
            n_realizations=20, **kw)
        assert c.results[0]["kriged_surface"]["volume_above"] == \
            pytest.approx(400.0, rel=1e-9)

    def test_kriging_without_regions_is_unchanged(self):
        from gstools_agent import analyze_kriging
        k = analyze_kriging(BX, BY, BZ, fit_variogram=False, variance=2.0,
                            len_scale=40.0, n_grid_x=5, n_grid_y=5)
        assert "regions" not in k.to_dict()


class TestJointCovariance:
    """Two one-cell regions either side of an observation (value 0, simple
    kriging about 0, exponential model, len_scale 10 m, the cells 5 m from
    it). With every cell above a level far below, V = z - level, so the
    correlation of the two totals is the correlation of the field.

    Exact conditional field (exponential = Markov, the datum screens):
    variance 1 - exp(-1) = 0.632, covariance 0. GSTools CondSRF keeps the
    unconditioned correlation exp(-1) = 0.37 across the datum -- why the
    default is kriging_residual."""

    XO = [0.0, 300.0, -300.0]
    YO = [0.0, 0.0, 0.0]
    ZO = [0.0, 0.0, 0.0]
    REG = [{"name": "left", "polygon": [[-5.5, -0.5], [-4.5, -0.5],
                                        [-4.5, 0.5], [-5.5, 0.5]],
            "level": -100.0},
           {"name": "right", "polygon": [[4.5, -0.5], [5.5, -0.5],
                                         [5.5, 0.5], [4.5, 0.5]],
            "level": -100.0}]

    def _corr(self, method, n):
        from gstools_agent import analyze_conditional_simulation
        r = analyze_conditional_simulation(
            self.XO, self.YO, self.ZO, self.REG, model_type="Exponential",
            variance=1.0, len_scale=10.0, kriging_type="simple",
            cell_size=1.0, n_realizations=n, seed=1,
            simulation_method=method)
        a, b = r.totals("left", -100.0), r.totals("right", -100.0)
        return float(np.corrcoef(a, b)[0, 1]), float(np.var(a))

    def test_kriging_residual_has_the_exact_conditional_covariance(self):
        corr, var = self._corr("kriging_residual", 3000)
        assert abs(corr) < 0.06
        assert var == pytest.approx(1 - math.exp(-1.0), rel=0.08)

    def test_condsrf_keeps_the_unconditioned_correlation(self):
        corr, var = self._corr("condsrf", 1500)
        assert corr > 0.2
        assert var == pytest.approx(1 - math.exp(-1.0), rel=0.12)


# =====================================================================
# Files and figures
# =====================================================================

class TestFilesAndFigures:

    def test_csv_text(self):
        r = _run([{"name": "b", "polygon": FOOT, "level": [96.0, 97.0]}],
                 n_realizations=30)
        real = r.realizations_csv().splitlines()
        assert real[0] == ("seed,volume_above:b@96,area_above:b@96,"
                           "volume_above:b@97,area_above:b@97")
        assert len(real) == 31
        prob = r.probability_csv().splitlines()
        assert prob[0] == "region,level,x,y,p_z_gt_level"
        assert len(prob) == 1 + 2 * r.n_cells
        kr = r.kriging_csv().splitlines()
        assert kr[0] == "x,y,kriged_mean,kriging_variance,kriging_std"
        assert len(kr) == 1 + r.lattice["nx"] * r.lattice["ny"]

    def test_figures_png_and_plotly_twins(self):
        from gstools_agent.conditional_figures import (
            exceedance_figure, probability_map_figure,
        )
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0},
                  {"name": "core", "polygon": INNER, "level": 97.0}],
                 names=NAMES, n_realizations=60,
                 bounds=[{"x": 20, "y": 40, "max": 96.0, "name": "B-8"}])
        png, pfig, n = exceedance_figure(r)
        assert png[:8] == b"\x89PNG\r\n\x1a\n" and n == 2
        assert pfig is not None and len(pfig.data) == 4
        png2, pmap, meta = probability_map_figure(r)
        assert png2[:8] == b"\x89PNG\r\n\x1a\n"
        lay = json.loads(pmap.to_json())["layout"]
        assert lay["yaxis"]["scaleanchor"] == "x"
        obs = [t for t in pmap.data if t.name == "Observations"][0]
        assert list(obs.text) == NAMES
        assert "value 96" in obs.hovertext[0] and "x 0 m" in obs.hovertext[0]
        assert meta["region"] == "b" and meta["level"] == 97.0


# =====================================================================
# Adapter
# =====================================================================

_ADAPTER_PARAMS = dict(
    x=BX, y=BY, values=BZ, names=NAMES,
    regions=[{"name": "bldg", "polygon": FOOT, "level": [96.0, 97.0]}],
    model_type="Exponential", variance=2.0, len_scale=40.0, cell_size=2.5,
    n_realizations=200, output_prefix="t")


def _walk_lists(obj, found):
    if isinstance(obj, dict):
        for v in obj.values():
            _walk_lists(v, found)
    elif isinstance(obj, list):
        found.append(len(obj))
        for v in obj:
            _walk_lists(v, found)
    return found


class TestAdapter:

    def test_unknown_parameter_refused(self):
        from funhouse_agent.dispatch import call_agent
        r = call_agent("gstools", "conditional_simulation",
                       dict(_ADAPTER_PARAMS, n_sims=10))
        assert "unknown parameter" in r["error"]

    def test_files_and_figures_land_in_the_working_folder(self, tmp_path,
                                                          monkeypatch):
        from funhouse_agent.dispatch import call_agent
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        r = call_agent("gstools", "conditional_simulation",
                       dict(_ADAPTER_PARAMS))
        assert "error" not in r, r.get("error")
        for name in ("t_realizations.csv", "t_probability.csv",
                     "t_kriging.csv", "t_exceedance.png",
                     "t_exceedance.plotly.json", "t_probability_map.png",
                     "t_probability_map.plotly.json"):
            assert (tmp_path / name).is_file(), name
        assert {f["figure"] for f in r["figures"]} == {"exceedance",
                                                        "probability_map"}
        assert all(f.get("plotly_json_path") for f in r["figures"])

    def test_json_has_no_grids(self, tmp_path, monkeypatch):
        from funhouse_agent.dispatch import call_agent
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        r = call_agent("gstools", "conditional_simulation",
                       dict(_ADAPTER_PARAMS, figures=False))
        assert "error" not in r, r.get("error")
        assert max(_walk_lists(r, [])) <= 20
        assert len(json.dumps(r)) < 20000
        for key in ("field", "grid_x", "krige_variance"):
            assert key not in r

    def test_condsrf_alias(self, tmp_path, monkeypatch):
        from funhouse_agent.dispatch import call_agent
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        r = call_agent("gstools", "condsrf",
                       dict(_ADAPTER_PARAMS, n_realizations=10,
                            figures=False, cell_size=5.0))
        assert r.get("simulation_method") == "condsrf", r.get("error")

    def test_kriging_large_grid_goes_to_a_file(self, tmp_path, monkeypatch):
        from funhouse_agent.dispatch import call_agent
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        base = dict(x=BX, y=BY, values=BZ, fit_variogram=False,
                    variance=2.0, len_scale=40.0)
        big = call_agent("gstools", "kriging",
                         dict(base, n_grid_x=60, n_grid_y=120,
                              output_prefix="kg"))
        assert "error" not in big, big.get("error")
        assert "field" not in big and "grid_x" not in big
        assert (tmp_path / "kg_grid.csv").is_file()
        lines = (tmp_path / "kg_grid.csv").read_text().splitlines()
        assert lines[0] == "x,y,value,kriging_variance"
        assert len(lines) == 1 + 60 * 120
        assert big["field_min"] <= big["field_mean"] <= big["field_max"]
        small = call_agent("gstools", "kriging",
                           dict(base, n_grid_x=10, n_grid_y=10))
        assert np.asarray(small["field"]).shape == (10, 10)

    def test_kriging_regions_through_the_adapter(self, tmp_path, monkeypatch):
        from funhouse_agent.dispatch import call_agent
        # its regions CSV goes to the working folder, not the repo
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        r = call_agent("gstools", "kriging", dict(
            x=BX, y=BY, values=BZ, fit_variogram=False, variance=2.0,
            len_scale=40.0, n_grid_x=3, n_grid_y=3,
            regions=[{"name": "bldg", "polygon": FOOT, "level": 97.0}],
            cell_size=2.5))
        assert "error" not in r, r.get("error")
        row = r["regions"][0]
        assert row["volume_above"] > 0 and row["cell_area_sum"] == 3000.0
        assert any("conditional_simulation" in n for n in r["notes"])

    def test_method_info_documents_every_parameter(self):
        from funhouse_agent.adapters.gstools_adapter import (
            _COND_VALID, _KRIGING_VALID, METHOD_INFO,
        )
        assert set(METHOD_INFO["conditional_simulation"]["parameters"]) \
            == set(_COND_VALID)
        assert set(METHOD_INFO["kriging"]["parameters"]) == \
            set(_KRIGING_VALID)
        for name, info in METHOD_INFO.items():
            for p, spec in info["parameters"].items():
                assert spec.get("description"), (name, p)


# =====================================================================
# Items from the lead's trace review of the field session (2026-10-15)
# =====================================================================

class _NaNRng:
    """Stands in for numpy's generator for one seed: every draw is NaN."""

    def standard_normal(self, n):
        return np.full(n, np.nan)


class TestNonFiniteGuard:
    """A realization with a NaN (the field session's own CondSRF script got
    one all-NaN field in a thousand and printed NaN statistics) is redrawn
    with a fresh seed, counted and reported; NaN never reaches a
    statistic."""

    def test_a_nan_realization_is_redrawn_and_reported(self, monkeypatch):
        orig = np.random.default_rng
        bad_seed = 11 + 5

        def fake(s=None):
            return _NaNRng() if s == bad_seed else orig(s)

        monkeypatch.setattr(np.random, "default_rng", fake)
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 n_realizations=50)
        assert r.redrawn == [{"replaced_seed": bad_seed,
                              "redrawn_seed": 11 + 1_000_000_000}]
        assert bad_seed not in r.realization_ids.tolist()
        assert 11 + 1_000_000_000 in r.realization_ids.tolist()
        d = r.to_dict()
        assert d["non_finite_redrawn"]["count"] == 1
        stats = d["results"][0]["volume_above"]
        assert all(math.isfinite(v) for v in stats.values())
        assert any("non-finite (NaN)" in w for w in r.warnings)

    def test_no_redraw_no_report(self):
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 n_realizations=20)
        assert r.redrawn == [] and "non_finite_redrawn" not in r.to_dict()

    def test_every_draw_nan_is_refused(self, monkeypatch):
        monkeypatch.setattr(np.random, "default_rng",
                            lambda s=None: _NaNRng())
        with pytest.raises(ValueError, match="non-finite"):
            _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 n_realizations=5)


class TestDefaultsFromThisCallsData:

    def test_variance_default_follows_the_observations_used(self):
        from gstools_agent import analyze_conditional_simulation
        reg = [{"name": "b", "polygon": FOOT, "level": 97.0}]
        a = analyze_conditional_simulation(BX, BY, BZ, reg, cell_size=2.5,
                                           n_realizations=20)
        b = analyze_conditional_simulation(BX + [40], BY + [45],
                                           BZ + [93.0], reg, cell_size=2.5,
                                           n_realizations=20)
        assert a.variance == pytest.approx(np.var(BZ))
        assert b.variance == pytest.approx(np.var(BZ + [93.0]))
        db = b.to_dict()["defaults_from_data"]
        assert db["variance"]["value"] == pytest.approx(np.var(BZ + [93.0]))
        assert "8 observations" in db["variance"]["basis"]
        assert db["observations"]["n"] == 8
        assert db["len_scale"]["value"] == pytest.approx(90.0 / 3.0)

    def test_given_values_are_not_echoed_as_defaults(self):
        r = _run([{"name": "b", "polygon": FOOT, "level": 97.0}],
                 n_realizations=10)
        d = r.to_dict()["defaults_from_data"]
        assert "variance" not in d and "len_scale" not in d

    def test_kriging_notes_echo_defaults_and_the_fit_override(self):
        from gstools_agent import analyze_kriging
        k = analyze_kriging(BX, BY, BZ, n_grid_x=4, n_grid_y=4,
                            fit_variogram=False)
        text = " ".join(k.notes)
        assert f"{np.var(BZ):.4g}" in text and "7 observations" in text
        k2 = analyze_kriging(BX, BY, BZ, n_grid_x=4, n_grid_y=4,
                             variance=50.0, len_scale=5.0)
        assert any("only starting guesses" in s for s in k2.notes)
        assert k2.variance != 50.0          # the fit overrode the guess


class TestPointwiseExpectation:

    def test_formula(self):
        from gstools_agent.regions import pointwise_gaussian
        p, e = pointwise_gaussian([10.0, 10.0, 12.0, 8.0],
                                  [2.0, 0.0, 0.0, 0.0], 10.0, "above")
        assert p[0] == pytest.approx(0.5)
        assert e[0] == pytest.approx(2.0 / math.sqrt(2 * math.pi))
        assert list(p[1:]) == [0.0, 1.0, 0.0]
        assert list(e[1:]) == [0.0, 2.0, 0.0]
        pb, eb = pointwise_gaussian([12.0], [1.0], 10.0, "below")
        d = -2.0
        Phi = 0.5 * (1 + math.erf(d / math.sqrt(2)))
        phi = math.exp(-0.5 * d * d) / math.sqrt(2 * math.pi)
        assert pb[0] == pytest.approx(Phi)
        assert eb[0] == pytest.approx(phi + d * Phi)

    def test_realization_mean_matches_the_pointwise_expectation(self):
        r = _run([{"name": "b", "polygon": FOOT, "level": [96.0, 98.0]}],
                 n_realizations=2000)
        for res in r.results:
            v, a = res["volume_above"], res["area_above"]
            pw = res["pointwise_expectation"]
            n = r.n_kept
            assert v["mean"] == pytest.approx(
                pw["volume_above"], abs=4 * v["std"] / math.sqrt(n))
            assert a["mean"] == pytest.approx(
                pw["area_above"], abs=4 * a["std"] / math.sqrt(n) + 1e-9)

    def test_kriging_regions_expectation_matches_the_simulation(self):
        from gstools_agent import analyze_kriging
        reg = [{"name": "b", "polygon": FOOT, "level": 97.0}]
        k = analyze_kriging(BX, BY, BZ, fit_variogram=False, n_grid_x=3,
                            n_grid_y=3, regions=reg, cell_size=2.5, **MODEL)
        row = k.to_dict()["regions"][0]
        s = _run(reg, n_realizations=2000)
        v = s.results[0]["volume_above"]
        assert row["volume_above"] == pytest.approx(
            s.results[0]["kriged_surface"]["volume_above"], rel=1e-9)
        assert row["expected_volume_above"] == pytest.approx(
            s.results[0]["pointwise_expectation"]["volume_above"], rel=1e-9)
        assert row["expected_volume_above"] == pytest.approx(
            v["mean"], abs=4 * v["std"] / math.sqrt(s.n_kept))
        assert row["expected_volume_above"] > row["volume_above"]
        assert 0 < row["mean_cell_probability_above"] < 1
        assert row["expected_area_above"] == pytest.approx(
            row["mean_cell_probability_above"] * row["cell_area_sum"])
        assert any("pointwise" in note for note in k.notes)

    def test_kriging_regions_csv_through_the_adapter(self, tmp_path,
                                                     monkeypatch):
        from funhouse_agent.dispatch import call_agent
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        r = call_agent("gstools", "kriging", dict(
            x=BX, y=BY, values=BZ, fit_variogram=False, variance=2.0,
            len_scale=40.0, n_grid_x=3, n_grid_y=3, output_prefix="kr",
            regions=[{"name": "bldg", "polygon": FOOT, "level": 97.0}],
            cell_size=2.5))
        assert "error" not in r, r.get("error")
        lines = (tmp_path / "kr_regions.csv").read_text().splitlines()
        assert lines[0] == ("region,level,x,y,kriged_mean,kriging_std,"
                            "p_z_gt_level")
        assert len(lines) == 1 + 480

    def test_kriging_returns_block_names_every_key(self, tmp_path,
                                                   monkeypatch):
        from funhouse_agent.adapters.gstools_adapter import METHOD_INFO
        from funhouse_agent.dispatch import call_agent
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        # call_agent adds ``defaults_used`` to every result (GM-115)
        documented = set(METHOD_INFO["kriging"]["returns"]) | {"defaults_used"}
        base = dict(x=BX, y=BY, values=BZ, variance=50.0, len_scale=5.0)
        for extra in (dict(n_grid_x=5, n_grid_y=5),
                      dict(n_grid_x=60, n_grid_y=60,
                           regions=[{"name": "b", "polygon": FOOT,
                                     "level": 97.0}], cell_size=2.5)):
            r = call_agent("gstools", "kriging", dict(base, **extra))
            assert "error" not in r, r.get("error")
            assert set(r) <= documented, set(r) - documented
