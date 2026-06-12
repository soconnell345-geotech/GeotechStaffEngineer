"""Tests for slope_stability.plotting (Phase 1, calc-viz)."""

import pytest

from slope_stability import (
    SlopeGeometry, SlopeSoilLayer, SoilNail, Geosynthetic, Anchor,
    analyze_slope, search_critical_surface,
)

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from slope_stability.plotting import (  # noqa: E402
    plot_cross_section, plot_trial_surface_map, plot_slice_forces,
    plot_interslice_forces, plot_mc_histogram, plot_fosm_tornado,
)


@pytest.fixture(scope="module")
def geom():
    return SlopeGeometry(
        surface_points=[(0, 20), (10, 20), (30, 10), (50, 10)],
        soil_layers=[
            SlopeSoilLayer("Fill", top_elevation=20, bottom_elevation=12,
                           gamma=19, phi=28, c_prime=5),
            SlopeSoilLayer("Clay", top_elevation=12, bottom_elevation=0,
                           gamma=18, phi=22, c_prime=12),
        ],
        gwt_points=[(0, 16), (30, 9), (50, 9)],
    )


@pytest.fixture(scope="module")
def geom_reinforced():
    return SlopeGeometry(
        surface_points=[(0, 20), (10, 20), (30, 10), (50, 10)],
        soil_layers=[
            SlopeSoilLayer("Clay", top_elevation=20, bottom_elevation=0,
                           gamma=18, phi=22, c_prime=12),
        ],
        nails=[SoilNail(x_head=18, z_head=16, length=8)],
        anchors=[Anchor(x_head=24, z_head=13, length=12, T_allow=80)],
        geosynthetics=[Geosynthetic(elevation=11, T_allow=40)],
    )


@pytest.fixture(scope="module")
def spencer_result(geom):
    return analyze_slope(geom, xc=22, yc=28, radius=20, method="spencer",
                         include_slice_data=True)


class TestCrossSection:
    def test_returns_figure(self, spencer_result, geom):
        fig = plot_cross_section(spencer_result, geom)
        assert fig is not None
        ax = fig.axes[0]
        # layers + surface + GWT + slip surface all drawn
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("Slip surface" in t for t in labels)
        assert any("GWT" in t for t in labels)
        assert any("Fill" in t for t in labels)
        plt.close(fig)

    def test_thrust_line_drawn_for_rigorous(self, spencer_result, geom):
        assert spencer_result.thrust_line is not None
        fig = plot_cross_section(spencer_result, geom)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("thrust" in t.lower() for t in labels)
        plt.close(fig)

    def test_reinforcement_drawn(self, geom_reinforced):
        res = analyze_slope(geom_reinforced, xc=22, yc=28, radius=20,
                            method="bishop", include_slice_data=True)
        fig = plot_cross_section(res, geom_reinforced)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("nail" in t.lower() for t in labels)
        assert any("Anchors" in t for t in labels)
        assert any("Geosynthetic" in t for t in labels)
        plt.close(fig)

    def test_tension_crack_annotated(self):
        g = SlopeGeometry(
            surface_points=[(0, 12), (8, 12), (20, 4), (32, 4)],
            soil_layers=[SlopeSoilLayer(
                "Stiff clay", top_elevation=12, bottom_elevation=-4,
                gamma=18, cu=45, analysis_mode="undrained")],
            tension_crack_depth=2.0, tension_crack_water_depth=1.0,
        )
        res = analyze_slope(g, xc=14, yc=20, radius=14, method="bishop",
                            include_slice_data=True)
        fig = plot_cross_section(res, g)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("Tension crack" in t for t in labels)
        plt.close(fig)

    def test_fos_required_annotation(self, spencer_result, geom):
        fig = plot_cross_section(spencer_result, geom, fos_required=2.0)
        texts = [t.get_text() for t in fig.axes[0].texts]
        assert any("UNSTABLE" in t for t in texts)
        plt.close(fig)


class TestTrialSurfaceMap:
    def test_circular_grid(self, geom):
        search = search_critical_surface(geom, nx=4, ny=4,
                                         method="bishop", n_slices=20)
        fig = plot_trial_surface_map(search, geom)
        assert len(fig.axes) >= 2  # main axes + colorbar
        plt.close(fig)

    def test_noncircular_trials_stored_and_drawn(self, geom):
        search = search_critical_surface(
            geom, surface_type="noncircular", n_trials=30, seed=7)
        assert len(search.trial_surfaces) > 0
        assert {"FOS", "points"} <= set(search.trial_surfaces[0])
        fig = plot_trial_surface_map(search, geom)
        plt.close(fig)

    def test_empty_search_raises(self, geom):
        from slope_stability.results import SearchResult
        with pytest.raises(ValueError, match="no drawable"):
            plot_trial_surface_map(SearchResult(), geom)


class TestSliceForces:
    def test_returns_figure(self, spencer_result):
        fig = plot_slice_forces(spencer_result)
        assert "FOS" in fig.axes[0].get_title()
        plt.close(fig)

    def test_requires_slice_data(self, geom):
        res = analyze_slope(geom, xc=22, yc=28, radius=20)
        with pytest.raises(ValueError, match="slice data"):
            plot_slice_forces(res)


class TestIntersliceForces:
    def test_rigorous_method(self, spencer_result, geom):
        fig = plot_interslice_forces(spencer_result, geom)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_nonrigorous_raises(self, geom):
        res = analyze_slope(geom, xc=22, yc=28, radius=20,
                            method="fellenius", include_slice_data=True)
        with pytest.raises(ValueError, match="rigorous"):
            plot_interslice_forces(res, geom)


class TestProbabilisticPlots:
    @pytest.fixture(scope="class")
    def variables(self):
        return {
            "phi:Clay": {"mean": 22, "cov": 0.10},
            "c_prime:Clay": {"mean": 12, "cov": 0.30, "dist": "lognormal"},
        }

    def test_mc_histogram(self, geom, variables):
        from slope_stability import monte_carlo_fos
        mc = monte_carlo_fos(geom, variables, xc=22, yc=28, radius=20,
                             n=100, seed=42)
        fig = plot_mc_histogram(mc)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert any("Lognormal" in t for t in labels)
        assert any("FOS = 1.0" in t for t in labels)
        plt.close(fig)

    def test_mc_histogram_requires_data(self):
        from slope_stability.probabilistic import MonteCarloResult
        with pytest.raises(ValueError, match="histogram"):
            plot_mc_histogram(MonteCarloResult())

    def test_fosm_tornado(self, geom, variables):
        from slope_stability import fosm_fos
        fosm = fosm_fos(geom, variables, xc=22, yc=28, radius=20)
        fig = plot_fosm_tornado(fosm)
        assert "Variance" in fig.axes[0].get_title()
        plt.close(fig)

    def test_fosm_tornado_requires_data(self):
        from slope_stability.probabilistic import FOSMResult
        with pytest.raises(ValueError, match="variance"):
            plot_fosm_tornado(FOSMResult())


class TestOptionalDependency:
    def test_module_imports_without_matplotlib_use(self):
        # plotting module itself must not import matplotlib at top level
        import importlib
        import slope_stability.plotting as mod
        importlib.reload(mod)

