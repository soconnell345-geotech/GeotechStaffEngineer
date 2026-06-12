"""End-to-end calc-package render tests for the calc-viz build
(slope_stability modern engine + fem2d), including graceful
degradation when matplotlib is unavailable (Phase 4).
"""

import builtins

import numpy as np
import pytest

from calc_package import generate_calc_package


# ---------------------------------------------------------------------
# Fixtures: one model per analysis type
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def slope_setup():
    from slope_stability import (
        SlopeGeometry, SlopeSoilLayer, Anchor, Geosynthetic,
        analyze_slope, search_critical_surface, monte_carlo_fos,
        fosm_fos,
    )
    geom = SlopeGeometry(
        surface_points=[(0, 20), (10, 20), (30, 10), (50, 10)],
        soil_layers=[
            SlopeSoilLayer("Fill", top_elevation=20,
                           bottom_elevation=12, gamma=19, phi=28,
                           c_prime=5),
            SlopeSoilLayer("Clay", top_elevation=12,
                           bottom_elevation=0, gamma=18, phi=22,
                           c_prime=12),
        ],
        gwt_points=[(0, 16), (30, 9), (50, 9)],
        anchors=[Anchor(x_head=24, z_head=13, length=12, T_allow=80)],
        geosynthetics=[Geosynthetic(elevation=11, T_allow=40)],
    )
    result = analyze_slope(geom, xc=22, yc=28, radius=20,
                           method="spencer", include_slice_data=True,
                           compare_methods=True)
    search = search_critical_surface(geom, nx=3, ny=3,
                                     method="bishop", n_slices=20)
    variables = {
        "phi:Clay": {"mean": 22, "cov": 0.10, "source": "lab"},
        "c_prime:Clay": {"mean": 12, "cov": 0.30,
                         "dist": "lognormal"},
    }
    mc = monte_carlo_fos(geom, variables, xc=22, yc=28, radius=20,
                         n=80, seed=42)
    fosm = fosm_fos(geom, variables, xc=22, yc=28, radius=20)
    analysis = {"geom": geom, "search": search, "mc": mc,
                "fosm": fosm, "variables": variables,
                "FOS_required": 1.5}
    return result, analysis


@pytest.fixture(scope="module")
def srm_setup():
    from fem2d import analyze_slope_srm
    layers = [dict(name="Soft clay fill", bottom_elevation=-6,
                   E=20000, nu=0.3, c=8, phi=15, gamma=19)]
    result = analyze_slope_srm(
        [(0, 10), (10, 10), (22, 4), (32, 4)], layers,
        nx=12, ny=6, srf_tol=0.1)
    return result, {"soil_layers": layers, "FOS_required": 1.3}


# ---------------------------------------------------------------------
# Full-feature renders
# ---------------------------------------------------------------------

class TestSlopeEndToEnd:
    def test_full_package(self, slope_setup):
        pytest.importorskip("matplotlib")
        result, analysis = slope_setup
        html = generate_calc_package(
            "slope_stability", result, analysis,
            project_name="Calc-Viz E2E", engineer="QA")
        # structure
        for token in [
            "Slope Geometry &amp; Soil Properties",
            "Analysis Method &amp; Factor of Safety",
            "Reinforcement", "Method Comparison",
            "Slice Force Table", "Critical Surface Search",
            "Probabilistic Analysis (Reliability)",
            "Stability Check",
        ]:
            assert token in html, token
        # tables populated
        assert "E_L (kN/m)" in html
        assert "Mobilized Reinforcement Forces" in html
        # figures embedded as base64 PNG
        assert html.count("data:image/png;base64") >= 5
        assert len(html) > 100_000

    def test_minimal_package_still_renders(self, slope_setup):
        result, analysis = slope_setup
        html = generate_calc_package(
            "slope_stability", result, {"geom": analysis["geom"]})
        assert "Slice Force Table" in html
        assert "Critical Surface Search" not in html


class TestFemEndToEnd:
    def test_srm_package(self, srm_setup):
        pytest.importorskip("matplotlib")
        result, analysis = srm_setup
        html = generate_calc_package(
            "fem2d", result, analysis,
            project_name="Calc-Viz FEM E2E", engineer="QA")
        for token in [
            "Model Summary", "Material Properties",
            "Strength Reduction Method (Slope Stability)",
            "SRF Trial History", "Plastic Zone Extent",
            "Results", "Checks",
        ]:
            assert token in html, token
        assert html.count("data:image/png;base64") >= 4

    def test_seepage_package(self):
        pytest.importorskip("matplotlib")
        from fem2d import analyze_seepage, generate_rect_mesh
        n, e = generate_rect_mesh(0, 10, -4, 0, 12, 6)
        head_bcs = (
            [(int(i), 6.0)
             for i in np.where(np.abs(n[:, 0]) < 0.01)[0]]
            + [(int(i), 2.0)
               for i in np.where(np.abs(n[:, 0] - 10) < 0.01)[0]])
        res = analyze_seepage(n, e, k=1e-5, head_bcs=head_bcs)
        html = generate_calc_package("fem2d", res, None)
        assert "Seepage Result Summary" in html
        assert "data:image/png;base64" in html

    def test_staged_package(self):
        from fem2d import (analyze_staged, ConstructionPhase,
                           assign_element_groups, generate_rect_mesh,
                           detect_boundary_nodes)
        nodes, elements = generate_rect_mesh(0, 12, -6, 0, 10, 5)
        bc = detect_boundary_nodes(nodes)
        groups = assign_element_groups(nodes, elements, {
            "lower": {"y_max": -3.0}, "upper": {"y_min": -3.0}})
        mats = [dict(E=25000, nu=0.3, c=20, phi=25, gamma=18)] \
            * len(elements)
        phases = [
            ConstructionPhase(name="Lower lift",
                              active_soil_groups=["lower"]),
            ConstructionPhase(name="Full height",
                              active_soil_groups=["lower", "upper"]),
        ]
        res = analyze_staged(nodes, elements, mats, 18.0, bc, groups,
                             phases)
        html = generate_calc_package("fem2d", res, None)
        assert "Construction Sequence" in html
        assert "Phase 2: Full height" in html


# ---------------------------------------------------------------------
# Graceful degradation without matplotlib
# ---------------------------------------------------------------------

@pytest.fixture
def no_matplotlib(monkeypatch):
    """Make any 'import matplotlib[...]' raise ImportError."""
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError(f"blocked for test: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


class TestNoMatplotlibDegradation:
    def test_slope_text_only(self, slope_setup, no_matplotlib):
        result, analysis = slope_setup
        html = generate_calc_package(
            "slope_stability", result, analysis,
            project_name="Degraded", engineer="QA")
        # no figures...
        assert "data:image/png;base64" not in html
        # ...but the full table/step content survives
        assert "Slice Force Table" in html
        assert "Probabilistic Analysis (Reliability)" in html
        assert "Method Comparison" in html
        assert "Stability Check" in html

    def test_fem_text_only(self, srm_setup, no_matplotlib):
        result, analysis = srm_setup
        html = generate_calc_package("fem2d", result, analysis)
        assert "data:image/png;base64" not in html
        assert "SRF Trial History" in html
        assert "Checks" in html

    def test_plotting_helpers_raise_import_error(self, slope_setup,
                                                 no_matplotlib):
        from slope_stability.plotting import plot_cross_section
        result, analysis = slope_setup
        with pytest.raises(ImportError):
            plot_cross_section(result, analysis["geom"])
