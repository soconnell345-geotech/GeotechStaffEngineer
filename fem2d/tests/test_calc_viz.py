"""Tests for fem2d.plotting and fem2d.calc_steps (Phase 3, calc-viz)."""

import numpy as np
import pytest

from fem2d import (
    analyze_slope_srm, analyze_foundation, analyze_seepage,
    analyze_staged, ConstructionPhase, assign_element_groups,
    generate_rect_mesh, generate_slope_mesh, detect_boundary_nodes,
    assign_layers_by_elevation, convert_to_t6,
)
from fem2d import calc_steps as cs
from calc_package.data_model import CalcSection, TableData, CalcStep

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fem2d.plotting import (  # noqa: E402
    plot_mesh, plot_deformed_mesh, plot_contour, plot_plastic_points,
    plot_srf_curve, plot_failure_mechanism, plot_seepage,
    CONTOUR_FIELDS,
)


@pytest.fixture(scope="module")
def srm_result():
    surface = [(0, 10), (10, 10), (22, 4), (32, 4)]
    layers = [dict(name="Soft clay fill", bottom_elevation=-6, E=20000,
                   nu=0.3, c=8, phi=15, gamma=19)]
    return analyze_slope_srm(surface, layers, nx=12, ny=6,
                             srf_tol=0.1, srf_range=(1.0, 3.0)), layers


@pytest.fixture(scope="module")
def foundation_result():
    return analyze_foundation(B=2.0, q=100.0, depth=8.0, E=25000,
                              nu=0.3, nx=14, ny=7)


@pytest.fixture(scope="module")
def seepage_result():
    n2, e2 = generate_rect_mesh(0, 10, -4, 0, 12, 6)
    head_bcs = (
        [(int(i), 6.0) for i in np.where(np.abs(n2[:, 0]) < 0.01)[0]]
        + [(int(i), 2.0)
           for i in np.where(np.abs(n2[:, 0] - 10) < 0.01)[0]])
    return analyze_seepage(n2, e2, k=1e-5, head_bcs=head_bcs)


@pytest.fixture(scope="module")
def staged_result():
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
    return analyze_staged(nodes, elements, mats, 18.0, bc, groups,
                          phases)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

class TestMeshPlot:
    def test_plain_mesh(self, foundation_result):
        r = foundation_result
        fig = plot_mesh(r.nodes, r.elements)
        text = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "T6" in text
        plt.close(fig)

    def test_materials_and_bcs(self):
        surface = [(0, 10), (10, 10), (22, 4), (32, 4)]
        nodes, elements = generate_slope_mesh(surface, 8.0, 10, 5,
                                              x_extend_left=2,
                                              x_extend_right=2)
        nodes, elements = convert_to_t6(nodes, elements)
        bc = detect_boundary_nodes(nodes)
        lid = assign_layers_by_elevation(nodes, elements, [-6])
        fig = plot_mesh(nodes, elements, material_ids=lid,
                        material_names=["Fill"], bc_nodes=bc)
        plt.close(fig)


class TestDeformedMesh:
    def test_auto_scale_annotated(self, foundation_result):
        fig = plot_deformed_mesh(foundation_result)
        text = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "scale" in text
        assert "u_max" in text
        plt.close(fig)

    def test_explicit_scale(self, foundation_result):
        fig = plot_deformed_mesh(foundation_result, scale=50)
        labels = [t.get_text()
                  for t in fig.axes[0].get_legend().get_texts()]
        assert any("x50" in t for t in labels)
        plt.close(fig)


class TestContours:
    @pytest.mark.parametrize("field", CONTOUR_FIELDS)
    def test_all_fields(self, foundation_result, field):
        fig = plot_contour(foundation_result, field=field)
        assert len(fig.axes) >= 2  # main + colorbar
        plt.close(fig)

    def test_unknown_field_raises(self, foundation_result):
        with pytest.raises(ValueError, match="Unknown field"):
            plot_contour(foundation_result, field="nope")


class TestSRMPlots:
    def test_plastic_points(self, srm_result):
        res, _ = srm_result
        assert res.plastic_points is not None
        assert res.plastic_points["n_plastic"] > 0
        fig = plot_plastic_points(res)
        text = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "Gauss points" in text
        plt.close(fig)

    def test_srf_curve(self, srm_result):
        res, _ = srm_result
        fig = plot_srf_curve(res)
        labels = [t.get_text()
                  for t in fig.axes[0].get_legend().get_texts()]
        assert any("FOS" in t for t in labels)
        plt.close(fig)

    def test_failure_mechanism(self, srm_result):
        res, _ = srm_result
        fig = plot_failure_mechanism(res)
        assert "Failure Mechanism" in fig.axes[0].get_title()
        plt.close(fig)

    def test_plastic_points_requires_srm(self, foundation_result):
        with pytest.raises(ValueError, match="plastic_points"):
            plot_plastic_points(foundation_result)

    def test_srf_curve_requires_srm(self, foundation_result):
        with pytest.raises(ValueError, match="srf_curve"):
            plot_srf_curve(foundation_result)


class TestSeepagePlot:
    def test_head_and_vectors(self, seepage_result):
        fig = plot_seepage(seepage_result)
        text = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "head" in text
        plt.close(fig)


# ---------------------------------------------------------------------
# calc_steps
# ---------------------------------------------------------------------

def _titles(sections):
    return [s.title for s in sections]


def _flat_text(sections):
    out = []
    for sec in sections:
        out.append(sec.title)
        for item in sec.items:
            if isinstance(item, CalcStep):
                out += [item.title, item.equation, item.notes,
                        str(item.result_value)]
            elif isinstance(item, TableData):
                out += [item.title, item.notes] + \
                    [str(h) for h in item.headers]
                for row in item.rows:
                    out += [str(c) for c in row]
            elif isinstance(item, str):
                out.append(item)
    return " | ".join(out)


class TestCalcStepsSRM:
    def test_sections(self, srm_result):
        res, layers = srm_result
        sections = cs.get_calc_steps(res, {"soil_layers": layers})
        titles = _titles(sections)
        assert "Model Summary" in titles
        assert "Strength Reduction Method (Slope Stability)" in titles
        assert "Results" in titles
        assert "Checks" in titles
        text = _flat_text(sections)
        assert "SRF Trial History" in text
        assert "Plastic Zone Extent" in text
        assert "Material Properties" in text
        assert "Soft clay fill" in text

    def test_input_summary(self, srm_result):
        res, layers = srm_result
        items = cs.get_input_summary(res, {"FOS_required": 1.3})
        text = " ".join(f"{i.name}={i.value}" for i in items)
        assert "FOS_req=1.3" in text
        assert "T6" in text

    def test_figures(self, srm_result):
        res, layers = srm_result
        figs = cs.get_figures(res, {"soil_layers": layers})
        titles = [f.title for f in figs]
        assert "Finite Element Mesh" in titles
        assert "Deformed Mesh" in titles
        assert "Failure Mechanism" in titles
        assert "Plastic Point Map" in titles
        assert "SRF vs Displacement" in titles
        for f in figs:
            assert len(f.image_base64) > 1000


class TestCalcStepsElastic:
    def test_sections_and_locations(self, foundation_result):
        sections = cs.get_calc_steps(foundation_result, None)
        text = _flat_text(sections)
        assert "Displacement Extremes" in text
        assert " at (" in text          # max-displacement locations
        assert "Stress Extremes" in text

    def test_figures(self, foundation_result):
        figs = cs.get_figures(foundation_result, None)
        titles = [f.title for f in figs]
        assert "Displacement Contours" in titles
        assert "Vertical Stress Contours" in titles


class TestCalcStepsSeepage:
    def test_sections(self, seepage_result):
        sections = cs.get_calc_steps(seepage_result, None)
        text = _flat_text(sections)
        assert "Seepage Result Summary" in text
        assert "Total head range" in text

    def test_figure(self, seepage_result):
        figs = cs.get_figures(seepage_result, None)
        assert [f.title for f in figs] == \
            ["Head Contours & Flow Vectors"]


class TestCalcStepsStaged:
    def test_phase_sections(self, staged_result):
        sections = cs.get_calc_steps(staged_result, None)
        titles = _titles(sections)
        assert "Construction Sequence" in titles
        assert "Phase 1: Lower lift" in titles
        assert "Phase 2: Full height" in titles

    def test_phase_figures(self, staged_result):
        figs = cs.get_figures(staged_result, None)
        assert len(figs) == 2
        assert "Lower lift" in figs[0].title


class TestRegistry:
    def test_fem2d_listed(self):
        from calc_package import list_supported_modules
        assert "fem2d" in list_supported_modules()

    def test_generate_html(self, foundation_result):
        pytest.importorskip("jinja2")
        from calc_package import generate_calc_package
        html = generate_calc_package("fem2d", foundation_result, None,
                                     project_name="T", engineer="E")
        assert "2D Finite Element Analysis" in html
        assert "data:image/png;base64" in html
