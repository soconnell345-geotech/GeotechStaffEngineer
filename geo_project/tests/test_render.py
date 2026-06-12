"""Echo-back render tests: PNG produced, vertex table correct, all templates."""

import pytest

from geo_project.render import EchoBack, echo_back, vertex_table
from geo_project.schema import (
    Anchor,
    GeosyntheticLayer,
    Loads,
    Nail,
    Reinforcement,
    Surcharge,
    Water,
)
from geo_project.templates import (
    benched_slope,
    cut_with_berm,
    embankment_on_foundation,
    simple_slope,
)
from geo_project.ingest import from_points, from_vision_draft

matplotlib = pytest.importorskip("matplotlib")


def _decorated_project():
    p = from_points(
        [(0, 10), (10, 10), (30, 0), (45, 0)],
        layer_boundaries={"clay_top": [(0, 2.0), (45, 1.0)]},
        layer_names=["Sand", "Clay"],
        gwt_points=[(0, 5.0), (45, -1.0)],
    )
    p.loads = Loads(surcharges=[Surcharge(q=12.0, x_start=0.0, x_end=8.0,
                                          label="traffic")], kh=0.1)
    p.reinforcement = Reinforcement(
        nails=[Nail(x_head=25.0, z_head=2.5, length=8.0)],
        anchors=[Anchor(x_head=20.0, z_head=5.0, length=10.0,
                        T_allow=150.0)],
        geosynthetics=[GeosyntheticLayer(elevation=1.0, T_allow=30.0)],
    )
    return p


def test_echo_back_writes_png(tmp_path):
    p = _decorated_project()
    out = tmp_path / "echo.png"
    eb = echo_back(p, str(out))
    assert isinstance(eb, EchoBack)
    assert eb.image_path == str(out)
    assert out.exists() and out.stat().st_size > 10_000  # a real figure
    assert "S1" in eb.vertex_table


def test_echo_back_without_path_returns_table_only():
    eb = echo_back(_decorated_project(), path=None)
    assert eb.image_path is None
    assert "Ground surface" in eb.vertex_table


def test_vertex_table_contents():
    p = _decorated_project()
    t = vertex_table(p)
    # Numbered surface vertices.
    assert "S1: (0, 10)" in t
    assert "S4: (45, 0)" in t
    # Boundary vertices keyed B<k>.<i>.
    assert "B1.1: (0, 2)" in t
    # Layers with resolved elevations.
    assert "Sand: top 10 m, bottom 2 m" in t
    assert "boundary 'clay_top'" in t
    # GWT, loads, reinforcement all listed.
    assert "W1: (0, 5)" in t
    assert "Q1: 12 kPa over x = 0 to 8 m (traffic)" in t
    assert "kh = 0.1" in t
    assert "N1: (25, 2.5) L=8 m @ 15 deg" in t
    assert "A1: (20, 5) L=10 m @ 15 deg, T=150 kN/m" in t
    assert "G1: z=1 m" in t
    assert "PROVENANCE: user" in t


def test_vision_draft_render_carries_warning(tmp_path):
    p = from_vision_draft({
        "surface_points": [(0, 8), (26, 0), (36, 0)],
    })
    eb = echo_back(p, str(tmp_path / "draft.png"))
    assert "UNCONFIRMED VISION DRAFT" in eb.vertex_table


def test_render_works_pre_materials_and_pre_analysis(tmp_path):
    # Straight from a template: empty strengths, no analyses — must render.
    p = simple_slope(10.0, 2.0, foundation_depth=6.0)
    eb = echo_back(p, str(tmp_path / "tpl.png"))
    assert (tmp_path / "tpl.png").exists()
    assert "Slope soil" in eb.vertex_table


@pytest.mark.parametrize("factory", [
    lambda: simple_slope(10.0, 2.0),
    lambda: benched_slope(15.0, 2.0, n_benches=2, bench_width=4.0),
    lambda: embankment_on_foundation(6.0, crest_width=12.0,
                                     foundation_depth=10.0),
    lambda: embankment_on_foundation(6.0, crest_width=12.0,
                                     foundation_depth=10.0, symmetric=True),
    lambda: cut_with_berm(8.0, 4.0, berm_width=6.0),
])
def test_every_template_renders(tmp_path, factory):
    p = factory()
    out = tmp_path / "out.png"
    eb = echo_back(p, str(out))
    assert out.exists() and out.stat().st_size > 5_000
    assert eb.vertex_table


def test_render_rejects_empty_geometry(tmp_path):
    from geo_project.schema import Project
    with pytest.raises(ValueError):
        echo_back(Project(), str(tmp_path / "x.png"))
