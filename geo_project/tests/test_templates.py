"""Template generator tests: geometry correctness + clean validation."""

import pytest

from geo_project.schema import Project
from geo_project.templates import (
    TEMPLATES,
    benched_slope,
    cut_with_berm,
    embankment_on_foundation,
    simple_slope,
)
from geo_project.validate import has_errors, summarize, validate


ALL_TEMPLATE_PROJECTS = {
    "simple_slope": lambda: simple_slope(10.0, 2.0),
    "simple_slope_foundation": lambda: simple_slope(10.0, 2.0,
                                                    foundation_depth=8.0),
    "benched_slope": lambda: benched_slope(15.0, 2.0, n_benches=2,
                                           bench_width=4.0),
    "embankment_half": lambda: embankment_on_foundation(
        6.0, crest_width=12.0, slope_ratio=2.5, foundation_depth=10.0),
    "embankment_full": lambda: embankment_on_foundation(
        6.0, crest_width=12.0, slope_ratio=2.5, foundation_depth=10.0,
        symmetric=True),
    "cut_with_berm": lambda: cut_with_berm(8.0, 4.0, berm_width=6.0),
}


@pytest.mark.parametrize("name", sorted(ALL_TEMPLATE_PROJECTS))
def test_every_template_validates_clean(name):
    p = ALL_TEMPLATE_PROJECTS[name]()
    assert isinstance(p, Project)
    assert p.geometry.provenance == "template"
    issues = validate(p)
    assert not has_errors(issues), f"{name}: {summarize(issues)}"
    # Surface strictly increasing in x.
    xs = [x for x, _ in p.geometry.surface_points]
    assert all(a < b for a, b in zip(xs, xs[1:]))
    # Every template logs its defaults in the assumption ledger.
    assert p.assumptions


def test_simple_slope_geometry():
    p = simple_slope(10.0, 2.0, crest_margin=15.0, toe_margin=12.0)
    assert p.geometry.surface_points == [
        (0.0, 10.0), (15.0, 10.0), (35.0, 0.0), (47.0, 0.0)]
    assert p.stratigraphy[0].bottom_elevation == -10.0
    assert p.layer_top(0) == 10.0


def test_simple_slope_with_foundation_layers():
    p = simple_slope(10.0, 2.0, foundation_depth=6.0)
    assert [L.name for L in p.stratigraphy] == ["Slope soil", "Foundation"]
    assert p.layer_bottom(0) == 0.0
    assert p.layer_bottom(1) == -6.0
    assert p.layer_top(1) == 0.0


def test_simple_slope_rejects_bad_inputs():
    with pytest.raises(ValueError):
        simple_slope(-1.0)
    with pytest.raises(ValueError):
        simple_slope(10.0, slope_ratio=0.0)


def test_benched_slope_geometry():
    p = benched_slope(12.0, slope_ratio=2.0, n_benches=2, bench_width=4.0,
                      crest_margin=10.0, toe_margin=10.0)
    pts = p.geometry.surface_points
    # crest flat, 3 lifts of 4 m height (8 m horizontal), 2 benches of 4 m.
    assert pts[0] == (0.0, 12.0)
    assert pts[1] == (10.0, 12.0)
    assert pts[2] == (18.0, 8.0)    # end of lift 1
    assert pts[3] == (22.0, 8.0)    # end of bench 1
    assert pts[4] == (30.0, 4.0)    # end of lift 2
    assert pts[5] == (34.0, 4.0)    # end of bench 2
    assert pts[6] == (42.0, 0.0)    # toe
    assert pts[7] == (52.0, 0.0)
    # z values descend monotonically.
    zs = [z for _, z in pts]
    assert all(a >= b for a, b in zip(zs, zs[1:]))


def test_benched_zero_benches_degenerates_to_simple():
    p = benched_slope(10.0, 2.0, n_benches=0)
    ref = simple_slope(10.0, 2.0)
    assert p.geometry.surface_points == ref.geometry.surface_points


def test_embankment_half_section_geometry():
    p = embankment_on_foundation(6.0, crest_width=12.0, slope_ratio=2.0,
                                 foundation_depth=10.0, margin=10.0)
    assert p.geometry.surface_points == [
        (0.0, 6.0), (6.0, 6.0), (18.0, 0.0), (28.0, 0.0)]
    assert [L.name for L in p.stratigraphy] == ["Embankment fill",
                                                "Foundation"]
    assert p.layer_bottom(0) == 0.0
    assert p.layer_bottom(1) == -10.0


def test_embankment_full_section_geometry():
    p = embankment_on_foundation(6.0, crest_width=12.0, slope_ratio=2.0,
                                 foundation_depth=10.0, margin=10.0,
                                 symmetric=True)
    pts = p.geometry.surface_points
    assert pts[0] == (0.0, 0.0)
    assert pts[2] == (22.0, 6.0)     # top of left slope
    assert pts[3] == (34.0, 6.0)     # end of crest
    assert pts[-1] == (56.0, 0.0)


def test_cut_with_berm_geometry():
    p = cut_with_berm(8.0, 4.0, berm_width=6.0, slope_ratio_upper=2.0,
                      slope_ratio_lower=1.5, crest_margin=10.0,
                      toe_margin=10.0)
    pts = p.geometry.surface_points
    assert pts[0] == (0.0, 12.0)
    assert pts[1] == (10.0, 12.0)
    assert pts[2] == (26.0, 4.0)     # bottom of upper face (16 m run)
    assert pts[3] == (32.0, 4.0)     # end of berm
    assert pts[4] == (38.0, 0.0)     # toe (6 m run for 4 m at 1.5H:1V)
    assert pts[5] == (48.0, 0.0)


def test_registry_complete():
    assert set(TEMPLATES) == {"simple_slope", "benched_slope",
                              "embankment_on_foundation", "cut_with_berm"}
    for fn in TEMPLATES.values():
        assert callable(fn)
        assert fn.__doc__  # LLM-facing docstrings exist
