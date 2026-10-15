"""Scale bar and north arrow for plans drawn as drawings (owner, 2026-10-15)."""

import pytest

from geotech_common import plan_furniture as pf


def test_nice_length_is_a_round_third_or_less():
    assert pf.nice_length(235.0) == 50.0
    assert pf.nice_length(41.0) == 10.0
    assert pf.nice_length(1200.0) == 200.0
    assert pf.nice_length(0.9) == 0.2


def test_north_follows_the_plan_or_points_up_and_says_so():
    assert pf.north_or_default(25.0) == (25.0, None)
    angle, note = pf.north_or_default(None)
    assert angle == 0.0 and "North drawn straight up" in note


def test_both_renderings_draw_the_furniture():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    texts = pf.mpl_scale_bar(ax, 10, 5, 20, 1.0, "m")
    texts += pf.mpl_north_arrow(ax, (90, 8), 30.0)
    assert [t.get_text() for t in texts] == ["0", "20 m", "N"]
    assert len(ax.patches) == pf.BAR_SEGMENTS
    plt.close(fig)
    go = pytest.importorskip("plotly.graph_objects")
    f = go.Figure()
    pf.plotly_scale_bar(f, 10, 5, 20, 1.0, "m")
    pf.plotly_north_arrow(f, (90, 8), 30.0)
    assert len(f.layout.shapes) == pf.BAR_SEGMENTS
    assert [a.text for a in f.layout.annotations] == ["0", "20 m", "", "N"]
