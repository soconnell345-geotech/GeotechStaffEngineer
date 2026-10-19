"""plot_data's own lettering and a plan's view (BACKLOG GM-146, GM-147,
GM-152, and GM-145 through the notes' check).

- GM-147: a plan's view takes in every outline drawn with a margin (a
  40 x 25 m outline lost its top edge; the view stopped at 24.76).
- GM-146: reference lines are obstacles for point names, as the data are.
- GM-152: the layout check covers the reference lines' names, the plan's
  furniture and the point names, not the notes alone.
- GM-145: hatching under a note is reported; a halo keeps it off.
"""

import pytest

from profile_figure import render_data_plot

pytest.importorskip("plotly")


@pytest.fixture
def closed(monkeypatch):
    """The axes' limits as the figure is closed."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    seen = {}
    real_close = plt.close

    def spy(fig=None):
        if fig is not None and getattr(fig, "axes", None):
            ax = fig.axes[0]
            seen["xlim"], seen["ylim"] = ax.get_xlim(), ax.get_ylim()
            seen["texts"] = {t.get_text(): t.xy for t in ax.texts
                             if hasattr(t, "xy")}
        return real_close(fig)

    monkeypatch.setattr(plt, "close", spy)
    return seen


def _outline(w, h, label="Site"):
    return {"x": [0, w, w, 0, 0], "y": [0, 0, h, h, 0], "label": label,
            "style": "polygon", "color": "black"}


BORINGS = {"x": [10, 30], "y": [12, 20], "label": "Borings",
           "style": "markers", "labels": ["B-1", "B-2"]}


# ---------------------------------------------------------------------------
# GM-147: the plan's view takes in every outline with a margin
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width_in", [5.0, 6.5, 8.0])
@pytest.mark.parametrize("with_points", [False, True])
@pytest.mark.parametrize("legend", [None, False])
def test_the_view_takes_in_the_whole_outline(closed, width_in, with_points,
                                             legend):
    series = [_outline(40, 25)] + ([BORINGS] if with_points else [])
    render_data_plot(series, plan_view=True, north_deg=0, title="Site plan",
                     width_in=width_in, legend=legend, interactive=False)
    (x0, x1), (y0, y1) = closed["xlim"], closed["ylim"]
    # every edge inside, with a margin of at least 1 % of the site
    assert x0 < -0.4 and x1 > 40.4, closed
    assert y0 < -0.25 and y1 > 25.25, closed


@pytest.mark.parametrize("w, h", [(140, 30), (18, 90), (400, 12), (5, 5)])
def test_long_and_narrow_sites_keep_their_edges_and_furniture_clear(
        closed, w, h):
    """The band and margins are sized in points as printed: the scale bar's
    names stay off a long flat site's property line, and the arrow off the
    bar on a narrow one."""
    res = render_data_plot([_outline(w, h)], plan_view=True, north_deg=20,
                           xlabel="Local grid (m)", interactive=False)
    (x0, x1), (y0, y1) = closed["xlim"], closed["ylim"]
    assert x0 < 0 < w < x1 and y0 < 0 < h < y1, closed
    assert res.warnings == [], res.warnings


def test_the_interactive_plan_has_the_pngs_view(closed):
    res = render_data_plot([_outline(40, 25), BORINGS], plan_view=True,
                           north_deg=0)
    lay = res.figure.layout
    assert tuple(lay.xaxis.range) == pytest.approx(closed["xlim"])
    assert tuple(lay.yaxis.range) == pytest.approx(closed["ylim"])
    # the equal aspect fits the plot's box; it never trims the range
    assert lay.xaxis.constrain == "domain" == lay.yaxis.constrain


# ---------------------------------------------------------------------------
# GM-146: reference lines are obstacles for point names
# ---------------------------------------------------------------------------

def test_point_names_keep_clear_of_reference_lines():
    """Points a few units right of a vertical line and just above a
    horizontal one: the names were put across the lines (top centre comes
    first); each has a clear spot, so none is crossed."""
    pts = {"x": [51, 53, 70, 30], "y": [10, 31, 30.6, 12],
           "label": "Samples", "style": "markers",
           "labels": ["S-1", "S-2", "S-3", "S-4"]}
    res = render_data_plot([pts], vlines=[{"value": 50, "label": "LL = 50"}],
                           hlines=[{"value": 30, "label": "Limit"}],
                           xlim=[0, 80], ylim=[0, 40], interactive=False)
    assert res.warnings == [], res.warnings


def test_a_name_with_no_clear_spot_is_reported():
    """A point boxed in by two reference lines closer than its name is
    wide: every spot crosses one, and the warning names the name and the
    line."""
    pts = {"x": [50.6], "y": [20.3], "label": "Samples",
           "style": "markers", "labels": ["Sample S-10"]}
    res = render_data_plot([pts], vlines=[50, 51.2], hlines=[20, 20.6],
                           xlim=[40, 60], ylim=[10, 30], interactive=False)
    assert len(res.warnings) == 1, res.warnings
    assert res.warnings[0].startswith(
        "series[0] point name 'Sample S-10': crossed by the reference line "
        "at"), res.warnings


# ---------------------------------------------------------------------------
# GM-152: the check covers reference-line names and the plan's furniture
# ---------------------------------------------------------------------------

def test_a_reference_line_name_moves_off_the_data_and_the_twin_follows():
    """A worked example's "Best estimate" stood on a data point at the
    right end of its line: the name takes the clearest end of its line,
    and the interactive chart puts it at the same end."""
    pts = {"x": [0, 1, 2, 3, 4], "y": [0.16, 0.14, 0.21, 0.21, 0.21],
           "label": "Cases", "style": "markers"}
    res = render_data_plot([pts], hlines=[{"value": 0.16,
                                           "label": "Best estimate"}],
                           ylim=[0, 1])
    assert res.warnings == [], res.warnings
    a = [a for a in res.figure.layout.annotations
         if a.text == "Best estimate"][0]
    assert (a.xanchor, a.yanchor) != ("right", "bottom")   # not top right


def test_a_vertical_line_name_crossed_by_the_data_is_reported():
    """Data along the top and the bottom of the plot: every end of the
    line is crossed, and the warning names the name and the series."""
    top = {"x": [0, 100], "y": [97, 97], "label": "Envelope",
           "style": "line"}
    low = {"x": [0, 100], "y": [3, 3], "label": "Floor",
           "style": "line"}
    res = render_data_plot([top, low], vlines=[{"value": 20,
                                                "label": "Lower bound"}],
                           xlim=[0, 100], ylim=[0, 100],
                           interactive=False)
    assert any(w.startswith("the name of the reference line at 20 "
                            "('Lower bound'): crossed by")
               for w in res.warnings), res.warnings


def test_a_horizontal_line_name_crossed_by_the_data_is_reported():
    left = {"x": [1, 1], "y": [0, 100], "label": "Pile",
            "style": "line"}
    right = {"x": [99, 99], "y": [0, 100], "label": "Wall",
             "style": "line"}
    res = render_data_plot([left, right], hlines=[{"value": 50,
                                                   "label": "Water table"}],
                           xlim=[0, 100], ylim=[0, 100], interactive=False)
    assert any(w.startswith("the name of the reference line at 50 "
                            "('Water table'): crossed by")
               for w in res.warnings), res.warnings


def test_a_note_on_the_scale_bar_is_reported_once_on_the_note(closed):
    res = render_data_plot([_outline(40, 25)], plan_view=True, north_deg=0,
                           interactive=False, annotations=[
                               {"text": "Stockpile area", "x": 0.73,
                                "y": 0.105, "coords": "axes"}])
    assert len(res.warnings) == 1, res.warnings
    w = res.warnings[0]
    assert w.startswith("annotations[0] 'Stockpile area': "), w
    assert "the scale bar" in w, w


def test_the_furniture_is_named_when_it_is_crossed(closed):
    """A reference line drawn down through the scale bar's end name: the
    warning names the furniture and the line."""
    site = _outline(40, 25)
    render_data_plot([site], plan_view=True, north_deg=0, interactive=False)
    x_end = closed["texts"]["10 m"][0]
    res = render_data_plot([site], plan_view=True, north_deg=0,
                           vlines=[x_end], interactive=False)
    assert any(w.startswith("the scale bar's '10 m': crossed by the "
                            "reference line at") for w in res.warnings), \
        res.warnings


# ---------------------------------------------------------------------------
# GM-145 through the notes: hatching under a note
# ---------------------------------------------------------------------------

HATCHED = {"x": [0, 10, 10, 0], "y": [0, 0, 5, 5], "label": "Soft clay",
           "style": "polygon", "hatch": "/", "color": "#555555"}


def test_hatching_under_a_note_is_reported():
    res = render_data_plot([HATCHED], annotations=[
        {"text": "Soft clay zone", "x": 5, "y": 2.5}],
        xlim=[0, 10], ylim=[0, 10], interactive=False)
    assert res.warnings == [
        "annotations[0] 'Soft clay zone': crossed by the hatching of 'Soft "
        "clay' (move it, or give it box: true or halo: true)"], res.warnings


@pytest.mark.parametrize("keep_off", [{"halo": True}, {"box": True}])
def test_a_halo_or_a_box_keeps_hatching_off_a_note(keep_off):
    res = render_data_plot([HATCHED], annotations=[
        dict({"text": "Soft clay zone", "x": 5, "y": 2.5}, **keep_off)],
        xlim=[0, 10], ylim=[0, 10], interactive=False)
    assert res.warnings == [], res.warnings


def test_a_name_inside_a_hatched_zone_is_kept_readable():
    """A point inside a hatched zone has no spot off the hatching: its name
    gets a thin ring of the background, so the hatch stops short of it."""
    pt = {"x": [5], "y": [2.5], "label": "Boring", "style": "markers",
          "labels": ["B-1"]}
    res = render_data_plot([HATCHED, pt], xlim=[0, 10], ylim=[0, 10],
                           interactive=False)
    assert res.warnings == [], res.warnings
