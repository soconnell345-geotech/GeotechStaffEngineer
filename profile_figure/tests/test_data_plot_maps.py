"""plot_data for plan views, outlines, named points and cases (2026-10-15).

Field session 2026-10-15: plan views were drawn with unequal axis scales
("annoying how the plots aren't 1:1 in the axes"), a footprint could only be
an unfilled line, points could not carry their boring names or hover text,
and the cases of a sensitivity study were drawn on a numeric "case index"
axis joined by lines (ticks at 2.5, 7.5 ...). These pin the general options
that fix each, in BOTH renderings: the PNG for the report and its interactive
twin for the chat.
"""

import io

import pytest

from profile_figure import render_data_plot

pytest.importorskip("plotly")

#: A long, low site: 400 m across, 160 m up.
FOOTPRINT = {"x": [80, 120, 120, 110, 110, 120, 120, 80],
             "y": [250, 250, 225, 225, 210, 210, 170, 170],
             "label": "Footprint", "style": "polygon", "fill": True}
POINTS = {"x": [90, 180, 400, 480], "y": [220, 290, 140, 170],
          "label": "Borings", "style": "markers",
          "labels": ["B-1", "B-2", "B-3", "B-4"],
          "hover": ["B-1<br>rock 12.0 m", "B-2", "B-3", "B-4"]}


def _png_size(png: bytes):
    from PIL import Image
    with Image.open(io.BytesIO(png)) as im:
        return im.size


# ---------------------------------------------------------------------------
# Drawn to scale
# ---------------------------------------------------------------------------

def test_equal_aspect_ties_the_scales_in_both_renderings():
    res = render_data_plot([FOOTPRINT, POINTS], equal_aspect=True)
    assert res.equal_aspect is True
    assert res.figure.layout.yaxis.scaleanchor == "x"
    assert res.figure.layout.yaxis.scaleratio == 1
    assert "drawn to scale" in res.summary()
    plain = render_data_plot([FOOTPRINT, POINTS])
    assert plain.figure.layout.yaxis.scaleanchor is None


def test_a_wide_site_gets_a_wide_figure():
    """The figure takes the data's proportions instead of a fixed box."""
    wide = render_data_plot([FOOTPRINT, POINTS], equal_aspect=True,
                            legend=False)
    w, h = _png_size(wide.png_bytes)
    assert w / h > 1.6
    tall = render_data_plot(
        [{"x": [0, 40, 40, 0], "y": [0, 0, 160, 160], "style": "polygon"}],
        equal_aspect=True)
    w2, h2 = _png_size(tall.png_bytes)
    assert h2 > w2


def test_the_png_really_is_one_to_one():
    """Measure the PNG's own axes: one metre across = one metre up."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    import profile_figure.data_plot as dp

    seen = {}
    real_close = plt.close

    def spy(fig=None):
        if fig is not None and hasattr(fig, "axes") and fig.axes:
            ax = fig.axes[0]
            bbox = ax.get_window_extent()
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            seen["px_per_x"] = bbox.width / abs(x1 - x0)
            seen["px_per_y"] = bbox.height / abs(y1 - y0)
        return real_close(fig)

    try:
        plt.close = spy
        dp.render_data_plot([FOOTPRINT, POINTS], equal_aspect=True)
    finally:
        plt.close = real_close
    assert seen["px_per_x"] == pytest.approx(seen["px_per_y"], rel=0.01)


def test_equal_aspect_is_refused_politely_on_cases_or_log_axes():
    res = render_data_plot([{"x": ["a", "b"], "y": [1, 2]}],
                           equal_aspect=True)
    assert res.equal_aspect is False
    assert any("equal_aspect ignored" in w for w in res.warnings)
    res = render_data_plot([{"x": [1, 10], "y": [1, 2]}], equal_aspect=True,
                           logx=True)
    assert res.equal_aspect is False


def test_the_legend_leaves_a_map_alone():
    res = render_data_plot([FOOTPRINT, POINTS], equal_aspect=True)
    legend = res.figure.layout.legend
    assert legend.x is None or legend.x >= 1.0       # Plotly's outside default
    inside = render_data_plot([FOOTPRINT, POINTS], equal_aspect=True,
                              legend_position="inside")
    assert inside.figure.layout.legend.x == pytest.approx(0.01)
    below = render_data_plot([FOOTPRINT, POINTS], legend_position="below")
    assert below.figure.layout.legend.orientation == "h"
    with pytest.raises(ValueError, match="legend_position"):
        render_data_plot([POINTS], legend_position="left")


def test_axis_limits():
    res = render_data_plot([POINTS], xlim=[0, 500], ylim=[100, 300])
    assert list(res.figure.layout.xaxis.range) == [0, 500]
    assert list(res.figure.layout.yaxis.range) == [100, 300]
    deep = render_data_plot([POINTS], ylim=[0, 20], depth_axis=True)
    assert list(deep.figure.layout.yaxis.range) == [20, 0]
    with pytest.raises(ValueError, match="xlim"):
        render_data_plot([POINTS], xlim=[5, 5])


# ---------------------------------------------------------------------------
# Outlines, markers, names and hover
# ---------------------------------------------------------------------------

def test_a_polygon_is_closed_and_filled():
    res = render_data_plot([FOOTPRINT])
    tr = res.figure.data[0]
    assert (tr.x[0], tr.y[0]) == (tr.x[-1], tr.y[-1])
    assert tr.fill == "toself"
    assert tr.fillcolor.startswith("rgba(") and tr.fillcolor.endswith(",0.2)")
    assert res.series[0]["n"] == len(FOOTPRINT["x"]) + 1
    custom = render_data_plot([dict(FOOTPRINT, fill="#ff0000",
                                    fill_opacity=0.5)])
    assert custom.figure.data[0].fillcolor == "rgba(255,0,0,0.5)"


def test_outlines_are_drawn_under_the_points():
    res = render_data_plot([POINTS, FOOTPRINT])
    assert [t.name for t in res.figure.data] == ["Footprint", "Borings"]


def test_fill_on_a_plain_line_is_a_warning_not_a_crash():
    res = render_data_plot([{"x": [1, 2], "y": [1, 2], "fill": True}])
    assert any("'fill' applies to style 'polygon'" in w
               for w in res.warnings)


def test_marker_shapes_outlines_and_hollow_markers():
    res = render_data_plot([
        {"x": [1, 2], "y": [1, 2], "style": "markers", "marker": "diamond",
         "marker_edge": "black", "marker_size": 7},
        {"x": [1, 2], "y": [2, 3], "style": "markers", "marker": "square",
         "marker_open": True}])
    a, b = res.figure.data
    assert a.marker.symbol == "diamond"
    assert a.marker.line.color == "black"
    assert a.marker.size == pytest.approx(14.0)
    assert b.marker.symbol == "square-open"
    with pytest.raises(ValueError, match="marker 'blob'"):
        render_data_plot([{"x": [1], "y": [1], "marker": "blob"}])
    with pytest.raises(ValueError, match="line_dash"):
        render_data_plot([{"x": [1], "y": [1], "line_dash": "wavy"}])


def test_points_carry_their_names_and_hover_text():
    res = render_data_plot([FOOTPRINT, POINTS], equal_aspect=True)
    pts = [t for t in res.figure.data if t.name == "Borings"][0]
    assert "text" in pts.mode
    assert list(pts.text) == ["B-1", "B-2", "B-3", "B-4"]
    # the twin puts each name where the PNG did (one spot per point)
    assert len(pts.textposition) == 4
    assert list(pts.hovertext)[0] == "B-1<br>rock 12.0 m"
    assert pts.hoverinfo == "text"
    named = [s for s in res.series if s["label"] == "Borings"][0]
    assert named["points_named"] == 4


def test_crowded_names_move_apart():
    """Two points almost on top of each other: their names take different
    spots instead of printing over each other."""
    res = render_data_plot([{"x": [0.0, 0.4, 50.0], "y": [0.0, 0.0, 10.0],
                             "style": "markers",
                             "labels": ["LEFT-ONE", "RIGHT-ONE", "far"]}])
    spots = list(res.figure.data[0].textposition)
    assert spots[0] != spots[1]


def test_names_must_match_the_points():
    with pytest.raises(ValueError, match="'labels'"):
        render_data_plot([{"x": [1, 2], "y": [1, 2], "labels": ["a"]}])
    with pytest.raises(ValueError, match="'hover'"):
        render_data_plot([{"x": [1, 2], "y": [1, 2], "hover": "one"}])


def test_dropped_points_drop_their_names_too():
    res = render_data_plot([{"x": [1, float("nan"), 3], "y": [1, 2, 3],
                             "style": "markers", "labels": ["a", "b", "c"]}])
    assert list(res.figure.data[0].text) == ["a", "c"]


def test_a_series_can_stay_out_of_the_legend():
    res = render_data_plot([POINTS, dict(FOOTPRINT, in_legend=False)])
    by_name = {t.name: t.showlegend for t in res.figure.data}
    assert by_name["Footprint"] is False
    assert res.figure.layout.showlegend is False     # one series left


# ---------------------------------------------------------------------------
# Cases on a case axis
# ---------------------------------------------------------------------------

def test_named_cases_go_along_x_in_order():
    cases = ["Exp 15 m", "Exp 30 m", "Sph 30 m"]
    res = render_data_plot([
        {"x": cases, "y": [0.4, 0.5, 0.45], "style": "markers",
         "label": "Upper"},
        {"x": ["Exp 30 m", "Gau 30 m"], "y": [0.1, 0.2], "style": "markers",
         "label": "Lower"}])
    fig = res.figure
    assert fig.layout.xaxis.type == "category"
    assert list(fig.layout.xaxis.categoryarray) == cases + ["Gau 30 m"]
    assert res.series[0]["x_categories"] == cases
    assert "x_range" not in res.series[0]


def test_cases_and_numbers_do_not_mix():
    with pytest.raises(ValueError, match="names \\(cases\\)"):
        render_data_plot([{"x": ["a", "b"], "y": [1, 2]},
                          {"x": [1, 2], "y": [1, 2]}])


def test_numbers_written_as_text_are_still_numbers():
    res = render_data_plot([{"x": ["1", "2.5"], "y": [1, 2]}])
    assert res.figure.layout.xaxis.type is None
    assert res.series[0]["x_range"] == [1.0, 2.5]


# ---------------------------------------------------------------------------
# Readable wherever it is shown
# ---------------------------------------------------------------------------

def test_the_interactive_chart_brings_its_own_colours():
    """Dark text on white paper, set on the figure itself, so a page theme
    cannot leave the title or the axis labels unreadable."""
    lay = render_data_plot([POINTS], title="T", xlabel="x (m)").figure.layout
    assert lay.paper_bgcolor == "white" and lay.plot_bgcolor == "white"
    assert lay.font.color == "#222222"
    assert lay.title.font.color == "#222222"
    assert lay.xaxis.tickfont.color == "#222222"


# ---------------------------------------------------------------------------
# A plan drawn as a drawing (owner, 2026-10-15: "There isn't much use to the
# grid lines or the multiple axis number... a typical plot with a scale bar
# ... a property line and north arrow any time we do a plan view")
# ---------------------------------------------------------------------------

PROPERTY = {"x": [20, 255, 255, 20], "y": [25, 25, 175, 175],
            "label": "Property line", "style": "polygon",
            "line_dash": "dashdot"}


def test_plan_view_is_a_drawing_not_a_chart():
    res = render_data_plot([PROPERTY, FOOTPRINT, POINTS], plan_view=True,
                           north_deg=20, xlabel="Local site grid, metres",
                           title="Plan")
    assert res.plan_view and res.equal_aspect
    assert not res.warnings
    lay = res.figure.layout
    assert lay.xaxis.visible is False and lay.yaxis.visible is False
    assert lay.yaxis.scaleanchor == "x"
    texts = [a.text for a in lay.annotations]
    assert "N" in texts                                   # the north arrow
    assert any(t.endswith(" m") for t in texts)           # the scale bar
    assert "Local site grid, metres" in texts             # its caption
    assert len(lay.shapes) == 4                           # the bar's segments
    assert "plan view" in res.summary()
    # drawing-sheet practice: the title sits at the left
    assert lay.title.x == 0.0 and lay.title.xanchor == "left"


def test_the_scale_bar_names_its_ends_only():
    """'0' and the length with its unit: a middle number crowded the end
    label on a short bar ('1020 m', live wave w7b)."""
    lay = render_data_plot([FOOTPRINT, POINTS], plan_view=True,
                           north_deg=0).figure.layout
    bar_text = [a.text for a in lay.annotations
                if not a.showarrow and a.text not in ("N",)]
    assert bar_text == ["0", "100 m"]


def test_the_north_arrow_points_where_it_is_told():
    import math
    for deg in (0, 20, 90):
        lay = render_data_plot([FOOTPRINT], plan_view=True,
                               north_deg=deg).figure.layout
        arrow = [a for a in lay.annotations if a.showarrow][0]
        # Plotly: the tail sits (ax, ay) px from the head, y downward
        tx, ty = -arrow.ax, arrow.ay            # head minus tail, y upward
        assert math.degrees(math.atan2(tx, ty)) % 360 == pytest.approx(
            deg % 360, abs=1e-6)


def test_no_north_given_drawn_up_and_said():
    """Owner, 2026-10-15: follow the plan the data came from (or the
    project's other plans); if none shows north, default to up."""
    res = render_data_plot([FOOTPRINT], plan_view=True)
    arrow = [a for a in res.figure.layout.annotations if a.showarrow]
    assert len(arrow) == 1
    assert arrow[0].ax == pytest.approx(0.0, abs=1e-9)    # straight up
    assert arrow[0].ay > 0
    assert any("North drawn straight up" in w for w in res.warnings)


def test_the_scale_bar_is_a_round_length():
    from profile_figure.data_plot import _nice_length
    # the largest 1, 2 or 5 x 10^k within a third of the width
    assert _nice_length(235.0) == 50.0
    assert _nice_length(41.0) == 10.0
    assert _nice_length(1200.0) == 200.0
    assert _nice_length(3.0) == 1.0
    assert _nice_length(1.4) == 0.2


def test_the_furniture_stays_off_the_site():
    """The scale bar sits in a margin below everything drawn."""
    res = render_data_plot([PROPERTY, FOOTPRINT], plan_view=True,
                           north_deg=0)
    bar_tops = [max(s.y0, s.y1) for s in res.figure.layout.shapes]
    assert max(bar_tops) < min(PROPERTY["y"])


def test_a_zone_can_be_hatched():
    zone = dict(FOOTPRINT, label="Zone", hatch="/", color="#c0504d",
                fill=False)
    res = render_data_plot([zone], plan_view=True, north_deg=0)
    tr = res.figure.data[0]
    assert tr.fill == "toself" and tr.fillpattern.shape == "/"
    assert tr.fillcolor == "rgba(0,0,0,0)"          # hatch only, no tint
    with pytest.raises(ValueError, match="hatch"):
        render_data_plot([dict(zone, hatch="zigzag")])
    res = render_data_plot([{"x": [1, 2], "y": [1, 2], "hatch": "/"}])
    assert any("'hatch' applies to style 'polygon'" in w
               for w in res.warnings)


def test_plan_view_refused_where_it_makes_no_sense():
    res = render_data_plot([{"x": ["a", "b"], "y": [1, 2]}], plan_view=True)
    assert res.plan_view is False
    assert any("plan_view ignored" in w for w in res.warnings)
