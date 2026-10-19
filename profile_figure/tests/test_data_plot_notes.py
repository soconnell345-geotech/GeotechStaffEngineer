"""plot_data places free text: notes (``annotations``), BACKLOG GM-109.

Live wave w9 (``module_work/live_smoke/runs/w9-figures/TRACE_REVIEW.md``
§2.7, fix 10): ``plot_data`` had no free text, so the figure helper carried
a plasticity chart's zone names on series of invisible markers (white
gaps, stray dots), drew it twice and moved to ``plot_code``. A note is text
at a point, in data units or in fractions of the plot, with an optional
leader arrow to the point it marks, an alignment and a plain box, drawn
the same in the PNG and in its interactive twin. Notes stay where they are
put; the layout check reports one that overlaps other lettering, that a
line runs through or that stands outside the plot.
"""

import io
import math

import pytest

from profile_figure import render_data_plot

pytest.importorskip("plotly")

CURVE = {"x": [0, 10, 20, 30, 40], "y": [0, 12, 18, 21, 22],
         "label": "Load test", "style": "both"}
RED, BLUE = "#ff0000", "#0000ff"


# ---------------------------------------------------------------------------
# What the PNG holds: the matplotlib figure as render_data_plot closes it,
# mapped to the saved PNG's own pixels (bbox_inches="tight", 150 dpi)
# ---------------------------------------------------------------------------

@pytest.fixture
def drawn(monkeypatch):
    """Spy on the figure as it is closed: each note's text artist, the
    axes' view and a function giving where a display point lies in the
    saved PNG."""
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib.text import Annotation

    seen = {}
    real_close = plt.close

    def spy(fig=None):
        if fig is not None and getattr(fig, "axes", None):
            ax = fig.axes[0]
            fig.canvas.draw()       # savefig left the boxes at its own dpi
            r = fig.canvas.get_renderer()
            tight = fig.get_tightbbox(r)                     # inches
            pad = float(matplotlib.rcParams["savefig.pad_inches"])
            dpi = fig.dpi

            def to_png(xy, save_dpi=150):
                x_in, y_in = xy[0] / dpi, xy[1] / dpi
                return ((x_in - (tight.x0 - pad)) * save_dpi,
                        ((tight.y1 + pad) - y_in) * save_dpi)

            seen.update(
                ax=ax, renderer=r, to_png=to_png,
                view=(tuple(ax.get_xlim()), tuple(ax.get_ylim())),
                notes={t.get_text(): t for t in ax.texts
                       if isinstance(t, Annotation)})
        return real_close(fig)

    monkeypatch.setattr(plt, "close", spy)
    return seen


def _image(png: bytes):
    from PIL import Image
    return Image.open(io.BytesIO(png)).convert("RGB")


def _share(img, centre, colour_test, half=4) -> float:
    """The share of pixels in a (2 half + 1)^2 window that pass."""
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    hits = total = 0
    for x in range(cx - half, cx + half + 1):
        for y in range(cy - half, cy + half + 1):
            if 0 <= x < img.width and 0 <= y < img.height:
                total += 1
                hits += colour_test(img.getpixel((x, y)))
    return hits / max(total, 1)


def _red(p):
    return p[0] > 200 and p[1] < 80 and p[2] < 80


def _blue(p):
    return p[2] > 150 and p[0] < 110 and p[1] < 110


def _box_centre(seen, text):
    from profile_figure.data_plot import _note_box
    bb = _note_box(seen["notes"][text], seen["renderer"])
    return ((bb.x0 + bb.x1) / 2.0, (bb.y0 + bb.y1) / 2.0), bb


def _plotly_note(res, text):
    hits = [a for a in res.figure.layout.annotations if a.text == text]
    assert len(hits) == 1, [a.text for a in res.figure.layout.annotations]
    return hits[0]


# ---------------------------------------------------------------------------
# Where a note stands, in both renderings
# ---------------------------------------------------------------------------

def test_a_note_in_data_units_stands_where_it_was_put(drawn):
    res = render_data_plot([CURVE], annotations=[
        {"text": "Plunging failure", "x": 30, "y": 8, "align": "left",
         "box": RED, "color": "white"}])
    assert not res.warnings
    # PNG: the text's left end at (30, 8), its box red where it was put
    t = drawn["notes"]["Plunging failure"]
    assert t.xycoords == "data" and t.xy == (30.0, 8.0)
    assert (t.get_ha(), t.get_va()) == ("left", "center")
    px, py = drawn["ax"].transData.transform((30.0, 8.0))
    (cx, cy), bb = _box_centre(drawn, "Plunging failure")
    assert cy == pytest.approx(py, abs=1.0)
    assert bb.x0 < px < bb.x0 + 12               # the box's pad, then text
    assert _share(_image(res.png_bytes), drawn["to_png"]((cx, cy)),
                  _red) > 0.5
    # the twin: the same place, the same side, the same box
    a = _plotly_note(res, "Plunging failure")
    assert (a.xref, a.yref, a.x, a.y) == ("x", "y", 30, 8)
    assert (a.xanchor, a.yanchor, a.showarrow) == ("left", "middle", False)
    assert a.bgcolor == "rgba(255,0,0,1)" and a.font.color == "white"


def test_a_note_in_fractions_of_the_plot(drawn):
    res = render_data_plot([CURVE], annotations=[
        {"text": "Groundwater not met", "x": 0.97, "y": 0.05,
         "coords": "axes", "align": "bottom right", "box": RED}])
    assert not res.warnings
    t = drawn["notes"]["Groundwater not met"]
    assert t.xycoords == "axes fraction"
    px, py = drawn["ax"].transAxes.transform((0.97, 0.05))
    (cx, cy), bb = _box_centre(drawn, "Groundwater not met")
    assert bb.x1 == pytest.approx(px, abs=4.0)           # its right end
    assert bb.y0 == pytest.approx(py, abs=4.0)           # its foot
    assert _share(_image(res.png_bytes), drawn["to_png"]((cx, cy)),
                  _red) > 0.4
    a = _plotly_note(res, "Groundwater not met")
    assert (a.xref, a.yref) == ("x domain", "y domain")
    assert (a.x, a.y, a.xanchor, a.yanchor) == (0.97, 0.05, "right",
                                                "bottom")


def test_a_leader_runs_from_the_text_to_the_point(drawn):
    res = render_data_plot([CURVE], annotations=[
        {"text": "Yield", "x": 12, "y": 21, "point": [20, 18],
         "color": BLUE}])
    assert not res.warnings
    t = drawn["notes"]["Yield"]
    assert t.xy == (20.0, 18.0) and t.xyann == (12.0, 21.0)
    assert t.arrow_patch is not None
    # the arrowhead is drawn at the point, in the note's colour
    head = drawn["ax"].transData.transform((20.0, 18.0))
    tail = drawn["ax"].transData.transform((12.0, 21.0))
    # a few pixels back from the point along the leader (shrinkB, then the
    # head itself)
    d = math.hypot(head[0] - tail[0], head[1] - tail[1])
    back = (head[0] + (tail[0] - head[0]) * 5 / d,
            head[1] + (tail[1] - head[1]) * 5 / d)
    assert _share(_image(res.png_bytes), drawn["to_png"](back), _blue,
                  half=5) > 0.05
    # the twin: the head at the point, the tail (the text) where it was put
    a = _plotly_note(res, "Yield")
    assert a.showarrow is True and a.arrowcolor == BLUE
    assert (a.xref, a.yref, a.x, a.y) == ("x", "y", 20, 18)
    assert (a.axref, a.ayref, a.ax, a.ay) == ("x", "y", 12, 21)


def test_a_note_in_fractions_with_a_leader_to_a_data_point(drawn):
    """Plotly draws a leader's tail in its head's units, so the note goes in
    data units for the PNG's view, and the twin is drawn over that view."""
    res = render_data_plot([CURVE], annotations=[
        {"text": "Peak", "x": 0.1, "y": 0.9, "coords": "axes",
         "align": "top left", "point": [40, 22]}])
    t = drawn["notes"]["Peak"]
    assert t.anncoords == "axes fraction" and t.xy == (40.0, 22.0)
    (x0, x1), (y0, y1) = drawn["view"]
    a = _plotly_note(res, "Peak")
    assert (a.x, a.y) == (40, 22)
    assert a.ax == pytest.approx(x0 + 0.1 * (x1 - x0))
    assert a.ay == pytest.approx(y0 + 0.9 * (y1 - y0))
    lay = res.figure.layout
    assert list(lay.xaxis.range) == pytest.approx([x0, x1])
    assert list(lay.yaxis.range) == pytest.approx([y0, y1])
    assert lay.xaxis.autorange is False


@pytest.mark.parametrize("align, ha, va, xanchor, yanchor", [
    (None, "center", "center", "center", "middle"),
    ("right", "right", "center", "right", "middle"),
    ("top", "center", "top", "center", "top"),
    ("Bottom-Left", "left", "bottom", "left", "bottom"),
    ("middle right", "right", "center", "right", "middle"),
])
def test_alignment_is_the_same_in_both(drawn, align, ha, va, xanchor,
                                       yanchor):
    note = {"text": "q", "x": 5, "y": 5}
    if align:
        note["align"] = align
    res = render_data_plot([CURVE], annotations=[note])
    t = drawn["notes"]["q"]
    assert (t.get_ha(), t.get_va()) == (ha, va)
    a = _plotly_note(res, "q")
    assert (a.xanchor, a.yanchor) == (xanchor, yanchor)


def test_notes_on_a_log_axis_and_a_case_axis(drawn):
    gsd = {"x": [50, 4.75, 0.425, 0.075], "y": [100, 80, 40, 12],
           "label": "B-1"}
    res = render_data_plot([gsd], logx=True, xlim=[100, 0.01],
                           annotations=[{"text": "Sand", "x": 1.0, "y": 60,
                                         "point": [4.75, 80]}])
    a = _plotly_note(res, "Sand")
    # Plotly takes positions on a log axis as their log10
    assert a.ax == pytest.approx(0.0) and a.x == pytest.approx(
        math.log10(4.75))
    assert a.ay == 60 and a.y == 80
    assert drawn["notes"]["Sand"].xyann == (1.0, 60.0)

    cases = render_data_plot(
        [{"x": ["Exp 15 m", "Exp 30 m", "Sph 30 m"], "y": [0.4, 0.5, 0.45],
          "style": "markers"}],
        annotations=[{"text": "chosen", "x": "Exp 30 m", "y": 0.52,
                      "align": "bottom"}])
    a = _plotly_note(cases, "chosen")
    assert (a.x, a.y) == (1, 0.52)               # the case's place on x
    assert cases.annotations[0]["at"] == ["Exp 30 m", 0.52]


def test_line_breaks_in_either_spelling(drawn):
    res = render_data_plot([CURVE], annotations=[
        {"text": "one\ntwo", "x": 5, "y": 15},
        {"text": "three<br>four", "x": 25, "y": 5}])
    assert "one<br>two" in [a.text for a in res.figure.layout.annotations]
    assert "three<br>four" in [a.text for a in res.figure.layout.annotations]
    assert "three\nfour" in drawn["notes"]


def test_a_plan_view_carries_a_note(drawn):
    site = {"x": [0, 120, 120, 0], "y": [0, 0, 80, 80], "style": "polygon",
            "label": "Property line", "line_dash": "dashdot"}
    borings = {"x": [30, 90], "y": [20, 60], "style": "markers",
               "label": "Borings", "labels": ["B-1", "B-2"]}
    res = render_data_plot(
        [site, borings], plan_view=True, north_deg=0,
        xlabel="Local site grid, metres",
        annotations=[{"text": "Existing building", "x": 60, "y": 40,
                      "box": RED, "color": "white"},
                     {"text": "Refusal at 4 m", "x": 100, "y": 20,
                      "point": [90, 60], "align": "top"}])
    assert res.plan_view and not res.warnings
    texts = [a.text for a in res.figure.layout.annotations]
    assert "N" in texts and "Local site grid, metres" in texts   # furniture
    a = _plotly_note(res, "Existing building")
    assert (a.xref, a.x, a.y) == ("x", 60, 40)
    b = _plotly_note(res, "Refusal at 4 m")
    assert b.showarrow and (b.x, b.y, b.ax, b.ay) == (90, 60, 100, 20)
    (cx, cy), _bb = _box_centre(drawn, "Existing building")
    px, py = drawn["ax"].transData.transform((60.0, 40.0))
    assert (cx, cy) == pytest.approx((px, py), abs=1.5)
    assert _share(_image(res.png_bytes), drawn["to_png"]((cx, cy)),
                  _red) > 0.4


def test_the_result_names_its_notes():
    res = render_data_plot([CURVE], annotations=[
        {"text": "a", "x": 5, "y": 15}, {"text": "b", "x": 25, "y": 5,
                                         "point": [30, 21]}])
    assert "2 notes" in res.summary()
    d = res.to_dict()
    assert d["annotations"] == [
        {"text": "a", "at": [5, 15], "coords": "data"},
        {"text": "b", "at": [25, 5], "coords": "data", "point": [30, 21]}]
    assert render_data_plot([CURVE]).to_dict()["annotations"] == []


# ---------------------------------------------------------------------------
# Bad input refused, saying what is wrong
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("note, words", [
    ({"txt": "a", "x": 1, "y": 1}, r"unknown key\(s\) \['txt'\]\. Did you "
                                   r"mean: txt -> text"),
    ({"text": "a", "x": "left", "y": 1}, "x must be a number"),
    ({"text": "a", "x": 1, "y": None}, "needs 'y'"),
    ({"text": "a", "x": 1, "y": float("nan")}, "finite"),
    ({"text": "a", "x": True, "y": 1}, "x must be a number"),
    ({"x": 1, "y": 1}, "needs 'text'"),
    ({"text": "  ", "x": 1, "y": 1}, "needs 'text'"),
    ({"text": "a", "x": 1, "y": 1, "coords": "paper"}, "coords 'paper'"),
    ({"text": "a", "x": 1, "y": 1, "align": "upper"}, "align 'upper'"),
    ({"text": "a", "x": 1, "y": 1, "align": "left right"}, "align"),
    ({"text": "a", "x": 1, "y": 1, "point": [3]}, r"point must be \[x, y\]"),
    ({"text": "a", "x": 1, "y": 1, "point": [3, "top"]},
     "point y must be a number"),
    ({"text": "a", "x": 1, "y": 1, "color": "blurple"}, "not a colour"),
    ({"text": "a", "x": 1, "y": 1, "box": 3}, "not a colour"),
    ({"text": "a", "x": 1, "y": 1, "font_size": "big"}, "font_size"),
    ({"text": "a", "x": 1, "y": 1, "font_size": float("nan")}, "font_size"),
    ("a note", "must be a dict"),
])
def test_bad_notes_are_refused(note, words):
    with pytest.raises(ValueError, match=words):
        render_data_plot([CURVE], annotations=[note])


def test_a_note_must_suit_the_axis():
    log_curve = {"x": [1, 10, 100], "y": [1, 10, 100]}
    with pytest.raises(ValueError, match="above zero on a log axis"):
        render_data_plot([log_curve], logy=True, annotations=[
            {"text": "a", "x": 5, "y": 0}])
    with pytest.raises(ValueError, match="not one of the cases"):
        render_data_plot([{"x": ["A", "B"], "y": [1, 2]}], annotations=[
            {"text": "a", "x": "C", "y": 1}])
    with pytest.raises(ValueError, match="list of"):
        render_data_plot([CURVE], annotations={"text": "a", "x": 1, "y": 1})
    # fractions of the plot are not data: no log rule for them
    ok = render_data_plot([log_curve], logy=True, annotations=[
        {"text": "a", "x": 0.5, "y": 0.0, "coords": "axes",
         "align": "bottom"}])
    assert ok.annotations[0]["coords"] == "axes"


# ---------------------------------------------------------------------------
# Notes stay put; names move round them; the layout check reports them
# ---------------------------------------------------------------------------

def test_point_names_keep_clear_of_a_note(drawn):
    from profile_figure.data_plot import _note_box, _overlap
    pts = {"x": [10, 30], "y": [10, 20], "style": "markers",
           "labels": ["P-1", "P-2"]}
    lims = {"xlim": [0, 40], "ylim": [0, 30]}
    alone = render_data_plot([pts], **lims).figure.data[0].textposition
    # a note over the spot P-1's name took, and the spots either side of
    # it: the name goes elsewhere, clear of the note; P-2 does not move
    res = render_data_plot([pts], **lims, annotations=[
        {"text": "A WIDE NOTE ACROSS THE TOP", "x": 10, "y": 10.7,
         "align": "bottom", "box": True}])
    spots = res.figure.data[0].textposition
    assert spots[0] != alone[0] and "top" not in spots[0]
    assert spots[1] == alone[1]
    r = drawn["renderer"]
    note = _note_box(drawn["notes"]["A WIDE NOTE ACROSS THE TOP"], r)
    name = drawn["notes"]["P-1"].get_window_extent(r)
    assert _overlap(note, name) == 0


def test_the_layout_check_reports_notes():
    """By the pixels drawn, with the plotting box's own check (review of
    GM-109: a box check made 21 false reports in 236 one-note figures)."""
    # two notes on top of each other
    res = render_data_plot([CURVE], annotations=[
        {"text": "First note", "x": 30, "y": 6},
        {"text": "Second note", "x": 30.5, "y": 6.3}])
    assert any(w.startswith("annotations[0] 'First note': overlaps "
                            "annotations[1]") for w in res.warnings)
    # a series through a note, named; with its own box the line is hidden
    on_line = {"text": "On the curve", "x": 20, "y": 18}
    res = render_data_plot([CURVE], annotations=[on_line])
    assert any("crossed by 'Load test' (move it, or give it box: true)"
               in w for w in res.warnings), res.warnings
    boxed = render_data_plot([CURVE], annotations=[dict(on_line, box=True)])
    assert not boxed.warnings, boxed.warnings
    # a reference line through a note
    res = render_data_plot([CURVE], hlines=[{"value": 6}], annotations=[
        {"text": "Near the limit", "x": 30, "y": 6}])
    assert any("crossed by the reference line at 6" in w
               for w in res.warnings), res.warnings
    # another note's leader through a note, named by its note
    res = render_data_plot([CURVE], xlim=[0, 40], ylim=[0, 30], annotations=[
        {"text": "Crossed", "x": 25, "y": 27},
        {"text": "Lead", "x": 3, "y": 27, "point": [38, 27],
         "align": "left"}])
    assert any(w.startswith("annotations[0] 'Crossed': crossed by the "
                            "leader of annotations[1]")
               for w in res.warnings), res.warnings
    assert not any("'Lead'" in w for w in res.warnings), res.warnings
    # over the title
    res = render_data_plot([CURVE], title="Load test", annotations=[
        {"text": "Over the title", "x": 0.5, "y": 1.06, "coords": "axes"}])
    assert any("overlaps the title" in w for w in res.warnings)
    # outside the plot; a leader to a point outside it
    res = render_data_plot([CURVE], xlim=[0, 40], annotations=[
        {"text": "Far", "x": 80, "y": 10},
        {"text": "Off", "x": 20, "y": 5, "point": [60, 10]}])
    far = [w for w in res.warnings if "'Far'" in w]
    assert far and "outside the plot" in far[0] and "xlim" in far[0]
    assert any("'Off'" in w and "the point it marks is outside the plot"
               in w for w in res.warnings)
    # a note placed clear of everything: nothing to say
    clean = render_data_plot([CURVE], annotations=[
        {"text": "Clear", "x": 30, "y": 6}])
    assert not clean.warnings


def test_no_false_reports_where_the_box_check_made_them():
    """The review's false reports: a note beside rotated case names, near
    the legend's padding, a two-line note over a line through its empty
    band, a boxed note under the title."""
    cases = {"x": ["Exponential 15 m", "Exponential 30 m",
                   "Spherical 30 m", "Gaussian 45 m"],
             "y": [0.42, 0.51, 0.47, 0.44], "style": "markers",
             "label": "P(exceed)"}
    res = render_data_plot([cases], legend=True, annotations=[
        {"text": "chosen", "x": "Exponential 30 m", "y": 0.505,
         "align": "left"}])
    assert not res.warnings, res.warnings
    two = render_data_plot([CURVE], hlines=[{"value": 15}], annotations=[
        {"text": "upper line\n\nlower line", "x": 30, "y": 15}])
    assert not two.warnings, two.warnings
    legend = render_data_plot(
        [CURVE, dict(CURVE, y=[1, 2, 3, 4, 5], label="Second")],
        legend=True, legend_position="inside", annotations=[
            {"text": "a note", "x": 0.5, "y": 0.5, "coords": "axes"}])
    assert not legend.warnings, legend.warnings
    titled = render_data_plot([CURVE], title="Load test", annotations=[
        {"text": "Under the title", "x": 0.5, "y": 0.97, "coords": "axes",
         "align": "top", "box": True}])
    assert not titled.warnings, titled.warnings


def test_the_check_leaves_the_picture_as_it_was(monkeypatch):
    import profile_figure.data_plot as dp
    notes = [{"text": "Yield", "x": 4, "y": 20, "point": [10, 12]},
             {"text": "Boxed", "x": 30, "y": 6, "box": True}]
    checked = dp.render_data_plot([CURVE], annotations=notes)
    monkeypatch.setattr(dp, "_check_lettering", lambda *a, **k: [])
    unchecked = dp.render_data_plot([CURVE], annotations=notes)
    assert checked.png_bytes == unchecked.png_bytes


def test_a_font_size_out_of_range_is_drawn_at_the_end_and_said():
    res = render_data_plot([CURVE], annotations=[
        {"text": "huge", "x": 0.5, "y": 0.5, "coords": "axes",
         "font_size": 40},
        {"text": "tiny", "x": 0.2, "y": 0.8, "coords": "axes",
         "font_size": 2}])
    assert ("annotations[0] font_size 40 pt drawn at 20 pt (a note is 6 to "
            "20 pt)") in res.warnings
    assert any(w.startswith("annotations[1] font_size 2 pt drawn at 6 pt")
               for w in res.warnings)
    ok = render_data_plot([CURVE], annotations=[
        {"text": "fine", "x": 0.5, "y": 0.5, "coords": "axes",
         "font_size": 12}])
    assert not ok.warnings


def test_a_failed_check_never_costs_the_figure(monkeypatch):
    import profile_figure.data_plot as dp

    def broken(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(dp, "_check_lettering", broken)
    res = dp.render_data_plot([CURVE], annotations=[
        {"text": "a", "x": 5, "y": 15}])
    assert res.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert any("not checked" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# Live wave w9's figures that went to plot_code for their text, drawn by
# plot_data (the data and the words of the briefs; no carrier series)
# ---------------------------------------------------------------------------

def _no_carriers(series):
    """No series is there only to carry text (w9: 0- or 1-point markers in
    white or black, named on the figure)."""
    for s in series:
        assert not (s.get("labels") and (s.get("marker_size") or 9) <= 1)


def test_w9_plasticity_chart_draws_its_zone_names_with_notes(drawn):
    """F57: the zone names were carried on invisible markers (two drafts),
    then the chart went to plot_code. With notes it is one plot_data call:
    zone and line names where they belong, a leader to the CL-ML band and
    to the sample whose name has no room beside it."""
    series = [
        {"x": [4, 25.5, 80], "y": [4, 4, 43.8], "label": "A-line",
         "style": "line", "color": "black", "line_width": 1.8,
         "in_legend": False},
        {"x": [16, 16, 74.667], "y": [0, 7.2, 60], "label": "U-line",
         "style": "line", "color": "dimgray", "line_dash": "dash",
         "in_legend": False},
        {"x": [16, 25.48, 29.59, 16], "y": [4, 4, 7, 7], "label": "CL-ML",
         "style": "polygon", "color": "gray", "fill": True,
         "fill_opacity": 0.25, "hatch": "/", "in_legend": False},
        {"x": [38, 55], "y": [18, 31], "label": "B-1", "style": "markers",
         "marker": "circle", "marker_edge": "black",
         "labels": ["B-1 S-4", "B-1 S-6"]},
        {"x": [46], "y": [14], "label": "B-2", "style": "markers",
         "marker": "square", "marker_edge": "black",
         "labels": ["B-2 S-3"]},
        {"x": [29, 62], "y": [9, 38], "label": "B-3", "style": "markers",
         "marker": "triangle", "marker_edge": "black",
         "labels": ["B-3 S-2", "B-3 S-5"]},
        {"x": [24], "y": [5], "label": "B-4", "style": "markers",
         "marker": "diamond", "marker_edge": "black"},
    ]
    notes = [
        {"text": "CL or OL", "x": 30, "y": 40},
        {"text": "CH or OH", "x": 70, "y": 46},
        {"text": "ML or OL", "x": 40, "y": 5},
        {"text": "MH or OH", "x": 68, "y": 18},
        {"text": "CL-ML", "x": 8, "y": 14, "point": [18, 5.5]},
        {"text": "A-line", "x": 78, "y": 39, "align": "top"},
        {"text": "U-line", "x": 66, "y": 54, "align": "right"},
        {"text": "B-4 S-1", "x": 34, "y": 1.5, "point": [24, 5],
         "align": "left"},
    ]
    _no_carriers(series)
    res = render_data_plot(series, annotations=notes,
                           vlines=[{"value": 50, "label": "LL = 50"}],
                           xlim=[0, 80], ylim=[0, 60], equal_aspect=True,
                           xlabel="Liquid limit, LL (%)",
                           ylabel="Plasticity index, PI (%)",
                           title="Plasticity chart", legend=True)
    # The LL = 50 line is an obstacle for point names, as the data are
    # (BACKLOG GM-146: it ran through two names here and no warning said
    # so). B-1 S-6 now stands clear of it; B-2 S-3 has no clear spot (the
    # line 12 px to the right of its point, the A-line to its left), so it
    # takes the thin dotted line and the check says so.
    assert res.warnings == [
        "series[4] point name 'B-2 S-3': crossed by the reference line at "
        "50 (no spot beside the point was clear: name it with a note and a "
        "leader, or leave the name out)"], res.warnings
    assert "8 notes" in res.summary()
    for n in notes:
        a = _plotly_note(res, n["text"])
        t = drawn["notes"][n["text"]]
        if "point" in n:
            assert (a.x, a.y) == tuple(n["point"])
            assert (a.ax, a.ay) == (n["x"], n["y"])
            assert t.xy == tuple(float(v) for v in n["point"])
            assert t.xyann == (float(n["x"]), float(n["y"]))
        else:
            assert (a.x, a.y) == (n["x"], n["y"]) and not a.showarrow
            assert t.xy == (float(n["x"]), float(n["y"]))
    # every sample is named or marked; no names in the legend's place
    assert [s["label"] for s in res.series][3:] == ["B-1", "B-2", "B-3",
                                                     "B-4"]


def test_w9_mohr_circles_draw_their_labels_with_notes(drawn):
    """F59: the circle labels and c' carried leaders, and the fitted values
    a text box: plot_code. With notes: true semicircles as series, leaders
    to the circle tops and to the intercept, the fit in a plain box in the
    corner of the plot."""
    s3, s1 = [50, 100, 200], [165, 310, 590]
    series, tops = [], []
    for i, (a, b) in enumerate(zip(s3, s1)):
        p, q = (a + b) / 2, (b - a) / 2
        th = [math.pi * k / 90 for k in range(91)]
        series.append({"x": [p + q * math.cos(t) for t in th],
                       "y": [q * math.sin(t) for t in th],
                       "label": f"Specimen {i + 1}", "style": "line",
                       "in_legend": False})
        tops.append((p, q))
    c, tanphi = 7.43, math.tan(math.radians(28.53))
    series.append({"x": [0, 700], "y": [c, c + 700 * tanphi],
                   "label": "Failure envelope", "style": "line",
                   "color": "black", "line_dash": "dash",
                   "in_legend": False})
    series.append({"x": [t[0] for t in tops], "y": [t[1] for t in tops],
                   "label": "Circle tops", "style": "markers",
                   "color": "#444444", "in_legend": False})
    notes = [
        {"text": "c' = 7.4 kPa", "x": 40, "y": 70, "point": [0, c],
         "align": "bottom"},
        {"text": "1: σ'3 = 50 kPa\nσ'1 = 165 kPa", "x": 60, "y": 150,
         "point": list(tops[0]), "align": "bottom"},
        {"text": "2: σ'3 = 100 kPa\nσ'1 = 310 kPa", "x": 205, "y": 230,
         "point": list(tops[1]), "align": "bottom"},
        {"text": "3: σ'3 = 200 kPa\nσ'1 = 590 kPa", "x": 420, "y": 300,
         "point": list(tops[2]), "align": "bottom"},
        {"text": "Failure envelope: φ' = 28.5°, c' = 7.4 kPa", "x": 0.03,
         "y": 0.97, "coords": "axes", "align": "top left", "box": True},
    ]
    _no_carriers(series)
    res = render_data_plot(series, annotations=notes, xlim=[-30, 720],
                           ylim=[0, 400], equal_aspect=True,
                           xlabel="Effective normal stress, σ' (kPa)",
                           ylabel="Shear stress, τ (kPa)",
                           title="CU triaxial: Mohr circles at failure")
    assert not res.warnings, res.warnings
    for n in notes[1:4]:
        a = _plotly_note(res, n["text"].replace("\n", "<br>"))
        assert a.showarrow and (a.x, a.y) == tuple(n["point"])
        assert drawn["notes"][n["text"]].xy == tuple(n["point"])
    box = _plotly_note(res, notes[4]["text"])
    assert (box.xref, box.x, box.y) == ("x domain", 0.03, 0.97)
    assert box.bgcolor == "rgba(255,255,255,1)"
    t = drawn["notes"][notes[4]["text"]]
    assert t.get_bbox_patch() is not None
