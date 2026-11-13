"""plot_data's panels, sections to scale and contours (BACKLOG GM-315):
what only model-written plot code drew before. Made-up numbers only."""

import io
import json

import pytest

pytest.importorskip("matplotlib")

from aec_agent_core.charts.data_plot import render_data_plot  # noqa: E402
from aec_agent_core.charts.figure_examples import (  # noqa: E402
    CODE_EXAMPLES, example)
from aec_agent_core.charts.layout_ink import (  # noqa: E402
    LETTERING_KEY, MIN_PRINT_PT, layout_check, parse_lettering, png_texts)
from aec_agent_core.charts.plot_tool import run_plot_data  # noqa: E402

NEW_EXAMPLES = ("wall_section_pressures", "settlement_stress_panels",
                "pile_design_panels", "contour_top_of_rock")
EL = [10.0, 6.0, 2.0]


def _chart(title, xs, **more):
    return {"title": title, "xlabel": "Stress (kPa)",
            "series": [{"x": xs, "y": EL, "style": "line", "label": title}],
            **more}


def _draw(**params):
    return render_data_plot(params.pop("series", None), **params)


def _size(png):
    from PIL import Image
    with Image.open(io.BytesIO(png)) as im:
        return im.size


# --- panels sharing the vertical axis --------------------------------------

def test_panels_draw_side_by_side_on_one_vertical_axis():
    res = _draw(panels=[_chart("Initial", [0, 40, 80]),
                        _chart("Increase", [30, 20, 10])],
                title="Two made-up stress profiles", ylabel="Elevation (m)")
    assert [p["kind"] for p in res.panels] == ["chart", "chart"]
    assert [s["panel"] for s in res.series] == [0, 1]
    assert res.ylabel == "Elevation (m)" and not res.warnings
    assert "2 panels on one vertical axis" in res.summary()
    assert res.figure is not None                  # chart panels have a twin
    # the twin is one row of subplots sharing y
    lay = res.figure.layout
    assert lay.yaxis2.matches == "y" and lay.xaxis2.anchor == "y2"


def test_panels_holding_one_quantity_share_one_x_scale():
    res = _draw(panels=[_chart("Initial", [0, 40, 180], share_x="stress"),
                        _chart("Increase", [30, 20, 10], share_x="stress"),
                        _chart("Other", [1, 2, 3])])
    a, b, c = (p["x_range"] for p in res.panels)
    assert a == b and a[0] == 0 and a[1] >= 180
    assert c != a and c[1] < 10          # not in the group: its own scale


def test_relative_widths_and_a_line_across_every_panel():
    res = _draw(panels=[_chart("Narrow", [0, 1, 2], width=1),
                        _chart("Wide", [0, 1, 2], width=2)],
                hlines=[{"value": 5.0, "label": "Water table"}],
                footnote="Made-up numbers.<br>Second line.")
    narrow, wide = (p["width_in"] for p in res.panels)
    assert wide == pytest.approx(2 * narrow, rel=0.02)
    assert res.footnote == "Made-up numbers.\nSecond line."
    shapes = [s for s in res.figure.layout.shapes if s.type == "line"]
    assert len(shapes) == 2 and {s.y0 for s in shapes} == {5.0}


def test_depth_axis_runs_down_in_panels():
    res = _draw(panels=[_chart("A", [0, 1, 2])], depth_axis=True,
                ylabel="Depth (m)")
    assert res.depth_axis
    y0, y1 = res.figure.layout.yaxis.range
    assert y0 > y1


def test_a_chart_panel_starts_at_zero_unless_xlim_says():
    res = _draw(panels=[_chart("A", [20, 40, 80]),
                        _chart("B", [20, 40, 80], xlim=[10, 100])])
    assert res.panels[0]["x_range"][0] == 0
    assert res.panels[1]["x_range"] == [10, 100]


# --- a section to scale beside its diagrams --------------------------------

def _section(**more):
    return {"kind": "section", "title": "Section, to scale",
            "series": [{"x": [0, 4, 4, 0], "y": [0, 0, 8, 8],
                        "style": "polygon", "fill": True, "label": "Wall"}],
            "xlim": [-1, 5], **more}


def _diagram(**more):
    return {"title": "Pressure", "xlabel": "Pressure (kPa)",
            "series": [{"x": [0, 30], "y": [8, 0], "style": "line",
                        "fill_to_x": 0, "label": "Pressure"}], **more}


def test_a_section_is_drawn_one_unit_across_as_one_unit_up():
    res = _draw(panels=[_section(), _diagram()], ylim=[-1, 9])
    sec = res.panels[0]
    assert sec["kind"] == "section" and res.equal_aspect
    # 6 units across and 10 up: the panel is 0.6 as wide as the axes are
    # high, whatever the figure's size
    tall = _draw(panels=[_section(), _diagram()], ylim=[-1, 9],
                 height_in=4.0)
    assert tall.panels[0]["width_in"] != sec["width_in"]
    for r, h_in in ((res, None), (tall, 4.0)):
        w = r.panels[0]["width_in"]
        assert 0.3 < w / 0.6 < 6.0
    assert "drawn to scale (1:1)" in res.summary()


def test_section_scale_is_exact_on_the_png():
    """The filled square of the section measures as many pixels across as
    up."""
    np = pytest.importorskip("numpy")
    from PIL import Image
    res = _draw(panels=[{"kind": "section", "xlim": [-1, 5], "series": [
        {"x": [0, 4, 4, 0], "y": [2, 2, 6, 6], "style": "polygon",
         "color": "#ff0000", "fill": "#ff0000", "fill_opacity": 1.0,
         "label": "Square"}]}, _diagram()], ylim=[-1, 9], dpi=200)
    im = np.asarray(Image.open(io.BytesIO(res.png_bytes)).convert("RGB"))
    red = (im[..., 0] > 200) & (im[..., 1] < 60) & (im[..., 2] < 60)
    ys, xs = np.nonzero(red)
    assert xs.size
    wide, high = xs.max() - xs.min(), ys.max() - ys.min()
    assert abs(wide - high) <= 2, (wide, high)


def test_a_section_too_wide_for_its_height_keeps_its_scale():
    res = _draw(panels=[_section(xlim=[-10, 30]), _diagram()], ylim=[-1, 9],
                height_in=7.0)
    assert any("to scale" in w and "height_in" in w for w in res.warnings)
    # the diagram keeps a readable width
    assert res.panels[1]["width_in"] >= 0.75


def test_fill_bands_dimension_and_arrow_are_drawn(tmp_path):
    plain = _draw(panels=[_section(), _diagram()], ylim=[-2, 9],
                  interactive=False)
    full = _draw(panels=[
        _section(bands=[{"top": 8, "bottom": 4, "label": "Sand",
                         "hatch": "."},
                        {"top": 4, "bottom": 0, "label": "Clay"}],
                 dimensions=[{"from": [0, 0], "to": [4, 0],
                              "offset": [0, -0.8], "text": "B = 4.0 m"}]),
        # the arrow's text runs back over it from the tail, so the tail
        # stands clear of the diagram by the text's length (from 25 kPa the
        # line at 19.9 kPa crossed it, and the layout check said so)
        _diagram(arrows=[{"from": [34, 2.7], "to": [0, 2.7],
                          "text": "P = 120 kN/m"}])],
        ylim=[-2, 9], interactive=False)
    assert not [w for w in full.warnings if "no interactive" not in w], \
        full.warnings
    assert full.png_bytes != plain.png_bytes
    assert _size(full.png_bytes)[0] == pytest.approx(
        _size(plain.png_bytes)[0], abs=40)


def test_fill_to_a_base_line_on_one_plot_and_in_its_twin():
    res = render_data_plot([{"x": [0, 1, 2], "y": [0, 5, 3],
                             "style": "line", "fill_to_y": 0,
                             "label": "Moment"}])
    assert not res.warnings
    fills = [t for t in res.figure.data if t.fill == "toself"]
    assert len(fills) == 1 and list(fills[0].y)[-3:-1] == [0, 0]
    with pytest.raises(ValueError, match="fill_to_x or fill_to_y"):
        render_data_plot([{"x": [0, 1], "y": [0, 1], "fill_to_x": 0,
                           "fill_to_y": 0}])
    with pytest.raises(ValueError, match="must be a number"):
        render_data_plot([{"x": [0, 1], "y": [0, 1], "fill_to_x": "zero"}])


def test_a_note_can_be_turned():
    res = render_data_plot(
        [{"x": [0, 1], "y": [0, 1], "label": "A"}],
        annotations=[{"text": "Along the plane", "x": 0.5, "y": 0.5,
                      "rotation": 90}])
    ann = [a for a in res.figure.layout.annotations
           if a.text == "Along the plane"]
    assert ann and ann[0].textangle == -90
    with pytest.raises(ValueError, match="rotation"):
        render_data_plot([{"x": [0, 1], "y": [0, 1]}],
                         annotations=[{"text": "t", "x": 0, "y": 0,
                                       "rotation": 400}])


def test_a_drawing_panel_has_no_twin_and_says_so():
    res = _draw(panels=[_section(), _diagram()], ylim=[-1, 9])
    assert res.figure is None
    (note,) = res.warnings
    assert note.startswith("no interactive chart") and "PNG" in note
    col = _draw(panels=[{"kind": "column", "bands": [
        {"top": 10, "bottom": 2, "label": "Clay"}]},
        _chart("A", [0, 1, 2])])
    assert col.figure is None and col.panels[0]["x_range"] == [0, 1]
    quiet = _draw(panels=[_section(), _diagram()], ylim=[-1, 9],
                  interactive=False)
    assert not quiet.warnings


# --- what is refused, by name ----------------------------------------------

@pytest.mark.parametrize("panels, more, words", [
    ([{"kind": "elevation", "series": []}], {}, "kind"),
    ([{"serie": []}], {}, "Did you mean: serie -> series"),
    ([{"title": "Empty"}], {}, "is empty"),
    ([_chart("A", ["a", "b", "c"])], {}, "must be numbers"),
    ([_chart("A", [0, 1, 2])], {"series": [{"x": [0], "y": [0]}]},
     "give each panel its own 'series'"),
    ([_chart("A", [0, 1, 2])], {"plan_view": True}, "not used with panels"),
    ([_chart("A", [0, 1, 2])], {"xlabel": "x"}, "its own 'xlabel'"),
    ([_chart("A", [0, 1, 2], bands=[{"top": 1}])], {}, "bottom"),
    ([_chart("A", [0, 1, 2], dimensions=[{"from": [0, 0], "to": [0, 0]}])],
     {}, "one point"),
    ([_chart("A", [0, 1, 2], arrows=[{"from": [0, 0], "tip": [1, 0]}])],
     {}, "unknown key"),
    ([_chart(str(k), [0, 1, 2]) for k in range(9)], {}, "at most 8"),
    ([_chart(str(k), [0, 1, 2]) for k in range(8)], {"width_in": 4.0},
     "too narrow"),
    ("not a list", {}, "non-empty list"),
    # review 2026-11-13: refused under their own place, not as a footnote
    # or a note
    ([_chart("A", [0, 1, 2], bands=[{"top": 1, "bottom": 0, "label": 5}])],
     {}, r"panels\[0\] bands\[0\] label must be text"),
    ([_chart("A", [0, 1, 2], bands=[{"top": 1, "bottom": 0,
                                     "color": "notacolour"}])],
     {}, r"panels\[0\] bands\[0\] color"),
    ([_chart("A", [0, 1, 2], arrows=[{"from": [0, 0], "to": [1, 0],
                                      "text": 12}])],
     {}, r"panels\[0\] arrows\[0\] text must be text"),
    ([dict(_chart("A", [0, 1, 2]), title=3)], {},
     r"panels\[0\] title must be text"),
])
def test_bad_panels_are_refused_by_name(panels, more, words):
    with pytest.raises(ValueError, match=words):
        _draw(panels=panels, **more)


def test_a_series_error_in_a_panel_says_which_panel():
    with pytest.raises(ValueError, match=r"panels\[1\]: series\[0\]"):
        _draw(panels=[_chart("A", [0, 1, 2]),
                      {"series": [{"x": [0, 1], "y": [0]}]}])


# --- contours ---------------------------------------------------------------

GRID = {"x": [0, 10, 20, 30], "y": [0, 10, 20],
        "z": [[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0],
              [3.0, 4.0, 5.0, 6.0]],
        "label": "Settlement (mm)"}
POINTS = {"x": [0, 30, 0, 30, 15], "y": [0, 0, 20, 20, 10],
          "z": [1.0, 4.0, 3.0, 6.0, 3.5], "label": "Settlement (mm)"}


def test_contours_from_a_grid_with_their_twin():
    res = render_data_plot(None, contour=dict(GRID, levels=[1, 2, 3, 4, 5,
                                                             6]),
                           equal_aspect=True, xlabel="x (m)",
                           ylabel="y (m)")
    c = res.contour
    assert c["from"] == "grid" and c["levels"] == [1, 2, 3, 4, 5, 6]
    assert c["z_range"] == [1.0, 6.0] and not c["filled"]
    assert not res.warnings and "contours of Settlement (mm)" in \
        res.summary()
    trace = res.figure.data[0]
    assert trace.type == "contour" and trace.contours.size == 1
    assert trace.contours.coloring == "lines" and trace.contours.showlabels


def test_filled_contours_name_their_colour_scale():
    res = render_data_plot(
        [{"x": [5, 25], "y": [5, 15], "style": "markers", "label": "Plates",
          "labels": ["SP-1", "SP-2"]}],
        contour=dict(GRID, filled=True), equal_aspect=True)
    assert res.contour["filled"] and not res.warnings
    levels = res.contour["levels"]
    assert levels[0] <= 1.0 and levels[-1] >= 6.0      # automatic: spans z
    trace = res.figure.data[0]
    assert trace.colorbar.title.text == "Settlement (mm)"
    assert trace.contours.start == levels[0]
    plain = render_data_plot(
        [{"x": [5, 25], "y": [5, 15], "style": "markers",
          "label": "Plates"}], equal_aspect=True, interactive=False)
    # the colour scale takes room beside the plot
    assert res.png_bytes != plain.png_bytes


def test_contours_from_scattered_points_are_triangulated():
    res = render_data_plot(None, contour=dict(POINTS, levels=4),
                           equal_aspect=True)
    assert res.contour["from"] == "points" and res.figure is None
    (note,) = res.warnings
    assert note.startswith("no interactive chart") and "scattered" in note


def test_contours_on_a_plan_keep_the_plan_furniture():
    res = render_data_plot(
        [{"x": [0, 30, 30, 0], "y": [0, 0, 20, 20], "style": "polygon",
          "line_dash": "dashdot", "label": "Property line"}],
        contour=dict(GRID, filled=True), plan_view=True, north_deg=0,
        xlabel="Local grid (m)")
    assert res.plan_view and res.contour and not res.warnings


def test_uneven_levels_have_no_twin():
    res = render_data_plot(None, contour=dict(GRID, levels=[1.5, 2, 4]))
    assert res.figure is None
    assert any("unevenly spaced" in w for w in res.warnings)


@pytest.mark.parametrize("change, words", [
    ({"label": ""}, "needs 'label'"),
    ({"z": [[1, 2], [3, 4]]}, "one row per y value"),
    ({"x": [0, 10, 5, 30]}, "must increase"),
    ({"z": [[1, 1, 1, 1]] * 3}, "one value only"),
    ({"levels": 1}, "levels is how many"),
    ({"levels": [50, 60]}, "no contour level lies within"),
    ({"colour": "red"}, "unknown key"),
    ({"z": [1, 2, 3]}, "one length"),
])
def test_a_bad_contour_is_refused_by_name(change, words):
    with pytest.raises(ValueError, match=words):
        render_data_plot(None, contour=dict(GRID, **change))


def test_points_on_one_line_cannot_be_contoured():
    with pytest.raises(ValueError, match="contour"):
        render_data_plot(None, contour={
            "x": [0, 1, 2, 3], "y": [0, 1, 2, 3], "z": [1, 2, 3, 4],
            "label": "q (kPa)"})


def test_contour_is_refused_on_a_log_or_case_axis():
    with pytest.raises(ValueError, match="linear axes"):
        render_data_plot(None, contour=GRID, logx=True)


def test_a_footnote_on_one_plot_is_in_both_renderings():
    res = render_data_plot([{"x": [0, 1], "y": [0, 1], "label": "A"}],
                           footnote="Source: made up.")
    assert res.footnote == "Source: made up."
    assert any(a.text == "Source: made up."
               for a in res.figure.layout.annotations)
    bare = render_data_plot([{"x": [0, 1], "y": [0, 1], "label": "A"}])
    assert _size(res.png_bytes)[1] > _size(bare.png_bytes)[1]


def test_a_long_footnote_is_broken_to_the_figures_width():
    """Review 2026-11-13: one line past the edge widened the saved PNG."""
    long = " ".join(["Made-up words for a long footnote."] * 8)
    res = render_data_plot([{"x": [0, 1], "y": [0, 1], "label": "A"}],
                           footnote=long, interactive=False)
    bare = render_data_plot([{"x": [0, 1], "y": [0, 1], "label": "A"}],
                            interactive=False)
    assert _size(res.png_bytes)[0] <= _size(bare.png_bytes)[0] + 2


# --- bounds (review 2026-11-13) ---------------------------------------------

@pytest.mark.parametrize("contour, words", [
    ({"x": list(range(600)), "y": list(range(600)),
      "z": [[0.0] * 600] * 600}, "at most 250,000"),
    ({"x": [0.0] * 50_001, "y": [0.0] * 50_001, "z": [0.0] * 50_001},
     "at most 50,000"),
    (dict(GRID, levels=[1 + k / 100 for k in range(31)]), "at most 30"),
    (dict(GRID, z=[[None] * 4] * 3), "no values"),
])
def test_a_contour_too_large_is_refused_in_plain_words(contour, words):
    with pytest.raises(ValueError, match=words):
        render_data_plot(None, contour=dict(contour, label="Z (m)"),
                         interactive=False)


def test_a_rough_contour_is_drawn_without_numbers_and_says_so():
    """Hundreds of small loops: numbering each took minutes."""
    import random
    import time
    rng = random.Random(7)
    xs = [float(k) for k in range(60)]
    z = [[rng.random() for _ in xs] for _ in xs]
    t0 = time.time()
    res = render_data_plot(None, contour={"x": xs, "y": xs, "z": z,
                                          "label": "Z (m)"},
                           interactive=False)
    assert time.time() - t0 < 20
    assert any("too many to number" in w for w in res.warnings)


def test_reference_chart_is_refused_with_panels_or_no_series(tmp_path,
                                                             monkeypatch):
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))

    def overlay(params):                # never reached
        raise AssertionError("overlay called")
    with pytest.raises(ValueError, match="not in panels"):
        run_plot_data({"panels": [_chart("A", [0, 1, 2])],
                       "reference_chart": "made_up", "title": "Made-up"},
                      overlay=overlay)
    with pytest.raises(ValueError, match="give those series"):
        run_plot_data({"contour": GRID, "reference_chart": "made_up",
                       "title": "Made-up"}, overlay=overlay)


# --- the tool ---------------------------------------------------------------

def test_the_tool_takes_panels_and_contour_without_series(tmp_path,
                                                          monkeypatch):
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    out = run_plot_data({"panels": [_chart("A", [0, 1, 2]),
                                    _chart("B", [2, 1, 0])],
                         "title": "Made-up panels",
                         "ylabel": "Elevation (m)",
                         "output_path": "panels_tool"})
    assert out["status"] == "success" and len(out["panels"]) == 2
    assert out["plotly_json_path"]
    out = run_plot_data({"contour": GRID, "title": "Made-up contours",
                         "output_path": "contour_tool"})
    assert out["status"] == "success" and out["contour"]["from"] == "grid"
    with pytest.raises(ValueError, match="series"):
        run_plot_data({"title": "Nothing to draw"})


def test_the_tool_checks_a_panels_series_keys():
    with pytest.raises(ValueError) as exc:
        run_plot_data({"panels": [{"series": [
            {"x": [0, 1], "y": [0, 1], "dash": "dot"}]}]})
    assert "panels[0].series" in str(exc.value)
    assert "line_dash" in str(exc.value)


def test_the_description_names_the_new_inputs():
    from aec_agent_core.charts.plot_tool import PLOT_DATA_INFO, VALID_PLOT
    for key in ("panels", "contour", "footnote"):
        assert key in VALID_PLOT and key in PLOT_DATA_INFO["parameters"]
    assert "fill_to_x" in PLOT_DATA_INFO["parameters"]["series"][
        "description"]
    assert not PLOT_DATA_INFO["parameters"]["series"]["required"]


# --- the code examples, redrawn without code -------------------------------

@pytest.mark.parametrize("name", NEW_EXAMPLES)
def test_the_examples_draw_with_nothing_on_the_layout_check(name):
    params = dict(example(name)["params"])
    res = render_data_plot(params.pop("series", None), **params)
    found = [w for w in res.warnings
             if not w.startswith("no interactive chart")]
    assert not found, found
    lettering = parse_lettering(png_texts(res.png_bytes)[LETTERING_KEY])
    # printed at the calc package's 6.5 in text width
    printed = lettering["smallest_pt"] * min(
        1.0, 6.5 / lettering["saved_width_in"])
    assert printed >= MIN_PRINT_PT, lettering
    assert json.dumps(params)                       # plain JSON throughout


def test_the_panel_examples_hold_the_code_examples_numbers():
    """One set of made-up values behind both renderings of a figure."""
    wall = example("wall_section_pressures")["params"]
    data = CODE_EXAMPLES["wall_pressures"]["data"]
    assert wall["title"] == data["title"]
    assert [p.get("kind", "chart") for p in wall["panels"]] == \
        ["section"] + ["chart"] * 3
    for panel, dia in zip(wall["panels"][1:], data["diagrams"]):
        assert panel["series"][0]["x"] == dia["kpa"]
        assert panel["series"][0]["fill_to_x"] == 0
        assert panel["share_x"] == "pressure"
    pile = example("pile_design_panels")["params"]
    assert pile["hlines"][0]["value"] == \
        CODE_EXAMPLES["pile_design_profile"]["data"]["neutral_plane"][
            "elevation"]
    assert pile["panels"][0]["kind"] == "column"


def test_the_full_layout_check_runs_on_panels(monkeypatch):
    """A note written across another is found by the pixel check and said
    in the warnings."""
    res = _draw(panels=[_chart("A", [0, 40, 80], annotations=[
        {"text": "One long note here", "x": 40, "y": 6},
        {"text": "Another long note", "x": 42, "y": 6}])],
        interactive=False)
    assert any("overlaps" in w for w in res.warnings), res.warnings
    assert layout_check is not None


def test_defaults_used_lists_only_what_a_panel_figure_used(tmp_path,
                                                           monkeypatch):
    """A call with panels leaves the single-plot defaults (grid, log axes,
    the legend's place) out of ``defaults_used``, and says what the left-out
    series stands for; a single plot lists them as before."""
    from aec_agent_core.charts.plot_tool import call_plot_data
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    out = call_plot_data({"panels": [_chart("Made-up stress", [0, 1, 2])],
                          "title": "Made-up panel figure",
                          "ylabel": "Elevation (m)", "output_path": "p"})
    assert out["status"] == "success", out
    used = out["defaults_used"]
    for name in ("grid", "logx", "logy", "equal_aspect", "plan_view",
                 "legend_position", "north_deg"):
        assert name not in used, name
    assert "panels or contour" in used["series"]
    assert used["depth_axis"] is False
    one = call_plot_data({"series": [{"x": [0, 1], "y": [0, 1],
                                      "label": "Made-up line"}],
                          "title": "Made-up line figure", "xlabel": "x (m)",
                          "ylabel": "y (m)", "output_path": "q"})
    assert one["defaults_used"]["grid"] is True
    assert "series" not in one["defaults_used"]
