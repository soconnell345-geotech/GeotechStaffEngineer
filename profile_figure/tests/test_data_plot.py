"""Tests for the generic data plot — the PNG and its interactive Plotly twin.

The two backends must draw the SAME figure: same cleaned series (including the
dropped non-finite points), same colors, same reference lines, same reversed
depth axis. The interactive chart is what the chat shows; the PNG is what a
PDF report embeds, so building the chart must never cost the image.
"""

import math

import pytest

from profile_figure import render_data_plot
from profile_figure.data_plot import build_data_plot_figure, _clean_series


SERIES = [
    {"x": [4, 8, 12, 9], "y": [1.5, 3.0, 4.5, 6.0], "label": "B-1 SPT N"},
    {"x": [6, 10, 14, 11], "y": [1.5, 3.0, 4.5, 6.0], "label": "B-2 SPT N",
     "style": "step"},
]


def _figure_of(**kwargs):
    fig = render_data_plot(SERIES, **kwargs).figure
    assert fig is not None, "plotly is a core dependency — figure expected"
    return fig


# ---------------------------------------------------------------------------
# Same data in both backends
# ---------------------------------------------------------------------------

def test_figure_draws_the_cleaned_series():
    pytest.importorskip("plotly")
    res = render_data_plot(SERIES, title="SPT vs depth")
    fig = res.figure
    assert len(fig.data) == len(SERIES)
    assert [t.name for t in fig.data] == ["B-1 SPT N", "B-2 SPT N"]
    assert list(fig.data[0].x) == [4, 8, 12, 9]
    assert list(fig.data[0].y) == [1.5, 3.0, 4.5, 6.0]
    # style "both" -> line + markers; style "step" -> the hv line shape
    assert fig.data[0].mode == "lines+markers"
    assert fig.data[1].mode == "lines" and fig.data[1].line.shape == "hv"
    assert fig.layout.title.text == "SPT vs depth"


def test_figure_and_png_drop_the_same_non_finite_points():
    pytest.importorskip("plotly")
    raw = [{"x": [1.0, float("nan"), 3.0], "y": [1.0, 2.0, 3.0],
            "label": "partial"}]
    res = render_data_plot(raw)
    assert res.warnings and "non-finite" in res.warnings[0]
    assert res.series[0]["n"] == 2                 # what the PNG drew
    assert list(res.figure.data[0].x) == [1.0, 3.0]
    assert list(res.figure.data[0].y) == [1.0, 3.0]


def test_builder_takes_the_cleaned_series_directly():
    pytest.importorskip("plotly")
    warnings = []
    cleaned = [_clean_series(s, i, warnings) for i, s in enumerate(SERIES)]
    fig = build_data_plot_figure(cleaned, title="T", xlabel="N", ylabel="z")
    assert fig is not None
    assert [t.name for t in fig.data] == ["B-1 SPT N", "B-2 SPT N"]
    assert fig.layout.xaxis.title.text == "N"
    assert fig.layout.yaxis.title.text == "z"


def test_explicit_series_color_wins_over_the_palette():
    pytest.importorskip("plotly")
    fig = _figure_of()
    first = fig.data[0].line.color
    custom = render_data_plot(
        [{"x": [1, 2], "y": [1, 2], "color": "#ff0000"}]).figure
    assert first != "#ff0000"
    assert custom.data[0].line.color == "#ff0000"


# ---------------------------------------------------------------------------
# Axes — the matplotlib semantics, mirrored
# ---------------------------------------------------------------------------

def test_depth_axis_reverses_y_and_puts_x_on_top():
    pytest.importorskip("plotly")
    fig = _figure_of(depth_axis=True)
    assert fig.layout.yaxis.autorange == "reversed"
    assert fig.layout.xaxis.side == "top"
    plain = _figure_of()
    assert plain.layout.yaxis.autorange is None
    assert plain.layout.xaxis.side is None


def test_log_axes_are_set_on_both_axes():
    pytest.importorskip("plotly")
    fig = _figure_of(logx=True, logy=True)
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.yaxis.type == "log"
    assert _figure_of().layout.xaxis.type is None


def test_grid_can_be_turned_off():
    pytest.importorskip("plotly")
    assert _figure_of(grid=True).layout.xaxis.showgrid is True
    assert _figure_of(grid=False).layout.xaxis.showgrid is False


# ---------------------------------------------------------------------------
# Reference lines and legend
# ---------------------------------------------------------------------------

def test_reference_lines_are_drawn_and_labelled():
    pytest.importorskip("plotly")
    fig = _figure_of(hlines=[{"value": 2.0, "label": "Water table"}],
                     vlines=[{"value": 10.0}])
    dashes = sorted(s.line.dash for s in fig.layout.shapes)
    assert dashes == ["dash", "dot"]               # hline dashed, vline dotted
    hline = [s for s in fig.layout.shapes if s.line.dash == "dash"][0]
    vline = [s for s in fig.layout.shapes if s.line.dash == "dot"][0]
    assert hline.y0 == 2.0 and vline.x0 == 10.0
    # a labelled line carries its label as an annotation (Plotly shapes have
    # no legend entry, where matplotlib's axhline does)
    assert [a.text for a in fig.layout.annotations] == ["Water table"]


def test_reference_lines_on_a_log_axis_use_log_coordinates():
    pytest.importorskip("plotly")
    # plotly.py does not convert shape coordinates for a log axis, so a line
    # at 100 must be drawn at log10(100) = 2.0 or it lands at 10**100.
    fig = render_data_plot([{"x": [1, 10, 100], "y": [1, 2, 3]}],
                           logx=True, vlines=[{"value": 100.0}]).figure
    assert fig.layout.shapes[0].x0 == pytest.approx(math.log10(100.0))
    # a non-positive value has no place on a log axis — skipped, not crashed
    ok = render_data_plot([{"x": [1, 10], "y": [1, 2]}], logx=True,
                          vlines=[{"value": 0.0}]).figure
    assert len(ok.layout.shapes) == 0


def test_legend_default_matches_the_png_rule():
    pytest.importorskip("plotly")
    one = render_data_plot([{"x": [1, 2], "y": [1, 2]}]).figure
    assert one.layout.showlegend is False           # single unlabelled series
    assert _figure_of().layout.showlegend is True   # two series
    labelled_ref = render_data_plot(
        [{"x": [1, 2], "y": [1, 2]}],
        hlines=[{"value": 1.5, "label": "allowable"}]).figure
    assert labelled_ref.layout.showlegend is True   # labelled reference line
    forced_off = render_data_plot(SERIES, legend=False).figure
    assert forced_off.layout.showlegend is False


# ---------------------------------------------------------------------------
# The PNG is never at risk
# ---------------------------------------------------------------------------

def test_interactive_false_skips_the_figure_and_keeps_the_png():
    pytest.importorskip("plotly")
    on = render_data_plot(SERIES, title="T", depth_axis=True)
    off = render_data_plot(SERIES, title="T", depth_axis=True,
                           interactive=False)
    assert on.figure is not None
    assert off.figure is None
    assert off.png_bytes == on.png_bytes           # byte-identical image
    assert off.to_dict() == on.to_dict()


def test_a_plotly_failure_costs_a_warning_not_the_png(monkeypatch):
    import profile_figure.data_plot as dp

    def boom(*_a, **_kw):
        raise RuntimeError("plotly exploded")

    monkeypatch.setattr(dp, "build_data_plot_figure", boom)
    res = dp.render_data_plot(SERIES)
    assert res.figure is None
    assert res.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
    assert any("interactive chart not built" in w for w in res.warnings)


def test_figure_stays_out_of_to_dict():
    pytest.importorskip("plotly")
    import json
    res = render_data_plot(SERIES)
    payload = json.dumps(res.to_dict())
    assert "figure" not in payload and "png_bytes" not in payload


def test_figure_round_trips_through_json():
    pytest.importorskip("plotly")
    import plotly.io as pio
    res = render_data_plot(SERIES, depth_axis=True, title="Round trip")
    again = pio.from_json(res.figure.to_json())
    assert len(again.data) == 2
    assert again.layout.yaxis.autorange == "reversed"
