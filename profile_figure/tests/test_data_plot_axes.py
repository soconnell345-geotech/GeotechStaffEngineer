"""Axis ranges as given, and the x axis on top (BACKLOG GM-107 fix 9, GM-106).

Live wave w9 (``module_work/live_smoke/runs/w9-figures/TRACE_REVIEW.md``
§2.7): ``xlim [100, 0.001]`` on a log axis, coarse on the left, was sorted
without a word, so the axis came out the other way; the result said
``success`` and only a look found it. The interactive twin ignored ``xlim``
on any log axis.
"""

import math

import pytest

pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt  # noqa: E402

from profile_figure import render_data_plot  # noqa: E402

CURVE = {"x": [50, 4.75, 0.425, 0.075, 0.01], "y": [100, 80, 45, 22, 10],
         "label": "B-1"}


@pytest.fixture
def drawn(monkeypatch):
    """The matplotlib figures render_data_plot draws, kept open."""
    figs = []
    real = plt.close

    def keep(fig=None):
        figs.append(fig)
    monkeypatch.setattr(plt, "close", keep)
    yield figs
    for f in figs:
        real(f)


def test_a_reversed_range_is_drawn_as_given_on_a_log_axis(drawn):
    res = render_data_plot([CURVE], logx=True, xlim=[100, 0.001])
    ax = drawn[0].axes[0]
    assert ax.get_xlim() == pytest.approx((100, 0.001))
    assert res.reversed_axes == ["x"]
    assert "x axis runs high to low" in res.summary()
    assert res.to_dict()["reversed_axes"] == ["x"]
    assert list(res.figure.layout.xaxis.range) == pytest.approx([2.0, -3.0])
    assert res.figure.layout.xaxis.type == "log"
    assert not res.warnings


def test_xlim_reaches_the_interactive_chart_on_a_log_axis():
    res = render_data_plot([CURVE], logx=True, xlim=[0.001, 100],
                           logy=False, ylim=[0, 100])
    assert list(res.figure.layout.xaxis.range) == pytest.approx([-3.0, 2.0])
    assert list(res.figure.layout.yaxis.range) == [0, 100]
    assert res.reversed_axes == []
    logy = render_data_plot([CURVE], logy=True, ylim=[1, 1000])
    assert list(logy.figure.layout.yaxis.range) == pytest.approx(
        [0.0, math.log10(1000)])


def test_a_log_axis_range_at_or_below_zero_is_left_out_and_said():
    res = render_data_plot([CURVE], logx=True, xlim=[0, 100])
    assert any("xlim ignored" in w for w in res.warnings)
    assert res.figure.layout.xaxis.range is None


def test_a_reversed_y_range_and_depth_axis(drawn):
    up = render_data_plot([CURVE], ylim=[100, 0])
    assert drawn[-1].axes[0].get_ylim() == pytest.approx((100, 0))
    assert up.reversed_axes == ["y"]
    # depth_axis sets the direction of y itself, whatever the order
    for ylim in ([0, 100], [100, 0]):
        deep = render_data_plot([CURVE], ylim=ylim, depth_axis=True)
        assert drawn[-1].axes[0].get_ylim() == pytest.approx((100, 0))
        assert list(deep.figure.layout.yaxis.range) == [100, 0]
        assert deep.reversed_axes == []


def test_x_on_top_puts_the_x_axis_along_the_top(drawn):
    res = render_data_plot([CURVE], x_on_top=True, xlabel="N60",
                           ylabel="Elevation (m)")
    ax = drawn[0].axes[0]
    assert ax.xaxis.get_label_position() == "top"
    assert ax.xaxis.get_ticks_position() == "top"
    assert ax.spines["top"].get_visible()
    assert res.figure.layout.xaxis.side == "top"
    assert res.x_on_top and "x axis along the top" in res.summary()
    assert res.figure.layout.yaxis.autorange != "reversed"   # elevation up
    plain = render_data_plot([CURVE])
    assert not plain.x_on_top and plain.figure.layout.xaxis.side is None
    # a plan is a drawing: no axes to move
    plan = render_data_plot([{"x": [0, 10], "y": [0, 5], "label": "a"}],
                            plan_view=True, x_on_top=True)
    assert not plan.x_on_top
