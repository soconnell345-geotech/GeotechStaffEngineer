"""The chart stack, the plotting box and the figure helper in the core
(BACKLOG INF-32, the app split's step 11 (b); plan
``module_work/APP_SPLIT_PLAN.md`` sections 2.3 and 2.8).

These run with only the core and planlens installed: a figure is drawn
through the core alone, with no reference chart lines handed in, and the
chart-lines registration (what an app with a reference library fills) is
the core's own. Made-up values only.
"""

from __future__ import annotations

import json
import os

import pytest

from aec_agent_core import charts


@pytest.fixture
def no_reference_charts():
    """No app's chart lines registered (restored afterwards)."""
    before = charts.reference_charts()
    charts.register_reference_charts(None)
    yield
    charts.register_reference_charts(before)


def _series():
    return [{"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.5, 2.5, 3.0],
             "label": "made-up curve"}]


def test_plot_data_draws_through_the_core_alone(tmp_path, no_reference_charts):
    from aec_agent_core.charts.plot_tool import run_plot_data
    out = tmp_path / "curve.png"
    res = run_plot_data({"series": _series(), "title": "A made-up curve",
                         "xlabel": "x (m)", "ylabel": "y (kPa)",
                         "output_path": str(out)})
    assert res.get("status", "success") == "success", res
    assert out.is_file() and out.stat().st_size > 1000
    with open(out, "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"


def test_without_chart_lines_reference_chart_is_refused_by_name(
        tmp_path, no_reference_charts):
    from aec_agent_core.charts.plot_tool import method_info, run_plot_data
    assert "reference_chart" not in method_info()["parameters"]
    with pytest.raises(Exception) as err:
        run_plot_data({"series": _series(), "reference_chart": "any",
                       "output_path": str(tmp_path / "x.png")})
    assert "reference_chart" in str(err.value)


def test_the_box_says_nothing_of_chart_lines_when_none_are_registered(
        no_reference_charts):
    from aec_agent_core import plot_box
    assert plot_box._chart_lines_path() == ""
    assert plot_box._chart_lines_note() == ""


def test_a_registered_provider_is_what_the_stack_reads(tmp_path,
                                                       no_reference_charts):
    """The registration an app fills: plot_data offers reference_chart and
    draws through the provider's overlay; the box finds its chart_lines
    file. The provider here is made up."""
    lines = tmp_path / "made_up_chart_lines.py"
    lines.write_text("def chart_names():\n    return ['made_up']\n",
                     encoding="utf-8")
    import importlib.util
    spec = importlib.util.spec_from_file_location("made_up_chart_lines",
                                                  lines)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seen = []

    class Provider:
        chart_lines = module
        parameter = {"type": "string", "description": "a made-up chart"}
        returns = "the made-up chart's lines"

        @staticmethod
        def overlay(params):
            seen.append(params["reference_chart"])
            params = dict(params)
            params.pop("reference_chart")
            return params, 0, "made-up chart drawn"

    charts.register_reference_charts(Provider())
    from aec_agent_core import plot_box
    from aec_agent_core.charts.plot_tool import method_info, run_plot_data
    assert "reference_chart" in method_info()["parameters"]
    assert plot_box._chart_lines_path() == str(lines)
    assert "made_up" in plot_box._chart_lines_note()
    res = run_plot_data({"series": _series(), "reference_chart": "made_up",
                         "output_path": str(tmp_path / "y.png")})
    assert seen == ["made_up"]
    assert res.get("status", "success") == "success", res


def test_the_box_loads_the_layout_check_from_the_core_chart_folder():
    """The plotting box's child loads layout_ink.py and plan_furniture.py
    by file from ``charts/`` beside ``plot_box/``: both moved together."""
    from aec_agent_core import plot_box
    here = os.path.dirname(os.path.abspath(plot_box.CHILD))
    folder = os.path.join(os.path.dirname(here), "charts")
    for name in ("layout_ink.py", "plan_furniture.py"):
        assert os.path.isfile(os.path.join(folder, name)), name


def test_the_worked_examples_are_in_the_core():
    from aec_agent_core.charts import figure_examples
    from aec_agent_core.deep import figure_helper
    names = figure_examples.example_names()
    # the nine geotech worked examples (plan 2.8): six plot_data calls and
    # three plot_code scripts
    assert len(names) + len(figure_examples.code_example_names()) == 9
    assert figure_helper.nearest_example("a made-up plan of points") \
        in names + [None]
    full = figure_examples.example(names[0])
    assert json.dumps(full)


def test_the_plotting_box_runs_code_in_the_core_alone(tmp_path):
    from aec_agent_core import plot_box
    res = plot_box.run_plot_code(
        "import matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots(figsize=(4, 3))\n"
        "ax.plot([0, 1, 2], [0, 1, 4], label='made-up')\n"
        "ax.set_xlabel('x (m)'); ax.set_ylabel('y (kPa)')\n"
        "ax.legend()\n",
        name="box_made_up", root=str(tmp_path))
    assert res.get("status") == "success", res
