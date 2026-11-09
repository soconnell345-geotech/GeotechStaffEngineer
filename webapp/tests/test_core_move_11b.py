"""The app side of step 11 (b) of the split (BACKLOG INF-32;
``module_work/inf32/CHECK_STEP11B.md``): the figure helper, the plotting
box, the chart stack and the worked examples are the shared core's; every
old import path (the modules under the two moved packages included) IS the
core's module, and GSE hands the core its reference chart lines through the
core's registration, as before the move.
"""

from __future__ import annotations

import importlib

import pytest

import funhouse_agent

STEP_11B = {
    "funhouse_agent.arg_checks": "aec_agent_core.arg_checks",
    "funhouse_agent.charts": "aec_agent_core.charts",
    "funhouse_agent.drawing_wait": "aec_agent_core.drawing_wait",
    "funhouse_agent.method_call": "aec_agent_core.method_call",
    "funhouse_agent.plot_box": "aec_agent_core.plot_box",
    "funhouse_agent.progress_notes": "aec_agent_core.progress_notes",
    "funhouse_agent.deep.figure_helper": "aec_agent_core.deep.figure_helper",
}

#: The modules under the two moved packages, by their old names.
UNDER = ["funhouse_agent.charts." + m for m in (
    "data_plot", "plot_tool", "layout_ink", "plan_furniture", "plotting",
    "figure_examples")] + ["funhouse_agent.plot_box." + m
                           for m in ("child", "figures")]


def test_the_table_holds_step_11b():
    for old, new in STEP_11B.items():
        assert funhouse_agent.CORE_ALIASES[old] == new


@pytest.mark.parametrize("old", sorted(STEP_11B) + UNDER)
def test_every_old_name_is_the_core_module(old):
    new = "aec_agent_core." + old.split(".", 1)[1]
    assert importlib.import_module(old) is importlib.import_module(new)
    parent, _, leaf = old.rpartition(".")
    assert getattr(importlib.import_module(parent), leaf) is \
        importlib.import_module(new)


def test_the_old_from_imports_give_the_core_objects():
    from funhouse_agent import plot_box
    from funhouse_agent.charts import register_reference_charts
    from funhouse_agent.dispatch import run_method
    import aec_agent_core.charts as core_charts
    import aec_agent_core.method_call as core_call
    import aec_agent_core.plot_box as core_box
    assert plot_box is core_box
    assert register_reference_charts is core_charts.register_reference_charts
    assert run_method is core_call.run_method


def test_a_patch_by_the_old_name_reaches_the_core(monkeypatch):
    monkeypatch.setattr("funhouse_agent.plot_box.CHILD", "elsewhere")
    from aec_agent_core import plot_box
    assert plot_box.CHILD == "elsewhere"


def test_gse_hands_in_the_reference_chart_lines():
    """GSE's profile_figure adapter registers the reference library's chart
    lines with the core's chart stack on import (and the geotech builds
    call it again); plot_data then offers reference_chart."""
    from aec_agent_core import charts
    from funhouse_agent.adapters import profile_figure_adapter as adapter
    has_lines = adapter.register_reference_charts()
    provider = charts.reference_charts()
    if not has_lines:
        pytest.skip("the installed reference library has no chart lines")
    import geotech_references.chart_lines as lines
    assert provider.chart_lines is lines
    from aec_agent_core.charts.plot_tool import method_info
    assert "reference_chart" in method_info()["parameters"]


def test_the_calc_helper_names_are_read_from_their_own_module():
    """``CALC_REDRAW_NOTE``, ``FigureLooks`` and ``MODULE_FIGURE_LOOK`` are
    the geotech calc helper's (``funhouse_agent.deep.calc_figures`` since
    INF-30 step 5); the figure helper, now the core's, no longer serves
    them under its own name (the core imports nothing of GSE)."""
    from funhouse_agent.deep import calc_figures, figure_helper
    for name in ("CALC_REDRAW_NOTE", "FigureLooks", "MODULE_FIGURE_LOOK"):
        assert hasattr(calc_figures, name)
        assert not hasattr(figure_helper, name)


@pytest.mark.parametrize("old,new", [
    ("geotech_common.layout_ink", "aec_agent_core.charts.layout_ink"),
    ("geotech_common.plan_furniture", "aec_agent_core.charts.plan_furniture"),
    ("profile_figure.data_plot", "aec_agent_core.charts.data_plot"),
    ("profile_figure.figure_examples",
     "aec_agent_core.charts.figure_examples"),
])
def test_step_6_names_still_reach_the_one_copy(old, new):
    """Step 6's stubs in GSE's packages (old name -> funhouse_agent.charts)
    land on the core's module through the 11 (b) alias."""
    assert importlib.import_module(old) is importlib.import_module(new)
