"""BACKLOG DR-35 (the owner's OWN-77, option (b)): a host hands the figure
helper its own worked examples; left out, everything is as before.

The core's figure helper takes ``examples=`` (an example set shaped like
``aec_agent_core.charts.figure_examples``) on ``build_figure_subagent``,
``FigureTasks``, ``make_figure_examples_tool``, ``nearest_example`` and
``brief_with_example``. Document Review hands in its own behind
``GEOTECH_REVIEW_FIGURE_EXAMPLES``; the geotech page hands in nothing and
keeps the core's set whatever the switch says.
"""

import json
from types import SimpleNamespace

import pytest

from aec_agent_core import review_flags
from aec_agent_core.charts import figure_examples
from aec_agent_core.deep import figure_helper
from aec_agent_core.deep.figure_helper import (EXAMPLE_HEADING, FigureTasks,
                                               brief_with_example,
                                               make_figure_examples_tool,
                                               nearest_example)

#: The figure_examples tool's description as it stood before DR-35, word
#: for word: with no example set handed in it must not move.
_OLD_DESCRIPTION = (
    "Worked examples of good engineering figures, each a complete "
    "plot_data call on made-up data with why it is drawn that way: "
    "a plan of borings and a footprint, an SPT profile against "
    "elevation, sensitivity cases, settlement against time, a load "
    "test, an exceedance curve, a wall section with its pressure "
    "diagrams, stress and pile profiles in panels, contours. No "
    "name: the list. name=<example>: "
    "that one in full. Adapt the nearest one; never copy its made-up "
    "numbers.")
_OLD_CODE_NOTE = (
    " The list also holds complete plot_code scripts (matplotlib, "
    "with the data each reads) to start from where plot_data "
    "cannot draw the figure; "
    "each entry says which kind it is.")


def _old_listing(with_code: bool) -> str:
    """The listing as the tool built it before DR-35, from the module's
    own functions."""
    names = figure_examples.example_names() + (
        figure_examples.code_example_names() if with_code else [])
    kinds = {"plot_data": "plot_data call", "plot_code": "plot_code script"}
    listing = [{"name": n, "kind": kinds[figure_examples.example(n)["kind"]],
                "shows": figure_examples.example(n)["shows"]} for n in names]
    return json.dumps({
        "examples": listing,
        "next": ("figure_examples(name=...) returns one: what it "
                 "shows, why it is drawn that way, and its complete "
                 "plot_data parameters"
                 + (", or its complete plot_code script and the "
                    "data it reads," if with_code else "")
                 + " to adapt.")}, ensure_ascii=False)


#: A made-up example set of one plot_data figure and no scripts.
_TINY = SimpleNamespace(
    EXAMPLES={"widget_tally": {
        "shows": "A tally of widgets by crate.",
        "why": ["x names the crates.", "Markers only.", "Starts at zero."],
        "params": {"title": "Widgets by crate", "xlabel": "Crate",
                   "ylabel": "Widgets", "series": [
                       {"x": ["C1", "C2"], "y": [3, 5],
                        "style": "markers", "label": "Widgets"}]}}},
    CODE_EXAMPLES={},
    TOOL_KINDS="a tally of widgets by crate")


@pytest.mark.parametrize("with_code", [False, True])
def test_with_no_set_the_tool_is_handed_exactly_what_it_was(with_code):
    tool = make_figure_examples_tool(plot_code=with_code)
    assert tool.description == _OLD_DESCRIPTION + (
        _OLD_CODE_NOTE if with_code else "")
    assert tool.invoke({"name": ""}) == _old_listing(with_code)
    names = figure_examples.example_names() + (
        figure_examples.code_example_names() if with_code else [])
    for n in names:
        assert tool.invoke({"name": n}) == json.dumps(
            figure_examples.example(n), ensure_ascii=False), n
    # the core's own set names its kinds where the tool used to
    assert figure_examples.TOOL_KINDS in tool.description


def test_with_no_set_the_attached_example_is_what_it_was():
    for name in figure_examples.example_names():
        assert figure_helper._example_of(name) == figure_examples.example(
            name)
    brief = ("A site plan of the borings and the building footprint, "
             "local grid in metres.")
    assert nearest_example(brief) == "plan_borings"
    out = brief_with_example(brief)
    assert EXAMPLE_HEADING in out and "'plan_borings'" in out
    assert FigureTasks().examples is None


def test_a_hosts_set_replaces_the_core_set_everywhere():
    tool = make_figure_examples_tool(plot_code=True, examples=_TINY)
    listing = json.loads(tool.invoke({"name": ""}))
    assert [e["name"] for e in listing["examples"]] == ["widget_tally"]
    # a set with no scripts says nothing of scripts, even with plot_code
    assert "plot_code" not in listing["next"]
    assert "plot_code" not in tool.description
    assert "a tally of widgets by crate" in tool.description
    assert "borings" not in tool.description
    full = json.loads(tool.invoke({"name": "widget tally"}))
    assert full["name"] == "widget_tally" and full["kind"] == "plot_data"
    assert full["matched"] == "'widget tally' read as 'widget_tally'"
    brief = "Draw the widget tally for the two crates."
    assert nearest_example(brief, examples=_TINY) == "widget_tally"
    assert nearest_example(brief) is None
    assert "'widget_tally'" in brief_with_example(brief, examples=_TINY)
    assert FigureTasks(examples=_TINY).examples is _TINY


def test_the_switch_is_off_by_default_and_kept_apart(monkeypatch):
    env = review_flags.FIGURE_EXAMPLES_ENV
    assert env == "GEOTECH_REVIEW_FIGURE_EXAMPLES"
    assert env in review_flags.LATER_ENVS
    assert env not in review_flags.ALL_ENVS
    assert all(env not in arm for arm in review_flags.ARMS.values())
    monkeypatch.delenv(env, raising=False)
    assert review_flags.review_figure_examples() is False
    monkeypatch.setenv(env, "1")
    assert review_flags.review_figure_examples() is True
    assert f"{env}=1" in review_flags.describe()
    with review_flags.switches({}):
        assert review_flags.review_figure_examples() is False


def test_the_core_default_ignores_the_switch(monkeypatch, tmp_path):
    """The geotech page builds its helper with no set: with the switch on
    it still gets the core's examples."""
    pytest.importorskip("deepagents")
    monkeypatch.setenv(review_flags.FIGURE_EXAMPLES_ENV, "1")
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    spec = figure_helper.build_figure_subagent()
    tool = next(t for t in spec["tools"] if t.name == "figure_examples")
    with_code = any(t.name == "plot_code" for t in spec["tools"])
    assert tool.description == _OLD_DESCRIPTION + (
        _OLD_CODE_NOTE if with_code else "")
    assert tool.invoke({"name": ""}) == _old_listing(with_code)
    mine = figure_helper.build_figure_subagent(examples=_TINY)
    tool = next(t for t in mine["tools"] if t.name == "figure_examples")
    assert [e["name"] for e in json.loads(tool.invoke(
        {"name": ""}))["examples"]] == ["widget_tally"]
    # only the examples tool differs: the prompt and description do not
    assert mine["system_prompt"] == spec["system_prompt"]
    assert mine["description"] == spec["description"]
