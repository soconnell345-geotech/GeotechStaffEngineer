"""The figure helper (owner's choice 2026-10-15, OWN-61 option B; GM-92).

"Go with B next, the figure helper. I assume the helper would be the Tufte
checklist and some actual good looking example plots."
"""

import json

import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from funhouse_agent import review_flags  # noqa: E402
from funhouse_agent.deep.figure_helper import (  # noqa: E402
    DEFAULT_FIGURE_MAX_MODEL_CALLS, FIGURE_CHECKLIST, FIGURE_HELPER_NAME,
    build_figure_subagent, make_figure_examples_tool)
from profile_figure.figure_examples import (EXAMPLES, example,  # noqa: E402
                                            example_names)


class _Quiet(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "quiet"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="done"))])


# ---------------------------------------------------------------------------
# The examples
# ---------------------------------------------------------------------------

def test_every_example_draws_cleanly_through_the_real_tool(tmp_path,
                                                           monkeypatch):
    """Each example is a complete plot_data call the helper's own plot_data
    tool accepts (strict arguments) and draws with no warning."""
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    spec = build_figure_subagent()
    plot = next(t for t in spec["tools"] if t.name == "plot_data")
    for name in example_names():
        params = dict(example(name)["params"], output_path=f"ex_{name}")
        out = json.loads(plot.invoke(params))
        assert out.get("status") == "success", (name, out)
        assert not out.get("warnings"), (name, out.get("warnings"))
        assert (tmp_path / f"ex_{name}.png").is_file()
        assert (tmp_path / f"ex_{name}.plotly.json").is_file()


def test_examples_follow_their_own_checklist():
    plan = example("plan_borings")["params"]
    # a plan is a drawing: plan_view, a north arrow, a property line
    assert plan["plan_view"] is True and plan["north_deg"] is not None
    assert any(s.get("line_dash") == "dashdot" for s in plan["series"])
    assert any(s.get("style") == "polygon" and s.get("fill")
               for s in plan["series"])
    assert all(s.get("labels") for s in plan["series"]
               if s.get("style") == "markers")
    cases = example("cases")["params"]
    assert all(s["style"] == "markers" for s in cases["series"])
    assert all(isinstance(v, str) for v in cases["series"][0]["x"])
    assert example("depth_profile")["params"]["depth_axis"] is True
    assert example("time_settlement")["params"]["logx"] is True
    for name in example_names():
        ex = example(name)
        assert ex["shows"] and len(ex["why"]) >= 3, name
        p = ex["params"]
        assert p.get("title") and p.get("xlabel"), name
        # a chart names both axes; a plan names its coordinate system
        assert p.get("plan_view") or p.get("ylabel"), name


def test_the_exceedance_example_marks_points_on_its_own_curve():
    import numpy as np
    curve, marks = example("exceedance")["params"]["series"]
    for x, y in zip(marks["x"], marks["y"]):
        assert np.interp(x, curve["x"], curve["y"]) == pytest.approx(
            y, abs=0.01)


def test_examples_are_copies():
    ex = example("cases")
    ex["params"]["title"] = "changed"
    assert EXAMPLES["cases"]["params"]["title"] != "changed"


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------

def test_figure_examples_tool():
    tool = make_figure_examples_tool()
    listing = json.loads(tool.invoke({}))
    assert [e["name"] for e in listing["examples"]] == example_names()
    one = json.loads(tool.invoke({"name": "depth_profile"}))
    assert one["name"] == "depth_profile" and "series" in one["params"]
    bad = json.loads(tool.invoke({"name": "pie_chart"}))
    assert "no example 'pie_chart'" in bad["error"]


def test_the_helper_draws_looks_and_counts():
    spec = build_figure_subagent()
    assert spec["name"] == FIGURE_HELPER_NAME
    names = {t.name for t in spec["tools"]}
    assert names == {"plot_data", "analyze_image", "list_files",
                     "read_text_file", "figure_examples", "calculate"}
    prompt = spec["system_prompt"]
    assert FIGURE_CHECKLIST in prompt
    for words in ("Tufte", "Data-ink", "plan_view", "north_deg",
                  "property line", "equal_aspect", "depth_axis",
                  "analyze_image", "At most two", "never invent"):
        assert words in prompt, words
    from funhouse_agent.deep.limits import ModelCallBudgetMiddleware
    budget = [m for m in spec["middleware"]
              if isinstance(m, ModelCallBudgetMiddleware)]
    assert budget and budget[0].run_limit == DEFAULT_FIGURE_MAX_MODEL_CALLS


def _captured_subagents(monkeypatch, **kw):
    import funhouse_agent.deep.agent as agent_mod
    captured = {}
    real = agent_mod.create_deep_agent

    def spy(**k):
        captured.update(k)
        return real(**k)
    monkeypatch.setattr(agent_mod, "create_deep_agent", spy)
    agent_mod.build_deep_agent(model=_Quiet(), reference_mode="off", **kw)
    return captured["subagents"]


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(review_flags.FIGURE_HELPER_ENV, raising=False)
    names = {s["name"] for s in _captured_subagents(monkeypatch)}
    assert FIGURE_HELPER_NAME not in names
    assert review_flags.ARMS["figure_helper"] == {
        review_flags.FIGURE_HELPER_ENV: "1"}


def test_the_switch_adds_it_with_the_guards(monkeypatch):
    monkeypatch.setenv(review_flags.FIGURE_HELPER_ENV, "1")
    specs = _captured_subagents(monkeypatch)
    fig = next(s for s in specs if s["name"] == FIGURE_HELPER_NAME)
    kinds = {type(m).__name__ for m in fig["middleware"]}
    assert {"ModelCallBudgetMiddleware", "ToolErrorGuard",
            "OutputLimitGuard", "AgentBusyRetry"} <= kinds
    assert "LOOKS" in fig["description"]
