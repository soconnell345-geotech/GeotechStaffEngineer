"""The figure helper (owner's choice 2026-10-15, OWN-61 option B; GM-92).

"Go with B next, the figure helper. I assume the helper would be the Tufte
checklist and some actual good looking example plots."

The owner's conventions for profiles and design figures (GM-97,
2026-10-15): elevation on the vertical axis by default ("most borings have
a slightly different ground surface elevation"), "kPa and kN
(everywhere!)", past pressure as test points or a design line by the stage
of analysis, and the wall drawn by default.
"""

import json
import re

import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from funhouse_agent import review_flags  # noqa: E402
from funhouse_agent.deep.figure_helper import (  # noqa: E402
    DEFAULT_FIGURE_MAX_MODEL_CALLS, FIGURE_CHECKLIST, FIGURE_HELPER_NAME,
    build_figure_subagent, make_figure_examples_tool)
from profile_figure.figure_examples import (  # noqa: E402
    CODE_EXAMPLES, EXAMPLES, code_example_names, example, example_names)


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
    assert example("time_settlement")["params"]["logx"] is True
    for name in example_names():
        assert example(name)["kind"] == "plot_data"


def test_the_profile_example_is_against_elevation():
    """GM-97: "use elevation as the default. You won't be able to get both
    to work together since most borings have a slightly different ground
    surface elevation"."""
    ex = example("spt_profile")
    p = ex["params"]
    assert not p.get("depth_axis")
    assert p["ylabel"].startswith("Elevation") and "depth" not in \
        p["ylabel"].lower()
    grounds = {re.search(r"ground El\. ([0-9.]+) m", s["label"]).group(1)
               for s in p["series"]}
    assert len(grounds) == 3, grounds          # three different ground levels
    for s in p["series"]:
        assert len(s["hover"]) == len(s["y"])
        assert all("below ground" in h for h in s["hover"])
    assert "different ground levels" in " ".join(ex["why"])
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
    code = example("wall_pressures")
    code["data"]["title"] = "changed"
    assert CODE_EXAMPLES["wall_pressures"]["data"]["title"] != "changed"


# ---------------------------------------------------------------------------
# The plot_code examples (GM-97)
# ---------------------------------------------------------------------------

def test_every_code_example_draws_cleanly_in_the_box(tmp_path):
    """Each script runs in the real locked box on its own data, saves one
    PNG and comes back with nothing on the layout check (lettering that
    overlaps, or that a drawn line runs through)."""
    pytest.importorskip("matplotlib")
    from funhouse_agent import plot_box
    if not plot_box.available():
        pytest.skip("the plotting box cannot run here")
    for name in code_example_names():
        ex = example(name)
        assert ex["kind"] == "plot_code"
        assert not plot_box.check_code(ex["code"]), name
        out = plot_box.run_plot_code(ex["code"], data=ex["data"], name=name,
                                     root=str(tmp_path))
        assert out["status"] == "success", (name, out.get("error"))
        assert not out.get("layout_check"), (name, out["layout_check"])
        assert not out.get("refused") and not out.get("warnings"), out
        (image,) = out["images"]
        assert image["file"] == f"{name}.png"
        assert (tmp_path / image["file"]).is_file()
        assert max(image["width_px"], image["height_px"]) <= 2048


def test_code_examples_follow_the_owners_conventions():
    """Elevation on the shared vertical axis, kPa and kN, lettering at least
    7 pt, the wall drawn (and how to draw its diagrams alone), test points
    against a design line."""
    for name in code_example_names():
        ex = example(name)
        assert ex["shows"] and len(ex["why"]) >= 4, name
        data = json.dumps(ex["data"], ensure_ascii=False)
        assert ex["data"]["elevation_label"].startswith("Elevation"), name
        assert "kPa" in ex["code"] + data, name
        sizes = [float(v) for v in re.findall(
            r"(?<![a-z])(?:fontsize|font\.size|titlesize|labelsize)"
            r"[\"']?\s*[=:]\s*([0-9.]+)", ex["code"])]
        assert sizes and min(sizes) >= 7, (name, sizes)
        assert "depth_axis" not in ex["code"] and "invert_yaxis" not in \
            ex["code"], name
    assert "kN/m" in json.dumps(example("wall_pressures")["data"],
                                ensure_ascii=False) + \
        example("wall_pressures")["code"]
    assert "kN" in example("pile_design_profile")["data"]["force_axis"][
        "label"]
    wall = example("wall_pressures")
    assert wall["data"]["wall"]["outline"]
    assert "set_aspect(\"equal\"" in wall["code"]
    assert "alone" in " ".join(wall["why"])
    settle = example("settlement_stress_profile")
    assert "design" in " ".join(settle["why"])
    assert "linestyle=\"none\"" in settle["code"]      # tests as points
    assert "Consolidation" in \
        settle["data"]["preconsolidation"]["tests"]["label"]
    pile = example("pile_design_profile")
    assert "neutral plane" in " ".join(pile["why"]).lower()


def test_the_settlement_script_draws_an_assumed_design_line(tmp_path):
    """The other stage of analysis: an assumed preconsolidation profile is
    a line named as one, beside the tests it was drawn through."""
    pytest.importorskip("matplotlib")
    from funhouse_agent import plot_box
    if not plot_box.available():
        pytest.skip("the plotting box cannot run here")
    ex = example("settlement_stress_profile")
    ex["data"]["preconsolidation"]["design"] = {
        "label": "Assumed for\ndesign", "elevation": [16.5, 14.5, 6.5],
        "kpa": [100, 82, 128]}
    out = plot_box.run_plot_code(ex["code"], data=ex["data"],
                                 root=str(tmp_path))
    assert out["status"] == "success", out.get("error")
    assert not out.get("layout_check"), out["layout_check"]


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------

def test_figure_examples_tool():
    tool = make_figure_examples_tool()
    listing = json.loads(tool.invoke({}))
    assert [e["name"] for e in listing["examples"]] == example_names()
    assert {e["kind"] for e in listing["examples"]} == {"plot_data call"}
    assert "plot_code" not in listing["next"]
    one = json.loads(tool.invoke({"name": "spt_profile"}))
    assert one["name"] == "spt_profile" and "series" in one["params"]
    # the old name and plain words still find it
    for want in ("depth profile", "depth_profile", "SPT profile"):
        near = json.loads(tool.invoke({"name": want}))
        assert near["name"] == "spt_profile", (want, near)
    bad = json.loads(tool.invoke({"name": "pie_chart"}))
    assert "no example 'pie_chart'" in bad["error"]
    # no plot_code: its scripts are neither listed nor handed out
    assert "plot_code" not in tool.description
    for name in code_example_names():
        got = json.loads(tool.invoke({"name": name}))
        assert got.get("kind") != "plot_code", (name, got)


def test_the_code_examples_come_with_plot_code():
    tool = make_figure_examples_tool(plot_code=True)
    assert "plot_code scripts" in tool.description
    listing = json.loads(tool.invoke({}))
    kinds = {e["name"]: e["kind"] for e in listing["examples"]}
    assert list(kinds) == example_names() + code_example_names()
    assert all(kinds[n] == "plot_code script" for n in code_example_names())
    assert all(kinds[n] == "plot_data call" for n in example_names())
    assert "plot_code script" in listing["next"]
    full = json.loads(tool.invoke({"name": "wall_pressures"}))
    assert full["kind"] == "plot_code"
    assert full["code"] == CODE_EXAMPLES["wall_pressures"]["code"]
    assert full["data"] == json.loads(json.dumps(
        CODE_EXAMPLES["wall_pressures"]["data"]))
    for name in code_example_names():
        # whole in one result: no cap cuts a script in two (deepagents moves
        # a result aside only past 20,000 tokens)
        assert len(tool.invoke({"name": name})) < 12_000, name
    for want, name in (("depth profile", "spt_profile"),
                       ("settlement profile", "settlement_stress_profile"),
                       ("pile design profile", "pile_design_profile"),
                       ("retaining wall pressures", "wall_pressures")):
        got = json.loads(tool.invoke({"name": want}))
        assert got["name"] == name, (want, got.get("name"))


def test_the_helper_draws_looks_and_counts(monkeypatch):
    monkeypatch.setenv(review_flags.PLOT_CODE_ENV, "0")
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
    # the examples tool offers no plot_code scripts without plot_code
    examples_tool = next(t for t in spec["tools"]
                         if t.name == "figure_examples")
    assert "plot_code" not in examples_tool.description
    from funhouse_agent.deep.limits import ModelCallBudgetMiddleware
    budget = [m for m in spec["middleware"]
              if isinstance(m, ModelCallBudgetMiddleware)]
    assert budget and budget[0].run_limit == DEFAULT_FIGURE_MAX_MODEL_CALLS


def test_the_checklist_carries_the_owners_conventions():
    """GM-97: elevation by default, kPa and kN, test points against a
    design line, the wall drawn."""
    text = " ".join(FIGURE_CHECKLIST.split())
    for words in ("A profile has ELEVATION on its vertical axis",
                  "only when no elevations are given",
                  "never depth and elevation on two axes",
                  "Settlement against time increases downward",
                  "stresses and pressures in kPa, forces in kN",
                  "Separate tests are points",
                  "a line for an assumed design profile",
                  "A wall or pile is drawn to scale with its diagrams",
                  "the diagram alone only when the structure is not known"):
        assert words in text, words
    assert "Depth and settlement increase downward" not in text
    from funhouse_agent.deep.figure_helper import (
        FIGURE_HELPER_DESCRIPTION, PLOT_CODE_DESCRIPTION_NOTE)
    assert "against elevation" in FIGURE_HELPER_DESCRIPTION
    assert "one elevation axis" in PLOT_CODE_DESCRIPTION_NOTE
    from funhouse_agent.adapters.profile_figure_adapter import METHOD_INFO
    plot = METHOD_INFO["plot_data"]
    assert "against elevation" in plot["brief"]
    depth = plot["parameters"]["depth_axis"]["description"]
    assert "elevation on y by default" in depth and "DOWNWARD" in depth


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


def test_on_by_default_and_off_with_both_switches_at_0(monkeypatch):
    """GM-227 (owner, GM-92, 2026-10-15: "on everywhere"): the helper is
    built with no variable set, and left out only with both switches at 0
    (plot_code brings it)."""
    for env in (review_flags.FIGURE_HELPER_ENV, review_flags.PLOT_CODE_ENV):
        monkeypatch.delenv(env, raising=False)
    assert review_flags.figure_helper() and review_flags.plot_code()
    names = {s["name"] for s in _captured_subagents(monkeypatch)}
    assert FIGURE_HELPER_NAME in names
    monkeypatch.setenv(review_flags.FIGURE_HELPER_ENV, "0")
    names = {s["name"] for s in _captured_subagents(monkeypatch)}
    assert FIGURE_HELPER_NAME in names
    monkeypatch.setenv(review_flags.PLOT_CODE_ENV, "0")
    names = {s["name"] for s in _captured_subagents(monkeypatch)}
    assert FIGURE_HELPER_NAME not in names
    assert review_flags.ARMS["figure_helper"] == {
        review_flags.FIGURE_HELPER_ENV: "1"}
    assert review_flags.ARMS["figure_helper_off"] == {
        review_flags.FIGURE_HELPER_ENV: "0", review_flags.PLOT_CODE_ENV: "0"}
    with review_flags.switches(review_flags.ARMS["figure_helper_off"]):
        assert not review_flags.figure_helper()
        assert not review_flags.plot_code()
    with review_flags.switches(review_flags.ARMS["baseline"]):
        assert review_flags.figure_helper() and review_flags.plot_code()
        described = review_flags.describe()
    assert "GEOTECH_FIGURE_HELPER=1 (default)" in described
    assert "GEOTECH_PLOT_CODE=1 (default)" in described


def test_plot_code_brings_the_helper_and_says_so(monkeypatch):
    """Live wave w8 (F55): with plot_code on the calc helper only, the main
    agent drew a wall with plot_data five times and ran out of steps. The
    plot_code switch brings the figure helper, whose description (what the
    main agent reads) says it draws in code what plot_data cannot."""
    monkeypatch.delenv(review_flags.FIGURE_HELPER_ENV, raising=False)
    monkeypatch.setenv(review_flags.PLOT_CODE_ENV, "1")
    specs = {s["name"]: s for s in _captured_subagents(
        monkeypatch, enable_calc_subagent=True)}
    fig = specs[FIGURE_HELPER_NAME]
    assert "plot_code" in {t.name for t in fig["tools"]}
    assert "draws in code (plot_code)" in fig["description"]
    assert "plot_code" in specs["calc"]["description"]
    # and its examples tool now lists the worked plot_code scripts
    examples_tool = next(t for t in fig["tools"]
                         if t.name == "figure_examples")
    listing = json.loads(examples_tool.invoke({}))
    names = [e["name"] for e in listing["examples"]]
    assert set(code_example_names()) <= set(names)
    assert "figure_examples lists worked plot_code scripts" in \
        fig["system_prompt"]


def test_the_switch_adds_it_with_the_guards(monkeypatch):
    monkeypatch.setenv(review_flags.FIGURE_HELPER_ENV, "1")
    specs = _captured_subagents(monkeypatch)
    fig = next(s for s in specs if s["name"] == FIGURE_HELPER_NAME)
    kinds = {type(m).__name__ for m in fig["middleware"]}
    assert {"ModelCallBudgetMiddleware", "ToolErrorGuard",
            "OutputLimitGuard", "AgentBusyRetry"} <= kinds
    assert "LOOKS" in fig["description"]
