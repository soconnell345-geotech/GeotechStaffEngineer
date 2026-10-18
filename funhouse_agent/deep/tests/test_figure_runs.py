"""The figure helper after live wave w9 (BACKLOG GM-107).

``module_work/live_smoke/runs/w9-figures/TRACE_REVIEW.md``: looks were
asked about the lettering only (fix 3); briefs prescribed styling and put
standards nobody gave into figures (fix 4); a helper drew six times
against a cap of three (fix 7); the main agent looked again at figures the
helper had looked at, twice telling the user something false (fix 8); a
reversed range was sorted without a word (fix 9); the main prompt never
named the helper (fix 11); no worked example was ever fetched (fix 13).

Everything new for the agent stays behind the figure switches: with them
off, the main agent's prompt and tools are as they were.
"""

import json

import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import (AIMessage, HumanMessage,  # noqa: E402
                                     ToolMessage)
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from funhouse_agent import review_flags  # noqa: E402
from funhouse_agent._fileio import working_dir_bound  # noqa: E402
from funhouse_agent.deep import keep_work  # noqa: E402
from funhouse_agent.deep.figure_helper import (  # noqa: E402
    EXAMPLE_HEADING, FIGURE_CHECKLIST, FIGURE_HELPER_DESCRIPTION,
    FIGURE_HELPER_NAME, FIGURE_HELPER_PROMPT, FIGURE_PRIMARY_NOTE,
    FigureTasks, brief_with_example, build_figure_subagent, looks_report,
    nearest_example)


class _Quiet(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "quiet"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="done"))])


def _no_switches(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)


def _built(monkeypatch, **kw):
    import funhouse_agent.deep.agent as agent_mod
    captured = {}
    real = agent_mod.create_deep_agent

    def spy(**k):
        captured.update(k)
        return real(**k)
    monkeypatch.setattr(agent_mod, "create_deep_agent", spy)
    agent_mod.build_deep_agent(model=_Quiet(), **kw)
    return captured


def _schema_props(tool) -> set:
    return set(tool.tool_call_schema.model_json_schema().get("properties")
               or {})


# ---------------------------------------------------------------------------
# the switches off: the main agent as it was
# ---------------------------------------------------------------------------

def test_switches_off_leave_the_default_agent_unchanged(monkeypatch):
    """With GEOTECH_FIGURE_HELPER and GEOTECH_PLOT_CODE off the main
    agent's prompt is exactly the domain prompt (plus the calc nudge when
    the calc helper is on), the calc helper's prompt exactly its own, and no
    tool or helper carries anything GM-107 added (plot_data's x_on_top and
    axis ranges are fix 9, a bug fix that applies by default)."""
    from funhouse_agent.deep import agent as agent_mod
    from funhouse_agent.deep.prompt import build_domain_prompt
    from funhouse_agent.dispatch import ANALYSIS_MODULES
    _no_switches(monkeypatch)
    for calc in (False, True):
        got = _built(monkeypatch, reference_mode="anytime",
                     enable_calc_subagent=calc)
        want = build_domain_prompt(ANALYSIS_MODULES, memory_enabled=False)
        if calc:
            want += "\n\n" + agent_mod._CALC_DELEGATION_NUDGE
        assert got["system_prompt"] == want
        assert not any(isinstance(m, FigureTasks)
                       for m in got["middleware"])
        names = {s["name"] for s in got["subagents"]}
        assert FIGURE_HELPER_NAME not in names
        tools = list(got["tools"]) + [t for s in got["subagents"]
                                      for t in s.get("tools") or []]
        assert "plot_code" not in {t.name for t in tools}
        for t in tools:
            assert "replaces" not in _schema_props(t), t.name
            assert "look_file" not in (t.description or ""), t.name
        if calc:
            spec = next(s for s in got["subagents"] if s["name"] == "calc")
            assert spec["system_prompt"] == (agent_mod._CALC_PREAMBLE
                                             + agent_mod._CALC_FRAMING)


def test_switches_off_plot_data_counts_per_name_as_before(tmp_path,
                                                         monkeypatch):
    from funhouse_agent.deep.tools import make_vision_tools
    _no_switches(monkeypatch)
    (tool,) = make_vision_tools(include={"plot_data"})
    rec = keep_work.HelperRecord("t", "general-purpose", "x")
    for k in range(3):
        rec.add_result("plot_code", f"c{k}", json.dumps({"status":
                                                         "success"}))
    token = keep_work._HELPER.set(rec)
    try:
        with working_dir_bound(str(tmp_path)):
            out = json.loads(tool.invoke({"series": [
                {"x": [0, 1], "y": [0, 1], "label": "a"}],
                "output_path": "one.png"}))
    finally:
        keep_work._HELPER.reset(token)
    assert out["status"] == "success"
    assert "draw" not in out and "redraw_note" not in out


# ---------------------------------------------------------------------------
# the switches on
# ---------------------------------------------------------------------------

def test_the_switch_names_the_helper_and_puts_its_middleware_inside_keepwork(
        monkeypatch):
    _no_switches(monkeypatch)
    monkeypatch.setenv(review_flags.FIGURE_HELPER_ENV, "1")
    got = _built(monkeypatch, reference_mode="off")
    assert got["system_prompt"].endswith("\n\n" + FIGURE_PRIMARY_NOTE)
    kinds = [type(m).__name__ for m in got["middleware"]]
    assert kinds.index("FigureTasks") == kinds.index("KeepWork") + 1
    fig = next(s for s in got["subagents"] if s["name"] == FIGURE_HELPER_NAME)
    plot = next(t for t in fig["tools"] if t.name == "plot_data")
    assert {"replaces", "x_on_top"} <= _schema_props(plot)
    assert "replaces:" in plot.description
    # the main agent's own redraw can retire a figure a helper drew this
    # turn (GM-151)
    main_plot = next(t for t in got["tools"] if t.name == "plot_data")
    assert "replaces" in _schema_props(main_plot)
    assert "by any helper" in main_plot.description
    for words in ("rather than drawing a draft yourself",
                  "do not look again at a figure it has looked at",
                  "ask the helper (it has the code)",
                  "put nothing in the brief that the user did not give"):
        assert words in FIGURE_PRIMARY_NOTE, words


def test_the_checklist_and_the_looks_ask_about_content():
    text = " ".join(FIGURE_CHECKLIST.split())
    assert "joined by straight lines or shown as points, never by a stair " \
        "or step unless the data is a step" in text
    assert "no test standard, classification, boundary or citation added " \
        "from memory" in text
    prompt = " ".join(FIGURE_HELPER_PROMPT.split())
    for words in ("read two or three plotted values",
                  "the checklist governs the styling",
                  "follow the checklist and say so",
                  "three draws in all whatever the tool or file name",
                  "replaces=", EXAMPLE_HEADING.lower().split()[0]):
        assert words.lower() in prompt.lower(), words
    desc = " ".join(FIGURE_HELPER_DESCRIPTION.split())
    assert "redraws until it passes" not in desc
    assert "Leave the styling to it" in desc
    assert "no test standard, classification or citation" in desc
    assert "what each of its looks found" in desc


# ---------------------------------------------------------------------------
# fix 13: the nearest worked example goes with the brief
# ---------------------------------------------------------------------------

def test_the_nearest_example_is_picked_from_the_brief():
    assert nearest_example("Plot the settlement of the plates against "
                           "time since filling began") == "time_settlement"
    assert nearest_example("A site plan of the borings and the building "
                           "footprint") == "plan_borings"
    assert nearest_example("the SPT N60 profile of B-1 to B-3") == \
        "spt_profile"
    # the code examples only where the helper has plot_code
    wall = "a cantilever wall with its earth pressures beside it"
    assert nearest_example(wall) is None
    assert nearest_example(wall, with_code=True) == "wall_pressures"
    assert nearest_example("factor of safety for the five cases of the "
                           "sensitivity study") == "cases"
    # a figure no example is like gets none, and a one-word name (a word
    # as common in a brief as 'case') needs more than itself
    mohr = ("Draw Mohr circles in effective stress at failure for the three "
            "CU triaxial tests on sample B-2 S-4, with the failure envelope; "
            "quote the result (c' and phi'); the governing case is test 2.")
    for brief in ("Mohr circles at failure", "a bar chart of the costs",
                  "the governing load case", mohr):
        assert nearest_example(brief, with_code=True) is None, brief
        assert brief_with_example(brief, with_code=True) == brief


def test_the_brief_carries_the_examples_reasons_once():
    brief = "Draw the settlement of the plates against time."
    out = brief_with_example(brief)
    assert out.startswith(brief) and EXAMPLE_HEADING in out
    assert "'time_settlement'" in out and "figure_examples(name=" in out
    assert "no legend" in out                     # one of its reasons
    assert brief_with_example(out) == out         # not twice
    assert brief_with_example("Mohr circles") == "Mohr circles"


# ---------------------------------------------------------------------------
# fix 8: the helper's looks come back from the record
# ---------------------------------------------------------------------------

def _record():
    rec = keep_work.HelperRecord("t1", FIGURE_HELPER_NAME, "draw it")
    rec.add_model(AIMessage(content="", tool_calls=[
        {"name": "plot_data", "id": "d1", "args": {}}]))
    rec.add_result("plot_data", "d1", json.dumps(
        {"status": "success", "output_path": "/conv/files/draft.png"}))
    rec.add_model(AIMessage(content="", tool_calls=[
        {"name": "analyze_image", "id": "l1",
         "args": {"attachment_key": "draft.png", "prompt": "values?"}}]))
    rec.add_result("analyze_image", "l1", json.dumps(
        {"analysis": "The curve reads 12 mm at day 10. " + "x" * 600}))
    rec.add_model(AIMessage(content="", tool_calls=[
        {"name": "plot_code", "id": "d2", "args": {}}]))
    rec.add_result("plot_code", "d2", json.dumps(
        {"status": "success", "images": [
            {"file": "final.png", "look_file": ".scratch/final_look.png"}]}))
    rec.add_model(AIMessage(content="", tool_calls=[
        {"name": "analyze_image", "id": "l2",
         "args": {"attachment_key": ".scratch/final_look.png"}}]))
    rec.add_result("analyze_image", "l2", json.dumps(
        {"analysis": {"answer": "labels clear"}}))
    rec.add_result("plot_code", "d3", json.dumps(
        {"status": "success", "images": [{"file": "final.png"}]}))
    rec.add_result("plot_code", "bad", json.dumps({"status": "error"}))
    return rec


def test_the_looks_report_is_written_from_the_record():
    note = looks_report(_record())
    assert note.startswith("FIGURE HELPER RECORD")
    assert "drew 3 time(s) (plot_code 2, plot_data 1) and looked 2" in note
    assert "look 1 at draft.png, after draw 1: The curve reads 12 mm" in note
    assert "look 2 at final.png, after draw 2" in note     # its look copy
    assert '"answer": "labels clear"' in note
    assert "Not looked at since its last draw: final.png." in note
    assert len(note) < 1500                           # answers are cut
    assert looks_report(keep_work.HelperRecord("t", "figure", "")) == ""


class _Request:
    """The parts of a tool-call request the middleware reads."""

    def __init__(self, call):
        self.tool_call = call

    def override(self, **kw):
        return _Request(kw.get("tool_call", self.tool_call))


def test_figure_tasks_inside_keepwork_briefs_and_reports(tmp_path):
    seen = {}

    def helper(request):
        # what the figure helper is handed, and what it does
        seen["brief"] = request.tool_call["args"]["description"]
        rec = keep_work.current_helper()
        for step in _record().steps:
            rec.steps.append(step)
        return ToolMessage(content="plates.png drawn", name="task",
                           tool_call_id=request.tool_call["id"])

    call = {"name": "task", "id": "task_1", "args": {
        "description": "Draw settlement of the plates against time.",
        "subagent_type": FIGURE_HELPER_NAME}}
    inner = FigureTasks()
    with working_dir_bound(str(tmp_path)):
        out = keep_work.KeepWork().wrap_tool_call(
            _Request(call), lambda r: inner.wrap_tool_call(r, helper))
    assert EXAMPLE_HEADING in seen["brief"]
    assert out.content.startswith("plates.png drawn")
    assert "FIGURE HELPER RECORD" in out.content
    # another helper's task is left alone
    other = {"name": "task", "id": "task_2", "args": {
        "description": "Draw settlement against time.",
        "subagent_type": "general-purpose"}}
    with working_dir_bound(str(tmp_path)):
        plain = keep_work.KeepWork().wrap_tool_call(
            _Request(other), lambda r: inner.wrap_tool_call(r, helper))
    assert EXAMPLE_HEADING not in seen["brief"]
    assert "FIGURE HELPER RECORD" not in plain.content


# ---------------------------------------------------------------------------
# through a real compiled agent
# ---------------------------------------------------------------------------

SERIES = [{"x": [0, 10, 45, 90], "y": [0, 12, 30, 41], "label": "SP-1"}]
BRIEF = "Draw the settlement of plates SP-1 and SP-2 against time."
SCRIPT = [
    ("plot_data", {"series": SERIES, "output_path": "draft.png"}),
    ("analyze_image", {"attachment_key": "draft.png",
                       "prompt": "Read two plotted values back."}),
    ("plot_data", {"series": SERIES, "output_path": "plates.png",
                   "replaces": ["draft.png"]}),
    ("analyze_image", {"attachment_key": "plates.png", "prompt": "check"}),
    ("plot_data", {"series": SERIES, "output_path": "plates.png"}),
]


class _Played(BaseChatModel):
    """Plays the main agent (asked ``PRIMARY``) and the figure helper."""

    helper_saw: list = []

    @property
    def _llm_type(self):
        return "played"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        first = next((m.content for m in messages
                      if isinstance(m, HumanMessage)), "")
        step = sum(1 for m in messages if isinstance(m, AIMessage))
        if first.startswith("PRIMARY"):
            reply = (AIMessage(content="", tool_calls=[{
                "name": "task", "id": "task_1",
                "args": {"description": BRIEF,
                         "subagent_type": FIGURE_HELPER_NAME}}])
                if step == 0 else AIMessage(content="All done."))
            return ChatResult(generations=[ChatGeneration(message=reply)])
        self.helper_saw.clear()
        self.helper_saw.append(first)
        self.helper_saw.extend(m.content for m in messages
                               if isinstance(m, ToolMessage))
        if step < len(SCRIPT):
            name, args = SCRIPT[step]
            reply = AIMessage(content="", tool_calls=[
                {"name": name, "id": f"h{step}", "args": args}])
        else:
            reply = AIMessage(content="plates.png: settlement against time.")
        return ChatResult(generations=[ChatGeneration(message=reply)])


class _Eyes:
    accepts_jpeg = False

    def vision_profile(self):
        return None

    def analyze_image(self, image, prompt):
        return "SP-1 reads 12 mm at day 10 and 41 mm at day 90."


def test_a_figure_task_through_the_real_agent(tmp_path, monkeypatch):
    _no_switches(monkeypatch)
    monkeypatch.setenv(review_flags.FIGURE_HELPER_ENV, "1")
    from funhouse_agent.deep.agent import build_deep_agent
    model = _Played()
    agent = build_deep_agent(model=model, engine=_Eyes(),
                             reference_mode="off")
    with working_dir_bound(str(tmp_path)), keep_work.turn_scope():
        state = agent.invoke({"messages": [{"role": "user",
                                            "content": "PRIMARY plates"}]},
                             config={"recursion_limit": 80})
    brief, *results = model.helper_saw
    assert brief.startswith(BRIEF) and "'time_settlement'" in brief
    draws = [json.loads(r) for r in results if '"analysis_type"' in r]
    assert [d.get("draw") for d in draws] == [1, 2, 3]
    assert sorted(draws[1]["retired"]) == ["draft.plotly.json", "draft.png"]
    assert "Do not draw again" in draws[2]["redraw_note"]
    assert not (tmp_path / "draft.png").exists()
    assert (tmp_path / "plates.png").exists()
    (task,) = [m for m in state["messages"]
               if isinstance(m, ToolMessage) and m.name == "task"]
    assert task.content.startswith("plates.png: settlement against time.")
    for words in ("FIGURE HELPER RECORD", "drew 3 time(s) (plot_data 3) "
                  "and looked 2", "look 1 at draft.png, after draw 1: SP-1 "
                  "reads 12 mm", "look 2 at plates.png, after draw 2",
                  "Not looked at since its last draw: plates.png."):
        assert words in task.content, words


def test_plot_data_in_another_helper_is_not_capped(tmp_path, monkeypatch):
    """With the switch on, the main agent's plot_data in a general helper
    (or the calc helper's) draws as many figures as it is asked for: a new
    figure is that figure's first draw (GM-151)."""
    from funhouse_agent.deep.tools import make_vision_tools
    _no_switches(monkeypatch)
    monkeypatch.setenv(review_flags.PLOT_CODE_ENV, "1")
    (tool,) = make_vision_tools(include={"plot_data"})
    series = [{"x": [0, 1], "y": [0, 1], "label": "a"}]
    for helper, capped in (("general-purpose", False), ("calc", False),
                           (FIGURE_HELPER_NAME, True)):
        rec = keep_work.HelperRecord("t", helper, "x")
        for k in range(3):
            rec.add_result("plot_data", f"c{k}",
                           json.dumps({"status": "success"}))
        token = keep_work._HELPER.set(rec)
        try:
            with working_dir_bound(str(tmp_path)):
                out = json.loads(tool.invoke({
                    "series": series, "output_path": f"{helper}_4.png"}))
        finally:
            keep_work._HELPER.reset(token)
        assert out["status"] == "success"
        if capped:
            assert out["draw"] == 4
            assert "Do not draw again" in out["redraw_note"]
        else:
            assert out["draw"] == 1 and "redraw_note" not in out, helper


def test_replaces_on_a_cut_short_result_says_nothing_was_retired(
        tmp_path, monkeypatch):
    from funhouse_agent.deep.tools import make_vision_tools
    _no_switches(monkeypatch)
    (tool,) = make_vision_tools(include={"plot_data"}, max_result_chars=300,
                                plot_data_replaces=True)
    import time
    with working_dir_bound(str(tmp_path)), keep_work.turn_scope():
        time.sleep(0.05)
        (tmp_path / "draft.png").write_bytes(b"x")     # a draft of this turn
        out = tool.invoke({"series": [{"x": [0, 1], "y": [0, 1],
                                       "label": "a"}],
                           "output_path": "final.png",
                           "replaces": ["draft.png"]})
    assert "nothing was retired: this result was cut short" in out
    assert (tmp_path / "draft.png").exists()


def test_plot_data_through_the_tool_takes_x_on_top_and_says_a_reversed_axis(
        tmp_path, monkeypatch):
    _no_switches(monkeypatch)
    spec = build_figure_subagent()
    plot = next(t for t in spec["tools"] if t.name == "plot_data")
    with working_dir_bound(str(tmp_path)):
        out = json.loads(plot.invoke({
            "series": [{"x": [50, 0.075, 0.01], "y": [100, 22, 10],
                        "label": "B-1"}],
            "logx": True, "xlim": [100, 0.001], "x_on_top": True,
            "output_path": "gsd.png"}))
    assert out["status"] == "success", out
    assert "x axis runs high to low" in out["summary"]
    assert "x axis along the top" in out["summary"]


def test_plot_data_through_the_tool_places_notes(tmp_path, monkeypatch):
    """GM-109: the deep agent's plot_data takes annotations, says what they
    are for, and a misspelt note key comes back with a did-you-mean."""
    from funhouse_agent.deep.tools import make_vision_tools
    _no_switches(monkeypatch)
    (tool,) = make_vision_tools(include={"plot_data"})
    assert "annotations" in _schema_props(tool)
    assert "annotations:" in tool.description
    assert "notes placed where you put them" in tool.description
    with working_dir_bound(str(tmp_path)):
        out = json.loads(tool.invoke({
            "series": [{"x": [0, 10, 20], "y": [0, 8, 12], "label": "Test"}],
            "annotations": [{"text": "Yield", "x": 4, "y": 10,
                             "point": [10, 8]}],
            "output_path": "notes.png"}))
        assert out["status"] == "success", out
        assert out["annotations"] == [{"text": "Yield", "at": [4, 10],
                                       "coords": "data", "point": [10, 8]}]
        bad = json.loads(tool.invoke({
            "series": [{"x": [0, 10], "y": [0, 8]}],
            "annotations": [{"text": "Yield", "x": 4, "y": 10,
                             "arrow": [10, 8]}],
            "output_path": "bad.png"}))
    assert bad.get("status") != "success"
    assert "arrow -> point" in json.dumps(bad)
