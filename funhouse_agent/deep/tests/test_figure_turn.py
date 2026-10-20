"""A figure is one thing across a turn, whoever draws it (BACKLOG GM-151).

``module_work/live_smoke/runs/w12-figures-judgment/TRACE_REVIEW.md``: in
F55j and F65 the main agent sent a SECOND helper to fix one flaw in a
figure, saved as ``<name>_v2``. The second helper could not retire the
first's file (two cards), its draw cap started again, and with no plotting
code it rebuilt the figure from a look at the old PNG; a calc helper's
package figures went out with no look (F18j). Here: a draw by any helper in
the turn counts on the figure's draws and can retire its earlier file; the
code that drew each figure is kept beside it, out of sight; a brief that
names a figure is handed that code; the calc helper is asked to look at the
figures a module method drew. Every plot_code draw is a real child process.
"""

import json
import os

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("matplotlib")

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from funhouse_agent import plot_box, review_flags  # noqa: E402
from funhouse_agent._fileio import working_dir_bound  # noqa: E402
from funhouse_agent.deep import keep_work  # noqa: E402
from funhouse_agent.deep.figure_helper import (  # noqa: E402
    CALC_REDRAW_NOTE, FIGURE_HELPER_DESCRIPTION, FIGURE_HELPER_NAME,
    FigureLooks, FigureTasks, looks_report)
from funhouse_agent.plot_box import figures  # noqa: E402

DRAW = ("import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\n"
        "ax.plot(data['x'], data['y'])\nax.set_xlabel('time (days)')\n"
        "save_figure(fig, {name!r})\n")
DATA = {"x": [0, 10, 45], "y": [0, 12, 30]}
SERIES = [{"x": [0, 10, 45], "y": [0, 12, 30], "label": "SP-1"}]


def _no_switches(monkeypatch):
    """Every switch off: unset, and those on by default set to 0 (GM-226,
    GM-227)."""
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    for env in review_flags.DEFAULT_ON_ENVS:
        monkeypatch.setenv(env, "0")


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """The conversation's working folder, with the plot-code switch on."""
    _no_switches(monkeypatch)
    monkeypatch.setenv(review_flags.PLOT_CODE_ENV, "1")
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    with working_dir_bound(str(tmp_path)):
        yield tmp_path


class _Request:
    """The parts of a tool-call request the middleware reads."""

    def __init__(self, call):
        self.tool_call = call

    def override(self, **kw):
        return _Request(kw.get("tool_call", self.tool_call))


def _in_helper(name, fn, task_id="t"):
    """Run ``fn(record)`` as a helper ``name`` would, inside the turn."""
    rec = keep_work.HelperRecord(task_id, name, "x")
    token = keep_work._HELPER.set(rec)
    try:
        return fn(rec)
    finally:
        keep_work._HELPER.reset(token)


def _plot_code(rec, name, replaces=None, call_id="d"):
    tool = plot_box.make_plot_code_tool()
    args = {"code": DRAW.format(name=name), "data": DATA}
    if replaces:
        args["replaces"] = replaces
    out = tool.invoke(args)
    rec.add_model(AIMessage(content="", tool_calls=[
        {"name": "plot_code", "id": call_id, "args": args}]))
    rec.add_result("plot_code", call_id, out)
    return json.loads(out)


def _plot_data(rec, name, call_id="p"):
    from funhouse_agent.deep.tools import make_vision_tools
    (tool,) = make_vision_tools(include={"plot_data"})
    out = tool.invoke({"series": SERIES, "output_path": name})
    rec.add_result("plot_data", call_id, out)
    return json.loads(out)


# ---------------------------------------------------------------------------
# fix 1: the figure belongs to the turn, whoever draws it
# ---------------------------------------------------------------------------

def test_a_second_helper_in_the_turn_retires_the_first_helpers_figure(
        folder):
    """F65: helper 1 drew the figure with plot_data; the helper sent to fix
    it draws ``_v2`` in code and names the first in replaces: one card."""
    with keep_work.turn_scope():
        first = _in_helper(FIGURE_HELPER_NAME,
                           lambda r: _plot_data(r, "gw_rain.png"), "t1")
        assert first["status"] == "success" and first["draw"] == 1
        assert (folder / "gw_rain.png").exists()
        second = _in_helper(FIGURE_HELPER_NAME, lambda r: _plot_code(
            r, "gw_rain_v2", replaces=["gw_rain"]), "t2")
    assert second["status"] == "success", second
    assert not (folder / "gw_rain.png").exists()
    assert not (folder / "gw_rain.plotly.json").exists()
    assert set(second["retired"]) >= {"gw_rain.png"}
    # it counts on from the first helper's draw, not from one again
    assert second["draw"] == 2 and second["continues"] == ["gw_rain.png"]
    assert "draw 2 of at most 3 of this figure this turn" in second["next"]


def test_the_draw_cap_counts_per_figure_across_the_turn(folder):
    """The cap is the figure's: the calc helper's third draw of a figure the
    figure helper drew twice is told to stop; its new figure is not."""
    with keep_work.turn_scope():
        def figure_helper(rec):
            return [_plot_code(rec, "wall", call_id=f"d{k}")
                    for k in (1, 2)]
        drawn = _in_helper(FIGURE_HELPER_NAME, figure_helper, "t1")
        assert [d["draw"] for d in drawn] == [1, 2]

        def calc(rec):
            return (_plot_code(rec, "wall", call_id="c1"),
                    _plot_code(rec, "moments", call_id="c2"))
        again, other = _in_helper("calc", calc, "t2")
    assert again["draw"] == 3
    assert "Do not draw again" in again["next"]
    assert "by any helper" in again["next"]
    assert other["draw"] == 1 and "Do not draw again" not in other["next"]


def test_outside_a_turn_the_helper_run_keeps_the_count(folder):
    """A host that opens no turn: the figures are booked on the helper."""
    def run(rec):
        a = _plot_code(rec, "fig", call_id="a")
        b = _plot_code(rec, "fig_v2", replaces=["fig.png"], call_id="b")
        return a, b, rec.figures
    a, b, book = _in_helper("calc", run)
    assert (a["draw"], b["draw"]) == (1, 2)
    assert "fig_v2" in book and not (folder / "fig.png").exists()


# ---------------------------------------------------------------------------
# fix 2: the code is kept beside each figure
# ---------------------------------------------------------------------------

def test_the_code_and_data_are_kept_in_the_hidden_folder(folder):
    with keep_work.turn_scope():
        out = _in_helper(FIGURE_HELPER_NAME,
                         lambda r: _plot_code(r, "plates"))
    (img,) = out["images"]
    assert img["code_file"] == ".figure_code/plates.plot.py"
    kept = figures.kept_code("plates.png", str(folder))
    assert kept["code"] == DRAW.format(name="plates")
    assert json.loads(kept["data"]) == DATA
    assert kept["figure"] == "plates.png"
    # plot_data keeps its arguments; a later draw in code replaces them
    with keep_work.turn_scope():
        res = _in_helper("calc", lambda r: _plot_data(r, "plan.png"))
        assert res["code_file"] == ".figure_code/plan.plot_data.json"
        args = figures.kept_code("plan", str(folder))["arguments"]
        assert args["series"] == SERIES
        _in_helper("calc", lambda r: _plot_code(r, "plan"))
    names = sorted(os.listdir(folder / ".figure_code"))
    assert "plan.plot_data.json" not in names and "plan.plot.py" in names


def test_a_retired_figures_code_goes_with_it(folder):
    with keep_work.turn_scope():
        _in_helper(FIGURE_HELPER_NAME, lambda r: _plot_code(r, "draft"))
        assert figures.kept_code("draft", str(folder)) is not None
        _in_helper(FIGURE_HELPER_NAME, lambda r: _plot_code(
            r, "final", replaces=["draft.png"]))
    assert figures.kept_code("draft", str(folder)) is None
    assert figures.kept_code("final", str(folder)) is not None


def test_the_code_is_not_among_the_files_the_model_is_shown(folder):
    """Neither the names the prompt lists nor list_files shows it."""
    from funhouse_agent.vision_tools import (dispatch_extended_tool,
                                             working_folder_names)
    with keep_work.turn_scope():
        _in_helper(FIGURE_HELPER_NAME, lambda r: _plot_code(r, "plates"))
    shown = " ".join(working_folder_names())
    assert "plates.png" in shown and ".figure_code" not in shown
    for depth in (1, 3):
        listed = dispatch_extended_tool(
            "list_files", {"path": ".", "depth": depth}, engine=None,
            attachments={})
        assert "plates.png" in listed, listed
        assert ".figure_code" not in listed and ".plot.py" not in listed \
            and ".data.json" not in listed, listed


def test_a_module_redraw_drops_the_code_it_did_not_run(folder):
    """A figure drawn in code, then drawn again by a module method under
    the same name: the kept code no longer drew what the file shows."""
    with keep_work.turn_scope():
        _in_helper("calc", lambda r: _plot_code(r, "wall"))
        assert figures.kept_code("wall", str(folder)) is not None
        _in_helper("calc", lambda r: FigureLooks().wrap_tool_call(
            _call_agent("calc_package", "render_figures"),
            _answer({"status": "success", "figures": [
                {"index": 1, "output_path": "wall.png"}]})))
    assert figures.kept_code("wall", str(folder)) is None


# ---------------------------------------------------------------------------
# fix 3: a redraw brief carries the code; a name alone removes nothing
# ---------------------------------------------------------------------------

def _task(helper, brief, task_id="task_9"):
    return {"name": "task", "id": task_id, "args": {
        "description": brief, "subagent_type": helper}}


def test_a_redraw_brief_carries_the_code(folder):
    """F55j: the brief to fix a figure names it and asks for ``_v2``; the
    helper is handed the code that drew it, and told to keep the name."""
    seen = {}

    def helper(request):
        seen["brief"] = request.tool_call["args"]["description"]
        return ToolMessage(content="done", name="task",
                           tool_call_id=request.tool_call["id"])

    with keep_work.turn_scope():
        _in_helper("calc", lambda r: _plot_code(r, "cantilever_wall"))
        brief = ("The pressure labels overlap on cantilever_wall.png. "
                 "Redraw it with the labels moved; save under the new name "
                 "cantilever_wall_v2.png.")
        for name in ("calc", FIGURE_HELPER_NAME):
            keep_work.KeepWork().wrap_tool_call(
                _Request(_task(name, brief)),
                lambda r: FigureTasks().wrap_tool_call(r, helper))
            got = seen["brief"]
            assert got.startswith(brief)
            assert figures.REDRAW_HEADING + ": cantilever_wall.png" in got
            assert DRAW.format(name="cantilever_wall").strip() in got
            assert '"x": [0, 10, 45]' in got                    # its data
            assert "whatever name the brief gives" in got
            assert "drawn 1 time(s) this turn" in got
            assert "unless the draw names it in replaces=" in got
        # a brief naming no figure, or a helper that draws none: unchanged
        for name, text in (("calc", "Check the wall's embedment."),
                           ("general-purpose", brief)):
            keep_work.KeepWork().wrap_tool_call(
                _Request(_task(name, text)), lambda r: FigureTasks(
                    ).wrap_tool_call(r, helper))
            assert seen["brief"] == text


def test_a_brief_naming_only_the_new_version_still_gets_the_code(folder):
    with keep_work.turn_scope():
        _in_helper(FIGURE_HELPER_NAME, lambda r: _plot_code(r, "gw_panels"))
    stems = figures.named_in("Save the fix as gw_panels_v2.png.",
                             str(folder))
    assert stems == ["gw_panels"]
    # a version kept in its own right is the one meant
    with keep_work.turn_scope():
        _in_helper(FIGURE_HELPER_NAME, lambda r: _plot_code(
            r, "gw_panels_v2"))
    assert figures.named_in("Fix gw_panels_v2.png.", str(folder)) == \
        ["gw_panels_v2"]
    # a word that is no figure's name is not a figure
    assert figures.named_in("the panels look fine", str(folder)) == []


def test_a_keep_both_brief_keeps_both_figures(folder):
    """The review's probe: a brief that asks for a second option beside a
    figure, under a name that reads as a version, keeps both files; the
    draws count on and the result says the first still shows."""
    def helper(request):
        drawn = _plot_code(keep_work.current_helper(), "wall_pressure_v2",
                           call_id="r1")
        assert "retired" not in drawn and drawn["draw"] == 2, drawn
        return ToolMessage(content="wall_pressure_v2.png drawn", name="task",
                           tool_call_id=request.tool_call["id"])

    with keep_work.turn_scope():
        _in_helper(FIGURE_HELPER_NAME,
                   lambda r: _plot_code(r, "wall_pressure"), "t1")
        out = keep_work.KeepWork().wrap_tool_call(
            _Request(_task(FIGURE_HELPER_NAME,
                           "Draw option B beside wall_pressure.png (keep "
                           "both), with the surcharge doubled; save as "
                           "wall_pressure_v2.png.")),
            lambda r: FigureTasks().wrap_tool_call(r, helper))
    assert (folder / "wall_pressure.png").exists()
    assert (folder / "wall_pressure_v2.png").exists()
    assert "wall_pressure_v2.png in .figure_code/wall_pressure_v2.plot.py" \
        in out.content
    assert "name it in the next helper's brief" in out.content


def test_a_redraw_under_another_name_retires_only_by_replaces(folder):
    with keep_work.turn_scope():
        _in_helper(FIGURE_HELPER_NAME, lambda r: _plot_code(r, "gw"), "t1")
        out = _in_helper(FIGURE_HELPER_NAME, lambda r: _plot_code(
            r, "gw_v2", replaces=["gw.png"]), "t2")
    assert out["retired"] == ["gw.png"] and out["draw"] == 2
    assert not (folder / "gw.png").exists()


def test_a_boring_named_v_1_is_not_a_version():
    assert figures.version_base("plan_v2") == "plan"
    assert figures.version_base("plan-rev3") == "plan"
    assert figures.version_base("plan version 2") == "plan"
    for name in ("boring_V-1", "boring", "sheet_10", "B-12", "case 3"):
        assert figures.version_base(name) is None, name


def test_boring_v_1_and_boring_are_two_figures(folder):
    with keep_work.turn_scope():
        _in_helper("calc", lambda r: _plot_code(r, "boring"), "t1")
        out = _in_helper("calc", lambda r: _plot_code(r, "boring_V-1"), "t2")
    assert out["draw"] == 1 and "continues" not in out
    assert not any("earlier version" in n for n in out.get("notes") or [])
    assert (folder / "boring.png").exists()
    assert figures.named_in("Redraw boring_V-1.png.", str(folder)) == \
        ["boring_V-1"]


def test_a_version_no_brief_named_is_left_and_said(folder):
    with keep_work.turn_scope():
        _in_helper(FIGURE_HELPER_NAME, lambda r: _plot_code(r, "sp"), "t1")
        out = _in_helper(FIGURE_HELPER_NAME,
                         lambda r: _plot_code(r, "sp_v2"), "t2")
    assert (folder / "sp.png").exists()
    assert out["draw"] == 2
    assert any("sp.png, an earlier version of this figure" in n
               for n in out["notes"])


def test_the_descriptions_say_how_to_change_a_figure():
    for text in (FIGURE_HELPER_DESCRIPTION, CALC_REDRAW_NOTE):
        text = " ".join(text.split())
        assert "that name" in text and "redraws it in place" in text
        assert "handed the code that drew the figure" in text
    desc = " ".join(plot_box.plot_code_description().split())
    assert "The code that drew each figure is kept beside it" in desc
    assert "by any helper" in desc


# ---------------------------------------------------------------------------
# fix 4 (TRACE_REVIEW fix 7): the calc helper looks at its figures
# ---------------------------------------------------------------------------

def _call_agent(agent, method, params=None, call_id="m1"):
    return _Request({"name": "call_agent", "id": call_id, "args": {
        "agent_name": agent, "method": method,
        "parameters": params or {}}})


def _answer(payload, call_id="m1"):
    return lambda request: ToolMessage(content=json.dumps(payload),
                                       name="call_agent",
                                       tool_call_id=call_id)


def test_a_module_figure_asks_the_calc_helper_for_a_look(
        folder, tmp_path_factory, monkeypatch):
    """A module names the file relative to the working folder, and the
    server's own folder is elsewhere (the review: no note ever fired)."""
    (folder / "profile.png").write_bytes(b"png")
    monkeypatch.chdir(tmp_path_factory.mktemp("server_cwd"))
    params = {"series": SERIES, "output_path": "profile.png"}
    with keep_work.turn_scope():
        out = _in_helper("calc", lambda r: FigureLooks().wrap_tool_call(
            _call_agent("profile_figure", "plot_data", params),
            _answer({"status": "success", "output_path": "profile.png"})))
    assert out.content.startswith('{"status": "success"')
    assert "LOOK at profile.png with analyze_image" in out.content
    assert "what it SHOWS against the data" in out.content
    assert "draw 1 of at most 3 of this figure this turn" in out.content
    assert ".figure_code/profile.plot_data.json" in out.content
    assert figures.kept_code("profile", str(folder))["arguments"] == params
    # an error, or a result with no picture, is left as it is
    with keep_work.turn_scope():
        plain = _in_helper("calc", lambda r: FigureLooks().wrap_tool_call(
            _call_agent("retaining_wall", "check"),
            _answer({"status": "success", "fs_sliding": 1.6})))
    assert "LOOK" not in plain.content


def test_a_module_figure_sent_as_a_json_string_keeps_its_arguments(
        folder, tmp_path_factory, monkeypatch):
    """GM-189: 'parameters' sent as a string holding the object runs as
    that object, and its figure keeps those arguments too."""
    (folder / "profile.png").write_bytes(b"png")
    monkeypatch.chdir(tmp_path_factory.mktemp("server_cwd"))
    params = {"series": SERIES, "output_path": "profile.png"}
    with keep_work.turn_scope():
        _in_helper("calc", lambda r: FigureLooks().wrap_tool_call(
            _call_agent("profile_figure", "plot_data", json.dumps(params)),
            _answer({"status": "success", "output_path": "profile.png"})))
    assert figures.kept_code("profile", str(folder))["arguments"] == params


def test_html_to_pdf_names_the_figures_no_look_read(folder):
    a, b = folder / "a.png", folder / "b.png"
    for p in (a, b):
        p.write_bytes(b"png")
    pdf = {"status": "success", "output_path": str(folder / "calc.pdf"),
           "embedded_images": [str(a), str(b)]}

    def run(rec):
        rec.add_model(AIMessage(content="", tool_calls=[
            {"name": "analyze_image", "id": "l1",
             "args": {"attachment_key": "a.png"}}]))
        rec.add_result("analyze_image", "l1", json.dumps(
            {"analysis": "reads 12 kPa at 3 m"}))
        return FigureLooks().wrap_tool_call(
            _call_agent("calc_package", "html_to_pdf"), _answer(pdf))
    out = _in_helper("calc", run)
    assert "FIGURES NOT LOOKED AT: this PDF embeds b.png," in out.content
    assert "a.png" not in out.content.split("FIGURES NOT LOOKED AT")[1]
    assert "build the PDF again" in out.content


def test_the_calc_helpers_record_names_its_figures_and_their_code():
    """The record holds the method's result as delivered: with FigureLooks'
    note after it, read past."""
    rec = keep_work.HelperRecord("t", "calc", "x")
    rec.add_result("call_agent", "m1", json.dumps({
        "status": "success", "figures": [
            {"index": 1, "output_path": "/conv/files/pressures.png"}]})
        + "\n\nLOOK at pressures.png with analyze_image {soon}")
    rec.add_result("plot_code", "d1", json.dumps({
        "status": "success", "images": [
            {"file": "wall.png",
             "code_file": ".figure_code/wall.plot.py"}]}))
    note = looks_report(rec)
    assert note.startswith("FIGURE RECORD of the calc helper")
    assert "drew 2 time(s) (call_agent 1, plot_code 1) and looked 0" in note
    assert "Not looked at since its last draw: pressures.png, wall.png." \
        in note
    assert "wall.png in .figure_code/wall.plot.py" in note


# ---------------------------------------------------------------------------
# the switches
# ---------------------------------------------------------------------------

def _built(monkeypatch, **kw):
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    import funhouse_agent.deep.agent as agent_mod

    class _Quiet(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "quiet"

        def bind_tools(self, tools, **k):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **k):
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="done"))])

    captured = {}
    real = agent_mod.create_deep_agent

    def spy(**k):
        captured.update(k)
        return real(**k)
    monkeypatch.setattr(agent_mod, "create_deep_agent", spy)
    agent_mod.build_deep_agent(model=_Quiet(), **kw)
    return captured


def test_the_calc_helper_gets_its_looks_only_behind_the_switch(monkeypatch):
    for on in (False, True):
        _no_switches(monkeypatch)
        if on:
            monkeypatch.setenv(review_flags.PLOT_CODE_ENV, "1")
        got = _built(monkeypatch, reference_mode="off",
                     enable_calc_subagent=True)
        calc = next(s for s in got["subagents"] if s["name"] == "calc")
        kinds = [type(m).__name__ for m in calc.get("middleware") or []]
        assert ("FigureLooks" in kinds) is on
        assert (CALC_REDRAW_NOTE in calc["description"]) is on
        if on:
            # inside the delivered log, so activity.jsonl shows its notes
            assert kinds.index("FigureLooks") == \
                kinds.index("DeliveredToolResults") + 1
