"""Records as files, the rest (BACKLOG RI-31).

Field session 2026-10-14, turn 3: the primary retyped 29,000 characters of
helper output into ``write_diggs``, and a blow count, a bottom depth, a
printed 60/15" and one exploration's coordinates went wrong on the way. The
first half (2026-10-14) let ``write_diggs`` read records files and named a
helper's saved files to the primary. Two ways round it were left:

* a helper that STOPS before saving (an error, its budget, a short answer
  after much reading) left its records only in its kept ``.md`` -- now the
  records it held are saved as ``kept/<helper>_records.json`` and named in
  the note the primary reads;
* a helper that "saved" with deepagents' ``write_file`` made a scratch file
  ``write_diggs`` cannot read -- now a helper's ``.json`` ``write_file`` is
  saved on the real disk with its own ``save_file``.

Offline: a scripted chat model plays the primary and its helpers through a
real compiled deep agent; the DIGGS writer is the real one.
"""

from __future__ import annotations

import json
import re

import pytest

pytest.importorskip("deepagents")
pytest.importorskip("pydantic")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

from funhouse_agent._fileio import working_dir_bound
from funhouse_agent.deep import held_records, keep_work
from funhouse_agent.deep.agent import build_deep_agent
from funhouse_agent.deep.scratch_guard import ScratchFilesystemGuard


def _m(v, unit="m"):
    return {"value": v, "unit": unit}


def _boring(iid, x):
    return {
        "investigation_id": iid, "kind": "boring", "depth_unit": "m",
        "x": x, "y": 0.0, "elevation": _m(4.6), "total_depth": _m(10.05),
        "layers": [
            {"top": _m(0.0), "bottom": _m(1.5), "description": "Gray poorly "
             "graded GRAVEL, loose, dry (fill)", "uscs": "GP"},
            {"top": _m(1.5), "bottom": _m(10.05), "description": "Brown "
             "poorly graded SAND, loose to dense, wet", "uscs": "SP"}],
        "samples": [{"sample_id": "S-1", "top": _m(1.0),
                     "bottom": _m(1.45), "kind": "spt"}],
        "spt": [{"depth_top": _m(1.0), "depth_bottom": _m(1.45),
                 "blows": [2, 1, 2], "n": 3, "sample_id": "S-1"}],
        "water": [{"depth": _m(3.0), "when": "while_drilling"}],
        "pages": [21],
    }


B1, B2 = _boring("B-1", 0.0), _boring("B-2", 25.0)
LAB = {"kind": "atterberg", "investigation_id": "B-1", "sample_id": "S-1",
       "depth_top": _m(1.0), "result": {"kind": "atterberg", "ll": 32.0,
                                        "pl": 18.0, "pi": 14.0}}

ROW = "B-7 | 3.0 m | N=17 | SM, brown silty sand | w=12% \n"


# ---------------------------------------------------------------------------
# Finding records by their shape
# ---------------------------------------------------------------------------

def test_records_are_found_by_shape_in_args_and_json_text():
    inv, lab = held_records.records_in({
        "agent_name": "subsurface", "method": "write_diggs",
        # parameters as models send them: JSON text
        "parameters": json.dumps({"investigations": [B1, B2],
                                  "lab_tests": [LAB]})})
    assert [i["investigation_id"] for i in inv] == ["B-1", "B-2"]
    assert lab == [LAB]
    # in prose, fenced, once each
    text = ("Here is what I have.\n```json\n" + json.dumps([B1]) + "\n```\n"
            "and again: " + json.dumps(B1))
    inv, lab = held_records.records_in(text)
    assert inv == [B1] and lab == []


@pytest.mark.parametrize("value", [
    {"investigation_id": "B-1"},                       # no log content
    {"investigation_id": "B-1", "layers": "SP over GP"},   # not a list
    {"id": "B-1", "layers": [{"top": 0}]},             # not the record's key
    {"investigation_id": "B-1", "layers": [{"top": 0}], "a": 1, "b": 2,
     "c": 3},                                          # mostly not a record
    {"kind": "atterberg"},                             # a kind and nothing
    {"kind": "other", "investigation_id": "B-1"},      # 'other', no result
    {"kind": "boring", "result": {"kind": "boring"}},  # not a lab kind
    {"q_ultimate_kPa": 512.0, "Nc": 30.1, "pages": [3]},   # a calc result
    [{"name": "B-1", "value": 3.0}, {"rows": [1, 2]}],
    "plain words about B-1 and its layers",
])
def test_json_that_is_not_records_is_not_taken(value):
    assert held_records.records_in(value) == ([], [])
    assert held_records.records_in(json.dumps(value)) == ([], [])


def _rec(*steps):
    rec = keep_work.HelperRecord("task_x", "general-purpose", "logs")
    for s in steps:
        if s[0] == "model":
            rec.add_model(AIMessage(content=s[1], tool_calls=s[2]))
        else:
            rec.add_result(s[1], s[2], s[3], s[4] if len(s) > 4 else None)
    return rec


def test_records_a_helper_already_saved_are_not_handed_over_twice():
    rec = _rec(
        ("model", "", [{"name": "save_file", "id": "s1",
                        "args": {"path": "b1.json",
                                 "content": json.dumps([B1])}}]),
        ("result", "save_file", "s1", json.dumps({"saved": "/w/b1.json"})))
    # its answer repeats B-1 and adds B-2: only B-2 is held back
    held = held_records.held_records(rec, json.dumps([B1, B2]))
    assert [i["investigation_id"] for i in held["investigations"]] == ["B-2"]
    assert held["source"] == "its answer"
    # everything was saved: nothing is held
    assert held_records.held_records(rec, json.dumps([B1])) is None


def test_the_source_holding_most_records_is_kept():
    rec = _rec(
        ("model", "B-1 first.", [{"name": "check", "id": "c1",
                                  "args": {"investigations": [B1, B2]}}]),
        ("result", "check", "c1", '{"error": "busy"}', "error"))
    held = held_records.held_records(rec, "Done: " + json.dumps([B1]))
    assert len(held["investigations"]) == 2
    assert held["source"].startswith("its check call")
    # a refused save_file is not a delivery
    rec = _rec(
        ("model", "", [{"name": "save_file", "id": "s1",
                        "args": {"path": "b.json",
                                 "content": json.dumps([B1, B2])}}]),
        ("result", "save_file", "s1", json.dumps({"error": "disk full",
                                                  "saved": "b.json"})))
    assert len(held_records.held_records(rec)["investigations"]) == 2


# ---------------------------------------------------------------------------
# Through a real compiled agent
# ---------------------------------------------------------------------------

class Played(BaseChatModel):
    """Answers each call from the conversation it is shown. The primary
    (asked ``PRIMARY <brief>``) delegates ``brief`` to a general-purpose
    helper, then runs ``after(task_result)``'s call or answers. A helper
    plays ``script``: one entry per step, a list of tool calls, a reply
    text, or an exception to raise."""

    brief: str = ""
    script: list = []
    after: object = None
    primary_first: list = []
    seen: list = []

    @property
    def _llm_type(self):
        return "played"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        first = next(m.content for m in messages
                     if isinstance(m, HumanMessage))
        step = sum(1 for m in messages if isinstance(m, AIMessage))
        self.seen.append(messages)
        if first.startswith("PRIMARY"):
            plan = list(self.primary_first) + [[{
                "name": "task", "id": "task_0",
                "args": {"description": self.brief,
                         "subagent_type": "general-purpose"}}]]
            if step < len(plan):
                reply = AIMessage(content="", tool_calls=plan[step])
            elif step == len(plan) and self.after is not None:
                told = next(m for m in messages if isinstance(m, ToolMessage)
                            and m.tool_call_id == "task_0").content
                reply = AIMessage(content="", tool_calls=self.after(told))
            else:
                reply = AIMessage(content="All done.")
            return ChatResult(generations=[ChatGeneration(message=reply)])
        move = self.script[min(step, len(self.script) - 1)]
        if isinstance(move, BaseException):
            raise move
        if isinstance(move, str):
            reply = AIMessage(content=move)
        else:
            reply = AIMessage(content="", tool_calls=[
                {**c, "id": c.get("id") or f"h{step}_{i}"}
                for i, c in enumerate(move)])
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _tools():
    def big_read(page: int) -> str:
        """Read one log page as rows."""
        return f"page {page}\n" + ROW * 400

    def check_logs(investigations: list) -> str:
        """Check boring-log records."""
        return json.dumps({"error": "the checking service is busy"})

    return [StructuredTool.from_function(big_read, name="big_read"),
            StructuredTool.from_function(check_logs, name="check_logs")]


def _write_diggs(name, *, lab=False):
    params = {"investigations_file": name, "output_path": "site.diggs.xml"}
    if lab:
        params["lab_tests_file"] = name
    return [{"name": "call_agent", "id": "w1",
             "args": {"agent_name": "subsurface", "method": "write_diggs",
                      "parameters": params}}]


def _run(model, folder):
    agent = build_deep_agent(model=model, extra_tools=_tools(),
                             reference_mode="off")
    with working_dir_bound(str(folder)):
        state = agent.invoke({"messages": [{"role": "user",
                                            "content": "PRIMARY go"}]},
                             config={"recursion_limit": 60})
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


def _by_id(results, call_id):
    return next(m for m in results if m.tool_call_id == call_id)


def test_a_helper_that_errors_leaves_its_records_as_a_file_the_primary_names(
        tmp_path):
    """The helper's largest call holds two borings (refused), then the model
    connection drops. The task result names a records file; the primary
    passes it to write_diggs, which writes valid DIGGS from it as given."""
    reads = [[{"name": "big_read", "args": {"page": p}}] for p in (30, 31)]
    model = Played(
        brief="HELPER: read logs B-1 and B-2 into records",
        script=reads + [[{"name": "check_logs",
                          "args": {"investigations": [B1, B2]}}],
                        RuntimeError("the connection to the model was "
                                     "reset")],
        after=lambda told: _write_diggs(json.loads(told)["records_file"]))
    results = _run(model, tmp_path)
    task = json.loads(_by_id(results, "task_0").content)
    assert "the connection to the model was reset" in task["error"]
    name = task["records_file"]
    assert re.fullmatch(r"kept/helper_[a-z0-9_-]+_records\.json", name), name
    assert name[:-len("_records.json")] + ".md" == task["kept"]
    assert "2 explorations (B-1, B-2)" in task["records"]
    assert "investigations_file" in task["records"]
    assert "never retype" in task["records"]
    saved = json.loads((tmp_path / name).read_text(encoding="utf-8"))
    assert saved == {"investigations": [B1, B2]}        # as it wrote them
    # the kept .md names it too
    assert name in (tmp_path / task["kept"]).read_text(encoding="utf-8")
    wrote = json.loads(_by_id(results, "w1").content)
    assert "error" not in wrote, wrote
    assert wrote["read_back"]["equal"], wrote["read_back"]
    assert wrote["sources"] == [{"from": name,
                                 "investigations": ["B-1", "B-2"],
                                 "lab_tests": 0}]
    assert (tmp_path / "site.diggs.xml").is_file()


def test_a_helper_that_gives_up_keeps_the_records_in_its_answer(tmp_path):
    """A short answer after a lot of reading, holding a boring and a lab
    test as JSON: the note after the answer names the records file, which
    the primary passes for both groups."""
    reads = [[{"name": "big_read", "args": {"page": p}}] for p in (30, 31)]
    answer = ("I ran out of room. What I have:\n```json\n"
              + json.dumps({"investigations": [B1], "lab_tests": [LAB]})
              + "\n```")
    assert len(answer) < keep_work.SHORT_ANSWER_CHARS
    holder = {}

    def after(told):
        holder["told"] = told
        name = re.search(r"are saved as '([^']+)'", told).group(1)
        return _write_diggs(name, lab=True)

    model = Played(brief="HELPER: logs to records", script=reads + [answer],
                   after=after)
    results = _run(model, tmp_path)
    told = holder["told"]
    assert told.startswith("I ran out of room.")
    assert "[Kept:" in told
    assert "[Records this helper held, from its answer" in told
    assert "1 exploration (B-1) and 1 lab test" in told
    assert "as investigations_file AND as lab_tests_file" in told
    name = re.search(r"are saved as '([^']+)'", told).group(1)
    assert json.loads((tmp_path / name).read_text(encoding="utf-8")) == {
        "investigations": [B1], "lab_tests": [LAB]}
    wrote = json.loads(_by_id(results, "w1").content)
    assert "error" not in wrote, wrote
    assert wrote["written"]["investigations"] == 1
    assert wrote["written"]["lab_tests"] == 1


def test_a_stopped_helper_whose_json_is_not_records_saves_no_records_file(
        tmp_path):
    reads = [[{"name": "big_read", "args": {"page": p}}] for p in (30, 31)]
    answer = ("Sorry, partial: ```json\n" + json.dumps(
        {"pages_read": [30, 31], "status": "partial",
         "rows": [{"id": "B-7", "depth": 3.0, "N": 17}]}) + "\n```")
    model = Played(brief="HELPER: logs", script=reads + [answer])
    results = _run(model, tmp_path)
    told = _by_id(results, "task_0").content
    assert "[Kept:" in told                       # the .md is kept as before
    assert "Records this helper held" not in told
    assert not list((tmp_path / "kept").glob("*records*"))


def test_a_helper_that_answers_in_full_saves_no_records_file(tmp_path):
    """Not stopped: its answer goes to the primary as before."""
    answer = "All borings:\n" + json.dumps([B1, B2]) + "\n" + "x" * 2_000
    model = Played(brief="HELPER: logs", script=[answer])
    results = _run(model, tmp_path)
    assert _by_id(results, "task_0").content.startswith("All borings:")
    assert not (tmp_path / "kept").exists()


# ---------------------------------------------------------------------------
# A helper's .json write_file is saved on the real disk
# ---------------------------------------------------------------------------

def test_a_helpers_json_write_file_lands_on_real_disk_for_write_diggs(
        tmp_path):
    """Before: the helper's write_file made a scratch file write_diggs could
    not read. Now: it is saved in the working folder under the same name,
    the helper is told so, and the primary passes it to write_diggs."""
    model = Played(
        brief="HELPER: B-1 and B-2 as a JSON file",
        script=[[{"name": "write_file",
                  "args": {"file_path": "/records/borings.json",
                           "content": json.dumps([B1, B2])}}],
                "Wrote the records."],
        after=lambda told: _write_diggs("records/borings.json"))
    results = _run(model, tmp_path)
    path = tmp_path / "records" / "borings.json"
    assert json.loads(path.read_text(encoding="utf-8")) == [B1, B2]
    # what the helper was told, as its write_file result
    helper_saw = [m for msgs in model.seen for m in msgs
                  if isinstance(m, ToolMessage) and m.name == "write_file"]
    data = json.loads(helper_saw[-1].content)
    assert data["file_exists"] is True
    assert "saved in the working folder with save_file instead, as " \
           "'records/borings.json'" in data["note"]
    # the primary is told the file's name in code (saved_files)
    told = _by_id(results, "task_0").content
    assert "Files this helper saved" in told and "borings.json" in told
    wrote = json.loads(_by_id(results, "w1").content)
    assert "error" not in wrote, wrote
    assert wrote["sources"] == [{"from": "records/borings.json",
                                 "investigations": ["B-1", "B-2"],
                                 "lab_tests": 0}]


def test_a_helpers_json_never_overwrites_a_file_it_did_not_save(tmp_path):
    """The working folder already holds 'borings.json' (another helper's
    records, or the user's upload): the helper's write_file of that name
    gets a name of its own and the file is left as it was. A file the
    helper saved itself is rewritten under its name."""
    theirs = json.dumps({"investigations": [B2]})
    (tmp_path / "borings.json").write_text(theirs, encoding="utf-8")

    def write(name, recs):
        return [{"name": "write_file",
                 "args": {"file_path": name, "content": json.dumps(recs)}}]

    model = Played(
        brief="HELPER: B-1 as a JSON file",
        script=[write("/borings.json", [B1]), write("/borings.json", [B1]),
                write("/mine.json", [B1]), write("/mine.json", [B1, B2]),
                "Wrote the records."])
    results = _run(model, tmp_path)
    assert (tmp_path / "borings.json").read_text(encoding="utf-8") == theirs
    own = tmp_path / "borings_task0.json"           # task_0's own name
    assert json.loads(own.read_text(encoding="utf-8")) == [B1]
    assert json.loads((tmp_path / "mine.json").read_text(
        encoding="utf-8")) == [B1, B2]
    assert not list(tmp_path.glob("mine_*.json"))
    by_call = {m.tool_call_id: json.loads(m.content)["note"]
               for msgs in model.seen for m in msgs
               if isinstance(m, ToolMessage) and m.name == "write_file"}
    notes = list(by_call.values())
    assert len(notes) == 4
    for note in notes[:2]:
        assert "as 'borings_task0.json'" in note
        assert "'borings.json' was already in the working folder and was " \
               "left as it was" in note
    assert "as 'mine.json'" in notes[3] and "already" not in notes[3]
    told = _by_id(results, "task_0").content
    assert "borings_task0.json" in told and "mine.json" in told


def test_nothing_is_kept_in_the_shared_temp_folder_on_a_shared_host(
        tmp_path, monkeypatch):
    """With no working folder bound on a shared host the kept files are
    not written at all -- the temp folder belongs to every user."""
    from funhouse_agent import _fileio
    monkeypatch.setattr(keep_work.tempfile, "gettempdir",
                        lambda: str(tmp_path))
    rec = _rec(("model", "", [{"name": "check", "id": "c1",
                               "args": {"investigations": [B1]}}]),
               ("result", "check", "c1", '{"error": "busy"}', "error"))
    _fileio.require_turn_binding(True)
    try:
        assert keep_work.save_held_records(rec) is None
        assert keep_work.write_helper_file(rec, "it stopped") is None
    finally:
        _fileio.require_turn_binding(False)
    assert not list(tmp_path.rglob("*"))
    # bound, the same records are kept in the conversation's folder
    with working_dir_bound(str(tmp_path / "conv")):
        kept = keep_work.save_held_records(rec)
    assert kept["name"].startswith("kept/helper_")
    assert (tmp_path / "conv" / kept["name"]).is_file()


def test_other_write_file_calls_stay_in_scratch(tmp_path):
    """The primary's own .json write_file, and a helper's .md, are scratch
    notes as before: nothing is written to the working folder."""
    model = Played(
        brief="HELPER: notes",
        primary_first=[[{"name": "write_file", "id": "p1",
                         "args": {"file_path": "/plan.json",
                                  "content": json.dumps([B1])}}]],
        script=[[{"name": "write_file",
                  "args": {"file_path": "/notes.md", "content": "# notes"}}],
                "Noted."])
    results = _run(model, tmp_path)
    assert "plan.json" in _by_id(results, "p1").content
    assert not (tmp_path / "plan.json").exists()
    assert not (tmp_path / "notes.md").exists()
    assert not list(tmp_path.rglob("*.json"))


def _req(name, **args):
    from langchain.agents.middleware.types import ToolCallRequest
    return ToolCallRequest(tool_call={"name": name, "args": args, "id": "c1"},
                           tool=None, state={"files": {}}, runtime=None)


def test_the_guard_saves_only_a_json_write_file_and_only_with_a_saver():
    seen = []

    def handler(r):
        seen.append(r)
        return ToolMessage(content="scratch", tool_call_id="c1",
                           name=r.tool_call["name"])

    saver = object()
    plain = ScratchFilesystemGuard()
    out = plain.wrap_tool_call(_req("write_file", file_path="/r.json",
                                    content="[]"), handler)
    assert out.content == "scratch" and seen[-1].tool_call["name"] == \
        "write_file"
    helper = ScratchFilesystemGuard(saver)
    for path in ("/notes.md", "/memories/r.json",
                 "/large_tool_results/abc.json"):
        helper.wrap_tool_call(_req("write_file", file_path=path,
                                   content="[]"), handler)
        assert seen[-1].tool_call["name"] == "write_file", path
    # a scratch name and a real path, both saved with the helper's tool
    for path, target in (("/sub/r.json", "sub/r.json"),
                         ("r.JSON", "r.JSON"),
                         ("/tmp/r.json", "/tmp/r.json")):
        helper.wrap_tool_call(_req("write_file", file_path=path,
                                   content="[1]"), handler)
        sent = seen[-1]
        assert sent.tool is saver
        assert sent.tool_call == {"name": "save_file", "id": "c1",
                                  "args": {"path": target, "content": "[1]"}}


def test_the_guard_reports_a_failed_save_as_an_error():
    def handler(r):
        return ToolMessage(content=json.dumps({"error": "disk full"}),
                           tool_call_id="c1", name="save_file",
                           status="error")

    out = ScratchFilesystemGuard(object()).wrap_tool_call(
        _req("write_file", file_path="/r.json", content="[]"), handler)
    assert out.status == "error" and out.name == "write_file"
    assert "disk full" in out.content and "run as save_file" in out.content


def test_helpers_with_save_file_carry_a_saver_and_the_primary_none(
        monkeypatch):
    import funhouse_agent.deep.agent as deep_agent
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel)
    captured = {}
    monkeypatch.setattr(deep_agent, "create_deep_agent",
                        lambda **kw: captured.update(kw) or object())
    deep_agent.build_deep_agent(
        GenericFakeChatModel(messages=iter([AIMessage(content="ok")])),
        enable_calc_subagent=True)

    def saver(middleware):
        (guard,) = [m for m in middleware
                    if isinstance(m, ScratchFilesystemGuard)]
        return guard._json_saver

    assert saver(captured["middleware"]) is None
    specs = {s["name"]: s for s in captured["subagents"]}
    for name in ("general-purpose", "calc"):
        assert getattr(saver(specs[name]["middleware"]), "name", None) == \
            "save_file", name
    for name in ("references", "reviewer"):
        assert saver(specs[name]["middleware"]) is None, name
