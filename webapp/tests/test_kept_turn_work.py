"""A turn that stops on an error keeps its work (BACKLOG TA-19).

Field session 2026-10-14: 51 minutes of helpers' extraction ended on a
rate-limit refusal; ``messages.json`` got the question and an error stub,
the error said "Nothing is lost ... ask the agent to continue", and the
continue turn started from nothing.

Offline: a scripted chat model through a real compiled deep agent, run by
the app's own turn worker (``turn_jobs``) and ``core.stream_turn``.
"""

from __future__ import annotations

import json
import os
import time

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

import webapp.core as core
import webapp.turn_jobs as tj
from webapp import turn_record as tr

ROW = "LB-1 | 1.5 m | N=22 | SP-SM | w=9% \n"


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    tj._JOBS.clear()
    yield


class Scripted(BaseChatModel):
    """Calls ``read_log`` on pages 98 and 99 at once, then -- on its next
    call -- raises ``fail`` (the turn's error); records what it is sent."""

    fail: str = "the model service dropped the connection"
    rows: int = 50
    seen: list = []

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        if not any(isinstance(m, AIMessage) for m in messages):
            reply = AIMessage(content="Reading both pages.", tool_calls=[
                {"name": "read_log", "id": "call_p98", "args": {"page": 98}},
                {"name": "read_log", "id": "call_p99", "args": {"page": 99}}])
            return ChatResult(generations=[ChatGeneration(message=reply)])
        raise RuntimeError(self.fail)


class Recorder(BaseChatModel):
    """Answers at once; records what it was sent (the continue turn)."""

    seen: list = []

    @property
    def _llm_type(self):
        return "recorder"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="Continuing from the kept logs."))])


def _read_log_tool(rows):
    def read_log(page: int) -> str:
        """Read one boring log page."""
        return f"page {page}\n" + ROW * rows
    return StructuredTool.from_function(read_log, name="read_log")


def _agent(model, rows=50):
    from funhouse_agent.deep.agent import build_deep_agent
    return build_deep_agent(model=model, extra_tools=[_read_log_tool(rows)],
                            reference_mode="off")


def _ctx(tid, files, transcript, prompt):
    return {"prompt": prompt, "temp_dir": files,
            "before": core.snapshot_dir(files), "staged_inputs": set(),
            "working_dir": files, "before_wd": None, "artifacts": [],
            "artifacts_before_len": 0, "transcript": transcript,
            "trace_on": False, "model": "test-model",
            "behavior": core.default_behavior()}


def _turn(agent, tid, prompt, messages, transcript):
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    transcript.append({"role": "user", "text": prompt})
    core.append_transcript(tid, transcript[-1])
    messages.append({"role": "user", "content": prompt})
    core.begin_partial(tid, prompt)
    job = tj.start_turn_job(agent, messages, tid, 60,
                            _ctx(tid, files, transcript, prompt))
    t0 = time.time()
    while not job.done and time.time() - t0 < 30:
        time.sleep(0.05)
    assert job.done
    return job


# ---------------------------------------------------------------------------
# End to end: the errored turn, then "continue"
# ---------------------------------------------------------------------------

def test_an_errored_turn_keeps_its_tool_results_and_continue_sees_them():
    """Before: messages.json = [question, error stub]; the continue turn's
    model saw no tool result. Now: the turn's calls and results are in the
    history, in valid order, and the continue turn's model is shown them."""
    tid = "KEEP1"
    core.ensure_conversation(tid)
    messages, transcript = [], []
    first = Scripted(seen=[])
    job = _turn(_agent(first), tid, "Extract the 2019 logs.", messages,
                transcript)
    err = job.result["error"] or ""
    assert "the model service dropped the connection" in err
    assert ("What this turn had done is kept: its 2 finished tool results "
            "are now in the conversation's history") in err

    saved = core.load_messages(tid)
    roles = [m["role"] for m in saved]
    assert roles == ["user", "assistant", "tool", "tool", "assistant"]
    calls = saved[1]["tool_calls"]
    assert {c["id"] for c in calls} == {"call_p98", "call_p99"}
    assert {m["tool_call_id"] for m in saved[2:4]} == {"call_p98", "call_p99"}
    assert all(ROW in m["content"] for m in saved[2:4])
    assert "The tool calls and results above are kept" in saved[-1]["content"]
    assert core.load_transcript(tid)[-1]["kept"]["tool_results"] == 2
    assert not os.path.exists(core.partial_messages_path(tid))   # cleared

    # "continue": the next turn is replayed on the kept history
    second = Recorder(seen=[])
    job2 = _turn(_agent(second), tid, "Are you able to continue?",
                 core.load_messages(tid), transcript)
    assert job2.result["error"] is None
    shown = second.seen[0]
    results = [m for m in shown if isinstance(m, ToolMessage)]
    assert len(results) == 2 and all(ROW in m.content for m in results)
    ai = next(m for m in shown if isinstance(m, AIMessage) and m.tool_calls)
    assert {c["id"] for c in ai.tool_calls} == \
        {m.tool_call_id for m in results}


def test_a_large_result_is_kept_whole_in_a_file_and_cut_in_the_history():
    tid = "KEEP2"
    core.ensure_conversation(tid)
    messages, transcript = [], []
    rows = 2_000                                   # ~74,000 characters
    job = _turn(_agent(Scripted(seen=[]), rows=rows), tid, "Read both.",
                messages, transcript)
    saved = core.load_messages(tid)
    tools = [m for m in saved if m["role"] == "tool"]
    assert all(len(m["content"]) < 6_000 for m in tools)
    assert all("the whole of it is in the file 'kept/turn1_read-log_"
               in m["content"] for m in tools)
    files = core.conversation_files_dir(tid)
    kept = sorted(os.listdir(os.path.join(files, "kept")))
    assert len(kept) == 2
    whole = open(os.path.join(files, "kept", kept[0]), encoding="utf-8").read()
    assert whole.count(ROW) == rows
    # the files are cards of the stopped turn, and named in the error
    entry = core.load_transcript(tid)[-1]
    assert sorted(os.path.basename(a) for a in entry["artifacts"]) == kept
    assert "2 of the largest also as files in the 'kept' folder" in \
        job.result["error"]


def test_a_turn_with_no_finished_result_says_so():
    """No tool returned: the error says there is nothing to pick up from,
    and the history is the old stub."""
    class FailsAtOnce(Scripted):
        def _generate(self, messages, stop=None, run_manager=None, **kw):
            raise RuntimeError("refused at once")
    tid = "KEEP3"
    core.ensure_conversation(tid)
    job = _turn(_agent(FailsAtOnce(seen=[])), tid, "q", [], [])
    assert "No tool had returned a result this turn" in job.result["error"]
    saved = core.load_messages(tid)
    assert [m["role"] for m in saved] == ["user", "assistant"]
    assert saved[-1]["content"] == core.NOTHING_KEPT


# ---------------------------------------------------------------------------
# A turn the process never finished
# ---------------------------------------------------------------------------

def test_an_interrupted_turn_is_folded_into_the_history_on_open():
    tid = "KEEP4"
    core.ensure_conversation(tid)
    core.append_transcript(tid, {"role": "user", "text": "Extract."})
    core.begin_partial(tid, "Extract.")
    rec = tr.TurnRecord(question={"role": "user", "content": "Extract."},
                        checkpoint_path=core.partial_messages_path(tid))
    rec.add([AIMessage(content="", id="a1", tool_calls=[
        {"name": "read_log", "id": "c1", "args": {"page": 98}},
        {"name": "read_log", "id": "c2", "args": {"page": 99}}]),
        ToolMessage(content="page 98\n" + ROW, tool_call_id="c1",
                    name="read_log")])
    # ... and then the process died: no turn end, no save
    got = core.recover_partial(tid)
    assert got["recovered"] and "its 1 finished tool result is kept" in \
        got["text"]
    saved = core.load_messages(tid)
    assert [m["role"] for m in saved] == ["user", "assistant", "tool", "tool",
                                          "assistant"]
    assert saved[0]["content"] == "Extract."
    assert saved[3]["content"] == tr.NOT_FINISHED   # c2 never returned
    assert not os.path.exists(core.partial_messages_path(tid))
    assert core.recover_partial(tid) is None


def test_the_activity_log_records_kept_helpers_and_their_queue(tmp_path):
    """The record says which helper's work went to which file, and how long
    a helper waited for a place under the cap."""
    from funhouse_agent._fileio import working_dir_bound
    from funhouse_agent.deep.tests.test_keep_work_offline import (
        Played, _big_read_tool)
    from funhouse_agent.deep.agent import build_deep_agent
    from webapp import activity_log as al
    briefs = ["HELPER error: logs A", "HELPER answer: logs B",
              "HELPER answer: logs C"]
    agent = build_deep_agent(
        model=Played(briefs=briefs, reads=2),
        extra_tools=[_big_read_tool(delay=0.3)], reference_mode="off")
    log = al.ActivityLogger(str(tmp_path / "conv"), turn=1)
    with working_dir_bound(str(tmp_path)):
        agent.invoke({"messages": [{"role": "user", "content": "PRIMARY"}]},
                     config={"recursion_limit": 80, "callbacks": [log]})
    recs = al.load(str(tmp_path / "conv"))
    kept = [r for r in recs if r["event"] == "helper_kept"]
    assert len(kept) == 1 and \
        kept[0]["path"].startswith("kept/helper_helper-error-logs-a")
    assert "stopped on an error" in kept[0]["why"]
    queued = [r for r in recs if r["event"] == "helper_queued"]
    assert queued and all(r["queued_s"] >= 0.5 for r in queued)


def test_the_messages_checkpoint_is_never_mirrored():
    assert core.is_in_flight_file("partial_messages.json")
    assert core.is_in_flight_file("partial_messages.json.part")


# ---------------------------------------------------------------------------
# The pieces
# ---------------------------------------------------------------------------

def test_complete_pairs_orders_and_closes_calls():
    msgs = [
        {"role": "tool", "content": "orphan", "tool_call_id": "zz"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"name": "a", "args": {}, "id": "1",
                         "type": "tool_call"},
                        {"name": "b", "args": {}, "id": "2",
                         "type": "tool_call"}]},
        {"role": "tool", "content": "r1", "tool_call_id": "1"},
        {"role": "assistant", "content": ""},              # empty: dropped
        {"role": "assistant", "content": "next",
         "tool_calls": [{"name": "c", "args": {}, "id": "3",
                         "type": "tool_call"}]},
        {"role": "tool", "content": "r1 again", "tool_call_id": "1"},
    ]
    out = tr.complete_pairs(msgs)
    assert [(m["role"], m.get("tool_call_id")) for m in out] == [
        ("assistant", None), ("tool", "1"), ("tool", "2"),
        ("assistant", None), ("tool", "3")]
    assert out[2]["content"] == tr.NOT_FINISHED
    assert out[4]["content"] == tr.NOT_FINISHED


def test_an_evicted_result_is_put_back_whole(tmp_path):
    from funhouse_agent.deep.keep_work import TurnScope
    rec = tr.TurnRecord()
    rec.scope = TurnScope()
    whole = "x" * 90_000
    rec.scope.note_full_result("c1", whole)
    rec.add([AIMessage(content="", id="a", tool_calls=[
        {"name": "task", "id": "c1", "args": {}}]),
        ToolMessage(content="Tool result too large, saved to "
                            "/large_tool_results/c1", tool_call_id="c1")])
    kept = rec.kept(str(tmp_path), turn=2)
    (path,) = kept.files
    assert open(path, encoding="utf-8").read() == whole
    assert "90,000 characters" in kept.messages[1]["content"]


def test_the_total_is_held_under_the_cap(tmp_path):
    rec = tr.TurnRecord()
    calls = [{"name": "r", "id": f"c{i}", "args": {}} for i in range(7)]
    rec.add([AIMessage(content="", id="a", tool_calls=calls)]
            + [ToolMessage(content=str(i) * 50_000, tool_call_id=f"c{i}")
               for i in range(7)])                 # 350,000 in all
    kept = rec.kept(str(tmp_path), turn=1)
    total = sum(len(m["content"]) for m in kept.messages
                if m["role"] == "tool")
    assert total <= tr.TOTAL_MAX_CHARS and len(kept.files) == 2


def test_saved_tool_calls_survive_the_round_trip(tmp_path):
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "read_log", "args": {"page": 3},
                             "id": "c1", "type": "tool_call"}]},
            {"role": "tool", "content": "rows", "tool_call_id": "c1",
             "name": "read_log"},
            {"role": "assistant", "content": "[stopped]"}]
    core.save_messages("RT", msgs, root=str(tmp_path))
    back = core.load_messages("RT", root=str(tmp_path))
    assert back[1]["tool_calls"][0]["id"] == "c1"
    assert back[1]["tool_calls"][0]["args"] == {"page": 3}
    assert back[2]["tool_call_id"] == "c1" and back[2]["name"] == "read_log"


def test_the_error_text_says_what_is_kept():
    class RateLimitError(Exception):
        pass
    line = tr.KeptWork(n_results=3, n_calls=3).line()
    busy = core.friendly_turn_error(
        RateLimitError("Too Many Requests"), kept=line)
    assert line in busy and "Wait a minute" in busy
    assert "Nothing is lost" not in busy
    plain = core.friendly_turn_error(ValueError("boom"), kept=line)
    assert plain.startswith("ValueError: boom") and line in plain
    # without a record: only what is always true
    bare = core.friendly_turn_error(RateLimitError("Too Many Requests"))
    assert "Nothing is lost" not in bare and "Files downloaded or saved" in bare
