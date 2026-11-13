"""A turn the tester can stop (BACKLOG TA-37), behind GEOTECH_STOP_BUTTON.

Deployment review 2026-11-10, section 2.1: nothing stopped a turn once it
was sent; a tester who asked the wrong thing waited it out while their
monthly budget paid. With the setting on, the page's Stop button sets a
flag on the turn's job; the turn checks it before each model call of every
agent, before each wait and before each pass, keeps its work the way an
interrupted turn's is kept, and its answer says it was stopped and what was
done. Unset, nothing changes.

Offline: scripted chat models through the page's real compiled agent, run
by the app's own turn worker (``turn_jobs``) and ``core.stream_turn``.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

import aec_document_review.core as core
import aec_document_review.turn_jobs as tj
from aec_agent_core.deep import keep_work as kw

ROW = "S-101 | note 4 | 4000 psi concrete\n"


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    monkeypatch.delenv(tj.STOP_BUTTON_ENV, raising=False)
    tj._JOBS.clear()
    yield
    tj._JOBS.clear()


class Reader(BaseChatModel):
    """Reads a sheet on every call until it has read ``reads`` sheets, then
    answers. Counts its calls."""

    reads: int = 3
    calls: list = []

    @property
    def _llm_type(self):
        return "reader"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.calls.append(len(messages))
        n = sum(1 for m in messages if isinstance(m, AIMessage)
                and m.tool_calls)
        if n < self.reads:
            reply = AIMessage(content="", tool_calls=[
                {"name": "read_sheet", "id": f"call_{n}",
                 "args": {"sheet": n + 1}}])
        else:
            reply = AIMessage(content="All sheets read: 4000 psi throughout.")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _tool(on_read=None):
    def read_sheet(sheet: int) -> str:
        """Read one drawing sheet."""
        if on_read is not None:
            on_read(sheet)
        return f"sheet {sheet}\n" + ROW * 5
    return StructuredTool.from_function(read_sheet, name="read_sheet")


def _agent(model, on_read=None):
    from aec_document_review.review_build import build_review_page_agent
    return build_review_page_agent(model=model, extra_tools=[_tool(on_read)],
                                   reference_mode="off")


def _ctx(files, transcript, prompt):
    return {"prompt": prompt, "temp_dir": files,
            "before": core.snapshot_dir(files), "staged_inputs": set(),
            "working_dir": files, "before_wd": None, "artifacts": [],
            "artifacts_before_len": 0, "transcript": transcript,
            "trace_on": False, "model": "test-model",
            "behavior": core.default_behavior()}


def _start(agent, tid, prompt="Check every sheet's concrete strength."):
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    transcript = [{"role": "user", "text": prompt}]
    core.append_transcript(tid, transcript[-1])
    messages = [{"role": "user", "content": prompt}]
    core.begin_partial(tid, prompt)
    return tj.start_turn_job(agent, messages, tid, 60,
                             _ctx(files, transcript, prompt))


def _wait(job, timeout=30.0):
    t0 = time.time()
    while not job.done and time.time() - t0 < timeout:
        time.sleep(0.02)
    assert job.done
    return job


def _events(tid):
    path = os.path.join(core.conversation_dir(tid), "activity.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# End to end, through the page's agent and the turn worker
# ---------------------------------------------------------------------------

def test_a_stopped_turn_ends_at_its_next_model_call_and_keeps_its_work(
        monkeypatch):
    """The tester presses Stop while the first sheet is being read: the
    model is not called again, the answer says the turn was stopped and
    what it did, the read result is kept in the history, and both the
    request and the stop are on the activity log."""
    monkeypatch.setenv(tj.STOP_BUTTON_ENV, "1")
    holder = {}

    def press_stop(sheet):
        assert holder["job"].request_stop() is True

    model = Reader(calls=[], reads=3)
    job = _start(_agent(model, on_read=press_stop), "STOP1")
    holder["job"] = job
    _wait(job)

    assert len(model.calls) == 1            # never called after the stop
    res = job.result
    assert res.get("stopped") is True
    assert res["error"] is None
    assert res["final"].startswith(core.STOPPED_LABEL)
    assert "Done before the stop: 1 call to read_sheet." in res["final"]
    assert "What this turn had done is kept" in res["final"]

    saved = core.load_messages("STOP1")
    assert [m["role"] for m in saved] == ["user", "assistant", "tool",
                                          "assistant"]
    assert ROW in saved[2]["content"]
    assert "The user stopped this turn" in saved[-1]["content"]
    entry = core.load_transcript("STOP1")[-1]
    assert entry["stopped"] is True and "error" not in entry
    assert entry["kept"]["tool_results"] == 1

    names = [e.get("event") for e in _events("STOP1")]
    assert "turn_stop_requested" in names
    assert "turn_stopped" in names
    assert names.index("turn_stop_requested") < names.index("turn_stopped")


def test_unset_the_turn_cannot_be_stopped_and_runs_as_before():
    """GEOTECH_STOP_BUTTON unset: the job takes no stop request and the turn
    runs to its answer."""
    holder = {}
    taken = []

    def press_stop(sheet):
        taken.append(holder["job"].request_stop())

    model = Reader(calls=[], reads=2)
    job = _start(_agent(model, on_read=press_stop), "STOP2")
    holder["job"] = job
    _wait(job)
    assert job.stoppable is False
    assert taken == [False, False]
    assert len(model.calls) == 3
    assert job.result["final"] == "All sheets read: 4000 psi throughout."
    assert "stopped" not in job.result
    assert "turn_stop_requested" not in [e.get("event")
                                         for e in _events("STOP2")]


def test_a_finished_job_takes_no_stop_request(monkeypatch):
    monkeypatch.setenv(tj.STOP_BUTTON_ENV, "on")
    job = _wait(_start(_agent(Reader(calls=[], reads=0)), "STOP3"))
    assert job.request_stop() is False
    assert "stopped" not in job.result


# ---------------------------------------------------------------------------
# The checks themselves
# ---------------------------------------------------------------------------

def test_check_stop_raises_only_in_a_stopped_turn():
    kw.check_stop()                                   # outside a turn
    ev = threading.Event()
    with kw.turn_scope(kw.TurnScope(stop_event=ev)) as scope:
        kw.check_stop()
        assert scope.spend_busy(5.0) is True
        ev.set()
        with pytest.raises(kw.TurnStopped):
            kw.check_stop()
        # the busy ladder may not wait in a stopped turn
        assert scope.spend_busy(5.0) is False
    with kw.turn_scope(kw.TurnScope()):               # no stop at all
        kw.check_stop()


def test_a_stopped_turn_raises_before_its_next_pass():
    """``stream_turn`` checks before it starts a pass: a turn stopped
    before it began never calls the agent."""
    class Never:
        def stream(self, *a, **k):
            raise AssertionError("the agent was called")
    ev = threading.Event()
    ev.set()
    with pytest.raises(kw.TurnStopped):
        list(core.stream_turn(Never(), [{"role": "user", "content": "q"}],
                              "STOP4", stop=ev))


def test_a_helpers_next_model_call_raises_in_a_stopped_turn():
    ev = threading.Event()
    called = []
    with kw.turn_scope(kw.TurnScope(stop_event=ev)):
        rec_mw = kw.HelperRecorder()
        rec_mw.wrap_model_call(object(), lambda r: called.append(1))
        ev.set()
        with pytest.raises(kw.TurnStopped):
            rec_mw.wrap_model_call(object(), lambda r: called.append(1))
    assert called == [1]


def test_the_wait_for_a_helper_place_ends_on_stop():
    """A helper waiting for a place (two run at once) stops waiting within
    a second of the stop, instead of waiting for the others to finish."""
    import contextvars
    places = kw._HelperPlaces(1)
    places.acquire()                                  # the one running
    ev = threading.Event()
    out = {}

    def waiter():
        try:
            places.acquire(stoppable=True)
            out["got"] = True
        except kw.TurnStopped:
            out["stopped"] = time.monotonic()

    with kw.turn_scope(kw.TurnScope(stop_event=ev)):
        ctx = contextvars.copy_context()
    t = threading.Thread(target=ctx.run, args=(waiter,), daemon=True)
    t.start()
    time.sleep(0.3)
    asked = time.monotonic()
    ev.set()
    t.join(3.0)
    assert "got" not in out
    assert out["stopped"] - asked < 1.5
    assert places._queue == []                        # left the queue
