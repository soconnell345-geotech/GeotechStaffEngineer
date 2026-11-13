"""Ceilings counted over the whole turn (BACKLOG TA-38). Repairs.

Deployment review 2026-11-10, section 2.1:

* a pass that ended without reaching the page's model-call budget could
  auto-continue with a fresh budget (TA-36 fixed only the pass the budget
  itself ended);
* the continuation calls after a reply cut off at the output limit were
  outside the budget, and an error in one sent the whole long call round
  again and dropped the text already paid for;
* the forced last reply could come back empty, and the answer then read
  "(no answer text)".

Offline: scripted chat models through the page's real compiled agent and
the turn worker, and the cut-off guard on its own.
"""

from __future__ import annotations

import os
import time

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

import aec_document_review.core as core
import aec_document_review.turn_jobs as tj
from aec_agent_core.deep import keep_work as kw
from aec_agent_core.deep.tool_guards import OutputLimitGuard

BUDGET_ENV = "GEOTECH_REVIEW_MAX_MODEL_CALLS"


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    tj._JOBS.clear()
    yield
    tj._JOBS.clear()


def _forced(messages) -> bool:
    """The call is the budget's forced last one (its nudge is the last
    message)."""
    last = messages[-1] if messages else None
    return (isinstance(last, HumanMessage)
            and "budget reached" in str(last.content).lower())


class Announcer(BaseChatModel):
    """Calls 1-2 read a sheet; call 3 announces a next step and stops (an
    auto-continue); every later call reads another sheet, unless it is the
    forced last call, which answers (or answers nothing, ``empty``)."""

    calls: list = []
    empty: bool = False
    announce_at: int = 3

    @property
    def _llm_type(self):
        return "announcer"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(_forced(messages))
        n = len(self.calls)
        if _forced(messages):
            reply = AIMessage(content="" if self.empty else
                              "Sheets 1-4 call for 4000 psi.")
        elif n == self.announce_at:
            reply = AIMessage(content="I'll open sheet 3 next.")
        else:
            reply = AIMessage(content="", tool_calls=[
                {"name": "read_sheet", "id": f"call_{n}",
                 "args": {"sheet": n}}])
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _tool():
    def read_sheet(sheet: int) -> str:
        """Read one drawing sheet."""
        return f"sheet {sheet}: 4000 psi"
    return StructuredTool.from_function(read_sheet, name="read_sheet")


def _agent(model):
    from aec_document_review.review_build import build_review_page_agent
    return build_review_page_agent(model=model, extra_tools=[_tool()],
                                   reference_mode="off")


def _run(agent, tid, prompt="What concrete strength do the sheets give?"):
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    transcript = [{"role": "user", "text": prompt}]
    core.append_transcript(tid, transcript[-1])
    messages = [{"role": "user", "content": prompt}]
    ctx = {"prompt": prompt, "temp_dir": files,
           "before": core.snapshot_dir(files), "staged_inputs": set(),
           "working_dir": files, "before_wd": None, "artifacts": [],
           "artifacts_before_len": 0, "transcript": transcript,
           "trace_on": False, "model": "test-model",
           "behavior": core.default_behavior()}
    job = tj.start_turn_job(agent, messages, tid, 60, ctx)
    t0 = time.time()
    while not job.done and time.time() - t0 < 30:
        time.sleep(0.02)
    assert job.done
    return job


# ---------------------------------------------------------------------------
# 1. One budget for every pass
# ---------------------------------------------------------------------------

def test_an_auto_continue_goes_on_with_what_is_left_of_the_budget(
        monkeypatch):
    """Budget 5: pass 1 makes three calls and ends on an announced step;
    the auto-continue pass has two left (one read, then the forced
    answer), not a fresh five. Before: eight calls in the turn."""
    monkeypatch.setenv(BUDGET_ENV, "5")
    model = Announcer(calls=[])
    job = _run(_agent(model), "CEIL1")
    assert len(model.calls) == 5
    assert model.calls[-1] is True                    # the forced answer
    assert model.calls.count(True) == 1
    assert job.result["final"].endswith("Sheets 1-4 call for 4000 psi.")
    assert job.result["error"] is None


# ---------------------------------------------------------------------------
# 2. The continuations of a cut-off reply
# ---------------------------------------------------------------------------

class _Req:
    """The little of a model request the guard uses."""

    def __init__(self, messages=()):
        self.messages = list(messages)

    def override(self, messages=None, **kw):
        return _Req(messages if messages is not None else self.messages)


def _cut(text):
    return AIMessage(content=text,
                     response_metadata={"finish_reason": "length"})


def test_continuations_are_counted_and_leave_the_forced_answer_its_call():
    """Primary budget 3, nothing used: the cut reply is continued once (the
    second continuation would leave no call for the forced answer), and the
    one continuation is counted against the budget. Before: two
    continuations, neither counted."""
    guard = OutputLimitGuard()
    sent = []

    def handler(req):
        sent.append(req)
        return _cut(f"part {len(sent)} ")

    scope = kw.TurnScope(whole_turn_budget=True)
    with kw.turn_scope(scope):
        kw.note_budget(3)
        out = guard.wrap_model_call(_Req(), handler)
        assert kw.extra_model_calls() == 1
    assert len(sent) == 2
    assert out.content == "part 1 part 2 "


def test_a_helpers_continuations_count_against_its_own_budget():
    guard = OutputLimitGuard()
    sent = []

    def handler(req):
        sent.append(req)
        return _cut("x")

    scope = kw.TurnScope(whole_turn_budget=True)
    rec = kw.HelperRecord("call_h", "reader", "read sheets")
    with kw.turn_scope(scope):
        token = kw._HELPER.set(rec)
        try:
            kw.note_budget(10)
            guard.wrap_model_call(_Req(), handler)
            assert kw.extra_model_calls() == 2
        finally:
            kw._HELPER.reset(token)
    assert rec.extra_model_calls == 2 and scope.primary_extra_calls == 0
    assert len(sent) == 3


def test_a_failed_continuation_keeps_the_text_paid_for():
    """The continuation raises: the cut reply is delivered as it is, and
    the error does not leave the guard (which sent the whole long call
    round again through the retries). Before: the error was raised."""
    guard = OutputLimitGuard()
    sent = []

    def handler(req):
        sent.append(req)
        if len(sent) == 1:
            return _cut("A long schedule, first half")
        raise RuntimeError("the model service dropped the connection")

    out = guard.wrap_model_call(_Req(), handler)
    assert out.content == "A long schedule, first half"
    assert len(sent) == 2


def test_outside_a_whole_turn_budget_nothing_is_counted():
    """The geotech page's turns (no whole-turn budget): as before."""
    guard = OutputLimitGuard()
    sent = []

    def handler(req):
        sent.append(req)
        return _cut("y")

    with kw.turn_scope(kw.TurnScope()):
        kw.note_budget(2)
        guard.wrap_model_call(_Req(), handler)
        assert kw.extra_model_calls() == 0
    assert len(sent) == 3


# ---------------------------------------------------------------------------
# 3. An empty forced reply
# ---------------------------------------------------------------------------

def test_an_empty_forced_reply_says_the_budget_ended_and_what_was_done(
        monkeypatch):
    """Budget 4: three reads, then the forced answer comes back empty. The
    answer says the budget ended and what the turn did; the read is kept in
    the history. Before: "(no answer text)" with an error to ask again."""
    monkeypatch.setenv(BUDGET_ENV, "4")
    model = Announcer(calls=[], empty=True, announce_at=0)
    job = _run(_agent(model), "CEIL2")
    assert len(model.calls) == 4 and model.calls[-1] is True
    final = job.result["final"]
    assert final.startswith(core.BUDGET_ENDED_LEAD)
    assert "What it did: 3 calls to read_sheet." in final
    assert "(no answer text)" not in final
    assert job.result["error"] is None
    entry = core.load_transcript("CEIL2")[-1]
    assert entry["budget_ended"] is True and "error" not in entry
    saved = core.load_messages("CEIL2")
    assert [m["role"] for m in saved] == (["user"] + ["assistant", "tool"] * 3
                                          + ["assistant"])
    assert "used its whole model-call budget" in saved[-1]["content"]
