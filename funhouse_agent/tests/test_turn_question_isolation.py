"""DR-37: two conversations' turns at once each hand their OWN question to
the vision side calls, helpers' included.

On Tiny Apps one process serves many people, and a turn's tools run in
worker threads. The question is a context variable bound by the turn worker
(``webapp.turn_jobs``); this runs two turns at once through that worker,
``core.stream_turn``, the heartbeat pump and a real compiled deep agent,
each turn making two looks at once and a ``task`` helper making its own. A
barrier holds the four main looks until all have started, so the turns
overlap. A process-wide variable in place of the context variable fails
this test (review of DR-36/37, 2026-11-08). All text here is made up.
"""

from __future__ import annotations

import os
import threading
import time

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

import webapp.core as core
import webapp.turn_jobs as tj
from funhouse_agent import turn_question
from funhouse_agent.vision_tools import ask_vision

QUESTIONS = {
    "ann": "Made-up question A: which sketch curve is tallest, ann?",
    "bo": "Made-up question B: how many boxes sit on sheet two, bo?",
}
HELPER_BRIEF = "helper look for "


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    # The conversations go under tmp_path, never the real store.
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    monkeypatch.setenv("GEOTECH_HEARTBEAT_S", "0.05")
    tj._JOBS.clear()
    yield
    tj._JOBS.clear()


class Engine:
    """A fake vision engine: hands back the prompt it was sent."""

    accepts_jpeg = False

    def analyze_image(self, image_bytes, prompt=""):
        return prompt


class Scripted(BaseChatModel):
    """The primary asks for two looks and one helper at once, then answers;
    the helper asks for one look, then answers. Which conversation it is in
    is read off the first human message."""

    @property
    def _llm_type(self):
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        first = next(str(m.content) for m in messages
                     if isinstance(m, HumanMessage))
        answered = any(isinstance(m, ToolMessage) for m in messages)
        if first.startswith(HELPER_BRIEF):
            who = first[len(HELPER_BRIEF):]
            msg = (AIMessage(content="helper done") if answered else
                   AIMessage(content="", tool_calls=[{
                       "name": "peek", "id": f"h-{who}",
                       "args": {"who": who, "where": "helper"}}]))
        else:
            who = "ann" if first == QUESTIONS["ann"] else "bo"
            msg = (AIMessage(content=f"done {who}") if answered else
                   AIMessage(content="", tool_calls=[
                       {"name": "peek", "id": f"p1-{who}",
                        "args": {"who": who, "where": "main1"}},
                       {"name": "peek", "id": f"p2-{who}",
                        "args": {"who": who, "where": "main2"}},
                       {"name": "task", "id": f"t-{who}",
                        "args": {"description": HELPER_BRIEF + who,
                                 "subagent_type": "general-purpose"}}]))
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _peek_tool(seen, lock, barrier, overlapped):
    def peek(who: str, where: str) -> str:
        """Take one made-up vision look."""
        if where.startswith("main"):
            try:
                barrier.wait()
                overlapped.append(where)
            except threading.BrokenBarrierError:
                pass
        sent = ask_vision(Engine(), b"\x89PNG made-up",
                          f"Read the sketch ({where}).", "image")
        with lock:
            seen.append((who, where, sent, turn_question.current()))
        return "looked"
    return StructuredTool.from_function(peek, name="peek")


def _ctx(files, prompt):
    return {"prompt": prompt, "temp_dir": files,
            "before": core.snapshot_dir(files), "staged_inputs": set(),
            "working_dir": files, "before_wd": None, "artifacts": [],
            "artifacts_before_len": 0,
            "transcript": [{"role": "user", "text": prompt}],
            "trace_on": False, "model": "test-model",
            "behavior": core.default_behavior()}


def test_two_turns_at_once_each_side_call_gets_its_own_question(tmp_path):
    from funhouse_agent.deep.agent import build_deep_agent
    seen, lock, overlapped = [], threading.Lock(), []
    barrier = threading.Barrier(4, timeout=15)    # 2 looks x 2 turns
    tool = _peek_tool(seen, lock, barrier, overlapped)
    agent = build_deep_agent(model=Scripted(), extra_tools=[tool],
                             subagent_extra_tools=[tool],
                             reference_mode="off")
    jobs = []
    for who, question in QUESTIONS.items():
        tid = f"TQ-ISO-{who}"
        assert core.conversation_dir(tid).startswith(str(tmp_path))
        core.ensure_conversation(tid)
        files = core.conversation_files_dir(tid)
        os.makedirs(files, exist_ok=True)
        core.begin_partial(tid, question)
        jobs.append(tj.start_turn_job(
            agent, [{"role": "user", "content": question}], tid, 60,
            _ctx(files, question)))
    t0 = time.time()
    while not all(j.done for j in jobs) and time.time() - t0 < 60:
        time.sleep(0.05)
    assert all(j.done for j in jobs)
    for job in jobs:
        assert not (job.result or {}).get("error"), job.result
    assert len(overlapped) == 4, "the four main looks did not run at once"
    assert {(w, where) for w, where, _, _ in seen} == {
        (w, where) for w in QUESTIONS
        for where in ("main1", "main2", "helper")}
    for who, where, sent, current in seen:
        other = "bo" if who == "ann" else "ann"
        assert current == QUESTIONS[who], (who, where, current)
        assert QUESTIONS[who] in sent, (who, where)
        assert QUESTIONS[other] not in sent, (who, where)
    assert turn_question.current() is None     # the caller's context untouched
