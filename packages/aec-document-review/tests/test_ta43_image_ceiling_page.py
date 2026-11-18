"""BACKLOG TA-43 on the page: a turn's image calls, counted and capped.

Through the page's real compiled agent and turn worker, with a scripted
chat model whose tool looks at an image through the core's vision engine
(a fake vision model). Unset, the turn runs as before and its record
carries the count. With ``GEOTECH_TURN_MAX_IMAGE_CALLS`` set, the running
turn's status shows the count against the ceiling, a look past it is
refused with a tool error, and the answer ends with one line that says so.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

import aec_document_review.core as core
import aec_document_review.turn_jobs as tj
from aec_agent_core import activity_log
from aec_agent_core.deep import keep_work as kw
from aec_agent_core.deep.limits import TURN_MAX_IMAGE_CALLS_ENV as ENV
from aec_agent_core.deep.vision_engine import (LangChainVisionEngine,
                                               describe_error)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
LOOKS = 3


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    tj._JOBS.clear()
    yield
    tj._JOBS.clear()


class Looker(BaseChatModel):
    """Calls 1-3 each look at one sheet; call 4 answers."""

    calls: list = []

    @property
    def _llm_type(self):
        return "looker"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(1)
        n = len(self.calls)
        if n <= LOOKS:
            reply = AIMessage(content="", tool_calls=[
                {"name": "look_at_sheet", "id": f"call_{n}",
                 "args": {"sheet": n}}])
        else:
            reply = AIMessage(content="Done.")
        return ChatResult(generations=[ChatGeneration(message=reply)])


class Eyes(BaseChatModel):
    """A fake vision model."""

    state: Any = None

    @property
    def _llm_type(self):
        return "eyes-fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        with self.state["lock"]:
            self.state["calls"] += 1
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="Door 508"))])


def _agent(eyes):
    engine = LangChainVisionEngine(eyes, detail="")

    def look_at_sheet(sheet: int) -> str:
        """Look at one drawing sheet."""
        try:
            return engine.analyze_image(PNG, f"Read sheet {sheet}.")
        except Exception as exc:   # as the core's vision tools report it
            return json.dumps({"error": describe_error(exc)})

    from aec_document_review.review_build import build_review_page_agent
    return build_review_page_agent(
        model=Looker(calls=[]),
        extra_tools=[StructuredTool.from_function(look_at_sheet,
                                                  name="look_at_sheet")],
        reference_mode="off")


def _run(agent, tid, prompt="What do the sheets show?"):
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


def _turn_end(tid):
    return [r for r in activity_log.load(core.conversation_dir(tid))
            if r.get("event") == "turn_end"][-1]


def _image_lines(job):
    return [e.get("text") for e in job.events
            if e.get("kind") == "tool_call"
            and str(e.get("text", "")).startswith("image calls")]


def _eyes():
    return Eyes(state={"calls": 0, "lock": threading.Lock()})


def test_unset_the_turn_runs_as_before_and_its_record_has_the_count():
    eyes = _eyes()
    job = _run(_agent(eyes), "TA43A")
    assert job.result["error"] is None
    assert job.result["final"] == "Done."
    assert _image_lines(job) == []
    assert eyes.state["calls"] == LOOKS
    end = _turn_end("TA43A")
    assert end["image_calls"] == LOOKS
    assert "image_limit" not in end and "image_calls_refused" not in end


def test_with_a_ceiling_the_status_shows_it_and_the_answer_says_so(
        monkeypatch):
    monkeypatch.setenv(ENV, "2")
    eyes = _eyes()
    job = _run(_agent(eyes), "TA43B")
    assert job.result["error"] is None
    assert eyes.state["calls"] == 2                  # the third not sent
    assert _image_lines(job) == ["image calls this turn: 1 of 2",
                                 "image calls this turn: 2 of 2"]
    refused = [e for e in job.events if e.get("kind") == "tool_result"
               and "ceiling of 2 image calls" in str(e.get("text"))]
    assert len(refused) == 1
    final = job.result["final"]
    assert final.startswith("Done.\n\n")
    assert final.endswith(kw.image_ceiling_told(_Refused(2, 1)))
    assert final.count("ceiling") == 1
    end = _turn_end("TA43B")
    assert (end["image_calls"], end["image_limit"],
            end["image_calls_refused"]) == (2, 2, 1)


def test_a_ceiling_the_turn_stays_under_says_nothing_at_the_end(
        monkeypatch):
    monkeypatch.setenv(ENV, "10")
    job = _run(_agent(_eyes()), "TA43C")
    assert job.result["final"] == "Done."
    assert _image_lines(job)[-1] == "image calls this turn: 3 of 10"


class _Refused:
    def __init__(self, limit, refused):
        self.image_limit, self.image_refused = limit, refused
