"""DR-37: the vision side call reads the turn's own user question as context.

Chart suite EV-35 (2026-11-08, TRACE_REVIEW section 4.2): ``analyze_image``
and ``render_region`` handed the vision model only the main model's
rewording of the question; on one task the rewording folded the answer
format into the question and the side call answered a narrower one. The
host now binds the user's latest message to the turn (as it binds the
working folder) and ``ask_vision`` adds it, capped, under the main model's
prompt, saying the prompt is what to answer. All text here is made up.
"""

from __future__ import annotations

import contextvars
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

pymupdf = pytest.importorskip("pymupdf")

from funhouse_agent import turn_question  # noqa: E402
from funhouse_agent import vision_tools  # noqa: E402
from funhouse_agent.vision_tools import (  # noqa: E402
    ask_vision, dispatch_extended_tool)

QUESTION = ("Which of the three made-up curves on the sketch peaks highest? "
            "Answer with the curve's colour only.")
PROMPT = "Name the curve that peaks highest and give its colour."


class Engine:
    """A fake vision engine that keeps every prompt it is sent."""

    accepts_jpeg = False

    def __init__(self):
        self.prompts = []

    def analyze_image(self, image_bytes, prompt=""):
        self.prompts.append(prompt)
        return "the blue curve"


def _png(w=320, h=200):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h), 0)
    pix.clear_with(255)
    return pix.tobytes("png")


def _pdf():
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "MADE-UP NOTE 12 A", fontsize=24)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.delenv("GEOTECH_VISION_PROBE", raising=False)
    vision_tools.clear_repeat_reads()
    yield
    vision_tools.clear_repeat_reads()


# -- the context block --------------------------------------------------------

def test_nothing_bound_leaves_the_prompt_as_it_was():
    assert turn_question.current() is None
    assert turn_question.with_question(PROMPT, "image") is PROMPT


def test_the_question_rides_under_the_prompt_as_context():
    with turn_question.bound(QUESTION):
        out = turn_question.with_question(PROMPT, "region")
    assert out.startswith(PROMPT)
    tail = out[len(PROMPT):]
    assert QUESTION in tail
    # The instructions say which is which.
    assert "is what to answer" in tail and "For context only" in tail
    assert "Do not answer it instead of the request" in tail
    assert turn_question.current() is None            # unbound on exit


def test_the_question_is_capped():
    long = "Made-up words about a sketch. " * 100
    with turn_question.bound(long):
        got = turn_question.current()
        assert len(got) == turn_question.MAX_CHARS and got.endswith("…")
        assert got in turn_question.with_question(PROMPT, "page")


def test_only_readings_for_the_main_model_get_it():
    with turn_question.bound(QUESTION):
        for kind in ("check", "find", "probe"):
            assert turn_question.with_question(PROMPT, kind) is PROMPT
        for kind in turn_question.QUESTION_KINDS:
            assert QUESTION in turn_question.with_question(PROMPT, kind)


def test_a_prompt_that_already_carries_the_question_is_left_alone():
    with turn_question.bound(QUESTION):
        same = f"{QUESTION}\n(read it off the sketch)"
        assert turn_question.with_question(same, "image") is same


def test_ask_vision_sends_it_and_says_so_on_the_record():
    from funhouse_agent.deep.vision_engine import current_side_call
    labels = []

    class Labelled(Engine):
        def analyze_image(self, image_bytes, prompt=""):
            labels.append(dict(current_side_call() or {}))
            return super().analyze_image(image_bytes, prompt)

    engine = Labelled()
    with turn_question.bound(QUESTION):
        ask_vision(engine, _png(), PROMPT, "image")
        ask_vision(engine, _png(), PROMPT, "check")
    assert QUESTION in engine.prompts[0]
    assert engine.prompts[1] == PROMPT
    assert labels[0]["user_question_chars"] == len(QUESTION)
    assert "user_question_chars" not in labels[1]


# -- through the tools --------------------------------------------------------

def test_analyze_image_reads_the_turn_question():
    engine = Engine()
    with turn_question.bound(QUESTION):
        out = json.loads(dispatch_extended_tool(
            "analyze_image", {"attachment_key": "sketch.png",
                              "prompt": PROMPT}, engine,
            {"sketch.png": _png()}))
    assert "error" not in out
    assert engine.prompts and all(PROMPT in p and QUESTION in p
                                  for p in engine.prompts)
    # The main model's request comes first; the question after it.
    p = engine.prompts[0]
    assert p.index(PROMPT) < p.index(QUESTION)


def test_render_region_on_an_image_reads_the_turn_question():
    engine = Engine()
    with turn_question.bound(QUESTION):
        out = json.loads(dispatch_extended_tool(
            "render_region", {"attachment_key": "sketch.png",
                              "prompt": PROMPT, "bbox": [10, 10, 200, 150]},
            engine, {"sketch.png": _png()}))
    assert "error" not in out
    assert engine.prompts and QUESTION in engine.prompts[-1]


def test_render_region_and_a_page_read_on_a_pdf_read_it():
    engine = Engine()
    pdf = _pdf()
    with turn_question.bound(QUESTION):
        dispatch_extended_tool(
            "render_region", {"attachment_key": "note.pdf", "page": 0,
                              "prompt": PROMPT, "bbox": [60, 70, 320, 120]},
            engine, {"note.pdf": pdf})
        dispatch_extended_tool(
            "analyze_pdf_page", {"attachment_key": "note.pdf", "page": 0,
                                 "prompt": PROMPT, "tiles": "off"},
            engine, {"note.pdf": pdf})
    assert len(engine.prompts) >= 2
    assert all(QUESTION in p for p in engine.prompts)


def test_a_tool_run_in_a_copied_context_sees_the_question():
    # LangGraph runs tools in threads with a COPY of the turn's context,
    # as it does for the bound working folder: no argument is needed.
    engine = Engine()

    def tool():
        return dispatch_extended_tool(
            "analyze_image", {"attachment_key": "sketch.png",
                              "prompt": PROMPT}, engine,
            {"sketch.png": _png()})

    with turn_question.bound(QUESTION):
        ctx = contextvars.copy_context()
    with ThreadPoolExecutor(1) as pool:
        pool.submit(ctx.run, tool).result()
    assert QUESTION in engine.prompts[0]


def test_no_question_bound_sends_the_prompt_alone():
    engine = Engine()
    dispatch_extended_tool(
        "analyze_image", {"attachment_key": "sketch.png", "prompt": PROMPT},
        engine, {"sketch.png": _png()})
    assert engine.prompts and not any("For context only" in p
                                      for p in engine.prompts)


# -- the host binds it per turn ----------------------------------------------

def test_the_app_turn_binds_the_users_own_message(tmp_path, monkeypatch):
    import webapp.core as core
    import webapp.turn_jobs as tj
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    tj._JOBS.clear()
    seen = []

    def stream_turn(agent, messages, thread_id, recursion_limit=None, **_kw):
        seen.append(turn_question.current())
        yield {"kind": "turn_done", "answer": "ok", "turn_tokens": 1}

    monkeypatch.setattr(core, "stream_turn", stream_turn)
    tid = "TQ1"
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    transcript = [{"role": "user", "text": QUESTION}]
    ctx = {"prompt": QUESTION, "temp_dir": files,
           "before": core.snapshot_dir(files), "staged_inputs": set(),
           "working_dir": files, "before_wd": None, "artifacts": [],
           "artifacts_before_len": 0, "transcript": transcript,
           "trace_on": False, "model": "test-model",
           "behavior": core.default_behavior()}
    core.begin_partial(tid, QUESTION)
    job = tj.start_turn_job(agent=object(),
                            messages=[{"role": "user", "content": QUESTION}],
                            thread_id=tid, recursion_limit=50, ctx=ctx)
    t0 = time.time()
    while not job.done and time.time() - t0 < 10:
        time.sleep(0.02)
    assert job.done
    assert seen == [QUESTION]
    assert turn_question.current() is None     # the caller's context untouched
