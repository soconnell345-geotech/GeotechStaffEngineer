"""What the record can answer: cost, cause, who did what (BACKLOG EV-14).

Field session 2026-10-14 (TRACE_REVIEW §6): ``usage`` null on every call on
the Prompter route; helper events carried no helper id; a 429 left only its
exception text; ``meta.json`` named the model only; tool durations mixed
work with queueing for a vision slot or the document lock; a 50,529-
character helper answer was cut at 32,000 and a summary at 8,000; vision
calls were bare ``model_start`` records with ``n_messages: 1``.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.tools import StructuredTool

import webapp.core as core
import webapp.turn_jobs as tj
from webapp import activity_log as al

pytest.importorskip("openai")
from webapp.engine_config import sdk_http_package  # noqa: E402

# the HTTP package openai's clients are built on (httpx; anthropic is on
# httpx2): transports and responses must come from the same one
httpx = sdk_http_package(__import__("openai"))


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    monkeypatch.delenv("GEOTECH_TOKENS_PER_MINUTE", raising=False)
    tj._JOBS.clear()
    yield


# ---------------------------------------------------------------------------
# Usage on the Prompter route (stream_usage)
# ---------------------------------------------------------------------------

def _sse(chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return body + "data: [DONE]\n\n"


def _stream_handler(sent, refuse_first=False):
    def handler(request):
        sent.append(json.loads(request.content or b"{}"))
        if refuse_first and len(sent) == 1:
            return httpx.Response(429, headers={"retry-after-ms": "1",
                                                "x-ratelimit-remaining-tokens":
                                                    "0"},
                                  json={"error": {
                                      "code": "rate_limit_exceeded",
                                      "message": "Too Many Requests"}})
        req = sent[-1]
        base = {"id": "c1", "object": "chat.completion.chunk", "created": 0,
                "model": "m"}
        chunks = [{**base, "choices": [{"index": 0, "delta": {
                       "role": "assistant", "content": "The answer."},
                       "finish_reason": None}]},
                  {**base, "choices": [{"index": 0, "delta": {},
                                        "finish_reason": "stop"}]}]
        if (req.get("stream_options") or {}).get("include_usage"):
            chunks.append({**base, "choices": [], "usage": {
                "prompt_tokens": 1200, "completion_tokens": 30,
                "total_tokens": 1230}})
        return httpx.Response(200, headers={"content-type":
                                            "text/event-stream"},
                              text=_sse(chunks))
    return handler


SECRET = "sk-test-NEVER-RECORDED-0123456789"


def _prompter_model(monkeypatch, handler, retries="0"):
    import webapp.tinyapps_engine as te
    real_init = httpx.Client.__init__

    def mocked_init(self, *a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        real_init(self, *a, **kw)

    # the constructor, not the class: the Prompter client is the openai
    # SDK's own class (a subclass of the real one), and a package first
    # imported while the class is swapped would subclass the swap for good
    monkeypatch.setattr(httpx.Client, "__init__", mocked_init)
    monkeypatch.setenv(te.MAX_RETRIES_ENV, retries)
    return te.build_chat_model(prompter=te.PrompterSettings(
        url="https://prompter.test/api/v1/chat/completions", model="m",
        api_key=SECRET))


def _turn(agent, tid, prompt="q"):
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    transcript = [{"role": "user", "text": prompt}]
    ctx = {"prompt": prompt, "temp_dir": files,
           "before": core.snapshot_dir(files), "staged_inputs": set(),
           "working_dir": files, "before_wd": None, "artifacts": [],
           "artifacts_before_len": 0, "transcript": transcript,
           "trace_on": True, "model": "m",
           "behavior": core.default_behavior()}
    job = tj.start_turn_job(agent, [{"role": "user", "content": prompt}],
                            tid, 40, ctx)
    t0 = time.time()
    while not job.done and time.time() - t0 < 30:
        time.sleep(0.05)
    assert job.done
    return job, al.load(core.conversation_dir(tid))


def _agent(model):
    from funhouse_agent.deep.agent import build_deep_agent
    return build_deep_agent(model=model, reference_mode="off")


def test_streamed_calls_on_the_prompter_route_carry_usage(monkeypatch):
    """Before (stream_usage off, as langchain-openai leaves it for a custom
    base_url): usage null and turn tokens 0. Now: the provider's count."""
    sent = []
    monkeypatch.setenv("GEOTECH_PROMPTER_STREAM_USAGE", "0")
    job, recs = _turn(_agent(_prompter_model(monkeypatch,
                                             _stream_handler(sent))), "U0")
    (end,) = [r for r in recs if r["event"] == "model_end"]
    assert end["usage"] is None and job.result["turn_tokens"] == 0
    assert "stream_options" not in sent[0]

    sent.clear()
    monkeypatch.delenv("GEOTECH_PROMPTER_STREAM_USAGE")
    job, recs = _turn(_agent(_prompter_model(monkeypatch,
                                             _stream_handler(sent))), "U1")
    assert sent[0]["stream"] is True
    assert sent[0]["stream_options"] == {"include_usage": True}
    (end,) = [r for r in recs if r["event"] == "model_end"]
    assert end["usage"]["input_tokens"] == 1200
    assert end["usage"]["total_tokens"] == 1230
    assert job.result["turn_tokens"] == 1230


def test_a_gateway_that_refuses_stream_options_costs_no_turn(monkeypatch):
    """The usage option refused (a 400 naming it): the call is asked again
    without it, and the model stops sending it."""
    sent = []
    ok = _stream_handler([])

    def handler(request):
        body = json.loads(request.content or b"{}")
        sent.append(body)
        if "stream_options" in body:
            return httpx.Response(400, json={"error": {
                "message": "Unrecognized request argument supplied: "
                           "stream_options", "code": "invalid_request"}})
        return ok(request)

    model = _prompter_model(monkeypatch, handler)
    job, recs = _turn(_agent(model), "U4")
    assert job.result["error"] is None
    assert job.result["final"] == "The answer."
    assert ["stream_options" in b for b in sent] == [True, False]
    assert model.stream_usage is False


def test_a_calls_http_tries_and_rate_limit_headers_are_kept(monkeypatch):
    """The client's own retry inside one model call: both HTTP answers and
    the 429's rate-limit headers are on the call's record."""
    sent = []
    model = _prompter_model(monkeypatch, _stream_handler(
        sent, refuse_first=True), retries="1")
    _job, recs = _turn(_agent(model), "U2")
    (end,) = [r for r in recs if r["event"] == "model_end"]
    assert end["http_tries"] == 2
    assert [t["status"] for t in end["http"]] == [429, 200]
    assert end["http"][0]["headers"]["x-ratelimit-remaining-tokens"] == "0"


def test_the_engine_and_policy_are_in_meta_and_turn_start(monkeypatch):
    model = _prompter_model(monkeypatch, _stream_handler([]))
    _job, recs = _turn(_agent(model), "U3")
    meta = core.load_meta("U3")
    eng = meta["engine"]
    assert eng["class"] == "PrompterChatOpenAI" and eng["model"] == "m"
    assert eng["client_max_retries"] == 0 and eng["stream_usage"] is True
    pol = eng["policy"]
    assert pol["agent_busy_tries"] == 5 and pol["max_helpers"] == 2
    assert pol["agent_ladder_s"] == [15.0, 30.0, 60.0, 120.0]
    assert pol["vision_max_inflight"] == 8
    start = next(r for r in recs if r["event"] == "turn_start")
    assert start["engine"] == eng
    for where in (json.dumps(eng), json.dumps(recs)):  # never the key
        assert SECRET not in where and "prompter.test" not in where


# ---------------------------------------------------------------------------
# Who did what: helpers
# ---------------------------------------------------------------------------

def test_every_event_inside_a_helper_names_its_task(tmp_path):
    from funhouse_agent._fileio import working_dir_bound
    from funhouse_agent.deep.agent import build_deep_agent
    from funhouse_agent.deep.tests.test_keep_work_offline import (
        Played, _big_read_tool)
    agent = build_deep_agent(
        model=Played(briefs=["HELPER answer: logs A",
                             "HELPER answer: logs B"], reads=1),
        extra_tools=[_big_read_tool(rows=800)], reference_mode="off")
    log = al.ActivityLogger(str(tmp_path / "c"), turn=1)
    with working_dir_bound(str(tmp_path)):
        agent.invoke({"messages": [{"role": "user", "content": "PRIMARY"}]},
                     config={"recursion_limit": 80, "callbacks": [log]})
    recs = al.load(str(tmp_path / "c"))
    inside = [r for r in recs if r["agent"] == "general-purpose"]
    assert inside and all(r.get("task_id") in ("task_0", "task_1")
                          for r in inside)
    by_task = {r["task_id"]: r["helper"] for r in inside}
    assert by_task == {"task_0": "HELPER answer: logs A",
                       "task_1": "HELPER answer: logs B"}
    primary_model = [r for r in recs if r["agent"] == "primary"
                     and r["event"] == "model_start"]
    assert primary_model and not any("task_id" in r for r in primary_model)
    # a result over 32,000 characters is kept whole beside the log
    big = [r for r in inside if r["event"] == "tool_end"
           and r["name"] == "big_read"]
    assert big and all(r["truncated"] and r["full"] for r in big)
    whole = open(os.path.join(str(tmp_path / "c"), big[0]["full"]),
                 encoding="utf-8").read()
    assert len(whole) == big[0]["chars"] > al.DEFAULT_MAX_CHARS


def test_a_long_model_text_is_kept_whole(tmp_path):
    log = al.ActivityLogger(str(tmp_path), turn=2)
    log.on_chat_model_start({}, [[]], run_id="r1")
    text = "summary line\n" * 1_000                   # 13,000 characters
    log.on_llm_end(LLMResult(generations=[[ChatGeneration(
        message=AIMessage(content=text))]]), run_id="r1")
    (rec,) = [r for r in al.load(str(tmp_path)) if r["event"] == "model_end"]
    assert rec["text_truncated"] and rec["text_full"]
    assert open(os.path.join(str(tmp_path), rec["text_full"]),
                encoding="utf-8").read() == text.strip()


# ---------------------------------------------------------------------------
# Cause: refusals
# ---------------------------------------------------------------------------

def test_a_model_error_keeps_the_refusals_facts(tmp_path):
    class _H(dict):
        def get(self, k, d=None):
            return super().get(k.lower(), d)

    class RateLimitError(Exception):
        status_code = 429

    exc = RateLimitError("Too Many Requests")
    exc.response = type("R", (), {"status_code": 429, "headers": _H({
        "retry-after": "20", "x-ratelimit-limit-tokens": "300000",
        "x-ratelimit-remaining-tokens": "0"})})()
    exc.geotech_busy = {"tries": 5, "waited_s": 225.0, "kind": "rate limit",
                        "why": "the ladder ran out"}
    log = al.ActivityLogger(str(tmp_path), turn=1)
    log.on_chat_model_start({}, [[]], run_id="m1")
    log.on_llm_error(exc, run_id="m1")
    (rec,) = [r for r in al.load(str(tmp_path)) if r["event"] == "model_error"]
    assert rec["status"] == 429 and rec["retry_after_s"] == 20.0
    assert rec["headers"]["x-ratelimit-limit-tokens"] == "300000"
    assert rec["busy_ladder"]["tries"] == 5


# ---------------------------------------------------------------------------
# Queueing: vision slots and the document lock; vision tied to its tool
# ---------------------------------------------------------------------------

class _Seen(BaseChatModel):
    @property
    def _llm_type(self):
        return "seen"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="a reading"))])


def _png(w=40, h=30):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_a_vision_call_is_tied_to_its_tool_page_and_tile(tmp_path,
                                                         monkeypatch):
    from funhouse_agent.deep.vision_engine import LangChainVisionEngine
    from funhouse_agent.vision_tools import ask_vision
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    engine = LangChainVisionEngine(_Seen(), detail="")

    def look(page: int) -> str:
        """Look at one tile of a page."""
        return ask_vision(engine, _png(), "read it", "tile",
                          where={"page": page, "tile": "r2c3"})

    tool = StructuredTool.from_function(look, name="analyze_pdf_page")
    log = al.ActivityLogger(str(tmp_path), turn=1)
    tool.invoke({"page": 24}, config={"callbacks": [log]})
    recs = al.load(str(tmp_path))
    (start,) = [r for r in recs if r["event"] == "model_start"]
    assert start["tool"] == "analyze_pdf_page"
    v = start["vision"]
    assert (v["kind"], v["page"], v["tile"]) == ("tile", 24, "r2c3")
    assert v["image_px"] == [40, 30] and "queued_s" in v
    (end,) = [r for r in recs if r["event"] == "tool_end"]
    assert end["vision_calls"] == 1 and end["vision_queued_s"] >= 0


def test_a_vision_slot_wait_is_measured(monkeypatch):
    from funhouse_agent.deep import vision_engine as ve
    monkeypatch.setenv(ve.INFLIGHT_ENV, "1")
    monkeypatch.setattr(ve, "_slots", None)
    held = threading.Event()
    free = threading.Event()

    def holder():
        with ve.call_slot("c"):
            held.set()
            free.wait(2)

    t = threading.Thread(target=holder)
    t.start()
    held.wait(2)
    threading.Timer(0.3, free.set).start()
    with ve.side_call(kind="tile") as info:
        with ve.call_slot("c"):
            pass
    t.join(2)
    assert info["queued_s"] >= 0.25


def test_a_wait_for_the_document_lock_is_measured_and_logged(tmp_path):
    from funhouse_agent import document_tools as dt
    raw = threading.RLock()

    class Entry:
        lock = raw

    class Kit:
        def _entry(self, handle):
            return Entry

    class Space:
        kit = Kit()

    dt._time_document_lock(Space(), {"handle": "doc_1"})
    timed = Entry.lock
    assert isinstance(timed, dt._TimedLock)
    held = threading.Event()

    def holder():
        with raw:
            held.set()
            time.sleep(0.3)

    th = threading.Thread(target=holder)
    th.start()
    held.wait(2)
    tally = [0.0]
    token = dt._LOCK_WAIT.set(tally)
    try:
        with timed:
            with timed:                       # re-entrant, as planlens nests
                pass
    finally:
        dt._LOCK_WAIT.reset(token)
    th.join(2)
    assert tally[0] >= 0.25

    # ... and the activity log puts it on the tool call's tool_end
    log = al.ActivityLogger(str(tmp_path), turn=1)
    log.on_tool_start({"name": "log_grid"}, "", run_id="t1", inputs={})
    log.on_custom_event("document_lock_wait",
                        {"tool": "log_grid", "waited_s": 14.5}, run_id="t1")
    log.on_tool_end("rows", run_id="t1")
    (end,) = [r for r in al.load(str(tmp_path)) if r["event"] == "tool_end"]
    assert end["lock_wait_s"] == 14.5
