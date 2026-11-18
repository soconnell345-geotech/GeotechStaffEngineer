"""BACKLOG INF-66: the run record tells a reported cache count from none.

Every ``model_end``'s ``usage`` has carried ``cache_read`` and
``cache_creation`` since 2026-11-05, and a reply that reports no cache count
records 0 for both. A 0 alone could not say "nothing was read from the cache"
apart from "the reply did not say". Now a call whose reply carried a cache
count (a reported 0 included) also has ``cache_reported: true`` beside its
``usage``, and ``turn_end`` has ``cache_reported_calls``. A turn whose
replies carry no cache count writes the record it wrote before, byte for
byte. Every usage object here is made up.
"""

import json
import uuid
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from aec_agent_core import activity_log as al
from aec_agent_core import palantir_sdk_engine as pse

RIDS = [uuid.UUID(int=n) for n in range(1, 9)]


def _call(log, rid, message, *, llm_output=None):
    log.on_chat_model_start({}, [[]], run_id=rid)
    gens = [] if message is None else [[ChatGeneration(message=message)]]
    log.on_llm_end(LLMResult(generations=gens, llm_output=llm_output),
                   run_id=rid)


def _unreported_replies():
    """Replies of four routes that carry token counts and no cache count:
    LangChain usage alone; langchain-anthropic's usage dict as a proxy that
    dropped the cache fields leaves it (both None); an OpenAI-style usage
    whose ``cached_tokens`` is None; and ``llm_output`` alone."""
    yield AIMessage(content="", usage_metadata={
        "input_tokens": 120, "output_tokens": 30, "total_tokens": 150}), None
    yield AIMessage(
        content="",
        usage_metadata={"input_tokens": 210, "output_tokens": 20,
                        "total_tokens": 230, "input_token_details": {}},
        response_metadata={"usage": {
            "input_tokens": 210, "output_tokens": 20,
            "cache_read_input_tokens": None,
            "cache_creation_input_tokens": None}}), None
    yield AIMessage(
        content="",
        usage_metadata={"input_tokens": 300, "output_tokens": 10,
                        "total_tokens": 310},
        response_metadata={"token_usage": {
            "prompt_tokens": 300, "completion_tokens": 10,
            "total_tokens": 310,
            "prompt_tokens_details": {"cached_tokens": None,
                                      "audio_tokens": None}}}), None
    yield None, {"token_usage": {"prompt_tokens": 40, "completion_tokens": 5,
                                 "total_tokens": 45}}


def _model_end_line(rid, usage):
    """A ``model_end`` line exactly as the record wrote it before the mark
    (keys in their order; fixed clock, no helper, no text, no tool call)."""
    return json.dumps({"turn": 1, "ts": 1000.0, "t": 0.0, "agent": "primary",
                       "event": "model_end", "run_id": str(rid),
                       "parent_run_id": None, "usage": usage,
                       "n_tool_calls": 0, "tool_names": [],
                       "duration_s": 0.0}, ensure_ascii=False)


def test_a_turn_without_cache_counts_writes_todays_record(tmp_path):
    """No reply reports a cache count: every ``model_end`` line is the line
    the record wrote before (the two zeros, no mark), and ``turn_end`` has
    today's ``usage`` and no ``cache_reported_calls``."""
    log = al.ActivityLogger(str(tmp_path), turn=1, clock=lambda: 1000.0)
    log.turn_start(prompt="q")
    for rid, (message, llm_output) in zip(RIDS, _unreported_replies()):
        _call(log, rid, message, llm_output=llm_output)
    log.turn_end(turn_tokens=735)
    with open(al.activity_path(str(tmp_path)), encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    ends = [ln for ln in lines if '"event": "model_end"' in ln]
    zero = {"cache_read": 0, "cache_creation": 0}
    assert ends == [
        _model_end_line(RIDS[0], {"input_tokens": 120, "output_tokens": 30,
                                  "total_tokens": 150, **zero}),
        _model_end_line(RIDS[1], {"input_tokens": 210, "output_tokens": 20,
                                  "total_tokens": 230, **zero}),
        _model_end_line(RIDS[2], {"input_tokens": 300, "output_tokens": 10,
                                  "total_tokens": 310, **zero}),
        _model_end_line(RIDS[3], {"input_tokens": 40, "output_tokens": 5,
                                  "total_tokens": 45, **zero})]
    end = json.loads(lines[-1])
    assert end["event"] == "turn_end"
    assert end["usage"] == {"input_tokens": 670, "output_tokens": 65,
                            "cache_read": 0, "cache_creation": 0, "calls": 4}
    assert "cache_reported_calls" not in end
    assert "cache_reported" not in json.dumps(lines)


def test_a_reported_zero_is_marked(tmp_path):
    """langchain-anthropic's own builder fed a reply whose cache counts are
    0: the same usage as before, and the mark says the 0 is a count."""
    ca = pytest.importorskip("langchain_anthropic.chat_models")
    um = ca._create_usage_metadata(NS(
        input_tokens=500, output_tokens=40, cache_read_input_tokens=0,
        cache_creation_input_tokens=0, cache_creation=None,
        output_tokens_details=None))
    log = al.ActivityLogger(str(tmp_path))
    _call(log, RIDS[0], AIMessage(content="", usage_metadata=um))
    rec = al.load(str(tmp_path))[-1]
    assert rec["usage"] == {"input_tokens": 500, "output_tokens": 40,
                            "total_tokens": 540, "cache_read": 0,
                            "cache_creation": 0}
    assert rec["cache_reported"] is True


def test_a_proxy_that_drops_the_counts_is_not_marked(tmp_path):
    """The same builder fed a reply without the cache fields (None), as a
    proxy may pass it on: zeros as before, and no mark."""
    ca = pytest.importorskip("langchain_anthropic.chat_models")
    um = ca._create_usage_metadata(NS(
        input_tokens=500, output_tokens=40, cache_read_input_tokens=None,
        cache_creation_input_tokens=None, cache_creation=None,
        output_tokens_details=None))
    log = al.ActivityLogger(str(tmp_path))
    _call(log, RIDS[0], AIMessage(content="", usage_metadata=um))
    rec = al.load(str(tmp_path))[-1]
    assert rec["usage"]["cache_read"] == 0
    assert rec["usage"]["cache_creation"] == 0
    assert "cache_reported" not in rec


def test_foundry_claude_sdk_reports_a_zero_and_no_count_apart():
    """The Foundry Claude SDK model's converter keeps a reported 0 as a
    count (``input_token_details``) and leaves it out when the SDK's usage
    has no cache fields; the input total is the same either way."""
    zero = pse._claude_usage(NS(input_tokens=9, output_tokens=3,
                                cache_read_input_tokens=0,
                                cache_creation_input_tokens=0))
    none = pse._claude_usage(NS(input_tokens=9, output_tokens=3))
    assert zero == {"input_tokens": 9, "output_tokens": 3,
                    "total_tokens": 12,
                    "input_token_details": {"cache_read": 0,
                                            "cache_creation": 0}}
    assert none == {"input_tokens": 9, "output_tokens": 3,
                    "total_tokens": 12}
    read = pse._claude_usage(NS(input_tokens=9, output_tokens=3,
                                cache_read_input_tokens=700,
                                cache_creation_input_tokens=None))
    assert read["input_tokens"] == 709
    assert read["input_token_details"] == {"cache_read": 700,
                                           "cache_creation": 0}


def test_a_none_count_does_not_hide_a_real_one_further_out(tmp_path):
    """A usage dict near the message with the cache fields None no longer
    stops the search: the count in ``llm_output`` is recorded (it was 0)."""
    log = al.ActivityLogger(str(tmp_path))
    msg = AIMessage(
        content="", usage_metadata={"input_tokens": 90, "output_tokens": 10,
                                    "total_tokens": 100},
        response_metadata={"usage": {"cache_read_input_tokens": None,
                                     "cache_creation_input_tokens": None}})
    _call(log, RIDS[0], msg, llm_output={"usage": {
        "cache_read_input_tokens": 70, "cache_creation_input_tokens": 5}})
    rec = al.load(str(tmp_path))[-1]
    assert rec["usage"]["cache_read"] == 70
    assert rec["usage"]["cache_creation"] == 5
    assert rec["cache_reported"] is True


def test_turn_end_counts_the_calls_that_reported(tmp_path):
    """Three calls: a write, a read, and one whose reply had no cache count.
    The sums are as before; ``cache_reported_calls`` says two of the three
    counted, so the third's 0 is no count."""
    log = al.ActivityLogger(str(tmp_path), turn=1)
    log.turn_start(prompt="q")
    for rid, details in zip(RIDS, ({"cache_read": 0, "cache_creation": 800},
                                   {"cache_read": 800, "cache_creation": 60},
                                   None)):
        um = {"input_tokens": 900, "output_tokens": 50, "total_tokens": 950}
        if details is not None:
            um["input_token_details"] = details
        _call(log, rid, AIMessage(content="", usage_metadata=um))
    log.turn_end(turn_tokens=2_850)
    recs = al.load(str(tmp_path))
    marks = [r.get("cache_reported") for r in recs
             if r["event"] == "model_end"]
    assert marks == [True, True, None]
    end = recs[-1]
    assert end["usage"] == {"input_tokens": 2_700, "output_tokens": 150,
                            "cache_read": 800, "cache_creation": 860,
                            "calls": 3}
    assert end["cache_reported_calls"] == 2
