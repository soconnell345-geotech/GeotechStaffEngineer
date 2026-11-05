"""BACKLOG INF-66: ``activity.jsonl`` keeps the prompt-cache split.

Each ``model_end``'s ``usage`` carries ``cache_read`` and ``cache_creation``
(part of ``input_tokens``, as LangChain counts them), taken from whatever the
route reports, and ``turn_end`` sums them beside the input and output totals.
Every usage object here is made up.
"""

import json
import os
import uuid

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from webapp import activity_log as al


def _end(log, message=None, *, llm_output=None, generation_info=None):
    rid = uuid.uuid4()
    log.on_chat_model_start({}, [[]], run_id=rid)
    gens = [] if message is None else [[ChatGeneration(
        message=message, generation_info=generation_info)]]
    log.on_llm_end(LLMResult(generations=gens, llm_output=llm_output),
                   run_id=rid)
    return al.load(log.conv_dir)[-1]


def test_anthropic_route_through_langchain_anthropic(tmp_path):
    """langchain-anthropic's own usage builder, fed a made-up Anthropic
    usage object: the reads and writes counted into the input and kept."""
    ca = pytest.importorskip("langchain_anthropic.chat_models")
    from types import SimpleNamespace
    raw = SimpleNamespace(input_tokens=7, output_tokens=40,
                          cache_read_input_tokens=1_000,
                          cache_creation_input_tokens=300,
                          cache_creation=None, output_tokens_details=None)
    um = ca._create_usage_metadata(raw)
    rec = _end(al.ActivityLogger(str(tmp_path)),
               AIMessage(content="ok", usage_metadata=um))
    assert rec["usage"] == {"input_tokens": 1_307, "output_tokens": 40,
                            "total_tokens": 1_347, "cache_read": 1_000,
                            "cache_creation": 300}


def test_anthropic_cache_write_by_ttl(tmp_path):
    """A write reported per TTL: langchain-anthropic sets ``cache_creation``
    to 0 and puts the count under the ephemeral keys."""
    um = {"input_tokens": 905, "output_tokens": 5, "total_tokens": 910,
          "input_token_details": {"cache_read": 0, "cache_creation": 0,
                                  "ephemeral_5m_input_tokens": 600,
                                  "ephemeral_1h_input_tokens": 300}}
    rec = _end(al.ActivityLogger(str(tmp_path)),
               AIMessage(content="", usage_metadata=um))
    assert rec["usage"]["cache_read"] == 0
    assert rec["usage"]["cache_creation"] == 900


def test_openai_route_through_langchain_openai(tmp_path):
    """langchain-openai's usage builder over an OpenAI-style usage dict:
    ``prompt_tokens_details.cached_tokens`` is a read; no write."""
    oa = pytest.importorskip("langchain_openai.chat_models.base")
    um = oa._create_usage_metadata(
        {"prompt_tokens": 2_000, "completion_tokens": 50,
         "total_tokens": 2_050,
         "prompt_tokens_details": {"cached_tokens": 1_536}})
    rec = _end(al.ActivityLogger(str(tmp_path)),
               AIMessage(content="", usage_metadata=um))
    assert rec["usage"] == {"input_tokens": 2_000, "output_tokens": 50,
                            "total_tokens": 2_050, "cache_read": 1_536,
                            "cache_creation": 0}


def test_openai_service_tier_prefix(tmp_path):
    um = {"input_tokens": 100, "output_tokens": 1, "total_tokens": 101,
          "input_token_details": {"priority_cache_read": 64, "priority": 36}}
    rec = _end(al.ActivityLogger(str(tmp_path)),
               AIMessage(content="", usage_metadata=um))
    assert rec["usage"]["cache_read"] == 64
    assert rec["usage"]["cache_creation"] == 0


def test_foundry_claude_sdk_model_usage(tmp_path):
    """The Foundry Claude SDK model's own converter (``_claude_usage``)."""
    pse = pytest.importorskip("webapp.palantir_sdk_engine")
    from types import SimpleNamespace
    um = pse._claude_usage(SimpleNamespace(
        input_tokens=4, output_tokens=12, cache_read_input_tokens=800,
        cache_creation_input_tokens=50))
    rec = _end(al.ActivityLogger(str(tmp_path)),
               AIMessage(content="", usage_metadata=um))
    assert rec["usage"]["input_tokens"] == 854
    assert rec["usage"]["cache_read"] == 800
    assert rec["usage"]["cache_creation"] == 50


def _model_ends(log):
    return [r for r in al.load(log.conv_dir) if r["event"] == "model_end"]


def _prompter_model(response, **kwargs):
    """The real Prompter chat model (Databricks/Funhouse) over a stand-in
    client whose ``create`` returns ``response``: no network."""
    from types import SimpleNamespace
    from funhouse_agent.deep.databricks_bridge import PrompterChatModel
    create = lambda **request: response                 # noqa: E731
    prompter = SimpleNamespace(chat_model="made-up-gpt", client=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    return PrompterChatModel(prompter=prompter, **kwargs)


def test_prompter_route_keeps_the_cached_count(tmp_path):
    """An openai ``ChatCompletion`` whose usage reports a cache read, through
    the real bridge conversion and the logger. The read is part of the
    input, as OpenAI counts it."""
    chat = pytest.importorskip("openai.types.chat")
    resp = chat.ChatCompletion.model_validate({
        "id": "c1", "object": "chat.completion", "created": 0,
        "model": "made-up-gpt",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 2_000, "completion_tokens": 50,
                  "total_tokens": 2_050,
                  "prompt_tokens_details": {"cached_tokens": 1_536}}})
    log = al.ActivityLogger(str(tmp_path))
    _prompter_model(resp).invoke([HumanMessage(content="q")],
                                 config={"callbacks": [log]})
    assert _model_ends(log)[-1]["usage"] == {
        "input_tokens": 2_000, "output_tokens": 50, "total_tokens": 2_050,
        "cache_read": 1_536, "cache_creation": 0}


def test_prompter_stream_keeps_the_cached_count(tmp_path):
    """The same over the bridge's streaming path: the usage-only last chunk
    carries the read."""
    from types import SimpleNamespace as NS
    usage = NS(prompt_tokens=900, completion_tokens=10, total_tokens=910,
               prompt_tokens_details=NS(cached_tokens=512))
    chunks = iter([
        NS(model="made-up-gpt", usage=None, choices=[NS(
            delta=NS(content="ok", tool_calls=None), finish_reason="stop")]),
        NS(model="made-up-gpt", usage=usage, choices=[])])
    log = al.ActivityLogger(str(tmp_path))
    model = _prompter_model(chunks, streaming_enabled=True)
    for _ in model.stream([HumanMessage(content="q")],
                          config={"callbacks": [log]}):
        pass
    got = _model_ends(log)[-1]["usage"]
    assert got["input_tokens"] == 900
    assert got["cache_read"] == 512 and got["cache_creation"] == 0


def test_foundry_gpt_sdk_routes_keep_the_cached_count(tmp_path, monkeypatch):
    """Foundry GPT named by API name (``PalantirSdkChatModel``) over the
    stand-in SDK of ``test_palantir_sdk_engine``: the chat route's usage
    object, then the Responses route's ``input_tokens_details``."""
    pytest.importorskip("webapp.palantir_sdk_engine")
    from types import SimpleNamespace as NS
    from webapp.tests import test_palantir_sdk_engine as fake
    reply = fake._responses_reply("ok")
    reply.open_ai_responses.usage = NS(
        input_tokens=800, output_tokens=7, total_tokens=807,
        input_tokens_details=NS(cached_tokens=768))
    fake._install_fake_responses(monkeypatch, [reply])
    from webapp.palantir_sdk_engine import PalantirSdkChatModel
    fake._FakeSdkModel.next_response = NS(
        choices=[NS(finish_reason="stop",
                    message=NS(content="ok", tool_calls=None))],
        usage=NS(prompt_tokens=700, completion_tokens=9, total_tokens=709,
                 prompt_tokens_details=NS(cached_tokens=640)),
        model="GPT_5_1")
    log = al.ActivityLogger(str(tmp_path))
    PalantirSdkChatModel(model_api_name="GPT_5_1", route="chat").invoke(
        [HumanMessage(content="q")], config={"callbacks": [log]})
    PalantirSdkChatModel.from_handles(
        "GPT_5_4", vision_model=fake._Handle()).invoke(
        [HumanMessage(content="q")], config={"callbacks": [log]})
    chat_end, responses_end = _model_ends(log)
    assert chat_end["usage"]["input_tokens"] == 700
    assert chat_end["usage"]["cache_read"] == 640
    assert responses_end["usage"]["input_tokens"] == 800
    assert responses_end["usage"]["cache_read"] == 768


def test_raw_anthropic_usage_in_response_metadata(tmp_path):
    um = {"input_tokens": 500, "output_tokens": 9, "total_tokens": 509}
    msg = AIMessage(content="", usage_metadata=um, response_metadata={
        "usage": {"input_tokens": 100, "output_tokens": 9,
                  "cache_read_input_tokens": 350,
                  "cache_creation_input_tokens": 50}})
    rec = _end(al.ActivityLogger(str(tmp_path)), msg)
    assert rec["usage"]["input_tokens"] == 500       # LangChain's, unchanged
    assert rec["usage"]["cache_read"] == 350
    assert rec["usage"]["cache_creation"] == 50


def test_llm_output_only_routes(tmp_path):
    log = al.ActivityLogger(str(tmp_path))
    rec = _end(log, llm_output={"token_usage": {
        "prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 64}}})
    assert rec["usage"] == {"input_tokens": 90, "output_tokens": 10,
                            "total_tokens": 100, "cache_read": 64,
                            "cache_creation": 0}
    rec = _end(log, llm_output={"usage": {
        "input_tokens": 10, "output_tokens": 2,
        "cache_read_input_tokens": 70, "cache_creation_input_tokens": 5}})
    assert rec["usage"]["cache_read"] == 70
    assert rec["usage"]["cache_creation"] == 5


def test_no_cache_fields_records_zero_and_no_usage_stays_none(tmp_path):
    log = al.ActivityLogger(str(tmp_path))
    rec = _end(log, AIMessage(content="", usage_metadata={
        "input_tokens": 11, "output_tokens": 3, "total_tokens": 14}))
    assert rec["usage"]["cache_read"] == 0
    assert rec["usage"]["cache_creation"] == 0
    rec = _end(log, AIMessage(content=""))
    assert rec["usage"] is None


def test_turn_end_sums_three_calls(tmp_path):
    """A turn of three calls: the first writes the prefix, the next two read
    it; ``turn_end.usage`` sums every count, a helper's call included."""
    log = al.ActivityLogger(str(tmp_path), turn=1)
    log.turn_start(prompt="q")
    calls = [(1_000, 1_000, 0, 50), (1_100, 0, 1_000, 40),
             (1_250, 0, 1_100, 60)]
    task = uuid.uuid4()
    for i, (inp, made, read, out) in enumerate(calls):
        um = {"input_tokens": inp, "output_tokens": out,
              "total_tokens": inp + out,
              "input_token_details": {"cache_read": read,
                                      "cache_creation": made}}
        rid = uuid.uuid4()
        if i == 2:          # the third call runs inside a helper
            log.on_tool_start({"name": "task"}, "", run_id=task,
                              inputs={"subagent_type": "calc"})
            log.on_chat_model_start({}, [[]], run_id=rid, parent_run_id=task)
        else:
            log.on_chat_model_start({}, [[]], run_id=rid)
        log.on_llm_end(LLMResult(generations=[[ChatGeneration(
            message=AIMessage(content="", usage_metadata=um))]]),
            run_id=rid, parent_run_id=task if i == 2 else None)
    log.turn_end(turn_tokens=3_500)
    end = al.load(str(tmp_path))[-1]
    assert end["event"] == "turn_end"
    assert end["usage"] == {"input_tokens": 3_350, "output_tokens": 150,
                            "cache_read": 2_100, "cache_creation": 1_000,
                            "calls": 3}
    assert end["turn_tokens"] == 3_500


def test_turn_end_counts_unmetered_and_omits_usage_with_no_call(tmp_path):
    log = al.ActivityLogger(str(tmp_path / "a"))
    log.turn_end()
    assert "usage" not in al.load(log.conv_dir)[-1]
    log = al.ActivityLogger(str(tmp_path / "b"))
    _end(log, AIMessage(content=""))
    _end(log, AIMessage(content="", usage_metadata={
        "input_tokens": 5, "output_tokens": 1, "total_tokens": 6}))
    log.turn_end()
    assert al.load(log.conv_dir)[-1]["usage"] == {
        "input_tokens": 5, "output_tokens": 1, "cache_read": 0,
        "cache_creation": 0, "calls": 1, "unmetered_calls": 1}


def test_old_records_still_load_and_count(tmp_path):
    """An activity file written before INF-66 (no cache keys) loads, and the
    review-suite reader's token totals neither need nor double-count them."""
    from funhouse_agent.review_eval.runner import _activity
    old = [{"turn": 1, "event": "model_end", "agent": "primary",
            "usage": {"input_tokens": 100, "output_tokens": 10,
                      "total_tokens": 110}},
           {"turn": 1, "event": "turn_end", "agent": "primary",
            "turn_tokens": 110}]
    new = {"turn": 2, "event": "model_end", "agent": "primary",
           "usage": {"input_tokens": 200, "output_tokens": 20,
                     "total_tokens": 220, "cache_read": 150,
                     "cache_creation": 40}}
    with open(os.path.join(tmp_path, al.FILE_NAME), "w",
              encoding="utf-8") as fh:
        for rec in old + [new]:
            fh.write(json.dumps(rec) + "\n")
    assert len(al.load(str(tmp_path))) == 3
    facts = _activity(str(tmp_path))
    assert facts["model_calls"] == 2
    assert facts["tokens_in"] == 300 and facts["tokens_out"] == 30
