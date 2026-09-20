"""The Prompter engine offline: the wire shape, the schema, the metering.

This is the engine the numbers come from, so the translation into OpenAI's
shape is worth pinning line by line. Three differences from the Claude engine
carry real risk and each has a test: a tool result is its own message, a tool
message's content must be a STRING so a rendered page has to travel
separately, and structured output is a strict JSON schema validated here
rather than a parsed object handed back by the SDK.
"""

from __future__ import annotations

import base64
import json

import pytest
from pydantic import BaseModel, Field

from report_ingest.engine import (
    CostMeter, PROMPTER_MODELS, PROMPTER_PRICES, PrompterEngine, Usage,
    image_block, price_for, strict_schema, text_block, tool_result_block,
    tool_use_block, user,
)


#: The leading bytes of each format, which is all either engine reads to
#: decide what it is sending.
PNG_BYTES = b"\x89PNG-ish"
JPEG_BYTES = b"\xff\xd8\xff-ish"


# -- a Prompter that records what it was asked --------------------------------

class _Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _Call:
    def __init__(self, cid, name, arguments):
        self.id = cid
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class _Details:
    def __init__(self, cached):
        self.cached_tokens = cached


class _Usage:
    def __init__(self, prompt=0, completion=0, cached=0):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = _Details(cached)


class _Response:
    def __init__(self, message, usage=None, model="gpt-4o-2024-11-20",
                 finish_reason="stop"):
        self.choices = [_Choice(message, finish_reason)]
        self.usage = usage or _Usage(100, 20)
        self.model = model


class _Completions:
    def __init__(self, response):
        self.response = response
        self.last = None

    def create(self, **kwargs):
        self.last = kwargs
        return self.response


class _Chat:
    def __init__(self, response):
        self.completions = _Completions(response)


class _Client:
    def __init__(self, response):
        self.chat = _Chat(response)


class FakePrompter:
    """Stands in for ``fh_prompter``: a raw client plus a ``chat`` method."""

    def __init__(self, response, with_client=True):
        self.response = response
        self._client = _Client(response) if with_client else None
        self.chat_calls = []

    def get_openai_instance(self):
        return self._client

    @property
    def client(self):
        return self._client

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return self.response


def _engine(response, with_client=True, **kw):
    prompter = FakePrompter(response, with_client=with_client)
    engine = PrompterEngine(prompter, meter=CostMeter(), **kw)
    return engine, prompter


def _sent(prompter):
    return prompter.client.chat.completions.last


# -- the model tiers ----------------------------------------------------------

def test_the_three_funhouse_tiers_are_the_model_names():
    assert PROMPTER_MODELS == ("funhouse-gpt-low", "funhouse-gpt-medium",
                               "funhouse-gpt-high")


def test_a_funhouse_tier_has_no_price_so_a_run_reports_tokens_not_dollars():
    assert Usage(input_tokens=10**6).dollars("funhouse-gpt-medium") == 0.0


def test_the_reply_records_the_deployment_that_actually_served_it():
    # A tier is an alias and the model behind it changes without notice, so a
    # scorecard has to keep what answered, not only what was asked for.
    engine, _ = _engine(_Response(_Message("hi"), model="gpt-4o-2024-11-20"))
    reply = engine.complete([user(text_block("go"))], tools=[{
        "name": "t", "description": "d", "input_schema": {}}])
    assert reply.model == "gpt-4o-2024-11-20"
    assert engine.served_by == "gpt-4o-2024-11-20"


# -- the wire shape -----------------------------------------------------------

def test_a_system_prompt_becomes_the_first_message():
    engine, prompter = _engine(_Response(_Message("ok")))
    engine.complete([user(text_block("go"))], system="be careful",
                    tools=[{"name": "t", "description": "d",
                            "input_schema": {}}])
    messages = _sent(prompter)["messages"]
    assert messages[0] == {"role": "system", "content": "be careful"}
    assert messages[1]["role"] == "user"


def test_tools_are_sent_in_openai_function_shape():
    engine, prompter = _engine(_Response(_Message("ok")))
    engine.complete([user(text_block("go"))], tools=[{
        "name": "read_page", "description": "the page's text",
        "input_schema": {"type": "object", "properties": {}}}])
    tool = _sent(prompter)["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "read_page"
    assert tool["function"]["parameters"] == {"type": "object",
                                              "properties": {}}
    assert _sent(prompter)["tool_choice"] == "auto"


def test_an_image_rides_as_a_data_uri_image_url():
    engine, prompter = _engine(_Response(_Message("ok")))
    engine.complete([user(text_block("look"))], images=[b"\x89PNG-ish"],
                    tools=[{"name": "t", "description": "d",
                            "input_schema": {}}])
    content = _sent(prompter)["messages"][-1]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    head, b64 = content[1]["image_url"]["url"].split(",", 1)
    assert head == "data:image/png;base64"
    assert base64.b64decode(b64) == b"\x89PNG-ish"


def test_a_jpeg_rides_as_a_jpeg_data_uri_not_a_png_one():
    """The ``png`` key holds a PNG or a JPEG, and the bytes decide which.

    The pages travel as JPEG since 5.21.2 -- a scanned page stored as PNG is
    a photograph in a lossless container, and thirty-six of them base64'd in
    one request is what the gateway refused. A JPEG announced as a PNG is a
    bad request, so the prefix is read off the bytes rather than assumed.
    """
    engine, prompter = _engine(_Response(_Message("ok")))
    engine.complete([user(text_block("look"), image_block(JPEG_BYTES))],
                    tools=[{"name": "t", "description": "d",
                            "input_schema": {}}])
    content = _sent(prompter)["messages"][-1]["content"]
    head, b64 = content[1]["image_url"]["url"].split(",", 1)
    assert head == "data:image/jpeg;base64"
    assert base64.b64decode(b64) == JPEG_BYTES


def test_a_png_block_still_rides_as_a_png_data_uri():
    engine, prompter = _engine(_Response(_Message("ok")))
    engine.complete([user(text_block("look"), image_block(PNG_BYTES))],
                    tools=[{"name": "t", "description": "d",
                            "input_schema": {}}])
    content = _sent(prompter)["messages"][-1]["content"]
    head, b64 = content[1]["image_url"]["url"].split(",", 1)
    assert head == "data:image/png;base64"
    assert base64.b64decode(b64) == PNG_BYTES


def test_the_claude_engine_names_the_media_type_off_the_bytes_too():
    from report_ingest.engine import ClaudeEngine

    png = ClaudeEngine._to_provider_block(image_block(PNG_BYTES))
    jpeg = ClaudeEngine._to_provider_block(image_block(JPEG_BYTES))
    assert png["source"]["media_type"] == "image/png"
    assert jpeg["source"]["media_type"] == "image/jpeg"


def test_an_unrecognised_picture_is_called_a_png_rather_than_refused():
    """A picture this package cannot name is still sent: the provider names
    a bad request, and raising here would lose a whole run over one page."""
    from report_ingest.engine import image_media_type

    assert image_media_type(b"GIF89a") == "image/png"
    assert image_media_type(b"") == "image/png"


def test_an_assistant_turn_with_tool_calls_round_trips():
    engine, prompter = _engine(_Response(_Message("ok")))
    engine.complete([
        user(text_block("go")),
        {"role": "assistant", "content": [
            text_block("looking"),
            tool_use_block("c1", "read_page", {"page": 7})]},
        {"role": "user", "content": [tool_result_block("c1", "page text")]},
    ], tools=[{"name": "read_page", "description": "d", "input_schema": {}}])
    messages = _sent(prompter)["messages"]
    assistant = messages[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "looking"
    call = assistant["tool_calls"][0]
    assert call["id"] == "c1" and call["type"] == "function"
    assert call["function"]["name"] == "read_page"
    assert json.loads(call["function"]["arguments"]) == {"page": 7}
    assert messages[2] == {"role": "tool", "tool_call_id": "c1",
                           "content": "page text"}


def test_a_tool_result_is_its_own_message_not_a_block_in_a_user_turn():
    engine, prompter = _engine(_Response(_Message("ok")))
    engine.complete([
        user(text_block("go")),
        {"role": "assistant", "content": [tool_use_block("c1", "t", {})]},
        {"role": "user", "content": [tool_result_block("c1", "a"),
                                     tool_result_block("c2", "b")]},
    ], tools=[{"name": "t", "description": "d", "input_schema": {}}])
    roles = [m["role"] for m in _sent(prompter)["messages"]]
    assert roles == ["user", "assistant", "tool", "tool"]


def test_a_picture_in_a_tool_result_travels_in_a_user_message_after_it():
    # An OpenAI tool message takes a string and nothing else, and the label
    # review's render_page answers with a picture. The tool message says the
    # picture follows; the picture follows.
    engine, prompter = _engine(_Response(_Message("ok")))
    engine.complete([
        user(text_block("go")),
        {"role": "assistant", "content": [tool_use_block("c1", "render", {})]},
        {"role": "user", "content": [tool_result_block(
            "c1", [text_block("page 7 at 80 dpi"), image_block(b"PNG")])]},
    ], tools=[{"name": "render", "description": "d", "input_schema": {}}])
    messages = _sent(prompter)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool",
                                             "user"]
    assert isinstance(messages[2]["content"], str)
    assert "picture follows" in messages[2]["content"]
    assert "page 7 at 80 dpi" in messages[2]["content"]
    kinds = [b["type"] for b in messages[3]["content"]]
    assert kinds == ["text", "image_url"]
    assert "c1" in messages[3]["content"][0]["text"]


def test_an_error_result_still_reaches_the_model_as_text():
    engine, prompter = _engine(_Response(_Message("ok")))
    engine.complete([
        user(text_block("go")),
        {"role": "assistant", "content": [tool_use_block("c1", "t", {})]},
        {"role": "user", "content": [tool_result_block(
            "c1", "page 9999 is outside this document", is_error=True)]},
    ], tools=[{"name": "t", "description": "d", "input_schema": {}}])
    tool_message = _sent(prompter)["messages"][2]
    assert "outside this document" in tool_message["content"]


def test_an_empty_tool_result_is_never_sent_as_an_empty_string():
    engine, prompter = _engine(_Response(_Message("ok")))
    engine.complete([
        user(text_block("go")),
        {"role": "assistant", "content": [tool_use_block("c1", "t", {})]},
        {"role": "user", "content": [tool_result_block("c1", "")]},
    ], tools=[{"name": "t", "description": "d", "input_schema": {}}])
    assert _sent(prompter)["messages"][2]["content"] == "(no content)"


def test_a_tool_use_block_cannot_be_smuggled_into_user_content():
    with pytest.raises(ValueError, match="separate messages"):
        PrompterEngine._content([tool_use_block("c1", "t", {})])


# -- structured output --------------------------------------------------------

class _Answer(BaseModel):
    verdict: str = Field(description="what it is")
    pages: int


def test_structured_output_is_sent_as_a_strict_json_schema():
    engine, prompter = _engine(_Response(_Message("ok")))
    engine.complete([user(text_block("go"))], output_format=_Answer,
                    tools=[{"name": "t", "description": "d",
                            "input_schema": {}}])
    fmt = _sent(prompter)["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "_Answer"
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"verdict", "pages"}


def test_the_json_that_comes_back_is_validated_into_the_model():
    body = json.dumps({"verdict": "geotechnical report", "pages": 94})
    engine, _ = _engine(_Response(_Message(body)))
    reply = engine.complete([user(text_block("go"))], output_format=_Answer,
                            tools=[{"name": "t", "description": "d",
                                    "input_schema": {}}])
    assert reply.parsed.verdict == "geotechnical report"
    assert reply.parsed.pages == 94


def test_json_that_does_not_fit_the_model_parses_as_nothing_not_as_garbage():
    engine, _ = _engine(_Response(_Message('{"verdict": "ok"}')))
    reply = engine.complete([user(text_block("go"))], output_format=_Answer,
                            tools=[{"name": "t", "description": "d",
                                    "input_schema": {}}])
    assert reply.parsed is None, (
        "a half-filled answer must read as no answer, so the caller raises "
        "rather than scoring a default")


def test_strict_schema_requires_every_field_of_every_nested_object():
    class Inner(BaseModel):
        a: str = "x"

    class Outer(BaseModel):
        inner: Inner
        note: str = "y"

    schema = strict_schema(Outer)
    assert schema["required"] == ["inner", "note"]
    assert schema["additionalProperties"] is False
    inner = schema["$defs"]["Inner"]
    assert inner["required"] == ["a"]
    assert inner["additionalProperties"] is False


def test_strict_schema_drops_the_keywords_strict_mode_refuses():
    """The first cluster run (2026-09-18): the three readers' schemas carried
    pydantic's default, minItems and maxItems, OpenAI refused the schema
    before any token was spent, and every reader call died with zero model
    calls. The limit survives as words in the description."""
    from typing import List, Optional

    class Read(BaseModel):
        blows: List[int] = Field(default_factory=list, min_length=1,
                                 max_length=3, description="the drives")
        depth: Optional[float] = Field(None, ge=0, le=100)
        code: str = Field("x", pattern="^[A-Z]{2}$")
        kind: str = Field("boring", description="what it is")

    schema = strict_schema(Read)
    text = json.dumps(schema)
    for word in ("default", "minItems", "maxItems", "minimum", "maximum",
                 "pattern"):
        assert f'"{word}"' not in text, word
    props = schema["properties"]
    assert props["blows"]["description"] == "the drives (1 to 3 items)"
    assert "(between 0 and 100)" in json.dumps(props["depth"])
    assert props["code"]["description"] == "(matching ^[A-Z]{2}$)"
    assert props["kind"]["description"] == "what it is"
    assert set(schema["required"]) == {"blows", "depth", "code", "kind"}


def test_a_fixed_length_tuple_becomes_a_plain_array_with_its_length_in_words():
    """The log and lab readers' provenance bbox is Tuple[float, float, float,
    float]; pydantic writes that as prefixItems (tuple validation), which
    strict mode does not implement."""
    from typing import Tuple

    class Prov(BaseModel):
        bbox: Tuple[float, float, float, float] = Field(
            description="left, top, right, bottom in page points")
        cell: Tuple[int, str] = (0, "")

    schema = strict_schema(Prov)
    assert "prefixItems" not in json.dumps(schema)
    bbox = schema["properties"]["bbox"]
    assert bbox["type"] == "array" and bbox["items"] == {"type": "number"}
    assert bbox["description"] == (
        "left, top, right, bottom in page points (exactly 4 items: number, "
        "number, number, number)")
    cell = schema["properties"]["cell"]
    assert cell["items"] == {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    assert Prov.model_validate_json('{"bbox": [1, 2, 3, 4], "cell": [1, "a"]}')


def test_every_structured_output_the_passes_send_is_clean_for_strict_mode():
    """Every pydantic model a pass hands to ``output_format`` must come out
    of strict_schema with none of the refused keywords, or the call is a
    Bad Request on the cluster and no offline test would ever notice."""
    import importlib

    from report_ingest.engine import STRICT_UNSUPPORTED

    refused = set(STRICT_UNSUPPORTED)

    def walk(node, hits):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in refused:
                    hits.add(key)
                walk(value, hits)
        elif isinstance(node, list):
            for value in node:
                walk(value, hits)

    checked = 0
    for name in ("report_ingest.triage", "report_ingest.label_review",
                 "report_ingest.log_reader", "report_ingest.lab_reader",
                 "report_ingest.narrative_reader",
                 "report_ingest.vision_labels"):
        mod = importlib.import_module(name)
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if (isinstance(obj, type) and issubclass(obj, BaseModel)
                    and obj is not BaseModel and obj.__module__ == name):
                hits = set()
                walk(strict_schema(obj), hits)
                assert not hits, f"{name}.{attr} still carries {sorted(hits)}"
                checked += 1
    assert checked >= 30, "the reader models were not found"


# -- metering ------------------------------------------------------------------

def test_tokens_are_read_from_the_openai_usage_shape():
    engine, _ = _engine(_Response(_Message("ok"),
                                  usage=_Usage(2000, 300, cached=1500)))
    reply = engine.complete([user(text_block("go"))], tools=[{
        "name": "t", "description": "d", "input_schema": {}}])
    assert reply.usage.input_tokens == 2000
    assert reply.usage.output_tokens == 300
    assert reply.usage.cache_read_tokens == 1500
    assert engine.meter.calls == 1


def test_tool_calls_come_back_with_their_arguments_parsed():
    message = _Message(None, [_Call("c1", "read_page", '{"page": 12}')])
    engine, _ = _engine(_Response(message, finish_reason="tool_calls"))
    reply = engine.complete([user(text_block("go"))], tools=[{
        "name": "read_page", "description": "d", "input_schema": {}}])
    assert reply.wants_tools
    assert reply.tool_calls[0].arguments == {"page": 12}
    assert reply.stop_reason == "tool_calls"
    assert reply.content[0]["type"] == "tool_use"


def test_unparseable_tool_arguments_become_an_empty_dict_not_a_crash():
    message = _Message(None, [_Call("c1", "read_page", "{not json")])
    engine, _ = _engine(_Response(message, finish_reason="tool_calls"))
    reply = engine.complete([user(text_block("go"))], tools=[{
        "name": "read_page", "description": "d", "input_schema": {}}])
    assert reply.tool_calls[0].arguments == {}


def test_a_refused_or_failed_call_is_an_error_not_a_blank_reply():
    engine, prompter = _engine(None)
    prompter.response = None
    prompter._client = _Client(None)
    with pytest.raises(RuntimeError, match="returned nothing"):
        engine.complete([user(text_block("go"))], tools=[{
            "name": "t", "description": "d", "input_schema": {}}])


# -- which path a call takes ----------------------------------------------------

def test_a_single_shot_call_goes_through_chat_to_keep_the_budget_guard():
    engine, prompter = _engine(_Response(_Message("ok")))
    engine.complete([user(text_block("go"))], system="be careful",
                    output_format=_Answer)
    assert prompter.chat_calls, "triage must keep the SDK's budget guard"
    assert prompter.chat_calls[0]["system"] == "be careful"
    assert prompter.chat_calls[0]["return_raw"] is True
    assert prompter.chat_calls[0]["response_format"]["type"] == "json_schema"
    assert prompter.client.chat.completions.last is None


def test_a_tool_loop_goes_through_the_raw_client():
    engine, prompter = _engine(_Response(_Message("ok")))
    engine.complete([user(text_block("go"))], tools=[{
        "name": "t", "description": "d", "input_schema": {}}])
    assert not prompter.chat_calls
    assert prompter.client.chat.completions.last is not None


def test_a_backend_with_no_openai_client_says_so_rather_than_failing_deep():
    engine, _ = _engine(_Response(_Message("ok")), with_client=False)
    with pytest.raises(RuntimeError, match="no OpenAI client"):
        engine.complete([user(text_block("go"))], tools=[{
            "name": "t", "description": "d", "input_schema": {}}])


def test_the_chat_path_can_be_turned_off_for_a_like_for_like_comparison():
    engine, prompter = _engine(_Response(_Message("ok")),
                               prefer_chat_for_single_calls=False)
    engine.complete([user(text_block("go"))], output_format=_Answer)
    assert not prompter.chat_calls
    assert prompter.client.chat.completions.last is not None


# -- a deployment that refuses a parameter --------------------------------------
#
# The first cluster run (2026-09-18) stopped on the triage call: the model
# behind the tier refused ``max_tokens`` and wanted ``max_completion_tokens``,
# prompter.chat() logged that and returned None, and the report FAILED. The
# engine now adapts once and remembers, as the app's Databricks bridge does.

_MAX_TOKENS_REFUSED = (
    "Error code: 400 - {'error': {'message': \"Unsupported parameter: "
    "'max_tokens' is not supported with this model. Use "
    "'max_completion_tokens' instead.\", 'type': 'invalid_request_error', "
    "'param': 'max_tokens', 'code': 'unsupported_parameter'}}")
_TEMPERATURE_REFUSED = (
    "Error code: 400 - {'error': {'message': \"Unsupported value: "
    "'temperature' does not support 0 with this model. Only the default (1) "
    "value is supported.\", 'type': 'invalid_request_error', "
    "'param': 'temperature', 'code': 'unsupported_value'}}")

_TOOL = [{"name": "t", "description": "d", "input_schema": {}}]


class _FussyCompletions(_Completions):
    """A reasoning-class deployment: refuses max_tokens and any temperature."""

    def __init__(self, response, refuse=("max_tokens", "temperature")):
        super().__init__(response)
        self.refuse = set(refuse)
        self.attempts = []

    def create(self, **kwargs):
        self.attempts.append(dict(kwargs))
        if "max_tokens" in self.refuse and "max_tokens" in kwargs:
            raise RuntimeError(_MAX_TOKENS_REFUSED)
        if "temperature" in self.refuse and "temperature" in kwargs:
            raise RuntimeError(_TEMPERATURE_REFUSED)
        self.last = kwargs
        return self.response


def _fussy_engine(refuse=("max_tokens", "temperature"), chat_returns=None):
    response = _Response(_Message("ok"))
    prompter = FakePrompter(response)
    prompter.client.chat.completions = _FussyCompletions(response, refuse)
    prompter.response = chat_returns            # what chat() hands back
    engine = PrompterEngine(prompter, meter=CostMeter())
    engine._sleep = lambda seconds: None        # never wait for real here
    return engine, prompter


def _attempts(prompter):
    return prompter.client.chat.completions.attempts


def test_a_refused_max_tokens_is_resent_as_max_completion_tokens():
    engine, prompter = _fussy_engine(refuse=("max_tokens",))
    reply = engine.complete([user(text_block("go"))], tools=_TOOL)
    attempts = _attempts(prompter)
    assert reply.text == "ok"
    assert len(attempts) == 2
    assert "max_tokens" in attempts[0]
    assert "max_completion_tokens" in attempts[1]
    assert "max_tokens" not in attempts[1]
    assert attempts[1]["max_completion_tokens"] == attempts[0]["max_tokens"]
    assert engine.token_key == "max_completion_tokens"
    assert engine.adaptations and "max_tokens" in engine.adaptations[0]


def test_the_lesson_is_kept_so_the_next_call_is_not_refused_first():
    engine, prompter = _fussy_engine(refuse=("max_tokens",))
    engine.complete([user(text_block("go"))], tools=_TOOL)
    engine.complete([user(text_block("again"))], tools=_TOOL)
    attempts = _attempts(prompter)
    assert len(attempts) == 3, "two calls, one refusal in all"
    assert "max_completion_tokens" in attempts[2]


def test_a_refused_temperature_is_dropped_and_stays_dropped():
    engine, prompter = _fussy_engine(refuse=("temperature",))
    engine.complete([user(text_block("go"))], tools=_TOOL)
    engine.complete([user(text_block("again"))], tools=_TOOL)
    attempts = _attempts(prompter)
    assert "temperature" in attempts[0]
    assert all("temperature" not in a for a in attempts[1:])
    assert engine.send_temperature is False


def test_both_refusals_cost_two_retries_in_all_not_two_per_call():
    engine, prompter = _fussy_engine()
    engine.complete([user(text_block("go"))], tools=_TOOL)
    engine.complete([user(text_block("again"))], tools=_TOOL)
    attempts = _attempts(prompter)
    # call 1: max_tokens refused, renamed; temperature refused, dropped.
    # call 2: sent in the accepted form first time.
    assert len(attempts) == 4
    assert "max_completion_tokens" in attempts[-1]
    assert "temperature" not in attempts[-1]
    assert len(engine.adaptations) == 2


def test_an_error_that_is_not_about_a_parameter_is_not_retried():
    engine, prompter = _fussy_engine(refuse=())

    def boom(**kwargs):
        raise RuntimeError("Error code: 500 - the server had an error")

    prompter.client.chat.completions.create = boom
    with pytest.raises(RuntimeError, match="server had an error"):
        engine.complete([user(text_block("go"))], tools=_TOOL)
    assert not engine.adaptations


_RATE_LIMITED = ("Error code: 429 - {'error': {'message': \"Your requests to "
                 "gpt-5.4 for funhouse-gpt-high in eastus2 have exceeded rate "
                 "limit.\", 'type': 'too_many_requests', 'param': None, "
                 "'code': 'rate_limit_exceeded'}}")


class RateLimitError(Exception):
    """Named like the SDK's, so the engine recognises it by name too."""


def test_a_rate_limited_call_waits_and_tries_again():
    """The first full cluster run lost 10 label reviews and 5 logs to 429s
    because nothing waited. Now the call waits, on the engine's ladder or
    the provider's retry-after, and the waits are recorded."""
    engine, prompter = _fussy_engine(refuse=())
    completions = prompter.client.chat.completions
    real = completions.create
    state = {"left": 2}

    def flaky(**kw):
        if state["left"]:
            state["left"] -= 1
            raise RateLimitError(_RATE_LIMITED)
        return real(**kw)

    completions.create = flaky
    slept = []
    engine._sleep = slept.append
    reply = engine.complete([user(text_block("go"))], tools=_TOOL)
    assert reply.text == "ok"
    assert slept == [15.0, 30.0], "the ladder, in order"
    assert sum("rate limited" in a for a in engine.adaptations) == 2


def test_a_rate_limit_that_never_lifts_is_raised_after_the_ladder():
    engine, prompter = _fussy_engine(refuse=())

    def always(**kw):
        raise RateLimitError(_RATE_LIMITED)

    prompter.client.chat.completions.create = always
    slept = []
    engine._sleep = slept.append
    with pytest.raises(RateLimitError):
        engine.complete([user(text_block("go"))], tools=_TOOL)
    assert slept == list(PrompterEngine.RATE_LIMIT_WAITS)


def test_the_providers_retry_after_beats_the_ladder():
    engine, prompter = _fussy_engine(refuse=())
    completions = prompter.client.chat.completions
    real = completions.create
    state = {"left": 1}

    class _Resp:
        headers = {"retry-after": "7"}

    def flaky(**kw):
        if state["left"]:
            state["left"] -= 1
            exc = RateLimitError(_RATE_LIMITED)
            exc.response = _Resp()
            raise exc
        return real(**kw)

    completions.create = flaky
    slept = []
    engine._sleep = slept.append
    engine.complete([user(text_block("go"))], tools=_TOOL)
    assert slept == [7.0]


def test_when_chat_swallows_the_refusal_the_call_falls_back_to_the_raw_client():
    engine, prompter = _fussy_engine(refuse=("max_tokens",), chat_returns=None)
    prompter._last_chat_error = RuntimeError(_MAX_TOKENS_REFUSED)
    reply = engine.complete([user(text_block("go"))], system="s",
                            output_format=_Answer)
    assert prompter.chat_calls, "chat was tried first, to keep the guard"
    assert reply.text == "ok", "and the raw client served the call"
    assert engine.token_key == "max_completion_tokens"
    assert any("chat() returned nothing" in a for a in engine.adaptations)


def test_after_chat_fails_once_the_run_stops_offering_it():
    engine, prompter = _fussy_engine(refuse=("max_tokens",), chat_returns=None)
    engine.complete([user(text_block("go"))], output_format=_Answer)
    engine.complete([user(text_block("again"))], output_format=_Answer)
    assert len(prompter.chat_calls) == 1
    assert engine.prefer_chat is False


def test_the_learned_token_key_is_what_chat_is_asked_for_too():
    engine, prompter = _fussy_engine(refuse=("max_tokens",))
    engine.complete([user(text_block("go"))], tools=_TOOL)     # learns
    prompter.response = _Response(_Message("ok"))
    engine.complete([user(text_block("simple"))], output_format=_Answer)
    assert "max_completion_tokens" in prompter.chat_calls[-1]
    assert "max_tokens" not in prompter.chat_calls[-1]


def test_when_chat_swallows_a_refusal_and_there_is_no_client_the_reason_is_reported():
    engine, prompter = _engine(None, with_client=False)
    prompter._last_chat_error = RuntimeError("quota exhausted")
    with pytest.raises(RuntimeError, match="quota exhausted"):
        engine.complete([user(text_block("go"))], output_format=_Answer)


# -- dollars, at the owner's own rates ----------------------------------------

def test_the_funhouse_deployments_are_priced_by_what_answered():
    """The owner's rates, read off the Funhouse budget page on 2026-09-18."""
    assert PROMPTER_PRICES["gpt-5.4-2026-03-05"] == (2.50, 15.00)
    assert PROMPTER_PRICES["gpt-5.1-2025-11-13"] == (1.25, 10.00)
    assert PROMPTER_PRICES["gpt-4.1-mini-2025-04-14"] == (0.40, 1.60)
    assert PROMPTER_PRICES["text-embedding-ada-002"] == (0.10, 0.00)
    # A TIER is an alias with no price: pricing by the tier would price a
    # name that means a different model next month.
    for tier in PROMPTER_MODELS:
        assert price_for(tier) is None


def test_a_gpt_54_call_of_a_thousand_in_and_a_hundred_out_costs_four_mills():
    usage = Usage(input_tokens=1000, output_tokens=100)
    # 1000 * $2.50/M + 100 * $15.00/M = $0.0025 + $0.0015
    assert usage.dollars("gpt-5.4-2026-03-05") == pytest.approx(0.004)


def test_an_unknown_deployment_costs_nothing_rather_than_being_guessed_at():
    usage = Usage(input_tokens=10**6, output_tokens=10**6)
    assert usage.dollars("gpt-9-does-not-exist") == 0.0
    assert usage.dollars("") == 0.0


def test_the_meter_prices_a_call_by_the_deployment_not_by_the_tier():
    meter = CostMeter()
    meter.add("funhouse-gpt-high",
              Usage(input_tokens=1000, output_tokens=100), 1.0,
              served_by="gpt-5.4-2026-03-05")

    assert meter.dollars == pytest.approx(0.004)
    assert meter.unpriced_calls == 0
    assert meter.priced is True
    # The scorecard still records the TIER that was asked for, with what
    # actually priced it beside.
    assert set(meter.by_model) == {"funhouse-gpt-high"}
    assert meter.by_model["funhouse-gpt-high"]["priced_as"] == [
        "gpt-5.4-2026-03-05"]


def test_a_tier_whose_deployment_is_unpriced_still_reports_tokens_only():
    meter = CostMeter()
    meter.add("funhouse-gpt-high", Usage(input_tokens=1000, output_tokens=100),
              1.0, served_by="gpt-7-preview")

    assert meter.dollars == 0.0
    assert meter.unpriced_calls == 1
    assert meter.priced is False
    assert meter.to_dict()["input_tokens"] == 1000
    assert "tokens, not dollars" in meter.summary()


def test_a_run_that_mixes_priced_and_unpriced_calls_says_so():
    meter = CostMeter()
    meter.add("funhouse-gpt-high", Usage(input_tokens=1000, output_tokens=100),
              1.0, served_by="gpt-5.4-2026-03-05")
    meter.add("funhouse-gpt-low", Usage(input_tokens=1000), 1.0,
              served_by="gpt-nobody-has-priced")

    assert meter.calls == 2
    assert meter.unpriced_calls == 1
    assert meter.priced is False, (
        "a dollar total that covers half the calls is a floor, not a total")
    assert meter.dollars == pytest.approx(0.004)


def test_a_reply_meters_itself_by_the_deployment_that_served_it():
    engine, _ = _engine(_Response(_Message("hi"),
                                  usage=_Usage(1000, 100),
                                  model="gpt-4.1-mini-2025-04-14"))
    engine.complete([user(text_block("go"))], tools=[{
        "name": "t", "description": "d", "input_schema": {}}])

    # 1000 * $0.40/M + 100 * $1.60/M
    assert engine.meter.dollars == pytest.approx(0.0004 + 0.00016)
    assert engine.meter.by_model["funhouse-gpt-medium"]["priced_as"] == [
        "gpt-4.1-mini-2025-04-14"]


def test_the_claude_list_prices_are_untouched():
    assert Usage(input_tokens=10**6).dollars("claude-opus-5") == 5.00
    assert price_for("claude-sonnet-5") == (2.00, 10.00)
