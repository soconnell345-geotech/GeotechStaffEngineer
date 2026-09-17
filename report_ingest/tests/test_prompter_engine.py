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
    CostMeter, PROMPTER_MODELS, PrompterEngine, Usage, image_block,
    strict_schema, text_block, tool_result_block, tool_use_block, user,
)


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
