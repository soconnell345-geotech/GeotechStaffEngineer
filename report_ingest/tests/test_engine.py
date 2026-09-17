"""The engine layer: neutral blocks, the cost meter, and the translation."""

from __future__ import annotations

import base64

import pytest

from report_ingest.engine import (
    CACHE_READ_RATE, CACHE_WRITE_RATE, ClaudeEngine, CostMeter, MODEL_PRICES,
    Usage, image_block, opaque_block, text_block, tool_result_block,
    tool_use_block, user,
)


class _Block:
    """A provider content block, shaped like the SDK's."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_dump(self, exclude_none=False):
        return dict(self.__dict__)


class _Usage:
    def __init__(self, i=0, o=0, r=0, w=0):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_read_input_tokens = r
        self.cache_creation_input_tokens = w


class _Response:
    def __init__(self, content, usage, stop_reason="end_turn", parsed=None):
        self.content = content
        self.usage = usage
        self.stop_reason = stop_reason
        self.parsed_output = parsed


class _Messages:
    def __init__(self, response):
        self.response = response
        self.last = None

    def create(self, **kwargs):
        self.last = kwargs
        return self.response

    def parse(self, **kwargs):
        self.last = kwargs
        return self.response


class _Client:
    def __init__(self, response):
        self.messages = _Messages(response)


def _engine(response, model="claude-opus-5"):
    return ClaudeEngine(model, client=_Client(response), meter=CostMeter())


def test_usage_dollars_uses_the_published_prices():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    price_in, price_out = MODEL_PRICES["claude-opus-5"]
    assert usage.dollars("claude-opus-5") == pytest.approx(
        price_in + price_out)


def test_cache_tokens_are_cheaper_than_plain_input():
    read = Usage(cache_read_tokens=1_000_000).dollars("claude-opus-5")
    write = Usage(cache_write_tokens=1_000_000).dollars("claude-opus-5")
    plain = Usage(input_tokens=1_000_000).dollars("claude-opus-5")
    assert read == pytest.approx(plain * CACHE_READ_RATE)
    assert write == pytest.approx(plain * CACHE_WRITE_RATE)


def test_an_unpriced_model_costs_nothing_rather_than_guessing():
    assert Usage(input_tokens=10**6).dollars("some-cluster-model") == 0.0


def test_the_meter_totals_per_model_and_overall():
    meter = CostMeter()
    meter.add("claude-opus-5", Usage(input_tokens=1000, output_tokens=200), 1.5)
    meter.add("claude-sonnet-5", Usage(input_tokens=500), 0.5)
    assert meter.calls == 2
    assert meter.input_tokens == 1500
    assert meter.seconds == pytest.approx(2.0)
    assert set(meter.by_model) == {"claude-opus-5", "claude-sonnet-5"}
    assert meter.by_model["claude-opus-5"]["calls"] == 1
    assert meter.dollars > meter.by_model["claude-sonnet-5"]["dollars"]
    assert meter.to_dict()["calls"] == 2


def test_blocks_translate_to_the_provider_shape():
    out = ClaudeEngine._to_provider([user(
        text_block("hello"),
        image_block(b"\x89PNG-ish"),
        tool_result_block("t1", "the answer"),
        tool_result_block("t2", [text_block("see this"),
                                 image_block(b"pixels")], is_error=True),
        opaque_block({"type": "thinking", "signature": "abc"}),
    )])
    blocks = out[0]["content"]
    assert blocks[0] == {"type": "text", "text": "hello"}
    assert blocks[1]["source"]["media_type"] == "image/png"
    assert base64.standard_b64decode(
        blocks[1]["source"]["data"]) == b"\x89PNG-ish"
    assert blocks[2] == {"type": "tool_result", "tool_use_id": "t1",
                         "content": "the answer"}
    assert blocks[3]["is_error"] is True
    assert blocks[3]["content"][1]["type"] == "image"
    # An opaque block goes through exactly as it arrived.
    assert blocks[4] == {"type": "thinking", "signature": "abc"}


def test_an_unknown_block_type_is_refused_not_guessed():
    with pytest.raises(ValueError, match="unknown block type"):
        ClaudeEngine._to_provider([user({"type": "video", "data": b""})])


def test_thinking_comes_back_as_an_opaque_block_to_echo_unchanged():
    thinking = _Block(type="thinking", thinking="", signature="sig-1")
    engine = _engine(_Response([thinking, _Block(type="text", text="hi")],
                               _Usage(i=10, o=2)))
    reply = engine.complete([user(text_block("go"))])
    assert reply.text == "hi"
    assert reply.content[0]["type"] == "opaque"
    assert reply.content[0]["data"]["signature"] == "sig-1"
    # And it round-trips back to the provider untouched.
    resent = ClaudeEngine._to_provider(
        [{"role": "assistant", "content": reply.content}])
    assert resent[0]["content"][0]["signature"] == "sig-1"


def test_tool_calls_come_back_parsed_and_metered():
    engine = _engine(_Response(
        [_Block(type="tool_use", id="c1", name="read_page", input={"page": 4})],
        _Usage(i=2000, o=50, r=1200), stop_reason="tool_use"))
    reply = engine.complete([user(text_block("go"))], tools=[{"name": "x"}])
    assert reply.wants_tools
    assert reply.tool_calls[0].name == "read_page"
    assert reply.tool_calls[0].arguments == {"page": 4}
    assert reply.stop_reason == "tool_use"
    assert engine.meter.input_tokens == 2000
    assert engine.meter.cache_read_tokens == 1200


def test_images_attach_to_the_last_user_message():
    engine = _engine(_Response([_Block(type="text", text="ok")], _Usage()))
    engine.complete([user(text_block("look"))], images=[b"png-1", b"png-2"])
    sent = engine._client.messages.last["messages"][-1]["content"]
    assert [b["type"] for b in sent] == ["text", "image", "image"]


def test_a_structured_request_goes_through_parse_not_create():
    engine = _engine(_Response([_Block(type="text", text="{}")], _Usage(),
                               parsed={"document_type": "other"}))
    reply = engine.complete([user(text_block("go"))], output_format=dict)
    assert reply.parsed == {"document_type": "other"}
    sent = engine._client.messages.last
    assert sent["output_format"] is dict
    # parse() takes no top-level cache_control; it rides in the body.
    assert "cache_control" not in sent
    assert sent["extra_body"]["cache_control"] == {"type": "ephemeral"}


def test_the_system_prompt_and_tools_are_passed_through():
    engine = _engine(_Response([_Block(type="text", text="ok")], _Usage()))
    engine.complete([user(text_block("go"))], system="be careful",
                    tools=[{"name": "read_page"}], max_tokens=4242)
    sent = engine._client.messages.last
    assert sent["system"] == "be careful"
    assert sent["tools"][0]["name"] == "read_page"
    assert sent["max_tokens"] == 4242
    assert sent["model"] == "claude-opus-5"


def test_the_prefix_is_cached_by_default_and_can_be_turned_off():
    engine = _engine(_Response([_Block(type="text", text="ok")], _Usage()))
    engine.complete([user(text_block("go"))])
    assert engine._client.messages.last["cache_control"] == {"type": "ephemeral"}
    plain = ClaudeEngine("claude-opus-5", client=_Client(
        _Response([_Block(type="text", text="ok")], _Usage())),
        meter=CostMeter(), auto_cache=False)
    plain.complete([user(text_block("go"))])
    assert "cache_control" not in plain._client.messages.last


def test_tool_use_block_copies_its_arguments():
    args = {"page": 1}
    block = tool_use_block("c1", "read_page", args)
    args["page"] = 99
    assert block["input"] == {"page": 1}
