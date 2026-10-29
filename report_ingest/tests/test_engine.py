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


# -- every request in the cost record (BACKLOG RI-21) -------------------------

class _Refusing(_Messages):
    def create(self, **kwargs):
        raise RuntimeError("Error code: 413 - request too large")


def test_a_claude_request_that_fails_is_recorded_then_raised():
    client = _Client(None)
    client.messages = _Refusing(None)
    engine = ClaudeEngine("claude-opus-5", client=client, meter=CostMeter())
    with pytest.raises(RuntimeError, match="too large"):
        engine.complete([user(text_block("go"))])
    meter = engine.meter
    assert (meter.calls, meter.failed) == (0, 1)
    assert meter.failed_by_kind == {"too_large": 1}
    (row,) = meter.to_dict()["per_call"]
    assert row["outcome"] == "error" and row["error_kind"] == "too_large"


def test_a_claude_reply_records_its_stop_reason():
    engine = _engine(_Response([_Block(type="text", text="hi")],
                               _Usage(10, 5), stop_reason="max_tokens"))
    engine.complete([user(text_block("go"))])
    (row,) = engine.meter.to_dict()["per_call"]
    assert row["outcome"] == "reply" and row["stop"] == "max_tokens"
    assert (row["input_tokens"], row["output_tokens"]) == (10, 5)


@pytest.mark.parametrize("text, kind", [
    ("Error code: 429 - too many requests", "rate_limit"),
    ("Error code: 413 - payload too large", "too_large"),
    ("Request timed out.", "timeout"),
    ("Unsupported parameter: 'max_tokens'", "parameter"),
    ("Prompter returned nothing", "no_reply"),
    ("Error code: 400 - prompt is too long: 215000 tokens > 200000 maximum",
     "too_large"),
    ("This model's maximum context length is 128000 tokens", "too_large"),
    ("Error code: 500 - the server had an error", "server_error"),
    ("Error code: 400 - the image could not be read", "other"),
])
def test_an_error_is_sorted_by_what_it_says(text, kind):
    from report_ingest.engine import FAILURE_KINDS, error_kind

    assert error_kind(RuntimeError(text)) == kind
    assert kind in FAILURE_KINDS


def test_a_failed_request_adds_no_reply_and_keeps_the_usage_it_carried():
    class _WithBody(Exception):
        body = {"usage": {"input_tokens": 700, "output_tokens": 0}}

    meter = CostMeter()
    meter.fail("m", _WithBody("cut off"), kind="other", seconds=3.0)
    meter.add("m", Usage(input_tokens=10, output_tokens=2), 1.0)
    out = meter.to_dict()
    assert (out["calls"], out["failed"], out["retried"]) == (1, 1, 0)
    assert out["input_tokens"] == 710, "what the error said it spent counts"
    assert out["seconds"] == 4.0
    assert out["by_model"]["m"]["failed"] == 1
    assert [r["call"] for r in out["per_call"]] == [1, 2]


def test_an_old_reader_of_the_cost_block_still_finds_its_keys():
    out = CostMeter().to_dict()
    for key in ("calls", "input_tokens", "output_tokens", "cache_read_tokens",
                "cache_write_tokens", "seconds", "dollars", "unpriced_calls",
                "by_model"):
        assert key in out


# -- recording never takes the place of the provider's error (RI-21 review) ---

class _Unreadable(RuntimeError):
    """An error whose text cannot be read: ``str()`` itself raises."""

    def __str__(self):
        raise ValueError("no text here")


class _BadBody(RuntimeError):
    body = {"usage": {"prompt_tokens": "n/a", "completion_tokens": None}}


class _Raising(_Messages):
    def __init__(self, error):
        super().__init__(None)
        self.error = error

    def create(self, **kwargs):
        raise self.error


def _failing_claude(error):
    client = _Client(None)
    client.messages = _Raising(error)
    return ClaudeEngine("claude-opus-5", client=client, meter=CostMeter())


@pytest.mark.parametrize("error", [_BadBody("Error code: 400 - bad"),
                                   _Unreadable()])
def test_an_error_that_will_not_read_still_comes_out_unchanged(error):
    engine = _failing_claude(error)
    with pytest.raises(type(error)) as raised:
        engine.complete([user(text_block("go"))])
    assert raised.value is error
    (row,) = engine.meter.to_dict()["per_call"]
    assert row["outcome"] == "error"
    assert engine.meter.failed == 1
    assert engine.meter.input_tokens == 0, "a count that is not one is none"


def test_an_unreadable_error_is_named_by_its_type():
    meter = CostMeter()
    meter.fail("m", _Unreadable())
    (row,) = meter.to_dict()["per_call"]
    assert row["error"] == "_Unreadable: (its message could not be read)"
    assert row["error_kind"] == "other"


def test_a_failure_while_recording_is_logged_and_the_error_still_raised(
        monkeypatch, caplog):
    engine = _failing_claude(RuntimeError("Error code: 500 - the provider"))

    def broken(*args, **kwargs):
        raise KeyError("the record is broken")

    monkeypatch.setattr(engine.meter, "_entry", broken)
    with pytest.raises(RuntimeError, match="the provider"):
        engine.complete([user(text_block("go"))])
    assert "could not record a failed request" in caplog.text


def test_a_failure_while_recording_a_reply_still_returns_the_reply(
        monkeypatch):
    engine = _engine(_Response([_Block(type="text", text="hi")],
                               _Usage(10, 5)))

    def broken(*args, **kwargs):
        raise KeyError("the record is broken")

    monkeypatch.setattr(engine.meter, "_entry", broken)
    reply = engine.complete([user(text_block("go"))])
    assert reply.text == "hi" and engine.meter.calls == 1


def test_counts_are_read_tolerantly():
    from report_ingest.engine import _count

    assert [_count(v) for v in (12, 12.0, "12", " 7 ", "n/a", None, -3,
                                True, float("nan"), [1])] == \
        [12, 12, 12, 7, None, None, None, None, None, None]


# -- reasoning that was not reported is not a reported 0 (RI-21 review) -------

def test_claude_reports_no_reasoning_split_so_none_is_counted():
    engine = _engine(_Response([_Block(type="text", text="hi")],
                               _Usage(10, 5)))
    reply = engine.complete([user(text_block("go"))])
    assert reply.usage.reasoning_tokens is None
    out = engine.meter.to_dict()
    assert out["reasoning_reported"] == 0 and out["reasoning_tokens"] == 0
    assert "reasoning_tokens" not in out["per_call"][0]


def test_the_meter_counts_the_replies_that_reported_reasoning():
    meter = CostMeter()
    meter.add("m", Usage(output_tokens=50, reasoning_tokens=30), 1.0)
    meter.add("m", Usage(output_tokens=50, reasoning_tokens=0), 1.0)
    meter.add("m", Usage(output_tokens=50), 1.0)
    out = meter.to_dict()
    assert (out["reasoning_tokens"], out["reasoning_reported"]) == (30, 2)
    assert [r.get("reasoning_tokens") for r in out["per_call"]] == \
        [30, 0, None]


# -- the per-request rows, compact (RI-21 review) -----------------------------

def test_the_model_and_deployment_are_written_once_not_on_every_row():
    meter = CostMeter()
    for _ in range(3):
        meter.add("tier-high", Usage(input_tokens=5), 1.0,
                  served_by="deploy-a")
    meter.add("tier-low", Usage(input_tokens=5), 1.0, served_by="deploy-b")
    meter.fail("tier-high", RuntimeError("Error code: 500"), attempt=1,
               retried=True)
    meter.add("tier-high", Usage(input_tokens=5), 1.0, served_by="deploy-a",
              attempt=2)
    out = meter.to_dict()
    assert out["per_call_defaults"] == {"model": "tier-high",
                                        "served_by": "deploy-a"}
    rows = out["per_call"]
    assert all("model" not in r and "served_by" not in r
               for r in rows[:3])
    assert (rows[3]["model"], rows[3]["served_by"]) == ("tier-low",
                                                        "deploy-b")
    assert "served_by" not in rows[4], "nothing served a failed request"
    assert "attempt" not in rows[4] and rows[5]["attempt"] == 2
    assert rows[5]["call"] == rows[4]["call"] == 5


# -- when each request went out, and a failed one's size (BACKLOG RI-126) -----
#
# Every number and time below is made up for the test.

#: A made-up moment: 2027-01-15T08:00:00Z.
_T0 = 1_800_000_000.0


class _FakeClock:
    """A clock that moves only when told; :meth:`sleep` moves it too, and
    remembers each wait, so nothing here waits for real."""

    def __init__(self, at: float = _T0):
        self.at = float(at)
        self.slept = []

    def time(self):
        return self.at

    def sleep(self, seconds):
        self.slept.append(round(float(seconds), 3))
        self.at += float(seconds)

    def tick(self, seconds):
        self.at += float(seconds)


class _Timed(_Messages):
    """A client whose requests take ``takes`` seconds on the fake clock,
    each answering with ``response`` or raising what ``errors`` holds."""

    def __init__(self, response, clock, takes, errors=()):
        super().__init__(response)
        self.clock = clock
        self.takes = takes
        self.errors = list(errors)

    def create(self, **kwargs):
        self.last = kwargs
        self.clock.tick(self.takes)
        if self.errors:
            raise self.errors.pop(0)
        return self.response

    parse = create


def _timed_claude(clock, takes, errors=(), pacer=None):
    client = _Client(None)
    client.messages = _Timed(
        _Response([_Block(type="text", text="ok")], _Usage(40, 4)), clock,
        takes, errors)
    engine = ClaudeEngine("claude-opus-5", client=client,
                          meter=CostMeter(clock=clock.time), pacer=pacer)
    engine._clock = clock.time
    return engine


def _png(width, height):
    """A PNG header with a made-up size: all the size estimate reads."""
    return (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0d" + b"IHDR"
            + int(width).to_bytes(4, "big") + int(height).to_bytes(4, "big")
            + b"\x00" * 4000)


class TestEveryRowSaysWhen:

    def test_a_reply_row_carries_its_start_and_its_duration(self):
        clock = _FakeClock()
        engine = _timed_claude(clock, takes=4.5)
        engine.complete([user(text_block("go"))])
        clock.tick(10.0)
        engine.complete([user(text_block("again"))])
        first, second = engine.meter.to_dict()["per_call"]
        assert first["started"] == "2027-01-15T08:00:00.000Z"
        assert first["duration"] == 4.5 and first["seconds"] == 4.5
        assert second["started"] == "2027-01-15T08:00:14.500Z"
        assert second["duration"] == 4.5

    def test_a_failed_row_carries_its_start_and_its_duration(self):
        clock = _FakeClock()
        engine = _timed_claude(clock, takes=2.25, errors=[
            RuntimeError("Error code: 500 - made up")])
        with pytest.raises(RuntimeError):
            engine.complete([user(text_block("go"))])
        (row,) = engine.meter.to_dict()["per_call"]
        assert row["outcome"] == "error"
        assert row["started"] == "2027-01-15T08:00:00.000Z"
        assert row["duration"] == 2.25

    def test_a_row_recorded_without_times_is_dated_by_the_meter(self):
        clock = _FakeClock()
        meter = CostMeter(clock=clock.time)
        meter.add("m", Usage(input_tokens=5), 2.0)
        meter.fail("m", RuntimeError("Error code: 500"), retried=True)
        reply, error = meter.to_dict()["per_call"]
        assert reply["started"] == "2027-01-15T07:59:58.000Z"
        assert reply["duration"] == 2.0
        assert error["started"] == "2027-01-15T08:00:00.000Z"
        assert error["duration"] == 0.0


class TestAFailedRequestsSize:

    def test_a_failed_request_carries_its_estimated_input_tokens(self):
        from report_ingest.engine import estimate_request

        clock = _FakeClock()
        engine = _timed_claude(clock, takes=1.0, errors=[
            RuntimeError("Error code: 429 - too many requests")])
        messages = [user(text_block("w" * 7000))]
        with pytest.raises(RuntimeError):
            engine.complete(messages, system="s" * 350)
        (row,) = engine.meter.to_dict()["per_call"]
        assert row["input_tokens"] == estimate_request(
            messages, system="s" * 350) == 2100
        assert row["input_tokens_estimated"] is True
        meter = engine.meter
        assert meter.input_tokens == 0, "an estimate is not in the totals"
        assert meter.by_model["claude-opus-5"]["input_tokens"] == 0

    def test_a_count_the_error_carried_is_kept_and_not_marked(self):
        class _WithBody(Exception):
            body = {"usage": {"input_tokens": 700, "output_tokens": 0}}

        meter = CostMeter()
        meter.fail("m", _WithBody("cut off"), estimated_in=9999)
        (row,) = meter.to_dict()["per_call"]
        assert row["input_tokens"] == 700
        assert "input_tokens_estimated" not in row

    def test_a_reply_is_not_estimated_without_a_pacer(self, monkeypatch):
        import report_ingest.engine as engine_module

        def refuse(*args, **kwargs):
            raise AssertionError("a reply with no pacer was estimated")

        monkeypatch.setattr(engine_module, "estimate_request", refuse)
        engine = _timed_claude(_FakeClock(), takes=1.0)
        assert engine.complete([user(text_block("go"))]).text == "ok"
        (row,) = engine.meter.to_dict()["per_call"]
        assert "input_tokens_estimated" not in row
        assert "paced" not in row
        assert "paced_seconds" not in engine.meter.to_dict()

    def test_a_tool_s_picture_is_sized_by_its_pixels_not_its_bytes(self):
        from report_ingest.engine import estimate_request

        picture = _png(1500, 1000)              # 2,000 tokens at 750 px
        sized = estimate_request([user(tool_result_block(
            "t1", [image_block(picture)]))])
        assert sized == 2000

    def test_the_tools_and_the_answer_schema_count(self):
        from report_ingest.engine import estimate_request, schema_tokens_est

        tools = [{"name": "zoom", "description": "d" * 1400}]
        bare = estimate_request([user(text_block("go"))])
        with_tools = estimate_request([user(text_block("go"))], tools=tools)
        assert with_tools > bare + 400            # 1,400 characters or so
        from pydantic import BaseModel

        class _Answer(BaseModel):
            soil: str

        answered = estimate_request([user(text_block("go"))],
                                    output_format=_Answer)
        assert answered == bare + schema_tokens_est(_Answer)

    def test_something_that_cannot_be_sized_is_none(self):
        from report_ingest.engine import estimate_request

        assert estimate_request([{"role": "user"}], tools=object()) is None


# -- pacing to a per-minute token limit (BACKLOG RI-127) ----------------------
#
# Every budget, size and time below is made up for the test.

def _pacer(clock, tokens_per_minute=100_000, **kw):
    from report_ingest.engine import RequestPacer

    return RequestPacer(tokens_per_minute, clock=clock.time,
                        sleep=clock.sleep, **kw)


class TestTheRequestPacer:

    def test_after_a_large_request_the_next_waits_out_the_minute(self):
        clock = _FakeClock()
        pacer = _pacer(clock)                    # large: over 50,000
        pacer.sent("model-a", 60_000)
        clock.tick(15.0)
        assert pacer.wait("model-a") == 45.0
        assert clock.slept == [45.0]
        assert (pacer.held, pacer.held_seconds) == (1, 45.0)
        assert pacer.wait("model-a") == 0.0, "the window has passed"

    def test_a_large_request_holds_its_own_model_only(self):
        clock = _FakeClock()
        pacer = _pacer(clock)
        pacer.sent("model-a", 60_000)
        assert pacer.wait("model-b") == 0.0
        assert clock.slept == []

    def test_a_request_at_or_under_the_fraction_holds_nothing(self):
        clock = _FakeClock()
        pacer = _pacer(clock, large_fraction=0.4)
        assert pacer.large_tokens == 40_000
        pacer.sent("model-a", 40_000)
        pacer.sent("model-a", None)              # could not be sized
        assert pacer.wait("model-a") == 0.0

    def test_a_rate_limit_backs_off_a_whole_window_unless_told(self):
        clock = _FakeClock()
        pacer = _pacer(clock)
        assert pacer.backoff(None, 15.0) == 60.0
        assert pacer.backoff(None, 120.0) == 120.0
        assert pacer.backoff(7.0, 15.0) == 7.0, "a retry-after is honoured"
        pacer.hold("model-a", pacer.backoff(None))
        clock.tick(20.0)
        assert pacer.wait("model-a") == 40.0

    @pytest.mark.parametrize("given", [
        {"tokens_per_minute": 0}, {"tokens_per_minute": -5},
        {"tokens_per_minute": True}, {"large_fraction": 0.0},
        {"large_fraction": 1.5}, {"window": 0.0}])
    def test_settings_that_mean_nothing_are_refused(self, given):
        from report_ingest.engine import RequestPacer

        kwargs = {"tokens_per_minute": 100_000, **given}
        with pytest.raises(ValueError):
            RequestPacer(kwargs.pop("tokens_per_minute"), **kwargs)

    def test_the_process_keeps_one_pacer_per_setting(self):
        from report_ingest.engine import shared_pacer

        a = shared_pacer(123_457)
        assert shared_pacer(123_457) is a
        assert shared_pacer(123_457, large_fraction=0.25) is not a
        assert shared_pacer(765_431) is not a

    def test_pace_engines_looks_through_wrappers_and_keeps_a_pacer(self):
        from report_ingest.engine import pace_engines

        clock = _FakeClock()
        pacer, own = _pacer(clock), _pacer(clock)
        bare = _timed_claude(clock, 1.0)
        mine = _timed_claude(clock, 1.0, pacer=own)

        class _Wrapper:
            def __init__(self, inner):
                self._inner = inner

        put = pace_engines([_Wrapper(_Wrapper(bare)), mine, None, object(),
                            bare], pacer)
        assert put == [bare]
        assert bare.pacer is pacer and mine.pacer is own
        assert pace_engines([bare], None) == []


class TestAPacedClaudeEngine:

    def test_after_a_large_request_the_next_is_held_not_refused(self):
        clock = _FakeClock()
        engine = _timed_claude(clock, takes=20.0, pacer=_pacer(clock))
        engine.complete([user(text_block("w" * 210_000))])   # 60,000 in
        clock.tick(5.0)
        engine.complete([user(text_block("small"))])
        # Its reply came back at 08:00:20, so the next waits to 08:01:20.
        assert clock.slept == [55.0], "the minute counts from its reply"
        first, second = engine.meter.to_dict()["per_call"]
        assert "paced" not in first
        assert second["paced"] == 55.0
        assert second["started"] == "2027-01-15T08:01:20.000Z"
        assert second["duration"] == 20.0
        assert second["seconds"] == 75.0, "the hold is in the call's time"
        assert engine.meter.to_dict()["paced_seconds"] == 55.0

    def test_a_rate_limit_holds_the_next_request_a_whole_window(self):
        clock = _FakeClock()
        engine = _timed_claude(clock, takes=1.0, pacer=_pacer(clock),
                               errors=[RuntimeError(
                                   "Error code: 429 - too many requests")])
        with pytest.raises(RuntimeError):
            engine.complete([user(text_block("small"))])
        engine.complete([user(text_block("small"))])
        assert clock.slept == [60.0]

    def test_small_requests_are_never_held(self):
        clock = _FakeClock()
        engine = _timed_claude(clock, takes=1.0, pacer=_pacer(clock))
        for _ in range(3):
            engine.complete([user(text_block("small"))])
        assert clock.slept == []
        assert "paced_seconds" not in engine.meter.to_dict()
