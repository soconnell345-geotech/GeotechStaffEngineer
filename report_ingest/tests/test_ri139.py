"""BACKLOG RI-139: the pacing follow-ups to RI-127.

(a) a pacer's hold before a request the route then refuses to send is
recorded; (b) ``paced_seconds`` reaches the attempt keys, results.json and
RESULTS, only where something was held; (c) a call's clock, size, attempt
and number are the call's own when calls run on several threads at once;
(d) a ``retry-after: 0`` is no wait, not the pacer's minute; (e) a rate
limit is known by its words in the message or the body, whatever the
status.

Every model name, size, count, time and message here is made up.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from report_ingest import cluster_scoring as cs
from report_ingest.engine import (
    CostMeter, PrompterEngine, RequestPacer, error_kind, estimate_request,
    is_rate_limit, text_block, user,
)

TOOL = [{"name": "look", "description": "Look again.",
         "input_schema": {"type": "object", "properties": {}}}]

#: 2027-01-15T08:00:00Z, a made-up moment far from any real run.
T0 = 1_800_000_000.0


class _Clock:
    """A clock that moves only when told; a sleep moves it and is kept."""

    def __init__(self, at=T0):
        self.at = at
        self.slept = []

    def time(self):
        return self.at

    def sleep(self, seconds):
        self.slept.append(float(seconds))
        self.at += float(seconds)


def _pacer(clock, tokens_per_minute=100_000):
    return RequestPacer(tokens_per_minute, clock=clock.time,
                        sleep=clock.sleep)


# -- a raw OpenAI-shaped client, as small as the engine needs ----------------

def _response(text, prompt=10, completion=2, model="made-up-deploy"):
    usage = SimpleNamespace(prompt_tokens=prompt,
                            completion_tokens=completion,
                            prompt_tokens_details=None,
                            completion_tokens_details=None)
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=usage, model=model)


class _Prompter:
    """Stands in for ``fh_prompter``: ``create`` answers every raw-client
    request; ``chat`` (single calls) answers with ``chat_reply``."""

    def __init__(self, create, with_client=True, chat_reply=None):
        self._client = (SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)))
            if with_client else None)
        self.chat_reply = chat_reply

    def get_openai_instance(self):
        return self._client

    def chat(self, **kwargs):
        return self.chat_reply


class RateLimitError(Exception):
    """Named like the SDK's."""


def _limited(text, retry_after=None, body=None, cls=RateLimitError):
    exc = cls(text)
    if retry_after is not None:
        exc.response = SimpleNamespace(headers={"retry-after": retry_after})
    if body is not None:
        exc.body = body
    return exc


def _flaky(errors, reply, clock=None, takes=0.0):
    """A ``create`` that raises ``errors`` in turn, then answers."""
    left = list(errors)

    def create(**kwargs):
        if clock is not None:
            clock.at += takes
        if left:
            raise left.pop(0)
        return reply

    return create


def _prompter_engine(create, clock, pacer=None, **kw):
    engine = PrompterEngine(_Prompter(create, **kw), meter=CostMeter(),
                            pacer=pacer)
    engine._clock = clock.time
    engine._sleep = clock.sleep
    return engine


# ---------------------------------------------------------------------------
# (a) a hold before a request that never went out
# ---------------------------------------------------------------------------

class TestAHoldBeforeARequestThatNeverWentOut:

    def test_the_foundry_engine_records_the_hold_on_a_row_of_its_own(self):
        pytest.importorskip("langchain_core")
        from langchain_core.messages import AIMessage

        from report_ingest.engine import ChatModelEngine
        from report_ingest.tests.chat_fixtures import ScriptedChatModel

        class _Refused(ValueError):
            nothing_sent = True

        said = AIMessage(content="ok",
                         response_metadata={"finish_reason": "stop"},
                         usage_metadata={"input_tokens": 10,
                                         "output_tokens": 5,
                                         "total_tokens": 15})
        clock = _Clock()
        model = ScriptedChatModel(replies=[said, _Refused("made-up refusal")])
        engine = ChatModelEngine(model)
        engine._clock = clock.time
        engine.pacer = _pacer(clock)
        engine.complete([user(text_block("w" * 210_000))])   # large: held next
        clock.at += 25.0
        with pytest.raises(_Refused):
            engine.complete([user(text_block("small"))])
        assert clock.slept == [35.0]
        meter = engine.meter
        assert (meter.calls, meter.failed) == (1, 0), "still no request"
        reply, held = meter.to_dict()["per_call"]
        assert "paced" not in reply
        assert held["outcome"] == "not_sent" and held["paced"] == 35.0
        assert held["started"] == "2027-01-15T08:01:00.000Z"
        assert held["error"] == "_Refused: made-up refusal"
        assert "error_kind" not in held and "input_tokens" not in held
        assert meter.to_dict()["paced_seconds"] == 35.0
        assert meter.failed_by_kind == {}

    def test_the_prompter_engine_with_no_client_records_its_hold_too(self):
        clock = _Clock()
        engine = _prompter_engine(None, clock, pacer=_pacer(clock),
                                  with_client=False,
                                  chat_reply=_response("ok"))
        engine.complete([user(text_block("w" * 210_000))])   # through chat()
        clock.at += 15.0
        with pytest.raises(RuntimeError, match="no OpenAI client"):
            engine.complete([user(text_block("small"))], tools=TOOL)
        assert clock.slept == [45.0]
        meter = engine.meter
        assert (meter.calls, meter.failed) == (1, 0)
        held = meter.to_dict()["per_call"][-1]
        assert held["outcome"] == "not_sent" and held["paced"] == 45.0
        assert meter.to_dict()["paced_seconds"] == 45.0

    def test_nothing_held_and_nothing_sent_records_nothing_as_before(self):
        clock = _Clock()
        engine = _prompter_engine(None, clock, pacer=_pacer(clock),
                                  with_client=False)
        with pytest.raises(RuntimeError, match="no OpenAI client"):
            engine.complete([user(text_block("small"))], tools=TOOL)
        assert engine.meter.per_call == []
        assert "paced_seconds" not in engine.meter.to_dict()


# ---------------------------------------------------------------------------
# (b) paced_seconds in the attempt keys, results.json and RESULTS
# ---------------------------------------------------------------------------

def _run_file(calls, paced=None, seconds=30.0, **extra):
    cost = {"calls": calls, "failed": 0, "retried": 0, "input_tokens": 100,
            "output_tokens": 10, "waited_seconds": 0.0, "dollars": 0.0}
    if paced is not None:
        cost["paced_seconds"] = paced
    return {"id": extra.pop("id", "R-1"), "run_date": "2027-01-15",
            "seconds": seconds, "cost": cost, **extra}


class TestPacedSecondsAcrossAttempts:

    def test_an_earlier_attempt_keeps_its_paced_seconds(self):
        entry = cs._attempt_entry(_run_file(3, paced=12.5, error="made up"))
        assert entry["cost"]["paced_seconds"] == 12.5
        assert entry["ended"] == "error"

    def test_an_attempt_nothing_held_carries_no_key(self):
        entry = cs._attempt_entry(_run_file(3))
        assert "paced_seconds" not in entry["cost"]

    def test_all_attempts_adds_them_up(self):
        earlier = [cs._attempt_entry(_run_file(2, paced=7.5)),
                   cs._attempt_entry(_run_file(1))]
        fields = cs._attempts_fields(earlier, {"calls": 4,
                                               "paced_seconds": 20.0}, 9.0)
        assert fields["all_attempts"]["paced_seconds"] == 27.5
        assert fields["all_attempts"]["attempts"] == 3

    def test_all_attempts_with_no_hold_is_what_it_was(self):
        earlier = [cs._attempt_entry(_run_file(2))]
        fields = cs._attempts_fields(earlier, {"calls": 4}, 9.0)
        assert "paced_seconds" not in fields["all_attempts"]
        assert set(fields["all_attempts"]) == {
            "attempts", "calls", "failed", "retried", "input_tokens",
            "output_tokens", "cache_read_tokens", "cache_write_tokens",
            "reasoning_tokens", "reasoning_reported", "unpriced_calls",
            "failed_by_kind", "dollars", "waited_seconds", "seconds"}


class TestPacedSecondsInTheResults:

    def test_a_stage_total_carries_them_only_once_one_file_has_them(self):
        total = cs._cost_totals()
        cs._add_cost(total, {"calls": 2, "failed": 0})
        assert "paced_seconds" not in total
        assert "held back" not in cs._failure_line(total)
        cs._add_cost(total, {"calls": 1, "failed": 0, "paced_seconds": 41.0})
        cs._add_cost(total, {"calls": 1, "failed": 0, "paced_seconds": 19.0})
        assert total["paced_seconds"] == 60.0
        assert "60 s held back by the pacer" in cs._failure_line(total)

    def test_the_ingest_results_count_every_attempt_s_holds(self):
        held = _run_file(2, paced=12.5, id="R-1", earlier_attempts=[
            {"run_date": "2027-01-14", "ended": "error", "seconds": 5.0,
             "cost": {"calls": 1, "paced_seconds": 7.5}}])
        plain = _run_file(3, id="R-2")
        out = cs._score_ingest({"R-1": held, "R-2": plain}, {}, "made-up")
        assert out["cost"]["paced_seconds"] == 20.0
        assert out["cost_last_attempt"]["paced_seconds"] == 12.5
        # The run file's own block, as results.json keeps it per report.
        assert out["per_report"][0]["cost"]["paced_seconds"] == 12.5

    def test_unfinished_reports_holds_reach_every_report(self):
        cost = cs._cost_totals()
        cs._add_cost(cost, {"calls": 2, "paced_seconds": 10.0})
        out = cs._unfinished_cost(cost, {"R-3": _run_file(1, paced=5.0,
                                                          error="made up")})
        assert out["cost_unfinished"]["paced_seconds"] == 5.0
        assert out["cost_every_report"]["paced_seconds"] == 15.0

    def test_with_no_hold_anywhere_the_results_carry_no_key(self):
        out = cs._score_ingest({"R-1": _run_file(2), "R-2": _run_file(3)},
                               {}, "made-up",
                               unfinished={"R-3": _run_file(1, error="x")})
        for block in ("cost", "cost_unfinished", "cost_every_report"):
            assert "paced_seconds" not in out[block], block


# ---------------------------------------------------------------------------
# (c) calls on two threads at once share one engine
# ---------------------------------------------------------------------------

class _ThreadClock:
    """A clock per thread, by thread name, so each call's times are its
    own and a row that took another call's times says so."""

    def __init__(self, starts):
        self.at = dict(starts)

    def time(self):
        return self.at[threading.current_thread().name]

    def advance(self, seconds):
        self.at[threading.current_thread().name] += float(seconds)


class TestTwoCallsOnTwoThreads:

    def test_each_row_carries_its_own_call_s_fields(self):
        """Call A's first request is rate limited only once call B's
        request is under way; B's reply comes back while A waits; then A is
        sent again. Each row must carry its own call's start, duration,
        size, attempt, number and deployment."""
        clock = _ThreadClock({"call-a": T0, "call-b": T0 + 500.0})
        a_in, b_in, a_waiting, b_done = (threading.Event(), threading.Event(),
                                         threading.Event(), threading.Event())
        a_tries = []

        def create(**kwargs):
            who = kwargs["messages"][-1]["content"][:1]
            if who == "A":
                a_tries.append(1)
                if len(a_tries) == 1:
                    clock.advance(2.0)
                    a_in.set()
                    assert b_in.wait(10), "call B never started"
                    raise RateLimitError("Error code: 429 - made-up limit")
                clock.advance(3.0)
                return _response("a", prompt=111, completion=11,
                                 model="made-up-deploy-a")
            b_in.set()
            assert a_waiting.wait(10), "call A never waited"
            clock.advance(7.0)
            return _response("b", prompt=222, completion=22,
                             model="made-up-deploy-b")

        def sleep(seconds):
            clock.advance(seconds)
            a_waiting.set()
            assert b_done.wait(10), "call B never came back"

        engine = PrompterEngine(_Prompter(create), meter=CostMeter())
        engine._clock = clock.time
        engine._sleep = sleep
        a_messages = [user(text_block("A" + "a" * 6999))]
        b_messages = [user(text_block("B" + "b" * 699))]
        replies, errors = {}, []

        def run(name, messages, done=None):
            try:
                replies[name] = engine.complete(messages, tools=TOOL)
            except BaseException as exc:            # noqa: BLE001 - reported
                errors.append(exc)
            finally:
                if done is not None:
                    done.set()

        a = threading.Thread(target=run, name="call-a",
                             args=("A", a_messages))
        b = threading.Thread(target=run, name="call-b",
                             args=("B", b_messages, b_done))
        a.start()
        # B begins only once A's first request is under way, so A's call
        # record was made first and B's after it.
        assert a_in.wait(10), "call A never sent"
        b.start()
        a.join(20)
        b.join(20)
        assert not a.is_alive() and not b.is_alive()
        assert errors == []

        a_error, b_reply, a_reply = engine.meter.per_call
        assert (a_error["call"], a_error["attempt"]) == (1, 1)
        assert a_error["outcome"] == "error"
        assert a_error["started"] == "2027-01-15T08:00:00.000Z"
        assert a_error["duration"] == 2.0 and a_error["waited"] == 15.0
        assert a_error["input_tokens"] == estimate_request(a_messages,
                                                           tools=TOOL)
        assert a_error["input_tokens"] != estimate_request(b_messages,
                                                           tools=TOOL)

        assert (b_reply["call"], b_reply["attempt"]) == (2, 1)
        assert b_reply["started"] == "2027-01-15T08:08:20.000Z"
        assert b_reply["duration"] == 7.0 and b_reply["seconds"] == 7.0
        assert b_reply["input_tokens"] == 222
        assert b_reply["served_by"] == "made-up-deploy-b"

        # Filed under its own call, though B's row came in between.
        assert (a_reply["call"], a_reply["attempt"]) == (1, 2)
        assert a_reply["started"] == "2027-01-15T08:00:17.000Z"
        assert a_reply["duration"] == 3.0 and a_reply["seconds"] == 20.0
        assert a_reply["input_tokens"] == 111
        assert a_reply["served_by"] == "made-up-deploy-a"

        assert replies["A"].model == "made-up-deploy-a"
        assert replies["B"].model == "made-up-deploy-b"
        # Nothing of either call is left on the engine.
        for name in ("_call", "_attempt", "_sent"):
            assert not hasattr(engine, name), name

    def test_one_thread_s_calls_are_numbered_as_before(self):
        clock = _Clock()
        engine = _prompter_engine(_flaky([_limited("Error code: 429 - x")],
                                         _response("ok")), clock)
        engine.complete([user(text_block("go"))], tools=TOOL)
        engine.complete([user(text_block("again"))], tools=TOOL)
        rows = engine.meter.per_call
        assert [(r["call"], r["attempt"], r["outcome"]) for r in rows] == [
            (1, 1, "error"), (1, 2, "reply"), (2, 1, "reply")]


# ---------------------------------------------------------------------------
# (d) a retry-after of 0
# ---------------------------------------------------------------------------

class TestARetryAfterOfZero:

    @pytest.mark.parametrize("header, seconds", [
        ("0", 0.0), ("7", 7.0), ("2.5", 2.5), ("-3", None), ("inf", None),
        ("nan", None), ("Wed, 21 Oct 2026 07:28:00 GMT", None)])
    def test_the_header_reads_as_seconds_or_not_at_all(self, header,
                                                       seconds):
        from report_ingest.engine import _retry_after_of

        assert _retry_after_of(_limited("x", retry_after=header)) == seconds

    def test_the_pacer_honours_a_zero_and_keeps_its_minute_for_none(self):
        clock = _Clock()
        pacer = _pacer(clock)
        assert pacer.backoff(0.0, 15.0) == 0.0
        assert pacer.backoff(None, 15.0) == 60.0, "said nothing: a minute"
        pacer.hold("made-up-model", pacer.backoff(0.0))
        assert pacer.wait("made-up-model") == 0.0
        assert clock.slept == []

    def test_a_paced_prompter_call_told_zero_is_sent_again_at_once(self):
        clock = _Clock()
        engine = _prompter_engine(
            _flaky([_limited("Error code: 429 - x", retry_after="0")],
                   _response("ok"), clock, takes=1.0),
            clock, pacer=_pacer(clock))
        engine.complete([user(text_block("go"))], tools=TOOL)
        assert clock.slept == [0.0], "not the minute"
        engine.complete([user(text_block("again"))], tools=TOOL)
        assert clock.slept == [0.0], "and the next request is not held"
        assert "paced_seconds" not in engine.meter.to_dict()

    def test_without_a_pacer_a_zero_takes_the_ladder_as_before(self):
        clock = _Clock()
        engine = _prompter_engine(
            _flaky([_limited("Error code: 429 - x", retry_after="0")],
                   _response("ok")), clock)
        engine.complete([user(text_block("go"))], tools=TOOL)
        assert clock.slept == [PrompterEngine.RATE_LIMIT_WAITS[0]]

    def test_a_give_up_told_zero_does_not_hold_the_next_request(self):
        pytest.importorskip("langchain_core")
        from langchain_core.messages import AIMessage

        from report_ingest.engine import ChatModelEngine
        from report_ingest.tests.chat_fixtures import ScriptedChatModel

        class RateLimitGaveUp(Exception):
            """Named like a host's client's, once its retries are spent."""

        said = AIMessage(content="ok",
                         response_metadata={"finish_reason": "stop"})
        clock = _Clock()
        model = ScriptedChatModel(replies=[
            _limited("made up", retry_after="0", cls=RateLimitGaveUp), said])
        engine = ChatModelEngine(model)
        engine._clock = clock.time
        engine.pacer = _pacer(clock)
        with pytest.raises(RateLimitGaveUp):
            engine.complete([user(text_block("small"))])
        engine.complete([user(text_block("small"))])
        assert clock.slept == []


# ---------------------------------------------------------------------------
# (e) a rate limit by its words, whatever the status
# ---------------------------------------------------------------------------

class _StatusError(Exception):
    """A provider error with a status and a body, as an SDK raises one."""

    def __init__(self, message, status=400, body=None):
        super().__init__(message)
        self.status_code = status
        self.body = body


SAYS_SO = [
    _StatusError("Error code: 400 - Throttled: retry later (made up)"),
    _StatusError("Error code: 400 - bad request",
                 body={"error": {"message": "Rate limit reached for made-up "
                                            "model", "code": "made_up"}}),
    _StatusError("Error code: 400", body="Too Many Requests"),
    _StatusError("Error code: 400 - request was THROTTLED by the gateway"),
]

SAYS_NOT = [
    _StatusError("Error code: 400 - the image could not be read (made up)"),
    _StatusError("Error code: 400 - bad request",
                 body={"error": {"message": "invalid schema: made up"}}),
    _StatusError("Error code: 400", body=None),
]


class TestARateLimitByItsWords:

    @pytest.mark.parametrize("error", SAYS_SO)
    def test_a_400_that_names_a_rate_limit_is_one(self, error):
        assert is_rate_limit(error) and PrompterEngine.is_rate_limit(error)
        assert error_kind(error) == "rate_limit"

    @pytest.mark.parametrize("error", SAYS_NOT)
    def test_a_400_that_does_not_is_not_guessed_from_its_status(self,
                                                                error):
        assert not is_rate_limit(error)
        assert error_kind(error) == "other"

    def test_a_body_that_will_not_read_is_no_rate_limit_and_no_error(self):
        class _Odd(Exception):
            @property
            def body(self):
                raise ValueError("made up")

        assert error_kind(_Odd("Error code: 400")) == "other"

    @pytest.mark.parametrize("error, slept, kind", [
        (SAYS_SO[1], [60.0], "rate_limit"), (SAYS_NOT[0], [], "other")])
    def test_the_foundry_engine_paces_only_the_one_that_says_so(
            self, error, slept, kind):
        pytest.importorskip("langchain_core")
        from langchain_core.messages import AIMessage

        from report_ingest.engine import ChatModelEngine
        from report_ingest.tests.chat_fixtures import ScriptedChatModel

        said = AIMessage(content="ok",
                         response_metadata={"finish_reason": "stop"})
        clock = _Clock()
        engine = ChatModelEngine(ScriptedChatModel(replies=[error, said]))
        engine._clock = clock.time
        engine.pacer = _pacer(clock)
        with pytest.raises(_StatusError):
            engine.complete([user(text_block("small"))])
        engine.complete([user(text_block("small"))])
        assert clock.slept == slept
        assert engine.meter.per_call[0]["error_kind"] == kind

    def test_the_prompter_engine_waits_out_one_that_says_so(self):
        clock = _Clock()
        engine = _prompter_engine(_flaky([SAYS_SO[0]], _response("ok")),
                                  clock)
        assert engine.complete([user(text_block("go"))],
                               tools=TOOL).text == "ok"
        assert clock.slept == [PrompterEngine.RATE_LIMIT_WAITS[0]]
        assert engine.meter.failed_by_kind == {"rate_limit": 1}

    def test_the_prompter_engine_raises_one_that_does_not(self):
        clock = _Clock()
        engine = _prompter_engine(_flaky([SAYS_NOT[0]], _response("ok")),
                                  clock)
        with pytest.raises(_StatusError):
            engine.complete([user(text_block("go"))], tools=TOOL)
        assert clock.slept == []
        assert engine.meter.failed_by_kind == {"other": 1}
