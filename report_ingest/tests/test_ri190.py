"""A request that fails with a server error is sent again (BACKLOG RI-190).

Build f lost a model-first log item to a Foundry HTTP 500 that nothing sent
again: the engine raised it at once and the item waited for a resume. The
engines now send a 5xx again on a short ladder, as they do a rate limit,
each request a row of the call in the cost record. No model is called and
nothing waits for real; every message, status and time here is made up.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from report_ingest.engine import (
    FAILURE_KINDS, SERVER_ERROR_LADDER, ChatModelEngine, PrompterEngine,
    error_kind, is_rate_limit, is_server_error, text_block, user,
)


class _StatusError(Exception):
    """A provider error with a status, as an SDK raises one."""

    def __init__(self, message, status=None, on_response=False):
        super().__init__(message)
        if on_response:
            self.response = SimpleNamespace(status_code=status, headers={})
        elif status is not None:
            self.status_code = status


class _ConjureLike(Exception):
    """Shaped like the error Foundry's model service raises: requests'
    words, the status on its response."""


def _conjure(status=500):
    error = _ConjureLike(f"{status} Server Error: Internal Server Error for "
                         f"url: https://made-up.example/api/llm/MADE_UP")
    error.response = SimpleNamespace(status_code=status, headers={})
    return error


SERVER_ERRORS = [
    _StatusError("made up", status=500),
    _StatusError("made up", status=502, on_response=True),
    _StatusError("Error code: 503 - made up"),
    _StatusError("Error code: 529 - {'type': 'overloaded_error'}"),
    _ConjureLike("500 Server Error: Internal Server Error (made up)"),
    _conjure(504),
]

NOT_SERVER_ERRORS = [
    _StatusError("Error code: 400 - the image could not be read (made up)",
                 status=400),
    _StatusError("Error code: 429 - too many requests", status=429),
    # A rate limit sent as a 5xx is a rate limit.
    _StatusError("Error code: 503 - rate limit reached (made up)", status=503),
    RuntimeError("the made-up gauge read 500 kPa"),
]


class TestAServerErrorIsKnown:

    @pytest.mark.parametrize("error", SERVER_ERRORS)
    def test_by_its_status_or_its_words(self, error):
        assert is_server_error(error)
        assert not is_rate_limit(error)
        assert error_kind(error) in ("server_error", "timeout")

    @pytest.mark.parametrize("error", NOT_SERVER_ERRORS)
    def test_and_nothing_else_is(self, error):
        assert not is_server_error(error)

    def test_on_the_error_it_was_raised_from(self):
        wrapped = RuntimeError("the model call failed (made up)")
        wrapped.__cause__ = _conjure(500)
        assert is_server_error(wrapped)

    def test_it_is_a_kind_of_its_own(self):
        assert "server_error" in FAILURE_KINDS
        assert error_kind(_conjure(500)) == "server_error"


class _Clock:
    """Moves only when told; a sleep moves it and is kept."""

    def __init__(self, at=1_800_000_000.0):     # 2027-01-15T08:00:00Z
        self.at = at
        self.slept = []

    def time(self):
        return self.at

    def sleep(self, seconds):
        self.slept.append(float(seconds))
        self.at += float(seconds)


def _rows(meter):
    return [{"attempt": 1, **row} for row in meter.to_dict()["per_call"]]


class TestTheChatModelEngine:

    @pytest.fixture(autouse=True)
    def _langchain(self):
        pytest.importorskip("langchain_core")

    def _engine(self, *replies):
        from report_ingest.tests.chat_fixtures import ScriptedChatModel

        model = ScriptedChatModel(replies=list(replies))
        engine = ChatModelEngine(model)
        clock = _Clock()
        engine._clock = clock.time
        engine._sleep = clock.sleep
        return engine, model, clock

    @staticmethod
    def _said(text):
        from langchain_core.messages import AIMessage

        return AIMessage(content=text,
                         response_metadata={"finish_reason": "stop"},
                         usage_metadata={"input_tokens": 10,
                                         "output_tokens": 5,
                                         "total_tokens": 15})

    def test_a_500_is_sent_again_and_the_reply_is_the_call_s(self):
        engine, model, clock = self._engine(_conjure(500), _conjure(502),
                                            self._said("ok"))
        reply = engine.complete([user(text_block("go"))])
        assert reply.text == "ok" and len(model.requests) == 3
        assert clock.slept == list(SERVER_ERROR_LADDER[:2])
        meter = engine.meter
        assert (meter.calls, meter.failed, meter.retried) == (1, 2, 2)
        assert meter.failed_by_kind == {"server_error": 2}
        rows = _rows(meter)
        assert [(r["call"], r["attempt"], r["outcome"]) for r in rows] == [
            (1, 1, "error"), (1, 2, "error"), (1, 3, "reply")]
        assert all(r["retried"] for r in rows[:2])
        assert [r["waited"] for r in rows[:2]] == list(SERVER_ERROR_LADDER[:2])
        assert rows[2]["seconds"] == sum(SERVER_ERROR_LADDER[:2])

    def test_one_that_never_clears_is_raised_after_the_ladder(self):
        errors = [_conjure(500) for _ in range(len(SERVER_ERROR_LADDER) + 1)]
        engine, model, clock = self._engine(*errors)
        with pytest.raises(_ConjureLike):
            engine.complete([user(text_block("go"))])
        assert len(model.requests) == len(SERVER_ERROR_LADDER) + 1
        assert clock.slept == list(SERVER_ERROR_LADDER)
        rows = _rows(engine.meter)
        assert [r.get("retried", False) for r in rows] == \
            [True] * len(SERVER_ERROR_LADDER) + [False]
        assert rows[-1]["attempt"] == len(SERVER_ERROR_LADDER) + 1
        assert len({r["call"] for r in rows}) == 1

    def test_anything_else_is_still_raised_at_once(self):
        engine, model, clock = self._engine(NOT_SERVER_ERRORS[0],
                                            self._said("not asked"))
        with pytest.raises(_StatusError):
            engine.complete([user(text_block("go"))])
        assert len(model.requests) == 1 and clock.slept == []
        (row,) = engine.meter.to_dict()["per_call"]
        assert "attempt" not in row and "retried" not in row


class TestThePrompterEngine:

    def test_a_500_is_sent_again_on_the_ladder(self):
        from report_ingest.tests.test_prompter_engine import (
            _TOOL, _fussy_engine)

        engine, prompter = _fussy_engine(refuse=())
        clock = _Clock()
        engine._sleep = clock.sleep
        create = prompter.client.chat.completions.create
        errors = [_StatusError("Error code: 500 - made up", status=500)]

        def flaky(**kwargs):
            if errors:
                raise errors.pop(0)
            return create(**kwargs)

        prompter.client.chat.completions.create = flaky
        reply = engine.complete([user(text_block("go"))], tools=_TOOL)
        assert reply.text == "ok"
        assert clock.slept == [PrompterEngine.SERVER_ERROR_WAITS[0]]
        rows = _rows(engine.meter)
        assert [(r["attempt"], r["outcome"]) for r in rows] == [
            (1, "error"), (2, "reply")]
        assert rows[0]["error_kind"] == "server_error" and rows[0]["retried"]
