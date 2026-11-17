"""One ladder for a busy or stuck model call (BACKLOG TA-24, done with
INF-75).

The deployment review of 2026-11-10: on the Prompter route the OpenAI
client asked a failed request again four times, 180 s each for a timeout,
INSIDE each try of the agent's own busy ladder and inside the vision
ladder. One stuck call could hold a turn 15 to 20 minutes before the tester
saw an error, and a look waited for a vision slot with no bound while
other testers' looks queued behind it with no message.

Now: the client asks again once by default; a call that got no answer in
time is asked again at most once in all, client and ladder together; and
the wait for a vision slot is bounded, after which the look is reported as
not read because the vision model is busy with other looks.

Everything here is made up: the address, the key, the model's name.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from aec_agent_core import activity_log
from aec_agent_core.activity_log import ActivityLogger
from aec_agent_core.deep import busy_retry, vision_engine
from aec_agent_core.deep.vision_engine import (
    FairSlots, LangChainVisionEngine, VisionCallTimeout, VisionSlotTimeout,
    describe_error, side_call,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class APITimeoutError(Exception):
    """Named as the OpenAI client's timeout is."""


class RateLimitError(Exception):
    def __init__(self, msg="Too Many Requests"):
        super().__init__(msg)
        self.status_code = 429


@pytest.fixture
def no_waits(monkeypatch):
    seen = []
    monkeypatch.setattr(busy_retry, "_sleep", seen.append)
    monkeypatch.setattr(vision_engine, "_sleep", seen.append)
    monkeypatch.delenv(busy_retry.TRIES_ENV, raising=False)
    monkeypatch.delenv(vision_engine.BUSY_TRIES_ENV, raising=False)
    monkeypatch.delenv("GEOTECH_TOKENS_PER_MINUTE", raising=False)
    return seen


# ---------------------------------------------------------------------------
# What is a timeout
# ---------------------------------------------------------------------------

def test_a_timeout_is_told_from_a_quick_refusal():
    gateway = Exception("made-up: upstream took too long")
    gateway.status_code = 504
    named = Exception("made-up")
    named.error_name = "MadeUp:LlmSocketTimeout"
    wrapped = RuntimeError("the call failed")
    wrapped.__cause__ = APITimeoutError("no answer")
    for exc in (APITimeoutError("no answer"), TimeoutError(),
                named, wrapped):
        assert vision_engine.is_timeout(exc), exc
    # a 504 is a timeout only after a slow try (review, 2026-11-13)
    assert not vision_engine.is_timeout(gateway)
    assert vision_engine.is_timeout(gateway,
                                    took_s=vision_engine.SLOW_TRY_S)
    busy = Exception("made-up: unavailable")
    busy.status_code = 503
    for exc in (RateLimitError(), busy, ConnectionResetError("reset"),
                ValueError("bad request"),
                VisionSlotTimeout("busy with other looks")):
        assert not vision_engine.is_timeout(exc), exc


def test_one_retry_of_a_timeout_in_all_client_and_ladder_together():
    none = SimpleNamespace()                        # a client with no retries
    zero = SimpleNamespace(max_retries=0)
    one = SimpleNamespace(max_retries=1)
    four = SimpleNamespace(max_retries=4)
    assert vision_engine.TIMEOUT_RETRIES == 1
    # the ladder's first timed-out try: asked again only where the client
    # has not already asked again itself
    assert vision_engine.may_ask_timeout_again(none, 1)
    assert vision_engine.may_ask_timeout_again(zero, 1)
    assert not vision_engine.may_ask_timeout_again(one, 1)
    assert not vision_engine.may_ask_timeout_again(four, 1)
    # and never a second time
    assert not vision_engine.may_ask_timeout_again(zero, 2)


# ---------------------------------------------------------------------------
# The agent's ladder
# ---------------------------------------------------------------------------

def _ladder_run(model, errors):
    """Run one model call through the agent's ladder; ``errors`` are raised
    by the first tries in turn. Returns (answer or the error, tries)."""
    asked = []

    def handler(request):
        asked.append(1)
        if len(asked) <= len(errors):
            raise errors[len(asked) - 1]
        return AIMessage(content="the answer")

    request = SimpleNamespace(model=model, messages=[], tools=[],
                              system_message=None)
    try:
        out = busy_retry.AgentBusyRetry().wrap_model_call(request, handler)
    except Exception as exc:  # noqa: BLE001 - the test reads it
        out = exc
    return out, len(asked)


def test_the_ladder_asks_a_timed_out_call_again_once_not_four_times(
        no_waits):
    """A client with no retries of its own: the ladder asks once more, and
    a second timeout ends the call. Before, it walked all five tries."""
    model = SimpleNamespace(max_retries=0)
    stuck = [APITimeoutError("no answer") for _ in range(5)]
    out, tries = _ladder_run(model, stuck)
    assert isinstance(out, APITimeoutError)
    assert tries == 2 and len(no_waits) == 1
    assert out.geotech_busy["tries"] == 2
    assert "no answer in time" in out.geotech_busy["why"]
    # one timeout, then an answer: the turn goes on
    out, tries = _ladder_run(model, [APITimeoutError("no answer")])
    assert getattr(out, "content", None) == "the answer" and tries == 2


def test_the_ladder_does_not_ask_again_what_the_client_already_did(
        no_waits):
    """The Prompter client asks a timed-out request again once itself: the
    ladder then adds no third ask."""
    model = SimpleNamespace(max_retries=1)
    out, tries = _ladder_run(model, [APITimeoutError("no answer")] * 5)
    assert isinstance(out, APITimeoutError)
    assert tries == 1 and no_waits == []


def test_a_quick_refusal_still_walks_the_whole_ladder(no_waits):
    """A rate limit costs a second, not the limit: its five tries and
    their waits are as they were, whatever the client's retries; only the
    timeouts among them are bounded."""
    model = SimpleNamespace(max_retries=1)
    out, tries = _ladder_run(model, [RateLimitError()] * 4)
    assert getattr(out, "content", None) == "the answer"
    assert tries == 5 and len(no_waits) == 4
    # a timeout after two rate limits is still the one that ends it
    no_waits.clear()
    out, tries = _ladder_run(model, [RateLimitError(), RateLimitError(),
                                     APITimeoutError("no answer"),
                                     RateLimitError()])
    assert isinstance(out, APITimeoutError) and tries == 3


def test_a_timeout_the_ladder_waits_on_says_so_on_the_record(no_waits,
                                                             monkeypatch):
    events = []
    monkeypatch.setattr(busy_retry, "_event",
                        lambda name, data: events.append((name, data)))
    model = SimpleNamespace(max_retries=0)
    _ladder_run(model, [APITimeoutError("no answer")])
    _ladder_run(model, [RateLimitError()])
    assert [name for name, _ in events] == ["model_busy", "model_busy"]
    assert events[0][1]["timed_out"] is True
    assert "timed_out" not in events[1][1]


# ---------------------------------------------------------------------------
# The Prompter client
# ---------------------------------------------------------------------------

def test_the_prompter_client_asks_again_once_by_default(monkeypatch):
    from aec_agent_core import tinyapps_engine as te
    monkeypatch.delenv(te.MAX_RETRIES_ENV, raising=False)
    assert te.DEFAULT_MAX_RETRIES == 1 and te.max_retries() == 1
    monkeypatch.setenv(te.MAX_RETRIES_ENV, "0")
    assert te.max_retries() == 0
    monkeypatch.setenv(te.MAX_RETRIES_ENV, "4")
    assert te.max_retries() == 4                     # still the override


def _stuck_prompter(monkeypatch, requests):
    """The page's own Prompter model against an endpoint that never
    answers in time."""
    pytest.importorskip("langchain_openai")
    from aec_agent_core import tinyapps_engine as te
    from aec_agent_core.engine_config import sdk_http_package
    # the HTTP package openai's clients are built on: the transport and
    # the timeout must come from the same one
    httpx = sdk_http_package(pytest.importorskip("openai"))

    def never_answers(request):
        requests.append(1)
        raise httpx.ReadTimeout("made-up: no answer", request=request)

    real_init = httpx.Client.__init__

    def mocked_init(self, *a, **kw):
        kw["transport"] = httpx.MockTransport(never_answers)
        real_init(self, *a, **kw)

    monkeypatch.setattr(httpx.Client, "__init__", mocked_init)
    return te.build_chat_model(prompter=te.PrompterSettings(
        url="https://prompter.test/api/v1/chat/completions", model="m",
        api_key="k"))


@pytest.mark.parametrize("client_retries", [None, "0"])
def test_a_stuck_prompter_call_is_two_requests_in_all(monkeypatch, no_waits,
                                                      client_retries):
    """End to end on the route the testers use: the endpoint never answers.
    Before: five requests a try, five tries, 25 requests at 180 s each
    (the turn's 900 s bound cut it at about 15 minutes). Now two requests
    in all, whichever of the client and the ladder asks again."""
    from aec_agent_core import tinyapps_engine as te
    if client_retries is None:
        monkeypatch.delenv(te.MAX_RETRIES_ENV, raising=False)
    else:
        monkeypatch.setenv(te.MAX_RETRIES_ENV, client_retries)
    requests: list = []
    model = _stuck_prompter(monkeypatch, requests)
    request = SimpleNamespace(model=model, messages=[], tools=[],
                              system_message=None)
    with pytest.raises(Exception) as caught:
        busy_retry.AgentBusyRetry().wrap_model_call(
            request, lambda r: model.invoke([HumanMessage(content="q")]))
    assert vision_engine.is_timeout(caught.value)
    assert len(requests) == 2
    # the ladder waited once only where the client did not ask again itself
    assert len(no_waits) == (0 if client_retries is None else 1)


# ---------------------------------------------------------------------------
# The vision ladder
# ---------------------------------------------------------------------------

class _Fails:
    """A vision model whose every ask raises ``error``."""

    def __init__(self, error, max_retries=None):
        self.error = error
        self.calls = 0
        if max_retries is not None:
            self.max_retries = max_retries

    def invoke(self, messages):
        self.calls += 1
        raise self.error


def test_the_vision_ladder_bounds_a_client_timeout_the_same_way(
        no_waits, monkeypatch):
    """With the side-call limit off, the client's own timeout reaches the
    vision ladder as a busy error: it was asked three times."""
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "0")
    once = _Fails(APITimeoutError("no answer"), max_retries=1)
    with pytest.raises(APITimeoutError):
        LangChainVisionEngine(once, detail="").analyze_image(PNG, "x")
    assert once.calls == 1
    plain = _Fails(APITimeoutError("no answer"))
    with pytest.raises(APITimeoutError) as caught:
        LangChainVisionEngine(plain, detail="").analyze_image(PNG, "x")
    assert plain.calls == 2
    assert "NOT read" in describe_error(caught.value)
    # a rate limit keeps its three tries
    limited = _Fails(RateLimitError(), max_retries=1)
    with pytest.raises(RateLimitError):
        LangChainVisionEngine(limited, detail="").analyze_image(PNG, "x")
    assert limited.calls == 3


# ---------------------------------------------------------------------------
# The wait for a vision slot
# ---------------------------------------------------------------------------

def test_a_wait_for_a_slot_can_be_bounded():
    slots = FairSlots(1)
    assert slots.acquire("alice") is True
    t0 = time.monotonic()
    assert slots.acquire("bob", timeout=0.2) is False
    assert 0.15 < time.monotonic() - t0 < 2
    assert slots.in_flight() == 1 and slots.in_flight("bob") == 0
    # the one who gave up left the line: the next waiter is served
    got = []
    t = threading.Thread(target=lambda: got.append(slots.acquire("carol")))
    t.start()
    time.sleep(0.05)
    slots.release("alice")
    t.join(2)
    assert got == [True] and slots.in_flight("carol") == 1
    # a slot that is free is taken at once, bound or no bound
    slots.release("carol")
    assert slots.acquire("bob", timeout=0.2) is True


def test_the_default_bound_and_its_setting(monkeypatch):
    monkeypatch.delenv(vision_engine.SLOT_WAIT_ENV, raising=False)
    assert vision_engine.slot_wait_s() == vision_engine.DEFAULT_SLOT_WAIT_S \
        == 600.0
    monkeypatch.setenv(vision_engine.SLOT_WAIT_ENV, "120")
    assert vision_engine.slot_wait_s() == 120.0
    monkeypatch.setenv(vision_engine.SLOT_WAIT_ENV, "0")
    assert vision_engine.slot_wait_s() is None            # no bound
    monkeypatch.delenv(vision_engine.SLOT_WAIT_ENV, raising=False)
    policy = busy_retry.retry_policy()
    assert policy["vision_slot_wait_s"] == 600.0
    assert policy["timeout_asks_again"] == 1


class _Holds:
    """A vision model whose first call holds its slot until released."""

    def __init__(self):
        self.calls = 0
        self.release = threading.Event()
        self.lock = threading.Lock()

    def invoke(self, messages):
        with self.lock:
            self.calls += 1
            n = self.calls
        if n == 1:
            self.release.wait(10)
        return AIMessage(content=f"answer {n}")


def test_a_look_that_cannot_get_a_slot_is_reported_not_left_waiting(
        monkeypatch, tmp_path):
    """Every slot held by another look: before, the second look waited as
    long as the first took, with no word. Now it gives up at the bound,
    says the vision model is busy with other looks, and is on the record."""
    monkeypatch.setenv(vision_engine.INFLIGHT_ENV, "1")
    monkeypatch.setenv(vision_engine.SLOT_WAIT_ENV, "0.3")
    monkeypatch.setenv(vision_engine.TIMEOUT_ENV, "30")
    model = _Holds()
    engine = LangChainVisionEngine(model, detail="")
    first = threading.Thread(
        target=lambda: engine.analyze_image(PNG, "x"), daemon=True)
    first.start()
    for _ in range(100):                      # until the first holds the slot
        if model.calls:
            break
        time.sleep(0.01)

    @tool
    def look_at_page(page: int) -> str:
        """Look at a page."""
        with side_call(kind="page", page=page):
            try:
                return engine.analyze_image(PNG, "what is here?")
            except Exception as exc:  # noqa: BLE001 - as the page's tools do
                return f"not read: {describe_error(exc)}"

    logger = ActivityLogger(str(tmp_path), turn=1)
    t0 = time.monotonic()
    try:
        out = look_at_page.invoke({"page": 3},
                                  config={"callbacks": [logger]})
        waited = time.monotonic() - t0
    finally:
        model.release.set()
        first.join(5)
    assert 0.25 < waited < 5
    assert "busy with other looks" in out and "NOT read" in out
    assert "\n" not in out                          # one plain line
    assert model.calls == 1                         # never sent, never retried
    events = [r for r in activity_log.load(str(tmp_path))
              if r.get("event") == "vision_slot_timeout"]
    assert len(events) == 1
    ev = events[0]
    assert ev["limit_s"] == 0.3 and ev["waited_s"] >= 0.3
    assert ev["in_flight"] == 1
    assert (ev["kind"], ev["page"]) == ("page", 3)
    assert ev["tool"] == "look_at_page"
    # not a side-call timeout: the call was never sent
    assert not [r for r in activity_log.load(str(tmp_path))
                if r.get("event") == "side_call_timeout"]


def test_a_slot_timeout_is_a_not_read_and_never_a_busy_retry():
    exc = VisionSlotTimeout("the vision model is busy with other looks: no "
                            "vision call could start within 600 s; this "
                            "image was NOT read — try again in a minute")
    assert isinstance(exc, VisionCallTimeout)       # read as "not read"
    assert vision_engine.busy_kind(exc) is None
    assert not vision_engine.refused_detail(exc)
    assert describe_error(exc) == str(exc)
    assert vision_engine.SLOT_TIMEOUT_EVENT in activity_log.APP_EVENTS
