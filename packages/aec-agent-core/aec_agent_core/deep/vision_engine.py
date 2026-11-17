"""``LangChainVisionEngine`` — the v5.0 vision adapter for the deepagents port.

The deepagents vision tools (``analyze_image`` / ``analyze_pdf_page`` /
``read_reference_figure`` in :mod:`funhouse_agent.deep.tools`) call
``engine.analyze_image(image_bytes, prompt) -> str`` exactly like the v1
``GenAIEngine`` protocol (see :class:`funhouse_agent.engine.ClaudeEngine`). v5
drives a single LangChain :class:`~langchain_core.language_models.chat_models.BaseChatModel`
for both text and vision, so this thin adapter lets that one model object satisfy
the vision surface — no separate Anthropic/OpenAI SDK client required.

It mirrors ``ClaudeEngine.analyze_image`` but emits the **standard LangChain
multimodal message shape** instead of a provider-specific one: a
:class:`~langchain_core.messages.HumanMessage` whose ``content`` is a list of
blocks ``[{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
{"type": "text", "text": prompt}]``. LangChain's provider integrations
(``langchain_anthropic``, ``langchain_openai``, …) translate that shape to the
right native format, so the same adapter works across providers.

Dependency-light: only ``langchain_core`` (for ``HumanMessage``) and stdlib
``base64``.
"""

from __future__ import annotations

import base64
import contextvars
import email.utils
import itertools
import logging
import os
import random
import re
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Iterator, Optional

log = logging.getLogger(__name__)

#: Seconds one vision SIDE call may run before it is given up and asked once
#: more (:data:`TIMEOUT_RETRIES`). Foundry brief 5 (2026-10-08): side calls
#: inherited the engine's 900 s read timeout, set for the primary model's long
#: reasoning, so one stalled tile held a turn for 907 s (and 343 s, and 903 s
#: in brief 4) while the p99 side call took 74 s. The clock starts when the
#: call holds its slot (:func:`call_slot`), so queueing behind other calls is
#: not counted. ``0`` = wait as long as the engine does. The primary agent's
#: own calls never pass through here and keep the engine's timeout.
#:
#: The limit is set from the measured tail (BACKLOG VIS-31, 2026-11-13). It
#: was 180 s. Foundry brief 7 (2026-11-12): 15,545 tile calls took 5 s at
#: the median and under 22 s for 99 %, and each of 34 stalled tiles held
#: its batch for the whole 180 s before a second ask that answered in 1 to
#: 2 s (about 350 s of one 841 s run). Brief 5 (another deployment,
#: 2026-10-08) had a 99th percentile of 74 s. 90 s is above both tails, so
#: on neither deployment is a call that would have answered given up as a
#: matter of course: about four times brief 7's 99th percentile and 1.2
#: times brief 5's. The trace review suggested 60 s; that is under brief
#: 5's 74 s, where more than one call in a hundred would be abandoned while
#: still working, asked a second time on an endpoint already slow, and
#: would keep its slot until it ended. Each timeout and each late end is
#: now an event (:data:`TIMEOUT_EVENT`, :data:`LATE_END_EVENT`; the late
#: end's ``held_s`` is how long the abandoned call really took), so the
#: next suite run measures whether 60 s would have cost any answer.
TIMEOUT_ENV = "GEOTECH_VISION_CALL_TIMEOUT_S"
DEFAULT_TIMEOUT_S = 90.0
#: How many more times a side call that timed out is asked.
TIMEOUT_RETRIES = 1


class VisionCallTimeout(TimeoutError):
    """A vision side call that did not answer within its time, every time it
    was asked: the image was not read."""


class VisionSlotTimeout(VisionCallTimeout):
    """No vision call slot came free within the wait bound
    (:func:`slot_wait_s`): the image was not read, and the call was never
    sent. Its text is the one plain line a tool result gives
    (:func:`describe_error`). Never asked again by the ladders here: the
    wait it reports is already the bound."""


#: The longest one look waits for a vision call slot (BACKLOG TA-24, a
#: repair: the wait had no bound). The process has :data:`INFLIGHT_ENV`
#: slots, and a call given up at its time limit keeps its slot until its
#: connection ends, so with every slot held by a stuck call the looks of
#: every tester queued with no end and no word. After this many seconds
#: the look is reported as not read, because the vision model is busy with
#: other looks. 600 s: at the measured 5 s median a side call takes (brief
#: 7), eight slots serve about 960 calls in that time, more than fifty
#: whole-page looks of 17 calls queued ahead, so a queue that is only long
#: does not reach it; a queue behind stuck calls does. ``0`` = no bound.
SLOT_WAIT_ENV = "GEOTECH_VISION_SLOT_WAIT_S"
DEFAULT_SLOT_WAIT_S = 600.0

#: The app event of a look that gave up waiting for a slot: ``waited_s``,
#: ``limit_s``, the calls ``in_flight`` then, and the look's ``kind``,
#: ``page`` and ``tile``.
SLOT_TIMEOUT_EVENT = "vision_slot_timeout"


def slot_wait_s() -> Optional[float]:
    """The bound on one look's wait for a vision slot in seconds
    (:data:`SLOT_WAIT_ENV`), or ``None`` for none."""
    raw = (os.environ.get(SLOT_WAIT_ENV) or "").strip()
    try:
        limit = float(raw) if raw else DEFAULT_SLOT_WAIT_S
    except ValueError:
        limit = DEFAULT_SLOT_WAIT_S
    return limit if limit > 0 else None


def call_timeout_s() -> Optional[float]:
    """The side-call time limit in seconds (:data:`TIMEOUT_ENV`), or
    ``None`` for none."""
    raw = (os.environ.get(TIMEOUT_ENV) or "").strip()
    try:
        limit = float(raw) if raw else DEFAULT_TIMEOUT_S
    except ValueError:
        limit = DEFAULT_TIMEOUT_S
    return limit if limit > 0 else None

#: At most this many vision side calls run at once in one process. A turn can
#: fan out several ways at once — the model asks for many pages in one step,
#: each look tiles a sheet four ways, the markup check reads four marks — and
#: nothing else bounds the product. On Foundry (2026-10-02) a few dozen calls
#: in flight overflowed the client's 10-connection pool and aborted the
#: process; any host with a rate limit pays for it too. Eight leaves room in
#: a 10-connection pool for the agent's own calls. ``0`` = no cap.
INFLIGHT_ENV = "GEOTECH_VISION_MAX_INFLIGHT"
DEFAULT_MAX_INFLIGHT = 8

#: A hard cap on one conversation's calls in flight, inside the process cap
#: (``0``, the default, = none). Fairness does not need it: a freed slot
#: always goes to the waiting conversation with the FEWEST calls in flight
#: (:class:`FairSlots`), so one tester's 30-call set read cannot queue a
#: second tester's one-page question behind all of it (live smoke wave 2a,
#: B13), while a tester alone still gets every slot. Set it to keep slots
#: free for newcomers outright.
PER_CONVERSATION_ENV = "GEOTECH_VISION_MAX_PER_CONVERSATION"


class FairSlots:
    """The process's vision-call slots, shared FAIRLY between conversations.

    At most ``limit`` holders at once. When a slot frees, the waiter served
    is the one whose conversation has the fewest calls in flight, the oldest
    first among equals — so a conversation alone uses every slot, and one
    that arrives while another holds them all is served at the next free
    slot instead of after the other's whole queue. ``per_conversation``
    (``0`` = none) also caps any one conversation's holders. Never re-entered
    by a holder: :func:`call_slot` wraps one model request only."""

    def __init__(self, limit: int, per_conversation: int = 0):
        self.limit = int(limit)
        self.per_conversation = max(0, int(per_conversation))
        self._cond = threading.Condition()
        self._in_use = 0
        self._by_conv: dict = {}
        self._waiting: list = []
        self._tickets = itertools.count()

    def in_flight(self, conversation=None) -> int:
        """Calls holding a slot (for one conversation, or all)."""
        with self._cond:
            if conversation is None:
                return self._in_use
            return self._by_conv.get(conversation, 0)

    def _eligible(self, conv) -> bool:
        return (not self.per_conversation
                or self._by_conv.get(conv, 0) < self.per_conversation)

    def _next(self):
        ready = [w for w in self._waiting if self._eligible(w[1])]
        if not ready:
            return None
        return min(ready, key=lambda w: (self._by_conv.get(w[1], 0), w[0]))

    def acquire(self, conversation="", timeout: Optional[float] = None
                ) -> bool:
        """Take a slot, waiting for one. With ``timeout`` (seconds) the
        wait is bounded: ``False`` when no slot came in that time, and the
        caller holds none (BACKLOG TA-24). ``True`` otherwise."""
        deadline = (None if timeout is None
                    else time.monotonic() + max(0.0, float(timeout)))
        with self._cond:
            me = (next(self._tickets), conversation)
            self._waiting.append(me)
            try:
                while not (self._in_use < self.limit and self._next() is me):
                    if deadline is None:
                        self._cond.wait()
                        continue
                    left = deadline - time.monotonic()
                    if left <= 0:
                        self._waiting.remove(me)
                        self._cond.notify_all()   # the next may be served
                        return False
                    self._cond.wait(left)
            except BaseException:
                self._waiting.remove(me)
                self._cond.notify_all()      # the next in line may be served
                raise
            self._waiting.remove(me)
            self._in_use += 1
            self._by_conv[conversation] = \
                self._by_conv.get(conversation, 0) + 1
            # Another slot may still be free for the next waiter in line.
            self._cond.notify_all()
            return True

    def release(self, conversation="") -> None:
        with self._cond:
            self._in_use = max(0, self._in_use - 1)
            n = self._by_conv.get(conversation, 0) - 1
            if n > 0:
                self._by_conv[conversation] = n
            else:
                self._by_conv.pop(conversation, None)
            self._cond.notify_all()


_slots_lock = threading.Lock()
_slots: Optional[tuple] = None          # ((limit, per_conv), FairSlots)


def _int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _semaphore() -> Optional[FairSlots]:
    limit = _int_env(INFLIGHT_ENV, DEFAULT_MAX_INFLIGHT)
    if limit <= 0:
        return None
    per = max(0, _int_env(PER_CONVERSATION_ENV, 0))
    global _slots
    with _slots_lock:
        if _slots is None or _slots[0] != (limit, per):
            _slots = ((limit, per), FairSlots(limit, per))
        return _slots[1]


def conversation_key() -> str:
    """Which conversation the call in this context belongs to: the working
    folder the host bound for the turn (as the document toolkits are kept,
    :mod:`funhouse_agent.document_tools`), ``""`` for a library caller with
    none."""
    try:
        from aec_agent_core._fileio import host_output_dir
        folder = host_output_dir()
    except Exception:  # noqa: BLE001 - fairness must never break a call
        return ""
    if not folder:
        return ""
    try:
        return os.path.normcase(os.path.realpath(folder))
    except (OSError, ValueError):
        return str(folder)


#: What the vision side call running in this context is (BACKLOG EV-14):
#: ``kind`` (tile, page, region, check...), ``page`` and ``tile`` when the
#: caller gave them, the image's pixel size and ``detail``, and ``queued_s``
#: (the wait for a vision slot). Set by :func:`side_call` around one look;
#: the activity log puts it on the call's ``model_start``. Field session
#: 2026-10-14: 607 of 916 side calls could not be tied to anything.
_SIDE_CALL: "contextvars.ContextVar[Optional[dict]]" = contextvars.ContextVar(
    "geotech_side_call", default=None)


@contextmanager
def side_call(**label) -> Iterator[dict]:
    """``with side_call(kind="tile", page=4, tile="r1c2"):`` -- label the
    vision call(s) made inside."""
    info = {k: v for k, v in label.items() if v is not None}
    token = _SIDE_CALL.set(info)
    try:
        yield info
    finally:
        try:
            _SIDE_CALL.reset(token)
        except (ValueError, RuntimeError):
            pass


def current_side_call() -> Optional[dict]:
    """The label of the vision side call running in this context, if any."""
    return _SIDE_CALL.get()


def _note_side_call(**facts) -> None:
    info = _SIDE_CALL.get()
    if info is not None:
        info.update({k: v for k, v in facts.items() if v is not None})


@contextmanager
def call_slot(conversation: Optional[str] = None) -> Iterator[None]:
    """Hold one of the process's vision-call slots for one model request.

    Wrap only the request itself, never work that makes further calls, so a
    holder never waits on a slot it needs to finish. ``conversation``
    defaults to this context's (:func:`conversation_key`); slots are shared
    fairly between conversations (:class:`FairSlots`). The wait is bounded
    (:func:`slot_wait_s`): past it :class:`VisionSlotTimeout` is raised,
    with a :data:`SLOT_TIMEOUT_EVENT` in the run record."""
    slots = _semaphore()
    if slots is None:
        _note_side_call(queued_s=0.0)
        yield
        return
    conv = conversation_key() if conversation is None else conversation
    t0 = time.monotonic()
    bound = slot_wait_s()
    if not slots.acquire(conv, timeout=bound):
        waited = round(time.monotonic() - t0, 3)
        _note_side_call(queued_s=waited)
        _event(SLOT_TIMEOUT_EVENT, {
            **_look_facts(), "waited_s": waited, "limit_s": bound,
            "in_flight": slots.in_flight()})
        raise VisionSlotTimeout(
            f"the vision model is busy with other looks: no vision call "
            f"could start within {bound:g} s; this image was NOT read — "
            f"try again in a minute")
    # The wait for the slot, on the call's record (EV-14): a slow tool is
    # told apart from a queued one.
    _note_side_call(queued_s=round(time.monotonic() - t0, 3))
    try:
        yield
    finally:
        slots.release(conv)


# ---------------------------------------------------------------------------
# A busy model: retried with backoff (live smoke wave 2a, B7)
# ---------------------------------------------------------------------------

#: How many times in all a side call is asked when the model answers "busy"
#: (429, 5xx, overloaded, a dropped connection). Several testers share one
#: Prompter key on Tiny Apps, and the SDK's own two quick retries are spent
#: in under a second; F27's overloaded tile survived only by accident.
BUSY_TRIES_ENV = "GEOTECH_VISION_BUSY_TRIES"
DEFAULT_BUSY_TRIES = 3
#: First wait before asking again, in seconds; doubled each time, with some
#: jitter so callers that failed together do not retry together.
BUSY_BACKOFF_S = 2.0
#: Longest wait honoured from a ``retry-after`` header, in seconds.
MAX_RETRY_AFTER_S = 30.0

#: The wait between tries (tests replace it).
_sleep = time.sleep
#: The clock an ask's running time is read off, from its slot to its end
#: (:data:`SLOW_TRY_S`, a late end's ``held_s``; tests replace it).
_clock = time.monotonic

_BUSY_TYPES = {
    "ratelimiterror", "internalservererror", "overloadederror",
    "serviceunavailableerror", "apiconnectionerror", "apitimeouterror",
    "badgatewayerror", "gatewaytimeouterror",
}
_BUSY_BODY_TYPES = {"overloaded_error", "rate_limit_error", "api_error",
                    "server_error", "rate_limit_exceeded"}
_BUSY_ERROR_NAMES = ("ratelimit", "overloaded", "unavailable", "timeout")


def _status_of(exc) -> Optional[int]:
    """The HTTP status an SDK error carries, if any."""
    for holder in (exc, getattr(exc, "response", None)):
        for attr in ("status_code", "status", "http_status"):
            v = getattr(holder, attr, None)
            if isinstance(v, int) and 100 <= v < 600:
                return v
    return None


def _body_error_type(exc) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error") if isinstance(body.get("error"), dict) \
            else body
        for key in ("type", "code"):
            v = err.get(key) if isinstance(err, dict) else None
            if isinstance(v, str):
                return v.lower()
    return ""


def _chain(exc):
    """``exc`` and the errors it was raised FROM (a wrapper's explicit
    cause; not an unrelated error it was raised while handling)."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__


def busy_kind(exc) -> Optional[str]:
    """What kind of "busy" ``exc`` is — ``rate limit``, ``overloaded``,
    ``server error 503``, ``connection`` — or ``None`` for an error that
    asking again would not fix. Read off the error's type, HTTP status and
    body, never off words in its message (F27's overload carried
    ``'details': None``, which a substring test took for a refused
    ``detail``).

    A rate limit the error NAMES is one whatever its status: Foundry's
    model service sends it as a 400 whose ``errorName`` is
    ``RateLimitsExceeded`` or a QoS exception (:func:`funhouse_agent.
    error_text.rate_limit_named`; BACKLOG INF-62). Any other 400 is not
    busy.

    An AI budget or quota that is USED UP is not busy, whatever its status
    (an ``insufficient_quota`` 429 included): asking again cannot fix it,
    so it is never retried (:func:`funhouse_agent.error_text.
    budget_exhausted`; live smoke wave 2c, D2)."""
    if isinstance(exc, VisionCallTimeout):
        return None                       # its own retry (TIMEOUT_RETRIES)
    if _budget_spent(exc):
        return None
    if _rate_limit_named(exc):
        return "rate limit"
    for e in _chain(exc):
        status = _status_of(e)
        body = _body_error_type(e)
        name = type(e).__name__.lower()
        if status == 429 or "ratelimit" in name or body in (
                "rate_limit_error", "rate_limit_exceeded"):
            return "rate limit"
        if status == 529 or "overloaded" in name or body == "overloaded_error":
            return "overloaded"
        if status is not None and 500 <= status < 600:
            return f"server error {status}"
        if status in (408, 409):
            return f"server error {status}"
        if name in _BUSY_TYPES or body in _BUSY_BODY_TYPES:
            return "server error"
        err_name = str(getattr(e, "error_name", "") or "").lower()
        if err_name and any(w in err_name for w in _BUSY_ERROR_NAMES):
            return "rate limit" if "ratelimit" in err_name else "server error"
        if isinstance(e, (ConnectionError, TimeoutError)):
            return "connection"
    return None


#: Error classes that mean "the request got no answer in time": the OpenAI
#: and Anthropic clients' ``APITimeoutError``, httpx's and requests'
#: timeouts.
_TIMEOUT_TYPES = {"apitimeouterror", "timeoutexception", "readtimeout",
                  "connecttimeout", "writetimeout", "pooltimeout", "timeout",
                  "readtimeouterror"}
#: HTTP statuses that mean the same seen from a gateway: it gave up waiting.
#: Only after a slow try (:data:`SLOW_TRY_S`): see :func:`is_timeout`.
_TIMEOUT_STATUSES = (408, 504)

#: How long a try must have lasted, in seconds, before a 408 or 504 counts
#: as a timeout (review of BACKLOG TA-24, 2026-11-13). A gateway that
#: answers 408 or 504 QUICKLY is refusing, not timing out, and keeps the
#: whole busy ladder, as before TA-24; one that answers it after a long wait
#: has cost about what a timeout costs. 20 s is a judgment, not a
#: measurement: it is about the 99th percentile of a side call in Foundry
#: brief 7 (under 22 s), so a try that lasted longer was waiting, not
#: working. No gateway's own time limit has been measured.
SLOW_TRY_S = 20.0


def is_timeout(exc, took_s: Optional[float] = None) -> bool:
    """Whether ``exc`` is a model request that got no answer IN TIME, as
    against one refused quickly (a rate limit, an overload): the client's
    own timeout, an error a service names a timeout, or a gateway's 408 or
    504 after a try that lasted at least :data:`SLOW_TRY_S` (``took_s``,
    the seconds the try took; a 408 or 504 with ``took_s`` unknown or
    shorter is a quick refusal). Read off the error's type, status and
    name, never its text.

    A timeout is the slow way to fail: each ask costs the whole limit. So
    one is asked again at most once in all, by the client and the ladders
    together (:func:`may_ask_timeout_again`; BACKLOG TA-24)."""
    if isinstance(exc, VisionSlotTimeout):
        return False
    for e in _chain(exc):
        if isinstance(e, TimeoutError):
            return True
        if type(e).__name__.lower() in _TIMEOUT_TYPES:
            return True
        if "timeout" in str(getattr(e, "error_name", "") or "").lower():
            return True
    if took_s is None or took_s < SLOW_TRY_S:
        return False
    return any(_status_of(e) in _TIMEOUT_STATUSES for e in _chain(exc))


def client_retries(model) -> int:
    """How many times ``model``'s own client asks a failed request again
    (its ``max_retries``); 0 for a model that keeps no such count."""
    value = getattr(model, "max_retries", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0


def may_ask_timeout_again(model, timeouts: int) -> bool:
    """Whether a ladder may ask again after its ``timeouts``-th timed-out
    try of one call: only while the client's own retries and the ladder's
    come to at most :data:`TIMEOUT_RETRIES` (one) in all. A client that
    already asks again by itself (``max_retries`` of 1 or more) has used
    the one retry inside the try, so the ladder asks no more; with a
    client that does not, the ladder asks once.

    What the rule bounds is the asks a ladder sees. A vision side call
    given up at its own limit (:data:`TIMEOUT_ENV`, shorter than the
    client's request timeout) is asked once more by
    :func:`invoke_side_call` whatever the client's retries, and each ask
    given up keeps running on its thread, where the client may still send
    its request once more: with the client asking again once, up to four
    requests reach the service for one look. Each ask given up is a
    :data:`TIMEOUT_EVENT` and each that ends a :data:`LATE_END_EVENT`, but
    the record cannot count the client's re-sends inside an ask: a request
    that timed out got no response, so the Tiny Apps route's per-call HTTP
    list (``http_tries``, :mod:`aec_agent_core.http_trace`) never sees it.

    Before (the deployment review of 2026-11-10): the Prompter client asked
    a timed-out request four more times at 180 s each inside every try of
    the agent's ladder, 15 minutes before the tester saw a word."""
    return int(timeouts) + client_retries(model) <= TIMEOUT_RETRIES


def _budget_spent(exc):
    """The :class:`~funhouse_agent.error_text.BudgetStop` of an error that
    says the AI budget is used up, else ``None``."""
    try:
        from aec_agent_core.error_text import budget_exhausted
        return budget_exhausted(exc)
    except Exception:  # noqa: BLE001 - classifying must never fail a call
        return None


def _rate_limit_named(exc) -> bool:
    """Whether an error NAMES a rate limit, as Foundry's 400s do
    (:func:`funhouse_agent.error_text.rate_limit_named`)."""
    try:
        from aec_agent_core.error_text import rate_limit_named
        return rate_limit_named(exc)
    except Exception:  # noqa: BLE001 - classifying must never fail a call
        return False


def _retry_after_s(exc) -> Optional[float]:
    """Seconds a ``retry-after`` (or ``retry-after-ms``) header asks for."""
    for e in _chain(exc):
        headers = getattr(getattr(e, "response", None), "headers", None)
        if headers is None:
            continue
        try:
            ms = headers.get("retry-after-ms")
            if ms is not None:
                return max(0.0, float(ms) / 1000.0)
            raw = headers.get("retry-after")
        except Exception:  # noqa: BLE001 - an odd header object
            continue
        if raw is None:
            continue
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            pass
        try:
            when = email.utils.parsedate_to_datetime(str(raw))
            return max(0.0, when.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            continue
    return None


def busy_tries() -> int:
    """How many times in all a busy side call is asked (at least 1)."""
    return max(1, _int_env(BUSY_TRIES_ENV, DEFAULT_BUSY_TRIES))


def _busy_wait_s(exc, attempt: int) -> float:
    after = _retry_after_s(exc)
    if after is not None:
        return min(after, MAX_RETRY_AFTER_S)
    base = BUSY_BACKOFF_S * (2 ** attempt)
    return base * random.uniform(0.75, 1.25)


def describe_error(exc) -> str:
    """One plain line saying why a vision call gave no reading — what a tool
    result (and so the user) is told instead of an SDK's raw error text."""
    if isinstance(exc, VisionCallTimeout):
        return str(exc)
    stop = _budget_spent(exc)
    if stop is not None:
        from aec_agent_core.error_text import BUDGET_HINT, budget_tool_note
        return f"{budget_tool_note(stop)} -- this image was NOT read. " \
               f"{BUDGET_HINT}"
    kind = busy_kind(exc)
    tries = getattr(exc, "vision_tries", None)
    if kind:
        asked = f" after {tries} tries" if tries and tries > 1 else ""
        what = {"rate limit": "busy (its rate limit was reached)",
                "overloaded": "busy (overloaded)",
                "connection": "unreachable (the connection failed)"}.get(
                    kind, f"busy ({kind})")
        return (f"the vision model was {what} and gave no answer{asked}; "
                f"this image was NOT read — try again in a minute")
    text = " ".join(str(exc).split())
    if len(text) > 300:
        text = text[:300] + " …"
    return f"{type(exc).__name__}: {text}"


_DETAIL_WORD = re.compile(r"\bdetail\b", re.IGNORECASE)


def refused_detail(exc) -> bool:
    """Whether ``exc`` is the model refusing the image ``detail`` value: a
    request error (400/422, or no status at all) that names the parameter —
    not a busy error whose body happens to hold the word ``details``."""
    if busy_kind(exc) is not None or isinstance(exc, VisionCallTimeout) \
            or _budget_spent(exc) is not None:
        return False
    status = _status_of(exc)
    if status is not None and status not in (400, 422):
        return False
    return bool(_DETAIL_WORD.search(str(exc)))


#: The app events a side call writes to the run record (the activity log's
#: ``APP_EVENTS``; BACKLOG INF-75). Foundry brief 7 (2026-11-12): 44 side
#: calls in 84 records were given up at the limit with only a log line, so
#: each left a ``model_start`` with no ``model_end`` and nothing said why.
#:
#: * ``side_call_timeout``: an ask got no answer within ``limit_s`` and was
#:   given up: the look's ``kind``, ``page`` and ``tile``, which ``ask`` of
#:   the side call it was and its ``ask_id`` (also on that ask's
#:   ``model_start``, in ``vision``), ``attempt`` (the timeouts so far) and
#:   ``asked_again`` (false: the image was reported as not read).
#: * ``side_call_late_end``: an ask that was given up ended after all,
#:   having held its vision slot ``held_s`` seconds: ``outcome``
#:   (``answered`` or ``error``, with the error's class) and the same
#:   ``ask_id``. An abandoned ask with no such event was still running, and
#:   still holding its slot, when the record stopped.
TIMEOUT_EVENT = "side_call_timeout"
LATE_END_EVENT = "side_call_late_end"


def _event(name: str, data: dict) -> None:
    """Write one app event to the run record, as
    :func:`aec_agent_core.deep.busy_retry._event` does. Never raises: a
    caller outside a run (a library call) has no record to write to."""
    try:
        from langchain_core.callbacks.manager import dispatch_custom_event
        dispatch_custom_event(name, data)
    except Exception:  # noqa: BLE001 - the record never fails a call
        pass


def _look_facts() -> dict:
    """What the look in this context is, for an event: ``kind``, ``page``
    and ``tile`` off its label (:func:`current_side_call`)."""
    info = _SIDE_CALL.get() or {}
    return {k: info[k] for k in ("kind", "page", "tile") if k in info}


def _invoke_once(model, messages, timeout: Optional[float],
                 facts: Optional[dict] = None,
                 timing: Optional[dict] = None):
    """``model.invoke(messages)`` holding a call slot, given up after
    ``timeout`` seconds of running.

    With a limit the call runs on its own thread in a COPY of the caller's
    context, so the run's callbacks (the activity log, the token count) still
    see it. A call given up is left to finish on that thread — it still holds
    its slot, so a stalled connection keeps counting against the process's
    cap rather than letting more calls pile onto the client's pool. When it
    does end, a :data:`LATE_END_EVENT` says how long it held the slot
    (``facts``: what the event says of the ask). A call that ends in time
    puts its running time, slot to end, in ``timing["ran_s"]``."""
    if timeout is None:
        with call_slot():
            t_run = _clock()
            try:
                return model.invoke(messages)
            finally:
                if timing is not None:
                    timing["ran_s"] = _clock() - t_run
    box: dict = {}
    started, done = threading.Event(), threading.Event()
    # Whether the caller gave the call up is settled under this lock, so a
    # call that ends as the limit passes is either returned or reported as
    # a late end, never both and never neither.
    settle = threading.Lock()

    def run():
        try:
            with call_slot():
                box["t_slot"] = _clock()
                started.set()
                box["value"] = model.invoke(messages)
        except BaseException as exc:  # noqa: BLE001 - handed to the caller
            box["error"] = exc
        finally:
            started.set()
            if box.get("t_slot") is not None:
                box["ran_s"] = _clock() - box["t_slot"]
            with settle:
                done.set()
                abandoned = bool(box.get("abandoned"))
            if abandoned:
                error = box.get("error")
                ran = box.get("ran_s")
                _event(LATE_END_EVENT, {
                    **(facts or {}), "limit_s": timeout,
                    "held_s": round(ran, 3) if ran is not None else None,
                    "outcome": "error" if error is not None else "answered",
                    **({"error": type(error).__name__}
                       if error is not None else {})})

    ctx = contextvars.copy_context()
    threading.Thread(target=ctx.run, args=(run,), daemon=True,
                     name="vision-side-call").start()
    started.wait()                      # queueing for a slot is not counted
    if not done.wait(timeout):
        with settle:
            if not done.is_set():
                box["abandoned"] = True
                raise VisionCallTimeout(
                    f"the vision call gave no answer within {timeout:g} s")
    if timing is not None and "ran_s" in box:
        timing["ran_s"] = box["ran_s"]
    if "error" in box:
        raise box["error"]
    return box["value"]


def _pace(model, messages):
    """``(pacer, ticket)`` after waiting for room on ``model``'s key, or
    ``(None, None)`` with pacing off. Never raises."""
    try:
        from aec_agent_core.deep.pacing import estimate_tokens, pace
        return pace(model, estimate_tokens(messages), what="vision")
    except Exception:  # noqa: BLE001 - pacing never fails a call
        return None, None


def _settle(pacer, ticket, response) -> None:
    if pacer is None:
        return
    try:
        from aec_agent_core.deep.pacing import usage_tokens
        pacer.settle(ticket, usage_tokens(response))
    except Exception:  # noqa: BLE001
        pass


def invoke_side_call(model, messages):
    """One vision side call: :func:`_invoke_once` under the side-call limit
    (:func:`call_timeout_s`), asked once more if it times out, and asked
    again with backoff — up to :func:`busy_tries` times in all, honouring a
    ``retry-after`` — while the model answers busy (:func:`busy_kind`). A
    timeout of the client's own, or a 408 or 504 after a slow ask
    (:func:`is_timeout`), is asked again at most once in all, client and
    ladder together (:func:`may_ask_timeout_again`). An ask given up at the
    side-call limit is asked once more whatever the client's retries, and
    its client may still re-send it on its own thread: up to four requests
    for one look with the client asking again once. The waits are spent
    holding no slot. Raises :class:`VisionCallTimeout` when
    no ask answered in time, and a busy error that outlasted its tries with
    ``vision_tries`` set on it — the caller reports that image as not read
    (:func:`describe_error`) instead of waiting on it.

    Each ask given up at the limit is a :data:`TIMEOUT_EVENT` in the run
    record, and each ask carries its number and an id on its own
    ``model_start`` (``vision.ask``, ``vision.ask_id``), so a start with no
    end can be matched to the timeout that left it (BACKLOG INF-75)."""
    timeout = call_timeout_s()
    timeouts = 0
    busy = 0
    tries = busy_tries()
    ask = 0
    while True:
        # Room on this key under the token pacer, when one is set
        # (funhouse_agent.deep.pacing; BACKLOG TA-22): waited for before
        # the slot, so a paced call holds none.
        pacer, ticket = _pace(model, messages)
        ask += 1
        ask_id = uuid.uuid4().hex[:12]
        _note_side_call(ask=ask, ask_id=ask_id)
        facts = {**_look_facts(), "ask": ask, "ask_id": ask_id}
        timing: dict = {}
        try:
            response = _invoke_once(model, messages, timeout, facts, timing)
            _settle(pacer, ticket, response)
            return response
        except VisionSlotTimeout:
            raise                         # the wait was the bound: not asked
        except VisionCallTimeout:
            timeouts += 1
            _event(TIMEOUT_EVENT, {
                **facts, "limit_s": timeout, "attempt": timeouts,
                "asked_again": timeouts <= TIMEOUT_RETRIES})
            if timeouts > TIMEOUT_RETRIES:
                raise VisionCallTimeout(
                    f"the vision call gave no answer within {timeout:g} s, "
                    f"{timeouts} times; this image was NOT read") from None
            log.warning("vision side call gave no answer within %g s; "
                        "asking once more", timeout)
        except Exception as exc:  # noqa: BLE001 - classified, then re-raised
            kind = busy_kind(exc)
            if kind is None:
                raise
            busy += 1
            # The client's own timeout (reached where the side-call limit
            # is off or longer), or a 408 or 504 after a slow ask: counted
            # with the side-call timeouts, and asked again at most once in
            # all (TA-24). A quick 408 or 504 keeps every try.
            timed_out = is_timeout(exc, timing.get("ran_s"))
            if timed_out:
                timeouts += 1
            if busy >= tries or (
                    timed_out and not may_ask_timeout_again(model, timeouts)):
                try:
                    exc.vision_tries = busy
                except Exception:  # noqa: BLE001 - a frozen exception type
                    pass
                raise
            wait = _busy_wait_s(exc, busy - 1)
            log.warning("vision side call: model busy (%s); asking again in "
                        "%.1f s (try %d of %d)", kind, wait, busy + 1, tries)
            _sleep(wait)


#: The fields a LangChain chat model keeps its output cap in: ``max_tokens``
#: (ChatAnthropic; ChatOpenAI, sent as ``max_completion_tokens`` or the
#: Responses API's ``max_output_tokens``; the Foundry SDK and Prompter
#: wrappers), then the other spellings some integrations use.
_CAP_FIELDS = ("max_tokens", "max_output_tokens", "max_completion_tokens")


def capped_model(model, cap):
    """``model`` with its per-answer output cap held to ``cap`` tokens, as a
    shallow copy that shares the original's client; ``model`` itself when
    its own cap is already that low, or when it keeps no cap a copy can
    change (then the call goes out as before).

    Live smoke wave 2c (E1): the vision side calls used the main model
    object, so since the output cap went to 32,000 (C5) one vision answer
    could run to 10,778 tokens (70 s) — on a reasoning model the cap holds
    the reasoning as well as the reply."""
    try:
        cap = int(cap)
    except (TypeError, ValueError):
        return model
    if cap <= 0:
        return model
    fields = getattr(type(model), "model_fields", None)
    copy = getattr(model, "model_copy", None)
    if not isinstance(fields, dict) or not callable(copy):
        return model
    for name in _CAP_FIELDS:
        if name not in fields:
            continue
        current = getattr(model, name, None)
        if isinstance(current, int) and not isinstance(current, bool) \
                and 0 < current <= cap:
            return model
        try:
            return copy(update={name: cap})
        except Exception:  # noqa: BLE001 - an uncopyable model keeps its cap
            return model
    return model


class LangChainVisionEngine:
    """Adapt a LangChain chat model to the minimal vision surface v5 needs.

    Implements only ``analyze_image`` — the single method the deepagents vision
    tools call through. Construct it with any LangChain
    :class:`~langchain_core.language_models.chat_models.BaseChatModel` (or
    anything exposing a compatible ``.invoke([messages]) -> AIMessage``):

        from langchain_anthropic import ChatAnthropic
        from aec_agent_core.deep.vision_engine import LangChainVisionEngine

        engine = LangChainVisionEngine(ChatAnthropic(model="claude-sonnet-4-6"))
        text = engine.analyze_image(png_bytes, "Read Kp off this chart.")

    Parameters
    ----------
    model : BaseChatModel
        The LangChain chat model that performs the vision call. It must be
        vision-capable for real images; for offline tests a fake model whose
        ``.invoke`` returns an ``AIMessage`` works.
    media_type : str, optional
        MIME type embedded in the data URI. ``None`` (default) reads it from
        the bytes — a render can be PNG or JPEG (see
        :mod:`funhouse_agent.vision_view`).
    detail : str, optional
        OpenAI's image ``detail`` level (``"high"``, ``"original"``...). The
        default follows the app's image budget
        (:func:`funhouse_agent.vision_view.detail`); ``""`` omits the field.
        A model that rejects the value is asked once more without it.
    """

    # Marks this as a vision-capable engine (parity with the v1 engines, which
    # expose feature sentinels like ``native_tool_calling``). It is NOT a
    # native-tool-calling engine — deepagents handles tool calling itself.
    native_tool_calling = False

    #: The data URI is labelled with the bytes' real type, so a JPEG render
    #: may be sent (see ``vision_view.render_view(allow_jpeg=...)``).
    accepts_jpeg = True

    #: :meth:`analyze_image` takes ``max_output_tokens``: a cap for that one
    #: answer (live smoke wave 2c, E1). Engines without this attribute are
    #: called as before.
    accepts_output_cap = True

    def __init__(self, model, media_type: Optional[str] = None,
                 detail: Optional[str] = None):
        self._model = model
        self._media_type = media_type
        self._detail = detail
        self._capped: dict = {}

    @property
    def model(self):
        """The wrapped LangChain chat model."""
        return self._model

    def _model_for(self, cap):
        """The model to ask with an output cap of ``cap`` tokens (a copy made
        once per cap, :func:`capped_model`); the model itself for none."""
        if not cap:
            return self._model
        m = self._capped.get(cap)
        if m is None:
            m = self._capped[cap] = capped_model(self._model, cap)
        return m

    def vision_profile(self):
        """What this model really is and what images it really takes —
        measured once per process by :mod:`funhouse_agent.vision_probe`
        (``None`` when probing is switched off)."""
        from aec_agent_core import vision_probe
        return vision_probe.profile_for(self._model)

    def analyze_image(
        self,
        image_input,
        user_prompt: str = "Describe this image.",
        max_output_tokens: Optional[int] = None,
    ) -> str:
        """Analyze an image with the wrapped LangChain model.

        Parameters
        ----------
        image_input : bytes, bytearray, or str
            Raw image bytes, or a path to an image file. Mirrors
            ``ClaudeEngine.analyze_image`` / the ``GenAIEngine`` contract.
        user_prompt : str
            What to extract / describe from the image.
        max_output_tokens : int, optional
            The most tokens this answer may take (reasoning included, on a
            reasoning model), never more than the model's own cap; ``None``
            = the model's own cap. An answer that stops there comes back
            with ``cut_off`` set.

        Returns
        -------
        str
            The model's text response. The ``content`` of the returned
            ``AIMessage`` is normalized to plain text (it may come back as a
            string or as a list of content blocks depending on the provider).
        """
        from langchain_core.messages import HumanMessage

        from aec_agent_core import vision_view

        if isinstance(image_input, (bytes, bytearray)):
            data = bytes(image_input)
        elif isinstance(image_input, str):
            with open(image_input, "rb") as f:
                data = f.read()
        else:
            raise TypeError(
                f"image_input must be bytes or file path, got {type(image_input)}"
            )

        media_type = self._media_type or vision_view.image_media_type(data)
        data_uri = f"data:{media_type};base64,{base64.b64encode(data).decode()}"
        detail = (self._detail if self._detail is not None
                  else vision_view.detail(self)) or None
        _note_side_call(image_px=_image_px(data), detail=detail,
                        image_bytes=len(data))
        # A failed probe's fallback budget, on this call's record in
        # activity.jsonl (DR-36). Read from the cache only: never a probe
        # that the call would not have made anyway.
        from aec_agent_core import vision_probe
        fallback = vision_view.probe_fallback_note(
            vision_probe.cached_profile(self._model))
        if fallback:
            _note_side_call(probe_fallback=fallback)

        def message(with_detail: bool) -> "HumanMessage":
            image_url = {"url": data_uri}
            if with_detail and detail:
                image_url["detail"] = detail
            return HumanMessage(content=[
                {"type": "image_url", "image_url": image_url},
                {"type": "text", "text": user_prompt},
            ])

        model = self._model_for(max_output_tokens)
        try:
            response = invoke_side_call(model, [message(True)])
        except VisionCallTimeout:
            raise
        except Exception as exc:
            # An older model (GPT-4.1, GPT-5.2) has no "original" detail; ask
            # once more at its default rather than fail the read. Only a
            # request error that names the parameter counts (F27: a busy
            # error's body held "details" and was asked again by accident).
            if not detail or not refused_detail(exc):
                raise
            response = invoke_side_call(model, [message(False)])
        text = _content_to_text(getattr(response, "content", response))
        if was_cut_off(response):
            # The answer stopped at the model's output limit (wave 2b, C5):
            # its end is missing, and the result must say so.
            text = VisionAnswer(text)
            text.cut_off = True
            log.warning("vision side call: the answer was cut off at the "
                        "output limit (%s tokens; %d characters of answer)",
                        max_output_tokens or "the model's", len(text))
        return text


def _image_px(data: bytes) -> Optional[list]:
    """``[width, height]`` of an image, read from its header; ``None``."""
    try:
        import io
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            return [int(im.size[0]), int(im.size[1])]
    except Exception:  # noqa: BLE001 - a record, never the call
        return None


class VisionAnswer(str):
    """A vision answer as text; ``cut_off`` is True when the model stopped
    at its output limit, so the end of the reading is missing."""

    cut_off = False


#: Finish reasons that mean "stopped at the output limit": OpenAI's
#: ``length``, Anthropic's ``max_tokens``, the Responses API's
#: ``max_output_tokens``.
_CUT_REASONS = {"length", "max_tokens", "max_output_tokens"}


def was_cut_off(response) -> bool:
    """Whether a model response stopped at its output limit, read off its
    metadata (``finish_reason`` / ``stop_reason`` / an incomplete Responses
    status), never off the text."""
    meta = getattr(response, "response_metadata", None)
    if not isinstance(meta, dict):
        return False
    reasons = [meta.get("finish_reason"), meta.get("stop_reason")]
    info = meta.get("generation_info")
    if isinstance(info, dict):
        reasons.append(info.get("finish_reason"))
    incomplete = meta.get("incomplete_details")
    if isinstance(incomplete, dict):
        reasons.append(incomplete.get("reason"))
    return any(isinstance(r, str) and r.lower() in _CUT_REASONS
               for r in reasons)


def _content_to_text(content) -> str:
    """Normalize an ``AIMessage.content`` (str or list of blocks) to text.

    LangChain providers may return ``content`` as a plain string or as a list
    of content blocks (e.g. ``[{"type": "text", "text": "..."}, ...]``). This
    flattens any text blocks and ignores non-text blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Standard LangChain text block, or a Claude-style {"type":"text"}.
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                elif "text" in block and isinstance(block["text"], str):
                    parts.append(block["text"])
        return "".join(parts)
    return str(content)


__all__ = ["LangChainVisionEngine", "call_slot", "INFLIGHT_ENV",
           "DEFAULT_MAX_INFLIGHT", "TIMEOUT_ENV", "DEFAULT_TIMEOUT_S",
           "TIMEOUT_RETRIES", "VisionCallTimeout", "call_timeout_s",
           "invoke_side_call", "FairSlots", "PER_CONVERSATION_ENV",
           "conversation_key", "busy_kind", "busy_tries", "describe_error",
           "refused_detail", "BUSY_TRIES_ENV", "DEFAULT_BUSY_TRIES",
           "VisionAnswer", "was_cut_off", "capped_model", "side_call",
           "current_side_call", "TIMEOUT_EVENT", "LATE_END_EVENT",
           "VisionSlotTimeout", "SLOT_WAIT_ENV", "DEFAULT_SLOT_WAIT_S",
           "SLOT_TIMEOUT_EVENT", "slot_wait_s", "is_timeout",
           "SLOW_TRY_S", "client_retries", "may_ask_timeout_again"]
