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
TIMEOUT_ENV = "GEOTECH_VISION_CALL_TIMEOUT_S"
DEFAULT_TIMEOUT_S = 180.0
#: How many more times a side call that timed out is asked.
TIMEOUT_RETRIES = 1


class VisionCallTimeout(TimeoutError):
    """A vision side call that did not answer within its time, every time it
    was asked: the image was not read."""


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

    def acquire(self, conversation="") -> None:
        with self._cond:
            me = (next(self._tickets), conversation)
            self._waiting.append(me)
            try:
                while not (self._in_use < self.limit and self._next() is me):
                    self._cond.wait()
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
        from funhouse_agent._fileio import host_output_dir
        folder = host_output_dir()
    except Exception:  # noqa: BLE001 - fairness must never break a call
        return ""
    if not folder:
        return ""
    try:
        return os.path.normcase(os.path.realpath(folder))
    except (OSError, ValueError):
        return str(folder)


@contextmanager
def call_slot(conversation: Optional[str] = None) -> Iterator[None]:
    """Hold one of the process's vision-call slots for one model request.

    Wrap only the request itself, never work that makes further calls, so a
    holder never waits on a slot it needs to finish. ``conversation``
    defaults to this context's (:func:`conversation_key`); slots are shared
    fairly between conversations (:class:`FairSlots`)."""
    slots = _semaphore()
    if slots is None:
        yield
        return
    conv = conversation_key() if conversation is None else conversation
    slots.acquire(conv)
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
    ``detail``)."""
    if isinstance(exc, VisionCallTimeout):
        return None                       # its own retry (TIMEOUT_RETRIES)
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
    if busy_kind(exc) is not None or isinstance(exc, VisionCallTimeout):
        return False
    status = _status_of(exc)
    if status is not None and status not in (400, 422):
        return False
    return bool(_DETAIL_WORD.search(str(exc)))


def _invoke_once(model, messages, timeout: Optional[float]):
    """``model.invoke(messages)`` holding a call slot, given up after
    ``timeout`` seconds of running.

    With a limit the call runs on its own thread in a COPY of the caller's
    context, so the run's callbacks (the activity log, the token count) still
    see it. A call given up is left to finish on that thread — it still holds
    its slot, so a stalled connection keeps counting against the process's
    cap rather than letting more calls pile onto the client's pool."""
    if timeout is None:
        with call_slot():
            return model.invoke(messages)
    box: dict = {}
    started, done = threading.Event(), threading.Event()

    def run():
        try:
            with call_slot():
                started.set()
                box["value"] = model.invoke(messages)
        except BaseException as exc:  # noqa: BLE001 - handed to the caller
            box["error"] = exc
        finally:
            started.set()
            done.set()

    ctx = contextvars.copy_context()
    threading.Thread(target=ctx.run, args=(run,), daemon=True,
                     name="vision-side-call").start()
    started.wait()                      # queueing for a slot is not counted
    if not done.wait(timeout):
        raise VisionCallTimeout(
            f"the vision call gave no answer within {timeout:g} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def invoke_side_call(model, messages):
    """One vision side call: :func:`_invoke_once` under the side-call limit
    (:func:`call_timeout_s`), asked once more if it times out, and asked
    again with backoff — up to :func:`busy_tries` times in all, honouring a
    ``retry-after`` — while the model answers busy (:func:`busy_kind`). The
    waits are spent holding no slot. Raises :class:`VisionCallTimeout` when
    no ask answered in time, and a busy error that outlasted its tries with
    ``vision_tries`` set on it — the caller reports that image as not read
    (:func:`describe_error`) instead of waiting on it."""
    timeout = call_timeout_s()
    timeouts = 0
    busy = 0
    tries = busy_tries()
    while True:
        try:
            return _invoke_once(model, messages, timeout)
        except VisionCallTimeout:
            timeouts += 1
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
            if busy >= tries:
                try:
                    exc.vision_tries = busy
                except Exception:  # noqa: BLE001 - a frozen exception type
                    pass
                raise
            wait = _busy_wait_s(exc, busy - 1)
            log.warning("vision side call: model busy (%s); asking again in "
                        "%.1f s (try %d of %d)", kind, wait, busy + 1, tries)
            _sleep(wait)


class LangChainVisionEngine:
    """Adapt a LangChain chat model to the minimal vision surface v5 needs.

    Implements only ``analyze_image`` — the single method the deepagents vision
    tools call through. Construct it with any LangChain
    :class:`~langchain_core.language_models.chat_models.BaseChatModel` (or
    anything exposing a compatible ``.invoke([messages]) -> AIMessage``):

        from langchain_anthropic import ChatAnthropic
        from funhouse_agent.deep.vision_engine import LangChainVisionEngine

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

    def __init__(self, model, media_type: Optional[str] = None,
                 detail: Optional[str] = None):
        self._model = model
        self._media_type = media_type
        self._detail = detail

    @property
    def model(self):
        """The wrapped LangChain chat model."""
        return self._model

    def vision_profile(self):
        """What this model really is and what images it really takes —
        measured once per process by :mod:`funhouse_agent.vision_probe`
        (``None`` when probing is switched off)."""
        from funhouse_agent import vision_probe
        return vision_probe.profile_for(self._model)

    def analyze_image(
        self,
        image_input,
        user_prompt: str = "Describe this image.",
    ) -> str:
        """Analyze an image with the wrapped LangChain model.

        Parameters
        ----------
        image_input : bytes, bytearray, or str
            Raw image bytes, or a path to an image file. Mirrors
            ``ClaudeEngine.analyze_image`` / the ``GenAIEngine`` contract.
        user_prompt : str
            What to extract / describe from the image.

        Returns
        -------
        str
            The model's text response. The ``content`` of the returned
            ``AIMessage`` is normalized to plain text (it may come back as a
            string or as a list of content blocks depending on the provider).
        """
        from langchain_core.messages import HumanMessage

        from funhouse_agent import vision_view

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

        def message(with_detail: bool) -> "HumanMessage":
            image_url = {"url": data_uri}
            if with_detail and detail:
                image_url["detail"] = detail
            return HumanMessage(content=[
                {"type": "image_url", "image_url": image_url},
                {"type": "text", "text": user_prompt},
            ])

        try:
            response = invoke_side_call(self._model, [message(True)])
        except VisionCallTimeout:
            raise
        except Exception as exc:
            # An older model (GPT-4.1, GPT-5.2) has no "original" detail; ask
            # once more at its default rather than fail the read. Only a
            # request error that names the parameter counts (F27: a busy
            # error's body held "details" and was asked again by accident).
            if not detail or not refused_detail(exc):
                raise
            response = invoke_side_call(self._model, [message(False)])
        text = _content_to_text(getattr(response, "content", response))
        if was_cut_off(response):
            # The answer stopped at the model's output limit (wave 2b, C5):
            # its end is missing, and the result must say so.
            text = VisionAnswer(text)
            text.cut_off = True
            log.warning("vision side call: the answer was cut off at the "
                        "model's output limit (%d characters)", len(text))
        return text


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
           "VisionAnswer", "was_cut_off"]
