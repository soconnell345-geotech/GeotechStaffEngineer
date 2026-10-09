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
import logging
import os
import threading
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

_slots_lock = threading.Lock()
_slots: Optional[tuple] = None          # (limit, BoundedSemaphore)


def _semaphore() -> Optional[threading.BoundedSemaphore]:
    raw = (os.environ.get(INFLIGHT_ENV) or "").strip()
    try:
        limit = int(raw) if raw else DEFAULT_MAX_INFLIGHT
    except ValueError:
        limit = DEFAULT_MAX_INFLIGHT
    if limit <= 0:
        return None
    global _slots
    with _slots_lock:
        if _slots is None or _slots[0] != limit:
            _slots = (limit, threading.BoundedSemaphore(limit))
        return _slots[1]


@contextmanager
def call_slot() -> Iterator[None]:
    """Hold one of the process's vision-call slots for one model request.

    Wrap only the request itself, never work that makes further calls, so a
    holder never waits on a slot it needs to finish."""
    sem = _semaphore()
    if sem is None:
        yield
        return
    with sem:
        yield


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
    (:func:`call_timeout_s`), asked once more if it times out. Raises
    :class:`VisionCallTimeout` when no ask answered in time — the caller
    reports that image as not read instead of waiting on it."""
    timeout = call_timeout_s()
    for attempt in range(TIMEOUT_RETRIES + 1):
        try:
            return _invoke_once(model, messages, timeout)
        except VisionCallTimeout:
            if attempt >= TIMEOUT_RETRIES:
                raise VisionCallTimeout(
                    f"the vision call gave no answer within {timeout:g} s, "
                    f"{attempt + 1} times; this image was NOT read") from None
            log.warning("vision side call gave no answer within %g s; "
                        "asking once more", timeout)


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
            # once more at its default rather than fail the read.
            if not detail or "detail" not in str(exc).lower():
                raise
            response = invoke_side_call(self._model, [message(False)])
        return _content_to_text(getattr(response, "content", response))


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
           "invoke_side_call"]
