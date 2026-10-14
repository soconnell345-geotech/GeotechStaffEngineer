"""A per-key token pacer over agent and vision calls (BACKLOG TA-22).

Field session 2026-10-14: the last refusals came with ONE helper running;
the minute before them carried at least ~295,000 tokens of tool results in
eight agent calls. That fits a tokens-per-minute limit on the deployment,
which no cap on helpers or vision slots bounds: what counts is how many
tokens go out on one key in a minute.

:class:`TokenPacer` keeps the tokens sent on one key over the last 60 s and
makes a call wait until its estimate fits under the limit. The estimate is
made from the request (characters / 4, a fixed cost per image) and
replaced by the provider's own count when the reply carries usage, so the
window is measured where it can be. One pacer per key
(:func:`pacer_for`): a key is the endpoint and a fingerprint of the API key
(never the key itself), so testers on keys of their own do not wait for
each other.

OFF unless ``GEOTECH_TOKENS_PER_MINUTE`` is set (tokens a minute on one
key; ``0`` = off): the deployment's real limit is not known to the app,
and a wrong guess would slow every turn. Set it a little under the
deployment's limit.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import deque
from typing import Any, Iterable, Optional

TPM_ENV = "GEOTECH_TOKENS_PER_MINUTE"

#: The window the limit is counted over, in seconds.
WINDOW_S = 60.0

#: Characters per token, for the estimate.
CHARS_PER_TOKEN = 4

#: Tokens counted for one image in a request (a high-detail tile set is of
#: this order on the GPT tiers; the provider's own count replaces it).
IMAGE_TOKENS = 1_500

#: The longest single sleep while waiting, so a settled estimate is seen
#: soon.
_MAX_NAP_S = 2.0

#: The wait and the clock (tests replace them).
_sleep = time.sleep
_now = time.monotonic


def tokens_per_minute() -> int:
    raw = (os.environ.get(TPM_ENV) or "").strip()
    try:
        return max(0, int(float(raw))) if raw else 0
    except ValueError:
        return 0


class TokenPacer:
    """Tokens sent on one key in the last :data:`WINDOW_S` seconds, at most
    ``limit``. A call larger than the whole limit goes when the window is
    empty (it can never fit better)."""

    def __init__(self, limit: int, clock=None):
        self.limit = int(limit)
        self._clock = clock or (lambda: _now())
        self._lock = threading.Lock()
        self._sent: deque = deque()       # [time, tokens, ticket]
        self._tickets = 0

    def _prune(self, now: float) -> None:
        while self._sent and now - self._sent[0][0] >= WINDOW_S:
            self._sent.popleft()

    def used(self) -> int:
        with self._lock:
            self._prune(self._clock())
            return sum(e[1] for e in self._sent)

    def acquire(self, tokens: int) -> tuple:
        """Wait until ``tokens`` fit in the window; ``(ticket, waited_s)``."""
        tokens = max(0, int(tokens))
        t0 = self._clock()
        while True:
            with self._lock:
                now = self._clock()
                self._prune(now)
                used = sum(e[1] for e in self._sent)
                if not self._sent or used + tokens <= self.limit:
                    self._tickets += 1
                    self._sent.append([now, tokens, self._tickets])
                    return self._tickets, now - t0
                nap = self._sent[0][0] + WINDOW_S - now
            _sleep(max(0.05, min(nap, _MAX_NAP_S)))

    def settle(self, ticket: int, tokens: Optional[int]) -> None:
        """Replace a call's estimate with the tokens the provider counted."""
        if tokens is None:
            return
        with self._lock:
            for e in self._sent:
                if e[2] == ticket:
                    e[1] = max(0, int(tokens))
                    return


_PACERS: dict = {}
_PACERS_LOCK = threading.Lock()


def _secret(value: Any) -> str:
    get = getattr(value, "get_secret_value", None)
    try:
        return str(get() if callable(get) else (value or ""))
    except Exception:  # noqa: BLE001
        return ""


def model_key(model: Any) -> str:
    """Which key a chat model's calls count against: its endpoint and a
    fingerprint of its API key (12 hex characters of a SHA-256; the key is
    never kept or logged), else its endpoint and model name."""
    seen = set()
    m = model
    # a model bound to tools, or a copy with a smaller cap, is the same key
    while m is not None and id(m) not in seen:
        seen.add(id(m))
        inner = getattr(m, "bound", None)
        if inner is None or inner is m:
            break
        m = inner
    base = (getattr(m, "openai_api_base", None) or getattr(m, "base_url", None)
            or getattr(getattr(m, "prompter", None), "base_url", None) or "")
    key = _secret(getattr(m, "openai_api_key", None)
                  or getattr(m, "api_key", None)
                  or getattr(getattr(m, "prompter", None), "api_key", None))
    name = (getattr(m, "model_name", None) or getattr(m, "model", None)
            or type(m).__name__)
    if key:
        return f"{base}|{hashlib.sha256(key.encode()).hexdigest()[:12]}"
    return f"{base}|{name}"


def pacer_for(model: Any) -> Optional[TokenPacer]:
    """The pacer of ``model``'s key, or ``None`` when pacing is off."""
    limit = tokens_per_minute()
    if limit <= 0:
        return None
    key = model_key(model)
    with _PACERS_LOCK:
        pacer = _PACERS.get(key)
        if pacer is None or pacer.limit != limit:
            pacer = _PACERS[key] = TokenPacer(limit)
        return pacer


def _content_chars(content: Any) -> tuple:
    """``(characters, images)`` of a message's content."""
    if isinstance(content, str):
        return len(content), 0
    chars = images = 0
    if isinstance(content, list):
        for b in content:
            if isinstance(b, str):
                chars += len(b)
            elif isinstance(b, dict):
                if b.get("type") in ("image", "image_url", "input_image"):
                    images += 1
                else:
                    chars += len(str(b.get("text") or ""))
    return chars, images


def estimate_tokens(messages: Iterable[Any], tools: Iterable[Any] = (),
                    system: Any = None) -> int:
    """A request's tokens: its text over :data:`CHARS_PER_TOKEN`, each tool
    schema likewise, :data:`IMAGE_TOKENS` an image."""
    chars = images = 0
    for m in list(messages or ()) + ([system] if system is not None else []):
        content = m.get("content") if isinstance(m, dict) \
            else getattr(m, "content", m)
        c, i = _content_chars(content)
        chars += c
        images += i
        for call in (getattr(m, "tool_calls", None) or []):
            chars += len(str(call.get("args") if isinstance(call, dict)
                             else call))
    for t in tools or ():
        try:
            import json
            chars += len(json.dumps(
                t if isinstance(t, dict)
                else {"name": getattr(t, "name", ""),
                      "description": getattr(t, "description", ""),
                      "args": getattr(t, "args", {})}, default=str))
        except Exception:  # noqa: BLE001
            chars += 500
    return chars // CHARS_PER_TOKEN + images * IMAGE_TOKENS


def usage_tokens(message: Any) -> Optional[int]:
    """Input plus output tokens the provider counted for a reply, if it
    said."""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if isinstance(total, (int, float)) and total > 0:
            return int(total)
        n = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
        return int(n) if n > 0 else None
    return None


def _event(name: str, data: dict) -> None:
    try:
        from langchain_core.callbacks.manager import dispatch_custom_event
        dispatch_custom_event(name, data)
    except Exception:  # noqa: BLE001 - logging never fails a call
        pass


def pace(model: Any, tokens: int, what: str = "agent") -> tuple:
    """Wait for room for ``tokens`` on ``model``'s key; ``(pacer, ticket)``
    to settle with the counted usage, or ``(None, None)`` when pacing is
    off. A wait of half a second or more is logged (``pacer_wait``)."""
    pacer = pacer_for(model)
    if pacer is None:
        return None, None
    ticket, waited = pacer.acquire(tokens)
    if waited >= 0.5:
        _event("pacer_wait", {"call": what, "waited_s": round(waited, 3),
                              "tokens": int(tokens), "limit": pacer.limit})
    return pacer, ticket


__all__ = ["TokenPacer", "pacer_for", "pace", "model_key", "estimate_tokens",
           "usage_tokens", "tokens_per_minute", "TPM_ENV", "WINDOW_S",
           "IMAGE_TOKENS", "CHARS_PER_TOKEN"]
