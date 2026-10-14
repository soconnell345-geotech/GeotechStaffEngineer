"""The HTTP tries inside one model call, for the activity log (BACKLOG EV-14).

Field session 2026-10-14: a refusal reached the log as one exception text;
the client's own retries happen inside one model call, so the record could
not say how many HTTP requests a call made, what each was answered, or what
the rate-limit headers said.

:func:`note_response` is an httpx response hook (the Prompter key route's
client installs it, ``webapp.tinyapps_engine``): it keeps each response's
status, ``retry-after`` and ``x-ratelimit-*`` headers for the thread that
made it. The activity log starts a list when a model call starts and takes
it when the call ends or fails (a synchronous call's callbacks and its HTTP
requests run on one thread). The body is never read here.
"""

from __future__ import annotations

import threading
from typing import List, Optional

_LOCAL = threading.local()

#: Responses kept per call at most (a call never makes more than a few).
MAX_KEPT = 20


def begin() -> None:
    """Start the list for the model call starting on this thread."""
    _LOCAL.tries = []


def note_response(response) -> None:
    """httpx response hook. Never raises, never reads the body."""
    try:
        tries = getattr(_LOCAL, "tries", None)
        if tries is None or len(tries) >= MAX_KEPT:
            return
        one = {"status": int(response.status_code)}
        headers = getattr(response, "headers", None) or {}
        for k, v in headers.items():
            lk = str(k).lower()
            if lk.startswith("x-ratelimit") or lk in ("retry-after",
                                                      "retry-after-ms"):
                one.setdefault("headers", {})[lk] = str(v)
        tries.append(one)
    except Exception:  # noqa: BLE001 - the record never costs a call
        pass


def take() -> Optional[List[dict]]:
    """The responses of the call that ran on this thread (``None`` when
    none was started), and the list is cleared."""
    tries = getattr(_LOCAL, "tries", None)
    _LOCAL.tries = None
    return tries


__all__ = ["begin", "note_response", "take", "MAX_KEPT"]
