"""A look at an image waits while a figure is being drawn in its folder.

Live wave w8 (2026-10-15, `module_work/live_smoke/runs/w8-plot-code/
TRACE_REVIEW.md` §2.3): the figure helper redrew a figure and looked at it in
the same step; the two calls ran side by side and the look read the PNG from
before the redraw, so a fault it had just fixed was reported again (twice
more in wave w7b with ``plot_data``). Drawing tools (``plot_code``,
``plot_data``) hold :func:`drawing` on the conversation folder while they
write; ``analyze_image`` on an image file there calls :func:`wait_until_drawn`
first.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from typing import Dict, Iterator, Optional

_COND = threading.Condition()
_ACTIVE: Dict[str, int] = {}

#: The longest a look waits for drawings in its folder (seconds).
DEFAULT_WAIT_S = 180.0


def _key(folder: Optional[str]) -> str:
    return os.path.normcase(os.path.abspath(folder)) if folder else ""


@contextlib.contextmanager
def drawing(folder: Optional[str]) -> Iterator[None]:
    """Mark a drawing in progress in ``folder`` for the block."""
    key = _key(folder)
    with _COND:
        _ACTIVE[key] = _ACTIVE.get(key, 0) + 1
    try:
        yield
    finally:
        with _COND:
            _ACTIVE[key] -= 1
            if _ACTIVE[key] <= 0:
                del _ACTIVE[key]
            _COND.notify_all()


def drawing_in(folder: Optional[str]) -> bool:
    with _COND:
        return _ACTIVE.get(_key(folder), 0) > 0


def wait_until_drawn(folder: Optional[str],
                     timeout: float = DEFAULT_WAIT_S) -> float:
    """Wait (up to ``timeout``) until no drawing runs in ``folder``; the
    seconds waited."""
    key = _key(folder)
    t0 = time.monotonic()
    with _COND:
        _COND.wait_for(lambda: _ACTIVE.get(key, 0) <= 0, timeout=timeout)
    return time.monotonic() - t0


__all__ = ["drawing", "drawing_in", "wait_until_drawn", "DEFAULT_WAIT_S"]
