"""Rendered page images waiting to be shown to the main model.

In the looking-only ``minimal`` build, ``analyze_pdf_page``
and ``render_region`` (and ``analyze_image`` of an image file, such as a
contact sheet) do not describe an image through a separate one-shot
vision call; they store it here and return its ``image_id``, and
:class:`funhouse_agent.deep.inline_images.InlineImageMiddleware` shows the
newest images to the reasoning model itself at its next call.

Inside a turn (:func:`aec_agent_core.deep.keep_work.turn_scope`) the images
are the TURN's own: held on its scope, at most :data:`MAX_IMAGES`, dropped
when the turn ends, so no other conversation's images can push one out
(a host runs several testers in one process). Outside a turn (a library
call, a probe) they wait in one process-wide store with the same bound.
Either holds bytes only and is never persisted: a replayed conversation
keeps the text of every result and simply no longer shows an image it no
longer has. The same bytes with the same description are one stored image.
An image pushed out by its own turn's later images leaves its description
behind (:func:`pushed_out`), so the model can be told it was not shown.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

#: Most images held at once (a turn rarely shows more than a few dozen).
MAX_IMAGES = 96

#: How many of the newest images each model call carries (the middleware's
#: ``keep``). Each is a full-budget render, and a request body has a size
#: limit on some gateways, so this is the knob to turn down (to 1) if a call
#: is refused as too large. Kept here, beside the store, so the tools that
#: store an image can tell the model how long it stays in view.
DEFAULT_KEEP = 2
KEEP_ENV = "GEOTECH_VISION_INLINE_KEEP"


def keep_from_env() -> int:
    """``GEOTECH_VISION_INLINE_KEEP`` (default 2, at least 1)."""
    try:
        return max(1, int(os.environ.get(KEEP_ENV, DEFAULT_KEEP)))
    except (TypeError, ValueError):
        return DEFAULT_KEEP

_LOCK = threading.Lock()
_STORE: "OrderedDict[str, Tuple[bytes, Dict[str, Any]]]" = OrderedDict()
#: The descriptions of the images pushed out of the process-wide store.
_GONE: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
#: The attribute of a turn's scope that holds its ``(images, pushed out)``.
SCOPE_ATTR = "inline_images"


def _shelf(make: bool = True):
    """``(images, pushed out)`` of the running turn, or the process-wide
    pair outside one. Called with ``_LOCK`` held."""
    try:
        from aec_agent_core.deep.keep_work import current_turn
        scope = current_turn()
    except Exception:  # noqa: BLE001 - no agent library: no turn
        scope = None
    if scope is None:
        return _STORE, _GONE
    if make:
        return scope.__dict__.setdefault(SCOPE_ATTR,
                                         (OrderedDict(), OrderedDict()))
    return scope.__dict__.get(SCOPE_ATTR) or ({}, {})


def put(data: bytes, meta: Optional[Dict[str, Any]] = None) -> str:
    """Keep ``data`` (image bytes) and return its id: the id it already has
    where the same bytes with the same ``meta`` are stored."""
    data, meta = bytes(data), dict(meta or {})
    with _LOCK:
        store, gone = _shelf()
        for known, (old, old_meta) in store.items():
            if old == data and \
                    {**old_meta, "shown": None} == {**meta, "shown": None}:
                store.move_to_end(known)
                return known
        image_id = "img_" + uuid.uuid4().hex[:12]
        store[image_id] = (data, meta)
        while len(store) > MAX_IMAGES:
            lost, (_, lost_meta) = store.popitem(last=False)
            gone[lost] = lost_meta
            while len(gone) > 4 * MAX_IMAGES:
                gone.popitem(last=False)
    return image_id


def get(image_id: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
    """``(bytes, meta)`` for ``image_id`` (the running turn's, or one stored
    outside a turn), or ``None`` once it has fallen out."""
    with _LOCK:
        return _shelf(False)[0].get(str(image_id)) or _STORE.get(str(image_id))


def pushed_out(image_id: str) -> Optional[Dict[str, Any]]:
    """The description of ``image_id`` when later images pushed it out of
    its store (its bytes are gone), else ``None``."""
    with _LOCK:
        return _shelf(False)[1].get(str(image_id)) or _GONE.get(str(image_id))


def clear() -> None:
    """Drop every stored image: the running turn's and the process-wide
    store's (tests)."""
    with _LOCK:
        for part in (*_shelf(False), _STORE, _GONE):
            part.clear()


__all__ = ["put", "get", "pushed_out", "clear", "MAX_IMAGES", "DEFAULT_KEEP", "KEEP_ENV",
           "keep_from_env"]
