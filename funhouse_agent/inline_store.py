"""Rendered page images waiting to be shown to the main model.

With ``GEOTECH_VISION_INLINE`` on (lean review agent only), ``analyze_pdf_page``
and ``render_region`` do not describe an image through a separate one-shot
vision call; they store it here and return its ``image_id``, and
:class:`funhouse_agent.deep.inline_images.InlineImageMiddleware` shows the
newest images to the reasoning model itself at its next call.

The store is process-wide and bounded (the oldest images fall out), holds
bytes only, and is never persisted: a replayed conversation keeps the text of
every result and simply no longer shows an image it no longer has.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

#: Most images held at once (a turn rarely shows more than a few dozen).
MAX_IMAGES = 96

_LOCK = threading.Lock()
_STORE: "OrderedDict[str, Tuple[bytes, Dict[str, Any]]]" = OrderedDict()


def put(data: bytes, meta: Optional[Dict[str, Any]] = None) -> str:
    """Keep ``data`` (image bytes) and return its id."""
    image_id = "img_" + uuid.uuid4().hex[:12]
    with _LOCK:
        _STORE[image_id] = (bytes(data), dict(meta or {}))
        while len(_STORE) > MAX_IMAGES:
            _STORE.popitem(last=False)
    return image_id


def get(image_id: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
    """``(bytes, meta)`` for ``image_id``, or ``None`` once it has fallen out."""
    with _LOCK:
        return _STORE.get(str(image_id))


def clear() -> None:
    """Drop every stored image (tests)."""
    with _LOCK:
        _STORE.clear()


__all__ = ["put", "get", "clear", "MAX_IMAGES"]
