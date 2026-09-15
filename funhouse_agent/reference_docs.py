"""Where the agent gets a reference PDF it needs to look at.

``read_reference_figure`` and ``view_worked_example_source`` render pages of
published PDFs that are not in the wheel. The local folder named by
``GEOTECH_REFERENCES_DOCS`` (or a source checkout's ``docs/``) is tried first.
When the PDF is not there, a fetcher registered by the host may supply it: the
web app registers one that downloads the file from the team SharePoint the
first time it is needed and keeps it in a local cache for the rest of the
session (owner decision 2026-09-15: fetch on demand, never at launch; the
session had just measured a 23 MB SharePoint download at about 1.3 s).

A fetcher is ``fn(filename, dest_path) -> path or None`` and may carry a
``description`` attribute ("SharePoint primary_references/") used in messages.
"""

from __future__ import annotations

import os
import tempfile
import threading
from typing import Callable, Optional

#: Env override for the local cache of fetched reference PDFs.
CACHE_ENV = "GEOTECH_REFERENCES_CACHE"

_FETCHER: Optional[Callable[[str, str], Optional[str]]] = None
_LOCK = threading.Lock()


def register_fetcher(fn: Optional[Callable[[str, str], Optional[str]]]) -> None:
    """Install (or, with ``None``, remove) the fallback source for PDFs."""
    global _FETCHER
    _FETCHER = fn


def fetcher_description() -> Optional[str]:
    return getattr(_FETCHER, "description", None) if _FETCHER else None


def cache_dir() -> str:
    """Local folder fetched PDFs are kept in."""
    return (os.environ.get(CACHE_ENV, "").strip()
            or os.path.join(tempfile.gettempdir(), "geotech_reference_pdfs"))


def _cached(name: str) -> Optional[str]:
    path = os.path.join(cache_dir(), name)
    return path if os.path.isfile(path) and os.path.getsize(path) > 0 else None


def fetch(filename: str) -> Optional[str]:
    """Local path of reference PDF ``filename`` from the cache or the
    registered fetcher, else ``None``. Never raises."""
    name = os.path.basename(str(filename or "")).strip()
    if not name:
        return None
    hit = _cached(name)
    if hit or _FETCHER is None:
        return hit
    with _LOCK:  # one download per file; a waiting caller finds it cached
        hit = _cached(name)
        if hit:
            return hit
        try:
            os.makedirs(cache_dir(), exist_ok=True)
            path = _FETCHER(name, os.path.join(cache_dir(), name))
        except Exception:  # noqa: BLE001
            return None
    return path if path and os.path.isfile(path) else None


def missing_message(original: str, filename: str) -> str:
    """The not-found error, saying where else was tried."""
    desc = fetcher_description()
    name = os.path.basename(str(filename or ""))
    if desc:
        return (f"{original} It is not available from {desc} either (looked "
                f"for '{name}'; the name must match exactly).")
    return original


__all__ = ["register_fetcher", "fetch", "cache_dir", "fetcher_description",
           "missing_message", "CACHE_ENV"]
