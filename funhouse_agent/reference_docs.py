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


#: Tests and page fingerprints pin the answer (``pin_chart_reads``): True
#: as on a host with the PDFs, False as on one without, None to look.
_PINNED: Optional[bool] = None


def pin_chart_reads(value: Optional[bool]) -> None:
    """Pin :func:`chart_reads_available` to ``value`` (None: look again)."""
    global _PINNED
    _PINNED = None if value is None else bool(value)


def _catalog_pdfs() -> list:
    """One figure catalog row for each source PDF the catalogs name:
    ``(reference, figure_number, pdf_page_index, pdf_path)``."""
    try:
        from geotech_references import _figures_db
        conn = _figures_db._ro_connect()
    except Exception:  # noqa: BLE001
        return []
    try:
        rows = conn.execute(
            "SELECT reference, figure_number, pdf_page_index, pdf_path "
            "FROM figures GROUP BY reference, pdf_path").fetchall()
        return [tuple(r) for r in rows]
    except Exception:  # noqa: BLE001
        return []
    finally:
        conn.close()


def _resolves(reference: str, figure_number: str, pdf_page_index=None,
              pdf_path: str = "") -> bool:
    """Whether the library's own ``resolve_pdf`` finds this figure's PDF
    (its docs folder, ``GEOTECH_REFERENCES_DOCS`` or a pack's folder), or
    a fetched copy is in the cache. Never fetches."""
    try:
        from geotech_references import _figures_db
        from funhouse_agent.adapters._reference_common import takes
        kw = {}
        if pdf_page_index not in (None, "") and takes(_figures_db.resolve_pdf,
                                                      "pdf_page_index"):
            kw["pdf_page_index"] = int(pdf_page_index)
        _figures_db.resolve_pdf(reference, figure_number, **kw)
        return True
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001 - an unknown figure is not readable
        return False
    if not pdf_path:
        try:
            from geotech_references import _figures_db
            pdf_path = _figures_db.figure_get(
                reference, figure_number,
                **({"pdf_page_index": int(pdf_page_index)}
                   if pdf_page_index not in (None, "") else {})
            ).get("pdf_path", "")
        except Exception:  # noqa: BLE001
            return False
    return bool(pdf_path) and _cached(os.path.basename(pdf_path)) is not None


def chart_reads_available() -> bool:
    """Whether a value can be read off a reference chart on this host: a
    fetcher is registered (the web app's SharePoint ``primary_references/``),
    or the source PDF of at least one figure catalog is found where the
    library's ``resolve_pdf`` looks (its docs folder,
    ``GEOTECH_REFERENCES_DOCS`` or a reference pack's folder) or is in the
    cache of fetched PDFs. Where not, no text the helpers read promises a
    chart read and ``read_reference_figure`` is not offered (2026-11-10
    reference review: the wheel ships no PDFs and Foundry has no fetcher,
    so every read there failed after the prompt had promised it)."""
    if _PINNED is not None:
        return _PINNED
    if _FETCHER is not None:
        return True
    return any(_resolves(*row) for row in _catalog_pdfs())


def figure_readable(reference: str, figure_number: str,
                    pdf_page_index=None) -> bool:
    """Whether this one figure can be read on this host (see
    :func:`chart_reads_available`): a fetcher may supply any PDF; else its
    own PDF must be found or cached."""
    if _PINNED is not None:
        return _PINNED
    if _FETCHER is not None:
        return True
    return _resolves(reference, figure_number, pdf_page_index)


#: What a figure_search hit says where its chart cannot be read here.
NO_CHART_READ = (
    "The source PDF for this figure is not on this host, so no value can "
    "be read off it here: this hit identifies the figure by its number, "
    "caption and page only. Do NOT report a value off this chart from the "
    "caption or from memory; say that the chart could not be read.")


def missing_message(original: str, filename: str) -> str:
    """The not-found error, saying where else was tried."""
    desc = fetcher_description()
    name = os.path.basename(str(filename or ""))
    if desc:
        return (f"{original} It is not available from {desc} either (looked "
                f"for '{name}'; the name must match exactly).")
    return original


__all__ = ["register_fetcher", "fetch", "cache_dir", "fetcher_description",
           "missing_message", "CACHE_ENV", "chart_reads_available",
           "figure_readable", "pin_chart_reads", "NO_CHART_READ"]
