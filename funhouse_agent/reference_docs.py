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
``description`` attribute ("SharePoint primary_references/") used in messages
and an ``available`` attribute, ``fn() -> set of file names or None``, that
lists what it can supply. What the helpers are told they can read rests on
evidence (2026-11-11 reference review, closure f): a fetcher that lists is
asked once a session, and a PDF counts as fetchable only when its name is in
that listing (a failed or empty listing promises nothing); a fetcher that
cannot list is trusted until its first failed fetch, which withdraws the
promise for the rest of the session. A name whose fetch failed is never
promised again that session.
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

#: The registered fetcher's one listing this session: ``(fetcher, names)``,
#: names a frozenset or None where the listing failed.
_LISTING: Optional[tuple] = None
_LIST_LOCK = threading.Lock()
#: Names whose fetch failed this session; never promised again.
_FAILED: set = set()
#: A fetcher that cannot list has failed a fetch this session.
_WITHDRAWN = False


def register_fetcher(fn: Optional[Callable[[str, str], Optional[str]]]) -> None:
    """Install (or, with ``None``, remove) the fallback source for PDFs.
    Registering a different fetcher forgets the last one's listing and
    failures; registering the same one again keeps them."""
    global _FETCHER, _LISTING, _WITHDRAWN
    if fn is not _FETCHER:
        _LISTING = None
        _FAILED.clear()
        _WITHDRAWN = False
    _FETCHER = fn


def _listing() -> Optional[frozenset]:
    """The names the registered fetcher lists, asked once a session; None
    when there is no fetcher, it cannot list, or the listing failed."""
    global _LISTING
    fetcher = _FETCHER
    lister = getattr(fetcher, "available", None) if fetcher else None
    if lister is None:
        return None
    with _LIST_LOCK:
        if _LISTING is not None and _LISTING[0] is fetcher:
            return _LISTING[1]
        try:
            names = lister()
            names = None if names is None else frozenset(
                os.path.basename(str(n)) for n in names)
        except Exception:  # noqa: BLE001 - a failed listing promises nothing
            names = None
        _LISTING = (fetcher, names)
        return names


def _fetcher_offers(name: str) -> bool:
    """Whether the registered fetcher is shown to hold PDF ``name``: in its
    listing (a fetcher that lists), or not yet failed this session (one
    that cannot); never a name whose fetch failed."""
    if _FETCHER is None or not name or name in _FAILED:
        return False
    if getattr(_FETCHER, "available", None) is None:
        return not _WITHDRAWN
    names = _listing()
    return bool(names) and name in names


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
            path = None
        if path and os.path.isfile(path):
            return path
        _failed(name)
    return None


def _failed(name: str) -> None:
    """Record a failed fetch: the name is not promised again this session,
    and a fetcher that cannot list promises nothing more."""
    global _WITHDRAWN
    _FAILED.add(name)
    if getattr(_FETCHER, "available", None) is None:
        _WITHDRAWN = True


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
    (its docs folder, ``GEOTECH_REFERENCES_DOCS`` or a pack's folder), a
    fetched copy is in the cache, or the registered fetcher is shown to
    hold it (:func:`_fetcher_offers`). Never fetches."""
    try:
        from geotech_references import _figures_db
        from funhouse_agent.adapters._reference_common import takes
    except Exception:  # noqa: BLE001
        return False

    def _page(func) -> dict:
        # a library before the 2026-11-10 review fetches by number only
        if pdf_page_index in (None, "") or not takes(func, "pdf_page_index"):
            return {}
        return {"pdf_page_index": int(pdf_page_index)}

    try:
        _figures_db.resolve_pdf(reference, figure_number,
                                **_page(_figures_db.resolve_pdf))
        return True
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001 - an unknown figure is not readable
        return False
    if not pdf_path:
        try:
            pdf_path = _figures_db.figure_get(
                reference, figure_number, **_page(_figures_db.figure_get)
            ).get("pdf_path", "")
        except Exception:  # noqa: BLE001
            return False
    name = os.path.basename(pdf_path or "")
    return bool(name) and (_cached(name) is not None
                           or _fetcher_offers(name))


def chart_reads_available() -> bool:
    """Whether a value can be read off a reference chart on this host: the
    source PDF of at least one figure catalog is found where the library's
    ``resolve_pdf`` looks (its docs folder, ``GEOTECH_REFERENCES_DOCS`` or
    a reference pack's folder), is in the cache of fetched PDFs, or is
    shown to be held by the registered fetcher (the web app's SharePoint
    ``primary_references/``: its name is in the folder's listing; see the
    module docstring). Where not, no text the helpers read promises a
    chart read and ``read_reference_figure`` is not offered (2026-11-10
    reference review: the wheel ships no PDFs and Foundry has no fetcher,
    so every read there failed after the prompt had promised it; closure
    f: a registered fetcher alone no longer counts as evidence)."""
    if _PINNED is not None:
        return _PINNED
    return any(_resolves(*row) for row in _catalog_pdfs())


def figure_readable(reference: str, figure_number: str,
                    pdf_page_index=None) -> bool:
    """Whether this one figure can be read on this host (see
    :func:`chart_reads_available`): its own PDF must be found, cached or
    shown to be held by the registered fetcher."""
    if _PINNED is not None:
        return _PINNED
    return _resolves(reference, figure_number, pdf_page_index)


def worked_example_pages_available() -> bool:
    """Whether a worked example's printed source page can be shown on this
    host (``view_worked_example_source``), by the same rule as
    :func:`chart_reads_available`: pinned; or the source PDF of at least
    one worked example with catalogued pages is found where the corpus
    looks (``GEOTECH_REFERENCES_DOCS``, a checkout's
    ``geotech-references/docs``), is in the cache of fetched PDFs, or is
    shown to be held by the registered fetcher."""
    if _PINNED is not None:
        return _PINNED
    try:
        from funhouse_agent import worked_examples as _we
        entries = [e for e in _we.load_examples()
                   if e.get("source_pdf_pages") and e.get("source_doc")]
    except Exception:  # noqa: BLE001
        return False
    seen = set()
    for entry in entries:
        doc = str(entry["source_doc"])
        if doc in seen:
            continue
        seen.add(doc)
        try:
            _we.resolve_source_pdf(entry)
            return True
        except Exception:  # noqa: BLE001 - absent here; maybe fetched
            name = os.path.basename(doc)
            if _cached(name) or _fetcher_offers(name):
                return True
    return False


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
    if not desc:
        return original
    names = _listing()
    if names is not None and name not in names:
        return (f"{original} '{name}' is not in {desc} either, so this "
                "PDF cannot be shown or read on this host.")
    return (f"{original} It is not available from {desc} either (looked "
            f"for '{name}'; the name must match exactly).")


__all__ = ["register_fetcher", "fetch", "cache_dir", "fetcher_description",
           "missing_message", "CACHE_ENV", "chart_reads_available",
           "figure_readable", "pin_chart_reads", "NO_CHART_READ",
           "worked_example_pages_available"]
