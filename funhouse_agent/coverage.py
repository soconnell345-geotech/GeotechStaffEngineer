"""Which pages of a document a tool call read or looked at, from the call
itself (its arguments and its result), never from what the model says it did.

:func:`pages_from_call` is what :mod:`funhouse_agent.cited_pages` (the pages
an answer cites, checked against the pages opened) and the review suite's
checks (:mod:`funhouse_agent.review_eval.checks`) read a conversation's tool
calls with; :func:`parse_pages` and :func:`compact` read and write page lists.

(The coverage ledger, the ``document_coverage`` tool and the gate behind
``GEOTECH_COVERAGE`` were removed in BACKLOG DR-40: run 7 measured 39/42 with
and without it, at +47 % input, and it was never re-measured;
``module_work/dr40/CHECK.md``. This module keeps the page parsers they
shared.)

"Read" means: looked at by a vision tool, or read as text by a text tool
(``log_grid`` counts as a text read of each page whose rows it returned in
full).

Pages are 0-based here, as in every tool; ``pdf_pages`` (one higher) are what
a reader cites.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, List, Optional, Tuple

#: Tools whose results say which pages were READ AS TEXT.
TEXT_TOOLS = ("read_document", "read_pdf_text")
#: Tools that LOOK at a page (whole or zoomed).
LOOK_TOOLS = ("analyze_pdf_page", "render_region")
#: A tool that read or looked at each page of a range itself (removed in
#: DR-39; kept so conversations recorded before it still parse).
SWEEP_TOOLS = ("sweep_pages",)
#: Tools that return a log's printed content row by row, each row naming its
#: page (planlens' ``log_grid``): a page whose rows all came back was read,
#: as text. Foundry brief 5 (N6).
GRID_TOOLS = ("log_grid",)
#: Not reads, by design: ``measure`` returns a position measured through the
#: page's scale, not the page's content - like a search, it answers one
#: question about a page; the look or the text read that found the thing is
#: the read. Nor are searches, thumbnails, ``find_like`` or
#: ``find_quantities``.


# ---------------------------------------------------------------------------
# Page lists
# ---------------------------------------------------------------------------

def parse_pages(spec: Any, n_pages: Optional[int] = None) -> List[int]:
    """0-based pages from an int, a list, a string like ``"2-4,7"``, or a
    dict of those (``{"logs": "7-10", "lab": [16, 17]}``, or
    ``{"pages": [...]}``: every value is read). Anything unreadable is
    dropped; ``n_pages`` bounds the result."""
    out: List[int] = []
    if spec is None or spec == "":
        return out
    if isinstance(spec, bool):
        return out
    if isinstance(spec, int):
        out = [spec]
    elif isinstance(spec, dict):
        for v in spec.values():
            out.extend(parse_pages(v, n_pages))
    elif isinstance(spec, (list, tuple, set)):
        for v in spec:
            out.extend(parse_pages(v, n_pages))
    else:
        for part in str(spec).replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"^(\d+)\s*(?:-|–|to)\s*(\d+)$", part)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                if a <= b and b - a <= 5000:
                    out.extend(range(a, b + 1))
            elif part.isdigit():
                out.append(int(part))
    if n_pages is not None:
        out = [p for p in out if 0 <= p < n_pages]
    return sorted(set(p for p in out if p >= 0))


def compact(pages: Iterable[int], offset: int = 0) -> str:
    """``[4,5,6,9]`` as ``"4-6,9"`` (``offset=1`` for viewer pages)."""
    ps = sorted(set(int(p) + offset for p in pages))
    if not ps:
        return ""
    out: List[str] = []
    start = prev = ps[0]
    for p in ps[1:]:
        if p == prev + 1:
            prev = p
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = p
    out.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(out)


# ---------------------------------------------------------------------------
# What a tool call read
# ---------------------------------------------------------------------------

def _json(result: Any) -> Optional[dict]:
    if isinstance(result, dict):
        return result
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def pages_from_call(name: str, args: Optional[dict], result: Any
                    ) -> Tuple[Optional[Tuple[str, str]],
                               List[Tuple[int, str]]]:
    """``(source_ref, [(page, how), ...])`` for one tool call.

    ``source_ref`` is ``("handle", h)`` or ``("source", key_or_path)``: how
    the call named its document. ``how`` is ``"text"`` (the result carried
    that page's text), ``"text_empty"`` (a text tool reached the page and it
    had no text layer), ``"look"`` (a vision tool looked at it) or
    ``"sweep"`` (a sweep asked about it, by text or by looking). Pages come
    from the RESULT where the result says which pages it returned (a read cut
    short by its size limit did not read the rest), from the arguments where
    a successful look is about one page. A call that failed read nothing.
    """
    args = args or {}
    data = _json(result)
    if data is not None and data.get("error"):
        return None, []
    if name == "read_document":
        if data is None:
            return None, []
        ref = ("handle", str(args.get("handle") or data.get("handle") or ""))
        return ref, [(p, "text") for p in
                     parse_pages(data.get("pages_returned"))]
    if name == "read_pdf_text":
        if data is None:
            return None, []
        src = str(args.get("source") or args.get("attachment_key")
                  or args.get("path") or data.get("source") or "")
        out = []
        for row in data.get("pages") or []:
            if not isinstance(row, dict) or not isinstance(row.get("page"),
                                                           int):
                continue
            how = "text" if row.get("has_text_layer") else "text_empty"
            out.append((row["page"], how))
        return ("source", src), out
    if name in LOOK_TOOLS:
        if result is None or (isinstance(result, str) and not result.strip()):
            return None, []
        src = str(args.get("attachment_key") or args.get("source") or "")
        # The page the tool READ (its result names it), else the one asked
        # for -- a 1-based pdf_page included (live smoke wave 2c).
        page = data.get("page") if data is not None else None
        if not isinstance(page, int) or isinstance(page, bool):
            try:
                pdf_page = args.get("pdf_page")
                page = (int(pdf_page) - 1 if pdf_page not in (None, "")
                        else int(args.get("page", 0) or 0))
            except (TypeError, ValueError):
                return None, []
        return ("source", src), [(page, "look")]
    if name in SWEEP_TOOLS:
        if data is None:
            return None, []
        src = str(args.get("source") or "")
        checked = set(parse_pages(data.get("pages_checked")))
        failed = {r.get("page") for r in data.get("unanswered") or []
                  if isinstance(r, dict)}
        return ("source", src), [(p, "sweep") for p in sorted(checked - failed)]
    if name in GRID_TOOLS:
        if data is None:
            return None, []
        ref = ("handle", str(args.get("handle") or data.get("handle") or ""))
        return ref, [(p, "text") for p in _grid_pages_returned(args, data)]
    return None, []


def _grid_pages_returned(args: dict, data: dict) -> List[int]:
    """The pages whose rows a ``log_grid`` result has returned IN FULL.

    The rows come a window at a time (``offset`` in, ``next_offset`` out),
    in page order, each naming its page. A page is complete in this window
    when a later page's rows follow it, or when this is the last window
    (no ``next_offset``); the last page of a window that goes on may go on
    in the next. A window after the first (``offset`` > 0) also completes
    the pages of the call that come before its first row's page: their rows
    were the earlier windows' (a page whose rows ended exactly at a window's
    end shows only here). A call with no rows (``rows=false``, a scan with
    no text) returns no page's content and reads nothing."""
    rows = data.get("rows")
    if not isinstance(rows, list):
        return []
    order: List[int] = []
    for row in rows:
        page = row.get("page") if isinstance(row, dict) else None
        if isinstance(page, int) and not isinstance(page, bool) \
                and (not order or order[-1] != page):
            order.append(page)
    if not order:
        return []
    more = data.get("next_offset") not in (None, "", False)
    done = order[:-1] if more else list(order)
    try:
        offset = int(args.get("offset") or data.get("offset") or 0)
    except (TypeError, ValueError):
        offset = 0
    if offset > 0:
        listed = parse_pages(data.get("pages"))
        if order[0] in listed:
            done.extend(listed[:listed.index(order[0])])
    return sorted(set(done))


# ---------------------------------------------------------------------------
# The inventory
# ---------------------------------------------------------------------------


__all__ = ["TEXT_TOOLS", "LOOK_TOOLS", "SWEEP_TOOLS", "GRID_TOOLS",
           "parse_pages", "compact", "pages_from_call"]
