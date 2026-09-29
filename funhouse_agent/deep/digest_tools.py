"""Digest tools on the lean review agent (``GEOTECH_REVIEW_DIGEST=1``).

WHY. A review of hundreds of pages, or of several PDFs, cannot be carried by
one agent loop: every page the reviewer reads stays in its context. These
tools let it hold SUMMARIES and pull DETAIL (plan of record
``module_work/REVIEW_ARCHITECTURE.md``, shape 2): the free layer of each
upload's digest (:mod:`funhouse_agent.review_digest`) - built from the PDF's
own text in seconds the first time it is asked for, saved in the
conversation's working folder and reused in later turns.

WHAT. Four tools, all reading only (they write nothing but the digest cache,
never a deliverable or a markup), so the reading helper gets them too:

* ``document_inventory`` - every upload at once: pages, page kinds, the
  pages to look at, printed page ranges, outline, sheet labels, markups, a
  small/large hint, and the sheets or standards cited that no upload holds;
* ``digest_search`` - exact-text search over one, some or all uploads;
* ``digest_pages`` - one compact row per page;
* ``digest_references`` - sheet, detail, standard, specification-section,
  table, figure and appendix references, with where each is cited and where
  a table or figure is captioned.

Every result is valid JSON under :data:`RESULT_CHARS`, strongest first; a
list that does not fit says how many were left out and gives ``next``.
Every page carries ``pdf_page`` (the number a viewer shows).
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from funhouse_agent import document_tools

#: The tools this module adds, in the order they are offered.
DIGEST_TOOLS = ("document_inventory", "digest_search", "digest_pages",
                "digest_references")

#: Every result stays under this many characters.
RESULT_CHARS = 11500

FREE_LAYER = ("Free layer: exact text only, no reading of pictures - a page "
              "flagged needs_look (drawing sheet, figure, scan, form, or "
              "text that is not what the page shows) must still be looked "
              "at.")

INVENTORY_DESCRIPTION = (
    "Take stock of all the uploaded documents at once, from their digests "
    "(built from each PDF's own text in seconds on first use and kept for "
    "later turns): pages, what kind each page is, which pages need looking "
    "at, printed page ranges, the outline, sheet numbers, review markups, a "
    "small/large hint for planning the review, and the sheets or standards "
    "the documents cite that no upload contains. Start a large or "
    "several-document review here. documents limits it to some uploads. "
    + FREE_LAYER)
SEARCH_DESCRIPTION = (
    "Search the exact text of the uploads' digests for words or a phrase: "
    "each matching line with its document, page (0-based), pdf_page, the "
    "line and its box in PDF points, best matches first. documents limits "
    "it to some uploads; pages ('0-40,90', 0-based) and kinds ('text', "
    "'drawing_sheet', 'figure', 'form', 'scanned', 'mixed') narrow it; pass "
    "next's offset to continue. Words drawn as lines or inside a picture "
    "are not found. " + FREE_LAYER)
PAGES_DESCRIPTION = (
    "One compact row per page of a document, from its digest: kind, sheet "
    "or printed page number, heading, words, markups, whether its text is "
    "reliable, needs_look, the references on it and a short excerpt. Use it "
    "to see what a run of pages holds before reading them. source = the "
    "document as uploaded; pages like '0-9,40' (0-based; every row also "
    "gives pdf_page). " + FREE_LAYER)
REFERENCES_DESCRIPTION = (
    "Cross-references found in the documents' text: sheet and detail "
    "callouts ('3/S-501'), standards ('STD. NO. 10.17'), specification "
    "sections, tables, figures and appendices, each with where it is cited "
    "(page, pdf_page, box) and, for a table, figure or appendix, where its "
    "caption is and its title. target narrows to one ('S-501', 'Table 5-1', "
    "'10.17'; 'Table 12-' for every table numbered 12-...); source to one "
    "document. Without a target, sheets and standards no upload contains "
    "come first. " + FREE_LAYER)


# ---------------------------------------------------------------------------
# Sources and digests
# ---------------------------------------------------------------------------

def _resolve(names: Optional[List[str]], attachments: Dict[str, bytes]
             ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """``({name: bytes or path}, errors)`` for the named documents, or for
    every upload when none are named."""
    if not names:
        return dict(attachments), []
    out: Dict[str, Any] = {}
    errors = []
    for name in names:
        try:
            out[str(name)] = document_tools.resolve_document_source(
                str(name), attachments)
        except Exception as exc:  # noqa: BLE001 - planlens' ToolError and kin
            err = {"document": str(name),
                   "error": str(exc) or type(exc).__name__}
            hint = getattr(exc, "hint", None)
            if hint:
                err["hint"] = hint
            errors.append(err)
    return out, errors


def _digests(sources: Dict[str, Any], root: Optional[str] = None
             ) -> Tuple[Dict[str, Any], List[Dict]]:
    from funhouse_agent.review_digest import build
    got, errors = {}, []
    for name, src in sources.items():
        try:
            got[name] = build(src, name=name, root=root)
        except Exception as exc:  # noqa: BLE001 - one bad upload is reported
            errors.append({"document": name,
                           "error": f"{type(exc).__name__}: {exc}"[:300]})
    return got, errors


def _dump(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                      default=str)


def _fit(out: Dict[str, Any], key: str, next_of: Callable[[int], Any],
         limit: int = RESULT_CHARS) -> str:
    """``out`` as JSON with its ``key`` list cut from the end to fit
    ``limit``; ``left_out`` says how many were cut and ``next`` where to
    continue (``next_of(index of the first one left out)``)."""
    items = list(out.get(key) or [])
    n = len(items)

    def dump(k: int) -> str:
        o = dict(out)
        o[key] = items[:k]
        if k < n:
            o["left_out"] = n - k
            o["next"] = next_of(k)
        return _dump(o)

    text = dump(n)
    k = n
    while k > 1 and len(text) > limit:
        k = max(1, k - max(1, k // 6))
        text = dump(k)
    return text


def _error(message: str, **extra) -> str:
    out = {"error": message}
    out.update(extra)
    return _dump(out)


# ---------------------------------------------------------------------------
# document_inventory
# ---------------------------------------------------------------------------

def _doc_overview(summary: Dict[str, Any], tight: int) -> Dict[str, Any]:
    """One document's overview, shortened as ``tight`` rises (0 = all)."""
    s = dict(summary)
    s.pop("search", None)
    caps = {0: (80, 60, 20), 1: (40, 30, 8), 2: (15, 12, 4), 3: (0, 0, 0)}
    n_outline, n_sheets, n_segments = caps[min(tight, 3)]
    for key, cap in (("outline", n_outline), ("sheet_labels", n_sheets),
                     ("segments", n_segments)):
        rows = s.get(key) or []
        if len(rows) > cap:
            s[key] = rows[:cap]
            s[key + "_more"] = len(rows) - cap
        if not s.get(key):
            s.pop(key, None)
    if tight >= 2:
        s.pop("roles", None)
        nl = dict(s.get("needs_look") or {})
        nl.pop("pages", None)
        s["needs_look"] = nl
    return s


#: How many cited-but-not-uploaded sheets (and citations of each) an
#: overview lists, as it is shortened to fit.
_MISSING_CAPS = {0: (40, 3), 1: (25, 2), 2: (12, 1), 3: (6, 1)}


def _missing(xref: Dict[str, Any], tight: int) -> Dict[str, Any]:
    n_items, n_cited = _MISSING_CAPS[min(tight, 3)]
    out: Dict[str, Any] = {}
    missing = xref.get("missing") or []
    if missing:
        out["cited_but_not_uploaded"] = [
            {"id": m["id"], "kinds": m["kinds"], "n": m["n"],
             "cited": [{k: c[k] for k in ("document", "page", "pdf_page",
                                          "text") if k in c}
                       for c in m["cited"][:n_cited]]}
            for m in missing[:n_items]]
        if len(missing) > n_items:
            out["cited_but_not_uploaded_more"] = len(missing) - n_items
            out["cited_but_not_uploaded_note"] = (
                "digest_references lists every one")
    found = xref.get("found") or []
    if found:
        out["cited_and_found"] = found[:max(5, n_items // 2)]
    if xref.get("note"):
        out["cited_note"] = xref["note"]
    return out


def inventory_result(sources: Dict[str, Any], errors: List[Dict[str, Any]],
                     limit: int = RESULT_CHARS,
                     root: Optional[str] = None) -> str:
    from funhouse_agent.review_digest import inventory_of
    inv = inventory_of(sources, root=root)
    errors = list(errors) + list(inv.get("errors") or [])
    xref = inv.get("cross_references") or {}
    docs = inv["documents"]

    def overview(tight: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "totals": inv["totals"], "shape_hint": inv["shape_hint"],
            "shape_note": (f"'small' = {inv['shape1_pages']} pages or fewer "
                           f"in all: read and look at every page yourself; "
                           f"'large' = work from these digests - search, "
                           f"page rows, references - and look only at the "
                           f"pages that matter"),
        }
        out.update(_missing(xref, tight))
        if errors:
            out["errors"] = errors[:10]
            if len(errors) > 10:
                out["errors_more"] = len(errors) - 10
        out["pages_note"] = ("pages are 0-based as the tools number them; "
                             "pdf_pages are what a viewer shows - cite "
                             "those")
        out["documents"] = [_doc_overview(d, tight) for d in docs]
        return out

    for tight in range(4):
        out = overview(tight)
        text = _dump(out)
        if len(text) <= limit:
            return text
    names = [d.get("document") for d in docs]
    return _fit(out, "documents",
                lambda k: {"documents": names[k:],
                           "note": "call document_inventory with these"},
                limit)


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------

def _as_list(v: Any) -> Optional[List[str]]:
    if v is None or v == "" or v == []:
        return None
    if isinstance(v, str):
        return [p.strip() for p in v.split(",") if p.strip()]
    return [str(x) for x in v]


def make_digest_tools(attachments: Optional[Dict[str, bytes]] = None,
                      working_dir: Optional[str] = None) -> list:
    """``document_inventory``, ``digest_search``, ``digest_pages`` and
    ``digest_references`` bound to this agent's uploads (the live dict the
    host mutates). None of them writes anything but the digest cache.

    ``working_dir`` is the conversation's working folder, bound when the
    agent is built: the digests are kept in its ``digest/`` folder even if
    the process-wide working folder is pointed elsewhere meanwhile (another
    conversation in the same process). Without it the working folder is
    looked up at each call."""
    from langchain_core.tools import StructuredTool
    from funhouse_agent.review_digest import root_for

    attachments = {} if attachments is None else attachments
    root = root_for(working_dir)

    def document_inventory(documents: Optional[List[str]] = None) -> str:
        sources, errors = _resolve(_as_list(documents), attachments)
        if not sources:
            return _error("no documents to take stock of",
                          hint="nothing is uploaded yet"
                          if not errors else "check the names", errors=errors)
        try:
            return inventory_result(sources, errors, root=root)
        except Exception as exc:  # noqa: BLE001 - a result, not a crash
            return _error(f"{type(exc).__name__}: {exc}"[:400])

    def digest_search(query: str, documents: Optional[List[str]] = None,
                      pages: str = "", kinds: Optional[List[str]] = None,
                      limit: int = 20, offset: int = 0) -> str:
        sources, errors = _resolve(_as_list(documents), attachments)
        if not sources:
            return _error("no documents to search", errors=errors)
        digests, errs = _digests(sources, root)
        errors += errs
        limit = max(1, min(int(limit or 20), 60))
        offset = max(0, int(offset or 0))
        kinds_l = _as_list(kinds)
        single = len(digests) == 1
        hits: List[Tuple[Tuple[int, int, int], Dict[str, Any]]] = []
        more = False
        total = 0
        capped = None
        order = {"exact": 0, "phrase": 1, "all_words": 2}
        for n_doc, (name, d) in enumerate(digests.items()):
            try:
                # One document pages through its own ranking; several are
                # merged, so each gives its best offset + limit and the
                # merged list is paged here.
                res = (d.search(query, pages=pages or None, kinds=kinds_l,
                                limit=limit, offset=offset) if single else
                       d.search(query, pages=pages or None, kinds=kinds_l,
                                limit=offset + limit, offset=0))
            except (IndexError, ValueError) as exc:
                errors.append({"document": name, "error": str(exc)})
                continue
            if res.get("error"):
                return _error(res["error"])
            total += int(res.get("total") or 0)
            capped = capped or res.get("capped")
            # Each document's hits come ranked; across documents the
            # better kind of match goes first, then rank, then upload order.
            for rank, h in enumerate(res["hits"]):
                h["document"] = name
                hits.append(((order.get(h.get("match"), 3), rank, n_doc), h))
            # a later page exists: this document alone ranks past it
            more = more or bool(res.get("next"))
        hits.sort(key=lambda kh: kh[0])
        start = 0 if single else offset
        shown = [h for _k, h in hits[start:start + limit]]
        out: Dict[str, Any] = {"query": " ".join(str(query or "").split())}
        if single:
            out["document"] = next(iter(digests))
            for h in shown:
                h.pop("document", None)
        if errors:
            out["errors"] = errors
        out["total"] = total
        if capped:
            out["capped"] = capped
        out["hits"] = shown
        if not shown:
            out["note"] = ("no line of text matches. Words drawn as lines or "
                           "inside a picture are not in the text: look at "
                           "the pages flagged needs_look before concluding "
                           "it is not there") if not offset else (
                "no more matches: every hit was on an earlier page")
        # ``next`` only when a later page holds hits.
        if shown and (more or len(hits) > start + limit):
            out["next"] = {"offset": offset + limit}
        return _fit(out, "hits", lambda k: {"offset": offset + k})

    def digest_pages(source: str, pages: str = "") -> str:
        sources, errors = _resolve([source], attachments)
        if not sources:
            return _dump(errors[0]) if errors else _error("no such document")
        digests, errs = _digests(sources, root)
        if not digests:
            return _dump(errs[0])
        name, d = next(iter(digests.items()))
        try:
            rows = d.pages(pages or None)
        except (IndexError, ValueError) as exc:
            return _error(str(exc), hint=f"pages are 0-based, 0 to "
                                         f"{d.n_pages - 1}")
        out: Dict[str, Any] = {"document": name, "pages_in_document":
                               d.n_pages, "rows": rows}
        wanted = [r["page"] for r in rows]

        def rest(k: int) -> Dict[str, str]:
            from funhouse_agent.review_digest.digest import compact_ranges
            return {"pages": compact_ranges(wanted[k:])}

        return _fit(out, "rows", rest)

    def digest_references(source: str = "", target: str = "",
                          offset: int = 0) -> str:
        from funhouse_agent.review_digest import cross_references
        sources, errors = _resolve([source] if source else None, attachments)
        if not sources:
            return _error("no documents to look in", errors=errors)
        digests, errs = _digests(sources, root)
        errors += errs
        offset = max(0, int(offset or 0))
        out: Dict[str, Any] = {}
        if errors:
            out["errors"] = errors
        if target and str(target).strip():
            items = []
            for name, d in digests.items():
                for r in d.references(target=target):
                    r["document"] = name
                    items.append(r)
            order = {"caption": 0, "listed": 1, "ref": 2}
            items.sort(key=lambda r: (order.get(r.get("role"), 3),
                                      list(digests).index(r["document"]),
                                      r["page"]))
            out["target"] = str(target).strip()
            out["occurrences"] = items[offset:]
            if not items:
                out["note"] = ("not found in the text of these documents; a "
                               "reference drawn as lines on a sheet is not "
                               "in the text")
            key = "occurrences"
        else:
            xref = cross_references(digests)
            groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for name, d in digests.items():
                for r in d.references():
                    g = groups.setdefault((r["kind"], r["id"]), {
                        "kind": r["kind"], "id": r["id"], "cited": 0,
                        "pages": []})
                    if r.get("target"):
                        g["_target"] = r["target"]
                    if r.get("role") == "caption":
                        g.setdefault("caption", {
                            "document": name, "page": r["page"],
                            "pdf_page": r["pdf_page"],
                            "title": r.get("title")})
                    elif r.get("role") == "ref":
                        g["cited"] += 1
                        where = {"document": name, "page": r["page"],
                                 "pdf_page": r["pdf_page"]}
                        if len(g["pages"]) < 6 and where not in g["pages"]:
                            g["pages"].append(where)
            from funhouse_agent.review_digest.references import sheet_key
            missing = {sheet_key(m["id"]) for m in xref.get("missing") or []}
            for g in groups.values():
                tgt = g.pop("_target", None)
                if tgt and sheet_key(tgt) in missing:
                    g["not_uploaded"] = True
                if not g["pages"]:
                    g.pop("pages")
            items = sorted(groups.values(), key=lambda g: (
                not g.get("not_uploaded"), -g["cited"], g["kind"], g["id"]))
            if len(digests) == 1:
                for g in items:
                    for p in g.get("pages", []):
                        p.pop("document", None)
                    if "caption" in g:
                        g["caption"].pop("document", None)
            out["references"] = items[offset:]
            if xref.get("note"):
                out["note"] = xref["note"]
            key = "references"
        return _fit(out, key, lambda k: {"offset": offset + k})

    return [
        StructuredTool.from_function(document_inventory,
                                     name="document_inventory",
                                     description=INVENTORY_DESCRIPTION),
        StructuredTool.from_function(digest_search, name="digest_search",
                                     description=SEARCH_DESCRIPTION),
        StructuredTool.from_function(digest_pages, name="digest_pages",
                                     description=PAGES_DESCRIPTION),
        StructuredTool.from_function(digest_references,
                                     name="digest_references",
                                     description=REFERENCES_DESCRIPTION),
    ]


__all__ = ["make_digest_tools", "DIGEST_TOOLS", "RESULT_CHARS",
           "INVENTORY_DESCRIPTION", "SEARCH_DESCRIPTION", "PAGES_DESCRIPTION",
           "REFERENCES_DESCRIPTION", "inventory_result"]
