"""Reading a built digest: its inventory, its page rows, search, references.

A :class:`Digest` is a folder written by :func:`funhouse_agent.review_digest.
build` (see that module for the layout). Everything here only READS it, so a
digest can be opened by any number of tools at once, in this turn or a later
one. Pages are 0-based everywhere, as the tools number them, and every page a
result names also carries ``pdf_page`` (page + 1), the number a PDF viewer
shows and a citation uses.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from collections import OrderedDict
from typing import (
    Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union)

from funhouse_agent.review_digest import references as R

#: Bumped whenever what a digest holds changes; an older digest is rebuilt.
FORMAT_VERSION = 1

INVENTORY_NAME = "inventory.json"
PAGES_NAME = "pages.jsonl"
INDEX_NAME = "index.sqlite"
REFERENCES_NAME = "references.json"

#: A search hit's line is shown up to this many characters.
SNIPPET_CHARS = 220
#: At most this many matching lines are ranked (and so can be paged through)
#: per search; a search with more says so and asks to be narrowed.
MAX_RANKED = 5000

PageSpec = Union[None, int, str, Sequence[int]]

#: Digests held open in this process, least recently used first. Bounded: a
#: long-lived host sees many conversations' uploads, and each digest holds
#: its page rows once read.
CACHE_SIZE = 16
_CACHE: "OrderedDict[str, Digest]" = OrderedDict()
_CACHE_GUARD = threading.Lock()


class DigestError(RuntimeError):
    """A document could not be digested at all (not a PDF or image)."""


# ---------------------------------------------------------------------------
# Page numbers
# ---------------------------------------------------------------------------

def compact_ranges(pages: Iterable[int]) -> str:
    """``[0, 1, 2, 5]`` -> ``"0-2,5"``."""
    nums = sorted(set(int(p) for p in pages))
    if not nums:
        return ""
    parts, start, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ",".join(parts)


def pdf_ranges(spec: str) -> str:
    """A 0-based range string as a viewer numbers pages: "0-2,5" -> "1-3,6"."""
    out = []
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        a, sep, b = part.partition("-")
        out.append(f"{int(a) + 1}-{int(b) + 1}" if sep else str(int(a) + 1))
    return ",".join(out)


def parse_pages(spec: PageSpec, n_pages: int) -> List[int]:
    """0-based pages from ``None``/``""`` (all), an int, a list or
    ``"0-4,9"``; pages past the end raise ``IndexError`` naming the range."""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return list(range(n_pages))
    if isinstance(spec, int):
        wanted = [spec]
    elif isinstance(spec, str):
        wanted = []
        for part in spec.replace(" ", "").split(","):
            if not part:
                continue
            a, sep, b = part.partition("-")
            if sep:
                lo, hi = int(a), int(b)
                if hi < lo:
                    lo, hi = hi, lo
                wanted.extend(range(lo, hi + 1))
            else:
                wanted.append(int(a))
    else:
        wanted = [int(p) for p in spec]
    bad = [p for p in wanted if not 0 <= p < n_pages]
    if bad:
        raise IndexError(f"page(s) {bad[:5]} out of range: the document has "
                         f"{n_pages} pages, numbered 0-{n_pages - 1}")
    return list(dict.fromkeys(wanted))


def _ranges(pages: Sequence[int]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for p in sorted(set(pages)):
        if out and p == out[-1][1] + 1:
            out[-1] = (out[-1][0], p)
        else:
            out.append((p, p))
    return out


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\w+", re.UNICODE)


def fts_phrase(query: str) -> Optional[str]:
    """The query as ONE quoted FTS5 phrase of its words - nothing the user
    typed can be read as FTS5 syntax (AND, NEAR, *, :, quotes)."""
    words = _TOKEN.findall(str(query or ""))
    return '"' + " ".join(words) + '"' if words else None


def fts_all_words(query: str) -> Optional[str]:
    """Every word of the query, each quoted, all required."""
    words = list(dict.fromkeys(_TOKEN.findall(str(query or ""))))
    if len(words) < 2:
        return None
    return " AND ".join(f'"{w}"' for w in words)


def _like_escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace("%", "\\%")
            .replace("_", "\\_"))


def _snippet(text: str, query: str) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= SNIPPET_CHARS:
        return text
    low = text.lower()
    at = low.find(query.lower().strip())
    if at < 0:
        words = _TOKEN.findall(query.lower())
        at = min((low.find(w) for w in words if low.find(w) >= 0), default=0)
    start = max(0, at - SNIPPET_CHARS // 3)
    piece = text[start:start + SNIPPET_CHARS]
    return ("…" if start else "") + piece + (
        "…" if start + SNIPPET_CHARS < len(text) else "")


# ---------------------------------------------------------------------------
# The digest
# ---------------------------------------------------------------------------

class Digest:
    """One document's digest folder (read-only)."""

    def __init__(self, folder: str, inventory: Dict[str, Any]):
        self.folder = folder
        self._inv = inventory
        self._rows: Optional[List[Dict[str, Any]]] = None
        self._refs: Optional[List[Dict[str, Any]]] = None
        self._lock = threading.Lock()

    # -- facts --------------------------------------------------------------
    @property
    def name(self) -> str:
        return self._inv.get("name") or os.path.basename(self.folder)

    @property
    def sha256(self) -> str:
        return self._inv["sha256"]

    @property
    def n_pages(self) -> int:
        return int(self._inv.get("pages") or 0)

    def inventory(self) -> Dict[str, Any]:
        """The document's inventory (a copy)."""
        return json.loads(json.dumps(self._inv))

    def summary(self, outline_levels: int = 1) -> Dict[str, Any]:
        """The inventory without its long lists: what an overview shows."""
        inv = self._inv
        out = {k: inv[k] for k in ("name", "pages", "kinds", "needs_look",
                                   "markups", "references") if k in inv}
        for k in ("no_text_layer", "unreliable_text", "unread_pages"):
            if inv.get(k, {}).get("pages"):
                out[k] = {kk: vv for kk, vv in inv[k].items()
                          if kk != "reasons"}
        if inv.get("roles"):
            out["roles"] = inv["roles"]
        if inv.get("segments"):
            out["segments"] = inv["segments"]
        outline = [e for e in inv.get("outline") or []
                   if (e.get("level") or 1) <= outline_levels]
        if outline:
            out["outline"] = outline
        if inv.get("sheet_labels"):
            out["sheet_labels"] = [
                {"sheet": s["sheet"], "page": s["page"],
                 "pdf_page": s["pdf_page"]} for s in inv["sheet_labels"]]
        out["search"] = inv.get("search")
        return out

    # -- pages --------------------------------------------------------------
    def _load_rows(self) -> List[Dict[str, Any]]:
        with self._lock:
            if self._rows is None:
                rows = []
                with open(os.path.join(self.folder, PAGES_NAME),
                          encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            rows.append(json.loads(line))
                self._rows = rows
            return self._rows

    def pages(self, spec: PageSpec = None) -> List[Dict[str, Any]]:
        """The page rows for ``spec`` (0-based; ``"0-4,9"``, a list, an int,
        or ``None`` for all)."""
        rows = self._load_rows()
        return [dict(rows[p]) for p in parse_pages(spec, len(rows))]

    def page(self, index: int) -> Dict[str, Any]:
        return self.pages([int(index)])[0]

    def sheet_labels(self) -> List[Dict[str, Any]]:
        return list(self._inv.get("sheet_labels") or [])

    # -- references ---------------------------------------------------------
    def references(self, target: Optional[str] = None,
                   kinds: Optional[Iterable[str]] = None
                   ) -> List[Dict[str, Any]]:
        """Every cross-reference, or those to ``target`` ("S-501",
        "Table 5-1", "10.17", "Table 12-" for every table numbered 12-...),
        optionally of some ``kinds``; captions first, then by page."""
        with self._lock:
            if self._refs is None:
                with open(os.path.join(self.folder, REFERENCES_NAME),
                          encoding="utf-8") as fh:
                    self._refs = list(json.load(fh).get("references") or [])
            refs = self._refs
        wanted = set(kinds or ())
        out = [dict(r) for r in refs if not wanted or r["kind"] in wanted]
        if target and str(target).strip():
            test = target_test(str(target))
            out = [r for r in out if test(r)]
            order = {"caption": 0, "listed": 1, "ref": 2}
            out.sort(key=lambda r: (order.get(r.get("role"), 3), r["page"]))
        return out

    # -- search -------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(os.path.join(self.folder, INDEX_NAME))
        con.row_factory = sqlite3.Row
        return con

    def search(self, query: str, pages: PageSpec = None,
               kinds: Optional[Iterable[str]] = None, limit: int = 20,
               offset: int = 0) -> Dict[str, Any]:
        """Lines matching ``query``: the exact wording first, then the
        words as a phrase, then all the words anywhere on one line. Each hit
        has page, pdf_page, line_id, the line (snippet), its box in
        displayed-page points, the page's kind and where the text came from
        when it is not the text layer. The user's text is quoted into FTS5,
        and a search FTS5 still refuses is run as a plain substring match.

        Paging: ``offset`` skips that many hits of the SAME ranking (every
        call ranks the best :data:`MAX_RANKED` matching lines, so the pages
        of one search never overlap), ``total`` says how many were ranked,
        and ``next`` is given only when a later page holds hits. A search
        that matches more lines than are ranked says so (``capped``)."""
        q = " ".join(str(query or "").split())
        limit = max(1, min(int(limit or 20), MAX_RANKED))
        offset = max(0, int(offset or 0))
        if not q:
            return {"query": q, "hits": [],
                    "error": "give words to search for"}
        where, args = self._filters(pages, kinds)
        # rowid -> (row, tier, rank); the hit dicts are built for the page
        # returned only, so paging deep into a big result stays cheap.
        found: Dict[int, Tuple[Any, int, float]] = {}
        capped = False
        method = "fts5" if self._inv.get("search") == "fts5" else "like"
        con = self._connect()
        try:
            if method == "fts5":
                try:
                    for tier, match in enumerate(
                            (fts_phrase(q), fts_all_words(q))):
                        if not match or len(found) >= MAX_RANKED:
                            continue
                        rows = self._fts(con, match, where, args, MAX_RANKED)
                        capped = capped or len(rows) >= MAX_RANKED
                        for row in rows:
                            if row["rowid"] not in found:
                                found[row["rowid"]] = (
                                    row, tier, float(row["rank"] or 0.0))
                except sqlite3.OperationalError:
                    method = "like"
                    found = {}
                    capped = False
                if not found and not _TOKEN.search(q):
                    method = "like"
            if method == "like":
                rows = self._like(con, q, where, args, MAX_RANKED)
                capped = len(rows) >= MAX_RANKED
                for row in rows:
                    found[row["rowid"]] = (row, 1, 0.0)
        finally:
            con.close()
        low = q.lower()

        def exact(entry) -> bool:
            return low in str(entry[0]["text"] or "").lower()

        ranked = sorted(found.values(), key=lambda e: (
            not exact(e), e[1], e[2], int(e[0]["page"]),
            int(e[0]["rowid"])))[:MAX_RANKED]
        rows_meta = self._load_rows()
        page_hits = []
        for entry in ranked[offset:offset + limit]:
            row, tier, _rank = entry
            h = self._hit(row, q)
            h["match"] = ("exact" if exact(entry) else
                          ("phrase" if tier == 0 else "all_words"))
            meta = (rows_meta[h["page"]]
                    if 0 <= h["page"] < len(rows_meta) else {})
            for k in ("printed", "sheet"):
                if meta.get(k):
                    h[k] = meta[k]
            if meta.get("needs_look"):
                h["needs_look"] = True
            page_hits.append(h)
        out: Dict[str, Any] = {"query": q, "hits": page_hits,
                               "method": method, "total": len(ranked)}
        if offset + limit < len(ranked):
            out["next"] = {"offset": offset + limit}
        if capped:
            out["capped"] = (f"more lines match than the {MAX_RANKED} "
                             f"ranked: narrow the search (more words, "
                             f"pages or kinds)")
        return out

    def _filters(self, pages: PageSpec, kinds) -> Tuple[str, List[Any]]:
        where, args = [], []
        if pages is not None and not (isinstance(pages, str)
                                      and not pages.strip()):
            wanted = parse_pages(pages, self.n_pages)
            spans = _ranges(wanted)
            where.append("(" + " OR ".join("l.page BETWEEN ? AND ?"
                                           for _ in spans) + ")")
            for a, b in spans:
                args += [a, b]
        kinds = [str(k) for k in (kinds or []) if str(k).strip()]
        if kinds:
            where.append("l.kind IN (" + ",".join("?" for _ in kinds) + ")")
            args += kinds
        return ((" AND " + " AND ".join(where)) if where else ""), args

    @staticmethod
    def _fts(con, match: str, where: str, args: List[Any], want: int):
        return con.execute(
            "SELECT l.rowid AS rowid, l.page, l.line_id, l.kind, l.source, "
            "l.text, l.x0, l.y0, l.x1, l.y1, l.author, "
            "bm25(lines_fts) AS rank FROM lines_fts "
            "JOIN lines l ON l.rowid = lines_fts.rowid "
            "WHERE lines_fts MATCH ?" + where +
            " ORDER BY rank, l.page, l.rowid LIMIT ?",
            [match] + list(args) + [want]).fetchall()

    @staticmethod
    def _like(con, q: str, where: str, args: List[Any], want: int):
        return con.execute(
            "SELECT l.rowid AS rowid, l.page, l.line_id, l.kind, l.source, "
            "l.text, l.x0, l.y0, l.x1, l.y1, l.author FROM lines l "
            "WHERE l.text LIKE ? ESCAPE '\\'" + where +
            " ORDER BY l.page, l.rowid LIMIT ?",
            ["%" + _like_escape(q) + "%"] + list(args) + [want]).fetchall()

    @staticmethod
    def _hit(row, q: str) -> Dict[str, Any]:
        hit: Dict[str, Any] = {
            "page": int(row["page"]), "pdf_page": int(row["page"]) + 1,
            "line_id": row["line_id"], "text": _snippet(row["text"], q),
            "bbox": [row["x0"], row["y0"], row["x1"], row["y1"]],
            "kind": row["kind"]}
        if row["source"] and row["source"] != "pdf_text":
            hit["source"] = row["source"]
        if row["author"]:
            hit["author"] = row["author"]
        return hit


# ---------------------------------------------------------------------------
# Matching a target
# ---------------------------------------------------------------------------

_KIND_WORDS = {
    "table": "table", "tables": "table", "tbl": "table",
    "figure": "figure", "figures": "figure", "fig": "figure",
    "appendix": "appendix", "appendices": "appendix", "app": "appendix",
    "attachment": "attachment", "exhibit": "attachment", "annex": "attachment",
    "enclosure": "attachment",
    "sheet": "sheet", "sheets": "sheet", "sht": "sheet", "dwg": "sheet",
    "drawing": "sheet",
    "detail": "detail", "details": "detail", "det": "detail",
    "std": "standard", "standard": "standard", "standards": "standard",
    "section": "section", "sections": "section", "sect": "section",
    "sec": "section", "spec": "spec_section", "specification":
    "spec_section",
}


def _norm(ident: str) -> str:
    return " ".join(str(ident).upper().replace(" ", " ").split())


def target_test(target: str):
    """A predicate for references to ``target``: "Table 5-1" (kind and id),
    "S-501" / "10.17" (any kind; a sheet or standard also matches its
    lettered sheets, 10.17 -> 10.17A), "Table 12-" (a prefix)."""
    t = " ".join(str(target).split()).strip()
    m = re.match(r"^([A-Za-z]+)\.?\s*(?:no\.?\s*|#\s*)?(.*)$", t)
    kind = None
    ident = t
    if m and m.group(1).lower() in _KIND_WORDS and m.group(2):
        kind = _KIND_WORDS[m.group(1).lower()]
        ident = m.group(2)
    ident = ident.strip().lstrip("#").strip()
    prefix = ident.endswith(("-", "."))
    want = _norm(ident)

    def test(ref: Dict[str, Any]) -> bool:
        rk = ref["kind"]
        if kind and not (rk == kind or {rk, kind} <= {"section",
                                                     "spec_section"}):
            return False
        rid = _norm(ref["id"])
        if prefix:
            return rid.startswith(want)
        if rid == want:
            return True
        if rk in R.SHEET_KINDS:
            tgt = ref.get("target") or ref["id"]
            if R.sheet_key(tgt) == R.sheet_key(want):
                return True
            # standard 10.17 <-> its lettered sheets 10.17A, 10.17B
            return R.label_matches(want, [tgt]) is not None
        return False

    return test


# ---------------------------------------------------------------------------
# Opening one
# ---------------------------------------------------------------------------

def load_digest(folder: str, sha256: Optional[str] = None
                ) -> Optional[Digest]:
    """The digest in ``folder`` when it is complete, of the current format
    and (if given) of this content; else ``None`` (it is to be built)."""
    folder = os.path.abspath(folder)
    path = os.path.join(folder, INVENTORY_NAME)
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return None
    with _CACHE_GUARD:
        cached = _CACHE.get(folder)
        if cached is not None:
            _CACHE.move_to_end(folder)
    if cached is not None and getattr(cached, "_mtime", None) == mtime and (
            sha256 is None or cached.sha256 == sha256):
        return cached
    try:
        with open(path, encoding="utf-8") as fh:
            inv = json.load(fh)
    except (OSError, ValueError):
        return None
    if inv.get("format") != FORMAT_VERSION:
        return None
    if sha256 is not None and inv.get("sha256") != sha256:
        return None
    for name in (PAGES_NAME, INDEX_NAME, REFERENCES_NAME):
        if not os.path.isfile(os.path.join(folder, name)):
            return None
    d = Digest(folder, inv)
    d._mtime = mtime
    with _CACHE_GUARD:
        _CACHE[folder] = d
        _CACHE.move_to_end(folder)
        while len(_CACHE) > CACHE_SIZE:
            _CACHE.popitem(last=False)
    return d


def cache_size() -> int:
    """How many digests this process holds open (at most
    :data:`CACHE_SIZE`)."""
    with _CACHE_GUARD:
        return len(_CACHE)


def clear_cache() -> None:
    """Forget the digests held open in this process (tests)."""
    with _CACHE_GUARD:
        _CACHE.clear()


__all__ = ["Digest", "DigestError", "FORMAT_VERSION", "load_digest",
           "parse_pages", "compact_ranges", "pdf_ranges", "fts_phrase",
           "fts_all_words", "target_test", "clear_cache", "cache_size",
           "CACHE_SIZE", "MAX_RANKED", "INVENTORY_NAME", "PAGES_NAME",
           "INDEX_NAME", "REFERENCES_NAME"]
