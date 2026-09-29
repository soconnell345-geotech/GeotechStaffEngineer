"""Building a document's digest (the FREE layer: code only, no model call).

``build(source)`` reads one PDF once with planlens and writes, under
``<root>/<sha256[:16]>/``:

* ``inventory.json`` - what the document is: name, sha256, pages, page kinds,
  pages with no text layer or unreliable text, the pages to look at,
  segments with their printed page ranges, the PDF's outline (bookmarks),
  sheet labels, markups and their authors, reference counts, format version;
* ``pages.jsonl`` - one compact row per page (see :func:`_row`);
* ``index.sqlite`` - an FTS5 index over every text line (page, line id, text,
  box in displayed-page points) and every markup's text;
* ``references.json`` - the cross-references found in the text
  (:mod:`funhouse_agent.review_digest.references`), each with page and box.

A digest is keyed by the document's CONTENT, so the same PDF uploaded again
(under any name) reuses it while its format version matches. An odd page
never stops a build: it is recorded as ``kind="unread"`` with the reason, and
flagged to be looked at.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
import weakref
from collections import Counter, OrderedDict
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union)

from funhouse_agent.review_digest import references as R
from funhouse_agent.review_digest.digest import (
    FORMAT_VERSION, INDEX_NAME, INVENTORY_NAME, PAGES_NAME, REFERENCES_NAME,
    Digest, DigestError, compact_ranges, load_digest, pdf_ranges)

Source = Union[bytes, bytearray, str, "os.PathLike[str]"]

#: The digests live in this folder of the conversation's working folder.
DIGEST_DIRNAME = "digest"
#: Page kinds whose content is a picture: they must be looked at.
NEEDS_LOOK_KINDS = ("drawing_sheet", "figure", "scanned", "form")
#: How long a page's excerpt runs.
EXCERPT_CHARS = 300
#: References listed on one page's row (the rest are in references.json).
ROW_REFERENCES = 12
#: A line on at least this fraction of pages (and 3) is a running header.
RUNNING_FRACTION = 0.2
#: Outline entries kept in the inventory.
MAX_OUTLINE = 1000

class _BuildLock:
    """One digest folder's build lock. A plain object (not a bare lock) so
    the registry can hold it weakly: it lives while a build holds it and is
    dropped after, so a long-lived host does not keep one per folder ever
    built."""

    __slots__ = ("lock", "__weakref__")

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def __enter__(self) -> "_BuildLock":
        self.lock.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.lock.release()


_BUILD_LOCKS: "weakref.WeakValueDictionary[str, _BuildLock]" = \
    weakref.WeakValueDictionary()
_BUILD_GUARD = threading.Lock()
#: Content hashes of files read, by (path, size, mtime), newest last.
PATH_SHA_SIZE = 256
_PATH_SHA: "OrderedDict[Tuple[str, int, int], str]" = OrderedDict()


# ---------------------------------------------------------------------------
# Where and what
# ---------------------------------------------------------------------------

def default_root() -> str:
    """``<working folder>/digest`` - the conversation's working folder the
    host sets (``GEOTECH_DEFAULT_OUTPUT_DIR``), read at call time."""
    from funhouse_agent._fileio import default_output_dir
    return os.path.abspath(os.path.join(default_output_dir() or os.getcwd(),
                                        DIGEST_DIRNAME))


def root_for(working_dir: Optional[str]) -> Optional[str]:
    """``<working_dir>/digest`` for a working folder bound when the agent was
    built; ``None`` (look the working folder up at call time) without one."""
    if not working_dir:
        return None
    return os.path.abspath(os.path.join(os.fspath(working_dir),
                                        DIGEST_DIRNAME))


def sha256_of(source: Source) -> str:
    """The document's content hash (a path's is remembered by size and
    modification time, so a big file is read once)."""
    if isinstance(source, (bytes, bytearray)):
        return hashlib.sha256(bytes(source)).hexdigest()
    path = os.path.abspath(os.fspath(source))
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    with _BUILD_GUARD:
        got = _PATH_SHA.get(key)
        if got is not None:
            _PATH_SHA.move_to_end(key)
    if got is None:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        got = h.hexdigest()
        with _BUILD_GUARD:
            _PATH_SHA[key] = got
            while len(_PATH_SHA) > PATH_SHA_SIZE:
                _PATH_SHA.popitem(last=False)
    return got


def _name_of(source: Source, name: Optional[str]) -> str:
    if name:
        return str(name)
    if isinstance(source, (bytes, bytearray)):
        return "document.pdf"
    return os.path.basename(os.fspath(source))


# ---------------------------------------------------------------------------
# Small text helpers
# ---------------------------------------------------------------------------

#: AutoCAD control codes left in hidden SHX text ("%%UPLAN VIEW", "2.1%%%").
_CAD_CODES = (("%%%", "%"), ("%%U", ""), ("%%u", ""), ("%%O", ""),
              ("%%o", ""), ("%%D", "°"), ("%%d", "°"),
              ("%%P", "±"), ("%%p", "±"), ("%%C", "Ø"),
              ("%%c", "Ø"))


def _clean(text: str, source: str) -> str:
    text = " ".join(str(text or "").split())
    if source == "cad_hidden_text" and "%%" in text:
        for a, b in _CAD_CODES:
            text = text.replace(a, b)
    return text


def _mask(text: str) -> str:
    """What stays the same when a line repeats from page to page."""
    return re.sub(r"\d+", "#", " ".join(text.lower().split()))


def _letters(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def _short(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= n:
        return text
    cut = text[: n - 1].rsplit(" ", 1)[0]
    return (cut or text[: n - 1]) + "…"


def _hex_label(label: Optional[str]) -> Optional[str]:
    """A page label some writers leave hex-encoded ("<FEFF005B...>")."""
    if not label:
        return None
    m = re.fullmatch(r"<\s*(FEFF[0-9A-Fa-f]*)\s*>", label.strip())
    if m:
        try:
            label = bytes.fromhex(m.group(1)).decode("utf-16-be").lstrip(
                "﻿")
        except (ValueError, UnicodeDecodeError):
            return None
    label = " ".join(label.split())
    return label or None


def _bbox(b: Sequence[float]) -> List[float]:
    return [round(float(v), 1) for v in b]


def _union(boxes: Iterable[Sequence[float]]) -> Optional[List[float]]:
    bs = [b for b in boxes if b is not None]
    if not bs:
        return None
    return _bbox((min(b[0] for b in bs), min(b[1] for b in bs),
                  max(b[2] for b in bs), max(b[3] for b in bs)))


# ---------------------------------------------------------------------------
# What a page prints about itself
# ---------------------------------------------------------------------------

#: A printed page number standing alone: "12", "5-2", "A-3", "iii", "Page 7".
_PRINTED = re.compile(
    r"^(?:(?i:page|pg\.?)\s+)?(?P<v>(?:[A-Z]{1,2}-)?\d{1,3}(?:-\d{1,3})?"
    r"|[ivxlc]{1,7}|[IVXLC]{2,7})$")


def _printed_label(lines, width: float, height: float, kind: str
                   ) -> Optional[str]:
    """The page number printed alone in the top or bottom band (planlens
    reads "Page N of M"; a manual prints "5-2" or "iii" by itself). Centred
    wins over a corner; drawing sheets are left to their title block."""
    if kind == "drawing_sheet":
        return None
    best = None
    for ln in lines:
        if ln.source != "pdf_text" or not ln.bbox:
            continue
        m = _PRINTED.match(ln.text.strip())
        if not m:
            continue
        x0, y0, x1, y1 = ln.bbox
        cx = (x0 + x1) / 2.0
        bottom = y0 >= 0.87 * height
        top = y1 <= 0.10 * height
        if not (bottom or top):
            continue
        centred = abs(cx - width / 2.0) <= 0.12 * width
        corner = x0 >= 0.75 * width
        if not (centred or (corner and kind in ("text", "mixed"))):
            continue
        score = (2 if bottom else 0) + (1 if centred else 0)
        if best is None or score > best[0]:
            best = (score, m.group("v"))
    return best[1] if best else None


def _title_block_sheet(lines) -> Optional[str]:
    """The sheet number a title block prints: the value beside or under a
    label word ("SHEET NO.", "STD. NO.", "DWG"), or one line saying both."""
    for ln in lines:
        m = R.LABEL_INLINE.match(ln.text)
        if m and R.is_sheet_label(m.group("v"), allow_plain=True):
            return m.group("v").rstrip(".").upper()
    for ln in lines:
        if not ln.bbox or not R.LABEL_WORD.match(ln.text):
            continue
        lx0, ly0, lx1, ly1 = ln.bbox
        h = max(ly1 - ly0, 2.0)
        best = None
        for other in lines:
            if other is ln or not other.bbox:
                continue
            if not R.is_sheet_label(other.text, allow_plain=True):
                continue
            ox0, oy0, ox1, oy1 = other.bbox
            below = (oy0 >= ly0 + 0.3 * h and oy0 - ly1 <= 3.0 * h
                     and not (ox1 < lx0 - 2 * h or ox0 > lx1 + 2 * h))
            right = (ox0 >= lx1 - 0.5 * h and ox0 - lx1 <= 8.0 * h
                     and abs((oy0 + oy1) / 2 - (ly0 + ly1) / 2) <= h)
            if below or right:
                d = math.hypot((ox0 + ox1) / 2 - (lx0 + lx1) / 2,
                               (oy0 + oy1) / 2 - (ly0 + ly1) / 2)
                if best is None or d < best[0]:
                    best = (d, other.text.strip().rstrip(".").upper())
        if best:
            return best[1]
    return None


def _sheet_label(lines, summary, index: int, name: str, n_pages: int
                 ) -> Tuple[Optional[str], Optional[str]]:
    """``(sheet, where it was read)``: the title block first, then the sheet
    reference planlens reads in the page's bands, then the PDF's own page
    label when it reads like a sheet number, then - for a one-page upload -
    a file name that is a sheet number ("S-501.pdf")."""
    got = _title_block_sheet(lines)
    if got:
        return got, "title block"
    sheet = getattr(summary, "sheet", None)
    if sheet and " of " not in sheet.lower() and R.is_sheet_label(sheet):
        return sheet.upper(), "page band"
    label = _hex_label(getattr(summary, "label", None))
    if label and label != str(index + 1) and R.is_sheet_label(label):
        return label.upper(), "page label"
    if n_pages == 1:
        stem = os.path.splitext(os.path.basename(name))[0]
        if R.is_sheet_label(stem):
            return stem.upper(), "file name"
    return None, None


#: The top and bottom bands where running headers and footers print
#: (manuals measured: the header ends by 10 % of the page, the page number
#: starts after 91 %; a numbered body heading starts below 11 %).
BAND = 0.105


def _band_lines(lines, height: float):
    """The lines in the page's top or bottom band, short enough to be a
    running header or footer."""
    return [ln for ln in lines if ln.bbox and len(ln.text) <= 80 and (
        ln.bbox[3] <= BAND * height or ln.bbox[1] >= (1.0 - BAND) * height)]


def _heading(lines, skip: set, summary) -> Optional[str]:
    """planlens' heading unless it is a running header or footer (a line in
    ``skip``); else the largest line that is not one (first on ties)."""
    head = getattr(summary, "heading", None)
    skipped = {" ".join(ln.text.split()) for ln in lines if ln.id in skip}
    if head and " ".join(head.split()) not in skipped:
        return _short(head, 100)
    best = None
    for ln in lines:
        text = ln.text
        if _letters(text) < 3 or ln.id in skip:
            continue
        size = ln.size or 0.0
        if best is None or size > best[0] + 0.1:
            best = (size, text)
    return _short(best[1], 100) if best else None


# ---------------------------------------------------------------------------
# One page: text groups and their references
# ---------------------------------------------------------------------------

class _Line:
    """A text line as the digest keeps it (text cleaned of CAD codes)."""
    __slots__ = ("id", "text", "bbox", "size", "source")

    def __init__(self, ln):
        self.id = str(ln.id)
        self.source = str(ln.source or "pdf_text")
        self.text = _clean(ln.text, self.source)
        self.bbox = tuple(float(v) for v in ln.bbox) if ln.bbox else None
        self.size = float(ln.size) if ln.size else None


def _groups(pc, lines: List[_Line]) -> List[List[_Line]]:
    """One group per text block (its lines joined, so a reference broken
    across a line break is still found), one per line outside a block."""
    by_id = {ln.id: ln for ln in lines}
    used = set()
    out: List[List[_Line]] = []
    for b in getattr(pc, "blocks", None) or []:
        members = [by_id[str(i)] for i in b.line_ids if str(i) in by_id]
        members = [m for m in members if m.id not in used]
        if members:
            out.append(members)
            used.update(m.id for m in members)
    out.extend([ln] for ln in lines if ln.id not in used)
    return out


def _caption_title(line: _Line, group: List[_Line], lines: List[_Line],
                   width: float) -> Optional[str]:
    """A caption printed alone ("Table 12-1") takes the line under it."""
    pos = group.index(line)
    if pos + 1 < len(group):
        return group[pos + 1].text
    if not line.bbox:
        return None
    x0, y0, x1, y1 = line.bbox
    h = max(y1 - y0, 2.0)
    cx = (x0 + x1) / 2
    best = None
    for other in lines:
        if other is line or not other.bbox or _letters(other.text) < 3:
            continue
        ox0, oy0, ox1, oy1 = other.bbox
        gap = oy0 - y1
        if -0.3 * h <= gap <= 2.5 * h and abs((ox0 + ox1) / 2 - cx) <= \
                0.4 * width:
            if best is None or gap < best[0]:
                best = (gap, other.text)
    return best[1] if best else None


#: A page with this many dotted-leader lines is a contents list: a caption
#: printed there is an entry in the list, not the table or figure itself.
CONTENTS_LEADERS = 5
_LEADER_LINE = re.compile(r"(?:\.\s?){4,}\s*\S{1,8}\s*$")


def _page_references(pc, lines: List[_Line], skip: set, width: float,
                     index: int, own_sheet: Optional[str]
                     ) -> List[Dict[str, Any]]:
    out = []
    own = R.sheet_key(own_sheet) if own_sheet else None
    contents = sum(1 for ln in lines
                   if _LEADER_LINE.search(ln.text)) >= CONTENTS_LEADERS
    for group in _groups(pc, lines):
        group = [ln for ln in group if ln.id not in skip]
        if not group:
            continue
        starts, parts, pos = [], [], 0
        for ln in group:
            starts.append(pos)
            parts.append(ln.text)
            pos += len(ln.text) + 1
        text = " ".join(parts)
        for ref in R.find_references(text, starts):
            involved = [ln for s, ln in zip(starts, group)
                        if s < ref["end"] and s + len(ln.text) > ref["start"]]
            # The sheet's own number in its title block is not a reference.
            tgt = ref.get("target")
            if own and tgt and R.sheet_key(tgt) == own:
                continue
            if contents and ref["role"] == "caption":
                ref["role"] = "listed"
            if ref["role"] in ("caption", "listed") and not ref.get("title") \
                    and involved:
                title = _caption_title(involved[-1], group, lines, width)
                if title:
                    ref["title"] = _short(title, 120)
            ref.pop("start", None)
            ref.pop("end", None)
            ref["page"] = index
            ref["pdf_page"] = index + 1
            if involved:
                ref["line_id"] = involved[0].id
                box = _union(ln.bbox for ln in involved)
                if box:
                    ref["bbox"] = box
                if involved[0].source != "pdf_text":
                    ref["source"] = involved[0].source
            out.append(ref)
    return out


def _excerpt(lines: List[_Line], skip: set) -> str:
    text = " ".join(ln.text for ln in lines if ln.id not in skip and ln.text)
    return _short(text, EXCERPT_CHARS)


def _row_ref(ref: Dict[str, Any]) -> str:
    kind = ref["kind"].replace("_", " ")
    s = f"{kind} {ref['id']}"
    if ref.get("role") in ("caption", "listed"):
        s += f" ({ref['role']})"
    return s


# ---------------------------------------------------------------------------
# Reading the document
# ---------------------------------------------------------------------------

def _open(source: Source, name: str):
    from planlens.document import Document
    if isinstance(source, (bytes, bytearray)):
        return Document(content=bytes(source), name=name)
    return Document(filepath=os.fspath(source), name=name)


def _read(source: Source, name: str, sha: str) -> Dict[str, Any]:
    """Everything the digest writes, from one pass over the document."""
    t0 = time.time()
    try:
        doc = _open(source, name)
    except Exception as exc:  # noqa: BLE001 - not a PDF planlens can open
        raise DigestError(f"'{name}' could not be opened as a PDF or image: "
                          f"{type(exc).__name__}: {exc}") from exc
    try:
        n = doc.n_pages
        read: Dict[int, Tuple[Any, Any, List[_Line]]] = {}
        unread: Dict[int, str] = {}
        for i in range(n):
            try:
                s = doc.summary(i)
                pc = doc.page(i, tables=False)
                lines = [_Line(ln) for ln in pc.lines
                         if str(ln.text or "").strip()]
                read[i] = (s, pc, lines)
            except Exception as exc:  # noqa: BLE001 - recorded, never raised
                unread[i] = f"{type(exc).__name__}: {str(exc)[:200]}"
        segments = _segments([read[i][0] for i in sorted(read)])
        roles = _roles(read)
        try:
            toc = doc.toc()
        except Exception:  # noqa: BLE001 - an unreadable outline is none
            toc = []
        try:
            meta = {k: v for k, v in (doc.metadata or {}).items()
                    if k in ("title", "author", "subject", "creator",
                             "producer") and v}
        except Exception:  # noqa: BLE001
            meta = {}
    finally:
        try:
            doc.close()
        except Exception:  # noqa: BLE001
            pass

    # Running headers and footers: a line in the top or bottom band of many
    # pages says nothing about any one of them (it would be every page's
    # heading and excerpt). Only the bands count: a numbered heading in the
    # body ("2. Subsurface Conditions") repeats its shape on every page too.
    running: set = set()
    if len(read) >= 4:
        counts: Counter = Counter()
        for s, _pc, lines in read.values():
            counts.update({_mask(ln.text) for ln in _band_lines(lines,
                                                                s.height)})
        floor = max(3, int(math.ceil(RUNNING_FRACTION * len(read))))
        running = {k for k, c in counts.items() if c >= floor}

    rows: List[Dict[str, Any]] = []
    refs: List[Dict[str, Any]] = []
    index_rows: List[Tuple] = []
    authors: Counter = Counter()
    n_markups = 0
    for i in range(n):
        if i in unread:
            rows.append({"page": i, "pdf_page": i + 1, "kind": "unread",
                         "unread": unread[i], "text_reliable": False,
                         "needs_look": True, "words": 0})
            continue
        s, pc, lines = read[i]
        kind = s.kind
        try:
            printed = (str(s.printed_page) if s.printed_page is not None
                       else _printed_label(lines, s.width, s.height, kind))
            sheet, sheet_from = _sheet_label(lines, s, i, name, n)
            skip = {ln.id for ln in _band_lines(lines, s.height)
                    if _mask(ln.text) in running}
            if printed:
                skip |= {ln.id for ln in lines
                         if ln.text.strip() in (printed, f"Page {printed}")}
            page_refs = _page_references(pc, lines, skip, s.width, i, sheet)
            for ref in page_refs:
                if printed:
                    ref["printed"] = printed
                if sheet:
                    ref["sheet"] = sheet
            heading = _heading(lines, skip, s)
            excerpt = _excerpt(lines, skip)
        except Exception as exc:  # noqa: BLE001 - keep the page, say why
            printed = sheet = sheet_from = heading = None
            page_refs, excerpt = [], ""
            unread_note = f"{type(exc).__name__}: {str(exc)[:200]}"
        else:
            unread_note = None
        refs.extend(page_refs)
        # No text layer: none at all (a CAD plot whose lettering is drawn
        # as lines, a scan), or planlens says the page needs OCR.
        no_text = ((int(s.n_text_chars) == 0 and kind != "blank")
                   or (bool(s.evidence.get("needs_ocr")) and s.text_reliable))
        row: Dict[str, Any] = {"page": i, "pdf_page": i + 1, "kind": kind}
        role = roles.get(i)
        if role:
            row["role"] = role
        label = _hex_label(s.label)
        if label and label != str(i + 1):
            row["label"] = label
        if sheet:
            row["sheet"] = sheet
            row["sheet_from"] = sheet_from
        if printed:
            row["printed"] = printed
        if heading:
            row["heading"] = heading
        row["words"] = int(s.n_words)
        if s.n_markups:
            row["markups"] = int(s.n_markups)
        row["text_reliable"] = bool(s.text_reliable)
        if no_text:
            row["no_text_layer"] = True
        if s.n_cad_text:
            row["hidden_cad_text"] = int(s.n_cad_text)
        row["needs_look"] = bool(kind in NEEDS_LOOK_KINDS
                                 or not s.text_reliable)
        if s.duplicate_of is not None:
            row["duplicate_of"] = int(s.duplicate_of)
        if s.segment is not None:
            row["segment"] = int(s.segment)
        if page_refs:
            shown = list(dict.fromkeys(_row_ref(r) for r in page_refs))
            row["references"] = shown[:ROW_REFERENCES]
            if len(shown) > ROW_REFERENCES:
                row["references_more"] = len(shown) - ROW_REFERENCES
        if excerpt:
            row["excerpt"] = excerpt
        if unread_note:
            row["partly_read"] = unread_note
        rows.append(row)
        for ln in lines:
            b = ln.bbox or (0.0, 0.0, 0.0, 0.0)
            index_rows.append((i, ln.id, kind, ln.source, ln.text,
                               round(b[0], 1), round(b[1], 1),
                               round(b[2], 1), round(b[3], 1), None))
        for mk in getattr(pc, "markups", None) or []:
            n_markups += 1
            authors[mk.author or "(no author)"] += 1
            text = " | ".join(x for x in (mk.text, mk.subject,
                                          getattr(mk, "appearance_text", None))
                              if x)
            if not text.strip():
                continue
            b = mk.bbox or (0.0, 0.0, 0.0, 0.0)
            index_rows.append((i, str(mk.id), kind, "markup", _clean(text, ""),
                               round(b[0], 1), round(b[1], 1),
                               round(b[2], 1), round(b[3], 1), mk.author))

    inventory = _inventory(name, sha, n, rows, segments, toc, meta, refs,
                           authors, n_markups, roles, len(index_rows))
    inventory["build_seconds"] = round(time.time() - t0, 2)
    return {"inventory": inventory, "rows": rows, "refs": refs,
            "index_rows": index_rows}


def _segments(summaries) -> List[Dict[str, Any]]:
    """planlens' segments over the pages that were read (it also stamps
    each summary with its segment)."""
    try:
        from planlens.document.structure import segments
        return segments(summaries)
    except Exception:  # noqa: BLE001 - no structure is not a failed build
        return []


#: A page role is kept only when a rule this sure of it fired: the page
#: names itself (0.9) or its appendix tab does (0.75). planlens' roles were
#: tuned on geotechnical reports; its guesses from a page's shape alone
#: (0.4) call a manual's cover a "calculation", so they are left out.
MIN_ROLE_CONFIDENCE = 0.75


def _roles(read) -> Dict[int, str]:
    """planlens' page roles (narrative, figure, plan, toc, divider, log, ...)
    from the text already read; none if the rules cannot run."""
    try:
        from planlens.document.roles import assign, facts_for
        facts = [facts_for(read[i][0], read[i][1].lines) for i in sorted(read)]
        return {r.page: r.role for r in assign(facts)
                if r.role != "other"
                and float(r.confidence) >= MIN_ROLE_CONFIDENCE}
    except Exception:  # noqa: BLE001 - roles are extra
        return {}


_LEADER_TAIL = re.compile(r"(?:\s*\.\s*){3,}\S*\s*$")


def _inventory(name, sha, n, rows, segments, toc, meta, refs, authors,
               n_markups, roles, n_index) -> Dict[str, Any]:
    by_page = {r["page"]: r for r in rows}
    kinds = Counter(r["kind"] for r in rows)
    look = [r["page"] for r in rows if r.get("needs_look")]
    no_text = [r["page"] for r in rows if r.get("no_text_layer")]
    unreliable = [r["page"] for r in rows if r["kind"] != "unread"
                  and not r.get("text_reliable", True)]
    unread = [r["page"] for r in rows if r["kind"] == "unread"]

    def printed_range(first: int, last: int) -> Optional[str]:
        labels = [by_page[p].get("printed") for p in range(first, last + 1)
                  if p in by_page and by_page[p].get("printed")]
        if not labels:
            return None
        return labels[0] if labels[0] == labels[-1] else \
            f"{labels[0]} to {labels[-1]}"

    segs = []
    for g in segments:
        pages = str(g.get("pages", ""))
        a, _, b = pages.partition("-")
        try:
            first, last = int(a), int(b or a)
        except ValueError:
            continue
        seg = {"id": g.get("id"), "title": _LEADER_TAIL.sub(
                   "", str(g.get("title") or "")).strip()[:100] or None,
               "pages": pages, "pdf_pages": pdf_ranges(pages),
               "n_pages": g.get("n_pages"), "kinds": g.get("kinds")}
        pr = printed_range(first, last)
        if pr:
            seg["printed"] = pr
        segs.append({k: v for k, v in seg.items() if v not in (None, "", {})})
    outline = []
    for e in toc or []:
        title = " ".join(str(e.get("title") or "").split())
        page = e.get("page")
        if not title or not isinstance(page, int) or not 0 <= page < n:
            continue
        item = {"level": e.get("level"), "title": title[:160], "page": page,
                "pdf_page": page + 1}
        if by_page.get(page, {}).get("printed"):
            item["printed"] = by_page[page]["printed"]
        outline.append(item)
    sheets = [{"page": r["page"], "pdf_page": r["pdf_page"],
               "sheet": r["sheet"], "from": r.get("sheet_from")}
              for r in rows if r.get("sheet")]
    ref_kinds = Counter(r["kind"] for r in refs if r.get("role") == "ref")
    captions = Counter(r["kind"] for r in refs if r.get("role") == "caption")
    inv: Dict[str, Any] = {
        "format": FORMAT_VERSION,
        "name": name,
        "sha256": sha,
        "pages": n,
        "kinds": dict(kinds.most_common()),
        "needs_look": {"pages": compact_ranges(look),
                       "pdf_pages": pdf_ranges(compact_ranges(look)),
                       "n": len(look)},
        "no_text_layer": {"pages": compact_ranges(no_text),
                          "pdf_pages": pdf_ranges(compact_ranges(no_text)),
                          "n": len(no_text)},
        "unreliable_text": {
            "pages": compact_ranges(unreliable),
            "pdf_pages": pdf_ranges(compact_ranges(unreliable)),
            "n": len(unreliable)},
        "segments": segs,
        "outline": outline[:MAX_OUTLINE],
        "sheet_labels": sheets,
        "markups": {"n": n_markups, "authors": dict(authors.most_common())},
        "references": {"cited": dict(ref_kinds.most_common()),
                       "captions": dict(captions.most_common())},
        "indexed_lines": n_index,
        "metadata": meta,
    }
    if roles:
        inv["roles"] = dict(Counter(roles.values()).most_common())
    if unread:
        inv["unread_pages"] = {"pages": compact_ranges(unread),
                               "pdf_pages": pdf_ranges(compact_ranges(unread)),
                               "reasons": {str(r["page"]): r["unread"]
                                           for r in rows
                                           if r["kind"] == "unread"}}
    if len(outline) > MAX_OUTLINE:
        inv["outline_more"] = len(outline) - MAX_OUTLINE
    return inv


# ---------------------------------------------------------------------------
# Writing it
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE lines (
    rowid   INTEGER PRIMARY KEY,
    page    INTEGER,
    line_id TEXT,
    kind    TEXT,
    source  TEXT,
    text    TEXT,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL,
    author  TEXT
);
CREATE INDEX lines_page ON lines (page);
"""
_FTS = ("CREATE VIRTUAL TABLE lines_fts USING fts5(text, content='lines', "
        "content_rowid='rowid', tokenize='porter unicode61')")


def _write_index(path: str, sha: str, index_rows: List[Tuple]) -> bool:
    """The search index; ``True`` when FTS5 is there, ``False`` when the
    digest falls back to plain LIKE search."""
    con = sqlite3.connect(path)
    try:
        con.executescript(_SCHEMA)
        con.executemany(
            "INSERT INTO lines (page, line_id, kind, source, text, x0, y0, "
            "x1, y1, author) VALUES (?,?,?,?,?,?,?,?,?,?)", index_rows)
        try:
            con.execute(_FTS)
            con.execute("INSERT INTO lines_fts(lines_fts) VALUES('rebuild')")
            fts = True
        except sqlite3.OperationalError:
            fts = False
        con.executemany("INSERT INTO meta (key, value) VALUES (?, ?)",
                        [("format", str(FORMAT_VERSION)), ("sha256", sha),
                         ("fts", "1" if fts else "0")])
        con.commit()
    finally:
        con.close()
    return fts


def _write(folder: str, data: Dict[str, Any]) -> None:
    inv = data["inventory"]
    fts = _write_index(os.path.join(folder, INDEX_NAME), inv["sha256"],
                       data["index_rows"])
    inv["search"] = "fts5" if fts else "like"
    with open(os.path.join(folder, PAGES_NAME), "w", encoding="utf-8") as fh:
        for row in data["rows"]:
            fh.write(json.dumps(row, ensure_ascii=False,
                                separators=(",", ":")) + "\n")
    with open(os.path.join(folder, REFERENCES_NAME), "w",
              encoding="utf-8") as fh:
        json.dump({"format": FORMAT_VERSION, "references": data["refs"]}, fh,
                  ensure_ascii=False, separators=(",", ":"))
    # The inventory last: its presence (with the right version) is what says
    # the digest is complete.
    with open(os.path.join(folder, INVENTORY_NAME), "w",
              encoding="utf-8") as fh:
        json.dump(inv, fh, ensure_ascii=False, indent=1)


def _lock_for(key: str) -> _BuildLock:
    """The build lock for one digest folder, shared by every caller that
    holds it at once (the caller's reference is what keeps it alive)."""
    with _BUILD_GUARD:
        lock = _BUILD_LOCKS.get(key)
        if lock is None:
            lock = _BuildLock()
            _BUILD_LOCKS[key] = lock
        return lock


def build(source: Source, name: Optional[str] = None,
          root: Optional[str] = None, force: bool = False) -> Digest:
    """The digest of one document (PDF bytes or a path), built on first use
    and reused while its content hash and format version match.

    Raises :class:`DigestError` only when the file cannot be opened at all;
    a page that cannot be read is recorded in the digest instead.
    """
    name = _name_of(source, name)
    sha = sha256_of(source)
    root = os.path.abspath(root or default_root())
    folder = os.path.join(root, sha[:16])
    with _lock_for(folder):
        if not force:
            existing = load_digest(folder, sha)
            if existing is not None:
                return existing
        data = _read(source, name, sha)
        os.makedirs(root, exist_ok=True)
        tmp = os.path.join(root, f".{sha[:16]}.{uuid.uuid4().hex[:8]}.tmp")
        os.makedirs(tmp)
        try:
            _write(tmp, data)
            if os.path.isdir(folder):
                shutil.rmtree(folder, ignore_errors=True)
            os.replace(tmp, folder)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            # Another process may have finished the same digest first.
            existing = load_digest(folder, sha)
            if existing is not None:
                return existing
            raise
    digest = load_digest(folder, sha)
    if digest is None:                       # pragma: no cover - just written
        raise DigestError(f"the digest of '{name}' could not be read back")
    return digest


# ---------------------------------------------------------------------------
# Several uploads at once
# ---------------------------------------------------------------------------

SHAPE1_PAGES_ENV = "GEOTECH_REVIEW_SHAPE1_PAGES"
DEFAULT_SHAPE1_PAGES = 20


def shape1_pages() -> int:
    """Pages at or below which a review is "small"
    (``GEOTECH_REVIEW_SHAPE1_PAGES``, default 20)."""
    try:
        return max(1, int(os.environ.get(SHAPE1_PAGES_ENV,
                                         DEFAULT_SHAPE1_PAGES)))
    except (TypeError, ValueError):
        return DEFAULT_SHAPE1_PAGES


def cross_references(digests: Mapping[str, Digest]) -> Dict[str, Any]:
    """Which sheets and standards the documents cite that no upload carries
    as a sheet label (the rule is :func:`references.label_matches`)."""
    labels: Dict[str, Tuple[str, int]] = {}
    for doc_name, d in digests.items():
        for s in d.sheet_labels():
            labels.setdefault(s["sheet"], (doc_name, s["page"]))
    cited: Dict[str, Dict[str, Any]] = {}
    for doc_name, d in digests.items():
        for ref in d.references():
            tgt = ref.get("target")
            if not tgt or ref.get("role") != "ref":
                continue
            key = R.sheet_key(tgt)
            entry = cited.setdefault(key, {"id": tgt, "kinds": set(),
                                           "n": 0, "cited": []})
            entry["kinds"].add(ref["kind"])
            entry["n"] += 1
            if len(entry["cited"]) < 5:
                entry["cited"].append({"document": doc_name,
                                       "page": ref["page"],
                                       "pdf_page": ref["pdf_page"],
                                       "text": ref.get("text")})
    missing, found = [], []
    for key, e in cited.items():
        e["kinds"] = sorted(e["kinds"])
        hit = R.label_matches(e["id"], labels)
        if hit is None:
            missing.append(e)
        else:
            doc_name, page = labels[hit]
            found.append({"id": e["id"], "sheet": hit, "document": doc_name,
                          "page": page, "pdf_page": page + 1,
                          "by_suffix": R.sheet_key(hit) != key})
    missing.sort(key=lambda e: (-e["n"], e["id"]))
    found.sort(key=lambda e: e["id"])
    out: Dict[str, Any] = {"missing": missing, "found": found,
                           "labels_known": len(labels)}
    if not labels and missing:
        out["note"] = ("no sheet labels were read from the uploads (a sheet "
                       "whose lettering is drawn as lines has none), so every "
                       "cited sheet is unmatched: check the title blocks by "
                       "looking")
    return out


def inventory_of(sources: Mapping[str, Source],
                 root: Optional[str] = None) -> Dict[str, Any]:
    """The digests of several uploads at once (built on first use): totals,
    one summary per document, a ``shape_hint`` and the cited sheets and
    standards that no upload contains."""
    digests: Dict[str, Digest] = {}
    errors = []
    docs = []
    for doc_name, src in sources.items():
        t0 = time.time()
        try:
            d = build(src, name=doc_name, root=root)
        except Exception as exc:  # noqa: BLE001 - one bad upload is reported
            errors.append({"document": doc_name,
                           "error": f"{type(exc).__name__}: {exc}"[:300]})
            continue
        digests[doc_name] = d
        summary = d.summary()
        summary["document"] = doc_name
        summary["seconds"] = round(time.time() - t0, 2)
        docs.append(summary)
    total = sum(d.n_pages for d in digests.values())
    limit = shape1_pages()
    out: Dict[str, Any] = {
        "documents": docs,
        "totals": {"documents": len(digests), "pages": total,
                   "needs_look": sum(d.inventory()["needs_look"]["n"]
                                     for d in digests.values()),
                   "markups": sum(d.inventory()["markups"]["n"]
                                  for d in digests.values())},
        "shape_hint": "small" if total <= limit else "large",
        "shape1_pages": limit,
        "cross_references": cross_references(digests),
    }
    if errors:
        out["errors"] = errors
    return out


__all__ = ["build", "inventory_of", "cross_references", "default_root",
           "root_for", "sha256_of", "shape1_pages", "DIGEST_DIRNAME",
           "NEEDS_LOOK_KINDS", "SHAPE1_PAGES_ENV", "DEFAULT_SHAPE1_PAGES"]
