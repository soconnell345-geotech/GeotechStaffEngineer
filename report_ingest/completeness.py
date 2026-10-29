"""Is this copy of the report the whole of it? (BACKLOG RI-184)

The owner's ruling of 2026-10-29: the report is the file. The figure and
table counts are what the file holds, and a copy whose text cites what the
file lacks is FLAGGED, so a record read off a partial copy says so. Two
things the pages themselves give away:

1. A PRINTED PAGE NUMBERING THAT RUNS SHORT. Pages that print "Page X of Y"
   with one Y and a rising X are one document's run. A printed page is
   counted missing only where the file has no page that could hold it: the
   run's pages and the file pages between them are counted, so a page that
   prints no number (a figure, a blank) is not taken for a missing one. A
   run that ends early, starts late or skips numbers it has no room for
   names the pages it lacks. Each document is named by where its run is,
   because the run may be the report's or an attachment's (build f: one
   copy lacked its last page; another two of a laboratory's certificates).
2. FIGURES, TABLES AND APPENDICES THE TEXT CITES THAT THE FILE DOES NOT
   HOLD. A citation is "Figure 3", "Figures 1 through 4", "Table 2",
   "Appendices A-D"; one beside a code or a standard ("Figure 22.1 of ASCE
   7") is another document's and is left out. One is held where a line
   leads with its caption or title on a page that is not a list (a contents
   page names many; a caption page few, :data:`MAX_CAPTIONS_ON_A_PAGE`).
   A figure page with no text to read is a figure the code cannot name, so
   figures are flagged missing only past the number of such pages.

:func:`completeness_qa` makes one ``partial`` QA entry, ranked ``open``,
for each document whose numbering is short and one for what the text cites
that the file lacks. It reads text only and calls no model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "MAX_CAPTIONS_ON_A_PAGE", "PICTURE_LABELS", "CITING_LABELS",
    "UNCHECKED_RUN_LABELS", "ShortRun", "short_runs", "cited_not_held",
    "completeness_qa", "page_texts",
]

#: A page whose lines lead with more captions or titles than this is a list
#: of them (contents, a list of figures), not the pages that hold them.
MAX_CAPTIONS_ON_A_PAGE = 3
#: Page labels whose pages may hold a figure the text layer cannot name.
PICTURE_LABELS: Tuple[str, ...] = ("figure", "plan", "profile", "photos")
#: How few characters a page has before it counts as having no text.
_NO_TEXT = 30
#: How far apart two numbered pages of one run may be beyond their numbers:
#: a figure or a blank page between two numbered ones prints none.
_SLACK = 3
#: A run that has not gone on for this many pages has ended.
_ENDED = 6

_RE_PAGE_OF = re.compile(r"\bPage\s*(\d{1,3})\s*of\s*(\d{1,3})\b", re.I)
#: A label: "3", "A-2", "4.1", "B".
_NUM = r"[A-Z]?-?\d{1,2}(?:[.\-]\d{1,2})?[A-Z]?(?![\d])"
#: A citation, on one line: "Water Table" at a line's end is not a table
#: whatever number the next line starts with.
_RE_CITE = re.compile(
    rf"\b(Figures?|Figs?\.|Tables?|Appendi(?:x|ces)|Plates?)[ \t]+"
    rf"((?:{_NUM}|[A-H]\b)(?:[ \t]*(?:,|and|&|through|to|thru|-|–)[ \t]*"
    rf"(?:{_NUM}|[A-H]\b))*)")
#: How near a citation a code's or a standard's name makes it that
#: document's figure or table ("the ASCE 7 table", "Figure 3 of the USGS
#: map"): characters before it, and after.
_ELSEWHERE_BEFORE, _ELSEWHERE_AFTER = 25, 30
#: The pages whose citations are the report's own, by label. A calculation
#: reproducing a textbook's pages cites the textbook's figures, and a
#: laboratory's or a design tool's printout its own tables.
CITING_LABELS: Tuple[str, ...] = ("narrative", "letter", "cover", "toc")
#: The pages whose "Page X of Y" runs are not checked, by label: a
#: calculation package reproduces a code's or a program's pages, and its
#: numbering is often theirs (build f: pages 3 and 4 of 4 of an excerpt
#: after two calculation sheets that print none).
UNCHECKED_RUN_LABELS: Tuple[str, ...] = ("calculation",)
#: Words beside a citation that make it another document's.
_RE_ELSEWHERE = re.compile(
    r"ASCE|USGS|\bIBC\b|\bCBC\b|AASHTO|\bACI\b|\bUFC\b|ASTM|FEMA|NAVFAC|"
    r"Eurocode|\bcode\b|Standard|Manual|Caltrans|FHWA|\bGEC\b|\bDM\s?7|"
    r"EM 1110|Section \d|Equation|design ?maps", re.I)
_RE_CAPTION = re.compile(
    rf"^\s*(FIGURE|Figure|FIG\.?|Fig\.?|PLATE|Plate|TABLE|Table|APPENDIX|"
    rf"Appendix)\s*(?:NO\.?\s*)?({_NUM}|[A-H]\b)", re.M)
_RE_RANGE = re.compile(
    rf"^({_NUM}|[A-H])\s*(?:through|to|thru|-|–)\s*({_NUM}|[A-H])$")
_KIND_WORD = {"figure": ("Figure", "Figures"), "table": ("Table", "Tables"),
              "appendix": ("Appendix", "Appendices")}


# ---------------------------------------------------------------------------
# the pages' text
# ---------------------------------------------------------------------------

def page_texts(doc: Any, pages: Optional[Sequence[int]] = None) -> List[str]:
    """Each page's text, its lines one to a line; "" for a page that will
    not read. ``pages`` default: every page of ``doc``."""
    out: List[str] = []
    for page in (range(doc.n_pages) if pages is None else pages):
        try:
            content = doc.page(int(page), tables=False)
        except Exception:                        # a page that will not read
            out.append("")
            continue
        out.append("\n".join(str(line.text or "") for line in content.lines))
    return out


# ---------------------------------------------------------------------------
# 1. a printed numbering that runs short
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShortRun:
    """One document's "Page X of Y" run, and the printed pages it lacks.
    ``pages`` are the file's 0-based pages that print the run's numbers."""

    of: int
    pages: Tuple[int, ...]
    printed: Tuple[int, ...]
    missing: Tuple[int, ...]


def _runs(hits: Mapping[int, Sequence[Tuple[int, int]]]
          ) -> List[List[Tuple[int, int, int]]]:
    """Chain each page's (x, y) to an open run with the same y whose last x
    is below it, within the slack; else start a run."""
    open_runs: List[List[Tuple[int, int, int]]] = []
    done: List[List[Tuple[int, int, int]]] = []
    for page in sorted(hits):
        for x, y in sorted(set(hits[page])):
            for run in open_runs:
                last_page, last_x, last_y = run[-1]
                if (last_y == y and last_x < x and page > last_page
                        and page - last_page <= x - last_x + _SLACK):
                    run.append((page, x, y))
                    break
            else:
                open_runs.append([(page, x, y)])
        for run in list(open_runs):
            if page - run[-1][0] > _ENDED:
                open_runs.remove(run)
                done.append(run)
    return sorted(done + open_runs, key=lambda run: run[0][0])


def _shadows(run: Sequence[Tuple[int, int, int]],
             runs: Sequence[Sequence[Tuple[int, int, int]]]) -> bool:
    """Is ``run`` a second numbering printed on another, longer run's
    pages, number for number?"""
    mine = {page: x for page, x, _y in run}
    for other in runs:
        if other is run or len(other) <= len(run):
            continue
        theirs = {page: x for page, x, _y in other}
        if all(theirs.get(page) == x for page, x in mine.items()):
            return True
    return False


def short_runs(texts: Sequence[str],
               labels: Optional[Mapping[int, str]] = None) -> List[ShortRun]:
    """Every "Page X of Y" run whose printed pages the file cannot hold.

    A document prints its footer on every page, so the pages that may hold
    a printed page a run lacks are few: between two of its numbered pages,
    any page; before its first or after its last, a page with no text to
    read (a scan); and before it, one page more for its page 1, which
    often prints no footer (a cover, a letter's first page). A run printed
    on the pages of a longer one, number for number, is that run's second
    numbering and is not counted, and neither is a run on pages
    ``labels`` gives as calculations alone (:data:`UNCHECKED_RUN_LABELS`).
    """
    labels = labels or {}
    hits: Dict[int, List[Tuple[int, int]]] = {}
    for page, text in enumerate(texts):
        found = [(int(x), int(y)) for x, y in _RE_PAGE_OF.findall(text or "")
                 if 0 < int(x) <= int(y)]
        if found:
            hits[page] = found
    runs = _runs(hits)
    covered = {page for run in runs for page, _x, _y in run}

    def blank(page: int) -> bool:
        return (0 <= page < len(texts) and page not in covered
                and len((texts[page] or "").strip()) < _NO_TEXT)

    out: List[ShortRun] = []
    for run in runs:
        if _shadows(run, runs) or (labels and all(
                labels.get(page) in UNCHECKED_RUN_LABELS
                for page, _x, _y in run)):
            continue
        of, lost = run[0][2], []
        first_page, first_x = run[0][0], run[0][1]
        page, held = first_page - 1, 0
        while held < first_x - 1 and blank(page):
            page, held = page - 1, held + 1
        need = first_x - 1 - held               # printed pages 1..need
        if need == 1 and 0 <= page and page not in covered:
            need = 0                            # its page 1, unnumbered
        lost += list(range(1, need + 1))
        for (p0, x0, _), (p1, x1, _) in zip(run, run[1:]):
            if x1 - x0 > p1 - p0:
                lost += list(range(x0 + 1, x0 + 1 + (x1 - x0) - (p1 - p0)))
        last_page, last_x = run[-1][0], run[-1][1]
        page, held = last_page + 1, 0
        while held < of - last_x and blank(page):
            page, held = page + 1, held + 1
        lost += list(range(last_x + 1 + held, of + 1))
        if lost:
            out.append(ShortRun(of=of,
                                pages=tuple(p for p, _x, _y in run),
                                printed=tuple(x for _p, x, _y in run),
                                missing=tuple(lost)))
    return out


# ---------------------------------------------------------------------------
# 2. what the text cites that the file does not hold
# ---------------------------------------------------------------------------

def _kind(word: str) -> str:
    word = word.lower()
    if word.startswith(("fig", "plate")):
        return "figure"
    return "table" if word.startswith("table") else "appendix"


def _norm(label: str) -> str:
    return re.sub(r"\s+", "", label).upper().lstrip("0") or "0"


def _expand(text: str) -> List[str]:
    """ "1 through 4" -> 1..4, "A-C" -> A..C, "2 and 3" -> 2, 3."""
    out: List[str] = []
    for part in re.split(r"\s*(?:,|and|&)\s*", text):
        part = part.strip()
        if not part:
            continue
        match = _RE_RANGE.match(part)
        if match:
            a, b = match.groups()
            if a.isdigit() and b.isdigit() and int(a) < int(b) <= int(a) + 30:
                out += [str(k) for k in range(int(a), int(b) + 1)]
                continue
            if (len(a) == len(b) == 1 and a.isalpha() and b.isalpha()
                    and a < b):
                out += [chr(k) for k in range(ord(a), ord(b) + 1)]
                continue
            if a.isdigit() and b.isdigit() and "-" in part:
                out.append(part)                 # "3-2": a label, not a range
                continue
            out += [a, b]
            continue
        out.append(part)
    return out


def cited_not_held(texts: Sequence[str],
                   labels: Optional[Mapping[int, str]] = None
                   ) -> Dict[str, Dict[str, List[int]]]:
    """``kind -> label -> the pages citing it`` for each figure, table and
    appendix the text cites that no page holds. ``labels`` are the page
    labels, when there are any: the report's own citations are on its
    :data:`CITING_LABELS` pages, a ``toc`` page holds nothing, and a picture
    page (:data:`PICTURE_LABELS`) whose figure the text does not name may
    hold any one figure. Without labels every page is read for both."""
    labels = labels or {}
    cited: Dict[Tuple[str, str], List[int]] = {}
    held: set = set()
    pictures = 0
    for page, text in enumerate(texts):
        text = text or ""
        label = labels.get(page)
        if not labels or label in CITING_LABELS:
            for match in _RE_CITE.finditer(text):
                near = (text[max(0, match.start() - _ELSEWHERE_BEFORE):
                             match.start()]
                        + " " + text[match.end():
                                     match.end() + _ELSEWHERE_AFTER])
                if _RE_ELSEWHERE.search(near):
                    continue
                for name in _expand(match.group(2)):
                    at = cited.setdefault(
                        (_kind(match.group(1)), _norm(name)), [])
                    if page not in at:
                        at.append(page)
        if label == "toc":
            continue
        captions = {(_kind(a), _norm(b)) for a, b in _RE_CAPTION.findall(text)}
        if len(captions) <= MAX_CAPTIONS_ON_A_PAGE:
            held |= captions
        if label in PICTURE_LABELS and not any(
                kind == "figure" for kind, _name in captions):
            pictures += 1
    out: Dict[str, Dict[str, List[int]]] = {}
    for (kind, label), pages in cited.items():
        if (kind, label) not in held:
            out.setdefault(kind, {})[label] = sorted(pages)
    figures = out.get("figure", {})
    if figures and len(figures) <= pictures:
        out.pop("figure")                        # they may be those pages
    return out


# ---------------------------------------------------------------------------
# the QA entries
# ---------------------------------------------------------------------------

def _span(pages: Sequence[int]) -> str:
    """0-based pages as the report's 1-based span: "pages 3-6"."""
    if not pages:
        return ""
    first, last = min(pages) + 1, max(pages) + 1
    return f"page {first}" if first == last else f"pages {first}-{last}"


def _and(parts: Sequence[str]) -> str:
    parts = list(parts)
    return parts[0] if len(parts) == 1 else \
        ", ".join(parts[:-1]) + " and " + parts[-1]


def _numbers(numbers: Sequence[Any]) -> str:
    """Labels in order, an unbroken run of whole numbers or of single
    letters written as a span: "1-3 and 5-8", "A-C", "3, 5 and A1"."""
    def rank(item: str) -> Optional[int]:
        if item.isdigit():
            return int(item)
        return ord(item) if len(item) == 1 and item.isalpha() else None

    spans: List[List[str]] = []
    for item in (str(n) for n in numbers):
        last = spans[-1][-1] if spans else None
        if (last is not None and rank(item) is not None
                and rank(last) is not None
                and item.isdigit() == last.isdigit()
                and rank(item) == rank(last) + 1):
            spans[-1].append(item)
        else:
            spans.append([item])
    return _and([span[0] if len(span) == 1 else f"{span[0]}-{span[-1]}"
                 for span in spans])


def completeness_qa(texts: Sequence[str],
                    labels: Optional[Mapping[int, str]] = None) -> list:
    """The ``partial`` QA entries, ranked ``open``, for a copy that looks
    incomplete: one per document whose printed numbering is short, and one
    for the figures, tables and appendices the text cites that the file
    does not hold. ``texts`` are the file's pages' text in page order
    (:func:`page_texts`); ``labels`` its page labels, when it has them."""
    from report_ingest.model import QAEntry

    out: List[Any] = []
    for run in short_runs(texts, labels):
        whole = run.of
        lacks = _numbers(run.missing)
        out.append(QAEntry(
            kind="partial", rank="open", where="document.completeness",
            detail=(f"this copy looks incomplete: the document numbered "
                    f"'Page X of {whole}' on {_span(run.pages)} lacks its "
                    f"page{'s' if len(run.missing) > 1 else ''} {lacks} of "
                    f"{whole}; the file has no page to hold "
                    f"{'them' if len(run.missing) > 1 else 'it'}"),
            values=[f"Page {n} of {whole}" for n in run.missing],
            pages=list(run.pages)))
    absent = cited_not_held(texts, labels)
    if absent:
        said = []
        citing: set = set()
        for kind in ("figure", "table", "appendix"):
            found = absent.get(kind)
            if not found:
                continue
            one, many = _KIND_WORD[kind]
            names = sorted(found, key=lambda s: (len(s), s))
            said.append(f"{one if len(names) == 1 else many} "
                        f"{_numbers(names)}")
            for pages in found.values():
                citing |= set(pages)
        out.append(QAEntry(
            kind="partial", rank="open", where="document.completeness",
            detail=(f"this copy looks incomplete: the text cites "
                    f"{_and(said)}, which the file does not hold"),
            values=[f"{kind} {label}: cited on {_span(pages)}"
                    for kind in ("figure", "table", "appendix")
                    for label, pages in sorted(absent.get(kind, {}).items())],
            pages=sorted(citing)))
    return out
