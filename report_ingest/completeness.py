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
   A kind is flagged only when the file holds none of the cited ones: a
   figure may sit on a page labelled a calculation or a laboratory sheet
   whose text prints no caption (build f), so one held figure means the
   others may be there too. A figure page whose text names no figure is
   one the code cannot name, so figures are flagged missing only past the
   number of such pages.

:func:`completeness_qa` makes one ``partial`` QA entry, ranked ``open``,
for each document whose numbering is short and one for what the text cites
that the file lacks. It reads text only and calls no model.

Given the record the readers filled, this module also checks what a geotechnical
report names and a partial copy drops (BACKLOG RI-185), with the same
caution (a whole report flagged is worse than a partial copy missed):

3. EXPLORATIONS NAMED, WITH NO LOG (:func:`named_not_logged`). A hole
   named at least twice on the report's own pages ("borings B-1 through
   B-8", "TP-3") that no log in the record carries
   (:func:`report_ingest.log_holes.names_match`) and no log page or bound
   document prints. A name is a hole's only in a series the record's own
   holes have or an exploration word introduces. One the text says was not
   drilled, was abandoned, was by others or is an earlier investigation's
   is not missing.
4. LABORATORY RESULTS NAMED, WITH NONE HELD (:func:`lab_said_not_held`).
   The record holds no laboratory test and the file no laboratory page,
   and the text says twice over that the results are in an appendix or
   attached to THIS document.
5. APPENDICES IN THE CONTENTS WITH NO PAGES
   (:func:`listed_appendices_not_held`). A contents page lists appendices
   by letter and title, the file holds some, and for another no page leads
   with its letter or its title and no page labelled for its content
   follows the last one held.

:func:`named_not_held_qa` makes one ``partial`` entry, ranked ``open``, that
names all three in the report's own words with the pages that name them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "MAX_CAPTIONS_ON_A_PAGE", "PICTURE_LABELS", "CITING_LABELS",
    "UNCHECKED_RUN_LABELS", "ShortRun", "short_runs", "cited_not_held",
    "completeness_qa", "page_texts",
    "MIN_MENTIONS", "LOG_LABELS", "named_not_logged", "lab_said_not_held",
    "listed_appendices_not_held", "named_not_held_qa",
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
#: The pages whose "Page X of Y" runs are not checked, by label: pages
#: that reproduce ANOTHER document's pages, whose numbering is that
#: document's and is excerpted on purpose. A calculation package reproduces
#: a code's or a program's pages (build f: pages 3 and 4 of 4 of an excerpt
#: after two calculation sheets that print none); a photo appendix
#: reproduces selected pages of an earlier investigation's photo log (MF-3:
#: six false notes on a complete report; RI-201). A laboratory's
#: certificates and the report's own pages are its own documents and are
#: checked (build f: a certificate printed as its pages 3-6 of 13 flags).
UNCHECKED_RUN_LABELS: Tuple[str, ...] = ("calculation", "photos")
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
    ``labels`` give as reproductions of another document's pages alone,
    calculations or photos (:data:`UNCHECKED_RUN_LABELS`; RI-201).
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
    appendix the text cites that no page holds, of a kind the file holds
    none of: where it holds some, the rest may be on pages whose text
    prints no caption (a figure on a page labelled a calculation or a
    laboratory sheet). ``labels`` are the page labels, when there are any:
    the report's own citations are on its :data:`CITING_LABELS` pages, a
    ``toc`` page holds nothing, and a picture page (:data:`PICTURE_LABELS`)
    whose figure the text does not name may hold any one figure. Without
    labels every page is read for both."""
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
    for kind in list(out):
        if any(key in held for key in cited if key[0] == kind):
            out.pop(kind)       # the file holds some: the rest may be uncaptioned
    figures = out.get("figure", {})
    if figures and len(figures) <= pictures:
        out.pop("figure")                        # they may be those pages
    return out


# ---------------------------------------------------------------------------
# 3-5. what the report names that the record and the file do not hold
#      (RI-185)
# ---------------------------------------------------------------------------

#: How many times the report's own pages must name a thing before its
#: absence is flagged: a single mention is thin evidence (RI-201).
MIN_MENTIONS = 2
#: The page labels that are one exploration's own record.
LOG_LABELS: Tuple[str, ...] = ("boring_log", "test_pit_log", "cpt_log",
                               "dcp_log")
#: Labels whose pages may print a hole's record besides the logs.
_MAY_HOLD_LOGS: Tuple[str, ...] = LOG_LABELS + ("field_test",
                                                "appended_report")
#: How much of a sentence either side of a mention is read for the words
#: that make it not missing (a table's cells have no sentence ends).
_NEAR = 150
#: How many of a page's first lines are read for a title.
_LEAD_LINES = 8
#: How long a quoted sentence is in an entry.
_QUOTE = 160

#: A hole's name as a report writes one: a series of letters and a number,
#: "B-1", "TP-3", "CPT-2", "MW-02A", "B1".
_HOLE = r"(?<![A-Za-z0-9\-–])([A-Z]{1,4})([-–]?)(\d{1,3}[A-Z]?)(?![A-Za-z0-9])"
_RE_HOLE = re.compile(_HOLE)
_RE_HOLE_RANGE = re.compile(
    _HOLE + r"\s*(?:through|thru|to|–|-)\s*" + _HOLE)
#: A word that introduces an exploration's name: "borings B-1, ...",
#: "test pits (TP-1 through TP-6)", "soundings designated CPT-1".
_RE_HOLE_WORD = re.compile(
    r"\b(?:borings?|bore\s?holes?|test\s?pits?|test\s?holes?|soundings?|"
    r"cone penetration tests?|CPTs?|probes?|hand augers?|auger holes?|"
    r"(?:monitoring |observation )?wells?|piezometers?)\b[^.;:]{0,25}?$",
    re.I)
#: What a name follows when it is not a hole's ("Figure B-1", "Sample S-3").
_RE_NOT_A_HOLE_HEAD = re.compile(
    r"(?:Figures?|Figs?\.?|Tables?|Appendi(?:x|ces)|Plates?|Sheets?|"
    r"Drawings?|Sections?|Samples?|Zones?|Types?|Class|Grade|Details?|"
    r"Photos?|Exhibits?|Attachments?|ASTM|AASHTO)[\s(]*$", re.I)
#: The names listed before a name: "B-1, B-2 and " of "B-1, B-2 and B-3".
#: What the list follows is what its last name follows.
_RE_LISTED_BEFORE = re.compile(
    r"(?:[A-Z]{1,4}[-–]?\d{1,3}[A-Z]?\s*(?:,|and|&|through|thru|to|,\s*and)"
    r"\s*)+$")
#: Words in a mention's sentence that make the hole one this report's
#: investigation did not drill, or another investigation's.
_RE_HOLE_NOT_MISSING = re.compile(
    r"\bnot\s+(?:be\s+|been\s+)?(?:drilled|performed|advanced|completed|"
    r"excavated|made|accessible|possible)|could not|unable to|abandon|"
    r"omitted|cancel|eliminated|deleted|relocated|inaccessible|"
    r"by others|\bprevious|\bprior\b|\bearlier\b|\bformer|historic|"
    r"\bpreliminary\b|\bproposed (?:boring|test pit|sounding|exploration)|"
    r"\bplanned\b|\bfuture\b|recommend|"
    r"(?:19|20)\d\d\s+(?:investigation|study|exploration|report)", re.I)
#: The text saying the logs are in another document, not this one.
_RE_ELSEWHERE_VOLUME = re.compile(
    r"under separate cover|separate(?:ly bound)? (?:volume|report|"
    r"document)|data report|submitted separately|provided separately", re.I)
_RE_SENTENCE_END = re.compile(r"(?<=[.;])\s+(?=[A-Z(])")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text or "")


def _sentence(flat: str, start: int, end: int) -> str:
    """The sentence of ``flat`` around ``[start, end)``, no more than
    :data:`_NEAR` characters either side."""
    lo, hi = max(0, start - _NEAR), min(len(flat), end + _NEAR)
    before = list(_RE_SENTENCE_END.finditer(flat, lo, start))
    if before:
        lo = before[-1].end()
    after = _RE_SENTENCE_END.search(flat, end, hi)
    if after:
        hi = after.start()
    return flat[lo:hi].strip()


def _head(flat: str, start: int) -> str:
    """What comes before the name at ``start``, and before the list of
    names it is in."""
    return _RE_LISTED_BEFORE.sub("", flat[max(0, start - 120):start])


def _hole_mentions(flat: str):
    """Each hole-like name in ``flat``: (start, end, series, hyphenated,
    names). A range "B-1 through B-8" of one series gives its names."""
    taken: List[Tuple[int, int]] = []
    for match in _RE_HOLE_RANGE.finditer(flat):
        s1, h1, n1, s2, _h2, n2 = match.groups()
        if (s1 == s2 and n1.isdigit() and n2.isdigit()
                and int(n1) < int(n2) <= int(n1) + 60):
            width = len(n1) if n1.startswith("0") else 0
            names = [f"{s1}{h1}{str(k).zfill(width)}"
                     for k in range(int(n1), int(n2) + 1)]
            taken.append((match.start(), match.end()))
            yield match.start(), match.end(), s1, bool(h1), names
    for match in _RE_HOLE.finditer(flat):
        if any(a <= match.start() < b for a, b in taken):
            continue
        series, hyphen, _number = match.groups()
        yield (match.start(), match.end(), series, bool(hyphen),
               [match.group(0)])


def named_not_logged(texts: Sequence[str], labels: Mapping[int, str],
                     record: Any, bound: Sequence[int] = ()
                     ) -> Dict[str, List[int]]:
    """``name -> the pages naming it`` for each exploration the report's
    own pages name at least :data:`MIN_MENTIONS` times that the record
    holds no log, sounding or pit record for.

    A name is a hole's only in a series (its letters, written with or
    without a hyphen as the series is) that the record's own holes have or
    that an exploration word introduces on these pages; "D50" beside
    borings named "D-1" is not one. Names after "Figure", "Sample" and the
    like are not holes'. A hole is held when a log in the record carries
    its name (:func:`report_ingest.log_holes.names_match`), or a log page,
    an appended report's page or a bound document's page prints it: the
    reader may have read it under another name, which
    ``log_holes.missing_holes_qa`` says, and a bound document's holes are
    its own record's. A hole whose sentence says it was not drilled, was
    abandoned, was by others or is a previous investigation's
    (:data:`_RE_HOLE_NOT_MISSING`) is not missing, and nothing is where
    the text says the logs are under separate cover. Nothing is flagged
    either while the holes missing are no more than the record's holes
    under names the text does not use plus the log pages with no text to
    read that no hole in the record was read from: they may be the same
    holes.
    """
    from report_ingest.log_holes import names_match

    bound = {int(p) for p in bound}
    recorded = [str(inv.investigation_id) for inv in
                getattr(record, "investigations", ()) or ()
                if getattr(inv, "investigation_id", "")]
    series = set()
    for name in recorded:
        found = _RE_HOLE.fullmatch(name.strip().upper())
        if found:
            series.add((found.group(1), bool(found.group(2))))

    citing = [(page, _flat(text)) for page, text in enumerate(texts)
              if page not in bound and labels.get(page) in CITING_LABELS]
    if any(_RE_ELSEWHERE_VOLUME.search(flat) for _page, flat in citing):
        return {}
    found = [(page, flat, list(_hole_mentions(flat)))
             for page, flat in citing]
    for _page, flat, mentions in found:
        for start, _end, letters, hyphen, _names in mentions:
            if _RE_HOLE_WORD.search(_head(flat, start)):
                series.add((letters, hyphen))

    named: Dict[str, List[int]] = {}          # name -> a page per mention
    excused: set = set()
    for page, flat, mentions in found:
        for start, end, letters, hyphen, names in mentions:
            if (letters, hyphen) not in series:
                continue
            if _RE_NOT_A_HOLE_HEAD.search(_head(flat, start)):
                continue
            if _RE_HOLE_NOT_MISSING.search(_sentence(flat, start, end)):
                excused |= set(names)
            for name in names:
                named.setdefault(name, []).append(page)

    elsewhere = " ".join(
        _flat(text).upper() for page, text in enumerate(texts)
        if page in bound or labels.get(page) in _MAY_HOLD_LOGS)
    printed = [m.group(0) for m in _RE_HOLE.finditer(elsewhere)]
    lacking = {
        name: sorted(set(pages)) for name, pages in named.items()
        if len(pages) >= MIN_MENTIONS and name not in excused
        and not any(names_match(name, other) for other in recorded)
        and not any(names_match(name, other) for other in printed)}
    if not lacking:
        return {}
    def unread(page: int) -> bool:
        return len((texts[page] or "").strip()) < _NO_TEXT

    if any(unread(page) for page in bound if 0 <= page < len(texts)):
        return {}                   # a scanned bound document may hold them
    unnamed = sum(1 for other in recorded
                  if not any(names_match(name, other) for name in named))
    scans = sum(1 for page in range(len(texts))
                if labels.get(page) in _MAY_HOLD_LOGS and unread(page))
    from_scans = sum(1 for other in recorded
                     if not any(names_match(other, name) for name in printed))
    if len(lacking) <= unnamed + max(0, scans - from_scans):
        return {}
    return lacking


#: A sentence about laboratory results ...
_RE_LAB = re.compile(r"laborator(?:y|ies)|\blab\b", re.I)
_RE_LAB_RESULTS = re.compile(r"results?|data\b|tests?\b|testing", re.I)
#: ... that says they are in this document: presented, included, attached,
#: in an appendix, an attachment or an enclosure. A table or a figure is
#: left out (it may sit on a narrative page the readers took as text), and
#: so is "on the boring logs" (those results are the logs').
_RE_LAB_INCLUDED = re.compile(
    r"\b(?:are|is|were|was|been)\s+(?:also\s+)?(?:presented|included|"
    r"provided|shown|contained|summari[sz]ed|given|tabulated|reported|"
    r"plotted|compiled|attached|enclosed|appended)\s+(?:in|on|as|at|to)\s+"
    r"(?:the\s+)?(?:attached\s+)?(?:Appendi(?:x|ces)|Attachments?|"
    r"Enclosures?)\b"
    r"|\b(?:are|is)\s+(?:also\s+)?(?:attached|enclosed|appended)\b"
    r"|\b(?:Appendi(?:x|ces)|Attachments?|Enclosures?)\s+\S{1,4}\s+"
    r"(?:presents?|contains?|includes?|provides?)\b", re.I)
#: Words that make a sentence about results that are NOT in this document.
_RE_LAB_NOT_HERE = re.compile(
    r"\bwill\b|\bto be\b|should be|\bshall\b|recommend|\bplanned\b|"
    r"\bproposed\b|pending|forthcoming|in progress|under separate cover|"
    r"separate (?:report|volume|letter|document)|"
    r"\bnot\s+(?:been\s+|yet\s+)?(?:performed|conducted|included|"
    r"available|completed)|\bno laboratory|by others|\bprevious|"
    r"\bprior\b|\bearlier\b|upon request|on file|\bif\b", re.I)
#: A contents line, or an appendix's title, that names laboratory results.
_RE_LAB_TITLE = re.compile(
    r"^\s*(?:APPENDIX|Appendix|ATTACHMENT|Attachment)\s+\S{1,4}\s*"
    r"[-–—:.]?\s*[^\n]*laborator[^\n]*$", re.I | re.M)
#: The tests a sentence may name, as the entry says them.
_LAB_KINDS: Tuple[Tuple[str, str], ...] = (
    ("sieve or grain size", r"sieve|grain[- ]size|gradation|particle[- ]size"
                            r"|hydrometer"),
    ("Atterberg limits", r"atterberg|liquid limit|plastic(?:ity)? (?:limit|"
                         r"index)"),
    ("consolidation", r"consolidation|oedometer"),
    ("triaxial", r"triaxial"),
    ("direct shear", r"direct shear"),
    ("unconfined compression", r"unconfined"),
    ("compaction", r"compaction|proctor"),
    ("corrosion", r"corrosi|resistivity|sulfate|sulphate|chloride"),
    ("moisture content", r"moisture content|water content"),
)
#: What a laboratory sheet's own first lines print.
_RE_LAB_SHEET = re.compile(
    r"laborator|sieve analysis|grain size|particle size|atterberg|"
    r"liquid limit|consolidation test|triaxial|direct shear|proctor|"
    r"compaction test|moisture content|unconfined compress", re.I)


def lab_said_not_held(texts: Sequence[str], labels: Mapping[int, str],
                      record: Any, bound: Sequence[int] = ()
                      ) -> List[Tuple[int, str]]:
    """``(page, the sentence or contents line)`` for each place the
    report's own pages say its laboratory results are in THIS document,
    when the record holds no laboratory test and the file no laboratory
    page; ``[]`` otherwise, and with fewer than :data:`MIN_MENTIONS` such
    places.

    "In this document" is a sentence about laboratory results or tests
    that says they are presented, included, attached, enclosed or the like
    in an appendix, an attachment or an enclosure
    (:data:`_RE_LAB_INCLUDED`), or a contents line titling an appendix with
    them (:data:`_RE_LAB_TITLE`). Results that will be, should be or are
    recommended to be done, that are under separate cover, by others or a
    previous investigation's (:data:`_RE_LAB_NOT_HERE`) are not, and a
    test the text only describes is not. The file holds a laboratory page
    when any page is labelled ``lab_test``, a bound document's included,
    when a page that is not the report's own text leads with a laboratory
    sheet's words, or when a bound page has no text to read.
    """
    if getattr(record, "lab_tests", None):
        return []
    bound = {int(p) for p in bound}
    for page, text in enumerate(texts):
        label = labels.get(page)
        if label == "lab_test":
            return []
        if page in bound and len((text or "").strip()) < _NO_TEXT:
            return []
        if label not in CITING_LABELS or page in bound:
            lead = "\n".join(_lead_lines(text))
            if _RE_LAB_SHEET.search(lead):
                return []
    said: List[Tuple[int, str]] = []
    for page, text in enumerate(texts):
        if page in bound or labels.get(page) not in CITING_LABELS:
            continue
        seen: set = set()
        for match in _RE_LAB_TITLE.finditer(text or ""):
            line = _flat(match.group(0)).strip()
            if not _RE_LAB_NOT_HERE.search(line) and line not in seen:
                seen.add(line)
                said.append((page, line))
        flat = _flat(text)
        for match in _RE_LAB_INCLUDED.finditer(flat):
            sentence = _sentence(flat, match.start(), match.end())
            if (_RE_LAB.search(sentence) and _RE_LAB_RESULTS.search(sentence)
                    and not _RE_LAB_NOT_HERE.search(sentence)
                    and sentence not in seen
                    and not any(sentence in line or line in sentence
                                for line in seen)):
                seen.add(sentence)
                said.append((page, sentence))
    return said if len(said) >= MIN_MENTIONS else []


def _lab_kinds(texts: Sequence[str], labels: Mapping[int, str],
               bound: Sequence[int] = ()) -> List[str]:
    """The tests the report's own pages name, in sentences that do not say
    they are planned or another document's."""
    bound = {int(p) for p in bound}
    found: set = set()
    for page, text in enumerate(texts):
        if page in bound or labels.get(page) not in CITING_LABELS:
            continue
        for sentence in _RE_SENTENCE_END.split(_flat(text)):
            if _RE_LAB_NOT_HERE.search(sentence):
                continue
            found |= {name for name, pattern in _LAB_KINDS
                      if re.search(pattern, sentence, re.I)}
    return [name for name, _pattern in _LAB_KINDS if name in found]


#: A contents line listing an appendix by letter and title; a dotted
#: leader and its page number are left off the title.
_RE_LISTED = re.compile(
    r"^\s*(?:APPENDIX|Appendix)\s+([A-Z]|\d{1,2})\b\s*[-–—:.]?\s*(.*?)"
    r"\s*(?:[.\s]{3,}\s*\S{1,6}|\s{2,}\S{1,6})?\s*$")
#: A listed appendix that is not in this document by its own line.
_RE_LISTED_NOT_HERE = re.compile(
    r"not used|not included|not applicable|reserved|omitted|"
    r"separate cover|separate(?:ly)? (?:bound|volume|report)|by others|"
    r"\bfuture\b|to be (?:provided|submitted)", re.I)
#: The page labels an appendix's title says its pages carry; the first
#: match is taken. A title that matches none is one the code cannot check
#: by content, and is never flagged.
_TITLE_CONTENT: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (r"\bCPT\b|cone penetr", ("cpt_log",)),
    (r"test pit", ("test_pit_log",)),
    (r"boring|bore\s?hole|\blogs?\b", LOG_LABELS),
    (r"laborator|\blab\b", ("lab_test",)),
    (r"calculation|computation", ("calculation",)),
    (r"photo", ("photos",)),
    (r"figures?\b|drawings?\b|\bplans?\b|\bmaps?\b", PICTURE_LABELS),
)
_SMALL_WORDS = frozenset(("of", "the", "and", "for", "a", "an", "to", "in",
                          "on", "with", "by"))


def _lead_lines(text: str) -> List[str]:
    return [line for line in (text or "").splitlines()
            if line.strip()][:_LEAD_LINES]


def _title_words(title: str) -> set:
    """A title's words, lower case, plurals and "-ing" folded, small words
    left out: "Laboratory Testing Results" and "LABORATORY TEST RESULT"."""
    out = set()
    for word in re.findall(r"[A-Za-z]{2,}", title.lower()):
        if word in _SMALL_WORDS:
            continue
        word = re.sub(r"(?:ing|s)$", "", word) if len(word) > 4 else word
        out.add(word)
    return out


def _leads_with_title(text: str, title: str) -> bool:
    """Does a page lead with a line alike ``title``: two thirds of the
    title's words or more, and at least two, on one of its first lines?"""
    want = _title_words(title)
    if len(want) < 2:
        return False
    for line in _lead_lines(text):
        if len(line) > 90:
            continue
        have = _title_words(line)
        if len(want & have) >= max(2, -(-2 * len(want) // 3)):
            return True
    return False


def listed_appendices_not_held(texts: Sequence[str],
                               labels: Mapping[int, str],
                               bound: Sequence[int] = ()
                               ) -> List[Tuple[str, str, int]]:
    """``(letter, title, the contents page)`` for each appendix a contents
    page (label ``toc``) lists by letter and title that the file plainly
    lacks, where it holds others of the list.

    An appendix is held when a page that is not a list leads a line with
    its letter (as :func:`cited_not_held` holds one), when a page leads
    with a line alike its title (two thirds of its words: "Laboratory Test
    Results" under "LABORATORY TESTING RESULT SHEETS"), or when a page
    labelled for its content (:data:`_TITLE_CONTENT`: logs for "Boring
    Logs", ``lab_test`` for "Laboratory Test Results") follows the last
    appendix held before it in the list. One whose title names no content
    the labels know is never flagged, nor one its own line says is not
    used or under separate cover. With no listed appendix held nothing is
    returned: that is :func:`cited_not_held`'s to say.
    """
    bound = {int(p) for p in bound}
    listed: List[Tuple[str, str, int]] = []
    last_toc = -1
    for page, text in enumerate(texts):
        if labels.get(page) != "toc" or page in bound:
            continue
        last_toc = page
        lines = (text or "").splitlines()
        for at, line in enumerate(lines):
            match = _RE_LISTED.match(line)
            if not match:
                continue
            letter, title = match.group(1).upper(), match.group(2).strip()
            if not title and at + 1 < len(lines) and \
                    not _RE_LISTED.match(lines[at + 1]):
                title = re.sub(r"\s*[.\s]{3,}\s*\S{1,6}\s*$", "",
                               lines[at + 1]).strip()
            if title and not any(letter == other for other, _t, _p in listed):
                listed.append((letter, title, page))
    if len(listed) < 2:
        return []

    by_letter: Dict[str, int] = {}
    for page, text in enumerate(texts):
        if labels.get(page) == "toc":
            continue
        captions = {(_kind(a), _norm(b))
                    for a, b in _RE_CAPTION.findall(text or "")}
        if len(captions) > MAX_CAPTIONS_ON_A_PAGE:
            continue
        for kind, name in captions:
            if kind == "appendix":
                by_letter.setdefault(name, page)
    held: Dict[str, int] = {}
    for letter, title, _page in listed:
        if _norm(letter) in by_letter:
            held[letter] = by_letter[_norm(letter)]
            continue
        for page, text in enumerate(texts):
            if labels.get(page) not in CITING_LABELS and \
                    _leads_with_title(text, title):
                held[letter] = page
                break
    if not held:
        return []

    out: List[Tuple[str, str, int]] = []
    after = last_toc
    for letter, title, page in listed:
        if letter in held:
            after = max(after, held[letter])
            continue
        if _RE_LISTED_NOT_HERE.search(title):
            continue
        content = next((found for pattern, found in _TITLE_CONTENT
                        if re.search(pattern, title, re.I)), None)
        if content is None:
            continue
        if any(labels.get(p) in content for p in range(after + 1, len(texts))):
            continue
        out.append((letter, title, page))
    return out


def _clip(text: str, n: int = _QUOTE) -> str:
    text = _flat(text).strip()
    return text if len(text) <= n else text[:n - 3].rstrip() + "..."


def named_not_held_qa(texts: Sequence[str],
                      labels: Optional[Mapping[int, str]], record: Any,
                      bound: Sequence[int] = ()) -> list:
    """The one ``partial`` QA entry, ranked ``open``, for what the report
    names that the record and the file do not hold (RI-185): explorations
    with no log, laboratory results, and appendices its contents list.
    ``[]`` when there is nothing to say, and without page labels (which
    pages are the report's own text cannot be told).

    ``texts`` and ``labels`` as :func:`completeness_qa` takes them;
    ``record`` is the report's own record after its readers; ``bound`` are
    the pages of the documents bound into the file, whose holes and sheets
    are their own records' and are not this report's to lack."""
    from report_ingest.model import QAEntry

    if not labels or record is None:
        return []
    holes = named_not_logged(texts, labels, record, bound)
    lab = lab_said_not_held(texts, labels, record, bound)
    appendices = listed_appendices_not_held(texts, labels, bound)
    if not (holes or lab or appendices):
        return []
    said: List[str] = []
    values: List[str] = []
    pages: set = set()
    if holes:
        names = sorted(holes, key=lambda s: (re.sub(r"\d+", "", s),
                                             len(s), s))
        where = sorted({p for at in holes.values() for p in at})
        said.append(
            f"the text names the exploration{'s' if len(names) > 1 else ''} "
            f"{_and(names)} ({_span(where)}), and the record holds no log "
            f"for {'them' if len(names) > 1 else 'it'}")
        values += [f"exploration {name}: named on {_span(holes[name])}"
                   for name in names]
        pages |= set(where)
    if lab:
        kinds = _lab_kinds(texts, labels, bound)
        where = sorted({page for page, _sentence_ in lab})
        said.append(
            f"the text says laboratory results are in this document "
            f"({_span(where)})"
            f"{', naming ' + _and(kinds) + ',' if kinds else ''} and the "
            f"record holds no laboratory page or test")
        values += [f"laboratory results, page {page + 1}: "
                   f"\"{_clip(sentence)}\"" for page, sentence in lab]
        pages |= set(where)
    if appendices:
        where = sorted({page for _l, _t, page in appendices})
        many = len(appendices) > 1
        said.append(
            f"the contents ({_span(where)}) list "
            f"{'Appendices' if many else 'Appendix'} "
            f"{_and([letter for letter, _t, _p in appendices])}, which the "
            f"file does not hold")
        values += [f"Appendix {letter}: \"{_clip(title)}\", listed on "
                   f"page {page + 1}" for letter, title, page in appendices]
        pages |= set(where)
    return [QAEntry(
        kind="partial", rank="open", where="document.completeness",
        detail="this copy looks incomplete: " + "; ".join(said),
        values=values, pages=sorted(pages))]


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
