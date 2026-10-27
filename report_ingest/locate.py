"""Placement by text match: where on its page a model's value is printed.

RI-87 piece 5 (``module_work/REPORT_INGEST_PLAN.md`` section 8.4). Under
model-first ingest the model gives each object its page and no box: it was
shown the pages, not the rows or cells a box could be copied from. This
module finds the box after the answer, from the words the page prints, in
planlens' one frame: the PDF text layer, or the Azure Document Intelligence
words on a scan (``Document.page(p, words=True)``).

1. **The page is checked** (:func:`page_for`). The pages one answer cites
   are read one way for the whole answer, by the builders' own rule
   (:func:`report_ingest.checks.pdf_local`): as the report's pages, or, with
   a note, all as positions in the pages the call was sent. A page outside
   the call on that reading becomes the call's first page at confidence
   0.4, with a note, as the builders do today. The note that an answer's
   cites mix the call's pages and others is the builder's (under
   ``checks="flag"``) and is not written again here.
2. **The object's printed words are found on its cited page.** Each kind of
   object has its anchors, tried in order, and the first that finds it once
   places it:

   - a log layer by its description's first words, then by its top depth
     printed on the row the depth ruler puts at that depth;
   - a sample or a drive by its label (or, where a form prints the label's
     letters and number in two columns, by the number on the sample's own
     rows), a drive then by its printed blow record, then either by its
     printed depth on the ruler's row; a sample not found by its label
     takes the box of the drive placed at its top on its page, since it
     came out of that drive, before its own printed depth;
   - a water reading by its printed depth on the line its note or a
     groundwater word names; one with no depth by the line its note names;
   - a laboratory test by its sample's label line, then by the row that
     prints its own results, then by its boring's label on a row that also
     prints its depth or one of its results (never a heading,
     :func:`boring_heading`); of several label lines, the one that names
     its boring, then prints its depth, then its own results; then by the
     sheet's title: a page-level box
     (:attr:`Placement.page_level`), but for the answer's only test on
     that page, whose place the sheet is;
   - a calculation value by its number where the printout around it (its
     line, its column's head, the headings above) carries its label; then
     where one place carries clearly more of the label than any other; then
     by its number where the page prints it once. A value that is words (a
     site class's letter) by those words on its label's line, or by the
     words themselves; never by its label alone;
   - a narrative answer by its quote, at the quote gate's own bar
     (:func:`report_ingest.narrative_reader.check_quotes`), boxed where
     planlens' search finds its first words; a quote the gate finds that
     search cannot box is placed on its page with no box (page level).

   A number is matched at its printed precision
   (:func:`report_ingest.calc_reader._precision_tol`), and never in a date,
   a time, a page's own numbering, a unit (``m3``) or a label (``S-3``).
   One candidate gives its box and the note "box from the text layer" (or
   "from DI words"); so do several along one printed line, with the line's
   box. Several on different lines give the page only and a note with the
   count, unless an anchor stands between them: on a log, the row the ruler
   puts at the object's depth, or the depth printed on the label's row. A
   page's rows are read after turning its words by the page's measured
   skew (:func:`page_skew`), so a scan's tilted line stays one row. A
   sheet scanned sideways (RI-149) is read in its upright frame, so its
   printed rows are rows here too, and the box found is given back in the
   page's frame as displayed (:class:`PageText`); a depth ruler is not
   asked on it.
3. **The call's other pages.** An object found by its own words (its
   label, description, name or quote; never by a number alone, a sheet's
   title or a boring's label) on exactly one other page of the call moves
   there, with a note. An object with no page at all (neither the call
   nor its record gives one) is counted, one ``no_page`` each.
4. **Not placed.** The object keeps its page and gets no box. On a page with
   text its confidence drops to :data:`NOT_FOUND_CONFIDENCE` and the item's
   QA note counts it ("N of M not found printed on the cited page"); on a
   page with no text nothing is lowered and the note says the page has no
   text. Nothing is ever removed.

Nothing in ingest calls this module yet (piece 7 will); it changes only the
records it is handed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple,
)

from report_ingest.calc_reader import _NUMBER_RE, _number, _precision_tol
from report_ingest.checks import PageNumbers, pdf_local
from report_ingest.lab_floor import TITLE_KINDS
from report_ingest.narrative_reader import (
    DEFAULT_CONFIDENCE, QUOTE_RATIO, _fold, _partial_ratio, check_quotes,
)
from report_ingest.turned import displayed_box, page_turn, upright_box

BBox = Tuple[float, float, float, float]

#: What a value is worth when its page prints text and its words were not
#: found there (section 8.4 step 4).
NOT_FOUND_CONFIDENCE = 0.6
#: What a value is worth when it cites a page outside its call (step 1).
OUTSIDE_CONFIDENCE = 0.4
#: The fuzzy score a text anchor needs: the quote gate's own bar.
TEXT_MIN_SCORE = QUOTE_RATIO
#: The share of a label's words (:func:`label_score`) a line must carry to
#: be the label's line. A label is the model's wording of the printed one
#: ("allowable bearing pressure" for "Allowable Bearing"), often with the
#: block or scenario it belongs to added, so half is the bar.
LABEL_MIN_SCORE = 50.0
#: How far ahead of the next line the best line's score must be for the
#: label to pick it; closer than this, both are candidates.
LABEL_MARGIN = 5.0
#: How many heading rows above a value are read as its headings and column
#: heads, and how many rows up (past rows of values) they are looked for.
CONTEXT_ROWS = 6
LOOKBACK_ROWS = 40
#: How far the best place's label score must lead the next for a value
#: printed in several places to take it, when no place carries half its
#: label.
LABEL_LEAD = 20.0
#: What a label word counts for when the printed word abbreviates it or is
#: it misread by a letter or two.
NEAR_WORD = 0.8
#: What a number in a label ("Layer 7", "Scenario 2") weighs: a row's or a
#: block's name, worth a word.
NUMBER_WEIGHT = 3.0
#: How many leading words of a description or a text value are searched
#: for. A layer's description runs over several printed lines and blocks; its
#: first words are on one.
LEAD_WORDS = 6
#: Points a candidate may sit above or below the row the ruler puts at its
#: depth and still be on that row (about one line of log text).
ROW_SLACK = 12.0
#: The steepest skew (degrees) a page's words are measured for, and the
#: least that turns them. Below it a page's rows are read as printed: the
#: measure reads an upright page up to about 0.2 degrees off, and at 0.3 a
#: word across a letter page sits under 3 points off its line, inside the
#: row's band.
SKEW_MAX_DEG = 3.0
SKEW_MIN_DEG = 0.3
#: How many points of each word's middle mark the page's lines for the skew
#: measure: a word's box is upright on a tilted line, so only its middle is
#: on the line.
SKEW_MARK = 5
#: A searched phrase this short (in characters) must match at
#: :data:`SHORT_MIN_SCORE`, not the quote bar.
SHORT_PHRASE = 20
SHORT_MIN_SCORE = 95.0
#: A text value this short (in characters) is looked for only beside its
#: label: printed alone it is a letter or a word any page prints.
SHORT_TEXT = 3

#: The statuses a placement ends in. ``placed`` and ``moved`` give a box
#: (but for a quote placed at page level).
PLACED, MOVED, SEVERAL, NOT_FOUND, NO_TEXT, NO_PAGE = (
    "placed", "moved", "several", "not_found", "no_text", "no_page")
STATUSES = (PLACED, MOVED, SEVERAL, NOT_FOUND, NO_TEXT, NO_PAGE)

#: The anchors that read an object's own words: its label, description,
#: name or quote. Only these move an object to another page of the call
#: (step 3); a number, a blow record or a row of result values printed on
#: another page, a sheet's title, or a boring's label (which every test
#: of that boring shares) says nothing about which object it is.
WORD_ANCHORS = frozenset((
    "label", "description", "sample label",
    "label and number", "number by its label", "label and text",
    "printed text", "quote", "quote on its page"))
#: The anchors that find an object's page, not its place on it: a sheet's
#: title, and a quote the gate finds on the page that search cannot box.
PAGE_ANCHORS = frozenset(("test title", "quote on its page"))

#: What the note says the box came from, by planlens' text source.
_SOURCE_WORDS = {"pdf_text": "the text layer", "azure_di": "DI words",
                 "ocr": "OCR words"}

#: Words that name a groundwater reading on a log.
_WATER_WORDS = re.compile(
    r"water|\bgw|\bwl\b|seepage|encountered|cave|phreatic|\bdry\b|"
    r"[▽▼∇]", re.I)

#: The characters a printed minus can be (calc_reader's number pattern).
_SIGNS = "-+‐‑‒–−"


# ---------------------------------------------------------------------------
# step 1: the page
# ---------------------------------------------------------------------------

def answer_pages(provs: Iterable[Any], pages: Sequence[int]) -> PageNumbers:
    """How one answer's cited pages are read, decided once for all of them
    (:func:`report_ingest.checks.pdf_local`): ``provs`` are the provenances
    of every object the answer gives (a log answer's every investigation),
    as the record holds them now. A record a ``flag`` builder made holds the
    report's pages already, and they are read as cited; reading the raw
    answer's cites instead would turn its positions twice."""
    return pdf_local([getattr(p, "page", None) for p in provs
                      if p is not None], pages)


def page_for(cited: Optional[int], pages: Sequence[int],
             numbers: PageNumbers) -> Tuple[int, str, Optional[float]]:
    """Section 8.4 step 1 for one cited page: ``(page, note, confidence)``.

    ``numbers`` says how the whole answer's cites are read
    (:func:`answer_pages`). A cite that reading puts on one of the call's
    pages stands, with the note that says it was read as a position when it
    was. Any other is a page the model was not shown: the value is kept, on
    the call's first page, at :data:`OUTSIDE_CONFIDENCE`. ``confidence`` is
    None when it is not lowered.
    """
    pages = [int(p) for p in pages]
    if not pages:
        raise ValueError("page_for needs the call's pages")
    if cited is not None:
        page, said = numbers.page(int(cited))
        if page in pages:
            return page, said, None
    said = "no page" if cited is None else f"page {int(cited)}"
    return (pages[0],
            f"the answer cited {said}, which is not one of the pages it was "
            f"sent ({_span(pages)}); kept on page {pages[0]} at confidence "
            f"{OUTSIDE_CONFIDENCE}", OUTSIDE_CONFIDENCE)


def _span(pages: Sequence[int]) -> str:
    pages = sorted(int(p) for p in pages)
    if len(pages) > 2 and pages == list(range(pages[0], pages[-1] + 1)):
        return f"{pages[0]}-{pages[-1]}"
    return ", ".join(str(p) for p in pages)


# ---------------------------------------------------------------------------
# the words a page prints
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Word:
    """One printed word with its box, folded for comparison, and the numbers
    it prints, each with half its last printed digit's place."""

    text: str
    bbox: BBox
    fold: str
    numbers: Tuple[Tuple[float, float], ...] = ()

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    def prints(self, value: Optional[float], first: bool = False) -> bool:
        """Whether this word prints ``value`` (only as its first number with
        ``first``: a depth is printed on its own or as a range's top, not as
        the third count of a blow record)."""
        if value is None or not self.numbers:
            return False
        numbers = self.numbers[:1] if first else self.numbers
        return any(same_number(value, n, tol) for n, tol in numbers)


@dataclass
class Row:
    """The words printed along one line of the page, left to right: a row
    of a form or a table runs across its columns, which a text layer's own
    lines often split."""

    words: List[Word] = field(default_factory=list)

    @property
    def bbox(self) -> BBox:
        return _union([w.bbox for w in self.words])

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class PageText:
    """One page's words and rows, where its text came from, and the skew its
    rows were read at (dy/dx, :func:`page_skew`).

    ``turn``: on a sheet scanned sideways (:func:`report_ingest.turned.
    page_turn`, RI-149), its turn, and every word's box is in the upright
    frame (:func:`report_ingest.turned.upright_box`), so its rows run across
    the page as the sheet prints them; :meth:`PageWords.displayed` takes a
    box back to the frame the record keeps. None on an upright page."""

    page: int
    source: str = ""
    words: List[Word] = field(default_factory=list)
    rows: List[Row] = field(default_factory=list)
    skew: float = 0.0
    turn: Any = None

    @property
    def has_text(self) -> bool:
        return bool(self.words)


def _printed_tol(token: str) -> float:
    """Half of the last printed digit's place, from the token as printed:
    ``6.30`` says 6.295 to 6.305, ``20`` says 19.5 to 20.5."""
    parts = re.split(r"[eE]", token)
    mantissa = parts[0]
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    exponent = 0
    if len(parts) > 1:                           # 1.5E-06: the exponent's
        exponent = int(_number(parts[1]) or 0)   # place, too
    return 0.5 * (10.0 ** max(-300, min(300, exponent - decimals)))


def numbers_in(text: str) -> Tuple[Tuple[float, float], ...]:
    """``(number, half its last printed digit's place)`` for every number a
    printed word carries, as calc_reader reads one.

    A dash after a digit is a range's (``5.0-6.5``, ``5-9-12``), not a
    minus, so the number after it is read unsigned. Digits straight after
    a letter, or after a letter and a dash (``m3``, ``N60``, ``S-3``), are
    a unit's or a label's, not a number printed. A comma followed by one
    or two digits (``12,5``) is also read as a decimal comma, ahead of the
    two numbers it otherwise is: the numbers come in printed order, and
    the first of ``1,50-1,95`` is 1.50.
    """
    cleaned = str(text or "").replace("\u00a0", " ")
    # (where its digits start, decimal comma first, number, tolerance)
    found: List[Tuple[int, int, float, float]] = []
    for match in _NUMBER_RE.finditer(cleaned):
        token = match.group(0)
        value = _number(token)
        if value is None:
            continue
        before = cleaned[match.start() - 1] if match.start() else ""
        if before.isalpha():
            continue                  # m3, N60, S-3: a unit's or a label's
        signed = token[0] in _SIGNS
        if signed and before and (before.isalnum() or before in ").'\""):
            value = abs(value)
        found.append((match.start() + signed, 1, value,
                      _printed_tol(token.replace(",", ""))))
    for match in re.finditer(r"(?<![\d,.])(\d+),(\d{1,2})(?![\d,])", cleaned):
        found.append((match.start(), 0,
                      float(f"{match.group(1)}.{match.group(2)}"),
                      0.5 * (10.0 ** -len(match.group(2)))))
    return tuple((value, tol) for _, _, value, tol in sorted(found))


def same_number(value: float, printed: float,
                printed_tol: Optional[float] = None) -> bool:
    """``printed`` is ``value`` at the precision both were given to.

    The calculation result check's own test (``_on_the_page``) holds the
    printed number to the precision of the value reported; given the
    printed number's own precision too, the stricter of the two holds, so a
    page printing 2.45 does not print a reported 2.5."""
    tol = _precision_tol(float(value))
    if printed_tol is not None:
        tol = min(tol, float(printed_tol))
    return abs(float(printed) - float(value)) <= tol


def _fold_token(text: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", str(text or "").lower())


def _union(boxes: Sequence[Sequence[float]]) -> BBox:
    boxes = [b for b in boxes if b is not None]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def page_skew(words: Sequence[Word]) -> float:
    """The slope (dy/dx) the page's printed lines run at, measured by
    planlens' own skew measure (:func:`planlens.document.raster.
    measure_skew`) on a mask marking each word's middle
    (:data:`SKEW_MARK` points across); 0 below :data:`SKEW_MIN_DEG`, or
    when there is too little to measure."""
    if len(words) < 2:
        return 0.0
    try:
        import numpy as np
        from planlens.document.raster import measure_skew
    except ImportError:                          # no raster extra installed
        return 0.0
    width = int(max(w.bbox[2] for w in words)) + 2
    height = int(max(w.bbox[3] for w in words)) + 2
    if width <= 2 or height <= 2:
        return 0.0
    mask = np.zeros((height, width), dtype=bool)
    for w in words:
        y = int(round(w.cy))
        x = int(round((w.bbox[0] + w.bbox[2]) / 2.0))
        if 0 <= y < height:
            mask[y, max(0, x - SKEW_MARK // 2):x + SKEW_MARK // 2 + 1] = True
    slope = float(measure_skew(mask, max_deg=SKEW_MAX_DEG))
    return slope if abs(math.degrees(math.atan(slope))) >= SKEW_MIN_DEG \
        else 0.0


def _rows_of(words: Sequence[Word], skew: float = 0.0) -> List[Row]:
    """The page's words grouped into printed rows, top to bottom.

    A word joins a row when its middle lies within the band of the row's
    first word and that word's middle lies within its own band, so a word
    set on its side (a turned column head, tall and narrow) does not gather
    every row it spans. The bands are read with the page's ``skew``
    (dy/dx) taken out, so a scan's tilted line is one row; the words keep
    their own boxes.
    """
    def band(w: Word) -> Tuple[float, float, float]:
        shift = skew * (w.bbox[0] + w.bbox[2]) / 2.0
        top, bottom = w.bbox[1] - shift, w.bbox[3] - shift
        return top, bottom, (top + bottom) / 2.0

    bands = {id(w): band(w) for w in words}
    rows: List[Row] = []
    for word in sorted(words, key=lambda w: (bands[id(w)][2], w.bbox[0])):
        top, bottom, middle = bands[id(word)]
        home = None
        for row in reversed(rows[-6:]):
            s_top, s_bottom, s_middle = bands[id(row.words[0])]
            if s_top <= middle <= s_bottom and top <= s_middle <= bottom:
                home = row
                break
        if home is None:
            rows.append(Row([word]))
        else:
            home.words.append(word)
    for row in rows:
        row.words.sort(key=lambda w: w.bbox[0])
    return rows


#: A word that prints a date or a time of day: no value of a log, a test or
#: a calculation, however its digits read.
_NOT_A_VALUE = re.compile(
    r"^\(?(\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}(:\d{2})?)"
    r"[,.;)]?$")
#: The words that number a page ("Page 3 of 7", "Sheet 2").
_NUMBERING = ("page", "pg", "sheet")


def _page_numbering(rows: Sequence[Row]) -> List[Word]:
    """The numbers of "Page 3 of 7" and "Sheet 2": the page's own
    numbering, not values printed on it."""
    out: List[Word] = []
    for row in rows:
        ws = row.words
        for i, w in enumerate(ws[:-1]):
            if w.fold not in _NUMBERING or not ws[i + 1].numbers:
                continue
            out.append(ws[i + 1])
            if i + 3 < len(ws) and ws[i + 2].fold == "of" and \
                    ws[i + 3].numbers:
                out.append(ws[i + 3])
    return out


class PageWords:
    """The words of a document's pages, read once each.

    ``doc`` is a planlens ``Document``, opened with its DI text source where
    the report has one (``Corpus.open_report(..., di="auto")`` and the
    ingest graph both do), so a scan's words are DI's in the same frame.
    """

    def __init__(self, doc: Any) -> None:
        self.doc = doc
        self._pages: Dict[int, PageText] = {}

    def __call__(self, page: int) -> PageText:
        page = int(page)
        if page not in self._pages:
            self._pages[page] = self._read(page)
        return self._pages[page]

    def _read(self, page: int) -> PageText:
        try:
            content = self.doc.page(page, words=True, tables=False)
        except Exception:                        # a page that will not read
            return PageText(page)
        turn = _turn_of(content)
        words: List[Word] = []
        sources: List[str] = []
        for line in content.lines:
            source = str(getattr(line, "source", "") or "")
            if source and source not in sources:
                sources.append(source)
            parts = list(line.words or []) or [(line.text, line.bbox)]
            for text, bbox in parts:
                text = str(text or "").strip()
                if not text or bbox is None:
                    continue
                box = tuple(float(v) for v in bbox)
                if turn is not None:
                    box = upright_box(box, turn)
                words.append(Word(text=text, bbox=box,
                                  fold=_fold_token(text),
                                  numbers=() if _NOT_A_VALUE.match(text)
                                  else numbers_in(text)))
        skew = page_skew(words)
        rows = _rows_of(words, skew)
        numbering = _page_numbering(rows)
        if numbering:
            plain = {id(w): replace(w, numbers=()) for w in numbering}
            words = [plain.get(id(w), w) for w in words]
            for row in rows:
                row.words = [plain.get(id(w), w) for w in row.words]
        return PageText(page, sources[0] if sources else "", words, rows,
                        skew, turn)

    def displayed(self, page: int, bbox: Optional[BBox]) -> Optional[BBox]:
        """A box found on ``page`` in the frame the record keeps: the page's
        as displayed (a sheet scanned sideways is searched in its upright
        frame, :class:`PageText`)."""
        turn = self(page).turn
        if bbox is None or turn is None:
            return bbox
        return displayed_box(bbox, turn)

    def search(self, phrase: str, page: int,
               min_score: float = TEXT_MIN_SCORE) -> List[Dict[str, Any]]:
        """planlens' fuzzy search for ``phrase`` on one page: one hit per
        text block or line it matches, best first, each with its box (in the
        page's words' frame, :class:`PageText`)."""
        if not str(phrase or "").strip():
            return []
        try:
            got = self.doc.search(phrase, pages=[int(page)], fuzzy=True,
                                  min_score=int(min_score),
                                  include_markups=False, max_hits=50,
                                  context_chars=0)
        except Exception:                        # a page that will not read
            return []
        hits = [h for h in got.get("hits") or [] if h.get("bbox")]
        turn = self(page).turn
        if turn is None:
            return hits
        return [{**h, "bbox": list(upright_box(h["bbox"], turn))}
                for h in hits]


def _turn_of(content: Any) -> Any:
    """A page's turn where its text runs a quarter turn from the page
    (:func:`report_ingest.turned.page_turn`), else None."""
    try:
        turn = page_turn(list(content.lines),
                         float(getattr(content, "width", 612.0) or 612.0),
                         float(getattr(content, "height", 792.0) or 792.0))
    except Exception:                            # lines that will not read
        return None
    return turn if turn.turned else None


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One place on a page where an anchor was found (``bbox`` None: the
    page, with no place on it to box)."""

    bbox: Optional[BBox]
    score: float = 100.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0


def number_rows(text: PageText, value: Optional[float], first: bool = False
                ) -> List[Tuple[Row, Word]]:
    """Every word printing ``value`` at its printed precision (as its first
    number, with ``first``), with its row."""
    if value is None:
        return []
    return [(row, w) for row in text.rows for w in row.words
            if w.prints(value, first)]


def number_hits(text: PageText, value: Optional[float], first: bool = False
                ) -> List[Candidate]:
    """Every word printing ``value`` at its printed precision."""
    return [Candidate(w.bbox) for _, w in number_rows(text, value, first)]


def _run_hits(row: Row, wanted: str) -> List[Tuple[int, int]]:
    """``(first, last)`` word indexes of each run of up to three words in
    ``row`` whose folded text is ``wanted``."""
    out: List[Tuple[int, int]] = []
    for i in range(len(row.words)):
        joined = ""
        for j in range(i, min(i + 3, len(row.words))):
            joined += row.words[j].fold
            if joined == wanted:
                out.append((i, j))
                break
            if not wanted.startswith(joined):
                break
    return out


def label_hits(text: PageText, label: str) -> List[Tuple[Row, Candidate]]:
    """Every place a label (``S-3``, ``B-12``, ``SS 4``) is printed as a word
    or a run of up to three words, with the row it is on."""
    wanted = _fold_token(label)
    if not wanted:
        return []
    out: List[Tuple[Row, Candidate]] = []
    for row in text.rows:
        for i, j in _run_hits(row, wanted):
            out.append((row, Candidate(_union([w.bbox for w in
                                               row.words[i:j + 1]]))))
    return out


def _row_prints(row: Row, value: Optional[float]) -> bool:
    return any(w.prints(value) for w in row.words)


def _row_has_label(row: Row, label: str) -> bool:
    wanted = _fold_token(label)
    return bool(wanted) and bool(_run_hits(row, wanted))


def lead(text: str, n: int = LEAD_WORDS) -> str:
    """The first ``n`` words of a text, the part a search looks for."""
    return " ".join(str(text or "").split()[:n])


def text_hits(words: PageWords, page: int, phrase: str,
              min_score: float = TEXT_MIN_SCORE) -> List[Candidate]:
    """The text blocks or lines where ``phrase``'s first words are printed:
    the best-scoring ones, if several score alike. A short phrase must
    match closely (:data:`SHORT_MIN_SCORE`): in "Unit B2" one wrong
    character is the whole difference from "Unit B3"."""
    wanted = lead(phrase)
    if len(wanted) <= SHORT_PHRASE:
        min_score = max(min_score, SHORT_MIN_SCORE)
    hits = words.search(wanted, page, min_score)
    if not hits:
        return []
    best = max(float(h.get("score") or 0.0) for h in hits)
    return [Candidate(tuple(float(v) for v in h["bbox"]),
                      float(h.get("score") or 0.0))
            for h in hits if float(h.get("score") or 0.0) >= best - 1.0]


#: Words any label may share with any line, which say nothing about which.
_STOPWORDS = frozenset((
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "of", "on",
    "or", "per", "the", "to", "with"))


def _same_word(want: str, have: str) -> float:
    """One label word against one printed word: 1 when they are the same
    word; :data:`NEAR_WORD` when one abbreviates the other (``coeff`` for
    ``coefficient``) or is the other with a letter or two misread; else 0."""
    if want == have:
        return 1.0
    if len(want) < 4 or len(have) < 3:
        return 0.0
    if want.startswith(have) or have.startswith(want):
        return NEAR_WORD
    if len(want) < 5:
        return 0.0
    if _partial_ratio(want, have) >= TEXT_MIN_SCORE and \
            abs(len(want) - len(have)) <= 2:
        return NEAR_WORD
    return 0.0


def _found(word: str, tokens: Sequence[str]) -> float:
    return max((_same_word(word, t) for t in tokens), default=0.0)


def _label_words(label: str) -> List[Tuple[str, float]]:
    """A label's words (less the words any line shares, ``of``, ``the``),
    each weighted by its length up to six letters, so ``footing``,
    ``width`` and ``B`` outweigh ``ft``; a number ("Layer 7", "Scenario 2")
    weighs :data:`NUMBER_WEIGHT`."""
    out = []
    for word in dict.fromkeys(_fold(label).split()):
        if word in _STOPWORDS:
            continue
        out.append((word, NUMBER_WEIGHT if word.isdigit()
                    else float(min(len(word), 6))))
    return out


def label_score(label: str, line: str, context: str = "") -> float:
    """How much of a label (a water reading's note, a value's name) a
    printed line carries, 0 to 100: the share of the label's weighted words
    (:func:`_label_words`) the line prints, a word found only in
    ``context`` (the headings above the line) counting half."""
    wanted = _label_words(label)
    if not wanted:
        return 0.0
    here = _fold(line).split()
    above = _fold(context).split()
    total = got = 0.0
    for word, weight in wanted:
        total += weight
        got += weight * max(_found(word, here), _found(word, above) / 2.0)
    return 100.0 * got / total


# ---------------------------------------------------------------------------
# the ruler: the anchor between several candidates on a log
# ---------------------------------------------------------------------------

def _unit(unit: Any) -> str:
    unit = str(unit or "").strip().lower().rstrip(".")
    if unit in ("ft", "feet", "foot", "'"):
        return "ft"
    if unit in ("m", "metre", "metres", "meter", "meters"):
        return "m"
    return unit


def ruler_for(rulers: Optional[Dict[int, Any]], page: int,
              unit: str, words: Optional["PageWords"] = None
              ) -> Optional[Any]:
    """The page's depth ruler, when its unit is the log's (or unknown).
    None on a sheet scanned sideways (``words``' page has a turn): its
    ruler was read in the displayed frame, its words are searched in the
    upright one (:class:`PageText`)."""
    if words is not None and words(page).turn is not None:
        return None
    ruler = (rulers or {}).get(int(page))
    if ruler is None or not getattr(ruler, "slope", 0.0):
        return None
    theirs, ours = _unit(getattr(ruler, "unit", "")), _unit(unit)
    if theirs and ours and theirs != ours:
        return None
    return ruler


def on_ruler_row(cands: Sequence[Candidate], ruler: Any,
                 top: Optional[float], bottom: Optional[float] = None
                 ) -> List[Candidate]:
    """The candidates on the rows the ruler puts between ``top`` and
    ``bottom`` (or on ``top``'s row), nearest ``top`` first: the one whose
    own line the ruler's ``top`` crosses, when only one is; else the nearest
    alone when it is clearly the nearest. All of them when there is no
    ruler to ask."""
    if ruler is None or top is None:
        return list(cands)
    y_top = ruler.y_at(float(top))
    if y_top is None:
        return list(cands)
    crossed = [c for c in cands if c.bbox[1] - 1.0 <= y_top <= c.bbox[3] + 1.0]
    if len(crossed) == 1:
        return crossed
    y_bottom = ruler.y_at(float(bottom)) if bottom is not None else None
    lo, hi = sorted((y_top, y_top if y_bottom is None else y_bottom))
    inside = [c for c in cands
              if lo - ROW_SLACK <= c.cy <= hi + ROW_SLACK]
    inside.sort(key=lambda c: abs(c.cy - y_top))
    if len(inside) > 1 and (abs(inside[1].cy - y_top)
                            - abs(inside[0].cy - y_top)) >= ROW_SLACK / 2:
        return inside[:1]
    return inside


# ---------------------------------------------------------------------------
# placing one object
# ---------------------------------------------------------------------------

@dataclass
class Placement:
    """Where one object of an answer was placed, and how.

    ``status``: ``placed`` (found once on its cited page), ``moved`` (found
    on one other page of the call), ``several`` (printed more than once and
    no anchor between them), ``not_found`` (its page prints text, but not
    its words), ``no_text`` (its page prints none) or ``no_page`` (it has
    no page to place it on). ``page_level``: placed by an anchor that finds
    the page, not the object's place on it (:data:`PAGE_ANCHORS`); its box,
    if any, is the sheet title's.
    """

    what: str
    kind: str
    cited: Optional[int]
    page: Optional[int]
    status: str
    bbox: Optional[BBox] = None
    anchor: str = ""
    source: str = ""
    candidates: int = 0
    note: str = ""
    page_level: bool = False

    @property
    def boxed(self) -> bool:
        return self.status in (PLACED, MOVED) and self.bbox is not None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "what": self.what, "kind": self.kind, "cited": self.cited,
            "page": self.page, "status": self.status,
            "bbox": ([round(v, 1) for v in self.bbox] if self.bbox
                     else None),
            "anchor": self.anchor, "source": self.source}
        if self.page_level:
            out["page_level"] = True
        if self.candidates > 1:
            out["candidates"] = self.candidates
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class Placements:
    """Every placement of one item, and the item's QA note. ``notes`` are
    the item's own further lines (objects with no provenance to write
    to)."""

    items: List[Placement] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for item in self.items:
            out[item.status] = out.get(item.status, 0) + 1
        return out

    def qa_note(self) -> str:
        """The item's one QA line: how many of its values were not found
        printed on their cited pages, and the other outcomes that are not a
        plain placement. Empty when every value was placed where cited."""
        counts = self.counts()
        total = len(self.items)
        parts = []
        if counts[NOT_FOUND]:
            parts.append(f"{counts[NOT_FOUND]} of {total} not found printed "
                         f"on the cited page (kept at confidence "
                         f"{NOT_FOUND_CONFIDENCE})")
        if counts[SEVERAL]:
            parts.append(f"{counts[SEVERAL]} of {total} printed in several "
                         f"places on the page, so given the page and no box")
        if counts[MOVED]:
            parts.append(f"{counts[MOVED]} of {total} found printed on "
                         f"another page of the call and moved there")
        page_level = sum(1 for p in self.items if p.page_level)
        if page_level:
            parts.append(f"{page_level} of {total} placed on the page only "
                         f"(by the sheet's title, or a quote found on the "
                         f"page that search could not box)")
        if counts[NO_TEXT]:
            parts.append(f"{counts[NO_TEXT]} of {total} on a page with no "
                         f"text layer and no DI text, so not placed")
        if counts[NO_PAGE]:
            parts.append(f"{counts[NO_PAGE]} of {total} with no page to "
                         f"place it on")
        return "; ".join(parts + [n for n in self.notes if n])

    def to_dict(self) -> Dict[str, Any]:
        return {"counts": self.counts(), "note": self.qa_note(),
                "items": [p.to_dict() for p in self.items]}


#: An anchor: its name, and what it finds on one page of a document.
Finder = Callable[["PageWords", int], List[Candidate]]
Anchor = Tuple[str, Finder]


def one_line(cands: Sequence[Candidate]) -> Optional[Candidate]:
    """Several candidates along one printed line (a value printed in two
    columns of one row, a label beside its own depth) are that line: their
    joint box. None when they are on different lines."""
    if len(cands) < 2 or any(c.bbox is None for c in cands):
        return None
    if max(c.bbox[1] for c in cands) >= min(c.bbox[3] for c in cands):
        return None
    return Candidate(_union([c.bbox for c in cands]),
                     min(c.score for c in cands))


def _on_page(words: PageWords, page: int, anchors: Sequence[Anchor]
             ) -> Optional[Tuple[str, Candidate, str, int]]:
    """``(PLACED, candidate, anchor, 1)`` from the first anchor that finds
    one candidate on the page, or several along one line; ``(SEVERAL,
    first, anchor, n)`` when the anchors found only several; None when none
    found anything."""
    if not words(page).has_text:
        return None
    several: Optional[Tuple[str, Candidate, str, int]] = None
    for name, find in anchors:
        cands = find(words, page)
        line = one_line(cands)
        if line is not None:
            return PLACED, line, name, 1
        if len(cands) == 1:
            return PLACED, cands[0], name, 1
        if cands and several is None:
            several = (SEVERAL, cands[0], name, len(cands))
    return several


def place(what: str, kind: str, cited: int, pages: Sequence[int],
          words: PageWords, anchors: Sequence[Anchor]) -> Placement:
    """Steps 2 to 4 for one object: its cited page, then the call's other
    pages, over its anchors in order (on the other pages, only those that
    read its own words: :data:`WORD_ANCHORS`)."""
    cited = int(cited)
    here = words(cited)
    got = _on_page(words, cited, anchors)
    if got is not None and got[0] == PLACED:
        return Placement(what, kind, cited, cited, PLACED,
                         words.displayed(cited, got[1].bbox),
                         got[2], here.source, 1,
                         _found_note(got[1], got[2], here.source),
                         got[2] in PAGE_ANCHORS)
    if got is not None:
        # Printed on its cited page, more than once: found where it was
        # cited, so the call's other pages are not asked.
        return Placement(what, kind, cited, cited, SEVERAL, None, got[2],
                         here.source, got[3],
                         f"printed in {got[3]} places on page {cited} with "
                         f"nothing to choose between them; the page is kept "
                         f"and no box given")
    own_words = [a for a in anchors if a[0] in WORD_ANCHORS]
    found_on: List[Tuple[int, Candidate, str, str]] = []
    for page in pages:
        if int(page) == cited or not own_words:
            continue
        hit = _on_page(words, int(page), own_words)
        if hit is not None and hit[0] == PLACED:
            found_on.append((int(page), hit[1], hit[2],
                             words(int(page)).source))
    if len(found_on) == 1:
        page, cand, name, source = found_on[0]
        return Placement(what, kind, cited, page, MOVED,
                         words.displayed(page, cand.bbox), name, source, 1,
                         f"not found printed on page {cited}; found on page "
                         f"{page} of the same call and moved there; "
                         f"{_found_note(cand, name, source)}",
                         name in PAGE_ANCHORS)
    if not here.has_text:
        return Placement(what, kind, cited, cited, NO_TEXT, None, "", "", 0,
                         f"page {cited} has no text layer and no DI text, so "
                         f"nothing on it could be placed")
    elsewhere = (f" (found on {len(found_on)} other pages of the call)"
                 if found_on else "")
    return Placement(what, kind, cited, cited, NOT_FOUND, None, "",
                     here.source, 0,
                     f"not found printed on page {cited}{elsewhere}; kept at "
                     f"confidence {NOT_FOUND_CONFIDENCE}")


def _box_note(source: str) -> str:
    return f"box from {_SOURCE_WORDS.get(source, source or 'the page text')}"


def _found_note(cand: Candidate, anchor: str, source: str) -> str:
    if cand.bbox is None:
        return "found on the page by the quote check; search could not box it"
    if anchor in PAGE_ANCHORS:
        return f"page only: {_box_note(source)} for the sheet's title"
    return _box_note(source)


def _join(*notes: str) -> str:
    return "; ".join(n for n in (str(x or "").strip() for x in notes) if n)


def apply(prov: Any, placement: Placement) -> None:
    """Put a placement on a record's provenance (a ``Provenance``).

    A placement moves the page and sets the box; the method stays the
    model's. A value not found on a page with text drops to
    :data:`NOT_FOUND_CONFIDENCE`; one printed in several places, or on a page
    with no text, keeps its confidence. Every outcome writes its note. The
    value itself is never touched.
    """
    if prov is None:
        return
    if placement.status in (PLACED, MOVED):
        prov.page = int(placement.page)
        prov.bbox = placement.bbox
    else:
        prov.bbox = None
        if placement.status == NOT_FOUND:
            prov.confidence = min(float(prov.confidence),
                                  NOT_FOUND_CONFIDENCE)
    prov.note = _join(prov.note, placement.note)


def _checked(prov: Any, pages: Sequence[int], numbers: PageNumbers,
             write: bool) -> int:
    """Step 1 on a record's provenance: the page it is placed on."""
    page, note, confidence = page_for(getattr(prov, "page", None), pages,
                                      numbers)
    if write and prov is not None and note:
        prov.page = page
        prov.note = _join(prov.note, note)
        if confidence is not None:
            prov.confidence = min(float(prov.confidence), confidence)
    return page


def _value(quantity: Any) -> Optional[float]:
    value = getattr(quantity, "value", quantity)
    return float(value) if isinstance(value, (int, float)) and not \
        isinstance(value, bool) else None


def _show(value: Optional[float]) -> str:
    return "?" if value is None else f"{value:g}"


# ---------------------------------------------------------------------------
# a log
# ---------------------------------------------------------------------------

def _printed_depth(value: Optional[float], unit: str,
                   rulers: Optional[Dict[int, Any]]) -> Anchor:
    """The depth printed on the row the ruler puts at it; anywhere on a page
    with no ruler. A word prints a depth as its first number: ``7.5`` or the
    top of ``7.5-9.0``, not the last count of ``4-7-9``."""
    def find(words: PageWords, page: int) -> List[Candidate]:
        return on_ruler_row(number_hits(words(page), value, first=True),
                            ruler_for(rulers, page, unit, words), value)
    return "printed depth", find


#: The number that ends a sample's label (``1`` of ``SS-1``, ``12a`` of
#: ``ST-12A``): what a form prints in its sample-number column when the
#: sampler's letters sit in a column of their own.
_LABEL_NUMBER = re.compile(r"[a-z]+([0-9]+[a-z]?)$")


def _labelled(label: str, top: Optional[float], bottom: Optional[float],
              unit: str, rulers: Optional[Dict[int, Any]]) -> Anchor:
    """A sample's or a drive's label; of several, the one on the rows the
    ruler puts at the sample, else the one whose row prints its depth (a
    small depth is any row's number; the ruler is asked first). A label
    printed nowhere
    whole (``SS-1`` as ``SS`` in one column and ``1`` in the next) is looked
    for by its number, on the rows the ruler puts between the sample's top
    and bottom only: a bare number anywhere else says nothing."""
    split = _LABEL_NUMBER.match(_fold_token(label))

    def find(words: PageWords, page: int) -> List[Candidate]:
        text = words(page)
        hits = label_hits(text, label)
        if len(hits) == 1:
            return [hits[0][1]]
        ruler = ruler_for(rulers, page, unit, words)
        if not hits:
            if split is None or ruler is None:
                return []
            parts = [Candidate(w.bbox) for w in text.words
                     if w.fold == split.group(1)]
            return [c for c in on_ruler_row(parts, ruler, top, bottom)
                    if _within(c, ruler, top, bottom)]
        if ruler is not None:
            near = on_ruler_row([c for _, c in hits], ruler, top, bottom)
            if len(near) == 1:
                return near
            hits = [(row, c) for row, c in hits if c in near] or hits
        same_row = [c for row, c in hits
                    if _row_prints(row, top) or _row_prints(row, bottom)]
        if len(same_row) == 1:
            return same_row
        return [c for _, c in hits]
    return "label", find


def _within(cand: Candidate, ruler: Any, top: Optional[float],
            bottom: Optional[float]) -> bool:
    """Whether a candidate sits on the rows between ``top`` and ``bottom``
    (one row's slack each way)."""
    if top is None:
        return False
    ys = [ruler.y_at(float(v)) for v in (top, bottom) if v is not None]
    ys = [y for y in ys if y is not None]
    if not ys:
        return False
    return min(ys) - ROW_SLACK <= cand.cy <= max(ys) + ROW_SLACK


def _described(description: str, top: Optional[float],
               bottom: Optional[float], unit: str,
               rulers: Optional[Dict[int, Any]]) -> Anchor:
    """A layer's description by its first words; of several, the one on the
    ruler's rows between its top and its bottom, nearest the top."""
    def find(words: PageWords, page: int) -> List[Candidate]:
        cands = text_hits(words, page, description)
        ruler = ruler_for(rulers, page, unit, words)
        if len(cands) <= 1 or ruler is None or top is None:
            return cands
        # A description is printed from its top down: one ending just above
        # the top's row is the layer before's.
        y_top = ruler.y_at(float(top))
        below = [c for c in cands
                 if y_top is None or c.cy >= y_top - ROW_SLACK / 3.0]
        return on_ruler_row(below, ruler, top, bottom)
    return "description", find


def _blow_record(blows: Sequence[Any], top: Optional[float],
                 bottom: Optional[float], unit: str,
                 rulers: Optional[Dict[int, Any]]) -> Anchor:
    """A drive's blow counts printed in order along one row (``5-9-12`` or
    three cells); of several, the one on the ruler's row."""
    counts = [float(b) for b in blows
              if (isinstance(b, int) and not isinstance(b, bool))
              or (isinstance(b, str) and b.strip().isdigit())]

    def find(words: PageWords, page: int) -> List[Candidate]:
        if len(counts) < 2:
            return []
        cands: List[Candidate] = []
        for row in words(page).rows:
            flat = [(i, n) for i, w in enumerate(row.words)
                    for n, _ in w.numbers]
            for k in range(len(flat) - len(counts) + 1):
                run = flat[k:k + len(counts)]
                if all(n == c for (_, n), c in zip(run, counts)):
                    cands.append(Candidate(_union(
                        [row.words[i].bbox
                         for i in range(run[0][0], run[-1][0] + 1)])))
        if len(cands) <= 1:
            return cands
        return on_ruler_row(cands, ruler_for(rulers, page, unit, words), top,
                            bottom)
    return "blow record", find


def _water(depth: Optional[float], note: str, unit: str,
           rulers: Optional[Dict[int, Any]]) -> Anchor:
    """A water reading on its printed line.

    Its line names it: a groundwater word, or the words of the reading's
    own note (a table row's "Casing Pulled", "Completion"). With a depth:
    the one place the depth is printed; else the one whose line its note
    names best; else the one line naming water that prints it; else the one
    on the ruler's row (a water table's rows are on no ruler, so its
    readings stay several there). With none (none encountered), the line
    its note names, where only one does."""
    def named(row: Row) -> bool:
        return bool(_WATER_WORDS.search(row.text)) or (
            bool(note.strip()) and label_score(note, row.text)
            >= LABEL_MIN_SCORE)

    def find(words: PageWords, page: int) -> List[Candidate]:
        text = words(page)
        if depth is None:
            if not note.strip():
                return []
            scored = [(label_score(note, row.text), row) for row in text.rows]
            best = max((s for s, _ in scored), default=0.0)
            if best < LABEL_MIN_SCORE:
                return []
            return [Candidate(row.bbox) for s, row in scored
                    if s >= best - LABEL_MARGIN]
        hits = number_rows(text, depth)
        if len(hits) <= 1:
            return [Candidate(w.bbox) for _, w in hits]
        if note.strip():
            scored = [(label_score(note, row.text), w) for row, w in hits]
            best = max(s for s, _ in scored)
            top = [w for s, w in scored if s > best - LABEL_MARGIN]
            if best >= LABEL_MIN_SCORE and len(top) == 1:
                return [Candidate(top[0].bbox)]
        on_named = [(row, w) for row, w in hits if named(row)]
        rows = {id(row): row for row, _ in on_named}
        if len(rows) == 1:
            return [Candidate(next(iter(rows.values())).bbox)]
        cands = [Candidate(w.bbox) for _, w in (on_named or hits)]
        return on_ruler_row(cands, ruler_for(rulers, page, unit, words),
                            depth) or cands
    return "printed depth", find


def investigation_targets(inv: Any, rulers: Optional[Dict[int, Any]] = None
                          ) -> List[Tuple[str, str, Any, List[Anchor]]]:
    """``(what, kind, object, anchors)`` for each layer, sample, drive and
    water reading of one investigation (an ``Investigation``)."""
    unit = str(getattr(inv, "depth_unit", "") or "")
    out: List[Tuple[str, str, Any, List[Anchor]]] = []
    for n, layer in enumerate(getattr(inv, "layers", None) or [], 1):
        top, bottom = _value(layer.top), _value(layer.bottom)
        anchors: List[Anchor] = []
        if str(layer.description or "").strip():
            anchors.append(_described(layer.description, top, bottom, unit,
                                      rulers))
        anchors.append(_printed_depth(top, unit, rulers))
        out.append((f"layer {n} at {_show(top)}", "layer", layer, anchors))
    for n, sample in enumerate(getattr(inv, "samples", None) or [], 1):
        top, bottom = _value(sample.top), _value(sample.bottom)
        anchors = []
        if str(sample.sample_id or "").strip():
            anchors.append(_labelled(sample.sample_id, top, bottom, unit,
                                     rulers))
        anchors.append(_printed_depth(top, unit, rulers))
        out.append((f"sample {sample.sample_id or n} at {_show(top)}",
                    "sample", sample, anchors))
    for n, drive in enumerate(getattr(inv, "spt", None) or [], 1):
        top, bottom = _value(drive.depth_top), _value(drive.depth_bottom)
        anchors = []
        if str(drive.sample_id or "").strip():
            anchors.append(_labelled(drive.sample_id, top, bottom, unit,
                                     rulers))
        # A drive's blow record is its own print; its depth is shared with
        # the sample it drove and with the layer that starts there.
        anchors.append(_blow_record(drive.blows, top, bottom, unit, rulers))
        anchors.append(_printed_depth(top, unit, rulers))
        out.append((f"drive {n} at {_show(top)}", "drive", drive, anchors))
    for n, water in enumerate(getattr(inv, "water", None) or [], 1):
        depth = _value(water.depth)
        out.append((f"water {n} at {_show(depth)}", "water", water,
                    [_water(depth, str(water.note or ""), unit, rulers)]))
    return out


def place_investigation(words: PageWords, inv: Any,
                        pages: Optional[Sequence[int]] = None,
                        rulers: Optional[Dict[int, Any]] = None,
                        write: bool = True,
                        numbers: Optional[PageNumbers] = None) -> Placements:
    """Place every layer, sample, drive and water reading of one log.

    ``pages`` are the call's pages (default: the log's own); ``rulers`` the
    depth rulers of its pages (``log_grid(...).rulers``), the anchor between
    several candidates. ``numbers``: how the answer's cited pages are read
    (:func:`answer_pages` over every investigation of the answer, which
    the caller decides once); by default from this log's own cites. With
    ``write`` each object's provenance takes its placement (:func:`apply`).
    """
    pages = [int(p) for p in (pages or getattr(inv, "pages", None) or [])]
    out = Placements()
    targets = investigation_targets(inv, rulers)
    if not pages:
        for what, kind, obj, _ in targets:
            out.items.append(_no_page(what, kind, getattr(obj, "prov", None)))
            if write:
                apply(getattr(obj, "prov", None), out.items[-1])
        return out
    if numbers is None:
        numbers = answer_pages([getattr(o, "prov", None)
                                for _, _, o, _ in targets], pages)
    placed: List[Tuple[Any, Placement]] = []
    for what, kind, obj, anchors in targets:
        cited = _checked(getattr(obj, "prov", None), pages, numbers, write)
        placed.append((obj, place(what, kind, cited, pages, words,
                                  anchors)))
    _from_drives(placed)
    for obj, got in placed:
        if write:
            apply(getattr(obj, "prov", None), got)
        out.items.append(got)
    return out


def _from_drives(placed: List[Tuple[Any, Placement]]) -> None:
    """A sample not placed by its label takes the box of the one drive
    placed at its top on its cited page: the sample came out of that drive,
    so the drive's printed line is the sample's. Its own depth alone is a
    weaker anchor: the layer that starts there prints it too."""
    drives = [(_value(obj.depth_top), got) for obj, got in placed
              if got.kind == "drive" and got.boxed]
    for i, (obj, got) in enumerate(placed):
        if got.kind != "sample" or got.status == NO_TEXT or (
                got.boxed and got.anchor != "printed depth"):
            continue
        top = _value(obj.top)
        same = {d.bbox: d for depth, d in drives
                if d.page == got.cited and top is not None
                and depth is not None and same_number(top, depth)}
        if len(same) != 1:
            continue
        drive = next(iter(same.values()))
        placed[i] = (obj, replace(
            got, page=drive.page, status=PLACED, bbox=drive.bbox,
            anchor="its drive", source=drive.source, candidates=1,
            note=f"the box of {drive.what}, placed at its top on page "
                 f"{drive.page}: the sample came out of that drive; "
                 f"{_box_note(drive.source)}"))


# ---------------------------------------------------------------------------
# laboratory tests
# ---------------------------------------------------------------------------

def _narrowed(hits: List[Tuple[Row, Candidate]],
              keeps: Sequence[Callable[[Row], bool]]) -> List[Candidate]:
    """Label hits narrowed one test at a time, until one is left: each test
    keeps the hits whose row passes it, and is passed over when it would
    keep none."""
    for keep in keeps:
        if len(hits) <= 1:
            break
        kept = [(row, c) for row, c in hits if keep(row)]
        if kept:
            hits = kept
    return [c for _, c in hits]


def _row_tests(boring: str, top: Optional[float], numbers: Sequence[float]
               ) -> List[Callable[[Row], bool]]:
    """What a test's row prints, in the order it narrows the test's label
    lines: its boring's label, its depth, then each of its own results."""
    out: List[Callable[[Row], bool]] = []
    if boring:
        out.append(lambda row: _row_has_label(row, boring))
    out.append(lambda row: _row_prints(row, top))
    out += [lambda row, n=n: _row_prints(row, n) for n in numbers]
    return out


def _sample_line(sample: str, boring: str, top: Optional[float],
                 numbers: Sequence[float] = ()) -> Anchor:
    """A test's sample label; of several (one sample label in each boring
    of a summary), narrowed by what the test's row prints
    (:func:`_row_tests`)."""
    def find(words: PageWords, page: int) -> List[Candidate]:
        return _narrowed(label_hits(words(page), sample),
                         _row_tests(boring, top, numbers))
    return "sample label", find


def _boring_line(boring: str, top: Optional[float],
                 numbers: Sequence[float] = ()) -> Anchor:
    """A test's boring label: a summary with no sample column, or a sample
    label the answer writes another way than the sheet. Only a label on a
    row that also prints the test's depth or one of its results is the
    test's line; a label alone is a heading, which every test of that
    boring shares (:func:`boring_heading`). Of several, narrowed by the
    depth and the results its row prints."""
    tests = _row_tests("", top, numbers)

    def find(words: PageWords, page: int) -> List[Candidate]:
        hits = [(row, c) for row, c in label_hits(words(page), boring)
                if any(test(row) for test in tests)]
        return _narrowed(hits, tests)
    return "boring label", find


def boring_heading(words: PageWords, page: int, test: Any) -> bool:
    """Whether a laboratory test's boring label is printed on the page only
    as a heading: on no row that also prints the test's depth or one of its
    results, so it is not the test's line (:func:`_boring_line`)."""
    boring = str(getattr(test, "investigation_id", "") or "").strip()
    if not boring:
        return False
    hits = label_hits(words(page), boring)
    tests = _row_tests("", _value(getattr(test, "depth_top", None)),
                       result_numbers(getattr(test, "result", None)))
    return bool(hits) and not any(prints(row) for row, _ in hits
                                  for prints in tests)


def _title(kind: str) -> Anchor:
    """The sheet's title: the first row from the top that the lab floor's
    title vocabulary (``TITLE_KINDS``) names this kind of test by; failing
    that, the first row it names any test by (a summary table titled for
    the limits it tabulates)."""
    own = [pattern for pattern, name in TITLE_KINDS if name == kind]
    every = [pattern for pattern, _ in TITLE_KINDS]

    def find(words: PageWords, page: int) -> List[Candidate]:
        rows = [(" ".join(row.text.lower().split()), row)
                for row in words(page).rows]
        for patterns in (own, every):
            for folded, row in rows:
                if any(re.search(p, folded) for p in patterns):
                    return [Candidate(row.bbox)]
        return []
    return "test title", find


def result_numbers(result: Any) -> List[float]:
    """The single numbers a test's result reports (liquid limit, fines,
    D50; not its curves or trials, which a sheet prints point by point), less
    zeros, which any sheet prints."""
    out: List[float] = []
    for name in type(result).model_fields if result is not None else ():
        got = getattr(result, name, None)
        if isinstance(got, bool) or isinstance(got, (list, tuple, dict)):
            continue
        number = _value(got) if hasattr(got, "value") else (
            float(got) if isinstance(got, (int, float)) else None)
        if number is not None and number != 0 and number not in out:
            out.append(number)
    return out


def _values_row(numbers: Sequence[float]) -> Anchor:
    """The row that prints the most of a test's own result numbers, two or
    more: a summary table's or a legend's line for this sample. Its box
    spans the numbers it prints."""
    def find(words: PageWords, page: int) -> List[Candidate]:
        if len(numbers) < 2:
            return []
        best, rows = 2, []
        for row in words(page).rows:
            hits = [w for w in row.words if any(w.prints(n) for n in numbers)]
            found = sum(1 for n in numbers
                        if any(w.prints(n) for w in row.words))
            if found > best:
                best, rows = found, []
            if found == best:
                rows.append(Candidate(_union([w.bbox for w in hits])))
        return rows
    return "result values", find


def lab_targets(test: Any) -> List[Anchor]:
    """The anchors of one laboratory test (a ``LabTest``): its sample's
    label; the row printing its own results; its boring's label; the
    sheet's title."""
    sample = str(getattr(test, "sample_id", "") or "").strip()
    boring = str(getattr(test, "investigation_id", "") or "").strip()
    top = _value(getattr(test, "depth_top", None))
    numbers = result_numbers(getattr(test, "result", None))
    anchors: List[Anchor] = []
    if sample:
        anchors.append(_sample_line(sample, boring, top, numbers))
    anchors.append(_values_row(numbers))
    if boring:
        anchors.append(_boring_line(boring, top, numbers))
    anchors.append(_title(str(getattr(test, "kind", "") or "")))
    return anchors


def _no_page(what: str, kind: str, prov: Any) -> Placement:
    """An object with no page to place it on: neither the call nor its
    record gives one."""
    return Placement(what, kind, getattr(prov, "page", None), None, NO_PAGE,
                     note="neither the call nor the record gives a page, so "
                          "there is no page to place it on")


def _first_prov(test: Any) -> Any:
    provs = list(getattr(test, "prov", None) or [])
    return provs[0] if provs else None


def place_lab_tests(words: PageWords, tests: Sequence[Any],
                    pages: Optional[Sequence[int]] = None,
                    write: bool = True,
                    numbers: Optional[PageNumbers] = None) -> Placements:
    """Place every test of one sheet (or window) by its first provenance.

    ``pages`` are the call's pages (default: each test's own ``pages``);
    ``numbers`` how the answer's cited pages are read (by default from the
    tests' own cites, :func:`answer_pages`). A test placed by the sheet's
    title is placed at page level, but for the answer's only test on that
    page: the sheet is that test's place. A test with no page to place it
    on is counted (``no_page``); one with no provenance is placed and
    noted, as nothing on the record can take its placement."""
    out = Placements()
    if pages and numbers is None:
        numbers = answer_pages([_first_prov(t) for t in tests], pages)
    checked: List[Tuple[Any, Any, str, List[int], Optional[int]]] = []
    for n, test in enumerate(tests, 1):
        own = [int(p) for p in (pages or getattr(test, "pages", None) or [])]
        prov = _first_prov(test)
        what = (f"{getattr(test, 'kind', 'test')} {n} "
                f"({getattr(test, 'sample_id', '') or 'no sample label'})")
        cited = (_checked(prov, own, numbers or answer_pages([prov], own),
                          write) if own else None)
        checked.append((test, prov, what, own, cited))
    on_page = Counter(c for *_, c in checked if c is not None)
    unwritten = 0
    for test, prov, what, own, cited in checked:
        if cited is None:
            out.items.append(_no_page(what, "test", prov))
            if write:
                apply(prov, out.items[-1])
            continue
        got = place(what, "test", cited, own, words, lab_targets(test))
        if got.page_level and got.anchor == "test title" and \
                on_page[got.page] == 1:
            got.page_level = False
            got.note = (f"{_box_note(got.source)}: the sheet's title, the "
                        f"place of the answer's only test on page "
                        f"{got.page}")
        if got.anchor in ("test title", "") and \
                boring_heading(words, got.page, test):
            got.note = _join(got.note, f"its boring's label is printed on "
                                       f"page {got.page} only as a heading, "
                                       f"on no row with its depth or "
                                       f"results")
        if prov is None:
            unwritten += 1
            got.note = _join(got.note, "the test has no provenance, so this "
                                       "placement is not written to it")
        elif write:
            apply(prov, got)
        out.items.append(got)
    if unwritten:
        out.notes.append(f"{unwritten} of {len(out.items)} test(s) have no "
                         f"provenance, so their placements are not written "
                         f"to the record")
    return out


# ---------------------------------------------------------------------------
# a calculation
# ---------------------------------------------------------------------------

def heading_rows(text: PageText, row: Row) -> List[Row]:
    """The heading rows above ``row``, nearest first: rows fewer than half
    of whose words print a number (titles, block names, column heads), up
    to :data:`CONTEXT_ROWS` of them, looked for past the rows of values in
    between up to :data:`LOOKBACK_ROWS` rows up."""
    at = next((i for i, r in enumerate(text.rows) if r is row), None)
    if at is None:
        return []
    out: List[Row] = []
    for above in reversed(text.rows[max(0, at - LOOKBACK_ROWS):at]):
        numeric = sum(1 for w in above.words if w.numbers)
        if numeric * 2 < len(above.words):
            out.append(above)
            if len(out) >= CONTEXT_ROWS:
                break
    return out


def _overlaps(a: Word, b: Word, slack: float = 4.0) -> bool:
    return a.bbox[0] <= b.bbox[2] + slack and a.bbox[2] >= b.bbox[0] - slack


def cell_matches(label: str, text: PageText, row: Row, word: Word
                 ) -> List[Tuple[float, float]]:
    """``(weight, how well found)`` for each of a label's words
    (:func:`_label_words`) around one printed value.

    The line's other words and the heads of the value's column (the words
    of a heading row straight above it) count whole. Other headings above
    count half -- titles, block names, ``key = value`` lines -- but not a
    column head straight above another value of the line: that names the
    other value. A number in the label ("Layer 7", "Scenario 2") names a
    row or a block: on the line it counts only as the line's first word,
    and the line's first word printing it is that row's name, not the
    value, which then matches nothing.
    """
    wanted = _label_words(label)
    if not wanted:
        return []
    first = row.words[0] if row.words else None
    if first is word and any(t == word.fold for t, _ in wanted
                             if t.isdigit()):
        return [(weight, 0.0) for _, weight in wanted]
    others = [w for w in row.words if w is not word]
    line = [t for w in others for t in _fold(w.text).split()]
    head = _fold(first.text).split() if first is not None and \
        first is not word else []
    values = [w for w in others if w.numbers]
    column: List[str] = []
    above: List[str] = []
    for heading in heading_rows(text, row):
        keyed = any(w.numbers for w in heading.words)
        for w in heading.words:
            tokens = _fold(w.text).split()
            if keyed:
                above.extend(tokens)
            elif _overlaps(w, word):
                column.extend(tokens)
            elif not any(_overlaps(w, v) for v in values):
                above.extend(tokens)
    out: List[Tuple[float, float]] = []
    for want, weight in wanted:
        near = head if want.isdigit() else line
        out.append((weight, max(_found(want, near), _found(want, column),
                                _found(want, above) / 2.0)))
    return out


def cell_score(label: str, text: PageText, row: Row, word: Word) -> float:
    """How much of a value's label the printout carries around one printed
    value, 0 to 100 (:func:`cell_matches`)."""
    matches = cell_matches(label, text, row, word)
    total = sum(weight for weight, _ in matches)
    return 100.0 * sum(w * d for w, d in matches) / total if total else 0.0


def _matched(name: str, number: float, words: PageWords, page: int
             ) -> List[Tuple[List[Tuple[float, float]], Candidate]]:
    text = words(page)
    return [(cell_matches(name, text, row, word), Candidate(word.bbox))
            for row, word in number_rows(text, number)]


def _label_and_number(name: str, number: float) -> Anchor:
    """The number where the printout around it carries half its label or
    more (:func:`cell_score`); of several, the best, where it leads the
    next by :data:`LABEL_MARGIN`."""
    def find(words: PageWords, page: int) -> List[Candidate]:
        scored = []
        for matches, cand in _matched(name, number, words, page):
            total = sum(weight for weight, _ in matches)
            score = 100.0 * sum(w * d for w, d in matches) / total \
                if total else 0.0
            if score >= LABEL_MIN_SCORE:
                scored.append((score, cand))
        if not scored:
            return []
        best = max(s for s, _ in scored)
        return [c for s, c in scored if s > best - LABEL_MARGIN]
    return "label and number", find


def _number_by_label(name: str, number: float) -> Anchor:
    """Of a number printed in several places, the one place that carries
    clearly more of its label than any other (leading the next by
    :data:`LABEL_LEAD`), counting only the label's words printed around
    one place or another: a load printed beside its case's name and again
    as a chart's tick label; a table's cell under its column's head on its
    scenario's row, where the label also names the soil only the title
    block prints."""
    def find(words: PageWords, page: int) -> List[Candidate]:
        matched = _matched(name, number, words, page)
        if len(matched) < 2:
            return []
        seen = [any(m[i][1] > 0 for m, _ in matched)
                for i in range(len(matched[0][0]))]
        total = sum(w for (w, _), s in zip(matched[0][0], seen) if s)
        if not total:
            return []
        scored = sorted(((100.0 * sum(w * d for (w, d), s in zip(m, seen)
                                      if s) / total, cand)
                         for m, cand in matched), key=lambda sc: -sc[0])
        if scored[0][0] - scored[1][0] >= LABEL_LEAD:
            return [scored[0][1]]
        return []
    return "number by its label", find


def _number_alone(number: float) -> Anchor:
    def find(words: PageWords, page: int) -> List[Candidate]:
        return number_hits(words(page), number)
    return "number", find


def _label_and_text(name: str, value: str) -> Anchor:
    """A short printed value (a site class's letter, a strength model's
    name) as whole words on a line that carries half its label or more; of
    several, the best line, where it leads by :data:`LABEL_MARGIN`."""
    wanted = _fold_token(value)

    def find(words: PageWords, page: int) -> List[Candidate]:
        text = words(page)
        scored: List[Tuple[float, Candidate]] = []
        for row in text.rows:
            hits = _run_hits(row, wanted) if wanted else []
            if not hits:
                continue
            context = " ".join(r.text for r in heading_rows(text, row))
            score = label_score(name, row.text, context)
            if score >= LABEL_MIN_SCORE:
                i, j = hits[0]
                scored.append((score, Candidate(_union(
                    [w.bbox for w in row.words[i:j + 1]]))))
        if not scored:
            return []
        best = max(s for s, _ in scored)
        return [c for s, c in scored if s > best - LABEL_MARGIN]
    return "label and text", find


def _phrase(name: str, phrase: str) -> Anchor:
    def find(words: PageWords, page: int) -> List[Candidate]:
        return text_hits(words, page, phrase)
    return name, find


def value_targets(value: Any) -> List[Anchor]:
    """The anchors of one calculation value (a ``NamedQuantity``): a number
    by its label, then alone; words by its label's line, then alone. Never
    by its label alone: a value whose own words are not printed is not
    found."""
    name = str(getattr(value, "name", "") or "")
    number = _value(getattr(value, "value", None))
    if number is not None:
        return [_label_and_number(name, number),
                _number_by_label(name, number), _number_alone(number)]
    anchors: List[Anchor] = []
    text = str(getattr(value, "text", "") or "").strip()
    if text and name:
        anchors.append(_label_and_text(name, text))
    if len(text) > SHORT_TEXT:
        anchors.append(_phrase("printed text", text))
    return anchors


def place_calculation(words: PageWords, calc: Any,
                      pages: Optional[Sequence[int]] = None,
                      write: bool = True,
                      numbers: Optional[PageNumbers] = None) -> Placements:
    """Place every input and result of one calculation printout.

    ``pages`` are the call's pages (default: the calculation's own);
    ``numbers`` how the answer's cited pages are read (by default from the
    values' own cites, :func:`answer_pages`)."""
    pages = [int(p) for p in (pages or getattr(calc, "pages", None) or [])]
    out = Placements()
    roles = (("input", getattr(calc, "inputs", None) or []),
             ("result", getattr(calc, "results", None) or []))
    if not pages:
        for role, values in roles:
            for value in values:
                prov = getattr(value, "prov", None)
                out.items.append(_no_page(f"{role} {value.name}", role,
                                          prov))
                if write:
                    apply(prov, out.items[-1])
        return out
    if numbers is None:
        numbers = answer_pages([getattr(v, "prov", None)
                                for _, values in roles for v in values],
                               pages)
    for role, values in roles:
        for value in values:
            prov = getattr(value, "prov", None)
            cited = _checked(prov, pages, numbers, write)
            got = place(f"{role} {value.name}", role, cited, pages, words,
                        value_targets(value))
            if write:
                apply(prov, got)
            out.items.append(got)
    return out


# ---------------------------------------------------------------------------
# a narrative's citations
# ---------------------------------------------------------------------------

def _quote(quote: str) -> List[Anchor]:
    """The quote on the page by the quote gate's own rule
    (:func:`check_quotes`): boxed where planlens' search finds its first
    words (``quote``); else on the page with no box (``quote on its page``,
    page level)."""
    passed: Dict[int, bool] = {}

    def on_page(words: PageWords, page: int) -> bool:
        from report_ingest.model import Citation

        if page not in passed:
            got = {}
            if words(page).has_text and _fold(quote):
                got = check_quotes(words.doc, {"q": [Citation(
                    page=page, quote=quote)]}, ["q"])
            passed[page] = got.get("q") == DEFAULT_CONFIDENCE
        return passed[page]

    def boxed(words: PageWords, page: int) -> List[Candidate]:
        return text_hits(words, page, quote)[:1] if on_page(words, page) \
            else []

    def page_only(words: PageWords, page: int) -> List[Candidate]:
        return [Candidate(None, QUOTE_RATIO)] if on_page(words, page) else []
    return [("quote", boxed), ("quote on its page", page_only)]


def place_citations(words: PageWords, citations: Dict[str, Sequence[Any]],
                    pages: Sequence[int]) -> Placements:
    """Check every citation of a narrative answer (``field -> [Citation]``)
    against its page, then the call's other pages.

    A citation carries no box (section 8.4), so nothing is written: the
    result says which quotes are on their pages, which were found on another
    page, and the box a viewer could show."""
    pages = [int(p) for p in pages]
    out = Placements()
    if not pages:
        out.items = [_no_page(f"{name} quote", "citation", cite)
                     for name, cites in citations.items() for cite in cites]
        return out
    every = [c for cites in citations.values() for c in cites]
    numbers = answer_pages(every, pages)
    for name, cites in citations.items():
        for cite in cites:
            cited = page_for(getattr(cite, "page", None), pages, numbers)[0]
            out.items.append(place(f"{name} quote", "citation", cited, pages,
                                   words, _quote(cite.quote)))
    return out
