"""Model-first ingest: which pages go in each call, and how they are sent.

BACKLOG RI-87, pieces 2 and 3 (``module_work/REPORT_INGEST_PLAN.md`` §8.2.1
and §8.7). Nothing in ingest calls this module yet: the model-first readers
(piece 7) and the graph wiring (piece 8) do, and only under
``GEOTECH_INGEST_MODEL_FIRST=on``. It makes no model call of its own.

SENDING THE PAGES (piece 2). A call carries its pages one of two ways, named
by ``GEOTECH_INGEST_MODEL_FIRST_PAGES`` (:func:`page_route`):

* ``pdf``: the pages cut out of the report into one small PDF
  (:func:`cut_pdf`), which the Anthropic API reads as both its text layer
  and its page images. At most :data:`MAX_PDF_PAGES` pages and
  :data:`MAX_PDF_BYTES` bytes (:func:`pdf_problem`); a cut larger than
  :data:`SANITIZE_ABOVE` is sanitised.
* ``pictures``: each page rendered at the size the model reads without
  shrinking it, followed by the page's text -- the text layer, or the Azure
  DI text on a scan -- between markers (:func:`picture_blocks`). At most
  :data:`MAX_PICTURES_PER_CALL` pictures, each and all together within the
  request limits (:func:`pictures_problem`). Both Foundry routes take
  pictures; whether they take a PDF is what the probe of piece 1 settles.
  The narrative's call sends TEXT FIRST (RI-175): a page with a usable
  text layer (:func:`text_pages`) goes as its text alone, and only the
  others as pictures (:func:`page_blocks`), so a report's narrative fits
  one call; on 20 pictures a page it was read in three or four parts.
* ``auto`` (the default): a PDF where the engine can send one, pictures
  otherwise (:func:`resolve_route`).

Either way the instruction says which report page each PDF page or picture
is (:func:`page_map_sentence`), so the model cites the report's own 0-based
page index.

GROUPING THE PAGES (piece 3). Which pages go in each call is decided from
the range labels and RI-73's page descriptions alone (:func:`group_pages`):

* **logs** (:func:`log_groups`): one log per call. A new log starts at a
  page whose description says it is a set's first sheet ("sheet 1 of 2"),
  or that logs another exploration than the current log's -- the
  describer's own ``exploration`` field (:func:`page_identifiers`) --
  unless it says in words that it is continued from the page before; a
  page that names no exploration joins the log before it. A log longer
  than :data:`MAX_LOG_PAGES` is cut into balanced parts, read by one call
  each and joined after. One exploration's logs can sit in more than one
  place in a report; joining them is the record's work, by exploration.
* **lab** (:func:`lab_groups`): each run of laboratory pages in windows of
  at most ``MAX_LAB_PAGES`` (``lab_reader.lab_windows``), the cuts moved to
  pages where the description's title or exploration changes.
* **calc** (:func:`calc_groups`): each run of calculation pages split into
  printouts where today's page heads say so (``calc_reader.
  split_calc_runs``, when the document is at hand) OR a description says a
  set's first sheet begins ("sheet 1 of 7"). Either signal alone mixed or
  padded truthed printouts that the two together keep apart (the offline
  score, stage b).
* **narrative** (:func:`narrative_groups`): the pages
  ``narrative_reader.reading_pages`` gives (cover, letter, contents,
  narrative and the front of the report), plus the report's own figure,
  plan and profile pages, in consecutive parts of at most
  :data:`MAX_NARRATIVE_PAGES` pages (one part on the pictures route),
  ranked for the reader to keep the front first when they do not all fit
  one request; for a report bound inside another, within that report's own
  pages.

Every group says why it starts where it does (:attr:`PageGroup.why`), so a
QA note or a trace review can name the rule that drew it. The rules are
LAYOUT JUDGMENT, scored offline against the answer sheets' item pages before
any call is paid for (``module_work/report_ingest_harness/
measure_grouping.py``).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple, Union)

__all__ = [
    "PAGES_ENV", "PAGE_ROUTES", "MAX_PDF_PAGES", "MAX_PDF_BYTES",
    "SANITIZE_ABOVE", "MAX_PICTURES_PER_CALL", "MAX_PICTURE_BYTES",
    "MAX_REQUEST_BYTES", "MAX_LOG_PAGES", "MAX_NARRATIVE_PAGES",
    "STAGE_LABELS", "NARRATIVE_LABELS", "FIGURE_LABELS", "TEXT_MARKERS",
    "PageGroup", "page_route", "resolve_route", "max_pages_per_call",
    "cut_pdf", "pdf_problem", "page_map_sentence", "picture_blocks",
    "text_pages", "page_blocks",
    "pictures_problem", "exploration_id", "page_identifiers",
    "page_identifier", "SHARED_ACROSS", "sheet_of",
    "carries_on", "is_divider", "divider_pages", "stage_pages", "runs",
    "log_groups", "lab_starts",
    "lab_groups", "calc_starts", "calc_groups", "calc_calls",
    "narrative_groups", "group_pages",
]


# ---------------------------------------------------------------------------
# the limits and the switch
# ---------------------------------------------------------------------------

#: The variable that says how pages reach the model.
PAGES_ENV = "GEOTECH_INGEST_MODEL_FIRST_PAGES"
#: Its settings: ``auto`` (the default), ``pdf`` or ``pictures``.
PAGE_ROUTES: Tuple[str, ...] = ("auto", "pdf", "pictures")

#: The Anthropic API's limits on one PDF document (pages, bytes).
MAX_PDF_PAGES = 100
MAX_PDF_BYTES = 32 * 1024 * 1024
#: A cut larger than this is sanitised (:func:`cut_pdf`).
SANITIZE_ABOVE = 8 * 1024 * 1024

#: The most pictures one call carries on the pictures route. Above 20
#: images in one request Claude's API holds every image to a stricter size
#: (2,000 px on a side), and claude.ai takes 20 a message (the vision page
#: of Claude's API documentation, read 2026-10-25). The Foundry wave
#: measures what its route takes; until then, 20.
MAX_PICTURES_PER_CALL = 20
#: One picture's size, base64-encoded, at most: Amazon Bedrock's 5 MB, the
#: lowest the same page gives (Claude's API takes 10 MB).
MAX_PICTURE_BYTES = 5 * 1024 * 1024
#: One request's size at most, pictures and words together: the standard
#: endpoints' 32 MB (lower on some partner platforms).
MAX_REQUEST_BYTES = 32 * 1024 * 1024

#: The most sheets one log call carries; a longer log is cut into parts.
MAX_LOG_PAGES = 12
#: The most pages one narrative call carries, on either route: on the
#: pictures route its pages with text go as text (RI-175).
MAX_NARRATIVE_PAGES = MAX_PDF_PAGES

#: Which labels each reader's groups are drawn from: the graph's own table
#: (``graph.ITEM_READERS``) for logs, lab and calc. Cone soundings and
#: dynamic probes keep today's sounding reader and are not grouped here.
STAGE_LABELS: Dict[str, Tuple[str, ...]] = {
    "logs": ("boring_log", "test_pit_log"),
    "lab": ("lab_test",),
    "calc": ("calculation",),
}
#: The labels the narrative reader is given today
#: (``narrative_reader.FRONT_LABELS``).
NARRATIVE_LABELS: Tuple[str, ...] = ("cover", "letter", "toc", "narrative")
#: The report's own figure series, added to the narrative's pages.
FIGURE_LABELS: Tuple[str, ...] = ("figure", "plan", "profile")

#: The markers a page's text is sent between, as the RI-73 describer sends
#: it.
TEXT_MARKERS: Tuple[str, str] = ("<<<PAGE TEXT", "PAGE TEXT>>>")


def page_route(value: Any = None) -> str:
    """The page route a setting names: ``value``, or the variable when
    ``value`` is None; unset or empty is ``auto``. A typo is refused."""
    if value is None:
        value = os.environ.get(PAGES_ENV)
    text = str(value or "").strip().lower()
    if not text:
        return "auto"
    if text not in PAGE_ROUTES:
        raise ValueError(f"{PAGES_ENV}={value!r}: use "
                         f"{', '.join(PAGE_ROUTES)}")
    return text


def resolve_route(setting: Any, accepts_pdf: bool) -> str:
    """``pdf`` or ``pictures`` for one engine. ``auto`` sends a PDF where the
    engine can send one; ``pdf`` on an engine that cannot is refused rather
    than sent as something the model would never see."""
    route = page_route(setting) if setting is not None else page_route()
    if route == "auto":
        return "pdf" if accepts_pdf else "pictures"
    if route == "pdf" and not accepts_pdf:
        raise ValueError(f"{PAGES_ENV}=pdf, but this engine cannot send a "
                         f"PDF; use pictures or auto")
    return route


def max_pages_per_call(route: str) -> int:
    """The most pages one call may carry on a route: the API's PDF limit,
    or :data:`MAX_PICTURES_PER_CALL` pictures."""
    if route == "pictures":
        return MAX_PICTURES_PER_CALL
    return MAX_PDF_PAGES


# ---------------------------------------------------------------------------
# piece 2: the pages as a PDF
# ---------------------------------------------------------------------------

def cut_pdf(source: Union[str, Path, bytes], pages: Optional[Sequence[int]],
            turns: Optional[Mapping[int, int]] = None
            ) -> Tuple[bytes, int, bool]:
    """``(the PDF's bytes, its page count, sanitized)``: ``pages`` (0-based,
    in order) copied out of ``source`` (a path, or a PDF's bytes) with their
    text layer and images, or the whole file for ``None``. Nothing is
    written to disk.

    A copied page keeps its whole resource dictionary, and some reports hang
    every font and form of the file on every page: four corpus items cut to
    25-284 MB, past the API's 32 MB. A cut larger than
    :data:`SANITIZE_ABOVE` is therefore SANITIZED (its content streams
    rewritten and the resources they never use dropped: 69 MB to 74 KB);
    on those four the text and a 40 dpi rendering came out identical. Not
    every page does -- one other corpus page renders differently once
    sanitized -- so a cut that fits is sent as copied.

    The limits are the caller's to check (:func:`pdf_problem`), because a
    cut too large to send is still worth recording.

    ``turns`` maps a page (0-based, of ``source``) whose text runs a quarter
    turn from the page (:func:`report_ingest.turned.page_turn`: a sheet
    scanned sideways) to that turn; its copy is rotated by the same turn
    clockwise, so it displays upright (RI-149). Only a copied page turns:
    the whole file (``pages`` None) is sent as it is."""
    import pymupdf

    def opened() -> Any:
        if isinstance(source, (bytes, bytearray)):
            return pymupdf.open(stream=bytes(source), filetype="pdf")
        return pymupdf.open(str(source))

    name = "the PDF" if isinstance(source, (bytes, bytearray)) \
        else Path(source).name
    if pages is None:
        data = (bytes(source) if isinstance(source, (bytes, bytearray))
                else Path(source).read_bytes())
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return data, doc.page_count, False
    with opened() as src:
        out = pymupdf.open()
        try:
            for page in pages:
                page = int(page)
                if not 0 <= page < src.page_count:
                    raise ValueError(f"page {page} is not in {name} "
                                     f"({src.page_count} pages)")
                out.insert_pdf(src, from_page=page, to_page=page)
                quarter = int((turns or {}).get(page, 0)) % 360
                if quarter:
                    copied = out[out.page_count - 1]
                    copied.set_rotation((copied.rotation + quarter) % 360)
            data = out.tobytes(garbage=3, deflate=True)
            if len(data) <= SANITIZE_ABOVE:
                return data, out.page_count, False
            for page in out:
                page.clean_contents(sanitize=True)
            return (out.tobytes(garbage=4, deflate=True, clean=True),
                    out.page_count, True)
        finally:
            out.close()


def pdf_problem(n_pages: int, n_bytes: int) -> str:
    """Why a PDF of this size cannot go in one call, or ``""`` if it can."""
    if int(n_pages) > MAX_PDF_PAGES or int(n_bytes) > MAX_PDF_BYTES:
        return (f"the PDF is {int(n_pages)} pages and {int(n_bytes):,} "
                f"bytes; the API takes at most {MAX_PDF_PAGES} pages and "
                f"{MAX_PDF_BYTES:,} bytes in one document")
    return ""


def page_map_sentence(pages: Sequence[int], route: str = "pdf",
                      whole: bool = False, text_first: bool = False) -> str:
    """The sentence that tells the model which report page each PDF page or
    picture is, so every page it cites is the report's 0-based index.

    The PDF wording is RI-86's, which the plain baseline measured: a page
    list ``[7, 8]`` gives "PDF page 1 is page 7, PDF page 2 is page 8".
    ``whole`` is a PDF of the whole report, whose pages need no map.
    ``text_first`` is a pictures-route call whose pages with a usable text
    layer go as their text alone (:func:`page_blocks`, RI-175)."""
    pages = [int(p) for p in pages]
    if route == "pictures" and text_first:
        return ("Each page is given under its page index: a page with a "
                "usable text layer as its text alone, and any other page as "
                "a picture followed by whatever text it has; wherever a page "
                "is asked for, give that page index (the report's 0-based "
                "page index).")
    if route == "pictures":
        return ("Each page is given as a picture headed by its page index, "
                "followed by the page's text; wherever a page is asked for, "
                "give that page index (the report's 0-based page index).")
    if whole:
        return ("Wherever a page is asked for, give the 0-based page index "
                "(the PDF's first page is 0).")
    mapping = ", ".join(f"PDF page {n + 1} is page {p}"
                        for n, p in enumerate(pages))
    return (f"The attached PDF holds only these pages, in order; wherever a "
            f"page is asked for, give the report's 0-based page index "
            f"({mapping}).")


# ---------------------------------------------------------------------------
# piece 2: the pages as pictures with their text
# ---------------------------------------------------------------------------

def _budget_for(model: Optional[str]) -> Any:
    """The largest picture ``model`` reads without shrinking it, or None
    (planlens' default size)."""
    if not model:
        return None
    from planlens.document.budget import budget_for_model

    found = budget_for_model(model)
    return found[0] if found else None


def _base64_size(n_bytes: int) -> int:
    return 4 * ((int(n_bytes) + 2) // 3)


def pictures_problem(blocks: Sequence[Mapping[str, Any]]) -> str:
    """Why a call of these blocks (:func:`picture_blocks`, and any words
    beside them) cannot be sent, or ``""`` if it can: too many pictures,
    one picture too large, or the whole request too large, each as
    :data:`MAX_PICTURES_PER_CALL`, :data:`MAX_PICTURE_BYTES` and
    :data:`MAX_REQUEST_BYTES` say, the pictures counted base64-encoded."""
    sizes = [_base64_size(len(b.get("png") or b"")) for b in blocks
             if b.get("type") == "image"]
    words = sum(len(str(b.get("text") or "").encode("utf-8"))
                for b in blocks if b.get("type") == "text")
    if len(sizes) > MAX_PICTURES_PER_CALL:
        return (f"{len(sizes)} pictures; one call carries at most "
                f"{MAX_PICTURES_PER_CALL}")
    big = [n for n, size in enumerate(sizes) if size > MAX_PICTURE_BYTES]
    if big:
        return (f"picture {big[0] + 1} is {sizes[big[0]]:,} bytes "
                f"base64-encoded; one picture may be at most "
                f"{MAX_PICTURE_BYTES:,}")
    total = sum(sizes) + words
    if total > MAX_REQUEST_BYTES:
        return (f"the pictures and words come to {total:,} bytes; one "
                f"request may be at most {MAX_REQUEST_BYTES:,}")
    return ""


def picture_blocks(doc: Any, pages: Sequence[int], *,
                   model: Optional[str] = None, budget: Any = None,
                   max_text_chars: Optional[int] = None
                   ) -> List[Dict[str, Any]]:
    """The pictures route: for each page, a line naming its page index, the
    page as a picture, then the page's text between :data:`TEXT_MARKERS`.

    The picture is rendered at the size ``model`` reads without shrinking it
    (``budget`` overrides), as the RI-73 describer renders it. The text is
    what a PDF block would carry, from planlens' frame: the page's text
    layer, or the Azure DI text on a page that has none. A page with no text
    says so, and the picture is then all there is. ``max_text_chars`` cuts a
    longer text with a note (None: the whole text). Whether the blocks can
    go in one call is :func:`pictures_problem`'s to say."""
    from report_ingest.engine import image_block, text_block
    from report_ingest.turned import upright_png

    if budget is None:
        budget = _budget_for(model)
    pages = [int(p) for p in pages]
    out: List[Dict[str, Any]] = []
    for n, page in enumerate(pages):
        text, turn, optical = _page_text(doc, page)
        turned = turn is not None and turn.turned
        out.append(text_block(
            f"Page {page} (picture {n + 1} of {len(pages)}"
            + (f"; its text runs a quarter turn from the page, so the "
               f"picture is shown turned upright" if turned else "")
            + "):"))
        png, _info = doc.render(page, budget=budget, fmt="auto")
        out.append(image_block(upright_png(png, turn) if turned else png))
        out.append(text_block(_text_words(doc, page, text, optical,
                                          max_text_chars)))
    return out


def _page_text(doc: Any, page: int) -> Tuple[str, Any, bool]:
    """``(the page's text, its quarter turn or None, read optically)``: the
    text layer, or the Azure DI text on a page that has none; a sheet
    scanned sideways (RI-149) has its text row by row as the sheet reads."""
    from report_ingest.turned import page_turn, reading_order

    content = doc.page(int(page), tables=False)
    text = content.text() or ""
    sources = sorted(set(getattr(content, "text_sources", []) or []))
    optical = any(s in ("ocr", "azure_di") for s in sources)
    try:
        turn = page_turn(list(content.lines),
                         float(getattr(content, "width", 612.0) or 612.0),
                         float(getattr(content, "height", 792.0) or 792.0))
    except Exception:                            # a page whose lines will not
        turn = None                              # read is sent as it is
    if turn is not None and turn.turned:
        text = "\n".join(str(ln.text or "") for ln in
                         reading_order(list(content.lines), turn))
    return text, turn, optical


def _reliable(doc: Any, page: int) -> bool:
    """Whether a page's text layer can be trusted (planlens' summary; a page
    whose summary will not read is taken at its word)."""
    try:
        return bool(getattr(doc.summary(int(page)), "text_reliable", True))
    except Exception:                            # a summary is a nicety
        return True


def _text_words(doc: Any, page: int, text: str, optical: bool,
                max_text_chars: Optional[int] = None) -> str:
    """A page's text between :data:`TEXT_MARKERS`, saying where it came
    from, or that the page carries none."""
    if not text.strip():
        return f"Page {page} carries no text: read it from the picture."
    cut = max_text_chars is not None and len(text) > int(max_text_chars)
    shown = text[:int(max_text_chars)] if cut else text
    where = ("as an optical reader read it from the picture" if optical
             else "as the PDF's own text layer carries it")
    if not _reliable(doc, page):
        where += (" -- this text layer is UNRELIABLE (a font with no "
                  "character map): trust the picture over it")
    lines = [f"Page {page}'s text, {where}, between the markers:",
             TEXT_MARKERS[0], shown, TEXT_MARKERS[1]]
    if cut:
        lines.append(f"[the text is cut at {int(max_text_chars)} "
                     f"characters; the picture shows the whole page]")
    return "\n".join(lines)


def text_pages(doc: Any, pages: Sequence[int]) -> List[int]:
    """The pages that go as their text alone in a call that sends text
    first (the narrative on the pictures route, RI-175): a reliable text
    layer of at least today's narrative reader's
    ``narrative_reader.MIN_TEXT_CHARS`` characters (a page with less goes
    as a picture there too). A page that will not read is not one."""
    from report_ingest.narrative_reader import MIN_TEXT_CHARS

    out: List[int] = []
    for page in pages:
        try:
            content = doc.page(int(page), tables=False)
        except Exception:                        # a page that will not read
            continue                             # goes as a picture
        chars = sum(len(str(line.text or "")) for line in content.lines)
        if chars >= MIN_TEXT_CHARS and _reliable(doc, page):
            out.append(int(page))
    return out


def page_blocks(doc: Any, page: int, *, as_text: bool,
                model: Optional[str] = None, budget: Any = None
                ) -> List[Dict[str, Any]]:
    """One page's blocks in a call that sends text first (RI-175): a page
    ``as_text`` (:func:`text_pages`) as its text alone, between
    :data:`TEXT_MARKERS` and naming its page index; any other as
    :func:`picture_blocks` sends it, a line naming it, the picture, then
    whatever text it has."""
    from report_ingest.engine import image_block, text_block
    from report_ingest.turned import upright_png

    page = int(page)
    text, turn, optical = _page_text(doc, page)
    if as_text and text.strip():
        return [text_block(_text_words(doc, page, text, optical))]
    if budget is None:
        budget = _budget_for(model)
    turned = turn is not None and turn.turned
    png, _info = doc.render(page, budget=budget, fmt="auto")
    return [text_block(
                f"Page {page} (as a picture: it has too little text to read "
                f"as text"
                + ("; its text runs a quarter turn from the page, so the "
                   "picture is shown turned upright" if turned else "")
                + "):"),
            image_block(upright_png(png, turn) if turned else png),
            text_block(_text_words(doc, page, text, optical))]


# ---------------------------------------------------------------------------
# piece 3: what a description says about identity and sets
# ---------------------------------------------------------------------------

def _described(record: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    """The description itself, from a saved record (``load_descriptions``)
    or a bare description."""
    if not record:
        return {}
    inner = record.get("description")
    return inner if isinstance(inner, Mapping) else record


#: Number signs ("#", "N°", "No.", "Nr:"): dropped, the code after them
#: kept.
_NUMBER_SIGN = re.compile(r"#|\bN\s*[°º]\.?|\bN[OR]\b\.?", re.I)
#: What separates the words of an identifier. A hyphen or an underscore
#: does not: it joins a code ("QX-7", "QX_7").
_SEPARATORS = re.compile(r"[\s.,;:/\\()\[\]{}'\"]+")
#: A code with letters before its number ("QX7", "QX7A") once hyphens are
#: gone.
_LETTERED = re.compile(r"^([A-Z]{1,6})0*(\d{1,5})([A-Z]{0,3})$")
#: A code of letters only ("QX-K").
_LETTERS_ONLY = re.compile(r"^[A-Z]{1,5}-[A-Z]{1,2}$")
#: The longest word joined to the number after it as the code's prefix
#: ("QX 7" is ``QX7``); a longer word names the number ("Hole 7").
_PREFIX_LETTERS = 3
#: The longest text read as one identifier.
_MAX_ID_CHARS = 60
#: A code found on pages that name this many other explorations or more is
#: shared by them -- a sample, a rig, a datum, a standard, a project number
#: -- and never names a page's exploration (:func:`page_identifiers`).
#: LAYOUT JUDGMENT.
SHARED_ACROSS = 2


def exploration_id(text: Any, strict: bool = False) -> Optional[str]:
    """``text``, the name of an exploration, as a key for comparing pages:
    upper case, without the words before the code, the number signs, the
    hyphens and the leading zeros. "Boring QX-04", "QX-4" and "QX 4" are
    all ``QX4``; "QX-K" stays ``QX-K``. A number with no letters of its own
    keeps the word that names it ("Bohrung 3" is ``BOHRUNG 3``), so it is
    never taken for another series' number 3. A name with a second one or a
    note in brackets ("QX-1 (ZW-1)", "Pit K (hand-written)") is read by the
    name outside them, unless only the brackets hold its number ("Hole No.
    (3)"). Text naming two ("QX-1 and QX-2", "Pit K and Pit M") names
    none; so does other text with two numbers ("QX-8 at 10 m", "Sheet 1 of
    2"), except a name the describer's field gives as one code with two
    numbers ("QX-19-001", "QX1-23"), which is folded (:func:`_folded`).
    Letters written apart ("Q. X", "Q-X") are one word.

    This only NORMALISES: what names an exploration is the describer's
    ``exploration`` field. ``strict`` is for text that may name anything --
    an identifier from a page's list, a title -- and refuses what is not
    one code: a bare number, letter or letters-only code with no word
    before it."""
    raw = " ".join(str(text or "").split())
    if not raw or (strict and len(raw) > _MAX_ID_CHARS):
        return None
    outside = " ".join(re.sub(r"\([^()]*\)", " ", raw).split())
    if outside and outside != raw and len(outside) <= _MAX_ID_CHARS:
        key = _code_key(outside, strict)
        numbered_inside = bool(re.search(r"\d", raw)) \
            and not re.search(r"\d", outside)
        if key and (not numbered_inside or re.search(r"[- ]", key)):
            return key
    key = _code_key(raw, strict) if len(raw) <= _MAX_ID_CHARS else None
    if key or strict:
        return key
    # The describer's field names one exploration: a name with two numbers
    # ("QX-19-001", "QX1-23") is still one name, folded.
    read = outside if outside and re.search(r"\d", outside) else raw
    return _folded(read) if len(read) <= _MAX_ID_CHARS else None


#: What joins two names in one text ("QX-1 & QX-2", "QX-1, QX-2", "QX-1
#: and QX-2").
_TWO_NAMES = re.compile(r"[&,+]|\band\b", re.I)


def _folded(text: str) -> Optional[str]:
    """One name that does not read as one code ("QX-19-001", "QX1-23",
    "QX-1/23"), folded to a key: upper case, without the number signs and
    the separators, each number without its leading zeros and kept apart
    from a number next to it by a hyphen ("QX-19-001" is ``QX19-1``). As in
    :func:`exploration_id`, the words before a code with letters of its own
    are dropped, and a code that starts with its number keeps the word
    before it. None for text naming two (:data:`_TWO_NAMES`) or with
    anything after the code."""
    if _TWO_NAMES.search(text):
        return None
    chunks = _NUMBER_SIGN.sub(" ", text.upper()).split()
    numbered = [n for n, c in enumerate(chunks) if re.search(r"\d", c)]
    if not numbered or numbered[0] != len(chunks) - 1:
        return None
    key = ""
    for run in re.findall(r"[A-Z]+|\d+", chunks[-1]):
        if run.isdigit():
            run = str(int(run))
            if key[-1:].isdigit():
                key += "-"
        key += run
    if not key or key[0].isalpha():
        return key or None
    words = _tokens(" ".join(chunks[:-1]))
    word = re.sub(r"[-_]", "", words[-1]) if words else ""
    if not word.isalpha():
        return key
    return word + key if len(word) <= _PREFIX_LETTERS else f"{word} {key}"


def _tokens(text: str) -> List[str]:
    """The words of ``text``; letters written apart ("Q. X", "Q.X") are one
    word, as "Q-X" is."""
    text = _NUMBER_SIGN.sub(" ", text.upper())
    text = re.sub(r"\s*-\s*", "-", text)
    out: List[str] = []
    spelt = False                    # the last word is letters written apart
    for token in _SEPARATORS.split(text):
        token = token.strip("-_")
        if not token:
            continue
        letter = len(token) == 1 and token.isalpha()
        if letter and spelt:
            out[-1] += token
            continue
        out.append(token)
        spelt = letter
    return out


def _code_key(text: str, strict: bool) -> Optional[str]:
    tokens = _tokens(text)
    numbered = [n for n, t in enumerate(tokens) if re.search(r"\d", t)]
    words = [t for t in tokens if t.isalpha() and len(t) > 2]
    if len(numbered) > 1 or len(words) != len(set(words)):
        return None                    # two numbers, or one word named twice
    if not numbered:
        if sum(1 for t in tokens if re.fullmatch(r"[A-Z]{1,2}", t)) > 1:
            return None                # two lettered names ("Pits K and M")
        last = tokens[-1] if tokens else ""
        word = tokens[-2] if len(tokens) > 1 else ""
        if _LETTERS_ONLY.match(last):
            # Alone in a list, letters only may as well be a soil's group
            # symbol ("XY-ZW") as an exploration.
            return None if strict and not word.isalpha() else last
        if re.fullmatch(r"[A-Z]{1,2}", last):
            if word.isalpha():
                return f"{word} {last}"
            return None if strict else last
        return None
    n = numbered[0]
    code = tokens[n]
    if len(re.findall(r"\d+", code)) > 1:
        return None
    word = re.sub(r"[-_]", "", tokens[n - 1]) if n > 0 else ""
    if code[0].isdigit():
        number = re.fullmatch(r"0*(\d{1,5})([A-Z]{0,3})", code)
        if not number:
            return None
        bare = number.group(1) + number.group(2)
        if word.isalpha() and len(word) <= _PREFIX_LETTERS:
            code = word + code
        elif word.isalpha():
            return f"{word} {bare}"
        else:
            return None if strict else bare
    lettered = _LETTERED.match(re.sub(r"[-_]", "", code))
    if not lettered:
        return None
    prefix, number, suffix = lettered.groups()
    return f"{prefix}{int(number)}{suffix}"


def _candidates(d: Mapping[str, Any]) -> List[str]:
    """The codes a description without an ``exploration`` field offers, in
    order: its identifiers, then its title, each read strictly; a code with
    letters of its own ("QX-4") before a number a word names ("Annex 2"),
    which is as often a part of the report as an exploration."""
    out: List[str] = []
    for text in list(d.get("identifiers") or []) + [d.get("title") or ""]:
        key = exploration_id(text, strict=True)
        if key and key not in out:
            out.append(key)
    return sorted(out, key=lambda k: " " in k)


def _same_numbers(a: str, b: str) -> bool:
    numbers = [[int(n) for n in re.findall(r"\d+", k)] for k in (a, b)]
    return bool(numbers[0]) and numbers[0] == numbers[1]


def page_identifiers(pages: Iterable[int],
                     descriptions: Mapping[int, Mapping[str, Any]]
                     ) -> Dict[int, Optional[str]]:
    """``page -> the exploration it logs`` (a key, :func:`exploration_id`),
    or None where the page names none.

    A description with an ``exploration`` field (RI-73's describer from
    prompt ``ri73-3``) is taken at its word. One saved before the field
    existed falls back to the codes it lists: the first of its identifiers,
    then its title, that is one code, a code with letters of its own before
    one a word numbers (:func:`_candidates`) -- passing over a code that
    recurs on pages naming :data:`SHARED_ACROSS` other explorations or
    more, which is a sample, a rig, a datum, a standard or a project number
    that every page prints. The same holds the other way round: a code
    listed first on a run of pages, beside :data:`SHARED_ACROSS` other
    codes or more that would each name a page in its place, is passed over
    too. Only a page that begins a set counts for this (a later sheet lists
    more of one log's samples), and not a code with the same numbers (the
    same exploration written another way). No list of words is consulted,
    so a form no report here prints is read the same way."""
    out: Dict[int, Optional[str]] = {}
    offered: Dict[int, List[str]] = {}
    for page in sorted({int(p) for p in pages}):
        d = _described(descriptions.get(page))
        if "exploration" in d:
            out[page] = exploration_id(d.get("exploration"))
        else:
            offered[page] = _candidates(d)
    shared: set = set()
    named: Dict[int, Optional[str]] = {}
    for _ in range(2 * len(offered) + 2):
        named = {p: next((k for k in keys if k not in shared), None)
                 for p, keys in offered.items()}
        rivalled: set = set()
        beside: Dict[str, set] = {}
        rivals: Dict[str, set] = {}      # within each run a code names
        run: Optional[str] = None
        for page, keys in offered.items():
            hole = named[page]
            if hole != run:
                rivals.pop(hole or "", None)
                run = hole
            if not hole:
                continue
            for key in keys:
                if key != hole:
                    beside.setdefault(key, set()).add(hole)
            # The code that would name the page without the one that does,
            # on a page that begins a set: a later sheet lists more of the
            # same log's samples and runs. A rival with the same numbers is
            # the same exploration written another way ("Pit 6", "QX6").
            sheet = sheet_of(descriptions.get(page))
            rival = next((k for k in keys if k not in shared and k != hole),
                         None)
            if rival and not (sheet and sheet[0] > 1) \
                    and not _same_numbers(rival, hole):
                rivals.setdefault(hole, set()).add(rival)
                if len(rivals[hole]) >= SHARED_ACROSS:
                    rivalled.add(hole)
        more = {k for k, holes in beside.items()
                if len(holes) >= SHARED_ACROSS} - shared
        if not more:
            # A code that names every page of a run it is first on -- listed
            # before each exploration's own -- is shared all the same, once
            # the codes beside it are settled.
            more = rivalled - shared
        if not more:
            break
        shared |= more
    out.update(named)
    return out


def page_identifier(record: Optional[Mapping[str, Any]]) -> Optional[str]:
    """One page's exploration, read alone (:func:`page_identifiers` reads a
    run of pages, which a description without the field needs)."""
    return page_identifiers([0], {0: record or {}})[0]


#: "k of n", "k/n", or k and n around a short word ("2 von 3"), in what a
#: description says about the set. A number run into another or a date
#: ("12/03/2031") is not a sheet.
_SHEET = re.compile(
    r"(?<![\d/.])(\d{1,3})\s*(?:of\b|/|\b[^\W\d_]{2,3}\b)\s*(\d{1,3})"
    r"(?![\d/.])", re.I)
#: The same where an identifier or the title names a sheet or a page.
_SHEET_WORDED = re.compile(
    r"\b(?:sheet|page|pg)\b\.?\s*(?:no\.?|number)?\s*:?\s*(\d{1,3})\s*"
    r"(?:of|/)\s*(\d{1,3})\b", re.I)
#: A continues field that says nothing is printed.
_NOTHING = re.compile(r"^\s*(none|no\b|not\b|nothing|n/?a\b|-+\s*$)", re.I)
#: Words that say a page carries on the one before it.
_CARRIES_ON = re.compile(r"continued\s+from|continuation\s+of|"
                         r"cont'?d\.?\s+from", re.I)
#: The longest set a sheet number is believed for: a larger "of" is the
#: page's place in the file or in a printed code, not a sheet of a set.
MAX_SHEETS = 50


def _sheet_in(text: str, pattern: "re.Pattern[str]"
              ) -> Optional[Tuple[int, int]]:
    for match in pattern.finditer(text):
        k, n = int(match.group(1)), int(match.group(2))
        if 1 <= k <= n <= MAX_SHEETS:
            return k, n
    return None


def sheet_of(record: Optional[Mapping[str, Any]]
             ) -> Optional[Tuple[int, int]]:
    """``(k, n)`` when the page's description says it is sheet k of a set of
    n, else None: from its ``continues`` field, then an identifier or the
    title that names a sheet or page."""
    d = _described(record)
    said = str(d.get("continues") or "")
    if said.strip() and not _NOTHING.match(said):
        found = _sheet_in(said, _SHEET)
        if found:
            return found
    for text in list(d.get("identifiers") or []) + [d.get("title") or ""]:
        found = _sheet_in(str(text), _SHEET_WORDED)
        if found:
            return found
    return None


def carries_on(record: Optional[Mapping[str, Any]]) -> bool:
    """Whether the page's description says in words that it carries on the
    page before ("continued from"). A sheet number past 1 is not enough: a
    run of appendix pages numbered "page 7 of 45" carries nothing on."""
    said = str(_described(record).get("continues") or "")
    return bool(_CARRIES_ON.search(said))


#: A kind that says the page is a divider ("appendix divider page"), not a
#: drawing's divider lines.
_DIVIDER = re.compile(r"\bdivider\b(?!\s+lines?\b)", re.I)


def is_divider(record: Optional[Mapping[str, Any]]) -> bool:
    """Whether the page's description says it is a divider: a tab page
    that opens an appendix or a part of the report and names it
    (BACKLOG RI-177). The labels give a divider its section's label
    (``range_labels``' conventions), so a reader would be sent it with the
    section's pages, or alone, and it holds nothing to read."""
    return bool(_DIVIDER.search(str(_described(record).get("kind") or "")))


def divider_pages(descriptions: Mapping[int, Mapping[str, Any]]
                  ) -> List[int]:
    """The pages the descriptions call dividers (:func:`is_divider`)."""
    return sorted(int(p) for p, record in descriptions.items()
                  if is_divider(record))


def _title_key(record: Optional[Mapping[str, Any]]) -> str:
    """A title folded for comparison: lower case, no digits, no
    punctuation."""
    text = str(_described(record).get("title") or "").lower()
    return " ".join(re.sub(r"[^a-z]+", " ", text).split())


# ---------------------------------------------------------------------------
# piece 3: the groups
# ---------------------------------------------------------------------------

@dataclass
class PageGroup:
    """The pages one reading call carries, and why they start where they
    do."""

    stage: str
    pages: List[int]
    #: The rule that opened this group, in words.
    why: str = ""
    #: What the group is: a log's exploration key, a calc run's first page.
    ident: str = ""
    #: ``(k, n)`` when a long log or run was cut into n parts; every part
    #: carries the same :attr:`key`.
    part: Optional[Tuple[int, int]] = None
    key: str = ""
    notes: List[str] = field(default_factory=list)
    #: The narrative's (RI-175): its pages in the order they are kept when
    #: they do not all fit one request -- the front of the report, then the
    #: pages labelled cover, letter, contents or narrative, then the figures
    #: -- and how many of them, the front, are kept whatever the request's
    #: size. Empty for every other group, whose pages all go.
    ranked: List[int] = field(default_factory=list)
    n_front: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"stage": self.stage, "pages": list(self.pages),
                "why": self.why, "ident": self.ident,
                "part": list(self.part) if self.part else None,
                "key": self.key, "notes": list(self.notes)}


def stage_pages(labels: Mapping[int, str], stage: str) -> List[int]:
    """The pages labelled for one reader, in order."""
    wanted = set(STAGE_LABELS[stage])
    return sorted(int(p) for p, label in labels.items() if label in wanted)


def runs(pages: Iterable[int]) -> List[List[int]]:
    """``pages`` as runs of consecutive pages."""
    out: List[List[int]] = []
    for page in sorted({int(p) for p in pages}):
        if out and page == out[-1][-1] + 1:
            out[-1].append(page)
        else:
            out.append([page])
    return out


def _balanced(pages: List[int], cap: int) -> List[List[int]]:
    """``pages`` in the fewest parts of at most ``cap``, their sizes
    differing by one page at most."""
    if len(pages) <= cap:
        return [pages]
    parts = -(-len(pages) // cap)
    size, extra = divmod(len(pages), parts)
    out: List[List[int]] = []
    start = 0
    for n in range(parts):
        end = start + size + (1 if n < extra else 0)
        out.append(pages[start:end])
        start = end
    return out


def log_groups(pages: Sequence[int],
               descriptions: Mapping[int, Mapping[str, Any]],
               max_pages: int = MAX_LOG_PAGES,
               breaks: Iterable[int] = ()) -> List[PageGroup]:
    """The log pages, one log per group, from their descriptions.

    Walking the pages in order, a page OPENS a new log when a page of
    ``breaks`` (a divider, RI-177) lies between it and the log page before,
    when its description says it is the first sheet of a set ("sheet 1 of
    2"), or when it logs an exploration (:func:`page_identifiers`) other
    than the current log's and does not say in words that it is continued
    from the page before -- after pages that name none, too, unless it is a
    later sheet of their set. Every other page joins the current log: a later
    sheet, a page of the current log's exploration, a page that names
    none. With no
    text layer every scanned log page used to share one planlens key; the
    descriptions name each page's hole from its picture.

    A sheet number past 1 does not keep a page in the log before it: a run
    of appendix pages printed "page 7 of 45" would carry every log on into
    the next.

    A log longer than ``max_pages`` is cut into balanced parts, which keep
    the log's :attr:`PageGroup.key` and are read by one call each."""
    out: List[PageGroup] = []
    current: Optional[PageGroup] = None
    holes: List[str] = []
    pages = sorted(int(p) for p in pages)
    cuts = sorted({int(p) for p in breaks})
    named = page_identifiers(pages, descriptions)
    for page in pages:
        record = descriptions.get(page)
        hole = named.get(page)
        known = hole is not None and hole in holes
        sheet = sheet_of(record)
        why = ""
        if current is None:
            why = "the first log page"
        elif any(current.pages[-1] < cut < page for cut in cuts):
            why = "the first log page after a divider"
        elif sheet and sheet[0] == 1:
            why = f"the first sheet of {sheet[1]}"
        elif hole and not known and not carries_on(record) \
                and (holes or not (sheet and sheet[0] > 1)):
            why = f"a new exploration identifier ({hole})"
        if why:
            current = PageGroup("logs", [page], ident=hole or "", why=why)
            out.append(current)
            holes = [hole] if hole else []
            continue
        if hole and holes and not known:
            current.notes.append(f"page {page} says it is continued from "
                                 f"the page before but names {hole}, not "
                                 f"{holes[0]}")
        current.pages.append(page)
        if hole and not known:
            holes.append(hole)
            current.ident = current.ident or hole
    final: List[PageGroup] = []
    for group in out:
        group.key = f"log-{group.pages[0]}"
        parts = _balanced(group.pages, max(1, int(max_pages)))
        if len(parts) == 1:
            final.append(group)
            continue
        for n, part in enumerate(parts):
            final.append(PageGroup(
                "logs", part, why=group.why if n == 0 else
                f"part {n + 1} of a log longer than {max_pages} pages",
                ident=group.ident, part=(n + 1, len(parts)), key=group.key,
                notes=list(group.notes) + [
                    f"a log of {len(group.pages)} pages, read in "
                    f"{len(parts)} parts"]))
    return final


def lab_starts(pages: Sequence[int],
               descriptions: Mapping[int, Mapping[str, Any]]) -> List[int]:
    """The pages of a run on which a new laboratory sheet begins, by the
    descriptions: a changed title, or an exploration identifier other than
    the page before's. A page that says it is continued from the page
    before never begins one."""
    out: List[int] = []
    pages = [int(p) for p in pages]
    named = page_identifiers(pages, descriptions)
    before: Optional[int] = None
    for page in pages:
        record = descriptions.get(page)
        if before is not None and not carries_on(record):
            hole, prior = named.get(page), named.get(before)
            if _title_key(record) != _title_key(descriptions.get(before)) \
                    or (hole and prior and hole != prior):
                out.append(page)
        before = page
    return out


def lab_groups(pages: Sequence[int],
               descriptions: Mapping[int, Mapping[str, Any]],
               size: Optional[int] = None) -> List[PageGroup]:
    """Each run of laboratory pages in windows of at most ``size``
    (``lab_reader.MAX_LAB_PAGES``), the cuts moved to the pages where a new
    sheet begins (:func:`lab_starts`) as ``lab_reader.lab_windows`` moves
    them."""
    from report_ingest.lab_reader import MAX_LAB_PAGES, lab_windows

    size = MAX_LAB_PAGES if size is None else int(size)
    out: List[PageGroup] = []
    for run in runs(pages):
        starts = lab_starts(run, descriptions)
        windows = lab_windows(run, size, starts)
        for n, window in enumerate(windows):
            why = ("a run of laboratory pages" if n == 0 else
                   "a new sheet begins here" if window[0] in starts else
                   "an even cut of a long run")
            out.append(PageGroup("lab", list(window), why=why,
                                 key=f"lab-{run[0]}",
                                 part=((n + 1, len(windows))
                                       if len(windows) > 1 else None)))
    return out


def calc_starts(pages: Sequence[int],
                descriptions: Mapping[int, Mapping[str, Any]]) -> List[int]:
    """The pages of a run on which a description says a new printout
    begins: the first sheet of a set of two or more ("sheet 1 of 7"), or a
    one-sheet set ("page 1 of 1") whose title differs from the page
    before's -- a program that prints "page 1 of 1" on every page of one
    printout keeps its title. A page with no sheet number begins nothing."""
    out: List[int] = []
    before: Optional[Mapping[str, Any]] = None
    for n, page in enumerate(pages):
        record = descriptions.get(int(page))
        sheet = sheet_of(record)
        if n and sheet and sheet[0] == 1 and not carries_on(record) \
                and (sheet[1] > 1 or _title_key(record)
                     != _title_key(before)):
            out.append(int(page))
        before = record
    return out


def _cut_at(pages: List[int], starts: Iterable[int]) -> List[List[int]]:
    """``pages`` cut before each of ``starts``."""
    wanted = {int(p) for p in starts}
    out: List[List[int]] = []
    for page in pages:
        if not out or page in wanted:
            out.append([page])
        else:
            out[-1].append(page)
    return out


def calc_groups(pages: Sequence[int], doc: Any = None,
                descriptions: Optional[Mapping[int, Mapping[str, Any]]] = None
                ) -> List[PageGroup]:
    """Each run of calculation pages, split into printouts.

    Two signals, and a new printout begins where EITHER says so:

    * the descriptions' sheet numbers (:func:`calc_starts`), which see a
      printout's "sheet 1 of n" even where two printouts carry one title;
    * today's page heads (``calc_reader.split_calc_runs``, when the document
      is at hand): a title naming a different calculation, a different
      program or a fresh banner, which see a program's output that numbers
      no sheets.

    A printout longer than two calls show (``calc_reader.MAX_RUN_PAGES``) is
    cut into balanced parts, as today. With the document, today's split
    runs over the whole run first -- its heads carry a title from page to
    page, so a run cut short first would lose it -- and the descriptions'
    starts cut its printouts further."""
    from report_ingest.calc_reader import MAX_RUN_PAGES, split_calc_runs

    out: List[PageGroup] = []
    for run in runs(pages):
        starts = calc_starts(run, descriptions) if descriptions else []
        if doc is not None:
            parts = [piece for part in split_calc_runs(doc, run)
                     for piece in _cut_at(part, starts)]
        else:
            parts = [part for piece in _cut_at(run, starts)
                     for part in _balanced(piece, MAX_RUN_PAGES)]
        for part in parts:
            why = ("a run of calculation pages" if part[0] == run[0]
                   else "the first sheet of a set, by its description"
                   if part[0] in starts else
                   "a new printout by its head, or a balanced cut")
            out.append(PageGroup("calc", list(part), why=why,
                                 key=f"calc-{part[0]}"))
    return out


def calc_calls(group: PageGroup) -> List[List[int]]:
    """The pages each call of one calculation group carries: one call up to
    ``MAX_CALC_PAGES`` pages, otherwise two windows with the first and the
    last page in both (``calc_reader.page_window``)."""
    from report_ingest.calc_reader import MAX_CALC_PAGES, page_window

    pages = list(group.pages)
    if len(pages) <= MAX_CALC_PAGES:
        return [pages]
    first = page_window(pages, 0)
    second = page_window(pages, max(0, MAX_CALC_PAGES - 2))
    return [first] + ([second] if second else [])


def narrative_groups(labels: Mapping[int, str], n_pages: int, *,
                     front_pages: Optional[int] = None,
                     max_pages: int = MAX_NARRATIVE_PAGES,
                     page_chars: Optional[Mapping[int, int]] = None,
                     max_chars: Optional[int] = None,
                     window: Optional[Sequence[int]] = None,
                     route: str = "pdf",
                     set_aside: Iterable[int] = ()
                     ) -> List[PageGroup]:
    """The narrative's pages in consecutive parts.

    The pages are ``narrative_reader.reading_pages``' rule with the range
    labels as its roles -- every page labelled cover, letter, contents or
    narrative, plus the first ``front_pages`` pages whatever their label --
    and the report's own figure, plan and profile pages, less the pages
    ``set_aside`` (the dividers, RI-177). Up to
    ``max_pages`` of them go in one call; more are read in consecutive
    parts of near one size. With ``page_chars`` and ``max_chars`` a part is
    also cut where its pages' text would pass the budget.

    ``window`` is a report BOUND INSIDE another, as ``reading_pages`` takes
    it: only the window's pages are read, and its front is the first
    ``front_pages`` pages OF IT, not of the file.

    Each part ranks its pages (:attr:`PageGroup.ranked`, RI-175) the way
    today's reader keeps them inside its budget: the front first, kept
    whatever the request's size, then the pages labelled cover, letter,
    contents or narrative, then the figures, each in page order. The reader
    leaves out the last-ranked pages that do not fit one request.

    On the ``pictures`` route the narrative is ONE part, whatever
    ``max_pages`` and ``max_chars`` say (RI-175): its call sends text
    first, so the request's own limit decides what goes
    (``model_first_read._Caller.fit``), leaving out rather than splitting;
    a front too large for one request is the only split, and it says so."""
    from report_ingest.narrative_reader import FRONT_PAGES

    front = FRONT_PAGES if front_pages is None else int(front_pages)
    total = int(n_pages)
    span = (sorted({int(p) for p in window if 0 <= int(p) < total})
            if window is not None else list(range(total)))
    inside = set(span)
    prose = {int(p) for p, label in labels.items()
             if label in NARRATIVE_LABELS and int(p) in inside}
    figures = {int(p) for p, label in labels.items()
               if label in FIGURE_LABELS and int(p) in inside}
    first = set(span[:max(0, front)])
    pages = sorted((prose | figures | first) - {int(p) for p in set_aside})
    if not pages:
        return []
    if route == "pictures":
        max_pages, max_chars = len(pages), None
    parts = _balanced(pages, max(1, int(max_pages)))
    if page_chars is not None and max_chars:
        cut: List[List[int]] = []
        for part in parts:
            used = 0
            cut.append([])
            for page in part:
                size = int(page_chars.get(page, 0))
                if cut[-1] and used + size > int(max_chars):
                    cut.append([])
                    used = 0
                cut[-1].append(page)
                used += size
        parts = cut

    def ranked(part: List[int]) -> Tuple[List[int], int]:
        tiers = ([p for p in part if p in first],
                 [p for p in part if p not in first and p in prose],
                 [p for p in part if p not in first and p not in prose])
        return [p for tier in tiers for p in tier], len(tiers[0])

    out: List[PageGroup] = []
    for n, part in enumerate(parts):
        order, n_front = ranked(part)
        out.append(PageGroup(
            "narrative", part, key="narrative",
            why=("the narrative and the front of the report" if n == 0
                 else "the narrative, continued"),
            part=(n + 1, len(parts)) if len(parts) > 1 else None,
            ranked=order, n_front=n_front))
    return out


def group_pages(labels: Mapping[int, str],
                descriptions: Mapping[int, Mapping[str, Any]], *,
                n_pages: int, doc: Any = None, route: str = "pdf",
                window: Optional[Sequence[int]] = None,
                front_pages: Optional[int] = None,
                page_chars: Optional[Mapping[int, int]] = None,
                max_chars: Optional[int] = None
                ) -> Dict[str, List[PageGroup]]:
    """Every reader's groups for one report: ``logs``, ``lab``, ``calc``
    and ``narrative``. ``window`` narrows all of them to a report bound
    inside another (:func:`narrative_groups`); ``front_pages``,
    ``page_chars`` and ``max_chars`` are the narrative's.

    A page the descriptions call a divider (:func:`divider_pages`,
    RI-177) is in no group, whatever its label, and a group never runs
    across one: the laboratory and calculation runs end at it, a log
    begins after it, and the narrative leaves it out.

    On the pdf ``route`` the narrative's parts are
    :data:`MAX_NARRATIVE_PAGES` long and cut on the text budget; on the
    pictures route it is one part (RI-175): a narrative page with a usable
    text layer goes as its text alone (:func:`page_blocks`), so the request
    limit and the picture cap are the request's to keep (the reader leaves
    out the last-ranked pages), not the part's."""
    aside = divider_pages(descriptions)
    labels = {int(p): str(label) for p, label in labels.items()
              if label and int(p) not in aside}
    if window is not None:
        inside = {int(p) for p in window}
        labels = {p: label for p, label in labels.items() if p in inside}
    return {
        "logs": log_groups(stage_pages(labels, "logs"), descriptions,
                           breaks=aside),
        "lab": lab_groups(stage_pages(labels, "lab"), descriptions),
        "calc": calc_groups(stage_pages(labels, "calc"), doc, descriptions),
        "narrative": narrative_groups(
            labels, n_pages, front_pages=front_pages,
            max_pages=MAX_NARRATIVE_PAGES, page_chars=page_chars,
            max_chars=max_chars, window=window, route=route,
            set_aside=aside),
    }
