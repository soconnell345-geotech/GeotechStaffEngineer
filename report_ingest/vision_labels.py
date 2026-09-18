"""The experiment: label a page by LOOKING at it, and nothing else.

planlens' rules label a page from what that page prints about itself, and
:mod:`report_ingest.label_review` corrects them with the whole report in
view. Both read TEXT. This module asks a different question, and asks it of
the cheapest model on the tier list: given the page as a PICTURE and the
eighteen-label vocabulary, what is this page?

WHY IT IS WORTH MEASURING. The rules reach 0.91 accuracy on the reports they
were built on and 0.79 on the next report anybody opens, and their worst
pages are the ones with no readable text at all -- a scanned appendix, a
plotted log, a photographed pit face. A rule cannot fire on those and the
review has to spend a tool call rendering each one. A vision-first pass
looks at every page by construction, so the pages the rules are worst at are
the ones it is under no handicap on. Whether that is worth its cost is a
measurement, not an opinion, and :mod:`report_ingest.cluster_scoring` makes
it against the SAME hand labels and the SAME scorer as the rules.

TWO MODES, AND THE TRADE BETWEEN THEM.

``mode="page"``
    One call per page, the page rendered whole. Every page gets the model's
    full attention and the page's own fine print is legible. A hundred pages
    is a hundred calls.
``mode="sheet"``
    One call per contact sheet of several pages, each thumbnail carrying its
    own page index, exactly as the label review's ``contact_sheet`` tool
    draws them. A hundred pages is seventeen calls at six a sheet, and each
    page is a thumbnail rather than a page. What that costs in accuracy is
    the thing being measured.

``outline_context=True`` prepends what the document prints about ITSELF --
the contents list, the lists of figures, tables and appendices, and every
divider and fly sheet -- to each call, so a run can put pure vision beside
vision that knows what appendix it is standing in.

WHAT PYTHON REFUSES, rather than passes on. A label outside the vocabulary
becomes ``other`` and a QA note says what the model said; a confidence
outside 0 to 1 is clipped; a page the reply never mentioned is
``unresolved`` and is never guessed at from its neighbours; a reply naming a
page that was not on the sheet is dropped with a note. The budget caps model
calls and every page past it is ``unresolved`` too, so a capped run is
visibly incomplete instead of quietly short.

THE DPI, AND WHY IT IS WHAT IT IS. A 4.1-class vision stack scales an image
to fit its own working size before it looks at it or charges for it: the
short side lands at about 768 px, and the price is counted in 512 px tiles
of what is left. :data:`DEFAULT_DPI` is 100, which puts a letter page at
850 x 1100 and an A4 page at 827 x 1169, both just above that ceiling.

The arithmetic is the whole argument, and it is one-sided. A letter page at
72 dpi is 612 x 792, which is UNDER the ceiling and so is not scaled at all:
four tiles, about 765 tokens. The same page at 100 dpi is scaled down to
768 x 994: four tiles, about 765 tokens. **The same price for a quarter more
pixels on the short side**, so 72 dpi is not the cheap option, it is the
same option with detail thrown away. Going the other way, 200 dpi is scaled
to the same 768 x 994 and costs the same tokens again, for four times the
bytes on the wire and not one pixel the model keeps. Everything from about
90 dpi upwards lands on that identical 768 px short side; 100 leaves room
for a page that is not letter-sized. ``dpi`` is a parameter because the
ceiling is the provider's, not ours, and it moves.

NO CONSTRAINTS IN THE SCHEMA. ``confidence`` is described as 0 to 1 rather
than declared with a minimum and a maximum, and ``reason`` is asked for in
twenty words rather than capped with a maxLength. Strict structured output
refuses schema keywords it does not implement, and a rejected call returns
nothing at all, while a number out of range is a thing Python can clip. The
gate belongs here, not in the schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from report_ingest.engine import Engine, text_block, user
from report_ingest.label_review import LABEL_DEFINITIONS

__all__ = [
    "VISION_LABELS", "MODES", "DEFAULT_DPI", "SHEET_PAGES", "SHEET_COLUMNS",
    "SHEET_THUMB_PX", "MAX_CONTEXT_CHARS",
    "VisionPageAnswer", "VisionSheetAnswer", "VisionPageLabel", "VisionLabels",
    "vision_system", "classify_pages_by_vision",
]

#: The vocabulary, and it is the label review's: a label this pass can return
#: that planlens cannot produce, or the other way round, would read as a miss
#: on every page carrying it. The one-line definitions are the review's too,
#: for the same reason -- two wordings of one vocabulary are two vocabularies,
#: and the second one's numbers would be the ones nobody checked.
VISION_LABELS: Tuple[str, ...] = tuple(LABEL_DEFINITIONS)

MODES: Tuple[str, ...] = ("page", "sheet")

#: See the module docstring: the largest render a 4.1-class vision stack will
#: keep, for a letter page and for A4.
DEFAULT_DPI = 100.0
#: Pages per call in ``sheet`` mode, and how they are laid out. Six in three
#: columns is two rows of thumbnails at planlens' largest thumbnail size.
SHEET_PAGES = 6
SHEET_COLUMNS = 3
SHEET_THUMB_PX = 400
#: Ceiling on the outline text when ``outline_context`` is on. It rides on
#: EVERY call in page mode, so a 30,000-character outline would cost more than
#: the pictures do.
MAX_CONTEXT_CHARS = 6000


# -- what the model returns -------------------------------------------------

class VisionPageAnswer(BaseModel):
    """One page, as the model read it off the picture."""

    page: int = Field(
        description="the 0-based page index of the page you are answering "
                    "for, as printed under the thumbnail or given to you")
    label: Literal[VISION_LABELS] = Field(     # type: ignore[valid-type]
        description="what this page IS; one of the eighteen labels")
    confidence: float = Field(
        description="how sure you are, from 0.0 to 1.0: 1.0 for a page that "
                    "names itself, about 0.5 for a guess from the layout, low "
                    "for a page you cannot make out")
    reason: str = Field(
        description="what on the page decided it, in 20 words or fewer")


class VisionSheetAnswer(BaseModel):
    """Every page on one contact sheet, one entry each."""

    pages: List[VisionPageAnswer] = Field(
        description="one entry for every page shown on the sheet and none "
                    "for any page that is not on it, in the order they are "
                    "shown")


# -- what this pass returns -------------------------------------------------

@dataclass(frozen=True)
class VisionPageLabel:
    """One page's label, as this pass settled it."""

    page: int
    label: str
    confidence: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"page": self.page, "label": self.label,
                "confidence": round(self.confidence, 3), "reason": self.reason}


@dataclass
class VisionLabels:
    """The labels a vision-first pass produced, and what it cost."""

    labels: List[VisionPageLabel]
    cost: Dict[str, Any] = field(default_factory=dict)
    unresolved: List[Dict[str, Any]] = field(default_factory=list)
    #: What Python refused or repaired, one line each. A label outside the
    #: vocabulary, a confidence out of range, a page the reply invented.
    qa: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = "page"
    dpi: float = DEFAULT_DPI
    outline_context: bool = False
    pages_asked: int = 0
    model_calls: int = 0
    budget: Optional[int] = None
    stopped_on_budget: bool = False
    model: str = ""

    @property
    def label_map(self) -> Dict[int, str]:
        """``page -> label``, which is what a scorer compares."""
        return {row.page: row.label for row in self.labels}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": {str(row.page): row.label for row in self.labels},
            "detail": [row.to_dict() for row in self.labels],
            "unresolved": [dict(u) for u in self.unresolved],
            "qa": [dict(q) for q in self.qa],
            "mode": self.mode,
            "dpi": round(float(self.dpi), 1),
            "outline_context": bool(self.outline_context),
            "pages_asked": self.pages_asked,
            "model_calls": self.model_calls,
            "budget": self.budget,
            "stopped_on_budget": self.stopped_on_budget,
            "model": self.model,
            "cost": dict(self.cost),
        }


# -- the prompt -------------------------------------------------------------

_SYSTEM_HEAD = """\
You are looking at pages of one engineering report and saying what each page
IS. You are given the pages as PICTURES. There is no text layer, no rule
label and no neighbouring context beyond what you can see: answer from the
page in front of you.

THE LABEL VOCABULARY. Every page gets exactly one of these eighteen.
"""

_SYSTEM_RULES = """
How to decide.

- Decide what the page IS, not what it is about. Prose discussing the
  laboratory results is 'narrative'; the laboratory's own result sheet is
  'lab_test'.
- If a page carries ONE exploration's own results, name that exploration --
  boring_log, test_pit_log, cpt_log or dcp_log -- and never figure. These
  results are often PLOTTED rather than tabulated, and a plot of one
  exploration's results is still that exploration's log however much it
  looks like a graph. 'figure' is for the report's own numbered figure
  series. When you can see it is an exploration's own results but not which
  kind, decide on what is being MEASURED, not on how it is drawn: blows per
  increment of penetration against depth is a DCP; continuous tip
  resistance, sleeve friction and pore pressure against depth is a CPT; a
  logged pit or trench face is a test pit; a drilled hole with driven
  samples and blow counts is a boring.
- 'other' is a real answer. Use it when the page is none of the above, or
  when you cannot tell from the picture, and say which in the reason.
- confidence is your own: 1.0 for a page that names itself, about 0.5 for a
  guess from the layout, low for a page you cannot make out. A low
  confidence costs nothing; a confident wrong answer costs a reader.
- reason: 20 words or fewer, naming what on the page decided it.
"""

_PAGE_TAIL = """
You are given ONE page at a time. Answer for that page, with the page index
you were given.
"""

_SHEET_TAIL = """
You are given a CONTACT SHEET of several pages laid out in a grid. Each
thumbnail is labelled beneath it with its 0-based page index and the page's
shape. Answer for EVERY page on the sheet, once each, using the page index
printed under that thumbnail. Do not answer for a page that is not on the
sheet, and do not leave one out: a page you cannot make out is 'other' at a
low confidence, which is an answer, and a page you skip is a hole.
"""


def vision_system(mode: str = "page") -> str:
    """The system prompt for one mode, vocabulary and all."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; the modes are {list(MODES)}")
    vocab = "\n".join(f"  {name}: {text}"
                      for name, text in LABEL_DEFINITIONS.items())
    tail = _PAGE_TAIL if mode == "page" else _SHEET_TAIL
    return _SYSTEM_HEAD + vocab + "\n" + _SYSTEM_RULES + tail


def _context_text(doc, outline: Any, max_chars: int) -> str:
    """What the document prints about itself, as lines a model reads."""
    from planlens.document.roles import document_outline

    from report_ingest.triage import outline_text

    if outline is None:
        outline = document_outline(doc)
    return outline_text(outline, max_chars=max_chars)


# -- the gates --------------------------------------------------------------

def _clip(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:                       # NaN
        return 0.0
    return min(1.0, max(0.0, number))


def _accept(answer: Any, page: int, qa: List[Dict[str, Any]]
            ) -> VisionPageLabel:
    """One answer through the Python gates, onto one page.

    ``page`` is authoritative: in page mode it is the page that was rendered,
    and in sheet mode it is the page the entry was matched to. A label
    outside the vocabulary becomes ``other`` and says so; a confidence
    outside 0 to 1 is clipped.
    """
    label = str(getattr(answer, "label", "") or "")
    reason = str(getattr(answer, "reason", "") or "")
    if label not in LABEL_DEFINITIONS:
        qa.append({"page": page, "note": f"label {label!r} is not in the "
                                         f"vocabulary; recorded as 'other'"})
        label = "other"
    raw = getattr(answer, "confidence", 0.0)
    confidence = _clip(raw)
    try:
        if float(raw) != confidence:
            qa.append({"page": page,
                       "note": f"confidence {raw} clipped to {confidence}"})
    except (TypeError, ValueError):
        qa.append({"page": page,
                   "note": f"confidence {raw!r} is not a number; read as 0.0"})
    return VisionPageLabel(page=page, label=label, confidence=confidence,
                           reason=reason)


# -- the pass ---------------------------------------------------------------

def classify_pages_by_vision(doc, engine: Engine, *,
                             pages: Any = None,
                             mode: str = "page",
                             dpi: float = DEFAULT_DPI,
                             budget: Optional[int] = None,
                             outline_context: bool = False,
                             sheet_pages: int = SHEET_PAGES,
                             outline: Any = None,
                             max_context_chars: int = MAX_CONTEXT_CHARS
                             ) -> VisionLabels:
    """Label pages from their pictures alone.

    Parameters
    ----------
    doc
        An open :class:`planlens.document.Document`.
    engine
        Anything satisfying :class:`report_ingest.engine.Engine`. The
        experiment is aimed at the cheapest Funhouse tier
        (``funhouse-gpt-low``), which is why it exists at all.
    pages
        Which pages to label: ``None`` for all of them, or anything
        planlens accepts -- an index, ``"0-9,14"``, or a sequence.
    mode
        ``"page"`` for one call per page, ``"sheet"`` for one call per
        contact sheet of ``sheet_pages`` pages.
    dpi
        What a page is rendered at in page mode. See the module docstring
        for why the default is 100.
    budget
        A ceiling on MODEL CALLS. Every page past it comes back
        ``unresolved`` rather than unlabelled-and-unmentioned.
    outline_context
        Prepend what the document prints about itself to every call.
    sheet_pages
        Pages per contact sheet in sheet mode.
    outline
        :func:`planlens.document.roles.document_outline` when the caller
        already has it; only read when ``outline_context`` is on.
    """
    from planlens.document.document import parse_pages

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; the modes are {list(MODES)}")
    wanted = parse_pages(pages, doc.n_pages)
    system = vision_system(mode)
    context = (_context_text(doc, outline, max_context_chars)
               if outline_context else "")

    labels: List[VisionPageLabel] = []
    unresolved: List[Dict[str, Any]] = []
    qa: List[Dict[str, Any]] = []
    spent = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "seconds": 0.0, "dollars": 0.0}
    model_name = ""
    stopped = False

    def charge(reply) -> None:
        nonlocal model_name
        spent["calls"] += 1
        spent["input_tokens"] += reply.usage.input_tokens
        spent["output_tokens"] += reply.usage.output_tokens
        spent["cache_read_tokens"] += reply.usage.cache_read_tokens
        spent["seconds"] += reply.seconds
        spent["dollars"] += reply.usage.dollars(reply.model)
        model_name = reply.model or model_name

    def out_of_budget() -> bool:
        return budget is not None and spent["calls"] >= int(budget)

    chunks: List[List[int]]
    if mode == "page":
        chunks = [[page] for page in wanted]
    else:
        step = max(1, int(sheet_pages))
        chunks = [wanted[i:i + step] for i in range(0, len(wanted), step)]

    for chunk in chunks:
        if out_of_budget():
            stopped = True
            for page in chunk:
                unresolved.append({
                    "page": page,
                    "why": f"the budget of {budget} model call(s) was spent "
                           f"before this page"})
            continue
        if mode == "page":
            _one_page(doc, engine, chunk[0], system, context, dpi,
                      labels, unresolved, qa, charge, len(wanted))
        else:
            _one_sheet(doc, engine, chunk, system, context,
                       labels, unresolved, qa, charge)

    spent["seconds"] = round(spent["seconds"], 1)
    spent["dollars"] = round(spent["dollars"], 4)
    return VisionLabels(
        labels=labels,
        cost=spent,
        unresolved=unresolved,
        qa=qa,
        mode=mode,
        dpi=float(dpi),
        outline_context=bool(outline_context),
        pages_asked=len(wanted),
        model_calls=spent["calls"],
        budget=None if budget is None else int(budget),
        stopped_on_budget=stopped,
        model=model_name or getattr(engine, "name", ""),
    )


def _ask(engine: Engine, body: str, system: str, png: bytes,
         output_format: Any, charge) -> Any:
    """One call: the words, the picture, the shape the answer must take."""
    reply = engine.complete([user(text_block(body))], system=system,
                            images=[png], output_format=output_format)
    charge(reply)
    return reply


def _one_page(doc, engine: Engine, page: int, system: str, context: str,
              dpi: float, labels: List[VisionPageLabel],
              unresolved: List[Dict[str, Any]], qa: List[Dict[str, Any]],
              charge, n_pages: int) -> None:
    png, info = doc.render(page, dpi=float(dpi))
    body = "\n".join(filter(None, [
        ("WHAT THIS DOCUMENT PRINTS ABOUT ITSELF\n" + context + "\n"
         if context else ""),
        f"The picture is page {page} of this document, rendered at "
        f"{info['dpi']:.0f} dpi. The document has {doc.n_pages} pages. Say "
        f"what page {page} is.",
    ]))
    reply = _ask(engine, body, system, png, VisionPageAnswer, charge)
    answer = reply.parsed
    if answer is None:
        unresolved.append({"page": page,
                           "why": f"the model returned no structured answer "
                                  f"(stop_reason {reply.stop_reason!r})"})
        return
    got = getattr(answer, "page", page)
    try:
        if int(got) != page:
            qa.append({"page": page,
                       "note": f"the answer named page {got}; it was asked "
                               f"about page {page} and is recorded there"})
    except (TypeError, ValueError):
        pass
    labels.append(_accept(answer, page, qa))


def _one_sheet(doc, engine: Engine, chunk: Sequence[int], system: str,
               context: str, labels: List[VisionPageLabel],
               unresolved: List[Dict[str, Any]], qa: List[Dict[str, Any]],
               charge) -> None:
    shown = list(chunk)
    sheets = doc.render_thumbnails(pages=shown, columns=min(SHEET_COLUMNS,
                                                            len(shown)),
                                   thumb_px=SHEET_THUMB_PX,
                                   per_sheet=len(shown))
    if not sheets:
        for page in shown:
            unresolved.append({"page": page,
                               "why": "no contact sheet could be rendered"})
        return
    png, info = sheets[0]
    body = "\n".join(filter(None, [
        ("WHAT THIS DOCUMENT PRINTS ABOUT ITSELF\n" + context + "\n"
         if context else ""),
        f"The picture is a contact sheet of pages "
        f"{', '.join(str(p) for p in shown)} of this document, which has "
        f"{doc.n_pages} pages. {info.get('legend', '')} Say what each of "
        f"those {len(shown)} pages is, one answer per page.",
    ]))
    reply = _ask(engine, body, system, png, VisionSheetAnswer, charge)
    answer = reply.parsed
    if answer is None:
        for page in shown:
            unresolved.append({
                "page": page,
                "why": f"the model returned no structured answer for this "
                       f"sheet (stop_reason {reply.stop_reason!r})"})
        return

    by_page: Dict[int, Any] = {}
    for entry in (getattr(answer, "pages", None) or []):
        try:
            page = int(getattr(entry, "page"))
        except (TypeError, ValueError, AttributeError):
            qa.append({"page": None,
                       "note": "an entry on this sheet named no page index "
                               "and was dropped"})
            continue
        if page not in shown:
            qa.append({"page": page,
                       "note": f"the reply named page {page}, which is not on "
                               f"this sheet; dropped rather than guessed"})
            continue
        if page in by_page:
            qa.append({"page": page,
                       "note": "the reply answered for this page twice; the "
                               "first answer is the one kept"})
            continue
        by_page[page] = entry

    for page in shown:
        entry = by_page.get(page)
        if entry is None:
            unresolved.append({
                "page": page,
                "why": "the reply left this page out of the sheet it was "
                       "shown on"})
            continue
        labels.append(_accept(entry, page, qa))
