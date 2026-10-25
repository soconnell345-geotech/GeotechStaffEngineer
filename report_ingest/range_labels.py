"""Page labels by whole ranges, from a description of every page (RI-73).

THE IDEA, AND WHY IT IS A FOURTH VOTER. The rules, the vision pass and the
label review each decide a page by what that page shows. The hand labels the
corpus is scored against decide it by the SECTION it sits in: a test pit's
photographs are that test pit's log, a calculation's plotted output is the
calculation, a laboratory report's certificate is a laboratory sheet. The
offline analysis of 2026-10-24 (stage b, 13 hand-labelled reports, 3,418
pages) put 43 % of the pages sent to the wrong reader down to that,
and a further 21 % to prose inside an appendix that the hand labels call
``other``: neither is a misreading of the page, and the review, a strong
model that looked at the page, did not infer either convention on its own.
So this pass does two things the others cannot:

1. :func:`describe_pages` -- ONE vision call per page, the page as a picture
   at the size the model looks at without shrinking it, with the page's own
   text (or the optical reader's, where the page has none), its place in the
   file and the number it prints on itself. The model DESCRIBES the page --
   its title as printed, what kind of page it is in plain words, the test,
   method or program it names, the identifiers it prints, whether it
   continues another page, its running header, its language, one sentence
   on what it shows -- and labels nothing. Each description is saved as it
   is made, so a run resumes.
2. :func:`label_by_ranges` -- ONE call per report (or per overlapping window
   of a long one): a model reads every description in order, with the
   outline the document prints about itself, and answers in SECTIONS (first
   page, last page, label, reason) plus any one-page exceptions. The
   labelling conventions are given in writing (:data:`LABEL_CONVENTIONS`),
   because without them a reader of the whole sequence would label by look
   as the review does. Python checks that the sections cover every page
   once, with no gap and no overlap.

TWO ARMS. ``plain`` gives the labeller the descriptions alone; ``hinted``
adds the rules' label on every line as a hint. Which is better is a
measurement (the ``range_labels`` stage of :mod:`report_ingest.
cluster_scoring`, scored by the reader each page is sent to).

THE CONVENTIONS ARE ONE REPLACEABLE TEXT. :data:`LABEL_CONVENTIONS` is the
recommended default of BACKLOG OWN-92, which asks the owner to settle them;
until the owner answers, this is what the labeller is told, and a run records
which text it was given (:func:`conventions_version`). A second text,
:data:`LABEL_CONVENTIONS_B`, is what the hand labels mostly do; it is there
to measure whether the labeller follows the conventions it is given, so that
OWN-92 can be answered on numbers. :data:`CONVENTIONS` names both.

BEHIND A SWITCH, OFF. In the graph the pass runs only when
``GEOTECH_INGEST_RANGE_LABELS`` (or ``ingest_report(range_labels=...)``)
names an arm; then its label settles every page it labelled
(:func:`report_ingest.label_vote.combine`). Off, no call is made and
the vote is what it was.

WHAT PYTHON REFUSES, rather than passes on: a label outside the vocabulary
becomes ``other`` with a QA note; a page no section covers, or two sections
cover with different labels, is asked about once more (the model is told
which pages) and is otherwise left UNRESOLVED -- never guessed from its
neighbours; a page outside the window is dropped with a note. A second
answer is taken only if it leaves fewer pages unsettled than the first.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterator, List, Mapping, Optional,
                    Sequence, Tuple)

from pydantic import BaseModel, Field, field_validator, model_validator

from report_ingest.engine import Engine, assistant, image_block, text_block, \
    user
from report_ingest.label_review import LABEL_DEFINITIONS

__all__ = [
    "ARMS", "RANGE_LABELS_ENV", "LABEL_CONVENTIONS", "DESCRIBE_SYSTEM",
    "DESCRIBE_PROMPT_VERSION", "MAX_PAGE_TEXT_CHARS", "DEFAULT_WINDOW",
    "DEFAULT_OVERLAP", "MIN_WINDOW", "PageDescription", "RangeSection",
    "RangeException", "RangeAnswer", "PageDescriptions", "RangeLabels",
    "describe_pages", "label_by_ranges", "labeller_system",
    "description_line", "check_sections", "unsettled", "balanced_windows",
    "range_labels_arm", "range_arm_of", "use_range_labels",
    "conventions_version", "load_descriptions", "LABELS_NOT_USED",
    "labeller_fingerprint", "labeller_vocabulary", "LABEL_CONVENTIONS_B",
    "LABELS_NOT_USED_B", "CONVENTIONS", "conventions_named",
]

#: The labeller's two arms: the descriptions alone, or with the rules' label
#: on every line as a hint.
ARMS: Tuple[str, ...] = ("plain", "hinted")

#: The switch: unset, ``0`` or ``off`` is off (the default); ``plain`` (or
#: ``1`` / ``on``) and ``hinted`` name an arm.
RANGE_LABELS_ENV = "GEOTECH_INGEST_RANGE_LABELS"

#: The page text a description call is given, at most; a longer page is cut
#: with a note, and the picture still shows all of it.
MAX_PAGE_TEXT_CHARS = 6000

#: Pages the labeller is shown in one call, and the overlap between two
#: windows of a report longer than that. A report up to this long is one
#: call. A window whose answer stops at the output cap is halved (down to
#: :data:`MIN_WINDOW`) and asked again.
DEFAULT_WINDOW = 300
DEFAULT_OVERLAP = 40
MIN_WINDOW = 60

#: Raised by a meter or a ceiling to say no further call may be made; the
#: describer stops sending pages when it sees one. By class name, because
#: the live harness's cap error is not this package's to import.
STOP_ERRORS: Tuple[str, ...] = ("CallCeilingReached", "WaveCapReached")

#: Changed whenever what the describer is asked changes, so a saved
#: description from another wording is never reused as if it were this one.
DESCRIBE_PROMPT_VERSION = "ri73-2"


# ---------------------------------------------------------------------------
# the conventions (BACKLOG OWN-92's recommended default)
# ---------------------------------------------------------------------------

#: How a page is labelled when its look and its section disagree. ONE
#: replaceable text: the owner's answer to OWN-92 replaces it, and a run
#: records which text it had (:func:`conventions_version`).
LABEL_CONVENTIONS = """\
HOW THESE REPORTS ARE LABELLED. Label a page by the SECTION it belongs to,
not by how the page looks on its own. In particular:

1. A photograph page, plot, chart, title page, cover sheet or certificate
   that sits inside a section takes that section's label. Photographs and
   typed field notes of a test pit inside the test pit records are
   test_pit_log. A calculation's title or header page, its plotted output,
   and the code or reference tables printed with it are calculation. A
   laboratory report's cover page, plots, certificates and test notes inside
   the laboratory results are lab_test.
2. Prose inside an appendix is 'other', not 'narrative': test procedures,
   general notes, key and legend sheets, soil classification criteria and
   an appendix's own contents page. 'narrative' is the report's own main
   text, including the tables printed inside it.
3. A tab or divider page takes the label of the section it opens, so
   there is no divider label.
4. A slip-sheet in a laboratory appendix (a page naming the test or the
   sample whose results follow) is lab_test.
5. Every page of a separate, complete report bound inside this one -- its
   own cover, letter, contents, text, figures and appendices -- is
   appended_report.
"""


#: The vocabulary's labels the conventions above never give (a divider
#: page takes its section's label), so the labeller is not offered them.
#: Replaced together with :data:`LABEL_CONVENTIONS`.
LABELS_NOT_USED: Tuple[str, ...] = ("divider",)


#: The conventions the hand labels mostly follow: a run of photographs is
#: its own section, a divider is a divider, and a sub-divider or slip-sheet
#: inside an appendix is ``other``.
LABEL_CONVENTIONS_B = """\
HOW THESE REPORTS ARE LABELLED. Label a page by the SECTION it belongs to,
not by how the page looks on its own, except where these rules say
otherwise. In particular:

1. A SINGLE photograph page, plot, chart, title page, cover sheet or
   certificate that sits inside a section takes that section's label. A
   calculation's title or header page, its plotted output, and the code or
   reference tables printed with it are calculation. A laboratory report's
   cover page, plots, certificates and test notes inside the laboratory
   results are lab_test. One photograph page among test pit logs is
   test_pit_log. But a RUN of two or more photograph pages -- a photo log,
   core box photographs, a set of site or test pit photographs -- is a
   section of its own, labelled photos, wherever it sits.
2. Prose inside an appendix is 'other', not 'narrative': test procedures,
   general notes, key and legend sheets, soil classification criteria and
   an appendix's own contents page. 'narrative' is the report's own main
   text, including the tables printed inside it.
3. A tab or divider page that opens an appendix or a main part of the
   report is divider.
4. A sub-divider or slip-sheet inside an appendix -- a page that only names
   the part, the test, the sample or the supplier whose pages follow -- is
   other.
5. Every page of a separate, complete report bound inside this one -- its
   own cover, letter, contents, text, figures and appendices -- is
   appended_report.
"""

#: Conventions B gives every label, the divider included.
LABELS_NOT_USED_B: Tuple[str, ...] = ()

#: The conventions a run can be given, by name: the text and the labels it
#: never gives.
CONVENTIONS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "A": (LABEL_CONVENTIONS, LABELS_NOT_USED),
    "B": (LABEL_CONVENTIONS_B, LABELS_NOT_USED_B),
}


def conventions_named(name: str) -> Tuple[str, Tuple[str, ...]]:
    """``(text, labels not used)`` for conventions ``A`` or ``B``."""
    key = str(name or "").strip().upper()
    if key not in CONVENTIONS:
        raise ValueError(f"conventions must be one of "
                         f"{', '.join(CONVENTIONS)}, not {name!r}")
    return CONVENTIONS[key]


def conventions_version(text: str = LABEL_CONVENTIONS) -> str:
    """A short fingerprint of the conventions text a run was given."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------

_ARM: ContextVar[Optional[str]] = ContextVar("ingest_range_labels",
                                             default=None)


def range_arm_of(value: Any) -> Optional[str]:
    """An arm from a switch value, ``None`` for off; refuses a typo."""
    text = str(value or "").strip().lower()
    if text in ("", "0", "off", "false", "no", "none"):
        return None
    if text in ("1", "on", "true", "yes"):
        return "plain"
    if text in ARMS:
        return text
    raise ValueError(f"{RANGE_LABELS_ENV}={value!r}: use off, "
                     f"{' or '.join(ARMS)}")


def range_labels_arm() -> Optional[str]:
    """The arm the switch names, or ``None`` (the default: off)."""
    forced = _ARM.get()
    if forced is not None:
        return range_arm_of(forced)
    return range_arm_of(os.environ.get(RANGE_LABELS_ENV))


@contextmanager
def use_range_labels(arm: Optional[str] = "plain") -> Iterator[None]:
    """Turn the range voter on (an arm) or off (``"off"``) inside."""
    token = _ARM.set(str(arm or "off"))
    try:
        yield
    finally:
        _ARM.reset(token)


# ---------------------------------------------------------------------------
# 1. the describer
# ---------------------------------------------------------------------------

class PageDescription(BaseModel):
    """One page, described and not labelled."""

    title: str = Field(
        description="the page's title or main heading EXACTLY as printed "
                    "(its own words, numbering and spelling; a title broken "
                    "over lines joined with spaces); a form's title block "
                    "counts; several headings: the most prominent first, "
                    "the others after ' | '; empty if the page prints none")
    kind: str = Field(
        description="what kind of page this is, in plain words, specific "
                    "enough to tell it from the pages around it")
    names: str = Field(
        description="the test, method, standard or computer program the "
                    "page names, as printed; empty if none")
    identifiers: List[str] = Field(
        description="every exploration, sample, test, figure, table, plate, "
                    "drawing or appendix identifier the page prints, as "
                    "printed, at most 30")
    continues: str = Field(
        description="what the page prints about being one of a set ('sheet 2 "
                    "of 3', 'continued', 'page 4 of 7'), as printed; empty "
                    "if nothing")
    header_footer: str = Field(
        description="any running header, footer or margin text naming the "
                    "document, part or appendix the page belongs to, as "
                    "printed; empty if none")
    issuer_date: str = Field(
        description="who issued the page and when, as printed in its title "
                    "block, header or footer (the organisation's name, a "
                    "project or job number, a date), quoted as printed; "
                    "empty if the page prints none")
    language: str = Field(
        description="the main language of the page's words, e.g. 'English'; "
                    "'none' for a page with no words")
    summary: str = Field(
        description="one sentence on what the page shows")

    @model_validator(mode="before")
    @classmethod
    def _no_nulls(cls, value: Any) -> Any:
        """A field sent as null is read as empty -- ``""``, or ``[]`` for the
        identifiers -- and each identifier as text, rather than the whole
        answer refused for it: a model leaves a field null where the page
        prints nothing, and that is what empty means here. A field that is
        MISSING is filled with its empty default, except a kind or a
        summary: an answer with no kind or no summary is not a
        description."""
        if not isinstance(value, dict):
            return value
        out = dict(value)
        for name in cls.model_fields:
            if name in out and out[name] is None:
                out[name] = [] if name == "identifiers" else ""
            elif name not in out and name not in ("kind", "summary"):
                out[name] = [] if name == "identifiers" else ""
        if isinstance(out.get("identifiers"), list):
            out["identifiers"] = [str(x) for x in out["identifiers"]
                                  if x is not None]
        return out

    @field_validator("identifiers", mode="before")
    @classmethod
    def _a_list(cls, value: Any) -> Any:
        """The identifiers as a list, however they came. A model sometimes
        sends the list as one string -- JSON, or the names separated by
        commas or lines -- and the description is no less good for it, so
        it is read rather than refused (a refused answer costs the page a
        second call, and a second refusal leaves it undescribed)."""
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                try:
                    parsed = json.loads(text)
                except ValueError:
                    parsed = None
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            parts = text.replace(";", ",").replace("\n", ",").split(",")
            return [part.strip() for part in parts if part.strip()]
        return value


DESCRIBE_SYSTEM = """\
You describe ONE page of an engineering report -- usually a geotechnical
investigation report with its appendices -- so that a reader who will never
see the page can tell from your description alone what the page is and which
part of the report it belongs to. That reader will decide where each run of
pages begins and ends, so what tells this page apart from its neighbours is
what matters most.

You are given the page as a picture, the text its PDF carries (the page's own
text layer, or an optical reader's where it has none), the page's position in
the file, and the page number the page prints on itself, if any. Read the
picture: the text can be incomplete, out of order or missing, and a page
with little or no text (a photograph, a plot, a scanned form) has to be
described from what it shows.

DESCRIBE, DO NOT CLASSIFY. Write what is there, in your own plain words for
the kind of page and in the page's own words for everything printed.

- title: the title or main heading exactly as printed. A form's title block
  counts. Empty if the page prints none.
- kind: specific plain words, such as 'boring log form', 'test pit log',
  'photograph page of a test pit', 'cone penetration test plot', 'dynamic
  probe table', 'particle size distribution sheet', 'plasticity chart',
  'laboratory summary table', 'laboratory certificate', 'calculation
  printout from a program', 'hand calculation sheet', 'site plan with
  exploration locations', 'subsurface cross-section', 'report text',
  'report text with a table', 'table of contents', 'appendix divider page',
  'report cover', 'transmittal letter', 'key to symbols', 'blank page'. Say
  so when a page belongs to a set its look alone would not show, e.g. 'title
  page of a calculation', 'notes page of a laboratory report'.
- names: the test, method, standard or program the page names.
- identifiers: every boring, test pit, probe, sample, test, figure, table,
  plate, drawing and appendix identifier the page prints.
- continues: what the page prints about being one of a set.
- header_footer: running header, footer or margin text that names the
  document, part or appendix -- this is often what says which section a
  page is in.
- issuer_date: who issued the page and when, as printed in its title block,
  header or footer -- the organisation's name, a project or job number, a
  date -- quoted as printed. A change in these from one page to the next is
  often where a different report bound into this one begins or ends.
- language: the main language of the page's words.
- summary: one sentence on what the page shows.

A page that is blank or unreadable is described as such; never invent text
that is not on the page.
"""


@dataclass
class PageDescriptions:
    """The descriptions of a document's pages, and what making them cost."""

    #: ``page -> the saved record``: the description itself under
    #: ``description``, with the page's printed number, its render and the
    #: model that made it.
    pages: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    #: Pages with no description: the call failed or gave no answer.
    unresolved: List[Dict[str, Any]] = field(default_factory=list)
    qa: List[Dict[str, Any]] = field(default_factory=list)
    #: What THIS call spent: pages read back from disk cost nothing here
    #: and their own cost stays in their files (:attr:`saved_cost`).
    cost: Dict[str, Any] = field(default_factory=dict)
    #: What every description in :attr:`pages` cost when it was made.
    saved_cost: Dict[str, Any] = field(default_factory=dict)
    reused: int = 0
    stopped: str = ""
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"n_described": len(self.pages), "reused": self.reused,
                "unresolved": [dict(u) for u in self.unresolved],
                "qa": [dict(q) for q in self.qa], "cost": dict(self.cost),
                "saved_cost": dict(self.saved_cost),
                "stopped": self.stopped, "model": self.model}


def _cost_row() -> Dict[str, Any]:
    return {"calls": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "dollars": 0.0, "seconds": 0.0}


def _charge(cost: Dict[str, Any], reply: Any) -> None:
    usage = reply.usage
    cost["calls"] += 1
    cost["input_tokens"] += int(usage.input_tokens or 0)
    cost["output_tokens"] += int(usage.output_tokens or 0)
    cost["cache_read_tokens"] += int(usage.cache_read_tokens or 0)
    cost["dollars"] = round(cost["dollars"]
                            + float(usage.dollars(reply.model or "")), 6)
    cost["seconds"] = round(cost["seconds"] + float(reply.seconds or 0.0), 2)


def _add_costs(total: Dict[str, Any], one: Mapping[str, Any]) -> None:
    for key, value in one.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = round(total.get(key, 0) + value, 6) \
                if isinstance(value, float) else total.get(key, 0) + value


def _engine_name(engine: Any) -> str:
    return str(getattr(engine, "model", "") or getattr(engine, "name", "")
               or type(engine).__name__)


def _budget_for(model: str) -> Any:
    """The largest picture ``model`` looks at without shrinking it, or None
    (then a page renders at planlens' default size)."""
    from planlens.document.budget import budget_for_model

    found = budget_for_model(model)
    return found[0] if found else None


def _why_not_taken(reply: Any) -> str:
    """Why a reply gave no description, in a line: what the engine said
    about the answer it did not take, or how the reply ended."""
    notes = [str(n) for n in (getattr(reply, "notes", None) or [])]
    if notes:
        return "; ".join(notes)[:300]
    if getattr(reply, "cut_off", False):
        return f"cut at the output cap ({reply.stop_reason})"
    return (f"no answer through the answer tool (stop "
            f"{getattr(reply, 'stop_reason', '') or 'unknown'})")


def _described(blob: Mapping[str, Any]) -> bool:
    """Whether a saved record describes its page: one with neither a kind
    nor a summary does not (even a blank page is described as one), and is
    asked for again rather than reused."""
    d = blob.get("description") or {}
    return bool(str(d.get("kind") or "").strip()
                or str(d.get("summary") or "").strip())


def _page_file(folder: str, page: int) -> str:
    return os.path.join(folder, f"p{int(page):04d}.json")


def load_descriptions(folder: str) -> Dict[int, Dict[str, Any]]:
    """Every saved description in ``folder``, by page."""
    out: Dict[int, Dict[str, Any]] = {}
    if not folder or not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if not (name.startswith("p") and name.endswith(".json")):
            continue
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as fh:
                blob = json.load(fh)
            out[int(blob["page"])] = blob
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


def _page_message(doc: Any, page: int, max_text_chars: int,
                  budget: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The words and the picture one description call carries, and the
    facts about the page the saved record keeps."""
    summary = doc.summary(page)
    content = doc.page(page, tables=False)
    text = content.text() or ""
    sources = sorted(set(getattr(content, "text_sources", []) or []))
    optical = any(s in ("ocr", "azure_di") for s in sources)
    printed = summary.printed_page
    printed_of = getattr(summary, "printed_of", None)
    cut = len(text) > max_text_chars
    shown = text[:max_text_chars] if cut else text
    lines = [f"This is page {page + 1} of {doc.n_pages} in the PDF file "
             f"(page index {page}).",
             (f"The page prints its own page number as {printed}"
              + (f" of {printed_of}" if printed_of else "") + "."
              if printed is not None else
              "No printed page number was found on the page.")]
    if not shown.strip():
        lines.append("The page carries no text: describe it from the "
                     "picture.")
    else:
        where = ("as an optical reader read it from the picture" if optical
                 else "as the PDF's own text layer carries it")
        if not getattr(summary, "text_reliable", True):
            where += (" -- this text layer is UNRELIABLE (a font with no "
                      "character map): trust the picture over it")
        lines += [f"The page's text, {where}, between the markers:",
                  "<<<PAGE TEXT", shown, "PAGE TEXT>>>"]
        if cut:
            lines.append(f"[the text is cut at {max_text_chars} characters; "
                         f"the picture shows the whole page]")
    lines.append("Describe this page.")
    png, info = doc.render(page, budget=budget, fmt="auto")
    facts = {"printed": printed, "printed_of": printed_of,
             "text_chars": len(text), "text_sources": sources,
             "render": {"dpi": info.get("dpi"),
                        "width_px": info.get("width_px"),
                        "height_px": info.get("height_px"),
                        "format": info.get("format"),
                        "budget": getattr(budget, "name", None)}}
    return [text_block("\n".join(lines)), image_block(png)], facts


def describe_pages(doc: Any, engine: Engine, *,
                   pages: Optional[Sequence[int]] = None,
                   folder: Optional[str] = None,
                   model: Optional[str] = None,
                   budget: Any = None,
                   max_text_chars: int = MAX_PAGE_TEXT_CHARS,
                   workers: int = 1,
                   redo: bool = False,
                   progress: Optional[Callable[[int, int], None]] = None,
                   deadline: Optional[float] = None,
                   max_calls: Optional[int] = None
                   ) -> PageDescriptions:
    """Describe every page (or ``pages``), one vision call per page.

    ``engine`` is the describer -- a parameter, like its model. ``model``
    names the model for the picture size (:mod:`planlens.document.budget`)
    and for the saved record; it defaults to the engine's own name. ``budget``
    overrides the picture size (an ``ImageBudget`` or a planlens budget
    name).

    ``folder`` is where each description is saved AS IT IS MADE, one
    ``pNNNN.json`` per page, and where a later call finds it again: a saved
    page is reused unless ``redo``. A folder holding another model's
    descriptions, or another wording's, is refused rather than mixed.

    ``workers`` above 1 sends that many pages at once; the descriptions are
    the same, only sooner. A call that fails is retried once if it gave no
    answer; one that raises leaves the page unresolved (not saved, so the
    next call asks again), and a stop error (:data:`STOP_ERRORS`) ends the
    pass with what is done kept. A page that needed a second call keeps
    why in its saved record (``retried``: what the engine said about the
    answer it did not take, or how the reply ended).

    ``deadline`` (a ``time.time()`` value) stops sending pages once it has
    passed; the pages in flight finish and are saved, and the pass reports
    ``stopped`` so a caller can come back for the rest. ``max_calls`` is a
    ceiling on this pass's model calls (the graph's
    ``Budgets.range_describe_calls``): reached, the pass stops the same way.

    ``progress(described, wanted)`` is called after each page is saved,
    outside the pass's lock, so a slow callback (a mirror sync) holds up
    only its own thread. Pages are drawn one at a time across threads.
    """
    name = str(model or _engine_name(engine))
    if budget is None:
        budget = _budget_for(name)
    wanted = list(range(doc.n_pages)) if pages is None \
        else sorted({int(p) for p in pages})
    out = PageDescriptions(cost=_cost_row(), saved_cost=_cost_row(),
                           model=name)
    saved = load_descriptions(folder) if folder and not redo else {}
    for page, blob in saved.items():
        if blob.get("model") != name or \
                blob.get("prompt") != DESCRIBE_PROMPT_VERSION:
            raise ValueError(
                f"{folder} holds descriptions made by "
                f"{blob.get('model')!r} with prompt {blob.get('prompt')!r}; "
                f"this pass is {name!r} with {DESCRIBE_PROMPT_VERSION!r}. "
                f"Use another folder rather than mix them.")
    todo = []
    for page in wanted:
        if page in saved and _described(saved[page]):
            out.pages[page] = saved[page]
            _add_costs(out.saved_cost, saved[page].get("cost") or {})
            out.reused += 1
        else:
            todo.append(page)
    if folder:
        os.makedirs(folder, exist_ok=True)

    lock = threading.Lock()
    drawing = threading.Lock()
    stop = threading.Event()
    calls = [0]

    def take_call() -> bool:
        with lock:
            if max_calls is not None and calls[0] >= int(max_calls):
                out.stopped = out.stopped or (
                    f"the ceiling of {int(max_calls)} describer call(s) was "
                    f"reached; pages left for later")
                stop.set()
                return False
            calls[0] += 1
            return True

    def one(page: int) -> None:
        if stop.is_set():
            return
        if deadline is not None and time.time() > deadline:
            with lock:
                out.stopped = out.stopped or "deadline: pages left for later"
            stop.set()
            return
        own = _cost_row()
        retried: List[str] = []
        try:
            with drawing:
                content, facts = _page_message(doc, page, max_text_chars,
                                               budget)
            answer = None
            for attempt in range(2):
                if not take_call():
                    if attempt == 0:
                        return                    # left for later, unasked
                    retried.append("the call ceiling was reached before a "
                                   "second call")
                    break
                reply = engine.complete([user(*content)],
                                        system=DESCRIBE_SYSTEM,
                                        output_format=PageDescription)
                _charge(own, reply)
                answer = reply.parsed
                if answer is not None:
                    break
                retried.append(_why_not_taken(reply))
        except Exception as exc:                  # one page, not the pass
            with lock:
                own["failed"] += 1
                _add_costs(out.cost, own)
                out.unresolved.append({
                    "page": page,
                    "why": f"{type(exc).__name__}: {str(exc)[:200]}"})
                if type(exc).__name__ in STOP_ERRORS:
                    out.stopped = f"{type(exc).__name__}: {str(exc)[:200]}"
                    stop.set()
            return
        with lock:
            _add_costs(out.cost, own)
            if answer is None:
                out.unresolved.append({
                    "page": page, "why": "no description taken: "
                                         + " | ".join(retried)})
                return
        blob = {"page": page, "pdf_page": page + 1, **facts,
                "description": answer.model_dump(), "model": name,
                "served_by": getattr(engine, "served_by", None) or name,
                "prompt": DESCRIBE_PROMPT_VERSION, "cost": own}
        if retried:
            blob["retried"] = retried
        if folder:
            tmp = _page_file(folder, page) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(blob, fh, indent=1, ensure_ascii=False)
            os.replace(tmp, _page_file(folder, page))
        with lock:
            out.pages[page] = blob
            _add_costs(out.saved_cost, own)
            done = len(out.pages)
        if progress is not None:
            progress(done, len(wanted))

    if int(workers) <= 1:
        for page in todo:
            if stop.is_set():
                break
            one(page)
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            futures = [pool.submit(one, page) for page in todo]
            for future in as_completed(futures):
                future.result()
    out.unresolved.sort(key=lambda row: row["page"])
    return out


# ---------------------------------------------------------------------------
# 2. the range labeller
# ---------------------------------------------------------------------------

class RangeSection(BaseModel):
    """A run of consecutive pages with one label."""

    first_page: int = Field(description="the section's first page index")
    last_page: int = Field(description="the section's last page index")
    label: str = Field(description="one label from the vocabulary")
    reason: str = Field(description="what decided it, in 20 words or fewer")


class RangeException(BaseModel):
    """One page inside a section whose label differs from the section's."""

    page: int = Field(description="the page index")
    label: str = Field(description="one label from the vocabulary")
    reason: str = Field(description="why this page differs from its "
                                    "section, in 20 words or fewer")


class RangeAnswer(BaseModel):
    """Every page of the window, in sections, with any one-page exceptions."""

    sections: List[RangeSection] = Field(
        description="consecutive sections in page order, covering every "
                    "page you were given exactly once")
    exceptions: List[RangeException] = Field(
        description="single pages whose label differs from the section "
                    "around them; empty when there are none")


_LABELLER_HEAD = """\
You label the pages of one engineering report -- usually a geotechnical
investigation report with its appendices -- by SECTIONS. You are given a
description of every page, in page order, written by someone who looked at
each page, and the outline the report prints about itself (its contents
lists, dividers and headings). Read the whole sequence first: where the main
text ends, where each appendix begins and what it holds, where a separate
report bound into this one begins and ends.

THE LABEL VOCABULARY. Every page gets exactly one of these.
"""

_LABELLER_RULES = """
HOW TO ANSWER.
- Answer in sections: runs of consecutive pages that share one label, each
  with its first and last page index, its label and a short reason. Cover
  every page you were given exactly once, in order, with no gap and no
  overlap. A section may be one page long.
- Page indexes are the numbers in square brackets at the start of each
  description line, counting from 0. They are NOT the numbers the pages
  print on themselves, which often start again in each appendix.
- Use an exception only for a single page whose label differs from the
  section around it (a figure page inside the report text, one log sheet
  between laboratory sheets). A run of two or more such pages is a section
  of its own.
- A page that says little about itself usually belongs with the pages
  around it: decide it from the section it sits in.
- 'issued' is who issued a page and when, as printed on it. Where it
  changes from the report's own, look at what the run of pages is. A
  laboratory's, driller's or other supplier's results inside an appendix
  take that appendix's label as usual. A complete report of its own -- its
  own cover, letter or contents and its own text -- bound into this one is
  appended_report, every page of it to its last; the issuer and date
  changing back is often where it ends.
- The reason names what decided the label (the divider that opens the
  section, the kind of page, the identifiers it shares with its neighbours),
  in 20 words or fewer.
"""

_HINT_NOTE = """
THE RULES' LABEL. Each line ends with 'rules:' and the label a set of
text-matching rules gave that page from its own text. Treat it as a hint,
not an answer: the rules are usually right on a page that names itself, and
least sure at the edges of a section, on pages with little text, and on
prose inside an appendix. Where the rules' label and the conventions above
disagree, the conventions decide.
"""


def labeller_vocabulary(not_used: Sequence[str] = LABELS_NOT_USED
                        ) -> Dict[str, str]:
    """The labels the labeller is offered: the vocabulary less the ones the
    conventions never give."""
    return {name: text for name, text in LABEL_DEFINITIONS.items()
            if name not in set(not_used)}


def labeller_system(hinted: bool = False,
                    conventions: str = LABEL_CONVENTIONS,
                    not_used: Sequence[str] = LABELS_NOT_USED) -> str:
    """The labeller's instructions: vocabulary, conventions, the answer."""
    vocab = "\n".join(f"  {name}: {text}" for name, text
                      in labeller_vocabulary(not_used).items())
    return (_LABELLER_HEAD + vocab + "\n\n" + conventions.strip() + "\n"
            + _LABELLER_RULES + (_HINT_NOTE if hinted else ""))


def labeller_fingerprint(hinted: bool = False,
                         conventions: str = LABEL_CONVENTIONS,
                         not_used: Sequence[str] = LABELS_NOT_USED) -> str:
    """A short fingerprint of the labeller's whole instructions for an arm:
    a saved answer from other instructions is not taken as this one's."""
    text = labeller_system(hinted, conventions, not_used)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _clean(text: Any, limit: int = 240) -> str:
    words = " ".join(str(text or "").split())
    return words if len(words) <= limit else words[:limit - 3] + "..."


def description_line(page: int, record: Optional[Mapping[str, Any]],
                     hint: Optional[str] = None) -> str:
    """One page's description as the labeller reads it: one line."""
    parts = [f"[{int(page)}]"]
    if not record:
        parts.append("(no description: this page could not be described)")
    else:
        d = record.get("description") or {}
        printed = record.get("printed")
        if printed is not None:
            of = record.get("printed_of")
            parts.append(f"printed {printed}" + (f"/{of}" if of else ""))
        for key, title in (("kind", "kind"), ("title", "title"),
                           ("names", "names"), ("continues", "continues"),
                           ("header_footer", "header"),
                           ("issuer_date", "issued"),
                           ("summary", "shows")):
            value = _clean(d.get(key))
            if value:
                parts.append(f"{title}: {value}")
        ids = [str(x).strip() for x in (d.get("identifiers") or [])
               if str(x).strip()]
        if ids:
            more = f" (+{len(ids) - 12} more)" if len(ids) > 12 else ""
            parts.append("ids: " + ", ".join(ids[:12]) + more)
        language = _clean(d.get("language"), 30)
        if language and language.lower() not in ("english", "en"):
            parts.append(f"language: {language}")
    if hint:
        parts.append(f"rules: {hint}")
    return " | ".join(parts)


def balanced_windows(pages: Sequence[int], window: int,
                     overlap: int) -> List[List[int]]:
    """``pages`` in as few windows of at most ``window`` as cover them,
    each sharing ``overlap`` pages with the next, all of near one size."""
    wanted = list(pages)
    n = len(wanted)
    size = max(1, int(window))
    if n <= size:
        return [wanted] if wanted else []
    overlap = max(0, min(int(overlap), size - 1))
    k = math.ceil((n - overlap) / (size - overlap))
    length = math.ceil((n + (k - 1) * overlap) / k)
    step = length - overlap
    out = []
    for i in range(k):
        start = min(i * step, n - length)
        out.append(wanted[start:start + length])
    return out


def check_sections(answer: Any, window: Sequence[int],
                   not_used: Sequence[str] = ()
                   ) -> Tuple[Dict[int, List[Tuple[str, str]]],
                              Dict[int, Tuple[str, str]], List[str],
                              List[Dict[str, Any]]]:
    """What an answer says about each page of ``window``, and its faults.

    Returns ``(claims, exceptions, problems, qa)``: ``claims`` maps each page
    to every ``(label, reason)`` a section gave it; ``exceptions`` each
    one-page exception; ``problems`` the coverage faults in words, for a
    re-ask; ``qa`` what Python changed (a label outside the vocabulary, a
    page outside the window). A label the active conventions do not offer
    (``not_used``) is read like one outside the vocabulary: noted, read as
    'other', and counted as a fault for the re-ask.

    A fault is a page the answer leaves unsettled (:func:`unsettled`): one
    no section covers, or two sections give different labels, with no
    exception to settle it. Two sections that give a page the SAME label
    have not left it in doubt, and an exception settles its page whatever
    the sections said, so neither is a fault.
    """
    inside = set(int(p) for p in window)
    low, high = (min(inside), max(inside)) if inside else (0, -1)
    claims: Dict[int, List[Tuple[str, str]]] = {p: [] for p in inside}
    exceptions: Dict[int, Tuple[str, str]] = {}
    qa: List[Dict[str, Any]] = []
    faults: List[str] = []

    def label_of(raw: Any, where: str) -> str:
        label = str(raw or "").strip()
        if label not in LABEL_DEFINITIONS or label in set(not_used):
            qa.append({"where": where, "note": f"label {label!r} is not in "
                                               f"the vocabulary; read as "
                                               f"'other'"})
            faults.append(f"label {label!r} given to {where} is not one of "
                          f"the labels offered")
            return "other"
        return label

    for section in getattr(answer, "sections", None) or []:
        first, last = int(section.first_page), int(section.last_page)
        if last < first:
            first, last = last, first
        label = label_of(section.label, f"section {first}-{last}")
        # Only the window's pages are walked: a section drawn to page
        # 100000 costs no more than one drawn to the window's end.
        start, stop = max(first, low), min(last, high)
        outside = (last - first + 1) - max(0, stop - start + 1)
        if outside:
            qa.append({"where": f"section {first}-{last}",
                       "note": f"{outside} page(s) outside the pages "
                               f"asked about were dropped"})
        for page in range(start, stop + 1):
            if page in inside:
                claims[page].append((label, str(section.reason or "")))
    for row in getattr(answer, "exceptions", None) or []:
        page = int(row.page)
        if page not in inside:
            qa.append({"where": f"exception {page}",
                       "note": "an exception outside the pages asked about "
                               "was dropped"})
            continue
        exceptions[page] = (label_of(row.label, f"exception {page}"),
                            str(row.reason or ""))
    gaps = sorted(p for p, said in claims.items() if not said
                  and p not in exceptions)
    split = sorted(p for p, said in claims.items() if p not in exceptions
                   and len({label for label, _r in said}) > 1)
    problems = list(faults)
    if gaps:
        problems.append(f"no section covers page(s) {_spans(gaps)}")
    if split:
        problems.append(f"sections with different labels cover page(s) "
                        f"{_spans(split)}")
    return claims, exceptions, problems, qa


def unsettled(claims: Mapping[int, Sequence[Tuple[str, str]]],
              exceptions: Mapping[int, Any]) -> List[int]:
    """The pages an answer leaves without one label: no section and no
    exception, or sections with different labels and no exception."""
    return sorted(page for page, said in claims.items()
                  if page not in exceptions
                  and len({label for label, _r in said}) != 1)


def _spans(pages: Sequence[int]) -> str:
    """``[1, 2, 3, 7]`` as ``1-3, 7``."""
    out: List[str] = []
    run: List[int] = []
    for page in list(pages) + [None]:
        if run and (page is None or page != run[-1] + 1):
            out.append(str(run[0]) if len(run) == 1
                       else f"{run[0]}-{run[-1]}")
            run = []
        if page is not None:
            run.append(page)
    return ", ".join(out)


@dataclass
class RangeLabels:
    """The labels a range pass settled, its sections, and what it cost."""

    labels: Dict[int, str] = field(default_factory=dict)
    #: The final sections, rebuilt from :attr:`labels` (a run of one label),
    #: each with the reason its first page was given.
    sections: List[Dict[str, Any]] = field(default_factory=list)
    exceptions: List[Dict[str, Any]] = field(default_factory=list)
    unresolved: List[int] = field(default_factory=list)
    qa: List[Dict[str, Any]] = field(default_factory=list)
    windows: List[Dict[str, Any]] = field(default_factory=list)
    cost: Dict[str, Any] = field(default_factory=dict)
    arm: str = "plain"
    model: str = ""
    conventions: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"arm": self.arm, "model": self.model,
                "conventions": self.conventions,
                "labels": {str(p): l for p, l in sorted(self.labels.items())},
                "sections": [dict(s) for s in self.sections],
                "exceptions": [dict(e) for e in self.exceptions],
                "unresolved": list(self.unresolved),
                "qa": [dict(q) for q in self.qa],
                "windows": [dict(w) for w in self.windows],
                "cost": dict(self.cost)}


def _window_message(window: Sequence[int], n_pages: int, outline: str,
                    descriptions: Mapping[int, Mapping[str, Any]],
                    hints: Optional[Mapping[int, str]]) -> str:
    first, last = window[0], window[-1]
    head = [f"This report has {n_pages} pages, indexes 0 to {n_pages - 1}."]
    if first > 0 or last < n_pages - 1:
        head.append(
            f"You are labelling pages {first} to {last} only; the rest of "
            f"the report is labelled separately. "
            + ("The window starts inside the report: the outline below says "
               "what comes before it. " if first > 0 else "")
            + ("The window ends before the report does." if last < n_pages - 1
               else ""))
    else:
        head.append("Label every page.")
    lines = head + ["", "THE OUTLINE THE REPORT PRINTS ABOUT ITSELF",
                    outline.strip() or "(none found)", "",
                    "THE PAGES, ONE DESCRIPTION A LINE"]
    for page in window:
        lines.append(description_line(
            page, descriptions.get(page),
            (hints or {}).get(page) if hints is not None else None))
    lines += ["", f"Answer in sections covering pages {first} to {last}, "
                  f"each exactly once."]
    return "\n".join(lines)


def _ask_window(engine: Engine, system: str, message: str,
                window: Sequence[int], cost: Dict[str, Any],
                repair: bool,
                not_used: Sequence[str] = ()) -> Dict[str, Any]:
    """One window: the call, the check, at most one re-ask. Returns what the
    window settled, or ``{"cut": True}`` for an answer that stopped at the
    output cap.

    The re-ask is a fresh call -- the same message, the first answer's
    sections in brief and what was wrong with them -- rather than the first
    turn replayed, because that turn's answer is a tool call that would need
    a tool result to follow it.
    """
    reply = engine.complete([user(text_block(message))], system=system,
                            output_format=RangeAnswer)
    _charge(cost, reply)
    calls = 1
    if reply.cut_off:
        return {"cut": True, "calls": calls}
    answer = reply.parsed
    problems: List[str] = ["no answer"]
    claims: Dict[int, List[Tuple[str, str]]] = {p: [] for p in window}
    exceptions: Dict[int, Tuple[str, str]] = {}
    qa: List[Dict[str, Any]] = []
    if answer is not None:
        claims, exceptions, problems, qa = check_sections(answer, window, not_used)
    repaired = False
    if problems and repair:
        if answer is None:
            note = ("A first answer to this came back with no sections "
                    "through the answer tool.")
        else:
            drawn = "; ".join(f"{s.first_page}-{s.last_page} {s.label}"
                              for s in answer.sections)
            note = (f"A first answer to this drew these sections: {drawn}. "
                    f"They leave pages without one label: "
                    + "; ".join(problems) + ".")
        again = engine.complete(
            [user(text_block(
                message + "\n\n" + note + f" Answer again, whole: every "
                f"page from {window[0]} to {window[-1]} in exactly one "
                f"section."))],
            system=system, output_format=RangeAnswer)
        _charge(cost, again)
        calls += 1
        if again.parsed is not None and not again.cut_off:
            c2, e2, p2, q2 = check_sections(again.parsed, window, not_used)
            # Taken only if it leaves fewer pages unsettled: counting the
            # KINDS of fault would let one gap be traded for 81 pages.
            if answer is None or len(unsettled(c2, e2)) < len(
                    unsettled(claims, exceptions)):
                claims, exceptions, problems, qa = c2, e2, p2, q2
                answer = again.parsed
                repaired = True
        elif answer is None and again.cut_off:
            return {"cut": True, "calls": calls}
    return {"claims": claims, "exceptions": exceptions, "problems": problems,
            "qa": qa, "calls": calls, "repaired": repaired,
            "answer": answer.model_dump() if answer is not None else None}


def label_by_ranges(descriptions: Mapping[int, Mapping[str, Any]],
                    n_pages: int, engine: Engine, *,
                    outline: str = "",
                    hints: Optional[Mapping[int, str]] = None,
                    window: int = DEFAULT_WINDOW,
                    overlap: int = DEFAULT_OVERLAP,
                    conventions: str = LABEL_CONVENTIONS,
                    not_used: Sequence[str] = LABELS_NOT_USED,
                    repair: bool = True,
                    model: Optional[str] = None) -> RangeLabels:
    """Label every page by sections from the descriptions, in one call per
    window.

    ``descriptions`` is ``page -> the saved record`` (what
    :func:`describe_pages` returns in ``pages``); a page with none is shown
    as such and still labelled. ``outline`` is the text the document prints
    about itself (:func:`report_ingest.triage.outline_text`). ``hints``, when
    given, is the ``hinted`` arm: the rules' label on every line. ``engine``
    is the labeller -- a parameter, like its model.

    A report longer than ``window`` pages is cut into overlapping windows
    (:func:`balanced_windows`); a page in two windows takes the label of the
    window it is further inside (the cut is the overlap's middle), and the
    two windows' disagreements there are counted in a QA note. A window
    whose answer stops at the output cap is halved and asked again, down to
    :data:`MIN_WINDOW` pages.

    A page no section covers, or that two sections give different labels
    (after one re-ask naming the pages), is UNRESOLVED; a one-page exception
    settles its page whatever the sections said.
    """
    hinted = hints is not None
    system = labeller_system(hinted=hinted, conventions=conventions,
                             not_used=not_used)
    out = RangeLabels(arm="hinted" if hinted else "plain",
                      model=str(model or _engine_name(engine)),
                      conventions=conventions_version(conventions),
                      cost=_cost_row())
    pages = list(range(int(n_pages)))
    queue = balanced_windows(pages, window, overlap)
    done: List[Tuple[List[int], Dict[str, Any]]] = []
    while queue:
        shown = queue.pop(0)
        message = _window_message(shown, n_pages, outline, descriptions,
                                  hints)
        got = _ask_window(engine, system, message, shown, out.cost, repair,
                          not_used)
        if got.get("cut"):
            if len(shown) // 2 < MIN_WINDOW:
                out.qa.append({"where": f"window {shown[0]}-{shown[-1]}",
                               "note": "the answer stopped at the output cap "
                                       "and the window is too short to "
                                       "halve; its pages are unresolved"})
                done.append((shown, {"claims": {p: [] for p in shown},
                                     "exceptions": {}, "problems": ["cut"],
                                     "qa": [], "calls": got["calls"],
                                     "repaired": False, "answer": None}))
                continue
            half = balanced_windows(shown, math.ceil(len(shown) / 2)
                                    + overlap // 2, overlap)
            out.qa.append({"where": f"window {shown[0]}-{shown[-1]}",
                           "note": f"the answer stopped at the output cap; "
                                   f"asked again in {len(half)} windows"})
            queue = half + queue
            continue
        done.append((shown, got))
    done.sort(key=lambda row: row[0][0])

    # Which window owns each page: the one it is furthest inside.
    owner: Dict[int, int] = {}
    for index, (shown, _got) in enumerate(done):
        for page in shown:
            depth = min(page - shown[0], shown[-1] - page)
            best = owner.get(page)
            if best is None:
                owner[page] = index
                continue
            other = done[best][0]
            if depth > min(page - other[0], other[-1] - page):
                owner[page] = index
    reasons: Dict[int, str] = {}
    disagreements = 0
    for index, (shown, got) in enumerate(done):
        out.windows.append({
            "first": shown[0], "last": shown[-1], "calls": got["calls"],
            "repaired": got["repaired"], "problems": list(got["problems"]),
            "answer": got["answer"]})
        for row in got["qa"]:
            out.qa.append(dict(row))
        for page in shown:
            claims = got["claims"].get(page) or []
            labels = {label for label, _r in claims}
            exception = got["exceptions"].get(page)
            if exception is not None:
                label, reason = exception
            elif len(labels) == 1:
                label, reason = claims[0]
            else:
                label, reason = None, ""
            if owner[page] != index:
                theirs = done[owner[page]][1]
                mine = label
                other_claims = theirs["claims"].get(page) or []
                other_exc = theirs["exceptions"].get(page)
                other = (other_exc[0] if other_exc else
                         other_claims[0][0] if len({l for l, _r in
                                                    other_claims}) == 1
                         else None)
                if mine and other and mine != other:
                    disagreements += 1
                continue
            if label is None:
                out.unresolved.append(page)
                continue
            out.labels[page] = label
            reasons[page] = reason
            if exception is not None:
                out.exceptions.append({"page": page, "label": label,
                                       "reason": reason})
    if disagreements:
        out.qa.append({"where": "windows",
                       "note": f"{disagreements} page(s) in the overlap of "
                               f"two windows got different labels from "
                               f"each; each took its nearer window's"})
    if out.unresolved:
        out.qa.append({"where": "coverage",
                       "note": f"{len(out.unresolved)} page(s) no section "
                               f"settled: {_spans(out.unresolved)}"})
    run: Optional[Dict[str, Any]] = None
    for page in sorted(out.labels):
        label = out.labels[page]
        if run and run["label"] == label and run["last_page"] == page - 1:
            run["last_page"] = page
            continue
        run = {"first_page": page, "last_page": page, "label": label,
               "reason": reasons.get(page, "")}
        out.sections.append(run)
    return out
