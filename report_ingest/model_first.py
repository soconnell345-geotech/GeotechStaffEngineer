"""Model-first report ingest (BACKLOG RI-87; the plan's section 8 in
``module_work/REPORT_INGEST_PLAN.md``).

Under ``GEOTECH_INGEST_MODEL_FIRST=on`` the model reads each item's pages in
one call and its answer becomes the record. This module grows piece by piece
(section 8.9). Today it holds the probe of section 8.7, which says whether a
route carries a PDF to the model whole, and piece 4: the answer schemas with
what section 8.5 adds, each stage's instruction, and where the new fields go
in the record. Nothing in ingest calls piece 4 yet (pieces 7 and 8 do, and
only under ``on``); today's readers and their schemas are untouched.

THE ANSWER SCHEMAS (:func:`answer_schema`). Subclasses of today's reading
classes, under the SAME names and docstrings, because the model reads both
in the answer tool ("Your answer, as a LogReading"): what it sees is
today's schema with the new fields added at the end of their class and
nothing else changed. A log gains the project name and number, the client
and every other header field it prints (:class:`LogReading`); a log's
sample gains recovery printed as a length, with its unit
(:class:`ReadSample`); a lab summary table's row gains its bottom elevation
(:class:`ReadRow`). Calc and narrative take today's schemas.

THE INSTRUCTIONS (:data:`INSTRUCTIONS`, :func:`instruction`). RI-86's plain
instructions, which matched the pipeline with one short paragraph a stage
(:data:`PLAIN_INSTRUCTIONS`, kept as measured for the plain baseline), with
the logs' paragraph naming what the log schema adds, the lab's asking for
Atterberg limits on a gradation sheet as an entry of their own, and the
narrative's carrying today's house conventions (RI-175). Versioned by
:data:`READ_PROMPT_VERSION`.

WHERE THE NEW FIELDS GO (:func:`apply_log_additions`,
:func:`apply_lab_additions`), after today's builders have made the record,
into slots the record already has, so no record field and no DIGGS element
is added: the project fields and the other header fields into
``Investigation.fields``, a recovery length into ``Sample.recovery``, a
row's bottom elevation into ``SummaryRow.other``.

THE PROBE (:func:`probe_pdf_route`, wave MF-0). Two calls over a two-page
PDF made in memory. Page 2 holds two made-up code words: the TEXT word in
its text layer alone (invisible text, as a scanned page's OCR layer is),
the PICTURE word in a picture alone, with no text under it. Page 1 and the
questions hold neither. A route that hands the model the whole PDF, text
layer and page pictures both, brings both words back; one that sends the
text layer alone brings back the text word only; one that drops the PDF,
or keeps its first page only, brings back neither.

1. The PDF call sends the PDF (:func:`~report_ingest.engine.pdf_block`)
   and asks for every code word on page 2. ``use_pdf`` is true only when
   BOTH words come back.
2. The pictures call is the control: the two pages as pictures, asking for
   the picture word. A model that reads it there but not from the PDF was
   let down by the route, not by its eyes.

Set ``GEOTECH_FOUNDRY_CLAUDE_PDF=1`` only if the PDF call brought both
words back on the Foundry Claude proxy; the switch does nothing on any
other route. The PDF call goes out even where the engine would not send a
PDF (:attr:`~report_ingest.engine.ChatModelEngine.accepts_pdf` off),
because finding out is the point. A call that raises ``ValueError`` was
refused before anything was sent (the ``palantir_models`` Claude class
refuses a PDF); any other error is a failed call, and the probe is run
again. Every probe draws fresh code words, so a cached reply to an earlier
probe cannot pass for this one.
"""

from __future__ import annotations

import random
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from report_ingest import lab_reader as _lab
from report_ingest import log_reader as _log
from report_ingest.engine import (FOUNDRY_CLAUDE_PDF_ENV,
                                  FOUNDRY_PROXY_LLM_TYPE, ChatModelEngine,
                                  engine_for, image_block, pdf_block,
                                  text_block, user)
from report_ingest.model import (Investigation, LabTest, Quantity,
                                 SummaryTableResult)
from report_ingest.model_first_pages import page_map_sentence

#: Made-up words a probe's code words are drawn from, with four digits after.
CODE_WORDS = ("AMBERWICK", "COPPERWREN", "MARIGOLD", "QUILLBANK",
              "TALLOWMERE", "WILLOWCREST")

#: What the PDF call asks. Neither code word is in it.
PDF_ASK = ("Page 2 of this document holds two code words. Reply with every "
           "code word you find on page 2, exactly as printed, separated by a "
           "space, and nothing else. If you find none, reply NONE.")

#: What the pictures call asks: the picture word, the only one it can see.
PICTURES_ASK = ("Page 2 shows a code word in a picture. Reply with that code "
                "word exactly as printed and nothing else. If you cannot see "
                "page 2, reply NONE.")

#: The resolution the pictures call renders the probe's pages at.
PICTURE_DPI = 100


def fresh_code_words() -> Tuple[str, str]:
    """Two different made-up code words no model has seen, each a word and
    four digits: ``(text word, picture word)``."""
    pick = random.SystemRandom()
    first, second = pick.sample(CODE_WORDS, 2)
    return (f"{first}-{pick.randint(1000, 9999)}",
            f"{second}-{pick.randint(1000, 9999)}")


def _word_picture(line: str) -> bytes:
    """``line`` drawn as a PNG, with no text layer anywhere."""
    import pymupdf

    scratch = pymupdf.open()
    try:
        page = scratch.new_page(width=460, height=48)
        page.insert_text((8, 32), line, fontsize=20)
        return page.get_pixmap(dpi=150).tobytes("png")
    finally:
        scratch.close()


def probe_document(text_word: str, picture_word: str) -> bytes:
    """The probe's two-page PDF, made in memory. Page 2 prints
    ``text_word`` in its text layer alone (invisible text) and shows
    ``picture_word`` in a picture alone; page 1 holds neither."""
    import pymupdf

    doc = pymupdf.open()
    try:
        first = doc.new_page()
        first.insert_text((72, 120), "Page 1 of 2", fontsize=20)
        first.insert_text((72, 156), "The code words are on the next page.",
                          fontsize=20)
        second = doc.new_page()
        second.insert_text((72, 120), "Page 2 of 2", fontsize=20)
        second.insert_text((72, 156), f"The text code word is {text_word}.",
                           fontsize=20, render_mode=3)
        second.insert_image(
            pymupdf.Rect(72, 200, 532, 248),
            stream=_word_picture(f"The picture code word is {picture_word}."))
        return doc.tobytes(garbage=3, deflate=True)
    finally:
        doc.close()


def _pictures(pdf: bytes) -> List[bytes]:
    """Each page of ``pdf`` as a PNG."""
    import pymupdf

    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return [page.get_pixmap(dpi=PICTURE_DPI).tobytes("png")
                for page in doc]


def _squashed(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def _ask(engine: Any, blocks: Sequence[Any],
         words: Tuple[str, str]) -> Dict[str, Any]:
    """One probe call: which code words its reply carries, what came back,
    or what the call raised (``outcome`` reply, refused or failed)."""
    started = time.time()
    try:
        reply = engine.complete([user(*blocks)])
    except Exception as exc:                            # noqa: BLE001
        return {"outcome": "refused" if isinstance(exc, ValueError)
                else "failed",
                "text_read": False, "picture_read": False, "reply": "",
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "seconds": round(time.time() - started, 1)}
    said = _squashed(reply.text)
    return {"outcome": "reply",
            "text_read": _squashed(words[0]) in said,
            "picture_read": _squashed(words[1]) in said,
            "reply": reply.text[:200], "error": None,
            "stop_reason": reply.stop_reason, "served_by": reply.model,
            "input_tokens": reply.usage.input_tokens,
            "output_tokens": reply.usage.output_tokens,
            "seconds": round(time.time() - started, 1)}


def _on_the_proxy(engine: Any) -> bool:
    """Whether the engine's model is the Foundry Claude proxy's class."""
    try:
        return getattr(getattr(engine, "chat_model", None), "_llm_type",
                       None) == FOUNDRY_PROXY_LLM_TYPE
    except Exception:                                   # noqa: BLE001
        return False


def _verdict(name: str, proxy: bool, accepts: Optional[bool],
             by_pdf: Dict[str, Any], by_pictures: Dict[str, Any]) -> str:
    if by_pdf["outcome"] == "failed":
        return (f"{name}: the PDF call failed ({by_pdf['error'][:200]}); "
                f"run the probe again.")
    if by_pdf["text_read"] and by_pdf["picture_read"]:
        whole = (f"{name}: the PDF reached the model whole: its text word "
                 f"and its picture word both came back.")
        if proxy:
            return f"{whole} Set {FOUNDRY_CLAUDE_PDF_ENV}=1."
        return (f"{whole} {FOUNDRY_CLAUDE_PDF_ENV} applies only to the "
                f"Foundry Claude proxy and does nothing for this route; "
                + ("the engine already sends a PDF here." if accepts else
                   "a PDF goes out here only from an engine built with "
                   "accepts_pdf=True."))
    if by_pdf["outcome"] == "refused":
        why = (f"the PDF was refused before it was sent "
               f"({by_pdf['error'][:200]})")
    elif by_pdf["text_read"]:
        why = ("only the PDF's text word came back, not its picture word: "
               "the route sends the text layer without the page pictures")
    elif by_pdf["picture_read"]:
        why = ("only the PDF's picture word came back, not its text word: "
               "the route sends the page pictures without the text layer")
    else:
        why = "the PDF reply carried neither code word"
    if by_pictures["picture_read"]:
        return (f"{name}: {why}. The pictures call read page 2, so leave "
                f"{FOUNDRY_CLAUDE_PDF_ENV} unset and send the pages as "
                f"pictures.")
    if by_pictures["outcome"] == "failed":
        return (f"{name}: {why}, and the pictures call failed "
                f"({by_pictures['error'][:200]}); run the probe again.")
    return (f"{name}: {why}, and the pictures call did not read page 2 "
            f"either, so the probe says nothing about the PDF route. Read "
            f"the replies.")


def probe_pdf_route(model: Any, *, words: Optional[Tuple[str, str]] = None
                    ) -> Dict[str, Any]:
    """Whether a PDF reaches the model whole on this route: two calls.

    ``model`` is an ingest engine or what :func:`engine_for` turns into one
    (a LangChain chat model above all). ``words`` (text word, picture word)
    is for tests; a probe draws its own. Nothing raises from a call: a
    refusal or an error is recorded on that call.

    Returns ``{"model", "words", "proxy", "accepts_pdf", "pdf",
    "pictures", "use_pdf", "says"}``: ``proxy`` is whether the model is the
    Foundry Claude proxy's class, ``accepts_pdf`` what the engine said
    before the probe, ``pdf`` and ``pictures`` the two calls (``outcome``,
    ``text_read``, ``picture_read``, ``reply``, ``error``, the tokens and
    seconds), ``use_pdf`` whether the PDF call brought both words back, and
    ``says`` the verdict in a sentence.
    """
    engine = engine_for(model)
    if engine is None:
        raise ValueError(f"no ingest engine for a {type(model).__name__}")
    words = tuple(words) if words else fresh_code_words()
    pdf = probe_document(*words)
    accepts = getattr(engine, "accepts_pdf", None)
    sender = engine
    if isinstance(engine, ChatModelEngine) and not engine.accepts_pdf:
        sender = ChatModelEngine(engine.chat_model, meter=engine.meter,
                                 max_tokens=engine.max_tokens,
                                 name=engine.name, accepts_pdf=True)
    by_pdf = _ask(sender, [pdf_block(pdf), text_block(PDF_ASK)], words)
    first, second = _pictures(pdf)
    by_pictures = _ask(engine, [text_block("Page 1:"), image_block(first),
                                text_block("Page 2:"), image_block(second),
                                text_block(PICTURES_ASK)], words)
    name = str(getattr(engine, "name", "") or type(engine).__name__)
    proxy = _on_the_proxy(engine)
    return {"model": name, "words": list(words), "proxy": proxy,
            "accepts_pdf": accepts, "pdf": by_pdf, "pictures": by_pictures,
            "use_pdf": bool(by_pdf["text_read"] and by_pdf["picture_read"]),
            "says": _verdict(name, proxy, accepts, by_pdf, by_pictures)}


# ---------------------------------------------------------------------------
# piece 4: the answer schemas with what section 8.5 adds
# ---------------------------------------------------------------------------
#
# MODEL-FACING. Each class below keeps its parent's name and docstring, and
# a field it retypes keeps its parent's description (read off the parent),
# so the answer schema the model sees is today's plus the new fields and
# nothing else. The new fields come last in their class: pydantic puts a
# subclass's own fields after the inherited ones, and an inherited field
# redefined here (a sample's recovery_percent) keeps its place.

#: Changed whenever what the model-first readers ask changes -- an
#: instruction below or a field the classes below add -- or what they are
#: shown, so a saved answer from another wording is never reused as if it
#: were this one. mf-2 (RI-149): a sheet scanned sideways is sent upright,
#: its text row by row as the sheet reads; the wording is mf-1's. mf-3
#: (build f): the narrative's outsideProject is offered yes, no or unclear,
#: no longer mixed (RI-174); the narrative is sent text first in one call
#: and carries today's house conventions, and the lab instruction asks for
#: Atterberg limits on a gradation sheet as an entry of their own (RI-175).
#: mf-4 (build g): the narrative carries the owner's house rules of
#: 2026-10-29 (RI-182: structures are buildings, the phase and its new
#: companion question projectPhaseOther, this investigation's holes only,
#: the listed hazard phrases only, the file's own figures and tables) and
#: today's rule for another client's report (RI-189).
READ_PROMPT_VERSION = "mf-4"


def _described(cls: Any, name: str) -> Optional[str]:
    """The description today's class gives field ``name``."""
    return cls.model_fields[name].description


class HeaderField(BaseModel):
    """A field the log's header prints that has no field of its own here."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="its label, as printed")
    value: str = Field(description="its value, as printed")


class ReadSample(_log.ReadSample):
    """One sample, and the index values printed against it on the log face."""

    model_config = ConfigDict(extra="forbid")

    recovery_percent: Optional[float] = Field(
        default=None,
        description="recovery printed as a percent, never worked out from a "
                    "length; null if not printed as a percent")
    recovery_length: Optional[float] = Field(
        default=None,
        description="recovery printed as a length, a number under a heading "
                    "such as REC (in) or RECOVERY (cm); null if not printed "
                    "as a length. Fill both this and recovery_percent only "
                    "where the log prints both")
    recovery_length_unit: str = Field(
        default="",
        description="the unit of recovery_length, as printed (in, cm, ft, "
                    "m ...); empty if not printed")


class LogReading(_log.LogReading):
    """One log as the model read it."""

    model_config = ConfigDict(extra="forbid")

    samples: List[ReadSample] = Field(
        default_factory=list,
        description=_described(_log.LogReading, "samples"))
    project: str = Field(
        default="",
        description="the project or job name the log prints, as printed; "
                    "empty if not printed")
    project_number: str = Field(
        default="",
        description="the project, job or contract number the log prints, as "
                    "printed; empty if not printed")
    client: str = Field(
        default="",
        description="the client or owner the log names, as printed; empty "
                    "if not printed")
    header_fields: List[HeaderField] = Field(
        default_factory=list,
        description="every other field the log's header prints that has no "
                    "field here, with its label and its value as printed")


class ReadRow(_lab.ReadRow):
    """One row of a summary-of-laboratory-tests table."""

    model_config = ConfigDict(extra="forbid")

    elevation_bottom: Optional[_lab.ReadQuantity] = Field(
        default=None,
        description="the elevation of the bottom of the sample, with its "
                    "unit as printed, when the table prints a range of "
                    "elevations (the elevation of its top goes in "
                    "elevation_top); null if not printed")


class ReadTest(_lab.ReadTest):
    """One test on one specimen, as the model read it.

    Flat on purpose: every kind of test fills the fields that belong to it
    and leaves the rest null. Python sorts them into the typed result.
    """

    model_config = ConfigDict(extra="forbid")

    rows: List[ReadRow] = Field(
        default_factory=list, description=_described(_lab.ReadTest, "rows"))


class LabSheetReading(_lab.LabSheetReading):
    """One laboratory sheet as the model read it."""

    model_config = ConfigDict(extra="forbid")

    tests: List[ReadTest] = Field(
        description=_described(_lab.LabSheetReading, "tests"))


#: The fields each class above adds, by class: what the model is asked that
#: today's schema does not ask. ``ReadSample.recovery_percent`` is not here:
#: it is today's field with a new description.
ADDED_FIELDS: Dict[str, Tuple[str, ...]] = {
    "LogReading": ("project", "project_number", "client", "header_fields"),
    "ReadSample": ("recovery_length", "recovery_length_unit"),
    "ReadRow": ("elevation_bottom",),
}


def answer_schema(stage: str) -> Any:
    """The answer schema a model-first call of ``stage`` offers as its
    answer tool: logs and lab take the classes above, calc and narrative
    today's (section 8.5 adds nothing to them)."""
    if stage == "logs":
        return LogReading
    if stage == "lab":
        return LabSheetReading
    if stage == "calc":
        from report_ingest.calc_reader import CalcReading
        return CalcReading
    if stage == "narrative":
        from report_ingest.narrative_reader import NarrativeReading
        return NarrativeReading
    raise ValueError(f"no model-first stage {stage!r}")


# ---------------------------------------------------------------------------
# piece 4: the instructions
# ---------------------------------------------------------------------------

#: RI-86's instructions, as the plain baseline measured them (MEASUREMENTS
#: 'CLAUDE LOCAL RUN 2'): one short paragraph a stage in general terms, with
#: the vocabulary the answer schema's own fields point at and nothing about
#: depth gates, halts, links or provenance rules.
#: ``module_work/report_ingest_harness/run_plain_baseline.py`` reads these,
#: so its runs keep the measured text.
PLAIN_INSTRUCTIONS: Dict[str, str] = {
    "logs": (
        "These pages are from a geotechnical report ({where}). Extract the "
        "exploration log on them (a boring, test pit or similar log): its "
        "header (the hole's identifier, location, elevation, total depth, "
        "dates, drilling method, hammer, who logged it), every layer with "
        "its depths, description and the USCS symbol where one is printed, "
        "every sample with its depths and any laboratory values printed "
        "beside it, every SPT or other blow-count record, and the water "
        "levels. Give depths in the log's own unit and say which unit. "
        "{pages_note} Return it with the answer tool."),
    "lab": (
        "These pages are from a geotechnical report ({where}). Extract the "
        "laboratory test results on them: for each test, its kind, the "
        "boring or test pit and the sample depth it belongs to, and every "
        "value it reports (index values, grading points, curve points, "
        "strengths and so on) with the units as printed. Give depths in the "
        "sheet's own unit and say which unit. {pages_note}\n\n"
        "Test kinds:\n{kinds}\n\nSeries names:\n{series}\n\n"
        "Return it with the answer tool."),
    "calc": (
        "These pages are from a geotechnical report ({where}). They hold one "
        "calculation. Extract it: what kind of calculation it is, the "
        "program that printed it if one is named, the method, what it is "
        "for, the labelled values it was given (inputs) and the labelled "
        "values it worked out (results), each with its label and unit as "
        "printed. {pages_note}\n\nCalculation kinds:\n{kinds}\n\n"
        "Return it with the answer tool."),
    "narrative": (
        "This is a geotechnical report ({where}). Answer each field of the "
        "answer tool from it, and leave a field null when the report does "
        "not say. {pages_note}\n\n{vocabulary}\n\n"
        "Return it with the answer tool."),
}

#: What each stage's model-first call is asked: the plain instruction, with
#: the logs' header and sample lists naming what the log schema adds (the
#: project name and number, the client, any other header field, recovery).
#: Calc is asked as the plain baseline was, and lab too but for one
#: sentence (RI-175): two tests on one sample are two entries, Atterberg
#: limits on a grading sheet among them, as today's lab reader asks. The
#: narrative carries today's house conventions as well
#: (``narrative_glossary.glossary_block``, RI-175: a count of explorations
#: not done is 0, null is "the report does not say", the mention
#: questions, the list items) and today's rule for another client's report
#: (RI-189). A summary row's bottom elevation is asked by
#: its field's own description. A change here changes
#: :data:`READ_PROMPT_VERSION`.
#: The sentence the model-first lab instruction adds to the plain one
#: (RI-175): build e's lab reading left out the Atterberg limits printed on
#: gradation sheets, which today's lab reader records as entries of their
#: own (its prompt's "TWO entries, one per kind").
LAB_TWO_TESTS = (
    "A sheet that reports two tests on one sample is two entries, one per "
    "kind, both with the same boring and depth: Atterberg limits printed on "
    "a gradation sheet are an atterberg entry of their own, beside the "
    "gradation entry.")

#: The paragraph the model-first narrative instruction adds after the house
#: conventions (RI-189): today's reader's rule 4 (``NARRATIVE_SYSTEM``),
#: which build f's model-first reading lacked and answered a property type
#: on reports written for other clients.
NARRATIVE_OTHER_CLIENT = (
    "A report about somebody else's project names no post: postName and "
    "propertyType are questions about one estate of government property. "
    "If this report was written for another client -- a private developer, "
    "a municipality, another agency -- set outsideProject to \"yes\", leave "
    "postName null and answer propertyType \"other\". Never infer a post "
    "from a city, a country or a project name.")

INSTRUCTIONS: Dict[str, str] = {
    **PLAIN_INSTRUCTIONS,
    "lab": PLAIN_INSTRUCTIONS["lab"].replace(
        " Give depths in the sheet's own unit",
        f" {LAB_TWO_TESTS} Give depths in the sheet's own unit"),
    # Today's house conventions, filled in by instruction() from
    # narrative_glossary.glossary_block, the text today's reader is given,
    # and today's rule for another client's report.
    "narrative": PLAIN_INSTRUCTIONS["narrative"].replace(
        "{vocabulary}\n\n",
        "{vocabulary}\n\n{glossary}\n\n" + NARRATIVE_OTHER_CLIENT + "\n\n"),
    "logs": (
        "These pages are from a geotechnical report ({where}). Extract the "
        "exploration log on them (a boring, test pit or similar log): its "
        "header (the hole's identifier, the project name and number, the "
        "client, location, elevation, total depth, dates, drilling method, "
        "hammer, who logged it, and any other field the header prints), "
        "every layer with its depths, description and the USCS symbol where "
        "one is printed, every sample with its depths, its recovery and any "
        "laboratory values printed beside it, every SPT or other blow-count "
        "record, and the water levels. Give depths in the log's own unit and "
        "say which unit. {pages_note} Return it with the answer tool."),
}


def _kinds_text(definitions: Dict[str, str]) -> str:
    return "\n".join(f"  {name}: {text}" for name, text in definitions.items())


def instruction(stage: str, report: str, pages: Sequence[int], *,
                whole: bool = False, route: str = "pdf",
                texts: Optional[Dict[str, str]] = None,
                text_first: bool = False) -> str:
    """The instruction for one call: ``texts[stage]`` (by default
    :data:`INSTRUCTIONS`) filled in for its report and pages.

    ``pages`` are the report's 0-based pages the call carries, in order;
    ``whole`` says they are the whole report. ``route`` is how they are sent
    (``pdf`` or ``pictures``), which sets the sentence saying which report
    page each PDF page or picture is (:func:`page_map_sentence`);
    ``text_first`` says the pages with a usable text layer go as their text
    alone (the narrative on the pictures route, RI-175)."""
    texts = INSTRUCTIONS if texts is None else texts
    pages = [int(p) for p in pages]
    if whole:
        where = f"report {report}, all {len(pages)} pages"
    else:
        where = (f"report {report}, page{'s' if len(pages) > 1 else ''} "
                 + ", ".join(str(p) for p in pages))
    note = page_map_sentence(pages, route=route, whole=whole,
                             text_first=text_first)
    fill: Dict[str, str] = {"where": where, "pages_note": note}
    if stage == "lab":
        fill.update(kinds=_kinds_text(_lab.KIND_DEFINITIONS),
                    series=_kinds_text(_lab.SERIES_NAMES))
    elif stage == "calc":
        from report_ingest.calc_reader import CALC_KIND_DEFINITIONS
        fill.update(kinds=_kinds_text(CALC_KIND_DEFINITIONS))
    elif stage == "narrative":
        from report_ingest.narrative_glossary import glossary_block
        from report_ingest.narrative_reader import _vocabulary_block
        fill.update(vocabulary=_vocabulary_block(), glossary=glossary_block())
    return texts[stage].format(**fill)


# ---------------------------------------------------------------------------
# piece 4: where the new fields go in the record
# ---------------------------------------------------------------------------
#
# After today's builder has made the record from the answer (the log's
# ``_Builder`` reads the fields it knows and ignores the rest), these put
# the new fields into slots the record already has. No record field is
# added, so report.record.json keeps its keys and the DIGGS writer its
# elements. A value that cannot go in is returned as an entry in the
# builders' own ``unresolved`` shape, never dropped without a word.

#: The ``Investigation.fields`` key each project field of a log goes under:
#: the names the log scorer's ``FIELD_MAP`` reads and planlens' log grid
#: gives the same header fields on today's path.
PROJECT_FIELD_KEYS: Dict[str, str] = {
    "project": "project", "project_number": "project_number",
    "client": "client",
}

#: The heading a summary row's bottom elevation is kept under in
#: ``SummaryRow.other``, the row's slot for a printed column with no field.
ELEVATION_BOTTOM_HEADING = "elevation bottom"


def _plain(text: Any) -> str:
    return " ".join(str(text or "").split())


def _folded(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


#: The units a model may give a "length" that is a percent; such a recovery
#: belongs in ``recovery_percent``.
PERCENT_UNITS: Tuple[str, ...] = ("%", "percent")


def _number_text(value: float) -> str:
    """A number as text without a float's tail: 412.5, 1234, 0.25."""
    return f"{float(value):.10g}"


def _unnumbered(label: str) -> str:
    """``label`` without the " (2)" :func:`_put_field` numbers a repeat with."""
    return re.sub(r"\s*\(\d+\)$", "", label)


def _put_field(fields: Dict[str, str], label: str, value: str) -> None:
    """``value`` into ``fields`` under ``label``. A label already there
    (spelled any way, numbered or not) with the same value is the same field
    and is left; one with another value is kept beside it under the next
    free number: nothing is dropped."""
    label, value = _plain(label).rstrip(":").strip(), _plain(value)
    if not label or not value:
        return
    label = _unnumbered(label) or label
    same = [k for k in fields
            if _folded(_unnumbered(k)) == _folded(label)]
    if any(_plain(fields[k]) == value for k in same):
        return
    if same:
        taken = {_folded(k) for k in fields}
        n = 2
        while _folded(f"{label} ({n})") in taken:
            n += 1
        label = f"{label} ({n})"
    fields[label] = value


def apply_log_additions(reading: Any, investigation: Investigation
                        ) -> Tuple[Investigation, List[Dict[str, Any]]]:
    """``investigation`` (today's log builder's record of ``reading``) with
    what :class:`LogReading` adds, and the values that could not go in.

    * ``project``, ``project_number`` and ``client`` go into
      ``Investigation.fields`` under :data:`PROJECT_FIELD_KEYS`;
    * each of ``header_fields`` goes into ``Investigation.fields`` under its
      label as printed (:func:`_put_field`);
    * a sample's ``recovery_length`` goes into ``Sample.recovery``, a
      Quantity in ``recovery_length_unit`` (empty when the log prints none,
      which leaves the unit unknown, as the record's rule is). One given in
      percent (:data:`PERCENT_UNITS`) is a percent recovery and goes into
      ``recovery_percent`` when that is empty.

    A recovery stays out, listed, when it is negative, when it is a percent
    above 100 or one that differs from the percent the sample already has,
    and when its sample is not in the record (the builder refused the
    sample's depth). Each sample is found by its label and top depth among
    the built samples not yet taken. Returns a copy; ``investigation`` is
    not changed.
    """
    out = investigation.model_copy(deep=True)
    unresolved: List[Dict[str, Any]] = []
    fields = dict(out.fields)
    for attribute, key in PROJECT_FIELD_KEYS.items():
        _put_field(fields, key, getattr(reading, attribute, ""))
    for printed in getattr(reading, "header_fields", None) or []:
        _put_field(fields, printed.name, printed.value)
    out.fields = fields

    built = list(out.samples)
    taken: set = set()
    for n, sample in enumerate(reading.samples):
        found = next((k for k in range(len(built)) if k not in taken
                      and built[k].sample_id == sample.sample_id
                      and built[k].top.value == float(sample.top)), None)
        if found is not None:
            taken.add(found)
        length = getattr(sample, "recovery_length", None)
        if length is None:
            continue
        length = float(length)
        unit = _plain(getattr(sample, "recovery_length_unit", ""))

        def listed(why: str) -> None:
            unresolved.append({
                "what": f"sample {sample.sample_id or n + 1} recovery",
                "page": sample.prov.page, "value": length, "why": why,
                "refused_by": "python"})

        if found is None:
            listed("its sample is not in the record")
            continue
        if length < 0.0:
            listed("a recovery cannot be negative")
            continue
        target = built[found]
        if unit.lower() in PERCENT_UNITS:
            if target.recovery_percent is None and length <= 100.0:
                target.recovery_percent = length
            elif length > 100.0:
                listed("a recovery in percent cannot be above 100")
            elif target.recovery_percent != length:
                listed(f"given in percent, and the sample's percent "
                       f"recovery is already {target.recovery_percent:g}")
            continue
        if target.recovery is None:
            target.recovery = Quantity(value=length, unit=unit)
        elif (target.recovery.value, target.recovery.unit) != (length, unit):
            listed("the sample already has another recovery length")
    return out, unresolved


def apply_lab_additions(reading: Any, tests: Sequence[LabTest]
                        ) -> Tuple[List[LabTest], List[Dict[str, Any]]]:
    """``tests`` (today's lab builder's records of ``reading``) with what
    :class:`LabSheetReading` adds, and the values that could not go in.

    A summary row's ``elevation_bottom`` goes into ``SummaryRow.other`` as
    (:data:`ELEVATION_BOTTOM_HEADING` with its unit in brackets, the number
    as printed): the record's row has no field for it, and ``other`` is its
    slot for a printed column with none (the lab scorer reads it).

    The builder makes one test per test read and one row per row read, in
    their order, so a row is paired with its reading by position and
    checked by its hole, sample and top depth. A value whose row cannot be
    paired stays out and is listed. A row whose ``other`` already holds the
    number (the model gave it there too) is not given it twice. Returns
    copies; ``tests`` are not changed.
    """
    out = [test.model_copy(deep=True) for test in tests]
    unresolved: List[Dict[str, Any]] = []
    paired = len(out) == len(reading.tests)
    for n, read in enumerate(reading.tests):
        rows = getattr(read, "rows", None) or []
        result = out[n].result if paired else None
        for m, row in enumerate(rows):
            bottom = getattr(row, "elevation_bottom", None)
            if bottom is None:
                continue
            got = None
            if (isinstance(result, SummaryTableResult)
                    and len(result.rows) == len(rows)
                    and result.rows[m].investigation_id
                    == row.investigation_id
                    and result.rows[m].sample_id == row.sample_id
                    and _same_depth(result.rows[m].depth_top,
                                    row.depth_top)):
                got = result.rows[m]
            if got is None:
                unresolved.append({
                    "what": f"summary row {m + 1} "
                            f"({row.investigation_id or '?'}) bottom "
                            f"elevation",
                    "page": read.prov.page, "value": float(bottom.value),
                    "why": "its row could not be found in the record, so "
                           "the value has no row to go in",
                    "refused_by": "python"})
                continue
            if _holds_number(got.other, float(bottom.value)):
                continue
            unit = _plain(bottom.unit)
            heading = (f"{ELEVATION_BOTTOM_HEADING} ({unit})" if unit
                       else ELEVATION_BOTTOM_HEADING)
            got.other.append((heading, _number_text(bottom.value)))
    return out, unresolved


def _same_depth(built: Optional[Quantity], read: Optional[float]) -> bool:
    """Whether a built row's top depth is the one read: both absent, or the
    same number."""
    if built is None or read is None:
        return built is None and read is None
    return float(built.value) == float(read)


def _holds_number(other: Sequence[Tuple[str, str]], value: float) -> bool:
    """Whether a row's ``other`` columns already print ``value``."""
    return any(abs(float(token) - value) < 1e-9
               for _heading, text in other
               for token in re.findall(r"-?\d+(?:\.\d+)?", str(text)))
