"""BACKLOG RI-200: the model-first narrative gets today's whole-report search.

Seven questions are as often answered in an appendix as in the prose
(siteClass, asceSevenVersion, seismicCodeUsed, soilCorrosion,
bearingCapacity, liquefactionPotential, reportDate). Today's narrative
reader searches every page for them and hands the passages over with their
page numbers; the model-first narrative now does the same. The report here
is made up: its prose names no ASCE 7 edition, and the edition is printed
only on an appendix calculation page.
"""

from __future__ import annotations

import pytest

from report_ingest import model_first as mf
from report_ingest import model_first_read as mfr
from report_ingest.model_first_pages import PageGroup
from report_ingest.narrative_reader import (DEFAULT_CONFIDENCE,
                                            NarrativeReading,
                                            RETRIEVAL_PHRASES)
from report_ingest.tests.fake_engine import FakeEngine

pymupdf = pytest.importorskip("pymupdf")

REPORT = "R00"
PROSE = "The made-up site is underlain by made-up silty sand over clay."
#: The appendix calculation page, and the only page naming the edition.
CALC_PAGE = 9
CALC_LINE = "Seismic parameters computed per ASCE 7-16 for the made-up site."


@pytest.fixture(scope="module")
def report_pdf():
    doc = pymupdf.open()
    for n in range(12):
        page = doc.new_page()
        lines = ([f"MADE-UP PROSE {n}", PROSE, PROSE, PROSE] if n < 4
                 else [f"MADE-UP APPENDIX CALCULATION {n}", CALC_LINE]
                 if n == CALC_PAGE else [f"MADE-UP FIGURE {n}"])
        for k, line in enumerate(lines):
            page.insert_text((72, 72 + 18 * k), line, fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture()
def doc(report_pdf):
    from planlens.document import open_document

    opened = open_document(report_pdf, name="made_up_retrieval")
    yield opened
    opened.close()


def _group():
    return PageGroup("narrative", [0, 1, 2, 3], key="narrative",
                     ranked=[0, 1, 2, 3], n_front=1)


def _answer():
    return {"final": NarrativeReading.model_validate({
        "projectName": "Made-up Bridge",
        "asceSevenVersion": "ASCE 7-16",
        "citations": [
            {"field": "projectName", "page": 0, "quote": "MADE-UP PROSE 0"},
            {"field": "asceSevenVersion", "page": CALC_PAGE,
             "quote": "computed per ASCE 7-16"}]})}


def _instruction(engine):
    return engine.calls[0]["messages"][0]["content"][-1]["text"]


def test_the_seven_questions_are_today_s():
    assert set(RETRIEVAL_PHRASES) == {
        "siteClass", "asceSevenVersion", "seismicCodeUsed", "soilCorrosion",
        "bearingCapacity", "liquefactionPotential", "reportDate"}


@pytest.mark.parametrize("pdf", [False, True])
def test_an_edition_printed_only_in_an_appendix_is_shown_and_kept(doc, pdf):
    engine = FakeEngine([_answer()])
    engine.accepts_pdf = pdf
    blob = mfr.read_narrative(doc, [_group()], engine, report_id=REPORT)

    said = _instruction(engine)
    assert "PASSAGES FOUND ELSEWHERE IN THE REPORT" in said
    assert f"page {CALC_PAGE}: " in said and "ASCE 7-16" in said
    # Before the instruction's last sentence, with the other addenda.
    assert said.rstrip().endswith("Return it with the answer tool.")
    assert blob["natural_hazards"]["asceSevenVersion"] == "ASCE 7-16"
    # Its citation stands on the appendix page: the quote gate found the
    # quote there, so the answer keeps the full confidence.
    assert blob["confidence"]["asceSevenVersion"] == DEFAULT_CONFIDENCE
    # A searched page is citable: no row saying the answer mixed its
    # call's pages with others.
    assert not [u for u in blob["unresolved"]
                if u.get("flag") == "page_numbers"]
    # The pages the narrative's groups hold are what it read.
    assert blob["pages"] == [0, 1, 2, 3]


def test_the_search_stays_inside_its_window(doc):
    engine = FakeEngine([_answer()])
    mfr.read_narrative(doc, [_group()], engine, report_id=REPORT,
                       window=list(range(CALC_PAGE)))
    assert "PASSAGES FOUND ELSEWHERE" not in _instruction(engine)


def test_retrieve_off_sends_the_instruction_alone(doc):
    engine = FakeEngine([_answer()])
    engine.accepts_pdf = True
    mfr.read_narrative(doc, [_group()], engine, report_id=REPORT,
                       retrieve=False)
    assert _instruction(engine) == mf.instruction(
        "narrative", REPORT, [0, 1, 2, 3], route="pdf")


def test_the_graph_hands_the_narrative_its_window(monkeypatch):
    """``_read_unit`` gives the model-first narrative the window today's
    narrative is searched in (a bound document's pages, or the parent's own
    under ``narrative_skips_bound``)."""
    from report_ingest import graph

    seen = {}

    def fake(doc, groups, engine, **kw):
        seen.update(kw)
        return {}

    monkeypatch.setattr(mfr, "read_narrative", fake)

    class _MF:
        engines = {"narrative": object()}
        routes = {"narrative": "pictures"}
        place = None
        max_request_tokens_in = None

    graph._read_unit(None, _MF(), "narrative", [_group()], REPORT, None,
                     [0, 1], window=[0, 1, 2, 3])
    assert seen["window"] == [0, 1, 2, 3]
