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


#: A previous investigation bound in at pages 4-9; page 6 holds the only
#: edition in the file, and it is the earlier report's (RI-94).
PRIOR_PAGES = "4-9"
PRIOR_LINE = "Seismic design per ASCE 7-05 for the earlier made-up building."
#: A phrase printed on every page, more often before a window than the
#: search's hit limit.
COMMON_LINE = "Liquefaction potential is low at the made-up site."


def _made_up_pdf(page_lines):
    doc = pymupdf.open()
    for lines in page_lines:
        page = doc.new_page()
        for k, line in enumerate(lines):
            page.insert_text((72, 72 + 18 * k), line, fontsize=9)
    data = doc.tobytes()
    doc.close()
    return data


def _opened(data, name):
    from planlens.document import open_document

    return open_document(data, name=name)


@pytest.fixture(scope="module")
def report_pdf():
    return _made_up_pdf(
        [f"MADE-UP PROSE {n}", PROSE, PROSE, PROSE] if n < 4
        else [f"MADE-UP APPENDIX CALCULATION {n}", CALC_LINE]
        if n == CALC_PAGE else [f"MADE-UP FIGURE {n}"] for n in range(12))


@pytest.fixture()
def doc(report_pdf):
    opened = _opened(report_pdf, "made_up_retrieval")
    yield opened
    opened.close()


@pytest.fixture()
def prior_doc():
    opened = _opened(_made_up_pdf(
        [f"MADE-UP PROSE {n}", PROSE, PROSE] if n < 4
        else [f"MADE-UP EARLIER REPORT {n}", PRIOR_LINE] if n == 6
        else [f"MADE-UP EARLIER REPORT {n}"] if n < 10
        else [f"MADE-UP FIGURE {n}"] for n in range(12)), "made_up_prior")
    yield opened
    opened.close()


@pytest.fixture()
def common_doc():
    opened = _opened(_made_up_pdf(
        [f"MADE-UP PAGE {n}", COMMON_LINE] for n in range(12)),
        "made_up_common")
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


def _prior_answer():
    return {"final": NarrativeReading.model_validate({
        "projectName": "Made-up Bridge",
        "citations": [
            {"field": "projectName", "page": 0, "quote": "MADE-UP PROSE 0"}]})}


@pytest.mark.parametrize("kind", ["appended_prior_report", "other", None])
def test_a_bound_previous_investigation_is_not_searched(prior_doc, kind):
    """The parent's prose names no edition; the earlier report bound in at
    pages 4-9 prints one on page 6. With no window (the default route), the
    parent is not offered it (review of RI-94/RI-200, finding 1)."""
    engine = FakeEngine([_prior_answer()])
    row = {"bound_id": "bound1", "pages": PRIOR_PAGES,
           "title": "Made-up Earlier Study"}
    row.update({"kind": kind} if kind else {})
    mfr.read_narrative(prior_doc, [_group()], engine, report_id=REPORT,
                       bound_documents=[row])
    said = str(engine.calls[0]["messages"])
    assert "ASCE 7-05" not in said
    assert "page 6: " not in said


@pytest.mark.parametrize("kind", ["volume", "data_report"])
def test_a_bound_volume_of_this_investigation_is_searched(prior_doc, kind):
    """A volume or data report of THIS investigation stays searchable."""
    engine = FakeEngine([_prior_answer()])
    mfr.read_narrative(prior_doc, [_group()], engine, report_id=REPORT,
                       bound_documents=[{"bound_id": "bound1",
                                         "pages": PRIOR_PAGES, "kind": kind}])
    said = _instruction(engine)
    assert "page 6: " in said and "ASCE 7-05" in said


def test_a_window_finds_a_common_phrase_inside_it(common_doc):
    """Every page prints the phrase, so the first four hits of the file
    are all before the window: the search runs over the window's pages,
    and the hit limit applies there (finding 2)."""
    from report_ingest.narrative_reader import retrieval_passages

    passages = retrieval_passages(common_doc,
                                  fields=["liquefactionPotential"],
                                  window=[8, 9, 10, 11])
    assert sorted(row["page"] for row in passages) == [8, 9, 10, 11]


def test_a_bound_child_is_offered_its_own_hits(common_doc):
    engine = FakeEngine([{"final": NarrativeReading.model_validate({
        "liquefactionPotential": "low",
        "citations": [{"field": "liquefactionPotential", "page": 8,
                       "quote": "Liquefaction potential is low"}]})}])
    child = PageGroup("narrative", [8, 9], key="narrative", ranked=[8, 9],
                      n_front=1)
    mfr.read_narrative(common_doc, [child], engine, report_id=REPORT,
                       window=[8, 9, 10, 11])
    said = _instruction(engine)
    assert "PASSAGES FOUND ELSEWHERE IN THE REPORT" in said
    assert "page 11: " in said and "page 3: " not in said


def test_positions_map_against_the_call_s_own_pages():
    """The call sent pages 20-29 and the search found pages 5 and 300. An
    answer citing 0, 3 and 9 gave positions in the ten pages it was sent;
    a cite of searched page 300 is the report's page (finding 3)."""
    reading = NarrativeReading.model_validate({"citations": [
        {"field": "projectName", "page": 0, "quote": "a"},
        {"field": "siteClass", "page": 3, "quote": "b"},
        {"field": "reportDate", "page": 9, "quote": "c"},
        {"field": "asceSevenVersion", "page": 300, "quote": "d"}]})
    out, _rows, said = mfr._narrative_on_report_pages(
        reading, list(range(20, 30)), [5, 300])
    assert [c.page for c in out.citations] == [20, 23, 29, 300]
    assert "positions" in said


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
