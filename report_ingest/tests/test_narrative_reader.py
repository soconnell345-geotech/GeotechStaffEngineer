"""The narrative reader, offline: what it sends, what it keeps, what it refuses.

Every test here runs on the synthetic report with the extended narrative
(``narrative_fixtures``) and a scripted engine. What is under test is never a
model: it is the brief the reader builds, the merge of a long narrative read
in parts, and the rules Python applies to what comes back -- the enumerations,
the citations, the units, the word limits and the two counts Python counts for
itself.
"""

from __future__ import annotations

import pytest

from report_ingest.model import GENERAL_FIELDS, NATURAL_HAZARD_FIELDS
from report_ingest.narrative_reader import (
    MAX_QUOTE_WORDS, NarrativeReading, ReadBearing, ReadCitation, ReadStratum,
    Unsettled, count_captions, iso_date, normalise_asce_version,
    normalise_site_class, read_narrative,
)
from report_ingest.tests.fake_engine import FakeEngine
from report_ingest.tests.narrative_fixtures import (
    BODY_PAGES, NARRATIVE_PAGES, build_narrative_report,
)

pytest.importorskip("planlens.document.roles")


@pytest.fixture(scope="module")
def gt():
    return build_narrative_report()


@pytest.fixture()
def doc(gt):
    from planlens.document import open_document
    document = open_document(gt.pdf, name="SYN")
    try:
        yield document
    finally:
        document.close()


def full_reading(**overrides) -> NarrativeReading:
    """What a reader of the fixture's narrative should come back with."""
    data = dict(
        documentType="geotechnical report",
        quickSummary="A geotechnical investigation for a residential "
                     "development, recommending spread footings.",
        projectName="Rosewood Terrace Development",
        projectNumber="26-118",
        geotechnicalEngineerFirm="Soil & Rock Consulting Engineers",
        testingProgramSummary="Four borings with hollow-stem augers and "
                              "three test pits.",
        boringCount=4,
        testPitCount=3,
        figureCount=2,
        recommendedFoundations=["spread footings"],
        bearingCapacity=["An allowable bearing pressure of 3,000 psf is "
                         "recommended for spread footings"],
        bearingCapacityValues=[ReadBearing(
            value=3000.0, unit="psf", foundation_type="spread footing",
            condition="on the dense residual soil", page=4,
            quote="An allowable bearing pressure of 3,000 psf is recommended")],
        strataList=[ReadStratum(
            name="residual soil", description="dense residual soil",
            top=0.0, bottom=6.0, depth_unit="m", uscs="SM", page=4)],
        liquefactionPotential="low",
        siteClass="Site Class D",
        seismicCodeUsed="ASCE 7-16",
        asceSevenVersion="ASCE 7-16",
        reportDate="14 March 2026",
        citations=[
            ReadCitation(field="boringCount", page=4,
                         quote="Four borings and three test pits were "
                               "completed for this study"),
            ReadCitation(field="siteClass", page=4,
                         quote="The site is classified as Site Class D"),
            ReadCitation(field="liquefactionPotential", page=4,
                         quote="The liquefaction potential of the site soils "
                               "is considered low"),
        ],
    )
    data.update(overrides)
    return NarrativeReading(**data)


def read(doc, engine, **kwargs):
    return read_narrative(doc, list(NARRATIVE_PAGES), engine,
                          body_pages=list(BODY_PAGES), report_id="SYN",
                          **kwargs)


class TestOneCall:
    """A narrative that fits in one call costs one call."""

    def test_answers_both_schemas(self, doc):
        engine = FakeEngine([{"final": full_reading()}])
        result = read(doc, engine)

        assert engine.n_calls == 1
        assert result.model_calls == 1
        assert result.general.documentType == "geotechnical report"
        assert result.general.boringCount == 4
        assert result.natural_hazards.liquefactionPotential == "low"
        assert result.natural_hazards.siteClass == "Site Class D"
        assert "boringCount" in result.answered
        assert "siteClass" in result.answered

    def test_the_brief_carries_the_pages_the_vocabulary_and_the_counts(
            self, doc):
        engine = FakeEngine([{"final": full_reading()}])
        read(doc, engine)

        sent = engine.calls[0]
        brief = sent["messages"][0]["content"][0]["text"]
        assert "=== page 2 ===" in brief and "=== page 4 ===" in brief
        assert "=== page 5 ===" not in brief          # the figure is not narrative
        assert "geotechnical report | environmental report" in brief
        assert "figures captioned in the main body: 1" in brief
        assert "ASCE 7-16" in brief                   # the page's own words
        assert sent["output_format"] is NarrativeReading
        assert sent["n_images"] == 0                  # prose is read, not seen

    def test_cost_is_metered(self, doc):
        engine = FakeEngine([{"final": full_reading()}])
        result = read(doc, engine)

        assert result.cost["calls"] == 1
        assert result.cost["input_tokens"] == 1000
        assert result.model == "fake-model"

    def test_nothing_the_narrative_does_not_say_is_answered(self, doc):
        engine = FakeEngine([{"final": full_reading()}])
        result = read(doc, engine)

        assert result.general.postName is None
        assert result.general.propertyType is None
        assert result.general.primeContractor is None
        assert result.natural_hazards.soilCorrosion is None


class TestTheTypedTwins:

    def test_a_bearing_pressure_keeps_its_printed_unit(self, doc):
        engine = FakeEngine([{"final": full_reading()}])
        result = read(doc, engine)

        (value,) = result.general.bearingCapacityValues
        assert value.value.value == 3000.0
        assert value.value.unit == "psf"
        assert value.foundation_type == "spread footing"
        assert round(value.value.to_si().value, 1) == 143.6   # kPa
        assert value.citation[0].page == 4

    def test_a_pressure_with_no_unit_is_refused(self, doc):
        reading = full_reading(bearingCapacityValues=[
            ReadBearing(value=3000.0, unit="", foundation_type="footing")])
        result = read(doc, FakeEngine([{"final": reading}]))

        assert result.general.bearingCapacityValues == []
        assert any(u["what"] == "bearingCapacityValues" and "no unit" in u["why"]
                   for u in result.unresolved)

    def test_the_profile_becomes_records(self, doc):
        result = read(doc, FakeEngine([{"final": full_reading()}]))

        (stratum,) = result.general.strataList
        assert stratum.name == "residual soil"
        assert stratum.uscs == "SM"
        assert stratum.top.unit == "m" and stratum.bottom.value == 6.0

    def test_a_stratum_whose_base_is_above_its_top_loses_the_base(self, doc):
        reading = full_reading(strataList=[ReadStratum(
            name="fill", top=5.0, bottom=1.0, depth_unit="m")])
        result = read(doc, FakeEngine([{"final": reading}]))

        (stratum,) = result.general.strataList
        assert stratum.top.value == 5.0 and stratum.bottom is None
        assert any("base is above the top" in u["why"]
                   for u in result.unresolved)

    def test_the_mentions_and_the_normalised_values(self, doc):
        reading = full_reading(soilCorrosion="yes", outsideProject="no",
                               citations=[ReadCitation(
                                   field="soilCorrosion", page=3,
                                   quote="Soil corrosivity testing was "
                                         "performed")])
        result = read(doc, FakeEngine([{"final": reading}]))

        assert result.natural_hazards.soilCorrosionAnswer.answer == "yes"
        assert result.natural_hazards.soilCorrosionAnswer.citation[0].page == 3
        assert result.general.outsideProjectAnswer.answer == "no"
        assert result.natural_hazards.siteClassNormalized == "D"
        assert result.natural_hazards.asceSevenVersionNormalized == "7-16"
        assert result.natural_hazards.reportDateISO == "2026-03-14"

    def test_a_verdict_keeps_its_reason_and_the_twin_keeps_the_verdict(
            self, doc):
        # The hand's own answers read "yes - six seismic refraction lines
        # across the site": the verdict is the answer and the reason is what
        # makes it worth having, so the field keeps both.
        reading = full_reading(
            geophysicalTestingMention="yes - six seismic refraction and MASW "
                                      "lines across the site",
            soilCorrosion="mixed - non-corrosive to ductile iron but high "
                          "chloride")
        result = read(doc, FakeEngine([{"final": reading}]))

        hazards = result.natural_hazards
        assert hazards.geophysicalTestingMention.startswith("yes - six")
        assert hazards.geophysicalTestingAnswer.answer == "yes"
        assert hazards.soilCorrosion.startswith("mixed -")
        assert hazards.soilCorrosionAnswer.answer == "mixed"

    def test_a_verdict_field_that_does_not_lead_with_one_is_refused(self, doc):
        reading = full_reading(
            siteResponseMention="a site response analysis was performed")
        result = read(doc, FakeEngine([{"final": reading}]))

        assert result.natural_hazards.siteResponseMention is None
        assert any(u["what"] == "siteResponseMention" and "yes, no, mixed"
                   in u["why"] for u in result.unresolved)
        assert result.natural_hazards.siteClassNormalized == "D"
        assert result.natural_hazards.asceSevenVersionNormalized == "7-16"
        assert result.natural_hazards.reportDateISO == "2026-03-14"


class TestWhatIsRefused:

    def test_a_value_outside_a_CLOSED_vocabulary_is_not_stored(self, doc):
        # documentType is the owner's own list, so an answer outside it is a
        # misreading rather than an unfamiliar answer.
        reading = full_reading(documentType="site investigation writeup")
        result = read(doc, FakeEngine([{"final": reading}]))

        assert result.general.documentType is None
        assert any(u["what"] == "documentType" and
                   u["value"] == "site investigation writeup"
                   for u in result.unresolved)

    def test_a_value_outside_an_OPEN_vocabulary_is_kept_as_written(self, doc):
        # The liquefaction words are only what has been seen so far. Refusing
        # the report's own verdict would lose it; the first draft of this
        # package did exactly that and the hand answers proved it wrong.
        reading = full_reading(liquefactionPotential="marginally liquefiable")
        result = read(doc, FakeEngine([{"final": reading}]))

        assert result.natural_hazards.liquefactionPotential == \
            "marginally liquefiable"
        assert not any(u["what"] == "liquefactionPotential"
                       for u in result.unresolved)

    def test_a_known_value_is_folded_onto_the_owners_spelling(self, doc):
        reading = full_reading(documentType="Geotechnical Report",
                               projectPhase="DESIGN-BUILD",
                               propertyType="new embassy or consulate "
                                            "compound")
        result = read(doc, FakeEngine([{"final": reading}]))

        assert result.general.documentType == "geotechnical report"
        assert result.general.projectPhase == "Design-build"
        assert result.general.propertyType == \
            "New embassy or consulate compound"

    def test_a_hazard_in_the_reports_own_terms_is_kept(self, doc):
        reading = full_reading(earthHazardsExposed=[
            "liquefaction-induced settlement", "flooding"])
        result = read(doc, FakeEngine([{"final": reading}]))

        assert result.natural_hazards.earthHazardsExposed == [
            "liquefaction-induced settlement", "flooding"]
        # Nothing was thrown away for being unfamiliar. (The field still
        # earns the standing "answered with no citation" note, which is a
        # different complaint about a different thing.)
        assert not any(u.get("value") for u in result.unresolved
                       if u["what"] == "earthHazardsExposed")

    def test_a_not_stated_string_is_stored_as_null(self, doc):
        reading = full_reading(primeContractor="N/A", postName="not stated")
        result = read(doc, FakeEngine([{"final": reading}]))

        assert result.general.primeContractor is None
        assert result.general.postName is None

    def test_a_citation_on_another_page_is_thrown_away(self, doc):
        reading = full_reading(citations=[
            ReadCitation(field="boringCount", page=7,
                         quote="BORING LOG NO. B-1")])
        result = read(doc, FakeEngine([{"final": reading}]))

        assert "boringCount" not in result.citations
        assert any(u.get("page") == 7 and "not in this narrative" in u["why"]
                   for u in result.unresolved)

    def test_a_citation_for_a_field_that_does_not_exist_is_thrown_away(
            self, doc):
        reading = full_reading(citations=[
            ReadCitation(field="soilColour", page=4, quote="brown")])
        result = read(doc, FakeEngine([{"final": reading}]))

        assert any("does not exist" in u["why"] for u in result.unresolved)

    def test_an_answer_with_no_citation_is_recorded(self, doc):
        result = read(doc, FakeEngine([{"final": full_reading()}]))

        assert any(u["what"] == "projectName" and
                   u["why"] == "answered with no citation"
                   for u in result.unresolved)

    def test_a_quote_is_cut_to_twenty_words(self, doc):
        long_quote = " ".join(f"word{i}" for i in range(40))
        reading = full_reading(citations=[
            ReadCitation(field="boringCount", page=4, quote=long_quote)])
        result = read(doc, FakeEngine([{"final": reading}]))

        quote = result.citations["boringCount"][0].quote
        assert len(quote.split()) == MAX_QUOTE_WORDS + 0
        assert quote.endswith("…")

    def test_a_summary_past_its_limit_is_kept_and_flagged(self, doc):
        long_summary = " ".join(["word"] * 130)
        reading = full_reading(quickSummary=long_summary)
        result = read(doc, FakeEngine([{"final": reading}]))

        assert result.general.quickSummary == long_summary
        assert any(u["what"] == "quickSummary" and "100-word" in u["why"]
                   for u in result.unresolved)

    def test_a_negative_count_is_refused(self, doc):
        reading = full_reading(cptCount=-1)
        result = read(doc, FakeEngine([{"final": reading}]))

        assert result.general.cptCount is None
        assert any("cannot be negative" in u["why"] for u in result.unresolved)

    def test_the_model_s_own_unsettled_list_comes_through(self, doc):
        reading = full_reading(unsettled=[Unsettled(
            what="the groundwater depth", page=3,
            why="the sentence is cut off at the page break")])
        result = read(doc, FakeEngine([{"final": reading}]))

        assert any(u["what"] == "the groundwater depth"
                   for u in result.unresolved)


class TestTheCountsPythonCounts:

    def test_the_counted_figures_beat_the_reading(self, doc):
        result = read(doc, FakeEngine([{"final": full_reading()}]))

        # The body captions one figure; the report's own list names two and
        # the reading answered two. What is stored is what was counted.
        assert result.general.figureCount == 1
        assert result.facts.stated_counts["figures"] == 2
        assert result.facts.counted["figures"] == 1
        assert any(u["what"] == "figureCount" and "captions" in u["why"]
                   for u in result.unresolved)

    def test_a_count_nothing_was_counted_for_keeps_the_reading(self, doc):
        reading = full_reading(tableCount=3)
        result = read(doc, FakeEngine([{"final": reading}]))

        assert result.facts.counted["tables"] == 0
        assert result.general.tableCount == 3

    def test_stated_counts_carry_what_the_narrative_said(self, doc):
        result = read(doc, FakeEngine([{"final": full_reading()}]))

        assert result.facts.stated_counts["borings"] == 4
        assert result.facts.stated_counts["test_pits"] == 3

    def test_count_captions_counts_labels_not_lines(self, doc):
        counted = count_captions(doc, list(BODY_PAGES))

        assert counted["figures"] == 1
        assert counted["figure_labels"] == ["figure 1"]


class TestALongNarrative:
    """Read in parts, merged in Python, summarised once."""

    def test_each_chunk_is_one_call_and_the_summaries_are_the_last(
            self, doc):
        from report_ingest.narrative_reader import SummaryReading
        first = full_reading(boringCount=4, quickSummary=None)
        second = full_reading(boringCount=None, testPitCount=3,
                              projectNumber="26-118", citations=[])
        engine = FakeEngine([
            {"final": first}, {"final": second}, {"final": second},
            {"final": SummaryReading(
                quickSummary="A due diligence study of a parking lot site.")},
        ])
        result = read(doc, engine, budget=8, max_chunk_chars=1200)

        assert engine.n_calls == 4                 # three parts, one summary
        assert result.general.boringCount == 4
        assert result.general.quickSummary == \
            "A due diligence study of a parking lot site."
        brief = engine.calls[1]["messages"][0]["content"][0]["text"]
        assert "This is part 2 of 3" in brief

    def test_two_parts_that_disagree_are_recorded_not_settled(self, doc):
        from report_ingest.narrative_reader import SummaryReading
        engine = FakeEngine([
            {"final": full_reading(boringCount=4)},
            {"final": full_reading(boringCount=6, citations=[])},
            {"final": full_reading(boringCount=6, citations=[])},
            {"final": SummaryReading()},
        ])
        result = read(doc, engine, budget=8, max_chunk_chars=1200)

        assert result.general.boringCount == 4     # the first answer stands
        assert any(u["what"] == "boringCount" and "differently" in u["why"]
                   for u in result.unresolved)

    def test_lists_from_two_parts_are_unioned(self, doc):
        from report_ingest.narrative_reader import SummaryReading
        engine = FakeEngine([
            {"final": full_reading(boringDictionary=["B-1", "B-2"])},
            {"final": full_reading(boringDictionary=["B-2", "B-3"],
                                   citations=[])},
            {"final": full_reading(boringDictionary=["B-4"], citations=[])},
            {"final": SummaryReading()},
        ])
        result = read(doc, engine, budget=8, max_chunk_chars=1200)

        assert result.general.boringDictionary == ["B-1", "B-2", "B-3", "B-4"]

    def test_a_budget_too_small_for_the_chunks_says_what_was_not_read(
            self, doc):
        engine = FakeEngine([{"final": full_reading()},
                             {"final": full_reading(citations=[])}])
        result = read(doc, engine, budget=2, max_chunk_chars=1200)

        assert engine.n_calls == 1                 # one chunk, one kept back
        assert any("were not read" in w for w in result.warnings)


class TestFailures:

    def test_no_structured_answer_is_an_error_not_an_empty_record(self, doc):
        engine = FakeEngine([{"text": "I could not read this."}])
        with pytest.raises(RuntimeError, match="no structured reading"):
            read(doc, engine)

    def test_no_pages_is_an_error(self, doc):
        with pytest.raises(ValueError, match="at least one page"):
            read_narrative(doc, [], FakeEngine([]))


class TestTheDeterministicNormalisations:

    @pytest.mark.parametrize("text,expected", [
        ("Site Class D", "D"), ("site class c/d", "CD"), ("D", "D"),
        ("Class E (soft clay)", "E"), ("stiff soil profile", None),
        (None, None),
    ])
    def test_site_class(self, text, expected):
        assert normalise_site_class(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("ASCE 7-16", "7-16"), ("ASCE/SEI 7-22 (2022)", "7-22"),
        ("ASCE 7", None), ("IBC 2018", None),
    ])
    def test_asce_version(self, text, expected):
        assert normalise_asce_version(text) == expected

    @pytest.mark.parametrize("text,expected", [
        ("14 March 2026", "2026-03-14"), ("March 14, 2026", "2026-03-14"),
        ("3/14/2026", "2026-03-14"), ("2026-03-14", "2026-03-14"),
        ("Sept 1, 2019", "2019-09-01"), ("some time in 2026", None),
        ("32 March 2026", None),
    ])
    def test_iso_date(self, text, expected):
        assert iso_date(text) == expected


def test_the_field_lists_are_the_owners_two_schemas():
    """The names are the owner's, verbatim, and nothing renamed them."""
    assert GENERAL_FIELDS[0] == "documentType"
    assert "geotechnicalEngineerFirm" in GENERAL_FIELDS
    assert "boringDictionary" in GENERAL_FIELDS
    assert len(GENERAL_FIELDS) == 25
    assert NATURAL_HAZARD_FIELDS[0] == "liquefactionPotential"
    assert "hazardAnalysisMention" in NATURAL_HAZARD_FIELDS
    assert len(NATURAL_HAZARD_FIELDS) == 12


class TestTheCallersOwnQuestions:
    """A question asked beside the schemas is answered under the same rules."""

    def test_it_is_put_in_the_brief_and_answered_with_its_citation(self, doc):
        from report_ingest.narrative_reader import ReadExtra

        reading = full_reading(extra_answers=[ReadExtra(
            question="What embedment is required?",
            answer="A minimum of 0.6 m below finished grade.",
            page=4, quote="with a minimum embedment of 0.6 m below finished "
                          "grade")])
        engine = FakeEngine([{"final": reading}])
        result = read(doc, engine,
                      questions=["What embedment is required?"])

        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "THE CALLER'S OWN QUESTIONS" in brief
        assert "1. What embedment is required?" in brief
        (answer,) = result.facts.extra_answers
        assert answer["answer"].startswith("A minimum of 0.6 m")
        assert answer["page"] == 4

    def test_a_question_the_narrative_does_not_answer_keeps_its_question(
            self, doc):
        from report_ingest.narrative_reader import ReadExtra

        reading = full_reading(extra_answers=[ReadExtra(
            question="Who paid for the drilling?", answer="not stated")])
        result = read(doc, FakeEngine([{"final": reading}]),
                      questions=["Who paid for the drilling?"])

        (answer,) = result.facts.extra_answers
        assert answer["question"] == "Who paid for the drilling?"
        assert answer["answer"] == ""

    def test_no_questions_means_no_block_in_the_brief(self, doc):
        engine = FakeEngine([{"final": full_reading()}])
        read(doc, engine)

        brief = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "THE CALLER'S OWN QUESTIONS" not in brief
