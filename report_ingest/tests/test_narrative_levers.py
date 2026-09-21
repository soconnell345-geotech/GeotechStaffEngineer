"""The narrative reader's 5.24.0 accuracy levers, offline.

The eight-report cluster run (2026-09-18) put the narrative reader at 64 %
recall and 74 % precision, and the per-question table said the losses were
mostly NOT comprehension: the identity fields were answered from pages the
reader was never shown, several fields were answered under a different house
convention from the hand's, and the free-text fields were scored on wording.
Five levers follow from that, and this file is the offline half of each:

1. the input page set -- the narrative plus the cover, the letter, the
   contents and the front of the report, inside a token budget;
2. the glossary and conventions block reaching the prompt;
3. per-question retrieval over the WHOLE report, with page numbers;
4. the five exploration answers taken from the logs, disagreements recorded;
5. the quote gate.

Nothing here calls a model. Lever 6, the lenient scoring view, is in
``test_narrative_scoring.py``'s company at the bottom of this file because it
is the same train.
"""

from __future__ import annotations

from dataclasses import dataclass


import pytest

from report_ingest.narrative_reader import (
    DEFAULT_CONFIDENCE, MIN_TEXT_CHARS, NarrativeReading, ReadCitation,
    RETRIEVAL_PHRASES, UNVERIFIED_CONFIDENCE, explorations_found,
    picture_pages, read_narrative, reading_pages, retrieval_passages,
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


def a_reading(**overrides) -> NarrativeReading:
    data = dict(
        documentType="geotechnical report",
        projectName="Rosewood Terrace Development",
        boringCount=4,
        testPitCount=3,
        siteClass="Site Class D",
        citations=[ReadCitation(
            field="siteClass", page=4,
            quote="The site is classified as Site Class D")],
    )
    data.update(overrides)
    return NarrativeReading(**data)


def read(doc, engine, **kwargs):
    kwargs.setdefault("retrieve", False)
    kwargs.setdefault("pictures", False)
    return read_narrative(doc, list(NARRATIVE_PAGES), engine,
                          body_pages=list(BODY_PAGES), report_id="SYN",
                          **kwargs)


def brief_of(engine, call: int = 0) -> str:
    return engine.calls[call]["messages"][0]["content"][0]["text"]


# ---------------------------------------------------------------------------
# lever 1: what the reader is given
# ---------------------------------------------------------------------------

class TestTheInputPageSet:
    """The narrative, the front matter, and the front of the report."""

    def test_the_labelled_pages_go_in_whatever_the_narrative_item_said(
            self, doc):
        # The synthetic report's cover is page 0 and its contents page 1;
        # neither is part of the narrative work item (pages 2-4).
        got = reading_pages(doc, NARRATIVE_PAGES, front_pages=0)

        assert got == [0, 1, 2, 3, 4]

    def test_the_front_of_the_report_goes_in_whatever_it_was_labelled(
            self, doc):
        # Pages 5-7 are a figure, a divider and a boring log. None is front
        # matter by label; all three are in the first eight pages.
        got = reading_pages(doc, NARRATIVE_PAGES, front_pages=8)

        assert got == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_the_set_is_deduplicated_and_in_page_order(self, doc):
        got = reading_pages(doc, [4, 2, 4, 3], front_pages=3)

        assert got == sorted(set(got))
        assert got == [0, 1, 2, 3, 4]

    def test_it_never_runs_past_the_end_of_the_report(self, doc):
        got = reading_pages(doc, NARRATIVE_PAGES, front_pages=500)

        assert got == list(range(doc.n_pages))

    def test_roles_that_will_not_compute_leave_the_narrative_and_the_front(
            self, doc):
        got = reading_pages(doc, NARRATIVE_PAGES, roles=[], front_pages=2)

        assert got == [0, 1, 2, 3, 4]

    def test_the_reader_is_given_the_whole_set_not_just_the_narrative(
            self, doc):
        engine = FakeEngine([{"final": a_reading()}])
        result = read(doc, engine, front_pages=6)

        brief = brief_of(engine)
        for page in (0, 1, 2, 3, 4, 5):
            assert f"=== page {page} ===" in brief
        assert result.pages == [0, 1, 2, 3, 4, 5]

    def test_the_token_budget_drops_the_later_pages_and_keeps_the_front(
            self, doc):
        engine = FakeEngine([{"final": a_reading()}])
        # A budget that cannot hold the whole set: the front matter is taken
        # first and in full, and what will not fit is dropped from the end.
        result = read(doc, engine, front_pages=3, token_budget=600)

        brief = brief_of(engine)
        assert "=== page 0 ===" in brief and "=== page 2 ===" in brief
        assert result.pages[:3] == [0, 1, 2]
        assert len(result.pages) < 5
        assert any("input budget" in w for w in result.warnings)

    def test_a_page_with_no_text_layer_is_sent_as_a_picture(self, doc):
        # The photographs page carries a caption and nothing else.
        assert picture_pages(doc, range(doc.n_pages))
        assert 10 in picture_pages(doc, range(doc.n_pages))
        chars = sum(len(line.text) for line in doc.page(10).lines)
        assert chars < MIN_TEXT_CHARS

        engine = FakeEngine([{"final": a_reading()}])
        read(doc, engine, front_pages=12, pictures=True)

        assert engine.calls[0]["n_images"] >= 1
        assert "PAGES SENT AS PICTURES" in brief_of(engine)

    def test_pictures_off_sends_none(self, doc):
        engine = FakeEngine([{"final": a_reading()}])
        read(doc, engine, front_pages=12, pictures=False)

        assert engine.calls[0]["n_images"] == 0


# ---------------------------------------------------------------------------
# lever 2: the glossary
# ---------------------------------------------------------------------------

class TestTheGlossaryReachesThePrompt:
    """The house rules are data the owner edits, and they are IN the brief."""

    def test_the_conventions_block_is_in_the_brief(self, doc):
        from report_ingest.narrative_glossary import CONVENTIONS

        engine = FakeEngine([{"final": a_reading()}])
        read(doc, engine, front_pages=0, roles=[])
        brief = brief_of(engine)

        assert "THE HOUSE CONVENTIONS" in brief
        assert "(DRAFT)" in brief
        for row in CONVENTIONS:
            assert row.rule.split(".")[0][:40] in brief

    def test_every_field_a_rule_names_exists_in_the_schemas(self):
        from report_ingest.model import GENERAL_FIELDS, NATURAL_HAZARD_FIELDS
        from report_ingest.narrative_glossary import unknown_fields

        assert unknown_fields(GENERAL_FIELDS + NATURAL_HAZARD_FIELDS) == []

    def test_a_rule_is_reachable_by_the_field_it_governs(self):
        from report_ingest.narrative_glossary import conventions_for

        rules = conventions_for("cptCount")
        assert any("0" in row.rule for row in rules)
        # The general null rule governs every field.
        assert conventions_for("projectName")

    def test_the_glossary_names_no_real_firm_or_site(self):
        from report_ingest import narrative_glossary
        from report_ingest.tests.test_log_templates import privacy_offences

        text = open(narrative_glossary.__file__, encoding="utf-8").read()
        assert privacy_offences(text) == []


# ---------------------------------------------------------------------------
# lever 3: retrieval over the whole report
# ---------------------------------------------------------------------------

class TestRetrieval:
    """Passages for the questions whose answers sit in a table."""

    def test_passages_carry_the_page_they_came_off(self, doc):
        rows = retrieval_passages(doc)

        assert rows
        for row in rows:
            assert isinstance(row["page"], int)
            assert 0 <= row["page"] < doc.n_pages
            assert row["passage"].strip()
            assert row["field"] in RETRIEVAL_PHRASES

    def test_the_brief_prints_each_passage_under_its_question(self, doc):
        engine = FakeEngine([{"final": a_reading()}])
        read(doc, engine, front_pages=0, roles=[], retrieve=True)
        brief = brief_of(engine)

        assert "PASSAGES FOUND ELSEWHERE IN THE REPORT" in brief
        assert "siteClass:" in brief
        assert "    page " in brief

    def test_a_citation_on_a_retrieved_page_is_accepted(self, doc):
        # Page 12 is a laboratory sheet: not in the narrative, not in the
        # front matter. An answer cited there is only citable because the
        # retrieval block put that page in front of the reader.
        rows = retrieval_passages(doc)
        outside = [r["page"] for r in rows if r["page"] not in (2, 3, 4)]
        if not outside:
            pytest.skip("this fixture retrieves nothing off the main body")
        page = outside[0]
        reading = a_reading(citations=[ReadCitation(
            field="soilCorrosion", page=page, quote="x")],
            soilCorrosion="yes - resistivity was measured")

        engine = FakeEngine([{"final": reading}])
        result = read(doc, engine, front_pages=0, roles=[], retrieve=True)

        assert "soilCorrosion" in result.citations

    def test_retrieval_off_costs_nothing_and_says_nothing(self, doc):
        engine = FakeEngine([{"final": a_reading()}])
        read(doc, engine, front_pages=0, roles=[], retrieve=False)

        assert "PASSAGES FOUND ELSEWHERE" not in brief_of(engine)


# ---------------------------------------------------------------------------
# lever 4: the five answers the appendix settles
# ---------------------------------------------------------------------------

@dataclass
class _Inv:
    """Just enough of an Investigation for the counter."""

    kind: str
    investigation_id: str = ""


class TestTheDeterministicExplorationAnswers:
    """The logs are a count of holes; the prose is a claim about them."""

    def test_the_counter_reads_the_logs_by_kind(self):
        got = explorations_found([
            _Inv("boring", "B-1"), _Inv("boring", "B-2"),
            _Inv("test_pit", "TP-1"),
        ])

        assert got["boringCount"] == 2
        assert got["testPitCount"] == 1
        assert got["cptCount"] == 0
        assert got["boringDictionary"] == ["B-1", "B-2"]
        assert got["testPitDictionary"] == ["TP-1"]

    def test_no_logs_at_all_yields_nothing_and_the_prose_stands(self, doc):
        assert explorations_found([]) == {}

        engine = FakeEngine([{"final": a_reading()}])
        result = read(doc, engine, front_pages=0, roles=[], investigations=[])

        assert result.general.boringCount == 4

    def test_the_logs_overrule_the_prose_and_the_split_is_recorded(
            self, doc):
        engine = FakeEngine([{"final": a_reading(boringCount=4)}])
        result = read(doc, engine, front_pages=0, roles=[],
                      investigations=[_Inv("boring", f"B-{n}")
                                      for n in range(1, 7)])

        assert result.general.boringCount == 6
        assert result.general.boringDictionary == [f"B-{n}"
                                                   for n in range(1, 7)]
        rows = [u for u in result.unresolved if u["what"] == "boringCount"]
        assert rows and "4" in rows[0]["value"]
        assert "the logs" in rows[0]["why"]
        # The narrative's own number is still on the record beside it.
        assert result.facts.stated_counts["borings"] == 4

    def test_agreement_is_silent(self, doc):
        engine = FakeEngine([{"final": a_reading(boringCount=2)}])
        result = read(doc, engine, front_pages=0, roles=[],
                      investigations=[_Inv("boring", "B-1"),
                                      _Inv("boring", "B-2")])

        assert result.general.boringCount == 2
        assert not [u for u in result.unresolved
                    if u["what"] == "boringCount" and "logs" in u["why"]]

    def test_a_zero_does_not_overrule_a_stated_number(self, doc):
        # No cone logs were read is not the same as no cones were pushed:
        # the labeller may have missed the item, or the soundings may be in
        # another volume. The narrative's number stands and the split is on
        # the record.
        engine = FakeEngine([{"final": a_reading(cptCount=5)}])
        result = read(doc, engine, front_pages=0, roles=[],
                      investigations=[_Inv("boring", "B-1")])

        assert result.natural_hazards is not None
        assert result.general.cptCount == 5
        rows = [u for u in result.unresolved if u["what"] == "cptCount"]
        assert rows and "stands" in rows[0]["why"]

    def test_a_zero_is_stored_where_the_narrative_said_nothing(self, doc):
        engine = FakeEngine([{"final": a_reading(testPitCount=None)}])
        result = read(doc, engine, front_pages=0, roles=[],
                      investigations=[_Inv("boring", "B-1")])

        assert result.general.testPitCount == 0


# ---------------------------------------------------------------------------
# lever 5: the quote gate
# ---------------------------------------------------------------------------

class TestTheQuoteGate:
    """A citation says a page and repeats its words. Both are checkable."""

    def test_an_answer_whose_quote_is_on_the_page_keeps_its_confidence(
            self, doc):
        engine = FakeEngine([{"final": a_reading()}])
        result = read(doc, engine, front_pages=0, roles=[])

        assert result.confidence["siteClass"] == DEFAULT_CONFIDENCE
        assert not [u for u in result.unresolved
                    if u["what"] == "siteClass"]

    def test_an_answer_whose_quote_is_not_on_the_page_is_downgraded(
            self, doc):
        reading = a_reading(citations=[ReadCitation(
            field="siteClass", page=4,
            quote="the site is underlain by glacial till to refusal")])
        engine = FakeEngine([{"final": reading}])
        result = read(doc, engine, front_pages=0, roles=[])

        # The ANSWER is kept -- it may still be right -- and it is flagged.
        assert result.natural_hazards.siteClass == "Site Class D"
        assert result.confidence["siteClass"] == UNVERIFIED_CONFIDENCE
        rows = [u for u in result.unresolved if u["what"] == "siteClass"]
        assert rows and "not on the page it cites" in rows[0]["why"]

    def test_an_answer_with_no_citation_keeps_its_confidence_and_is_flagged(
            self, doc):
        engine = FakeEngine([{"final": a_reading(citations=[])}])
        result = read(doc, engine, front_pages=0, roles=[])

        assert result.confidence["siteClass"] == DEFAULT_CONFIDENCE
        assert [u for u in result.unresolved
                if u["what"] == "siteClass"
                and "no citation" in u["why"]]

    def test_the_confidence_map_covers_every_answered_field(self, doc):
        engine = FakeEngine([{"final": a_reading()}])
        result = read(doc, engine, front_pages=0, roles=[])

        assert set(result.confidence) == set(result.answered)
        assert result.to_dict()["confidence"]["siteClass"] == pytest.approx(
            DEFAULT_CONFIDENCE)


# ---------------------------------------------------------------------------
# lever 6: the lenient scoring view
# ---------------------------------------------------------------------------

class TestLenientScoring:
    """The four free-text fields, judged as a reviewer would judge them."""

    def test_a_reworded_recommendation_is_strict_miss_lenient_hit(self):
        from report_ingest.narrative_scoring import same_value

        hand = ["shallow spread footings bearing on the dense residual soil"]
        got = ["spread footings on dense residual soil"]

        assert not same_value("recommendedFoundations", hand, got)
        assert same_value("recommendedFoundations", hand, got, lenient=True)

    def test_a_shared_number_and_unit_is_enough(self):
        from report_ingest.narrative_scoring import same_value

        # Worded differently enough to fail BOTH the strict ratio and the
        # lenient one; what carries it is the pressure they both print.
        hand = ["3,000 psf for spread footings"]
        got = ["spread footings, 3000 psf net allowable"]

        assert not same_value("bearingCapacity", hand, got)
        assert same_value("bearingCapacity", hand, got, lenient=True)

    def test_a_bare_number_carries_nothing_a_number_with_a_unit_does(self):
        from report_ingest.narrative_scoring import _numbers_with_units

        # Two sentences that share the bare number 3 share nothing; two that
        # share 3000 psf are about the same recommendation.
        assert _numbers_with_units("3 boreholes") == set()
        assert _numbers_with_units("3,000 psf") == {(3000.0, "psf")}
        assert _numbers_with_units(["20 feet", "8 in"]) == {(20.0, "ft"),
                                                            (8.0, "in")}

    def test_the_same_number_in_a_different_unit_is_a_different_value(self):
        from report_ingest.narrative_scoring import _numbers_with_units

        assert not (_numbers_with_units("3000 psf allowable")
                    & _numbers_with_units("3000 kPa allowable"))

    def test_two_different_recommendations_stay_different(self):
        from report_ingest.narrative_scoring import same_value

        hand = ["mat foundations founded in the weathered rock"]
        got = ["3000 psf net allowable"]

        assert not same_value("bearingCapacity", hand, got)
        assert not same_value("bearingCapacity", hand, got, lenient=True)

    def test_feet_and_ft_are_one_unit(self):
        from report_ingest.narrative_scoring import same_value

        assert same_value("strata",
                          "residual soil to 20 feet then weathered rock",
                          "weathered rock below 20 ft", lenient=True)

    def test_lenience_does_not_reach_an_enumeration_a_count_or_an_id_list(
            self):
        from report_ingest.narrative_scoring import same_value

        assert not same_value("boringCount", 4, 5, lenient=True)
        assert not same_value("documentType", "geotechnical report",
                              "environmental report", lenient=True)
        assert not same_value("boringDictionary", ["B-1"], ["B-12"],
                              lenient=True)

    def test_the_two_views_are_scored_side_by_side(self):
        from report_ingest.model import GeneralFacts, NaturalHazardFacts
        from report_ingest.narrative_scoring import score_narrative

        truth = {"id": "X", "general": {
            "recommendedFoundations": [
                "shallow spread footings bearing on dense residual soil"],
            "boringCount": 4}, "natural_hazards": {}}
        general = GeneralFacts(
            recommendedFoundations=["spread footings on dense residual soil"],
            boringCount=4)
        score = score_narrative(truth, general, NaturalHazardFacts(),
                                report="X")

        assert score.recall.found == 1 and score.recall.total == 2
        assert score.lenient_recall.found == 2
        assert score.lenient_recall.total == 2
        row = next(r for r in score.fields
                   if r.field == "recommendedFoundations")
        assert row.ok is False and row.lenient_ok is True

    def test_a_miss_on_a_field_a_house_rule_governs_is_a_convention_miss(
            self):
        from report_ingest.narrative_scoring import dominant_miss, miss_kind

        assert miss_kind("cptCount", "missed") == "convention"
        assert miss_kind("earthHazardsExposed", "invented") == "convention"
        assert miss_kind("projectName", "wrong") == "wrong"
        assert miss_kind("projectName", "right") == ""
        assert dominant_miss(["wrong", "wrong", "missed"]) == "wrong"
        assert dominant_miss([]) == ""
        # A tie goes to the kind that tells the reader what to do about it.
        assert dominant_miss(["convention", "missed"]) == "convention"
