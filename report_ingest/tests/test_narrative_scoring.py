"""The narrative scorer: three numbers, and what each kind of field forgives.

The rules under test are the ones that decide whether a run looks good: that a
report saying nothing and a reader saying nothing agree but earn no recall,
that a firm's name survives an ampersand and an "Inc.", that a site class
survives its own words, and that a summary is judged on presence and length
and never on its prose.
"""

from __future__ import annotations

import pytest

from report_ingest.model import GeneralFacts, NaturalHazardFacts
from report_ingest.narrative_scoring import (
    FIELD_KINDS, LIST_HIT_JACCARD, kind_of, same_value, score_narrative,
)


def truth(**sections):
    return {"general": sections.get("general", {}),
            "natural_hazards": sections.get("natural_hazards", {}),
            "notes": ""}


def score(hand, general=None, hazards=None):
    return score_narrative(hand, general or GeneralFacts(),
                           hazards or NaturalHazardFacts(), report="R00")


class TestTheKinds:

    def test_every_field_of_both_schemas_has_a_kind(self):
        from report_ingest.model import GENERAL_FIELDS, NATURAL_HAZARD_FIELDS

        for name in GENERAL_FIELDS + NATURAL_HAZARD_FIELDS:
            assert kind_of(name) in ("enum", "int", "string", "list",
                                     "summary")
        assert len(FIELD_KINDS) == len(GENERAL_FIELDS) + \
            len(NATURAL_HAZARD_FIELDS)

    @pytest.mark.parametrize("name,kind", [
        ("documentType", "enum"), ("liquefactionPotential", "enum"),
        ("boringCount", "int"), ("figureCount", "int"),
        ("geotechnicalEngineerFirm", "string"), ("siteClass", "string"),
        ("boringDictionary", "list"), ("earthHazardsExposed", "list"),
        ("quickSummary", "summary"), ("naturalHazardSummary", "summary"),
    ])
    def test_the_kinds_are_what_they_should_be(self, name, kind):
        assert kind_of(name) == kind


class TestComparingOneAnswer:

    def test_an_enumeration_is_exact_after_folding(self):
        assert same_value("documentType", "geotechnical report",
                          "Geotechnical Report")
        assert not same_value("liquefactionPotential", "low", "moderate")

    def test_a_count_is_a_count(self):
        assert same_value("boringCount", 4, 4)
        assert not same_value("boringCount", 4, 5)

    def test_a_firm_survives_its_punctuation(self):
        assert same_value("geotechnicalEngineerFirm",
                          "Soil & Rock Consulting Engineers",
                          "Soil and Rock Consulting Engineers, Inc.")

    def test_a_shared_proper_noun_is_enough(self):
        assert same_value("projectName", "Rosewood Terrace Development",
                          "Rosewood Terrace")

    def test_two_different_firms_do_not_match(self):
        assert not same_value("geotechnicalEngineerFirm",
                              "Soil & Rock Consulting Engineers",
                              "Atlantic Geotechnical Group")

    def test_a_site_class_survives_its_own_words(self):
        assert same_value("siteClass", "D", "Site Class D")
        assert not same_value("siteClass", "D", "Site Class C")

    def test_a_date_survives_its_format(self):
        assert same_value("reportDate", "2026-03-14", "14 March 2026")
        assert not same_value("reportDate", "2026-03-14", "15 March 2026")

    def test_an_asce_edition_survives_its_wording(self):
        assert same_value("asceSevenVersion", "ASCE 7-16",
                          "ASCE/SEI 7-16 (2016)")

    def test_a_list_is_set_overlap(self):
        assert same_value("boringDictionary", ["B-1", "B-2"],
                          ["B-1", "B-2", "B-3"])       # 2/3 overlap
        assert not same_value("boringDictionary", ["B-1", "B-2"],
                              ["B-9", "B-8"])
        assert LIST_HIT_JACCARD <= 0.67

    def test_null_against_anything_is_a_miss_and_null_against_null_is_not(
            self):
        assert same_value("boringCount", None, None)
        assert not same_value("boringCount", 4, None)
        assert not same_value("boringCount", None, 4)


class TestTheThreeNumbers:

    def test_a_reader_that_says_nothing_agrees_but_recalls_nothing(self):
        result = score(truth(general={"boringCount": 4,
                                      "projectName": "Rosewood"}))

        assert result.recall.found == 0 and result.recall.total == 2
        assert result.precision.total == 0        # it gave no answers
        assert result.agreement.found > 30        # and agreed about the rest
        assert [row.verdict for row in result.fields
                if row.field == "boringCount"] == ["missed"]

    def test_a_reader_that_invents_loses_precision_not_recall(self):
        result = score(truth(general={"boringCount": 4}),
                       GeneralFacts(boringCount=4, postName="Somewhere"))

        assert result.recall.found == 1 and result.recall.total == 1
        assert result.precision.found == 1 and result.precision.total == 2
        assert any(row.verdict == "invented" and row.field == "postName"
                   for row in result.fields)

    def test_a_wrong_answer_costs_both(self):
        result = score(truth(general={"boringCount": 4}),
                       GeneralFacts(boringCount=6))

        assert result.recall.found == 0 and result.precision.found == 0
        assert any(row.verdict == "wrong" for row in result.fields)

    def test_the_kinds_are_scored_separately(self):
        result = score(
            truth(general={"boringCount": 4,
                           "documentType": "geotechnical report",
                           "boringDictionary": ["B-1", "B-2"]}),
            GeneralFacts(boringCount=4, documentType="geotechnical report",
                         boringDictionary=["B-1"]))

        assert result.kind("int").found == 1
        assert result.kind("enum").found == 1
        assert result.kind("list").found == 0      # 0.5 overlap, under the bar

    def test_the_items_inside_the_lists_are_pooled(self):
        result = score(
            truth(general={"boringDictionary": ["B-1", "B-2", "B-3"]}),
            GeneralFacts(boringDictionary=["B-1", "B-2", "B-9"]))

        assert result.list_items.found == 2 and result.list_items.total == 3
        assert result.list_item_recall.found == 2
        assert result.list_item_recall.total == 3


class TestTheSummaries:

    def test_presence_is_scored_where_the_hand_wrote_one(self):
        hand = truth(general={"quickSummary": "A due diligence study."})
        written = score(hand, GeneralFacts(quickSummary="A study of a site."))
        silent = score(hand)

        assert written.summaries_present.found == 1
        assert silent.summaries_present.found == 0
        assert silent.summaries_present.total == 1

    def test_the_words_themselves_are_never_compared(self):
        hand = truth(general={"quickSummary": "A due diligence study."})
        result = score(hand, GeneralFacts(
            quickSummary="Something else entirely, but written."))

        assert result.summaries_present.found == 1

    def test_the_word_limit_is_scored(self):
        long_one = " ".join(["word"] * 130)
        result = score(truth(general={"quickSummary": "x"}),
                       GeneralFacts(quickSummary=long_one))

        assert result.summaries_within_limit.found == 0
        assert result.summaries_within_limit.total == 1
        assert any(row.field == "quickSummary" and row.verdict == "too_long"
                   for row in result.fields)

    def test_a_summary_is_not_in_the_recall_or_precision_counts(self):
        result = score(truth(general={"quickSummary": "x"}),
                       GeneralFacts(quickSummary="y"))

        assert result.recall.total == 0
        assert result.precision.total == 0


class TestOneWholeReport:

    def test_the_reader_runs_and_is_scored(self, tmp_path):
        pytest.importorskip("planlens.document.roles")
        from planlens.document import open_document
        from report_ingest.narrative_scoring import (
            narrative_pages_of, score_one_report,
        )
        from report_ingest.tests.fake_engine import FakeEngine
        from report_ingest.tests.narrative_fixtures import (
            build_narrative_report,
        )
        from report_ingest.tests.test_graph import narrative_turn

        doc = open_document(build_narrative_report().pdf, name="SYN")
        try:
            pages, body = narrative_pages_of(doc)
            assert pages == [2, 3, 4]
            assert 5 in body                       # the figure page
            hand = truth(
                general={"boringCount": 4, "testPitCount": 3,
                         "geotechnicalEngineerFirm":
                             "Soil and Rock Consulting Engineers"},
                natural_hazards={"siteClass": "D",
                                 "liquefactionPotential": "low"})
            result = score_one_report(hand, doc,
                                      FakeEngine([narrative_turn()]),
                                      report="SYN")
        finally:
            doc.close()

        assert result.error is None
        assert result.recall.found == 5 and result.recall.total == 5
        assert result.model_calls == 1
        assert result.pages == [2, 3, 4]

    def test_a_document_with_no_narrative_is_an_error_not_a_crash(self):
        pytest.importorskip("planlens.document.roles")
        from report_ingest.narrative_scoring import score_one_report

        class NoItems:
            pass

        result = score_one_report({"general": {}}, _EmptyDoc(), None,
                                  report="R00")
        assert "no narrative" in (result.error or "")


class _EmptyDoc:
    """A document planlens finds no work items in."""

    n_pages = 0

    def page_map(self):
        return []

    def page(self, index, **kwargs):
        raise IndexError(index)
