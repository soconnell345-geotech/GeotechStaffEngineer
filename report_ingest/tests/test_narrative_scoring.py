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
    FIELD_KINDS, LIST_HIT_JACCARD, is_piece, kind_of, rescore_saved,
    same_value, score_narrative,
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

    def test_a_hole_list_ignores_leading_zeros_but_not_tens(self):
        assert same_value("boringDictionary", ["B-1", "B-2"],
                          ["B-01", "B-02"])
        assert not same_value("boringDictionary", ["B-1", "B-2"],
                              ["B-10", "B-20"])

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


class TestTheTruthFilesOwnRules:
    """``_alternates`` and ``_skip``: the hand's two ways of being fair."""

    def test_an_alternate_answer_scores_as_correct(self):
        hand = truth(general={"boringCount": 14})
        hand["_alternates"] = {"boringCount": [18]}

        assert score(hand, GeneralFacts(boringCount=18)).recall.found == 1
        assert score(hand, GeneralFacts(boringCount=14)).recall.found == 1
        assert score(hand, GeneralFacts(boringCount=9)).recall.found == 0

    def test_a_null_among_the_alternates_accepts_not_stated(self):
        hand = truth(general={"tableCount": 6})
        hand["_alternates"] = {"tableCount": [None]}
        result = score(hand)

        assert result.recall.found == 1
        assert [row.verdict for row in result.fields
                if row.field == "tableCount"] == ["missed"]

    def test_an_alternate_list_is_a_whole_alternative_answer(self):
        hand = truth(natural_hazards={"earthHazardsExposed": ["flooding"]})
        hand["_alternates"] = {"earthHazardsExposed": [
            ["flooding", "landslide"], ["flooding", "volcanos"]]}

        exact = score(hand, hazards=NaturalHazardFacts(
            earthHazardsExposed=["flooding", "landslide"]))
        wrong = score(hand, hazards=NaturalHazardFacts(
            earthHazardsExposed=["karst"]))

        assert exact.recall.found == 1
        assert wrong.recall.found == 0

    def test_an_empty_list_among_the_alternates_accepts_silence(self):
        hand = truth(natural_hazards={
            "earthHazardsExposed": ["liquefaction-induced settlement"]})
        hand["_alternates"] = {"earthHazardsExposed": [[]]}

        assert score(hand).recall.found == 1

    def test_the_items_of_the_best_matching_list_are_the_ones_pooled(self):
        hand = truth(general={"boringDictionary": ["B-1"]})
        hand["_alternates"] = {"boringDictionary": [["B-1", "B-2", "B-3"]]}
        result = score(hand, GeneralFacts(
            boringDictionary=["B-1", "B-2", "B-3"]))

        assert result.recall.found == 1
        assert result.list_item_recall.found == 3
        assert result.list_item_recall.total == 3

    def test_a_skipped_field_is_in_no_count_at_all(self):
        hand = truth(general={"tableCount": 6, "boringCount": 4})
        hand["_skip"] = ["tableCount"]
        result = score(hand, GeneralFacts(tableCount=99, boringCount=4))

        assert result.skipped == ["tableCount"]
        assert result.recall.total == 1            # boringCount alone
        assert result.precision.total == 1
        assert result.kind("int").total == 1
        assert [row.verdict for row in result.fields
                if row.field == "tableCount"] == ["skipped"]

    def test_a_skipped_field_is_not_counted_in_agreement_either(self):
        hand = truth(general={"tableCount": 6})
        hand["_skip"] = ["tableCount"]
        with_skip = score(hand)
        without = score(truth(general={"tableCount": 6}))

        assert with_skip.agreement.total == without.agreement.total - 1

    def test_a_skipped_summary_is_not_scored_for_presence(self):
        hand = truth(general={"quickSummary": "A due diligence study."})
        hand["_skip"] = ["quickSummary"]

        assert score(hand).summaries_present.total == 0


class TestTheTwoKindsOfList:

    def test_an_identifier_survives_its_punctuation(self):
        assert same_value("boringDictionary", ["B-1", "B-2"], ["B1", "b 2"])

    def test_two_identifiers_that_merely_look_alike_are_two(self):
        # B-1 is a substring of B-12: a fuzzy matcher would call them one.
        result = score(truth(general={"boringDictionary": ["B-1", "B-2"]}),
                       GeneralFacts(boringDictionary=["B-12", "B-21"]))

        assert result.recall.found == 0
        assert result.list_item_recall.found == 0

    def test_a_prose_item_matches_on_resemblance(self):
        # The hand's bearingCapacity runs to fifteen words an item; exact
        # matching would score transcription rather than reading.
        assert same_value(
            "bearingCapacity",
            ["An allowable bearing pressure of 3,000 psf is recommended for "
             "spread footings bearing on the dense residual soil"],
            ["an allowable bearing pressure of 3,000 psf is recommended for "
             "spread footings"])

    def test_one_truth_item_cannot_be_matched_twice(self):
        result = score(
            truth(general={"recommendedFoundations": ["spread footings"]}),
            GeneralFacts(recommendedFoundations=["spread footings",
                                                 "spread footings on fill"]))

        # Two predictions, one truth item: one hit, so the items' precision
        # is one of two rather than two of two.
        assert result.list_items.found == 1
        assert result.list_items.total == 2


#: Invented hand items with numbers in them, for the piece tests.
ALLOWABLE = "3,700 psf allowable for zephyr footings at least 2 feet wide"
METRIC = "zephyr footings 6.5 m wide, 185 kPa"


class TestThePieces:
    """BACKLOG RI-13: options listed one per item, in the LENIENT view only.

    Every value here is invented. The strict view must give what one-to-one
    pairing gives (the scorer before RI-13); the lenient view leaves a piece
    of a paired truth item out of the overlap.
    """

    HAND = ["zephyr footings on blue gravel",
            "deep option of quill piles or orbit piers"]
    SPLIT = ["zephyr footings on blue gravel", "deep option", "quill piles",
             "orbit piers"]

    def test_options_listed_one_by_one_are_right_in_the_lenient_view(self):
        result = score(truth(general={"recommendedFoundations": self.HAND}),
                       GeneralFacts(recommendedFoundations=self.SPLIT))

        (row,) = [r for r in result.fields
                  if r.field == "recommendedFoundations"]
        # Strict: two hits and two extras, 2/4, under the bar, as before.
        assert row.verdict == "wrong" and row.jaccard == 0.5
        assert result.recall.found == 0
        # Lenient: the two pile types are pieces of the second item.
        assert row.lenient_ok
        assert result.lenient_recall.found == 1

    def test_the_pooled_item_counts_are_strict_and_unchanged(self):
        result = score(truth(general={"recommendedFoundations": self.HAND}),
                       GeneralFacts(recommendedFoundations=self.SPLIT))

        assert result.list_items.found == 2
        assert result.list_items.total == 4
        assert result.list_item_recall.found == 2
        assert result.list_item_recall.total == 2

    # (case, field, hand, answer, strict overlap as before RI-13, lenient)
    REVIEW_CASES = [
        ("a word inside a longer word", "recommendedFoundations",
         ["zephyr footings on weathered sandstone"],
         ["zephyr footings on weathered sandstone", "sand"], 0.5, False),
        ("a word ending a longer word", "recommendedFoundations",
         ["orbit slab on blue backfill"],
         ["orbit slab on blue backfill", "fill"], 0.5, False),
        ("a different number", "bearingCapacity",
         ["1,234 psf allowable for zephyr footings"],
         ["1,234 psf allowable for zephyr footings", "1,834 psf"],
         0.5, False),
        ("a stray letter", "structureList", ["kite tower"],
         ["kite tower", "k"], 0.5, False),
        # Numbers are whole words, and a piece is an unbroken run: none of
        # these is part of the item, though each is made of its loose words
        # once "3,700" falls apart into "3" and "700" and "6.5" into "6" and
        # "5".
        ("another load", "bearingCapacity", [ALLOWABLE],
         [ALLOWABLE, "2,700 psf"], 0.5, False),
        ("another width", "bearingCapacity", [ALLOWABLE],
         [ALLOWABLE, "3 feet"], 0.5, False),
        ("loose words of the item", "bearingCapacity", [ALLOWABLE],
         [ALLOWABLE, "wide footings 2,700"], 0.5, False),
        ("the tail of a number", "bearingCapacity", [ALLOWABLE],
         [ALLOWABLE, "700 psf"], 0.5, False),
        ("the tail of a decimal", "bearingCapacity", [METRIC],
         [METRIC, "5 kPa"], 0.5, False),
        ("the head of a decimal", "bearingCapacity", [METRIC],
         [METRIC, "6 kPa"], 0.5, False),
        ("a decimal width cut short", "bearingCapacity", [METRIC],
         [METRIC, "5 m"], 0.5, False),
        ("one item cut into single words", "recommendedFoundations",
         ["deep option of quill piles or orbit piers"],
         ["deep", "option", "of", "quill", "piles", "or", "orbit", "piers"],
         0.125, False),
        # What words cannot see. Each is right in the LENIENT view only,
        # which is why pieces never count in the strict one.
        ("an option the report rules out", "recommendedFoundations",
         ["orbit piers, since quill piles are not recommended"],
         ["orbit piers", "quill piles"], 0.5, True),
        ("the same item twice", "structureList", ["kite tower"],
         ["kite tower", "Kite Tower."], 0.5, True),
    ]

    @pytest.mark.parametrize(
        "field_name, hand, got, strict_overlap, lenient",
        [case[1:] for case in REVIEW_CASES],
        ids=[case[0] for case in REVIEW_CASES])
    def test_the_strict_view_is_unchanged_and_the_lenient_view_is_pinned(
            self, field_name, hand, got, strict_overlap, lenient):
        from report_ingest.narrative_scoring import _list_overlap

        assert not same_value(field_name, hand, got)
        jaccard, hits, given, wanted, pieces = _list_overlap(
            field_name, hand, got)
        assert jaccard == strict_overlap and pieces == 0
        assert same_value(field_name, hand, got, lenient=True) is lenient

    # (hand item, a real second value, a garbled piece of the hand item)
    BETTER_WORSE = [
        ("1,234 psf allowable for zephyr footings",
         "2,345 psf allowable for orbit mats", "1,834 psf"),
        (ALLOWABLE, "9 ksf on zephyr rock", "2,700 psf"),
        (ALLOWABLE, "9 ksf on zephyr rock", "3 feet"),
        (ALLOWABLE, "9 ksf on zephyr rock", "wide footings 2,700"),
        (METRIC, "9 ksf on zephyr rock", "5 kPa"),
        (METRIC, "9 ksf on zephyr rock", "6 kPa"),
    ]

    @pytest.mark.parametrize("item, real, garbled", BETTER_WORSE)
    def test_a_worse_answer_does_not_beat_a_better_one(self, item, real,
                                                       garbled):
        from report_ingest.narrative_scoring import _list_overlap

        hand = [item]
        better, worse = hand + [real], hand + [garbled]
        for lenient in (False, True):
            assert same_value("bearingCapacity", hand, worse,
                              lenient=lenient) \
                <= same_value("bearingCapacity", hand, better,
                              lenient=lenient)
            assert _list_overlap("bearingCapacity", hand, worse, lenient)[0] \
                <= _list_overlap("bearingCapacity", hand, better, lenient)[0]
        assert not same_value("bearingCapacity", hand, worse, lenient=True)

    def test_a_true_piece_of_a_number_item_still_counts(self):
        # The run of words as printed, numbers whole, is still a piece.
        for piece in ("3,700 psf allowable", "3700 psf",
                      "at least 2 feet wide"):
            assert same_value("bearingCapacity", [ALLOWABLE],
                              [ALLOWABLE, piece], lenient=True), piece
        assert same_value("bearingCapacity", [METRIC], [METRIC, "185 kPa"],
                          lenient=True)

    def test_an_item_takes_no_more_pieces_than_the_options_it_lists(self):
        from report_ingest.narrative_scoring import _list_overlap, _options

        assert _options("deep option of quill piles or orbit piers") == 2
        assert _options("deep option: quill piles or orbit piers") == 3
        assert _options(ALLOWABLE) == 1          # a comma in a number
        assert _options(METRIC) == 2
        # Two options, so two pieces at most: the split of the pile types
        # is right, the eight single words are not.
        assert _list_overlap("recommendedFoundations", self.HAND,
                             self.SPLIT, True) == (1.0, 2, 4, 2, 2)
        words = self.HAND[1].split()
        assert _list_overlap("recommendedFoundations", self.HAND[1:],
                             words, True)[4] == 2

    def test_an_item_that_is_no_piece_still_counts_against(self):
        for lenient in (False, True):
            assert not same_value("structureList", ["kite tower"],
                                  ["kite tower", "moat bridge"],
                                  lenient=lenient)
        # Not a lenient field: no pieces in either view.
        assert not same_value("earthHazardsExposed", ["landslide"],
                              ["landslide", "landslide"], lenient=True)

    def test_a_piece_is_whole_words_of_the_item(self):
        assert is_piece("orbit piers", self.HAND[1])
        assert is_piece("Quill Piles.", self.HAND[1])
        assert not is_piece("sand", "weathered sandstone")
        assert not is_piece("fill", "blue backfill")
        assert not is_piece("1,834 psf", "1,234 psf allowable")

    def test_a_piece_is_an_unbroken_run_in_the_items_order(self):
        assert is_piece("quill piles or orbit", self.HAND[1])
        assert not is_piece("quill orbit", self.HAND[1])
        assert not is_piece("piers orbit", self.HAND[1])

    def test_a_number_is_one_word(self):
        assert is_piece("3,700 psf", ALLOWABLE)
        assert is_piece("3700 psf", ALLOWABLE)
        assert is_piece("6.5 m wide", METRIC)
        assert not is_piece("700 psf", ALLOWABLE)
        assert not is_piece("3 feet", ALLOWABLE)
        assert not is_piece("5 m", METRIC)
        assert not is_piece("6 m", METRIC)

    def test_a_longer_item_is_a_second_answer_not_a_piece(self):
        assert not is_piece("spread footings on fill", "spread footings")

    def test_a_shared_capitalised_word_does_not_make_a_piece(self):
        # The string rule's shared-proper-noun test passes on any capitalised
        # first word; a piece has to be made of the item's own words.
        assert not is_piece("Zephyr wells", "Zephyr footings on blue gravel")

    def test_an_identifier_list_has_no_pieces(self):
        # B-1 is no piece of B-12: two holes, an overlap of one in two.
        for lenient in (False, True):
            assert not same_value("boringDictionary", ["B-12"],
                                  ["B-12", "B-1"], lenient=lenient)


class TestRescoreSaved:
    """A saved run file scored again with the current rules, no model."""

    def _saved(self, hand, general=None, hazards=None):
        result = score(hand, general, hazards)
        result.pages = [3, 4]
        result.model_calls = 2
        return {"id": "R00", "score": result.to_dict()}

    def test_the_run_file_keeps_a_long_answer_whole(self):
        items = [f"made-up recommendation number {n} with a long tail of "
                 f"words" for n in range(6)]
        blob = self._saved(truth(general={"bearingCapacity": items[:1]}),
                           GeneralFacts(bearingCapacity=items))
        (row,) = [r for r in blob["score"]["fields"]
                  if r["field"] == "bearingCapacity"]

        assert row["got"].endswith("…") and len(row["got"]) == 120
        assert row["answer"] == items

    def test_a_rescore_gives_back_the_same_score(self):
        hand = truth(general={"boringCount": 4,
                              "recommendedFoundations": ["orbit mat"],
                              "quickSummary": "A study of a made-up site."})
        blob = self._saved(hand, GeneralFacts(
            boringCount=4, recommendedFoundations=["orbit mat"],
            quickSummary="A made-up summary."))

        again = rescore_saved(hand, blob)

        assert again.to_dict()["recall"] == blob["score"]["recall"]
        assert again.to_dict()["precision"] == blob["score"]["precision"]
        assert again.summaries_present.found == 1
        assert again.pages == [3, 4] and again.model_calls == 2

    def test_a_rescore_applies_the_current_rules(self):
        hand = truth(general={"recommendedFoundations":
                              TestThePieces.HAND})
        blob = self._saved(hand, GeneralFacts(
            recommendedFoundations=TestThePieces.SPLIT))
        for row in blob["score"]["fields"]:          # as an older scorer
            if row["field"] == "recommendedFoundations":   # left it
                row.update(lenient_ok=False)

        again = rescore_saved(hand, blob)

        assert [(r.verdict, r.lenient_ok) for r in again.fields
                if r.field == "recommendedFoundations"] == [("wrong", True)]

    def test_an_old_run_file_with_cut_answers_is_not_rescored(self):
        hand = truth(general={"boringCount": 4})
        blob = self._saved(hand, GeneralFacts(boringCount=4))
        for row in blob["score"]["fields"]:
            del row["answer"]

        assert rescore_saved(hand, blob) is None
        assert rescore_saved(hand, {"score": {"fields": []}}) is None

    def test_a_saved_error_is_carried_over(self):
        hand = truth(general={"boringCount": 4})
        saved = {"id": "R00", "score": {
            "report": "R00", "fields": [], "pages": [],
            "model_calls": 1, "error": "no narrative work item in this "
                                       "document"}}

        again = rescore_saved(hand, saved)

        assert again.error == "no narrative work item in this document"
        assert again.report == "R00" and again.model_calls == 1
        assert again.recall.total == 0


class TestTheVerdictQuestions:

    def test_the_verdict_is_the_answer_and_the_reason_is_not_scored(self):
        assert same_value(
            "geophysicalTestingMention",
            "yes - six seismic refraction and MASW lines across the site",
            "yes - a shear wave velocity survey was carried out")

    def test_a_different_verdict_is_a_different_answer(self):
        assert not same_value("soilCorrosion",
                              "yes - highly corrosive, chloride to 9,151 ppm",
                              "no - not corrosive")

    def test_mixed_is_its_own_verdict(self):
        assert same_value("soilCorrosion", "mixed - non-corrosive to pipe",
                          "mixed - corrosive to steel only")
        assert not same_value("soilCorrosion", "mixed - non-corrosive to pipe",
                              "yes - corrosive")

    def test_an_answer_with_no_verdict_matches_nothing(self):
        assert not same_value("siteResponseMention",
                              "a site response analysis was performed",
                              "yes - a one-dimensional analysis")
