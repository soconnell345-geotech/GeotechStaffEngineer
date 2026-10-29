"""The two checks on a narrative answer once it is read (BACKLOG RI-178,
RI-179): repeated list items dropped, and a summary over its limit sent
back once to be shortened. Every value here is made up.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from report_ingest.narrative_checks import (
    drop_repeats, over_limit, shorten_summaries,
)
from report_ingest.narrative_reader import NarrativeReading, SummaryReading
from report_ingest.tests.fake_engine import FakeEngine


def _words(n: int, word: str = "gravel") -> str:
    return " ".join([word] * n)


# ---------------------------------------------------------------------------
# repeated list items
# ---------------------------------------------------------------------------

class TestDropRepeats:

    def test_the_same_item_folded_is_dropped_and_the_first_kept(self):
        values = {"recommendedFoundations": [
            "Spread footings on fill", "spread footings, on fill",
            "driven piles"]}
        out, rows = drop_repeats(values)

        assert out["recommendedFoundations"] == ["Spread footings on fill",
                                                 "driven piles"]
        assert rows == [{
            "what": "recommendedFoundations",
            "why": "repeats an item: the same as 'Spread footings on fill'; "
                   "dropped",
            "value": "spread footings, on fill"}]
        # The answer given is not changed in place.
        assert len(values["recommendedFoundations"]) == 3

    def test_a_short_name_that_is_another_item_s_abbreviation_goes(self):
        out, rows = drop_repeats({"structureList": [
            "QRS", "Made-up Pump House (MPH)", "Quarry Rest Shelter (QRS)",
            "MPH", "Store"]})

        assert out["structureList"] == ["Made-up Pump House (MPH)",
                                        "Quarry Rest Shelter (QRS)", "Store"]
        assert [row["value"] for row in rows] == ["QRS", "MPH"]

    def test_a_bearing_sentence_with_the_same_pressures_goes(self):
        out, rows = drop_repeats({"bearingCapacity": [
            "An allowable bearing pressure of 2,750 psf for footings.",
            "Footings: allowable bearing pressure 2750 psf (summary).",
            "A pressure of 2,750 psf, increased by 150 psf per foot of "
            "embedment.",
            "Walls may use 95 kPa."]})

        # The third gives a second pressure and is a different item.
        assert len(out["bearingCapacity"]) == 3
        assert rows[0]["value"].startswith("Footings: allowable")

    def test_a_sentence_with_no_pressure_is_never_a_repeat_by_pressure(self):
        items = ["Footings bear on the dense sand.",
                 "Mats bear on the stiff clay."]
        out, rows = drop_repeats({"bearingCapacity": items})

        assert out["bearingCapacity"] == items and rows == []

    def test_one_hole_spelled_two_ways_is_listed_once(self):
        out, rows = drop_repeats({"boringDictionary": [
            "QB1", "QB-2", "QB-1", "QB-02", "QB-10"]})

        assert out["boringDictionary"] == ["QB1", "QB-2", "QB-10"]
        assert len(rows) == 2

    def test_a_hole_with_text_after_it_goes_when_the_hole_is_listed(self):
        out, rows = drop_repeats({"testPitDictionary": [
            "QP-3 (offset 10 ft east)", "QP-3", "QP-4",
            "QP-4/QP-5", "QP-6 (abandoned)"]})

        # QP-4/QP-5 names another hole, and QP-6 is not listed on its own.
        assert out["testPitDictionary"] == ["QP-3", "QP-4", "QP-4/QP-5",
                                            "QP-6 (abandoned)"]
        assert [row["value"] for row in rows] == ["QP-3 (offset 10 ft east)"]

    def test_other_fields_and_short_lists_are_left_alone(self):
        values = {"structureList": ["Made-up Hall"], "projectName": "X",
                  "earthHazardsExposed": ["flooding", "flooding"]}
        out, rows = drop_repeats(values)

        assert out == values and rows == []

    def test_the_builder_drops_them_and_recounts_the_structures(self):
        from report_ingest.narrative_reader import _Builder

        builder = _Builder([0, 1])
        builder.add(NarrativeReading(
            structureList=["Made-up Hall (MH)", "MH", "Made-up Annex"],
            structureCount=3))
        general, _hazards, _facts = builder.build([0, 1], {}, None)

        assert general.structureList == ["Made-up Hall (MH)",
                                         "Made-up Annex"]
        assert general.structureCount == 2
        whats = [row["what"] for row in builder.unresolved]
        assert "structureList" in whats and "structureCount" in whats


# ---------------------------------------------------------------------------
# a summary over its limit
# ---------------------------------------------------------------------------

def _sections(**summaries):
    general = SimpleNamespace(quickSummary=None, testingProgramSummary=None)
    hazards = SimpleNamespace(naturalHazardSummary=None,
                              seismicParameterSummary=None)
    for name, text in summaries.items():
        setattr(general if hasattr(general, name) else hazards, name, text)
    return general, hazards


class TestShortenSummaries:

    def test_nothing_over_its_limit_makes_no_call(self):
        general, hazards = _sections(quickSummary=_words(100))
        engine = FakeEngine([])

        assert shorten_summaries(general, hazards, engine, []) == 0
        assert engine.calls == []

    def test_one_call_shortens_every_summary_over_its_limit(self):
        general, hazards = _sections(quickSummary=_words(104),
                                     naturalHazardSummary=_words(201, "silt"),
                                     testingProgramSummary=_words(90))
        unresolved = [{"what": "quickSummary",
                       "why": "104 words, past the 100-word limit; kept as "
                              "written"}]
        charged = []
        engine = FakeEngine([{"final": SummaryReading(
            quickSummary=_words(95), naturalHazardSummary=_words(180))}])

        made = shorten_summaries(general, hazards, engine, unresolved,
                                 charged.append)

        assert made == 1 and len(engine.calls) == 1 and len(charged) == 1
        assert engine.calls[0]["output_format"] is SummaryReading
        sent = engine.calls[0]["messages"][0]["content"][0]["text"]
        assert "quickSummary (104 words; its limit is 100)" in sent
        assert "naturalHazardSummary (201 words; its limit is 200)" in sent
        assert "testingProgramSummary" not in sent
        assert general.quickSummary == _words(95)
        assert hazards.naturalHazardSummary == _words(180)
        assert general.testingProgramSummary == _words(90)
        # The summary's own row says what became of it; the other gets one.
        assert unresolved[0]["why"] == ("104 words, past the 100-word limit; "
                                        "shortened once, to 95 words")
        assert unresolved[1] == {
            "what": "naturalHazardSummary",
            "why": "201 words, past the 200-word limit; shortened once, to "
                   "180 words"}
        assert over_limit(general, hazards) == {}

    def test_a_shortening_still_over_the_limit_is_not_taken(self):
        general, hazards = _sections(quickSummary=_words(110))
        unresolved = []
        engine = FakeEngine([{"final": SummaryReading(
            quickSummary=_words(103))}])

        shorten_summaries(general, hazards, engine, unresolved)

        assert general.quickSummary == _words(110)
        assert "did not bring it inside; kept as written" in \
            unresolved[0]["why"]

    def test_a_call_that_fails_costs_nothing_but_the_shortening(self):
        from report_ingest.engine import CallCeilingReached

        general, hazards = _sections(quickSummary=_words(110))
        unresolved = []
        engine = FakeEngine([{"raise": CallCeilingReached("made-up cap")}])

        assert shorten_summaries(general, hazards, engine, unresolved) == 0
        assert general.quickSummary == _words(110)
        assert "CallCeilingReached" in unresolved[0]["why"]


class TestTodaysReaderShortens:

    @pytest.fixture(scope="class")
    def gt(self):
        pytest.importorskip("planlens.document.roles")
        from report_ingest.tests.narrative_fixtures import (
            build_narrative_report,
        )
        return build_narrative_report()

    @pytest.fixture()
    def doc(self, gt):
        from planlens.document import open_document
        document = open_document(gt.pdf, name="SYN")
        try:
            yield document
        finally:
            document.close()

    def test_today_s_reader_sends_an_over_long_summary_back_once(self, doc):
        from report_ingest.narrative_reader import read_narrative
        from report_ingest.tests.narrative_fixtures import NARRATIVE_PAGES

        engine = FakeEngine([
            {"final": NarrativeReading(quickSummary=_words(106))},
            {"final": SummaryReading(quickSummary=_words(60))}])
        result = read_narrative(doc, NARRATIVE_PAGES, engine, front_pages=0,
                                roles=[], retrieve=False)

        assert result.general.quickSummary == _words(60)
        assert result.model_calls == 2 and result.cost["calls"] == 2
        assert any(row["what"] == "quickSummary" and
                   row["why"].endswith("shortened once, to 60 words")
                   for row in result.unresolved)

    def test_the_model_first_reader_sends_it_back_once_too(self):
        pymupdf = pytest.importorskip("pymupdf")
        from planlens.document import open_document
        from report_ingest import model_first_read as mfr
        from report_ingest.model_first_pages import PageGroup

        made = pymupdf.open()
        for n in range(3):
            made.new_page().insert_text((72, 72), f"MADE-UP PAGE {n}")
        data = made.tobytes()
        made.close()
        engine = FakeEngine([
            {"final": NarrativeReading(
                projectName="Made-up Annex", quickSummary=_words(103),
                structureList=["Made-up Annex (MA)", "MA"])},
            {"final": SummaryReading(quickSummary=_words(70))}])
        engine.accepts_pdf = True
        opened = open_document(data, name="made_up")
        try:
            blob = mfr.read_narrative(
                opened, [PageGroup("narrative", [0, 1, 2], key="narrative")],
                engine, report_id="R00")
        finally:
            opened.close()

        assert len(engine.calls) == 2
        assert engine.calls[1]["output_format"] is SummaryReading
        assert blob["general"]["quickSummary"] == _words(70)
        assert blob["general"]["structureList"] == ["Made-up Annex (MA)"]
        assert blob["general"]["structureCount"] == 1
        assert blob["cost"]["calls"] == 2
        assert any(row.get("what") == "quickSummary" and
                   row["why"].endswith("shortened once, to 70 words")
                   for row in blob["unresolved"])

    def test_today_s_reader_keeps_it_when_the_budget_is_spent(self, doc):
        from report_ingest.narrative_reader import read_narrative
        from report_ingest.tests.narrative_fixtures import NARRATIVE_PAGES

        engine = FakeEngine([
            {"final": NarrativeReading(quickSummary=_words(106))}])
        result = read_narrative(doc, NARRATIVE_PAGES, engine, budget=1,
                                front_pages=0, roles=[], retrieve=False)

        assert result.general.quickSummary == _words(106)
        assert result.model_calls == 1
