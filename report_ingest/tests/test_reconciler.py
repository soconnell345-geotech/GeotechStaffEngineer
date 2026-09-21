"""The reconciler: what it links, what it counts, and what it refuses to settle.

Pure Python over synthetic records, so every test here is about a rule rather
than a model. The rule that matters most is the one about disagreements: the
tests below check that both values survive a conflict, in the record, with
their pages.
"""

from __future__ import annotations

import pytest

from report_ingest.model import (
    AtterbergResult, BearingValue, Calculation, GeneralFacts, LabTest,
    MoistureDensityResult, NamedQuantity, NaturalHazardFacts, QAEntry,
    Quantity,
)
from report_ingest.reconciler import (
    DEPTH_TOL_M, depth_m, fold_id, reconcile, si_view,
)
from report_ingest.tests.fake_engine import FakeEngine
from report_ingest.tests.record_fixtures import (
    atterberg, boring, build_record, general_facts, gradation, orphan_sheet,
    page_context, pit, q, summary_table,
)


def kinds_of(record, kind):
    return [entry for entry in record.qa if entry.kind == kind]


class TestTheLabToGroundLink:

    def test_a_sheet_is_linked_by_its_printed_hole_and_depth(self):
        record = reconcile(build_record())

        sheet = record.lab_tests[0]
        assert sheet.investigation_id == "B-1"       # untouched, as printed
        assert sheet.linked_investigation_id == "B-1"
        assert sheet.linked_sample_id == "S-1"
        assert sheet.linked_depth_delta_m == 0.0

    def test_the_identifier_is_matched_however_it_is_written(self):
        sheet = atterberg(investigation="b 1", sample="")
        record = reconcile(build_record(lab_tests=[sheet]))

        assert record.lab_tests[0].linked_investigation_id == "B-1"
        assert fold_id("B-1") == fold_id("b 1") == "B1"

    def test_the_depth_match_is_unit_aware(self):
        # The sheet prints metres and the log prints feet: 0.61 m is the
        # 2.0 ft sample, and nothing but a conversion says so.
        sheet = atterberg(sample="", depth_ft=0.61)
        sheet.depth_top = Quantity(value=0.61, unit="m")
        record = reconcile(build_record(lab_tests=[sheet]))

        assert record.lab_tests[0].linked_sample_id == "S-1"

    def test_a_depth_beyond_the_tolerance_does_not_link(self):
        sheet = atterberg(sample="", depth_ft=4.0)   # 1.22 m; S-1 is 0.61 m
        record = reconcile(build_record(lab_tests=[sheet]))

        assert record.lab_tests[0].linked_sample_id == ""
        assert any("no sample" in e.detail for e in kinds_of(record, "partial"))

    def test_a_depth_inside_the_tolerance_links(self):
        # 2.4 ft is 0.73 m; the sample runs 0.61 m to 1.07 m, so it is inside.
        sheet = atterberg(sample="", depth_ft=2.4)
        record = reconcile(build_record(lab_tests=[sheet]))

        assert record.lab_tests[0].linked_sample_id == "S-1"
        assert record.lab_tests[0].linked_depth_delta_m <= DEPTH_TOL_M

    def test_a_sheet_naming_a_hole_that_is_not_here_is_recorded(self):
        record = reconcile(build_record(lab_tests=[orphan_sheet()]))

        (entry,) = [e for e in kinds_of(record, "partial")
                    if "not among the explorations" in e.detail]
        assert entry.values == ["B-9"]
        assert entry.pages == [15]
        assert record.lab_tests[0].linked_investigation_id == ""

    def test_a_sheet_naming_no_hole_at_all_is_recorded(self):
        sheet = atterberg(investigation="", sample="")
        record = reconcile(build_record(lab_tests=[sheet]))

        assert any("names no exploration" in e.detail
                   for e in kinds_of(record, "partial"))

    def test_a_summary_table_reports_the_holes_it_names_that_are_missing(self):
        table = summary_table()
        table.result.rows[0].investigation_id = "B-7"
        record = reconcile(build_record(lab_tests=[table]))

        (entry,) = [e for e in kinds_of(record, "partial")
                    if "summary table" in e.detail]
        assert entry.values == ["B-7"]


def _calc(kind, subject="", results=(), pages=(20, 21)):
    """One calculation, with its results as ``(label, value, unit)``.

    A unit of ``None`` means the page printed a WORD -- a site class, an OK
    -- and the value travels as text.
    """
    return Calculation(
        kind=kind, subject=subject, pages=list(pages),
        results=[NamedQuantity(
            name=name,
            value=(None if unit is None
                   else Quantity(value=float(value), unit=unit)),
            text=("" if unit is not None else str(value)))
            for name, value, unit in results])


class TestTheCalculationLink:
    """A calculation whose subject names a boring belongs with that boring."""

    def test_a_subject_naming_a_hole_is_linked_to_it(self):
        record = reconcile(build_record(calculations=[
            _calc("settlement", subject="Settlement beneath B-1")]))

        assert record.calculations[0].linked_investigation_id == "B-1"
        assert record.calculations[0].subject == "Settlement beneath B-1"

    def test_a_subject_naming_no_hole_links_to_nothing(self):
        record = reconcile(build_record(calculations=[
            _calc("settlement", subject="the north wing mat")]))

        assert record.calculations[0].linked_investigation_id == ""

    def test_a_subject_naming_a_hole_this_report_does_not_carry(self):
        record = reconcile(build_record(calculations=[
            _calc("settlement", subject="Settlement beneath B-9")]))

        assert record.calculations[0].linked_investigation_id == ""


class TestTheCalculationsAgainstTheNarrative:
    """A conflict between the prose and the appendix is RECORDED, never
    settled: a report whose text recommends one pressure over an appendix
    that computed another is telling a reviewer something."""

    def _general(self, **over):
        facts = general_facts()
        for name, value in over.items():
            setattr(facts, name, value)
        return facts

    def test_a_bearing_pressure_that_agrees_raises_nothing(self):
        general = self._general(bearingCapacityValues=[BearingValue(
            value=Quantity(value=150.0, unit="kPa"))])
        record = reconcile(build_record(general=general, calculations=[
            _calc("shallow_foundation_bearing",
                  results=[("Allowable bearing pressure", 150.0, "kPa")])]))

        assert not [e for e in record.qa
                    if e.where.startswith("calculations.")]

    def test_a_bearing_pressure_that_differs_is_a_disagreement(self):
        general = self._general(bearingCapacityValues=[BearingValue(
            value=Quantity(value=150.0, unit="kPa"))])
        record = reconcile(build_record(general=general, calculations=[
            _calc("shallow_foundation_bearing",
                  results=[("Allowable bearing pressure", 224.0, "kPa")])]))

        (entry,) = [e for e in record.qa
                    if e.where == "calculations.shallow_foundation_bearing"]
        assert entry.kind == "disagreement"
        assert entry.values == ["calculation 224 kPa", "narrative 150 kPa"]
        assert entry.pages == [20, 21]

    def test_the_two_are_compared_in_si(self):
        """A calculation in psf beside a recommendation in kPa is ONE number
        and must not read as a conflict with itself."""
        general = self._general(bearingCapacityValues=[BearingValue(
            value=Quantity(value=150.0, unit="kPa"))])
        record = reconcile(build_record(general=general, calculations=[
            _calc("shallow_foundation_bearing",
                  results=[("Allowable bearing pressure", 3132.6, "psf")])]))

        assert not [e for e in record.qa
                    if e.where.startswith("calculations.")]

    def test_a_result_the_page_does_not_call_a_bearing_pressure_is_ignored(
            self):
        """A printout states dozens of pressures and only the one it CALLS a
        bearing pressure is the recommendation."""
        general = self._general(bearingCapacityValues=[BearingValue(
            value=Quantity(value=150.0, unit="kPa"))])
        record = reconcile(build_record(general=general, calculations=[
            _calc("shallow_foundation_bearing",
                  results=[("Overburden pressure at founding level",
                            30.0, "kPa")])]))

        assert not [e for e in record.qa
                    if e.where.startswith("calculations.")]

    def test_a_settlement_the_prose_states_is_compared(self):
        general = self._general(bearingCapacity=[
            "Total settlements are estimated at 25 mm."])
        record = reconcile(build_record(general=general, calculations=[
            _calc("settlement",
                  results=[("Total settlement", 74.0, "mm")])]))

        (entry,) = [e for e in record.qa
                    if e.where == "calculations.settlement"]
        assert entry.kind == "disagreement"
        assert "74 mm" in entry.values[0] and "25 mm" in entry.values[1]

    def test_a_settlement_the_prose_does_not_state_raises_nothing(self):
        record = reconcile(build_record(calculations=[
            _calc("settlement", results=[("Total settlement", 74.0, "mm")])]))

        assert not [e for e in record.qa
                    if e.where.startswith("calculations.")]

    def test_a_site_class_that_differs_is_a_disagreement(self):
        hazards = NaturalHazardFacts(siteClass="Site Class D")
        record = reconcile(build_record(
            natural_hazards=hazards,
            calculations=[_calc("site_response", results=[
                ("Seismic Site Class", "C", None)])]))

        (entry,) = [e for e in record.qa
                    if e.where == "calculations.site_response"]
        assert entry.kind == "disagreement"

    def test_a_site_class_that_agrees_raises_nothing(self):
        hazards = NaturalHazardFacts(siteClass="Site Class C")
        record = reconcile(build_record(
            natural_hazards=hazards,
            calculations=[_calc("site_response", results=[
                ("Seismic Site Class", "C", None)])]))

        assert not [e for e in record.qa
                    if e.where.startswith("calculations.")]


class TestTheCounts:

    def test_a_narrative_count_that_the_appendix_does_not_bear_out(self):
        general = general_facts()
        general.boringCount = 3
        record = reconcile(build_record(general=general))

        (entry,) = [e for e in kinds_of(record, "count_mismatch")
                    if e.where == "general.boringCount"]
        assert entry.values == ["narrative 3", "found 1"]
        # The narrative is NOT corrected: what it said is what it said.
        assert record.general.boringCount == 3

    def test_a_count_that_agrees_says_nothing(self):
        record = reconcile(build_record())

        assert not [e for e in kinds_of(record, "count_mismatch")
                    if e.where == "general.boringCount"]

    def test_a_named_boring_with_no_log_is_recorded(self):
        general = general_facts()
        general.boringDictionary = ["B-1", "B-2"]
        record = reconcile(build_record(general=general))

        (entry,) = [e for e in kinds_of(record, "count_mismatch")
                    if "no log in this report carries" in e.detail]
        assert entry.values == ["B-2"]

    def test_a_log_the_narrative_does_not_name_is_recorded(self):
        record = reconcile(build_record(
            investigations=[boring(), boring("B-4"), pit()]))

        (entry,) = [e for e in kinds_of(record, "count_mismatch")
                    if "does not name" in e.detail]
        assert entry.values == ["B-4"]

    def test_what_was_found_is_written_onto_the_narrative_facts(self):
        record = reconcile(build_record())

        assert record.narrative.found_counts == {
            "borings": 1, "test_pits": 1, "lab_tests": 3}
        assert record.narrative.found_ids["boring"] == ["B-1"]


class TestTheSummaryTableCrossCheck:

    def test_a_disagreement_keeps_both_values_and_both_pages(self):
        record = reconcile(build_record(lab_tests=[
            atterberg(ll=38.0), summary_table(ll_for_s1=41.0)]))

        # The row's liquid limit AND the plasticity index it implies both
        # disagree with the sheet, and both are reported: a reviewer opening
        # the page needs to see everything that does not match.
        found = {e.where.split(".")[1].split(" ")[0]: e
                 for e in kinds_of(record, "conflict")}
        assert set(found) == {"ll", "pi"}
        entry = found["ll"]
        assert "summary table 41" in entry.values[0]
        assert "atterberg sheet 38" in entry.values[1]
        assert entry.pages == [12, 14]
        # Neither value was changed by the finding.
        assert record.lab_tests[0].result.ll == 38.0
        assert record.lab_tests[1].result.rows[0].ll == 41.0

    def test_values_that_agree_raise_nothing(self):
        record = reconcile(build_record(lab_tests=[
            atterberg(ll=38.0), gradation(), summary_table(ll_for_s1=38.0)]))

        assert kinds_of(record, "conflict") == []

    def test_a_row_is_only_compared_with_the_sheet_for_its_own_specimen(self):
        # The table's B-1 S-3 row carries fines; the atterberg sheet at 2 ft
        # is a different specimen and must not be compared with it.
        record = reconcile(build_record(lab_tests=[
            atterberg(ll=38.0), summary_table(ll_for_s1=38.0)]))

        assert kinds_of(record, "conflict") == []

    def test_a_value_only_one_side_carries_is_not_a_conflict(self):
        sheet = LabTest(kind="moisture_content", investigation_id="B-1",
                        sample_id="S-1", depth_top=q(2.0, "ft"), pages=[16],
                        result=MoistureDensityResult(kind="moisture_content",
                                                     wc=24.1))
        record = reconcile(build_record(lab_tests=[sheet, summary_table()]))

        assert kinds_of(record, "conflict") == []

    def test_an_engine_comments_and_changes_nothing(self):
        from report_ingest.reconciler import _ConflictComment, _ConflictComments

        engine = FakeEngine([{"final": _ConflictComments(comments=[
            _ConflictComment(index=1,
                             comment="the table is a transcription; open "
                                     "page 12 and read the sheet")])}])
        record = reconcile(build_record(lab_tests=[
            atterberg(ll=38.0), summary_table(ll_for_s1=41.0)]), engine=engine)

        first = kinds_of(record, "conflict")[0]
        assert "Reviewer note: the table is a transcription" in first.detail
        assert len(first.values) == 2
        assert engine.n_calls == 1

    def test_no_engine_means_no_call(self):
        record = reconcile(build_record(lab_tests=[
            atterberg(ll=38.0), summary_table(ll_for_s1=41.0)]))

        assert kinds_of(record, "conflict")[0].detail.count("Reviewer") == 0


class TestUnitsAndPages:

    def test_a_unit_with_no_conversion_is_recorded_and_the_value_kept(self):
        sheet = atterberg()
        sheet.depth_top = Quantity(value=3.0, unit="fathoms")
        record = reconcile(build_record(lab_tests=[sheet]))

        (entry,) = kinds_of(record, "unconverted")
        assert "fathoms" in entry.detail
        assert record.lab_tests[0].depth_top.value == 3.0

    def test_the_si_view_converts_and_leaves_the_record_as_printed(self):
        record = reconcile(build_record())
        view = si_view(record)

        assert record.investigations[0].total_depth.unit == "ft"
        assert view["investigations.B-1"]["total_depth"]["unit"] == "m"
        assert round(view["investigations.B-1"]["total_depth"]["value"],
                     3) == 9.144
        assert round(view["general"]["bearingCapacityValues[0]"]["value"],
                     1) == 143.6

    def test_a_key_content_page_in_no_work_item_is_recorded(self):
        record = reconcile(build_record(), **page_context())

        (entry,) = [e for e in kinds_of(record, "skipped")
                    if e.where == "pages.calculation"]
        assert entry.pages == [20, 21]

    def test_a_page_with_no_text_and_no_di_is_recorded(self):
        record = reconcile(build_record(), **page_context())

        (entry,) = kinds_of(record, "unreadable")
        assert entry.pages == [20]          # 19 had a DI result

    def test_without_labels_the_page_checks_are_skipped_not_guessed(self):
        record = reconcile(build_record())

        assert kinds_of(record, "skipped") == []
        assert kinds_of(record, "unreadable") == []

    def test_what_a_reader_could_not_settle_is_carried_through(self):
        record = reconcile(build_record(), reader_unresolved=[
            {"what": "siteClass", "why": "the sentence is cut off",
             "page": 4, "value": "Site Class ?"}])

        (entry,) = [e for e in kinds_of(record, "partial")
                    if e.where == "siteClass"]
        assert entry.pages == [4] and entry.values == ["Site Class ?"]


class TestTheShapeOfThePass:

    def test_it_returns_the_same_record_and_keeps_what_was_there(self):
        record = build_record(lab_tests=[orphan_sheet()])
        record.qa.append(QAEntry(kind="note", where="earlier",
                                 detail="something the graph recorded"))
        out = reconcile(record)

        assert out is record
        assert out.qa[0].where == "earlier"
        assert len(out.qa) > 1

    def test_an_empty_record_reconciles_to_nothing(self):
        from report_ingest.model import ReportRecord

        record = reconcile(ReportRecord())

        assert record.qa == []
        assert record.narrative.found_counts == {"lab_tests": 0}

    def test_depth_m_refuses_a_unit_it_cannot_convert(self):
        assert depth_m(Quantity(value=1.0, unit="ft")) == pytest.approx(0.3048)
        assert depth_m(Quantity(value=1.0, unit="fathoms")) is None
        assert depth_m(None) is None
