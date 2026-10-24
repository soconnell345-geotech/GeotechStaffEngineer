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
    DEPTH_TOL_M, depth_m, fold_id, hole_key, reconcile, si_view,
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


def _label_notes(record):
    return [e for e in kinds_of(record, "note")
            if "head of a longer label" in e.detail]


class TestTheHoleInsideALabel:
    """RI-07: a sheet names its hole the way the laboratory writes it --
    without the log's leading zero, or at the head of a specimen label --
    and the link still finds it. Only holes the logs hold are looked for,
    and the printed words are never changed."""

    def test_a_leading_zero_on_the_log_is_the_same_hole(self):
        sheet = atterberg(investigation="B-1")
        record = reconcile(build_record(
            investigations=[boring("B-01"), pit()], lab_tests=[sheet]))

        assert sheet.investigation_id == "B-1"       # untouched, as printed
        assert sheet.linked_investigation_id == "B-01"
        assert sheet.linked_sample_id == "S-1"
        assert _label_notes(record) == []            # a whole match

    def test_only_the_zeros_that_start_a_number_are_dropped(self):
        assert hole_key("MW-02") == hole_key("mw 2") == "MW2"
        assert hole_key("B-10") == "B10"
        assert hole_key("B-100A") == "B100A"

    def test_a_hole_at_the_head_of_the_printed_boring_is_linked(self):
        sheet = atterberg(investigation="B-1 S-1", sample="")
        record = reconcile(build_record(lab_tests=[sheet]))

        assert sheet.investigation_id == "B-1 S-1"
        assert sheet.linked_investigation_id == "B-1"
        assert sheet.linked_sample_id == "S-1"
        (note,) = _label_notes(record)
        assert note.values == ["'B-1 S-1' -> B-1"]
        assert note.pages == [12]

    def test_a_hole_at_the_head_of_the_sample_label_is_linked(self):
        # The sheet printed no boring; its specimen is "B-1 2.0-3.5 ft".
        sheet = atterberg(investigation="", sample="B-1 2.0-3.5 ft")
        record = reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_investigation_id == "B-1"
        assert sheet.linked_sample_id == "S-1"       # by the depth
        assert not any("names no exploration" in e.detail
                       for e in kinds_of(record, "partial"))
        (note,) = _label_notes(record)
        assert note.values == ["'B-1 2.0-3.5 ft' -> B-1"]

    def test_a_printed_boring_outranks_the_sample_label(self):
        # The sheet names B-9, which is not here; a label that happens to
        # begin with B-1 does not overrule what the sheet printed.
        sheet = atterberg(investigation="B-9", sample="B-1 S-1")
        record = reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_investigation_id == ""
        assert any("not among the explorations" in e.detail
                   for e in kinds_of(record, "partial"))

    @pytest.mark.parametrize("printed", ["B-12", "B-1A", "B-1A S-1"])
    def test_a_hole_is_not_found_inside_a_longer_name(self, printed):
        sheet = atterberg(investigation=printed, sample="")
        record = reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_investigation_id == ""
        assert _label_notes(record) == []

    @pytest.mark.parametrize("label", ["B-1, 2.0 ft", "B-1/S-1", "B-1 (2.0)",
                                       "B-1  S-1  2.0-3.5"])
    def test_the_hole_ends_where_a_word_of_the_label_ends(self, label):
        sheet = atterberg(investigation="", sample=label)
        reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_investigation_id == "B-1"

    def test_the_longest_hole_at_the_head_wins(self):
        sheet = atterberg(investigation="B-1A S-1", sample="")
        reconcile(build_record(
            investigations=[boring("B-1"), boring("B-1A")],
            lab_tests=[sheet]))

        assert sheet.linked_investigation_id == "B-1A"

    def test_a_hole_named_by_a_number_alone_is_not_read_from_a_label(self):
        # A log called "2" and a specimen called "2": as likely a sample
        # number as a hole, so the label is not taken for the hole.
        sheet = atterberg(investigation="", sample="2")
        record = reconcile(build_record(
            investigations=[boring("2")], lab_tests=[sheet]))

        assert sheet.linked_investigation_id == ""
        assert any("names no exploration" in e.detail
                   for e in kinds_of(record, "partial"))

    def test_a_hole_named_by_a_number_still_links_when_printed_whole(self):
        sheet = atterberg(investigation="2")
        reconcile(build_record(investigations=[boring("2")],
                               lab_tests=[sheet]))

        assert sheet.linked_investigation_id == "2"

    @pytest.mark.parametrize("label", ["B-1", "B-1 40.0 ft"])
    def test_a_name_the_logs_give_a_sample_is_not_read_as_a_hole(self, label):
        # Boring B-1 AND the pit's bulk sample B-1: a label reading B-1 is a
        # sample number as likely as a hole, and B-1 has no sample at the
        # sheet's 40 ft to say otherwise.
        sheet = atterberg(investigation="", sample=label, depth_ft=40.0)
        record = reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_investigation_id == ""
        assert _label_notes(record) == []

    def test_a_hole_that_is_also_a_sample_name_links_at_its_own_depth(self):
        # The same name, but B-1's log holds a sample at the sheet's 2.0 ft.
        sheet = atterberg(investigation="", sample="B-1 2.0 ft")
        reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_investigation_id == "B-1"
        assert sheet.linked_sample_id == "S-1"

    def test_a_printed_boring_that_is_also_a_sample_name_links_whole(self):
        sheet = atterberg(investigation="S-2", sample="")
        reconcile(build_record(
            investigations=[boring("B-1"), boring("S-2")],
            lab_tests=[sheet]))

        assert sheet.linked_investigation_id == "S-2"

    def test_a_summary_row_spelt_otherwise_is_linked_as_a_sheet_is(self):
        # RI-56: a row is linked by the sheet's rules into its own link
        # fields, so "B-01" is the logged B-1 and not a hole missing.
        table = summary_table()
        table.result.rows[0].investigation_id = "B-01"
        record = reconcile(build_record(lab_tests=[table]))

        row = table.result.rows[0]
        assert (row.linked_investigation_id, row.linked_sample_id) == \
            ("B-1", "S-1")
        assert row.investigation_id == "B-01"           # as printed
        assert not [e for e in kinds_of(record, "partial")
                    if "summary table" in e.detail]


def _sounding(name):
    hole = boring(name)
    hole.kind, hole.samples, hole.spt = "cpt", [], []
    return hole


def _no_sample_names(hole):
    hole.samples, hole.spt = [], []
    return hole


def _linked(investigations, printed="", label="", depth_ft=2.0):
    sheet = atterberg(investigation=printed, sample=label, depth_ft=depth_ft)
    reconcile(build_record(investigations=investigations, lab_tests=[sheet]))
    return sheet


class TestTheReviewProbes:
    """The RI-07 review's inputs that linked a test to the wrong hole: a
    depth run onto the hole's number, a composite of several holes, and two
    logged spellings of one number. Each now links the right hole or none."""

    B12 = [f"B-{i}" for i in range(1, 13)]

    @pytest.mark.parametrize("printed,label,holes,want", [
        ("", "B-1 2.0-3.5 ft", "B", "B-1"),
        ("B-1 2.0", "", "B", "B-1"),
        ("", "B-1, 2.0 ft", "B", "B-1"),
        ("", "B-1 2'", "B", "B-1"),
        ("", "B-1-2", "B", "B-1"),
        ("", "B-1 (2.0)", "B", "B-1"),
        ("", "BH-1 1.5-1.95 m", "BH", "BH-1"),
    ])
    def test_a_depth_after_the_hole_is_not_part_of_its_number(
            self, printed, label, holes, want):
        names = [f"{holes}-{i}" for i in range(1, 13)]
        sheet = _linked([boring(n) for n in names], printed, label)
        assert sheet.linked_investigation_id == want

    @pytest.mark.parametrize("printed,label", [
        ("", "B-1/B-2"), ("", "B-1 & B-2"), ("B-1, B-2", ""),
        ("", "B-1 to B-4 composite"),
    ])
    def test_a_composite_of_several_holes_names_no_one_hole(
            self, printed, label):
        sheet = _linked([boring(f"B-{i}") for i in range(1, 5)],
                        printed, label)
        assert sheet.linked_investigation_id == ""

    @pytest.mark.parametrize("printed,logged,want", [
        ("B-7", ["B-07", "B-7"], "B-7"),
        ("B-07", ["B-7", "B-07"], "B-07"),
        ("B-007", ["B-07", "B-7"], ""),
    ])
    def test_two_logged_spellings_of_one_number_are_two_holes(
            self, printed, logged, want):
        sheet = _linked([boring(n) for n in logged], printed)
        assert sheet.linked_investigation_id == want

    def test_a_bare_label_is_not_taken_for_a_sounding(self):
        # Logs that read no sample names give no other guard: "S-1" alone is
        # as likely the sample's number as the sounding's name.
        sheet = _linked([_no_sample_names(boring("B-1")), _sounding("S-1")],
                        label="S-1")
        assert sheet.linked_investigation_id == ""

    def test_a_label_that_says_more_is_still_not_taken_for_a_sounding(self):
        sheet = _linked([_no_sample_names(boring("B-1")), _sounding("S-1")],
                        label="S-1 2.0 ft")
        assert sheet.linked_investigation_id == ""

    @pytest.mark.parametrize("label", ["MW-2", "MW-2 2.0 ft"])
    def test_a_sample_label_is_taken_for_a_monitoring_well(self, label):
        well = boring("MW-2")
        well.kind = "well"
        sheet = _linked([well], label=label)
        assert sheet.linked_investigation_id == "MW-2"

    def test_a_bare_label_is_taken_for_a_boring_of_that_name(self):
        sheet = _linked([_no_sample_names(boring("B-1")),
                         _no_sample_names(boring("S-1"))], label="S-1")
        assert sheet.linked_investigation_id == "S-1"

    def test_a_label_another_hole_holds_as_a_sample_there_is_refused(self):
        # B-1 is a boring and TP-1's bulk sample; at 2.0 ft both readings
        # have a sample, so the label names neither for certain.
        tp = pit("TP-1")
        tp.samples[0].top = q(2.0, "ft")
        sheet = _linked([boring("B-1"), tp], label="B-1")
        assert sheet.linked_investigation_id == ""

    def test_a_sample_named_in_the_label_is_taken_at_its_depth(self):
        from report_ingest.model import Sample
        hole = boring("B-1")
        hole.samples.append(Sample(sample_id="U-1", top=q(2.0, "ft"),
                                   bottom=q(2.5, "ft"), kind="shelby"))
        by_depth = _linked([hole], label="B-1 2.0 ft")
        assert by_depth.linked_sample_id == "S-1"        # the first at 2.0 ft
        hole = boring("B-1")
        hole.samples.append(Sample(sample_id="U-1", top=q(2.0, "ft"),
                                   bottom=q(2.5, "ft"), kind="shelby"))
        by_name = _linked([hole], label="B-1 U-1")
        assert by_name.linked_sample_id == "U-1"

    def test_a_sample_named_in_the_label_elsewhere_is_still_taken(self):
        # RI-56: the label names S-3, which the log puts at 14.5 ft; the
        # sheet prints 2 ft, where the log has S-1. The label is followed.
        sheet = _linked([boring("B-1")], label="B-1 S-3")
        assert sheet.linked_sample_id == "S-3"


class TestTheLabelBeforeTheDepth:
    """RI-56: a printed label that names one of the hole's samples is that
    sample; the depth decides only where no label names one, and a label
    the depth disagrees with is followed and said."""

    def _conflicts(self, record):
        return [e for e in kinds_of(record, "conflict")
                if e.where.endswith(".lab_tests")]

    def test_a_label_the_depth_disagrees_with_is_followed_and_said(self):
        sheet = atterberg(sample="S-3", depth_ft=2.0)
        record = reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_sample_id == "S-3"
        (entry,) = self._conflicts(record)
        assert entry.rank == "open"
        assert entry.where == "B-1.lab_tests"
        assert "Linked to S-3 by the label" in entry.detail
        assert "has S-1 there" in entry.detail
        assert entry.values == ["label B-1 S-3", "depth 2 ft"]
        assert entry.pages == [12]

    def test_a_label_the_depth_agrees_with_raises_nothing(self):
        sheet = atterberg(sample="S-3", depth_ft=14.5)
        record = reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_sample_id == "S-3"
        assert self._conflicts(record) == []

    def test_a_label_naming_no_sample_is_linked_by_depth_alone(self):
        sheet = atterberg(sample="Bag 4", depth_ft=7.0)
        record = reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_sample_id == "S-2"
        assert self._conflicts(record) == []

    def test_sheets_cut_from_one_sample_are_one_entry(self):
        one = atterberg(sample="S-3", depth_ft=2.0, page=12)
        two = atterberg(sample="S-3", depth_ft=2.0, page=13)
        record = reconcile(build_record(lab_tests=[one, two]))

        (entry,) = self._conflicts(record)
        assert entry.detail.startswith("2 lab sheets name sample S-3")
        assert entry.pages == [12, 13]

    def test_a_label_naming_several_samples_links_to_none(self):
        sheet = atterberg(sample="S-2,S-3", depth_ft=2.0)
        record = reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_investigation_id == "B-1"
        assert sheet.linked_sample_id == ""
        (entry,) = self._conflicts(record)
        assert "samples S-2, S-3 of B-1" in entry.detail
        assert "to no sample" in entry.detail
        # the conflict says why, so no second entry for the missing sample
        assert not [e for e in record.qa
                    if "no sample of it sits within" in e.detail]

    def test_an_interval_meeting_several_named_samples_is_a_composite(self):
        sheet = atterberg(sample="S-2,S-3", depth_ft=7.0)
        sheet.depth_bottom = q(16.0, "ft")
        record = reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_investigation_id == "B-1"
        assert sheet.linked_sample_id == ""
        (entry,) = self._conflicts(record)
        assert "puts S-2, S-3 within" in entry.detail
        assert "to no sample" in entry.detail

    def test_the_numbers_of_a_depth_range_are_not_another_hole(self):
        # A record with a hole named by a bare number: "4.0-5.5" in the
        # label does not name it, so S-3 is still the label's sample.
        sheet = atterberg(investigation="B-1", sample="S-3 4.0-5.5",
                          depth_ft=2.0)
        record = reconcile(build_record(
            investigations=[boring("B-1"), boring("4")], lab_tests=[sheet]))

        assert sheet.linked_sample_id == "S-3"
        (entry,) = self._conflicts(record)
        assert "Linked to S-3 by the label" in entry.detail

    @pytest.mark.parametrize("label", ["Hole 2 S-3", "2/S-3", "BH 2, S-3"])
    def test_a_bare_number_naming_another_hole_is_that_hole(self, label):
        # A sheet of hole 1 whose label names hole 2 by its number: S-3 is
        # not hole 1's S-3, so the depth decides.
        sheet = atterberg(investigation="1", sample=label, depth_ft=2.0)
        record = reconcile(build_record(
            investigations=[boring("1"), boring("2")], lab_tests=[sheet]))

        assert sheet.linked_investigation_id == "1"
        assert sheet.linked_sample_id == "S-1"
        assert self._conflicts(record) == []

    @pytest.mark.parametrize("label", ["S-3, 2 - 3.5 ft", "S-3 2 to 3",
                                       "S-3 at 2 ft"])
    def test_a_number_in_a_range_or_a_depth_is_no_hole(self, label):
        sheet = atterberg(investigation="1", sample=label, depth_ft=2.0)
        record = reconcile(build_record(
            investigations=[boring("1"), boring("2")], lab_tests=[sheet]))

        assert sheet.linked_sample_id == "S-3"
        (entry,) = self._conflicts(record)
        assert "Linked to S-3 by the label" in entry.detail

    def test_the_log_s_depth_is_said_in_the_printed_unit_and_rounded(self):
        hole = boring("B-1")
        hole.samples[2].top = q(4.4651, "m")
        sheet = atterberg(sample="S-3", depth_ft=2.0)
        record = reconcile(build_record(investigations=[hole],
                                        lab_tests=[sheet]))

        (entry,) = self._conflicts(record)
        assert "a depth of 2 ft; its log puts S-3 at 14.6 ft" in entry.detail

    def test_several_named_samples_one_at_the_depth_is_that_one(self):
        sheet = atterberg(sample="S-2,S-3", depth_ft=14.5)
        record = reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_sample_id == "S-3"
        assert self._conflicts(record) == []

    def test_a_printed_interval_holding_the_log_s_point_raises_nothing(self):
        # The sheet prints the interval it sampled; the log holds one point
        # inside it, 0.4 ft below the printed top.
        hole = boring("B-1")
        hole.samples[2].top, hole.samples[2].bottom = q(13.6, "ft"), None
        sheet = atterberg(sample="S-3", depth_ft=13.2)
        sheet.depth_bottom = q(14.9, "ft")
        record = reconcile(build_record(investigations=[hole],
                                        lab_tests=[sheet]))

        assert sheet.linked_sample_id == "S-3"
        assert self._conflicts(record) == []

    def test_the_printed_interval_is_widened_by_the_tolerance(self):
        # 11.8-14.2 ft printed, 14.5-16.0 ft logged: 0.3 ft apart.
        sheet = atterberg(sample="S-3", depth_ft=11.8)
        sheet.depth_bottom = q(14.2, "ft")
        record = reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_sample_id == "S-3"
        assert self._conflicts(record) == []

    def test_an_interval_short_of_the_sample_is_still_said(self):
        sheet = atterberg(sample="S-3", depth_ft=2.0)
        sheet.depth_bottom = q(3.5, "ft")
        record = reconcile(build_record(lab_tests=[sheet]))

        assert sheet.linked_sample_id == "S-3"
        (entry,) = self._conflicts(record)
        assert "within" not in entry.detail and "has S-1 there" in \
            entry.detail

    def test_a_sample_word_beside_another_hole_s_name_is_not_counted(self):
        # "S-3 (B-2)" printed under B-1: the label names another hole, so its
        # sample word names no sample of B-1 and the depth decides.
        sheet = atterberg(investigation="B-1", sample="S-3 (B-2)",
                          depth_ft=2.0)
        record = reconcile(build_record(
            investigations=[boring("B-1"), boring("B-2")],
            lab_tests=[sheet]))

        assert sheet.linked_investigation_id == "B-1"
        assert sheet.linked_sample_id == "S-1"
        assert self._conflicts(record) == []

    def test_a_summary_row_is_linked_by_its_own_columns(self):
        table = summary_table()
        record = reconcile(build_record(lab_tests=[table]))

        links = [(r.linked_investigation_id, r.linked_sample_id)
                 for r in table.result.rows]
        assert links == [("B-1", "S-1"), ("B-1", "S-3")]
        assert table.linked_investigation_id == ""      # the table: none
        assert self._conflicts(record) == []

    def test_a_summary_row_the_depth_disagrees_with_is_said(self):
        table = summary_table()
        table.result.rows[1].depth_top = q(7.0, "ft")    # S-2's depth
        record = reconcile(build_record(lab_tests=[table]))

        assert table.result.rows[1].linked_sample_id == "S-3"
        (entry,) = self._conflicts(record)
        assert entry.detail.startswith("1 summary row names sample S-3")
        assert entry.pages == [14]

    def test_a_summary_row_naming_no_hole_is_left_unlinked(self):
        table = summary_table()
        table.result.rows[0].investigation_id = ""
        table.result.rows[0].sample_id = ""
        record = reconcile(build_record(lab_tests=[table]))

        assert table.result.rows[0].linked_investigation_id == ""
        assert not [e for e in kinds_of(record, "partial")
                    if "summary table" in e.detail]


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

    def test_a_zero_padded_subject_links_to_the_hole(self):
        record = reconcile(build_record(
            investigations=[boring("B-01"), pit()],
            calculations=[_calc("settlement", subject="Settlement at B-1")]))

        assert record.calculations[0].linked_investigation_id == "B-01"

    def test_a_subject_is_not_linked_where_two_logs_share_the_key(self):
        # "B-07" and "B-7" logged as two holes are two holes: the zero fold
        # does not choose between them (an exact spelling still does).
        both = [boring("B-07"), boring("B-7"), pit()]
        loose = reconcile(build_record(
            investigations=both,
            calculations=[_calc("settlement", subject="Settlement at B-007")]))
        exact = reconcile(build_record(
            investigations=[boring("B-07"), boring("B-7"), pit()],
            calculations=[_calc("settlement", subject="Settlement at B-7")]))

        assert loose.calculations[0].linked_investigation_id == ""
        assert exact.calculations[0].linked_investigation_id == "B-7"

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

    def test_a_zero_padded_name_is_the_same_hole(self):
        general = general_facts()
        general.boringDictionary = ["B-1"]
        record = reconcile(build_record(
            general=general, investigations=[boring("B-01"), pit()]))

        assert not [e for e in kinds_of(record, "count_mismatch")
                    if "no log in this report carries" in e.detail
                    or "does not name" in e.detail]

    def test_two_different_names_are_still_apart(self):
        general = general_facts()
        general.boringDictionary = ["B-1", "B-10"]
        record = reconcile(build_record(
            general=general, investigations=[boring("B-01"), pit()]))

        (entry,) = [e for e in kinds_of(record, "count_mismatch")
                    if "no log in this report carries" in e.detail]
        assert entry.values == ["B-10"]

    def test_the_names_are_reported_as_printed(self):
        general = general_facts()
        general.boringDictionary = ["B-2", "B-1"]
        record = reconcile(build_record(
            general=general, investigations=[boring("B-01"), boring("B-003"),
                                             pit()]))

        (missing,) = [e for e in kinds_of(record, "count_mismatch")
                      if "no log in this report carries" in e.detail]
        (extra,) = [e for e in kinds_of(record, "count_mismatch")
                    if "does not name" in e.detail]
        assert missing.values == ["B-2"]
        assert extra.values == ["B-003"]

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

    def test_a_value_with_no_unit_printed_is_named_and_not_converted(self):
        # RI-20: until then a blank unit converted as the dimensionless one,
        # so this depth of 3 went into the SI view as 3 m.
        sheet = atterberg()
        sheet.depth_top = Quantity(value=3.0, unit="")
        record = reconcile(build_record(lab_tests=[sheet]))

        (entry,) = kinds_of(record, "unconverted")
        assert "no unit printed" in entry.detail
        assert "unknown" in entry.detail
        assert entry.values == ["lab_tests.depth_top"]
        assert record.lab_tests[0].depth_top.value == 3.0
        view = si_view(record)
        assert "depth_top" not in view["lab_tests.atterberg[0]"]

    def test_a_ph_has_no_unit_and_is_not_named(self):
        from report_ingest.model import ChemicalResult
        test = LabTest(kind="chemical", investigation_id="B-1",
                       depth_top=q(2.0, "ft"),
                       result=ChemicalResult(pH=Quantity(value=7.4, unit="")))
        record = reconcile(build_record(lab_tests=[test]))

        assert kinds_of(record, "unconverted") == []

    def test_a_water_analysis_in_mg_per_litre_needs_no_note(self):
        # RI-20: mg/L had no entry, so every water analysis carried an
        # "unconverted" note though DIGGS takes it as printed.
        from report_ingest.diggs_writer import (
            diggs_roundtrip_gate, diggs_schema_gate, write_diggs,
        )
        from report_ingest.model import ChemicalResult
        test = LabTest(kind="chemical", investigation_id="B-1",
                       depth_top=q(2.0, "ft"),
                       result=ChemicalResult(
                           # a pH has no unit: written with no uom (the
                           # schema has no code for "none" there)
                           pH=Quantity(value=7.4, unit=""),
                           sulfate=Quantity(value=250.0, unit="mg/L"),
                           chloride=Quantity(value=80.0, unit="mg/l"),
                           conductivity=Quantity(value=1500.0,
                                                 unit="uS/cm")))
        record = reconcile(build_record(lab_tests=[test]))
        assert kinds_of(record, "unconverted") == []

        notes = []
        xml = write_diggs(record, notes=notes)
        assert not any("Sulfate" in line or "Chloride" in line
                       or "Conductivity" in line or "pH" in line
                       for line in notes[0].skipped), notes[0].skipped
        assert "<uom>1</uom>" not in xml
        assert "<uom>mg/L</uom>" in xml or ">mg/L<" in xml
        assert diggs_schema_gate(xml)[0] is True
        ok, diffs = diggs_roundtrip_gate(
            xml, record.investigations, project=record.project,
            lab_tests=record.lab_tests, notes=notes[0])
        assert ok, diffs

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



# -- a log against itself (BACKLOG RI-28) ------------------------------------

def _m(v):
    return Quantity(value=v, unit="m")


def _hole(spt=(), td=None, layers=(), water=(), samples=()):
    from report_ingest.model import (
        SPT, Investigation, Layer, Sample, WaterLevel,
    )
    return Investigation(
        investigation_id="B-7", depth_unit="m", pages=[30, 31],
        total_depth=_m(td) if td is not None else None,
        layers=[Layer(top=_m(a), bottom=_m(b)) for a, b in layers],
        samples=[Sample(sample_id=sid, top=_m(a), bottom=_m(b), kind="spt")
                 for sid, a, b in samples],
        spt=[SPT(**row) for row in spt],
        water=[WaterLevel(depth=_m(d), when="at_completion") for d in water])


def _reconciled(*holes):
    from report_ingest.model import ReportRecord
    return reconcile(ReportRecord(investigations=list(holes)))


class TestALogAgainstItself:

    def test_drives_that_fill_their_interval_raise_nothing(self):
        rows = [{"depth_top": _m(t), "depth_bottom": _m(t + 0.45),
                 "blows": [3, 4, 5], "increment": _m(0.15)}
                for t in (1.05, 2.55, 4.05)]
        record = _reconciled(_hole(rows, td=6.0, layers=[(0, 6.0)]))
        assert record.qa == []

    def test_a_refusal_s_distance_in_the_wrong_unit_is_a_conflict(self):
        rows = [{"depth_top": _m(3.5), "depth_bottom": _m(3.65),
                 "blows": ['60/15"'], "refusal": True}]
        (entry,) = kinds_of(_reconciled(_hole(rows)), "conflict")
        assert "0.381 m" in entry.detail and "0.150 m" in entry.detail
        assert entry.pages == [30, 31]

    def test_a_refusal_that_stops_short_of_its_interval_is_fine(self):
        rows = [{"depth_top": _m(10.05), "depth_bottom": _m(10.5),
                 "blows": [32, "50/5cm"], "increment": _m(0.15),
                 "refusal": True}]
        assert _reconciled(_hole(rows)).qa == []

    @pytest.mark.parametrize("lengths, conflicts", [
        ((6, 6, 2.5), 0),       # its own lengths end in a halt (RI-50)
        ((6, 6, 6), 0),         # three whole drives that fill it
        ((6, 6, 9), 1),         # 21 in of drives in an 18 in interval
    ])
    def test_a_halt_shown_by_the_records_own_lengths_stops_short(
            self, lengths, conflicts):
        inch = 0.0254
        rows = [{"depth_top": _m(1.5), "depth_bottom": _m(1.5 + 18 * inch),
                 "blows": [16, 29, 50],
                 "increments": [Quantity(value=v, unit="in")
                                for v in lengths]}]
        record = _reconciled(_hole(rows))
        assert len(kinds_of(record, "conflict")) == conflicts

    @pytest.mark.parametrize("refusal, conflicts", [
        (False, 1),     # read as one 6 in drive in an 18 in interval
        (True, 0),      # the reader's own refusal flag: checked as a halt
    ])
    def test_a_lone_count_over_50_is_checked_as_read(self, refusal,
                                                     conflicts):
        # One whole number, no distance and no lengths: it may be the N
        # filed as a drive, and the drive sum is what catches that
        # (RI-50 review).
        inch = 0.0254
        rows = [{"depth_top": _m(1.5), "depth_bottom": _m(1.5 + 18 * inch),
                 "blows": [73], "refusal": refusal}]
        record = _reconciled(_hole(rows))
        assert len(kinds_of(record, "conflict")) == conflicts

    def test_a_lone_counts_length_with_no_unit_does_not_silence_it(self):
        # The length says nothing (RI-70 (4)): the count is as lone as one
        # with no length, and its drive sum is still checked.
        inch = 0.0254
        rows = [{"depth_top": _m(1.5), "depth_bottom": _m(1.5 + 18 * inch),
                 "blows": [67], "increments": [Quantity(value=6, unit="")]}]
        record = _reconciled(_hole(rows))
        assert len(kinds_of(record, "conflict")) == 1

    def test_a_record_that_halts_before_its_third_drive_is_no_form(self):
        # Two drives, the second over 50 and no refusal flag: a halt, not a
        # record of a form that drives twice (RI-50).
        rows = [{"depth_top": _m(t), "blows": b}
                for t, b in ((1.0, [1, 2, 3]), (2.5, [4, 5, 6]),
                             (4.0, [21, 57]), (5.5, [9, 9, 9]))]
        assert kinds_of(_reconciled(_hole(rows)), "count_mismatch") == []

    def test_the_interval_comes_from_the_linked_sample_when_the_drive_has_none(
            self):
        rows = [{"depth_top": _m(1.0), "blows": [3, 4], "sample_id": "S-1",
                 "increment": _m(0.15)}]
        hole = _hole(rows, samples=[("S-1", 1.0, 1.45)])
        (entry,) = kinds_of(_reconciled(hole), "conflict")
        assert "sample S-1" in entry.detail

    def test_a_drive_count_unlike_the_rest_of_the_log_is_named(self):
        rows = [{"depth_top": _m(t), "blows": b}
                for t, b in ((1.0, [1, 2, 3]), (2.5, [4, 5, 6]),
                             (4.0, [7, 8]), (5.5, [9, 9, 9]))]
        (entry,) = kinds_of(_reconciled(_hole(rows)), "count_mismatch")
        assert "4.00 m" in entry.detail and "give 3" in entry.detail
        # too few records to know the form: nothing
        assert _reconciled(_hole(rows[2:])).qa == []

    def test_depths_below_the_total_depth_are_one_entry_per_hole(self):
        hole = _hole(td=20.1, layers=[(0, 12.0), (12.0, 21.1)],
                     water=[20.5])
        (entry,) = kinds_of(_reconciled(hole), "out_of_range")
        assert "below its total depth of 20.10 m" in entry.detail
        assert "layer 2 bottom 21.10 m" in entry.detail
        assert "water (at_completion) 20.50 m" in entry.detail
        # a header that rounds the total depth is not caught out
        ok = _hole(td=20.0, layers=[(0, 20.04)])
        assert _reconciled(ok).qa == []


# -- the rank of each entry (BACKLOG RI-08) ----------------------------------

class TestTheRank:
    """Each entry says whether it needs a person: ``open``, ``changed`` or
    ``note``, from what it says rather than which pass wrote it."""

    def test_the_reconciler_s_own_entries_are_ranked_by_what_they_say(self):
        sheet = atterberg()
        sheet.depth_top = Quantity(value=3.0, unit="fathoms")
        nameless = atterberg(investigation="", sample="")
        record = reconcile(build_record(
            lab_tests=[atterberg(ll=38.0), summary_table(ll_for_s1=41.0),
                       orphan_sheet(), sheet, nameless]),
            reader_unresolved=[
                {"what": "depth", "refused_by": "python",
                 "why": "depth 99 is outside what the scale on this log "
                        "reads (0 to 20); refused rather than accepted"},
                {"what": "input Wall load", "refused_by": "python",
                 "why": "the unit 'tons/ft' is not one the record can "
                        "convert; the value is kept as printed and nothing "
                        "converts it"},
                {"what": "page 3", "refused_by": "budget",
                 "why": "the reader did not get to this page"},
                {"what": "hammer type", "refused_by": "reader",
                 "why": "the log names no hammer"}])

        def rank(where_or_detail):
            (entry,) = [e for e in record.qa
                        if where_or_detail in e.where
                        or where_or_detail in e.detail]
            return entry.rank

        # both values recorded, nothing settled them
        assert rank("lab_tests.ll at") == "open"
        # the sheet names a hole the logs do not hold
        assert rank("'B-9'") == "open"
        # a value read and thrown out; a page nobody read
        assert rank("refused rather than accepted") == "changed"
        assert rank("did not get to this page") == "changed"
        # kept as printed; a sheet that names no hole; the reader's own note
        assert rank("tons/ft") == "note"
        assert rank("fathoms") == "note"
        assert rank("names no exploration") == "note"
        assert rank("the log names no hammer") == "note"

    def test_a_log_s_own_checks_leave_the_values_and_are_open(self):
        rows = [{"depth_top": _m(t), "blows": b}
                for t, b in ((1.0, [1, 2, 3]), (2.5, [4, 5, 6]),
                             (4.0, [7, 8]), (5.5, [9, 9, 9]))]
        hole = _hole(rows, td=5.0, layers=[(0, 6.0)])
        record = _reconciled(hole)

        assert {e.kind: e.rank for e in record.qa} == {
            "count_mismatch": "open", "out_of_range": "open"}

    def test_the_two_voters_split_is_a_change_and_a_kept_value_a_note(self):
        from report_ingest.graph import _vote_qa

        qa = []
        _vote_qa(qa, {
            "disagreements": [{
                "what": "layer at 1.0 m: bottom", "where": "B-1.layers",
                "floor": "2.0 m", "model": "2.5 m", "kept": "model",
                "why": "the model named a box and a note"}],
            "kept": [{"what": "sample at 3.0 m: recovery", "page": 4}]},
            pages=[4])

        assert [(e.kind, e.rank) for e in qa] == [
            ("disagreement", "changed"), ("note", "note")]

    @pytest.mark.parametrize("what, floor, model", [
        ("water at 3 m: depth", "3 m", "not_encountered"),
        ("kind", "atterberg", "gradation"),
    ])
    def test_a_split_whose_record_carries_both_is_open(self, what, floor,
                                                       model):
        # Two water readings, or two lab kinds, stand side by side in the
        # record: the line says so, and nothing settled it (RI-70 (8)).
        from report_ingest.graph import _vote_qa

        qa = []
        _vote_qa(qa, {"disagreements": [{
            "what": what, "where": "B-1", "floor": floor, "model": model,
            "kept": "both", "why": "both are in the record"}]}, pages=[4])
        (entry,) = qa
        assert "; the record carries both. Both are in the record." in \
            entry.detail
        assert "both's" not in entry.detail
        assert entry.rank == "open"

    def test_the_ones_that_need_a_person_come_first_and_are_counted(self):
        record = reconcile(build_record(lab_tests=[
            atterberg(ll=38.0), summary_table(ll_for_s1=41.0),
            orphan_sheet()]))
        record.qa.insert(0, QAEntry(kind="note", where="workflow",
                                    detail="read on the standard workflow"))

        ranked = record.qa_ranked()
        order = ["open", "changed", "note"]
        assert [order.index(e.rank) for e in ranked] == sorted(
            order.index(e.rank) for e in record.qa)
        assert record.qa[0].where == "workflow"   # the record's own order
        counts = record.counts()
        assert counts["qa"] == len(record.qa)
        assert counts["qa_open"] + counts["qa_changed"] + counts[
            "qa_note"] == len(record.qa)
        assert counts["qa_open"] == record.qa_by_rank()["open"] >= 3

    def test_a_record_written_before_ranks_is_ranked_on_loading(self):
        import json
        from report_ingest.model import ReportRecord

        old = {"qa": [
            {"kind": "conflict", "where": "lab_tests.ll at B-1 2 ft",
             "detail": "the summary table and the atterberg sheet give "
                       "different values for ll; both are recorded",
             "values": [], "pages": [12, 14]},
            {"kind": "out_of_range", "where": "result Moment",
             "detail": "the unit 'kN-m' is not one the record can convert; "
                       "the value is kept as printed and nothing converts "
                       "it", "values": [], "pages": [40]}]}
        record = ReportRecord.model_validate_json(json.dumps(old))

        assert [e.rank for e in record.qa] == ["open", "note"]
        again = json.loads(record.model_dump_json())
        assert [e["rank"] for e in again["qa"]] == ["open", "note"]
        # a rank given is a rank kept
        assert QAEntry(kind="note", detail="x", rank="open").rank == "open"

    # -- one test for each rule of model._SAID_RANK, through the code that
    #    writes the entry

    def test_a_driven_record_whose_bottom_is_not_below_its_top_is_open(self):
        rows = [{"depth_top": _m(3.0), "depth_bottom": _m(2.55),
                 "blows": [3, 4, 5], "increment": _m(0.15)}]
        record = _reconciled(_hole(rows, td=6.0, layers=[(0, 6.0)]))

        (entry,) = [e for e in record.qa if "is not below its top"
                    in e.detail]
        assert (entry.kind, entry.rank) == ("out_of_range", "open")

    def test_a_lab_test_with_no_sample_at_its_depth_is_open(self):
        record = reconcile(build_record(lab_tests=[
            atterberg(sample="S-9", depth_ft=11.0)]))

        (entry,) = [e for e in record.qa if "no sample of it sits within"
                    in e.detail]
        assert (entry.kind, entry.rank) == ("partial", "open")

    def test_a_hole_in_the_summary_table_the_logs_lack_is_open(self):
        table = summary_table()
        table.result.rows[1].investigation_id = "B-9"
        record = reconcile(build_record(lab_tests=[table]))

        (entry,) = [e for e in record.qa
                    if e.where == "lab_tests.summary_table"]
        assert "are not among those read from this report" in entry.detail
        assert (entry.kind, entry.rank) == ("partial", "open")

    def test_a_reader_s_note_that_says_not_among_is_its_own_kind(self):
        # The lab rules match the reconciler's whole sentence, not the two
        # words a reader's free text may carry.
        record = reconcile(build_record(), reader_unresolved=[
            {"what": "hammer", "why": "the hammer type is not among those "
                                      "the legend lists"}])

        (entry,) = [e for e in record.qa if e.where == "hammer"]
        assert (entry.kind, entry.rank) == ("partial", "note")

    def test_pages_with_no_text_read_from_the_picture_are_a_note(self):
        record = reconcile(build_record(), no_text_pages=[3, 4], di_pages=[])

        (entry,) = [e for e in record.qa if e.kind == "unreadable"]
        assert "read from the picture" in entry.detail
        assert entry.rank == "note"

    def test_a_narrative_with_no_page_of_text_unread_is_a_change(self):
        from types import SimpleNamespace

        from report_ingest.graph import _narrative_items
        qa = []
        item = SimpleNamespace(kind="narrative", id="item_1", pages=[1, 2])
        assert _narrative_items([item], "standard", {1, 2}, qa) == ([], [])

        (entry,) = qa
        assert (entry.kind, entry.rank) == ("unreadable", "changed")

    def test_the_logs_answer_over_the_narrative_s_is_a_change(self):
        from types import SimpleNamespace

        from report_ingest.graph import _counts_from_logs
        record = build_record(general=general_facts().model_copy(
            update={"boringCount": 4}))
        record.narrative.pages = [3]
        items = [SimpleNamespace(kind="boring_log", id="item_5"),
                 SimpleNamespace(kind="test_pit_log", id="item_6")]
        qa = []
        _counts_from_logs(record, qa, "settle", items, items)

        assert record.general.boringCount == 1
        (entry,) = qa
        assert "over the narrative's" in entry.detail
        assert (entry.kind, entry.rank) == ("note", "changed")

    def test_a_label_the_review_moved_on_an_agreed_page_is_a_change(
            self, tmp_path):
        import json
        from types import SimpleNamespace

        from report_ingest.graph import Budgets, _Labelling, _review, _Spend
        from report_ingest.label_vote import PageChoice
        (tmp_path / "review.json").write_text(json.dumps({
            "asked_pages": [10],
            "changes": [{"page": 3, "from": "narrative", "to": "figure",
                         "reason": "a full-page drawing",
                         "evidence": "read_page"}]}))
        choices = [PageChoice(page=3, label="narrative", confidence=0.9,
                              agreed=True, policy="structural"),
                   PageChoice(page=10, label="figure", confidence=0.5,
                              agreed=False, policy="structural")]
        labels = {3: "narrative", 10: "figure"}
        qa = []
        _review(SimpleNamespace(n_pages=20), None, Budgets(), str(tmp_path),
                True, [], None, None, _Labelling(), choices, labels, [10],
                _Spend(), qa, [])

        assert labels[3] == "figure"
        (entry,) = [e for e in qa if e.where == "labels.page3"]
        assert "the review moved it" in entry.detail
        assert (entry.kind, entry.rank) == ("note", "changed")
        # the split nobody settled stays open
        (entry,) = [e for e in qa if e.where == "labels.page10"]
        assert (entry.kind, entry.rank) == ("label_disagreement", "open")

    @pytest.mark.parametrize("floor, model, rank", [
        # made-up values, each of a shape the saved runs show
        ("0.183", "0.183 g", "note"),               # a unit on one side
        ("12", "12%", "note"),
        ("37.0 feet", "37 feet", "note"),           # how it is printed
        ("= 0.730E+01 :unselected:", "7.3", "note"),
        ("47 K91305", "47K91305", "note"),
        ("13,750 psf", "13750 psf", "note"),
        ("TP-01", "TP-1", "note"),
        ("S-4", "4", "note"),                       # a series prefix
        ("4", "S4", "note"),
        ("12", "max 12", "changed"),                # a word apart from it
        ("approx 12", "12", "changed"),
        ("12", "El 12", "changed"),
        ("6 in", "6 ft", "changed"),                # two units
        ("2.0 m", "2.5 m", "changed"),              # two numbers
        ("-12.5", "12.5", "changed"),               # a sign
        ("12.5", "-12.5 kN-m", "changed"),
        ("\u221212.5", "12.5", "changed"),          # the minus sign
        ("12.5", "\u221212.5 kN-m", "changed"),
        ("\u221212.5", "-12.5", "note"),            # one minus, two glyphs
        ("<0.5", "0.5", "changed"),                 # a qualifier
        (">100", "100", "changed"),
        ("12 (refusal)", "12", "changed"),
        ("4", "4A", "changed"),                     # a letter run on
        ("4A", "4B", "changed"),
        ("S-4", "U-4", "changed"),                  # two names
        ("B-1", "B-1A", "changed"),
        ("12", "", "changed"),                      # one side read nothing
        ("4-6-8", "4-6", "changed"),
    ])
    def test_a_split_on_how_one_number_is_printed_is_a_note(
            self, floor, model, rank):
        from report_ingest.graph import _vote_qa
        qa = []
        _vote_qa(qa, {"disagreements": [{
            "what": "input Unit Weight: value", "where": "calc.inputs",
            "floor": floor, "model": model, "floor_method": "grid",
            "model_method": "model", "kept": "model",
            "why": "the model named a box and a note"}]}, pages=[4])

        (entry,) = qa
        assert (entry.kind, entry.rank) == ("disagreement", rank)

    def test_a_minus_is_a_sign_only_where_it_leads_a_number(self):
        from report_ingest.model import _number_parts
        assert _number_parts("4-5-6")[0] == [4.0, 5.0, 6.0]
        assert _number_parts("s-4")[0] == [4.0]
        assert _number_parts("-12.5 kn-m")[0] == [-12.5]
        assert _number_parts("= \u221212.5")[0] == [-12.5]
        assert _number_parts("1.5e-03 m")[0] == [0.0015]


# -- a continuation sheet read as a log of its own (BACKLOG RI-08) -----------

def _sheet(name, pages, sheet="", samples=(), layers=(), td=None, unit="m",
           kind="boring"):
    from report_ingest.model import Investigation, Layer, Sample
    return Investigation(
        investigation_id=name, kind=kind, depth_unit=unit, pages=list(pages),
        sheet=sheet,
        total_depth=Quantity(value=td, unit=unit) if td is not None else None,
        layers=[Layer(top=Quantity(value=a, unit=unit),
                      bottom=Quantity(value=b, unit=unit), description=d)
                for a, b, d in layers],
        samples=[Sample(sample_id=sid, top=Quantity(value=a, unit=unit),
                        bottom=Quantity(value=b, unit=unit), kind=kind)
                 for sid, a, b, kind in samples])


def _folded(*sheets, lab=()):
    from report_ingest.model import ReportRecord
    return reconcile(ReportRecord(investigations=list(sheets),
                                  lab_tests=list(lab)))


class TestContinuationSheets:

    def test_a_printed_sheet_two_of_two_is_folded_into_its_first_sheet(self):
        # Made-up values, of the shape the saved runs show.
        first = _sheet("BH-6", [70], "1 of 2", td=12.0,
                       samples=[("S1", 1.0, 1.45, "spt"),
                                ("C4", 10.5, 12.0, "core")],
                       layers=[(0.0, 2.6, "CLAY"), (2.6, 12.0, "ROCK")])
        later = _sheet("BH-6", [71], "2 of 2", td=17.4,
                       samples=[("C4", 10.5, 12.0, "core"),
                                ("C5", 12.0, 13.5, "core")],
                       layers=[(12.0, 17.4, "ROCK, fresher")])
        sheet = LabTest(kind="unconfined_rock", investigation_id="BH-6",
                        depth_top=Quantity(value=12.8, unit="m"), pages=[95])
        record = _folded(first, later, lab=[sheet])

        (hole,) = record.investigations
        assert hole.pages == [70, 71]
        assert [s.sample_id for s in hole.samples] == ["S1", "C4", "C5"]
        assert len(hole.layers) == 3
        assert hole.total_depth.value == 17.4
        (entry,) = [e for e in record.qa if e.where == "BH-6.sheets"]
        assert "sheet 2 of 2" in entry.detail and entry.rank == "changed"
        assert any("17.4 m kept as the deeper" in v for v in entry.values)
        assert any("printed again" in v for v in entry.values)
        # the reason it matters: a test on the second sheet finds its sample
        assert record.lab_tests[0].linked_sample_id == "C5"
        assert record.narrative.found_counts["borings"] == 1

    def test_with_no_sheet_numbers_the_depths_must_carry_on(self):
        first = _sheet("6", [44], samples=[("", d, d + 0.45, "spt")
                                           for d in (1.0, 9.0, 18.0)])
        later = _sheet("6", [45], samples=[("", d, d + 0.45, "spt")
                                           for d in (17.2, 21.0, 24.5)])
        (hole,) = _folded(first, later).investigations
        assert hole.pages == [44, 45] and len(hole.samples) == 6

    def test_a_second_log_from_the_ground_surface_is_another_hole(self):
        first = _sheet("B-1", [10], samples=[("", 1.5, 2.0, "spt"),
                                             ("", 9.0, 9.5, "spt")])
        later = _sheet("B-1", [11], samples=[("", 0.5, 1.0, "spt"),
                                             ("", 20.0, 20.5, "spt")])
        assert len(_folded(first, later).investigations) == 2

    def test_a_first_sheet_or_a_far_page_or_another_unit_is_not_folded(self):
        a = dict(samples=[("", 1.0, 1.5, "spt"), ("", 9.0, 9.5, "spt")])
        b = dict(samples=[("", 10.0, 10.5, "spt"), ("", 14.0, 14.5, "spt")])
        cases = [
            (_sheet("B-1", [10], "1 of 2", **a),
             _sheet("B-1", [11], "1 of 2", **b)),         # a new log
            (_sheet("B-1", [10], **a), _sheet("B-1", [14], **b)),  # far
            (_sheet("B-1", [10], **a),
             _sheet("B-1", [11], unit="ft", **b)),         # another unit
            (_sheet("", [10], **a), _sheet("", [11], **b)),  # no name
        ]
        for first, later in cases:
            assert len(_folded(first, later).investigations) == 2

    def test_a_sheet_two_printing_only_the_number_is_the_same_hole(self):
        first = _sheet("BH No. 8", [90], "1 of 2", td=8.6,
                       samples=[("", 1.0, 1.5, "spt")])
        later = _sheet("8", [91], "2 of 2", td=13.2)
        record = _folded(first, later)
        (hole,) = record.investigations
        assert hole.investigation_id == "BH No. 8"
        (entry,) = [e for e in record.qa if e.where.endswith(".sheets")]
        assert "printed '8'" in entry.detail
        # another series' number is not the same hole
        other = _folded(_sheet("B-5", [60], "1 of 2"),
                        _sheet("TP-5", [61], "2 of 2"))
        assert len(other.investigations) == 2

    def test_three_sheets_read_as_three_logs_are_one_hole(self):
        sheets = [_sheet("B-9", [p], f"{k} of 3",
                         samples=[("", t, t + 0.45, "spt")])
                  for k, p, t in ((1, 40, 1.0), (2, 41, 12.0),
                                  (3, 42, 24.0))]
        record = _folded(*sheets)
        (hole,) = record.investigations
        assert hole.pages == [40, 41, 42] and len(hole.samples) == 3
        (entry,) = [e for e in record.qa if e.where == "B-9.sheets"]
        assert "read as 3 logs" in entry.detail

    def test_a_restated_row_that_prints_something_else_is_kept_twice(self):
        first = _sheet("B-1", [10], "1 of 2",
                       samples=[("S-4", 9.5, 11.0, "spt")])
        later = _sheet("B-1", [11], "2 of 2",
                       samples=[("S-5", 9.5, 11.0, "spt")])
        (hole,) = _folded(first, later).investigations
        assert [s.sample_id for s in hole.samples] == ["S-4", "S-5"]

    def test_a_test_pit_then_a_boring_of_one_name_are_two_holes(self):
        first = _sheet("TP-1", [10], "1 of 2", kind="test_pit",
                       samples=[("", 0.5, 1.0, "bulk")])
        later = _sheet("TP-1", [11], "2 of 2",
                       samples=[("", 3.0, 3.45, "spt")])
        assert len(_folded(first, later).investigations) == 2
        # nor with no sheet numbers, where the depths carry on
        first = _sheet("1", [10], kind="test_pit",
                       samples=[("", 0.5, 1.0, "bulk"), ("", 2.0, 2.5, "bulk")])
        later = _sheet("1", [11], samples=[("", 2.4, 2.85, "spt"),
                                           ("", 6.0, 6.45, "spt")])
        assert len(_folded(first, later).investigations) == 2

    def test_a_sounding_is_never_folded(self):
        for kind in ("cpt", "dcp"):
            first = _sheet("C-1", [20], "1 of 2", kind=kind, layers=[
                (0.0, 8.0, "")])
            later = _sheet("C-1", [21], "2 of 2", kind=kind, layers=[
                (8.0, 16.0, "")])
            assert len(_folded(first, later).investigations) == 2

    def test_a_sheet_two_whose_depths_start_again_says_so(self):
        # The printed number joins the sheets; the depths do not carry on,
        # and the note says it for the person checking the fold.
        def first():
            return _sheet("B-4", [30], "1 of 2", samples=[
                ("", 1.0, 1.45, "spt"), ("", 6.0, 6.45, "spt")])
        later = _sheet("B-4", [31], "2 of 2",
                       samples=[("", 0.5, 0.95, "spt"), ("", 4.0, 4.45, "spt")])
        record = _folded(first(), later)
        assert len(record.investigations) == 1
        (entry,) = [e for e in record.qa if e.where == "B-4.sheets"]
        assert ("it prints sheet 2 of 2, but its depths (0.50-4.45 m) start "
                "above where the first's (1.00-6.45 m) leave off"
                in entry.detail)
        assert entry.rank == "open"      # a person should look at the pages
        # one that carries on does not say it
        later = _sheet("B-4", [31], "2 of 2",
                       samples=[("", 7.5, 7.95, "spt")])
        (entry,) = [e for e in _folded(first(), later).qa
                    if e.where == "B-4.sheets"]
        assert "start above" not in entry.detail and entry.rank == "changed"


class TestTheSummaryRowToSheetMatch:
    """A sheet is checked against a summary row of the same hole, B-1 or B-01."""

    def test_a_zero_padded_sheet_is_the_rows_hole(self):
        from report_ingest.reconciler import _sheets_for
        row = summary_table().result.rows[0]            # B-1, S-1, 2 ft
        padded = atterberg(investigation="B-01")
        other = atterberg(investigation="B-10")

        assert _sheets_for(row, [padded, other]) == [padded]
