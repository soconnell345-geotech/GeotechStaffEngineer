"""RI-140: a summary-table row joins a laboratory sheet only where the two
report on one specimen. The match by hole and depth window alone joined an
index-test row to another sample's chemistry sheet in the same hole, so the
cross-check compared nothing and the checklist took the row's missing
Atterberg sheet for found. Now a row and a sheet join only when they share a
test (the sheet's kind is one the row's columns name, or both carry one
value), or a sample both name alike; of the sheets of one kind the closest
interval wins, and a tie is said, not guessed.

Made-up values only: no hole, sample, depth or number here is from a report.
"""

from __future__ import annotations

from report_ingest import checklist as CL
from report_ingest.model import (
    AtterbergResult, CBRResult, ChemicalResult, ConsolidationResult,
    DocumentFacts, GradationResult, Investigation, LabTest, Layer,
    MoistureDensityResult, OtherResult, Project, Quantity, ReportRecord,
    Sample, StrengthResult, SummaryRow, SummaryTableResult,
)
from report_ingest.reconciler import (
    _heading_kinds, _names_no_specimen, _reading, _sample_keys, _sheets_for,
    _summary_matches, reconcile,
)


def _m(value):
    return Quantity(value=value, unit="m")


def _ppm(value):
    return Quantity(value=value, unit="ppm")


def _row(hole="QB-4", sample="", top=2.40, bottom=None, **cells):
    return SummaryRow(investigation_id=hole, sample_id=sample,
                      depth_top=_m(top) if top is not None else None,
                      depth_bottom=_m(bottom) if bottom is not None else None,
                      **cells)


def _atterberg(hole="QB-4", sample="", top=2.40, bottom=None, page=21,
               **values):
    return LabTest(
        kind="atterberg", investigation_id=hole, sample_id=sample,
        depth_top=_m(top), depth_bottom=_m(bottom) if bottom else None,
        pages=[page], source_report="SYN",
        result=AtterbergResult(**(values or {"ll": 47.0, "pl": 22.0,
                                             "pi": 25.0})))


def _chemical(hole="QB-4", sample="", top=2.40, page=30, **values):
    return LabTest(
        kind="chemical", investigation_id=hole, sample_id=sample,
        depth_top=_m(top), pages=[page], source_report="SYN",
        result=ChemicalResult(**(values or {"pH": 7.1,
                                            "sulfate": _ppm(118.0)})))


def _index_row(**kw):
    """A row reporting index tests only: limits and a water content."""
    cells = {"ll": 47.0, "pl": 22.0, "pi": 25.0, "wc": 18.3}
    cells.update(kw.pop("cells", {}))
    return _row(**kw, **cells)


def _summary(*rows):
    return LabTest(kind="summary_table", pages=[19], source_report="SYN",
                   result=SummaryTableResult(rows=list(rows)))


def _record(*lab):
    """A record with QB-4 and QB-9 logged; their one sample each sits far
    below the depths tested here, so nothing links a sheet to a sample."""
    holes = [Investigation(
        investigation_id=name, kind="boring", depth_unit="m",
        total_depth=_m(15.0),
        layers=[Layer(top=_m(0.0), bottom=_m(15.0),
                      description="Brown sandy CLAY", uscs="CL")],
        samples=[Sample(sample_id="T-20", top=_m(11.0), bottom=_m(11.45),
                        kind="spt")],
        pages=[5], source_report="SYN") for name in ("QB-4", "QB-9")]
    return ReportRecord(document=DocumentFacts(report_id="SYN", n_pages=40),
                        project=Project(name="Made-up Yard"),
                        investigations=holes, lab_tests=list(lab))


def _match(rows, sheets):
    joined, ties = _summary_matches(rows, sheets)
    return joined, ties


def _check(record):
    rows = {r["id"]: r for r in CL.run_checklist(record)["code_checks"]}
    return rows["completeness.summary_and_sheets"]


def _table_notes(record):
    return [e for e in record.qa
            if e.where.startswith("lab_tests.summary_table at")]


class TestASharedTest:

    def test_an_index_row_and_a_chemistry_sheet_at_its_depth_stay_apart(self):
        row, chem = _index_row(), _chemical()
        joined, ties = _match([row], [chem])

        assert joined == [[]] and ties == []
        assert _sheets_for(row, [chem]) == []

    def test_placeholder_cells_do_not_make_a_shared_test(self):
        # The table prints dashes and n/a in its chemistry columns.
        row = _index_row(cells={"pH": "--", "sulfate": "n/a"})
        assert _match([row], [_chemical()]) == ([[]], [])

    def test_the_checklist_lists_the_row_whose_sheet_is_missing(self):
        record = reconcile(_record(_chemical(), _summary(_index_row())))
        sh = _check(record)

        assert sh["status"] == "fail", sh
        # RI-142: said by the tests the row tabulates (its water content is
        # a note: labs often report it on the summary only).
        assert ("summary rows short of a sheet: QB-4 at 2.4 m (no atterberg "
                "sheet)") in sh["detail"]
        assert not [e for e in record.qa if e.kind == "conflict"]

    def test_a_row_joins_its_atterberg_sheet_and_not_the_chemistry(self):
        row, att, chem = _index_row(), _atterberg(), _chemical()
        joined, ties = _match([row], [chem, att])

        assert joined == [[att]] and ties == []

    def test_a_disagreeing_value_on_the_joined_sheet_is_a_conflict(self):
        row = _index_row(cells={"ll": 52.0})
        record = reconcile(_record(_chemical(), _atterberg(),
                                   _summary(row)))

        (conflict,) = [e for e in record.qa if e.kind == "conflict"]
        assert conflict.where == "lab_tests.ll at QB-4 2.4 m"
        assert conflict.values == ["summary table 52", "atterberg sheet 47"]
        assert sorted(conflict.pages) == [19, 21]

    def test_a_non_plastic_row_joins_its_sheet_that_prints_no_limits(self):
        row = _row(pl="NP")
        att = _atterberg(non_plastic=True)
        assert _match([row], [att, _chemical()]) == ([[att]], [])

    def test_a_value_both_carry_joins_them(self):
        # A chemistry sheet printing the water content beside its results:
        # the row's water content is a test the two have in common.
        row = _row(wc=18.3)
        chem = _chemical(pH=7.1, wc=21.6)
        assert _match([row], [chem]) == ([[chem]], [])

    def test_the_same_sample_named_joins_without_a_shared_column(self):
        row = _row(sample="T-3", ll=47.0)
        chem = _chemical(sample="T-3")
        assert _match([row], [chem]) == ([[chem]], [])


class TestTheClosestInterval:

    def test_the_exact_interval_beats_a_window_match(self):
        row = _index_row(top=2.40, bottom=2.85)
        exact = _atterberg(top=2.40, bottom=2.85, page=21)
        near = _atterberg(top=2.50, bottom=2.95, page=22)
        joined, ties = _match([row], [near, exact])

        assert joined == [[exact]] and ties == []

    def test_the_nearer_top_wins_when_neither_is_exact(self):
        row = _index_row(top=2.40)
        nearer = _atterberg(top=2.47, page=21)
        farther = _atterberg(top=2.52, page=22)
        assert _match([row], [farther, nearer]) == ([[nearer]], [])

    def test_a_sheet_stays_with_the_row_whose_interval_is_its_own(self):
        own = _index_row(top=3.00)
        next_row = _index_row(top=3.10)
        att = _atterberg(top=3.00)
        joined, ties = _match([own, next_row], [att])

        assert joined == [[att], []] and ties == []
        assert _sheets_for(next_row, [att], rows=[own, next_row]) == []
        assert _sheets_for(next_row, [att]) == [att]     # the row alone

    def test_each_kind_picks_its_own_closest(self):
        row = _row(top=2.40, ll=47.0, pH=7.1)
        att = _atterberg(top=2.40)
        chem_near = _chemical(top=2.45)
        chem_far = _chemical(top=2.52, page=31)
        joined, _ties = _match([row], [chem_far, att, chem_near])

        assert {id(s) for s in joined[0]} == {id(att), id(chem_near)}

    def test_one_test_read_as_several_all_join(self):
        # One specimen's Atterberg test read as two (a test over two pages):
        # nothing tells them apart, so both are the row's.
        row = _index_row()
        first = _atterberg(sample="T-3", page=21)
        second = _atterberg(sample="T-3", page=22)
        joined, ties = _match([row], [first, second])

        assert joined == [[first, second]] and ties == []


class TestATie:

    def test_two_specimens_as_close_join_neither(self):
        row = _index_row(top=2.40)
        upper = _atterberg(sample="T-3", top=2.30, page=21)
        lower = _atterberg(sample="T-4", top=2.50, page=22)
        joined, ties = _match([row], [upper, lower])

        assert joined == [[]]
        ((t_rows, t_sheets),) = ties
        assert t_rows == [row]
        assert {id(s) for s in t_sheets} == {id(upper), id(lower)}

    def test_a_sample_both_name_breaks_the_tie(self):
        row = _index_row(sample="T-4", top=2.40)
        upper = _atterberg(sample="T-3", top=2.30, page=21)
        lower = _atterberg(sample="T-4", top=2.50, page=22)
        assert _match([row], [upper, lower]) == ([[lower]], [])

    def test_a_sheet_as_close_to_two_rows_joins_neither(self):
        upper = _index_row(sample="T-3", top=2.30)
        lower = _index_row(sample="T-4", top=2.50)
        att = _atterberg(top=2.40)
        joined, ties = _match([upper, lower], [att])

        assert joined == [[], []]
        ((t_rows, t_sheets),) = ties
        assert t_sheets == [att]
        assert {id(r) for r in t_rows} == {id(upper), id(lower)}

    def test_the_tie_is_a_note_and_no_conflict(self):
        row = _index_row(top=2.40, cells={"ll": 60.0})
        record = reconcile(_record(
            _atterberg(sample="T-3", top=2.30, page=21),
            _atterberg(sample="T-4", top=2.50, page=22),
            _summary(row)))

        assert not [e for e in record.qa if e.kind == "conflict"]
        (note,) = _table_notes(record)
        assert (note.kind, note.rank) == ("note", "open")
        assert note.where == "lab_tests.summary_table at QB-4 2.4 m"
        assert note.values[0] == "atterberg sheet QB-4 T-3 at 2.3 m"
        assert "2 atterberg" in note.detail
        assert "cross-checked against none of them" in note.detail
        assert sorted(note.pages) == [19, 21, 22]

    def test_the_checklist_is_unsure_on_a_tie_alone(self):
        record = reconcile(_record(
            _atterberg(sample="T-3", top=2.30, page=21),
            _atterberg(sample="T-4", top=2.50, page=22),
            _summary(_index_row(top=2.40))))
        sh = _check(record)

        assert sh["status"] == "unsure", sh
        # No row is listed as missing a sheet; the row's water content, on
        # no sheet, is a note (RI-142).
        assert "short of a sheet" not in sh["detail"]
        assert "naming no test, with no sheet" not in sh["detail"]
        assert "1 summary row(s) tabulate water content with no sheet" in \
            sh["detail"]
        assert "not on the summary" not in sh["detail"]
        assert ("as close to more than one specimen, not matched (look at "
                "the pages): QB-4 at 2.4 m (2 sheets: atterberg)"
                in sh["detail"])

    def test_a_row_with_no_hole_among_sheets_of_two_holes_is_a_tie(self):
        row = _index_row(hole="", top=2.40)
        here = _atterberg(hole="QB-4", top=2.40, page=21)
        there = _atterberg(hole="QB-9", top=2.40, page=22)
        joined, ties = _match([row], [here, there])

        assert joined == [[]]
        ((t_rows, t_sheets),) = ties
        assert {id(s) for s in t_sheets} == {id(here), id(there)}

    def test_its_sample_picks_the_hole(self):
        row = _index_row(hole="", sample="L-61", top=2.40)
        here = _atterberg(hole="QB-4", sample="L-61", top=2.40, page=21)
        there = _atterberg(hole="QB-9", sample="L-75", top=2.40, page=22)
        assert _match([row], [here, there]) == ([[here]], [])

    def test_a_row_with_a_hole_only_takes_a_sheet_of_one_specimen(self):
        row = _row(top=None, ll=47.0)
        only = _atterberg(top=2.40)
        assert _match([row], [only]) == ([[only]], [])
        two = [_atterberg(top=2.40), _atterberg(top=4.10, page=22)]
        joined, ties = _match([row], two)
        assert joined == [[]] and len(ties) == 1


class TestARowNamingNoSpecimen:

    def test_it_joins_nothing_and_is_no_tie(self):
        row = _row(hole="", top=None, ll=47.0)
        assert _names_no_specimen(row)
        assert _match([row], [_atterberg(), _atterberg(hole="QB-9")]) == \
            ([[]], [])

    def test_the_checklist_counts_it_apart(self):
        record = reconcile(_record(
            _atterberg(),
            _summary(_index_row(), _row(hole="", top=None, ll=47.0))))
        sh = _check(record)

        assert sh["status"] == "pass", sh
        assert sh["detail"].startswith("all 1 summary rows have a sheet")
        assert ("1 summary row(s) print no hole, sample or depth and are "
                "matched to no sheet") in sh["detail"]


class TestSampleNames:

    def test_the_printed_and_the_linked_name_both_count(self):
        sheet = _atterberg(sample="S-5")
        sheet.linked_sample_id = "5"
        assert _sample_keys(sheet) == {"S5", "5"}
        assert _match([_row(sample="S-5", ll=47.0)], [sheet]) == \
            ([[sheet]], [])

    def test_the_hole_at_the_head_of_a_label_is_dropped(self):
        assert _sample_keys(_atterberg(sample="QB-4, T-03")) == {"T3"}


def _sheet(kind, result, sample="", top=2.40, bottom=None, page=40):
    return LabTest(kind=kind, investigation_id="QB-4", sample_id=sample,
                   depth_top=_m(top),
                   depth_bottom=_m(bottom) if bottom is not None else None,
                   pages=[page], source_report="SYN", result=result)


def _kpa(value):
    return Quantity(value=value, unit="kPa")


def _kn(value):
    return Quantity(value=value, unit="kN/m3")


class TestTheTestsAColumnNames:

    def test_a_qu_names_a_uu_triaxial_sheet_that_reports_su(self):
        row = _row(qu=_kpa(164.0))
        uu = _sheet("triaxial", StrengthResult(kind="triaxial",
                                               test_type="UU", su=_kpa(82.0)))
        assert _match([row], [uu]) == ([[uu]], [])

    def test_a_uscs_symbol_names_the_limits_and_the_grading(self):
        row = _row(uscs="CL")
        att = _atterberg()
        grad = _sheet("gradation", GradationResult(fines_percent=58.0),
                      page=41)
        chem = _chemical()
        joined, _ties = _match([row], [chem, att, grad])
        assert [s.kind for s in joined[0]] == ["atterberg", "gradation"]

    def test_headings_kept_in_other_name_their_tests(self):
        assert _heading_kinds("CBR (%)") == {"cbr"}
        assert _heading_kinds("Gs") == {"specific_gravity"}
        assert _heading_kinds("Specific Gravity") == {"specific_gravity"}
        assert _heading_kinds("k (cm/s)") == {"permeability"}
        assert _heading_kinds("Permeability") == {"permeability"}
        assert _heading_kinds("p'c (kPa)") == {"swell_consolidation"}
        assert _heading_kinds("Cc") == {"swell_consolidation"}
        assert _heading_kinds("Cr") == {"swell_consolidation"}
        assert _heading_kinds("Swell pressure") == {"swell_consolidation",
                                                    "cbr"}
        assert _heading_kinds("C.B.R. % 4 day soak") == {"cbr"}
        assert _heading_kinds("kPa") == set()
        assert _heading_kinds("Remarks") == set()

    def test_an_other_column_joins_its_sheet(self):
        cases = [
            ("CBR", "11", _sheet("cbr", CBRResult(cbr_percent=11.0))),
            ("Gs", "2.68", _sheet("specific_gravity", OtherResult(
                kind="specific_gravity", fields={"Gs": 2.68}))),
            ("k (cm/s)", "3E-6", _sheet("permeability", OtherResult(
                kind="permeability", fields={"k": "3E-6"}))),
            ("Cc", "0.27", _sheet("swell_consolidation", ConsolidationResult(
                test_type="consolidation", cc=0.27))),
        ]
        for heading, cell, sheet in cases:
            row = _row(other=[(heading, cell)])
            assert _match([row], [sheet]) == ([[sheet]], []), heading
            blank = _row(other=[(heading, "--")])
            assert _match([blank], [_sheet(sheet.kind, sheet.result,
                                           top=2.48)]) == ([[]], []), heading

    def test_a_dry_density_meets_a_consolidation_sheets_dry_unit_weight(self):
        row = _row(dry_density=_kn(15.9))
        oed = _sheet("swell_consolidation", ConsolidationResult(
            test_type="consolidation", cc=0.27, dry_unit_weight=_kn(15.9)))
        assert _match([row], [oed]) == ([[oed]], [])

    def test_a_row_naming_no_test_keeps_its_exact_interval_sheet(self):
        row = _row(top=2.40, bottom=2.85, description="Grey silty CLAY")
        exact = _atterberg(top=2.40, bottom=2.85, page=21)
        window = _chemical(top=2.48, page=30)
        joined, ties = _match([row], [window, exact])
        assert joined == [[exact]] and ties == []


class TestOneSampleSeveralSheets:

    def test_a_hydrometer_sheet_printing_the_top_only_joins_its_sieve(self):
        row = _row(sample="T-3", top=2.40, bottom=2.85, fines_percent=58.0)
        sieve = _sheet("gradation", GradationResult(fines_percent=58.0),
                       sample="T-3", top=2.40, bottom=2.85, page=50)
        hydro = _sheet("gradation", GradationResult(hydrometer=True,
                                                    clay_percent=21.0),
                       sample="T-3", top=2.40, page=58)
        joined, ties = _match([row], [sieve, hydro])
        assert joined == [[sieve, hydro]] and ties == []

    def test_a_sheet_of_another_interval_stays_out(self):
        row = _row(sample="T-3", top=2.40, bottom=2.85, fines_percent=58.0)
        sieve = _sheet("gradation", GradationResult(fines_percent=58.0),
                       sample="T-3", top=2.40, bottom=2.85, page=50)
        other = _sheet("gradation", GradationResult(fines_percent=40.0),
                       sample="T-3", top=2.50, page=58)
        assert _match([row], [sieve, other]) == ([[sieve]], [])


class TestAValueAlone:

    def test_another_samples_sheet_printing_a_shared_value_stays_apart(self):
        # The row's own sheets are missing; a chemistry sheet of the next
        # sample, a few centimetres off, prints a water content too.
        row = _index_row(sample="T-3")
        chem = _chemical(sample="T-4", top=2.49, pH=6.9, wc=24.0)
        assert _match([row], [chem]) == ([[]], [])
        record = reconcile(_record(chem, _summary(row)))
        assert not [e for e in record.qa if e.kind == "conflict"]
        assert ("summary rows short of a sheet: QB-4 T-3 at 2.4 m (no "
                "atterberg sheet)") in _check(record)["detail"]

    def test_a_shared_value_joins_where_a_side_prints_no_sample(self):
        row = _index_row()
        chem = _chemical(sample="T-4", top=2.49, pH=6.9, wc=24.0)
        assert _match([row], [chem]) == ([[chem]], [])


class TestTheRowThatSharesATest:

    def test_a_sheet_goes_to_the_row_that_tabulates_its_test(self):
        # The limits row names the sample; the grading row, the interval.
        limits = _row(sample="T-3", top=2.40, ll=47.0)
        grading = _row(top=2.40, bottom=2.85, fines_percent=58.0)
        att = _atterberg(sample="T-3", page=21)
        grad = _sheet("gradation", GradationResult(fines_percent=61.0),
                      sample="T-3", top=2.40, page=50)
        joined, ties = _match([limits, grading], [att, grad])
        assert joined == [[att], [grad]] and ties == []


class TestTheChecklistsTabulatedKinds:

    def test_a_chemistry_sheet_beside_an_index_table_is_not_missing(self):
        record = reconcile(_record(
            _atterberg(sample="T-3"),
            _sheet("moisture_content", MoistureDensityResult(wc=18.3),
                   sample="T-3", page=22),
            _chemical(sample="T-4", top=2.49, pH=6.9, wc=24.0),
            _summary(_index_row(sample="T-3"))))
        sh = _check(record)

        assert sh["status"] == "pass", sh
        assert "not on the summary" not in sh["detail"]
        assert ("sheets of kinds the summary does not tabulate, not counted: "
                "chemical") in sh["detail"]


class TestEmptyCells:

    def test_spelled_out_not_tested_is_no_reading(self):
        for cell in ("not tested", "Not Tested", "n.t.", "N/T", "not run",
                     "n/a", "--"):
            assert not _reading(cell), cell
        for cell in ("NP", "ND", "<10", "Nil", 0.0):
            assert _reading(cell), cell

    def test_a_not_tested_cell_names_no_test(self):
        row = _index_row(cells={"pH": "Not tested", "sulfate": "not run"})
        assert _match([row], [_chemical()]) == ([[]], [])
