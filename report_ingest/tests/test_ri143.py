"""RI-143: summary rows and sheets the match could not place.

Four gaps on saved records, each fixed in general terms:

1. a summary row that prints no boring of its own and names its hole in
   another column ("Site ref.", "Location") is placed in that hole, and
   rows under a cell spanning them take the hole of the row above;
2. a sheet that prints its depth with no unit is read in the record's one
   depth unit, and one that prints its depth only in its sample label is
   read from there, to link it and to match it to a row;
3. a chemistry sheet's percent solids is no water content, in the lab
   floor's vocabulary and in the reader's gate;
4. bracketed units no longer name a test (a resistivity's ``kΩ``), two bare
   sample numbers of one numbering are two specimens, an unlogged hole is
   matched by its own name after a word for an exploration, and a boring
   printed as a bare number is the one hole the logs give that number.

The reviews' fixes: a label naming two holes links neither, whatever word
heads it, by name or by the second's number; a joined number no logged
hole carries is the sample's; a bracket's symbol beside its unit ("(k,
m/s)") still names its test, and a resistance's prefix or an element's
symbol names none; one numbering is off where a row prints a number a sheet
of another specimen prints; blank rows ending above another hole deeper
down are left unplaced only where that hole has samples at their depths;
the solids gate refuses a water content equal to the percent solids except
at the two values the definitions allow and, under ``flag``, keeps it as
doubtful; the checklist reads depths as the match does, and a kind the
record holds no sheet of at all is a note, unsure where a bound report
read on its own may hold it.

Made-up values only: no hole, sample, depth or number here is from a report.
"""

from __future__ import annotations

import pytest

from report_ingest import checklist as CL
from report_ingest.lab_floor import (
    SOLIDS_EQUAL_TO_WATER, _field_of, names_solids, solids_can_equal_water,
)
from report_ingest.lab_reader import (
    LabSheetReading, ReadField, ReadProv, ReadReported, ReadTest, _Builder,
    read_lab_sheet,
)
from report_ingest.model import (
    AtterbergResult, BoundReport, ChemicalResult, DocumentFacts,
    GradationResult, Investigation, LabTest, Layer, Project, Quantity,
    ReportRecord, Sample, SummaryRow, SummaryTableResult,
)
from report_ingest.reconciler import (
    _heading_kinds, _name_after_word, _numbers_apart, _summary_matches,
    depth_unit, reconcile,
)


def _m(value):
    return Quantity(value=value, unit="m")


def _bare(value):
    return Quantity(value=value, unit="")


def _hole(name, tops=(1.0, 2.0, 3.0), kind="boring", unit="m"):
    return Investigation(
        investigation_id=name, kind=kind, depth_unit=unit,
        total_depth=Quantity(value=12.0, unit=unit),
        layers=[Layer(top=Quantity(value=0.0, unit=unit),
                      bottom=Quantity(value=12.0, unit=unit),
                      description="Olive clayey SAND", uscs="SM")],
        samples=[Sample(sample_id=f"S-{i + 1}",
                        top=Quantity(value=t, unit=unit),
                        bottom=Quantity(value=t + 0.45, unit=unit),
                        kind="spt") for i, t in enumerate(tops)],
        pages=[5], source_report="SYN")


def _record(holes, *lab):
    return ReportRecord(document=DocumentFacts(report_id="SYN", n_pages=60),
                        project=Project(name="Made-up Quay"),
                        investigations=list(holes), lab_tests=list(lab))


def _row(hole="", sample="", top=1.0, other=(), **cells):
    cells = cells or {"sand_percent": 41.0}
    return SummaryRow(investigation_id=hole, sample_id=sample,
                      depth_top=_m(top) if top is not None else None,
                      other=list(other), **cells)


def _summary(*rows):
    return LabTest(kind="summary_table", pages=[30], source_report="SYN",
                   result=SummaryTableResult(rows=list(rows)))


def _grading(hole="QX-2", sample="", top=None, bottom=None, page=33,
             sand=41.0):
    return LabTest(kind="gradation", investigation_id=hole, sample_id=sample,
                   depth_top=top, depth_bottom=bottom, pages=[page],
                   source_report="SYN",
                   result=GradationResult(sand_percent=sand,
                                          gravel_percent=3.0))


def _rows_of(record):
    table = next(t for t in record.lab_tests if t.kind == "summary_table")
    return table.result.rows


def _notes(record, words):
    return [e for e in record.qa if words in e.detail]


# -- (1) a hole named in another column --------------------------------------

class TestHoleInAnotherColumn:

    def test_a_bare_number_under_a_site_marking_heading(self):
        row = _row(sample="L-31", other=[("Site ref.", "71")])
        record = reconcile(_record([_hole("BH No. 71"), _hole("QX-2")],
                                   _summary(row)))
        (got,) = _rows_of(record)
        assert got.linked_investigation_id == "BH No. 71"
        assert got.linked_sample_id == "S-1"
        assert got.investigation_id == ""            # printed, unchanged
        (note,) = _notes(record, "name their hole in another column")
        assert note.kind == "note"
        assert "'Site ref.' '71' -> BH No. 71" in note.values

    def test_a_logged_name_under_a_location_heading(self):
        row = _row(top=2.0, other=[("Location", "QX-2")])
        record = reconcile(_record([_hole("QX-2")], _summary(row)))
        assert _rows_of(record)[0].linked_investigation_id == "QX-2"

    def test_rows_under_a_cell_spanning_them_take_the_hole_above(self):
        rows = [_row(top=1.0, other=[("Site ref.", "QX-2")]),
                _row(top=2.0), _row(top=3.0),
                _row(top=0.5)]                     # shallower: a new group
        record = reconcile(_record([_hole("QX-2")], _summary(*rows)))
        got = [r.linked_investigation_id for r in _rows_of(record)]
        assert got == ["QX-2", "QX-2", "QX-2", ""]
        (note,) = _notes(record, "name their hole in another column")
        assert "2 row(s) below them" in note.detail

    def test_no_carry_where_the_hole_has_no_sample_at_the_row_s_depth(self):
        rows = [_row(top=1.0, other=[("Site ref.", "QX-2")]),
                _row(top=7.0)]
        record = reconcile(_record([_hole("QX-2")], _summary(*rows)))
        assert [r.linked_investigation_id for r in _rows_of(record)] == \
            ["QX-2", ""]

    def test_a_row_printing_an_unread_hole_stops_the_carry(self):
        rows = [_row(top=1.0, other=[("Site ref.", "QX-2")]),
                _row(top=2.0, other=[("Site ref.", "CBR9")]),
                _row(top=3.0)]
        record = reconcile(_record([_hole("QX-2")], _summary(*rows)))
        assert [r.linked_investigation_id for r in _rows_of(record)] == \
            ["QX-2", "", ""]

    def test_a_sample_heading_names_no_hole(self):
        row = _row(other=[("Sample boring", "QX-2"), ("Lab No.", "71")])
        record = reconcile(_record([_hole("QX-2"), _hole("BH No. 71")],
                                   _summary(row)))
        assert _rows_of(record)[0].linked_investigation_id == ""

    def test_the_row_s_own_boring_comes_first(self):
        row = _row(hole="QX-2", other=[("Site ref.", "71")])
        record = reconcile(_record([_hole("QX-2"), _hole("BH No. 71")],
                                   _summary(row)))
        assert _rows_of(record)[0].linked_investigation_id == "QX-2"


class TestHoleAfterAWord:

    def test_bore_hole_no_and_a_bracketed_name(self):
        sheet = _grading(hole="BORE HOLE NO. (QX-2)", top=_m(1.0))
        record = reconcile(_record([_hole("QX-2")], sheet))
        assert record.lab_tests[0].linked_investigation_id == "QX-2"
        assert record.lab_tests[0].investigation_id == "BORE HOLE NO. (QX-2)"

    def test_a_boring_word_and_a_number_names_the_numbered_boring(self):
        sheet = _grading(hole="B/H 71, U3", top=_m(2.0))
        record = reconcile(_record(
            [_hole("71"), _hole("TP-71", kind="test_pit")], sheet))
        assert record.lab_tests[0].linked_investigation_id == "71"

    def test_a_general_word_and_a_number_two_holes_carry_is_no_link(self):
        sheet = _grading(hole="Hole 71", top=_m(2.0))
        record = reconcile(_record(
            [_hole("71"), _hole("TP-71", kind="test_pit")], sheet))
        assert record.lab_tests[0].linked_investigation_id == ""

    def test_a_bare_number_boring_is_the_hole_the_logs_number_so(self):
        row = _row(hole="64", top=2.0)
        record = reconcile(_record([_hole("BH No. 64"), _hole("QX-01")],
                                   _summary(row)))
        assert _rows_of(record)[0].linked_investigation_id == "BH No. 64"

    def test_a_bare_number_two_series_carry_names_neither(self):
        row = _row(hole="38", top=2.0)
        record = reconcile(_record([_hole("BH-38"), _hole("QX-038")],
                                   _summary(row)))
        assert _rows_of(record)[0].linked_investigation_id == ""

    def test_an_unlogged_hole_is_matched_by_its_own_name(self):
        row = _row(hole="QZ - 4", top=9.0)
        sheet = _grading(hole="BORE HOLE NO. (QZ-4)", top=_m(9.0))
        joined, ties = _summary_matches([row], [sheet])
        assert joined == [[sheet]] and ties == []

    def test_a_hole_and_a_sample_run_together_are_no_one_name(self):
        assert _name_after_word("BORE HOLE NO. (QZ-4)") == "QZ-4"
        assert _name_after_word("BH. QZ - 4") == "QZ - 4"
        assert _name_after_word("BH47, U9") == ""
        assert _name_after_word("BH 83") == ""
        assert _name_after_word("QZ-4") == ""


# -- (2) a depth printed with no unit, or only in the label -------------------

class TestDepthWithNoUnit:

    def test_the_record_s_one_unit(self):
        assert depth_unit(_record([_hole("QX-2")])) == "m"
        mixed = _record([_hole("QX-2"), _hole("QX-3", unit="ft")])
        assert depth_unit(mixed) == ""

    def test_a_unitless_sheet_links_to_its_sample_in_the_record_s_unit(self):
        sheet = _grading(top=_bare(2.0), bottom=_bare(2.45))
        record = reconcile(_record([_hole("QX-2")], sheet))
        got = record.lab_tests[0]
        assert got.linked_investigation_id == "QX-2"
        assert got.linked_sample_id == "S-2"
        assert got.depth_top.unit == ""                  # kept as printed
        (note,) = _notes(record, "print their depths with no unit")
        assert "read in m" in note.detail

    def test_a_record_of_two_units_leaves_the_depth_unread(self):
        sheet = _grading(top=_bare(2.0))
        record = reconcile(_record([_hole("QX-2"),
                                    _hole("QX-3", unit="ft")], sheet))
        assert record.lab_tests[0].linked_investigation_id == "QX-2"
        assert record.lab_tests[0].linked_sample_id == ""
        assert not _notes(record, "print their depths with no unit")

    def test_a_unitless_sheet_joins_its_row(self):
        row = _row(hole="QX-2", top=1.7)
        sheet = _grading(sample="1.7-3.2", top=_bare(1.7), bottom=_bare(3.2))
        assert _summary_matches([row], [sheet], unit="m") == ([[sheet]], [])
        assert _summary_matches([row], [sheet]) == ([[]], [])

    def test_a_depth_printed_only_in_the_sample_label(self):
        row = _row(hole="QX-2", sample="U3", top=4.0)
        sheet = _grading(sample="QX-2, U3, 4.00-4.45m")
        assert _summary_matches([row], [sheet]) == ([[sheet]], [])

    def test_a_label_depth_places_nothing_without_a_hole(self):
        row = _row(hole="QX-2", top=4.0)
        sheet = _grading(hole="", sample="4.00-4.45")
        assert _summary_matches([row], [sheet], unit="m") == ([[]], [])


# -- (3) percent solids is no water content ----------------------------------

class TestPercentSolids:

    def test_the_floor_keeps_solids_as_printed(self):
        for label in ("% Solids", "Percent solids", "Total Solids, %",
                      "Solids (%)", "Dry matter"):
            assert _field_of(label) == "as_printed", label
            assert names_solids(label), label

    def test_other_solids_headings_keep_their_meaning(self):
        assert _field_of("Specific gravity of the solids") != "as_printed"
        assert not names_solids("Specific gravity of the solids")
        assert not names_solids("Total dissolved solids")
        assert _field_of("Moisture content (%)") == "wc"

    def test_a_water_content_a_resistivity_test_ran_at(self):
        assert _field_of("Water content at minimum resistivity (%)") == \
            "as_printed"
        assert _field_of("Moisture content, as received (%)") == "wc"

    def _built(self, fields, wc=81.0):
        reading = LabSheetReading(
            depth_unit="m", pages_read=[40],
            tests=[ReadTest(kind="chemical", investigation_id="QX-2",
                            sample_id="S-3", depth_top=3.0, wc=wc,
                            pH=ReadReported(value=7.4),
                            fields=[ReadField(name=n, value=v)
                                    for n, v in fields],
                            prov=ReadProv(page=40))])
        builder = _Builder(reading, [40], "R-made-up", [])
        (test,) = builder.build()
        assert isinstance(test.result, ChemicalResult)
        return test.result, builder.unresolved

    def test_the_reader_refuses_the_solids_number_as_a_water_content(self):
        result, unresolved = self._built([("% Solids", "81.0")])
        assert result.wc is None
        assert any("share of dry solids" in u["why"] for u in unresolved)

    def test_a_water_content_the_sheet_also_prints_as_one_stays(self):
        result, _u = self._built([("% Solids", "81.0"),
                                  ("Moisture content", "81.0")])
        assert result.wc == 81.0

    def test_a_water_content_beside_another_solids_number_stays(self):
        result, unresolved = self._built([("% Solids", "73.91")], wc=18.2)
        assert result.wc == 18.2
        assert not any("dry solids" in u["why"] for u in unresolved)


# -- (4) headings, numbers and the cross-check -------------------------------

class TestHeadingsAndNumbers:

    def test_a_bracketed_unit_names_no_test(self):
        assert "permeability" not in _heading_kinds(
            "Minimum Resistivity (kΩ - cm)")
        assert "permeability" in _heading_kinds("K (cm/s)")
        assert "permeability" in _heading_kinds("Permeability (k)")

    def test_two_numbers_of_one_numbering_are_two_specimens(self):
        rows = [_row(hole="QX-2", sample="8807", top=1.0),
                _row(hole="QX-2", sample="8819", top=2.0)]
        sheets = [_grading(sample="8814", top=_m(1.0)),
                  _grading(sample="8819", top=_m(2.0))]
        assert _numbers_apart(rows[0], sheets[0])
        joined, ties = _summary_matches(rows, sheets)
        assert joined == [[], [sheets[1]]] and ties == []

    def test_numbers_of_two_numberings_say_nothing(self):
        rows = [_row(hole="QX-2", sample="7", top=1.0)]
        sheets = [_grading(sample="8814", top=_m(1.0))]
        assert _summary_matches(rows, sheets) == ([sheets], [])

    def test_a_name_with_a_letter_is_no_bare_number(self):
        assert not _numbers_apart(_row(sample="S-5"), _grading(sample="5"))

    def test_a_row_with_no_hole_is_not_placed_by_the_numbers(self):
        # The numbered sheets are another hole's specimens; the row prints
        # no hole, so the sheet left over is still not known to be its own.
        rows = [_row(hole="QX-2", sample="8814", top=1.0),
                _row(hole="", sample="8816", top=1.0)]
        sheets = [_grading(sample="8814", top=_m(1.0)),
                  _grading(hole="QX-3", sample="QX-3 S-1", top=_m(1.0))]
        joined, ties = _summary_matches(rows, sheets)
        assert joined == [[sheets[0]], []]
        assert ties == [([rows[1]], sheets)]


# -- review fixes: composite labels, brackets, numberings, runs, solids -----

def _linked(holes, boring="", sample=""):
    sheet = _grading(hole=boring, sample=sample, top=_m(2.0))
    record = reconcile(_record(holes, sheet))
    return record.lab_tests[0].linked_investigation_id


class TestCompositeLabels:

    def test_a_label_naming_two_holes_of_one_series_links_neither(self):
        bh = [_hole("BH-47"), _hole("BH-48")]
        tp = [_hole("TP-47", kind="test_pit"),
              _hole("TP-48", kind="test_pit")]
        assert _linked(bh, "BH-47 / BH-48 composite") == ""
        assert _linked(bh, "BH-47 & BH-48") == ""
        assert _linked(bh, "BH-47 to BH-48 composite") == ""
        assert _linked(tp, "TP-47, TP-48") == ""
        assert _linked(tp, "", "TP-47/TP-48 composite") == ""

    def test_a_second_hole_printed_as_its_number_links_neither(self):
        bh = [_hole("BH-47"), _hole("BH-48")]
        tp = [_hole("TP-47", kind="test_pit"),
              _hole("TP-48", kind="test_pit")]
        assert _linked(bh, "BH-47/48 composite") == ""
        assert _linked(bh, "BH-47 & 48") == ""
        assert _linked(tp, "TP-47, 48") == ""
        assert _linked(tp, "", "TP-47/48 composite") == ""

    def test_a_hole_of_another_series_after_it_links_neither(self):
        holes = [_hole("BH-47"), _hole("TP-71", kind="test_pit")]
        assert _linked(holes, "BH-47 / TP-71 composite") == ""

    def test_a_joined_number_no_logged_hole_carries_is_the_sample(self):
        bh = [_hole("BH-47"), _hole("BH-48")]
        assert _linked(bh, "BH-47, 3") == "BH-47"
        assert _linked(bh, "BH-47 / 12") == "BH-47"
        assert _linked(bh, "BH-47, 2.0 m") == "BH-47"

    def test_numbered_logs_named_twice_after_a_word_link_neither(self):
        holes = [_hole("47"), _hole("48")]
        assert _linked(holes, "BH 47 / BH 48 composite") == ""
        assert _linked(holes, "Boring 47 & 48") == ""
        assert _linked(holes, "BH 47 / 48") == ""

    def test_one_hole_after_a_word_still_links(self):
        assert _linked([_hole("47"), _hole("48")], "BH 47, U9") == "47"
        assert _linked([_hole("47"), _hole("48")], "Boring 47") == "47"
        assert _linked([_hole("KW-12"), _hole("KW-13")],
                       "BORE HOLE NO. (KW-12)") == "KW-12"
        assert _linked([_hole("KW-12"), _hole("KW-13")],
                       "BORE HOLE NO. (KW-12) / (KW-13)") == ""

    def test_a_hole_after_a_word_then_its_sample_s_number_links(self):
        # No hole 3 or 12 is logged: the number is the sample's.
        holes = [_hole("47"), _hole("48")]
        for label in ("BH 47, 3", "BH 47, 12", "BH 47 / 3", "BH 47, 2.0 m",
                      "BH 47, 2-2.45"):
            assert _linked(holes, label) == "47", label


class TestBracketsInHeadings:

    def test_a_symbol_beside_its_unit_names_the_test(self):
        for heading, kind in (
                ("Hydraulic conductivity (k, m/s)", "permeability"),
                ("Coefficient (k, cm/s)", "permeability"),
                ("Perm. (k, ft/day)", "permeability"),
                ("Bearing ratio (CBR, %)", "cbr"),
                ("Soaked (CBR %)", "cbr"),
                ("Particle density (Gs, Mg/m3)", "specific_gravity"),
                ("Compression (Cc, -)", "swell_consolidation")):
            assert kind in _heading_kinds(heading), heading

    def test_units_and_concentrations_name_no_test(self):
        for heading in ("Minimum resistivity (kΩ - cm)",
                        "Resistivity (k ohm-cm)", "Resistivity (kΩ.cm)",
                        "Dry density (kN/m3)", "Potassium (K, mg/L)",
                        "Chromium (Cr, mg/kg)"):
            assert _heading_kinds(heading) == set(), heading

    def test_a_resistance_spelt_with_a_hyphen_names_no_test(self):
        for heading in ("Minimum resistivity (k-ohm cm)",
                        "Resistivity (K-Ohm-cm)", "Resistivity (kohm-cm)",
                        "Resistivity (k ohm cm)"):
            assert _heading_kinds(heading) == set(), heading

    def test_an_element_s_symbol_names_no_test(self):
        for heading in ("Potassium (K)", "Potassium (K) mg/kg",
                        "Chromium (Cr)", "Potassium, K"):
            assert _heading_kinds(heading) == set(), heading
        assert "permeability" in _heading_kinds("Permeability (k)")
        assert "permeability" in _heading_kinds("K (cm/s)")


class TestNumberingThatStartsAgain:

    def test_sheets_numbering_each_hole_afresh_are_another_numbering(self):
        # The table numbers its specimens 1 to 6 down two holes; each
        # hole's sheets number theirs 1 to 3. They agree in QX-2 only, and
        # QX-3's rows still join their sheets.
        rows = [_row(hole="QX-2", sample=str(n), top=float(n))
                for n in (1, 2, 3)]
        rows += [_row(hole="QX-3", sample=str(n + 3), top=float(n))
                 for n in (1, 2, 3)]
        sheets = [_grading(hole=hole, sample=str(n), top=_m(float(n)))
                  for hole in ("QX-2", "QX-3") for n in (1, 2, 3)]
        joined, ties = _summary_matches(rows, sheets)
        assert joined == [[s] for s in sheets] and ties == []

    def test_a_number_on_another_specimen_of_the_hole_is_two_numberings(self):
        # As above, but QX-3's sheets number theirs 5 to 7: no sheet
        # number repeats, yet the table's 5 and 6 are sheets' numbers at
        # other depths of QX-3.
        rows = [_row(hole="QX-2", sample=str(n), top=float(n))
                for n in (1, 2, 3)]
        rows += [_row(hole="QX-3", sample=str(n + 3), top=float(n))
                 for n in (1, 2, 3)]
        sheets = [_grading(hole="QX-2", sample=str(n), top=_m(float(n)))
                  for n in (1, 2, 3)]
        sheets += [_grading(hole="QX-3", sample=str(n + 4),
                            top=_m(float(n))) for n in (1, 2, 3)]
        joined, ties = _summary_matches(rows, sheets)
        assert joined == [[s] for s in sheets] and ties == []

    def test_one_pair_agreeing_by_chance_is_no_one_numbering(self):
        # QX-2's row and sheet at 1.0 both print 3; the table's 4 and 5 are
        # sheets' numbers in the other hole.
        rows = [_row(hole="QX-2", sample="3", top=1.0),
                _row(hole="QX-2", sample="4", top=2.0),
                _row(hole="QX-3", sample="5", top=1.0),
                _row(hole="QX-3", sample="6", top=2.0)]
        sheets = [_grading(hole="QX-2", sample="3", top=_m(1.0)),
                  _grading(hole="QX-2", sample="5", top=_m(2.0)),
                  _grading(hole="QX-3", sample="2", top=_m(1.0)),
                  _grading(hole="QX-3", sample="4", top=_m(2.0))]
        joined, ties = _summary_matches(rows, sheets)
        assert joined == [[s] for s in sheets] and ties == []

    def test_a_numbering_running_on_through_the_holes_holds_in_each(self):
        # The sheets' numbers run on from hole to hole; the pair in QX-3
        # shows the table shares them, so QX-2's row printing another
        # number than the sheet at its depth is another specimen.
        rows = [_row(hole="QX-2", sample="8807", top=1.0),
                _row(hole="QX-3", sample="8819", top=2.0)]
        sheets = [_grading(hole="QX-2", sample="8814", top=_m(1.0)),
                  _grading(hole="QX-3", sample="8819", top=_m(2.0))]
        joined, ties = _summary_matches(rows, sheets)
        assert joined == [[], [sheets[1]]] and ties == []


class TestRunsUnderASpanningCell:

    @staticmethod
    def _linked_rows(tops_below, top_below=4.0):
        rows = [_row(top=1.0, other=[("Site ref.", "QX-2")]),
                _row(top=2.0), _row(top=3.0),
                _row(top=top_below, other=[("Site ref.", "QX-3")])]
        record = reconcile(_record(
            [_hole("QX-2"), _hole("QX-3", tops=tops_below)],
            _summary(*rows)))
        return record, [r.linked_investigation_id for r in _rows_of(record)]

    def test_rows_both_holes_have_samples_at_are_unplaced(self):
        # The cell may be printed on the middle row of the group it spans:
        # the blank rows may as well be QX-3's, whose log has samples there.
        record, got = self._linked_rows((1.0, 2.0, 3.0, 4.0))
        assert got == ["QX-2", "", "", "QX-3"]
        (note,) = _notes(record, "name their hole in another column")
        assert "2 row(s) printing it blank above a row of another hole " \
            "deeper down, whose log has a sample at their depths too, are " \
            "left unplaced" in note.detail

    def test_rows_the_hole_below_has_no_sample_at_keep_the_hole_above(self):
        _record_, got = self._linked_rows((5.0, 6.0), top_below=5.0)
        assert got == ["QX-2", "QX-2", "QX-2", "QX-3"]

    def test_rows_above_the_last_the_hole_below_lacks_keep_their_hole(self):
        # QX-3 has no sample at 2.0, so the group QX-3's cell spans starts
        # below it; the row at 3.0 may be either hole's.
        _record_, got = self._linked_rows((3.0, 4.0))
        assert got == ["QX-2", "QX-2", "", "QX-3"]

    def test_a_run_ending_where_the_depth_starts_again_keeps_its_hole(self):
        rows = [_row(top=1.0, other=[("Site ref.", "QX-2")]),
                _row(top=2.0), _row(top=3.0),
                _row(top=1.0, other=[("Site ref.", "QX-3")])]
        record = reconcile(_record([_hole("QX-2"), _hole("QX-3")],
                                   _summary(*rows)))
        assert [r.linked_investigation_id for r in _rows_of(record)] == \
            ["QX-2", "QX-2", "QX-2", "QX-3"]


def _reading(fields, wc):
    return LabSheetReading(
        depth_unit="m", pages_read=[40],
        tests=[ReadTest(kind="chemical", investigation_id="QX-2",
                        sample_id="S-3", depth_top=3.0, wc=wc,
                        pH=ReadReported(value=7.4),
                        fields=[ReadField(name=n, value=v)
                                for n, v in fields],
                        prov=ReadProv(page=40))])


class TestSolidsGateByArithmetic:
    """Percent solids S = 100 Ms / M; a water content is 100 Mw / M = 100
    - S on a wet basis, 100 Mw / Ms = 100 (100 - S) / S on a dry one. The
    two are one number only at S = 50 and S = 50 (sqrt 5 - 1) = 61.80."""

    @staticmethod
    def _built(fields, wc, **kw):
        builder = _Builder(_reading(fields, wc), [40], "R-made-up", [], **kw)
        (test,) = builder.build()
        return test, builder.unresolved

    def test_the_rule_is_the_two_definitions(self):
        assert SOLIDS_EQUAL_TO_WATER[0] == 50.0
        assert abs(SOLIDS_EQUAL_TO_WATER[1] - 61.803) < 0.001
        for k in range(100, 1000):
            s = k / 10
            alike = round(100 - s, 1) == s or \
                round(100 * (100 - s) / s, 1) == s
            assert solids_can_equal_water(f"{s:.1f}") == alike, s

    def test_the_printed_digits_set_the_tolerance(self):
        for printed in ("50", "61.8", "62", "61.80", "61,8 %"):
            assert solids_can_equal_water(printed), printed
        for printed in ("61.83", "51", "60", "76.9", "", "n/a"):
            assert not solids_can_equal_water(printed), printed

    def test_a_moisture_printing_another_number_keeps_the_solids_out(self):
        test, unresolved = self._built(
            [("% Solids", "76.9"), ("Moisture (%)", "23.1")], 76.9)
        assert test.result.wc is None
        assert any("refused rather than accepted" in u["why"]
                   for u in unresolved)

    def test_at_50_or_61_8_percent_solids_an_equal_number_is_kept(self):
        for printed in ("50.0", "61.8"):
            test, unresolved = self._built([("% Solids", printed)],
                                           float(printed))
            assert test.result.wc == float(printed) and not unresolved

    def test_a_neighbouring_sheet_s_moisture_keeps_nothing(self, tmp_path):
        # The window holds the chemistry sheet and an Atterberg sheet
        # printing a natural moisture content: the solids still are not
        # the chemistry specimen's water content.
        pymupdf = pytest.importorskip("pymupdf")
        documents = pytest.importorskip("planlens.document")
        from report_ingest.tests.fake_engine import FakeEngine
        pdf = pymupdf.open()
        for lines in (("ANALYTICAL RESULTS", "Sample QX-2 S-3", "pH 7.4",
                       "% Solids 76.9"),
                      ("ATTERBERG LIMITS",
                       "Natural moisture content (%) 23.1")):
            page = pdf.new_page(width=612, height=792)
            for i, text in enumerate(lines):
                page.insert_text((72, 72 + 18 * i), text, fontsize=11)
        path = tmp_path / "sheets.pdf"
        pdf.save(str(path))
        pdf.close()
        reading = _reading([("% Solids", "76.9")], 76.9)
        reading.pages_read = [0, 1]
        reading.tests[0].prov = ReadProv(page=0)
        doc = documents.open_document(str(path))
        try:
            result = read_lab_sheet(doc, [0, 1],
                                    FakeEngine([{"final": reading}]))
        finally:
            doc.close()
        (chem,) = [t for t in result.model_tests if t.kind == "chemical"]
        assert chem.result.wc is None
        assert all(getattr(t.result, "wc", None) != 76.9
                   for t in result.tests)

    def test_under_flag_the_number_is_doubtful_and_kept(self):
        test, unresolved = self._built([("% Solids", "52.37")], 52.37,
                                       checks="flag")
        assert test.result.wc == 52.37
        (entry,) = [u for u in unresolved if "dry solids" in u["why"]]
        assert "kept at confidence 0.5 for review" in entry["why"]
        assert "no real sheet prints this" not in entry["why"]
        assert test.prov[0].confidence == 0.5

    def test_a_water_analysis_s_solids_are_no_share_of_dry_solids(self):
        for label in ("Total dissolved solids (%)", "Suspended solids (%)",
                      "Volatile solids (%)", "Total suspended solids, %"):
            assert not names_solids(label), label
            assert _field_of(label) != "as_printed", label


class TestTheChecklistReadsDepthsAsTheMatchDoes:

    def test_a_unitless_depth_is_read_in_the_record_s_unit(self):
        sheet = _grading(top=_bare(2.0))
        assert CL._printed(sheet, "m")[1] == 2.0
        assert CL._printed(sheet)[1] is None


class TestAKindTheRecordHoldsNoSheetOf:

    @staticmethod
    def _check(*lab, bound=()):
        record = _record([_hole("QX-2")], *lab)
        record.bound_documents = list(bound)
        rows = {r["id"]: r for r in
                CL.run_checklist(reconcile(record))["code_checks"]}
        return rows["completeness.summary_and_sheets"]

    @staticmethod
    def _lab():
        rows = [_row(hole="QX-2", top=t, ll=40.0, sand_percent=41.0)
                for t in (1.0, 2.0)]
        return [_grading(top=_m(t)) for t in (1.0, 2.0)] + [_summary(*rows)]

    def test_limits_with_no_atterberg_sheet_in_the_record_are_a_note(self):
        sh = self._check(*self._lab())
        assert sh["status"] == "pass", sh
        assert "the record holds no atterberg sheet; 2 summary row(s) " \
            "tabulate it (not counted as missing: there is no sheet to be " \
            "missing)" in sh["detail"]

    def test_a_report_bound_in_it_and_read_on_its_own_makes_it_unsure(self):
        bound = [BoundReport(bound_id="bound1", report_id="SYN-b1",
                             pages="45-58", read=True,
                             counts={"lab_tests": 3}),
                 BoundReport(bound_id="bound2", report_id="SYN-b2",
                             pages="59-70", read=False)]
        sh = self._check(*self._lab(), bound=bound)
        assert sh["status"] == "unsure", sh
        assert "a report bound in this one, outside this record, may hold " \
            "it" in sh["detail"]
        assert "bound1 (SYN-b1, read on its own, 3 lab test(s))" in \
            sh["detail"]
        assert "bound2 (SYN-b2, not read)" in sh["detail"]

    def test_a_bound_report_holding_no_laboratory_test_changes_nothing(self):
        bound = [BoundReport(bound_id="bound1", report_id="SYN-b1",
                             pages="45-58", read=True,
                             counts={"lab_tests": 0})]
        sh = self._check(*self._lab(), bound=bound)
        assert sh["status"] == "pass", sh
        assert "bound1" not in sh["detail"]

    def test_limits_with_an_atterberg_sheet_for_another_row_are_a_gap(self):
        limits = LabTest(kind="atterberg", investigation_id="QX-2",
                         depth_top=_m(1.0), pages=[35], source_report="SYN",
                         result=AtterbergResult(ll=40.0, pl=20.0, pi=20.0))
        sh = self._check(*self._lab(), limits)
        assert sh["status"] == "fail", sh
        assert "1 with no atterberg sheet" in sh["detail"]
        assert "the record holds no atterberg sheet" not in sh["detail"]
