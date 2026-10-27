"""Two holes under one printed name (BACKLOG RI-135).

A report prints a second series of logs whose hole names reuse the first
series' ("QX-1" twice), so the record holds two holes under one name. They
are kept apart by what their pages print, each with a qualifier beside the
printed name; a QA entry names them; the laboratory link refuses to guess
between them; the counts take them as two holes; and the DIGGS file gives
each its own feature under the printed name.

One hole whose sheets the fold did not join (a later sheet a page further
on, a sounding carried on to the next page) is NOT two holes: its logs are
counted as one hole, said in one QA entry, and a sheet naming it goes to
the log holding its sample, else the first, as before.

Every hole name, firm, date, station, title, page and value here is made up.
"""

from types import SimpleNamespace

import pytest

from report_ingest.diggs_writer import (
    NS, QUALIFIER_PARAMETER, diggs_roundtrip_gate, diggs_schema_gate,
    write_diggs,
)
from report_ingest.model import (
    BoundReport, DCPData, DCPPoint, DrillingDetails, Investigation, LabTest,
    Layer, MoistureDensityResult, Quantity, ReportRecord, Sample, SummaryRow,
    SummaryTableResult, hole_label, qa_rank,
)
from report_ingest.narrative_reader import explorations_found
from report_ingest.reconciler import _may_join, reconcile
from report_ingest.shared_names import (
    PagePrint, _date_key, hole_identity, hole_parts, name_groups, pages_span,
    tell_apart,
)


def _m(value):
    return Quantity(value=value, unit="m")


def _hole(name="QX-1", pages=(10,), elevation=None, depth=8.0, **over):
    data = dict(
        investigation_id=name, pages=list(pages), depth_unit="m",
        total_depth=_m(depth) if depth is not None else None,
        elevation=_m(elevation) if elevation is not None else None,
        layers=[Layer(top=_m(0.0), bottom=_m(depth or 5.0),
                      description="brown silty clay")],
        samples=[Sample(sample_id="S-1", top=_m(1.0), bottom=_m(1.5)),
                 Sample(sample_id="S-2", top=_m(3.0), bottom=_m(3.5))])
    data.update(over)
    return Investigation(**data)


def _sheet(name="QX-1", page=40, depth=1.0, **over):
    data = dict(kind="moisture_content", investigation_id=name,
                depth_top=_m(depth), pages=[page],
                result=MoistureDensityResult(wc=17.5))
    data.update(over)
    return LabTest(**data)


def _outline(*tabs):
    """A planlens-like outline: each tab ``(page, words)``."""
    return SimpleNamespace(
        dividers=[SimpleNamespace(page=p, text=t) for p, t in tabs],
        appendices=[])


class _Doc:
    """A planlens-like document: ``{page: [line text, ...]}``, top down."""

    def __init__(self, pages):
        self.pages = pages

    def page(self, index):
        lines = [SimpleNamespace(text=text, bbox=(10.0, 600.0 + 12 * n,
                                                  300.0, 610.0 + 12 * n))
                 for n, text in enumerate(self.pages.get(index, ()))]
        return SimpleNamespace(lines=lines)


def _twins_record(**first_over):
    """Two made-up holes printed under "QX-1": the first series' at page
    10, the second's at page 20, with different ground elevations; both
    logged down from the ground, so their depths overlap."""
    first = dict(elevation=52.0, depth=8.0)
    first.update(first_over)
    return ReportRecord(investigations=[
        _hole("QX-1", [10], **first),
        _hole("QX-1", [20], elevation=43.0, depth=15.0)])


def _stationed():
    """The same two holes, each printing its station and no elevation."""
    return ReportRecord(investigations=[
        _hole("QX-1", [10], station="10+00"),
        _hole("QX-1", [20], station="55+00", depth=15.0)])


def _entries(record, where):
    return [e for e in record.qa if e.where == where]


# ---------------------------------------------------------------------------
# kept apart, each qualified, and said
# ---------------------------------------------------------------------------

class TestKeptApart:
    def test_two_holes_of_one_name_get_a_qualifier_each(self):
        record = reconcile(_twins_record())
        first, second = record.investigations
        assert first.investigation_id == second.investigation_id == "QX-1"
        assert first.qualifier == "ground elevation 52 m"
        assert second.qualifier == "ground elevation 43 m"

    def test_one_qa_entry_names_them_open_and_says_what_tells_them_apart(self):
        record = reconcile(_twins_record())
        (entry,) = _entries(record, "investigations.QX-1")
        assert entry.kind == "note" and entry.rank == "open"
        assert "two holes are printed under the name 'QX-1'" in entry.detail
        assert "told apart by their ground elevation" in entry.detail
        # The total depth tells nothing: a cut-off log prints a shallower one.
        assert "total depth" not in entry.detail
        assert "fold did not join" not in entry.detail
        assert entry.values == ["QX-1 (ground elevation 52 m): page 10",
                                "QX-1 (ground elevation 43 m): page 20"]
        assert entry.pages == [10, 20]
        # The rank follows the words, so a record ranked on loading agrees.
        assert qa_rank(entry.kind, entry.detail) == "open"

    def test_a_different_total_depth_alone_does_not_tell_them_apart(self):
        record = _twins_record(elevation=None, depth=8.0)
        record.investigations[1].elevation = None
        tell_apart(record)
        assert [i.qualifier for i in record.investigations] == [
            "page 10", "page 20"]

    def test_a_hole_whose_name_is_its_own_keeps_no_qualifier(self):
        record = _twins_record()
        record.investigations.append(_hole("QX-2", [30], elevation=50.0))
        reconcile(record)
        assert record.investigations[2].qualifier == ""
        assert "qualifier" not in record.investigations[2].model_dump(
            mode="json")
        assert record.investigations[0].model_dump(
            mode="json")["qualifier"] == "ground elevation 52 m"

    def test_the_qualifier_survives_the_json_round_trip(self):
        record = reconcile(_twins_record())
        again = ReportRecord.model_validate(record.model_dump(mode="json"))
        assert [i.qualifier for i in again.investigations] == [
            "ground elevation 52 m", "ground elevation 43 m"]

    def test_by_report_section_first_from_the_outline(self):
        record = _twins_record()
        _twins, qa = tell_apart(record, outline=_outline(
            (8, "Appendix B Logs of the borings"),
            (18, "Appendix D Earlier investigation")))
        assert [i.qualifier for i in record.investigations] == [
            "under 'Appendix B Logs of the borings'",
            "under 'Appendix D Earlier investigation'"]
        assert "told apart by their report section" in qa[0].detail

    def test_one_section_for_both_falls_to_the_title_the_logs_print(self):
        record = _twins_record(elevation=None)
        record.investigations[1].elevation = None
        doc = _Doc({10: ["Alder Engineering", "LOG OF BORING QX-1"],
                    20: ["STRATUM CHART, TEST HOLE QX-1", "Depth 1.0 m"]})
        tell_apart(record, doc=doc,
                   outline=_outline((8, "Appendix B Logs of the borings")))
        assert [i.qualifier for i in record.investigations] == [
            "titled 'LOG OF BORING QX-1'",
            "titled 'STRATUM CHART, TEST HOLE QX-1'"]

    def test_one_title_for_both_does_not_tell_them_apart(self):
        record = _twins_record(elevation=None, depth=15.0)
        record.investigations[1].elevation = None
        doc = _Doc({10: ["LOG OF BORING QX-1"], 20: ["LOG OF BORING QX-1"]})
        _twins, qa = tell_apart(record, doc=doc)
        assert [i.qualifier for i in record.investigations] == [
            "page 10", "page 20"]
        assert "nothing their pages print tells them apart" in qa[0].detail
        assert "one may be the other printed again" in qa[0].detail
        assert qa[0].rank == "open"

    def test_a_title_that_only_adds_continued_is_the_same_title(self):
        record = _twins_record(elevation=None, depth=15.0)
        record.investigations[1].elevation = None
        doc = _Doc({10: ["LOG OF BORING QX-1"],
                    20: ["LOG OF BORING QX-1 (continued)"]})
        tell_apart(record, doc=doc)
        # Their depths overlap, so two holes; the title tells them nothing.
        assert [i.qualifier for i in record.investigations] == [
            "page 10", "page 20"]

    def test_a_fact_only_one_log_prints_tells_nothing(self):
        record = _twins_record(depth=15.0)
        record.investigations[1].elevation = None
        tell_apart(record)
        assert [i.qualifier for i in record.investigations] == [
            "page 10", "page 20"]

    def test_by_firm_project_and_date(self):
        record = ReportRecord(investigations=[
            _hole("QX-1", [10], drilling=DrillingDetails(
                contractor="Alder Drilling Co."),
                date_started="05/19/2044"),
            _hole("QX-1", [20], drilling=DrillingDetails(
                contractor="Birch Boring Ltd."),
                date_started="09/02/2051")])
        tell_apart(record)
        assert [i.qualifier for i in record.investigations] == [
            "drilled by Alder Drilling Co.", "drilled by Birch Boring Ltd."]
        record.investigations[1].drilling.contractor = "Alder Drilling Co."
        record.investigations[0].fields = {"Project No.": "QX-44-01"}
        record.investigations[1].fields = {"Project No.": "OLD-77-B"}
        tell_apart(record)
        assert record.investigations[0].qualifier == \
            "Project No.: QX-44-01"
        record.investigations[1].fields = {"Project No.": "QX-44-01"}
        tell_apart(record)
        assert record.investigations[0].qualifier == "started 05/19/2044"

    def test_two_dates_one_reading_makes_one_day_do_not_tell_them_apart(self):
        record = ReportRecord(investigations=[
            _hole("QX-1", [10], date_started="3/8/2044"),
            _hole("QX-1", [20], date_started="8/3/2044")])
        tell_apart(record)
        assert [i.qualifier for i in record.investigations] == [
            "page 10", "page 20"]

    def test_a_contractor_field_is_a_firm_not_a_project(self):
        record = ReportRecord(investigations=[
            _hole("QX-1", [10], fields={"Drilling Contractor": "Alder Co."}),
            _hole("QX-1", [20], fields={"Drilling Contractor": "Birch Ltd."})])
        _twins, qa = tell_apart(record)
        assert record.investigations[0].qualifier == \
            "Drilling Contractor: Alder Co."
        assert "told apart by their firm" in qa[0].detail

    def test_by_bound_report(self):
        record = _twins_record(elevation=None)
        record.investigations[1].elevation = None
        record.bound_documents = [BoundReport(
            bound_id="b1", title="Earlier QX-series report", first_page=18,
            last_page=25)]
        tell_apart(record)
        assert record.investigations[0].qualifier == \
            "in the report itself, not a bound one"
        assert record.investigations[1].qualifier == \
            "in bound report b1 (Earlier QX-series report)"

    def test_three_holes_of_one_name(self):
        record = _twins_record()
        record.investigations.append(_hole("QX-1", [30], elevation=61.0))
        _twins, qa = tell_apart(record)
        assert len({i.qualifier for i in record.investigations}) == 3
        assert "three holes are printed under the name" in qa[0].detail

    def test_names_written_differently_are_not_one_name_here(self):
        holes = [_hole("QX-01", [10]), _hole("QX-1", [20])]
        assert name_groups(holes) == []

    def test_continuation_sheets_still_fold_into_one_hole(self):
        record = ReportRecord(investigations=[
            _hole("QX-1", [10], depth=None, sheet="Sheet 1 of 2",
                  layers=[Layer(top=_m(0.0), bottom=_m(6.0),
                                description="clay")]),
            _hole("QX-1", [11], depth=12.0, sheet="Sheet 2 of 2",
                  layers=[Layer(top=_m(6.0), bottom=_m(12.0),
                                description="sand")],
                  samples=[Sample(sample_id="S-4", top=_m(7.0))])])
        reconcile(record)
        assert len(record.investigations) == 1
        assert record.investigations[0].qualifier == ""
        assert not _entries(record, "investigations.QX-1")

    def test_told_apart_twice_gives_the_same_qualifiers(self):
        record = _twins_record()
        tell_apart(record)
        once = [i.qualifier for i in record.investigations]
        tell_apart(record)
        assert [i.qualifier for i in record.investigations] == once

    def test_the_section_before_the_first_tab(self):
        page_print = PagePrint(outline=_outline((15, "Appendix A")))
        assert page_print.section(3)[0] == -1
        assert page_print.section(15) == (15, "Appendix A")
        assert PagePrint().section(3) is None

    def test_pages_span(self):
        assert pages_span([3, 4, 5, 9]) == "pages 3-5, 9"
        assert pages_span([7]) == "page 7"


class TestDates:
    def test_a_date_read_one_way_is_one_day(self):
        assert _date_key("10/29/47") == _date_key("10/29/2047")
        assert len(_date_key("10/29/47")) == 1
        assert _date_key("29 Oct 2047") == _date_key("10/29/2047")
        assert _date_key("2047-10-29") == _date_key("Oct. 29, 2047")

    def test_the_day_and_the_month_are_not_guessed(self):
        assert len(_date_key("3/8/47")) == 2
        assert _date_key("6/6/47") == frozenset({(47, 6, 6)})

    def test_no_number_no_date(self):
        assert _date_key("n/a") is None
        assert _date_key("") is None


# ---------------------------------------------------------------------------
# one hole read as several logs is not two holes
# ---------------------------------------------------------------------------

def _cut_off():
    """Made-up boring ZK-7: sheets 1 and 2 read as one log (pages 30-31);
    sheet 3's page not read as a log; sheet 4 of 4 on page 33, a log of its
    own."""
    first = _hole("ZK-7", [30, 31], sheet="Sheet 1 of 4", depth=None,
                  layers=[Layer(top=_m(0.0), bottom=_m(7.0),
                                description="grey clay")],
                  samples=[Sample(sample_id="S-1", top=_m(1.0),
                                  bottom=_m(1.5)),
                           Sample(sample_id="S-2", top=_m(3.0),
                                  bottom=_m(3.5)),
                           Sample(sample_id="S-4", top=_m(6.0),
                                  bottom=_m(6.5))])
    last = _hole("ZK-7", [33], sheet="Sheet 4 of 4", depth=12.0,
                 layers=[Layer(top=_m(9.0), bottom=_m(12.0),
                               description="dense sand")],
                 samples=[Sample(sample_id="S-7", top=_m(10.0),
                                 bottom=_m(10.5))])
    return ReportRecord(investigations=[first, last])


_CUT_OFF_DOC = _Doc({30: ["LOG OF BORING ZK-7"],
                     33: ["LOG OF BORING ZK-7 (continued)"]})


class TestOneHoleReadAsSeveralLogs:
    def test_a_later_sheet_a_page_further_on_is_not_a_second_hole(self):
        record = _cut_off()
        record.lab_tests = [_sheet("ZK-7", page=40, depth=1.0,
                                   sample_id="S-1"),
                            _sheet("ZK-7", page=41, depth=10.0,
                                   sample_id="S-7")]
        reconcile(record, doc=_CUT_OFF_DOC)
        assert len(record.investigations) == 2
        assert [i.qualifier for i in record.investigations] == ["", ""]
        first, deep = record.lab_tests
        # Linked as before, and each to the log holding its sample.
        assert first.linked_investigation_id == "ZK-7"
        assert first.linked_sample_id == "S-1"
        assert deep.linked_investigation_id == "ZK-7"
        assert deep.linked_sample_id == "S-7"
        assert first.linked_qualifier == deep.linked_qualifier == ""
        # Which of the hole's logs, by its pages.
        assert first.linked_log_pages == "pages 30-31"
        assert deep.linked_log_pages == "page 33"
        assert not [e for e in record.qa if e.kind == "partial"
                    and e.where == "lab_tests.ZK-7"]

    def test_a_sample_name_on_two_of_its_logs_goes_where_the_depth_agrees(
            self):
        # S-4 is printed on sheet 1 at 6.0 m and again on sheet 4 at 9.0 m.
        record = _cut_off()
        record.investigations[1].samples.insert(
            0, Sample(sample_id="S-4", top=_m(9.0), bottom=_m(9.5)))
        record.lab_tests = [_sheet("ZK-7", page=40, depth=9.0,
                                   sample_id="S-4")]
        reconcile(record, doc=_CUT_OFF_DOC)
        test = record.lab_tests[0]
        assert test.linked_sample_id == "S-4"
        assert not [e for e in record.qa if e.kind == "conflict"]
        assert test.linked_log_pages == "page 33"

    def test_it_is_said_once_open_with_what_says_so(self):
        record = reconcile(_cut_off(), doc=_CUT_OFF_DOC)
        (entry,) = _entries(record, "investigations.ZK-7")
        assert entry.kind == "note" and entry.rank == "open"
        assert qa_rank(entry.kind, entry.detail) == "open"
        assert "sheets the fold did not join" in entry.detail
        assert "it prints sheet 4 of 4" in entry.detail
        assert "carry on from where the earlier log's" in entry.detail
        assert "only '(continued)' added" in entry.detail
        assert "holes under one printed name" not in entry.detail
        assert entry.values[0] == "ZK-7: pages 30-31"
        assert entry.values[1].startswith("ZK-7: page 33 (")
        assert entry.pages == [30, 31, 33]

    def test_counted_as_one_hole(self):
        record = reconcile(_cut_off(), doc=_CUT_OFF_DOC)
        assert explorations_found(record.investigations)["boringCount"] == 1
        first, last = record.investigations
        assert hole_identity(first) == hole_identity(last)

    def test_the_printed_sheet_number_alone_says_so(self):
        record = ReportRecord(investigations=[
            _hole("QX-3", [12], sheet="Sheet 1 of 3"),
            _hole("QX-3", [14], sheet="Sheet 3 of 3", samples=[], layers=[],
                  depth=None)])
        holes, why, unsure = hole_parts(record.investigations)
        assert len(holes) == 1 and not unsure
        assert why[id(record.investigations[1])] == "it prints sheet 3 of 3"

    @pytest.mark.parametrize("sheet", ["Sheet 2 of 5", "Sheet 1 of 1"])
    def test_a_first_sheet_or_one_of_another_set_is_another_hole(self, sheet):
        record = ReportRecord(investigations=[
            _hole("QX-3", [12], sheet="Sheet 1 of 2"),
            _hole("QX-3", [14], sheet=sheet, samples=[], layers=[],
                  depth=None)])
        holes, _why, _unsure = hole_parts(record.investigations)
        assert len(holes) == 2

    def test_a_log_nothing_could_be_read_off_is_taken_as_the_same_hole(self):
        # The second log's page gave the readers nothing: no depth, no sheet
        # number, no title, no header field. Nothing says it is another hole.
        record = ReportRecord(investigations=[
            _hole("QX-3", [12], station="20+00"),
            _hole("QX-3", [14], samples=[], layers=[], depth=None)])
        record.lab_tests = [_sheet("QX-3", page=40, depth=3.0)]
        reconcile(record)
        assert [i.qualifier for i in record.investigations] == ["", ""]
        (entry,) = _entries(record, "investigations.QX-3")
        assert entry.rank == "open"
        assert "may be one hole's sheets the fold did not join, or two " \
            "holes" in entry.detail
        assert "counted as one hole, as before" in entry.detail
        test = record.lab_tests[0]
        assert test.linked_investigation_id == "QX-3"
        assert test.linked_sample_id == "S-2"
        assert explorations_found(record.investigations)["boringCount"] == 1

    def test_but_a_different_series_with_no_sign_is_another_hole(self):
        record = ReportRecord(investigations=[
            _hole("QX-3", [12], station="20+00"),
            _hole("QX-3", [14], station="64+00", samples=[], layers=[],
                  depth=None)])
        reconcile(record)
        assert [i.qualifier for i in record.investigations] == [
            "at station 20+00", "at station 64+00"]

    def test_a_soundings_readings_carried_on_to_the_next_page(self):
        def probe(pages, depths):
            return Investigation(
                investigation_id="DP-2", kind="dcp", pages=pages,
                depth_unit="m", dcp=DCPData(points=[
                    DCPPoint(depth=_m(d), blows=4.0) for d in depths]))
        record = ReportRecord(investigations=[
            probe([50], [0.3, 0.6, 0.9, 1.2, 1.5]),
            probe([51], [1.8, 2.1, 2.4])])
        reconcile(record)
        assert [i.qualifier for i in record.investigations] == ["", ""]
        (entry,) = _entries(record, "investigations.DP-2")
        assert "its readings (1.80-2.40 m) carry on" in entry.detail
        first, later = record.investigations
        assert hole_identity(first) == hole_identity(later)

    def test_two_soundings_both_from_the_ground_are_two(self):
        def probe(pages, depths):
            return Investigation(
                investigation_id="DP-2", kind="dcp", pages=pages,
                depth_unit="m", dcp=DCPData(points=[
                    DCPPoint(depth=_m(d), blows=4.0) for d in depths]))
        record = ReportRecord(investigations=[
            probe([50], [0.3, 0.6, 0.9]), probe([51], [0.3, 0.6])])
        reconcile(record)
        assert [i.qualifier for i in record.investigations] == [
            "page 50", "page 51"]

    def test_two_series_with_overlapping_depths_are_still_two_holes(self):
        # The true twins' shape: two series, each logged from the ground,
        # different strata; the second even prints "sheet 2 of 2".
        record = ReportRecord(investigations=[
            _hole("QX-1", [10], layers=[Layer(
                top=_m(0.0), bottom=_m(8.0), description="brown clay")]),
            _hole("QX-1", [20], sheet="Sheet 2 of 2", layers=[Layer(
                top=_m(0.0), bottom=_m(15.0), description="grey sand")])])
        record.lab_tests = [_sheet(page=40)]
        reconcile(record, doc=_Doc({10: ["LOG OF BORING QX-1"],
                                    20: ["STRATUM CHART, TEST HOLE QX-1"]}))
        assert [i.qualifier for i in record.investigations] == [
            "titled 'LOG OF BORING QX-1'",
            "titled 'STRATUM CHART, TEST HOLE QX-1'"]
        (entry,) = _entries(record, "investigations.QX-1")
        assert "holes under one printed name" in entry.detail
        assert "fold did not join" not in entry.detail
        assert record.lab_tests[0].linked_investigation_id == ""

    def test_one_twin_read_as_two_logs(self):
        record = _stationed()
        record.investigations[1].sheet = "Sheet 1 of 2"
        record.investigations.append(_hole(
            "QX-1", [22], sheet="Sheet 2 of 2", depth=15.0,
            layers=[Layer(top=_m(8.0), bottom=_m(15.0),
                          description="sand")],
            samples=[Sample(sample_id="S-5", top=_m(9.0), bottom=_m(9.5))]))
        record.lab_tests = [_sheet(page=40, depth=9.0,
                                   fields={"Station": "55+00"})]
        reconcile(record)
        a, b, b_more = record.investigations
        assert a.qualifier == "at station 10+00"
        assert b.qualifier == b_more.qualifier == "at station 55+00"
        (entry,) = _entries(record, "investigations.QX-1")
        assert "two holes are printed under the name 'QX-1'" in entry.detail
        assert "QX-1 (at station 55+00) is printed on two logs taken as " \
            "one hole's sheets the fold did not join" in entry.detail
        assert entry.values == ["QX-1 (at station 10+00): page 10",
                                "QX-1 (at station 55+00): pages 20, 22"]
        test = record.lab_tests[0]
        assert test.linked_qualifier == "at station 55+00"
        assert test.linked_sample_id == "S-5"
        assert test.linked_log_pages == "page 22"
        assert explorations_found(record.investigations)["boringCount"] == 2


def _two_series():
    """Made-up: two holes printed as QV-2 in one appendix, each its own
    "Sheet 1 of 1", firm and date; the second's first sample was taken
    below the first's last, so its depths seem to carry on."""
    return ReportRecord(investigations=[
        _hole("QV-2", [60], sheet="Sheet 1 of 1", depth=4.0,
              date_started="10/29/2047",
              drilling=DrillingDetails(contractor="Larch Drilling"),
              samples=[Sample(sample_id="S-1", top=_m(1.0), bottom=_m(1.5)),
                       Sample(sample_id="S-2", top=_m(3.0),
                              bottom=_m(3.5))]),
        _hole("QV-2", [62], sheet="Sheet 1 of 1", depth=9.0,
              date_started="05/19/2044",
              drilling=DrillingDetails(contractor="Hornbeam Drilling"),
              samples=[Sample(sample_id="S-1", top=_m(6.0), bottom=_m(6.5)),
                       Sample(sample_id="S-2", top=_m(8.0),
                              bottom=_m(8.5))])])


_APPENDIX = _outline((55, "Appendix B Logs of the borings"))


class TestWhatIsPrintedOutweighsDepthsCarryingOn:
    def test_a_second_series_deeper_down_is_another_hole(self):
        record = _two_series()
        holes, _why, _unsure = hole_parts(record.investigations)
        assert len(holes) == 2
        reconcile(record, outline=_APPENDIX)
        assert [i.qualifier for i in record.investigations] == [
            "drilled by Larch Drilling", "drilled by Hornbeam Drilling"]
        (entry,) = _entries(record, "investigations.QV-2")
        assert "two holes are printed under the name 'QV-2'" in entry.detail
        assert "fold did not join" not in entry.detail
        assert explorations_found(record.investigations)["boringCount"] == 2

    def test_its_sheets_are_not_linked_to_the_first_holes_samples(self):
        record = _two_series()
        record.lab_tests = [
            _sheet("QV-2", page=70, depth=6.0, sample_id="S-1"),
            _sheet("QV-2", page=71, depth=8.0, sample_id="S-2"),
            _sheet("QV-2", page=72, depth=8.0, sample_id="S-2",
                   lab="Hornbeam Drilling soils laboratory")]
        reconcile(record, outline=_APPENDIX)
        first, second, by_firm = record.lab_tests
        for refused in (first, second):
            assert refused.linked_investigation_id == ""
            assert refused.linked_sample_id == ""
        assert by_firm.linked_qualifier == "drilled by Hornbeam Drilling"
        assert by_firm.linked_sample_id == "S-2"
        assert by_firm.linked_log_pages == ""
        assert not [e for e in record.qa if e.kind == "conflict"]

    @pytest.mark.parametrize("firm, holes", [("Hornbeam Drilling", 2),
                                             ("Larch Drilling", 1)])
    def test_a_sparse_first_log(self, firm, holes):
        # One sample at 1.5 m, then a log whose first sample is at 3 m: the
        # depths alone would carry on; a different firm says another hole.
        record = ReportRecord(investigations=[
            _hole("QV-2", [60],
                  drilling=DrillingDetails(contractor="Larch Drilling"),
                  samples=[Sample(sample_id="S-1", top=_m(1.5))]),
            _hole("QV-2", [61], drilling=DrillingDetails(contractor=firm),
                  samples=[Sample(sample_id="S-1", top=_m(3.0)),
                           Sample(sample_id="S-2", top=_m(5.0))])])
        found, why, _unsure = hole_parts(record.investigations)
        assert len(found) == holes
        if holes == 1:
            assert why[id(record.investigations[1])].startswith(
                "its depths (3.00-5.00 m) carry on")

    def test_a_printed_sheet_number_of_the_set_still_says_a_later_sheet(self):
        # A sheet number of the earlier log's set is printed: it is that
        # hole's later sheet, though its header was read with another date.
        record = ReportRecord(investigations=[
            _hole("QV-2", [60], sheet="Sheet 1 of 2",
                  date_started="10/29/2047"),
            _hole("QV-2", [61], sheet="Sheet 2 of 2",
                  date_started="05/19/2044",
                  samples=[Sample(sample_id="S-3", top=_m(6.0))])])
        found, why, _unsure = hole_parts(record.investigations)
        assert len(found) == 1
        assert "it prints sheet 2 of 2" in why[id(record.investigations[1])]


# ---------------------------------------------------------------------------
# the laboratory link does not guess
# ---------------------------------------------------------------------------

class TestLabLink:
    def test_a_sheet_that_prints_nothing_telling_is_linked_to_neither(self):
        record = _twins_record()
        record.lab_tests = [_sheet(page=40)]
        reconcile(record)
        test = record.lab_tests[0]
        assert test.linked_investigation_id == ""
        assert test.linked_sample_id == "" and test.linked_qualifier == ""
        (entry,) = [e for e in _entries(record, "lab_tests.QX-1")
                    if e.kind == "partial"]
        assert entry.rank == "open"
        assert "print nothing that tells which" in entry.detail
        assert "linked to neither" in entry.detail
        assert entry.values == ["moisture_content: page 40"]
        # Not said as a hole the logs lack: the logs hold two.
        assert not any("not among the explorations" in e.detail
                       for e in record.qa)

    def test_many_refused_are_listed_with_how_many_more(self):
        record = _twins_record()
        record.lab_tests = [_sheet(page=40 + n) for n in range(14)]
        reconcile(record)
        (entry,) = [e for e in _entries(record, "lab_tests.QX-1")
                    if e.kind == "partial"]
        assert len(entry.values) == 13
        assert entry.values[-1] == "... 2 more"

    def test_an_elevation_the_sheet_prints_chooses_no_hole(self):
        record = _twins_record()
        record.lab_tests = [_sheet(page=40, depth=3.0, elevation=_m(40.0))]
        reconcile(record)
        assert record.lab_tests[0].linked_investigation_id == ""

    def test_the_station_it_prints_puts_it_at_one_hole(self):
        record = _stationed()
        record.lab_tests = [_sheet(page=40, depth=3.0,
                                   fields={"Station": "55+00"})]
        reconcile(record)
        test = record.lab_tests[0]
        assert test.linked_investigation_id == "QX-1"
        assert test.linked_qualifier == "at station 55+00"
        assert test.linked_sample_id == "S-2"
        (entry,) = _entries(record, "lab_tests.QX-1")
        assert entry.kind == "note" and entry.rank == "note"
        assert "linked to one of them by what they print" in entry.detail
        assert "at station 55+00" in entry.values[0]

    def test_its_pages_under_one_holes_report_section(self):
        record = _twins_record()
        record.lab_tests = [_sheet(page=12), _sheet(page=40)]
        reconcile(record, outline=_outline(
            (8, "Appendix B Logs and lab"), (18, "Appendix D Earlier work"),
            (35, "Appendix E Site photographs")))
        first, last = record.lab_tests
        assert first.linked_qualifier == "under 'Appendix B Logs and lab'"
        assert last.linked_investigation_id == ""

    def test_its_sample_date_is_one_holes_drilling_date(self):
        record = ReportRecord(investigations=[
            _hole("QX-1", [10], date_started="05/19/2044"),
            _hole("QX-1", [20], date_started="09/23/2051")])
        record.lab_tests = [_sheet(fields={"Date Sampled": "9/23/51"})]
        reconcile(record)
        assert record.lab_tests[0].linked_qualifier == "started 09/23/2051"

    def test_a_sample_date_read_two_ways_chooses_no_hole(self):
        record = ReportRecord(investigations=[
            _hole("QX-1", [10], date_started="05/19/2044"),
            _hole("QX-1", [20], date_started="04/06/2051")])
        record.lab_tests = [_sheet(fields={"Date Sampled": "4/6/51"})]
        reconcile(record)
        assert record.investigations[1].qualifier == "started 04/06/2051"
        assert record.lab_tests[0].linked_investigation_id == ""

    def test_the_firm_it_prints(self):
        record = ReportRecord(investigations=[
            _hole("QX-1", [10], drilling=DrillingDetails(
                contractor="Alder Drilling Co.")),
            _hole("QX-1", [20], drilling=DrillingDetails(
                contractor="Birch Boring Ltd."))])
        record.lab_tests = [_sheet(lab="Birch Boring Ltd. soils laboratory")]
        reconcile(record)
        assert record.lab_tests[0].linked_qualifier == \
            "drilled by Birch Boring Ltd."

    def test_a_sheet_pointing_at_both_is_linked_to_neither(self):
        record = ReportRecord(investigations=[
            _hole("QX-1", [10], drilling=DrillingDetails(
                contractor="Alder Drilling Co.")),
            _hole("QX-1", [20], drilling=DrillingDetails(
                contractor="Birch Boring Ltd."))])
        record.lab_tests = [_sheet(page=12,
                                   lab="Birch Boring Ltd. soils laboratory")]
        reconcile(record, outline=_outline((8, "Appendix B Logs"),
                                           (18, "Appendix D Earlier work")))
        assert record.lab_tests[0].linked_investigation_id == ""

    def test_a_name_of_one_hole_is_linked_as_before(self):
        record = ReportRecord(investigations=[_hole("QX-2", [10])])
        record.lab_tests = [_sheet("QX-2")]
        reconcile(record)
        test = record.lab_tests[0]
        assert test.linked_investigation_id == "QX-2"
        assert test.linked_sample_id == "S-1"
        assert "linked_qualifier" not in test.model_dump(mode="json")
        assert not _entries(record, "lab_tests.QX-2")

    def test_summary_rows_each_by_what_they_print(self):
        record = _stationed()
        record.lab_tests = [LabTest(
            kind="summary_table", pages=[45],
            result=SummaryTableResult(rows=[
                SummaryRow(investigation_id="QX-1", depth_top=_m(1.0),
                           wc=14.0),
                SummaryRow(investigation_id="QX-1", depth_top=_m(3.0),
                           other=[("Station", "10+00")], wc=16.0),
                SummaryRow(investigation_id="QX-2", depth_top=_m(1.0),
                           wc=18.0)]))]
        reconcile(record)
        refused, kept, other = record.lab_tests[0].result.rows
        assert refused.linked_investigation_id == ""
        assert kept.linked_investigation_id == "QX-1"
        assert kept.linked_qualifier == "at station 10+00"
        assert kept.linked_sample_id == "S-2"
        partial = [e for e in _entries(record, "lab_tests.QX-1")
                   if e.kind == "partial"]
        assert partial and "1 summary row(s) name 'QX-1'" in \
            partial[0].detail
        # QX-2 is a hole no log holds, said as before; QX-1 is not.
        (missing,) = [e for e in record.qa
                      if e.where == "lab_tests.summary_table"
                      and "not among those read" in e.detail]
        assert missing.values == ["QX-2"]

    def test_a_row_below_printing_its_hole_blank_takes_the_row_aboves(self):
        record = _stationed()
        record.lab_tests = [LabTest(
            kind="summary_table", pages=[45],
            result=SummaryTableResult(rows=[
                SummaryRow(depth_top=_m(1.0),
                           other=[("Location", "QX-1"),
                                  ("Station", "55+00")], wc=14.0),
                SummaryRow(depth_top=_m(3.0), wc=16.0)]))]
        reconcile(record)
        first, below = record.lab_tests[0].result.rows
        assert first.linked_qualifier == "at station 55+00"
        assert below.linked_investigation_id == "QX-1"
        assert below.linked_qualifier == "at station 55+00"

    def test_a_row_linked_to_neither_is_not_joined_to_one_twins_sheet(self):
        row = SummaryRow(investigation_id="QX-1", depth_top=_m(3.0), wc=17.5)
        sheet = _sheet(depth=3.0, linked_investigation_id="QX-1",
                       linked_qualifier="at station 55+00")
        assert _may_join(row, sheet) is False
        row.linked_investigation_id = "QX-1"
        row.linked_qualifier = "at station 10+00"
        assert _may_join(row, sheet) is False
        row.linked_qualifier = "at station 55+00"
        assert _may_join(row, sheet) is True


# ---------------------------------------------------------------------------
# counted as two holes
# ---------------------------------------------------------------------------

class TestCounts:
    def test_the_narratives_counts_take_them_as_two_holes(self):
        record = reconcile(_twins_record())
        found = explorations_found(record.investigations)
        assert found["boringCount"] == 2
        assert found["boringDictionary"] == ["QX-1", "QX-1"]

    def test_one_hole_read_twice_and_not_told_apart_is_still_one(self):
        holes = [_hole("QX-1", [10]), _hole("QX-01", [11])]
        assert explorations_found(holes)["boringCount"] == 1

    def test_the_identity_a_hole_is_counted_by(self):
        record = reconcile(_twins_record())
        first, second = record.investigations
        assert hole_identity(first) != hole_identity(second)
        assert hole_identity(_hole("QX-01")) == hole_identity(_hole("QX-1"))
        assert hole_label(first) == "QX-1 (ground elevation 52 m)"
        assert hole_label(_hole("QX-2")) == "QX-2"

    def test_the_question_lists_hole_count(self):
        from report_ingest.question_list import _holes
        record = reconcile(_twins_record())
        assert _holes(record.investigations)[0] == 2

    def test_the_reconcilers_own_count_check(self):
        record = _twins_record()
        record.general.boringCount = 2
        reconcile(record)
        assert not [e for e in record.qa if e.kind == "count_mismatch"]


# ---------------------------------------------------------------------------
# the DIGGS file
# ---------------------------------------------------------------------------

def _features(xml):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)
    gml = f"{{{NS['gml']}}}"
    out = []
    for element in root.iter(f"{{{NS['diggs']}}}Borehole"):
        out.append((element.get(f"{gml}id"),
                    element.find(f"{gml}name").text))
    return root, out


def _test_holes(root):
    xlink = "{http://www.w3.org/1999/xlink}"
    return [t.find(f"{{{NS['diggs']}}}samplingFeatureRef")
            .get(f"{xlink}href").lstrip("#")
            for t in root.iter(f"{{{NS['diggs']}}}Test")
            if t.find(f".//{{{NS['diggs']}}}location") is not None
            and t.find(f"{{{NS['diggs']}}}samplingFeatureRef") is not None]


class TestDiggs:
    def _record(self):
        record = _stationed()
        record.lab_tests = [_sheet(page=40, depth=3.0,
                                   fields={"Station": "55+00"}),
                            _sheet(page=41)]
        return reconcile(record)

    def test_distinct_ids_under_the_printed_name(self):
        record = self._record()
        notes = []
        xml = write_diggs(record, notes=notes)
        assert diggs_schema_gate(xml)[0] is True
        _root, features = _features(xml)
        assert [name for _gid, name in features] == ["QX-1", "QX-1"]
        assert len({gid for gid, _name in features}) == 2
        assert notes[0].shared_names == ["QX-1"]
        assert QUALIFIER_PARAMETER in xml
        assert "at station 55+00" in xml

    def test_a_linked_result_is_in_its_own_hole_and_a_refused_one_nowhere(
            self):
        record = self._record()
        notes = []
        xml = write_diggs(record, notes=notes)
        root, features = _features(xml)
        second = features[1][0]
        lab = [gid for gid in _test_holes(root)
               if gid in {g for g, _n in features}]
        assert lab == [second]
        assert any("2 holes are printed under the name 'QX-1'"
                   in line for line in notes[0].skipped)
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert ok, diffs

    def test_the_gate_looks_in_the_one_hole_it_was_linked_to(self):
        record = self._record()
        xml = write_diggs(record)
        record.lab_tests[0].linked_qualifier = "at station 10+00"
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert not ok and any("no result at that hole and depth" in d
                              for d in diffs)

    def test_two_logs_nobody_reconciled_keep_the_old_placing(self):
        record = ReportRecord(investigations=[_hole("QX-1", [10]),
                                              _hole("QX-1", [20])])
        record.lab_tests = [_sheet(page=40)]
        xml = write_diggs(record)
        root, features = _features(xml)
        assert features[1][0] in _test_holes(root)
        assert QUALIFIER_PARAMETER not in xml

    def test_a_twin_left_out_of_the_file_takes_no_other_twins_results(self):
        # ZK-4 at ground 30 m has depths in an unknown unit, so it is left
        # out of the file; the ZK-4 at ground 21 m is written.
        gone = _hole("ZK-4", [10], elevation=30.0, units_known=False,
                     qualifier="ground elevation 30 m")
        kept = _hole("ZK-4", [20], elevation=21.0,
                     qualifier="ground elevation 21 m")
        record = ReportRecord(investigations=[gone, kept])
        record.lab_tests = [
            _sheet("ZK-4", page=40, depth=1.0,
                   linked_investigation_id="ZK-4",
                   linked_qualifier="ground elevation 30 m"),
            _sheet("ZK-4", page=41, depth=3.0),
            _sheet("ZK-4", page=42, depth=3.0,
                   linked_investigation_id="ZK-4",
                   linked_qualifier="ground elevation 21 m")]
        notes = []
        xml = write_diggs(record, notes=notes)
        root, features = _features(xml)
        assert [name for _gid, name in features] == ["ZK-4"]
        # Only the one linked to the written twin is in it.
        assert _test_holes(root) == [features[0][0]]
        skipped = notes[0].skipped
        assert any("lab test 1" in line and "is left out of this file"
                   in line for line in skipped)
        assert any("lab test 2" in line and "2 holes are printed under the "
                   "name 'ZK-4'" in line for line in skipped)
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert ok, diffs
        # The gate asks the same: one linked to the written twin is looked
        # for there, and missed when it is not.
        record.lab_tests[0].linked_qualifier = "ground elevation 21 m"
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert not ok and any("no result at that hole and depth" in d
                              for d in diffs)

    def test_one_holes_logs_each_take_the_results_linked_to_them(self):
        # Not the name's last log, as before: the log holding the sample.
        record = _cut_off()
        record.lab_tests = [
            _sheet("ZK-7", page=40, depth=1.0, sample_id="S-1"),
            _sheet("ZK-7", page=41, depth=10.0, sample_id="S-7"),
            LabTest(kind="summary_table", pages=[45],
                    result=SummaryTableResult(rows=[
                        SummaryRow(investigation_id="ZK-7", sample_id="S-2",
                                   depth_top=_m(3.0), wc=21.5)]))]
        reconcile(record, doc=_CUT_OFF_DOC)
        xml = write_diggs(record)
        root, features = _features(xml)
        assert QUALIFIER_PARAMETER not in xml
        first, last = (gid for gid, _name in features)
        assert _test_holes(root) == [first, last, first]
        (row,) = record.lab_tests[2].result.rows
        assert row.linked_log_pages == "pages 30-31"
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert ok, diffs
        # The gate looks in the log it was linked to, and nowhere else.
        record.lab_tests[0].linked_log_pages = "page 33"
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert not ok and any("no result at that hole and depth" in d
                              for d in diffs)

    def test_one_twins_logs_each_take_the_results_linked_to_them(self):
        record = _stationed()
        record.investigations[1].sheet = "Sheet 1 of 2"
        record.investigations.append(_hole(
            "QX-1", [22], sheet="Sheet 2 of 2", depth=15.0,
            layers=[Layer(top=_m(8.0), bottom=_m(15.0),
                          description="sand")],
            samples=[Sample(sample_id="S-5", top=_m(9.0), bottom=_m(9.5))]))
        record.lab_tests = [
            _sheet(page=40, depth=3.0, fields={"Station": "55+00"}),
            _sheet(page=41, depth=9.0, fields={"Station": "55+00"})]
        reconcile(record)
        xml = write_diggs(record)
        root, features = _features(xml)
        _a, b, b_more = (gid for gid, _name in features)
        assert _test_holes(root) == [b, b_more]
        assert [t.linked_log_pages for t in record.lab_tests] == [
            "page 20", "page 22"]
        ok, diffs = diggs_roundtrip_gate(xml, record)
        assert ok, diffs


# ---------------------------------------------------------------------------
# the summary page
# ---------------------------------------------------------------------------

def test_the_summary_page_shows_each_qualifier_beside_the_name():
    from report_ingest.writers import summary_markdown
    record = _stationed()
    record.lab_tests = [_sheet(page=40, depth=3.0,
                               fields={"Station": "55+00"})]
    page = summary_markdown(reconcile(record))
    assert "QX-1 (at station 10+00)" in page
    assert "QX-1 (at station 55+00)" in page


@pytest.mark.parametrize("cls", [LabTest, SummaryRow])
@pytest.mark.parametrize("field", ["linked_qualifier", "linked_log_pages"])
def test_the_link_qualifier_is_written_only_when_set(cls, field):
    data = {"kind": "moisture_content"} if cls is LabTest else {}
    assert field not in cls(**data).model_dump(mode="json")
    dumped = cls(**{field: "page 20"}, **data).model_dump(mode="json")
    assert dumped[field] == "page 20"
    assert getattr(cls.model_validate(dumped), field) == "page 20"
