"""Where a bound document begins and ends, decided on plain values.

No document, no model, no engine: :mod:`report_ingest.bound` is arithmetic
over a triage answer and a page-label map, and this is that arithmetic. The
graph's own tests (``test_graph.TestAReportBoundInsideAReport``) then run the
whole thing on the synthetic report.
"""

from __future__ import annotations

from types import SimpleNamespace

from report_ingest.bound import (
    MIN_BOUND_PAGES, BoundRange, bound_ranges, compact_pages, label_runs,
    narrative_block, parse_range, triage_runs,
)


def _profile(*entries):
    """A triage profile with just the field this module reads."""
    return SimpleNamespace(bound_together=list(entries))


def _labels(*runs, n_pages: int = 30):
    """A page-label map where the given page ranges are appended_report."""
    out = {page: "narrative" for page in range(n_pages)}
    for run in runs:
        for page in run:
            out[page] = "appended_report"
    return out


class TestReadingAPageRange:

    def test_a_simple_range(self):
        assert parse_range("112-118", 200) == list(range(112, 119))

    def test_a_split_range_and_a_single_page(self):
        assert parse_range("0-2,9", 30) == [0, 1, 2, 9]

    def test_an_en_dash_is_a_dash(self):
        assert parse_range("4–6", 30) == [4, 5, 6]

    def test_a_backwards_range_is_read_forwards(self):
        assert parse_range("18-15", 30) == [15, 16, 17, 18]

    def test_pages_past_the_end_are_clipped_not_raised(self):
        # A model that counted wrong about a 20-page file still told us
        # something true about its first pages.
        assert parse_range("18-25", 20) == [18, 19]

    def test_rubbish_is_an_empty_list(self):
        assert parse_range("the appendix", 20) == []
        assert parse_range(None, 20) == []

    def test_a_list_of_pages_is_accepted(self):
        assert parse_range([3, 1, 2, 2], 20) == [1, 2, 3]


class TestSpellingARange:

    def test_a_run_is_a_dash(self):
        assert compact_pages([15, 16, 17, 18]) == "15-18"

    def test_a_single_page_is_a_number(self):
        assert compact_pages([7]) == "7"

    def test_two_runs_are_comma_separated(self):
        assert compact_pages([0, 1, 2, 9, 10]) == "0-2,9-10"

    def test_nothing_is_empty(self):
        assert compact_pages([]) == ""


class TestWhatEachSourceSays:

    def test_the_labels_give_their_appended_report_runs(self):
        labels = _labels(range(5, 9), range(20, 23))
        assert label_runs(labels) == [[5, 6, 7, 8], [20, 21, 22]]

    def test_triage_splits_one_entry_into_its_contiguous_runs(self):
        rows = triage_runs(
            _profile({"kind": "volume", "pages": "0-2,9-11",
                      "title": "Volume 2"}), 30)
        assert [row["pages"] for row in rows] == [[0, 1, 2], [9, 10, 11]]
        assert all(row["kind"] == "volume" for row in rows)
        assert all(row["title"] == "Volume 2" for row in rows)

    def test_a_profile_with_nothing_bound_says_nothing(self):
        assert triage_runs(_profile(), 30) == []
        assert triage_runs(None, 30) == []


class TestTheUnionOfTheTwo:

    def test_both_sources_agreeing_is_one_document_said_by_both(self):
        ranges, qa = bound_ranges(
            _profile({"kind": "appended_prior_report", "pages": "10-15",
                      "title": "An earlier study"}),
            _labels(range(10, 16)), 30)

        (row,) = ranges
        assert row.pages == list(range(10, 16))
        assert row.said_by == ("planlens", "triage")
        assert row.title == "An earlier study"
        assert row.bound_id == "bound1"
        assert not [e for e in qa if e.where == "bound.extent"]

    def test_the_rules_alone_still_make_a_document(self):
        ranges, qa = bound_ranges(_profile(), _labels(range(10, 16)), 30)

        (row,) = ranges
        assert row.said_by == ("planlens",)
        assert row.kind == "appended_prior_report"
        assert not qa

    def test_triage_alone_still_makes_a_document(self):
        ranges, _qa = bound_ranges(
            _profile({"kind": "data_report", "pages": "10-15", "title": ""}),
            _labels(), 30)

        (row,) = ranges
        assert row.said_by == ("triage",) and row.kind == "data_report"

    def test_overlapping_edges_become_the_union_with_a_note(self):
        ranges, qa = bound_ranges(
            _profile({"kind": "appended_prior_report", "pages": "9-14",
                      "title": ""}),
            _labels(range(10, 16)), 30)

        (row,) = ranges
        assert row.pages == list(range(9, 16))
        assert row.spec == "9-15"
        (note,) = [e for e in qa if e.where == "bound.extent"]
        assert note.values == ["triage 9-14", "planlens 10-15"]
        assert "union" in note.detail

    def test_adjacent_claims_are_one_document(self):
        # A tab at page 9 and the pages of the report behind it are one
        # thing, not two documents that happen to touch.
        ranges, _qa = bound_ranges(
            _profile({"kind": "other", "pages": "9", "title": ""}),
            _labels(range(10, 16)), 30)

        (row,) = ranges
        assert row.pages == list(range(9, 16))

    def test_two_separate_documents_stay_two(self):
        ranges, _qa = bound_ranges(
            _profile(), _labels(range(4, 9), range(20, 26)), 30)

        assert [row.spec for row in ranges] == ["4-8", "20-25"]
        assert [row.bound_id for row in ranges] == ["bound1", "bound2"]

    def test_a_run_under_the_floor_is_appended_pages_not_a_document(self):
        short = list(range(10, 10 + MIN_BOUND_PAGES - 1))
        ranges, qa = bound_ranges(_profile(), _labels(short), 30)

        assert ranges == []
        (note,) = [e for e in qa if e.where == "bound.short_run"]
        assert note.pages == short
        assert "stay in this record" in note.detail

    def test_the_floor_can_be_lowered_by_the_caller(self):
        ranges, _qa = bound_ranges(_profile(), _labels([10, 11]), 30,
                                   min_pages=2)

        assert [row.spec for row in ranges] == ["10-11"]

    def test_nothing_bound_is_nothing_said(self):
        assert bound_ranges(_profile(), _labels(), 30) == ([], [])


class TestTheRangeItself:

    def test_it_knows_its_own_edges(self):
        row = BoundRange(bound_id="bound1", pages=[15, 16, 17, 18])

        assert (row.first_page, row.last_page, row.n_pages) == (15, 18, 4)
        assert row.spec == "15-18"
        assert row.to_dict()["pages"] == "15-18"


class TestWhatTheNarrativeReaderIsTold:

    def test_it_names_the_pages_and_says_what_to_do_about_them(self):
        text = narrative_block([{"pages": "15-18", "title": "An older study",
                                 "firm": "Another firm", "date": "2019"}])

        assert "pages 15-18" in text
        assert "An older study - Another firm - 2019" in text
        assert "previousInvestigationCount" in text
        assert "boringDictionary" in text

    def test_a_document_that_prints_no_title_still_gets_a_line(self):
        text = narrative_block([{"pages": "15-18"}])

        assert "title not printed" in text

    def test_nothing_bound_is_no_block_at_all(self):
        assert narrative_block([]) == ""
        assert narrative_block(None) == ""
