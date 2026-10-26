"""The builders' check mode helpers (RI-87 piece 6): made-up page numbers."""

from __future__ import annotations

import pytest

from report_ingest.checks import (
    IMPOSSIBLE, LISTED, PageNumbers, check_mode, doubtful, pdf_local,
)
from report_ingest.model import qa_rank


class TestPdfLocal:
    """An answer's page numbers are read one way for the whole answer."""

    def test_report_pages_are_read_as_cited(self):
        numbers = pdf_local([10, 11, 11], [10, 11])
        assert not numbers.positions and numbers.mixed == ""
        assert numbers.page(11) == (11, "")

    def test_positions_are_read_as_the_reports_pages(self):
        numbers = pdf_local([0, 1, None], [10, 11])
        assert numbers.positions and numbers.mixed == ""
        page, note = numbers.page(1)
        assert page == 11
        assert note == ("the answer cited page 1, a position in the 2 "
                        "page(s) it was sent; read as the report's page 11")

    def test_one_cite_past_the_positions_makes_them_all_pages(self):
        numbers = pdf_local([0, 1, 57], [10, 11])
        assert not numbers.positions
        assert numbers.page(0) == (0, "")

    def test_cites_counted_from_1_are_not_moved_one_by_one(self):
        """Pages 3 to 5 cited 1, 2, 3: 3 is past the last position, so the
        answer is read as the report's pages; 1 and 2 stay as cited, not
        moved to 4 and 5 while 3 stays 3."""
        numbers = pdf_local([1, 2, 3], [3, 4, 5])
        assert not numbers.positions
        assert [numbers.page(c)[0] for c in (1, 2, 3)] == [1, 2, 3]
        # pages from 3 cannot be positions among 3, so the note says only
        # that the answer mixes them
        assert numbers.mixed == (
            "the page numbers this answer cites mix this call's pages and "
            "others: 3 is among this call's pages and 1, 2 are not; they "
            "were read as the report's pages")

    def test_cites_from_0_on_pages_from_1_are_positions_with_a_note(self):
        numbers = pdf_local([0, 1, 2], [1, 2, 3])
        assert numbers.positions
        assert [numbers.page(c)[0] for c in (0, 1, 2)] == [1, 2, 3]
        assert numbers.mixed == (
            "the page numbers this answer cites mix this call's pages and "
            "others: 1, 2 are among this call's pages and 0 is not, and with "
            "the call's pages 1, 2, 3 each could also be a position among "
            "the 3 pages sent; all were read as positions")

    def test_a_mix_read_as_pages_says_so(self):
        numbers = pdf_local([1, 4], [1, 2, 3])
        assert not numbers.positions
        assert numbers.mixed.endswith(
            "they were read as the report's pages")

    def test_any_mix_is_noted(self):
        """RI-89 review: a mix is noted whether or not the call's pages
        could also be positions."""
        assert pdf_local([10, 57], [10, 11]).mixed == (
            "the page numbers this answer cites mix this call's pages and "
            "others: 10 is among this call's pages and 57 is not; they were "
            "read as the report's pages")
        assert pdf_local([57], [10, 11]).mixed == ""

    def test_no_cites_is_no_change(self):
        assert pdf_local([], [4, 5]) == PageNumbers((4, 5))
        assert pdf_local([None], []).page(3) == (3, "")


class TestWords:

    def test_the_mode_is_checked(self):
        assert check_mode("flag") == "flag"
        with pytest.raises(ValueError):
            check_mode("lenient")

    def test_listed_and_impossible_rank_open(self):
        assert IMPOSSIBLE.endswith(LISTED)
        assert qa_rank("out_of_range", f"x; {IMPOSSIBLE}") == "open"
        assert qa_rank("out_of_range", f"x; it is {LISTED}") == "open"
        assert qa_rank("out_of_range", doubtful("x")) == "open"
