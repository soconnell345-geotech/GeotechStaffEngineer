"""Review of BACKLOG DR-25: ``scope="finding"`` looks for each term one
piece at a time, so a term that ends at the end of its piece (a table
row's last cell) is found whatever row follows it. Made-up values."""

from aec_agent_core.review_eval.checks import check_stated_together

#: A cell that answers its head with no, at the end of what is read.
_NO_CELL = {"re": r"carried\?:\s*no(?=\s*$)"}
_ROWS = ("| Sheet | Item | Carried? |\n|---|---|---|\n"
         "| A-301 | Parapet 109.30 | No |\n")


def test_a_term_at_the_end_of_a_row_is_found_when_a_row_follows():
    last = check_stated_together(_ROWS, ["a-301", _NO_CELL], scope="finding")
    assert last[0], last
    more = check_stated_together(_ROWS + "| A-101 | Stair | Yes |\n",
                                 ["a-301", _NO_CELL], scope="finding")
    assert more[0], more


def test_a_term_in_no_piece_is_still_named_missing():
    ok, detail = check_stated_together(
        "| Sheet | Item | Carried? |\n|---|---|---|\n"
        "| A-301 | Parapet | Yes |\n| A-101 | Stair | Yes |\n",
        ["a-301", _NO_CELL], scope="finding")
    assert not ok and detail.startswith("missing:")
