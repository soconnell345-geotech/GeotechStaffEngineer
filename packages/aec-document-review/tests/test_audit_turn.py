"""Audit of one turn of Document Review, 2026-11-18
(``module_work/audit_2026_11_18/TURN.md``): what the tester is told."""

import aec_document_review.core as core

KEPT = ("What this turn had done is kept: its 3 finished tool results are "
        "now in the conversation's history.")


def test_a_refused_sign_in_says_what_the_turn_kept():
    """A 401 in the middle of a turn dropped the sentence saying what was
    kept and said "nothing is lost" in its place."""
    exc = RuntimeError("Error code: 401 - Unauthorized (made-up)")
    text = core.friendly_turn_error(exc, kept=KEPT)
    assert "refused this app's sign-in" in text
    assert KEPT in text
    assert "nothing is lost" not in text
    # with nothing kept to name, the text is as it was
    plain = core.friendly_turn_error(exc)
    assert "files saved or downloaded this turn are still in the " \
           "conversation" in plain

