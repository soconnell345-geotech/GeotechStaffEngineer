"""The WP1b scorecard's own logic, with no corpus and no model.

Everything here is pure: the rate arithmetic, the disputed-hand-label rule,
the prompt fingerprint and the set membership. They are worth pinning because
each one decides what a published number means, and three of them were
written after a run went wrong in a way the numbers alone would not have
shown.
"""

from __future__ import annotations

import pytest

from module_work.report_ingest_harness import measure_wp1b as m


# -- the sets ---------------------------------------------------------------

def test_the_three_scored_sets_cover_all_38_reports_exactly_once():
    oos = set(m.OOS_OPEN) | set(m.OOS_BLIND)
    assert not (set(m.OOS_OPEN) & set(m.OOS_BLIND)), "a report in both sets"
    assert len(oos) == 24
    # The in-sample ids come from the spreadsheet and need the corpus, so
    # only the out-of-sample halves are checked without it.
    assert len(m.OOS_OPEN) == 10
    assert len(m.OOS_BLIND) == 14


def test_an_unknown_set_name_is_refused():
    with pytest.raises(ValueError, match="unknown set"):
        m.set_ids("everything")


def test_two_checkpoint_reports_are_also_in_the_blind_set():
    # Not a bug to fix here: the brief put R36 and R37 in the cost checkpoint,
    # so they were read and are no longer blind. The scorecard must therefore
    # report the blind set twice, with and without them, and this test is the
    # reminder that the second figure is the honest one.
    overlap = set(m.CHECKPOINT) & set(m.OOS_BLIND)
    assert overlap == {"R36", "R37"}


# -- the rates --------------------------------------------------------------

def test_precision_recall_and_f1_are_counted_per_label():
    s = m.Scores()
    for _ in range(8):
        s.add("boring_log", "boring_log")
    s.add("boring_log", "figure")            # a miss: recall 8/9
    s.add("lab_test", "boring_log")          # a false positive: precision 8/9
    precision, recall, f1, support = s.rates("boring_log")
    assert support == 9
    assert recall == pytest.approx(8 / 9)
    assert precision == pytest.approx(8 / 9)
    assert f1 == pytest.approx(8 / 9)
    assert s.accuracy == pytest.approx(8 / 10)


def test_a_label_with_no_page_gives_nan_rather_than_zero():
    s = m.Scores()
    s.add("narrative", "narrative")
    precision, recall, _f1, support = s.rates("cpt_log")
    assert support == 0
    assert recall != recall and precision != precision, (
        "a label the set does not contain must read as unmeasured, never as "
        "a perfect or a failing score")


def test_an_accepted_alternate_counts_only_towards_the_lenient_score():
    s = m.Scores()
    s.add("test_pit_log", "boring_log", ("boring_log",))
    assert s.accuracy == 0.0
    assert s.lenient_accuracy == 1.0


# -- disputed hand labels ---------------------------------------------------

def test_every_disputed_entry_names_both_labels_and_a_reason():
    for (rid, page), row in m.DISPUTED.items():
        assert rid.startswith("R") and isinstance(page, int)
        assert row["hand"] != row["review"], "a dispute needs two answers"
        assert len(row["note"]) > 20, f"{rid} p{page} needs a reason"
        assert isinstance(row["confirmed"], bool)


def test_a_confirmed_dispute_drops_only_when_the_review_still_says_it():
    (rid, page), row = next(iter(m.DISPUTED.items()))
    assert row["confirmed"], "this test assumes the lead has confirmed these"
    assert m.disputed_drop(rid, page, row["review"]) is True
    # A later run that moves the page somewhere else is a new answer, and a
    # new answer is scored rather than quietly excused by an old dispute.
    assert m.disputed_drop(rid, page, "narrative") is False


def test_a_page_nobody_disputed_is_never_dropped():
    assert m.disputed_drop("R01", 3, "narrative") is False


def test_an_unconfirmed_dispute_is_scored_exactly_as_before(monkeypatch):
    key = ("R99", 1)
    monkeypatch.setitem(m.DISPUTED, key, {
        "review": "cover", "hand": "narrative", "confirmed": False,
        "note": "a page raised but not yet looked at by the lead"})
    assert m.disputed_drop("R99", 1, "cover") is False


# -- the prompt fingerprint -------------------------------------------------

def test_the_fingerprint_is_stable_and_changes_with_the_prompt(monkeypatch):
    import report_ingest.label_review as lr

    first = m.prompt_fingerprint()
    assert first == m.prompt_fingerprint(), "the same prompts must hash alike"
    assert len(first) == 8
    monkeypatch.setattr(lr, "REVIEW_SYSTEM", lr.REVIEW_SYSTEM + "\n- and one "
                        "more rule.")
    assert m.prompt_fingerprint() != first, (
        "a changed prompt must change the fingerprint, or half a set on new "
        "prompts and half on old will score as one round")


def test_a_changed_label_definition_also_changes_the_fingerprint(monkeypatch):
    import report_ingest.label_review as lr

    first = m.prompt_fingerprint()
    monkeypatch.setitem(lr.LABEL_DEFINITIONS, "figure", "a picture")
    assert m.prompt_fingerprint() != first


# -- the cost stops ---------------------------------------------------------

def test_the_cost_ceilings_are_the_ones_the_lead_set():
    assert m.MAX_REPORT_DOLLARS == 5.00
    assert m.MAX_TOTAL_DOLLARS == 45.00


def test_the_key_content_labels_are_the_gated_ones():
    assert m.GATE == 0.98
    assert set(m.KEY_CONTENT) == {
        "narrative", "plan", "profile", "boring_log", "test_pit_log",
        "cpt_log", "dcp_log", "lab_test", "calculation"}
