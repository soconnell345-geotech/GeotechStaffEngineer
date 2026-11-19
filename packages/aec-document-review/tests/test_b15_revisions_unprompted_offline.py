"""The second question on issues 0 and 1 of the drawing set
(``coord-second-issue``, brief 15) - offline, no model, no network.

``coord-revisions`` asks, for each change, whether it is clouded and listed
in the revision block, so it cannot show whether a reviewer finds a change
nobody marked without being told to look. ``coord-second-issue`` asks the
same two documents what a contractor asks who priced the earlier issue, and
scores apart: each of the two marked changes found, the unmarked change
found (sheet and new value together), and the answer saying that this
change is not marked. These tests hold the task to that: its question names
nothing to look for, ``coord-revisions`` is as it was, and each check takes
the made-up replies it should and no other. All made-up values.
"""

import hashlib
import re

import pytest

pytest.importorskip("pymupdf")

from aec_agent_core.review_eval import checks as C  # noqa: E402
from aec_document_review.review_eval import (  # noqa: E402
    coordination_fixture as F, score_review_suite)
from aec_document_review.review_eval.tasks import (  # noqa: E402
    COORDINATION_TASKS, OPEN_TASKS, _UNFLAGGED)

BY_ID = {t.id: t for t in COORDINATION_TASKS}
NEW = "coord-second-issue"
FOUND = "bearing-pressure"
SAID = "bearing-pressure, said not flagged"
LABELS = ["core-dimension", "door-hardware", FOUND, SAID]


def _passed(answer: str, task_id: str = NEW):
    return [c["label"] for c in BY_ID[task_id].checks
            if C.run_check(c, answer)["passed"]]


def _table(heads, *rows):
    def line(cells):
        return "| " + " | ".join(cells) + " |"
    return "\n".join([line(heads), line(["---"] * len(heads))] +
                     [line(r) for r in rows])


# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------

def test_the_task_is_listed_beside_its_twin_on_the_same_documents():
    ids = [t.id for t in COORDINATION_TASKS]
    assert ids.count(NEW) == 1 and len(ids) == 17 and len(OPEN_TASKS) == 55
    assert ids.index(NEW) == ids.index("coord-revisions") + 1
    task, twin = BY_ID[NEW], BY_ID["coord-revisions"]
    assert task in OPEN_TASKS
    assert task.documents == twin.documents == [
        "fixture_coord_set_prev", "fixture_coord_set"]
    assert (task.category, task.doc_type, task.page, task.split) == (
        twin.category, twin.doc_type, twin.page, twin.split)
    assert [c["label"] for c in task.checks] == LABELS
    assert {c["type"] for c in task.checks} == {"stated_together"}
    # a selection by the twin's id still runs the twin alone
    assert not NEW.startswith("coord-revisions")


def test_the_question_names_nothing_to_look_for():
    q = BY_ID[NEW].question.lower()
    for word in ("cloud", "revision", "rev.", "block", "table", "flag",
                 "delta", "listed", "unannounced", "silent", "unmarked",
                 "marked", "undocumented", "s-001", "a-401", "a-601", "107",
                 "200", "150", "kpa", "bearing", "7500", "hw-", "note",
                 "three", "two changes"):
        assert word not in q, word
    # the reviewer's own words would not pass a check either
    assert _passed(BY_ID[NEW].question) == []
    assert not re.search(_UNFLAGGED["re"], q)


def test_the_twin_is_exactly_as_it_was():
    twin = BY_ID["coord-revisions"]
    assert twin.question == (
        "These are two issues of the same drawing set. What changed in "
        "issue 1? For each change, say whether it is clouded and listed in "
        "the sheet's revision block.")
    assert [c["label"] for c in twin.checks] == [
        "core-dimension", "door-hardware",
        "bearing-pressure, and that it is not flagged"]
    # its question, checks and truth, as recorded before this task was added
    digest = hashlib.sha256(repr(twin.to_dict()).encode()).hexdigest()
    assert digest == TWIN_DIGEST


#: sha256 of ``repr(coord-revisions.to_dict())`` on the commit this branch
#: starts from (the 1.3.0 release commit).
TWIN_DIGEST = "7dc000ac3e4039fc39f3c4ea4571298dc5c5c8777713039f3b4aa938469c88b7"


def test_the_checks_are_the_twins_but_found_and_said_are_apart():
    task, twin = {c["label"]: c for c in BY_ID[NEW].checks}, {
        c["label"]: c for c in BY_ID["coord-revisions"].checks}
    for label in ("core-dimension", "door-hardware"):
        assert task[label] == twin[label]
    old = twin["bearing-pressure, and that it is not flagged"]
    assert task[FOUND]["terms"] == old["terms"][:2]
    assert task[SAID]["terms"] == old["terms"]
    assert task[SAID]["terms"][-1] is _UNFLAGGED
    assert task[SAID]["scope"] == "finding"


def test_the_truth_is_the_fixtures_own():
    truth = BY_ID[NEW].truth
    for _key, _sheet, what, flagged in F.REVISIONS:
        if flagged:
            assert what in truth
    assert "150 kPa to 200 kPa with no cloud and no entry" in truth
    assert _passed(truth) == LABELS
    assert "coordination_fixture.py" in truth


def test_no_false_alarm_check_reads_this_task():
    """The task scores no false alarms: its sound sheets, named whole in an
    answer with words that say nothing is wrong or something is missing,
    cost nothing; only the two checks every task gets could fail."""
    task = BY_ID[NEW]
    assert all(c["type"] != "no_false_alarm" for c in task.checks)
    auto = [c for c in task.all_checks() if c not in task.checks]
    assert [c["type"] for c in auto] == ["not_contains", "not_contains"]
    sound = PERFECT + (
        "\nNo change on G-002, C-101, A-101, A-102, A-201, A-301, A-501, "
        "S-101, S-201, S-202 or S-501: the door schedule is not missing a "
        "row, detail 3/A-501 is not broken, and footing F1 is unchanged. "
        "Issue 0 and issue 1 were compared sheet by sheet.")
    for check in task.all_checks():
        assert C.run_check(check, sound)["passed"], check["label"]


def test_the_dry_run_lists_the_task_with_its_documents(tmp_path):
    out = score_review_suite(out_dir=str(tmp_path), dry_run=True, ids=[NEW])
    text = out["results_md"]
    assert "1 tasks" in text or "1 task" in text
    assert NEW in text and "MISSING" not in text.upper()
    alone = score_review_suite(out_dir=str(tmp_path / "twin"), dry_run=True,
                               ids=["coord-revisions"])["results_md"]
    assert NEW not in alone


# ---------------------------------------------------------------------------
# The stand-in answers
# ---------------------------------------------------------------------------

PERFECT = """Three things differ between the two issues and should be \
looked at again before the price is confirmed:

- A-401, enlarged core plans: the bay between grids B and C is now 7500 \
(was 7200). Clouded, revision 1.
- A-601, door schedule: door 107 now takes hardware set HW-2 (was HW-1). \
Clouded, revision 1.
- S-001, structural note 2: the allowable bearing pressure is now 200 kPa \
(was 150 kPa). This change is not clouded and the sheet's revision block \
has no entry for it, so confirm it with the engineer before relying on it.
"""

# reads only what the later issue marks as changed
CARELESS = """Two revisions affect the price:

- A-401: grids B to C revised to 7500 (revision 1, REVISED CORE LAYOUT).
- A-601: door 107 hardware revised to HW-2 (revision 1).

S-001 and the other structural sheets remain at revision 0, so the \
structural price stands."""

# finds all three and says nothing of how the third was issued
FOUND_ONLY = """- A-401: grids B to C are 7500, were 7200.
- A-601: door 107 is HW-2, was HW-1.
- S-001: allowable bearing pressure is 200 kPa, was 150 kPa."""

NOTHING = ("The later issue matches the earlier one on every sheet I "
           "compared: G-001, A-401, A-601, S-001 and S-501. I found no "
           "differences that affect the price.")


def test_the_stand_in_answers_score_as_they_should():
    assert _passed(PERFECT) == LABELS
    assert _passed(CARELESS) == ["core-dimension", "door-hardware"]
    assert _passed(FOUND_ONLY) == LABELS[:3]
    assert _passed(NOTHING) == [] and _passed("") == []
    # the twin reads the same answers its own way: nothing here moves it
    assert _passed(PERFECT, "coord-revisions") == [
        "core-dimension", "door-hardware",
        "bearing-pressure, and that it is not flagged"]
    assert _passed(FOUND_ONLY, "coord-revisions") == [
        "core-dimension", "door-hardware"]


_HEADS = ["Sheet", "What it says now", "Was", "Flagged?"]

#: For each check, made-up replies that must pass it and replies that must
#: fail it.
REPLIES = {
    "core-dimension": ([
        "A-401: the B to C bay on the enlarged plans is 7500 now.",
        "1. Core layout (A401): grid B-C dimension 7,500 mm, up from 7,200.",
        _table(_HEADS, ["A-401", "B-C = 7.5 m", "7.2 m", "Yes"]),
    ], [
        "A-401 was revised (REVISED CORE LAYOUT).",
        "A-401 still dimensions grids B to C at 7200.",
        "Sheets compared: A-401, A-501, A-601.",
    ]),
    "door-hardware": ([
        "Door 107 now has hardware set HW-2.",
        "- A-601: 107 changed from HW-1 to HW 2.",
        _table(_HEADS, ["A-601", "Door 107: HW2", "HW-1", "Yes"]),
    ], [
        "A-601: door hardware revised.",
        "Door 107 has hardware set HW-1.",
        "Door 108 is HW-2 and EL. 107.20 is unchanged.",
    ]),
    FOUND: ([
        "S-001 note 2 gives an allowable bearing pressure of 200 kPa; the "
        "earlier issue gave 150 kPa.",
        "- S001: bearing pressure 200 kPa (was 150).",
        "S-001 reads 200 kPa where the earlier issue read 150 kPa, which "
        "looks like a drafting error.",
        _table(_HEADS, ["S-001", "200 kPa allowable bearing", "150 kPa",
                        "No"]),
    ], [
        "S-001: the allowable bearing pressure is 150 kPa.",
        "The structural notes on S-001 are unchanged.",
        "S-001 note 3 calls for 200 mm of compacted granular fill.",
        "The bearing pressure went up by a third.",
    ]),
    SAID: ([
        "S-001 note 2 now gives 200 kPa (was 150 kPa); the change is not "
        "clouded.",
        "- S-001: allowable bearing pressure 200 kPa, was 150 kPa. Never "
        "listed in the revision block.",
        "- S-001: 200 kPa bearing (150 before)\n  Unmarked: the sheet "
        "still shows revision 0.",
        "S-001 silently changes the bearing pressure to 200 kPa.",
        _table(_HEADS, ["A-401", "B-C = 7500", "7200", "Yes"],
               ["S-001", "200 kPa allowable bearing", "150 kPa", "No"]),
        "### Changes without a revision mark\n\n- S-001 note 2: 200 kPa, "
        "was 150 kPa.",
    ], [
        # found, and the rest of the finding says nothing of its marking
        "S-001 note 2 gives an allowable bearing pressure of 200 kPa; the "
        "earlier issue gave 150 kPa.",
        "S-001 reads 200 kPa where the earlier issue read 150 kPa, which "
        "looks like a drafting error.",
        # said of another change, in the neighbouring finding
        "- A-401: B to C is 7500. Not clouded on the upper plan.\n"
        "- S-001: allowable bearing pressure 200 kPa, was 150 kPa.",
        # every row answered yes
        _table(_HEADS, ["A-401", "B-C = 7500", "7200", "Yes"],
               ["S-001", "200 kPa allowable bearing", "150 kPa", "Yes"]),
        # an all-clear beside it
        "- S-001: allowable bearing pressure 200 kPa, was 150 kPa.\n"
        "- No other differences were found.",
        # the sheet and the word in one finding, the value in another
        "- S-001 is at revision 0 and has no cloud.\n"
        "- The memorandum recommends 200 kPa.",
    ]),
}


def test_every_check_has_its_replies():
    assert sorted(REPLIES) == sorted(LABELS)
    for right, wrong in REPLIES.values():
        assert len(right) >= 3 and len(wrong) >= 3


@pytest.mark.parametrize("label", LABELS)
def test_each_check_takes_the_right_replies_and_no_wrong_one(label):
    (check,) = [c for c in BY_ID[NEW].checks if c["label"] == label]
    right, wrong = REPLIES[label]
    for reply in right:
        got = C.run_check(check, reply)
        assert got["passed"], (reply, got["detail"])
    for reply in wrong:
        assert not C.run_check(check, reply)["passed"], reply


# ---------------------------------------------------------------------------
# What the fourth check does with two answers a reviewer may well give
# ---------------------------------------------------------------------------

def test_a_change_called_a_drafting_error_is_found_and_not_said():
    """Finding the change and calling it a mistake is not saying nobody
    marked it: the third check passes, the fourth does not."""
    for reply in (
            "- S-001 note 2: allowable bearing pressure reads 200 kPa, was "
            "150 kPa. Likely a drafting error; ask the engineer.",
            "S-001 gives 200 kPa where the earlier issue gave 150 kPa. This "
            "looks like a typo and should be queried."):
        assert _passed(reply) == [FOUND], reply
    # the limit: "not" in the same finding, said of something else, passes
    loose = ("- S-001 note 2: 200 kPa, was 150 kPa. Probably a drafting "
             "error, as the footings on S-501 were not resized.")
    assert _passed(loose) == [FOUND, SAID]


def test_a_table_with_a_flagged_column_is_read_row_by_row():
    """Three changes in a table with a yes/no column: the fourth check
    reads the S-001 row with its column head, so "No" there passes and
    "Yes" does not, whatever the other rows say."""
    def rows(s001):
        return _table(
            ["Sheet", "Change", "Flagged?"],
            ["A-401", "Grids B to C 7200 to 7500", "Yes"],
            ["A-601", "Door 107 HW-1 to HW-2", "Yes"],
            ["S-001", "Bearing pressure 150 kPa to 200 kPa", s001])
    assert _passed(rows("No")) == LABELS
    assert _passed(rows("Yes")) == LABELS[:3]
    assert _passed(rows("")) == LABELS[:3]
    # the limit: a "No" under any other head passes too
    other = _table(["Sheet", "Change", "Affects our price?"],
                   ["S-001", "Bearing pressure 150 kPa to 200 kPa", "No"])
    assert _passed(other) == [FOUND, SAID]
