"""The review of ``coord-second-issue`` (brief 15): twenty made-up answers
to its question, each a full finding with a reason and an action as a
reviewer writes one, read by the task's fourth check ("bearing-pressure,
said not flagged") - offline, no model, no network.

Ten find the S-001 change and do NOT say it is unmarked; each holds a
"no", "not", "never" or "without" about something else (the footings, the
price, the report), as such findings do. Ten say it, each in other words.
The check must fail the first ten and pass the second ten, and the third
check (the change found) must pass all twenty. All made-up values.
"""

import hashlib

import pytest

pytest.importorskip("pymupdf")

from aec_agent_core.review_eval import checks as C  # noqa: E402
from aec_document_review.review_eval.tasks import (  # noqa: E402
    COORDINATION_TASKS, _UNFLAGGED)

BY_ID = {t.id: t for t in COORDINATION_TASKS}
NEW = "coord-second-issue"
FOUND = "bearing-pressure"
SAID = "bearing-pressure, said not flagged"


def _check(label):
    (check,) = [c for c in BY_ID[NEW].checks if c["label"] == label]
    return check


def _table(heads, *rows):
    def line(cells):
        return "| " + " | ".join(cells) + " |"
    return "\n".join([line(heads), line(["---"] * len(heads))] +
                     [line(r) for r in rows])


#: Found, and nothing said of how the change was issued.
NOT_SAID = {
    "footings-not-resized": (
        "- **S-001, structural note 2.** The allowable bearing pressure now "
        "reads 200 kPa; the earlier issue read 150 kPa. Why it matters: the "
        "footing schedule on S-501 has not changed, so the footings are "
        "still sized for the lower value. Action: ask the engineer whether "
        "the footings will be resized before you confirm the concrete "
        "price."),
    "no-price-change-yet": (
        "3. S-001 (General Structural Notes): note 2 gives an allowable "
        "bearing pressure of 200 kPa, up from 150 kPa. On its own this is "
        "no change to quantities, but it may alter subgrade preparation "
        "and proof-rolling. Recommend an RFI to the structural engineer."),
    "without-a-report": (
        "- S-001: allowable bearing pressure is 200 kPa (was 150 kPa).\n"
        "  - Reason: a one-third increase without a revised geotechnical "
        "report behind it is unusual.\n"
        "  - Action: request the basis of the 200 kPa value and price "
        "ground improvement as an allowance until it is confirmed."),
    "do-not-rely": (
        "**S-001, note 2: bearing pressure 150 kPa to 200 kPa.** The "
        "design basis of every footing moved. Do not rely on the earlier "
        "footing take-off until the engineer confirms which value governs; "
        "carry the earlier sizes in the price for now."),
    "table-affects-price": _table(
        ["Sheet", "What it says now", "Was", "Affects our price?", "Action"],
        ["A-401", "Grids B to C 7500", "7200", "Yes", "Re-measure the core"],
        ["A-601", "Door 107 HW-2", "HW-1", "Yes", "Re-quote hardware"],
        ["S-001", "Allowable bearing pressure 200 kPa", "150 kPa", "No",
         "Confirm with the engineer"]),
    "table-quantities": _table(
        ["Sheet", "Now", "Before", "Quantity change", "Next step"],
        ["S-001", "Note 2: 200 kPa allowable bearing", "150 kPa",
         "None until footings are redesigned", "RFI to the engineer"]),
    "never-tested": (
        "- S-001 note 2: 200 kPa allowable bearing pressure, was 150 kPa. "
        "A value this high is never assumed for fill; check it against the "
        "memorandum and ask who approved it. Hold the earthworks price "
        "until it is answered."),
    "not-a-saving": (
        "S-001 now states an allowable bearing pressure of 200 kPa where "
        "the earlier issue stated 150 kPa. This is not a saving you can "
        "bank: S-501 still schedules F1 to F3 at their earlier sizes. Ask "
        "the engineer to confirm the footing sizes in writing."),
    "typo": (
        "4. **S-001** reads 200 kPa for the allowable bearing pressure "
        "(150 kPa before). Nothing else on the structural sheets moved "
        "with it, so this may be a drafting error and not a design change. "
        "Query it before you reprice the foundations."),
    "under-heading": (
        "### Structural\n\n"
        "- S-001, note 2: allowable bearing pressure is now 200 kPa "
        "(150 kPa in the earlier issue). No footing on S-501 was made "
        "smaller to match. Action: confirm the design basis with the "
        "engineer; no repricing until then."),
}

#: Found, and said not to be marked, each in other words.
IS_SAID = {
    "no-cloud": (
        "- **S-001, structural note 2.** The allowable bearing pressure now "
        "reads 200 kPa; the earlier issue read 150 kPa. There is no cloud "
        "around the note. Why it matters: footing sizes follow from this "
        "value. Action: ask the engineer to confirm it."),
    "not-clouded": (
        "3. S-001: allowable bearing pressure 200 kPa (was 150 kPa). The "
        "note is not clouded, so it is easy to miss. Recommend an RFI to "
        "the structural engineer before the foundation price is fixed."),
    "unclouded": (
        "S-001 note 2 now gives 200 kPa for the allowable bearing pressure "
        "against 150 kPa before, an unclouded change. Have the engineer "
        "confirm it and reissue the sheet."),
    "not-in-revision-table": (
        "- S-001: bearing pressure is 200 kPa, was 150 kPa.\n"
        "  - This change is not in the revision table of the sheet.\n"
        "  - Action: confirm with the engineer which value governs."),
    "no-revision-row": (
        "**S-001, note 2: bearing pressure 150 kPa to 200 kPa.** The "
        "sheet has no revision row for it and still reads revision 0. "
        "Carry the earlier footing sizes until the engineer answers."),
    "silent": (
        "S-001 is a silent change: note 2 gives an allowable bearing "
        "pressure of 200 kPa where the earlier issue gave 150 kPa. Raise "
        "it with the engineer, since the footings depend on it."),
    "unannounced": (
        "4. **S-001** reads 200 kPa for the allowable bearing pressure "
        "(150 kPa before). The change is unannounced. Query it before you "
        "reprice the foundations."),
    "without-revision-entry": (
        "- S-001 note 2: 200 kPa allowable bearing pressure, was 150 kPa, "
        "changed without a revision entry. Ask who approved it and hold "
        "the earthworks price until it is answered."),
    "table-not-flagged": _table(
        ["Sheet", "What it says now", "Was", "How it was issued", "Action"],
        ["A-401", "Grids B to C 7500", "7200", "Clouded, revision 1",
         "Re-measure the core"],
        ["S-001", "Allowable bearing pressure 200 kPa", "150 kPa",
         "Not flagged anywhere", "Confirm with the engineer"]),
    "no-delta": (
        "### Structural\n\n"
        "- S-001, note 2: allowable bearing pressure is now 200 kPa "
        "(150 kPa in the earlier issue). No delta beside the note. "
        "Action: confirm the design basis with the engineer."),
}


@pytest.mark.parametrize("name", sorted(NOT_SAID) + sorted(IS_SAID))
def test_every_one_of_the_twenty_finds_the_change(name):
    reply = NOT_SAID.get(name) or IS_SAID[name]
    got = C.run_check(_check(FOUND), reply)
    assert got["passed"], got["detail"]


@pytest.mark.parametrize("name", sorted(NOT_SAID))
def test_found_and_not_said_unmarked_fails_the_fourth_check(name):
    got = C.run_check(_check(SAID), NOT_SAID[name])
    assert not got["passed"], got["detail"]


@pytest.mark.parametrize("name", sorted(IS_SAID))
def test_found_and_said_unmarked_passes_the_fourth_check(name):
    got = C.run_check(_check(SAID), IS_SAID[name])
    assert got["passed"], got["detail"]


#: Other ways of saying it, and of not saying it, that the term reads.
MORE = [
    ("- S-001: bearing pressure is 200 kPa, was 150 kPa. It isn't clouded.",
     True),
    ("- S-001: bearing pressure is 200 kPa, was 150 kPa, neither clouded "
     "nor listed.", True),
    ("- S-001: bearing pressure is 200 kPa, was 150 kPa. The revision block "
     "does not show it.", True),
    (_table(["Sheet", "Now", "Was", "Clouded?"],
            ["S-001", "200 kPa allowable bearing", "150 kPa", "No"]), True),
    ("- S-001: bearing pressure is 200 kPa, was 150 kPa. Flag it to the "
     "engineer - do not reprice the footings yet.", False),
    ("- S-001: bearing pressure is 200 kPa, was 150 kPa. Do not confirm "
     "the price until this has been flagged with the engineer.", False),
    (_table(["Sheet", "Now", "Was", "Clouded?"],
            ["S-001", "200 kPa allowable bearing", "150 kPa", "Yes"]),
     False),
]


@pytest.mark.parametrize("reply,said", MORE)
def test_other_wordings(reply, said):
    assert C.run_check(_check(FOUND), reply)["passed"]
    assert C.run_check(_check(SAID), reply)["passed"] is said


def test_the_twin_and_its_term_are_as_they_were():
    """The repair is this task's alone: ``coord-revisions`` and
    ``_UNFLAGGED`` as on the commit the builder's branch starts from."""
    twin = BY_ID["coord-revisions"]
    digest = hashlib.sha256(repr(twin.to_dict()).encode()).hexdigest()
    assert digest == ("7dc000ac3e4039fc39f3c4ea4571298dc5c5c8777713039f3b4a"
                      "a938469c88b7")
    assert twin.checks[-1]["terms"][-1] is _UNFLAGGED
    assert hashlib.sha256(_UNFLAGGED["re"].encode()).hexdigest() == (
        UNFLAGGED_DIGEST)
    assert _check(SAID)["terms"][-1] is not _UNFLAGGED
    # the truth still passes all four of its own checks
    assert all(C.run_check(c, BY_ID[NEW].truth)["passed"]
               for c in BY_ID[NEW].checks)


UNFLAGGED_DIGEST = "bd95e1d5eaabb836c18933b1950155d5c9c7d49d0489c28b29997d8be9a81f8e"
