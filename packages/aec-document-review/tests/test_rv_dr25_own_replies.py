"""Review of BACKLOG DR-25: the nine checks against replies the reviewer
wrote, not the builder. Each test names what a careful reviewer would
write (it must pass) and what says the opposite (it must not). Made-up
values, the fixture's own building."""

import pytest

from aec_agent_core.review_eval.checks import check_stated_together
from aec_document_review.review_eval.tasks import _PLANTED_CHECKS


def _passes(key, reply):
    check = dict(_PLANTED_CHECKS[key])
    check.pop("type")
    return check_stated_together(reply, **check)[0]


# -- a sheet left behind, in plain words -----------------------------------

@pytest.mark.parametrize("reply", [
    "The sections (A-301) contradict the elevations: the parapet is at "
    "109.30 on one and 109.60 on the other.",
    "The parapet change on A-201 was not picked up on A-301.",
    "A-301 shows the old parapet level of 109.30.",
    "- A-301: parapet level 109.30\n  - should read 109.60 to agree with "
    "A-201",
])
def test_a_twin_left_behind_in_plain_words(reply):
    assert _passes("parapet-twin", reply)


@pytest.mark.parametrize("reply", [
    "A-301 no longer shows the old parapet level; it reads 109.60.",
    "A-301 does not contradict A-201 on the parapet.",
    "A-301: the parapet should be looked at again; I did not open it.",
])
def test_a_twin_that_followed_is_not_left_behind(reply):
    assert not _passes("parapet-twin", reply)


@pytest.mark.parametrize("reply", [
    "A-101 shows door 106, which is no longer in the schedule (A-601 has "
    "110 instead).",
    "Door 106 on A-101 has no row in the door schedule any more; the "
    "schedule's 110 has no tag.",
])
def test_a_tag_with_no_schedule_row(reply):
    assert _passes("door-renumbered", reply)


def test_a_tag_that_was_renumbered_too():
    assert not _passes(
        "door-renumbered",
        "Door 106 became 110 on the first floor plan A-101 and in the "
        "schedule; the two agree.")


# -- looked for and not found ----------------------------------------------

@pytest.mark.parametrize("key, reply", [
    ("window-anchorage",
     "I could not find the anchorage calculations anywhere in the package."),
    ("window-anchorage", "The submittal has nothing for SD-05 Design Data."),
    ("window-anchorage",
     "Anchorage calculations: not found in the submittal."),
    ("window-section-missing",
     "Section 08 80 00 cannot be found in the manual's table of contents."),
    ("window-section-missing",
     "The reference to Section 08 80 00 does not resolve."),
    ("window-section-missing",
     "I could not find a Section 08 80 00 in the project manual."),
    ("window-water-test",
     "- Finish: complies.\n"
     "- The laboratory report leaves out the water penetration test.\n"
     "- Field water test of 3.3: after installation, not due yet."),
])
def test_looked_for_and_not_found(key, reply):
    assert _passes(key, reply)


@pytest.mark.parametrize("key, reply", [
    ("window-anchorage",
     "I found the anchorage calculations on page 4 of the package."),
    ("window-section-missing",
     "Does Section 08 80 00 resolve to a section in the manual?"),
    ("window-section-missing",
     "Section 08 80 00 can be found in the manual's contents."),
    ("window-water-test",
     "The field water test of 3.3 could not be found, which is expected "
     "before installation."),
])
def test_found_is_not_missing(key, reply):
    assert not _passes(key, reply)
