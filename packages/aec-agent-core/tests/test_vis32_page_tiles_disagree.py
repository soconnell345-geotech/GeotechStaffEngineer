"""BACKLOG VIS-32: a page reading that the tiles contradict says so.

A review run asked about one door tag by name; the whole-page answer placed
that tag where the close-up tiles read another tag of the same series, and
nothing in the result named the pair, so the agent dropped the item between
the two readings. ``_merge_found`` now names such a pair once, in
``found_note``, and says the tiles are the closer look. A read with no such
pair returns exactly what it returned before (the cases below were written
against the code before this change and pass on it unchanged).

Made-up answers on a made-up 999 x 999 pt page: one point per grid step on
the page, half a point per step on the one tile, so every box is exact. No
model, no network.
"""

from __future__ import annotations

import copy
import json

import pytest

from aec_agent_core import vision_tools

PAGE_VIEW = [0.0, 0.0, 999.0, 999.0]
TILE_VIEW = [0.0, 0.0, 499.5, 499.5]
#: The same place on the page grid and on tile r1c1's grid.
HERE_PAGE = "[400, 400, 420, 415]"
HERE_TILE = "[800, 800, 840, 830]"
HERE = [400.0, 400.0, 420.0, 415.0]

AGREE = ("'found' lists every located thing with its page box and where it "
         "was seen (the page answer and the tiles agree on what is there).")


def _read(page_answer: str, tile_answer: str) -> dict:
    return {"page": 0, "view": list(PAGE_VIEW), "analysis": page_answer,
            "tiles": [{"tile": "r1c1", "view": list(TILE_VIEW),
                       "analysis": tile_answer}]}


def _merged(page_answer: str, tile_answer: str, what: str = "page") -> dict:
    out = _read(page_answer, tile_answer)
    vision_tools._merge_found(out, what)
    return out


def _only_note(named: str) -> str:
    return (f"1 thing(s) were found ONLY in the tiles, not in the whole-page "
            f"answer: {named}. The tiles read small lettering larger — treat "
            f"these as found, and zoom on each before relying on it. 'found' "
            f"lists every located thing with its page box and where it was "
            f"seen.")


# --------------------------------------------------------------------------
# No disagreement: the result is what it was (each expected value below is
# what the code before VIS-32 returns).
# --------------------------------------------------------------------------

AS_BEFORE = {
    "the page and the tile read the same tag": (
        f"Door D-508 {HERE_PAGE}", f"Door D-508 {HERE_TILE}",
        [{"what": "Door D-508", "page_bbox": HERE,
          "seen_in": ["page", "r1c1"]}],
        AGREE),
    "a thing only the tile saw, of another series": (
        f"Door D-508 {HERE_PAGE}",
        f"Door D-508 {HERE_TILE}\nWindow W-12 [900, 100, 920, 115]",
        [{"what": "Window W-12", "page_bbox": [450.0, 50.0, 460.0, 57.5],
          "seen_in": ["r1c1"], "tiles_only": True},
         {"what": "Door D-508", "page_bbox": HERE,
          "seen_in": ["page", "r1c1"]}],
        _only_note("Window W-12 at [450.0, 50.0, 460.0, 57.5] (r1c1)")),
    "two tags of different series at one place": (
        f"Detail A-501 {HERE_PAGE}", f"Door 508 {HERE_TILE}",
        [{"what": "Door 508", "page_bbox": HERE,
          "seen_in": ["page", "r1c1"]}],
        AGREE),
    "the tile reads the page's tag elsewhere": (
        f"Door 509 {HERE_PAGE}",
        f"Door 508 {HERE_TILE}\nDoor 509 [100, 900, 140, 930]",
        [{"what": "Door 508", "page_bbox": HERE,
          "seen_in": ["page", "r1c1"]},
         {"what": "Door 509", "page_bbox": [50.0, 450.0, 70.0, 465.0],
          "seen_in": ["r1c1"], "tiles_only": True}],
        _only_note("Door 509 at [50.0, 450.0, 70.0, 465.0] (r1c1)")),
    "two tags of one series far apart": (
        "Door 509 [100, 100, 120, 115]", f"Door 508 {HERE_TILE}",
        [{"what": "Door 509", "page_bbox": [100.0, 100.0, 120.0, 115.0],
          "seen_in": ["page"]},
         {"what": "Door 508", "page_bbox": HERE, "seen_in": ["r1c1"],
          "tiles_only": True}],
        _only_note(f"Door 508 at {HERE} (r1c1)")),
    "one tag written two ways": (
        f"Door D-509 {HERE_PAGE}", f"Door 509 {HERE_TILE}",
        [{"what": "Door 509", "page_bbox": HERE,
          "seen_in": ["page", "r1c1"]}],
        AGREE),
}


@pytest.mark.parametrize("case", sorted(AS_BEFORE))
def test_without_a_disagreement_the_result_is_as_before(case):
    page, tile, found, note = AS_BEFORE[case]
    out = _merged(page, tile)
    expected = _read(page, tile)
    expected.update(found=found, found_note=note)
    assert json.dumps(out) == json.dumps(expected)


# --------------------------------------------------------------------------
# A disagreement is named once, in found_note, and nothing else moves.
# --------------------------------------------------------------------------

def _without_the_line(page, tile, monkeypatch, what="page"):
    with monkeypatch.context() as m:
        m.setattr(vision_tools, "_place_disagreements",
                  lambda *a, **k: [])
        return _merged(page, tile, what)


def test_the_tag_placed_where_the_tile_reads_another_is_named(monkeypatch):
    """The item's case: the page answer, asked about door 509 by name,
    placed it where the tile reads door 508. Plain numbers carry no code,
    so the two readings merge into one row; before, the note said the two
    readings agreed."""
    page, tile = f"the requested door tag 509 {HERE_PAGE}", \
        f"Door 508 {HERE_TILE}"
    out = _merged(page, tile)
    note = out["found_note"]
    assert "agree" not in note
    assert "DISAGREE" in note
    assert "'the requested door tag 509' at " + str(HERE) in note
    assert "tile r1c1 read 'Door 508' at " + str(HERE) in note
    assert "closer look" in note and "zoom" in note
    assert note.count("509") == 1 and note.count("508") == 1
    # only the note moved: the rows and every other key are as before
    before = _without_the_line(page, tile, monkeypatch)
    assert {k: v for k, v in out.items() if k != "found_note"} == \
        {k: v for k, v in before.items() if k != "found_note"}
    assert before["found_note"] == AGREE


def test_a_row_seen_only_on_the_page_next_to_one_only_in_the_tile(
        monkeypatch):
    """The item's "Done when": tags with letters stay two rows, one seen
    only in the page answer, one only in the tile under another code; the
    line follows the found-only-in-tiles note."""
    page, tile = f"Door D-509 {HERE_PAGE}", f"Door D-508 {HERE_TILE}"
    out = _merged(page, tile)
    assert sorted(r["seen_in"] for r in out["found"]) == [["page"], ["r1c1"]]
    before = _without_the_line(page, tile, monkeypatch)
    assert out["found"] == before["found"]
    assert before["found_note"] == _only_note(f"Door D-508 at {HERE} (r1c1)")
    assert out["found_note"].startswith(before["found_note"] + " ")
    added = out["found_note"][len(before["found_note"]) + 1:]
    assert added.startswith("The whole-page answer and the tiles DISAGREE "
                            "at 1 place: 'Door D-509' at ")
    assert "tile r1c1 read 'Door D-508'" in added


def test_an_image_read_in_tiles_says_it_too():
    out = _merged(f"Tag P-12 {HERE_PAGE}", f"Tag P-13 {HERE_TILE}", "image")
    note = out["found_note"]
    assert "whole-image answer and the tiles DISAGREE" in note
    assert "'Tag P-12' at " + str(HERE) + " in the image answer" in note
    assert all("bbox" in r for r in out["found"])


def test_the_nearest_reading_is_the_one_named():
    """Two tags of the series near the page's: the nearer is named, once."""
    tile = (f"Door 508 {HERE_TILE}\n"
            "Door 510 [800, 860, 840, 890]")        # 15 pt lower on the page
    out = _merged(f"Door 509 {HERE_PAGE}", tile)
    note = out["found_note"]
    assert "read 'Door 508'" in note and "510" not in note


def test_the_same_pair_twice_in_the_page_answer_is_named_once():
    page = f"Door 509 {HERE_PAGE}\nSummary: door 509 {HERE_PAGE}"
    out = _merged(page, f"Door 508 {HERE_TILE}")
    assert out["found_note"].count("DISAGREE at 1 place") == 1
    assert out["found_note"].count("508") == 1


def test_no_tiles_no_line():
    out = {"page": 0, "view": list(PAGE_VIEW),
           "analysis": f"Door 509 {HERE_PAGE}"}
    before = copy.deepcopy(out)
    vision_tools._merge_found(out)
    assert out == before


@pytest.mark.parametrize("label, tags", [
    ("Door tag 509", {"509"}),
    ("D-509 (45 min)", {"D-509", "45"}),
    ("callout 2/A-302", {"A-302"}),
    ("2 hr wall, 1.5 m high", set()),
    ("CORE / COPY ROOM", set()),
])
def test_tags_are_the_words_with_a_digit(label, tags):
    assert vision_tools._tags(label) == tags
