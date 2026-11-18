"""BACKLOG TA-43: a count and a ceiling on a turn's image calls.

A page look is up to 17 image calls, so a 50-sheet set read sheet by sheet
is about 850, and the model-call budget counts none of them. Now every
image sent to the vision model for a separate reading is counted on the
turn (images shown inline to the agent's own model are not) and the count
is on the turn's record. With ``GEOTECH_TURN_MAX_IMAGE_CALLS`` set, a look past
the ceiling is refused (the image is not sent) with a tool error that says
so, and each look's result says its count against the ceiling. Unset (the
default), nothing is refused and every result is what it was.

A fake vision model and made-up pages built here: no model, no network.
"""

from __future__ import annotations

import json
import threading
import uuid
from typing import Any, List, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from aec_agent_core import activity_log, vision_tools
from aec_agent_core.activity_log import ActivityLogger
from aec_agent_core.deep import keep_work as kw
from aec_agent_core.deep import limits
from aec_agent_core.deep.vision_engine import (LangChainVisionEngine,
                                               describe_error)

ENV = limits.TURN_MAX_IMAGE_CALLS_ENV
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class _Eyes(BaseChatModel):
    """A fake vision model: answers every look, and counts them."""

    state: Any = None

    @property
    def _llm_type(self) -> str:
        return "eyes-fake"

    def _generate(self, messages, stop: Optional[List[str]] = None,
                  run_manager=None, **kwargs) -> ChatResult:
        with self.state["lock"]:
            self.state["calls"] += 1
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="Door 508 [100, 100, 120, 115]"))])


def _engine():
    model = _Eyes(state={"calls": 0, "lock": threading.Lock()})
    return LangChainVisionEngine(model, detail=""), model.state


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    vision_tools.clear_repeat_reads()
    yield
    vision_tools.clear_repeat_reads()


def _sheet() -> bytes:
    pymupdf = pytest.importorskip("pymupdf")
    d = pymupdf.open()
    page = d.new_page(width=612, height=396)
    page.insert_text((72, 100), "DOOR 508", fontsize=10)
    return d.tobytes()


def _page_tool(engine, pdf):
    from aec_agent_core.deep.vision_kit import make_vision_tools
    tools = make_vision_tools(engine=engine, attachments={"s": pdf})
    return next(t for t in tools if t.name == "analyze_pdf_page")


def _read(tool, prompt="List the door tags.", tiles="off") -> str:
    return tool.invoke({"attachment_key": "s", "page": 0, "prompt": prompt,
                        "tiles": tiles})


# --------------------------------------------------------------------------
# Unset: nothing moves
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value, limit", [
    (None, 0), ("", 0), ("0", 0), ("off", 0), ("many", 0), ("-3", 0),
    ("40", 40)])
def test_the_ceiling_is_off_unless_set(monkeypatch, value, limit):
    if value is not None:
        monkeypatch.setenv(ENV, value)
    assert limits.turn_max_image_calls() == limit
    assert kw.TurnScope().image_limit == limit


def test_unset_every_look_is_made_and_counted():
    engine, state = _engine()
    with kw.turn_scope(kw.TurnScope()) as scope:
        answers = [engine.analyze_image(PNG, "Read it.") for _ in range(20)]
    assert all(a.startswith("Door 508") for a in answers)
    assert state["calls"] == 20
    assert (scope.image_calls, scope.image_refused, scope.image_limit) == \
        (20, 0, 0)


def test_unset_a_page_read_returns_what_it_returned():
    """The tool result under a turn with no ceiling is the result with no
    turn at all, character for character: no key, no line."""
    pdf = _sheet()
    engine, _state = _engine()
    tool = _page_tool(engine, pdf)
    outside = _read(tool)
    vision_tools.clear_repeat_reads()
    with kw.turn_scope(kw.TurnScope()) as scope:
        inside = _read(tool)
    assert inside == outside
    assert "image_calls" not in inside and "ceiling" not in inside
    assert scope.image_calls == 1
    raw = json.dumps({"analysis": "made up"})
    with kw.turn_scope(kw.TurnScope()):
        assert kw.with_image_calls(raw, {"calls": 3, "refused": 0}) is raw
        assert kw.image_calls_status([0]) is None


def test_outside_a_turn_nothing_is_counted_or_refused(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    engine, state = _engine()
    for _ in range(3):
        engine.analyze_image(PNG, "Read it.")
    assert state["calls"] == 3


# --------------------------------------------------------------------------
# Set: the ceiling
# --------------------------------------------------------------------------

def test_past_the_ceiling_a_look_is_refused_and_not_sent(monkeypatch):
    monkeypatch.setenv(ENV, "2")
    engine, state = _engine()
    with kw.turn_scope(kw.TurnScope()) as scope:
        engine.analyze_image(PNG, "Read it.")
        engine.analyze_image(PNG, "Read it.")
        with pytest.raises(kw.ImageBudgetSpent) as caught:
            engine.analyze_image(PNG, "Read it.")
    assert state["calls"] == 2
    assert (scope.image_calls, scope.image_refused) == (2, 1)
    said = describe_error(caught.value)
    assert said == kw.IMAGE_CEILING_ERROR.format(limit=2)
    assert "ceiling of 2 image calls" in said
    assert "NOT read" in said and "Finish with what you have" in said


def test_a_refused_page_read_is_a_plain_tool_error(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    pdf = _sheet()
    engine, state = _engine()
    tool = _page_tool(engine, pdf)
    with kw.turn_scope(kw.TurnScope()):
        first = json.loads(_read(tool))
        second = json.loads(_read(tool, prompt="List the windows."))
    assert list(first)[0] == "image_calls"
    assert first["image_calls"].startswith(
        "this look made 1 image call; this turn has made 1 of its ceiling "
        "of 1. The ceiling is met: any further look this turn is refused, "
        "whatever a note in this result suggests")
    assert first["analysis"].startswith("Door 508")
    assert second == {"error": kw.IMAGE_CEILING_ERROR.format(limit=1)}
    assert state["calls"] == 1


def test_a_look_cut_by_the_ceiling_says_what_was_not_read(monkeypatch):
    """A page in 2 x 2 tiles is five image calls; with three left, three
    are made, two are refused, and the result says so."""
    monkeypatch.setenv(ENV, "3")
    pdf = _sheet()
    engine, state = _engine()
    tool = _page_tool(engine, pdf)
    with kw.turn_scope(kw.TurnScope()) as scope:
        raw = _read(tool, tiles="2")
    out = json.loads(raw)
    assert (scope.image_calls, scope.image_refused) == (3, 2)
    assert state["calls"] == 3
    assert "error" not in out
    assert "2 more were refused at the ceiling" in out["image_calls"]
    rest = json.dumps({k: v for k, v in out.items() if k != "image_calls"})
    assert "ceiling of 3 image calls" in rest      # the parts not read say why
    # at least one tile was refused; its note invites no look (review fix 3)
    assert out["tiles_not_read"].endswith(
        "NOT read: this turn's image ceiling was met")
    assert "render_region" not in out["tiles_not_read"]


CAPPED = "this turn's image ceiling was met"


def test_at_the_ceiling_a_look_says_no_further_look_will_be_made(
        monkeypatch):
    """Release review: a result's own notes may still suggest a zoom or
    tiles; once the ceiling is met the count's line says any further look
    is refused, in place of the "near" warning."""
    monkeypatch.setenv(ENV, "10")
    raw = json.dumps({"analysis": "made up",
                      "found_note": "zoom on each before relying on it"})
    with kw.turn_scope(kw.TurnScope()) as scope:
        assert kw.image_ceiling_met() is False
        scope.image_calls = 10
        met = json.loads(kw.with_image_calls(raw, {"calls": 1}))
        assert kw.image_ceiling_met() is True
    assert met["image_calls"] == (
        "this look made 1 image call; this turn has made 10 of its ceiling "
        "of 10" + kw.IMAGE_MET_NOTE)
    assert "ceiling is near" not in met["image_calls"]
    assert kw.image_ceiling_met() is False          # outside a turn


def test_no_ceiling_is_never_met():
    with kw.turn_scope(kw.TurnScope()) as scope:
        scope.image_calls = 500
        assert kw.image_ceiling_met() is False


PAGE_VIEW = [0.0, 0.0, 999.0, 999.0]
TILE_VIEW = [0.0, 0.0, 499.5, 499.5]
FOUND_HEAD = ("'found' lists every located thing with its page box and "
              "where it was seen (")


def _merged(tiles):
    out = {"page": 0, "view": list(PAGE_VIEW),
           "analysis": "Door D-508 [400, 400, 420, 415]", "tiles": tiles}
    vision_tools._merge_found(out)
    return out["found_note"]


def test_tiles_that_were_not_read_are_not_said_to_agree():
    """Release review: when the whole-page look takes the turn's last image
    call every tile is refused, and the merged list is the page answer's
    alone. Its note must not say the tiles agree. The same holds for a
    tile that failed for any other reason."""
    ceiling = kw.IMAGE_CEILING_ERROR.format(limit=1)
    read = {"tile": "r1c1", "view": list(TILE_VIEW),
            "analysis": "Door D-508 [800, 800, 840, 830]"}
    none_read = _merged([{"tile": "r1c1", "error": ceiling},
                         {"tile": "r1c2", "error": ceiling}])
    assert none_read == (
        FOUND_HEAD + "from the page answer alone: no tile was read, so "
        "nothing here was checked against a closer look).")
    assert "agree" not in none_read
    one_failed = _merged([read, {"tile": "r1c2",
                                 "error": "the vision model timed out"}])
    assert one_failed == (
        FOUND_HEAD + "the page answer and the tiles that were read agree "
        "on what is there; 1 tile was not read).")
    two_refused = _merged([read, {"tile": "r1c2", "error": ceiling},
                           {"tile": "r2c1", "error": ceiling}])
    assert two_refused.endswith("; 2 tiles were not read).")
    # every tile read: the text it has always had, byte for byte
    assert _merged([read]) == (
        FOUND_HEAD + "the page answer and the tiles agree on what is "
        "there).")


def test_tiles_not_read_at_the_ceiling_invite_no_look():
    """Review fix 3: a tile refused at the ceiling is named with the reason
    alone; reading again or zooming would be refused too."""
    ceiling = kw.IMAGE_CEILING_ERROR.format(limit=3)
    read = [{"tile": "r1c1", "analysis": "Door 508"},
            {"tile": "r1c2", "analysis": "Door 509"}]
    note = vision_tools._tiles_not_read_note
    notes = [
        note(read + [{"tile": "r2c1", "error": ceiling}]),
        note(read + [{"tile": "r2c1", "error": ceiling},
                     {"tile": "r2c2", "error": ceiling}]),
        note([{"tile": "r1c1", "error": ceiling},
              {"tile": "r1c2", "error": ceiling}]),
        note(read + [{"tile": "r2c1", "error": "the vision model is busy"},
                     {"tile": "r2c2", "error": ceiling}]),
    ]
    assert notes == [
        f"tile r2c1 was NOT read: {CAPPED}",
        f"tiles r2c1, r2c2 were NOT read: {CAPPED}",
        f"tiles r1c1, r1c2 were NOT read: {CAPPED}",
        f"tile r2c2 was NOT read: {CAPPED}; tile r2c1 failed: the vision "
        f"model is busy",
    ]
    assert not any("render_region" in n or "again" in n for n in notes)


def test_tiles_not_read_for_another_reason_say_what_they_said():
    """Every other reason keeps the text it had before TA-43, byte for
    byte (the expected strings are that code's)."""
    why = "the vision model timed out"
    note = vision_tools._tiles_not_read_note
    assert note([{"tile": "r1c1", "analysis": "Door 508"},
                 {"tile": "r2c2", "error": why}]) == (
        "tile r2c2 failed: the vision model timed out; the rest were read. "
        "That part of the page was NOT read in close-up: read it with "
        "render_region(bbox=<the failed tile's view>) before relying on it")
    assert note([{"tile": "r1c1", "error": why},
                 {"tile": "r1c2", "error": why}]) == (
        "NO tile was read (the vision model timed out); only the whole-page "
        "view above was, and it was asked for the layout only: read the "
        "page again, or zoom with render_region, before relying on its "
        "lettering")
    assert note([{"tile": "r1c1", "analysis": "Door 508"}]) is None


def test_unset_a_tiled_read_returns_what_it_returned():
    """With the setting unset a page read in tiles is the same under a turn
    as outside one: nothing refused, no note."""
    pdf = _sheet()
    engine, _state = _engine()
    tool = _page_tool(engine, pdf)
    outside = _read(tool, tiles="2")
    vision_tools.clear_repeat_reads()
    with kw.turn_scope(kw.TurnScope()) as scope:
        inside = _read(tool, tiles="2")
    assert inside == outside
    assert "tiles_not_read" not in inside and "ceiling" not in inside
    assert scope.image_calls == 5


def test_from_80_percent_each_look_warns(monkeypatch):
    monkeypatch.setenv(ENV, "10")
    raw = json.dumps({"analysis": "made up"})
    with kw.turn_scope(kw.TurnScope()) as scope:
        scope.image_calls = 7
        quiet = json.loads(kw.with_image_calls(raw, {"calls": 2}))
        scope.image_calls = 8
        near = json.loads(kw.with_image_calls(raw, {"calls": 1}))
        text = kw.with_image_calls("plain text", {"calls": 1})
        none = kw.with_image_calls(raw, {"calls": 0, "refused": 1})
    assert quiet["image_calls"] == ("this look made 2 image calls; this "
                                    "turn has made 7 of its ceiling of 10")
    assert "ceiling is near" in near["image_calls"]
    assert "ceiling is met" not in near["image_calls"]
    assert near["analysis"] == "made up"
    assert text.startswith("[image calls] this look made 1 image call;")
    assert text.endswith("\n\nplain text")
    assert none is raw


def test_the_status_line_moves_with_the_count(monkeypatch):
    monkeypatch.setenv(ENV, "5")
    shown = [0]
    with kw.turn_scope(kw.TurnScope()) as scope:
        assert kw.image_calls_status(shown) is None
        scope.image_calls = 2
        assert kw.image_calls_status(shown) == "image calls this turn: 2 of 5"
        assert kw.image_calls_status(shown) is None
        scope.image_calls = 5
        assert kw.image_calls_status(shown) == "image calls this turn: 5 of 5"


def test_the_tester_is_told_only_when_a_look_was_refused(monkeypatch):
    monkeypatch.setenv(ENV, "4")
    scope = kw.TurnScope()
    scope.image_calls = 4
    assert kw.image_ceiling_told(scope) is None
    scope.image_refused = 1
    assert kw.image_ceiling_told(scope) == (
        "_This turn reached its ceiling of 4 image calls: 1 image call after "
        "that was not made, so the answer rests on what was read before. "
        "Ask to continue for more._")
    scope.image_refused = 3
    assert "3 image calls after that were not made" in \
        kw.image_ceiling_told(scope)
    monkeypatch.delenv(ENV)
    assert kw.image_ceiling_told(kw.TurnScope()) is None
    assert kw.image_ceiling_told(None) is None


# --------------------------------------------------------------------------
# The record
# --------------------------------------------------------------------------

def _turn_end(tmp_path, scope, looks: int):
    engine, _state = _engine()
    log = ActivityLogger(str(tmp_path), turn=1)
    with kw.turn_scope(scope):
        log.on_chat_model_start({}, [[]], run_id=uuid.uuid4())
        for _ in range(looks):
            try:
                engine.analyze_image(PNG, "Read it.")
            except kw.ImageBudgetSpent:
                pass
    log.turn_end(turn_tokens=0)          # the scope has closed, as on a page
    (end,) = [r for r in activity_log.load(str(tmp_path))
              if r["event"] == "turn_end"]
    return end


def test_the_count_is_on_the_turn_record_without_a_ceiling(tmp_path):
    end = _turn_end(tmp_path, kw.TurnScope(), 4)
    assert end["image_calls"] == 4
    assert "image_limit" not in end and "image_calls_refused" not in end


def test_the_record_names_the_ceiling_and_the_refusals(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, "3")
    end = _turn_end(tmp_path, kw.TurnScope(), 5)
    assert (end["image_calls"], end["image_limit"],
            end["image_calls_refused"]) == (3, 3, 2)


def test_a_record_with_no_turn_scope_has_no_count(tmp_path):
    log = ActivityLogger(str(tmp_path), turn=1)
    log.on_chat_model_start({}, [[]], run_id=uuid.uuid4())
    log.turn_end(turn_tokens=0)
    (end,) = [r for r in activity_log.load(str(tmp_path))
              if r["event"] == "turn_end"]
    assert "image_calls" not in end
