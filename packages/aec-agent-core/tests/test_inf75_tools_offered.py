"""The record shows what each offered tool said, not only its name
(BACKLOG INF-75).

Foundry brief 7 (2026-11-12): ``tools_offered_sha`` on ``turn_end`` hashes
the sorted tool names, so a run offered ``show_me`` on its looking tools
and a run that was not had the same value, and the activity record alone
could not show the argument was offered. A second field,
``tools_offered_spec_sha``, now hashes each tool's name, description and
argument schema; ``tools_offered_sha`` is unchanged for older readers.
"""

from __future__ import annotations

import uuid

from langchain_core.messages import HumanMessage, SystemMessage

from aec_agent_core import activity_log
from aec_agent_core.activity_log import ActivityLogger


def _spec(name, description, properties):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties}}}


PAGE = {"page": {"type": "integer"}}
LOOK = _spec("look_at_page", "Look at one page.", PAGE)
LOOK_SHOW = _spec("look_at_page", "Look at one page.",
                  {**PAGE, "show_me": {"type": "boolean"}})
READ = _spec("read_text", "Read a page's text.", PAGE)


def _turn_end(tmp_path, tools=None, engine=None, turn=1):
    logger = ActivityLogger(str(tmp_path), turn=turn)
    logger.turn_start(prompt="q", engine=engine)
    if tools is not None:
        logger.on_chat_model_start(
            {"name": "fake"},
            [[SystemMessage(content="the rules"), HumanMessage(content="q")]],
            run_id=uuid.uuid4(), invocation_params={"tools": tools})
    logger.turn_end()
    return [r for r in activity_log.load(str(tmp_path))
            if r.get("event") == "turn_end"][-1]


def test_an_added_argument_changes_the_spec_hash_not_the_names_hash(
        tmp_path):
    a = _turn_end(tmp_path / "a", [LOOK, READ])
    b = _turn_end(tmp_path / "b", [LOOK_SHOW, READ])
    assert a["tools_offered_sha"] == b["tools_offered_sha"]     # as before
    assert a["tools_offered_spec_sha"] != b["tools_offered_spec_sha"]
    assert a["tools_offered_spec_from"] == "calls"


def test_the_same_tools_in_another_order_hash_the_same(tmp_path):
    a = _turn_end(tmp_path / "a", [LOOK, READ])
    b = _turn_end(tmp_path / "b", [READ, LOOK])
    assert a["tools_offered_spec_sha"] == b["tools_offered_spec_sha"]


def test_a_changed_description_changes_the_spec_hash(tmp_path):
    reworded = _spec("read_text", "Read the text of a page.", PAGE)
    a = _turn_end(tmp_path / "a", [LOOK, READ])
    b = _turn_end(tmp_path / "b", [LOOK, reworded])
    assert a["tools_offered_spec_sha"] != b["tools_offered_spec_sha"]


def test_the_other_specification_shapes_are_read(tmp_path):
    """The name at the top (the Responses and Anthropic shapes)."""
    flat = {"type": "function", "name": "look_at_page",
            "description": "Look at one page.",
            "parameters": {"type": "object", "properties": PAGE}}
    claude = {"name": "look_at_page", "description": "Look at one page.",
              "input_schema": {"type": "object", "properties": PAGE}}
    a = _turn_end(tmp_path / "a", [LOOK])
    b = _turn_end(tmp_path / "b", [flat])
    c = _turn_end(tmp_path / "c", [claude])
    assert a["tools_offered_spec_sha"] == b["tools_offered_spec_sha"] \
        == c["tools_offered_spec_sha"]


def test_a_client_that_keeps_its_tools_uses_the_builds_fingerprint(tmp_path):
    """Where the model calls do not show their tools, the page's own
    fingerprint of the built agent's tools stands in, and the record says
    which it was."""
    engine = {"build": {"tools": ["look_at_page", "read_text"]},
              "tools_sha": "0123456789abcdef"}
    rec = _turn_end(tmp_path, engine=engine)
    assert rec["tools_offered_spec_sha"] == "0123456789abcdef"
    assert rec["tools_offered_spec_from"] == "build"
    assert rec["tools_offered_from"] == "build"


def test_a_turn_with_no_tools_known_has_no_spec_hash(tmp_path):
    rec = _turn_end(tmp_path)
    assert "tools_offered_spec_sha" not in rec
    assert "tools_offered_sha" not in rec
