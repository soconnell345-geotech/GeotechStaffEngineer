"""A references or reviewer consult's result says how its lookups went,
counted by code (2026-11-10 reference-library review, item 5).

The helper's own "not found" had covered lookups that errored, came back
empty or were never made. The block is built from the helper's record
(``HelperRecord``), never from its prose. Made-up lookups throughout.
"""

import dataclasses
import json

import pytest

pytest.importorskip("deepagents")

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from funhouse_agent.deep import keep_work  # noqa: E402
from funhouse_agent.deep.lookup_report import (  # noqa: E402
    BLOCK_HEAD, LOOKUPS_NOTE, MAX_LISTED, LookupReport, lookup_block,
    lookup_outcomes)


def _record(results, subagent="references", budget=False):
    """A helper record with one model step calling each lookup, then each
    result: ``results`` is [(tool, args, text, status)]."""
    rec = keep_work.HelperRecord("t1", subagent, "a made-up question")
    calls = [{"name": tool, "id": f"c{i}", "args": args}
             for i, (tool, args, _t, _s) in enumerate(results)]
    rec.add_model(AIMessage(content="", tool_calls=calls))
    for i, (tool, _a, text, status) in enumerate(results):
        rec.add_result(tool, f"c{i}", text, status)
    rec.budget_reached = budget
    return rec


def _lookup(agent, method, **params):
    return {"agent_name": agent, "method": method, "parameters": params}


OK = json.dumps({"method": "x", "result": 1.23})
EMPTY_SEARCH = json.dumps({"result": [], "shown": 0,
                           "defaults_used": {"limit": 5}})
ERROR = json.dumps({"error": "KeyError: \"Section '99.9' not found in "
                             "reference 'made_up'\""})
CUT = OK[:-1] + ', "more": "' + "x" * 50 + '"\n...[truncated 4000 chars].'


def test_a_clean_consult_says_so_in_one_line():
    rec = _record([("call_agent", _lookup("gec12", "made_up_table", phi=30),
                    OK, None)])
    block = lookup_block(rec)
    assert block.startswith(f"[{BLOCK_HEAD}: 1 made; none errored")
    assert "\n" not in block
    assert "do not fill" not in block


def test_errored_empty_and_cut_are_counted_and_named():
    rec = _record([
        ("call_agent", _lookup("gec12", "made_up_section", section_id="99.9"),
         ERROR, None),
        ("call_agent", _lookup("reference_db", "reference_search",
                               query="made up words"), EMPTY_SEARCH, None),
        ("call_agent", _lookup("gec12", "made_up_text"), CUT, None),
        ("describe_method", {"agent_name": "gec12"}, ERROR, None),
    ])
    block = lookup_block(rec)
    head = block.splitlines()[0]
    # describe_method is not a lookup
    assert "3 made; 1 errored, 1 came back empty, 1 cut at the size limit" \
        in head
    assert ("- gec12.made_up_section errored: Section '99.9' not found in "
            "reference 'made_up'") in block
    assert '- reference_db.reference_search "made up words" came back empty' \
        in block
    assert "made_up_text" not in block          # cut is counted, not listed
    assert "do not fill them from memory" in head


def test_a_tool_error_status_and_a_listed_error_count_as_errors():
    rec = _record([
        ("read_reference_figure", {"reference": "made_up",
                                   "figure_number": "9-9"},
         "vision not available", "error"),
        ("call_agent", _lookup("made_up", "m"),
         json.dumps({"result": [{"error": "no such figure"}]}), None),
    ])
    outcomes = lookup_outcomes(rec)
    assert [o["outcome"] for o in outcomes] == ["errored", "errored"]
    assert outcomes[0]["name"] == "read_reference_figure made_up figure 9-9"
    assert outcomes[1]["reason"] == "no such figure"


def test_no_lookup_made_is_said():
    rec = _record([])
    block = lookup_block(rec)
    assert block.startswith(f"[{BLOCK_HEAD}: no lookup was made.")
    assert "do not fill them from memory" in block


def test_ending_on_the_budget_is_said():
    rec = _record([("call_agent", _lookup("gec12", "m"), OK, None)],
                  budget=True)
    block = lookup_block(rec)
    assert "ended on its call budget" in block
    assert "do not fill them from memory" in block


def test_a_long_list_is_cut_to_its_first_lookups():
    rec = _record([("call_agent", _lookup("made_up", f"m{i}"), ERROR, None)
                   for i in range(MAX_LISTED + 3)])
    block = lookup_block(rec)
    assert block.count("\n- made_up.") == MAX_LISTED
    assert "- and 3 more]" in block


class _Request:
    def __init__(self, name, subagent_type=None):
        args = {"description": "x"}
        if subagent_type:
            args["subagent_type"] = subagent_type
        self.tool_call = {"name": name, "id": "t1", "args": args}


def _run(request, rec, result):
    """LookupReport around a handler run while ``rec`` is the running
    helper, as KeepWork sets it."""
    token = keep_work._HELPER.set(rec)
    try:
        return LookupReport().wrap_tool_call(request, lambda r: result)
    finally:
        keep_work._HELPER.reset(token)


@pytest.mark.parametrize("helper", ["references", "reviewer"])
def test_the_consult_result_gets_the_block(helper):
    rec = _record([("call_agent", _lookup("gec12", "m"), ERROR, None)],
                  subagent=helper)
    out = _run(_Request("task", helper), rec,
               ToolMessage(content="Not found.", tool_call_id="t1"))
    assert out.content.startswith("Not found.\n\n[" + BLOCK_HEAD)
    assert "gec12.m errored" in out.content


def test_a_command_result_gets_the_block():
    @dataclasses.dataclass
    class Command:
        update: dict

    rec = _record([("call_agent", _lookup("gec12", "m"), OK, None)])
    msg = ToolMessage(content="Value 1.23.", tool_call_id="t1")
    out = _run(_Request("task", "references"), rec,
               Command(update={"messages": [msg]}))
    assert BLOCK_HEAD in out.update["messages"][0].content


def test_other_helpers_and_tools_are_left_alone():
    msg = ToolMessage(content="done", tool_call_id="t1")
    calc = _record([("call_agent", _lookup("bearing_capacity", "m"), OK,
                     None)], subagent="calc")
    assert _run(_Request("task", "calc"), calc, msg) is msg
    rec = _record([])
    assert _run(_Request("call_agent"), rec, msg) is msg


def test_the_geotech_page_has_it_inside_keepwork():
    from webapp.tests import page_builds
    call = page_builds.fingerprint("geotech_default")["app"]["calls"][0]
    classes = [m["class"] for m in call["middleware"]]
    assert classes.index("LookupReport") > classes.index("KeepWork")
    assert BLOCK_HEAD in LOOKUPS_NOTE
