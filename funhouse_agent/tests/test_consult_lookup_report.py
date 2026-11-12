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
    # the exception's name is kept, its quoted message unquoted
    assert ("- gec12.made_up_section errored: KeyError: Section '99.9' not "
            "found in reference 'made_up'") in block
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


def _any_word(section, lacks):
    return {"reference": "made_up", "section_id": section,
            "match": "any word", "lacks": lacks}


def test_a_search_with_only_any_word_hits_is_no_match():
    """The library fills a search no section answers with hits holding
    some of the words, marked "any word" (2026-11-10 review, core-05):
    such a search found nothing for the query."""
    partial = json.dumps({"result": [_any_word("1.1", ["zzz"]),
                                     _any_word("2.3", ["zzz", "yyy"])],
                          "shown": 2, "matched": "any word",
                          "words_not_found": ["zzz"]})
    mixed = json.dumps({"result": [{"reference": "made_up",
                                    "section_id": "4.1"},
                                   _any_word("2.3", ["yyy"])],
                        "shown": 2, "matched": "any word"})
    figure_hits = json.dumps({"result": [
        dict(_any_word("9-9", ["zzz"]), figure_number="9-9",
             read_value="To read a numeric value ...")], "shown": 1})
    rec = _record([
        ("call_agent", _lookup("reference_db", "reference_search",
                               query="made up yyy"), mixed, None),
        ("call_agent", _lookup("reference_db", "reference_search",
                               query="made up zzz"), partial, None),
        ("call_agent", _lookup("figure_db", "figure_search",
                               query="made up zzz"), figure_hits, None),
    ])
    outcomes = lookup_outcomes(rec)
    assert [o["outcome"] for o in outcomes] == ["ok", "empty", "empty"]
    block = lookup_block(rec)
    assert "3 made; 0 errored, 2 came back empty" in block
    assert ('- reference_db.reference_search "made up zzz" came back empty: '
            "no hit holds every word") in block
    assert "holds all the words asked for" in LOOKUPS_NOTE


def test_a_refused_scope_is_an_error_with_its_message():
    """A search scope the library refuses ("dm7", an unknown name) comes
    back as an error naming the valid ids, never as "nothing found"."""
    refused = json.dumps({"result": [{"error": (
        "'dm7' names two references here: dm7_1 (made up), dm7_2 (made "
        "up); give one")}]})
    rec = _record([("call_agent", _lookup("figure_db", "figure_search",
                                          query="made up", reference="dm7"),
                    refused, None)])
    [o] = lookup_outcomes(rec)
    assert o["outcome"] == "errored"
    assert "dm7_1" in o["reason"]


def test_no_lookup_made_is_said():
    rec = _record([])
    block = lookup_block(rec)
    assert block.startswith(f"[{BLOCK_HEAD}: no lookup was made.")
    # nothing failed: the answer simply comes from no reference
    assert "Nothing in the answer comes from the references" in block
    assert "which lookups failed" not in block


def _chart_read(figure, analysis):
    return json.dumps({"reference": "made_up", "figure_number": figure,
                       "caption": "A made-up chart", "analysis": analysis})


def test_a_chart_read_off_the_wrong_page_came_back_empty():
    """The reader said the figure is not on the page: no value was read,
    so the read is no answer (review follow-up 1)."""
    rec = _record([
        ("read_reference_figure", {"reference": "made_up",
                                   "figure_number": "7-1"},
         _chart_read("7-1", "Figure 7-1 is not on this page; the page shows "
                            "a table of made-up values."), None),
        ("read_reference_figure", {"reference": "made_up",
                                   "figure_number": "7-2"},
         _chart_read("7-2", "NOT ON THIS PAGE: the page is text."), None),
        ("read_reference_figure", {"reference": "made_up",
                                   "figure_number": "7-3"},
         _chart_read("7-3", "Following the made-up curve, the value is 1.2. "
                            "The legend is not on this page."), None),
    ])
    outcomes = lookup_outcomes(rec)
    assert [o["outcome"] for o in outcomes] == ["empty", "empty", "ok"]
    block = lookup_block(rec)
    assert "3 made; 0 errored, 2 came back empty" in block
    assert ("- read_reference_figure made_up figure 7-1 came back empty: "
            "the reader said the figure is not on the page") in block


def test_said_not_on_page_matches_the_whole_figure_number():
    """Asked for 7-1, Figure 7-15 is another figure; asked for 7-15, so
    is 7-150 (review follow-up 5)."""
    from funhouse_agent.reference_vision import said_not_on_page
    assert not said_not_on_page("Figure 7-15 is not on this page.", "7-1")
    assert not said_not_on_page("Figure 7-150 is not on this page.", "7-15")
    assert not said_not_on_page("Figure 17-1 is not on this page.", "7-1")
    assert not said_not_on_page("Figure 7-1.5 is not on this page.", "7-1")
    assert said_not_on_page("Figure 7-1 is not on this page.", "7-1")
    assert said_not_on_page("Figure 7-15's curves are not on this page.",
                            "7-15")
    assert said_not_on_page("FIGURE D-1 does not appear on this page.", "d-1")
    assert said_not_on_page("Figure 2.5 is not on this page.", "2.5")


def test_a_search_for_another_topic_does_not_answer_an_empty_one():
    """Recovery is keyed on the query for a search: an empty search stays
    listed with its query when a search for a different topic finds hits;
    the same query asked again (case and spacing aside) answers it
    (review follow-up 2)."""
    hit = json.dumps({"result": [{"reference": "made_up",
                                  "section_id": "1.1"}], "shown": 1})
    rec = _record([
        ("call_agent", _lookup("reference_db", "reference_search",
                               query="made up topic one"), EMPTY_SEARCH, None),
        ("call_agent", _lookup("reference_db", "reference_search",
                               query="made up topic two"), hit, None),
        ("call_agent", _lookup("figure_db", "figure_search",
                               query="Made Up  chart"), ERROR, None),
        ("call_agent", _lookup("figure_db", "figure_search",
                               query="made up chart"), hit, None),
    ])
    rows = lookup_outcomes(rec)
    assert [r["recovered"] for r in rows] == [False, False, True, False]
    block = lookup_block(rec)
    assert "1 came back empty" in block
    assert ('- reference_db.reference_search "made up topic one" came back '
            "empty") in block
    assert ('- failed, then answered by a later call: figure_db.figure_search '
            '"made up chart"]') in block


def test_ending_on_the_budget_is_said():
    rec = _record([("call_agent", _lookup("gec12", "m"), OK, None)],
                  budget=True)
    block = lookup_block(rec)
    assert "ended on its call budget, so its answer may be incomplete" \
        in block
    assert "do not fill" not in block       # nothing failed


def test_a_retried_lookup_that_answered_is_not_a_failure():
    """The review's fake-model run: an argument slip, then the same method
    answers; the answer used the second value."""
    rec = _record([
        ("call_agent", _lookup("gec12", "made_up_chart", phi=80),
         json.dumps({"error": "ValueError: phi=80 is outside the chart"}),
         None),
        ("call_agent", _lookup("gec12", "made_up_chart", phi=34), OK, None),
    ])
    rows = lookup_outcomes(rec)
    assert rows[0]["outcome"] == "errored" and rows[0]["recovered"]
    block = lookup_block(rec)
    head = block.splitlines()[0]
    assert ("2 made; each one that failed was answered by a later call for "
            "the same lookup.") in head
    assert "do not fill" not in block
    assert block.endswith("- failed, then answered by a later call: "
                          "gec12.made_up_chart]")


def test_an_error_never_answered_is_still_a_failure():
    rec = _record([
        ("call_agent", _lookup("gec12", "made_up_chart", phi=34), OK, None),
        ("call_agent", _lookup("gec12", "made_up_chart", phi=80),
         json.dumps({"error": "ValueError: phi=80 is outside the chart"}),
         None),
        ("call_agent", _lookup("gec12", "other_made_up"), ERROR, None),
    ])
    rows = lookup_outcomes(rec)
    # an answer BEFORE the error does not recover it
    assert [r["recovered"] for r in rows] == [False, False, False]
    block = lookup_block(rec)
    assert "3 made; 2 errored" in block
    assert "Tell the user which lookups failed" in block
    assert "answered by a later call" not in block


def test_a_failure_recovered_beside_one_that_is_not():
    rec = _record([
        ("call_agent", _lookup("gec12", "a_made_up"), ERROR, None),
        ("call_agent", _lookup("gec12", "a_made_up", phi=30), OK, None),
        ("call_agent", _lookup("gec12", "b_made_up"), ERROR, None),
    ], budget=True)
    block = lookup_block(rec)
    lines = block.splitlines()
    assert "3 made; 1 errored, 0 came back empty" in lines[0]
    assert "may be incomplete" in lines[0]
    assert "Tell the user which lookups failed" in lines[0]
    assert lines[1].startswith("- gec12.b_made_up errored: KeyError")
    assert lines[2] == "- failed, then answered by a later call: gec12.a_made_up]"


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


def test_a_block_that_fails_is_logged_and_the_answer_kept(monkeypatch,
                                                          caplog):
    from funhouse_agent.deep import lookup_report

    def boom(rec):
        raise RuntimeError("made-up failure")

    monkeypatch.setattr(lookup_report, "lookup_block", boom)
    msg = ToolMessage(content="Value 1.23.", tool_call_id="t1")
    with caplog.at_level("WARNING", logger=lookup_report.__name__):
        out = _run(_Request("task", "references"), _record([]), msg)
    assert out is msg
    assert "lookups block was not added" in caplog.text
    assert "made-up failure" in caplog.text


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
