"""The suite's results table counts what the record holds (BACKLOG INF-75).

Foundry brief 7 (2026-11-12): RESULTS read "errors 0" and "failed calls 0"
for a run in which three tool calls were refused, and for runs in which
vision side calls were given up at their time limit. The table counted
only ``model_error`` events and failed runs. It now also counts, per run
and per arm, the side calls given up at the limit and the tool calls that
were refused, read off ``activity.jsonl``.
"""

from __future__ import annotations

import json
import uuid

from aec_agent_core import activity_log
from aec_agent_core.activity_log import ActivityLogger
from aec_agent_core.deep.strict_args import unknown_arguments_error
from aec_agent_core.review_eval import runner
from aec_agent_core.review_eval.runner import record_counts, summarize
from aec_document_review.review_eval.tasks import Task


def _tool(logger, name, args, call_id, *, result=None, error=None,
          delivered=None, parent=None):
    """One tool call through the logger's own handlers, as a run makes it."""
    rid = uuid.uuid4()
    logger.on_tool_start({"name": name}, json.dumps(args), run_id=rid,
                         parent_run_id=parent, inputs=args,
                         tool_call_id=call_id)
    if error is not None:
        logger.on_tool_error(error, run_id=rid, parent_run_id=parent)
    else:
        logger.on_tool_end(result, run_id=rid, parent_run_id=parent)
    if delivered is not None:
        logger.on_custom_event("tool_delivered", {
            "tool_call_id": call_id, "name": name, "content": delivered,
            "status": "error" if error is not None else "success",
            "refused": False}, run_id=parent or rid)
    return rid


def _record(tmp_path):
    return activity_log.load(str(tmp_path))


def test_a_tool_that_raised_is_one_refused_call_not_two(tmp_path):
    """A raising tool leaves a ``tool_error`` and, for the error message the
    model is given, a ``tool_refused``: one call."""
    logger = ActivityLogger(str(tmp_path), turn=1)
    _tool(logger, "read_document", {"pages": "1-3"}, "call-1",
          error=ValueError("bad page range"),
          delivered='{"error": "read_document failed"}')
    _tool(logger, "read_document", {"pages": "1"}, "call-2",
          result="page one", delivered="page one")
    recs = _record(tmp_path)
    assert [r["event"] for r in recs if r["event"].startswith("tool_")] == [
        "tool_start", "tool_error", "tool_refused", "tool_start", "tool_end"]
    error = next(r for r in recs if r["event"] == "tool_error")
    refused = next(r for r in recs if r["event"] == "tool_refused")
    assert error["tool_call_id"] == "call-1"
    assert refused["tool_call_id"] == "call-1" and refused["raised"] is True
    counts = record_counts(recs)
    assert counts["tool_calls_refused"] == 1
    assert counts["tool_calls_refused_by"] == {
        "raised": 1, "no_tool_ran": 0, "unknown_arguments": 0}


def test_a_call_no_tool_ran_for_is_counted(tmp_path):
    """An unknown tool name: nothing ran, the model got a refusal."""
    logger = ActivityLogger(str(tmp_path), turn=1)
    logger.on_custom_event("tool_delivered", {
        "tool_call_id": "call-9", "name": "open_drawing",
        "content": "open_drawing is not a valid tool", "status": "error",
        "refused": True}, run_id=uuid.uuid4())
    recs = _record(tmp_path)
    refused = next(r for r in recs if r["event"] == "tool_refused")
    assert "raised" not in refused
    assert record_counts(recs)["tool_calls_refused_by"] == {
        "raised": 0, "no_tool_ran": 1, "unknown_arguments": 0}


def test_a_call_refused_for_an_unknown_argument_is_counted(tmp_path):
    """Brief 7's three: the tool returned its own refusal, a ``tool_end``
    like any other, and the table read 0."""
    logger = ActivityLogger(str(tmp_path), turn=1)
    refusal = unknown_arguments_error("analyze_pdf_page", ["pdf_pages"],
                                      ["document", "page", "pdf_page"])
    for n in range(3):
        _tool(logger, "analyze_pdf_page", {"pdf_pages": "4-13"},
              f"call-{n}", result=refusal, delivered=refusal)
    _tool(logger, "analyze_pdf_page", {"pdf_page": 4}, "call-3",
          result='{"read": "sheet A-101"}', delivered='{"read": "sheet A-101"}')
    counts = record_counts(_record(tmp_path))
    assert counts["tool_calls_refused"] == 3
    assert counts["tool_calls_refused_by"]["unknown_arguments"] == 3


def test_an_older_record_pairs_the_error_and_the_refusal_by_name():
    """Before the refusal carried ``raised``, the pair is matched by tool
    name and agent, so an old record is not counted twice."""
    recs = [
        {"event": "tool_error", "agent": "primary", "name": "measure",
         "run_id": "r1", "error": "ValueError: x"},
        {"event": "tool_refused", "agent": "primary", "name": "measure",
         "tool_call_id": "c1", "text": "measure failed"},
        {"event": "tool_refused", "agent": "primary", "name": "open_plan",
         "tool_call_id": "c2", "text": "not a valid tool"},
    ]
    assert record_counts(recs)["tool_calls_refused_by"] == {
        "raised": 1, "no_tool_ran": 1, "unknown_arguments": 0}


def test_side_call_timeouts_and_unended_calls_are_counted():
    recs = [
        {"event": "model_start", "run_id": "m1"},
        {"event": "side_call_timeout", "ask_id": "a1", "asked_again": True,
         "limit_s": 90.0, "kind": "tile", "page": 4},
        {"event": "model_start", "run_id": "m2"},
        {"event": "side_call_timeout", "ask_id": "a2", "asked_again": False,
         "limit_s": 90.0, "kind": "tile", "page": 4},
        {"event": "model_start", "run_id": "m3"},
        {"event": "model_end", "run_id": "m3"},
        {"event": "model_start", "run_id": "m4"},
        {"event": "model_error", "run_id": "m4"},
        {"event": "side_call_late_end", "ask_id": "a1", "held_s": 301.2},
    ]
    counts = record_counts(recs)
    assert counts["side_call_timeouts"] == 2
    assert counts["side_calls_not_read"] == 1
    assert counts["side_call_late_ends"] == 1
    assert counts["model_calls_unended"] == 2


def _task(tid):
    return Task(id=tid, question="q", documents=[], category="locate",
                doc_type="drawing_set", checks=[])


def _row(md, arm):
    header = next(l for l in md.splitlines() if l.startswith("| arm |"))
    row = next(l for l in md.splitlines() if l.startswith(f"| {arm} |"))
    cols = [c.strip() for c in header.strip("|").split("|")]
    vals = [c.strip() for c in row.strip("|").split("|")]
    return dict(zip(cols, vals))


def test_the_table_has_the_two_counts_beside_errors_and_failed_calls():
    ok = {"score": {"passed": True, "checks_passed": 1, "checks_total": 1}}
    runs = {"baseline": {
        "a": dict(ok, side_call_timeouts=2, tool_calls_refused=3),
        "b": dict(ok, side_call_timeouts=1, tool_calls_refused=0)}}
    md = summarize(runs, ["baseline"], [_task("a"), _task("b")],
                   {"model": "m", "versions": {}, "updated": "now"})
    got = _row(md, "baseline")
    assert got["side-call timeouts"] == "3"
    assert got["refused tool calls"] == "3"
    assert got["errors"] == "0" and got["failed calls"] == "0"


def test_a_run_saved_before_the_counts_reads_not_recorded_not_zero():
    ok = {"score": {"passed": True, "checks_passed": 1, "checks_total": 1}}
    runs = {"baseline": {"a": dict(ok), "b": dict(ok)}}
    md = summarize(runs, ["baseline"], [_task("a"), _task("b")],
                   {"model": "m", "versions": {}, "updated": "now"})
    got = _row(md, "baseline")
    assert got["side-call timeouts"] == "·"
    assert got["refused tool calls"] == "·"


def test_rescoring_a_saved_run_fills_the_counts_its_record_can_give(tmp_path):
    """A run saved before the table had them: the refused calls and the
    unended model calls are in any record; timeouts are only in a record
    that wrote the events, so an old one leaves them unset."""
    logger = ActivityLogger(str(tmp_path), turn=1)
    refusal = unknown_arguments_error("analyze_pdf_page", ["pdf_pages"],
                                      ["document", "page", "pdf_page"])
    _tool(logger, "analyze_pdf_page", {"pdf_pages": "4-13"}, "call-0",
          result=refusal, delivered=refusal)
    saved = {"answer": "", "files": [], "tool_calls": [],
             "score": {"passed": True, "checks_passed": 0, "checks_total": 0}}
    out = runner.rescore_saved(saved, _task("a"), str(tmp_path))
    assert out["tool_calls_refused"] == 1
    assert out["model_calls_unended"] == 0
    assert "side_call_timeouts" not in out
