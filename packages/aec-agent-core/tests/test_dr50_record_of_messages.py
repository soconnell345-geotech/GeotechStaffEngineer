"""A helper's record made after the fact from its own messages
(``keep_work.record_of_messages``, BACKLOG DR-50 part 3): the minimal
build's helper is not a ``task`` call ``KeepWork`` wraps, so its open items
are read from the messages its run returns. No model; made-up values."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from aec_agent_core.deep import keep_work

DOC = "made_up_set.pdf"
ASK = "Read the made-up note in this box."


def _ask(name, args, cid, text=""):
    return AIMessage(content=text, tool_calls=[{"name": name, "args": args,
                                                "id": cid}])


def _messages():
    return [
        HumanMessage(content="LOOKING JOB: pages 1-2 of the made-up set."),
        _ask("render_region", {"attachment_key": DOC, "pdf_page": 1,
                               "prompt": ASK, "bbox": [10, 10, 200, 80]},
             "z1", "Zooming on the note."),
        ToolMessage(content=json.dumps({"error": "made-up render failure"}),
                    tool_call_id="z1", name="render_region"),
        _ask("render_region", {"attachment_key": DOC, "pdf_page": 1,
                               "prompt": ASK, "bbox": [0, 0, 150, 60]}, "z2"),
        ToolMessage(content=json.dumps({"page": 0, "pdf_page": 1,
                                        "bbox": [0, 0, 150, 60]}),
                    tool_call_id="z2", name="render_region"),
        _ask("analyze_pdf_page", {"attachment_key": DOC, "pdf_page": 2},
             "p2"),
        ToolMessage(content="made-up tool crash", tool_call_id="p2",
                    name="analyze_pdf_page", status="error"),
        AIMessage(content="Findings: none."),
    ]


def test_the_record_holds_the_helper_s_steps_in_order():
    rec = keep_work.record_of_messages(_messages(), subagent="page_looker",
                                       brief="LOOKING JOB")
    assert rec.subagent == "page_looker" and rec.brief == "LOOKING JOB"
    assert [s["kind"] for s in rec.steps] == [
        "model", "result", "model", "result", "model", "result", "model"]
    assert rec.steps[0]["text"] == "Zooming on the note."
    assert rec.steps[0]["calls"][0]["id"] == "z1"
    assert [s["status"] for s in rec.steps if s["kind"] == "result"] == [
        "success", "success", "error"]
    assert rec.model_calls == 4 and rec.n_results == 3
    assert rec.last_words() == "Findings: none."
    # no running helper is set: what reads it is as it was
    assert keep_work.current_helper() is None


def test_its_open_items_are_the_default_reader_s():
    """The zoom redone in an overlapping box after its failure came back is
    settled; the look that failed is listed, as ``open_items`` lists it for
    a recorded reader."""
    rec = keep_work.record_of_messages(_messages())
    assert keep_work.open_items(rec) == [
        f"- {DOC} page 2: analyze_pdf_page failed: made-up tool crash"]


def test_nothing_to_read_makes_an_empty_record():
    for msgs in (None, [], [HumanMessage(content="brief")]):
        rec = keep_work.record_of_messages(msgs)
        assert rec.steps == [] and keep_work.open_items(rec) == []
