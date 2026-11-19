"""Audit of one turn of Document Review, 2026-11-18
(``module_work/audit_2026_11_18/TURN.md``): the record's side."""

import os

from aec_agent_core.activity_log import ActivityLogger


def test_two_cut_texts_of_one_turn_keep_a_whole_copy_each(tmp_path):
    """Tool-call ids of one provider all start alike (``toolu_01...``).
    The whole copy of a cut text was named by the id's first 8 characters,
    so the second call's copy overwrote the first and the first record's
    ``full`` pointed at the other call's text."""
    log = ActivityLogger(str(tmp_path), turn=1)
    first = "FIRST " * 20000
    second = "SECOND " * 20000
    _, cut1, _, name1 = log._cap_whole(first, 1000, "delivered_read_document",
                                       "toolu_01AAAAmadeup0001")
    _, cut2, _, name2 = log._cap_whole(second, 1000,
                                       "delivered_read_document",
                                       "toolu_01BBBBmadeup0002")
    assert cut1 and cut2 and name1 and name2
    assert name1 != name2
    with open(os.path.join(str(tmp_path), name1), encoding="utf-8") as fh:
        assert fh.read() == first
    with open(os.path.join(str(tmp_path), name2), encoding="utf-8") as fh:
        assert fh.read() == second


def test_a_run_id_keeps_the_name_it_had(tmp_path):
    """A callback's run id (a UUID) names its copy as before."""
    log = ActivityLogger(str(tmp_path), turn=2)
    _, _, _, name = log._cap_whole(
        "x" * 5000, 1000, "tool_end_big",
        "01a1f1d2-4877-737f-86ec-990f33368da1")
    assert name.endswith("turn2_tool-end-big_01a1f1d2.txt")


def _model_end(tmp_path, message):
    import json
    import uuid
    from langchain_core.outputs import ChatGeneration, LLMResult
    log = ActivityLogger(str(tmp_path), turn=1)
    log.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]),
                   run_id=uuid.uuid4())
    with open(os.path.join(str(tmp_path), "activity.jsonl"),
              encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if r.get("event") == "model_end"][-1]


def test_the_record_says_a_reply_stopped_at_the_output_limit(tmp_path):
    """A text reply still cut after its continuations is delivered as it
    is; nothing in the record said any reply had been cut."""
    from langchain_core.messages import AIMessage
    cut = _model_end(tmp_path / "a", AIMessage(
        content="Part 1 of an answer that is cut",
        response_metadata={"finish_reason": "length"}))
    assert cut.get("output_cut") is True
    whole = _model_end(tmp_path / "b", AIMessage(
        content="A whole answer.",
        response_metadata={"finish_reason": "stop"}))
    assert "output_cut" not in whole
