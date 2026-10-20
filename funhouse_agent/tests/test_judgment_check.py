"""The judgment-call check's parts (BACKLOG GM-174): the list finder shared
with the live smoke harness (:mod:`funhouse_agent.answer_sections`), the
source test and the "no reference found" words
(:mod:`funhouse_agent.judgment_check`). The turn itself is driven in
``webapp/tests/test_judgment_check_turn.py``."""

from __future__ import annotations

import pytest

from funhouse_agent import answer_sections as sections
from funhouse_agent import judgment_check as jc


# --- the list finder ------------------------------------------------------

def test_items_are_the_top_level_entries_with_their_lines():
    part = sections.section(
        "Result.\n\n## Judgment calls\n\n"
        "1. **Wall.** Smooth.\n   The bonded case halves the strut force.\n"
        "   - a sub-point\n"
        "2. **Water.** None given.\n\n"
        "These are mine to check.\n\n## References\n- GEC-4\n",
        "judgment_calls")
    calls = sections.items(part)
    assert len(calls) == 2
    assert "bonded case" in calls[0] and "sub-point" in calls[0]
    assert calls[1] == "2. **Water.** None given."     # the paragraph after
                                                       # the list is not one


def test_items_read_bullets_a_table_or_paragraphs():
    bullets = "**Judgment calls**\n- one\n* two\n+ three\n"
    assert sections.items(sections.section(bullets, "judgment_calls")) == [
        "- one", "* two", "+ three"]
    table = ("### Judgment calls\n| Call | Basis |\n|---|---|\n"
             "| Smooth wall | GEC-4 Sec. 5 |\n| No water | judgment |\n")
    rows = sections.items(sections.section(table, "judgment_calls"))
    assert rows == ["| Smooth wall | GEC-4 Sec. 5 |", "| No water | judgment |"]
    paras = ("Judgment calls:\n\n**Wall.** Smooth.\n\n**Water.** None.\n")
    assert len(sections.items(sections.section(paras, "judgment_calls"))) == 2


def test_no_list_and_an_empty_list():
    assert sections.section("No calls here.", "judgment_calls") is None
    assert sections.items(None) == []
    assert jc.judgment_calls("Plain answer.") is None
    assert jc.judgment_calls("## Judgment calls\n") == []


def test_the_harness_reads_the_same_list():
    """The live smoke harness's ``answer_section`` is this finder, so a wave
    measures the list the app acted on."""
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    sys.path.insert(0, os.path.join(root, "module_work"))
    try:
        det = pytest.importorskip("live_smoke.detectors")
    finally:
        sys.path.remove(os.path.join(root, "module_work"))
    text = "## Judgment calls\n1. a\n2. b\n## Next\n"
    assert det.answer_section(text, "judgment_calls") == \
        sections.section(text, "judgment_calls")
    assert det._SCOPES is sections.SCOPES


# --- the source test --------------------------------------------------------

@pytest.mark.parametrize("call", [
    "Smooth wall, per GEC-4 Section 5.4.",
    "K0 from UFC 3-220-10 §7-6.2.",
    "Nq off Fig. 6-16 of the manual.",
    "Figs. 6-16 and 6-17.",
    "Ka by Eq. 5-14.",
    "Values from Table 5-8.",
    "As printed on p. 23.",
    "Chapter 7 of the driven-pile manual.",
    "App. A.5 weak-rock curves.",
    "Long-term case, as Duncan & Wright (2014) advise.",
    "K0 = 0.658 by Jaky (1944).",
    "Load factor from AASHTO LRFD 3.11.5.",
    "The tool's choices basis: FHWA NHI-06-088.",
])
def test_calls_with_a_source(call):
    assert jc.has_source(call)


@pytest.mark.parametrize("call", [
    "Smooth wall (the tool default).",
    "GEC-6 recommends Vesic.",
    "Two lifts were used, to 1.5 m and then 5 m.",
    "Drained parameters with no groundwater.",
    "The configuration is symmetric.",
    "See the table above.",
])
def test_calls_with_no_source(call):
    assert not jc.has_source(call)


@pytest.mark.parametrize("call, said", [
    ("Strut preload. This rests on judgment, no reference found.", True),
    ("Engineering judgment; no published source was found.", True),
    ("Judgment: I found no reference for this.", True),
    ("I found no reference for it.", False),          # judgment unnamed
    ("This rests on engineering judgment.", False),   # nothing looked for
    ("No reference basis beyond common practice.", False),
])
def test_the_no_reference_found_words(call, said):
    assert jc.says_none_found(call) is said


def test_check_names_only_the_unsourced_calls():
    answer = ("## Judgment calls\n"
              "1. **Wall.** Smooth, per GEC-4 Section 5.4.\n"
              "2. **Water.** None given.\n"
              "3. **Preload.** Rests on judgment, no reference found.\n")
    result = jc.check(answer)
    assert result["n_calls"] == 3
    # no consult: "no reference found" cannot be true yet, so it counts
    assert result["unsourced"] == ["Water. None given.",
                                   "Preload. Rests on judgment, no reference "
                                   "found."]
    assert result["acts"] is True and result["reason"] == jc.NO_CONSULT
    note = jc.note_for_model(result)
    assert note.startswith(jc.NOTE_PREFIX) and "- Water. None given." in note
    assert "`references` helper once, in one task" in note
    assert "one line each, asking only for its source" in note
    assert "This note comes once." in note
    # GM-187: after a consult, a call with no source still goes back
    consulted = jc.check(answer, references_ran=True)
    assert consulted["unsourced"] == ["Water. None given."]
    assert consulted["acts"] is True
    assert consulted["reason"] == jc.AFTER_CONSULT
    after = jc.note_for_model(consulted)
    assert after.startswith(jc.NOTE_PREFIX)
    assert "- Water. None given." in after and "Preload" not in after
    assert "Do not ask the helper again." in after
    assert "rests on judgment, no reference found" in after
    assert "This note comes once." in after
    # every call sourced after the consult: nothing to send back
    sourced = answer.replace("None given.", "None, per GEC-4 Section 5.6.")
    assert jc.check(sourced, references_ran=True)["acts"] is False


def test_a_long_call_is_cut_for_the_note():
    head = jc.call_head("- " + "word " * 80)
    assert len(head) <= jc.MAX_CALL_CHARS + 4 and head.endswith("...")


def test_the_check_sees_a_references_task_in_either_message_shape():
    from langchain_core.messages import AIMessage
    check = jc.TurnJudgmentCheck(mode="retry")
    check.saw([AIMessage(content="", tool_calls=[{
        "name": "task", "id": "1", "args": {"subagent_type": "calc"}}])])
    assert check.references_ran is False
    check.saw([{"role": "assistant", "tool_calls": [{
        "id": "2", "type": "function", "function": {
            "name": "task",
            "arguments": '{"subagent_type": "references"}'}}]}])
    assert check.references_ran is True


def test_feedback_is_once_and_finish_never_changes_the_answer():
    logged = []
    check = jc.TurnJudgmentCheck(mode="retry", log=logged.append)
    answer = "## Judgment calls\n1. Smooth wall.\n"
    assert check.feedback(answer).startswith(jc.NOTE_PREFIX)
    assert check.feedback(answer) is None
    assert check.finish(answer) == answer
    assert [r["action"] for r in logged] == ["fed_back", "none"]
    assert [r["reason"] for r in logged] == [jc.NO_CONSULT] * 2
    assert check.retry_status == jc.RETRY_STATUS
    off = jc.TurnJudgmentCheck(mode="off")
    assert off.feedback(answer) is None


def test_once_a_turn_whichever_note_went():
    """GM-187: sent back for no consult, the answer is not sent back again
    for calls still unsourced after the consult it asked for; sent back
    after a consult, it is not sent back again either."""
    answer = "## Judgment calls\n1. Smooth wall.\n"
    first = jc.TurnJudgmentCheck(mode="retry")
    assert first.feedback(answer)
    first.references_ran = True                 # the consult the note asked for
    assert first.feedback(answer) is None
    second = jc.TurnJudgmentCheck(mode="retry")
    second.references_ran = True
    note = second.feedback(answer)
    assert "Do not ask the helper again." in note
    assert second.retry_status == jc.RETRY_STATUS_AFTER_CONSULT
    assert second.feedback(answer) is None


def test_mode_needs_the_switch_and_a_build_that_asks_for_a_consult(
        monkeypatch):
    from funhouse_agent import review_flags

    class Agent:
        geotech_judgment_consult = True

    # on by default (GM-226, 2026-10-20)
    monkeypatch.delenv(review_flags.JUDGMENT_CALLS_ENV, raising=False)
    assert jc.mode_for(Agent()) == "retry"
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "0")
    assert jc.mode_for(Agent()) == "off"
    monkeypatch.setenv(review_flags.JUDGMENT_CALLS_ENV, "1")
    assert jc.mode_for(Agent()) == "retry"
    assert jc.mode_for(object()) == "off"


@pytest.mark.parametrize("call", [
    # w13 F51j, word for word: no consult ran
    "**Smooth wall contact (default) instead of bonded.** The default is the "
    "Rankine, no wall friction assumption, and I found no reference for it.",
    "**Smooth wall contact.** This rests on judgment, no reference found.",
])
def test_no_reference_found_with_no_consult_is_unsourced(call):
    """The exemption counts only after a consult: with none, nothing was
    looked for, so the call is sent back."""
    answer = "## Judgment calls\n\n- " + call + "\n"
    assert jc.check(answer)["acts"] is True
    assert jc.check(answer)["n_unsourced"] == 1
