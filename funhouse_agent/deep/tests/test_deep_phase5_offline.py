"""Offline tests for the v5.0 deepagents Phase 5 A/B eval harness (NO API key,
NO network, NO real model).

Phase 5 adds ``funhouse_agent.deep.eval_harness`` — an A/B runner + scorer that
compares the v1 agent against the v2 deepagents agent on the project's 68-question
suite. These tests cover the WHOLE pipeline offline:

  (a) The suite loader reads ``geotech_test_suite.json`` and returns the expected
      count/shape, and the suite is confirmed to carry NO expected answers (so
      correctness is honestly "not auto-scorable").
  (b) The two adapters (v1 AgentResult / v2 message list -> QAResult) correctly
      flag tool errors and extract answer/trace/rounds/usage.
  (c) The P1 hallucination-on-error heuristic on SYNTHETIC traces: an error +
      falsely-confident answer flags; an error that IS acknowledged does NOT; a
      clean run does NOT.
  (d) ``score_run`` aggregates the P1 rate, tool-error rate, and errors/question
      correctly, and ``render_markdown_table`` produces a metric|v1|v2|delta
      table.
  (e) ``run_ab`` + ``write_results`` over the STUB v1/v2 agents on a few questions
      produces a results JSON + markdown table with BOTH agents' metrics — no
      real model.

Run from the worktree root with the venv python::

    cd <worktree>
    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_deep_phase5_offline.py -v
"""

import json

import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from funhouse_agent.deep import eval_harness as eh
from funhouse_agent.react_support import AgentResult


# ===========================================================================
# (a) Suite loader
# ===========================================================================

def test_load_suite_shape_and_count():
    """The packaged suite loads as a list of {id, module, question} dicts."""
    questions = eh.load_suite()
    assert isinstance(questions, list)
    assert len(questions) >= 60  # the suite is ~68 questions
    for q in questions:
        assert "id" in q and "question" in q and "module" in q
    ids = [q["id"] for q in questions]
    assert "BC-1" in ids  # a known anchor question


def test_suite_has_no_expected_answers():
    """Correctness is NOT auto-scorable: the suite carries no expected answers."""
    questions = eh.load_suite()
    assert eh.suite_has_expected_answers(questions) is False
    c = eh.score_correctness([], questions)
    assert c["auto_scorable"] is False
    assert c["pass_rate"] is None
    assert "not auto-scorable" in c["note"].lower()


def test_suite_with_expected_answers_is_numeric_scorable():
    """A FUTURE suite variant with expected answers unlocks numeric grading."""
    questions = [
        {"id": "X-1", "module": "m", "question": "q", "expected": 100.0,
         "tolerance": 0.1},
    ]
    assert eh.suite_has_expected_answers(questions) is True
    qa = eh.QAResult(qid="X-1", module="m", agent="v1",
                     answer="The answer is about 105 kPa.")
    c = eh.score_correctness([qa], questions)
    assert c["auto_scorable"] is True
    assert c["method"] == "numeric_tolerance"
    assert c["pass_rate"] == 1.0  # 105 within 10% of 100


# ===========================================================================
# (b) Adapters
# ===========================================================================

def _v1_result(answer, tool_calls=None, errors=None, rounds=2):
    return AgentResult(
        answer=answer,
        tool_calls=tool_calls or [],
        rounds=rounds,
        total_time_s=0.0,
        errors=errors or [],
    )


def test_v1_adapter_flags_error_from_result_preview():
    """A v1 tool call whose result_preview is an error JSON is flagged errored."""
    res = _v1_result(
        "answer text that is long enough to be substantive for the heuristic.",
        tool_calls=[
            {"round": 1, "tool_name": "describe_method",
             "arguments": {"agent_name": "bearing_capacity", "method": "vesic"},
             "result_preview": '{"error": "Unknown method \'vesic\'."}'},
            {"round": 2, "tool_name": "call_agent",
             "arguments": {"agent_name": "bearing_capacity"},
             "result_preview": '{"q_ultimate_kPa": 512.3}'},
        ],
        errors=[{"round": 1, "type": "dispatch", "tool": "describe_method",
                 "message": "Unknown method 'vesic'."}],
    )
    qa = eh._v1_to_qa("BC-1", "bearing_capacity", res, latency_s=1.2)
    assert qa.agent == "v1"
    assert qa.rounds == 2
    assert len(qa.trace) == 2
    assert qa.trace[0].errored is True
    assert qa.trace[1].errored is False
    assert qa.has_tool_error is True
    assert qa.n_tool_errors == 1
    assert qa.usage is None  # v1 ClaudeEngine exposes no token usage
    assert qa.latency_s == 1.2


def test_v1_adapter_clean_run_has_no_error():
    qa = eh._v1_to_qa(
        "BC-2", "bearing_capacity",
        _v1_result("clean answer with a value of 350 kPa, nothing went wrong.",
                   tool_calls=[{"round": 1, "tool_name": "call_agent",
                                "arguments": {}, "result_preview": '{"ok": 1}'}]),
        latency_s=0.5,
    )
    assert qa.has_tool_error is False
    assert qa.n_tool_errors == 0


def _v2_messages(final_text, *, tool_error=False, with_usage=True):
    """Build a synthetic deepagents message list (call + result + final)."""
    call = AIMessage(
        content="",
        tool_calls=[{"name": "call_agent",
                     "args": {"agent_name": "bearing_capacity"},
                     "id": "call_1"}],
        usage_metadata=({"input_tokens": 1000, "output_tokens": 100,
                         "total_tokens": 1100} if with_usage else None),
    )
    if tool_error:
        result = ToolMessage(content='{"error": "boom"}', tool_call_id="call_1",
                             name="call_agent", status="error")
    else:
        result = ToolMessage(content='{"q_ultimate_kPa": 512.3}',
                             tool_call_id="call_1", name="call_agent")
    final = AIMessage(
        content=final_text,
        usage_metadata=({"input_tokens": 1300, "output_tokens": 80,
                         "total_tokens": 1380} if with_usage else None),
    )
    return {"messages": [HumanMessage(content="q"), call, result, final]}


def test_v2_adapter_extracts_answer_trace_rounds_usage():
    msgs = _v2_messages("The ultimate bearing capacity is about 512 kPa.")
    qa = eh._v2_to_qa("BC-1", "bearing_capacity", msgs, latency_s=3.4)
    assert qa.agent == "v2"
    assert "512" in qa.answer
    assert qa.rounds == 1  # one AIMessage issued a tool call
    assert len(qa.trace) == 1
    assert qa.trace[0].name == "call_agent"
    assert qa.trace[0].errored is False
    assert qa.has_tool_error is False
    assert qa.usage is not None
    assert qa.usage["total_tokens"] == 1100 + 1380
    assert qa.latency_s == 3.4


def test_v2_adapter_flags_tool_error_from_status_and_json():
    msgs = _v2_messages("Done — the value is 512 kPa.", tool_error=True)
    qa = eh._v2_to_qa("BC-1", "bearing_capacity", msgs, latency_s=1.0)
    assert qa.trace[0].errored is True
    assert qa.has_tool_error is True
    assert qa.errors  # an error entry was recorded


def test_v2_adapter_no_usage_returns_none():
    msgs = _v2_messages("Plain answer.", with_usage=False)
    qa = eh._v2_to_qa("X", "m", msgs, latency_s=0.0)
    assert qa.usage is None


# ===========================================================================
# (c) P1 hallucination-on-error heuristic
# ===========================================================================

def _qa(answer, *, errored, agent="v1", exception=None):
    """Build a QAResult with one (optionally errored) tool call."""
    return eh.QAResult(
        qid="Q", module="m", agent=agent, answer=answer,
        trace=[eh.ToolCallRecord(name="call_agent", args={}, errored=errored)],
        rounds=1, exception=exception,
    )


def test_p1_flags_error_with_confident_answer():
    """Error in trace + confident, substantive answer + no acknowledgement => P1."""
    qa = _qa(
        "The ultimate bearing capacity is 512 kPa and the allowable is 170 kPa. "
        "These results are reliable for the given parameters.",
        errored=True,
    )
    assert eh.is_hallucination_on_error(qa) is True


def test_p1_not_flagged_when_failure_acknowledged():
    """An error that the answer acknowledges is NOT a P1 hallucination."""
    qa = _qa(
        "The tool call failed with an unknown-method error, so I could not "
        "compute a reliable bearing capacity. Please retry with valid inputs.",
        errored=True,
    )
    assert eh.is_hallucination_on_error(qa) is False


def test_p1_not_flagged_for_clean_run():
    """No tool error => never a P1 hallucination, even with a confident answer."""
    qa = _qa(
        "The ultimate bearing capacity is 512 kPa and the allowable is 170 kPa.",
        errored=False,
    )
    assert eh.is_hallucination_on_error(qa) is False


def test_p1_not_flagged_for_empty_or_short_answer():
    """An error with no substantive answer (e.g. max-rounds) is not a P1."""
    qa = _qa("[Reached max rounds.]", errored=True)
    assert eh.is_hallucination_on_error(qa) is False


def test_p1_not_flagged_when_ask_crashed():
    """A hard exception is a crash, not a confabulation."""
    qa = _qa(
        "Some confident-looking text that is plenty long to be substantive here.",
        errored=True, exception="RuntimeError: boom",
    )
    assert eh.is_hallucination_on_error(qa) is False


def test_acknowledgement_lexicon():
    assert eh.answer_acknowledges_failure("the call errored out") is True
    assert eh.answer_acknowledges_failure("I was unable to compute it") is True
    assert eh.answer_acknowledges_failure("approximately 500 kPa") is True
    assert eh.answer_acknowledges_failure("the value is exactly 512 kPa") is False


# ===========================================================================
# (d) Aggregate scoring + markdown table
# ===========================================================================

def _run_questions():
    return [
        {"id": "Q1", "module": "m", "question": "q1"},
        {"id": "Q2", "module": "m", "question": "q2"},
        {"id": "Q3", "module": "m", "question": "q3"},
        {"id": "Q4", "module": "m", "question": "q4"},
    ]


def test_score_run_metrics_on_synthetic_records():
    """One P1 hallucination out of four => p1 rate 0.25; error rate tracks too."""
    qas = [
        # Q1: error + confident answer -> P1 hallucination.
        eh.QAResult(qid="Q1", module="m", agent="v1",
                    answer="The bearing capacity is 512 kPa, fully reliable here.",
                    trace=[eh.ToolCallRecord("call_agent", {}, errored=True)],
                    rounds=2, latency_s=1.0,
                    errors=[{"tool": "call_agent", "message": "boom"}]),
        # Q2: error but acknowledged -> NOT a hallucination (still a tool error).
        eh.QAResult(qid="Q2", module="m", agent="v1",
                    answer="The tool failed, so I could not compute a value.",
                    trace=[eh.ToolCallRecord("describe_method", {}, errored=True)],
                    rounds=1, latency_s=2.0,
                    errors=[{"tool": "describe_method", "message": "unknown"}]),
        # Q3: clean run.
        eh.QAResult(qid="Q3", module="m", agent="v1",
                    answer="The bearing capacity is 600 kPa for these inputs.",
                    trace=[eh.ToolCallRecord("call_agent", {}, errored=False)],
                    rounds=1, latency_s=1.5),
        # Q4: clean run.
        eh.QAResult(qid="Q4", module="m", agent="v1",
                    answer="Settlement is 25 mm under the applied load.",
                    trace=[eh.ToolCallRecord("call_agent", {}, errored=False)],
                    rounds=3, latency_s=0.5),
    ]
    m = eh.score_run(qas, _run_questions())
    assert m["n_questions"] == 4
    assert m["p1_count"] == 1
    assert m["p1_hallucination_rate"] == pytest.approx(0.25)
    assert m["p1_question_ids"] == ["Q1"]
    assert m["tool_error_rate"] == pytest.approx(0.5)  # Q1 + Q2 have errors
    assert m["errors_per_question"] == pytest.approx(0.5)  # 2 errors / 4 q
    assert m["avg_rounds"] == pytest.approx((2 + 1 + 1 + 3) / 4)
    assert m["avg_latency_s"] == pytest.approx((1.0 + 2.0 + 1.5 + 0.5) / 4)
    assert m["total_tokens"] is None  # no usage on these records
    assert m["correctness"]["auto_scorable"] is False


def test_render_markdown_table_has_metric_rows():
    ab = {
        "model": "claude-sonnet-4-6",
        "n_questions": 4,
        "v1": {"metrics": eh.score_run(
            [eh.QAResult(qid="Q1", module="m", agent="v1",
                         answer="confident reliable answer, 512 kPa, all good here.",
                         trace=[eh.ToolCallRecord("call_agent", {}, errored=True)],
                         rounds=2, latency_s=1.0, errors=[{"message": "x"}])],
            _run_questions()[:1])},
        "v2": {"metrics": eh.score_run(
            [eh.QAResult(qid="Q1", module="m", agent="v2",
                         answer="the tool errored so I could not finish.",
                         trace=[eh.ToolCallRecord("call_agent", {}, errored=True)],
                         rounds=1, latency_s=1.0, errors=[{"message": "x"}])],
            _run_questions()[:1])},
    }
    table = eh.render_markdown_table(ab)
    assert "| Metric | v1 | v2 | delta" in table
    assert "P1 hallucination-on-error rate" in table
    assert "Tool-error rate" in table
    assert "claude-sonnet-4-6" in table
    # v1 hallucinated (100%), v2 did not (0%): the delta row should read "better".
    p1_line = [ln for ln in table.splitlines()
               if ln.startswith("| P1 hallucination-on-error rate")][0]
    assert "100.0%" in p1_line and "0.0%" in p1_line
    assert "better" in p1_line
    # Correctness section is honest about not being auto-scorable.
    assert "not auto-scorable" in table.lower()


# ===========================================================================
# (e) End-to-end --dry-run over STUB agents (no model)
# ===========================================================================

def test_run_ab_with_stub_agents(tmp_path):
    """run_ab over the stub v1/v2 agents produces both agents' metrics offline."""
    questions = eh.load_suite()
    ab = eh.run_ab(
        questions,
        v1_agent=eh._StubV1Agent(),
        v2_agent=eh._StubV2Agent(),
        limit=3,
        verbose=False,
    )
    assert ab["n_questions"] == 3
    assert len(ab["v1"]["results"]) == 3
    assert len(ab["v2"]["results"]) == 3
    # The stub v2 agent reports token usage; the stub v1 does not.
    assert ab["v1"]["metrics"]["total_tokens"] is None
    assert ab["v2"]["metrics"]["total_tokens"] is not None
    # Both stub agents return clean (no-error) traces => zero P1 hallucinations.
    assert ab["v1"]["metrics"]["p1_count"] == 0
    assert ab["v2"]["metrics"]["p1_count"] == 0


def test_write_results_produces_json_and_markdown(tmp_path):
    questions = eh.load_suite()
    ab = eh.run_ab(
        questions, v1_agent=eh._StubV1Agent(), v2_agent=eh._StubV2Agent(),
        limit=2, verbose=False,
    )
    ab["model"] = "DRY-RUN-STUB"
    out = tmp_path / "ab_results.json"
    paths = eh.write_results(ab, out)
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "markdown_table" in data
    assert data["v1"]["metrics"]["n_questions"] == 2
    assert data["v2"]["metrics"]["n_questions"] == 2
    md = tmp_path / "ab_results.json.md"
    assert md.exists()
    assert "A/B eval" in md.read_text(encoding="utf-8")


def test_cli_dry_run_writes_outputs(tmp_path):
    """The CLI --dry-run path runs the whole harness offline and writes outputs."""
    out = tmp_path / "cli_ab.json"
    rc = eh.main(["--dry-run", "--limit", "2", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["model"] == "DRY-RUN-STUB"
    assert data["n_questions"] == 2
    assert "v1" in data and "v2" in data


def test_run_ab_records_hard_exception(tmp_path):
    """A crashing agent is recorded as an exception, not allowed to abort the run."""

    class _Boom:
        def ask(self, q):
            raise RuntimeError("v1 boom")

        def invoke(self, payload, *a, **k):
            raise RuntimeError("v2 boom")

    questions = eh.load_suite()
    ab = eh.run_ab(
        questions, v1_agent=_Boom(), v2_agent=_Boom(), limit=1, verbose=False,
    )
    v1_res = ab["v1"]["results"][0]
    v2_res = ab["v2"]["results"][0]
    assert "v1 boom" in (v1_res["exception"] or "")
    assert "v2 boom" in (v2_res["exception"] or "")
    # Exceptions are crashes, not P1 hallucinations.
    assert ab["v1"]["metrics"]["p1_count"] == 0
    assert ab["v1"]["metrics"]["exception_rate"] == pytest.approx(1.0)
