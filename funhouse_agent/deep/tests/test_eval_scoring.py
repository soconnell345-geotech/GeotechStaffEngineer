"""Offline tests for the v5.1 expected-answer auto-scoring in eval_harness
(NO API key, NO network, NO real model).

Covers the v5.1 "Correctness is not auto-scored" gap-closure
(docs/V5.0_TODO.md section C):

  (a) The packaged suite now CARRIES ``expected`` answer keys (module-run
      ground truth) on calc-type questions.
  (b) :func:`default_judge` — numeric extraction + tolerance compare (with
      ``alt`` unit variants) and keyword matching; questions WITHOUT
      ``expected`` are SKIPPED (None), never failed.
  (c) :func:`score_correctness` — aggregates graded/passed/skipped with the
      default judge; legacy scalar ``expected`` still grades; a no-expected
      question set is still honestly "not auto-scorable".
  (d) :func:`make_llm_judge` — the OPT-IN LLM judge hook, exercised with a
      STUB model (no live calls): PASS/FAIL parsing and the unparseable->skip
      path.

Run from the worktree root with the venv python::

    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_eval_scoring.py -v
"""

import pytest

from funhouse_agent.deep import eval_harness as eh


def _qa(qid, answer):
    return eh.QAResult(qid=qid, module="m", agent="v2", answer=answer)


def _traced_qa(qid, answer, calls):
    """QAResult with a trace of ``(tool_name, errored)`` tuples."""
    return eh.QAResult(
        qid=qid, module="m", agent="v2", answer=answer,
        trace=[eh.ToolCallRecord(name=n, errored=e) for n, e in calls],
    )


# ===========================================================================
# (a) The packaged suite carries expected answer keys
# ===========================================================================

def test_packaged_suite_carries_expected_answers():
    suite = eh.load_suite()
    assert eh.suite_has_expected_answers(suite) is True
    with_expected = [q for q in suite if "expected" in q]
    assert len(with_expected) >= 25  # 31 as of v5.1
    for q in with_expected:
        exp = q["expected"]
        assert isinstance(exp, dict)
        assert exp.get("values") or exp.get("keywords"), q["id"]
        assert "source" in exp, q["id"]  # ground-truth provenance is mandatory
        for entry in exp.get("values", []):
            assert "value" in entry and "rtol" in entry, q["id"]


# ===========================================================================
# (b) default_judge
# ===========================================================================

def test_default_judge_numeric_within_tolerance_passes():
    q = {"id": "X", "expected": {"values": [{"name": "q_ult", "value": 1593,
                                             "rtol": 0.15}]}}
    assert eh.default_judge(q, _qa("X", "The ultimate capacity is 1600 kPa.")) is True


def test_default_judge_numeric_outside_tolerance_fails():
    q = {"id": "X", "expected": {"values": [{"name": "q_ult", "value": 1593,
                                             "rtol": 0.15}]}}
    assert eh.default_judge(q, _qa("X", "The capacity is 900 kPa.")) is False


def test_default_judge_matches_any_number_not_just_first():
    """Answers list inputs before results — any-number matching handles that."""
    q = {"id": "X", "expected": {"values": [{"name": "s", "value": 41.8,
                                             "rtol": 0.1}]}}
    ans = "For B = 3 m, Cc = 0.25 and e0 = 0.9, the settlement is 42 mm."
    assert eh.default_judge(q, _qa("X", ans)) is True


def test_default_judge_alt_values_cover_unit_variants():
    q = {"id": "X", "expected": {"values": [{"name": "s_mm", "value": 41.8,
                                             "rtol": 0.1, "alt": [0.042]}]}}
    assert eh.default_judge(q, _qa("X", "Settlement is about 0.042 m.")) is True
    assert eh.default_judge(q, _qa("X", "Settlement is about 0.30 m.")) is False


def test_default_judge_keywords_case_insensitive():
    q = {"id": "X", "expected": {"keywords": ["class D"]}}
    assert eh.default_judge(q, _qa("X", "This is Site CLASS D per ASCE 7.")) is True
    assert eh.default_judge(q, _qa("X", "This is Site Class C.")) is False


def test_default_judge_requires_all_values_and_keywords():
    q = {"id": "X", "expected": {
        "values": [{"name": "ka", "value": 0.295, "rtol": 0.03}],
        "keywords": ["Rankine"],
    }}
    assert eh.default_judge(q, _qa("X", "Rankine Ka = 0.295.")) is True
    assert eh.default_judge(q, _qa("X", "Ka = 0.295.")) is False        # keyword missing
    assert eh.default_judge(q, _qa("X", "Rankine Ka = 0.50.")) is False  # value off


def test_default_judge_skips_question_without_expected():
    assert eh.default_judge({"id": "X", "question": "q"}, _qa("X", "42")) is None


def test_default_judge_legacy_scalar_expected_still_grades():
    q = {"id": "X", "expected": 100.0, "tolerance": 0.1}
    assert eh.default_judge(q, _qa("X", "About 105 kPa.")) is True
    assert eh.default_judge(q, _qa("X", "About 150 kPa.")) is False


def test_number_extraction_handles_commas_and_exponents():
    assert 1593.0 in eh._extract_numbers("q_ult = 1,593 kPa")
    assert 0.042 in eh._extract_numbers("4.2e-2 m of settlement")


# ===========================================================================
# (c) score_correctness with the default judge
# ===========================================================================

def test_score_correctness_grades_expected_and_skips_rest():
    questions = [
        {"id": "A", "expected": {"values": [{"name": "v", "value": 100, "rtol": 0.05}]}},
        {"id": "B", "expected": {"values": [{"name": "v", "value": 200, "rtol": 0.05}]}},
        {"id": "C"},  # no expected -> skipped, NOT failed
    ]
    qas = [
        _qa("A", "The result is 101."),   # pass
        _qa("B", "The result is 150."),   # fail
        _qa("C", "Some discursive answer with the number 7 in it."),
    ]
    c = eh.score_correctness(qas, questions)
    assert c["auto_scorable"] is True
    assert c["method"] == "expected"
    assert c["n_graded"] == 2
    assert c["n_passed"] == 1
    assert c["n_skipped"] == 1
    assert c["pass_rate"] == pytest.approx(0.5)


def test_score_correctness_without_any_expected_is_not_auto_scorable():
    questions = [{"id": "A"}, {"id": "B"}]
    qas = [_qa("A", "answer"), _qa("B", "answer")]
    c = eh.score_correctness(qas, questions)
    assert c["auto_scorable"] is False
    assert c["pass_rate"] is None
    assert c["n_skipped"] == 2
    assert "not auto-scorable" in c["note"].lower()


def test_score_run_correctness_block_uses_default_judge():
    questions = [
        {"id": "A", "expected": {"values": [{"name": "v", "value": 50, "rtol": 0.1}]}},
        {"id": "B"},
    ]
    qas = [_qa("A", "Computed value: 51 kPa."), _qa("B", "n/a")]
    m = eh.score_run(qas, questions)
    assert m["correctness"]["auto_scorable"] is True
    assert m["correctness"]["n_graded"] == 1
    assert m["correctness"]["n_passed"] == 1
    assert m["correctness"]["n_skipped"] == 1


def test_render_suite_markdown_reports_correctness_when_scorable():
    questions = [
        {"id": "A", "expected": {"values": [{"name": "v", "value": 50, "rtol": 0.1}]}},
    ]
    qa = _qa("A", "Computed value: 51 kPa.")
    d = qa.to_dict()
    d["question"] = "What is v?"
    suite_result = {
        "model": "stub", "n": 1, "results": [d],
        "metrics": eh.score_run([qa], questions),
    }
    md = eh.render_suite_markdown(suite_result)
    assert "Correctness" in md
    assert "1/1" in md
    assert "auto-scored" in md.lower()


# ===========================================================================
# (d) make_llm_judge — opt-in, exercised with a STUB model (no live calls)
# ===========================================================================

class _StubModel:
    """Canned chat model: records the prompt, returns a fixed reply."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        class _Msg:
            def __init__(self, content):
                self.content = content
        return _Msg(self.reply)


def test_llm_judge_parses_pass_and_fail():
    q = {"id": "A", "question": "What is Ka?",
         "expected": {"values": [{"name": "ka", "value": 0.295, "rtol": 0.03}]}}
    qa = _qa("A", "Ka is 0.295 by Rankine.")

    judge_pass = eh.make_llm_judge(_StubModel("PASS — value matches."))
    judge_fail = eh.make_llm_judge(_StubModel("FAIL: the value is wrong."))
    assert judge_pass(q, qa) is True
    assert judge_fail(q, qa) is False


def test_llm_judge_unparseable_verdict_skips():
    judge = eh.make_llm_judge(_StubModel("I am not sure about this one."))
    assert judge({"id": "A", "question": "q"}, _qa("A", "ans")) is None


def test_llm_judge_prompt_includes_question_expected_and_answer():
    model = _StubModel("PASS")
    judge = eh.make_llm_judge(model)
    q = {"id": "A", "question": "What is the bearing capacity?",
         "expected": {"values": [{"name": "q_ult", "value": 1593, "rtol": 0.15}]}}
    judge(q, _qa("A", "About 1600 kPa."))
    prompt = model.prompts[0]
    assert "What is the bearing capacity?" in prompt
    assert "1593" in prompt
    assert "About 1600 kPa." in prompt
    assert "PASS or FAIL" in prompt


def test_llm_judge_feeds_score_correctness():
    questions = [{"id": "A", "question": "q"}, {"id": "B", "question": "q"}]
    qas = [_qa("A", "ans"), _qa("B", "ans")]
    c = eh.score_correctness(qas, questions,
                             judge_fn=eh.make_llm_judge(_StubModel("PASS")))
    assert c["auto_scorable"] is True
    assert c["method"] == "llm_judge"
    assert c["n_graded"] == 2
    assert c["n_passed"] == 2


# ===========================================================================
# (e) error_recovered — the P1 recovered-after-error split
# ===========================================================================

# A substantive answer with no acknowledgement word -> is_hallucination_on_error.
_HALLUC_ANSWER = ("The ultimate bearing capacity is about 512 kPa and the "
                  "allowable capacity is 170 kPa for this footing.")


def test_error_recovered_true_when_same_tool_later_succeeds():
    """Errored call, then a LATER non-errored call of the same tool -> recovered."""
    qa = _traced_qa("A", _HALLUC_ANSWER, [
        ("call_agent", True),    # first attempt errors
        ("call_agent", False),   # retry of the SAME tool succeeds
    ])
    assert eh.error_recovered(qa) is True


def test_error_recovered_false_when_no_later_success():
    """Errored call with no later success of that tool -> not recovered."""
    qa = _traced_qa("A", _HALLUC_ANSWER, [
        ("describe_method", False),
        ("call_agent", True),    # errors and is never retried successfully
    ])
    assert eh.error_recovered(qa) is False


def test_error_recovered_false_when_only_a_different_tool_succeeds():
    """A later success of a DIFFERENT tool does not recover the errored one."""
    qa = _traced_qa("A", _HALLUC_ANSWER, [
        ("call_agent", True),        # this tool errors...
        ("list_methods", False),     # ...and only a different tool succeeds after
    ])
    assert eh.error_recovered(qa) is False


def test_error_recovered_false_when_no_error_in_trace():
    """No errored call -> nothing to recover from -> False."""
    qa = _traced_qa("A", _HALLUC_ANSWER, [("call_agent", False)])
    assert eh.error_recovered(qa) is False


def test_error_recovered_requires_every_errored_tool_to_recover():
    """All errored tool names must recover; one unrecovered -> False."""
    recovered = _traced_qa("A", _HALLUC_ANSWER, [
        ("call_agent", True), ("call_agent", False),
        ("describe_method", True), ("describe_method", False),
    ])
    assert eh.error_recovered(recovered) is True
    one_unrecovered = _traced_qa("B", _HALLUC_ANSWER, [
        ("call_agent", True), ("call_agent", False),
        ("describe_method", True),   # never retried successfully
    ])
    assert eh.error_recovered(one_unrecovered) is False


def test_score_run_splits_p1_into_recovered_and_unrecovered():
    """score_run adds the additive recovered/unrecovered P1 split (+ qid lists)."""
    recovered = _traced_qa("REC", _HALLUC_ANSWER, [
        ("call_agent", True), ("call_agent", False)])
    unrecovered = _traced_qa("UNREC", _HALLUC_ANSWER, [
        ("call_agent", True)])
    clean = _traced_qa("OK", _HALLUC_ANSWER, [("call_agent", False)])

    m = eh.score_run([recovered, unrecovered, clean],
                     [{"id": "REC"}, {"id": "UNREC"}, {"id": "OK"}])
    assert m["p1_count"] == 2                       # both errored + unhedged
    assert m["p1_recovered_count"] == 1
    assert m["p1_unrecovered_count"] == 1
    assert m["p1_recovered_question_ids"] == ["REC"]
    assert m["p1_unrecovered_question_ids"] == ["UNREC"]
    # The split sums back to the P1 total (no double counting).
    assert m["p1_recovered_count"] + m["p1_unrecovered_count"] == m["p1_count"]


def test_render_suite_markdown_shows_recovered_split():
    """The run-metrics P1 line surfaces the recovered-after-error split."""
    recovered = _traced_qa("REC", _HALLUC_ANSWER, [
        ("call_agent", True), ("call_agent", False)])
    d = recovered.to_dict(); d["question"] = "q?"
    suite_result = {
        "model": "stub", "n": 1, "results": [d],
        "metrics": eh.score_run([recovered], [{"id": "REC"}]),
    }
    md = eh.render_suite_markdown(suite_result)
    assert "recovered-after-error" in md
    assert "likely false positives" in md


# ===========================================================================
# (f) check_optional_deps — the graceful optional-dependency preflight
# ===========================================================================

def test_check_optional_deps_reports_missing_and_present():
    """A guaranteed-missing import is reported; an importable one is not."""
    deps = {"json": "stdlib", "totally_missing_pkg_zzz_9x": "full"}
    missing = eh.check_optional_deps(deps)
    assert missing == [{"import": "totally_missing_pkg_zzz_9x", "extra": "full"}]


def test_check_optional_deps_empty_when_all_present():
    assert eh.check_optional_deps({"json": "x", "sys": "y"}) == []


def test_check_optional_deps_never_raises_and_sorts():
    """Never raises on a broken import; output is sorted by import name."""
    deps = {"zzz_missing_b": "full", "aaa_missing_a": "deep"}
    missing = eh.check_optional_deps(deps)
    assert [d["import"] for d in missing] == ["aaa_missing_a", "zzz_missing_b"]


def test_optional_deps_map_covers_eval_named_backends():
    """The 10 backends that failed on the 2026-07-05 cluster run are all mapped."""
    eval_named = {"gstools", "pygef", "python_ags4", "pydiggs", "ezdxf",
                  "SALib", "pystrata", "eqsig", "liquepy", "openseespy"}
    assert eval_named <= set(eh.OPTIONAL_DEPS)


def test_check_optional_deps_default_returns_shaped_list():
    """The no-arg call returns a list of {import, extra} dicts (env-dependent)."""
    out = eh.check_optional_deps()
    assert isinstance(out, list)
    for d in out:
        assert set(d) == {"import", "extra"}
        assert d["import"] in eh.OPTIONAL_DEPS
