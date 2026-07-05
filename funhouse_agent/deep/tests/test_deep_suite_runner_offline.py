"""Offline tests for the v2-ONLY suite runner (NO API key, NO network, NO model).

The v5.0 deepagents migration adds a Funhouse-friendly, single-agent suite
runner — :func:`funhouse_agent.deep.eval_harness.run_suite` — so a non-developer
owner can run the whole geotech suite through the v2 deepagents agent on their
OWN Funhouse Prompter/GPT model (no Claude key, no v1 agent, no A/B). These tests
drive that runner offline by:

  (a) Stubbing the compiled v2 agent (an object whose ``.invoke(...)`` returns a
      canned deepagents-style message list — same shape as
      ``eval_harness._StubV2Agent``) and monkeypatching ``build_deep_agent`` so
      ``run_suite`` never builds a real model.
  (b) Passing a tiny explicit ``questions=`` list (the test seam) so the loader
      is bypassed and the run is fast + deterministic.

Coverage:
  * ``run_suite`` returns ``{model, n, results, metrics}`` with one QAResult dict
    per question, each carrying the agent's answer + the question text.
  * a per-question exception is recorded on that result and does NOT abort the run.
  * ``out=`` writes BOTH a results JSON and a readable markdown review; the
    markdown contains the questions + answers.
  * ``render_suite_markdown`` renders a synthetic suite_result top-to-bottom
    (metrics header + per-question question/answer/tools).

Run from the worktree root with the venv python::

    cd <worktree>
    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_deep_suite_runner_offline.py -v
"""

import json

import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from funhouse_agent.deep import eval_harness as eh


# ===========================================================================
# A stub compiled v2 agent: .invoke() returns a canned deepagents message list
# ===========================================================================

class _StubDeepAgent:
    """Canned v2 agent — its ``.invoke`` returns a per-question message list.

    Mirrors the deepagents invoke shape (``{"messages": [...]}``) with one
    ``call_agent`` tool call + result and a final assistant answer, so the whole
    v2 adapter + scorer path runs offline. The final answer echoes the question
    so the test can assert the answer landed in the right slot.
    """

    def __init__(self, *, tool_error=False):
        self.tool_error = tool_error
        self.calls = 0

    def invoke(self, payload, *args, **kwargs):
        self.calls += 1
        question = ""
        for m in payload.get("messages", []):
            if isinstance(m, dict) and m.get("role") == "user":
                question = m.get("content", "")
        call = AIMessage(
            content="",
            tool_calls=[{
                "name": "call_agent",
                "args": {"agent_name": "bearing_capacity",
                         "method": "bearing_capacity_analysis"},
                "id": "call_1",
            }],
            usage_metadata={"input_tokens": 1200, "output_tokens": 150,
                            "total_tokens": 1350},
        )
        if self.tool_error:
            result = ToolMessage(content='{"error": "boom"}',
                                 tool_call_id="call_1", name="call_agent",
                                 status="error")
        else:
            result = ToolMessage(content='{"q_ultimate_kPa": 512.3}',
                                 tool_call_id="call_1", name="call_agent")
        final = AIMessage(
            content=f"Answer for: {question} -> the value is about 512 kPa.",
            usage_metadata={"input_tokens": 1400, "output_tokens": 90,
                            "total_tokens": 1490},
        )
        return {"messages": [HumanMessage(content=question), call, result, final]}


def _patch_build(monkeypatch, agent):
    """Monkeypatch ``build_deep_agent`` (imported lazily inside run_suite)."""
    import funhouse_agent.deep.agent as agent_mod
    monkeypatch.setattr(agent_mod, "build_deep_agent", lambda *a, **k: agent)


def _tiny_suite():
    return [
        {"id": "BC-1", "module": "bearing_capacity",
         "question": "Bearing capacity of a 2 m footing, phi=30?"},
        {"id": "ST-1", "module": "settlement",
         "question": "Settlement under a 100 kPa load?"},
    ]


# ===========================================================================
# (a) run_suite over a stub agent + explicit questions
# ===========================================================================

def test_run_suite_returns_per_question_results(monkeypatch):
    """run_suite returns model/n/results/metrics with one answer per question."""
    _patch_build(monkeypatch, _StubDeepAgent())
    suite = _tiny_suite()
    out = eh.run_suite("fake-model", questions=suite, verbose=False)

    assert out["n"] == 2
    assert out["model"] == "fake-model"  # string model -> used verbatim as label
    assert len(out["results"]) == 2

    r0, r1 = out["results"]
    assert r0["qid"] == "BC-1" and r0["module"] == "bearing_capacity"
    assert r1["qid"] == "ST-1"
    # The answer landed in the right slot (the stub echoes the question).
    assert "512" in r0["answer"]
    assert r0["question"] == suite[0]["question"]
    assert r1["question"] == suite[1]["question"]
    # The v2 trace was extracted (one call_agent tool call, no error).
    assert len(r0["trace"]) == 1
    assert r0["trace"][0]["name"] == "call_agent"
    assert r0["has_tool_error"] is False
    # Metrics are the v2 process metrics via score_run.
    m = out["metrics"]
    assert m["n_questions"] == 2
    assert m["p1_count"] == 0
    assert m["tool_error_rate"] == 0.0
    assert m["total_tokens"] is not None  # the stub reports usage


def test_run_suite_respects_limit(monkeypatch):
    """limit= runs only the first N questions."""
    _patch_build(monkeypatch, _StubDeepAgent())
    out = eh.run_suite("fake-model", questions=_tiny_suite(), limit=1,
                       verbose=False)
    assert out["n"] == 1
    assert len(out["results"]) == 1
    assert out["results"][0]["qid"] == "BC-1"


def test_run_suite_default_loads_packaged_suite(monkeypatch):
    """questions=None loads the packaged 76-question suite via load_suite."""
    _patch_build(monkeypatch, _StubDeepAgent())
    out = eh.run_suite("fake-model", limit=3, verbose=False)
    assert out["n"] == 3
    # The first packaged question is BC-1 (a known anchor).
    assert out["results"][0]["qid"] == "BC-1"


def test_run_suite_records_per_question_exception(monkeypatch):
    """A crashing .invoke is recorded on that result and does not abort the run."""

    class _CrashOnce:
        def __init__(self):
            self.n = 0

        def invoke(self, payload, *a, **k):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("kaboom on Q1")
            return _StubDeepAgent().invoke(payload)

    _patch_build(monkeypatch, _CrashOnce())
    out = eh.run_suite("fake-model", questions=_tiny_suite(), verbose=False)
    assert out["n"] == 2
    # Q1 crashed -> exception recorded; Q2 still ran.
    assert "kaboom on Q1" in (out["results"][0]["exception"] or "")
    assert out["results"][1]["exception"] is None
    assert "512" in out["results"][1]["answer"]
    # A crash is NOT a P1 hallucination, but it IS an exception.
    assert out["metrics"]["p1_count"] == 0
    assert out["metrics"]["exception_rate"] == pytest.approx(0.5)


def test_run_suite_records_missing_optional_deps(monkeypatch):
    """The optional-dependency preflight lands in metrics + the markdown banner."""
    _patch_build(monkeypatch, _StubDeepAgent())
    monkeypatch.setattr(
        eh, "check_optional_deps",
        lambda *a, **k: [{"import": "gstools", "extra": "gstools"},
                         {"import": "pygef", "extra": "subsurface"}],
    )
    out = eh.run_suite("fake-model", questions=_tiny_suite(), verbose=False)
    missing = out["metrics"]["missing_optional_deps"]
    assert [d["import"] for d in missing] == ["gstools", "pygef"]

    md = eh.render_suite_markdown(out)
    assert "Missing optional packages" in md
    assert "gstools" in md and "pygef" in md
    assert "geotech-staff-engineer[deep,full]" in md


def test_run_suite_no_banner_when_all_deps_present(monkeypatch):
    """No missing-deps banner when the preflight comes back empty."""
    _patch_build(monkeypatch, _StubDeepAgent())
    monkeypatch.setattr(eh, "check_optional_deps", lambda *a, **k: [])
    out = eh.run_suite("fake-model", questions=_tiny_suite(), verbose=False)
    assert out["metrics"]["missing_optional_deps"] == []
    assert "Missing optional packages" not in eh.render_suite_markdown(out)


def test_run_suite_object_model_label_includes_model_id(monkeypatch):
    """A non-string model labels as ClassName(model_id) for the review header."""
    _patch_build(monkeypatch, _StubDeepAgent())

    class _FakeModel:
        model = "funhouse-gpt-high"

    out = eh.run_suite(_FakeModel(), questions=_tiny_suite()[:1], verbose=False)
    assert out["model"] == "_FakeModel(funhouse-gpt-high)"


# ===========================================================================
# (b) out= writes JSON + readable markdown review
# ===========================================================================

def test_run_suite_writes_json_and_markdown(monkeypatch, tmp_path):
    """out= writes a results JSON and a sibling .md review with Qs + answers."""
    _patch_build(monkeypatch, _StubDeepAgent())
    suite = _tiny_suite()
    out = tmp_path / "suite_results.json"
    result = eh.run_suite("fake-model", questions=suite, out=out, verbose=False)

    # JSON exists and carries the folded-in markdown review.
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n"] == 2
    assert "markdown_review" in data
    assert len(data["results"]) == 2

    # Markdown review exists and contains BOTH questions + their answers.
    md = tmp_path / "suite_results.json.md"
    assert md.exists()
    text = md.read_text(encoding="utf-8")
    assert suite[0]["question"] in text
    assert suite[1]["question"] in text
    assert "512" in text  # the answer text
    assert "not auto-scored" in text.lower()  # the honesty note
    assert "BC-1" in text and "ST-1" in text
    # The returned dict matches what was written.
    assert result["n"] == data["n"]


# ===========================================================================
# (c) render_suite_markdown on a synthetic suite_result
# ===========================================================================

def test_render_suite_markdown_structure():
    """render_suite_markdown lays out a metrics header + per-question blocks."""
    qa_good = eh.QAResult(
        qid="BC-1", module="bearing_capacity", agent="v2",
        answer="The ultimate bearing capacity is about 512 kPa.",
        trace=[eh.ToolCallRecord(
            "call_agent",
            {"agent_name": "bearing_capacity", "method": "bearing_capacity_analysis"},
            errored=False)],
        rounds=1, latency_s=2.0,
        usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )
    qa_err = eh.QAResult(
        qid="ST-1", module="settlement", agent="v2",
        answer="The settlement is 25 mm, fully reliable for these inputs here.",
        trace=[eh.ToolCallRecord("call_agent", {"agent_name": "settlement"},
                                 errored=True)],
        rounds=2, latency_s=3.0,
        errors=[{"tool": "call_agent", "message": "boom"}],
    )
    d0 = qa_good.to_dict(); d0["question"] = "Bearing capacity of a 2 m footing?"
    d1 = qa_err.to_dict(); d1["question"] = "Settlement under 100 kPa?"
    suite_result = {
        "model": "funhouse-gpt-high",
        "n": 2,
        "results": [d0, d1],
        "metrics": eh.score_run([qa_good, qa_err],
                                [{"id": "BC-1"}, {"id": "ST-1"}]),
    }

    md = eh.render_suite_markdown(suite_result)
    # Header.
    assert "# v2 (deepagents) suite review" in md
    assert "funhouse-gpt-high" in md
    assert "not auto-scored" in md.lower()
    assert "## Run metrics" in md
    assert "P1 hallucination-on-error rate" in md
    assert "Tool-error rate" in md
    # Per-question blocks: questions, answers, tools.
    assert "Bearing capacity of a 2 m footing?" in md
    assert "Settlement under 100 kPa?" in md
    assert "512 kPa" in md
    assert "25 mm" in md
    # call_agent rendered as agent.method; the errored call is flagged.
    assert "bearing_capacity.bearing_capacity_analysis" in md
    assert "[errored]" in md
    assert "_errors:" in md  # the error note line


def test_render_suite_markdown_handles_empty_and_exception():
    """An empty answer + a hard exception both render cleanly."""
    qa_exc = eh.QAResult(qid="X-1", module="m", agent="v2",
                         exception="RuntimeError: boom")
    d = qa_exc.to_dict(); d["question"] = "some question"
    suite_result = {
        "model": "m", "n": 1, "results": [d],
        "metrics": eh.score_run([qa_exc], [{"id": "X-1"}]),
    }
    md = eh.render_suite_markdown(suite_result)
    assert "_(no answer produced)_" in md
    assert "_exception: RuntimeError: boom_" in md
    assert "(none)" in md  # no tool calls in the trace


def test_write_suite_results_paths(tmp_path):
    """write_suite_results returns both written paths and folds markdown in."""
    qa = eh.QAResult(qid="BC-1", module="bearing_capacity", agent="v2",
                     answer="about 512 kPa", trace=[], rounds=0)
    d = qa.to_dict(); d["question"] = "q?"
    suite_result = {"model": "m", "n": 1, "results": [d],
                    "metrics": eh.score_run([qa], [{"id": "BC-1"}])}
    out = tmp_path / "r.json"
    paths = eh.write_suite_results(suite_result, out)
    assert paths["json"] == str(out)
    assert paths["markdown"] == str(tmp_path / "r.json.md")
    assert (tmp_path / "r.json").exists()
    assert (tmp_path / "r.json.md").exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "markdown_review" in data


# ===========================================================================
# (d) Public export wiring
# ===========================================================================

def test_run_suite_exported_from_deep_package():
    """`from funhouse_agent.deep import run_suite` works (and the eval_harness path)."""
    from funhouse_agent.deep import run_suite as run_suite_pkg
    from funhouse_agent.deep.eval_harness import run_suite as run_suite_eh
    assert run_suite_pkg is run_suite_eh
    from funhouse_agent.deep import render_suite_markdown, write_suite_results  # noqa: F401
