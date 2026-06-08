"""Offline tests for accurate token tracking (NO API key, NO network, NO model).

The owner runs the v5.0 deepagents agent in Funhouse on GPT and watches token
spend (one chatbot conversation hit 800k tokens). Two root problems were fixed:

1. ``PrompterChatModel._generate`` put usage in ``generation_info["usage"]`` but
   never set ``AIMessage.usage_metadata`` (nor ``response_metadata["model_name"]``),
   so the standard LangChain aggregators saw nothing on the Funhouse/GPT path.
2. The eval's trace-summed ``_v2_usage`` missed sub-agent (references / reviewer)
   model calls; the run now aggregates EVERY model call via the
   ``get_usage_metadata_callback`` context manager (sub-agents included).

These tests verify, fully offline:

  (a) ``_to_usage_metadata`` maps an OpenAI-style usage object/dict (total-only,
      in+out, full, none) to the LangChain ``UsageMetadata`` shape — including the
      Funhouse total-only case where ``input_tokens = total`` so any in+out
      aggregator still yields the right TOTAL.
  (b) ``PrompterChatModel._generate`` sets ``usage_metadata`` on the returned
      AIMessage (driven by the same fake-prompter stub pattern as
      ``test_deep_phase2_offline.py``), and that usage flows through a real
      ``get_usage_metadata_callback``.
  (c) The eval token surfacing: ``score_run`` exposes ``max_total_tokens``,
      ``render_suite_markdown`` shows Total/Avg/Max token header fields + a
      per-question ``tokens:`` line, ``QAResult.to_dict`` exposes ``total_tokens``,
      and ``run_suite`` verbose prints a per-question ``+tok (cumulative ...)``
      tally. The per-question callback capture is exercised by monkeypatching
      ``get_usage_metadata_callback`` to a fake yielding a KNOWN total.

Run from the worktree root with the venv python::

    cd <worktree>
    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_deep_token_tracking_offline.py -v
"""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from funhouse_agent.deep import eval_harness as eh
from funhouse_agent.deep.databricks_bridge import (
    PrompterChatModel,
    _to_usage_metadata,
)


# ===========================================================================
# (a) _to_usage_metadata mapping
# ===========================================================================

def test_to_usage_metadata_full_split():
    """prompt+completion+total -> input/output/total verbatim."""
    um = _to_usage_metadata(
        SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    assert um == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_to_usage_metadata_split_without_total():
    """prompt+completion but no total -> total computed as in+out."""
    um = _to_usage_metadata(
        SimpleNamespace(prompt_tokens=12, completion_tokens=6)
    )
    assert um == {"input_tokens": 12, "output_tokens": 6, "total_tokens": 18}


def test_to_usage_metadata_total_only_funhouse_case():
    """ONLY a combined total (the Funhouse case): input=total, output=0, total=total.

    This keeps any aggregator that sums input+output yielding the right TOTAL —
    the owner cares about the total; the in/out split is best-effort.
    """
    um = _to_usage_metadata(SimpleNamespace(total_tokens=800_000))
    assert um == {"input_tokens": 800_000, "output_tokens": 0,
                  "total_tokens": 800_000}
    # Any in+out aggregator still recovers the right TOTAL.
    assert um["input_tokens"] + um["output_tokens"] == 800_000


def test_to_usage_metadata_dict_input():
    """A plain dict usage (not an object) maps the same way."""
    assert _to_usage_metadata({"total_tokens": 4242}) == {
        "input_tokens": 4242, "output_tokens": 0, "total_tokens": 4242,
    }
    assert _to_usage_metadata(
        {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}
    ) == {"input_tokens": 3, "output_tokens": 7, "total_tokens": 10}


def test_to_usage_metadata_none_and_empty():
    """No usage at all -> None (so the field is honestly absent)."""
    assert _to_usage_metadata(None) is None
    assert _to_usage_metadata(SimpleNamespace()) is None
    assert _to_usage_metadata({}) is None


# ===========================================================================
# (b) PrompterChatModel._generate sets usage_metadata (+ flows through callback)
# ===========================================================================

def _make_response(content, usage, finish_reason="stop", model="fake-gpt"):
    """A canned OpenAI-shaped response (SimpleNamespace) with a usage object."""
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage, model=model)


class _FakeCompletions:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakePrompter:
    """Fake PrompterAPI: only .client.chat.completions.create + .chat_model."""

    def __init__(self, response, chat_model="fake-gpt"):
        self.chat_model = chat_model
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions(response))
        )


def test_generate_sets_usage_metadata_total_only():
    """A response whose usage object carries ONLY total_tokens -> the AIMessage
    has usage_metadata with the correct total (Funhouse total-only path)."""
    resp = _make_response("hi", SimpleNamespace(total_tokens=800_000))
    model = PrompterChatModel(prompter=_FakePrompter(resp))
    result = model._generate([HumanMessage(content="hi")])
    ai = result.generations[0].message
    assert ai.usage_metadata == {"input_tokens": 800_000, "output_tokens": 0,
                                 "total_tokens": 800_000}
    # The legacy generation_info usage blob is still present.
    assert result.generations[0].generation_info["usage"]["total_tokens"] == 800_000
    # model_name is set so the usage callback will record this call.
    assert result.generations[0].generation_info["model_name"] == "fake-gpt"


def test_generate_sets_usage_metadata_split():
    """prompt+completion -> usage_metadata input/output/total."""
    resp = _make_response(
        "hi", SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                              total_tokens=15)
    )
    model = PrompterChatModel(prompter=_FakePrompter(resp))
    ai = model._generate([HumanMessage(content="hi")]).generations[0].message
    assert ai.usage_metadata == {"input_tokens": 10, "output_tokens": 5,
                                 "total_tokens": 15}


def test_generate_no_usage_leaves_metadata_unset():
    """A response with no usage object -> usage_metadata stays None."""
    resp = _make_response("hi", None)
    model = PrompterChatModel(prompter=_FakePrompter(resp))
    ai = model._generate([HumanMessage(content="hi")]).generations[0].message
    assert ai.usage_metadata is None


def test_prompter_usage_flows_through_real_callback():
    """The whole point: usage from PrompterChatModel is aggregated by the real
    get_usage_metadata_callback (which needs BOTH usage_metadata AND
    response_metadata['model_name'])."""
    resp = _make_response("hi", SimpleNamespace(total_tokens=12_345))
    model = PrompterChatModel(prompter=_FakePrompter(resp))
    with get_usage_metadata_callback() as cb:
        model.invoke([HumanMessage(content="hi")], config={"callbacks": [cb]})
    agg = dict(cb.usage_metadata)
    assert "fake-gpt" in agg
    assert agg["fake-gpt"]["total_tokens"] == 12_345


# ===========================================================================
# (c) Eval token surfacing
# ===========================================================================

def test_qaresult_to_dict_exposes_total_tokens():
    """QAResult.to_dict surfaces total_tokens (and the in/out split)."""
    qa = eh.QAResult(
        qid="BC-1", module="bearing_capacity", agent="v2",
        usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )
    d = qa.to_dict()
    assert d["total_tokens"] == 120
    assert d["input_tokens"] == 100
    assert d["output_tokens"] == 20
    # No usage -> all None.
    bare = eh.QAResult(qid="X", module="m", agent="v2").to_dict()
    assert bare["total_tokens"] is None
    assert bare["input_tokens"] is None


def test_score_run_reports_max_total_tokens():
    """score_run adds max_total_tokens (the single most expensive question)."""
    cheap = eh.QAResult(qid="A", module="m", agent="v2",
                        usage={"total_tokens": 5_000})
    pricey = eh.QAResult(qid="B", module="m", agent="v2",
                         usage={"total_tokens": 90_000})
    m = eh.score_run([cheap, pricey], [{"id": "A"}, {"id": "B"}])
    assert m["total_tokens"] == 95_000
    assert m["max_total_tokens"] == 90_000
    assert m["avg_tokens"] == pytest.approx(47_500)


def test_score_run_no_usage_tokens_none():
    """No usage anywhere -> token metrics are None, not a misleading zero."""
    qa = eh.QAResult(qid="A", module="m", agent="v2")
    m = eh.score_run([qa], [{"id": "A"}])
    assert m["total_tokens"] is None
    assert m["max_total_tokens"] is None
    assert m["avg_tokens"] is None


def test_render_suite_markdown_shows_token_fields_and_per_question_line():
    """The review markdown shows Total/Avg/Max token header fields and a
    per-question tokens: line (with the in/out split when both are nonzero)."""
    qa_split = eh.QAResult(
        qid="BC-1", module="bearing_capacity", agent="v2",
        answer="The ultimate bearing capacity is about 512 kPa.",
        trace=[eh.ToolCallRecord("call_agent",
                                 {"agent_name": "bearing_capacity"})],
        usage={"input_tokens": 12_000, "output_tokens": 6_432,
               "total_tokens": 18_432},
    )
    qa_total_only = eh.QAResult(
        qid="ST-1", module="settlement", agent="v2",
        answer="Settlement is about 25 mm.",
        trace=[eh.ToolCallRecord("call_agent", {"agent_name": "settlement"})],
        usage={"input_tokens": 90_000, "output_tokens": 0,
               "total_tokens": 90_000},
    )
    d0 = qa_split.to_dict(); d0["question"] = "Bearing capacity?"
    d1 = qa_total_only.to_dict(); d1["question"] = "Settlement?"
    suite_result = {
        "model": "funhouse-gpt-high", "n": 2, "results": [d0, d1],
        "metrics": eh.score_run([qa_split, qa_total_only],
                                [{"id": "BC-1"}, {"id": "ST-1"}]),
    }
    md = eh.render_suite_markdown(suite_result)

    # Header token fields, formatted with thousands separators.
    assert "Total tokens: 108,432" in md
    assert "Avg tokens / question: 54,216" in md
    assert "Max tokens (one question): 90,000" in md
    # Per-question token line: split shown only when both in/out are nonzero.
    assert "_tokens: 18,432 (12,000 in / 6,432 out)_" in md
    assert "_tokens: 90,000_" in md  # total-only -> no split shown


def test_render_suite_markdown_tokens_na_when_absent():
    """A question with no usage shows tokens: n/a (header too)."""
    qa = eh.QAResult(qid="X", module="m", agent="v2", answer="ok")
    d = qa.to_dict(); d["question"] = "q?"
    suite_result = {"model": "m", "n": 1, "results": [d],
                    "metrics": eh.score_run([qa], [{"id": "X"}])}
    md = eh.render_suite_markdown(suite_result)
    assert "_tokens: n/a_" in md
    assert "Total tokens: n/a" in md
    assert "Max tokens (one question): n/a" in md


# ----- callback capture in run_suite (monkeypatched callback) --------------

class _StubDeepAgent:
    """Canned v2 agent: returns a deepagents-style message list. Its messages
    carry usage_metadata too (so the trace-fallback path also has tokens)."""

    def invoke(self, payload, *args, **kwargs):
        question = ""
        for m in payload.get("messages", []):
            if isinstance(m, dict) and m.get("role") == "user":
                question = m.get("content", "")
        call = AIMessage(
            content="",
            tool_calls=[{"name": "call_agent",
                         "args": {"agent_name": "bearing_capacity"},
                         "id": "c1"}],
            usage_metadata={"input_tokens": 1, "output_tokens": 1,
                            "total_tokens": 2},
        )
        result = ToolMessage(content='{"q_ultimate_kPa": 512.3}',
                             tool_call_id="c1", name="call_agent")
        final = AIMessage(content=f"Answer: {question} -> 512 kPa.",
                          usage_metadata={"input_tokens": 1, "output_tokens": 1,
                                          "total_tokens": 2})
        return {"messages": [call, result, final]}


@contextmanager
def _fake_callback_cm(total_per_call):
    """A get_usage_metadata_callback stand-in yielding a KNOWN per-call total.

    Each `with` block yields a fresh handler whose ``usage_metadata`` reports one
    model with ``total_tokens = total_per_call`` — simulating the real callback
    aggregating a run (sub-agents included) to a known value, with no model.
    """
    cb = SimpleNamespace(usage_metadata={
        "funhouse-gpt": {"input_tokens": 0, "output_tokens": 0,
                         "total_tokens": total_per_call},
    })
    yield cb


def _patch_build(monkeypatch, agent):
    import funhouse_agent.deep.agent as agent_mod
    monkeypatch.setattr(agent_mod, "build_deep_agent", lambda *a, **k: agent)


def _patch_callback(monkeypatch, total_per_call):
    """Monkeypatch the callback get_usage_metadata_callback used inside _run_one."""
    monkeypatch.setattr(
        "langchain_core.callbacks.get_usage_metadata_callback",
        lambda *a, **k: _fake_callback_cm(total_per_call),
    )


def test_run_suite_uses_callback_total_over_trace(monkeypatch):
    """run_suite prefers the callback total (sub-agents included) over the
    trace-summed usage. The fake callback reports 50,000 per question, which
    must override the stub's trace total of 4."""
    _patch_build(monkeypatch, _StubDeepAgent())
    _patch_callback(monkeypatch, 50_000)
    suite = [
        {"id": "BC-1", "module": "bearing_capacity", "question": "q1"},
        {"id": "ST-1", "module": "settlement", "question": "q2"},
    ]
    out = eh.run_suite("fake-model", questions=suite, verbose=False)
    # Each question's authoritative total is the callback's 50,000, not 4.
    assert out["results"][0]["total_tokens"] == 50_000
    assert out["results"][1]["total_tokens"] == 50_000
    m = out["metrics"]
    assert m["total_tokens"] == 100_000
    assert m["max_total_tokens"] == 50_000
    assert m["avg_tokens"] == pytest.approx(50_000)


def test_run_suite_falls_back_to_trace_when_callback_empty(monkeypatch):
    """If the callback records nothing (e.g. a stub model), the trace-summed
    usage is kept — nothing regresses."""
    _patch_build(monkeypatch, _StubDeepAgent())
    _patch_callback(monkeypatch, 0)  # callback yields a zero/empty total
    suite = [{"id": "BC-1", "module": "bearing_capacity", "question": "q1"}]
    out = eh.run_suite("fake-model", questions=suite, verbose=False)
    # Falls back to the trace-summed usage (2 + 2 from the two AIMessages).
    assert out["results"][0]["total_tokens"] == 4


def test_run_suite_verbose_prints_cumulative_token_tally(monkeypatch, capsys):
    """run_suite verbose prints a per-question +tok (cumulative ...) line."""
    _patch_build(monkeypatch, _StubDeepAgent())
    _patch_callback(monkeypatch, 18_432)
    suite = [
        {"id": "BC-1", "module": "bearing_capacity", "question": "q1"},
        {"id": "DS-2", "module": "drilled_shaft", "question": "q2"},
    ]
    eh.run_suite("fake-model", questions=suite, verbose=True)
    printed = capsys.readouterr().out
    # Per-question line: id, module, this-question tokens, running cumulative.
    assert "+18,432 tok" in printed
    assert "(cumulative 18,432)" in printed
    assert "(cumulative 36,864)" in printed
    assert "DS-2 (drilled_shaft)" in printed


# ===========================================================================
# (d) Chatbot (DeepNotebookChat) running token total
# ===========================================================================

def test_notebook_format_token_line():
    from funhouse_agent.deep.notebook import _format_token_line
    line = _format_token_line(18_432, 142_907)
    assert line == "tokens this turn: 18,432 | conversation total: 142,907"


def test_notebook_sum_callback_tokens():
    from funhouse_agent.deep.notebook import _sum_callback_tokens
    # Two models (e.g. primary + a differently-named sub-agent model) summed.
    agg = {
        "gpt-a": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        "gpt-b": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10},
    }
    assert _sum_callback_tokens(agg) == 130
    # total_tokens missing -> falls back to input+output.
    assert _sum_callback_tokens({"m": {"input_tokens": 3, "output_tokens": 4}}) == 7
    # Nothing recorded -> 0 (a turn with no usage adds nothing).
    assert _sum_callback_tokens({}) == 0
    assert _sum_callback_tokens(None) == 0


class _StubStreamAgent:
    """A compiled-deep-agent stand-in: a .stream() yielding canned token chunks
    and recording the config it was handed (to confirm a callback is attached)."""

    def __init__(self, tokens=("Hello ", "world")):
        from langchain_core.messages import AIMessageChunk
        self._items = [
            ("messages", (AIMessageChunk(content=t), {"langgraph_node": "model"}))
            for t in tokens
        ]
        self.last_config = None

    def stream(self, inp, config=None, *, stream_mode=None):
        self.last_config = config
        yield from self._items


def test_notebook_ask_and_print_emits_token_line_and_tracks_total():
    """ask_and_print prints the running-token line and updates the cumulative
    total. The stub does not call a real model, so the callback aggregates 0 —
    proving the wiring is graceful when no usage is reported (total stays 0)."""
    from funhouse_agent.deep.notebook import DeepNotebookChat
    import io

    chat = DeepNotebookChat(_StubStreamAgent())
    buf = io.StringIO()
    final = chat.ask_and_print("hi", file=buf)
    assert final == "Hello world"
    printed = buf.getvalue()
    # The token line is emitted under the answer.
    assert "tokens this turn:" in printed
    assert "conversation total:" in printed
    # A usage-metadata callback was attached to the stream config.
    assert len(chat._agent.last_config["callbacks"]) == 1
    assert chat._agent.last_config["configurable"]["thread_id"]
    # No real model -> 0 tokens; the running total is exposed and is 0.
    assert chat.total_tokens == 0


def test_notebook_running_total_accumulates_across_turns(monkeypatch):
    """With the callback monkeypatched to a known per-turn total, the running
    conversation total accumulates across turns and resets to 0."""
    from funhouse_agent.deep.notebook import DeepNotebookChat
    import io

    # Patch the callback the notebook imports lazily inside _run_stream.
    monkeypatch.setattr(
        "langchain_core.callbacks.get_usage_metadata_callback",
        lambda *a, **k: _fake_callback_cm(10_000),
    )
    chat = DeepNotebookChat(_StubStreamAgent())
    chat.ask_and_print("turn 1", file=io.StringIO())
    assert chat.total_tokens == 10_000
    chat.ask_and_print("turn 2", file=io.StringIO())
    assert chat.total_tokens == 20_000
    # reset() zeroes the running total.
    chat.reset()
    assert chat.total_tokens == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
