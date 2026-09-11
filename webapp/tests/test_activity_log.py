"""Always-on activity log (owner feedback 2026-09-11): every tool call, tool
result and model call — primary AND sub-agents — lands in
``<conversation>/activity.jsonl`` regardless of the "Show turn details" box.

Three layers:
1. the handler driven with synthetic callback sequences (attribution by
   ``task`` nesting, truncation, never raises);
2. the REAL stack — ``build_deep_agent`` with a scripted model that delegates
   to the ``calc`` sub-agent — proving callbacks propagate into sub-agents on
   the installed deepagents/langgraph and that the sub-agent's own tool calls
   are attributed to ``calc``;
3. ``turn_jobs`` writes the turn envelope with tracing OFF.
"""

import json
import os
import uuid

import pytest

pytest.importorskip("langchain_core")

from webapp import activity_log as al


def _u():
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# 1. synthetic
# ---------------------------------------------------------------------------

def test_nested_task_attributes_inner_calls_to_subagent(tmp_path):
    log = al.ActivityLogger(str(tmp_path), turn=2)
    task_id, inner_id, model_id = _u(), _u(), _u()
    log.on_tool_start({"name": "task"}, "", run_id=task_id,
                      inputs={"description": "run bearing",
                              "subagent_type": "calc"})
    assert log.agent == "calc"
    log.on_chat_model_start({"name": "ChatFake"}, [[]], run_id=model_id,
                            parent_run_id=task_id)
    log.on_tool_start({"name": "call_agent"}, "", run_id=inner_id,
                      parent_run_id=task_id,
                      inputs={"agent": "bearing_capacity",
                              "params": {"B": 2.0}})
    log.on_tool_end("q_ult = 1159 kPa", run_id=inner_id, parent_run_id=task_id)
    log.on_tool_end("calc done: 1159 kPa", run_id=task_id)
    assert log.agent == "primary"

    recs = al.load(str(tmp_path))
    ev = [(r["event"], r["agent"], r.get("name")) for r in recs]
    assert ev == [
        ("tool_start", "primary", "task"),
        ("model_start", "calc", None),
        ("tool_start", "calc", "call_agent"),
        ("tool_end", "calc", "call_agent"),
        ("tool_end", "primary", "task"),
    ]
    assert recs[0]["subagent"] == "calc"
    assert recs[2]["args"] == {"agent": "bearing_capacity",
                               "params": {"B": 2.0}}
    assert recs[3]["result"] == "q_ult = 1159 kPa"
    assert recs[3]["truncated"] is False and recs[3]["chars"] == 16
    assert recs[4]["subagent"] == "calc"
    assert all(r["turn"] == 2 for r in recs)
    assert log.records_written == 5 and log.last_error is None


def test_result_truncation_is_marked(tmp_path):
    log = al.ActivityLogger(str(tmp_path), max_chars=100)
    rid = _u()
    log.on_tool_start({"name": "call_agent"}, "", run_id=rid, inputs={})
    log.on_tool_end("x" * 5000, run_id=rid)
    rec = al.load(str(tmp_path))[-1]
    assert rec["truncated"] is True and rec["chars"] == 5000
    assert rec["result"].startswith("x" * 100)
    assert "truncated: 5000 chars total" in rec["result"]


def test_tool_error_pops_task_and_records_error(tmp_path):
    log = al.ActivityLogger(str(tmp_path))
    tid = _u()
    log.on_tool_start({"name": "task"}, "", run_id=tid,
                      inputs={"subagent_type": "references"})
    log.on_tool_error(RuntimeError("boom"), run_id=tid)
    assert log.agent == "primary"
    rec = al.load(str(tmp_path))[-1]
    assert rec["event"] == "tool_error" and rec["error"] == "RuntimeError: boom"
    assert rec["agent"] == "primary" and rec["subagent"] == "references"


def test_model_end_usage_and_tool_call_count(tmp_path):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult
    log = al.ActivityLogger(str(tmp_path))
    rid = _u()
    log.on_chat_model_start({"kwargs": {"model": "gpt-x"}}, [[]], run_id=rid)
    msg = AIMessage(content="", tool_calls=[
        {"name": "call_agent", "args": {}, "id": "c1"}],
        usage_metadata={"input_tokens": 120, "output_tokens": 30,
                        "total_tokens": 150})
    log.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]),
                   run_id=rid)
    recs = al.load(str(tmp_path))
    assert recs[0]["event"] == "model_start" and recs[0]["model"] == "gpt-x"
    assert recs[1]["event"] == "model_end"
    assert recs[1]["usage"] == {"input_tokens": 120, "output_tokens": 30,
                                "total_tokens": 150}
    assert recs[1]["n_tool_calls"] == 1
    assert recs[1]["duration_s"] is not None


def test_never_raises_when_unwritable(tmp_path):
    blocker = tmp_path / "conv"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    log = al.ActivityLogger(str(blocker))
    rid = _u()
    log.turn_start("hi")
    log.on_tool_start({"name": "t"}, "", run_id=rid, inputs={})
    log.on_tool_end("ok", run_id=rid)
    log.turn_end()
    assert log.records_written == 0
    assert log.last_error


def test_weird_payloads_are_serialised(tmp_path):
    log = al.ActivityLogger(str(tmp_path))
    rid = _u()
    log.on_tool_start(None, "raw input", run_id=rid, name="named_via_kw")
    log.on_tool_end({"content": [{"type": "text", "text": "part one"},
                                 "part two"]}, run_id=rid)
    recs = al.load(str(tmp_path))
    assert recs[0]["name"] == "named_via_kw" and recs[0]["args"] == "raw input"
    assert recs[1]["name"] == "named_via_kw"
    assert "part one" in recs[1]["result"]


def test_load_filters_by_turn(tmp_path):
    al.ActivityLogger(str(tmp_path), turn=1).turn_start("a")
    al.ActivityLogger(str(tmp_path), turn=2).turn_start("b")
    assert [r["prompt"] for r in al.load(str(tmp_path), turn=2)] == ["b"]
    assert len(al.load(str(tmp_path))) == 2
    assert al.load(str(tmp_path / "missing")) == []


# ---------------------------------------------------------------------------
# 2. REAL stack: build_deep_agent + scripted model delegating to `calc`
# ---------------------------------------------------------------------------

def _scripted_model():
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import Field

    class Scripted(BaseChatModel):
        """Primary: delegate once via `task`, then answer. Calc sub-agent
        (recognised by its first human message being the delegation text):
        call `list_agents` once, then answer."""
        seen: list = Field(default_factory=list)

        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

        @property
        def _llm_type(self):
            return "scripted"

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.seen.append(list(messages))
            humans = [m for m in messages if isinstance(m, HumanMessage)]
            first = str(humans[0].content) if humans else ""
            is_calc = first.startswith("DELEGATED:")
            n_tool_msgs = sum(1 for m in messages
                              if getattr(m, "type", "") == "tool")
            if is_calc:
                if n_tool_msgs == 0:
                    msg = AIMessage(content="", tool_calls=[{
                        "name": "list_agents", "args": {}, "id": "c_inner"}])
                else:
                    msg = AIMessage(content="calc result: q_ult = 1159 kPa")
            else:
                if n_tool_msgs == 0:
                    msg = AIMessage(content="", tool_calls=[{
                        "name": "task",
                        "args": {"description": "DELEGATED: run bearing",
                                 "subagent_type": "calc"},
                        "id": "c_task"}])
                else:
                    msg = AIMessage(content="Final: 1159 kPa.")
            return ChatResult(generations=[ChatGeneration(message=msg)])

    return Scripted()


def test_real_stack_subagent_calls_are_logged_and_attributed(tmp_path):
    pytest.importorskip("deepagents")
    from funhouse_agent.deep.agent import build_deep_agent
    agent = build_deep_agent(_scripted_model(), enable_calc_subagent=True,
                             reference_mode="off")
    log = al.ActivityLogger(str(tmp_path), turn=1)
    log.turn_start("bearing capacity?")
    out = agent.invoke(
        {"messages": [{"role": "user", "content": "bearing capacity?"}]},
        config={"callbacks": [log],
                "configurable": {"thread_id": "act-1"}})
    log.turn_end(answer_chars=len(str(out["messages"][-1].content)))
    recs = al.load(str(tmp_path))
    assert log.last_error is None, log.last_error
    tool_recs = [r for r in recs if r["event"] in ("tool_start", "tool_end")]
    by_name = {(r["event"], r["name"]): r for r in tool_recs}
    # the primary's delegation ...
    assert by_name[("tool_start", "task")]["agent"] == "primary"
    assert by_name[("tool_start", "task")]["subagent"] == "calc"
    assert by_name[("tool_start", "task")]["args"]["subagent_type"] == "calc"
    # ... and the sub-agent's OWN tool call, attributed to calc, in full
    assert by_name[("tool_start", "list_agents")]["agent"] == "calc"
    assert by_name[("tool_end", "list_agents")]["agent"] == "calc"
    assert "bearing_capacity" in by_name[("tool_end", "list_agents")]["result"]
    # the task's result comes back to the primary
    assert by_name[("tool_end", "task")]["agent"] == "primary"
    assert "1159" in by_name[("tool_end", "task")]["result"]
    # model calls on both sides
    models = [r["agent"] for r in recs if r["event"] == "model_end"]
    assert "primary" in models and "calc" in models
    assert recs[0]["event"] == "turn_start" and recs[-1]["event"] == "turn_end"


# ---------------------------------------------------------------------------
# 3. turn_jobs writes the envelope with tracing OFF
# ---------------------------------------------------------------------------

def test_turn_job_writes_activity_with_trace_off(monkeypatch, tmp_path):
    import time
    import webapp.core as core
    import webapp.turn_jobs as tj
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    tj._JOBS.clear()
    tid = "ACT1"
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    seen = {}

    def fake_stream(agent, messages, thread_id, recursion_limit=None,
                    callbacks=None):
        seen["callbacks"] = callbacks
        yield {"kind": "token", "text": "hi"}
        yield {"kind": "turn_done", "answer": "hi", "turn_tokens": 7}

    monkeypatch.setattr(core, "stream_turn", fake_stream)
    transcript = [{"role": "user", "text": "q1"}]
    ctx = {"prompt": "q1", "temp_dir": files,
           "before": core.snapshot_dir(files), "staged_inputs": set(),
           "working_dir": files, "before_wd": None, "artifacts": [],
           "artifacts_before_len": 0, "transcript": transcript,
           "trace_on": False, "model": "m", "behavior": core.default_behavior()}
    job = tj.start_turn_job(object(), [], tid, None, ctx)
    t0 = time.time()
    while not job.done and time.time() - t0 < 10:
        time.sleep(0.02)
    assert job.done
    assert isinstance(seen["callbacks"][0], al.ActivityLogger)
    recs = al.load(core.conversation_dir(tid))
    assert [r["event"] for r in recs] == ["turn_start", "turn_end"]
    assert recs[0]["turn"] == 1 and recs[0]["prompt"] == "q1"
    assert recs[1]["turn_tokens"] == 7 and recs[1]["error"] is None
    # trace.jsonl still gated by the toggle
    assert not os.path.isfile(core.trace_path(tid))
