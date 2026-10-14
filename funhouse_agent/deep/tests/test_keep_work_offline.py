"""A ``task`` helper that stops hands back what it read (BACKLOG TA-21), and
at most N helpers run at once (TA-22).

Field session 2026-10-14: four helpers held extracted data when they stopped
(one 21 borings, 53,000 characters) and returned two apologies and two error
strings; five started within 12 ms on one key.

Offline: one scripted chat model plays the primary and its helpers (each
call is answered from the conversation it is shown), through a real
compiled deep agent.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

from funhouse_agent._fileio import working_dir_bound
from funhouse_agent.deep import keep_work
from funhouse_agent.deep.agent import build_deep_agent

ROW = "B-7 | 3.0 m | N=17 | SM, brown silty sand | w=12% \n"


def _first_human(messages) -> str:
    for m in messages:
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def _n_ai(messages) -> int:
    return sum(1 for m in messages if isinstance(m, AIMessage))


class Played(BaseChatModel):
    """Answers each call from the conversation it is shown: the primary
    (asked ``PRIMARY``) delegates ``briefs`` to helpers, then answers; a
    helper (asked ``HELPER <how>``) reads, then ends ``<how>``."""

    briefs: list = []
    reads: int = 2

    @property
    def _llm_type(self):
        return "played"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        first = _first_human(messages)
        step = _n_ai(messages)
        if first.startswith("PRIMARY"):
            if step == 0:
                reply = AIMessage(content="", tool_calls=[
                    {"name": "task", "id": f"task_{i}",
                     "args": {"description": b,
                              "subagent_type": "general-purpose"}}
                    for i, b in enumerate(self.briefs)])
            else:
                reply = AIMessage(content="All done.")
            return ChatResult(generations=[ChatGeneration(message=reply)])
        how = (first.split()[1].rstrip(":") if len(first.split()) > 1
               else "answer")
        if step < self.reads:
            reply = AIMessage(content=f"Reading page {30 + step}.",
                              tool_calls=[{"name": "big_read",
                                           "id": f"r{step}_{id(messages)}",
                                           "args": {"page": 30 + step}}])
        elif how == "error":
            raise RuntimeError("the connection to the model was reset")
        elif how == "apology":
            reply = AIMessage(content="Sorry, I could not finish the logs.")
        else:
            reply = AIMessage(content="FULL DATA:\n" + ROW * 80)
        return ChatResult(generations=[ChatGeneration(message=reply)])


def _big_read_tool(rows: int = 400, delay: float = 0.0, live=None):
    def big_read(page: int) -> str:
        """Read one log page as rows."""
        if live is not None:
            with live["lock"]:
                live["now"] += 1
                live["max"] = max(live["max"], live["now"])
        try:
            if delay:
                time.sleep(delay)
            return f"page {page}\n" + ROW * rows
        finally:
            if live is not None:
                with live["lock"]:
                    live["now"] -= 1
    return StructuredTool.from_function(big_read, name="big_read")


def _agent(briefs, reads=2, tool=None):
    model = Played(briefs=list(briefs), reads=reads)
    return build_deep_agent(model=model, extra_tools=[tool or _big_read_tool()],
                            reference_mode="off")


def _run(agent, folder):
    with working_dir_bound(str(folder)):
        return agent.invoke({"messages": [{"role": "user",
                                           "content": "PRIMARY go"}]},
                            config={"recursion_limit": 80})


def _task_results(state):
    return [m for m in state["messages"]
            if isinstance(m, ToolMessage) and m.name == "task"
            or (isinstance(m, ToolMessage)
                and str(m.tool_call_id).startswith("task_"))]


# ---------------------------------------------------------------------------
# TA-21: a helper that stops
# ---------------------------------------------------------------------------

def test_a_helper_that_errors_returns_a_file_and_a_summary(tmp_path):
    """Before: the primary got only the error string. Now: the error, the
    kept file's name and a summary made in code."""
    state = _run(_agent(["HELPER error: read the 2026 logs, pages 30-31"]),
                 tmp_path)
    (res,) = _task_results(state)
    data = json.loads(res.content)
    assert "the connection to the model was reset" in data["error"]
    assert data["kept"].startswith("kept/helper_")
    assert "2 tool calls" in data["summary"] and "big_read x2" in data["summary"]
    assert "Pages its calls named: 30, 31" in data["summary"]
    path = tmp_path / data["kept"]
    text = path.read_text(encoding="utf-8")
    # everything it read, in full, and what it was asked
    assert text.count(ROW) == 2 * 400
    assert "read the 2026 logs, pages 30-31" in text
    assert "stopped on an error" in text


def test_the_index_offsets_point_at_each_result(tmp_path):
    state = _run(_agent(["HELPER error: logs"]), tmp_path)
    data = json.loads(_task_results(state)[0].content)
    text = (tmp_path / data["kept"]).read_text(encoding="utf-8")
    import re
    offsets = [int(m.replace(",", "")) for m in
               re.findall(r"characters at offset ([\d,]+)", text)]
    assert len(offsets) == 2
    for off in offsets:
        assert text[off:].startswith("page 3"), text[off:off + 40]


def test_a_helper_that_gives_up_keeps_its_answer_and_a_file(tmp_path):
    """An apology after a lot of reading: the answer is passed on as it
    was, with the kept file named after it."""
    state = _run(_agent(["HELPER apology: read logs"]), tmp_path)
    (res,) = _task_results(state)
    assert res.content.startswith("Sorry, I could not finish the logs.")
    assert "[Kept: its answer (35 characters) is much shorter" in res.content
    rel = res.content.split("in the file '")[1].split("'")[0]
    assert (tmp_path / rel).is_file()


def test_a_helper_that_answers_in_full_keeps_no_file(tmp_path):
    state = _run(_agent(["HELPER answer: read logs"]), tmp_path)
    (res,) = _task_results(state)
    assert res.content.startswith("FULL DATA:")
    assert "[Kept:" not in res.content
    assert not (tmp_path / "kept").exists()


def test_a_helper_that_read_little_keeps_nothing_on_error(tmp_path):
    tool = _big_read_tool(rows=3)
    state = _run(_agent(["HELPER error: one page"], tool=tool), tmp_path)
    data = json.loads(_task_results(state)[0].content)
    assert "kept" not in data and "Nothing it read was worth keeping" in \
        data["hint"]


def test_the_general_purpose_prompt_asks_for_partial_data():
    from funhouse_agent.deep.agent import HELPER_PARTIAL_RULE
    assert "Never answer with only an apology" in HELPER_PARTIAL_RULE


def test_budget_and_short_answer_rules():
    rec = keep_work.HelperRecord("t1", "general-purpose", "brief")
    rec.add_result("big_read", "c1", "x" * 30_000)
    assert keep_work.gave_up(rec, "y" * 5_000) is None
    assert "much shorter" in keep_work.gave_up(rec, "sorry")
    rec.budget_reached = True
    assert "budget" in keep_work.gave_up(rec, "y" * 5_000)
    # a references consult answers briefly by design: never kept for that
    ref = keep_work.HelperRecord("t2", "references", "Kp?")
    ref.add_result("call_agent", "c1", "x" * 30_000)
    assert keep_work.gave_up(ref, "Kp = 5.3 (DM7.2 Fig 4-12)") is None


def test_a_budget_stop_marks_the_running_helper():
    from funhouse_agent.deep.limits import ModelCallBudgetMiddleware

    class Req:
        state = {"run_model_call_count": 2}

    rec = keep_work.HelperRecord("t1", "general-purpose", "b")
    token = keep_work._HELPER.set(rec)
    try:
        assert ModelCallBudgetMiddleware(3)._is_final_budgeted_call(Req())
    finally:
        keep_work._HELPER.reset(token)
    assert rec.budget_reached


# ---------------------------------------------------------------------------
# TA-22: helpers at once
# ---------------------------------------------------------------------------

def test_at_most_two_helpers_run_at_once(tmp_path, monkeypatch):
    monkeypatch.delenv(keep_work.MAX_HELPERS_ENV, raising=False)
    live = {"now": 0, "max": 0, "lock": threading.Lock()}
    tool = _big_read_tool(rows=1, delay=0.15, live=live)
    briefs = [f"HELPER answer: part {i}" for i in range(5)]
    state = _run(_agent(briefs, reads=1, tool=tool), tmp_path)
    assert len(_task_results(state)) == 5
    assert live["max"] <= keep_work.DEFAULT_MAX_HELPERS == 2


def test_the_cap_is_configurable(monkeypatch):
    monkeypatch.setenv(keep_work.MAX_HELPERS_ENV, "4")
    assert keep_work.max_helpers() == 4
    monkeypatch.setenv(keep_work.MAX_HELPERS_ENV, "0")
    assert keep_work.max_helpers() == 0
    monkeypatch.setenv(keep_work.MAX_HELPERS_ENV, "x")
    assert keep_work.max_helpers() == keep_work.DEFAULT_MAX_HELPERS


def test_places_are_served_first_come_and_freed_on_error():
    places = keep_work._HelperPlaces(1)
    order = []
    places.acquire()

    def later(i):
        places.acquire()
        order.append(i)
        places.release()

    threads = []
    for i in range(3):
        t = threading.Thread(target=later, args=(i,))
        t.start()
        threads.append(t)
        time.sleep(0.05)
    places.release()
    for t in threads:
        t.join(2)
    assert order == [0, 1, 2]
