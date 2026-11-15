"""To-do updates beside the next call, behind a setting (BACKLOG DR-43 part 2).

``GEOTECH_REVIEW_TODOS_WITH_CALLS`` (off unless set) adds one sentence to the
page's rule on the to-do list and to the reading helper's prompt. Unset, both
prompts are what they were. The suite's runner counts the steps whose only
tool call is ``write_todos`` and their share of the input tokens.
"""

import json

import pytest

pytest.importorskip("planlens.tools")

from langchain_core.messages import AIMessage  # noqa: E402

from aec_agent_core import activity_log, review_flags  # noqa: E402
from aec_agent_core.review_eval.runner import (  # noqa: E402
    record_counts, summarize, todo_only_steps)
from aec_document_review import review_prompt as P  # noqa: E402
from aec_document_review.review_build import build_review_page_agent  # noqa: E402
from tests.test_review_harness_offline import (  # noqa: E402
    FakeEngine, _model, _review_kwargs)
from planlens.testing import build_synthetic_review_document  # noqa: E402

# The page's rule as master wrote it (6e2b94c): the bullet that ends the
# list of ways of working.
MASTER_TODO_BULLET = (
    "- **Plan multi-step work with `write_todos`** — a full-set review, a\n"
    "  specification-versus-submittal check, a comment-response round — and keep\n"
    "  it updated. Skip it for a single question.")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


@pytest.fixture
def on(monkeypatch):
    for k, v in review_flags.ARMS["todos_with_calls"].items():
        monkeypatch.setenv(k, v)


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

def test_the_switch_is_declared_like_the_others(monkeypatch):
    assert review_flags.TODOS_WITH_CALLS_ENV == "GEOTECH_REVIEW_TODOS_WITH_CALLS"
    assert review_flags.TODOS_WITH_CALLS_ENV in review_flags.ALL_ENVS
    assert review_flags.TODOS_WITH_CALLS_ENV not in review_flags.DEFAULT_ON_ENVS
    assert "todos_with_calls" in review_flags.__all__
    assert "TODOS_WITH_CALLS_ENV" in review_flags.__all__
    assert review_flags.ARMS["todos_with_calls"] == {
        review_flags.TODOS_WITH_CALLS_ENV: "1"}
    assert "GEOTECH_REVIEW_TODOS_WITH_CALLS" in review_flags.__doc__
    assert review_flags.todos_with_calls() is False
    for value in ("", "0", "off"):
        monkeypatch.setenv(review_flags.TODOS_WITH_CALLS_ENV, value)
        assert review_flags.todos_with_calls() is False
    monkeypatch.setenv(review_flags.TODOS_WITH_CALLS_ENV, "1")
    assert review_flags.todos_with_calls() is True


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------

def test_unset_the_prompts_are_byte_identical_to_masters():
    assert P.build_document_review_prompt() == P.DOCUMENT_REVIEW_PROMPT
    assert MASTER_TODO_BULLET in P.DOCUMENT_REVIEW_PROMPT
    assert P.build_document_review_reader_prompt() == \
        P.DOCUMENT_REVIEW_READER_PROMPT
    assert "same step" not in P.DOCUMENT_REVIEW_PROMPT
    assert "same step" not in P.DOCUMENT_REVIEW_READER_PROMPT


def test_set_the_rule_says_to_update_the_list_beside_the_next_call(on):
    prompt = P.build_document_review_prompt()
    flat = " ".join(prompt.split())
    assert P.TODOS_WITH_CALLS_RULE in flat
    assert "same step as your next tool call" in flat
    assert "sends the whole conversation again" in flat
    # the rest of today's rule stays, and the sentence sits inside it
    assert "Plan multi-step work with `write_todos`" in prompt
    assert "Skip it for a single question. Update the list" in flat
    # everything else in the prompt is untouched
    assert prompt.replace(" " + P.TODOS_WITH_CALLS_RULE, "") == \
        P.DOCUMENT_REVIEW_PROMPT
    # with the memory note too
    with_memory = P.build_document_review_prompt(memory_enabled=True)
    assert P.TODOS_WITH_CALLS_RULE in " ".join(with_memory.split())


def test_set_the_reading_helper_gets_the_line_unset_it_does_not(on):
    reader = P.build_document_review_reader_prompt()
    assert reader.startswith(P.DOCUMENT_REVIEW_READER_PROMPT)
    assert P.TODOS_WITH_CALLS_RULE in " ".join(reader.split())


def test_the_helper_spec_carries_the_prompt_the_switch_decides(monkeypatch):
    from aec_document_review import review_build
    spec_of = review_build._review_reader_subagent
    import inspect
    assert "build_document_review_reader_prompt" in inspect.getsource(spec_of)


def test_the_built_page_agent_is_sent_the_rule_only_when_set(monkeypatch):
    gt = build_synthetic_review_document()
    seen = {}
    for label in ("off", "on"):
        if label == "on":
            monkeypatch.setenv(review_flags.TODOS_WITH_CALLS_ENV, "1")
        model = _model([AIMessage(content="done")])
        agent = build_review_page_agent(
            model, attachments={"review_set.pdf": gt.pdf},
            engine=FakeEngine(), **_review_kwargs())
        agent.invoke({"messages": [{"role": "user", "content": "Hi"}]},
                     config={"recursion_limit": 50})
        first = model.seen[0][0].content
        seen[label] = first if isinstance(first, str) else "".join(
            b.get("text", "") for b in first if isinstance(b, dict))
    assert P.TODOS_WITH_CALLS_RULE not in " ".join(seen["off"].split())
    assert P.TODOS_WITH_CALLS_RULE in " ".join(seen["on"].split())


# ---------------------------------------------------------------------------
# A step with write_todos beside another call runs and is recorded
# ---------------------------------------------------------------------------

def test_a_step_with_write_todos_beside_another_call_runs_and_is_recorded(
        on, tmp_path):
    gt = build_synthetic_review_document()
    engine = FakeEngine("a letter with a stamp")
    both = AIMessage(content="", tool_calls=[
        {"name": "write_todos", "id": "call_todos_1", "args": {"todos": [
            {"content": "Read page 1", "status": "in_progress"},
            {"content": "Answer", "status": "pending"}]}},
        {"name": "analyze_pdf_page", "id": "call_look_1",
         "args": {"attachment_key": "review_set.pdf", "page": 0,
                  "tiles": "off"}}])
    model = _model([both, AIMessage(content="Page 1 is a letter.")])
    agent = build_review_page_agent(
        model, attachments={"review_set.pdf": gt.pdf}, engine=engine,
        **_review_kwargs())
    log = activity_log.ActivityLogger(str(tmp_path), turn=1)
    out = agent.invoke({"messages": [{"role": "user", "content": "Page 1?"}]},
                       config={"recursion_limit": 100, "callbacks": [log]})
    assert out["messages"][-1].content == "Page 1 is a letter."
    assert len(engine.prompts) == 1                 # the look ran
    assert out.get("todos")                         # the list was updated
    records = activity_log.load(str(tmp_path))
    started = [r["name"] for r in records if r["event"] == "tool_start"]
    assert "write_todos" in started and "analyze_pdf_page" in started
    ends = [r for r in records if r["event"] == "tool_end"]
    assert {r["name"] for r in ends} >= {"write_todos", "analyze_pdf_page"}
    first_step = [r for r in records if r["event"] == "model_end"][0]
    assert first_step["n_tool_calls"] == 2
    # a step that holds another call is not a to-do-only step
    assert todo_only_steps(records)["todo_only_steps"] == 0


# ---------------------------------------------------------------------------
# The runner's count
# ---------------------------------------------------------------------------

def _end(agent, n, tokens):
    return {"agent": agent, "event": "model_end", "n_tool_calls": n,
            "usage": {"input_tokens": tokens, "output_tokens": 10}}


def _tool(agent, name):
    return {"agent": agent, "event": "tool_start", "name": name, "args": {}}


MADE_UP = [
    _end("primary", 1, 1000), _tool("primary", "write_todos"),     # only
    _end("primary", 2, 2000), _tool("primary", "write_todos"),     # beside
    _tool("primary", "read_document"),
    _end("primary", 1, 3000), _tool("primary", "task"),            # helper
    _end("reader", 1, 500), _tool("reader", "write_todos"),        # helper's
    _end("reader", 0, 700),
    {"agent": "primary", "event": "model_end", "n_tool_calls": 0,   # a side
     "vision": {"kind": "page"}, "usage": {"input_tokens": 9999}},  # call
    _end("primary", 1, 4000), _tool("primary", "write_todos"),     # only
    _end("primary", 0, 5000),
]


def test_the_runner_counts_todo_only_steps_and_their_input_share():
    got = todo_only_steps(MADE_UP)
    assert got["todo_only_steps"] == 3
    assert got["todo_only_input_tokens"] == 1000 + 500 + 4000
    assert got["model_input_tokens"] == (1000 + 2000 + 3000 + 500 + 700
                                         + 4000 + 5000)
    assert got["todo_only_input_share"] == round(5500 / 16200, 4)
    assert record_counts(MADE_UP)["todo_only_steps"] == 3


def test_a_record_with_no_tool_names_or_usage_counts_nothing():
    assert todo_only_steps([])["todo_only_steps"] == 0
    assert todo_only_steps([])["todo_only_input_share"] is None
    bare = [{"agent": "a", "event": "model_end", "n_tool_calls": 1},
            _tool("a", "write_todos")]
    got = todo_only_steps(bare)
    assert got["todo_only_steps"] == 1 and got["todo_only_input_share"] is None


def _arm_row(md, arm):
    rows = [ln for ln in md.splitlines() if ln.startswith("|")]
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    for ln in rows[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if cells[0] == arm:
            return dict(zip(header, cells))
    raise AssertionError(arm)


def test_results_md_and_results_json_carry_the_count_for_every_arm():
    from aec_document_review.review_eval.tasks import Task
    task = Task.__new__(Task)
    task.category = task.doc_type = task.page = "x"
    task.id = "a"
    base = {"score": {"passed": True, "checks_passed": 1, "checks_total": 1},
            "model_calls": 3, "tool_calls": [], "tokens_in": 10,
            "tokens_out": 1, "seconds": 1}
    runs = {"baseline": {"a": dict(base, **todo_only_steps(MADE_UP))},
            "todos_with_calls": {"a": dict(base, todo_only_steps=0,
                                           todo_only_input_tokens=0,
                                           model_input_tokens=800)},
            "old": {"a": dict(base)}}
    md = summarize(runs, ["baseline", "todos_with_calls", "old"], [],
                   {"updated": "now", "model": "m"})
    assert _arm_row(md, "baseline")["to-do-only steps"] == "3"
    assert _arm_row(md, "baseline")["their share of input tokens"] == "34.0 %"
    assert _arm_row(md, "todos_with_calls")["to-do-only steps"] == "0"
    assert _arm_row(md, "todos_with_calls")[
        "their share of input tokens"] == "0.0 %"
    # a run saved before the count existed is not zero
    assert _arm_row(md, "old")["to-do-only steps"] == "·"
    assert _arm_row(md, "old")["their share of input tokens"] == "·"
    # the per-run keys are what results.json holds
    assert json.loads(json.dumps(runs["baseline"]["a"]))[
        "todo_only_input_share"] == round(5500 / 16200, 4)
