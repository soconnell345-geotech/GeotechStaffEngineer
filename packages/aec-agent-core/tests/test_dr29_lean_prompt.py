"""BACKLOG DR-29: the agent library's generic prompt off a page, behind a
switch (``GEOTECH_REVIEW_LEAN_PROMPT``, off unless set).

:class:`aec_agent_core.deep.lean_prompt.LeanFrameworkPrompt` removes the
library's base prompt word for word and swaps its to-do and helper sections
and the ``write_todos`` and ``task`` tool descriptions for a page's short
texts, all or nothing: where any of the library's texts is not found word
for word, the call goes out exactly as it came and the turn's record says
which. These tests hand the middleware made-up library texts, so they hold
on any installed version of the library; the page's own build is tested in
Document Review (``test_dr29_lean_prompt_page.py``).
"""

from __future__ import annotations

import pytest

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel)
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool

from aec_agent_core import activity_log, review_flags
from aec_agent_core.deep import keep_work
from aec_agent_core.deep import lean_prompt as L

# Made-up "library" texts, shaped like the library's.
BASE = "You are a made-up agent.\n\n## Core\n\n- Read relevant files first."
TODO_SECTION = "## `write_todos`\n\nA long made-up section about to-do lists."
TASK_SECTION = "## `task` (spawner)\n\nA long made-up section about helpers."
TODO_TOOL = "A long made-up description of the to-do tool, with examples."
TASK_TOOL = ("Launch a made-up helper.\n\nAvailable agents:\n"
             "{available_agents}\n\nExample: {{\"a\": 1}} and more.")
TEXTS = {L.BASE: BASE, L.TODO_SECTION: TODO_SECTION,
         L.TASK_SECTION: TASK_SECTION, L.TODO_TOOL: TODO_TOOL,
         L.TASK_TOOL: TASK_TOOL}

PAGE = "You review documents. Cite every page."
HELPERS = "- reader: reads pages.\n- figure: draws figures."
LIST = "\n\nAvailable subagent types:\n\n" + HELPERS

SHORT_TODO = "## `write_todos`\n\nShort to-do text."
SHORT_TODO_TOOL = "Short to-do tool text."
SHORT_TASK = "## `task`\n\nShort helper text."
SHORT_TASK_TOOL = "Short task tool text:\n{available_agents}"


def _middleware(texts=None):
    return L.LeanFrameworkPrompt(
        todo_prompt=SHORT_TODO, todo_tool_description=SHORT_TODO_TOOL,
        task_prompt=SHORT_TASK, task_tool_description=SHORT_TASK_TOOL,
        texts=TEXTS if texts is None else texts)


def _tool(name, description):
    def run(text: str = "") -> str:
        return "ok"
    return StructuredTool.from_function(run, name=name,
                                        description=description)


def _tools():
    task_desc = (TASK_TOOL.replace("{available_agents}", HELPERS)
                 .replace("{{", "{").replace("}}", "}"))
    return [_tool("write_todos", TODO_TOOL), _tool("task", task_desc),
            _tool("open_document", "Open a document.")]


def _blocks():
    """The system message as the library composes it: the page's prompt and
    the base prompt, then one block per middleware section."""
    return SystemMessage(content=[
        {"type": "text", "text": PAGE + "\n\n" + BASE},
        {"type": "text", "text": "\n\n" + TODO_SECTION},
        {"type": "text", "text": "\n\n" + TASK_SECTION + LIST}])


def _text(message):
    c = message.content
    return c if isinstance(c, str) else "".join(b["text"] for b in c)


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

def test_the_switch_is_declared_like_the_three_newest(monkeypatch):
    env = review_flags.LEAN_PROMPT_ENV
    assert env == "GEOTECH_REVIEW_LEAN_PROMPT"
    assert env in review_flags.LATER_ENVS
    assert env not in review_flags.ALL_ENVS
    assert env not in review_flags.DEFAULT_ON_ENVS
    assert all(env not in arm for arm in review_flags.ARMS.values())
    assert "lean_prompt" not in review_flags.ARMS
    assert {"LEAN_PROMPT_ENV", "lean_prompt"} <= set(review_flags.__all__)
    assert env in review_flags.__doc__
    monkeypatch.delenv(env, raising=False)
    assert review_flags.lean_prompt() is False
    assert env not in review_flags.describe()
    for off in ("", "0", "off", "no"):
        monkeypatch.setenv(env, off)
        assert review_flags.lean_prompt() is False
    monkeypatch.setenv(env, "1")
    assert review_flags.lean_prompt() is True
    assert f"{env}=1" in review_flags.describe()
    # an arm that does not name it clears it; the suite's pair sets it
    with review_flags.switches({}):
        assert review_flags.lean_prompt() is False
    with review_flags.switches({env: "1"}):
        assert review_flags.lean_prompt() is True


# ---------------------------------------------------------------------------
# The swap
# ---------------------------------------------------------------------------

def test_set_the_base_prompt_goes_and_the_sections_and_tools_get_short():
    message, tools, outcome = _middleware().lean(_blocks(), _tools())
    assert outcome["applied"] is True
    text = _text(message)
    assert text == (PAGE + "\n\n" + SHORT_TODO + "\n\n" + SHORT_TASK + LIST)
    assert BASE not in text and TODO_SECTION not in text \
        and TASK_SECTION not in text
    by_name = {t.name: t for t in tools}
    assert by_name["write_todos"].description == SHORT_TODO_TOOL
    assert by_name["task"].description == "Short task tool text:\n" + HELPERS
    assert by_name["open_document"].description == "Open a document."
    # the tools keep their names and arguments; the originals are untouched
    originals = _tools()
    assert [t.name for t in tools] == [t.name for t in originals]
    assert by_name["task"].args == originals[1].args
    assert outcome["system_chars_cut"] == len(_text(_blocks())) - len(text)
    assert outcome["tool_chars_cut"] == (
        len(originals[0].description) - len(SHORT_TODO_TOOL)
        + len(originals[1].description)
        - len(by_name["task"].description))


def test_set_a_prompt_held_as_one_text_is_cut_the_same_way():
    whole = SystemMessage(content=_text(_blocks()))
    message, _, outcome = _middleware().lean(whole, _tools())
    assert outcome["applied"] is True
    assert message.content == (PAGE + "\n\n" + SHORT_TODO + "\n\n"
                               + SHORT_TASK + LIST)


def test_a_tool_the_call_does_not_offer_is_not_counted_missing():
    _, tools, outcome = _middleware().lean(_blocks(), _tools()[2:])
    assert outcome["applied"] is True
    assert [t.description for t in tools] == ["Open a document."]


# ---------------------------------------------------------------------------
# All or nothing
# ---------------------------------------------------------------------------

def _not_applied(texts=None, message=None, tools=None):
    message = _blocks() if message is None else message
    tools = _tools() if tools is None else tools
    out_message, out_tools, outcome = _middleware(texts).lean(message, tools)
    assert outcome["applied"] is False
    # never a partial cut: the very same message and tools
    assert out_message is message
    assert all(a is b for a, b in zip(out_tools, tools))
    assert len(out_tools) == len(tools)
    return outcome["not_found"]


@pytest.mark.parametrize("label", [L.BASE, L.TODO_SECTION, L.TASK_SECTION,
                                   L.TODO_TOOL, L.TASK_TOOL])
def test_a_text_the_installed_library_no_longer_defines_cuts_nothing(label):
    assert _not_applied({**TEXTS, label: None}) == [label]


def test_a_base_prompt_not_in_the_prompt_word_for_word_cuts_nothing():
    reworded = {**TEXTS, L.BASE: BASE.replace("made-up", "reworded")}
    assert _not_applied(reworded) == [L.BASE]


def test_a_section_or_tool_text_not_as_the_library_wrote_it_cuts_nothing():
    message = SystemMessage(content=[
        {"type": "text", "text": PAGE + "\n\n" + BASE},
        {"type": "text", "text": "\n\n" + TODO_SECTION + " (a profile's)"},
        {"type": "text", "text": "\n\nA profile's own helper text." + LIST}])
    assert _not_applied(message=message) == [L.TASK_SECTION]
    tools = _tools()
    tools[0] = _tool("write_todos", TODO_TOOL + " Changed.")
    tools[1] = _tool("task", "A profile's own task description.")
    assert _not_applied(tools=tools) == [L.TODO_TOOL, L.TASK_TOOL]
    no_slot = {**TEXTS, L.TASK_TOOL: "A task description with no list."}
    assert _not_applied(no_slot) == [L.TASK_TOOL]


def test_every_text_not_found_is_named_in_the_library_s_order():
    assert _not_applied({k: None for k in TEXTS}) == [
        L.BASE, L.TODO_SECTION, L.TASK_SECTION, L.TODO_TOOL, L.TASK_TOOL]


def test_the_installed_library_s_texts_are_read_by_name():
    texts = L.library_texts()
    assert set(texts) == {label for label, _, _ in L.LIBRARY_TEXTS}
    assert all(v is None or (isinstance(v, str) and v.strip())
               for v in texts.values())


# ---------------------------------------------------------------------------
# The turn's record
# ---------------------------------------------------------------------------

class _Model(FakeMessagesListChatModel):
    seen: list = []
    bound: list = []

    def bind_tools(self, tools, **kw):
        self.bound.append(list(tools))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="done"))])


def _turn(tmp_path, middleware):
    from langchain.agents import create_agent
    model = _Model(responses=[AIMessage(content="x")], seen=[], bound=[])
    agent = create_agent(model, tools=_tools(),
                         system_prompt=_text(_blocks()),
                         middleware=[middleware] if middleware else [])
    log = activity_log.ActivityLogger(str(tmp_path), turn=1)
    with keep_work.turn_scope():
        agent.invoke({"messages": [{"role": "user", "content": "Hi"}]},
                     config={"callbacks": [log], "recursion_limit": 20})
        log.turn_end(turn_tokens=0)
    end = [r for r in activity_log.load(str(tmp_path))
           if r["event"] == "turn_end"][-1]
    return model, end


def test_the_turn_s_record_says_the_prompt_was_cut(tmp_path):
    model, end = _turn(tmp_path, _middleware())
    assert end["lean_prompt"]["applied"] is True
    assert end["lean_prompt"]["system_chars_cut"] > 0
    assert BASE not in _text(model.seen[0][0])
    sent = {t.name: t.description for t in model.bound[0]}
    assert sent["write_todos"] == SHORT_TODO_TOOL


def test_the_turn_s_record_says_when_a_text_was_not_found(tmp_path):
    model, end = _turn(tmp_path, _middleware({**TEXTS, L.BASE: None}))
    assert end["lean_prompt"] == {"applied": False, "not_found": [L.BASE]}
    assert _text(model.seen[0][0]) == _text(_blocks())
    sent = {t.name: t.description for t in model.bound[0]}
    assert sent["write_todos"] == TODO_TOOL


def test_without_the_middleware_the_record_has_no_lean_prompt(tmp_path):
    _, end = _turn(tmp_path, None)
    assert "lean_prompt" not in end


def test_a_call_not_applied_is_kept_over_a_later_one_and_helpers_are_not():
    with keep_work.turn_scope() as scope:
        keep_work.note_lean_prompt({"applied": False, "not_found": ["x"]})
        keep_work.note_lean_prompt({"applied": True})
        assert scope.lean_prompt == {"applied": False, "not_found": ["x"]}
        assert keep_work.lean_prompt_facts(scope) == {
            "lean_prompt": {"applied": False, "not_found": ["x"]}}
    with keep_work.turn_scope() as scope:
        token = keep_work._HELPER.set(object())
        try:
            keep_work.note_lean_prompt({"applied": True})
        finally:
            keep_work._HELPER.reset(token)
        assert scope.lean_prompt is None
        assert keep_work.lean_prompt_facts(scope) == {}
    keep_work.note_lean_prompt({"applied": True})      # outside a turn
    assert keep_work.lean_prompt_facts(None) == {}
