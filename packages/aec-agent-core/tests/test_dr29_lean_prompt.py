"""BACKLOG DR-29: the agent library's generic prompt off a page, behind a
switch (``GEOTECH_REVIEW_LEAN_PROMPT``, off unless set).

:class:`aec_agent_core.deep.lean_prompt.LeanFrameworkPrompt` removes the
library's base prompt word for word and swaps its to-do and helper sections
and the ``write_todos`` and ``task`` tool descriptions for a page's short
texts: only the texts the installed library defines (deepagents 0.6 defines
all five; 0.7 no base prompt and no helper section), all or nothing among
those: where one is not found word for word, the call goes out exactly as
it came and the turn's record says which. These tests hand the middleware
made-up library texts in both shapes, so they hold on any installed version
of the library; the page's own build is tested in Document Review
(``test_dr29_lean_prompt_page.py``).
"""

from __future__ import annotations

import sys
import types

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
#: The later library's shape: no base prompt and no helper section.
FEWER = {**TEXTS, L.BASE: None, L.TASK_SECTION: None}
ORDER = [label for label, _, _ in L.LIBRARY_TEXTS]

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
    assert outcome["cut"] == ORDER
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
    assert outcome["cut"] == [L.BASE, L.TODO_SECTION, L.TASK_SECTION]
    assert [t.description for t in tools] == ["Open a document."]


# ---------------------------------------------------------------------------
# Fewer texts defined (the later library's shape): those are cut
# ---------------------------------------------------------------------------

def _fewer_blocks(todo=TODO_SECTION):
    """The system message as the later library composes it: the page's
    prompt alone, then the to-do section; no helper section."""
    return SystemMessage(content=[
        {"type": "text", "text": PAGE},
        {"type": "text", "text": "\n\n" + todo}])


def test_with_fewer_texts_defined_those_are_cut_and_the_rest_not_required():
    message, tools, outcome = _middleware(FEWER).lean(_fewer_blocks(),
                                                      _tools())
    assert outcome["applied"] is True
    assert outcome["cut"] == [L.TODO_SECTION, L.TODO_TOOL, L.TASK_TOOL]
    assert _text(message) == PAGE + "\n\n" + SHORT_TODO
    assert SHORT_TASK not in _text(message)     # no section to swap
    by_name = {t.name: t.description for t in tools}
    assert by_name["write_todos"] == SHORT_TODO_TOOL
    assert by_name["task"] == "Short task tool text:\n" + HELPERS
    assert by_name["open_document"] == "Open a document."
    assert outcome["system_chars_cut"] == len(TODO_SECTION) - len(SHORT_TODO)


def test_with_fewer_texts_defined_each_must_still_be_found_word_for_word():
    reworded = _fewer_blocks(TODO_SECTION.replace("long", "longer"))
    assert _not_applied(FEWER, message=reworded) == [L.TODO_SECTION]
    tools = _tools()
    tools[1] = _tool("task", "A profile's own task description.")
    assert _not_applied(FEWER, message=_fewer_blocks(), tools=tools) == [
        L.TASK_TOOL]


@pytest.mark.parametrize("label", ORDER)
def test_a_text_the_installed_library_does_not_define_is_not_required(label):
    message, tools, outcome = _middleware({**TEXTS, label: None}).lean(
        _blocks(), _tools())
    assert outcome["applied"] is True
    assert outcome["cut"] == [other for other in ORDER if other != label]
    # what the library does not define is left as it is
    by_name = {t.name: t.description for t in tools}
    if label in (L.BASE, L.TODO_SECTION, L.TASK_SECTION):
        assert TEXTS[label] in _text(message)
    elif label == L.TODO_TOOL:
        assert by_name["write_todos"] == TODO_TOOL
    else:
        assert by_name["task"] == _tools()[1].description


def test_a_library_that_defines_none_of_the_texts_changes_nothing():
    message = _blocks()
    out, tools, outcome = _middleware({k: None for k in TEXTS}).lean(
        message, _tools())
    assert outcome == {"applied": True, "cut": [], "system_chars_cut": 0,
                       "tool_chars_cut": 0}
    assert out is message
    assert [t.description for t in tools] == [
        t.description for t in _tools()]


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
    tools = [_tool("task", "Another task description."),
             _tool("write_todos", "Another to-do description.")]
    assert _not_applied(message=SystemMessage(content=PAGE),
                        tools=tools) == ORDER


def test_the_installed_library_s_texts_are_read_by_name():
    texts = L.library_texts()
    assert set(texts) == set(ORDER)
    assert all(v is None or (isinstance(v, str) and v.strip())
               for v in texts.values())
    # langchain's to-do texts are defined on every version the core allows
    assert texts[L.TODO_SECTION] and texts[L.TODO_TOOL]


def test_a_name_a_module_serves_only_through_getattr_is_not_defined(
        monkeypatch):
    """The later library keeps its old base prompt as a deprecated
    stand-in served by the module's ``__getattr__`` and puts it in no
    prompt; that is not a text it defines."""
    module = types.ModuleType("made_up_agent_library")
    module.REAL = "A real text."
    module.EMPTY = "  "

    def __getattr__(name):
        if name == "OLD":
            return "A deprecated stand-in."
        raise AttributeError(name)

    module.__getattr__ = __getattr__
    monkeypatch.setitem(sys.modules, "made_up_agent_library", module)
    assert module.OLD == "A deprecated stand-in."
    table = (("real", "made_up_agent_library", "REAL"),
             ("old", "made_up_agent_library", "OLD"),
             ("empty", "made_up_agent_library", "EMPTY"),
             ("gone", "made_up_agent_library", "GONE"),
             ("no module", "made_up_agent_library_absent", "REAL"))
    assert L.library_texts(table) == {
        "real": "A real text.", "old": None, "empty": None, "gone": None,
        "no module": None}


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


def _turn(tmp_path, middleware, *, tools=None, system_prompt=None):
    from langchain.agents import create_agent
    model = _Model(responses=[AIMessage(content="x")], seen=[], bound=[])
    if middleware is None:
        middleware = []
    elif not isinstance(middleware, list):
        middleware = [middleware]
    agent = create_agent(model, tools=_tools() if tools is None else tools,
                         system_prompt=(_text(_blocks()) if system_prompt
                                        is None else system_prompt),
                         middleware=middleware)
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
    assert end["lean_prompt"]["cut"] == ORDER
    assert end["lean_prompt"]["system_chars_cut"] > 0
    assert BASE not in _text(model.seen[0][0])
    sent = {t.name: t.description for t in model.bound[0]}
    assert sent["write_todos"] == SHORT_TODO_TOOL


def test_the_turn_s_record_says_when_a_text_was_not_found(tmp_path):
    reworded = {**TEXTS, L.BASE: BASE.replace("made-up", "reworded")}
    model, end = _turn(tmp_path, _middleware(reworded))
    assert end["lean_prompt"] == {"applied": False, "not_found": [L.BASE]}
    assert _text(model.seen[0][0]) == _text(_blocks())
    sent = {t.name: t.description for t in model.bound[0]}
    assert sent["write_todos"] == TODO_TOOL


def test_last_it_cuts_the_to_do_section_a_middleware_before_it_writes(
        tmp_path):
    """The later library's arrangement: the core attaches langchain's own
    to-do middleware, which writes the to-do section; the lean prompt,
    kept after it, sees and cuts it (langchain's real to-do texts)."""
    from langchain.agents.middleware import TodoListMiddleware
    real = L.library_texts()
    texts = {**FEWER, L.TODO_SECTION: real[L.TODO_SECTION],
             L.TODO_TOOL: real[L.TODO_TOOL], L.TASK_TOOL: None}
    tools = _tools()[2:]
    model, end = _turn(tmp_path, [TodoListMiddleware(), _middleware(texts)],
                       tools=tools, system_prompt=PAGE)
    assert end["lean_prompt"]["applied"] is True
    assert end["lean_prompt"]["cut"] == [L.TODO_SECTION, L.TODO_TOOL]
    assert _text(model.seen[0][0]) == PAGE + "\n\n" + SHORT_TODO
    sent = {t.name: t.description for t in model.bound[0]}
    assert sent["write_todos"] == SHORT_TODO_TOOL
    # put ahead of the to-do middleware it would never see the section,
    # and so it cuts nothing
    ahead = tmp_path / "ahead"
    ahead.mkdir()
    model, end = _turn(ahead, [_middleware(texts), TodoListMiddleware()],
                       tools=tools, system_prompt=PAGE)
    assert end["lean_prompt"] == {"applied": False,
                                  "not_found": [L.TODO_SECTION]}
    assert real[L.TODO_SECTION] in _text(model.seen[0][0])


def test_the_core_attaches_the_to_do_middleware_ahead_of_one_kept_last(
        monkeypatch):
    """``compile_agent`` adds the to-do middleware where deepagents did not
    (0.7): at the end as before, but ahead of a middleware marked
    ``keep_last``."""
    from langchain.agents.middleware import AgentMiddleware

    from aec_agent_core.deep import assembly
    if assembly._TodoMiddleware is None:
        pytest.skip("no to-do middleware installed")
    monkeypatch.delenv("GEOTECH_INSTRUCTIONS", raising=False)

    class Plain(AgentMiddleware):
        pass

    seen = []

    def create(**kwargs):
        seen.append(list(kwargs["middleware"]))
        names = {"write_todos": None} if len(seen) > 1 else {}
        return types.SimpleNamespace(nodes={"tools": types.SimpleNamespace(
            bound=types.SimpleNamespace(tools_by_name=names))})

    first, second, lean = Plain(), Plain(), _middleware()
    assembly.compile_agent(create, model=None, tools=[], system_prompt="P",
                           subagents=[],
                           create_kwargs={"middleware": [first, lean]})
    assert seen[0] == [first, lean]
    assert seen[1][0] is first and seen[1][2] is lean
    assert isinstance(seen[1][1], assembly._TodoMiddleware)
    seen.clear()
    assembly.compile_agent(create, model=None, tools=[], system_prompt="P",
                           subagents=[],
                           create_kwargs={"middleware": [first, second]})
    assert seen[1][:2] == [first, second] and len(seen[1]) == 3
    assert isinstance(seen[1][2], assembly._TodoMiddleware)


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
