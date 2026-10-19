"""deepagents' ``execute`` and ``delete`` hidden on the geotech page
(BACKLOG GM-176).

Neither can run on the app's in-memory backend, and live wave w13b showed
both kinds of harm: F55j's calc helper called ``execute`` with no command,
and F18j's unknown-tool error listed seven tools the consult cannot use.
Every geotech-page agent -- the primary, ``calc``, ``general-purpose``,
``figure``, ``references`` and ``reviewer`` -- is offered neither, and an
unknown-tool error lists no hidden tool. deepagents leaves ``execute`` off
the menu itself when its backend cannot run it, so these tests make the
backend say it can (as a sandbox backend would, or a deepagents release
that offers it regardless) and check the app's own filter.
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from funhouse_agent import review_flags
from funhouse_agent.deep.agent import build_deep_agent
from funhouse_agent.deep.figure_helper import FIGURE_HELPER_NAME
from funhouse_agent.deep.limits import UNRUNNABLE_TOOLS, HideTools

HELPERS = ("calc", "general-purpose", FIGURE_HELPER_NAME, "references",
           "reviewer")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


@pytest.fixture
def can_execute(monkeypatch):
    """deepagents' backend check says yes, so only the app's filter keeps
    ``execute`` off a menu (and deepagents adds its "Execute Tool"
    prompt section)."""
    import deepagents.middleware.filesystem as fs
    monkeypatch.setattr(fs, "supports_execution", lambda backend: True)


def _scripted(replies, seen, menus):
    """A chat model answering from ``replies`` (``(text, [(name, args)])``)
    and recording each call's messages and tool menu."""

    class Scripted(BaseChatModel):
        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kw):
            menus.append([t.get("name") if isinstance(t, dict)
                          else getattr(t, "name", None) for t in tools])
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            seen.append(list(messages))
            text, calls = replies.pop(0) if replies else ("done", [])
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=text, tool_calls=[
                    {"name": n, "args": a, "id": f"c{len(seen)}_{i}"}
                    for i, (n, a) in enumerate(calls)]))])

    return Scripted()


def _system(messages) -> str:
    sm = next((m for m in messages if getattr(m, "type", "") == "system"),
              None)
    if sm is None:
        return ""
    content = sm.content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                       for b in content)
    return str(content)


def _geotech(model, monkeypatch):
    monkeypatch.setenv(review_flags.FIGURE_HELPER_ENV, "1")
    return build_deep_agent(model, enable_calc_subagent=True)


def _run(agent):
    return agent.invoke({"messages": [{"role": "user", "content": "go"}]},
                        config={"recursion_limit": 80})


def test_no_geotech_page_agent_is_offered_execute_or_delete(
        monkeypatch, can_execute):
    replies = []
    for name in HELPERS:
        replies += [("", [("task", {"subagent_type": name,
                                    "description": "Say done."})]),
                    ("done", [])]
    replies.append(("All done.", []))
    seen, menus = [], []
    agent = _geotech(_scripted(replies, seen, menus), monkeypatch)
    # registered (nothing that refers to them breaks), never offered
    assert {"execute", "read_file"} <= set(
        agent.nodes["tools"].bound.tools_by_name)
    _run(agent)
    assert not replies, "every helper ran"
    assert len(menus) == 2 * len(HELPERS) + 1
    for menu in menus:
        assert not set(menu) & UNRUNNABLE_TOOLS, menu
    # the primary keeps the scratch filesystem; so does every helper but
    # the references consult, which hides it (REFERENCE_HIDDEN_TOOLS)
    assert "read_file" in menus[0] and "task" in menus[0]
    for i, name in enumerate(HELPERS, start=1):
        assert "task" not in menus[2 * i - 1], name
    for messages in seen:
        assert "Execute Tool" not in _system(messages)


def test_the_review_page_still_hides_them(monkeypatch, can_execute):
    from webapp.profiles import DOCUMENT_REVIEW
    seen, menus = [], []
    agent = build_deep_agent(_scripted([("Hello.", [])], seen, menus),
                             **DOCUMENT_REVIEW.build_kwargs())
    _run(agent)
    assert not set(menus[-1]) & UNRUNNABLE_TOOLS
    assert "read_file" not in menus[-1]          # its own, wider filter
    assert "Execute Tool" not in _system(seen[-1])


@pytest.mark.parametrize("who", ["primary", "calc"])
def test_an_unknown_tool_error_lists_no_hidden_tool(monkeypatch, who):
    """F18j [189]: the error named every REGISTERED tool, hidden or not."""
    made_up = ("render_pdf_page", {"path": "calc.pdf"})
    if who == "primary":
        replies = [("", [made_up]), ("ok", [])]
    else:
        replies = [("", [("task", {"subagent_type": "calc",
                                   "description": "Look at the PDF."})]),
                   ("", [made_up]), ("done", []), ("ok", [])]
    seen, menus = [], []
    agent = _geotech(_scripted(replies, seen, menus), monkeypatch)
    _run(agent)
    told = [m for msgs in seen for m in msgs if isinstance(m, ToolMessage)
            and "render_pdf_page" in str(m.content)]
    assert told, "the unknown-tool error reached the model"
    listed = str(told[-1].content).split("try one of", 1)
    assert len(listed) == 2, told[-1].content
    names = {n.strip(" '\"[].") for n in listed[1].split("]")[0].split(",")}
    assert names and not names & UNRUNNABLE_TOOLS, names
    assert "read_file" in names                 # the page keeps it


def _req(names, system=None):
    from langchain_core.messages import SystemMessage

    class Req:
        def __init__(self, tools, sm):
            self.tools, self.system_message = tools, sm

        def override(self, tools=None, system_message=None):
            return Req(self.tools if tools is None else tools,
                       self.system_message if system_message is None
                       else system_message)

    return Req([type("T", (), {"name": n})() for n in names],
               SystemMessage(content=system) if system is not None else None)


def test_hide_tools_takes_a_hidden_tools_prompt_section_with_it():
    hide = HideTools(UNRUNNABLE_TOOLS)
    text = ("Intro.\n\n## Filesystem Tools\nls, read_file.\n\n"
            "## Execute Tool `execute`\nRun a shell command.\n\n"
            "## Next\nMore.\n")
    out = []
    hide.wrap_model_call(_req(["execute", "delete", "read_file"], text),
                         out.append)
    assert [t.name for t in out[0].tools] == ["read_file"]
    sm = out[0].system_message.content
    assert "Execute Tool" not in sm and "Run a shell" not in sm
    assert "## Filesystem Tools" in sm and "## Next\nMore." in sm
    # no section to take: the prompt is the very same text
    plain = "Intro.\n\n\n\nBody."
    hide.wrap_model_call(_req(["read_file"], plain), out.append)
    assert out[1].system_message.content is not None
    assert out[1].system_message.content == plain


def test_hide_tools_scrubs_an_unknown_tool_errors_list():
    hide = HideTools(UNRUNNABLE_TOOLS)

    class Req:
        tool = None

    err = ToolMessage(content=(
        "Error: render_pdf_page is not a valid tool, try one of "
        "[read_file, execute, call_agent, delete]."), tool_call_id="1")
    out = hide.wrap_tool_call(Req(), lambda r: err)
    assert out.content == ("Error: render_pdf_page is not a valid tool, try "
                           "one of [read_file, call_agent].")

    class Known:
        tool = object()

    same = hide.wrap_tool_call(Known(), lambda r: err)
    assert same is err                           # a real tool's result
