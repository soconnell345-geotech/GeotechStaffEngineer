"""BACKLOG DR-29 on the page: the agent library's generic prompt off Document
Review's default build, behind ``GEOTECH_REVIEW_LEAN_PROMPT`` (off unless
set).

Set, the page's main agent is sent its system prompt without deepagents'
base prompt, with short to-do and helper sections, and short ``write_todos``
and ``task`` descriptions (the lean build's texts; ``review_agent.LEAN_*``),
all or nothing (:mod:`aec_agent_core.deep.lean_prompt`). Unset, what the
model is sent is what it was. The page is built on a fake model and run one
step; nothing calls a model or the network.

The cut needs deepagents 0.6's texts: deepagents 0.7 no longer appends a
base prompt or writes a helper section, so there the page is left as it is
and the record says which texts were not found (the tests below that need
the 0.6 texts skip where the installed library lacks them, and the one that
needs their absence skips where it has them).
"""

import importlib.metadata
import os
import re

import pytest

pytest.importorskip("planlens.tools")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel)
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from aec_agent_core import activity_log, review_flags  # noqa: E402
from aec_agent_core.deep import keep_work  # noqa: E402
from aec_agent_core.deep import lean_prompt as L  # noqa: E402
from aec_document_review import review_agent as RA  # noqa: E402
from aec_document_review.review_build import (  # noqa: E402
    REVIEW_READER_DESCRIPTION, build_review_page_agent)
from aec_document_review.review_prompt import build_document_review_prompt  # noqa: E402
from tests import page_builds  # noqa: E402
from tests.test_review_harness_offline import (  # noqa: E402
    FakeEngine, _review_kwargs)

ENV = review_flags.LEAN_PROMPT_ENV
LIB = L.library_texts()
DEFINED = all(LIB.values())
SECTIONS = (L.BASE, L.TODO_SECTION, L.TASK_SECTION)

needs_texts = pytest.mark.skipif(
    not DEFINED, reason="the installed agent library lacks the 0.6 texts")


def _version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:  # noqa: BLE001
        return None


#: The system prompt's saving in characters on the libraries the laptop
#: runs (deepagents 0.6.8, langchain 1.3.4): the blank line and the base
#: prompt (2 + 2,258), the to-do section (1,370 for 314) and the helper
#: section (2,144 for 148). Measured on this branch: 17,582 characters off,
#: 12,270 on.
LAPTOP = ("0.6.8", "1.3.4")
LAPTOP_SAVING = 2 + 2258 + (1370 - 314) + (2144 - 148)
LAPTOP_CHARS = (17582, 12270)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.LATER_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


class Cap(FakeMessagesListChatModel):
    """Records what each model call was sent; answers "done"."""

    seen: list = []
    bound: list = []

    def bind_tools(self, tools, **kw):
        self.bound.append(list(tools))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="done"))])


def _text(message):
    c = message.content
    return c if isinstance(c, str) else "".join(
        b.get("text", "") for b in c if isinstance(b, dict))


def _turn(monkeypatch, tmp_path, value):
    """One step of the page's default build with the switch at ``value``
    (None: unset): ``(system text, {tool: description}, turn_end)``."""
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    model = Cap(responses=[AIMessage(content="x")], seen=[], bound=[])
    agent = build_review_page_agent(model, engine=FakeEngine(),
                                    **_review_kwargs())
    folder = tmp_path / f"turn_{value}"
    folder.mkdir()
    log = activity_log.ActivityLogger(str(folder), turn=1)
    with keep_work.turn_scope():
        agent.invoke({"messages": [{"role": "user", "content": "Hi"}]},
                     config={"recursion_limit": 50, "callbacks": [log]})
        log.turn_end(turn_tokens=0)
    end = [r for r in activity_log.load(str(folder))
           if r["event"] == "turn_end"][-1]
    tools = {(t.get("name") if isinstance(t, dict) else t.name):
             (t.get("description") if isinstance(t, dict) else t.description)
             for t in model.bound[0]}
    return _text(model.seen[0][0]), tools, end


def _helpers(task_description):
    before, after = (part.replace("{{", "{").replace("}}", "}")
                     for part in LIB[L.TASK_TOOL].split(L.AGENTS_SLOT))
    return task_description[len(before):len(task_description) - len(after)]


# ---------------------------------------------------------------------------
# Unset: today's page
# ---------------------------------------------------------------------------

def test_unset_the_page_is_sent_today_s_prompt_and_tool_texts(
        monkeypatch, tmp_path):
    text, tools, end = _turn(monkeypatch, tmp_path, None)
    for value in ("0", "off"):
        assert _turn(monkeypatch, tmp_path, value)[:2] == (text, tools)
    # the library's own texts, as the library wrote them
    for label in SECTIONS:
        if LIB[label]:
            assert LIB[label] in text
    if LIB[L.TODO_TOOL]:
        assert tools["write_todos"] == LIB[L.TODO_TOOL]
    if LIB[L.TASK_TOOL]:
        assert tools["task"].endswith(
            LIB[L.TASK_TOOL].split(L.AGENTS_SLOT)[1]
            .replace("{{", "{").replace("}}", "}"))
    for short in (RA.LEAN_TODO_PROMPT, RA.LEAN_TASK_PROMPT):
        assert short not in text
    assert "lean_prompt" not in end


def test_the_middleware_is_on_the_main_agent_only_and_only_when_set():
    def middleware(calls):
        mains, helpers = [], []
        for call in calls:
            kw = call["kwargs"]
            mains += [type(m).__name__ for m in kw.get("middleware") or []]
            for spec in kw.get("subagents") or []:
                helpers += [type(m).__name__
                            for m in spec.get("middleware") or []]
        return mains, helpers

    _, calls, _ = page_builds.build("review_default", capture=True)
    mains, helpers = middleware(calls)
    assert "HideScratchFilesystem" in mains
    assert "LeanFrameworkPrompt" not in mains + helpers
    _, calls, _ = page_builds.build("review_default", capture=True,
                                    settings={ENV: "1"})
    mains, helpers = middleware(calls)
    assert mains.count("LeanFrameworkPrompt") == 1
    assert mains.index("LeanFrameworkPrompt") == \
        mains.index("HideScratchFilesystem") + 1
    assert "LeanFrameworkPrompt" not in helpers
    # the looking-only build is not touched
    _, calls, _ = page_builds.build("review_minimal", capture=True,
                                    settings={ENV: "1"})
    assert "LeanFrameworkPrompt" not in sum(middleware(calls), [])


def test_the_switch_is_read_by_the_review_page_s_build_alone():
    """The geotech page, the helpers and the minimal build never read it."""
    import aec_agent_core
    import aec_document_review
    readers, users = [], []
    for package in (aec_agent_core, aec_document_review):
        root = os.path.dirname(package.__file__)
        for folder, _, files in os.walk(root):
            for name in files:
                if not name.endswith(".py"):
                    continue
                path = os.path.join(folder, name)
                with open(path, encoding="utf-8") as fh:
                    source = fh.read()
                rel = os.path.relpath(path, os.path.dirname(root))
                if ".lean_prompt()" in source:
                    readers.append(rel.replace(os.sep, "/"))
                if re.search(r"(?<!class )LeanFrameworkPrompt\(", source):
                    users.append(rel.replace(os.sep, "/"))
    assert readers == ["aec_document_review/review_build.py"]
    assert users == ["aec_document_review/review_build.py"]


# ---------------------------------------------------------------------------
# Set: the cut, word for word
# ---------------------------------------------------------------------------

@needs_texts
def test_set_the_base_prompt_goes_and_the_sections_are_short(
        monkeypatch, tmp_path):
    off, off_tools, _ = _turn(monkeypatch, tmp_path, None)
    on, on_tools, end = _turn(monkeypatch, tmp_path, "1")
    expected = (off.replace("\n\n" + LIB[L.BASE], "", 1)
                .replace(LIB[L.TODO_SECTION], RA.LEAN_TODO_PROMPT, 1)
                .replace(LIB[L.TASK_SECTION], RA.LEAN_TASK_PROMPT, 1))
    assert on == expected
    for label in SECTIONS:
        assert LIB[label] not in on
    # the page's own prompt and deepagents' list of helpers stay
    assert build_document_review_prompt() in on
    assert "Available subagent types:" in on
    assert "- general-purpose: " + REVIEW_READER_DESCRIPTION in on
    # the two tools' descriptions are short; the list of helpers is kept
    assert on_tools["write_todos"] == RA.LEAN_TODO_TOOL_DESCRIPTION
    assert on_tools["task"] == RA.LEAN_TASK_TOOL_DESCRIPTION.replace(
        L.AGENTS_SLOT, _helpers(off_tools["task"]))
    assert "general-purpose" in on_tools["task"]
    assert "page_reader" not in on_tools["task"] + on
    # every other tool is offered word for word as before
    assert set(on_tools) == set(off_tools)
    for name in set(off_tools) - {"write_todos", "task"}:
        assert on_tools[name] == off_tools[name], name
    assert end["lean_prompt"] == {
        "applied": True, "system_chars_cut": len(off) - len(on),
        "tool_chars_cut": (len(off_tools["write_todos"])
                           + len(off_tools["task"])
                           - len(on_tools["write_todos"])
                           - len(on_tools["task"]))}


@needs_texts
def test_the_system_prompt_s_length_off_and_on(monkeypatch, tmp_path):
    off = len(_turn(monkeypatch, tmp_path, None)[0])
    on = len(_turn(monkeypatch, tmp_path, "1")[0])
    print(f"\nDR-29 system prompt: {off} characters off, {on} on, "
          f"{off - on} saved")
    assert off - on == (2 + len(LIB[L.BASE])
                        + len(LIB[L.TODO_SECTION]) - len(RA.LEAN_TODO_PROMPT)
                        + len(LIB[L.TASK_SECTION]) - len(RA.LEAN_TASK_PROMPT))
    assert (len(RA.LEAN_TODO_PROMPT), len(RA.LEAN_TASK_PROMPT)) == (314, 148)
    if (_version("deepagents"), _version("langchain")) == LAPTOP:
        assert off - on == LAPTOP_SAVING == 5312
        # the lengths themselves move with the page's own prompt, so they
        # are stated, not pinned: LAPTOP_CHARS on this branch
        assert on == off - LAPTOP_SAVING


# ---------------------------------------------------------------------------
# Set, and a text not found: nothing is cut, and the record says so
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("moved", ["gone", "reworded"])
def test_an_upgrade_that_moved_the_base_prompt_leaves_the_page_as_it_is(
        monkeypatch, tmp_path, moved):
    off, off_tools, _ = _turn(monkeypatch, tmp_path, None)
    base = None if moved == "gone" else "You are a deep agent, reworded."
    monkeypatch.setattr(L, "library_texts", lambda: {**LIB, L.BASE: base})
    on, on_tools, end = _turn(monkeypatch, tmp_path, "1")
    assert (on, on_tools) == (off, off_tools)
    assert end["lean_prompt"]["applied"] is False
    assert L.BASE in end["lean_prompt"]["not_found"]
    assert "system_chars_cut" not in end["lean_prompt"]


@pytest.mark.skipif(DEFINED, reason="the installed agent library has the "
                                    "0.6 texts")
def test_on_a_library_without_the_texts_the_page_is_left_as_it_is(
        monkeypatch, tmp_path):
    off, off_tools, _ = _turn(monkeypatch, tmp_path, None)
    on, on_tools, end = _turn(monkeypatch, tmp_path, "1")
    assert (on, on_tools) == (off, off_tools)
    assert end["lean_prompt"]["applied"] is False
    assert set(end["lean_prompt"]["not_found"]) >= {
        label for label, text in LIB.items() if not text}
