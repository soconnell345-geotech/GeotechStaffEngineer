"""BACKLOG DR-29 on the page: the agent library's generic prompt off Document
Review's default build, behind ``GEOTECH_REVIEW_LEAN_PROMPT`` (off unless
set).

Set, the page's main agent is sent its system prompt without deepagents'
base prompt, with short to-do and helper sections, and short ``write_todos``
and ``task`` descriptions (the lean build's texts; ``review_agent.LEAN_*``):
each of those the INSTALLED library defines, all or nothing among them
(:mod:`aec_agent_core.deep.lean_prompt`). deepagents 0.6 defines all five;
0.7 writes no base prompt and no helper section, so there the to-do section
and the two tool descriptions are cut. The tests below run on whichever
version is installed and check each text it defines. Unset, what the model
is sent is what it was. The page is built on a fake model and run one step;
nothing calls a model or the network.
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
ORDER = [label for label, _, _ in L.LIBRARY_TEXTS]
SECTIONS = (L.BASE, L.TODO_SECTION, L.TASK_SECTION)
#: The page's short text for each section (None: the base prompt goes).
SHORT = {L.BASE: None, L.TODO_SECTION: RA.LEAN_TODO_PROMPT,
         L.TASK_SECTION: RA.LEAN_TASK_PROMPT}


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


def _frame(template):
    return [part.replace("{{", "{").replace("}}", "}")
            for part in template.split(L.AGENTS_SLOT)]


def _helpers(task_description, template=None):
    before, after = _frame(template or LIB[L.TASK_TOOL])
    return task_description[len(before):len(task_description) - len(after)]


def _expected(off, off_tools, texts):
    """What a call should carry with the switch set, from what it carried
    unset: each section ``texts`` defines cut or made short, and each tool
    text it defines swapped; ``(system text, tools, cut)``."""
    on = off
    for label in SECTIONS:
        if texts[label]:
            assert on.count(texts[label]) == 1, label
            on = (on.replace("\n\n" + texts[label], "", 1)
                  if SHORT[label] is None
                  else on.replace(texts[label], SHORT[label], 1))
    tools = dict(off_tools)
    if texts[L.TODO_TOOL] and "write_todos" in tools:
        tools["write_todos"] = RA.LEAN_TODO_TOOL_DESCRIPTION
    if texts[L.TASK_TOOL] and "task" in tools:
        tools["task"] = RA.LEAN_TASK_TOOL_DESCRIPTION.replace(
            L.AGENTS_SLOT, _helpers(off_tools["task"], texts[L.TASK_TOOL]))
    cut = [label for label in ORDER if texts[label] and (
        label in SECTIONS
        or (label == L.TODO_TOOL and "write_todos" in off_tools)
        or (label == L.TASK_TOOL and "task" in off_tools))]
    return on, tools, cut


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
    assert "LeanFrameworkPrompt" not in helpers
    # the main agent's last middleware, so it sees what the model gets
    main = [c for c in calls if c["kind"] == "create_deep_agent"]
    assert len(main) == 1
    names = [type(m).__name__ for m in main[0]["kwargs"]["middleware"]]
    assert names[-1] == "LeanFrameworkPrompt"
    assert main[0]["kwargs"]["middleware"][-1].keep_last is True
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

def test_set_each_text_the_installed_library_defines_is_cut_or_short(
        monkeypatch, tmp_path):
    """Runs on whichever library is installed: every text it defines is
    gone from what the model is handed and the page's short text is there;
    the rest is as it was."""
    defined = [label for label in ORDER if LIB[label]]
    print(f"\nDR-29 texts the installed library defines: {defined}")
    # both shapes the core allows define the to-do texts and the task text
    assert {L.TODO_SECTION, L.TODO_TOOL, L.TASK_TOOL} <= set(defined)
    off, off_tools, _ = _turn(monkeypatch, tmp_path, None)
    on, on_tools, end = _turn(monkeypatch, tmp_path, "1")
    expected, expected_tools, cut = _expected(off, off_tools, LIB)
    assert on == expected
    assert on_tools == expected_tools
    for label in SECTIONS:
        if LIB[label]:
            assert LIB[label] not in on, label
            if SHORT[label]:
                assert SHORT[label] in on, label
    # the to-do tool is offered on every version (the core attaches it where
    # deepagents does not), and its text is short
    assert on_tools["write_todos"] == RA.LEAN_TODO_TOOL_DESCRIPTION
    assert LIB[L.TODO_TOOL] not in on_tools["write_todos"]
    # the task tool's fixed text is gone; the page's list of helpers stays
    before, after = _frame(LIB[L.TASK_TOOL])
    assert before not in on_tools["task"] and after not in on_tools["task"]
    assert on_tools["task"].startswith(
        RA.LEAN_TASK_TOOL_DESCRIPTION.split(L.AGENTS_SLOT)[0])
    assert "- general-purpose: " + REVIEW_READER_DESCRIPTION \
        in on_tools["task"]
    assert "page_reader" not in on_tools["task"] + on
    # the page's own prompt stays; the helper list after a helper section
    # stays where the library writes one
    assert build_document_review_prompt() in on
    if LIB[L.TASK_SECTION]:
        assert "Available subagent types:" in on
        assert "- general-purpose: " + REVIEW_READER_DESCRIPTION in on
    assert end["lean_prompt"] == {
        "applied": True, "cut": cut, "system_chars_cut": len(off) - len(on),
        "tool_chars_cut": sum(len(off_tools[n]) - len(on_tools[n])
                              for n in ("write_todos", "task"))}


def test_the_system_prompt_s_length_off_and_on(monkeypatch, tmp_path):
    off = len(_turn(monkeypatch, tmp_path, None)[0])
    on = len(_turn(monkeypatch, tmp_path, "1")[0])
    print(f"\nDR-29 system prompt on deepagents {_version('deepagents')}, "
          f"langchain {_version('langchain')}: {off} characters off, {on} "
          f"on, {off - on} saved")
    saving = 0
    for label in SECTIONS:
        if LIB[label]:
            saving += (2 + len(LIB[label]) if SHORT[label] is None
                       else len(LIB[label]) - len(SHORT[label]))
    assert off - on == saving > 0
    assert (len(RA.LEAN_TODO_PROMPT), len(RA.LEAN_TASK_PROMPT)) == (314, 148)
    if (_version("deepagents"), _version("langchain")) == LAPTOP:
        assert off - on == LAPTOP_SAVING == 5312
        # the lengths themselves move with the page's own prompt, so they
        # are stated, not pinned: LAPTOP_CHARS on this branch
        assert on == off - LAPTOP_SAVING


# ---------------------------------------------------------------------------
# Set, and fewer texts defined: those are cut, the rest is not required
# ---------------------------------------------------------------------------

def test_a_library_without_a_base_prompt_or_helper_section_cuts_the_rest(
        monkeypatch, tmp_path):
    """deepagents 0.7's shape on any installed version: no base prompt and
    no helper section defined, so the to-do section and the two tool texts
    are cut and whatever else the prompt holds stays word for word."""
    fewer = {**LIB, L.BASE: None, L.TASK_SECTION: None}
    off, off_tools, _ = _turn(monkeypatch, tmp_path, None)
    monkeypatch.setattr(L, "library_texts", lambda: dict(fewer))
    on, on_tools, end = _turn(monkeypatch, tmp_path, "1")
    expected, expected_tools, cut = _expected(off, off_tools, fewer)
    assert cut == [L.TODO_SECTION, L.TODO_TOOL, L.TASK_TOOL]
    assert (on, on_tools) == (expected, expected_tools)
    for label in (L.BASE, L.TASK_SECTION):
        if LIB[label]:
            assert LIB[label] in on, label
    assert RA.LEAN_TASK_PROMPT not in on
    assert end["lean_prompt"]["applied"] is True
    assert end["lean_prompt"]["cut"] == cut


# ---------------------------------------------------------------------------
# Set, and a text the library defines not found: nothing is cut
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label", [L.BASE, L.TODO_SECTION, L.TODO_TOOL])
def test_an_upgrade_that_reworded_a_text_leaves_the_page_as_it_is(
        monkeypatch, tmp_path, label):
    off, off_tools, _ = _turn(monkeypatch, tmp_path, None)
    reworded = "A made-up text an upgrade put in place of the old one."
    monkeypatch.setattr(L, "library_texts",
                        lambda: {**LIB, label: reworded})
    on, on_tools, end = _turn(monkeypatch, tmp_path, "1")
    assert (on, on_tools) == (off, off_tools)
    assert end["lean_prompt"] == {"applied": False, "not_found": [label]}
