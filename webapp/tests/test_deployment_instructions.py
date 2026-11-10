"""A deployment's instructions on both pages' agents (REF-64 (B)).

Built through ``webapp.tests.page_builds`` (``webapp.core.build_agent``
with each page's profile, a fake model, every ``GEOTECH_*`` unset) with
``GEOTECH_INSTRUCTIONS`` pointing at a folder of made-up text. Unset, the
builds are the recorded fingerprints (``test_page_fingerprints.py``)."""

import pytest

from aec_agent_core.deep import instructions as ins
from webapp.tests import page_builds

pytest.importorskip("deepagents")

TAIL = "\n\n" + ins.PROTECTED_RULES + "\n\n" + ins.HEADING + "\n\n"


@pytest.fixture()
def folder(tmp_path):
    (tmp_path / "all.md").write_text("Report lengths in metres.",
                                     encoding="utf-8")
    (tmp_path / "main.md").write_text("Main agent rule.", encoding="utf-8")
    (tmp_path / "references.md").write_text("Prefer pack Example-1.",
                                            encoding="utf-8")
    return str(tmp_path)


def _main_call(calls):
    return next(c for c in calls if c["kind"] == "create_deep_agent")


@pytest.mark.parametrize("name", ["geotech_default", "review_default"])
def test_unset_build_has_no_appended_record(name):
    agent, calls, _ = page_builds.build(name, capture=True)
    assert "appended" not in agent.geotech_build
    call = _main_call(calls)
    assert ins.HEADING not in call["kwargs"]["system_prompt"]
    for spec in call["kwargs"]["subagents"]:
        assert ins.HEADING not in (spec.get("system_prompt") or "")


@pytest.mark.parametrize("name", ["geotech_default", "review_default"])
def test_every_role_gets_its_block_last(name, folder):
    agent, calls, _ = page_builds.build(
        name, capture=True, settings={ins.ENV: folder})
    kw = _main_call(calls)["kwargs"]
    assert kw["system_prompt"].endswith(
        TAIL + "Report lengths in metres.\n\nMain agent rule.")
    roles = agent.geotech_build["appended"]["roles"]
    assert roles["main"]["files"] == ["all.md", "main.md"]
    for spec in kw["subagents"]:
        if "runnable" in spec:
            continue
        expected = "Report lengths in metres."
        if spec["name"] == "references":
            expected += "\n\nPrefer pack Example-1."
        assert spec["system_prompt"].endswith(TAIL + expected), spec["name"]
        assert spec["name"] in roles
    # the review page's reading helper is "general-purpose" too
    assert "general-purpose" in roles


def test_the_only_fingerprint_difference_is_the_prompts(folder, monkeypatch):
    """With references.md alone, the librarian's prompt hash is the one
    difference (plan 3.6)."""
    import os
    os.remove(os.path.join(folder, "all.md"))
    os.remove(os.path.join(folder, "main.md"))
    before = page_builds.fingerprint("geotech_default")["app"]
    real = page_builds.build
    monkeypatch.setattr(page_builds, "build", lambda name, **kw: real(
        name, **{**kw, "settings": {ins.ENV: folder}}))
    after = page_builds.fingerprint("geotech_default")["app"]
    diff = page_builds.differences(before, after)
    assert diff and all("system_prompt" in d for d in diff), diff
    assert len(diff) == 1 and diff[0].startswith("/calls[0]/subagents["), diff
    index = int(diff[0].split("subagents[")[1].split("]")[0])
    assert before["calls"][0]["subagents"][index]["name"] == "references"


def test_an_unreadable_folder_still_builds(tmp_path):
    agent, calls, _ = page_builds.build(
        "review_default", capture=True,
        settings={ins.ENV: str(tmp_path / "missing")})
    rec = agent.geotech_build["appended"]
    assert rec["roles"] == {} and "FileNotFoundError" in rec["error"]
    assert ins.HEADING not in _main_call(calls)["kwargs"]["system_prompt"]


def test_both_routes_of_each_compiled_helper_get_its_block(tmp_path,
                                                           monkeypatch):
    """report_ingest and report_library are each reached two ways: the
    primary tool of that name and ``task`` to the sub-agent. Every route
    gets the block read when the agent was built, and the record names it
    (the review of REF-64 (B) found the two tools had none)."""
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel)
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from funhouse_agent import document_tools
    from funhouse_agent.deep import agent as agent_mod
    import report_ingest.engine as engine_mod
    import report_ingest.subagent as subagent_mod
    from report_ingest.tests.library_fixtures import build_library

    library = tmp_path / "library"
    library.mkdir()
    build_library(library)
    inst = tmp_path / "inst"
    inst.mkdir()
    (inst / "all.md").write_text("Everyone rule.", encoding="utf-8")
    (inst / "report_library.md").write_text("Library rule.",
                                            encoding="utf-8")
    (inst / "report_ingest.md").write_text("Ingest rule.", encoding="utf-8")
    monkeypatch.setenv(ins.ENV, str(inst))

    systems = []        # every system prompt an engine or the model got

    class Inner:
        name = "inner"

        def complete(self, messages, *, system=None, **kw):
            systems.append(system)
            return "ok"

    class Chat(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            systems.append("\n".join(m.content for m in messages
                                     if m.type == "system"))
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="done"))])

    def fake_run_ingest(source, questions, *, engine=None,
                        vision_engine=None, **kw):
        engine.complete([], system="SHIPPED INGEST PROMPT")
        vision_engine.complete([], system="SHIPPED VOTER PROMPT")
        return subagent_mod.IngestSummary(error="stopped by the test")

    monkeypatch.setattr(engine_mod, "engine_for",
                        lambda obj, model=None: Inner())
    monkeypatch.setattr(subagent_mod, "run_ingest", fake_run_ingest)
    monkeypatch.setattr(document_tools, "resolve_document_source",
                        lambda source, attachments=None: source)
    built = {}
    monkeypatch.setattr(agent_mod, "create_deep_agent", lambda **kw: (
        built.update(kw), type("Agent", (), {})())[1])
    agent = agent_mod.build_deep_agent(
        Chat(responses=[AIMessage(content="x")]), enable_report_ingest=True,
        enable_report_library=True, library_root=str(library))
    # read at build: an edit after it reaches the next build, not this one
    (inst / "report_ingest.md").write_text("Edited later.", encoding="utf-8")
    roles = agent.geotech_build["appended"]["roles"]
    assert roles["report_ingest"]["files"] == ["all.md", "report_ingest.md"]
    assert roles["report_library"]["files"] == ["all.md",
                                                "report_library.md"]
    tools = {t.name: t for t in built["tools"]}
    subs = {s["name"]: s for s in built["subagents"]}

    ingest_tail = TAIL + "Everyone rule.\n\nIngest rule."
    for route in (
            lambda: tools["report_ingest"].func(source="made_up.pdf"),
            lambda: subs["report_ingest"]["runnable"].invoke(
                {"messages": [HumanMessage(content="Read made_up.pdf")]})):
        del systems[:]
        route()
        assert systems == ["SHIPPED INGEST PROMPT" + ingest_tail,
                           "SHIPPED VOTER PROMPT" + ingest_tail]

    library_tail = TAIL + "Everyone rule.\n\nLibrary rule."
    question = "Which reports are there?"
    for route in (
            lambda: tools["report_library"].func(question=question),
            lambda: subs["report_library"]["runnable"].invoke(
                {"messages": [HumanMessage(content=question)],
                 "question": question})):
        del systems[:]
        route()
        assert systems and all(s.endswith(library_tail)
                               and s.count(ins.HEADING) == 1
                               for s in systems), systems

    # no instructions: the factory's engine is the one engine_for built
    assert isinstance(agent_mod._report_ingest_engine_factory(object())(),
                      Inner)


def test_the_turn_record_carries_it(folder):
    """``engine_record`` (each turn_start's ``engine``) carries the
    appended record under ``build``; absent when unset."""
    from webapp.core import engine_record
    agent, _, _ = page_builds.build("review_default",
                                    settings={ins.ENV: folder})
    rec = engine_record(agent)["build"]["appended"]
    assert rec["roles"]["main"]["sha"] and rec["roles"]["main"]["chars"]
    plain, _, _ = page_builds.build("review_default")
    assert "appended" not in engine_record(plain).get("build", {})
