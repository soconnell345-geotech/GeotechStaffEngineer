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


def test_report_ingest_and_library_blocks_are_recorded(tmp_path, monkeypatch):
    """The two compiled helpers: the library's prompt takes its block, the
    ingest's engine appends its block to every call, both recorded."""
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel)
    from langchain_core.messages import AIMessage
    from funhouse_agent.deep import agent as agent_mod
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

    seen = {}

    def fake_library(model, **kw):
        seen["library_prompt"] = kw.get("extra_system_prompt")
        return {"name": "report_library", "description": "d",
                "runnable": object()}

    import report_ingest.library_agent as library_agent
    monkeypatch.setattr(library_agent, "build_report_library_subagent",
                        fake_library)
    monkeypatch.setattr(agent_mod, "create_deep_agent",
                        lambda **kw: type("Agent", (), {})())
    model = FakeMessagesListChatModel(responses=[AIMessage(content="ok")])
    agent = agent_mod.build_deep_agent(
        model, enable_report_ingest=True, enable_report_library=True,
        library_root=str(library))
    roles = agent.geotech_build["appended"]["roles"]
    assert roles["report_library"]["files"] == ["all.md",
                                                "report_library.md"]
    assert seen["library_prompt"].endswith("Everyone rule.\n\nLibrary rule.")
    assert seen["library_prompt"].startswith(ins.PROTECTED_RULES)
    assert roles["report_ingest"]["files"] == ["all.md", "report_ingest.md"]

    # the ingest's engine: its system prompt gets the block on every call
    calls = {}

    class Inner:
        name = "inner"

        def complete(self, messages, *, system=None, **kw):
            calls["system"] = system
            return "ok"

    import report_ingest.engine as engine_mod
    monkeypatch.setattr(engine_mod, "engine_for",
                        lambda obj, model=None: Inner())
    engine = agent_mod._report_ingest_engine_factory(object())()
    engine.complete([], system="SHIPPED INGEST PROMPT")
    assert calls["system"] == ("SHIPPED INGEST PROMPT" + TAIL
                               + "Everyone rule.\n\nIngest rule.")
    assert engine.appended["files"] == ["all.md", "report_ingest.md"]

    monkeypatch.delenv(ins.ENV)
    plain = agent_mod._report_ingest_engine_factory(object())()
    assert isinstance(plain, Inner)
