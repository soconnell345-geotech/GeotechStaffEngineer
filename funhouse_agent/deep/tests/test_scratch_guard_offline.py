"""Scratch-filesystem guard and the delegation rules that go with it.

Field feedback 2026-09-15 (Nairobi SOE re-run), N1/N12/N14: agents ran
deepagents' scratch ``grep``/``read_file`` on real files about 35 times and
got "No matches found" / "not found"; a calc sub-agent that could not "find"
its own earlier report rebuilt it with placeholder text where the numbers
belonged; and a sub-agent renamed soil layers and changed the wrong ones.
"""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from funhouse_agent.deep.scratch_guard import ScratchFilesystemGuard, looks_real


def _req(name, **args):
    return SimpleNamespace(tool_call={"name": name, "args": args, "id": "c1"},
                           state={"files": {"/notes/a.md": {}}})


def _run(req):
    called = []

    def handler(r):
        called.append(r)
        return ToolMessage(content="scratch", tool_call_id="c1")

    return ScratchFilesystemGuard().wrap_tool_call(req, handler), bool(called)


def test_read_file_on_a_real_file_is_redirected(tmp_path):
    f = tmp_path / "SOE_sensitivity_review.html"
    f.write_text("<td>29.878</td>")
    out, called = _run(_req("read_file", file_path=str(f)))
    assert not called
    assert out.status == "error" and "read_text_file" in out.content


def test_grep_under_a_real_root_is_redirected():
    out, called = _run(_req("grep", pattern="PYWall",
                            path="/root/.geotech_webapp/conversations/x/files"))
    assert not called and "search_document" in out.content


def test_write_file_to_tmp_is_redirected():
    out, called = _run(_req("write_file", file_path="/tmp/report.html",
                            content="x"))
    assert not called and "save_file" in out.content


@pytest.mark.parametrize("name", ["ls", "glob"])
def test_listing_a_real_folder_points_to_list_files(name):
    out, called = _run(_req(name, path="/Workspace/Users/x", pattern="*.pdf"))
    assert not called and "list_files" in out.content


@pytest.mark.parametrize("name, args", [
    ("read_file", {"file_path": "/notes/a.md"}),        # the agent's own note
    ("ls", {"path": "/notes"}),
    ("write_file", {"file_path": "/draft.md", "content": "x"}),
    ("read_file", {"file_path": "/memories/AGENTS.md"}),
    ("read_file", {"file_path": "/large_tool_results/abc"}),
    ("grep", {"pattern": "x"}),                          # no path
    ("call_agent", {"agent_name": "soe"}),               # not a scratch tool
])
def test_scratch_calls_pass_through(name, args):
    out, called = _run(_req(name, **args))
    assert called and out.content == "scratch"


def test_async_path(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x")

    async def handler(r):
        return ToolMessage(content="scratch", tool_call_id="c1")

    out = asyncio.run(ScratchFilesystemGuard().awrap_tool_call(
        _req("read_file", file_path=str(f)), handler))
    assert "read_text_file" in out.content


def test_looks_real():
    assert looks_real("/tmp/x") and looks_real("/Volumes/a/b")
    assert looks_real("C:\\Users\\x") and looks_real("/Workspace")
    assert not looks_real("/") and not looks_real("/notes.md")
    assert not looks_real("/tmpfile.md")


def test_every_agent_in_the_build_carries_the_guard(monkeypatch):
    pytest.importorskip("deepagents")
    import funhouse_agent.deep.agent as deep_agent
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel)
    from langchain_core.messages import AIMessage
    captured = {}
    monkeypatch.setattr(deep_agent, "create_deep_agent",
                        lambda **kw: captured.update(kw) or object())
    deep_agent.build_deep_agent(
        GenericFakeChatModel(messages=iter([AIMessage(content="ok")])),
        enable_calc_subagent=True, enable_setup_agent=True)
    assert sum(isinstance(m, ScratchFilesystemGuard)
               for m in captured["middleware"]) == 1
    specs = {s["name"]: s for s in captured["subagents"]}
    assert set(specs) >= {"references", "reviewer", "calc", "model_setup",
                          "general-purpose"}
    for name, spec in specs.items():
        assert sum(isinstance(m, ScratchFilesystemGuard)
                   for m in spec["middleware"]) == 1, name
    assert "tools" not in specs["general-purpose"]  # inherits primary tools


def test_calc_reads_real_text_and_carries_the_rules():
    from funhouse_agent.deep.agent import (_CALC_DELEGATION_NUDGE,
                                           build_calc_subagent)
    spec = build_calc_subagent()
    assert "read_text_file" in [t.name for t in spec["tools"]]
    prompt = spec["system_prompt"]
    assert "NEVER WRITE PLACEHOLDERS" in prompt
    assert "per prior analysis" in prompt          # the phrase it wrote
    assert "LAYER NAMES" in prompt
    assert "NO memory" in _CALC_DELEGATION_NUDGE
    assert "check the numbers are in it" in _CALC_DELEGATION_NUDGE
