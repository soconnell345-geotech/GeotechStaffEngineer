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


# ---------------------------------------------------------------------------
# 2026-10 Foundry eval: '.' is the scratch root, and a missed read says why
# ---------------------------------------------------------------------------

BIG_ID = "3a22beec-e0ce-4ed1-a4f1-f48017855348"


def _request(name, files=None, **args):
    from langchain.agents.middleware.types import ToolCallRequest
    return ToolCallRequest(tool_call={"name": name, "args": args, "id": "c1"},
                           tool=None, state={"files": files or {}},
                           runtime=None)


def _fake_fs(files):
    """A handler that behaves like deepagents' read_file / grep on ``files``."""
    seen = []

    def handler(r):
        seen.append(r.tool_call["args"])
        name, args = r.tool_call["name"], r.tool_call["args"]
        if name == "read_file":
            path = args["file_path"]
            if path in files:
                return ToolMessage(content=f"     1\t{files[path]}",
                                   tool_call_id="c1", name="read_file")
            return ToolMessage(content=f"Error: File '{path}' not found",
                               tool_call_id="c1", name="read_file")
        return ToolMessage(content=f"searched {args.get('path')}",
                           tool_call_id="c1", name=name)

    return handler, seen


@pytest.mark.parametrize("name, arg", [("grep", "path"), ("ls", "path"),
                                       ("glob", "path")])
@pytest.mark.parametrize("dot", [".", "./"])
def test_dot_is_the_scratch_root_not_the_real_disk(name, arg, dot):
    # LP-2: grep(path='.') was refused four times as "on the real disk";
    # and deepagents itself reads '.' as an empty path '/.', so the guard
    # sends it to '/', where the scratch notes and saved results are.
    handler, seen = _fake_fs({})
    out = ScratchFilesystemGuard().wrap_tool_call(
        _request(name, pattern="x", **{arg: dot}), handler)
    assert seen == [{"pattern": "x", arg: "/"}]
    assert out.content == "searched /"


@pytest.mark.parametrize("path", ["/large_tool_results/abc",
                                  "/conversation_history/x.md",
                                  "/memories/AGENTS.md"])
def test_deepagents_own_areas_pass_through(path):
    handler, seen = _fake_fs({path: "data"})
    out = ScratchFilesystemGuard().wrap_tool_call(
        _request("read_file", {path: "data"}, file_path=path), handler)
    assert "data" in out.content and len(seen) == 1


def test_read_by_a_mangled_name_reads_the_saved_result():
    # RDD-2/RDD-4: the model read '<id>.txt'; deepagents saves '<id>' (it
    # writes every '.' in an id as '_', so it never makes a '.txt').
    files = {f"/large_tool_results/{BIG_ID}": "row 0: FOS 1.38"}
    handler, seen = _fake_fs(files)
    out = ScratchFilesystemGuard().wrap_tool_call(
        _request("read_file", files,
                 file_path=f"/large_tool_results/{BIG_ID}.txt", limit=5),
        handler)
    assert seen[-1] == {"file_path": f"/large_tool_results/{BIG_ID}",
                        "limit": 5}
    assert "FOS 1.38" in out.content
    assert out.content.startswith("(Nothing is saved at")


def test_read_of_a_result_that_was_never_saved_says_so():
    handler, _ = _fake_fs({})
    out = ScratchFilesystemGuard().wrap_tool_call(
        _request("read_file", {}, file_path=f"/large_tool_results/{BIG_ID}.txt"),
        handler)
    assert out.status == "error"
    assert "Nothing has been saved there" in out.content
    assert "use that message" in out.content
    assert "Do not report a value you have not read" in out.content


def test_saved_results_are_listed_when_the_name_is_wrong():
    files = {"/large_tool_results/call_12": "x"}
    handler, seen = _fake_fs(files)
    out = ScratchFilesystemGuard().wrap_tool_call(
        _request("read_file", files, file_path="/large_tool_results/call_1"),
        handler)
    # a merely similar name is NEVER served: it would be another call's data
    assert len(seen) == 1
    assert out.status == "error" and "call_12" in out.content


def test_missing_scratch_note_lists_what_exists():
    files = {"/notes/a.md": "x"}
    handler, _ = _fake_fs(files)
    out = ScratchFilesystemGuard().wrap_tool_call(
        _request("read_file", files, file_path="/notes/b.md"), handler)
    assert "not found" in out.content and "/notes/a.md" in out.content
    assert "read_text_file" in out.content


def test_missed_read_async():
    files = {f"/large_tool_results/{BIG_ID}": "row 0"}
    handler, _ = _fake_fs(files)

    async def ahandler(r):
        return handler(r)

    out = asyncio.run(ScratchFilesystemGuard().awrap_tool_call(
        _request("read_file", files,
                 file_path=f"/large_tool_results/{BIG_ID}.txt"), ahandler))
    assert "row 0" in out.content


def test_a_successful_read_is_untouched():
    files = {"/notes/a.md": "hello"}
    handler, seen = _fake_fs(files)
    out = ScratchFilesystemGuard().wrap_tool_call(
        _request("read_file", files, file_path="/notes/a.md"), handler)
    assert out.content == "     1\thello" and len(seen) == 1


def test_offloaded_result_round_trip_through_the_real_agent():
    """End to end on the installed deepagents: a result too large to return
    is saved under /large_tool_results/<id>; reading it by '<id>.txt' gets
    it, reading a SMALL result's id gets the explanation, and grep on '.'
    finds the saved text."""
    pytest.importorskip("deepagents")
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel)
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import StructuredTool

    from funhouse_agent.deep.agent import build_deep_agent

    small_id = "1250dbca-8180-486a-ad07-240d109ff0b9"

    def call(name, args, cid):
        return AIMessage(content="", tool_calls=[
            {"name": name, "args": args, "id": cid, "type": "tool_call"}])

    class Scripted(FakeMessagesListChatModel):
        script: list = []

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            msg = self.script.pop(0) if self.script else AIMessage(
                content="done")
            return ChatResult(generations=[ChatGeneration(message=msg)])

    def big_tool() -> str:
        """A very large result."""
        return "\n".join(f"row {i}: " + "x" * 60 for i in range(3000))

    def small_tool() -> str:
        """A small result."""
        return '{"FOS": 1.38}'

    model = Scripted(responses=[AIMessage(content="x")], script=[
        call("big_tool", {}, BIG_ID),
        call("read_file", {"file_path": f"/large_tool_results/{BIG_ID}.txt",
                           "limit": 3}, "r1"),
        call("small_tool", {}, small_id),
        call("read_file", {"file_path":
                           f"/large_tool_results/{small_id}.txt"}, "r2"),
        call("grep", {"pattern": "row 2999", "path": "."}, "r3"),
    ])
    agent = build_deep_agent(model, extra_tools=[
        StructuredTool.from_function(big_tool, name="big_tool"),
        StructuredTool.from_function(small_tool, name="small_tool")])
    out = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    replies = {m.tool_call_id: m for m in out["messages"]
               if isinstance(m, ToolMessage)}
    assert f"/large_tool_results/{BIG_ID}" in out["files"]
    assert "row 0:" in str(replies["r1"].content)
    assert replies["r2"].status == "error"
    assert "only when it was too large" in str(replies["r2"].content)
    assert BIG_ID in str(replies["r3"].content)


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
