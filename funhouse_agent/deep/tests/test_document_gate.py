"""The document and drawing tools wait for a document on the geotech page
(BACKLOG INF-67).

On a geotech turn with no document they are off the model's tool list;
an attached document, or one fetched mid-turn, puts them on from the next
model call (of the same turn); the Document Review page always has them.
Every model here is scripted: no model call, no network, made-up files.
"""

from __future__ import annotations

import os

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from funhouse_agent import review_flags
from funhouse_agent._fileio import working_dir_bound
from funhouse_agent.deep.agent import build_deep_agent
from funhouse_agent.deep.document_gate import (DOCUMENT_TOOLS,
                                               DocumentToolsGate)

#: Kept on the list without a document: the prompt has the agent read back
#: a PDF it built itself.
ALWAYS = {"read_pdf_text", "analyze_pdf_page"}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("GEOTECH_DEFAULT_OUTPUT_DIR", raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


def _scripted(replies, menus):
    class Scripted(BaseChatModel):
        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kw):
            menus.append({t.get("name") if isinstance(t, dict)
                          else getattr(t, "name", None) for t in tools})
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            text, calls = replies.pop(0) if replies else ("done", [])
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content=text, tool_calls=[
                    {"name": n, "args": a, "id": f"c{len(menus)}_{i}"}
                    for i, (n, a) in enumerate(calls)]))])

    return Scripted()


def _run(agent, folder):
    with working_dir_bound(str(folder)):
        return agent.invoke(
            {"messages": [{"role": "user", "content": "Ka for phi 30?"}]},
            config={"recursion_limit": 40})


def _doc_tools_in(menu):
    return set(menu) & DOCUMENT_TOOLS


def test_a_geotech_turn_with_no_document_has_none_of_them(tmp_path):
    menus = []
    agent = build_deep_agent(_scripted([("Ka = 0.333.", [])], menus))
    # registered, so a call by name still runs; only the list is shorter
    registered = set(agent.nodes["tools"].bound.tools_by_name)
    assert {"open_document", "annotate_document", "render_region"} \
        <= registered
    _run(agent, tmp_path)
    assert len(menus) == 1
    assert not _doc_tools_in(menus[0]), _doc_tools_in(menus[0])
    assert ALWAYS <= menus[0]
    assert "call_agent" in menus[0]


def test_an_attached_document_turns_them_on(tmp_path):
    menus = []
    agent = build_deep_agent(_scripted([("Opened.", [])], menus),
                             attachments={"made_up.pdf": b"%PDF-1.4 x"})
    _run(agent, tmp_path)
    assert _doc_tools_in(menus[0]) == DOCUMENT_TOOLS & set(
        agent.nodes["tools"].bound.tools_by_name)
    assert {"open_document", "annotate_document"} <= menus[0]


def test_an_upload_after_the_build_turns_them_on(tmp_path):
    """The web app adds uploads to the SAME dict the agent was built with."""
    menus = []
    attachments = {}
    agent = build_deep_agent(_scripted([("a", []), ("b", [])], menus),
                             attachments=attachments)
    _run(agent, tmp_path)
    attachments["made_up.png"] = b"\x89PNG made up"
    _run(agent, tmp_path)
    assert not _doc_tools_in(menus[0])
    assert "open_document" in menus[1]


def test_a_document_fetched_mid_turn_turns_them_on_for_the_next_call(
        tmp_path):
    """A fetch tool's result is in the turn's messages: the next model call
    of the same turn has the document tools."""

    @tool
    def sharepoint_download_file(path: str, save_as: str = "",
                                 refresh: bool = False) -> str:
        """Download a made-up file."""
        (tmp_path / "made_up.pdf").write_bytes(b"%PDF-1.4 x")
        return "Downloaded 'made_up.pdf' into the working folder."

    menus = []
    replies = [("", [("sharepoint_download_file", {"path": "x/made_up.pdf"})]),
               ("", [("open_document", {"source": "made_up.pdf"})]),
               ("Read it.", [])]
    agent = build_deep_agent(_scripted(replies, menus),
                             extra_tools=[sharepoint_download_file])
    _run(agent, tmp_path)
    assert len(menus) == 3, "one turn, three model calls"
    assert not _doc_tools_in(menus[0])
    assert "open_document" in menus[1] and "annotate_document" in menus[1]
    assert "open_document" in menus[2]


def test_the_hosts_signal_turns_them_on_mid_turn(tmp_path):
    """The web app's signal (its downloads ledger) flips inside a tool; the
    next model call of the turn has the document tools."""
    held = []

    @tool
    def fetch_made_up_file(name: str) -> str:
        """Fetch a made-up file by some route the gate has no name for."""
        held.append(name)
        return "fetched"

    menus = []
    replies = [("", [("fetch_made_up_file", {"name": "made_up.pdf"})]),
               ("Done.", [])]
    agent = build_deep_agent(_scripted(replies, menus),
                             extra_tools=[fetch_made_up_file],
                             document_signal=lambda: bool(held))
    _run(agent, tmp_path)
    assert not _doc_tools_in(menus[0])
    assert "open_document" in menus[1]


def test_no_working_folder_bound_hides_nothing():
    """Library or notebook use: a document may be any path."""
    menus = []
    agent = build_deep_agent(_scripted([("ok", [])], menus))
    agent.invoke({"messages": [{"role": "user", "content": "hi"}]},
                 config={"recursion_limit": 40})
    assert {"open_document", "annotate_document"} <= menus[0]


def test_a_signal_that_raises_offers_the_tools(tmp_path):
    def broken():
        raise OSError("made-up failure")

    gate = DocumentToolsGate({}, signal=broken)
    with working_dir_bound(str(tmp_path)):
        assert gate.holds_document([]) is True


def test_the_review_page_keeps_them(tmp_path):
    from webapp.profiles import DOCUMENT_REVIEW
    menus = []
    agent = build_deep_agent(_scripted([("Hello.", [])], menus),
                             **DOCUMENT_REVIEW.build_kwargs())
    _run(agent, tmp_path)
    assert {"open_document", "read_document", "search_document",
            "render_region", "annotate_document"} <= menus[-1]


# ---------------------------------------------------------------------------
# The web app's signal (webapp.geotech_page.document_signal)
# ---------------------------------------------------------------------------

def test_web_app_signal_reads_the_downloads_ledger(tmp_path):
    from webapp import core
    from webapp.geotech_page import document_signal
    conv = tmp_path / "conv"
    files = conv / "files"
    files.mkdir(parents=True)
    holds = document_signal(str(files))
    with working_dir_bound(str(files)):
        assert holds() is False
        local = files / "made_up.pdf"
        local.write_bytes(b"%PDF-1.4 x")
        core.record_download(str(conv), "Shared/made_up.pdf", str(local), 10)
        assert holds() is True


def test_web_app_signal_counts_a_folder_the_user_chose(tmp_path):
    from webapp.geotech_page import document_signal
    files = tmp_path / "conv" / "files"
    files.mkdir(parents=True)
    chosen = tmp_path / "their_project"
    chosen.mkdir()
    holds = document_signal(str(files))
    with working_dir_bound(str(chosen)):
        assert holds() is True
    with working_dir_bound(str(files)):
        assert holds() is False
    assert os.path.isdir(chosen)
