"""Helpers reach the conversation's kept page readings (BACKLOG TA-28).

Field session 2026-10-14, turn 3: the helpers looked again at 25 pages that
turns 1 and 2 had already looked at. The turn's note named the 80 kept
readings, but only the primary sees that note, a helper knows only its
brief, and no helper passed ``reuse=true``; one helper's tiles read SPT
tables that matched turn 1's transcription row for row, and it returned a
162-character apology.

``kept_readings`` is a read-only tool every agent with the looking tools
has -- the primary, deepagents' general-purpose helper (it inherits the
primary's tools) and the review page's reading helper. Without ``pdf_page``
it lists the readings BY REFERENCE (page, kind of look, what it was asked,
length, a one-line summary); with ``document`` + ``pdf_page`` it returns one
reading whole, the same result ``reuse=true`` gives.

Offline: readings are kept the way the vision tools keep them, and a
scripted chat model plays the primary and its helper through a real
compiled deep agent. Synthetic names throughout.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from funhouse_agent import vision_tools as vt
from funhouse_agent._fileio import working_dir_bound
from funhouse_agent.deep.agent import build_deep_agent
from funhouse_agent.deep.tools import make_vision_tools

DOC = "site_report.pdf"
PDF = b"%PDF-1.4 synthetic report bytes"
SPT_ROWS = ("S-1 | 1.05-1.50 m | 3-4-6 | N=10\n"
            "S-2 | 2.55-3.00 m | 5-7-9 | N=16\n") * 40


def _reading(text, tiles=0):
    out = {"analysis": text, "view": [0, 0, 612, 792], "pdf_page": None}
    if tiles:
        out["tiles"] = [{"tile": i, "analysis": f"tile {i}: " + SPT_ROWS,
                         "view": [0, 0, 306, 396]} for i in range(tiles)]
    return json.dumps(out)


@pytest.fixture
def kept(tmp_path):
    """A working folder with five kept readings of two documents."""
    vt.clear_read_log()
    with working_dir_bound(str(tmp_path)):
        vt._keep_reading(DOC, PDF, 27, "page", "Transcribe the SPT table",
                         _reading("Boring B-1 log. " + SPT_ROWS))
        vt._keep_reading(DOC, PDF, 28, "page+tiles 2x2",
                         "Transcribe the SPT table on this log",
                         _reading("Boring B-2 log overview.", tiles=4))
        vt._keep_reading(DOC, PDF, 27, vt.ZOOM_VIEW,
                         "Read the sample box depths",
                         _reading("Sample boxes at 1.05-1.50 m and "
                                  "2.55-3.00 m."),
                         region=[40.0, 300.0, 200.0, 420.0])
        vt._keep_reading(DOC, PDF, 22, "page", "Where are the borings?",
                         _reading("Figure 2, exploration plan, 15 symbols."))
        vt._keep_reading("appendix.pdf", b"%PDF other", 4, "page",
                         "Read the lab summary",
                         _reading("Atterberg limits table, 6 rows."))
    yield str(tmp_path)
    vt.clear_read_log()


def _call(folder, **args):
    return json.loads(vt.dispatch_kept_readings(args, folder=folder))


# ---------------------------------------------------------------------------
# The list: by reference, never the readings themselves
# ---------------------------------------------------------------------------

def test_the_list_gives_each_reading_by_reference(kept):
    out = _call(kept)
    assert out["kept"] == 5
    assert out["pages_with_readings"] == {DOC: "23, 28-29",
                                          "appendix.pdf": "5"}
    rows = out["readings"][DOC]
    assert [(r["pdf_page"], r["kind"]) for r in rows] == [
        (23, "page"), (28, "page"), (28, "zoom"), (29, "page+tiles 2x2")]
    zoom = rows[2]
    assert zoom["bbox"] == [40.0, 300.0, 200.0, 420.0]
    assert zoom["asked"] == "Read the sample box depths"
    first = rows[1]
    assert first["summary"].startswith("Boring B-1 log. S-1 | 1.05-1.50 m")
    assert len(first["summary"]) <= vt.KEPT_SUMMARY_CHARS
    assert first["chars"] > 2000                 # its length, not its text
    # small: the rows are referenced, not pasted
    assert SPT_ROWS not in json.dumps(out)
    assert len(json.dumps(out)) < 3000
    assert "reuse=true" in out["how"] and "pdf_page=N" in out["how"]


def test_the_list_narrows_by_document_pages_and_kind(kept):
    out = _call(kept, document="SITE_REPORT.PDF", pdf_pages="28-29")
    assert out["kept"] == 3 and list(out["readings"]) == [DOC]
    assert {r["pdf_page"] for r in out["readings"][DOC]} == {28, 29}
    zooms = _call(kept, kind="zoom")
    assert zooms["kept"] == 1
    tiled = _call(kept, kind="tiles", document="site_report")  # a part names it
    assert [r["pdf_page"] for r in tiled["readings"][DOC]] == [29]
    assert list(_call(kept, pdf_pages=[5])["readings"]) == ["appendix.pdf"]


def test_a_long_list_pages_on(kept, monkeypatch):
    monkeypatch.setattr(vt, "KEPT_LIST_MAX", 2)
    one = _call(kept)
    assert one["showing"] == "1-2 of 5" and one["next_offset"] == 2
    two = _call(kept, offset=2)
    assert two["showing"] == "3-4 of 5" and two["next_offset"] == 4
    last = _call(kept, offset=4)
    assert last["showing"] == "5-5 of 5" and "next_offset" not in last


# ---------------------------------------------------------------------------
# One reading, whole, on request
# ---------------------------------------------------------------------------

def test_one_page_comes_back_whole_as_reuse_gives_it(kept):
    out = _call(kept, document=DOC, pdf_page=28)
    assert out["document"] == DOC and out["pdf_page"] == 28
    assert out["kind"] == "page"
    assert out["analysis"] == "Boring B-1 log. " + SPT_ROWS
    assert "Transcribe the SPT table" in out["reused"]
    # the page's zoom is named, by reference
    (other,) = out["also_kept_for_this_page"]
    assert other["kind"] == "zoom" and "summary" not in other
    # the same reading analyze_pdf_page(reuse=true) hands back
    with working_dir_bound(kept):
        earlier = vt._earlier_reading(PDF, 27, "off")
    assert json.loads(vt._reused_result(earlier))["analysis"] == \
        out["analysis"]


def test_a_zoom_by_its_box_and_a_tiled_page(kept):
    zoom = _call(kept, document=DOC, pdf_page=28,
                 bbox=[40, 300, 200, 420])
    assert zoom["kind"] == "zoom" and "1.05-1.50 m" in zoom["analysis"]
    assert "Read the sample box depths" in zoom["reused"]
    tiled = _call(kept, document=DOC, pdf_page=29)
    assert tiled["kind"] == "page+tiles 2x2" and len(tiled["tiles"]) == 4


def test_what_is_not_kept_is_said_plainly(kept):
    gone = _call(kept, document=DOC, pdf_page=40)
    assert "no reading of p. 40" in gone["error"]
    assert gone["pages_with_readings"][DOC] == "23, 28-29"
    assert "analyze_pdf_page" in gone["hint"]
    nodoc = _call(kept, document="other.pdf")
    assert "no readings of 'other.pdf'" in nodoc["error"]
    nozoom = _call(kept, document=DOC, pdf_page=23, kind="zoom")
    assert "no kept zoom of p. 23" in nozoom["error"]
    assert nozoom["kept_for_this_page"][0]["kind"] == "page"
    bad = _call(kept, kind="sideways")
    assert "is not one of page, tiles, zoom" in bad["error"]
    assert "error" in _call(kept, pdf_pages="0-3")       # 1-based


def test_a_page_in_two_documents_asks_which(kept):
    with working_dir_bound(kept):
        vt._keep_reading("appendix.pdf", b"%PDF other", 22, "page",
                         "Read the plan", _reading("Another plan."))
    out = _call(kept, pdf_page=23)
    assert "2 documents" in out["error"]
    assert set(out["documents"]) == {DOC, "appendix.pdf"}


def test_nothing_kept_says_how_readings_get_kept(tmp_path):
    vt.clear_read_log()
    out = _call(str(tmp_path))
    assert out["kept"] == 0 and "analyze_pdf_page" in out["note"]


# ---------------------------------------------------------------------------
# The tool, and who has it
# ---------------------------------------------------------------------------

def test_the_tool_describes_itself_and_reads_the_bound_folder(kept):
    tools = {t.name: t for t in make_vision_tools()}
    tool = tools["kept_readings"]
    assert tool.description == vt.KEPT_READINGS_DESCRIPTION
    assert "helpers have this tool too" in tool.description
    assert "Read-only" in tool.description
    with working_dir_bound(kept):
        listed = json.loads(tool.invoke({"pdf_pages": "28"}))
        whole = json.loads(tool.invoke({"document": DOC, "pdf_page": 28}))
    assert listed["kept"] == 2
    assert whole["analysis"].startswith("Boring B-1 log.")
    # unknown arguments are refused by name, like every primary tool
    with working_dir_bound(kept):
        refused = json.loads(tool.invoke({"page": 3}))
    assert "has no argument 'page'" in refused["error"]


def test_the_review_page_offers_it_only_with_its_switch(monkeypatch):
    """The geotech page always has kept_readings; the testers' review page
    waits for the suite (owner's rule: new behaviour behind an OFF switch
    until measured), arm ``kept_readings``."""
    from funhouse_agent import review_flags
    from funhouse_agent.deep.review_agent import (
        KEPT_READINGS_TOOL, READER_TOOLS, REVIEW_TOOLS)
    assert KEPT_READINGS_TOOL == "kept_readings"
    assert "kept_readings" not in REVIEW_TOOLS + READER_TOOLS
    monkeypatch.delenv(review_flags.KEPT_READINGS_ENV, raising=False)
    assert not review_flags.kept_readings_on_review_page()
    monkeypatch.setenv(review_flags.KEPT_READINGS_ENV, "1")
    assert review_flags.kept_readings_on_review_page()
    assert review_flags.ARMS["kept_readings"] == {
        review_flags.KEPT_READINGS_ENV: "1"}


def _first_human(messages) -> str:
    for m in messages:
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


class Played(BaseChatModel):
    """The primary delegates one brief; the helper lists the kept readings
    for its pages, takes one whole, and answers from it."""

    seen: dict = {}

    @property
    def _llm_type(self):
        return "played"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        first = _first_human(messages)
        n_ai = sum(1 for m in messages if isinstance(m, AIMessage))
        if first.startswith("PRIMARY"):
            reply = (AIMessage(content="", tool_calls=[{
                "name": "task", "id": "task_0",
                "args": {"description": "HELPER transcribe the SPT on "
                                        "site_report.pdf pages 28-29",
                         "subagent_type": "general-purpose"}}])
                if n_ai == 0 else AIMessage(content="Done."))
            return ChatResult(generations=[ChatGeneration(message=reply)])
        results = [m for m in messages if isinstance(m, ToolMessage)]
        self.seen["helper_results"] = [m.content for m in results]
        if n_ai == 0:
            reply = AIMessage(content="", tool_calls=[{
                "name": "kept_readings", "id": "k1",
                "args": {"document": DOC, "pdf_pages": "28-29"}}])
        elif n_ai == 1:
            reply = AIMessage(content="", tool_calls=[{
                "name": "kept_readings", "id": "k2",
                "args": {"document": DOC, "pdf_page": 28}}])
        else:
            reply = AIMessage(content="From the kept reading of p. 28: "
                                      "S-1 N=10, S-2 N=16.")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def test_a_helper_reaches_earlier_readings_without_looking_again(kept):
    """The field session's gap, closed: a general-purpose helper, briefed
    only with its pages, finds the readings an earlier turn kept and takes
    one whole -- no vision call is made (there is no engine to make one)."""
    model = Played(seen={})
    agent = build_deep_agent(model=model, reference_mode="off")
    with working_dir_bound(kept):
        state = agent.invoke({"messages": [{"role": "user",
                                            "content": "PRIMARY go"}]},
                             config={"recursion_limit": 60})
    listed, whole = model.seen["helper_results"]
    listed = json.loads(listed)
    assert listed["pages_with_readings"] == {DOC: "28-29"}
    assert SPT_ROWS not in json.dumps(listed)             # by reference
    whole = json.loads(whole)
    assert whole["analysis"] == "Boring B-1 log. " + SPT_ROWS
    task = [m for m in state["messages"] if isinstance(m, ToolMessage)
            and m.tool_call_id == "task_0"]
    assert task and "S-1 N=10" in str(task[0].content)
