"""A reading helper's cut text read reaches the main agent, and a zoom it
redid in another box does not (BACKLOG DR-50), offline.

The page's real build runs one turn on a scripted model and a fake vision
engine, as ``test_dr46_reader_boundary_offline.py`` does. Two reading
helpers split a made-up text-heavy set. One reads six pages of text in one
``read_document`` call, which stops at its size limit with a cursor to go
on from, and answers without going on; it also zooms with a box given two
ways (the zoom fails) and zooms again in another box with the same
question. The other looks at one page and finds nothing wrong. Every value
is made up.
"""

import json
import re

import pytest

pytest.importorskip("planlens.tools")
pymupdf = pytest.importorskip("pymupdf")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage, HumanMessage, ToolMessage)
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from aec_agent_core import inline_store, review_flags  # noqa: E402
from aec_agent_core._fileio import working_dir_bound  # noqa: E402
from aec_agent_core.deep import keep_work  # noqa: E402
from aec_document_review.review_build import build_review_page_agent  # noqa: E402
from tests.test_review_harness_offline import (  # noqa: E402
    FakeEngine, _review_kwargs)

DOC = "made_up_spec.pdf"
JOB_TEXT = (f"READING JOB T: open {DOC}, read pdf pages 1-6 and list every "
            "clause that names a test.")
JOB_LOOK = f"READING JOB L: look at pdf page 1 of {DOC} and list its heading."
ASK = "Read the made-up heading lettering in this box."
ANSWER_T = "Findings: none on these pages.\nOpen items: none."
ANSWER_L = "Findings: none on this page.\nOpen items: none."


def _made_up_spec(pages=6, lines=100) -> bytes:
    doc = pymupdf.open()
    for p in range(pages):
        page = doc.new_page(width=612, height=792)
        for i in range(lines):
            page.insert_text(
                (24, 14 + i * 7.6),
                f"Made-up clause {p + 1}.{i:03d}: the quick grey fox notes "
                "nothing", fontsize=6)
    data = doc.tobytes()
    doc.close()
    return data


def _text(m) -> str:
    c = m.content
    if isinstance(c, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                       for b in c)
    return c or ""


def _call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                              "id": cid}])


class Played(BaseChatModel):
    """Plays the main agent and both helpers from what each call shows."""

    seen: list = []

    @property
    def _llm_type(self):
        return "played"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        first = next((_text(m) for m in messages
                      if isinstance(m, HumanMessage)), "")
        results = [m for m in messages if isinstance(m, ToolMessage)]
        who = ("T" if first.startswith("READING JOB T") else
               "L" if first.startswith("READING JOB L") else "main")
        self.seen.append((who, list(messages)))
        msg = self._step(who, results)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _step(self, who, results):
        n = len(results)
        if who == "main":
            if n == 0:
                return AIMessage(content="", tool_calls=[
                    {"name": "task", "id": "task_T", "args": {
                        "description": JOB_TEXT,
                        "subagent_type": "general-purpose"}},
                    {"name": "task", "id": "task_L", "args": {
                        "description": JOB_LOOK,
                        "subagent_type": "general-purpose"}}])
            return AIMessage(content="Done.")
        if who == "L":
            if n == 0:
                return _call("analyze_pdf_page", {
                    "attachment_key": DOC, "pdf_page": 1, "tiles": "off",
                    "prompt": "Read the heading."}, "look_l1")
            return AIMessage(content=ANSWER_L)
        if n == 0:
            return _call("open_document", {"source": DOC}, "open_t")
        if n == 1:
            handle = re.search(r'"handle"\s*:\s*"([^"]+)"',
                               _text(results[0])).group(1)
            return _call("read_document", {"handle": handle,
                                           "pdf_pages": "1-6"}, "read_t")
        if n == 2:
            # a box given two ways: the zoom fails
            return _call("render_region", {
                "attachment_key": DOC, "pdf_page": 1, "prompt": ASK,
                "bbox": [24, 10, 300, 60], "view": [0, 0, 612, 792],
                "image_box": [0, 0, 500, 80]}, "zoom_t1")
        if n == 3:
            # the same question in another box
            return _call("render_region", {
                "attachment_key": DOC, "pdf_page": 1, "prompt": ASK,
                "bbox": [20, 8, 320, 70]}, "zoom_t2")
        return AIMessage(content=ANSWER_T)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    inline_store.clear()
    yield
    inline_store.clear()


def _turn(folder):
    model = Played(seen=[])
    agent = build_review_page_agent(
        model, attachments={DOC: _made_up_spec()},
        engine=FakeEngine(reply="Heading: MADE-UP SPECIFICATION."),
        **_review_kwargs())
    with working_dir_bound(str(folder)):
        out = agent.invoke(
            {"messages": [{"role": "user",
                           "content": "Review the specification."}]},
            config={"recursion_limit": 200})
    assert out["messages"][-1].content == "Done."
    last = [m for who, m in model.seen if who == "main"][-1]
    got = {m.tool_call_id: _text(m) for m in last
           if isinstance(m, ToolMessage)}
    assert set(got) == {"task_T", "task_L"}
    # what the text helper's tools returned, as it received them
    helper = [m for who, m in model.seen if who == "T"][-1]
    tools = {m.tool_call_id: _text(m) for m in helper
             if isinstance(m, ToolMessage)}
    return got["task_T"], got["task_L"], tools


CUT = re.compile(
    r"\n- (\S+) page (\d+): a text read \(read_document\) was cut at its "
    r"size limit, so the rest of the text it asked for was not read \(the "
    r"tool's cursor to go on from: next=(\{[^\n]*\})\)")


def test_a_cut_text_read_reaches_the_main_agent_and_a_redone_zoom_does_not(
        tmp_path):
    text, look, tools = _turn(tmp_path)

    # the helper's text read stopped at its size limit, with a cursor
    read = json.loads(tools["read_t"])
    assert isinstance(read.get("next"), dict), tools["read_t"][:400]
    # its first zoom failed, its second (another box, same question) read
    assert "error" in tools["zoom_t1"], tools["zoom_t1"][:400]
    assert "error" not in tools["zoom_t2"][:200], tools["zoom_t2"][:400]

    # 1. the cut reaches the main agent, with the tool's own cursor, though
    # the helper's answer says "Open items: none."
    assert text.startswith(ANSWER_T)
    found = CUT.search(text)
    assert found, text
    assert found.group(1) == read["handle"]
    assert json.loads(found.group(3)) == read["next"]
    first = int(re.match(r"\d+", str(read["next"]["pages"])).group(0))
    assert int(found.group(2)) == first + 1

    # 2. the zoom it redid is not listed as unread (the second zoom's own
    # flags, such as lettering past its box's edges, still are)
    assert "render_region failed" not in text
    assert "failed" not in text.split(keep_work.OPEN_ITEMS_HEADING, 1)[1]

    # 3. a helper with no cut text and no failed look: its answer and its
    # kept-file note only, as before
    assert keep_work.OPEN_ITEMS_HEADING not in look
    assert re.fullmatch(
        re.escape(ANSWER_L) + r"\n\n\[This reading helper's whole record "
        r"\(1 tool results, [\d,]+ characters\) is kept in the file "
        r"'kept/reader_[^']+\.md' in the working folder\. Read it with "
        r"read_text_file -- its index gives the offset of each result -- to "
        r"compare what it read with other pages, instead of reading the "
        r"same pages again\.\]", look), look
