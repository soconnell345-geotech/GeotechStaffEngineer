"""The minimal build's helper hands back the open items its tools flagged,
by the default build's reader's own code (BACKLOG DR-50, part 3), offline.

The page's minimal build (``GEOTECH_REVIEW_AGENT=minimal``) runs one turn
on a scripted model and a fake vision engine, as
``test_dr50_reader_open_items_offline.py`` does for the default build. The
main agent hands two looking jobs to ``task`` helpers. Helper T reads six
pages of text in one call that stops at its size limit, zooms (the render
fails) and zooms again with the same question in a box that overlaps the
first, zooms on another page (the render fails, and it does not redo it),
and looks at an uploaded picture whose vision answer is cut off at its
output limit; it then answers that nothing is open. Helper L looks at one
page and finds nothing wrong. Every value is made up.

The arm's helper has no text tool: it reads by looking. The test adds the
core's real ``read_pdf_text`` to the helper's tools only, to show that the
same code carries a cut text read in this build too.
"""

import json

import pytest

pytest.importorskip("planlens.tools")
pymupdf = pytest.importorskip("pymupdf")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage, HumanMessage, ToolMessage)
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from aec_agent_core import inline_store, review_flags, vision_view  # noqa: E402
from aec_agent_core._fileio import working_dir_bound  # noqa: E402
from aec_agent_core.deep import keep_work  # noqa: E402
from aec_agent_core.deep.vision_engine import VisionAnswer  # noqa: E402
from aec_agent_core.deep.vision_kit import make_vision_tools  # noqa: E402
from aec_document_review import minimal_agent  # noqa: E402
from aec_document_review.review_build import build_review_page_agent  # noqa: E402
from tests.test_review_harness_offline import (  # noqa: E402
    FakeEngine, _review_kwargs)

DOC = "made_up_spec.pdf"
IMG = "made_up_label.png"
JOB_T = (f"LOOKING JOB T: in {DOC}, pdf pages 1-6, list every clause that "
         f"names a test; also read the label in {IMG}.")
JOB_L = f"LOOKING JOB L: look at pdf page 3 of {DOC} and give its heading."
ASK = "Read the made-up heading lettering in this box."
BOX_FAILS = [24, 10, 300, 60]        # page 1: the render fails
BOX_REDO = [20, 8, 320, 70]          # page 1: overlaps it, renders
BOX_FAILS_2 = [30, 100, 200, 160]    # page 2: fails, never redone
ANSWER_T = "Findings: none on these pages.\nCould not check: nothing."
ANSWER_L = "Findings: none on this page.\nCould not check: nothing."


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


def _made_up_picture() -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 240, 90), 0)
    pix.set_rect(pix.irect, (255, 255, 255))
    pix.set_rect(pymupdf.IRect(20, 30, 220, 60), (0, 0, 0))
    return pix.tobytes("png")


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
        who = ("T" if first.startswith("LOOKING JOB T") else
               "L" if first.startswith("LOOKING JOB L") else "main")
        self.seen.append((who, list(messages)))
        msg = self._step(who, len(results))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _step(self, who, n):
        if who == "main":
            if n < 2:
                job, cid = ((JOB_T, "task_T") if n == 0 else
                            (JOB_L, "task_L"))
                return _call("task", {"description": job,
                                      "subagent_type": "page_looker"}, cid)
            return AIMessage(content="Done.")
        if who == "L":
            if n == 0:
                return _call("analyze_pdf_page", {
                    "attachment_key": DOC, "pdf_page": 3,
                    "prompt": "Read the heading."}, "look_l1")
            return AIMessage(content=ANSWER_L)
        steps = [
            ("read_pdf_text", {"source": DOC, "pdf_pages": "1-6"}, "read_t"),
            ("render_region", {"attachment_key": DOC, "pdf_page": 1,
                               "prompt": ASK, "bbox": BOX_FAILS}, "zoom_t1"),
            ("render_region", {"attachment_key": DOC, "pdf_page": 1,
                               "prompt": ASK, "bbox": BOX_REDO}, "zoom_t2"),
            ("render_region", {"attachment_key": DOC, "pdf_page": 2,
                               "prompt": ASK, "bbox": BOX_FAILS_2},
             "zoom_t3"),
            ("analyze_image", {"attachment_key": IMG,
                               "prompt": "Read the made-up label."},
             "look_t"),
        ]
        if n < len(steps):
            return _call(*steps[n])
        return AIMessage(content=ANSWER_T)


def _cut_answer(prompt):
    answer = VisionAnswer("Label: MADE-UP 12")
    answer.cut_off = True
    return answer


@pytest.fixture(autouse=True)
def _minimal(monkeypatch):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    for k, v in review_flags.ARMS["minimal"].items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    # a render of the two failing boxes fails, as a render can
    real_render = vision_view.render_view

    def render(source, *args, **kw):
        box = kw.get("bbox")
        if box is not None and [float(v) for v in box] in (
                [float(v) for v in BOX_FAILS],
                [float(v) for v in BOX_FAILS_2]):
            raise ValueError("made-up render failure")
        return real_render(source, *args, **kw)

    monkeypatch.setattr(vision_view, "render_view", render)
    # the helper's tools, with the core's text read added (see the module
    # docstring); the main agent's tools are the arm's own
    real_tools = minimal_agent._looking_tools

    def tools(engine, attachments, save_fn, markup_author, max_result_chars,
              *, with_output):
        out = real_tools(engine, attachments, save_fn, markup_author,
                         max_result_chars, with_output=with_output)
        if not with_output:
            out += make_vision_tools(
                engine=engine, attachments=attachments,
                include={"read_pdf_text"}, max_result_chars=max_result_chars)
        return out

    monkeypatch.setattr(minimal_agent, "_looking_tools", tools)
    inline_store.clear()
    yield
    inline_store.clear()


def _turn(folder):
    model = Played(seen=[])
    agent = build_review_page_agent(
        model, attachments={DOC: _made_up_spec(), IMG: _made_up_picture()},
        engine=FakeEngine(reply=_cut_answer), **_review_kwargs())
    assert agent.geotech_review_agent == "minimal"
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
    # what helper T's tools returned, as it received them
    helper = [m for who, m in model.seen if who == "T"][-1]
    tools = {m.tool_call_id: _text(m) for m in helper
             if isinstance(m, ToolMessage)}
    return got["task_T"], got["task_L"], tools


def test_the_minimal_helper_hands_back_the_open_items_its_tools_flagged(
        tmp_path):
    text, look, tools = _turn(tmp_path)

    # what the helper's tools returned: a text read cut at its size limit,
    # a zoom that failed and its redo in an overlapping box, a zoom on
    # page 2 that failed, and a picture read whose answer was cut off
    read = json.loads(tools["read_t"])
    assert read.get("truncated") is True, tools["read_t"][:400]
    assert json.loads(tools["zoom_t1"]) == {"error": "made-up render failure"}
    assert "error" not in json.loads(tools["zoom_t2"]), tools["zoom_t2"][:400]
    assert json.loads(tools["zoom_t3"]) == {"error": "made-up render failure"}
    assert "cut_off" in json.loads(tools["look_t"]), tools["look_t"][:400]

    # the main agent gets the helper's answer and, after it, the open items
    # in the default build's reader's own words, though the answer says
    # nothing is open: the failed zoom never redone, the unfinished picture
    # read, the cut text read. No line for the zoom redone in an
    # overlapping box, and no kept file (this build cannot read one).
    nxt = read["pdf_pages_returned"][-1]
    expected = [
        f"- {DOC} page 2 (a zoom): render_region failed: made-up render "
        "failure",
        f"- {IMG}: the vision answer was cut off at its output limit (the "
        "whole image), so the end of that reading is missing",
        f"- {DOC} page {nxt}: a text read (read_pdf_text) was cut at its size "
        "limit, so the rest of the text it asked for was not read (the "
        "tool's note: " + read["truncated_note"].rstrip(".") + ")",
    ]
    assert text == (ANSWER_T + "\n\n"
                    + keep_work.open_items_note(expected, None)), text
    assert f"{DOC} page 1" not in text
    assert "kept/" not in text and "read_text_file" not in text

    # a helper with no open item: its answer exactly as before
    assert look == ANSWER_L

