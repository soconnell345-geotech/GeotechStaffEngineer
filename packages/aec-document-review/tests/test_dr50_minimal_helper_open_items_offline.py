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

A second turn hands one job to helper V, which zooms the way the arm's own
prompt asks: ``view`` + ``image_box``, with no question. A zoom that fails
and is not redone is listed; one redone at the same ``view`` +
``image_box`` is not; one followed by a good zoom elsewhere on the page
stays listed, and so does one followed by a good zoom in a merely
overlapping place (with no question, no redo rule settles it: the note
names more than it must, the safe side).

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
JOB_V = (f"LOOKING JOB V: look at pdf pages 3-6 of {DOC} and read each "
         "page's table.")
ANSWER_V = "Findings: none on these pages.\nCould not check: nothing."
VIEW = [0, 0, 612, 792]              # a whole page's view, in page points
IB_FAILS = [400, 300, 600, 420]      # pages 3-6: the first render fails
IB_FAR = [50, 800, 250, 950]         # far from it on the same page
IB_OVERLAP = [500, 380, 700, 500]    # overlaps it
#: 0-based page -> a point in page points: the first render of a window
#: that holds it fails (helper V's zooms; set by its test only)
FAIL_ONCE: dict = {}


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
    jobs: list = [("T", JOB_T), ("L", JOB_L)]

    @property
    def _llm_type(self):
        return "played"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        first = next((_text(m) for m in messages
                      if isinstance(m, HumanMessage)), "")
        results = [m for m in messages if isinstance(m, ToolMessage)]
        who = next((w for w in ("T", "L", "V")
                    if first.startswith(f"LOOKING JOB {w}")), "main")
        self.seen.append((who, list(messages)))
        msg = self._step(who, len(results))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _step(self, who, n):
        if who == "main":
            if n < len(self.jobs):
                w, job = self.jobs[n]
                return _call("task", {"description": job,
                                      "subagent_type": "page_looker"},
                             f"task_{w}")
            return AIMessage(content="Done.")
        if who == "V":
            # the arm's own shape: view + image_box, no question
            def zoom(page, image_box, cid):
                return ("render_region", {
                    "attachment_key": DOC, "pdf_page": page, "view": VIEW,
                    "image_box": image_box}, cid)
            steps = [
                zoom(4, IB_FAILS, "zoom_v1"),     # fails, never redone
                zoom(5, IB_FAILS, "zoom_v2"),     # fails
                zoom(5, IB_FAILS, "zoom_v3"),     # the same place: reads
                zoom(6, IB_FAILS, "zoom_v4"),     # fails
                zoom(6, IB_FAR, "zoom_v5"),       # elsewhere: reads
                zoom(3, IB_FAILS, "zoom_v6"),     # fails
                zoom(3, IB_OVERLAP, "zoom_v7"),   # overlapping: reads
            ]
            if n < len(steps):
                return _call(*steps[n])
            return AIMessage(content=ANSWER_V)
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
        point = FAIL_ONCE.get(kw.get("page"))
        if (box is not None and point is not None
                and box[0] <= point[0] <= box[2]
                and box[1] <= point[1] <= box[3]):
            FAIL_ONCE.pop(kw.get("page"))
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
    FAIL_ONCE.clear()
    yield
    FAIL_ONCE.clear()
    inline_store.clear()


def _turn(folder, jobs=(("T", JOB_T), ("L", JOB_L))):
    model = Played(seen=[], jobs=list(jobs))
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
    assert set(got) == {f"task_{w}" for w, _job in jobs}
    # what each helper's tools returned, as it received them
    tools = {}
    for w, _job in jobs:
        helper = [m for who, m in model.seen if who == w][-1]
        tools[w] = {m.tool_call_id: _text(m) for m in helper
                    if isinstance(m, ToolMessage)}
    return got, tools


def test_the_minimal_helper_hands_back_the_open_items_its_tools_flagged(
        tmp_path):
    got, tools = _turn(tmp_path)
    text, look, tools = got["task_T"], got["task_L"], tools["T"]

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


def test_the_minimal_helper_s_own_zooms_by_view_with_no_question(tmp_path):
    x0, y0, x1, y1 = vision_view.image_box_to_page(VIEW, IB_FAILS)
    for pdf_page in (3, 4, 5, 6):
        FAIL_ONCE[pdf_page - 1] = ((x0 + x1) / 2, (y0 + y1) / 2)
    got, tools = _turn(tmp_path, jobs=[("V", JOB_V)])
    tools = tools["V"]

    # what the helper's zooms returned
    for cid in ("zoom_v1", "zoom_v2", "zoom_v4", "zoom_v6"):
        assert json.loads(tools[cid]) == {
            "error": "made-up render failure"}, (cid, tools[cid][:400])
    for cid in ("zoom_v3", "zoom_v5", "zoom_v7"):
        assert "error" not in json.loads(tools[cid]), (cid, tools[cid][:400])

    # page 4: failed, never redone; page 6: failed, then a good zoom far
    # away (it stays; before, any later good zoom by view on the page
    # dropped it); page 3: failed, then a good zoom in an overlapping place
    # (with no question no redo rule settles it: it stays); page 5: redone
    # at the same view + image_box, so not listed
    expected = [f"- {DOC} page {p} (a zoom): render_region failed: made-up "
                "render failure" for p in (4, 6, 3)]
    assert got["task_V"] == (ANSWER_V + "\n\n"
                             + keep_work.open_items_note(expected, None)), \
        got["task_V"]
