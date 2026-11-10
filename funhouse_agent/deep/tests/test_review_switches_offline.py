"""The review switches and the inline image tools - offline, fake models.

What is left of the M1 tests (findings, overview) after BACKLOG DR-39
removed the findings tools, the overview orientation and the lean build:
the suite's arms are exactly what ``review_flags.ARMS`` says, an arm clears
the settings it does not name, and an image FILE given to ``analyze_image``
with ``inline_image_files`` (the ``minimal`` build) is shown to the main
model rather than described.
"""

import json
import os

import pytest

pytest.importorskip("planlens.tools")
pymupdf = pytest.importorskip("pymupdf")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel)
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from funhouse_agent import (  # noqa: E402
    document_tools, inline_store, review_flags, vision_tools)
from funhouse_agent.deep.agent import build_deep_agent  # noqa: E402
from funhouse_agent.deep.tools import make_vision_tools  # noqa: E402
from planlens.testing import build_synthetic_review_document  # noqa: E402

GEOTECH_WORDS = ("drawing_ir", "digitize_drawing", "query_drawing",
                 "snip_region", "call_agent", "figure_db")


@pytest.fixture(autouse=True)
def _no_switches(monkeypatch, tmp_path):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    monkeypatch.delenv("GEOTECH_REVIEW_MAX_MODEL_CALLS", raising=False)
    monkeypatch.delenv(document_tools.MARKUP_AUTHOR_ENV, raising=False)
    # the working folder: where findings.json and every output lands
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(work))


@pytest.fixture(scope="module")
def gt():
    return build_synthetic_review_document()


def _review_kwargs():
    from webapp.profiles import DOCUMENT_REVIEW
    return DOCUMENT_REVIEW.build_kwargs()


class Scripted(FakeMessagesListChatModel):
    """Plays ``script`` (AIMessages or callables of the request messages) and
    records what it was sent."""

    script: list = []
    seen: list = []

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        step = self.script.pop(0) if self.script else AIMessage(content="done")
        msg = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _model(script):
    return Scripted(responses=[AIMessage(content="x")], script=list(script),
                    seen=[])


def _call(name, args, i=1):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                              "id": f"call_{name}_{i}"}])


def _tools_of(agent):
    return agent.nodes["tools"].bound.tools_by_name


class FakeEngine:
    accepts_jpeg = False

    def __init__(self, reply="seen"):
        self.reply = reply
        self.prompts = []

    def vision_profile(self):
        return None

    def analyze_image(self, image, prompt):
        self.prompts.append(prompt)
        return self.reply


# ---------------------------------------------------------------------------
# The contact sheet shown to the main model (the minimal build's inline images)
# ---------------------------------------------------------------------------

@pytest.fixture
def sheet_png(gt, tmp_path):
    from planlens.document import Document
    doc = Document(content=gt.pdf)
    try:
        png, _info = doc.render_thumbnails()[0]
    finally:
        doc.close()
    # in the working folder: the read tools read the conversation's files
    path = tmp_path / "work" / "contact_sheet_1.png"
    path.write_bytes(png)
    return str(path)


def _analyze_image(inline, attachments=None):
    eng = FakeEngine()
    tool = next(t for t in make_vision_tools(
        engine=eng, attachments=attachments or {}, inline_images=inline,
        inline_image_files=inline)
        if t.name == "analyze_image")
    return tool, eng


def test_inline_analyze_image_of_a_file_is_shown_not_described(sheet_png):
    inline_store.clear()
    tool, eng = _analyze_image(True)
    out = json.loads(tool.invoke({"attachment_key": sheet_png,
                                  "prompt": "which pages are drawings?"}))
    assert eng.prompts == []                       # no one-shot call
    data, meta = inline_store.get(out["image_id"])
    assert data[:4] == b"\x89PNG" and "contact_sheet_1.png" in meta["label"]
    assert "look at it yourself" in out["note"]
    # an upload named by its key keeps the vision call (and, since VIS-09,
    # the image's view for zooming, never an image_id)
    with open(sheet_png, "rb") as fh:
        tool, eng = _analyze_image(True, {"photo.png": fh.read()})
    out = json.loads(tool.invoke({"attachment_key": "photo.png"}))
    assert out["analysis"] == "seen" and "image_id" not in out
    assert out["view"][:2] == [0, 0] and len(eng.prompts) == 1
    # switch off: unchanged (the same read again in this conversation would
    # be served as a repeat, B6 — forget it so the call is made)
    from funhouse_agent.vision_tools import clear_repeat_reads
    clear_repeat_reads()
    tool, eng = _analyze_image(False)
    out = json.loads(tool.invoke({"attachment_key": sheet_png}))
    assert out["analysis"] == "seen" and "image_id" not in out
    assert len(eng.prompts) == 1


def test_inline_analyze_image_refuses_what_is_not_an_image(tmp_path):
    """A text file is not shown and not sent as a picture: it is refused,
    naming the tool that reads it (wave 2b, C6: every file used to go out
    labelled PNG, and an .xlsx came back as a 400 from the API)."""
    not_image = tmp_path / "work" / "notes.txt"
    not_image.write_text("plain text", encoding="utf-8")
    tool, eng = _analyze_image(True)
    out = json.loads(tool.invoke({"attachment_key": str(not_image)}))
    assert "read_text_file" in out["error"] and eng.prompts == []


def test_page_images_are_labelled_as_before():
    from funhouse_agent.deep.inline_images import image_message
    inline_store.clear()
    page = inline_store.put(b"\x89PNG page", {"page": 2, "pdf_page": 3,
                                              "view": [0, 0, 10, 10]})
    msg = image_message([page])
    assert msg.content[1]["text"] == (f"Image {page}: PDF page 3 (tool page "
                                      f"2), view [0, 0, 10, 10]")


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------

def test_the_arms_are_the_baseline_and_one_change_each():
    A = review_flags
    assert A.ARMS == {
        "baseline": {},
        # 5.32: the looking-only measuring stick, test_review_minimal_offline
        "minimal": {A.AGENT_ENV: "minimal"},
        # plan W4's report checklist (the geotech page only since RI-34;
        # the coverage ledger it once came with went in DR-40)
        "checklist": {A.CHECKLIST_ENV: "1"},
        # EV-18 cited pages no tool opened, pinned in
        # funhouse_agent/tests/test_cited_pages.py
        "cited_pages_note": {A.CITED_PAGES_ENV: "note"},
        "cited_pages_retry": {A.CITED_PAGES_ENV: "retry"},
        # GM-92 the figure helper, pinned in
        # funhouse_agent/deep/tests/test_figure_helper.py
        "figure_helper": {A.FIGURE_HELPER_ENV: "1"},
        # GM-98 plotting code in a locked box, pinned in
        # funhouse_agent/tests/test_plot_box.py
        "plot_code": {A.PLOT_CODE_ENV: "1"},
        # GM-227 both on by default; these take them away
        "figure_helper_off": {A.FIGURE_HELPER_ENV: "0", A.PLOT_CODE_ENV: "0"},
        "plot_code_off": {A.PLOT_CODE_ENV: "0"},
        # TA-33 an answer at the step cap, pinned in webapp/tests/test_core.py
        "step_cap_answer": {A.STEP_CAP_ANSWER_ENV: "1"},
        # EV-21 cited documents nothing looked up, pinned in
        # funhouse_agent/tests/test_cited_documents.py
        "cited_documents_note": {A.CITED_DOCUMENTS_ENV: "note"},
        "cited_documents_retry": {A.CITED_DOCUMENTS_ENV: "retry"},
        # GM-116 judgment calls, pinned in
        # funhouse_agent/deep/tests/test_judgment_calls.py
        "judgment_calls": {A.JUDGMENT_CALLS_ENV: "1"},
        # GM-226 on by default; this takes it away
        "judgment_calls_off": {A.JUDGMENT_CALLS_ENV: "0"},
        # DR-38 the main model looks for itself where it asks to, pinned in
        # aec_document_review/tests/test_review_show_me_offline.py
        "show_me": {A.SHOW_ME_ENV: "1"},
    }


def test_the_legacy_build_accepts_and_ignores_working_dir(tmp_path):
    kw = _review_kwargs()
    a = build_deep_agent(_model([]), working_dir=str(tmp_path), **kw)
    b = build_deep_agent(_model([]), **kw)
    assert set(_tools_of(a)) == set(_tools_of(b))
    assert not hasattr(a, "geotech_review_agent")


@pytest.mark.parametrize("keep,words", [(None, "among the newest 2 views"),
                                        ("1", "while it is the newest view"),
                                        ("3", "among the newest 3 views")])
def test_inline_notes_say_how_long_an_image_stays_shown(sheet_png, monkeypatch,
                                                        keep, words):
    """Item 10: only the newest GEOTECH_VISION_INLINE_KEEP images are shown,
    and the notes say so."""
    if keep:
        monkeypatch.setenv(inline_store.KEEP_ENV, keep)
    assert words in vision_tools.inline_note()
    tool, _eng = _analyze_image(True)
    out = json.loads(tool.invoke({"attachment_key": sheet_png}))
    assert words in out["note"] and "look at it yourself" in out["note"]
    from funhouse_agent.deep.inline_images import InlineImageMiddleware
    assert InlineImageMiddleware().keep == int(keep or 2)


def test_an_arm_clears_the_switch_settings_it_does_not_name(monkeypatch):
    """Item 13: an arm is exactly what it says - settings included."""
    from funhouse_agent.deep.review_agent import MAX_MODEL_CALLS_ENV
    from funhouse_agent.plot_box import SETTINGS_ENVS as PLOT_CODE_SETTINGS
    assert set(review_flags.SETTINGS_ENVS) == {
        inline_store.KEEP_ENV, MAX_MODEL_CALLS_ENV} | set(PLOT_CODE_SETTINGS)
    assert not set(review_flags.SETTINGS_ENVS) & set(review_flags.ALL_ENVS)
    mine = {inline_store.KEEP_ENV: "1", MAX_MODEL_CALLS_ENV: "9"}
    for k, v in mine.items():
        monkeypatch.setenv(k, v)
    with review_flags.switches(review_flags.ARMS["minimal"]):
        assert not any(os.environ.get(k) for k in mine)
    with review_flags.switches({**review_flags.ARMS["minimal"],
                                MAX_MODEL_CALLS_ENV: "30"}):
        assert os.environ.get(MAX_MODEL_CALLS_ENV) == "30"
        assert os.environ.get(inline_store.KEEP_ENV) is None
    assert {k: os.environ.get(k) for k in mine} == mine     # restored
