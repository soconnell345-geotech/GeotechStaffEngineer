"""The Document Review harness changes of 2026-09-26 — offline, fake models.

Each change is behind a switch (``funhouse_agent.review_flags``); these tests
pin both sides: switched off, the page is what it was; switched on, it does
what the review asked for.
"""

import json

import pytest

pytest.importorskip("planlens.tools")
fitz = pytest.importorskip("fitz")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from funhouse_agent import (  # noqa: E402
    document_tools, inline_store, review_flags, vision_tools, vision_view)
from funhouse_agent.deep.agent import build_deep_agent  # noqa: E402
from funhouse_agent.deep.tools import make_vision_tools  # noqa: E402
from planlens.testing import build_synthetic_review_document  # noqa: E402

GEOTECH_WORDS = ("drawing_ir", "digitize_drawing", "query_drawing",
                 "snip_region", "call_agent", "figure_db")


@pytest.fixture(autouse=True)
def _no_switches(monkeypatch):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    monkeypatch.delenv("GEOTECH_REVIEW_MAX_MODEL_CALLS", raising=False)


@pytest.fixture(scope="module")
def gt():
    return build_synthetic_review_document()


def _review_kwargs():
    from webapp.profiles import DOCUMENT_REVIEW
    return DOCUMENT_REVIEW.build_kwargs()


class Scripted(FakeMessagesListChatModel):
    """A chat model that plays ``script`` (a list of AIMessages or callables
    of the request messages) and records what it was sent."""

    script: list = []
    seen: list = []
    bound: list = []

    def bind_tools(self, tools, **kw):
        self.bound.append([getattr(t, "name", None) or t.get("name")
                           if isinstance(t, dict) else getattr(t, "name", None)
                           for t in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        step = self.script.pop(0) if self.script else AIMessage(content="done")
        msg = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _model(script):
    return Scripted(responses=[AIMessage(content="x")], script=list(script),
                    seen=[], bound=[])


def _call(name, args, i=1):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                              "id": f"call_{name}_{i}"}])


def _system_text(messages):
    content = messages[0].content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b)
                       for b in content)
    return content


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
        return self.reply(prompt) if callable(self.reply) else self.reply


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------

def test_an_arm_s_other_variables_are_set_and_restored(monkeypatch):
    """Review finding 2: a custom arm's budget must not be dropped."""
    monkeypatch.delenv("GEOTECH_REVIEW_MAX_MODEL_CALLS", raising=False)
    import os
    with review_flags.switches({review_flags.AGENT_ENV: "lean",
                                "GEOTECH_REVIEW_MAX_MODEL_CALLS": "20"}):
        assert os.environ["GEOTECH_REVIEW_MAX_MODEL_CALLS"] == "20"
    assert "GEOTECH_REVIEW_MAX_MODEL_CALLS" not in os.environ


def test_switches_set_exactly_the_arm_and_restore(monkeypatch):
    monkeypatch.setenv(review_flags.VISION_INLINE_ENV, "1")
    with review_flags.switches(review_flags.ARMS["lean"]):
        assert review_flags.lean_agent()
        assert not review_flags.vision_inline()      # not in the arm: unset
    assert review_flags.vision_inline()              # restored
    assert not review_flags.lean_agent()
    assert set(review_flags.ARMS) >= {"baseline", "lean", "grounded",
                                      "inline", "sweep"}
    assert review_flags.ARMS["baseline"] == {}


# ---------------------------------------------------------------------------
# The lean agent (GEOTECH_REVIEW_AGENT=lean)
# ---------------------------------------------------------------------------

def test_switch_off_builds_the_page_exactly_as_before():
    kw = _review_kwargs()
    legacy = dict(kw)
    legacy.pop("review_page")
    a = build_deep_agent(_model([]), **kw)
    b = build_deep_agent(_model([]), **legacy)
    assert set(_tools_of(a)) == set(_tools_of(b))
    assert not hasattr(a, "geotech_review_agent")


def test_lean_agent_carries_only_review_tools(monkeypatch):
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    agent = build_deep_agent(_model([]), **_review_kwargs())
    names = set(_tools_of(agent))
    assert getattr(agent, "geotech_review_agent", None) == "lean"
    assert not names & {"execute", "glob", "grep", "ls", "read_file",
                        "write_file", "edit_file", "read_reference_figure",
                        "view_worked_example_source", "read_pdf_text",
                        "call_agent", "list_agents", "sweep_pages"}
    assert {"open_document", "read_document", "search_document",
            "analyze_pdf_page", "render_region", "annotate_document",
            "write_todos", "task"} <= names
    # Word output is offered wherever python-docx is installed.
    assert ("write_docx" in names) == vision_tools.docx_available()
    for t in _tools_of(agent).values():
        for word in GEOTECH_WORDS:
            assert word not in (t.description or ""), (t.name, word)


def test_lean_prompt_is_the_review_prompt_without_the_coding_agent(monkeypatch):
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    model = _model([AIMessage(content="ok")])
    agent = build_deep_agent(model, **_review_kwargs())
    agent.invoke({"messages": [{"role": "user", "content": "hi"}]})
    text = _system_text(model.seen[0])
    assert text.startswith("You are a document-review assistant")
    for gone in ("scratch filesystem", "You are a deep agent",
                 "Mimic existing style", "drawing_ir"):
        assert gone not in text, gone
    assert "pdf_page" in text          # the citation convention


def test_lean_budget_ends_a_long_request_with_an_answer(monkeypatch):
    from webapp import core
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    monkeypatch.setenv("GEOTECH_REVIEW_MAX_MODEL_CALLS", "6")

    def step(messages):
        return AIMessage(content="FINDINGS SO FAR")  # used once tools vanish

    model = _model([])
    calls = {"n": 0}

    def generate(self, messages, stop=None, run_manager=None, **kw):
        calls["n"] += 1
        self.seen.append(list(messages))
        tools_offered = self.bound and self.bound[-1]
        self.bound.append([])
        msg = (_call("list_files", {"path": "."}, calls["n"])
               if tools_offered else step(messages))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    monkeypatch.setattr(Scripted, "_generate", generate)
    agent = build_deep_agent(model, **_review_kwargs())
    out = list(core.stream_turn(agent, [{"role": "user", "content": "go"}],
                                "t", recursion_limit=50))
    done = [e for e in out if e["kind"] == "turn_done"][0]
    assert calls["n"] == 6
    assert "FINDINGS SO FAR" in done["answer"]
    last = model.seen[-1][-1]
    assert isinstance(last, HumanMessage) and "Step budget" in str(last.content)


def test_reader_helper_has_the_review_rules_and_no_writing_tools(monkeypatch):
    from funhouse_agent.deep.review_agent import READER_TOOLS, make_reader_tool
    tools = make_vision_tools(engine=FakeEngine(), include=set(READER_TOOLS))
    model = _model([AIMessage(content="- finding (sheet S-1), read from text")])
    tool = make_reader_tool(model, tools)
    out = tool.invoke({"description": "read page 1 of x.pdf",
                       "subagent_type": "page_reader"})
    assert "finding" in out
    text = _system_text(model.seen[0])
    assert "## How you read" in text and "pdf_page" in text
    assert "annotate_document" not in {t.name for t in tools}
    assert "write_docx" not in {t.name for t in tools}


def test_attachment_note_names_only_lean_tools_when_lean(monkeypatch, tmp_path):
    from webapp import core
    att = core.Attachment(key="a.pdf", path=str(tmp_path / "a.pdf"), size=1)
    legacy = core.attachment_note([att], review=True)
    assert "drawing_ir" in legacy          # switch off: unchanged
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    lean = core.attachment_note([att], review=True)
    assert "open_document" in lean and "drawing_ir" not in lean
    assert "drawing_ir" in core.attachment_note([att])   # geotech page


# ---------------------------------------------------------------------------
# Page numbers a viewer shows (unswitched fix)
# ---------------------------------------------------------------------------

def test_document_results_carry_the_viewer_page(gt):
    atts = {"r.pdf": gt.pdf}
    opened = json.loads(document_tools.dispatch_document_tool(
        "open_document", {"source": "r.pdf"}, attachments=atts,
        max_chars=14000))
    handle = opened["handle"]
    hits = json.loads(document_tools.dispatch_document_tool(
        "search_document", {"handle": handle, "pattern": "GENERAL NOTE"},
        attachments=atts, max_chars=14000))["hits"]
    assert hits and all(h["pdf_page"] == h["page"] + 1 for h in hits)
    text = json.loads(document_tools.dispatch_document_tool(
        "read_document", {"handle": handle, "pages": "0"}, attachments=atts,
        max_chars=14000))["text"]
    assert "=== page 0 [pdf_page 1]" in text


def test_viewer_pages_never_push_a_result_past_its_limit():
    raw = json.dumps({"hits": [{"page": i} for i in range(50)]})
    assert document_tools.with_viewer_pages(raw, limit=len(raw)) == raw
    assert '"pdf_page":50' in document_tools.with_viewer_pages(raw)
    assert document_tools.with_viewer_pages("not json") == "not json"


# ---------------------------------------------------------------------------
# Grounding the vision call
# ---------------------------------------------------------------------------

def test_vision_prompt_is_unchanged_with_the_switches_off():
    assert vision_tools._vision_prompt("find X", (0, 0, 10, 10), None) == \
        vision_view.with_grid("find X")
    assert vision_tools._lines_for_context(b"", 0) is None


def test_text_context_lists_the_view_s_lines_or_says_there_are_none(gt):
    lines, reliable = vision_view.page_lines(gt.pdf, gt.sheet_page)
    assert any("GENERAL NOTE A" in t for t, _ in lines)
    doc = fitz.open(stream=gt.pdf, filetype="pdf")
    rect = doc[gt.sheet_page].rect
    whole = vision_view.text_context(lines, (0, 0, rect.width, rect.height),
                                     reliable)
    assert "GENERAL NOTE A" in whole and "0-999" in whole
    empty = vision_view.text_context(lines, (1, 1, 2, 2), reliable)
    assert "NO text layer" in empty


def test_render_region_tells_the_vision_call_the_text_layer(gt, monkeypatch):
    monkeypatch.setenv(review_flags.VISION_TEXT_ENV, "1")
    eng = FakeEngine()
    tool = next(t for t in make_vision_tools(engine=eng,
                                             attachments={"r.pdf": gt.pdf})
                if t.name == "render_region")
    x, y = gt.sheet_text_upright_origin
    out = json.loads(tool.invoke({"attachment_key": "r.pdf",
                                  "page": gt.sheet_page,
                                  "bbox": [x - 20, y - 40, x + 200, y + 20]}))
    assert out["pdf_page"] == gt.sheet_page + 1
    assert "GENERAL NOTE A" in eng.prompts[0]


def test_structured_locations_come_back_as_page_boxes(gt, monkeypatch):
    monkeypatch.setenv(review_flags.VISION_STRUCTURED_ENV, "1")
    eng = FakeEngine('The note.\nLOCATED: [{"what": "note", "text": "GENERAL '
                     'NOTE A", "box": [0, 0, 999, 999], "sure": true}]')
    tool = next(t for t in make_vision_tools(engine=eng,
                                             attachments={"r.pdf": gt.pdf})
                if t.name == "render_region")
    out = json.loads(tool.invoke({"attachment_key": "r.pdf", "page": 0,
                                  "bbox": [50, 50, 250, 150]}))
    assert "LOCATED" in eng.prompts[0]
    assert out["analysis"] == "The note."
    item = out["located"][0]
    assert item["text"] == "GENERAL NOTE A"
    assert item["page_bbox"] == pytest.approx(out["view"], abs=0.2)


def test_split_located_keeps_an_answer_it_cannot_parse():
    text = "no list here\nLOCATED: [broken"
    assert vision_view.split_located(text, (0, 0, 10, 10)) == (text, [])


# ---------------------------------------------------------------------------
# The main model looks (GEOTECH_VISION_INLINE)
# ---------------------------------------------------------------------------

def test_inline_render_stores_the_image_and_makes_no_vision_call(gt):
    inline_store.clear()
    eng = FakeEngine()
    tool = next(t for t in make_vision_tools(engine=eng,
                                             attachments={"r.pdf": gt.pdf},
                                             inline_images=True)
                if t.name == "render_region")
    out = json.loads(tool.invoke({"attachment_key": "r.pdf", "page": 0,
                                  "bbox": [50, 50, 250, 150]}))
    assert eng.prompts == []
    assert inline_store.get(out["image_id"]) is not None
    assert "look at it" in out["note"]


def test_inline_middleware_shows_the_newest_images_to_the_model(gt,
                                                                monkeypatch):
    inline_store.clear()
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    monkeypatch.setenv(review_flags.VISION_INLINE_ENV, "1")
    model = _model([
        _call("render_region", {"attachment_key": "r.pdf", "page": 0,
                                "bbox": [50, 50, 250, 150]}),
        AIMessage(content="I can see the heading."),
    ])
    agent = build_deep_agent(model, attachments={"r.pdf": gt.pdf},
                             **_review_kwargs())
    agent.invoke({"messages": [{"role": "user", "content": "look"}]})
    second = model.seen[1]
    last = second[-1]
    assert isinstance(last, HumanMessage)
    kinds = [b.get("type") for b in last.content]
    assert "image_url" in kinds
    # never saved into the conversation: the state holds text results only
    final = agent.invoke({"messages": [{"role": "user", "content": "x"}]})
    assert not any(isinstance(m, HumanMessage) and isinstance(m.content, list)
                   for m in final["messages"])


def test_probe_sends_the_image_after_the_tool_result():
    from funhouse_agent.deep.inline_images import probe
    model = _model([AIMessage(content="It is red.")])
    out = probe(model)
    assert out == {"ok": True, "answer": "It is red.", "error": None}
    sent = model.seen[0]
    assert isinstance(sent[-2], ToolMessage)
    assert isinstance(sent[-1], HumanMessage)
    assert any(b.get("type") == "image_url" for b in sent[-1].content)

    class Refuses(Scripted):
        def _generate(self, messages, stop=None, run_manager=None, **kw):
            raise ValueError("images not allowed after tool messages")

    bad = probe(Refuses(responses=[AIMessage(content="x")], script=[],
                        seen=[], bound=[]))
    assert not bad["ok"] and "images not allowed" in bad["error"]


def test_newest_image_ids_stop_at_the_user_turn():
    from funhouse_agent.deep.inline_images import newest_image_ids
    inline_store.clear()
    a = inline_store.put(b"\x89PNG a", {})
    b = inline_store.put(b"\x89PNG b", {})
    c = inline_store.put(b"\x89PNG c", {})
    msgs = [ToolMessage(content=json.dumps({"image_id": a}), tool_call_id="1"),
            HumanMessage(content="next question"),
            ToolMessage(content=json.dumps({"image_id": b}), tool_call_id="2"),
            ToolMessage(content=json.dumps({"image_id": c}), tool_call_id="3")]
    assert newest_image_ids(msgs, keep=2) == [b, c]
    assert newest_image_ids(msgs, keep=5) == [b, c]


def test_inline_images_reach_the_budget_s_final_call(gt, monkeypatch):
    """Review finding 4: the last budgeted call must still see the pages."""
    inline_store.clear()
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    monkeypatch.setenv(review_flags.VISION_INLINE_ENV, "1")
    monkeypatch.setenv("GEOTECH_REVIEW_MAX_MODEL_CALLS", "4")   # the floor
    look = {"attachment_key": "r.pdf", "page": 0, "bbox": [50, 50, 250, 150]}
    model = _model([_call("render_region", look, i) for i in (1, 2, 3)]
                   + [AIMessage(content="From the image: the heading.")])
    agent = build_deep_agent(model, attachments={"r.pdf": gt.pdf},
                             **_review_kwargs())
    agent.invoke({"messages": [{"role": "user", "content": "look"}]})
    assert len(model.seen) == 4
    final = model.seen[3]
    assert "Step budget" in str(final[-1].content)          # the last call
    assert any(isinstance(m, HumanMessage) and isinstance(m.content, list)
               and any(b.get("type") == "image_url" for b in m.content)
               for m in final)


def test_a_failing_reading_helper_does_not_end_the_turn():
    """Review finding 5."""
    from funhouse_agent.deep.review_agent import make_reader_tool

    class Boom(Scripted):
        def _generate(self, messages, stop=None, run_manager=None, **kw):
            raise RuntimeError("429 rate limited")

    tool = make_reader_tool(Boom(responses=[AIMessage(content="x")],
                                 script=[], seen=[], bound=[]), [])
    out = tool.invoke({"description": "read page 1"})
    assert "reading helper failed" in out and "429" in out


def test_located_lists_are_trimmed_to_fit_not_cut_mid_json():
    """Review finding 3."""
    item = {"what": "x" * 40, "text": "y" * 40, "image_box": [1, 2, 3, 4],
            "page_bbox": [1.0, 2.0, 3.0, 4.0], "sure": True}
    out = {"page": 0, "analysis": "a" * 3000, "located": [dict(item)] * 60,
           "tiles": [{"tile": f"r{i}", "analysis": "b" * 1500,
                      "located": [dict(item)] * 30} for i in range(16)]}
    vision_tools._fit_located(out, vision_tools.TILED_RESULT_CHARS)
    vision_tools._fit_tiles(out)
    text = json.dumps(out)
    assert len(text) <= vision_tools.TILED_RESULT_CHARS
    json.loads(text)
    region = {"page": 0, "analysis": "c" * 2000, "located": [dict(item)] * 80}
    vision_tools._fit_region(region)
    assert len(json.dumps(region)) <= vision_tools.REGION_RESULT_CHARS
    assert region.get("located_cut")


def test_split_located_keeps_text_after_the_list():
    text, items = vision_view.split_located(
        'Answer.\nLOCATED: [{"what": "a", "box": [0, 0, 999, 999]}]\nMore.',
        (0, 0, 100, 100))
    assert text == "Answer.\nMore." and items[0]["page_bbox"] == \
        [0.0, 0.0, 100.0, 100.0]


def test_page_lines_unreadable_is_none_not_no_text_layer():
    assert vision_view.page_lines(b"not a pdf", 0) is None


def test_small_host_caps_keep_document_results_whole(gt):
    """Review finding 8: the viewer pages never push past the host's cap."""
    out = document_tools.dispatch_document_tool(
        "open_document", {"source": "r.pdf"}, attachments={"r.pdf": gt.pdf},
        max_chars=1000, cap=1100)
    assert len(out) <= 1100
    json.loads(out)


# ---------------------------------------------------------------------------
# sweep_pages (GEOTECH_REVIEW_SWEEP)
# ---------------------------------------------------------------------------

def test_sweep_answers_every_page_and_reports_per_page(monkeypatch):
    from planlens.testing.submittal_fixtures import build_synthetic_submittal
    from funhouse_agent.deep.sweep import sweep
    pdf = build_synthetic_submittal().pdf

    def page_answer(prompt):
        yes = "SCALE" in prompt or "1:100" in prompt
        return json.dumps({"relevant": yes, "answer": "scale 1:100" if yes
                           else "", "items": [{"text": "SCALE: 1:100",
                                               "box": [0, 0, 100, 100]}]
                           if yes else [], "sure": True})

    eng = FakeEngine(page_answer)
    text_model = _model([lambda m: AIMessage(content=page_answer(
        m[-1].content))] * 20)
    out = sweep(pdf, "Which pages state a drawing scale?", eng,
                model=text_model)
    assert out["pages_checked"] == "0-10"
    rel = {r["page"] for r in out["relevant"]}
    assert rel == {8, 9}
    assert all(r["pdf_page"] == r["page"] + 1 for r in out["relevant"])
    looked = [r for r in out["relevant"] if r["how"] == "looked"]
    assert looked and "page_bbox" in looked[0]["items"][0]


def test_sweep_asks_for_pixels_and_converts_them_with_the_size_sent():
    """2026-10-07: a sweep's whole-page items are asked for in pixels of the
    image as sent and converted exactly; each also carries a padded
    zoom_bbox, because a whole-page box says where to look, not where to put
    a mark."""
    import re as _re
    from planlens.testing.submittal_fixtures import build_synthetic_submittal
    from funhouse_agent.deep.sweep import sweep
    pdf = build_synthetic_submittal().pdf
    seen = {}

    def page_answer(prompt):
        m = _re.search(r"pixels of this (\d+) x (\d+) image", prompt)
        if not m:
            return json.dumps({"relevant": False, "answer": "", "items": [],
                               "sure": True})
        w, h = int(m.group(1)), int(m.group(2))
        seen["size"] = (w, h)
        return json.dumps({"relevant": True, "answer": "a tag", "sure": True,
                           "items": [{"text": "T", "px": [w / 4, h / 4,
                                                          w / 2, h / 2]}]})

    out = sweep(pdf, "Where is the tag?", FakeEngine(page_answer),
                pages="8", look="always")
    (row,) = out["relevant"]
    (item,) = row["items"]
    with fitz.open(stream=pdf, filetype="pdf") as doc:
        r = doc[8].rect
    assert item["page_bbox"] == pytest.approx(
        [r.width / 4, r.height / 4, r.width / 2, r.height / 2], abs=0.1)
    zx0, zy0, zx1, zy1 = item["zoom_bbox"]
    assert zx0 < r.width / 4 - 0.09 * r.width
    assert "zoom_bbox" in out["note"] and "not where to put a mark" in out["note"]


def test_sweep_tool_is_offered_only_with_its_switch(monkeypatch):
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    assert "sweep_pages" not in _tools_of(
        build_deep_agent(_model([]), **_review_kwargs()))
    monkeypatch.setenv(review_flags.SWEEP_ENV, "1")
    assert "sweep_pages" in _tools_of(
        build_deep_agent(_model([]), **_review_kwargs()))
