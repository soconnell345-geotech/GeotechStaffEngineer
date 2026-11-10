"""The Document Review harness changes of 2026-09-26 — offline, fake models.

Each change is behind a switch (``aec_agent_core.review_flags``); these tests
pin both sides: switched off, the page is what it was; switched on, it does
what the review asked for.
"""

import json

import pytest

pytest.importorskip("planlens.tools")
pymupdf = pytest.importorskip("pymupdf")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from aec_agent_core import (  # noqa: E402
    document_tools, inline_store, review_flags, vision_tools, vision_view)
from aec_document_review.review_build import build_review_page_agent as build_deep_agent  # noqa: E402
from aec_agent_core.deep.vision_kit import make_vision_tools  # noqa: E402
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
    from aec_document_review.profiles import DOCUMENT_REVIEW
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
    with review_flags.switches({review_flags.AGENT_ENV: "minimal",
                                "GEOTECH_REVIEW_MAX_MODEL_CALLS": "20"}):
        assert os.environ["GEOTECH_REVIEW_MAX_MODEL_CALLS"] == "20"
    assert "GEOTECH_REVIEW_MAX_MODEL_CALLS" not in os.environ


def test_switches_set_exactly_the_arm_and_restore(monkeypatch):
    monkeypatch.setenv(review_flags.STEP_CAP_ANSWER_ENV, "1")
    with review_flags.switches(review_flags.ARMS["minimal"]):
        assert review_flags.minimal_agent()
        assert not review_flags.step_cap_answer()  # not in the arm: unset
    assert review_flags.step_cap_answer()          # restored
    assert not review_flags.minimal_agent()
    assert set(review_flags.ARMS) >= {"baseline", "minimal"}
    assert review_flags.ARMS["baseline"] == {}


def test_the_removed_experiments_are_gone():
    """BACKLOG DR-39: the lean build and the switches and arms on it are
    removed; a notebook still asking for an old arm is told so."""
    for arm in ("lean", "grounded", "inline", "sweep", "overview",
                "geometry", "digest", "kept_readings"):
        assert arm not in review_flags.ARMS
    for name in ("lean_agent", "vision_inline", "sweep", "findings",
                 "overview", "geometry", "digest",
                 "kept_readings_on_review_page", "overview_pages"):
        assert not hasattr(review_flags, name), name


def test_the_unmeasured_vision_switches_are_gone():
    """BACKLOG DR-40: the vision switches that never measured a gain are
    removed with their code; a notebook that still sets one gets the default
    look."""
    assert "turn_retry" not in review_flags.ARMS
    for name in ("vision_text_context", "vision_structured",
                 "vision_turn_retry", "VISION_TEXT_ENV",
                 "VISION_STRUCTURED_ENV", "VISION_TURN_RETRY_ENV"):
        assert not hasattr(review_flags, name), name
    for name in ("text_context", "page_lines", "with_locations",
                 "split_located", "patch_align", "align_to_patches",
                 "PATCH_ALIGN_ENV"):
        assert not hasattr(vision_view, name), name
    for name in ("_lines_for_context", "_split_located", "_settle_turn",
                 "remembered_turn", "clear_turn_memory", "_fit_located"):
        assert not hasattr(vision_tools, name), name





# ---------------------------------------------------------------------------
# The default page's own fixes, every switch off (BACKLOG DR-01 - DR-03)
# ---------------------------------------------------------------------------
def _looping_model(final="FINDINGS SO FAR"):
    """Calls list_files whenever tools are offered; answers once they go."""
    model = _model([])
    calls = {"n": 0}

    def generate(self, messages, stop=None, run_manager=None, **kw):
        calls["n"] += 1
        self.seen.append(list(messages))
        tools_offered = self.bound and self.bound[-1]
        self.bound.append([])
        msg = (_call("list_files", {"path": "."}, calls["n"])
               if tools_offered else AIMessage(content=final))
        return ChatResult(generations=[ChatGeneration(message=msg)])

    return model, calls, generate




#: The default page's own tools with every switch off (DR-04), besides the
#: planlens document tools and the optional ones (each offered where its
#: library imports).
REVIEW_PAGE_TOOLS = {"analyze_image", "analyze_pdf_page", "render_region",
                     "calculate", "list_files", "read_pdf_text",
                     "read_text_file", "save_file"}

#: deepagents' own tools: planning, the helper and the scratch filesystem
#: (hidden from this page's menu by HideScratchFilesystem); which of them a
#: build has depends on the deepagents version.
DEEPAGENTS_OWN = {"write_todos", "task", "ls", "read_file", "write_file",
                  "edit_file", "delete", "glob", "grep", "execute"}






def test_default_page_budget_ends_a_long_request_with_an_answer(monkeypatch):
    """DR-03: the step cap ended long turns in GraphRecursionError after
    about 17 model calls; the budget's last call answers instead."""
    from aec_document_review import core
    monkeypatch.setenv("GEOTECH_REVIEW_MAX_MODEL_CALLS", "6")
    model, calls, generate = _looping_model()
    monkeypatch.setattr(Scripted, "_generate", generate)
    agent = build_deep_agent(model, **_review_kwargs())
    assert not hasattr(agent, "geotech_review_agent")       # the default build
    out = list(core.stream_turn(agent, [{"role": "user", "content": "go"}],
                                "t", recursion_limit=50))
    done = [e for e in out if e["kind"] == "turn_done"][0]
    assert calls["n"] == 6
    assert "FINDINGS SO FAR" in done["answer"]
    last = model.seen[-1][-1]
    assert isinstance(last, HumanMessage) and "Step budget" in str(last.content)


def test_default_page_step_floor_fits_the_default_budget(monkeypatch):
    """The host's cap is raised enough for all 40 calls of the default
    budget on this build (REVIEW_PAGE_STEPS_PER_CALL)."""
    from aec_document_review import core
    from aec_document_review.review_agent import DEFAULT_MAX_MODEL_CALLS
    model, calls, generate = _looping_model()
    monkeypatch.setattr(Scripted, "_generate", generate)
    agent = build_deep_agent(model, **_review_kwargs())
    out = list(core.stream_turn(agent, [{"role": "user", "content": "go"}],
                                "t", recursion_limit=core.DEFAULT_BEHAVIOR[
                                    "recursion_limit"]))
    assert calls["n"] == DEFAULT_MAX_MODEL_CALLS
    assert "FINDINGS SO FAR" in [e for e in out
                                 if e["kind"] == "turn_done"][0]["answer"]


def test_default_page_budget_off_restores_the_step_cap(monkeypatch):
    """GEOTECH_REVIEW_MAX_MODEL_CALLS=off is the way back on a live host."""
    from langgraph.errors import GraphRecursionError
    from aec_document_review import core
    monkeypatch.setenv("GEOTECH_REVIEW_MAX_MODEL_CALLS", "off")
    model, calls, generate = _looping_model()
    monkeypatch.setattr(Scripted, "_generate", generate)
    agent = build_deep_agent(model, **_review_kwargs())
    assert not getattr(agent, "geotech_min_recursion_limit", None)
    with pytest.raises(GraphRecursionError):
        list(core.stream_turn(agent, [{"role": "user", "content": "go"}],
                              "t", recursion_limit=20))




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

def test_vision_prompt_is_the_agent_s_prompt_with_the_location_rule():
    assert vision_tools._vision_prompt("find X") == \
        vision_view.with_grid("find X")


# ---------------------------------------------------------------------------
# The main model looks (the minimal build's inline images)
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
    monkeypatch.setenv(review_flags.AGENT_ENV, "minimal")
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
    from aec_agent_core.deep.inline_images import probe
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
    from aec_agent_core.deep.inline_images import newest_image_ids
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
    monkeypatch.setenv(review_flags.AGENT_ENV, "minimal")
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
    """Review finding 5 (the minimal build's helper; the lean build's went
    with BACKLOG DR-39)."""
    from aec_document_review.minimal_agent import _helper_tool

    class Boom(Scripted):
        def _generate(self, messages, stop=None, run_manager=None, **kw):
            raise RuntimeError("429 rate limited")

    tool = _helper_tool(Boom(responses=[AIMessage(content="x")],
                             script=[], seen=[], bound=[]), FakeEngine(), [])
    out = tool.invoke({"description": "read page 1"})
    assert "helper failed" in out and "429" in out


def test_long_results_are_trimmed_to_fit_not_cut_mid_json():
    """Review finding 3."""
    out = {"page": 0, "analysis": "a" * 3000,
           "tiles": [{"tile": f"r{i}", "analysis": "b" * 3000}
                     for i in range(16)]}
    vision_tools._fit_tiles(out)
    text = json.dumps(out)
    assert len(text) <= vision_tools.TILED_RESULT_CHARS
    json.loads(text)
    region = {"page": 0, "analysis": "c" * 20000}
    vision_tools._fit_region(region)
    assert len(json.dumps(region)) <= vision_tools.REGION_RESULT_CHARS


def test_small_host_caps_keep_document_results_whole(gt):
    """Review finding 8: the viewer pages never push past the host's cap."""
    out = document_tools.dispatch_document_tool(
        "open_document", {"source": "r.pdf"}, attachments={"r.pdf": gt.pdf},
        max_chars=1000, cap=1100)
    assert len(out) <= 1100
    json.loads(out)
