"""The page's main model looks for itself where it asks to
(``GEOTECH_VISION_SHOW_ME``, BACKLOG DR-38), offline.

Today every look on the Document Review page is a separate vision call that
writes a description, and the agent's own model reasons from that text.
Behind the switch the looking tools take ``show_me``: with it no vision call
runs and the image is put in front of the agent's model at its next step.
Left off, a look is today's. The model chooses per call.
"""

import json

import pytest

pytest.importorskip("planlens.tools")
pymupdf = pytest.importorskip("pymupdf")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from aec_agent_core import inline_store, review_flags  # noqa: E402
from aec_agent_core.deep.vision_kit import (  # noqa: E402
    SHOW_ME_NOTES, make_vision_tools)
from aec_document_review.review_build import build_review_page_agent  # noqa: E402
from tests.test_review_harness_offline import (  # noqa: E402
    FakeEngine, _call, _model, _review_kwargs, _tools_of)
from planlens.testing import build_synthetic_review_document  # noqa: E402

LOOKING = ("analyze_pdf_page", "render_region", "analyze_image")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    inline_store.clear()
    yield
    inline_store.clear()


@pytest.fixture(scope="module")
def gt():
    return build_synthetic_review_document()


def _picture() -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 320, 200), False)
    pix.set_rect(pix.irect, (30, 90, 200))
    return pix.tobytes("png")


def _agent(model, gt, engine=None, **attachments):
    return build_review_page_agent(
        model, attachments={"review_set.pdf": gt.pdf, **attachments},
        engine=engine or FakeEngine(), **_review_kwargs())


def _images(messages):
    return [b for m in messages if isinstance(m, HumanMessage)
            and isinstance(m.content, list)
            for b in m.content if isinstance(b, dict)
            and b.get("type") == "image_url"]


def _schema(tool):
    return tool.args_schema.model_json_schema()["properties"]


# ---------------------------------------------------------------------------
# Off: nothing changes
# ---------------------------------------------------------------------------

def test_off_the_tools_have_no_show_me_and_a_look_is_a_vision_call(gt):
    assert not review_flags.show_me()
    engine = FakeEngine("a letter with a stamp")
    model = _model([_call("analyze_pdf_page",
                          {"attachment_key": "review_set.pdf", "page": 0,
                           "tiles": "off"}),
                    lambda messages: AIMessage(
                        content=f"images seen: {len(_images(messages))}")])
    agent = _agent(model, gt, engine)
    tools = _tools_of(agent)
    for name in LOOKING:
        assert "show_me" not in _schema(tools[name]), name
        assert "show_me" not in tools[name].description, name
    out = agent.invoke({"messages": [{"role": "user", "content": "Page 1?"}]},
                       config={"recursion_limit": 200})
    assert out["messages"][-1].content == "images seen: 0"
    assert len(engine.prompts) == 1


def test_off_a_call_that_names_show_me_is_refused_by_name(gt):
    """The argument does not exist with the switch off: it is refused, not
    dropped silently (the kit's strict arguments)."""
    engine = FakeEngine()
    tools = {t.name: t for t in make_vision_tools(
        engine=engine, attachments={"review_set.pdf": gt.pdf},
        include={"analyze_pdf_page"})}
    out = tools["analyze_pdf_page"].invoke(
        {"attachment_key": "review_set.pdf", "page": 0, "show_me": True})
    assert "show_me" in str(out) and engine.prompts == []


# ---------------------------------------------------------------------------
# On: the model chooses per call
# ---------------------------------------------------------------------------

@pytest.fixture
def on(monkeypatch):
    for k, v in review_flags.ARMS["show_me"].items():
        monkeypatch.setenv(k, v)
    assert review_flags.show_me()


def test_on_the_three_looking_tools_take_show_me_and_say_when(on, gt):
    agent = _agent(_model([]), gt)
    tools = _tools_of(agent)
    for name in LOOKING:
        field = _schema(tools[name])["show_me"]
        assert field["type"] == "boolean" and field["default"] is False
        assert tools[name].description.endswith(SHOW_ME_NOTES[name])
    page = tools["analyze_pdf_page"].description
    assert "judge with your own eyes" in page and "READ OUT" in page
    # no other tool gains it, and the prompt is the page's own, unchanged
    others = [n for n, t in tools.items() if n not in LOOKING
              and "show_me" in json.dumps(_schema(t))]
    assert others == []
    assert getattr(agent, "geotech_review_agent", None) != "minimal"


def test_on_a_look_with_show_me_is_shown_and_no_vision_call_runs(on, gt):
    engine = FakeEngine()
    seen = []

    def after_look(messages):
        seen.append(len(_images(messages)))
        last = next(m for m in reversed(messages)
                    if getattr(m, "type", "") == "tool")
        result = json.loads(last.content)
        assert result["image_id"].startswith("img_")
        assert "shown to you with your next step" in result["note"]
        assert "analysis" not in result
        return _call("render_region", {
            "attachment_key": "review_set.pdf", "page": 0,
            "view": result["view"], "image_box": [100, 100, 400, 300],
            "show_me": True}, 2)

    def after_zoom(messages):
        seen.append(len(_images(messages)))
        return AIMessage(content="The stamp is on the letter.")

    model = _model([_call("analyze_pdf_page",
                          {"attachment_key": "review_set.pdf", "page": 0,
                           "show_me": True}),
                    after_look, after_zoom])
    out = _agent(model, gt, engine).invoke(
        {"messages": [{"role": "user", "content": "Where is the stamp?"}]},
        config={"recursion_limit": 200})
    assert out["messages"][-1].content == "The stamp is on the letter."
    assert seen == [1, 2]                 # the page, then the page and zoom
    assert engine.prompts == []           # nobody else looked
    # the images went into the requests, never into the conversation
    assert _images(out["messages"]) == []


def test_on_a_look_without_show_me_is_still_a_vision_call(on, gt):
    engine = FakeEngine("a ruled table of quantities")
    seen = []

    def after(messages):
        seen.append(len(_images(messages)))
        last = next(m for m in reversed(messages)
                    if getattr(m, "type", "") == "tool")
        assert "image_id" not in json.loads(last.content)
        return AIMessage(content="read")

    model = _model([_call("analyze_pdf_page",
                          {"attachment_key": "review_set.pdf", "page": 2,
                           "tiles": "off", "prompt": "Read the table."}),
                    after])
    _agent(model, gt, engine).invoke(
        {"messages": [{"role": "user", "content": "Read page 3."}]},
        config={"recursion_limit": 200})
    assert seen == [0]
    assert len(engine.prompts) == 1 and "Read the table." in engine.prompts[0]


def test_on_an_uploaded_picture_is_shown_too(on, gt):
    """Before the switch an upload's first look was always a vision call,
    inline build or not (BACKLOG VIS-26, DR-38)."""
    engine = FakeEngine()
    seen = []

    def after(messages):
        seen.append(len(_images(messages)))
        return AIMessage(content="A blue panel.")

    model = _model([_call("analyze_image", {"attachment_key": "site.png",
                                            "show_me": True}), after])
    out = _agent(model, gt, engine, **{"site.png": _picture()}).invoke(
        {"messages": [{"role": "user", "content": "What is in the photo?"}]},
        config={"recursion_limit": 200})
    assert out["messages"][-1].content == "A blue panel."
    assert seen == [1] and engine.prompts == []
    # without show_me the same upload goes to the vision call
    model = _model([_call("analyze_image", {"attachment_key": "site.png"}),
                    lambda messages: AIMessage(
                        content=str(len(_images(messages))))])
    out = _agent(model, gt, engine, **{"site.png": _picture()}).invoke(
        {"messages": [{"role": "user", "content": "And again?"}]},
        config={"recursion_limit": 200})
    assert out["messages"][-1].content == "0" and len(engine.prompts) == 1


def test_on_the_reading_helper_is_shown_what_it_asks_to_see(on, gt):
    """The helper holds the primary's looking tools, so it has show_me, and
    its own model is shown the image (not the primary's)."""
    engine = FakeEngine()
    seen = {"helper": [], "primary": []}

    def helper_after(messages):
        seen["helper"].append(len(_images(messages)))
        return AIMessage(content="Helper: page 1 is a letter.")

    def primary_after(messages):
        seen["primary"].append(len(_images(messages)))
        return AIMessage(content="Done.")

    model = _model([
        _call("task", {"description": "Look at page 1 of review_set.pdf.",
                       "subagent_type": "general-purpose"}),
        _call("analyze_pdf_page", {"attachment_key": "review_set.pdf",
                                   "page": 0, "show_me": True}, 2),
        helper_after, primary_after])
    out = _agent(model, gt, engine).invoke(
        {"messages": [{"role": "user", "content": "What is page 1?"}]},
        config={"recursion_limit": 200})
    assert out["messages"][-1].content == "Done."
    assert seen == {"helper": [1], "primary": [0]}
    assert engine.prompts == []


def test_the_suite_has_the_arm(on):
    assert review_flags.ARMS["show_me"] == {
        "GEOTECH_VISION_SHOW_ME": "1"}
    assert "GEOTECH_VISION_SHOW_ME" in review_flags.ALL_ENVS
    with review_flags.switches({}):
        assert not review_flags.show_me()     # baseline clears it
    assert review_flags.show_me()


def test_the_reading_helper_does_not_hang_on_the_agent_library_s_spec(
        gt, monkeypatch):
    """BACKLOG INF-71: the helper used to start from a spec imported from a
    private name; had the name moved, the page silently got the stock
    helper with every tool. It is declared here now, whatever that name
    holds."""
    from aec_agent_core.deep import assembly
    from aec_document_review import review_build
    from aec_document_review.review_agent import READER_TOOLS
    monkeypatch.setattr(assembly, "_GENERAL_PURPOSE_SPEC", None)
    spec = review_build._review_reader_subagent(
        list(_tools_of(_agent(_model([]), gt)).values()))
    assert spec["name"] == "general-purpose"
    assert {t.name for t in spec["tools"]} <= set(READER_TOOLS)
    assert spec["tools"] and "annotate_document" not in {
        t.name for t in spec["tools"]}
    assert set(spec) == {"name", "description", "system_prompt", "tools",
                         "middleware"}
