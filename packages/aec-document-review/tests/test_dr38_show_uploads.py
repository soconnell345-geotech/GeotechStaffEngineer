"""The page's main model is shown an uploaded image itself
(``GEOTECH_VISION_SHOW_UPLOADS``, BACKLOG DR-38), offline, fake models.

Today an upload's first look is always a vision side call, and the main
model reasons from the text that call writes. ``show_me`` lets the model ask
to see the image, and no model asked. Behind this switch the main agent's
``analyze_image`` of an uploaded image file shows the image to the main
model at its next step, with no vision call; everything else is today's.
"""

import base64
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("planlens.tools")
pymupdf = pytest.importorskip("pymupdf")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from aec_agent_core import inline_store, review_flags, vision_tools  # noqa: E402
from aec_agent_core.deep import keep_work, upload_view  # noqa: E402
from aec_agent_core.deep.vision_kit import make_vision_tools  # noqa: E402
from aec_document_review.review_build import build_review_page_agent  # noqa: E402
from tests.test_review_harness_offline import (  # noqa: E402
    FakeEngine, _call, _model, _review_kwargs)
from planlens.testing import build_synthetic_review_document  # noqa: E402

ENV = review_flags.SHOW_UPLOADS_ENV


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.LATER_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("GEOTECH_TURN_MAX_IMAGE_CALLS", raising=False)
    monkeypatch.delenv("GEOTECH_VISION_DETAIL", raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    inline_store.clear()
    yield
    inline_store.clear()


@pytest.fixture(scope="module")
def gt():
    return build_synthetic_review_document()


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert review_flags.show_uploads()


class SameModelEngine(FakeEngine):
    """A vision engine that reads images with the page's main model, as the
    page's default engine (``LangChainVisionEngine(model)``) does."""

    def __init__(self, model, reply="a bar chart"):
        super().__init__(reply)
        self.model = model


def _chart() -> bytes:
    """A made-up picture: a white field with three dark bars."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 200), False)
    pix.set_rect(pix.irect, (255, 255, 255))
    for i, h in enumerate((60, 120, 90)):
        pix.set_rect(pymupdf.IRect(40 + 80 * i, 190 - h, 90 + 80 * i, 190),
                     (20, 40, 160))
    return pix.tobytes("png")


def _images(messages):
    return [b for m in messages if isinstance(m, HumanMessage)
            and isinstance(m.content, list)
            for b in m.content if isinstance(b, dict)
            and b.get("type") == "image_url"]


def _last_tool(messages):
    return json.loads(next(m for m in reversed(messages)
                           if getattr(m, "type", "") == "tool").content)


def _agent(model, gt, engine, **attachments):
    return build_review_page_agent(
        model, attachments={"review_set.pdf": gt.pdf, **attachments},
        engine=engine, **_review_kwargs())


def _turn(agent, text="What is the tallest bar?"):
    return agent.invoke({"messages": [{"role": "user", "content": text}]},
                        config={"recursion_limit": 200})


def _look_tool(engine, attachments):
    return {t.name: t for t in make_vision_tools(
        engine=engine, attachments=attachments,
        include={"analyze_image"})}["analyze_image"]


def _marked(fn):
    """``fn()`` run as the main agent's analyze_image runs under the
    switch (the middleware's mark)."""
    token = upload_view._ON.set(True)
    try:
        return fn()
    finally:
        upload_view._ON.reset(token)


# ---------------------------------------------------------------------------
# Off: nothing changes
# ---------------------------------------------------------------------------

def test_off_an_upload_s_look_is_today_s_vision_call(gt, monkeypatch):
    """Unset, the new code is never reached: the look is a vision call, no
    image goes to the model and the record has no new line."""
    def never(*a, **k):
        raise AssertionError("show_upload ran with the switch unset")
    monkeypatch.setattr(upload_view, "show_upload", never)
    seen = []

    def after(messages):
        seen.append(len(_images(messages)))
        assert "analysis" in _last_tool(messages)
        return AIMessage(content="The middle bar.")

    model = _model([_call("analyze_image", {"attachment_key": "chart.png",
                                            "prompt": "Read the bars."}),
                    after])
    engine = SameModelEngine(model)
    agent = _agent(model, gt, engine, **{"chart.png": _chart()})
    out = _turn(agent)
    assert out["messages"][-1].content == "The middle bar."
    assert seen == [0] and len(engine.prompts) == 1
    assert "Read the bars." in engine.prompts[0]
    assert "show_uploads" not in agent.geotech_build
    assert ENV not in agent.geotech_build["switches"]


def test_set_but_without_the_page_s_middleware_nothing_changes(on):
    """A build that does not carry the middleware (the geotech page, the
    minimal build, a helper) looks as today even with the variable set."""
    engine = FakeEngine("three bars")
    out = json.loads(_look_tool(engine, {"chart.png": _chart()}).invoke(
        {"attachment_key": "chart.png", "prompt": "Read it."}))
    assert out["analysis"] == "three bars" and "image_id" not in out
    assert len(engine.prompts) == 1


# ---------------------------------------------------------------------------
# On: the main model sees the upload
# ---------------------------------------------------------------------------

def test_on_an_upload_is_shown_to_the_main_model_and_no_vision_call_runs(
        on, gt):
    seen = []

    def after(messages):
        blocks = _images(messages)
        seen.append(len(blocks))
        result = _last_tool(messages)
        assert result["image_id"].startswith("img_")
        assert "analysis" not in result and "zoom_hint" not in result
        assert "uploaded image is shown to you" in result["note"]
        assert result["view"] == [0, 0, 300, 200]
        url = blocks[0]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1])[:8] == \
            b"\x89PNG\r\n\x1a\n"
        return AIMessage(content="The middle bar is tallest.")

    model = _model([_call("analyze_image", {"attachment_key": "chart.png",
                                            "prompt": "Read the bars."}),
                    after])
    engine = SameModelEngine(model)
    agent = _agent(model, gt, engine, **{"chart.png": _chart()})
    out = _turn(agent)
    assert out["messages"][-1].content == "The middle bar is tallest."
    assert seen == [1] and engine.prompts == []
    # in the request only, never in the saved conversation
    assert _images(out["messages"]) == []
    assert agent.geotech_build["show_uploads"].startswith("on: ")
    assert f"{ENV}=1" in agent.geotech_build["switches"]


def test_on_one_shown_image_is_one_image_call_under_the_ceiling(
        on, monkeypatch):
    monkeypatch.setenv("GEOTECH_TURN_MAX_IMAGE_CALLS", "2")
    engine = FakeEngine()
    tool = _look_tool(engine, {"chart.png": _chart()})
    args = {"attachment_key": "chart.png"}
    with keep_work.turn_scope() as scope:
        first = json.loads(_marked(lambda: tool.invoke(args)))
        assert scope.image_calls == 1 and first["image_id"]
        assert first["image_calls"].startswith(
            "this look made 1 image call; this turn has made 1 of its "
            "ceiling of 2")
        _marked(lambda: tool.invoke(args))
        assert scope.image_calls == 2
        third = json.loads(_marked(lambda: tool.invoke(args)))
        assert scope.image_refused == 1
        assert "image_id" not in third
        assert "reached its ceiling of 2 image calls" in json.dumps(third)
        assert keep_work.image_call_facts(scope)["image_calls"] == 2
    assert engine.prompts == []


def test_on_a_pdf_upload_is_today_s_answer_word_for_word(on, gt):
    engine = FakeEngine()
    tool = _look_tool(engine, {"review_set.pdf": gt.pdf})
    args = {"attachment_key": "review_set.pdf"}
    today = tool.invoke(args)
    put = []
    real_put = inline_store.put
    inline_store.put = lambda *a, **k: put.append(1) or real_put(*a, **k)
    try:
        assert _marked(lambda: tool.invoke(args)) == today
    finally:
        inline_store.put = real_put
    assert put == [] and "image_id" not in today
    assert engine.prompts == []


def test_on_a_file_opened_by_its_path_is_not_shown(on, tmp_path):
    path = tmp_path / "sketch.png"
    path.write_bytes(_chart())
    engine = FakeEngine("a sketch")
    out = json.loads(_marked(lambda: _look_tool(engine, {}).invoke(
        {"attachment_key": str(path)})))
    assert out.get("analysis") == "a sketch" and "image_id" not in out


def test_on_an_image_too_large_to_show_is_read_and_the_result_says_so(
        on, monkeypatch):
    monkeypatch.setattr(vision_tools, "INLINE_IMAGE_MAX_BYTES", 100)
    engine = FakeEngine("three bars")
    out = json.loads(_marked(lambda: _look_tool(
        engine, {"chart.png": _chart()}).invoke(
            {"attachment_key": "chart.png"})))
    assert out["analysis"] == "three bars" and "image_id" not in out
    assert "not shown to you" in out["not_shown"]
    assert "300 x 200 px" in out["not_shown"]
    assert len(engine.prompts) == 1


def test_on_tiles_and_turn_are_said_to_be_unused(on):
    out = json.loads(_marked(lambda: _look_tool(
        FakeEngine(), {"chart.png": _chart()}).invoke(
            {"attachment_key": "chart.png", "tiles": "2", "turn": 90})))
    assert out["image_id"]
    assert "tiles is not used" in out["unused"]
    assert "turn is not used" in out["unused"]


def test_on_a_model_that_takes_no_images_gets_today_s_look(on, gt):
    seen = []

    def after(messages):
        seen.append(len(_images(messages)))
        return AIMessage(content="read")

    model = _model([_call("analyze_image", {"attachment_key": "chart.png"}),
                    after])
    model.profile = {"image_inputs": False}
    engine = SameModelEngine(model)
    agent = _agent(model, gt, engine, **{"chart.png": _chart()})
    _turn(agent)
    assert seen == [0] and len(engine.prompts) == 1
    note = agent.geotech_build["show_uploads"]
    assert note.startswith("off: ") and "takes no images" in note


def test_on_images_read_by_another_model_leave_the_page_as_today(on, gt):
    """The main model is not known to take images when the page's images
    go to another model."""
    model = _model([_call("analyze_image", {"attachment_key": "chart.png"}),
                    AIMessage(content="read")])
    engine = FakeEngine()
    agent = _agent(model, gt, engine, **{"chart.png": _chart()})
    _turn(agent)
    assert len(engine.prompts) == 1
    assert agent.geotech_build["show_uploads"].startswith(
        "off: the page's images are read by another model")
    ok, why = upload_view.takes_images(model, None)
    assert not ok and "no vision engine" in why


def test_on_the_reading_helper_s_look_stays_a_vision_call(on, gt):
    """Only the main agent's analyze_image is marked: the helper, which
    holds the same tool, reads the upload through a vision call."""
    seen = {"helper": [], "primary": []}

    def helper_after(messages):
        seen["helper"].append(len(_images(messages)))
        assert "analysis" in _last_tool(messages)
        return AIMessage(content="Helper: three bars.")

    def primary_after(messages):
        seen["primary"].append(len(_images(messages)))
        return AIMessage(content="Done.")

    model = _model([
        _call("task", {"description": "Read the bars in chart.png.",
                       "subagent_type": "general-purpose"}),
        _call("analyze_image", {"attachment_key": "chart.png"}, 2),
        helper_after, primary_after])
    engine = SameModelEngine(model)
    out = _turn(_agent(model, gt, engine, **{"chart.png": _chart()}))
    assert out["messages"][-1].content == "Done."
    assert seen == {"helper": [0], "primary": [0]}
    assert len(engine.prompts) == 1


def test_the_middleware_marks_only_analyze_image():
    mw = upload_view.UploadImageMiddleware(engine=FakeEngine())
    got = {}
    for name in ("task", "render_region", "analyze_image"):
        mw.wrap_tool_call(SimpleNamespace(tool_call={"name": name}),
                          lambda r, n=name: got.__setitem__(
                              n, upload_view.active()))
    assert got == {"task": False, "render_region": False,
                   "analyze_image": True}
    assert upload_view.active() is False


def test_the_image_goes_as_the_side_call_s_image_part(on, monkeypatch):
    """The same ``image_url`` block the vision side call sends, with the
    engine's detail, which each engine family's chat model translates."""
    monkeypatch.setenv("GEOTECH_VISION_DETAIL", "high")
    image_id = json.loads(_marked(lambda: _look_tool(
        FakeEngine(), {"chart.png": _chart()}).invoke(
            {"attachment_key": "chart.png"})))["image_id"]
    from langchain_core.messages import ToolMessage
    mw = upload_view.UploadImageMiddleware(engine=FakeEngine())
    request = SimpleNamespace(
        messages=[HumanMessage(content="q"), ToolMessage(
            content=json.dumps({"image_id": image_id}), tool_call_id="c1")],
        override=lambda messages: SimpleNamespace(messages=messages))
    shown = mw._with_images(request).messages[-1]
    block = _images([shown])[0]
    assert set(block) == {"type", "image_url"}
    assert block["image_url"]["detail"] == "high"
    assert block["image_url"]["url"].startswith("data:image/png;base64,")
    # the OpenAI-compatible route passes it through as it is
    from aec_agent_core.deep.databricks_bridge import _lc_message_to_openai
    assert block in _lc_message_to_openai(shown)["content"]
    # the hosted platform's SDK route reads it as an image
    try:
        from aec_agent_core.palantir_sdk_engine import _image_url
    except Exception:                                  # noqa: BLE001
        return
    assert _image_url(block) == block["image_url"]["url"]


def test_the_switch_is_a_later_switch_the_suite_can_set(on):
    assert ENV == "GEOTECH_VISION_SHOW_UPLOADS"
    assert ENV in review_flags.LATER_ENVS
    assert ENV not in review_flags.ALL_ENVS
    with review_flags.switches({}):
        assert not review_flags.show_uploads()
    with review_flags.switches({ENV: "1"}):
        assert review_flags.show_uploads()
        assert f"{ENV}=1" in review_flags.describe()
    assert review_flags.show_uploads()
