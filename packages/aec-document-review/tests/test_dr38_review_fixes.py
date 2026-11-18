"""DR-38's review (``module_work/dr38/REVIEW.md``): what the main model is
told about an upload it is shown under ``GEOTECH_VISION_SHOW_UPLOADS``,
offline, fake models, made-up pictures."""

import io
import json

import pytest

pytest.importorskip("planlens.tools")
pymupdf = pytest.importorskip("pymupdf")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from aec_agent_core import inline_store  # noqa: E402
from tests.test_dr38_show_uploads import (  # noqa: E402,F401
    SameModelEngine, _agent, _chart, _clean, _images, _look_tool, _marked,
    _turn, gt, on)
from tests.test_review_harness_offline import FakeEngine, _model  # noqa: E402


def _looks(names):
    """One model step that looks at every upload in ``names`` at once."""
    return AIMessage(content="", tool_calls=[
        {"name": "analyze_image", "args": {"attachment_key": name},
         "id": f"call_{i}"} for i, name in enumerate(names)])


def _said(messages):
    """The text of the image message the model was handed."""
    return " ".join(
        b.get("text", "") for m in messages if isinstance(m, HumanMessage)
        and isinstance(m.content, list) for b in m.content
        if isinstance(b, dict) and b.get("type") == "text")


def test_an_upload_left_out_of_a_batch_is_named_as_not_shown(on, gt):
    """Three uploads looked at in one step, two images shown at a time: the
    model is told which one it was not shown, by name."""
    seen = []

    def after(messages):
        seen.append((len(_images(messages)), _said(messages)))
        return AIMessage(content="done")

    model = _model([_looks(["first.png", "second.png", "third.png"]), after])
    agent = _agent(model, gt, SameModelEngine(model), **{
        "first.png": _chart(), "second.png": _chart(), "third.png": _chart()})
    _turn(agent)
    shown, said = seen[0]
    assert shown == 2
    assert "NOT shown" in said and "first.png" in said
    not_shown = said.split("NOT shown", 1)[1]
    assert "second.png" not in not_shown and "third.png" not in not_shown


def test_a_batch_that_fits_says_nothing_more(on, gt):
    seen = []

    def after(messages):
        seen.append((len(_images(messages)), _said(messages)))
        return AIMessage(content="done")

    model = _model([_looks(["first.png", "second.png"]), after])
    agent = _agent(model, gt, SameModelEngine(model), **{
        "first.png": _chart(), "second.png": _chart()})
    _turn(agent)
    assert seen[0][0] == 2 and "NOT shown" not in seen[0][1]


def test_a_shown_upload_keeps_the_note_today_s_read_gives(on):
    """A three-page TIFF: today's read says only its first page was sent,
    and so does the result of a look that shows it."""
    Image = pytest.importorskip("PIL.Image")
    pages = [Image.new("RGB", (120, 80), c) for c in ("white", "red", "blue")]
    buf = io.BytesIO()
    pages[0].save(buf, format="TIFF", save_all=True, append_images=pages[1:])
    tool = _look_tool(FakeEngine("read"), {"scan.tif": buf.getvalue()})
    today = json.loads(tool.invoke({"attachment_key": "scan.tif"}))
    assert "page 1 of 3" in today["image_note"]
    shown = json.loads(_marked(lambda: tool.invoke(
        {"attachment_key": "scan.tif"})))
    assert inline_store.get(shown["image_id"]) is not None
    assert shown["image_note"] == today["image_note"]
