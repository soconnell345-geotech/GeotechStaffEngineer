"""Two limits of ``GEOTECH_VISION_SHOW_UPLOADS`` noted with 1.3.0, repaired:

1. a second look at one upload stored a second copy of its image;
2. the shown image waited in ONE store of 96 images for the whole process,
   so another conversation's images could push it out before its turn came
   to be shown, and nothing said so.

The images a turn stores are now the turn's own (on its scope, gone when it
ends), the same bytes with the same description are one stored image, and
an upload pushed out by its own turn's later images is said, in its tool
result and in the turn's record, not to have been shown.

Made-up images and names only.
"""

import contextvars
import io
import json

import pytest

pytest.importorskip("langchain.agents.middleware")
PIL = pytest.importorskip("PIL.Image")

from langchain_core.messages import (  # noqa: E402
    AIMessage, HumanMessage, ToolMessage)

from aec_agent_core import inline_store, vision_tools  # noqa: E402
from aec_agent_core.deep import keep_work, upload_view  # noqa: E402


class _Engine:
    model = None
    accepts_jpeg = False

    def analyze_image(self, image, prompt, **kw):  # pragma: no cover
        raise AssertionError("a shown upload runs no vision call")


class _Request:
    def __init__(self, messages):
        self.messages = messages

    def override(self, messages):
        return _Request(messages)


@pytest.fixture(autouse=True)
def _clean():
    inline_store.clear()
    yield
    inline_store.clear()


def _png(shade: int = 255) -> bytes:
    buf = io.BytesIO()
    PIL.new("RGB", (64, 48), (shade, shade, shade)).save(buf, "PNG")
    return buf.getvalue()


def _look(attachments, key) -> dict:
    """The main agent's ``analyze_image`` of the upload ``key`` under the
    switch (the middleware's mark), in the running turn."""
    token = upload_view._ON.set(True)
    try:
        return json.loads(vision_tools._dispatch_analyze_image(
            {"attachment_key": key, "prompt": "what is it"},
            _Engine(), attachments))
    finally:
        upload_view._ON.reset(token)


def _in_turn(scope):
    """A context in which ``scope`` is the running turn, as one tester's
    turn runs beside another's in one process."""
    ctx = contextvars.copy_context()
    ctx.run(keep_work._TURN.set, scope)
    return ctx


def _step(results):
    """One step's messages: the user's turn, the model's calls, and the
    tool results ``results`` (their JSON texts)."""
    calls = [{"name": "analyze_image", "args": {}, "id": f"call_{i}"}
             for i in range(len(results))]
    return [HumanMessage(content="What do the photos show?"),
            AIMessage(content="", tool_calls=calls),
            *[ToolMessage(content=raw, tool_call_id=f"call_{i}")
              for i, raw in enumerate(results)]]


# ---------------------------------------------------------------------------
# (1) a second look stores no second copy
# ---------------------------------------------------------------------------
def test_a_second_look_at_one_upload_stores_no_second_copy():
    files = {"site_photo.png": _png(), "other_photo.png": _png(120)}
    scope = keep_work.TurnScope()
    with keep_work.turn_scope(scope):
        first = _look(files, "site_photo.png")
        other = _look(files, "other_photo.png")
        second = _look(files, "site_photo.png")
        held = scope.__dict__.get("inline_images", ({},))[0]
        assert second["image_id"] == first["image_id"]
        assert other["image_id"] != first["image_id"]
        assert len(held) == 2
        # the second look makes it the newest again: it is the last to go
        assert list(held)[-1] == first["image_id"]
        assert "shown to you" in second["note"]


def test_the_same_bytes_under_another_description_are_another_image():
    with keep_work.turn_scope():
        a = inline_store.put(_png(), {"page": 0, "view": [0, 0, 64, 48]})
        b = inline_store.put(_png(), {"page": 1, "view": [0, 0, 64, 48]})
        again = inline_store.put(_png(), {"page": 0, "view": [0, 0, 64, 48]})
        held = keep_work.current_turn().__dict__["inline_images"][0]
    assert a != b and again == a and len(held) == 2


# ---------------------------------------------------------------------------
# (2) another conversation's images push nothing out
# ---------------------------------------------------------------------------
def test_another_conversation_s_images_cannot_push_an_upload_out():
    mine, theirs = keep_work.TurnScope(), keep_work.TurnScope()
    me, them = _in_turn(mine), _in_turn(theirs)
    out = me.run(_look, {"site_photo.png": _png()}, "site_photo.png")
    for i in range(inline_store.MAX_IMAGES + 4):
        them.run(inline_store.put, _png(i), {"label": f"photo {i}"})
    # and images stored outside any turn (a library call, a probe)
    for i in range(inline_store.MAX_IMAGES + 4):
        inline_store.put(_png(i), {"label": f"loose {i}"})
    got = me.run(inline_store.get, out["image_id"])
    assert got is not None and got[1]["source"] == "site_photo.png"
    # the model call that follows still carries it
    request = me.run(upload_view.UploadImageMiddleware(keep=2)._with_images,
                     _Request(_step([json.dumps(out)])))
    assert any(isinstance(b, dict) and b.get("type") == "image_url"
               for b in request.messages[-1].content)
    # its result is as the tool gave it, and its record has no such key
    assert json.loads(request.messages[2].content) == out
    assert "uploads_not_shown" not in keep_work.image_call_facts(mine)
    # and the other turn never sees it
    assert them.run(inline_store.get, out["image_id"]) is None


def test_a_turn_s_images_are_its_own_and_go_with_it():
    scope = keep_work.TurnScope()
    with keep_work.turn_scope(scope):
        out = _look({"site_photo.png": _png()}, "site_photo.png")
        assert inline_store.get(out["image_id"]) is not None
        assert not inline_store._STORE          # nothing process-wide
    assert "inline_images" not in scope.__dict__
    assert inline_store.get(out["image_id"]) is None


# ---------------------------------------------------------------------------
# (2) an upload no longer there when its turn comes is said not to be shown
# ---------------------------------------------------------------------------
def test_an_upload_pushed_out_before_it_is_shown_is_said_not_to_be_shown():
    scope = keep_work.TurnScope()
    middleware = upload_view.UploadImageMiddleware(keep=2)
    with keep_work.turn_scope(scope):
        seen = _look({"first_photo.png": _png(7)}, "first_photo.png")
        # shown at the model call that follows its look
        earlier = middleware._with_images(_Request(_step([json.dumps(seen)])))
        assert len([b for b in earlier.messages[-1].content
                    if b.get("type") == "image_url"]) == 1
        # one later step looks at more uploads than the turn holds
        files = {f"photo_{i:03d}.png": _png(i % 250)
                 for i in range(inline_store.MAX_IMAGES + 1)}
        # same shade, another name: another upload, another stored image
        results = [_look(files, name) for name in files]
        assert len({r["image_id"] for r in results}) == len(files)
        assert inline_store.get(results[0]["image_id"]) is None
        request = middleware._with_images(_Request(
            _step([json.dumps(seen)] + [json.dumps(r) for r in results])))
    tools = [json.loads(m.content) for m in request.messages
             if isinstance(m, ToolMessage)]
    lost = tools[1]
    assert "photo_000.png was NOT shown to you" in lost["not_shown"]
    assert "never in front of you" in lost["not_shown"]
    assert "note" not in lost       # no longer says it is shown
    # the upload that WAS shown before it fell out is not called unseen,
    # and the ones still held keep their results
    assert "not_shown" not in tools[0] and "not_shown" not in tools[2]
    assert "shown to you" in tools[2]["note"]
    # the turn's record names it, once however many calls follow
    assert keep_work.image_call_facts(scope)["uploads_not_shown"] == [
        "photo_000.png"]
    with keep_work.turn_scope(scope):
        pass
    assert keep_work.image_call_facts(scope)["uploads_not_shown"] == [
        "photo_000.png"]

