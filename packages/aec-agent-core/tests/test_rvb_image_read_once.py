"""Release review of 2026-11-18 (whole change, B): DR-38 meets EV-20. With
``GEOTECH_VISION_SHOW_UPLOADS`` on, an upload that is NOT shown to the main
model (too large once shrunk, or not an image) is handed to today's read,
which went through ``_dispatch_analyze_image`` a second time and wrote the
run record's ``image_read`` event twice for the one tool call.

Made-up values only.
"""

import io
import json

import pytest

from aec_agent_core import vision_tools
from aec_agent_core.deep import keep_work, upload_view

PIL = pytest.importorskip("PIL.Image")


class _Engine:
    model = None
    accepts_jpeg = False

    def analyze_image(self, image, prompt, **kw):
        keep_work.take_image_call()
        return "a made-up reading"


def _png() -> bytes:
    buf = io.BytesIO()
    PIL.new("RGB", (64, 48), "white").save(buf, "PNG")
    return buf.getvalue()


def _look(monkeypatch, attachments, key):
    events = []
    monkeypatch.setattr(vision_tools, "_note_image_read",
                        lambda k, data, src: events.append((k, src)))
    scope = keep_work.TurnScope()
    with keep_work.turn_scope(scope):
        token = upload_view._ON.set(True)
        try:
            raw = vision_tools._dispatch_analyze_image(
                {"attachment_key": key, "prompt": "what is it"},
                _Engine(), attachments)
        finally:
            upload_view._ON.reset(token)
    return events, scope, raw


def test_an_upload_shown_is_noted_once(monkeypatch):
    events, scope, raw = _look(monkeypatch, {"photo.png": _png()},
                               "photo.png")
    assert "image_id" in json.loads(raw)
    assert len(events) == 1 and scope.image_calls == 1


def test_an_upload_too_large_to_show_is_noted_once(monkeypatch):
    monkeypatch.setattr(vision_tools, "INLINE_IMAGE_MAX_BYTES", 10)
    events, scope, raw = _look(monkeypatch, {"photo.png": _png()},
                               "photo.png")
    assert "not_shown" in json.loads(raw)       # read by a vision call
    assert scope.image_calls == 1
    assert events == [("photo.png", "attachment")]


def test_an_upload_that_is_not_an_image_is_noted_once(monkeypatch):
    events, _scope, _raw = _look(
        monkeypatch, {"notes.txt": b"made-up words, not an image"},
        "notes.txt")
    assert events == [("notes.txt", "attachment")]
