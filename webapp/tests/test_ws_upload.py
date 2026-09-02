"""Tests for the websocket-based uploader (proxy-safe attach path)."""

import base64
import os

import pytest

from webapp import ws_upload


class TestUploadMode:
    def test_default_http(self, monkeypatch):
        monkeypatch.delenv("GEOTECH_UPLOAD_MODE", raising=False)
        assert ws_upload.upload_mode() == "http"

    def test_ws_selected(self, monkeypatch):
        monkeypatch.setenv("GEOTECH_UPLOAD_MODE", "ws")
        assert ws_upload.upload_mode() == "ws"
        monkeypatch.setenv("GEOTECH_UPLOAD_MODE", "WS ")
        assert ws_upload.upload_mode() == "ws"

    def test_garbage_falls_back_to_http(self, monkeypatch):
        monkeypatch.setenv("GEOTECH_UPLOAD_MODE", "carrier-pigeon")
        assert ws_upload.upload_mode() == "http"


class TestDecodeComponentValue:
    def test_empty_and_none(self):
        assert ws_upload.decode_component_value(None) == ([], [])
        assert ws_upload.decode_component_value([]) == ([], [])

    def test_round_trip(self):
        payload = b"%PDF fake report bytes"
        value = [{"name": "report.pdf",
                  "b64": base64.b64encode(payload).decode(),
                  "size": len(payload)}]
        pairs, errors = ws_upload.decode_component_value(value)
        assert errors == []
        assert pairs == [("report.pdf", payload)]

    def test_multiple_files(self):
        value = [{"name": f"f{i}.txt",
                  "b64": base64.b64encode(f"data{i}".encode()).decode()}
                 for i in range(3)]
        pairs, errors = ws_upload.decode_component_value(value)
        assert [n for n, _ in pairs] == ["f0.txt", "f1.txt", "f2.txt"]
        assert not errors

    def test_bad_base64_reported_not_raised(self):
        value = [{"name": "ok.txt", "b64": base64.b64encode(b"x").decode()},
                 {"name": "bad.txt", "b64": "!!!not-base64!!!"}]
        pairs, errors = ws_upload.decode_component_value(value)
        assert [n for n, _ in pairs] == ["ok.txt"]
        assert len(errors) == 1

    def test_oversize_rejected(self):
        big = b"x" * (ws_upload.MAX_FILE_MB * 1024 * 1024 + 1)
        value = [{"name": "huge.bin", "b64": base64.b64encode(big).decode()}]
        pairs, errors = ws_upload.decode_component_value(value)
        assert pairs == []
        assert "cap" in errors[0]

    def test_empty_file_rejected(self):
        value = [{"name": "empty.txt", "b64": ""}]
        pairs, errors = ws_upload.decode_component_value(value)
        assert pairs == []
        assert "empty" in errors[0]

    def test_non_list_value(self):
        pairs, errors = ws_upload.decode_component_value({"name": "x"})
        assert pairs == [] and len(errors) == 1


class TestComponentAssets:
    def test_component_html_ships(self):
        # The wheel must carry the component page (pyproject package-data).
        path = os.path.join(ws_upload._COMPONENT_DIR, "index.html")
        assert os.path.exists(path)
        src = open(path, encoding="utf-8").read()
        # Core protocol messages present
        for token in ("streamlit:componentReady", "streamlit:setComponentValue",
                      "streamlit:render", "streamlit:setFrameHeight"):
            assert token in src, f"missing protocol message {token}"


class TestWsModeAppBoot:
    """The app must render cleanly with the ws uploader active (AppTest)."""

    def test_app_boots_in_ws_mode(self, monkeypatch, tmp_path):
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest
        monkeypatch.setenv("GEOTECH_UPLOAD_MODE", "ws")
        monkeypatch.setenv("GEOTECH_CONVERSATIONS_DIR", str(tmp_path))
        app = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "app.py")
        at = AppTest.from_file(app, default_timeout=60).run()
        assert not at.exception, [str(e.value) for e in at.exception]
