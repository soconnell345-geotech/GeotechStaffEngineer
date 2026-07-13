"""Tests for funhouse_agent._fileio — verified file writes + placeholder rescue.

Regression for the live Databricks failure (2026-06-12): a 241 kB calc-package
HTML written to /Workspace via plain file I/O "succeeded" but the workspace
stored the literal 11-byte string PLACEHOLDER. Tool responses must detect a
target filesystem that did not store the content and rescue to the temp dir.
"""

import contextlib
import json
import os
import sys
import tempfile
import types

import pytest

from funhouse_agent._fileio import (
    default_output_dir, rescue_write, save_verified, workspace_api_upload,
    workspace_write_hint, written_file_problem,
)


@contextlib.contextmanager
def _fake_databricks_sdk(client_factory):
    """Inject a fake ``databricks.sdk`` module exposing ``WorkspaceClient``.

    databricks-sdk is an OPTIONAL dependency not installed in this venv, so the
    real SDK path is exercised against a mock injected into ``sys.modules``.
    """
    saved = {k: sys.modules.get(k) for k in ("databricks", "databricks.sdk")}
    sdk = types.ModuleType("databricks.sdk")
    sdk.WorkspaceClient = client_factory
    dbk = types.ModuleType("databricks")
    dbk.sdk = sdk
    sys.modules["databricks"] = dbk
    sys.modules["databricks.sdk"] = sdk
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class _RecordingWorkspace:
    """A stand-in WorkspaceClient.workspace with a configurable read-back size."""

    def __init__(self, store, readback_size="match", upload_exc=None,
                 status_exc=None):
        self._store = store
        self._readback = readback_size
        self._upload_exc = upload_exc
        self._status_exc = status_exc

    def mkdirs(self, path):
        self._store.setdefault("mkdirs", []).append(path)

    def upload(self, path, content, **kwargs):
        if self._upload_exc:
            raise self._upload_exc
        # The real SDK takes a binary stream; accept a stream or raw bytes.
        data = content.read() if hasattr(content, "read") else bytes(content)
        self._store["upload"] = (path, data, kwargs)

    def get_status(self, path):
        if self._status_exc:
            raise self._status_exc
        sent = len(self._store.get("upload", ("", b""))[1])
        size = sent if self._readback == "match" else self._readback
        return types.SimpleNamespace(size=size)


def _client_factory(store, **kw):
    def make():
        client = types.SimpleNamespace()
        client.workspace = _RecordingWorkspace(store, **kw)
        return client
    return make


class TestWrittenFileProblem:
    def test_exact_content_ok(self, tmp_path):
        f = tmp_path / "a.html"
        f.write_bytes(b"<!DOCTYPE html><html>body</html>")
        assert written_file_problem(str(f), b"<!DOCTYPE html><html>body</html>") is None

    def test_crlf_inflated_text_ok(self, tmp_path):
        # Text-mode writes on Windows translate \n -> \r\n; must not flag.
        content = "<!DOCTYPE html>\n<html>\nbody\n</html>\n"
        f = tmp_path / "a.html"
        with open(f, "w", encoding="utf-8") as fh:  # newline translation ON
            fh.write(content)
        assert written_file_problem(str(f), content.encode("utf-8")) is None

    def test_placeholder_file_detected(self, tmp_path):
        f = tmp_path / "a.html"
        f.write_bytes(b"PLACEHOLDER")
        problem = written_file_problem(str(f), b"<!DOCTYPE html>" + b"x" * 240000)
        assert problem is not None and "11 bytes" in problem

    def test_wrong_head_detected(self, tmp_path):
        f = tmp_path / "a.html"
        f.write_bytes(b"Z" * 500)
        problem = written_file_problem(str(f), b"<!DOCTYPE html>" + b"x" * 100)
        assert problem is not None and "does not start" in problem

    def test_missing_file(self, tmp_path):
        problem = written_file_problem(str(tmp_path / "nope.html"), b"x")
        assert problem is not None and "no file exists" in problem

    def test_no_expected_content_size_only(self, tmp_path):
        f = tmp_path / "a.pdf"
        f.write_bytes(b"")
        assert "empty" in written_file_problem(str(f), None)
        f.write_bytes(b"%PDF-1.4 ...")
        assert written_file_problem(str(f), None) is None


class TestRescueWrite:
    def test_rescue_lands_in_tempdir_verified(self):
        content = b"<!DOCTYPE html>rescued"
        rescue = rescue_write("/Workspace/Users/x/out_test_fileio.html", content)
        assert rescue is not None
        try:
            assert os.path.dirname(rescue) == os.path.abspath(tempfile.gettempdir())
            assert open(rescue, "rb").read() == content
        finally:
            os.remove(rescue)

    def test_rescue_refuses_same_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
        target = str(tmp_path / "x.html")
        assert rescue_write(target, b"abc") is None


class TestDefaultOutputDir:
    def test_local_default_is_cwd(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        monkeypatch.delenv("GEOTECH_DEFAULT_OUTPUT_DIR", raising=False)
        assert default_output_dir() == ""

    def test_databricks_uses_tempdir(self, monkeypatch):
        monkeypatch.delenv("GEOTECH_DEFAULT_OUTPUT_DIR", raising=False)
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
        assert default_output_dir() == tempfile.gettempdir()

    def test_env_override_wins_and_is_absolute(self, monkeypatch, tmp_path):
        # The host-chosen working folder (GEOTECH_DEFAULT_OUTPUT_DIR) is the
        # highest-precedence DEFAULT — it beats even the Databricks temp fallback.
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
        assert default_output_dir() == os.path.abspath(str(tmp_path))

    def test_blank_env_is_ignored(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", "   ")
        assert default_output_dir() == ""


class TestWorkspaceHint:
    def test_hint_only_for_workspace_paths(self):
        assert "PLACEHOLDER" in workspace_write_hint("/Workspace/Users/x/a.html")
        assert workspace_write_hint("/tmp/a.html") == ""


class TestCalcPackagePlaceholderRescue:
    def test_build_response_detects_and_rescues(self, tmp_path, monkeypatch):
        """Simulate the live failure: the writer reports success but the
        target stored only PLACEHOLDER."""
        import calc_package as cp
        from funhouse_agent.adapters.calc_package import _build_response

        big_html = "<!DOCTYPE html>" + "x" * 5000

        def fake_generate(module, result, analysis, output_path, format, **meta):
            with open(output_path, "w") as f:
                f.write("PLACEHOLDER")
            return big_html

        monkeypatch.setattr(cp, "generate_calc_package", fake_generate)
        out = str(tmp_path / "pkg_test_fileio.html")
        resp = _build_response("lateral_pile", None, None,
                               {"output_path": out}, analysis_type="Lateral Pile")
        assert resp["status"] == "error"
        assert resp["file_exists"] is False
        assert "did not store the content" in resp["error"]
        assert resp.get("rescue_path")
        try:
            assert open(resp["rescue_path"], "rb").read() == big_html.encode()
        finally:
            os.remove(resp["rescue_path"])

    def test_build_response_good_write_still_success(self, tmp_path):
        from funhouse_agent.adapters.calc_package import _generate_bearing_capacity_package
        out = str(tmp_path / "bc.html")
        resp = _generate_bearing_capacity_package({
            "width": 2.0, "unit_weight": 18.0, "friction_angle": 30.0,
            "output_path": out,
        })
        assert resp["status"] == "success"
        assert resp["file_exists"] is True
        assert "rescue_path" not in resp


class TestWorkspaceApiUpload:
    """The optional databricks-sdk durable /Workspace write path."""

    def test_sdk_absent_returns_not_ok(self):
        # databricks-sdk is not installed in this venv → import fails → ok False,
        # so the caller falls back to a plain write + verify.
        res = workspace_api_upload("/Workspace/Users/x/a.txt", b"hi")
        assert res["ok"] is False
        assert "error" in res

    def test_sdk_present_success_verified(self):
        store = {}
        with _fake_databricks_sdk(_client_factory(store)):
            res = workspace_api_upload("/Workspace/Users/x/report.pdf", b"PDFDATA")
        assert res == {"ok": True, "size": 7, "verified": True}
        assert store["upload"][0] == "/Workspace/Users/x/report.pdf"
        assert store["upload"][1] == b"PDFDATA"
        assert store["mkdirs"] == ["/Workspace/Users/x"]

    def test_str_content_is_encoded(self):
        store = {}
        with _fake_databricks_sdk(_client_factory(store)):
            res = workspace_api_upload("/Workspace/x.txt", "héllo")
        assert res["ok"] is True
        assert store["upload"][1] == "héllo".encode("utf-8")

    def test_size_mismatch_is_not_ok(self):
        store = {}
        with _fake_databricks_sdk(_client_factory(store, readback_size=1)):
            res = workspace_api_upload("/Workspace/x.txt", b"abcdef")
        assert res["ok"] is False and "stored 1 bytes" in res["error"]

    def test_upload_raises_is_not_ok(self):
        store = {}
        with _fake_databricks_sdk(
                _client_factory(store, upload_exc=RuntimeError("boom"))):
            res = workspace_api_upload("/Workspace/x.txt", b"data")
        assert res["ok"] is False and "boom" in res["error"]

    def test_size_unavailable_is_ok_unverified(self):
        store = {}
        with _fake_databricks_sdk(
                _client_factory(store, status_exc=RuntimeError("no status"))):
            res = workspace_api_upload("/Workspace/x.txt", b"data")
        assert res == {"ok": True, "size": 4, "verified": False}


class TestSaveFileWorkspaceRouting:
    """_dispatch_save_file routes /Workspace default-writer saves through the API."""

    def test_workspace_default_writer_uses_api_on_success(self, monkeypatch):
        import funhouse_agent._fileio as fio
        import funhouse_agent.vision_tools as vt

        calls = {}

        def fake_api(path, content):
            calls["api"] = path
            return {"ok": True, "size": 11, "verified": True}

        monkeypatch.setattr(fio, "workspace_api_upload", fake_api)

        def no_write(path, content):
            raise AssertionError("plain writer must not run when the API succeeds")

        monkeypatch.setattr(vt, "_default_save_fn", no_write)
        res = json.loads(vt._dispatch_save_file(
            {"path": "/Workspace/Users/x/out.txt", "content": "hello world"},
            vt._default_save_fn))
        assert res["save_method"] == "workspace_api"
        assert res["file_exists"] is True
        assert res["file_size_bytes"] == 11
        assert calls["api"] == "/Workspace/Users/x/out.txt"

    def test_workspace_api_failure_falls_back_to_plain_write(self, tmp_path,
                                                             monkeypatch):
        import funhouse_agent._fileio as fio
        import funhouse_agent.vision_tools as vt

        monkeypatch.setattr(fio, "workspace_api_upload",
                            lambda p, c: {"ok": False, "error": "sdk absent"})
        target = tmp_path / "out.txt"

        def writer(path, content):
            mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
            with open(target, mode) as f:
                f.write(content)
            return str(target)

        monkeypatch.setattr(vt, "_default_save_fn", writer)
        res = json.loads(vt._dispatch_save_file(
            {"path": "/Workspace/Users/x/out.txt", "content": "hello"},
            vt._default_save_fn))
        assert res["file_exists"] is True
        assert res.get("save_method") != "workspace_api"
        assert "workspace_api_note" in res

    def test_non_workspace_path_never_calls_api(self, tmp_path, monkeypatch):
        import funhouse_agent._fileio as fio
        import funhouse_agent.vision_tools as vt

        def boom_api(path, content):
            raise AssertionError("API must not be tried for a non-/Workspace path")

        monkeypatch.setattr(fio, "workspace_api_upload", boom_api)
        out = tmp_path / "a.txt"
        res = json.loads(vt._dispatch_save_file(
            {"path": str(out), "content": "hi"}, vt._default_save_fn))
        assert res["file_exists"] is True

    def test_custom_save_fn_workspace_path_not_bypassed(self, monkeypatch):
        import funhouse_agent._fileio as fio
        import funhouse_agent.vision_tools as vt

        def boom_api(path, content):
            raise AssertionError("API must not run for a custom save_fn")

        monkeypatch.setattr(fio, "workspace_api_upload", boom_api)
        res = json.loads(vt._dispatch_save_file(
            {"path": "/Workspace/Users/x/out.txt", "content": "hi"},
            lambda p, c: f"/dbfs/{p}"))
        assert res["file_exists"] is False
        assert "note" in res  # custom-fn remote-path note; API correctly skipped

    def test_save_fn_exception_rescues_content(self):
        import funhouse_agent.vision_tools as vt

        def raising(path, content):
            raise PermissionError("denied")

        res = json.loads(vt._dispatch_save_file(
            {"path": "rescue_me_test_fileio.txt", "content": "IMPORTANT"},
            raising))
        assert "error" in res
        assert res.get("rescue_path")
        try:
            assert open(res["rescue_path"], "rb").read() == b"IMPORTANT"
        finally:
            os.remove(res["rescue_path"])


class TestSaveVerified:
    """The reusable high-level verified save (used by plot adapters etc.)."""

    def test_local_success(self, tmp_path):
        out = tmp_path / "a.html"
        res = save_verified(str(out), "<html>hi</html>")
        assert res["file_exists"] is True
        assert res["file_size_bytes"] == out.stat().st_size > 0
        assert os.path.isabs(res["saved"])
        assert "error" not in res

    def test_bytes_content(self, tmp_path):
        out = tmp_path / "a.bin"
        res = save_verified(str(out), b"\x00\x01\x02BIN")
        assert res["file_exists"] is True
        assert out.read_bytes() == b"\x00\x01\x02BIN"

    def test_workspace_uses_api(self):
        store = {}
        with _fake_databricks_sdk(_client_factory(store)):
            res = save_verified("/Workspace/Users/x/plot.html", "<html>fig</html>")
        assert res["save_method"] == "workspace_api"
        assert res["file_exists"] is True
        assert store["upload"][0] == "/Workspace/Users/x/plot.html"

    def test_workspace_api_absent_falls_back(self, tmp_path, monkeypatch):
        import funhouse_agent._fileio as fio
        target = tmp_path / "out.html"

        def writer(path, content):
            mode = "wb" if isinstance(content, (bytes, bytearray)) else "w"
            with open(target, mode) as f:
                f.write(content)
            return str(target)

        monkeypatch.setattr(fio, "_local_write", writer)
        res = save_verified("/Workspace/Users/x/out.html", "<html>hi</html>")
        assert res["file_exists"] is True
        assert res.get("save_method") != "workspace_api"
        assert "workspace_api_note" in res

    def test_placeholder_write_rescues(self, tmp_path, monkeypatch):
        import funhouse_agent._fileio as fio
        target = tmp_path / "ph.html"

        def placeholder(path, content):
            target.write_bytes(b"PLACEHOLDER")
            return str(target)

        monkeypatch.setattr(fio, "_local_write", placeholder)
        res = save_verified(str(target), "<html>" + "y" * 3000 + "</html>")
        assert res["file_exists"] is False
        assert "did not store the content" in res["error"]
        assert res.get("rescue_path")
        os.remove(res["rescue_path"])

    def test_writer_exception_rescues(self, monkeypatch):
        import funhouse_agent._fileio as fio

        def raising(path, content):
            raise PermissionError("denied")

        monkeypatch.setattr(fio, "_local_write", raising)
        res = save_verified("save_verified_exc_test.html", "IMPORTANT")
        assert "error" in res
        assert res.get("rescue_path")
        try:
            assert open(res["rescue_path"], "rb").read() == b"IMPORTANT"
        finally:
            os.remove(res["rescue_path"])


class TestSaveFilePlaceholderRescue:
    def test_save_file_detects_placeholder(self, tmp_path, monkeypatch):
        import funhouse_agent.vision_tools as vt

        def placeholder_writer(path, content):
            abs_path = os.path.abspath(path)
            with open(abs_path, "wb") as f:
                f.write(b"PLACEHOLDER")
            return abs_path

        monkeypatch.setattr(vt, "_default_save_fn", placeholder_writer)
        out = str(tmp_path / "report_test_fileio.html")
        result = json.loads(vt.dispatch_extended_tool(
            "save_file", {"path": out, "content": "<!DOCTYPE html>" + "y" * 2000},
            engine=None, attachments=None, save_fn=None))
        assert result["file_exists"] is False
        assert "did not store the content" in result["error"]
        assert result.get("rescue_path")
        try:
            assert b"y" * 2000 in open(result["rescue_path"], "rb").read()
        finally:
            os.remove(result["rescue_path"])
