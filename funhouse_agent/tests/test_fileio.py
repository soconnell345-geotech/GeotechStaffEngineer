"""Tests for funhouse_agent._fileio — verified file writes + placeholder rescue.

Regression for the live Databricks failure (2026-06-12): a 241 kB calc-package
HTML written to /Workspace via plain file I/O "succeeded" but the workspace
stored the literal 11-byte string PLACEHOLDER. Tool responses must detect a
target filesystem that did not store the content and rescue to the temp dir.
"""

import json
import os
import tempfile

import pytest

from funhouse_agent._fileio import (
    default_output_dir, rescue_write, workspace_write_hint,
    written_file_problem,
)


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
        assert default_output_dir() == ""

    def test_databricks_uses_tempdir(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
        assert default_output_dir() == tempfile.gettempdir()


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
