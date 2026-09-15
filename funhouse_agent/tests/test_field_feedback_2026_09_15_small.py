"""Small fixes from field feedback 2026-09-15 (Nairobi SOE re-run).

N7  html_to_pdf refused <img src="file:///tmp/x.png">: the Linux path lost its
    leading slash and was looked up inside the working folder.
N9  analyze_pdf_page / read_pdf_text / open_document refused the bare name of
    a file the SharePoint download had just put in the working folder.
N11 describe_method('SOE', ...) failed three times: module names are
    lower-case in the registry.
"""

import json
import os

import pytest

from funhouse_agent.adapters.calc_package import _resolve_image_path


# -- N7 ---------------------------------------------------------------------

def test_file_uri_keeps_an_absolute_posix_path():
    assert _resolve_image_path("file:///tmp/soe_moment_sensitivity.png") == \
        os.path.abspath("/tmp/soe_moment_sensitivity.png")


def test_file_uri_windows_drive_and_percent_encoding(tmp_path):
    png = tmp_path / "moment sensitivity.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    uri = png.as_uri()                       # file:///C:/.../moment%20sensitivity.png
    assert os.path.normcase(_resolve_image_path(uri)) == \
        os.path.normcase(str(png))


def test_html_to_pdf_embeds_a_file_uri_image(tmp_path):
    pytest.importorskip("fitz")
    from funhouse_agent.dispatch import call_agent
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=60, height=40)
    page.draw_rect(fitz.Rect(5, 5, 55, 35), color=(0, 0, 1), fill=(0, 0, 1))
    png = tmp_path / "fig.png"
    page.get_pixmap(dpi=72).save(str(png))
    out = call_agent("calc_package", "html_to_pdf", {
        "html": f'<html><body><p>x</p><img src="{png.as_uri()}"></body></html>',
        "output_path": str(tmp_path / "r.pdf")})
    assert out.get("status") == "success", out
    assert out.get("images_embedded") == 1


# -- N9 ---------------------------------------------------------------------

def test_bare_name_resolves_in_the_working_folder(tmp_path, monkeypatch):
    fitz = pytest.importorskip("fitz")
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    name = "315000-001-00 Excavation Support Dwg & Calcs.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "PRESSURE DIAGRAM FOR SAND")
    doc.save(str(tmp_path / name))
    from funhouse_agent.vision_tools import dispatch_extended_tool
    out = json.loads(dispatch_extended_tool("read_pdf_text", {"source": name},
                                            engine=None, attachments={}))
    assert "PRESSURE DIAGRAM FOR SAND" in json.dumps(out)
    from funhouse_agent.vision_tools import _resolve_attachment_or_path
    data, kind = _resolve_attachment_or_path(name, {})
    assert kind == "path" and data[:4] == b"%PDF"


def test_unknown_bare_name_still_explains(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    from funhouse_agent.vision_tools import _resolve_attachment_or_path
    with pytest.raises(FileNotFoundError, match="working folder"):
        _resolve_attachment_or_path("missing.pdf", {})


def test_document_tools_resolve_bare_name(tmp_path, monkeypatch):
    from funhouse_agent import document_tools
    if not document_tools.available():
        pytest.skip("planlens.tools not installed")
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4\n")
    assert os.path.normcase(document_tools._resolve("a.pdf")) == \
        os.path.normcase(str(tmp_path / "a.pdf"))


# -- N11 --------------------------------------------------------------------

@pytest.mark.parametrize("name", ["SOE", "Soe", " soe ", "soe"])
def test_module_names_ignore_case(name):
    from funhouse_agent.dispatch import call_agent, describe_method, list_methods
    assert "error" not in describe_method(name, "apparent_pressure")
    assert "error" not in list_methods(name)
    out = call_agent(name, "apparent_pressure", {
        "excavation_depth": 6.0,
        "layers": [{"thickness": 20, "unit_weight": 18, "friction_angle": 30}]})
    assert out.get("type") == "sand"


def test_unknown_module_still_errors():
    from funhouse_agent.dispatch import call_agent
    assert "Unknown module" in call_agent("SOEX", "x", {})["error"]
