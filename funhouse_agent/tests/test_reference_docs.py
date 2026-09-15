"""Reference PDFs fetched on first use when not in the local docs folder.

Owner decision 2026-09-15: the PDFs live in SharePoint
GSE_app/primary_references and are downloaded only when a chart or worked
example page from them is first needed (never at launch).
"""

import json
import os
import shutil

import pytest

from funhouse_agent import reference_docs


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv(reference_docs.CACHE_ENV, str(tmp_path / "cache"))
    reference_docs.register_fetcher(None)
    yield
    reference_docs.register_fetcher(None)


def _pdf(path, text="Figure 4-12"):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text)
    doc.save(str(path))
    return path


def test_fetch_downloads_once_then_uses_the_cache():
    calls = []

    def fetcher(name, dest):
        calls.append(name)
        with open(dest, "wb") as fh:
            fh.write(b"%PDF-1.4")
        return dest

    reference_docs.register_fetcher(fetcher)
    first = reference_docs.fetch("docs/GEC 12 Vol 3.pdf")
    again = reference_docs.fetch("GEC 12 Vol 3.pdf")
    assert first == again and os.path.basename(first) == "GEC 12 Vol 3.pdf"
    assert calls == ["GEC 12 Vol 3.pdf"]


def test_failed_fetch_and_the_message():
    def fetcher(name, dest):
        raise FileNotFoundError(name)

    fetcher.description = "SharePoint primary_references/"
    reference_docs.register_fetcher(fetcher)
    assert reference_docs.fetch("x.pdf") is None
    msg = reference_docs.missing_message("source PDF not found.", "docs/x.pdf")
    assert "SharePoint primary_references/" in msg and "'x.pdf'" in msg


def test_without_a_fetcher_nothing_changes():
    assert reference_docs.fetch("x.pdf") is None
    assert reference_docs.missing_message("orig", "x.pdf") == "orig"


class _Eng:
    def analyze_image(self, image_bytes, prompt):
        return "Kp about 5.5"


def test_read_reference_figure_uses_the_fetcher(tmp_path, monkeypatch):
    from geotech_references import _figures_db
    src = _pdf(tmp_path / "src.pdf")

    def not_local(reference, figure_number):
        raise FileNotFoundError("source PDF for dm7_2 4-12 not found")

    monkeypatch.setattr(_figures_db, "resolve_pdf", not_local)
    monkeypatch.setattr(_figures_db, "figure_get", lambda r, f: {
        "figure_number": "4-12", "caption": "Log spiral Ka and Kp",
        "pdf_path": "docs/ufc_3_220_20_2025.pdf", "pdf_page_index": 0,
        "page_estimated": False})
    asked = []

    def fetcher(name, dest):
        asked.append(name)
        shutil.copy(src, dest)
        return dest

    reference_docs.register_fetcher(fetcher)
    from funhouse_agent.vision_tools import dispatch_extended_tool
    out = json.loads(dispatch_extended_tool(
        "read_reference_figure",
        {"reference": "dm7_2", "figure_number": "4-12", "prompt": "Kp?"},
        _Eng(), {}))
    assert "error" not in out, out
    assert asked == ["ufc_3_220_20_2025.pdf"]


def test_read_reference_figure_error_names_sharepoint(monkeypatch):
    from geotech_references import _figures_db

    def not_local(reference, figure_number):
        raise FileNotFoundError("source PDF for dm7_2 4-12 not found")

    monkeypatch.setattr(_figures_db, "resolve_pdf", not_local)

    def fetcher(name, dest):
        return None

    fetcher.description = "SharePoint primary_references/"
    reference_docs.register_fetcher(fetcher)
    from funhouse_agent.vision_tools import dispatch_extended_tool
    out = json.loads(dispatch_extended_tool(
        "read_reference_figure", {"reference": "dm7_2", "figure_number": "4-12"},
        _Eng(), {}))
    assert "SharePoint primary_references/" in out["error"]


def test_worked_example_page_uses_the_fetcher(tmp_path, monkeypatch):
    from funhouse_agent import worked_examples as we
    src = _pdf(tmp_path / "src.pdf", "Table G-1")

    def not_local(entry):
        raise FileNotFoundError("source PDF not found")

    monkeypatch.setattr(we, "resolve_source_pdf", not_local)
    reference_docs.register_fetcher(
        lambda name, dest: shutil.copy(src, dest) and dest)
    from funhouse_agent.vision_tools import dispatch_extended_tool
    out = json.loads(dispatch_extended_tool(
        "view_worked_example_source",
        {"example_id": "WE-PAVE-3", "pdf_page": 1, "prompt": "What is shown?"},
        _Eng(), {}))
    assert "error" not in out, out
