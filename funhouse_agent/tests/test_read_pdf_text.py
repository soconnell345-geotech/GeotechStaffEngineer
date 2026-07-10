"""Offline tests for read_pdf_text + attachment/real-path resolution (v5.4.1).

No API key / vision model needed — read_pdf_text is pure PyMuPDF, and the
real-path fallback for analyze_* is exercised with engine=None (a path that
resolves then hits "vision not available" proves the resolution happened).
"""

import json

import pytest

from funhouse_agent.vision_tools import (
    dispatch_extended_tool, _resolve_attachment_or_path, _parse_pages,
    sanitize_upload_name, iter_upload_files, EXTENDED_TOOLS,
)

fitz = pytest.importorskip("fitz")  # PyMuPDF


def _make_pdf(n_text=2, scanned=False):
    """A programmatic PDF: n_text text-layer pages + optional 1 scanned page."""
    doc = fitz.open()
    for i in range(n_text):
        pg = doc.new_page()
        pg.insert_text((72, 72),
                       f"PAGE {i}\nBoring log B-{i + 1}: SPT N=15, w=22%.\n"
                       + ("lorem ipsum " * 40))
    if scanned:
        tmp = fitz.open()
        t = tmp.new_page()
        t.insert_text((72, 72), "scanned only")
        pix = t.get_pixmap()
        sp = doc.new_page()
        sp.insert_image(sp.rect, pixmap=pix)   # image only -> no text layer
        tmp.close()
    data = doc.tobytes()
    doc.close()
    return data


def _call(tool, args, engine=None, attachments=None):
    return json.loads(dispatch_extended_tool(tool, args, engine,
                                             attachments or {}))


class TestReadPdfText:
    def test_registered(self):
        assert "read_pdf_text" in EXTENDED_TOOLS

    def test_attachment_key_text_extraction(self):
        out = _call("read_pdf_text", {"source": "rpt", "pages": "0-1"},
                    attachments={"rpt": _make_pdf()})
        assert out["source_type"] == "attachment"
        assert out["pages_returned"] == [0, 1]
        assert out["pages"][0]["has_text_layer"] is True
        assert "SPT N=15" in out["pages"][0]["text"]

    def test_engine_none_ok(self):
        out = _call("read_pdf_text", {"source": "rpt"}, engine=None,
                    attachments={"rpt": _make_pdf()})
        assert "error" not in out and out["pages_returned"]

    def test_real_path(self, tmp_path):
        p = tmp_path / "r.pdf"
        p.write_bytes(_make_pdf())
        out = _call("read_pdf_text", {"source": str(p)})
        assert out["source_type"] == "path"
        assert "SPT N=15" in out["pages"][0]["text"]

    def test_attachment_takes_precedence_over_path(self, tmp_path):
        p = tmp_path / "dup.pdf"
        p.write_bytes(_make_pdf(n_text=3))          # 3-page file on disk
        att = {str(p): _make_pdf(n_text=1)}         # 1-page attachment same name
        out = _call("read_pdf_text", {"source": str(p)}, attachments=att)
        assert out["source_type"] == "attachment"
        assert out["n_pages_total"] == 1            # the attachment won

    def test_scanned_page_flagged(self):
        out = _call("read_pdf_text", {"source": "rpt", "pages": "0-2"},
                    attachments={"rpt": _make_pdf(n_text=2, scanned=True)})
        p2 = out["pages"][2]
        assert p2["has_text_layer"] is False
        assert p2["text"] == ""
        assert "no text layer" in p2["note"]
        assert "analyze_pdf_page" in p2["note"]
        assert out["scanned_pages"] == [2]
        assert "analyze_pdf_page" in out["scanned_note"]

    def test_truncation_flagged(self):
        out = _call("read_pdf_text",
                    {"source": "rpt", "pages": [0, 1], "max_chars": 150},
                    attachments={"rpt": _make_pdf(n_text=2)})
        assert out["truncated"] is True
        assert "budget" in out["truncated_note"]

    def test_pages_int_list_range(self):
        att = {"rpt": _make_pdf(n_text=4)}
        assert _call("read_pdf_text", {"source": "rpt", "pages": 2},
                     attachments=att)["pages_returned"] == [2]
        assert _call("read_pdf_text", {"source": "rpt", "pages": [0, 3]},
                     attachments=att)["pages_returned"] == [0, 3]
        assert _call("read_pdf_text", {"source": "rpt", "pages": "1-3"},
                     attachments=att)["pages_returned"] == [1, 2, 3]

    def test_out_of_range_clamped_with_note(self):
        out = _call("read_pdf_text", {"source": "rpt", "pages": "0-9"},
                    attachments={"rpt": _make_pdf(n_text=2)})
        assert out["pages_returned"] == [0, 1]
        assert "out of range" in out["page_request_note"]

    def test_default_pages(self):
        out = _call("read_pdf_text", {"source": "rpt"},
                    attachments={"rpt": _make_pdf(n_text=3)})
        assert out["pages_returned"] == [0, 1, 2]

    def test_not_found_error_mentions_keys_and_paths(self):
        out = _call("read_pdf_text", {"source": "nope"},
                    attachments={"rpt": b"x"})
        assert "not found" in out["error"]
        assert "rpt" in out["error"]
        assert "filesystem path" in out["error"]

    def test_non_pdf_bytes_error(self):
        out = _call("read_pdf_text", {"source": "bad"},
                    attachments={"bad": b"not a pdf at all"})
        assert "error" in out and "PDF" in out["error"]


class TestRealPathFallback:
    def test_analyze_pdf_page_resolves_real_path(self, tmp_path):
        p = tmp_path / "r.pdf"
        p.write_bytes(_make_pdf())
        # engine=None: resolution succeeds, then vision is unavailable — the
        # "vision not available" error proves the path was resolved+rendered.
        out = _call("analyze_pdf_page", {"attachment_key": str(p), "page": 0},
                    engine=None)
        assert "Vision not available" in out["error"]

    def test_analyze_image_missing_key_mentions_paths(self):
        out = _call("analyze_image", {"attachment_key": "missing"},
                    attachments={"a": b"x"})
        assert "not found" in out["error"]
        assert "filesystem path" in out["error"]
        assert "'a'" in out["error"]

    def test_resolve_helper(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"PDFBYTES")
        data, src = _resolve_attachment_or_path(str(p), {})
        assert src == "path" and data == b"PDFBYTES"
        data2, src2 = _resolve_attachment_or_path("k", {"k": b"ATT"})
        assert src2 == "attachment" and data2 == b"ATT"
        with pytest.raises(FileNotFoundError):
            _resolve_attachment_or_path("nope", {"k": b"ATT"})


class TestUploadHelpers:
    def test_sanitize_basename_and_chars(self):
        assert sanitize_upload_name("Mali Report v2.pdf") == "Mali_Report_v2.pdf"
        assert sanitize_upload_name("/tmp/sub/dir/a b.pdf") == "a_b.pdf"
        assert sanitize_upload_name("C:\\x\\y z.PDF") == "y_z.PDF"
        assert sanitize_upload_name("") == "file"
        assert sanitize_upload_name(None) == "file"

    def test_iter_upload_files_8x_tuple(self):
        got = list(iter_upload_files(
            ({"name": "a.pdf", "content": b"AA"},
             {"name": "b.png", "content": bytearray(b"BB")})))
        assert got == [("a.pdf", b"AA"), ("b.png", b"BB")]

    def test_iter_upload_files_7x_dict(self):
        got = list(iter_upload_files({"a.pdf": {"content": b"AA"}}))
        assert got == [("a.pdf", b"AA")]

    def test_iter_upload_files_empty(self):
        assert list(iter_upload_files(None)) == []
        assert list(iter_upload_files({})) == []


class TestParsePages:
    def test_forms(self):
        assert _parse_pages(None, 5, 3) == ([0, 1, 2], None)
        assert _parse_pages(2, 5, 3)[0] == [2]
        assert _parse_pages("1-3", 5, 3)[0] == [1, 2, 3]
        assert _parse_pages([0, 4], 5, 3)[0] == [0, 4]
        assert _parse_pages("2", 5, 3)[0] == [2]

    def test_out_of_range(self):
        pages, note = _parse_pages("2-9", 5, 3)
        assert pages == [2, 3, 4] and "out of range" in note

    def test_unparseable_falls_back(self):
        pages, note = _parse_pages("garbage", 5, 3)
        assert pages == [0, 1, 2] and "could not parse" in note
