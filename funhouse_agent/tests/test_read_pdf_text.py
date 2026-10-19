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


class TestViewerPages:
    """One page convention (BACKLOG GM-154 / VIS-27): ``pages`` counts from
    0 as every tool's does, ``pdf_pages`` from 1 as a viewer (and
    html_to_pdf's page_contents) do, and every page in the result carries
    both. Live wave w13: a helper read ``pages`` "3-5" as 1-based and cited
    its tables one page low."""

    def test_pdf_pages_reads_the_pages_a_viewer_shows(self):
        att = {"rpt": _make_pdf(n_text=6)}
        out = _call("read_pdf_text", {"source": "rpt", "pdf_pages": "4-5"},
                    attachments=att)
        assert out["pages_returned"] == [3, 4]
        assert out["pdf_pages_returned"] == [4, 5]
        assert [p["pdf_page"] for p in out["pages"]] == [4, 5]
        assert "PAGE 3" in out["pages"][0]["text"]       # the 4th page
        assert "pdf_page" in out["page_numbers"]
        assert "html_to_pdf" in out["page_numbers"]

    def test_every_page_carries_its_viewer_number(self):
        out = _call("read_pdf_text", {"source": "rpt", "pages": "0-2"},
                    attachments={"rpt": _make_pdf(n_text=2, scanned=True)})
        assert [(p["page"], p["pdf_page"]) for p in out["pages"]] == \
            [(0, 1), (1, 2), (2, 3)]
        assert "pdf_page 3" in out["pages"][2]["note"]
        assert "pdf_page [3]" in out["scanned_note"]

    def test_pdf_pages_forms(self):
        att = {"rpt": _make_pdf(n_text=6)}
        for spec, want in ((2, [1]), ([1, 6], [0, 5]), ("1-2,5", [0, 1, 4])):
            got = _call("read_pdf_text", {"source": "rpt", "pdf_pages": spec},
                        attachments=att)["pages_returned"]
            assert got == want, spec

    def test_pdf_page_zero_is_refused(self):
        out = _call("read_pdf_text", {"source": "rpt", "pdf_pages": "0-2"},
                    attachments={"rpt": _make_pdf()})
        assert "error" in out and "count from 1" in out["error"]

    def test_both_given_must_agree(self):
        att = {"rpt": _make_pdf(n_text=4)}
        out = _call("read_pdf_text",
                    {"source": "rpt", "pages": "1-2", "pdf_pages": "2-3"},
                    attachments=att)
        assert out["pages_returned"] == [1, 2]
        out = _call("read_pdf_text",
                    {"source": "rpt", "pages": "3", "pdf_pages": "3"},
                    attachments=att)
        assert "not both" in out["error"]

    def test_out_of_range_said_in_the_numbers_given(self):
        out = _call("read_pdf_text", {"source": "rpt", "pdf_pages": "2-4"},
                    attachments={"rpt": _make_pdf(n_text=2)})
        assert out["pdf_pages_returned"] == [2]
        assert "pdf_page(s) [3]" in out["page_request_note"]
        assert "pdf_page 1-2" in out["page_request_note"]

    def test_truncation_names_the_next_viewer_pages(self):
        out = _call("read_pdf_text",
                    {"source": "rpt", "pages": [0, 1], "max_chars": 150},
                    attachments={"rpt": _make_pdf(n_text=2)})
        # The first page fits; the second (pdf_page 2) is cut short.
        assert out["pages"][-1]["text"].endswith("[page text truncated]")
        assert "pdf_pages='2-9'" in out["truncated_note"]

    def test_the_agent_tool_takes_pdf_pages(self):
        from funhouse_agent.deep.tools import make_vision_tools
        (tool,) = make_vision_tools(attachments={"rpt": _make_pdf(n_text=5)},
                                    include={"read_pdf_text"})
        out = json.loads(tool.invoke({"source": "rpt", "pdf_pages": 5}))
        assert out["pdf_pages_returned"] == [5]
        assert "pdf_pages" in tool.description
        assert "pdf_pages" in tool.args


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

    def test_comma_lists(self):
        assert _parse_pages("0-1,3", 5, 3)[0] == [0, 1, 3]
        assert _parse_pages(" 4 , 2 ", 5, 3)[0] == [2, 4]

    def test_a_huge_range_is_not_built_page_by_page(self):
        pages, note = _parse_pages("3-999999999", 5, 3)
        assert pages == [3, 4] and "out of range" in note
