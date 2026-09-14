"""Deep-agent document review tools (planlens.tools) — offline, no engine.

Uses planlens' synthetic review document: a report page with a review stamp, a
/Rotate 90 drawing sheet with a reviewer's callout, a contractor's reply, a
cloud, an arrow and hidden CAD text, a ruled table, a blank page and a scan.
"""

import json

import pytest

pytest.importorskip("planlens.tools")
fitz = pytest.importorskip("fitz")

from funhouse_agent import document_tools  # noqa: E402
from funhouse_agent.deep.tools import make_vision_tools  # noqa: E402
from planlens.testing import build_synthetic_review_document  # noqa: E402

NAMES = set(document_tools.DOCUMENT_TOOL_NAMES)


@pytest.fixture(scope="module")
def gt():
    return build_synthetic_review_document()


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


def _invoke(tool, **kwargs):
    return json.loads(tool.invoke(kwargs))


def test_document_tools_are_on_the_primary_surface():
    names = {t.name for t in make_vision_tools(engine=None)}
    assert NAMES <= names


def test_restricted_surfaces_stay_restricted():
    names = {t.name for t in make_vision_tools(engine=None,
                                               include={"save_file"})}
    assert names == {"save_file"}


def test_review_flow_over_an_attachment(gt):
    tools = make_vision_tools(engine=None,
                              attachments={"submittal.pdf": gt.pdf})
    opened = _invoke(_tool(tools, "open_document"), source="submittal.pdf")
    handle = opened["handle"]
    assert opened["n_pages"] == 5
    assert opened["pages_by_kind"]["drawing_sheet"] == str(gt.sheet_page)

    hits = _invoke(_tool(tools, "search_document"), handle=handle,
                   pattern="embedment")
    assert hits["n_hits"] == 2

    text = _invoke(_tool(tools, "read_document"), handle=handle,
                   pages=str(gt.sheet_page), with_locations=True)["text"]
    assert gt.sheet_text_upright in text
    assert "points at 1500,900" in text
    # The reviewer's words appear only under the markups heading.
    assert text.index("review markup") < text.index("CONFIRM THE PILE")

    marks = _invoke(_tool(tools, "document_markups"), handle=handle,
                    author="Reviewer")
    assert marks["n_markups"] == 2

    # pages accepts a list as well as a range string.
    rows = _invoke(_tool(tools, "document_page_map"), handle=handle,
                   pages=[0, 1])["rows"]
    assert [r["page"] for r in rows] == [0, 1]


def test_results_fit_the_cap_as_valid_json(gt):
    tools = make_vision_tools(engine=None, attachments={"s.pdf": gt.pdf},
                              max_result_chars=3000,
                              reference_result_chars=3000)
    handle = _invoke(_tool(tools, "open_document"), source="s.pdf")["handle"]
    raw = _tool(tools, "read_document").invoke(
        {"handle": handle, "with_locations": True})
    assert len(raw) <= 3000
    assert "next" in json.loads(raw)      # paged, not string-truncated


def test_truncation_disabled_still_returns_json(gt):
    tools = make_vision_tools(engine=None, attachments={"s.pdf": gt.pdf},
                              max_result_chars=0)
    handle = _invoke(_tool(tools, "open_document"), source="s.pdf")["handle"]
    assert "next" not in _invoke(_tool(tools, "read_document"), handle=handle)


def test_real_path_and_unknown_source(gt, tmp_path):
    path = tmp_path / "set.pdf"
    path.write_bytes(gt.pdf)
    tools = make_vision_tools(engine=None, attachments={"other.pdf": b"%PDF"})
    assert _invoke(_tool(tools, "open_document"),
                   source=str(path))["n_pages"] == 5
    missing = _invoke(_tool(tools, "open_document"), source="nope.pdf")
    assert "other.pdf" in missing["hint"]


def test_attachments_do_not_leak_between_conversations(gt):
    mine = make_vision_tools(engine=None, attachments={"mine.pdf": gt.pdf})
    theirs = make_vision_tools(engine=None, attachments={})
    assert "handle" in _invoke(_tool(mine, "open_document"), source="mine.pdf")
    assert "error" in _invoke(_tool(theirs, "open_document"),
                              source="mine.pdf")


def test_look_lines_name_the_apps_vision_tools(gt):
    tools = make_vision_tools(engine=None, attachments={"s.pdf": gt.pdf})
    opened = _invoke(_tool(tools, "open_document"), source="s.pdf")
    assert opened["source"] == "s.pdf"          # what analyze_pdf_page takes
    assert opened["pages_to_view"] == f"{gt.sheet_page},{gt.scanned_page}"
    assert "analyze_pdf_page(attachment_key=" in opened["pages_to_view_note"]
    text = _invoke(_tool(tools, "read_document"), handle=opened["handle"],
                   pages=str(gt.scanned_page))["text"]
    assert "! look: image-only page" in text
    assert "render_region(attachment_key=" in text
    names = {t.name for t in tools}
    assert "render_page" not in names            # the app's own vision tools


def test_an_image_upload_is_reviewable(gt):
    src = fitz.open("pdf", gt.pdf)
    png = src[gt.scanned_page].get_pixmap(dpi=40).tobytes("png")
    src.close()
    tools = make_vision_tools(engine=None, attachments={"scan.png": png})
    opened = _invoke(_tool(tools, "open_document"), source="scan.png")
    assert opened["kind"] == "image" and opened["pages_by_kind"] == {"scanned": "0"}


def test_structure_and_thumbnails_reach_the_agent(tmp_path):
    from planlens.testing import build_synthetic_submittal
    sub = build_synthetic_submittal()
    tools = make_vision_tools(engine=None, attachments={"sub.pdf": sub.pdf})
    opened = _invoke(_tool(tools, "open_document"), source="sub.pdf")
    assert any(s["title"] == sub.appendix_title for s in opened["segments"])
    structure = _invoke(_tool(tools, "document_structure"),
                        handle=opened["handle"])
    assert structure["n_segments"] == len(sub.expected_segments)
    thumbs = _invoke(_tool(tools, "render_page_thumbnails"),
                     handle=opened["handle"], pages="0-3")
    assert thumbs["sheets"][0]["pages"] == "0-3"
    assert "analyze_image(attachment_key=" in thumbs["note"]
    # The written PNG resolves through the same path the vision tools use.
    from funhouse_agent.vision_tools import _resolve_attachment_or_path
    data, kind = _resolve_attachment_or_path(thumbs["sheets"][0]["image_path"], {})
    assert kind == "path" and data[:8] == b"\x89PNG\r\n\x1a\n"


def test_without_the_tool_layer_the_tools_are_hidden(monkeypatch):
    monkeypatch.setattr(document_tools, "available", lambda: False)
    names = {t.name for t in make_vision_tools(engine=None)}
    assert not (NAMES & names)
    out = json.loads(document_tools.dispatch_document_tool(
        "open_document", {"source": "x.pdf"}))
    assert "planlens" in out["error"]
