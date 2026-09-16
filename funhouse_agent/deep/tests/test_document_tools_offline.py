"""Deep-agent document review tools (planlens.tools) — offline, no engine.

Uses planlens' synthetic review document: a report page with a review stamp, a
/Rotate 90 drawing sheet with a reviewer's callout, a contractor's reply, a
cloud, an arrow and hidden CAD text, a ruled table, a blank page and a scan.
"""

import copy
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


def _older_planlens(monkeypatch, *, without=(), drop_search_params=()):
    """Make the installed planlens look like one that predates a feature.

    The app is wired against a planlens the cluster may not have yet: the
    cluster installs from PyPI. Editing the published specs is exactly what a
    older package looks like to the bridge, and it proves the older-planlens
    path without installing one.
    """
    from planlens.tools import specs as planlens_specs
    kept = []
    for spec in planlens_specs.TOOL_SPECS:
        if spec["name"] in without:
            continue
        if spec["name"] == "search_document" and drop_search_params:
            spec = copy.deepcopy(spec)
            for param in drop_search_params:
                spec["parameters"]["properties"].pop(param, None)
        kept.append(spec)
    monkeypatch.setattr(planlens_specs, "TOOL_SPECS", kept)


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


# -- tools and parameters a newer planlens adds --------------------------------

def test_find_quantities_is_offered_when_the_installed_planlens_has_it():
    if not document_tools.has_tool("find_quantities"):
        pytest.skip("installed planlens predates find_quantities")
    names = {t.name for t in make_vision_tools(engine=None)}
    assert "find_quantities" in names
    assert "find_quantities" in document_tools.document_tool_names()


def test_find_quantities_is_hidden_on_an_older_planlens(monkeypatch):
    _older_planlens(monkeypatch, without=("find_quantities",))
    names = {t.name for t in make_vision_tools(engine=None)}
    assert "find_quantities" not in names
    assert NAMES <= names                      # the seven are unaffected
    assert "find_quantities" not in document_tools.document_tool_names()


def test_stated_quantities_come_back_located_and_within_the_cap(gt):
    if not document_tools.has_tool("find_quantities"):
        pytest.skip("installed planlens predates find_quantities")
    tools = make_vision_tools(engine=None, attachments={"report.pdf": gt.pdf},
                              max_result_chars=3000,
                              reference_result_chars=3000)
    handle = _invoke(_tool(tools, "open_document"),
                     source="report.pdf")["handle"]
    raw = _tool(tools, "find_quantities").invoke(
        {"handle": handle, "pages": str(gt.narrative_page)})
    assert len(raw) <= 3000
    out = json.loads(raw)                      # valid JSON, not truncated
    assert "error" not in out
    assert out["n_mentions"] >= 1
    rows = out["quantities"]
    # The narrative states "approximately 40-foot centers" and "20 to 35 feet":
    # a mention carries the unit the page wrote, its kind and its page.
    assert any("ft" in row for row in rows)
    assert any("[length]" in row for row in rows)
    assert any(f"p{gt.narrative_page}" in row for row in rows)
    # Filtering reaches planlens with the spec's own parameter names.
    none = _invoke(_tool(tools, "find_quantities"), handle=handle,
                   pages=str(gt.narrative_page), kinds=["volume"])
    assert none["n_mentions"] == 0


def test_fuzzy_search_finds_a_word_with_a_letter_wrong(gt):
    if not document_tools.search_supports_fuzzy():
        pytest.skip("installed planlens predates fuzzy search")
    pytest.importorskip("rapidfuzz",
                        reason="fuzzy search needs the planlens 'text' extra")
    tools = make_vision_tools(engine=None, attachments={"s.pdf": gt.pdf})
    search = _tool(tools, "search_document")
    handle = _invoke(_tool(tools, "open_document"), source="s.pdf")["handle"]
    # The reviewer's callout says EMBEDMENT; this asks for it with one letter
    # substituted, the way a scan or SHX lettering comes back.
    exact = _invoke(search, handle=handle, pattern="EMBEDMANT")
    assert exact["n_hits"] == 0
    loose = _invoke(search, handle=handle, pattern="EMBEDMANT", fuzzy=True)
    assert loose["n_hits"] >= 1
    assert any("EMBEDMENT" in json.dumps(hit) for hit in loose["hits"])


def test_fuzzy_on_an_older_planlens_is_refused_without_a_call(gt, monkeypatch):
    tools = make_vision_tools(engine=None, attachments={"s.pdf": gt.pdf})
    search = _tool(tools, "search_document")
    handle = _invoke(_tool(tools, "open_document"), source="s.pdf")["handle"]
    _older_planlens(monkeypatch, drop_search_params=("fuzzy", "min_score"))
    calls = []
    monkeypatch.setattr(document_tools, "dispatch_document_tool",
                        lambda *a, **k: calls.append(a) or "{}")
    out = _invoke(search, handle=handle, pattern="EMBEDMANT", fuzzy=True)
    assert "fuzzy" in out["error"] and not calls
    # An exact search still goes through untouched.
    _invoke(search, handle=handle, pattern="EMBEDMENT")
    assert calls
