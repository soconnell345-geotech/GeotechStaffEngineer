"""The app's binding to planlens.review (BACKLOG PL-29 (b), split step 9 (b)).

The document tools that only wrapped planlens live in ``planlens.review``;
``aec_agent_core.document_tools`` hands it the app's host. What is checked
here is the binding, offline with fake engines and a made-up sheet:

* the process's host is the app's (its confinement, scratch folder, hints,
  the find_like budget setting);
* each agent build gets its own host: two builds made one after the other
  sign and look with their own signer and engine, never the last one's;
* an engine that cannot see images leaves the marks unchecked (no
  ``check`` block, as before), and with no engine ``find_like`` returns
  unverified matches;
* a marked copy, a markup check and a ``find_like`` search run end to end
  through the agent's own tools;
* the Word/Excel reader keeps its plain-Markdown fallback.
"""

import json

import pytest

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("PIL.Image")

from planlens import review  # noqa: E402

from aec_agent_core import document_tools  # noqa: E402
from aec_agent_core.deep.vision_kit import make_vision_tools  # noqa: E402

pytestmark = pytest.mark.skipif(
    not document_tools.has_tool("annotate_document"),
    reason="installed planlens has no annotate_document")

TAG = (300.0, 200.0, 330.0, 220.0)


def _sheet() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=396)
    page.draw_rect(pymupdf.Rect(*TAG), color=(0, 0, 0), width=1.0)
    page.insert_text((TAG[0] + 3, TAG[3] - 6), "ZQ7", fontsize=8)
    data = doc.tobytes()
    doc.close()
    return data


class Seer:
    """Says the crop encloses the tag; records each prompt."""

    def __init__(self, name):
        self.name = name
        self.prompts = []

    def analyze_image(self, image_bytes, prompt):
        self.prompts.append(prompt)
        return json.dumps({"encloses": True, "inside": "ZQ7", "sure": True})


class Blind:
    """An engine with no analyze_image."""


def _tools(engine, folder, monkeypatch, author=None):
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(folder))
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    return {t.name: t for t in make_vision_tools(
        engine=engine, attachments={"sheet.pdf": _sheet()},
        markup_author=author)}


def _mark(tools, name="sheet_marked.pdf"):
    return json.loads(tools["annotate_document"].invoke({
        "handle": "sheet.pdf", "output_path": name,
        "markups": [{"kind": "box", "page": 0, "bbox": list(TAG),
                     "comment": "check this tag", "target": "ZQ7"}]}))


def _authors(path):
    with pymupdf.open(path) as doc:
        return {a.info.get("title") for page in doc
                for a in (page.annots() or [])}


def test_the_process_host_is_the_apps():
    host = review.current()
    assert host is document_tools.BASE_HOST
    assert host.vision_hint == document_tools.VISION_HINT
    assert host.image_view_hint == document_tools.IMAGE_VIEW_HINT
    assert host.scratch_dir == ".scratch"
    assert host.look is None                 # a look is per build
    assert "Document Review" in host.own_mark_words


def test_two_builds_keep_their_own_signer_and_look(tmp_path, monkeypatch):
    a, b = Seer("a"), Seer("b")
    tools_a = _tools(a, tmp_path / "one", monkeypatch,
                     author="Reviewer One via App (AI draft)")
    tools_b = _tools(b, tmp_path / "two", monkeypatch,
                     author="Reviewer Two via App (AI draft)")
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path / "one"))
    out_a = _mark(tools_a)
    n_a, n_b = len(a.prompts), len(b.prompts)
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path / "two"))
    out_b = _mark(tools_b)
    assert "error" not in out_a and "error" not in out_b
    assert n_a >= 1 and n_b == 0             # build A's look checked A's copy
    assert len(b.prompts) >= 1 and len(a.prompts) == n_a
    assert _authors(tmp_path / "one" / "sheet_marked.pdf") == {
        "Reviewer One via App (AI draft)"}
    assert _authors(tmp_path / "two" / "sheet_marked.pdf") == {
        "Reviewer Two via App (AI draft)"}
    assert out_a["check"] and out_b["check"]


def test_no_author_given_signs_with_the_setting_read_now(tmp_path,
                                                         monkeypatch):
    tools = _tools(Seer("c"), tmp_path, monkeypatch)
    monkeypatch.setenv(document_tools.MARKUP_AUTHOR_ENV,
                       "Made-up Desk (AI draft)")
    out = _mark(tools)
    assert "error" not in out
    assert _authors(tmp_path / "sheet_marked.pdf") == {
        "Made-up Desk (AI draft)"}


def test_an_engine_that_cannot_see_leaves_the_marks_unchecked(tmp_path,
                                                              monkeypatch):
    out = _mark(_tools(Blind(), tmp_path, monkeypatch))
    assert "error" not in out and "check" not in out
    assert (tmp_path / "sheet_marked.pdf").is_file()


def test_the_build_host_reads_through_its_own_engine():
    seer = Seer("d")
    host = document_tools.host_for(seer, "Someone (AI draft)")
    assert host.look(b"png", "what is here?", "check")
    assert seer.prompts == ["what is here?"]
    assert host.signed_by() == "Someone (AI draft)"
    assert document_tools.host_for(None).look is None
    assert document_tools.checking_host(
        document_tools.host_for(Blind()), Blind()).look is None


def test_find_like_runs_through_the_agents_tool(tmp_path, monkeypatch):
    pytest.importorskip("planlens.document.findlike")
    tf = pytest.importorskip("planlens.testing.tag_fixtures")
    gt = tf.build_synthetic_tag_set(n_pages=1)
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    tools = {t.name: t for t in make_vision_tools(
        engine=None, attachments={"set.pdf": gt.pdf})}
    if "find_like" not in tools:
        pytest.skip("find_like is not offered here (no OpenCV)")
    out = json.loads(tools["find_like"].invoke({
        "attachment_key": "set.pdf", "page": 0,
        "bbox": list(gt.example_bbox), "text": "GCE"}))
    assert "error" not in out
    assert "UNVERIFIED" in out.get("note", "")   # no engine: no reads


def test_the_office_reader_keeps_its_fallback(monkeypatch):
    from aec_agent_core import office_text

    def old_reader(path):                    # no picture arguments
        raise TypeError("unexpected keyword argument 'image_dir'")

    monkeypatch.setattr(office_text, "read_office",
                        lambda path, **kw: old_reader(path))
    monkeypatch.setattr(office_text, "office_to_markdown",
                        lambda path: "| a | b |")
    assert document_tools._read_office("x.xlsx", "x.xlsx", image_dir="d",
                                       image_ref=".scratch") == (
        "| a | b |", {})
