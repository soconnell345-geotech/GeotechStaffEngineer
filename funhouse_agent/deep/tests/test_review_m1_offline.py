"""Review milestone M1 (S1.5 findings, S1.2 overview) - offline, fake models.

Pins both sides of each new switch (``funhouse_agent.review_flags``):
switched off, the page is what it was; switched on, the finding format, its
ledger, its quote check and its renderers work, and an image file is shown
to the main model itself when the inline switch is on.
"""

import json
import os
import sys

import pytest

pytest.importorskip("planlens.tools")
fitz = pytest.importorskip("fitz")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel)
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from funhouse_agent import (  # noqa: E402
    document_tools, inline_store, review_flags, review_findings as RF,
    vision_tools)
from funhouse_agent.deep.agent import build_deep_agent  # noqa: E402
from funhouse_agent.deep.tools import make_vision_tools  # noqa: E402
from planlens.testing import build_synthetic_review_document  # noqa: E402

GEOTECH_WORDS = ("drawing_ir", "digitize_drawing", "query_drawing",
                 "snip_region", "call_agent", "figure_db")
FINDINGS_TOOLS = {"record_finding", "update_finding", "list_findings",
                  "findings_report"}


@pytest.fixture(autouse=True)
def _no_switches(monkeypatch, tmp_path):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    monkeypatch.delenv("GEOTECH_REVIEW_MAX_MODEL_CALLS", raising=False)
    monkeypatch.delenv(document_tools.MARKUP_AUTHOR_ENV, raising=False)
    # the working folder: where findings.json and every output lands
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(work))


@pytest.fixture(scope="module")
def gt():
    return build_synthetic_review_document()


def _review_kwargs():
    from webapp.profiles import DOCUMENT_REVIEW
    return DOCUMENT_REVIEW.build_kwargs()


class Scripted(FakeMessagesListChatModel):
    """Plays ``script`` (AIMessages or callables of the request messages) and
    records what it was sent."""

    script: list = []
    seen: list = []

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        step = self.script.pop(0) if self.script else AIMessage(content="done")
        msg = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _model(script):
    return Scripted(responses=[AIMessage(content="x")], script=list(script),
                    seen=[])


def _call(name, args, i=1):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args,
                                              "id": f"call_{name}_{i}"}])


def _tools_of(agent):
    return agent.nodes["tools"].bound.tools_by_name


class FakeEngine:
    accepts_jpeg = False

    def __init__(self, reply="seen"):
        self.reply = reply
        self.prompts = []

    def vision_profile(self):
        return None

    def analyze_image(self, image, prompt):
        self.prompts.append(prompt)
        return self.reply


def _finding(**over):
    base = dict(statement="The pile embedment is not shown.", severity="major",
                confidence="high", evidence="read",
                citations=[{"document": "a.pdf", "page": 0,
                            "quote": "pile embedment"}])
    base.update(over)
    return RF.Finding(**base)


# ---------------------------------------------------------------------------
# The format
# ---------------------------------------------------------------------------

def test_a_finding_is_validated_with_a_clear_message():
    with pytest.raises(ValueError, match="severity must be one of"):
        _finding(severity="high")
    with pytest.raises(ValueError, match="confidence must be one of"):
        _finding(confidence="sure")
    with pytest.raises(ValueError, match="evidence must be one of"):
        _finding(evidence="guessed")
    with pytest.raises(ValueError, match="status must be one of"):
        _finding(status="open")
    with pytest.raises(ValueError, match="needs a statement"):
        _finding(statement="   ")
    with pytest.raises(ValueError, match="at least one citation"):
        _finding(citations=[])
    with pytest.raises(ValueError, match="0-based integer"):
        _finding(citations=[{"document": "a.pdf", "page": -1}])
    with pytest.raises(ValueError, match="needs the document"):
        _finding(citations=[{"document": "", "page": 0}])
    with pytest.raises(ValueError, match="bbox"):
        _finding(citations=[{"document": "a.pdf", "page": 0,
                             "bbox": [1, 2, 3]}])
    # enums are read case-insensitively and stored lower-case
    f = _finding(severity="Major", confidence="HIGH", evidence="Seen")
    assert (f.severity, f.confidence, f.evidence, f.status) == \
        ("major", "high", "seen", "draft")


def test_a_finding_round_trips_through_json():
    f = _finding(id="F3", disciplines=["geotechnical"], related=["F1"],
                 source="review agent",
                 citations=[{"document": "a.pdf", "page": 4, "sheet": "S-1",
                             "printed_page": "12", "bbox": [1, 2, 30, 40],
                             "quote": "EL 1704 m"}])
    c = f.citations[0]
    assert c.pdf_page == 5                       # always page + 1
    again = RF.Finding.from_dict(json.loads(json.dumps(f.to_dict())))
    assert again == f
    assert again.to_dict() == f.to_dict()
    # a viewer page given alone is read as page - 1
    assert RF.Citation.from_dict({"document": "a.pdf", "pdf_page": 3}).page == 2


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------

PAGES = {
    "a.pdf": {0: "The pile   embedment is 6 m\nbelow the pile cap. See NOTE 4.",
              1: "",                                  # no text layer
              2: "B-1"},
}


def _page_text(document, page):
    doc = PAGES[document]                             # KeyError: unknown doc
    if page not in doc:
        raise IndexError(f"no page {page}")
    return doc[page]


def test_quotes_pass_fail_and_a_wholly_unfound_finding_is_downgraded():
    ok = _finding(citations=[{"document": "a.pdf", "page": 0,
                              "quote": "Pile embedment is 6 M below"}])
    bad = _finding(citations=[{"document": "a.pdf", "page": 0,
                               "quote": "concrete strength 5000 psi"}])
    mixed = _finding(citations=[
        {"document": "a.pdf", "page": 0, "quote": "see note 4"},
        {"document": "a.pdf", "page": 0, "quote": "grout pressure 2 MPa"}])
    scan = _finding(citations=[{"document": "a.pdf", "page": 1,
                                "quote": "anything"}])
    unknown = _finding(citations=[{"document": "zzz.pdf", "page": 0,
                                   "quote": "pile embedment"}])
    off_end = _finding(citations=[{"document": "a.pdf", "page": 9,
                                   "quote": "pile embedment"}])
    short_page = _finding(citations=[{"document": "a.pdf", "page": 2,
                                      "quote": "boring B-1 was drilled to "
                                               "30 ft"}])
    no_quote = _finding(citations=[{"document": "a.pdf", "page": 0,
                                    "bbox": [0, 0, 10, 10]}])
    out = RF.verify_quotes([ok, bad, mixed, scan, unknown, off_end,
                            short_page, no_quote], _page_text)
    assert len(out) == 8                              # never deleted
    assert ok.quote_verified is True and ok.confidence == "high"
    assert "found" in ok.quote_note and "PDF p. 1" in ok.quote_note
    assert bad.quote_verified is False and bad.confidence == "low"
    assert "NOT found" in bad.quote_note and "lowered" in bad.quote_note
    # one quote found: flagged, not downgraded
    assert mixed.quote_verified is False and mixed.confidence == "high"
    # a page with no text cannot be checked: neither pass nor fail
    assert scan.quote_verified is None and scan.confidence == "high"
    assert "no text layer" in scan.quote_note
    assert unknown.quote_verified is False and unknown.confidence == "low"
    assert off_end.quote_verified is False and off_end.confidence == "low"
    # a short page cannot "contain" a long quote
    assert short_page.quote_verified is False
    assert no_quote.quote_verified is None and no_quote.quote_note is None


def test_quote_score_normalises_and_works_without_rapidfuzz(monkeypatch):
    text = "Use “Type II” cement — see Section 03 30 00."
    assert RF.quote_score('use "type ii" cement - see', text) == 100.0
    fuzzy_with = RF.quote_score("use type II cemnt", text)
    monkeypatch.setitem(sys.modules, "rapidfuzz", None)   # import -> error
    fuzzy_without = RF.quote_score("use type II cemnt", text)
    assert fuzzy_without >= 85 and fuzzy_with >= 85
    assert RF.quote_score("steel pipe piles", text) < 60


def test_planlens_page_text_reads_the_cited_page(gt, tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(gt.pdf)
    with RF.planlens_page_text({"r.pdf": gt.pdf,
                                str(path): str(path)}) as page_text:
        assert gt.heading in page_text("r.pdf", gt.narrative_page)
        # a path source is found by its file name alone
        assert gt.heading in page_text("report.pdf", gt.narrative_page)
        with pytest.raises(KeyError):
            page_text("other.pdf", 0)
        with pytest.raises(IndexError):
            page_text("r.pdf", 99)
        f = _finding(citations=[{"document": "r.pdf",
                                 "page": gt.narrative_page,
                                 "quote": " ".join(gt.phrase_across_lines)}])
        RF.verify_quotes([f], page_text)
    assert f.quote_verified is True


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def test_ledger_adds_updates_lists_and_persists(monkeypatch, tmp_path):
    ledger = RF.FindingsLedger()
    assert ledger.load() == []                         # no file: empty
    assert ledger.path == str(tmp_path / "work" / RF.LEDGER_NAME)
    a = ledger.add(_finding(disciplines=["Structural"]))
    b = ledger.add(_finding(statement="Legend missing.", severity="minor"))
    assert (a.id, b.id) == ("F1", "F2")
    assert (tmp_path / "work" / "findings.json").is_file()
    c = ledger.add(_finding(id="F1", statement="dup id"))   # re-numbered
    assert c.id == "F3"
    assert ledger.update("f1", status="confirmed").status == "confirmed"
    with pytest.raises(ValueError):
        ledger.update("F1", severity="huge")
    with pytest.raises(KeyError):
        ledger.update("F9", status="rejected")
    assert [f.id for f in ledger.list(status="confirmed")] == ["F1"]
    assert [f.id for f in ledger.list(severity=["minor"])] == ["F2"]
    assert [f.id for f in ledger.list(discipline="structural")] == ["F1"]
    # a new object reads what the first one wrote
    again = RF.FindingsLedger()
    assert [f.id for f in again.load()] == ["F1", "F2", "F3"]
    assert again.get("F1").status == "confirmed"
    # the ledger follows the working folder the host points it at
    other = tmp_path / "other"
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(other))
    assert ledger.load() == []
    assert ledger.add(_finding()).id == "F1"
    assert (other / "findings.json").is_file()


def test_an_unreadable_ledger_is_set_aside_not_written_over(tmp_path):
    path = tmp_path / "work" / "findings.json"
    path.write_text("{not json", encoding="utf-8")
    ledger = RF.FindingsLedger()
    assert ledger.load() == []
    assert (tmp_path / "work" / "findings.unreadable.json").read_text(
        encoding="utf-8") == "{not json"
    assert ledger.add(_finding()).id == "F1"


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _rendered_findings():
    return [
        _finding(id="F1", citations=[{"document": "plans.pdf", "page": 3,
                                      "sheet": "S-2"}]),
        _finding(id="F2", severity="critical", statement="Bearing | 3 ksf "
                 "is unsupported.", citations=[
                     {"document": "report.pdf", "page": 11,
                      "printed_page": "12"}]),
        _finding(id="F3", severity="info", confidence="low",
                 quote_verified=False,
                 citations=[{"document": "report.pdf", "page": 0}]),
    ]


def test_markdown_comment_log():
    md = RF.to_markdown(_rendered_findings())
    lines = md.splitlines()
    assert lines[0] == "| No. | Severity | Finding | Where | Evidence |"
    assert "Sheet S-2 (plans.pdf)" in md          # two documents: named
    assert "p. 12 (report.pdf)" in md
    assert "PDF p. 1 (report.pdf)" in md          # never the 0-based index
    assert "Bearing \\| 3 ksf" in md              # a pipe cannot split a row
    assert "low confidence; quote not found on the cited page" in md
    assert all(line.count("|") - line.count("\\|") == 6
               for line in lines[:5])
    one_doc = RF.to_markdown(_rendered_findings()[:1])
    assert "Sheet S-2 |" in one_doc and "(plans.pdf)" not in one_doc
    assert RF.to_markdown([]) == "_No findings recorded._\n"
    order = [f.id for f in RF.sort_findings(_rendered_findings())]
    assert order == ["F2", "F1", "F3"]            # most severe first


def test_markdown_renders_as_a_word_table(tmp_path):
    docx = pytest.importorskip("docx")
    out = json.loads(vision_tools._dispatch_write_docx(
        {"path": str(tmp_path / "work" / "log.docx"),
         "markdown": RF.to_markdown(_rendered_findings()),
         "title": "Review comments"}, vision_tools._default_save_fn))
    assert "error" not in out, out
    doc = docx.Document(str(tmp_path / "work" / "log.docx"))
    table = doc.tables[0]
    assert [c.text for c in table.rows[0].cells] == [
        "No.", "Severity", "Finding", "Where", "Evidence"]
    assert len(table.rows) == 4
    assert "Bearing | 3 ksf is unsupported." in table.rows[2].cells[2].text


def test_markups_go_on_through_planlens_annotate_and_read_back(gt, tmp_path):
    from funhouse_agent.review_eval.checks import check_pdf_markups
    from planlens.document import Document
    from planlens.document.markup_writer import MarkupSpec
    x, y = gt.sheet_text_upright_origin
    findings = [
        _finding(id="F1", citations=[{"document": "r.pdf",
                                      "page": gt.narrative_page,
                                      "quote": gt.heading}]),
        _finding(id="F2", evidence="seen", citations=[
            {"document": "r.pdf", "page": gt.sheet_page,
             "bbox": [x - 10, y - 30, x + 200, y + 10]},
            {"document": "other.pdf", "page": 0, "quote": "elsewhere"}]),
        _finding(id="F3", severity="minor", citations=[
            {"document": "/tmp/up/r.pdf", "page": gt.table_page}]),
        _finding(id="F4", quote_verified=False, confidence="low", citations=[
            {"document": "r.pdf", "page": gt.sheet_page, "quote": "not there",
             "bbox": [x - 10, y - 30, x + 200, y + 10]}]),
    ]
    marks = RF.to_markups(findings, "r.pdf")
    assert [m["kind"] for m in marks] == ["highlight", "box", "note", "box"]
    assert marks[0]["quote"] == gt.heading and marks[0]["page"] == 0
    assert marks[0]["comment"].startswith("F1 (major): ")
    assert marks[3]["comment"].endswith("[low confidence]")
    for m in marks:                                # planlens' own input check
        MarkupSpec.from_dict(m)
    opened = json.loads(document_tools.dispatch_document_tool(
        "open_document", {"source": "r.pdf"}, attachments={"r.pdf": gt.pdf},
        max_chars=6000))
    out_path = str(tmp_path / "r_marked.pdf")
    written = json.loads(document_tools.dispatch_document_tool(
        "annotate_document", {"handle": opened["handle"],
                              "output_path": out_path, "markups": marks,
                              "author": "M1 test (AI draft)"},
        attachments={"r.pdf": gt.pdf}, max_chars=6000))
    assert written["n_written"] == 4 and written["n_skipped"] == 0, written
    ok, detail = check_pdf_markups("", files=[out_path], min=4,
                                   author_contains="M1 test")
    assert ok, detail
    doc = Document(filepath=out_path)
    try:
        mine = [m for m in doc.markups() if "M1 test" in (m.author or "")]
    finally:
        doc.close()
    assert sorted(m.page for m in mine) == sorted(
        [gt.narrative_page, gt.sheet_page, gt.table_page, gt.sheet_page])
    assert any("F1 (major)" in (m.text or "") for m in mine)


# ---------------------------------------------------------------------------
# The tools (GEOTECH_REVIEW_FINDINGS, lean agent only)
# ---------------------------------------------------------------------------

def test_findings_tools_only_with_lean_and_the_switch(monkeypatch):
    def names():
        return set(_tools_of(build_deep_agent(_model([]),
                                              **_review_kwargs())))
    assert not names() & FINDINGS_TOOLS                  # everything off
    monkeypatch.setenv(review_flags.FINDINGS_ENV, "1")
    assert not names() & FINDINGS_TOOLS                  # legacy page: no
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    monkeypatch.delenv(review_flags.FINDINGS_ENV)
    assert not names() & FINDINGS_TOOLS                  # lean alone: no
    monkeypatch.setenv(review_flags.FINDINGS_ENV, "1")
    agent = build_deep_agent(_model([]), **_review_kwargs())
    tools = _tools_of(agent)
    assert FINDINGS_TOOLS <= set(tools)
    for name in FINDINGS_TOOLS:
        for word in GEOTECH_WORDS:
            assert word not in (tools[name].description or ""), (name, word)
    # the tool schema reaches an OpenAI-style endpoint without references
    from langchain_core.utils.function_calling import convert_to_openai_tool
    for name in ("record_finding", "update_finding"):
        schema = json.dumps(convert_to_openai_tool(tools[name]))
        assert "$ref" not in schema and "$defs" not in schema, name


def _tools(attachments):
    from funhouse_agent.deep.findings_tools import make_findings_tools
    return {t.name: t for t in make_findings_tools(
        attachments, markup_author="M1 test (AI draft)")}


def test_record_list_and_report(gt, tmp_path):
    tools = _tools({"r.pdf": gt.pdf})
    rec = tools["record_finding"]
    good = json.loads(rec.invoke({
        "statement": "The report heading is present.", "severity": "info",
        "confidence": "high", "evidence": "read",
        "citations": [{"document": "r.pdf", "page": gt.narrative_page,
                       "quote": gt.heading}]}))
    assert good["recorded"] == "F1" and good["quote_verified"] is True
    bad = json.loads(rec.invoke({
        "statement": "A 5000 psi concrete strength is specified.",
        "severity": "major", "confidence": "high", "evidence": "read",
        "disciplines": ["structural"],
        "citations": [{"document": "r.pdf", "page": gt.narrative_page,
                       "quote": "5000 psi concrete strength"}]}))
    assert bad["recorded"] == "F2" and bad["quote_verified"] is False
    assert bad["confidence"] == "low" and "look at that page" in bad["note"]
    error = json.loads(rec.invoke({
        "statement": "x", "severity": "huge", "confidence": "high",
        "evidence": "read", "citations": [{"document": "r.pdf", "page": 0}]}))
    assert "severity must be one of" in error["error"]
    listed = json.loads(tools["list_findings"].invoke({}))
    assert listed["count"] == 2
    assert [f["id"] for f in json.loads(tools["list_findings"].invoke(
        {"discipline": "Structural"}))["findings"]] == ["F2"]
    assert json.loads(tools["list_findings"].invoke(
        {"severity": "critical"}))["count"] == 0
    # Markdown, and a marked-up copy written into the working folder
    rep = json.loads(tools["findings_report"].invoke(
        {"format": "markdown", "marked_up_pdf": True}))
    assert rep["count"] == 2
    assert rep["markdown"].index("F2") < rep["markdown"].index("F1")
    marked = rep["marked_up"][0]
    # F1 highlights its quote; F2's quote is not on the page, so its comment
    # goes on that page as a note instead of being lost
    assert marked["document"] == "r.pdf" and marked["n_written"] == 2, marked
    assert marked["placed_as_notes"] == 1 and marked["n_skipped"] == 0
    assert os.path.dirname(marked["output_path"]) == str(tmp_path / "work")
    # its own name: never annotate_document's <document>_marked.pdf
    assert os.path.basename(marked["output_path"]) == "r_findings.pdf"
    # redone from the findings, not appended to: the same count again
    again = json.loads(tools["findings_report"].invoke(
        {"format": "markdown", "marked_up_pdf": True}))["marked_up"][0]
    assert again["n_written"] == marked["n_written"]
    # a rejected finding is left out of the deliverables
    RF.FindingsLedger().update("F2", status="rejected")
    rep = json.loads(tools["findings_report"].invoke({"format": "markdown"}))
    assert rep["count"] == 1 and "F2" not in rep["markdown"]


def test_report_as_word(gt, tmp_path):
    """The comment log goes through write_docx's own writer with the host's
    save function: the page's working folder and its download card."""
    docx = pytest.importorskip("docx")
    from webapp import core
    from funhouse_agent.deep.findings_tools import make_findings_tools
    artifacts = []
    save_fn = core.make_save_fn(str(tmp_path / "work"), artifacts)
    tools = {t.name: t for t in make_findings_tools({"r.pdf": gt.pdf},
                                                    save_fn=save_fn)}
    tools["record_finding"].invoke({
        "statement": "The boring table lists four borings.",
        "severity": "info", "confidence": "high", "evidence": "read",
        "citations": [{"document": "r.pdf", "page": gt.table_page}]})
    rep = json.loads(tools["findings_report"].invoke(
        {"path": "comments", "title": "Comments"}))
    assert rep["docx"].get("saved"), rep
    path = tmp_path / "work" / "comments.docx"
    assert path.is_file() and artifacts == [str(path)]
    assert rep["docx"].get("note")                  # the host's saved note
    text = "\n".join(c.text for row in docx.Document(str(path)).tables[0].rows
                     for c in row.cells)
    assert "The boring table lists four borings." in text
    assert "PDF p. 3" in text


def test_nothing_to_report_is_said_plainly():
    rep = json.loads(_tools({})["findings_report"].invoke({}))
    assert "no findings" in rep["error"]


def test_the_lean_agent_records_a_finding_end_to_end(gt, monkeypatch,
                                                    tmp_path):
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    monkeypatch.setenv(review_flags.FINDINGS_ENV, "1")
    model = _model([
        _call("record_finding", {
            "statement": "The heading reads as a geotechnical report.",
            "severity": "info", "confidence": "high", "evidence": "read",
            "citations": [{"document": "r.pdf", "page": 0,
                           "quote": gt.heading, "pdf_page": 1}]}),
        AIMessage(content="Recorded F1."),
    ])
    agent = build_deep_agent(model, attachments={"r.pdf": gt.pdf},
                             **_review_kwargs())
    agent.invoke({"messages": [{"role": "user", "content": "review"}]})
    saved = RF.FindingsLedger().load()
    assert [f.id for f in saved] == ["F1"]
    assert saved[0].quote_verified is True
    assert saved[0].source == "review agent"


# ---------------------------------------------------------------------------
# The contact sheet shown to the main model (inline + lean)
# ---------------------------------------------------------------------------

@pytest.fixture
def sheet_png(gt, tmp_path):
    from planlens.document import Document
    doc = Document(content=gt.pdf)
    try:
        png, _info = doc.render_thumbnails()[0]
    finally:
        doc.close()
    # in the working folder: the read tools read the conversation's files
    path = tmp_path / "work" / "contact_sheet_1.png"
    path.write_bytes(png)
    return str(path)


def _analyze_image(inline, attachments=None):
    eng = FakeEngine()
    tool = next(t for t in make_vision_tools(
        engine=eng, attachments=attachments or {}, inline_images=inline,
        inline_image_files=inline)
        if t.name == "analyze_image")
    return tool, eng


def test_inline_analyze_image_of_a_file_is_shown_not_described(sheet_png):
    inline_store.clear()
    tool, eng = _analyze_image(True)
    out = json.loads(tool.invoke({"attachment_key": sheet_png,
                                  "prompt": "which pages are drawings?"}))
    assert eng.prompts == []                       # no one-shot call
    data, meta = inline_store.get(out["image_id"])
    assert data[:4] == b"\x89PNG" and "contact_sheet_1.png" in meta["label"]
    assert "look at it yourself" in out["note"]
    # an upload named by its key keeps the vision call
    with open(sheet_png, "rb") as fh:
        tool, eng = _analyze_image(True, {"photo.png": fh.read()})
    out = json.loads(tool.invoke({"attachment_key": "photo.png"}))
    assert out == {"analysis": "seen"} and len(eng.prompts) == 1
    # switch off: unchanged (the same read again in this conversation would
    # be served as a repeat, B6 — forget it so the call is made)
    from funhouse_agent.vision_tools import clear_repeat_reads
    clear_repeat_reads()
    tool, eng = _analyze_image(False)
    out = json.loads(tool.invoke({"attachment_key": sheet_png}))
    assert out == {"analysis": "seen"} and len(eng.prompts) == 1


def test_inline_analyze_image_refuses_what_is_not_an_image(tmp_path):
    """A text file is not shown and not sent as a picture: it is refused,
    naming the tool that reads it (wave 2b, C6: every file used to go out
    labelled PNG, and an .xlsx came back as a 400 from the API)."""
    not_image = tmp_path / "work" / "notes.txt"
    not_image.write_text("plain text", encoding="utf-8")
    tool, eng = _analyze_image(True)
    out = json.loads(tool.invoke({"attachment_key": str(not_image)}))
    assert "read_text_file" in out["error"] and eng.prompts == []


def test_the_lean_inline_agent_sees_the_contact_sheet(sheet_png, monkeypatch):
    inline_store.clear()
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    monkeypatch.setenv(review_flags.VISION_INLINE_ENV, "1")
    monkeypatch.setenv(review_flags.OVERVIEW_ENV, "1")    # both switches
    eng = FakeEngine()
    model = _model([
        _call("analyze_image", {"attachment_key": sheet_png,
                                "prompt": "which pages are drawings?"}),
        AIMessage(content="Page 1 is a drawing sheet."),
    ])
    agent = build_deep_agent(model, engine=eng, **_review_kwargs())
    tools = _tools_of(agent)
    assert "shown to you" in tools["analyze_image"].description
    agent.invoke({"messages": [{"role": "user", "content": "orient"}]})
    assert eng.prompts == []
    last = model.seen[1][-1]
    assert isinstance(last, HumanMessage)
    assert any(b.get("type") == "image_url" for b in last.content)
    assert any("image file contact_sheet_1.png" in b.get("text", "")
               for b in last.content)


def test_the_lean_agent_without_inline_still_describes_the_image(
        sheet_png, monkeypatch):
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    eng = FakeEngine()
    agent = build_deep_agent(_model([]), engine=eng, **_review_kwargs())
    tool = _tools_of(agent)["analyze_image"]
    assert "shown to you" not in tool.description
    assert json.loads(tool.invoke({"attachment_key": sheet_png})) == \
        {"analysis": "seen"}


def test_page_images_are_labelled_as_before():
    from funhouse_agent.deep.inline_images import image_message
    inline_store.clear()
    page = inline_store.put(b"\x89PNG page", {"page": 2, "pdf_page": 3,
                                              "view": [0, 0, 10, 10]})
    msg = image_message([page])
    assert msg.content[1]["text"] == (f"Image {page}: PDF page 3 (tool page "
                                      f"2), view [0, 0, 10, 10]")


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------

def test_new_switches_are_cleared_by_every_arm(monkeypatch):
    assert {review_flags.FINDINGS_ENV, review_flags.OVERVIEW_ENV} <= set(
        review_flags.ALL_ENVS)
    monkeypatch.setenv(review_flags.FINDINGS_ENV, "1")
    monkeypatch.setenv(review_flags.OVERVIEW_ENV, "1")
    with review_flags.switches(review_flags.ARMS["lean"]):
        assert not review_flags.findings() and not review_flags.overview()
    assert review_flags.findings() and review_flags.overview()
    with review_flags.switches(review_flags.ARMS["overview"]):
        assert review_flags.overview() and review_flags.lean_agent()
        assert not review_flags.findings()
    assert "GEOTECH_REVIEW_FINDINGS=1" in review_flags.describe()


def test_arms_are_the_old_ones_plus_overview():
    A = review_flags
    assert A.ARMS == {
        "baseline": {},
        "lean": {A.AGENT_ENV: "lean"},
        "grounded": {A.AGENT_ENV: "lean", A.VISION_TEXT_ENV: "1",
                     A.VISION_STRUCTURED_ENV: "1"},
        "inline": {A.AGENT_ENV: "lean", A.VISION_TEXT_ENV: "1",
                   A.VISION_STRUCTURED_ENV: "1", A.VISION_INLINE_ENV: "1"},
        "sweep": {A.AGENT_ENV: "lean", A.VISION_TEXT_ENV: "1",
                  A.VISION_STRUCTURED_ENV: "1", A.SWEEP_ENV: "1"},
        "overview": {A.AGENT_ENV: "lean", A.VISION_TEXT_ENV: "1",
                     A.VISION_STRUCTURED_ENV: "1", A.OVERVIEW_ENV: "1"},
        # S1.1, pinned in test_review_geometry_offline.py
        "geometry": {A.AGENT_ENV: "lean", A.VISION_TEXT_ENV: "1",
                     A.VISION_STRUCTURED_ENV: "1", A.GEOMETRY_ENV: "1"},
        # shape 2 digest, pinned in test_review_digest_tools_offline.py
        "digest": {A.AGENT_ENV: "lean", A.VISION_TEXT_ENV: "1",
                   A.VISION_STRUCTURED_ENV: "1", A.DIGEST_ENV: "1"},
        # 5.32: the looking-only measuring stick, test_review_minimal_offline
        "minimal": {A.AGENT_ENV: "minimal"},
        # plan W4 (coverage in code), pinned in test_coverage_offline.py
        "coverage": {A.COVERAGE_ENV: "1"},
        "checklist": {A.COVERAGE_ENV: "1", A.CHECKLIST_ENV: "1"},
    }


def test_overview_pages_setting(monkeypatch):
    assert review_flags.overview_pages() == 20
    monkeypatch.setenv(review_flags.OVERVIEW_PAGES_ENV, "35")
    assert review_flags.overview_pages() == 35
    monkeypatch.setenv(review_flags.OVERVIEW_PAGES_ENV, "lots")
    assert review_flags.overview_pages() == 20


# ---------------------------------------------------------------------------
# Fixes from the independent review of the M1 work (2026-09-28)
# ---------------------------------------------------------------------------

def test_an_unreliable_text_layer_leaves_the_quote_unchecked():
    """Item 2a: planlens' text_reliable=False means the string is not what
    the page says, so a quote missing from it proves nothing."""
    f = _finding(citations=[{"document": "a.pdf", "page": 0,
                             "quote": "pile embedment is 6 m"}])
    RF.verify_quotes([f], lambda d, p: RF.CitedPage(
        "AB�CD�EF", text_reliable=False))
    assert f.quote_verified is None and f.confidence == "high"
    assert "text layer is not what the page shows" in f.quote_note


def test_an_unreliable_page_from_planlens_is_unchecked(tmp_path):
    import planlens.testing as PT
    build = getattr(PT, "build_unmapped_text_pdf", None)
    if build is None:
        pytest.skip("this planlens has no unmapped-text fixture")
    with RF.planlens_page_text({"u.pdf": build(0.5)}) as page_text:
        page = page_text("u.pdf", 0)
        assert page.text_reliable is False and str(page)   # it HAS text
        f = _finding(citations=[{"document": "u.pdf", "page": 0,
                                 "quote": "the pile cap is 2 m thick"}])
        RF.verify_quotes([f], page_text)
    assert f.quote_verified is None and f.confidence == "high"


def test_a_quote_in_a_review_markup_is_found(gt):
    """Item 2b: a reviewer's comment (and a stamp's drawn wording) is on the
    page too, though not in its text layer."""
    comment = _finding(citations=[{"document": "r.pdf", "page": gt.sheet_page,
                                   "quote": gt.reviewer_comment}])
    stamp = _finding(citations=[{"document": "r.pdf",
                                 "page": gt.narrative_page,
                                 "quote": gt.stamp_wording}])
    with RF.planlens_page_text({"r.pdf": gt.pdf}) as page_text:
        assert gt.reviewer_comment not in str(page_text("r.pdf",
                                                        gt.sheet_page))
        RF.verify_quotes([comment, stamp], page_text)
    assert comment.quote_verified is True, comment.quote_note
    assert "review markup" in comment.quote_note
    assert stamp.quote_verified is True, stamp.quote_note


def test_a_seen_finding_keeps_its_confidence_when_its_quote_is_not_in_text():
    """Item 2c: lettering read by looking is often not in the text layer."""
    seen = _finding(evidence="seen", citations=[
        {"document": "a.pdf", "page": 0, "quote": "W=5' MIN."}])
    read = _finding(evidence="read", citations=[
        {"document": "a.pdf", "page": 0, "quote": "W=5' MIN."}])
    RF.verify_quotes([seen, read], _page_text)
    assert seen.quote_verified is False and seen.confidence == "high"
    assert "NOT found" in seen.quote_note and "SEEN" in seen.quote_note
    assert read.quote_verified is False and read.confidence == "low"


def _open(source, attachments):
    return json.loads(document_tools.dispatch_document_tool(
        "open_document", {"source": source}, attachments=attachments,
        max_chars=6000))["handle"]


def test_a_citation_by_handle_is_resolved_and_named(gt, tmp_path):
    """Item 3: open_document returns doc_...; a finding citing it is checked
    against that document and recorded under the document's name."""
    atts = {"r.pdf": gt.pdf}
    handle = _open("r.pdf", atts)
    assert handle.startswith("doc_")
    tools = _tools(atts)
    out = json.loads(tools["record_finding"].invoke({
        "statement": "The heading is present.", "severity": "info",
        "confidence": "high", "evidence": "read",
        "citations": [{"document": handle, "page": gt.narrative_page,
                       "quote": gt.heading}]}))
    assert out["quote_verified"] is True, out
    saved = RF.FindingsLedger().get(out["recorded"])
    assert saved.citations[0].document == "r.pdf"
    rep = json.loads(tools["findings_report"].invoke(
        {"format": "markdown", "marked_up_pdf": True}))
    assert rep["marked_up"][0]["n_written"] == 1, rep
    # a handle opened from a PATH (not an upload) resolves too
    path = tmp_path / "work" / "copy.pdf"
    path.write_bytes(gt.pdf)
    by_path = _open(str(path), {})
    assert by_path != handle
    out = json.loads(_tools({})["record_finding"].invoke({
        "statement": "Heading.", "severity": "info", "confidence": "high",
        "evidence": "read",
        "citations": [{"document": by_path, "page": 0,
                       "quote": gt.heading}]}))
    assert out["quote_verified"] is True, out
    # a handle that is not open is still "not found", and says so
    out = json.loads(tools["record_finding"].invoke({
        "statement": "x.", "severity": "info", "confidence": "high",
        "evidence": "read", "citations": [{"document": "doc_0123456789",
                                           "page": 0, "quote": "anything"}]}))
    assert out["quote_verified"] is False and "not found" in out["quote_check"]


def test_update_finding_corrects_rejects_and_rechecks(gt):
    """Item 4."""
    tools = _tools({"r.pdf": gt.pdf})
    assert "update_finding" in tools["record_finding"].description
    rec = tools["record_finding"]
    f1 = json.loads(rec.invoke({
        "statement": "Heading present.", "severity": "info",
        "confidence": "high", "evidence": "read",
        "citations": [{"document": "r.pdf", "page": 0,
                       "quote": gt.heading}]}))["recorded"]
    f2 = json.loads(rec.invoke({
        "statement": "Wrong reading.", "severity": "major",
        "confidence": "high", "evidence": "read",
        "citations": [{"document": "r.pdf", "page": 0,
                       "quote": gt.heading}]}))["recorded"]
    up = tools["update_finding"]
    # new citations are checked again: this quote is not on the page
    out = json.loads(up.invoke({"id": f1, "citations": [
        {"document": "r.pdf", "page": 0, "quote": "5000 psi concrete"}]}))
    assert out["quote_verified"] is False and out["confidence"] == "low"
    out = json.loads(up.invoke({"id": f1.lower(), "confidence": "medium",
                                "citations": [{"document": "r.pdf", "page": 0,
                                               "quote": gt.heading}]}))
    assert out["quote_verified"] is True and out["confidence"] == "medium"
    out = json.loads(up.invoke({"id": f1, "status": "confirmed",
                                "severity": "minor",
                                "statement": "Heading reads correctly."}))
    saved = RF.FindingsLedger().get(f1)
    assert (saved.status, saved.severity, saved.statement) == (
        "confirmed", "minor", "Heading reads correctly.")
    assert saved.quote_verified is True             # untouched citations
    out = json.loads(up.invoke({"id": f2, "status": "rejected"}))
    assert out["status"] == "rejected" and "withdrawn" in out["note"]
    # a rejected finding is left out of the list and the report by default
    listed = json.loads(tools["list_findings"].invoke({}))
    assert [f["id"] for f in listed["findings"]] == [f1]
    assert listed["rejected_not_shown"] == 1
    asked = json.loads(tools["list_findings"].invoke({"status": "rejected"}))
    assert [f["id"] for f in asked["findings"]] == [f2]
    rep = json.loads(tools["findings_report"].invoke({"format": "markdown"}))
    assert rep["count"] == 1 and f2 not in rep["markdown"]
    # errors are results
    assert "no finding" in json.loads(up.invoke({"id": "F99",
                                                 "status": "draft"}))["error"]
    assert "nothing to change" in json.loads(up.invoke({"id": f1}))["error"]
    assert "severity must be one of" in json.loads(up.invoke(
        {"id": f1, "severity": "huge"}))["error"]
    assert RF.FindingsLedger().get(f1).severity == "minor"   # unchanged


def test_the_report_never_overwrites_the_agents_own_marked_copy(gt, tmp_path):
    """Item 5: annotate_document's default <stem>_marked.pdf is the agent's
    (or user's) hand-placed markup; the findings copy has its own name."""
    mine = tmp_path / "work" / "r_marked.pdf"
    mine.write_bytes(b"%PDF-1.4 the agent's own markup")
    tools = _tools({"r.pdf": gt.pdf})
    tools["record_finding"].invoke({
        "statement": "Heading.", "severity": "info", "confidence": "high",
        "evidence": "read", "citations": [{"document": "r.pdf", "page": 0,
                                           "quote": gt.heading}]})
    rep = json.loads(tools["findings_report"].invoke(
        {"format": "markdown", "marked_up_pdf": True}))
    assert rep["marked_up"][0]["output_path"].endswith("r_findings.pdf")
    assert mine.read_bytes() == b"%PDF-1.4 the agent's own markup"


def test_two_agents_keep_their_own_ledgers_and_digests(gt, monkeypatch,
                                                      tmp_path):
    """Item 6: the working folder is bound when each agent is built, so a
    tab that re-points GEOTECH_DEFAULT_OUTPUT_DIR cannot redirect another
    conversation's findings or digests."""
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    monkeypatch.setenv(review_flags.FINDINGS_ENV, "1")
    monkeypatch.setenv(review_flags.DIGEST_ENV, "1")
    a_dir, b_dir, elsewhere = (tmp_path / n for n in ("a", "b", "elsewhere"))
    for d in (a_dir, b_dir, elsewhere):
        d.mkdir()
    atts = {"r.pdf": gt.pdf}
    a = build_deep_agent(_model([]), attachments=atts,
                         working_dir=str(a_dir), **_review_kwargs())
    b = build_deep_agent(_model([]), attachments=atts,
                         working_dir=str(b_dir), **_review_kwargs())
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(elsewhere))
    for agent, words in ((a, "Finding in A."), (b, "Finding in B.")):
        tools = _tools_of(agent)
        tools["record_finding"].invoke({
            "statement": words, "severity": "info", "confidence": "high",
            "evidence": "read", "citations": [{"document": "r.pdf",
                                               "page": 0,
                                               "quote": gt.heading}]})
        inv = json.loads(tools["document_inventory"].invoke({}))
        assert inv["totals"]["pages"] == 5, inv
    assert [f.statement for f in RF.FindingsLedger(
        str(a_dir / "findings.json")).load()] == ["Finding in A."]
    assert [f.statement for f in RF.FindingsLedger(
        str(b_dir / "findings.json")).load()] == ["Finding in B."]
    assert (a_dir / "digest").is_dir() and (b_dir / "digest").is_dir()
    assert os.listdir(elsewhere) == []
    assert a.geotech_working_dir == str(a_dir)


def test_the_legacy_build_accepts_and_ignores_working_dir(tmp_path):
    kw = _review_kwargs()
    a = build_deep_agent(_model([]), working_dir=str(tmp_path), **kw)
    b = build_deep_agent(_model([]), **kw)
    assert set(_tools_of(a)) == set(_tools_of(b))
    assert not hasattr(a, "geotech_review_agent")


def test_the_released_inline_arm_describes_image_files_by_a_vision_call(
        sheet_png):
    """Item 9: the 5.30 ``inline`` arm (no overview switch) keeps the
    one-shot analyze_image, exactly as released."""
    from funhouse_agent.deep.review_agent import REVIEW_DESCRIPTIONS
    inline_store.clear()
    eng = FakeEngine()
    with review_flags.switches(review_flags.ARMS["inline"]):
        agent = build_deep_agent(_model([]), engine=eng, **_review_kwargs())
        tool = _tools_of(agent)["analyze_image"]
        assert tool.description == REVIEW_DESCRIPTIONS["analyze_image"]
        out = json.loads(tool.invoke({"attachment_key": sheet_png,
                                      "prompt": "which pages?"}))
        # the page tools DO go inline in that arm
        assert "shown to you" in _tools_of(agent)["analyze_pdf_page"
                                                  ].description
    assert out == {"analysis": "seen"} and eng.prompts == ["which pages?"]
    # overview alone does not show image files either
    with review_flags.switches({review_flags.AGENT_ENV: "lean",
                                review_flags.OVERVIEW_ENV: "1"}):
        agent = build_deep_agent(_model([]), engine=eng, **_review_kwargs())
        out = json.loads(_tools_of(agent)["analyze_image"].invoke(
            {"attachment_key": sheet_png}))
    assert out == {"analysis": "seen"}


@pytest.mark.parametrize("keep,words", [(None, "among the newest 2 views"),
                                        ("1", "while it is the newest view"),
                                        ("3", "among the newest 3 views")])
def test_inline_notes_say_how_long_an_image_stays_shown(sheet_png, monkeypatch,
                                                        keep, words):
    """Item 10: only the newest GEOTECH_VISION_INLINE_KEEP images are shown,
    and the notes say so."""
    if keep:
        monkeypatch.setenv(inline_store.KEEP_ENV, keep)
    assert words in vision_tools.inline_note()
    tool, _eng = _analyze_image(True)
    out = json.loads(tool.invoke({"attachment_key": sheet_png}))
    assert words in out["note"] and "look at it yourself" in out["note"]
    from funhouse_agent.deep.inline_images import InlineImageMiddleware
    assert InlineImageMiddleware().keep == int(keep or 2)


def test_a_ledger_that_cannot_be_read_just_now_is_not_set_aside(
        gt, monkeypatch, tmp_path):
    """Item 11: only content that is not JSON is corrupt; a transient read
    error must not start a fresh ledger over the findings."""
    import builtins
    ledger = RF.FindingsLedger()
    ledger.add(_finding())
    path = tmp_path / "work" / "findings.json"
    before = path.read_text(encoding="utf-8")
    real_open = builtins.open

    def flaky(file, *a, **kw):
        if os.path.abspath(str(file)) == str(path):
            raise PermissionError(13, "locked by another process", str(file))
        return real_open(file, *a, **kw)

    with monkeypatch.context() as m:
        m.setattr(RF, "open", flaky, raising=False)
        with pytest.raises(OSError):
            ledger.load()
        with pytest.raises(OSError):
            ledger.add(_finding(statement="second"))
        tools = _tools({"r.pdf": gt.pdf})
        assert "could not save" in json.loads(tools["record_finding"].invoke({
            "statement": "x.", "severity": "info", "confidence": "high",
            "evidence": "read", "citations": [{"document": "r.pdf",
                                               "page": 0}]}))["error"]
        for name, args in (("list_findings", {}), ("findings_report", {}),
                           ("update_finding", {"id": "F1",
                                               "status": "draft"})):
            assert "could not be read" in json.loads(
                tools[name].invoke(args))["error"], name
    assert not (tmp_path / "work" / "findings.unreadable.json").exists()
    assert path.read_text(encoding="utf-8") == before
    assert [f.id for f in RF.FindingsLedger(str(path)).load()] == ["F1"]


def test_the_report_counts_what_planlens_skipped_not_what_it_listed(
        monkeypatch):
    """Item 12: planlens cuts its list of skipped rows to fit; its own
    n_skipped is the count."""
    from funhouse_agent.deep import findings_tools as FT
    calls = []

    def fake(name, args, attachments=None, max_chars=None, cap=None):
        calls.append(name)
        if name == "open_document":
            return json.dumps({"handle": "doc_feedbeef00"})
        if not args["append"]:
            return json.dumps({"output_path": "/x/r_findings.pdf",
                               "n_written": 1, "n_skipped": 15,
                               "skipped": [{"index": 0, "reason": "no text"}],
                               "skipped_truncated_after": 1})
        return json.dumps({"output_path": "/x/r_findings.pdf",
                           "n_written": 1, "n_skipped": 0})

    monkeypatch.setattr(FT.document_tools, "dispatch_document_tool", fake)
    monkeypatch.setattr(FT.document_tools, "has_tool", lambda n: True)
    findings = [_finding(id="F1", citations=[{"document": "r.pdf", "page": 0,
                                              "quote": "pile embedment"}])]
    entry = FT._marked_up(findings, {}, "t")[0]
    assert calls == ["open_document", "annotate_document", "annotate_document"]
    assert entry["placed_as_notes"] == 1
    assert entry["n_skipped"] == 14 and entry["skipped_not_listed"] == 14


def test_an_arm_clears_the_switch_settings_it_does_not_name(monkeypatch):
    """Item 13: an arm is exactly what it says - settings included."""
    from funhouse_agent.deep.review_agent import MAX_MODEL_CALLS_ENV
    from funhouse_agent.review_digest.build import SHAPE1_PAGES_ENV
    assert set(review_flags.SETTINGS_ENVS) == {
        review_flags.OVERVIEW_PAGES_ENV, SHAPE1_PAGES_ENV,
        inline_store.KEEP_ENV, MAX_MODEL_CALLS_ENV}
    assert not set(review_flags.SETTINGS_ENVS) & set(review_flags.ALL_ENVS)
    mine = {review_flags.OVERVIEW_PAGES_ENV: "5", SHAPE1_PAGES_ENV: "3",
            inline_store.KEEP_ENV: "1", MAX_MODEL_CALLS_ENV: "9"}
    for k, v in mine.items():
        monkeypatch.setenv(k, v)
    with review_flags.switches(review_flags.ARMS["lean"]):
        assert not any(os.environ.get(k) for k in mine)
        assert review_flags.overview_pages() == 20
    with review_flags.switches({**review_flags.ARMS["overview"],
                                review_flags.OVERVIEW_PAGES_ENV: "30"}):
        assert review_flags.overview_pages() == 30
        assert os.environ.get(SHAPE1_PAGES_ENV) is None
    assert {k: os.environ.get(k) for k in mine} == mine     # restored
