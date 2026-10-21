"""Cited pages checked against the pages a conversation opened (EV-18).

Field session 2026-10-14 (TRACE_REVIEW Part 3, C5): an answer gave four page
lists as "source traceability" for placeholder borings, and the DIGGS file
carried them; they were even splits of a section, not pages anything had
read. These tests drive the general check with synthetic records: the
citation parser, the record of pages opened (the activity log, the coverage
ledger, the vision records), the write_diggs cross-check entry, the
findings tool, the answer check and the turn object.
"""

import json
import os

import pytest

from funhouse_agent import cited_pages as cp
from funhouse_agent import review_flags


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    cp.clear_cache()
    yield
    cp.clear_cache()


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def _cites(text, names=()):
    return [(c.kind, c.numbers, c.label, c.elsewhere)
            for c in cp.find_citations(text, names)]


def test_pdf_citations_count_as_a_viewer_does():
    got = _cites("the plan (Figure 2, PDF page 24); logs on PDF pp. "
                 "108\u2013119; pdf_page 7; boring 13 (PDF 133-134); page 9 "
                 "of the PDF; viewer page 5")
    assert ("pdf", [24], "", "") in got
    assert ("pdf", list(range(108, 120)), "", "") in got
    assert ("pdf", [7], "", "") in got
    assert ("pdf", [133, 134], "", "") in got
    assert ("pdf", [9], "", "") in got
    assert ("pdf", [5], "", "") in got


def test_bare_page_numbers_may_be_printed_or_the_viewers():
    got = _cites("see p. 12, pages 3, 5 and 7, pp. 20-22 and Page 3 of 10")
    assert ("page", [12], "", "") in got
    assert ("page", [3], "", "") in got
    assert ("page", [5], "", "") in got and ("page", [7], "", "") in got
    # a hyphenated number is a range, or the page printed "20-22"
    assert ("page", [20, 21, 22], "20-22", "") in got
    # "Page 3 of 10" is page 3, not page 10
    assert not any(n == [10] for _k, n, _l, _e in got)


def test_printed_labels_are_labels():
    got = _cites("the table is on page A-3, the method on p. 5-2 and the "
                 "contents on p. iv")
    assert ("label", [], "a-3", "") in got
    assert ("label", [], "5-2", "") in got
    assert ("label", [], "iv", "") in got


@pytest.mark.parametrize("text", [
    "on that page I found the log",
    "a 40-page report",
    "the pages in Appendix B",
    "a multi-page log",
    "Table 5-2 and Figure 3",
    "section 4.2 on page 4.2",
    "Supporting pages\n2026 program",
])
def test_words_that_are_not_page_citations(text):
    assert _cites(text) == []


def test_a_mixed_list_keeps_each_items_own_numbering():
    got = _cites("PDF pages 3 and page 7")
    assert ("pdf", [3], "", "") in got
    assert ("page", [7], "", "") in got


def test_a_table_column_headed_pages_is_read():
    text = ("| Program | Supporting pages |\n|---|---|\n"
            "| A | 13\u201314, 24 |\n| B | 28 |\n\n"
            "| ID | PDF page |\n|--|--|\n| B-1 | 21, 22 |\n")
    got = _cites(text)
    assert ("page", [13, 14], "13-14", "") in got
    assert ("page", [24], "", "") in got and ("page", [28], "", "") in got
    assert ("pdf", [21], "", "") in got and ("pdf", [22], "", "") in got


def test_a_page_of_another_published_document_is_not_this_conversations():
    got = cp.find_citations(
        "Ka from GEC-12 (p. 7-29); Das (2010), p. 350; (Bowles 1996, p. "
        "512); UFC 3-220-10 page 45; per DM7.2 p. 7-12")
    assert all(c.elsewhere for c in got), [(c.shown, c.elsewhere)
                                           for c in got]


def test_a_boring_name_or_a_date_beside_a_page_is_not_another_document():
    got = cp.find_citations(
        "B-7 (PDF pages 30-31) was drilled 3 Mar 2021, page 5; the "
        "TM-1 log (p. 12)")
    assert got and not any(c.elsewhere for c in got)


def test_a_clause_naming_the_conversations_document_is_checked():
    got = cp.find_citations("GEC-12 is quoted in report.pdf (p. 4)",
                            ["report.pdf"])
    assert got and not got[0].elsewhere


# ---------------------------------------------------------------------------
# The record of pages opened
# ---------------------------------------------------------------------------

def _conv(tmp_path):
    conv = tmp_path / "conv"
    (conv / "files").mkdir(parents=True)
    (conv / "meta.json").write_text("{}", encoding="utf-8")
    return conv


def _write_events(conv, calls, turn=1):
    """Tool calls as the activity log records them."""
    with open(conv / "activity.jsonl", "a", encoding="utf-8") as fh:
        for i, call in enumerate(calls):
            rid = call.get("run_id") or f"r{turn}_{i}_{call['name']}"
            fh.write(json.dumps({"turn": turn, "event": "tool_start",
                                 "agent": call.get("agent", "primary"),
                                 "name": call["name"], "run_id": rid,
                                 "args": call.get("args", {})}) + "\n")
            end = {"turn": turn, "event": call.get("event", "tool_end"),
                   "agent": call.get("agent", "primary"),
                   "name": call["name"], "run_id": rid,
                   "truncated": bool(call.get("full"))}
            result = call.get("result", {})
            text = result if isinstance(result, str) else json.dumps(result)
            if end["event"] == "tool_error":
                end["error"] = text
            else:
                end["result"] = text
            if call.get("full"):
                end["full"] = call["full"]
            fh.write(json.dumps(end) + "\n")


HISTORY = [
    {"name": "open_document", "args": {"source": "report.pdf"},
     "result": {"handle": "doc_abc", "name": "report.pdf", "n_pages": 160}},
    {"name": "read_document", "args": {"handle": "doc_abc", "pages": "0-2"},
     "result": {"handle": "doc_abc", "pages_returned": "0-2"}},
    {"name": "analyze_pdf_page", "agent": "general-purpose",
     "args": {"attachment_key": "report.pdf", "page": 9},
     "result": {"page": 9, "pdf_page": 10, "analysis": "a log"}},
    {"name": "log_grid", "args": {"handle": "doc_abc", "pages": "20-21"},
     "result": {"handle": "doc_abc", "pages": [20, 21],
                "rows": [{"page": 20}, {"page": 21}], "next_offset": 40}},
    {"name": "search_document", "args": {"handle": "doc_abc",
                                         "pattern": "boring"},
     "result": {"handle": "doc_abc", "hits": [{"page": 30,
                                               "text": "boring 12"}]}},
    {"name": "read_pdf_text", "args": {"source": "report.pdf",
                                       "pages": "40"},
     "result": {"pages": [{"page": 40, "has_text_layer": False}]}},
    {"name": "render_region", "event": "tool_error",
     "args": {"attachment_key": "report.pdf", "page": 70},
     "result": "boom"},
]


def test_the_record_says_how_each_page_was_reached(tmp_path):
    conv = _conv(tmp_path)
    _write_events(conv, HISTORY)
    opened = cp.pages_opened(str(conv), folder=str(conv / "files"))
    assert opened.knows_text and "activity log" in opened.records
    # the handle and the attachment key are one document
    assert list(opened.pages) == ["report.pdf"]
    status = {p: opened.status(p) for p in (0, 2, 9, 20, 21, 30, 40, 70, 99)}
    assert status == {0: "opened", 2: "opened", 9: "opened", 20: "opened",
                      21: "opened", 30: "hit", 40: "text_empty",
                      70: "never", 99: "never"}
    assert opened.n_pages["report.pdf"] == 160


def test_a_result_cut_short_is_read_from_its_whole_copy(tmp_path):
    conv = _conv(tmp_path)
    (conv / "activity_full").mkdir()
    whole = json.dumps({"handle": "doc_abc", "pages_returned": "50-51",
                        "text": "x" * 100})
    (conv / "activity_full" / "t1.txt").write_text(whole, encoding="utf-8")
    _write_events(conv, HISTORY[:1] + [
        {"name": "read_document", "args": {"handle": "doc_abc"},
         "result": whole[:20] + "\n...[truncated]",
         "full": "activity_full/t1.txt"}])
    opened = cp.pages_opened(str(conv), folder=str(conv / "files"))
    assert opened.status(50) == opened.status(51) == "opened"


def test_the_log_is_read_again_as_it_grows(tmp_path):
    conv = _conv(tmp_path)
    _write_events(conv, HISTORY[:2])
    assert cp.pages_opened(str(conv), str(conv / "files")).status(9) \
        == "never"
    _write_events(conv, HISTORY[2:3], turn=2)
    assert cp.pages_opened(str(conv), str(conv / "files")).status(9) \
        == "opened"


def test_the_coverage_ledger_and_the_vision_reads_count(tmp_path):
    conv = _conv(tmp_path)
    (conv / "coverage.json").write_text(json.dumps({"documents": {
        "sha1:x": {"name": "report.pdf", "source": "report.pdf",
                   "inventory": {"n_pages": 10},
                   "reads": {"4": {"how": ["text"]},
                             "5": {"how": ["text_empty"]}}}}}),
        encoding="utf-8")
    (conv / "reads.json").write_text(json.dumps({"reads": [
        {"document": "report.pdf", "page": 7, "tool": "analyze_pdf_page"}]}),
        encoding="utf-8")
    opened = cp.pages_opened(str(conv), folder=str(conv / "files"))
    assert opened.knows_text
    assert {"coverage ledger", "vision reads"} <= set(opened.records)
    assert opened.status(4) == "opened"
    assert opened.status(5) == "text_empty"
    assert opened.status(7) == "opened"


def test_the_record_folder_is_found_from_the_working_folder(tmp_path):
    conv = _conv(tmp_path)
    _write_events(conv, HISTORY[:1])
    from funhouse_agent._fileio import working_dir_bound
    with working_dir_bound(str(conv / "files")):
        assert cp.record_dir() == str(conv)
    elsewhere = tmp_path / "custom"
    elsewhere.mkdir()
    with working_dir_bound(str(elsewhere)):
        assert cp.record_dir() is None
        with cp.record_bound(str(conv)):          # the turn binds it
            assert cp.record_dir() == str(conv)
        assert cp.record_dir() is None


# ---------------------------------------------------------------------------
# A written file's pages
# ---------------------------------------------------------------------------

def _opened(tmp_path):
    conv = _conv(tmp_path)
    _write_events(conv, HISTORY)
    return cp.pages_opened(str(conv), folder=str(conv / "files"))


def test_each_record_citing_a_page_nobody_opened_gets_one_entry(tmp_path):
    opened = _opened(tmp_path)
    groups = {"investigations": [
        {"investigation_id": "B-1", "pages": [0, 1]},
        {"investigation_id": "B-2", "pages": [2],
         "layers": [{"top": {"value": 0, "unit": "m",
                             "prov": {"page": 9}}}]},
        {"investigation_id": "B-12", "pages": list(range(107, 119))},
        {"investigation_id": "B-14", "pages": [20, 21, 22, 23]},
        {"investigation_id": "B-20"}],
        "lab_tests": [
            {"kind": "atterberg", "investigation_id": "B-1",
             "sample_id": "S-1", "pages": [30]},
            {"kind": "gradation", "investigation_id": "B-1",
             "sample_id": "S-2", "pages": [40]}]}
    entries, summary = cp.record_entries(groups, opened=opened)
    by = {e["where"]: e for e in entries}
    assert set(by) == {"B-12", "B-14", "B-1 S-1 (atterberg)",
                       "B-1 S-2 (gradation)"}
    assert all(e["kind"] == cp.ENTRY_KIND for e in entries)
    assert by["B-12"]["pages"] == list(range(107, 119))
    assert "PDF pages 108-119" in by["B-12"]["detail"]
    assert "any of them" in by["B-12"]["detail"]
    assert by["B-14"]["pages"] == [22, 23]
    assert "PDF 21-22 were read" in by["B-14"]["detail"]
    assert "only in search results" in by["B-1 S-1 (atterberg)"]["detail"]
    assert "no text layer" in by["B-1 S-2 (gradation)"]["detail"]
    assert summary["checked"] is True
    assert summary["not_read"] == 12 + 2 + 1 + 1


def test_a_prov_page_counts_as_cited(tmp_path):
    opened = _opened(tmp_path)
    groups = {"investigations": [{
        "investigation_id": "B-3", "pages": [0],
        "prov": [{"page": 0}, {"page": 77}]}]}
    entries, _ = cp.record_entries(groups, opened=opened)
    assert [e["pages"] for e in entries] == [[77]]


def test_without_a_record_of_text_reads_nothing_is_claimed(tmp_path):
    empty = cp.PagesOpened()
    entries, summary = cp.record_entries(
        {"investigations": [{"investigation_id": "B-1", "pages": [5]}]},
        opened=empty)
    assert entries == []
    assert summary["checked"] is False and "no activity log" in \
        summary["why"]


def test_records_without_pages_are_not_checked(tmp_path):
    entries, summary = cp.record_entries(
        {"investigations": [{"investigation_id": "B-1"}]},
        opened=_opened(tmp_path))
    assert entries == [] and summary["why"] == "no record cites a page"


def test_unopened_citations_for_the_findings_tool(tmp_path):
    opened = _opened(tmp_path)
    got = cp.unopened_citations([("report.pdf", 9), ("report.pdf", 70),
                                 ("doc_zzz", 0)], opened=opened)
    assert got == [{"document": "report.pdf", "page": 70, "pdf_page": 71,
                    "status": "never"}]
    assert cp.unopened_citations([("report.pdf", 70)],
                                 opened=cp.PagesOpened()) is None


# ---------------------------------------------------------------------------
# The answer
# ---------------------------------------------------------------------------

def _labels(rows):
    """A label source: the same printed numbers for every document."""
    return lambda opened, key: {p: set(ls) for p, ls in rows.items()}


def test_answer_pages_nobody_opened_are_listed(tmp_path):
    opened = _opened(tmp_path)
    res = cp.check_answer(
        "B-1 is on PDF page 10. Boring 12 (PDF pages 108-119) and the lab "
        "sheet (PDF page 31). Logs on PDF pages 1-5.",
        opened=opened, label_source=_labels({}))
    assert res["checked"] and res["n_cited"] == 4
    never = [r["cited"] for r in res.get("never_opened", [])]
    assert never == ["PDF pages 108-119"]
    assert [r["cited"] for r in res["hit_only"]] == ["PDF page 31"]
    part = res["partly_opened"][0]
    assert part["cited"] == "PDF pages 1-5"
    assert part["not_opened"] == "PDF pages 4-5" and part["opened"] == \
        "3 of 5"


def test_a_printed_number_counts_when_its_page_was_opened(tmp_path):
    opened = _opened(tmp_path)
    # PDF page 10 (0-based 9) prints "12"; PDF page 12 was never opened.
    labels = _labels({9: {"12"}, 60: {"13"}})
    res = cp.check_answer("the log is on page 12", opened=opened,
                          label_source=labels)
    assert "never_opened" not in res and "unverified" not in res
    res = cp.check_answer("the plan is on page 13", opened=opened,
                          label_source=labels)
    row = res["never_opened"][0]
    assert row["pages"] == "PDF page 61"
    assert row["read_as"] == "printed page number"


def test_a_bare_number_nobody_opened_either_way_is_listed(tmp_path):
    res = cp.check_answer("see page 50", opened=_opened(tmp_path),
                          label_source=_labels({}))
    assert res["never_opened"][0]["pages"] == "PDF page 50"


def test_printed_labels_resolve_through_the_page_map(tmp_path):
    opened = _opened(tmp_path)
    labels = _labels({2: {"a-3"}, 44: {"b-1"}})
    res = cp.check_answer("table on page A-3; figure on page B-1; and "
                          "page Z-9", opened=opened, label_source=labels)
    assert [r["cited"] for r in res["never_opened"]] == ["page B-1"]
    assert [r["cited"] for r in res["unverified"]] == ["page Z-9"]


def test_without_the_printed_numbers_a_bare_number_is_unverified(tmp_path):
    res = cp.check_answer("page 50 and PDF page 50", opened=_opened(tmp_path),
                          label_source=lambda o, k: None)
    assert [r["cited"] for r in res["unverified"]] == ["page 50"]
    assert [r["cited"] for r in res["never_opened"]] == ["PDF page 50"]


def test_a_pdf_page_beyond_every_document_is_said(tmp_path):
    res = cp.check_answer("PDF page 500", opened=_opened(tmp_path),
                          label_source=_labels({}))
    assert res["never_opened"][0]["note"] == \
        "no document read here has that page"


def test_other_documents_pages_are_not_checked(tmp_path):
    res = cp.check_answer("Kp from GEC-12 (p. 7-29) and UFC 3-220-10 page "
                          "45", opened=_opened(tmp_path),
                          label_source=_labels({}))
    assert "never_opened" not in res
    assert len(res["other_documents"]) == 2


def test_an_answer_with_no_record_is_not_checked():
    res = cp.check_answer("PDF page 3", opened=cp.PagesOpened())
    assert res["checked"] is False and res["n_cited"] == 1
    assert cp.check_answer("no pages here")["n_cited"] == 0


def test_the_notes_name_the_pages():
    res = {"never_opened": [{"cited": "PDF pages 108-119",
                             "pages": "PDF pages 108-119"},
                            {"cited": "page 13", "pages": "PDF page 61"}]}
    user = cp.note_for_user(res)
    assert "PDF pages 108-119" in user and "page 13 (PDF page 61)" in user
    assert "no tool or helper read or looked at" in user
    model = cp.note_for_model(res)
    assert model.startswith(cp.NOTE_PREFIX)
    assert "- page 13 (PDF page 61)" in model and "comes once" in model
    assert cp.note_for_user({}) is None and cp.note_for_model({}) is None


def test_the_printed_numbers_a_page_map_row_carries():
    class Row:
        printed_page = 12
        label = "A-3"
        header = "PROJECT X | 2-1"
        footer = "Geotechnical report | iii"
    assert cp.summary_labels(Row()) == {"12", "a-3", "2-1", "iii"}


def test_a_real_pdfs_printed_numbers_are_read(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pytest.importorskip("planlens")
    conv = _conv(tmp_path)
    pdf = pymupdf.open()
    for n in (11, 12, 13):
        page = pdf.new_page(width=612, height=792)
        page.insert_text((72, 120), f"Body text of the report, sheet {n}.",
                         fontsize=11)
        page.insert_text((290, 770), f"Page {n}", fontsize=10)
    pdf.save(str(conv / "files" / "report.pdf"))
    pdf.close()
    _write_events(conv, [
        {"name": "analyze_pdf_page",
         "args": {"attachment_key": "report.pdf", "page": 1},
         "result": {"page": 1, "analysis": "page 12"}}])
    from funhouse_agent._fileio import working_dir_bound
    with working_dir_bound(str(conv / "files")):
        opened = cp.pages_opened(str(conv))
        res = cp.check_answer("as printed on page 12; and see page 13",
                              opened=opened)
    rows = res.get("never_opened") or []
    assert [r["cited"] for r in rows] == ["page 13"], res
    assert rows[0]["pages"] == "PDF page 3"


# ---------------------------------------------------------------------------
# One turn
# ---------------------------------------------------------------------------

def _turn(tmp_path, mode):
    conv = _conv(tmp_path)
    _write_events(conv, HISTORY)
    logged = []
    check = cp.TurnPageCheck(record_folder=str(conv), log=logged.append,
                             mode=mode, label_source=_labels({}))
    return check, logged


BAD = "Boring 12 is on PDF pages 108-119."
GOOD = "Boring 12 is on PDF page 10."


def test_off_records_and_leaves_the_answer(tmp_path):
    check, logged = _turn(tmp_path, "off")
    assert check.feedback(BAD) is None
    assert check.finish(BAD) == BAD
    assert logged[-1]["action"] == "none" and logged[-1]["mode"] == "off"
    assert logged[-1]["never_opened"][0]["cited"] == "PDF pages 108-119"


def test_note_appends_the_note(tmp_path):
    check, logged = _turn(tmp_path, "note")
    out = check.finish(BAD)
    assert out.startswith(BAD) and "Page check" in out
    assert logged[-1]["action"] == "noted"
    assert check.finish(GOOD) == GOOD


def test_retry_tells_the_model_once(tmp_path):
    check, logged = _turn(tmp_path, "retry")
    note = check.feedback(BAD)
    assert note and note.startswith(cp.NOTE_PREFIX)
    assert logged[-1]["action"] == "fed_back"
    assert check.feedback(BAD) is None              # once a turn
    assert "Page check" in check.finish(BAD)        # still cites them
    assert check.finish(GOOD) == GOOD


def test_answers_citing_nothing_are_not_recorded(tmp_path):
    check, logged = _turn(tmp_path, "note")
    assert check.finish("No page is cited here.") == "No page is cited here."
    assert logged == []


def test_the_switch():
    assert review_flags.cited_pages() == "off"
    for value, mode in (("note", "note"), ("1", "note"), ("on", "note"),
                        ("retry", "retry"), ("RETRY", "retry"),
                        ("0", "off"), ("nonsense", "off")):
        with review_flags.switches({review_flags.CITED_PAGES_ENV: value}):
            assert review_flags.cited_pages() == mode
    assert review_flags.ARMS["cited_pages_note"] == {
        review_flags.CITED_PAGES_ENV: "note"}
    assert review_flags.ARMS["cited_pages_retry"] == {
        review_flags.CITED_PAGES_ENV: "retry"}
    assert review_flags.CITED_PAGES_ENV in review_flags.ALL_ENVS


# ---------------------------------------------------------------------------
# The writers
# ---------------------------------------------------------------------------

def _m(v, unit="m"):
    return {"value": v, "unit": unit}


def _hole(iid, pages):
    return {"investigation_id": iid, "kind": "boring", "depth_unit": "m",
            "total_depth": _m(10.0),
            "layers": [{"top": _m(0.0), "bottom": _m(10.0),
                        "description": "Brown SAND", "uscs": "SP"}],
            "pages": pages}


def _write_diggs(params):
    pytest.importorskip("pydantic")
    from funhouse_agent.dispatch import call_agent
    res = call_agent("subsurface", "write_diggs", params)
    return json.loads(res) if isinstance(res, str) else res


def test_write_diggs_names_a_record_citing_pages_nobody_read(tmp_path):
    conv = _conv(tmp_path)
    _write_events(conv, HISTORY)
    from funhouse_agent._fileio import working_dir_bound
    with working_dir_bound(str(conv / "files")):
        res = _write_diggs({
            "investigations": [_hole("B-1", [20, 21]),
                               _hole("B-12", list(range(107, 119)))],
            "output_path": "site.diggs.xml"})
    assert "error" not in res, res
    assert res["file_exists"]                       # never refused
    cc = res["cross_checks"]
    entries = [e for e in cc["entries"] if e["kind"] == cp.ENTRY_KIND]
    assert [e["where"] for e in entries] == ["B-12"]
    assert entries[0]["pages"] == list(range(107, 119))
    assert cc["by_kind"][cp.ENTRY_KIND] == 1
    assert cp.ENTRY_KIND in res["verdict"]
    assert res["cited_pages"]["checked"] is True
    assert res["cited_pages"]["not_read"] == 12


def test_write_diggs_with_every_page_read_raises_nothing(tmp_path):
    conv = _conv(tmp_path)
    _write_events(conv, HISTORY)
    from funhouse_agent._fileio import working_dir_bound
    with working_dir_bound(str(conv / "files")):
        res = _write_diggs({"investigations": [_hole("B-1", [20, 21])],
                            "output_path": "site.diggs.xml"})
    assert res["cross_checks"]["n"] == 0, res["cross_checks"]
    assert res["cited_pages"] == {"checked": True, "pages_cited": 2,
                                  "not_read": 0, "against": "activity log"}


def test_write_diggs_on_a_host_with_no_record_says_so(tmp_path):
    from funhouse_agent._fileio import working_dir_bound
    with working_dir_bound(str(tmp_path)):
        res = _write_diggs({"investigations": [_hole("B-1", [20, 21])],
                            "output_path": "site.diggs.xml"})
    assert res["cross_checks"]["n"] == 0
    assert res["cited_pages"]["checked"] is False


def test_write_diggs_reads_the_record_the_turn_bound(tmp_path):
    conv = _conv(tmp_path)
    _write_events(conv, HISTORY)
    custom = tmp_path / "custom"                    # a chosen working folder
    custom.mkdir()
    from funhouse_agent._fileio import working_dir_bound
    with working_dir_bound(str(custom)), cp.record_bound(str(conv)):
        res = _write_diggs({"investigations": [_hole("B-12", [107])],
                            "output_path": "site.diggs.xml"})
    assert [e["where"] for e in res["cross_checks"]["entries"]] == ["B-12"]


def test_record_finding_names_cited_pages_nobody_opened(tmp_path):
    pytest.importorskip("langchain_core")
    conv = _conv(tmp_path)
    _write_events(conv, HISTORY)
    from funhouse_agent._fileio import working_dir_bound
    from funhouse_agent.deep.findings_tools import (RECORD_DESCRIPTION,
                                                    make_findings_tools)
    assert "pages_not_opened" in RECORD_DESCRIPTION
    tools = {t.name: t for t in make_findings_tools(
        working_dir=str(conv / "files"))}
    with working_dir_bound(str(conv / "files")):
        out = json.loads(tools["record_finding"].invoke({
            "statement": "The boring log omits the water level.",
            "severity": "minor", "confidence": "medium", "evidence": "seen",
            "citations": [{"document": "report.pdf", "page": 9},
                          {"document": "report.pdf", "page": 70}]}))
    assert out["pages_not_opened"] == ["report.pdf PDF page 71"], out
    assert "page_check" in out


def test_a_run_the_page_map_names_is_where_not_what(tmp_path):
    conv = _conv(tmp_path)
    _write_events(conv, [
        {"name": "open_document", "args": {"source": "report.pdf"},
         "result": {"handle": "doc_abc", "name": "report.pdf",
                    "n_pages": 160, "pages_by_kind": {
                        "text": "0-39", "form": "40-89,120-125"},
                    "pdf_pages_by_kind": {"text": "1-40"},
                    "segments": [{"id": 3, "pages": "107-155",
                                  "title": "older investigation"}]}},
        {"name": "read_document", "args": {"handle": "doc_abc"},
         "result": {"handle": "doc_abc", "pages_returned": "40-41"}}])
    opened = cp.pages_opened(str(conv), folder=str(conv / "files"))
    assert {(0, 39), (40, 89), (120, 125), (107, 155)} <= opened.runs
    res = cp.check_answer(
        "The logs are in PDF pages 41-90 and the older investigation in "
        "PDF pages 108-156; boring 12 is on PDF pages 108-119 and the "
        "plan on PDF page 121.", opened=opened, label_source=_labels({}))
    assert [r["cited"] for r in res["page_map"]] == [
        "PDF pages 41-90", "PDF pages 108-156"]
    assert [r["cited"] for r in res["never_opened"]] == [
        "PDF pages 108-119", "PDF page 121"]
