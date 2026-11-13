"""Offline tests of ``measure_placement``'s scoring, on made-up rows.

The script itself reads the private corpus; what is checked here is the
arithmetic it reports: two boxes on one printed line, the pipeline object
that is the same object, the answer's pages read once, the 90 % gate with
page-level placements kept out of it, the open and blind split and the
RESULTS.md it writes.

Run::

    .venv/Scripts/python -m pytest module_work/report_ingest_harness/tests/test_measure_placement.py -q
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from module_work.report_ingest_harness import measure_placement as mp


def test_two_boxes_on_one_line_and_on_one_spot():
    got = mp.compare([100, 200, 140, 210], 3,
                     {"page": 3, "bbox": [300, 201, 340, 211]})
    assert got["same_page"] and got["same_line"] and not got["same_spot"]
    got = mp.compare([100, 200, 140, 210], 3,
                     {"page": 3, "bbox": [120, 205, 160, 215]})
    assert got["same_line"] and got["same_spot"]
    got = mp.compare([100, 200, 140, 210], 3,
                     {"page": 3, "bbox": [100, 240, 140, 250]})
    assert not got["same_line"] and got["dy"] == 40.0
    assert mp.compare([0, 0, 1, 1], 3, {"page": 4, "bbox": [0, 0, 1, 1]}) == {
        "same_page": False}


def test_a_log_object_pairs_by_depth_then_label():
    theirs = [{"kind": "sample", "top": 5.0, "label": "S-2", "page": 0,
               "bbox": [1, 2, 3, 4]},
              {"kind": "sample", "top": 5.0, "label": "S-3", "page": 0,
               "bbox": [1, 2, 3, 4]},
              {"kind": "drive", "top": 5.0, "label": "", "page": 0,
               "bbox": None}]
    mine = SimpleNamespace(top=SimpleNamespace(value=5.0), sample_id="S-3")
    assert mp.pair("logs", "sample", mine, theirs)["label"] == "S-3"
    mine = SimpleNamespace(top=SimpleNamespace(value=7.5), sample_id="S-3")
    assert mp.pair("logs", "sample", mine, theirs) is None


def test_a_calculation_value_pairs_by_number_then_label():
    theirs = [{"kind": "input", "name": "Footing width", "top": 6.5,
               "page": 0, "bbox": [1, 2, 3, 4]},
              {"kind": "input", "name": "Footing length", "top": 6.5,
               "page": 0, "bbox": [1, 2, 3, 4]}]
    mine = SimpleNamespace(name="Footing length L",
                           value=SimpleNamespace(value=6.5))
    assert mp.pair("calc", "input", mine, theirs)["name"] == "Footing length"


def _row(status, kind="input", same_line=None, page_level=False):
    row = {"what": f"{kind} x", "kind": kind, "cited": 2, "page": 2,
           "status": status, "bbox": [1, 2, 3, 4] if status == "placed"
           else None, "anchor": "number" if status == "placed" else "",
           "source": "pdf_text", "page_words": 40}
    if page_level:
        row.update(anchor="test title", page_level=True)
    if same_line is not None:
        row["pipeline"] = {"same_page": True, "same_line": same_line,
                           "same_spot": same_line, "dy": 0.0 if same_line
                           else 30.0, "their_page": 2,
                           "their_bbox": [1, 40, 3, 44],
                           "mine_reads": "6.5", "theirs_reads": "7.0"}
    return row


def _item(rows, which="open", name="calc__RXX_p2"):
    counts = {s: sum(1 for r in rows if r["status"] == s)
              for s in ("placed", "moved", "several", "not_found", "no_text")}
    return {"item": name, "report": "RXX", "pages": [2], "set": which,
            "step1": {"cited": len(rows)}, "note": "", "counts": counts,
            "objects": rows, "pipeline_objects": len(rows),
            "pipeline_boxed": len(rows)}


def test_the_gate_counts_values_on_pages_with_text(tmp_path):
    rows = [_row("placed", same_line=True)] * 8 + [
        _row("placed", same_line=False), _row("not_found"),
        _row("no_text"), _row("no_text")]
    summary = mp.stage_summary([_item(rows)])
    assert (summary["total"], summary["with_text"], summary["placed"]) == (
        12, 10, 9)
    assert summary["rate"] == 0.9
    assert summary["compare"]["paired"] == 9
    assert summary["compare"]["same_line"] == 8

    mp.write_results(tmp_path, {"calc": [_item(rows)]})
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "| calc | open | 12 | 10 | 9/10 (90.0 %) |" in text
    assert "PASS" in text
    assert "8/9 (88.9 %)" in text
    assert "### Same page, another line than the other box" in text
    saved = json.loads((tmp_path / "results.json").read_text(
        encoding="utf-8"))
    assert saved["summaries"]["calc"]["placed"] == 9


def test_below_the_gate_is_a_miss(tmp_path):
    rows = [_row("placed")] * 8 + [_row("several")] * 2
    mp.write_results(tmp_path, {"calc": [_item(rows)]})
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "| 8/10 (80.0 %) |" in text and "MISS" in text


def test_a_page_level_placement_is_kept_out_of_the_gate(tmp_path):
    rows = [_row("placed")] * 8 + [_row("placed", page_level=True)] * 2
    summary = mp.stage_summary([_item(rows)])
    assert (summary["placed"], summary["page_level"]) == (8, 2)
    assert summary["rate"] == 0.8


def test_boring_headings_are_counted_apart_from_the_gate(tmp_path):
    def heading(page_level):
        row = _row("placed", kind="test", page_level=page_level)
        row.update(anchor="test title", test_kind="moisture_content",
                   boring_heading=True)
        return row
    rows = ([_row("placed", kind="test")] * 8
            + [heading(True), heading(False)])
    summary = mp.stage_summary([_item(rows)])
    assert (summary["placed"], summary["boring_headings"]) == (9, 2)
    assert summary["boring_headings_by_outcome"] == {"page level": 1,
                                                     "title spot": 1}
    mp.write_results(tmp_path, {"lab": [_item(rows)]})
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "| 9/10 (90.0 %) | 1 | 8/10 (80.0 %) | 2 |" in text


def test_lab_titles_are_counted_both_ways_and_by_kind(tmp_path):
    def title(kind, page_level):
        row = _row("placed", kind="test", page_level=page_level)
        row.update(anchor="test title", test_kind=kind)
        return row
    rows = ([_row("placed", kind="test")] * 7
            + [title("summary_table", False), title("compaction", False),
               title("atterberg", True)])
    summary = mp.stage_summary([_item(rows)])
    assert (summary["placed"], summary["titles_apart"],
            summary["page_level"]) == (9, 7, 1)
    assert summary["sheet_titles_by_kind"] == {"summary_table": 1,
                                               "compaction": 1}
    assert summary["page_level_by_kind"] == {"atterberg": 1}
    blind = [_item([title("other", False)], "blind", "lab__RBLIND")]
    mp.write_results(tmp_path, {"lab": [_item(rows)] + blind})
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "| lab | open | 10 | 10 | 9/10 (90.0 %) | 1 | 7/10 (70.0 %) |" \
        in text
    assert "| lab | blind | other 1 | none |" in text
    assert "lab__RBLIND" not in text


def test_blind_items_are_counted_and_not_listed(tmp_path):
    open_rows = [_row("placed")] * 9 + [_row("not_found")]
    blind_rows = [_row("placed")] * 4 + [_row("not_found")]
    items = [_item(open_rows, "open", "calc__ROPEN"),
             _item(blind_rows, "blind", "calc__RBLIND")]
    mp.write_results(tmp_path, {"calc": items},
                     {"calc_sonnet": ("calc", [_item(open_rows)])})
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "| calc | open | 10 | 10 | 9/10 (90.0 %) |" in text
    assert "| calc | blind | 5 | 5 | 4/5 (80.0 %) |" in text
    assert "| calc | all | 15 | 15 | 13/15 (86.7 %) |" in text
    assert "| calc_sonnet | open | 10 |" in text
    assert "calc__ROPEN" in text and "calc__RBLIND" not in text
    assert "Blind items (1), counts only" in text


def test_an_answers_pages_are_read_once_for_all_its_cites():
    reading = SimpleNamespace(inputs=[
        SimpleNamespace(prov=SimpleNamespace(page=0)),
        SimpleNamespace(prov=SimpleNamespace(page=1))], results=[])
    got = mp.check_answer("calc", reading, [40, 41])
    assert got["pdf_local"] == 2
    assert [v.prov.page for v in reading.inputs] == [40, 41]
    reading = SimpleNamespace(inputs=[
        SimpleNamespace(prov=SimpleNamespace(page=41)),
        SimpleNamespace(prov=SimpleNamespace(page=99))], results=[])
    got = mp.check_answer("calc", reading, [40, 41])
    assert (got["cited"], got["outside"]) == (1, 1)
    assert [v.prov.page for v in reading.inputs] == [41, 40]


def test_a_held_out_record_has_its_boxes_taken_off():
    from report_ingest.model import Calculation

    calc = Calculation.model_validate({
        "kind": "other", "pages": [3],
        "inputs": [{"name": "Width", "value": {"value": 6.5, "unit": "ft"},
                    "prov": {"page": 3, "method": "model", "confidence": 0.9,
                             "bbox": [1, 2, 3, 4]}}]})
    assert mp.strip_boxes("calc", calc) == [(3, [1.0, 2.0, 3.0, 4.0])]
    assert calc.inputs[0].prov.bbox is None


# ---------------------------------------------------------------------------
# RI-99 (3): a lab test's two boxes, in three groups
# ---------------------------------------------------------------------------

def _lab_page(tmp_path):
    """A made-up sheet: a heading line naming the sample, its results on a
    row of their own, and another sample's row below."""
    import pymupdf
    from planlens.document import open_document
    from report_ingest.locate import PageWords

    pdf = pymupdf.open()
    page = pdf.new_page()
    for x, y, text in [
            (72, 90, "Sample:"), (130, 90, "S-2"), (220, 90, "Depth 5.0 ft"),
            (72, 150, "LL"), (160, 150, "PL"), (250, 150, "PI"),
            (72, 164, "41"), (160, 164, "19"), (250, 164, "22"),
            (72, 240, "S-7"), (160, 240, "55"), (250, 240, "31")]:
        page.insert_text((x, y), text, fontsize=10)
    path = tmp_path / "made_lab.pdf"
    pdf.save(str(path))
    pdf.close()
    doc = open_document(str(path))
    return doc, PageWords(doc)(0)


def _lab_test():
    from report_ingest.model import LabTest

    return LabTest.model_validate({
        "kind": "atterberg", "investigation_id": "TB-4", "sample_id": "S-2",
        "pages": [0], "result": {"kind": "atterberg", "ll": 41, "pl": 19,
                                 "pi": 22},
        "prov": [{"page": 0, "method": "model"}]})


def test_a_lab_tests_two_boxes_are_put_in_three_groups(tmp_path):
    doc, text = _lab_page(tmp_path)
    try:
        test = _lab_test()
        results = [70, 155, 270, 167]          # the row of 41 19 22
        label = [128, 81, 150, 93]             # "S-2" in the heading line
        other = [70, 231, 270, 243]            # another sample's row
        same = {"same_page": True, "same_line": True}
        apart = {"same_page": True, "same_line": False}
        assert mp.where_beside(same, text, results, results, test) \
            == "same_line"
        assert mp.where_beside(apart, text, results, label, test) \
            == "same_block"
        assert mp.where_beside(apart, text, label, results, test) \
            == "same_block"
        assert mp.where_beside(apart, text, results, other, test) \
            == "elsewhere"
        assert mp.where_beside({"same_page": False}, text, results, label,
                               test) is None
    finally:
        doc.close()


def test_another_samples_row_sharing_one_number_is_elsewhere(tmp_path):
    """RI-99 review: a summary table where another sample's row prints one
    of the test's numbers. Placement on that row and the pipeline on the
    test's own sample's row are in different samples' rows: elsewhere."""
    import pymupdf
    from planlens.document import open_document
    from report_ingest.locate import PageWords

    pdf = pymupdf.open()
    page = pdf.new_page()
    for x, y, text in [(72, 150, "S-2"), (160, 150, "41"), (250, 150, "19"),
                       (340, 150, "22"),
                       (72, 240, "S-7"), (160, 240, "55"), (250, 240, "19"),
                       (340, 240, "36")]:
        page.insert_text((x, y), text, fontsize=10)
    path = tmp_path / "made_summary.pdf"
    pdf.save(str(path))
    pdf.close()
    doc = open_document(str(path))
    try:
        text = PageWords(doc)(0)
        s2_row = [70, 141, 360, 153]
        s7_row = [70, 231, 360, 243]
        apart = {"same_page": True, "same_line": False}
        assert mp.where_beside(apart, text, s7_row, s2_row, _lab_test()) \
            == "elsewhere"
    finally:
        doc.close()


def _lab_row(where):
    row = _row("placed", kind="test", same_line=(where == "same_line"))
    row["pipeline"]["where"] = where
    return row


def test_the_lab_groups_are_reported_apart_and_only_elsewhere_is_listed(
        tmp_path):
    rows = [_lab_row("same_line")] * 2 + [_lab_row("same_block")] * 5 + [
        _lab_row("elsewhere")]
    summary = mp.stage_summary([_item(rows, name="lab__RXX_p2")])
    assert [summary["compare"]["lab_" + w] for w in mp.BESIDE] == [2, 5, 1]
    assert summary["compare"]["same_line"] == 2          # unchanged
    mp.write_results(tmp_path, {"lab": [_item(rows, name="lab__RXX_p2")]})
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "## Lab: where the two boxes sit" in text
    assert "| lab | open | 8 | 2/8 (25.0 %) | 5/8 (62.5 %) | " \
           "1/8 (12.5 %) |" in text
    listed = text.split("### Same page, another line than the other box")[1]
    assert listed.count("| lab__RXX_p2 |") == 1          # the one elsewhere


def test_a_stage_with_no_lab_pairs_has_no_lab_table(tmp_path):
    rows = [_row("placed", same_line=True)]
    mp.write_results(tmp_path, {"calc": [_item(rows)]})
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "## Lab: where the two boxes sit" not in text
