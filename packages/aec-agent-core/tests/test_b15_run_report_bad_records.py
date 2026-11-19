"""The run report on records that are not well formed (brief 15, BACKLOG
EV-17): a view box holding nan, null, text or too few numbers, a box outside
its page, a look with no page, a page number beyond the document, a page
image that cannot be made, a document that is gone, a last line cut off, an
empty record, an older record.

The report is still made; what cannot be drawn is left out of the drawing,
counted, and said near the top, one line a kind; nothing is repaired. The
runs are the made-up run of ``test_ev17_run_report``, one record changed.
"""

from __future__ import annotations

import json
import re

import pytest

from aec_agent_core import run_report
from test_ev17_run_report import DOC, made_up_run
from test_ev17b_page_viewer import VIEWER_DATA, _pdf

NAN = float("nan")
LEFT = re.compile(r"<div class='warn' id='left-out'>.*?</div>", re.S)


def _run(folder, pdf=True):
    made_up_run(folder, max_chars=100_000)      # every result kept whole
    if pdf:
        _pdf(folder)
    return folder


def _change(folder, event, name, nth, fn, key="result"):
    """Changes the ``nth`` record of one tool in the activity log."""
    path = folder / "activity.jsonl"
    lines, k = path.read_text(encoding="utf-8").splitlines(), 0
    for i, line in enumerate(lines):
        rec = json.loads(line)
        if rec.get("event") == event and rec.get("name") == name:
            if k == nth:
                held = rec[key] if key == "args" else json.loads(rec[key])
                fn(held)
                rec[key] = held if key == "args" else json.dumps(held)
                lines[i] = json.dumps(rec)
            k += 1
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _result(folder, name, nth, **fields):
    _change(folder, "tool_end", name, nth, lambda d: d.update(fields))


def _viewer(page):
    """The viewer's data, read as a browser reads it: no nan, no Infinity."""
    def refuse(word):
        raise ValueError(f"{word} in the viewer's data")
    m = VIEWER_DATA.search(page)
    assert m, "no viewer data in the page"
    return json.loads(m.group(1), parse_constant=refuse)


def _left(page):
    m = LEFT.search(page)
    return re.findall(r"<li>(.*?)</li>", m.group(0)) if m else []


def _line(kind, n):
    return f"{run_report.LEFT_OUT[kind]}: <b>{n}</b>"


def _looks(data, page):
    return data["docs"][0]["pages"][str(page)]["l"]


# the main agent's own zoom (the run's second render_region) is on page 2
BAD_BOXES = {"nan": [50.0, NAN, 200.0, 220.0],
             "infinity": [50.0, 60.0, float("inf"), 220.0],
             "null": None, "text": "50,60,200,220",
             "numbers as text": ["50", "60", "200", "220"],
             "three numbers": [50.0, 60.0, 200.0],
             "no area": [200.0, 60.0, 50.0, 220.0]}


@pytest.mark.parametrize("kind", sorted(BAD_BOXES))
def test_a_box_that_cannot_be_drawn_is_left_out_and_counted(tmp_path, kind):
    _result(_run(tmp_path), "render_region", 1, view=BAD_BOXES[kind])
    page = run_report.build_report(str(tmp_path))
    data = _viewer(page)
    assert _left(page) == [_line("box", 1)]
    zoom = [lk for lk in _looks(data, 2) if lk[1] == "zoom"]
    # listed, with no box; the box that was asked for is not put in its place
    assert len(zoom) == 1 and zoom[0][2] is None
    entry = data["docs"][0]["pages"]["2"]
    assert data["rf"][entry["rf"]] == (
        f"Not drawn here. {run_report.LEFT_OUT['box']}: 1.")
    # part 4 is still made, with the page's other boxes
    assert page.count("<figure") == 3 and 'class="bz"' in page
    assert "its view box in the record cannot be drawn" in page
    assert not re.search(r'="(nan|inf)', page)
    assert "could not be made" not in page


def test_a_whole_page_view_and_a_tile_holding_nan(tmp_path):
    def change(d):
        d["view"] = [0.0, 0.0, NAN, 792.0]
        d["tiles"][0]["view"] = [NAN, 0, 306, 396]
    _change(_run(tmp_path, pdf=False), "tool_end", "analyze_pdf_page", 0,
            change)
    page = run_report.build_report(str(tmp_path))
    looks = _looks(_viewer(page), 2)
    assert _left(page) == [_line("box", 2)]
    assert [lk[2] for lk in looks if lk[1] == "page"] == [None]
    assert sum(1 for lk in looks if lk[1] == "tile" and lk[2]) == 3
    assert page.count("<figure") == 3 and not re.search(r'="(nan|inf)', page)


@pytest.mark.parametrize("view", [[700.0, 900.0, 900.0, 1100.0],
                                  [500.0, 700.0, 900.0, 1100.0]])
def test_a_box_outside_its_page_is_not_drawn_or_moved(tmp_path, view):
    _result(_run(tmp_path), "render_region", 1, view=view)
    page = run_report.build_report(str(tmp_path))
    assert _left(page) == [_line("off_page", 1)]
    zoom = [lk for lk in _looks(_viewer(page), 2) if lk[1] == "zoom"]
    assert zoom[0][2] is None
    assert "its view box reaches outside the page" in page
    assert 'x="700"' not in page and 'x="500"' not in page


def test_a_box_a_point_past_the_page_edge_is_drawn(tmp_path):
    _result(_run(tmp_path), "analyze_pdf_page", 0,
            view=[0.0, 0.0, 612.9, 793.0])
    page = run_report.build_report(str(tmp_path))
    assert _left(page) == []
    assert [0.0, 0.0, 612.9, 793.0] in [lk[2] for lk in
                                        _looks(_viewer(page), 2)]


def test_without_the_pdf_a_far_box_cannot_be_told_and_is_not_counted(
        tmp_path):
    _result(_run(tmp_path, pdf=False), "render_region", 1,
            view=[700.0, 900.0, 900.0, 1100.0])
    page = run_report.build_report(str(tmp_path))
    assert _left(page) == []
    assert [700.0, 900.0, 900.0, 1100.0] in [
        lk[2] for lk in _looks(_viewer(page), 2)]


def _page_of_the_helpers_look(folder, value):
    def change(d):
        d.pop("page", None)
        d.pop("pdf_page", None)
        if value is not None:
            d["page"] = value
    _change(folder, "tool_end", "analyze_pdf_page", 1, change)
    _change(folder, "tool_start", "analyze_pdf_page", 1, change, "args")


def test_a_look_whose_page_is_not_a_number_is_put_on_no_page(tmp_path):
    _page_of_the_helpers_look(_run(tmp_path), "three")
    page = run_report.build_report(str(tmp_path))
    data = _viewer(page)
    assert _left(page) == [_line("no_page", 1)]
    # it was on page 3; it is not moved to the first page
    assert "l" not in data["docs"][0]["pages"]["0"]
    assert "l" not in data["docs"][0]["pages"]["3"]
    assert page.count("<figure") == 2


def test_a_look_that_names_no_page_is_on_the_tools_first_page(tmp_path):
    """As before this change, and no fault: the tool's own default is its
    first page, and a look at an image has no page to name."""
    _page_of_the_helpers_look(_run(tmp_path), None)
    page = run_report.build_report(str(tmp_path))
    assert _left(page) == []
    assert len(_looks(_viewer(page), 0)) == 1


def test_a_vision_record_with_no_page_is_counted(tmp_path):
    """``reads.json`` alone, one row whole and one with no page."""
    rows = [{"document": DOC, "page": 1, "view": "page"},
            {"document": DOC, "view": "page"},
            {"document": DOC, "page": 2, "view": "zoom",
             "region": [10.0, float("nan"), 50.0, 60.0]}]
    (tmp_path / "reads.json").write_text(json.dumps({"reads": rows}),
                                         encoding="utf-8")
    page = run_report.build_report(str(tmp_path))
    data = _viewer(page)
    assert _left(page) == [_line("box", 1), _line("no_page", 1)]
    assert _looks(data, 1)[0][2] is None and _looks(data, 2)[0][2] is None


@pytest.mark.parametrize("where", [40, -3])
def test_a_look_on_a_page_the_document_has_not(tmp_path, where):
    _result(_run(tmp_path), "render_region", 1, page=where)
    page = run_report.build_report(str(tmp_path))
    data = _viewer(page)
    assert _left(page) == [_line("past_end", 1)]
    assert data["docs"][0]["n"] == 6 and page.count("data-page=") == 6
    assert page.count("<figure") == 3 and f"p-1-{where}" not in page


def test_a_text_read_far_beyond_the_document(tmp_path):
    _result(_run(tmp_path), "read_document", 0, pages_returned="0-1,5000")
    page = run_report.build_report(str(tmp_path))
    assert _left(page) == [_line("past_end", 1)]
    assert page.count("data-page=") == 6


def test_a_cut_last_line_and_empty_records_are_counted(tmp_path):
    _run(tmp_path)
    log = tmp_path / "activity.jsonl"
    text = log.read_text(encoding="utf-8")
    log.write_text(text + "{}\n\n[]\n" + text.splitlines()[-1][:-40],
                   encoding="utf-8")
    page = run_report.build_report(str(tmp_path))
    assert _left(page) == [_line("bad_line", 3)]
    assert page.count("<figure") == 3 and _viewer(page)["docs"][0]["n"] == 6


def test_an_empty_activity_log_still_gives_a_report(tmp_path):
    (tmp_path / "activity.jsonl").write_text("", encoding="utf-8")
    page = run_report.build_report(str(tmp_path))
    assert _left(page) == [] and _viewer(page)["docs"] == []


def test_every_kind_has_its_own_line_above_the_contents(tmp_path):
    _run(tmp_path)
    _result(tmp_path, "render_region", 0, view=[NAN, 0, 1, 2])
    _result(tmp_path, "render_region", 1, view=[700.0, 900.0, 900.0, 1100.0])
    _result(tmp_path, "read_document", 0, pages_returned="0-1,77")
    _page_of_the_helpers_look(tmp_path, "three")
    with open(tmp_path / "activity.jsonl", "a", encoding="utf-8") as fh:
        fh.write('{"turn": 1, "event": "tool_st')
    page = run_report.build_report(str(tmp_path))
    assert _left(page) == [_line(kind, 1) for kind in run_report.LEFT_OUT]
    assert page.index("id='left-out'") < page.index("<nav class='toc'>")
    _viewer(page)


def test_a_pdf_that_cannot_be_drawn_or_is_gone_leaves_no_record_out(
        tmp_path):
    """As before this change: outlines only, and the reason in part 4."""
    _run(tmp_path)
    (tmp_path / DOC).write_bytes(b"%PDF-1.4 made-up, not a document")
    page = run_report.build_report(str(tmp_path))
    assert _left(page) == [] and page.count("<figure") == 3
    assert f"{DOC} could not be opened" in page
    (tmp_path / DOC).unlink()
    page = run_report.build_report(str(tmp_path))
    assert _left(page) == [] and "their PDF was not found" in page
    assert _viewer(page)["docs"][0]["pages"]["2"]["x"] == "nopdf"


def test_an_older_record_without_the_newer_fields_leaves_nothing_out(
        tmp_path):
    """Results with no ``view`` or ``pdf_page``, records with no turn, task
    id, usage or text: an older shape is no fault and is not counted."""
    _run(tmp_path, pdf=False)
    log = tmp_path / "activity.jsonl"
    drop = ("turn", "task_id", "helper", "usage", "text", "vision",
            "tool_run_id", "duration_s", "chars", "truncated")
    out = []
    for line in log.read_text(encoding="utf-8").splitlines():
        rec = {k: v for k, v in json.loads(line).items() if k not in drop}
        if rec.get("event") == "tool_end" and rec.get("name") in (
                "analyze_pdf_page", "render_region"):
            res = json.loads(rec["result"])
            res.pop("view", None), res.pop("pdf_page", None)
            rec["result"] = json.dumps(res)
        out.append(json.dumps(rec))
    log.write_text("\n".join(out) + "\n", encoding="utf-8")
    page = run_report.build_report(str(tmp_path))
    data = _viewer(page)
    assert _left(page) == [] and page.count("<figure") == 3
    # a zoom with no view back keeps the box that was asked for, as before
    assert [50.0, 60.0, 200.0, 220.0] in [lk[2] for lk in _looks(data, 2)]


def test_a_well_formed_run_says_nothing_is_left_out(tmp_path):
    page = run_report.build_report(str(_run(tmp_path)))
    assert "left-out" not in page and "Not drawn here" not in page
    assert all(not k.startswith("left-") for k in _viewer(page)["rf"])
