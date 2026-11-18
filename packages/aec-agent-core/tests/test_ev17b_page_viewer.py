"""The run report's page viewer (BACKLOG EV-17 part b), and the turns of a
record with a transcript and no activity log.

The record is the made-up run of ``test_ev17_run_report`` (written by the
real writer), or hand-built files of the same shape; the PDFs are made in
the test.
"""

from __future__ import annotations

import base64
import html
import json
import re
import shutil
import subprocess

import pytest

from aec_agent_core import run_report
from test_ev17_run_report import (BRIEF, DOC, FINAL, NETWORK, QUESTION,
                                  _transcript, made_up_run)

VIEWER_DATA = re.compile(r'<script type="application/json" '
                         r'id="viewer-data">(.*?)</script>', re.S)


def _data(page: str) -> dict:
    m = VIEWER_DATA.search(page)
    assert m, "no viewer data in the page"
    return json.loads(m.group(1))


def _pdf(folder, n=6, name=DOC):
    fitz = pytest.importorskip("fitz")
    pdf = fitz.open()
    for i in range(n):
        page = pdf.new_page(width=612, height=792)
        page.insert_text((72, 72), f"made-up sheet {i + 1}")
        page.draw_rect(fitz.Rect(100, 200, 400, 500), width=1)
    pdf.save(str(folder / name))
    pdf.close()
    return fitz


def _long_side(fitz, b64: str) -> int:
    pix = fitz.Pixmap(base64.b64decode(b64))
    return max(pix.width, pix.height)


def test_every_box_names_its_call_step_tool_and_agent(tmp_path):
    made_up_run(tmp_path)
    data = _data(run_report.build_report(str(tmp_path)))
    (doc,) = data["docs"]
    assert doc["name"] == DOC and doc["n"] == 6 and doc["i"] == 1
    assert sorted(doc["pages"]) == [str(p) for p in range(6)]
    plan = doc["pages"]["2"]
    looks = {(n, kind, label) for n, kind, _b, label, _t, _s in plan["l"]}
    assert looks == {(3, "page", ""), (3, "tile", "r1c1"),
                     (3, "tile", "r1c2"), (3, "tile", "r2c1"),
                     (3, "tile", "r2c2"), (8, "zoom", "")}
    zoom = [lk for lk in plan["l"] if lk[1] == "zoom"][0]
    assert zoom[2] == [50, 60, 200, 220] and zoom[4] == "render_region"
    schedule = doc["pages"]["4"]["l"]
    assert [(lk[0], lk[1], lk[2]) for lk in schedule] == [
        (6, "zoom", [100, 200, 400, 500])]
    # how each page was taken in, with its calls
    assert doc["pages"]["0"]["m"] == {"text": [2], "contact": [4]}
    assert "m" not in doc["pages"]["5"] and "l" not in doc["pages"]["5"]
    # each call: who made it and in which step, what was asked, the start
    # of what came back
    calls = data["calls"]
    assert calls["3"]["tool"] == "analyze_pdf_page"
    assert (calls["3"]["who"], calls["3"]["step"]) == ("Main agent", 2)
    assert (calls["8"]["who"], calls["8"]["step"]) == ("Main agent", 4)
    assert (calls["6"]["who"], calls["6"]["step"]) == (
        "Helper 1 (general-purpose)", 1)
    assert calls["7"]["who"] == "Helper 1 (general-purpose)"
    assert '"bbox": [100, 200, 400, 500]' in calls["6"]["asked"]
    assert "made-up marks B1 and B2" in calls["6"]["back"]
    # a cut result: its start comes from the whole copy, its size whole
    assert calls["3"]["back_n"] > run_report.VIEW_BACK_CHARS
    assert len(calls["3"]["back"]) == run_report.VIEW_BACK_CHARS
    assert set(calls) == {"2", "3", "4", "6", "7", "8"}
    assert data["steps"] is True and data["multi"] is False


def test_the_viewer_is_in_the_same_file_and_reaches_nothing_else(tmp_path):
    made_up_run(tmp_path)
    page = run_report.build_report(str(tmp_path))
    for pattern in NETWORK:
        assert not pattern.search(page), pattern.pattern
    assert '<div id="viewer" hidden' in page
    for label in ("Previous page looked at", "Previous page", "Next page",
                  "Next page looked at", "Close"):
        assert f">{label}</button>" in page
    assert "getElementById('viewer-data')" in page
    # the grid's documents carry their index, the squares their page; a
    # drawing in part 4 offers the viewer (shown by the script)
    assert '<div class="doc" id="doc-1" data-i="1">' in page
    assert "<a class='vw-open' hidden href='#view-1-3'>" in page
    assert "Choose a square to open its page in the page viewer" in page
    assert page.index('<div id="viewer"') > page.index('<section id="record"')


def test_a_record_text_cannot_end_the_viewer_data(tmp_path):
    """A tool result holding ``</script>`` stays inside the JSON block."""
    trap = "made-up </script><script>window.x=1</script> & more"
    recs = [
        {"turn": 1, "ts": 1.0, "agent": "primary", "event": "turn_start",
         "prompt": QUESTION, "model": "made-up-model"},
        {"turn": 1, "ts": 2.0, "agent": "primary", "event": "model_end",
         "run_id": "m1", "n_tool_calls": 1},
        {"turn": 1, "ts": 3.0, "agent": "primary", "event": "tool_start",
         "name": "render_region", "run_id": "r1",
         "args": {"attachment_key": DOC, "page": 0,
                  "bbox": [10, 20, 110, 220]}},
        {"turn": 1, "ts": 4.0, "agent": "primary", "event": "tool_end",
         "name": "render_region", "run_id": "r1",
         "result": json.dumps({"page": 0, "view": [10, 20, 110, 220],
                               "analysis": trap})},
    ]
    (tmp_path / "activity.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    page = run_report.build_report(str(tmp_path))
    block = VIEWER_DATA.search(page).group(1)
    assert "<" not in block and ">" not in block and "&" not in block
    assert trap in json.loads(block)["calls"]["1"]["back"]
    assert page.count("<script") == 2         # the data and the script


def test_page_images_for_the_viewer_when_the_pdf_is_found(tmp_path):
    run_dir, docs = tmp_path / "run", tmp_path / "docs"
    run_dir.mkdir()
    docs.mkdir()
    made_up_run(run_dir)
    fitz = _pdf(docs)
    page = run_report.build_report(str(run_dir), documents=str(docs))
    data = _data(page)
    (doc,) = data["docs"]
    pages = doc["pages"]
    # the pages looked at: large for the viewer, small in part 4, once each
    for p in ("2", "3", "4"):
        assert _long_side(fitz, pages[p]["i"]) == run_report.VIEW_IMAGE_PX
        assert page.count(pages[p]["i"]) == 1
    assert page.count('<image href="data:image/jpeg;base64,') == 3
    # the other pages: small, for the viewer only
    for p in ("0", "1", "5"):
        assert _long_side(fitz, pages[p]["i"]) == run_report.PAGE_IMAGE_PX
    assert doc["r"] == [0, 0, 612, 792]
    assert all("r" not in e and "x" not in e for e in pages.values())
    assert "Pages looked at: 3; drawn at 900 pixels" in page
    assert "Other pages: 3; drawn at 420 pixels" in page


def test_caps_bound_the_images_and_say_what_was_left_out(tmp_path,
                                                         monkeypatch):
    run_dir, docs = tmp_path / "run", tmp_path / "docs"
    run_dir.mkdir()
    docs.mkdir()
    made_up_run(run_dir)
    _pdf(docs)
    monkeypatch.setattr(run_report, "MAX_VIEW_IMAGE_BYTES", 1)
    monkeypatch.setattr(run_report, "MAX_OTHER_IMAGE_BYTES", 1)
    page = run_report.build_report(str(run_dir), documents=str(docs))
    pages = _data(page)["docs"][0]["pages"]
    # one large image, then the small one from part 4 (no "i", no reason)
    assert "i" in pages["2"]
    assert all("i" not in pages[p] and "x" not in pages[p]
               for p in ("3", "4"))
    assert page.count('<image href="data:image/jpeg;base64,') == 3
    # other pages: the one taken in first drawn, then the cap
    assert "i" in pages["0"]
    assert pages["1"]["x"] == "cap" and pages["5"]["x"] == "cap"
    assert "only at 420: 2 (the large images stop" in page
    assert "first: 1; not drawn: 2 (their images stop" in page
    # a page with no image still lists its looks
    assert pages["4"]["l"][0][:2] == [6, "zoom"]

    monkeypatch.setattr(run_report, "MAX_PAGE_IMAGES", 1)
    page = run_report.build_report(str(run_dir), documents=str(docs))
    pages = _data(page)["docs"][0]["pages"]
    assert pages["3"]["x"] == "cap" and pages["4"]["x"] == "cap"
    assert page.count('<image href="data:image/jpeg;base64,') == 1
    assert "not drawn: 2 (at most 1 pages looked at are drawn" in page
    schedule = page[page.index('id="p-1-4"'):]
    schedule = schedule[:schedule.index("</figure>")]
    assert html.escape("outline at the PDF page's size") in schedule
    assert '<a href="#c6">#6</a> zoom' in schedule

    monkeypatch.setattr(run_report, "MAX_PAGE_IMAGES", 300)
    monkeypatch.setattr(run_report, "MAX_OTHER_IMAGE_BYTES", 10 ** 9)
    monkeypatch.setattr(run_report, "MAX_IMAGES_DRAWN", 2)
    pages = _data(run_report.build_report(
        str(run_dir), documents=str(docs)))["docs"][0]["pages"]
    # two images in all: page 3 large and small, page 4 small (from part 4)
    assert sum("i" in e for e in pages.values()) == 1
    assert "i" in pages["2"] and "x" not in pages["3"]
    assert [p for p, e in sorted(pages.items()) if e.get("x") == "cap"] \
        == ["0", "1", "4", "5"]


def test_without_a_pdf_the_viewer_draws_outlines_and_says_why(tmp_path):
    made_up_run(tmp_path)
    data = _data(run_report.build_report(str(tmp_path)))
    pages = data["docs"][0]["pages"]
    assert data["docs"][0]["r"] is None
    assert pages["2"]["r"] == [0, 0, 612, 792] and pages["2"]["rf"] == "view"
    assert pages["4"]["rf"] == "boxes"
    assert all(e.get("x") == "nopdf" for e in pages.values())
    assert "--documents" in data["why"]["nopdf"]


def test_the_viewer_from_the_vision_records_alone(tmp_path):
    (tmp_path / "reads.json").write_text(json.dumps({"reads": [
        {"document": DOC, "page": 1, "pdf_page": 2, "view": "zoom",
         "region": [10, 20, 30, 40], "tool": "render_region",
         "prompt": "made-up", "when": 1.0}]}), encoding="utf-8")
    page = run_report.build_report(str(tmp_path))
    data = _data(page)
    look = data["docs"][0]["pages"]["1"]["l"][0]
    assert look == [0, "zoom", [10, 20, 30, 40], "", "render_region",
                    "reads.json"]
    assert data["steps"] is False and data["calls"] == {}
    assert data["empty"].startswith("No look recorded")
    assert "could not be made" not in page


def test_a_viewer_that_cannot_be_made_is_one_line(tmp_path, monkeypatch):
    made_up_run(tmp_path)

    def broken(self):
        raise RuntimeError(f"made-up failure in {tmp_path / 'x.json'}")

    monkeypatch.setattr(run_report.Report, "_viewer_data", broken)
    page = run_report.build_report(str(tmp_path))
    assert "The page viewer could not be made from the record: " \
           "RuntimeError: made-up failure in" in page
    assert str(tmp_path) not in page and tmp_path.as_posix() not in page
    assert 'id="viewer"' not in page and "viewer-data" not in page
    assert "Traceback" not in page
    assert 'id="p-1-2"' in page                   # part 4 is still there


def test_the_viewer_script_parses(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed here")
    script = tmp_path / "viewer.js"
    script.write_text(run_report._SCRIPT + run_report._VIEWER_SCRIPT,
                      encoding="utf-8")
    out = subprocess.run([node, "--check", str(script)],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr


# -- the follow-up to part (a): a transcript and no activity log ----------

def test_a_transcript_without_its_activity_log_lists_each_answer(tmp_path):
    second = "Made-up second question about the made-up beam B2."
    _transcript(tmp_path, (QUESTION, FINAL), (second, "Made-up reply."))
    page = run_report.build_report(str(tmp_path))
    steps = page[page.index('<section id="steps">'):]
    steps = steps[:steps.index('<section id="taken">')]
    assert "No activity.jsonl: the steps cannot be shown." in steps
    one = steps[steps.index('id="turn-1"'):steps.index('id="turn-2"')]
    two = steps[steps.index('id="turn-2"'):]
    assert html.escape(QUESTION) in one and html.escape(FINAL) in one
    assert f"The answer as the user received it: {len(FINAL)} characters" \
        in one
    assert "<details open>" in one
    assert html.escape(second) in two and "Made-up reply." in two
    assert "Files shown with the answer: made_up_markup.pdf" in two


def test_a_suite_summary_without_its_activity_log_lists_each_answer(
        tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({
        "task": "made-up-task", "arm": "base", "question": QUESTION,
        "turns": [{"question": QUESTION, "answer": FINAL},
                  {"question": BRIEF}]}), encoding="utf-8")
    page = run_report.build_report(str(tmp_path))
    steps = page[page.index('<section id="steps">'):]
    steps = steps[:steps.index('<section id="taken">')]
    assert html.escape(FINAL) in steps and html.escape(BRIEF) in steps
    assert "The record holds no answer as the user received it for this " \
           "turn." in steps[steps.index('id="turn-2"'):]


def test_no_question_and_no_answer_is_one_plain_line(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"title": "Made-up"}),
                                        encoding="utf-8")
    page = run_report.build_report(str(tmp_path))
    assert "no other file of the record holds a question or an answer" \
        in page
    assert "could not be made" not in page
