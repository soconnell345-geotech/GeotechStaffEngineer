"""The offline run report (BACKLOG EV-17): one HTML file a reader opens to
see what a saved run did, step by step, what it took in and where it looked.

The record is written by the real writer (``ActivityLogger``'s callbacks, as
LangChain calls them), with made-up documents, questions and answers; the
older-shape and missing-file cases are hand-built files of the same shape.
"""

from __future__ import annotations

import html
import json
import re
import time
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from aec_agent_core import run_report
from aec_agent_core.activity_log import ActivityLogger
from aec_agent_core.deep import keep_work
from aec_agent_core.deep.vision_engine import side_call

DOC = "made_up_plan.pdf"
QUESTION = "Check the made-up beam schedule against the made-up plan."
BRIEF = ("Read the made-up schedule on PDF page 5 and list each beam mark "
         "with its size; say which marks you could not read.")
FINAL = "The schedule lists B1 and B2; the plan shows B1 only."

#: The calls of the made-up run, in the order the report must show them.
EXPECTED_CALLS = ["open_document", "read_document", "analyze_pdf_page",
                  "render_page_thumbnails", "task", "render_region",
                  "analyze_pdf_page", "render_region"]


def _reply(text: str = "", calls=(), tokens=(100, 10)) -> LLMResult:
    msg = AIMessage(content=text, tool_calls=[
        {"name": n, "args": a, "id": f"call_{i}"}
        for i, (n, a) in enumerate(calls)],
        usage_metadata={"input_tokens": tokens[0],
                        "output_tokens": tokens[1],
                        "total_tokens": sum(tokens)})
    return LLMResult(generations=[[ChatGeneration(message=msg)]])


class _Run:
    """Drives one ActivityLogger the way a turn does."""

    def __init__(self, folder, max_chars: int = 400):
        self.log = ActivityLogger(str(folder), turn=1, max_chars=max_chars)
        self.top = uuid.uuid4()
        self.log.on_chain_start({}, {}, run_id=self.top, parent_run_id=None)

    def model(self, parent, text="", calls=(), tokens=(100, 10),
              model="made-up-model"):
        rid = uuid.uuid4()
        self.log.on_chat_model_start({"kwargs": {"model": model}},
                                     [[HumanMessage(content="made-up")]],
                                     run_id=rid, parent_run_id=parent)
        self.log.on_llm_end(_reply(text, calls, tokens), run_id=rid,
                            parent_run_id=parent)

    def tool_start(self, parent, name, args):
        rid = uuid.uuid4()
        self.log.on_tool_start({"name": name}, json.dumps(args), run_id=rid,
                               parent_run_id=parent, inputs=args)
        return rid

    def tool_end(self, rid, parent, result):
        out = result if isinstance(result, str) else json.dumps(result)
        self.log.on_tool_end(out, run_id=rid, parent_run_id=parent)

    def tool(self, parent, name, args, result, looks=()):
        rid = self.tool_start(parent, name, args)
        for label, text in looks:
            with side_call(**label):
                self.model(rid, text, tokens=(50, 5), model="made-up-vision")
        self.tool_end(rid, parent, result)
        return rid


def _tiles(page_w=612.0, page_h=792.0):
    half_w, half_h = page_w / 2, page_h / 2
    return [{"tile": f"r{r + 1}c{c + 1}",
             "view": [c * half_w, r * half_h, (c + 1) * half_w,
                      (r + 1) * half_h],
             "analysis": "made-up tile reading " + "x" * 300}
            for r in range(2) for c in range(2)]


def made_up_run(folder, *, helper_id_on_task_call: bool = True,
                max_chars: int = 400, question: str = QUESTION) -> None:
    """One turn: the main agent opens, reads, looks at and makes a contact
    sheet of a six-page made-up document, starts one helper (a zoom and a
    whole-page look), zooms itself and answers."""
    run = _Run(folder, max_chars=max_chars)
    log, top = run.log, run.top
    log.turn_start(prompt=question, model="made-up-model")
    run.model(top, "I will open the document.",
              [("open_document", {"source": DOC})], tokens=(1000, 20))
    run.tool(top, "open_document", {"source": DOC},
             {"handle": "h1", "name": DOC, "n_pages": 6})
    run.model(top, "Reading the notes and looking at the plan.",
              [("read_document", {"handle": "h1"}),
               ("analyze_pdf_page", {"attachment_key": DOC, "page": 2}),
               ("render_page_thumbnails", {"handle": "h1"})],
              tokens=(2000, 40))
    run.tool(top, "read_document", {"handle": "h1"},
             {"handle": "h1", "pages_returned": "0-1",
              "text": "=== page 0\nmade-up general notes"})
    run.tool(top, "analyze_pdf_page",
             {"attachment_key": DOC, "page": 2, "tiles": "2x2"},
             {"page": 2, "pdf_page": 3, "view": [0.0, 0.0, 612.0, 792.0],
              "analysis": "made-up layout", "tiles": _tiles()},
             looks=[({"kind": "overview", "page": 2}, "made-up overview"),
                    ({"kind": "tile", "page": 2, "tile": "r1c1"},
                     "made-up tile")])
    run.tool(top, "render_page_thumbnails", {"handle": "h1"},
             {"handle": "h1", "sheets": [{"pages": "0-3", "n_pages": 4}]})
    task_args = {"description": BRIEF, "subagent_type": "general-purpose"}
    run.model(top, "A helper reads the schedule page.",
              [("task", task_args)], tokens=(3000, 60))
    helper = keep_work.HelperRecord("call_task_1", "general-purpose", BRIEF)
    if helper_id_on_task_call:
        token = keep_work._HELPER.set(helper)
        task = run.tool_start(top, "task", task_args)
    else:
        task = run.tool_start(top, "task", task_args)
        token = keep_work._HELPER.set(helper)
    try:
        run.model(task, "Zooming on the schedule.",
                  [("render_region", {}), ("analyze_pdf_page", {})],
                  tokens=(500, 15))
        run.tool(task, "render_region",
                 {"attachment_key": DOC, "page": 4,
                  "bbox": [100, 200, 400, 500]},
                 {"page": 4, "pdf_page": 5,
                  "view": [100.0, 200.0, 400.0, 500.0],
                  "analysis": "made-up marks B1 and B2"},
                 looks=[({"kind": "region", "page": 4}, "B1, B2")])
        run.tool(task, "analyze_pdf_page", {"attachment_key": DOC, "page": 3},
                 {"page": 3, "pdf_page": 4, "view": [0.0, 0.0, 612.0, 792.0],
                  "analysis": "made-up elevation"})
        run.model(task, "Beam marks B1 and B2.", tokens=(600, 12))
        run.tool_end(task, top, "Beam marks B1 and B2 on PDF page 5.")
    finally:
        keep_work._HELPER.reset(token)
    run.model(top, "Zooming on the plan's beam.",
              [("render_region", {})], tokens=(4000, 30))
    run.tool(top, "render_region",
             {"attachment_key": DOC, "page": 2, "bbox": [50, 60, 200, 220]},
             {"page": 2, "pdf_page": 3, "view": [50.0, 60.0, 200.0, 220.0],
              "analysis": "made-up beam B1"})
    run.model(top, FINAL, tokens=(5000, 25))
    log.turn_end(turn_tokens=17_000, answer_chars=len(FINAL))
    log.turn_files([{"path": "files/made_up_notes.md", "bytes": 120,
                     "how": "new"}])


def _counts(page: str, doc: str = DOC) -> dict:
    m = re.search(r'<p class="counts" data-doc="' + re.escape(doc)
                  + r'">(.*?)</p>', page)
    assert m, "no page counts for the document"
    return {k: int(v) for k, v in re.findall(
        r'<span data-kind="(\w+)">[^<]*: <b>(\d+)</b>', m.group(1))}


def _total(page: str, key: str) -> str:
    m = re.search(r'<tr data-total="' + key + r'"><th>[^<]*</th>'
                  r'<td class="r">([^<]*)</td>', page)
    assert m, f"no total {key}"
    return m.group(1)


NETWORK = [re.compile(p, re.I) for p in (
    r"""(?:src|href|action)\s*=\s*["']?\s*(?:https?:|//|ftp:)""",
    r"@import", r"url\(\s*['\"]?\s*(?:https?:|//)", r"<link\b",
    r"<script[^>]+src", r"<iframe", r"\bfetch\(", r"XMLHttpRequest")]


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("helper_id_on_task_call", [True, False])
def test_every_call_in_order_and_the_helper_under_its_brief(
        tmp_path, helper_id_on_task_call):
    made_up_run(tmp_path, helper_id_on_task_call=helper_id_on_task_call)
    page = run_report.build_report(str(tmp_path))
    calls = re.findall(r'<div class="call" id="c(\d+)"><div class="ch">'
                       r'<span class="no">#\d+</span> <b>(\w+)</b>', page)
    assert [name for _n, name in calls] == EXPECTED_CALLS
    assert [int(n) for n, _name in calls] == list(range(1, 9))
    # the helper's block holds its brief, then its two calls, before what
    # it handed back; the main agent's last zoom comes after
    block = page.index('<details class="helper" open>')
    assert page.index(BRIEF) > block      # the brief is shown once, there
    order = [block, page.index(BRIEF), page.index('id="c6"'),
             page.index('id="c7"'),
             page.index("What the helper handed back"), page.index('id="c8"')]
    assert order == sorted(order)
    assert "Helper 1 (general-purpose)" in page
    # the model's own words, the main agent's and the helper's
    for words in ("I will open the document.", "Zooming on the schedule.",
                  FINAL):
        assert words in page


def test_page_grid_counts_and_totals(tmp_path):
    made_up_run(tmp_path)
    page = run_report.build_report(str(tmp_path))
    assert _counts(page) == {"text": 2, "whole": 1, "tiles": 1, "zoom": 2,
                             "contact": 4, "never": 1}
    assert _total(page, "tool_calls") == "8"
    assert _total(page, "helpers") == "1"
    assert _total(page, "model_calls") == "7"
    assert _total(page, "vision_calls") == "3"
    assert _total(page, "tokens_in") == f"{1000 + 2000 + 3000 + 500 + 600 + 4000 + 5000 + 150:,}"
    assert _total(page, "tokens_out") == f"{20 + 40 + 60 + 15 + 12 + 30 + 25 + 15:,}"
    # the tile boxes came from the whole copy the log kept of a cut result
    assert "whole copy from activity_full/" in page
    assert "files/made_up_notes.md" in page
    assert QUESTION in page


def test_boxes_numbered_as_the_steps(tmp_path):
    made_up_run(tmp_path)
    page = run_report.build_report(str(tmp_path))
    plan = page[page.index('id="p-1-2"'):]
    plan = plan[:plan.index("</figure>")]
    assert plan.count('<rect class="bt"') == 4         # the four tiles
    assert plan.count('<rect class="bw"') == 1         # the whole page
    assert '<rect class="bz" x="50" y="60" width="150" height="160"' in plan
    assert '<a href="#c8">' in plan and '<a href="#c3">' in plan
    schedule = page[page.index('id="p-1-4"'):]
    assert '<rect class="bz" x="100" y="200" width="300" height="300"' \
        in schedule[:schedule.index("</figure>")]
    # part 2 links each look to its drawing
    assert 'href="#p-1-4"' in page
    # no PDF anywhere: outlines, said once
    assert page.count("Drawn as outlines only") == 1


def test_self_contained_and_says_what_it_holds(tmp_path):
    made_up_run(tmp_path)
    page = run_report.build_report(str(tmp_path))
    for pattern in NETWORK:
        assert not pattern.search(page), pattern.pattern
    assert page.index(html.escape(run_report.CONTENT_NOTE)) < page.index("<nav")
    assert page.startswith("<!doctype html>")
    assert "prefers-color-scheme:dark" in page and "@media print" in page


def test_pages_drawn_behind_the_boxes_when_the_pdf_is_found(tmp_path):
    fitz = pytest.importorskip("fitz")
    run_dir, docs = tmp_path / "run", tmp_path / "docs"
    run_dir.mkdir()
    docs.mkdir()
    made_up_run(run_dir)
    pdf = fitz.open()
    for i in range(6):
        pdf.new_page(width=612, height=792).insert_text(
            (72, 72), f"made-up sheet {i + 1}")
    pdf.save(str(docs / DOC))
    pdf.close()
    page = run_report.build_report(str(run_dir), documents=str(docs))
    assert page.count("data:image/jpeg;base64,") == 3  # pages 3, 4 and 5
    assert "Drawn as outlines only" not in page
    assert "drawn from the PDF" in page


def test_a_missing_activity_log_is_named_not_a_traceback(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({
        "task": "made-up-task", "arm": "base", "question": QUESTION,
        "files": ["files/made_up_memo.md"], "seconds": 12.5,
        "model_calls": 3}), encoding="utf-8")
    page = run_report.build_report(str(tmp_path))
    assert "This record has no activity.jsonl" in page
    assert re.search(r"<td>activity\.jsonl</td><td class=warn>not found", page)
    assert QUESTION in page and "files/made_up_memo.md" in page
    assert "could not be made" not in page


def test_an_older_record_still_reads(tmp_path):
    """No task_id, no usage, no vision labels, a result cut short with no
    whole copy beside it, and an unreadable line."""
    recs = [
        {"turn": 1, "ts": 1.0, "agent": "primary", "event": "turn_start",
         "prompt": QUESTION, "model": "made-up-model"},
        {"turn": 1, "ts": 2.0, "agent": "primary", "event": "model_end",
         "run_id": "m1", "n_tool_calls": 1},
        {"turn": 1, "ts": 3.0, "agent": "primary", "event": "tool_start",
         "name": "task", "run_id": "t1", "subagent": "general-purpose",
         "args": {"description": BRIEF, "subagent_type": "general-purpose"}},
        {"turn": 1, "ts": 4.0, "agent": "general-purpose",
         "event": "model_end", "run_id": "m2", "n_tool_calls": 1},
        {"turn": 1, "ts": 5.0, "agent": "general-purpose",
         "event": "tool_start", "name": "render_region", "run_id": "r1",
         "args": {"attachment_key": DOC, "page": 1}},
        {"turn": 1, "ts": 6.0, "agent": "general-purpose",
         "event": "tool_end", "name": "render_region", "run_id": "r1",
         "result": '{"page": 1, "view": [10, 20, 110, 220], "analysis": "mad',
         "truncated": True, "chars": 90000},
        {"turn": 1, "ts": 7.0, "agent": "primary", "event": "tool_end",
         "name": "task", "run_id": "t1", "result": "made-up helper answer"},
        {"turn": 1, "ts": 9.0, "agent": "primary", "event": "turn_end",
         "duration_s": 8.0},
    ]
    lines = [json.dumps(r) for r in recs]
    lines.insert(3, "{not json")
    (tmp_path / "activity.jsonl").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    page = run_report.build_report(str(tmp_path))
    assert "1 unreadable lines skipped" in page
    assert "Helper records carry no task_id" in page
    assert "Model calls carry no usage" in page
    assert "could not be made" not in page
    calls = re.findall(r'id="c\d+"><div class="ch"><span class="no">#\d+'
                       r'</span> <b>(\w+)</b>', page)
    assert calls == ["task", "render_region"]
    assert page.index(BRIEF) < page.index('id="c2"')
    assert "cut in the record at" in page
    # the zoom's box, found in the cut text
    assert '<rect class="bz" x="10" y="20" width="100" height="200"' in page
    assert _total(page, "tokens_in") == "not recorded"


def test_suite_run_folder_with_its_vision_records(tmp_path):
    """A review suite's task folder: run.json holds the whole question (the
    activity log keeps 500 characters) and files/.scratch the vision tools'
    records, which add a look the log does not hold."""
    long_q = QUESTION + " " + "Made-up detail. " * 60
    made_up_run(tmp_path)
    (tmp_path / "run.json").write_text(json.dumps({
        "task": "made-up-task", "arm": "base", "question": long_q,
        "turns": [{"question": long_q, "answer": FINAL}],
        "files": ["files/made_up_notes.md"]}), encoding="utf-8")
    scratch = tmp_path / "files" / ".scratch"
    scratch.mkdir(parents=True)
    (scratch / "reads.json").write_text(json.dumps({"reads": [
        {"document": DOC, "page": 5, "pdf_page": 6, "view": "page",
         "tool": "analyze_pdf_page", "prompt": "made-up", "when": 1.0},
        {"document": DOC, "page": 2, "pdf_page": 3, "view": "page+tiles 2x2",
         "tool": "analyze_pdf_page", "prompt": "made-up", "when": 1.0}]}),
        encoding="utf-8")
    page = run_report.build_report(str(tmp_path))
    assert "a review suite run (task made-up-task, arm base)" in page
    assert long_q.strip() in page
    assert _counts(page)["whole"] == 2 and _counts(page)["never"] == 0
    assert "from reads.json" in page


def _transcript(folder, *pairs) -> None:
    """``transcript.jsonl`` in the shape the pages' writer gives it: one
    line per entry, as ``append_transcript`` (aec_document_review/core.py)
    appends it, with the entries the page makes: ``{"role": "user",
    "text": prompt}`` (app.py) and ``{"role": "assistant", "text": final,
    "artifacts": [...]}`` (turn_jobs.py, artifacts relative to files/)."""
    with open(folder / "transcript.jsonl", "a", encoding="utf-8") as fh:
        for question, answer in pairs:
            fh.write(json.dumps({"role": "user", "text": question}) + "\n")
            fh.write(json.dumps({"role": "assistant", "text": answer,
                                 "artifacts": ["made_up_markup.pdf"]})
                     + "\n")


def test_page_conversation_folder(tmp_path):
    long_q = QUESTION + " " + "Made-up detail. " * 60
    made_up_run(tmp_path, question=long_q)
    (tmp_path / "meta.json").write_text(json.dumps({
        "title": "Made-up review", "turn_count": 1}), encoding="utf-8")
    _transcript(tmp_path, (long_q, FINAL))
    page = run_report.build_report(str(tmp_path))
    assert "a page conversation: Made-up review" in page
    assert "made_up_markup.pdf" in page
    assert f"The answer as the user received it: {len(FINAL)} characters" \
        in page
    assert f"<pre>{html.escape(FINAL)}</pre>" in page
    # the whole question from the transcript, not the log's first 500
    assert long_q in page
    assert "the activity log keeps the first 500 characters" not in page
    assert "holds no answer as the user received it" not in page


def test_a_turn_with_no_answer_says_so(tmp_path):
    made_up_run(tmp_path)
    page = run_report.build_report(str(tmp_path))
    assert "The record holds no answer as the user received it for this " \
           "turn." in page


def test_a_transcript_without_its_activity_log_shows_the_questions(tmp_path):
    created = 1_700_000_000.25
    (tmp_path / "meta.json").write_text(json.dumps({
        "title": "Made-up review", "created": created, "updated": created,
        "turn_count": 2}), encoding="utf-8")
    second = "Made-up second question about the made-up beam B2."
    _transcript(tmp_path, (QUESTION, FINAL), (second, "Made-up reply."))
    page = run_report.build_report(str(tmp_path))
    asked = re.search(r"<tr><th>Asked</th><td>(.*?)</td></tr>", page)
    assert asked, "no Asked row"
    assert f"Turn 1: <span class='said'>{html.escape(QUESTION)}</span>" \
        in asked.group(1)
    assert f"Turn 2: <span class='said'>{html.escape(second)}</span>" \
        in asked.group(1)
    # "When" is a date and time, not meta.json's number of seconds
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(created))
    assert f"<tr><th>When</th><td>{when}</td></tr>" in page
    assert "1700000000" not in page


def test_the_grid_says_no_look_recorded_without_the_activity_log(tmp_path):
    (tmp_path / "reads.json").write_text(json.dumps({"reads": [
        {"document": DOC, "page": 3, "pdf_page": 4, "view": "page",
         "tool": "analyze_pdf_page", "prompt": "made-up", "when": 1.0}]}),
        encoding="utf-8")
    page = run_report.build_report(str(tmp_path))
    assert _counts(page) == {"text": 0, "whole": 1, "tiles": 0, "zoom": 0,
                             "contact": 0, "never": 3}
    assert "no look recorded: <b>3</b>" in page
    assert 'title="PDF page 1: no look recorded"' in page
    assert "never opened" not in page
    # with the log, nothing changes
    made_up_run(tmp_path)
    page = run_report.build_report(str(tmp_path))
    assert "never opened" in page and "no look recorded" not in page


def test_command_line_writes_only_where_asked(tmp_path, capsys):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    made_up_run(run_dir)
    assert run_report.main([str(run_dir)]) == 0
    beside = run_dir / run_report.REPORT_FILE
    assert beside.is_file()
    beside.unlink()
    other = tmp_path / "elsewhere.html"
    assert run_report.main([str(run_dir), "-o", str(other)]) == 0
    assert other.is_file() and not beside.exists()
    out = capsys.readouterr().out
    assert run_report.CONTENT_NOTE in out
    assert run_report.main([str(tmp_path / "missing")]) == 2
    err = capsys.readouterr().err
    assert "No such run folder" in err and "Traceback" not in err
