"""The owner's question list in the ingest run (BACKLOG RI-101 part 1) and
the list of posts for its post-name row (BACKLOG RI-170).

Offline. Every question, key, post and name here is invented: the owner's
list, its code map and the real list of posts are private and never read
here. The graph runs reuse the synthetic report and the scripted engine of
``test_graph``; unset, the graph snapshot test
(``test_graph_model_first.TestOffIsToday``) holds the run byte for byte.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from report_ingest import question_list as ql
from report_ingest.question_list import Answer, Post, QuestionListError
from report_ingest.tests.fake_engine import FakeEngine

HEADINGS = ["Output", "Category", "Data Type", "Allow Null",
            "Other Validation", "Prompt", "Key", "Answered by"]

#: A made-up list of three rows: a screening row, the post-name row and a
#: row code answers.
ROWS = [
    ("Study?", "Screening", "Boolean", None, None, "Is this a study?",
     "is_study", "Report text (model)"),
    ("Office", "Site", "String", True, None, "Which office is named?",
     "office_name", "Report text (model)"),
    ("Plans", "Paperwork", "Integer or list of integers", True, None,
     "Which pages are plans?", "plan_pages", "Code"),
]

CODE_MAP = {"readings": {"plan_pages": "pages_labelled:plan"},
            "post_name": "office_name"}

#: A made-up list of three posts, every name invented.
POSTS = [Post("Quellbrook", "Quellbrook Field Office"),
         Post("Varnhollow", "Varnhollow Survey Station"),
         Post("Tessidor", "Tessidor Liaison Office")]


def _workbook(path: Path, rows=ROWS, code_map=CODE_MAP) -> Path:
    import openpyxl

    book = openpyxl.Workbook()
    book.active.title = "Rules"
    book.active.append(["Rules (a heading)"])
    book.active.append(["Answer from the pages."])
    sheet = book.create_sheet("Questions")
    sheet.append(HEADINGS)
    for row in rows:
        sheet.append(list(row))
    book.save(path)
    if code_map is not None:
        path.with_name(path.stem + ql.CODE_MAP_SUFFIX).write_text(
            json.dumps(code_map), encoding="utf-8")
    return path


def _posts_json(path: Path, posts=POSTS) -> Path:
    path.write_text(json.dumps([
        {"name": p.name, "full_name": p.full_name, "region": "made-up"}
        for p in posts]), encoding="utf-8")
    return path


def _answer(value, pages=(1,)):
    return {"value": value, "status": "answered", "pdf_pages": list(pages),
            "note": ""}


# ---------------------------------------------------------------------------
# the list of posts (RI-170)
# ---------------------------------------------------------------------------

class TestLoadPosts:

    def test_json_list_and_object(self, tmp_path):
        assert ql.load_posts(_posts_json(tmp_path / "p.json")) == POSTS
        wrapped = tmp_path / "w.json"
        wrapped.write_text(json.dumps({"posts": [{"Name": "Quellbrook"}]}),
                           encoding="utf-8")
        assert ql.load_posts(wrapped) == [Post("Quellbrook", "")]

    def test_csv_by_its_headings(self, tmp_path):
        path = tmp_path / "p.csv"
        path.write_text("Region,Name,Full Name\n"
                        "made-up,Quellbrook,Quellbrook Field Office\n"
                        "made-up,,no name here\n"
                        "made-up,Tessidor,\n", encoding="utf-8")
        assert ql.load_posts(path) == [
            Post("Quellbrook", "Quellbrook Field Office"),
            Post("Tessidor", "")]

    @pytest.mark.parametrize("text,suffix", [
        ("not json", ".json"), ('{"other": []}', ".json"),
        ("city,country\nx,y\n", ".csv")])
    def test_a_file_it_cannot_use_is_refused(self, tmp_path, text, suffix):
        path = tmp_path / f"p{suffix}"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(QuestionListError):
            ql.load_posts(path)
        with pytest.raises(QuestionListError):
            ql.load_posts(tmp_path / "missing.json")


class TestMatchPost:

    def test_the_name_exactly_folded(self):
        found = ql.match_post("  QUELLBROOK. ", POSTS)
        assert (found.post, found.how) == (POSTS[0], "exact name")

    def test_then_the_full_name(self):
        found = ql.match_post("tessidor liaison-office", POSTS)
        assert (found.post, found.how) == (POSTS[2], "full name")

    def test_then_a_close_spelling(self):
        found = ql.match_post("Varnholow", POSTS)
        assert found.post == POSTS[1]
        assert found.how == "close spelling (similarity 0.95)"

    def test_no_post_close_enough_is_no_match(self):
        found = ql.match_post("Ostravel", POSTS)
        assert found.post is None and "at least 0.85 alike" in found.why

    def test_two_equally_close_match_none(self):
        twins = [Post("Brenmoor"), Post("Brenmoar"), POSTS[0]]
        found = ql.match_post("Brenmoer", twins)
        assert found.post is None and len(found.tied) == 2
        assert "equally close" in found.why

    def test_two_posts_of_one_name_match_none(self):
        found = ql.match_post("Quellbrook", POSTS + [Post("Quellbrook",
                                                         "Another Office")])
        assert found.post is None and len(found.tied) == 2


class TestApplyPosts:

    @pytest.fixture()
    def qlist(self, tmp_path):
        return ql.load_questions(_workbook(tmp_path / "q.xlsx"))

    def test_the_code_map_names_the_row(self, qlist):
        assert qlist.post_key == "office_name"

    def test_a_match_keeps_the_printed_name_beside_the_post(self, qlist):
        answers = {"office_name": Answer(value="Varnholow",
                                         status="answered")}
        assert ql.apply_posts(qlist, answers, POSTS) == []
        got = answers["office_name"]
        assert (got.value, got.printed) == ("Varnhollow", "Varnholow")
        assert got.matched_by.startswith("close spelling")
        assert "as printed: 'Varnholow'" in got.note
        assert "by close spelling" in got.note

    def test_no_match_is_an_open_note_and_the_value_stays(self, qlist):
        answers = {"office_name": Answer(value="Ostravel",
                                         status="answered")}
        lines = ql.apply_posts(qlist, answers, POSTS)
        assert len(lines) == 1 and "matches no one post" in lines[0]
        got = answers["office_name"]
        assert (got.value, got.printed, got.matched_by) == (
            "Ostravel", None, "")
        assert "unmatched" in got.note

    def test_a_row_not_answered_is_left_alone(self, qlist):
        answers = {"office_name": Answer(status="not said")}
        assert ql.apply_posts(qlist, answers, POSTS) == []
        assert answers["office_name"].note == ""

    @pytest.mark.parametrize("key", ["no_such_key", "plan_pages",
                                     "is_study"])
    def test_a_post_name_that_does_not_fit_the_list(self, tmp_path, key):
        with pytest.raises(QuestionListError, match="post_name"):
            ql.load_questions(_workbook(tmp_path / "q.xlsx", code_map={
                **CODE_MAP, "post_name": key}))

    def test_no_post_name_row_matches_nothing(self, tmp_path):
        qlist = ql.load_questions(_workbook(
            tmp_path / "q.xlsx", code_map={"readings": {}}))
        answers = {"office_name": Answer(value="Ostravel",
                                         status="answered")}
        assert qlist.post_key == ""
        assert ql.apply_posts(qlist, answers, POSTS) == []
        assert answers["office_name"].value == "Ostravel"


# ---------------------------------------------------------------------------
# a resumed run does not pay twice
# ---------------------------------------------------------------------------

def test_a_kept_reply_is_given_back_for_the_same_request(tmp_path):
    qlist = ql.load_questions(_workbook(tmp_path / "q.xlsx"))
    schema = ql.answer_schema(ql.screening_questions(qlist))
    messages = [{"role": "user", "content": [{"type": "text",
                                              "text": "made-up"}]}]
    path = tmp_path / "items" / ql.CALLS_NAME
    inner = FakeEngine([{"final": {"is_study": _answer(True)}}])
    first = ql._ReplayEngine(inner, path, resume=True)
    first.complete(messages, output_format=schema)
    assert inner.n_calls == 1 and path.is_file()
    again = ql._ReplayEngine(FakeEngine([]), path, resume=True)
    reply = again.complete(messages, output_format=schema)
    assert reply.parsed.is_study.value is True
    assert reply.usage.tokens_in == 0
    # Not resuming, or another request, calls the model.
    fresh = FakeEngine([{"final": {"is_study": _answer(False)}}])
    ql._ReplayEngine(fresh, path, resume=False).complete(
        messages, output_format=schema)
    assert fresh.n_calls == 1


# ---------------------------------------------------------------------------
# the graph (RI-101 part 1)
# ---------------------------------------------------------------------------

def _graph_run(folder: Path, extra_turns=(), env_extra=None, engine=None):
    pytest.importorskip("planlens.document.roles")
    from report_ingest import graph
    from report_ingest.tests import test_graph as tg
    from report_ingest.tests.narrative_fixtures import build_narrative_report

    folder.mkdir(parents=True, exist_ok=True)
    pdf = folder / "SYN.pdf"
    pdf.write_bytes(build_narrative_report().pdf)
    out = folder / "out"
    engine = engine or FakeEngine(tg.full_script() + list(extra_turns))
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("GEOTECH_INGEST_")
           and k not in (ql.LIST_ENV, ql.POSTS_ENV)}
    env.update(env_extra or {})
    with mock.patch.dict(os.environ, env, clear=True):
        record = graph.ingest_report(pdf, engine, out_dir=out,
                                     report_id="SYN", model_first="off")
    return record, out, engine


def _sheet_rows(path: Path):
    import openpyxl

    book = openpyxl.load_workbook(path, read_only=True)
    try:
        rows = list(book["Answers"].iter_rows(values_only=True))
        run = list(book["Run"].iter_rows(values_only=True))
    finally:
        book.close()
    heads = rows[0]
    return ({r[heads.index("Key")]: dict(zip(heads, r)) for r in rows[1:]},
            {r[0]: r[1] for r in run if r and len(r) > 1})


QL_TURNS = [{"final": {"is_study": _answer(True)}},
            {"final": {"office_name": _answer("Varnholow", pages=(2,))}}]


def test_the_graph_names_the_lists_own_setting():
    from report_ingest import graph

    assert graph.QUESTION_LIST_ENV == ql.LIST_ENV


def test_unset_the_graph_asks_nothing_and_writes_no_sheet(tmp_path):
    from report_ingest import graph
    from report_ingest.tests import test_graph as tg

    def refuse(*_a, **_k):
        raise AssertionError("the question list ran with its setting unset")

    with mock.patch.object(graph, "_question_list", refuse):
        record, out, engine = _graph_run(tmp_path)
    assert engine.n_calls == len(tg.full_script())
    assert not (out / ql.SHEET_NAME).exists()
    assert not (out / "items" / ql.CALLS_NAME).exists()
    assert not any(e.where.startswith("question_list") for e in record.qa)


def test_set_the_sheet_is_written_beside_the_record(tmp_path):
    base, _, base_engine = _graph_run(tmp_path / "unset")
    listed = _workbook(tmp_path / "q.xlsx")
    posts = _posts_json(tmp_path / "posts.json")
    record, out, engine = _graph_run(
        tmp_path / "set", QL_TURNS,
        {ql.LIST_ENV: str(listed), ql.POSTS_ENV: str(posts)})
    sheet = out / ql.SHEET_NAME
    assert sheet.is_file() and (out / "report.record.json").is_file()
    assert engine.n_calls == base_engine.n_calls + 2
    # The two calls are counted in the run's cost.
    assert record.document.model_calls == base.document.model_calls + 2
    # The narrative's own answers are left as they are.
    assert record.narrative == base.narrative
    rows, run = _sheet_rows(sheet)
    assert rows["is_study"]["Answer"] == "yes"
    office = rows["office_name"]
    assert office["Answer"] == "Varnhollow"
    assert "as printed: 'Varnholow'" in office["Note"]
    assert rows["plan_pages"]["From"] == "code"
    assert run["Question list"] == "q.xlsx"
    assert run["List of posts"] == "posts.json"
    assert not any(e.where.startswith("question_list") for e in record.qa)


def test_an_unmatched_post_is_an_open_qa_entry(tmp_path):
    listed = _workbook(tmp_path / "q.xlsx")
    posts = _posts_json(tmp_path / "posts.json")
    turns = [QL_TURNS[0],
             {"final": {"office_name": _answer("Ostravel")}}]
    record, out, _ = _graph_run(tmp_path / "set", turns, {
        ql.LIST_ENV: str(listed), ql.POSTS_ENV: str(posts)})
    entries = [e for e in record.qa if e.where == "question_list.office_name"]
    assert len(entries) == 1 and entries[0].rank == "open"
    rows, _ = _sheet_rows(out / ql.SHEET_NAME)
    assert rows["office_name"]["Answer"] == "Ostravel"


def test_without_posts_the_row_is_answered_as_printed(tmp_path):
    listed = _workbook(tmp_path / "q.xlsx")
    _, out, _ = _graph_run(tmp_path / "set", QL_TURNS,
                           {ql.LIST_ENV: str(listed)})
    rows, run = _sheet_rows(out / ql.SHEET_NAME)
    assert rows["office_name"]["Answer"] == "Varnholow"
    assert "as printed" not in (rows["office_name"]["Note"] or "")
    assert "List of posts" not in run


def test_a_list_it_cannot_read_is_a_qa_entry_not_a_failed_run(tmp_path):
    record, out, engine = _graph_run(tmp_path, env_extra={
        ql.LIST_ENV: str(tmp_path / "missing.xlsx")})
    assert (out / "report.record.json").is_file()
    assert not (out / ql.SHEET_NAME).exists()
    entries = [e for e in record.qa if e.where == "question_list"]
    assert len(entries) == 1 and entries[0].kind == "not_run"
    assert "could not be read" in entries[0].detail


def test_a_bad_posts_file_leaves_the_row_as_printed(tmp_path):
    listed = _workbook(tmp_path / "q.xlsx")
    bad = tmp_path / "posts.json"
    bad.write_text("not json", encoding="utf-8")
    record, out, _ = _graph_run(tmp_path / "set", QL_TURNS, {
        ql.LIST_ENV: str(listed), ql.POSTS_ENV: str(bad)})
    rows, _ = _sheet_rows(out / ql.SHEET_NAME)
    assert rows["office_name"]["Answer"] == "Varnholow"
    assert any(e.kind == "not_run" and "list of posts" in e.detail
               for e in record.qa)


def test_a_failing_call_is_a_qa_entry_not_a_failed_run(tmp_path):
    listed = _workbook(tmp_path / "q.xlsx")
    record, out, _ = _graph_run(
        tmp_path, [{"raise": RuntimeError("made-up outage")}],
        {ql.LIST_ENV: str(listed)})
    assert (out / "report.record.json").is_file()
    assert any(e.kind == "not_run" and "made-up outage" in e.detail
               for e in record.qa)


def test_a_resumed_run_reuses_the_kept_replies(tmp_path):
    from report_ingest.tests import test_graph as tg

    listed = _workbook(tmp_path / "q.xlsx")
    env = {ql.LIST_ENV: str(listed)}
    _graph_run(tmp_path, QL_TURNS, env)
    # The second run is given the graph's script again (whatever it reuses
    # from the folder) and nothing for the question list.
    record, out, engine = _graph_run(
        tmp_path, env_extra=env,
        engine=_Lenient(tg.full_script()))
    assert not engine.asked_with_list
    rows, _ = _sheet_rows(out / ql.SHEET_NAME)
    assert rows["office_name"]["Answer"] == "Varnholow"


class _Lenient(FakeEngine):
    """The graph's script, and a note of any question-list call."""

    asked_with_list = False

    def complete(self, messages, **kwargs):
        schema = kwargs.get("output_format")
        if getattr(schema, "__name__", "") in ("ScreeningAnswers",
                                               "QuestionAnswers"):
            self.asked_with_list = True
            raise AssertionError("a kept reply was asked for again")
        return super().complete(messages, **kwargs)
