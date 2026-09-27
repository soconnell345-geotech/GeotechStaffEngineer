"""The Document Review suite itself — offline, no model, no network.

The suite is only worth running if its checks accept a right answer and
reject a wrong one, its documents resolve, and its runner scores and resumes
the way it says. Those are pinned here.
"""

import json
import os

import pytest

pytest.importorskip("planlens.tools")
fitz = pytest.importorskip("fitz")

from funhouse_agent import review_flags  # noqa: E402
from funhouse_agent.review_eval import checks as C  # noqa: E402
from funhouse_agent.review_eval import documents as D  # noqa: E402
from funhouse_agent.review_eval.tasks import (  # noqa: E402
    CATEGORIES, DOC_TYPES, OPEN_TASKS, Task, load_tasks, select)

FILE_CHECKS = {"file_produced", "pdf_markups", "docx_contains", "tool_used"}


@pytest.fixture(autouse=True)
def _no_switches(monkeypatch):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def test_tasks_are_well_formed():
    ids = [t.id for t in OPEN_TASKS]
    assert len(ids) == len(set(ids))
    assert len(OPEN_TASKS) >= 25
    for t in OPEN_TASKS:
        assert t.category in CATEGORIES, t.id
        assert t.doc_type in DOC_TYPES, t.id
        assert t.truth, t.id
        for d in t.documents:
            assert d in D.DOCUMENTS, (t.id, d)
        for c in t.all_checks():
            assert c["type"] in C.CHECKS, (t.id, c)
    # varied on purpose: several document types and every kind of question
    assert len({t.doc_type for t in OPEN_TASKS}) >= 7
    assert {t.category for t in OPEN_TASKS} >= {"locate", "summarize",
                                                "count", "check", "compare",
                                                "markups", "produce"}
    # the synthetic fixtures stay the smallest part of the suite
    synthetic = [t for t in OPEN_TASKS
                 if any(d.startswith("fixture_") for d in t.documents)]
    assert len(synthetic) <= 3


@pytest.mark.parametrize("task", OPEN_TASKS, ids=lambda t: t.id)
def test_the_truth_passes_its_own_checks_and_silence_does_not(task):
    answer_checks = [c for c in task.all_checks()
                     if c["type"] not in FILE_CHECKS]
    for c in answer_checks:
        r = C.run_check(c, task.truth)
        assert r["passed"], (c, r["detail"])
    for c in task.checks:
        if c["type"] in FILE_CHECKS or c["type"] == "not_contains":
            continue
        assert not C.run_check(c, "")["passed"], c


def test_load_tasks_adds_a_private_set_and_select_narrows(tmp_path):
    extra = tmp_path / "blind.json"
    extra.write_text(json.dumps({"tasks": [{
        "id": "blind-1", "question": "q", "documents": ["x.pdf"],
        "category": "locate", "doc_type": "drawing_set", "split": "blind",
        "checks": [{"type": "contains_all", "terms": ["y"]}]}]}),
        encoding="utf-8")
    tasks = load_tasks([str(extra)])
    assert any(t.id == "blind-1" for t in tasks)
    assert [t.id for t in select(tasks, split="blind")] == ["blind-1"]
    assert all(t.id.startswith("meck-")
               for t in select(tasks, ids=["meck-"]))
    assert Task.from_dict(OPEN_TASKS[0].to_dict()).to_dict() == \
        OPEN_TASKS[0].to_dict()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def test_normalize_and_alternatives():
    assert C.normalize("2’6” – **Bold**") == "2'6\" - bold"
    assert C.term_found(["x", "2'-6"], C.normalize("a 2’-6” curb"))
    assert C.term_found({"re": r"\b3600\b"}, C.normalize("3600 psi"))
    assert not C.term_found({"re": r"\b3600\b"}, C.normalize("36000 psi"))


def test_guarded_terms_do_not_match_inside_bigger_numbers():
    from funhouse_agent.review_eval.tasks import _ft, _inch, _pct
    assert C.check_contains_all("max cross slope 5%", [_pct("5")])[0]
    assert not C.check_contains_all("max grade 15%", [_pct("5")])[0]
    assert C.check_contains_all("a 6-inch pipe", [_inch("6")])[0]
    assert not C.check_contains_all("a 16 in pipe", [_inch("6")])[0]
    assert C.check_contains_all("50' R/W", [_ft("50")])[0]
    assert not C.check_contains_all("150 ft", [_ft("50")])[0]


def test_set_match_scores_recall_and_precision():
    vocab = ["10.25A", "20.00A", "20.00B", "30.01"]
    ok, _ = C.check_set_match("Sheets 10.25A, 20.00A and 20.00B.", vocab,
                              ["10.25A", "20.00A", "20.00B"],
                              min_recall=1.0, min_precision=0.75)
    assert ok
    bad, detail = C.check_set_match("10.25A, 20.00A, 20.00B and 30.01", vocab,
                                    ["10.25A", "20.00A", "20.00B"],
                                    min_recall=1.0, min_precision=0.8)
    assert not bad and "30.01" in detail
    assert not C.check_set_match("sheet 120.00A", vocab, ["20.00A"])[0]


def test_cites_accepts_viewer_or_printed_pages():
    assert C.check_cites("see PDF page 29", pages=[29])[0]
    assert C.check_cites("(p. 29)", pages=[29])[0]
    assert C.check_cites("printed page 5-2", printed=["5-2"])[0]
    assert not C.check_cites("page 28", pages=[29], printed=["5-2"])[0]


def test_auto_checks_catch_giving_up_and_zero_based_citations():
    from funhouse_agent.review_eval.tasks import AUTO_CHECKS
    give_up, zero = AUTO_CHECKS
    assert not C.run_check(give_up, "The sheet is too blurry to read.")["passed"]
    assert not C.run_check(zero, "It is on page 0.")["passed"]
    assert C.run_check(zero, "It is on page 10.")["passed"]


def test_file_checks_read_the_produced_files(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_paragraph("Concrete 3600 psi; ramp 8.33%; surface S9.5B")
    memo = tmp_path / "memo.docx"
    d.save(memo)
    pdf = tmp_path / "sheet_marked.pdf"
    doc = fitz.open()
    page = doc.new_page()
    annot = page.add_text_annot((72, 72), "Confirm 8.33% applies over the ramp")
    annot.set_info(title="Review suite (AI draft)")
    annot.update()
    doc.save(pdf)
    files = [str(memo), str(pdf)]
    assert C.check_file_produced("", files, ext=".docx")[0]
    assert C.check_file_produced("", files, ext=".pdf", name_contains="marked")[0]
    assert not C.check_file_produced("", files, ext=".xlsx")[0]
    assert C.check_docx_contains("", files, terms=["3600", "8.33", "S9.5B"])[0]
    assert C.check_pdf_markups("", files, min=1, pages=[0],
                               text_contains="8.33")[0]
    assert not C.check_pdf_markups("", files, min=1, pages=[1])[0]


def test_a_broken_check_fails_rather_than_raising():
    r = C.run_check({"type": "no_such_check"}, "x")
    assert not r["passed"]
    r = C.run_check({"type": "contains_all"}, "x")      # missing 'terms'
    assert not r["passed"] and "check error" in r["detail"]
    s = C.score([r, {"passed": True, "info": True}])
    assert s == {"checks_passed": 0, "checks_total": 1, "passed": False}


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

def test_fixtures_and_sets_resolve():
    name, data = D.resolve("fixture_submittal")
    assert name == "submittal.pdf" and fitz.open(stream=data,
                                                 filetype="pdf").page_count == 11
    try:
        name, data = D.resolve("meck_set")
    except D.MissingDocument:
        pytest.skip("public documents not in this checkout")
    assert fitz.open(stream=data, filetype="pdf").page_count == 10


def test_a_missing_document_is_reported_not_raised_by_the_runner(tmp_path):
    from funhouse_agent.review_eval.runner import run_task
    task = Task(id="t", question="q", documents=["no_such_file.pdf"],
                category="locate", doc_type="drawing_set",
                checks=[{"type": "contains_all", "terms": ["x"]}])
    res = run_task(task, model=None, arm="baseline", arm_env={},
                   docs_dir=str(tmp_path), run_dir=str(tmp_path / "run"))
    assert res["skipped"] and "missing document" in res["error"]


def test_collect_public_docs_copies_what_it_finds(tmp_path):
    out = D.collect_public_docs(tmp_path / "docs")
    assert set(out["copied"]) | set(out["missing"]) == set(D.public_files())
    for name in out["copied"]:
        assert (tmp_path / "docs" / name).is_file()


# ---------------------------------------------------------------------------
# The runner, end to end, on a scripted model
# ---------------------------------------------------------------------------

def _scripted_model(answer):
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel)
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class Model(FakeMessagesListChatModel):
        calls: list = []

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            self.calls.append(len(messages))
            tool_msgs = [m for m in messages if getattr(m, "type", "") == "tool"]
            if not tool_msgs:
                msg = AIMessage(content="", tool_calls=[{
                    "name": "open_document",
                    "args": {"source": "review_set.pdf"}, "id": "c1"}])
            else:
                msg = AIMessage(content=answer)
            return ChatResult(generations=[ChatGeneration(message=msg)])

    return Model(responses=[AIMessage(content="x")], calls=[])


def test_runner_scores_both_arms_writes_results_and_resumes(tmp_path):
    from funhouse_agent.review_eval import score_review_suite
    task = next(t for t in OPEN_TASKS if t.id == "fixture-markups")
    model = _scripted_model(task.truth + " Cited on PDF page 2.")
    out = tmp_path / "suite"
    res = score_review_suite(model, ids=["fixture-markups"],
                             arms=("baseline", "lean"), out_dir=out,
                             verbose=False)
    for arm in ("baseline", "lean"):
        run = res["results"][arm]["fixture-markups"]
        assert run["error"] is None, run.get("traceback")
        assert run["score"]["passed"], run["checks"]
        assert run["model_calls"] >= 2
        assert any(c["name"] == "open_document" for c in run["tool_calls"])
        assert (out / "runs" / arm / "fixture-markups" / "run.json").is_file()
        assert (out / "runs" / arm / "fixture-markups" /
                "activity.jsonl").is_file()
    md = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "| baseline | 1/1 |" in md and "| lean | 1/1 |" in md
    assert "no task changed outcome" in md
    # A second run resumes: nothing is asked of the model again.
    n = len(model.calls)
    score_review_suite(model, ids=["fixture-markups"],
                       arms=("baseline", "lean"), out_dir=out, verbose=False)
    assert len(model.calls) == n


def test_runner_retries_a_failed_run(tmp_path):
    from funhouse_agent.review_eval import score_review_suite
    run_dir = tmp_path / "runs" / "baseline" / "fixture-markups"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({"error": "boom"}),
                                      encoding="utf-8")
    task = next(t for t in OPEN_TASKS if t.id == "fixture-markups")
    model = _scripted_model(task.truth)
    res = score_review_suite(model, ids=["fixture-markups"],
                             out_dir=tmp_path, verbose=False)
    assert res["results"]["baseline"]["fixture-markups"]["error"] is None


def test_a_retry_is_not_scored_on_the_failed_attempt_s_files(tmp_path):
    """Review finding 1."""
    from funhouse_agent.review_eval.runner import run_task
    run_dir = tmp_path / "run"
    (run_dir / "files").mkdir(parents=True)
    (run_dir / "files" / "old_marked.pdf").write_bytes(b"%PDF-stale")
    (run_dir / "run.json").write_text('{"error": "boom"}', encoding="utf-8")
    task = next(t for t in OPEN_TASKS if t.id == "fixture-markups")
    res = run_task(task, _scripted_model(task.truth), arm="baseline",
                   arm_env={}, docs_dir=None, run_dir=str(run_dir))
    assert not any("old_marked" in f for f in res["files"])
    assert (run_dir / "run.json").is_file()          # kept for the runner


def test_the_step_cap_is_a_result_and_is_not_paid_for_twice(tmp_path):
    """Review finding 6: a GraphRecursionError is the page's own behaviour."""
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel)
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from funhouse_agent.review_eval import score_review_suite

    class Loops(FakeMessagesListChatModel):
        calls: list = []

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            self.calls.append(1)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="", tool_calls=[{"name": "list_files",
                                         "args": {"path": "."},
                                         "id": f"c{len(self.calls)}"}]))])

    model = Loops(responses=[AIMessage(content="x")], calls=[])
    res = score_review_suite(model, ids=["fixture-markups"], out_dir=tmp_path,
                             recursion_limit=8, verbose=False)
    run = res["results"]["baseline"]["fixture-markups"]
    assert run["error"] is None and "GraphRecursionError" in run["outcome_error"]
    assert not run["score"]["passed"]
    n = len(model.calls)
    score_review_suite(model, ids=["fixture-markups"], out_dir=tmp_path,
                       recursion_limit=8, verbose=False)
    assert len(model.calls) == n
    assert "step caps" in res["results_md"]


def test_a_broken_document_is_recorded_not_raised(tmp_path, monkeypatch):
    """Review finding 9."""
    from funhouse_agent.review_eval.runner import run_task
    monkeypatch.setitem(D.DOCUMENTS, "broken", {"fixture": "no_such_fixture"})
    task = Task(id="t", question="q", documents=["broken"], category="locate",
                doc_type="drawing_set", checks=[])
    res = run_task(task, None, arm="baseline", arm_env={}, docs_dir=None,
                   run_dir=str(tmp_path / "r"))
    assert res["skipped"] and "no_such_fixture" in res["error"]


def test_checks_reject_the_false_passes_the_review_found():
    """Review finding 7, case by case."""
    from funhouse_agent.review_eval.tasks import (
        AUTO_CHECKS, _MECK_ALIASES, _ft, _inch)
    assert not C.check_contains_all("see paragraph 5-4 in the manual",
                                    [_inch("4")])[0]
    assert not C.check_contains_all("per 'Note 4' of the sheet", [_ft("4")])[0]
    assert not C.check_contains_all("14' of media", [_ft("4")])[0]
    assert not C.check_cites("step. 12 of the process", pages=[12])[0]
    assert not C.check_cites("30 mpg 30", pages=[30])[0]
    assert C.check_cites("see pages 3, 5, and 7", pages=[7])[0]
    assert C.check_set_match("sheets 10.25A and 20.00A/B",
                             ["10.25A", "20.00A", "20.00B"],
                             ["10.25A", "20.00A", "20.00B"],
                             aliases=_MECK_ALIASES)[0]
    give_up = AUTO_CHECKS[0]
    assert C.run_check(give_up, "I zoomed in at higher resolution.")["passed"]
    assert not C.run_check(give_up, "Please send a higher-resolution copy "
                                     "of the sheet.")["passed"]
    dup = next(t for t in OPEN_TASKS if t.id == "fixture-duplicate-page")
    wrong = "No page is duplicated; all 8 pages are distinct; page 3 is unique."
    assert not all(C.run_check(c, wrong)["passed"] for c in dup.all_checks())
    bio = next(t for t in OPEN_TASKS if t.id == "set-find-bioretention")
    assert not all(C.run_check(c, "Sheet 21.01 shows 14' of media")["passed"]
                   for c in bio.checks)


def test_dry_run_checks_documents_without_a_model(tmp_path):
    from funhouse_agent.review_eval import score_review_suite
    res = score_review_suite(None, ids=["fixture-"], out_dir=tmp_path,
                             dry_run=True, verbose=False)
    assert "fixture_submittal -> submittal.pdf" in res["results_md"]


def test_unknown_arm_is_refused(tmp_path):
    from funhouse_agent.review_eval import score_review_suite
    with pytest.raises(ValueError):
        score_review_suite(None, arms=("nope",), out_dir=tmp_path,
                           dry_run=True)


def test_summary_reports_what_an_arm_fixed_and_broke():
    from funhouse_agent.review_eval.runner import summarize
    tasks = [Task(id="a", question="q", documents=[], category="locate",
                  doc_type="drawing_set", checks=[]),
             Task(id="b", question="q", documents=[], category="check",
                  doc_type="calc_package", checks=[])]
    ok = {"score": {"passed": True, "checks_passed": 1, "checks_total": 1}}
    no = {"score": {"passed": False, "checks_passed": 0, "checks_total": 1},
          "checks": [{"label": "x", "passed": False, "detail": "missing: y"}]}
    runs = {"baseline": {"a": ok, "b": no}, "lean": {"a": no, "b": ok}}
    md = summarize(runs, ["baseline", "lean"], tasks,
                   {"model": "m", "versions": {}, "updated": "now"})
    assert "`lean` FIXES b" in md and "`lean` BREAKS a" in md
    assert "missing: y" in md
