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
    CATEGORIES, DOC_TYPES, OPEN_TASKS, PAGES, Task, load_tasks, select)

FILE_CHECKS = {"file_produced", "pdf_markups", "markups_on_targets",
               "markups_point_at",
               "docx_contains", "tool_used", "pages_covered"}


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
    # the synthetic fixtures stay the smallest part of the suite (four since
    # 5.32: the two drawn-lettering tag sets measure markup placement and
    # whole-set coverage, which no public document in the suite can; three
    # more since plan W4 on one synthetic report, because whether EVERY data
    # page of a report was read can only be scored on a report whose every
    # page is known by construction)
    synthetic = [t for t in OPEN_TASKS
                 if any(d.startswith("fixture_") for d in t.documents)]
    assert len(synthetic) <= 7
    assert len(synthetic) < len(OPEN_TASKS) / 4
    for t in OPEN_TASKS:
        assert t.page in PAGES, t.id


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


def test_run_json_says_what_the_run_ran_on(tmp_path, monkeypatch):
    """Foundry brief 4: run.json carried neither the commit the wheel was
    built from nor the vision profile the probe measured. Both are recorded
    now, never by a new probe."""
    from funhouse_agent import vision_probe
    from funhouse_agent.review_eval import runner
    from funhouse_agent.review_eval.runner import run_task
    task = next(t for t in OPEN_TASKS if t.id == "fixture-markups")
    model = _scripted_model(task.truth)
    runner._commit.cache_clear()
    monkeypatch.setenv("GEOTECH_APP_COMMIT", "a4ef417")
    res = run_task(task, model, arm="baseline", arm_env={}, docs_dir=None,
                   run_dir=str(tmp_path / "run"))
    assert res["commits"]["app"] == "a4ef417"
    assert "planlens" in res["commits"]
    assert res["versions"]["geotech-staff-engineer"]
    assert res["vision_profile"] == {"probe": "off"}       # the test's switch
    # with the probe on, the profile measured for the run's model is copied
    monkeypatch.setenv(vision_probe.PROBE_ENV, "1")
    prof = vision_probe.VisionProfile(answered_by="gpt-5.4", general="openai-high",
                                      detailed="openai-original",
                                      source="probe", max_edge=2048)
    monkeypatch.setitem(vision_probe._CACHE, vision_probe._cache_key(model),
                        prof)
    got = runner._vision_profile(model)
    assert got["max_edge"] == 2048 and "at most 2048 px" in got["summary"]
    runner._commit.cache_clear()


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


# ---------------------------------------------------------------------------
# Review of the suite's checks, 2026-09-28: wrong answers that passed and
# right answers that failed
# ---------------------------------------------------------------------------

def _passes(task_id, answer):
    task = next(t for t in OPEN_TASKS if t.id == task_id)
    return all(C.run_check(c, answer)["passed"] for c in task.all_checks())


_APPENDICES_RIGHT = [
    "The manual has 14 appendices:\n- **Appendix A** - References\n"
    "- **Appendix B** - Airfield/Heliport Design Analysis Outline\n"
    "- **Appendix C** - Recommended Contract Drawing Outline\n"
    "- **Appendix D** - Waiver Processing Procedures\n"
    "- **Appendix E** - Flexural Strength and Modulus of Bituminous Concrete\n"
    "- **Appendix F** - Curves for Effective Strain Repetitions\n"
    "- **Appendix G** - Preparation of Bituminous Cylindrical Specimens\n"
    "- **Appendix H** - Dynamic Modulus of Bituminous Mixtures\n"
    "- **Appendix I** - Estimating the Modulus of Elasticity\n"
    "- **Appendix J** - Modulus of Unbound Granular Base\n"
    "- **Appendix K** - Fatigue of Stabilized Soils\n"
    "- **Appendix L** - Resilient Modulus of Subgrade Material\n"
    "- **Appendix M** - Fatigue Life of Bituminous Concrete\n"
    "- **Appendix N** - Resilient Modulus of Granular Base Material",
    "| Appendix | Covers | PDF page |\n|---|---|---|\n| A | References | 431 |"
    "\n| B | Design analysis outline | 439 |\n| C | Contract drawing outline "
    "| 446 |\n| D | Waivers | 452 |\n| E | Flexural strength | 457 |\n| F | "
    "Effective strain repetitions | 460 |\n| G | Bituminous cylindrical "
    "specimens | 482 |\n| H | Dynamic modulus | 484 |\n| I | Estimating the "
    "modulus of elasticity | 487 |\n| J | Unbound granular base | 491 |\n| K "
    "| Stabilized soils | 495 |\n| L | Resilient modulus of subgrade "
    "material | 500 |\n| M | Fatigue life | 522 |\n| N | Resilient modulus of "
    "granular base material | 528 |",
    "Appendices: A. References; B. Design analysis outline; C. Contract "
    "drawings; D. Waivers; E. Flexural strength; F. Strain repetitions; G. "
    "Cylindrical specimens; H. Dynamic modulus; I. Estimating the modulus; J. "
    "Unbound base; K. Stabilized soils; L. Subgrade material; M. Fatigue "
    "life; N. Resilient modulus of granular base.",
    "References (Appendix A), design analysis outline (Appendix B), contract "
    "drawings (Appendix C), waivers (Appendix D), flexural strength (Appendix "
    "E), strain repetitions (Appendix F), cylindrical specimens (Appendix G), "
    "dynamic modulus (Appendix H), estimating the modulus (Appendix I), "
    "unbound materials (Appendix J), stabilized soils (Appendix K), subgrade "
    "material (Appendix L), fatigue life (Appendix M), resilient modulus of "
    "granular base (Appendix N).",
]
_APPENDICES_WRONG = [
    # the letters matched to the wrong topics (shifted by one)
    "Appendix A - Design analysis outline\nAppendix B - Contract drawing "
    "outline\nAppendix C - Waivers\nAppendix D - Flexural strength\nAppendix "
    "E - Strain repetitions\nAppendix F - Cylindrical specimens\nAppendix G - "
    "Dynamic modulus\nAppendix H - Estimating the modulus\nAppendix I - "
    "Unbound granular base\nAppendix J - Stabilized soils\nAppendix K - "
    "Subgrade material\nAppendix L - Fatigue life\nAppendix M - Resilient "
    "modulus of granular base\nAppendix N - References",
    # "I could not find the appendices", then the topics without letters
    "I could not find the list of appendices in the text I read. From the "
    "chapter references they appear to cover references, a design analysis "
    "outline, contract drawings, waivers, flexural strength, strain "
    "repetitions, cylindrical specimens, dynamic modulus, estimating the "
    "modulus, unbound materials, stabilized soils, subgrade material, fatigue "
    "life and the resilient modulus of granular base.",
    # half of them, the rest as bare topics
    "Appendix A - References; Appendix B - Design analysis outline; Appendix "
    "C - Contract drawings; Appendix D - Waivers; Appendix E - Flexural "
    "strength; Appendix F - Strain repetitions; Appendix G - Cylindrical "
    "specimens. I did not get to H to N, which cover dynamic modulus, "
    "estimating the modulus, unbound bases, stabilized soils, subgrade, "
    "fatigue life and granular base.",
]
_CH12_RIGHT = [
    "Chapter 12 has eight tables:\n- Table 12-1: Example of Mixed Traffic "
    "Design\n- Table 12-2: Stress-Strength Ratios and Allowable Coverages\n- "
    "Table 12-3: Fatigue Damage Summary Sheet for Mixed Traffic\n- Table "
    "12-4: Pass-to-Coverage Ratios\n- Table 12-5: Design Example for Primary "
    "(Channelized) Traffic Areas\n- Table 12-6: Design Example for Secondary "
    "(Unchannelized) Traffic Areas\n- Table 12-7: Recommended Spacing of "
    "Transverse Contraction Joints\n- Table 12-8: Dowel Size and Spacing",
    "| Table | Title | Page |\n|---|---|---|\n| 12-1 | Example of Mixed "
    "Traffic Design | PDF p. 196 (printed 12-6) |\n| 12-2 | Stress-Strength "
    "Ratios | 198 |\n| 12-3 | Fatigue Damage Summary Sheet | 200 |\n| 12-4 | "
    "Pass-to-Coverage Ratios | 201 |\n| 12-5 | Primary (Channelized) Traffic "
    "| 205 |\n| 12-6 | Secondary (Unchannelized) Traffic | 207 |\n| 12-7 | "
    "Spacing of Transverse Contraction Joints | 211 |\n| 12-8 | Dowel Size "
    "and Spacing | 212 |",
]
_CH12_WRONG = [
    # 4 of the 8, the other four as bare topics
    "Chapter 12 tables: Table 12-1 Example of Mixed Traffic Design; Table "
    "12-2 Stress-Strength Ratios; Table 12-3 Fatigue Damage Summary Sheet; "
    "Table 12-4 Pass-to-Coverage Ratios. The chapter also discusses "
    "channelized and unchannelized traffic areas, transverse contraction "
    "joint spacing and dowels.",
    # every number with the wrong title
    "Table 12-1 Stress-Strength Ratios; Table 12-2 Fatigue Damage Summary; "
    "Table 12-3 Pass-to-Coverage; Table 12-4 Channelized traffic example; "
    "Table 12-5 Unchannelized traffic example; Table 12-6 Joint spacing; "
    "Table 12-7 Dowels; Table 12-8 Mixed traffic design.",
    "I could not find the table captions for Chapter 12. It covers mixed "
    "traffic design, stress-strength ratios, a fatigue damage summary, "
    "pass-to-coverage ratios, channelized and unchannelized traffic, joint "
    "spacing and dowels.",
]
_ASCE7_RIGHT = [
    "ASCE 7 Chapters 1, 2, 6, 7, 11, 12, 13, 15 and 26",
    "Chapter 3 modifies ASCE 7-22 Chapter 1 (general, UFC 3-1), Chapter 2 "
    "(load combinations, 3-2), Chapter 6 (tsunami loads, 3-3), Chapter 7 "
    "(snow loads, 3-4), Chapter 11 (seismic design criteria, 3-5), Chapter "
    "12 (seismic design requirements for building structures, 3-6), Chapter "
    "13 (nonstructural components, 3-7), Chapter 15 (nonbuilding structures, "
    "3-8) and Chapter 26 (wind loads, 3-9).",
    "| ASCE 7 chapter | Subject | UFC section |\n|---|---|---|\n| 1 | General "
    "| 3-1 |\n| 2 | Combinations of loads | 3-2 |\n| 6 | Tsunami loads | 3-3 "
    "|\n| 7 | Snow loads | 3-4 |\n| 11 | Seismic design criteria | 3-5 |\n| "
    "12 | Building structures | 3-6 |\n| 13 | Nonstructural components | 3-7 "
    "|\n| 15 | Nonbuilding structures | 3-8 |\n| 26 | Wind loads | 3-9 |",
]
_ASCE7_WRONG = [
    "I could not find which ASCE 7 chapters Chapter 3 modifies; it covers "
    "general requirements, load combinations, tsunami, snow, seismic and "
    "wind loads.",
    # two chapters swapped with each other's subject
    "Chapter 1 - General; Chapter 2 - Load combinations; Chapter 6 - Snow "
    "loads; Chapter 7 - Tsunami loads; Chapter 11 - Seismic design criteria; "
    "Chapter 12 - Building structures; Chapter 13 - Nonstructural "
    "components; Chapter 15 - Nonbuilding structures; Chapter 26 - Wind "
    "loads.",
    "It modifies ASCE 7 Chapters 1, 2, 7 and 26.",
]
_TRAP_RIGHT = [
    "L=10'-0\" MIN., W=5'-0\" MIN., X=7'-0\" MIN.; also 1.5' MIN. and 21\" "
    "MIN.",
    "L = 10' minimum, W = 5' minimum, X = 7' minimum, 1.5' minimum, 21 in. "
    "minimum",
]
_TRAP_WRONG = [
    "L=10' MIN., W=5' MAX., X=7' MIN., 1.5' MIN., 21\" MIN.",
    "L=10' MIN., W=5' (as drawn), X=7' MIN., 1.5' MIN., 21\" MIN.",
    "W=5' (as drawn), X=7' MIN., L=10' MIN.; 1.5' MIN., 21\" MIN.",
    "L=10'-6\" MIN., W=5' MIN., X=7' MIN., 1.5' MIN., 21\" MIN.",
]


@pytest.mark.parametrize("task_id,right,wrong", [
    ("ufc260-appendices", _APPENDICES_RIGHT, _APPENDICES_WRONG),
    ("ufc260-ch12-tables", _CH12_RIGHT, _CH12_WRONG),
    ("ufc301-asce7-chapters", _ASCE7_RIGHT, _ASCE7_WRONG),
    ("meck-trap-dimensions", _TRAP_RIGHT, _TRAP_WRONG),
    ("meck-bioretention-section-dims",
     ["Section A-A: 10'-0\" MIN. across the top and 4'-0\" MIN. of depth.",
      "10'-0\" MIN.; 4' MIN."],
     ["Section A-A: 10'-6\" MIN. and 4'-0\" MAX."]),
])
def test_the_checks_take_right_answers_and_refuse_wrong_ones(task_id, right,
                                                             wrong):
    task = next(t for t in OPEN_TASKS if t.id == task_id)
    assert _passes(task_id, task.truth)
    for answer in right:
        assert _passes(task_id, answer), answer
    for answer in wrong:
        assert not _passes(task_id, answer), answer


def test_drawing_notation_is_a_minimum():
    """Item 8: `10'-0" MIN.` is 10 feet minimum; `10'-6" MIN.` is not."""
    from funhouse_agent.review_eval.tasks import _min
    yes = ["10'-0\" MIN.", "10' - 0\" min", "10'-0\" minimum", "10' MIN.",
           "at least 10'-0\"", "10 ft 0 in minimum"]
    no = ["10'-6\" MIN.", "110' MIN.", "10'-0\" MAX.", "min 10'-6\""]
    for text in yes:
        assert C.term_found(_min("10"), C.normalize(text)), text
    for text in no:
        assert not C.term_found(_min("10"), C.normalize(text)), text


def test_labelled_set_reads_ids_with_their_titles():
    items = {"table 1": {"ids": [r"\btable\s*1\b"], "titles": ["soils"]},
             "table 2": {"ids": [r"\btable\s*2\b"], "titles": ["rock"]}}
    ok, _ = C.check_labelled_set("Table 1: soils. Table 2: rock.", items)
    assert ok
    ok, detail = C.check_labelled_set("Table 1: rock. Table 2: soils.", items)
    assert not ok and "another item's title" in detail
    assert C.check_labelled_set("Soils (Table 1); rock (Table 2).", items)[0]
    # titles optional: an id alone counts, one with the other's title not
    assert C.check_labelled_set("Tables 1 and 2", items,
                                require_title=False,
                                enum_lead=r"\btables?\s*",
                                enum_tokens={"table 1": "1",
                                             "table 2": "2"})[0]
    assert not C.check_labelled_set("Table 1 - rock; table 2 - rock", items,
                                    require_title=False)[0]


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


def test_summary_counts_failed_calls_inside_finished_runs():
    """A run the agent finished can still hide failed model calls (Foundry,
    2026-10-01: dropped connections, rate limits on vision calls); they get
    their own column beside the whole-run errors."""
    from funhouse_agent.review_eval.runner import summarize
    tasks = [Task(id="a", question="q", documents=[], category="locate",
                  doc_type="drawing_set", checks=[]),
             Task(id="b", question="q", documents=[], category="check",
                  doc_type="calc_package", checks=[])]
    ok = {"score": {"passed": True, "checks_passed": 1, "checks_total": 1}}
    runs = {"sweep": {"a": dict(ok, model_errors=3),
                      "b": dict(ok, model_errors=2, error="ReadTimeout")}}
    md = summarize(runs, ["sweep"], tasks,
                   {"model": "m", "versions": {}, "updated": "now"})
    header = next(l for l in md.splitlines() if l.startswith("| arm |"))
    row = next(l for l in md.splitlines() if l.startswith("| sweep |"))
    cols = [c.strip() for c in header.strip("|").split("|")]
    vals = [c.strip() for c in row.strip("|").split("|")]
    got = dict(zip(cols, vals))
    assert got["failed calls"] == "5" and got["errors"] == "1"
    assert got["step caps"] == "0"


# -- marks on targets (field session 2026-10-01) -----------------------------

def _tag_marks(tmp_path, boxes, name="marked.pdf", kind="circle"):
    from planlens.document.markup_writer import write_markups
    from planlens.testing.tag_fixtures import build_synthetic_tag_set
    out = str(tmp_path / name)
    write_markups(build_synthetic_tag_set().pdf, out, [
        {"kind": kind, "page": 0, "comment": "GCE", "label": "GCE",
         "bbox": list(b)} for b in boxes], author="AI")
    return out


def _gce_callouts(page=0, text="GCE"):
    from planlens.testing.tag_fixtures import build_synthetic_tag_set
    return [t.bbox for t in build_synthetic_tag_set().tags
            if t.page == page and t.text == text and t.kind == "callout"]


ON_TARGET = {"fixture": "tags", "text": "GCE", "kind": "callout", "page": 0}


def test_rings_on_every_callout_pass(tmp_path):
    pdf = _tag_marks(tmp_path, _gce_callouts())
    ok, detail = C.check_markups_on_targets("", [pdf], **ON_TARGET)
    assert ok, detail
    assert "7/7 targets" in detail


def test_rings_at_invented_coordinates_fail(tmp_path):
    pdf = _tag_marks(tmp_path, [(430, 250, 500, 320), (520, 300, 590, 370)])
    ok, detail = C.check_markups_on_targets("", [pdf], **ON_TARGET)
    assert not ok and "0/7 targets" in detail


def test_look_alikes_circled_too_cost_precision(tmp_path):
    boxes = _gce_callouts() + _gce_callouts(text="GCG") + \
        _gce_callouts(text="QCE")
    pdf = _tag_marks(tmp_path, boxes)
    ok, detail = C.check_markups_on_targets("", [pdf], **ON_TARGET)
    assert not ok and "7/7 targets" in detail and "precision 0.54" in detail


def test_one_blanket_box_over_the_sheet_is_not_a_hit(tmp_path):
    pdf = _tag_marks(tmp_path, [(50, 50, 1150, 750)], kind="box")
    ok, detail = C.check_markups_on_targets("", [pdf], **ON_TARGET)
    assert not ok and "0/7 targets" in detail


# -- where a comment points (Foundry brief 4, 2026-10-07) ----------------------

def _comment_sheet(tmp_path, markups, name="commented.pdf"):
    """A sheet with two notes that both state a maximum, and the given
    comments written on a copy (displayed-frame points)."""
    from planlens.document.markup_writer import write_markups
    doc = fitz.open()
    page = doc.new_page(width=792, height=612)
    page.insert_text((60, 228), "SLOPE SHALL NOT EXCEED 6.25% MAX.",
                     fontsize=7)
    page.insert_text((420, 430), "SLOPE UP TO 5% (6.2% MAX.)", fontsize=7)
    src = doc.tobytes()
    doc.close()
    out = str(tmp_path / name)
    write_markups(src, out, markups, author="AI")
    return out


#: The first note's printed line, as a task would carry it (measured once).
_NOTE = [{"page": 0, "box": [60.0, 222.0, 190.0, 229.0], "name": "the note"}]
_ASK = "DRAFT: please confirm the 6.25% maximum."


def test_a_comment_on_the_right_line_points_at_it(tmp_path):
    for i, mark in enumerate([
            {"kind": "callout", "points_at": [60.0, 226.0]},
            {"kind": "note", "point": [143.0, 225.0]},
            {"kind": "box", "bbox": [58.0, 221.0, 192.0, 230.0]},
            {"kind": "highlight", "quote": "SHALL NOT EXCEED 6.25%"}]):
        pdf = _comment_sheet(tmp_path, [dict(mark, page=0, comment=_ASK)],
                             name=f"right{i}.pdf")
        ok, detail = C.check_markups_point_at(
            "", [pdf], targets=_NOTE, text_contains="6.25", pad=4.0)
        assert ok, (mark, detail)
        assert "ON it" in detail


def test_a_comment_on_another_note_fails_however_it_mentions_the_figure(
        tmp_path):
    """Brief 4, GPT-5.4 baseline: the comment mentioned 8.33 and pointed at
    the section label stating 8.3 - the old check passed it."""
    pdf = _comment_sheet(tmp_path, [
        {"kind": "callout", "page": 0, "comment": _ASK,
         "quote": "SLOPE UP TO 5% (6.2% MAX.)"}])
    assert C.check_pdf_markups("", [pdf], min=1, pages=[0],
                               text_contains="6.25")[0]
    ok, detail = C.check_markups_point_at(
        "", [pdf], targets=_NOTE, text_contains="6.25", pad=4.0)
    assert not ok and "recall 0.00" in detail
    # and two comments, one right one wrong: recall met, precision says so
    pdf2 = _comment_sheet(tmp_path, [
        {"kind": "callout", "page": 0, "comment": _ASK,
         "quote": "SLOPE UP TO 5% (6.2% MAX.)"},
        {"kind": "note", "page": 0, "comment": _ASK, "point": [61, 225]}],
        name="both.pdf")
    ok, detail = C.check_markups_point_at(
        "", [pdf2], targets=_NOTE, text_contains="6.25")
    assert ok and "precision 0.50" in detail
    assert not C.check_markups_point_at(
        "", [pdf2], targets=_NOTE, text_contains="6.25",
        min_precision=1.0)[0]


def test_point_at_needs_targets_a_pdf_and_a_matching_comment(tmp_path):
    assert not C.check_markups_point_at("", [], targets=_NOTE)[0]
    assert "no targets" in C.check_markups_point_at("", ["x.pdf"])[1]
    pdf = _comment_sheet(tmp_path, [
        {"kind": "note", "page": 0, "comment": "unrelated", "point": [61, 225]}])
    ok, detail = C.check_markups_point_at("", [pdf], targets=_NOTE,
                                          text_contains="6.25")
    assert not ok and "no markup says" in detail


def test_the_ramp_note_target_on_the_public_sheet(tmp_path):
    """The produce-markup task's target on the public sheet 10.31A, against
    the anchors the brief-4 runs wrote: the five on note 4's line pass, the
    one on the section label fails."""
    from planlens.document.markup_writer import write_markups
    from funhouse_agent.review_eval.tasks import MECK_1031A_RAMP_NOTE
    path = D.find_file("10.31A.pdf")
    if not path:
        pytest.skip("10.31A.pdf is not in this checkout")
    task = next(t for t in OPEN_TASKS if t.id == "produce-markup")
    check = next(c for c in task.all_checks()
                 if c["type"] == "markups_point_at")
    ask = "DRAFT: Please confirm the 8.33% maximum applies over the run."
    cases = {(60.0, 227.4): True, (143.0, 227.3): True,
             (491.8, 434.4): False}
    for (x, y), want in cases.items():
        out = str(tmp_path / f"m_{int(x)}.pdf")
        write_markups(path, out, [{"kind": "callout", "page": 0,
                                   "comment": ask, "points_at": [x, y]}],
                      author="AI")
        r = C.run_check(check, "", files=[out])
        assert r["passed"] is want, ((x, y), r["detail"])
    assert check["targets"] is MECK_1031A_RAMP_NOTE


def test_the_tag_fixture_resolves_as_a_document():
    name, data = D.resolve("fixture_tags")
    assert name == "tag_set.pdf" and data[:4] == b"%PDF"


def test_a_hyphenated_compound_matches_the_spaced_term():
    """Foundry run 2026-10-02: four arms answered 'Edge-of-pavement
    elevation' and failed a check for 'edge of pavement'."""
    text = C.normalize("- **Edge-of-pavement elevation** at the right end")
    assert C.term_found("edge of pavement", text)
    assert C.term_found("edge-of-pavement", C.normalize("edge of pavement"))
    # digits keep their hyphens: M-278 is not "m 278" by this rule
    assert not C.term_found("m 278", C.normalize("AASHTO M-278"))


def test_latex_number_separators_read_as_plain_numbers():
    """Foundry rc2 run 2026-10-03: a correct answer wrote q_ult as
    ``1{,}195.3`` and failed both 1195 patterns."""
    terms = [[{"re": r"(?<![\w.,/-])1195(?!\d)"},
              {"re": r"(?<![\w.,/-])1,195(?!\d)"}]]
    assert C.check_contains_all(r"\(q_{ult} = 1{,}195.3\) kPa", terms)[0]
    assert C.check_contains_all(r"$q_{ult} = 1\,195.3$ kPa", terms)[0]
    assert not C.check_contains_all("q_ult = 1,159.3 kPa", terms)[0]


# -- whole-set coverage (pages_listed) ---------------------------------------

def test_pages_named_reads_lists_ranges_and_pdf_pages():
    assert C.pages_named("FPG callouts are on pages 4, 12 and 20.", 24) == \
        {4, 12, 20}
    assert C.pages_named("PDF page 12; also page 20 and p. 4", 24) == \
        {4, 12, 20}
    assert C.pages_named("sheets 3 & 9, pp. 10-12", 24) == {3, 9, 10, 11, 12}
    # Foundry rc2 run 2026-10-03: the colon form lost the list
    assert C.pages_named("FPG penetration callouts occur on PDF pages: "
                         "**4, 12, 19, and 20**", 24) == {4, 12, 19, 20}
    # rc3 run 2026-10-04: the same list bulleted, read as no pages at all
    assert C.pages_named("FPG penetration callouts with leaders appear on "
                         "PDF pages:\n\n- **4**\n- **12**\n- **19**\n- **20**"
                         "\n\nI visually checked all 24 sheets.", 24) == \
        {4, 12, 19, 20}
    # a bullet is not a range: "- 4\n- 12" is two pages, not 4 to 12
    assert C.pages_named("pages:\n- 4\n- 6", 24) == {4, 6}
    # a wide range talks about the whole set, it lists nothing found
    assert C.pages_named("pages 1-24 have legend rows", 24) == set()
    assert C.pages_named("page 31", 24) == set()


def test_the_long_set_check_wants_all_three_and_little_else():
    chk = {"expected": [4, 12, 20], "n_pages": 24, "min_recall": 1.0,
           "min_precision": 0.75}
    assert C.check_pages_listed("pages 4, 12 and 20", **chk)[0]
    assert not C.check_pages_listed("pages 4 and 12", **chk)[0]      # missed
    assert not C.check_pages_listed("pages 2, 4, 9, 12, 20", **chk)[0]
    assert not C.check_pages_listed("I could not tell.", **chk)[0]


def test_the_long_tag_set_puts_fpg_where_the_task_says():
    from funhouse_agent.review_eval.documents import (LONG_SET_PAGES,
                                                      LONG_SET_RARE, resolve)
    from planlens.testing.tag_fixtures import build_synthetic_tag_set
    gt = build_synthetic_tag_set(n_pages=LONG_SET_PAGES, gce_growth=0,
                                 extra_callouts=LONG_SET_RARE)
    pages = sorted({t.page + 1 for t in gt.tags
                    if t.text == "FPG" and t.kind == "callout"})
    task = next(t for t in OPEN_TASKS if t.id == "set-long-rare-tag")
    assert pages == task.checks[0]["expected"] == [4, 12, 20]
    name, data = resolve("fixture_tags_long")
    import fitz                      # bytes differ by PyMuPDF's file id
    with fitz.open(stream=data, filetype="pdf") as doc:
        assert name == "long_tag_set.pdf" and doc.page_count == 24


def test_rescore_rechecks_a_saved_run_without_a_model(tmp_path):
    """A check fixed after the run (Foundry rc2, 2026-10-03) re-scores the
    saved answer; no model is called, and the old score is kept."""
    import json
    from funhouse_agent.review_eval import score_review_suite
    tid = "calc-bearing-consistency"
    run_dir = tmp_path / "runs" / "baseline" / tid
    run_dir.mkdir(parents=True)
    answer = (r"Yes. \(q_{ult} = 1{,}195.3\) kPa (printed p. 3), FS = 3.0 "
              r"(p. 1), so \(q_{all} = 398.4\) kPa, matching p. 4.")
    stale = {"pass": False, "passed": 2, "total": 3}
    (run_dir / "run.json").write_text(json.dumps(
        {"task": tid, "arm": "baseline", "answer": answer, "files": [],
         "tool_calls": [], "error": None, "score": stale}), encoding="utf-8")

    class _NoModel:
        model = "none"

        def invoke(self, *a, **k):
            raise AssertionError("a rescore must not call the model")

        bind_tools = invoke

    out = score_review_suite(model=_NoModel(), out_dir=tmp_path,
                             arms=("baseline",), ids=[tid], rescore=True,
                             verbose=False)
    run = out["results"]["baseline"][tid]
    assert run["score_before_rescore"] == stale
    saved = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert saved["score"] == run["score"] != stale
    assert run["score"]["passed"], run["checks"]
