"""The cluster entry point offline: the guards, the scoring, the report.

No model is called and no corpus is needed. Every report is pre-seeded as a
finished run file, which is exactly the state a resumed notebook is in, so
the whole of the scoring and rendering path runs for real -- the set split,
the disputed drop, the blind-set second figure, the verdicts, and the
RESULTS.md that comes home.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from report_ingest import cluster_scoring as cs


@pytest.fixture(autouse=True)
def todays_readers(monkeypatch):
    """These runs script today's readers: model first, on for every stage
    when the setting is unset since BACKLOG RI-186, is set off for them; a
    test of model first asks for it."""
    from report_ingest.graph import MODEL_FIRST_ENV

    monkeypatch.setenv(MODEL_FIRST_ENV, "off")


def _run_blob(rid, n_pages, rules, final, changes=(), cost=None):
    return {
        "id": rid, "run_date": "2026-09-17", "n_pages": n_pages,
        "engine": "prompter", "triage_model": "funhouse-gpt-medium",
        "review_model": "funhouse-gpt-medium", "served_by": ["gpt-4o-2026"],
        "rules_labels": {str(k): v for k, v in rules.items()},
        "profile": {"document_type": "geotechnical report",
                    "workflow": "standard", "bound_together": [],
                    "toc_agreement": "partial", "scan_fraction": 0.0,
                    "rationale": "a whole report"},
        "review": {"final_labels": {str(k): v for k, v in final.items()},
                   "changes": list(changes), "rejected_changes": [],
                   "structure": [], "unresolved": [], "notes": "",
                   "tool_calls": 12, "budget": 60, "stopped_on_budget": False,
                   "model_calls": 5},
        "cost": cost or {"calls": 5, "input_tokens": 40000,
                         "output_tokens": 3000, "cache_read_tokens": 0,
                         "dollars": 0.0, "seconds": 90.0},
        "seconds": 95.0,
    }


@pytest.fixture()
def cluster(tmp_path):
    """A reports folder and an out_dir with every report already run."""
    reports_dir = tmp_path / "corpus"
    reports_dir.mkdir()
    out = tmp_path / "out"
    (out / "runs").mkdir(parents=True)
    (out / "triage").mkdir(parents=True)

    # R36 is in the cost checkpoint and so no longer blind; R31 was never
    # opened by anyone. Both are in the blind set.
    for rid in ("R36", "R31"):
        (reports_dir / f"{rid}.pdf").write_bytes(b"%PDF-1.7\n")

    # R36: the rules called page 3 a figure, the review made it a plan, and
    # the hand label says plan. A fix.
    (out / "runs" / "R36.json").write_text(json.dumps(_run_blob(
        "R36", 94,
        {3: "figure", 7: "boring_log"},
        {3: "plan", 7: "boring_log"},
        changes=[{"page": 3, "from": "figure", "to": "plan",
                  "reason": "the figure list calls it the location plan",
                  "evidence": "read_page"}])), encoding="utf-8")
    # R31: the rules were right and the review broke it.
    (out / "runs" / "R31.json").write_text(json.dumps(_run_blob(
        "R31", 120,
        {5: "lab_test", 9: "narrative"},
        {5: "figure", 9: "narrative"},
        changes=[{"page": 5, "from": "lab_test", "to": "figure",
                  "reason": "it is plotted", "evidence": "render_page"}])),
        encoding="utf-8")

    labels = {
        "R36": {"3": {"label": "plan", "alternates": []},
                "7": {"label": "boring_log", "alternates": []}},
        "R31": {"5": {"label": "lab_test", "alternates": []},
                "9": {"label": "narrative", "alternates": []}},
    }
    oos = tmp_path / "oos_labels.json"
    oos.write_text(json.dumps(labels), encoding="utf-8")
    return reports_dir, out, oos


# -- the guards ---------------------------------------------------------------

@pytest.mark.parametrize("workspace", ["/Workspace/Users/x/out",
                                       "/workspace/shared/out"])
def test_a_workspace_out_dir_is_allowed_and_gets_a_note(workspace):
    """The owner keeps things in the workspace folder; refusing it was wrong.

    Until this train a workspace ``out_dir`` raised. The owner's own rule of
    2026-09-20 names the workspace folder as one of the two durable places
    here, so it is allowed and the note says what the better shape is.
    """
    note = cs.out_dir_note(Path(workspace))
    assert note and "durable here" in note
    assert "durable_dir" in note


@pytest.mark.parametrize("plain", ["/tmp/report_ingest",
                                   "/Volumes/main/geo/out"])
def test_an_ordinary_out_dir_says_nothing(plain):
    assert cs.out_dir_note(Path(plain)) is None


def test_a_workspace_out_dir_runs_rather_than_raising(tmp_path, capsys,
                                                       monkeypatch):
    """And the note is printed where the owner will see it.

    Folder creation is stubbed: a real ``/Workspace`` cannot be created by
    a normal user on Linux (CI failed on that for months), and on Windows
    the call made a stray ``C:\\Workspace`` folder."""
    monkeypatch.setattr(cs.Path, "mkdir", lambda self, *a, **k: None)
    with pytest.raises(FileNotFoundError):     # no reports; it got past the
        cs.score_on_cluster(                   # guard, which is the point
            reports_dir=tmp_path / "nothing", out_dir="/Workspace/x/out",
            prompter=object())
    assert "workspace folder" in capsys.readouterr().out


def test_no_prompter_is_refused_rather_than_read_from_the_environment():
    with pytest.raises(ValueError, match="pass the live fh_prompter"):
        cs.score_on_cluster(reports_dir="/tmp/nowhere", prompter=None)


def test_a_missing_corpus_says_what_it_expected(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"R01\.pdf"):
        cs.score_on_cluster(reports_dir=tmp_path / "gone",
                            out_dir=tmp_path / "out", prompter=object())


def test_an_unknown_set_name_is_refused():
    with pytest.raises(ValueError, match="unknown set"):
        cs._set_ids("everything", None)


# -- resuming -----------------------------------------------------------------

def test_a_report_with_a_run_file_is_skipped_so_a_resume_is_free(cluster,
                                                                 capsys):
    reports_dir, out, oos = cluster
    cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                        prompter=object(), oos_labels=oos,
                        sets=("oos_blind",))
    printed = capsys.readouterr().out
    assert "R36: already done, skipping" in printed
    assert "R31: already done, skipping" in printed


def test_the_three_sets_together_are_every_report_in_the_corpus():
    ids = set(cs.OOS_OPEN) | set(cs.OOS_BLIND)
    assert len(ids) == 24, "the out-of-sample halves must not overlap"
    # insample comes from the spreadsheet at run time; 14 + 24 = 38.


# -- the scoring --------------------------------------------------------------

def test_the_set_is_scored_before_and_after_from_the_saved_runs(cluster):
    reports_dir, out, oos = cluster
    results = cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                                  prompter=object(), oos_labels=oos,
                                  sets=("oos_blind",))
    blind = results["sets"]["oos_blind"]
    assert blind["n_reports"] == 2
    assert blind["after"]["pages"] == 4
    # One fix and one break: accuracy is unmoved, which is the point of
    # grading each change rather than reading the total.
    assert blind["before"]["accuracy"] == pytest.approx(0.75)
    assert blind["after"]["accuracy"] == pytest.approx(0.75)
    assert blind["verdicts"]["fixed"] == 1
    assert blind["verdicts"]["broke"] == 1


def test_the_blind_set_is_reported_again_without_the_checkpoint_reports(
        cluster):
    reports_dir, out, oos = cluster
    results = cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                                  prompter=object(), oos_labels=oos,
                                  sets=("oos_blind",))
    clean = results["sets"]["oos_blind"]["never_seen"]
    assert clean["excluded"] == ["R36"], "R36 was read at the checkpoint"
    assert clean["reports"] == ["R31"]
    # R31 alone: the rules had both pages right, the review broke one.
    assert clean["before"]["accuracy"] == pytest.approx(1.0)
    assert clean["after"]["accuracy"] == pytest.approx(0.5)


def test_results_json_is_serializable_and_drops_the_working_state(cluster):
    reports_dir, out, oos = cluster
    cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                        prompter=object(), oos_labels=oos,
                        sets=("oos_blind",))
    blob = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert "_scores" not in blob["sets"]["oos_blind"]
    assert blob["sets"]["oos_blind"]["after"]["pages"] == 4


def test_results_md_carries_the_tables_and_names_nobody(cluster):
    reports_dir, out, oos = cluster
    cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                        prompter=object(), oos_labels=oos,
                        sets=("oos_blind",))
    text = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "# WP1b on the cluster" in text
    assert "strict accuracy" in text
    assert "What triage said" in text
    assert "honest blind figure" in text
    assert "R36" in text and "R31" in text
    # The reasons a change carries can name a firm; they stay in runs/.
    assert "the figure list calls it the location plan" not in text


def test_a_run_without_out_of_sample_labels_still_reports_triage(cluster):
    reports_dir, out, _ = cluster
    results = cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                                  prompter=object(), sets=("oos_blind",))
    assert results["sets"]["oos_blind"]["after"]["pages"] == 0
    assert {r["id"] for r in results["triage"]} == {"R31", "R36"}
    assert (out / "RESULTS.md").read_text(encoding="utf-8").count("|") > 10


def test_the_deployment_that_served_the_run_is_carried_into_the_report(
        cluster):
    reports_dir, out, oos = cluster
    results = cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                                  prompter=object(), oos_labels=oos,
                                  sets=("oos_blind",))
    assert results["served_by"] == ["gpt-4o-2026"]
    assert "gpt-4o-2026" in (out / "RESULTS.md").read_text(encoding="utf-8")


# -- the owner's own folder, under the reports' own names ---------------------

MANIFEST = """| ID | file | origin | pages | text | image | blank | toi | chars | sizes |
|---|---|---|---|---|---|---|---|---|---|
| R36 | `Reports_PDF/Harbour Widening Geotech.pdf` | public | 94 | 94 | 0 | 0 | 0 | 2000 | 8.5x11.0x94 |
| R31 | `Riverside Pumping Station 2018.pdf` | private | 120 | 120 | 0 | 0 | 0 | 1900 | 8.5x11.0x120 |
| R32 | `Reports_PDF/Never Uploaded.pdf` | private | 60 | 60 | 0 | 0 | 0 | 1900 | 8.5x11.0x60 |
"""


@pytest.fixture()
def named(cluster, tmp_path):
    """The same two runs, but the PDFs carry their original names.

    R32 is in the manifest and not in the folder, which is the state a
    part-uploaded folder is in.
    """
    reports_dir, out, oos = cluster
    for path in reports_dir.glob("R*.pdf"):
        path.unlink()
    (reports_dir / "Reports_PDF").mkdir()
    (reports_dir / "Reports_PDF" / "Harbour Widening Geotech.pdf"
     ).write_bytes(b"%PDF-1.7\n")
    (reports_dir / "Riverside Pumping Station 2018.pdf"
     ).write_bytes(b"%PDF-1.7\n")
    manifest = tmp_path / "small" / "MANIFEST.md"
    manifest.parent.mkdir()
    manifest.write_text(MANIFEST, encoding="utf-8")
    return reports_dir, manifest, out, oos


def test_reports_under_their_own_names_are_scored_through_the_manifest(named):
    reports_dir, manifest, out, oos = named
    results = cs.score_on_cluster(reports_dir=reports_dir, manifest=manifest,
                                  out_dir=out, prompter=object(),
                                  oos_labels=oos, sets=("oos_blind",))
    assert results["sets"]["oos_blind"]["n_reports"] == 2
    assert results["sets"]["oos_blind"]["after"]["pages"] == 4


def test_a_report_the_folder_does_not_have_is_named_and_skipped(named,
                                                                capsys):
    reports_dir, manifest, out, oos = named
    results = cs.score_on_cluster(reports_dir=reports_dir, manifest=manifest,
                                  out_dir=out, prompter=object(),
                                  oos_labels=oos, sets=("oos_blind",))
    assert "R32" in results["absent"]
    assert "R32" in capsys.readouterr().out
    assert "R32" in (out / "RESULTS.md").read_text(encoding="utf-8")


def test_the_older_corpus_dir_argument_still_works(cluster):
    reports_dir, out, oos = cluster
    results = cs.score_on_cluster(corpus_dir=reports_dir, out_dir=out,
                                  prompter=object(), oos_labels=oos,
                                  sets=("oos_blind",))
    assert results["sets"]["oos_blind"]["n_reports"] == 2


def test_a_run_with_no_reports_folder_at_all_is_refused(tmp_path):
    with pytest.raises(ValueError, match="pass reports_dir"):
        cs.score_on_cluster(out_dir=tmp_path / "out", prompter=object())


def test_tokens_are_totalled_and_reported_instead_of_dollars(cluster):
    reports_dir, out, oos = cluster
    results = cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                                  prompter=object(), oos_labels=oos,
                                  sets=("oos_blind",))
    assert results["totals"]["input_tokens"] == 80000
    assert results["totals"]["output_tokens"] == 6000
    text = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "80,000 tokens in" in text
    assert "no per-token price" in text


class TestTheTotalsCountEveryStage:
    """BACKLOG RI-173: the totals, and the run's last printed line, count
    every stage's failed items and requests. Every number is made up."""

    def test_a_labels_run_alone_keeps_its_totals_and_says_none(
            self, cluster, capsys):
        reports_dir, out, oos = cluster
        results = cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                                      prompter=object(), oos_labels=oos,
                                      sets=("oos_blind",))
        totals = results["totals"]
        assert (totals["stages"], totals["failed_items"]) == (["labels"], 0)
        assert totals["input_tokens"] == 80000
        assert "labels_cost" not in results
        printed = capsys.readouterr().out.rstrip().splitlines()
        assert printed[-1] == "failures by stage: labels none"

    def test_a_stages_failed_items_and_requests_reach_the_totals(
            self, logs_cluster, capsys):
        _reports_dir, out, _truth = logs_cluster
        path = out / "logs" / "R36_p38.json"
        blob = json.loads(path.read_text(encoding="utf-8"))
        blob["after"]["error"] = "TypeError: made up"
        blob["cost"].update(failed=3, failed_by_kind={"other": 3})
        path.write_text(json.dumps(blob), encoding="utf-8")
        capsys.readouterr()

        # object() has no chat(): the failed log's retry fails again.
        results = _logs_run(logs_cluster)
        totals = results["totals"]
        assert totals["stages"] == ["logs"]
        assert totals["failed_items_by_stage"] == {"logs": ["R36_p38"]}
        assert totals["items_by_stage"] == {"logs": 2}
        assert totals["failed_items"] == 1
        assert totals["failed"] == results["logs"]["cost"]["failed"]
        assert totals["calls"] == results["logs"]["cost"]["calls"]
        saved = json.loads((out / "results.json").read_text("utf-8"))
        assert saved["totals"]["failed_items"] == 1
        last = capsys.readouterr().out.rstrip().splitlines()[-1]
        assert last.startswith("failures by stage: logs 1 of 2 item(s) not "
                               "read (R36_p38)")

    def test_every_stage_adds_into_the_totals(self):
        def cost(calls, failed):
            total = cs._cost_totals()
            cs._add_cost(total, {"calls": calls, "failed": failed,
                                 "input_tokens": 10 * calls,
                                 "failed_by_kind": {"other": failed}})
            return total

        results = {
            "stages": ["labels", "logs", "ingest", "range_labels", "vote"],
            "failures": {}, "triage": [{"id": "R1"}], "totals": cost(2, 0),
            "logs": {"failures": {"R2_p1": "made up"},
                     "per_log": [{"log_id": "R1_p4", "error": "made up"},
                                 {"log_id": "R1_p5", "error": None}],
                     "cost": cost(5, 4)},
            "ingest": {"failures": {"R3": "made up"}, "per_report": [],
                       "cost": cost(1, 0), "cost_every_report": cost(3, 1)},
            "range_labels": {"failures": {}, "per_report": [{"id": "R1"}],
                             "cost": {"describe": cost(1, 0),
                                      "label": cost(1, 2)}},
            "vote": {"per_report": [{"id": "R1"}]},
        }
        cs._fold_totals(results)
        totals = results["totals"]
        assert (totals["calls"], totals["failed"]) == (12, 7)
        assert totals["input_tokens"] == 120
        assert totals["failed_by_kind"] == {"other": 7}
        assert totals["failed_items_by_stage"] == {
            "labels": [], "logs": ["R1_p4", "R2_p1"], "ingest": ["R3"],
            "range_labels": [], "vote": []}
        assert totals["failed_requests_by_stage"] == {
            "labels": 0, "logs": 4, "ingest": 1, "range_labels": 2,
            "vote": 0}
        assert totals["items_by_stage"]["logs"] == 3
        assert totals["failed_items"] == 3
        assert results["labels_cost"]["calls"] == 2
        assert cs._failures_line(totals) == (
            "failures by stage: labels none; logs 2 of 3 item(s) not read "
            "(R1_p4, R2_p1), 4 failed request(s); ingest 1 of 1 item(s) not "
            "read (R3), 1 failed request(s); range_labels 2 failed "
            "request(s); vote none")


# ---------------------------------------------------------------------------
# the logs stage (WP2b)
# ---------------------------------------------------------------------------

def _log_blob(log_id, report, set_name, before, after, calls=1):
    """A finished log run, the state a resumed notebook is in."""
    def side(found, total):
        return {
            "log_id": log_id, "report": report, "stage": "x",
            "scores": {"n_value": {"found": found, "total": total,
                                   "rate": found / total if total else None}},
            "overall": {"found": found, "total": total,
                        "rate": found / total if total else None},
            "cost": {}, "model_calls": calls, "unresolved": 0, "changes": 0,
            "error": None,
        }
    return {
        "log_id": log_id, "report": report, "run_date": "2026-09-17",
        "set": set_name, "model": "funhouse-gpt-high",
        "served_by": "gpt-4o-2026", "pages": [38],
        "before": side(*before), "after": side(*after),
        "cost": {"calls": calls, "input_tokens": 30000, "output_tokens": 2000,
                 "cache_read_tokens": 0, "dollars": 0.0, "seconds": 40.0},
        "seconds": 42.0,
    }


@pytest.fixture()
def logs_cluster(tmp_path):
    """A truth folder and an out_dir with every log already run."""
    reports_dir = tmp_path / "corpus"
    reports_dir.mkdir()
    for rid in ("R36", "R31"):
        (reports_dir / f"{rid}.pdf").write_bytes(b"%PDF-1.7\n")
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    (truth_dir / "OPEN.txt").write_text("R36\n", encoding="utf-8")
    for rid, page in (("R36", 38), ("R31", 278)):
        (truth_dir / f"{rid}_p{page}.json").write_text(json.dumps({
            "id": f"{rid}_p{page}", "pages": [page], "depth_unit": "ft",
            "fields": {}, "layers": [], "samples": [], "water": []}),
            encoding="utf-8")
    out = tmp_path / "out"
    (out / "runs").mkdir(parents=True)
    (out / "triage").mkdir(parents=True)
    (out / "logs").mkdir(parents=True)
    (out / "logs" / "R36_p38.json").write_text(
        json.dumps(_log_blob("R36_p38", "R36", "open", (6, 10), (9, 10))),
        encoding="utf-8")
    (out / "logs" / "R31_p278.json").write_text(
        json.dumps(_log_blob("R31_p278", "R31", "blind", (4, 10), (7, 10))),
        encoding="utf-8")
    return reports_dir, out, truth_dir


def _logs_run(logs_cluster, **over):
    reports_dir, out, truth_dir = logs_cluster
    kwargs = dict(reports_dir=reports_dir, out_dir=out, prompter=object(),
                  stages=("logs",), truth_dir=truth_dir, sets=())
    kwargs.update(over)
    return cs.score_on_cluster(**kwargs)


def test_an_unknown_stage_is_refused():
    with pytest.raises(ValueError, match="unknown stage"):
        cs.score_on_cluster(reports_dir=".", prompter=object(),
                            stages=("labels", "sideways"))


def test_no_stage_at_all_is_refused():
    with pytest.raises(ValueError, match="at least one stage"):
        cs.score_on_cluster(reports_dir=".", prompter=object(), stages=())


def test_the_logs_stage_without_truth_is_refused_before_anything_runs(
        tmp_path):
    with pytest.raises(ValueError, match="truth_dir"):
        cs.score_on_cluster(reports_dir=tmp_path, prompter=object(),
                            stages=("logs",))


def test_the_default_stage_is_still_labels_alone(cluster):
    reports_dir, out, oos = cluster
    results = cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                                  prompter=object(), oos_labels=oos)
    assert results["stages"] == ["labels"]
    assert "logs" not in results


def test_a_log_with_a_run_file_is_skipped_so_a_resume_is_free(logs_cluster):
    results = _logs_run(logs_cluster)
    assert results["logs"]["n_logs"] == 2
    # Nothing was opened and no model was called: object() has no chat().
    assert [r["log_id"] for r in results["logs"]["per_log"]] == [
        "R31_p278", "R36_p38"]


def test_a_saved_failure_is_retried_not_skipped(logs_cluster, capsys):
    """The first cluster run (2026-09-18) wrote six model refusals as
    finished runs, and the next run carried every one forward as done."""
    reports_dir, out, truth_dir = logs_cluster
    path = out / "logs" / "R31_p278.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    blob["after"]["error"] = "RuntimeError: Prompter returned nothing"
    path.write_text(json.dumps(blob), encoding="utf-8")

    _logs_run(logs_cluster)
    printed = capsys.readouterr().out

    assert "R31_p278: previous attempt failed" in printed
    assert "R31_p278: already done" not in printed
    assert "R36_p38: already done, skipping" in printed


def test_a_saved_failure_is_read_off_every_stage_shape():
    from report_ingest.cluster_scoring import _saved_failure

    assert _saved_failure({"after": {"error": "x"}}) == "x"        # logs, lab
    assert _saved_failure({"score": {"error": "y"}}) == "y"        # narrative
    assert _saved_failure({"error": "z"}) == "z"
    assert _saved_failure({"after": {"error": None}, "score": {}}) is None
    assert _saved_failure({}) is None


def test_before_and_after_are_reported_apart(logs_cluster):
    logs = _logs_run(logs_cluster)["logs"]
    everything = logs["sets"]["all"]
    assert everything["before"]["n_value"] == {"found": 10, "total": 20}
    assert everything["after"]["n_value"] == {"found": 16, "total": 20}


def test_open_and_blind_are_split_by_the_open_file(logs_cluster):
    logs = _logs_run(logs_cluster)["logs"]
    assert logs["open_set"] == ["R36"]
    assert logs["sets"]["open"]["logs"] == ["R36_p38"]
    assert logs["sets"]["blind"]["logs"] == ["R31_p278"]


def test_the_open_set_can_be_overridden(logs_cluster):
    logs = _logs_run(logs_cluster, open_reports=["R31"])["logs"]
    assert logs["sets"]["open"]["logs"] == ["R31_p278"]


def test_cost_is_totalled_per_log(logs_cluster):
    logs = _logs_run(logs_cluster)["logs"]
    assert logs["cost"]["input_tokens"] == 60000
    assert logs["cost"]["calls"] == 2


def test_results_md_carries_the_log_tables_and_names_nobody(logs_cluster):
    _reports_dir, out, _truth = logs_cluster
    _logs_run(logs_cluster)
    text = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "WP2b on the cluster" in text
    assert "R36_p38" in text and "R31_p278" in text
    assert "before" in text and "after" in text
    assert ".pdf" not in text


def test_floor_and_model_are_printed_beside_before_and_after(logs_cluster):
    """Foundry brief 4: the blind logs' index went 44 % before -> 0 % after,
    and RESULTS.md could not say whether the floor's record ever held the
    values or the model overruled them: the floor and the model, each scored
    alone, are printed in their own columns (an old run file without them
    shows a dash and the table says how many logs the column is over)."""
    _reports_dir, out, _truth = logs_cluster
    path = out / "logs" / "R31_p278.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    blob["after"]["floor_alone"] = {
        "scores": {"n_value": {"found": 5, "total": 10}},
        "overall": {"found": 5, "total": 10}}
    blob["after"]["model_alone"] = {
        "scores": {"n_value": {"found": 3, "total": 10}},
        "overall": {"found": 3, "total": 10}}
    path.write_text(json.dumps(blob), encoding="utf-8")
    logs = _logs_run(logs_cluster)["logs"]
    assert logs["sets"]["blind"]["n_floor"] == 1
    assert logs["sets"]["all"]["n_model"] == 1
    text = (out / "RESULTS.md").read_text(encoding="utf-8")
    head = next(l for l in text.splitlines() if l.startswith("metric"))
    assert head.split() == ["metric", "before", "floor", "model", "after"]
    blind = text.split("## blind")[1]
    n_row = next(l for l in blind.splitlines() if l.startswith("n_value"))
    assert n_row.split()[1:] == ["40%", "4/10", "50%", "5/10", "30%",
                                 "3/10", "70%", "7/10"]
    assert "`floor` is over the 1 of 2 log(s)" in text
    per = next(l for l in text.splitlines() if l.startswith("R36_p38"))
    assert per.count(" -") >= 2                       # no floor, no model


def test_the_logs_results_are_serialisable(logs_cluster):
    _reports_dir, out, _truth = logs_cluster
    _logs_run(logs_cluster)
    blob = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert blob["stages"] == ["logs"]
    assert blob["logs"]["n_logs"] == 2


def test_both_stages_run_in_one_call(logs_cluster, tmp_path):
    reports_dir, out, truth_dir = logs_cluster
    # The same two reports, with a finished label run each, in the same
    # out_dir the finished log runs are in -- a notebook that asked for both.
    (out / "runs" / "R36.json").write_text(json.dumps(_run_blob(
        "R36", 94, {3: "figure"}, {3: "plan"},
        changes=[{"page": 3, "from": "figure", "to": "plan",
                  "reason": "the figure list calls it the location plan",
                  "evidence": "read_page"}])), encoding="utf-8")
    (out / "runs" / "R31.json").write_text(json.dumps(_run_blob(
        "R31", 120, {5: "lab_test"}, {5: "lab_test"})), encoding="utf-8")
    oos = tmp_path / "oos_labels.json"
    oos.write_text(json.dumps({
        "R36": {"3": {"label": "plan", "alternates": []}},
        "R31": {"5": {"label": "lab_test", "alternates": []}}}),
        encoding="utf-8")

    results = cs.score_on_cluster(
        reports_dir=reports_dir, out_dir=out, prompter=object(),
        oos_labels=oos, stages=("labels", "logs"), truth_dir=truth_dir)
    assert results["stages"] == ["labels", "logs"]
    assert results["logs"]["n_logs"] == 2
    text = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "WP1b on the cluster" in text and "WP2b on the cluster" in text


def test_a_truth_folder_that_is_not_there_is_refused(logs_cluster, tmp_path):
    with pytest.raises(FileNotFoundError, match="no hand-truthed logs"):
        _logs_run(logs_cluster, truth_dir=tmp_path / "nowhere")


def test_a_log_whose_report_is_absent_is_named_and_skipped(logs_cluster,
                                                           tmp_path):
    reports_dir, out, truth_dir = logs_cluster
    (out / "logs" / "R31_p278.json").unlink()
    logs = _logs_run(logs_cluster)["logs"]
    assert logs["n_logs"] == 1
    assert "R31_p278" not in [r["log_id"] for r in logs["per_log"]]


# ---------------------------------------------------------------------------
# the lab stage
# ---------------------------------------------------------------------------

class TestTheLabStage:
    def test_the_stage_is_offered_and_needs_its_truth(self):
        from report_ingest.cluster_scoring import STAGE_NAMES, score_on_cluster

        assert "lab" in STAGE_NAMES
        with pytest.raises(ValueError, match="lab_truth_dir"):
            score_on_cluster(reports_dir="anywhere", prompter=object(),
                             stages=("lab",))

    def test_the_scorecard_renders_ids_kinds_and_rates_and_nothing_else(self):
        """RESULTS.md is the file that comes home, so it carries no wording
        from a page."""
        from report_ingest.cluster_scoring import _render_lab, _score_lab

        done = {
            "gradation__R36_p52": {
                "sheet_id": "gradation__R36_p52", "report": "R36",
                "kind": "gradation", "set": "open", "served_by": "gpt-x",
                "before": {"scores": {"index": {"found": 4, "total": 6},
                                      "series": {"found": 5, "total": 5}},
                           "overall": {"found": 9, "total": 11}},
                "after": {"scores": {"kind": {"found": 1, "total": 1},
                                     "link": {"found": 1, "total": 1},
                                     "index": {"found": 6, "total": 6},
                                     "series": {"found": 5, "total": 5}},
                          "overall": {"found": 13, "total": 13},
                          "model_calls": 1, "tool_calls": 0,
                          "unresolved": 0, "changes": 0, "error": None},
                "cost": {"calls": 1, "input_tokens": 9000,
                         "output_tokens": 700, "cache_read_tokens": 0,
                         "dollars": 0.0},
                "seconds": 12.0,
            },
            "triaxial__R35_p45": {
                "sheet_id": "triaxial__R35_p45", "report": "R35",
                "kind": "triaxial", "set": "blind", "served_by": "gpt-x",
                "before": {"scores": {"index": {"found": 0, "total": 7}},
                           "overall": {"found": 0, "total": 7}},
                "after": {"scores": {"kind": {"found": 1, "total": 1},
                                     "link": {"found": 1, "total": 1},
                                     "index": {"found": 5, "total": 7}},
                          "overall": {"found": 7, "total": 9},
                          "model_calls": 2, "tool_calls": 1,
                          "unresolved": 1, "changes": 1, "error": None},
                "cost": {"calls": 2, "input_tokens": 14000,
                         "output_tokens": 900, "cache_read_tokens": 0,
                         "dollars": 0.0},
                "seconds": 30.0,
            },
        }
        scored = _score_lab(done, {"chemical__R17_p155": "not in the folder"},
                            "funhouse-gpt-high", ("R36", "R28"))
        assert scored["n_sheets"] == 2
        assert scored["sets"]["open"]["n_sheets"] == 1
        assert scored["sets"]["blind"]["n_sheets"] == 1
        assert scored["kinds"]["gradation"]["n_sheets"] == 1
        text = "\n".join(_render_lab(scored))
        assert "gradation__R36_p52" in text and "triaxial__R35_p45" in text
        assert "chemical__R17_p155: not in the folder" in text
        # kind and link are the model's alone, and the table says so.
        assert "no before column" in text
        assert "Per kind" in text and "Per sheet" in text

    def test_floor_model_and_values_read_are_printed_beside_after(self):
        """Foundry brief 4: the run files kept floor_alone and model_alone
        but RESULTS.md printed only before and after, and an 80 % -> 37 %
        gradation drop could not be told apart from a linking failure."""
        from report_ingest.cluster_scoring import _render_lab, _score_lab

        def row(sheet, read=True):
            after = {"scores": {"kind": {"found": 1, "total": 1},
                                "link": {"found": 0, "total": 1},
                                "series": {"found": 1, "total": 28}},
                     "overall": {"found": 2, "total": 30},
                     "model_calls": 1, "tool_calls": 0, "unresolved": 0,
                     "changes": 0, "error": None,
                     "floor_alone": {"scores": {"series": {"found": 20,
                                                           "total": 28}},
                                     "overall": {"found": 20, "total": 30}},
                     "model_alone": {"scores": {"series": {"found": 1,
                                                           "total": 28}},
                                     "overall": {"found": 2, "total": 30}}}
            if read:
                after["unlinked"] = {
                    "scores": {"series": {"found": 27, "total": 28}},
                    "overall": {"found": 27, "total": 28}}
            return {"sheet_id": sheet, "report": "R15", "kind": "gradation",
                    "set": "blind", "served_by": "",
                    "before": {"scores": {"series": {"found": 28,
                                                     "total": 28}},
                               "overall": {"found": 28, "total": 28}},
                    "after": after,
                    "cost": {"calls": 1, "input_tokens": 0,
                             "output_tokens": 0, "cache_read_tokens": 0,
                             "dollars": 0.0},
                    "seconds": 1.0}

        done = {"gradation__R15_p82": row("gradation__R15_p82"),
                "gradation__R17_p114": row("gradation__R17_p114",
                                           read=False)}
        scored = _score_lab(done, {}, "m", ())
        blind = scored["sets"]["blind"]
        assert blind["n_unlinked"] == 1 and blind["n_floor"] == 2
        text = "\n".join(_render_lab(scored))
        head = next(l for l in text.splitlines() if l.startswith("metric"))
        assert head.split() == ["metric", "before", "floor", "model",
                                "after", "read"]
        series = next(l for l in text.splitlines()
                      if l.startswith("series"))
        assert "40/56" in series and "2/56" in series and "27/28" in series
        assert "`read` is over the 1 of 2 sheet(s)" in text
        per = next(l for l in text.splitlines()
                   if l.startswith("gradation__R15_p82"))
        assert "27/28" in per and "20/30" in per

    def test_a_sheet_moved_into_the_open_set_moves_in_the_scorecard(self):
        """The split is decided at scoring time, never frozen into a run."""
        from report_ingest.cluster_scoring import _score_lab

        row = {
            "sheet_id": "gradation__R06_p67", "report": "R06",
            "kind": "gradation", "set": "blind", "served_by": "",
            "before": {"scores": {}, "overall": {"found": 0, "total": 0}},
            "after": {"scores": {}, "overall": {"found": 0, "total": 0},
                      "model_calls": 1, "tool_calls": 0, "unresolved": 0,
                      "changes": 0, "error": None},
            "cost": {"calls": 1, "input_tokens": 0, "output_tokens": 0,
                     "cache_read_tokens": 0, "dollars": 0.0},
            "seconds": 1.0,
        }
        row["set"] = "open"          # what _run_lab does on a resumed run
        scored = _score_lab({"gradation__R06_p67": row}, {}, "m", ("R06",))
        assert scored["sets"]["open"]["n_sheets"] == 1
        assert "blind" not in scored["sets"]

    def test_the_specimen_label_arm_is_stated_and_a_mix_is_flagged(self):
        """RI-07: a run with the specimen-label paragraph on is a different
        arm, and a resumed run must not quietly mix the two."""
        from report_ingest.cluster_scoring import _render_lab, _score_lab

        def row(sheet, **over):
            base = {
                "sheet_id": sheet, "report": "R06", "kind": "gradation",
                "set": "open", "served_by": "",
                "before": {"scores": {}, "overall": {"found": 0, "total": 0}},
                "after": {"scores": {}, "overall": {"found": 0, "total": 0},
                          "model_calls": 1, "tool_calls": 0,
                          "unresolved": 0, "changes": 0, "error": None},
                "cost": {"calls": 1, "input_tokens": 0, "output_tokens": 0,
                         "cache_read_tokens": 0, "dollars": 0.0},
                "seconds": 1.0}
            base.update(over)
            return base

        old = _score_lab({"a": row("a")}, {}, "m", ("R06",))
        assert old["link_from_label"] == [False]   # written before the switch
        assert "GEOTECH_INGEST_LAB_LINK_FROM_LABEL): off." in \
            "\n".join(_render_lab(old))
        mixed = _score_lab({"a": row("a"), "b": row("b", link_from_label=True)},
                           {}, "m", ("R06",))
        assert mixed["link_from_label"] == [False, True]
        assert "MIXED" in "\n".join(_render_lab(mixed))
        # RI-56's per-page floor switch is stated the same way.
        assert old["floor_per_page"] == [False]
        assert "GEOTECH_INGEST_LAB_FLOOR_PER_PAGE): off." in \
            "\n".join(_render_lab(old))
        mixed = _score_lab({"a": row("a"), "b": row("b", floor_per_page=True)},
                           {}, "m", ("R06",))
        assert mixed["floor_per_page"] == [False, True]
        assert ("GEOTECH_INGEST_LAB_FLOOR_PER_PAGE): off and on -- MIXED"
                in "\n".join(_render_lab(mixed)))


# ---------------------------------------------------------------------------
# the calc stage (WP5)
# ---------------------------------------------------------------------------

def _calc_row(calc_id, report, kind, which, **over):
    row = {
        "calc_id": calc_id, "report": report, "kind": kind, "set": which,
        "served_by": "gpt-x", "pages": [41, 42, 43], "n_pages": 3,
        "before": {"scores": {"program": {"found": 1, "total": 1},
                              "inputs": {"found": 9, "total": 13},
                              "results": {"found": 1, "total": 4}},
                   "overall": {"found": 11, "total": 18}},
        "after": {"scores": {"kind": {"found": 1, "total": 1},
                             "program": {"found": 1, "total": 1},
                             "method": {"found": 1, "total": 1},
                             "subject": {"found": 1, "total": 1},
                             "inputs": {"found": 12, "total": 13},
                             "results": {"found": 4, "total": 4}},
                  "overall": {"found": 20, "total": 21},
                  "kind_read": kind, "model_calls": 1, "tool_calls": 0,
                  "unresolved": 0, "misplaced": 1, "floor_values": 48,
                  "disagreements": 1, "kept": 34, "added": 6,
                  "reconciled": 8, "error": None},
        "cost": {"calls": 1, "input_tokens": 21000, "output_tokens": 1400,
                 "cache_read_tokens": 0, "dollars": 0.0},
        "seconds": 19.0,
    }
    row.update(over)
    return row


class TestTheCalcStage:
    def test_the_stage_is_offered_and_needs_its_truth(self):
        from report_ingest.cluster_scoring import STAGE_NAMES, score_on_cluster

        assert "calc" in STAGE_NAMES
        with pytest.raises(ValueError, match="calc_truth_dir"):
            score_on_cluster(reports_dir="anywhere", prompter=object(),
                             stages=("calc",))

    def test_the_scorecard_renders_ids_kinds_and_rates_and_nothing_else(self):
        """RESULTS.md is the file that comes home, so it carries no wording
        from a page."""
        from report_ingest.cluster_scoring import _render_calc, _score_calc

        done = {
            "settlement__R18_p41": _calc_row(
                "settlement__R18_p41", "R18", "settlement", "open"),
            "pavement__R29_p127": _calc_row(
                "pavement__R29_p127", "R29", "pavement", "open", n_pages=4),
        }
        scored = _score_calc(
            done, {"slope_stability__R23_p386": "not in the folder"},
            "funhouse-gpt-high", ("R18", "R29"))

        assert scored["n_calcs"] == 2 and scored["n_pages"] == 7
        assert scored["sets"]["open"]["n_calcs"] == 2
        assert "blind" not in scored["sets"]
        assert scored["kinds"]["settlement"]["n_calcs"] == 1
        text = "\n".join(_render_calc(scored))
        assert "settlement__R18_p41" in text and "pavement__R29_p127" in text
        assert "slope_stability__R23_p386: not in the folder" in text
        assert "no before column" in text
        assert "Per kind" in text and "Per run" in text

    def test_an_empty_open_set_says_there_is_no_blind_set(self):
        """Which is the state this train ships in, and a scorecard that did
        not say so would read as if the numbers were blind."""
        from report_ingest.cluster_scoring import _render_calc, _score_calc

        scored = _score_calc(
            {"settlement__R18_p41": _calc_row(
                "settlement__R18_p41", "R18", "settlement", "open")},
            {}, "m", ())
        text = "\n".join(_render_calc(scored))

        assert "THERE IS NO BLIND SET" in text.upper()
        assert "IN-SAMPLE" in text.upper()

    def test_a_run_moved_into_the_open_set_moves_in_the_scorecard(self):
        """The split is decided at scoring time, never frozen into a run."""
        from report_ingest.cluster_scoring import _score_calc

        row = _calc_row("settlement__R18_p41", "R18", "settlement", "blind")
        row["set"] = "open"         # what _run_calc does on a resumed run
        scored = _score_calc({"settlement__R18_p41": row}, {}, "m", ("R18",))

        assert scored["sets"]["open"]["n_calcs"] == 1
        assert "blind" not in scored["sets"]

    def test_a_saved_run_is_read_back_and_not_paid_for_again(self, tmp_path):
        """The stage is restartable: a run file on disk is the answer."""
        from report_ingest.cluster_scoring import _run_calc

        truth_dir = tmp_path / "calc"
        truth_dir.mkdir()
        (truth_dir / "settlement__R18_p41.json").write_text(json.dumps({
            "id": "settlement__R18_p41", "report": "R18",
            "kind": "settlement", "pages": [41]}), encoding="utf-8")
        (truth_dir / "OPEN.txt").write_text("R18\n", encoding="utf-8")
        out = tmp_path / "out"
        (out / "calc").mkdir(parents=True)
        (out / "calc" / "settlement__R18_p41.json").write_text(
            json.dumps(_calc_row("settlement__R18_p41", "R18", "settlement",
                                 "blind")), encoding="utf-8")

        class _Corpus:
            def present_ids(self):
                return ["R18"]

            def open_report(self, *a, **k):      # pragma: no cover - guard
                raise AssertionError("a saved run must not be re-read")

        scored = _run_calc(_Corpus(), object(), "m", out, truth_dir,
                           budget=2, redo=False, max_calcs=None,
                           open_reports=None)

        assert scored["n_calcs"] == 1
        assert scored["per_calc"][0]["set"] == "open"   # rescored, not frozen

    def test_a_truth_file_for_a_report_that_is_not_there_is_a_failure(
            self, tmp_path):
        from report_ingest.cluster_scoring import _run_calc

        truth_dir = tmp_path / "calc"
        truth_dir.mkdir()
        (truth_dir / "settlement__R99_p41.json").write_text(json.dumps({
            "id": "settlement__R99_p41", "report": "R99",
            "kind": "settlement", "pages": [41]}), encoding="utf-8")

        class _Corpus:
            def present_ids(self):
                return ["R18"]

        scored = _run_calc(_Corpus(), object(), "m", tmp_path / "out",
                           truth_dir, budget=2, redo=False, max_calcs=None,
                           open_reports=None)

        assert scored["n_calcs"] == 0
        assert "R99 is not in the reports folder" in             scored["failures"]["settlement__R99_p41"]

    @staticmethod
    def _picture_run(tmp_path, monkeypatch, turns):
        """``_run_calc`` on the two-page printout, reading for real on a
        fake engine in the Prompter's place; returns ``(run, add_truth,
        out)``."""
        pytest.importorskip("pymupdf")
        import report_ingest.engine as engine_module
        from report_ingest.calc_reader import PAGE_IMAGES_ENV
        from report_ingest.cluster_scoring import _run_calc
        from report_ingest.tests.calc_fixtures import build_two_page_printout
        from report_ingest.tests.fake_engine import FakeEngine
        from report_ingest.tests.test_calc_reader import _open

        def fake_engine(prompter, model, meter=None, **kwargs):
            engine = FakeEngine(turns(), name=model)
            engine.served_by = "fake-deployment"
            return engine

        monkeypatch.delenv(PAGE_IMAGES_ENV, raising=False)
        monkeypatch.setattr(engine_module, "PrompterEngine", fake_engine)
        pdf = build_two_page_printout().pdf
        truth_dir = tmp_path / "calc"
        truth_dir.mkdir()
        out = tmp_path / "out"
        (out / "calc").mkdir(parents=True)

        class _Corpus:
            def present_ids(self):
                return ["R90"]

            def open_report(self, *a, **k):
                return _open(pdf, tmp_path)

        def add_truth(calc_id):
            (truth_dir / f"{calc_id}.json").write_text(json.dumps({
                "id": calc_id, "report": "R90", "kind": "lateral_pile",
                "pages": [0, 1]}), encoding="utf-8")

        def run():
            return _run_calc(_Corpus(), object(), "m", out, truth_dir,
                             budget=2, redo=False, max_calcs=None,
                             open_reports=None)

        return run, add_truth, out

    def test_the_picture_arm_is_recorded_on_off_and_mixed(self, tmp_path,
                                                          monkeypatch):
        """RI-67: the run files and RESULTS say which arm of
        GEOTECH_INGEST_CALC_PAGE_IMAGES each run read in, as the lab stage
        does its switch, and a resumed run that mixed the two says so."""
        from report_ingest.calc_reader import use_page_images
        from report_ingest.cluster_scoring import _render_calc
        from report_ingest.tests.test_calc_reader import _reading

        run, add_truth, out = self._picture_run(
            tmp_path, monkeypatch,
            lambda: [{"final": _reading(kind="lateral_pile")}])
        add_truth("lateral_pile__R90_p0")
        with use_page_images():
            on = run()
        saved = json.loads((out / "calc" / "lateral_pile__R90_p0.json")
                           .read_text(encoding="utf-8"))
        assert saved["page_images"] is True
        assert saved["model"] == "m" and "engine" not in saved  # as before
        assert on["page_images"] == [True]
        assert on["per_calc"][0]["page_images"] is True
        assert ("Every page as a picture (RI-17, "
                "GEOTECH_INGEST_CALC_PAGE_IMAGES): on.") in \
            "\n".join(_render_calc(on))

        add_truth("lateral_pile__R90_p7")    # read with the switch off
        mixed = run()
        assert json.loads((out / "calc" / "lateral_pile__R90_p7.json")
                          .read_text(encoding="utf-8"))["page_images"] \
            is False
        assert mixed["page_images"] == [False, True]
        text = "\n".join(_render_calc(mixed))
        assert "GEOTECH_INGEST_CALC_PAGE_IMAGES): off and on -- MIXED" in text

    def test_a_failed_reading_is_put_in_the_arm_it_was_asked_for(
            self, tmp_path, monkeypatch):
        from report_ingest.calc_reader import use_page_images

        run, add_truth, out = self._picture_run(tmp_path, monkeypatch,
                                                lambda: [])
        add_truth("lateral_pile__R90_p0")
        with use_page_images():
            scored = run()
        saved = json.loads((out / "calc" / "lateral_pile__R90_p0.json")
                           .read_text(encoding="utf-8"))
        assert saved["after"]["error"]          # the reading did fail
        assert saved["page_images"] is True
        assert scored["page_images"] == [True]

    def test_a_run_file_from_before_the_switch_was_recorded_says_so(self):
        """Not 'off': RI-17's arms ran before RI-67 recorded them."""
        from report_ingest.cluster_scoring import _render_calc, _score_calc

        scored = _score_calc(
            {"settlement__R18_p41": _calc_row(
                "settlement__R18_p41", "R18", "settlement", "open")},
            {}, "m", ())
        assert scored["page_images"] == [None]
        text = "\n".join(_render_calc(scored))
        assert "GEOTECH_INGEST_CALC_PAGE_IMAGES): not recorded;" in text


# ---------------------------------------------------------------------------
# the narrative stage (WP4)
# ---------------------------------------------------------------------------

def _narrative_blob(rid: str, which: str, recall, precision):
    """One report's saved narrative run, as the stage writes it."""
    found, total = recall
    got, gave = precision
    return {
        "id": rid, "run_date": "2026-09-17", "set": which,
        "model": "funhouse-gpt-high", "served_by": "gpt-5.1-2026",
        "score": {
            "report": rid,
            "recall": {"found": found, "total": total},
            "precision": {"found": got, "total": gave},
            "agreement": {"found": 30, "total": 37},
            "by_kind": {"int": {"found": 1, "total": 2},
                        "string": {"found": found - 1, "total": total - 2},
                        "enum": {"found": 1, "total": 1}},
            "list_items": {"precision": {"found": 3, "total": 4},
                           "recall": {"found": 3, "total": 5}},
            "summaries": {"present": {"found": 2, "total": 3},
                          "within_limit": {"found": 2, "total": 2}},
            "fields": [
                {"field": "boringCount", "kind": "int", "verdict": "right",
                 "ok": True},
                {"field": "postName", "kind": "string", "verdict": "missed",
                 "ok": False},
                {"field": "projectNumber", "kind": "string",
                 "verdict": "invented", "ok": False},
            ],
            "unresolved": 2, "model_calls": 1,
            "cost": {}, "pages": [2, 3, 4], "error": None,
        },
        "cost": {"calls": 1, "input_tokens": 21000, "output_tokens": 1200,
                 "cache_read_tokens": 0, "dollars": 0.0},
        "seconds": 41.0,
    }


class TestTheNarrativeStage:
    """The WP4 stage: scored per report, open and blind apart, restartable."""

    def test_it_is_a_stage_name(self):
        assert "narrative" in cs.STAGE_NAMES

    def test_without_hand_answers_it_is_refused_before_anything_runs(
            self, tmp_path):
        with pytest.raises(ValueError, match="narrative_truth_dir"):
            cs.score_on_cluster(reports_dir=tmp_path, prompter=object(),
                                stages=("narrative",))

    def test_the_sets_are_split_and_the_totals_add_up(self):
        from report_ingest.cluster_scoring import _score_narrative

        done = {"R36": _narrative_blob("R36", "open", (9, 10), (9, 11)),
                "R22": _narrative_blob("R22", "blind", (5, 10), (5, 12))}
        scored = _score_narrative(done, {"R19": "no narrative item"},
                                  "funhouse-gpt-high", ("R36",))

        assert scored["n_reports"] == 2
        assert scored["sets"]["open"]["reports"] == ["R36"]
        assert scored["sets"]["blind"]["reports"] == ["R22"]
        everything = scored["sets"]["all"]["totals"]
        assert everything["recall"] == {"found": 14, "total": 20}
        assert everything["precision"] == {"found": 14, "total": 23}
        assert everything["summaries.present"] == {"found": 4, "total": 6}

    def test_the_per_question_table_says_which_questions_are_hard(self):
        from report_ingest.cluster_scoring import _score_narrative

        done = {"R36": _narrative_blob("R36", "open", (9, 10), (9, 11)),
                "R22": _narrative_blob("R22", "blind", (5, 10), (5, 12))}
        scored = _score_narrative(done, {}, "m", ("R36",))

        assert scored["fields"]["boringCount"] == {
            "right": 2, "asked": 2, "missed": 0, "invented": 0, "wrong": 0}
        assert scored["fields"]["postName"]["missed"] == 2
        # An invented answer is counted but was never asked for.
        assert scored["fields"]["projectNumber"]["asked"] == 0
        assert scored["fields"]["projectNumber"]["invented"] == 2

    def test_the_report_names_nobody_and_carries_the_rates(self):
        from report_ingest.cluster_scoring import (
            _render_narrative, _score_narrative,
        )

        done = {"R36": _narrative_blob("R36", "open", (9, 10), (9, 11)),
                "R22": _narrative_blob("R22", "blind", (5, 10), (5, 12))}
        scored = _score_narrative(done, {"R19": "no narrative item"}, "m",
                                  ("R36",))
        text = "\n".join(_render_narrative(scored))

        assert "R36" in text and "R22" in text
        assert "R19: no narrative item" in text
        assert "Per question" in text and "Per report" in text
        assert "90% 9/10" in text                  # R36's recall
        assert "presence and word limit only" in text

    def test_a_report_moved_into_the_open_set_moves_in_the_scorecard(self):
        from report_ingest.cluster_scoring import _score_narrative

        row = _narrative_blob("R06", "blind", (4, 8), (4, 9))
        row["set"] = "open"           # what _run_narrative does on a resume
        scored = _score_narrative({"R06": row}, {}, "m", ("R06",))

        assert scored["sets"]["open"]["n_reports"] == 1
        assert "blind" not in scored["sets"]

    def test_the_open_set_is_read_off_the_truth_folder(self, tmp_path):
        from report_ingest.cluster_scoring import (
            DEFAULT_OPEN_NARRATIVE, _open_set_or,
        )

        assert _open_set_or(tmp_path, DEFAULT_OPEN_NARRATIVE) == \
            DEFAULT_OPEN_NARRATIVE
        (tmp_path / "OPEN.txt").write_text("R15\nR28\n", encoding="utf-8")
        assert _open_set_or(tmp_path, DEFAULT_OPEN_NARRATIVE) == ("R15", "R28")


# ---------------------------------------------------------------------------
# the vision stage (WP5)
# ---------------------------------------------------------------------------

def _vision_blob(rid, n_pages, rules, seen, unresolved=(), qa=(),
                 mode="page", dpi=100.0, calls=None, windows=0,
                 dollars=0.0):
    """One report's saved vision run, as the stage writes it."""
    return {
        "id": rid, "run_date": "2026-09-17", "n_pages": n_pages,
        "model": "funhouse-gpt-low", "served_by": "gpt-4.1-2026",
        "mode": mode, "dpi": dpi, "outline_context": False, "detail": None,
        "rules_labels": {str(k): v for k, v in rules.items()},
        "vision": {
            "labels": {str(k): v for k, v in seen.items()},
            "detail": [{"page": k, "label": v, "confidence": 0.8,
                        "reason": "the title block says so"}
                       for k, v in seen.items()],
            "unresolved": [dict(u) for u in unresolved],
            "qa": [dict(q) for q in qa],
            "mode": mode, "dpi": dpi, "outline_context": False,
            "pages_asked": n_pages, "model_calls": calls or len(seen),
            "budget": None, "stopped_on_budget": False,
            "model": "gpt-4.1-2026",
            "cost": {"mode": mode, "windows": windows},
        },
        "cost": {"calls": calls or len(seen), "input_tokens": 13000,
                 "output_tokens": 900, "cache_read_tokens": 0,
                 "dollars": dollars, "seconds": 60.0},
        "seconds": 62.0,
    }


@pytest.fixture()
def vision_cluster(cluster):
    """The two reports of ``cluster``, with a vision run beside each.

    R36 keeps its label run, so it prints three columns; R31's is removed,
    so it prints two. That is the state a run makes when the vision stage is
    asked for on its own.
    """
    reports_dir, out, oos = cluster
    (out / "vision").mkdir(parents=True)
    # R36: vision agrees with the hand on page 3 and misses page 7.
    (out / "vision" / "R36.json").write_text(json.dumps(_vision_blob(
        "R36", 94, {3: "figure", 7: "boring_log"},
        {3: "plan", 7: "figure"})), encoding="utf-8")
    # R31: vision gets both right where the review broke one.
    (out / "vision" / "R31.json").write_text(json.dumps(_vision_blob(
        "R31", 120, {5: "lab_test", 9: "narrative"},
        {5: "lab_test"}, unresolved=[{"page": 9, "why": "no answer"}])),
        encoding="utf-8")
    return reports_dir, out, oos


def _vision_run(vision_cluster, **over):
    reports_dir, out, oos = vision_cluster
    kwargs = dict(reports_dir=reports_dir, out_dir=out, prompter=object(),
                  oos_labels=oos, sets=("oos_blind",),
                  stages=("vision_labels",))
    kwargs.update(over)
    return cs.score_on_cluster(**kwargs)


class TestTheVisionStage:
    """WP5: the same pages, labelled from their pictures, scored alike."""

    def test_it_is_a_stage_name_and_the_default_is_still_labels_alone(self,
                                                                     cluster):
        reports_dir, out, oos = cluster
        assert "vision_labels" in cs.STAGE_NAMES
        results = cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                                      prompter=object(), oos_labels=oos)
        assert results["stages"] == ["labels"]
        assert "vision_labels" not in results

    def test_a_report_with_a_run_file_is_skipped_so_a_resume_is_free(
            self, vision_cluster, capsys):
        results = _vision_run(vision_cluster)
        printed = capsys.readouterr().out
        assert "R36: already done, skipping" in printed
        assert results["vision_labels"]["n_reports"] == 2

    def test_an_unknown_vision_mode_is_refused(self, vision_cluster):
        with pytest.raises(ValueError, match="unknown vision_mode"):
            _vision_run(vision_cluster, vision_mode="contact")

    def test_the_default_model_is_the_cheapest_tier(self, vision_cluster):
        results = _vision_run(vision_cluster)
        assert results["vision_labels"]["model"] == "funhouse-gpt-low"

    def test_the_three_columns_are_scored_on_the_same_hand_labels(
            self, vision_cluster):
        seen = _vision_run(vision_cluster)["vision_labels"]
        blind = seen["sets"]["oos_blind"]
        # Four hand-labelled pages. Rules: 3/4 (R36 p3 wrong). Review: R36
        # fixed p3 and R31 broke p5, so 3/4 as well. Vision: p3 right, p7
        # wrong, p5 right, p9 unresolved and so scored as 'other' -- 2/4.
        assert blind["rules"]["pages"] == 4
        assert blind["vision"]["pages"] == 4
        assert blind["rules"]["accuracy"] == pytest.approx(0.75)
        assert blind["review"]["accuracy"] == pytest.approx(0.75)
        assert blind["vision"]["accuracy"] == pytest.approx(0.5)

    def test_an_unresolved_page_is_scored_and_not_excused(self,
                                                          vision_cluster):
        results = _vision_run(vision_cluster)["vision_labels"]
        row = [r for r in results["per_report"] if r["id"] == "R31"][0]
        assert row["unresolved"] == 1
        assert row["labelled"] == 1
        assert row["vision"] == pytest.approx(0.5), (
            "the page with no answer counts as 'other', which is wrong here")

    def test_the_review_column_only_appears_where_the_review_ran(
            self, vision_cluster):
        _reports_dir, out, _oos = vision_cluster
        (out / "runs" / "R31.json").unlink()
        results = _vision_run(vision_cluster)["vision_labels"]
        blind = results["sets"]["oos_blind"]
        assert blind["n_with_review"] == 1
        assert blind["n_reports"] == 2
        # R31 has no review, so its per-report review cell is empty rather
        # than a copy of the rules.
        row = [r for r in results["per_report"] if r["id"] == "R31"][0]
        assert row["review"] is None
        assert row["rules"] is not None and row["vision"] is not None

    def test_the_blind_set_is_a_summary_with_no_per_report_line(
            self, vision_cluster):
        _reports_dir, out, _oos = vision_cluster
        _vision_run(vision_cluster)
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        body = text.split("# WP5 on the cluster")[1]
        assert "summary only" in body
        assert "honest blind figure" in body
        assert "R31" not in body.split("## Cost")[0], (
            "a blind figure read report by report stops being blind")

    def test_results_md_carries_the_three_columns_and_names_nobody(
            self, vision_cluster):
        _reports_dir, out, _oos = vision_cluster
        _vision_run(vision_cluster)
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "# WP5 on the cluster" in text
        for column in ("rules", "+review", "vision"):
            assert f"P {column}" in text and f"R {column}" in text
        assert "strict accuracy" in text
        # A vision reason names what the model saw and can carry a firm.
        assert "the title block says so" not in text

    def test_the_settings_the_run_used_are_printed(self, vision_cluster):
        _reports_dir, out, _oos = vision_cluster
        results = _vision_run(vision_cluster)
        assert results["vision_labels"]["settings"] == {
            "mode": "page", "dpi": 100.0, "outline_context": False,
            "fallback": True}
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "mode `page`" in text and "100 dpi" in text

    def test_the_sheet_mode_and_the_outline_flag_reach_the_settings(
            self, vision_cluster):
        results = _vision_run(vision_cluster, vision_mode="sheet",
                              vision_dpi=72.0, vision_outline_context=True)
        assert results["vision_labels"]["settings"] == {
            "mode": "sheet", "dpi": 72.0, "outline_context": True,
            "fallback": True}

    def test_cost_is_totalled_per_report(self, vision_cluster):
        cost = _vision_run(vision_cluster)["vision_labels"]["cost"]
        assert cost["input_tokens"] == 26000
        assert cost["output_tokens"] == 1800

    def test_the_results_are_serialisable_without_the_working_state(
            self, vision_cluster):
        _reports_dir, out, _oos = vision_cluster
        _vision_run(vision_cluster)
        blob = json.loads((out / "results.json").read_text(encoding="utf-8"))
        assert blob["stages"] == ["vision_labels"]
        assert "_scores" not in blob["vision_labels"]["sets"]["oos_blind"]
        assert blob["vision_labels"]["sets"]["oos_blind"]["vision"]["pages"] \
            == 4

    def test_both_whole_report_stages_run_in_one_call(self, vision_cluster):
        _reports_dir, out, _oos = vision_cluster
        results = _vision_run(vision_cluster,
                              stages=("labels", "vision_labels"))
        assert results["stages"] == ["labels", "vision_labels"]
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "WP1b on the cluster" in text and "WP5 on the cluster" in text

    def test_a_report_the_folder_does_not_have_is_named_and_skipped(
            self, vision_cluster, tmp_path):
        reports_dir, out, oos = vision_cluster
        (out / "vision" / "R31.json").unlink()
        (reports_dir / "R31.pdf").unlink()
        (out / "runs" / "R31.json").unlink()
        results = _vision_run(vision_cluster)
        assert results["vision_labels"]["failures"] == {}, (
            "a report that is in neither the folder nor a run file is absent, "
            "not a failure")
        assert "R31" in results["absent"]


class TestTheThreeColumnTable:
    """``columns_label_table`` widens the before-and-after table to N runs."""

    def test_it_prints_a_pair_of_columns_per_run(self):
        from report_ingest.scoring import Scores, columns_label_table

        rules, vision = Scores(), Scores()
        for _ in range(4):
            rules.add("plan", "plan")
            vision.add("plan", "figure")
        rows = columns_label_table([("rules", rules), ("vision", vision)],
                                   ("plan",))
        assert "P rules" in rows[0] and "R vision" in rows[0]
        assert rows[1].startswith("plan")
        assert rows[1].count("1.000") == 2, "the rules got all four right"
        assert rows[1].count("0.000") >= 1, "vision got none of them"

    def test_a_run_that_did_not_happen_simply_has_no_columns(self):
        from report_ingest.scoring import Scores, columns_label_table

        only = Scores()
        only.add("toc", "toc")
        rows = columns_label_table([("rules", only)])
        assert "P rules" in rows[0]
        assert "vision" not in rows[0]

    def test_an_unmeasured_label_reads_as_a_dash_not_a_zero(self):
        from report_ingest.scoring import Scores, columns_label_table

        rules = Scores()
        rules.add("plan", "plan")
        rows = columns_label_table([("rules", rules)], ("cpt_log",))
        assert "--" in rows[1], (
            "a label the set holds no page of is unmeasured, and must read as "
            "neither a perfect score nor a failing one")


class TestOneTruthRoot:
    """The five sets of hand truth travel to the cluster as ONE folder.

    They are private, so they are uploaded by hand before every run. Five
    Volume paths that must each be right is five chances for one to be
    stale while the run still starts and scores against it; a root holding
    ``logs/``, ``lab/``, ``calc/``, ``soundings/`` and ``narrative/`` is one
    thing to get right.
    """

    def test_a_root_with_the_subfolders_is_split_by_stage(self, tmp_path):
        for name in ("logs", "lab", "calc", "soundings", "narrative"):
            (tmp_path / name).mkdir()

        logs, lab, narrative, calc, soundings = cs._truth_dirs(
            tmp_path, None, None)

        assert logs == tmp_path / "logs"
        assert lab == tmp_path / "lab"
        assert narrative == tmp_path / "narrative"
        assert calc == tmp_path / "calc"
        assert soundings == tmp_path / "soundings"

    def test_a_folder_of_log_truth_files_is_still_used_as_it_stands(self,
                                                                   tmp_path):
        """The older single-folder form: no `logs/` inside, so it IS the
        log truth folder. Breaking this would silently stop finding truth
        that is right there."""
        (tmp_path / "R06_p51.json").write_text("{}", encoding="utf-8")

        logs, lab, narrative, calc, soundings = cs._truth_dirs(
            tmp_path, None, None)

        assert logs == tmp_path
        assert lab is None and narrative is None and calc is None
        assert soundings is None

    def test_an_explicit_per_stage_folder_wins_over_the_root(self, tmp_path):
        for name in ("logs", "lab", "calc", "narrative"):
            (tmp_path / name).mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()

        logs, lab, narrative, calc, _soundings = cs._truth_dirs(
            tmp_path, elsewhere, elsewhere)

        assert logs == tmp_path / "logs"
        assert lab == elsewhere and narrative == elsewhere
        assert calc == tmp_path / "calc"

    def test_an_explicit_calc_folder_wins_over_the_root(self, tmp_path):
        for name in ("logs", "calc"):
            (tmp_path / name).mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()

        _logs, _lab, _narr, calc, _snd = cs._truth_dirs(
            tmp_path, None, None, elsewhere)

        assert calc == elsewhere

    def test_an_explicit_sounding_folder_wins_over_the_root(self, tmp_path):
        for name in ("logs", "soundings"):
            (tmp_path / name).mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()

        *_rest, soundings = cs._truth_dirs(tmp_path, None, None, None,
                                           elsewhere)

        assert soundings == elsewhere

    def test_no_truth_at_all_is_all_nones(self):
        assert cs._truth_dirs(None, None, None) == (None,) * 5

    def test_the_soundings_stage_is_one_of_the_stages(self):
        assert "soundings" in cs.STAGE_NAMES
        assert cs.TRUTH_SUBDIRS["soundings"] == "soundings"

    def test_a_root_missing_the_calc_folder_is_refused_by_that_stage(
            self, tmp_path):
        (tmp_path / "logs").mkdir()

        with pytest.raises(ValueError, match="calc/"):
            cs.score_on_cluster(reports_dir=tmp_path, prompter=object(),
                                stages=("calc",), truth_dir=tmp_path)

    def test_a_root_missing_a_stages_folder_is_refused_by_that_stage(
            self, tmp_path):
        """A root with logs/ but no narrative/ must not start the narrative
        stage against the root itself."""
        (tmp_path / "logs").mkdir()

        with pytest.raises(ValueError, match="narrative/"):
            cs.score_on_cluster(reports_dir=tmp_path, prompter=object(),
                                stages=("narrative",), truth_dir=tmp_path)

    def test_the_lab_stage_reads_its_folder_out_of_the_root(self, tmp_path):
        """Refused for a MISSING lab folder, accepted past the check once the
        root has one -- it then fails later, on the corpus, not on truth."""
        (tmp_path / "lab").mkdir()

        with pytest.raises(FileNotFoundError, match="no reports"):
            cs.score_on_cluster(reports_dir=tmp_path, prompter=object(),
                                stages=("lab",), truth_dir=tmp_path,
                                out_dir=tmp_path / "out")

@pytest.fixture()
def document_cluster(cluster):
    """The same two reports, run in DOCUMENT mode: windows, not pages."""
    reports_dir, out, oos = cluster
    (out / "vision").mkdir(parents=True)
    (out / "vision" / "R36.json").write_text(json.dumps(_vision_blob(
        "R36", 94, {3: "figure", 7: "boring_log"}, {3: "plan", 7: "figure"},
        mode="document", calls=3, windows=3)), encoding="utf-8")
    (out / "vision" / "R31.json").write_text(json.dumps(_vision_blob(
        "R31", 120, {5: "lab_test", 9: "narrative"},
        {5: "lab_test", 9: "narrative"},
        mode="document", calls=4, windows=4)), encoding="utf-8")
    return reports_dir, out, oos


class TestDocumentMode:
    """The third vision mode on the cluster: the whole report in view."""

    def test_it_is_a_mode_the_stage_accepts(self, document_cluster):
        results = cs.score_on_cluster(
            reports_dir=document_cluster[0], out_dir=document_cluster[1],
            prompter=object(), oos_labels=document_cluster[2],
            sets=("oos_blind",), stages=("vision_labels",),
            vision_mode="document")
        assert results["vision_labels"]["settings"]["mode"] == "document"

    def test_the_window_and_the_image_cap_reach_the_settings(self,
                                                             document_cluster):
        results = cs.score_on_cluster(
            reports_dir=document_cluster[0], out_dir=document_cluster[1],
            prompter=object(), oos_labels=document_cluster[2],
            sets=("oos_blind",), stages=("vision_labels",),
            vision_mode="document", vision_window=24,
            vision_images_per_call=40, vision_detail="low")
        settings = results["vision_labels"]["settings"]
        assert settings == {"mode": "document", "dpi": 100.0,
                            "outline_context": False, "fallback": True,
                            "detail": "low",
                            "window": 24, "images_per_call": 40}

    def test_a_page_mode_run_records_no_window_settings(self,
                                                        vision_cluster):
        """A setting only document mode has is recorded only for a document
        run, so a page-mode header does not print a window it never had."""
        results = _vision_run(vision_cluster)
        assert results["vision_labels"]["settings"] == {
            "mode": "page", "dpi": 100.0, "outline_context": False,
            "fallback": True}

    def test_the_defaults_are_the_measured_cap_and_the_thirty_six_window(
            self, document_cluster):
        from report_ingest.vision_labels import (
            DOCUMENT_WINDOW, MAX_IMAGES_PER_CALL,
        )

        results = cs.score_on_cluster(
            reports_dir=document_cluster[0], out_dir=document_cluster[1],
            prompter=object(), oos_labels=document_cluster[2],
            sets=("oos_blind",), stages=("vision_labels",),
            vision_mode="document")
        settings = results["vision_labels"]["settings"]
        assert settings["window"] == DOCUMENT_WINDOW == 36
        assert settings["images_per_call"] == MAX_IMAGES_PER_CALL == 50

    def test_an_unknown_detail_is_refused(self, document_cluster):
        with pytest.raises(ValueError, match="unknown vision_detail"):
            cs.score_on_cluster(
                reports_dir=document_cluster[0], out_dir=document_cluster[1],
                prompter=object(), oos_labels=document_cluster[2],
                sets=("oos_blind",), stages=("vision_labels",),
                vision_detail="medium")

    def test_the_header_names_the_mode_the_window_and_the_detail(
            self, document_cluster):
        _reports_dir, out, _oos = document_cluster
        cs.score_on_cluster(
            reports_dir=document_cluster[0], out_dir=out,
            prompter=object(), oos_labels=document_cluster[2],
            sets=("oos_blind", "insample"), stages=("vision_labels",),
            vision_mode="document", vision_detail="low")
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "mode `document`" in text
        assert "detail `low`" in text
        assert "window 36 pages, at most 50 images a call" in text
        assert "whole report in thumbnail beside it" in text

    def test_the_per_report_line_carries_the_windows_the_calls_covered(
            self, document_cluster, monkeypatch):
        # The blind set prints a summary and no per-report line, so the two
        # reports are put in the OPEN set to exercise the table itself.
        _reports_dir, out, _oos = document_cluster
        monkeypatch.setattr(cs, "OOS_OPEN", ("R36", "R31"))
        results = cs.score_on_cluster(
            reports_dir=document_cluster[0], out_dir=out,
            prompter=object(), oos_labels=document_cluster[2],
            sets=("oos_open",), stages=("vision_labels",),
            vision_mode="document")
        rows = {r["id"]: r for r in results["vision_labels"]["per_report"]}
        assert rows["R36"]["windows"] == 3
        assert rows["R36"]["calls"] == 3
        assert rows["R31"]["windows"] == 4
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "wins" in text, "the window column is in the per-report table"
        assert "one window is one call in document mode" in text
        line = [ln for ln in text.splitlines() if ln.startswith("R36")][0]
        cells = line.split()
        assert cells[-5] == "3", "the window count is on R36's line"
        assert cells[-4] == "--", (
            "no window was refused on size, so the split column is a dash")
        assert "`split` is how many windows the gateway refused" in text

    def test_page_mode_prints_a_dash_where_it_has_no_windows(
            self, vision_cluster, monkeypatch):
        _reports_dir, out, _oos = vision_cluster
        monkeypatch.setattr(cs, "OOS_OPEN", ("R36", "R31"))
        results = _vision_run(vision_cluster, sets=("oos_open",))
        rows = {r["id"]: r for r in results["vision_labels"]["per_report"]}
        assert rows["R36"]["windows"] == 0, (
            "page mode has calls, not windows")
        line = [ln for ln in (out / "RESULTS.md").read_text(
            encoding="utf-8").splitlines() if ln.startswith("R36")][0]
        assert "--" in line, "a dash, not a 0 that reads as 'it did nothing'"

    def test_the_three_columns_are_still_one_scorer_on_one_truth(
            self, document_cluster):
        results = cs.score_on_cluster(
            reports_dir=document_cluster[0], out_dir=document_cluster[1],
            prompter=object(), oos_labels=document_cluster[2],
            sets=("oos_blind",), stages=("vision_labels",),
            vision_mode="document")
        blind = results["vision_labels"]["sets"]["oos_blind"]
        # R36 p3 right, p7 wrong; R31 both right. Three of four.
        assert blind["vision"]["pages"] == 4
        assert blind["vision"]["accuracy"] == pytest.approx(0.75)
        assert blind["rules"]["accuracy"] == pytest.approx(0.75)


class TestDollarsInTheResults:
    """The Cost block prints dollars when a deployment was priced."""

    def test_an_unpriced_run_still_reports_tokens_and_no_dollar_figure(
            self, vision_cluster):
        _reports_dir, out, _oos = vision_cluster
        _vision_run(vision_cluster)
        block = (out / "RESULTS.md").read_text(
            encoding="utf-8").split("## Cost")[-1]
        assert "26,000 tokens in" in block
        assert "$" not in block
        assert "no per-token price" in block

    def test_a_priced_run_prints_dollars_beside_the_tokens(self, cluster):
        reports_dir, out, oos = cluster
        (out / "vision").mkdir(parents=True)
        (out / "vision" / "R36.json").write_text(json.dumps(_vision_blob(
            "R36", 94, {3: "figure", 7: "boring_log"},
            {3: "plan", 7: "figure"}, dollars=1.25)), encoding="utf-8")
        (out / "vision" / "R31.json").write_text(json.dumps(_vision_blob(
            "R31", 120, {5: "lab_test", 9: "narrative"},
            {5: "lab_test", 9: "narrative"}, dollars=0.75)), encoding="utf-8")
        cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                            prompter=object(), oos_labels=oos,
                            sets=("oos_blind",), stages=("vision_labels",))
        block = (out / "RESULTS.md").read_text(
            encoding="utf-8").split("# WP5 on the cluster")[1]
        cost = block.split("## Cost")[-1]
        assert "$2.00" in cost, "the two reports together"
        assert "$1.00" in cost, "and per report"
        assert "owner's own Funhouse rates" in cost
        assert "the DEPLOYMENT that answered" in cost
        assert "read a total as a floor" in cost


class TestTheReadmeCell:
    """Every keyword the README's cells pass must be one the function takes.

    The README is the owner's notebook cell: a parameter renamed here and
    not there is a TypeError on the cluster, minutes into a run, with the
    reports already uploaded.
    """

    @staticmethod
    def _cell_keywords():
        import re
        from pathlib import Path

        readme = (Path(cs.__file__).parent / "README.md").read_text(
            encoding="utf-8")
        names = set()
        for block in readme.split("score_on_cluster(")[1:]:
            call = block.split(")")[0]
            names |= set(re.findall(r"^\s*(\w+)\s*=", call, re.MULTILINE))
        return names

    def test_every_keyword_the_readme_passes_exists(self):
        import inspect

        taken = set(inspect.signature(cs.score_on_cluster).parameters)
        used = self._cell_keywords()
        assert used, "no score_on_cluster cell found in the README"
        assert used <= taken, f"the README passes {sorted(used - taken)}"

    def test_the_readme_shows_every_vision_mode_in_a_cell(self):
        """Both cells, whatever the alignment: the one to run and the one to
        try. The five-stage cell recommends `sheet`, which the 2026-09-20
        corpus run made the mode to run; the vision section's own cell shows
        `document`, which is the mode still being priced."""
        import re
        from pathlib import Path

        readme = (Path(cs.__file__).parent / "README.md").read_text(
            encoding="utf-8")
        shown = set(re.findall(r'vision_mode\s*=\s*"(\w+)"', readme))
        assert {"sheet", "document"} <= shown, shown
        assert "vision_detail" in readme
        assert "50" in readme and "images" in readme



class TestTheSplitAndTheFallbackOnTheCluster:
    """What 5.21.2 added to the stage: a window that halves itself when the
    gateway refuses its body, and one page-mode call for a page nothing
    answered for. Both are COUNTS in the saved run, so the table and the
    progress line read them off disk like everything else."""

    def _with_counts(self, document_cluster, splits=0, fallback_pages=0):
        reports_dir, out, oos = document_cluster
        blob = json.loads((out / "vision" / "R36.json").read_text(
            encoding="utf-8"))
        blob["vision"]["cost"]["splits"] = splits
        blob["vision"]["cost"]["fallback_pages"] = fallback_pages
        (out / "vision" / "R36.json").write_text(json.dumps(blob),
                                                 encoding="utf-8")
        return reports_dir, out, oos

    def test_the_split_count_reaches_the_per_report_row(self,
                                                        document_cluster):
        _reports, out, oos = self._with_counts(document_cluster, splits=2,
                                               fallback_pages=3)
        results = cs.score_on_cluster(
            reports_dir=document_cluster[0], out_dir=out, prompter=object(),
            oos_labels=oos, sets=("oos_blind",), stages=("vision_labels",),
            vision_mode="document")
        rows = {r["id"]: r for r in results["vision_labels"]["per_report"]}
        assert rows["R36"]["splits"] == 2
        assert rows["R36"]["fallback_pages"] == 3
        assert rows["R31"]["splits"] == 0

    def test_the_split_column_prints_the_count_and_a_dash(
            self, document_cluster, monkeypatch):
        _reports, out, oos = self._with_counts(document_cluster, splits=2)
        monkeypatch.setattr(cs, "OOS_OPEN", ("R36", "R31"))
        cs.score_on_cluster(
            reports_dir=document_cluster[0], out_dir=out, prompter=object(),
            oos_labels=oos, sets=("oos_open",), stages=("vision_labels",),
            vision_mode="document")
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        header = [ln for ln in text.splitlines() if "split" in ln
                  and "wins" in ln][0]
        assert "split" in header
        r36 = [ln for ln in text.splitlines() if ln.startswith("R36")][0]
        r31 = [ln for ln in text.splitlines() if ln.startswith("R31")][0]
        assert r36.split()[-4] == "2"
        assert r31.split()[-4] == "--", (
            "a report that never hit the body limit prints a dash")

    def test_the_header_says_the_pictures_are_jpeg_and_what_split_means(
            self, document_cluster):
        _reports, out, oos = document_cluster
        cs.score_on_cluster(
            reports_dir=document_cluster[0], out_dir=out, prompter=object(),
            oos_labels=oos, sets=("oos_blind",), stages=("vision_labels",),
            vision_mode="document")
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "JPEG" in text
        assert "request whose BODY is too large" in text

    def test_the_fallback_flag_is_on_by_default_and_can_be_turned_off(
            self, document_cluster):
        _reports, out, oos = document_cluster
        on = cs.score_on_cluster(
            reports_dir=document_cluster[0], out_dir=out, prompter=object(),
            oos_labels=oos, sets=("oos_blind",), stages=("vision_labels",),
            vision_mode="document")
        assert on["vision_labels"]["settings"]["fallback"] is True
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "page-mode fallback on" in text

        off = cs.score_on_cluster(
            reports_dir=document_cluster[0], out_dir=out, prompter=object(),
            oos_labels=oos, sets=("oos_blind",), stages=("vision_labels",),
            vision_mode="document", vision_fallback=False)
        assert off["vision_labels"]["settings"]["fallback"] is False
        assert "no fallback" in (out / "RESULTS.md").read_text(
            encoding="utf-8")


# ---------------------------------------------------------------------------
# the durable mirror (5.22.0)
# ---------------------------------------------------------------------------

from report_ingest.tests.test_mirror import BrokenFM, FakeFM  # noqa: E402


@pytest.fixture()
def unrun(tmp_path):
    """A corpus of two reports with NOTHING done yet, and no out_dir files."""
    reports_dir = tmp_path / "corpus"
    reports_dir.mkdir()
    for rid in ("R36", "R31"):
        (reports_dir / f"{rid}.pdf").write_bytes(b"%PDF-1.7\n")
    out = tmp_path / "out_521"
    labels = {
        "R36": {"3": {"label": "plan", "alternates": []}},
        "R31": {"5": {"label": "lab_test", "alternates": []}},
    }
    oos = tmp_path / "oos_labels.json"
    oos.write_text(json.dumps(labels), encoding="utf-8")
    return reports_dir, out, oos


def _fake_run_one(recorder):
    """A ``_run_one`` that writes its run file and calls no model."""

    def run_one(rid, corpus, prompter, model, triage_model, out_dir):
        recorder.append(rid)
        blob = _run_blob(rid, 10, {3: "plan"}, {3: "plan"})
        (out_dir / "runs" / f"{rid}.json").write_text(
            json.dumps(blob), encoding="utf-8")
        (out_dir / "triage" / f"{rid}.json").write_text(
            json.dumps(blob["profile"]), encoding="utf-8")
        return blob

    return run_one


class TestTheDurableMirror:
    """The output goes somewhere a cluster restart cannot reach."""

    def test_a_run_with_no_mirror_says_so_and_runs_anyway(self, cluster,
                                                          capsys):
        reports_dir, out, oos = cluster
        cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                            prompter=object(), oos_labels=oos,
                            sets=("oos_blind",))
        printed = capsys.readouterr().out
        assert "no durable mirror" in printed
        assert "sharepoint=fh_sp_client" in printed

    def test_what_the_mirror_holds_is_restored_before_anything_runs(
            self, unrun, capsys):
        reports_dir, out, oos = unrun
        fm = FakeFM()
        remote = f"{cs.DEFAULT_FOLDER}/{out.name}"
        fm.put(f"{remote}/runs/R36.json",
               json.dumps(_run_blob("R36", 10, {3: "figure"},
                                    {3: "plan"})).encode("utf-8"))
        # Only R36's PDF: R31 has neither a PDF nor a run and is skipped, so
        # nothing but the restore decides what this run scores.
        (reports_dir / "R31.pdf").unlink()

        results = cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                                      prompter=object(), oos_labels=oos,
                                      sets=("oos_blind",), sharepoint=fm)
        printed = capsys.readouterr().out
        assert "restored 1 file(s)" in printed
        assert "R36: already done, skipping" in printed
        assert (out / "runs" / "R36.json").is_file()
        assert results["sets"]["oos_blind"]["n_reports"] == 1

    def test_each_run_file_is_on_the_mirror_before_the_next_report_starts(
            self, unrun, monkeypatch):
        reports_dir, out, oos = unrun
        fm = FakeFM()
        started: list = []
        seen_at_start: list = []

        run_one = _fake_run_one(started)

        def spy(rid, *args, **kwargs):
            seen_at_start.append((rid, sorted(fm.tree)))
            return run_one(rid, *args, **kwargs)

        monkeypatch.setattr(cs, "_run_one", spy)
        cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                            prompter=object(), oos_labels=oos,
                            sets=("oos_blind",), sharepoint=fm)

        assert len(started) == 2, "both reports ran"
        first, second = started
        # When the SECOND report began, the FIRST one's run file was already
        # on the mirror. That is the whole point: a restart mid-run costs
        # the report in flight and nothing else.
        _rid, tree_at_second = seen_at_start[1]
        remote = f"{cs.DEFAULT_FOLDER}/{out.name}"
        assert f"{remote}/runs/{first}.json" in tree_at_second
        assert f"{remote}/triage/{first}.json" in tree_at_second
        assert f"{remote}/runs/{second}.json" in fm.tree, "and then the second"
        assert f"{remote}/RESULTS.md" in fm.tree, "and the results at the end"

    def test_a_failing_mirror_warns_once_and_the_run_finishes(
            self, unrun, monkeypatch, capsys):
        reports_dir, out, oos = unrun
        monkeypatch.setattr(cs, "_run_one", _fake_run_one([]))
        results = cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                                      prompter=object(), oos_labels=oos,
                                      sets=("oos_blind",),
                                      sharepoint=BrokenFM())
        printed = capsys.readouterr().out
        assert results["sets"]["oos_blind"]["n_reports"] == 2, "it finished"
        assert (out / "RESULTS.md").is_file()
        # One warning, not one per file per report.
        assert printed.count("the durable mirror failed") == 1
        assert "upload(s) FAILED" in printed

    def test_a_plain_folder_takes_the_run_too(self, cluster, tmp_path):
        reports_dir, out, oos = cluster
        durable = tmp_path / "durable"
        cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                            prompter=object(), oos_labels=oos,
                            sets=("oos_blind",), durable_dir=durable)
        assert (durable / out.name / "RESULTS.md").is_file()
        assert (durable / out.name / "runs" / "R36.json").is_file()

    def test_the_fh_sp_client_object_itself_is_accepted(self, cluster):
        reports_dir, out, oos = cluster
        fm = FakeFM()
        client = type("FhSpClient", (), {"file_manager": fm})()
        cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                            prompter=object(), oos_labels=oos,
                            sets=("oos_blind",), sharepoint=client)
        assert f"{cs.DEFAULT_FOLDER}/{out.name}/RESULTS.md" in fm.tree

    def test_the_remote_folder_can_be_moved(self, cluster):
        reports_dir, out, oos = cluster
        fm = FakeFM()
        cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                            prompter=object(), oos_labels=oos,
                            sets=("oos_blind",), sharepoint=fm,
                            sharepoint_folder="somewhere/else")
        assert f"somewhere/else/{out.name}/RESULTS.md" in fm.tree

    def test_a_workspace_durable_dir_is_the_point_of_the_mirror(self):
        """The workspace folder is where the owner keeps a run, not a trap."""
        assert cs.Mirror(durable_dir="/Workspace/somewhere").active
        assert cs.out_dir_note(Path("/Workspace/Users/x/out"))


# ---------------------------------------------------------------------------
# the vote stage (WP6)
# ---------------------------------------------------------------------------

def _vote_vision(rid, n_pages, rules, seen, rules_confidence,
                 vision_confidence):
    """A saved vision run carrying both voters and both confidences."""
    return {
        "id": rid, "run_date": "2026-09-20", "n_pages": n_pages,
        "model": "funhouse-gpt-low", "served_by": "gpt-4.1-mini",
        "mode": "sheet", "dpi": 100.0, "outline_context": False,
        "detail": None, "fallback": True,
        "rules_labels": {str(k): v for k, v in rules.items()},
        "rules_confidence": {str(k): v
                             for k, v in rules_confidence.items()},
        "vision": {
            "labels": {str(k): v for k, v in seen.items()},
            "detail": [{"page": k, "label": v,
                        "confidence": vision_confidence[k],
                        "reason": "the sheet says so"}
                       for k, v in seen.items()],
            "unresolved": [], "qa": [], "mode": "sheet", "dpi": 100.0,
            "outline_context": False, "pages_asked": n_pages,
            "model_calls": 2, "budget": None, "stopped_on_budget": False,
            "model": "gpt-4.1-mini", "cost": {"mode": "sheet"},
        },
        "cost": {"calls": 2, "input_tokens": 9000, "output_tokens": 400,
                 "cache_read_tokens": 0, "dollars": 0.02, "seconds": 20.0},
        "seconds": 21.0,
    }


#: R36 is the IN-SAMPLE report the trust table is learned on. Six pages,
#: hand-labelled: two the rules win, one vision wins, three they agree on
#: (two right, one wrong). Every number the tests assert falls out of this
#: table by hand.
R36_RULES = {0: "calculation", 1: "calculation", 2: "plan", 3: "figure",
             4: "narrative", 5: "lab_test"}
R36_VISION = {0: "narrative", 1: "narrative", 2: "plan", 3: "plan",
              4: "narrative", 5: "lab_test"}
R36_HAND = {0: "calculation", 1: "calculation", 2: "plan", 3: "plan",
            4: "narrative", 5: "figure"}
R36_RULES_CONF = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.4, 4: 0.9, 5: 0.9}
R36_VISION_CONF = {0: 0.5, 1: 0.5, 2: 0.8, 3: 0.85, 4: 0.8, 5: 0.8}

#: R31 is the OUT-OF-SAMPLE report the table is APPLIED to. Four pages, all
#: four a disagreement, one per branch of every policy.
R31_RULES = {0: "calculation", 1: "figure", 2: "other", 3: "lab_test"}
R31_VISION = {0: "narrative", 1: "photos", 2: "boring_log", 3: "figure"}
R31_HAND = {0: "calculation", 1: "photos", 2: "boring_log", 3: "lab_test"}
R31_RULES_CONF = {0: 0.9, 1: 0.4, 2: 0.4, 3: 0.9}
R31_VISION_CONF = {0: 0.5, 1: 0.8, 2: 0.9, 3: 0.6}


@pytest.fixture()
def vote_cluster(tmp_path, monkeypatch):
    """One in-sample report and one out-of-sample one, both already seen.

    ``_set_ids`` is pinned so the sets are these two reports and nothing
    else, and both reports' hand labels come from the out-of-sample file, so
    the whole fixture is four files and no spreadsheet.
    """
    reports_dir = tmp_path / "corpus"
    reports_dir.mkdir()
    for rid in ("R36", "R31"):
        (reports_dir / f"{rid}.pdf").write_bytes(b"%PDF-1.7\n")
    out = tmp_path / "out_vote"
    (out / "runs").mkdir(parents=True)
    (out / "triage").mkdir()
    (out / "vision").mkdir()

    (out / "vision" / "R36.json").write_text(json.dumps(_vote_vision(
        "R36", 6, R36_RULES, R36_VISION, R36_RULES_CONF, R36_VISION_CONF)),
        encoding="utf-8")
    (out / "vision" / "R31.json").write_text(json.dumps(_vote_vision(
        "R31", 4, R31_RULES, R31_VISION, R31_RULES_CONF, R31_VISION_CONF)),
        encoding="utf-8")

    oos = tmp_path / "oos_labels.json"
    oos.write_text(json.dumps({
        "R36": {str(p): {"label": v, "alternates": []}
                for p, v in R36_HAND.items()},
        "R31": {str(p): {"label": v, "alternates": []}
                for p, v in R31_HAND.items()},
    }), encoding="utf-8")

    pinned = {"insample": ("R36",), "oos_open": ("R31",), "oos_blind": ()}

    def set_ids(name, corpus):
        return pinned[name]

    monkeypatch.setattr(cs, "_set_ids", set_ids)
    return reports_dir, out, oos


def _vote_run(vote_cluster, **over):
    reports_dir, out, oos = vote_cluster
    kwargs = dict(reports_dir=reports_dir, out_dir=out, prompter=object(),
                  oos_labels=oos, sets=("insample", "oos_open"),
                  stages=("vote",))
    kwargs.update(over)
    return cs.score_on_cluster(**kwargs)


class TestTheVoteStage:
    """WP6: what a disagreement between the voters is worth, with no model."""

    def test_it_is_a_stage_name_and_is_not_in_the_default(self, cluster):
        reports_dir, out, oos = cluster
        assert "vote" in cs.STAGE_NAMES
        results = cs.score_on_cluster(reports_dir=reports_dir, out_dir=out,
                                      prompter=object(), oos_labels=oos)
        assert results["stages"] == ["labels"]
        assert "vote" not in results

    def test_it_refuses_to_run_without_a_vision_folder_and_says_where(
            self, vote_cluster):
        reports_dir, out, oos = vote_cluster
        for path in (out / "vision").glob("*.json"):
            path.unlink()
        with pytest.raises(FileNotFoundError) as caught:
            _vote_run(vote_cluster)
        message = str(caught.value)
        assert "vision" in message and str(out / "vision") in message
        assert "vision_dirs" in message

    def test_a_named_vision_folder_is_read_instead(self, vote_cluster,
                                                   tmp_path):
        reports_dir, out, oos = vote_cluster
        elsewhere = tmp_path / "an_earlier_run" / "vision"
        elsewhere.mkdir(parents=True)
        (elsewhere / "R36.json").write_text(
            (out / "vision" / "R36.json").read_text(encoding="utf-8"),
            encoding="utf-8")
        for path in (out / "vision").glob("*.json"):
            path.unlink()
        vote = _vote_run(vote_cluster, vision_dirs=[elsewhere])["vote"]
        assert vote["reports"] == ["R36"]

    def test_no_model_is_ever_asked_for(self, vote_cluster):
        """The prompter is not even touched: a vote is arithmetic on disk."""

        class Explodes:
            def __getattr__(self, name):
                raise AssertionError(f"the vote called the model: {name}")

        _vote_run(vote_cluster, prompter=Explodes())

    # -- 1. agreement ------------------------------------------------------

    def test_the_agreement_rate_and_what_it_is_worth(self, vote_cluster):
        vote = _vote_run(vote_cluster)["vote"]
        agree = vote["sets"]["insample"]["agreement"]
        # Six pages, three of them agreed (2, 4, 5).
        assert agree["pages"] == 6
        assert agree["agree"] == 3
        assert agree["agreement"] == pytest.approx(0.5)
        # Of the three they agreed on, two were right: page 5 is lab_test to
        # both voters and figure to the hand.
        assert agree["agreed"] == {"pages": 3, "correct": 2,
                                   "accuracy": pytest.approx(0.6667)}
        # Of the three they split on, the rules had two and vision one, and
        # between them they had all three.
        split = agree["disagreed"]
        assert split["pages"] == 3
        assert split["rules_correct"] == 2
        assert split["vision_correct"] == 1
        assert split["either_correct"] == 3
        assert split["neither_correct"] == 0
        assert split["ceiling"] == pytest.approx(1.0)

    def test_an_unresolved_page_is_a_disagreement_rather_than_an_excuse(
            self, vote_cluster):
        reports_dir, out, oos = vote_cluster
        blob = json.loads((out / "vision" / "R36.json").read_text(
            encoding="utf-8"))
        blob["vision"]["labels"].pop("4")        # the pass never answered
        (out / "vision" / "R36.json").write_text(json.dumps(blob),
                                                 encoding="utf-8")
        vote = _vote_run(vote_cluster)["vote"]
        rows = json.loads((out / "vote" / "R36.json").read_text(
            encoding="utf-8"))["disagreements"]
        page4 = [r for r in rows if r["page"] == 4]
        assert page4 and page4[0]["vision"] == "other"
        # Page 4 was one of the three the voters agreed on. It is now one of
        # the four they do not, and the accuracy of the agreed pages falls
        # with it -- the page nothing could answer for is exactly the page a
        # second look is for.
        agree = vote["sets"]["insample"]["agreement"]
        assert agree["agree"] == 2
        assert agree["disagreed"]["pages"] == 4

    # -- 2. the trust table ------------------------------------------------

    def test_the_trust_table_is_learned_on_the_in_sample_report_alone(
            self, vote_cluster):
        vote = _vote_run(vote_cluster)["vote"]
        assert vote["trust_learned_on"] == ["R36"]
        table = vote["trust"]
        # R36's two calculation pages: the rules had both, vision neither.
        assert table["calculation"] == {"pages": 2, "rules": 2, "vision": 0,
                                        "winner": "rules"}
        # R36's one figure page: vision had it.
        assert table["figure"] == {"pages": 1, "rules": 0, "vision": 1,
                                   "winner": "vision"}
        # `other` and `lab_test` split only on R31, which is out of sample.
        assert "other" not in table
        assert "lab_test" not in table

    def test_a_class_the_table_never_saw_goes_to_vision(self, vote_cluster):
        _reports_dir, out, _oos = vote_cluster
        _vote_run(vote_cluster)
        rows = {r["page"]: r for r in json.loads(
            (out / "vote" / "R31.json").read_text(encoding="utf-8")
        )["disagreements"]}
        # `other` is not in the table, so trust takes vision's boring_log.
        assert rows[2]["chosen"]["trust"] == "boring_log"
        # `calculation` is, and the table says believe the rules.
        assert rows[0]["chosen"]["trust"] == "calculation"

    # -- 3. the three policies --------------------------------------------

    def test_the_three_policies_are_scored_beside_the_voters(self,
                                                             vote_cluster):
        scores = _vote_run(vote_cluster)["vote"]["sets"]["oos_open"]["scores"]
        # R31: four pages, four disagreements, by hand.
        assert scores["rules"]["accuracy"] == pytest.approx(0.5)
        assert scores["vision"]["accuracy"] == pytest.approx(0.5)
        # trust: calculation -> rules (right), figure -> vision (right),
        # other -> vision (right), lab_test -> vision (wrong).
        assert scores["trust"]["accuracy"] == pytest.approx(0.75)
        # structural: the rules keep calculation, other and lab_test, vision
        # keeps figure. That loses page 2 and keeps page 3.
        assert scores["structural"]["accuracy"] == pytest.approx(0.75)
        # confidence: the more confident voter wins every page, and here
        # that voter is right every time.
        assert scores["confidence"]["accuracy"] == pytest.approx(1.0)

    def test_each_policy_s_chosen_label_is_written_down_per_page(
            self, vote_cluster):
        _reports_dir, out, _oos = vote_cluster
        _vote_run(vote_cluster)
        rows = {r["page"]: r for r in json.loads(
            (out / "vote" / "R31.json").read_text(encoding="utf-8")
        )["disagreements"]}
        assert rows[3]["chosen"] == {"trust": "figure",
                                     "structural": "lab_test",
                                     "confidence": "lab_test"}
        assert rows[3]["rules_confidence"] == pytest.approx(0.9)
        assert rows[3]["vision_confidence"] == pytest.approx(0.6)

    def test_confidence_ties_go_to_the_rules(self):
        from report_ingest.vote import PageVote, combine

        tied = PageVote("R01", 1, "calculation", "narrative",
                        rules_confidence=0.7, vision_confidence=0.7)
        assert combine(tied, "confidence") == "calculation"
        louder = PageVote("R01", 1, "calculation", "narrative",
                          rules_confidence=0.7, vision_confidence=0.71)
        assert combine(louder, "confidence") == "narrative"

    def test_the_review_is_a_third_column_where_a_label_run_sits_beside_it(
            self, vote_cluster):
        _reports_dir, out, _oos = vote_cluster
        final = dict(R36_RULES)
        final[3] = "plan"                       # the review fixed that one
        (out / "runs" / "R36.json").write_text(json.dumps(_run_blob(
            "R36", 6, R36_RULES, final)), encoding="utf-8")
        vote = _vote_run(vote_cluster)["vote"]
        assert vote["n_with_review"] == 1
        scores = vote["sets"]["insample"]["scores"]
        assert scores["review"]["accuracy"] == pytest.approx(0.8333)
        assert "review" not in vote["sets"]["oos_open"]["scores"], (
            "R31 has no label run, so it prints no review column")

    # -- 4. the disagreement set -------------------------------------------

    def test_what_a_targeted_review_of_the_splits_would_have_to_manage(
            self, vote_cluster):
        vote = _vote_run(vote_cluster)["vote"]
        # R31: nothing agreed, four to review, so the review alone has to
        # carry the gate: 0.98 * 4 / 4.
        oos = vote["sets"]["oos_open"]
        assert oos["agreement"]["disagreed"]["fraction"] == pytest.approx(1.0)
        assert oos["required_review_accuracy"] == pytest.approx(0.98)
        # R36: three agreed pages carry two hits, so a review of the other
        # three would have to be better than perfect. It says so.
        insample = vote["sets"]["insample"]
        assert insample["required_review_accuracy"] == pytest.approx(1.2933,
                                                                     abs=1e-4)

    def test_the_results_file_says_a_gate_out_of_reach_is_out_of_reach(
            self, vote_cluster):
        _reports_dir, out, _oos = vote_cluster
        _vote_run(vote_cluster)
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert ">1.000" in text

    # -- 5. the file that comes home ---------------------------------------

    def test_the_results_file_carries_the_section_and_its_five_parts(
            self, vote_cluster):
        _reports_dir, out, _oos = vote_cluster
        _vote_run(vote_cluster)
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "# Vote: rules, vision and the review as voters" in text
        for heading in ("## 1. Agreement",
                        "## 2. Per-label trust",
                        "## 3. The combined labels",
                        "## 4. The disagreement set",
                        "## 5. Per report"):
            assert heading in text, heading
        for column in ("trust", "struct", "conf"):
            assert f"P {column}" in text, column
        assert "No model was called" in text

    def test_the_results_file_carries_no_reason_and_no_confidence(
            self, vote_cluster):
        _reports_dir, out, _oos = vote_cluster
        _vote_run(vote_cluster)
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        # A vision reason names what the model saw on a page and can carry a
        # firm; it stays in the run files on the cluster.
        assert "the sheet says so" not in text

    def test_the_qa_file_lists_every_split_and_nothing_else(self,
                                                            vote_cluster):
        _reports_dir, out, _oos = vote_cluster
        _vote_run(vote_cluster)
        blob = json.loads((out / "vote" / "R36.json").read_text(
            encoding="utf-8"))
        assert blob["pages_compared"] == 6
        assert [r["page"] for r in blob["disagreements"]] == [0, 1, 3]
        assert blob["trust_table_learned_on"] == ["R36"]
        assert blob["rules_from"] == "saved run", (
            "the saved run carries rules_confidence, so no PDF is reopened")
        row = blob["disagreements"][2]
        assert row["rules"] == "figure" and row["vision"] == "plan"
        assert row["hand"] == "plan"
        assert "reason" not in json.dumps(blob)

    def test_the_vote_files_are_mirrored_like_everything_else(self,
                                                              vote_cluster):
        _reports_dir, out, _oos = vote_cluster
        fm = FakeFM()
        _vote_run(vote_cluster, sharepoint=fm)
        remote = f"{cs.DEFAULT_FOLDER}/{out.name}"
        assert f"{remote}/vote/R36.json" in fm.tree
        assert f"{remote}/RESULTS.md" in fm.tree

    def test_the_results_json_drops_the_live_scorers(self, vote_cluster):
        _reports_dir, out, _oos = vote_cluster
        _vote_run(vote_cluster)
        blob = json.loads((out / "results.json").read_text(encoding="utf-8"))
        assert blob["stages"] == ["vote"]
        assert "_scores" not in blob["vote"]["sets"]["insample"]
        assert blob["vote"]["sets"]["oos_open"]["scores"]["confidence"][
            "accuracy"] == pytest.approx(1.0)

    def test_the_blind_set_is_reported_without_a_per_report_line(
            self, vision_cluster):
        """The honest-blind row appears, and no report is named beside it."""
        reports_dir, out, oos = vision_cluster
        results = cs.score_on_cluster(
            reports_dir=reports_dir, out_dir=out, prompter=object(),
            oos_labels=oos, sets=("oos_blind",), stages=("vote",))
        assert "honest_blind" in results["vote"]["sets"]
        # R36 sat in the cost checkpoint; R31 never did.
        assert results["vote"]["sets"]["honest_blind"]["reports"] == ["R31"]
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "## 5. Per report" not in text
        assert "stops being blind" in text

    def test_a_run_file_without_a_confidence_says_so_rather_than_inventing(
            self, vote_cluster):
        """A pre-5.22.0 vision run has labels and no confidence. planlens is
        asked again for it; where the PDF will not open (these are stubs),
        the saved labels stand and the confidence policy simply has nothing
        to weigh."""
        _reports_dir, out, _oos = vote_cluster
        blob = json.loads((out / "vision" / "R31.json").read_text(
            encoding="utf-8"))
        blob.pop("rules_confidence")
        (out / "vision" / "R31.json").write_text(json.dumps(blob),
                                                 encoding="utf-8")
        _vote_run(vote_cluster)
        qa = json.loads((out / "vote" / "R31.json").read_text(
            encoding="utf-8"))
        assert qa["rules_from"] == "saved run, no confidence"
        assert "rules_confidence" not in qa["disagreements"][0]


# -- the ingest stage: the whole pipeline, per report --------------------------

@pytest.fixture()
def ingest_cluster(tmp_path, monkeypatch):
    """One synthetic report on disk, and a fake engine in the Prompter's place.

    The stage builds its engine through ``report_ingest.engine.PrompterEngine``
    at call time, so that name is patched to hand back a ``FakeEngine``
    replaying whatever script the test put in ``scripts["turns"]``. No
    model, no credential, no network; the whole graph runs for real.
    """
    from report_ingest.tests import test_graph as tg
    from report_ingest.tests.fake_engine import FakeEngine
    import report_ingest.engine as engine_module

    reports_dir = tmp_path / "corpus"
    reports_dir.mkdir()
    (reports_dir / "R36.pdf").write_bytes(tg.build_narrative_report().pdf)
    out = tmp_path / "out_ingest"
    monkeypatch.setattr(
        cs, "_set_ids",
        lambda name, corpus: ("R36",) if name == "oos_open" else ())
    # Two engines per report, as in production: the readers on the tier
    # asked for, and the page-label vision voter on the cheap one. Each
    # replays its own script.
    scripts = {"turns": [], "vision": tg.vision_turns(), "engines": []}

    def fake_engine(prompter, model, meter=None, **kwargs):
        turns = (scripts["vision"] if model == "funhouse-gpt-low"
                 else scripts["turns"])
        engine = FakeEngine(list(turns or []), name=model)
        engine.served_by = "fake-deployment"
        scripts["engines"].append(engine)
        return engine

    monkeypatch.setattr(engine_module, "PrompterEngine", fake_engine)
    return reports_dir, out, scripts


def _ingest_run(ingest_cluster, **over):
    reports_dir, out, _scripts = ingest_cluster
    kwargs = dict(reports_dir=reports_dir, out_dir=out, prompter=object(),
                  sets=("oos_open",), stages=("ingest",))
    kwargs.update(over)
    return cs.score_on_cluster(**kwargs)


class TestTheIngestStage:

    def test_it_is_a_stage_name_and_not_in_the_default(self):
        assert "ingest" in cs.STAGE_NAMES
        assert cs.STAGE_NAMES[-1] == "ingest"

    def test_the_whole_pipeline_runs_and_leaves_the_record_and_its_exports(
            self, ingest_cluster):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        results = _ingest_run(ingest_cluster)

        folder = out / "ingest" / "R36"
        for name in ("report.record.json", "report.summary.md",
                     "report.page.md", "report.diggs.xml", "qa.json",
                     "run.json", "triage.json", "vision.json", "labels.json"):
            assert (folder / name).is_file(), name
        assert (out / "ingest" / "reports.db").is_file()
        assert (folder / "items").is_dir()
        ingest = results["ingest"]
        assert ingest["n_reports"] == 1 and ingest["failures"] == {}
        (row,) = ingest["per_report"]
        assert row["id"] == "R36" and row["n_pages"] == 22
        assert row["workflow"] == "standard"
        assert row["counts"]["investigations"] == 2
        assert row["investigations_by_kind"] == {"boring": 1, "test_pit": 1}
        assert row["lab_by_kind"] == {"atterberg": 1, "gradation": 1}
        assert row["narrative"]["answered"] > 0
        assert row["diggs"]["written"] == "yes"
        assert row["diggs"]["schema"] in ("valid", "not run")
        assert row["diggs"]["roundtrip"] == "equal"
        assert row["qa_by_kind"].get("disagreement", 0) >= 1
        assert row["triage_reused"] is False and row["review_reused"] is False
        assert scripts["engines"][0].n_calls == len(tg.main_script())
        assert scripts["engines"][1].n_calls == len(tg.vision_turns())
        # The voters agreed on every page, so the review never ran.
        assert not (folder / "review.json").exists()
        labels = row["labels"]
        assert labels["policy"] == "structural"
        assert labels["review_mode"] == "disagreements"
        assert labels["split"] == 0 and labels["review_changed"] == 0
        assert labels["pages_voted"] == 22

    def test_results_md_carries_the_section_and_names_nobody(
            self, ingest_cluster):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _ingest_run(ingest_cluster)
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "# Ingest: the record and its exports" in text
        assert "## Per report" in text and "## Totals" in text
        assert "## The page labels: the vote, and what went to the review" \
            in text
        assert "R36" in text and "1 boring, 1 test_pit" in text
        assert "yes/valid/equal" in text or "yes/not run/equal" in text
        assert "Rosewood" not in text          # the project name stays out
        blob = json.loads((out / "results.json").read_text(encoding="utf-8"))
        assert blob["ingest"]["per_report"][0]["id"] == "R36"

    def test_a_second_call_is_free(self, ingest_cluster, capsys):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _ingest_run(ingest_cluster)
        scripts["turns"] = []
        scripts["vision"] = []
        results = _ingest_run(ingest_cluster)
        assert "R36: already done, skipping" in capsys.readouterr().out
        assert results["ingest"]["n_reports"] == 1
        # Two engines for the one report that ran, and none for the second
        # call: the reader tier and the vision voter's cheaper one.
        assert len(scripts["engines"]) == 2

    def test_a_saved_label_run_is_reused_so_the_review_is_not_paid_twice(
            self, ingest_cluster):
        from planlens.document import open_document
        from planlens.document.roles import page_roles
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        with open_document(str(reports_dir / "R36.pdf")) as doc:
            rules = {r.page: r.role for r in page_roles(doc)}
        (out / "runs").mkdir(parents=True)
        (out / "runs" / "R36.json").write_text(json.dumps(
            _run_blob("R36", 22, rules, rules)), encoding="utf-8")
        # No triage turn and no review turns: both come off the saved run.
        # The identity call for the report bound in at pages 15-18 and that
        # report's own two readers are NOT saved by a label run and are paid.
        scripts["turns"] = ([tg.identity_turn(), tg.narrative_turn()]
                            + [tg.log_turn("B-1"),
                               tg.log_turn("TP-1", "test_pit")]
                            + [tg.lab_turn("atterberg", 12),
                               tg.lab_turn("gradation", 13)]
                            + [tg.calc_turn()]
                            + tg.bound_reader_turns())
        results = _ingest_run(ingest_cluster)
        (row,) = results["ingest"]["per_report"]
        assert row["triage_reused"] is True and row["review_reused"] is True
        assert scripts["engines"][0].n_calls == len(scripts["turns"])
        review = json.loads((out / "ingest" / "R36" / "review.json")
                            .read_text(encoding="utf-8"))
        assert review["reused_from"].endswith("R36.json")
        assert row["counts"]["investigations"] == 2

    def test_the_record_is_scored_where_hand_truth_exists(
            self, ingest_cluster, tmp_path):
        from report_ingest.tests import test_graph as tg
        from report_ingest.tests.narrative_fixtures import build_narrative_report
        reports_dir, out, scripts = ingest_cluster
        gt = build_narrative_report()
        truth = tmp_path / "truth"
        (truth / "narrative").mkdir(parents=True)
        (truth / "logs").mkdir()
        (truth / "narrative" / "R36.json").write_text(json.dumps({
            "id": "R36", "general": gt.general,
            "natural_hazards": gt.natural_hazards}), encoding="utf-8")
        (truth / "logs" / "R36_p7.json").write_text(json.dumps({
            "id": "R36_p7", "pages": [7, 8], "depth_unit": "m",
            "fields": {"boring_id": "B-1"},
            "layers": [{"top": 0.0, "uscs": "CL"}],
            "samples": [{"top": 1.5, "blows": [4, 6, 8]}],
            "water": []}), encoding="utf-8")
        scripts["turns"] = tg.main_script()
        results = _ingest_run(ingest_cluster, truth_dir=truth)
        (row,) = results["ingest"]["per_report"]
        (log_score,) = row["scores"]["logs"]
        assert log_score["id"] == "R36_p7"
        assert log_score["overall"]["found"] >= 3
        assert log_score["records"] == ["parent"]
        assert row["scores"]["narrative"]["recall"]["total"] > 0
        assert row["scores"]["lab"] == []
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "## Scored against the hand truth" in text
        totals = results["ingest"]["totals"]["scores"]
        assert totals["logs"]["n"] == 1 and totals["narrative"]["n"] == 1

    def test_a_truth_on_a_bound_reports_pages_meets_that_reports_record(
            self, ingest_cluster, tmp_path):
        # BACKLOG RI-14: a page is read on one side of the boundary only, so
        # the log on page 18 (inside the report bound in at 15-18) is in the
        # bound record, not the parent's -- and is scored there.
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        truth = tmp_path / "truth"
        (truth / "logs").mkdir(parents=True)
        (truth / "logs" / "R36_p18.json").write_text(json.dumps({
            "id": "R36_p18", "pages": [18], "depth_unit": "m",
            "fields": {"boring_id": "BH-1"},
            "layers": [{"top": 0.0, "uscs": "CL"}],
            "samples": [{"top": 1.5, "blows": [3, 5, 7]}],
            "water": []}), encoding="utf-8")
        scripts["turns"] = tg.main_script()
        results = _ingest_run(ingest_cluster, truth_dir=truth)
        (row,) = results["ingest"]["per_report"]
        assert row["investigations_by_kind"] == {"boring": 1, "test_pit": 1}
        (log_score,) = row["scores"]["logs"]
        assert log_score["overall"]["found"] >= 3
        # And the row says where it met it, so a rise is not read as a
        # reader reading better.
        assert log_score["records"] == ["bound1"]
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "R36 R36_p18 (bound1)" in text

    def test_the_ri14_switches_reach_the_graph_and_the_run_file(
            self, ingest_cluster):
        from report_ingest.model import ReportRecord
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        results = _ingest_run(ingest_cluster,
                              ingest_counts_from_logs="zeros",
                              ingest_narrative_skips_bound=True)
        (row,) = results["ingest"]["per_report"]
        assert row["switches"] == {"ingest_bound": True,
                                   "lab_sheets_per_call": None,
                                   "counts_from_logs": "zeros",
                                   "narrative_skips_bound": True}
        folder = out / "ingest" / "R36"
        record = ReportRecord.model_validate_json(
            (folder / "report.record.json").read_text(encoding="utf-8"))
        assert record.general.cptCount == 0
        assert (folder / "items" / "item_2-skips-bound.json").is_file()

    def test_by_default_the_ri14_switches_are_off(self, ingest_cluster):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        (row,) = _ingest_run(ingest_cluster)["ingest"]["per_report"]
        assert row["switches"]["counts_from_logs"] == "off"
        assert row["switches"]["narrative_skips_bound"] is False

    def test_a_report_the_folder_does_not_have_is_named_and_skipped(
            self, ingest_cluster, monkeypatch):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        monkeypatch.setattr(
            cs, "_set_ids",
            lambda name, corpus: ("R36", "R31") if name == "oos_open" else ())
        scripts["turns"] = tg.main_script()
        results = _ingest_run(ingest_cluster)
        assert "R31" in results["absent"]
        assert results["ingest"]["n_reports"] == 1

    def test_the_run_files_are_mirrored_like_everything_else(
            self, ingest_cluster, tmp_path):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        durable = tmp_path / "durable"
        _ingest_run(ingest_cluster, durable_dir=durable)
        assert (durable / out.name / "ingest" / "R36" / "run.json").is_file()
        assert (durable / out.name / "ingest" / "R36"
                / "report.diggs.xml").is_file()


# -- failed requests and reasoning in the cost block (BACKLOG RI-21) -----------

def _rewrite(path, change):
    blob = json.loads(path.read_text(encoding="utf-8"))
    change(blob)
    path.write_text(json.dumps(blob), encoding="utf-8")


class TestFailedRequestsInTheCost:

    def test_run_files_from_before_say_not_recorded_rather_than_zero(
            self, logs_cluster):
        _reports_dir, out, _truth = logs_cluster
        logs = _logs_run(logs_cluster)["logs"]
        assert logs["cost"]["files"] == 2
        assert logs["cost"]["files_counting_failures"] == 0
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "failed requests: not recorded" in text

    def test_failed_and_resent_requests_are_totalled_and_printed(
            self, logs_cluster):
        _reports_dir, out, _truth = logs_cluster

        def failed(blob):
            blob["cost"].update({
                "failed": 3, "retried": 2, "reasoning_tokens": 1200,
                "reasoning_reported": 1, "waited_seconds": 45.0,
                "failed_by_kind": {"rate_limit": 2, "timeout": 1},
                "per_call": [{"call": 1, "attempt": 1}]})

        for name in ("R36_p38.json", "R31_p278.json"):
            _rewrite(out / "logs" / name, failed)
        cost = _logs_run(logs_cluster)["logs"]["cost"]
        assert (cost["failed"], cost["retried"]) == (6, 4)
        assert cost["reasoning_tokens"] == 2400
        assert cost["failed_by_kind"] == {"rate_limit": 4, "timeout": 2}
        assert cost["calls"] == 2              # replies only, as before
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert ("failed requests, not among the calls above: 6 (4 rate "
                "limited, 2 timed out), 4 of them sent again by the engine, "
                "90 s waiting to send again; reasoning: 2,400 of the output "
                "tokens") in text

    def test_a_mix_of_old_and_new_run_files_says_how_many_counted(self):
        total = cs._cost_totals()
        cs._add_cost(total, {"calls": 4})
        cs._add_cost(total, {"calls": 5, "failed": 0, "retried": 0})
        line = cs._failure_line(total)
        assert line.startswith("failed requests, not among the calls "
                               "above: 0, 0 of them sent again")
        assert "counted in 1 of 2 run files" in line

    def test_the_ingest_results_keep_totals_not_the_row_per_request(self):
        row = {"id": "R36", "n_pages": 3, "seconds": 1.0,
               "cost": {"calls": 2, "failed": 1, "retried": 1,
                        "per_call": [{"call": 1}, {"call": 2}]}}
        scored = cs._score_ingest({"R36": row}, {}, "m")
        (kept,) = scored["per_report"]
        assert "per_call" not in kept["cost"]
        assert kept["cost"]["failed"] == 1
        assert "per_call" in row["cost"]       # the run file is untouched
        assert scored["cost"]["failed"] == 1


# -- the vote stage needs no prompter (BACKLOG RI-08) ---------------------------

class TestTheVoteNeedsNoPrompter:

    def test_the_vote_alone_runs_with_none(self, vote_cluster):
        results = _vote_run(vote_cluster, prompter=None)
        assert "vote" in results

    def test_a_stage_that_calls_the_model_still_needs_one(self, tmp_path):
        with pytest.raises(ValueError, match=r"\['labels'\] call the model"):
            cs.score_on_cluster(reports_dir=tmp_path, prompter=None,
                                stages=("vote", "labels"))


# -- what the visual-scales voters did, in RESULTS.md (BACKLOG RI-05, G3) -----

class TestTheVisualScalesBlock:

    def test_old_run_files_print_no_block(self, logs_cluster):
        _reports_dir, out, _truth = logs_cluster
        logs = _logs_run(logs_cluster)["logs"]
        assert logs["visual_scales"]["files_not_recorded"] == 2
        assert "## Visual scales" not in (out / "RESULTS.md").read_text(
            encoding="utf-8")

    def test_the_log_votes_and_page_sources_are_counted(self, logs_cluster):
        _reports_dir, out, _truth = logs_cluster

        def scanned(blob):
            blob["visual_scales_on"] = True
            blob["page_sources"] = {"pages": 1, "raster": 1, "di_text": 1}
            blob["after"]["visual_scales"] = {
                "visual_scales": True, "label_calls": 1, "pixel_layers": 3,
                "layer_votes": [{"agreement": "agree"},
                                {"agreement": "disagree"},
                                {"agreement": "no_record_layer"}]}

        def off(blob):
            blob["visual_scales_on"] = False
            blob["page_sources"] = {"pages": 1, "raster": 0, "di_text": 0}

        _rewrite(out / "logs" / "R36_p38.json", scanned)
        _rewrite(out / "logs" / "R31_p278.json", off)
        counts = _logs_run(logs_cluster)["logs"]["visual_scales"]
        assert (counts["files"], counts["files_on"]) == (2, 1)
        assert (counts["pages"], counts["raster"], counts["di_text"]) == \
            (2, 1, 1)
        assert counts["label_calls"] == 1 and counts["pixel_layers"] == 3
        assert counts["layer_votes"] == {"agree": 1, "disagree": 1,
                                         "no_record_layer": 1}
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        block = text.split("## Visual scales")[1].split("## Cost")[0]
        assert "on for 1 of 2 run files" in block
        assert "pages read: 2, of them 1 scans and 1 with their text" in block
        assert "label-reading calls: 1" in block
        assert "1 agree, 1 disagree, 1 with no record layer" in block

    def test_plot_and_trace_checks_are_counted(self):
        counts = cs._visual_counts()
        cs._add_checks(counts, plot={
            "status": "checked", "label_calls": 1, "table_points": 9,
            "agree": 6, "disagree": [{"sieve": "#200"}],
            "no_marker_at_size": 2})
        cs._add_checks(counts, plot={"status": "no_plot", "label_calls": 0})
        cs._add_checks(counts, trace={
            "status": "checked", "label_calls": 2, "channels": {
                "qc": {"compared": 5, "agree": 4, "disagree": [{"depth": 3}],
                       "several": 1, "no_trace": 2},
                "fs": {"status": "no_panel"}}})
        assert counts["label_calls"] == 3
        assert counts["plot_checks"] == {"checked": 1, "no_plot": 1,
                                         "unavailable": 0}
        assert counts["plot_points"] == {"table": 9, "agree": 6,
                                         "disagree": 1,
                                         "no_marker_at_size": 2}
        assert counts["trace_points"] == {"compared": 5, "agree": 4,
                                          "disagree": 1, "several": 1,
                                          "no_trace": 2}
        assert counts["trace_channels_no_panel"] == 1
        counts.update(files=1, files_on=1)
        text = "\n".join(cs._visual_lines(counts))
        assert "grading curves checked against the table: 1" in text
        assert "plotted soundings traced by code: 1" in text

    def test_the_scores_carry_what_the_voters_recorded(self):
        from report_ingest.lab_scoring import LabScore
        from report_ingest.log_scoring import LogScore
        from report_ingest.sounding_scoring import SoundingScore

        plain = LogScore(log_id="R36_p38", report="R36", stage="record")
        assert "visual_scales" not in plain.to_dict()
        log = LogScore(log_id="R36_p38", report="R36", stage="record",
                       visual_scales={"label_calls": 1})
        assert log.to_dict()["visual_scales"] == {"label_calls": 1}
        lab = LabScore(sheet_id="gradation__R36_p40", report="R36",
                       kind="gradation", stage="record",
                       plot_check={"status": "no_plot"})
        assert lab.to_dict()["plot_check"] == {"status": "no_plot"}
        sounding = SoundingScore(sounding_id="cpt__R36_p50", report="R36",
                                 kind="cpt", stage="record",
                                 code_trace={"status": "checked"})
        assert sounding.to_dict()["code_trace"] == {"status": "checked"}
        assert "visual_scales" not in sounding.to_dict()


# -- calls by step, per document (BACKLOG RI-21) -------------------------------

class TestCallsByStep:

    @staticmethod
    def _write(path, blob):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob), encoding="utf-8")

    def test_the_report_and_each_bound_report_are_counted_apart(
            self, tmp_path):
        folder = tmp_path / "R36"
        self._write(folder / "triage.json", {"cost": {"calls": 1}})
        self._write(folder / "vision.json", {"cost": {"calls": 4}})
        self._write(folder / "items" / "a.json",
                    {"calculation": {}, "cost": {"calls": 2}})
        self._write(folder / "items" / "b.json",
                    {"calculation": {}, "cost": {"calls": 1}})
        self._write(folder / "items" / "c.json",
                    {"general": {}, "cost": {"calls": 1}})
        self._write(folder / "bound" / "bound1" / "identity.json",
                    {"cost": {"calls": 1}})
        self._write(folder / "bound" / "bound1" / "items" / "d.json",
                    {"investigation": {"kind": "boring"}, "cost": {"calls": 2},
                     "printed_depths": [],
                     "visual_scales": {"label_calls": 1, "pixel_layers": 2}})
        self._write(folder / "bound" / "bound1" / "items" / "e.json",
                    {"investigation": {"kind": "cpt"}, "cost": {"calls": 1},
                     "tabulated": False, "floor_points": 0,
                     "code_trace": {"status": "no_plot", "label_calls": 0}})
        self._write(folder / "bound" / "bound1" / "items" / "f.json",
                    {"tests": [], "cost": {"calls": 1},
                     "plot_check": {"status": "unavailable"}})

        got = cs._ingest_steps(folder)
        steps = got["calls_by_step"]
        assert steps["report"] == {"triage": 1, "vision": 4, "calc": 3,
                                   "calc items": 2, "narrative": 1,
                                   "narrative items": 1}
        assert steps["bound1"] == {"identity": 1, "log": 2, "log items": 1,
                                   "sounding": 1, "sounding items": 1,
                                   "lab": 1, "lab items": 1}
        visual = got["visual_scales"]
        assert visual["label_calls"] == 1 and visual["pixel_layers"] == 2
        assert visual["trace_checks"]["no_plot"] == 1
        assert visual["plot_checks"]["unavailable"] == 1

        # The report's cost counts two calls no file holds (a reading that
        # failed keeps no file): they show as unattributed, on its row.
        row = {"id": "R36", "visual_scales_on": True,
               "cost": {"calls": 16}, **got}
        text = "\n".join(cs._step_lines([row]))
        assert "## Calls by step" in text
        lines = text.splitlines()
        report_line = next(line for line in lines
                           if line.startswith("R36     report"))
        assert report_line.split()[2:] == ["1", "4", "-", "-", "-", "-",
                                           "-", "3/2", "1/1", "9", "+2"]
        bound_line = next(line for line in lines
                          if line.startswith("R36     bound1"))
        assert bound_line.split()[-1] == "5", "no unattributed cell"

    def _two_narratives(self, tmp_path, newer_is_plain):
        import os
        folder = tmp_path / "R90"
        self._write(folder / "items" / "n1.json",
                    {"general": {}, "cost": {"calls": 4}})
        self._write(folder / "items" / "n1-skips-bound.json",
                    {"general": {}, "cost": {"calls": 2}})
        self._write(folder / "items" / "c.json",
                    {"calculation": {}, "cost": {"calls": 1}})
        newer, older = (("n1", "n1-skips-bound") if newer_is_plain
                        else ("n1-skips-bound", "n1"))
        os.utime(folder / "items" / f"{older}.json", (1_000_000, 1_000_000))
        os.utime(folder / "items" / f"{newer}.json", (2_000_000, 2_000_000))
        return folder

    def test_a_reread_narrative_is_counted_once_by_the_recorded_key(
            self, tmp_path):
        # The newer file is the plain one, but the run says the bound
        # window was skipped: the key, not the file time, picks the file.
        folder = self._two_narratives(tmp_path, newer_is_plain=True)
        got = cs._ingest_steps(folder, True)
        row = got["calls_by_step"]["report"]
        assert row["narrative"] == 2 and row["narrative items"] == 1
        assert got["superseded_item_files"] == {"report": {"narrative": {
            "files": 1, "calls": 4, "chosen_by": "key"}}}
        text = "\n".join(cs._step_lines([{"id": "R90", "cost": {"calls": 3},
                                          **got}]))
        assert "1 older narrative item file(s)" in text
        assert "(4 calls)" in text
        assert "picked by the run's recorded narrative_skips_bound" in text

    def test_the_switch_off_picks_the_plain_file(self, tmp_path):
        folder = self._two_narratives(tmp_path, newer_is_plain=False)
        row = cs._ingest_steps(folder, False)["calls_by_step"]["report"]
        assert row["narrative"] == 4

    def test_no_recorded_switch_falls_back_to_file_time_and_says_so(
            self, tmp_path):
        folder = self._two_narratives(tmp_path, newer_is_plain=True)
        got = cs._ingest_steps(folder)
        assert got["calls_by_step"]["report"]["narrative"] == 4
        assert got["superseded_item_files"]["report"]["narrative"][
            "chosen_by"] == "mtime"
        text = "\n".join(cs._step_lines([{"id": "R90", "cost": {"calls": 5},
                                          **got}]))
        assert "newest by file time" in text

    def test_one_narrative_file_has_nothing_superseded(self, tmp_path):
        folder = tmp_path / "R91"
        self._write(folder / "items" / "n.json",
                    {"general": {}, "cost": {"calls": 3}})
        got = cs._ingest_steps(folder, True)
        assert "superseded_item_files" not in got
        assert got["calls_by_step"]["report"]["narrative items"] == 1

    def _lab_folder(self, tmp_path):
        folder = tmp_path / "R92"
        self._write(folder / "items" / "item_7.json",
                    {"tests": [], "cost": {"calls": 2}})
        self._write(folder / "items" / "item_7-per3.json",
                    {"tests": [], "cost": {"calls": 5}})
        self._write(folder / "items" / "item_8.json",
                    {"tests": [], "cost": {"calls": 1}})
        return folder

    def test_a_lab_item_saved_whole_and_split_is_counted_once(
            self, tmp_path):
        folder = self._lab_folder(tmp_path)
        got = cs._ingest_steps(folder, None, 3)
        row = got["calls_by_step"]["report"]
        assert row["lab"] == 6 and row["lab items"] == 2
        assert got["superseded_item_files"]["report"]["lab"] == {
            "files": 1, "calls": 2, "chosen_by": "lab_sheets_per_call"}
        text = "\n".join(cs._step_lines([{"id": "R92", "cost": {"calls": 6},
                                          **got}]))
        assert "1 lab item file(s) of the other reading" in text
        assert "(2 calls)" in text

    def test_with_the_setting_off_the_whole_lab_file_is_the_one(
            self, tmp_path):
        folder = self._lab_folder(tmp_path)
        got = cs._ingest_steps(folder)
        assert got["calls_by_step"]["report"]["lab"] == 3
        assert got["superseded_item_files"]["report"]["lab"]["calls"] == 5

    def test_a_lone_whole_lab_file_is_never_superseded(self, tmp_path):
        folder = tmp_path / "R93"
        self._write(folder / "items" / "item_8.json",
                    {"tests": [], "cost": {"calls": 1}})
        got = cs._ingest_steps(folder, None, 3)
        assert "superseded_item_files" not in got
        assert got["calls_by_step"]["report"]["lab items"] == 1

    def test_a_log_item_read_whole_beside_its_parts_is_counted_apart(
            self, tmp_path):
        # A folder reused from a run before RI-117: the log item read whole
        # (item_4) beside the two parts it is now cut into, one per hole.
        folder = tmp_path / "R94"
        self._write(folder / "items" / "item_4.json",
                    {"investigation": {}, "cost": {"calls": 3}})
        self._write(folder / "items" / "item_4a.json",
                    {"investigation": {}, "cost": {"calls": 2}})
        self._write(folder / "items" / "item_4b.json",
                    {"investigation": {}, "cost": {"calls": 1}})
        self._write(folder / "items" / "item_5.json",
                    {"investigation": {}, "cost": {"calls": 1}})
        got = cs._ingest_steps(folder)
        row = got["calls_by_step"]["report"]
        assert row["log"] == 4 and row["log items"] == 3
        assert got["superseded_item_files"]["report"]["log"] == {
            "files": 1, "calls": 3, "chosen_by": "split_parts"}
        text = "\n".join(cs._step_lines([{"id": "R94", "cost": {"calls": 4},
                                          **got}]))
        assert "1 log item file(s) read whole before the item was cut" in text
        assert "(3 calls)" in text

    def test_a_log_item_with_its_bound_pages_handed_over_is_cut_alike(
            self, tmp_path):
        # graph.py keys an item whose bound pages were handed over "-own",
        # the whole item and its parts alike: item_7-own beside item_7a-own
        # and item_7b-own is the same reuse as item_7 beside its parts.
        folder = tmp_path / "R96"
        self._write(folder / "items" / "item_7-own.json",
                    {"investigation": {}, "cost": {"calls": 3}})
        self._write(folder / "items" / "item_7a-own.json",
                    {"investigation": {}, "cost": {"calls": 2}})
        self._write(folder / "items" / "item_7b-own.json",
                    {"investigation": {}, "cost": {"calls": 1}})
        got = cs._ingest_steps(folder)
        row = got["calls_by_step"]["report"]
        assert row["log"] == 3 and row["log items"] == 2
        assert got["superseded_item_files"]["report"]["log"] == {
            "files": 1, "calls": 3, "chosen_by": "split_parts"}

    def test_a_part_without_the_wholes_suffix_is_not_its_part(
            self, tmp_path):
        # item_8a-own is not a part of item_8 (a reading before the bound
        # pages were handed over): the suffix must match.
        folder = tmp_path / "R97"
        self._write(folder / "items" / "item_8.json",
                    {"investigation": {}, "cost": {"calls": 3}})
        self._write(folder / "items" / "item_8a-own.json",
                    {"investigation": {}, "cost": {"calls": 2}})
        got = cs._ingest_steps(folder)
        assert "superseded_item_files" not in got
        assert got["calls_by_step"]["report"]["log items"] == 2

    def test_a_calc_item_beside_its_parts_is_not_a_log(self, tmp_path):
        folder = tmp_path / "R95"
        self._write(folder / "items" / "item_6.json",
                    {"calculation": {}, "cost": {"calls": 1}})
        self._write(folder / "items" / "item_6a.json",
                    {"investigation": {}, "cost": {"calls": 1}})
        got = cs._ingest_steps(folder)
        assert "superseded_item_files" not in got

    def test_the_ingest_stage_writes_them_into_its_run_file(
            self, ingest_cluster):
        from report_ingest.tests import test_graph as tg
        _reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        results = _ingest_run(ingest_cluster)
        run = json.loads((out / "ingest" / "R36" / "run.json").read_text(
            encoding="utf-8"))
        assert "report" in run["calls_by_step"]
        assert run["visual_scales_on"] is False
        assert results["ingest"]["visual_scales"]["files"] == 1
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "## Calls by step" in text


def test_page_sources_count_the_scans_and_never_fail_a_run():
    pytest.importorskip("pymupdf")
    fixtures = pytest.importorskip("planlens.testing.visual_scale_fixtures")
    scan = fixtures.build_log(fixtures.LogVariant("ri_scan", skew_deg=0.4,
                                                  seed=101))
    assert cs._page_sources(scan.open(), [0]) == {
        "pages": 1, "raster": 1, "di_text": 0}
    assert cs._page_sources(object(), [3, 4]) == {
        "pages": 2, "raster": 0, "di_text": 0}


# -- the RI-21 review's fixes ------------------------------------------------

class TestReasoningNotReported:

    def test_no_reply_that_reported_reasoning_says_not_reported(self):
        total = cs._cost_totals()
        cs._add_cost(total, {"calls": 3, "failed": 0, "retried": 0,
                             "reasoning_tokens": 0, "reasoning_reported": 0})
        assert cs._failure_line(total).endswith("reasoning: not reported")

    def test_some_replies_reporting_says_how_many(self):
        total = cs._cost_totals()
        cs._add_cost(total, {"calls": 3, "failed": 0, "retried": 0,
                             "reasoning_tokens": 900,
                             "reasoning_reported": 2})
        assert cs._failure_line(total).endswith(
            "reasoning: 900 of the output tokens, in the 2 of 3 calls that "
            "reported it")


class TestTheRunFileWriting:

    def test_the_rows_are_one_per_line_and_read_back_whole(self):
        blob = {"id": "R36", "cost": {
            "calls": 2, "per_call_defaults": {"model": "m"},
            "per_call": [{"call": 1, "outcome": "reply"},
                         {"call": 2, "outcome": "error",
                          "error": "RuntimeError: a \"quoted\" word"}]},
            "seconds": 1.0}
        text = cs._dumps(blob)
        assert json.loads(text) == blob
        lines = text.splitlines()
        assert '      {"call":1,"outcome":"reply"},' in lines
        assert any(line.startswith('      {"call":2,') for line in lines)

    def test_a_run_file_with_no_rows_is_written_as_before(self):
        blob = {"id": "R36", "cost": {"calls": 0}}
        assert cs._dumps(blob) == json.dumps(blob, indent=2)


class TestCallsByStepNeverFailsARun:

    def test_odd_files_are_passed_over_and_counted(self, tmp_path):
        folder = tmp_path / "R36"
        items = folder / "items"
        items.mkdir(parents=True)
        (items / "a.json").write_text("[1, 2]", encoding="utf-8")
        (items / "b.json").write_text("{not json", encoding="utf-8")
        (items / "c.json").write_text(json.dumps({
            "investigation": {}, "cost": {"calls": "two"},
            "visual_scales": {"label_calls": "x",
                              "layer_votes": ["agree", None]},
            "plot_check": ["checked"]}), encoding="utf-8")
        (folder / "triage.json").write_text(json.dumps({"cost": "free"}),
                                            encoding="utf-8")
        got = cs._ingest_steps(folder)
        assert got["calls_by_step"]["report"] == {"log": 0, "log items": 1}
        assert got["step_files_unreadable"] == 2
        assert got["visual_scales"]["label_calls"] == 0

    def test_a_count_that_raises_still_returns(self, tmp_path, monkeypatch):
        folder = tmp_path / "R36"
        (folder / "items").mkdir(parents=True)
        (folder / "items" / "a.json").write_text(
            json.dumps({"tests": [], "cost": {"calls": 1}}),
            encoding="utf-8")

        def broken(blob):
            raise KeyError("an item shape nobody expected")

        monkeypatch.setattr(cs, "_item_step", broken)
        got = cs._ingest_steps(folder)
        assert got["step_files_unreadable"] == 1
        assert got["calls_by_step"] == {"report": {}}

    def test_a_log_read_of_a_cone_sounding_is_still_the_log_reader(self):
        """A log-reader item is a log-reader item, whatever the kind of
        investigation it read; the sounding reader's result says whether
        the sheet was tabulated, and the log reader's never does."""
        assert cs._item_step({"investigation": {"kind": "cpt"},
                              "printed_depths": []}) == "log"
        assert cs._item_step({"investigation": {"kind": "boring"},
                              "tabulated": True}) == "sounding"
        assert cs._item_step({"investigation": {"kind": "dcp"},
                              "floor_points": 3}) == "sounding"


def test_a_reused_label_run_carries_its_changes_and_their_evidence(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "R01.json").write_text(json.dumps({"review": {
        "final_labels": {"3": "figure", "4": "plan"},
        "changes": [
            {"page": 3, "from": "other", "to": "figure",
             "evidence": "render_page"},
            {"page": 4, "from": "other", "to": "plan", "evidence": "ledger"},
        ]}}), encoding="utf-8")
    folder = tmp_path / "out" / "R01"
    cs._reuse_label_run(folder, runs, "R01")
    saved = json.loads((folder / "review.json").read_text(encoding="utf-8"))
    assert [(c["page"], c["evidence"]) for c in saved["changes"]] == [
        (3, "render_page"), (4, "ledger")]
    assert saved["final_labels"] == {"3": "figure", "4": "plan"}


def test_a_reused_run_without_saved_changes_keeps_only_final_labels(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "R02.json").write_text(json.dumps({"review": {
        "final_labels": {"1": "toc"}}}), encoding="utf-8")
    folder = tmp_path / "out" / "R02"
    cs._reuse_label_run(folder, runs, "R02")
    saved = json.loads((folder / "review.json").read_text(encoding="utf-8"))
    assert "changes" not in saved


# ---------------------------------------------------------------------------
# a chat model in the Prompter's place (BACKLOG RI-67)
# ---------------------------------------------------------------------------

from report_ingest.engine import (  # noqa: E402
    PrompterEngine as _RealPrompterEngine,
)


@pytest.fixture()
def chat_cluster(ingest_cluster, monkeypatch):
    """``ingest_cluster`` for a run on a chat model: the Prompter's engine is
    a class again (``engine_for`` checks against it), and one that fails the
    item it is built for, because a chat-model run must not build one."""
    import report_ingest.engine as engine_module

    class _NoPrompter(_RealPrompterEngine):
        def __init__(self, *args, **kwargs):
            raise AssertionError("a chat-model run built a Prompter engine")

    monkeypatch.setattr(engine_module, "PrompterEngine", _NoPrompter)
    return ingest_cluster


class _Said:
    """A pass's answer as the stage saves it: only ``to_dict`` is read."""

    def __init__(self, blob):
        self._blob = blob

    def to_dict(self):
        return dict(self._blob)


def _stub_label_passes(monkeypatch, handed):
    """Triage and the label review as stubs that keep the engine they were
    handed and call nothing: which engine that is is what is under test."""
    import report_ingest.label_review as label_review
    import report_ingest.triage as triage_module

    blob = _run_blob("R36", 22, {3: "plan"}, {3: "plan"})

    def triage(doc, roles, outline, engine=None, facts=None):
        handed.append(("triage", engine))
        return _Said(blob["profile"])

    def review_labels(doc, roles, outline, profile, engine=None):
        handed.append(("review", engine))
        return _Said(blob["review"])

    monkeypatch.setattr(triage_module, "triage", triage)
    monkeypatch.setattr(label_review, "review_labels", review_labels)


#: The keys a labels run file has always had, in order: a Prompter run
#: writes exactly these and nothing more.
LABEL_RUN_KEYS = ["id", "run_date", "n_pages", "engine", "triage_model",
                  "review_model", "served_by", "rules_labels",
                  "rules_confidence", "profile", "review", "cost", "seconds"]


class TestAChatModelInThePromptersPlace:
    """A LangChain chat model -- on Foundry, Claude through the workspace's
    own model classes -- reads in the Prompter's place, for the readers and
    the vision pass alike; with none passed nothing changes."""

    @pytest.fixture(autouse=True)
    def _langchain(self):
        pytest.importorskip("langchain_core")

    @staticmethod
    def _chat(name, turns=()):
        from report_ingest.tests.chat_fixtures import (
            ScriptedChatModel, from_turns,
        )
        return ScriptedChatModel(replies=from_turns(list(turns)),
                                 model_name=name)

    # -- the Prompter run is the run it always was ---------------------------

    def test_a_prompter_label_run_writes_what_it_always_wrote(
            self, ingest_cluster, monkeypatch):
        reports_dir, out, scripts = ingest_cluster
        handed: list = []
        _stub_label_passes(monkeypatch, handed)

        results = cs.score_on_cluster(
            reports_dir=reports_dir, out_dir=out, prompter=object(),
            sets=("oos_open",), stages=("labels",))

        blob = json.loads((out / "runs" / "R36.json").read_text(
            encoding="utf-8"))
        assert list(blob) == LABEL_RUN_KEYS
        assert blob["engine"] == "prompter"
        assert blob["review_model"] == blob["triage_model"] == \
            "funhouse-gpt-high"
        assert [engine for _, engine in handed] == scripts["engines"]
        assert results["engine"] == "prompter"
        assert "chat_model" not in results
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "triage and label review through Prompter" in text
        assert "Engine:" not in text

    def test_a_prompter_ingest_records_no_engine_fields(self,
                                                        ingest_cluster):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()

        results = _ingest_run(ingest_cluster)

        (row,) = results["ingest"]["per_report"]
        assert row["model"] == "funhouse-gpt-high"
        for key in ("engine", "chat_model", "vision", "triage"):
            assert key not in row, key
        assert [e.name for e in scripts["engines"]] == [
            "funhouse-gpt-high", "funhouse-gpt-low"]
        assert "engine" not in results["ingest"]
        assert "vision_model" not in results["ingest"]

    # -- a chat model reads ---------------------------------------------------

    def test_the_label_passes_read_on_the_chat_model(self, chat_cluster,
                                                     monkeypatch):
        from report_ingest.engine import ChatModelEngine
        reports_dir, out, scripts = chat_cluster
        handed: list = []
        _stub_label_passes(monkeypatch, handed)
        reader = self._chat("made-up-reader")

        results = cs.score_on_cluster(
            reports_dir=reports_dir, out_dir=out, prompter=None,
            chat_model=reader, sets=("oos_open",), stages=("labels",))

        assert scripts["engines"] == []          # no Prompter engine at all
        assert [what for what, _ in handed] == ["triage", "review"]
        for _, engine in handed:
            assert isinstance(engine, ChatModelEngine)
            assert engine.chat_model is reader
        blob = json.loads((out / "runs" / "R36.json").read_text(
            encoding="utf-8"))
        assert blob["engine"] == "chat_model"
        assert blob["review_model"] == blob["triage_model"] == \
            "made-up-reader"
        assert blob["chat_model"] == "ScriptedChatModel"
        assert results["engine"] == "chat_model"
        assert results["review_model"] == "made-up-reader"
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "triage and label review through a chat model" in text
        assert "Review model `made-up-reader`" in text
        assert "Engine: `chat_model` over `ScriptedChatModel`" in text

    def test_the_vision_stage_looks_through_the_vision_chat_model(
            self, chat_cluster, monkeypatch):
        import report_ingest.vision_labels as vision_labels
        from report_ingest.engine import ChatModelEngine
        reports_dir, out, scripts = chat_cluster
        handed: list = []

        class _Seen:
            labels: dict = {}
            unresolved: list = []
            model_calls = 0
            cost: dict = {}

            def to_dict(self):
                return {"labels": {}, "unresolved": [], "qa": [], "cost": {}}

        def classify(doc, engine, **kwargs):
            handed.append(engine)
            return _Seen()

        monkeypatch.setattr(vision_labels, "classify_pages_by_vision",
                            classify)
        reader, looker = (self._chat("made-up-reader"),
                          self._chat("made-up-looker"))

        results = cs.score_on_cluster(
            reports_dir=reports_dir, out_dir=out, prompter=None,
            chat_model=reader, vision_chat_model=looker,
            sets=("oos_open",), stages=("vision_labels",))

        (engine,) = handed
        assert isinstance(engine, ChatModelEngine)
        assert engine.chat_model is looker
        blob = json.loads((out / "vision" / "R36.json").read_text(
            encoding="utf-8"))
        assert (blob["model"], blob["engine"], blob["chat_model"]) == (
            "made-up-looker", "chat_model", "ScriptedChatModel")
        seen = results["vision_labels"]
        assert (seen["model"], seen["engine"]) == ("made-up-looker",
                                                   "chat_model")
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "Vision model `made-up-looker`" in text
        assert "Engine: `chat_model` over `ScriptedChatModel`" in text

    def test_a_whole_ingest_reads_and_votes_on_two_chat_models(
            self, chat_cluster):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = chat_cluster
        reader = self._chat("made-up-reader", tg.main_script())
        looker = self._chat("made-up-looker", tg.vision_turns())

        results = _ingest_run(chat_cluster, prompter=None,
                              chat_model=reader, vision_chat_model=looker)

        ingest = results["ingest"]
        assert ingest["failures"] == {}
        assert scripts["engines"] == []
        assert len(reader.requests) == len(tg.main_script())
        assert len(looker.requests) == len(tg.vision_turns())
        assert not reader.replies and not looker.replies
        (row,) = ingest["per_report"]
        assert row["diggs"]["written"] == "yes"
        assert (row["model"], row["engine"], row["chat_model"]) == (
            "made-up-reader", "chat_model", "ScriptedChatModel")
        assert row["vision"] == {"model": "made-up-looker",
                                 "engine": "chat_model",
                                 "chat_model": "ScriptedChatModel"}
        assert "triage" not in row          # the triage is the reader's tier
        assert ingest["engine"] == "chat_model"
        assert ingest["vision_model"] == "made-up-looker"
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "Model `made-up-reader`" in text
        assert ("The vision voter: `chat_model` over `ScriptedChatModel`, "
                "model `made-up-looker`") in text

    def test_a_callable_is_asked_once_for_each_model_name(self,
                                                          chat_cluster):
        from report_ingest.tests import test_graph as tg
        models = {
            "made-up-reader": self._chat("made-up-reader", tg.main_script()),
            "made-up-looker": self._chat("made-up-looker",
                                         tg.vision_turns())}
        asked: list = []

        def factory(name):
            asked.append(name)
            return models[name]

        results = _ingest_run(chat_cluster, prompter=None,
                              chat_model=factory, model="made-up-reader",
                              vision_model="made-up-looker")

        assert sorted(asked) == ["made-up-looker", "made-up-reader"]
        (row,) = results["ingest"]["per_report"]
        assert row["model"] == "made-up-reader"
        assert row["vision"]["model"] == "made-up-looker"
        assert not any(m.replies for m in models.values())

    # -- what is refused, and when ------------------------------------------

    @pytest.mark.parametrize("bad", [object(), lambda name: None])
    def test_what_is_no_chat_model_is_refused_before_anything_runs(
            self, chat_cluster, bad):
        reports_dir, out, scripts = chat_cluster
        with pytest.raises(ValueError, match="must be a LangChain chat model"):
            _ingest_run(chat_cluster, prompter=None, chat_model=bad)
        assert not (out / "ingest").exists()
        assert scripts["engines"] == []

    def test_a_stage_left_on_the_prompter_still_needs_it(self):
        looker = self._chat("made-up-looker")
        with pytest.raises(ValueError, match="pass the live fh_prompter"):
            cs.score_on_cluster(reports_dir="anywhere", prompter=None,
                                vision_chat_model=looker,
                                stages=("ingest",))


# ---------------------------------------------------------------------------
# the readers' switches in the ingest's run.json (BACKLOG RI-67)
# ---------------------------------------------------------------------------

class TestTheIngestRecordsTheReadersSwitches:

    @pytest.fixture(autouse=True)
    def _no_switch_in_the_environment(self, monkeypatch):
        from report_ingest.calc_reader import PAGE_IMAGES_ENV
        from report_ingest.lab_reader import (
            FLOOR_PER_PAGE_ENV, LINK_FROM_LABEL_ENV,
        )
        monkeypatch.delenv(PAGE_IMAGES_ENV, raising=False)
        monkeypatch.delenv(LINK_FROM_LABEL_ENV, raising=False)
        monkeypatch.delenv(FLOOR_PER_PAGE_ENV, raising=False)

    def test_both_are_off_by_default_and_said_so(self, ingest_cluster):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()

        (row,) = _ingest_run(ingest_cluster)["ingest"]["per_report"]

        assert row["calc_page_images"] is False
        assert row["lab_link_from_label"] is False
        assert row["lab_floor_per_page"] is False
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert ("Every calculation page as a picture (RI-17, "
                "GEOTECH_INGEST_CALC_PAGE_IMAGES): off.") in text
        assert ("Specimen-label paragraph (RI-07, "
                "GEOTECH_INGEST_LAB_LINK_FROM_LABEL): off.") in text
        assert ("Per-page lab floor in the brief (RI-56, "
                "GEOTECH_INGEST_LAB_FLOOR_PER_PAGE): off.") in text

    def test_the_per_page_floor_switch_on_is_recorded_on(
            self, ingest_cluster):
        from report_ingest.lab_reader import use_floor_per_page
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()

        with use_floor_per_page():
            (row,) = _ingest_run(ingest_cluster)["ingest"]["per_report"]

        assert row["lab_floor_per_page"] is True
        assert row["lab_link_from_label"] is False
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "GEOTECH_INGEST_LAB_FLOOR_PER_PAGE): on." in text

    def test_the_calc_switch_on_is_recorded_on(self, ingest_cluster):
        from report_ingest.calc_reader import use_page_images
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()

        with use_page_images():
            (row,) = _ingest_run(ingest_cluster)["ingest"]["per_report"]

        assert row["calc_page_images"] is True
        assert row["lab_link_from_label"] is False
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "GEOTECH_INGEST_CALC_PAGE_IMAGES): on." in text

    def test_a_run_json_from_before_says_not_recorded(self):
        text = "\n".join(cs._render_ingest(cs._score_ingest(
            {"R90": {"id": "R90", "cost": {}}}, {}, "m")))
        assert "GEOTECH_INGEST_CALC_PAGE_IMAGES): not recorded;" in text


class TestMeasurementPlumbingRI69:
    """BACKLOG RI-69: DI withheld on request, one chat model per name, a chat
    run's notes in its own words, a label run from the other engine said."""

    @pytest.fixture
    def langchain(self):
        pytest.importorskip("langchain_core")

    # -- (1) di_dir=False ------------------------------------------------------

    def test_none_falls_back_to_a_di_folder_and_false_withholds_it(
            self, tmp_path):
        from report_ingest.corpus import Corpus
        reports = tmp_path / "reports"
        reports.mkdir()
        (tmp_path / "di").mkdir()
        (tmp_path / "di" / "R01.json.gz").write_bytes(b"x")

        fallback = Corpus(reports, di_dir=None)
        assert fallback.di_dir == tmp_path / "di"
        assert fallback.di_withheld is False
        assert fallback.di_file("R01").is_file()

        withheld = Corpus(reports, di_dir=False)
        assert withheld.di_withheld is True
        assert not withheld.di_file("R01").is_file()
        assert withheld.has_di("R01") is False
        assert withheld.load_di("R01") is None

    def test_a_withheld_run_says_so_and_a_normal_run_does_not(
            self, ingest_cluster, monkeypatch):
        reports_dir, out, scripts = ingest_cluster
        (reports_dir.parent / "di").mkdir()
        handed: list = []
        _stub_label_passes(monkeypatch, handed)

        results = cs.score_on_cluster(
            reports_dir=reports_dir, out_dir=out, prompter=object(),
            di_dir=False, sets=("oos_open",), stages=("labels",))

        assert results["di_withheld"] is True
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "Document Intelligence text: withheld" in text
        saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
        assert saved["di_withheld"] is True

        plain_out = out.parent / "out_plain"
        plain = cs.score_on_cluster(
            reports_dir=reports_dir, out_dir=plain_out, prompter=object(),
            sets=("oos_open",), stages=("labels",))
        assert "di_withheld" not in plain
        assert "Document Intelligence" not in (
            plain_out / "RESULTS.md").read_text(encoding="utf-8")

    # -- (2) one request per name --------------------------------------------

    def test_a_factory_is_asked_once_when_model_and_vision_model_match(
            self, langchain):
        from report_ingest.engine import CostMeter
        from report_ingest.tests.chat_fixtures import ScriptedChatModel
        asked: list = []

        def factory(name):
            asked.append(name)
            return ScriptedChatModel(replies=[], model_name=name)

        engines = cs._Engines(None, factory)
        first = engines.build("made-up-model", CostMeter())
        second = engines.build("made-up-model", CostMeter(), vision=True)
        assert asked == ["made-up-model"]
        assert first.chat_model is second.chat_model

    def test_two_different_factories_with_one_name_are_each_asked_once(
            self, langchain):
        from report_ingest.engine import CostMeter
        from report_ingest.tests.chat_fixtures import ScriptedChatModel
        asked: list = []

        def make(tag):
            def factory(name):
                asked.append(tag)
                return ScriptedChatModel(replies=[], model_name=name)
            return factory

        main, looker = make("main"), make("vision")
        engines = cs._Engines(None, main, looker)
        a = engines.build("made-up-model", CostMeter())
        b = engines.build("made-up-model", CostMeter(), vision=True)
        engines.build("made-up-model", CostMeter())
        engines.build("made-up-model", CostMeter(), vision=True)
        assert asked == ["main", "vision"]
        assert a.chat_model is not b.chat_model

    # -- (3) a chat run's notes -----------------------------------------------

    def test_a_chat_run_names_no_funhouse_in_its_cost_and_served_by_notes(
            self, chat_cluster, monkeypatch, langchain):
        reports_dir, out, scripts = chat_cluster
        handed: list = []
        _stub_label_passes(monkeypatch, handed)
        reader = TestAChatModelInThePromptersPlace._chat("made-up-reader")

        cs.score_on_cluster(
            reports_dir=reports_dir, out_dir=out, prompter=None,
            chat_model=reader, sets=("oos_open",), stages=("labels",))

        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "budget endpoint" not in text
        assert "Funhouse publishes" not in text
        assert "tier is an alias" not in text
        assert "provider's own usage report" in text

    def test_the_served_by_line_of_a_chat_run_names_no_tier_alias(
            self, chat_cluster, monkeypatch, langchain):
        reports_dir, out, scripts = chat_cluster
        _stub_label_passes(monkeypatch, [])
        scored = cs._score

        def with_served_by(*args, **kwargs):
            results = scored(*args, **kwargs)
            results["served_by"] = ["made-up-deployment"]
            return results

        monkeypatch.setattr(cs, "_score", with_served_by)
        cs.score_on_cluster(
            reports_dir=reports_dir, out_dir=out, prompter=None,
            chat_model=TestAChatModelInThePromptersPlace._chat(
                "made-up-reader"),
            sets=("oos_open",), stages=("labels",))
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "Served by: made-up-deployment." in text.splitlines()

    def test_a_prompter_run_prints_the_notes_it_always_printed(
            self, ingest_cluster, monkeypatch):
        reports_dir, out, scripts = ingest_cluster
        _stub_label_passes(monkeypatch, [])
        cs.score_on_cluster(
            reports_dir=reports_dir, out_dir=out, prompter=object(),
            sets=("oos_open",), stages=("labels",))
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert ("A Funhouse tier is an alias and the deployment behind it "
                "changes, so this is what actually answered.") in text
        assert ("Funhouse publishes no per-token price for a capability "
                "tier") in text
        assert "Funhouse's own budget endpoint" in text
        # the stage's results make no difference to a Prompter note
        cost = {"dollars": 0.0}
        assert cs._price_note(cost, {"engine": "prompter"}) == \
            cs._price_note(cost) == cs._price_note(cost, {})
        cost = {"dollars": 1.0}
        assert cs._price_note(cost, {"engine": "prompter"}) == \
            cs._price_note(cost)

    def test_a_prompter_ingest_run_file_carries_no_new_key(
            self, ingest_cluster):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        results = _ingest_run(ingest_cluster)
        (row,) = results["ingest"]["per_report"]
        blob = json.loads((out / "ingest" / "R36" / "run.json").read_text(
            encoding="utf-8"))
        for key in ("di_withheld", "label_run_engine"):
            assert key not in row and key not in blob
        assert "di_withheld" not in results
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "other engine" not in text

    # -- (6) a label run from the other engine -------------------------------

    def test_an_ingest_that_reuses_the_other_engines_label_run_says_so(
            self, chat_cluster, langchain, monkeypatch):
        from report_ingest.tests import test_graph as tg
        # The scripted chat answers do not all fit the readers' schemas, so
        # some items go unread; this test is about the label run's engine,
        # not about unread items (RI-110 keeps such a report out of the rows).
        monkeypatch.setattr(cs, "_item_error_fields",
                            lambda record, bound=(): {})
        reports_dir, out, scripts = chat_cluster
        runs = out / "runs"
        runs.mkdir(parents=True)
        blob = _run_blob("R36", 22, {3: "plan"}, {3: "plan"})
        blob["engine"] = "prompter"
        (runs / "R36.json").write_text(json.dumps(blob), encoding="utf-8")
        reader = TestAChatModelInThePromptersPlace._chat(
            "made-up-reader", tg.main_script())
        looker = TestAChatModelInThePromptersPlace._chat(
            "made-up-looker", tg.vision_turns())

        results = _ingest_run(chat_cluster, prompter=None,
                              chat_model=reader, vision_chat_model=looker)

        (row,) = results["ingest"]["per_report"]
        assert row["review_reused"] or row["triage_reused"]
        assert row["label_run_engine"] == "prompter"
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "Reused a saved label run made on the other engine" in text
        assert "R36 (prompter)" in text


# -- range_labels (BACKLOG RI-73) ---------------------------------------------

def _range_blob(rid, n_pages, arm, rules, labels, dollars=0.0):
    """One report's saved range run for one arm, as the stage writes it,
    made by the stage's default models from today's instructions."""
    from report_ingest.range_labels import (DESCRIBE_PROMPT_VERSION,
                                            labeller_fingerprint)

    return {
        "id": rid, "run_date": "2026-10-24", "n_pages": n_pages, "arm": arm,
        "describe_model": "funhouse-gpt-low",
        "range_model": "funhouse-gpt-high",
        "describe_prompt": DESCRIBE_PROMPT_VERSION,
        "labeller_prompt": labeller_fingerprint(arm == "hinted"),
        "conventions": "made-up", "describer": {"model": "made-up-eyes"},
        "labeller": {"model": "made-up-reader"}, "served_by": [],
        "rules_labels": {str(k): v for k, v in rules.items()},
        "labels": {str(k): v for k, v in labels.items()},
        "ranges": {"labels": {str(k): v for k, v in labels.items()},
                   "sections": [], "exceptions": [], "unresolved": [],
                   "qa": [], "windows": []},
        "describe": {"n_described": n_pages},
        "describe_cost": {"calls": n_pages, "input_tokens": 9000,
                          "output_tokens": 900, "dollars": 0.5},
        "cost": {"calls": 1, "input_tokens": 4000, "output_tokens": 600,
                 "dollars": dollars},
        "seconds": 30.0,
    }


@pytest.fixture()
def range_cluster(cluster):
    """The two reports of ``cluster``, each with both arms already run."""
    reports_dir, out, oos = cluster
    rules = {"R36": {3: "figure", 7: "boring_log"},
             "R31": {5: "lab_test", 9: "narrative"}}
    arms = {"plain": {"R36": {3: "plan", 7: "figure"},
                      "R31": {5: "lab_test", 9: "narrative"}},
            "hinted": {"R36": {3: "plan", 7: "boring_log"},
                       "R31": {5: "lab_test", 9: "narrative"}}}
    for arm, per_report in arms.items():
        (out / "ranges" / arm).mkdir(parents=True)
        for rid, labels in per_report.items():
            (out / "ranges" / arm / f"{rid}.json").write_text(json.dumps(
                _range_blob(rid, 100, arm, rules[rid], labels, 0.25)),
                encoding="utf-8")
    return reports_dir, out, oos


def _range_run(range_cluster, **over):
    reports_dir, out, oos = range_cluster
    kwargs = dict(reports_dir=reports_dir, out_dir=out, prompter=object(),
                  oos_labels=oos, sets=("oos_blind",),
                  stages=("range_labels",))
    kwargs.update(over)
    return cs.score_on_cluster(**kwargs)


class TestTheRangeStage:
    """RI-73: the rules and both arms on one set of hand labels."""

    def test_it_is_a_stage_and_a_resume_is_free(self, range_cluster, capsys):
        assert "range_labels" in cs.STAGE_NAMES
        results = _range_run(range_cluster)
        assert "R36: already done, skipping" in capsys.readouterr().out
        assert results["range_labels"]["sets"]["oos_blind"]["n_reports"] == 2

    def test_each_column_is_scored_strictly_and_by_harm(self, range_cluster):
        row = _range_run(range_cluster)["range_labels"]["sets"]["oos_blind"]
        columns = row["columns"]
        # Four hand pages. Rules: R36 p3 figure for plan (neither read):
        # wrong, not harmful. Plain: R36 p7 figure for boring_log: wrong and
        # harmful (a log no reader reads). Hinted: all four right.
        assert columns["rules"]["accuracy"] == pytest.approx(0.75)
        assert columns["rules"]["harmful"] == 0
        assert columns["plain"]["accuracy"] == pytest.approx(0.75)
        assert columns["plain"]["harmful"] == 1
        assert columns["hinted"]["accuracy"] == pytest.approx(1.0)
        assert columns["hinted"]["harmful"] == 0

    def test_the_describer_is_counted_once_a_report_not_once_an_arm(
            self, range_cluster):
        cost = _range_run(range_cluster)["range_labels"]["cost"]
        assert cost["describe"]["calls"] == 200
        assert cost["label"]["calls"] == 4
        assert cost["label"]["dollars"] == pytest.approx(1.0)

    def test_an_unknown_arm_is_refused(self, range_cluster):
        with pytest.raises(ValueError, match="range_arms"):
            _range_run(range_cluster, range_arms=("plain", "guessed"),
                       redo=True)

    def test_results_carry_ids_and_numbers_and_no_working_state(
            self, range_cluster):
        reports_dir, out, oos = range_cluster
        _range_run(range_cluster)
        blob = json.loads((out / "results.json").read_text(encoding="utf-8"))
        assert "_scores" not in blob["range_labels"]["sets"]["oos_blind"]
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "# RI-73: every page described" in text
        assert "RI-73 on the cluster" not in text
        assert "HARMFUL pages" in text
        assert "made up" not in text

    def test_a_run_drawn_over_an_undescribed_page_is_not_skipped(
            self, range_cluster, capsys):
        reports_dir, out, oos = range_cluster
        path = out / "ranges" / "plain" / "R31.json"
        blob = json.loads(path.read_text(encoding="utf-8"))
        blob["describe"]["unresolved"] = [{"page": 9, "why": "made up"}]
        path.write_text(json.dumps(blob), encoding="utf-8")
        results = _range_run(range_cluster)
        printed = capsys.readouterr().out
        assert "R36: already done, skipping" in printed
        assert "R31: already done" not in printed
        assert "R31" in results["range_labels"]["failures"]

    @pytest.mark.parametrize("key, value", [
        ("labeller_prompt", "made-up-old"), ("describe_prompt", "ri73-0"),
        ("range_model", "made-up-other-labeller"),
        ("describe_model", "made-up-other-eyes")])
    def test_a_run_from_other_instructions_or_models_is_not_skipped(
            self, range_cluster, capsys, key, value):
        reports_dir, out, oos = range_cluster
        path = out / "ranges" / "hinted" / "R31.json"
        blob = json.loads(path.read_text(encoding="utf-8"))
        blob[key] = value
        path.write_text(json.dumps(blob), encoding="utf-8")
        _range_run(range_cluster)
        printed = capsys.readouterr().out
        assert "R36: already done, skipping" in printed
        assert "R31: already done" not in printed

    def test_a_corrupt_run_file_is_not_finished_and_the_run_goes_on(
            self, range_cluster, capsys):
        reports_dir, out, oos = range_cluster
        (out / "ranges" / "plain" / "R31.json").write_text(
            "{ not json", encoding="utf-8")
        assert cs._finished_range_run(
            out / "ranges" / "plain" / "R31.json") is None
        results = _range_run(range_cluster)
        printed = capsys.readouterr().out
        assert "R36: already done, skipping" in printed
        assert "R31" in results["range_labels"]["failures"]

    def test_the_saved_runs_are_read_back_without_a_call(self, range_cluster):
        reports_dir, out, oos = range_cluster
        saved = cs._saved_ranges(out, ["R31", "R36", "R99"],
                                 ("plain", "hinted"), "funhouse-gpt-low",
                                 "funhouse-gpt-high")
        assert sorted(saved["runs"]["plain"]) == ["R31", "R36"]
        assert sorted(saved["runs"]["hinted"]) == ["R31", "R36"]
        other = cs._saved_ranges(out, ["R31"], ("plain",),
                                 "made-up-eyes", "funhouse-gpt-high")
        assert other["runs"]["plain"] == {}


def test_a_describing_pass_keeps_its_cost_record_beside_the_runs(tmp_path):
    from report_ingest.engine import CostMeter, Usage

    meter = CostMeter()
    cs._keep_meter(tmp_path, "R99", meter)
    assert not tmp_path.exists() or not list(tmp_path.iterdir())
    entry = meter.add("made-up-model", Usage(input_tokens=10), 1.0)
    entry["notes"] = ["an answer was not taken: made up"]
    cs._keep_meter(tmp_path / "meters", "R99", meter)
    (saved,) = list((tmp_path / "meters").glob("R99-*.json"))
    blob = json.loads(saved.read_text(encoding="utf-8"))
    assert blob["id"] == "R99" and blob["cost"]["calls"] == 1
    assert "an answer was not taken: made up" in saved.read_text(
        encoding="utf-8")


def test_the_mirror_is_synced_every_n_pages_described():
    synced = []
    progress = cs._mirror_every(3, lambda: synced.append(1))
    for done in range(1, 10):
        progress(done, 9)
    assert len(synced) == 3
    assert cs._mirror_every(3, None) is None


def test_another_conventions_runs_go_in_their_own_folder_and_column():
    assert cs._range_column("hinted") == "hinted"
    assert cs._range_column("hinted", "b") == "hinted_B"


def test_a_conventions_a_run_is_not_a_finished_conventions_b_run(
        range_cluster):
    reports_dir, out, oos = range_cluster
    path = out / "ranges" / "hinted" / "R31.json"
    assert cs._finished_range_run(path, "hinted", "funhouse-gpt-low",
                                  "funhouse-gpt-high") is not None
    assert cs._finished_range_run(path, "hinted", "funhouse-gpt-low",
                                  "funhouse-gpt-high", "B") is None
    saved = cs._saved_ranges(out, ["R31", "R36"], ("hinted",),
                             "funhouse-gpt-low", "funhouse-gpt-high", "B")
    assert saved["arms"] == ["hinted_B"] and saved["runs"]["hinted_B"] == {}


def test_an_unknown_conventions_name_is_refused(range_cluster, tmp_path):
    reports_dir, out, oos = range_cluster
    with pytest.raises(ValueError, match="conventions must be one of"):
        cs._run_ranges(None, None, "made-up-eyes", "made-up-reader", out,
                       [], arms=("hinted",), workers=1, redo=False,
                       conventions="Z")


# -- model first as a second arm (BACKLOG RI-87 piece 9) ---------------------

#: The made-up models each reader stage reads on under ``on``.
FIRST_MODELS = {"logs": "fake-logs", "lab": "fake-lab", "calc": "fake-calc",
                "narrative": "fake-narrative"}


@pytest.fixture()
def first_cluster(tmp_path, monkeypatch):
    """The synthetic report, one made-up answer sheet per reader stage, and
    engines made by model name, each replaying the script ``scripts`` holds
    for that name (none: an engine that may not be called)."""
    from report_ingest.tests.fake_engine import FakeEngine
    from report_ingest.tests.narrative_fixtures import build_narrative_report
    import report_ingest.engine as engine_module

    reports_dir = tmp_path / "corpus"
    reports_dir.mkdir()
    (reports_dir / "R36.pdf").write_bytes(build_narrative_report().pdf)
    truth = tmp_path / "truth"
    sheets = {
        "logs": ("R36_p7", {"id": "R36_p7", "pages": [7, 8],
                            "depth_unit": "m", "fields": {},
                            "layers": [{"top": 0.0, "uscs": "CL"}],
                            "samples": [], "water": []}),
        "lab": ("atterberg__R36_p12", {
            "id": "atterberg__R36_p12", "kind": "atterberg",
            "depth_unit": "m",
            "tests": [{"investigation_id": "B-1", "depth_top": 4.5,
                       "ll": 38, "pl": 19, "pi": 19}]}),
        "calc": ("lateral_pile__R36_p20", {
            "id": "lateral_pile__R36_p20", "report": "R36",
            "kind": "lateral_pile", "pages": [20],
            "inputs": [{"name": "Pile-head load", "value": 120.0,
                        "unit": "kN", "page": 20}], "results": []}),
        "narrative": ("R36", {"id": "R36", "general": {
            "projectName": "Rosewood Terrace Development"},
            "natural_hazards": {}}),
    }
    for stage, (name, blob) in sheets.items():
        (truth / stage).mkdir(parents=True)
        (truth / stage / f"{name}.json").write_text(json.dumps(blob),
                                                    encoding="utf-8")
    scripts = {"by_model": {}, "engines": []}

    def fake_engine(prompter, model, meter=None, **kwargs):
        engine = FakeEngine(list(scripts["by_model"].get(model) or []),
                            name=model)
        engine.accepts_pdf = True
        engine.served_by = "fake-deployment"
        scripts["engines"].append(engine)
        return engine

    monkeypatch.setattr(engine_module, "PrompterEngine", fake_engine)
    return reports_dir, tmp_path / "out", truth, scripts


def _first_scripts():
    """Made-up model-first answers for the four reader stages."""
    from report_ingest.tests import test_graph as tg
    from report_ingest.tests import test_graph_model_first as tgm

    return {"fake-logs": [tgm._log("B-1", "boring", 7, [
                (0.0, 31.5, "Brown sandy lean CLAY, stiff, moist")])],
            "fake-lab": [tg.lab_turn("atterberg", 12)],
            "fake-calc": [tgm._calc()],
            "fake-narrative": [tg.narrative_turn()]}


def _first_run(first_cluster, stages, **over):
    reports_dir, out, truth, _scripts = first_cluster
    kwargs = dict(reports_dir=reports_dir, out_dir=out, prompter=object(),
                  stages=stages, sets=(), model="fake-today",
                  truth_dir=truth / "logs", lab_truth_dir=truth / "lab",
                  calc_truth_dir=truth / "calc",
                  narrative_truth_dir=truth / "narrative")
    kwargs.update(over)
    return cs.score_on_cluster(**kwargs)


class TestModelFirstAsASecondArm:
    """``model_first`` on the reader stages and the ingest: the same items,
    the same scorers, and every run file naming its arm (BACKLOG RI-87,
    plan section 8.8). No model is called."""

    def test_on_reads_each_stage_on_its_own_model_and_names_the_arm(
            self, first_cluster):
        from report_ingest.model_first import READ_PROMPT_VERSION

        pytest.importorskip("report_ingest.locate")
        _reports, out, _truth, scripts = first_cluster
        scripts["by_model"] = _first_scripts()
        results = _first_run(first_cluster, tuple(FIRST_MODELS),
                             model_first="on", stage_models=FIRST_MODELS)
        called = {e.name: e.n_calls for e in scripts["engines"]}
        assert called == {name: 1 for name in FIRST_MODELS.values()}
        for stage, name in FIRST_MODELS.items():
            assert results[stage]["model"] == name
            assert results[stage]["model_first"] == ["on"]
            assert results[stage]["routes"] == ["pdf"]
            assert results[stage]["prompt_versions"] == [READ_PROMPT_VERSION]
        log = json.loads((out / "logs" / "R36_p7.json").read_text("utf-8"))
        assert (log["model_first"], log["route"], log["prompt_version"]) == (
            "on", "pdf", READ_PROMPT_VERSION)
        assert log["calls"] == [{"pages": [7, 8], "stop_reason": "end_turn",
                                 "cut_off": False, "answered": True}]
        assert log["placement"]["values"] > 0
        assert log["after"]["error"] is None
        # Scored by today's scorer: the layer the answer sheet names was
        # read, and there is no floor.
        assert log["after"]["scores"]["layer_top"]["found"] == 1
        assert log["after"]["record"] == log["after"]["model_record"]
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "Model first (RI-87, GEOTECH_INGEST_MODEL_FIRST): `on`" in text
        assert "Placed at a spot (plan section 8.4, gate 90 %)" in text

    def test_off_reads_as_today_and_both_arms_share_the_before_column(
            self, first_cluster, tmp_path):
        from report_ingest.tests import test_graph as tg

        pytest.importorskip("report_ingest.locate")
        _reports, out, _truth, scripts = first_cluster
        scripts["by_model"] = {"fake-today": [tg.log_turn("B-1")],
                               **_first_scripts()}
        off = _first_run(first_cluster, ("logs",), model_first="off")
        on = _first_run(first_cluster, ("logs",), model_first="on",
                        stage_models=FIRST_MODELS,
                        out_dir=tmp_path / "out_on")
        assert off["logs"]["model"] == "fake-today"
        assert off["logs"]["model_first"] == ["off"]
        assert "routes" not in off["logs"]
        saved_off = json.loads(
            (out / "logs" / "R36_p7.json").read_text("utf-8"))
        saved_on = json.loads(
            (tmp_path / "out_on" / "logs" / "R36_p7.json").read_text("utf-8"))
        assert saved_off["model_first"] == "off" and "route" not in saved_off
        # The grid is read with no model call, so it is the same in both.
        assert saved_off["before"]["scores"] == saved_on["before"]["scores"]
        assert set(saved_off["after"]["scores"]) == set(
            saved_on["after"]["scores"])
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "Model first (RI-87, GEOTECH_INGEST_MODEL_FIRST): `off`." \
            in text

    @pytest.mark.parametrize("redo", [False, True])
    def test_an_item_read_in_the_other_arm_is_a_failure_and_is_kept(
            self, logs_cluster, redo, capsys):
        """The seeded run files say no arm, so they were read with it off;
        an ``on`` run names them and never overwrites them, even with
        ``redo``, and builds no engine for them."""
        _reports, out, _truth = logs_cluster
        before = (out / "logs" / "R36_p38.json").read_bytes()
        results = _logs_run(logs_cluster, model_first="on", redo=redo)
        failures = results["logs"]["failures"]
        assert set(failures) == {"R36_p38", "R31_p278"}
        assert "read with model first off and this run is on" in \
            failures["R36_p38"]
        assert (out / "logs" / "R36_p38.json").read_bytes() == before
        assert "skipped -- its run file was read with model first off" in \
            capsys.readouterr().out

    def test_stage_models_name_the_reader_stages_and_a_route_is_checked(
            self, logs_cluster):
        with pytest.raises(ValueError, match="stage_models names"):
            _logs_run(logs_cluster, model_first="on",
                      stage_models={"soundings": "fake-x"})
        with pytest.raises(ValueError):
            _logs_run(logs_cluster, model_first="on", page_route="fax")
        with pytest.raises(ValueError):
            _logs_run(logs_cluster, model_first="sometimes")

    def test_stage_models_under_off_set_todays_reader(
            self, logs_cluster, capsys):
        """Both arms on the same models: off reads a named stage on its
        model, today's reader."""
        results = _logs_run(logs_cluster, model_first="off",
                            stage_models={"logs": "fake-logs"})
        assert results["logs"]["model"] == "fake-logs"
        printed = capsys.readouterr().out
        assert ("model first: off (today's readers); logs on fake-logs, "
                "lab on funhouse-gpt-high") in printed

    def test_the_variable_sets_the_arm_when_none_is_passed(
            self, logs_cluster, monkeypatch):
        monkeypatch.setenv("GEOTECH_INGEST_MODEL_FIRST", "on")
        results = _logs_run(logs_cluster)
        assert set(results["logs"]["failures"]) == {"R36_p38", "R31_p278"}

    def test_the_ingest_on_runs_the_graph_model_first_and_says_so(
            self, ingest_cluster, monkeypatch, tmp_path):
        """The describer, the labeller and each reader on its own model;
        the triage alone seeded from a label run; ``run.json`` names the
        arm, the routes, the models, the versions and what placement found;
        the labels are scored for pages sent to the wrong reader. The
        laboratory window is answered here: a group no answer came back for
        is a failed item, and the report no result (RI-187)."""
        from report_ingest.model_first import (READ_PROMPT_VERSION,
                                               LabSheetReading)
        from report_ingest.range_labels import DESCRIBE_PROMPT_VERSION
        from report_ingest.tests import test_graph_model_first as tgm
        from report_ingest.tests.fake_engine import FakeEngine
        import report_ingest.engine as engine_module

        pytest.importorskip("report_ingest.locate")
        reports_dir, out, _scripts = ingest_cluster
        main, describer, labeller, stages = tgm._engines()
        stages["lab"].turns = [{"final": LabSheetReading.model_validate({
            "depth_unit": "m",
            "tests": [{"kind": "atterberg", "investigation_id": "B-1",
                       "sample_id": "S-1", "depth_top": 1.5, "ll": 41.0,
                       "pl": 19.0, "pi": 22.0, "prov": {"page": 12}}]})}]
        by_name = {"fake-main": main, "fake-describer": describer,
                   "fake-labeller": labeller,
                   **{FIRST_MODELS[s]: e for s, e in stages.items()}}

        def fake_engine(prompter, model, meter=None, **kwargs):
            engine = by_name.get(model) or FakeEngine([], name=model)
            engine.served_by = "fake-deployment"
            return engine

        monkeypatch.setattr(engine_module, "PrompterEngine", fake_engine)
        hand = tmp_path / "hand.json"
        hand.write_text(json.dumps({"R36": {
            "7": {"label": "boring_log", "alternates": []},
            "20": {"label": "calculation", "alternates": []}}}),
            encoding="utf-8")
        results = _ingest_run(
            ingest_cluster, model="fake-main", model_first="on",
            stage_models=FIRST_MODELS, describe_model="fake-describer",
            range_model="fake-labeller", oos_labels=hand)
        assert results["ingest"]["failures"] == {}
        assert describer.n_calls == 22 and labeller.n_calls == 2
        assert {s: e.n_calls for s, e in stages.items()} == {
            "narrative": 2, "logs": 3, "lab": 1, "calc": 1}
        run = json.loads((out / "ingest" / "R36" / "run.json").read_text(
            "utf-8"))
        assert run["model_first"] == "on"
        assert run["routes"] == {s: "pdf" for s in FIRST_MODELS}
        assert run["models"] == {**FIRST_MODELS, "describer": "fake-describer",
                                 "labeller": "fake-labeller"}
        assert run["prompt_version"] == READ_PROMPT_VERSION
        assert run["ranges"]["arm"] == "plain"
        assert run["ranges"]["describe_prompt"] == DESCRIBE_PROMPT_VERSION
        assert run["placement"]["values"] > 0
        assert run["labels"]["harmful"] == {"rules": 0, "final": 0}
        assert results["ingest"]["model_first"] == ["on"]
        assert results["ingest"]["routes"] == ["pdf"]
        assert not (out / "ingest" / "R36" / "vision.json").exists()
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "wrong rdr" in text and "`on`" in text

    def test_the_ingest_off_reads_each_named_stage_on_its_model(
            self, ingest_cluster, monkeypatch):
        """Off with ``stage_models``: today's readers each on their stage's
        model, the triage and the identity call on ``model``, the vision
        voter on its own; ``run.json`` names the readers' models."""
        from report_ingest.tests import test_graph as tg
        from report_ingest.tests.fake_engine import FakeEngine
        import report_ingest.engine as engine_module

        _reports, out, _scripts = ingest_cluster
        turns = {
            "fake-main": [tg.triage_turn(), tg.identity_turn()],
            "funhouse-gpt-low": tg.vision_turns(),
            "fake-narrative": [tg.narrative_turn(),
                               tg.bound_narrative_turn()],
            "fake-logs": [tg.log_turn("B-1"), tg.log_turn("TP-1", "test_pit"),
                          tg.log_turn("BH-1")],
            "fake-lab": [tg.lab_turn("atterberg", 12),
                         tg.lab_turn("gradation", 13)],
            "fake-calc": [tg.calc_turn()]}
        by_name = {name: FakeEngine(list(script), name=name)
                   for name, script in turns.items()}

        def fake_engine(prompter, model, meter=None, **kwargs):
            engine = by_name.get(model) or FakeEngine([], name=model)
            engine.served_by = "fake-deployment"
            return engine

        monkeypatch.setattr(engine_module, "PrompterEngine", fake_engine)
        results = _ingest_run(ingest_cluster, model="fake-main",
                              model_first="off", stage_models=FIRST_MODELS)
        assert results["ingest"]["failures"] == {}
        assert {name: e.n_calls for name, e in by_name.items()} == {
            name: len(script) for name, script in turns.items()}
        run = json.loads((out / "ingest" / "R36" / "run.json").read_text(
            "utf-8"))
        assert run["model_first"] == "off"
        assert run["models"] == FIRST_MODELS
        assert results["ingest"]["model_first"] == ["off"]

    def test_the_ingest_off_says_off_and_scores_the_wrong_reader_pages(
            self, ingest_cluster, tmp_path):
        from report_ingest.tests import test_graph as tg

        _reports, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        hand = tmp_path / "hand.json"
        hand.write_text(json.dumps({"R36": {
            "7": {"label": "boring_log", "alternates": []},
            "12": {"label": "narrative", "alternates": []}}}),
            encoding="utf-8")
        results = _ingest_run(ingest_cluster, model_first="off",
                              oos_labels=hand)
        (row,) = results["ingest"]["per_report"]
        assert row["model_first"] == "off" and "routes" not in row
        # Page 12 is a laboratory sheet the hand calls narrative: a page
        # sent to the lab reader rather than the narrative's.
        assert row["labels"]["harmful"] == {"rules": 1, "final": 1}
        assert results["ingest"]["totals"]["labels"]["harmful_final"] == 1


# -- token limits, the arm stamp and the placement file (BACKLOG RI-120) -------

class _Counted:
    """An engine that counts its calls, standing in for a real one."""

    accepts_pdf = True

    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return "reply"


def _meter(tokens_in, tokens_out=0):
    from report_ingest.engine import CostMeter, Usage

    meter = CostMeter()
    meter.add("made-up-model", Usage(input_tokens=tokens_in,
                                     output_tokens=tokens_out), 0.1)
    return meter


class TestTokenBrake:
    def test_no_limit_wraps_nothing(self):
        inner = _Counted()
        assert cs._TokenBrake(_meter(10 ** 6), None, None).wrap(inner) is inner

    def test_a_call_past_the_report_limit_is_refused_before_it_is_sent(self):
        from report_ingest.engine import CallCeilingReached, Usage

        meter, inner = _meter(900), _Counted()
        brake = cs._TokenBrake(meter, 1000, None)
        engine = brake.wrap(inner)
        assert engine.complete([]) == "reply" and inner.calls == 1
        assert engine.accepts_pdf is True          # attributes pass through
        meter.add("made-up-model", Usage(input_tokens=150), 0.1)
        with pytest.raises(CallCeilingReached):
            engine.complete([])
        assert inner.calls == 1
        assert brake.fields()["token_brake"]["limit"] == "max_report_tokens_in"

    def test_the_total_limit_counts_input_and_output(self):
        from report_ingest.engine import CallCeilingReached

        brake = cs._TokenBrake(_meter(400, 700), None, 1000)
        with pytest.raises(CallCeilingReached):
            brake.wrap(_Counted()).complete([])
        assert brake.fields()["token_brake"]["limit"] == "max_total_tokens"

    def test_cache_writes_and_reads_count_as_tokens_in(self):
        """BACKLOG RI-171: a route that reports nearly all its input as
        cache writes and reads (Foundry's Claude) still trips both limits.
        The numbers are made up."""
        from report_ingest.engine import CallCeilingReached, CostMeter, Usage

        meter = CostMeter()
        meter.add("made-up-model", Usage(input_tokens=4,
                                         cache_write_tokens=700,
                                         cache_read_tokens=300,
                                         output_tokens=50), 0.1)
        assert meter.tokens_in == 1004
        assert meter.to_dict()["tokens_in"] == 1004
        report = cs._TokenBrake(meter, 1000, None)
        with pytest.raises(CallCeilingReached, match="1,004 tokens in"):
            report.wrap(_Counted()).complete([])
        entry = report.fields()["token_brake"]
        assert (entry["tokens_in"], entry["input_tokens"],
                entry["cache_write_tokens"], entry["cache_read_tokens"],
                entry["output_tokens"]) == (1004, 4, 700, 300, 50)
        total = cs._TokenBrake(meter, None, 1054)
        with pytest.raises(CallCeilingReached):
            total.wrap(_Counted()).complete([])
        assert cs._TokenBrake(meter, None, 1055).wrap(
            _Counted()).complete([]) == "reply"

    def test_a_run_files_cost_counts_its_cache_tokens(self):
        cost = {"input_tokens": 4, "cache_write_tokens": 700,
                "cache_read_tokens": 300, "output_tokens": 50}
        assert cs._tokens_of(cost) == 1054
        assert cs._attempts_in([{"cost": cost},
                                {"cost": {"input_tokens": 10}}]) == 1014
        # A stage's total keeps the cache writes too.
        total = cs._cost_totals()
        cs._add_cost(total, cost)
        assert (total["cache_write_tokens"], cs._tokens_of(total)) == (
            700, 1054)

    def test_all_attempts_carries_one_tokens_in_number(self):
        """BACKLOG RI-171: a run file's stop check reads one number off
        ``all_attempts``, every attempt's plain input, cache writes and
        cache reads together. Made-up counts."""
        earlier = [{"run_date": "2027-01-15", "ended": "error",
                    "seconds": 3.0,
                    "cost": {"calls": 2, "input_tokens": 6,
                             "cache_write_tokens": 1500,
                             "cache_read_tokens": 200}},
                   {"run_date": "2027-01-16", "ended": "token_brake",
                    "seconds": 2.0, "cost": {"calls": 1, "input_tokens": 90}}]
        now = {"calls": 3, "input_tokens": 4, "cache_write_tokens": 700,
               "cache_read_tokens": 300, "output_tokens": 50}
        total = cs._attempts_fields(earlier, now, 4.0)["all_attempts"]
        assert total["tokens_in"] == 6 + 1500 + 200 + 90 + 4 + 700 + 300
        assert total["tokens_in"] == cs._attempts_in(earlier) + 1004
        assert (total["input_tokens"], total["cache_write_tokens"],
                total["cache_read_tokens"]) == (100, 2200, 500)

    def test_the_cost_line_shows_the_three_parts(self):
        from report_ingest.engine import CostMeter, Usage, tokens_in_words

        assert tokens_in_words({"input_tokens": 60,
                                "cache_write_tokens": 1700,
                                "cache_read_tokens": 500}) == (
            "2,260 tokens in (60 input, 1,700 cache write, 500 cache read)")
        assert tokens_in_words({"input_tokens": 900}) == "900 tokens in"
        meter = CostMeter()
        meter.add("made-up-model", Usage(input_tokens=60,
                                         cache_write_tokens=1700,
                                         cache_read_tokens=500,
                                         output_tokens=40), 2.0)
        assert ("1 calls, 2,260 tokens in (60 input, 1,700 cache write, "
                "500 cache read), 40 out") in meter.summary()

    def test_a_brake_that_never_fired_adds_nothing_to_the_run_file(self):
        brake = cs._TokenBrake(_meter(10), 1000, 1000)
        brake.wrap(_Counted()).complete([])
        assert brake.fields() == {}

    def test_the_arm_engines_are_wrapped_and_the_callers_dict_is_not(self):
        brake = cs._TokenBrake(_meter(0), 1000, None)
        kwargs = {"model_first": "on", "describe_engine": _Counted(),
                  "range_engine": _Counted(),
                  "stage_engines": {"logs": _Counted()}}
        wrapped = brake.wrap_kwargs(kwargs)
        assert isinstance(wrapped["describe_engine"], cs._BrakedEngine)
        assert isinstance(wrapped["stage_engines"]["logs"], cs._BrakedEngine)
        assert isinstance(kwargs["describe_engine"], _Counted)
        assert wrapped["model_first"] == "on"


class TestArmStamp:
    def test_a_folder_is_held_to_the_arm_it_was_started_in(self, tmp_path):
        folder = tmp_path / "ingest" / "R1"
        folder.mkdir(parents=True)
        cs._stamp_arm(folder, cs._FirstArm("off"))
        run_file = folder / "run.json"            # no run file yet
        assert cs._arm_clash(run_file, cs._FirstArm("off")) is None
        clash = cs._arm_clash(run_file, cs._FirstArm("on"))
        assert "started with model first off" in clash
        assert "fresh out_dir" in clash

    def test_the_stamp_is_written_once(self, tmp_path):
        cs._stamp_arm(tmp_path, cs._FirstArm("off"))
        cs._stamp_arm(tmp_path, cs._FirstArm("on"))
        assert json.loads((tmp_path / "arm.json").read_text("utf-8")) == {
            "model_first": "off"}

    def test_a_folder_with_no_stamp_is_judged_by_its_run_file_as_before(
            self, tmp_path):
        run_file = tmp_path / "run.json"
        run_file.write_text(json.dumps({"model_first": "off"}), "utf-8")
        assert cs._arm_clash(run_file, cs._FirstArm("on")) is not None
        assert cs._arm_clash(run_file, cs._FirstArm("off")) is None


class TestTheIngestStageLimits:
    def test_the_stage_stamps_the_folder_and_the_other_arm_cannot_take_it(
            self, ingest_cluster):
        _reports, out, _scripts = ingest_cluster
        _ingest_run(ingest_cluster, model="fake-main")
        folder = out / "ingest" / "R36"
        assert json.loads((folder / "arm.json").read_text("utf-8")) == {
            "model_first": "off"}
        (folder / "run.json").unlink()            # an interrupted folder
        results = _ingest_run(ingest_cluster, model="fake-main",
                              model_first="on", stage_models=FIRST_MODELS)
        assert "started with model first off" in (
            results["ingest"]["failures"]["R36"])
        assert not (folder / "run.json").exists()

    def test_a_zero_token_ceiling_stops_before_the_first_report(
            self, ingest_cluster, capsys):
        _reports, out, _scripts = ingest_cluster
        results = _ingest_run(ingest_cluster, model="fake-main",
                              max_total_tokens=0)
        assert not (out / "ingest" / "R36").exists()
        assert results["ingest"]["totals"]["pages"] == 0
        assert "stopping before R36" in capsys.readouterr().out

    def test_the_default_has_no_token_limit(self, ingest_cluster):
        _reports, out, _scripts = ingest_cluster
        _ingest_run(ingest_cluster, model="fake-main")
        run = json.loads((out / "ingest" / "R36" / "run.json").read_text(
            "utf-8"))
        assert "token_brake" not in run


class TestTheRequestPacingSetting:
    """``score_on_cluster(tokens_per_minute=...)`` reaches the graph's
    ``Budgets`` (BACKLOG RI-127); unset it is None there. The budget is
    made up."""

    @staticmethod
    def _budgets(ingest_cluster, monkeypatch, **over):
        from report_ingest import graph

        seen = {}

        def stop(doc, engine, *args, **kwargs):
            seen["budgets"] = kwargs["budgets"]
            raise RuntimeError("made-up stop")

        monkeypatch.setattr(graph, "ingest_report", stop)
        _ingest_run(ingest_cluster, model="fake-main", **over)
        return seen["budgets"]

    def test_the_setting_reaches_the_budgets(self, ingest_cluster,
                                             monkeypatch):
        budgets = self._budgets(ingest_cluster, monkeypatch,
                                tokens_per_minute=123_467)
        assert budgets.tokens_per_minute == 123_467

    def test_unset_it_is_none(self, ingest_cluster, monkeypatch):
        assert self._budgets(ingest_cluster,
                             monkeypatch).tokens_per_minute is None


class TestThePlacementFile:
    def test_the_graph_writes_what_it_placed_and_replaces_it_each_run(
            self, tmp_path):
        from report_ingest import graph

        found = [{"counts": {"placed": 2, "not_found": 1}, "items": []}]
        graph._write_placement(str(tmp_path), {"mf-a": found})
        got = json.loads((tmp_path / "placement.json").read_text("utf-8"))
        assert got == {"mf-a": found}
        graph._write_placement(str(tmp_path), {})
        assert json.loads((tmp_path / "placement.json").read_text(
            "utf-8")) == {}

    def test_the_run_file_counts_the_placement_file_and_ignores_item_files(
            self, tmp_path):
        from report_ingest.cluster_scoring import _Engines
        from report_ingest.model_first import READ_PROMPT_VERSION

        folder = tmp_path / "R1"
        (folder / "items").mkdir(parents=True)
        (folder / "bound" / "b1").mkdir(parents=True)
        (folder / "placement.json").write_text(json.dumps({
            "mf-a": [{"counts": {"placed": 2, "not_found": 1}, "items": []}]}),
            "utf-8")
        (folder / "bound" / "b1" / "placement.json").write_text(json.dumps({
            "mf-b": [{"counts": {"placed": 4}, "items": []}]}), "utf-8")
        (folder / "items" / "mf-stale.json").write_text(json.dumps({
            "placement": [{"counts": {"placed": 50}, "items": []}]}), "utf-8")
        engine = _Counted()
        kwargs = {"stage_engines": {s: engine for s in cs.FIRST_STAGES},
                  "describe_engine": engine, "range_engine": engine}
        names = {s: "made-up" for s in cs.FIRST_STAGES}
        names.update(describer="made-up", labeller="made-up")
        fields = cs._ingest_arm_fields(cs._FirstArm("on"),
                                       _Engines.of(object()), kwargs,
                                       names, folder)
        assert fields["prompt_version"] == READ_PROMPT_VERSION
        assert fields["placement"]["counts"]["placed"] == 6
        assert fields["placement"]["counts"]["not_found"] == 1

    def test_a_stale_file_is_replaced_when_the_read_starts(
            self, tmp_path, monkeypatch):
        import types
        from report_ingest import graph

        monkeypatch.setattr(graph, "_join_by_hole", lambda *a, **k: None)
        (tmp_path / "placement.json").write_text(
            json.dumps({"mf-old": [{"counts": {"placed": 9}}]}), "utf-8")
        record = types.SimpleNamespace(investigations=[])
        graph._read_model_first(None, None, str(tmp_path), True, "R1", {},
                                None, record, None, [], [], "standard", [])
        # No units read: an empty file, never the earlier run's.
        assert json.loads((tmp_path / "placement.json").read_text(
            "utf-8")) == {}


def _metered(monkeypatch, scripts):
    """The fixture's fake engines, each charging its reply to the meter it
    is built on, as a real engine does (the fixture's do not)."""
    import report_ingest.engine as engine_module
    from report_ingest.engine import Usage
    from report_ingest.tests.fake_engine import FakeEngine

    def build(prompter, model, meter=None, **kwargs):
        turns = (scripts["vision"] if model == "funhouse-gpt-low"
                 else scripts["turns"])

        class Metered(FakeEngine):
            def complete(self, messages, **kw):
                reply = super().complete(messages, **kw)
                if meter is not None:
                    meter.add(model, Usage(input_tokens=1000,
                                           output_tokens=100), 0.0)
                return reply

        engine = Metered(list(turns or []), name=model)
        engine.served_by = "fake-deployment"
        return engine

    monkeypatch.setattr(engine_module, "PrompterEngine", build)


class TestTheBrakeInsideARun:
    def test_a_braked_report_is_marked_ends_the_run_and_is_retried(
            self, ingest_cluster, monkeypatch, capsys):
        from report_ingest.tests import test_graph as tg

        _reports, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _metered(monkeypatch, scripts)
        _ingest_run(ingest_cluster, model="fake-main",
                    max_report_tokens_in=2500)
        run_file = out / "ingest" / "R36" / "run.json"
        braked = json.loads(run_file.read_text("utf-8"))
        assert braked["token_brake"]["limit"] == "max_report_tokens_in"
        assert not braked.get("error")
        printed = capsys.readouterr().out
        assert "stopping after R36" in printed        # the run ended
        saved = {p.name for p in (out / "ingest" / "R36" / "items").glob(
            "*.json")} if (out / "ingest" / "R36" / "items").is_dir() else set()

        # A second call with no limit does not skip it as done.
        _ingest_run(ingest_cluster, model="fake-main")
        printed = capsys.readouterr().out
        assert "cut short by the token brake" in printed
        assert "already done" not in printed
        again = json.loads(run_file.read_text("utf-8"))
        assert "token_brake" not in again
        kept = {p.name for p in (out / "ingest" / "R36" / "items").glob(
            "*.json")}
        assert saved <= kept                        # saved items kept

    def test_what_a_failed_report_spent_counts_toward_the_total(
            self, ingest_cluster, monkeypatch, capsys):
        from report_ingest import graph

        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = []
        _metered(monkeypatch, scripts)
        (reports_dir / "R37.pdf").write_bytes(
            (reports_dir / "R36.pdf").read_bytes())
        monkeypatch.setattr(
            cs, "_set_ids",
            lambda name, corpus: ("R36", "R37") if name == "oos_open" else ())

        def boom(doc, engine, *args, **kwargs):
            engine.complete([{"role": "user", "content": "x"}])
            raise RuntimeError("made-up failure")

        monkeypatch.setattr(graph, "ingest_report", boom)
        scripts["turns"] = [{"final": None}]
        results = _ingest_run(ingest_cluster, model="fake-main",
                              max_total_tokens=1000)
        assert "made-up failure" in results["ingest"]["failures"]["R36"]
        assert "stopping before R37" in capsys.readouterr().out
        assert not (out / "ingest" / "R37" / "arm.json").exists()


class TestAGiveUpIsRetried:
    """BACKLOG RI-110: an item the model could not be got to read leaves the
    report unfinished, so a resume reads it again."""

    def test_a_forced_give_up_is_marked_counted_and_retried_on_resume(
            self, ingest_cluster, monkeypatch, capsys):
        from report_ingest import graph
        from report_ingest.tests import test_graph as tg

        _reports, out, scripts = ingest_cluster
        full = tg.main_script()
        scripts["turns"] = full[:5] + full[7:]      # the two lab turns out
        real = graph._read_lab

        def give_up(*args, **kwargs):
            raise RuntimeError("made-up rate limit: retries exhausted")

        monkeypatch.setattr(graph, "_read_lab", give_up)
        results = _ingest_run(ingest_cluster)
        run_file = out / "ingest" / "R36" / "run.json"
        saved = json.loads(run_file.read_text("utf-8"))
        assert saved["item_errors"]
        assert "made-up rate limit" in saved["item_errors"][0]["detail"]
        assert not saved.get("error")
        # The run's failed count includes the report with an unread item.
        assert "R36" in results["ingest"]["failures"]
        # ... and it is not listed among the finished reports.
        assert results["ingest"]["per_report"] == []
        assert results["ingest"]["n_reports"] == 0

        # A resume does not skip it as done; it reads the lab items again.
        monkeypatch.setattr(graph, "_read_lab", real)
        scripts["turns"] = [full[5], full[6]] + full[7:]
        _ingest_run(ingest_cluster)
        printed = capsys.readouterr().out
        assert "could not be read (a give-up); retrying" in printed
        assert "already done" not in printed
        again = json.loads(run_file.read_text("utf-8"))
        assert "item_errors" not in again

    def test_a_label_review_that_did_not_finish_leaves_the_report_unfinished(
            self, ingest_cluster, monkeypatch, capsys):
        """BACKLOG RI-173: a failed label review is a failed item of the
        ingest stage, in the totals too, and a resume asks for it again --
        not a report counted as done with only a partial QA entry."""
        from report_ingest import label_review
        from report_ingest.tests import test_graph as tg

        _reports, out, scripts = ingest_cluster
        asked = []

        def fails(*args, **kwargs):
            asked.append(1)
            raise RuntimeError("made-up review failure")

        monkeypatch.setattr(label_review, "review_labels", fails)
        scripts["turns"] = tg.main_script()
        results = _ingest_run(ingest_cluster, review_mode="all")
        saved = json.loads((out / "ingest" / "R36" / "run.json")
                           .read_text("utf-8"))
        (row,) = saved["item_errors"]
        assert row["where"] == "labels.review"
        assert "made-up review failure" in row["detail"]
        assert "R36" in results["ingest"]["failures"]
        assert results["ingest"]["per_report"] == []
        assert results["totals"]["failed_items_by_stage"]["ingest"] == ["R36"]
        assert not (out / "ingest" / "R36" / "review.json").exists()

        scripts["turns"] = tg.main_script()
        _ingest_run(ingest_cluster, review_mode="all")
        assert "already done" not in capsys.readouterr().out
        assert len(asked) == 2

    def test_results_cover_every_finished_report_in_the_folder(
            self, ingest_cluster, monkeypatch):
        from report_ingest.tests import test_graph as tg

        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _ingest_run(ingest_cluster)
        # A second call that is asked for another (absent) report only.
        monkeypatch.setattr(
            cs, "_set_ids",
            lambda name, corpus: ("R99",) if name == "oos_open" else ())
        results = _ingest_run(ingest_cluster)
        assert [r["id"] for r in results["ingest"]["per_report"]] == ["R36"]


# -- every attempt's cost (BACKLOG RI-124) -------------------------------------
#
# Every number below is made up for the test.

def _attempt_cost(calls, tokens_in, tokens_out, **extra):
    """A run file's cost block, as CostMeter.to_dict writes it, cut down."""
    cost = {"calls": calls, "failed": 0, "retried": 0, "failed_by_kind": {},
            "input_tokens": tokens_in, "output_tokens": tokens_out,
            "reasoning_tokens": 0, "reasoning_reported": 0,
            "cache_read_tokens": 0, "cache_write_tokens": 0,
            "seconds": 1.5, "waited_seconds": 0.0, "dollars": 0.0,
            "unpriced_calls": calls, "by_model": {"made-up-model": {}},
            "per_call_defaults": {}, "per_call": [{"call": 1}]}
    cost.update(extra)
    return cost


class TestTheBrakeCountsEveryAttempt:
    def test_the_earlier_attempts_input_counts_toward_the_report_limit(self):
        from report_ingest.engine import CallCeilingReached, Usage

        meter, inner = _meter(300), _Counted()
        brake = cs._TokenBrake(meter, 1000, None, earlier_in=650)
        engine = brake.wrap(inner)
        assert engine.complete([]) == "reply"       # 950 sent of 1,000
        meter.add("made-up-model", Usage(input_tokens=70), 0.1)
        with pytest.raises(CallCeilingReached, match="earlier attempts"):
            engine.complete([])
        assert inner.calls == 1
        assert brake.fields()["token_brake"] == {
            "limit": "max_report_tokens_in", "tokens_in": 370,
            "input_tokens": 370, "cache_write_tokens": 0,
            "cache_read_tokens": 0, "output_tokens": 0,
            "earlier_tokens_in": 650}

    def test_the_same_meter_alone_would_have_gone_on(self):
        meter, inner = _meter(370), _Counted()
        brake = cs._TokenBrake(meter, 1000, None)
        brake.wrap(inner).complete([])
        assert inner.calls == 1 and brake.fields() == {}

    def test_the_run_limit_still_counts_this_call_only(self):
        brake = cs._TokenBrake(_meter(100, 50), None, 1000, earlier_in=5000)
        brake.wrap(_Counted()).complete([])
        assert brake.fields() == {}


class TestEveryAttemptIsCarried:
    def test_an_attempt_is_read_off_the_run_file_it_replaces(self):
        braked = {"id": "R1", "run_date": "2026-01-02", "seconds": 12.5,
                  "token_brake": {"limit": "max_report_tokens_in"},
                  "cost": _attempt_cost(
                      4, 4100, 410, failed=2, retried=1,
                      failed_by_kind={"rate_limit": 2},
                      cache_read_tokens=30)}
        (entry,) = cs._earlier_attempts(braked)
        assert entry["ended"] == "token_brake"
        assert entry["run_date"] == "2026-01-02"
        assert entry["seconds"] == 12.5               # the file's wall clock
        assert entry["cost"]["calls"] == 4
        assert entry["cost"]["failed_by_kind"] == {"rate_limit": 2}
        assert entry["cost"]["cache_read_tokens"] == 30
        for key in ("per_call", "per_call_defaults", "by_model"):
            assert key not in entry["cost"]

    def test_attempts_chain_and_add_up(self):
        first = {"id": "R1", "run_date": "2026-01-02", "seconds": 12.5,
                 "item_errors": [{"where": "lab"}],
                 "cost": _attempt_cost(4, 4100, 410, failed=2, retried=1,
                                       failed_by_kind={"rate_limit": 2})}
        earlier = cs._earlier_attempts(first)
        # The second attempt failed outright: its file has no wall clock,
        # so its calls' own seconds stand in.
        cost = _attempt_cost(3, 2200, 230, failed=1,
                             failed_by_kind={"timeout": 1}, seconds=4.0)
        second = {"id": "R1", "run_date": "2026-01-03", "error": "made up",
                  "cost": cost,
                  **cs._attempts_fields(earlier, cost, cost["seconds"])}
        assert second["all_attempts"]["calls"] == 7
        both = cs._earlier_attempts(second)
        assert [e["ended"] for e in both] == ["item_errors", "error"]
        assert [e["seconds"] for e in both] == [12.5, 4.0]
        fields = cs._attempts_fields(both, _attempt_cost(5, 900, 90), 7.0)
        total = fields["all_attempts"]
        assert total["attempts"] == 3
        assert (total["calls"], total["failed"], total["retried"]) == (12, 3, 1)
        assert (total["input_tokens"], total["output_tokens"]) == (7200, 730)
        assert total["failed_by_kind"] == {"rate_limit": 2, "timeout": 1}
        assert total["seconds"] == 23.5
        assert len(fields["earlier_attempts"]) == 2
        assert cs._attempts_in(both) == 6300

    def test_a_first_attempt_adds_nothing(self):
        assert cs._attempts_fields([], _attempt_cost(2, 10, 1), 1.0) == {}

    def test_a_run_file_from_before_carries_no_attempts(self):
        assert cs._attempt_rows({"id": "R1", "cost": {}}) == []
        assert cs._attempt_rows({"earlier_attempts": "not a list"}) == []


def _ingest_row(rid, cost, seconds, earlier=None):
    row = {"id": rid, "n_pages": 3, "seconds": seconds, "cost": cost}
    if earlier is not None:
        row["earlier_attempts"] = earlier
    return row


class TestTheResultsCountEveryAttempt:
    def test_the_cost_block_counts_every_attempt_and_says_so(self):
        earlier = [{"run_date": "2026-01-02", "ended": "item_errors",
                    "seconds": 5.0,
                    "cost": _attempt_cost(8, 9000, 800, failed=7, retried=6,
                                          failed_by_kind={"rate_limit": 7},
                                          cache_read_tokens=100)}]
        rows = {"R7": _ingest_row("R7", _attempt_cost(3, 3000, 300), 2.0,
                                  earlier),
                "R8": _ingest_row("R8", _attempt_cost(2, 1000, 100), 1.0)}
        scored = cs._score_ingest(rows, {}, "m")
        cost = scored["cost"]
        assert (cost["calls"], cost["failed"]) == (13, 7)
        assert cost["input_tokens"] == 13000
        assert cost["failed_by_kind"] == {"rate_limit": 7}
        assert cost["seconds"] == 8.0
        own = scored["cost_last_attempt"]
        assert (own["calls"], own["failed"], own["input_tokens"]) == (
            5, 0, 4000)
        assert own["seconds"] == 3.0
        assert scored["reports_with_earlier_attempts"] == {"R7": 1}
        assert "every attempt" in scored["cost_covers"]
        text = "\n".join(cs._render_ingest(scored))
        assert ("13 model calls over every attempt, 13,100 tokens in "
                "(13,000 input, 0 cache write, 100 cache read), 1,200 "
                "output, 8 s") in text
        assert ("failed requests, not among the calls above: 7 (7 rate "
                "limited), 6 of them sent again") in text
        assert ("each report's last attempt alone: 5 model calls, 4,000 "
                "tokens in, 400 output, 3 s, 0 failed requests") in text
        assert "read in more than one attempt: R7 (2 attempts)" in text
        assert "count EVERY attempt" in text

    def test_with_no_report_read_twice_the_results_are_as_before(self):
        rows = {"R8": _ingest_row("R8", _attempt_cost(2, 1000, 100), 1.0)}
        scored = cs._score_ingest(rows, {}, "m")
        for key in ("cost_last_attempt", "reports_with_earlier_attempts",
                    "cost_covers"):
            assert key not in scored
        assert scored["cost"]["calls"] == 2
        text = "\n".join(cs._render_ingest(scored))
        assert "2 model calls, 1,000 tokens in, 100 output, 1 s" in text
        assert "every attempt" not in text
        assert "last attempt alone" not in text


class TestEveryAttemptInARun:
    def test_a_braked_report_retried_keeps_both_and_the_brake_counts_both(
            self, ingest_cluster, monkeypatch, capsys):
        from report_ingest.tests import test_graph as tg

        _reports, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _metered(monkeypatch, scripts)                # 1,000 in a call
        run_file = out / "ingest" / "R36" / "run.json"
        _ingest_run(ingest_cluster, model="fake-main",
                    max_report_tokens_in=2500)
        first = json.loads(run_file.read_text("utf-8"))
        assert first["token_brake"]["limit"] == "max_report_tokens_in"
        for key in ("earlier_attempts", "all_attempts"):
            assert key not in first                     # a first attempt
        assert "earlier_tokens_in" not in first["token_brake"]
        sent = first["cost"]["input_tokens"]
        assert sent == 1000 * first["cost"]["calls"] and sent >= 2500
        capsys.readouterr()

        # A limit 1,500 above what the first attempt sent: counting both
        # attempts, two more calls start (at +0 and +1,000) and the third
        # is refused; counting this attempt alone it would allow more.
        _ingest_run(ingest_cluster, model="fake-main",
                    max_report_tokens_in=sent + 1500)
        second = json.loads(run_file.read_text("utf-8"))
        assert second["cost"]["calls"] == 2
        assert second["token_brake"]["earlier_tokens_in"] == sent
        (entry,) = second["earlier_attempts"]
        assert entry["ended"] == "token_brake"
        assert entry["cost"]["calls"] == first["cost"]["calls"]
        assert entry["cost"]["input_tokens"] == sent
        both = second["all_attempts"]
        assert both["attempts"] == 2
        assert both["calls"] == first["cost"]["calls"] + 2
        assert both["input_tokens"] == sent + 2000
        printed = capsys.readouterr().out
        assert (f"stopping after R36: {sent + 2000:,} tokens in over its "
                f"attempts") in printed

        # A third call with no limit finishes the report: RESULTS counts
        # all three attempts and keeps the last beside them. The first
        # attempt saved the triage, so the readers' script starts after it.
        scripts["turns"] = tg.main_script()[1:]
        results = _ingest_run(ingest_cluster, model="fake-main")
        third = json.loads(run_file.read_text("utf-8"))
        assert "token_brake" not in third
        assert [e["ended"] for e in third["earlier_attempts"]] == [
            "token_brake", "token_brake"]
        calls = (first["cost"]["calls"] + 2 + third["cost"]["calls"])
        assert third["all_attempts"]["calls"] == calls
        ingest = results["ingest"]
        assert ingest["cost"]["calls"] == calls
        assert ingest["cost_last_attempt"]["calls"] == third["cost"]["calls"]
        assert ingest["reports_with_earlier_attempts"] == {"R36": 2}
        saved = json.loads((out / "results.json").read_text("utf-8"))
        assert saved["ingest"]["cost"]["calls"] == calls
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert f"{calls} model calls over every attempt" in text
        assert "R36 (3 attempts)" in text

    def test_a_report_whose_attempts_used_its_allowance_is_not_read_again(
            self, ingest_cluster, monkeypatch, capsys):
        from report_ingest.tests import test_graph as tg

        _reports, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _metered(monkeypatch, scripts)
        run_file = out / "ingest" / "R36" / "run.json"
        _ingest_run(ingest_cluster, model="fake-main",
                    max_report_tokens_in=2500)
        before = run_file.read_text("utf-8")
        sent = json.loads(before)["cost"]["input_tokens"]
        capsys.readouterr()
        results = _ingest_run(ingest_cluster, model="fake-main",
                              max_report_tokens_in=sent)
        assert "not retried" in capsys.readouterr().out
        assert "per-report ceiling" in results["ingest"]["failures"]["R36"]
        assert run_file.read_text("utf-8") == before   # left as it was

    def test_a_give_up_retried_keeps_both_attempts(
            self, ingest_cluster, monkeypatch):
        from report_ingest import graph
        from report_ingest.tests import test_graph as tg

        _reports, out, scripts = ingest_cluster
        full = tg.main_script()
        scripts["turns"] = full[:5] + full[7:]      # the two lab turns out
        _metered(monkeypatch, scripts)
        real = graph._read_lab

        def give_up(*args, **kwargs):
            raise RuntimeError("made-up rate limit: retries exhausted")

        monkeypatch.setattr(graph, "_read_lab", give_up)
        _ingest_run(ingest_cluster)
        run_file = out / "ingest" / "R36" / "run.json"
        first = json.loads(run_file.read_text("utf-8"))
        assert first["item_errors"] and first["cost"]["calls"] > 0
        assert "earlier_attempts" not in first

        monkeypatch.setattr(graph, "_read_lab", real)
        scripts["turns"] = [full[5], full[6]] + full[7:]
        results = _ingest_run(ingest_cluster)
        again = json.loads(run_file.read_text("utf-8"))
        assert "item_errors" not in again
        (entry,) = again["earlier_attempts"]
        assert entry["ended"] == "item_errors"
        assert entry["cost"]["calls"] == first["cost"]["calls"]
        assert entry["seconds"] == first["seconds"]
        both = again["all_attempts"]
        assert both["calls"] == (first["cost"]["calls"]
                                 + again["cost"]["calls"])
        assert both["input_tokens"] == (first["cost"]["input_tokens"]
                                        + again["cost"]["input_tokens"])
        cost = results["ingest"]["cost"]
        assert cost["calls"] == both["calls"]
        assert results["ingest"]["cost_last_attempt"]["calls"] == (
            again["cost"]["calls"])

    def test_an_error_then_a_finished_read_keeps_the_errors_cost(
            self, ingest_cluster, monkeypatch):
        from report_ingest import graph
        from report_ingest.tests import test_graph as tg

        _reports, out, scripts = ingest_cluster
        _metered(monkeypatch, scripts)
        real = graph.ingest_report

        def boom(doc, engine, *args, **kwargs):
            engine.complete([{"role": "user", "content": "x"}])
            raise RuntimeError("made-up failure")

        monkeypatch.setattr(graph, "ingest_report", boom)
        scripts["turns"] = [{"final": None}]
        _ingest_run(ingest_cluster, model="fake-main")
        run_file = out / "ingest" / "R36" / "run.json"
        failed = json.loads(run_file.read_text("utf-8"))
        assert failed["error"] and failed["cost"]["calls"] == 1
        assert "earlier_attempts" not in failed

        monkeypatch.setattr(graph, "ingest_report", real)
        scripts["turns"] = tg.main_script()
        results = _ingest_run(ingest_cluster, model="fake-main")
        done = json.loads(run_file.read_text("utf-8"))
        (entry,) = done["earlier_attempts"]
        assert entry["ended"] == "error"
        assert entry["cost"]["calls"] == 1
        assert entry["cost"]["input_tokens"] == 1000
        assert done["all_attempts"]["calls"] == done["cost"]["calls"] + 1
        assert results["ingest"]["cost"]["calls"] == (
            done["cost"]["calls"] + 1)

    def test_a_redo_starts_the_report_afresh(self, ingest_cluster,
                                             monkeypatch):
        from report_ingest.tests import test_graph as tg

        _reports, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _metered(monkeypatch, scripts)
        _ingest_run(ingest_cluster, model="fake-main",
                    max_report_tokens_in=2500)
        _ingest_run(ingest_cluster, model="fake-main", redo=True)
        again = json.loads((out / "ingest" / "R36" / "run.json").read_text(
            "utf-8"))
        assert "earlier_attempts" not in again
        assert "all_attempts" not in again

    def test_a_first_run_writes_what_it_always_wrote(self, ingest_cluster):
        from report_ingest.tests import test_graph as tg

        _reports, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        results = _ingest_run(ingest_cluster)
        run = json.loads((out / "ingest" / "R36" / "run.json").read_text(
            "utf-8"))
        for key in ("earlier_attempts", "all_attempts"):
            assert key not in run
        for key in ("cost_last_attempt", "reports_with_earlier_attempts",
                    "cost_covers", "cost_unfinished", "cost_every_report"):
            assert key not in results["ingest"]
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "over every attempt" not in text
        assert "last attempt alone" not in text
        assert "reports not finished" not in text
        assert "every report, finished or not" not in text


class TestUnfinishedReportsInTheCost:
    """BACKLOG RI-124: a report still unfinished when a call ends is not a
    result, but what it spent, over every attempt, is in the Cost block."""

    def test_a_braked_report_alone_is_counted_as_unfinished(
            self, ingest_cluster, monkeypatch):
        from report_ingest.tests import test_graph as tg

        _reports, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _metered(monkeypatch, scripts)                # 1,000 in a call
        results = _ingest_run(ingest_cluster, model="fake-main",
                              max_report_tokens_in=2500)
        ingest = results["ingest"]
        assert ingest["n_reports"] == 0 and ingest["cost"]["calls"] == 0
        left = ingest["cost_unfinished"]
        assert left["reports"] == ["R36"]
        assert (left["calls"], left["input_tokens"]) == (3, 3000)
        every = ingest["cost_every_report"]
        assert (every["calls"], every["input_tokens"]) == (3, 3000)
        assert "token brake" in ingest["failures"]["R36"]
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert ("reports not finished (listed under failures): 1, 3 calls, "
                "3,000 in / 300 out, 0 failed requests, over every "
                "attempt") in text
        assert ("every report, finished or not: 3 calls, 3,000 in / 300 "
                "out, 0 failed requests") in text

    def test_the_run_goes_on_after_a_report_skipped_at_its_limit(
            self, ingest_cluster, monkeypatch, capsys):
        from report_ingest.tests import test_graph as tg

        reports_dir, out, scripts = ingest_cluster
        (reports_dir / "R37.pdf").write_bytes(
            (reports_dir / "R36.pdf").read_bytes())
        monkeypatch.setattr(
            cs, "_set_ids",
            lambda name, corpus: ("R36", "R37") if name == "oos_open" else ())
        scripts["turns"] = tg.main_script()
        _metered(monkeypatch, scripts)
        _ingest_run(ingest_cluster, model="fake-main",
                    max_report_tokens_in=2500)
        assert "stopping after R36" in capsys.readouterr().out
        assert not (out / "ingest" / "R37").exists()  # the run ended there
        r36 = out / "ingest" / "R36" / "run.json"
        before = r36.read_text("utf-8")
        sent = json.loads(before)["cost"]["input_tokens"]

        # The same limit as R36 has already sent: R36 is skipped, and the
        # run goes on to R37, which the brake then stops in its turn.
        results = _ingest_run(ingest_cluster, model="fake-main",
                              max_report_tokens_in=sent)
        printed = capsys.readouterr().out
        assert "R36: not retried" in printed
        assert "stopping after R37" in printed
        assert r36.read_text("utf-8") == before
        r37 = json.loads((out / "ingest" / "R37" / "run.json").read_text(
            "utf-8"))
        assert r37["token_brake"]["limit"] == "max_report_tokens_in"
        ingest = results["ingest"]
        assert "per-report ceiling" in ingest["failures"]["R36"]
        assert "token brake" in ingest["failures"]["R37"]
        left = ingest["cost_unfinished"]
        assert left["reports"] == ["R36", "R37"]
        assert left["calls"] == (json.loads(before)["cost"]["calls"]
                                 + r37["cost"]["calls"])

    def test_a_report_left_unfinished_by_an_earlier_call_is_listed(
            self, ingest_cluster, monkeypatch):
        from report_ingest.tests import test_graph as tg

        _reports, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _metered(monkeypatch, scripts)
        _ingest_run(ingest_cluster, model="fake-main",
                    max_report_tokens_in=2500)
        # A second call asked for another (absent) report only.
        monkeypatch.setattr(
            cs, "_set_ids",
            lambda name, corpus: ("R99",) if name == "oos_open" else ())
        ingest = _ingest_run(ingest_cluster, model="fake-main")["ingest"]
        assert ingest["failures"]["R36"].startswith(
            "left unfinished by an earlier call: cut short by the token "
            "brake")
        assert ingest["cost_unfinished"]["reports"] == ["R36"]
        assert ingest["cost_unfinished"]["calls"] == 3

    def test_the_unfinished_line_counts_every_attempt_and_names_the_kinds(
            self):
        done = {"R8": _ingest_row("R8", _attempt_cost(2, 1000, 100), 1.0)}
        unfinished = {"R9": {
            "id": "R9", "error": "made up",
            "cost": _attempt_cost(1, 400, 0, failed=3, retried=2,
                                  failed_by_kind={"rate_limit": 3},
                                  seconds=2.0),
            "earlier_attempts": [{
                "run_date": "2026-01-02", "ended": "item_errors",
                "seconds": 6.0,
                "cost": _attempt_cost(5, 5000, 500, failed=1,
                                      failed_by_kind={"timeout": 1})}]}}
        scored = cs._score_ingest(done, {"R9": "made up"}, "m", unfinished)
        assert scored["cost"]["calls"] == 2           # finished reports only
        left = scored["cost_unfinished"]
        assert (left["calls"], left["failed"], left["input_tokens"]) == (
            6, 4, 5400)
        assert left["failed_by_kind"] == {"rate_limit": 3, "timeout": 1}
        assert left["seconds"] == 8.0
        every = scored["cost_every_report"]
        assert (every["calls"], every["failed"], every["input_tokens"]) == (
            8, 4, 6400)
        assert every["seconds"] == 9.0
        text = "\n".join(cs._render_ingest(scored))
        assert ("reports not finished (listed under failures): 1, 6 calls, "
                "5,400 in / 500 out, 4 failed requests (3 rate limited, 1 "
                "timed out), over every attempt") in text
        assert ("every report, finished or not: 8 calls, 6,400 in / 600 "
                "out, 4 failed requests, 9 s") in text
        assert "per report: 2.0 calls" in text        # averages: finished

    def test_no_unfinished_report_adds_nothing(self):
        done = {"R8": _ingest_row("R8", _attempt_cost(2, 1000, 100), 1.0)}
        for unfinished in (None, {}):
            scored = cs._score_ingest(done, {}, "m", unfinished)
            assert "cost_unfinished" not in scored
            assert "cost_every_report" not in scored


class TestIngestSwitchLinesRI128:

    @staticmethod
    def _row(rid, **switches):
        return {"id": rid, "cost": {}, "switches": switches}

    def test_differing_switches_say_mixed_with_counts(self):
        rows = [self._row("R90", ingest_bound=True, counts_from_logs="off",
                          narrative_skips_bound=False),
                self._row("R91", ingest_bound=True,
                          counts_from_logs="fill",
                          narrative_skips_bound=False)]
        text = "\n".join(cs._switch_lines(rows))
        assert ("`counts_from_logs`): fill for 1 report and off for 1 "
                "report -- MIXED") in text
        assert "`narrative_skips_bound`): off for 2 reports." in text
        assert "`ingest_bound`): on for 2 reports." in text

    def test_an_old_run_file_says_not_recorded_and_the_block_renders(self):
        text = "\n".join(cs._render_ingest(cs._score_ingest(
            {"R90": {"id": "R90", "cost": {}},
             "R91": {"id": "R91", "cost": {},
                     "switches": {"counts_from_logs": "off"}}}, {}, "m")))
        assert ("`counts_from_logs`): off for 1 report and not recorded "
                "for 1 report -- MIXED") in text


    def test_lab_sheets_per_call_is_a_switch_line_and_none_is_off(self):
        rows = [self._row("R90", lab_sheets_per_call=None),
                self._row("R91", lab_sheets_per_call=3)]
        text = "\n".join(cs._switch_lines(rows))
        assert ("`lab_sheets_per_call`): 3 for 1 report and off for 1 "
                "report -- MIXED") in text


# -- a token ceiling across builds (BACKLOG RI-133) ----------------------------
#
# Every number below is made up for the test.

def _folder_run(out, rid, **blob):
    folder = out / "ingest" / rid
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.json").write_text(
        json.dumps({"id": rid, **blob}), encoding="utf-8")


class TestTheCeilingCountsTheFolder:
    def test_the_folder_spend_counts_each_run_file_once(self, tmp_path):
        first = cs._FirstArm()
        # Finished, with the sum over its attempts.
        _folder_run(tmp_path, "R90", counts={"qa": 0},
                    cost=_attempt_cost(1, 700, 70),
                    all_attempts={"input_tokens": 3000, "output_tokens": 300})
        # Unfinished, no sum: its own cost plus the earlier attempts.
        _folder_run(tmp_path, "R91", error="made-up",
                    cost=_attempt_cost(1, 200, 20),
                    earlier_attempts=[{"cost": {"input_tokens": 50,
                                                "output_tokens": 5}}])
        # Finished, a first attempt.
        _folder_run(tmp_path, "R92", counts={"qa": 0},
                    cost=_attempt_cost(1, 400, 40))
        assert cs._folder_tokens(tmp_path, first) == (
            3300 + 220 + 55 + 440)
        assert cs._folder_tokens(tmp_path / "none", first) == 0

    def test_a_second_call_stops_at_the_ceiling_counting_the_first(
            self, ingest_cluster, monkeypatch, capsys):
        from report_ingest.tests import test_graph as tg

        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _metered(monkeypatch, scripts)
        _ingest_run(ingest_cluster, model="fake-main")
        spent = cs._folder_tokens(out, cs._FirstArm())
        assert spent > 0
        (reports_dir / "R37.pdf").write_bytes(
            (reports_dir / "R36.pdf").read_bytes())
        monkeypatch.setattr(
            cs, "_set_ids",
            lambda name, corpus: ("R36", "R37") if name == "oos_open" else ())
        capsys.readouterr()
        # The done report is skipped (counted once, by the folder); the
        # second call has spent nothing itself yet is already at the ceiling.
        results = _ingest_run(ingest_cluster, model="fake-main",
                              max_total_tokens=spent)
        printed = capsys.readouterr().out
        assert "R36: already done, skipping" in printed
        assert "stopping before R37" in printed
        assert f"the folder had spent {spent:,} tokens" in printed
        why = results["ingest"]["failures"]["(token ceiling)"]
        assert f"the folder had spent {spent:,} tokens" in why
        assert not (out / "ingest" / "R37" / "arm.json").exists()

    def test_a_ceiling_above_the_folder_spend_lets_the_call_run(
            self, ingest_cluster, monkeypatch):
        from report_ingest.tests import test_graph as tg

        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _metered(monkeypatch, scripts)
        results = _ingest_run(ingest_cluster, model="fake-main",
                              max_total_tokens=10 ** 9)
        assert "(token ceiling)" not in results["ingest"]["failures"]
        assert (out / "ingest" / "R36" / "arm.json").exists()

    def test_none_changes_nothing(self, ingest_cluster, monkeypatch, capsys):
        from report_ingest.tests import test_graph as tg

        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.main_script()
        _metered(monkeypatch, scripts)
        _folder_run(out, "R90", counts={"qa": 0},
                    cost=_attempt_cost(1, 10 ** 9, 0))
        results = _ingest_run(ingest_cluster, model="fake-main")
        printed = capsys.readouterr().out
        assert "token ceiling" not in printed
        assert "(token ceiling)" not in results["ingest"]["failures"]
        assert (out / "ingest" / "R36" / "arm.json").exists()
