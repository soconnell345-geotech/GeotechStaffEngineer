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

@pytest.mark.parametrize("bad", ["/Workspace/Users/x/out",
                                 "/dbfs/Workspace/out", "dbfs:/Workspace/o"])
def test_a_workspace_out_dir_is_refused_before_anything_runs(bad):
    with pytest.raises(ValueError, match="non-durable"):
        cs._check_out_dir(Path(bad))


@pytest.mark.parametrize("good", ["/tmp/report_ingest",
                                  "/Volumes/main/geo/out"])
def test_tmp_and_volume_paths_are_accepted(good):
    cs._check_out_dir(Path(good))


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
    assert "80,000 input tokens" in text
    assert "no per-token price" in text


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
    """The three sets of hand truth travel to the cluster as ONE folder.

    They are private, so they are uploaded by hand before every run. Three
    Volume paths that must each be right is three chances for one to be
    stale while the run still starts and scores against it; a root holding
    ``logs/``, ``lab/`` and ``narrative/`` is one thing to get right.
    """

    def test_a_root_with_the_three_subfolders_is_split_by_stage(self,
                                                               tmp_path):
        for name in ("logs", "lab", "narrative"):
            (tmp_path / name).mkdir()

        logs, lab, narrative = cs._truth_dirs(tmp_path, None, None)

        assert logs == tmp_path / "logs"
        assert lab == tmp_path / "lab"
        assert narrative == tmp_path / "narrative"

    def test_a_folder_of_log_truth_files_is_still_used_as_it_stands(self,
                                                                   tmp_path):
        """The older single-folder form: no `logs/` inside, so it IS the
        log truth folder. Breaking this would silently stop finding truth
        that is right there."""
        (tmp_path / "R06_p51.json").write_text("{}", encoding="utf-8")

        logs, lab, narrative = cs._truth_dirs(tmp_path, None, None)

        assert logs == tmp_path
        assert lab is None and narrative is None

    def test_an_explicit_per_stage_folder_wins_over_the_root(self, tmp_path):
        for name in ("logs", "lab", "narrative"):
            (tmp_path / name).mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()

        logs, lab, narrative = cs._truth_dirs(tmp_path, elsewhere, elsewhere)

        assert logs == tmp_path / "logs"
        assert lab == elsewhere and narrative == elsewhere

    def test_no_truth_at_all_is_three_nones(self):
        assert cs._truth_dirs(None, None, None) == (None, None, None)

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
        assert "26,000 input tokens" in block
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

    def test_only_out_dir_is_refused_for_workspace(self):
        """A durable_dir is the owner's own finding and is not second-guessed."""
        with pytest.raises(ValueError, match="non-durable"):
            cs._check_out_dir(Path("/Workspace/Users/x/out"))
        assert cs.Mirror(durable_dir="/Workspace/somewhere").active


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
    scripts = {"turns": [], "engines": []}

    def fake_engine(prompter, model, meter=None, **kwargs):
        engine = FakeEngine(list(scripts["turns"]), name=model)
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
        scripts["turns"] = tg.full_script()
        results = _ingest_run(ingest_cluster)

        folder = out / "ingest" / "R36"
        for name in ("report.record.json", "report.summary.md",
                     "report.page.md", "report.diggs.xml", "qa.json",
                     "run.json", "triage.json", "review.json"):
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
        assert row["diggs"]["schema"] in ("valid", "not checked here")
        assert row["diggs"]["roundtrip"] == "equal"
        assert row["qa_by_kind"].get("disagreement", 0) >= 1
        assert row["triage_reused"] is False and row["review_reused"] is False
        assert scripts["engines"][0].n_calls == len(tg.full_script())

    def test_results_md_carries_the_section_and_names_nobody(
            self, ingest_cluster):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.full_script()
        _ingest_run(ingest_cluster)
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "# Ingest: the record and its exports" in text
        assert "## Per report" in text and "## Totals" in text
        assert "R36" in text and "1 boring, 1 test_pit" in text
        assert "yes/valid/equal" in text or "yes/not checked here/equal" in text
        assert "Rosewood" not in text          # the project name stays out
        blob = json.loads((out / "results.json").read_text(encoding="utf-8"))
        assert blob["ingest"]["per_report"][0]["id"] == "R36"

    def test_a_second_call_is_free(self, ingest_cluster, capsys):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.full_script()
        _ingest_run(ingest_cluster)
        scripts["turns"] = []
        results = _ingest_run(ingest_cluster)
        assert "R36: already done, skipping" in capsys.readouterr().out
        assert results["ingest"]["n_reports"] == 1
        assert len(scripts["engines"]) == 1     # no second engine was built

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
        scripts["turns"] = ([tg.narrative_turn()]
                            + [tg.log_turn("B-1"),
                               tg.log_turn("TP-1", "test_pit")]
                            + [tg.lab_turn("atterberg", 12),
                               tg.lab_turn("gradation", 13)])
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
        scripts["turns"] = tg.full_script()
        results = _ingest_run(ingest_cluster, truth_dir=truth)
        (row,) = results["ingest"]["per_report"]
        (log_score,) = row["scores"]["logs"]
        assert log_score["id"] == "R36_p7"
        assert log_score["overall"]["found"] >= 3
        assert row["scores"]["narrative"]["recall"]["total"] > 0
        assert row["scores"]["lab"] == []
        text = (out / "RESULTS.md").read_text(encoding="utf-8")
        assert "## Scored against the hand truth" in text
        totals = results["ingest"]["totals"]["scores"]
        assert totals["logs"]["n"] == 1 and totals["narrative"]["n"] == 1

    def test_a_report_the_folder_does_not_have_is_named_and_skipped(
            self, ingest_cluster, monkeypatch):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        monkeypatch.setattr(
            cs, "_set_ids",
            lambda name, corpus: ("R36", "R31") if name == "oos_open" else ())
        scripts["turns"] = tg.full_script()
        results = _ingest_run(ingest_cluster)
        assert "R31" in results["absent"]
        assert results["ingest"]["n_reports"] == 1

    def test_the_run_files_are_mirrored_like_everything_else(
            self, ingest_cluster, tmp_path):
        from report_ingest.tests import test_graph as tg
        reports_dir, out, scripts = ingest_cluster
        scripts["turns"] = tg.full_script()
        durable = tmp_path / "durable"
        _ingest_run(ingest_cluster, durable_dir=durable)
        assert (durable / out.name / "ingest" / "R36" / "run.json").is_file()
        assert (durable / out.name / "ingest" / "R36"
                / "report.diggs.xml").is_file()
