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
