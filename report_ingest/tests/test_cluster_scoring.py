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
