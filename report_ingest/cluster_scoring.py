"""Score the two model passes on the cluster, against the real model.

The app runs in Funhouse against OpenAI models through Prompter. A score
measured on any other model measures a model that will never do the work, so
the numbers that count come from here: one function the owner calls in a
notebook with the live ``fh_prompter`` object already in hand.

    from report_ingest.cluster_scoring import score_on_cluster

    score_on_cluster(
        reports_dir = "/Volumes/main/geotech/reports",
        manifest    = "/Volumes/main/geotech/wp1b/MANIFEST.md",
        labels_xlsx = "/Volumes/main/geotech/wp1b/trial_pages_working_r2.xlsx",
        oos_labels  = "/Volumes/main/geotech/wp1b/oos_labels.json",
        di_dir      = "/Volumes/main/geotech/report_di",
        out_dir     = "/tmp/report_ingest_wp1b",
        prompter    = fh_prompter,
        model       = "funhouse-gpt-high",   # the tier the app runs on
    )

THE REPORTS STAY WHERE THEY ARE. ``reports_dir`` is the folder the owner
already has on the cluster, under the file names their authors gave them;
the manifest's source-file column is what turns a file name into an ID, so
that one small file has to travel with the run. The DI results are read in
either form -- ``<ID>.json.gz`` or ``DI_data_<original stem>.json``.

NO CREDENTIAL IS READ OR STORED. Authentication is whatever the passed-in
``fh_prompter`` was built with. Nothing here touches an environment
variable, a key file or a secret scope.

WHERE THE OUTPUT GOES. ``out_dir`` must be ``/tmp`` or a Volume. Writes to
``/Workspace`` are non-durable and permission-blocked on this cluster
(``docs/DATABRICKS_INSTALL.md``), and a run that appears to succeed and
leaves nothing behind is the worst outcome, so a ``/Workspace`` path is
refused up front rather than discovered at the end.

IT IS RESTARTABLE. Every report writes ``runs/<ID>.json`` as it finishes and
a later call skips any report that already has one. A detached notebook, an
expired token or a 429 storm costs the reports that had not finished, not
the ones that had. Delete a file to redo that report; pass ``redo=True`` to
redo all of them.

WHAT IT WRITES into ``out_dir``::

    runs/<ID>.json        rules, profile, review, cost -- one per report
    triage/<ID>.json      the document profile alone, for the owner's audit
    results.json          every summary table as data
    RESULTS.md            the same tables to read, and to bring back

``RESULTS.md`` is the file to bring home. It carries IDs, labels, counts and
rates and nothing else -- never a page heading, a change's reason or a triage
rationale, any of which can name a firm, a project or a person. Those stay
in ``runs/`` and ``triage/``, which stay on the cluster unless the owner
moves them deliberately.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from report_ingest.corpus import Corpus
from report_ingest.scoring import (
    CHECKPOINT, GATE, KEY_CONTENT, OOS_BLIND, OOS_OPEN, Scores, disputed_drop,
    gate_failures, label_table, verdict_for,
)

__all__ = ["score_on_cluster", "SET_NAMES", "STAGE_NAMES"]

SET_NAMES: Tuple[str, ...] = ("insample", "oos_open", "oos_blind")

#: The measurements this run can make. ``labels`` is WP1b -- triage and the
#: label review over whole reports. ``logs`` is WP2b -- the log grid and then
#: the log reader over each hand-truthed log, scored before and after.
#: ``lab`` is WP3 -- the page's own tables and then the lab reader over each
#: hand-truthed laboratory sheet, scored the same two ways.
STAGE_NAMES: Tuple[str, ...] = ("labels", "logs", "lab", "narrative")

#: The reports the log rules were allowed to be tuned on, when no ``OPEN.txt``
#: sits beside the truth files. Everything else is scored as blind.
DEFAULT_OPEN_LOGS: Tuple[str, ...] = ("R36", "R37", "R06", "R07", "R15",
                                      "R28")
#: The same, for the laboratory sheets.
DEFAULT_OPEN_LAB: Tuple[str, ...] = ("R36", "R28", "R17", "R06")

#: Paths a run must not write to. A Workspace write looks like it worked and
#: then is not there.
_REFUSED_PREFIXES = ("/Workspace", "/dbfs/Workspace", "dbfs:/Workspace")


def _check_out_dir(out_dir: Path) -> None:
    text = str(out_dir).replace("\\", "/")
    for bad in _REFUSED_PREFIXES:
        if text.startswith(bad):
            raise ValueError(
                f"out_dir {out_dir} is under {bad}, where writes are "
                f"non-durable and permission-blocked on this cluster. Use "
                f"/tmp/... or a Volume path (/Volumes/...).")


#: The subfolder each scoring stage keeps its hand truth in, under one truth
#: root. The names are the stage names, so ``stages`` and the folder listing
#: read the same way.
TRUTH_SUBDIRS: Dict[str, str] = {"logs": "logs", "lab": "lab",
                                 "narrative": "narrative"}


def _truth_dirs(truth_dir: Any, lab_truth_dir: Any, narrative_truth_dir: Any
                ) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """``(logs, lab, narrative)`` truth folders, from a ROOT or from three.

    ONE FOLDER TRAVELS TO THE CLUSTER, not three. The hand truth is private,
    so it is uploaded by hand before every run; asking for three Volume paths
    that must each be right invites one of them to be stale while the run
    still starts and scores the wrong thing. So ``truth_dir`` may be a root
    holding ``logs/``, ``lab/`` and ``narrative/`` -- the stage names -- and
    each stage takes its own folder from it.

    The older three-argument form still works and always wins: an explicit
    ``lab_truth_dir`` or ``narrative_truth_dir`` overrides the root, and a
    ``truth_dir`` that holds the log truth files DIRECTLY, with no ``logs/``
    in it, is used as-is the way it always was. Nothing here touches the
    disk beyond asking whether a subfolder exists, and a missing folder is
    left as ``None`` so the caller raises the message that names the stage.
    """
    root = Path(truth_dir) if truth_dir is not None else None

    def under(name: str) -> Optional[Path]:
        if root is None:
            return None
        candidate = root / TRUTH_SUBDIRS[name]
        return candidate if candidate.is_dir() else None

    logs = under("logs") or root
    lab = Path(lab_truth_dir) if lab_truth_dir is not None else under("lab")
    narrative = (Path(narrative_truth_dir)
                 if narrative_truth_dir is not None else under("narrative"))
    return logs, lab, narrative


def _set_ids(name: str, corpus: Corpus) -> Tuple[str, ...]:
    if name == "insample":
        return tuple(corpus.mapped_ids()) if corpus.labels_available else ()
    if name == "oos_open":
        return OOS_OPEN
    if name == "oos_blind":
        return OOS_BLIND
    raise ValueError(f"unknown set {name!r}; the sets are {SET_NAMES}")


def _truth_for(rid: str, corpus: Corpus, oos: Dict[str, Dict[int, dict]],
               mapped: Sequence[str]
               ) -> Tuple[Dict[int, str], Dict[int, Tuple[str, ...]], str]:
    """``(label per page, acceptable alternates, where it came from)``.

    The spreadsheet wins when a report has one: it labels every page, and the
    out-of-sample file samples five.
    """
    if rid in mapped:
        try:
            return ({pl.page0: pl.label for pl in corpus.labels_for(rid)}, {},
                    "spreadsheet")
        except (FileNotFoundError, KeyError, ImportError):
            pass
    rows = oos.get(rid)
    if rows:
        return ({p: v["label"] for p, v in rows.items()},
                {p: tuple(v.get("alternates") or ()) for p, v in rows.items()},
                "lead, 5 pages")
    return {}, {}, "none"


def _run_one(rid: str, corpus: Corpus, prompter: Any, model: str,
             triage_model: str, out_dir: Path) -> dict:
    """Rules, triage and review on one report. Writes its own run file."""
    from planlens.document.roles import document_outline, page_roles

    from report_ingest.engine import CostMeter, PrompterEngine
    from report_ingest.label_review import review_labels
    from report_ingest.triage import document_facts, triage

    started = time.time()
    meter = CostMeter()
    doc = corpus.open_report(rid, di="auto", warn=False)
    try:
        roles = page_roles(doc)
        outline = document_outline(doc)
        facts = document_facts(doc, roles)
        triage_engine = PrompterEngine(prompter, triage_model, meter=meter)
        profile = triage(doc, roles, outline, engine=triage_engine,
                         facts=facts)
        review_engine = PrompterEngine(prompter, model, meter=meter)
        review = review_labels(doc, roles, outline, profile,
                               engine=review_engine)
        rules = {r.page: r.role for r in roles}
        n_pages = doc.n_pages
        served_by = sorted({x for x in (triage_engine.served_by,
                                        review_engine.served_by) if x})
    finally:
        doc.close()

    blob = {
        "id": rid,
        "run_date": date.today().isoformat(),
        "n_pages": n_pages,
        "engine": "prompter",
        "triage_model": triage_model,
        "review_model": model,
        "served_by": served_by,
        "rules_labels": {str(k): v for k, v in sorted(rules.items())},
        "profile": profile.to_dict(),
        "review": review.to_dict(),
        "cost": meter.to_dict(),
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / "runs" / f"{rid}.json").write_text(
        json.dumps(blob, indent=2), encoding="utf-8")
    (out_dir / "triage" / f"{rid}.json").write_text(
        json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return blob


def score_on_cluster(reports_dir: Any = None, labels_xlsx: Any = None,
                     di_dir: Any = None, out_dir: Any = "/tmp/report_ingest",
                     prompter: Any = None, model: str = "funhouse-gpt-high",
                     sets: Sequence[str] = SET_NAMES, *,
                     manifest: Any = None,
                     triage_model: Optional[str] = None,
                     oos_labels: Any = None,
                     max_total_dollars: Optional[float] = None,
                     max_reports: Optional[int] = None,
                     redo: bool = False,
                     corpus_dir: Any = None,
                     stages: Sequence[str] = ("labels",),
                     truth_dir: Any = None,
                     lab_truth_dir: Any = None,
                     narrative_truth_dir: Any = None,
                     log_budget: int = 6,
                     lab_budget: int = 4,
                     narrative_budget: int = 8,
                     open_reports: Optional[Sequence[str]] = None,
                     open_lab_reports: Optional[Sequence[str]] = None,
                     open_narrative_reports: Optional[Sequence[str]] = None
                     ) -> Dict[str, Any]:
    """Run the rules, triage and the label review over the corpus, and score.

    Parameters
    ----------
    reports_dir
        The folder holding the report PDFs, named either ``R01.pdf`` ... or
        as their authors named them. In the second case the manifest is what
        says which file is which, so it is required. (``corpus_dir`` is the
        older name for this argument and still works.)
    manifest
        The corpus manifest (``MANIFEST.md``). Defaults to one sitting beside
        the reports; pass it when the reports folder is not yours to write
        to, which on the cluster it is not.
    labels_xlsx, di_dir
        The hand-label spreadsheet and the Azure Document Intelligence
        results (``<ID>.json.gz`` or ``DI_data_<original stem>.json``).
        Without the spreadsheet the run measures but cannot score the
        in-sample set; without the DI results the scanned pages are read
        from their own text layer alone.
    out_dir
        Where everything is written. ``/tmp/...`` or a Volume, never
        ``/Workspace``.
    prompter
        The live ``fh_prompter``. Required: there is no other way in.
    model, triage_model
        Funhouse tiers. ``triage_model`` defaults to ``model``; triage is one
        call over a ledger and a cheaper tier is usually enough, the review
        is the one that needs the reasoning.
    sets
        Which of ``insample``, ``oos_open``, ``oos_blind`` to run. Together
        they are all 38 reports, so the default gives every report a triage
        profile.
    oos_labels
        The lead's out-of-sample labels (``labels.json``). Without it the two
        out-of-sample sets run and produce profiles but score nothing.
    max_total_dollars
        Stop before starting another report once the run has spent this
        much. Funhouse publishes no per-token price for a tier, so this only
        bites if a price is known; ``max_reports`` is the reliable cap.
    redo
        Re-run reports that already have a run file. Off by default, which is
        what makes a detached notebook cheap to resume.
    stages
        Which measurements to make. ``("labels",)`` is WP1b -- triage and the
        label review over whole reports. ``"logs"`` is WP2b -- ``log_grid``
        and then the log reader over each hand-truthed log, scored before
        (the grid alone) and after (the record the reader built). ``"lab"``
        is WP3 -- the page's own detected tables and then the lab reader over
        each hand-truthed laboratory sheet, scored the same two ways.
        ``"narrative"`` is WP4 -- the narrative reader over every report that
        has a hand answer, scored field by field on recall, precision and the
        flattering agreement. Pass any combination; all four is
        ``stages=("labels", "logs", "lab", "narrative")``.
    truth_dir
        ONE truth root for every scoring stage: a folder holding ``logs/``,
        ``lab/`` and ``narrative/``, named after the stages that read them.
        The hand truth is private, so it is uploaded by hand before a run and
        does not live in the wheel; one folder to upload is one thing to get
        right rather than three. A ``truth_dir`` holding the log truth files
        directly, with no ``logs/`` in it, still works as it always did.
    lab_truth_dir, narrative_truth_dir
        The older per-stage form, and it still overrides the root. Pass them
        when the three sets of truth are not in one place.

        In every case an ``OPEN.txt`` beside the truth files names the
        reports whose pages the prompts were allowed to be tuned against;
        everything else is scored as blind.
    log_budget, lab_budget
        Model calls a reader may spend per log and per sheet. Each reader's
        own ceiling -- six and four -- holds whatever these say.
    open_reports, open_lab_reports
        Override an open set instead of reading its ``OPEN.txt``.
    """
    if prompter is None:
        raise ValueError(
            "pass the live fh_prompter; this module reads no credential of "
            "its own")
    reports_dir = reports_dir if reports_dir is not None else corpus_dir
    if reports_dir is None:
        raise ValueError("pass reports_dir: the folder holding the PDFs")
    stages = tuple(stages)
    unknown = [s for s in stages if s not in STAGE_NAMES]
    if unknown:
        raise ValueError(
            f"unknown stage(s) {unknown}; the stages are {list(STAGE_NAMES)}")
    if not stages:
        raise ValueError(f"pass at least one stage: {list(STAGE_NAMES)}")
    logs_truth, lab_truth, narrative_truth = _truth_dirs(
        truth_dir, lab_truth_dir, narrative_truth_dir)
    if "logs" in stages and logs_truth is None:
        raise ValueError(
            "the 'logs' stage scores the reader against the hand-truthed "
            "logs, so it needs truth_dir -- either the folder of "
            "<ID>_p<page>.json files itself, or one truth root with a "
            "'logs/' folder in it. They are private and do not ship in the "
            "wheel.")
    if "lab" in stages and lab_truth is None:
        raise ValueError(
            "the 'lab' stage scores the reader against the hand-truthed "
            "laboratory sheets, so it needs them: either lab_truth_dir -- "
            "the folder of <kind>__<ID>_p<page>.json files -- or a 'lab/' "
            "folder inside truth_dir. They are private and do not ship in "
            "the wheel.")
    if "narrative" in stages and narrative_truth is None:
        raise ValueError(
            "the 'narrative' stage scores the reader against the hand "
            "answers, so it needs them: either narrative_truth_dir -- the "
            "folder of <ID>.json files, each holding the owner's two schemas "
            "answered by hand with null for 'not stated' -- or a "
            "'narrative/' folder inside truth_dir. They are private and do "
            "not ship in the wheel.")
    out = Path(out_dir)
    _check_out_dir(out)
    for sub in ("runs", "triage"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if "logs" in stages:
        (out / "logs").mkdir(parents=True, exist_ok=True)
    if "lab" in stages:
        (out / "lab").mkdir(parents=True, exist_ok=True)
    if "narrative" in stages:
        (out / "narrative").mkdir(parents=True, exist_ok=True)

    corpus = Corpus(reports_dir, manifest=manifest, di_dir=di_dir,
                    labels_xlsx=labels_xlsx, cache_dir=out)
    if not corpus.available:
        raise FileNotFoundError(
            f"no reports at {reports_dir}: expected R01.pdf ... R38.pdf "
            f"there, or the original file names plus a manifest "
            f"({corpus.manifest}) whose source-file column names them")
    triage_model = triage_model or model

    oos: Dict[str, Dict[int, dict]] = {}
    if oos_labels:
        blob = json.loads(Path(oos_labels).read_text(encoding="utf-8"))
        oos = {rid: {int(p): v for p, v in pages.items()}
               for rid, pages in blob.items()}
    mapped = corpus.mapped_ids() if corpus.labels_available else []

    asked: List[str] = []
    if "labels" in stages:
        for name in sets:
            for rid in _set_ids(name, corpus):
                if rid not in asked:
                    asked.append(rid)
    # A cluster folder may hold a subset of the 38 the manifest lists. Say
    # which are not there once, up front, rather than failing them one at a
    # time deep in the run; a report already scored keeps its run file.
    present = set(corpus.present_ids())
    wanted = [rid for rid in asked
              if rid in present or (out / "runs" / f"{rid}.json").is_file()]
    absent = [rid for rid in asked if rid not in wanted]
    if max_reports:
        wanted = wanted[:int(max_reports)]

    print(f"report ingest on the cluster: stage(s) {', '.join(stages)}; "
          f"model {model}, triage {triage_model}; out_dir {out}")
    if "labels" in stages:
        print(f"  labels: {len(wanted)} report(s)")
    if absent:
        print(f"  not in {corpus.reports_dir} and skipped: "
              f"{', '.join(absent)}")
    spent = 0.0
    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, rid in enumerate(wanted, 1):
        run_file = out / "runs" / f"{rid}.json"
        if run_file.is_file() and not redo:
            done[rid] = json.loads(run_file.read_text(encoding="utf-8"))
            print(f"  [{n}/{len(wanted)}] {rid}: already done, skipping")
            continue
        if max_total_dollars is not None and spent > max_total_dollars:
            print(f"  stopping before {rid}: spent ${spent:.2f}, past the "
                  f"${max_total_dollars:.2f} ceiling")
            break
        try:
            blob = _run_one(rid, corpus, prompter, model, triage_model, out)
        except KeyboardInterrupt:
            print("  interrupted; what is finished is on disk and a later "
                  "call resumes")
            break
        except Exception as exc:                     # keep the run going
            failures[rid] = f"{type(exc).__name__}: {exc}"
            print(f"  [{n}/{len(wanted)}] {rid}: FAILED -- {failures[rid]}")
            traceback.print_exc()
            continue
        done[rid] = blob
        spent += blob["cost"].get("dollars", 0.0)
        review = blob["review"]
        print(f"  [{n}/{len(wanted)}] {rid}: {blob['n_pages']} pp, "
              f"{len(review['changes'])} change(s), "
              f"{review['tool_calls']}/{review['budget']} tool calls, "
              f"{blob['cost']['calls']} model calls, "
              f"{blob['cost']['input_tokens']:,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")

    results = _score(done, corpus, oos, mapped, sets, model, triage_model,
                     failures, absent)
    results["stages"] = list(stages)
    lines = _render(results) if "labels" in stages else _header(results)

    if "logs" in stages:
        logs = _run_logs(corpus, prompter, model, out, logs_truth,
                         budget=log_budget, redo=redo,
                         max_logs=max_reports, open_reports=open_reports)
        results["logs"] = logs
        lines += _render_logs(logs)

    if "lab" in stages:
        lab = _run_lab(corpus, prompter, model, out, lab_truth,
                       budget=lab_budget, redo=redo, max_sheets=max_reports,
                       open_reports=open_lab_reports)
        results["lab"] = lab
        lines += _render_lab(lab)

    if "narrative" in stages:
        narrative = _run_narrative(
            corpus, prompter, model, out, narrative_truth,
            budget=narrative_budget, redo=redo, max_reports=max_reports,
            open_reports=open_narrative_reports)
        results["narrative"] = narrative
        lines += _render_narrative(narrative)

    (out / "results.json").write_text(json.dumps(_plain(results), indent=2),
                                      encoding="utf-8")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out / 'RESULTS.md'} -- that is the file to bring back")
    print(f"per-report runs in {out / 'runs'}, profiles in {out / 'triage'}")
    if "logs" in stages:
        print(f"per-log runs in {out / 'logs'}")
    if "lab" in stages:
        print(f"per-sheet runs in {out / 'lab'}")
    if "narrative" in stages:
        print(f"per-report narrative runs in {out / 'narrative'}")
    return results


def _score(done: Dict[str, dict], corpus: Corpus,
           oos: Dict[str, Dict[int, dict]], mapped: Sequence[str],
           sets: Sequence[str], model: str, triage_model: str,
           failures: Dict[str, str],
           absent: Sequence[str] = ()) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "date": date.today().isoformat(),
        "engine": "prompter",
        "review_model": model,
        "triage_model": triage_model,
        "served_by": sorted({s for b in done.values()
                             for s in (b.get("served_by") or [])}),
        "failures": dict(failures),
        "absent": list(absent),
        "sets": {},
        "triage": [],
        "totals": {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                   "cache_read_tokens": 0, "dollars": 0.0, "seconds": 0.0},
    }
    for rid in sorted(done):
        blob = done[rid]
        cost = blob["cost"]
        for key in ("calls", "input_tokens", "output_tokens",
                    "cache_read_tokens"):
            out["totals"][key] += cost.get(key, 0)
        out["totals"]["dollars"] += cost.get("dollars", 0.0)
        out["totals"]["seconds"] += blob.get("seconds", 0.0)
        profile = blob["profile"]
        out["triage"].append({
            "id": rid,
            "document_type": profile["document_type"],
            "workflow": profile["workflow"],
            "bound_together": len(profile["bound_together"]),
            "toc_agreement": profile["toc_agreement"],
            "scan_fraction": profile["scan_fraction"],
        })

    for name in sets:
        ids = [rid for rid in _set_ids(name, corpus) if rid in done]
        before, after = Scores(), Scores()
        clean_before, clean_after = Scores(), Scores()
        verdicts: Dict[str, int] = {}
        per_report: List[dict] = []
        dropped = 0
        for rid in ids:
            blob = done[rid]
            rules = {int(k): v for k, v in blob["rules_labels"].items()}
            final = {int(k): v
                     for k, v in blob["review"]["final_labels"].items()}
            hand, alternates, source = _truth_for(rid, corpus, oos, mapped)
            never_seen = name == "oos_blind" and rid not in CHECKPOINT
            hits_before = hits_after = scored = 0
            for page, want in sorted(hand.items()):
                alts = alternates.get(page, ())
                was, now = rules.get(page, "other"), final.get(page, "other")
                if disputed_drop(rid, page, now):
                    dropped += 1
                    continue
                before.add(want, was, alts)
                after.add(want, now, alts)
                if never_seen:
                    clean_before.add(want, was, alts)
                    clean_after.add(want, now, alts)
                scored += 1
                hits_before += int(was == want)
                hits_after += int(now == want)
            for change in blob["review"]["changes"]:
                verdict = verdict_for(rid, int(change["page"]),
                                      change["from"], change["to"],
                                      hand.get(int(change["page"])))
                verdicts[verdict] = verdicts.get(verdict, 0) + 1
            per_report.append({
                "id": rid, "pages": blob["n_pages"], "scored": scored,
                "truth": source,
                "before": (hits_before / scored) if scored else None,
                "after": (hits_after / scored) if scored else None,
                "changes": len(blob["review"]["changes"]),
                "rejected": len(blob["review"]["rejected_changes"]),
                "unresolved": len(blob["review"]["unresolved"]),
                "tool_calls": blob["review"]["tool_calls"],
                "budget": blob["review"]["budget"],
                "calls": blob["cost"]["calls"],
                "input_tokens": blob["cost"]["input_tokens"],
                "output_tokens": blob["cost"]["output_tokens"],
                "seconds": blob["seconds"],
            })
        row: Dict[str, Any] = {
            "reports": ids,
            "n_reports": len(ids),
            "disputed_dropped": dropped,
            "before": before.to_dict(),
            "after": after.to_dict(),
            "verdicts": verdicts,
            "gate_failures": gate_failures(after),
            "per_report": per_report,
            "_scores": (before, after),
        }
        if name == "oos_blind" and clean_after.n:
            row["never_seen"] = {
                "reports": [r for r in ids if r not in CHECKPOINT],
                "excluded": [r for r in ids if r in CHECKPOINT],
                "before": clean_before.to_dict(),
                "after": clean_after.to_dict(),
                "gate_failures": gate_failures(clean_after),
                "_scores": (clean_before, clean_after),
            }
        out["sets"][name] = row
    return out


def _plain(value: Any) -> Any:
    """The results with the live :class:`Scores` objects taken out.

    ``_scores`` is how the renderer gets at the rates without recomputing
    them; it cannot be serialized, and a key beginning with an underscore is
    the signal that it is working state rather than a result.
    """
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()
                if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# the logs stage
# ---------------------------------------------------------------------------

def _open_set(truth_dir: Path,
              override: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """The reports the log rules were allowed to be tuned on."""
    if override:
        return tuple(override)
    open_file = truth_dir / "OPEN.txt"
    if open_file.is_file():
        names = tuple(line.strip() for line
                      in open_file.read_text(encoding="utf-8").splitlines()
                      if line.strip())
        if names:
            return names
    return DEFAULT_OPEN_LOGS


def _run_logs(corpus: Corpus, prompter: Any, model: str, out: Path,
              truth_dir: Path, *, budget: int, redo: bool,
              max_logs: Optional[int],
              open_reports: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Grid and reader over every hand-truthed log, before and after.

    Restartable the same way the labels stage is: each log writes
    ``logs/<log id>.json`` as it finishes and a later call skips it.
    """
    from report_ingest.engine import CostMeter, PrompterEngine
    from report_ingest.log_scoring import METRICS, score_one_log

    if not truth_dir.is_dir():
        raise FileNotFoundError(
            f"no hand-truthed logs at {truth_dir}; the 'logs' stage scores "
            f"against them and cannot run without them")
    truths = []
    for path in sorted(truth_dir.glob("*.json")):
        try:
            truths.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping {path.name}: {type(exc).__name__}: {exc}")
    if max_logs:
        truths = truths[:int(max_logs)]
    openset = _open_set(truth_dir, open_reports)
    print(f"  logs: {len(truths)} hand-truthed log(s); open set "
          f"{', '.join(openset)}")

    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, truth in enumerate(truths, 1):
        log_id = str(truth.get("id") or f"log{n}")
        report = log_id.split("_")[0]
        run_file = out / "logs" / f"{log_id}.json"
        if run_file.is_file() and not redo:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
            # The open/blind split is decided at SCORING time, not frozen
            # into the run file. A log moved into the open set after it ran
            # must move in the scorecard too, or the blind figure quietly
            # keeps crediting a log somebody has since looked at.
            blob["set"] = "open" if report in openset else "blind"
            done[log_id] = blob
            print(f"  [{n}/{len(truths)}] {log_id}: already done, skipping")
            continue
        if report not in set(corpus.present_ids()):
            failures[log_id] = f"{report} is not in the reports folder"
            print(f"  [{n}/{len(truths)}] {log_id}: skipped -- "
                  f"{failures[log_id]}")
            continue
        meter = CostMeter()
        engine = PrompterEngine(prompter, model, meter=meter)
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(report, di="auto", warn=False)
            before, after = score_one_log(truth, doc, engine, budget=budget,
                                          report_id=report)
        except KeyboardInterrupt:
            print("  interrupted; what is finished is on disk and a later "
                  "call resumes")
            break
        except Exception as exc:                     # keep the run going
            failures[log_id] = f"{type(exc).__name__}: {exc}"
            print(f"  [{n}/{len(truths)}] {log_id}: FAILED -- "
                  f"{failures[log_id]}")
            traceback.print_exc()
            continue
        finally:
            if doc is not None:
                doc.close()
        blob = {
            "log_id": log_id,
            "report": report,
            "run_date": date.today().isoformat(),
            "set": "open" if report in openset else "blind",
            "model": model,
            "served_by": engine.served_by,
            "pages": [int(p) for p in truth.get("pages") or []],
            "before": before.to_dict(),
            "after": after.to_dict(),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
        }
        run_file.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        done[log_id] = blob
        gain = (after.total.found - before.total.found)
        print(f"  [{n}/{len(truths)}] {log_id}: "
              f"{before.total.found}/{before.total.total} -> "
              f"{after.total.found}/{after.total.total} "
              f"({gain:+d}), {after.model_calls} model call(s), "
              f"{blob['cost']['input_tokens']:,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")

    return _score_logs(done, failures, model, openset)


# ---------------------------------------------------------------------------
# the lab stage
# ---------------------------------------------------------------------------

def _run_lab(corpus: Corpus, prompter: Any, model: str, out: Path,
             truth_dir: Path, *, budget: int, redo: bool,
             max_sheets: Optional[int],
             open_reports: Optional[Sequence[str]]) -> Dict[str, Any]:
    """Tables and reader over every hand-truthed laboratory sheet.

    Restartable the same way the other stages are: each sheet writes
    ``lab/<sheet id>.json`` as it finishes and a later call skips it.
    """
    from report_ingest.engine import CostMeter, PrompterEngine
    from report_ingest.lab_scoring import pages_of, report_of, score_one_sheet

    if not truth_dir.is_dir():
        raise FileNotFoundError(
            f"no hand-truthed laboratory sheets at {truth_dir}; the 'lab' "
            f"stage scores against them and cannot run without them")
    truths = []
    for path in sorted(truth_dir.glob("*.json")):
        try:
            truths.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping {path.name}: {type(exc).__name__}: {exc}")
    if max_sheets:
        truths = truths[:int(max_sheets)]
    openset = _open_set(truth_dir, open_reports) if (
        open_reports or (truth_dir / "OPEN.txt").is_file()) \
        else DEFAULT_OPEN_LAB
    print(f"  lab: {len(truths)} hand-truthed sheet(s); open set "
          f"{', '.join(openset)}")

    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, truth in enumerate(truths, 1):
        sheet_id = str(truth.get("id") or f"sheet{n}")
        report = report_of(truth)
        run_file = out / "lab" / f"{sheet_id}.json"
        if run_file.is_file() and not redo:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
            # The open/blind split is decided at SCORING time, never frozen
            # into a run file: a sheet moved into the open set after it ran
            # has to move in the scorecard too, or the blind figure quietly
            # keeps crediting a page somebody has since looked at.
            blob["set"] = "open" if report in openset else "blind"
            done[sheet_id] = blob
            print(f"  [{n}/{len(truths)}] {sheet_id}: already done, skipping")
            continue
        if report not in set(corpus.present_ids()):
            failures[sheet_id] = f"{report} is not in the reports folder"
            print(f"  [{n}/{len(truths)}] {sheet_id}: skipped -- "
                  f"{failures[sheet_id]}")
            continue
        meter = CostMeter()
        engine = PrompterEngine(prompter, model, meter=meter)
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(report, di="auto", warn=False)
            before, after = score_one_sheet(truth, doc, engine,
                                            budget=budget, report_id=report)
        except KeyboardInterrupt:
            print("  interrupted; what is finished is on disk and a later "
                  "call resumes")
            break
        except Exception as exc:                     # keep the run going
            failures[sheet_id] = f"{type(exc).__name__}: {exc}"
            print(f"  [{n}/{len(truths)}] {sheet_id}: FAILED -- "
                  f"{failures[sheet_id]}")
            traceback.print_exc()
            continue
        finally:
            if doc is not None:
                doc.close()
        blob = {
            "sheet_id": sheet_id,
            "report": report,
            "kind": str(truth.get("kind") or ""),
            "run_date": date.today().isoformat(),
            "set": "open" if report in openset else "blind",
            "model": model,
            "served_by": engine.served_by,
            "pages": pages_of(truth),
            "before": before.to_dict(),
            "after": after.to_dict(),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
        }
        run_file.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        done[sheet_id] = blob
        gain = after.total.found - before.total.found
        print(f"  [{n}/{len(truths)}] {sheet_id}: "
              f"{before.total.found}/{before.total.total} -> "
              f"{after.total.found}/{after.total.total} "
              f"({gain:+d}), {after.model_calls} model call(s), "
              f"{blob['cost']['input_tokens']:,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")
    return _score_lab(done, failures, model, openset)


def _lab_totals(rows: Sequence[dict], stage: str
                ) -> Dict[str, Dict[str, int]]:
    """``metric -> {found, total}`` summed over a set of sheets."""
    from report_ingest.lab_scoring import METRICS

    out = {m: {"found": 0, "total": 0} for m in METRICS}
    for row in rows:
        for metric, score in (row[stage].get("scores") or {}).items():
            if metric not in out:
                out[metric] = {"found": 0, "total": 0}
            out[metric]["found"] += int(score.get("found") or 0)
            out[metric]["total"] += int(score.get("total") or 0)
    return out


def _score_lab(done: Dict[str, dict], failures: Dict[str, str], model: str,
               openset: Sequence[str]) -> Dict[str, Any]:
    rows = [done[k] for k in sorted(done)]
    sets: Dict[str, Any] = {}
    for name in ("open", "blind", "all"):
        group = [r for r in rows if name == "all" or r["set"] == name]
        if not group:
            continue
        sets[name] = {
            "sheets": [r["sheet_id"] for r in group],
            "n_sheets": len(group),
            "before": _lab_totals(group, "before"),
            "after": _lab_totals(group, "after"),
        }
    by_kind: Dict[str, Any] = {}
    for kind in sorted({r.get("kind") or "?" for r in rows}):
        group = [r for r in rows if (r.get("kind") or "?") == kind]
        by_kind[kind] = {
            "n_sheets": len(group),
            "before": _lab_totals(group, "before"),
            "after": _lab_totals(group, "after"),
        }
    cost = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "dollars": 0.0, "seconds": 0.0}
    for row in rows:
        for key in ("calls", "input_tokens", "output_tokens",
                    "cache_read_tokens"):
            cost[key] += row["cost"].get(key, 0)
        cost["dollars"] += row["cost"].get("dollars", 0.0)
        cost["seconds"] += row.get("seconds", 0.0)
    return {
        "date": date.today().isoformat(),
        "model": model,
        "served_by": sorted({r.get("served_by") for r in rows
                             if r.get("served_by")}),
        "open_set": list(openset),
        "n_sheets": len(rows),
        "failures": dict(failures),
        "sets": sets,
        "kinds": by_kind,
        "per_sheet": [{
            "sheet_id": r["sheet_id"], "set": r["set"],
            "kind": r.get("kind") or "?",
            "before": r["before"]["overall"], "after": r["after"]["overall"],
            "model_calls": r["after"].get("model_calls", 0),
            "tool_calls": r["after"].get("tool_calls", 0),
            "unresolved": r["after"].get("unresolved", 0),
            "changes": r["after"].get("changes", 0),
            "error": r["after"].get("error"),
            "input_tokens": r["cost"].get("input_tokens", 0),
            "output_tokens": r["cost"].get("output_tokens", 0),
            "dollars": r["cost"].get("dollars", 0.0),
            "seconds": r.get("seconds", 0.0),
        } for r in rows],
        "cost": cost,
    }


def _render_lab(lab: Dict[str, Any]) -> List[str]:
    """The lab stage, as tables that carry IDs, kinds, counts and rates only."""
    from report_ingest.lab_scoring import (
        DEPTH_TOL_M, EXACT_TOL, METRICS, MODEL_ONLY, PASSING_TOL,
    )

    out: List[str] = [
        "", "# WP3 on the cluster: the lab reader through Prompter", "",
        f"Run {lab['date']}. Model `{lab['model']}`, "
        f"{lab['n_sheets']} hand-truthed sheet(s).",
        "",
        f"**before** is what the page's own detected TABLES hold; **after** "
        f"is what the reader's records hold. A table has no idea what test "
        f"it is on or which boring it belongs to, so "
        f"{' and '.join('`' + m + '`' for m in MODEL_ONLY)} have "
        f"no before column at all, and the others ask of the tables "
        f"only whether the number is on the page. Tolerances: a depth links "
        f"within {DEPTH_TOL_M} m, compared in metres whatever the sheet "
        f"prints; an index value is exact to {EXACT_TOL}; a grading within "
        f"{PASSING_TOL} percent; a curve within the tolerance its own truth "
        f"file states.",
    ]
    if lab.get("served_by"):
        out.append(f"Served by: {', '.join(lab['served_by'])}.")
    if lab.get("failures"):
        out += ["", "**Sheets that failed and are NOT in any number below:**"]
        out += [f"- {sheet_id}: {why}"
                for sheet_id, why in lab["failures"].items()]
    out += ["", f"Open set (the reports whose lab pages were looked at): "
                f"{', '.join(lab['open_set'])}. Everything else is blind."]

    for name in ("open", "blind", "all"):
        row = lab["sets"].get(name)
        if not row:
            continue
        out += ["", f"## {name} -- {row['n_sheets']} sheet(s)", "", "```",
                f"{'metric':<10}{'before':>14}{'after':>14}"]
        for metric in METRICS:
            before, after = row["before"].get(metric), row["after"].get(metric)
            if not after or not (before["total"] or after["total"]):
                continue
            out.append(f"{metric:<10}{_rate(before):>14}{_rate(after):>14}")
        before_all = {"found": sum(v["found"] for v in row["before"].values()),
                      "total": sum(v["total"] for v in row["before"].values())}
        after_all = {"found": sum(v["found"] for v in row["after"].values()),
                     "total": sum(v["total"] for v in row["after"].values())}
        out += [f"{'OVERALL':<10}{_rate(before_all):>14}"
                f"{_rate(after_all):>14}", "```"]

    out += ["", "## Per kind", "", "```",
            f"{'kind':<20}{'sheets':>7}{'before':>14}{'after':>14}"]
    for kind, row in lab.get("kinds", {}).items():
        before_all = {"found": sum(v["found"] for v in row["before"].values()),
                      "total": sum(v["total"] for v in row["before"].values())}
        after_all = {"found": sum(v["found"] for v in row["after"].values()),
                     "total": sum(v["total"] for v in row["after"].values())}
        out.append(f"{kind:<20}{row['n_sheets']:>7}"
                   f"{_rate(before_all):>14}{_rate(after_all):>14}")
    out.append("```")

    out += ["", "## Per sheet", "", "```",
            f"{'sheet':<28}{'set':<7}{'before':>12}{'after':>12}{'calls':>7}"
            f"{'zoom':>6}{'unres':>7}{'look':>6}{'in':>10}{'out':>8}{'s':>7}"]
    for r in lab["per_sheet"]:
        if r.get("error"):
            out.append(f"{r['sheet_id']:<28}{r['set']:<7}ERROR "
                       f"{str(r['error'])[:50]}")
            continue
        out.append(
            f"{r['sheet_id']:<28}{r['set']:<7}"
            f"{_rate(r['before']):>12}{_rate(r['after']):>12}"
            f"{r['model_calls']:>7}{r['tool_calls']:>6}{r['unresolved']:>7}"
            f"{r['changes']:>6}{r['input_tokens']:>10,}"
            f"{r['output_tokens']:>8}{r['seconds']:>7.0f}")
    out.append("```")
    out += ["", "`zoom` is how many times the reader magnified a plot. "
                "`unres` is what it could not settle plus what Python "
                "refused -- an impossible depth, a percentage past 100, a "
                "liquid limit below the plastic limit, a grading running "
                "the wrong way. `look` is every value it took from the "
                "picture and every curve it digitised.", ""]
    cost = lab["cost"]
    n = max(1, lab["n_sheets"])
    out += ["## Cost", "", "```",
            f"{cost['calls']} model calls, {cost['input_tokens']:,} input "
            f"tokens (+{cost['cache_read_tokens']:,} the provider cached), "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s",
            f"per sheet: {cost['calls'] / n:.1f} calls, "
            f"{cost['input_tokens'] / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s",
            "```", "",
            "Funhouse publishes no per-token price for a capability tier, so "
            "this reports TOKENS. Read the spend from Funhouse's own budget "
            "endpoint for the same window."]
    return out


# ---------------------------------------------------------------------------
# the narrative stage (WP4)
# ---------------------------------------------------------------------------

#: The reports whose narrative answers were written with the reader's output
#: in view, when the truth folder carries no OPEN.txt. Everything else is
#: blind, and the blind figure is the one that means anything.
DEFAULT_OPEN_NARRATIVE: Tuple[str, ...] = ("R36", "R05")


def _run_narrative(corpus: Corpus, prompter: Any, model: str, out: Path,
                   truth_dir: Path, *, budget: int, redo: bool,
                   max_reports: Optional[int],
                   open_reports: Optional[Sequence[str]]) -> Dict[str, Any]:
    """The narrative reader over every report that has a hand answer.

    The stage reads the truth files that are PRESENT and skips every report
    without one: the hand answers arrive a few reports at a time, and a stage
    that failed on the ones not yet written would be unusable until the last
    one was.

    Restartable the same way the other stages are: each report writes
    ``narrative/<ID>.json`` as it finishes and a later call skips it.
    """
    from report_ingest.engine import CostMeter, PrompterEngine
    from report_ingest.narrative_scoring import score_one_report

    if not truth_dir.is_dir():
        raise FileNotFoundError(
            f"no hand answers at {truth_dir}; the 'narrative' stage scores "
            f"against them and cannot run without them")
    truths: List[Tuple[str, dict]] = []
    for path in sorted(truth_dir.glob("*.json")):
        rid = path.stem
        try:
            truths.append((rid, json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  skipping {path.name}: {type(exc).__name__}: {exc}")
    if max_reports:
        truths = truths[:int(max_reports)]
    openset = (tuple(open_reports) if open_reports
               else _open_set_or(truth_dir, DEFAULT_OPEN_NARRATIVE))
    present = set(corpus.present_ids())
    print(f"  narrative: {len(truths)} hand-answered report(s); open set "
          f"{', '.join(openset)}")

    done: Dict[str, dict] = {}
    failures: Dict[str, str] = {}
    for n, (rid, truth) in enumerate(truths, 1):
        run_file = out / "narrative" / f"{rid}.json"
        if run_file.is_file() and not redo:
            blob = json.loads(run_file.read_text(encoding="utf-8"))
            # The open/blind split is decided at SCORING time, never frozen
            # into a run file, for the same reason the lab stage does it.
            blob["set"] = "open" if rid in openset else "blind"
            done[rid] = blob
            print(f"  [{n}/{len(truths)}] {rid}: already done, skipping")
            continue
        if rid not in present:
            failures[rid] = f"{rid} is not in the reports folder"
            print(f"  [{n}/{len(truths)}] {rid}: skipped -- {failures[rid]}")
            continue
        meter = CostMeter()
        engine = PrompterEngine(prompter, model, meter=meter)
        started = time.time()
        doc = None
        try:
            doc = corpus.open_report(rid, di="auto", warn=False)
            score = score_one_report(truth, doc, engine, budget=budget,
                                     report=rid)
        except KeyboardInterrupt:
            print("  interrupted; what is finished is on disk and a later "
                  "call resumes")
            break
        except Exception as exc:                     # keep the run going
            failures[rid] = f"{type(exc).__name__}: {exc}"
            print(f"  [{n}/{len(truths)}] {rid}: FAILED -- {failures[rid]}")
            traceback.print_exc()
            continue
        finally:
            if doc is not None:
                doc.close()
        blob = {
            "id": rid,
            "run_date": date.today().isoformat(),
            "set": "open" if rid in openset else "blind",
            "model": model,
            "served_by": engine.served_by,
            "score": score.to_dict(),
            "cost": meter.to_dict(),
            "seconds": round(time.time() - started, 1),
        }
        run_file.write_text(json.dumps(blob, indent=2), encoding="utf-8")
        done[rid] = blob
        recall = score.recall
        print(f"  [{n}/{len(truths)}] {rid}: recall "
              f"{recall.found}/{recall.total}, precision "
              f"{score.precision.found}/{score.precision.total}, "
              f"{score.model_calls} model call(s), "
              f"{blob['cost']['input_tokens']:,} in / "
              f"{blob['cost']['output_tokens']:,} out, "
              f"{blob['seconds']:.0f} s")
    return _score_narrative(done, failures, model, openset)


def _open_set_or(truth_dir: Path,
                 fallback: Sequence[str]) -> Tuple[str, ...]:
    """The reports whose answers were written with the output in view."""
    open_file = truth_dir / "OPEN.txt"
    if open_file.is_file():
        names = tuple(line.strip() for line
                      in open_file.read_text(encoding="utf-8").splitlines()
                      if line.strip())
        if names:
            return names
    return tuple(fallback)


def _narrative_totals(rows: Sequence[dict]) -> Dict[str, Dict[str, int]]:
    """The three numbers, the kinds and the summaries, summed over reports."""
    from report_ingest.narrative_scoring import KINDS

    names = ["recall", "precision", "agreement"]
    out: Dict[str, Dict[str, int]] = {
        name: {"found": 0, "total": 0} for name in names}
    for kind in KINDS:
        out[f"kind.{kind}"] = {"found": 0, "total": 0}
    for name in ("list_items.precision", "list_items.recall",
                 "summaries.present", "summaries.within_limit"):
        out[name] = {"found": 0, "total": 0}
    for row in rows:
        score = row.get("score") or {}
        for name in names:
            part = score.get(name) or {}
            out[name]["found"] += int(part.get("found") or 0)
            out[name]["total"] += int(part.get("total") or 0)
        for kind, part in (score.get("by_kind") or {}).items():
            key = f"kind.{kind}"
            out.setdefault(key, {"found": 0, "total": 0})
            out[key]["found"] += int(part.get("found") or 0)
            out[key]["total"] += int(part.get("total") or 0)
        for group in ("list_items", "summaries"):
            for name, part in (score.get(group) or {}).items():
                key = f"{group}.{name}"
                out.setdefault(key, {"found": 0, "total": 0})
                out[key]["found"] += int(part.get("found") or 0)
                out[key]["total"] += int(part.get("total") or 0)
    return out


def _field_table(rows: Sequence[dict]) -> Dict[str, Dict[str, int]]:
    """``field -> {right, asked, missed, invented, wrong}`` over the reports.

    The per-field table is what says which QUESTIONS are hard, which is the
    thing a prompt change is actually aimed at. A field nobody hand-answered
    shows an asked of 0 and is not a failure.
    """
    out: Dict[str, Dict[str, int]] = {}
    for row in rows:
        for entry in ((row.get("score") or {}).get("fields") or []):
            name = entry.get("field")
            if not name:
                continue
            cell = out.setdefault(name, {"right": 0, "asked": 0, "missed": 0,
                                         "invented": 0, "wrong": 0})
            verdict = entry.get("verdict")
            if verdict in ("missed", "wrong", "right", "too_long"):
                cell["asked"] += 1
            if verdict == "right":
                cell["right"] += 1
            elif verdict in ("missed", "invented", "wrong"):
                cell[verdict] += 1
    return out


def _score_narrative(done: Dict[str, dict], failures: Dict[str, str],
                     model: str, openset: Sequence[str]) -> Dict[str, Any]:
    rows = [done[k] for k in sorted(done)]
    sets: Dict[str, Any] = {}
    for name in ("open", "blind", "all"):
        group = [r for r in rows if name == "all" or r["set"] == name]
        if not group:
            continue
        sets[name] = {"reports": [r["id"] for r in group],
                      "n_reports": len(group),
                      "totals": _narrative_totals(group)}
    cost = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "dollars": 0.0, "seconds": 0.0}
    for row in rows:
        for key in ("calls", "input_tokens", "output_tokens",
                    "cache_read_tokens"):
            cost[key] += row["cost"].get(key, 0)
        cost["dollars"] += row["cost"].get("dollars", 0.0)
        cost["seconds"] += row.get("seconds", 0.0)
    return {
        "date": date.today().isoformat(),
        "model": model,
        "served_by": sorted({r.get("served_by") for r in rows
                             if r.get("served_by")}),
        "open_set": list(openset),
        "n_reports": len(rows),
        "failures": dict(failures),
        "sets": sets,
        "fields": _field_table(rows),
        "per_report": [{
            "id": r["id"], "set": r["set"],
            "recall": (r["score"].get("recall") or {}),
            "precision": (r["score"].get("precision") or {}),
            "agreement": (r["score"].get("agreement") or {}),
            "model_calls": r["score"].get("model_calls", 0),
            "unresolved": r["score"].get("unresolved", 0),
            "error": r["score"].get("error"),
            "input_tokens": r["cost"].get("input_tokens", 0),
            "output_tokens": r["cost"].get("output_tokens", 0),
            "seconds": r.get("seconds", 0.0),
        } for r in rows],
        "cost": cost,
    }


def _render_narrative(narrative: Dict[str, Any]) -> List[str]:
    """The narrative stage, as tables carrying IDs, counts and rates only."""
    from report_ingest.narrative_scoring import (
        FUZZY_RATIO, KINDS, LIST_HIT_JACCARD,
    )

    out: List[str] = [
        "", "# WP4 on the cluster: the narrative reader through Prompter", "",
        f"Run {narrative['date']}. Model `{narrative['model']}`, "
        f"{narrative['n_reports']} hand-answered report(s).",
        "",
        "**recall** is of the questions the report DOES answer, how many came "
        "back right; **precision** is of the answers the reader gave, how "
        "many were right; **agreement** counts every field including the ones "
        "both sides left null, and flatters. Enumerations and counts are "
        f"exact; a string matches at a partial ratio of {FUZZY_RATIO} or on a "
        f"shared proper noun; a list at a Jaccard overlap of "
        f"{LIST_HIT_JACCARD}; the four prose summaries are scored for "
        "presence and word limit only, because whether a summary is a good "
        "summary is a person's call.",
    ]
    if narrative.get("served_by"):
        out.append(f"Served by: {', '.join(narrative['served_by'])}.")
    if narrative.get("failures"):
        out += ["", "**Reports that failed and are NOT in any number "
                    "below:**"]
        out += [f"- {rid}: {why}"
                for rid, why in narrative["failures"].items()]
    out += ["", f"Open set (answers written with the output in view): "
                f"{', '.join(narrative['open_set'])}. Everything else is "
                f"blind."]

    for name in ("open", "blind", "all"):
        row = narrative["sets"].get(name)
        if not row:
            continue
        totals = row["totals"]
        out += ["", f"## {name} -- {row['n_reports']} report(s)", "", "```",
                f"{'metric':<26}{'rate':>16}"]
        for metric in ("recall", "precision", "agreement"):
            out.append(f"{metric:<26}{_rate(totals[metric]):>16}")
        out.append("")
        for kind in KINDS:
            cell = totals.get(f"kind.{kind}")
            if cell and cell["total"]:
                out.append(f"{'  recall, ' + kind:<26}{_rate(cell):>16}")
        for label, key in (("list items, precision", "list_items.precision"),
                           ("list items, recall", "list_items.recall"),
                           ("summaries written", "summaries.present"),
                           ("summaries in limit",
                            "summaries.within_limit")):
            cell = totals.get(key)
            if cell and cell["total"]:
                out.append(f"{'  ' + label:<26}{_rate(cell):>16}")
        out.append("```")

    fields = narrative.get("fields") or {}
    asked = {name: cell for name, cell in fields.items() if cell["asked"]}
    if asked:
        out += ["", "## Per question", "", "```",
                f"{'field':<30}{'asked':>7}{'right':>7}{'missed':>8}"
                f"{'wrong':>7}{'invented':>10}"]
        for name in sorted(asked, key=lambda k: (-asked[k]["asked"], k)):
            cell = asked[name]
            out.append(f"{name:<30}{cell['asked']:>7}{cell['right']:>7}"
                       f"{cell['missed']:>8}{cell['wrong']:>7}"
                       f"{cell['invented']:>10}")
        out.append("```")

    out += ["", "## Per report", "", "```",
            f"{'report':<8}{'set':<7}{'recall':>14}{'precision':>14}"
            f"{'calls':>7}{'unres':>7}{'in':>10}{'out':>8}{'s':>7}"]
    for r in narrative["per_report"]:
        if r.get("error"):
            out.append(f"{r['id']:<8}{r['set']:<7}ERROR "
                       f"{str(r['error'])[:50]}")
            continue
        out.append(
            f"{r['id']:<8}{r['set']:<7}{_rate(r['recall']):>14}"
            f"{_rate(r['precision']):>14}{r['model_calls']:>7}"
            f"{r['unresolved']:>7}{r['input_tokens']:>10,}"
            f"{r['output_tokens']:>8,}{r['seconds']:>7.0f}")
    out.append("```")

    cost = narrative["cost"]
    n = max(1, narrative["n_reports"])
    out += ["", "## Cost", "", "```",
            f"{cost['calls']} model calls, {cost['input_tokens']:,} input "
            f"tokens (+{cost['cache_read_tokens']:,} the provider cached), "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s",
            f"per report: {cost['calls'] / n:.1f} calls, "
            f"{cost['input_tokens'] / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s",
            "```", "",
            "Funhouse publishes no per-token price for a capability tier, so "
            "this reports TOKENS. Read the spend from Funhouse's own budget "
            "endpoint for the same window."]
    return out


def _totals(rows: Sequence[dict], stage: str) -> Dict[str, Dict[str, int]]:
    """``metric -> {found, total}`` summed over a set of logs."""
    from report_ingest.log_scoring import METRICS

    out = {m: {"found": 0, "total": 0} for m in METRICS}
    for row in rows:
        for metric, score in (row[stage].get("scores") or {}).items():
            if metric not in out:
                out[metric] = {"found": 0, "total": 0}
            out[metric]["found"] += int(score.get("found") or 0)
            out[metric]["total"] += int(score.get("total") or 0)
    return out


def _score_logs(done: Dict[str, dict], failures: Dict[str, str], model: str,
                openset: Sequence[str]) -> Dict[str, Any]:
    rows = [done[k] for k in sorted(done)]
    sets: Dict[str, Any] = {}
    for name in ("open", "blind", "all"):
        group = [r for r in rows if name == "all" or r["set"] == name]
        if not group:
            continue
        sets[name] = {
            "logs": [r["log_id"] for r in group],
            "n_logs": len(group),
            "before": _totals(group, "before"),
            "after": _totals(group, "after"),
        }
    cost = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "dollars": 0.0, "seconds": 0.0}
    for row in rows:
        for key in ("calls", "input_tokens", "output_tokens",
                    "cache_read_tokens"):
            cost[key] += row["cost"].get(key, 0)
        cost["dollars"] += row["cost"].get("dollars", 0.0)
        cost["seconds"] += row.get("seconds", 0.0)
    return {
        "date": date.today().isoformat(),
        "model": model,
        "served_by": sorted({r.get("served_by") for r in rows
                             if r.get("served_by")}),
        "open_set": list(openset),
        "n_logs": len(rows),
        "failures": dict(failures),
        "sets": sets,
        "per_log": [{
            "log_id": r["log_id"], "set": r["set"],
            "before": r["before"]["overall"], "after": r["after"]["overall"],
            "model_calls": r["after"].get("model_calls", 0),
            "unresolved": r["after"].get("unresolved", 0),
            "changes": r["after"].get("changes", 0),
            "error": r["after"].get("error"),
            "input_tokens": r["cost"].get("input_tokens", 0),
            "output_tokens": r["cost"].get("output_tokens", 0),
            "dollars": r["cost"].get("dollars", 0.0),
            "seconds": r.get("seconds", 0.0),
        } for r in rows],
        "cost": cost,
    }


def _rate(row: Dict[str, int]) -> str:
    if not row["total"]:
        return "     -"
    return f"{row['found'] / row['total']:5.0%} {row['found']}/{row['total']}"


def _render_logs(logs: Dict[str, Any]) -> List[str]:
    """The logs stage, as tables that carry IDs, counts and rates only."""
    from report_ingest.log_scoring import (
        LAYER_TOL_M, METRICS, SAMPLE_TOL_M, WATER_TOL_M,
    )

    out: List[str] = [
        "", "# WP2b on the cluster: the log reader through Prompter", "",
        f"Run {logs['date']}. Model `{logs['model']}`, "
        f"{logs['n_logs']} hand-truthed log(s).",
        "",
        f"**before** is what `log_grid` alone recovered; **after** is what "
        f"the reader's record holds. The grid runs once and is handed to the "
        f"reader, so the difference between the two columns is the model and "
        f"nothing else. Tolerances: samples, index values and water "
        f"{SAMPLE_TOL_M} m, layer tops {LAYER_TOL_M} m "
        f"(water {WATER_TOL_M} m); depths compared in metres whatever the "
        f"log prints. N values exact.",
    ]
    if logs.get("served_by"):
        out.append(f"Served by: {', '.join(logs['served_by'])}.")
    if logs.get("failures"):
        out += ["", "**Logs that failed and are NOT in any number below:**"]
        out += [f"- {log_id}: {why}"
                for log_id, why in logs["failures"].items()]
    out += ["", f"Open set (the logs the rules were tuned on): "
                f"{', '.join(logs['open_set'])}. Everything else is blind."]

    for name in ("open", "blind", "all"):
        row = logs["sets"].get(name)
        if not row:
            continue
        out += ["", f"## {name} -- {row['n_logs']} log(s)", "", "```",
                f"{'metric':<16}{'before':>14}{'after':>14}"]
        for metric in METRICS:
            before, after = row["before"].get(metric), row["after"].get(metric)
            if not before or not (before["total"] or after["total"]):
                continue
            out.append(f"{metric:<16}{_rate(before):>14}{_rate(after):>14}")
        before_all = {"found": sum(v["found"] for v in row["before"].values()),
                      "total": sum(v["total"] for v in row["before"].values())}
        after_all = {"found": sum(v["found"] for v in row["after"].values()),
                     "total": sum(v["total"] for v in row["after"].values())}
        out += [f"{'OVERALL':<16}{_rate(before_all):>14}"
                f"{_rate(after_all):>14}", "```"]

    out += ["", "## Per log", "", "```",
            f"{'log':<14}{'set':<7}{'before':>12}{'after':>12}{'calls':>7}"
            f"{'unres':>7}{'look':>6}{'in':>10}{'out':>8}{'s':>7}"]
    for r in logs["per_log"]:
        if r.get("error"):
            out.append(f"{r['log_id']:<14}{r['set']:<7}ERROR "
                       f"{str(r['error'])[:60]}")
            continue
        out.append(
            f"{r['log_id']:<14}{r['set']:<7}"
            f"{_rate(r['before']):>12}{_rate(r['after']):>12}"
            f"{r['model_calls']:>7}{r['unresolved']:>7}{r['changes']:>6}"
            f"{r['input_tokens']:>10,}{r['output_tokens']:>8,}"
            f"{r['seconds']:>7.0f}")
    out.append("```")
    out += ["", "`unres` is what the reader could not settle plus what "
                "Python refused (a depth off the page). `look` is how many "
                "values it took from the picture rather than the rows.", ""]
    cost = logs["cost"]
    n = max(1, logs["n_logs"])
    out += ["## Cost", "", "```",
            f"{cost['calls']} model calls, {cost['input_tokens']:,} input "
            f"tokens (+{cost['cache_read_tokens']:,} the provider cached), "
            f"{cost['output_tokens']:,} output, {cost['seconds']:.0f} s",
            f"per log: {cost['calls'] / n:.1f} calls, "
            f"{cost['input_tokens'] / n:,.0f} in, "
            f"{cost['output_tokens'] / n:,.0f} out, "
            f"{cost['seconds'] / n:.0f} s",
            "```", "",
            "Funhouse publishes no per-token price for a capability tier, so "
            "this reports TOKENS. Read the spend from Funhouse's own budget "
            "endpoint for the same window."]
    return out


def _header(results: Dict[str, Any]) -> List[str]:
    """The heading a run writes when the labels stage did not run."""
    return ["# Report ingest on the cluster", "",
            f"Run {results['date']}. Stage(s): "
            f"{', '.join(results.get('stages') or ())}."]


def _render(results: Dict[str, Any]) -> List[str]:
    """RESULTS.md: IDs, labels, counts and rates. Nothing that names anyone."""
    out: List[str] = [
        "# WP1b on the cluster: triage and label review through Prompter",
        "",
        f"Run {results['date']}. Review model `{results['review_model']}`, "
        f"triage `{results['triage_model']}`.",
    ]
    if results.get("served_by"):
        out.append(f"Served by: {', '.join(results['served_by'])}. A Funhouse "
                   f"tier is an alias and the deployment behind it changes, "
                   f"so this is what actually answered.")
    if results.get("absent"):
        out += ["", f"Not in the reports folder, so not run: "
                    f"{', '.join(results['absent'])}."]
    if results.get("failures"):
        out += ["", "**Reports that failed and are NOT in any number below:**"]
        out += [f"- {rid}: {why}" for rid, why in results["failures"].items()]

    for name, row in results["sets"].items():
        if not row["n_reports"]:
            continue
        before, after = row["_scores"]
        out += ["", f"## {name} -- {row['n_reports']} report(s), "
                    f"{after.n} scored pages", ""]
        if row["disputed_dropped"]:
            out.append(f"{row['disputed_dropped']} page(s) dropped from both "
                       f"scores as confirmed disputed hand labels.")
            out.append("")
        out += ["```",
                f"{'':<24}{'before':>10}{'after':>10}",
                f"{'strict accuracy':<24}{before.accuracy:>10.3f}"
                f"{after.accuracy:>10.3f}",
                f"{'accepting alternates':<24}"
                f"{before.lenient_accuracy:>10.3f}"
                f"{after.lenient_accuracy:>10.3f}",
                "",
                f"key content (the gate is {GATE:.2f} on both rates after "
                f"review)"]
        out += label_table(before, after, KEY_CONTENT)
        out.append("below the gate after review: "
                   + (", ".join(row["gate_failures"]) or "none"))
        out += ["", "every label"]
        out += label_table(before, after)
        out += ["", "what the review's changes did, against the hand labels"]
        for verdict in ("fixed", "broke", "still_wrong", "disputed",
                        "unscored"):
            out.append(f"  {verdict:<14}{row['verdicts'].get(verdict, 0):>5}")
        out += ["",
                f"{'report':<8}{'pages':>7}{'scored':>8}{'before':>9}"
                f"{'after':>8}{'chg':>5}{'tools':>8}{'calls':>7}"
                f"{'in':>10}{'out':>9}{'s':>7}"]
        for r in row["per_report"]:
            before_txt = ("   --  " if r["before"] is None
                          else f"{r['before']:>9.3f}")
            after_txt = ("   --  " if r["after"] is None
                         else f"{r['after']:>8.3f}")
            out.append(f"{r['id']:<8}{r['pages']:>7}{r['scored']:>8}"
                       f"{before_txt}{after_txt}{r['changes']:>5}"
                       f"{r['tool_calls']:>5}/{r['budget']:<2}"
                       f"{r['calls']:>7}{r['input_tokens']:>10,}"
                       f"{r['output_tokens']:>9,}{r['seconds']:>7.0f}")
        out.append("```")

        clean = row.get("never_seen")
        if clean:
            cb, ca = clean["_scores"]
            out += ["",
                    f"### the same set minus {', '.join(clean['excluded'])}, "
                    f"which the cost checkpoint used and so are no longer "
                    f"blind",
                    "",
                    f"{len(clean['reports'])} reports nobody has opened, "
                    f"{ca.n} pages. **This is the honest blind figure.**",
                    "", "```",
                    f"{'':<24}{'before':>10}{'after':>10}",
                    f"{'strict accuracy':<24}{cb.accuracy:>10.3f}"
                    f"{ca.accuracy:>10.3f}",
                    f"{'accepting alternates':<24}"
                    f"{cb.lenient_accuracy:>10.3f}"
                    f"{ca.lenient_accuracy:>10.3f}", ""]
            out += label_table(cb, ca, KEY_CONTENT)
            out.append("below the gate after review: "
                       + (", ".join(clean["gate_failures"]) or "none"))
            out.append("```")

    out += ["", "## What triage said", "",
            "Enumerated fields only. The rationale and the anomalies stay in "
            "`triage/<ID>.json` on the cluster.", "",
            "| report | document_type | workflow | bound | toc | scan |",
            "|---|---|---|---|---|---|"]
    for r in results["triage"]:
        out.append(f"| {r['id']} | {r['document_type']} | {r['workflow']} | "
                   f"{r['bound_together']} | {r['toc_agreement']} | "
                   f"{r['scan_fraction']:.2f} |")

    totals = results["totals"]
    out += ["", "## Cost", "", "```",
            f"{totals['calls']} model calls, "
            f"{totals['input_tokens']:,} input tokens "
            f"(+{totals['cache_read_tokens']:,} the provider cached), "
            f"{totals['output_tokens']:,} output, "
            f"{totals['seconds']:.0f} s",
            "```", "",
            "Funhouse publishes no per-token price for a capability tier, so "
            "this reports TOKENS. Read the spend from Funhouse's own budget "
            "endpoint for the same window."]
    return out
