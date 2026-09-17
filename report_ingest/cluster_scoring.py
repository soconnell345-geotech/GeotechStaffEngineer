"""Score the two model passes on the cluster, against the real model.

The app runs in Funhouse against OpenAI models through Prompter. A score
measured on any other model measures a model that will never do the work, so
the numbers that count come from here: one function the owner calls in a
notebook with the live ``fh_prompter`` object already in hand.

    from report_ingest.cluster_scoring import score_on_cluster

    score_on_cluster(
        corpus_dir="/Volumes/main/geotech/report_corpus",
        labels_xlsx="/Volumes/main/geotech/report_corpus/trial_pages.xlsx",
        di_dir="/Volumes/main/geotech/report_di",
        out_dir="/tmp/report_ingest_wp1b",
        prompter=fh_prompter,
        model="funhouse-gpt-medium",
    )

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

__all__ = ["score_on_cluster", "SET_NAMES"]

SET_NAMES: Tuple[str, ...] = ("insample", "oos_open", "oos_blind")

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


def score_on_cluster(corpus_dir: Any, labels_xlsx: Any = None,
                     di_dir: Any = None, out_dir: Any = "/tmp/report_ingest",
                     prompter: Any = None, model: str = "funhouse-gpt-medium",
                     sets: Sequence[str] = SET_NAMES, *,
                     triage_model: Optional[str] = None,
                     oos_labels: Any = None,
                     max_total_dollars: Optional[float] = None,
                     max_reports: Optional[int] = None,
                     redo: bool = False) -> Dict[str, Any]:
    """Run the rules, triage and the label review over the corpus, and score.

    Parameters
    ----------
    corpus_dir, labels_xlsx, di_dir
        Where the PDFs (``R01.pdf`` ...), the hand-label spreadsheet and the
        Azure Document Intelligence results are. Only ``corpus_dir`` is
        required; without the spreadsheet the run measures but cannot score
        the in-sample set.
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
    """
    if prompter is None:
        raise ValueError(
            "pass the live fh_prompter; this module reads no credential of "
            "its own")
    out = Path(out_dir)
    _check_out_dir(out)
    for sub in ("runs", "triage"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    corpus = Corpus(corpus_dir, di_dir=di_dir, labels_xlsx=labels_xlsx,
                    cache_dir=out)
    if not corpus.available:
        raise FileNotFoundError(
            f"no corpus at {corpus_dir}: expected R01.pdf ... R38.pdf there")
    triage_model = triage_model or model

    oos: Dict[str, Dict[int, dict]] = {}
    if oos_labels:
        blob = json.loads(Path(oos_labels).read_text(encoding="utf-8"))
        oos = {rid: {int(p): v for p, v in pages.items()}
               for rid, pages in blob.items()}
    mapped = corpus.mapped_ids() if corpus.labels_available else []

    wanted: List[str] = []
    for name in sets:
        for rid in _set_ids(name, corpus):
            if rid not in wanted:
                wanted.append(rid)
    if max_reports:
        wanted = wanted[:int(max_reports)]

    print(f"WP1b on the cluster: {len(wanted)} report(s); review {model}, "
          f"triage {triage_model}; out_dir {out}")
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
                     failures)
    lines = _render(results)
    (out / "results.json").write_text(json.dumps(_plain(results), indent=2),
                                      encoding="utf-8")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out / 'RESULTS.md'} -- that is the file to bring back")
    print(f"per-report runs in {out / 'runs'}, profiles in {out / 'triage'}")
    return results


def _score(done: Dict[str, dict], corpus: Corpus,
           oos: Dict[str, Dict[int, dict]], mapped: Sequence[str],
           sets: Sequence[str], model: str, triage_model: str,
           failures: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "date": date.today().isoformat(),
        "engine": "prompter",
        "review_model": model,
        "triage_model": triage_model,
        "served_by": sorted({s for b in done.values()
                             for s in (b.get("served_by") or [])}),
        "failures": dict(failures),
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
