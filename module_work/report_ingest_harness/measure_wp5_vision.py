"""WP5 measurement: labelling a page by LOOKING at it, against the hand.

Run from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_wp5_vision
    ... --set checkpoint            # which reports
    ... --mode sheet                # six pages a call instead of one
    ... --mode document             # a window of pages, whole report in view
    ... --detail low                # 85 tokens an image instead of a page
    ... --outline-context           # let it see what the document says it is
    ... --reuse                     # re-score the saved runs, no model at all
    ... --append --note "what changed this round"

WHAT IT MEASURES. planlens' rules label a page from what that page prints
about itself and :mod:`report_ingest.label_review` corrects them with the
whole report in view. Both read TEXT. This measures the third way: the page
as a PICTURE to a cheap model, with the eighteen-label vocabulary and
nothing else, through
:func:`report_ingest.vision_labels.classify_pages_by_vision`.

THE THREE COLUMNS ARE SCORED BY ONE SCORER. ``rules``, ``+review`` and
``vision`` all go through :class:`report_ingest.scoring.Scores` against the
same hand labels, because a vision number measured by a scorer of its own
would be a number nobody could set beside the rules. ``+review`` appears
only for a report whose WP1b run is saved beside this one.

THE NUMBERS THAT COUNT COME FROM THE CLUSTER. The app runs in Funhouse
against OpenAI models through Prompter, and the experiment is aimed at the
cheapest tier there. This script drives the DEVELOPMENT engine so the
prompt can be iterated here at a keystroke, and its figures go into the
ledger marked as a checkpoint. The real run is
``score_on_cluster(..., stages=("vision_labels",),
vision_model="funhouse-gpt-low")``.

``--reuse`` re-scores the saved runs and calls no model at all, so tuning
the scorecard is free and the whole of this file can be exercised offline.

PRIVACY. Reports are IDs and pages are numbers. The model's REASON for a
page can quote a title block, so it is written to the gitignored run file
and never to the scorecard or the ledger.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import corpus
from module_work.report_ingest_harness import measure_wp1b as wp1b
from report_ingest.scoring import (   # the cluster scores with these too
    GATE, KEY_CONTENT, Scores, columns_label_table, disputed_drop,
    gate_failures,
)
from report_ingest.vision_labels import (
    DEFAULT_DPI, DETAIL_LEVELS, DOCUMENT_OVERLAP, DOCUMENT_WINDOW,
    MAX_IMAGES_PER_CALL, MODES, SHEET_PAGES, classify_pages_by_vision,
)

LEDGER = corpus.RAW_DIR.parent / "MEASUREMENTS.md"
SECTION = "## WP5 -- labelling a page by looking at it"
#: Where the full, private detail of a run goes. Gitignored, like all of
#: ``raw/``: the reasons name what the model saw on a page.
RUNS_DIR = corpus.CHECKS_DIR / "wp5_vision"
#: Where the WP1b runs are, for the ``+review`` column.
REVIEW_RUNS_DIR = wp1b.RUNS_DIR

#: The development model. Never quoted as the system's accuracy: the tier
#: this experiment is really about is ``funhouse-gpt-low``.
DEV_MODEL = "claude-haiku-4-5"


# -- ground truth and the sets ----------------------------------------------

def set_ids(name: str) -> Tuple[str, ...]:
    """The reports in one set. The same sets the label scorecard uses."""
    return wp1b.set_ids(name)


def oos_labels() -> Dict[str, Dict[int, dict]]:
    if not wp1b.OOS_LABELS.is_file():
        return {}
    blob = json.loads(wp1b.OOS_LABELS.read_text(encoding="utf-8"))
    return {rid: {int(p): v for p, v in pages.items()}
            for rid, pages in blob.items()}


def review_labels_for(rid: str) -> Dict[int, str]:
    """The reviewed labels a saved WP1b run left behind, or nothing.

    Nothing means the report prints two columns rather than three, which is
    honest; filling the column from the rules would print the rules twice.
    """
    path = REVIEW_RUNS_DIR / f"{rid}.json"
    if not path.is_file():
        return {}
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    final = ((blob.get("review") or {}).get("final_labels")) or {}
    return {int(k): v for k, v in final.items()}


# -- running one report ------------------------------------------------------

def run_one(rid: str, engine: Any, *, mode: str = "page",
            dpi: float = DEFAULT_DPI, outline_context: bool = False,
            sheet_pages: int = SHEET_PAGES, budget: Optional[int] = None,
            di: str = "auto", reuse: bool = False,
            detail: Optional[str] = None,
            window: int = DOCUMENT_WINDOW,
            overlap: int = DOCUMENT_OVERLAP,
            images_per_call: int = MAX_IMAGES_PER_CALL,
            fallback: bool = True) -> dict:
    """One report through the vision pass, saved, and restartable.

    ``reuse`` re-reads the saved run instead of calling a model, which is
    what makes tuning the scorecard free.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_file = RUNS_DIR / f"{rid}.json"
    if reuse and run_file.is_file():
        return json.loads(run_file.read_text(encoding="utf-8"))
    if reuse:
        return {"id": rid, "error": "no saved run to reuse"}

    from planlens.document.roles import document_outline, page_roles

    doc = None
    try:
        doc = corpus.open_report(rid, di=di, warn=False)
        roles = page_roles(doc)
        outline = document_outline(doc) if outline_context else None
        seen = classify_pages_by_vision(
            doc, engine, mode=mode, dpi=dpi, budget=budget,
            outline_context=outline_context, sheet_pages=sheet_pages,
            outline=outline, detail=detail, window=window, overlap=overlap,
            images_per_call=images_per_call, fallback=fallback)
        blob = {
            "id": rid,
            "run_date": datetime.date.today().isoformat(),
            "n_pages": doc.n_pages,
            "model": getattr(engine, "name", ""),
            "mode": mode, "dpi": float(dpi), "detail": detail,
            "outline_context": bool(outline_context),
            "rules_labels": {str(r.page): r.role for r in roles},
            "vision": seen.to_dict(),
            "cost": dict(seen.cost),
            "error": None,
        }
    except Exception as exc:                      # measure the rest of them
        return {"id": rid, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if doc is not None:
            doc.close()
    run_file.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return blob


# -- scoring -----------------------------------------------------------------

def score(runs: Sequence[dict], oos: Dict[str, Dict[int, dict]]
          ) -> Dict[str, Any]:
    """The three columns, one scorer, one set of hand labels."""
    rules, review, vision = Scores(), Scores(), Scores()
    per_report: List[dict] = []
    dropped = 0
    with_review = 0
    for blob in runs:
        rid = str(blob.get("id") or "")
        if blob.get("error"):
            per_report.append({"id": rid, "error": blob["error"]})
            continue
        rule_labels = {int(k): v
                       for k, v in (blob.get("rules_labels") or {}).items()}
        seen = blob.get("vision") or {}
        vision_labels = {int(k): v
                         for k, v in (seen.get("labels") or {}).items()}
        reviewed = review_labels_for(rid)
        with_review += int(bool(reviewed))
        hand, alternates, source = wp1b.truth_for(rid, oos)
        hits = {"rules": 0, "review": 0, "vision": 0}
        scored = 0
        for page, want in sorted(hand.items()):
            alts = alternates.get(page, ())
            was = rule_labels.get(page, "other")
            now = reviewed.get(page, "other")
            saw = vision_labels.get(page, "other")
            if reviewed and disputed_drop(rid, page, now):
                dropped += 1
                continue
            rules.add(want, was, alts)
            vision.add(want, saw, alts)
            if reviewed:
                review.add(want, now, alts)
            scored += 1
            hits["rules"] += int(was == want)
            hits["review"] += int(now == want)
            hits["vision"] += int(saw == want)
        cost = blob.get("cost") or {}
        per_report.append({
            "id": rid, "pages": blob.get("n_pages", 0), "scored": scored,
            "truth": source,
            "rules": (hits["rules"] / scored) if scored else None,
            "review": ((hits["review"] / scored)
                       if scored and reviewed else None),
            "vision": (hits["vision"] / scored) if scored else None,
            "labelled": len(seen.get("labels") or {}),
            "unresolved": len(seen.get("unresolved") or []),
            "qa": len(seen.get("qa") or []),
            "calls": cost.get("calls", 0),
            "input_tokens": cost.get("input_tokens", 0),
            "output_tokens": cost.get("output_tokens", 0),
            "dollars": cost.get("dollars", 0.0),
            "error": None,
        })
    return {"rules": rules, "review": review, "vision": vision,
            "per_report": per_report, "disputed_dropped": dropped,
            "n_with_review": with_review}


def columns(scored: Dict[str, Any]) -> List[Tuple[str, Scores]]:
    """The columns this run actually has, in the order they are read."""
    out = [("rules", scored["rules"])]
    if scored["review"].n:
        out.append(("+review", scored["review"]))
    out.append(("vision", scored["vision"]))
    return out


def report(scored: Dict[str, Any], *, blind: bool = False) -> str:
    """The scorecard. IDs, labels, counts and rates; never a reason."""
    cols = columns(scored)
    out: List[str] = [
        f"{'':<24}" + "".join(f"{n:>10}" for n, _s in cols),
        f"{'strict accuracy':<24}"
        + "".join(f"{s.accuracy:>10.3f}" for _n, s in cols),
        f"{'accepting alternates':<24}"
        + "".join(f"{s.lenient_accuracy:>10.3f}" for _n, s in cols),
        "",
        f"key content (the gate is {GATE:.2f} on both rates)",
    ]
    out += columns_label_table(cols, KEY_CONTENT)
    out.append("vision below the gate: "
               + (", ".join(gate_failures(scored["vision"])) or "none"))
    out += ["", "every label"]
    out += columns_label_table(cols)
    if scored["disputed_dropped"]:
        out += ["", f"{scored['disputed_dropped']} page(s) dropped from every "
                    f"column as confirmed disputed hand labels."]
    if blind:
        # A blind figure read report by report stops being blind the moment
        # somebody goes looking for which report dragged it down.
        out += ["", "blind set: summary only, no per-report line."]
        return "\n".join(out)

    out += ["",
            f"{'report':<8}{'pages':>7}{'scored':>8}{'rules':>8}{'review':>8}"
            f"{'vision':>8}{'unres':>7}{'qa':>5}{'calls':>7}{'in':>10}"
            f"{'out':>9}{'$':>8}"]
    for row in scored["per_report"]:
        if row.get("error"):
            out.append(f"{row['id']:<8}ERROR {str(row['error'])[:60]}")
            continue

        def cell(value: Optional[float]) -> str:
            return "   --   " if value is None else f"{value:>8.3f}"

        out.append(f"{row['id']:<8}{row['pages']:>7}{row['scored']:>8}"
                   f"{cell(row['rules'])}{cell(row['review'])}"
                   f"{cell(row['vision'])}{row['unresolved']:>7}"
                   f"{row['qa']:>5}{row['calls']:>7}"
                   f"{row['input_tokens']:>10,}{row['output_tokens']:>9,}"
                   f"{row['dollars']:>8.3f}")
    good = [r for r in scored["per_report"] if not r.get("error")]
    calls = sum(r["calls"] for r in good)
    spent = sum(r["dollars"] for r in good)
    out += ["", f"{calls} model call(s) over {len(good)} report(s), "
                f"${spent:.2f} at list price."]
    return "\n".join(out)


# -- the command line --------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="checkpoint", dest="set_name",
                    choices=wp1b.SET_NAMES)
    ap.add_argument("--only", default="", help="comma-separated report IDs")
    ap.add_argument("--mode", default="page", choices=MODES)
    ap.add_argument("--dpi", type=float, default=DEFAULT_DPI)
    ap.add_argument("--sheet-pages", type=int, default=SHEET_PAGES,
                    help="pages per contact sheet in sheet mode")
    ap.add_argument("--detail", default=None, choices=DETAIL_LEVELS,
                    help="image_url.detail on every picture; 'low' is about "
                         "85 tokens an image instead of a page's four tiles")
    ap.add_argument("--window", type=int, default=DOCUMENT_WINDOW,
                    help="full-size pages in one document-mode call")
    ap.add_argument("--overlap", type=int, default=DOCUMENT_OVERLAP,
                    help="pages the next document-mode window sees again")
    ap.add_argument("--images-per-call", type=int,
                    default=MAX_IMAGES_PER_CALL,
                    help="the endpoint's ceiling on images in one request; "
                         "50 was measured on the cluster on 2026-09-18")
    ap.add_argument("--outline-context", action="store_true",
                    help="give it what the document prints about itself")
    ap.add_argument("--no-fallback", action="store_true",
                    help="do NOT give a page left unresolved by a sheet or "
                         "a window one page-mode call of its own")
    ap.add_argument("--budget", type=int, default=None,
                    help="ceiling on model calls per report")
    ap.add_argument("--di", default="auto", choices=("auto", "all", "none"))
    ap.add_argument("--model", default=DEV_MODEL,
                    help="the DEVELOPMENT model; its numbers are a "
                         "checkpoint, never a result")
    ap.add_argument("--reuse", action="store_true",
                    help="re-score the saved runs and call no model at all")
    ap.add_argument("--append", action="store_true",
                    help="append the scorecard to MEASUREMENTS.md")
    ap.add_argument("--note", default="", help="what changed this round")
    args = ap.parse_args(argv)

    only = tuple(x.strip() for x in args.only.split(",") if x.strip())
    ids = [rid for rid in set_ids(args.set_name)
           if not only or rid in only]
    if not ids:
        print("no reports matched")
        return 1

    engine: Any = None
    if not args.reuse:
        from report_ingest.engine import ClaudeEngine
        engine = ClaudeEngine(args.model)

    runs: List[dict] = []
    for n, rid in enumerate(ids, 1):
        print(f"[{n}/{len(ids)}] {rid} ...", flush=True)
        runs.append(run_one(rid, engine, mode=args.mode, dpi=args.dpi,
                            outline_context=args.outline_context,
                            sheet_pages=args.sheet_pages, budget=args.budget,
                            di=args.di, reuse=args.reuse, detail=args.detail,
                            window=args.window, overlap=args.overlap,
                            images_per_call=args.images_per_call,
                            fallback=not args.no_fallback))

    scored = score(runs, oos_labels())
    text = report(scored, blind=args.set_name == "oos_blind")
    print(text)
    failed = [r["id"] for r in runs if r.get("error")]
    if failed and len(failed) == len(runs):
        print(f"\nEVERY report failed ({len(failed)}). Nothing was measured.",
              file=sys.stderr)
        return 2
    if args.append:
        stamp = datetime.date.today().isoformat()
        block = (f"\n### WP5 vision scorecard, {stamp} -- DEVELOPMENT engine "
                 f"{args.model} (a checkpoint, never a result)\n\n"
                 + (f"{args.note}\n\n" if args.note else "")
                 + f"Set {args.set_name}, mode {args.mode}, "
                   f"{args.dpi:.0f} dpi, outline context "
                   f"{'on' if args.outline_context else 'off'}"
                 + (f", detail {args.detail}" if args.detail else "")
                 + (f", window {args.window} pages with {args.overlap} "
                    f"overlapping, at most {args.images_per_call} images a "
                    f"call" if args.mode == "document" else "")
                 + ". `rules` is "
                   f"planlens' per-page rules, `+review` the rules with the "
                   f"label review's changes applied, `vision` a model "
                   f"looking at the page and nothing else; all three scored "
                   f"against the same hand labels by the same scorer, and an "
                   f"unresolved page counts as `other`.\n\n"
                   f"```\n{text}\n```\n")
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(block)
        print(f"\nappended to {LEDGER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
