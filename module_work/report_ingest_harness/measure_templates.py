"""Measurement: the log-TEMPLATE recogniser, and what it is worth to the grid.

Run from the repo root::

    .venv/Scripts/python -m module_work.report_ingest_harness.measure_templates
    ... --non-logs 30         # how many non-log pages to test precision on
    ... --seed 20260920       # the draw is seeded, so the set is reproducible
    ... --append              # write the scorecard into MEASUREMENTS.md

NO MODEL, NO KEY, NO NETWORK. Everything here is ``log_grid`` and rapidfuzz
over pages already on this machine, so it is free to re-run after every change
to a fingerprint.

WHAT IT MEASURES, in three parts.

**Recognition.** Over the hand-truthed logs: which family the recogniser
claims each page for, against the family that actually printed it (taken from
the fingerprint file's own families and the corpus, and written down here as
``EXPECTED``). Per family: how many of that family's logs it found (recall)
and how many of its claims were right (precision).

**Precision off the logs.** A seeded random draw of pages that are NOT logs,
from the same reports. Every match on one of them is a false positive, and
this is the number that says whether a fingerprint is claiming letterhead
rather than a form.

**What the column map is worth.** For each recognised log the grid's own
floor (``seed_from_grid``) is scored against the hand truth TWICE -- once with
the template's ``column_map`` and once without -- through the same
``log_scoring.score_record`` the cluster uses. The difference is the
fingerprint's whole contribution to the record, with no model involved.

PRIVACY. Reports are IDs, logs are ``<ID>_p<page>`` and families are printed
as ``family A``, ``family B`` in ledger order. The FIRM NAMES live only in the
private fingerprint file; nothing here prints one, and MEASUREMENTS.md, which
IS committed, gets only IDs, letters, counts and rates.

That includes the EVIDENCE. A fingerprint's own phrases are the firm's name,
its gINT report name and its field labels, so :func:`report` prints which
GROUP carried each hit and how well, and never the phrase itself: ``footer
(100); title (100)`` rather than the words. To see the words, call
:func:`report_ingest.log_templates.recognise` directly on a machine that has
the private file; do not put them in a committed scorecard.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from module_work.report_ingest_harness import corpus as C
from report_ingest.log_floor import seed_from_grid
from report_ingest.log_scoring import METRICS, Score, score_record
from report_ingest.log_templates import (
    MATCH_THRESHOLD, TemplateMatch, column_names, load_templates, recognise,
)

TRUTH_DIR = C.RAW_DIR / "truth" / "logs"
TEMPLATES = C.RAW_DIR / "truth" / "templates.json"
LEDGER = C.RAW_DIR.parent / "MEASUREMENTS.md"

#: Which family actually printed each hand-truthed log -- the lead's own
#: reading of the corpus, 2026-09-20 -- lives in the PRIVATE fingerprint file
#: under ``"expected"``, keyed by report, valued with the family name as that
#: file spells it. It is not here because a family name is a firm and this
#: file is committed. A report absent from that map is printed on a form no
#: fingerprint describes, and every match on it is a false positive.
def expected_families(path: Any) -> Dict[str, str]:
    """``report -> family``, out of the private fingerprint file."""
    here = Path(path)
    if not here.is_file():
        return {}
    blob = json.loads(here.read_text(encoding="utf-8"))
    if not isinstance(blob, dict):
        return {}
    return {str(k): str(v) for k, v in (blob.get("expected") or {}).items()}

#: The page labels that are logs. A non-log page is anything else, and the
#: precision draw comes from those.
LOG_LABELS = frozenset(("boring_log", "test_pit_log", "cpt_log", "dcp_log"))


# ---------------------------------------------------------------------------
# families as letters
# ---------------------------------------------------------------------------

class Anonymiser:
    """``family -> "family A"``, stable in first-seen order.

    The committed ledger must carry the measurement and not the client list,
    and a letter carries the measurement perfectly well: what matters is that
    two fingerprints of one family are one family and that a family's numbers
    can be read.
    """

    def __init__(self) -> None:
        self._seen: Dict[str, str] = {}

    def __call__(self, family: Optional[str]) -> str:
        if not family:
            return "-"
        if family not in self._seen:
            letter = chr(ord("A") + len(self._seen))
            self._seen[family] = f"family {letter}"
        return self._seen[family]

    @property
    def families(self) -> List[str]:
        return list(self._seen.values())


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def load_truth(only: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    if not TRUTH_DIR.is_dir():
        raise FileNotFoundError(
            f"no hand-truthed logs ({TRUTH_DIR} is gitignored and exists only "
            f"where the owner put it)")
    out: List[Dict[str, Any]] = []
    for path in sorted(TRUTH_DIR.glob("*.json")):
        truth = json.loads(path.read_text(encoding="utf-8"))
        if only and truth["id"].split("_")[0] not in only:
            continue
        out.append(truth)
    return out


def _floor_scores(truth: Dict[str, Any], grid: Any, pages: Sequence[int],
                  match: Optional[TemplateMatch]
                  ) -> Tuple[Any, Optional[Any], int]:
    """``(without the map, with it, columns the map named)``."""
    plain = score_record(truth, [seed_from_grid(grid, pages)])
    if match is None or not match.column_map:
        return plain, None, 0
    named = column_names(grid, match)
    mapped = score_record(truth,
                          [seed_from_grid(grid, pages, template=match)])
    return plain, mapped, len(named)


def run_logs(truths: Sequence[Dict[str, Any]], templates: Sequence[Any],
             di: str, expected: Dict[str, str]) -> List[Dict[str, Any]]:
    from planlens.document.loggrid import log_grid

    rows: List[Dict[str, Any]] = []
    for truth in truths:
        log_id = truth["id"]
        rid = log_id.split("_")[0]
        pages = [int(p) for p in truth["pages"]]
        doc = None
        try:
            doc = C.open_report(rid, di=di, warn=False)
            grid = log_grid(doc, pages)
            match = recognise(doc, pages[0], grid=grid, templates=templates)
            plain, mapped, named = _floor_scores(truth, grid, pages, match)
        except Exception as exc:                 # measure the rest of them
            rows.append({"log": log_id, "report": rid,
                         "error": f"{type(exc).__name__}: {exc}"})
            continue
        finally:
            if doc is not None:
                doc.close()
        rows.append({
            "log": log_id, "report": rid,
            "want": expected.get(rid, ""),
            "got": match.family if match else "",
            "name": match.name if match else "",
            "confidence": round(match.confidence, 3) if match else 0.0,
            "margin": round(match.margin, 3) if match else 0.0,
            "evidence": list(match.evidence) if match else [],
            "columns_named": named,
            "plain": plain, "mapped": mapped,
        })
    return rows


def non_log_pages(reports: Sequence[str], n: int,
                  seed: int) -> List[Tuple[str, int, str]]:
    """A seeded draw of ``(report, page, hand label)`` off the hand labels.

    The draw is from the HAND labels, so a page counted as "not a log" is one
    a person said is not a log, and a false positive here is a real one.
    """
    pool: List[Tuple[str, int, str]] = []
    mapped = set(C.CORPUS.mapped_ids())
    for rid in reports:
        if rid not in mapped:
            continue
        for label in C.CORPUS.labels_for(rid):
            if label.label not in LOG_LABELS:
                pool.append((rid, int(label.page0), label.label))
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def run_non_logs(draw: Sequence[Tuple[str, int, str]],
                 templates: Sequence[Any], di: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    by_report: Dict[str, List[Tuple[int, str]]] = {}
    for rid, page, label in draw:
        by_report.setdefault(rid, []).append((page, label))
    for rid in sorted(by_report):
        doc = None
        try:
            doc = C.open_report(rid, di=di, warn=False)
            for page, label in sorted(by_report[rid]):
                match = recognise(doc, page, templates=templates)
                rows.append({"report": rid, "page": page, "hand": label,
                             "got": match.family if match else "",
                             "name": match.name if match else "",
                             "confidence": (round(match.confidence, 3)
                                            if match else 0.0)})
        except Exception as exc:                 # measure the rest of them
            rows.append({"report": rid, "page": -1, "hand": "",
                         "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if doc is not None:
                doc.close()
    return rows


# ---------------------------------------------------------------------------
# the scorecard
# ---------------------------------------------------------------------------

def _rate(found: int, total: int) -> str:
    if not total:
        return "     -"
    return f"{found / total:5.0%} {found}/{total}"


def _metric(score: Any, name: str) -> Optional[Score]:
    return None if score is None else score.scores.get(name)


_RE_GROUP = re.compile(r"^(\w+)\s")


def _redact(evidence: Sequence[str]) -> str:
    """The evidence as a per-group tally, with the form's own words gone.

    A fingerprint's phrases ARE the firm: its name, its gINT report name, its
    field labels. What carries the diagnostic weight is which GROUP matched
    and how many of its phrases did, so that is what is printed --
    ``footer 1, title 4, columns 6`` -- and never the words. To see the words,
    call ``report_ingest.log_templates.recognise`` on a machine that has the
    private file.
    """
    counts: Dict[str, int] = {}
    for part in evidence:
        match = _RE_GROUP.match(str(part))
        if match:
            counts[match.group(1)] = counts.get(match.group(1), 0) + 1
    order = ("footer", "title", "columns")
    return ", ".join(f"{name} {counts[name]}" for name in order
                     if name in counts)


def report(log_rows: Sequence[Dict[str, Any]],
           non_log_rows: Sequence[Dict[str, Any]],
           anon: Anonymiser, threshold: float) -> str:
    out: List[str] = [
        f"threshold {threshold:.2f}; family names are letters here and the "
        f"evidence names its GROUP and not the form's own words -- both live "
        f"only in the private fingerprint file",
        "",
        f"{'log':<12}{'want':<11}{'got':<11}{'conf':>6}{'margin':>8}"
        f"{'cols':>6}  evidence",
    ]
    for row in log_rows:
        if row.get("error"):
            out.append(f"{row['log']:<12}ERROR {row['error'][:60]}")
            continue
        out.append(
            f"{row['log']:<12}{anon(row['want']):<11}{anon(row['got']):<11}"
            f"{row['confidence']:>6.2f}{row['margin']:>8.2f}"
            f"{row['columns_named']:>6}  {_redact(row['evidence'])[:64]}")

    # -- recognition, per family ------------------------------------------
    good = [r for r in log_rows if not r.get("error")]
    families = sorted({r["want"] for r in good if r["want"]}
                      | {r["got"] for r in good if r["got"]})
    out += ["", "Recognition on the hand-truthed logs",
            f"{'family':<12}{'logs':>6}{'recall':>14}{'precision':>16}"]
    for family in families:
        want = [r for r in good if r["want"] == family]
        got = [r for r in good if r["got"] == family]
        hit = sum(1 for r in want if r["got"] == family)
        out.append(f"{anon(family):<12}{len(want):>6}"
                   f"{_rate(hit, len(want)):>14}"
                   f"{_rate(sum(1 for r in got if r['want'] == family), len(got)):>16}")
    unclaimed = [r for r in good if not r["want"]]
    out.append(f"{'no template':<12}{len(unclaimed):>6}"
               f"{'-':>14}"
               f"{_rate(sum(1 for r in unclaimed if not r['got']), len(unclaimed)):>16}")

    # -- precision off the logs -------------------------------------------
    clean = [r for r in non_log_rows if not r.get("error")]
    claimed = [r for r in clean if r["got"]]
    out += ["", f"Pages the hand says are NOT logs: {len(clean)} drawn, "
                f"{len(claimed)} claimed by a template"]
    for row in claimed[:12]:
        out.append(f"  {row['report']} p{row['page']:<5}{row['hand']:<16}"
                   f"{anon(row['got'])} {row['confidence']:.2f}")
    if len(claimed) > 12:
        out.append(f"  ... and {len(claimed) - 12} more")

    # -- what the column map is worth --------------------------------------
    with_map = [r for r in good if r.get("mapped") is not None]
    out += ["", f"The grid's floor, scored against the hand truth, on the "
                f"{len(with_map)} log(s) a template claimed",
            f"{'metric':<14}{'no column map':>16}{'with it':>16}"]
    for metric in METRICS + ("overall",):
        plain_found = plain_total = mapped_found = mapped_total = 0
        for row in with_map:
            if metric == "overall":
                one, two = row["plain"].total, row["mapped"].total
            else:
                one = _metric(row["plain"], metric)
                two = _metric(row["mapped"], metric)
            if one is not None:
                plain_found += one.found
                plain_total += one.total
            if two is not None:
                mapped_found += two.found
                mapped_total += two.total
        if not plain_total and not mapped_total:
            continue
        out.append(f"{metric:<14}{_rate(plain_found, plain_total):>16}"
                   f"{_rate(mapped_found, mapped_total):>16}")
    out += ["", "Per log, overall floor score",
            f"{'log':<12}{'family':<11}{'no map':>14}{'with map':>14}"]
    for row in with_map:
        out.append(f"{row['log']:<12}{anon(row['got']):<11}"
                   f"{_rate(row['plain'].total.found, row['plain'].total.total):>14}"
                   f"{_rate(row['mapped'].total.found, row['mapped'].total.total):>14}")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated report IDs")
    ap.add_argument("--di", default="auto", choices=("auto", "all", "none"))
    ap.add_argument("--templates", default=str(TEMPLATES),
                    help="the PRIVATE fingerprint file")
    ap.add_argument("--non-logs", type=int, default=30,
                    help="how many non-log pages to test precision on")
    ap.add_argument("--seed", type=int, default=20260920,
                    help="the seed for the non-log draw")
    ap.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    ap.add_argument("--append", action="store_true",
                    help="append the scorecard to MEASUREMENTS.md")
    ap.add_argument("--note", default="", help="what changed this round")
    args = ap.parse_args(argv)

    templates = load_templates(args.templates)
    if not templates:
        print(f"no fingerprints at {args.templates}; nothing to measure",
              file=sys.stderr)
        return 1
    only = tuple(x.strip() for x in args.only.split(",") if x.strip()) or None
    truths = load_truth(only)
    if not truths:
        print("no hand-truthed logs matched")
        return 1

    print(f"{len(templates)} fingerprint(s), {len(truths)} truthed log(s)",
          flush=True)
    expected = expected_families(args.templates)
    if not expected:
        print("the fingerprint file carries no 'expected' map; every log "
              "will read as belonging to no family", file=sys.stderr)
    log_rows = run_logs(truths, templates, args.di, expected)
    reports = sorted({row["report"] for row in log_rows})
    draw = non_log_pages(reports, args.non_logs, args.seed)
    print(f"{len(draw)} non-log page(s) drawn with seed {args.seed}",
          flush=True)
    non_log_rows = run_non_logs(draw, templates, args.di)

    anon = Anonymiser()
    for row in log_rows:                     # letters in report order
        anon(row.get("want") or "")
    text = report(log_rows, non_log_rows, anon, args.threshold)
    print(text)

    if args.append:
        stamp = datetime.date.today().isoformat()
        block = (f"\n### Log-template recogniser, {stamp} -- no model, no "
                 f"network\n\n"
                 + (f"{args.note}\n\n" if args.note else "")
                 + f"```\n{text}\n```\n")
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(block)
        print(f"\nappended to {LEDGER}")
    return 0


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main())
