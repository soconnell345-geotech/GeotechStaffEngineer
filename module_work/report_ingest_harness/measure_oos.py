"""Out-of-sample check for the page labels (WP1 review step, lead's hand labels).

The WP1 scorecard is scored on the 14 hand-labelled reports, all overseas-facility
reports from a few firms. This script scores the same rules on 120 pages the lead
labelled by eye from contact sheets (five random pages from each of the 24
reports WITHOUT spreadsheet labels: six US commercial firms, a state DOT, two
non-English reports, 1990s scans). Labels live in the private folder
``raw/checks/oos/labels.json`` (primary label + acceptable alternates).

Run:  python -m module_work.report_ingest_harness.measure_oos
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from . import corpus


BLIND = ("R17", "R19", "R22", "R25", "R26", "R27", "R31", "R32", "R33", "R34",
         "R35", "R36", "R37", "R38")   # never shown to the rules builder
OPEN = ("R01", "R02", "R03", "R04", "R05", "R06", "R07", "R08", "R10", "R14")


def main() -> int:
    from planlens.document.roles import page_roles  # WP1

    which = sys.argv[1] if len(sys.argv) > 1 else "all"   # all | blind | open
    keep = {"all": None, "blind": set(BLIND), "open": set(OPEN)}[which]
    oos = Path(corpus.RAW_DIR) / "checks" / "oos"
    labels = json.load(open(oos / "labels.json", encoding="utf-8"))
    if keep is not None:
        labels = {k: v for k, v in labels.items() if k in keep}
    print(f"scoring the {which} set: {len(labels)} reports")
    rows = []
    for rid, pages in sorted(labels.items()):
        doc = corpus.open_report(rid, di="auto")
        try:
            roles = {r.page: r for r in page_roles(doc)}
        finally:
            doc.close()
        for p, want in pages.items():
            r = roles.get(int(p))
            pred = r.role if r else "?"
            ok = pred == want["label"]
            ok_alt = ok or pred in want["alternates"]
            rows.append((rid, int(p), want["label"], pred, ok, ok_alt,
                         (getattr(r, "evidence", {}) or {}).get("rule", "") if r else ""))
    n = len(rows)
    strict = sum(1 for x in rows if x[4]); lenient = sum(1 for x in rows if x[5])
    print(f"out-of-sample pages: {n}; strict accuracy {strict/n:.2f}; "
          f"accepting alternates {lenient/n:.2f}")
    per = collections.defaultdict(lambda: [0, 0, 0])   # label -> [support, tp, predicted]
    for _, _, want, pred, ok, _, _ in rows:
        per[want][0] += 1; per[pred][2] += 1
        if ok: per[want][1] += 1
    print(f"{'label':<16}{'support':>8}{'recall':>8}{'precision':>10}")
    for lab in sorted(per):
        s, tp, pc = per[lab]
        rec = tp / s if s else float('nan'); prec = tp / pc if pc else float('nan')
        print(f"{lab:<16}{s:>8}{rec:>8.2f}{prec:>10.2f}")
    print("\nmisses (strict):")
    for rid, p, want, pred, ok, ok_alt, rule in rows:
        if not ok:
            flag = "" if not ok_alt else "  (alternate accepted)"
            print(f"  {rid} p{p:<4} hand={want:<16} pred={pred:<16} {rule}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
