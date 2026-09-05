"""Observational scoring: composition proposals vs native DXF truth.

Runs find_leaders / find_dimensions / find_bubble_callouts on the plotted
Mecklenburg PDFs and compares against the native-entity truth JSON
(model-space inches from the DXF). REPORT, NOT GATE: real-drafting recall
is the known-unknown being measured — do not tune the heuristics to these
numbers; document them.

Coordinate mapping: planlens.ir.align.fit_plot_transform fits the full
plot transform (rotation 0/90/180/270 + uniform scale + offset) by anchor
voting on native leader/dimension/text anchor points — the Mecklenburg
sheets plot at 72 pt/in ROTATED 270 degrees, which a translation-only fit
cannot see (the Phase-3 discovery; rms 0.02-0.03 pt, 100% anchors matched
on the annotation-rich sheets). Matches are counted within 18 pt (0.25 in)
after transform. OCR coverage has its own committed check:
ocr_coverage_check.py (independent-verifier methodology; 'partial' =
containment matches carry much of the headline coverage number, exact-only
is far lower — both are printed).

Usage:  python module_work/drawing_ground_truth/score_compositions.py

RESULTS LEDGER (do not tune to these; grow the representation and re-run):

Phase-2 baseline (2026-09-04, translation-only transform): 0/25, 0/16.

Phase 3 (2026-09-04, same day — after the rotation+scale+offset fit,
fill-cluster arrowheads, fold-blind triangle alignment, and no-text
confidence renormalization):

- TRANSFORM TRUTH: the Mecklenburg sheets plot ROTATED 270 deg at
  exactly 72 pt/model-inch (fit rms 0.02-0.03 pt, 100% of anchor
  points matched) — Phase 2's 0/25 was PARTLY a transform artifact.
  The old translation-only fit could never land on these plots.
- Leader tip recall 11/25 (21.01: 3/13, 3001: 5/7, 10.31A: 3/5).
  Residual misses, diagnosed per tip on 21.01: four tips have NO
  plotted fragments at all within arrow range (arrowheads suppressed
  or off-window), and the rest are sparse micro-dot tips (1-3 dots)
  sitting inside stipple texture whose local density spike is below
  the 2x background gate — indistinguishable from texture by pure
  local geometry at this scale.
- Dimension end recall 1/16 — DIAGNOSED, distinct root cause: native
  dims plot their shaft as TWO collinear halves around a centered
  text gap; each half carries one arrowhead, so the both-ends-on-one-
  shaft model never fires on them (the high-confidence dim proposals
  found are the sheet's OTHER, manually-drafted dims). Fix = a
  split-shaft pairing leg in find_dimensions (v2, documented in the
  drawing memo; not attempted this phase).
- Bubbles: 40/40 count match on 10.31A (unchanged).
- OCR leg (planlens.ocr, RapidOCR, auto-rotation): on-page truth-text
  coverage 92% / 92% / 88% exact+partial on 21.01 / 3001 / 10.31A,
  with median coordinate error 5.8 / 7.0 / 1.4 pt after the page-
  /Rotate derotation fix. Full end-to-end: search_drawing_set with
  ocr_text=true finds "MECKLENBURG" on the no-text-layer 21.01 sheet
  (0 without OCR, flagged inconclusive; ~2.5 min first pass at 300
  dpi, instant cached follow-ups).
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

MECK = os.path.join(HERE, "mecklenburg")
SCALE = 72.0          # points per model inch (paper-scale sheets)
MATCH_TOL = 18.0      # pt (0.25 in) after transform
MIN_CONF = 0.3        # observational: look at the permissive band too


def _match(truth_xy, prop_xy, tol=MATCH_TOL):
    """Greedy one-to-one matching; returns number matched."""
    import math
    unused = list(prop_xy)
    n = 0
    for t in truth_xy:
        best, bd = None, None
        for i, p in enumerate(unused):
            d = math.hypot(p[0] - t[0], p[1] - t[1])
            if d <= tol and (bd is None or d < bd):
                best, bd = i, d
        if best is not None:
            unused.pop(best)
            n += 1
    return n


def _leader_tips(m):
    """Model-space tip points: LEADER vertices[0] + each MULTILEADER
    leader-line's first vertex (the arrow end in both)."""
    tips = [L["vertices"][0] for L in m.get("leaders", [])
            if L.get("vertices")]
    for ML in m.get("multileaders", []):
        for ln in ML.get("leader_lines", []):
            if ln:
                tips.append(ln[0])
                break  # one tip per multileader (avoid double-count)
    return tips


def main():
    from planlens.ir import from_pdf_vector, queries as q
    from planlens.ir.align import fit_plot_transform

    sheets = sorted(f[:-11] for f in os.listdir(MECK)
                    if f.endswith(".truth.json"))
    print(f"{'sheet':8s} {'ents':>6s} | truth L/D | fit(rot,scale,rms) | "
          f"props L/D | matched L D | secs")
    totals = dict(tl=0, td=0, ml=0, md=0)
    for s in sheets:
        truth = json.load(open(os.path.join(MECK, s + ".truth.json")))
        m = truth["spaces"]["model"]
        n_lead = (len(m.get("leaders", [])) + len(m.get("multileaders", [])))
        t_dim = m.get("dimensions", [])

        pdf = os.path.join(MECK, s + ".pdf")
        t0 = time.time()
        ir = from_pdf_vector(pdf)
        leads = q.find_leaders(ir, min_confidence=MIN_CONF,
                               exclude_dimensions=True)
        dims = q.find_dimensions(ir, min_confidence=MIN_CONF)
        dt = time.time() - t0

        # Transform: rotation/scale/offset fit on annotation anchor chains
        # (leader + multileader vertices + dimension defpoints) — the
        # translation-only fit failed on rotated/real-scale plots.
        chains = [L["vertices"] for L in m.get("leaders", [])
                  if L.get("vertices")]
        chains += [ln for ML in m.get("multileaders", [])
                   for ln in ML.get("leader_lines", []) if ln]
        anchors = [v for c in chains for v in c]
        anchors += [d["defpoint"] for d in t_dim if d.get("defpoint")]
        fit = (fit_plot_transform(anchors, ir, anchor_chains=chains)
               if len(anchors) >= 3 else None)

        ml = md = 0
        fit_desc = "no-fit"
        if fit is not None:
            fit_desc = (f"r{fit['rotation_deg']:<3d}s{fit['scale']:6.2f} "
                        f"rms{fit['rms']:.2f}")
            tips = [fit["apply"](tuple(v)) for v in _leader_tips(m)]
            ml = _match(tips, [p["tip_xy"] for p in leads])
            dps = [fit["apply"](tuple(d["defpoint"])) for d in t_dim
                   if d.get("defpoint")]
            # A dimension defpoint sits at one END of the dimension line.
            ends = ([tuple(p["end_a_xy"]) for p in dims]
                    + [tuple(p["end_b_xy"]) for p in dims])
            md = _match(dps, ends)

        totals["tl"] += n_lead
        totals["td"] += len(t_dim)
        totals["ml"] += ml
        totals["md"] += md
        print(f"{s:8s} {len(ir.entities):>6d} | "
              f"{n_lead:2d}/{len(t_dim):2d}     | {fit_desc:18s} | "
              f"{len(leads):3d}/{len(dims):3d}   | "
              f"{ml:2d}/{n_lead:2d} {md:2d}/{len(t_dim):2d} | {dt:5.1f}")
    print(f"\nTOTAL leader tip recall  {totals['ml']}/{totals['tl']}"
          f"  dimension end recall {totals['md']}/{totals['td']}")
    print("(observational; greedy match within "
          f"{MATCH_TOL:g} pt after rotation+scale+offset anchor fit)")


if __name__ == "__main__":
    main()
