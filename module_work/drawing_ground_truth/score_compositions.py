"""Observational scoring: composition proposals vs native DXF truth.

Runs find_leaders / find_dimensions / find_bubble_callouts on the plotted
Mecklenburg PDFs and compares against the native-entity truth JSON
(model-space inches from the DXF). REPORT, NOT GATE: real-drafting recall
is the known-unknown being measured — do not tune the heuristics to these
numbers; document them.

Coordinate mapping: sheets are drawn at paper scale in inches and plotted
at 72 pt/in; the residual offset between model space and the PDF page is
estimated by aligning the truth text bbox with the IR text... SHX sheets
have no IR text, so alignment uses the overall linework bbox instead
(median-shift fit). Matches are counted within a tolerance of 18 pt
(0.25 in) after transform — generous on purpose, since the fit itself is
approximate.

Usage:  python module_work/drawing_ground_truth/score_compositions.py

RESULTS (2026-09-04 run, post Phase-2 hardening — the honest read):

- Leader/dimension tip recall vs native truth: 0/25 and 0/16. NOT a
  scoring artifact: probing a truth tip on sheet 21.01 shows Mecklenburg
  plots render arrowheads as clusters of ~0.06-pt micro-segments (solid
  fill dots), not triangles — invisible to the closed/open-triangle
  candidate model by representation, and micro-dot clustering can't be
  naively added because line-style dots flood the sheets (3,417 tiny
  segments, 205 clusters on 21.01). A fill-cluster arrowhead leg is
  Phase-3 work alongside B7 (these SHX sheets need a raster pass for
  text anyway).
- False-positive floor after hardening: 5 of 10 sheets report ZERO
  spurious dimensions; dense sheets (3001) still over-propose
  (36 D vs 10 truth) — some may be visually-real leaders drafted as
  plain lines (native-entity truth undercounts those); unverified.
- Bubbles: 10.31A proposes exactly 40 vs 40 native circles (count
  match; positions not yet cross-checked).
- Perf on real sheets after the endpoint-grid + id-map work: worst
  sheet 6 s (was >600 s before indexing).

These numbers are the drawing memo's real-drafting baseline — do not
tune to them; grow the representation model (Phase 3) and re-run.
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


def _fit_offset(truth_pts, ir_bbox):
    """Median offset mapping SCALE*model -> IR page points (no rotation)."""
    if not truth_pts or ir_bbox is None:
        return None
    xs = sorted(p[0] * SCALE for p in truth_pts)
    ys = sorted(p[1] * SCALE for p in truth_pts)
    tb = (xs[0], ys[0], xs[-1], ys[-1])
    # Align centers of the truth extent and the IR linework extent.
    return (0.5 * (ir_bbox[0] + ir_bbox[2]) - 0.5 * (tb[0] + tb[2]),
            0.5 * (ir_bbox[1] + ir_bbox[3]) - 0.5 * (tb[1] + tb[3]))


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
