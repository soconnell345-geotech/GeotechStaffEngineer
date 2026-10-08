"""Private spot check of planlens' visual scales against a real report.

    python spot_check.py LOGS_PDF TRUTH [--extra-truth PATH]
                         [--grading-pdf PDF] [--json OUT] [--seed N]

Nothing private is in this file: the report and every true number come in
through the arguments (the lead's ``scales_truth.json`` and an optional
extra truth file kept beside it, both in a git-ignored folder). What it
prints is counts and error sizes only.

What it checks (no model is called; where a model would read a printed
number, the value comes from the truth file, as the app's label-reading
call would supply it):

1. Scanned log sheets: ``log_grid`` first asks for the depth labels' values;
   they are supplied; every stratum line found is compared with the true
   boundaries (error, inside the reading's own +/-, missed, extra). The
   same boundaries are then measured from rough boxes with ``measure``
   (the agent's path), counting right, listed-not-chosen and wrong snaps.
2. Vector log sheets: the depth labels against the ticks drawn beside them
   (offset of each label centre from its tick) and how far tying the ruler
   to the ticks moved each layer top.
3. Trial-pit sheets drawn to no scale: refused or not.
4. Scanned grading sheets: the axis labels' values supplied; each plotted
   point measured from a rough box and compared with the printed table.

Exit code 0 always (a report, not a gate); ``--json`` saves the numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from typing import Any, Dict, List, Optional

from planlens.document import Document
from planlens.document import loggrid as LG
from planlens.document.measuring import measure
from planlens.document.scalefinder import find_scales
from planlens.document.scales import unrotate_point

#: Rough-box error of a box read off a zoomed view (the design's E5 pixel
#: regime), points.
BOX_SD_PT = 3.0
#: Draws per true boundary / plotted point.
DRAWS = 5
#: Truth's own accuracy (the lead's method), metres.
TRUTH_PM_M = 0.05


def _stats(errs: List[float]) -> Dict[str, Any]:
    a = [abs(e) for e in errs]
    if not a:
        return {"n": 0}
    return {"n": len(a), "median_abs": round(statistics.median(a), 4),
            "max_abs": round(max(a), 4),
            "mean": round(statistics.fmean(errs), 4)}


def _pct(k: int, n: int) -> str:
    return f"{k}/{n}" + (f" ({100.0 * k / n:.0f} %)" if n else "")


# -- truth -------------------------------------------------------------------

def load_truth(truth: str, extra: Optional[str]) -> Dict[str, Any]:
    lead = json.load(open(truth, encoding="utf-8"))
    ext = json.load(open(extra, encoding="utf-8")) if extra else {}
    logs = dict(ext.get("logs", {}))
    sheets = {s["page"]: dict(s) for s in logs.get("scanned_sheets", [])}
    dpi = float(logs.get("frame_dpi", 130))
    for s in lead.get("sheets", []):
        # the lead's measurement wins where both exist
        base = sheets.get(s["page"], {})
        sheets[s["page"]] = {**base, "page": s["page"],
                             "depth_from_m": s["depth_from_m"],
                             "frame_px": s["frame_px"],
                             "boundaries_m": s["boundaries_m"],
                             "boundaries_from": "lead"}
    for s in sheets.values():
        s["frame_pt"] = [p * 72.0 / dpi for p in s["frame_px"]]
    logs["scanned_sheets"] = [sheets[k] for k in sorted(sheets)]
    return {"logs": logs, "grading": ext.get("grading")}


# -- label values, as a reader of the printed numbers would give them ----------

def log_label_values(boxes, sheet, step: float) -> List[float]:
    """Each label's printed depth: the label stands just above the depth it
    marks, inside the sheet's depth frame (0 at its top, 10 steps at its
    bottom), so the printed number is the whole step nearest its place."""
    top, bottom = sheet["frame_pt"]
    n = float(sheet.get("steps_per_sheet", 10))
    out = []
    for b in boxes:
        cy = (b[1] + b[3]) / 2.0
        d = (cy - top) / (bottom - top) * n * step
        out.append(round(sheet["depth_from_m"] + step * round(d / step), 6))
    return out


def axis_label_values(boxes, scale, frame_extent, ax) -> List[float]:
    """An axis label's printed value, as a reader of the label gives it.

    The axis prints its values evenly (a value a step, a decade a step on a
    log axis), starting at the chart's leading edge (left for x, top for y).
    Each label is counted in steps from that edge, the step being the
    labels' own median spacing, so a far edge drawn wider than the grid
    cannot shift the values.
    """
    x0, y0, _x1, _y1 = frame_extent
    if scale.axis == "x":
        cs = [(b[0] + b[2]) / 2.0 for b in boxes]
        edge = x0
        order = sorted(ax["printed"])                     # left to right
        if ax["left"] > ax["right"]:
            order = order[::-1]
    else:
        cs = [(b[1] + b[3]) / 2.0 for b in boxes]
        edge = y0
        order = sorted(ax["printed"], reverse=ax["top"] > ax["bottom"])
    srt = sorted(cs)
    gaps = sorted(b - a for a, b in zip(srt, srt[1:]))
    # a missing label doubles one gap; the median is a single step
    step = gaps[len(gaps) // 2] if gaps else 1.0
    out = []
    for c in cs:
        k = int(round((c - edge) / step))
        out.append(order[max(0, min(len(order) - 1, k))])
    return out


# -- 1. scanned log sheets -----------------------------------------------------

def scanned_logs(doc, logs, rng) -> Dict[str, Any]:
    step = float(logs.get("label_step_m", 1.0))
    per_sheet = []
    errs, inside, inside_truth, missed, extra = [], 0, 0, 0, 0
    pms = []
    snaps = {"right": 0, "listed": 0, "wrong": 0, "unsnapped_inside": 0,
             "unsnapped_outside": 0, "total": 0}
    scale_ratio = []
    for sh in logs["scanned_sheets"]:
        p = sh["page"]
        g = LG.log_grid(doc, [p])
        nv = g.needs_values.get(p)
        row: Dict[str, Any] = {"page": p, "truth_lines":
                               len(sh["boundaries_m"])}
        if not nv:
            row["result"] = ("no ruler found" if not g.rulers
                             else "read without values")
            per_sheet.append(row)
            missed += len(sh["boundaries_m"])
            continue
        boxes = nv["labels"]
        row["labels_found"] = len(boxes)
        vals = log_label_values(boxes, sh, step)
        g = LG.log_grid(doc, [p], values={p: vals})
        if not g.rulers:
            row["result"] = "values refused"
            per_sheet.append(row)
            missed += len(sh["boundaries_m"])
            continue
        r = g.rulers[p] if isinstance(g.rulers, dict) else g.rulers[0]
        row["anchor_rule"] = r.evidence.get("anchor_rule_kind")
        row["ruler_pm_m"] = round(r.plus_minus or 0.0, 4)
        # the scale from the labels alone against the frame's own span
        top, bottom = sh["frame_pt"]
        frame_per_pt = (10.0 * step) / (bottom - top)
        scale_ratio.append(r.slope / frame_per_pt - 1.0)
        tops = [ly for ly in g.layers if ly.source == "stratum_rule"]
        row["lines_found"] = len(tops)
        used = set()
        sheet_errs = []
        for t in sh["boundaries_m"]:
            best = min(((abs(ly.top - t), i) for i, ly in enumerate(tops)
                        if i not in used), default=None)
            if best is None or best[0] > 0.5:
                missed += 1
                continue
            used.add(best[1])
            ly = tops[best[1]]
            e = ly.top - t
            errs.append(e)
            sheet_errs.append(e)
            pms.append(ly.plus_minus or 0.0)
            inside += abs(e) <= (ly.plus_minus or 0.0)
            inside_truth += abs(e) <= math.hypot(ly.plus_minus or 0.0,
                                                 TRUTH_PM_M)
        extra += len(tops) - len(used)
        row["extra_lines"] = len(tops) - len(used)
        row["error_m"] = _stats(sheet_errs)
        per_sheet.append(row)
        # the agent's path: a rough box round each true line, measured
        sc_vals = {nv["scale"]: vals}
        desc = next((c for c in g.columns if "description" in c.names), None)
        if desc is None:
            continue
        for t in sh["boundaries_m"]:
            y = r.y_at(t)
            if y is None:
                continue
            for _ in range(DRAWS):
                cy = y + rng.gauss(0.0, BOX_SD_PT)
                cx = (desc.x0 + desc.x1) / 2.0 + rng.gauss(0.0, BOX_SD_PT)
                half = 0.35 * (desc.x1 - desc.x0)
                box = [cx - half, cy - 1.5, cx + half, cy + 1.5]
                res = measure(doc, p, where=box, kind="line",
                              pad=2.5 * BOX_SD_PT, values=sc_vals)
                snaps["total"] += 1
                v = res.get("value")
                if res.get("ambiguous"):
                    alts = [a.get("value") or {} for a in
                            res.get("alternatives", [])]
                    ok = any(abs(_first_number(a) - t) <= max(
                        a.get("plus_minus", 0), 0.03) for a in alts
                        if _first_number(a) is not None)
                    snaps["listed"] += 1 if ok else 0
                    snaps["wrong"] += 0 if ok else 1
                    continue
                if not v:
                    snaps["wrong"] += 1
                    continue
                d = _first_number(v)
                if res.get("unsnapped"):
                    key = ("unsnapped_inside" if abs(d - t) <=
                           v["plus_minus"] else "unsnapped_outside")
                    snaps[key] += 1
                elif abs(d - t) <= math.hypot(v["plus_minus"], TRUTH_PM_M):
                    snaps["right"] += 1
                else:
                    snaps["wrong"] += 1
    n = len(errs)
    return {
        "sheets": len(logs["scanned_sheets"]),
        "sheets_with_ruler": sum(1 for r in per_sheet if "anchor_rule" in r),
        "labels_found_per_sheet": sorted({r.get("labels_found", 0)
                                          for r in per_sheet}),
        "anchor_rules": sorted({r.get("anchor_rule") or "none"
                                for r in per_sheet}),
        "truth_lines": sum(r["truth_lines"] for r in per_sheet),
        "matched": n, "missed": missed, "extra_lines": extra,
        "error_m": _stats(errs),
        "inside_own_pm": _pct(inside, n),
        "inside_pm_and_truth_pm": _pct(inside_truth, n),
        "plus_minus_m": _stats(pms),
        "label_scale_vs_frame_span": _stats(scale_ratio),
        "measure_from_rough_boxes": snaps,
        "per_sheet": per_sheet,
    }


def _first_number(v: Dict[str, Any]) -> Optional[float]:
    for k, x in v.items():
        if k in ("plus_minus", "unit", "confidence", "display",
                 "plus_minus_pt", "warnings"):
            continue
        if isinstance(x, (int, float)):
            return float(x)
    return None


# -- 2. vector log sheets ------------------------------------------------------

def vector_logs(doc, pages) -> Dict[str, Any]:
    offs, shifts, rules, resid = [], [], [], []
    for p in pages:
        ps = find_scales(doc, p)
        sc = next((s for s in ps.scales() if s.axis == "y" and s.usable
                   and s.quantity in ("depth", "elevation")), None)
        if sc is None:
            rules.append("none")
            continue
        rules.append(sc.anchor_rule_kind)
        resid.append(sc.residual_pt)
        for a in sc.anchors:
            if a.kind == "frame_line" or a.box is None:
                continue
            u, v = (a.box[0] + a.box[2]) / 2.0, (a.box[1] + a.box[3]) / 2.0
            centre = sc.along(u, v)
            offs.append(centre - a.position)
        # how far tying the ruler to the ticks moved each layer top
        g = LG.log_grid(doc, [p])
        orig = LG._vector_anchor
        try:
            LG._vector_anchor = lambda d, pg: None
            g0 = LG.log_grid(Document(content=doc._doc.tobytes()), [p])
        finally:
            LG._vector_anchor = orig
        for a, b in zip(g.layers, g0.layers):
            shifts.append(a.top - b.top)
    return {"pages": len(pages), "anchor_rules": rules,
            "label_centre_minus_tick_pt": _stats(offs),
            "fit_residual_pt": _stats(resid),
            "layer_top_shift_m": _stats(shifts)}


# -- 3. pits -------------------------------------------------------------------

def pits(doc, pages) -> Dict[str, Any]:
    refused = 0
    for p in pages:
        ps = find_scales(doc, p)
        if not [s for s in ps.scales() if s.axis == "y"
                and s.quantity in ("depth", "elevation")]:
            refused += 1
    return {"pages": len(pages), "refused": refused}


# -- 4. grading sheets ----------------------------------------------------------

def grading(doc, gt, rng) -> Dict[str, Any]:
    ax = gt["axes"]
    sizes = gt["sizes_mm"]
    out_sheets = []
    e_pass, e_size, inside, n_pts = [], [], 0, 0
    counts = {"right": 0, "listed": 0, "wrong": 0, "unsnapped": 0,
              "total": 0}
    for sh in gt["sheets"]:
        p = sh["page"]
        ps = find_scales(doc, p)
        frame = next((f for f in ps.frames if f.kind == "plot"
                      and "x" in f.scales and f.scales["x"].transform
                      == "log10"), None)
        row: Dict[str, Any] = {"page": p}
        if frame is None:
            row["result"] = "no log-x chart found"
            out_sheets.append(row)
            continue
        row["axes_found"] = sorted(frame.scales)
        vals = {}
        for k, sc in frame.scales.items():
            if sc.needs_values:
                vals[sc.id] = axis_label_values(sc.label_boxes, sc,
                                                frame.extent, ax[k])
        ps = find_scales(doc, p, values=vals) if vals else ps
        frame = ps.frame_of(f"{frame.id}.x") or frame
        sx, sy = frame.scales.get("x"), frame.scales.get("y")
        if not (sx and sy and sx.usable and sy.usable):
            row["result"] = "an axis is missing or refused"
            out_sheets.append(row)
            continue
        points = []
        for size, pct in zip(sizes, sh["passing_pct"]):
            points.append((size, pct))
            if pct >= 100.0:
                break               # the first 100 is the last point drawn
        sheet_pass = []
        for size, pct in points:
            u, v = sx.position_of(size), sy.position_of(pct)
            if u is None or v is None:
                continue
            x, y = unrotate_point(u, v, sx.angle_deg)
            n_pts += 1
            for _ in range(DRAWS):
                cx = x + rng.gauss(0.0, BOX_SD_PT)
                cy = y + rng.gauss(0.0, BOX_SD_PT)
                res = measure(doc, p, where=[cx - 2, cy - 2, cx + 2, cy + 2],
                              kind="point", pad=2.5 * BOX_SD_PT,
                              values=vals)
                counts["total"] += 1
                if res.get("ambiguous"):
                    counts["listed"] += 1
                    continue
                if res.get("unsnapped") or not res.get("value"):
                    counts["unsnapped"] += 1
                    continue
                rx = res["value"].get(sx.id) or {}
                ry = res["value"].get(sy.id) or {}
                vx, vy = _first_number(rx), _first_number(ry)
                if vx is None or vy is None:
                    counts["unsnapped"] += 1
                    continue
                ep = vy - pct
                es = math.log10(vx) - math.log10(size)
                ok = (abs(ep) <= ry["plus_minus"] + 0.5
                      and abs(vx - size) <= rx["plus_minus"] + 0.005 * size)
                # a neighbouring marker taken for this one is a wrong snap
                if abs(ep) > 3.0 or abs(es) > 0.05:
                    counts["wrong"] += 1
                    continue
                counts["right"] += 1
                e_pass.append(ep)
                e_size.append(es)
                sheet_pass.append(ep)
                inside += ok
        row["points"] = len(points)
        row["passing_error_pct"] = _stats(sheet_pass)
        out_sheets.append(row)
    snapped = len(e_pass)
    return {"sheets": len(gt["sheets"]),
            "sheets_read": sum(1 for r in out_sheets if "points" in r),
            "table_points": n_pts, "measurements": counts,
            "passing_error_pct": _stats(e_pass),
            "size_error_log10": _stats(e_size),
            "inside_pm_plus_print": _pct(inside, snapped),
            "per_sheet": out_sheets}


# -- main ----------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs_pdf")
    ap.add_argument("truth")
    ap.add_argument("--extra-truth")
    ap.add_argument("--grading-pdf")
    ap.add_argument("--json")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    truth = load_truth(a.truth, a.extra_truth)
    doc = Document(filepath=a.logs_pdf)
    out: Dict[str, Any] = {}
    out["scanned_logs"] = scanned_logs(doc, truth["logs"], rng)
    if truth["logs"].get("vector_pages"):
        out["vector_logs"] = vector_logs(doc, truth["logs"]["vector_pages"])
    if truth["logs"].get("pit_pages"):
        out["pits"] = pits(doc, truth["logs"]["pit_pages"])
    if a.grading_pdf and truth.get("grading"):
        out["grading"] = grading(Document(filepath=a.grading_pdf),
                                 truth["grading"], rng)
    text = json.dumps(out, indent=1, default=str)
    print(text)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
