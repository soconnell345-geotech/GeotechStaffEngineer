"""The visual-scales measurement harness: every synthetic fixture, no model.

WHAT IT MEASURES (design section 7, module_work/VISUAL_SCALES_DESIGN.md):

- for every fixture variant in ``planlens.testing.visual_scale_fixtures``:
  whether a scale was found, its residual and anchor rule, and that a page
  not drawn to scale is refused;
- for every thing the fixture states (a stratum contact, a plotted marker, a
  curve read-off, a plan distance, a boring's coordinates): rough boxes drawn
  from the MEASURED error distributions of a vision model's boxes —
  a zoom (sd 1 pt), pixel boxes off a whole sheet (sd 2, 4 and 6 pt) and the
  old 0-999 grid (sd 40 and 70 pt, window +/-79 pt) — each measured with
  ``planlens.document.measuring.measure``;
- the error of each value against the truth, whether the truth lies inside
  the reported +/-, how often the snap landed on the right thing, and how
  often the result admitted ambiguity instead of choosing.

THE GATE (design section 7): on the fixtures, at least 95 % of readings
carry the truth inside their +/-, and no reading with a wrong snap is
returned without its alternatives. The exit code is 0 when the gate holds.

Label values for scans with no text come from the fixture's own printed
labels (``ScaleFixture.values_for``): the stand-in for the app's one vision
call over numbered crops. A misread experiment checks that one wrong label
value is dropped and named, and two are refused.

Usage::

    python module_work/scales_harness/harness.py            # everything
    python module_work/scales_harness/harness.py --quick    # fewer draws
    python module_work/scales_harness/harness.py --only log,plot --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

#: Box regimes: (name, error model, pad passed as the location error, draws
#: per reading in full mode). A positive number is the sd (pt) of a normal
#: error on each axis (pixel boxes measured 1-6 pt off, zooms about 1 pt);
#: ``"grid"`` is the old 0-999 grid, whose boxes were measured 57-91 pt off:
#: an error of that size in a random direction, with the worst of it as the
#: view's location error.
REGIMES = (
    ("zoom", 1.0, 3.0, 4),
    ("pixel_sd2", 2.0, 6.0, 4),
    ("pixel_sd4", 4.0, 11.0, 4),
    ("pixel_sd6", 6.0, 16.0, 4),
    ("old_grid", "grid", 91.0, 4),
)

#: A snap is RIGHT when it lands this close to the thing (points).
SNAP_TOL_PT = 1.6


def _value_of(v: Any) -> Optional[float]:
    """The numeric value out of a measure value dict."""
    if not isinstance(v, dict):
        return None
    for k, x in v.items():
        if k in ("plus_minus", "unit", "confidence", "display",
                 "plus_minus_pt", "warnings"):
            continue
        if isinstance(x, (int, float)):
            return float(x)
    return None


def _pm_of(v: Any) -> Optional[float]:
    return float(v["plus_minus"]) if isinstance(v, dict) and \
        v.get("plus_minus") is not None else None


def _error(rng, sd):
    if sd == "grid":
        r = rng.uniform(57.0, 91.0)
        a = rng.uniform(0.0, 2.0 * math.pi)
        return r * math.cos(a), r * math.sin(a)
    return rng.normal(0.0, sd), rng.normal(0.0, sd)


def _rough_box(rng, truth, sd, kind: str):
    """A box a model might draw round the thing: shifted, resized."""
    bx = truth.box_pt
    cx, cy = (bx[0] + bx[2]) / 2.0, (bx[1] + bx[3]) / 2.0
    w, h = bx[2] - bx[0], bx[3] - bx[1]
    dx, dy = _error(rng, sd)
    if kind == "line":
        # a box round a line: across part of its run, a few points tall
        w = w * rng.uniform(0.5, 1.0)
        h = max(h, 2.0) + rng.uniform(0.0, 4.0)
    else:
        w = max(w, 2.0) * rng.uniform(0.8, 1.6)
        h = max(h, 2.0) * rng.uniform(0.8, 1.6)
    return [cx + dx - w / 2, cy + dy - h / 2, cx + dx + w / 2, cy + dy + h / 2]


def _shift_box(rng, bx, sd):
    dx, dy = _error(rng, sd)
    return [bx[0] + dx, bx[1] + dy, bx[2] + dx, bx[3] + dy]


def _supply_values(fx, doc):
    """Values for every scale waiting for them, read off the fixture."""
    from planlens.document.scalefinder import find_scales
    ps = find_scales(doc, fx.page)
    pend = ps.needing_values()
    vals = {s.id: fx.values_for(s.label_boxes) for s in pend}
    return (vals or None), ps, pend


def _truth_value_for(reading, axis_key: str) -> Optional[float]:
    for k, v in reading.values.items():
        if k.endswith("." + axis_key) or k == axis_key:
            return v
    return None


def _point_check(res, reading) -> Tuple[Optional[bool], List[float]]:
    """Are both coordinates of a point inside their +/- ? Errors in value."""
    vals = res.get("value") or {}
    if not isinstance(vals, dict):
        return None, []
    errs = []
    inside = True
    any_ = False
    for sid, vd in vals.items():
        if sid == "needs_values":
            continue
        v, pm = _value_of(vd), _pm_of(vd)
        if v is None:
            continue
        if sid.endswith(".x") or sid.endswith("easting"):
            key = "easting" if sid.endswith("easting") else "x"
        elif sid.endswith(".y") or sid.endswith("northing"):
            key = "northing" if sid.endswith("northing") else "y"
        else:
            continue
        t = _truth_value_for(reading, key)
        if t is None:
            continue
        any_ = True
        err = v - t
        errs.append(err)
        if pm is None or abs(err) > pm + 1e-9:
            inside = False
    return (inside if any_ else None), errs


def run_fixture(fx, rng, quick: bool, regimes) -> Dict[str, Any]:
    from planlens.document.measuring import measure
    from planlens.document.scalefinder import find_scales
    doc = fx.open()
    t0 = time.time()
    vals, ps0, pend = _supply_values(fx, doc)
    ps = find_scales(doc, fx.page, values=vals) if vals else ps0
    t_find = time.time() - t0
    rep: Dict[str, Any] = {
        "name": fx.name, "kind": fx.kind, "raster": fx.raster,
        "seconds_find": round(t_find, 2),
        "scales": [{"id": s.id, "quantity": s.quantity,
                    "transform": s.transform, "needs_values": s.needs_values,
                    "residual_pt": round(s.residual_pt, 3),
                    "plus_minus_pt": round(s.plus_minus_pt, 3)
                    if s.usable else None,
                    "anchor_rule": s.anchor_rule_kind,
                    "confidence": s.confidence,
                    "per_point": s.b if s.usable else None}
                   for s in ps.scales()],
        "n_pending_before_values": len(pend),
        "readings": [],
    }
    if fx.not_to_scale:
        made = [s for s in ps.scales() if s.usable and s.axis in ("x", "y")]
        rep["refused_ok"] = not made
        doc.close()
        return rep
    for i, rd in enumerate(fx.readings):
        for name, sd, pad, n in regimes:
            draws = 1 if quick else n
            for _d in range(draws):
                box = _rough_box(rng, rd, sd, rd.kind)
                kw: Dict[str, Any] = {"pad": pad, "values": vals}
                if rd.kind == "curve":
                    k, v = next(iter(rd.at.items()))
                    kw["at"] = {k.split(".")[-1]: v}
                if rd.kind == "distance":
                    kw["to"] = _shift_box(rng, rd.to_box_pt, sd)
                    box = _shift_box(rng, rd.box_pt, sd)
                try:
                    res = measure(doc, fx.page, where=box, kind=rd.kind, **kw)
                except Exception as exc:          # a harness must not stop
                    rep["readings"].append({"i": i, "regime": name,
                                            "error": repr(exc)})
                    continue
                rep["readings"].append(_score(res, rd, i, name))
    doc.close()
    return rep


def _score(res, rd, i, regime) -> Dict[str, Any]:
    out: Dict[str, Any] = {"i": i, "regime": regime, "tag": rd.tag,
                           "kind": rd.kind}
    if res.get("ambiguous"):
        out["outcome"] = "ambiguous"
        # Is the truth among the alternatives?
        alts = res.get("alternatives") or []
        out["truth_listed"] = any(_alt_is_truth(a, rd) for a in alts)
        return out
    snapped = res.get("snapped_to")
    unsnapped = bool(res.get("unsnapped"))
    out["outcome"] = "unsnapped" if unsnapped else (
        "snapped" if snapped else "no_snap")
    if rd.kind == "point":
        inside, errs = _point_check(res, rd)
        out["inside"] = inside
        out["errors"] = errs
        if snapped and snapped.get("at_pt"):
            d = math.dist(snapped["at_pt"], rd.at_pt)
            out["snap_error_pt"] = round(d, 3)
            out["wrong_snap"] = d > SNAP_TOL_PT + 2.5
    elif rd.kind == "distance":
        v = (res.get("value") or {})
        val, pm = _value_of(v), _pm_of(v)
        if val is not None:
            out["error"] = val - rd.value
            out["inside"] = pm is not None and abs(val - rd.value) <= pm + 1e-9
            ends = res.get("ends") or []
            if ends and all(e.get("how") == "snapped" for e in ends):
                d1 = math.dist(ends[0]["at_pt"], rd.at_pt)
                d2 = math.dist(ends[1]["at_pt"], rd.to_pt)
                out["wrong_snap"] = max(d1, d2) > SNAP_TOL_PT + 2.5
                out["outcome"] = "snapped"
            else:
                out["outcome"] = "unsnapped"
    else:
        v = res.get("value")
        val, pm = _value_of(v), _pm_of(v)
        if val is not None:
            out["error"] = val - rd.value
            out["inside"] = pm is not None and abs(val - rd.value) <= pm + 1e-9
            out["plus_minus"] = pm
        if snapped:
            pt = snapped.get("at_pt")
            if pt is None and snapped.get("bbox"):
                b = snapped["bbox"]
                pt = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if rd.kind == "line":
                moved = res.get("snapped_to", {}).get("bbox")
                # a line is right when its position across the run matches
                b = moved
                yc = (b[1] + b[3]) / 2 if b else None
                xc = rd.at_pt[0]
                # compare at the truth's x: allow the line's own skew
                d = abs(yc - rd.at_pt[1]) if yc is not None else 99
                out["snap_error_pt"] = round(d, 3)
                out["wrong_snap"] = d > SNAP_TOL_PT + 2.0
            elif pt is not None:
                d = math.dist(pt, rd.at_pt)
                out["snap_error_pt"] = round(d, 3)
                out["wrong_snap"] = d > SNAP_TOL_PT + 2.5
    return out


def _alt_is_truth(a, rd) -> bool:
    pt = a.get("at_pt")
    if pt is None and a.get("bbox"):
        b = a["bbox"]
        pt = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    if pt is None:
        return False
    if rd.kind == "line":
        return abs(pt[1] - rd.at_pt[1]) <= SNAP_TOL_PT + 2.0
    return math.dist(pt, rd.at_pt) <= SNAP_TOL_PT + 2.5


def misread_experiment() -> Dict[str, Any]:
    """One wrong label value is dropped and named; two are refused."""
    from planlens.document.scalefinder import find_scales
    from planlens.testing.visual_scale_fixtures import build_log, log_variants
    out = {"one_wrong_dropped": 0, "one_wrong_total": 0,
           "two_wrong_refused": 0, "two_wrong_total": 0}
    for v in [v for v in log_variants() if v.raster][:6]:
        fx = build_log(v)
        doc = fx.open()
        ps = find_scales(doc, 0)
        pend = [s for s in ps.needing_values() if s.quantity == "depth"]
        if not pend:
            doc.close()
            continue
        s = pend[0]
        vals = fx.values_for(s.label_boxes)
        good = [x for x in vals]
        one = list(good)
        k = len(one) // 2
        if one[k] is not None:
            one[k] = one[k] + 2.0 * (one[1] - one[0] if one[1] and one[0]
                                     is not None else 1.0)
        ps1 = find_scales(doc, 0, values={s.id: one})
        s1 = ps1.get(s.id)
        out["one_wrong_total"] += 1
        if s1 is not None and s1.usable and s1.provenance.get("dropped"):
            if abs(s1.b - fx.scales["depth"]["per_point"]) <= \
                    0.002 * fx.scales["depth"]["per_point"]:
                out["one_wrong_dropped"] += 1
        two = list(good)
        if two[1] is not None and two[-2] is not None:
            two[1] = two[1] + 3.0 * (good[1] - good[0])
            two[-2] = two[-2] - 2.5 * (good[1] - good[0])
        ps2 = find_scales(doc, 0, values={s.id: two})
        s2 = ps2.get(s.id)
        out["two_wrong_total"] += 1
        if s2 is not None and not s2.usable:
            out["two_wrong_refused"] += 1
        doc.close()
    return out


def summarise(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    tot = defaultdict(int)
    by_kind: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_regime: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    errors: Dict[str, List[float]] = defaultdict(list)
    failures: List[Dict[str, Any]] = []
    for rep in reports:
        if "refused_ok" in rep:
            tot["not_to_scale"] += 1
            tot["refused_ok"] += int(rep["refused_ok"])
            continue
        for r in rep["readings"]:
            k = rep["kind"]
            g = r["regime"]
            if "error" in r and isinstance(r["error"], str):
                tot["exceptions"] += 1
                failures.append({"fixture": rep["name"], **r})
                continue
            tot["calls"] += 1
            by_regime[g]["calls"] += 1
            oc = r.get("outcome")
            by_regime[g][oc] += 1
            if oc == "ambiguous":
                by_regime[g]["truth_listed"] += int(r.get("truth_listed", False))
                continue
            inside = r.get("inside")
            if inside is None:
                by_regime[g]["no_value"] += 1
                continue
            tot["readings"] += 1
            by_kind[k]["readings"] += 1
            by_regime[g]["readings"] += 1
            tot["inside"] += int(bool(inside))
            by_kind[k]["inside"] += int(bool(inside))
            by_regime[g]["inside"] += int(bool(inside))
            if oc == "snapped":
                by_regime[g]["snapped_ok"] += int(not r.get("wrong_snap"))
                if r.get("wrong_snap"):
                    tot["wrong_snap_no_alts"] += 1
                    by_kind[k]["wrong_snap_no_alts"] += 1
                    failures.append({"fixture": rep["name"], **r})
            if not inside:
                failures.append({"fixture": rep["name"], **r})
            if "error" in r:
                errors[k].append(abs(r["error"]))
            for e in r.get("errors", []):
                errors[k].append(abs(e))
    def pct(a, b):
        return round(100.0 * a / b, 2) if b else None
    out = {
        "readings": tot["readings"], "inside": tot["inside"],
        "inside_pct": pct(tot["inside"], tot["readings"]),
        "wrong_snaps_without_alternatives": tot["wrong_snap_no_alts"],
        "calls": tot["calls"], "exceptions": tot["exceptions"],
        "not_to_scale_refused": f"{tot['refused_ok']} of {tot['not_to_scale']}",
        "by_kind": {k: {**v, "inside_pct": pct(v["inside"], v["readings"])}
                    for k, v in by_kind.items()},
        "by_regime": {g: {**v,
                          "ambiguous_pct": pct(v["ambiguous"], v["calls"]),
                          "snap_success_pct": pct(v["snapped_ok"],
                                                  v["calls"]),
                          "inside_pct": pct(v["inside"], v["readings"])}
                      for g, v in by_regime.items()},
        "median_abs_error": {k: (sorted(v)[len(v) // 2] if v else None)
                             for k, v in errors.items()},
        "max_abs_error": {k: (max(v) if v else None)
                          for k, v in errors.items()},
        "failures_shown": failures[:40],
        "n_failures": len(failures),
    }
    out["gate"] = bool(out["inside_pct"] is not None
                       and out["inside_pct"] >= 95.0
                       and out["wrong_snaps_without_alternatives"] == 0
                       and tot["exceptions"] == 0
                       and tot["refused_ok"] == tot["not_to_scale"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--json", default="")
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    import numpy as np
    from planlens.testing.visual_scale_fixtures import all_fixtures
    rng = np.random.default_rng(args.seed)
    only = {s for s in args.only.split(",") if s}
    t0 = time.time()
    fixtures = [fx for fx in all_fixtures() if not only or fx.kind in only]
    reports = []
    for fx in fixtures:
        t = time.time()
        rep = run_fixture(fx, rng, args.quick, REGIMES)
        reports.append(rep)
        if args.verbose:
            n = len([r for r in rep["readings"] if "inside" in r])
            ok = sum(1 for r in rep["readings"] if r.get("inside"))
            print(f"{fx.name:<40} {time.time() - t:5.1f}s  scales "
                  f"{len(rep['scales'])}  inside {ok}/{n}"
                  + ("  refused " + str(rep.get("refused_ok"))
                     if "refused_ok" in rep else ""), flush=True)
    summ = summarise(reports)
    summ["misread"] = misread_experiment()
    summ["fixtures"] = len(fixtures)
    summ["seconds"] = round(time.time() - t0, 1)
    print(json.dumps({k: v for k, v in summ.items()
                      if k != "failures_shown"}, indent=1))
    if args.verbose and summ["failures_shown"]:
        print("first failures:")
        for f in summ["failures_shown"][:25]:
            print("  ", f)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"summary": summ, "fixtures": reports}, fh, indent=1,
                      default=str)
    gate = summ["gate"] and (summ["misread"]["one_wrong_dropped"]
                             == summ["misread"]["one_wrong_total"]) and (
        summ["misread"]["two_wrong_refused"]
        == summ["misread"]["two_wrong_total"])
    print("GATE", "PASS" if gate else "FAIL")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
