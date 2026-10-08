"""Visual scales in report ingest: geometry as a VOTER, behind its own setting.

The W3 design (``module_work/VISUAL_SCALES_DESIGN.md`` §6.5, owner decision
6) puts planlens' scales under three readers: the log floor gets layer tops
measured from the stratum lines a scan draws, the lab reader gets a check of
a plotted grading curve against the sheet's own table, and the sounding
reader gets a code reading of a plotted trace to set beside the model's
digitising. Each changes the reader's numbers, so each waits for a cluster
run: **off by default** (``GEOTECH_INGEST_VISUAL_SCALES``, or
:func:`use_visual_scales` around a run), and "a voter first" — what code
measures is set beside what the record holds, and disagreements are QA
entries; only the log floor's layer tops enter the floor itself, and only
with the setting on.

OFF (the default) the readers behave as before the planlens scales build
as nearly as the installed planlens allows: ``log_grid`` now finds stratum
lines in a scan's pixels, and :func:`log_grid_for` folds those boundaries
back into the layer above (the grid's own ruler and columns from the pixels
stay — they are planlens' reading of the page, not a layer top).

ON:

* **Logs** (:func:`log_grid_for`): a scan with no text has its depth labels
  read by one structured call over numbered crops (the app's own sheet,
  :mod:`funhouse_agent.scale_labels`) — an extra call, where the design
  hoped to fold it into the reader's first call: the floor must have its
  depths BEFORE the model is shown it. Layer tops measured from the pixels,
  each with its +/-, enter the floor (:func:`report_ingest.log_floor.
  seed_from_grid`), where the merge keeps them unless the model brings
  evidence, and records every split.
* **Lab** (:func:`plot_check`): a gradation the sheet TABULATES is set
  against the curve it PLOTS — every plotted marker read through the plot's
  fitted axes. The table stays the record; a point that disagrees is a
  ``plot_vs_table`` QA entry.
* **Soundings** (:func:`trace_check`): each digitised point is read again by
  code where its channel's trace crosses that depth; agreement and
  disagreement are counted per channel, and disagreements are QA entries.
  The record keeps the model's series.
* **zoom_plot** (:func:`zoom_plot`, the ONE helper behind the three readers'
  tool): with the setting on, a crop that holds a fitted plot also carries
  code's reading of it (its axes and its plotted markers).
"""

from __future__ import annotations

import dataclasses
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

ENV = "GEOTECH_INGEST_VISUAL_SCALES"

_FORCED: ContextVar[Optional[bool]] = ContextVar("ingest_visual_scales",
                                                 default=None)

#: Code's value and the other voter's disagree past this many times code's
#: +/- (the owner's "about 3x", decision 4).
DISAGREE_FACTOR = 3.0
#: ... and never closer than these, in the value's own unit: a percent
#: passing is printed to whole percent, a grading size to a few percent.
MIN_PERCENT_GAP = 1.0
#: A plotted marker belongs to a tabulated sieve within this many decades.
SIZE_MATCH_DECADES = 0.03
#: Most QA entries one check writes (the rest are counted in the note).
MAX_QA_ROWS = 8


def enabled() -> bool:
    """Whether the visual-scales voters run (default: off)."""
    forced = _FORCED.get()
    if forced is not None:
        return forced
    return (os.environ.get(ENV) or "").strip().lower() in (
        "1", "on", "true", "yes")


@contextmanager
def use_visual_scales(on: bool = True) -> Iterator[None]:
    """Turn the voters on (or off) for the code inside the block."""
    token = _FORCED.set(bool(on))
    try:
        yield
    finally:
        _FORCED.reset(token)


# ---------------------------------------------------------------------------
# Reading a scan's printed labels through the ingest's engine
# ---------------------------------------------------------------------------

class LabelCell(BaseModel):
    """One numbered cell of the label sheet."""

    model_config = ConfigDict(extra="forbid")

    number: int = Field(description="the cell's number, #N")
    text: str = Field(description="the number exactly as printed, with "
                                  "every trailing zero; '-' when the cell "
                                  "cannot be read")


class LabelReading(BaseModel):
    """What the label sheet says, cell by cell."""

    model_config = ConfigDict(extra="forbid")

    labels: List[LabelCell] = Field(
        default_factory=list, description="one entry per numbered cell")


LABEL_SYSTEM = (
    "You read the numbers printed on a page's scale - a depth ruler, a plot "
    "axis, a scale bar - from numbered crops of them. You copy what is "
    "printed and never compute or guess.")

LABEL_PROMPT = (
    "Each numbered cell (#N) shows ONE number printed on a scale, cut from "
    "a page and enlarged. Give every cell: its number and the number it "
    "shows exactly as printed (digits, decimal point, minus or plus sign, "
    "every trailing zero: 4.0, not 4; no units). Where a cell shows no "
    "number or you cannot read it with confidence, give '-'.")


def _cache(doc: Any) -> Dict[Any, Optional[str]]:
    c = getattr(doc, "_ingest_label_cache", None)
    if c is None:
        c = {}
        try:
            setattr(doc, "_ingest_label_cache", c)
        except Exception:                       # pragma: no cover
            pass
    return c


def read_labels(doc: Any, items: Sequence[Tuple[int, Sequence[float]]],
                engine: Any, charge: Optional[Callable[[Any], None]] = None
                ) -> Tuple[List[Optional[str]], Dict[str, Any]]:
    """The printed text of each label box in ``items`` (``None`` where not
    read), by one structured call per contact sheet; remembered on the
    document so a second reader of the same page spends nothing."""
    from funhouse_agent.scale_labels import label_sheets, normalise_label
    from report_ingest.engine import image_block, text_block, user

    cache = _cache(doc)
    texts: List[Optional[str]] = [None] * len(items)
    info: Dict[str, Any] = {"calls": 0, "labels": len(items), "read": 0,
                            "from_memory": 0, "errors": []}
    todo: List[int] = []
    for i, (page, box) in enumerate(items):
        key = (int(page), tuple(round(float(v), 1) for v in box))
        if key in cache:
            texts[i] = cache[key]
            info["from_memory"] += 1
        else:
            todo.append(i)
    if todo and engine is not None:
        sub = [items[i] for i in todo]
        for png, idx, _size in label_sheets(doc, sub):
            try:
                reply = engine.complete(
                    [user(text_block(LABEL_PROMPT), image_block(png))],
                    system=LABEL_SYSTEM, output_format=LabelReading)
            except Exception as exc:            # a voter's call, not the run
                info["errors"].append(f"{type(exc).__name__}: {exc}")
                continue
            info["calls"] += 1
            if charge is not None:
                charge(reply)
            parsed = reply.parsed
            got: Dict[int, Optional[str]] = {}
            for cell in (getattr(parsed, "labels", None) or []):
                got[int(cell.number)] = normalise_label(cell.text)
            for k in idx:
                i = todo[k]
                texts[i] = got.get(k + 1)
                page, box = items[i]
                cache[(int(page), tuple(round(float(v), 1)
                                        for v in box))] = texts[i]
    info["read"] = sum(1 for t in texts if t is not None)
    return texts, info


def page_scales(doc: Any, page: int, engine: Any = None,
                charge: Optional[Callable[[Any], None]] = None,
                spent: Optional[Dict[str, Any]] = None):
    """``(PageScales, values)`` for ``page``, the labels of any scale waiting
    for them read through ``engine`` (and the values that fitted them)."""
    from planlens.document.scalefinder import find_scales
    ps = find_scales(doc, int(page))
    pend = ps.needing_values()
    values: Dict[str, List[Optional[str]]] = {}
    if pend and engine is not None:
        items: List[Tuple[int, Sequence[float]]] = []
        spans = []
        for sc in pend:
            spans.append((sc.id, len(items), len(items) + len(sc.label_boxes)))
            items.extend((int(page), b) for b in sc.label_boxes)
        texts, info = read_labels(doc, items, engine, charge)
        if spent is not None:
            spent["label_calls"] = spent.get("label_calls", 0) + info["calls"]
        for sid, a, b in spans:
            values[sid] = texts[a:b]
        if any(t is not None for t in texts):
            ps = find_scales(doc, int(page), values=values)
    return ps, values


def meter() -> Tuple[Dict[str, Any], Callable[[Any], None]]:
    """A cost dict and the hook that charges a reply to it — for a caller
    (a scorer) that reads a grid's labels outside a reader."""
    spent: Dict[str, Any] = {"calls": 0, "input_tokens": 0,
                             "output_tokens": 0, "cache_read_tokens": 0,
                             "seconds": 0.0, "dollars": 0.0}

    def charge(reply: Any) -> None:
        spent["calls"] += 1
        spent["input_tokens"] += reply.usage.input_tokens
        spent["output_tokens"] += reply.usage.output_tokens
        spent["cache_read_tokens"] += reply.usage.cache_read_tokens
        spent["seconds"] += reply.seconds
        spent["dollars"] += reply.usage.dollars(reply.model)

    return spent, charge


def add_cost(cost: Dict[str, Any], extra: Dict[str, Any]) -> None:
    """Fold ``extra`` (a :func:`meter` dict) into a reader's ``cost``."""
    if not extra or not extra.get("calls"):
        return
    for k, v in extra.items():
        if isinstance(v, (int, float)):
            cost[k] = (cost.get(k) or 0) + v


# ---------------------------------------------------------------------------
# Logs: the grid, its labels, and the off switch
# ---------------------------------------------------------------------------

def _pixel_layer(ly: Any) -> bool:
    return (getattr(ly, "source", "") == "stratum_rule"
            and (getattr(ly, "evidence", None) or {}).get("found_in")
            == "pixels")


def without_pixel_layers(grid: Any) -> Any:
    """``grid`` with every layer boundary found in a scan's PIXELS folded
    into the layer above — the floor as it was before planlens read stratum
    lines off scans. The grid itself is not changed; a copy is returned."""
    layers = list(getattr(grid, "layers", None) or [])
    if not any(_pixel_layer(ly) for ly in layers):
        return grid
    folded: List[Any] = []
    for ly in layers:
        if folded and _pixel_layer(ly):
            prev = folded[-1]
            desc = " ".join(d for d in (prev.description, ly.description)
                            if d).strip()
            box = prev.bbox
            if box is not None and ly.bbox is not None:
                box = (min(box[0], ly.bbox[0]), min(box[1], ly.bbox[1]),
                       max(box[2], ly.bbox[2]), max(box[3], ly.bbox[3]))
            elif box is None:
                box = ly.bbox
            folded[-1] = dataclasses.replace(
                prev, bottom=ly.bottom, description=desc, bbox=box,
                pages=tuple(sorted(set(prev.pages) | set(ly.pages))))
        else:
            folded.append(ly)
    try:
        return dataclasses.replace(grid, layers=folded)
    except TypeError:                           # pragma: no cover - not a
        grid.layers = folded                    # dataclass: change in place
        return grid


def log_grid_for(doc: Any, pages: Sequence[int], engine: Any = None,
                 charge: Optional[Callable[[Any], None]] = None
                 ) -> Tuple[Any, Dict[str, Any]]:
    """``(grid, info)``: ``log_grid`` over one log's pages, as the setting
    says. OFF: pixel-found layer boundaries folded away. ON: a textless
    scan's depth labels read (one structured call over numbered crops) and
    the grid read again with them; ``info`` says what was read."""
    from planlens.document.loggrid import log_grid
    pages = [int(p) for p in pages]
    grid = log_grid(doc, pages)
    if not enabled():
        return without_pixel_layers(grid), {}
    info: Dict[str, Any] = {"visual_scales": True}
    nv = getattr(grid, "needs_values", None) or {}
    if nv and engine is not None:
        items: List[Tuple[int, Sequence[float]]] = []
        spans = []
        for pg in sorted(nv):
            boxes = list(nv[pg].get("labels") or [])
            spans.append((int(pg), len(items), len(items) + len(boxes)))
            items.extend((int(pg), b) for b in boxes)
        texts, linfo = read_labels(doc, items, engine, charge)
        values = {pg: texts[a:b] for pg, a, b in spans}
        info["labels_read"] = {str(pg): ", ".join("?" if t is None else t
                                                  for t in vals)
                               for pg, vals in values.items()}
        info["label_calls"] = linfo["calls"]
        if any(t is not None for t in texts):
            grid = log_grid(doc, pages, values=values)
    info["pixel_layers"] = sum(1 for ly in grid.layers if _pixel_layer(ly))
    return grid, info


def layer_votes(grid: Any, record_layers: Sequence[Any],
                record_unit: str = "") -> List[Dict[str, Any]]:
    """Every layer top measured from a scan's pixels set against the
    record's nearest layer top: the geometry voter, row by row.

    It matters most where the scan has no text: there the floor holds no
    layer (a layer is its words, and the words are the model's to read off
    the picture), so the measured tops can only VOTE on the model's.
    """
    unit = (getattr(grid, "unit", None) or "").strip()
    if record_unit and unit and record_unit.strip() != unit:
        return []
    tops = []
    for ly in getattr(grid, "layers", None) or []:
        if _pixel_layer(ly) and ly.top is not None:
            tops.append(ly)
    rec = []
    for ly in record_layers or []:
        q = getattr(ly, "top", None)
        v = getattr(q, "value", q)
        if isinstance(v, (int, float)):
            rec.append(float(v))
    rows: List[Dict[str, Any]] = []
    for ly in tops:
        pm = float(ly.plus_minus or 0.0)
        row: Dict[str, Any] = {"geometry_top": round(float(ly.top), 4),
                               "plus_minus": round(pm, 4),
                               "page": int(ly.pages[0]) if ly.pages else None}
        if (ly.evidence or {}).get("dashed"):
            row["dashed"] = True
        if not rec:
            row.update({"record_top": None, "agreement": "no_record_layer"})
        else:
            near = min(rec, key=lambda t: abs(t - float(ly.top)))
            gap = abs(near - float(ly.top))
            row.update({"record_top": near, "gap": round(gap, 4),
                        "agreement": ("agree" if gap <= max(
                            DISAGREE_FACTOR * pm, 1e-6) else "disagree")})
        rows.append(row)
    return rows


def layer_votes_qa(votes: Sequence[Dict[str, Any]], where: str,
                   unit: str, pages: Sequence[int]) -> List[Any]:
    """QA entries for :func:`layer_votes`: each disagreement, and a note."""
    from report_ingest.model import QAEntry
    if not votes:
        return []
    out = []
    dis = [v for v in votes if v.get("agreement") == "disagree"]
    for v in dis[:MAX_QA_ROWS]:
        out.append(QAEntry(
            kind="disagreement", where=where,
            detail=(f"a stratum line measured on the scan puts a layer top "
                    f"at {v['geometry_top']:g} +/- {v['plus_minus']:g} "
                    f"{unit}; the record's nearest top is "
                    f"{v['record_top']:g} {unit}, {v['gap']:g} away (more "
                    f"than {DISAGREE_FACTOR:g}x the measurement's +/-)"),
            values=[f"geometry {v['geometry_top']:g}",
                    f"record {v['record_top']:g}"],
            pages=[v["page"]] if v.get("page") is not None
            else [int(p) for p in pages]))
    agree = sum(1 for v in votes if v.get("agreement") == "agree")
    none = sum(1 for v in votes if v.get("agreement") == "no_record_layer")
    out.append(QAEntry(
        kind="note", where=where,
        detail=(f"layer tops measured from the scan's stratum lines: "
                f"{len(votes)}; {agree} agree with the record's tops within "
                f"{DISAGREE_FACTOR:g}x their +/-, {len(dis)} disagree"
                + (f", {none} with no record layer to compare" if none
                   else "")),
        pages=[int(p) for p in pages]))
    return out


# ---------------------------------------------------------------------------
# Plots: markers and traces read through fitted axes
# ---------------------------------------------------------------------------

def _number(v: Any) -> Optional[float]:
    if not isinstance(v, dict):
        return None
    for k, x in v.items():
        if k in ("plus_minus", "unit", "confidence", "display",
                 "plus_minus_pt", "warnings"):
            continue
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return float(x)
    return None


def _plot_frames(ps: Any) -> List[Any]:
    return [f for f in getattr(ps, "frames", []) or []
            if getattr(f, "kind", "") in ("plot", "chart", "profile")]


def _axis(frame: Any, axis: str) -> Optional[Any]:
    for sc in frame.scales.values():
        if sc.axis == axis and sc.usable:
            return sc
    return None


def plot_markers(doc: Any, page: int, frame: Any,
                 values: Optional[Dict[str, Any]] = None
                 ) -> List[Dict[str, float]]:
    """Every plotted marker in ``frame`` read on both axes:
    ``[{x, y, x_pm, y_pm}]`` (markers code could not read on both are left
    out)."""
    from planlens.document.measuring import measure
    xs, ys = _axis(frame, "x"), _axis(frame, "y")
    if xs is None or ys is None:
        return []
    res = measure(doc, int(page), where=list(frame.extent), kind="point",
                  pad=0.0, values=values or None)
    rows = []
    if res.get("ambiguous"):
        cands = [a.get("values") or {} for a in res.get("alternatives") or []]
    elif isinstance(res.get("value"), dict) and not res.get("unsnapped"):
        cands = [res["value"]]
    else:
        cands = []
    for vals in cands:
        vx, vy = vals.get(xs.id), vals.get(ys.id)
        x, y = _number(vx), _number(vy)
        if x is None or y is None:
            continue
        rows.append({"x": x, "y": y,
                     "x_pm": float((vx or {}).get("plus_minus") or 0.0),
                     "y_pm": float((vy or {}).get("plus_minus") or 0.0)})
    return rows


def _grading_points(test: Any) -> List[Tuple[float, float, str]]:
    """``(size_mm, percent, sieve)`` of a tabulated gradation."""
    res = getattr(test, "result", None)
    if getattr(res, "kind", "") != "gradation":
        return []
    out = []
    for p in getattr(res, "percent_passing", None) or []:
        if p.size is None or p.size.value is None:
            continue
        unit = (p.size.unit or "mm").strip().lower()
        size = float(p.size.value)
        if unit in ("in", "inch", "inches", '"'):
            size *= 25.4
        elif unit in ("um", "µm", "micron", "microns"):
            size /= 1000.0
        elif unit not in ("mm", ""):
            continue
        if size > 0:
            out.append((size, float(p.percent_passing), p.sieve or ""))
    return out


def plot_check(doc: Any, pages: Sequence[int], floor_tests: Sequence[Any],
               engine: Any = None,
               charge: Optional[Callable[[Any], None]] = None
               ) -> Optional[Dict[str, Any]]:
    """A tabulated gradation set against the grading curve the sheet plots.

    For each floor gradation with sizes: the sheet's plot whose x axis is a
    log axis (particle size) and y axis a linear one (percent passing) is
    found, every plotted marker read through its fitted axes, and each
    table point compared with the marker at its size. ``None`` when the
    sheet tabulates no gradation; otherwise a dict the QA section reads.
    """
    tables = [(t, pts) for t in floor_tests
              for pts in [_grading_points(t)] if len(pts) >= 3]
    if not tables:
        return None
    markers: List[Dict[str, float]] = []
    where = None
    spent: Dict[str, Any] = {}
    for page in pages:
        try:
            ps, values = page_scales(doc, page, engine, charge, spent)
        except Exception as exc:                # noqa: BLE001 - a voter
            return {"status": "unavailable",
                    "note": f"the plot could not be read: "
                            f"{type(exc).__name__}: {exc}"}
        for fr in _plot_frames(ps):
            xs, ys = _axis(fr, "x"), _axis(fr, "y")
            if xs is None or ys is None or xs.transform != "log10" \
                    or ys.transform != "linear":
                continue
            got = plot_markers(doc, page, fr, values)
            if len(got) > len(markers):
                markers, where = got, {"page": int(page), "frame": fr.id}
    out: Dict[str, Any] = {"label_calls": spent.get("label_calls", 0)}
    if not markers:
        out.update({"status": "no_plot",
                    "note": "no plotted grading curve with readable axes "
                            "and markers was found on the sheet"})
        return out
    rows: List[Dict[str, Any]] = []
    agree = 0
    unmatched = 0
    for _test, pts in tables:
        for size, pct, sieve in pts:
            near = [m for m in markers if m["x"] > 0
                    and abs(math.log10(m["x"]) - math.log10(size))
                    <= max(SIZE_MATCH_DECADES,
                           3.0 * m["x_pm"] / (m["x"] * math.log(10.0)))]
            if not near:
                unmatched += 1
                continue
            m = min(near, key=lambda m: abs(math.log10(m["x"])
                                            - math.log10(size)))
            gap = abs(m["y"] - pct)
            tol = max(MIN_PERCENT_GAP, DISAGREE_FACTOR * m["y_pm"])
            if gap <= tol:
                agree += 1
            else:
                rows.append({"sieve": sieve, "size_mm": round(size, 4),
                             "table": pct, "plot": round(m["y"], 2),
                             "plot_plus_minus": round(m["y_pm"], 2),
                             "gap": round(gap, 2)})
    out.update({"status": "checked", **(where or {}),
                "markers": len(markers),
                "table_points": sum(len(p) for _t, p in tables),
                "agree": agree, "disagree": rows,
                "no_marker_at_size": unmatched})
    return out


def trace_check(doc: Any, pages: Sequence[int], investigation: Any,
                engine: Any = None,
                charge: Optional[Callable[[Any], None]] = None
                ) -> Optional[Dict[str, Any]]:
    """Each digitised sounding point read again by code, where its
    channel's trace crosses that depth. ``None`` when the series was not
    digitised (a tabulated sheet is exact already)."""
    from planlens.document.measuring import measure
    from planlens.document.scales import unrotate_point
    data = None if investigation is None else (
        investigation.cpt or investigation.dcp)
    if data is None or not getattr(data, "digitised", False) \
            or not data.points:
        return None
    channels = (("qc", "fs", "u2") if investigation.cpt is not None
                else ("blows", "penetration", "index"))
    series: Dict[str, List[Tuple[float, float]]] = {}
    for pt in data.points:
        d = getattr(pt.depth, "value", None)
        if d is None:
            continue
        for ch in channels:
            v = getattr(pt, ch, None)
            v = getattr(v, "value", v)
            if isinstance(v, (int, float)):
                series.setdefault(ch, []).append((float(d), float(v)))
    if not series:
        return None
    spent: Dict[str, Any] = {}
    frames = []
    for page in pages:
        try:
            ps, values = page_scales(doc, page, engine, charge, spent)
        except Exception:                       # noqa: BLE001 - a voter
            continue
        for fr in _plot_frames(ps):
            xs, ys = _axis(fr, "x"), _axis(fr, "y")
            if xs is not None and ys is not None:
                frames.append((int(page), fr, xs, ys, values))
    out: Dict[str, Any] = {"label_calls": spent.get("label_calls", 0),
                           "channels": {}}
    if not frames:
        out["status"] = "no_plot"
        out["note"] = ("no plotted panel with readable axes was found on "
                       "the sheet")
        return out
    for ch, pts in series.items():
        # The panel this channel is plotted in: the one whose x axis spans
        # most of the values the model read for it, and whose y axis spans
        # its depths.
        best = None
        for page, fr, xs, ys, values in frames:
            lo_x = sorted((xs.value_at(xs.along(fr.extent[0], fr.extent[1])),
                           xs.value_at(xs.along(fr.extent[2], fr.extent[1]))))
            lo_y = sorted((ys.value_at(ys.along(fr.extent[0], fr.extent[1])),
                           ys.value_at(ys.along(fr.extent[0], fr.extent[3]))))
            span = (lo_x[1] - lo_x[0]) or 1.0
            inside = sum(1 for d, v in pts
                         if lo_x[0] - 0.05 * span <= v <= lo_x[1] + 0.05 * span
                         and lo_y[0] <= d <= lo_y[1])
            if best is None or inside > best[0]:
                best = (inside, page, fr, xs, ys, values)
        if best is None or best[0] < max(3, 0.8 * len(pts)):
            out["channels"][ch] = {"status": "no_panel",
                                   "note": "no panel's axes span this "
                                           "channel's values"}
            continue
        _n, page, fr, xs, ys, values = best
        row = {"panel": fr.id, "page": page, "compared": 0, "agree": 0,
               "disagree": [], "several": 0, "no_trace": 0}
        x_mid = (fr.extent[0] + fr.extent[2]) / 2.0
        for d, v in pts:
            s = ys.position_of(d)
            if s is None:
                row["no_trace"] += 1
                continue
            yd = unrotate_point(x_mid, s, ys.angle_deg)[1] \
                if hasattr(ys, "angle_deg") else s
            box = [fr.extent[0] + 1.0, yd - 2.0, fr.extent[2] - 1.0, yd + 2.0]
            try:
                res = measure(doc, page, where=box, kind="curve",
                              at={ys.id: d}, pad=1.0, values=values or None)
            except Exception:                   # noqa: BLE001
                row["no_trace"] += 1
                continue
            val = res.get("value")
            if res.get("ambiguous"):
                alts = [a for a in res.get("alternatives") or []
                        if _number(a.get("value")) is not None]
                if len(alts) == 1:
                    val = alts[0]["value"]
                else:
                    row["several"] += 1
                    continue
            num = _number(val)
            if num is None or res.get("unsnapped"):
                row["no_trace"] += 1
                continue
            pm = float(val.get("plus_minus") or 0.0)
            row["compared"] += 1
            gap = abs(num - v)
            if gap <= max(DISAGREE_FACTOR * pm, 1e-9):
                row["agree"] += 1
            else:
                row["disagree"].append({"depth": d, "model": v,
                                        "code": round(num, 4),
                                        "code_plus_minus": round(pm, 4),
                                        "gap": round(gap, 4)})
        out["channels"][ch] = row
    out["status"] = "checked"
    return out


def plot_check_qa(check: Optional[Dict[str, Any]], pages: Sequence[int]
                  ) -> List[Any]:
    """QA entries for a :func:`plot_check`."""
    from report_ingest.model import QAEntry
    if not check or check.get("status") != "checked":
        return []
    pg = [int(check["page"])] if check.get("page") is not None \
        else [int(p) for p in pages]
    rows = check.get("disagree") or []
    out = [QAEntry(
        kind="plot_vs_table", where="lab.gradation",
        detail=(f"the plotted grading curve read by code disagrees with "
                f"the sheet's own table at {r['sieve'] or r['size_mm']} "
                f"({r['size_mm']} mm): table {r['table']} %, plot "
                f"{r['plot']} +/- {r['plot_plus_minus']} %; the record "
                f"keeps the table"),
        values=[f"table {r['table']}", f"plot {r['plot']}"], pages=pg)
        for r in rows[:MAX_QA_ROWS]]
    out.append(QAEntry(
        kind="note", where="lab.gradation",
        detail=(f"plot against table: {check.get('agree', 0)} of "
                f"{check.get('table_points', 0)} tabulated point(s) agree "
                f"with the plotted curve read by code, {len(rows)} "
                f"disagree, {check.get('no_marker_at_size', 0)} have no "
                f"plotted marker at their size"),
        pages=pg))
    return out


def trace_check_qa(check: Optional[Dict[str, Any]], pages: Sequence[int]
                   ) -> List[Any]:
    """QA entries for a :func:`trace_check`: one per channel that has
    disagreements, and one note with the counts."""
    from report_ingest.model import QAEntry
    if not check or check.get("status") != "checked":
        return []
    out = []
    parts = []
    for ch, row in (check.get("channels") or {}).items():
        if row.get("status") == "no_panel":
            parts.append(f"{ch}: no panel")
            continue
        dis = row.get("disagree") or []
        parts.append(f"{ch}: {row.get('agree', 0)}/{row.get('compared', 0)} "
                     f"agree, {row.get('several', 0)} with several traces, "
                     f"{row.get('no_trace', 0)} with none")
        if dis:
            worst = max(dis, key=lambda r: r["gap"])
            out.append(QAEntry(
                kind="disagreement", where=f"sounding.{ch}",
                detail=(f"code read the {ch} trace at {row['compared']} "
                        f"digitised depth(s): {len(dis)} differ from the "
                        f"model's reading by more than "
                        f"{DISAGREE_FACTOR:g}x code's +/- (largest at "
                        f"{worst['depth']:g}: model {worst['model']:g}, "
                        f"code {worst['code']:g} +/- "
                        f"{worst['code_plus_minus']:g}); the record keeps "
                        f"the model's series"),
                values=[f"model {worst['model']:g}",
                        f"code {worst['code']:g}"],
                pages=[int(row.get("page", pages[0] if pages else 0))]))
    out.append(QAEntry(kind="note", where="sounding",
                       detail="code reading of the plotted traces: "
                              + "; ".join(parts),
                       pages=[int(p) for p in pages]))
    return out


# ---------------------------------------------------------------------------
# The one zoom_plot
# ---------------------------------------------------------------------------

def _code_note(doc: Any, page: int, box: Sequence[float], engine: Any = None,
               charge: Optional[Callable[[Any], None]] = None
               ) -> Optional[str]:
    """What code reads of a plot the crop holds: its axes and its markers."""
    ps, values = page_scales(doc, page, engine, charge)
    lines: List[str] = []
    for fr in _plot_frames(ps):
        e = fr.extent
        ix = max(0.0, min(e[2], box[2]) - max(e[0], box[0]))
        iy = max(0.0, min(e[3], box[3]) - max(e[1], box[1]))
        area = max(1.0, (e[2] - e[0]) * (e[3] - e[1]))
        if ix * iy < 0.5 * area:
            continue
        xs, ys = _axis(fr, "x"), _axis(fr, "y")
        if xs is None or ys is None:
            pend = [s for s in fr.scales.values() if s.needs_values]
            if pend:
                lines.append(f"plot {fr.id}: its axis labels have no text "
                             f"and were not read; code cannot read values "
                             f"off it")
            continue
        lines.append(f"plot {fr.id}: x {xs.transform}, y {ys.transform}, "
                     f"both fitted to the drawn grid and printed labels")
        pts = plot_markers(doc, page, fr, values)
        if pts:
            shown = "; ".join(f"({p['x']:.4g}, {p['y']:.4g})"
                              for p in pts[:30])
            lines.append(f"  {len(pts)} plotted marker(s) read by code "
                         f"(x, y): {shown}"
                         + (" ..." if len(pts) > 30 else ""))
    if not lines:
        return None
    return ("CODE'S READING OF THIS CROP (measured from the drawing, a second "
            "voter beside your own reading - where they differ, say so):\n"
            + "\n".join(lines))


def zoom_plot(doc: Any, pages: Sequence[int], arguments: Dict[str, Any], *,
              dpi: float, noun: str, instruction: str,
              zooms: List[Dict[str, Any]], engine: Any = None,
              charge: Optional[Callable[[Any], None]] = None) -> List[Any]:
    """The readers' ``zoom_plot`` tool: a magnified crop of one region of a
    page, as content blocks. With visual scales on, a crop holding a fitted
    plot also carries code's reading of it."""
    from report_ingest.engine import image_block, text_block
    page = int(arguments.get("page", pages[0]))
    if page not in list(pages):
        raise ValueError(f"page {page} is not part of this {noun} "
                         f"({list(pages)})")
    bbox = arguments.get("bbox")
    if not bbox or len(list(bbox)) != 4:
        raise ValueError("bbox must be four numbers: x0, y0, x1, y1")
    box = [float(v) for v in bbox]
    png, info = doc.render(page, bbox=box, dpi=dpi)
    zooms.append({"page": page, "bbox": box,
                  "why": str(arguments.get("why") or "")})
    blocks: List[Any] = [text_block(
        f"page {page}, box {[round(v, 1) for v in info['clip']]} at "
        f"{info['dpi']} dpi ({info['width_px']}x{info['height_px']} px). "
        + instruction), image_block(png)]
    if enabled():
        try:
            note = _code_note(doc, page, box, engine, charge)
        except Exception:                       # noqa: BLE001 - a voter
            note = None
        if note:
            blocks.insert(1, text_block(note))
    return blocks


__all__ = ["enabled", "use_visual_scales", "ENV", "read_labels",
           "page_scales", "without_pixel_layers", "log_grid_for",
           "plot_markers", "plot_check", "trace_check", "plot_check_qa",
           "trace_check_qa", "zoom_plot", "LabelReading", "LabelCell"]
