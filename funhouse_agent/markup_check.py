"""Check that every mark placed by LOCATION encloses what it names.

``annotate_document`` writes a box, circle or callout wherever it is told to.
A field session (2026-10-01) asked for red circles round tags an agent had
found by looking; the circles went on at coordinates the agent wrote from
nothing, in empty paper, and nothing noticed until the reviewer opened the
file. The owner: "make sure it never draws anything based on inferred
locations ... Probably should also visually confirm its markups."

So after the marked copy is written, each mark the caller placed by a box or a
point (not by a quote, which the text layer anchors exactly) is checked the way
a person would: a crop of the MARKED copy around the mark is rendered and a
one-shot vision call is asked whether the red mark encloses — or, for a
callout, points at — what the mark's label (or comment) names. The tool's
result then says which marks are confirmed, which are misplaced and which the
look could not settle, so the agent redoes or removes the misplaced ones before
handing the file over.
"""

from __future__ import annotations

import contextvars
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

#: Kinds whose place on the page is the point of the mark.
CHECK_KINDS = ("box", "circle", "highlight", "callout")

#: Anchors a caller supplied as geometry — the ones that can be wrong.
GEOMETRY_ANCHORS = ("bbox", "point", "points_at")

#: Most marks checked in one call (each is one vision call); the rest are
#: reported as not checked.
MAX_CHECKS = 60

#: Checks run at once.
WORKERS = 4

#: Context round a mark in the crop: at least this many points each side,
#: and at least this fraction of the mark's own size.
CONTEXT_PT = 36.0
CONTEXT_FRAC = 1.0

_WORD = {"box": "rectangle", "circle": "ring", "highlight": "highlight",
         "callout": "arrow"}

_PROMPT = (
    "This is a crop of a page that a review tool has just marked up. A red "
    "{word} has been drawn on it{label_note}. The question is whether the "
    "mark is in the right place.\n\nThe mark is meant to {verb}: {expected}\n\n"
    "Look at what is actually {where} the red {word} — not at the mark's own "
    "red label text beside it.{close_q} Reply with ONLY a JSON object: "
    '{{"encloses": true or false, "inside": "the exact text or symbol {where} '
    "the red {word}, or 'nothing' if it is blank paper or linework only\", "
    '{close_key}"sure": true or false}}')

#: Enclosing marks are also checked for SIZE. Foundry rc3 run (2026-10-04):
#: told six times that its rings were misplaced, an agent widened them until
#: each took in its tag somewhere — rings 70-125 pt across round 10 pt tags —
#: and the check confirmed them. A ring that wide on a dense sheet does not
#: say which thing it means.
#:
#: 5.32.0 asked the model whether the mark was "drawn closely round" the
#: thing. On GPT-5.4 (Funhouse, 2026-10-07) that rejected CORRECT rings — a
#: 19 x 14 pt ring centred on a 10 x 4 pt tag was "drawn far wider than it" —
#: because any ring that fits a small tag is a few times its size. So size is
#: now MEASURED, not judged: the look returns where the thing is in the crop
#: (a 0-999 box), and the mark is too wide only when its area is more than
#: :data:`MAX_AREA_FACTOR` times the thing's. Good rings measured 3-10x, the
#: blanket rings 70-180x.
_WHERE_Q = (" Also give where that thing is in this image, as a box on a "
            "0-999 grid (0,0 top-left, 999,999 bottom-right), or null if it "
            "is not there.")
_WHERE_KEY = '"thing_box": [x0, y0, x1, y1] or null, '

#: A mark this many times the area of the thing it names is a blanket, not a
#: mark (the suite's own placement check uses 60).
MAX_AREA_FACTOR = 30.0

#: The smallest area a thing is taken to have, in pt² — a tag's lettering
#: box can be read very thin.
MIN_THING_AREA = 25.0


def _too_wide(mark_bbox: Any, thing_box: Any, view: Any) -> Optional[float]:
    """The mark's area over the thing's, when that is more than
    :data:`MAX_AREA_FACTOR`; else ``None`` (including when the look gave no
    usable box — size is then not judged at all)."""
    from funhouse_agent import vision_view
    try:
        tb = [float(v) for v in thing_box]
        mb = [float(v) for v in mark_bbox]
        if len(tb) != 4 or len(mb) != 4:
            return None
        x0, y0, x1, y1 = vision_view.image_box_to_page(list(view), tb)
    except (TypeError, ValueError):
        return None
    thing = max((x1 - x0) * (y1 - y0), MIN_THING_AREA)
    mark = max(0.0, mb[2] - mb[0]) * max(0.0, mb[3] - mb[1])
    ratio = mark / thing
    return ratio if ratio > MAX_AREA_FACTOR else None


def _rows_with_specs(result: Dict[str, Any], specs: Sequence[Any]
                     ) -> List[Dict[str, Any]]:
    """Pair each written row with the spec it came from. Rows come back in
    spec order with the skipped ones left out, so the skipped indices say
    which spec each row is."""
    skipped = {int(s.get("index")) for s in (result.get("skipped") or [])
               if isinstance(s, dict) and s.get("index") is not None}
    order = [i for i in range(len(specs)) if i not in skipped]
    out = []
    for row, i in zip(result.get("written") or [], order):
        spec = specs[i] if isinstance(specs[i], dict) else {}
        out.append({"index": i, "row": row, "spec": spec})
    return out


def marks_to_check(result: Dict[str, Any], specs: Sequence[Any]
                   ) -> List[Dict[str, Any]]:
    """The written marks whose place came from caller geometry."""
    return [r for r in _rows_with_specs(result, specs)
            if r["row"].get("kind") in CHECK_KINDS
            and r["row"].get("anchored_by") in GEOMETRY_ANCHORS]


def _expected(spec: Dict[str, Any]) -> str:
    text = str(spec.get("label") or spec.get("comment") or "").strip()
    return text[:160] or "(no label or comment given)"


def _crop_box(row: Dict[str, Any]) -> List[float]:
    if row.get("kind") == "callout" and row.get("points_at"):
        x, y = row["points_at"]
        box = [x - 6.0, y - 6.0, x + 6.0, y + 6.0]
    else:
        box = list(row.get("bbox") or [0.0, 0.0, 0.0, 0.0])
    w, h = box[2] - box[0], box[3] - box[1]
    mx = max(CONTEXT_PT, CONTEXT_FRAC * w)
    my = max(CONTEXT_PT, CONTEXT_FRAC * h)
    return [box[0] - mx, box[1] - my, box[2] + mx, box[3] + my]


def _parse(text: Any) -> Optional[Dict[str, Any]]:
    if isinstance(text, dict):
        return text
    m = re.search(r"\{.*\}", str(text or ""), re.DOTALL)
    if not m:
        return None
    try:
        got = json.loads(m.group(0))
    except ValueError:
        return None
    return got if isinstance(got, dict) else None


def _check_one(pdf_bytes: bytes, item: Dict[str, Any], engine) -> Dict[str, Any]:
    from funhouse_agent import vision_view

    row, spec = item["row"], item["spec"]
    kind = row.get("kind")
    word = _WORD.get(kind, "mark")
    out = {"index": item["index"], "page": row.get("page"),
           "pdf_page": (row.get("page") or 0) + 1, "kind": kind,
           "names": _expected(spec), "bbox": row.get("bbox")}
    encloses_kind = kind in ("box", "circle")
    crop = _crop_box(row)
    try:
        image, info = vision_view.render_view(
            pdf_bytes, page=int(row.get("page") or 0), bbox=crop,
            pad_frac=0.0, engine=engine)
        answer = engine.analyze_image(image, _PROMPT.format(
            word=word,
            label_note=(" with a short red label beside it"
                        if spec.get("label") else ""),
            verb=("point at" if kind == "callout" else "enclose"),
            where=("at the tip of" if kind == "callout" else "inside"),
            close_q=(_WHERE_Q if encloses_kind else ""),
            close_key=(_WHERE_KEY if encloses_kind else ""),
            expected=_expected(spec)))
    except Exception as exc:  # noqa: BLE001 - a failed check is reported
        out.update(verdict="not_checked",
                   reason=f"{type(exc).__name__}: {str(exc)[:160]}")
        return out
    got = _parse(answer)
    if got is None:
        out.update(verdict="unsure", seen=str(answer)[:160])
        return out
    encloses, sure = got.get("encloses"), got.get("sure")
    out["seen"] = str(got.get("inside") or "")[:120]
    ratio = (_too_wide(row.get("bbox"), got.get("thing_box"),
                       (info or {}).get("clip") or crop)
             if encloses is True and encloses_kind else None)
    if ratio is not None:
        # Round the right thing, but drawn so wide it does not single it out.
        out["verdict"] = "misplaced" if sure is not False else "unsure"
        out["seen"] = (out["seen"] + f" — but the mark is {ratio:.0f} times "
                       "the area of the thing")[:160]
    elif encloses is True:
        out["verdict"] = "confirmed" if sure is not False else "unsure"
    elif encloses is False:
        out["verdict"] = "misplaced" if sure is not False else "unsure"
    else:
        out["verdict"] = "unsure"
    return out


def check_marks(output_pdf: str, result: Dict[str, Any], specs: Sequence[Any],
                engine, max_checks: int = MAX_CHECKS,
                workers: int = WORKERS) -> Optional[Dict[str, Any]]:
    """Look at every geometry-placed mark on the marked copy. Returns the
    ``check`` block for the tool result, or ``None`` when there is nothing to
    check or no engine to check with."""
    items = marks_to_check(result, specs)
    if not items or engine is None or not hasattr(engine, "analyze_image"):
        return None
    with open(output_pdf, "rb") as fh:
        pdf_bytes = fh.read()
    todo, later = items[:max_checks], items[max_checks:]
    # Each worker runs in a COPY of this call's context, so the activity log
    # attributes its vision call to this turn (as sweep_pages does).
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(contextvars.copy_context().run, _check_one,
                               pdf_bytes, it, engine) for it in todo]
        checks = [f.result() for f in futures]
    by = {v: [c for c in checks if c["verdict"] == v]
          for v in ("confirmed", "misplaced", "unsure", "not_checked")}
    block: Dict[str, Any] = {
        "checked": len(checks),
        "confirmed": len(by["confirmed"]),
        "misplaced": [{k: c.get(k) for k in ("index", "pdf_page", "kind",
                                             "names", "bbox", "seen")}
                      for c in by["misplaced"]],
        "unsure": [{k: c.get(k) for k in ("index", "pdf_page", "kind",
                                          "names", "seen")}
                   for c in by["unsure"]],
    }
    if by["not_checked"]:
        block["not_checked"] = [{k: c.get(k) for k in ("index", "pdf_page",
                                                       "reason")}
                                for c in by["not_checked"]]
    if later:
        block["over_limit"] = len(later)
    bad = len(by["misplaced"]) + len(by["unsure"])
    block["note"] = (
        "Every mark placed by a box or point was looked at on the marked copy."
        if not bad else
        f"{bad} mark(s) are misplaced or could not be confirmed. Do not hand "
        "the file over as it is: find each one again (look at the page, take "
        "the view + image_box of what you find) and write the copy again with "
        "append=false, leaving out any mark you cannot place. A mark "
        "written from memory or by estimate lands in the wrong place, and "
        "widening a mark until it takes the thing in does not place it: zoom "
        "until you can box the thing itself.")
    return block


__all__ = ["CHECK_KINDS", "GEOMETRY_ANCHORS", "check_marks", "marks_to_check"]
