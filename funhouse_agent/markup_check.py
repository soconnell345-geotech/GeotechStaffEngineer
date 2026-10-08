"""Check that every mark is on what it is meant to mark.

``annotate_document`` writes a box, circle, callout, highlight or note wherever
it is told to. A field session (2026-10-01) asked for red circles round tags an
agent had found by looking; the circles went on at coordinates the agent wrote
from nothing, in empty paper, and nothing noticed until the reviewer opened the
file. The owner: "make sure it never draws anything based on inferred
locations ... Probably should also visually confirm its markups."

So after the marked copy is written, each mark is checked the way a person
would: a crop of the MARKED copy around the mark is rendered and a one-shot
vision call is asked what is actually there and whether that is the thing the
mark is meant to mark. The tool's result then says which marks are confirmed,
which are misplaced and which the look could not settle, so the agent redoes
or removes the misplaced ones before handing the file over.

**What a mark is compared with** (Foundry brief 4, 2026-10-07). The check
used to ask whether what is under a mark matched its label or, with none, its
COMMENT. A review comment is a request ("Please confirm the 8.33 % maximum
...") rather than the name of a thing, so on 10 checks of comments placed on
the right line none was confirmed and 8 "misplaced" verdicts were wrong. The
thing is now named apart from the comment — the agent's ``target``, else the
``label``, else the ``quote`` the mark was anchored on — and the question is
whether the mark is on THAT thing and whether the comment could be about it.
With nothing named, the question is whether the mark is on the thing the
comment is about. A callout's or a note's look reads the whole line or object
at its spot, not the one letter under the arrow's tip.

**Every anchor is checked.** Quote-anchored marks and sticky notes were not,
and after "misplaced" verdicts every agent re-placed its comment by quote or as
a note, where the check went quiet — which is how a comment on the WRONG note
reached a file (GPT-5.4, brief 4). A quote anchor lands exactly on its words;
what can still be wrong is the choice of words, and the comment question asks
exactly that.

**"Encloses" is measured, not judged.** For a box or a ring the look returns
where the thing is; the mark encloses it when the thing's centre lies inside
the mark with :data:`ENCLOSE_SLACK_PT` of slack — the suite's own rule
(``review_eval.checks.check_markups_on_targets``). The side call took the end
of a leader's shoulder as part of a tag ("- GCE") and said tight rings that
held the whole tag did not enclose it (4 wrong rejections of 70 good rings,
brief 4); its reading of what is there still decides WHICH thing it is (a QCE,
an arrowhead), and size is still measured (:data:`MAX_AREA_FACTOR`).
"""

from __future__ import annotations

import contextvars
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Kinds whose place on the page is the point of the mark — every kind but a
#: reply, which is threaded onto an existing comment rather than placed.
CHECK_KINDS = ("box", "circle", "highlight", "callout", "note")

#: Anchors a caller supplied as geometry.
GEOMETRY_ANCHORS = ("bbox", "point", "points_at")

#: Anchors found from the text layer: exact on their words, but the words can
#: be the wrong ones.
QUOTE_ANCHORS = ("quote", "quote_fuzzy")

#: Every anchor a checked mark may have come from.
CHECKED_ANCHORS = GEOMETRY_ANCHORS + QUOTE_ANCHORS

#: Kinds drawn ROUND a thing: whether they enclose it is measured.
ENCLOSING_KINDS = ("box", "circle")

#: Kinds that mark a spot on a line or an object: the look reads the whole
#: line, and the crop is wide enough to hold one.
POINTING_KINDS = ("callout", "note")

#: Most marks checked in one call (each is one vision call); the rest are
#: reported as not checked.
MAX_CHECKS = 60

#: Checks run at once.
WORKERS = 4

#: Context round a mark in the crop: at least this many points each side,
#: and at least this fraction of the mark's own size.
CONTEXT_PT = 36.0
CONTEXT_FRAC = 1.0

#: ... and, for a callout's tip or a note's corner, this much each side
#: across, so the whole line of text at the spot is in the crop (a notes line
#: on a drawing runs 150-250 pt; the 8.33 % line of brief 4 was 165 pt).
POINTING_CONTEXT_X_PT = 120.0

#: A thing is enclosed when its centre is inside the mark with this much
#: slack, in points — the suite's placement check uses the same.
ENCLOSE_SLACK_PT = 4.0

#: How each kind is described to the look: what it is, what it is meant to do
#: to the thing, and where the thing is relative to it.
_KIND_WORDS = {
    "box": ("red rectangle", "enclose", "inside the red rectangle"),
    "circle": ("red ring", "enclose", "inside the red ring"),
    "highlight": ("yellow highlight", "cover", "under the yellow highlight"),
    "callout": ("red arrow (a callout: a comment box with a leader ending in "
                "an arrowhead)", "point at", "at the tip of the arrow"),
    "note": ("yellow sticky-note icon", "sit on",
             "at the icon's top-left corner — the note is about what that "
             "corner sits on, part of which may be under the icon"),
}

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
#: MEASURED, not judged: the look returns where the thing is in the crop,
#: and the mark is too wide only when its area is more than
#: :data:`MAX_AREA_FACTOR` times the thing's. Good rings measured 3-10x, the
#: blanket rings 70-180x.
#:
#: Where the thing is comes back in PIXELS of the crop as sent (``thing_px``)
#: and is converted with the crop's own size — the convention every look uses
#: since GPT-5.4's 0-999 boxes were measured scaled by 0.87-1.10 between
#: identical calls while its pixel boxes were within a few points
#: (2026-10-07). An old-style ``thing_box`` on the 0-999 grid still counts.
_WHERE_Q = ("where that thing is in this {w} x {h} pixel image, as a box "
            "[x0, y0, x1, y1] in pixels (0,0 top-left), or null if it is not "
            "there")

#: A mark this many times the area of the thing it names is a blanket, not a
#: mark (the suite's own placement check uses 60).
MAX_AREA_FACTOR = 30.0

#: The smallest area a thing is taken to have, in pt² — a tag's lettering
#: box can be read very thin.
MIN_THING_AREA = 25.0


def _thing_on_page(thing_box: Any, view: Any, size: Any = None
                   ) -> Optional[Tuple[float, float, float, float]]:
    """The look's box of the thing as PDF points, or ``None`` when it gave no
    usable box. ``thing_box`` is in pixels of the crop when its ``size``
    (w, h as sent) is given, otherwise on the 0-999 grid."""
    from funhouse_agent import vision_view
    try:
        tb = [float(v) for v in thing_box]
        if len(tb) != 4:
            return None
        if size:
            return vision_view.px_box_to_page(list(view), tb, size)
        return vision_view.image_box_to_page(list(view), tb)
    except (TypeError, ValueError):
        return None


def _too_wide(mark_bbox: Any, thing_box: Any, view: Any,
              size: Any = None) -> Optional[float]:
    """The mark's area over the thing's, when that is more than
    :data:`MAX_AREA_FACTOR`; else ``None`` (including when the look gave no
    usable box — size is then not judged at all). ``thing_box`` is in pixels
    of the crop when its ``size`` (w, h as sent) is given, otherwise on the
    0-999 grid."""
    page_box = _thing_on_page(thing_box, view, size)
    try:
        mb = [float(v) for v in mark_bbox]
    except (TypeError, ValueError):
        return None
    if page_box is None or len(mb) != 4:
        return None
    x0, y0, x1, y1 = page_box
    thing = max((x1 - x0) * (y1 - y0), MIN_THING_AREA)
    mark = max(0.0, mb[2] - mb[0]) * max(0.0, mb[3] - mb[1])
    ratio = mark / thing
    return ratio if ratio > MAX_AREA_FACTOR else None


def _encloses(mark_bbox: Any, thing: Optional[Sequence[float]],
              slack: float = ENCLOSE_SLACK_PT) -> Optional[bool]:
    """Whether the thing's centre lies inside the mark, ``slack`` points
    each way; ``None`` without a usable box of either."""
    try:
        mx0, my0, mx1, my1 = (float(v) for v in mark_bbox)
        tx0, ty0, tx1, ty1 = (float(v) for v in thing)
    except (TypeError, ValueError):
        return None
    cx, cy = (tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0
    return (mx0 - slack <= cx <= mx1 + slack
            and my0 - slack <= cy <= my1 + slack)


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
        spec = _spec_view(specs[i] if isinstance(specs[i], dict) else {})
        out.append({"index": i, "row": row, "spec": spec})
    return out


def _spec_view(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The spec as planlens read it: an ``anchor`` object's fields lifted to
    the top (planlens reads them so, with a note)."""
    anchor = spec.get("anchor")
    if not isinstance(anchor, dict):
        return spec
    out = {k: v for k, v in spec.items() if k != "anchor"}
    for k, v in anchor.items():
        out.setdefault(k, v)
    return out


def marks_to_check(result: Dict[str, Any], specs: Sequence[Any]
                   ) -> List[Dict[str, Any]]:
    """The written marks to look at: every kind in :data:`CHECK_KINDS`,
    however it was anchored (:data:`CHECKED_ANCHORS`)."""
    return [r for r in _rows_with_specs(result, specs)
            if r["row"].get("kind") in CHECK_KINDS
            and r["row"].get("anchored_by") in CHECKED_ANCHORS]


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _target(spec: Dict[str, Any]) -> Tuple[str, str]:
    """``(what the mark is meant to be on, where that came from)`` — the
    agent's ``target``, else the ``label``, else the ``quote``; ``("", "")``
    when the spec names no thing (the comment then says what it is about)."""
    for key in ("target", "label", "quote"):
        value = _text(spec.get(key), 160)
        if value:
            return value, key
    return "", ""


def _expected(spec: Dict[str, Any]) -> str:
    """What the result says the mark was compared with."""
    target, _src = _target(spec)
    return target or _text(spec.get("comment"), 160) or \
        "(no target, label, quote or comment given)"


def _fold(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def _same_words(a: str, b: str) -> bool:
    return bool(_fold(a)) and _fold(a) == _fold(b)


_SOURCE_WORDS = {
    "target": "what the mark is on",
    "label": "the mark's label",
    "quote": "the printed words the mark was anchored on",
}


def _prompt(kind: str, spec: Dict[str, Any], size: Sequence[int]
            ) -> Tuple[str, bool]:
    """The look's prompt for one mark, and whether it asks about the comment
    as well as the thing (``comment_fits``)."""
    word, verb, where = _KIND_WORDS.get(kind, ("mark", "mark", "at the mark"))
    target, source = _target(spec)
    comment = _text(spec.get("comment"), 300)
    ask_comment = bool(comment) and not _same_words(comment, target)
    label_note = (" with a short red label beside it" if spec.get("label")
                  else "")
    placed = "placed" if kind == "note" else "drawn"
    parts = [f"This is a crop of a page that a review tool has just marked "
             f"up. A {word} has been {placed} on it{label_note}. The question is "
             f"whether the mark is on the right thing."]
    if target:
        parts.append(f"The mark is meant to {verb}: {target}   "
                     f"({_SOURCE_WORDS[source]})")
    if comment:
        parts.append(f'The review comment on this mark: "{comment}". A '
                     f"review comment is a remark or a request ABOUT the "
                     f"thing marked; it need not repeat the words printed "
                     f"there.")
    if not target:
        parts.append(f"The mark is meant to {verb} the thing that comment "
                     f"is about.")
    whole = (" Read the WHOLE line of text, or the whole object, there — "
             "not a single letter or stroke." if kind in POINTING_KINDS
             or kind == "highlight" else "")
    own = ("the icon itself" if kind == "note"
           else "the mark's own red label or comment box")
    parts.append(f"Look at what is actually {where} — not at {own}.{whole}")
    named = "the thing named above" if target else \
        "the thing the comment is about"
    spot = where.split(" —")[0]
    keys = [("same_thing", f"true if what is there IS {named} (a "
                           f"paraphrase, a partial reading or a misprint of "
                           f"it counts; a different item that only shares a "
                           f"word or a number does not), else false")]
    asked = ask_comment and bool(target)
    if asked:
        keys.append(("comment_fits", "true if the review comment could be "
                                     "about what is there; false only if it "
                                     "is plainly about something else "
                                     "(another note, value or item)"))
    if kind in ENCLOSING_KINDS:
        keys.append(("encloses", f"true if the {word} goes all the way round "
                                 f"that thing"))
    keys.append(("inside", f"the exact text or symbol {spot} (the whole line "
                           f"or object), or 'nothing' if it is blank paper "
                           f"or linework only"))
    if kind in ENCLOSING_KINDS and size:
        keys.append(("thing_px", _WHERE_Q.format(
            w=int(size[0]), h=int(size[1])).strip()))
    keys.append(("sure", "true or false"))
    skeleton = "{" + ", ".join(f'"{k}": ...' for k, _d in keys) + "}"
    parts.append("Reply with ONLY a JSON object " + skeleton + ", where:\n"
                 + "\n".join(f"- {k}: {d}" for k, d in keys))
    return "\n\n".join(parts), asked


def _crop_box(row: Dict[str, Any]) -> List[float]:
    kind = row.get("kind")
    if kind == "callout" and row.get("points_at"):
        x, y = row["points_at"]
        box = [x - 6.0, y - 6.0, x + 6.0, y + 6.0]
    elif kind == "note" and row.get("bbox"):
        # A sticky note's icon hangs from the spot it was put on (its
        # top-left corner); what it is about is there.
        x, y = row["bbox"][0], row["bbox"][1]
        box = [x - 6.0, y - 6.0, x + 18.0, y + 18.0]
    else:
        box = list(row.get("bbox") or [0.0, 0.0, 0.0, 0.0])
    w, h = box[2] - box[0], box[3] - box[1]
    mx = max(CONTEXT_PT, CONTEXT_FRAC * w)
    my = max(CONTEXT_PT, CONTEXT_FRAC * h)
    if kind in POINTING_KINDS:
        mx = max(mx, POINTING_CONTEXT_X_PT)
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


def _bool(value: Any) -> Optional[bool]:
    """A yes/no the look wrote, as True / False, else ``None``."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower() if value is not None else ""
    if text in ("true", "yes"):
        return True
    if text in ("false", "no"):
        return False
    return None


def _names_match(seen: str, target: str) -> Optional[bool]:
    """``True`` when the look's reading of what is there contains the named
    thing or is contained in it (``"- GCE"`` for ``GCE``: the end of a
    leader read as part of the tag); ``None`` when that cannot say — a
    different reading is left to the look's own verdict."""
    a, b = _fold(seen), _fold(target)
    if len(a) < 2 or len(b) < 2:
        return None
    return True if (b in a or a in b) else None


def _verdict(kind: str, got: Dict[str, Any], row: Dict[str, Any],
             spec: Dict[str, Any], clip: Any, size: Any,
             asked_comment: bool) -> Tuple[str, str]:
    """``(verdict, what to add to "seen")`` for one look's answer."""
    sure = _bool(got.get("sure"))
    encl_said = _bool(got.get("encloses"))
    target, _src = _target(spec)
    identity = _bool(got.get("same_thing"))
    if identity is None and target:
        identity = _names_match(str(got.get("inside") or ""), target)
    fits = _bool(got.get("comment_fits")) if asked_comment else None

    def decided(ok: bool) -> str:
        if sure is False:
            return "unsure"
        return "confirmed" if ok else "misplaced"

    if fits is False:
        return decided(False), " — the comment is about something else"
    if kind not in ENCLOSING_KINDS:
        ok = identity if identity is not None else encl_said
        return ("unsure", "") if ok is None else (decided(ok), "")
    if got.get("thing_px") is not None:
        thing_raw, thing_size = got.get("thing_px"), size
    else:                       # an old-style answer on the 0-999 grid
        thing_raw, thing_size = got.get("thing_box"), None
    thing = _thing_on_page(thing_raw, clip, thing_size)
    if identity is False:
        return decided(False), ""
    if thing is not None and identity is True:
        inside = _encloses(row.get("bbox"), thing)
        if inside is False:
            return decided(False), " — beside the mark, not inside it"
        ratio = _too_wide(row.get("bbox"), thing_raw, clip, thing_size)
        if ratio is not None:
            # Round the right thing, but drawn so wide it does not single it
            # out.
            return decided(False), (f" — but the mark is {ratio:.0f} times "
                                    f"the area of the thing")
        if inside:
            return decided(True), ""
    # No measurable box, or the look did not say which thing it saw: its own
    # yes/no (and size, where it gave a box) decide, as before.
    ok = encl_said if identity is None else (identity and encl_said)
    if ok is None:
        return "unsure", ""
    if ok:
        ratio = _too_wide(row.get("bbox"), thing_raw, clip, thing_size)
        if ratio is not None:
            return decided(False), (f" — but the mark is {ratio:.0f} times "
                                    f"the area of the thing")
    return decided(bool(ok)), ""


def _check_one(pdf_bytes: bytes, item: Dict[str, Any], engine) -> Dict[str, Any]:
    from funhouse_agent import vision_view

    row, spec = item["row"], item["spec"]
    kind = row.get("kind")
    out = {"index": item["index"], "page": row.get("page"),
           "pdf_page": (row.get("page") or 0) + 1, "kind": kind,
           "names": _expected(spec), "bbox": row.get("bbox"),
           "anchored_by": row.get("anchored_by")}
    crop = _crop_box(row)
    try:
        image, info = vision_view.render_view(
            pdf_bytes, page=int(row.get("page") or 0), bbox=crop,
            pad_frac=0.0, engine=engine)
        size = (int(info["width_px"]), int(info["height_px"]))
        prompt, asked_comment = _prompt(kind, spec, size)
        answer = engine.analyze_image(image, prompt)
    except Exception as exc:  # noqa: BLE001 - a failed check is reported
        out.update(verdict="not_checked",
                   reason=f"{type(exc).__name__}: {str(exc)[:160]}")
        return out
    got = _parse(answer)
    if got is None:
        out.update(verdict="unsure", seen=str(answer)[:160])
        return out
    clip = (info or {}).get("clip") or crop
    verdict, why = _verdict(kind, got, row, spec, clip, size, asked_comment)
    out["verdict"] = verdict
    out["seen"] = (str(got.get("inside") or "")[:120] + why)[:160]
    return out


def check_marks(output_pdf: str, result: Dict[str, Any], specs: Sequence[Any],
                engine, max_checks: int = MAX_CHECKS,
                workers: int = WORKERS) -> Optional[Dict[str, Any]]:
    """Look at every mark on the marked copy. Returns the ``check`` block for
    the tool result, or ``None`` when there is nothing to check or no engine
    to check with."""
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
        "Every mark was looked at on the marked copy, and each is on the "
        "thing it is meant to mark."
        if not bad else
        f"{bad} mark(s) are misplaced or could not be confirmed ('seen' says "
        "what is actually there). Do not hand the file over as it is: find "
        "the thing each comment is about again and write the copy again "
        "with append=false, leaving out any mark you cannot place. For a "
        "thing found by looking, zoom with render_region until it is legible "
        "in a view of 300 pt or less and take THAT zoom's view + the thing's "
        "image_box; for words, quote the words the comment is about. Every "
        "anchor — box, point, quote or note — is checked the same way, so "
        "switching anchor does not settle a verdict. Widening a mark until "
        "it takes the thing in does not place it: zoom until you can box the "
        "thing itself. Name what a mark is on with target when the comment "
        "is a request rather than the thing's name.")
    return block


__all__ = ["CHECK_KINDS", "GEOMETRY_ANCHORS", "QUOTE_ANCHORS",
           "CHECKED_ANCHORS", "check_marks", "marks_to_check"]
