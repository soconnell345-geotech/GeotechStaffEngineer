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
``quote`` the mark was anchored on — and the question is whether the mark is
on THAT thing and whether the comment could be about it. With nothing named,
the question is whether the mark is on the thing the comment is about. A
callout's or a note's look reads the whole line or object at its spot, not the
one letter under the arrow's tip.

**A label is what the reader sees, never what the mark is on** (live smoke
2c, E3). The check used to take the ``label`` as the thing's name when no
``target`` was given. A reviewer's label is usually a verdict or a question
("460.1?", "e = 0.444", "1.056 ok"), so on F38's red-line 10 of 18 boxes on
the right numbers came back misplaced; the agent rebuilt them with ``target``
and no labels, and the delivered red-line had no visible text. The label is
now shown to the look as display text to disregard, never compared with what
is under the mark (its PLACE is still measured: see the labels below).

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

**A mark in an area is judged by the area** (live smoke 2a, 2026-10-09). Asked
for "a note on the title block", an agent boxed the blank part of the title
strip and named it ``target: title block``; the look saw blank paper in the
box, took the comment ("Checked - live smoke test") for one about something
else, and the comment never became visible. What a mark names may be a THING
it goes round or an AREA it sits in (a title block, a table, a margin): the
look now says which, a mark inside the area it names is on it (measured where
the look gives the area's box), and a general remark or a stamp fits any mark.

**Labels are checked too, by measurement.** A label is placed by planlens,
not by the agent, and nothing looked at it: on two /Rotate 270 sheets every
label printed over the sheet title, 90-150 pt from its box, and the answer
told the user each sat "beside the box". A label more than
:data:`LABEL_MAX_GAP_PT` from its mark, running off the page, or printed over
the drawing's lettering (:data:`LABEL_MAX_LETTERING`) is reported.

**One frame.** Every box here — a row's ``bbox``, ``label_bbox`` and
``points_at``, the crop, the look's box converted back — is in planlens'
displayed frame (PDF points as the page is shown, ``/Rotate`` applied,
top-left origin): the frame ``Document.render`` crops in and PyMuPDF's
``get_pixmap(clip=...)`` renders in, so nothing is rotated here.
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
_WHERE_Q = ("where that thing is in this {w} x {h} pixel image (for an "
            "area, the part of it the image shows), as a box "
            "[x0, y0, x1, y1] in pixels (0,0 top-left), or null if it is not "
            "there")

#: A mark this many times the area of the thing it names is a blanket, not a
#: mark (the suite's own placement check uses 60).
MAX_AREA_FACTOR = 30.0

#: The smallest area a thing is taken to have, in pt² — a tag's lettering
#: box can be read very thin.
MIN_THING_AREA = 25.0

#: A label further than this from its mark (points, box to box) does not
#: read as that mark's — the live-smoke detector's threshold (H6 a).
LABEL_MAX_GAP_PT = 20.0

#: A label is over the drawing's lettering when, in any square of it the
#: label's height across, this share of the paper is ink — not counting
#: lines that run right across the label (a border it crosses is still
#: legible). Measured on the live-smoke sheets: labels printed over a title
#: or a note held 0.09-0.12 in their worst square; clear labels 0, a label
#: crossing a sheet border 0 (0.07 before the border was taken out).
LABEL_MAX_LETTERING = 0.04

#: The share of a row or column of the label's render that makes it a line
#: running right across the label, and the render's pixels per point.
LABEL_LINE_SHARE = 0.9
LABEL_PX_PER_PT = 2.0

#: A pixel darker than this (0-255 grey) is ink.
INK_LEVEL = 128


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


def _within(mark_bbox: Any, area: Optional[Sequence[float]],
            slack: float = ENCLOSE_SLACK_PT) -> Optional[bool]:
    """Whether the mark's centre lies inside an area's box, ``slack``
    points each way; ``None`` without a usable box of either."""
    return _encloses(area, mark_bbox, slack)


def _gap(a: Sequence[float], b: Sequence[float]) -> float:
    """The distance between two boxes, points (0 when they touch)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def _lettering_under(page, box: Sequence[float]) -> Optional[float]:
    """The worst share of ink under ``box`` (displayed frame) in any square
    of it the label's height across, on a render of the page WITHOUT its
    annotations (the label itself is one), lines that run right across the
    label left out; ``None`` when the box cannot be rendered."""
    import fitz
    import numpy as np

    x0, y0, x1, y1 = (float(v) for v in box)
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(LABEL_PX_PER_PT,
                                                 LABEL_PX_PER_PT),
                              clip=fitz.Rect(x0, y0, x1, y1),
                              colorspace=fitz.csGRAY, alpha=False,
                              annots=False)
    except Exception:  # noqa: BLE001 - an unrendered label is not judged
        return None
    if pix.width < 2 or pix.height < 2:
        return None
    ink = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.stride)[:, :pix.width] < INK_LEVEL
    ink = ink.copy()
    ink[ink.mean(axis=1) >= LABEL_LINE_SHARE, :] = False
    ink[:, ink.mean(axis=0) >= LABEL_LINE_SHARE] = False
    rows, cols = ink.shape
    side = min(rows, cols)
    step = max(1, side // 2)
    if cols >= rows:
        windows = [ink[:, i:i + side]
                   for i in range(0, max(1, cols - side + 1), step)]
    else:
        windows = [ink[i:i + side, :]
                   for i in range(0, max(1, rows - side + 1), step)]
    return max(float(w.mean()) for w in windows if w.size)


def _label_problem(page, row: Dict[str, Any]) -> Optional[str]:
    """What is wrong with a written row's visible label, or ``None``: too
    far from its mark, off the page, or over the drawing's lettering."""
    try:
        lb = [float(v) for v in row["label_bbox"]]
        mb = [float(v) for v in row["bbox"]]
    except (KeyError, TypeError, ValueError):
        return None
    if len(lb) != 4 or len(mb) != 4:
        return None
    gap = _gap(lb, mb)
    if gap > LABEL_MAX_GAP_PT:
        return (f"the label is {gap:.0f} pt from its mark, so it does not "
                f"read as that mark's")
    r = page.rect
    if lb[0] < r.x0 - 1 or lb[1] < r.y0 - 1 or lb[2] > r.x1 + 1 \
            or lb[3] > r.y1 + 1:
        return "the label runs off the page"
    share = _lettering_under(page, lb)
    if share is not None and share >= LABEL_MAX_LETTERING:
        return ("the label is printed over the drawing's own lettering, "
                "where neither can be read")
    return None


def check_labels(pdf_bytes: bytes, result: Dict[str, Any],
                 specs: Sequence[Any]) -> List[Dict[str, Any]]:
    """Every written row's visible label, measured on the marked copy: the
    ones too far from their mark, off the page or over the drawing's
    lettering, each with what is wrong. No vision call."""
    items = [it for it in _rows_with_specs(result, specs)
             if isinstance(it["row"], dict) and it["row"].get("label")
             and it["row"].get("label_bbox") is not None]
    if not items:
        return []
    import fitz
    out: List[Dict[str, Any]] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for it in items:
            row = it["row"]
            try:
                page = doc[int(row.get("page") or 0)]
            except (IndexError, ValueError, TypeError):
                continue
            problem = _label_problem(page, row)
            if problem:
                out.append({"index": it["index"],
                            "pdf_page": int(row.get("page") or 0) + 1,
                            "label": row.get("label"),
                            "label_bbox": row.get("label_bbox"),
                            "problem": problem})
    finally:
        doc.close()
    return out


def _label_reads_supported() -> bool:
    """Whether the installed planlens takes ``label_reads``."""
    try:
        from planlens.document.markup_writer import MarkupSpec
        return "label_reads" in MarkupSpec.fields_accepted()
    except Exception:  # noqa: BLE001 - an older planlens simply lacks it
        return False


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
    agent's ``target``, else the ``quote`` it was anchored on; ``("", "")``
    when the spec names no thing (the comment then says what it is about).
    Never the ``label``: that is the text the reader sees beside the mark,
    often a verdict ("460.1?"), and need not match anything printed (E3)."""
    for key in ("target", "quote"):
        value = _text(spec.get(key), 160)
        if value:
            return value, key
    return "", ""


def _expected(spec: Dict[str, Any]) -> str:
    """What the result says the mark was compared with."""
    target, _src = _target(spec)
    return target or _text(spec.get("comment"), 160) or \
        "(no target, quote or comment given)"


def _fold(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def _same_words(a: str, b: str) -> bool:
    return bool(_fold(a)) and _fold(a) == _fold(b)


_SOURCE_WORDS = {
    "target": "what the mark is on",
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
    label = _text(spec.get("label"), 80)
    label_note = " with a short red label beside it" if label else ""
    placed = "placed" if kind == "note" else "drawn"
    parts = [f"This is a crop of a page that a review tool has just marked "
             f"up. A {word} has been {placed} on it{label_note}. The question is "
             f"whether the mark is on the right thing."]
    if label:
        # E3: the label is display text, often a verdict ("460.1?"); taken
        # for the thing's name it failed 10 good boxes of 18 (F38).
        parts.append(f'The red label reads "{label}". It is the reviewer\'s '
                     f"own display text (a tag, a verdict or a question), NOT "
                     f"the name of what is marked: never judge the mark by "
                     f"whether the label matches what is there.")
    if target:
        parts.append(f"The mark is meant to {verb}: {target}   "
                     f"({_SOURCE_WORDS[source]})")
        parts.append("What is named may be a single thing (a tag, a line of "
                     "text, a cell, a symbol) or an AREA of the sheet (a "
                     "title block, a table, a margin, a drawing view). A "
                     "mark placed in the blank paper INSIDE a named area is "
                     "on that area: judge it by the area it sits in, not by "
                     "the empty paper under it.")
    if comment:
        parts.append(f'The review comment on this mark: "{comment}". A '
                     f"review comment is a remark or a request ABOUT the "
                     f"thing marked; it need not repeat the words printed "
                     f"there.")
    if not target and comment:
        parts.append(f"The mark is meant to {verb} the thing that comment "
                     f"is about.")
    elif not target:
        parts.append(f"Nothing names what the mark is meant to {verb}: "
                     f"judge whether it {verb}s one definite thing (a tag, a "
                     f"line of text, a symbol).")
    whole = (" Read the WHOLE line of text, or the whole object, there — "
             "not a single letter or stroke." if kind in POINTING_KINDS
             or kind == "highlight" else "")
    own = ("the icon itself" if kind == "note"
           else "the mark's own red label or comment box")
    parts.append(f"Look at what is actually {where} — not at {own}.{whole}")
    named = ("the thing named above" if target else
             "the thing the comment is about" if comment else
             "one definite thing")
    spot = where.split(" —")[0]
    keys = [("same_thing", f"true if what is there IS {named} (a "
                           f"paraphrase, a partial reading or a misprint of "
                           f"it counts; a different item that only shares a "
                           f"word or a number does not), else false")]
    if target:
        keys.append(("in_area", "true if what is named is an AREA (a title "
                                "block, a table, a margin, a view) and the "
                                "mark lies inside it, blank paper and all; "
                                "false if it is a single thing, or the mark "
                                "is outside it"))
    asked = ask_comment and bool(target)
    if asked:
        keys.append(("comment_fits", "true if the review comment could be "
                                     "about what is there or about the "
                                     "named thing or area — a general remark "
                                     "or a stamp ('checked', 'reviewed', "
                                     "'approved') fits any mark; false only "
                                     "if it is plainly about something else "
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
    in_area = _bool(got.get("in_area")) if target else None
    if got.get("thing_px") is not None:
        thing_raw, thing_size = got.get("thing_px"), size
    else:                       # an old-style answer on the 0-999 grid
        thing_raw, thing_size = got.get("thing_box"), None
    thing = _thing_on_page(thing_raw, clip, thing_size)

    def decided(ok: bool) -> str:
        if sure is False:
            return "unsure"
        return "confirmed" if ok else "misplaced"

    if fits is False:
        return decided(False), " — the comment is about something else"
    if in_area is True:
        # A mark in the blank paper of the area it names (a note on a title
        # block): on that area — measured against the area's box where the
        # look gave one, never against the paper under the mark.
        if thing is not None and _within(row.get("bbox"), thing) is False:
            return decided(False), " — outside the area it names"
        return decided(True), " — inside the area it names"
    if kind not in ENCLOSING_KINDS:
        ok = identity if identity is not None else encl_said
        return ("unsure", "") if ok is None else (decided(ok), "")
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
        from funhouse_agent.vision_tools import ask_vision
        answer = ask_vision(engine, image, prompt, "check")
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
    try:
        labels = check_labels(pdf_bytes, result, specs)
    except Exception as exc:  # noqa: BLE001 - the marks' verdicts stand
        labels = []
        block["labels_not_checked"] = f"{type(exc).__name__}: {exc}"[:160]
    if labels:
        block["labels"] = labels
    bad = len(by["misplaced"]) + len(by["unsure"])
    notes = []
    if not bad and not labels:
        notes.append("Every mark was looked at on the marked copy, and each "
                     "is on the thing it is meant to mark; every label sits "
                     "clear beside its mark.")
    if bad:
        notes.append(
            f"{bad} mark(s) are misplaced or could not be confirmed ('seen' "
            "says what is actually there). Do not hand the file over as it "
            "is: find the thing each comment is about again and write the "
            "copy again with append=false, leaving out any mark you cannot "
            "place. For a thing found by looking, zoom with render_region "
            "until it is legible in a view of 300 pt or less and take THAT "
            "zoom's view + the thing's image_box; for words, quote the words "
            "the comment is about. Every anchor — box, point, quote or note "
            "— is checked the same way, so switching anchor does not settle "
            "a verdict. Widening a mark until it takes the thing in does not "
            "place it: zoom until you can box the thing itself. Name what a "
            "mark is on with target when the comment is a request rather "
            "than the thing's name; a mark in the blank part of an area (a "
            "title block, a margin) names that area as its target. A label "
            "is only the text the reader sees and is never compared with "
            "what is under the mark: add target= and keep your labels.")
    if labels:
        along = (" Where the drawing's own lettering runs up or down the "
                 "page, give label_reads (up or down) so the label runs the "
                 "same way." if _label_reads_supported() else "")
        notes.append(
            f"{len(labels)} label(s) are not clear of the drawing or not "
            "beside their mark ('labels' says which and why). A label is "
            "what the reader sees on the sheet, so do not describe it as "
            "beside its mark: write the copy again with append=false and a "
            "shorter label, or no label (the comment still carries the "
            "words)." + along)
    block["note"] = " ".join(notes)
    return block


# ---------------------------------------------------------------------------
# A mark named by its words, and what a marked copy holds (live smoke 2c)
# ---------------------------------------------------------------------------

#: How planlens' annotate result opens. Bob's model (F47, E10) read it, with
#: ``n_written: 1``, as "the copy holds only this one note", and told the
#: user the five markups already on the document were missing. They were not.
_PLANLENS_LEAD = "a NEW file: the document you opened is unchanged."


def _copy_markups(pdf_bytes: bytes, signer: str) -> List[Dict[str, Any]]:
    """Every markup in a PDF, each as ``{"m", "label", "ours", "parent"}``:
    ``label`` when it is a mark's visible label (planlens ties a label to its
    mark with ``/IRT`` and ``/RT /Group``), ``parent`` that mark, ``ours``
    when this app wrote it (signed ``signer`` or as this app)."""
    import fitz
    from planlens.document.annotations import extract_annotations

    from funhouse_agent.document_tools import _is_label, _ours

    doc = fitz.open(stream=bytes(pdf_bytes), filetype="pdf")
    try:
        out: List[Dict[str, Any]] = []
        for i in range(doc.page_count):
            marks, _cad = extract_annotations(doc[i], i)
            by_id = {m.id: m for m in marks}
            for m in marks:
                label = m.xref is not None and _is_label(doc, m.xref)
                out.append({"m": m, "label": label,
                            "ours": _ours(m.author, signer),
                            "parent": (by_id.get(m.in_reply_to)
                                       if label and m.in_reply_to else None)})
        return out
    finally:
        doc.close()


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def name_marks_by_words(pdf_bytes: bytes, remove: Any, signer: str
                        ) -> Tuple[List[Any], List[Dict[str, str]]]:
    """``(remove, refused)`` for ``annotate_document(remove=...)`` on the
    marked copy ``pdf_bytes``: each entry of words that names exactly ONE
    mark this app wrote — in its comment OR in its visible label — becomes
    that mark's id, so the mark is removed with its label.

    Live smoke 2c (E9): F47's ``remove: ["Embedment: not shown"]`` was a
    label's text; the remover matched comments only, said "no mark this app
    wrote says that", and removing by id took another list-and-remove round.

    Words naming no mark, or several, are taken out of the list and
    returned in ``refused`` (rows shaped like the remover's ``not_removed``)
    — never handed on, since the remover, matching comments only, would
    take the one mark whose COMMENT says them although another's label
    does too. Ids, and entries without words, are left for the remover."""
    from funhouse_agent.document_tools import _MARKUP_ID

    entries = ([remove] if isinstance(remove, (str, dict))
               else list(remove or []))
    rows = _copy_markups(pdf_bytes, signer)
    out: List[Any] = []
    refused: List[Dict[str, str]] = []
    for raw in entries:
        if isinstance(raw, dict):
            want = "" if raw.get("id") else str(raw.get("text") or "")
        else:
            want = str(raw or "")
        want = want.strip()
        if not want or _MARKUP_ID.match(want):
            out.append(raw)
            continue
        low = want.lower()
        hits: Dict[Tuple[int, Any], Any] = {}
        for r in rows:
            m = r["m"]
            if not r["ours"] or low not in (m.text or "").lower():
                continue
            if not r["label"]:
                hits[(m.page, m.xref)] = m
            elif r["parent"] is not None:
                p = r["parent"]
                hits[(p.page, p.xref)] = p
        if len(hits) == 1:
            out.append(next(iter(hits.values())).id)
        elif hits:
            ids = ", ".join(m.id for m in list(hits.values())[:8])
            refused.append({"remove": want, "reason": (
                f"{len(hits)} marks say that in their comment or label "
                f"({ids}): give the id of the one meant")})
        else:
            refused.append({"remove": want, "reason": (
                "no mark this app wrote says that, in its comment or its "
                "label")})
    return out, refused


def describe_copy(result: Dict[str, Any], signer: str) -> None:
    """Say in ``result`` what the marked copy at ``result["output_path"]``
    holds, counted off the file (live smoke 2c, E10): ``in_file`` — its
    markups (a mark's visible label is part of the mark, not counted
    apart), the ones this call added, the ones kept from before and who
    wrote those, and any this call removed — and a ``note`` that opens by
    saying so in words. Raises when the file cannot be read; the caller
    leaves the result as it was."""
    import os

    path = str(result.get("output_path") or "")
    with open(path, "rb") as fh:
        data = fh.read()
    marks = [r for r in _copy_markups(data, signer) if not r["label"]]
    added = int(result.get("n_written") or 0)
    total = len(marks)
    kept = max(0, total - added)
    by: Dict[str, int] = {}
    ours = 0
    for r in marks:
        if r["ours"]:
            ours += 1
        else:
            who = str(r["m"].author or "").strip() or "unsigned"
            by[who] = by.get(who, 0) + 1
    if ours > added:
        by["this app, earlier calls"] = ours - added
    removed = len(result.get("removed") or [])
    block: Dict[str, Any] = {"markups": total, "added_by_this_call": added,
                             "kept_from_before": kept}
    if by:
        block["kept_by_author"] = by
    if removed:
        block["removed_by_this_call"] = removed
    result["in_file"] = block

    said = (f"'{os.path.basename(path)}' holds {_plural(total, 'markup')}: "
            f"{added} added by this call and {kept} kept from before")
    if by:
        said += " (" + ", ".join(f"{who} {n}" for who, n in by.items()) + ")"
    if removed:
        said += f"; {removed} removed by this call"
    said += (". Every markup already on the document is in the copy (remove "
             "takes out only this app's marks); the document you opened is "
             "unchanged.")
    note = str(result.get("note") or "").strip()
    if note.startswith(_PLANLENS_LEAD):
        note = note[len(_PLANLENS_LEAD):].strip()
    result["note"] = said + (" " + note if note else "")


__all__ = ["CHECK_KINDS", "GEOMETRY_ANCHORS", "QUOTE_ANCHORS",
           "CHECKED_ANCHORS", "check_labels", "check_marks",
           "describe_copy", "marks_to_check", "name_marks_by_words"]
