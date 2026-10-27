"""Pages whose text runs turned: their reading order, rows and pictures.

A report scanned with its sheets fed sideways holds pages whose picture,
and so whose text, runs a quarter turn from the page: a landscape grading
chart scanned onto a portrait page, its lines reading bottom to top. The
page itself is not rotated (its ``/Rotate`` is 0); the optical reader says
the content is skewed about 90 degrees and gives every line its own
direction (planlens' ``TextLine.rotation``: degrees counter-clockwise AS
DISPLAYED, 90 reading bottom to top).

Every reader that walks such a page by its displayed frame reads it wrong
(BACKLOG RI-149, RI-150). Sorted top to bottom and left to right, the lines
of a turned page come in the order in which each one STARTS along the
turned direction, which follows its length and not the row it is printed
on: a legend's four rows, each a sample's label, its description and its
percentages, came out with the percentages in another order than the
labels, and the reader gave each sample another sample's values. The top
band of the displayed page is a side of the turned sheet, so its title is
not where a reader looks for it. And the picture is sent sideways.

This module reads a page's turn from its lines (:func:`page_turn`) and
gives the lines in the turned page's own frame: its reading order (rows
from the turned page's top, each row in the direction its text reads,
:func:`reading_order`), its printed rows (:func:`rows`), its top band
(:func:`top_band`), and the quarter turn that shows its picture upright
(:func:`upright_png`). Every box a record keeps stays in the displayed
frame: only the order and the grouping change. A reader that shows a model
the upright picture lists the page's boxes in that picture's frame
(:func:`upright_box`, :func:`upright_size`), so a box read off the picture
and a box copied from the listing are in one frame, and takes every box the
model gives back to the displayed frame (:func:`displayed_box`) before
anything uses it.

THE ROWS ARE DE-SKEWED. A scan is never fed quite square: on the sheet that
raised RI-149 the content runs 89 degrees, not 90, and across a 400 pt row
that drifts the row's far end 6 pt across -- half a row pitch, so the
percentages at the end of a legend row sat nearer the NEXT row's label than
their own in the displayed frame. The rows here are taken across the text's
own direction, the length-weighted mean of the turned lines' rotations, so
a row's far end stays on its row.

An upright page (no turn) keeps the displayed order every reader used
before; :func:`page_turn` calls a page turned only when most of its text,
by characters, runs within :data:`QUARTER_TOL` of one quarter turn, and
clearly more of it than runs upright (:data:`TURN_MARGIN`).

LAYOUT JUDGMENT: :data:`QUARTER_TOL`, :data:`TURNED_SHARE` and
:data:`TURN_MARGIN` are set from the corpus's scans (lines on a turned sheet
read 85-95 degrees; selection marks and stray glyphs read 0; over the
corpus's 7,500 pages every page with at least 0.6 of its letters turned
carries at least twice as many turned letters as upright ones, while a form
printing an upright header over a turned chart, and a map whose street
names run turned, carry 1.0 to 1.35 times as many), and :data:`ROW_TOL`
from its legends (a row pitch of about 12 pt, a line about 10 pt thick);
none is from a standard.
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

__all__ = ["Turn", "TurnVote", "page_turn", "turn_vote", "turn_of",
           "reading_order", "rows", "top_band", "upright_png", "upright_box",
           "displayed_box", "upright_size", "line_thickness", "runs_with",
           "QUARTER_TOL", "TURNED_SHARE", "TURN_MARGIN", "ROW_TOL"]

#: A line within this many degrees of a quarter turn reads that way.
QUARTER_TOL = 15.0
#: The share of a page's text, by characters, that must run one quarter
#: turn (90, 180 or 270) for the page to be read in that turn's frame.
TURNED_SHARE = 0.5
#: How many times the upright lines' characters the turned ones must be: a
#: form with an upright header over a landscape chart splits its letters
#: about evenly between the two (1.0 to 1.1 times), a sheet scanned
#: sideways does not (its upright letters are stray marks and stamps).
TURN_MARGIN = 1.5
#: Two lines are on one printed row when their centres, taken across the
#: text's own direction, lie within this share of the page's median line
#: thickness of each other.
ROW_TOL = 0.5


@dataclass(frozen=True)
class Turn:
    """How a page's text runs: ``quarter`` (0, 90, 180 or 270 degrees
    counter-clockwise as displayed) and ``angle``, the text's own direction
    (the quarter plus the scan's skew), with the displayed page's size."""

    quarter: int = 0
    angle: float = 0.0
    width: float = 612.0
    height: float = 792.0

    @property
    def turned(self) -> bool:
        return self.quarter != 0

    def _axes(self) -> Tuple[float, float, float, float]:
        """``(dx, dy, nx, ny)``: the reading direction and the direction
        across it, toward the following rows, in displayed coordinates
        (y down)."""
        theta = math.radians(self.angle)
        return (math.cos(theta), -math.sin(theta),
                math.sin(theta), math.cos(theta))

    def frame(self, bbox: Sequence[float]) -> Tuple[float, float, float, float]:
        """A displayed box as ``(u0, v0, u1, v1)`` in the text's own frame:
        u along the reading direction, v across it toward the following
        rows (the turned page's x and y, up to an offset)."""
        x0, y0, x1, y1 = (float(v) for v in bbox)
        dx, dy, nx, ny = self._axes()
        us = [x * dx + y * dy for x in (x0, x1) for y in (y0, y1)]
        vs = [x * nx + y * ny for x in (x0, x1) for y in (y0, y1)]
        return min(us), min(vs), max(us), max(vs)

    def centre_v(self, bbox: Sequence[float]) -> float:
        """Where a box's centre lies across the text's own direction."""
        x0, y0, x1, y1 = (float(v) for v in bbox)
        _dx, _dy, nx, ny = self._axes()
        return ((x0 + x1) / 2.0) * nx + ((y0 + y1) / 2.0) * ny


def _quarter(rotation: Any) -> Optional[Tuple[int, float]]:
    """``(quarter, offset)`` for a line's rotation within
    :data:`QUARTER_TOL` of a quarter turn, else None."""
    if rotation is None:
        return None
    try:
        r = float(rotation) % 360.0
    except (TypeError, ValueError):
        return None
    q = int(round(r / 90.0)) % 4 * 90
    off = (r - q + 180.0) % 360.0 - 180.0
    if abs(off) > QUARTER_TOL:
        return None
    return q, off


#: A tick box an optical reader reports as a line of its own ("[ ]",
#: "[x]"): it has no direction, and a grading chart's grid read as tick
#: boxes would outvote its words.
_SELECTION_MARK = re.compile(r"^\s*\[\s*[xX]?\s*\]\s*$")


def _letters(line: Any) -> int:
    """The letters and digits a line carries, for the vote on its page's
    turn: none for a tick box."""
    text = str(getattr(line, "text", "") or "")
    if _SELECTION_MARK.match(text):
        return 0
    return sum(1 for ch in text if ch.isalnum())


@dataclass(frozen=True)
class TurnVote:
    """How a page's letters vote on its turn: the letters and digits on
    lines running each quarter turn, the page's total (lines of unknown
    direction counted in the total only), and each quarter's
    length-weighted skew."""

    votes: Tuple[Tuple[int, float], ...]
    total: float
    skew: Tuple[Tuple[int, float], ...]

    def of(self, quarter: int) -> float:
        return dict(self.votes).get(quarter, 0.0)

    @property
    def best(self) -> int:
        """The turned quarter with the most letters (90, 180 or 270)."""
        return max((90, 180, 270), key=self.of)

    @property
    def share(self) -> float:
        """:attr:`best`'s share of the page's letters."""
        return self.of(self.best) / self.total if self.total > 0 else 0.0

    @property
    def margin(self) -> float:
        """:attr:`best`'s letters over the upright lines' letters
        (infinite where no line runs upright)."""
        up = self.of(0)
        return self.of(self.best) / up if up > 0 else math.inf


def turn_vote(lines: Sequence[Any]) -> TurnVote:
    """The vote :func:`page_turn` decides by. A line whose direction is
    unknown (None) does not vote, and neither does a tick box ("[ ]",
    "[x]")."""
    votes = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    skew = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    weight = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    total = 0.0
    for line in lines or ():
        n = float(_letters(line))
        if n <= 0:
            continue
        total += n
        got = _quarter(getattr(line, "rotation", 0.0))
        if got is None:
            continue
        q, off = got
        votes[q] += n
        # The skew is weighted by the square of the line's length: an
        # optical reader measures a long line's direction far better than a
        # short one's (a two-digit axis number reads 88 or 91 degrees on a
        # sheet whose long lines all read 88.4), and it is over the long
        # lines' length that a wrong angle drifts a row onto the next.
        length = _length(line, q)
        skew[q] += length * length * off
        weight[q] += length * length
    mean = {q: (skew[q] / weight[q] if weight[q] > 0 else 0.0) for q in skew}
    return TurnVote(tuple(sorted(votes.items())), total,
                    tuple(sorted(mean.items())))


def page_turn(lines: Sequence[Any], width: float = 612.0,
              height: float = 792.0) -> Turn:
    """The way a page's text runs, read from its lines.

    A quarter turn wins when its lines carry at least
    :data:`TURNED_SHARE` of the page's letters and digits AND at least
    :data:`TURN_MARGIN` times the letters of the upright lines; else the
    page is upright (quarter 0). The margin is what tells a sheet scanned
    sideways from a form with an upright header over a landscape chart,
    whose two halves split the vote about evenly. The angle is the
    length-weighted mean of the winning lines' rotations: the text's own
    direction, skew included."""
    vote = turn_vote(lines)
    if vote.total <= 0:
        return Turn(0, 0.0, float(width), float(height))
    best = vote.best
    if vote.share < TURNED_SHARE or vote.margin < TURN_MARGIN:
        return Turn(0, 0.0, float(width), float(height))
    return Turn(best, best + dict(vote.skew)[best], float(width),
                float(height))


def _length(line: Any, quarter: int) -> float:
    """A line's length along its own direction, from its displayed box."""
    try:
        x0, y0, x1, y1 = (float(v) for v in line.bbox)
    except (AttributeError, TypeError, ValueError):
        return 1.0
    return max(1.0, abs(y1 - y0) if quarter in (90, 270) else abs(x1 - x0))


def turn_of(doc: Any, page: int) -> Turn:
    """:func:`page_turn` for one page of an open document; upright for a
    page that will not read."""
    try:
        content = doc.page(page)
        return page_turn(list(content.lines),
                         float(getattr(content, "width", 612.0) or 612.0),
                         float(getattr(content, "height", 792.0) or 792.0))
    except Exception:                         # a page that will not read
        return Turn()


def runs_with(line: Any, turn: Turn) -> bool:
    """Whether a line of words runs the page's way (its rotation within
    :data:`QUARTER_TOL` of the page's quarter turn)."""
    got = _quarter(getattr(line, "rotation", 0.0))
    return got is not None and got[0] == turn.quarter


_runs_with = runs_with


def line_thickness(lines: Sequence[Any], turn: Turn) -> float:
    """The median thickness, across the text's direction, of the lines of
    words that run the page's way (tick boxes, stray glyphs and lines
    running another way left out): the scale :data:`ROW_TOL` is a share
    of."""
    return _thickness(lines, turn)


def _thickness(lines: Sequence[Any], turn: Turn) -> float:
    """:func:`line_thickness`."""
    words = [ln for ln in lines if _letters(ln) >= 2 and _runs_with(ln, turn)]
    sizes = sorted(f[3] - f[1] for f in (turn.frame(ln.bbox)
                                         for ln in words or lines)
                   if f[3] > f[1])
    if not sizes:
        return 10.0
    return max(1.0, sizes[len(sizes) // 2])


def rows(lines: Sequence[Any], turn: Turn) -> List[List[Any]]:
    """The page's printed rows in the text's own frame: each a list of
    lines in reading order, the rows from the turned page's top.

    A line joins the row before it when its centre, across the text's
    direction, lies within :data:`ROW_TOL` of the median line thickness of
    that row's mean."""
    lines = [ln for ln in lines or () if getattr(ln, "bbox", None)]
    if not lines:
        return []
    tol = ROW_TOL * _thickness(lines, turn)
    ordered = sorted(lines, key=lambda ln: (turn.centre_v(ln.bbox),
                                            turn.frame(ln.bbox)[0]))
    out: List[List[Any]] = []
    centre: List[float] = []
    for line in ordered:
        v = turn.centre_v(line.bbox)
        if out and abs(v - centre[-1]) <= tol:
            out[-1].append(line)
            n = len(out[-1])
            centre[-1] += (v - centre[-1]) / n
            continue
        out.append([line])
        centre.append(v)
    return [sorted(row, key=lambda ln: turn.frame(ln.bbox)[0]) for row in out]


def reading_order(lines: Sequence[Any], turn: Turn) -> List[Any]:
    """The lines in the order the page reads: on an upright page the
    displayed order every reader has used (top to bottom, then left to
    right); on a turned page its rows from the turned page's top, each in
    the direction its text reads."""
    if not turn.turned:
        return sorted(lines, key=lambda ln: (round(ln.bbox[1], 1),
                                             ln.bbox[0]))
    return [line for row in rows(lines, turn) for line in row]


def top_band(lines: Sequence[Any], turn: Turn, frac: float = 0.22
             ) -> List[Any]:
    """The lines in the top ``frac`` of the page as it reads, in reading
    order: on an upright page those whose top is within ``frac`` of the
    displayed height (the floor's title band); on a turned page those
    within ``frac`` of the turned page's own height."""
    if not turn.turned:
        return [ln for ln in reading_order(lines, turn)
                if ln.bbox[1] <= frac * turn.height]
    corners = turn.frame((0.0, 0.0, turn.width, turn.height))
    limit = corners[1] + frac * (corners[3] - corners[1])
    return [ln for ln in reading_order(lines, turn)
            if turn.frame(ln.bbox)[1] <= limit]


def upright_box(bbox: Sequence[float], turn: Turn
                ) -> Tuple[float, float, float, float]:
    """A displayed box in the upright picture's frame (points, top-left
    origin) that :func:`upright_png` turns the page into: the quarter turn
    only, no skew. :func:`displayed_box` takes it back."""
    x0, y0, x1, y1 = (float(v) for v in bbox)
    w, h = turn.width, turn.height
    if turn.quarter == 90:
        return (h - y1, x0, h - y0, x1)
    if turn.quarter == 270:
        return (y0, w - x1, y1, w - x0)
    if turn.quarter == 180:
        return (w - x1, h - y1, w - x0, h - y0)
    return (x0, y0, x1, y1)


def displayed_box(bbox: Sequence[float], turn: Turn
                  ) -> Tuple[float, float, float, float]:
    """A box in the upright picture's frame (:func:`upright_box`) back in
    the page's frame as displayed, the frame every record keeps. A box read
    off an upright picture, or copied from a listing given in its frame,
    comes back through here before anything uses it."""
    u0, v0, u1, v1 = (float(v) for v in bbox)
    u0, u1 = sorted((u0, u1))
    v0, v1 = sorted((v0, v1))
    w, h = turn.width, turn.height
    if turn.quarter == 90:
        return (v0, h - u1, v1, h - u0)
    if turn.quarter == 270:
        return (w - v1, u0, w - v0, u1)
    if turn.quarter == 180:
        return (w - u1, h - v1, w - u0, h - v0)
    return (u0, v0, u1, v1)


def upright_size(turn: Turn) -> Tuple[float, float]:
    """The upright picture's width and height in points: the displayed
    page's, swapped by a quarter turn."""
    if turn.quarter in (90, 270):
        return turn.height, turn.width
    return turn.width, turn.height


def upright_png(png: bytes, turn: Turn) -> bytes:
    """A rendered page or crop turned by the quarter that sets its text
    upright (text reading bottom to top is turned a quarter clockwise), or
    the image unchanged when the page is upright or the picture will not
    turn."""
    if not turn.turned or not png:
        return png
    try:
        from PIL import Image
    except Exception:                         # no PIL: send it as it is
        return png
    try:
        image = Image.open(io.BytesIO(png))
        fmt = image.format or "PNG"
        # PIL's ROTATE_n turns counter-clockwise by n: text turned a
        # quarter counter-clockwise (90) is set upright by a quarter
        # clockwise (ROTATE_270), and so on.
        op = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
              270: Image.Transpose.ROTATE_90}[turn.quarter]
        out = io.BytesIO()
        image.transpose(op).save(out, format=fmt)
        return out.getvalue()
    except Exception:                         # a picture that will not turn
        return png
