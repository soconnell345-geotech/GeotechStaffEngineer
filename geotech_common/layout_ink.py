"""The layout check: lettering a reader cannot read cleanly, judged by the
pixels a figure draws as it is saved (BACKLOG GM-108, moved here for GM-109).

Shared by the plotting box (:mod:`funhouse_agent.plot_box.child`, which
loads this file by its path so the rest of ``geotech_common`` stays out of
the box) and ``profile_figure.data_plot`` (the notes of ``plot_data``).

The check swaps ``draw`` on matplotlib's base classes while it saves the
figure into memory. That swap is process-wide, so one check runs at a time
(:data:`_LOCK`), and a draw on any other thread passes straight through to
the original: a figure another thread saves meanwhile is drawn as usual.

Stdlib only at the top: matplotlib and numpy are imported where used.
"""

from __future__ import annotations

import io
import threading
import time

#: The layout check judges lettering by the pixels drawn (live wave w9: 22
#: of its 36 reports were false while it judged by boxes -- rotated text by
#: its upright box, legend text through the legend's opaque frame, text
#: through its own white box -- and lettering outside the axes was never
#: tested). The measures are the w9 trace review's own (§9): a line runs
#: through lettering when at least LINE_MIN_PX of its pixels show on the
#: lettering's glyphs, grown by one pixel, in the finished picture (at 200
#: dpi; scaled with the dpi); a pixel shows when the picture with the line
#: differs from the picture without it by more than VISIBLE_DIFF over the
#: three colour channels (of 255 each).
LINE_MIN_PX = 3
VISIBLE_DIFF = 60
#: A glyph pixel: drawn at an alpha above this (of 255).
INK_ALPHA = 40
#: Lettering closer than this on one line reads as one word ("200" against
#: "0" reads "2000"): a quarter of the smaller lettering's size (letters in
#: a word sit about 0.13 em apart, words about 0.45 em), and never under
#: TOUCH_PT (2 px at 100 dpi).
TOUCH_EM = 0.25
TOUCH_PT = 1.44
#: A box under lettering (its own bbox, a legend's frame) at least this
#: opaque is the lettering's footprint.
BOX_OPAQUE = 0.5
#: The check gives up ("unchecked") rather than slow the save past this
#: many seconds per figure, or hold more than this many MB of drawing kept
#: on the lettering (the box has 2 GB): thousands of texts, or a thousand
#: lines under a large opaque legend.
LAYOUT_SECONDS = 3.0
LAYOUT_MB = 256

#: One check at a time in a process: the check swaps ``draw`` on
#: matplotlib's base classes from the start of its save to the end.
_LOCK = threading.Lock()


class _Unchecked(Exception):
    """The layout is too large to check; the message says why."""


def layout_check(fig, fname: str, dpi=200, *, texts=None, seconds=None,
                 mb=None) -> dict:
    """Lettering the reader cannot read cleanly, judged by the pixels drawn
    in the figure as it is saved at ``dpi`` (not from a look):

    - ``text_overlaps``: lettering whose ink overlaps other lettering, or
      runs into it on one line (closer than :data:`TOUCH_PT`: "200" against
      "0" reads "2000"). Lettering on its own opaque box (a bbox, a legend's
      frame) has the box as its footprint; an opaque legend is one footprint,
      "the legend".
    - ``text_on_a_line``: lettering that a line, a shape's edge, markers, a
      tick mark or an axes frame shows through, or that a filled shape is
      drawn over. A line hidden by a box drawn over it (the lettering's own
      box, a legend's frame) is not through the lettering.
    - ``text_cut_off``: lettering the figure edge or its axes' edge cuts.

    Every drawn text counts, inside the axes or out (titles, axis and tick
    labels, legend entries, figure text); tick labels outside the view are
    never drawn, and an annotation is its text: its leader arrow is a line
    like any other (it runs through other lettering, never its own), as is
    an arrow drawn alone. Up to eight of each, each ``[words, other]``; a
    look decides.

    ``texts``, a list of text artists, keeps only the findings on those
    (all of them, in the order given): ``text_overlaps`` ``[k, words,
    artist]`` (the other lettering, which may be another of ``texts``),
    ``text_on_a_line`` ``[k, what, artist]`` (what shows through, and its
    artist: a note's leader is its note) and ``text_cut_off`` ``[k,
    where]``, ``k`` the index in ``texts``.

    A figure too large to check within ``seconds`` (default
    :data:`LAYOUT_SECONDS`) or ``mb`` (default :data:`LAYOUT_MB`) is
    ``unchecked``."""
    out = {"file": fname}
    try:
        found = _ink_check(fig, int(dpi), seconds=seconds, mb=mb)
    except _Unchecked as exc:
        out["unchecked"] = str(exc)
        return out
    except Exception as exc:                               # noqa: BLE001
        out["unchecked"] = (f"the layout could not be measured "
                            f"({type(exc).__name__}: {exc})")[:200]
        return out
    if texts is None:
        for key in ("text_overlaps", "text_on_a_line", "text_cut_off"):
            if found.get(key):
                out[key] = found[key][:8]
        return out
    where = {id(a): k for k, a in enumerate(texts)}
    raw = found.get("raw") or {}
    pairs = []
    for a, wa, b, wb in raw.get("text_overlaps", []):
        if id(a) in where:
            pairs.append([where[id(a)], wb, b])
        if id(b) in where:
            pairs.append([where[id(b)], wa, a])
    lines = [[where[id(a)], what, by]
             for a, _w, what, by in raw.get("text_on_a_line", [])
             if id(a) in where]
    cut = [[where[id(a)], at] for a, _w, at in raw.get("text_cut_off", [])
           if id(a) in where]
    for key, rows in (("text_overlaps", pairs), ("text_on_a_line", lines),
                      ("text_cut_off", cut)):
        if rows:
            out[key] = sorted(rows, key=lambda r: r[0])
    return out


def _leaf_kinds() -> list:
    """The classes whose ``draw`` draws every leaf of a figure: lettering,
    lines, shapes, arrows (their draw is their own), collections (meshes
    draw on their own) and images."""
    import matplotlib.collections as mcoll
    import matplotlib.image as mimage
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Patch
    from matplotlib.text import Text
    kinds = [Text, Line2D, Patch, mcoll.Collection]
    if "draw" in vars(FancyArrowPatch):
        kinds.append(FancyArrowPatch)
    # pcolormesh, a colorbar's fill, tripcolor: their draw is their own
    for name in ("QuadMesh", "TriMesh"):
        k = getattr(mcoll, name, None)
        if k is not None and "draw" in vars(k):
            kinds.append(k)
    base = getattr(mimage, "_ImageBase", None)
    if base is not None and "draw" in vars(base):
        kinds.append(base)
    return kinds


def _ink_check(fig, dpi: int, *, seconds=None, mb=None) -> dict:
    """Save ``fig`` into memory as the plotting box's ``save_figure`` saves
    it (``bbox_inches="tight"``, so lettering outside the axes and the
    canvas is in the frame), at ``dpi``, noting each leaf artist as it is
    drawn; after the last one, measure them in that frame (:class:`_Ink`).
    One check at a time (:data:`_LOCK`); a draw on another thread meanwhile
    is the original draw."""
    with _LOCK:
        return _ink_check_locked(fig, dpi, seconds, mb)


def _ink_check_locked(fig, dpi: int, seconds, mb) -> dict:
    from matplotlib.artist import Artist
    from matplotlib.text import Text
    seconds = LAYOUT_SECONDS if seconds is None else float(seconds)
    kinds = _leaf_kinds()
    originals = {k: vars(k)["draw"] for k in kinds}
    state = {"drawn": [], "placed": {}, "off": None, "depth": 0,
             "found": None, "seconds": seconds,
             "mb": LAYOUT_MB if mb is None else float(mb),
             "deadline": time.monotonic() + seconds}
    checking = threading.get_ident()

    def wrap(orig):
        def draw(self, renderer, *args, **kwargs):
            # another thread's figure is drawn as usual; a leaf's own parts
            # (a text's box) are the leaf; the second buffer's draws are
            # the check's own
            if (threading.get_ident() != checking or state["depth"]
                    or renderer is state["off"]):
                return orig(self, renderer, *args, **kwargs)
            state["drawn"].append((self, orig))
            if isinstance(self, Text):
                # where it is drawn: 3D text is placed only while it draws
                state["placed"][id(self)] = (self, self._x, self._y,
                                             self._rotation)
            if "draw_path" not in vars(renderer):
                # the saving pass: the leaf is drawn alone later, and this
                # picture is thrown away (the containers still lay out)
                return None
            state["depth"] += 1
            try:
                return orig(self, renderer, *args, **kwargs)
            finally:
                state["depth"] -= 1
        draw._supports_rasterization = getattr(
            orig, "_supports_rasterization", False)
        return draw

    class _Measure(Artist):
        """Drawn last (the highest z-order), when the frame is the saved
        one and every leaf has been drawn in place."""

        def draw(self, renderer):
            drawn, state["drawn"] = state["drawn"], []
            placed, state["placed"] = state["placed"], {}
            # matplotlib's sizing pass for bbox_inches="tight" draws nothing
            if "draw_path" in vars(renderer):
                return
            moved = []
            try:
                for a, x, y, rot in placed.values():
                    now = (a._x, a._y, a._rotation)
                    if any(u is not v for u, v in zip(now, (x, y, rot))):
                        moved.append((a,) + now)
                        a._x, a._y, a._rotation = x, y, rot
                state["found"] = _Ink(fig, renderer, drawn, state).found()
            finally:
                for a, x, y, rot in moved:
                    a._x, a._y, a._rotation = x, y, rot

    probe = _Measure()
    probe.set_zorder(1e12)
    probe.set_in_layout(False)
    for k, orig in originals.items():
        setattr(k, "draw", wrap(orig))
    try:
        fig.add_artist(probe)
        fig.savefig(io.BytesIO(), format="rgba", dpi=dpi,
                    bbox_inches="tight")
    finally:
        for k, orig in originals.items():
            setattr(k, "draw", orig)
        try:
            probe.remove()
        except Exception:                                  # noqa: BLE001
            pass
    return state["found"] or {}


def _grow(mask, k: int):
    """``mask`` grown by ``k`` pixels each way (a square)."""
    if k <= 0:
        return mask
    rows = mask.copy()
    for d in range(1, k + 1):
        rows[d:] |= mask[:-d]
        rows[:-d] |= mask[d:]
    out = rows.copy()
    for d in range(1, k + 1):
        out[:, d:] |= rows[:, :-d]
        out[:, :-d] |= rows[:, d:]
    return out


def _meet(a, b, grow: int = 0) -> bool:
    """Whether two masks placed in the frame (``(row, col, mask)``) share a
    pixel, the first grown by ``grow`` pixels."""
    import numpy as np
    ra, ca, ma = a
    rb, cb, mb = b
    if grow:
        ma = _grow(np.pad(ma, grow), grow)
        ra, ca = ra - grow, ca - grow
    r0, r1 = max(ra, rb), min(ra + ma.shape[0], rb + mb.shape[0])
    c0, c1 = max(ca, cb), min(ca + ma.shape[1], cb + mb.shape[1])
    if r1 <= r0 or c1 <= c0:
        return False
    return bool((ma[r0 - ra:r1 - ra, c0 - ca:c1 - ca]
                 & mb[r0 - rb:r1 - rb, c0 - cb:c1 - cb]).any())


def _ink_rows(mask, row0: int):
    """The first and last frame rows of a mask's ink, or None."""
    import numpy as np
    rows = np.nonzero(mask.any(axis=1))[0]
    return (row0 + int(rows[0]), row0 + int(rows[-1])) if len(rows) else None


def _figure_roles(fig) -> dict:
    """What some leaves are, by ``id``: grid lines, tick marks, the panels'
    and the figure's own backgrounds, the axes' frame lines (by side), each
    legend's frame, entries and symbols, and each annotation's leader arrow
    (to its annotation)."""
    from matplotlib.figure import FigureBase
    from matplotlib.legend import Legend
    from matplotlib.text import Annotation
    roles = {"grid": set(), "ticks": set(), "panels": set(), "spines": {},
             "frames": {}, "in_legend": {}, "symbols": set(), "leaders": {}}
    for t in fig.findobj(lambda a: isinstance(a, Annotation)):
        if t.arrow_patch is not None:
            roles["leaders"][id(t.arrow_patch)] = t
    for f in fig.findobj(lambda a: isinstance(a, FigureBase)):
        roles["panels"].add(id(f.patch))
    for ax in fig.get_axes():
        roles["panels"].add(id(ax.patch))
        for side, sp in ax.spines.items():
            roles["spines"][id(sp)] = side
        for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None),
                     getattr(ax, "zaxis", None)):
            if axis is None:
                continue
            if getattr(axis, "gridlines", None) is not None:
                roles["grid"].add(id(axis.gridlines))     # a 3D axis's grid
            for tick in axis.majorTicks + axis.minorTicks:
                roles["grid"].add(id(tick.gridline))
                roles["ticks"].update((id(tick.tick1line),
                                       id(tick.tick2line)))
    for leg in fig.findobj(lambda a: isinstance(a, Legend)):
        frame = leg.get_frame()
        roles["frames"][id(frame)] = leg
        for a in leg.findobj():
            if a is leg or a is frame:
                continue
            roles["in_legend"][id(a)] = leg
            if not hasattr(a, "get_text"):
                roles["symbols"].add(id(a))
    return roles


class _Ink:
    """The lettering of one saved figure, measured by what each artist
    draws.

    Each leaf artist is drawn again ALONE into a second buffer of the saved
    frame, in the order the figure drew it (a shape's fill and its edge
    apart; a text's glyphs apart from its box), and kept where it falls on
    a piece of lettering. Per piece of lettering the leaves are then laid
    over one another in that order, as the picture was made, so what shows
    is known: a line under an opaque box shows nowhere; a line under a box
    85 % white still shows."""

    def __init__(self, fig, renderer, drawn, state):
        import numpy as np
        if time.monotonic() > state["deadline"]:
            raise _Unchecked(f"too many texts to check in time "
                             f"({state['seconds']:g} s per figure); a look "
                             f"decides")
        from matplotlib.backends.backend_agg import RendererAgg
        self.np = np
        self.W, self.H = int(renderer.width), int(renderer.height)
        self.dpi = float(renderer.dpi)
        self.off = RendererAgg(self.W, self.H, self.dpi)
        state["off"] = self.off
        self.buf = np.asarray(self.off.buffer_rgba())
        self.gap = max(1, int(round(TOUCH_PT * self.dpi / 72.0)))
        self.min_px = max(2, int(round(LINE_MIN_PX * self.dpi / 200.0)))
        self.deadline = state["deadline"]
        self.seconds, self.mb = state["seconds"], state["mb"]
        self.kept = 0           # bytes of drawing kept on the lettering
        self.roles = _figure_roles(fig)
        self.leaves = [(a, o) for a, o in drawn if a.get_visible()]
        self._find_lettering(renderer)
        self.boxes = np.array(self.regions, dtype=float).reshape(-1, 4)

    def _in_time(self):
        if time.monotonic() > self.deadline:
            raise _Unchecked(
                f"too many texts to check in time ({len(self.names)} "
                f"texts, {self.seconds:g} s per figure); a look decides")

    # -- where the lettering is -------------------------------------------

    def _box(self, bb, pad: int):
        """A frame box ``(r0, r1, c0, c1)`` round a display bbox, or None
        when it falls outside the frame."""
        np = self.np
        c0 = max(0, int(np.floor(bb.x0)) - pad)
        c1 = min(self.W, int(np.ceil(bb.x1)) + pad)
        r0 = max(0, self.H - int(np.ceil(bb.y1)) - pad)
        r1 = min(self.H, self.H - int(np.floor(bb.y0)) + pad)
        return (r0, r1, c0, c1) if r1 > r0 and c1 > c0 else None

    def _find_lettering(self, renderer):
        from matplotlib.text import Text
        from matplotlib.transforms import Bbox
        self.names = {}         # leaf index -> its words
        self.extent = {}        # leaf index -> its display bbox
        self.region = {}        # leaf index -> region index
        self.regions = []       # (r0, r1, c0, c1)
        self.reach = {}         # leaf index -> display bbox it can draw in
        pt = self.dpi / 72.0
        for i, (a, _orig) in enumerate(self.leaves):
            box = None
            if isinstance(a, Text):
                words = " ".join(a.get_text().split())
                if not words:
                    continue
                bb = Text.get_window_extent(a, renderer)
                self.names[i] = words[:40]
                self.extent[i] = bb
                bp = a.get_bbox_patch()
                if bp is not None and bp.get_visible():
                    # placed in this frame (the saving pass drew no text)
                    a.update_bbox_position_size(renderer)
                    bb = Bbox.union([bb, bp.get_window_extent(renderer)])
                halo = 3 * pt if a.get_path_effects() else 0
                self.reach[i] = bb.padded(halo + 2 * pt + 4)
                box = self._box(bb, self.gap + 2 + int(halo))
            elif (id(a) in self.roles["frames"]
                  and self._opaque_frame(self.roles["frames"][id(a)])):
                self.names[i] = "the legend"
                box = self._box(a.get_window_extent(renderer), 1)
            if box is not None:
                self.region[i] = len(self.regions)
                self.regions.append(box)

    def _opaque_frame(self, leg) -> bool:
        frame = leg.get_frame()
        return bool(frame.get_visible() and frame.get_fill()
                    and frame.get_facecolor()[3] >= BOX_OPAQUE)

    def _boxed(self, text) -> bool:
        """Whether a text is drawn on its own opaque box (or a halo)."""
        bp = text.get_bbox_patch()
        return bool(text.get_path_effects()) or bool(
            bp is not None and bp.get_visible() and bp.get_fill()
            and bp.get_facecolor()[3] >= BOX_OPAQUE)

    # -- drawing each leaf alone ------------------------------------------

    def _parts(self, i, a):
        """``(part, kind)`` for each separate drawing of leaf ``a``: kind
        ``glyph`` (a text's letters alone: its mask), ``text`` (a text as
        drawn, with its box), ``stroke`` (a line, an edge, markers: what can
        run through lettering), ``face`` (a fill, an image: what can hide
        it), ``grid`` (drawn, never reported) and ``unclipped`` (a text's
        letters without its clip). An arrow (an annotation's leader, which
        it draws apart from its text, or one drawn alone) is a ``stroke``."""
        from matplotlib.collections import Collection
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrowPatch, Patch
        from matplotlib.text import Text
        roles = self.roles
        if isinstance(a, Text):
            if i not in self.names:
                return []
            parts = []
            bp = a.get_bbox_patch()
            if a.get_path_effects() or (bp is not None
                                        and bp.get_visible()):
                parts.append(("glyph", "glyph"))
            parts.append(("full", "text"))
            if a.get_clip_on() and (a.get_clip_box() is not None
                                    or a.get_clip_path() is not None):
                parts.append(("unclipped", "unclipped"))
            return parts
        if isinstance(a, Line2D):
            return [("full", "grid" if id(a) in roles["grid"] else "stroke")]
        if isinstance(a, FancyArrowPatch):
            return [("full", "stroke")]
        if isinstance(a, Patch):
            if (id(a) in roles["panels"] or id(a) in roles["frames"]
                    or not hasattr(a, "_facecolor")
                    or not hasattr(a, "_edgecolor")):
                return [("full", "face")]
            if id(a) in roles["spines"]:
                return [("full", "stroke")]
            parts = []
            if (a.get_fill() and a.get_facecolor()[3] > 0) or a.get_hatch():
                parts.append(("face", "face"))
            if (a.get_linewidth() > 0 and a.get_edgecolor()[3] > 0
                    and a.get_linestyle() not in ("None", "none", "", " ")):
                parts.append(("edge", "stroke"))
            return parts
        if isinstance(a, Collection):
            if id(a) in roles["grid"]:
                return [("full", "grid")]
            if self._lines(a):
                return [("full", "stroke")]
            # fill_between, stackplot, a mesh: the fill hides, the edges
            # run through (as a shape's)
            np = self.np
            try:
                a.update_scalarmappable()
                faces = np.asarray(a.get_facecolor(), float).reshape(-1, 4)
                edges = np.asarray(a.get_edgecolor(), float).reshape(-1, 4)
                widths = np.atleast_1d(np.asarray(a.get_linewidth(), float))
            except Exception:                              # noqa: BLE001
                return [("full", "face")]
            parts = []
            if (faces[:, 3] > 0).any() or a.get_hatch():
                parts.append(("cface", "face"))
            if (edges[:, 3] > 0).any() and (widths > 0).any():
                parts.append(("cedge", "stroke"))
            return parts
        return [("full", "face")]

    def _lines(self, a) -> bool:
        """A collection that is lines or markers, drawn whole as a line."""
        from matplotlib.collections import LineCollection, PathCollection
        return isinstance(a, (LineCollection, PathCollection)) or (
            hasattr(a, "filled") and not a.filled)

    def _draw(self, a, orig, part):
        """Draw one part of leaf ``a`` alone into the second buffer."""
        off = self.off
        if part == "full":
            orig(a, off)
        elif part in ("glyph", "unclipped"):
            bp = a.get_bbox_patch()
            effects = a.get_path_effects()
            shown = bp.get_visible() if bp is not None else None
            try:
                if bp is not None:
                    bp.set_visible(False)
                if effects:
                    a.set_path_effects([])
                if part == "unclipped":
                    a.set_clip_on(False)
                orig(a, off)
            finally:
                if bp is not None:
                    bp.set_visible(shown)
                if effects:
                    a.set_path_effects(effects)
                if part == "unclipped":
                    a.set_clip_on(True)
        elif part in ("cface", "cedge"):        # a collection's fill or edges
            np = self.np
            # its colours as drawn, held while it draws (a mapped
            # collection would map them again)
            edges = np.asarray(a.get_edgecolor(), float).reshape(-1, 4)
            kept = (a._facecolors, a._edgecolors, a._hatch)
            a.update_scalarmappable = lambda: None
            try:
                if part == "cface":
                    a._edgecolors = np.zeros((1, 4))
                else:
                    a._edgecolors = edges
                    a._facecolors = np.zeros((1, 4))
                    a._hatch = None
                orig(a, off)
            finally:
                a._facecolors, a._edgecolors, a._hatch = kept
                del a.update_scalarmappable
        else:                                   # a shape's fill or edge
            kept = (a._facecolor, a._edgecolor, a._hatch)
            try:
                if part == "face":
                    a._edgecolor = (0.0, 0.0, 0.0, 0.0)
                else:
                    a._facecolor = (0.0, 0.0, 0.0, 0.0)
                    a._hatch = None
                orig(a, off)
            finally:
                a._facecolor, a._edgecolor, a._hatch = kept

    def _near(self, i, a):
        """The regions leaf ``a`` can draw in: a clipped leaf draws inside
        its clip box, a text round itself; anything else anywhere."""
        if i in self.reach:
            bb = self.reach[i]
        elif a.get_clip_on() and a.get_clip_box() is not None:
            bb = a.get_clip_box().padded(2)
        else:
            return range(len(self.regions))
        r0, r1, c0, c1 = self.boxes.T
        return self.np.nonzero((c0 <= bb.x1) & (bb.x0 <= c1)
                               & (r0 <= self.H - bb.y0)
                               & (self.H - bb.y1 <= r1))[0].tolist()

    def _collect(self):
        """Draw every leaf part alone, keeping what falls on each region.
        A region that is not lettering (a legend's frame) keeps only its
        own leaf: nothing else is read there."""
        from matplotlib.text import Text
        self.layers = [[] for _ in self.regions]
        owner = {k: i for i, k in self.region.items()}
        buf = self.buf
        for i, (a, orig) in enumerate(self.leaves):
            self._in_time()
            parts = self._parts(i, a)
            if not parts or not self.regions:
                continue
            near = self._near(i, a)
            for n, (part, kind) in enumerate(parts):
                if part == "unclipped" and i not in self.region:
                    continue
                self._draw(a, orig, part)
                inked = []
                look = ([self.region[i]] if part == "unclipped" else near)
                for k in look:
                    r0, r1, c0, c1 = self.regions[k]
                    crop = buf[r0:r1, c0:c1]
                    if crop[..., 3].any():
                        keep = owner[k] == i or isinstance(
                            self.leaves[owner[k]][0], Text)
                        inked.append((k, crop.copy() if keep else None))
                for k, crop in inked:
                    if crop is not None:
                        self.layers[k].append(((i, n), kind, i, crop))
                        self.kept += crop.nbytes
                    r0, r1, c0, c1 = self.regions[k]
                    buf[r0:r1, c0:c1] = 0
                if self.kept > self.mb * 2 ** 20:
                    raise _Unchecked(
                        f"too much drawn on the lettering to check in "
                        f"memory (over {self.mb:g} MB); a look decides")
                if part == "unclipped":
                    # it may draw outside its own region: clear it all
                    self.off.clear()

    # -- what shows ---------------------------------------------------------

    def _mask(self, i, kinds):
        """Leaf ``i``'s ink in its own region from the first of ``kinds``
        drawn, as ``(row, col, bool mask)``; None when it drew nothing."""
        k = self.region.get(i)
        if k is None:
            return None
        r0, _r1, c0, _c1 = self.regions[k]
        for kind in kinds:
            for _order, kd, leaf, crop in self.layers[k]:
                if leaf == i and kd == kind:
                    return r0, c0, crop[..., 3] > INK_ALPHA
        return None

    def _what(self, a, kind) -> str:
        from matplotlib.collections import Collection, PathCollection
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrowPatch, Patch
        roles = self.roles
        if isinstance(a, FancyArrowPatch):
            note = roles["leaders"].get(id(a))
            words = " ".join(note.get_text().split())[:40] if note else ""
            return f"the leader of '{words}'" if words else "an arrow"
        if kind == "face":
            if id(a) in roles["panels"]:
                return "an axes panel drawn over it"
            if isinstance(a, Patch):
                return "a filled shape drawn over it"
            return "an image or filled area drawn over it"
        if id(a) in roles["spines"]:
            return f"the {roles['spines'][id(a)]} frame line"
        if id(a) in roles["ticks"]:
            return "a tick mark"
        if id(a) in roles["symbols"]:
            return "a legend symbol"
        if isinstance(a, Patch) or (isinstance(a, Collection)
                                    and not self._lines(a)):
            return "an outline"
        label = str(a.get_label() or "")
        if label and not label.startswith("_"):
            return label[:40]
        if isinstance(a, PathCollection) or (
                isinstance(a, Line2D)
                and a.get_linestyle() in ("None", "none", "", " ")):
            return "markers"
        return "a line"

    def _through(self, i):
        """The leaf showing most on text ``i``'s glyphs (grown one pixel):
        a line, an edge or markers anywhere in the order, or a fill drawn
        after the text. ``(pixels, what, artist)`` or None."""
        np = self.np
        k = self.region[i]
        mine = self._mask(i, ("glyph", "text"))
        if mine is None or not mine[2].any():
            return None
        grown = _grow(mine[2], 1)
        layers = sorted((e for e in self.layers[k]
                         if e[1] not in ("glyph", "unclipped")),
                        key=lambda e: e[0])
        at = next((e[0] for e in layers if e[2] == i), None)

        text = self.leaves[i][0]
        leaders = self.roles["leaders"]

        def tested(e):
            order, kind, leaf, _crop = e
            # never the text itself, nor its own leader (which starts at it)
            if leaf == i or leaders.get(id(self.leaves[leaf][0])) is text:
                return False
            if kind == "stroke":
                return True
            # a fill drawn over the text; a legend's frame over lettering is
            # the legend's footprint (text_overlaps)
            return (kind == "face" and at is not None and order > at
                    and id(self.leaves[leaf][0]) not in self.roles["frames"])
        if not any(tested(e) for e in layers):
            return None
        # only the glyph pixels count: lay the leaves over one another
        # there, leaving out any that draw nothing on them
        layers = [e for e in layers if e[3][..., 3][grown].any()]
        if not any(tested(e) for e in layers):
            return None
        alphas = [e[3][..., 3][grown].astype(np.float32) / 255.0
                  for e in layers]
        above = [None] * len(layers)
        clear = np.ones(alphas[0].shape, np.float32)
        for n in range(len(layers) - 1, -1, -1):
            above[n] = clear
            clear = clear * (1.0 - alphas[n])
        under = np.full(alphas[0].shape + (3,), 255.0, np.float32)
        best = None
        for n, e in enumerate(layers):
            a = alphas[n][..., None]
            colour = e[3][..., :3][grown].astype(np.float32)
            if tested(e):
                shows = (above[n][..., None] * a
                         * np.abs(colour - under)).sum(axis=1)
                px = int((shows > VISIBLE_DIFF).sum())
                if px >= self.min_px and (best is None or px > best[0]):
                    leaf = self.leaves[e[2]][0]
                    best = (px, self._what(leaf, e[1]), leaf)
            under = a * colour + (1.0 - a) * under
        return best

    def found(self) -> dict:
        from matplotlib.text import Text
        from matplotlib.transforms import Bbox
        self._collect()
        texts = [i for i in self.region if isinstance(self.leaves[i][0],
                                                       Text)]
        texts.sort()
        # each finding by artist, for a check over chosen texts
        self.raw = {"text_overlaps": [], "text_on_a_line": [],
                    "text_cut_off": []}
        on_a_line, cut = [], []
        for i in texts:
            self._in_time()
            hit = self._through(i)
            if hit:
                on_a_line.append([self.names[i], hit[1]])
                # a leader's artist is its annotation
                by = self.roles["leaders"].get(id(hit[2]), hit[2])
                self.raw["text_on_a_line"].append(
                    (self.leaves[i][0], self.names[i], hit[1], by))
        for i in sorted(self.names):
            a = self.leaves[i][0]
            if not isinstance(a, Text):
                continue
            seen = self._mask(i, ("glyph", "text"))
            whole = self._mask(i, ("unclipped",))
            if seen is not None and whole is not None:
                shown, all_ink = int(seen[2].sum()), int(whole[2].sum())
                if shown and all_ink - shown >= max(self.min_px,
                                                    0.02 * all_ink):
                    cut.append([self.names[i], "the edge of its axes"])
                    self.raw["text_cut_off"].append(
                        (a, self.names[i], "the edge of its axes"))
                    continue
            bb = self.extent[i]
            if a.get_clip_on() and a.get_clip_box() is not None:
                # a clipped text reaches only as far as its clip
                bb = Bbox.intersection(bb, a.get_clip_box())
                if bb is None:
                    continue
            if (bb.x0 < -1 or bb.y0 < -1 or bb.x1 > self.W + 1
                    or bb.y1 > self.H + 1):
                cut.append([self.names[i], "the figure edge"])
                self.raw["text_cut_off"].append(
                    (a, self.names[i], "the figure edge"))
        return {"text_overlaps": self._overlaps(texts),
                "text_on_a_line": on_a_line, "text_cut_off": cut,
                "raw": self.raw}

    def _overlaps(self, texts) -> list:
        """Pairs of lettering whose footprints share ink, or whose letters
        come within :attr:`gap` pixels on one line."""
        roles = self.roles
        items = []
        for i in texts:
            mask = self._mask(i, ("glyph", "text"))
            if mask is None or not mask[2].any():
                continue
            a = self.leaves[i][0]
            foot = mask
            drawn = self._alpha(i, "text")
            if self._boxed(a) and drawn is not None:
                # the letters and the opaque box (or halo) they sit on
                foot = (mask[0], mask[1],
                        mask[2] | (drawn >= 255 * BOX_OPAQUE))
            items.append((i, mask, foot, roles["in_legend"].get(id(a)),
                          float(a.get_fontsize())))
        for i in sorted(self.region):
            a = self.leaves[i][0]
            alpha = self._alpha(i, "face")
            if id(a) in roles["frames"] and i in self.names \
                    and alpha is not None:
                r0, _r1, c0, _c1 = self.regions[self.region[i]]
                items.append((i, None, (r0, c0, alpha >= 255 * BOX_OPAQUE),
                              roles["frames"][id(a)], None))
        np = self.np
        # each footprint's frame box, to pass over pairs far apart at once
        top = np.array([e[2][0] for e in items], float)
        left = np.array([e[2][1] for e in items], float)
        bottom = top + [e[2][2].shape[0] for e in items]
        right = left + [e[2][2].shape[1] for e in items]
        reach = max([self._gap(e[4]) for e in items if e[4]] + [self.gap])
        pairs = []
        for x in range(len(items)):
            self._in_time()
            near = np.nonzero((top - reach <= bottom[x])
                              & (top[x] - reach <= bottom)
                              & (left - reach <= right[x])
                              & (left[x] - reach <= right))[0]
            for y in near[near > x].tolist():
                i, gi, fi, li, si = items[x]
                j, gj, fj, lj, sj = items[y]
                frame_i, frame_j = gi is None, gj is None
                size = min(s for s in (si, sj, 1e9) if s is not None)
                g = self._gap(size)
                if (frame_i and frame_j) or self._apart(fi, fj, g):
                    continue
                if li is not None and li is lj:
                    if frame_i or frame_j:
                        continue                  # an entry on its own frame
                    hit = self._close(gi, gj, g)
                else:
                    # an entry of an opaque legend meets others as "the
                    # legend"
                    if ((li is not None and not frame_i
                         and self._opaque_frame(li))
                            or (lj is not None and not frame_j
                                and self._opaque_frame(lj))):
                        continue
                    # a box is its text's footprint: a label's box against
                    # another's letters reads as one word too (live wave
                    # w9: "24" on its white box against "Medium")
                    hit = self._close(fi, fj, g)
                if hit:
                    if frame_i:
                        pairs.append([self.names[j], "the legend"])
                    elif frame_j:
                        pairs.append([self.names[i], "the legend"])
                    else:
                        pairs.append([self.names[i], self.names[j]])
                    self.raw["text_overlaps"].append(
                        (self.leaves[i][0], self.names[i],
                         self.leaves[j][0], self.names[j]))
        return pairs

    def _alpha(self, i, kind="text"):
        k = self.region[i]
        for _order, kd, leaf, crop in self.layers[k]:
            if leaf == i and kd == kind:
                return crop[..., 3]
        return None

    def _gap(self, size_pt: float) -> int:
        """The one-word gap in pixels for lettering ``size_pt`` tall."""
        pt = max(TOUCH_PT, TOUCH_EM * min(size_pt, 1e3))
        return max(1, int(round(pt * self.dpi / 72.0)))

    @staticmethod
    def _apart(f, h, g: int) -> bool:
        """Two placed masks whose frames are more than ``g`` apart."""
        return (f[0] - g > h[0] + h[2].shape[0]
                or h[0] - g > f[0] + f[2].shape[0]
                or f[1] - g > h[1] + h[2].shape[1]
                or h[1] - g > f[1] + f[2].shape[1])

    def _close(self, a, b, g: int) -> bool:
        """Two footprints (letters, or letters on their box) overlap, or
        come within ``g`` pixels with their ink on the same rows."""
        if _meet(a, b):
            return True
        ra, rb = _ink_rows(a[2], a[0]), _ink_rows(b[2], b[0])
        if ra is None or rb is None or min(ra[1], rb[1]) < max(ra[0],
                                                                 rb[0]):
            return False
        return _meet(a, b, grow=g)
