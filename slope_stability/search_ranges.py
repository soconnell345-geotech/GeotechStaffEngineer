"""
Automatic search ranges from the slope's own geometry (BACKLOG GM-57).

When the caller gives no entry / exit windows or centre ranges, the search
used to set them from the MODEL extent: entry in the first 40 % of the
profile, exit in the last 40 %, circle centres over the whole profile from
1 unit above the highest ground to 2 H above it. The answer then depended on
how the model was drawn. A slope off the middle of its model had its critical
circle cut off by a window (the steep phi = 0 slope of V-054, drawn from its
toe: the toe circle exits the crest inside the exit window's lower bound).
A model drawn wide put the split between the windows far out on the level
ground, so only large circles were admissible, and spread the centre grid
thin. The ranges now follow the slope.

- **The slope.** The faces of the ground surface (`slope_faces`: runs of
  segments steeper than 5 deg rising the same way), grouped by the way they
  rise. A group's toe is its lowest toe, its crest its highest crest, H the
  height between them. A group less than a third as tall as the tallest is
  left out (a ditch beside an embankment); faces rising both ways (an
  embankment, a dam, a cut) each give a group.
- **The base.** The depth d from a group's toe down to the bottom of the
  deepest layer under the face, the lowest a trial surface may go. This is
  Taylor's firm base, DH below the crest, so d = (D - 1) H (FHWA NHI-06-088
  Fig. 6-15, after Taylor 1948; Janbu 1968, Fig. 6-16, depth factor d = D/H
  measured from the toe). A firm layer above the model base only shortens
  the critical circle, so the model base gives the wider range.
- **Centre heights** from 0.1 H to max(2 H, d) above the highest ground over
  the centre range. Janbu's charts (FHWA NHI-06-088 Figs. 6-16 and 6-19) put
  the critical centre 1.35 H to 3 H above the toe for depth factors up to
  d = 3; the midpoint circles rise with d, so the top rises with d once d
  passes 2 H. (The old bottom, 1 unit above the crest, did not scale.)
- **Reach.** How far a circle can come out beyond the toe and go in behind
  the crest: the deepest circle the centre grid holds, tangent to the base
  from a centre at the top of the grid, Y above the toe:
  sqrt(d (2 Y + d)) beyond the toe from a centre above the toe (none when
  the base is at the toe: no circle can come out beyond it), and
  sqrt((Y + d)^2 - (Y - H)^2) behind the crest from a centre above the
  crest. Taylor's critical circles lie inside these with room to spare
  (independent phi = 0 optimisations, as in GM-63): 2:1 slope, D = 2, out
  1.19 H beyond the toe and in 1.76 H behind the crest, against reaches of
  2.65 H and 3.46 H; D = 3, 2.69 H / 3.2 H against 4.0 H / 4.58 H; 1:1
  slope, D = 2.25, 1.96 H / 2.45 H against 3.01 H / 3.75 H.
- **Windows.** The end of a surface on a face group's toe side lies between
  the toe's reach and the face's mid-length (horizontally); the end on the
  crest side between the mid-length and the crest's reach. A surface must
  cross the middle of the slope, as the old split between the windows
  meant, but measured on the slope. With several groups the windows are
  joined. The automatic ranges are a default for the slope as a whole, not
  a judgment about the ground: a failure confined to part of the face (a
  weak layer over rock above the toe, a slide above the water outside the
  slope; FHWA NHI-06-088 p. 6-38) is searched for by giving the windows,
  and the result says where the ranges ran (`search_ranges`) and where the
  governing surface lies (`governing_location`).
- **Centre x** from the lowest to the highest end the windows allow: a
  circle's centre lies between its ends, so the grid spacing follows the
  slope, not the model width.
- **The model's ends.** A range is cut at the end of the ground profile and
  the cut is recorded (`AutoRanges.clipped`). When the governing surface
  ends at a cut, the GM-63 profile-end check gives the lower FOS of a
  circle past the end if it finds one; if not, the search still says the
  model is shorter than the slope calls for (`search_supplements`).
- **No slope face** (level ground): the old fractions of the profile.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from slope_stability.geometry import SlopeGeometry

# A face group less than this fraction of the tallest group's height is
# left out of the ranges (the toe-circle family still covers its toe).
_MIN_GROUP_HEIGHT = 1.0 / 3.0
# Centre heights above the highest ground over the centre range, in H: the
# bottom, and the top's least value (the top is max(this H, d)).
_Y_LO_H = 0.1
_Y_HI_H = 2.0
# Numerical slack on the reach beyond the toe, in H, so that a surface
# through the toe of a slope standing on its base stays in its window ...
_REACH_SLACK_H = 0.02
# ... and a bound the model's end cuts by less than this, in H, is not
# recorded as cut.
_CLIP_TOL_H = 0.05
# The windows meet at the face's mid-length.
_MIDDLE_PART = 0.5


@dataclass
class AutoRanges:
    """The automatic search ranges and how they were set.

    ``clipped`` maps a bound ('entry_lo', 'exit_hi') cut at the end of the
    ground profile to the x the slope called for. ``slopes`` has one entry
    per face group used: ``rises`` (+1 toward +x), ``toe`` and ``crest``
    (x, z), ``height`` H, ``depth_to_base`` d below the toe and the
    ``reach_toe`` / ``reach_crest`` beyond them. Empty for level ground
    (``basis`` then says the old fractions of the profile were used).
    """
    x_entry_range: Tuple[float, float]
    x_exit_range: Tuple[float, float]
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    clipped: Dict[str, float] = field(default_factory=dict)
    slopes: List[Dict[str, Any]] = field(default_factory=list)
    basis: str = "slope"

    def to_dict(self) -> Dict[str, Any]:
        r = lambda v: round(v, 3)   # noqa: E731
        return {
            "basis": self.basis,
            "slopes": [{k: ([r(c) for c in v] if isinstance(v, tuple)
                            else (r(v) if isinstance(v, float) else v))
                        for k, v in s.items()} for s in self.slopes],
            "clipped": {k: r(v) for k, v in self.clipped.items()},
        }


def _base_under(geom: SlopeGeometry, xs) -> float:
    """Lowest elevation of the bottom of the deepest layer at the given x."""
    return min(min(L.bottom_at(x) for L in geom.soil_layers) for x in xs)


def _fraction_ranges(geom: SlopeGeometry) -> AutoRanges:
    """The old ranges, fractions of the profile (level ground only)."""
    x_min, x_max = geom.x_range
    w = x_max - x_min
    zs = [z for _, z in geom.surface_points]
    z_max, h = max(zs), max(zs) - min(zs)
    return AutoRanges(
        x_entry_range=(x_min, x_min + 0.4 * w),
        x_exit_range=(x_min + 0.6 * w, x_max),
        x_range=(x_min, x_max),
        y_range=(z_max + 1.0, z_max + 2.0 * h),
        basis="no slope face: fractions of the ground profile")


def face_groups(geom: SlopeGeometry) -> List[Dict[str, Any]]:
    """The face groups (see the module docstring), each a dict with
    ``rises``, ``toe``, ``crest`` and ``height``; a group less than a third
    as tall as the tallest is left out."""
    from slope_stability.search_supplements import slope_faces

    faces = slope_faces(geom)
    groups = []
    for s in (1, -1):
        fs = [f for f in faces if f["rises"] == s]
        if not fs:
            continue
        # the lowest toe and the highest crest; of equals, the one nearer
        # the rest of the slope (a ditch on the crest is not the crest)
        toe = min((f["toe"] for f in fs), key=lambda p: (p[1], -s * p[0]))
        crest = max((f["crest"] for f in fs), key=lambda p: (p[1], -s * p[0]))
        h = crest[1] - toe[1]
        if h > 0:
            groups.append({"rises": s, "toe": toe, "crest": crest,
                           "height": h})
    if not groups:
        return []
    h_max = max(g["height"] for g in groups)
    return [g for g in groups if g["height"] >= _MIN_GROUP_HEIGHT * h_max]


def automatic_ranges(geom: SlopeGeometry) -> AutoRanges:
    """Entry / exit windows and centre ranges set from the slope (GM-57).

    See the module docstring for the basis. Returns an `AutoRanges`.
    """
    groups = face_groups(geom)
    if not groups:
        return _fraction_ranges(geom)
    h_max = max(g["height"] for g in groups)
    x_min, x_max = geom.x_range
    for g in groups:
        (xt, zt), (xc, _) = g["toe"], g["crest"]
        g["depth_to_base"] = max(
            zt - _base_under(geom, (xt, 0.5 * (xt + xc), xc)), 0.0)
    d_max = max(g["depth_to_base"] for g in groups)
    rise_top = max(_Y_HI_H * h_max, d_max)

    # The windows need the grid top (the reach) and the grid top needs the
    # ground over the centre range (the windows): take the highest crest
    # first, then the highest ground over the windows found.
    z_top = max(g["crest"][1] for g in groups)
    for _ in range(2):
        lo_ends, hi_ends, lo_inner, hi_inner = [], [], [], []
        for g in groups:
            (xt, zt), (xc, _) = g["toe"], g["crest"]
            h, d = g["height"], g["depth_to_base"]
            y_top = z_top + rise_top - zt          # grid top above the toe
            g["reach_toe"] = max(_REACH_SLACK_H * h,
                                 math.sqrt(d * (2.0 * y_top + d)))
            g["reach_crest"] = math.sqrt(max(
                (y_top + d) ** 2 - (y_top - h) ** 2, 0.0))
            inner = _MIDDLE_PART * abs(xc - xt)
            if g["rises"] > 0:
                lo_ends.append(xt - g["reach_toe"])
                hi_ends.append(xc + g["reach_crest"])
            else:
                lo_ends.append(xc - g["reach_crest"])
                hi_ends.append(xt + g["reach_toe"])
            lo_inner.append(min(xt, xc) + inner)
            hi_inner.append(max(xt, xc) - inner)
        lo_want, hi_want = min(lo_ends), max(hi_ends)
        lo, hi = max(lo_want, x_min), min(hi_want, x_max)
        z_new = max([z for x, z in geom.surface_points if lo <= x <= hi]
                    + [geom.ground_elevation_at(lo),
                       geom.ground_elevation_at(hi)])
        if z_new <= z_top:
            break
        z_top = z_new

    clipped = {}
    if lo_want < x_min - _CLIP_TOL_H * h_max:
        clipped["entry_lo"] = lo_want
    if hi_want > x_max + _CLIP_TOL_H * h_max:
        clipped["exit_hi"] = hi_want
    slopes = [{"rises": g["rises"], "toe": g["toe"], "crest": g["crest"],
               "height": g["height"], "depth_to_base": g["depth_to_base"],
               "reach_toe": g["reach_toe"], "reach_crest": g["reach_crest"]}
              for g in groups]
    return AutoRanges(
        x_entry_range=(lo, max(lo_inner)),
        x_exit_range=(min(hi_inner), hi),
        x_range=(lo, hi),
        y_range=(z_top + _Y_LO_H * h_max, z_top + rise_top),
        clipped=clipped, slopes=slopes)


def resolve_ranges(geom: SlopeGeometry,
                   x_entry_range=None, x_exit_range=None,
                   x_range=None, y_range=None,
                   ) -> Tuple[Tuple, Tuple, Tuple, Tuple, Optional[AutoRanges]]:
    """Fill the ranges the caller left as None from `automatic_ranges`.

    An automatic centre x range spans the windows actually used (given or
    automatic) and the slope's own, so caller-given windows stay inside the
    grid. Returns (x_entry_range, x_exit_range, x_range, y_range, auto);
    ``auto`` is None when the caller gave all four.
    """
    if None not in (x_entry_range, x_exit_range, x_range, y_range):
        return x_entry_range, x_exit_range, x_range, y_range, None
    auto = automatic_ranges(geom)
    if x_entry_range is None:
        x_entry_range = auto.x_entry_range
    if x_exit_range is None:
        x_exit_range = auto.x_exit_range
    if x_range is None:
        x_range = (min(auto.x_range[0], x_entry_range[0]),
                   max(auto.x_range[1], x_exit_range[1]))
    if y_range is None:
        y_range = auto.y_range
    return x_entry_range, x_exit_range, x_range, y_range, auto
