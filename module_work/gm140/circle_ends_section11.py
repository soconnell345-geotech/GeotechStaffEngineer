"""GM-140 "Where a circle ends" (GM-343): the rows of the account's
section 11 this step is expected to move, and the searches whose trial
circles meet the ground at more than two points.

``step2_section11.py``, ``step3_section11.py`` and ``step4_section11.py``
are run unchanged for their own rows. This script adds the rows for this
step: the vertical cut against Taylor (B16), the circles the brief
measured, and every search where circles were refused as two masses.

    PYTHONPATH=. python module_work/gm140/circle_ends_section11.py           # this tree
    PYTHONPATH=<a master worktree> python <this file>                        # master

One line per measured value. It runs on both trees: fields that exist only
after the step are read with getattr. Each geometry is taken from the test
module that pins it, or is the brief's (nothing else is re-entered).
"""
import math
import warnings

warnings.simplefilter("ignore")

from slope_stability.analysis import (analyze_slope,           # noqa: E402
                                      search_critical_surface)
from slope_stability.geometry import (SlopeGeometry,           # noqa: E402
                                      SlopeSoilLayer)
from slope_stability.slip_surface import CircularSlipSurface   # noqa: E402
from slope_stability.tests import test_ground_crossings as tgc  # noqa: E402
from slope_stability.tests import test_search_supplements as tss  # noqa: E402
from slope_stability.tests import test_choices as tch          # noqa: E402


def vertical_cut():
    """The brief's cut (`_vertical_cut()` of tests/test_m_alpha.py)."""
    return SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (20.001, 5.0), (45, 5.0)],
        soil_layers=[SlopeSoilLayer("clay", 5.0, -10.0, 18.0, cu=20.0,
                                    analysis_mode="undrained")])


def f3(v):
    return "none" if v is None else f"{v:.4f}"


def one(name, geom, **kw):
    try:
        r = analyze_slope(geom, **kw)
        arc = getattr(r, "arc", None)
        print(f"{name}: FOS {f3(r.FOS)}, x {r.x_entry:.3f} to {r.x_exit:.3f}"
              + (f", arc {arc['index'] or 'as given'} of {arc['n_arcs']}" if arc else ""))
    except ValueError as exc:
        print(f"{name}: refused ({str(exc)[:110]})")


def search(name, geom, **kw):
    r = search_critical_surface(geom, **kw)
    c = r.critical
    split = getattr(r, "circles_split", None)
    print(f"{name}: FOS {f3(c.FOS if c else None)}"
          + (f", x {c.x_entry:.3f} to {c.x_exit:.3f}" if c else "")
          + f", governs {r.governing_surface}, candidates {r.candidates}"
          + f", refused as two masses {r.n_rejected_ground_crossings}"
          + (f", circles split {split}" if split is not None else ""))


def main():
    cut = vertical_cut()
    print("--- B16: the vertical cut, Taylor 3.83 x 20 / (18 x 5) = 0.8511")
    for m in ("bishop", "fellenius", "janbu", "spencer",
              "morgenstern_price"):
        search(f"vertical cut, {m} search, default", cut, method=m)
    for n in (30, 120):
        one(f"Taylor's circle (13.02, 10.96) R 13.00, Bishop, {n} slices",
            cut, xc=13.02, yc=10.96, radius=13.00, method="bishop",
            n_slices=n)
    r_toe = math.hypot(20.0 - 13.02, 10.96)
    for n in (30, 120):
        one(f"Taylor's circle through the toe (R {r_toe:.4f}), Bishop, "
            f"{n} slices", cut, xc=13.02, yc=10.96, radius=r_toe,
            method="bishop", n_slices=n)
    one("Taylor's circle through the toe, Spencer, 60 slices", cut,
        xc=13.02, yc=10.96, radius=r_toe, method="spencer", n_slices=60)
    try:
        ended = CircularSlipSurface(13.02, 10.96, 13.00, 20.0, 24.57)
        one("Taylor's circle R 13.00 ended at the toe by its two ends, "
            "Bishop, 120 slices", cut, slip_surface=ended, method="bishop",
            n_slices=120)
    except TypeError:
        print("Taylor's circle R 13.00 ended at the toe by its two ends: "
              "a circle takes no ends on this tree")
    one("circle leaving the face 2 % of H above the toe (through "
        "(20.00002, 0.1) and the crest at x 24.57)", cut,
        slip_surface=_through((20.00002, 0.1), (24.57, 5.0), 13.0),
        method="bishop", n_slices=120)
    one("circle leaving the face 5 % of H above the toe", cut,
        slip_surface=_through((20.00005, 0.25), (24.57, 5.0), 13.0),
        method="bishop", n_slices=120)
    one("the search's circle before (18.48, 6.23) R 6.27", cut, xc=18.48,
        yc=6.23, radius=6.27, method="bishop")

    print("--- circles refused as two masses before (GM-122's cases)")
    w10 = tgc._w10()
    one("w10 circle (10, 39.5) R 40, Bishop", w10, xc=10.0, yc=39.5,
        radius=40.0, method="bishop", n_slices=40)
    one("w10 circle (10, 39.5) R 40, Spencer", w10, xc=10.0, yc=39.5,
        radius=40.0, method="spencer", n_slices=40)
    one("w10 circle (0, 50) R 50.92, Bishop", w10, xc=0.0, yc=50.0,
        radius=50.92, method="bishop", n_slices=40)
    search("w10 default search, Bishop, 40 slices", w10, method="bishop",
           n_slices=40)
    search("#85 clay face without its support, Bishop", tgc._clay85(),
           method="bishop")

    print("--- steep faces and toe circles (B12)")
    search("V-054 slope as given (ends at the toe), Bishop, 40 slices",
           tss._toe_geom(), method="bishop", n_slices=40)
    search("windows given: entry 0 to 4, exit 14 to 20 (test_choices)",
           tch._geom(), x_entry_range=(0, 4), x_exit_range=(14, 20))


def _through(p, q, r):
    """The circle of radius r through p and q with its centre above."""
    (x1, z1), (x2, z2) = p, q
    mx, mz = 0.5 * (x1 + x2), 0.5 * (z1 + z2)
    c = math.hypot(x2 - x1, z2 - z1)
    t = math.sqrt(r * r - 0.25 * c * c)
    return CircularSlipSurface(mx - t * (z2 - z1) / c,
                               mz + t * (x2 - x1) / c, r)


if __name__ == "__main__":
    main()
