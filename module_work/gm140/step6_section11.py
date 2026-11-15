"""GM-140 step 6 (one method across the search): the benchmarks of the
account's section 11 that this step's brief expects to move and the earlier
scripts do not measure.

``step2_section11.py``, ``step3_section11.py`` and
``circle_ends_section11.py`` are run unchanged for their own rows (Slide2
#85 and #86 with supports, ACADS 1(a), V-043, V-031, V-044, V-039, the
probabilistic cases, ACADS 4, V-054, V-032, V-042, Duncan's Example 1, the
vertical cut, B14). This script adds the rows for step 6:

* each planar family against its closed form, by every method named, with
  how many of the family's planes the method solves (GM-56);
* the searches whose families were scored by a method other than the one
  named, with what each family reports and under which method;
* the noncircular searches against the critical circle by the same method.

    PYTHONPATH=. python module_work/gm140/step6_section11.py          # this tree
    PYTHONPATH=<a master worktree> python <this file>                 # master

One line per measured value. It runs on both trees: the fields and options
that exist only after the step are read with getattr and passed only where
the tree takes them. Each geometry is taken from the test module that pins
it (nothing is re-entered by hand).
"""
import inspect
import math
import sys
import time
import warnings

warnings.simplefilter("ignore")

from slope_stability.analysis import search_critical_surface   # noqa: E402
from slope_stability import search as S                        # noqa: E402
from slope_stability import search_supplements as SS           # noqa: E402

NEW = "f_interslice" in inspect.signature(search_critical_surface).parameters
METHODS = ("fellenius", "bishop", "janbu", "spencer", "morgenstern_price",
           "gle")


def r4(v):
    return None if v is None or v >= 999.0 else round(v, 4)


def planes(name, geom, closed, n_slices=30):
    """Every plane of the family (`check_infinite_slope`'s own spans)
    scored by each method through the search's own `_compute_fos`: how many
    the method solves, the lowest, and its difference from the closed
    form. (Before step 6 `_compute_fos` scored a polyline named for Bishop
    by Spencer; the family itself was scored by Janbu whatever was named.)"""
    for m in METHODS:
        n = n_ok = 0
        best = None
        for run in SS.cohesionless_face_runs(geom):
            lo, hi = run["x_from"], run["x_to"]
            w = hi - lo
            for k in SS._PLANAR_SPLITS:
                for i in range(k):
                    a, b = lo + w * i / k, lo + w * (i + 1) / k
                    slip = SS._planar_surface(geom, a, b, n_slices)
                    if slip is None:
                        continue
                    n += 1
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f = S._compute_fos(geom, slip, m, n_slices,
                                           window_span=b - a)
                    if f < 999.0:
                        n_ok += 1
                        best = f if best is None else min(best, f)
        diff = (None if best is None
                else f"{100 * (best / closed - 1):+.2f} %")
        print(f"planes | {name} | {m}: solved {n_ok} of {n}; lowest "
              f"{r4(best)}; closed form {closed:.4f} ({diff})", flush=True)


def show(name, res, t=None):
    c = res.critical
    out = {
        "FOS": None if c is None else r4(c.FOS),
        "method": None if c is None else c.method,
        "governs": res.governing_surface,
        "candidates": dict(res.candidates),
    }
    for key in ("candidate_methods", "named_method", "method_used",
                "named_method_lowest", "screen_unsolved",
                "circle_check"):
        if hasattr(res, key):
            out[key] = getattr(res, key)
    if t is not None:
        out["seconds"] = round(t, 1)
    print(f"search | {name}: {out}", flush=True)
    return res


def search(name, geom, **kw):
    if not NEW:
        kw.pop("f_interslice", None)
    t0 = time.perf_counter()
    res = search_critical_surface(geom, **kw)
    return show(name, res, time.perf_counter() - t0)


def planar():
    from slope_stability.tests.test_search_supplements import _fill_geom
    from slope_stability.tests.test_duncan_verification import (
        TestDuncanExample3)
    from slope_stability.tests.test_m_alpha import _duncan_ex4
    from slope_stability.tests.test_search_supplements import _closed_form
    beta86 = math.degrees(math.atan(0.8))
    planes("#86 without supports, c' = 0 (tan 37 / tan 38.66)",
           _fill_geom(), _closed_form(beta86, 37.0))
    planes("#86 without supports, ru = 0.2", _fill_geom(ru=0.2),
           _closed_form(beta86, 37.0, ru=0.2))
    g = _fill_geom()
    g.kh = 0.1
    planes("#86 without supports, kh = 0.1", g,
           _closed_form(beta86, 37.0, kh=0.1))
    beta2 = math.degrees(math.atan(0.5))
    planes("Duncan Ex. 3 geometry, dry (tan 40 / tan 26.57)",
           TestDuncanExample3()._geom(), _closed_form(beta2, 40.0))
    planes("Duncan Ex. 3, kh = 0.15", TestDuncanExample3()._geom(kh=0.15),
           _closed_form(beta2, 40.0, kh=0.15))
    beta3 = math.degrees(math.atan(1.0 / 3.0))
    planes("Duncan Ex. 4, sand face (tan 38 / tan 18.43)", _duncan_ex4(),
           _closed_form(beta3, 38.0))


def searches():
    from slope_stability.tests.test_search_supplements import (
        _fill_geom, _toe_geom)
    from slope_stability.tests.test_m_alpha import (_duncan_ex4,
                                                   _weak_over_rock)
    from slope_stability.tests.test_duncan_verification import (
        TestDuncanExample3)
    for m in ("bishop", "fellenius", "janbu", "spencer",
              "morgenstern_price"):
        search(f"#86 without supports, c' = 0, {m}", _fill_geom(), method=m)
    search("#86 without supports, GLE, constant function", _fill_geom(),
           method="gle", f_interslice="constant")
    search("#86 without supports, Bishop, circles kept off the face",
           _fill_geom(), method="bishop", x_entry_range=(0.0, 5.0),
           x_exit_range=(30.0, 75.0))
    search("#86 without supports, Spencer, circles kept off the face",
           _fill_geom(), method="spencer", x_entry_range=(0.0, 5.0),
           x_exit_range=(30.0, 75.0))
    search("Duncan Ex. 3 geometry, kh = 0.15, Spencer",
           TestDuncanExample3()._geom(kh=0.15), method="spencer")
    search("Duncan Ex. 4, Bishop", _duncan_ex4(), method="bishop")
    search("Duncan Ex. 4, Spencer", _duncan_ex4(), method="spencer")
    search("V-054 steep phi = 0 slope, Spencer, 40 slices", _toe_geom(),
           method="spencer", n_slices=40)
    search("V-054 steep phi = 0 slope, GLE constant, 40 slices",
           _toe_geom(), method="gle", n_slices=40, f_interslice="constant")
    search("B14 weak layer over rock, default search, Spencer",
           _weak_over_rock(), method="spencer")


def noncircular():
    from slope_stability.tests.test_search_modern import _acads_geom
    import validation_examples.test_published_v043_slope_reinforced as v43
    import validation_examples.test_published_v028_e8 as v28
    from slope_stability.tests.test_new_search import _two_layer_slope
    acads = dict(n_slices=25, x_entry_range=(20.0, 32.0),
                 x_exit_range=(46.0, 68.0))
    search("ACADS 1(a), circle grid, Spencer, 25 slices", _acads_geom(),
           method="spencer", n_slices=25)
    for m in ("spencer", "bishop", "morgenstern_price", "janbu"):
        search(f"ACADS 1(a), random noncircular, 200 trials, seed 42, {m}",
               _acads_geom(), method=m, surface_type="noncircular",
               n_trials=200, n_points=6, seed=42, **acads)
    search("ACADS 1(a), DE, seed 1, Spencer", _acads_geom(),
           method="spencer", surface_type="noncircular_de", n_points=6,
           seed=1, **acads)
    for m in ("spencer", "morgenstern_price", "janbu"):
        search(f"ACADS 1(a), particle swarm, seed 1, {m} named",
               _acads_geom(), method=m, surface_type="pso", nx=5, ny=4,
               n_trials=200, n_points=6, seed=1, **acads)
    for m in ("spencer", "morgenstern_price"):
        search(f"two-layer slope, weak-layer search, seed 1, {m} named",
               _two_layer_slope(), method=m, surface_type="weak_layer",
               n_trials=150, seed=1)
    search("V-043 sand, circle grid, Spencer", v43._geom("sand"),
           method="spencer")
    search("V-043 sand, DE, seed 1, Spencer", v43._geom("sand"),
           method="spencer", surface_type="noncircular_de", n_slices=30,
           seed=1)
    search("V-028 ACADS 4, weak-layer search, 700 trials, seed 3, Spencer",
           v28._geom(surcharges=v28._ZONES), surface_type="weak_layer",
           method="spencer", n_trials=700, n_points=6, n_slices=30, seed=3,
           x_entry_range=(20.0, 50.0), x_exit_range=(58.0, 84.0))


def main(argv):
    parts = {"planar": planar, "searches": searches,
             "noncircular": noncircular}
    for name in ([a for a in argv if a in parts] or list(parts)):
        parts[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
