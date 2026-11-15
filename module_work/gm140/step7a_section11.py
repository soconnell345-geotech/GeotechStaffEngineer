"""GM-140 step 7a (search coverage as printed, the noncircular search): the
benchmarks of the account's section 11 that this step's brief expects to
move, before and after.

``step4_section11.py`` is run unchanged for every pinned test of the
validation set (the circular values, which must not move). This script adds
the noncircular rows the brief lists:

* ACADS 4 (V-028, B11): the weak-layer search, Spencer, Morgenstern-Price
  and Janbu (refine False and True where the tree takes it), and the
  random noncircular search;
* V-043 (sand): differential evolution, and the random search with 2,000
  trials, seeds 1 to 5 (the spread);
* ACADS 1(a): the random search by Spencer;
* B14 (weak layer over rock): random with the default windows, DE directed
  at the face, the circle search by Spencer;
* Duncan's Example 1, D = 2: DE and random (seeds 1 to 3);
* V-028's DE, seeds 1 to 3.

    PYTHONPATH=. python module_work/gm140/step7a_section11.py         # this tree
    PYTHONPATH=<a master worktree> python <this file>                # master

One line per measured value, with its run time. It runs on both trees:
``refine`` False on a noncircular search is passed only where the tree's
noncircular searches take it (the step adds it). Each geometry is taken from
the test module that pins it.
"""
import sys
import time
import warnings

warnings.simplefilter("ignore")

from slope_stability.analysis import search_critical_surface   # noqa: E402

try:
    from slope_stability import search_refine   # noqa: F401
    NEW = True
except ImportError:
    NEW = False


def r4(v):
    return None if v is None or v >= 999.0 else round(v, 4)


def run(name, geom, **kw):
    if not NEW and kw.get("surface_type", "circular") != "circular":
        if kw.get("refine") is False:
            print(f"search | {name}: (not on this tree)", flush=True)
            return None
        kw.pop("refine", None)
    t0 = time.perf_counter()
    res = search_critical_surface(geom, **kw)
    t = time.perf_counter() - t0
    c = res.critical
    out = {"FOS": None if c is None else r4(c.FOS),
           "method": None if c is None else c.method,
           "governs": res.governing_surface,
           "flags": list(getattr(res, "critical_flags", []) or []),
           "seconds": round(t, 1)}
    low = getattr(res, "lowest_unflagged", None)
    if low:
        out["lowest_unflagged"] = low.get("FOS")
    mins = getattr(res, "local_minima", None)
    if mins:
        out["local_minima"] = [(m.get("start"), m.get("F_start"),
                                m.get("FOS")) for m in mins]
    ref = getattr(res, "refinement", None)
    if ref and ref.get("final_increment"):
        out["final_increment"] = ref["final_increment"]["increment"]
        out["wedges"] = [(w["layer"], w["FOS"]) for w in ref["wedges"]]
    if c is not None and getattr(c, "slip_points", None):
        out["points"] = [(round(x, 2), round(z, 2)) for x, z in
                         c.slip_points]
    print(f"search | {name}: {out}", flush=True)
    return res


def acads4():
    import validation_examples.test_published_v028_e8 as v28
    kw = dict(n_trials=700, n_points=6, n_slices=30, seed=3,
              x_entry_range=(20.0, 50.0), x_exit_range=(58.0, 84.0))
    for m in ("spencer", "morgenstern_price", "janbu"):
        for refine in (False, True):
            run(f"ACADS 4, weak-layer search, {m}, refine={refine}",
                v28._geom(surcharges=v28._ZONES), surface_type="weak_layer",
                method=m, refine=refine, **kw)
    run("ACADS 4, random noncircular, 700 trials, seed 3, spencer",
        v28._geom(surcharges=v28._ZONES), surface_type="noncircular",
        method="spencer", **kw)
    for seed in (1, 2, 3):
        run(f"ACADS 4 (V-028), DE, seed {seed}, spencer",
            v28._geom(surcharges=v28._ZONES), surface_type="noncircular_de",
            method="spencer", n_points=6, n_slices=30, seed=seed,
            x_entry_range=(20.0, 50.0), x_exit_range=(58.0, 84.0))


def v043():
    import validation_examples.test_published_v043_slope_reinforced as v43
    run("V-043 sand, DE, seed 1, spencer", v43._geom("sand"),
        method="spencer", surface_type="noncircular_de", n_slices=30, seed=1)
    for seed in (1, 2, 3, 4, 5):
        run(f"V-043 sand, random, 2000 trials, seed {seed}, spencer",
            v43._geom("sand"), method="spencer", surface_type="noncircular",
            n_trials=2000, n_slices=30, seed=seed)


def acads1a():
    from slope_stability.tests.test_search_modern import _acads_geom
    run("ACADS 1(a), random, 200 trials, seed 42, spencer", _acads_geom(),
        method="spencer", surface_type="noncircular", n_trials=200,
        n_points=6, n_slices=25, seed=42, x_entry_range=(20.0, 32.0),
        x_exit_range=(46.0, 68.0))


def b14():
    from slope_stability.tests.test_m_alpha import _weak_over_rock
    run("B14, circle search, spencer", _weak_over_rock(), method="spencer")
    run("B14, random, default windows, seed 1, spencer", _weak_over_rock(),
        method="spencer", surface_type="noncircular", seed=1)
    # directed at the face: the toe side's window on the face and the
    # flat below, the other behind the crest
    run("B14, DE, directed (toe side 18-30, crest side 30-50), seed 1, "
        "spencer", _weak_over_rock(), method="spencer",
        surface_type="noncircular_de", seed=1, x_entry_range=(18.0, 30.0),
        x_exit_range=(30.0, 50.0))


def duncan1():
    from slope_stability.tests.test_duncan_verification import \
        TestDuncanExample1
    g, _, _ = TestDuncanExample1()._long_geom(4.0, 2)
    run("Duncan Ex. 1, D = 2, DE, seed 1, spencer", g, method="spencer",
        surface_type="noncircular_de", seed=1)
    for seed in (1, 2, 3):
        run(f"Duncan Ex. 1, D = 2, random, seed {seed}, spencer", g,
            method="spencer", surface_type="noncircular", seed=seed)


def main(argv):
    parts = {"acads4": acads4, "v043": v043, "acads1a": acads1a,
             "b14": b14, "duncan1": duncan1}
    for name in ([a for a in argv if a in parts] or list(parts)):
        parts[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
