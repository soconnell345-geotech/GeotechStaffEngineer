"""GM-140 step 2: the published validation set (the account's section 11),
measured on the code this script is run against.

Run it twice, once on master and once on the branch, and set the two
outputs side by side: that is the before / after table of
``module_work/gm140/STEP2_CHECK.md``.

    python module_work/gm140/step2_section11.py            # this tree
    PYTHONPATH=<a master worktree> python <this file>      # master

It prints one line per measured value. The benchmarks whose pinned tests
pass unchanged on both trees are not recomputed here: the pins hold, and
the CHECK file says so from the test run. What is measured here is every
benchmark the step's brief expects to move, with each geometry taken from
the test module that pins it (nothing is re-entered by hand):

* Slide2 #85 with and without its support, #86 with its tiebacks, ACADS
  1(a): searches by Bishop, Spencer and GLE
  (slope_stability/tests/test_search_reinforcement.py).
* V-043 (sand fill), V-031, V-044: the pinned circles and searches.
* V-039: the yield coefficient of the #104-structure demonstration.
* The FOSM and Monte Carlo cases: B4 (Duncan 2000), V-030, V-042, V-045.
* ACADS 4 (V-028, B11): the declared family, the weak-layer search and the
  calc-package report of that search.
* The count the brief asks for: #85 without its support, 6 x 6 Spencer
  search, circles left without a rigorous solution, and where they lie.

Works on both trees: fields that exist only after the step are read with
getattr, and the old approximate solver's calls are counted on a tree that
still has it.
"""

import json
import sys
import time
import warnings

warnings.simplefilter("ignore")

from slope_stability.analysis import (analyze_slope,           # noqa: E402
                                      search_critical_surface)
from slope_stability.slip_surface import (CircularSlipSurface,  # noqa: E402
                                          PolylineSlipSurface)
from slope_stability.slices import build_slices                 # noqa: E402
import slope_stability.methods as _methods                      # noqa: E402

OUT = {}


def put(key, value):
    OUT[key] = value
    print(f"{key}: {value}", flush=True)


def fos(x, nd=4):
    return None if x is None else round(float(x), nd)


def unsolved(res):
    """Trials left without a solution, as the result counts them (None on
    a tree whose results do not count them)."""
    rec = getattr(res, "no_solution", None)
    if rec is None:
        return None
    return {f"{r['family']}/{r['method']}":
            (f"{r['n']} ({r['no_driving']} no driving)"
             if r.get("no_driving") else r["n"]) for r in rec}


def crit(res):
    c = res.critical
    if c is None:
        return None
    return {"FOS": fos(c.FOS), "solved": getattr(c, "method_solved", True),
            "used": getattr(c, "method_used", ""),
            "governs": res.governing_surface,
            "candidates": dict(res.candidates),
            "no_solution": unsolved(res)}


def timed(fn):
    t = time.time()
    out = fn()
    return out, round(time.time() - t, 1)


# ---------------------------------------------------------------------------
# Slide2 #85, #86, ACADS 1(a)
# ---------------------------------------------------------------------------

def slide85_86_acads():
    from slope_stability.tests.test_search_reinforcement import (
        _slide85, _slide86, _acads, _slide85c, _SLIDE2_85_CIRCLE)

    # The count the brief asks for. On a tree that still has the old
    # approximate solver, count its calls and where the circles lie.
    calls = []
    old = getattr(_methods, "spencer_fos_legacy", None)
    if old is not None:
        def counted(slices, slip, **kw):
            out = old(slices, slip, **kw)
            calls.append((slip.xc, slip.yc, slip.radius))
            return out
        _methods.spencer_fos_legacy = counted
    g = _slide85(anchor=False)
    res, t = timed(lambda: search_critical_surface(g, method="spencer",
                                                   nx=6, ny=6))
    if old is not None:
        _methods.spencer_fos_legacy = old
    put("85 no support, Spencer 6x6", crit(res))
    put("85 no support, Spencer 6x6, run time s", t)
    put("85 no support, Spencer 6x6, surfaces evaluated",
        res.n_surfaces_evaluated)
    put("85 no support, Spencer 6x6, circles handed to the old solver",
        len(calls) if old is not None else "none: removed")
    c = res.critical
    b = analyze_slope(g, xc=c.xc, yc=c.yc, radius=c.radius, method="bishop")
    put("85 no support, Bishop on the Spencer search's circle", fos(b.FOS))
    rb = search_critical_surface(g, method="bishop", nx=6, ny=6)
    put("85 no support, Bishop 6x6", crit(rb))

    for m in ("bishop", "spencer", "gle"):
        res, t = timed(lambda: search_critical_surface(_slide85c(),
                                                       method=m))
        put(f"85 one support (active), {m} search", crit(res))
        put(f"85 one support (active), {m} search, run time s", t)
        r = analyze_slope(_slide85c(), method=m, **_SLIDE2_85_CIRCLE)
        put(f"85 one support, {m} on Slide2's circle", fos(r.FOS))
    res = search_critical_surface(_slide85c("passive"), method="gle")
    put("85 one support (passive), gle search", crit(res))

    for m in ("bishop", "spencer", "gle"):
        res = search_critical_surface(_slide86(), method=m,
                                      x_entry_range=(0.0, 4.0),
                                      x_exit_range=(30.0, 75.0))
        put(f"86 five tiebacks, {m} search", crit(res))
    res = search_critical_surface(_slide86(tiebacks=False), method="spencer")
    put("86 no supports, spencer search", crit(res))

    for m in ("bishop", "spencer", "gle", "janbu"):
        res = search_critical_surface(_acads(), method=m)
        put(f"ACADS 1(a), {m} search", crit(res))


# ---------------------------------------------------------------------------
# V-043, V-031, V-044, V-026
# ---------------------------------------------------------------------------

def circles():
    import validation_examples.test_published_v043_slope_reinforced as v43
    import validation_examples.test_published_v031_slope as v31
    import validation_examples.test_published_v044_slope_submerged as v44

    g = v43._geom("sand")
    res = search_critical_surface(g, method="spencer")
    put("V-043 sand fill, default Spencer search (refined)", crit(res))
    for name, (xc, yc, r) in (("pinned circle", v43._SAND_CIRC),
                              ("critical circle", v43._SAND_CRITICAL)):
        one = analyze_slope(g, xc=xc, yc=yc, radius=r, method="spencer")
        put(f"V-043 sand fill, Spencer on the {name}", fos(one.FOS))
    g31 = v31._v031_geom()
    for m in ("bishop", "spencer", "gle"):
        r = analyze_slope(g31, xc=v31._V031_CIRCLE[0], yc=v31._V031_CIRCLE[1],
                          radius=v31._V031_CIRCLE[2], method=m, n_slices=50)
        put(f"V-031 pinned circle, {m}", fos(r.FOS))
    res = search_critical_surface(g31, method="spencer")
    put("V-031, Spencer search", crit(res))
    g44 = v44._geom(75)
    for m in ("bishop", "spencer", "gle"):
        r = analyze_slope(g44, xc=v44._XC, yc=v44._YC, radius=v44._R,
                          method=m, n_slices=40)
        put(f"V-044 pinned circle, {m}",
            {"FOS": fos(r.FOS), "solved": getattr(r, "method_solved", True),
             "lambda or theta": fos(r.theta_spencer if m == "spencer"
                                    else r.lambda_mp)})


# ---------------------------------------------------------------------------
# V-039 (Newmark)
# ---------------------------------------------------------------------------

def newmark():
    from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
    from slope_stability.newmark import yield_acceleration
    geom = SlopeGeometry(
        surface_points=[(0, 0), (12, 0), (30, 12), (55, 12)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=12,
                                    bottom_elevation=-20, gamma=19.0,
                                    phi=25.0, c_prime=15.0)])
    slip = CircularSlipSurface(30.0, 26.0, 24.0)
    for m in ("spencer", "bishop", "morgenstern_price", "janbu"):
        ya = yield_acceleration(geom, slip_surface=slip, method=m,
                                n_slices=40)
        put(f"V-039 structure, ky by {m}",
            {"ky": fos(ya.ky, 5), "FOS static": fos(ya.fos_static),
             "converged": ya.converged,
             "reason": getattr(ya, "reason", "")})


# ---------------------------------------------------------------------------
# FOSM and Monte Carlo: B4, V-030, V-042, V-045
# ---------------------------------------------------------------------------

def probabilistic():
    from slope_stability.probabilistic import fosm_fos, monte_carlo_fos
    import validation_examples.test_published_v030_fosm_slope as v30
    import validation_examples.test_published_v042_slope_probabilistic as v42
    import validation_examples.test_published_v045_slope_correlated as v45

    def mc_row(mc):
        return {"n": mc.n, "mean": fos(mc.fos_mean), "cov": fos(mc.fos_cov),
                "pf": fos(mc.pf, 5),
                "beta_LN": fos(mc.beta_lognormal, 3),
                "no solution": getattr(mc, "n_no_solution", None)}

    def fosm_row(fr):
        return {"F_mlv": fos(fr.fos_mlv), "cov": fos(fr.cov_f),
                "beta_LN": fos(fr.beta_lognormal, 3),
                "pf_LN": fos(fr.pf_lognormal, 5)}

    # Each test module's own calls, re-run through its test functions'
    # inputs: the module-level helpers and constants only.
    for name, mod in (("V-030", v30), ("V-042", v42), ("V-045", v45)):
        tests = [t for t in dir(mod) if t.startswith("test_")]
        ok, bad = [], []
        for t in tests:
            try:
                getattr(mod, t)()
                ok.append(t)
            except Exception as exc:            # a pin that moved
                bad.append(f"{t}: {type(exc).__name__}: "
                           f"{str(exc).splitlines()[0][:160]}")
        put(f"{name} pinned tests", {"pass": len(ok), "fail": bad})

    g = v42._geom()
    res = search_critical_surface(g, method="bishop",
                                  surface_type="entry_exit", n_trials=800,
                                  n_slices=30, seed=1)
    put("V-042 Bishop entry/exit search", crit(res))
    surf = dict(xc=v42._XC, yc=v42._YC, radius=v42._R)
    for m in ("bishop", "spencer"):
        fr = fosm_fos(g, v42._VARS, method=m, n_slices=30, **surf)
        put(f"V-042 FOSM on the pinned circle, {m}", fosm_row(fr))
        mc = monte_carlo_fos(g, v42._VARS, method=m, n=2000, seed=5,
                             n_slices=30, **surf)
        put(f"V-042 Monte Carlo (2000, seed 5), {m}", mc_row(mc))

    # B4, Duncan (2000): the test of slope_stability/tests that pins it
    import slope_stability.tests.test_probabilistic as tp
    tests = [t for t in dir(tp) if t.startswith("Test")]
    ok, bad = 0, []
    for cls_name in tests:
        cls = getattr(tp, cls_name)
        for t in [t for t in dir(cls) if t.startswith("test_")]:
            try:
                fn = getattr(cls(), t)
                if fn.__code__.co_argcount > 1:
                    continue                    # needs a fixture: by pytest
                fn()
                ok += 1
            except Exception as exc:
                bad.append(f"{cls_name}.{t}: {str(exc).splitlines()[0][:120]}")
    put("B4 and the module's probabilistic tests without fixtures",
        {"pass": ok, "fail": bad})


# ---------------------------------------------------------------------------
# ACADS 4 (V-028, B11)
# ---------------------------------------------------------------------------

def acads4():
    import validation_examples.test_published_v028_e8 as e8
    from slope_stability.gle import gle_fos, janbu_fos

    geom = e8._geom(surcharges=e8._ZONES)
    lowest, n, n_unsolved = {}, 0, {"spencer": 0, "morgenstern_price": 0,
                                    "janbu": 0}
    for pts in e8._family():
        slip = PolylineSlipSurface(points=pts)
        slices = build_slices(geom, slip, e8._FAMILY_SLICES)
        n += 1
        sp = gle_fos(slices, slip, f_interslice="constant")
        mp = gle_fos(slices, slip, f_interslice="half_sine")
        try:
            fj = janbu_fos(slices, slip)[0]
            j_ok = fj < 900.0
        except ValueError:
            fj, j_ok = None, False
        for name, value, solved in (("spencer", sp.fos, sp.converged),
                                    ("morgenstern_price", mp.fos,
                                     mp.converged),
                                    ("janbu", fj, j_ok)):
            if not solved:
                n_unsolved[name] += 1
            elif name not in lowest or value < lowest[name][0]:
                lowest[name] = (value, pts)
    put("ACADS 4 family (81 surfaces), lowest solved",
        {k: fos(v[0]) for k, v in lowest.items()})
    put("ACADS 4 family, surfaces without a solution", n_unsolved)

    kw = dict(surface_type="weak_layer", method="spencer", n_trials=700,
              n_points=6, n_slices=30, seed=3, x_entry_range=(20.0, 50.0),
              x_exit_range=(58.0, 84.0))
    res, t = timed(lambda: search_critical_surface(geom, **kw))
    c = res.critical
    put("ACADS 4 weak-layer search (seed 3, 700 trials), Spencer",
        None if c is None else
        {"FOS": fos(c.FOS), "x": [round(c.x_entry, 2), round(c.x_exit, 2)],
         "nonconverged": res.n_rejected_nonconverged,
         "no_solution": unsolved(res), "run time s": t})
    if c is not None:
        slip = PolylineSlipSurface(points=c.slip_points)
        one = analyze_slope(geom, slip_surface=slip, method="spencer",
                            compare_methods=True)
        put("ACADS 4, analyze_slope on the search's surface",
            {"spencer": fos(one.FOS),
             "solved": getattr(one, "method_solved", True),
             "used": getattr(one, "method_used", ""),
             "theta": fos(one.theta_spencer, 2),
             "janbu corrected": fos(one.FOS_janbu),
             "morgenstern_price": fos(one.FOS_morgenstern_price)})

    # the calc-package report of the same search
    try:
        from funhouse_agent.adapters.calc_package import (
            _generate_slope_report_package)
        params = {
            "surface_points": [list(p) for p in e8._SURF],
            "soil_layers": [
                {"name": "upper", "top_elevation": 40.0,
                 "bottom_elevation": 15.0, "gamma": 18.84, "phi": 20.0,
                 "c_prime": 28.5,
                 "bottom_boundary_points": [list(p) for p in e8._SEAM_TOP]},
                {"name": "weak", "top_elevation": 40.0,
                 "bottom_elevation": 15.0, "gamma": 18.84, "phi": 10.0,
                 "c_prime": 0.0,
                 "bottom_boundary_points": [list(p) for p in e8._SEAM_BOT]},
                {"name": "lower", "top_elevation": 40.0,
                 "bottom_elevation": 15.0, "gamma": 18.84, "phi": 20.0,
                 "c_prime": 28.5}],
            "gwt_points": [list(p) for p in e8._PIEZO],
            "method": "spencer", "surface_type": "weak_layer",
            "n_trials": 700, "n_points": 6, "n_slices": 30, "seed": 3,
            "x_entry_range": [20.0, 50.0], "x_exit_range": [58.0, 84.0],
            "output_path": "/tmp/gm140_step2_acads4_report.html",
            "format": "html",
        }
        out = _generate_slope_report_package(params)
        put("ACADS 4 calc-package report (no loads: the package has no "
            "surcharge zones), headline",
            {k: out.get(k) for k in ("status", "FOS", "method", "is_stable",
                                     "governing_surface", "candidates",
                                     "no_solution", "error")})
    except Exception as exc:                    # report what stopped it
        put("ACADS 4 calc-package report", f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Where the circles without a Spencer solution lie (#85 without its support)
# ---------------------------------------------------------------------------

def where85():
    """The 6 x 6 Spencer search of #85 without its support: each trial
    circle Spencer has no solution on, by reason and by where it lies. The
    slope: toe (15, 10), crest (25, 30), H = 20, a rigid base at the toe's
    level. Branch only (it reads the single-surface solver's reasons)."""
    import math
    import statistics
    import slope_stability.search as search
    if not hasattr(search, "solve_method"):
        put("85 no support, where the unsolved circles lie",
            "not on this tree")
        return
    from slope_stability.tests.test_search_reinforcement import _slide85
    real = search.solve_method
    rows = []

    def spy(slices, slip, method, *a, **k):
        sol = real(slices, slip, method, *a, **k)
        if method == "spencer" and not sol.solved:
            rows.append((slip.xc, slip.yc, slip.radius,
                         slices[0].x_left, slices[-1].x_right,
                         slices[0].z_base, max(abs(sl.alpha)
                                               for sl in slices),
                         sol.reason.split(" (")[0][:70]))
        return sol

    search.solve_method = spy
    try:
        search_critical_surface(_slide85(anchor=False), method="spencer",
                                nx=6, ny=6)
    finally:
        search.solve_method = real
    n = len(rows)
    reasons = {}
    for r in rows:
        key = r[7].split(" still differ")[0]
        reasons[key] = reasons.get(key, 0) + 1
    q = lambda v: [round(x, 2) for x in (min(v), statistics.median(v),
                                         max(v))]
    put("85 no support, Spencer solves without a solution (all families)",
        n)
    put("85 no support, unsolved: reasons", reasons)
    # (the ground starts at the toe, x = 15, and rises to the crest at 25)
    put("85 no support, unsolved: lower end on the face above the toe "
        "(x > 15.05)", sum(1 for r in rows if r[3] > 15.05))
    put("85 no support, unsolved: lower end at the toe",
        sum(1 for r in rows if r[3] <= 15.05))
    put("85 no support, unsolved: lower end's height above the toe, in H "
        "(min, median, max)",
        q([max(r[3] - 15.0, 0.0) * 2.0 / 20.0 for r in rows]))
    put("85 no support, unsolved: centre height above the crest, in H "
        "(min, median, max)", q([(r[1] - 30.0) / 20.0 for r in rows]))
    put("85 no support, unsolved: upper end behind the crest, in H "
        "(min, median, max)", q([(r[4] - 25.0) / 20.0 for r in rows]))
    put("85 no support, unsolved: steepest base, degrees "
        "(min, median, max)", q([math.degrees(r[6]) for r in rows]))


def main(argv):
    parts = {"85": slide85_86_acads, "circles": circles, "newmark": newmark,
             "prob": probabilistic, "acads4": acads4, "where85": where85}
    chosen = [a for a in argv if a in parts] or list(parts)
    for name in chosen:
        print(f"--- {name}", flush=True)
        parts[name]()
    for a in argv:
        if a.endswith(".json"):
            with open(a, "w") as fh:
                json.dump(OUT, fh, indent=1, default=str)


if __name__ == "__main__":
    main(sys.argv[1:])
