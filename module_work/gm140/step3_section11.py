"""GM-140 step 3: the benchmarks of the account's section 11 that this
step's brief expects to move and ``step2_section11.py`` does not measure.

``step2_section11.py`` is run unchanged for its own rows (Slide2 #85 and
#86, ACADS 1(a), V-043, V-031, V-044, V-039, the probabilistic cases and
ACADS 4). This script adds the rows for step 3: the searches that left
surfaces out for m_alpha, the toe-circle family's below-base test, and the
flags a search's governing surface carries.

    PYTHONPATH=. python module_work/gm140/step3_section11.py          # this tree
    PYTHONPATH=<a master worktree> python <this file>                 # master

One line per measured value. It runs on both trees: the fields and options
that exist only after the step are read with getattr and passed only where
the tree takes them. Each geometry is taken from the test module that pins
it (nothing is re-entered by hand).
"""
import inspect
import sys
import warnings

warnings.simplefilter("ignore")

from slope_stability.analysis import (analyze_slope,           # noqa: E402
                                      search_critical_surface)

NEW = "low_m_alpha" in inspect.signature(search_critical_surface).parameters


def show(name, res):
    c = res.critical
    out = {
        "FOS": None if c is None else round(c.FOS, 4),
        "governs": res.governing_surface,
        "candidates": dict(res.candidates),
        "left out for m_alpha": res.n_rejected_m_alpha,
        "flags": getattr(res, "critical_flags", None),
        "lowest unflagged": (getattr(res, "lowest_unflagged", None)
                             or {}).get("FOS"),
        "flagged trials": (sum(r["n_flagged"] for r in res.trial_flags)
                           if hasattr(res, "trial_flags") else None),
    }
    if c is not None and getattr(c, "FOS_bishop", None) is not None:
        out["Bishop / OMS on it"] = (
            round(c.FOS_bishop, 4),
            None if c.FOS_fellenius is None else round(c.FOS_fellenius, 4))
    print(f"{name}: {out}", flush=True)
    return res


def search(name, geom, **kw):
    """The search as given, then (on a tree with the option) the same
    search with surfaces of m_alpha below 0.2 left out, the rule before
    step 3."""
    show(name, search_critical_surface(geom, **kw))
    if NEW:
        show(name + " [low_m_alpha='refuse']",
             search_critical_surface(geom, low_m_alpha="refuse", **kw))


def slide85():
    from slope_stability.tests.test_search_reinforcement import _slide85c
    search("#85 one support, active, Bishop", _slide85c(), method="bishop")
    search("#85 one support, passive, Bishop", _slide85c("passive"),
           method="bishop")
    for convention in ("active", "passive"):
        for rule in ("frictional", "every_slice"):
            search(f"#85 {convention}, reverse curvature discarded, Bishop, "
                   f"m_alpha_rule {rule}",
                   _slide85c(convention, reverse_curvature="discard"),
                   method="bishop", m_alpha_rule=rule)


def circles():
    from slope_stability.tests.test_search_supplements import _toe_geom
    from slope_stability.tests.test_m_alpha import (
        _vertical_cut, _duncan_ex4, _weak_over_rock)
    from slope_stability.tests.test_duncan_verification import \
        TestDuncanExample1
    from slope_stability.tests.test_ground_crossings import _clay85
    from slope_stability.tests.test_search_modern import _acads_geom
    import validation_examples.test_published_v031_slope as v31
    import validation_examples.test_published_v042_slope_probabilistic as v42
    search("V-054 steep phi = 0 slope, toe circle, Bishop, 40 slices",
           _toe_geom(), method="bishop", n_slices=40)
    search("V-032 (Slide2 #61), entry/exit, Bishop", v31._v032_geom(),
           method="bishop", surface_type="entry_exit", nx=10, ny=10,
           n_slices=30, x_entry_range=(0.0, 6.0), x_exit_range=(6.0, 20.0))
    search("V-042 (Hassan and Wolff), entry/exit, Bishop", v42._geom(),
           method="bishop", surface_type="entry_exit", n_trials=800,
           n_slices=30, seed=1)
    ex1 = TestDuncanExample1()
    for flat, d in ((4.0, 2), (4.0, 3)):
        g, _, _ = ex1._long_geom(flat, d)
        search(f"Duncan Ex. 1, D = {d}, {flat:g} H each side, Bishop", g,
               method="bishop")
    search("Vertical cut in undrained clay, Bishop", _vertical_cut(),
           method="bishop")
    search("Vertical cut in undrained clay, Bishop, every slice",
           _vertical_cut(), method="bishop", m_alpha_rule="every_slice")
    search("Duncan Ex. 4, Janbu, 30 slices", _duncan_ex4(), method="janbu",
           n_slices=30)
    search("B14 weak layer over rock, default search, Bishop",
           _weak_over_rock(), method="bishop")
    search("GM-130: #85's clay face without its support, Janbu", _clay85(),
           method="janbu")
    search("ACADS 1(a), random noncircular, 200 trials, seed 42, Spencer",
           _acads_geom(), method="spencer", surface_type="noncircular",
           n_trials=200, n_points=6, n_slices=25, seed=42,
           x_entry_range=(20.0, 32.0), x_exit_range=(46.0, 68.0))


def single():
    """Given circles: Bishop's value, the Ordinary Method of Slices value
    for the same circle, and what is reported."""
    from slope_stability.tests.test_own_solution_or_no_number import (
        _fill_slope, _NO_SPENCER_CIRCLE)
    import validation_examples.test_published_v029_slope as v29
    b = analyze_slope(_fill_slope(c=40.0), method="bishop",
                      **_NO_SPENCER_CIRCLE)
    f = analyze_slope(_fill_slope(c=40.0), method="fellenius",
                      **_NO_SPENCER_CIRCLE)
    print("made-up fill slope (c' 40), circle (8, 12) R 12: reported "
          f"{b.FOS:.4f}; Bishop "
          f"{getattr(b, 'FOS_bishop', None) or b.FOS:.4f}; OMS {f.FOS:.4f}; "
          f"flags {getattr(b, 'flags', None)}", flush=True)
    GAMMA_W = 9.81

    def phreatic(x):
        return 2.0 + 6.0 * max(min(x / 60.0, 1.0), 0.0)
    ppp = [(float(x), float(z), GAMMA_W * max(phreatic(x) - z, 0.0))
           for x in range(-5, 66, 5) for z in [-6, -3, 0, 2, 4, 6, 8, 10]]
    kw = dict(xc=v29._A5_XC, yc=v29._A5_YC, radius=v29._A5_R,
              method="bishop", n_slices=40)
    g = analyze_slope(v29._a5_geom(ppp), **kw)
    line = (f"V-029 made-up pore-pressure grid on the ACADS 5 slope, "
            f"Bishop: {g.FOS:.4f}; flags {getattr(g, 'flags', None)}")
    if NEW:
        z = analyze_slope(v29._a5_geom(ppp), negative_friction_term="zero",
                          **kw)
        line += f"; with negative_friction_term='zero' {z.FOS:.4f}"
    print(line, flush=True)


def main(argv):
    parts = {"slide85": slide85, "circles": circles, "single": single}
    for name in ([a for a in argv if a in parts] or list(parts)):
        parts[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
