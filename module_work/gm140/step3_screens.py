"""GM-140 step 3, item 5: the noncircular screens, measured.

The module's noncircular screens (the account's D22) are numerical
stand-ins not found in print: slice count, span, agreement between moment
axes, agreement between slicings, and the refusal of a factor of safety
below 1.5 on a jagged surface. The brief asks: on every noncircular case of
the account's section 11, per screen, how many trials does it refuse, and
how many of those do the printed checks (EM 1110-2-1902 C-10a; Whitman and
Bailey 1967, p. 487) flag? A screen that refuses only surfaces the printed
checks flag is removed; one that refuses surfaces they pass stays.

    PYTHONPATH=. python module_work/gm140/step3_screens.py [case ...]

The three screens applied after the solve are counted by the search itself
(``SearchResult.screen_refusals``: ``n`` and ``n_flagged``). The two
applied before any solve (slice count, span) are solved here, by Spencer's
method, to see what the printed checks say of them: this script wraps the
screen's own function and changes nothing in the search.

Each geometry is taken from the test module that pins it.
"""
import json
import sys
import time
import warnings

warnings.simplefilter("ignore")

import slope_stability.search as S                              # noqa: E402
from slope_stability.acceptability import surface_findings, flags  # noqa: E402
from slope_stability.analysis import search_critical_surface     # noqa: E402
from slope_stability.solution import solve_method                # noqa: E402

PRE = {}      # screen -> {"n", "solved", "flagged", "unflagged_fos": [...]}


def _wrap():
    orig = S._noncircular_admissible

    def wrapped(slip, slices, geom, n_slices_requested, window_span=None):
        ok = orig(slip, slices, geom, n_slices_requested, window_span)
        if not ok:
            screen = ("slice_count"
                      if len(slices) < max(5, n_slices_requested // 3)
                      else "span")
            rec = PRE.setdefault(screen, {"n": 0, "solved": 0, "flagged": 0,
                                          "unflagged_fos": []})
            rec["n"] += 1
            try:
                reinf = S.trial_reinforcement(geom, slip)
                sol = solve_method(slices, slip, "spencer", "half_sine",
                                   1e-4, reinf)
            except (ValueError, ZeroDivisionError, RuntimeError):
                sol = None
            if sol is not None and sol.solved:
                rec["solved"] += 1
                if flags(surface_findings(geom, slip, slices, sol)):
                    rec["flagged"] += 1
                else:
                    rec["unflagged_fos"].append(round(sol.fos, 3))
        return ok
    S._noncircular_admissible = wrapped


def run(name, geom, **kw):
    PRE.clear()
    t = time.time()
    res = search_critical_surface(geom, **kw)
    c = res.critical
    pre = {k: {"n": v["n"], "solved": v["solved"], "flagged": v["flagged"],
               "unflagged": len(v["unflagged_fos"]),
               "unflagged_lowest": (min(v["unflagged_fos"])
                                    if v["unflagged_fos"] else None)}
           for k, v in PRE.items()}
    out = {
        "FOS": None if c is None else round(c.FOS, 4),
        "critical_flags": res.critical_flags,
        "lowest_unflagged": (res.lowest_unflagged or {}).get("FOS"),
        "n_evaluated": res.n_surfaces_evaluated,
        "no_solution": sum(r["n"] for r in res.no_solution),
        "screens_after_solve": res.screen_refusals,
        "screens_before_solve": pre,
        "refused_lowest": (res.refused_lowest or {}).get("FOS"),
        "trial_flags": [(r["family"], r["method"], r["n_scored"],
                         r["n_flagged"], r["by_check"])
                        for r in res.trial_flags],
        "s": round(time.time() - t, 1),
    }
    print(name + ": " + json.dumps(out), flush=True)
    return out


def cases():
    from validation_examples.test_published_v028_e8 import _geom as v028, \
        _ZONES
    from validation_examples.test_published_v028_slope import _v028_geom
    from validation_examples.test_published_v043_slope_reinforced import \
        _geom as v043
    from slope_stability.tests.test_m_alpha import _w11, _weak_over_rock
    win28 = dict(x_entry_range=(20.0, 50.0), x_exit_range=(58.0, 84.0))
    out = {
        "V-028 ACADS 4, weak-layer search, loads, seed 3": lambda: run(
            "V-028 ACADS 4, weak-layer search, loads, seed 3",
            v028(surcharges=_ZONES), surface_type="weak_layer",
            method="spencer", n_trials=700, n_points=6, n_slices=30, seed=3,
            **win28),
        "V-028 ACADS 4, weak-layer search, no loads, 40 slices": lambda: run(
            "V-028 ACADS 4, weak-layer search, no loads, 40 slices",
            _v028_geom(), surface_type="weak_layer", method="spencer",
            n_trials=700, n_points=6, n_slices=40, seed=3, **win28),
        "V-028 ACADS 4, random noncircular, loads, seed 3": lambda: run(
            "V-028 ACADS 4, random noncircular, loads, seed 3",
            v028(surcharges=_ZONES), surface_type="noncircular",
            method="spencer", n_trials=700, n_points=6, n_slices=30, seed=3,
            **win28),
        "B14 weak layer over rock, DE directed, seed 1": lambda: run(
            "B14 weak layer over rock, DE directed, seed 1",
            _weak_over_rock(), surface_type="noncircular_de",
            method="spencer", x_entry_range=(25.5, 28.0),
            x_exit_range=(29.0, 40.0), n_points=7, seed=1),
        "B14 weak layer over rock, random default windows, seed 1":
            lambda: run(
                "B14 weak layer over rock, random default windows, seed 1",
                _weak_over_rock(), surface_type="noncircular",
                method="spencer", n_trials=500, seed=1),
        "w11 cut over sandstone, DE, seed 3": lambda: run(
            "w11 cut over sandstone, DE, seed 3", _w11(),
            surface_type="noncircular_de", method="spencer",
            x_entry_range=(20, 30), x_exit_range=(28, 60), n_slices=40,
            n_points=5, seed=3),
    }
    for seed in (1, 2, 3):
        out[f"V-043 sand fill, DE, seed {seed}"] = (
            lambda seed=seed: run(
                f"V-043 sand fill, DE, seed {seed}", v043("sand"),
                method="spencer", surface_type="noncircular_de",
                n_slices=30, seed=seed))
    for seed in (1, 2):
        out[f"V-043 sand fill, random 2000, seed {seed}"] = (
            lambda seed=seed: run(
                f"V-043 sand fill, random 2000, seed {seed}", v043("sand"),
                method="spencer", surface_type="noncircular", n_trials=2000,
                n_slices=30, seed=seed))
    return out


def main(argv):
    _wrap()
    all_cases = cases()
    chosen = [k for k in all_cases if any(a in k for a in argv)] \
        if argv else list(all_cases)
    for k in chosen:
        all_cases[k]()


if __name__ == "__main__":
    main(sys.argv[1:])
