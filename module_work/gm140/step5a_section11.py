"""GM-140 step 5a: the published validation set (the account's section 11),
measured before and after, with nothing re-entered by hand.

``step2_section11.py`` (run unchanged for this step too) recomputes the
benchmarks step 2 moved. This script covers the rest of section 11 a
different way: it runs the tests that pin the set and writes down every
factor of safety those tests compute, so the same run on master and on
the branch gives the before / after table of ``STEP5A_CHECK.md``.

It is a pytest plugin. Every call a TEST makes to one of the module's
entry points (``analyze_slope``, ``compare_methods_table``,
``search_critical_surface``, ``rapid_drawdown_fos``,
``search_rapid_drawdown``, ``yield_acceleration``, ``fosm_fos``,
``monte_carlo_fos``, and the solvers ``fellenius_fos``, ``bishop_fos``,
``gle_fos``, ``janbu_fos``) is recorded as one line: the test, the call's
number within it, the function and the value. Calls the module makes
inside itself are not recorded. Nothing is changed in what the tests
compute or assert.

    # this tree (the branch)
    python module_work/gm140/step5a_section11.py run  OUT_DIR
    # master: a worktree of master first on the path
    PYTHONPATH=<master worktree> python <this file> run OUT_DIR_MASTER \\
        --rootdir <master worktree>
    # the table: every recorded value that differs
    python module_work/gm140/step5a_section11.py diff OUT_DIR_MASTER OUT_DIR

``run`` passes any further arguments to pytest (``-n 4`` works: each
worker writes its own file).
"""

import json
import os
import sys

_OUT_ENV = "GM140_STEP5A_OUT"

# (module, name): what a test may call
_ENTRY = [
    ("slope_stability.analysis", "analyze_slope"),
    ("slope_stability.analysis", "compare_methods_table"),
    ("slope_stability.analysis", "search_critical_surface"),
    ("slope_stability.analysis", "rapid_drawdown_fos"),
    ("slope_stability.analysis", "search_rapid_drawdown"),
    ("slope_stability.rapid_drawdown", "rapid_drawdown_fos"),
    ("slope_stability.rapid_drawdown", "search_rapid_drawdown"),
    ("slope_stability.newmark", "yield_acceleration"),
    ("slope_stability.probabilistic", "fosm_fos"),
    ("slope_stability.probabilistic", "monte_carlo_fos"),
    ("slope_stability.methods", "fellenius_fos"),
    ("slope_stability.methods", "bishop_fos"),
    ("slope_stability.gle", "gle_fos"),
    ("slope_stability.gle", "janbu_fos"),
    ("slope_stability", "analyze_slope"),
    ("slope_stability", "search_critical_surface"),
]

_state = {"item": None, "k": 0, "depth": 0, "fh": None}


def _r(x, nd=4):
    try:
        return None if x is None else round(float(x), nd)
    except (TypeError, ValueError):
        return None


def _value(out):
    """The numbers of a result, small enough for one line."""
    if isinstance(out, (int, float)):
        return _r(out)
    if isinstance(out, tuple):
        return [_value(o) for o in out[:3]]
    if isinstance(out, dict):
        if "rows" in out:
            return {r["method"]: r["FOS"] for r in out["rows"]}
        return {k: _r(v) for k, v in out.items()
                if isinstance(v, (int, float))}
    d = {}
    for name in ("FOS", "fos", "FOS_fellenius", "FOS_bishop", "FOS_spencer",
                 "FOS_morgenstern_price", "FOS_janbu", "stage1_fos", "ky",
                 "beta", "mean_FOS", "FOS_mean", "pf", "probability_of_failure",
                 "n_drained_substituted"):
        v = getattr(out, name, None)
        if isinstance(v, (int, float)):
            d[name] = _r(v)
    crit = getattr(out, "critical", None)
    if crit is not None:
        d["critical_FOS"] = _r(getattr(crit, "FOS", None))
        d["governs"] = getattr(out, "governing_surface", None)
    dd = getattr(out, "drawdown", None)
    if dd is not None:
        d["stage1_fos"] = _r(getattr(dd, "stage1_fos", None))
    return d


def _wrap(fn, label):
    def recorded(*a, **kw):
        if _state["item"] is None or _state["depth"]:
            return fn(*a, **kw)
        _state["depth"] += 1
        try:
            out = fn(*a, **kw)
        except Exception as e:                      # noqa: BLE001
            _line(label, {"raised": type(e).__name__})
            raise
        finally:
            _state["depth"] -= 1
        _line(label, _value(out))
        return out
    recorded.__name__ = getattr(fn, "__name__", label)
    recorded.__doc__ = getattr(fn, "__doc__", None)
    recorded.__wrapped__ = fn
    return recorded


def _line(label, value):
    _state["k"] += 1
    fh = _state["fh"]
    if fh is None:
        os.makedirs(os.environ[_OUT_ENV], exist_ok=True)
        fh = _state["fh"] = open(os.path.join(
            os.environ[_OUT_ENV], f"calls_{os.getpid()}.jsonl"), "a")
    fh.write(json.dumps({"test": _state["item"], "k": _state["k"],
                         "fn": label, "value": value}) + "\n")
    fh.flush()


def pytest_configure(config):
    if not os.environ.get(_OUT_ENV):
        return
    import importlib
    wrappers = {}
    for mod, name in _ENTRY:
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        fn = getattr(m, name, None)
        if fn is None or hasattr(fn, "__wrapped_by_gm140__"):
            continue
        w = wrappers.get(id(fn))
        if w is None:
            w = wrappers[id(fn)] = _wrap(fn, name)
            w.__wrapped_by_gm140__ = True
        setattr(m, name, w)


def pytest_runtest_setup(item):
    _state["item"] = item.nodeid
    _state["k"] = 0


def pytest_runtest_teardown(item):
    _state["item"] = None


# ---------------------------------------------------------------------------

def _tests(root):
    import glob
    files = [os.path.join(root, "slope_stability", "tests")]
    for f in sorted(glob.glob(os.path.join(
            root, "validation_examples", "test_published_*.py"))):
        with open(f, encoding="utf-8") as fh:
            if "slope_stability" in fh.read():
                files.append(f)
    return files


def _load(d):
    out = {}
    for name in sorted(os.listdir(d)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(d, name)) as fh:
            for line in fh:
                rec = json.loads(line)
                test = rec["test"].replace("\\", "/")
                out[(test, rec["k"], rec["fn"])] = rec["value"]
    return out


def main(argv):
    if len(argv) >= 2 and argv[0] == "run":
        out = os.path.abspath(argv[1])
        rest = argv[2:]
        root = os.getcwd()
        if "--rootdir" in rest:
            root = os.path.abspath(rest[rest.index("--rootdir") + 1])
        os.environ[_OUT_ENV] = out
        here = os.path.dirname(os.path.abspath(__file__))
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [p for p in (os.environ.get("PYTHONPATH"), here) if p])
        sys.path.insert(0, here)
        import pytest
        return pytest.main(["-q", "-p", "step5a_section11",
                            "-p", "no:cacheprovider", *rest, *_tests(root)])
    if len(argv) == 3 and argv[0] == "diff":
        before, after = _load(argv[1]), _load(argv[2])
        keys = sorted(set(before) & set(after))
        only = len(set(before) ^ set(after))
        n = 0
        for key in keys:
            b, a = before[key], after[key]
            if b != a:
                n += 1
                print(f"{key[0]} #{key[1]} {key[2]}\n    before {b}\n"
                      f"    after  {a}")
        print(f"\n{len(before)} values before, {len(after)} after, "
              f"{len(keys)} in both, {n} of those differ; {only} calls are "
              f"in one run only (tests added or changed by the step)")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
