"""A Python warning raised inside a method reaches the agent (BACKLOG GM-316;
``aec_agent_core.call_warnings``). Made-up modules and values only."""

import threading
import types
import warnings

import numpy as np
import pytest

from aec_agent_core import call_warnings as CW
from aec_agent_core.hooks import clear_hook, get_hook, hooks, set_hook
from aec_agent_core.method_call import run_method

PKG = "gm316calc"


def _code(name: str, source: str) -> dict:
    """The globals of a made-up module ``name`` holding ``source``'s
    functions (so a warning they raise is attributed to ``name``)."""
    g = {"__name__": name, "warnings": warnings, "np": np}
    exec(compile(source, f"<{name}>", "exec"), g)
    return g


CALC = _code(f"{PKG}.core", """
def warn_once(params):
    warnings.warn("tip below the profile: " + str(params.get("x", 1)))
    return {"value": 1.0}

def warn_many(params):
    for i in range(params["n"]):
        warnings.warn(f"note {i}")
    return {"value": 2.0}

def warn_deprecated(params):
    warnings.warn("old name", DeprecationWarning)
    warnings.warn("future", FutureWarning)
    return {"value": 3.0}

def overflow(params):
    with np.errstate(over="warn"):
        v = float(np.exp(np.array([1000.0]))[0])
    return {"value": v}

def warn_then_fail(params):
    warnings.warn("layers stop short")
    raise ValueError("no soil at the toe")

def as_list(params):
    warnings.warn("a list result")
    return [1, 2, 3]

def keeps_own(params):
    warnings.warn("own text")
    warnings.warn("python only")
    return {"value": 4.0, "warnings": ["own text", "module said"]}

def call_other(params):
    return OTHER(params)

def call_plotting(params):
    PLOT(params)
    return {"value": 5.0}

def overshoot(params):
    warnings.warn("stacklevel too high", stacklevel=3)
    return {"value": 6.0}
""")

OTHER_MOD = _code("gm316thirdparty.lib", """
def other(params):
    warnings.warn("third-party chatter")
    with np.errstate(divide="warn"):
        np.log(np.array([0.0]))
    return {"value": 7.0}
""")
PLOT_MOD = _code("matplotlib.gm316fake", """
def plot(params):
    warnings.warn("tight layout not applied", stacklevel=2)
""")
CALC["OTHER"] = OTHER_MOD["other"]
CALC["PLOT"] = PLOT_MOD["plot"]


def _mod(*names):
    m = types.SimpleNamespace()
    m.METHOD_REGISTRY = {n: CALC[n] for n in names}
    m.METHOD_INFO = {n: {"parameters": {}, "open_parameters": True}
                     for n in names}
    return m


@pytest.fixture(autouse=True)
def _packages():
    """The made-up package stands for an app's calculation packages."""
    before = hooks().get("method_call.warning_packages")
    set_hook("method_call.warning_packages", lambda: (PKG,))
    yield
    if before is None:
        clear_hook("method_call.warning_packages")
    else:
        set_hook("method_call.warning_packages", before)


def _run(name, params=None):
    return run_method(_mod(name), "fake", name, params or {})


def test_a_warning_raised_in_a_handler_is_in_the_result():
    r = _run("warn_once")
    assert r["warnings"] == ["tip below the profile: 1"]
    assert r["value"] == 1.0
    assert list(r)[0] == "warnings"      # first: kept by a cut for size


def test_two_calls_in_a_row_both_carry_it():
    # pytest's own default action shows a warning once per location; the
    # second call must still get it.
    assert _run("warn_once")["warnings"] == ["tip below the profile: 1"]
    assert _run("warn_once")["warnings"] == ["tip below the profile: 1"]


def test_calls_at_once_in_threads_do_not_see_each_others():
    barrier = threading.Barrier(2)
    CALC["BARRIER"] = barrier
    src = _code(f"{PKG}.threads", """
def slow(params):
    BARRIER.wait()
    warnings.warn("from " + params["who"])
    BARRIER.wait()
    return {"who": params["who"]}
""")
    src["BARRIER"] = barrier
    m = types.SimpleNamespace(
        METHOD_REGISTRY={"slow": src["slow"]},
        METHOD_INFO={"slow": {"parameters": {}, "open_parameters": True}})
    out = {}

    def go(who):
        out[who] = run_method(m, "fake", "slow", {"who": who})

    ts = [threading.Thread(target=go, args=(w,)) for w in ("a", "b")]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert out["a"]["warnings"] == ["from a"]
    assert out["b"]["warnings"] == ["from b"]


def test_a_warning_outside_a_call_behaves_as_before():
    _run("warn_once")                      # the hook is installed
    assert getattr(warnings.showwarning, "_aec_call_warnings", False)
    seen = []

    def previous(message, category, filename, lineno, file=None, line=None):
        seen.append(str(message))

    hook = CW._make_hook(previous)
    hook(UserWarning("outside"), UserWarning, "x.py", 1)
    assert seen == ["outside"]
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        CALC["warn_once"]({"x": 9})
    assert [str(w.message) for w in rec] == ["tip below the profile: 9"]


def test_noise_is_not_added():
    r = _run("warn_deprecated")
    assert "warnings" not in r
    r = _run("call_plotting")
    assert "warnings" not in r


def test_a_numpy_floating_point_warning_reaches_the_agent_wherever_raised():
    r = _run("overflow")
    assert r["warnings"] == ["overflow encountered in exp"]
    r = _run("call_other")                 # a third-party library's code
    assert r["warnings"] == ["divide by zero encountered in log"]


def test_a_stacklevel_one_too_high_still_reaches_the_agent():
    assert _run("overshoot")["warnings"] == ["stacklevel too high"]


def test_an_error_keeps_the_warnings_raised_before_it():
    r = _run("warn_then_fail")
    assert r["error"].startswith("ValueError")
    assert r["warnings"] == ["layers stop short"]


def test_a_result_that_is_not_a_dict_is_returned_unchanged():
    assert _run("as_list") == [1, 2, 3]


def test_the_modules_own_warnings_come_first_without_duplicates():
    r = _run("keeps_own")
    assert r["warnings"] == ["own text", "module said", "python only"]


def test_at_most_twenty_then_a_count():
    r = _run("warn_many", {"n": 25})
    assert r["warnings"][:20] == [f"note {i}" for i in range(20)]
    assert r["warnings"][20:] == ["and 5 more"]


def test_no_file_path_or_line_number_in_the_text():
    text = _run("warn_once")["warnings"][0]
    assert "<gm316calc" not in text and "line" not in text


def test_the_lock_based_catch_in_layout_ink_still_works():
    from aec_agent_core.charts import layout_ink
    src = _code(f"{PKG}.ink", """
def both(params):
    with LOCK, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn("hidden by the catch")
    warnings.warn("shown after it")
    return {"value": 8.0}
""")
    src["LOCK"] = layout_ink._WARNINGS_LOCK
    m = types.SimpleNamespace(
        METHOD_REGISTRY={"both": src["both"]},
        METHOD_INFO={"both": {"parameters": {}, "open_parameters": True}})
    r = run_method(m, "fake", "both", {})
    assert r["warnings"] == ["shown after it"]
    assert getattr(warnings.showwarning, "_aec_call_warnings", False)


def test_a_call_inside_a_call_passes_its_warnings_up():
    with CW.collecting() as outer:
        r = _run("warn_once")
    assert r["warnings"] == ["tip below the profile: 1"]
    assert outer.texts == ["tip below the profile: 1"]


def test_without_the_hook_only_the_cores_own_frame_counts():
    clear_hook("method_call.warning_packages")
    assert get_hook("method_call.warning_packages") is None
    assert "warnings" not in _run("warn_once")
    assert _run("overflow")["warnings"] == ["overflow encountered in exp"]
