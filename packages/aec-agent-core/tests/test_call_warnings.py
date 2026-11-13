"""A Python warning raised inside a method reaches the agent (BACKLOG GM-316;
``aec_agent_core.call_warnings``). Made-up modules and values only."""

import sys
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

def warn_same(params):
    for _ in range(params["n"]):
        warnings.warn("the same note")
    return {"value": 9.0}
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


def test_so_far_gives_the_calls_warnings_while_it_runs():
    """Review B4: a calc package reads the warnings raised so far in its
    call, to print them before it is written; outside a call, none."""
    g = _code(f"{PKG}.package", """
def package(params):
    warnings.warn("pile below the profile")
    return {"printed": SO_FAR()}
""")
    g["SO_FAR"] = CW.so_far
    m = types.SimpleNamespace(
        METHOD_REGISTRY={"package": g["package"]},
        METHOD_INFO={"package": {"parameters": {}, "open_parameters": True}})
    r = run_method(m, "fake", "package", {})
    assert r["printed"] == ["pile below the profile"]
    assert r["warnings"] == ["pile below the profile"]
    assert CW.so_far() == []


def test_two_calls_in_a_row_both_carry_it():
    # pytest's own default action shows a warning once per location; the
    # second call must still get it.
    assert _run("warn_once")["warnings"] == ["tip below the profile: 1"]
    assert _run("warn_once")["warnings"] == ["tip below the profile: 1"]


def test_a_numpy_warning_repeated_in_later_calls_still_reaches_them():
    """The hook's path (raised from C): the registry key is taken out so
    the next call sees it again."""
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        for _ in range(3):
            assert _run("overflow")["warnings"] == [
                "overflow encountered in exp"]


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


def test_a_catch_warnings_block_does_not_hide_a_packages_own_warning():
    """An app package's warning is recorded before any filter acts (review
    A1), so a package cannot silence its own warning with catch_warnings:
    a trial calculation must not raise it (drilled_shaft, lateral_pile).
    The layout_ink lock block still runs, and the hook is still there."""
    from aec_agent_core.charts import layout_ink
    src = _code(f"{PKG}.ink", """
def both(params):
    with LOCK, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.warn("inside the catch")
    warnings.warn("after it")
    return {"value": 8.0}
""")
    src["LOCK"] = layout_ink._WARNINGS_LOCK
    m = types.SimpleNamespace(
        METHOD_REGISTRY={"both": src["both"]},
        METHOD_INFO={"both": {"parameters": {}, "open_parameters": True}})
    r = run_method(m, "fake", "both", {})
    assert r["warnings"] == ["inside the catch", "after it"]
    assert getattr(warnings.showwarning, "_aec_call_warnings", False)


#: The filter two interleaved ``catch_warnings`` blocks leave behind.
BLANKET = ("ignore", None, Warning, None, 0)


def test_an_ignore_everything_filter_in_front_does_not_stop_it():
    """Review A1: with an "ignore" filter for every warning at the front of
    warnings.filters, a package's warning is still in each call's result,
    on the first call and the hundredth."""
    with warnings.catch_warnings():        # puts the filters back after
        warnings.simplefilter("ignore")
        assert warnings.filters[0] == BLANKET
        for i in range(100):
            r = _run("warn_once", {"x": i % 3})
            assert r["warnings"] == [f"tip below the profile: {i % 3}"], i
        r = _run("warn_then_fail")
        assert r["warnings"] == ["layers stop short"]


def test_the_filter_two_interleaved_blocks_leave_does_not_stop_it():
    """Review A1, p2_leak: thread A enters catch_warnings, B enters, A
    exits, B exits; B puts back A's copy, the blanket ignore filter left at
    the front for good. Calls made after still carry their warnings."""
    a_in, b_in, a_out = (threading.Event() for _ in range(3))

    def a():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a_in.set()
            b_in.wait(10)
        a_out.set()

    def b():
        a_in.wait(10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b_in.set()
            a_out.wait(10)

    with warnings.catch_warnings():        # puts the filters back after
        ts = [threading.Thread(target=a), threading.Thread(target=b)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(20)
        assert warnings.filters[0] == BLANKET   # left behind
        for i in range(100):
            assert _run("warn_once")["warnings"] == [
                "tip below the profile: 1"], i


def test_outside_a_call_a_warning_is_shown_as_before():
    """Outside a call the wrapper only hands on: shown once per location
    under the default action, at the line it names (stacklevel too)."""
    src = _code("gm316outside.mod", """
def plain():
    warnings.warn("plain")

def up():
    warnings.warn("attributed to the caller", stacklevel=2)

def twice():
    for _ in range(2):
        plain()
    up()
""")
    shown = []

    def recorder(message, category, filename, lineno, file=None,
                 line=None):
        shown.append((str(message), filename, lineno))

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        warnings.showwarning = recorder
        CW.ensure_hook()                   # our hook over the recorder
        assert getattr(warnings.warn, "_aec_call_warnings", False)
        src["twice"]()
    assert shown == [("plain", "<gm316outside.mod>", 3),
                     ("attributed to the caller", "<gm316outside.mod>", 11)]


@pytest.mark.skipif(sys.version_info < (3, 12),
                    reason="skip_file_prefixes is new in Python 3.12")
def test_skip_file_prefixes_still_names_the_same_frame():
    lib = _code("gm316skip.lib", """
def warn_here():
    warnings.warn("skipped over", skip_file_prefixes=("<gm316skip",))
""")
    caller = _code("gm316outside.caller", """
def go():
    LIB()
""")
    caller["LIB"] = lib["warn_here"]
    shown = []
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = (
            lambda m, c, f, n, file=None, line=None: shown.append((f, n)))
        CW.ensure_hook()
        caller["go"]()
        with CW.collecting():
            caller["go"]()
    assert shown == [("<gm316outside.caller>", 3)] * 2


def test_within_a_call_a_recorder_sees_every_occurrence():
    """Review C11: a package's warning repeated in one call reaches an
    "always" recorder each time (pytest.warns), and the result once."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = _run("warn_same", {"n": 4})
    assert r["warnings"] == ["the same note"]
    assert [str(w.message) for w in rec].count("the same note") == 4


def test_a_warning_seen_outside_a_call_first_still_reaches_the_next_call():
    """Review C6: the registry has the same text at the same line from a
    direct call; a call through run_method still records it."""
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        CALC["warn_once"]({"x": 4})        # outside any call
        assert _run("warn_once", {"x": 4})["warnings"] == [
            "tip below the profile: 4"]


def test_logging_capture_warnings_round_trip(caplog):
    """Review C17: logging.captureWarnings over the hook; the result and
    logging both get the warning; captureWarnings(False) puts back what
    was there."""
    import logging
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        CW.ensure_hook()
        before = warnings.showwarning
        logging.captureWarnings(True)
        try:
            with caplog.at_level(logging.WARNING, logger="py.warnings"):
                r = _run("warn_once", {"x": 7})
            assert r["warnings"] == ["tip below the profile: 7"]
            assert any("tip below the profile: 7" in m
                       for m in caplog.messages)
        finally:
            logging.captureWarnings(False)
        assert warnings.showwarning is before


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
