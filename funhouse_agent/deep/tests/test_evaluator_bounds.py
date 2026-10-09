"""The three safe expression evaluators refuse a huge integer power instead of
stalling (live smoke wave 2a).

``calculate`` (``funhouse_agent/deep/calculate_tool.py``), reliability's
``_compile_g`` and pystra's ``_compile_limit_state`` all evaluate an
AST-checked expression with Python's own arithmetic, and Python evaluates
``int ** int`` EXACTLY: ``9**9**9`` -- or ``floor(9)**floor(9)**floor(9)``,
or an integer variable raised the same way -- builds a 370-million-digit
integer and holds the interpreter for minutes. Every power is now bounded
before it is computed.

The stall checks run in a child process with a timeout, so a regression
fails the test instead of hanging the suite.
"""

import math
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent("""
    import json, time
    from funhouse_agent.deep.calculate_tool import evaluate
    from funhouse_agent.adapters.reliability_adapter import _compile_g
    from pystra_agent.pystra_utils import _compile_limit_state
    out = {}
    t0 = time.time()
    out["calc_literal"] = evaluate("9**9**9")
    out["calc_floor"] = evaluate("floor(9)**floor(9)**floor(9)")
    out["calc_var"] = evaluate("ceil(a)**ceil(a)**ceil(a)", {"a": 9})
    out["calc_big_product"] = evaluate("floor(1e308)*floor(1e308)")
    try:
        _compile_g("a**a**a", ["a"])({"a": 9})
        out["g"] = "no error"
    except OverflowError as exc:
        out["g"] = "OverflowError: " + str(exc)
    try:
        _compile_limit_state("9**9**9 + R", ["R"])(R=1.0)
        out["lsf"] = "no error"
    except OverflowError as exc:
        out["lsf"] = "OverflowError: " + str(exc)
    out["seconds"] = time.time() - t0
    print(json.dumps(out))
""")


def test_no_evaluator_stalls_on_a_huge_integer_power():
    proc = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True,
                          text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    import json
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["seconds"] < 5, out
    for key in ("calc_literal", "calc_floor", "calc_var", "calc_big_product"):
        assert "error" in out[key] and "value" not in out[key], (key, out[key])
    assert "1e308" in out["calc_floor"]["error"]
    assert out["g"].startswith("OverflowError") and "1e308" in out["g"]
    assert out["lsf"].startswith("OverflowError") and "1e308" in out["lsf"]


def test_ordinary_powers_still_work():
    from funhouse_agent.deep.calculate_tool import evaluate
    from funhouse_agent.adapters.reliability_adapter import (
        _bounded_pow, _compile_g)
    assert evaluate("ceil(2)**10")["value"] == 1024.0
    assert evaluate("2**-3")["value"] == pytest.approx(0.125)
    assert evaluate("(-2)**3")["value"] == -8.0
    assert evaluate("tan(radians(45 - phi/2))**2", {"phi": 30})["value"] == \
        pytest.approx(1 / 3)
    assert evaluate("1e2**2")["value"] == pytest.approx(1e4)
    # an exact integer power up to about 1e308 is still exact
    assert _bounded_pow(2, 1000) == 2 ** 1000
    assert _bounded_pow(1, 10 ** 12) == 1 and _bounded_pow(-1, 10 ** 9 + 1) == -1
    assert _bounded_pow(0, 10 ** 12) == 0
    with pytest.raises(OverflowError):
        _bounded_pow(2, 1100)
    # float powers are Python's own (they overflow at once by themselves)
    with pytest.raises(OverflowError):
        _bounded_pow(10.0, 400.0)
    g = _compile_g("R**2 - S", ["R", "S"])
    assert g({"R": 3, "S": 1}) == 8
    # a variable cannot shadow the bounded power
    g2 = _compile_g("__gse_pow__**2", ["__gse_pow__"])
    assert g2({"__gse_pow__": 3}) == 9


def test_pystra_limit_state_powers_are_bounded_but_work_on_arrays():
    import numpy as np
    from pystra_agent.pystra_utils import _compile_limit_state
    f = _compile_limit_state("R**2 - S", ["R", "S"])
    assert list(f(R=np.array([2.0, 3.0]), S=1.0)) == [3.0, 8.0]
    assert f(R=3, S=1) == 8
    with pytest.raises(OverflowError):                 # 2**1200, exact
        _compile_limit_state("floor(R)**(floor(R)*600)", ["R"])(R=2.0)


def test_an_overlong_expression_is_refused():
    from funhouse_agent.adapters.reliability_adapter import _compile_g
    from funhouse_agent.deep.calculate_tool import evaluate
    from pystra_agent.pystra_utils import _compile_limit_state
    long_expr = "+".join(["x"] * 6000)
    with pytest.raises(ValueError, match="too long"):
        _compile_g(long_expr, ["x"])
    with pytest.raises(ValueError, match="too long"):
        _compile_limit_state(long_expr, ["x"])
    assert "too long" in evaluate(long_expr, {"x": 1})["error"]
    # a long but ordinary expression is fine
    assert math.isclose(evaluate("+".join(["x"] * 200), {"x": 1})["value"],
                        200.0)
    # the power rewrite is iterative: a 3,000-term chain compiles as before
    chain = "+".join(["x**2"] * 1500)
    assert _compile_g(chain, ["x"])({"x": 2}) == 6000
    assert _compile_limit_state(chain, ["x"])(x=2) == 6000
    # and powers inside calls, conditionals and other powers are all bounded
    # (each result is 2**1200 or more: over the bound, yet quick to compute
    # if a regression ever let it through, so this cannot hang the suite)
    for expr in ("max(1, x**(x*600))", "1 if x < 0 else x**(x*600)",
                 "(x**x)**(x*300)", "x**x**(x*6)"):
        with pytest.raises(OverflowError):
            _compile_g(expr, ["x"])({"x": 2})
        with pytest.raises(OverflowError):
            _compile_limit_state(expr, ["x"])(x=2)
    # the rewrite keeps Python's right-associative powers
    assert _compile_g("(x**x)**x", ["x"])({"x": 3}) == 27 ** 3
    assert _compile_g("x**x**x", ["x"])({"x": 3}) == 3 ** 27
    assert _compile_limit_state("x**x**x", ["x"])(x=3) == 3 ** 27
