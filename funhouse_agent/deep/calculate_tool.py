"""``calculate`` -- evaluate a stated formula exactly, instead of in the head.

Live smoke wave 1 (geotech questions, G3): the prompt forbids arithmetic and
no tool replaced it. Where no module method computes a value -- su from a
cone's qt with a chosen Nkt, a SHANSEP ratio, a unit conversion, a ratio of
two tool outputs -- the agents either gave no number (REF-7) or did the sum
themselves and flagged it "not tool-validated" (GC-1, REF-3, REF-5, REF-6).

This tool evaluates one arithmetic expression over named numbers with the
SAME AST-whitelisted evaluator the reliability module uses for its
limit-state functions (``reliability_adapter._compile_g``: arithmetic,
comparisons, conditionals and a fixed list of math functions; no names,
attributes, strings or calls beyond those). Integer constants are made
floats first, so a huge power overflows at once instead of building an
enormous integer; and the evaluator refuses any exact integer power above
about 1e308 before computing it (``floor(9)**floor(9)**floor(9)`` held the
process for minutes, live smoke wave 2a), so no expression can stall a turn.

One call evaluates several expressions over the same variables (BACKLOG
GM-169): a list, a ``{name: expression}`` map whose names are variables of
the expressions after them, or a string that is a list of expressions. Live
wave w13: one helper made 10 single calls after its list was refused, another
17.
"""

from __future__ import annotations

import ast
import json
import math
import statistics
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.tools import StructuredTool

#: The tool's model-facing description.
CALCULATE_DESCRIPTION = (
    "Evaluate an arithmetic formula exactly -- a published equation or "
    "correlation you found (e.g. su = (qt - sigma_v0) / Nkt), a unit "
    "conversion, a ratio of tool outputs -- instead of working it out "
    "yourself. 'expression' uses + - * / ** %, parentheses, pi and the "
    "functions sqrt log log10 exp sin cos tan "
    "asin acos atan atan2 sinh cosh tanh radians degrees ceil floor abs min "
    "max (trigonometry in radians: tan(radians(phi))), erf erfc, and the "
    "standard normal norm_cdf norm_pdf norm_ppf (e.g. the chance a value "
    "with mean m and std s exceeds L: 1 - norm_cdf((L - m) / s)). "
    "'variables' maps each name in the expression to a number, e.g. "
    "{\"qt\": 1250, \"sigma_v0\": 95, \"Nkt\": 14}, or to a LIST of "
    "numbers used through count sum mean median std (sample) min max "
    "percentile(L, p) count_above(L, v) count_below(L, v), e.g. "
    "count_above(z, 873.5) / count(z). Count, add up or average a list here, "
    "never by eye. Several formulas go in ONE call: 'expression' may be a "
    "list of expressions, or an object of name -> expression, e.g. "
    "{\"Ka\": \"tan(radians(45 - phi/2))**2\", \"Pa\": \"0.5*Ka*g*H**2\"} "
    "(each name's value can be used in the expressions after it), all "
    "over the same 'variables'; each comes back in 'results'. Returns the "
    "value with the expression and inputs; "
    "report all three. Where a module method computes the quantity "
    "(capacity, settlement, FOS...), call that method instead.")


def _constants_as_names(expression: str, values: Dict[str, float]) -> str:
    """``expression`` with every number literal replaced by a name bound, as
    a float, in ``values`` (``__k0``, ``__k1``...): integers become floats
    (a huge power overflows at once), and a literal such as ``1e-3`` cannot
    trip the evaluator's identifier check on its ``e``."""
    tree = ast.parse(str(expression), mode="eval")

    class _ToName(ast.NodeTransformer):
        def visit_Constant(self, node):          # noqa: N802 - ast API
            v = node.value
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                name = f"__k{len(values)}"
                values[name] = float(v)
                return ast.copy_location(ast.Name(id=name, ctx=ast.Load()),
                                         node)
            return node

    return ast.unparse(_ToName().visit(tree))


#: The longest list a variable may hold.
_MAX_LIST = 100_000


def _numbers(value, fn: str) -> list:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{fn}() takes a list variable")
    if not value:
        raise ValueError(f"{fn}() of an empty list")
    return list(value)


def _percentile(values, p):
    """Linear-interpolation percentile (numpy's default), p in 0-100."""
    xs = sorted(_numbers(values, "percentile"))
    if not 0.0 <= p <= 100.0:
        raise ValueError("percentile(L, p) takes p from 0 to 100")
    pos = (len(xs) - 1) * p / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _count_above(values, v):
    return float(sum(1 for x in _numbers(values, "count_above") if x > v))


def _count_below(values, v):
    return float(sum(1 for x in _numbers(values, "count_below") if x < v))


#: Functions over a list variable (GM-94: a field session's footprint
#: fractions were counted by eye from a grid, 10 of 40 wrong).
LIST_FUNCS = {
    "count": lambda L: float(len(_numbers(L, "count"))),
    "sum": lambda L: math.fsum(_numbers(L, "sum")),
    "mean": lambda L: math.fsum(_numbers(L, "mean")) / len(L),
    "median": lambda L: statistics.median(_numbers(L, "median")),
    "std": lambda L: statistics.stdev(_numbers(L, "std")),
    "percentile": _percentile,
    "count_above": _count_above,
    "count_below": _count_below,
}


def _echo(values: Dict[str, Any]) -> Dict[str, Any]:
    """The inputs as echoed: a long list is summarised, not repeated."""
    out = {}
    for k, v in values.items():
        if isinstance(v, list) and len(v) > 20:
            out[k] = f"[{len(v)} values, {min(v):g} to {max(v):g}]"
        else:
            out[k] = v
    return out


def _bind(variables) -> Tuple[Dict[str, Any], Optional[str]]:
    """``(values, None)`` -- each variable as a float or a list of floats --
    or ``({}, why)`` for one that is neither."""
    raw = variables if isinstance(variables, dict) else {}
    values: Dict[str, Any] = {}
    for name, v in raw.items():
        try:
            if isinstance(v, (list, tuple)):
                if len(v) > _MAX_LIST:
                    return {}, (f"variable '{name}' holds {len(v):,} "
                                f"values; the limit is {_MAX_LIST:,}")
                values[str(name)] = [float(x) for x in v]
            else:
                values[str(name)] = float(v)
        except (TypeError, ValueError):
            return {}, (f"variable '{name}' is not a number or a list of "
                        f"numbers ({str(v)[:80]!r})")
    return values, None


def evaluate(expression: str, variables: Optional[Dict[str, Any]] = None
             ) -> Dict[str, Any]:
    """``{"value", "expression", "variables"}`` or ``{"error", ...}``."""
    from funhouse_agent.adapters.reliability_adapter import (
        _MAX_EXPR_CHARS, _compile_g)
    expr = str(expression or "").strip()
    if len(expr) > _MAX_EXPR_CHARS:
        return {"error": f"the expression is too long ({len(expr):,} "
                         f"characters; the limit is {_MAX_EXPR_CHARS:,})",
                "expression": expr[:200] + "..."}
    values, problem = _bind(variables)
    if problem:
        return {"error": problem, "expression": expr}
    if not expr:
        return {"error": "give 'expression', e.g. '(qt - sigma_v0) / Nkt' "
                         "with 'variables' {\"qt\": 1250, ...}"}
    try:
        bound = dict(values)
        g = _compile_g(_constants_as_names(expr, bound), list(bound),
                       extra_funcs=LIST_FUNCS)
        value = g(bound)
    except ValueError as exc:
        import re
        msg = str(exc).replace("g_expression", "expression")
        msg = re.sub(r"'__k\d+',?\s*", "", msg).replace(", ]", "]")
        return {"error": msg, "expression": expr}
    except SyntaxError as exc:
        return {"error": f"not a valid expression: {exc.msg}",
                "expression": expr}
    except ZeroDivisionError:
        return {"error": "division by zero", "expression": expr,
                "variables": _echo(values)}
    except TypeError as exc:
        if any(isinstance(v, list) for v in values.values()):
            return {"error": "a list variable goes into a list function "
                             "(count, sum, mean, median, std, min, max, "
                             "percentile(L, p), count_above(L, v), "
                             "count_below(L, v)); arithmetic takes numbers "
                             f"({exc})",
                    "expression": expr}
        return {"error": f"TypeError: {exc}", "expression": expr}
    except (OverflowError, ArithmeticError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "expression": expr}
    except Exception as exc:                           # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "expression": expr}
    if isinstance(value, bool):
        out: Any = value
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return {"error": f"the result is not a finite number ({value})",
                    "expression": expr, "variables": _echo(values)}
        try:
            out = float(value)
        except OverflowError:                # an exact integer above 1e308
            return {"error": "the result is too large to be a number "
                             "(above about 1e308)",
                    "expression": expr, "variables": _echo(values)}
    else:
        return {"error": f"the expression did not give a number ({value!r})",
                "expression": expr}
    return {"value": out, "expression": expr, "variables": _echo(values)}


#: The most expressions one call evaluates.
MAX_EXPRESSIONS = 50


def _several(expression) -> Optional[List[Tuple[Optional[str], str]]]:
    """``[(name or None, expression), ...]`` when ``expression`` holds
    several -- a list, a ``{name: expression}`` map, or a string that is
    itself a list or tuple of expressions (``"[a/3, b*4]"``, the form live
    wave w13 refused) -- else ``None``."""
    if isinstance(expression, dict):
        return [(str(k), str(v)) for k, v in expression.items()]
    if isinstance(expression, (list, tuple)):
        return [(None, str(v)) for v in expression]
    if isinstance(expression, str):
        from funhouse_agent.adapters.reliability_adapter import (
            _MAX_EXPR_CHARS)
        if len(expression) > _MAX_EXPR_CHARS:
            return None                 # evaluate says it is too long
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            if isinstance(tree.body, (ast.List, ast.Tuple)) \
                    and tree.body.elts:
                return [(None, ast.unparse(e)) for e in tree.body.elts]
        except Exception:  # noqa: BLE001 - SyntaxError, RecursionError...
            return None                 # evaluate words the error
    return None


def evaluate_many(items: List[Tuple[Optional[str], str]],
                  variables: Optional[Dict[str, Any]] = None
                  ) -> Dict[str, Any]:
    """Each expression over the same ``variables``: ``{"results",
    "variables"}``, ``results`` a list, or a ``{name: result}`` map when the
    expressions are named -- then each name's value is a variable of the
    expressions after it. One expression's error is that one's alone."""
    if not items:
        return {"error": "give at least one expression"}
    if len(items) > MAX_EXPRESSIONS:
        return {"error": f"{len(items)} expressions in one call; the limit "
                         f"is {MAX_EXPRESSIONS}"}
    values, problem = _bind(variables)
    if problem:
        return {"error": problem}
    named = all(name is not None for name, _ in items)
    known = dict(values)
    results: Any = {} if named else []
    for name, expr in items:
        if named and name in known:
            out = {"error": f"'{name}' is already a variable; give this "
                            "result another name", "expression": expr}
        else:
            out = evaluate(expr, known)
            out.pop("variables", None)
        if named:
            results[name] = out
            if isinstance(out.get("value"), float):
                known[name] = out["value"]
        else:
            results.append(out)
    return {"results": results, "variables": _echo(values)}


def calculate_any(expression, variables=None) -> Dict[str, Any]:
    """:func:`evaluate` for one expression, :func:`evaluate_many` for
    several (see :func:`_several`)."""
    items = _several(expression)
    if items is None:
        return evaluate(expression, variables)
    return evaluate_many(items, variables)


def make_calculate_tool() -> StructuredTool:
    """The ``calculate`` tool (strict arguments, JSON result)."""
    from funhouse_agent.deep.strict_args import strict_tool

    def calculate(expression: Union[str, List[str], Dict[str, str]],
                  variables: Optional[Dict[str, Any]] = None) -> str:
        """Evaluate an arithmetic formula exactly."""
        return json.dumps(calculate_any(expression, variables), default=str)

    return strict_tool(StructuredTool.from_function(
        calculate, name="calculate", description=CALCULATE_DESCRIPTION))


__all__ = ["make_calculate_tool", "evaluate", "evaluate_many",
           "calculate_any", "CALCULATE_DESCRIPTION", "LIST_FUNCS",
           "MAX_EXPRESSIONS"]
