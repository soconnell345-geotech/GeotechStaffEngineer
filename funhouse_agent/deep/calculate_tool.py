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
"""

from __future__ import annotations

import ast
import json
import math
from typing import Any, Dict, Optional

from langchain_core.tools import StructuredTool

#: The tool's model-facing description.
CALCULATE_DESCRIPTION = (
    "Evaluate an arithmetic formula exactly -- a published equation or "
    "correlation you found (e.g. su = (qt - sigma_v0) / Nkt), a unit "
    "conversion, a ratio of tool outputs -- instead of working it out "
    "yourself. 'expression' uses + - * / ** %, parentheses, pi and the "
    "functions sqrt log log10 exp sin cos tan "
    "asin acos atan atan2 sinh cosh tanh radians degrees ceil floor abs min "
    "max (trigonometry in radians: tan(radians(phi))). 'variables' maps each "
    "name in the expression to a number, e.g. {\"qt\": 1250, \"sigma_v0\": "
    "95, \"Nkt\": 14}. Returns the value with the expression and inputs; "
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
    raw = variables if isinstance(variables, dict) else {}
    values: Dict[str, float] = {}
    for name, v in raw.items():
        try:
            values[str(name)] = float(v)
        except (TypeError, ValueError):
            return {"error": f"variable '{name}' is not a number ({v!r})",
                    "expression": expr}
    if not expr:
        return {"error": "give 'expression', e.g. '(qt - sigma_v0) / Nkt' "
                         "with 'variables' {\"qt\": 1250, ...}"}
    try:
        bound = dict(values)
        g = _compile_g(_constants_as_names(expr, bound), list(bound))
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
                "variables": values}
    except (OverflowError, ArithmeticError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "expression": expr}
    except Exception as exc:                           # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}", "expression": expr}
    if isinstance(value, bool):
        out: Any = value
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return {"error": f"the result is not a finite number ({value})",
                    "expression": expr, "variables": values}
        try:
            out = float(value)
        except OverflowError:                # an exact integer above 1e308
            return {"error": "the result is too large to be a number "
                             "(above about 1e308)",
                    "expression": expr, "variables": values}
    else:
        return {"error": f"the expression did not give a number ({value!r})",
                "expression": expr}
    return {"value": out, "expression": expr, "variables": values}


def make_calculate_tool() -> StructuredTool:
    """The ``calculate`` tool (strict arguments, JSON result)."""
    from funhouse_agent.deep.strict_args import strict_tool

    def calculate(expression: str,
                  variables: Optional[Dict[str, float]] = None) -> str:
        """Evaluate an arithmetic formula exactly."""
        return json.dumps(evaluate(expression, variables), default=str)

    return strict_tool(StructuredTool.from_function(
        calculate, name="calculate", description=CALCULATE_DESCRIPTION))


__all__ = ["make_calculate_tool", "evaluate", "CALCULATE_DESCRIPTION"]
