"""
A model given as an arithmetic expression of the sampled variables.

Lets a one-call Sobol / Morris analysis evaluate the model itself (live smoke
G9, 2026-10-08: a sample matrix of 1,280 rows came back to an agent that
cannot loop over it). Same restriction as the reliability engines'
``g_expression``: identifiers are the variable names and a fixed set of math
functions; the parsed tree may contain arithmetic, comparisons and calls to
those functions only (no attributes, subscripts, strings, lambdas or
comprehensions), and it runs with no builtins.
"""

import ast
import math


_MATH_FUNCS = {
    name: getattr(math, name)
    for name in ("sqrt", "log", "exp", "sin", "cos", "tan", "pi", "asin",
                 "acos", "atan", "atan2", "sinh", "cosh", "tanh", "log10",
                 "ceil", "floor", "radians", "degrees")
}
_MATH_FUNCS.update({"abs": abs, "min": min, "max": max})

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
    ast.Constant, ast.Load, ast.IfExp, ast.Compare, ast.BoolOp,
    ast.And, ast.Or,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.FloorDiv, ast.USub, ast.UAdd,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
)


def compile_expression(expr, var_names):
    """Compile ``expr`` into ``f(values: dict) -> float``.

    Parameters
    ----------
    expr : str
        e.g. ``"(c + 18*z*tan(radians(phi))) / 40"``.
    var_names : list of str
        The sampled variables the expression may use.

    Raises
    ------
    ValueError
        On an unknown identifier or any construct outside plain arithmetic.
    """
    if not expr or not str(expr).strip():
        raise ValueError("expression is empty")
    var_names = list(var_names)
    allowed = set(var_names) | set(_MATH_FUNCS)
    try:
        tree = ast.parse(str(expr), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"expression is not valid: {exc}") from None
    # Identifiers are checked on the parsed names (a regex scan would read
    # the exponent of 1e-3 as an identifier "e").
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in allowed:
            raise ValueError(
                f"Unknown identifier '{node.id}' in expression. Allowed: the "
                f"variables {var_names} and the math functions "
                f"{sorted(_MATH_FUNCS)}.")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"expression may only contain arithmetic, comparisons and "
                f"the allowed math functions (rejected: "
                f"{type(node).__name__}).")
        if isinstance(node, ast.Call) and (
                node.keywords or not (isinstance(node.func, ast.Name)
                                      and node.func.id in _MATH_FUNCS)):
            raise ValueError(
                f"function calls are limited to {sorted(_MATH_FUNCS)}")
        if isinstance(node, ast.Constant) and \
                not isinstance(node.value, (int, float)):
            raise ValueError("constants must be numbers")
    code = compile(tree, "<expression>", "eval")
    namespace = {"__builtins__": {}}
    namespace.update(_MATH_FUNCS)

    def f(values):
        local = {k: values[k] for k in var_names}
        return eval(code, namespace, local)  # nosec B307 — AST-whitelisted

    return f
