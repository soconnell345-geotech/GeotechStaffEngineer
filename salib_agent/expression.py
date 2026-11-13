"""
A model given as an arithmetic expression of the sampled variables.

Lets a one-call Sobol / Morris analysis evaluate the model itself (live smoke
G9, 2026-10-08: a sample matrix of 1,280 rows came back to an agent that
cannot loop over it). Same restriction as the reliability engines'
``g_expression``: identifiers are the variable names and a fixed set of math
functions; the parsed tree may contain arithmetic, comparisons and calls to
those functions only (no attributes, subscripts, strings, lambdas or
comprehensions), and it runs with no builtins.

It also takes the reliability engines' LIMITS (GM-323), the same numbers as
``aec_agent_core.safe_expr`` and ``pystra_agent.pystra_utils``: the
expression is at most ``_MAX_EXPR_CHARS`` long and ``_MAX_NESTING`` levels
deep, and every power is computed by :func:`_bounded_pow`, which refuses an
exact integer result above about 1e308 before computing it (Python
evaluates ``int ** int`` exactly, so ``9**9**9`` would hold the interpreter
for minutes). An analysis module imports no other module's code, so the
limits are restated here; ``salib_agent/tests`` pins them to the reliability
engine's.
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


#: An integer power whose exact result would need more bits than this
#: (2**1024 is about 1.8e308, the largest float) is refused before it is
#: computed. Float powers overflow at once by themselves and are left alone.
_MAX_POW_BITS = 1024

#: The longest expression accepted, and the deepest nesting (a chain such
#: as ``x+x+x+...`` nests one level per term).
_MAX_EXPR_CHARS = 10_000
_MAX_NESTING = 200
_TOO_DEEP = ("expression is too long or too deeply nested to read; split "
             "it into shorter steps")


def _bounded_pow(base, exp):
    """``base ** exp``, refusing an exact integer result above about 1e308."""
    if isinstance(base, int) and isinstance(exp, int) and exp > 0:
        bits = abs(base).bit_length()
        if bits > 1 and (bits - 1) * exp > _MAX_POW_BITS:
            raise OverflowError(
                "a power whose result is larger than about 1e308 was "
                "refused before it was computed")
    return base ** exp


def _check_nesting(tree):
    """Refuse a tree nested deeper than ``_MAX_NESTING`` levels, measured
    with an explicit stack (Python's own tree walks recurse)."""
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > _MAX_NESTING:
            raise ValueError(_TOO_DEEP)
        stack.extend((c, depth + 1) for c in ast.iter_child_nodes(node))


def _bound_powers(tree, var_names) -> str:
    """Rewrite every ``a ** b`` in the (already checked) ``tree`` as a call
    to :func:`_bounded_pow`; returns the name it is bound to, chosen so no
    variable shadows it. Iterative, children before parents, so a nested
    power (a ** b ** c) is rewritten inside the call replacing its parent."""
    name = "__gse_pow__"
    while name in set(var_names):
        name += "_"

    def _call(node):
        func = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
        return ast.copy_location(ast.Call(
            func=func, args=[node.left, node.right], keywords=[]), node)

    def _is_pow(node):
        return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)

    for parent in reversed(list(ast.walk(tree))):
        for field, value in ast.iter_fields(parent):
            if _is_pow(value):
                setattr(parent, field, _call(value))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if _is_pow(item):
                        value[i] = _call(item)
    return name


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
        On an unknown identifier, any construct outside plain arithmetic,
        or an expression longer or more deeply nested than the limits.
    OverflowError
        From the compiled function, for an integer power above about 1e308.
    """
    if not expr or not str(expr).strip():
        raise ValueError("expression is empty")
    if len(str(expr)) > _MAX_EXPR_CHARS:
        raise ValueError(
            f"expression is too long ({len(str(expr)):,} characters; the "
            f"limit is {_MAX_EXPR_CHARS:,})")
    var_names = list(var_names)
    allowed = set(var_names) | set(_MATH_FUNCS)
    try:
        tree = ast.parse(str(expr), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"expression is not valid: {exc}") from None
    except (RecursionError, MemoryError):
        raise ValueError(_TOO_DEEP) from None
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
    pow_name = _bound_powers(tree, var_names)
    _check_nesting(tree)
    try:
        code = compile(tree, "<expression>", "eval")
    except (RecursionError, MemoryError):
        raise ValueError(_TOO_DEEP) from None
    namespace = {"__builtins__": {}}
    namespace.update(_MATH_FUNCS)
    namespace[pow_name] = _bounded_pow

    def f(values):
        local = {k: values[k] for k in var_names}
        return eval(code, namespace, local)  # nosec B307 — AST-whitelisted

    return f
