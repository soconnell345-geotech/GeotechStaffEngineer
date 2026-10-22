"""A restricted arithmetic expression, compiled to a function of named values.

:func:`_compile_g` parses an expression over named variables and the math
functions in :data:`_MATH_FUNCS` and refuses anything but arithmetic,
comparisons and those calls; every power is bounded and the length capped,
so no expression can stall the interpreter. The reliability adapter's limit
states and the ``calculate`` tool both use it. Moved here from
:mod:`funhouse_agent.adapters.reliability_adapter` (INF-30 step 5), which
imports these names from here.
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
# The normal distribution (GM-94): erf/erfc, and the standard normal's CDF,
# density and inverse CDF, e.g. the probability that a kriged value with
# its kriging standard deviation exceeds a level.
_STD_NORMAL = __import__("statistics").NormalDist()
_MATH_FUNCS.update({"erf": math.erf, "erfc": math.erfc,
                    "norm_cdf": _STD_NORMAL.cdf,
                    "norm_pdf": _STD_NORMAL.pdf,
                    "norm_ppf": _STD_NORMAL.inv_cdf})

#: An integer power whose result would need more bits than this (2**1024 is
#: about 1.8e308, the largest float) is refused BEFORE it is computed. Python
#: evaluates ``int ** int`` exactly, so ``9**9**9`` -- or ``floor(9)`` raised
#: the same way, or an integer variable -- builds a 370-million-digit integer
#: and holds the interpreter for minutes (live smoke wave 2a). A float power
#: overflows at once by itself and is left alone.
_MAX_POW_BITS = 1024

#: The longest expression accepted. With every power bounded, only a chain of
#: multiplications can still grow an integer, and this keeps that chain short.
_MAX_EXPR_CHARS = 10_000
_TOO_DEEP = ("{what} is too long or too deeply nested to read; split it "
             "into shorter steps.")


_MAX_NESTING = 200


def _check_nesting(tree, what):
    """Refuse a tree nested deeper than ``_MAX_NESTING`` levels (a long chain
    of terms such as ``x+x+x+...`` nests one level per term). Measured with
    an explicit stack: Python's own tree walks recurse, and give up (or on
    3.11 overflow the C stack) well before a long expression is a problem."""
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > _MAX_NESTING:
            raise ValueError(_TOO_DEEP.format(what=what))
        stack.extend((c, depth + 1) for c in ast.iter_child_nodes(node))


def _compile_deep(tree, filename, what):
    """``compile(tree, filename, "eval")`` after the nesting check; a
    RecursionError or MemoryError that still gets through is the same
    refusal, never a crash."""
    _check_nesting(tree, what)
    try:
        return compile(tree, filename, "eval")
    except (RecursionError, MemoryError):
        raise ValueError(_TOO_DEEP.format(what=what)) from None


def _bounded_pow(base, exp):
    """``base ** exp``, refusing an exact integer result above about 1e308.

    Only a Python ``int`` raised to a positive ``int`` is checked (``bool`` is
    an ``int``); floats and numpy values overflow quickly on their own."""
    if isinstance(base, int) and isinstance(exp, int) and exp > 0:
        bits = abs(base).bit_length()
        if bits > 1 and (bits - 1) * exp > _MAX_POW_BITS:
            raise OverflowError(
                "a power whose result is larger than about 1e308 was "
                "refused before it was computed")
    return base ** exp


def _bound_powers(tree, var_names) -> str:
    """Rewrite every ``a ** b`` in the (already checked) ``tree`` as a call to
    :func:`_bounded_pow`; the name it is bound to, chosen so no variable
    shadows it."""
    name = "__gse_pow__"
    while name in set(var_names):
        name += "_"

    def _call(node):
        # both new nodes take the power's location (ast.fix_missing_locations
        # recurses, so it is not used)
        func = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
        return ast.copy_location(ast.Call(
            func=func, args=[node.left, node.right], keywords=[]), node)

    def _is_pow(node):
        return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)

    # Iterative, children before parents (reversed breadth-first order), so
    # a long chain cannot hit the recursion limit and a nested power
    # (a ** b ** c) is rewritten inside the call that replaces its parent.
    for parent in reversed(list(ast.walk(tree))):
        for field, value in ast.iter_fields(parent):
            if _is_pow(value):
                setattr(parent, field, _call(value))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if _is_pow(item):
                        value[i] = _call(item)
    return name


def _compile_g(expr: str, var_names, extra_funcs=None):
    """Compile a limit-state/FOS expression string into g(values_dict).

    Same restricted-eval pattern as the pystra adapter: only the variable
    names and math functions are allowed identifiers. Every power is bounded
    (:func:`_bounded_pow`) and the expression's length capped, so no
    expression can stall the interpreter on a huge integer.
    """
    if not expr or not str(expr).strip():
        raise ValueError(
            "g_expression is required, e.g. 'R - S' (margin) or "
            "'(c + q*tan(radians(phi)))/tau' (FOS).")
    if len(str(expr)) > _MAX_EXPR_CHARS:
        raise ValueError(
            f"g_expression is too long ({len(str(expr)):,} characters; the "
            f"limit is {_MAX_EXPR_CHARS:,}).")
    funcs = dict(_MATH_FUNCS)
    if extra_funcs:
        funcs.update(extra_funcs)
    allowed = set(var_names) | set(funcs)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"g_expression is not a valid expression: {exc}")
    except (RecursionError, MemoryError):
        raise ValueError(_TOO_DEEP.format(what="g_expression")) from None
    # Identifiers are checked on the PARSED names, not by a regex over the
    # text: a regex reads the exponent of 1e-3 as an identifier 'e' and
    # refused scientific notation.
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in allowed:
            raise ValueError(
                f"Unknown identifier '{node.id}' in g_expression. Allowed "
                f"variables: {sorted(var_names)}; plus math functions "
                f"like sqrt/log/exp/tan/radians.")
    # Beyond the identifier allowlist: refuse any AST construct outside pure
    # arithmetic — no attribute access, subscripts, strings, f-strings,
    # lambdas, comprehensions, or starred/keyword call tricks can reach
    # execution, and the code runs with no builtins.
    _OK = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
           ast.Constant, ast.Load, ast.IfExp, ast.Compare, ast.BoolOp,
           ast.And, ast.Or,
           ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
           ast.FloorDiv, ast.USub, ast.UAdd,
           ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
    for node in ast.walk(tree):
        if not isinstance(node, _OK):
            raise ValueError(
                f"g_expression may only contain arithmetic, comparisons, "
                f"and the allowed math functions (rejected: "
                f"{type(node).__name__}).")
        if isinstance(node, ast.Call):
            if node.keywords or not (isinstance(node.func, ast.Name)
                                     and node.func.id in funcs):
                raise ValueError(
                    "g_expression function calls are limited to the "
                    f"allowed math functions: {sorted(funcs)}")
        if isinstance(node, ast.Constant) and \
                not isinstance(node.value, (int, float)):
            raise ValueError(
                "g_expression constants must be numbers, got "
                f"{type(node.value).__name__}.")
    pow_name = _bound_powers(tree, var_names)
    code = _compile_deep(tree, "<g_expression>", "g_expression")
    ns = {"__builtins__": {}}
    ns.update(funcs)
    ns[pow_name] = _bounded_pow

    def g(values):
        local = {k: values[k] for k in var_names}
        return eval(code, ns, local)   # nosec B307 — AST-whitelisted above

    return g
