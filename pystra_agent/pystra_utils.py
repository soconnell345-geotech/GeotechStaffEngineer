"""
Utilities for pystra integration and compatibility.
"""

import sys

import numpy as np

from geotech_common.optional_packages import import_error


def pystra_import_error():
    """Why pystra cannot be imported (not installed, or its import's own
    error), or None when it can (BACKLOG INF-54)."""
    return import_error("pystra")


def has_pystra() -> bool:
    """
    Check if pystra is available.

    Returns:
        True if pystra can be imported, False otherwise
    """
    return pystra_import_error() is None


_patches_applied = False


def _is_v2() -> bool:
    """True for pystra 2.x, whose FORM/SORM/Monte Carlo API replaced 1.x's."""
    import pystra
    return hasattr(pystra, "FORM")


class _Run:
    """Plain record of one analysis, the same whichever pystra ran it.

    Arrays are in pystra's own order (random variables only, constants left
    out); ``None`` where pystra reports no value (a quantity its version
    does not give, or a run that did not converge).
    """

    beta = pf = alpha = x = u = kappa = None
    iterations = 0
    beta_breitung = pf_breitung = None
    #: FORM: pystra's own convergence flag (2.x), or its stopping rule read
    #: back (1.x has no flag: it stops at the tolerances or at the
    #: iteration limit, so "stopped below the limit" is its convergence).
    converged = None
    converged_source = ""
    message = ""
    #: pystra's own count of limit-state evaluations (2.x); None on 1.x.
    n_evaluations = None
    #: SORM stage: whether it was run, pystra's flag and message.
    sorm_converged = None
    sorm_message = ""
    #: Monte Carlo: samples actually run, and the seed used (None = fresh).
    n_samples = None
    seed = None


def run_form(sm, lsf, with_sorm: bool = False) -> _Run:
    """Run FORM (and SORM when asked) on pystra 1.x or 2.x.

    Nothing is filled in for a stage that failed: an unconverged FORM on
    2.x has no index, probability or design point (pystra gives none), and a
    failed SORM leaves ``beta_breitung``/``pf_breitung``/``kappa`` at None
    with pystra's message in ``sorm_message`` (GM-322).
    """
    import pystra

    out = _Run()
    if _is_v2():
        # on_failure="return" hands back pystra's record of an unconverged
        # run (status, message, residuals) instead of raising.
        form = pystra.FORM(sm, lsf, on_failure="return")
        res = form.run()
        out.converged = bool(res.converged)
        out.converged_source = "pystra's own flag (FORMResult.converged)"
        out.message = str(res.message)
        out.n_evaluations = int(res.n_limit_state_evaluations)
        out.beta = None if res.beta is None else float(res.beta)
        out.pf = (None if res.failure_probability is None
                  else float(res.failure_probability))
        out.alpha = None if res.alpha is None else np.asarray(res.alpha)
        out.x = None if res.design_point_x is None else np.asarray(res.design_point_x)
        out.u = None if res.design_point_u is None else np.asarray(res.design_point_u)
        out.iterations = int(res.iterations)
        if with_sorm:
            if not out.converged:
                out.sorm_converged = False
                out.sorm_message = (
                    "SORM was not run: it is fitted at FORM's design point, "
                    "and FORM did not converge")
                return out
            sres = pystra.SORM(sm, lsf, form=form, on_failure="return").run()
            out.sorm_converged = bool(sres.converged)
            out.sorm_message = str(sres.message)
            out.n_evaluations += int(sres.n_limit_state_evaluations)
            out.beta_breitung = None if sres.beta is None else float(sres.beta)
            out.pf_breitung = (
                None if sres.failure_probability is None
                else float(sres.failure_probability))
            out.kappa = (
                None if sres.curvatures is None
                else np.atleast_1d(np.asarray(sres.curvatures, dtype=float)))
        return out

    options = pystra.AnalysisOptions()
    options.setPrintOutput(False)
    form = pystra.Form(sm, lsf, options)
    form.run()
    out.beta = float(form.getBeta())
    out.pf = float(form.getFailure().flat[0])
    out.alpha = np.asarray(form.getAlpha())
    out.x = np.asarray(form.getDesignPoint(uspace=False))
    out.u = np.asarray(form.getDesignPoint(uspace=True))
    out.iterations = int(form.i)
    # 1.x keeps no flag. Its loop ends when both tolerances are met OR at
    # the iteration limit, so a run that used every iteration is reported
    # as not converged (it may have met the tolerances on its last pass;
    # pystra 1.x does not say).
    out.converged = out.iterations < int(options.getImax())
    out.converged_source = (
        "pystra 1.x keeps no flag: read from its stopping rule (stopped "
        "before the iteration limit)")
    out.message = ("Converged" if out.converged
                   else f"FORM iteration limit ({int(options.getImax())}) "
                        f"reached")
    if with_sorm:
        try:
            sorm = pystra.Sorm(sm, lsf, options, form)
            sorm.run()
            out.beta_breitung = float(np.ravel(sorm.betag_breitung)[0])
            out.pf_breitung = float(np.ravel(sorm.pf2_breitung)[0])
            out.kappa = np.atleast_1d(np.asarray(sorm.kappa, dtype=float))
        except Exception as exc:  # the failure is reported, not hidden
            out.beta_breitung = out.pf_breitung = out.kappa = None
            out.sorm_converged = False
            out.sorm_message = f"{type(exc).__name__}: {exc}"
        else:
            # 1.x marks an undefined Breitung formula (1 + beta*kappa <= 0
            # for a curvature) by printing a line and setting both values
            # to exactly 0; a real result never has pf = 0 with beta = 0.
            ok = (np.isfinite(out.beta_breitung)
                  and np.isfinite(out.pf_breitung)
                  and not (out.pf_breitung == 0.0
                           and out.beta_breitung == 0.0))
            out.sorm_converged = bool(ok)
            out.sorm_message = (
                "Fitted" if ok
                else "The Breitung formula is undefined for the fitted "
                     "curvatures")
            if not ok:
                out.beta_breitung = out.pf_breitung = None
    return out


def run_monte_carlo(sm, lsf, n_samples: int, seed=None) -> _Run:
    """Run crude Monte Carlo on pystra 1.x or 2.x.

    Every one of the ``n_samples`` asked for is run: pystra's own default
    stops as soon as the coefficient of variation of pf reaches 0.05, which
    left the count run below the count reported (GM-322). ``seed`` is local
    to the call: numpy's global generator is neither read nor left changed.
    With ``seed=None`` the run is seeded from fresh entropy and differs
    from run to run.
    """
    import pystra

    out = _Run()
    out.seed = seed
    if _is_v2():
        mc = pystra.CrudeMonteCarlo(
            sm, lsf,
            options=pystra.SimulationOptions(n_samples=n_samples,
                                             target_cov=0.0),
            rng=seed)
        res = mc.run()
        out.beta = float(res.beta)
        out.pf = float(res.failure_probability)
        out.n_samples = int(res.n_samples)
        out.n_evaluations = int(res.n_limit_state_evaluations)
        return out

    options = pystra.AnalysisOptions()
    options.setPrintOutput(False)
    options.setSamples(n_samples)
    options.target_cov = -1.0  # 1.x has no setter; never reached, so no early stop
    mc = pystra.CrudeMonteCarlo(options, lsf, sm)
    # 1.x samples from numpy's global generator. Give it this call's seed
    # (or fresh entropy) and put the caller's generator state back after.
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        mc.run()
    finally:
        np.random.set_state(state)
    out.beta = float(np.ravel(mc.getBeta())[0])
    out.pf = float(np.ravel(mc.getFailure())[0])
    out.n_samples = int(mc.k)
    return out


def _apply_numpy2_patches():
    """
    Fix pystra for numpy >= 2.0 scalar conversion.

    pystra has bugs with numpy >=2.0 where jacobian(), x_to_u(), and u_to_x()
    return arrays instead of scalars in some contexts. This function monkey-patches
    the Transformation class to ensure proper scalar extraction.

    This function is idempotent and safe to call multiple times.
    """
    global _patches_applied
    if _patches_applied:
        return

    import numpy as np

    # Check numpy version
    if np.lib.NumpyVersion(np.__version__) < '2.0.0':
        _patches_applied = True
        return  # No patches needed for numpy < 2.0

    import pystra

    # pystra 2.x renamed Transformation.jacobian to jacobian_u_wrt_x and
    # already extracts scalars correctly, so there is nothing to patch.
    if not hasattr(pystra.Transformation, "jacobian"):
        _patches_applied = True
        return

    # Save originals (for reference, though we won't restore them)
    _orig_jacobian = pystra.Transformation.jacobian
    _orig_x_to_u = pystra.Transformation.x_to_u
    _orig_u_to_x = pystra.Transformation.u_to_x

    def _patched_jacobian(self, u, x, marg):
        """
        Build Jacobian matrix with proper scalar extraction.

        Original pystra Transformation.jacobian does:
            u = inv_T @ u; J[i][i] = marg[i].jacobian(u[i], x[i]); J = T @ J
        But numpy 2.x fails on scalar assignment when results are matrices.
        Distribution.jacobian(u, x) expects array-like args (uses u.size).
        """
        nrv = len(marg)
        u_t = np.asarray(np.dot(self.inv_T, u)).flatten()
        x_flat = np.asarray(x).flatten()
        J_u_x = np.zeros((nrv, nrv))
        for i in range(nrv):
            # Pass 1-element arrays — Distribution.jacobian uses u.size
            ui_arr = np.atleast_1d(u_t[i])
            xi_arr = np.atleast_1d(x_flat[i])
            jac_val = marg[i].jacobian(ui_arr, xi_arr)
            J_u_x[i][i] = float(np.asarray(jac_val).flat[0])
        J_u_x = np.asarray(np.dot(self.T, J_u_x))
        return J_u_x

    def _patched_x_to_u(self, x, marg):
        """
        Transform from physical to standard normal space with scalar extraction.

        Original pystra does:
            u[i] = marg[i].x_to_u(x[i]); u = T @ u
        Distribution.x_to_u expects scalar float.
        """
        nrv = len(marg)
        u = np.zeros(nrv)
        for i in range(nrv):
            xi = float(np.asarray(x[i]).flat[0])
            u_val = marg[i].x_to_u(xi)
            u[i] = float(np.asarray(u_val).flat[0])
        u = np.asarray(np.dot(self.T, u)).flatten()
        return u

    def _patched_u_to_x(self, u, marg):
        """
        Transform from standard normal to physical space with scalar extraction.

        Original pystra does:
            z = inv_T @ u; x[i] = marg[i].u_to_x(z[i])
        Distribution.u_to_x expects scalar float.
        """
        nrv = len(marg)
        z = np.asarray(np.dot(self.inv_T, u)).flatten()
        x = np.zeros(nrv)
        for i in range(nrv):
            zi = float(z[i])
            x_val = marg[i].u_to_x(zi)
            x[i] = float(np.asarray(x_val).flat[0])
        return x

    # Apply patches
    pystra.Transformation.jacobian = _patched_jacobian
    pystra.Transformation.x_to_u = _patched_x_to_u
    pystra.Transformation.u_to_x = _patched_u_to_x

    _patches_applied = True


_LSF_FUNCS = (
    'sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'pi', 'asin', 'acos', 'atan',
    'atan2', 'sinh', 'cosh', 'tanh', 'log10', 'ceil', 'floor',
)

#: An integer power whose exact result would need more bits than this (2**1024
#: is about 1.8e308, the largest float) is refused before it is computed:
#: Python evaluates ``int ** int`` exactly, and ``9**9**9`` builds a
#: 370-million-digit integer that holds the interpreter for minutes (live
#: smoke wave 2a). Float and numpy powers overflow at once and are left alone.
_MAX_POW_BITS = 1024

#: The longest expression accepted (with powers bounded, only a long chain of
#: multiplications could still grow an integer).
_MAX_EXPR_CHARS = 10_000
_MAX_NESTING = 200
_TOO_DEEP = ("Limit state expression is too long or too deeply nested to "
             "read; split it into shorter steps")


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
    """Refuse a tree nested deeper than ``_MAX_NESTING`` levels (one per term
    of a long chain), measured with an explicit stack: Python's own tree
    walks recurse and give up, or overflow the C stack, long before."""
    import ast
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > _MAX_NESTING:
            raise ValueError(_TOO_DEEP)
        stack.extend((c, depth + 1) for c in ast.iter_child_nodes(node))


def _compile_limit_state(expr_str: str, var_names: list) -> callable:
    """
    Compile a limit state expression string into a callable function.

    The expression is parsed and checked as an AST before anything runs:
    identifiers must be the variable names or the math functions below, and
    the tree may hold only arithmetic, comparisons, ``a if c else b`` and
    calls to those functions (no attributes, subscripts, strings, lambdas,
    comprehensions or keyword calls). It runs with no builtins. Identifiers
    are taken from the parsed tree, not a regex over the text, so scientific
    notation (``1e-3``) is accepted.

    Args:
        expr_str: String expression like "R - S" or "R**2 - 1e-3*S"
        var_names: List of variable names that can appear in expression

    Returns:
        Callable taking the variables as keyword arguments (pystra calls
        ``expression(**kwargs)``)

    Raises:
        ValueError: If expression contains unknown identifiers or any
        construct outside plain arithmetic

    Examples:
        >>> f = _compile_limit_state("R - S", ["R", "S"])
        >>> f(R=200, S=100)
        100
    """
    import ast
    import math

    if not expr_str or not str(expr_str).strip():
        raise ValueError("Limit state expression cannot be empty")
    if len(str(expr_str)) > _MAX_EXPR_CHARS:
        raise ValueError(
            f"Limit state expression is too long ({len(str(expr_str)):,} "
            f"characters; the limit is {_MAX_EXPR_CHARS:,})")
    var_names = list(var_names)
    funcs = {name: getattr(math, name) for name in _LSF_FUNCS
             if hasattr(math, name)}
    funcs.update({'abs': abs, 'min': min, 'max': max})
    allowed = set(var_names) | set(funcs)

    try:
        tree = ast.parse(str(expr_str), mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"Limit state expression is not valid: {exc}") from None
    except (RecursionError, MemoryError):
        raise ValueError(_TOO_DEEP) from None

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in allowed:
            raise ValueError(
                f"Unknown identifier '{node.id}' in limit state expression. "
                f"Allowed: {', '.join(sorted(var_names))}"
            )
    ok = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name,
          ast.Constant, ast.Load, ast.IfExp, ast.Compare, ast.BoolOp,
          ast.And, ast.Or,
          ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
          ast.FloorDiv, ast.USub, ast.UAdd,
          ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
    for node in ast.walk(tree):
        if not isinstance(node, ok):
            raise ValueError(
                f"Limit state expression may only contain arithmetic, "
                f"comparisons and the allowed math functions (rejected: "
                f"{type(node).__name__}).")
        if isinstance(node, ast.Call) and (
                node.keywords or not (isinstance(node.func, ast.Name)
                                      and node.func.id in funcs)):
            raise ValueError(
                f"Limit state function calls are limited to "
                f"{sorted(funcs)}")
        if isinstance(node, ast.Constant) and \
                not isinstance(node.value, (int, float)):
            raise ValueError("Limit state constants must be numbers")

    # pystra hands the limit state whole arrays of sample points. Plain
    # arithmetic works on them as it stands; a math function (math.sqrt
    # takes one number), a comparison or ``a if c else b`` does not, so an
    # expression with any of those is evaluated one point at a time.
    pointwise = any(isinstance(node, (ast.Call, ast.IfExp, ast.BoolOp,
                                      ast.Compare))
                    for node in ast.walk(tree))

    # Every ``a ** b`` becomes a call to _bounded_pow (after the checks, so
    # the name is not one an expression can write), under a name no variable
    # shadows.
    pow_name = "__gse_pow__"
    while pow_name in set(var_names):
        pow_name += "_"

    def _call(node):
        # both new nodes take the power's location (ast.fix_missing_locations
        # recurses, so it is not used)
        func = ast.copy_location(ast.Name(id=pow_name, ctx=ast.Load()), node)
        return ast.copy_location(ast.Call(
            func=func, args=[node.left, node.right], keywords=[]), node)

    def _is_pow(node):
        return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)

    # Iterative, children before parents (reversed breadth-first order): a
    # long chain cannot hit the recursion limit, and a nested power
    # (a ** b ** c) is rewritten inside the call that replaces its parent.
    for parent in reversed(list(ast.walk(tree))):
        for field, value in ast.iter_fields(parent):
            if _is_pow(value):
                setattr(parent, field, _call(value))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if _is_pow(item):
                        value[i] = _call(item)
    _check_nesting(tree)
    try:
        code = compile(tree, "<limit_state>", "eval")
    except (RecursionError, MemoryError):
        raise ValueError(_TOO_DEEP) from None
    namespace = {"__builtins__": {}}
    namespace.update(funcs)
    namespace[pow_name] = _bounded_pow

    def lsf(**kwargs):
        local = {k: kwargs[k] for k in var_names}
        if not pointwise or all(np.ndim(v) == 0 for v in local.values()):
            return eval(code, namespace, local)  # nosec B307 — AST-whitelisted
        columns = np.broadcast_arrays(
            *[np.asarray(local[k], dtype=float) for k in var_names])
        out = np.empty(columns[0].shape, dtype=float)
        for idx in np.ndindex(out.shape):
            point = {k: float(c[idx]) for k, c in zip(var_names, columns)}
            out[idx] = eval(code, namespace, point)  # nosec B307
        return out

    return lsf


def _create_pystra_variable(var_dict: dict):
    """
    Create a pystra distribution object from a variable specification dict.

    Args:
        var_dict: Dict with keys:
            - name: str (variable name)
            - dist: str (distribution type)
            - mean: float (for most distributions)
            - stdv: float (for most distributions)
            - value: float (for constant)
            - Additional params for specific distributions

    Returns:
        pystra distribution object

    Raises:
        ValueError: If distribution type unknown or required params missing
        ImportError: If pystra not available
    """
    import pystra

    name = var_dict.get("name")
    dist = var_dict.get("dist", "").lower()

    if not name:
        raise ValueError("Variable must have 'name' field")
    if not dist:
        raise ValueError(f"Variable '{name}' must have 'dist' field")

    # Map distribution names to pystra classes
    if dist == "normal":
        mean = var_dict.get("mean")
        stdv = var_dict.get("stdv")
        if mean is None or stdv is None:
            raise ValueError(f"Normal distribution '{name}' requires 'mean' and 'stdv'")
        return pystra.Normal(name, mean, stdv)

    elif dist == "lognormal":
        mean = var_dict.get("mean")
        stdv = var_dict.get("stdv")
        if mean is None or stdv is None:
            raise ValueError(f"Lognormal distribution '{name}' requires 'mean' and 'stdv'")
        return pystra.Lognormal(name, mean, stdv)

    elif dist == "gumbel":
        mean = var_dict.get("mean")
        stdv = var_dict.get("stdv")
        if mean is None or stdv is None:
            raise ValueError(f"Gumbel distribution '{name}' requires 'mean' and 'stdv'")
        return pystra.Gumbel(name, mean, stdv)

    elif dist == "uniform":
        a = var_dict.get("a")
        b = var_dict.get("b")
        if a is None or b is None:
            raise ValueError(f"Uniform distribution '{name}' requires 'a' and 'b'")
        if _is_v2():
            return pystra.Uniform(name, lower=a, upper=b)
        # 1.x reads (mean, stdv) unless input_type is given, then (a, b)
        return pystra.Uniform(name, a, b, input_type=1)

    elif dist == "constant":
        value = var_dict.get("value")
        if value is None:
            raise ValueError(f"Constant distribution '{name}' requires 'value'")
        return pystra.Constant(name, value)

    elif dist == "weibull":
        mean = var_dict.get("mean")
        stdv = var_dict.get("stdv")
        if mean is None or stdv is None:
            raise ValueError(f"Weibull distribution '{name}' requires 'mean' and 'stdv'")
        return pystra.Weibull(name, mean, stdv)

    elif dist == "gamma_dist":
        mean = var_dict.get("mean")
        stdv = var_dict.get("stdv")
        if mean is None or stdv is None:
            raise ValueError(f"Gamma distribution '{name}' requires 'mean' and 'stdv'")
        return pystra.Gamma(name, mean, stdv)

    elif dist == "beta":
        mean = var_dict.get("mean")
        stdv = var_dict.get("stdv")
        if mean is None or stdv is None:
            raise ValueError(f"Beta distribution '{name}' requires 'mean' and 'stdv'")
        return pystra.Beta(name, mean, stdv)

    else:
        raise ValueError(
            f"Unknown distribution type '{dist}'. "
            f"Supported: normal, lognormal, gumbel, uniform, constant, weibull, gamma_dist, beta"
        )


def _build_stochastic_model(variables: list, correlation: list = None):
    """
    Build a pystra StochasticModel from variable list and optional correlation.

    Args:
        variables: List of dicts with variable specifications
        correlation: Optional correlation matrix (list of lists)

    Returns:
        tuple: (StochasticModel, list of pystra variable objects, list of variable names)

    Raises:
        ValueError: If correlation matrix dimensions don't match number of variables
    """
    import pystra

    # Create pystra variables
    pystra_vars = [_create_pystra_variable(v) for v in variables]
    var_names = [v.get("name") for v in variables]

    # Build stochastic model
    sm = pystra.StochasticModel()
    for pvar in pystra_vars:
        if _is_v2():
            sm.add_variable(pvar)
        else:
            sm.addVariable(pvar)

    # Set correlation if provided
    if correlation is not None:
        n_vars = len(pystra_vars)
        if len(correlation) != n_vars:
            raise ValueError(
                f"Correlation matrix has {len(correlation)} rows but {n_vars} variables"
            )
        for row in correlation:
            if len(row) != n_vars:
                raise ValueError(
                    f"Correlation matrix must be {n_vars}x{n_vars}, "
                    f"found row with {len(row)} columns"
                )
        if _is_v2():
            sm.set_correlation(pystra.CorrelationMatrix(correlation))
        else:
            sm.setCorrelation(pystra.CorrelationMatrix(correlation))

    return sm, pystra_vars, var_names
