"""
Utilities for pystra integration and compatibility.
"""

import sys


def has_pystra() -> bool:
    """
    Check if pystra is available.

    Returns:
        True if pystra can be imported, False otherwise
    """
    try:
        import pystra
        return True
    except ImportError:
        return False


_patches_applied = False


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


def _compile_limit_state(expr_str: str, var_names: list) -> callable:
    """
    Compile a limit state expression string into a callable function.

    Args:
        expr_str: String expression like "R - S" or "R**2 - S"
        var_names: List of variable names that can appear in expression

    Returns:
        Callable that takes keyword arguments matching var_names

    Raises:
        ValueError: If expression contains unknown identifiers

    Examples:
        >>> f = _compile_limit_state("R - S", ["R", "S"])
        >>> f(R=200, S=100)
        100
    """
    import re
    import math

    # Security: only allow math operations and variable names
    allowed = set(var_names) | {
        'abs', 'min', 'max', 'sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'pi',
        'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'log10', 'ceil', 'floor'
    }

    # Extract all identifiers from expression
    tokens = re.findall(r'[a-zA-Z_]\w*', expr_str)
    for token in tokens:
        if token not in allowed:
            raise ValueError(
                f"Unknown identifier '{token}' in limit state expression. "
                f"Allowed: {', '.join(sorted(var_names))}"
            )

    # Build lambda function string
    arg_str = ", ".join(var_names)
    func_str = f"lambda {arg_str}: {expr_str}"

    # Provide math functions in namespace
    ns = {
        name: getattr(math, name)
        for name in [
            'sqrt', 'log', 'exp', 'sin', 'cos', 'tan', 'pi',
            'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh',
            'log10', 'ceil', 'floor'
        ]
        if hasattr(math, name)
    }
    ns['abs'] = abs
    ns['min'] = min
    ns['max'] = max

    # Compile with restricted builtins
    return eval(func_str, {"__builtins__": {}}, ns)


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
        return pystra.Uniform(name, a, b)

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
        sm.setCorrelation(pystra.CorrelationMatrix(correlation))

    return sm, pystra_vars, var_names
