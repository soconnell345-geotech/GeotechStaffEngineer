"""
Finite difference beam-column solver for laterally loaded piles.

Solves the governing differential equation:
    EI * d4y/dz4 + Q * d2y/dz2 + Es * y = 0

where:
    EI = flexural rigidity of pile (kN-m^2)
    Q  = axial load (kN), positive in compression
    y  = lateral deflection (m)
    z  = depth (m)
    Es = secant modulus of soil reaction p/y (kN/m^2)

Uses the finite difference method with iterative solution to handle
the nonlinear p-y curve soil response.

Sign conventions:
    M = EI * d2y/dz2          (moment, positive for positive curvature)
    V = EI * d3y/dz3 + Q * dy/dz  (shear, includes axial load effect)

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).

References
----------
- COM624P Manual: FHWA-SA-91-048 (Wang & Reese, 1993), Part II.
- Hetenyi, M. (1946). "Beams on Elastic Foundation."
- Reese, L.C. & Van Impe, W.F. (2001). "Single Piles and Pile Groups
  Under Lateral Loading."
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded


@dataclass
class SolverResult:
    """Container for finite difference solver output.

    Attributes
    ----------
    z : numpy.ndarray
        Depth array (m), length n+1. Depth is measured from the ground
        surface: nodes above grade (free/stickup length) have z < 0.
    y : numpy.ndarray
        Lateral deflection (m).
    slope : numpy.ndarray
        Rotation/slope (radians).
    moment : numpy.ndarray
        Bending moment (kN-m).
    shear : numpy.ndarray
        Shear force (kN).
    soil_reaction : numpy.ndarray
        Soil reaction p (kN/m).
    Es : numpy.ndarray
        Secant soil modulus (kN/m^2).
    iterations : int
        Number of iterations to converge.
    converged : bool
        True if solution converged within tolerance.
    """
    z: np.ndarray
    y: np.ndarray
    slope: np.ndarray
    moment: np.ndarray
    shear: np.ndarray
    soil_reaction: np.ndarray
    Es: np.ndarray
    iterations: int
    converged: bool


def solve_lateral_pile(
    pile_length: float,
    EI_values: np.ndarray,
    py_functions: list,
    Vt: float = 0.0,
    Mt: float = 0.0,
    Q: float = 0.0,
    head_condition: str = 'free',
    rotational_stiffness: float = 0.0,
    n_elements: int = 100,
    tolerance: float = 1e-5,
    max_iterations: int = 100,
    pile_diameter: float = 1.0,
    stickup: float = 0.0,
) -> SolverResult:
    """Solve the laterally loaded pile problem using finite differences.

    Parameters
    ----------
    pile_length : float
        Embedded pile length below the ground surface (m).
    EI_values : numpy.ndarray
        Flexural rigidity at each node (kN-m^2). Length n+1 (spanning the
        full structural length, stickup + embedment, top to tip), or
        length 1 for a uniform pile.
    py_functions : list of callables
        List of functions, one per node. Each takes (y, z, b) and returns p.
        Length n+1. Functions at nodes above grade (z < 0) are ignored —
        zero soil resistance is enforced over the stickup length.
    Vt : float
        Lateral load at pile head (kN). Positive in direction of deflection.
        Applied at the TOP of the stickup when stickup > 0.
    Mt : float
        Applied moment at pile head (kN-m). Applied at the top of the stickup.
    Q : float
        Axial load on pile (kN). Positive in compression.
    head_condition : str
        'free': specified shear and moment (default).
        'fixed': specified shear and zero rotation (fixed head).
        'partial': specified shear and rotational stiffness.
        Boundary conditions are applied at the top of the stickup.
    rotational_stiffness : float
        Rotational stiffness at pile head (kN-m/rad). Only used when
        head_condition='partial'.
    n_elements : int
        Number of pile segments over the total (stickup + embedded) length.
        Default 100.
    tolerance : float
        Convergence tolerance on deflection. Default 1e-5.
    max_iterations : int
        Maximum number of iterations. Default 100.
    pile_diameter : float
        Pile diameter (m). Passed to p-y functions.
    stickup : float
        Above-ground free (unsupported) length of pile (m). Default 0
        (head load applied at the ground surface — behavior identical to
        the pre-stickup solver). When > 0, the finite difference mesh is
        extended above z=0 with zero soil resistance (p = 0) over the
        stickup, and the head load / boundary conditions are applied at
        the top of the stickup. For a free-headed pile this is statically
        equivalent (for the embedded response) to applying
        M = Mt + Vt*stickup at grade, provided Q = 0 (with Q != 0 a
        P-delta moment also develops over the stickup, which this mesh
        captures and the equivalent-load shortcut does not).

    Returns
    -------
    SolverResult
        Solution containing deflection, moment, shear, slope, and soil reaction.
        The depth array spans [-stickup, pile_length].
    """
    if stickup < 0:
        raise ValueError(f"stickup must be >= 0, got {stickup}")

    n = n_elements
    total_length = pile_length + stickup
    h = total_length / n
    n_nodes = n + 1

    z_top = -stickup if stickup > 0 else 0.0
    z = np.linspace(z_top, pile_length, n_nodes)

    # Nodes at or below the ground surface (small tolerance for the node
    # that lands exactly at grade in floating point).
    grade_tol = 1e-9 * max(total_length, 1.0)
    embedded = z >= -grade_tol

    # With a stickup the soil stiffness is discontinuous at z = 0 (no soil
    # above, Es below). For a node that lands exactly ON the discontinuity,
    # collocating with the full one-sided Es introduces an O(h) error in
    # the embedded response; the standard treatment is the average of the
    # two-sided limits, i.e. Es/2 at the grade node. (Not applied when
    # stickup = 0, where the grade node is the loaded boundary — preserves
    # the original solver behavior exactly.)
    soil_weight = np.ones(n_nodes)
    if stickup > 0:
        soil_weight[np.abs(z) <= grade_tol] = 0.5

    # Ensure EI_values has correct length
    if len(EI_values) == 1:
        EI = np.full(n_nodes, EI_values[0])
    elif len(EI_values) == n_nodes:
        EI = EI_values.copy()
    else:
        EI = np.interp(z, np.linspace(z_top, pile_length, len(EI_values)), EI_values)

    # Initial guess for Es (soil secant modulus); zero above grade
    Es = np.zeros(n_nodes)
    for i in range(n_nodes):
        if not embedded[i]:
            continue
        small_y = 0.001
        try:
            p_init = py_functions[i](small_y, z[i], pile_diameter)
            if abs(p_init) > 0:
                Es[i] = abs(p_init) / small_y
        except (ZeroDivisionError, ValueError):
            pass

    # Iterative solution
    converged = False
    y_prev = np.zeros(n_nodes)
    iterations = 0
    diff = 1.0

    for iteration in range(max_iterations):
        iterations = iteration + 1

        # Assemble and solve the full system
        Y_full = _assemble_and_solve(
            n, h, EI, Es * soil_weight, Q, Vt, Mt, head_condition,
            rotational_stiffness
        )

        # Extract real node deflections: Y[2] through Y[n+2]
        y_new = Y_full[2:n + 3]

        # Update Es from p-y curves (above-grade nodes keep Es = 0)
        Es_new = np.zeros(n_nodes)
        for i in range(n_nodes):
            if not embedded[i]:
                continue
            if abs(y_new[i]) > 1e-12:
                p_val = py_functions[i](y_new[i], z[i], pile_diameter)
                Es_new[i] = abs(p_val / y_new[i])
            else:
                Es_new[i] = Es[i]

        # Check convergence (skip first iteration)
        if iteration > 0:
            max_y = max(abs(y_new.max()), abs(y_new.min()), 1e-12)
            diff = np.max(np.abs(y_new - y_prev)) / max_y
            if diff < tolerance:
                converged = True
                Es = Es_new
                break

        # Update Es with relaxation for stability
        if iteration < 3:
            Es = Es_new
        else:
            Es = 0.5 * Es + 0.5 * Es_new

        y_prev = y_new.copy()

    y = y_new

    # ---- Compute derived quantities from full Y array ----
    # Slope: dy/dz using central differences (fictitious nodes for boundaries)
    slope = np.zeros(n_nodes)
    slope[0] = (Y_full[3] - Y_full[1]) / (2.0 * h)  # uses y[-1] and y[1]
    for i in range(1, n_nodes - 1):
        slope[i] = (y[i + 1] - y[i - 1]) / (2.0 * h)
    slope[-1] = (Y_full[n + 3] - Y_full[n + 1]) / (2.0 * h)  # uses y[n+1] and y[n-1]

    # Moment: M = EI * d2y/dz2
    moment = np.zeros(n_nodes)
    moment[0] = EI[0] * (Y_full[1] - 2.0 * Y_full[2] + Y_full[3]) / (h * h)
    for i in range(1, n_nodes - 1):
        moment[i] = EI[i] * (y[i - 1] - 2.0 * y[i] + y[i + 1]) / (h * h)
    moment[-1] = EI[-1] * (Y_full[n + 1] - 2.0 * Y_full[n + 2] + Y_full[n + 3]) / (h * h)

    # Shear: V = dM/dz + Q * dy/dz
    shear = np.zeros(n_nodes)
    shear[0] = Vt
    shear[-1] = 0.0
    for i in range(1, n_nodes - 1):
        shear[i] = (moment[i + 1] - moment[i - 1]) / (2.0 * h) + Q * slope[i]

    # Soil reaction from p-y curves (zero above grade)
    soil_reaction = np.array([
        py_functions[i](y[i], z[i], pile_diameter) if embedded[i] else 0.0
        for i in range(n_nodes)
    ])

    if not converged:
        warnings.warn(
            f"Solver did not converge after {max_iterations} iterations. "
            f"Last relative change: {diff:.2e}. Try increasing max_iterations "
            f"or n_elements."
        )

    return SolverResult(
        z=z, y=y, slope=slope, moment=moment, shear=shear,
        soil_reaction=soil_reaction, Es=Es,
        iterations=iterations, converged=converged,
    )


def _assemble_and_solve(
    n: int,
    h: float,
    EI: np.ndarray,
    Es: np.ndarray,
    Q: float,
    Vt: float,
    Mt: float,
    head_condition: str,
    rotational_stiffness: float,
) -> np.ndarray:
    """Assemble the full (n+5) system in banded form and solve.

    Uses n+5 unknowns: 2 fictitious nodes above the pile head (y[-2], y[-1]),
    n+1 real nodes (y[0]..y[n]), and 2 fictitious nodes below the tip
    (y[n+1], y[n+2]).

    Mapping: Y[j] corresponds to y[j-2].
        Y[0]=y[-2], Y[1]=y[-1], Y[2]=y[0], ..., Y[n+2]=y[n], Y[n+3]=y[n+1], Y[n+4]=y[n+2]

    Rows 0,1: head boundary conditions
    Rows 2..n+2: equilibrium equations at real nodes
    Rows n+3,n+4: tip boundary conditions

    The interior equilibrium rows are pentadiagonal (|i-j| <= 2); the head
    shear row (row 0, columns 0..4) and the tip shear row (row n+4,
    columns n..n+4) widen the band to |i-j| <= 4. The system is therefore
    solved with ``scipy.linalg.solve_banded`` with (l, u) = (4, 4) —
    O(n) instead of the previous dense O(n^3) ``np.linalg.solve`` — which
    matters for the iterative p-y / cracked-EI loops and capacity sweeps.

    Returns
    -------
    numpy.ndarray
        Full solution array Y of length n+5.
    """
    ab, F = _assemble_banded(
        n, h, EI, Es, Q, Vt, Mt, head_condition, rotational_stiffness
    )

    # --- Solve (banded LAPACK gbsv) ---
    try:
        Y = solve_banded((_BAND, _BAND), ab, F)
    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix in solver. Returning zero deflections.")
        Y = np.zeros(n + 5)

    return Y


# Half-bandwidth of the assembled system (set by the head/tip shear
# boundary-condition rows; interior rows are pentadiagonal, |i-j| <= 2).
_BAND = 4


def _assemble_banded(
    n: int,
    h: float,
    EI: np.ndarray,
    Es: np.ndarray,
    Q: float,
    Vt: float,
    Mt: float,
    head_condition: str,
    rotational_stiffness: float,
) -> tuple:
    """Assemble the (n+5) system in LAPACK banded storage.

    Returns (ab, F) where ``ab`` is the (2*_BAND+1, n+5) diagonal-ordered
    band matrix (``ab[_BAND + i - j, j] = K[i, j]``) suitable for
    ``scipy.linalg.solve_banded``, and ``F`` is the right-hand side.
    The matrix entries are identical to the previous dense assembly.
    """
    N = n + 5
    h2 = h * h
    h3 = h2 * h
    h4 = h2 * h2

    ab = np.zeros((2 * _BAND + 1, N))
    F = np.zeros(N)

    def add(i, j, val):
        ab[_BAND + i - j, j] += val

    # --- Equilibrium equations for real nodes (rows 2 to n+2) ---
    for i in range(n + 1):
        j = i + 2  # index in the full Y array
        add(j, j - 2, EI[i] / h4)
        add(j, j - 1, -4.0 * EI[i] / h4 + Q / h2)
        add(j, j,     6.0 * EI[i] / h4 - 2.0 * Q / h2 + Es[i])
        add(j, j + 1, -4.0 * EI[i] / h4 + Q / h2)
        add(j, j + 2, EI[i] / h4)

    # --- Head boundary conditions ---

    # Row 0: Shear at pile head = Vt
    # V(0) = EI[0]*(Y[4]-2Y[3]+2Y[1]-Y[0])/(2h^3) + Q*(Y[3]-Y[1])/(2h) = Vt
    add(0, 0, -EI[0] / (2.0 * h3))
    add(0, 1, EI[0] / h3 - Q / (2.0 * h))
    add(0, 3, -EI[0] / h3 + Q / (2.0 * h))
    add(0, 4, EI[0] / (2.0 * h3))
    F[0] = Vt

    # Row 1: Depends on head condition
    if head_condition == 'free':
        # Moment at head = Mt:  EI[0]*(Y[1]-2Y[2]+Y[3])/h^2 = Mt
        add(1, 1, EI[0] / h2)
        add(1, 2, -2.0 * EI[0] / h2)
        add(1, 3, EI[0] / h2)
        F[1] = Mt
    elif head_condition == 'fixed':
        # Slope = 0:  (Y[3]-Y[1])/(2h) = 0
        add(1, 1, -1.0)
        add(1, 3, 1.0)
        F[1] = 0.0
    elif head_condition == 'partial':
        # M = -Kr * slope
        # EI[0]*(Y[1]-2Y[2]+Y[3])/h^2 = -Kr*(Y[3]-Y[1])/(2h)
        # => EI[0]*(Y[1]-2Y[2]+Y[3])/h^2 + Kr*(Y[3]-Y[1])/(2h) = 0
        Kr = rotational_stiffness
        add(1, 1, EI[0] / h2 - Kr / (2.0 * h))
        add(1, 2, -2.0 * EI[0] / h2)
        add(1, 3, EI[0] / h2 + Kr / (2.0 * h))
        F[1] = 0.0

    # --- Tip boundary conditions ---

    # Row n+3: Moment at tip = 0
    # EI[n]*(Y[n+1]-2Y[n+2]+Y[n+3])/h^2 = 0
    add(n + 3, n + 1, EI[n] / h2)
    add(n + 3, n + 2, -2.0 * EI[n] / h2)
    add(n + 3, n + 3, EI[n] / h2)
    F[n + 3] = 0.0

    # Row n+4: Shear at tip = 0
    # EI[n]*(Y[n+4]-2Y[n+3]+2Y[n+1]-Y[n])/(2h^3) + Q*(Y[n+3]-Y[n+1])/(2h) = 0
    add(n + 4, n,     -EI[n] / (2.0 * h3))
    add(n + 4, n + 1, EI[n] / h3 - Q / (2.0 * h))
    add(n + 4, n + 3, -EI[n] / h3 + Q / (2.0 * h))
    add(n + 4, n + 4, EI[n] / (2.0 * h3))
    F[n + 4] = 0.0

    return ab, F


def hetenyi_solution(
    pile_length: float,
    EI: float,
    Es_constant: float,
    Vt: float,
    Mt: float = 0.0,
    n_points: int = 101,
) -> SolverResult:
    """Closed-form Hetenyi solution for pile in uniform elastic soil.

    For validation of the finite difference solver. Solves the case of a
    semi-infinite beam on an elastic foundation with a point load and
    moment at the free end.

    The solution is:
        y(z) = exp(-beta*z) * [A*cos(beta*z) + B*sin(beta*z)]

    where:
        beta = (Es / (4*EI))^0.25
        A = Vt/(2*beta^3*EI) + Mt/(2*beta^2*EI)
        B = -Mt/(2*beta^2*EI)

    Parameters
    ----------
    pile_length : float
        Pile length (m). Should be long enough that beta*L > 4.
    EI : float
        Flexural rigidity (kN-m^2).
    Es_constant : float
        Constant soil modulus (kN/m^2). p = Es * y.
    Vt : float
        Lateral load at pile head (kN).
    Mt : float
        Moment at pile head (kN-m). Default 0.
    n_points : int
        Number of output points. Default 101.

    Returns
    -------
    SolverResult
        Analytical solution.

    References
    ----------
    Hetenyi, M. (1946). "Beams on Elastic Foundation."
    """
    beta = (Es_constant / (4.0 * EI)) ** 0.25

    z = np.linspace(0, pile_length, n_points)

    # Coefficients
    A = Vt / (2.0 * beta**3 * EI) + Mt / (2.0 * beta**2 * EI)
    B = -Mt / (2.0 * beta**2 * EI)

    y = np.zeros(n_points)
    slope = np.zeros(n_points)
    moment = np.zeros(n_points)
    shear = np.zeros(n_points)

    for i, zi in enumerate(z):
        bz = beta * zi
        e_bz = np.exp(-bz)
        cos_bz = np.cos(bz)
        sin_bz = np.sin(bz)

        # y = exp(-bz) * [A*cos + B*sin]
        y[i] = e_bz * (A * cos_bz + B * sin_bz)

        # slope = beta * exp(-bz) * [(-A+B)*cos + (-A-B)*sin]
        slope[i] = beta * e_bz * ((-A + B) * cos_bz + (-A - B) * sin_bz)

        # M = EI*y'' = 2*beta^2*EI * exp(-bz) * [-B*cos + A*sin]
        moment[i] = 2.0 * beta**2 * EI * e_bz * (-B * cos_bz + A * sin_bz)

        # V = EI*y''' + Q*dy/dz (Q=0 for Hetenyi)
        # = 2*beta^3*EI * exp(-bz) * [(A+B)*cos + (B-A)*sin]
        shear[i] = 2.0 * beta**3 * EI * e_bz * ((A + B) * cos_bz + (B - A) * sin_bz)

    soil_reaction = Es_constant * y

    return SolverResult(
        z=z, y=y, slope=slope, moment=moment, shear=shear,
        soil_reaction=soil_reaction,
        Es=np.full(n_points, Es_constant),
        iterations=0, converged=True,
    )
