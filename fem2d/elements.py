"""
Finite element formulations for 2D plane-strain analysis.

Provides CST (3-node constant strain triangle) and Q4 (4-node
isoparametric quadrilateral) element stiffness and force routines,
plus Euler-Bernoulli beam elements for structural members.

All soil elements use DOF ordering: [u1, v1, u2, v2, ...].
Beam elements use: [ui, vi, theta_i, uj, vj, theta_j].
Node numbering is counter-clockwise for 2D elements.
"""

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Beam element dataclass
# ---------------------------------------------------------------------------

@dataclass
class BeamElement:
    """2D Euler-Bernoulli beam element.

    Parameters
    ----------
    node_i : int — start node index.
    node_j : int — end node index.
    EA : float — axial stiffness (kN).
    EI : float — flexural stiffness (kN·m²).
    weight_per_m : float — self-weight per unit length (kN/m). Default 0.
    """
    node_i: int
    node_j: int
    EA: float
    EI: float
    weight_per_m: float = 0.0

# 2×2 Gauss quadrature points and weights
_GP = 1.0 / np.sqrt(3.0)
_GAUSS_PTS_2x2 = [(-_GP, -_GP), (_GP, -_GP), (_GP, _GP), (-_GP, _GP)]


# ---------------------------------------------------------------------------
# CST (3-node triangle)
# ---------------------------------------------------------------------------

def cst_area(coords):
    """Compute triangle area from (3,2) coordinate array.

    Parameters
    ----------
    coords : (3, 2) array
        [[x1, y1], [x2, y2], [x3, y3]], counter-clockwise.

    Returns
    -------
    float
        Positive area.
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))


def cst_B(coords):
    """Compute the constant B matrix (3×6) for a CST element.

    Parameters
    ----------
    coords : (3, 2) array

    Returns
    -------
    B : (3, 6) array
        Strain-displacement matrix.
    A : float
        Element area.
    """
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    A2 = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    A = abs(A2) / 2.0

    b1, c1 = y2 - y3, x3 - x2
    b2, c2 = y3 - y1, x1 - x3
    b3, c3 = y1 - y2, x2 - x1

    B = (1.0 / A2) * np.array([
        [b1, 0, b2, 0, b3, 0],
        [0, c1, 0, c2, 0, c3],
        [c1, b1, c2, b2, c3, b3],
    ])
    return B, A


def cst_stiffness(coords, D, t=1.0):
    """Element stiffness matrix for a CST element.

    Parameters
    ----------
    coords : (3, 2) array
    D : (3, 3) array — constitutive matrix.
    t : float — thickness (1.0 for plane strain per-unit-length).

    Returns
    -------
    K : (6, 6) array
    """
    B, A = cst_B(coords)
    return t * A * (B.T @ D @ B)


def cst_body_force(coords, bx, by, t=1.0):
    """Consistent body force vector for a CST element.

    Parameters
    ----------
    coords : (3, 2) array
    bx, by : float — body force per unit volume.
    t : float — thickness.

    Returns
    -------
    f : (6,) array
    """
    A = cst_area(coords)
    f_node = t * A / 3.0
    return f_node * np.array([bx, by, bx, by, bx, by])


def cst_stress(coords, D, u_e):
    """Compute element stress from nodal displacements.

    Parameters
    ----------
    coords : (3, 2) array
    D : (3, 3) array
    u_e : (6,) array — element nodal displacements.

    Returns
    -------
    sigma : (3,) array — [sigma_x, sigma_y, tau_xy]
    epsilon : (3,) array — [eps_x, eps_y, gamma_xy]
    """
    B, _ = cst_B(coords)
    epsilon = B @ u_e
    sigma = D @ epsilon
    return sigma, epsilon


# ---------------------------------------------------------------------------
# Q4 (4-node isoparametric quad)
# ---------------------------------------------------------------------------

def q4_shape_derivs(xi, eta):
    """Shape function derivatives in natural coordinates.

    Returns
    -------
    dN_dxi : (4,) array
    dN_deta : (4,) array
    N : (4,) array — shape function values.
    """
    N = np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta),
    ]) / 4.0
    dN_dxi = np.array([
        -(1 - eta), (1 - eta), (1 + eta), -(1 + eta),
    ]) / 4.0
    dN_deta = np.array([
        -(1 - xi), -(1 + xi), (1 + xi), (1 - xi),
    ]) / 4.0
    return dN_dxi, dN_deta, N


def q4_B_detJ(coords, xi, eta):
    """Compute B matrix and Jacobian determinant at a natural coordinate point.

    Parameters
    ----------
    coords : (4, 2) array
    xi, eta : float — natural coordinates.

    Returns
    -------
    B : (3, 8) array
    detJ : float
    N : (4,) array
    """
    dN_dxi, dN_deta, N = q4_shape_derivs(xi, eta)
    J = np.array([dN_dxi, dN_deta]) @ coords  # (2, 2)
    detJ = np.linalg.det(J)
    Jinv = np.linalg.inv(J)
    dN_dxy = Jinv @ np.array([dN_dxi, dN_deta])  # (2, 4)

    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2 * i] = dN_dxy[0, i]
        B[1, 2 * i + 1] = dN_dxy[1, i]
        B[2, 2 * i] = dN_dxy[1, i]
        B[2, 2 * i + 1] = dN_dxy[0, i]
    return B, detJ, N


def q4_stiffness(coords, D, t=1.0):
    """Element stiffness matrix for a Q4 element (2×2 Gauss quadrature).

    Parameters
    ----------
    coords : (4, 2) array
    D : (3, 3) array
    t : float

    Returns
    -------
    K : (8, 8) array
    """
    K = np.zeros((8, 8))
    for xi, eta in _GAUSS_PTS_2x2:
        B, detJ, _ = q4_B_detJ(coords, xi, eta)
        K += B.T @ D @ B * detJ * t
    return K


def q4_body_force(coords, bx, by, t=1.0):
    """Consistent body force vector for a Q4 element.

    Parameters
    ----------
    coords : (4, 2) array
    bx, by : float — body force per unit volume.
    t : float

    Returns
    -------
    f : (8,) array
    """
    f = np.zeros(8)
    for xi, eta in _GAUSS_PTS_2x2:
        _, detJ, N = q4_B_detJ(coords, xi, eta)
        for i in range(4):
            f[2 * i] += N[i] * bx * detJ * t
            f[2 * i + 1] += N[i] * by * detJ * t
    return f


def q4_stress(coords, D, u_e):
    """Compute element stress at centroid from nodal displacements.

    Returns stress and strain at the element centroid (xi=0, eta=0).
    """
    B, _, _ = q4_B_detJ(coords, 0.0, 0.0)
    epsilon = B @ u_e
    sigma = D @ epsilon
    return sigma, epsilon


# ---------------------------------------------------------------------------
# 2D Euler-Bernoulli beam element
# ---------------------------------------------------------------------------

def beam2d_stiffness(coords_ij, EA, EI):
    """Element stiffness matrix for a 2-node Euler-Bernoulli beam.

    DOF ordering: [ui, vi, theta_i, uj, vj, theta_j].

    Parameters
    ----------
    coords_ij : (2, 2) array — [[xi, yi], [xj, yj]].
    EA : float — axial stiffness (kN).
    EI : float — flexural stiffness (kN*m^2).

    Returns
    -------
    K_global : (6, 6) array — stiffness in global coordinates.
    T : (6, 6) array — transformation matrix (local → global).
    L : float — element length.
    """
    dx = coords_ij[1, 0] - coords_ij[0, 0]
    dy = coords_ij[1, 1] - coords_ij[0, 1]
    L = math.sqrt(dx ** 2 + dy ** 2)
    c = dx / L  # cos(alpha)
    s = dy / L  # sin(alpha)

    # Local stiffness: axial + bending
    k_a = EA / L
    k1 = 12.0 * EI / L ** 3
    k2 = 6.0 * EI / L ** 2
    k3 = 4.0 * EI / L
    k4 = 2.0 * EI / L

    K_local = np.array([
        [ k_a,   0,    0,  -k_a,   0,    0  ],
        [  0,   k1,   k2,    0,  -k1,   k2  ],
        [  0,   k2,   k3,    0,  -k2,   k4  ],
        [-k_a,   0,    0,   k_a,   0,    0  ],
        [  0,  -k1,  -k2,    0,   k1,  -k2  ],
        [  0,   k2,   k4,    0,  -k2,   k3  ],
    ])

    # Rotation transformation: local → global
    T = np.array([
        [ c,  s,  0,  0,  0,  0],
        [-s,  c,  0,  0,  0,  0],
        [ 0,  0,  1,  0,  0,  0],
        [ 0,  0,  0,  c,  s,  0],
        [ 0,  0,  0, -s,  c,  0],
        [ 0,  0,  0,  0,  0,  1],
    ])

    K_global = T.T @ K_local @ T
    return K_global, T, L


def beam2d_internal_forces(coords_ij, EA, EI, u_beam_6):
    """Compute internal forces for a 2-node beam element.

    Parameters
    ----------
    coords_ij : (2, 2) array — [[xi, yi], [xj, yj]].
    EA : float — axial stiffness.
    EI : float — flexural stiffness.
    u_beam_6 : (6,) array — beam DOF displacements [ui, vi, thi, uj, vj, thj].

    Returns
    -------
    dict with keys:
        'axial_i', 'shear_i', 'moment_i' — forces at node i.
        'axial_j', 'shear_j', 'moment_j' — forces at node j.
        'length' — element length.
    """
    K_global, T, L = beam2d_stiffness(coords_ij, EA, EI)

    # Global forces
    f_global = K_global @ u_beam_6

    # Transform to local for axial/shear/moment
    u_local = T @ u_beam_6
    dx = coords_ij[1, 0] - coords_ij[0, 0]
    dy = coords_ij[1, 1] - coords_ij[0, 1]
    c = dx / L
    s = dy / L

    # Local stiffness for internal forces
    k_a = EA / L
    k1 = 12.0 * EI / L ** 3
    k2 = 6.0 * EI / L ** 2
    k3 = 4.0 * EI / L
    k4 = 2.0 * EI / L

    K_local = np.array([
        [ k_a,   0,    0,  -k_a,   0,    0  ],
        [  0,   k1,   k2,    0,  -k1,   k2  ],
        [  0,   k2,   k3,    0,  -k2,   k4  ],
        [-k_a,   0,    0,   k_a,   0,    0  ],
        [  0,  -k1,  -k2,    0,   k1,  -k2  ],
        [  0,   k2,   k4,    0,  -k2,   k3  ],
    ])

    f_local = K_local @ u_local

    return {
        'axial_i': float(f_local[0]),
        'shear_i': float(f_local[1]),
        'moment_i': float(f_local[2]),
        'axial_j': float(f_local[3]),
        'shear_j': float(f_local[4]),
        'moment_j': float(f_local[5]),
        'length': float(L),
    }


# ---------------------------------------------------------------------------
# Triangle quadrature (area coordinates, weights sum to 1)
# ---------------------------------------------------------------------------

# Rules keyed by number of points. Each entry: (points (n_gp, 3) area
# coordinates [L1, L2, L3], weights (n_gp,) summing to 1).
# 1-pt: exact degree 1 (CST). 3-pt interior: exact degree 2 (T6 default —
# integrates the straight-sided T6 stiffness exactly). 6-pt Dunavant:
# exact degree 4 (option for plasticity accuracy studies).
_TRI_6A = 0.445948490915965
_TRI_6B = 0.091576213509771
TRI_GAUSS = {
    1: (np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]),
        np.array([1.0])),
    3: (np.array([
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ]),
        np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])),
    6: (np.array([
            [1.0 - 2.0 * _TRI_6A, _TRI_6A, _TRI_6A],
            [_TRI_6A, 1.0 - 2.0 * _TRI_6A, _TRI_6A],
            [_TRI_6A, _TRI_6A, 1.0 - 2.0 * _TRI_6A],
            [1.0 - 2.0 * _TRI_6B, _TRI_6B, _TRI_6B],
            [_TRI_6B, 1.0 - 2.0 * _TRI_6B, _TRI_6B],
            [_TRI_6B, _TRI_6B, 1.0 - 2.0 * _TRI_6B],
        ]),
        np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)),
}


# ---------------------------------------------------------------------------
# T6 (6-node quadratic triangle)
# ---------------------------------------------------------------------------
# Node ordering: corners 0, 1, 2 (CCW) then midsides 3 (between 0-1),
# 4 (between 1-2), 5 (between 2-0).
# Shape functions in area coordinates L1, L2, L3 (Smith & Griffiths):
#   N_i = L_i (2 L_i - 1)   for corners
#   N_3 = 4 L1 L2,  N_4 = 4 L2 L3,  N_5 = 4 L3 L1

def t6_shape(L):
    """T6 shape functions at an area-coordinate point.

    Parameters
    ----------
    L : (3,) array-like — area coordinates [L1, L2, L3].

    Returns
    -------
    N : (6,) array
    """
    L1, L2, L3 = L
    return np.array([
        L1 * (2.0 * L1 - 1.0),
        L2 * (2.0 * L2 - 1.0),
        L3 * (2.0 * L3 - 1.0),
        4.0 * L1 * L2,
        4.0 * L2 * L3,
        4.0 * L3 * L1,
    ])


def t6_shape_derivs(L):
    """T6 shape functions and natural-coordinate derivatives.

    Uses (xi, eta) = (L2, L3) as the independent natural coordinates,
    with L1 = 1 - xi - eta.

    Returns
    -------
    N : (6,) array
    dN_dxi : (6,) array
    dN_deta : (6,) array
    """
    L1, L2, L3 = L
    N = t6_shape(L)
    dN_dxi = np.array([
        -(4.0 * L1 - 1.0),
        4.0 * L2 - 1.0,
        0.0,
        4.0 * (L1 - L2),
        4.0 * L3,
        -4.0 * L3,
    ])
    dN_deta = np.array([
        -(4.0 * L1 - 1.0),
        0.0,
        4.0 * L3 - 1.0,
        -4.0 * L2,
        4.0 * L2,
        4.0 * (L1 - L3),
    ])
    return N, dN_dxi, dN_deta


def t6_B_detJ(coords, L):
    """B matrix and Jacobian determinant for a T6 element at one point.

    Isoparametric formulation — valid for straight or (mildly) curved
    edges. For straight-sided elements with midside nodes at edge
    midpoints, detJ = 2A (constant).

    Parameters
    ----------
    coords : (6, 2) array — node coordinates (ordering above).
    L : (3,) array-like — area coordinates of the evaluation point.

    Returns
    -------
    B : (3, 12) array — strain-displacement matrix.
    detJ : float — Jacobian determinant (integration: dA = detJ/2 dxi deta).
    N : (6,) array — shape function values.
    """
    N, dN_dxi, dN_deta = t6_shape_derivs(L)
    J = np.array([dN_dxi, dN_deta]) @ coords  # (2, 2)
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    Jinv = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
    dN_dxy = Jinv @ np.array([dN_dxi, dN_deta])  # (2, 6)

    B = np.zeros((3, 12))
    B[0, 0::2] = dN_dxy[0]
    B[1, 1::2] = dN_dxy[1]
    B[2, 0::2] = dN_dxy[1]
    B[2, 1::2] = dN_dxy[0]
    return B, detJ, N


def t6_stiffness(coords, D, t=1.0, n_gp=3):
    """Element stiffness matrix for a T6 element.

    Parameters
    ----------
    coords : (6, 2) array
    D : (3, 3) array — constitutive matrix.
    t : float — thickness.
    n_gp : int — quadrature rule (1, 3, or 6). 3 is exact for
        straight-sided elastic T6.

    Returns
    -------
    K : (12, 12) array
    """
    pts, wts = TRI_GAUSS[n_gp]
    K = np.zeros((12, 12))
    for L, w in zip(pts, wts):
        B, detJ, _ = t6_B_detJ(coords, L)
        K += (B.T @ D @ B) * (0.5 * w * detJ * t)
    return K


def t6_body_force(coords, bx, by, t=1.0, n_gp=3):
    """Consistent body force vector for a T6 element.

    For straight-sided T6, the classic result: corner nodes receive zero,
    midside nodes receive A/3 each.

    Returns
    -------
    f : (12,) array
    """
    pts, wts = TRI_GAUSS[n_gp]
    f = np.zeros(12)
    for L, w in zip(pts, wts):
        _, dN_dxi, dN_deta = t6_shape_derivs(L)
        J = np.array([dN_dxi, dN_deta]) @ coords
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        N = t6_shape(L)
        scale = 0.5 * w * detJ * t
        f[0::2] += N * bx * scale
        f[1::2] += N * by * scale
    return f


def t6_stress(coords, D, u_e):
    """Element centroid stress/strain for a T6 element.

    Returns
    -------
    sigma : (3,) array — [sigma_x, sigma_y, tau_xy] at the centroid.
    epsilon : (3,) array
    """
    B, _, _ = t6_B_detJ(coords, np.array([1, 1, 1]) / 3.0)
    epsilon = B @ u_e
    sigma = D @ epsilon
    return sigma, epsilon
