"""
Tests for T6 (6-node quadratic triangle) elements: shape functions,
quadrature, consistent loads, mesh conversion, and the elastic pipeline.
"""

import numpy as np
import pytest

from fem2d.elements import (
    TRI_GAUSS, t6_shape, t6_shape_derivs, t6_B_detJ,
    t6_stiffness, t6_body_force, t6_stress,
    cst_stiffness, cst_body_force,
)
from fem2d.mesh import (
    convert_to_t6, t6_corner_elements, t6_boundary_edges,
    generate_rect_mesh, detect_boundary_nodes,
)
from fem2d.materials import elastic_D
from fem2d.assembly import (
    assemble_stiffness, assemble_gravity, assemble_surface_load,
    apply_bcs_penalty, solve_linear, recover_element_stresses,
)
from fem2d.solver import solve_elastic


# Reference T6 element (straight sides, midsides at midpoints)
T6_COORDS = np.array([
    [0.0, 0.0], [2.0, 0.0], [0.0, 1.5],
    [1.0, 0.0], [1.0, 0.75], [0.0, 0.75],
])


class TestQuadrature:
    def test_weights_sum_to_one(self):
        for n_gp, (pts, wts) in TRI_GAUSS.items():
            assert np.isclose(wts.sum(), 1.0), f"rule {n_gp}"
            assert np.allclose(pts.sum(axis=1), 1.0), f"rule {n_gp}"

    def test_3pt_exact_quadratic(self):
        # Integrate L1^2 over unit triangle: exact = A/6 with A*sum(w L1^2)
        pts, wts = TRI_GAUSS[3]
        # int L1^2 dA = A * 2!/(2+2)! * 2 = A/6
        val = (wts * pts[:, 0] ** 2).sum()
        assert np.isclose(val, 1.0 / 6.0)

    def test_6pt_exact_quartic(self):
        pts, wts = TRI_GAUSS[6]
        # int L1^4 dA = A * 4! * 2 / 6! = A/15
        val = (wts * pts[:, 0] ** 4).sum()
        assert np.isclose(val, 1.0 / 15.0)


class TestT6ShapeFunctions:
    # Area coordinates of the 6 nodes (corners then midsides)
    NODE_L = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5],
    ], dtype=float)

    def test_kronecker_delta(self):
        for i, L in enumerate(self.NODE_L):
            N = t6_shape(L)
            expected = np.zeros(6)
            expected[i] = 1.0
            assert np.allclose(N, expected), f"node {i}"

    def test_partition_of_unity(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            r = rng.random(3)
            L = r / r.sum()
            N, dN_dxi, dN_deta = t6_shape_derivs(L)
            assert np.isclose(N.sum(), 1.0)
            assert np.isclose(dN_dxi.sum(), 0.0, atol=1e-12)
            assert np.isclose(dN_deta.sum(), 0.0, atol=1e-12)

    def test_detJ_equals_2A_straight_sides(self):
        A = 0.5 * 2.0 * 1.5
        for L in TRI_GAUSS[3][0]:
            _, detJ, _ = t6_B_detJ(T6_COORDS, L)
            assert np.isclose(detJ, 2.0 * A)


class TestT6Element:
    def test_stiffness_symmetric_psd(self):
        D = elastic_D(1e4, 0.3)
        K = t6_stiffness(T6_COORDS, D)
        assert np.allclose(K, K.T, atol=1e-8)
        eigvals = np.linalg.eigvalsh(K)
        # 3 rigid body modes -> 3 near-zero eigenvalues, rest positive
        assert np.sum(np.abs(eigvals) < 1e-6) == 3
        assert np.all(eigvals > -1e-6)

    def test_stiffness_3pt_vs_6pt_identical(self):
        # 3-pt rule is exact for straight-sided T6 (integrand degree 2)
        D = elastic_D(1e4, 0.3)
        K3 = t6_stiffness(T6_COORDS, D, n_gp=3)
        K6 = t6_stiffness(T6_COORDS, D, n_gp=6)
        assert np.allclose(K3, K6, rtol=1e-10)

    def test_constant_strain_patch(self):
        """Linear displacement field -> exact constant strain at all GPs."""
        a, b, c, d = 1e-3, 2e-3, -5e-4, 3e-4
        # u = a*x + b*y, v = c*x + d*y
        u_e = np.zeros(12)
        u_e[0::2] = a * T6_COORDS[:, 0] + b * T6_COORDS[:, 1]
        u_e[1::2] = c * T6_COORDS[:, 0] + d * T6_COORDS[:, 1]
        expected = np.array([a, d, b + c])
        for L in TRI_GAUSS[6][0]:
            B, _, _ = t6_B_detJ(T6_COORDS, L)
            eps = B @ u_e
            assert np.allclose(eps, expected, atol=1e-12)

    def test_quadratic_field_exact(self):
        """T6 reproduces a quadratic displacement field exactly."""
        # u = x^2, v = x*y -> eps_xx = 2x, eps_yy = x, gamma = y + 0...
        u_e = np.zeros(12)
        u_e[0::2] = T6_COORDS[:, 0] ** 2
        u_e[1::2] = T6_COORDS[:, 0] * T6_COORDS[:, 1]
        for L in TRI_GAUSS[3][0]:
            B, _, N = t6_B_detJ(T6_COORDS, L)
            x = N @ T6_COORDS[:, 0]
            y = N @ T6_COORDS[:, 1]
            eps = B @ u_e
            assert np.allclose(eps, [2 * x, x, y], atol=1e-10)

    def test_consistent_gravity_midside_only(self):
        """Straight-sided T6: corners get 0, midsides get A/3 each."""
        A = 0.5 * 2.0 * 1.5
        f = t6_body_force(T6_COORDS, 0.0, -1.0)
        fy = f[1::2]
        assert np.allclose(fy[:3], 0.0, atol=1e-12)
        assert np.allclose(fy[3:], -A / 3.0)
        assert np.isclose(fy.sum(), -A)


class TestConvertToT6:
    def test_counts_and_midsides(self):
        nodes, elements = generate_rect_mesh(0, 2, 0, 1, 2, 1)
        n_corner = len(nodes)
        nodes6, elems6 = convert_to_t6(nodes, elements)
        # Euler: edges = (3*n_elem + n_boundary_edges)/2
        assert elems6.shape == (len(elements), 6)
        assert np.array_equal(elems6[:, :3], elements)
        # Midside coordinates are edge midpoints
        for e in range(len(elements)):
            n0, n1, n2, m01, m12, m20 = elems6[e]
            assert np.allclose(nodes6[m01], 0.5 * (nodes6[n0] + nodes6[n1]))
            assert np.allclose(nodes6[m12], 0.5 * (nodes6[n1] + nodes6[n2]))
            assert np.allclose(nodes6[m20], 0.5 * (nodes6[n2] + nodes6[n0]))
        # Shared edges produce one midside node (no duplicates)
        mids = elems6[:, 3:].ravel()
        n_unique_edges = len(set(mids.tolist()))
        assert len(nodes6) == n_corner + n_unique_edges

    def test_corner_extraction(self):
        nodes, elements = generate_rect_mesh(0, 1, 0, 1, 1, 1)
        _, elems6 = convert_to_t6(nodes, elements)
        assert np.array_equal(t6_corner_elements(elems6), elements)

    def test_boundary_edges_3node(self):
        nodes, elements = generate_rect_mesh(0, 2, 0, 1, 2, 1)
        nodes6, elems6 = convert_to_t6(nodes, elements)
        # Find top corner edge
        top = [n for n in range(len(nodes)) if abs(nodes[n, 1] - 1.0) < 1e-9]
        top.sort(key=lambda n: nodes[n, 0])
        corner_edges = [(top[i], top[i + 1]) for i in range(len(top) - 1)]
        edges3 = t6_boundary_edges(elems6, corner_edges)
        for (ni, nm, nj) in edges3:
            assert np.allclose(nodes6[nm], 0.5 * (nodes6[ni] + nodes6[nj]))

    def test_midside_nodes_detected_in_bcs(self):
        nodes, elements = generate_rect_mesh(0, 2, -1, 0, 4, 2)
        nodes6, elems6 = convert_to_t6(nodes, elements)
        bc = detect_boundary_nodes(nodes6)
        # Base of 4x2 mesh: 5 corner nodes + 4 midside nodes = 9
        assert len(bc['fixed_base']) == 9


class TestT6ElasticPipeline:
    def test_gravity_column_stress(self):
        """sigma_yy = -gamma*depth (linear field — T6 near-exact)."""
        gamma, depth = 20.0, 10.0
        nodes, elements = generate_rect_mesh(0, 4, -depth, 0, 4, 8)
        nodes6, elems6 = convert_to_t6(nodes, elements)
        bc = detect_boundary_nodes(nodes6)
        D = elastic_D(2e4, 0.3)
        u, stresses, strains = solve_elastic(nodes6, elems6, D, gamma, bc)
        # Element centroid depths (corner average)
        cent_y = nodes6[elems6[:, :3]].mean(axis=1)[:, 1]
        expected = gamma * cent_y  # tension-positive: compression negative
        assert np.allclose(stresses[:, 1], expected, rtol=0.02, atol=0.5)

    def test_t6_beats_cst_on_bending(self):
        """Tip-loaded cantilever: T6 captures bending; CST locks.

        Analytical (Timoshenko, plane strain): delta = PL^3/(3EI) +
        shear term. T6 should be within ~5%; CST on the same corner mesh
        is much stiffer.
        """
        Lb, h = 8.0, 1.0
        E, nu, P = 1e6, 0.25, 1.0
        nodes, elements = generate_rect_mesh(0, Lb, -h / 2, h / 2, 16, 2)
        D = elastic_D(E, nu)

        def tip_deflection(nds, elems):
            n_dof = 2 * len(nds)
            K = assemble_stiffness(nds, elems, D)
            F = np.zeros(n_dof)
            # Distribute tip load over right-edge nodes
            right = [n for n in range(len(nds))
                     if abs(nds[n, 0] - Lb) < 1e-9]
            for n in right:
                F[2 * n + 1] = -P / len(right)
            # Clamp left edge
            left = [n for n in range(len(nds)) if abs(nds[n, 0]) < 1e-9]
            bc = {'fixed_base': np.array(left), 'roller_left': np.array([]),
                  'roller_right': np.array([])}
            K_bc, F_bc = apply_bcs_penalty(K, F, bc)
            u = solve_linear(K_bc, F_bc)
            tip = [n for n in right if abs(nds[n, 1]) < 0.26]
            return float(np.mean(u[[2 * n + 1 for n in tip]]))

        d_cst = abs(tip_deflection(nodes, elements))
        nodes6, elems6 = convert_to_t6(nodes, elements)
        d_t6 = abs(tip_deflection(nodes6, elems6))

        E_star = E / (1.0 - nu ** 2)  # plane strain
        I = h ** 3 / 12.0
        d_beam = P * Lb ** 3 / (3.0 * E_star * I)
        # T6 within 8% of beam theory (incl. shear deformation ~0.2%)
        assert abs(d_t6 - d_beam) / d_beam < 0.08, (d_t6, d_beam)
        # CST locks: at least 25% too stiff on this mesh
        assert d_cst < 0.75 * d_t6, (d_cst, d_t6)

    def test_quadratic_surface_load_total(self):
        nodes, elements = generate_rect_mesh(0, 2, -1, 0, 2, 1)
        nodes6, elems6 = convert_to_t6(nodes, elements)
        top = [n for n in range(len(nodes)) if abs(nodes[n, 1]) < 1e-9]
        top.sort(key=lambda n: nodes[n, 0])
        corner_edges = [(top[i], top[i + 1]) for i in range(len(top) - 1)]
        edges3 = t6_boundary_edges(elems6, corner_edges)
        q = -10.0
        F = assemble_surface_load(nodes6, edges3, 0.0, q)
        assert np.isclose(F[1::2].sum(), q * 2.0)
        assert np.isclose(F[0::2].sum(), 0.0)

    def test_uniform_load_settlement_matches_oedometer(self):
        """Laterally confined column under uniform q: u_y = q*H/M."""
        E, nu, q, H = 1e4, 0.3, 50.0, 4.0
        nodes, elements = generate_rect_mesh(0, 2, -H, 0, 2, 4)
        nodes6, elems6 = convert_to_t6(nodes, elements)
        bc = detect_boundary_nodes(nodes6)
        top = [n for n in range(len(nodes)) if abs(nodes[n, 1]) < 1e-9]
        top.sort(key=lambda n: nodes[n, 0])
        corner_edges = [(top[i], top[i + 1]) for i in range(len(top) - 1)]
        edges3 = t6_boundary_edges(elems6, corner_edges)
        D = elastic_D(E, nu)
        u, _, _ = solve_elastic(nodes6, elems6, D, 0.0, bc,
                                surface_loads=[(edges3, 0.0, -q)])
        M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))  # constrained modulus
        expected = q * H / M
        top6 = [n for n in range(len(nodes6))
                if abs(nodes6[n, 1]) < 1e-9]
        settle = -np.mean(u[[2 * n + 1 for n in top6]])
        assert np.isclose(settle, expected, rtol=1e-3)
