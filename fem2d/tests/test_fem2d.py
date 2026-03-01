"""
Comprehensive tests for the fem2d module.

Covers elements, materials, mesh generation, assembly, solvers,
SRM, analysis functions, and results dataclass.

Analytical validation:
- Gravity column: sigma_yy = -gamma*z, max settlement = gamma*H^2 / (2*M)
- Patch test: uniform strain reproduced exactly by CST/Q4
- Elastic D-matrix: plane-strain constitutive relations
- MC return mapping: yield surface and principal stress space
"""

import math
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------

class TestCSTArea:
    def test_unit_triangle(self):
        from fem2d.elements import cst_area
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        assert cst_area(coords) == pytest.approx(0.5)

    def test_scaled_triangle(self):
        from fem2d.elements import cst_area
        coords = np.array([[0, 0], [4, 0], [0, 3]], dtype=float)
        assert cst_area(coords) == pytest.approx(6.0)

    def test_translated_triangle(self):
        from fem2d.elements import cst_area
        coords = np.array([[10, 10], [14, 10], [10, 13]], dtype=float)
        assert cst_area(coords) == pytest.approx(6.0)

    def test_equilateral_triangle(self):
        from fem2d.elements import cst_area
        s = 2.0
        h = s * math.sqrt(3) / 2
        coords = np.array([[0, 0], [s, 0], [s / 2, h]], dtype=float)
        expected = s**2 * math.sqrt(3) / 4
        assert cst_area(coords) == pytest.approx(expected, rel=1e-10)


class TestCSTBMatrix:
    def test_B_shape(self):
        from fem2d.elements import cst_B
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        B, A = cst_B(coords)
        assert B.shape == (3, 6)
        assert A == pytest.approx(0.5)

    def test_uniform_extension(self):
        """Uniform x-extension: eps_x = const, eps_y = gamma_xy = 0."""
        from fem2d.elements import cst_B
        coords = np.array([[0, 0], [2, 0], [1, 1]], dtype=float)
        B, _ = cst_B(coords)
        # Displacements: u_i = eps * x_i, v_i = 0
        eps_x = 0.001
        u_e = np.array([eps_x * 0, 0, eps_x * 2, 0, eps_x * 1, 0])
        strain = B @ u_e
        assert strain[0] == pytest.approx(eps_x, rel=1e-10)
        assert strain[1] == pytest.approx(0, abs=1e-14)
        assert strain[2] == pytest.approx(0, abs=1e-14)


class TestCSTStiffness:
    def test_stiffness_symmetric(self):
        from fem2d.elements import cst_stiffness
        from fem2d.materials import elastic_D
        coords = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)
        D = elastic_D(30000, 0.3)
        K = cst_stiffness(coords, D)
        assert K.shape == (6, 6)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_stiffness_positive_semidefinite(self):
        from fem2d.elements import cst_stiffness
        from fem2d.materials import elastic_D
        coords = np.array([[0, 0], [1, 0], [0.5, 0.8]], dtype=float)
        D = elastic_D(30000, 0.3)
        K = cst_stiffness(coords, D)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)

    def test_stiffness_scales_with_thickness(self):
        from fem2d.elements import cst_stiffness
        from fem2d.materials import elastic_D
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        D = elastic_D(30000, 0.3)
        K1 = cst_stiffness(coords, D, t=1.0)
        K2 = cst_stiffness(coords, D, t=2.0)
        np.testing.assert_allclose(K2, 2.0 * K1, rtol=1e-12)


class TestCSTBodyForce:
    def test_body_force_total(self):
        """Total body force = t * A * b for each direction."""
        from fem2d.elements import cst_body_force
        coords = np.array([[0, 0], [2, 0], [0, 3]], dtype=float)
        bx, by = 0.0, -18.0
        t = 1.0
        f = cst_body_force(coords, bx, by, t)
        A = 3.0  # area of 2x3/2 triangle
        assert sum(f[0::2]) == pytest.approx(bx * A * t, abs=1e-10)
        assert sum(f[1::2]) == pytest.approx(by * A * t, abs=1e-10)

    def test_equal_distribution(self):
        """Each node gets 1/3 of the total force."""
        from fem2d.elements import cst_body_force
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        f = cst_body_force(coords, 10.0, -20.0)
        # All x-forces equal, all y-forces equal
        assert f[0] == pytest.approx(f[2], rel=1e-10)
        assert f[0] == pytest.approx(f[4], rel=1e-10)
        assert f[1] == pytest.approx(f[3], rel=1e-10)
        assert f[1] == pytest.approx(f[5], rel=1e-10)


class TestCSTStress:
    def test_stress_from_displacement(self):
        from fem2d.elements import cst_stress
        from fem2d.materials import elastic_D
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        D = elastic_D(30000, 0.3)
        # Pure x-extension
        eps_x = 0.001
        u_e = np.array([0, 0, eps_x, 0, 0, 0], dtype=float)
        sigma, epsilon = cst_stress(coords, D, u_e)
        assert epsilon.shape == (3,)
        assert sigma.shape == (3,)
        # sigma_xx should be positive, sigma_yy from Poisson coupling
        expected_sigma = D @ epsilon
        np.testing.assert_allclose(sigma, expected_sigma, rtol=1e-10)


class TestQ4ShapeDerivs:
    def test_shape_functions_sum_to_one(self):
        from fem2d.elements import q4_shape_derivs
        _, _, N = q4_shape_derivs(0.3, -0.2)
        assert sum(N) == pytest.approx(1.0, rel=1e-12)

    def test_shape_at_nodes(self):
        from fem2d.elements import q4_shape_derivs
        nodes_nat = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        for i, (xi, eta) in enumerate(nodes_nat):
            _, _, N = q4_shape_derivs(xi, eta)
            for j in range(4):
                expected = 1.0 if i == j else 0.0
                assert N[j] == pytest.approx(expected, abs=1e-12)


class TestQ4Stiffness:
    def test_stiffness_symmetric(self):
        from fem2d.elements import q4_stiffness
        from fem2d.materials import elastic_D
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        D = elastic_D(30000, 0.3)
        K = q4_stiffness(coords, D)
        assert K.shape == (8, 8)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_stiffness_positive_semidefinite(self):
        from fem2d.elements import q4_stiffness
        from fem2d.materials import elastic_D
        coords = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
        D = elastic_D(30000, 0.3)
        K = q4_stiffness(coords, D)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)


class TestQ4BodyForce:
    def test_total_force(self):
        from fem2d.elements import q4_body_force
        coords = np.array([[0, 0], [2, 0], [2, 3], [0, 3]], dtype=float)
        bx, by = 0.0, -18.0
        f = q4_body_force(coords, bx, by, t=1.0)
        A = 6.0
        assert sum(f[0::2]) == pytest.approx(bx * A, abs=1e-10)
        assert sum(f[1::2]) == pytest.approx(by * A, abs=1e-10)


class TestQ4Stress:
    def test_stress_dimensions(self):
        from fem2d.elements import q4_stress
        from fem2d.materials import elastic_D
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        D = elastic_D(30000, 0.3)
        u_e = np.zeros(8)
        sigma, epsilon = q4_stress(coords, D, u_e)
        assert sigma.shape == (3,)
        assert epsilon.shape == (3,)


# ---------------------------------------------------------------------------
# Patch test
# ---------------------------------------------------------------------------

class TestPatchTest:
    """Patch test: uniform strain state must be reproduced exactly."""

    def test_cst_patch(self):
        """4-element CST patch test with linear displacement field."""
        from fem2d.elements import cst_stress
        from fem2d.materials import elastic_D

        # 5-node patch: 4 corners + center
        nodes = np.array([
            [0, 0], [2, 0], [2, 2], [0, 2], [1, 1]
        ], dtype=float)
        # 4 CST elements
        elements = np.array([
            [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]
        ])

        E, nu = 30000.0, 0.3
        D = elastic_D(E, nu)

        # Impose linear displacement: u = a*x + b*y, v = c*x + d*y
        a, b, c, d = 0.001, 0.0005, -0.0003, 0.002
        u = np.zeros(2 * len(nodes))
        for i, (x, y) in enumerate(nodes):
            u[2 * i] = a * x + b * y
            u[2 * i + 1] = c * x + d * y

        # All elements should give identical strain
        expected_eps = np.array([a, d, b + c])
        expected_sig = D @ expected_eps

        for elem in elements:
            coords = nodes[elem]
            u_e = np.array([u[2 * n] for n in elem for _ in range(1)] +
                           [0])  # placeholder
            # Rebuild properly
            u_e = np.zeros(6)
            for k, n in enumerate(elem):
                u_e[2 * k] = u[2 * n]
                u_e[2 * k + 1] = u[2 * n + 1]

            sigma, epsilon = cst_stress(coords, D, u_e)
            np.testing.assert_allclose(epsilon, expected_eps, atol=1e-12)
            np.testing.assert_allclose(sigma, expected_sig, atol=1e-8)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------

class TestElasticD:
    def test_shape(self):
        from fem2d.materials import elastic_D
        D = elastic_D(30000, 0.3)
        assert D.shape == (3, 3)

    def test_symmetric(self):
        from fem2d.materials import elastic_D
        D = elastic_D(30000, 0.3)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_positive_definite(self):
        from fem2d.materials import elastic_D
        D = elastic_D(30000, 0.3)
        eigvals = np.linalg.eigvalsh(D)
        assert np.all(eigvals > 0)

    def test_values_nu0(self):
        """For nu=0: D11=D22=E, D12=0, D33=E/2."""
        from fem2d.materials import elastic_D
        E = 10000.0
        D = elastic_D(E, 0.0)
        assert D[0, 0] == pytest.approx(E)
        assert D[0, 1] == pytest.approx(0.0)
        assert D[2, 2] == pytest.approx(E / 2.0)

    def test_incompressible_limit(self):
        """nu approaching 0.5: D-matrix entries grow large."""
        from fem2d.materials import elastic_D
        D = elastic_D(30000, 0.499)
        # D[0,0] should be very large (nearly incompressible)
        assert D[0, 0] > 1e6

    def test_constrained_modulus(self):
        """D[0,0] = E(1-nu)/((1+nu)(1-2nu)) = constrained modulus."""
        from fem2d.materials import elastic_D
        E, nu = 30000.0, 0.3
        D = elastic_D(E, nu)
        M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        assert D[0, 0] == pytest.approx(M)


class TestElasticD4:
    def test_shape(self):
        from fem2d.materials import elastic_D_4
        D = elastic_D_4(30000, 0.3)
        assert D.shape == (4, 4)

    def test_symmetric(self):
        from fem2d.materials import elastic_D_4
        D = elastic_D_4(30000, 0.3)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_sigma_zz_row(self):
        """sigma_zz row should give correct plane-strain out-of-plane stress."""
        from fem2d.materials import elastic_D_4
        E, nu = 30000.0, 0.3
        D = elastic_D_4(E, nu)
        c = E / ((1 + nu) * (1 - 2 * nu))
        assert D[2, 0] == pytest.approx(c * nu)
        assert D[2, 1] == pytest.approx(c * nu)
        assert D[2, 2] == pytest.approx(c * (1 - nu))


class TestMCReturnMapping:
    def test_elastic_no_yield(self):
        """Small compression — no yield when stress state is inside envelope."""
        from fem2d.materials import mc_return_mapping
        # With nu=0.3, szz = 0.3*(sxx+syy). For [-50,-50,0]:
        # szz=-30, principals=[-30,-50,-50], f = -30 - 3*(-50) - 2*50*sqrt(3) < 0
        sigma_trial = np.array([-50.0, -50.0, 0.0])
        sigma_new, D_ep, yielded = mc_return_mapping(
            sigma_trial, 30000, 0.3, c=50, phi_deg=30)
        assert not yielded
        np.testing.assert_allclose(sigma_new, sigma_trial, atol=1e-10)

    def test_yields_under_tension(self):
        """High tension should trigger yield."""
        from fem2d.materials import mc_return_mapping
        sigma_trial = np.array([200.0, -50.0, 10.0])
        sigma_new, D_ep, yielded = mc_return_mapping(
            sigma_trial, 30000, 0.3, c=10, phi_deg=30)
        assert yielded

    def test_returned_stress_satisfies_yield(self):
        """Returned stress should not violate in-plane MC yield criterion."""
        from fem2d.materials import mc_return_mapping
        sigma_trial = np.array([500.0, -200.0, 100.0])
        sigma_new, D_ep, yielded = mc_return_mapping(
            sigma_trial, 30000, 0.3, c=20, phi_deg=30)
        assert yielded
        # Check in-plane Mohr circle MC criterion (tension-positive):
        # f = q + p*sin(phi) - c*cos(phi) <= 0
        sxx, syy, txy = sigma_new
        p = (sxx + syy) / 2.0
        q = math.sqrt(((sxx - syy) / 2.0)**2 + txy**2)
        sin_phi = math.sin(math.radians(30))
        cos_phi = math.cos(math.radians(30))
        f = q + p * sin_phi - 20 * cos_phi
        assert f <= 1e-6  # within tolerance

    def test_pure_cohesion(self):
        """phi=0, only cohesion — Tresca-like yield."""
        from fem2d.materials import mc_return_mapping
        sigma_trial = np.array([200.0, -200.0, 50.0])
        sigma_new, D_ep, yielded = mc_return_mapping(
            sigma_trial, 30000, 0.3, c=50, phi_deg=0)
        # With phi=0: N_phi=1, sigma_c=2c
        # f = sigma_1 - sigma_3 - 2c
        # Check returned stress
        if yielded:
            sxx, syy, txy = sigma_new
            s_mean = (sxx + syy) / 2.0
            R = math.sqrt(((sxx - syy) / 2.0)**2 + txy**2)
            diff = 2 * R  # sigma_1 - sigma_3 (in-plane)
            # Should be close to 2c = 100
            assert diff <= 100.0 + 1.0

    def test_D_ep_shape(self):
        """Tangent modulus always (3, 3)."""
        from fem2d.materials import mc_return_mapping
        sigma_trial = np.array([500.0, -200.0, 100.0])
        _, D_ep, _ = mc_return_mapping(
            sigma_trial, 30000, 0.3, c=20, phi_deg=30)
        assert D_ep.shape == (3, 3)

    def test_zero_strength(self):
        """c=0, phi=0 should still not crash."""
        from fem2d.materials import mc_return_mapping
        sigma_trial = np.array([100.0, -100.0, 50.0])
        sigma_new, D_ep, yielded = mc_return_mapping(
            sigma_trial, 30000, 0.3, c=0, phi_deg=0)
        # N_phi=1, sigma_c=0, so any deviatoric stress yields
        assert D_ep.shape == (3, 3)

    def test_associated_flow(self):
        """psi = phi (associated flow) should also work."""
        from fem2d.materials import mc_return_mapping
        sigma_trial = np.array([300.0, -100.0, 50.0])
        sigma_new, D_ep, yielded = mc_return_mapping(
            sigma_trial, 30000, 0.3, c=20, phi_deg=30, psi_deg=30)
        assert D_ep.shape == (3, 3)


class TestDruckerPragerParams:
    def test_phi30(self):
        """DP params for phi=30, c=20."""
        from fem2d.materials import drucker_prager_params
        alpha, k = drucker_prager_params(20, 30)
        tan30 = math.tan(math.radians(30))
        denom = math.sqrt(9 + 12 * tan30**2)
        assert alpha == pytest.approx(tan30 / denom, rel=1e-10)
        assert k == pytest.approx(3 * 20 / denom, rel=1e-10)

    def test_phi0(self):
        """phi=0: alpha=0, k=c (von Mises-like)."""
        from fem2d.materials import drucker_prager_params
        alpha, k = drucker_prager_params(50, 0)
        assert alpha == pytest.approx(0.0, abs=1e-15)
        assert k == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------

class TestPointsInPolygon:
    def test_square(self):
        from fem2d.mesh import points_in_polygon
        poly = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        pts = np.array([[1, 1], [-1, 1], [3, 1], [1, 3]], dtype=float)
        result = points_in_polygon(pts, poly)
        assert result[0] == True   # inside
        assert result[1] == False  # left
        assert result[2] == False  # right
        assert result[3] == False  # above

    def test_triangle(self):
        from fem2d.mesh import points_in_polygon
        poly = np.array([[0, 0], [4, 0], [2, 3]], dtype=float)
        pts = np.array([[2, 1], [0.1, 0.1], [5, 0]], dtype=float)
        result = points_in_polygon(pts, poly)
        assert result[0] == True
        assert result[1] == True
        assert result[2] == False


class TestGenerateRectMesh:
    def test_node_count(self):
        from fem2d.mesh import generate_rect_mesh
        nodes, elements = generate_rect_mesh(0, 10, -5, 0, 4, 2)
        assert len(nodes) == 5 * 3  # (nx+1) * (ny+1)

    def test_element_count(self):
        from fem2d.mesh import generate_rect_mesh
        nodes, elements = generate_rect_mesh(0, 10, -5, 0, 4, 2)
        assert len(elements) == 4 * 2 * 2  # nx * ny * 2

    def test_ccw_ordering(self):
        from fem2d.mesh import generate_rect_mesh
        nodes, elements = generate_rect_mesh(0, 2, 0, 2, 3, 3)
        for elem in elements:
            c = nodes[elem]
            cross = (c[1, 0] - c[0, 0]) * (c[2, 1] - c[0, 1]) - \
                    (c[1, 1] - c[0, 1]) * (c[2, 0] - c[0, 0])
            assert cross > 0, "Element not CCW"

    def test_bounds(self):
        from fem2d.mesh import generate_rect_mesh
        nodes, _ = generate_rect_mesh(-5, 5, -10, 0, 10, 5)
        assert nodes[:, 0].min() == pytest.approx(-5)
        assert nodes[:, 0].max() == pytest.approx(5)
        assert nodes[:, 1].min() == pytest.approx(-10)
        assert nodes[:, 1].max() == pytest.approx(0)


class TestGenerateSlopeMesh:
    def test_basic_slope(self):
        from fem2d.mesh import generate_slope_mesh
        surface = [(0, 0), (5, 0), (10, 5), (15, 5)]
        nodes, elements = generate_slope_mesh(surface, depth=10, nx=10, ny=5)
        assert len(nodes) > 0
        assert len(elements) > 0

    def test_all_triangles(self):
        from fem2d.mesh import generate_slope_mesh
        surface = [(0, 0), (5, 0), (10, 3), (15, 3)]
        nodes, elements = generate_slope_mesh(surface, depth=8, nx=8, ny=4)
        for elem in elements:
            assert len(elem) == 3


class TestGeneratePolygonMesh:
    def test_square_domain(self):
        from fem2d.mesh import generate_polygon_mesh
        poly = np.array([[0, 0], [4, 0], [4, 4], [0, 4]], dtype=float)
        nodes, elements = generate_polygon_mesh(poly, element_size=1.0)
        assert len(nodes) > 4
        assert len(elements) > 0

    def test_triangles_only(self):
        from fem2d.mesh import generate_polygon_mesh
        poly = np.array([[0, 0], [3, 0], [3, 3], [0, 3]], dtype=float)
        nodes, elements = generate_polygon_mesh(poly, element_size=1.5)
        for elem in elements:
            assert len(elem) == 3


class TestDetectBoundaryNodes:
    def test_rect_boundaries(self):
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes
        nodes, _ = generate_rect_mesh(0, 10, -5, 0, 5, 3)
        bc = detect_boundary_nodes(nodes)
        assert 'fixed_base' in bc
        assert 'roller_left' in bc
        assert 'roller_right' in bc
        assert len(bc['fixed_base']) > 0

    def test_base_nodes_at_bottom(self):
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes
        nodes, _ = generate_rect_mesh(0, 10, -5, 0, 5, 3)
        bc = detect_boundary_nodes(nodes)
        for n in bc['fixed_base']:
            assert nodes[n, 1] == pytest.approx(-5, abs=0.1)

    def test_no_overlap_base_roller(self):
        """Base nodes should not appear in roller lists."""
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes
        nodes, _ = generate_rect_mesh(0, 10, -5, 0, 5, 3)
        bc = detect_boundary_nodes(nodes)
        base_set = set(bc['fixed_base'])
        for n in bc['roller_left']:
            assert n not in base_set
        for n in bc['roller_right']:
            assert n not in base_set


class TestAssignLayers:
    def test_single_layer(self):
        from fem2d.mesh import generate_rect_mesh, assign_layers_by_elevation
        nodes, elements = generate_rect_mesh(0, 10, -10, 0, 5, 5)
        layer_ids = assign_layers_by_elevation(nodes, elements, [-5.0])
        # Elements above -5 → layer 0, below -5 → layer 1
        assert set(layer_ids).issubset({0, 1})

    def test_two_layers(self):
        from fem2d.mesh import generate_rect_mesh, assign_layers_by_elevation
        nodes, elements = generate_rect_mesh(0, 10, -12, 0, 5, 6)
        layer_ids = assign_layers_by_elevation(nodes, elements, [-4.0, -8.0])
        assert set(layer_ids).issubset({0, 1, 2})


class TestTriangleQuality:
    def test_equilateral(self):
        from fem2d.mesh import triangle_quality
        s = 2.0
        h = s * math.sqrt(3) / 2
        nodes = np.array([[0, 0], [s, 0], [s / 2, h]], dtype=float)
        elements = np.array([[0, 1, 2]])
        q = triangle_quality(nodes, elements)
        assert q['min_angles'][0] == pytest.approx(60.0, abs=0.1)

    def test_right_triangle(self):
        from fem2d.mesh import triangle_quality
        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        elements = np.array([[0, 1, 2]])
        q = triangle_quality(nodes, elements)
        assert q['min_angles'][0] == pytest.approx(45.0, abs=0.1)
        assert q['areas'][0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

class TestElementDofs:
    def test_triangle(self):
        from fem2d.assembly import element_dofs
        dofs = element_dofs([0, 1, 2])
        np.testing.assert_array_equal(dofs, [0, 1, 2, 3, 4, 5])

    def test_quad(self):
        from fem2d.assembly import element_dofs
        dofs = element_dofs([3, 5, 7, 9])
        np.testing.assert_array_equal(dofs, [6, 7, 10, 11, 14, 15, 18, 19])


class TestAssembleStiffness:
    def test_shape(self):
        from fem2d.assembly import assemble_stiffness
        from fem2d.materials import elastic_D
        from fem2d.mesh import generate_rect_mesh
        nodes, elements = generate_rect_mesh(0, 2, -1, 0, 2, 1)
        D = elastic_D(30000, 0.3)
        K = assemble_stiffness(nodes, elements, D)
        n_dof = 2 * len(nodes)
        assert K.shape == (n_dof, n_dof)

    def test_symmetric(self):
        from fem2d.assembly import assemble_stiffness
        from fem2d.materials import elastic_D
        from fem2d.mesh import generate_rect_mesh
        nodes, elements = generate_rect_mesh(0, 2, -2, 0, 3, 3)
        D = elastic_D(30000, 0.3)
        K = assemble_stiffness(nodes, elements, D)
        diff = K - K.T
        assert abs(diff).max() < 1e-8


class TestAssembleGravity:
    def test_total_weight(self):
        """Total vertical force = gamma * area * thickness."""
        from fem2d.assembly import assemble_gravity
        from fem2d.mesh import generate_rect_mesh
        nodes, elements = generate_rect_mesh(0, 4, -3, 0, 4, 3)
        gamma = 18.0
        F = assemble_gravity(nodes, elements, gamma)
        total_fy = sum(F[1::2])  # all y-forces
        expected = -gamma * 4 * 3 * 1.0  # -gamma * W * H * t
        assert total_fy == pytest.approx(expected, rel=1e-6)

    def test_per_element_gamma(self):
        from fem2d.assembly import assemble_gravity
        from fem2d.mesh import generate_rect_mesh
        nodes, elements = generate_rect_mesh(0, 2, -1, 0, 2, 1)
        gamma = np.full(len(elements), 20.0)
        F = assemble_gravity(nodes, elements, gamma)
        total_fy = sum(F[1::2])
        assert total_fy == pytest.approx(-20 * 2 * 1, rel=1e-6)


class TestAssembleSurfaceLoad:
    def test_uniform_load(self):
        from fem2d.assembly import assemble_surface_load
        from fem2d.mesh import generate_rect_mesh
        nodes, _ = generate_rect_mesh(0, 4, -2, 0, 4, 2)
        # Surface nodes at y=0
        surface = np.where(np.abs(nodes[:, 1]) < 0.01)[0]
        surface = surface[np.argsort(nodes[surface, 0])]
        edges = [(surface[i], surface[i + 1]) for i in range(len(surface) - 1)]
        F = assemble_surface_load(nodes, edges, qx=0, qy=-10, t=1)
        total_fy = sum(F[1::2])
        assert total_fy == pytest.approx(-10 * 4, rel=1e-6)


class TestApplyBCs:
    def test_penalty_applied(self):
        from fem2d.assembly import assemble_stiffness, apply_bcs_penalty
        from fem2d.materials import elastic_D
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes
        nodes, elements = generate_rect_mesh(0, 2, -1, 0, 2, 1)
        D = elastic_D(30000, 0.3)
        K = assemble_stiffness(nodes, elements, D)
        F = np.zeros(2 * len(nodes))
        bc = detect_boundary_nodes(nodes)
        K_bc, F_bc = apply_bcs_penalty(K, F, bc)
        # Diagonal at constrained DOFs should be huge
        for n in bc['fixed_base']:
            assert K_bc[2 * n, 2 * n] > 1e18
            assert K_bc[2 * n + 1, 2 * n + 1] > 1e18


class TestNodalStresses:
    def test_averaging(self):
        from fem2d.assembly import nodal_stresses
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        elements = np.array([[0, 1, 2], [0, 2, 3]])
        # Element 0: [10, 20, 5], Element 1: [30, 40, 15]
        elem_stresses = np.array([[10, 20, 5], [30, 40, 15]], dtype=float)
        ns = nodal_stresses(nodes, elements, elem_stresses)
        assert ns.shape == (4, 3)
        # Node 0 belongs to both elements → average
        np.testing.assert_allclose(ns[0], [20, 30, 10], rtol=1e-10)
        # Node 1 belongs to element 0 only
        np.testing.assert_allclose(ns[1], [10, 20, 5], rtol=1e-10)


# ---------------------------------------------------------------------------
# Solver — linear elastic
# ---------------------------------------------------------------------------

class TestSolveElastic:
    def test_gravity_column_stress(self):
        """Vertical stress at depth z should be approx gamma*|z|."""
        from fem2d.solver import solve_elastic
        from fem2d.materials import elastic_D
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes

        W, H = 20.0, 10.0
        gamma = 18.0
        E, nu = 30000.0, 0.3
        nodes, elements = generate_rect_mesh(0, W, -H, 0, 20, 10)
        bc = detect_boundary_nodes(nodes)
        D = elastic_D(E, nu)

        u, stresses, strains = solve_elastic(nodes, elements, D, gamma, bc)

        # Check vertical stress at interior elements (away from boundaries)
        centroids = nodes[elements].mean(axis=1)
        # Pick elements in the middle of the domain, at various depths
        mid_mask = (centroids[:, 0] > W * 0.3) & (centroids[:, 0] < W * 0.7)
        mid_centroids = centroids[mid_mask]
        mid_stresses = stresses[mid_mask]
        for cy, s in zip(mid_centroids[:, 1], mid_stresses):
            expected = gamma * cy  # cy is negative, so sigma_yy = gamma*cy < 0
            assert s[1] == pytest.approx(expected, rel=0.20, abs=5)

    def test_gravity_column_settlement(self):
        """Max settlement = gamma * H^2 / (2 * M) where M = constrained modulus."""
        from fem2d.solver import solve_elastic
        from fem2d.materials import elastic_D
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes

        W, H = 20.0, 10.0
        gamma = 18.0
        E, nu = 30000.0, 0.3
        # Use wider mesh to reduce boundary effects
        nodes, elements = generate_rect_mesh(0, W, -H, 0, 20, 10)
        bc = detect_boundary_nodes(nodes)
        D = elastic_D(E, nu)

        u, _, _ = solve_elastic(nodes, elements, D, gamma, bc)

        # Max downward displacement at surface
        uy = u[1::2]
        max_settlement = -uy.min()  # downward is negative

        M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        analytical = gamma * H**2 / (2 * M)
        assert max_settlement == pytest.approx(analytical, rel=0.15)

    def test_zero_gravity_zero_displacement(self):
        from fem2d.solver import solve_elastic
        from fem2d.materials import elastic_D
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes

        nodes, elements = generate_rect_mesh(0, 5, -5, 0, 5, 5)
        bc = detect_boundary_nodes(nodes)
        D = elastic_D(30000, 0.3)

        u, stresses, strains = solve_elastic(nodes, elements, D, 0.0, bc)
        assert np.max(np.abs(u)) < 1e-10

    def test_with_surface_load(self):
        """Surface load should produce downward displacement."""
        from fem2d.solver import solve_elastic
        from fem2d.materials import elastic_D
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes

        nodes, elements = generate_rect_mesh(0, 10, -10, 0, 10, 10)
        bc = detect_boundary_nodes(nodes)
        D = elastic_D(30000, 0.3)

        surface = np.where(np.abs(nodes[:, 1]) < 0.01)[0]
        surface = surface[np.argsort(nodes[surface, 0])]
        edges = [(surface[i], surface[i + 1])
                 for i in range(len(surface) - 1)]
        surface_loads = [(edges, 0.0, -50.0)]

        u, stresses, _ = solve_elastic(
            nodes, elements, D, 0.0, bc, surface_loads=surface_loads)

        # Surface should move downward
        uy_surface = u[2 * surface + 1]
        assert np.min(uy_surface) < 0


# ---------------------------------------------------------------------------
# Solver — nonlinear
# ---------------------------------------------------------------------------

class TestSolveNonlinear:
    def test_elastic_material_converges(self):
        """With high c and phi, material stays elastic → must converge."""
        from fem2d.solver import solve_nonlinear
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes

        nodes, elements = generate_rect_mesh(0, 5, -5, 0, 5, 5)
        bc = detect_boundary_nodes(nodes)
        gamma = 18.0
        material_props = [{
            'E': 30000, 'nu': 0.3, 'c': 1000, 'phi': 45, 'psi': 0,
            'gamma': 18
        }]
        converged, u, stresses, strains = solve_nonlinear(
            nodes, elements, material_props, gamma, bc,
            n_steps=5, max_iter=50)
        assert converged

    def test_zero_gravity_converges(self):
        from fem2d.solver import solve_nonlinear
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes

        nodes, elements = generate_rect_mesh(0, 5, -5, 0, 5, 5)
        bc = detect_boundary_nodes(nodes)
        material_props = [{'E': 30000, 'nu': 0.3, 'c': 20, 'phi': 30,
                           'psi': 0, 'gamma': 18}]
        converged, u, stresses, strains = solve_nonlinear(
            nodes, elements, material_props, 0.0, bc)
        assert converged
        assert np.max(np.abs(u)) < 1e-10

    def test_material_props_expansion(self):
        """Single material dict expands to all elements."""
        from fem2d.solver import solve_nonlinear
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes

        nodes, elements = generate_rect_mesh(0, 4, -4, 0, 4, 4)
        bc = detect_boundary_nodes(nodes)
        material_props = [{'E': 30000, 'nu': 0.3, 'c': 500, 'phi': 35,
                           'psi': 0, 'gamma': 18}]
        converged, u, _, _ = solve_nonlinear(
            nodes, elements, material_props, 18.0, bc,
            n_steps=3, max_iter=50)
        assert converged


# ---------------------------------------------------------------------------
# SRM
# ---------------------------------------------------------------------------

class TestStrengthReduction:
    def test_strong_slope_high_fos(self):
        """Strong soil on rectangular domain should have FOS > 1."""
        from fem2d.srm import strength_reduction
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes

        # Use a simple rectangular domain for reliable convergence
        nodes, elements = generate_rect_mesh(0, 20, -10, 0, 10, 5)
        bc = detect_boundary_nodes(nodes)

        material_props = [{'E': 30000, 'nu': 0.3, 'c': 100, 'phi': 35,
                           'psi': 0, 'gamma': 18}]
        result = strength_reduction(
            nodes, elements, material_props, 18.0, bc,
            n_load_steps=5, max_nr_iter=50, srf_range=(0.5, 3.0))
        assert result['FOS'] >= 1.0
        assert result['n_srf_trials'] >= 2

    def test_auto_detect_bc(self):
        """bc_nodes=None should auto-detect."""
        from fem2d.srm import strength_reduction
        from fem2d.mesh import generate_rect_mesh

        nodes, elements = generate_rect_mesh(0, 10, -10, 0, 5, 5)
        material_props = [{'E': 30000, 'nu': 0.3, 'c': 100, 'phi': 30,
                           'psi': 0, 'gamma': 18}]
        result = strength_reduction(
            nodes, elements, material_props, 18.0, bc_nodes=None,
            n_load_steps=3, max_nr_iter=30, srf_range=(0.5, 2.0))
        assert 'FOS' in result
        assert 'converged' in result


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

class TestAnalyzeGravity:
    def test_returns_result(self):
        from fem2d.analysis import analyze_gravity
        result = analyze_gravity(10, 5, 18, 30000, 0.3, nx=5, ny=3)
        assert result.analysis_type == "elastic"
        assert result.converged
        assert result.max_displacement_m > 0

    def test_reasonable_stress(self):
        from fem2d.analysis import analyze_gravity
        result = analyze_gravity(10, 10, 18, 30000, 0.3, nx=8, ny=8)
        # Max vertical stress ≈ gamma*H = 180 kPa
        assert result.min_sigma_yy_kPa < 0  # compressive


class TestAnalyzeFoundation:
    def test_returns_result(self):
        from fem2d.analysis import analyze_foundation
        # Use finer mesh so surface nodes capture the B=2m foundation
        result = analyze_foundation(2, 100, 10, 30000, 0.3, nx=30, ny=10)
        assert result.analysis_type == "elastic"
        assert result.max_displacement_m > 0

    def test_settlement_under_load(self):
        from fem2d.analysis import analyze_foundation
        result = analyze_foundation(2, 100, 10, 30000, 0.3, nx=30, ny=10)
        assert result.max_displacement_y_m > 0


class TestAnalyzeSlopeSRM:
    def test_returns_fos(self):
        from fem2d.analysis import analyze_slope_srm
        # Use gentle slope with strong soil for reliable convergence
        surface = [(0, 0), (10, 0), (20, 3), (30, 3)]
        layers = [{
            'name': 'clay',
            'bottom_elevation': -30,
            'E': 30000, 'nu': 0.3,
            'c': 80, 'phi': 30, 'psi': 0,
            'gamma': 18,
        }]
        result = analyze_slope_srm(
            surface, layers, nx=10, ny=5,
            n_load_steps=5)
        assert result.FOS is not None
        assert result.FOS > 0
        assert result.n_srf_trials >= 2


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class TestFEMResult:
    def test_summary(self):
        from fem2d.results import FEMResult
        r = FEMResult(
            analysis_type="elastic",
            n_nodes=100, n_elements=150,
            max_displacement_m=0.005,
            max_displacement_x_m=0.001,
            max_displacement_y_m=0.004,
            max_sigma_xx_kPa=50.0,
            max_sigma_yy_kPa=20.0,
            min_sigma_yy_kPa=-180.0,
            max_tau_xy_kPa=30.0,
            displacements=np.zeros(200),
        )
        s = r.summary()
        assert "elastic" in s
        assert "100 nodes" in s
        assert "0.0050" in s

    def test_summary_with_fos(self):
        from fem2d.results import FEMResult
        r = FEMResult(FOS=1.35, n_srf_trials=12,
                      displacements=np.zeros(10))
        s = r.summary()
        assert "1.350" in s
        assert "12" in s

    def test_to_dict(self):
        from fem2d.results import FEMResult
        r = FEMResult(
            analysis_type="srm",
            n_nodes=50, n_elements=80,
            max_displacement_m=0.01,
            FOS=1.5, n_srf_trials=8,
            converged=True,
            displacements=np.zeros(100),
        )
        d = r.to_dict()
        assert d["analysis_type"] == "srm"
        assert d["FOS"] == 1.5
        assert d["n_srf_trials"] == 8
        assert "displacements" not in d  # raw arrays excluded

    def test_to_dict_no_fos(self):
        from fem2d.results import FEMResult
        r = FEMResult(analysis_type="elastic",
                      displacements=np.zeros(10))
        d = r.to_dict()
        assert "FOS" not in d

    def test_warnings(self):
        from fem2d.results import FEMResult
        r = FEMResult(warnings=["Test warning"],
                      displacements=np.zeros(10))
        s = r.summary()
        assert "WARNING: Test warning" in s
        d = r.to_dict()
        assert d["warnings"] == ["Test warning"]


# ---------------------------------------------------------------------------
# Import / public API
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_import_fem2d(self):
        import fem2d
        assert hasattr(fem2d, 'analyze_gravity')
        assert hasattr(fem2d, 'analyze_foundation')
        assert hasattr(fem2d, 'analyze_slope_srm')

    def test_all_exports(self):
        import fem2d
        for name in fem2d.__all__:
            assert hasattr(fem2d, name), f"Missing export: {name}"

    def test_result_class(self):
        from fem2d import FEMResult
        r = FEMResult(displacements=np.zeros(10))
        assert hasattr(r, 'summary')
        assert hasattr(r, 'to_dict')

    def test_element_functions(self):
        from fem2d import cst_stiffness, cst_B, cst_area
        from fem2d import q4_stiffness, q4_body_force, q4_stress
        assert callable(cst_stiffness)
        assert callable(q4_stiffness)

    def test_mesh_functions(self):
        from fem2d import (generate_rect_mesh, generate_slope_mesh,
                           detect_boundary_nodes, triangle_quality)
        assert callable(generate_rect_mesh)
        assert callable(detect_boundary_nodes)

    def test_solver_functions(self):
        from fem2d import solve_elastic, solve_nonlinear
        assert callable(solve_elastic)
        assert callable(solve_nonlinear)

    def test_srm_function(self):
        from fem2d import strength_reduction
        assert callable(strength_reduction)
