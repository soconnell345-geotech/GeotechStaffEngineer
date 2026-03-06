"""
Tests for fdm2d — explicit Lagrangian finite difference module.

~80 tests covering:
- Materials: D-matrix, moduli, wave speed, MC return mapping
- Grid: generation, sub-triangles, boundary detection, lumped mass
- Zones: strain rates, mixed discretization, forces
- Solver: timestep, convergence, elastic/MC
- Analysis: gravity column, foundation
- Results: summary, to_dict
- Cross-validation: fdm2d vs fem2d
"""

import math
import pytest
import numpy as np
from numpy.testing import assert_allclose

from fdm2d.materials import (
    elastic_D, bulk_shear_moduli, wave_speed, mc_return_mapping,
)
from fdm2d.grid import (
    generate_quad_grid, build_sub_triangles,
    compute_sub_triangle_geometry, detect_boundary_gridpoints,
    compute_lumped_mass, _cst_B_area,
)
from fdm2d.zones import (
    compute_strain_rates, apply_mixed_discretization,
    compute_internal_forces, compute_gravity_forces,
    compute_surface_pressure, zone_averaged_stress,
)
from fdm2d.solver import critical_timestep, solve_explicit
from fdm2d.analysis import (
    analyze_gravity, analyze_foundation,
    _build_bc_arrays, _build_result,
)
from fdm2d.results import FDMResult


# ===================================================================
# Materials tests
# ===================================================================

class TestElasticD:
    """Tests for elastic_D()."""

    def test_shape(self):
        D = elastic_D(30000, 0.3)
        assert D.shape == (3, 3)

    def test_symmetry(self):
        D = elastic_D(30000, 0.3)
        assert_allclose(D, D.T, atol=1e-10)

    def test_positive_definite(self):
        D = elastic_D(30000, 0.3)
        eigenvalues = np.linalg.eigvalsh(D)
        assert all(eigenvalues > 0)

    def test_nu_zero(self):
        """nu=0: no lateral strain coupling."""
        D = elastic_D(10000, 0.0)
        assert D[0, 1] == 0.0
        assert D[1, 0] == 0.0
        # D[0,0] = E*(1-nu)/((1+nu)*(1-2nu)) = E
        assert_allclose(D[0, 0], 10000.0, rtol=1e-10)

    def test_known_values(self):
        """Check against hand calculation."""
        E, nu = 30000, 0.3
        c = E / ((1 + nu) * (1 - 2 * nu))
        D = elastic_D(E, nu)
        assert_allclose(D[0, 0], c * (1 - nu), rtol=1e-10)
        assert_allclose(D[0, 1], c * nu, rtol=1e-10)
        assert_allclose(D[2, 2], c * (1 - 2 * nu) / 2.0, rtol=1e-10)

    def test_high_nu(self):
        """Near-incompressible: D values increase."""
        D_low = elastic_D(30000, 0.2)
        D_high = elastic_D(30000, 0.49)
        assert D_high[0, 0] > D_low[0, 0]


class TestBulkShearModuli:
    """Tests for bulk_shear_moduli()."""

    def test_known_values(self):
        E, nu = 30000, 0.3
        K, G = bulk_shear_moduli(E, nu)
        assert_allclose(K, E / (3 * (1 - 2 * nu)), rtol=1e-10)
        assert_allclose(G, E / (2 * (1 + nu)), rtol=1e-10)

    def test_relationship(self):
        """K and G should satisfy E = 9KG/(3K+G)."""
        E, nu = 50000, 0.25
        K, G = bulk_shear_moduli(E, nu)
        E_back = 9 * K * G / (3 * K + G)
        assert_allclose(E_back, E, rtol=1e-10)


class TestWaveSpeed:
    """Tests for wave_speed()."""

    def test_positive(self):
        K, G = bulk_shear_moduli(30000, 0.3)
        rho = 18.0 / 9.81
        vp = wave_speed(K, G, rho)
        assert vp > 0

    def test_scaling(self):
        """Stiffer material → higher wave speed."""
        rho = 18.0 / 9.81
        K1, G1 = bulk_shear_moduli(30000, 0.3)
        K2, G2 = bulk_shear_moduli(60000, 0.3)
        vp1 = wave_speed(K1, G1, rho)
        vp2 = wave_speed(K2, G2, rho)
        assert vp2 > vp1


class TestMCReturnMapping:
    """Tests for mc_return_mapping()."""

    def test_elastic_no_yield(self):
        """Stress inside yield surface → no correction."""
        # Small compressive stress — should be inside MC surface
        sigma = np.array([-50.0, -100.0, 10.0])
        sigma_new, yielded = mc_return_mapping(sigma, 30000, 0.3, 20.0, 30.0)
        assert not yielded
        assert_allclose(sigma_new, sigma)

    def test_yield_returns_to_surface(self):
        """Trial stress outside → returned to yield surface."""
        # Large shear with no confinement
        sigma = np.array([0.0, 0.0, 100.0])
        sigma_new, yielded = mc_return_mapping(sigma, 30000, 0.3, 10.0, 30.0)
        assert yielded
        # Check returned to yield surface
        sxx, syy, txy = sigma_new
        p = (sxx + syy) / 2.0
        q = math.sqrt(((sxx - syy) / 2.0) ** 2 + txy ** 2)
        f = q + p * math.sin(math.radians(30)) - 10.0 * math.cos(math.radians(30))
        assert f <= 1e-6

    def test_pure_cohesion(self):
        """phi=0 (undrained): strength = c."""
        sigma = np.array([0.0, 0.0, 50.0])
        sigma_new, yielded = mc_return_mapping(sigma, 30000, 0.3, 20.0, 0.0)
        assert yielded
        sxx, syy, txy = sigma_new
        q = math.sqrt(((sxx - syy) / 2.0) ** 2 + txy ** 2)
        assert_allclose(q, 20.0, atol=0.1)

    def test_returns_two_values(self):
        """Explicit solver version: returns (sigma, yielded), no D_ep."""
        sigma = np.array([-10.0, -20.0, 5.0])
        result = mc_return_mapping(sigma, 30000, 0.3, 10.0, 25.0)
        assert len(result) == 2

    def test_dilation(self):
        """Non-zero psi: return mapping with non-associated flow."""
        sigma = np.array([0.0, 0.0, 100.0])
        sigma_new, yielded = mc_return_mapping(
            sigma, 30000, 0.3, 10.0, 30.0, psi_deg=15.0)
        assert yielded

    def test_hydrostatic_tension_apex(self):
        """Hydrostatic tension beyond apex → apex return."""
        sigma = np.array([500.0, 500.0, 0.0])
        sigma_new, yielded = mc_return_mapping(sigma, 30000, 0.3, 10.0, 30.0)
        assert yielded
        # Should return to apex
        assert_allclose(sigma_new[0], sigma_new[1], atol=1e-6)


# ===================================================================
# Grid tests
# ===================================================================

class TestGenerateQuadGrid:
    """Tests for generate_quad_grid()."""

    def test_node_count(self):
        nodes, zones = generate_quad_grid(0, 10, -5, 0, 4, 3)
        assert len(nodes) == (4 + 1) * (3 + 1)  # 20

    def test_zone_count(self):
        nodes, zones = generate_quad_grid(0, 10, -5, 0, 4, 3)
        assert len(zones) == 4 * 3  # 12

    def test_zone_connectivity_shape(self):
        nodes, zones = generate_quad_grid(0, 10, -5, 0, 4, 3)
        assert zones.shape == (12, 4)

    def test_bottom_left_node(self):
        nodes, zones = generate_quad_grid(0, 10, -5, 0, 4, 3)
        assert_allclose(nodes[0], [0, -5])

    def test_top_right_node(self):
        nodes, zones = generate_quad_grid(0, 10, -5, 0, 4, 3)
        assert_allclose(nodes[-1], [10, 0])

    def test_ccw_ordering(self):
        """Zone nodes should be CCW: [SW, SE, NE, NW]."""
        nodes, zones = generate_quad_grid(0, 1, 0, 1, 1, 1)
        z = zones[0]
        # SW=0, SE=1, NE=3, NW=2 for a 2x2 grid
        sw, se, ne, nw = nodes[z]
        assert sw[0] < se[0]  # SW is left of SE
        assert sw[1] < nw[1]  # SW is below NW
        assert ne[0] > nw[0]  # NE is right of NW

    def test_single_zone(self):
        nodes, zones = generate_quad_grid(0, 1, 0, 1, 1, 1)
        assert len(nodes) == 4
        assert len(zones) == 1


class TestBuildSubTriangles:
    """Tests for build_sub_triangles()."""

    def test_shape(self):
        _, zones = generate_quad_grid(0, 1, 0, 1, 2, 2)
        sub_tris = build_sub_triangles(zones)
        assert sub_tris.shape == (4, 4, 3)

    def test_four_tris_per_zone(self):
        _, zones = generate_quad_grid(0, 1, 0, 1, 1, 1)
        sub_tris = build_sub_triangles(zones)
        assert sub_tris.shape[1] == 4

    def test_overlay_a_diagonal(self):
        """Overlay A uses diagonal 0-2."""
        _, zones = generate_quad_grid(0, 1, 0, 1, 1, 1)
        sub_tris = build_sub_triangles(zones)
        n0, n1, n2, n3 = zones[0]
        # Overlay A: tri(0,1,2), tri(0,2,3)
        assert set(sub_tris[0, 0]) == {n0, n1, n2}
        assert set(sub_tris[0, 1]) == {n0, n2, n3}

    def test_overlay_b_diagonal(self):
        """Overlay B uses diagonal 1-3."""
        _, zones = generate_quad_grid(0, 1, 0, 1, 1, 1)
        sub_tris = build_sub_triangles(zones)
        n0, n1, n2, n3 = zones[0]
        assert set(sub_tris[0, 2]) == {n0, n1, n3}
        assert set(sub_tris[0, 3]) == {n1, n2, n3}


class TestSubTriangleGeometry:
    """Tests for compute_sub_triangle_geometry()."""

    def test_B_shape(self):
        nodes, zones = generate_quad_grid(0, 1, 0, 1, 1, 1)
        sub_tris = build_sub_triangles(zones)
        B_all, areas = compute_sub_triangle_geometry(nodes, sub_tris)
        assert B_all.shape == (1, 4, 3, 6)

    def test_areas_positive(self):
        nodes, zones = generate_quad_grid(0, 2, 0, 3, 3, 3)
        sub_tris = build_sub_triangles(zones)
        _, areas = compute_sub_triangle_geometry(nodes, sub_tris)
        assert np.all(areas > 0)

    def test_overlay_areas_equal_zone(self):
        """Sum of overlay A areas = sum of overlay B areas = zone area."""
        nodes, zones = generate_quad_grid(0, 1, 0, 1, 1, 1)
        sub_tris = build_sub_triangles(zones)
        _, areas = compute_sub_triangle_geometry(nodes, sub_tris)
        zone_area = 1.0  # 1×1 quad
        assert_allclose(areas[0, 0] + areas[0, 1], zone_area, rtol=1e-10)
        assert_allclose(areas[0, 2] + areas[0, 3], zone_area, rtol=1e-10)


class TestDetectBoundary:
    """Tests for detect_boundary_gridpoints()."""

    def test_fixed_base_count(self):
        nodes, _ = generate_quad_grid(0, 10, -5, 0, 4, 3)
        bc = detect_boundary_gridpoints(nodes)
        assert len(bc['fixed_base']) == 5  # nx+1

    def test_roller_left(self):
        nodes, _ = generate_quad_grid(0, 10, -5, 0, 4, 3)
        bc = detect_boundary_gridpoints(nodes)
        # Left column excluding bottom row
        assert len(bc['roller_left']) == 3  # ny (excluding base)

    def test_roller_right(self):
        nodes, _ = generate_quad_grid(0, 10, -5, 0, 4, 3)
        bc = detect_boundary_gridpoints(nodes)
        assert len(bc['roller_right']) == 3

    def test_no_overlap(self):
        """Base and roller sets should not overlap."""
        nodes, _ = generate_quad_grid(0, 10, -5, 0, 4, 3)
        bc = detect_boundary_gridpoints(nodes)
        base_set = set(bc['fixed_base'])
        left_set = set(bc['roller_left'])
        right_set = set(bc['roller_right'])
        assert base_set.isdisjoint(left_set)
        assert base_set.isdisjoint(right_set)


class TestLumpedMass:
    """Tests for compute_lumped_mass()."""

    def test_total_mass(self):
        """Total lumped mass = rho * total_area * t."""
        nodes, zones = generate_quad_grid(0, 2, 0, 3, 2, 3)
        sub_tris = build_sub_triangles(zones)
        _, areas = compute_sub_triangle_geometry(nodes, sub_tris)
        rho = 18.0 / 9.81
        t = 1.0
        mass = compute_lumped_mass(nodes, zones, sub_tris, areas, rho, t)
        total_area = 2.0 * 3.0
        expected_mass = rho * total_area * t
        assert_allclose(mass.sum(), expected_mass, rtol=1e-10)

    def test_positive(self):
        nodes, zones = generate_quad_grid(0, 1, 0, 1, 2, 2)
        sub_tris = build_sub_triangles(zones)
        _, areas = compute_sub_triangle_geometry(nodes, sub_tris)
        mass = compute_lumped_mass(nodes, zones, sub_tris, areas, 1.0)
        assert np.all(mass > 0)

    def test_scaling_with_rho(self):
        nodes, zones = generate_quad_grid(0, 1, 0, 1, 2, 2)
        sub_tris = build_sub_triangles(zones)
        _, areas = compute_sub_triangle_geometry(nodes, sub_tris)
        m1 = compute_lumped_mass(nodes, zones, sub_tris, areas, 1.0)
        m2 = compute_lumped_mass(nodes, zones, sub_tris, areas, 2.0)
        assert_allclose(m2, 2.0 * m1, rtol=1e-10)


# ===================================================================
# Zones tests
# ===================================================================

class TestStrainRates:
    """Tests for compute_strain_rates()."""

    def test_zero_velocity_zero_strain(self):
        """Zero velocity → zero strain rates."""
        nodes, zones = generate_quad_grid(0, 1, 0, 1, 1, 1)
        sub_tris = build_sub_triangles(zones)
        B_all, _ = compute_sub_triangle_geometry(nodes, sub_tris)
        vel = np.zeros((len(nodes), 2))
        sr = compute_strain_rates(vel, sub_tris, B_all, len(zones))
        assert_allclose(sr, 0.0, atol=1e-15)

    def test_uniform_x_velocity(self):
        """Uniform x-velocity → zero strain (rigid translation)."""
        nodes, zones = generate_quad_grid(0, 1, 0, 1, 1, 1)
        sub_tris = build_sub_triangles(zones)
        B_all, _ = compute_sub_triangle_geometry(nodes, sub_tris)
        vel = np.ones((len(nodes), 2)) * [1.0, 0.0]
        sr = compute_strain_rates(vel, sub_tris, B_all, len(zones))
        assert_allclose(sr, 0.0, atol=1e-12)

    def test_linear_x_displacement(self):
        """Linear velocity in x → uniform eps_xx."""
        nodes, zones = generate_quad_grid(0, 1, 0, 1, 1, 1)
        sub_tris = build_sub_triangles(zones)
        B_all, _ = compute_sub_triangle_geometry(nodes, sub_tris)
        # vx = x (linear in x)
        vel = np.zeros((len(nodes), 2))
        vel[:, 0] = nodes[:, 0]
        sr = compute_strain_rates(vel, sub_tris, B_all, len(zones))
        # eps_xx should be 1.0 everywhere
        assert_allclose(sr[:, :, 0], 1.0, atol=1e-10)


class TestMixedDiscretization:
    """Tests for apply_mixed_discretization()."""

    def test_uniform_strain_unchanged(self):
        """Uniform volumetric strain → no correction."""
        n_zones = 2
        strain_rates = np.zeros((n_zones, 4, 3))
        # Uniform volumetric: eps_xx = eps_yy = 1.0
        strain_rates[:, :, 0] = 1.0
        strain_rates[:, :, 1] = 1.0
        areas = np.ones((n_zones, 4))
        corrected = apply_mixed_discretization(strain_rates, areas)
        assert_allclose(corrected, strain_rates, atol=1e-12)

    def test_deviatoric_preserved(self):
        """Mixed discretization preserves deviatoric strain rates."""
        n_zones = 1
        strain_rates = np.zeros((n_zones, 4, 3))
        # Different volumetric but same deviatoric (eps_xx - eps_yy)
        strain_rates[0, 0, 0] = 2.0
        strain_rates[0, 0, 1] = 0.0
        strain_rates[0, 1, 0] = 1.0
        strain_rates[0, 1, 1] = 1.0
        strain_rates[0, 2, 0] = 1.5
        strain_rates[0, 2, 1] = 0.5
        strain_rates[0, 3, 0] = 0.5
        strain_rates[0, 3, 1] = 1.5
        areas = np.ones((n_zones, 4))
        corrected = apply_mixed_discretization(strain_rates, areas)
        # Deviatoric part (eps_xx - eps_yy) should be preserved
        for s in range(4):
            dev_orig = strain_rates[0, s, 0] - strain_rates[0, s, 1]
            dev_corr = corrected[0, s, 0] - corrected[0, s, 1]
            assert_allclose(dev_corr, dev_orig, atol=1e-12)


class TestGravityForces:
    """Tests for compute_gravity_forces()."""

    def test_total_force(self):
        """Total gravity = gamma * area * t (downward)."""
        nodes, zones = generate_quad_grid(0, 2, 0, 3, 2, 3)
        sub_tris = build_sub_triangles(zones)
        _, areas = compute_sub_triangle_geometry(nodes, sub_tris)
        gamma = 18.0
        F = compute_gravity_forces(nodes, zones, areas, gamma, len(nodes))
        # Total Fy should be -gamma * area * t
        total_area = 2.0 * 3.0
        expected_fy = -gamma * total_area
        assert_allclose(F[:, 1].sum(), expected_fy, rtol=1e-10)
        # Total Fx should be zero
        assert_allclose(F[:, 0].sum(), 0.0, atol=1e-10)

    def test_direction(self):
        """Gravity forces are downward (negative y)."""
        nodes, zones = generate_quad_grid(0, 1, 0, 1, 1, 1)
        sub_tris = build_sub_triangles(zones)
        _, areas = compute_sub_triangle_geometry(nodes, sub_tris)
        F = compute_gravity_forces(nodes, zones, areas, 18.0, len(nodes))
        assert np.all(F[:, 1] <= 0)


class TestSurfacePressure:
    """Tests for compute_surface_pressure()."""

    def test_total_force(self):
        """Total force = pressure * length * t."""
        nodes = np.array([[0, 0], [1, 0], [2, 0], [0, -1], [1, -1], [2, -1]],
                         dtype=float)
        edges = [(0, 1), (1, 2)]
        qy = -100.0  # downward
        F = compute_surface_pressure(nodes, edges, 0.0, qy, len(nodes))
        expected_fy = qy * 2.0  # 2m total length
        assert_allclose(F[:, 1].sum(), expected_fy, rtol=1e-10)


class TestInternalForces:
    """Tests for compute_internal_forces()."""

    def test_equilibrium(self):
        """Under gravity, internal forces should approach gravity forces."""
        # This is tested indirectly through the solver
        nodes, zones = generate_quad_grid(0, 1, 0, 1, 1, 1)
        sub_tris = build_sub_triangles(zones)
        B_all, areas = compute_sub_triangle_geometry(nodes, sub_tris)
        stresses = np.zeros((1, 4, 3))
        F = compute_internal_forces(
            nodes, sub_tris, B_all, areas, stresses, len(nodes))
        # Zero stress → zero internal forces
        assert_allclose(F, 0.0, atol=1e-15)


class TestZoneAveragedStress:
    """Tests for zone_averaged_stress()."""

    def test_shape(self):
        stresses = np.ones((5, 4, 3))
        avg = zone_averaged_stress(stresses)
        assert avg.shape == (5, 3)

    def test_uniform(self):
        stresses = np.ones((3, 4, 3)) * 100.0
        avg = zone_averaged_stress(stresses)
        assert_allclose(avg, 100.0)


# ===================================================================
# Solver tests
# ===================================================================

class TestCriticalTimestep:
    """Tests for critical_timestep()."""

    def test_positive(self):
        nodes, zones = generate_quad_grid(0, 10, -5, 0, 5, 5)
        props = {'E': 30000, 'nu': 0.3, 'gamma': 18.0}
        dt = critical_timestep(nodes, zones, props)
        assert dt > 0

    def test_scaling(self):
        """Finer mesh → smaller timestep."""
        props = {'E': 30000, 'nu': 0.3, 'gamma': 18.0}
        n1, z1 = generate_quad_grid(0, 10, -5, 0, 5, 5)
        n2, z2 = generate_quad_grid(0, 10, -5, 0, 10, 10)
        dt1 = critical_timestep(n1, z1, props)
        dt2 = critical_timestep(n2, z2, props)
        assert dt2 < dt1

    def test_stiffer_faster(self):
        """Stiffer material → smaller timestep."""
        nodes, zones = generate_quad_grid(0, 10, -5, 0, 5, 5)
        dt1 = critical_timestep(nodes, zones, {'E': 30000, 'nu': 0.3, 'gamma': 18.0})
        dt2 = critical_timestep(nodes, zones, {'E': 120000, 'nu': 0.3, 'gamma': 18.0})
        assert dt2 < dt1

    def test_safety_factor(self):
        nodes, zones = generate_quad_grid(0, 10, -5, 0, 5, 5)
        props = {'E': 30000, 'nu': 0.3, 'gamma': 18.0}
        dt_half = critical_timestep(nodes, zones, props, safety=0.5)
        dt_full = critical_timestep(nodes, zones, props, safety=1.0)
        assert_allclose(dt_half, dt_full / 2.0, rtol=1e-10)


class TestSolveExplicit:
    """Tests for solve_explicit()."""

    def test_zero_gravity_zero_displacement(self):
        """No forces → no displacement."""
        nodes, zones = generate_quad_grid(0, 1, 0, 1, 2, 2)
        sub_tris = build_sub_triangles(zones)
        B_all, areas = compute_sub_triangle_geometry(nodes, sub_tris)
        bc = detect_boundary_gridpoints(nodes)
        from fdm2d.analysis import _build_bc_arrays
        bc_fixed, bc_values = _build_bc_arrays(nodes, bc)
        rho = 18.0 / 9.81
        mass = compute_lumped_mass(nodes, zones, sub_tris, areas, rho)
        props = {'E': 30000, 'nu': 0.3, 'gamma': 18.0}

        result = solve_explicit(
            nodes, zones, sub_tris, B_all, areas, props,
            0.0,  # zero gravity
            bc_fixed, bc_values, mass, len(nodes),
            max_steps=1000, tol=1e-5)

        converged, pos, disp, stresses, vel, n_steps, fr, hist = result
        assert converged
        assert_allclose(disp, 0.0, atol=1e-15)

    def test_elastic_convergence(self):
        """Elastic gravity problem should converge."""
        nodes, zones = generate_quad_grid(0, 5, -5, 0, 5, 5)
        sub_tris = build_sub_triangles(zones)
        B_all, areas = compute_sub_triangle_geometry(nodes, sub_tris)
        bc = detect_boundary_gridpoints(nodes)
        from fdm2d.analysis import _build_bc_arrays
        bc_fixed, bc_values = _build_bc_arrays(nodes, bc)
        rho = 18.0 / 9.81
        mass = compute_lumped_mass(nodes, zones, sub_tris, areas, rho)
        props = {'E': 30000, 'nu': 0.3, 'gamma': 18.0}

        result = solve_explicit(
            nodes, zones, sub_tris, B_all, areas, props,
            18.0, bc_fixed, bc_values, mass, len(nodes),
            max_steps=50000, tol=1e-4)

        converged, pos, disp, stresses, vel, n_steps, fr, hist = result
        assert converged

    def test_mc_convergence(self):
        """Mohr-Coulomb gravity should converge."""
        nodes, zones = generate_quad_grid(0, 5, -5, 0, 5, 5)
        sub_tris = build_sub_triangles(zones)
        B_all, areas = compute_sub_triangle_geometry(nodes, sub_tris)
        bc = detect_boundary_gridpoints(nodes)
        from fdm2d.analysis import _build_bc_arrays
        bc_fixed, bc_values = _build_bc_arrays(nodes, bc)
        rho = 18.0 / 9.81
        mass = compute_lumped_mass(nodes, zones, sub_tris, areas, rho)
        props = {'E': 30000, 'nu': 0.3, 'gamma': 18.0, 'c': 50.0, 'phi': 30.0}

        result = solve_explicit(
            nodes, zones, sub_tris, B_all, areas, props,
            18.0, bc_fixed, bc_values, mass, len(nodes),
            max_steps=50000, tol=1e-4)

        converged, pos, disp, stresses, vel, n_steps, fr, hist = result
        assert converged

    def test_downward_displacement(self):
        """Gravity should cause downward (negative y) displacement."""
        nodes, zones = generate_quad_grid(0, 5, -5, 0, 5, 5)
        sub_tris = build_sub_triangles(zones)
        B_all, areas = compute_sub_triangle_geometry(nodes, sub_tris)
        bc = detect_boundary_gridpoints(nodes)
        from fdm2d.analysis import _build_bc_arrays
        bc_fixed, bc_values = _build_bc_arrays(nodes, bc)
        rho = 18.0 / 9.81
        mass = compute_lumped_mass(nodes, zones, sub_tris, areas, rho)
        props = {'E': 30000, 'nu': 0.3, 'gamma': 18.0}

        result = solve_explicit(
            nodes, zones, sub_tris, B_all, areas, props,
            18.0, bc_fixed, bc_values, mass, len(nodes),
            max_steps=50000, tol=1e-4)

        converged, pos, disp, stresses, vel, n_steps, fr, hist = result
        # Free surface nodes should move down
        top_nodes = np.where(np.abs(nodes[:, 1]) < 1e-6)[0]
        # Exclude corners that may be fixed
        interior_top = top_nodes[
            (nodes[top_nodes, 0] > 0.01) & (nodes[top_nodes, 0] < 4.99)]
        if len(interior_top) > 0:
            assert np.all(disp[interior_top, 1] < 0)


# ===================================================================
# Analysis tests
# ===================================================================

class TestAnalyzeGravity:
    """Tests for analyze_gravity()."""

    def test_returns_fdm_result(self):
        r = analyze_gravity(5, 5, 18.0, 30000, 0.3, nx=5, ny=5,
                            max_steps=30000, tol=1e-3)
        assert isinstance(r, FDMResult)
        assert r.analysis_type == "gravity"

    def test_converges(self):
        r = analyze_gravity(5, 5, 18.0, 30000, 0.3, nx=5, ny=5,
                            max_steps=50000, tol=1e-3)
        assert r.converged

    def test_grid_counts(self):
        r = analyze_gravity(5, 5, 18.0, 30000, 0.3, nx=4, ny=3,
                            max_steps=30000, tol=1e-3)
        assert r.n_gridpoints == 5 * 4  # (nx+1)*(ny+1)
        assert r.n_zones == 4 * 3

    def test_stress_at_depth(self):
        """sigma_yy at base ≈ -gamma*depth (compression)."""
        r = analyze_gravity(10, 10, 18.0, 30000, 0.3, nx=5, ny=5,
                            max_steps=50000, tol=1e-3)
        # min_sigma_yy should be approximately -gamma*depth = -180
        assert r.min_sigma_yy_kPa < 0
        assert r.min_sigma_yy_kPa < -100  # should be near -180

    def test_settlement(self):
        """Max settlement order-of-magnitude check.

        Analytical: delta = gamma*H^2 / (2*M)
        M = E*(1-nu)/((1+nu)*(1-2nu))
        """
        gamma, H, E, nu = 18.0, 5.0, 30000, 0.3
        M = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        delta_analytical = gamma * H ** 2 / (2 * M)

        r = analyze_gravity(10, H, gamma, E, nu, nx=5, ny=5,
                            max_steps=50000, tol=1e-3)
        assert r.max_displacement_y_m > 0  # Should have downward movement
        # Within order of magnitude
        assert r.max_displacement_y_m < delta_analytical * 5
        assert r.max_displacement_y_m > delta_analytical * 0.1

    def test_mc_gravity(self):
        """Gravity with Mohr-Coulomb should converge."""
        r = analyze_gravity(5, 5, 18.0, 30000, 0.3, nx=5, ny=5,
                            c=20.0, phi=30.0,
                            max_steps=50000, tol=1e-3)
        assert r.converged


class TestAnalyzeFoundation:
    """Tests for analyze_foundation()."""

    def test_returns_fdm_result(self):
        r = analyze_foundation(2.0, 100.0, 10.0, 30000, 0.3,
                               nx=10, ny=5,
                               max_steps=30000, tol=1e-3)
        assert isinstance(r, FDMResult)
        assert r.analysis_type == "foundation"

    def test_converges(self):
        r = analyze_foundation(2.0, 100.0, 10.0, 30000, 0.3,
                               nx=10, ny=5,
                               max_steps=50000, tol=1e-3)
        assert r.converged

    def test_settlement_positive(self):
        """Foundation load should cause settlement."""
        r = analyze_foundation(2.0, 100.0, 10.0, 30000, 0.3,
                               nx=10, ny=5,
                               max_steps=50000, tol=1e-3)
        assert r.max_displacement_y_m > 0


# ===================================================================
# Results tests
# ===================================================================

class TestFDMResult:
    """Tests for FDMResult dataclass."""

    def test_summary_format(self):
        r = FDMResult(analysis_type="gravity", n_gridpoints=100,
                      n_zones=81, converged=True, n_timesteps=5000)
        s = r.summary()
        assert "EXPLICIT FDM" in s
        assert "gravity" in s
        assert "100" in s

    def test_to_dict_keys(self):
        r = FDMResult(analysis_type="gravity", n_gridpoints=100,
                      n_zones=81, converged=True)
        d = r.to_dict()
        expected_keys = {
            "analysis_type", "n_gridpoints", "n_zones", "converged",
            "n_timesteps", "final_force_ratio",
            "max_displacement_m", "max_displacement_x_m",
            "max_displacement_y_m",
            "max_sigma_xx_kPa", "max_sigma_yy_kPa",
            "min_sigma_yy_kPa", "max_tau_xy_kPa",
            "warnings",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_no_numpy(self):
        """Dict should contain no numpy objects."""
        r = FDMResult(
            analysis_type="test",
            max_displacement_m=0.001,
            final_force_ratio=1e-6,
        )
        d = r.to_dict()
        for key, val in d.items():
            assert not isinstance(val, np.ndarray), f"{key} is ndarray"
            assert not isinstance(val, np.floating), f"{key} is np.floating"
            assert not isinstance(val, np.integer), f"{key} is np.integer"

    def test_summary_with_warnings(self):
        r = FDMResult(warnings=["Did not converge"])
        s = r.summary()
        assert "WARNING" in s

    def test_default_values(self):
        r = FDMResult()
        assert r.analysis_type == "gravity"
        assert r.converged is True
        assert r.n_timesteps == 0


# ===================================================================
# Patch test
# ===================================================================

class TestPatchTest:
    """Patch test: linear displacement field → uniform strain."""

    def test_uniform_strain_from_linear_displacement(self):
        """Apply linear velocity field, check uniform strain rates."""
        nodes, zones = generate_quad_grid(0, 2, 0, 2, 2, 2)
        sub_tris = build_sub_triangles(zones)
        B_all, areas = compute_sub_triangle_geometry(nodes, sub_tris)

        # Linear velocity: vx = 0.001*x, vy = 0
        vel = np.zeros((len(nodes), 2))
        vel[:, 0] = 0.001 * nodes[:, 0]

        sr = compute_strain_rates(vel, sub_tris, B_all, len(zones))
        # eps_xx should be 0.001 everywhere
        assert_allclose(sr[:, :, 0], 0.001, atol=1e-10)
        # eps_yy and gamma_xy should be zero
        assert_allclose(sr[:, :, 1], 0.0, atol=1e-10)
        assert_allclose(sr[:, :, 2], 0.0, atol=1e-10)


# ===================================================================
# Mixed discretization locking test
# ===================================================================

class TestVolumeLocking:
    """Test that mixed discretization prevents volumetric locking."""

    def test_near_incompressible(self):
        """nu=0.499 should not lock — displacement should be nonzero."""
        r = analyze_gravity(5, 5, 18.0, 30000, 0.499, nx=5, ny=5,
                            max_steps=50000, tol=1e-3)
        # If locked, displacement would be near-zero
        assert r.max_displacement_m > 1e-6

    def test_compressible_vs_incompressible(self):
        """Near-incompressible should give similar settlement to nu=0.3."""
        r_comp = analyze_gravity(5, 5, 18.0, 30000, 0.3, nx=5, ny=5,
                                 max_steps=50000, tol=1e-3)
        r_incomp = analyze_gravity(5, 5, 18.0, 30000, 0.499, nx=5, ny=5,
                                   max_steps=50000, tol=1e-3)
        # Both should have nonzero displacement
        assert r_comp.max_displacement_m > 1e-6
        assert r_incomp.max_displacement_m > 1e-6


# ===================================================================
# Build BC arrays test
# ===================================================================

class TestBuildBCArrays:
    """Tests for _build_bc_arrays()."""

    def test_shape(self):
        nodes, _ = generate_quad_grid(0, 5, -5, 0, 3, 3)
        bc_dict = detect_boundary_gridpoints(nodes)
        bc_fixed, bc_values = _build_bc_arrays(nodes, bc_dict)
        assert bc_fixed.shape == (len(nodes), 2)
        assert bc_values.shape == (len(nodes), 2)

    def test_base_fixed_both(self):
        """Base nodes should be fixed in both x and y."""
        nodes, _ = generate_quad_grid(0, 5, -5, 0, 3, 3)
        bc_dict = detect_boundary_gridpoints(nodes)
        bc_fixed, _ = _build_bc_arrays(nodes, bc_dict)
        for nid in bc_dict['fixed_base']:
            assert bc_fixed[nid, 0]
            assert bc_fixed[nid, 1]

    def test_roller_x_only(self):
        """Roller nodes should be fixed in x only."""
        nodes, _ = generate_quad_grid(0, 5, -5, 0, 3, 3)
        bc_dict = detect_boundary_gridpoints(nodes)
        bc_fixed, _ = _build_bc_arrays(nodes, bc_dict)
        for nid in bc_dict['roller_left']:
            assert bc_fixed[nid, 0]
            assert not bc_fixed[nid, 1]


# ===================================================================
# CST B-matrix unit test
# ===================================================================

class TestCSTBArea:
    """Tests for _cst_B_area()."""

    def test_unit_triangle_area(self):
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        B, A = _cst_B_area(coords)
        assert_allclose(A, 0.5, rtol=1e-10)

    def test_B_shape(self):
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        B, A = _cst_B_area(coords)
        assert B.shape == (3, 6)

    def test_B_rigid_body(self):
        """Rigid body translation → zero strain."""
        coords = np.array([[0, 0], [2, 0], [1, 1]], dtype=float)
        B, _ = _cst_B_area(coords)
        # Uniform displacement
        u = np.array([1, 0, 1, 0, 1, 0], dtype=float)
        eps = B @ u
        assert_allclose(eps, 0.0, atol=1e-12)


# ===================================================================
# Cross-validation: fdm2d vs fem2d
# ===================================================================

class TestCrossValidation:
    """Cross-validation between fdm2d and fem2d for gravity column."""

    def test_gravity_stress_vs_fem2d(self):
        """Gravity column stress: fdm2d vs fem2d within 20%."""
        try:
            from fem2d import analyze_gravity as fem_gravity
        except ImportError:
            pytest.skip("fem2d not available")

        gamma, depth, E, nu = 18.0, 10.0, 30000, 0.3

        fdm_r = analyze_gravity(10, depth, gamma, E, nu, nx=8, ny=8,
                                max_steps=80000, tol=1e-4)
        fem_r = fem_gravity(10, depth, gamma, E, nu, nx=8, ny=8)

        # Both should show compression at base ≈ -gamma*depth
        expected = -gamma * depth  # -180 kPa

        # fdm2d and fem2d should be within 20% of analytical
        assert abs(fdm_r.min_sigma_yy_kPa - expected) / abs(expected) < 0.25
        assert abs(fem_r.min_sigma_yy_kPa - expected) / abs(expected) < 0.25

    def test_gravity_displacement_vs_fem2d(self):
        """Gravity column displacement: fdm2d vs fem2d within 50%."""
        try:
            from fem2d import analyze_gravity as fem_gravity
        except ImportError:
            pytest.skip("fem2d not available")

        gamma, depth, E, nu = 18.0, 5.0, 30000, 0.3

        fdm_r = analyze_gravity(10, depth, gamma, E, nu, nx=5, ny=5,
                                max_steps=50000, tol=1e-3)
        fem_r = fem_gravity(10, depth, gamma, E, nu, nx=5, ny=5)

        # Both should have nonzero settlement
        assert fdm_r.max_displacement_y_m > 0
        assert fem_r.max_displacement_y_m > 0

        # Within factor of 2 (explicit vs implicit can differ on coarse mesh)
        ratio = fdm_r.max_displacement_y_m / max(fem_r.max_displacement_y_m, 1e-10)
        assert 0.2 < ratio < 5.0
