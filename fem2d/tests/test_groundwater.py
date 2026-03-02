"""
Tests for groundwater pressures, seepage, and Biot consolidation.

Covers porewater.py functions, solver integration with pore pressures,
seepage solver, and coupled consolidation.

Analytical validations:
- Hydrostatic: u = gamma_w * depth_below_gwt
- 1D Darcy flow: v = k * dh/dx (constant for uniform head gradient)
- Terzaghi consolidation: settlement increases, pp decreases over time
"""

import math
import numpy as np
import pytest


# ===========================================================================
# Phase 1: Pore Pressure Tests
# ===========================================================================

class TestComputePorePressures:
    """Test compute_pore_pressures() with various GWT inputs."""

    def test_constant_gwt(self):
        from fem2d.porewater import compute_pore_pressures
        nodes = np.array([
            [0.0, 0.0],    # at GWT level
            [1.0, -5.0],   # 5m below GWT
            [2.0, 2.0],    # 2m above GWT
        ])
        pp = compute_pore_pressures(nodes, gwt=0.0, gamma_w=9.81)
        assert pp[0] == pytest.approx(0.0)
        assert pp[1] == pytest.approx(9.81 * 5.0, rel=1e-10)
        assert pp[2] == pytest.approx(0.0)  # above GWT

    def test_deep_point_pressure(self):
        from fem2d.porewater import compute_pore_pressures
        nodes = np.array([[5.0, -10.0]])
        pp = compute_pore_pressures(nodes, gwt=0.0, gamma_w=9.81)
        assert pp[0] == pytest.approx(9.81 * 10.0, rel=1e-10)

    def test_above_gwt_zero(self):
        from fem2d.porewater import compute_pore_pressures
        nodes = np.array([[0.0, 5.0], [1.0, 3.0], [2.0, 1.0]])
        pp = compute_pore_pressures(nodes, gwt=0.0)
        np.testing.assert_array_equal(pp, 0.0)

    def test_polyline_gwt(self):
        from fem2d.porewater import compute_pore_pressures
        nodes = np.array([
            [0.0, -2.0],   # 2m below GWT (gwt=0 at x=0)
            [5.0, -2.0],   # at GWT level (gwt=-2 at x=5)
            [10.0, -6.0],  # 2m below GWT (gwt=-4 at x=10)
        ])
        # GWT slopes from z=0 at x=0 to z=-4 at x=10
        gwt_poly = np.array([[0.0, 0.0], [10.0, -4.0]])
        pp = compute_pore_pressures(nodes, gwt=gwt_poly, gamma_w=10.0)
        # At x=0: gwt=0, node at -2, depth=2m, pp=20
        assert pp[0] == pytest.approx(20.0)
        # At x=5: gwt=-2, node at -2, depth=0m, pp=0
        assert pp[1] == pytest.approx(0.0)
        # At x=10: gwt=-4, node at -6, depth=2m, pp=20
        assert pp[2] == pytest.approx(20.0)

    def test_per_node_artesian(self):
        from fem2d.porewater import compute_pore_pressures
        nodes = np.array([
            [0.0, -5.0],
            [1.0, -5.0],
        ])
        # Artesian: head at node 0 is 5m above, node 1 is at -5m
        heads = np.array([5.0, -5.0])
        pp = compute_pore_pressures(nodes, gwt=heads, gamma_w=9.81)
        # Node 0: head=5, z=-5, depth_below=10m
        assert pp[0] == pytest.approx(9.81 * 10.0, rel=1e-10)
        # Node 1: head=-5, z=-5, depth_below=0
        assert pp[1] == pytest.approx(0.0)

    def test_invalid_gwt_raises(self):
        from fem2d.porewater import compute_pore_pressures
        nodes = np.array([[0.0, 0.0]])
        with pytest.raises(ValueError):
            compute_pore_pressures(nodes, gwt=np.array([1.0, 2.0]))


class TestElementPorePressures:
    """Test element_pore_pressures() centroidal averaging."""

    def test_centroidal_average(self):
        from fem2d.porewater import element_pore_pressures
        nodes = np.array([[0, 0], [1, 0], [0.5, 1], [1.5, 1]])
        elements = np.array([[0, 1, 2], [1, 3, 2]])
        nodal_pp = np.array([10.0, 20.0, 30.0, 0.0])
        pp_elem = element_pore_pressures(nodes, elements, nodal_pp)
        assert pp_elem[0] == pytest.approx(20.0)    # (10+20+30)/3
        assert pp_elem[1] == pytest.approx(50.0 / 3) # (20+0+30)/3

    def test_uniform_pore_pressure(self):
        from fem2d.porewater import element_pore_pressures
        nodes = np.array([[0, 0], [1, 0], [0, 1]])
        elements = np.array([[0, 1, 2]])
        nodal_pp = np.array([15.0, 15.0, 15.0])
        pp_elem = element_pore_pressures(nodes, elements, nodal_pp)
        assert pp_elem[0] == pytest.approx(15.0)


class TestEffectiveStressCorrection:
    """Test effective_stress_correction()."""

    def test_simple_subtraction(self):
        from fem2d.porewater import effective_stress_correction
        sigma_total = np.array([-100.0, -200.0, 50.0])  # [sx, sy, txy]
        u_pore = 50.0
        sigma_eff = effective_stress_correction(sigma_total, u_pore)
        # sigma' = sigma - u*[1,1,0]
        np.testing.assert_allclose(sigma_eff, [-150.0, -250.0, 50.0])

    def test_zero_pore_pressure(self):
        from fem2d.porewater import effective_stress_correction
        sigma_total = np.array([-100.0, -200.0, 30.0])
        sigma_eff = effective_stress_correction(sigma_total, 0.0)
        np.testing.assert_allclose(sigma_eff, sigma_total)

    def test_shear_unchanged(self):
        from fem2d.porewater import effective_stress_correction
        sigma_total = np.array([0.0, 0.0, 100.0])
        sigma_eff = effective_stress_correction(sigma_total, 50.0)
        assert sigma_eff[2] == pytest.approx(100.0)


class TestPorePressureForce:
    """Test pore_pressure_force() assembly."""

    def test_force_vector_shape(self):
        from fem2d.porewater import pore_pressure_force
        nodes = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=float)
        elements = np.array([[0, 1, 2]])
        nodal_pp = np.array([10.0, 10.0, 10.0])
        F = pore_pressure_force(nodes, elements, nodal_pp)
        assert F.shape == (6,)

    def test_nonzero_below_gwt(self):
        from fem2d.porewater import pore_pressure_force
        nodes = np.array([[0, -1], [1, -1], [0.5, 0]], dtype=float)
        elements = np.array([[0, 1, 2]])
        nodal_pp = np.array([9.81, 9.81, 0.0])
        F = pore_pressure_force(nodes, elements, nodal_pp)
        # Should be nonzero since there is a non-zero pore pressure
        assert np.linalg.norm(F) > 0

    def test_zero_above_gwt(self):
        from fem2d.porewater import pore_pressure_force
        nodes = np.array([[0, 5], [1, 5], [0.5, 6]], dtype=float)
        elements = np.array([[0, 1, 2]])
        nodal_pp = np.array([0.0, 0.0, 0.0])
        F = pore_pressure_force(nodes, elements, nodal_pp)
        np.testing.assert_allclose(F, 0.0, atol=1e-15)


class TestSolverWithPorePressure:
    """Test elastic/nonlinear solver integration with pore pressures."""

    def test_elastic_gravity_with_hydrostatic_pp(self):
        """Elastic gravity + hydrostatic pp produces effective stress."""
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes
        from fem2d.materials import elastic_D
        from fem2d.solver import solve_elastic
        from fem2d.porewater import compute_pore_pressures

        nodes, elements = generate_rect_mesh(0, 5, -5, 0, 5, 5)
        bc_nodes = detect_boundary_nodes(nodes)
        D = elastic_D(30000, 0.3)
        gamma = 20.0

        # Without pore pressures
        u_dry, stresses_dry, _ = solve_elastic(
            nodes, elements, D, gamma, bc_nodes)

        # With hydrostatic pore pressures (GWT at surface)
        pp = compute_pore_pressures(nodes, gwt=0.0, gamma_w=9.81)
        u_wet, stresses_wet, _ = solve_elastic(
            nodes, elements, D, gamma, bc_nodes, pore_pressures=pp)

        # Displacements should differ (pp adds upward body force)
        assert not np.allclose(u_dry, u_wet)

        # Vertical displacement: wet should settle less (buoyancy)
        max_settle_dry = np.min(u_dry[1::2])
        max_settle_wet = np.min(u_wet[1::2])
        assert abs(max_settle_wet) < abs(max_settle_dry)

    def test_nonlinear_with_pp_converges(self):
        """Nonlinear solver with MC and pore pressures converges."""
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes
        from fem2d.solver import solve_nonlinear
        from fem2d.porewater import compute_pore_pressures

        nodes, elements = generate_rect_mesh(0, 5, -5, 0, 5, 5)
        bc_nodes = detect_boundary_nodes(nodes)
        mat = [{'E': 30000, 'nu': 0.3, 'c': 10, 'phi': 25, 'psi': 0}]
        gamma = 20.0
        pp = compute_pore_pressures(nodes, gwt=0.0, gamma_w=9.81)

        converged, u, stresses, strains = solve_nonlinear(
            nodes, elements, mat, gamma, bc_nodes,
            n_steps=5, pore_pressures=pp)
        assert converged


# ===========================================================================
# Phase 2: Seepage Tests
# ===========================================================================

class TestCSTPermeabilityMatrix:
    """Test cst_permeability_matrix()."""

    def test_shape_3x3(self):
        from fem2d.porewater import cst_permeability_matrix
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        H = cst_permeability_matrix(coords, k=1e-5)
        assert H.shape == (3, 3)

    def test_symmetric(self):
        from fem2d.porewater import cst_permeability_matrix
        coords = np.array([[0, 0], [2, 0], [1, 1.5]], dtype=float)
        H = cst_permeability_matrix(coords, k=1e-4)
        np.testing.assert_allclose(H, H.T, atol=1e-15)

    def test_positive_semidefinite(self):
        from fem2d.porewater import cst_permeability_matrix
        coords = np.array([[0, 0], [1, 0], [0.5, 0.8]], dtype=float)
        H = cst_permeability_matrix(coords, k=1e-5)
        eigs = np.linalg.eigvalsh(H)
        assert all(e >= -1e-14 for e in eigs)

    def test_scales_with_k(self):
        from fem2d.porewater import cst_permeability_matrix
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        H1 = cst_permeability_matrix(coords, k=1e-5)
        H2 = cst_permeability_matrix(coords, k=2e-5)
        np.testing.assert_allclose(H2, 2.0 * H1, rtol=1e-10)


class TestAssembleFlowSystem:
    """Test assemble_flow_system()."""

    def test_global_shape(self):
        from fem2d.porewater import assemble_flow_system
        from fem2d.mesh import generate_rect_mesh
        nodes, elements = generate_rect_mesh(0, 2, -2, 0, 4, 4)
        H, q = assemble_flow_system(nodes, elements, k=1e-5)
        n = len(nodes)
        assert H.shape == (n, n)
        assert q.shape == (n,)

    def test_symmetric(self):
        from fem2d.porewater import assemble_flow_system
        from fem2d.mesh import generate_rect_mesh
        nodes, elements = generate_rect_mesh(0, 2, -2, 0, 3, 3)
        H, _ = assemble_flow_system(nodes, elements, k=1e-5)
        H_dense = H.toarray()
        np.testing.assert_allclose(H_dense, H_dense.T, atol=1e-15)

    def test_per_element_k(self):
        from fem2d.porewater import assemble_flow_system
        from fem2d.mesh import generate_rect_mesh
        nodes, elements = generate_rect_mesh(0, 1, -1, 0, 2, 2)
        n_elem = len(elements)
        k_arr = np.full(n_elem, 1e-5)
        H, _ = assemble_flow_system(nodes, elements, k=k_arr)
        assert H.shape == (len(nodes), len(nodes))


class TestApplyHeadBCs:
    """Test apply_head_bcs()."""

    def test_prescribed_value_enforced(self):
        from fem2d.porewater import assemble_flow_system, apply_head_bcs
        from fem2d.mesh import generate_rect_mesh
        from scipy.sparse.linalg import spsolve

        nodes, elements = generate_rect_mesh(0, 1, -1, 0, 2, 2)
        H, q = assemble_flow_system(nodes, elements, k=1e-5)

        # Fix all nodes to h=5.0
        head_bcs = [(i, 5.0) for i in range(len(nodes))]
        H_bc, q_bc = apply_head_bcs(H, q, head_bcs)
        h = spsolve(H_bc.tocsc(), q_bc)
        np.testing.assert_allclose(h, 5.0, atol=1e-4)


class TestSolveSeepage:
    """Test solve_seepage()."""

    def test_uniform_head_zero_velocity(self):
        """All nodes at same head → zero velocity."""
        from fem2d.porewater import solve_seepage
        from fem2d.mesh import generate_rect_mesh

        nodes, elements = generate_rect_mesh(0, 4, -4, 0, 4, 4)
        head_bcs = [(i, 10.0) for i in range(len(nodes))]
        result = solve_seepage(nodes, elements, k=1e-5, head_bcs=head_bcs)

        np.testing.assert_allclose(result['head'], 10.0, atol=1e-4)
        np.testing.assert_allclose(result['velocity'], 0.0, atol=1e-10)

    def test_1d_darcy_flow(self):
        """Linear head drop: left h=10, right h=0 → horizontal flow."""
        from fem2d.porewater import solve_seepage
        from fem2d.mesh import generate_rect_mesh

        # Wide short domain: flow from left to right
        nodes, elements = generate_rect_mesh(0, 10, -2, 0, 20, 4)
        k = 1e-4  # m/s

        # Left nodes: h=10, right nodes: h=0
        left_nodes = np.where(np.abs(nodes[:, 0]) < 0.01)[0]
        right_nodes = np.where(np.abs(nodes[:, 0] - 10.0) < 0.01)[0]

        head_bcs = [(int(n), 10.0) for n in left_nodes]
        head_bcs += [(int(n), 0.0) for n in right_nodes]

        result = solve_seepage(nodes, elements, k=k, head_bcs=head_bcs)

        # Gradient dh/dx = (0-10)/10 = -1.0 m/m
        # Velocity vx = -k * dh/dx = -k * (-1) = k = 1e-4 m/s (rightward)
        vel = result['velocity']
        vx_avg = np.mean(vel[:, 0])
        assert abs(vx_avg - k) < k * 0.15  # within 15%


class TestSeepageVelocity:
    """Test seepage_velocity()."""

    def test_direction_and_magnitude(self):
        from fem2d.porewater import seepage_velocity

        nodes = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        elements = np.array([[0, 1, 2]])
        # Linear head field: h = x → grad_h = [1, 0]
        head = nodes[:, 0].copy()  # h = x
        k = 1e-4

        vel = seepage_velocity(nodes, elements, head, k)
        # v = -k * grad(h) = -k * [1, 0] = [-1e-4, 0]
        assert vel.shape == (1, 2)
        assert vel[0, 0] == pytest.approx(-k, rel=1e-10)
        assert vel[0, 1] == pytest.approx(0.0, abs=1e-14)


class TestSeepageResult:
    """Test SeepageResult dataclass."""

    def test_summary_and_to_dict(self):
        from fem2d.results import SeepageResult
        r = SeepageResult(
            n_nodes=100, n_elements=150,
            max_head_m=10.0, min_head_m=0.0,
            max_pore_pressure_kPa=98.1,
            max_velocity_m_per_s=1e-5,
            total_flow_m3_per_s_per_m=1e-4,
        )
        s = r.summary()
        assert "SEEPAGE" in s
        assert "98.1" in s

        d = r.to_dict()
        assert d["n_nodes"] == 100
        assert d["max_head_m"] == 10.0


class TestAnalyzeSeepage:
    """Test analyze_seepage() high-level API."""

    def test_returns_seepage_result(self):
        from fem2d.analysis import analyze_seepage
        from fem2d.mesh import generate_rect_mesh
        from fem2d.results import SeepageResult

        nodes, elements = generate_rect_mesh(0, 4, -4, 0, 4, 4)

        # Left side: h=10, right side: h=0
        left_nodes = np.where(np.abs(nodes[:, 0]) < 0.01)[0]
        right_nodes = np.where(np.abs(nodes[:, 0] - 4.0) < 0.01)[0]
        head_bcs = [(int(n), 10.0) for n in left_nodes]
        head_bcs += [(int(n), 0.0) for n in right_nodes]

        result = analyze_seepage(nodes, elements, k=1e-5, head_bcs=head_bcs)
        assert isinstance(result, SeepageResult)
        assert result.n_nodes > 0
        assert result.max_head_m >= result.min_head_m
        assert result.max_velocity_m_per_s > 0


# ===========================================================================
# Phase 3: Consolidation Tests
# ===========================================================================

class TestCSTCouplingMatrix:
    """Test cst_coupling_matrix()."""

    def test_shape_6x3(self):
        from fem2d.porewater import cst_coupling_matrix
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        Q = cst_coupling_matrix(coords)
        assert Q.shape == (6, 3)

    def test_nonzero_entries(self):
        from fem2d.porewater import cst_coupling_matrix
        coords = np.array([[0, 0], [2, 0], [1, 1.5]], dtype=float)
        Q = cst_coupling_matrix(coords)
        assert np.linalg.norm(Q) > 0


class TestCSTCompressibilityMatrix:
    """Test cst_compressibility_matrix()."""

    def test_shape_3x3(self):
        from fem2d.porewater import cst_compressibility_matrix
        coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
        S = cst_compressibility_matrix(coords)
        assert S.shape == (3, 3)

    def test_symmetric(self):
        from fem2d.porewater import cst_compressibility_matrix
        coords = np.array([[0, 0], [2, 0], [1, 1.5]], dtype=float)
        S = cst_compressibility_matrix(coords)
        np.testing.assert_allclose(S, S.T, atol=1e-20)

    def test_positive_definite(self):
        from fem2d.porewater import cst_compressibility_matrix
        coords = np.array([[0, 0], [1, 0], [0.5, 0.8]], dtype=float)
        S = cst_compressibility_matrix(coords)
        eigs = np.linalg.eigvalsh(S)
        assert all(e > 0 for e in eigs)


class TestAssembleCoupling:
    """Test assemble_coupling()."""

    def test_global_shape(self):
        from fem2d.porewater import assemble_coupling
        from fem2d.mesh import generate_rect_mesh
        nodes, elements = generate_rect_mesh(0, 2, -2, 0, 3, 3)
        Q = assemble_coupling(nodes, elements)
        n = len(nodes)
        assert Q.shape == (2 * n, n)


class TestAssembleCompressibility:
    """Test assemble_compressibility()."""

    def test_global_shape_symmetric(self):
        from fem2d.porewater import assemble_compressibility
        from fem2d.mesh import generate_rect_mesh
        nodes, elements = generate_rect_mesh(0, 2, -2, 0, 3, 3)
        S = assemble_compressibility(nodes, elements)
        n = len(nodes)
        assert S.shape == (n, n)
        S_dense = S.toarray()
        np.testing.assert_allclose(S_dense, S_dense.T, atol=1e-20)


class TestSolveConsolidation:
    """Test solve_consolidation() — 1D Terzaghi column."""

    def _setup_column(self):
        """Create a simple consolidation problem."""
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes
        from fem2d.porewater import compute_pore_pressures

        width = 2.0
        depth = 10.0
        nodes, elements = generate_rect_mesh(0, width, -depth, 0, 3, 10)
        bc_nodes = detect_boundary_nodes(nodes)

        mat = [{'E': 10000, 'nu': 0.3}]
        gamma = 18.0
        k = 1e-6

        # Drain at top only
        top_nodes = np.where(np.abs(nodes[:, 1]) < 0.01)[0]
        head_bcs = [(int(n), 0.0) for n in top_nodes]

        # Initial hydrostatic pore pressure (GWT at surface)
        pp_0 = compute_pore_pressures(nodes, gwt=0.0, gamma_w=9.81)

        return nodes, elements, mat, gamma, bc_nodes, k, head_bcs, pp_0

    def test_settlement_increases_over_time(self):
        from fem2d.porewater import solve_consolidation

        nodes, elements, mat, gamma, bc_nodes, k, head_bcs, pp_0 = \
            self._setup_column()

        times = np.array([0, 100, 1000, 10000, 100000])
        result = solve_consolidation(
            nodes, elements, mat, gamma, bc_nodes,
            k=k, head_bcs=head_bcs, time_steps=times,
            pore_pressures_0=pp_0)

        # Settlement magnitude should generally increase
        # (more negative = more settlement)
        settlements = result['settlements']
        # Later settlements should be at least as large as early ones
        # (settlement becomes more negative over time)
        assert settlements[-1] <= settlements[0] or abs(settlements[-1]) >= abs(settlements[0]) * 0.99

    def test_pore_pressure_decreases_over_time(self):
        from fem2d.porewater import solve_consolidation

        nodes, elements, mat, gamma, bc_nodes, k, head_bcs, pp_0 = \
            self._setup_column()

        times = np.array([0, 1000, 10000, 100000, 1000000])
        result = solve_consolidation(
            nodes, elements, mat, gamma, bc_nodes,
            k=k, head_bcs=head_bcs, time_steps=times,
            pore_pressures_0=pp_0)

        pp_hist = result['pore_pressures']
        # Max pore pressure should decrease over time
        max_pp_early = np.max(pp_hist[1])
        max_pp_late = np.max(pp_hist[-1])
        assert max_pp_late <= max_pp_early

    def test_converged(self):
        from fem2d.porewater import solve_consolidation

        nodes, elements, mat, gamma, bc_nodes, k, head_bcs, pp_0 = \
            self._setup_column()

        times = np.array([0, 1000, 10000])
        result = solve_consolidation(
            nodes, elements, mat, gamma, bc_nodes,
            k=k, head_bcs=head_bcs, time_steps=times,
            pore_pressures_0=pp_0)
        assert result['converged']


class TestConsolidationResult:
    """Test ConsolidationResult dataclass."""

    def test_summary_and_to_dict(self):
        from fem2d.results import ConsolidationResult
        r = ConsolidationResult(
            n_nodes=50, n_elements=80,
            n_time_steps=5,
            max_settlement_m=-0.005,
            max_excess_pore_pressure_kPa=50.0,
            degree_of_consolidation=0.75,
            converged=True,
        )
        s = r.summary()
        assert "CONSOLIDATION" in s
        assert "0.75" in s

        d = r.to_dict()
        assert d["n_nodes"] == 50
        assert d["converged"] is True
        assert d["degree_of_consolidation"] == 0.75


class TestAnalyzeConsolidation:
    """Test analyze_consolidation() high-level API."""

    def test_returns_consolidation_result(self):
        from fem2d.analysis import analyze_consolidation
        from fem2d.results import ConsolidationResult

        soil = [{'E': 10000, 'nu': 0.3, 'gamma': 18}]
        result = analyze_consolidation(
            width=2.0, depth=5.0,
            soil_layers=soil, k=1e-5,
            load_q=50.0,
            time_points=[0, 100, 1000, 10000],
            nx=4, ny=8)

        assert isinstance(result, ConsolidationResult)
        assert result.n_time_steps == 4
        assert result.converged


# ===========================================================================
# Phase: Integration Tests
# ===========================================================================

class TestSlopeWithGWT:
    """SRM FOS with GWT should be less than FOS without GWT."""

    def test_gwt_lowers_fos(self):
        from fem2d.analysis import analyze_slope_srm

        # Steep slope with weak soil to ensure FOS < 3.0
        surface = [(0, 10), (5, 10), (15, 0), (20, 0)]
        soil = [{
            'name': 'clay',
            'bottom_elevation': -10,
            'E': 20000, 'nu': 0.3,
            'c': 5, 'phi': 15, 'gamma': 18,
        }]

        # Dry slope
        result_dry = analyze_slope_srm(surface, soil,
                                       nx=15, ny=8, srf_tol=0.05)
        fos_dry = result_dry.FOS

        # Saturated slope (GWT at top of slope)
        result_wet = analyze_slope_srm(surface, soil,
                                       nx=15, ny=8, srf_tol=0.05,
                                       gwt=10.0, gamma_w=9.81)
        fos_wet = result_wet.FOS

        # GWT should reduce FOS
        assert fos_wet < fos_dry

    def test_polyline_gwt_slope(self):
        """Polyline GWT on a slope should produce a valid FOS."""
        from fem2d.analysis import analyze_slope_srm

        surface = [(0, 10), (5, 10), (15, 0), (20, 0)]
        soil = [{
            'name': 'clay',
            'bottom_elevation': -10,
            'E': 20000, 'nu': 0.3,
            'c': 5, 'phi': 15, 'gamma': 18,
        }]

        # GWT follows terrain but 3m below surface
        gwt_poly = np.array([[0, 7], [5, 7], [15, -3], [20, -3]])

        result = analyze_slope_srm(surface, soil,
                                   nx=15, ny=8, srf_tol=0.05,
                                   gwt=gwt_poly)
        assert result.FOS > 0
        assert result.converged


class TestSeepageThenMechanical:
    """Solve seepage → use pore pressures in mechanical analysis."""

    def test_coupled_seepage_mechanical(self):
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes
        from fem2d.porewater import solve_seepage
        from fem2d.materials import elastic_D
        from fem2d.solver import solve_elastic

        nodes, elements = generate_rect_mesh(0, 10, -10, 0, 8, 8)
        bc_nodes = detect_boundary_nodes(nodes)

        # Seepage: left head = 10m, right head = 0m
        left_nodes = np.where(np.abs(nodes[:, 0]) < 0.01)[0]
        right_nodes = np.where(np.abs(nodes[:, 0] - 10.0) < 0.01)[0]
        head_bcs = [(int(n), 10.0) for n in left_nodes]
        head_bcs += [(int(n), 0.0) for n in right_nodes]

        seep_result = solve_seepage(nodes, elements, k=1e-5,
                                     head_bcs=head_bcs)
        pp = seep_result['pore_pressures']

        # Mechanical analysis with seepage pore pressures
        D = elastic_D(30000, 0.3)
        u, stresses, strains = solve_elastic(
            nodes, elements, D, gamma=20.0, bc_nodes=bc_nodes,
            pore_pressures=pp)

        # Should produce valid displacement field
        assert not np.any(np.isnan(u))
        assert np.max(np.abs(u)) > 0


class TestArtesianCondition:
    """Per-node elevated heads → higher pore pressures."""

    def test_artesian_higher_pp(self):
        from fem2d.porewater import compute_pore_pressures

        nodes = np.array([
            [0.0, -5.0],
            [1.0, -5.0],
            [2.0, -5.0],
        ])

        # Normal hydrostatic (GWT at surface)
        pp_normal = compute_pore_pressures(nodes, gwt=0.0, gamma_w=9.81)

        # Artesian: head elevated to 10m above surface
        heads_artesian = np.array([10.0, 10.0, 10.0])
        pp_artesian = compute_pore_pressures(nodes, gwt=heads_artesian,
                                              gamma_w=9.81)

        # Artesian pp should be higher than normal
        assert np.all(pp_artesian > pp_normal)


class TestExcavationWithGWT:
    """Test excavation analysis with groundwater."""

    def test_excavation_with_gwt_runs(self):
        from fem2d.analysis import analyze_excavation

        soil = [{
            'name': 'clay',
            'bottom_elevation': -20,
            'E': 30000, 'nu': 0.3,
            'c': 10, 'phi': 25, 'psi': 0, 'gamma': 19,
        }]
        result = analyze_excavation(
            width=5, depth=3, wall_depth=8,
            soil_layers=soil, wall_EI=50000, wall_EA=5e6,
            nx=12, ny=8, gwt=-1.0, gamma_w=9.81)
        assert result.converged


class TestPorePressureForceEquilibrium:
    """Check that pp force is consistent with analytical."""

    def test_single_element_force(self):
        """For uniform pp, F_pp should equal t * A * B^T * m * pp."""
        from fem2d.porewater import pore_pressure_force
        from fem2d.elements import cst_B, cst_area

        coords = np.array([[0, 0], [2, 0], [1, 1.5]], dtype=float)
        nodes = coords
        elements = np.array([[0, 1, 2]])
        pp_val = 25.0
        nodal_pp = np.array([pp_val, pp_val, pp_val])

        F = pore_pressure_force(nodes, elements, nodal_pp, t=1.0)

        # Manual calculation
        B, A = cst_B(coords)
        m = np.array([1, 1, 0])
        F_expected = A * (B.T @ m) * pp_val
        np.testing.assert_allclose(F, F_expected, atol=1e-10)


class TestSeepageDamProblem:
    """Dam seepage: high head upstream, low head downstream."""

    def test_dam_head_decreases(self):
        from fem2d.porewater import solve_seepage
        from fem2d.mesh import generate_rect_mesh

        nodes, elements = generate_rect_mesh(0, 20, -10, 0, 15, 8)

        # Upstream (left): h=10, downstream (right): h=2
        left = np.where(np.abs(nodes[:, 0]) < 0.01)[0]
        right = np.where(np.abs(nodes[:, 0] - 20.0) < 0.01)[0]
        head_bcs = [(int(n), 10.0) for n in left]
        head_bcs += [(int(n), 2.0) for n in right]

        result = solve_seepage(nodes, elements, k=1e-5, head_bcs=head_bcs)

        # Head should decrease from left to right
        # Pick interior nodes at mid-height
        mid_y = np.where(np.abs(nodes[:, 1] + 5.0) < 1.5)[0]
        if len(mid_y) > 2:
            x_sorted = np.argsort(nodes[mid_y, 0])
            heads_sorted = result['head'][mid_y[x_sorted]]
            # Should be monotonically decreasing
            diffs = np.diff(heads_sorted)
            assert np.all(diffs <= 0.01)  # allow small numerical noise


class TestConsolidationStability:
    """Solution stable for large time steps (implicit scheme)."""

    def test_large_time_step_stable(self):
        from fem2d.porewater import solve_consolidation, compute_pore_pressures
        from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes

        nodes, elements = generate_rect_mesh(0, 2, -5, 0, 3, 8)
        bc_nodes = detect_boundary_nodes(nodes)
        mat = [{'E': 10000, 'nu': 0.3}]
        gamma = 18.0
        k = 1e-5

        top_nodes = np.where(np.abs(nodes[:, 1]) < 0.01)[0]
        head_bcs = [(int(n), 0.0) for n in top_nodes]
        pp_0 = compute_pore_pressures(nodes, gwt=0.0, gamma_w=9.81)

        # Very large time step — should still converge (implicit)
        times = np.array([0, 1e6, 1e8])
        result = solve_consolidation(
            nodes, elements, mat, gamma, bc_nodes,
            k=k, head_bcs=head_bcs, time_steps=times,
            pore_pressures_0=pp_0)

        assert result['converged']
        assert not np.any(np.isnan(result['displacements']))
        assert not np.any(np.isnan(result['pore_pressures']))
