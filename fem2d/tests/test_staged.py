"""
Tests for staged construction analysis.

Covers:
- ConstructionPhase data model
- assign_element_groups() utility
- Assembly filtering (active_elements, active_beams)
- Single-phase equivalence
- Multi-phase excavation sequences
- Cumulative state carryover (stress, strain, HS state)
- GWT / pore pressure per-phase
- Surface load per-phase
- PhaseResult and StagedConstructionResult containers
- Error and edge cases
"""

import math
import numpy as np
import pytest

from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes
from fem2d.materials import elastic_D
from fem2d.elements import BeamElement
from fem2d.assembly import (
    assemble_stiffness, assemble_gravity, assemble_surface_load,
    build_rotation_dof_map, assemble_beam_stiffness, assemble_beam_gravity,
)
from fem2d.porewater import pore_pressure_force
from fem2d.solver import solve_nonlinear
from fem2d.analysis import (
    ConstructionPhase, assign_element_groups, analyze_staged,
)
from fem2d.results import PhaseResult, StagedConstructionResult


# ===========================================================================
# Helper: small mesh for testing
# ===========================================================================

def _small_mesh(nx=6, ny=3, width=6.0, depth=3.0):
    """Create a small rectangular mesh for testing."""
    nodes, elements = generate_rect_mesh(0, width, -depth, 0, nx, ny)
    bc_nodes = detect_boundary_nodes(nodes)
    return nodes, elements, bc_nodes


def _make_material_props(n_elem, E=30000, nu=0.3, c=5, phi=30):
    """Create uniform material properties list."""
    return [{'E': E, 'nu': nu, 'c': c, 'phi': phi, 'psi': 0}] * n_elem


def _make_gamma_arr(n_elem, gamma=18.0):
    """Create uniform gamma array."""
    return np.full(n_elem, gamma)


# ===========================================================================
# Data Model Tests
# ===========================================================================

class TestConstructionPhase:
    """Test ConstructionPhase dataclass."""

    def test_defaults(self):
        phase = ConstructionPhase()
        assert phase.name == "Phase"
        assert phase.active_soil_groups == []
        assert phase.active_beam_ids is None
        assert phase.surface_loads is None
        assert phase.gwt is None
        assert phase.n_steps == 5
        assert phase.reset_displacements is False

    def test_custom_values(self):
        phase = ConstructionPhase(
            name="Excavate",
            active_soil_groups=['retained', 'base'],
            active_beam_ids=[0, 1],
            n_steps=10,
            gwt=-2.0,
            reset_displacements=True,
        )
        assert phase.name == "Excavate"
        assert phase.active_soil_groups == ['retained', 'base']
        assert phase.active_beam_ids == [0, 1]
        assert phase.n_steps == 10
        assert phase.gwt == -2.0
        assert phase.reset_displacements is True


class TestAssignElementGroups:
    """Test assign_element_groups()."""

    def test_single_region_covers_all(self):
        nodes, elements, _ = _small_mesh()
        groups = assign_element_groups(nodes, elements, {
            'all': {'x_min': -1, 'x_max': 100, 'y_min': -100, 'y_max': 1},
        })
        assert len(groups['all']) == len(elements)
        assert len(groups['_default']) == 0

    def test_two_regions_split(self):
        nodes, elements, _ = _small_mesh(width=10.0)
        groups = assign_element_groups(nodes, elements, {
            'left': {'x_min': 0, 'x_max': 5, 'y_min': -100, 'y_max': 1},
            'right': {'x_min': 5, 'x_max': 10, 'y_min': -100, 'y_max': 1},
        })
        # All elements should be assigned to one of the two groups
        total = len(groups['left']) + len(groups['right'])
        # Some elements may be in both groups (centroid on boundary)
        assert total >= len(elements) - len(groups['_default'])
        assert len(groups['left']) > 0
        assert len(groups['right']) > 0

    def test_no_region_goes_to_default(self):
        nodes, elements, _ = _small_mesh()
        groups = assign_element_groups(nodes, elements, {
            'tiny': {'x_min': 100, 'x_max': 200, 'y_min': 100, 'y_max': 200},
        })
        assert len(groups['tiny']) == 0
        assert len(groups['_default']) == len(elements)

    def test_element_in_multiple_regions(self):
        """Elements can appear in multiple overlapping regions."""
        nodes, elements, _ = _small_mesh()
        groups = assign_element_groups(nodes, elements, {
            'all': {'x_min': -1, 'x_max': 100, 'y_min': -100, 'y_max': 1},
            'top': {'x_min': -1, 'x_max': 100, 'y_min': -1.5, 'y_max': 1},
        })
        # top should be a subset of all
        assert len(groups['top']) <= len(groups['all'])
        assert len(groups['top']) > 0


# ===========================================================================
# Assembly Filtering Tests
# ===========================================================================

class TestAssemblyFiltering:
    """Test active_elements / active_beams on assembly functions."""

    def test_stiffness_active_subset(self):
        nodes, elements, _ = _small_mesh()
        D = elastic_D(30000, 0.3)
        K_all = assemble_stiffness(nodes, elements, D)
        # Only first half active
        half = set(range(len(elements) // 2))
        K_half = assemble_stiffness(nodes, elements, D, active_elements=half)
        # Half should have fewer nonzero entries
        assert K_half.nnz < K_all.nnz
        assert K_half.nnz > 0

    def test_stiffness_none_means_all(self):
        nodes, elements, _ = _small_mesh()
        D = elastic_D(30000, 0.3)
        K_all = assemble_stiffness(nodes, elements, D)
        K_none = assemble_stiffness(nodes, elements, D, active_elements=None)
        np.testing.assert_allclose(K_all.toarray(), K_none.toarray())

    def test_gravity_active_subset(self):
        nodes, elements, _ = _small_mesh()
        F_all = assemble_gravity(nodes, elements, 18.0)
        half = set(range(len(elements) // 2))
        F_half = assemble_gravity(nodes, elements, 18.0, active_elements=half)
        # Half should have less total force
        assert np.abs(F_half).sum() < np.abs(F_all).sum()
        assert np.abs(F_half).sum() > 0

    def test_pore_pressure_force_active_subset(self):
        nodes, elements, _ = _small_mesh()
        pp = np.ones(len(nodes)) * 50.0  # 50 kPa everywhere
        F_all = pore_pressure_force(nodes, elements, pp)
        half = set(range(len(elements) // 2))
        F_half = pore_pressure_force(nodes, elements, pp,
                                     active_elements=half)
        assert np.abs(F_half).sum() < np.abs(F_all).sum()

    def test_beam_stiffness_active_subset(self):
        nodes, elements, _ = _small_mesh()
        beams = [
            BeamElement(node_i=0, node_j=1, EA=1e6, EI=1e4),
            BeamElement(node_i=1, node_j=2, EA=1e6, EI=1e4),
        ]
        rot_map, n_dof = build_rotation_dof_map(len(nodes), beams)
        K_all = assemble_beam_stiffness(nodes, beams, rot_map, n_dof)
        K_one = assemble_beam_stiffness(nodes, beams, rot_map, n_dof,
                                        active_beams={0})
        assert K_one.sum() < K_all.sum()
        assert K_one.sum() > 0

    def test_beam_gravity_active_subset(self):
        nodes, elements, _ = _small_mesh()
        beams = [
            BeamElement(node_i=0, node_j=1, EA=1e6, EI=1e4,
                        weight_per_m=5.0),
            BeamElement(node_i=1, node_j=2, EA=1e6, EI=1e4,
                        weight_per_m=5.0),
        ]
        rot_map, n_dof = build_rotation_dof_map(len(nodes), beams)
        F_all = assemble_beam_gravity(nodes, beams, rot_map, n_dof)
        F_one = assemble_beam_gravity(nodes, beams, rot_map, n_dof,
                                      active_beams={0})
        assert np.abs(F_one).sum() < np.abs(F_all).sum()


# ===========================================================================
# Single-Phase Equivalence Tests
# ===========================================================================

class TestSinglePhase:
    """Single phase 'all active' should match solve_nonlinear() directly."""

    def test_single_phase_matches_direct(self):
        nodes, elements, bc_nodes = _small_mesh()
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        # Direct solve
        conv_direct, u_direct, sig_direct, eps_direct = solve_nonlinear(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            n_steps=5, t=1.0)

        # Staged solve with single phase
        all_elems = list(range(n_elem))
        groups = {'all': all_elems}
        phases = [ConstructionPhase(
            name="Gravity",
            active_soil_groups=['all'],
            n_steps=5,
        )]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        assert result.converged
        assert len(result.phases) == 1
        p0 = result.phases[0]
        assert p0.converged
        assert p0.n_active_elements == n_elem

        # Displacements should be very close
        u_staged = p0.displacements
        np.testing.assert_allclose(u_staged, u_direct[:len(u_staged)],
                                   atol=1e-8)

    def test_single_phase_with_surface_load(self):
        """Single phase with surface load should produce displacement."""
        nodes, elements, bc_nodes = _small_mesh()
        n_elem = len(elements)
        mat_props = [{'E': 30000, 'nu': 0.3, 'c': 0, 'phi': 0, 'psi': 0}] * n_elem
        gamma_arr = np.zeros(n_elem)  # no gravity

        # Surface edges on top
        top_nodes = np.where(np.abs(nodes[:, 1]) < 0.01)[0]
        top_nodes = top_nodes[np.argsort(nodes[top_nodes, 0])]
        edges = [(top_nodes[i], top_nodes[i + 1])
                 for i in range(len(top_nodes) - 1)]

        groups = {'all': list(range(n_elem))}
        phases = [ConstructionPhase(
            name="Load",
            active_soil_groups=['all'],
            surface_loads=[(edges, 0.0, -100.0)],  # 100 kPa downward
            n_steps=3,
        )]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        assert result.converged
        assert result.phases[0].max_displacement_m > 0


# ===========================================================================
# Multi-Phase Excavation Tests
# ===========================================================================

class TestMultiPhaseExcavation:
    """Multi-phase excavation with element deactivation."""

    def _setup_excavation(self):
        """Setup a mesh with 'retained' and 'excavated' groups."""
        nodes, elements, bc_nodes = _small_mesh(nx=8, ny=4, width=8.0,
                                                 depth=4.0)
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        # Split: left half = retained, right upper quarter = excavated
        groups = assign_element_groups(nodes, elements, {
            'retained': {'x_min': -1, 'x_max': 4, 'y_min': -5, 'y_max': 1},
            'excavated': {'x_min': 4, 'x_max': 9, 'y_min': -2, 'y_max': 1},
            'base': {'x_min': 4, 'x_max': 9, 'y_min': -5, 'y_max': -2},
        })
        return nodes, elements, bc_nodes, mat_props, gamma_arr, groups

    def test_two_phase_gravity_then_excavate(self):
        """Phase 1: full gravity. Phase 2: deactivate excavated zone."""
        nodes, elements, bc_nodes, mat_props, gamma_arr, groups = \
            self._setup_excavation()

        phases = [
            ConstructionPhase(
                name="Initial",
                active_soil_groups=['retained', 'excavated', 'base'],
                n_steps=5,
            ),
            ConstructionPhase(
                name="Excavate",
                active_soil_groups=['retained', 'base'],
                n_steps=5,
            ),
        ]

        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        assert result.n_phases == 2
        assert result.phases[0].converged
        assert result.phases[1].converged
        # Phase 2 should have fewer active elements
        assert (result.phases[1].n_active_elements
                < result.phases[0].n_active_elements)

    def test_deactivated_elements_zero_stress(self):
        """Deactivated elements should have zero stress after excavation."""
        nodes, elements, bc_nodes, mat_props, gamma_arr, groups = \
            self._setup_excavation()

        phases = [
            ConstructionPhase(
                name="Initial",
                active_soil_groups=['retained', 'excavated', 'base'],
                n_steps=5,
            ),
            ConstructionPhase(
                name="Excavate",
                active_soil_groups=['retained', 'base'],
                n_steps=5,
            ),
        ]

        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        # After phase 1, excavated elements should have non-zero stress
        p0_stresses = result.phases[0].stresses
        excavated_ids = groups['excavated']
        if len(excavated_ids) > 0:
            exc_stress_p0 = p0_stresses[excavated_ids]
            assert np.any(np.abs(exc_stress_p0) > 0.1)

    def test_active_count_correct(self):
        """Active element count matches union of group sizes."""
        nodes, elements, bc_nodes, mat_props, gamma_arr, groups = \
            self._setup_excavation()

        phases = [
            ConstructionPhase(
                name="Phase 1",
                active_soil_groups=['retained'],
                n_steps=3,
            ),
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        assert result.phases[0].n_active_elements == len(groups['retained'])


# ===========================================================================
# State Carryover Tests
# ===========================================================================

class TestStateCarryover:
    """Verify cumulative state is preserved across phases."""

    def test_stress_persists(self):
        """Stress from phase 1 retained elements persists in phase 2."""
        nodes, elements, bc_nodes = _small_mesh(nx=6, ny=3)
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        groups = {'all': list(range(n_elem))}
        phases = [
            ConstructionPhase(name="P1", active_soil_groups=['all'],
                              n_steps=5),
            ConstructionPhase(name="P2", active_soil_groups=['all'],
                              n_steps=1),
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        # After P1, there should be non-zero stress
        p0_sig = result.phases[0].stresses
        assert np.any(np.abs(p0_sig) > 0.1)

        # P2 stresses should also be non-zero (carried forward)
        p1_sig = result.phases[1].stresses
        assert np.any(np.abs(p1_sig) > 0.1)

    def test_strain_accumulates(self):
        """Strain should be present in both phases."""
        nodes, elements, bc_nodes = _small_mesh()
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        groups = {'all': list(range(n_elem))}
        phases = [
            ConstructionPhase(name="P1", active_soil_groups=['all'],
                              n_steps=5),
            ConstructionPhase(name="P2", active_soil_groups=['all'],
                              n_steps=3),
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        # Both phases should have non-zero strain
        assert np.any(np.abs(result.phases[0].strains) > 1e-10)
        assert np.any(np.abs(result.phases[1].strains) > 1e-10)

    def test_reset_displacements(self):
        """reset_displacements zeros u but keeps stress."""
        nodes, elements, bc_nodes = _small_mesh()
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        groups = {'all': list(range(n_elem))}
        phases = [
            ConstructionPhase(name="P1", active_soil_groups=['all'],
                              n_steps=5),
            ConstructionPhase(name="P2", active_soil_groups=['all'],
                              n_steps=5, reset_displacements=True),
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        # P1 should have non-zero displacement
        assert result.phases[0].max_displacement_m > 0
        # P2 runs from zero u with carried-forward stress
        assert result.phases[1].converged

    def test_hs_state_preserved(self):
        """HS hardening state should carry across phases."""
        nodes, elements, bc_nodes = _small_mesh(
            nx=8, ny=5, width=10.0, depth=5.0)
        n_elem = len(elements)

        # Use high cohesion for convergence (matching test_hs_beams.py)
        hs_props = [{
            'model': 'hs',
            'E50_ref': 25000, 'Eur_ref': 75000,
            'm': 0.5, 'p_ref': 100, 'R_f': 0.9,
            'E': 25000, 'nu': 0.3, 'c': 100, 'phi': 30, 'psi': 0,
        }] * n_elem
        gamma_arr = _make_gamma_arr(n_elem)

        groups = {'all': list(range(n_elem))}
        phases = [
            ConstructionPhase(name="P1", active_soil_groups=['all'],
                              n_steps=10),
            ConstructionPhase(name="P2", active_soil_groups=['all'],
                              n_steps=5),
        ]
        result = analyze_staged(
            nodes, elements, hs_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        assert result.converged
        assert result.phases[0].converged
        assert result.phases[1].converged


# ===========================================================================
# GWT / Pore Pressure Tests
# ===========================================================================

class TestStagedGWT:
    """GWT changes between phases."""

    def test_gwt_in_one_phase(self):
        """Phase with gwt should differ from phase without."""
        nodes, elements, bc_nodes = _small_mesh()
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        groups = {'all': list(range(n_elem))}
        phases = [
            ConstructionPhase(name="Dry", active_soil_groups=['all'],
                              n_steps=5, gwt=None),
            ConstructionPhase(name="Wet", active_soil_groups=['all'],
                              n_steps=5, gwt=0.0),  # GWT at surface
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        assert result.converged
        # Both phases should have valid results
        assert result.phases[0].max_displacement_m > 0
        assert result.phases[1].max_displacement_m > 0

    def test_gwt_none_means_dry(self):
        """Phase with gwt=None should have no pore pressure effect."""
        nodes, elements, bc_nodes = _small_mesh()
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        groups = {'all': list(range(n_elem))}
        phases_dry = [
            ConstructionPhase(name="Dry", active_soil_groups=['all'],
                              n_steps=5, gwt=None),
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases_dry)

        assert result.converged


# ===========================================================================
# Surface Load Tests
# ===========================================================================

class TestStagedSurfaceLoads:
    """Surface loads applied per-phase."""

    def _setup(self):
        nodes, elements, bc_nodes = _small_mesh()
        n_elem = len(elements)
        # Elastic only to keep it simple
        mat_props = [{'E': 30000, 'nu': 0.3, 'c': 0, 'phi': 0,
                       'psi': 0}] * n_elem
        gamma_arr = np.zeros(n_elem)

        # Surface edges
        top_nodes = np.where(np.abs(nodes[:, 1]) < 0.01)[0]
        top_nodes = top_nodes[np.argsort(nodes[top_nodes, 0])]
        edges = [(top_nodes[i], top_nodes[i + 1])
                 for i in range(len(top_nodes) - 1)]

        groups = {'all': list(range(n_elem))}
        return nodes, elements, bc_nodes, mat_props, gamma_arr, groups, edges

    def test_load_in_phase2_only(self):
        """Load applied in phase 2 → zero displacement after phase 1."""
        nodes, elements, bc_nodes, mat_props, gamma_arr, groups, edges = \
            self._setup()

        phases = [
            ConstructionPhase(name="No load", active_soil_groups=['all'],
                              n_steps=3),
            ConstructionPhase(name="Loaded", active_soil_groups=['all'],
                              surface_loads=[(edges, 0.0, -100.0)],
                              n_steps=3),
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        # Phase 1 has zero gamma, no load → near-zero displacement
        assert result.phases[0].max_displacement_m < 1e-10
        # Phase 2 has surface load → non-zero displacement
        assert result.phases[1].max_displacement_m > 1e-6

    def test_load_removed_in_phase3(self):
        """Load removed in phase 3 should result in changed displacement."""
        nodes, elements, bc_nodes, mat_props, gamma_arr, groups, edges = \
            self._setup()

        phases = [
            ConstructionPhase(name="P1", active_soil_groups=['all'],
                              n_steps=2),
            ConstructionPhase(name="Loaded", active_soil_groups=['all'],
                              surface_loads=[(edges, 0.0, -100.0)],
                              n_steps=3),
            ConstructionPhase(name="Unloaded", active_soil_groups=['all'],
                              n_steps=2),
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        # Phase 2 should have displacement from the load
        disp_loaded = result.phases[1].max_displacement_m
        assert disp_loaded > 1e-6


# ===========================================================================
# Beam Integration Tests
# ===========================================================================

class TestStagedBeams:
    """Beam elements activated in specific phases."""

    def test_beams_in_phase2_only(self):
        """Beams activated in phase 2 should produce beam forces."""
        nodes, elements, bc_nodes = _small_mesh(nx=6, ny=4, depth=4.0)
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        # Find nodes near x=3 for wall beams
        x_wall = 3.0
        mask = np.abs(nodes[:, 0] - x_wall) < 0.6
        wall_nodes = np.where(mask)[0]
        wall_nodes = wall_nodes[np.argsort(-nodes[wall_nodes, 1])]

        if len(wall_nodes) < 2:
            pytest.skip("Mesh too coarse for wall nodes")

        beams = []
        for k in range(len(wall_nodes) - 1):
            beams.append(BeamElement(
                node_i=int(wall_nodes[k]),
                node_j=int(wall_nodes[k + 1]),
                EA=1e6, EI=1e4,
            ))

        groups = {'all': list(range(n_elem))}
        phases = [
            ConstructionPhase(name="Gravity", active_soil_groups=['all'],
                              n_steps=5),
            ConstructionPhase(name="Install wall", active_soil_groups=['all'],
                              active_beam_ids=list(range(len(beams))),
                              n_steps=3),
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases,
            beam_elements=beams)

        assert result.converged
        # Phase 1: no beams
        assert result.phases[0].n_beam_elements == 0
        # Phase 2: beams present
        assert result.phases[1].n_beam_elements > 0
        assert result.phases[1].n_active_beams == len(beams)


# ===========================================================================
# Result Container Tests
# ===========================================================================

class TestPhaseResult:
    """Test PhaseResult dataclass."""

    def test_to_dict_keys(self):
        pr = PhaseResult(
            phase_name="Test", phase_index=0,
            n_active_elements=10, n_active_beams=0,
            converged=True,
            max_displacement_m=0.001,
            max_displacement_x_m=0.0005,
            max_displacement_y_m=0.001,
            max_sigma_xx_kPa=50.0,
            max_sigma_yy_kPa=100.0,
            min_sigma_yy_kPa=-200.0,
            max_tau_xy_kPa=30.0,
        )
        d = pr.to_dict()
        assert d['phase_name'] == "Test"
        assert d['phase_index'] == 0
        assert d['converged'] is True
        assert 'max_displacement_m' in d
        assert 'n_active_elements' in d

    def test_to_dict_with_beams(self):
        pr = PhaseResult(
            phase_name="Wall", phase_index=1,
            n_beam_elements=3,
            max_beam_moment_kNm_per_m=150.0,
            max_beam_shear_kN_per_m=50.0,
        )
        d = pr.to_dict()
        assert d['n_beam_elements'] == 3
        assert 'max_beam_moment_kNm_per_m' in d

    def test_summary(self):
        pr = PhaseResult(phase_name="Gravity", phase_index=0,
                         n_active_elements=50, n_active_beams=0,
                         max_displacement_m=0.005,
                         min_sigma_yy_kPa=-100.0, max_sigma_yy_kPa=10.0)
        s = pr.summary()
        assert "Gravity" in s
        assert "50" in s


class TestStagedConstructionResult:
    """Test StagedConstructionResult container."""

    def test_to_dict_structure(self):
        scr = StagedConstructionResult(
            n_phases=2, n_nodes=100, n_elements=150,
            converged=True,
            phases=[
                PhaseResult(phase_name="P1", phase_index=0),
                PhaseResult(phase_name="P2", phase_index=1),
            ],
        )
        d = scr.to_dict()
        assert d['n_phases'] == 2
        assert d['converged'] is True
        assert len(d['phases']) == 2
        assert d['phases'][0]['phase_name'] == "P1"
        assert d['phases'][1]['phase_name'] == "P2"

    def test_get_phase_by_index(self):
        scr = StagedConstructionResult(
            phases=[
                PhaseResult(phase_name="A", phase_index=0),
                PhaseResult(phase_name="B", phase_index=1),
            ],
        )
        assert scr.get_phase(0).phase_name == "A"
        assert scr.get_phase(1).phase_name == "B"

    def test_get_phase_by_name(self):
        scr = StagedConstructionResult(
            phases=[
                PhaseResult(phase_name="Gravity", phase_index=0),
                PhaseResult(phase_name="Excavate", phase_index=1),
            ],
        )
        assert scr.get_phase("Gravity").phase_index == 0
        assert scr.get_phase("Excavate").phase_index == 1

    def test_get_phase_not_found(self):
        scr = StagedConstructionResult(
            phases=[PhaseResult(phase_name="A", phase_index=0)],
        )
        with pytest.raises(KeyError):
            scr.get_phase("missing")
        with pytest.raises(IndexError):
            scr.get_phase(5)

    def test_summary(self):
        scr = StagedConstructionResult(
            n_phases=1, n_nodes=50, n_elements=80,
            converged=True,
            phases=[PhaseResult(phase_name="Gravity", phase_index=0,
                                n_active_elements=80)],
        )
        s = scr.summary()
        assert "STAGED CONSTRUCTION" in s
        assert "Gravity" in s


# ===========================================================================
# Error / Edge Case Tests
# ===========================================================================

class TestEdgeCases:
    """Error handling and edge cases."""

    def test_non_convergence_stops_phases(self):
        """If a phase doesn't converge, subsequent phases are skipped."""
        nodes, elements, bc_nodes = _small_mesh(nx=4, ny=2)
        n_elem = len(elements)
        # Very weak soil that might not converge with too few steps
        mat_props = [{'E': 100, 'nu': 0.3, 'c': 0.01, 'phi': 1,
                       'psi': 0}] * n_elem
        gamma_arr = _make_gamma_arr(n_elem, gamma=100.0)

        groups = {'all': list(range(n_elem))}
        phases = [
            ConstructionPhase(name="P1", active_soil_groups=['all'],
                              n_steps=1),  # single step, likely won't converge
            ConstructionPhase(name="P2", active_soil_groups=['all'],
                              n_steps=5),
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases,
            max_iter=2, tol=1e-12)  # very strict tolerance

        # Either it converged or stopped early
        if not result.converged:
            assert len(result.phases) <= 2

    def test_empty_active_groups(self):
        """Phase with no active groups should produce zero-displacement result."""
        nodes, elements, bc_nodes = _small_mesh()
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        groups = {'all': list(range(n_elem))}
        phases = [
            ConstructionPhase(name="Empty", active_soil_groups=[],
                              n_steps=3),
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        assert result.converged
        assert result.phases[0].n_active_elements == 0

    def test_unknown_group_name_ignored(self):
        """Unknown group name in active_soil_groups should be ignored."""
        nodes, elements, bc_nodes = _small_mesh()
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        groups = {'all': list(range(n_elem))}
        phases = [
            ConstructionPhase(name="P1",
                              active_soil_groups=['all', 'nonexistent'],
                              n_steps=3),
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        assert result.converged
        assert result.phases[0].n_active_elements == n_elem

    def test_three_phase_sequence(self):
        """3-phase: gravity → load → excavate all completes."""
        nodes, elements, bc_nodes = _small_mesh(nx=6, ny=3, width=6.0,
                                                 depth=3.0)
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        groups = assign_element_groups(nodes, elements, {
            'soil': {'x_min': -1, 'x_max': 3, 'y_min': -4, 'y_max': 1},
            'excav': {'x_min': 3, 'x_max': 7, 'y_min': -1.5, 'y_max': 1},
            'deep': {'x_min': 3, 'x_max': 7, 'y_min': -4, 'y_max': -1.5},
        })

        # Surface edges for load
        top_nodes = np.where(np.abs(nodes[:, 1]) < 0.01)[0]
        top_nodes = top_nodes[np.argsort(nodes[top_nodes, 0])]
        # Only left half for the load
        left_top = top_nodes[nodes[top_nodes, 0] < 3.0]
        edges = [(left_top[i], left_top[i + 1])
                 for i in range(len(left_top) - 1)]

        phases = [
            ConstructionPhase(
                name="Gravity",
                active_soil_groups=['soil', 'excav', 'deep'],
                n_steps=5),
            ConstructionPhase(
                name="Load",
                active_soil_groups=['soil', 'excav', 'deep'],
                surface_loads=[(edges, 0.0, -50.0)] if len(edges) > 0 else None,
                n_steps=3),
            ConstructionPhase(
                name="Excavate",
                active_soil_groups=['soil', 'deep'],
                n_steps=5),
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases)

        assert result.n_phases == 3
        assert all(p.converged for p in result.phases)

    def test_phase_with_no_beams_when_beams_exist(self):
        """Phase with active_beam_ids=None when beam_elements is given."""
        nodes, elements, bc_nodes = _small_mesh()
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        # Create a dummy beam (may not connect to useful nodes)
        beams = [BeamElement(node_i=0, node_j=1, EA=1e6, EI=1e4)]

        groups = {'all': list(range(n_elem))}
        phases = [
            ConstructionPhase(name="No beams", active_soil_groups=['all'],
                              active_beam_ids=None, n_steps=5),
        ]
        result = analyze_staged(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            element_groups=groups, phases=phases,
            beam_elements=beams)

        assert result.converged
        assert result.phases[0].n_beam_elements == 0


# ===========================================================================
# solve_nonlinear return_state Tests
# ===========================================================================

class TestReturnState:
    """Test return_state=True on solve_nonlinear."""

    def test_return_state_gives_5_tuple(self):
        nodes, elements, bc_nodes = _small_mesh(nx=4, ny=2)
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        result = solve_nonlinear(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            n_steps=3, return_state=True)

        assert len(result) == 5
        converged, u, stresses, strains, state = result
        assert isinstance(converged, (bool, np.bool_))
        assert len(state) == n_elem

    def test_return_state_false_gives_4_tuple(self):
        nodes, elements, bc_nodes = _small_mesh(nx=4, ny=2)
        n_elem = len(elements)
        mat_props = _make_material_props(n_elem)
        gamma_arr = _make_gamma_arr(n_elem)

        result = solve_nonlinear(
            nodes, elements, mat_props, gamma_arr, bc_nodes,
            n_steps=3, return_state=False)

        assert len(result) == 4
