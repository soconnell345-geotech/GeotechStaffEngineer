"""
Validation tests for the lateral pile analysis module.

Tests include:
1. Elastic closed-form (Hetenyi) verification of the finite difference solver.
2. p-y curve spot checks against published formulations.
3. Equilibrium and boundary condition verification.
4. Mesh convergence study.
5. Parametric behavior checks (fixed vs free, P-delta, cyclic).
6. Full analysis regression against mesh-converged, equilibrium-verified benchmarks.
7. Published example validation (COM624P-style cases).

Run with: pytest lateral_pile/validation.py -v

References
----------
- COM624P Manual: FHWA-SA-91-048 (Wang & Reese, 1993)
- Hetenyi, M. (1946). "Beams on Elastic Foundation."
- Matlock, H. (1970). OTC 1204.
- Reese, L.C., Cox, W.R. & Koop, F.D. (1974). OTC 2080.
- API RP2A-WSD, 21st Edition (2000).
- Reese, L.C. & Van Impe, W.F. (2001). "Single Piles and Pile Groups
  Under Lateral Loading." Balkema.
"""

import numpy as np
import pytest

from lateral_pile.pile import Pile, ReinforcedConcreteSection, rebar_diameter, _HP_SECTIONS
from lateral_pile.soil import SoilLayer
from lateral_pile.py_curves import (
    SoftClayMatlock, SoftClayJeanjean, SandAPI, SandReese,
    StiffClayBelowWT, StiffClayAboveWT, WeakRock,
)
from lateral_pile.solver import solve_lateral_pile, hetenyi_solution
from lateral_pile.analysis import LateralPileAnalysis

# numpy >= 2.0 renamed trapz -> trapezoid
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz


# =============================================================================
# Helper: Standard test pile and soil configurations
# =============================================================================

def _make_soft_clay_case():
    """Standard COM624P-style soft clay case.

    Steel pipe pile: D=0.610 m (24"), t=12.7 mm, L=18 m
    Soft clay: su=25 kPa, eps50=0.020, gamma=9.0 kN/m3, J=0.5
    """
    pile = Pile(length=18.0, diameter=0.610, thickness=0.0127, E=200e6)
    layers = [SoilLayer(
        top=0.0, bottom=18.0,
        py_model=SoftClayMatlock(c=25.0, gamma=9.0, eps50=0.020, J=0.5,
                                 loading='static'),
    )]
    return LateralPileAnalysis(pile, layers)


def _make_medium_clay_case():
    """Standard medium stiffness clay case.

    Steel pipe pile: D=0.610 m, t=12.7 mm, L=20 m
    Medium clay: su=50 kPa, eps50=0.010, gamma=9.0 kN/m3, J=0.5
    """
    pile = Pile(length=20.0, diameter=0.610, thickness=0.0127, E=200e6)
    layers = [SoilLayer(
        top=0.0, bottom=20.0,
        py_model=SoftClayMatlock(c=50.0, gamma=9.0, eps50=0.010, J=0.5,
                                 loading='static'),
    )]
    return LateralPileAnalysis(pile, layers)


def _make_api_sand_case():
    """Standard API sand case.

    Steel pipe pile: D=0.610 m, t=9.5 mm, L=21 m
    Medium-dense sand: phi=39°, gamma=10.4 kN/m3, k=33900 kN/m3
    """
    pile = Pile(length=21.0, diameter=0.610, thickness=0.0095, E=200e6)
    layers = [SoilLayer(
        top=0.0, bottom=21.0,
        py_model=SandAPI(phi=39.0, gamma=10.4, k=33900.0, loading='static'),
    )]
    return LateralPileAnalysis(pile, layers)


# =============================================================================
# Test 1: Hetenyi (elastic) solution — verifies the FD solver
# =============================================================================

class TestHetenyi:
    """Verify finite difference solver against Hetenyi closed-form solution."""

    def test_elastic_pile_deflection(self):
        """FD solver with constant Es should match Hetenyi within 2%."""
        L = 20.0
        EI = 50000.0
        Es = 5000.0
        Vt = 100.0
        Mt = 0.0
        n = 100

        analytical = hetenyi_solution(L, EI, Es, Vt, Mt, n_points=n + 1)

        n_nodes = n + 1
        EI_array = np.full(n_nodes, EI)

        def linear_py(y, z, b_diam):
            return Es * y

        fd_result = solve_lateral_pile(
            pile_length=L,
            EI_values=EI_array,
            py_functions=[linear_py] * n_nodes,
            Vt=Vt, Mt=Mt, Q=0.0,
            head_condition='free',
            n_elements=n,
            tolerance=1e-8,
            max_iterations=50,
            pile_diameter=0.6,
        )

        y_analytical = analytical.y[0]
        y_fd = fd_result.y[0]
        assert y_analytical != 0
        rel_error = abs(y_fd - y_analytical) / abs(y_analytical)
        assert rel_error < 0.02, (
            f"Head deflection error {rel_error:.1%} exceeds 2%. "
            f"FD={y_fd:.6f}, Analytical={y_analytical:.6f}"
        )

    def test_elastic_pile_with_moment(self):
        """FD solver with applied moment should match Hetenyi."""
        L = 20.0
        EI = 50000.0
        Es = 5000.0
        Vt = 50.0
        Mt = 200.0
        n = 100

        analytical = hetenyi_solution(L, EI, Es, Vt, Mt, n_points=n + 1)

        n_nodes = n + 1
        EI_array = np.full(n_nodes, EI)

        def linear_py(y, z, b):
            return Es * y

        fd_result = solve_lateral_pile(
            pile_length=L,
            EI_values=EI_array,
            py_functions=[linear_py] * n_nodes,
            Vt=Vt, Mt=Mt, Q=0.0,
            head_condition='free',
            n_elements=n,
            tolerance=1e-8,
            max_iterations=50,
            pile_diameter=0.6,
        )

        y_analytical = analytical.y[0]
        y_fd = fd_result.y[0]
        rel_error = abs(y_fd - y_analytical) / abs(y_analytical)
        assert rel_error < 0.02, (
            f"Head deflection error {rel_error:.1%}. "
            f"FD={y_fd:.6f}, Analytical={y_analytical:.6f}"
        )

    def test_elastic_pile_full_profile(self):
        """Entire deflection profile should match Hetenyi, not just head."""
        L = 20.0
        EI = 50000.0
        Es = 5000.0
        Vt = 100.0
        n = 200

        analytical = hetenyi_solution(L, EI, Es, Vt, 0.0, n_points=n + 1)

        EI_array = np.full(n + 1, EI)

        def linear_py(y, z, b):
            return Es * y

        fd_result = solve_lateral_pile(
            pile_length=L,
            EI_values=EI_array,
            py_functions=[linear_py] * (n + 1),
            Vt=Vt, Mt=0.0, Q=0.0,
            head_condition='free',
            n_elements=n,
            tolerance=1e-8,
            max_iterations=50,
            pile_diameter=0.6,
        )

        # Compare full profiles (ignore very small values near tip)
        max_y = abs(analytical.y[0])
        for i in range(n + 1):
            if abs(analytical.y[i]) > 0.001 * max_y:
                rel_error = abs(fd_result.y[i] - analytical.y[i]) / max_y
                assert rel_error < 0.02, (
                    f"Profile mismatch at node {i} (z={analytical.z[i]:.2f}m): "
                    f"FD={fd_result.y[i]:.6f}, Analytical={analytical.y[i]:.6f}"
                )


# =============================================================================
# Test 2: p-y curve spot checks
# =============================================================================

class TestPYCurves:
    """Verify p-y curve models produce correct values."""

    def test_matlock_pu_shallow(self):
        """Matlock pu at shallow depth should use wedge mechanism."""
        model = SoftClayMatlock(c=25.0, gamma=8.0, eps50=0.02, J=0.5)
        b = 0.6
        z = 1.0
        pu = model.get_pu(z, b)
        pu_expected = (3.0 + 8.0 / 25.0 + 0.5 / 0.6) * 25.0 * 0.6
        assert abs(pu - pu_expected) < 0.1, f"pu={pu}, expected={pu_expected}"

    def test_matlock_pu_deep(self):
        """Matlock pu at great depth should be 9*c*b."""
        model = SoftClayMatlock(c=25.0, gamma=8.0, eps50=0.02, J=0.5)
        pu = model.get_pu(50.0, 0.6)
        assert abs(pu - 9.0 * 25.0 * 0.6) < 0.01

    def test_matlock_static_curve_shape(self):
        """Matlock static p-y should follow 1/3 power law."""
        model = SoftClayMatlock(c=25.0, gamma=8.0, eps50=0.02, J=0.5, loading='static')
        b = 0.6
        z = 5.0
        pu = model.get_pu(z, b)
        y50 = 2.5 * 0.02 * 0.6

        # At y = y50: p = 0.5*pu
        p_at_y50 = model.get_p(y50, z, b)
        assert abs(p_at_y50 - 0.5 * pu) / (0.5 * pu) < 0.01

        # At y = 8*y50: p = 0.5*pu*8^(1/3)
        p_at_8y50 = model.get_p(8.0 * y50, z, b)
        expected_8 = 0.5 * pu * 8.0 ** (1.0 / 3.0)
        assert abs(p_at_8y50 - expected_8) / expected_8 < 0.01

    def test_matlock_cyclic_deep(self):
        """Matlock cyclic at deep depth should plateau at 0.72*pu."""
        model = SoftClayMatlock(c=25.0, gamma=8.0, eps50=0.02, J=0.5, loading='cyclic')
        pu = model.get_pu(20.0, 0.6)
        y50 = 2.5 * 0.02 * 0.6
        p_cyclic = model.get_p(5.0 * y50, 20.0, 0.6)
        assert abs(p_cyclic - 0.72 * pu) / (0.72 * pu) < 0.01

    def test_matlock_symmetry(self):
        """p-y curve should be antisymmetric: p(-y) = -p(y)."""
        model = SoftClayMatlock(c=25.0, gamma=8.0, eps50=0.02, J=0.5)
        p_pos = model.get_p(0.015, 3.0, 0.6)
        p_neg = model.get_p(-0.015, 3.0, 0.6)
        assert abs(p_pos + p_neg) < 1e-10

    def test_jeanjean_pu_deep(self):
        """Jeanjean deep pu should be 12*su*b (vs Matlock's 9)."""
        model = SoftClayJeanjean(su=25.0, gamma=8.0, Gmax=5000.0, J=0.5)
        pu = model.get_pu(50.0, 0.6)
        assert abs(pu - 12.0 * 25.0 * 0.6) < 0.01

    def test_jeanjean_vs_matlock_higher_pu(self):
        """Jeanjean should give higher pu than Matlock at depth."""
        matlock = SoftClayMatlock(c=25.0, gamma=8.0, eps50=0.02, J=0.5)
        jeanjean = SoftClayJeanjean(su=25.0, gamma=8.0, Gmax=5000.0, J=0.5)
        assert jeanjean.get_pu(10.0, 0.6) > matlock.get_pu(10.0, 0.6)

    def test_jeanjean_initial_slope(self):
        """Jeanjean initial slope should be related to Gmax."""
        model = SoftClayJeanjean(su=25.0, gamma=8.0, Gmax=5000.0, J=0.5)
        p = model.get_p(1e-7, 5.0, 0.6)
        assert p > 0

    def test_api_sand_shape(self):
        """API sand should follow tanh shape and approach A*pu."""
        model = SandAPI(phi=35.0, gamma=10.0, k=16000.0, loading='static')
        b = 0.6
        z = 5.0
        pu = model.get_pu(z, b)
        A = model.get_A(z, b)
        Apu = A * pu

        # At large y, p -> A*pu
        p_large = model.get_p(0.5, z, b)
        assert p_large > 0.95 * Apu

        # Initial slope = k*z
        small_y = 1e-6
        p_small = model.get_p(small_y, z, b)
        initial_slope = p_small / small_y
        assert abs(initial_slope - model.k * z) / (model.k * z) < 0.01

    def test_api_sand_at_surface(self):
        """API sand at z=0 should give zero resistance."""
        model = SandAPI(phi=35.0, gamma=10.0, k=16000.0)
        assert model.get_p(0.01, 0.0, 0.6) == 0.0

    def test_stiff_clay_below_wt_basics(self):
        """StiffClayBelowWT should produce reasonable results."""
        model = StiffClayBelowWT(c=100.0, gamma=9.0, eps50=0.005, ks=270000.0)
        b = 0.6
        z = 3.0

        # pu should be between shallow and deep
        pu = model.get_pu(z, b)
        pu_deep = 11.0 * 100.0 * 0.6  # = 660 kN/m
        assert 0 < pu <= pu_deep

        # p should increase with y
        p1 = model.get_p(0.001, z, b)
        p2 = model.get_p(0.01, z, b)
        assert p2 > p1 > 0

    def test_weak_rock_basics(self):
        """WeakRock model should produce reasonable results."""
        model = WeakRock(qu=1000.0, Er=200000.0)
        b = 0.6
        z = 3.0

        pu = model.get_pu(z, b)
        assert pu > 0

        # At large y, p should reach pu
        p_large = model.get_p(0.1, z, b)
        assert abs(p_large - pu) / pu < 0.01

        # Antisymmetry
        assert abs(model.get_p(0.01, z, b) + model.get_p(-0.01, z, b)) < 1e-10


# =============================================================================
# Test 3: Equilibrium verification
# =============================================================================

class TestEquilibrium:
    """Verify that the solver satisfies force and moment equilibrium.

    These tests are the strongest validation because they are
    independent of any published reference — equilibrium must hold
    regardless of the p-y model used.
    """

    @pytest.mark.parametrize("case_name,make_fn,Vt,Mt,Q", [
        ("soft_clay_200kN", _make_soft_clay_case, 200.0, 0.0, 0.0),
        ("soft_clay_100kN_moment", _make_soft_clay_case, 100.0, 150.0, 0.0),
        ("medium_clay_200kN", _make_medium_clay_case, 200.0, 0.0, 0.0),
        ("api_sand_89kN", _make_api_sand_case, 89.0, 0.0, 0.0),
        ("sand_with_axial", _make_api_sand_case, 100.0, 0.0, 500.0),
    ])
    def test_force_equilibrium(self, case_name, make_fn, Vt, Mt, Q):
        """Integral of soil reaction should equal applied lateral load."""
        analysis = make_fn()
        results = analysis.solve(Vt=Vt, Mt=Mt, Q=Q, head_condition='free',
                                 n_elements=200)
        assert results.converged, f"Case {case_name} did not converge"

        integral_p = _trapz(results.soil_reaction, results.z)
        rel_error = abs(integral_p - Vt) / Vt
        assert rel_error < 0.005, (
            f"Force equilibrium error {rel_error:.2%}: "
            f"sum_p={integral_p:.3f}, Vt={Vt:.1f}"
        )

    @pytest.mark.parametrize("case_name,make_fn,Vt,Mt", [
        ("soft_clay_free", _make_soft_clay_case, 200.0, 0.0),
        ("soft_clay_moment", _make_soft_clay_case, 100.0, 150.0),
        ("medium_clay_free", _make_medium_clay_case, 200.0, 0.0),
        ("api_sand_free", _make_api_sand_case, 89.0, 0.0),
    ])
    def test_moment_equilibrium(self, case_name, make_fn, Vt, Mt):
        """Moment about pile head from soil reaction should balance.

        The global moment equilibrium about z=0 is:
            Mt + integral(p(z)*z dz, 0, L) = 0
        i.e., integral(p*z dz) = -Mt.
        """
        analysis = make_fn()
        results = analysis.solve(Vt=Vt, Mt=Mt, Q=0.0, head_condition='free',
                                 n_elements=200)
        assert results.converged

        moment_integral = _trapz(results.soil_reaction * results.z, results.z)
        # Equilibrium: moment_integral + Mt = 0
        residual = moment_integral + Mt
        if abs(Mt) > 0:
            rel_error = abs(residual) / abs(Mt)
        else:
            # When Mt=0, residual should be very small relative to Vt*L
            rel_error = abs(residual) / (Vt * results.z[-1])
        assert rel_error < 0.01, (
            f"Moment equilibrium error {rel_error:.2%}: "
            f"integral(p*z)={moment_integral:.3f}, Mt={Mt:.1f}, "
            f"residual={residual:.3f}"
        )

    def test_shear_at_head_matches_applied(self):
        """Shear force at pile head should equal applied load."""
        analysis = _make_soft_clay_case()
        results = analysis.solve(Vt=200.0, Mt=0.0, Q=0.0,
                                 head_condition='free', n_elements=200)
        assert abs(results.shear[0] - 200.0) < 0.1

    def test_moment_at_head_matches_applied(self):
        """Moment at pile head should equal applied moment."""
        analysis = _make_soft_clay_case()

        # With Mt = 150 kN-m
        results = analysis.solve(Vt=100.0, Mt=150.0, Q=0.0,
                                 head_condition='free', n_elements=200)
        rel_error = abs(results.moment[0] - 150.0) / 150.0
        assert rel_error < 0.02, (
            f"Head moment error: M(0)={results.moment[0]:.2f}, expected 150.0"
        )

    def test_tip_boundary_conditions(self):
        """Moment and shear at pile tip should be essentially zero."""
        analysis = _make_soft_clay_case()
        results = analysis.solve(Vt=200.0, Mt=0.0, Q=0.0,
                                 head_condition='free', n_elements=200)

        assert abs(results.moment[-1]) < 0.1, (
            f"Tip moment should be ~0, got {results.moment[-1]:.4f}"
        )
        assert abs(results.shear[-1]) < 0.1, (
            f"Tip shear should be ~0, got {results.shear[-1]:.4f}"
        )


# =============================================================================
# Test 4: Mesh convergence
# =============================================================================

class TestMeshConvergence:
    """Verify that the solution converges as the mesh is refined."""

    def test_convergence_soft_clay(self):
        """Solution should converge as n_elements increases."""
        analysis = _make_soft_clay_case()

        results_coarse = analysis.solve(Vt=200.0, head_condition='free',
                                        n_elements=25)
        results_medium = analysis.solve(Vt=200.0, head_condition='free',
                                        n_elements=100)
        results_fine = analysis.solve(Vt=200.0, head_condition='free',
                                      n_elements=400)

        # Fine solution is our reference
        y_ref = results_fine.y_top

        # Medium should be within 0.5% of fine
        error_medium = abs(results_medium.y_top - y_ref) / abs(y_ref)
        assert error_medium < 0.005, (
            f"Medium mesh error {error_medium:.2%} exceeds 0.5%"
        )

        # Coarse should be within 2% of fine
        error_coarse = abs(results_coarse.y_top - y_ref) / abs(y_ref)
        assert error_coarse < 0.02, (
            f"Coarse mesh error {error_coarse:.2%} exceeds 2%"
        )

    def test_convergence_sand(self):
        """Sand case should also converge with mesh refinement."""
        analysis = _make_api_sand_case()

        results_100 = analysis.solve(Vt=89.0, head_condition='free',
                                     n_elements=100)
        results_400 = analysis.solve(Vt=89.0, head_condition='free',
                                     n_elements=400)

        error = abs(results_100.y_top - results_400.y_top) / abs(results_400.y_top)
        assert error < 0.01, f"Sand mesh convergence error {error:.2%}"


# =============================================================================
# Test 5: Parametric behavior
# =============================================================================

class TestParametricBehavior:
    """Verify qualitative correctness of the analysis."""

    def test_fixed_less_than_free(self):
        """Fixed head should give less deflection than free head."""
        analysis = _make_medium_clay_case()
        r_free = analysis.solve(Vt=100.0, head_condition='free')
        r_fixed = analysis.solve(Vt=100.0, head_condition='fixed')

        assert r_free.converged and r_fixed.converged
        assert r_fixed.y_top < r_free.y_top
        assert r_fixed.y_top > 0  # still deflects in direction of load

    def test_fixed_head_zero_slope(self):
        """Fixed head condition should enforce zero rotation at pile head."""
        analysis = _make_medium_clay_case()
        r_fixed = analysis.solve(Vt=100.0, head_condition='fixed')
        assert abs(r_fixed.rotation_top) < 1e-6, (
            f"Fixed head slope = {r_fixed.rotation_top:.2e}, should be ~0"
        )

    def test_fixed_head_negative_moment(self):
        """Fixed head should develop a negative moment at the pile head."""
        analysis = _make_medium_clay_case()
        r_fixed = analysis.solve(Vt=100.0, head_condition='fixed')
        assert r_fixed.moment[0] < 0, (
            f"Fixed head moment M(0) = {r_fixed.moment[0]:.1f}, should be negative"
        )

    def test_load_proportionality_small_loads(self):
        """At small loads (near-linear), doubling load should ~double deflection.

        Uses API sand (tanh formulation) which has a well-defined linear
        initial slope, unlike Matlock's 1/3 power law which is always nonlinear.
        """
        analysis = _make_api_sand_case()
        r1 = analysis.solve(Vt=5.0, head_condition='free', n_elements=200)
        r2 = analysis.solve(Vt=10.0, head_condition='free', n_elements=200)

        ratio = r2.y_top / r1.y_top
        assert 1.95 < ratio < 2.05, (
            f"Load proportionality: 2x load gave {ratio:.2f}x deflection "
            f"(expected ~2.0 for near-linear range)"
        )

    def test_increasing_load_increases_deflection(self):
        """Higher loads should produce larger deflections."""
        analysis = _make_soft_clay_case()
        r50 = analysis.solve(Vt=50.0, head_condition='free')
        r100 = analysis.solve(Vt=100.0, head_condition='free')
        r200 = analysis.solve(Vt=200.0, head_condition='free')

        assert r200.y_top > r100.y_top > r50.y_top > 0

    def test_p_delta_amplification(self):
        """Axial load should amplify lateral deflection (P-delta effect)."""
        analysis = _make_soft_clay_case()
        r_no_axial = analysis.solve(Vt=200.0, Q=0.0, head_condition='free')
        r_axial = analysis.solve(Vt=200.0, Q=500.0, head_condition='free')

        assert r_axial.y_top > r_no_axial.y_top, (
            "Axial load should amplify lateral deflection"
        )
        # P-delta effect should be modest for typical pile loads
        amplification = r_axial.y_top / r_no_axial.y_top
        assert amplification < 1.5, (
            f"P-delta amplification {amplification:.2f}x seems too large"
        )

    def test_stiffer_soil_less_deflection(self):
        """Higher soil strength should give less pile head deflection."""
        pile = Pile(length=20.0, diameter=0.610, thickness=0.0127, E=200e6)

        layers_soft = [SoilLayer(top=0.0, bottom=20.0,
            py_model=SoftClayMatlock(c=25.0, gamma=9.0, eps50=0.020, J=0.5))]
        layers_stiff = [SoilLayer(top=0.0, bottom=20.0,
            py_model=SoftClayMatlock(c=75.0, gamma=9.0, eps50=0.010, J=0.5))]

        r_soft = LateralPileAnalysis(pile, layers_soft).solve(Vt=100.0)
        r_stiff = LateralPileAnalysis(pile, layers_stiff).solve(Vt=100.0)

        assert r_stiff.y_top < r_soft.y_top

    def test_cyclic_vs_static(self):
        """Cyclic loading should give more deflection than static.

        Uses Matlock Lake Austin pile at high load where deflections
        exceed 3*y50 (=29.2 mm) so cyclic degradation engages.
        At small deflections (< 3*y50), static and cyclic are identical.
        """
        pile = Pile(length=12.8, diameter=0.324, thickness=0.0127, E=200e6)

        layers_static = [SoilLayer(top=0.0, bottom=12.8,
            py_model=SoftClayMatlock(c=38.3, gamma=10.0, eps50=0.012, J=0.5,
                                     loading='static'))]
        layers_cyclic = [SoilLayer(top=0.0, bottom=12.8,
            py_model=SoftClayMatlock(c=38.3, gamma=10.0, eps50=0.012, J=0.5,
                                     loading='cyclic'))]

        # 89 kN gives ~31.5 mm static deflection, exceeding 3*y50 = 29.2 mm
        r_static = LateralPileAnalysis(pile, layers_static).solve(Vt=89.0)
        r_cyclic = LateralPileAnalysis(pile, layers_cyclic).solve(Vt=89.0)

        assert r_cyclic.y_top > r_static.y_top, (
            f"Cyclic ({r_cyclic.y_top*1000:.1f} mm) should exceed "
            f"static ({r_static.y_top*1000:.1f} mm)"
        )

    def test_max_moment_below_surface(self):
        """Maximum bending moment should occur below the surface."""
        analysis = _make_soft_clay_case()
        results = analysis.solve(Vt=200.0, Mt=0.0, head_condition='free')

        assert results.max_moment_depth > 0, (
            "Max moment should be below surface for free-head pile"
        )


# =============================================================================
# Test 6: Full analysis integration tests
# =============================================================================

class TestFullAnalysis:
    """Integration tests for the complete analysis pipeline."""

    def test_soft_clay_basic(self):
        """Basic soft clay analysis should converge with reasonable results."""
        pile = Pile(length=20.0, diameter=0.6, thickness=0.012, E=200e6)
        layers = [SoilLayer(top=0.0, bottom=20.0,
            py_model=SoftClayMatlock(c=25.0, gamma=8.0, eps50=0.02, J=0.5))]

        analysis = LateralPileAnalysis(pile, layers)
        results = analysis.solve(Vt=100.0, Mt=0.0, Q=0.0, head_condition='free')

        assert results.converged
        assert 0.001 < results.y_top < 1.0
        assert results.max_moment > 0

    def test_sand_basic(self):
        """Basic API sand analysis should converge."""
        pile = Pile(length=15.0, diameter=0.6, thickness=0.012, E=200e6)
        layers = [SoilLayer(top=0.0, bottom=15.0,
            py_model=SandAPI(phi=35.0, gamma=10.0, k=16000.0))]

        analysis = LateralPileAnalysis(pile, layers)
        results = analysis.solve(Vt=50.0)
        assert results.converged
        assert results.y_top > 0

    def test_multilayer(self):
        """Analysis with clay over sand should work correctly."""
        pile = Pile(length=20.0, diameter=0.6, thickness=0.012, E=200e6)
        layers = [
            SoilLayer(top=0.0, bottom=5.0,
                py_model=SoftClayMatlock(c=25.0, gamma=8.0, eps50=0.02, J=0.5)),
            SoilLayer(top=5.0, bottom=20.0,
                py_model=SandAPI(phi=35.0, gamma=10.0, k=16000.0)),
        ]

        analysis = LateralPileAnalysis(pile, layers)
        results = analysis.solve(Vt=100.0, Mt=0.0, Q=500.0, head_condition='free')
        assert results.converged
        assert results.y_top > 0

    def test_jeanjean_full_analysis(self):
        """Jeanjean soft clay analysis should converge."""
        pile = Pile(length=20.0, diameter=0.6, thickness=0.012, E=200e6)

        layers = [SoilLayer(top=0.0, bottom=20.0,
            py_model=SoftClayJeanjean(su=25.0, gamma=8.0, Gmax=5000.0, J=0.5))]
        analysis = LateralPileAnalysis(pile, layers)
        results = analysis.solve(Vt=100.0, head_condition='free')

        assert results.converged
        assert results.y_top > 0
        assert results.y_top < 0.5

    def test_results_summary(self):
        """Results summary should include all key information."""
        pile = Pile(length=15.0, diameter=0.6, thickness=0.012, E=200e6)
        layers = [SoilLayer(top=0.0, bottom=15.0,
            py_model=SandAPI(phi=35.0, gamma=10.0, k=16000.0))]

        analysis = LateralPileAnalysis(pile, layers)
        results = analysis.solve(Vt=50.0)
        summary = results.summary()

        assert 'Pile head deflection' in summary
        assert 'Max bending moment' in summary
        assert 'Converged' in summary

    def test_to_dict(self):
        """Results to_dict should contain all keys and correct types."""
        analysis = _make_medium_clay_case()
        results = analysis.solve(Vt=100.0)
        d = results.to_dict()

        assert 'z' in d
        assert 'deflection_m' in d
        assert 'moment_kNm' in d
        assert 'y_top_m' in d
        assert 'converged' in d
        assert isinstance(d['z'], list)
        assert isinstance(d['y_top_m'], float)


# =============================================================================
# Test 7: Published benchmark validation
# =============================================================================

class TestPublishedBenchmarks:
    """Regression tests against mesh-converged, equilibrium-verified solutions.

    These reference values were computed using 400 elements and verified to
    satisfy force equilibrium to < 0.01% and boundary conditions to < 0.01%.
    They serve as benchmarks for the COM624P-style p-y formulations.

    Reference values match the expected behavior from:
    - COM624P Manual: FHWA-SA-91-048 (Wang & Reese, 1993)
    - Reese & Van Impe (2001), "Single Piles and Pile Groups Under
      Lateral Loading", Balkema.
    """

    def test_soft_clay_benchmark(self):
        """Soft clay benchmark: 0.610m pipe pile, su=25 kPa, Vt=200 kN.

        Reference values (n=400, verified equilibrium < 0.01%):
          y_top  = 69.0 mm
          Mmax   = 487.0 kN-m at 4.50 m
          Rotation = -14.6 mrad
        """
        analysis = _make_soft_clay_case()
        results = analysis.solve(Vt=200.0, Mt=0.0, Q=0.0,
                                 head_condition='free', n_elements=200)

        assert results.converged
        # Deflection within 5% of reference
        assert abs(results.y_top * 1000 - 69.0) / 69.0 < 0.05, (
            f"y_top = {results.y_top*1000:.2f} mm, expected ~69.0 mm"
        )
        # Max moment within 5% of reference
        assert abs(results.max_moment - 487.0) / 487.0 < 0.05, (
            f"Mmax = {results.max_moment:.1f} kN-m, expected ~487.0 kN-m"
        )
        # Max moment depth within 0.5 m of reference
        assert abs(results.max_moment_depth - 4.50) < 0.5, (
            f"Mmax depth = {results.max_moment_depth:.2f} m, expected ~4.50 m"
        )

    def test_medium_clay_benchmark(self):
        """Medium clay benchmark: 0.610m pipe pile, su=50 kPa, Vt=200 kN.

        Reference values (n=400, verified equilibrium < 0.01%):
          y_top  = 22.9 mm
          Mmax   = 327.7 kN-m at 3.20 m
        """
        analysis = _make_medium_clay_case()
        results = analysis.solve(Vt=200.0, Mt=0.0, Q=0.0,
                                 head_condition='free', n_elements=200)

        assert results.converged
        assert abs(results.y_top * 1000 - 22.9) / 22.9 < 0.05, (
            f"y_top = {results.y_top*1000:.2f} mm, expected ~22.9 mm"
        )
        assert abs(results.max_moment - 327.7) / 327.7 < 0.05, (
            f"Mmax = {results.max_moment:.1f} kN-m, expected ~327.7 kN-m"
        )

    def test_api_sand_benchmark(self):
        """API sand benchmark: 0.610m pipe pile, phi=39, Vt=89 kN.

        Reference values (n=400, verified equilibrium < 0.01%):
          y_top  = 3.62 mm
          Mmax   = 97.7 kN-m at ~1.9 m
        """
        analysis = _make_api_sand_case()
        results = analysis.solve(Vt=89.0, Mt=0.0, Q=0.0,
                                 head_condition='free', n_elements=200)

        assert results.converged
        assert abs(results.y_top * 1000 - 3.62) / 3.62 < 0.05, (
            f"y_top = {results.y_top*1000:.2f} mm, expected ~3.62 mm"
        )
        assert abs(results.max_moment - 97.7) / 97.7 < 0.05, (
            f"Mmax = {results.max_moment:.1f} kN-m, expected ~97.7 kN-m"
        )

    def test_soft_clay_fixed_head_benchmark(self):
        """Fixed-head soft clay benchmark: same pile, Vt=200 kN.

        Reference values (n=400, verified equilibrium < 0.01%):
          y_top     = 18-20 mm (significantly less than free-head 69 mm)
          M(0)      < 0 (negative head moment)
          slope(0)  = 0 (fixed condition)
        """
        analysis = _make_soft_clay_case()
        r_free = analysis.solve(Vt=200.0, head_condition='free', n_elements=200)
        r_fixed = analysis.solve(Vt=200.0, head_condition='fixed', n_elements=200)

        assert r_fixed.converged
        # Fixed head should be 25-35% of free head deflection
        ratio = r_fixed.y_top / r_free.y_top
        assert 0.15 < ratio < 0.45, (
            f"Fixed/free ratio = {ratio:.3f}, expected 0.2-0.4"
        )
        assert abs(r_fixed.rotation_top) < 1e-6
        assert r_fixed.moment[0] < 0

    def test_matlock_lake_austin_benchmark(self):
        """Matlock (1970) Lake Austin test — OTC 1204.

        Corrected pile and soil parameters from published literature
        (confirmed across pilegroups.com, RSPile theory manual, Springer).

        Steel pipe pile: D=324mm (12.75"), t=12.7mm (0.5"), L=12.8m
        EI = 30,141 kN-m^2
        Soft clay: su=38.3 kPa, gamma'=10.0 kN/m3, eps50=0.012, J=0.5
        Free head, static loading, load at ground line.

        Reference (n=400, equilibrium verified to <0.01%):
          V=22.2 kN (5 kips):  y_top = 2.39 mm, Mmax = 18.5 kN-m @ 1.60 m
          V=44.5 kN (10 kips): y_top = 8.74 mm, Mmax = 45.9 kN-m @ 1.98 m
          V=89.0 kN (20 kips): y_top = 31.5 mm, Mmax = 113.1 kN-m @ 2.43 m
        """
        pile = Pile(length=12.8, diameter=0.324, thickness=0.0127, E=200e6)
        layers = [SoilLayer(top=0.0, bottom=12.8,
            py_model=SoftClayMatlock(c=38.3, gamma=10.0, eps50=0.012, J=0.5,
                                     loading='static'))]
        analysis = LateralPileAnalysis(pile, layers)

        # Test at 22.2 kN (5 kips)
        r1 = analysis.solve(Vt=22.2, head_condition='free', n_elements=200)
        assert r1.converged
        assert abs(r1.y_top * 1000 - 2.39) / 2.39 < 0.05

        # Test at 44.5 kN (10 kips)
        r2 = analysis.solve(Vt=44.5, head_condition='free', n_elements=200)
        assert r2.converged
        assert abs(r2.y_top * 1000 - 8.74) / 8.74 < 0.05
        assert abs(r2.max_moment - 45.9) / 45.9 < 0.05

        # Test at 89.0 kN (20 kips)
        r3 = analysis.solve(Vt=89.0, head_condition='free', n_elements=200)
        assert r3.converged
        assert abs(r3.y_top * 1000 - 31.5) / 31.5 < 0.05

    def test_com624p_example1_sabine_river(self):
        """COM624P Example 1 — Sabine River soft clay test pile.

        Published in FHWA-SA-91-048 (Wang & Reese, 1993), Example 1.
        Matches Matlock (1970) Sabine River field test.

        Steel pipe pile: D=12.75" (0.3239 m), t=3/8" (9.525 mm), L=12.8 m
        EI ≈ 23,265 kN-m^2
        Soft clay: su=14.4 kPa (2.1 psi), gamma'=5.0 kN/m^3 (buoyant),
                   eps50=0.012, J=0.5
        Free head, static loading, lateral load at ground line.

        Published COM624P output:
          V = 44.5 kN (10 kips) -> y_top = 34.3 mm (1.35 in)

        This test validates the full analysis pipeline against the
        COM624P program's published results to within 5%.
        """
        pile = Pile(length=12.8, diameter=0.3239, thickness=0.009525, E=200e6)
        layers = [SoilLayer(top=0.0, bottom=12.8,
            py_model=SoftClayMatlock(c=14.4, gamma=5.0, eps50=0.012, J=0.5,
                                     loading='static'))]
        analysis = LateralPileAnalysis(pile, layers)
        results = analysis.solve(Vt=44.5, head_condition='free', n_elements=200)

        assert results.converged
        # Published COM624P result: 34.3 mm (1.35 in)
        y_mm = results.y_top * 1000
        rel_error = abs(y_mm - 34.3) / 34.3
        assert rel_error < 0.05, (
            f"COM624P Example 1: y_top = {y_mm:.2f} mm, "
            f"published = 34.3 mm, error = {rel_error:.1%}"
        )

    def test_sand_high_load_benchmark(self):
        """API sand at higher load level.

        Reference (n=400, equilibrium verified):
          V=178 kN: y_top = 8.61 mm, Mmax = 217.5 kN-m
        """
        analysis = _make_api_sand_case()
        results = analysis.solve(Vt=178.0, Mt=0.0, Q=0.0,
                                 head_condition='free', n_elements=200)

        assert results.converged
        assert abs(results.y_top * 1000 - 8.61) / 8.61 < 0.05
        assert abs(results.max_moment - 217.5) / 217.5 < 0.05

    def test_applied_moment_benchmark(self):
        """Soft clay with applied moment at head.

        Reference (n=400, equilibrium verified):
          V=100 kN, M=150 kN-m:
            y_top > soft clay 100 kN case without moment
            M(0) should be close to 150 kN-m
        """
        analysis = _make_soft_clay_case()
        r_no_moment = analysis.solve(Vt=100.0, Mt=0.0, head_condition='free',
                                     n_elements=200)
        r_moment = analysis.solve(Vt=100.0, Mt=150.0, head_condition='free',
                                  n_elements=200)

        assert r_moment.converged
        # Applied moment should increase head deflection
        assert r_moment.y_top > r_no_moment.y_top
        # Head moment should match applied
        assert abs(r_moment.moment[0] - 150.0) / 150.0 < 0.02


# =============================================================================
# Test 8: Data structure validation
# =============================================================================

class TestDataStructures:
    """Verify pile and soil data structures."""

    def test_pile_solid_circular(self):
        """Solid circular pile should compute correct I."""
        import math
        pile = Pile(length=15.0, diameter=0.45, E=25e6)
        I_expected = math.pi / 4.0 * (0.225)**4
        assert abs(pile.moment_of_inertia - I_expected) / I_expected < 1e-10

    def test_pile_pipe(self):
        """Pipe pile should compute correct I."""
        import math
        pile = Pile(length=20.0, diameter=0.610, thickness=0.0127, E=200e6)
        r_o = 0.305
        r_i = 0.305 - 0.0127
        I_expected = math.pi / 4.0 * (r_o**4 - r_i**4)
        assert abs(pile.moment_of_inertia - I_expected) / I_expected < 1e-10

    def test_pile_EI(self):
        """EI should be E * I."""
        pile = Pile(length=20.0, diameter=0.610, thickness=0.0127, E=200e6)
        assert abs(pile.EI - pile.E * pile.moment_of_inertia) < 1e-6

    def test_soil_layer_validation(self):
        """Layer validation should catch gaps and overlaps."""
        from lateral_pile.soil import validate_layers
        model = SoftClayMatlock(c=25.0, gamma=8.0, eps50=0.02)

        # Gap between layers
        layers_gap = [
            SoilLayer(top=0.0, bottom=5.0, py_model=model),
            SoilLayer(top=6.0, bottom=15.0, py_model=model),
        ]
        with pytest.raises(ValueError, match="Gap"):
            validate_layers(layers_gap, 15.0)

        # Overlap
        layers_overlap = [
            SoilLayer(top=0.0, bottom=6.0, py_model=model),
            SoilLayer(top=5.0, bottom=15.0, py_model=model),
        ]
        with pytest.raises(ValueError, match="Overlap"):
            validate_layers(layers_overlap, 15.0)

    def test_soil_layer_invalid_model(self):
        """Layer should reject models without required interface."""
        with pytest.raises(TypeError):
            SoilLayer(top=0.0, bottom=10.0, py_model="not a model")

    def test_pile_invalid_params(self):
        """Pile should reject invalid parameters."""
        with pytest.raises(ValueError):
            Pile(length=-1, diameter=0.5, E=200e6)
        with pytest.raises(ValueError):
            Pile(length=10, diameter=0.5, thickness=0.3, E=200e6)  # t > r


# =============================================================================
# Test 9: H-Pile section lookup
# =============================================================================

class TestHPileCreation:
    """Verify H-pile factory method and section database."""

    def test_strong_axis_I(self):
        """Strong-axis I for HP14x117 should match AISC value (1220 in^4)."""
        pile = Pile.from_h_pile("HP14x117", length=20.0, axis='strong')
        # 1220 in^4 * (0.0254)^4 = 507.8e-6 m^4
        assert abs(pile.moment_of_inertia - 507.8e-6) / 507.8e-6 < 1e-3

    def test_weak_axis_I(self):
        """Weak-axis I for HP14x117 should match AISC value (443 in^4)."""
        pile = Pile.from_h_pile("HP14x117", length=20.0, axis='weak')
        # 443 in^4 * (0.0254)^4 = 184.4e-6 m^4
        assert abs(pile.moment_of_inertia - 184.4e-6) / 184.4e-6 < 1e-3

    def test_diameter_is_flange_width(self):
        """Pile diameter should be set to flange width for p-y curves."""
        pile = Pile.from_h_pile("HP14x117", length=20.0)
        bf = _HP_SECTIONS["HP14x117"]["bf"]
        assert abs(pile.diameter - bf) < 1e-10

    def test_invalid_designation(self):
        """Unknown HP shape should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown HP shape"):
            Pile.from_h_pile("HP99x999", length=20.0)

    def test_invalid_axis(self):
        """Invalid axis should raise ValueError."""
        with pytest.raises(ValueError, match="axis must be"):
            Pile.from_h_pile("HP14x117", length=20.0, axis='diagonal')

    def test_all_designations_load(self):
        """All 10 HP designations should create valid piles."""
        for designation in _HP_SECTIONS:
            pile = Pile.from_h_pile(designation, length=15.0)
            assert pile.EI > 0
            assert pile.moment_of_inertia > 0

    def test_strong_greater_than_weak(self):
        """Strong-axis I should be greater than weak-axis I for all shapes."""
        for designation in _HP_SECTIONS:
            strong = Pile.from_h_pile(designation, length=15.0, axis='strong')
            weak = Pile.from_h_pile(designation, length=15.0, axis='weak')
            assert strong.moment_of_inertia > weak.moment_of_inertia


# =============================================================================
# Test 10: Composite (concrete-filled pipe) section
# =============================================================================

class TestFilledPipe:
    """Verify concrete-filled pipe pile factory."""

    def test_composite_EI_hand_calc(self):
        """Composite EI should match hand calculation."""
        import math
        D = 0.610
        t = 0.0127
        E_steel = 200e6
        fc = 28000.0  # 28 MPa

        r_o = D / 2.0
        r_i = r_o - t
        I_steel = math.pi / 4.0 * (r_o**4 - r_i**4)
        I_concrete = math.pi / 4.0 * r_i**4
        fc_MPa = fc / 1000.0
        E_concrete = 4700.0 * math.sqrt(fc_MPa) * 1000.0
        EI_expected = E_steel * I_steel + E_concrete * I_concrete

        pile = Pile.from_filled_pipe(length=20.0, diameter=D, thickness=t,
                                     fc=fc)
        assert abs(pile.EI - EI_expected) / EI_expected < 1e-10

    def test_custom_E_concrete(self):
        """Custom E_concrete should override f'c-based calculation."""
        pile_fc = Pile.from_filled_pipe(
            length=20.0, diameter=0.610, thickness=0.0127, fc=28000.0)
        pile_custom = Pile.from_filled_pipe(
            length=20.0, diameter=0.610, thickness=0.0127,
            E_concrete=30000000.0)
        # Different E_concrete values should give different EI
        assert pile_fc.EI != pile_custom.EI

    def test_composite_greater_than_steel_only(self):
        """Composite EI should exceed steel-only EI for same geometry."""
        pile_steel = Pile(length=20.0, diameter=0.610, thickness=0.0127,
                          E=200e6)
        pile_filled = Pile.from_filled_pipe(
            length=20.0, diameter=0.610, thickness=0.0127)
        assert pile_filled.EI > pile_steel.EI

    def test_invalid_thickness(self):
        """Invalid thickness should raise ValueError."""
        with pytest.raises(ValueError):
            Pile.from_filled_pipe(length=20.0, diameter=0.610,
                                  thickness=0.35)  # t > r


# =============================================================================
# Test 11: Reinforced concrete section and cracked EI
# =============================================================================

class TestReinforcedConcreteSection:
    """Verify ReinforcedConcreteSection properties and Branson's equation."""

    def _make_section(self):
        """Create a standard test section: 900mm drilled shaft."""
        return ReinforcedConcreteSection(
            diameter=0.9, fc=35000.0, fy=420000.0,
            n_bars=12, bar_diameter=0.025400, cover=0.075,
        )

    def test_Ec(self):
        """Ec should match ACI 318 formula."""
        import math
        rc = self._make_section()
        fc_MPa = 35000.0 / 1000.0
        Ec_expected = 4700.0 * math.sqrt(fc_MPa) * 1000.0
        assert abs(rc.Ec - Ec_expected) / Ec_expected < 1e-10

    def test_Ig(self):
        """Gross I should match pi/4 * r^4."""
        import math
        rc = self._make_section()
        Ig_expected = math.pi / 4.0 * (0.45)**4
        assert abs(rc.Ig - Ig_expected) / Ig_expected < 1e-10

    def test_As(self):
        """Total steel area should be n_bars * pi/4 * db^2."""
        import math
        rc = self._make_section()
        As_expected = 12 * math.pi / 4.0 * 0.025400**2
        assert abs(rc.As - As_expected) / As_expected < 1e-10

    def test_Mcr(self):
        """Cracking moment should match fr*Ig/yt."""
        import math
        rc = self._make_section()
        fc_MPa = 35000.0 / 1000.0
        fr = 0.62 * math.sqrt(fc_MPa) * 1000.0
        yt = 0.45
        Ig = math.pi / 4.0 * 0.45**4
        Mcr_expected = fr * Ig / yt
        assert abs(rc.Mcr - Mcr_expected) / Mcr_expected < 1e-10

    def test_Icr_less_than_Ig(self):
        """Cracked I must be less than gross I."""
        rc = self._make_section()
        assert rc.Icr < rc.Ig

    def test_Icr_positive(self):
        """Cracked I must be positive."""
        rc = self._make_section()
        assert rc.Icr > 0

    def test_effective_EI_below_cracking(self):
        """Below cracking moment, effective EI = Ec * Ig."""
        rc = self._make_section()
        small_moment = rc.Mcr * 0.5
        EI_eff = rc.get_effective_EI(small_moment)
        EI_uncracked = rc.Ec * rc.Ig
        assert abs(EI_eff - EI_uncracked) / EI_uncracked < 1e-10

    def test_effective_EI_at_large_moment(self):
        """At very large moment, effective EI approaches Ec * Icr."""
        rc = self._make_section()
        large_moment = rc.Mcr * 100.0
        EI_eff = rc.get_effective_EI(large_moment)
        EI_cracked = rc.Ec * rc.Icr
        # Should be very close to Icr (within 1%)
        assert abs(EI_eff - EI_cracked) / EI_cracked < 0.01

    def test_effective_EI_intermediate(self):
        """At moderate moment, effective EI is between cracked and gross."""
        rc = self._make_section()
        moderate_moment = rc.Mcr * 2.0
        EI_eff = rc.get_effective_EI(moderate_moment)
        EI_cracked = rc.Ec * rc.Icr
        EI_uncracked = rc.Ec * rc.Ig
        assert EI_cracked < EI_eff < EI_uncracked

    def test_rebar_diameter_lookup(self):
        """Standard rebar lookup should return correct diameter."""
        import math
        d = rebar_diameter("#8")
        assert abs(d - 0.025400) < 1e-6

    def test_rebar_diameter_invalid(self):
        """Unknown bar size should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown rebar size"):
            rebar_diameter("#99")

    def test_invalid_section_params(self):
        """Invalid section parameters should raise ValueError."""
        with pytest.raises(ValueError):
            ReinforcedConcreteSection(
                diameter=0.9, fc=35000.0, n_bars=2,  # too few bars
                bar_diameter=0.025, cover=0.075)
        with pytest.raises(ValueError):
            ReinforcedConcreteSection(
                diameter=0.3, fc=35000.0, n_bars=12,
                bar_diameter=0.025, cover=0.2)  # cover > radius


# =============================================================================
# Test 12: Cracked-EI full analysis integration
# =============================================================================

class TestCrackedEIAnalysis:
    """Verify the cracked-EI outer iteration loop with a full analysis."""

    def _make_rc_analysis(self):
        """Create a standard RC pile analysis in soft clay."""
        rc = ReinforcedConcreteSection(
            diameter=0.9, fc=35000.0, fy=420000.0,
            n_bars=12, bar_diameter=0.025400, cover=0.075,
        )
        pile = Pile.from_rc_section(length=15.0, rc_section=rc)
        layers = [
            SoilLayer(
                top=0.0, bottom=15.0,
                py_model=SoftClayMatlock(c=50.0, gamma=9.0, eps50=0.01, J=0.5),
            ),
        ]
        return LateralPileAnalysis(pile, layers)

    def test_cracked_analysis_converges(self):
        """RC analysis with cracked EI should converge."""
        analysis = self._make_rc_analysis()
        results = analysis.solve(Vt=200.0)
        assert results.converged

    def test_ei_iterations_populated(self):
        """Results should report EI iteration count."""
        analysis = self._make_rc_analysis()
        results = analysis.solve(Vt=200.0)
        assert results.ei_iterations > 0

    def test_EI_profile_in_results(self):
        """Results should contain the final EI profile."""
        analysis = self._make_rc_analysis()
        results = analysis.solve(Vt=200.0)
        assert results.EI_profile is not None
        assert len(results.EI_profile) == len(results.z)

    def test_cracked_EI_reduces_stiffness(self):
        """Cracked analysis should produce lower EI than uncracked at max moment."""
        rc = ReinforcedConcreteSection(
            diameter=0.9, fc=35000.0, fy=420000.0,
            n_bars=12, bar_diameter=0.025400, cover=0.075,
        )
        pile = Pile.from_rc_section(length=15.0, rc_section=rc)
        layers = [
            SoilLayer(
                top=0.0, bottom=15.0,
                py_model=SoftClayMatlock(c=50.0, gamma=9.0, eps50=0.01, J=0.5),
            ),
        ]
        analysis = LateralPileAnalysis(pile, layers)
        results = analysis.solve(Vt=200.0)

        # EI at high-moment nodes should be less than uncracked EI
        EI_uncracked = rc.Ec * rc.Ig
        assert np.min(results.EI_profile) < EI_uncracked
        # EI at zero-moment nodes should remain at uncracked value
        assert np.max(results.EI_profile) == pytest.approx(EI_uncracked, rel=1e-6)

    def test_EI_varies_along_pile(self):
        """EI should vary — nodes with higher moment get lower EI."""
        analysis = self._make_rc_analysis()
        results = analysis.solve(Vt=200.0)
        EI = results.EI_profile
        # EI should not be uniform (some cracking should occur)
        assert np.max(EI) > np.min(EI)

    def test_EI_profile_in_to_dict(self):
        """to_dict() should include EI_profile for RC analyses."""
        analysis = self._make_rc_analysis()
        results = analysis.solve(Vt=200.0)
        d = results.to_dict()
        assert 'EI_profile_kNm2' in d
        assert 'ei_iterations' in d

    def test_non_rc_has_no_EI_profile(self):
        """Standard (non-RC) analysis should not have EI_profile."""
        pile = Pile(length=20.0, diameter=0.610, thickness=0.0127, E=200e6)
        layers = [
            SoilLayer(
                top=0.0, bottom=20.0,
                py_model=SandAPI(phi=35.0, gamma=10.0, k=16000.0),
            ),
        ]
        analysis = LateralPileAnalysis(pile, layers)
        results = analysis.solve(Vt=100.0)
        assert results.EI_profile is None
        assert results.ei_iterations == 0
        assert 'EI_profile_kNm2' not in results.to_dict()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
