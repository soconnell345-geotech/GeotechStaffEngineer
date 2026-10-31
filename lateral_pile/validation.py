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
- API RP2A-WSD, 21st Edition with Supplement 3 (2007).
- FHWA GEC 9, FHWA-HIF-18-031 (2018).
- Reese, L.C. & Van Impe, W.F. (2001). "Single Piles and Pile Groups
  Under Lateral Loading." Balkema.
"""

import numpy as np
import pytest

from lateral_pile.pile import Pile, ReinforcedConcreteSection, rebar_diameter, _HP_SECTIONS
from lateral_pile.soil import SoilLayer
from lateral_pile.py_curves import (
    SoftClayMatlock, SoftClayJeanjean, SandAPI, SandReese,
    StiffClayBelowWT, StiffClayAboveWT, WeakRock, SandLiquefied,
)
from lateral_pile.solver import solve_lateral_pile, hetenyi_solution
from lateral_pile.analysis import LateralPileAnalysis
from geotech_common.numpy_compat import trapezoid as _trapz


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

    def test_sand_reese_three_part_curve(self):
        """SandReese's 'simplified' construction must be a genuine 3-part
        curve (linear -> parabola -> plateau), not bilinear (LP-1)."""
        model = SandReese(phi=35.0, gamma=9.0, k=22000.0, loading='static',
                          construction='simplified')
        z, b = 3.0, 0.6
        yu = 3.0 * b / 80.0
        pus, pud = model.get_pu(z, b)
        A = max(0.9, 3.0 - 0.8 * z / b)
        p_ult = A * min(pus, pud)

        # Plateau reached at yu and held beyond.
        assert model.get_p(yu, z, b) == pytest.approx(p_ult, rel=1e-9)
        assert model.get_p(2.0 * yu, z, b) == pytest.approx(p_ult, rel=1e-9)

        # Parabolic softening at 0.5*yu: at most the 1/3-power value and below
        # the plateau (a bilinear curve would still be on its initial line here).
        p_half = model.get_p(0.5 * yu, z, b)
        assert p_half <= p_ult * (0.5 ** (1.0 / 3.0)) + 1e-9
        assert p_half < p_ult

        # Monotonic, with a genuine intermediate-stiffness region (interior
        # slopes strictly between 0 and the initial slope k*z) -- a bilinear
        # curve would have none.
        y, p = model.get_py_curve(z, b, n_points=200)
        assert np.all(np.diff(p) >= -1e-9)
        slopes = np.diff(p) / np.diff(y)
        interior = slopes[1:-1]
        k_init = 22000.0 * z
        assert np.sum((interior > 1e-6) & (interior < 0.5 * k_init)) >= 3

    def test_sand_reese_construction_default_is_reese1974(self):
        """construction defaults to 'reese1974', GEC 9 App. A.4's printed
        curve (GM-287; 'simplified' before); the two constructions differ."""
        full = SandReese(phi=35.0, gamma=9.0, k=22000.0)
        assert full.construction == 'reese1974'
        simp = SandReese(phi=35.0, gamma=9.0, k=22000.0, construction='simplified')
        z, b = 3.0, 0.6
        ym = b / 60.0
        # default vs explicit reese1974 are identical; simplified differs at the m-point
        assert full.get_p(ym, z, b) == pytest.approx(
            SandReese(phi=35.0, gamma=9.0, k=22000.0,
                      construction='reese1974').get_p(ym, z, b), rel=1e-12)
        assert full.get_p(ym, z, b) != pytest.approx(simp.get_p(ym, z, b), rel=1e-6)
        with pytest.raises(ValueError):
            SandReese(phi=35.0, gamma=9.0, k=22000.0, construction='bogus')

    def test_sand_reese_full_four_segment_curve(self):
        """The full Reese (1974) construction is a genuine four-segment curve with
        the A/B chart m-point: pm = B*pu at ym = b/60, pu_curve = A*pu at yu = 3b/80,
        a straight m-segment between them, and a plateau beyond yu."""
        from lateral_pile.py_curves import _reese_A, _reese_B
        model = SandReese(phi=35.0, gamma=9.0, k=22000.0, loading='static',
                          construction='reese1974')
        z, b = 3.0, 0.6
        zb = z / b
        ym, yu = b / 60.0, 3.0 * b / 80.0
        pus, pud = model.get_pu(z, b)
        pu = min(pus, pud)
        pm_expected = _reese_B(zb, False) * pu
        pu_curve_expected = _reese_A(zb, False) * pu
        # m-point and ultimate resistances match the A/B charts
        assert model.get_p(ym, z, b) == pytest.approx(pm_expected, rel=1e-6)
        assert model.get_p(yu, z, b) == pytest.approx(pu_curve_expected, rel=1e-6)
        # plateau held beyond yu
        assert model.get_p(2.0 * yu, z, b) == pytest.approx(pu_curve_expected, rel=1e-9)
        # m-point below ultimate (B < A), curve monotonic non-decreasing
        assert pm_expected < pu_curve_expected
        y, p = model.get_py_curve(z, b, n_points=300)
        assert np.all(np.diff(p) >= -1e-9)
        # straight m-segment between ym and yu (near-constant slope there)
        m_expected = (pu_curve_expected - pm_expected) / (yu - ym)
        assert (model.get_p(0.5 * (ym + yu), z, b)
                == pytest.approx(pm_expected + m_expected * 0.5 * (yu - ym), rel=0.02))

    def test_sand_reese_A_B_coefficients_asymptotes(self):
        """The re-digitized Reese A/B coefficients hit the PRINTED asymptotes of
        COM624P (FHWA-SA-91-048) Figs 3.12/3.13 at x/b > 5 — Fig 3.12 prints ONE
        annotation "A = 0.88" (static and cyclic merge at depth); Fig 3.13 prints
        "B_c = 0.55, B_s = 0.5" — and the m-point stays below the ultimate
        (B < A) for BOTH loadings at every tabulated depth, which the manual's
        curves satisfy as printed (verified 2026-07-18,
        module_work/wiki_verification/reese_sand_com624p.md)."""
        from lateral_pile.py_curves import _reese_A, _reese_B, _REESE_AB_ZB
        assert _reese_A(5.0, False) == pytest.approx(0.88, abs=0.001)
        assert _reese_A(6.0, True) == pytest.approx(0.88, abs=0.001)
        assert _reese_B(5.0, False) == pytest.approx(0.50, abs=0.001)
        assert _reese_B(6.0, True) == pytest.approx(0.55, abs=0.001)
        for zb in _REESE_AB_ZB:
            assert _reese_B(zb, False) < _reese_A(zb, False)  # static m < ult
            assert _reese_B(zb, True) < _reese_A(zb, True)    # cyclic m < ult

    def test_sand_reese_cyclic_four_segment_and_deep_degeneracy(self):
        """With the manual-verified cyclic tables (COM624P Figs 3.12/3.13), the
        cyclic curve is a genuine four-segment shape at EVERY depth: A_c > B_c
        both at z/b=2 (1.01 vs 0.82) and at the deep asymptotes (A=0.88 vs
        B_c=0.55 — the old premise that they coincide at 0.55 conflated B_c's
        printed asymptote with A's). No warnings, monotone non-decreasing."""
        import warnings
        from lateral_pile.py_curves import _reese_A, _reese_B
        b = 0.6
        ym, yu = b / 60.0, 3.0 * b / 80.0
        model = SandReese(phi=35.0, gamma=9.0, k=22000.0, loading='cyclic',
                          construction='reese1974')
        # z/b = 2: proper four-segment (m-point strictly below the ultimate).
        z2 = 2.0 * b
        pu2 = min(*model.get_pu(z2, b))
        pm2, puc2 = _reese_B(2.0, True) * pu2, _reese_A(2.0, True) * pu2
        assert pm2 < puc2
        assert model.get_p(ym, z2, b) == pytest.approx(pm2, rel=1e-6)
        assert model.get_p(yu, z2, b) == pytest.approx(puc2, rel=1e-6)
        assert model.get_p(2.0 * yu, z2, b) == pytest.approx(puc2, rel=1e-9)  # plateau
        y2, p2 = model.get_py_curve(z2, b, n_points=300)
        assert np.all(np.diff(p2) >= -1e-9)
        # z/b = 5: printed asymptotes A = 0.88 > B_c = 0.55 — still a proper
        # m-segment at depth; no warning/crash, plateau at A*pu.
        z5 = 5.0 * b
        pu5 = min(*model.get_pu(z5, b))
        pm5, puc5 = _reese_B(5.0, True) * pu5, _reese_A(5.0, True) * pu5
        assert pm5 < puc5
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            y5, p5 = model.get_py_curve(z5, b, n_points=300)
        assert np.all(np.diff(p5) >= -1e-9)
        assert model.get_p(3.0 * yu, z5, b) == pytest.approx(0.88 * pu5, rel=1e-9)

    def test_reese_sand_ps_as_printed_gec9(self):
        """p_s of Reese sand from GEC 9 Eqs. A-22 to A-24 (p. 227), GM-287.
        At phi = 30 deg: alpha 15, beta 60, K0 0.4 and Ka 1/3, so C2 =
        tan 60 / tan 30 - 1/3 and C3 = (1/3)(tan^8 60 - 1) + 0.4 tan 30
        tan^4 60. A-23's third term carries a z, as COM624P Eq. 3.31 (p.
        341) prints it; GEC 9 prints it without. API RP 2A Fig. 6.8.6-1
        plots the same coefficients, 1.914 / 2.679 / 28.56 read at 30 deg,
        which only the term with its z reproduces."""
        import math
        from lateral_pile.py_curves import _reese_sand_coefficients
        t, r = math.tan, math.radians
        C1, C2, C3 = _reese_sand_coefficients(30.0)
        c1 = (0.4 * t(r(30)) * math.sin(r(60)) / (t(r(30)) * math.cos(r(15)))
              + t(r(60)) ** 2 * t(r(15)) / t(r(30))
              + 0.4 * t(r(60)) * (t(r(30)) * math.sin(r(60)) - t(r(15))))
        assert C1 == pytest.approx(c1, rel=1e-12)
        assert C1 == pytest.approx(1.912, abs=0.001)
        assert C2 == pytest.approx(3.0 - 1.0 / 3.0, rel=1e-12)
        assert C3 == pytest.approx(80.0 / 3.0 + 0.4 * t(r(30)) * 9.0, rel=1e-12)
        assert (C1, C2, C3) == pytest.approx((1.914, 2.679, 28.56), rel=0.01)
        # in the model: p_st = gamma z (C1 z + C2 D), p_sd = C3 D gamma z
        pst, psd = SandReese(phi=30.0, gamma=10.0, k=20000.0).get_pu(2.0, 0.6)
        assert pst == pytest.approx(10.0 * 2.0 * (c1 * 2.0 + C2 * 0.6), rel=1e-12)
        assert psd == pytest.approx(C3 * 0.6 * 10.0 * 2.0, rel=1e-12)

    def test_api_sand_coefficients_read_off_fig_6_8_6_1(self):
        """API sand C1, C2, C3 from API RP 2A-WSD Fig. 6.8.6-1 (p. 70), read
        off its vector curves at whole degrees, linear between (GM-287)."""
        from lateral_pile.py_curves import (
            _api_sand_coefficients, _API_SAND_PHI, _API_SAND_C1,
            _API_SAND_C2, _API_SAND_C3)
        assert _api_sand_coefficients(30.0) == pytest.approx(
            (1.914, 2.679, 28.56), rel=0.002)
        i = list(_API_SAND_PHI).index(35.0)
        assert _api_sand_coefficients(35.5) == pytest.approx(
            tuple((c[i] + c[i + 1]) / 2.0
                  for c in (_API_SAND_C1, _API_SAND_C2, _API_SAND_C3)))
        with pytest.warns(UserWarning):
            assert _api_sand_coefficients(45.0) == pytest.approx(
                _api_sand_coefficients(40.0))
        C1, C2, C3 = _api_sand_coefficients(30.0)
        pu = SandAPI(phi=30.0, gamma=10.0, k=20000.0).get_pu(2.0, 0.6)
        assert pu == pytest.approx(min((C1 * 2.0 + C2 * 0.6) * 10.0 * 2.0,
                                       C3 * 0.6 * 10.0 * 2.0))

    def test_sand_k_table_a3_by_relative_density(self):
        """GEC 9 Table A-3 (p. 228) gives sand k by relative density only:
        no friction angle is mapped to it, and no phi 40 value (GM-287)."""
        from lateral_pile.py_curves import _sand_k_recommendation
        assert _sand_k_recommendation("medium", below_wt=True) == pytest.approx(
            60.0 * 271.447)
        assert _sand_k_recommendation("dense", below_wt=False) == pytest.approx(
            225.0 * 271.447)
        with pytest.raises((ValueError, KeyError, AttributeError)):
            _sand_k_recommendation(35.0)

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

    def test_liquefied_sand_pu_increases_with_depth(self):
        """Liquefied sand pu should increase with depth."""
        model = SandLiquefied(diameter=0.6)
        pu_1 = model.get_pu(1.0, 0.6)
        pu_5 = model.get_pu(5.0, 0.6)
        assert pu_5 > pu_1 > 0

    def test_liquefied_sand_as_printed_gec9(self):
        """GEC 9 Eq. A-43 (p. 232) takes y in mm and gives p in kN/m, times
        P_d of Eq. A-44 (p. 233); before GM-287 y entered in metres. At
        z = 1 m, y = 50 mm, D = 0.6 m: about 3.4 kN/m, inside the printed
        range (p up to about 85 lb/in, 14.9 kN/m)."""
        import math
        model = SandLiquefied(diameter=0.6)
        z = 1.0
        A = 3e-7 * (z + 1.0) ** 6.05
        B = 2.80 * (z + 1.0) ** 0.11
        C = 2.85 * (z + 1.0) ** -0.41
        Pd = 3.81 * math.log(0.6) + 5.6
        p = model.get_p(0.050, z, 0.6)
        assert p == pytest.approx(A * (B * 50.0) ** C * Pd, rel=1e-12)
        assert 1.0 < p < 14.9

    def test_liquefied_sand_concave_up_shape(self):
        """Liquefied sand curve should be concave-up (stiffness increases)."""
        model = SandLiquefied(diameter=0.6)
        z, b = 3.0, 0.6
        # Secant stiffness should increase with deflection (concave-up)
        p1 = model.get_p(0.02, z, b)
        p2 = model.get_p(0.10, z, b)
        secant_1 = p1 / 0.02
        secant_2 = p2 / 0.10
        assert secant_2 > secant_1

    def test_liquefied_sand_cap_at_150mm(self):
        """Resistance should cap at y = 0.15 m."""
        model = SandLiquefied(diameter=0.6)
        z, b = 5.0, 0.6
        pu = model.get_pu(z, b)
        p_at_150 = model.get_p(0.15, z, b)
        p_at_200 = model.get_p(0.20, z, b)
        assert abs(p_at_150 - pu) < 1e-10
        assert abs(p_at_200 - pu) < 1e-10

    def test_liquefied_sand_antisymmetry(self):
        """p(-y) should equal -p(y)."""
        model = SandLiquefied(diameter=0.6)
        p_pos = model.get_p(0.05, 3.0, 0.6)
        p_neg = model.get_p(-0.05, 3.0, 0.6)
        assert abs(p_pos + p_neg) < 1e-10

    def test_liquefied_sand_surface_zero(self):
        """At z=0, resistance should be zero."""
        model = SandLiquefied(diameter=0.6)
        assert model.get_p(0.05, 0.0, 0.6) == 0.0
        assert model.get_pu(0.0, 0.6) == 0.0

    def test_liquefied_sand_very_soft(self):
        """Liquefied sand should be much softer than intact API sand."""
        liq = SandLiquefied(diameter=0.6)
        intact = SandAPI(phi=35.0, gamma=10.0, k=16000.0)
        z, b = 3.0, 0.6
        p_liq = liq.get_p(0.05, z, b)
        p_intact = intact.get_p(0.05, z, b)
        # Liquefied should be at least 10x softer
        assert p_liq < p_intact / 10.0

    def test_liquefied_sand_diameter_effect(self):
        """Larger diameter should give higher resistance (pd correction)."""
        small = SandLiquefied(diameter=0.3)
        large = SandLiquefied(diameter=1.0)
        z, b = 3.0, 0.6
        assert large.get_pu(z, b) > small.get_pu(z, b)

    def test_liquefied_sand_get_py_curve(self):
        """get_py_curve should return arrays of correct shape."""
        model = SandLiquefied(diameter=0.6)
        y_arr, p_arr = model.get_py_curve(3.0, 0.6, n_points=30)
        assert len(y_arr) == 30
        assert len(p_arr) == 30
        assert y_arr[0] == 0.0
        assert p_arr[0] == 0.0
        # All positive for positive y
        assert all(p >= 0 for p in p_arr)


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

    def test_partial_head_between_free_and_fixed(self):
        """A rotational spring at the head (head_condition='partial') gives a
        deflection and head moment between the free and fixed heads, moving
        toward fixed as the spring stiffens (GEC 9 6.7.3: deflection grows
        as fixity falls), with M = Kr x rotation. Before GM-287 the spring's
        sign was reversed and drove the rotation: at Kr 1e4 kN-m/rad the
        head deflected more than a free one."""
        analysis = _make_soft_clay_case()
        free = analysis.solve(Vt=100.0, head_condition='free')
        fixed = analysis.solve(Vt=100.0, head_condition='fixed')
        prev_y, prev_m = free.y_top, 0.0
        for kr in (1e3, 1e4, 1e5, 1e6):
            r = analysis.solve(Vt=100.0, head_condition='partial',
                               rotational_stiffness=kr)
            assert r.converged
            assert fixed.y_top < r.y_top < prev_y
            assert fixed.moment[0] < r.moment[0] < prev_m
            assert r.moment[0] == pytest.approx(kr * r.rotation_top, rel=1e-3)
            prev_y, prev_m = r.y_top, r.moment[0]

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

        Uses the Matlock (1970) test pile at high load where deflections
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
          y_top  = 3.68 mm
          Mmax   = 98.9 kN-m at ~1.9 m
        (3.62 mm and 97.7 kN-m before GM-287 took C1, C2, C3 off API RP 2A
        Fig. 6.8.6-1 in place of a table no page prints.)
        """
        analysis = _make_api_sand_case()
        results = analysis.solve(Vt=89.0, Mt=0.0, Q=0.0,
                                 head_condition='free', n_elements=200)

        assert results.converged
        assert abs(results.y_top * 1000 - 3.68) / 3.68 < 0.05, (
            f"y_top = {results.y_top*1000:.2f} mm, expected ~3.68 mm"
        )
        assert abs(results.max_moment - 98.9) / 98.9 < 0.05, (
            f"Mmax = {results.max_moment:.1f} kN-m, expected ~98.9 kN-m"
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

    def test_matlock_1970_test_pile_benchmark(self):
        """Matlock (1970), OTC 1204: the instrumented test pile in the
        stronger of his two soft clays.

        Pile and soil parameters as the literature reports the test; the
        values below are this module's own results, not published ones.

        Steel pipe pile: D=324mm (12.75"), t=12.7mm (0.5"), L=12.8m
        EI = 30,141 kN-m^2
        Soft clay: su=38.3 kPa, gamma'=10.0 kN/m3, eps50=0.012, J=0.5
        Free head, static loading, load at ground line.

        Module results (n=400, equilibrium verified to <0.01%):
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

    def test_gec9_example_b1_its_pole(self):
        """FHWA GEC 9 (2018) App. B.1, pp. 235-242: the ITS pole on a 54 in
        drilled shaft 45 ft long, its top 1.5 ft above the ground, free
        head, loads at the top. Table B-1 (p. 236, depths from the top): Reese sand,
        phi 30 deg, to 15 ft (gamma' 115 pcf, k 90 pci); Reese sand below
        the water table to 30 ft (52.6 pcf, 60 pci); soft clay to 40 ft (su
        300 psf, 47.6 pcf, eps50 0.02; Matlock with J 0.5, GEC 9 p. 218);
        Reese sand to the tip (57.6 pcf, 80 pci). f'c 4,000 psi, 20 #10
        bars, 3 in of cover to the bars. Printed (LPILE, p. 239): service loads
        (144 kip-ft, 3 kip, 9 kip axial) give a head deflection of 0.03 in;
        strength loads (672 kip-ft, 14 kip, 12 kip) a maximum moment of 727
        kip-ft and a maximum shear of 35.5 kip.

        LPILE took the shaft's nonlinear EI; here the shaft keeps its
        uncracked transformed section (composite_section_ei), so above the
        cracking moment (608 kip-ft) it is a little stiff and its moment a
        little high: 735.6 kip-ft (+1.2 %), shear 36.8 kip (+3.6 %). Before
        GM-287, with sand coefficients from a table no page prints: 0.0333
        in, 744.7 kip-ft, 35.2 kip."""
        from lateral_pile.composite_section import composite_section_ei
        FT, IN, PCF, PSF = 0.3048, 0.0254, 0.157087, 0.0478803
        PCI, KIP, PSI = 271.447, 4.448222, 6.894757
        bar = rebar_diameter("#10")
        sec = composite_section_ei(
            "reinforced_concrete", fc=4000 * PSI, diameter=54 * IN,
            n_bars=20, bar_diameter=bar,
            bar_circle_diameter=54 * IN - 2 * (3 * IN + bar / 2))
        Ec = ReinforcedConcreteSection(diameter=54 * IN, fc=4000 * PSI).Ec
        pile = Pile(length=43.5 * FT, diameter=54 * IN, E=Ec,
                    moment_of_inertia=sec.EI / Ec)
        layers = [
            SoilLayer(0.0, 13.5 * FT,
                      SandReese(phi=30.0, gamma=115 * PCF, k=90 * PCI)),
            SoilLayer(13.5 * FT, 28.5 * FT,
                      SandReese(phi=30.0, gamma=52.6 * PCF, k=60 * PCI)),
            SoilLayer(28.5 * FT, 38.5 * FT,
                      SoftClayMatlock(c=300 * PSF, gamma=47.6 * PCF,
                                      eps50=0.02, J=0.5)),
            SoilLayer(38.5 * FT, 43.5 * FT,
                      SandReese(phi=30.0, gamma=57.6 * PCF, k=80 * PCI)),
        ]
        analysis = LateralPileAnalysis(pile, layers)
        service = analysis.solve(Vt=3 * KIP, Mt=144 * KIP * FT, Q=9 * KIP,
                                 stickup=1.5 * FT)
        strength = analysis.solve(Vt=14 * KIP, Mt=672 * KIP * FT,
                                  Q=12 * KIP, stickup=1.5 * FT)
        assert service.converged and strength.converged
        y_in = service.y_top / IN
        m_kft = strength.max_moment / (KIP * FT)
        v_kip = strength.max_shear / KIP
        # the module's own values
        assert y_in == pytest.approx(0.0333, rel=0.01)
        assert m_kft == pytest.approx(735.6, rel=0.005)
        assert v_kip == pytest.approx(36.78, rel=0.005)
        # against the print: 0.03 in to two places, 727 kip-ft, 35.5 kip
        assert 0.025 <= y_in < 0.035
        assert m_kft == pytest.approx(727.0, rel=0.02)
        assert v_kip == pytest.approx(35.5, rel=0.05)

    @pytest.mark.parametrize("v_kip, y_top_in, m_max_in_kip", [
        (20.0, 0.145, 661.0),     # printed summary table, p. 126
        (40.0, 0.623, 1740.0),
    ])
    def test_com624p_example_1(self, v_kip, y_top_in, m_max_in_kip):
        """COM624P Example 1 (FHWA-SA-91-048, printed pp. 101-126): a
        14HP89, strong axis (I 904 in^4, E 29,000 ksi), 300 in long, head
        at the ground, free, 39 kips axial, in uniform stiff clay above the
        water table (Welch & Reese: c 14 psi, gamma 0.069 lb/in^3, eps50
        0.007), static, 120 increments. Head deflection and maximum moment
        as printed on p. 126; the printed p-y curves (pp. 121-122) are
        pinned in tests/test_stiff_clay_above_wt.py (GM-202)."""
        IN, PSI, PCI, KIP = 0.0254, 6.894757, 271.4471, 4.448222
        b = 14.7 * IN
        pile = Pile(length=300 * IN, diameter=b, E=29e6 * PSI,
                    moment_of_inertia=904 * IN ** 4)
        clay = StiffClayAboveWT(c=14.0 * PSI, gamma=0.069 * PCI, eps50=0.007)
        r = LateralPileAnalysis(pile, [SoilLayer(0.0, 300 * IN, clay)]).solve(
            Vt=v_kip * KIP, Q=39.0 * KIP, n_elements=120)
        assert r.converged
        assert r.y_top / IN == pytest.approx(y_top_in, rel=0.01)
        assert r.max_moment / (KIP * IN) == pytest.approx(m_max_in_kip,
                                                          rel=0.01)

    def test_soft_clay_regression_case_source_unknown(self):
        """Regression case, source unknown: NOT a published check.

        From the module's first commit until GM-209 this case was labelled
        "COM624P Example 1" with a "published" 34.3 mm (1.35 in). It is
        not: the manual's Example 1 is the 14HP89 in stiff clay above
        (test_com624p_example_1). No printed source for 34.3 mm was found
        (GM-209 searched the COM624P manual, Matlock 1970 OTC 1204, GEC 9
        and other lateral-pile literature on file). Matlock's
        instrumented pile had a 0.5 in wall, not this 3/8 in, and his
        free-head deflections are printed for cyclic loading only. 34.3 mm
        is within 0.3 % of what this module computes for these inputs, so
        the case guards against change, not against a published answer.

        Steel pipe pile: D 0.3239 m, t 9.525 mm, L 12.8 m (EI about 23,265
        kN m^2). Soft clay: su 14.4 kPa, gamma' 5.0 kN/m^3, eps50 0.012,
        J 0.5. Free head, static, load at the ground line, 44.5 kN.
        """
        pile = Pile(length=12.8, diameter=0.3239, thickness=0.009525, E=200e6)
        layers = [SoilLayer(top=0.0, bottom=12.8,
            py_model=SoftClayMatlock(c=14.4, gamma=5.0, eps50=0.012, J=0.5,
                                     loading='static'))]
        analysis = LateralPileAnalysis(pile, layers)
        results = analysis.solve(Vt=44.5, head_condition='free', n_elements=200)

        assert results.converged
        # The module's own result (34.41 mm at GM-209), not a published one
        y_mm = results.y_top * 1000
        rel_error = abs(y_mm - 34.3) / 34.3
        assert rel_error < 0.01, (
            f"soft clay regression case: y_top = {y_mm:.2f} mm, "
            f"pinned 34.3 mm, change = {rel_error:.1%}"
        )

    def test_sand_high_load_benchmark(self):
        """API sand at higher load level.

        Reference (n=400, equilibrium verified):
          V=178 kN: y_top = 9.06 mm, Mmax = 225.1 kN-m
        (8.61 mm and 217.5 kN-m before GM-287 took C1, C2, C3 off API RP 2A
        Fig. 6.8.6-1 in place of a table no page prints.)
        """
        analysis = _make_api_sand_case()
        results = analysis.solve(Vt=178.0, Mt=0.0, Q=0.0,
                                 head_condition='free', n_elements=200)

        assert results.converged
        assert abs(results.y_top * 1000 - 9.06) / 9.06 < 0.05
        assert abs(results.max_moment - 225.1) / 225.1 < 0.05

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
