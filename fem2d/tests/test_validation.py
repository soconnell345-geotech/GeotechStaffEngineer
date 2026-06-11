"""
Published-benchmark validation suite (see fem2d/VALIDATION.md).

Centerpiece tests:
- Griffiths & Lane (1999) Example 1 homogeneous slope: published FE FOS
  1.4, Bishop & Morgenstern 1.380.
- Griffiths & Lane (1999) Example 4 undrained two-layer slope: Taylor
  1.47 (cu2/cu1=1), plateau ~2.1 (cu2/cu1=2).
- Prandtl strip footing Nc = 2 + pi = 5.14: T6 within a few percent,
  CST locks (never collapses) — the element-quality contrast.
- Elastic closed form (1D gravity compression).
- Bishop cross-check vs slope_stability on a shared geometry.

All FEM-heavy tests are marked slow. Assertion bands document the
measured accuracy recorded in VALIDATION.md — they are tight enough to
catch regressions, wide enough for cross-platform numeric noise.
"""

import numpy as np
import pytest

from fem2d.analysis import analyze_gravity, analyze_slope_srm
from fem2d.mesh import (generate_rect_mesh, generate_slope_mesh,
                        convert_to_t6, detect_boundary_nodes,
                        t6_boundary_edges)
from fem2d.solver import build_nl_context, run_nl
from fem2d.srm import strength_reduction


# ============================================================================
# Griffiths & Lane (1999) Example 1 — homogeneous 2:1 slope, FOS 1.4
# ============================================================================

GL_H = 10.0
GL_GAMMA = 20.0
EX1_SURF = [(0.0, GL_H), (1.2 * GL_H, GL_H), (3.2 * GL_H, 0.0),
            (4.4 * GL_H, 0.0)]
EX1_PROPS = [{'E': 1e5, 'nu': 0.3, 'c': 10.0, 'phi': 20.0, 'psi': 0.0,
              'gamma': GL_GAMMA}]


@pytest.fixture(scope='module')
def gl99_ex1_t6():
    """T6 32x12, D=1 (0.5 m base sliver), bisection tol 0.02 (~10 s)."""
    nodes, elements = generate_slope_mesh(EX1_SURF, depth=0.5,
                                          nx=32, ny=12)
    nodes, elements = convert_to_t6(nodes, elements)
    return strength_reduction(nodes, elements, EX1_PROPS, GL_GAMMA,
                              tol=0.02, h_ref=GL_H)


@pytest.mark.slow
class TestGriffithsLaneExample1:

    def test_fos_vs_published(self, gl99_ex1_t6):
        """Published FE 1.4, B&M 1.380; T6 measured 1.34 (-4%)."""
        fos = gl99_ex1_t6['FOS']
        print(f"\n  GL99 Ex1 T6 32x12: FOS={fos:.3f} (published 1.4)")
        assert 1.28 <= fos <= 1.48
        assert gl99_ex1_t6['fos_basis'] == 'nonconvergence'

    def test_dimensionless_displacement_curve(self, gl99_ex1_t6):
        """Curve must reproduce GL99 Table 2: flat (~0.38-0.45 at
        SRF<=1) then a knee toward failure."""
        srf, dim = gl99_ex1_t6['srf_curve']
        early = dim[srf <= 1.0]
        assert len(early) >= 1
        assert np.all((early > 0.30) & (early < 0.55))
        # rising toward failure
        assert dim[np.argmax(srf)] > early.min()


# ============================================================================
# Griffiths & Lane (1999) Example 4 — undrained two-layer, D=2
# ============================================================================

EX4_SURF = [(0.0, GL_H), (1.2 * GL_H, GL_H), (3.2 * GL_H, 0.0),
            (5.6 * GL_H, 0.0)]
CU1 = 0.25 * GL_GAMMA * GL_H  # 50 kPa


def _ex4_run(ratio, nu, tol=0.02):
    nodes, elements = generate_slope_mesh(EX4_SURF, depth=GL_H,
                                          nx=40, ny=16)
    nodes, elements = convert_to_t6(nodes, elements)
    cents = nodes[elements[:, :3]].mean(axis=1)
    props = [{'E': 1e5, 'nu': nu,
              'c': CU1 if cy >= 0.0 else CU1 * ratio,
              'phi': 0.0, 'psi': 0.0, 'gamma': GL_GAMMA}
             for cy in cents[:, 1]]
    return strength_reduction(nodes, elements, props, GL_GAMMA,
                              tol=tol, h_ref=GL_H)


@pytest.mark.slow
class TestGriffithsLaneExample4:

    def test_homogeneous_taylor_bracketed(self):
        """cu2/cu1 = 1, published 1.47 (Taylor). The undrained
        (incompressible) case is the formulation's weak spot: nu=0.49
        locks high (+~25%), nu=0.30 yields out-of-plane early (-~10%).
        The published value must be BRACKETED (VALIDATION.md sec 2).
        """
        lo = _ex4_run(1.0, 0.30)['FOS']
        hi = _ex4_run(1.0, 0.49)['FOS']
        print(f"\n  GL99 Ex4 ratio 1.0: nu=0.30 -> {lo:.3f}, "
              f"nu=0.49 -> {hi:.3f} (published 1.47)")
        assert lo - 0.05 <= 1.47 <= hi + 0.05
        assert 1.0 < lo < hi < 2.1

    def test_strong_foundation(self):
        """cu2/cu1 = 2: toe circle, published ~2.1 (nu=0.30 run)."""
        r = _ex4_run(2.0, 0.30)
        print(f"\n  GL99 Ex4 ratio 2.0: FOS={r['FOS']:.3f} "
              f"(published ~2.1)")
        assert 1.95 <= r['FOS'] <= 2.45

    def test_weak_foundation_drops_fos(self):
        """cu2/cu1 = 0.6: weak-foundation mechanism, FOS < 1."""
        r = _ex4_run(0.6, 0.30)
        print(f"\n  GL99 Ex4 ratio 0.6: FOS={r['FOS']:.3f}")
        assert r['FOS'] < 1.05


# ============================================================================
# Prandtl strip footing — Nc = 2 + pi (T6 vs locked CST)
# ============================================================================

def _footing_collapse(element_type, nx=40, ny=20, q_app=900.0,
                      n_steps=45):
    B, c = 2.0, 100.0
    nodes, elements = generate_rect_mesh(-5 * B, 5 * B, -5 * B, 0, nx, ny)
    n_corner = len(nodes)
    if element_type == 't6':
        nodes, elements = convert_to_t6(nodes, elements)
    bc = detect_boundary_nodes(nodes)
    surf = np.where(np.abs(nodes[:n_corner, 1]) < 1e-9)[0]
    loaded = surf[(nodes[surf, 0] >= -B / 2 - 1e-9) &
                  (nodes[surf, 0] <= B / 2 + 1e-9)]
    loaded = loaded[np.argsort(nodes[loaded, 0])]
    edges = [(loaded[i], loaded[i + 1]) for i in range(len(loaded) - 1)]
    if element_type == 't6':
        edges = t6_boundary_edges(elements, edges)
    props = [{'E': 1e5, 'nu': 0.3, 'c': c, 'phi': 0.0, 'psi': 0.0,
              'gamma': 0.0}]
    ctx = build_nl_context(nodes, elements, props, 0.0, bc,
                           surface_loads=[(edges, 0.0, -q_app)])
    res = run_nl(ctx, n_steps=n_steps, max_iter=1000, tol=1e-4,
                 method='elastic')
    n_ok = len(res['iterations']) - (0 if res['converged'] else 1)
    return q_app * n_ok / n_steps / c, res['converged']


@pytest.mark.slow
class TestPrandtlFooting:

    def test_t6_within_a_few_percent(self):
        nc, _ = _footing_collapse('t6')
        print(f"\n  Prandtl footing T6: Nc~{nc:.2f} (exact 5.14)")
        assert 4.8 <= nc <= 5.5

    def test_cst_locks_and_overshoots(self):
        """CST must NOT be trusted for collapse: it locks and carries
        far beyond the exact limit load."""
        nc, converged = _footing_collapse('cst')
        print(f"\n  Prandtl footing CST: carried Nc~{nc:.2f} "
              f"(exact 5.14) conv={converged}")
        assert nc > 7.0  # >35% overshoot: classic CST locking


# ============================================================================
# Elastic closed form
# ============================================================================

class TestElasticClosedForm:

    def test_gravity_settlement_t6(self):
        """1D gravity compression: w = gamma H^2 / (2 M), within 5%."""
        W, D, G, E, NU = 40.0, 10.0, 18.0, 30000.0, 0.3
        M = E * (1 - NU) / ((1 + NU) * (1 - 2 * NU))
        w_exact = G * D ** 2 / (2 * M)
        res = analyze_gravity(width=W, depth=D, gamma=G, E=E, nu=NU,
                              nx=40, ny=10, element_type='t6')
        err = abs(res.max_displacement_y_m - w_exact) / w_exact
        assert err < 0.05, f"T6 settlement error {err:.1%}"


# ============================================================================
# Cross-check vs slope_stability Bishop (shared geometry)
# ============================================================================

@pytest.mark.slow
class TestBishopCrossCheck:

    SURFACE = [(0, 0), (10, 0), (30, 10), (50, 10)]
    LAYER = {'name': 'clay', 'bottom_elevation': -10,
             'E': 30000, 'nu': 0.3, 'c': 10.0, 'phi': 15.0, 'psi': 0,
             'gamma': 18.0}

    def test_srm_t6_vs_bishop(self):
        from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
        from slope_stability.analysis import search_critical_surface

        geom = SlopeGeometry(
            surface_points=self.SURFACE,
            soil_layers=[SlopeSoilLayer(
                name='clay', top_elevation=10, bottom_elevation=-20,
                gamma=18.0, phi=15.0, c_prime=10.0)])
        bishop = search_critical_surface(
            geom, method='bishop', nx=15, ny=15).critical.FOS

        srm = analyze_slope_srm(
            surface_points=self.SURFACE, soil_layers=[self.LAYER],
            depth=5.0, nx=40, ny=20, srf_tol=0.02, x_extend=0.0,
            element_type='t6')

        print(f"\n  Bishop={bishop:.3f}  SRM T6={srm.FOS:.3f}")
        # SRM sits above Bishop at this density and refines toward it
        # (1.419 @ 40x20, 1.306 @ 56x24 vs Bishop 1.173 — VALIDATION.md)
        assert srm.FOS >= bishop - 0.05
        assert (srm.FOS - bishop) / bishop < 0.30
