"""
Phase-4 NR solver hardening tests.

The tangent path (method='tangent') with continuum elastoplastic tangent,
K_T reform interval (modified NR), divergence cutback, optional line
search, and the dual residual+displacement convergence criterion.
"""

import numpy as np
import pytest

from fem2d.mesh import generate_slope_mesh, detect_boundary_nodes, \
    convert_to_t6
from fem2d.solver import build_nl_context, run_nl, solve_nonlinear, \
    _material_arrays


SURFACE = [(0.0, 6.0), (6.0, 6.0), (18.0, 0.0), (24.0, 0.0)]


def _problem(element_type='cst', c=8.0, phi=20.0):
    nodes, elements = generate_slope_mesh(SURFACE, depth=6.0, nx=10, ny=5)
    if element_type == 't6':
        nodes, elements = convert_to_t6(nodes, elements)
    props = [{'E': 1e5, 'nu': 0.3, 'c': c, 'phi': phi, 'psi': 0.0,
              'gamma': 20.0}]
    bc = detect_boundary_nodes(nodes)
    return nodes, elements, props, bc


def _solve(method, element_type='cst', **kw):
    nodes, elements, props, bc = _problem(element_type)
    conv, u, s, e = solve_nonlinear(
        nodes, elements, props, 20.0, bc, n_steps=5, max_iter=200,
        tol=1e-6, method=method, **kw)
    return conv, u, s


class TestTangentMethod:

    def test_tangent_matches_elastic_method(self):
        """Both methods must converge to the same equilibrium."""
        conv_e, u_e, s_e = _solve('elastic')
        conv_t, u_t, s_t = _solve('tangent')
        assert conv_e and conv_t
        scale = np.max(np.abs(u_e))
        assert np.max(np.abs(u_t - u_e)) < 0.02 * scale
        assert np.allclose(s_t, s_e, atol=2.0)  # kPa

    def test_tangent_t6(self):
        conv_e, u_e, _ = _solve('elastic', element_type='t6')
        conv_t, u_t, _ = _solve('tangent', element_type='t6')
        assert conv_e and conv_t
        scale = np.max(np.abs(u_e))
        assert np.max(np.abs(u_t - u_e)) < 0.02 * scale

    def test_tangent_fewer_iterations_than_elastic(self):
        """The elastoplastic tangent must beat constant stiffness."""
        nodes, elements, props, bc = _problem()
        ctx = build_nl_context(nodes, elements, props, 20.0, bc)
        mats = _material_arrays(props, len(elements))
        r_e = run_nl(ctx, n_steps=2, max_iter=500, tol=1e-6,
                     method='elastic', mats_override=mats)
        r_t = run_nl(ctx, n_steps=2, max_iter=500, tol=1e-6,
                     method='tangent', mats_override=mats)
        assert r_e['converged'] and r_t['converged']
        assert r_t['n_iter_total'] < r_e['n_iter_total']

    def test_reform_interval_modified_nr(self):
        """Modified NR (reform every 3 its) reaches the same answer."""
        conv1, u1, _ = _solve('tangent', reform_interval=1)
        conv3, u3, _ = _solve('tangent', reform_interval=3)
        assert conv1 and conv3
        scale = np.max(np.abs(u1))
        assert np.max(np.abs(u3 - u1)) < 0.01 * scale

    def test_line_search_converges_same(self):
        conv0, u0, _ = _solve('tangent', line_search=False)
        conv1, u1, _ = _solve('tangent', line_search=True)
        assert conv0 and conv1
        scale = np.max(np.abs(u0))
        assert np.max(np.abs(u1 - u0)) < 0.01 * scale


class TestDualConvergence:

    def test_disp_tol_accepts_converged_state(self):
        """Dual criterion (du + residual cap) converges on a stable
        problem and agrees with the residual-only solution."""
        conv_r, u_r, _ = _solve('elastic')
        conv_d, u_d, _ = _solve('elastic', disp_tol=1e-8)
        assert conv_r and conv_d
        scale = np.max(np.abs(u_r))
        assert np.max(np.abs(u_d - u_r)) < 0.02 * scale

    def test_disp_tol_gated_by_residual_cap(self):
        """A huge disp_tol alone must NOT trigger acceptance while the
        residual is far from equilibrium (cap gates it)."""
        nodes, elements, props, bc = _problem(c=2.0, phi=10.0)
        ctx = build_nl_context(nodes, elements, props, 20.0, bc)
        mats = _material_arrays(props, len(elements))
        res = run_nl(ctx, n_steps=1, max_iter=5, tol=1e-12,
                     method='elastic', mats_override=mats,
                     disp_tol=1e6, disp_residual_cap=1e-12)
        # neither criterion satisfiable in 5 iterations on a yielding
        # problem with an impossible residual cap
        assert not res['converged']


class TestDivergenceCutback:

    def test_failing_slope_returns_gracefully(self):
        """Tangent method on a failing slope: cutbacks exhaust, the
        solver reports non-convergence with finite displacements."""
        nodes, elements, props, bc = _problem(c=0.5, phi=5.0)
        ctx = build_nl_context(nodes, elements, props, 20.0, bc)
        mats = _material_arrays(props, len(elements))
        res = run_nl(ctx, n_steps=2, max_iter=30, tol=1e-6,
                     method='tangent', mats_override=mats, max_cutbacks=2)
        assert not res['converged']
        assert np.all(np.isfinite(res['u']))

    def test_max_cutbacks_zero_fails_immediately(self):
        nodes, elements, props, bc = _problem(c=0.5, phi=5.0)
        ctx = build_nl_context(nodes, elements, props, 20.0, bc)
        mats = _material_arrays(props, len(elements))
        r0 = run_nl(ctx, n_steps=2, max_iter=30, tol=1e-6,
                    method='tangent', mats_override=mats, max_cutbacks=0)
        r2 = run_nl(ctx, n_steps=2, max_iter=30, tol=1e-6,
                    method='tangent', mats_override=mats, max_cutbacks=3)
        # more cutbacks -> at least as many committed load steps
        assert len(r2['iterations']) >= len(r0['iterations'])


class TestAPIStability:

    def test_solve_nonlinear_signature(self):
        import inspect
        sig = inspect.signature(solve_nonlinear)
        for name in ('method', 'reform_interval', 'disp_tol',
                     'max_cutbacks', 'line_search', 'n_gp', 'return_gp'):
            assert name in sig.parameters
        assert sig.parameters['method'].default == 'elastic'
