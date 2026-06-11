"""
Phase-3 SRM robustness tests.

Covers: strength-reduction policy (c, tan(phi), psi cap, srm_field, HS),
failure detection (non-convergence + dimensionless-displacement blowup),
adaptive bracket+bisect outputs (srf_history, srf_curve, fos_basis),
stall-window early abort, and element_type plumbing through
analyze_slope_srm.

References: Griffiths & Lane (1999); PLAXIS psi_red = min(psi, phi_red).
"""

import math

import numpy as np
import pytest

from fem2d.analysis import analyze_slope_srm
from fem2d.mesh import generate_slope_mesh
from fem2d.srm import strength_reduction, _reduce_props


# ============================================================================
# Reduction policy (pure-python, instant)
# ============================================================================

def _props(c=10.0, phi=20.0, psi=0.0, **extra):
    p = {'E': 1e5, 'nu': 0.3, 'c': c, 'phi': phi, 'psi': psi,
         'gamma': 20.0}
    p.update(extra)
    return p


class TestReducePolicy:

    def test_c_and_tanphi_reduced(self):
        rp = _reduce_props([_props()], 2.0, 'c_phi')[0]
        assert rp['c'] == pytest.approx(5.0)
        expected_phi = math.degrees(
            math.atan(math.tan(math.radians(20.0)) / 2.0))
        assert rp['phi'] == pytest.approx(expected_phi)

    def test_psi_capped_at_reduced_phi(self):
        """PLAXIS policy: when phi_red < psi, psi_red = phi_red."""
        rp = _reduce_props([_props(phi=30.0, psi=30.0)], 2.0, 'c_phi')[0]
        assert rp['phi'] < 30.0
        assert rp['psi'] == pytest.approx(rp['phi'])

    def test_psi_below_reduced_phi_unchanged(self):
        rp = _reduce_props([_props(phi=30.0, psi=5.0)], 1.2, 'c_phi')[0]
        assert rp['psi'] == pytest.approx(5.0)

    def test_field_c_only(self):
        rp = _reduce_props([_props()], 2.0, 'c')[0]
        assert rp['c'] == pytest.approx(5.0)
        assert rp['phi'] == pytest.approx(20.0)

    def test_field_phi_only(self):
        rp = _reduce_props([_props()], 2.0, 'phi')[0]
        assert rp['c'] == pytest.approx(10.0)
        assert rp['phi'] < 20.0

    def test_phi_zero_undrained(self):
        rp = _reduce_props([_props(c=30.0, phi=0.0)], 1.5, 'c_phi')[0]
        assert rp['phi'] == 0.0
        assert rp['c'] == pytest.approx(20.0)

    def test_hs_stiffness_params_untouched(self):
        """HS coverage: only strengths reduced, stiffness unchanged."""
        mp = _props(model='hs', E50_ref=3e4, Eur_ref=9e4, m=0.5,
                    p_ref=100.0, R_f=0.9)
        rp = _reduce_props([mp], 2.0, 'c_phi')[0]
        assert rp['c'] == pytest.approx(5.0)
        for k in ('E50_ref', 'Eur_ref', 'm', 'p_ref', 'R_f', 'model'):
            assert rp[k] == mp[k]


# ============================================================================
# Shared small slope (kept tiny — factorized once per SRM call)
# ============================================================================

SURFACE = [(0.0, 6.0), (6.0, 6.0), (18.0, 0.0), (24.0, 0.0)]
LAYER = {'name': 'soil', 'bottom_elevation': -6.0,
         'E': 1e5, 'nu': 0.3, 'c': 8.0, 'phi': 20.0, 'psi': 0.0,
         'gamma': 20.0}


@pytest.fixture(scope='module')
def srm_cst():
    return analyze_slope_srm(SURFACE, [LAYER], nx=12, ny=6,
                             srf_tol=0.05, element_type='cst')


def _tiny_mesh():
    nodes, elements = generate_slope_mesh(SURFACE, depth=6.0, nx=10, ny=5)
    props = [{'E': 1e5, 'nu': 0.3, 'c': 8.0, 'phi': 20.0, 'psi': 0.0,
              'gamma': 20.0}]
    return nodes, elements, props


# ============================================================================
# Result structure / SRF curve
# ============================================================================

class TestSRMOutputs:

    def test_fos_reasonable(self, srm_cst):
        assert srm_cst.FOS is not None
        assert 0.5 < srm_cst.FOS <= 3.0

    def test_history_fields(self, srm_cst):
        hist = srm_cst.srf_history
        assert len(hist) >= 3
        for h in hist:
            for key in ('srf', 'max_disp_m', 'dimensionless_disp',
                        'converged', 'failed', 'n_iter'):
                assert key in h
            assert np.isfinite(h['dimensionless_disp'])

    def test_srf_curve(self, srm_cst):
        srf, dim = srm_cst.srf_curve
        assert len(srf) == len(dim) >= 2
        assert np.all(np.isfinite(srf)) and np.all(np.isfinite(dim))
        # Curve only contains converged trials
        assert np.all(srf <= srm_cst.FOS + 0.05)

    def test_fos_basis(self, srm_cst):
        assert srm_cst.fos_basis in (
            'nonconvergence', 'blowup', 'range_exhausted')

    def test_displacement_grows_toward_failure(self, srm_cst):
        """GL99 Fig. 2: dimensionless displacement rises near the FOS."""
        srf, dim = srm_cst.srf_curve
        order = np.argsort(srf)
        assert dim[order][-1] >= dim[order][0]


# ============================================================================
# Failure detection options
# ============================================================================

class TestFailureDetection:

    def test_blowup_disabled_matches_nonconvergence(self):
        """blowup_factor=None: pure GL99 non-convergence criterion."""
        nodes, elements, props = _tiny_mesh()
        r_nc = strength_reduction(nodes, elements, props, 20.0,
                                  tol=0.05, blowup_factor=None)
        assert r_nc['fos_basis'] in ('nonconvergence', 'range_exhausted')
        r_def = strength_reduction(nodes, elements, props, 20.0, tol=0.05)
        # Blowup can only LOWER the detected FOS (extra failure mode)
        assert r_def['FOS'] <= r_nc['FOS'] + 0.051
        assert abs(r_def['FOS'] - r_nc['FOS']) < 0.35

    def test_bisection_tolerance_respected(self):
        nodes, elements, props = _tiny_mesh()
        res = strength_reduction(nodes, elements, props, 20.0, tol=0.05)
        assert res['converged']
        # Final bracket width <= tol: last stable and first failed trial
        # around the FOS differ by no more than tol
        stable = [h['srf'] for h in res['srf_history'] if not h['failed']]
        failed = [h['srf'] for h in res['srf_history'] if h['failed']]
        if stable and failed:
            lo = max(s for s in stable if s <= res['FOS'] + 1e-9)
            hi = min(f for f in failed if f >= res['FOS'] - 1e-9)
            assert hi - lo <= 0.05 + 1e-9

    def test_stall_window_cuts_iteration_short(self):
        """Stall check aborts a trial whose residual stops improving."""
        from fem2d.mesh import detect_boundary_nodes
        from fem2d.solver import build_nl_context, run_nl
        nodes, elements, props = _tiny_mesh()
        bc = detect_boundary_nodes(nodes)
        ctx = build_nl_context(nodes, elements, props, 20.0, bc)
        # Unreachable tol + always-firing stall ratio: aborts at
        # window+1 iterations (stall fires when the recent best residual
        # is not better than stall_ratio x the earlier best)
        res = run_nl(ctx, n_steps=1, max_iter=500, tol=0.0,
                     method='elastic', stall_window=10, stall_ratio=0.0)
        assert not res['converged']
        assert res['n_iter_total'] <= 12

    def test_stall_window_default_off(self):
        """Default: trials run to the full ceiling (GL99 behavior)."""
        import inspect
        sig = inspect.signature(strength_reduction)
        assert sig.parameters['stall_window'].default is None

    def test_srm_field_c_equals_cphi_for_undrained(self):
        """phi=0 soil: reducing 'c' alone == reducing 'c_phi'."""
        nodes, elements, props = _tiny_mesh()
        props_u = [dict(props[0], c=25.0, phi=0.0)]
        r1 = strength_reduction(nodes, elements, props_u, 20.0,
                                tol=0.05, srm_field='c_phi')
        r2 = strength_reduction(nodes, elements, props_u, 20.0,
                                tol=0.05, srm_field='c')
        assert r1['FOS'] == pytest.approx(r2['FOS'], abs=1e-9)


# ============================================================================
# element_type plumbing
# ============================================================================

class TestElementTypePlumbing:

    def test_t6_default_runs(self):
        res = analyze_slope_srm(SURFACE, [LAYER], nx=10, ny=5,
                                srf_tol=0.05)
        assert res.FOS is not None
        assert 0.5 < res.FOS <= 3.0
        assert res.fos_basis in (
            'nonconvergence', 'blowup', 'range_exhausted')

    def test_t6_not_stiffer_than_cst(self, srm_cst):
        """T6 must not report a HIGHER FOS than locked CST (same mesh)."""
        res_t6 = analyze_slope_srm(SURFACE, [LAYER], nx=12, ny=6,
                                   srf_tol=0.05, element_type='t6')
        assert res_t6.FOS <= srm_cst.FOS + 0.15
