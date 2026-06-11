"""
Tests for the 3D principal-stress Mohr-Coulomb return mapping
(materials.mc_return_principal) and the vectorized solver core paths.
"""

import numpy as np
import pytest

from fem2d.materials import (
    mc_return_principal, mc_return_mapping, elastic_D,
)
from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes, convert_to_t6
from fem2d.solver import solve_nonlinear


def _f_mc(sig4, c, phi_deg):
    """MC yield value from full principal decomposition (tension-positive)."""
    sig4 = np.atleast_2d(sig4)
    c = np.broadcast_to(np.asarray(c, dtype=float), (len(sig4),))
    phi = np.radians(np.broadcast_to(np.asarray(phi_deg, dtype=float),
                                     (len(sig4),)))
    p = 0.5 * (sig4[:, 0] + sig4[:, 1])
    R = np.sqrt((0.5 * (sig4[:, 0] - sig4[:, 1])) ** 2 + sig4[:, 3] ** 2)
    sv = np.sort(np.column_stack([p + R, p - R, sig4[:, 2]]), axis=1)
    s1, s3 = sv[:, 2], sv[:, 0]
    return (s1 - s3) + (s1 + s3) * np.sin(phi) - 2 * c * np.cos(phi)


class TestMCReturnPrincipal:
    def test_elastic_state_untouched(self):
        sig = np.array([[-10.0, -20.0, -9.0, 3.0]])
        s, Dep, y, reg = mc_return_principal(sig, 3e4, 0.3, 100.0, 30.0)
        assert not y[0] and reg[0] == 0
        assert np.allclose(s, sig)
        assert np.allclose(Dep[0], elastic_D(3e4, 0.3))

    def test_elastic_tangent_rotation_invariant(self):
        """Elastic Dep equals the elastic D for any principal angle."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            sig = np.array([rng.uniform(-50, 0, 4)])
            sig[0, 3] = rng.uniform(-20, 20)
            s, Dep, y, _ = mc_return_principal(sig, 2e4, 0.25, 500.0, 35.0)
            if not y[0]:
                assert np.allclose(Dep[0], elastic_D(2e4, 0.25), atol=1e-6)

    def test_random_returns_land_on_surface(self):
        """Property test: f(returned) <= tol, all regions valid."""
        rng = np.random.default_rng(0)
        N = 5000
        sig = np.column_stack([
            rng.uniform(-500, 200, N), rng.uniform(-500, 200, N),
            rng.uniform(-500, 200, N), rng.uniform(-200, 200, N)])
        c = rng.uniform(0.1, 50, N)
        phi = rng.uniform(0, 45, N)
        psi = np.minimum(phi, rng.uniform(0, 20, N))
        s_new, Dep, y, reg = mc_return_principal(sig, 3e4, 0.3, c, phi, psi)
        f_after = _f_mc(s_new, c, phi)
        assert f_after.max() < 1e-6
        # all four plastic regions exercised by this sample
        assert set(np.unique(reg[y])) == {1, 2, 3, 4}
        # non-yielded untouched
        assert np.allclose(s_new[~y], sig[~y])

    def test_szz_governed_case_old_model_missed(self):
        """sxx=syy compressive, szz near zero: szz is the major principal
        stress. The old in-plane return sees q_inplane = 0 and misses
        yield entirely; the 3D return correctly yields."""
        sig = np.array([[-100.0, -100.0, -1.0, 0.0]])
        c, phi = 10.0, 30.0
        assert _f_mc(sig, c, phi)[0] > 0  # truly at yield

        # old in-plane model: no yield detected (the defect)
        _, _, yielded_old = mc_return_mapping(
            np.array([-100.0, -100.0, 0.0]), 3e4, 0.3, c, phi)
        assert not yielded_old

        # new model: yields and returns to the surface
        s_new, _, y, reg = mc_return_principal(sig, 3e4, 0.3, c, phi)
        assert y[0] and reg[0] in (2, 3)
        assert abs(_f_mc(s_new, c, phi)[0]) < 1e-8

    def test_matches_old_return_when_szz_intermediate(self):
        """psi=0, szz strictly intermediate: the 3D return reduces to the
        in-plane return (regression against the legacy model)."""
        sig4 = np.array([[-50.0, -150.0, -100.0, 20.0]])
        s_new, _, y, reg = mc_return_principal(sig4, 3e4, 0.3, 5.0, 25.0, 0.0)
        s_old, _, y_old = mc_return_mapping(
            np.array([-50.0, -150.0, 20.0]), 3e4, 0.3, 5.0, 25.0, 0.0)
        assert y[0] and y_old and reg[0] == 1
        assert np.allclose(s_new[0, [0, 1, 3]], s_old, rtol=1e-10)
        assert np.isclose(s_new[0, 2], -100.0)  # szz untouched (psi=0 main)

    def test_apex_return(self):
        sig = np.array([[50.0, 60.0, 55.0, 1.0]])
        c, phi = 10.0, 30.0
        s_new, _, y, reg = mc_return_principal(sig, 3e4, 0.3, c, phi)
        apex = c * np.cos(np.radians(phi)) / np.sin(np.radians(phi))
        assert reg[0] == 4
        assert np.allclose(s_new[0, :3], apex)
        assert np.isclose(s_new[0, 3], 0.0)

    def test_tresca_phi_zero(self):
        """phi=0 (undrained): return caps s1-s3 at 2*cu; no apex."""
        cu = 25.0
        sig = np.array([[-10.0, -110.0, -60.0, 0.0]])
        s_new, _, y, reg = mc_return_principal(sig, 1e4, 0.49, cu, 0.0)
        assert y[0] and reg[0] != 4
        assert np.isclose(_f_mc(s_new, cu, 0.0)[0], 0.0, atol=1e-9)
        # mean stress preserved for the Tresca main-plane return
        assert np.isclose(s_new[0, :3].sum(), sig[0, :3].sum())

    def test_dilation_changes_return_direction(self):
        """psi=0 preserves mean stress on the main plane; psi > 0 has a
        volumetric flow component (return moves p toward compression in
        stress space — confined dilation builds compression)."""
        sig = np.array([[-50.0, -250.0, -150.0, 0.0]])
        s0, _, _, _ = mc_return_principal(sig, 3e4, 0.3, 5.0, 30.0, 0.0)
        s1, _, _, _ = mc_return_principal(sig, 3e4, 0.3, 5.0, 30.0, 15.0)
        p_tr = sig[0, :3].mean()
        p0 = s0[0, :3].mean()
        p1 = s1[0, :3].mean()
        assert np.isclose(p0, p_tr)  # non-dilative: p preserved
        assert p1 < p0               # dilative: return has -mean component


class TestSolverCoreT6:
    """Nonlinear solver with the new GP core on T6 elements."""

    def _setup(self):
        nodes, elements = generate_rect_mesh(0, 10, -5, 0, 10, 5)
        n6, e6 = convert_to_t6(nodes, elements)
        bc = detect_boundary_nodes(n6)
        return n6, e6, bc

    def test_t6_mc_gravity_converges(self):
        n6, e6, bc = self._setup()
        mp = [{'E': 3e4, 'nu': 0.3, 'c': 20.0, 'phi': 25.0, 'psi': 0.0}]
        conv, u, s, e = solve_nonlinear(n6, e6, mp, 19.0, bc, n_steps=5)
        assert conv
        # K0 stress state: syy ~ gamma*z at element centroids
        assert s[:, 1].min() < -80.0

    def test_elastic_and_tangent_methods_agree(self):
        n6, e6, bc = self._setup()
        mp = [{'E': 3e4, 'nu': 0.3, 'c': 10.0, 'phi': 20.0, 'psi': 0.0}]
        conv1, u1, s1, _ = solve_nonlinear(
            n6, e6, mp, 19.0, bc, n_steps=5, method='elastic')
        conv2, u2, s2, _ = solve_nonlinear(
            n6, e6, mp, 19.0, bc, n_steps=5, method='tangent')
        assert conv1 and conv2
        assert np.allclose(u1, u2, atol=5e-5 * max(np.abs(u1).max(), 1e-12) /
                           1e-3)  # within ~5% of max displacement
        assert np.allclose(np.abs(u1).max(), np.abs(u2).max(), rtol=0.05)

    def test_return_gp_exposes_szz(self):
        n6, e6, bc = self._setup()
        mp = [{'E': 3e4, 'nu': 0.3, 'c': 20.0, 'phi': 25.0, 'psi': 0.0}]
        out = solve_nonlinear(n6, e6, mp, 19.0, bc, n_steps=3,
                              return_gp=True)
        conv, u, s, e, sig_gp, eps_gp = out
        assert sig_gp.shape == (len(e6), 3, 4)
        # szz compressive below surface, between nu*(sxx+syy) bounds
        assert sig_gp[:, :, 2].min() < -10.0

    def test_q4_still_supported(self):
        # Build a small structured Q4 mesh manually
        x = np.linspace(0, 4, 5)
        y = np.linspace(-2, 0, 3)
        xx, yy = np.meshgrid(x, y)
        nodes = np.column_stack([xx.ravel(), yy.ravel()])
        elems = []
        for j in range(2):
            for i in range(4):
                n0 = j * 5 + i
                elems.append([n0, n0 + 1, n0 + 6, n0 + 5])
        elems = np.array(elems)
        bc = detect_boundary_nodes(nodes)
        mp = [{'E': 2e4, 'nu': 0.3, 'c': 15.0, 'phi': 20.0, 'psi': 0.0}]
        conv, u, s, e = solve_nonlinear(nodes, elems, mp, 18.0, bc, n_steps=3)
        assert conv
        assert s.shape == (8, 3)
