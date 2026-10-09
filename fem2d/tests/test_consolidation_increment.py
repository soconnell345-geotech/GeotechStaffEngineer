"""Consolidation under a load increment — live smoke G6 (CON-1), 2026-10-09.

Before the fix:
- the monolithic solve applied self-weight AND the load undrained at t = 0,
  so p0 carried the self-weight response (386 kPa under a 100 kPa load on a
  20 m column) and U was the dissipation of both;
- the drained boundary took the head (m) as a pressure (kPa): gwt = 20 m gave
  20 kPa at the surface instead of gamma_w * 20 = 196 kPa;
- the staggered flow acted on the TOTAL pore pressure with no elevation term
  (and k in m/s used as a mobility), so a hydrostatic field "consolidated"
  away and produced settlement under no load change;
- U came back only at the final time, and the first requested time was taken
  as the loading instant (one time point always gave U = 0);
- the solver stepped directly on the requested output times, so a sparse
  list (1e3, 1e7, 1e8 s) was integrated in three giant steps.

Closed form: Terzaghi 1-D, drained top, impermeable base,
  c = k_mob / S,  S = 1/M + 1/M_oed  (fem2d's storage, alpha = 1),
  p0 = M / (M_oed + M) * q             (undrained Biot response, alpha = 1),
  U(Tv) = 1 - sum 2/m^2 exp(-m^2 Tv),  m = (2j+1) pi / 2,
  p(z,t)/p0 = sum 2/m sin(m Z) exp(-m^2 Tv),  Z = depth / H (0 at the
  drained top, 1 at the impermeable base).
Tolerances follow fem2d/VALIDATION.md section 5 (Biot u-p: the FLAC < 5%
envelope; Crank-Nicolson reaches < 1% on the base pressure).
"""

import math

import numpy as np
import pytest

from fem2d import analyze_consolidation
from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes
from fem2d.porewater import solve_consolidation, compute_pore_pressures

# The V-023 / CON-1 column in fem2d units (kPa, m, s).
K, G, M = 5e5, 2e5, 4e6
E = 9 * K * G / (3 * K + G)
NU = (3 * K - 2 * G) / (2 * (3 * K + G))
M_OED = K + 4 * G / 3
Q = 100.0
H = 20.0
S = 1.0 / M + 1.0 / M_OED
P0 = M / (M_OED + M) * Q                     # 83.92 kPa


def _U_terzaghi(T, n=200):
    return 1.0 - sum(2.0 / ((2 * j + 1) * math.pi / 2) ** 2
                     * math.exp(-((2 * j + 1) * math.pi / 2) ** 2 * T)
                     for j in range(n))


def _p_ratio_terzaghi(Z, T, n=200):
    return sum(2.0 / ((2 * j + 1) * math.pi / 2)
               * math.sin((2 * j + 1) * math.pi / 2 * Z)
               * math.exp(-((2 * j + 1) * math.pi / 2) ** 2 * T)
               for j in range(n))


def _column(k_mob, time_points, gamma=0.0, gwt=0.0, theta=0.5, ny=40):
    return analyze_consolidation(
        width=2.0, depth=H,
        soil_layers=[{"E": E, "nu": NU, "gamma": gamma,
                      "bottom_elevation": -H}],
        k=k_mob, load_q=Q, time_points=time_points, gwt=gwt,
        n_w=M, nx=2, ny=ny, consolidation_scheme="monolithic", theta=theta)


class TestCON1Regression:
    """The live CON-1 input: E = 529412 kPa, nu = 0.3235, M = 4e6 kPa,
    mobility 1e-10 m2/(kPa.s), 100 kPa, gwt = 20, gamma = 18."""

    def _con1(self, time_points, gwt=20.0, gamma=18.0, theta=0.5):
        return analyze_consolidation(
            width=2.0, depth=20.0,
            soil_layers=[{"E": 529412, "nu": 0.3235, "gamma": gamma,
                          "bottom_elevation": 0.0}],
            k=1e-10, load_q=100.0, time_points=time_points, gwt=gwt,
            n_w=4e6, nx=4, ny=40, consolidation_scheme="monolithic",
            theta=theta)

    def test_p0_is_the_load_response_not_386(self):
        r = self._con1([1e3, 1e7, 1e8])
        p0 = np.asarray(r.excess_pore_pressures[0])
        # Interior undrained excess = M/(M_oed + M) q = 83.9 kPa (<= q),
        # never the 386 kPa self-weight-contaminated value.
        assert np.median(p0) == pytest.approx(P0, rel=0.01)
        assert r.max_excess_pore_pressure_kPa < 1.25 * Q

    def test_self_weight_and_water_table_do_not_enter_the_increment(self):
        a = self._con1([1e5], gwt=20.0, gamma=18.0)
        b = self._con1([1e5], gwt=0.0, gamma=0.0)
        np.testing.assert_allclose(a.excess_pore_pressures,
                                   b.excess_pore_pressures, atol=1e-6)
        np.testing.assert_allclose(a.settlements, b.settlements, atol=1e-9)

    def test_U_per_time_matches_terzaghi(self):
        r = self._con1([1e3, 1e7, 1e8])
        d = r.to_dict()
        c = 1e-10 / S
        assert d["time_s"] == [0.0, 1e3, 1e7, 1e8]
        U = d["degree_of_consolidation_by_time"]
        assert U[0] == 0.0
        assert U[2] == pytest.approx(_U_terzaghi(c * 1e7 / 400.0), abs=0.01)
        assert U[3] == pytest.approx(1.0, abs=1e-3)
        # The final value is where it was, and agrees with the history.
        assert d["degree_of_consolidation"] == U[-1]
        # Settlement per time ends at the drained value q H / M_oed.
        assert d["final_drained_settlement_m"] == pytest.approx(
            -100.0 * 20.0 / 766_670.0, rel=0.01)
        assert d["surface_settlement_m_by_time"][-1] == pytest.approx(
            d["final_drained_settlement_m"], rel=0.01)

    def test_one_time_point_is_an_elapsed_time(self):
        """A single requested time is measured from loading, not taken as
        the loading instant (it used to give U = 0 every time)."""
        r = self._con1([1e7], theta=1.0)
        d = r.to_dict()
        assert d["time_s"] == [0.0, 1e7]
        assert d["degree_of_consolidation"] > 0.95


class TestTerzaghiValidation:
    """fem2d/VALIDATION.md section 5 problem through the public wrapper,
    sparse output times (the sub-stepped schedule does the integration)."""

    K_MOB = 1e-7                       # m2/(kPa.s)
    C = K_MOB / S                      # 0.0643 m2/s

    def test_average_U_vs_tv(self):
        tv = [0.05, 0.1, 0.2, 0.5, 1.0]
        r = _column(self.K_MOB, [T * H ** 2 / self.C for T in tv])
        U = r.to_dict()["degree_of_consolidation_by_time"][1:]
        for T, u in zip(tv, U):
            assert u == pytest.approx(_U_terzaghi(T), abs=0.02), (
                f"Tv={T}: FE U={u:.4f} vs Terzaghi {_U_terzaghi(T):.4f}")

    def test_isochrones(self):
        tv = [0.05, 0.2, 0.5]
        r = _column(self.K_MOB, [T * H ** 2 / self.C for T in tv])
        nodes, _ = generate_rect_mesh(0, 2.0, -H, 0, 2, 40)
        col = np.where(np.abs(nodes[:, 0]) < 1e-6)[0]
        p = np.asarray(r.excess_pore_pressures)
        p0 = P0
        for i, T in enumerate(tv, start=1):
            for n in col:
                depth = -nodes[n, 1]
                if depth < 2.0:          # skip the drained boundary layer
                    continue
                Z = depth / H
                an = _p_ratio_terzaghi(Z, T)
                fe = p[i][n] / p0
                assert abs(fe - an) < 0.05, (
                    f"Tv={T} z={depth:.1f}: FE {fe:.3f} vs {an:.3f}")


class TestHeadBoundaryUnits:
    """Drained boundaries are total heads (m) -> u = gamma_w (h - z)."""

    def _setup(self, gwt):
        nodes, elements = generate_rect_mesh(0, 2.0, -10.0, 0, 3, 10)
        bc = detect_boundary_nodes(nodes)
        top = np.where(np.abs(nodes[:, 1]) < 0.01)[0]
        pp0 = compute_pore_pressures(nodes, gwt=gwt, gamma_w=9.81)
        return nodes, elements, bc, top, pp0

    def test_staggered_surface_pressure_is_gamma_w_times_head(self):
        nodes, elements, bc, top, pp0 = self._setup(gwt=20.0)
        r = solve_consolidation(
            nodes, elements, [{"E": 10000, "nu": 0.3}], 18.0, bc,
            k=1e-6, head_bcs=[(int(n), 20.0) for n in top],
            time_steps=[0, 1e3, 1e5], pore_pressures_0=pp0)
        p = np.asarray(r["pore_pressures"])
        # 20 m of water above the surface: 196.2 kPa, not 20.
        assert p[-1][top] == pytest.approx(9.81 * 20.0, rel=1e-6)

    def test_staggered_hydrostatic_field_does_not_dissipate(self):
        """A hydrostatic field is equilibrium: no excess, no flow, no
        settlement change over time (it used to drain to the surface value
        and add ~50 % to the settlement)."""
        nodes, elements, bc, top, pp0 = self._setup(gwt=0.0)
        r = solve_consolidation(
            nodes, elements, [{"E": 10000, "nu": 0.3}], 18.0, bc,
            k=1e-6, head_bcs=[(int(n), 0.0) for n in top],
            time_steps=[0, 1e3, 1e5, 1e7], pore_pressures_0=pp0)
        p = np.asarray(r["pore_pressures"])
        np.testing.assert_allclose(p[-1], pp0, atol=1e-6)
        assert r["max_excess_pore_pressure_kPa"] == pytest.approx(0.0,
                                                                  abs=1e-6)
        s = np.asarray(r["settlements"])
        assert s[-1] == pytest.approx(s[0], rel=1e-9)

    def test_negative_time_rejected(self):
        nodes, elements, bc, top, pp0 = self._setup(gwt=0.0)
        with pytest.raises(ValueError, match=">= 0"):
            solve_consolidation(
                nodes, elements, [{"E": 10000, "nu": 0.3}], 18.0, bc,
                k=1e-6, head_bcs=[(int(n), 0.0) for n in top],
                time_steps=[-1.0, 10.0], pore_pressures_0=pp0)


def test_staggered_result_says_it_has_no_transient():
    r = analyze_consolidation(
        width=2.0, depth=5.0,
        soil_layers=[{"E": 10000, "nu": 0.3, "gamma": 18}],
        k=1e-5, load_q=50.0, time_points=[100, 1000], nx=4, ny=8)
    d = r.to_dict()
    assert d["scheme"] == "staggered"
    assert d["degree_of_consolidation_by_time"] == [1.0, 1.0, 1.0]
    assert any("monolithic" in n for n in d["notes"])
