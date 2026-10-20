"""Crank-Nicolson start-up (BACKLOG GM-07) and fill placed under water
(GM-09), 2026-10-20.

GM-07. With theta = 0.5 the undrained excess pore pressure, which jumps
to 0 at the drained boundary at t = 0, rang at the drained boundary for the
rest of the analysis whenever the first step was long next to the boundary
element's h^2 / c: Crank-Nicolson damps those components by
(1 - lambda dt / 2) / (1 + lambda dt / 2) -> -1 per step. The fix is
Rannacher's start-up (Rannacher 1984): the first steps are each taken as two
fully implicit half-steps, which keeps second-order accuracy.

GM-09. Fill flagged in the consolidation column was refused below the water
table. Now the water standing on the soil in place before placement is part
of the initial stage and the fill loads the soil by its buoyant unit weight
gamma - gamma_w below the water table (FHWA NHI-06-088, 2006, Eq. 2-7,
p. 2-6). Hand checks: initial stage gamma'_clay H_c^2 / (2 M); load stage
drained q (H_c + h_f) / M + (fill self-compression + fill load x H_c) / M.
"""

import math

import numpy as np
import pytest

from fem2d import analyze_consolidation
from fem2d.mesh import generate_rect_mesh, detect_boundary_nodes
from fem2d.porewater import solve_consolidation

from fem2d.tests.test_consolidation_increment import (
    E, NU, M, Q, H, S, P0, _U_terzaghi, _p_ratio_terzaghi)

K_MOB = 1e-10
C_V = K_MOB / S
GW = 9.81


def _column(Tv, startup_steps=2, theta=0.5, ny=40):
    return analyze_consolidation(
        width=2.0, depth=H,
        soil_layers=[{"E": E, "nu": NU, "gamma": 0.0,
                      "bottom_elevation": -H}],
        k=K_MOB, load_q=Q, time_points=np.asarray(Tv) * H ** 2 / C_V,
        n_w=M, nx=2, ny=ny, consolidation_scheme="monolithic",
        theta=theta, startup_steps=startup_steps)


def _depths(ny=40):
    nodes, _ = generate_rect_mesh(0, 2.0, -H, 0, 2, ny)
    return -nodes[:, 1]


class TestCrankNicolsonStartup:
    TV = [1.0, 2.0, 4.0, 16.0]

    def test_ringing_before_and_gone_after(self):
        """Output times from Tv = 1: the first internal step is Tv 0.01,
        some 16 times the boundary element's h^2 / c. Without the start-up
        the excess goes negative and stays (-1.4 kPa at Tv = 16, U = 1);
        with it the excess never goes below zero and is 0 at Tv = 16."""
        old = np.asarray(_column(self.TV, startup_steps=0)
                         .excess_pore_pressures)
        new = np.asarray(_column(self.TV).excess_pore_pressures)
        assert old[-1].min() < -1.0                      # the ringing
        assert new[1:].min() > -1e-4                  # round-off
        assert np.abs(new[-1]).max() < 1e-3

    def test_single_late_output_time(self):
        """One output time (the G6 / fill-column case): the internal
        schedule's first step is two decades below it. Before: +/-41.7 kPa
        left at U = 1; after: none."""
        r_old = analyze_consolidation(
            width=2.0, depth=20.0,
            soil_layers=[{"E": 529412, "nu": 0.3235, "gamma": 18.0,
                          "bottom_elevation": 0.0}],
            k=1e-10, load_q=100.0, time_points=[1e8], n_w=4e6, nx=4,
            ny=40, consolidation_scheme="monolithic", theta=0.5,
            startup_steps=0)
        r_new = analyze_consolidation(
            width=2.0, depth=20.0,
            soil_layers=[{"E": 529412, "nu": 0.3235, "gamma": 18.0,
                          "bottom_elevation": 0.0}],
            k=1e-10, load_q=100.0, time_points=[1e8], n_w=4e6, nx=4,
            ny=40, consolidation_scheme="monolithic", theta=0.5)
        assert np.abs(np.asarray(r_old.excess_pore_pressures)[-1]).max() > 30
        assert np.abs(np.asarray(r_new.excess_pore_pressures)[-1]).max() \
            < 1e-3

    @pytest.mark.parametrize("Tv_list", [[0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
                                         [1.0, 2.0, 4.0, 16.0]])
    def test_terzaghi_U_and_isochrones(self, Tv_list):
        """U against Terzaghi's closed form within 0.006 (the T6/T3
        drained-boundary layer at the smallest Tv), the isochrones
        p(z, t) within 0.1 kPa of the series (P0 = 83.9 kPa) below 0.5 m."""
        r = _column(Tv_list)
        U = r.to_dict()["degree_of_consolidation_by_time"]
        ex = np.asarray(r.excess_pore_pressures)
        z = _depths()
        for i, T in enumerate(Tv_list):
            assert U[i + 1] == pytest.approx(_U_terzaghi(T), abs=0.006)
            p_t = np.array([_p_ratio_terzaghi(zz / H, T) for zz in z]) * P0
            assert np.abs(ex[i + 1] - p_t)[z >= 0.5].max() < 0.1

    def test_second_order_with_startup(self):
        """Uniform steps to Tv = 0.3, error at the end against 2,560 steps:
        halving dt quarters the error with the start-up (ratio 4.0); plain
        Crank-Nicolson at 10 steps is 28 kPa off from the ringing."""
        nodes, el = generate_rect_mesh(0, 2.0, -H, 0, 2, 20)
        bc = detect_boundary_nodes(nodes)
        top = np.where(np.abs(nodes[:, 1]) < 1e-6)[0]
        top = top[np.argsort(nodes[top, 0])]
        edges = [(top[i], top[i + 1]) for i in range(len(top) - 1)]
        T = 0.3 * H ** 2 / C_V

        def run(n, su):
            r = solve_consolidation(
                nodes, el, [{"E": E, "nu": NU}], 0.0, bc, k=K_MOB,
                head_bcs=[(int(t), 0.0) for t in top],
                time_steps=np.linspace(0, T, n + 1), n_w=M,
                surface_loads=[(edges, 0.0, -Q)], scheme="monolithic",
                theta=0.5, startup_steps=su)
            return np.asarray(r['pore_pressures'])[-1]

        ref = run(2560, 2)
        e = [np.abs(run(n, 2) - ref).max() for n in (10, 20, 40)]
        assert e[0] / e[1] == pytest.approx(4.0, rel=0.05)
        assert e[1] / e[2] == pytest.approx(4.0, rel=0.05)
        assert np.abs(run(10, 0) - ref).max() > 10.0

    def test_backward_euler_unaffected(self):
        a = np.asarray(_column([0.1, 1.0], theta=1.0).excess_pore_pressures)
        b = np.asarray(_column([0.1, 1.0], theta=1.0, startup_steps=0)
                       .excess_pore_pressures)
        np.testing.assert_array_equal(a, b)

    def test_choice_stated(self):
        ch = {c["what"]: c for c in _column([1.0]).to_dict()["choices"]}
        su = [c for w, c in ch.items() if w.startswith("time-stepping")][0]
        assert su["chosen"] == 2 and "Rannacher" in su["basis"]
        assert not any(w.startswith("time-stepping") for w in
                       {c["what"] for c in _column([1.0], theta=1.0)
                        .to_dict()["choices"]})


# ---------------------------------------------------------------------------
# GM-09: fill placed under water
# ---------------------------------------------------------------------------

E_C, NU_C = 529412.0, 0.3235
M_OED = E_C * (1 - NU_C) / ((1 + NU_C) * (1 - 2 * NU_C))
M_W = 4e6


def _fill_column(gwt, scheme="monolithic"):
    fill = {"name": "fill", "E": E_C, "nu": NU_C, "gamma": 20.0,
            "bottom_elevation": -2.0, "fill": True}
    clay = {"name": "clay", "E": E_C, "nu": NU_C, "gamma": 18.0,
            "bottom_elevation": -20.0}
    return analyze_consolidation(
        width=2.0, depth=20.0, soil_layers=[fill, clay], k=1e-10,
        load_q=Q, time_points=[1e8], gwt=gwt, n_w=M_W, nx=4, ny=40,
        consolidation_scheme=scheme, theta=0.5)


# (gwt, load on the clay from the fill (kPa), integral of the fill's own
# effective stress over its 2 m (kPa m)); fill gamma 20, clay 18.
_GB = 20.0 - GW
_CASES = [
    (1.0, 2 * _GB, 2 * _GB),            # under 1 m of ponded water
    (0.0, 2 * _GB, 2 * _GB),            # water at the fill's top
    (-1.0, 20.0 + _GB, 10.0 + 20.0 + _GB / 2),   # top metre dry
    (-2.0, 40.0, 40.0),                 # in the dry (as before)
]


class TestFillUnderWater:
    @pytest.mark.parametrize("scheme", ["monolithic", "staggered"])
    @pytest.mark.parametrize("gwt,on_clay,self_int", _CASES)
    def test_hand_check(self, scheme, gwt, on_clay, self_int):
        d = _fill_column(gwt, scheme).to_dict()
        s0 = (18.0 - GW) * 18.0 ** 2 / (2 * M_OED)
        s_load = (Q * 20.0 + self_int + on_clay * 18.0) / M_OED
        assert d['stages'][0]['surface_settlement_m'] == pytest.approx(
            -s0, rel=0.005)
        assert d['stages'][1]['surface_settlement_m'] == pytest.approx(
            -s_load, rel=0.005)

    @pytest.mark.parametrize("gwt,on_clay,_", _CASES)
    def test_undrained_excess_in_the_clay(self, gwt, on_clay, _):
        """p0 deep in the clay = M / (M_oed + M) (q + fill load)."""
        r = _fill_column(gwt)
        nodes, _e = generate_rect_mesh(0, 2.0, -20.0, 0, 4, 40)
        p0 = np.asarray(r.excess_pore_pressures)[0]
        assert np.median(p0[nodes[:, 1] < -10]) == pytest.approx(
            M_W / (M_OED + M_W) * (Q + on_clay), rel=1e-6)

    def test_water_level_and_buoyant_weight_in_choices(self):
        d = _fill_column(1.0).to_dict()
        ch = {c["what"]: c for c in d["choices"]}
        w = ch["water table elevation gwt (m)"]
        assert w["chosen"] == 1.0 and w["how"] == "given"
        assert "ponded" in w["basis"]
        f = ch["fill unit weight below the water table"]
        assert f["chosen"] == f"{20.0 - GW:.2f} kN/m3"
        assert "Eq. 2-7, p. 2-6" in f["basis"]
        assert any("under water" in n for n in d["notes"])

    def test_dry_fill_choice(self):
        ch = {c["what"]: c for c in _fill_column(-2.0).to_dict()["choices"]}
        assert ch["fill unit weight"]["chosen"] == "total unit weight gamma"
        assert "fill unit weight below the water table" not in ch
