"""The seepage flow is a flow through a section (GM-222, 2026-11-05).

``analyze_seepage`` reports ``flow_m3_per_s_per_m``: the inflow across the
fixed-head boundaries per metre run, from the nodal reactions of the flow
equations at the constrained-head nodes, with inflow = outflow checked in
``flow_balance``. Before, it reported ``total_flow_m3_per_s_per_m`` =
sum of |v| x A x t over the elements, which grew with the slice thickness
t and is not the flow through any section.

Checks:

1. One-dimensional Darcy flow, exact: q = k i y (UFS 3-220-10, 16 January
   2026, Eq. 6-4, p. 286).
2. The printed flow-net example: UFS 3-220-10 (2026) Figure 6-5 "Flow Net
   Example Calculations", p. 292, on the flow net of Figure 6-3 (p. 288):
   an impervious dam embedded 20 ft in 40 ft of isotropic permeable soil
   on an impervious base, head loss 36 ft, k = 1e-3 cm/s, N_f = 3,
   N_d = 9: q = k h_L (N_f/N_d) W (Eq. 6-9, p. 290) = 34 ft^3/day per ft.
   The dam's width is not printed; it scales 20 ft off Figure 6-3 (the
   embedment, printed 20 ft, scales 19.6 ft at the same scale), and the
   soil is drawn about 60 ft either side of the dam's centre. Accepted
   within 10 %: the printed q comes from a hand-drawn net counted in whole
   flow channels (3), and the unprinted width moves the FEM flow by about
   7 % for +-4 ft. The FEM gives 32.4 ft^3/day per ft (-5 %).
3. No result depends on t; a prescribed inflow leaves at the fixed heads.
"""

import numpy as np
import pytest

from fem2d.analysis import analyze_seepage
from fem2d.mesh import generate_rect_mesh
from fem2d.porewater import FLOW_BALANCE_TOLERANCE, solve_seepage

FT = 0.3048
FT3_PER_DAY_PER_FT = FT ** 2 / 86400.0   # m^3/s per m


def _rect(L, y, nx, ny):
    nodes, els = generate_rect_mesh(0.0, L, 0.0, y, nx, ny)
    return np.asarray(nodes), np.asarray(els)


def _darcy(t=1.0):
    L, y, k, h1, h2 = 12.0, 3.0, 2.0e-5, 8.0, 5.0
    nodes, els = _rect(L, y, 12, 6)
    bcs = ([(i, h1) for i, (x, _) in enumerate(nodes) if x < 1e-9]
           + [(i, h2) for i, (x, _) in enumerate(nodes) if x > L - 1e-9])
    return analyze_seepage(nodes, els, k, bcs, t=t), k * (h1 - h2) / L * y


def _figure_6_3(half=60.0, h=2.0, B=20.0, D=20.0, T=40.0, t=1.0):
    """The soil under the dam of UFS 3-220-10 Figure 6-3, in metres.

    A structured grid (ft) through x = +-B/2 and y = T - D, the dam's
    embedded part removed (its faces impervious), the base and the far
    sides impervious; head 76 ft upstream (datum at the base of the
    permeable soil, as Figure 6-5 sets it) and 40 ft downstream.
    """
    xs = np.arange(-half, half + 1e-9, h)
    ys = np.arange(0.0, T + 1e-9, h)
    nx = len(xs)
    nodes = np.array([[x, y] for y in ys for x in xs])
    els = []
    for j in range(len(ys) - 1):
        for i in range(nx - 1):
            xc = 0.5 * (xs[i] + xs[i + 1])
            yc = 0.5 * (ys[j] + ys[j + 1])
            if abs(xc) < B / 2 and yc > T - D:
                continue
            a, b = j * nx + i, j * nx + i + 1
            c, d = b + nx, a + nx
            els += [[a, b, d], [b, c, d]] if (i + j) % 2 == 0 \
                else [[a, b, c], [a, c, d]]
    els = np.array(els)
    used = np.unique(els)
    remap = -np.ones(len(nodes), dtype=int)
    remap[used] = np.arange(len(used))
    nodes = nodes[used] * FT
    els = remap[els]
    bcs = []
    for n, (x, y) in enumerate(nodes):
        if abs(y - T * FT) < 1e-9:
            if x <= -B / 2 * FT + 1e-9:
                bcs.append((n, 76.0 * FT))
            elif x >= B / 2 * FT - 1e-9:
                bcs.append((n, 40.0 * FT))
    k = 1.0e-3 / 100.0          # 1e-3 cm/s, as printed
    return analyze_seepage(nodes, els, k, bcs, t=t)


class TestDarcy1D:
    def test_flow_is_k_i_y(self):
        r, q = _darcy()
        assert r.flow_m3_per_s_per_m == pytest.approx(q, rel=1e-9)
        assert r.outflow_m3_per_s_per_m == pytest.approx(q, rel=1e-9)

    def test_balance_and_boundaries(self):
        r, q = _darcy()
        fb = r.flow_balance
        assert fb["passed"] is True
        assert fb["mismatch"] < 1e-9
        assert fb["tolerance"] == FLOW_BALANCE_TOLERANCE
        by_head = {b["head_m"]: b for b in r.flow_boundaries}
        assert by_head[8.0]["net_inflow_m3_per_s_per_m"] == \
            pytest.approx(q, rel=1e-9)
        assert by_head[5.0]["net_inflow_m3_per_s_per_m"] == \
            pytest.approx(-q, rel=1e-9)
        assert r.flow_measured_on.startswith("inflow")

    def test_does_not_grow_with_thickness(self):
        r1, _ = _darcy(t=1.0)
        r3, _ = _darcy(t=3.0)
        assert r3.flow_m3_per_s_per_m == pytest.approx(
            r1.flow_m3_per_s_per_m, rel=1e-9)
        np.testing.assert_allclose(r3.head, r1.head, rtol=1e-9)


class TestFigure6_5:
    """UFS 3-220-10 (2026) Figure 6-5, p. 292: q = 34 ft^3/day per ft."""

    PRINTED_Q = 34.0 * FT3_PER_DAY_PER_FT

    @pytest.fixture(scope="class")
    @classmethod
    def result(cls):
        return _figure_6_3()

    def test_flow_against_printed(self, result):
        assert result.flow_m3_per_s_per_m == pytest.approx(
            self.PRINTED_Q, rel=0.10)

    def test_flow_balance(self, result):
        assert result.flow_balance["passed"] is True
        assert result.outflow_m3_per_s_per_m == pytest.approx(
            result.flow_m3_per_s_per_m, rel=1e-9)

    def test_independent_of_thickness(self, result):
        r = _figure_6_3(t=2.5)
        assert r.flow_m3_per_s_per_m == pytest.approx(
            result.flow_m3_per_s_per_m, rel=1e-9)

    def test_to_dict(self, result):
        d = result.to_dict()
        assert "total_flow_m3_per_s_per_m" not in d
        assert d["flow_m3_per_s_per_m"] == pytest.approx(
            result.flow_m3_per_s_per_m, rel=1e-3)
        assert d["flow_balance"]["passed"] is True
        assert len(d["flow_boundaries"]) == 2
        assert d["flow_measured_on"]


def test_prescribed_inflow_leaves_at_the_fixed_heads():
    nodes, els = _rect(10.0, 4.0, 10, 4)
    bcs = [(i, 2.0) for i, (x, _) in enumerate(nodes) if x > 10.0 - 1e-9]
    well = int(np.argmin(np.hypot(nodes[:, 0] - 3.0, nodes[:, 1] - 2.0)))
    Q, t = 1.0e-4, 2.0           # over the slice
    out = solve_seepage(nodes, els, 1e-5, bcs, t=t, flow_bcs=[(well, Q)])
    assert out["inflow"] == pytest.approx(0.0, abs=1e-15)
    assert out["outflow"] == pytest.approx(Q / t, rel=1e-9)
    assert out["prescribed_flow"] == pytest.approx(Q / t, rel=1e-12)
    assert out["flow_balance"]["passed"] is True
    assert out["flow_rate"] == pytest.approx(Q / t, rel=1e-9)
    assert out["flow_measured_on"].startswith("outflow")
