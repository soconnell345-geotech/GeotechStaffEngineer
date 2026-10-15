"""A dewatered excavation: the wall carries the net water pressure and the
water seeps under its tip (GM-73, 2026-10-15).

Until 2026-10-15 ``analyze_excavation`` kept an excavation below the water
table full of water to the table, so the water pressures on the wall's two
faces balanced. ``excavation_water='dewatered'`` (now the default) lowers
the water inside to the excavation level at each lift, holds the table
outside, and takes the pore pressures from the steady-state seepage under
the wall (``porewater.seepage_heads``; the wall impermeable).

Checks:

1. The seepage solution against the closed form for a sheet pile driven
   into a pervious layer on an impervious base (the conformal-mapping
   solution of the Laplace equation, derived below).
2. The benchmark: FHWA GEC 4, Ground Anchors and Anchored Systems
   (FHWA-IF-99-015, 1999), section 5.2.9 "Water Pressures", pp. 62-63,
   Figures 31-32 (after CIRIA 1984): an excavation H deep, embedment d, the
   table j below the ground, the water inside i below the excavation level.
   Assuming the head lost uniformly along the shortest flow path (2d + H -
   i - j, down the back of the wall and up the front):
       U_f = 2 (d + H - j)(d - i) / (2d + H - i - j) gamma_w    (Eq. 15)
       U_c = (H + i - j) 2 (d - i) / (2d + H - i - j) gamma_w   (Eq. 16)
   and with no seepage the net pressure is U_n = (H + i - j) gamma_w
   (Eq. 17). The flow net puts more of the head loss near the tip and in
   front of the wall than the uniform assumption (check 1: at mid-
   penetration the exact head is 0.84 of the difference, the uniform
   assumption 0.75), so the FEM net pressure at the excavation level lies
   between Eq. 16 and Eq. 17, and the pore pressure at the tip above
   Eq. 15. GEC 4 itself says the simplified pressures are "greater than
   hydrostatic in front of the wall and less than hydrostatic behind"
   (p. 62); that is checked too.
3. Consistency: water inside at the table level reproduces 'full'; an
   impervious layer under the tip gives the no-seepage pressure (Eq. 17).
"""

import math

import numpy as np
import pytest
from scipy.integrate import quad

from fem2d import analyze_excavation
from fem2d.analysis import _free_water_on_wall, _graded_coords
from fem2d.mesh import convert_to_t6, generate_rect_mesh
from fem2d.porewater import seepage_heads, solve_seepage

GW = 9.81


# ---------------------------------------------------------------------------
# 1. The seepage solution against the closed form
# ---------------------------------------------------------------------------

def sheet_pile_head(depth, s, T, H):
    """Total head on the upstream face of a sheet pile driven s into a
    pervious layer T thick on an impervious base, infinite both ways, head
    H on the upstream ground and 0 on the downstream ground.

    By antisymmetry the head is H/2 below the tip. In the half-strip
    upstream of the pile, zeta = cosh(pi z / T) (z = x + i y, the ground
    y = 0, the pile on x = 0) maps the flow region onto the upper half
    plane: the upstream ground to (1, inf) (h = H), the pile face to
    (c, 1), c = cos(pi s / T) (no flow), the line below the tip to (-1, c)
    (h = H/2), the base to (-inf, -1) (no flow). The Schwarz-Christoffel
    integral F = int d(tau) / sqrt((tau + 1)(tau - c)(tau - 1)) maps that
    onto a rectangle on which the head is linear, so on the face (zeta =
    cos(pi depth / T)) h = H/2 + (H/2) I(zeta) / I(1), with I(zeta) =
    int_c^zeta d(tau) / sqrt((tau + 1)(tau - c)(1 - tau)); with tau = c +
    (1 - c) sin^2(theta) the integrand is 2 / sqrt(1 + c + (1 - c)
    sin^2(theta)), regular.
    """
    c = math.cos(math.pi * s / T)

    def f(th):
        return 2.0 / math.sqrt(1.0 + c + (1.0 - c) * math.sin(th) ** 2)

    zeta = math.cos(math.pi * depth / T)
    th = math.asin(math.sqrt(min(max((zeta - c) / (1.0 - c), 0.0), 1.0)))
    return H / 2 + H / 2 * quad(f, 0.0, th)[0] / quad(f, 0.0, math.pi / 2)[0]


def _cut_mesh(xs, ys, s, quadratic=True):
    """Structured mesh through xs, ys, cut along x = 0 above y = -s (the
    elements right of it get their own copies of the column's nodes)."""
    idx, el = generate_rect_mesh(0.0, float(len(xs) - 1), 0.0,
                                 float(len(ys) - 1), len(xs) - 1,
                                 len(ys) - 1)
    ix = np.rint(idx[:, 0]).astype(int)
    iy = np.rint(idx[:, 1]).astype(int)
    nodes = np.column_stack([xs[ix], ys[iy]])
    if quadratic:
        nodes, el = convert_to_t6(nodes, el)
    el = np.array(el)
    col = np.where((np.abs(nodes[:, 0]) < 1e-9)
                   & (nodes[:, 1] > -s + 1e-9))[0]
    copy = {int(n): len(nodes) + i for i, n in enumerate(col)}
    nodes = np.vstack([nodes, nodes[col]])
    cx = nodes[el[:, :3]].mean(axis=1)[:, 0]
    for e in np.where(cx > 0)[0]:
        el[e] = [copy.get(int(n), int(n)) for n in el[e]]
    return nodes, el, sorted(col, key=lambda n: -nodes[n, 1]), cx


@pytest.fixture(scope="module")
def sheet_pile():
    s, T, H = 5.0, 10.0, 1.0
    L = 6.0 * T                     # the exact far field to 1e-4 H
    xs = _graded_coords([-L, 0.0, L], 120)
    ys = _graded_coords([-T, -s, 0.0], 40)
    nodes, el, face, cx = _cut_mesh(xs, ys, s)
    up = set(np.unique(el[cx < 0]).tolist())
    down = set(np.unique(el[cx > 0]).tolist())
    bcs = {}
    for n in np.where(np.abs(nodes[:, 1]) < 1e-9)[0]:
        if n in up and n not in down:
            bcs[int(n)] = H
        elif n in down and n not in up:
            bcs[int(n)] = 0.0
    h = seepage_heads(nodes, el, 1.0, bcs)
    return dict(s=s, T=T, H=H, nodes=nodes, h=h, face=face)


def test_seepage_heads_match_the_closed_form(sheet_pile):
    """The head on the upstream face at a quarter, half and three quarters
    of the penetration, and at the tip, within 1 % of the head difference
    (the FEM converges as h^(1/2) only at the two corners where the flow
    turns: the top of the pile and its tip)."""
    sp = sheet_pile
    nodes, h, s = sp['nodes'], sp['h'], sp['s']
    for frac in (0.25, 0.5, 0.75):
        n = min(sp['face'], key=lambda m: abs(-nodes[m, 1] - frac * s))
        exact = sheet_pile_head(-nodes[n, 1], s, sp['T'], sp['H'])
        assert h[n] == pytest.approx(exact, abs=0.01 * sp['H'])
    tip = int(np.where((np.abs(nodes[:, 0]) < 1e-9)
                       & (np.abs(nodes[:, 1] + s) < 1e-9))[0][0])
    assert h[tip] == pytest.approx(0.5 * sp['H'], abs=1e-6)


def test_uniform_head_loss_is_below_the_flow_net():
    """Why GEC 4's Eqs. 15-16 (the head lost uniformly along the wall) give
    less than the flow net behind the wall: at mid-penetration the exact
    head is 0.84 of the difference, the uniform assumption 0.75."""
    exact = sheet_pile_head(2.5, 5.0, 10.0, 1.0)
    assert exact == pytest.approx(0.8416, abs=1e-3)
    assert exact > 0.75 + 0.05


def test_seepage_heads_agree_with_the_cst_solver():
    nodes, el = generate_rect_mesh(0.0, 10.0, -5.0, 0.0, 10, 5)
    left = np.where(np.abs(nodes[:, 0]) < 1e-9)[0]
    right = np.where(np.abs(nodes[:, 0] - 10.0) < 1e-9)[0]
    k = np.linspace(1e-5, 4e-5, len(el))
    bcs = [(int(n), 3.0) for n in left] + [(int(n), 1.0) for n in right]
    ref = solve_seepage(nodes, el, k, bcs)['head']
    h = seepage_heads(nodes, el, k, bcs)
    np.testing.assert_allclose(h, ref, atol=1e-6)


def test_seepage_heads_need_a_prescribed_head():
    nodes, el = generate_rect_mesh(0.0, 2.0, -1.0, 0.0, 2, 1)
    with pytest.raises(ValueError, match="no prescribed head"):
        seepage_heads(nodes, el, 1.0, {})


# ---------------------------------------------------------------------------
# 2. The benchmark: GEC 4 section 5.2.9 (pp. 62-63)
# ---------------------------------------------------------------------------

H_EXC, D_EMB = 6.0, 6.0             # excavation depth, embedment (m)
SAND = [{"name": "Sand", "bottom_elevation": -60, "E": 40000, "nu": 0.3,
         "gamma": 20.0, "c": 0.0, "phi": 35.0, "psi": 0.0}]
STRUTS = [{"depth": 1.0, "stiffness": 1e5}, {"depth": 4.0, "stiffness": 1e5}]


def gec4(H, d, i=0.0, j=0.0):
    """GEC 4 Eqs. 15, 16 and 17 (kPa)."""
    L = 2 * d + H - i - j
    return (2 * (d + H - j) * (d - i) / L * GW,
            (H + i - j) * 2 * (d - i) / L * GW,
            (H + i - j) * GW)


@pytest.fixture(scope="module")
def propped():
    """A sheet pile wall propped at 1 m and 4 m in a 6 m excavation in
    sand, embedded 6 m, the water table at the ground; wide (60 m), so the
    flow in front is not confined by the centre line."""
    out = {}
    for mode in ('full', 'dewatered'):
        out[mode] = analyze_excavation(
            width=60.0, depth=H_EXC, wall_depth=H_EXC + D_EMB,
            soil_layers=SAND, wall_EI=50300.0, wall_EA=3.36e6, gwt=0.0,
            struts=STRUTS, nx=20, ny=10, excavation_water=mode)
    return out


def test_benchmark_converges(propped):
    assert propped['dewatered'].converged
    assert propped['full'].converged
    assert propped['dewatered'].excavation_water == 'dewatered'


def test_net_water_pressure_between_gec4_and_no_seepage(propped):
    """GEC 4 Eq. 16 (39.2 kPa) < FEM (47.3) < Eq. 17 (58.9); the tip
    pressure above Eq. 15 (78.5 kPa; FEM 83.4) by less than 10 %."""
    st = propped['dewatered'].excavation_stages[-1]
    u_f, u_c, u_n = gec4(H_EXC, D_EMB)
    assert (u_f, u_c, u_n) == pytest.approx((78.48, 39.24, 58.86), abs=0.01)
    net = st['net_water_pressure_at_excavation_level_kPa']
    assert u_c < net < u_n
    assert net == pytest.approx(47.3, abs=1.0)
    assert u_f < st['tip_pore_pressure_kPa'] < 1.10 * u_f
    assert st['tip_head_m'] == pytest.approx(
        -(H_EXC + D_EMB) + st['tip_pore_pressure_kPa'] / GW)
    assert st['water_inside_depth_m'] == pytest.approx(H_EXC)


def test_the_model_domain_is_large_enough(propped):
    """The same flow net in a domain eight times larger (retained side,
    excavation and depth 96 m): the model's (retained side and base two
    wall depths away, the far side impervious) is within 2 %."""
    H, d = H_EXC, D_EMB
    W, T, L = H + d, 96.0, 96.0
    xs = _graded_coords([-L, 0.0, L], 80)
    ys = _graded_coords([-T, -W, -H, 0.0], 60)
    nodes, el, face, cx = _cut_mesh(xs, ys, W)
    cy = nodes[el[:, :3]].mean(axis=1)[:, 1]
    keep = ~((cx > 0) & (cy > -H))
    el, cx = el[keep], cx[keep]
    left = set(np.unique(el[cx < 0]).tolist())
    right = set(np.unique(el[cx > 0]).tolist())
    bcs = {n: 0.0 for n in left - right if nodes[n, 1] >= -1e-9}
    bcs.update({n: -H for n in right if abs(nodes[n, 1] + H) < 1e-9})
    h = seepage_heads(nodes, el, 1.0, bcs)
    tip = int(np.where((np.abs(nodes[:, 0]) < 1e-9)
                       & (np.abs(nodes[:, 1] + W) < 1e-9))[0][0])
    back = [n for n in left - right if abs(nodes[n, 0]) < 1e-9
            and abs(nodes[n, 1] + H) < 1e-9][0]
    st = propped['dewatered'].excavation_stages[-1]
    assert st['tip_pore_pressure_kPa'] == pytest.approx(
        GW * (h[tip] + W), rel=0.02)
    assert st['net_water_pressure_at_excavation_level_kPa'] == \
        pytest.approx(GW * (h[back] + H), rel=0.02)


def test_pressures_above_hydrostatic_in_front_below_behind(propped):
    """GEC 4 p. 62: with seepage the pore pressures are greater than
    hydrostatic in front of the wall and less than hydrostatic behind it;
    the net pressure is zero at the tip and largest at the excavation
    level."""
    rows = propped['dewatered'].wall_water_pressures
    for r in rows:
        z = r['depth_m']
        if 0.0 < z < H_EXC + D_EMB:
            assert r['retained_side_kPa'] < GW * z
        if H_EXC < z < H_EXC + D_EMB:
            assert r['excavation_side_kPa'] > GW * (z - H_EXC)
        if z < H_EXC:
            assert r['excavation_side_kPa'] == 0.0      # dewatered
    assert rows[-1]['net_kPa'] == pytest.approx(0.0, abs=1e-9)
    peak = max(rows, key=lambda r: r['net_kPa'])
    assert peak['depth_m'] == pytest.approx(H_EXC)


def test_dewatered_loads_the_wall_more(propped):
    full, dew = propped['full'], propped['dewatered']
    assert dew.max_beam_moment_kNm_per_m > 5 * full.max_beam_moment_kNm_per_m
    assert all(a['force_kN_per_m'] > 2 * b['force_kN_per_m']
               for a, b in zip(dew.strut_forces, full.strut_forces))
    assert full.excavation_stages[-1]['tip_head_m'] == pytest.approx(0.0)
    assert full.excavation_stages[-1][
        'net_water_pressure_at_excavation_level_kPa'] == pytest.approx(0.0)
    assert dew.excavation_stages[-1]['tip_head_m'] < -3.0


def test_water_lowered_with_the_lifts(propped):
    rows = propped['dewatered'].excavation_stages
    assert [r['excavation_depth_m'] for r in rows] == [1.0, 4.0, 6.0]
    assert [r['water_inside_depth_m'] for r in rows] == \
        pytest.approx([1.0, 4.0, 6.0])
    nets = [r['net_water_pressure_at_excavation_level_kPa'] for r in rows]
    assert nets == sorted(nets) and nets[0] > 0
    for r in rows:
        assert 0 < r['upward_gradient_in_front'] < r['critical_gradient']
    assert rows[-1]['critical_gradient'] == pytest.approx((20.0 - GW) / GW)


def test_notes_give_the_condition_and_its_basis(propped):
    notes = " ".join(propped['dewatered'].notes)
    for text in ("excavation_water='dewatered'", "UFC 3-220-20",
                 "Example 7-2", "GEC 4", "EM 1110-2-2504",
                 "Eqs. 15-16", "critical gradient"):
        assert text in notes
    assert not propped['dewatered'].warnings
    assert any("full of water" in w for w in propped['full'].warnings)


def test_tables_show_the_water(propped):
    r = propped['dewatered']
    tables = {t.name: t for t in r.get_tables()}
    t = tables['excavation_water']
    assert t.printed
    assert t.column("Pore pressure at the tip") == [
        s['tip_pore_pressure_kPa'] for s in r.excavation_stages]
    assert not tables['wall_water_pressures'].printed
    from fem2d.calc_steps import get_calc_steps
    sec = [s for s in get_calc_steps(r, {}) if s.title == "Excavation Stages"]
    assert any(getattr(i, 'title', '') == "Water on the Wall by Stage"
               for i in sec[0].items)
    d = r.to_dict()
    assert d['excavation_water'] == 'dewatered'
    assert len(d['wall_water_pressures']) == len(r.wall_water_pressures)


# ---------------------------------------------------------------------------
# 3. Consistency
# ---------------------------------------------------------------------------

CLAY = [{"name": "Clay", "bottom_elevation": -20, "E": 30000, "nu": 0.3,
         "gamma": 18, "c": 25, "phi": 20, "psi": 0}]
QUICK = dict(width=10, depth=5, wall_depth=10, wall_EI=50000, wall_EA=5e6,
             nx=16, ny=8, n_steps=3)


@pytest.mark.parametrize("contact", ['smooth', 'bonded'])
def test_water_inside_at_the_table_is_full(contact):
    """The seepage path with the water inside held at the table gives the
    same answer as 'full': the water on the wall's faces (through the
    soil on each side, and the free water in the excavation) balances
    node by node. For 'bonded' this also checks that the cut wall tied in
    both directions is the wall sharing the soil's nodes."""
    full = analyze_excavation(soil_layers=CLAY, gwt=-2.0, **QUICK,
                              wall_contact=contact, excavation_water='full')
    level = analyze_excavation(soil_layers=CLAY, gwt=-2.0, **QUICK,
                               wall_contact=contact,
                               excavation_water='level',
                               water_level_inside=2.0)
    assert level.max_beam_moment_kNm_per_m == pytest.approx(
        full.max_beam_moment_kNm_per_m, rel=1e-4)
    assert level.max_beam_shear_kN_per_m == pytest.approx(
        full.max_beam_shear_kN_per_m, rel=1e-4)
    assert level.excavation_stages[-1]['wall_head_ux_m'] == pytest.approx(
        full.excavation_stages[-1]['wall_head_ux_m'], rel=1e-4, abs=1e-8)
    assert level.excavation_stages[-1][
        'net_water_pressure_at_excavation_level_kPa'] == pytest.approx(
        0.0, abs=1e-6)


def test_impervious_layer_under_the_tip_is_the_no_seepage_case():
    """A wall cut off into an impervious layer (k a million times smaller):
    no flow, so the net water pressure at the excavation level is the
    unbalanced hydrostatic gamma_w H (GEC 4 Eq. 17; EM 1110-2-2504 para
    4-5a)."""
    layers = [dict(CLAY[0], bottom_elevation=-9.0, k=1e-5),
              dict(CLAY[0], name="Cut-off", bottom_elevation=-20, k=1e-11)]
    r = analyze_excavation(soil_layers=layers, gwt=0.0, **QUICK)
    st = r.excavation_stages[-1]
    assert st['net_water_pressure_at_excavation_level_kPa'] == \
        pytest.approx(GW * 5.0, rel=0.01)


def test_water_standing_in_the_excavation_pushes_the_wall():
    """'level': water 2.5 m deep inside a 5 m excavation stands on the
    floor and against the wall's face above it."""
    r = analyze_excavation(soil_layers=CLAY, gwt=0.0, **QUICK,
                           excavation_water='level', water_level_inside=2.5)
    assert r.converged
    st = r.excavation_stages[-1]
    assert st['water_inside_depth_m'] == pytest.approx(2.5)
    for row in r.wall_water_pressures:
        if 2.5 <= row['depth_m'] <= 5.0:
            assert row['excavation_side_kPa'] >= GW * (row['depth_m'] - 2.5) \
                - 1e-9
        if row['depth_m'] < 2.5:
            assert row['excavation_side_kPa'] == 0.0
    dew = analyze_excavation(soil_layers=CLAY, gwt=0.0, **QUICK)
    assert 0 < st['net_water_pressure_at_excavation_level_kPa'] < \
        dew.excavation_stages[-1]['net_water_pressure_at_excavation_level_kPa']


def test_free_water_on_the_wall_resultant():
    """Water 4 m deep against a wall face: resultant gamma_w h^2 / 2
    toward the retained side, acting h / 3 above the floor."""
    y = np.linspace(0.0, -6.0, 13)              # corner, midside, ...
    nodes = np.column_stack([np.zeros_like(y), y])
    wall = list(range(len(y)))
    for quadratic in (True, False):
        F = _free_water_on_wall(nodes, wall, -1.0, -5.0, GW, quadratic)
        fx = F[0::2]
        assert fx.sum() == pytest.approx(-GW * 16 / 2)
        assert (fx * (y + 5.0)).sum() == pytest.approx(-GW * 64 / 6)
        assert np.all(F[1::2] == 0.0)


def test_quick_condition_is_warned():
    """A short embedment below a deep dewatered excavation: the upward
    gradient in front reaches gamma'/gamma_w."""
    stiff = [{"name": "s", "bottom_elevation": -100, "E": 1e6, "nu": 0.3,
              "gamma": 17.0, "c": 1e4, "phi": 0, "psi": 0}]
    r = analyze_excavation(width=40, depth=6, wall_depth=9,
                           soil_layers=stiff, wall_EI=1e6, wall_EA=1e7,
                           gwt=0.0, nx=16, ny=8, n_steps=1)
    st = r.excavation_stages[-1]
    assert st['upward_gradient_in_front'] >= st['critical_gradient']
    assert any("quick" in w for w in r.warnings)


@pytest.mark.parametrize("kwargs, match", [
    (dict(excavation_water='pumped'), "excavation_water must be"),
    (dict(excavation_water='level'), "needs water_level_inside"),
    (dict(water_level_inside=2.0), "only with excavation_water='level'"),
    (dict(excavation_water='level', water_level_inside=6.0),
     "takes wells"),
    (dict(excavation_water='level', water_level_inside=0.5, gwt=-1.0),
     "above the water table outside"),
    (dict(gwt=0.5), "above the ground"),
    (dict(gwt=np.array([[-20.0, -1.0], [20.0, -1.0]])), "one elevation"),
    (dict(soil_layers=[dict(CLAY[0], k=1e-5),
                       dict(CLAY[0], bottom_elevation=-30)]), "'k' > 0"),
])
def test_bad_water_input_is_refused(kwargs, match):
    args = dict(soil_layers=[dict(CLAY[0], bottom_elevation=-10),
                             dict(CLAY[0], bottom_elevation=-30)],
                gwt=0.0, **QUICK)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        analyze_excavation(**args)
