"""Wall friction between 0 and phi' in a staged excavation (GM-72, 2026-10-19).

``analyze_excavation(wall_contact='interface')`` joins the wall to each soil
face by zero-thickness joints lumped at the node pairs: the normal tie of
the smooth wall, and a tangential force limited by Mohr-Coulomb,
|T| <= N' tan(delta) + c_a L.

Checks:

1. The limits: a joint with delta = 0 is the smooth wall exactly; a joint
   too strong to slip is the bonded wall.
2. The joints against the classical thrusts, on a rigid wall 4 m high
   translated away from (active) or into (passive) a sand with phi' = 30
   deg: Coulomb's active thrust (UFC 3-220-20, 2025, Table 4-2, p. 194:
   Coulomb K_A = 0.30 at delta/phi' = 0.5 and 1) and Caquot-Kerisel's
   passive thrust (Caltrans Trenching and Shoring Manual, 2025, Figure
   4-20, p. 4-30: Kp = 6.3 for phi' = 30 deg and a level surface at
   delta = phi', times R = 0.686, 0.746, 0.878 at delta/phi' = 0.4, 0.5,
   0.7), horizontal components (P_N = P cos delta, Figure 4-20).
3. The USS Steel Sheet Piling Design Manual cantilever (the benchmark of
   test_excavation_staged.py) with delta/phi' = 0.5, the ratio the manual
   uses for its passive coefficient (Kp = 6.56, Fig. 5A): the conventional
   method with Coulomb's active coefficient at the same delta.
4. The default (smooth) is unchanged; the inputs are checked.
"""

import math

import numpy as np
import pytest

from fem2d import analyze_excavation
from fem2d.analysis import (ConstructionPhase, _geostatic_stresses,
                            analyze_staged)
from fem2d.elements import BeamElement
from fem2d.mesh import convert_to_t6, generate_rect_mesh

from fem2d.tests.test_excavation_staged import (
    D_WALL, H, PCF, PZ27_EA, PZ27_EI, SAND, uss_cantilever)


# ---------------------------------------------------------------------------
# A rigid wall translated against a sand block (check 2)
# ---------------------------------------------------------------------------

def coulomb_ka(phi, delta):
    """Coulomb's active coefficient, vertical wall, level backfill (the
    thrust inclined at delta to the wall's normal)."""
    p, d = math.radians(phi), math.radians(delta)
    return math.cos(p) ** 2 / (math.cos(d) * (1.0 + math.sqrt(
        math.sin(p + d) * math.sin(p) / math.cos(d))) ** 2)


def translate_wall(phi, delta, move, psi, Hw=4.0, gamma=18.0, E=20000.0,
                   nu=0.3, nx=20, ny=10, n_steps=20):
    """A rigid wall on x = 0 from the ground to -Hw, the sand to its right
    (x from 0 to 5 Hw, down to -2 Hw, base fixed, far side on rollers; the
    face below the wall on rollers, so the soil under the heel can move
    down or up). In-situ K0 stresses with the wall held, then the wall
    translated by ``move`` (m; negative = away from the soil, active) by a
    stiff spring and a matching force at its top and its heel. The wall
    joins the soil face above its heel by the same ties and joints
    ``analyze_excavation`` builds (the heel node sits on the rollers).

    Returns (horizontal thrust on the wall, vertical shear on the wall
    (+ up), the staged result)."""
    B = 5.0 * Hw
    nodes, elems = generate_rect_mesh(0.0, B, -2.0 * Hw, 0.0, nx, 2 * ny)
    nodes, elems = convert_to_t6(nodes, elems)
    n0 = len(nodes)
    on_face = np.where((np.abs(nodes[:, 0]) < 1e-9)
                       & (nodes[:, 1] > -Hw - 1e-9))[0]
    face = sorted((int(i) for i in on_face), key=lambda i: -nodes[i, 1])
    wall = list(range(n0, n0 + len(face)))
    nodes = np.vstack([nodes, nodes[face]])
    bc = {'fixed_base': [int(i) for i in np.where(
              np.abs(nodes[:n0, 1] + 2.0 * Hw) < 1e-9)[0]],
          'roller_left': [int(i) for i in np.where(
              (np.abs(nodes[:n0, 0]) < 1e-9)
              & (nodes[:n0, 1] < -Hw + 1e-9))[0]],
          'roller_right': [int(i) for i in np.where(
              np.abs(nodes[:n0, 0] - B) < 1e-9)[0]]}
    beams = [BeamElement(node_i=wall[i], node_j=wall[i + 1], EA=1e7,
                         EI=1e6) for i in range(len(wall) - 1)]
    h = Hw / ny
    y = nodes[face, 1]
    trib = np.zeros(len(face))
    for i in range(0, len(face) - 2, 2):          # T6 edges: 1/6, 2/3, 1/6
        L = abs(y[i] - y[i + 2])
        trib[i] += L / 6.0
        trib[i + 1] += 2.0 * L / 3.0
        trib[i + 2] += L / 6.0
    G = E / (2.0 * (1.0 + nu))
    k_tie = 1e3 * E
    springs, joints = [], []
    for k, (w, s) in enumerate(zip(wall[:-1], face[:-1])):
        j = len(springs)
        springs.append({'node': w, 'node_b': s, 'kx': k_tie, 'ky': 0.0})
        if delta > 0:
            joints.append({'spring': j, 'face': -1,
                           'tan_delta': math.tan(math.radians(delta)),
                           'ks': G * trib[k] / (0.5 * h),
                           'length': trib[k]})
    n_ties = len(springs)
    kp = 1e2 * E
    springs.append({'node': wall[0], 'kx': kp})
    springs.append({'node': wall[-1], 'kx': kp, 'ky': kp})
    mats = [{'E': E, 'nu': nu, 'c': 0.0, 'phi': phi, 'psi': psi,
             'gamma': gamma}] * len(elems)
    sig0 = _geostatic_stresses(nodes, elems, [mats[0]],
                               np.zeros(len(elems), int), None, None, 1.0)
    push = np.zeros(2 * len(nodes))
    push[2 * wall[0]] = push[2 * wall[-1]] = kp * move
    every = dict(active_soil_groups=['soil'],
                 active_beam_ids=list(range(len(beams))),
                 active_spring_ids=list(range(len(springs))))
    r = analyze_staged(
        nodes, elems, mats, gamma, bc, {'soil': list(range(len(elems)))},
        [ConstructionPhase(name='K0', n_steps=2, **every),
         ConstructionPhase(name='translate', n_steps=n_steps,
                           nodal_forces=push, **every)],
        beam_elements=beams, springs=springs, initial_stresses=sig0,
        interfaces=joints or None, max_iter=1000, tol=1e-7,
        settled_ratio=0.1, settled_residual_cap=1e-2)
    last = r.phases[-1]
    P_h = abs(sum(s['force_x_kN_per_m'] for s in last.spring_forces
                  if s['spring'] < n_ties))
    P_v = sum(row['shear_kN_per_m'] for row in (last.interface_forces or []))
    return P_h, P_v, r


# Coulomb K_A for phi' = 30: the formula, checked against UFC 3-220-20
# (2025) Table 4-2, p. 194 (0.33, 0.30, 0.30 at delta/phi' = 0, 0.5, 1)
@pytest.mark.parametrize("delta,printed", [(0.0, 0.33), (15.0, 0.30),
                                           (30.0, 0.30)])
def test_coulomb_ka_as_printed(delta, printed):
    assert coulomb_ka(30.0, delta) == pytest.approx(printed, abs=0.005)


# Tolerance 5 % on the thrust: the heel node is held by the rollers below
# it, not by the wall (its share of the pressure, L p / 6 on the last
# element side, is 3 % of the thrust on this mesh), and the stress at the
# heel corner is singular; the passive values are also read off a chart
# (Kp = 6.3 +/- 0.1, 1.6 %).
TOL = 0.05


@pytest.mark.parametrize("delta", [0.0, 15.0, 30.0])
def test_active_thrust_approaches_coulomb(delta):
    """Wall moved 1 % of its height away: the thrust reaches Coulomb's at
    the joints' delta and the friction is fully mobilized (shear =
    tan(delta) x thrust): FE 46.8 / 41.0 / 37.0 against 48.0 / 41.9 / 37.1
    kN/m (-2.5, -2.1, -0.2 %)."""
    P_h, P_v, r = translate_wall(30.0, delta, -0.04, psi=0.0)
    assert r.converged
    ref = 0.5 * 18.0 * 4.0 ** 2 * coulomb_ka(30.0, delta) \
        * math.cos(math.radians(delta))
    assert P_h == pytest.approx(ref, rel=TOL)
    # the soil settles past the wall: it drags the wall down by tan(delta)
    assert -P_v == pytest.approx(math.tan(math.radians(delta)) * P_h,
                                 abs=0.02 * P_h)


# Caltrans T&S Manual (2025), Figure 4-20, p. 4-30: Kp at phi' = 30 deg,
# beta = 0 (read 6.3), and its table of R for -delta/phi'
KP0_30 = 6.3
R_30 = {0.4: 0.686, 0.5: 0.746, 0.7: 0.878}


@pytest.mark.slow
@pytest.mark.parametrize("ratio", [0.0, 0.4, 0.5, 0.7])
def test_passive_thrust_approaches_caquot_kerisel(ratio):
    """Wall pushed 5 % of its height into the soil (associated flow,
    psi = phi', the plasticity the log-spiral solution assumes): FE 433.0 /
    594.6 / 646.9 / 748.5 kN/m against Rankine 432 and Caquot-Kerisel
    608.6 / 653.6 / 743.4 (+0.2, -2.3, -1.0, +0.7 %)."""
    delta = ratio * 30.0
    P_h, P_v, r = translate_wall(30.0, delta, 0.2, psi=30.0)
    assert r.converged
    kp = 3.0 if ratio == 0 else R_30[ratio] * KP0_30
    ref = 0.5 * 18.0 * 4.0 ** 2 * kp * math.cos(math.radians(delta))
    assert P_h == pytest.approx(ref, rel=TOL)
    if ratio:
        assert P_v > 0                         # the soil heaves past the wall


# ---------------------------------------------------------------------------
# The excavation: limits, the USS cantilever, the default (checks 1, 3, 4)
# ---------------------------------------------------------------------------

def _uss(**kw):
    return analyze_excavation(width=50.0, depth=H, wall_depth=H + D_WALL,
                              soil_layers=SAND, wall_EI=PZ27_EI,
                              wall_EA=PZ27_EA, **kw)


@pytest.fixture(scope="module")
def runs():
    return {
        'smooth': _uss(),
        'delta0': _uss(wall_contact='interface', interface_delta=0.0),
        'half': _uss(wall_contact='interface', interface_ratio=0.5),
        'full': _uss(wall_contact='interface', interface_ratio=1.0),
        'strong': _uss(wall_contact='interface', interface_delta=0.0,
                       interface_adhesion=1e4),
        'bonded': _uss(wall_contact='bonded'),
    }


def _head(r):
    return r.excavation_stages[-1]['wall_head_ux_m']


def test_frictionless_interface_is_the_smooth_wall(runs):
    a, b = runs['delta0'], runs['smooth']
    assert a.converged and b.converged
    assert a.max_beam_moment_kNm_per_m == b.max_beam_moment_kNm_per_m
    assert _head(a) == _head(b)
    assert a.wall_interface is None            # no joint carries shear


def test_unslipping_interface_approaches_bonded(runs):
    """Adhesion 10 MPa: no joint slips; the joints' elastic shear is all
    that separates them from the bonded wall (M 65.60 vs 65.65 kN-m/m,
    head 26.9 vs 26.4 mm)."""
    a, b = runs['strong'], runs['bonded']
    assert a.converged and b.converged
    assert a.max_beam_moment_kNm_per_m == pytest.approx(
        b.max_beam_moment_kNm_per_m, rel=0.01)
    assert _head(a) == pytest.approx(_head(b), rel=0.05)
    assert a.wall_interface['retained_side']['joints_slipping'] == 0


def test_wall_friction_orders_the_wall_forces(runs):
    """More wall friction, smaller moment and movement: smooth 155.2,
    delta/phi' 0.5 98.7, 1.0 80.9, bonded 65.7 kN-m/m."""
    m = [runs[k].max_beam_moment_kNm_per_m
         for k in ('smooth', 'half', 'full', 'bonded')]
    h = [_head(runs[k]) for k in ('smooth', 'half', 'full', 'bonded')]
    assert all(r.converged for r in runs.values())
    assert m == sorted(m, reverse=True)
    assert h == sorted(h, reverse=True)


def test_retained_side_mobilizes_the_given_delta(runs):
    """delta/phi' = 0.5: the joints slip and the given 17.5 deg is
    mobilized. delta = phi' = 35: the soil next to the wall yields before
    the joints slip (23.6 deg mobilized), more than at 0.5, less than
    given."""
    w = runs['half'].wall_interface['retained_side']
    assert w['mobilized_delta_deg'] == pytest.approx(17.5, abs=1.0)
    assert w['joints_slipping'] >= 0.75 * w['joints']
    assert w['wall_shear_kN_per_m'] < 0        # the soil drags the wall down
    full = runs['full'].wall_interface['retained_side']
    assert w['mobilized_delta_deg'] + 2.0 < full['mobilized_delta_deg'] \
        <= 35.0
    for key in ('half', 'full'):
        front = runs[key].wall_interface['excavation_side']
        assert front['wall_shear_kN_per_m'] > 0    # the soil in front heaves


def test_uss_cantilever_with_wall_friction(runs):
    """The USS example (pp. 86-90) with its own delta/phi' = 0.5: the
    conventional method with Coulomb's K_A at delta = 17.5 deg (0.246) and
    the manual's Kp = 6.56 (Fig. 5A), horizontal components (cos delta):
    M_max 98.7 kN-m/m; the FEM wall with interface_ratio = 0.5: 98.7
    (+0.1 %). (The manual's own Ka = 0.27, Kp = 6.56 without cos delta give
    117.1.) The tolerance is the smooth benchmark's, +/-10 %."""
    c = math.cos(math.radians(17.5))
    le = uss_cantilever(H, 115 * PCF, 65 * PCF, coulomb_ka(35.0, 17.5) * c,
                        6.56 * c)
    assert le["M_max"] == pytest.approx(98.7, abs=0.1)
    assert runs['half'].max_beam_moment_kNm_per_m == pytest.approx(
        le["M_max"], rel=0.10)


def test_choices_and_notes(runs):
    d = runs['half'].to_dict()
    whats = {c['what']: c for c in d['choices']}
    assert whats["wall contact (wall friction)"]['chosen'] == 'interface'
    assert whats["wall friction ratio delta/phi'"]['chosen'] == 0.5
    assert whats["wall adhesion c_a (kPa)"]['how'] == 'default'
    assert "Table 4-1, p. 184" in \
        whats["wall contact (wall friction)"]['basis']
    assert any("wall_contact='interface'" in n and "17.5 deg" in n
               for n in d['notes'])
    assert set(d['wall_interface']) >= {'retained_side', 'excavation_side'}
    smooth = runs['smooth'].to_dict()['choices'][0]
    assert smooth['chosen'] == 'smooth' and smooth['how'] == 'default'
    assert 'Wall friction, retained side' in runs['half'].summary()


@pytest.mark.parametrize("kw,match", [
    (dict(wall_contact='interface'), "interface_delta"),
    (dict(wall_contact='interface', interface_delta=17,
          interface_ratio=0.5), "one of them"),
    (dict(wall_contact='interface', interface_delta=50), "0 to 45"),
    (dict(wall_contact='interface', interface_ratio=1.2), "0 to 1"),
    (dict(wall_contact='interface', interface_delta=17,
          interface_adhesion=-1), "adhesion"),
    (dict(interface_delta=17), "only with"),
    (dict(wall_contact='bonded', interface_adhesion=5), "only with"),
])
def test_bad_inputs(kw, match):
    with pytest.raises(ValueError, match=match):
        analyze_excavation(width=10, depth=2, wall_depth=5,
                           soil_layers=SAND[:1], wall_EI=5e4, wall_EA=5e6,
                           nx=8, ny=6, **kw)


def test_dewatered_excavation_with_wall_friction():
    """With a water table and seepage under the wall the joints use the
    effective normal force (the water pressure the wall carries taken off):
    the run converges and the retained-side delta mobilized stays at or
    below the given one."""
    r = analyze_excavation(
        width=20.0, depth=3.0, wall_depth=7.0, gwt=-1.0, nx=20, ny=12,
        soil_layers=[{'E': 30000, 'nu': 0.3, 'c': 2.0, 'phi': 32.0,
                      'psi': 0.0, 'gamma': 19.0}],
        wall_EI=5e4, wall_EA=5e6, struts=[{'depth': 0.5,
                                           'stiffness': 2e4}],
        wall_contact='interface', interface_delta=16.0)
    assert r.converged
    w = r.wall_interface['retained_side']
    assert 0 < w['mobilized_delta_deg'] <= 16.0 + 0.5
