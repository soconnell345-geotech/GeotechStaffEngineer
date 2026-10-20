"""The out-of-plane thickness ``t`` of the excavation and staged analyses
(GM-204, 2026-10-20).

fem2d is plane strain per metre run: ``t`` is the thickness of the slice
the model stands for. The soil elements' stiffness and loads are
integrated over it; until 2026-10-20 the members given per metre run
(the wall's EA and EI, struts and springs, wall/soil joints, a phase's
nodal forces) were not multiplied by it, so at t = 1.5 the soil was 1.5
times stiffer and heavier than the wall: on the probe excavation the
wall's moment went 11.40 -> 16.79 kN-m/m and its shear 7.69 -> 11.27 kN/m,
a strut's force 4.35 -> 6.22 kN/m, while the displacements did not move.

Now every member is scaled with the soil and every force is reported per
metre run, so the results per metre are the same at any t.
"""

import numpy as np
import pytest

from fem2d import analyze_excavation
from fem2d.analysis import ConstructionPhase, analyze_staged
from fem2d.elements import BeamElement
from fem2d.mesh import detect_boundary_nodes, generate_rect_mesh

REL = 1e-6

LAYERS = [
    {'E': 40000, 'nu': 0.3, 'c': 1, 'phi': 30, 'psi': 0.0, 'gamma': 18,
     'bottom_elevation': -6.0},
    {'E': 20000, 'nu': 0.3, 'c': 5, 'phi': 28, 'psi': 0.0, 'gamma': 18,
     'model': 'hs', 'E50_ref': 25000, 'Eur_ref': 75000, 'm': 0.5,
     'p_ref': 100, 'R_f': 0.9, 'bottom_elevation': -8.0},
]
BASE = dict(width=10.0, depth=2.0, wall_depth=5.0, soil_layers=LAYERS,
            wall_EI=5.0e4, wall_EA=5.0e6, nx=10, ny=8, n_steps=4)
CASES = {
    # a strut, the water lowered to a level inside: the free water on the
    # wall's face is a phase's nodal forces
    'strut_water_level': dict(
        struts=[{'depth': 0.5, 'stiffness': 2.0e4}], gwt=-0.5,
        excavation_water='level', water_level_inside=1.0),
    # wall friction: the joints' stiffness, adhesion and water force
    'interface_water': dict(
        wall_contact='interface', interface_delta=17.0,
        interface_adhesion=2.0, gwt=-1.0),
}


def _close(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    scale = max(float(np.max(np.abs(a))), 1e-9)
    assert np.allclose(a, b, rtol=REL, atol=REL * scale), \
        f"max difference {float(np.max(np.abs(a - b)))} of {scale}"


def _beam_table(beam_forces):
    return [[bf.axial_i, bf.shear_i, bf.moment_i,
             bf.axial_j, bf.shear_j, bf.moment_j] for bf in beam_forces]


@pytest.mark.parametrize('case', sorted(CASES))
def test_excavation_wall_per_metre_independent_of_t(case):
    r1 = analyze_excavation(t=1.0, **BASE, **CASES[case])
    r2 = analyze_excavation(t=1.5, **BASE, **CASES[case])
    assert r1.converged and r2.converged
    _close(r1.max_beam_moment_kNm_per_m, r2.max_beam_moment_kNm_per_m)
    _close(r1.max_beam_shear_kN_per_m, r2.max_beam_shear_kN_per_m)
    _close(_beam_table(r1.beam_forces), _beam_table(r2.beam_forces))
    _close(r1.displacements, r2.displacements)
    _close(r1.stresses, r2.stresses)
    for key in ('max_beam_moment_kNm_per_m', 'max_beam_shear_kN_per_m'):
        _close(r1.wall_envelope[key], r2.wall_envelope[key])
    if r1.strut_forces:
        _close([s['force_kN_per_m'] for s in r1.strut_forces],
               [s['force_kN_per_m'] for s in r2.strut_forces])
    p1, p2 = r1.staged.phases[-1], r2.staged.phases[-1]
    if p1.interface_forces:
        for key in ('normal_kN_per_m', 'shear_kN_per_m', 'limit_kN_per_m'):
            _close([f[key] for f in p1.interface_forces],
                   [f[key] for f in p2.interface_forces])


def _staged_with_beams(t):
    """A wall of beams (with self-weight) in a block, a spring to ground at
    its head, a tie between two nodes, a strip load and a water table, in
    two phases."""
    nodes, elements = generate_rect_mesh(0.0, 10.0, -6.0, 0.0, 10, 6)
    bc = detect_boundary_nodes(nodes)
    col = sorted((i for i, n in enumerate(nodes)
                  if abs(n[0] - 5.0) < 1e-9 and n[1] >= -4.0 - 1e-9),
                 key=lambda i: -nodes[i, 1])
    beams = [BeamElement(node_i=int(a), node_j=int(b), EA=5.0e6, EI=5.0e4,
                         weight_per_m=2.0) for a, b in zip(col, col[1:])]
    top = sorted((i for i, n in enumerate(nodes)
                  if abs(n[1]) < 1e-9 and n[0] < 4.5),
                 key=lambda i: nodes[i, 0])
    edges = [(top[i], top[i + 1]) for i in range(len(top) - 1)]
    right = [i for i, n in enumerate(nodes)
             if abs(n[1]) < 1e-9 and abs(n[0] - 8.0) < 1e-9][0]
    springs = [{'node': col[0], 'kx': 5.0e3},
               {'node': col[0], 'node_b': right, 'kx': 2.0e3}]
    nf = np.zeros(2 * len(nodes))
    nf[2 * col[1]] = 3.0                    # kN/m per m run
    mp = [{'E': 20000, 'nu': 0.3, 'c': 10, 'phi': 25, 'psi': 0}]
    groups = {'all': list(range(len(elements)))}
    phases = [
        ConstructionPhase(name='in situ', active_soil_groups=['all'],
                          gwt=-3.0),
        ConstructionPhase(name='wall and load', active_soil_groups=['all'],
                          active_beam_ids=list(range(len(beams))),
                          active_spring_ids=[0, 1], gwt=-3.0,
                          surface_loads=[(edges, 0.0, -50.0)],
                          nodal_forces=nf)]
    return analyze_staged(nodes, elements, mp, 18.0, bc, groups, phases,
                          beam_elements=beams, t=t, springs=springs)


def test_staged_beams_springs_per_metre_independent_of_t():
    r1, r2 = _staged_with_beams(1.0), _staged_with_beams(1.5)
    assert r1.converged and r2.converged
    p1, p2 = r1.phases[-1], r2.phases[-1]
    assert p1.max_beam_moment_kNm_per_m > 1.0       # the wall is loaded
    _close(p1.max_beam_moment_kNm_per_m, p2.max_beam_moment_kNm_per_m)
    _close(p1.max_beam_shear_kN_per_m, p2.max_beam_shear_kN_per_m)
    _close(_beam_table(p1.beam_forces), _beam_table(p2.beam_forces))
    _close([[s['force_x_kN_per_m'], s['force_y_kN_per_m']]
            for s in p1.spring_forces],
           [[s['force_x_kN_per_m'], s['force_y_kN_per_m']]
            for s in p2.spring_forces])
    _close(p1.displacements, p2.displacements)
    _close(p1.stresses, p2.stresses)
