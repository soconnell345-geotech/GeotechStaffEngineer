"""GM-308 (a): the Anderson-accelerated constant-stiffness iteration.

It is an option of the footing capacity (``iteration='accelerated'``);
no default changes. Every other caller keeps the constant-stiffness
iteration, pinned here to the values master gave before GM-308 (a
strength reduction to 12 significant digits, a staged excavation to 8).
Made-up values throughout.
"""

import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from fem2d import analyze_footing_capacity
from fem2d.analysis import analyze_excavation, analyze_slope_srm
from fem2d.solver import _anderson_step


# ---------------------------------------------------------------------------
# Callers other than the footing: unchanged to the digit
# ---------------------------------------------------------------------------

SURFACE = [(0.0, 6.0), (6.0, 6.0), (18.0, 0.0), (24.0, 0.0)]
LAYER = {'name': 'soil', 'bottom_elevation': -6.0, 'E': 1e5, 'nu': 0.3,
         'c': 8.0, 'phi': 20.0, 'psi': 0.0, 'gamma': 20.0}
CLAY = [{"name": "clay", "bottom_elevation": -20, "E": 20000, "nu": 0.35,
         "c": 20, "phi": 20, "psi": 0, "gamma": 18}]


def test_strength_reduction_unchanged():
    """Values from master 9ae6560, before GM-308."""
    r = analyze_slope_srm(SURFACE, [LAYER], nx=12, ny=6, srf_tol=0.05)
    assert r.FOS == 1.588
    assert float(np.abs(r.displacements).sum()) == pytest.approx(
        5.937264836335416, rel=1e-12)
    assert float(r.max_displacement_m) == pytest.approx(
        0.038335241746602605, rel=1e-12)


def test_staged_excavation_unchanged():
    """Values from master 9ae6560, before GM-308. rel 1e-8: master itself
    reads sum |u| 2.5e-11 and the moments about 5e-10 apart between
    machines and numpy/scipy builds (GM-308 review), so a tighter pin
    fails on master too."""
    e = analyze_excavation(
        width=10, depth=5, wall_depth=10, soil_layers=CLAY, wall_EI=50000,
        wall_EA=5e6, nx=20, ny=10, n_steps=5,
        struts=[{"depth": 1.5, "stiffness": 50000}])
    assert float(np.abs(e.displacements).sum()) == pytest.approx(
        5.419050307962344, rel=1e-8)
    assert float(e.max_displacement_m) == pytest.approx(
        0.04056509255721257, rel=1e-8)
    moments = [float(b.moment_j) for b in e.beam_forces[:3]]
    assert moments == pytest.approx(
        [1.7096224744901747, 11.968060662234393, -20.787306443747184],
        rel=1e-8)


# ---------------------------------------------------------------------------
# The Anderson step itself
# ---------------------------------------------------------------------------

def test_anderson_step_solves_a_linear_system_faster():
    """On a linear system K u = F with a poor 'K0' the accelerated
    iteration converges where the plain one barely moves (Walker & Ni
    2011: Anderson(m) on a linear problem is GMRES-like)."""
    n = 60
    rng = np.random.default_rng(3)
    k = 1.0 + rng.random(n) * 50.0
    K = diags([k, -0.4 * np.ones(n - 1), -0.4 * np.ones(n - 1)],
              [0, -1, 1]).tocsc()
    K0 = diags(np.full(n, k.max())).tocsc()
    lu = splu(K0)
    F = rng.random(n)

    def run(accel, iters):
        u = np.zeros(n)
        hist = {}
        for _ in range(iters):
            R = F - K @ u
            u = u + (_anderson_step(lu, u, R, hist, 10) if accel
                     else lu.solve(R))
        return np.linalg.norm(F - K @ u) / np.linalg.norm(F)

    assert run(True, 60) < 1e-6
    assert run(False, 60) > 1e-3


def test_anderson_first_step_is_the_constant_stiffness_step():
    lu = splu(diags([2.0, 4.0]).tocsc())
    du = _anderson_step(lu, np.zeros(2), np.array([2.0, 2.0]), {}, 10)
    assert du == pytest.approx([1.0, 0.5])


# ---------------------------------------------------------------------------
# The footing: inputs, choices, and the frictional answer
# ---------------------------------------------------------------------------

def _iteration_choice(r):
    return next((c for c in r.choices if c['what'] == 'nonlinear iteration'),
                None)


def test_bad_iteration_refused():
    with pytest.raises(ValueError, match="iteration"):
        analyze_footing_capacity(B=2.0, c=50.0, iteration='newton')


def test_accelerated_option_on_a_frictional_soil():
    """20x10, c = 20 kPa, phi = 25 at max_iter 400: the default
    constant-stiffness iteration reads 455.6 kPa; iteration='accelerated'
    reads above it. Its value is not pinned: near collapse the accelerated
    answer varies by machine (542.4, 564.1 at 400 on the machines checked;
    GM-308 review)."""
    kw = dict(B=2.0, c=20.0, phi=25.0, gamma=18.0, nx=20, ny=10)
    a = analyze_footing_capacity(max_iter=400, iteration='accelerated', **kw)
    ch = _iteration_choice(a)
    assert ch['chosen'] == 'accelerated' and ch['how'] == 'given'
    assert 'Anderson' in ch['basis']
    c = analyze_footing_capacity(max_iter=400, **kw)
    assert c.q_ult_kPa == pytest.approx(455.6, abs=0.1)
    assert a.q_ult_kPa > c.q_ult_kPa
    # the max_iter warning: a lower bound by default; with the option q_ult
    # can move either way with max_iter, so it says it is not a bound
    assert any('lower bound' in w for w in c.warnings)
    assert all('not a bound' in w and 'lower bound' not in w
               for w in a.warnings if 'max_iter' in w)
    # left out: not listed (the tool does not take it); output as before
    assert _iteration_choice(c) is None
    explicit = analyze_footing_capacity(max_iter=400,
                                        iteration='constant_stiffness', **kw)
    assert explicit.q_ult_kPa == c.q_ult_kPa
    ch = _iteration_choice(explicit)
    assert ch['chosen'] == 'constant_stiffness' and ch['how'] == 'given'
    assert any('accelerated' in x for x in ch['alternatives'])


def test_clay_default_unchanged():
    """phi = 0: 20x10 c = 50 kPa reads 274.2 kPa as in VALIDATION.md §3."""
    r = analyze_footing_capacity(B=2.0, c=50.0, nx=20, ny=10)
    assert r.q_ult_kPa == pytest.approx(274.2, abs=0.1)
    assert _iteration_choice(r) is None


@pytest.mark.slow
def test_default_mesh_frictional_answer_accelerated():
    """Default 40x20 mesh, c = 20 kPa, phi = 25: 520.7 kPa with the
    accelerated iteration at max_iter 4000 (and 16000), where the default
    reads 455.6, 477.3 and 499.0 at 1000, 4000 and 16000. At max_iter 1000
    the accelerated value varies by machine (499.0 or 520.7; GM-308
    review), so it is pinned at 4000."""
    r = analyze_footing_capacity(B=2.0, c=20.0, phi=25.0, gamma=18.0,
                                 max_iter=4000, iteration='accelerated')
    assert r.q_ult_kPa == pytest.approx(520.7, abs=0.1)


def test_rigid_footing_takes_the_iteration_choice():
    r = analyze_footing_capacity(B=2.0, c=20.0, phi=25.0, gamma=18.0,
                                 footing='rigid', nx=20, ny=10,
                                 n_load_steps=6, iteration='accelerated')
    assert _iteration_choice(r)['chosen'] == 'accelerated'
    assert r.q_ult_kPa > 20.0
