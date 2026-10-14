"""The excavation wall is one straight line of beams (GM-70, 2026-10-14).

Until 2026-10-14 ``create_wall_elements`` took every node within ``tol`` of
the wall line. On a T6 mesh the midside nodes of horizontal edges sit at
x = +/- dx/2, inside the 0.6 dx tolerance ``analyze_excavation`` passes, so
the "wall" hopped between three node columns (horizontal beam segments
across the excavation face) and its moments came out about 4x too large;
and where x = 0 was not a mesh column the wall straddled two columns.
Now: one column, the one nearest the wall line; ``analyze_excavation`` puts
a node column on x = 0 and a node row at the wall tip; T3 and T6 meshes
agree within mesh tolerance.
"""

import numpy as np
import pytest

from fem2d import analyze_excavation
from fem2d.analysis import create_wall_elements
from fem2d.mesh import convert_to_t6, generate_rect_mesh

CLAY = [{"name": "Clay", "bottom_elevation": -20, "E": 30000, "nu": 0.3,
         "gamma": 18, "c": 25, "phi": 20, "psi": 0}]


def _wall(result):
    nodes = result.nodes
    xi = [nodes[b.node_i, 0] for b in result.beam_forces]
    xj = [nodes[b.node_j, 0] for b in result.beam_forces]
    yi = [nodes[b.node_i, 1] for b in result.beam_forces]
    yj = [nodes[b.node_j, 1] for b in result.beam_forces]
    return np.array(xi), np.array(xj), np.array(yi), np.array(yj)


def test_create_wall_elements_takes_one_column_on_a_t6_mesh():
    nodes, elements = generate_rect_mesh(-10, 10, -10, 0, 10, 5)
    nodes, elements = convert_to_t6(nodes, elements)
    dx = 2.0
    beams, ids = create_wall_elements(nodes, x_wall=0.0, y_top=0.0,
                                      y_bottom=-6.0, EA=1e6, EI=1e4,
                                      tol=0.6 * dx)
    assert np.all(np.abs(nodes[ids, 0]) < 1e-9)
    lengths = [np.hypot(*(nodes[b.node_j] - nodes[b.node_i])) for b in beams]
    assert np.allclose(lengths, 1.0)               # dy/2 on the T6 mesh
    assert nodes[ids[0], 1] == pytest.approx(0.0)
    assert nodes[ids[-1], 1] == pytest.approx(-6.0)


@pytest.mark.parametrize("element_type", ["t3", "t6"])
def test_excavation_wall_is_vertical_with_node_spacing(element_type):
    r = analyze_excavation(width=10, depth=5, wall_depth=10,
                           soil_layers=CLAY, wall_EI=50000, wall_EA=5e6,
                           nx=30, ny=15, element_type=element_type)
    xi, xj, yi, yj = _wall(r)
    assert np.all(np.abs(xi) < 1e-9) and np.all(np.abs(xj) < 1e-9)
    lengths = np.array([b.length for b in r.beam_forces])
    assert np.allclose(yi - yj, lengths)           # straight down
    assert np.allclose(yj[:-1], yi[1:])            # node to next node
    # the node spacing of the mesh (the row moved onto the tip may make
    # the last interval or two shorter, never longer than 1.5 spacings)
    spacing = np.median(lengths)
    assert np.all(lengths <= 1.5 * spacing + 1e-9)
    assert yi[0] == pytest.approx(0.0)
    assert yj[-1] == pytest.approx(-10.0)          # the wall tip as asked


def test_wall_on_x0_when_x0_is_not_a_uniform_column():
    """width 6, wall 8, nx 10: dx = 3.8 m, and 16/3.8 is not whole, so
    before GM-70 the wall straddled the columns at -0.8 and +1.1 m."""
    r = analyze_excavation(width=6, depth=3, wall_depth=8,
                           soil_layers=CLAY, wall_EI=5000, wall_EA=1e6,
                           nx=10, ny=6, n_steps=3)
    xi, xj, _, yj = _wall(r)
    assert np.all(np.abs(xi) < 1e-9) and np.all(np.abs(xj) < 1e-9)
    assert yj[-1] == pytest.approx(-8.0)


@pytest.fixture(scope="module")
def t3_t6():
    out = {}
    for et in ("t3", "t6"):
        out[et] = analyze_excavation(
            width=10, depth=5, wall_depth=10, soil_layers=CLAY,
            wall_EI=50000, wall_EA=5e6, nx=60, ny=30, element_type=et)
    return out


def _head_ux(r):
    return float(np.asarray(r.displacements)[2 * r.beam_forces[0].node_i])


def _moment_above(r, z):
    return max(max(abs(b.moment_i), abs(b.moment_j)) for b in r.beam_forces
               if r.nodes[b.node_j, 1] >= z - 1e-9)


def test_t3_and_t6_agree_on_the_wall(t3_t6):
    """Wall head deflection and the moments above the excavation level
    agree within mesh tolerance (60 x 30 mesh: 1.4 % and 0.8 %). The
    whole-wall maximum does not: on T3 it sits in a stress concentration
    at the wall tip (18.9 vs 14.2 kN-m/m), which mesh refinement does not
    remove; see fem2d/DESIGN.md."""
    t3, t6 = t3_t6["t3"], t3_t6["t6"]
    assert _head_ux(t6) == pytest.approx(_head_ux(t3), rel=0.05)
    assert _moment_above(t6, -5.0) == pytest.approx(_moment_above(t3, -5.0),
                                                    rel=0.05)


def test_no_horizontal_beam_segments_in_the_default_mesh(t3_t6):
    for r in t3_t6.values():
        xi, xj, _, _ = _wall(r)
        assert np.max(np.abs(xj - xi)) < 1e-9
