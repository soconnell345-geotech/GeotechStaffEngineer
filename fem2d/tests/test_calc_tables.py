"""fem2d calculation tables (backlog GM-39): the values in the tables are
the result's own."""

import numpy as np
import pytest

from fem2d import analyze_excavation
from fem2d.analysis import (
    ConstructionPhase, analyze_consolidation, analyze_staged,
)
from fem2d.mesh import (
    convert_to_t6, detect_boundary_nodes, generate_rect_mesh,
    t6_boundary_edges,
)

SOIL = [{"name": "Clay", "bottom_elevation": -20, "E": 30000, "nu": 0.3,
         "gamma": 18, "c": 25, "phi": 20, "psi": 0}]


def _by_name(tables):
    return {t.name: t for t in tables}


@pytest.fixture(scope="module")
def excavation():
    return analyze_excavation(
        width=10, depth=5, wall_depth=10, soil_layers=SOIL,
        wall_EI=50000, wall_EA=5e6, nx=10, ny=6, n_steps=3,
        struts=[{"depth": 1.5, "stiffness": 50000}])


def test_beam_forces_are_the_results_own(excavation):
    t = _by_name(excavation.get_tables())["beam_forces"]
    assert t.printed and t.n_rows == len(excavation.beam_forces)
    bf = excavation.beam_forces[2]
    row = t.rows[2]
    assert row[t.column_index("Bending moment M i")] == bf.moment_i
    assert row[t.column_index("Shear force V j")] == bf.shear_j
    assert row[t.column_index("y i")] == float(excavation.nodes[bf.node_i,
                                                                1])
    m = [abs(v) for v in t.column("Bending moment M i")
         + t.column("Bending moment M j")]
    assert max(m) == pytest.approx(excavation.max_beam_moment_kNm_per_m)


def test_strut_forces_csv_only(excavation):
    t = _by_name(excavation.get_tables())["strut_forces"]
    assert not t.printed
    assert t.column("Force") == [s["force_kN_per_m"]
                                 for s in excavation.strut_forces]


def test_srm_trials_are_the_history():
    from fem2d.results import FEMResult
    hist = [{"srf": 1.0, "max_disp_m": 0.01, "dimensionless_disp": 0.5,
             "converged": True, "failed": False, "n_iter": 12,
             "settled": False, "final_residual": 1e-6},
            {"srf": 1.4, "max_disp_m": 0.09, "dimensionless_disp": 4.0,
             "converged": False, "failed": True, "n_iter": 200,
             "settled": False, "final_residual": 1e-2}]
    r = FEMResult(analysis_type="srm", FOS=1.3, srf_history=hist)
    (t,) = r.get_tables()
    assert not t.printed
    assert t.column("SRF") == [1.0, 1.4]
    assert t.column("Failed") == [False, True]


def test_stage_movements_are_the_phase_summaries():
    nodes, elements = generate_rect_mesh(-10, 10, -10, 0, 20, 10)
    n_corner = len(nodes)
    nodes, elements = convert_to_t6(nodes, elements)
    bc = detect_boundary_nodes(nodes)
    surf = np.where(np.abs(nodes[:n_corner, 1]) < 1e-9)[0]
    foot = surf[np.abs(nodes[surf, 0]) <= 1.0 + 1e-9]
    foot = foot[np.argsort(nodes[foot, 0])]
    edges = t6_boundary_edges(
        elements, [(foot[i], foot[i + 1]) for i in range(len(foot) - 1)])
    mats = [{'E': 20000.0, 'nu': 0.3, 'c': 1e6, 'phi': 0.0,
             'psi': 0.0}] * len(elements)
    groups = {'all': list(range(len(elements)))}
    phases = [
        ConstructionPhase(name="Gravity", active_soil_groups=['all']),
        ConstructionPhase(name="Footing", active_soil_groups=['all'],
                          surface_loads=[(edges, 0.0, -100.0)]),
    ]
    res = analyze_staged(nodes, elements, mats, 18.0, bc, groups, phases)
    (t,) = res.get_tables()
    assert t.printed and t.n_rows == 2
    for row, p in zip(t.rows, res.phases):
        assert row[t.column_index("This stage: settlement")] == \
            p.delta_displacement["settlement_m"]
        assert row[t.column_index("Cumulative settlement")] == \
            p.cumulative_displacement["settlement_m"]
    assert t.column("Phase") == ["Gravity", "Footing"]


def test_consolidation_curve_is_the_results():
    r = analyze_consolidation(
        width=2.0, depth=10.0,
        soil_layers=[{"E": 5000.0, "nu": 0.3, "gamma": 18.0,
                      "bottom_elevation": 0.0}],
        k=1e-9, load_q=50.0, time_points=[1e4, 1e6, 1e8], gwt=10.0,
        nx=2, ny=10)
    (t,) = r.get_tables()
    assert t.column("Time t") == [float(x) for x in r.times]
    assert t.column("Surface settlement") == [float(s) for s in
                                              r.settlements]
