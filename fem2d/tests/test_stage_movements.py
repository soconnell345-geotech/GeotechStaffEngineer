"""Movements per construction stage (owner direction, 2026-10-09).

"FEM is flexible and could consider self weight depending on construction
sequence. Construction phasing will determine output (ie output the delta
movement per stage)."

Every staged analysis, and the consolidation column, reports per stage:
its own (delta) movement, the cumulative movement from the start, and the
movement since a reference stage (``reset_displacements_after``; default:
after the first / initial stage, with a ``judgment`` record when that
default changes the numbers materially).

Checks are against independent solves on the same mesh (linear elasticity,
so superposition is exact) and the 1-D closed forms of the consolidation
column:
- a gravity stage then a footing-load stage: the load stage's delta equals
  the load-only solve; the gravity stage's equals the gravity-only solve;
  cumulative = gravity + load (to round-off);
- a fill group placed in a later stage: its stage delta equals the solve
  under the fill's self-weight alone (the fill is placed stress-free);
- a stage in which nothing changes moves nothing (Gauss-point state carry);
- the column: initial stage = (gamma - gamma_w) H^2 / (2 M_oed), the load
  stage unchanged from G6 (U = 1.000 CON-1 key), cumulative = the sum.
"""

import numpy as np
import pytest

from fem2d.analysis import (
    ConstructionPhase, analyze_staged, assign_element_groups,
    analyze_consolidation,
)
from fem2d.materials import elastic_D
from fem2d.mesh import (
    generate_rect_mesh, detect_boundary_nodes, convert_to_t6,
    t6_boundary_edges,
)
from fem2d.solver import solve_elastic

E, NU, GAMMA = 20000.0, 0.3, 18.0
ELASTIC = {'E': E, 'nu': NU, 'c': 1e6, 'phi': 0.0, 'psi': 0.0}


def _footing_model():
    nodes, elements = generate_rect_mesh(-10, 10, -10, 0, 20, 10)
    n_corner = len(nodes)
    nodes, elements = convert_to_t6(nodes, elements)
    bc = detect_boundary_nodes(nodes)
    surf = np.where(np.abs(nodes[:n_corner, 1]) < 1e-9)[0]
    foot = surf[np.abs(nodes[surf, 0]) <= 1.0 + 1e-9]
    foot = foot[np.argsort(nodes[foot, 0])]
    edges = t6_boundary_edges(
        elements, [(foot[i], foot[i + 1]) for i in range(len(foot) - 1)])
    return nodes, elements, bc, edges


def _footing_phases(edges, **kw0):
    return [
        ConstructionPhase(name="Gravity", active_soil_groups=['all'], **kw0),
        ConstructionPhase(name="Footing", active_soil_groups=['all'],
                          surface_loads=[(edges, 0.0, -100.0)]),
    ]


@pytest.fixture(scope="module")
def footing():
    nodes, elements, bc, edges = _footing_model()
    mats = [ELASTIC] * len(elements)
    groups = {'all': list(range(len(elements)))}
    res = analyze_staged(nodes, elements, mats, GAMMA, bc, groups,
                         _footing_phases(edges))
    D = elastic_D(E, NU)
    u_grav, _, _ = solve_elastic(nodes, elements, D, GAMMA, bc)
    u_load, _, _ = solve_elastic(nodes, elements, D, 0.0, bc,
                                 surface_loads=[(edges, 0.0, -100.0)])
    return dict(nodes=nodes, elements=elements, bc=bc, edges=edges,
                mats=mats, groups=groups, res=res,
                u_grav=u_grav, u_load=u_load)


class TestGravityThenFooting:
    def test_load_stage_delta_is_the_load_only_result(self, footing):
        p1 = footing['res'].phases[1]
        scale = np.abs(footing['u_load']).max()
        np.testing.assert_allclose(p1.delta_displacements,
                                   footing['u_load'], atol=1e-9 * scale)

    def test_gravity_stage_is_its_own_stage(self, footing):
        p0 = footing['res'].phases[0]
        scale = np.abs(footing['u_grav']).max()
        np.testing.assert_allclose(p0.delta_displacements,
                                   footing['u_grav'], atol=1e-9 * scale)
        assert p0.cumulative_displacement == p0.delta_displacement

    def test_cumulative_is_gravity_plus_load(self, footing):
        p1 = footing['res'].phases[1]
        total = footing['u_grav'] + footing['u_load']
        np.testing.assert_allclose(p1.displacements, total,
                                   atol=1e-9 * np.abs(total).max())
        uy = total[1::2]
        assert p1.cumulative_displacement['settlement_m'] == \
            pytest.approx(uy.min(), abs=1e-6)

    def test_summaries_and_reference(self, footing):
        d = footing['res'].to_dict()
        p1 = d['phases'][1]
        u_load_y = footing['u_load'][1::2]
        assert p1['delta_displacement']['settlement_m'] == \
            pytest.approx(u_load_y.min(), abs=1e-6)
        # Under the footing centre, on the surface.
        assert p1['delta_displacement']['settlement_at_xy'] == [0.0, 0.0]
        # Default: reset after stage 0 -> since-reference = this stage.
        assert p1['displacement_since_reference'] == p1['delta_displacement']
        assert p1['reference_stage'] == "end of stage 0 'Gravity'"
        assert d['phases'][0]['reference_stage'] == "start of analysis"
        assert d['displacement_reference'] == {
            "reset_after_stages": [0],
            "measured_from": "end of stage 0 'Gravity'",
            "chosen_by": "default"}

    def test_backward_compatible_cumulative_keys(self, footing):
        p1 = footing['res'].phases[1]
        assert p1.max_displacement_m == pytest.approx(
            p1.cumulative_displacement['total_m'], abs=1e-6)

    def test_judgment_when_the_default_changes_the_numbers(self, footing):
        d = footing['res'].to_dict()
        j = d['judgment']
        assert set(j) == {"question", "options", "used", "why"}
        assert j['question'] == "Which stage are movements measured from?"
        names = [o['name'] for o in j['options']]
        assert names == ["from end of the initial (gravity) stage",
                         "cumulative from start"]
        assert j['used'] == names[0]
        for o in j['options']:
            assert set(o) == {"name", "source", "assumptions", "applies",
                              "result"}
            assert o['applies'] is True
        s_delta = abs(d['phases'][1]['delta_displacement']['settlement_m'])
        s_cum = abs(d['phases'][1]['cumulative_displacement']['settlement_m'])
        assert f"settlement {s_delta * 1000:.1f} mm" in j['options'][0]['result']
        assert f"settlement {s_cum * 1000:.1f} mm" in j['options'][1]['result']
        assert "reset_displacements_after" in j['why']


class TestReferenceOptions:
    def _run(self, footing, phases=None, **kw):
        f = footing
        return analyze_staged(
            f['nodes'], f['elements'], f['mats'], GAMMA, f['bc'],
            f['groups'], phases or _footing_phases(f['edges']), **kw)

    def test_start_reports_cumulative_and_no_judgment(self, footing):
        r = self._run(footing, reset_displacements_after="start")
        p1 = r.phases[1]
        assert p1.displacement_since_reference == p1.cumulative_displacement
        assert p1.reference_stage == "start of analysis"
        assert r.judgment is None
        assert r.displacement_reference['chosen_by'] == "user"
        assert r.displacement_reference['reset_after_stages'] == []

    def test_explicit_choice_has_no_judgment(self, footing):
        for ref in (0, "Gravity", "initial"):
            r = self._run(footing, reset_displacements_after=ref)
            assert r.judgment is None, ref
            p1 = r.phases[1]
            assert p1.displacement_since_reference == p1.delta_displacement

    def test_phase_reset_flag_is_the_users_choice(self, footing):
        phases = _footing_phases(footing['edges'])
        phases[1].reset_displacements = True
        r = self._run(footing, phases=phases)
        assert r.judgment is None
        assert r.displacement_reference['chosen_by'] == "user"
        p1 = r.phases[1]
        assert p1.displacement_since_reference == p1.delta_displacement

    def test_reset_is_reporting_only(self, footing):
        """Resetting never changes the solution (it used to zero u in the
        solver, which re-converged to the same total anyway)."""
        phases = _footing_phases(footing['edges'])
        phases[1].reset_displacements = True
        r = self._run(footing, phases=phases)
        np.testing.assert_allclose(r.phases[1].displacements,
                                   footing['res'].phases[1].displacements,
                                   atol=1e-12)

    def test_three_stages_reset_after_the_middle_one(self, footing):
        edges = footing['edges']
        phases = _footing_phases(edges) + [
            ConstructionPhase(name="More load", active_soil_groups=['all'],
                              surface_loads=[(edges, 0.0, -150.0)])]
        r = self._run(footing, phases=phases, reset_displacements_after=1)
        p0, p1, p2 = r.phases
        # Stages up to the reference are measured from the start.
        np.testing.assert_allclose(p1.displacements_since_reference,
                                   p1.displacements)
        np.testing.assert_allclose(p2.displacements_since_reference,
                                   p2.displacements - p1.displacements,
                                   atol=1e-15)
        # The third stage's own movement is the extra 50 kPa: 0.5 x load.
        np.testing.assert_allclose(p2.delta_displacements,
                                   0.5 * footing['u_load'],
                                   atol=1e-9 * np.abs(footing['u_load']).max())

    @pytest.mark.parametrize("bad", [5, -1, "nope", True, 0.5])
    def test_bad_reference_rejected(self, footing, bad):
        with pytest.raises(ValueError, match="reset_displacements_after"):
            self._run(footing, reset_displacements_after=bad)

    def test_weightless_first_stage_needs_no_judgment(self, footing):
        f = footing
        r = analyze_staged(f['nodes'], f['elements'], f['mats'], 0.0, f['bc'],
                           f['groups'], _footing_phases(f['edges']))
        assert r.judgment is None
        assert r.phases[0].delta_displacement['total_m'] == 0.0


class TestFillPlacedInAStage:
    """Soil placed during construction: its self-weight movement belongs in
    the stage that places it, and it is placed stress-free (its strain
    counts from placement, not from the movement its shared nodes made
    before it existed)."""

    @pytest.fixture(scope="class")
    def fill(self):
        nodes, elements = generate_rect_mesh(0, 10, -8, 0, 10, 8)
        nodes, elements = convert_to_t6(nodes, elements)
        bc = detect_boundary_nodes(nodes)
        groups = assign_element_groups(nodes, elements, {
            'native': {'y_max': -2.0}, 'fill': {'y_min': -2.0}})
        mats = [ELASTIC] * len(elements)
        phases = [
            ConstructionPhase(name="Gravity", active_soil_groups=['native']),
            ConstructionPhase(name="Place fill",
                              active_soil_groups=['native', 'fill']),
        ]
        r = analyze_staged(nodes, elements, mats, GAMMA, bc, groups, phases)
        g_fill = np.zeros(len(elements))
        g_fill[groups['fill']] = GAMMA
        u_fill, _, _ = solve_elastic(nodes, elements, elastic_D(E, NU),
                                     g_fill, bc)
        return r, u_fill

    def test_fill_stage_delta_is_the_fill_weight_alone(self, fill):
        r, u_fill = fill
        np.testing.assert_allclose(r.phases[1].delta_displacements, u_fill,
                                   atol=1e-9 * np.abs(u_fill).max())

    def test_fill_settlement_is_reported_in_its_stage(self, fill):
        r, u_fill = fill
        d1 = r.to_dict()['phases'][1]['delta_displacement']
        assert d1['settlement_m'] == pytest.approx(u_fill[1::2].min(),
                                                   abs=1e-6)
        assert d1['settlement_m'] < -1e-3


def test_a_stage_that_changes_nothing_moves_nothing():
    """Plastic T6 state is carried per Gauss point: element averages moved
    a no-change stage by ~0.08 mm (1.5 % of the gravity settlement)."""
    nodes, elements = generate_rect_mesh(0, 6, -4, 0, 6, 4)
    nodes, elements = convert_to_t6(nodes, elements)
    bc = detect_boundary_nodes(nodes)
    mats = [{'E': E, 'nu': NU, 'c': 1.0, 'phi': 20.0, 'psi': 0.0}] \
        * len(elements)
    groups = {'all': list(range(len(elements)))}
    phases = [ConstructionPhase(name="Gravity", active_soil_groups=['all']),
              ConstructionPhase(name="No change", active_soil_groups=['all'])]
    r = analyze_staged(nodes, elements, mats, GAMMA, bc, groups, phases,
                       max_iter=500)
    assert r.converged
    assert r.phases[0].delta_displacement['settlement_m'] < -1e-3
    assert np.abs(r.phases[1].delta_displacements).max() < 1e-9


# ---------------------------------------------------------------------------
# Consolidation column: initial (self-weight) stage + load stage
# ---------------------------------------------------------------------------

E_C, NU_C, M_W = 529412.0, 0.3235, 4e6
M_OED = E_C * (1 - NU_C) / ((1 + NU_C) * (1 - 2 * NU_C))


def _con1(**kw):
    """The live CON-1 column (G6): 20 m, 100 kPa, water 20 m above the
    surface (ponded), monolithic."""
    args = dict(width=2.0, depth=20.0,
                soil_layers=[{"E": E_C, "nu": NU_C, "gamma": 18.0,
                              "bottom_elevation": 0.0}],
                k=1e-10, load_q=100.0, time_points=[1e3, 1e7, 1e8],
                gwt=20.0, n_w=M_W, nx=4, ny=40,
                consolidation_scheme="monolithic", theta=0.5)
    args.update(kw)
    return analyze_consolidation(**args)


@pytest.fixture(scope="module")
def con1():
    return _con1()


class TestConsolidationStages:
    def test_load_stage_unchanged_and_reported_by_default(self, con1):
        d = con1.to_dict()
        assert d['degree_of_consolidation'] == 1.0          # CON-1 key
        load = d['stages'][1]
        assert load['surface_settlement_m_by_time'] == \
            d['surface_settlement_m_by_time']
        assert d['surface_settlement_m_by_time'][-1] == pytest.approx(
            -100.0 * 20.0 / M_OED, rel=0.01)
        assert d['displacement_reference'] == {
            "reset_after_stage": 0,
            "measured_from": "end of stage 0 'initial' (self-weight)",
            "chosen_by": "default"}

    def test_initial_stage_is_the_buoyant_self_weight(self, con1):
        """Ponded water (gwt 20 m above the surface) loads the surface, so
        sigma'v = (gamma - gamma_w) z and s0 = gamma' H^2 / (2 M_oed)."""
        s0 = con1.to_dict()['stages'][0]['surface_settlement_m']
        assert s0 == pytest.approx(-(18.0 - 9.81) * 20.0 ** 2 / (2 * M_OED),
                                   rel=0.005)

    def test_cumulative_is_initial_plus_load_at_every_time(self, con1):
        st = con1.to_dict()['stages']
        s0 = st[0]['surface_settlement_m']
        for s, c in zip(st[1]['surface_settlement_m_by_time'],
                        st[1]['cumulative_surface_settlement_m_by_time']):
            assert c == pytest.approx(s0 + s, abs=2e-6)
        assert st[1]['degree_of_consolidation_by_time'] == \
            con1.to_dict()['degree_of_consolidation_by_time']

    def test_judgment_on_the_default(self, con1):
        j = con1.to_dict()['judgment']
        assert j['used'] == "from end of the initial (gravity) stage"
        st = con1.to_dict()['stages']
        load = abs(st[1]['surface_settlement_m']) * 1000
        cum = abs(st[1]['cumulative_surface_settlement_m']) * 1000
        assert f"settlement {load:.1f} mm" in j['options'][0]['result']
        assert f"settlement {cum:.1f} mm" in j['options'][1]['result']

    def test_start_reports_cumulative(self, con1):
        r = _con1(reset_displacements_after="start")
        d = r.to_dict()
        assert 'judgment' not in d
        assert d['surface_settlement_m_by_time'] == \
            d['stages'][1]['cumulative_surface_settlement_m_by_time']
        assert d['max_settlement_m'] == pytest.approx(
            d['stages'][1]['cumulative_surface_settlement_m'], abs=1e-6)
        # U is the load stage's and does not depend on the reference.
        assert d['degree_of_consolidation_by_time'] == \
            con1.to_dict()['degree_of_consolidation_by_time']

    def test_explicit_initial_has_no_judgment(self):
        assert 'judgment' not in _con1(
            reset_displacements_after="initial").to_dict()

    def test_bad_reference_rejected(self):
        with pytest.raises(ValueError, match="reset_displacements_after"):
            _con1(reset_displacements_after="load")

    def test_weightless_column_has_no_initial_movement(self):
        r = _con1(soil_layers=[{"E": E_C, "nu": NU_C, "gamma": 0.0,
                                "bottom_elevation": -20.0}], gwt=0.0)
        d = r.to_dict()
        assert d['stages'][0]['surface_settlement_m'] == 0.0
        assert 'judgment' not in d


def test_staggered_reports_the_load_stage_by_default():
    """The staggered scheme used to report settlement including
    self-weight; it now follows the reference stage like the monolithic
    one. Drained 1-D: load stage q H / M_oed, initial gamma' H^2/(2 M_oed)."""
    H, q, Ec, nuc = 10.0, 50.0, 10000.0, 0.3
    m = Ec * (1 - nuc) / ((1 + nuc) * (1 - 2 * nuc))
    r = analyze_consolidation(
        width=2.0, depth=H, soil_layers=[{"E": Ec, "nu": nuc, "gamma": 18}],
        k=1e-5, load_q=q, time_points=[100, 1000], nx=4, ny=8)
    d = r.to_dict()
    assert d['surface_settlement_m_by_time'][-1] == pytest.approx(
        -q * H / m, rel=0.01)
    assert d['stages'][0]['surface_settlement_m'] == pytest.approx(
        -(18 - 9.81) * H ** 2 / (2 * m), rel=0.02)
    assert d['stages'][1]['cumulative_surface_settlement_m'] == \
        pytest.approx(d['stages'][0]['surface_settlement_m']
                      + d['stages'][1]['surface_settlement_m'], abs=2e-6)
    assert 'judgment' in d


def _fill_column(gwt=-2.0, fill_top=True):
    fill = {"name": "fill", "E": E_C, "nu": NU_C, "gamma": 18.0,
            "bottom_elevation": -2.0, "fill": fill_top}
    clay = {"name": "clay", "E": E_C, "nu": NU_C, "gamma": 18.0,
            "bottom_elevation": -20.0, "fill": not fill_top}
    return analyze_consolidation(
        width=2.0, depth=20.0, soil_layers=[fill, clay],
        k=1e-10, load_q=100.0, time_points=[1e8], gwt=gwt, n_w=M_W, nx=4,
        ny=40, consolidation_scheme="monolithic", theta=0.5)


def test_fill_layer_self_weight_moves_into_the_load_stage():
    """A 2 m top layer flagged fill, placed with the load on 18 m of clay
    with the water table at the clay surface: the initial stage carries the
    clay's buoyant self-weight only, and the load stage's drained end state
    carries q + the (dry) fill's weight:
      s_initial = g' 18^2 / (2 M);  s_load = (20 q + 38 g) / M."""
    q, g, gp = 100.0, 18.0, 18.0 - 9.81
    d = _fill_column().to_dict()
    assert d['stages'][0]['surface_settlement_m'] == pytest.approx(
        -gp * 18.0 ** 2 / (2 * M_OED), rel=0.01)
    assert d['final_drained_settlement_m'] == pytest.approx(
        -(20 * q + 38 * g) / M_OED, rel=0.01)
    assert d['surface_settlement_m_by_time'][-1] == pytest.approx(
        d['final_drained_settlement_m'], rel=0.01)
    assert "fill" in d['stages'][1]['description']
    # The fill weight is applied undrained: p0 exceeds the load-only one.
    assert d['max_excess_pore_pressure_kPa_by_time'][0] > \
        _con1(time_points=[1e8], gwt=0.0).to_dict()[
            'max_excess_pore_pressure_kPa_by_time'][0]


def test_underwater_fill_is_refused():
    with pytest.raises(ValueError, match="above the water table"):
        _fill_column(gwt=0.0)


def test_fill_must_be_the_top_layer():
    with pytest.raises(ValueError, match="TOP layers"):
        _fill_column(fill_top=False)
