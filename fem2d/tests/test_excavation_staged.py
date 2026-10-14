"""A staged excavation in front of a wall (GM-71, 2026-10-14).

Until 2026-10-14 ``analyze_excavation`` switched off the self-weight of the
excavated elements and kept their stiffness, in one gravity solve: the wall
head moved INTO the retained soil and the wall forces were not those of an
excavation. It now runs on ``analyze_staged``: in-situ K0 stresses, then the
excavated elements removed lift by lift with the forces they carried
released, the wall wished in place and each strut installed stress-free.

Benchmark: USS Steel Sheet Piling Design Manual (United States Steel, July
1984, reprinted by FHWA), Design Examples, "Design of Cantilevered Sheet
Pile Wall - Granular Soil", pp. 86-90: H = 14 ft, gamma = 115 pcf above the
dredge line, gamma' = 65 pcf below it (low water at the dredge line on both
sides, no net water), phi = 35 deg, Ka = 0.27, Kp = 6.56 (delta/phi = -0.5,
Fig. 5A). By the conventional assumed pressure diagram and statics
(p. 87): D = 10.5 ft (used 13.5 ft, +28.5 %); zero shear y + x = 1.0 + 4.0
ft below the dredge line and M_max = 26,000 ft-lb/ft (pp. 88-89).

The FEM wall is smooth (no wall friction, ``wall_contact='smooth'``), so the
same method is applied with the Rankine Kp (delta = 0) and the wall is
lengthened by the manual's own rule (D = 1.3 D0; "20 to 40 %", p. 88).
"""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from fem2d import analyze_excavation
from fem2d.analysis import ConstructionPhase, analyze_staged
from fem2d.elements import BeamElement
from fem2d.mesh import (convert_to_t6, detect_boundary_nodes,
                        generate_rect_mesh)
from fem2d.porewater import standing_water_forces

FT = 0.3048                 # m
PCF = 0.157087464           # kN/m3 per pcf
FTLB = 0.00444822162        # kN-m/m per ft-lb/ft


def uss_cantilever(H, g, gp, Ka, Kp):
    """The manual's conventional method for a cantilever in granular soil
    (pp. 86-89): D0 from sum M = 0 (p. 87), then the zero-shear point and
    M_max (pp. 88-89). Any consistent units."""
    PA1 = g * H * Ka

    def parts(D):
        PA2 = PA1 + gp * D * Ka
        PE = gp * D * (Kp - Ka) - PA1
        PJ = gp * D * (Kp - Ka) + g * H * Kp
        Z = ((PE - PA1) * D - H * PA1) / (PE + PJ)
        M = (0.5 * H * PA1 * (D + H / 3) + PA1 * D ** 2 / 2
             + (PE + PJ) * Z ** 2 / 6 - (PE + PA2) * D ** 2 / 6
             + (PA2 - PA1) * D ** 2 / 6)
        return M, Z

    D0 = brentq(lambda D: parts(D)[0], 0.3 * H, 5 * H)
    y = PA1 / (gp * (Kp - Ka))
    P1, P2 = 0.5 * PA1 * H, 0.5 * PA1 * y
    x = math.sqrt(2 * (P1 + P2) / (gp * (Kp - Ka)))
    M_max = P1 * (H / 3 + y + x) + P2 * (2 * y / 3 + x) - (P1 + P2) * x / 3
    return {"D0": D0, "Z_at": lambda D: parts(D)[1], "y": y, "x": x,
            "M_max": M_max}


# ---------------------------------------------------------------------------
# 1. The limit-equilibrium helper reproduces the published example
# ---------------------------------------------------------------------------

def test_uss_example_reproduced():
    le = uss_cantilever(14.0, 115.0, 65.0, 0.27, 6.56)
    assert le["D0"] == pytest.approx(10.5, rel=0.01)       # 10.43 ft
    assert le["Z_at"](10.5) == pytest.approx(1.60, abs=0.01)
    assert le["y"] == pytest.approx(1.06, abs=0.01)
    assert le["x"] == pytest.approx(4.0, abs=0.01)
    # published 26,000 with y and x rounded to 1.0 and 4.0 ft
    assert le["M_max"] == pytest.approx(26000, rel=0.02)   # 26,334


# ---------------------------------------------------------------------------
# 2. The FEM cantilever against the same method with delta = 0
# ---------------------------------------------------------------------------

H = 14 * FT
KA = math.tan(math.radians(27.5)) ** 2          # 0.271 (the manual's 0.27)
KP = math.tan(math.radians(62.5)) ** 2          # 3.690, Rankine
K0 = 1 - math.sin(math.radians(35.0))           # 0.426, Jaky
LE = uss_cantilever(H, 115 * PCF, 65 * PCF, KA, KP)      # m, kN-m/m
D_WALL = 1.3 * LE["D0"]                         # 5.84 m
SAND = [
    {"name": "Medium sand above the dredge line", "bottom_elevation": -H,
     "E": 40000, "nu": 0.3, "gamma": 115 * PCF, "c": 0.0, "phi": 35.0,
     "psi": 0.0},
    {"name": "Medium sand below (effective unit weight)",
     "bottom_elevation": -60.0, "E": 40000, "nu": 0.3, "gamma": 65 * PCF,
     "c": 0.0, "phi": 35.0, "psi": 0.0},
]
PZ27_EI = 50300.0       # kN-m2/m: I = 184.2 in4/ft, E = 29,000 ksi
PZ27_EA = 3.36e6        # kN/m: A = 7.94 in2/ft


@pytest.fixture(scope="module")
def cantilever():
    return analyze_excavation(
        width=50.0, depth=H, wall_depth=H + D_WALL, soil_layers=SAND,
        wall_EI=PZ27_EI, wall_EA=PZ27_EA)


def _wall_rows(r):
    """(depth of node i, depth of node j, beam force) top to bottom."""
    return [(-r.nodes[b.node_i, 1], -r.nodes[b.node_j, 1], b)
            for b in r.beam_forces]


def _moment_at_dredge(r):
    for zi, zj, b in _wall_rows(r):
        if abs(zj - H) < 1e-6:
            return abs(b.moment_j)
    raise AssertionError("no wall node at the excavation level")


def test_le_values_for_the_fem_wall():
    assert LE["D0"] / FT == pytest.approx(14.73, abs=0.02)
    assert LE["M_max"] / FTLB == pytest.approx(33804, rel=1e-3)
    assert LE["M_max"] == pytest.approx(150.4, abs=0.1)


def test_converges(cantilever):
    assert cantilever.converged
    assert cantilever.analysis_type == "excavation"


def test_in_situ_stage_does_not_move(cantilever):
    """The K0 stresses are in equilibrium with the weight: stage 0 settles
    nothing."""
    d0 = cantilever.staged.phases[0].delta_displacement
    assert d0["total_m"] < 1e-9


def test_wall_moves_toward_the_excavation(cantilever):
    r = cantilever
    wall = [b.node_i for b in r.beam_forces] + [r.beam_forces[-1].node_j]
    ux = np.asarray(r.displacements)[2 * np.array(wall)]
    depth = -r.nodes[wall, 1]
    assert np.all(ux[depth <= H + 1e-9] > 0)
    assert ux[0] == pytest.approx(ux.max())                 # the head
    # Caltrans T&S Manual (2025) Table 4-1: active is reached at
    # Delta/H = 0.002 in medium dense sand
    assert ux[0] / H > 0.002
    assert r.excavation_stages[-1]["wall_head_ux_m"] == pytest.approx(ux[0])


def test_retained_side_is_at_active(cantilever):
    """The wall moved far beyond 0.002 H, so the retained side is at the
    active limit: between Ka and K0, and within a few percent of Ka (a
    smooth wall: Rankine)."""
    m_d = _moment_at_dredge(cantilever)
    gamma = 115 * PCF
    k_mob = 6.0 * m_d / (gamma * H ** 3)            # M = K gamma H^3 / 6
    assert KA <= k_mob < K0
    assert k_mob == pytest.approx(KA, rel=0.10)     # 0.280
    # the soil next to the wall: sigma_x / sigma_y = Ka
    r = cantilever
    cent = r.nodes[r.elements].mean(axis=1)
    touches = np.any(np.abs(r.nodes[r.elements][:, :, 0]) < 1e-9, axis=1)
    sel = np.where(touches & (cent[:, 0] < 0)
                   & (cent[:, 1] > -0.85 * H))[0]
    ratio = r.stresses[sel, 0] / r.stresses[sel, 1]
    assert len(sel) >= 4
    np.testing.assert_allclose(ratio, KA, rtol=0.05)


def test_moment_matches_the_conventional_method(cantilever):
    """M_max within 10 % of the manual's method with the FEM's wall
    friction (delta = 0): 155.1 against 150.4 kN-m/m (+3 %), at about the
    zero-shear depth."""
    r = cantilever
    m_le = LE["M_max"]
    assert r.max_beam_moment_kNm_per_m == pytest.approx(m_le, rel=0.10)
    z_le = H + LE["y"] + LE["x"]                     # 6.57 m
    zi, zj, b = max(_wall_rows(r), key=lambda t: max(abs(t[2].moment_i),
                                                     abs(t[2].moment_j)))
    z_fem = zj if abs(b.moment_j) >= abs(b.moment_i) else zi
    assert z_fem == pytest.approx(z_le, abs=0.5)


def test_smooth_wall_carries_no_drag(cantilever):
    assert max(abs(b.axial_i) for b in cantilever.beam_forces) < 0.5


def test_result_mesh_is_the_soil_left(cantilever):
    r = cantilever
    cent = r.nodes[r.elements].mean(axis=1)
    assert not np.any((cent[:, 0] > 0) & (cent[:, 1] > -H))
    assert r.n_elements == len(r.elements) == len(r.stresses)
    assert r.n_elements < len(r.staged.elements)


def test_result_reports_the_stages(cantilever):
    r = cantilever
    d = r.to_dict()
    assert [s["name"] for s in d["excavation_stages"]] == [
        f"Excavate to {H:.2f} m"]
    assert d["wall_envelope"]["max_beam_moment_kNm_per_m"] == pytest.approx(
        r.max_beam_moment_kNm_per_m)
    assert any("K0 = 0.426" in n for n in d["notes"])
    # The notes name the wall contact actually modelled (live smoke
    # 2026-10-14, F51: they said "shares nodes ... fully rough" for the
    # default smooth wall, and the agent called its answer unconservative).
    contact = [n for n in d["notes"] if "wall_contact=" in n]
    assert len(contact) == 1 and "smooth wall" in contact[0]
    assert "shares nodes" not in contact[0]
    assert "Excavation stages" in r.summary()
    tables = {t.name: t for t in r.get_tables()}
    t = tables["excavation_stages"]
    assert t.printed
    assert t.column("Wall head u_x") == [
        s["wall_head_ux_m"] for s in r.excavation_stages]


def test_bonded_wall_hangs_the_retained_soil():
    """Why 'smooth' is the default: a wall sharing the soil's nodes takes
    the retained soil's weight in axial force (more than the horizontal
    thrust above the excavation level, which no wall friction angle up to
    phi' allows) and the thrust falls below the active value."""
    r = analyze_excavation(
        width=50.0, depth=H, wall_depth=H + D_WALL, soil_layers=SAND,
        wall_EI=PZ27_EI, wall_EA=PZ27_EA, nx=20, ny=10,
        wall_contact="bonded")
    above = [b for zi, zj, b in _wall_rows(r) if zj <= H + 1e-6]
    thrust = abs(above[-1].shear_j)
    n_max = max(abs(b.axial_j) for b in above)
    assert n_max > math.tan(math.radians(35.0)) * thrust
    k_mob = 6.0 * _moment_at_dredge(r) / (115 * PCF * H ** 3)
    assert k_mob < KA
    assert any("shares nodes" in n and "wall_contact='bonded'" in n
               for n in r.notes)


# ---------------------------------------------------------------------------
# 3. Lifts and struts (a clay that stands; quick meshes)
# ---------------------------------------------------------------------------

CLAY = [{"name": "Clay", "bottom_elevation": -20, "E": 30000, "nu": 0.3,
         "gamma": 18, "c": 25, "phi": 20, "psi": 0}]


@pytest.fixture(scope="module")
def propped():
    return analyze_excavation(
        width=10, depth=5, wall_depth=10, soil_layers=CLAY, wall_EI=50000,
        wall_EA=5e6, nx=20, ny=10, n_steps=5,
        struts=[{"depth": 1.5, "stiffness": 50000}])


def test_default_lifts_stop_at_the_strut(propped):
    names = [p.phase_name for p in propped.staged.phases]
    assert names == ["In-situ stresses (K0)", "Excavate to 1.50 m",
                     "Excavate to 5.00 m (struts at 1.50 m installed)"]
    assert propped.converged


def test_strut_carries_only_the_movement_after_installation(propped):
    r = propped
    sf = r.strut_forces[0]
    assert sf["installed_in_stage"] == 2
    node = sf["node_id"]
    u1 = np.asarray(r.staged.phases[1].displacements)[2 * node]
    u2 = np.asarray(r.staged.phases[2].displacements)[2 * node]
    assert sf["force_kN_per_m"] == pytest.approx(50000 * (u2 - u1),
                                                 rel=1e-6)
    assert sf["force_kN_per_m"] > 0                 # holds the wall back
    assert r.excavation_stages[0]["strut_forces_kN_per_m"] == []
    assert r.excavation_stages[1]["strut_forces_kN_per_m"] == [
        pytest.approx(sf["force_kN_per_m"])]


def test_wall_moves_toward_the_excavation_in_every_lift(propped):
    for st in propped.excavation_stages:
        assert st["max_wall_ux_m"] > 0


def test_calc_package_shows_the_stages(propped):
    from fem2d.calc_steps import get_calc_steps, get_figures
    sections = get_calc_steps(propped, {})
    titles = [sec.title for sec in sections]
    assert "Excavation Stages" in titles
    table = [i for i in sections[titles.index("Excavation Stages")].items
             if getattr(i, "title", "") == "Wall by Stage"][0]
    assert len(table.rows) == len(propped.excavation_stages)
    assert len(get_figures(propped, {})) >= 3


def test_strut_at_the_ground_is_installed_before_digging():
    r = analyze_excavation(
        width=10, depth=5, wall_depth=10, soil_layers=CLAY, wall_EI=50000,
        wall_EA=5e6, nx=16, ny=8, n_steps=3,
        struts=[{"depth": 0.0, "stiffness": 50000}])
    assert len(r.staged.phases) == 2
    assert r.strut_forces[0]["installed_in_stage"] == 1
    assert r.strut_forces[0]["force_kN_per_m"] > 0


def test_a_stop_is_added_at_a_strut_level():
    r = analyze_excavation(
        width=10, depth=5, wall_depth=10, soil_layers=CLAY, wall_EI=50000,
        wall_EA=5e6, nx=16, ny=8, n_steps=3, lifts=[2.0],
        struts=[{"depth": 3.0, "stiffness": 50000}])
    assert [s["excavation_depth_m"] for s in r.excavation_stages] == [
        2.0, 3.0, 5.0]
    assert r.excavation_stages[2]["struts_installed_m"] == [3.0]


@pytest.mark.parametrize("kwargs, match", [
    (dict(wall_depth=5.0), "wall_depth"),
    (dict(struts=[{"depth": 5.0, "stiffness": 1e4}]), "above the final"),
    (dict(lifts=[3.0, 2.0]), "increasing"),
    (dict(lifts=[6.0]), "below the excavation depth"),
    (dict(wall_contact="rough"), "wall_contact"),
])
def test_bad_input_is_refused(kwargs, match):
    args = dict(width=10, depth=5, wall_depth=10, soil_layers=CLAY,
                wall_EI=50000, wall_EA=5e6, nx=10, ny=6)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        analyze_excavation(**args)


def test_water_above_the_excavation_level_is_said():
    r = analyze_excavation(
        width=10, depth=3, wall_depth=8, soil_layers=CLAY, wall_EI=50000,
        wall_EA=5e6, nx=12, ny=8, n_steps=3, gwt=-1.0)
    assert r.converged
    assert any("full of water" in w for w in r.warnings)
    dry = analyze_excavation(
        width=10, depth=3, wall_depth=8, soil_layers=CLAY, wall_EI=50000,
        wall_EA=5e6, nx=12, ny=8, n_steps=3, gwt=-4.0)
    assert not any("full of water" in w for w in dry.warnings)


# ---------------------------------------------------------------------------
# 4. The staged path underneath
# ---------------------------------------------------------------------------

def _block(nx=8, ny=4):
    nodes, elements = generate_rect_mesh(0.0, 8.0, -4.0, 0.0, nx, ny)
    nodes, elements = convert_to_t6(nodes, elements)
    return nodes, elements, detect_boundary_nodes(nodes)


def test_beam_placed_later_carries_only_later_movement():
    """GM-61: gravity, then a wall installed with nothing else changed.
    Before 2026-10-14 the 'no beams' gravity phase still had the beams'
    stiffness, and the install phase reported forces from every movement
    since the start (the backlog item measured 186 kN/m axial and
    9 kN-m/m with no movement in the stage)."""
    nodes, elements, bc = _block()
    col = np.where(np.abs(nodes[:, 0] - 4.0) < 1e-9)[0]
    col = col[np.argsort(-nodes[col, 1])][:5]
    beams = [BeamElement(node_i=int(a), node_j=int(b), EA=5e6, EI=5e4)
             for a, b in zip(col[:-1], col[1:])]
    mats = [{'E': 20000.0, 'nu': 0.3, 'c': 0.0, 'phi': 0.0}] * len(elements)
    groups = {'all': list(range(len(elements)))}
    phases = [ConstructionPhase("Gravity", ['all'], n_steps=2),
              ConstructionPhase("Install wall", ['all'],
                                active_beam_ids=list(range(len(beams))),
                                n_steps=2)]
    r = analyze_staged(nodes, elements, mats, 18.0, bc, groups, phases,
                       beam_elements=beams)
    assert r.converged
    assert r.phases[0].beam_forces is None
    for bf in r.phases[1].beam_forces:
        assert abs(bf.axial_i) < 1e-6 and abs(bf.moment_i) < 1e-6
    # and gravity with the beams left out is gravity without beams
    r0 = analyze_staged(nodes, elements, mats, 18.0, bc, groups,
                        [phases[0]])
    np.testing.assert_allclose(r.phases[0].displacements,
                               r0.phases[0].displacements, atol=1e-12)


def test_spring_placed_later_carries_only_later_movement():
    nodes, elements, bc = _block()
    top = int(np.where((np.abs(nodes[:, 0] - 4.0) < 1e-9)
                       & (np.abs(nodes[:, 1]) < 1e-9))[0][0])
    mats = [{'E': 20000.0, 'nu': 0.3, 'c': 0.0, 'phi': 0.0}] * len(elements)
    groups = {'all': list(range(len(elements)))}
    phases = [ConstructionPhase("Gravity", ['all'], n_steps=1),
              ConstructionPhase("Prop", ['all'], active_spring_ids=[0],
                                n_steps=1)]
    springs = [{'node': top, 'kx': 1e4, 'ky': 1e4}]
    r = analyze_staged(nodes, elements, mats, 18.0, bc, groups, phases,
                       springs=springs)
    sf1 = r.phases[1].spring_forces[0]
    assert abs(sf1['force_y_kN_per_m']) < 1e-9      # nothing changed
    assert sf1['placed_in_phase'] == 1


def test_ponded_water_does_not_lift_the_ground():
    """Water standing on the ground changes total stress and pore pressure
    alike: the effective stresses and movements of a level saturated
    ground do not change (Terzaghi's principle). With the water's load on
    the free surface (default) they do not; without it, the pore pressure
    at the surface pushes the ground up."""
    nodes, elements, bc = _block()
    mats = [{'E': 20000.0, 'nu': 0.3, 'c': 0.0, 'phi': 0.0}] * len(elements)
    groups = {'all': list(range(len(elements)))}

    def run(gwt, water=True):
        ph = [ConstructionPhase("Gravity", ['all'], gwt=gwt, n_steps=1)]
        return analyze_staged(nodes, elements, mats, 20.0, bc, groups, ph,
                              water_on_free_faces=water,
                              reset_displacements_after="start")

    at_surface = run(0.0).phases[0].displacements
    ponded = run(2.0).phases[0].displacements
    np.testing.assert_allclose(ponded, at_surface, atol=1e-9)
    lifted = run(2.0, water=False).phases[0].displacements
    assert np.max(lifted[1::2] - at_surface[1::2]) > 1e-4


def test_standing_water_is_minus_p_n():
    """One T6 element face under 10 kPa: the resultant is p L, inward."""
    nodes, elements = generate_rect_mesh(0.0, 2.0, -1.0, 0.0, 1, 1)
    nodes, elements = convert_to_t6(nodes, elements)
    pp = np.where(np.abs(nodes[:, 1]) < 1e-9, 10.0, 0.0)
    F = standing_water_forces(nodes, elements, pp)
    top = np.abs(nodes[:, 1]) < 1e-9
    assert F[1::2][top].sum() == pytest.approx(-10.0 * 2.0)
    assert abs(F[0::2].sum()) < 1e-9
