"""GM-13 / GM-14: the critical-surface search must not miss the toe circle
of a steep slope or the shallow planar (infinite-slope) failure of a
cohesionless face. Both misses reported a factor of safety that was too HIGH.

GM-13 (V-054 isotropic anchor): H = 5 m at 45 deg, gamma = 20, su = 40 kPa,
the model starting at the toe. The true critical is the toe circle
(2.5, 4.0, R = 4.717), Bishop 2.287 (brute force; published 2.16 FLAC /
Taylor ~2.2). The automatic centre grid gave ~2.78: the circle exits the
crest at x = 7.1, inside the automatic exit window's lower bound (15), and
its centre lies below the crest, under the automatic centre grid.

GM-14 (Slide2 #86 / Duncan & Wright 2005 Fig 7.28, supports removed):
c' = 0, phi' = 37 deg, gamma = 130 pcf fill, face (0,0)-(30,24) ft on rock.
The infinite slope gives tan 37 / (24/30) = 0.942; the circular search gave
~1.57 because every circle it tries is deeper than the shallow planar slide.
"""

import math

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import (
    analyze_slope, search_critical_surface, infinite_slope_fos,
)
from slope_stability.search_supplements import (
    slope_faces, cohesionless_face_runs, check_infinite_slope,
    search_toe_circles, TOE_CIRCLE, INFINITE_SLOPE, CIRCULAR, NONCIRCULAR,
)

_TAN_RATIO_86 = math.tan(math.radians(37.0)) / (24.0 / 30.0)   # 0.9419


def _toe_geom(mirror=False):
    pts = [(0, 0), (5, 5), (25, 5)]
    if mirror:
        pts = [(-x, z) for x, z in reversed(pts)]
    return SlopeGeometry(
        surface_points=pts,
        soil_layers=[SlopeSoilLayer(
            name="clay", top_elevation=5.0, bottom_elevation=-10.0,
            gamma=20.0, analysis_mode="undrained", cu=40.0)])


def _fill_geom(mirror=False, ru=0.0):
    pts = [(0, 0), (30, 24), (75, 24)]
    if mirror:
        pts = [(-x, z) for x, z in reversed(pts)]
    return SlopeGeometry(
        surface_points=pts,
        soil_layers=[SlopeSoilLayer(
            name="fill", top_elevation=24.0, bottom_elevation=0.0,
            gamma=130.0, phi=37.0, c_prime=0.0, ru=ru)])


# ---------------------------------------------------------------------------
# GM-13 toe circles
# ---------------------------------------------------------------------------

def test_brute_force_toe_circle_value():
    """The anchor: the toe circle (2.5, 4.0, R = 4.717) is Bishop 2.287."""
    f = analyze_slope(_toe_geom(), method="bishop", n_slices=40, xc=2.5,
                      yc=4.0, radius=math.hypot(2.5, 4.0)).FOS
    assert f == pytest.approx(2.287, abs=0.002)


def test_search_finds_toe_circle():
    """GM-13: the default search now reports the toe circle (~2.287), not
    the window-limited grid circle (~2.78), and says so."""
    res = search_critical_surface(_toe_geom(), method="bishop", n_slices=40)
    c = res.critical
    assert c.FOS == pytest.approx(2.287, abs=0.02)
    assert c.FOS <= 2.287 + 0.005
    assert res.governing_surface == TOE_CIRCLE
    assert c.is_circular
    assert c.x_entry == pytest.approx(0.0, abs=0.05)       # through the toe
    assert c.x_exit == pytest.approx(7.1, abs=0.6)
    assert res.candidates[CIRCULAR] > 2.7                  # the old answer
    assert res.candidates[TOE_CIRCLE] == pytest.approx(c.FOS, abs=1e-4)
    assert any("toe circle governs" in w for w in res.warnings)


def test_toe_circle_mirrored_slope():
    """Same slope falling toward +x: the same toe circle and FOS."""
    res = search_critical_surface(_toe_geom(mirror=True), method="bishop",
                                  n_slices=40)
    assert res.critical.FOS == pytest.approx(2.287, abs=0.02)
    assert res.governing_surface == TOE_CIRCLE
    assert res.critical.x_exit == pytest.approx(0.0, abs=0.05)


def test_toe_circles_off_reproduces_old_search():
    """toe_circles=False is the previous behaviour, with an edge warning: the
    window-limited circle exits on the lower edge of the automatic exit
    window."""
    res = search_critical_surface(_toe_geom(), method="bishop", n_slices=40,
                                  toe_circles=False)
    assert res.critical.FOS == pytest.approx(2.784, abs=0.01)
    assert res.governing_surface == CIRCULAR
    assert any("edge of the automatic x_exit_range" in w
               for w in res.warnings)


def test_toe_circles_honour_given_windows():
    """A caller-given exit window that excludes the critical toe circle's
    crest end keeps it out: the best toe circle allowed ends on the window
    edge (still below the grid's 2.78), and the edge is reported."""
    res = search_critical_surface(_toe_geom(), method="bishop", n_slices=40,
                                  x_entry_range=(0.0, 10.0),
                                  x_exit_range=(15.0, 25.0))
    c = res.critical
    assert 2.6 < c.FOS < res.candidates[CIRCULAR]
    assert 15.0 <= c.x_exit <= 15.5
    assert any("edge of the given x_exit_range" in w for w in res.warnings)


def test_slope_faces_benched():
    """Each bench face has its own toe; flat stretches end a face."""
    g = SlopeGeometry(
        surface_points=[(0, 0), (10, 10), (20, 10), (30, 20), (40, 20)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=20,
                                    bottom_elevation=-10, gamma=19,
                                    phi=30, c_prime=10)])
    faces = slope_faces(g)
    assert [f["toe"] for f in faces] == [(0, 0), (20, 10)]
    assert [f["crest"] for f in faces] == [(10, 10), (30, 20)]
    assert all(f["rises"] == 1 for f in faces)


def test_toe_search_respects_centre_ranges():
    """A caller-given centre range is honoured by the toe-circle family."""
    out = search_toe_circles(_toe_geom(), n_slices=40, y_range=(6.0, 15.0))
    assert out["FOS"] > 2.287 + 0.01
    assert out["yc"] >= 6.0


# ---------------------------------------------------------------------------
# GM-14 infinite slope (shallow planar) in cohesionless soil
# ---------------------------------------------------------------------------

def test_search_finds_infinite_slope_86():
    """GM-14: Slide2 #86 without supports -- the default search reports the
    shallow planar slide at tan(phi')/tan(beta) = 0.942, not ~1.57."""
    res = search_critical_surface(_fill_geom(), method="bishop")
    c = res.critical
    assert c.FOS == pytest.approx(_TAN_RATIO_86, abs=0.01)
    assert res.governing_surface == INFINITE_SLOPE
    assert not c.is_circular and len(c.slip_points) == 4
    assert c.method == "Janbu"           # force equilibrium, as the closed form
    assert res.candidates[CIRCULAR] > 1.5        # the old answer
    chk = res.infinite_slope_checks
    assert len(chk) == 1
    assert chk[0]["layer"] == "fill"
    assert chk[0]["slope_angle_deg"] == pytest.approx(38.66, abs=0.01)
    assert chk[0]["FOS_dry_closed_form"] == pytest.approx(_TAN_RATIO_86,
                                                          abs=1e-4)
    assert any("infinite slope" in w for w in res.warnings)
    d = res.to_dict()
    assert d["governing_surface"] == INFINITE_SLOPE
    assert d["critical"]["surface_type"] == "noncircular"
    assert d["infinite_slope_checks"][0]["FOS"] == pytest.approx(
        c.FOS, abs=1e-3)


def _closed_form(beta_deg, phi_deg, kh=0.0, ru=0.0):
    """Infinite slope, c' = 0, vertical-column ru, kh acting downslope
    (Duncan, Wright & Brandon 2014, Ch. 14):
    F = tan phi' (cos b - kh sin b - ru / cos b) / (sin b + kh cos b)."""
    b = math.radians(beta_deg)
    return (math.tan(math.radians(phi_deg))
            * (math.cos(b) - kh * math.sin(b) - ru / math.cos(b))
            / (math.sin(b) + kh * math.cos(b)))


def test_infinite_slope_matches_closed_form():
    """The planar surfaces are real LE surfaces, scored by force
    equilibrium (Janbu), and land within 0.3 % above the closed form."""
    beta = math.degrees(math.atan(0.8))
    closed = infinite_slope_fos(beta, 37.0, 130.0).FOS
    assert closed == pytest.approx(_closed_form(beta, 37.0), rel=1e-9)
    out = check_infinite_slope(_fill_geom())
    assert closed <= out["FOS"] <= closed * 1.003


def test_infinite_slope_honours_pore_pressure_and_kh():
    """With ru on the layer, or a pseudo-static kh, the planar surface still
    gives the closed form: (1 - ru/cos^2 b) tan phi'/tan b with ru (as
    infinite_slope_fos 'ru'), (cos b - kh sin b) tan phi'/(sin b + kh cos b)
    with kh -- also with the face falling toward +x."""
    beta = math.degrees(math.atan(0.8))
    closed_ru = infinite_slope_fos(beta, 37.0, 130.0, water_condition="ru",
                                   ru=0.2).FOS
    out = check_infinite_slope(_fill_geom(ru=0.2))
    assert closed_ru <= out["FOS"] <= closed_ru * 1.003
    for mirror in (False, True):
        g = _fill_geom(mirror=mirror)
        g.kh = 0.1
        out = check_infinite_slope(g)
        closed_kh = _closed_form(beta, 37.0, kh=0.1)
        assert closed_kh <= out["FOS"] <= closed_kh * 1.003, mirror


def test_infinite_slope_mirrored_and_noncircular_search():
    """The face falling toward +x, and a noncircular search, both report it."""
    res = search_critical_surface(_fill_geom(mirror=True), method="spencer",
                                  surface_type="noncircular", n_trials=100,
                                  seed=1)
    assert res.critical.FOS == pytest.approx(_TAN_RATIO_86, abs=0.01)
    assert res.governing_surface == INFINITE_SLOPE
    assert res.candidates[NONCIRCULAR] > res.critical.FOS


def test_infinite_slope_excluded_by_windows_is_warned():
    """Windows that leave the face out (the 'global failure' windows of a
    supported wall) keep the planar slide out -- and the result says the
    infinite slope may govern."""
    res = search_critical_surface(_fill_geom(), method="bishop",
                                  x_entry_range=(0.0, 3.0),
                                  x_exit_range=(40.0, 75.0))
    assert res.governing_surface in (CIRCULAR, TOE_CIRCLE)
    assert res.critical.FOS > 1.0
    assert res.critical.x_exit >= 40.0 - 1e-6
    assert res.infinite_slope_checks[0]["FOS"] is None
    assert "x_exit_range" in res.infinite_slope_checks[0]["note"]
    assert any("infinite slope may govern" in w for w in res.warnings)


def test_infinite_slope_off_and_cohesive_soil():
    """infinite_slope=False skips the family; a c' > 0 face has no
    cohesionless run at all."""
    res = search_critical_surface(_fill_geom(), method="bishop",
                                  infinite_slope=False, toe_circles=False)
    assert res.governing_surface == CIRCULAR
    assert res.critical.FOS > 1.5
    g = _fill_geom()
    g.soil_layers[0].c_prime = 5.0
    assert cohesionless_face_runs(g) == []


def test_cohesionless_runs_follow_the_outcrop():
    """A face cutting a cohesive layer over a sand layer: only the sand
    outcrop is a run."""
    g = SlopeGeometry(
        surface_points=[(0, 0), (20, 20), (40, 20)],
        soil_layers=[
            SlopeSoilLayer(name="clay", top_elevation=20, bottom_elevation=10,
                           gamma=19, phi=25, c_prime=15),
            SlopeSoilLayer(name="sand", top_elevation=10, bottom_elevation=-5,
                           gamma=19, phi=34, c_prime=0)])
    runs = cohesionless_face_runs(g)
    assert len(runs) == 1
    assert runs[0]["layer"] == "sand"
    assert runs[0]["x_from"] == pytest.approx(0.0, abs=0.01)
    assert runs[0]["x_to"] == pytest.approx(10.0, abs=0.5)
