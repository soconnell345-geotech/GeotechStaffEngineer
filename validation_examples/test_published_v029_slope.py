"""Phase E / v5.4 E3 validation — pore-pressure GRID / TIN input.

Slide2 Verification #10 = ACADS 5 (Giam & Donald 1989) [manual pp. 55-58]: a
homogeneous 1:2 excavation (beta=26.56 deg, c'=11 kPa / phi'=28 deg /
gamma=20 kN/m3) whose pore pressure is supplied as a discrete GRID interpolated
from the Figure 10.2 flow net, plus ponded water. Published (Table 10.2): Bishop
1.498, Spencer 1.500, GLE 1.500, Janbu-corrected 1.457; referee 1.53 [Giam].

V-029 was previously **N/A (scope)**: the module had NO pore-pressure-grid input
(only a piezometric surface ``gwt_points`` and a per-layer ``ru``). E3 adds the
capability — ``SlopeGeometry.pore_pressure_points`` (scattered (x, z, u) triples),
interpolated at each slice base by ``build_pore_pressure_interpolator`` (linear on
the Delaunay triangulation, nearest-node fallback outside the hull / for
degenerate point sets, suction clamped to 0). It overrides the piezometric-line /
ru base pore pressure and is wired through the search path; the ponded-water
buttress still comes from ``gwt_points`` (set both for a pool over a flow-net
field).

VERDICT: **CAPABILITY BUILT (flips V-029's N/A-scope) — validated by construction.**
The published 1.498 itself is NOT pinned here: the manual's Figure 10.2 flow-net
grid and the excavation extent did not survive text extraction, so reproducing the
exact referee surface would require inventing the un-extractable inputs (forbidden).
Instead the capability is validated RIGOROUSLY and honestly:
  * a grid encoding a hydrostatic field reproduces the ``gwt_points`` piezometric-
    line base pore pressure to machine precision (identical FOS), so the new path
    is exact where the two overlap;
  * TIN linear interpolation is exact for a linear field; the nearest fallback
    returns finite, non-negative values outside the hull;
  * on a reconstructed 1:2 ACADS-5-style section a flow-net-style grid lowers the
    FOS the expected way (dry ~1.89 -> ~1.25). The reconstructed number is a
    capability DEMONSTRATION, not a published match. NOT tuned.

See slope_stability/geometry.py (pore_pressure_points / build_pore_pressure_
interpolator), slices.py (slice-base override), VALIDATION.md B8, INVENTORY V-029.
"""

import pytest

from slope_stability.geometry import (
    SlopeGeometry, SlopeSoilLayer, build_pore_pressure_interpolator,
)
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.slices import build_slices
from slope_stability.analysis import analyze_slope, search_critical_surface
from geotech_common.water import GAMMA_W


# ── equivalence geometry (WT above the toe bench -> ponding present) ─────────
_SURF = [(0, 0), (10, 0), (30, 10), (50, 10)]
_WT = 5.0
_GWT = [(-1.0, _WT), (51.0, _WT)]
# a valid deep circle whose base (el -0.4) is well below the WT (nonzero u)
_XC, _YC, _R = 14.29, 21.86, 22.27


def _geom(gwt=None, ppp=None):
    return SlopeGeometry(
        surface_points=_SURF,
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=10.0,
                                    bottom_elevation=-8.0, gamma=20.0,
                                    phi=28.0, c_prime=11.0)],
        gwt_points=gwt, pore_pressure_points=ppp)


def _hydrostatic_grid(wt):
    """Grid of (x, z, u) reproducing the hydrostatic field u = gw*max(wt-z, 0)."""
    return [(float(x), float(z), GAMMA_W * max(wt - z, 0.0))
            for x in range(-2, 53, 5) for z in [-8, -5, 0, 5, 10]]


def test_v029_grid_reproduces_piezo_line_base_pressure():
    """With the SAME gwt_points on both (so ponding + submerged weight match),
    a grid encoding the identical hydrostatic field reproduces the piezometric-
    line base pore pressure per slice, and hence the FOS, to machine precision."""
    ppp = _hydrostatic_grid(_WT)
    slip = CircularSlipSurface(_XC, _YC, _R)
    s_line = build_slices(_geom(gwt=_GWT), slip, 40)
    s_grid = build_slices(_geom(gwt=_GWT, ppp=ppp), slip, 40)
    assert sum(s.pore_pressure for s in s_line) > 500.0          # nonzero field
    for a, b in zip(s_line, s_grid):
        assert a.pore_pressure == pytest.approx(b.pore_pressure, abs=1e-6)
    for m in ("bishop", "spencer", "gle"):
        f_line = analyze_slope(_geom(gwt=_GWT), xc=_XC, yc=_YC, radius=_R,
                               method=m, n_slices=40).FOS
        f_grid = analyze_slope(_geom(gwt=_GWT, ppp=ppp), xc=_XC, yc=_YC,
                               radius=_R, method=m, n_slices=40).FOS
        assert f_grid == pytest.approx(f_line, abs=1e-6)


def test_v029_grid_lowers_fos_vs_dry():
    """The grid pore pressure reduces the FOS relative to the dry slope on the
    same surface (the whole point of supplying a flow-net field)."""
    ppp = _hydrostatic_grid(_WT)
    f_dry = analyze_slope(_geom(), xc=_XC, yc=_YC, radius=_R,
                          method="bishop", n_slices=40).FOS
    f_grid = analyze_slope(_geom(ppp=ppp), xc=_XC, yc=_YC, radius=_R,
                           method="bishop", n_slices=40).FOS
    assert f_grid < f_dry


def test_v029_tin_linear_interpolation_is_exact():
    """Linear interpolation on the Delaunay triangulation is EXACT for a linear
    field u = 3 + 0.5x + 2z (barycentric linear reproduces a plane)."""
    def u(x, z):
        return 3.0 + 0.5 * x + 2.0 * z
    pts = [(x, z, u(x, z)) for x in (0, 5, 10, 15) for z in (0, 4, 8)]
    interp = build_pore_pressure_interpolator(pts)
    for x in (2.0, 7.5, 12.3):
        for z in (1.0, 3.5, 6.2):
            assert interp(x, z) == pytest.approx(max(u(x, z), 0.0), abs=1e-9)


def test_v029_fallback_and_suction_clamp():
    """Outside the convex hull the nearest-node value is used (finite, >= 0); a
    degenerate (< 3-point) set falls back to nearest; suction (negative u)
    is clamped to 0."""
    interp = build_pore_pressure_interpolator([(0, 0, 10.0), (10, 0, 20.0),
                                               (5, 10, 30.0)])
    v = interp(100.0, 100.0)
    assert v == v and v >= 0.0                                   # finite, clamped
    two = build_pore_pressure_interpolator([(0, 0, 42.0), (10, 0, 7.0)])
    assert two(0.5, 0.5) == pytest.approx(42.0)                  # nearest of two
    suction = build_pore_pressure_interpolator([(0, 0, -50.0), (1, 0, -50.0),
                                                (0, 1, -50.0)])
    assert suction(0.3, 0.3) == 0.0                              # clamp <0 -> 0


def test_v029_grid_wires_through_search():
    """A critical-surface search evaluates the grid pore pressure per trial
    surface and returns a valid, sane minimum."""
    ppp = _hydrostatic_grid(_WT)
    res = search_critical_surface(
        _geom(ppp=ppp), method="bishop", surface_type="circular",
        nx=6, ny=6, n_slices=25, x_entry_range=(0.0, 12.0),
        x_exit_range=(28.0, 50.0))
    assert res.critical is not None
    assert 0.5 < res.critical.FOS < 2.5


# ── ACADS-5-style capability demonstration (reconstructed section) ───────────
_A5_SURF = [(0, 0), (20, 0), (40, 10), (60, 10)]     # 1:2 cut, 10 m high
_A5_XC, _A5_YC, _A5_R = 25.714, 19.143, 19.101       # dry critical circle


def _a5_geom(ppp=None):
    return SlopeGeometry(
        surface_points=_A5_SURF,
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=10.0,
                                    bottom_elevation=-6.0, gamma=20.0,
                                    phi=28.0, c_prime=11.0)],
        pore_pressure_points=ppp)


def test_v029_acads5_flownet_grid_demonstration():
    """Capability demonstration on the ACADS-5 soil/slope (c'=11/phi'=28/gamma=20,
    1:2 cut) with a flow-net-STYLE declining-phreatic pore-pressure grid. This
    exercises the grid on a realistic excavation; it is NOT a match to the
    published 1.498 (the manual's exact flow net is not extractable). The grid
    lowers the dry FOS ~1.89 to ~1.25."""
    def phreatic(x):
        return 2.0 + 6.0 * max(min(x / 60.0, 1.0), 0.0)
    ppp = [(float(x), float(z), GAMMA_W * max(phreatic(x) - z, 0.0))
           for x in range(-5, 66, 5) for z in [-6, -3, 0, 2, 4, 6, 8, 10]]
    f_dry = analyze_slope(_a5_geom(), xc=_A5_XC, yc=_A5_YC, radius=_A5_R,
                          method="bishop", n_slices=40).FOS
    f_grid = analyze_slope(_a5_geom(ppp), xc=_A5_XC, yc=_A5_YC, radius=_A5_R,
                           method="bishop", n_slices=40).FOS
    assert f_dry == pytest.approx(1.893, abs=0.05)      # ours dry (pinned)
    assert f_grid == pytest.approx(1.247, abs=0.05)     # ours grid (pinned)
    assert f_grid < f_dry                               # flow-net field destabilises
