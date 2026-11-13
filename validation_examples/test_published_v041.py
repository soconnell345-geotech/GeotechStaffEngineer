"""Phase E / v5.4 E1 validation — rapid-drawdown CRITICAL-SURFACE SEARCH.

``slope_stability.search_rapid_drawdown`` finds the minimum-FOS slip surface with
the *rapid-drawdown* strength substituted per trial (Corps 2-stage / Duncan-
Wright-Wong 3-stage), reusing the module's existing search machinery
(grid / entry-exit / random / DE) through their new ``fos_fn`` hook — no search
internals are duplicated. This closes the V5.3 "B2a-search" follow-up that left
Slide2 #98 (Walter Bouldin) DEFERRED because its published values are search
MINIMA and ``rapid_drawdown_fos`` scores only one specified surface.

Two validation legs:

WRAPPER MECHANICS (exact geometry) — Slide2 #95/#96 homogeneous EM 1110-2-1902
dam (gamma=135 pcf, c'=0/phi'=30, R cR=1200 psf/phiR=16; face (0,0)-(220,73)-
(312,110)-(380,110) ft). On this EXACT section a circular search finds a minimum
(FOS 0.90) at or below the published specified circle's single-surface FOS
(1.207 at the flat-pool default), and the stage-level detail recomputed on the
winning surface reproduces the search FOS to machine precision — i.e. the search
truly evaluates the drawdown strength per surface. (The #95/#96 published 1.347 /
1.443 are single-SPECIFIED-surface results, validated in V-037/V-038, not search
minima, so they are not the target here.)

#98 WALTER BOULDIN (approximate geometry) — the published SEARCH minima are
Corps 2-stage 0.931 and Duncan-Wright-Wong 3-stage 1.039 (Duncan, Wright & Wong
1990). The cross-section is only RECOVERED (INVENTORY): flat-stacked layers
(Clayey Sandy Gravel 0-17 / Cretaceous Clay 17-30 / Micaceous Sand 30-51 / Clayey
Silty Sand 51-60 ft), with the real pinch-outs simplified and the riprap face
veneer omitted; the exact figure is Rocscience's.

VERDICT (#98): **CONVENTION (search validated; geometry approximate).** With the
recovered section the search minima are Corps 0.836 and DWW 0.951 (0.935
before GM-140 step 9 put every stage as EM 1110-2-1902 Appendix G prints it),
10 % and 8.5 % BELOW the published 0.931 / 1.039, with the correct method
ordering (DWW > Corps). Being low in every procedure and across two search
types is consistent with the recovered section (flat layers, no riprap
veneer) and not with the wrapper, whose mechanics are exact on the #95/#96
section above. NOT tuned; the recovered geometry is left as-is.

See slope_stability/rapid_drawdown.py::search_rapid_drawdown, VALIDATION.md
(V-041), and validation_examples/INVENTORY.md (#98).
"""

import math

import pytest

from geotech_common.water import GAMMA_W
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.rapid_drawdown import (
    search_rapid_drawdown, rapid_drawdown_fos, RapidDrawdownSearchResult,
)
from slope_stability.search_supplements import CIRCULAR, INFINITE_SLOPE

FT, PSF, PCF = 0.3048, 0.04788, 0.157087

# ── #95/#96 homogeneous dam (EXACT geometry) ────────────────────────────────
_FACE = [(0, 0), (220, 73), (312, 110), (380, 110)]


def _dam_9596():
    return SlopeGeometry(
        surface_points=[(x * FT, z * FT) for x, z in _FACE],
        soil_layers=[SlopeSoilLayer(
            name="fill", top_elevation=110 * FT, bottom_elevation=-1.0 * FT,
            gamma=135 * PCF, phi=30.0, c_prime=0.0,
            R_c=1200 * PSF, R_phi=16.0)])


# ── #98 Walter Bouldin (APPROXIMATE recovered geometry, INVENTORY) ──────────
_SURF_98 = [(0, 0), (100, 40), (140, 60), (180, 60)]


def _dam_98():
    return SlopeGeometry(
        surface_points=[(x * FT, z * FT) for x, z in _SURF_98],
        soil_layers=[
            SlopeSoilLayer(name="ClayeySiltySand", top_elevation=60 * FT,
                           bottom_elevation=51 * FT, gamma=128 * PCF,
                           c_prime=240 * PSF, phi=32.7, R_c=650 * PSF, R_phi=13.0),
            SlopeSoilLayer(name="MicaceousSand", top_elevation=51 * FT,
                           bottom_elevation=30 * FT, gamma=123 * PCF,
                           c_prime=220 * PSF, phi=22.5, R_c=450 * PSF, R_phi=11.0),
            SlopeSoilLayer(name="CretaceousClay", top_elevation=30 * FT,
                           bottom_elevation=17 * FT, gamma=124 * PCF,
                           c_prime=180 * PSF, phi=19.0, R_c=180 * PSF, R_phi=13.0),
            SlopeSoilLayer(name="ClayeySandyGravel", top_elevation=17 * FT,
                           bottom_elevation=-2 * FT, gamma=125 * PCF,
                           c_prime=0.0, phi=40.0, R_phi=None),   # free-draining
        ])


def _search_98(method):
    return search_rapid_drawdown(
        _dam_98(), 47 * FT, 15 * FT, method=method, surface_type="circular",
        nx=6, ny=5, x_range=(30 * FT, 140 * FT), y_range=(60 * FT, 150 * FT),
        n_slices=20)


# ── WRAPPER MECHANICS (exact #95/#96 geometry) ──────────────────────────────

def test_v041_search_finds_min_at_or_below_specified_surface():
    """A circular drawdown search on the EXACT #95/#96 section finds circles
    (least 0.90) at or below the published specified circle's single-surface
    FOS (1.207, flat-pool default), proving the drawdown strength is
    evaluated per trial surface inside the search loop. A shallow planar
    slide on the upper face governs at 0.783, against Corps 2-stage's
    infinite slope, (gamma' / gamma) tan(phi') / tan(beta) = 0.772. Before
    GM-131 sliced a polyline at its corners, no planar surface solved the
    full-pool stage, so that family dropped out of the search and the
    circle (0.895) was reported."""
    r = search_rapid_drawdown(
        _dam_9596(), 110 * FT, 24 * FT, method="corps_2stage",
        surface_type="circular", nx=4, ny=3,
        x_range=(140 * FT, 200 * FT), y_range=(190 * FT, 240 * FT), n_slices=20)
    single = rapid_drawdown_fos(
        _dam_9596(), 110 * FT, 24 * FT, xc=169.5 * FT, yc=210 * FT,
        radius=210 * FT, method="corps_2stage", n_slices=20).FOS
    assert isinstance(r, RapidDrawdownSearchResult)
    assert r.critical is not None
    assert single == pytest.approx(1.207, abs=0.02)      # V-037 single surface
    assert r.search.candidates[CIRCULAR] == pytest.approx(0.902, abs=0.03)
    gamma = 135 * PCF
    closed = ((gamma - GAMMA_W) / gamma * math.tan(math.radians(30.0))
              / ((110 - 73) / (312 - 220)))              # 0.772
    assert r.search.governing_surface == INFINITE_SLOPE
    assert r.FOS == pytest.approx(closed, rel=0.02)
    assert r.FOS <= single + 1e-6                        # search is >= as critical
    assert r.n_surfaces_evaluated > 0


def test_v041_detail_reproduces_search_fos():
    """The stage-level drawdown detail recomputed on the winning surface equals
    the search minimum (the search and the single-surface solve are the same
    computation on that surface)."""
    r = search_rapid_drawdown(
        _dam_9596(), 110 * FT, 24 * FT, method="corps_2stage",
        surface_type="circular", nx=4, ny=3,
        x_range=(140 * FT, 200 * FT), y_range=(190 * FT, 240 * FT), n_slices=20)
    assert r.drawdown is not None
    assert r.drawdown.FOS == pytest.approx(r.FOS, abs=1e-9)
    assert r.drawdown.stage1_fos > 1.0                   # full pool is stable
    assert len(r.drawdown.sigma_fc) == r.drawdown.n_slices
    d = r.to_dict()
    assert d["FOS"] == pytest.approx(r.FOS, abs=1e-4)
    assert "drawdown_detail" in d and "search" in d


# ── #98 WALTER BOULDIN published search minima ──────────────────────────────

def test_v041_98_corps_2stage_search_minimum():
    """#98 Corps 2-stage search minimum = 0.837 vs published 0.931 (~10% low);
    residual = approximate recovered geometry (flat layers, no riprap veneer),
    not the wrapper (mechanics exact above). NOT tuned."""
    r = _search_98("corps_2stage")
    assert r.FOS == pytest.approx(0.837, abs=0.03)       # ours (pinned)
    assert r.FOS == pytest.approx(0.931, rel=0.15)       # within 15% of published
    assert r.critical is not None and r.critical.is_circular


def test_v041_98_duncan_3stage_and_ordering():
    """#98 DWW 3-stage search minimum = 0.951 vs published 1.039 (8.5 % low), and
    stays ABOVE the Corps 2-stage minimum (published ordering DWW > Corps)."""
    r_c = _search_98("corps_2stage")
    r_d = _search_98("duncan_3stage")
    # GM-140 step 9: 0.935 -> 0.951; printed 1.039 (Slide2 #98; Duncan,
    # Wright and Wong 1990), -8.5 %, on a recovered section
    assert r_d.FOS == pytest.approx(0.951, abs=0.03)
    assert r_d.FOS == pytest.approx(1.039, rel=0.15)     # within 15% of published
    assert r_d.FOS > r_c.FOS                             # DWW > Corps (published)


# ── API / robustness ────────────────────────────────────────────────────────

def test_v041_noncircular_search_runs():
    """The noncircular leg reuses the same machinery and returns a valid,
    non-degenerate drawdown minimum (guards reject sliver / non-converged
    surfaces)."""
    r = search_rapid_drawdown(
        _dam_98(), 47 * FT, 15 * FT, method="corps_2stage",
        surface_type="noncircular", n_trials=150, n_points=5, seed=3,
        x_entry_range=(0 * FT, 60 * FT), x_exit_range=(60 * FT, 150 * FT),
        n_slices=20)
    assert r.critical is not None
    assert 0.3 < r.FOS < 2.5                             # a real, non-artifact FOS
    assert r.critical.slip_points is not None            # noncircular surface


def test_v041_bad_arguments_rejected():
    g = _dam_98()
    with pytest.raises(ValueError):
        search_rapid_drawdown(g, 47 * FT, 15 * FT, method="bogus")
    with pytest.raises(ValueError):
        search_rapid_drawdown(g, 47 * FT, 15 * FT, surface_type="bogus")
    with pytest.raises(ValueError):
        search_rapid_drawdown(g, 15 * FT, 47 * FT)       # to >= from
