"""Unit tests for the rapid-drawdown method (v5.3 B2a): USACE 2-stage and
Duncan-Wright-Wong 3-stage. These pin the framework BEHAVIOUR (the stub is
closed; both methods run; undrained drawdown is more critical than drained;
the 3-stage is less conservative than the 2-stage; free-draining layers stay
drained). Quantitative validation vs Slide2 #95/#96 lives in
validation_examples/test_published_v037_slope.py.
"""

import math

import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import rapid_drawdown_fos
from slope_stability.rapid_drawdown import RapidDrawdownResult

FT, PSF, PCF = 0.3048, 0.04788, 0.157087


def _dam(R=True):
    """Homogeneous embankment (EM 1110-2-1902 Appendix G): 3:1-ish upstream face,
    crest el 110 ft, base el 0. Material gamma=135 pcf, c'=0/phi'=30, R-envelope
    cR=1200 psf/phiR=16 (unless R=False -> free-draining)."""
    lay = SlopeSoilLayer(
        name="fill", top_elevation=110 * FT, bottom_elevation=-5.0,
        gamma=135 * PCF, phi=30.0, c_prime=0.0,
        R_c=(1200 * PSF if R else 0.0), R_phi=(16.0 if R else None))
    return SlopeGeometry(
        surface_points=[(0.0, 0.0), (305 * FT, 110 * FT), (380 * FT, 110 * FT)],
        soil_layers=[lay])


def _rd(geom, method):
    return rapid_drawdown_fos(geom, 110 * FT, 24 * FT, xc=169.5 * FT,
                              yc=210 * FT, radius=210 * FT, method=method,
                              n_slices=50)


def test_stub_is_closed_and_returns_result():
    r = _rd(_dam(), "corps_2stage")
    assert isinstance(r, RapidDrawdownResult)
    assert r.FOS > 0
    assert r.stage1_fos > r.FOS       # full pool is more stable than drawdown
    d = r.to_dict()
    assert d["method"] == "corps_2stage" and "stage1_FOS" in d


def test_convergence_flags_reported():
    """Both GLE stages converge on the well-posed dam, so the flags are True and
    no warning is emitted (regression guard for the dead-ternary fix)."""
    r = _rd(_dam(), "duncan_3stage")
    assert r.stage1_converged is True and r.stage3_converged is True
    assert r.warnings == []
    d = r.to_dict()
    assert d["stage3_converged"] is True and "warnings" not in d


def test_rapid_drawdown_more_critical_than_drained():
    """The undrained rapid-drawdown FOS is well below the free-draining (drained)
    drawdown FOS -- the whole point of the analysis."""
    undrained = _rd(_dam(R=True), "corps_2stage").FOS
    drained = _rd(_dam(R=False), "corps_2stage").FOS
    assert undrained < 0.75 * drained


def test_3stage_less_conservative_than_2stage():
    """Duncan-Wright-Wong 3-stage (Kc-interpolated undrained strength) gives a
    HIGHER FOS than the USACE 2-stage combined envelope."""
    f2 = _rd(_dam(), "corps_2stage").FOS
    f3 = _rd(_dam(), "duncan_3stage").FOS
    assert f3 >= f2
    assert 1.0 < f2 < 1.6 and 1.0 < f3 < 1.7


def test_lowe_karafiath_is_duncan_without_stage3():
    """Lowe & Karafiath (2-stage) uses the SAME Kc-interpolated stage-2 undrained
    strength as Duncan-Wright-Wong but omits stage 3, so its FOS is >= the 3-stage
    (stage 3 only substitutes a LOWER strength), it makes no drained substitutions,
    and its per-slice Kc diagnostics match the 3-stage run (same stage 1 + 2)."""
    lk = _rd(_dam(), "lowe_karafiath")
    dww = _rd(_dam(), "duncan_3stage")
    assert lk.method == "lowe_karafiath"
    assert lk.FOS >= dww.FOS - 1e-6
    assert lk.n_drained_substituted == 0
    # identical stage-1/stage-2 => identical consolidation stresses and Kc
    assert lk.sigma_fc == pytest.approx(dww.sigma_fc, abs=1e-9)
    assert lk.Kc == pytest.approx(dww.Kc, abs=1e-9)
    # and it is never more conservative than the Corps combined-envelope 2-stage
    assert lk.FOS >= _rd(_dam(), "corps_2stage").FOS - 1e-6


def test_undrained_slices_and_diagnostics():
    r = _rd(_dam(), "duncan_3stage")
    assert r.n_undrained_slices == r.n_slices   # homogeneous low-perm dam
    assert len(r.sigma_fc) == 50 and len(r.Kc) == 50
    Kf = (1 + math.sin(math.radians(30))) / (1 - math.sin(math.radians(30)))
    assert all(1.0 <= k <= Kf + 1e-6 for k in r.Kc)


def test_free_draining_layer_stays_drained():
    """A layer with R_phi=None is free-draining: no undrained slices."""
    r = _rd(_dam(R=False), "duncan_3stage")
    assert r.n_undrained_slices == 0


def test_stage1_seepage_phreatic_raises_fos():
    """The optional steady-seepage stage-1 phreatic (declining through the dam)
    lowers stage-1 pore pressure -> higher consolidation stress -> higher
    mobilized undrained strength -> higher drawdown FOS than the conservative
    flat full-pool default. A flat phreatic at the pool level reproduces the
    default (the option's identity case)."""
    g = _dam()
    flat = _rd(g, "corps_2stage").FOS
    seepage = rapid_drawdown_fos(
        g, 110 * FT, 24 * FT, xc=169.5 * FT, yc=210 * FT, radius=210 * FT,
        method="corps_2stage", n_slices=50,
        stage1_phreatic_points=[(0.0, 110 * FT), (380 * FT, 80 * FT)]).FOS
    assert seepage > flat
    identity = rapid_drawdown_fos(
        g, 110 * FT, 24 * FT, xc=169.5 * FT, yc=210 * FT, radius=210 * FT,
        method="corps_2stage", n_slices=50,
        stage1_phreatic_points=[(-1.0, 110 * FT), (381 * FT, 110 * FT)]).FOS
    assert identity == pytest.approx(flat, abs=0.01)


def test_validation_errors():
    g = _dam()
    with pytest.raises(ValueError):
        rapid_drawdown_fos(g, 24 * FT, 110 * FT, xc=169.5 * FT, yc=210 * FT,
                           radius=210 * FT)          # to >= from
    with pytest.raises(ValueError):
        rapid_drawdown_fos(g, 110 * FT, 24 * FT, method="bogus",
                           xc=169.5 * FT, yc=210 * FT, radius=210 * FT)
    with pytest.raises(ValueError):
        rapid_drawdown_fos(g, 110 * FT, 24 * FT)     # no slip surface


# ---------------------------------------------------------------------------
# Kf per EM 1110-2-1902 App. G (wiki-verification 2026-07-18): Eq. G-7 closed
# form for c'=0; printed stress-dependent Eq. G-8 for c'>0 (p. G-7).
# ---------------------------------------------------------------------------

def test_Kf_G7_closed_form_when_c_zero():
    from slope_stability.rapid_drawdown import _Kf
    phi = 30.0
    K7 = (1 + math.sin(math.radians(phi))) / (1 - math.sin(math.radians(phi)))
    assert _Kf(phi) == pytest.approx(K7)
    assert _Kf(phi, 0.0, 100.0) == pytest.approx(K7)


def test_Kf_G8_exceeds_G7_and_converges():
    """G-8 with c'>0 exceeds G-7 at finite stress (conservative: smaller w)
    and converges to G-7 as sigma'_fc >> c'."""
    from slope_stability.rapid_drawdown import _Kf
    phi, c = 30.0, 10.0
    K7 = _Kf(phi)
    k_low, k_mid, k_high = _Kf(phi, c, 30.0), _Kf(phi, c, 300.0), _Kf(phi, c, 3e5)
    assert k_low > k_mid > K7          # monotone toward G-7 from above
    assert k_high == pytest.approx(K7, rel=1e-3)
    # Printed G-8 hand check: phi=30, c'=10, sigma=100:
    # (100+8.6603)(1.5) / ((100-8.6603)(0.5)) = 162.99/45.67 = 3.569
    assert _Kf(30.0, 10.0, 100.0) == pytest.approx(3.5690, rel=1e-3)


def test_Kf_G8_capped_below_c_cos_phi():
    from slope_stability.rapid_drawdown import _Kf
    assert _Kf(30.0, 10.0, 5.0) == pytest.approx(1e6)   # sigma < c'*cos(phi)


def test_G8_conservative_direction_on_c_prime_soil():
    """For c' > 0 the G-8 Kf exceeds G-7, so the interpolation weight
    w = (Kc-1)/(Kf-1) shrinks and the stage-2 strength s_R + w*(s_D - s_R)
    moves TOWARD the R envelope (conservative when s_D > s_R). The c'=0 case
    is byte-identical to pre-G-8 behaviour by construction (G-8 -> G-7), which
    the published validations V-037/041/048/052 pin."""
    from slope_stability.rapid_drawdown import _Kf
    phi, c, sfc, Kc = 30.0, 10.0, 100.0, 2.0
    s_R, s_D = 40.0, 90.0                      # s_D > s_R (usual case)
    w7 = (Kc - 1.0) / (_Kf(phi) - 1.0)
    w8 = (Kc - 1.0) / (_Kf(phi, c, sfc) - 1.0)
    assert w8 < w7
    s7 = s_R + w7 * (s_D - s_R)
    s8 = s_R + w8 * (s_D - s_R)
    assert s_R < s8 < s7 < s_D


def test_a_search_with_no_surface_reports_no_fos():
    """GM-104: the solver's 999.9 marker was reported as the FOS when the
    search found no admissible surface."""
    from slope_stability.rapid_drawdown import RapidDrawdownSearchResult
    r = RapidDrawdownSearchResult(FOS=999.9, method="duncan", search=None)
    assert r.no_surface_found
    d = r.to_dict()
    assert d["FOS"] is None and d["no_surface_found"] is True
    assert "999.9" not in r.summary() and "no admissible surface" in r.summary()


# ---------------------------------------------------------------------------
# The stage-3 Fellenius normal under the remaining pool (GM-57 review)
# ---------------------------------------------------------------------------

def _pilarcitos_true():
    """Pilarcitos' true upstream section (V-052b): 2:1 for the lower 58 ft,
    3:1 to the crest; c' = 0, phi' = 45, R-envelope 60 psf / 23 deg."""
    face = [(0, 0), (116, 58), (176, 78), (260, 78)]
    return SlopeGeometry(
        surface_points=[(x * FT, z * FT) for x, z in face],
        soil_layers=[SlopeSoilLayer(
            name="fill", top_elevation=78 * FT, bottom_elevation=-8 * FT,
            gamma=135 * PCF, phi=45.0, c_prime=0.0,
            R_c=60 * PSF, R_phi=23.0)])


def test_pond_changes_no_fellenius_effective_normal():
    """Water ponded above a slice raises its total normal stress and its pore
    pressure equally, so the stage-3 Fellenius estimate of the effective
    normal is the same with 5 m of pond as with the water at the ground."""
    from slope_stability.rapid_drawdown import (
        _fellenius_effective_normal, _pool_geometry)
    from slope_stability.slices import build_slices
    from slope_stability.slip_surface import CircularSlipSurface
    g = SlopeGeometry(
        surface_points=[(0.0, 0.0), (30.0, 10.0), (50.0, 10.0)],
        soil_layers=[SlopeSoilLayer(name="s", top_elevation=10.0,
                                    bottom_elevation=-10.0, gamma=20.0,
                                    phi=30.0, c_prime=0.0)])
    slip = CircularSlipSurface(10.0, 25.0, 25.0)
    wet = build_slices(_pool_geometry(g, 5.0), slip, 30)
    assert sum(s.pond_force > 0.0 for s in wet) >= 5
    for s in wet:
        if s.pond_force <= 0.0:
            continue
        dry_top = dict(vars(s))
        q = s.pond_force / s.width
        dry_top.update(weight=s.weight - s.pond_force, pond_force=0.0,
                       pore_pressure=s.pore_pressure - q)
        bare = type(s)(**dry_top)
        assert _fellenius_effective_normal(s) == pytest.approx(
            _fellenius_effective_normal(bare), abs=1e-9)
        assert _fellenius_effective_normal(s) > 0.0


def test_dww_toe_circle_under_the_remaining_pool():
    """GM-57 review: the fully automatic DWW search of the true Pilarcitos
    section reported 0.052 on a toe circle wholly under the remaining pool.
    The Fellenius stage-3 normal counted the pond's pressure on the base in
    full but its weight only at cos^2(alpha); sigma' clamped to zero on all
    20 slices and the drained substitution left no strength. With the pond
    taken out of both terms the circle gives 1.57 (2.048 with the rigorous
    'gle' normal); the Corps and Lowe-Karafiath values, which have no stage
    3, do not change."""
    g = _pilarcitos_true()
    circle = dict(xc=-5.613427617517621, yc=31.343655235035243,
                  radius=31.84231877926432)
    fel = rapid_drawdown_fos(g, 72 * FT, 37 * FT, method="duncan_3stage",
                             n_slices=20, **circle)
    gle = rapid_drawdown_fos(g, 72 * FT, 37 * FT, method="duncan_3stage",
                             n_slices=20, stage3_effective_normal="gle",
                             **circle)
    assert fel.FOS == pytest.approx(1.57, abs=0.01)       # was 0.052
    assert gle.FOS == pytest.approx(2.048, abs=0.01)
    assert fel.FOS <= gle.FOS                             # Fellenius stays the
    lk = rapid_drawdown_fos(g, 72 * FT, 37 * FT,          # conservative basis
                            method="lowe_karafiath", n_slices=20, **circle)
    assert fel.FOS <= lk.FOS + 1e-9

