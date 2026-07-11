"""V-034 — Slide2 Verification #34: Wolff & Harr (1987) Clarence Cannon Dam,
correlated c'-phi' probabilistic analysis. [manual pp. 132-133]

Validates the CORRELATED SCALAR-PAIR reliability capability (train item 1 —
generalizes the F1 su-law (a,b) correlation to arbitrary scalar pairs) against
Slide2 #34's published correlation coefficients (Table 34.1):
  Phase I fill : c'=2230+/-1150 lb/ft^2, phi'=6.34+/-7.87 deg, rho(c',phi')=+0.11
  Phase II fill: c'=2901.6+/-1079.8 lb/ft^2, phi'=14.8+/-9.44 deg, rho= -0.51
  Sand drain   : c'=0, phi'=30 (not varied)

VERDICT: PASS (correlated-pair capability) / N/A-scope (full-slope FOS + Pf).

The correlation FEATURE is validated with #34's own published rho values on a
representative 2-material embankment: fosm_fos / monte_carlo_fos accept the
`correlations` list, add the FOSM cross-term (surfacing a `corr(...)` variance
entry), and the correlation demonstrably shifts COV_F away from the independent
value. FOSM and MC land in the same band; the residual (~8%) is the FOSM
linearization limit at #34's EXTREME phi' COV (~1.2 for Phase I) — a documented
Taylor-series limitation at very high COV, not a feature defect (the machinery
matches MC to <1% at moderate COV; see slope_stability/tests/test_probabilistic).

N/A-scope: the FULL #34 result (deterministic FOS 2.333 GLE / 2.383 Spencer /
2.36 Wolff-Harr; Pf 3.55e-3 Slide) is NOT reproduced — Slide2 Fig 34.2 is a
rendered multi-material section with NO labeled coordinates, a figure-only
non-circular critical surface, omitted "non-labeled soil layers", and (the
manual's own note) unit weights Slide "had to use to match the factor of
safety". The geometry does not survive honest extraction, so this test
validates the CAPABILITY against the published correlation coefficients, not the
slope result. Related: #33 (El-Ramly dyke) is N/A-scope (figure-only, and c'=0
so no c'-phi' correlation); #35 (Cannon Dam reliability indices) is
SKIPPED(source) per the manual's admission that missing inputs were selected to
match the FOS. See RESULTS/INVENTORY.
"""
import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.probabilistic import fosm_fos, monte_carlo_fos

_PSF, _PCF = 0.04788, 0.157087
# Slide2 #34 Table 34.1 -> SI (kPa, deg, kN/m3)
_P1 = dict(c=2230*_PSF, sc=1150*_PSF, phi=6.34, sp=7.87, g=150*_PCF)   # Phase I
_P2 = dict(c=2901.6*_PSF, sc=1079.8*_PSF, phi=14.8, sp=9.44, g=150*_PCF)  # Phase II
_SURFACE = [(0, 10), (10, 10), (30, 20), (50, 20)]  # representative 2:1, 10 m
# Bishop critical circle from an entry-exit search (seed=2), pinned.
_XC, _YC, _R = 19.21666763854572, 25.616662455190777, 24.76207708108703
_VARS = {
    "c_prime:PhaseI": {"std": _P1["sc"]}, "phi:PhaseI": {"std": _P1["sp"]},
    "c_prime:PhaseII": {"std": _P2["sc"]}, "phi:PhaseII": {"std": _P2["sp"]},
}
_CORR = [("c_prime:PhaseI", "phi:PhaseI", 0.11),
         ("c_prime:PhaseII", "phi:PhaseII", -0.51)]


def _geom():
    return SlopeGeometry(surface_points=_SURFACE, soil_layers=[
        SlopeSoilLayer(name="PhaseII", top_elevation=20, bottom_elevation=10,
                       gamma=_P2["g"], gamma_sat=_P2["g"], c_prime=_P2["c"],
                       phi=_P2["phi"], analysis_mode="drained"),
        SlopeSoilLayer(name="PhaseI", top_elevation=10, bottom_elevation=0.0,
                       gamma=_P1["g"], gamma_sat=_P1["g"], c_prime=_P1["c"],
                       phi=_P1["phi"], analysis_mode="drained"),
    ])


def test_v034_feature_consumes_published_correlations():
    """FOSM accepts #34's published rho values (Table 34.1), surfaces the
    cross-term as corr(...) entries, and the correlation shifts COV_F off the
    independent value."""
    indep = fosm_fos(_geom(), _VARS, xc=_XC, yc=_YC, radius=_R,
                     method="bishop", n_slices=30)
    corr = fosm_fos(_geom(), _VARS, xc=_XC, yc=_YC, radius=_R,
                    method="bishop", n_slices=30, correlations=_CORR)
    assert corr.fos_mlv == pytest.approx(3.445, abs=0.02)
    assert indep.cov_f == pytest.approx(0.396, abs=0.02)
    assert corr.cov_f == pytest.approx(0.413, abs=0.02)
    assert corr.cov_f != pytest.approx(indep.cov_f, abs=1e-4)   # correlation matters
    pct = corr.variable_variance_pct
    assert "corr(c_prime:PhaseI,phi:PhaseI)" in pct
    assert "corr(c_prime:PhaseII,phi:PhaseII)" in pct


def test_v034_fosm_and_mc_same_band_at_high_cov():
    """FOSM and Monte Carlo agree within the high-COV band on #34's stats
    (extreme phi' COV ~1.2 -> ~8% FOSM/MC gap from Taylor linearization, not a
    feature defect)."""
    rf = fosm_fos(_geom(), _VARS, xc=_XC, yc=_YC, radius=_R,
                  method="bishop", n_slices=30, correlations=_CORR)
    mc = monte_carlo_fos(_geom(), _VARS, xc=_XC, yc=_YC, radius=_R,
                         method="bishop", n=4000, seed=11, n_slices=30,
                         correlations=_CORR)
    assert mc.fos_cov == pytest.approx(rf.cov_f, rel=0.15)
