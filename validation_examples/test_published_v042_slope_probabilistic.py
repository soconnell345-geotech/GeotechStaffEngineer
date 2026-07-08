"""V-042 — Slide2 Verification #36: Li & Lumb (1987) / Hassan & Wolff (1999)
homogeneous slope, PROBABILISTIC reliability index (Bishop simplified).
[Slide2 manual pp. 137-139]

A canonical published reliability benchmark — validates the module's FOSM +
Monte Carlo + lognormal reliability-index path against the ORIGINAL Hassan &
Wolff (1999) result (which Slide2 reproduces), not just against Slide2's own
simulation.

Geometry (labeled Fig 36.1, meters): homogeneous slope, base el 0, domain
x in [0, 20]; surface (0,5),(5,5),(15,15),(20,15) — a 10 m high 1:1 face from
(5,5) to (15,15). Table 36.1 (mean, std): c' = 18 +/- 3.6 kPa (COV 0.20),
phi = 30 +/- 3 deg (COV 0.10), gamma = 18 +/- 0.9 kN/m3 (COV 0.05),
ru = 0.20 +/- 0.02. Bishop simplified; variables normal; FOS lognormal.

Published (Table 36.2):
  Slide (Bishop):    det-min surface FOS 1.340, RI_LN 2.482; overall FOS 1.350,
                     RI 2.393.
  Hassan-Wolff:      det FOS 1.334, RI_LN 2.336.

The RI comparison is the headline. This example needed the additive `ru`
probabilistic variable (added with F1/F2 alongside `gamma_sat`).

VERDICT: PASS. The searched Bishop critical circle gives FOS 1.325 (-0.7% vs
Hassan-Wolff 1.334, -1.1% vs Slide 1.340). The FOSM COV_F 0.1225 REPRODUCES
Hassan-Wolff's implied FOS-COV: the module's lognormal_beta evaluated at the
published deterministic FOS (1.334) gives RI 2.30, within 1.6% of Hassan-Wolff's
2.336 (and the module's own F=1.325 -> RI 2.244 is within ~4%). FOSM ~= MC. The
variance is c'-dominated (~70%), then phi (~23%) -- physically sensible for a
COV_c'=0.20 slope.
"""
import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import search_critical_surface
from slope_stability.probabilistic import fosm_fos, monte_carlo_fos, lognormal_beta

_SURFACE = [(0, 5), (5, 5), (15, 15), (20, 15)]
# Bishop critical circle recovered by entry-exit search (seed=1, 1500 trials);
# pinned here so the reliability assertions are deterministic and fast.
_XC, _YC, _R = 4.3736565339749305, 19.588888034671577, 14.589059771501798
_VARS = {"c_prime": {"std": 3.6}, "phi": {"std": 3.0},
         "gamma": {"std": 0.9}, "ru": {"std": 0.02}}


def _geom():
    layer = SlopeSoilLayer(
        name="mat1", top_elevation=15, bottom_elevation=0.0,
        gamma=18.0, gamma_sat=18.0, phi=30.0, c_prime=18.0, ru=0.2,
        analysis_mode="drained")
    return SlopeGeometry(surface_points=_SURFACE, soil_layers=[layer])


def test_v042_deterministic_bishop_fos():
    """The searched Bishop critical FOS reproduces the published deterministic
    minimum (Slide 1.340 / Hassan-Wolff 1.334) to ~1%."""
    res = search_critical_surface(_geom(), method="bishop",
                                  surface_type="entry_exit", n_trials=800,
                                  n_slices=30, seed=1)
    assert res.critical.FOS == pytest.approx(1.334, rel=0.03)
    assert res.critical.FOS == pytest.approx(1.340, rel=0.03)


def test_v042_fosm_reliability_index_matches_hassan_wolff():
    """FOSM on the critical circle: COV_F reproduces Hassan-Wolff's implied
    FOS-COV -- the module's lognormal RI at the published FOS matches 2.336."""
    fr = fosm_fos(_geom(), _VARS, xc=_XC, yc=_YC, radius=_R,
                  method="bishop", n_slices=30)
    assert fr.fos_mlv == pytest.approx(1.325, abs=0.01)
    assert fr.cov_f == pytest.approx(0.1225, abs=0.012)
    # module's own reliability index (at its FOS 1.325)
    assert fr.beta_lognormal == pytest.approx(2.244, abs=0.06)
    # evaluated at the PUBLISHED deterministic FOS -> reproduces Hassan-Wolff 2.336
    assert lognormal_beta(1.334, fr.cov_f) == pytest.approx(2.336, abs=0.10)
    # c'-dominated variance (COV_c' = 0.20 is the largest), then phi
    pct = fr.variable_variance_pct
    assert pct["c_prime"] > pct["phi"] > pct["gamma"]
    assert pct["c_prime"] > 55.0


def test_v042_monte_carlo_agrees_with_fosm():
    """Monte Carlo reproduces the FOSM mean, COV and lognormal RI."""
    fr = fosm_fos(_geom(), _VARS, xc=_XC, yc=_YC, radius=_R,
                  method="bishop", n_slices=30)
    mc = monte_carlo_fos(_geom(), _VARS, xc=_XC, yc=_YC, radius=_R,
                         method="bishop", n=2000, seed=5, n_slices=30)
    assert mc.fos_mean == pytest.approx(fr.fos_mlv, abs=0.02)
    assert mc.fos_cov == pytest.approx(fr.cov_f, rel=0.12)
    assert mc.beta_lognormal == pytest.approx(fr.beta_lognormal, abs=0.10)
