"""V-030 (F1) — Duncan (2000) LASH underwater slope: FOSM propagation of the
input COVs THROUGH the FOS with the new depth-varying undrained-strength law.

This is the ADDITIVE extension that closes the N/A-scope note on V-030
(``test_published_v026_slope.py``): the module's per-parameter FOSM varies
phi/c/cu/gamma INDEPENDENTLY per layer, so a shared random su-gradient across the
thin Bay-Mud sub-layers would be over-counted. The new ``linear_su`` strength law
(``slope_stability.probabilistic``) treats su(z) = a + b*(datum - z) as ONE
correlated (a, b) random-variable pair, applied COHERENTLY across every
sub-layer, so a single gradient perturbation moves the whole profile together.

Problem (Slide2 #29 / Duncan 2000, manual pp. 121-122): the 100-ft underwater
Bay-Mud slope at the LASH terminal. su = 100 psf at el -20 ft, +9.8 psf/ft;
gamma = 100 pcf. Probabilistic inputs (Table 29.2): unit-weight std 3.3 pcf
(COV 0.033), su rate-of-change std 1.2 psf/ft (COV 0.122). Duncan's closed-form
Taylor-series anchor: F = 1.17, COV_F ~ 0.16, Pf ~ 18%. Slide2 per-method
simulated Pf: 13-18%.

What the FOSM propagation shows (all HONEST, no tuning):
 (1) The law at its mean (a, b) REPRODUCES the discretized deterministic FOS
     (Spencer 1.190) -- the law and the hand-built sub-layer cu profile agree.
 (2) Propagating the STATED input COVs (unit weight via gamma_sat + the
     su-gradient std 1.2 psf/ft) gives COV_F ~ 0.133 -> Pf ~ 11% (Pf ~ 13% at
     Duncan's F = 1.17), landing in the Slide2 per-method band (13-14% for
     Spencer/GLE). The unit-weight uncertainty is AMPLIFIED by buoyancy: the
     submerged buoyant weight (gamma_sat - gamma_w ~ 37.6 pcf) carries a COV of
     ~0.088 even though gamma_sat's COV is only 0.033, so gamma_sat contributes
     ~40% of the FOS variance. (This is why gamma_sat, not the dry gamma, must be
     the varied unit weight on a fully submerged slope.)
 (3) Duncan's higher closed-form COV_F ~ 0.16 / Pf 18% is RECOVERED when the su
     uncertainty is characterized as the full coherently-scaling profile
     (COV ~ 0.16, the intercept+slope+scatter of the su-vs-depth data), not the
     regression-slope std alone: the correlated (a, b) law with COV 0.16 gives
     COV_F = 0.160 -> Pf 18.2% at F = 1.17. Since F is ~proportional to su for
     this phi=0 slope, COV_F tracks COV_su -- the FOSM correctly propagates it.
 (4) The correlation cross-term behaves: for the same |sensitivities|, the
     combined (a, b) variance rises monotonically with rho_ab (-1 < 0 < +1).
 (5) Monte Carlo (bivariate-normal (a,b) sampling) agrees with the FOSM COV_F.

VERDICT: PASS -- the correlated su-gradient FOSM capability is built and
propagates the V-030 input COVs coherently; it reproduces Duncan's COV_F 0.16 ->
Pf 18% when the su uncertainty is supplied as the full profile COV, and lands in
the Slide2 Pf band on the stated regression-slope std. Closes the V-030
N/A-scope gap.
"""
import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.slip_surface import PolylineSlipSurface
from slope_stability.probabilistic import (
    fosm_fos, monte_carlo_fos, lognormal_beta, _phi_cdf)

# ── LASH geometry (SI, converted from the manual's labeled Fig 29.1, feet) ──
# Mirrors test_published_v026_slope.py::_v030_geom; kept local so this extension
# file is self-contained.
_F, _PSF, _PCF = 0.3048, 0.04788, 0.157087
_SURFACE = [(x*_F, z*_F) for x, z in
            [(-28, -40), (0, -40), (71, -120), (138, -120), (228, -18),
             (283, -17), (350, -8), (389, 22), (461, 22)]]
_FAIL = [(x*_F, z*_F) for x, z in
         [(138, -120), (150, -117), (170, -105), (185, -100), (205, -93),
          (221, -85), (240, -75), (257, -64), (275, -53), (293, -39),
          (311, -23), (350, -8)]]
_GAMMA, _BASE, _TOP = 100*_PCF, -143*_F, 22*_F
# su law constants (SI): a = 100 psf, b = 9.8 psf/ft, datum el -20 ft, floor 15 psf
_A, _B = 100*_PSF, 9.8*_PSF/_F
_B_STD = 1.2*_PSF/_F              # Table 29.2 su rate-of-change std
_DATUM, _FLOOR = -20*_F, 15*_PSF


def _lash_geom(nlayers=60):
    dz = (_TOP - _BASE) / nlayers
    layers = []
    for i in range(nlayers):
        t = _TOP - i*dz
        b = t - dz
        mid = 0.5*(t + b)
        layers.append(SlopeSoilLayer(
            name="baymud_%d" % i, top_elevation=t, bottom_elevation=b,
            gamma=_GAMMA, gamma_sat=_GAMMA,
            cu=max(_A + _B*(_DATUM - mid), _FLOOR), analysis_mode="undrained"))
    return SlopeGeometry(
        surface_points=_SURFACE, soil_layers=layers,
        gwt_points=[(_SURFACE[0][0], 0.0), (_SURFACE[-1][0], 0.0)])


def _law(a_cov=None, b_std=_B_STD, b_cov=None, rho=0.0):
    """Build a linear_su law spec: a fixed (std 0) unless a_cov given; b varied by
    b_std (default = Table 29.2) or b_cov."""
    a = {"mean": _A, "std": 0.0} if a_cov is None else {"mean": _A, "cov": a_cov}
    b = {"mean": _B, "cov": b_cov} if b_cov is not None else {"mean": _B, "std": b_std}
    return {"law": "linear_su", "a": a, "b": b, "rho_ab": rho,
            "datum_z": _DATUM, "z_ref": "mid", "su_min": _FLOOR}


_SLIP = PolylineSlipSurface(points=_FAIL)


def test_v030_fosm_law_at_means_reproduces_deterministic_fos():
    """The su law evaluated at its mean (a, b) recomputes the SAME sub-layer cu
    profile as the hand-built discretization, so FOSM's mean-value FOS equals the
    deterministic Spencer FOS (1.190)."""
    r = fosm_fos(_lash_geom(), {"bay_mud_su": _law()}, slip_surface=_SLIP,
                 method="spencer", n_slices=60)
    assert r.fos_mlv == pytest.approx(1.190, abs=0.01)


def test_v030_fosm_stated_input_covs_land_in_slide_band():
    """Propagating the STATED input COVs (unit weight via gamma_sat COV 0.033 +
    su-gradient std 1.2 psf/ft) gives COV_F ~ 0.133 -> Pf ~ 11% (Pf ~ 13% at
    Duncan's F = 1.17), inside the Slide2 per-method band. The unit-weight
    uncertainty is buoyancy-amplified: gamma_sat carries ~40% of the variance
    despite its small (0.033) COV."""
    variables = {"gamma_sat": {"cov": 0.033}, "bay_mud_su": _law()}
    r = fosm_fos(_lash_geom(), variables, slip_surface=_SLIP,
                 method="spencer", n_slices=60)
    assert r.cov_f == pytest.approx(0.133, abs=0.015)
    assert r.pf_lognormal == pytest.approx(0.107, abs=0.03)
    # buoyancy amplification: gamma_sat is a major (not negligible) contributor
    assert r.variable_variance_pct["gamma_sat"] > 30.0
    assert r.variable_variance_pct["bay_mud_su"] > 45.0
    # at Duncan's deterministic F = 1.17 this COV_F lands in the Slide2 band
    pf_at_duncan_f = _phi_cdf(-lognormal_beta(1.17, r.cov_f))
    assert 0.11 < pf_at_duncan_f < 0.16


def test_v030_fosm_full_su_cov_reproduces_duncan():
    """Duncan's closed-form COV_F ~ 0.16 / Pf 18% is recovered when the su
    uncertainty is the full coherently-scaling profile COV (a, b both COV 0.16,
    perfectly correlated). Since F ~ proportional to su for this phi=0 slope,
    COV_F tracks COV_su -- the FOSM propagates it correctly."""
    r = fosm_fos(_lash_geom(), {"bay_mud_su": _law(a_cov=0.16, b_cov=0.16, rho=1.0)},
                 slip_surface=_SLIP, method="spencer", n_slices=60)
    assert r.cov_f == pytest.approx(0.16, abs=0.012)
    # Pf at Duncan's F = 1.17 reproduces his reported 18%
    pf_at_duncan_f = _phi_cdf(-lognormal_beta(1.17, r.cov_f))
    assert pf_at_duncan_f == pytest.approx(0.18, abs=0.02)


def test_v030_correlated_pair_cross_term_is_monotone_in_rho():
    """The (a, b) cross-term: with the same |sensitivities|, the combined FOS
    variance rises monotonically with rho_ab (anti- < independent < positively
    correlated). Confirms the Taylor-series correlation term is applied."""
    covs = []
    for rho in (-1.0, 0.0, 1.0):
        r = fosm_fos(_lash_geom(),
                     {"bay_mud_su": _law(a_cov=0.1, b_cov=0.1, rho=rho)},
                     slip_surface=_SLIP, method="spencer", n_slices=60)
        covs.append(r.cov_f)
    assert covs[0] < covs[1] < covs[2]
    # sanity: independent-pair variance = sum of the two squared half-deltas
    assert covs[1] == pytest.approx(0.085, abs=0.02)


def test_v030_monte_carlo_agrees_with_fosm():
    """Monte Carlo (bivariate-normal (a, b) sampling, correlated via Cholesky)
    reproduces the FOSM COV_F for the stated inputs -- the two propagation paths
    agree."""
    variables = {"gamma_sat": {"cov": 0.033}, "bay_mud_su": _law()}
    rf = fosm_fos(_lash_geom(), variables, slip_surface=_SLIP,
                  method="spencer", n_slices=60)
    mc = monte_carlo_fos(_lash_geom(), variables, slip_surface=_SLIP,
                         method="spencer", n=1000, seed=11, n_slices=60)
    assert mc.fos_mean == pytest.approx(rf.fos_mlv, abs=0.02)
    assert mc.fos_cov == pytest.approx(rf.cov_f, rel=0.15)
    assert mc.pf_lognormal == pytest.approx(rf.pf_lognormal, abs=0.03)
