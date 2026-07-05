"""Phase E validation — GEC-10 Appendix A drilled-shaft design example (V-006..V-008).

Source: GEC-10 (FHWA-NHI-18-024) Appendix A design example, Steps 11.5-11.6.
See validation_examples/INVENTORY.md entries V-006/007/008 and RESULTS.md.

v5.3 coverage Batch 2: the `drilled_shaft` module now offers the GEC-10 RATIONAL
side-resistance chains as opt-in, default-preserving methods —
`beta_method="rational"` (cohesionless OCR/Ko chain) and `alpha_method="rational"`
(cohesive Chen-2011 alpha with the UU/UC->CIUC transform) — so V-006 and V-007 run
through the HIGH-LEVEL `DrillShaftAnalysis` path with the built-in chains (no
hand-fed coefficients). The base-resistance unit formula (V-008) already matches.

Units: example is US customary; modules are SI. 1 ft=0.3048 m, 1 kip=4.448 kN,
1 ksf=47.88 kPa, 1 tsf=95.76 kPa, 1 psf=0.04788 kPa.
"""

import math

import pytest

from drilled_shaft import (
    DrillShaft, ShaftSoilLayer, ShaftSoilProfile, DrillShaftAnalysis,
)
from drilled_shaft.side_resistance import (
    alpha_cohesive, beta_cohesionless,
    phi_prime_from_N1_60, preconsolidation_stress, k0_from_ocr,
    beta_cohesionless_rational, su_to_ciuc, alpha_cohesive_rational, PA,
)
from drilled_shaft.end_bearing import end_bearing_cohesionless

FT = 0.3048
KIP = 4.448
KSF = 47.88
PSF = 0.04788


# --- V-008 base resistance in sand: 0.60*N60 (tsf) == 57.5*N60 (kPa) ----------

def test_v008_base_unit_resistance_matches():
    """Unit base resistance qb and area math match the example to <0.1%
    (with the large-diameter reduction removed, per V-008b)."""
    N60 = 41
    B_m = 8 * FT
    area = math.pi / 4 * B_m**2
    qb_kPa = 57.5 * min(N60, 50)            # module unit formula
    assert qb_kPa / KSF == pytest.approx(49.2, rel=0.01)   # example 49.2 ksf
    RBN_unreduced = qb_kPa * area
    assert RBN_unreduced / KIP == pytest.approx(2473, rel=0.01)  # example 2473 kips


def test_v008b_large_diameter_reduction_is_applied():
    """The module applies the O'Neill-Reese 1.27/Db reduction for Db>1.27 m
    (GEC-10 13.3.4.3). The example value is unreduced — documented CONVENTION
    difference, not a bug. This test pins the module's (correct) behavior."""
    N60 = 41
    B_m = 8 * FT                            # 2.4384 m > 1.27 m
    area = math.pi / 4 * B_m**2
    RBN_module = end_bearing_cohesionless(N60, area, diameter=B_m)
    expected_factor = 1.27 / B_m            # 0.521
    RBN_unreduced = 57.5 * N60 * area
    assert RBN_module == pytest.approx(RBN_unreduced * expected_factor, rel=1e-6)
    # i.e. the module returns ~1289 kips vs the example's unreduced 2473 kips
    assert RBN_module / KIP == pytest.approx(1289, rel=0.02)


# --- V-006 beta (sand): GEC-10 rational OCR/Ko chain reproduces beta=0.41 -----

def test_v006_rational_beta_chain_intermediates():
    """The GEC-10 Appendix A rational beta chain, step by step (Layer 3 sand).

    phi' from (N1)60=21; sigma'p from N60=30; OCR uses the NO-SCOUR sigma'v
    (4,645 psf); Ko = (1-sin phi')*OCR^(sin phi'); beta = Ko*tan(delta=phi')."""
    phi = phi_prime_from_N1_60(21)
    assert phi == pytest.approx(40.0, abs=0.5)                 # 39.66 deg (pub 40)

    sigma_p = preconsolidation_stress(30, PA)                 # 0.47*pa*N60^0.6
    assert sigma_p / PSF == pytest.approx(7654, rel=0.01)      # pub 7,654 psf

    sigma_v_noscour = 4645 * PSF
    OCR = sigma_p / sigma_v_noscour
    assert OCR == pytest.approx(1.65, abs=0.02)               # pub 1.65

    Ko = k0_from_ocr(phi, OCR)
    assert Ko == pytest.approx(0.49, abs=0.02)               # pub 0.49

    beta = beta_cohesionless_rational(phi, OCR)               # delta = phi'
    assert beta == pytest.approx(0.41, rel=0.05)              # pub 0.41


def test_v006_high_level_rational_beta():
    """High-level DrillShaftAnalysis(beta_method='rational') reproduces the
    published Layer 3 beta and nominal side resistance RSN through the built-in
    chain — no hand-fed coefficients.

    The profile models the SCOUR-adjusted design stress (sigma'v ~ 2,266 psf at
    the layer mid-depth, for fs) with the layer's `sigma_v_ref` carrying the
    NO-SCOUR stress (4,645 psf, for OCR). A 3 m cased cover supplies the
    overburden without contributing side resistance."""
    B = 8 * FT
    dz = 20 * FT
    shaft = DrillShaft(diameter=B, length=3.0 + dz, casing_depth=3.0)
    soil = ShaftSoilProfile(layers=[
        # Cover: permanently cased -> excluded from side resistance, but its
        # weight raises sigma'v at the Layer-3 mid-depth to the scour value.
        ShaftSoilLayer(3.0, "cohesionless", 18.0, phi=30),
        ShaftSoilLayer(dz, "cohesionless", 17.9, phi=40, N60=30, N1_60=21,
                       sigma_v_ref=4645 * PSF, description="Layer 3 sand"),
    ])
    result = DrillShaftAnalysis(shaft=shaft, soil=soil,
                                beta_method="rational").compute()

    sand = next(lb for lb in result.layer_breakdown
                if lb["soil_type"] == "cohesionless" and "rational" in lb["method"])
    beta = float(sand["method"].split("=")[1].split()[0])
    assert beta == pytest.approx(0.41, rel=0.05)                  # pub 0.41
    # scour-adjusted design stress at mid-depth ~ 2,266 psf
    assert sand["sigma_v_kPa"] / PSF == pytest.approx(2266, rel=0.03)
    assert sand["fs_kPa"] / PSF == pytest.approx(936, rel=0.05)   # pub fSN 936 psf
    assert sand["side_resistance_kN"] / KIP == pytest.approx(470.7, rel=0.07)  # RSN


def test_v006_default_beta_still_depth_based():
    """Default beta_method='depth' is byte-identical to the pre-v5.3 behavior:
    the deep Layer-3 sand floors at the O'Neill-Reese beta=0.25."""
    for z_ft in (40, 60, 80, 100):
        assert beta_cohesionless(z_ft * FT) == pytest.approx(0.25, abs=1e-6)


# --- V-007 alpha (clay): GEC-10 rational su-transform chain reproduces 0.47 ---

def test_v007_rational_alpha_chain_intermediates():
    """The GEC-10 Appendix A rational alpha chain, step by step (Layer 4 clay).

    su(UU)=1,750 psf transformed to CIUC via Eq 10-16 (the UC pair 0.893/0.513,
    as the worked example applies); alpha = 0.30 + 0.17/(su(CIUC)/pa)."""
    su_uu = 1750 * PSF
    sigma_v0 = 2114 * PSF
    su_ciuc = su_to_ciuc(su_uu, sigma_v0, "uc")
    assert su_ciuc / PSF == pytest.approx(2057, rel=0.01)         # pub 2,057 psf

    alpha = alpha_cohesive_rational(su_ciuc, PA)
    assert alpha == pytest.approx(0.47, rel=0.05)                 # pub 0.47

    fs = alpha * su_ciuc                                          # fs = alpha*su_CIUC
    assert fs / PSF == pytest.approx(976, rel=0.05)               # pub fSN 976 psf


def test_v007_high_level_rational_alpha():
    """High-level DrillShaftAnalysis(alpha_method='rational', su_test_type='uc')
    reproduces the published Layer 4 alpha and RSN through the built-in chain.

    The clay sits between a cased cover (overburden -> sigma'vo ~ 2,114 psf) and a
    bearing layer below, so the full 15 ft is active (the bottom-1D cohesive
    exclusion falls in the bearing layer, not the clay)."""
    B = 8 * FT
    dz = 15 * FT
    shaft = DrillShaft(diameter=B, length=3.0 + dz + 3.0, casing_depth=3.0)
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(3.0, "cohesive", 18.0, cu=100.0),          # cased cover
        ShaftSoilLayer(dz, "cohesive", 20.3, cu=1750 * PSF, description="Layer 4 clay"),
        ShaftSoilLayer(3.0, "cohesionless", 19.0, phi=35),        # bearing layer / tip
    ])
    result = DrillShaftAnalysis(shaft=shaft, soil=soil,
                                alpha_method="rational", su_test_type="uc").compute()

    clay = next(lb for lb in result.layer_breakdown
                if lb.get("description") == "Layer 4 clay")
    alpha = float(clay["method"].split("=")[1].split()[0])
    assert alpha == pytest.approx(0.47, rel=0.05)                 # pub 0.47
    # full 15 ft active (no bottom-1D exclusion eating into the clay)
    active = clay["effective_bottom_m"] - clay["effective_top_m"]
    assert active == pytest.approx(dz, rel=0.02)
    assert clay["fs_kPa"] / PSF == pytest.approx(976, rel=0.05)   # pub fSN 976 psf
    assert clay["side_resistance_kN"] / KIP == pytest.approx(368.1, rel=0.07)  # RSN


def test_v007_default_alpha_still_aashto():
    """Default alpha_method='aashto' is byte-identical to the pre-v5.3 behavior:
    the AASHTO simplified alpha=0.55 for cu/pa<=1.5."""
    cu = 1750 * PSF
    assert alpha_cohesive(cu, PA) == pytest.approx(0.55, abs=1e-6)
