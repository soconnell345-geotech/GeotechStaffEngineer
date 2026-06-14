"""Phase E validation — GEC-10 Appendix A drilled-shaft design example (V-006..V-008).

Source: GEC-10 (FHWA-NHI-18-024) Appendix A design example, Steps 11.5-11.6.
See validation_examples/INVENTORY.md entries V-006/007/008 and RESULTS.md.

Key finding: the `drilled_shaft` module implements the SIMPLIFIED
O'Neill-Reese (1999) / AASHTO methods (depth-based beta, alpha=0.55), not the
GEC-10 (2018) RATIONAL chains (OCR-based beta, su-transform alpha) used in the
Appendix A example. So the side-resistance entries are documented method-scope
gaps; the base-resistance unit formula DOES match exactly.

Units: example is US customary; modules are SI. 1 ft=0.3048 m, 1 kip=4.448 kN,
1 ksf=47.88 kPa, 1 tsf=95.76 kPa, 1 psf=0.04788 kPa.
"""

import math

import pytest

from drilled_shaft.side_resistance import alpha_cohesive, beta_cohesionless
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


# --- V-007 alpha (clay): module is AASHTO 0.55, example is rational ----------

def test_v007_alpha_method_scope_gap():
    """Documents the coverage gap: module returns the AASHTO simplified alpha
    (0.55 for cu/pa<=1.5), NOT the GEC-10 rational su-transform alpha (0.47).
    Both are defensible; the module simply implements the simplified method."""
    cu = 1750 * PSF
    pa = 2114 * PSF
    alpha_module = alpha_cohesive(cu, pa)
    assert alpha_module == pytest.approx(0.55, abs=1e-6)      # AASHTO simplified
    example_rational_alpha = 0.47
    assert abs(alpha_module - example_rational_alpha) > 0.05  # genuinely different methods


# --- V-006 beta (sand): module is depth-based, example is rational OCR -------

def test_v006_beta_method_scope_gap():
    """Documents the coverage gap: module returns the O'Neill-Reese depth-based
    beta (floored at 0.25 for deep layers), NOT the GEC-10 rational OCR-based
    beta (0.41). Recorded as a coverage gap, module not tuned to the example."""
    # Layer 3 sand sits deep (>40 ft); the depth formula is at its 0.25 floor.
    for z_ft in (40, 60, 80, 100):
        assert beta_cohesionless(z_ft * FT) == pytest.approx(0.25, abs=1e-6)
    example_rational_beta = 0.41
    assert abs(0.25 - example_rational_beta) > 0.05
