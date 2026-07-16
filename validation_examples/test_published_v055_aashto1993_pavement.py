"""V-055: AASHTO 1993 Guide printed worked examples, end-to-end through
pavement_design.

Published targets (AASHTO Guide for Design of Pavement Structures, 1993):

1. Figure 3.1 (printed II-32) flexible design chart worked example:
   W18 = 5x10^6, R = 95% (ZR = -1.645), So = 0.35, MR = 5000 psi,
   dPSI = 1.9 -> printed solution SN = 5.0 (nomograph, 0.1 resolution).
2. Figure 3.7 (printed II-45/46) rigid design chart worked example:
   W18 = 5.1x10^6, k = 72 pci, Ec = 5x10^6 psi, S'c = 650 psi, J = 3.2,
   Cd = 1.0, So = 0.29, R = 95%, dPSI = 4.5-2.5 = 1.7 -> printed solution
   D = 10.0 in (nomograph, nearest half-inch).
3. Figure 2.4 (printed II-15) effective roadbed MR worked example:
   12 monthly MR values -> sum(uf) = 3.72, effective MR = 5,000 psi.

These run through the FULL orchestrator (reliability table -> solve ->
layer split / rounding -> forward check), not the bare equations, so they
validate the module's assembly of the guide's procedure.
"""

import pytest

from pavement_design import (PavementLayer, design_flexible_pavement,
                             design_rigid_pavement)


def test_v055a_flexible_figure_3_1_example():
    res = design_flexible_pavement(
        w18=5e6, reliability_pct=95, so=0.35, mr_psi=5000, delta_psi=1.9,
        layers=[
            PavementLayer("asphalt", modulus_psi=400000),
            PavementLayer("granular_base", modulus_psi=30000),
            PavementLayer("granular_subbase", modulus_psi=11000),
        ])
    # Printed SN = 5.0 (nomograph read-off; equation solve 4.95-5.0).
    assert res.sn_required == pytest.approx(5.0, abs=0.06)
    # ZR from Table 4.1 at R=95%.
    assert res.zr == pytest.approx(-1.645, abs=1e-3)
    # The designed section must satisfy its own forward check.
    assert res.sn_provided >= res.sn_required
    assert res.w18_capacity >= res.w18
    assert res.adequate


def test_v055b_rigid_figure_3_7_example():
    res = design_rigid_pavement(
        w18=5.1e6, sc_psi=650, ec_psi=5e6, reliability_pct=95, so=0.29,
        delta_psi=1.7, j=3.2, cd=1.0, k_pci=72, pt=2.5)
    # Printed D = 10.0 in (nearest half-inch nomograph; solve ~9.7-10.0).
    assert 9.6 <= res.d_required_in <= 10.05
    assert res.d_provided_in == pytest.approx(10.0, abs=0.51)
    assert res.adequate


def test_v055c_effective_mr_figure_2_4_example():
    res = design_flexible_pavement(
        w18=5e6, reliability_pct=95, so=0.35, delta_psi=1.9,
        monthly_mr_psi=[20000, 20000, 2500, 4000, 4000, 7000, 7000, 7000,
                        7000, 7000, 4000, 20000],
        layers=[PavementLayer("asphalt", a=0.44)])
    # Printed effective MR = 5,000 psi.
    assert res.effective_mr_psi == pytest.approx(5000, rel=0.05)
    # And the flexible solve at that MR reproduces the Figure 3.1 SN.
    assert res.sn_required == pytest.approx(5.0, abs=0.1)
