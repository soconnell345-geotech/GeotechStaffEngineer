"""The alpha method's charts as UFC 3-220-20 and FHWA GEC-12 print them
(GM-255): UFC Table 6-16, Fig 6-14 and Eq 6-9; GEC-12 Fig. 7-17."""

import numpy as np
import pytest

from geotech_common import alpha_method as am

PSF = am.PSF
PA = am.PA_KPA


def test_table_6_16_consistencies_at_the_printed_bounds():
    assert am.ufc_consistency(799 * PSF) == "soft"
    assert am.ufc_consistency(800 * PSF) == "firm"
    assert am.ufc_consistency(1500 * PSF) == "stiff"
    assert am.ufc_consistency(3000 * PSF) == "very stiff"
    assert am.ufc_consistency(6000 * PSF) == "very stiff"
    assert am.ufc_consistency(6001 * PSF) == "harder than very stiff"


def test_soft_clay_takes_alpha_1_whatever_z():
    for zb in (5.0, 10.0, 60.0):
        assert am.alpha_ufc(799 * PSF, zb) == (1.0, am.UFC_TABLE_SOFT)


def test_ufc_drag_example_reads_0_5_off_curve_a():
    """App. B-5.4 (pp. 636-637): the stiff clay, su 3000 psf, Z/b 65,
    alpha 0.5 off Fig 6-14."""
    alpha, source = am.alpha_ufc(3000 * PSF, 65.0)
    assert source == am.UFC_FIG
    assert alpha == pytest.approx(0.50, abs=0.005)


def test_eq_6_9_is_as_printed_and_gives_the_examples_720_psf():
    """Eq 6-9 (p. 416): alpha = 0.28 (su/Pa)^-0.44; App. B-5.4's
    transition zone, su 3000 psf, takes 720 psf."""
    su = 3000 * PSF
    assert am.alpha_ufc_eq_6_9(su) == pytest.approx(
        0.28 * (su / PA) ** -0.44)
    assert am.alpha_ufc_eq_6_9(su) * 3000 == pytest.approx(720, abs=1)
    assert am.alpha_ufc_eq_6_9(1.0) == 1.0          # at most 1.0


def test_curves_a_and_b_and_between():
    su = 1.0 * PA
    a, _ = am.alpha_ufc(su, 40.0)
    b, _ = am.alpha_ufc(su, 10.0)
    mid, _ = am.alpha_ufc(su, 25.0)
    assert a == pytest.approx(0.854, abs=0.005)
    assert b == pytest.approx(0.635, abs=0.005)
    assert mid == pytest.approx(0.5 * (a + b))
    assert am.alpha_ufc(su, 80.0)[0] == a           # A from 40b on


def test_the_curves_fall_from_1_and_end_flat():
    for x, y in (am.UFC_CURVE_A, am.UFC_CURVE_B):
        assert np.all(np.diff(x) > 0)
        assert max(y) <= 1.0 and min(y) >= 0.23
        assert x[-1] == am.UFC_CURVE_END
    # curve A runs along alpha 1.0 to su/Pa 0.72
    assert am.alpha_ufc(0.70 * PA, 50.0)[0] == 1.0


def test_off_the_chart_is_held_and_said():
    notes = []
    alpha, source = am.alpha_ufc(3.0 * PA, 50.0, notes)
    assert source == am.UFC_FIG_END
    assert alpha == pytest.approx(am.UFC_CURVE_A[1][-1])
    assert "past the end" in notes[0]
    notes = []
    alpha, _ = am.alpha_ufc(1.0 * PA, 5.0, notes)
    assert alpha == am.alpha_ufc(1.0 * PA, 10.0)[0]
    assert "less than curve B's 10" in notes[0]


def test_fig_7_17_moved_here_and_axial_pile_reads_the_same():
    from axial_pile import tomlinson
    assert tomlinson.adhesion_fig_7_17 is am.adhesion_fig_7_17
    assert tomlinson.CHART_END_KSF is am.CHART_END_KSF
    # Ca = su at the chart's start on the solid D = 10b curve (0.26/0.25)
    ca = am.adhesion_fig_7_17(0.25 * am.KSF, 10.0, am.SOLID)
    assert ca / am.KSF == pytest.approx(0.26)
    with pytest.raises(ValueError, match="curve must be one of"):
        am.adhesion_fig_7_17(50.0, 20.0, "steel")
