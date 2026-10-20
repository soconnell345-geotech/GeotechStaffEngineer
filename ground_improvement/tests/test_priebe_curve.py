"""GEC-13's printed Priebe curve, Fig. 5-21 curve (4) (GM-221).

GEC-13 (FHWA-NHI-16-027) Vol. I prints no Priebe formula. It prints the
Priebe curve, Fig. 5-21 curve (4) "PRIEBE phi = 37.5°" (p. 5-36), reads it
in Example 2 (Sec. 4.3.2, Fig. 5-27, p. 5-49) and says on p. 5-37 that
"the Priebe curve may be used to evaluate the upper bound effectiveness".
The curve was traced off the rendered page on its grid lines and drawn
back over the figure; a read is good to about one line width, +/- 0.03.

Example 2: columns 3.0 ft at 5.7 ft, A/A_sc = (5.7 / 3)^2 = 3.6; "Using
the Priebe curve from Figure 5-27" the settlement ratio is 2.7, so 27 in.
becomes 10 in. Fig. 5-27's arrows are drawn at A/A_sc 3.53 and n' 2.63,
labelled "3.6" and "2.7".
"""

import pytest

from ground_improvement.aggregate_piers import priebe_basic_improvement_factor
from ground_improvement.settlement_ratio import (
    PRIEBE_CURVE_4, priebe_curve_settlement_ratio)

TOL = 0.03  # one line width on the printed figure


def _curve(area_ratio):
    return priebe_curve_settlement_ratio(1.0 / area_ratio)


@pytest.mark.parametrize("area_ratio, n_read", [
    (1.6, 6.01), (2.0, 4.82), (2.5, 3.80), (3.0, 3.11), (3.5, 2.63),
    (4.0, 2.30), (5.0, 1.96), (6.0, 1.79), (8.0, 1.49)])
def test_the_curve_as_read_off_the_figure(area_ratio, n_read):
    assert _curve(area_ratio) == pytest.approx(n_read, abs=TOL)


def test_example_2_reads_the_curve_where_its_arrows_are_drawn():
    # the arrows of Fig. 5-27 as drawn: A/A_sc 3.53, n' 2.63
    assert _curve(3.53) == pytest.approx(2.63, abs=TOL)
    # at the example's A/A_sc = (5.7 / 3)^2 the curve gives 2.55, where
    # the example's text rounds its read to 2.7: S = 27 / 2.55 = 10.6 in.
    a = (5.7 / 3.0) ** 2
    assert _curve(a) == pytest.approx(2.55, abs=TOL)
    assert 27.0 / _curve(a) == pytest.approx(10.6, abs=0.15)


def test_the_formula_against_the_printed_curve():
    """Priebe's basic n0 at phi_c = 37.5 deg, nu = 1/3: within 1.5 % of the
    printed curve at A/A_c 2.25 to 3.5, 2 to 7 % above it from 3.75 to 8,
    and 6 to 31 % above it below 2.25."""
    for area_ratio, n_read in PRIEBE_CURVE_4.items():
        f = priebe_basic_improvement_factor(1.0 / area_ratio, 37.5)
        off = f / n_read - 1.0
        if 2.25 <= area_ratio <= 3.5:
            assert abs(off) < 0.015, (area_ratio, off)
        elif area_ratio >= 3.75:
            assert 0.015 < off < 0.07, (area_ratio, off)
        else:
            assert 0.06 < off < 0.32, (area_ratio, off)


def test_between_the_reads_linear_in_a_s():
    a0, a1 = 1 / 3.0, 1 / 3.25
    mid = (a0 + a1) / 2
    assert priebe_curve_settlement_ratio(mid) == pytest.approx(
        (PRIEBE_CURVE_4[3.0] + PRIEBE_CURVE_4[3.25]) / 2)


@pytest.mark.parametrize("as_ratio", [0.1, 0.7])
def test_off_the_drawn_curve_is_refused(as_ratio):
    with pytest.raises(ValueError, match="method 'formula'"):
        priebe_curve_settlement_ratio(as_ratio)


def test_a_bad_area_ratio_is_refused():
    with pytest.raises(ValueError, match="replacement ratio"):
        priebe_curve_settlement_ratio(1.0)
