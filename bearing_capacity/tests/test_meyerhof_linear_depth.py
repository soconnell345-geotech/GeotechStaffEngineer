"""BACKLOG GM-309: Meyerhof's own depth factors, linear in k = Df/B at every
depth, as UFC 3-220-20 Table 5-6 (p. 294) prints them, offered as
``depth_factor_method="linear"`` (Meyerhof only). The default ("hansen":
k = arctan(Df/B) beyond Df/B = 1) is unchanged.

Table 5-6, Depth row: phi = 0: d_c = 1 + 0.2 k (K_P)^0.5, d_q = d_gamma = 1;
phi' > 10 deg: d_q = d_gamma = 1 + 0.1 k (K_P)^0.5; K_P = tan^2(pi/4 +
phi'/2), k = Df/B (the table's notes, no limit on Df/B). Values below are
made up and worked by hand."""

import math

import pytest

from bearing_capacity.calc_steps import (_depth_factor_equation,
                                         _depth_factor_reference)
from bearing_capacity.capacity import BearingCapacityAnalysis
from bearing_capacity.factors import (DEPTH_FACTOR_BASIS,
                                      DEPTH_FACTOR_METHODS, depth_factors)
from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer


def _analysis(width=1.6, depth=4.0, layer=None, **kw):
    return BearingCapacityAnalysis(
        footing=Footing(width=width, length=width, depth=depth,
                        shape="square"),
        soil=BearingSoilProfile(layer1=SoilLayer(**(layer or dict(
            cohesion=0.0, friction_angle=34.0, unit_weight=18.5)))),
        factor_method="meyerhof", **kw)


def _choice(result, what):
    found = [c for c in result.choices if c["what"] == what]
    assert len(found) == 1
    return found[0]


def test_linear_is_an_offered_value_with_its_printed_basis():
    assert "linear" in DEPTH_FACTOR_METHODS
    assert "Table 5-6" in DEPTH_FACTOR_BASIS["linear"]
    assert "p. 294" in DEPTH_FACTOR_BASIS["linear"]


def test_linear_reproduces_table_5_6_by_hand_beyond_df_b_one():
    # phi 34 deg, B 1.6 m, Df 4.0 m: Df/B = 2.5
    # K_P = tan^2(45 + 17) = tan^2(62 deg) = 3.5371, sqrt = 1.8807
    # d_c = 1 + 0.2 x 2.5 x 1.8807 = 1.9404
    # d_q = d_gamma = 1 + 0.1 x 2.5 x 1.8807 = 1.4702
    dc, dq, dg = depth_factors(34.0, 4.0, 1.6, "meyerhof", "linear")
    assert dc == pytest.approx(1.9404, abs=5e-4)
    assert dq == pytest.approx(1.4702, abs=5e-4)
    assert dg == dq
    # 'hansen' takes k = arctan(2.5) = 1.1903 rad: smaller factors
    hc, hq, hg = depth_factors(34.0, 4.0, 1.6, "meyerhof", "hansen")
    # GM-310: left out, Meyerhof's factors are his own (linear)
    assert depth_factors(34.0, 4.0, 1.6, "meyerhof") == (dc, dq, dg)
    assert hq == pytest.approx(1 + 0.1 * 1.1903 * 1.8807, abs=5e-4)
    assert hq < dq and hc < dc


def test_linear_reproduces_table_5_6_by_hand_at_phi_zero():
    # phi = 0: K_P = 1; Df/B = 3.0 / 1.2 = 2.5: d_c = 1 + 0.2 x 2.5 = 1.5
    assert depth_factors(0.0, 3.0, 1.2, "meyerhof", "linear") == \
        pytest.approx((1.5, 1.0, 1.0))
    # 'hansen': 1 + 0.2 arctan(2.5) = 1.2381
    assert depth_factors(0.0, 3.0, 1.2, "meyerhof", "hansen")[0] == \
        pytest.approx(1.2381, abs=5e-4)


@pytest.mark.parametrize("phi", [0.0, 28.0, 34.0])
@pytest.mark.parametrize("df_b", [0.0, 0.4, 0.75, 1.0])
def test_linear_and_the_default_agree_up_to_df_b_one(phi, df_b):
    # below Df/B = 1 both take k = Df/B (Table 5-6)
    assert depth_factors(phi, df_b * 2.0, 2.0, "meyerhof", "linear") == \
        pytest.approx(depth_factors(phi, df_b * 2.0, 2.0, "meyerhof",
                                    "hansen"))


def test_linear_below_df_b_one_by_hand():
    # phi 28 deg, Df/B = 0.9 / 1.5 = 0.6: K_P = tan^2(59 deg) = 2.7698,
    # sqrt = 1.6643; d_c = 1 + 0.2 x 0.6 x 1.6643 = 1.1997,
    # d_q = 1 + 0.1 x 0.6 x 1.6643 = 1.0999
    dc, dq, dg = depth_factors(28.0, 0.9, 1.5, "meyerhof", "linear")
    assert (dc, dq, dg) == pytest.approx((1.1997, 1.0999, 1.0999), abs=5e-4)


def test_linear_goes_with_meyerhof_only():
    with pytest.raises(ValueError, match="meyerhof"):
        depth_factors(30.0, 4.0, 1.6, "vesic", "linear")
    with pytest.raises(ValueError, match="factor_method='meyerhof'"):
        BearingCapacityAnalysis(
            footing=Footing(width=1.6, depth=4.0, shape="square"),
            soil=BearingSoilProfile(layer1=SoilLayer(
                friction_angle=30.0, unit_weight=18.0)),
            depth_factor_method="linear")


def test_analysis_carries_the_linear_factors_and_a_larger_capacity():
    lin = _analysis(depth_factor_method="linear").compute()
    dflt = _analysis(depth_factor_method="hansen").compute()
    assert lin.dq == pytest.approx(1.4702, abs=5e-4)
    assert lin.q_ultimate > dflt.q_ultimate
    # only the depth factors differ: the overburden and self-weight terms
    # scale by d_q / d_gamma
    assert lin.term_overburden / dflt.term_overburden == \
        pytest.approx(lin.dq / dflt.dq, rel=1e-9)


def test_the_choice_reports_linear_and_hansen_as_alternative():
    r = _analysis(depth_factor_method="linear").compute()
    c = _choice(r, "depth factors")
    # outside a tool call "given" is judged by the value: linear is now
    # Meyerhof's default (GM-310)
    assert c["how"] == "default"
    assert c["chosen"].startswith("k = Df/B = 2.500")
    assert "(depth_factor_method 'linear')" in c["chosen"]
    assert "Table 5-6" in c["basis"] and "every depth" in c["basis"]
    assert any("depth_factor_method 'hansen'" in a for a in c["alternatives"])


def test_hansen_is_the_option_and_lists_linear_as_alternative():
    r = _analysis(depth_factor_method="hansen").compute()
    c = _choice(r, "depth factors")
    assert c["how"] == "given"
    assert "arctan(Df/B)" in c["chosen"]
    assert "(depth_factor_method 'hansen')" in c["chosen"]
    assert "Brinch Hansen's method only" in c["basis"]
    alt = [a for a in c["alternatives"] if "'linear'" in a]
    assert len(alt) == 1 and "1.940, 1.470, 1.470" in alt[0]


# GM-310 (the call for OWN-131, 2026-11-12): with Meyerhof's method the
# depth factors follow his own Table 5-6 at every depth by default.

def test_meyerhof_default_is_his_own_table_and_warns_beyond_df_b_one():
    import warnings as w
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        r = _analysis().compute()
    lin = _analysis(depth_factor_method="linear").compute()
    han = _analysis(depth_factor_method="hansen").compute()
    assert r.q_ultimate == lin.q_ultimate > han.q_ultimate
    assert r.dq == pytest.approx(1.4702, abs=5e-4)
    c = _choice(r, "depth factors")
    assert c["how"] == "default"
    assert "(depth_factor_method 'linear')" in c["chosen"]
    # the warning, in the choice and as a Python warning: no limit, and
    # Brinch Hansen's factors and q_ult beside it
    for text in (c["chosen"], " ".join(str(x.message) for x in caught)):
        assert "Df/B = 2.50 is beyond 1" in text
        assert "grow without limit" in text
        assert f"q_ult = {han.q_ultimate:,.0f} kPa" in text
        assert f"{han.dc:.3f}, {han.dq:.3f}, {han.dgamma:.3f}" in text
        assert "Table 5-8 note B, p. 295" in text


def test_no_warning_up_to_df_b_one_or_with_vesic():
    import warnings as w
    shallow = BearingCapacityAnalysis(
        footing=Footing(width=1.6, depth=1.6, shape="square"),
        soil=BearingSoilProfile(layer1=SoilLayer(
            friction_angle=34.0, unit_weight=18.5)),
        factor_method="meyerhof")
    assert shallow.depth_factor_method == "linear"
    vesic = BearingCapacityAnalysis(
        footing=Footing(width=1.6, depth=4.0, shape="square"),
        soil=BearingSoilProfile(layer1=SoilLayer(
            friction_angle=34.0, unit_weight=18.5)))
    assert vesic.depth_factor_method == "hansen"
    with w.catch_warnings():
        w.simplefilter("error")
        r = shallow.compute()
        vesic.compute()
    assert "beyond 1" not in _choice(r, "depth factors")["chosen"]
    # up to Df/B = 1 the two forms are the same number
    assert r.q_ultimate == pytest.approx(BearingCapacityAnalysis(
        footing=Footing(width=1.6, depth=1.6, shape="square"),
        soil=BearingSoilProfile(layer1=SoilLayer(
            friction_angle=34.0, unit_weight=18.5)),
        factor_method="meyerhof",
        depth_factor_method="hansen").compute().q_ultimate)


def test_vesic_does_not_list_linear():
    r = BearingCapacityAnalysis(
        footing=Footing(width=1.6, depth=4.0, shape="square"),
        soil=BearingSoilProfile(layer1=SoilLayer(
            friction_angle=34.0, unit_weight=18.5))).compute()
    c = _choice(r, "depth factors")
    assert not [a for a in c["alternatives"] if "'linear'" in a]


def test_calc_steps_show_the_linear_form():
    a = _analysis(depth_factor_method="linear")
    eq = _depth_factor_equation("meyerhof", 34.0, "linear")
    assert "k = D_f/B at every depth" in eq and "arctan" not in eq
    assert "d_q = d_γ = 1.0" in _depth_factor_equation(
        "meyerhof", 0.0, "linear")
    assert "Table 5-6" in _depth_factor_reference(a)
    assert "Table 5-8" not in _depth_factor_reference(a)


def test_adapter_takes_linear():
    from funhouse_agent.adapters.bearing_capacity import (
        METHOD_INFO, run_bearing_capacity_analysis)
    p = METHOD_INFO["bearing_capacity_analysis"]["parameters"][
        "depth_factor_method"]
    assert "linear" in p["allowed_values"] and "'linear'" in p["description"]
    result, _ = run_bearing_capacity_analysis(dict(
        width=1.6, length=1.6, depth=4.0, shape="square",
        friction_angle=34.0, unit_weight=18.5, factor_method="meyerhof",
        depth_factor_method="linear"))
    assert result.dq == pytest.approx(1.4702, abs=5e-4)
    assert math.isfinite(result.q_ultimate)
