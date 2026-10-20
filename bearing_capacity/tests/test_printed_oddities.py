"""BACKLOG GM-214: two printed oddities in the bearing capacity factors.

1. UFC 3-220-20 Table 5-8 (p. 296) prints d_c = 1 + k (1 - sin phi')^2
   N_q/N_c, half of what its note C correspondence formula gives from its
   own d_q. The module keeps the note C form (its printed s_c and i_c follow
   that formula exactly, and at phi -> 0 it meets Table 5-7's 1 + 0.4 k)
   and lists the printed value as an alternative.
2. Meyerhof's i_gamma = (1 - theta/phi')^2 (Table 5-6, p. 294) was clamped
   at 0, which does nothing: past phi it grows again, to 1 at 2 phi. The
   table takes theta < phi', so theta >= phi' is refused. The other factors
   that turn back up past their range (a load angle below 0 or from 90,
   Vesic's g_q past 45 deg, b_q past alpha tan phi = 1) are refused too.
"""

import math

import pytest

from bearing_capacity.calc_steps import (
    _depth_factor_reference, _inclination_factor_equation,
)
from bearing_capacity.capacity import BearingCapacityAnalysis
from bearing_capacity.factors import (
    base_inclination_factors, bearing_capacity_Nc, bearing_capacity_Nq,
    depth_factors, depth_parameter_k, ground_inclination_factors,
    inclination_factors, ufc_table_5_8_printed_dc,
)
from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer


def _analysis(phi=32.0, c=0.0, depth=1.0, **kw):
    footing = Footing(width=2.0, depth=depth, shape="square")
    soil = BearingSoilProfile(layer1=SoilLayer(
        cohesion=c, friction_angle=phi, unit_weight=19.0))
    return BearingCapacityAnalysis(footing=footing, soil=soil, **kw)


# ── 1. d_c: Table 5-8 as printed against its note C ──────────────────────

@pytest.mark.parametrize("phi", [20.0, 32.0, 40.0])
@pytest.mark.parametrize("df_b", [0.5, 2.0])
def test_table_5_8_printed_dc_is_half_the_note_c_formula(phi, df_b):
    B = 2.0
    dc, dq, _ = depth_factors(phi, df_b * B, B, "vesic")
    printed = ufc_table_5_8_printed_dc(phi, df_b * B, B)
    assert dc - 1.0 == pytest.approx(2.0 * (printed - 1.0), rel=1e-12)
    # the note C formula from Table 5-8's d_q, written out
    p, k = math.radians(phi), depth_parameter_k(df_b * B, B)
    Nq, Nc = bearing_capacity_Nq(phi), bearing_capacity_Nc(phi)
    assert dc == pytest.approx(
        1.0 + 2.0 * k * (1.0 - math.sin(p)) ** 2 * Nq / Nc, rel=1e-12)


@pytest.mark.parametrize("phi", [20.0, 32.0, 40.0])
def test_table_5_8_printed_sc_and_ic_follow_note_c(phi):
    """The same table's s_c and i_c are the correspondence formula
    x_c = x_q - (1 - x_q)/(N_c tan phi) applied to its s_q and i_q: only
    its d_c is not."""
    p = math.radians(phi)
    Nq, Nc = bearing_capacity_Nq(phi), bearing_capacity_Nc(phi)

    def corr(xq):
        return xq - (1.0 - xq) / (Nc * math.tan(p))

    b_l = 0.6
    sq = 1.0 + math.sin(p) * b_l                      # Table 5-8 s_q
    assert 1.0 + b_l * (Nq / Nc) * math.cos(p) == pytest.approx(
        corr(sq), rel=1e-12)                          # printed s_c
    iq = 0.7
    assert (iq * Nq - 1.0) / (Nq - 1.0) == pytest.approx(
        corr(iq), rel=1e-12)                          # printed i_c


def test_at_phi_near_zero_note_c_meets_table_5_7():
    """Table 5-7 (phi = 0): d_c = 1 + 0.4 k. Note C's d_c tends to
    1 + 2 k/5.14 = 1 + 0.39 k; the printed Table 5-8 form to 1 + 0.19 k."""
    B, Df = 2.0, 1.0
    k = depth_parameter_k(Df, B)
    dc = depth_factors(1e-4, Df, B, "vesic")[0]
    assert (dc - 1.0) / k == pytest.approx(2.0 / 5.14, abs=1e-3)
    assert (ufc_table_5_8_printed_dc(1e-4, Df, B) - 1.0) / k == \
        pytest.approx(1.0 / 5.14, abs=1e-3)
    assert depth_factors(0.0, Df, B, "vesic")[0] == pytest.approx(
        1.0 + 0.4 * k)


def test_the_default_dc_is_unchanged():
    # master 86e631b: phi 32, c 10, 2 m square
    r = _analysis(c=10.0).compute()
    assert r.dc == pytest.approx(1.1443074476293416, rel=1e-12)
    assert r.q_ultimate == pytest.approx(1830.1, abs=0.05)
    r = _analysis(c=10.0, depth=4.0).compute()
    assert r.dc == pytest.approx(1.3195396112219269, rel=1e-12)
    assert r.q_ultimate == pytest.approx(4855.78, abs=0.005)


def test_the_depth_choice_states_both_forms():
    r = _analysis(c=10.0).compute()
    d = [c for c in r.choices if c["what"] == "depth factors"][0]
    assert "correspondence formula of its note C" in d["basis"]
    assert "half of that" in d["basis"]
    assert "Table 5-7's 1 + 0.4 k" in d["basis"]
    printed = ufc_table_5_8_printed_dc(32.0, 1.0, 2.0)
    alt = [a for a in d["alternatives"] if "Table 5-8 (p. 296)" in a]
    assert alt == [
        f"d_c = {printed:.3f}, as UFC 3-220-20 Table 5-8 (p. 296) prints "
        "it, 1 + k (1 - sin phi)^2 N_q/N_c: half of what the "
        "correspondence formula its note C names gives (not an option of "
        "this tool)"]
    # not for Meyerhof, phi = 0 or the other depth methods
    for kw in ({"factor_method": "meyerhof"}, {"phi": 0.0, "c": 50.0},
               {"depth_factor_method": "gec6"}):
        r = _analysis(**{"c": 10.0, **kw}).compute()
        d = [c for c in r.choices if c["what"] == "depth factors"][0]
        assert not any("Table 5-8 (p. 296)" in a
                       for a in d.get("alternatives", []))


def test_the_calc_steps_name_note_c():
    assert "note C" in _depth_factor_reference(_analysis(c=10.0))
    assert "note C" not in _depth_factor_reference(
        _analysis(phi=0.0, c=50.0))


# ── 2. Meyerhof's i_gamma and theta < phi' ───────────────────────────────

def test_meyerhof_below_phi_is_as_before():
    r = _analysis(load_inclination=30.0, factor_method="meyerhof").compute()
    assert r.igamma == pytest.approx((1.0 - 30.0 / 32.0) ** 2, rel=1e-12)
    assert r.q_ultimate == pytest.approx(215.2, abs=0.05)


@pytest.mark.parametrize("beta", [32.0, 40.0, 64.0])
def test_meyerhof_refuses_theta_at_or_beyond_phi(beta):
    """Before GM-214, phi 32 on a 2 m square: q_ult 199.4 kPa at 32 deg
    (i_gamma 0), 176.7 at 40, then 496.2 at 64 (i_gamma back to 1)."""
    with pytest.raises(ValueError, match=r"theta = atan\(H/V\) < phi'"):
        _analysis(load_inclination=beta, factor_method="meyerhof").compute()
    with pytest.raises(ValueError, match="Table 5-6"):
        inclination_factors(32.0, beta, 0.0, 2.0, 2.0, None, "meyerhof")


def test_meyerhof_phi_zero_takes_no_i_gamma():
    """phi = 0: N_gamma = 0, i_gamma = 1, and theta < phi' has no phi'."""
    assert inclination_factors(0.0, 10.0, 50.0, 2.0, 2.0, None,
                               "meyerhof")[2] == 1.0


def test_two_layers_refuse_on_the_lower_layer_too():
    footing = Footing(width=2.0, depth=1.0, shape="square")
    soil = BearingSoilProfile(
        layer1=SoilLayer(friction_angle=36.0, unit_weight=19.0,
                         thickness=1.0),
        layer2=SoilLayer(friction_angle=25.0, unit_weight=18.0))
    with pytest.raises(ValueError, match="phi = 25 deg"):
        BearingCapacityAnalysis(footing=footing, soil=soil,
                                load_inclination=30.0,
                                factor_method="meyerhof").compute()


def test_vesic_lists_meyerhof_as_none_beyond_phi():
    """Vesic's factors with c = 0 stand to 45 deg; the Meyerhof
    alternative says it has none there."""
    r = _analysis(load_inclination=40.0).compute()
    i = [c for c in r.choices if c["what"] == "load inclination factors"][0]
    assert i["alternatives"][0] == (
        "Meyerhof's angle-only factors, UFC 3-220-20 Table 5-6 (p. 294): "
        "none at 40 deg: the table takes theta = atan(H/V) < phi' = 32 deg")
    r = _analysis(load_inclination=10.0).compute()
    i = [c for c in r.choices if c["what"] == "load inclination factors"][0]
    assert "i_gamma = 0.473" in i["alternatives"][0]


def test_the_calc_steps_show_the_limit():
    assert "β < φ" in _inclination_factor_equation(
        _analysis(load_inclination=10.0, factor_method="meyerhof"), 32.0)


# ── the other factors that turn back up past their range ────────────────

@pytest.mark.parametrize("method", ["vesic", "meyerhof"])
@pytest.mark.parametrize("beta", [-10.0, 90.0, 120.0])
def test_load_inclination_outside_0_to_90_is_refused(method, beta):
    """Before: Vesic at -10 deg gave i_q 1.28 and q_ult 1,555.9 kPa against
    1,158.8 for a vertical load (phi 32, c 0, 2 m square)."""
    with pytest.raises(ValueError, match="0 to below 90"):
        inclination_factors(0.0 if beta >= 90 else 32.0, beta, 50.0, 2.0,
                            2.0, 500.0, method)


@pytest.mark.parametrize("omega", [-10.0, 50.0, 60.0])
def test_ground_slope_outside_0_to_45_is_refused(omega):
    """Before: (1 - tan omega)^2 is 0 at 45 deg, then 0.536 at 60 (q_ult
    621.0 kPa against 0.0 at 45); -10 deg gave 1.38 (1,603.4 kPa)."""
    with pytest.raises(ValueError, match="0 to 45 deg"):
        ground_inclination_factors(32.0, omega)
    with pytest.raises(ValueError, match="ground_slope"):
        _analysis(ground_slope=omega).compute()


def test_ground_slope_to_45_is_as_before():
    assert ground_inclination_factors(32.0, 30.0)[1] == pytest.approx(
        (1.0 - math.tan(math.radians(30.0))) ** 2)
    assert ground_inclination_factors(32.0, 45.0)[1] == pytest.approx(
        0.0, abs=1e-12)


def test_base_tilt_past_alpha_tan_phi_one_is_refused():
    # phi 40: alpha tan phi = 1 at 68.3 deg
    assert base_inclination_factors(40.0, 60.0)[1] == pytest.approx(
        (1.0 - math.radians(60.0) * math.tan(math.radians(40.0))) ** 2)
    with pytest.raises(ValueError, match="alpha tan\\(phi\\) above 1"):
        base_inclination_factors(40.0, 70.0)
    # GEC-6 Table 5-5: upward +; a negative tilt stays as printed
    assert base_inclination_factors(30.0, -5.0)[1] > 1.0
