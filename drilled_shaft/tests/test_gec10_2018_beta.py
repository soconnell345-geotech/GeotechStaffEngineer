"""GM-241: side resistance in cohesionless soil follows GEC-10 (2018) as
printed, by default.

GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.1, Step 5 of the design for axial
load, pp. 10-10 to 10-12: R_SN = pi B dz f_SN, f_SN = beta sigma'v on the
layer's average sigma'v (Eqs. 10-5, 10-7); beta by Eq. 10-13 with sigma'p
from N60 (Eq. 10-11 or 10-12), limited to its value at 7.5 ft; K/Ko = 1
(p. 10-11), 2/3 where construction quality is not controlled (p. 10-12).
Sand takes side resistance from the ground surface (Sec. 10.3.5.1 prints
no top zone; Examples 10-1 and 10-5), clay and rock leave out the top
1.5 m (p. 10-17), unless top_exclusion is given. The depth beta of
O'Neill and Reese (1999), the default until GM-241, stays by name.
Example 10-1 is V-058 in ``validation_examples``.
"""

import math

import pytest

from drilled_shaft import (DrillShaft, DrillShaftAnalysis, ShaftSoilLayer,
                           ShaftSoilProfile)
from drilled_shaft.capacity import SHALLOW_BETA_DEPTH
from drilled_shaft.side_resistance import (PA, beta_gec10,
                                           preconsolidation_stress_gec10)


def _ds2(**kw):
    """The eval question DS-2: a 1.2 m shaft 15 m long in sand, phi' 34,
    19 kN/m3, N60 15, no water table."""
    soil = ShaftSoilProfile(layers=[ShaftSoilLayer(
        30.0, "cohesionless", 19.0, phi=34.0, N60=15.0)])
    return DrillShaftAnalysis(shaft=DrillShaft(diameter=1.2, length=15.0),
                              soil=soil, **kw)


def _choice(result, what):
    (c,) = [c for c in result.choices if c["what"].startswith(what)]
    return c


def test_ds2_is_the_hand_derivation():
    """Each step of Eqs. 10-5 to 10-13 by hand, over 0-15 m: the sand
    from the ground surface, as Examples 10-1 and 10-5 take it."""
    sv = 19.0 * 15.0 / 2                               # 142.5 kPa
    sp = PA * 0.47 * 15.0 ** 0.6                       # 241.8 kPa
    s = math.sin(math.radians(34.0))
    Ko = (1 - s) * (sp / sv) ** s                      # 0.592
    beta = Ko * math.tan(math.radians(34.0))           # 0.400
    Qs = math.pi * 1.2 * 15.0 * beta * sv              # 3,220.3 kN
    r = _ds2().compute()
    assert Qs == pytest.approx(3220.3, abs=0.1)
    assert r.Q_side_sand == pytest.approx(Qs, rel=1e-9)
    (lb,) = r.layer_breakdown
    st = lb["beta_steps"]
    assert (st["sigma_p_kPa"], st["OCR"], st["Ko"], st["beta"]) == (
        pytest.approx(sp, abs=1e-3), pytest.approx(sp / sv, abs=1e-3),
        pytest.approx(Ko, abs=1e-3), pytest.approx(beta, abs=1e-3))
    assert r.Q_ultimate == pytest.approx(4195.7, abs=0.1)


def test_the_depth_beta_stays_by_name():
    """beta = 1.5 - 0.245 sqrt(z, m) = 0.829 at the centre of 0-15 m (the
    metric form of 1.5 - 0.135 sqrt(z, ft), NHI-06-089 Eq. 9-42), f_s =
    0.829 x 142.5 = 118.1 kPa (Eq. 9-43)."""
    r = _ds2(beta_method="depth").compute()
    beta = 1.5 - 0.245 * math.sqrt(7.5)
    Qs = math.pi * 1.2 * 15.0 * beta * 19.0 * 7.5
    assert r.Q_side_sand == pytest.approx(Qs, rel=1e-3)
    assert r.Q_ultimate == pytest.approx(7656.0, abs=0.1)
    assert "rational" not in r.layer_breakdown[0]["method"]


def test_the_method_choice_lists_its_alternatives_with_values():
    c = _choice(_ds2().compute(), "side resistance in cohesionless soil")
    assert c["how"] == "default"
    assert "Eq. 10-13" in c["chosen"] and "3,220 kN" in c["chosen"]
    assert "Sec. 10.3.5.1" in c["basis"] and "pp. 10-10 to 10-12" in c["basis"]
    alts = " | ".join(c["alternatives"])
    assert "sub-layers of 0.1 m" in alts and "Q_ult 3,953 kN" in alts
    assert "beta_method 'depth'" in alts and "Q_ult 7,656 kN" in alts
    c = _choice(_ds2(beta_method="depth").compute(),
                "side resistance in cohesionless soil")
    assert c["how"] == "given" and "O'Neill and Reese" in c["chosen"]
    assert "Q_ult 4,196 kN" in c["alternatives"][0]


def test_each_judgment_is_a_choice_with_its_alternatives():
    r = _ds2().compute()
    top = _choice(r, "side resistance near the ground surface")
    assert top["how"] == "default" and "(0-1.5 m)" in top["chosen"]
    assert "p. 10-17" in top["basis"] and "Sec. 10.3.5.2" in top["basis"]
    assert "Examples 10-1 and 10-5" in top["basis"]
    assert "Q_ult 3,998 kN" in top["alternatives"][0]
    k = _choice(r, "K/Ko")
    assert k["how"] == "default" and "2/3 of its" in k["basis"]
    assert "Q_ult 3,122 kN" in k["alternatives"][0]
    sp = _choice(r, "preconsolidation stress")
    assert "m = 0.6" in sp["chosen"] and sp["how"] == "default"
    assert any("'silty_sand'" in a and "Q_ult 5,335 kN" in a
               for a in sp["alternatives"])
    assert any("'gravel'" in a for a in sp["alternatives"])


def test_K_over_Ko_scales_beta_and_is_bounded():
    full = _ds2().compute()
    two_thirds = _ds2(K_over_Ko=2 / 3).compute()
    assert two_thirds.Q_side_sand == pytest.approx(
        full.Q_side_sand * 2 / 3, rel=1e-9)
    assert _choice(two_thirds, "K/Ko")["how"] == "given"
    for bad in (0.0, 1.2):
        with pytest.raises(ValueError, match="p. 10-12"):
            _ds2(K_over_Ko=bad)


def test_beta_near_the_surface_is_limited_to_its_value_at_7_5_ft():
    """A layer whose average sigma'v is shallower than 7.5 ft takes the
    OCR at 7.5 ft; f_SN stays beta times its own sigma'v."""
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(2.0, "cohesionless", 19.0, phi=34.0, N60=20.0),
        ShaftSoilLayer(20.0, "cohesionless", 19.0, phi=34.0, N60=20.0)])
    r = DrillShaftAnalysis(shaft=DrillShaft(diameter=1.0, length=10.0),
                           soil=soil).compute()
    first = r.layer_breakdown[0]
    st = first["beta_steps"]
    sv_75 = 19.0 * SHALLOW_BETA_DEPTH
    assert st["shallow_limit"] is True
    assert st["sigma_v_for_OCR_kPa"] == pytest.approx(sv_75, abs=1e-3)
    want = beta_gec10(34.0, preconsolidation_stress_gec10(20.0) / sv_75)
    assert st["beta"] == pytest.approx(want["beta"], abs=1e-3)
    assert first["sigma_v_kPa"] == pytest.approx(19.0 * 1.0, abs=0.05)
    assert first["fs_kPa"] == pytest.approx(want["beta"] * 19.0 * 1.0,
                                            abs=0.05)
    assert r.layer_breakdown[1]["beta_steps"]["shallow_limit"] is False
    assert "7.5 ft" in _choice(r, "beta near the ground surface")["chosen"]


def test_the_gravel_correlation_is_eq_10_12():
    assert preconsolidation_stress_gec10(20.0, "gravel") == pytest.approx(
        PA * 0.15 * 20.0)
    assert preconsolidation_stress_gec10(20.0, "silty_sand") == pytest.approx(
        PA * 0.47 * 20.0 ** 0.8)
    with pytest.raises(ValueError, match="sigma_p_correlation"):
        ShaftSoilLayer(5.0, "cohesionless", 19.0, phi=34,
                       sigma_p_correlation="clay")


def test_no_limit_on_f_sn_and_a_choice_says_so():
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(10.0, "cohesionless", 20.0, phi=40.0, N60=30.0),
        ShaftSoilLayer(15.0, "cohesionless", 20.0, phi=40.0, OCR=4.0,
                       N60=30.0)])
    r = DrillShaftAnalysis(shaft=DrillShaft(diameter=1.0, length=25.0),
                           soil=soil).compute()
    deep = r.layer_breakdown[1]
    assert deep["fs_kPa"] > 200.0
    assert deep["fs_kPa"] == pytest.approx(
        deep["beta_steps"]["beta"] * deep["sigma_v_kPa"], abs=0.2)
    c = _choice(r, "upper limit on f_SN")
    assert "191 kPa" in c["alternatives"][0]


def test_a_side_layer_with_no_n60_is_refused_with_the_way_out():
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(5.0, "cohesionless", 18.0, phi=32.0),
        ShaftSoilLayer(15.0, "cohesive", 18.0, cu=80.0)])
    a = DrillShaftAnalysis(shaft=DrillShaft(diameter=1.0, length=12.0),
                           soil=soil)
    with pytest.raises(ValueError, match="needs N60") as err:
        a.compute()
    assert "0-5 m" in str(err.value) and "'depth'" in str(err.value)
    # a casing over it leaves it out, and the depth beta takes it
    DrillShaftAnalysis(shaft=DrillShaft(diameter=1.0, length=12.0,
                                        casing_depth=5.0), soil=soil).compute()
    DrillShaftAnalysis(shaft=DrillShaft(diameter=1.0, length=12.0),
                       soil=soil, beta_method="depth").compute()


def test_the_average_stress_is_exact_across_the_water_table():
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(10.0, "cohesionless", 19.0, phi=34.0, N60=20.0)],
        gwt_depth=3.0)
    s15, s3 = 19.0 * 1.5, 19.0 * 3.0
    s10 = s3 + 7.0 * (19.0 - soil.gamma_w)
    want = (1.5 * (s15 + s3) / 2 + 7.0 * (s3 + s10) / 2) / 8.5
    assert soil.average_effective_stress(1.5, 10.0) == pytest.approx(want)
    assert soil.average_effective_stress(4.0, 6.0) == pytest.approx(
        soil.effective_stress_at_depth(5.0))


def test_the_top_zone_is_by_soil_type_unless_given():
    """Left out: none in sand (Sec. 10.3.5.1; Examples 10-1 and 10-5),
    the top 1.5 m in clay (p. 10-17). Given: every layer."""
    soil = ShaftSoilProfile(layers=[
        ShaftSoilLayer(1.0, "cohesive", 18.0, cu=60.0),
        ShaftSoilLayer(29.0, "cohesionless", 19.0, phi=34.0, N60=15.0)])
    shaft = DrillShaft(diameter=1.2, length=15.0)
    r = DrillShaftAnalysis(shaft=shaft, soil=soil).compute()
    clay, sand = r.layer_breakdown
    assert clay["method"] == "excluded"
    assert sand["effective_top_m"] == 1.0
    top = _choice(r, "side resistance near the ground surface")
    assert top["how"] == "default"
    assert "cohesionless soil within the top 1.5 m (1-1.5 m)" in \
        top["chosen"]
    assert "none in cohesive soil there (0-1 m)" in top["chosen"]
    given = DrillShaftAnalysis(shaft=shaft, soil=soil,
                               top_exclusion=1.5).compute()
    assert given.layer_breakdown[1]["effective_top_m"] == 1.5
    top = _choice(given, "side resistance near the ground surface")
    assert top["how"] == "given"
    assert "none over the top 1.5 m, in every layer" in top["chosen"]
    assert f"Q_ult {r.Q_ultimate:,.0f} kN" in top["alternatives"][0]
    # DS-2: all sand, so top_exclusion 0 is the default's value
    zero = _ds2(top_exclusion=0.0).compute()
    assert zero.layer_breakdown[0]["effective_top_m"] == 0.0
    assert zero.Q_ultimate == pytest.approx(_ds2().compute().Q_ultimate)
    assert [c for c in zero.choices
            if c["what"] == "side resistance near the ground surface"] == []
    with pytest.raises(ValueError, match="top_exclusion"):
        _ds2(top_exclusion=-1.0)
    # the casing, where deeper, takes its place, in sand too
    cased = DrillShaftAnalysis(
        shaft=DrillShaft(diameter=1.2, length=15.0, casing_depth=3.0),
        soil=_ds2().soil).compute()
    assert cased.layer_breakdown[0]["effective_top_m"] == 3.0
    assert [c for c in cased.choices
            if c["what"] == "side resistance near the ground surface"] == []


def test_the_length_sweep_takes_the_settings():
    a = _ds2(K_over_Ko=2 / 3, top_exclusion=0.0)
    (row,) = a.capacity_vs_depth(depth_min=15.0, depth_max=15.0, n_points=1)
    assert row["Q_ultimate_kN"] == pytest.approx(a.compute().Q_ultimate,
                                                 abs=0.1)


def test_the_calc_steps_print_each_step_of_eq_10_13():
    from drilled_shaft.calc_steps import get_calc_steps
    a = _ds2()
    steps = {s.title: s for sec in get_calc_steps(a.compute(), a)
             for s in sec.items if hasattr(s, "title")}
    assert "Beta Method (Cohesionless Soils), GEC-10 (2018) Eq. 10-13" \
        in steps
    (layer,) = [s for t, s in steps.items() if t.startswith("Layer 1:")]
    sub = layer.substitution
    assert "0.47 × 15^0.6 = 241.8 kPa" in sub
    assert "OCR = 241.8 / 142.5 = 1.697" in sub
    assert "= 0.400" in sub and "f_SN = 0.400 × 142.5 = 56.9 kPa" in sub
    assert layer.result_value == "3,220.3"
    top = steps["Top Exclusion Zone"]
    assert top.result_value == "1.50 / 0.00"
    assert "cohesionless soil and rock: max(0, 0.00) = 0.00 m" in         top.substitution
