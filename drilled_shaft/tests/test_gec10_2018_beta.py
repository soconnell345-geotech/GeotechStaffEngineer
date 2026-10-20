"""GM-241: side resistance in cohesionless soil follows GEC-10 (2018) as
printed, by default.

GEC-10 (FHWA-NHI-18-024) Sec. 10.3.5.1, Step 5 of the design for axial
load, pp. 10-10 to 10-12: R_SN = pi B dz f_SN, f_SN = beta sigma'v on the
layer's average sigma'v (Eqs. 10-5, 10-7); beta by Eq. 10-13 with sigma'p
from N60 (Eq. 10-11 or 10-12), limited to its value at 7.5 ft; K/Ko = 1
(p. 10-11), 2/3 where construction quality is not controlled (p. 10-12).
The depth beta of O'Neill and Reese (1999), the default until GM-241,
stays by name. Example 10-1 is V-058 in ``validation_examples``.
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
    """Each step of Eqs. 10-5 to 10-13 by hand, over 1.5-15 m."""
    sv = 19.0 * (1.5 + 15.0) / 2                       # 156.75 kPa
    sp = PA * 0.47 * 15.0 ** 0.6                       # 241.8 kPa
    s = math.sin(math.radians(34.0))
    Ko = (1 - s) * (sp / sv) ** s                      # 0.562
    beta = Ko * math.tan(math.radians(34.0))           # 0.379
    Qs = math.pi * 1.2 * 13.5 * beta * sv              # 3,022.6 kN
    r = _ds2().compute()
    assert Qs == pytest.approx(3022.6, abs=0.1)
    assert r.Q_side_sand == pytest.approx(Qs, rel=1e-9)
    (lb,) = r.layer_breakdown
    st = lb["beta_steps"]
    assert (st["sigma_p_kPa"], st["OCR"], st["Ko"], st["beta"]) == (
        pytest.approx(sp, abs=1e-3), pytest.approx(sp / sv, abs=1e-3),
        pytest.approx(Ko, abs=1e-3), pytest.approx(beta, abs=1e-3))
    assert r.Q_ultimate == pytest.approx(3998.1, abs=0.1)


def test_the_depth_beta_is_unchanged_by_name():
    r = _ds2(beta_method="depth").compute()
    assert r.Q_ultimate == pytest.approx(7328.0, abs=0.1)
    assert "rational" not in r.layer_breakdown[0]["method"]


def test_the_method_choice_lists_its_alternatives_with_values():
    c = _choice(_ds2().compute(), "side resistance in cohesionless soil")
    assert c["how"] == "default"
    assert "Eq. 10-13" in c["chosen"] and "3,023 kN" in c["chosen"]
    assert "Sec. 10.3.5.1" in c["basis"] and "pp. 10-10 to 10-12" in c["basis"]
    alts = " | ".join(c["alternatives"])
    assert "sub-layers of 0.1 m" in alts and "Q_ult 3,890 kN" in alts
    assert "beta_method 'depth'" in alts and "Q_ult 7,328 kN" in alts
    c = _choice(_ds2(beta_method="depth").compute(),
                "side resistance in cohesionless soil")
    assert c["how"] == "given" and "O'Neill and Reese" in c["chosen"]
    assert "Q_ult 3,998 kN" in c["alternatives"][0]


def test_each_judgment_is_a_choice_with_its_alternatives():
    r = _ds2().compute()
    top = _choice(r, "side resistance near the ground surface")
    assert "p. 10-17" in top["basis"] and "Example 10-1" in top["basis"]
    assert "Q_ult 4,196 kN" in top["alternatives"][0]
    k = _choice(r, "K/Ko")
    assert k["how"] == "default" and "2/3 of its" in k["basis"]
    assert "Q_ult 2,991 kN" in k["alternatives"][0]
    sp = _choice(r, "preconsolidation stress")
    assert "m = 0.6" in sp["chosen"] and sp["how"] == "default"
    assert any("'silty_sand'" in a and "Q_ult 5,067 kN" in a
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
    assert first["sigma_v_kPa"] == pytest.approx(19.0 * 1.75, abs=0.05)
    assert first["fs_kPa"] == pytest.approx(want["beta"] * 19.0 * 1.75,
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


def test_the_top_zone_is_an_input():
    zero = _ds2(top_exclusion=0.0).compute()
    assert zero.layer_breakdown[0]["effective_top_m"] == 0.0
    assert zero.Q_ultimate == pytest.approx(4195.9, abs=0.5)
    assert [c for c in zero.choices
            if c["what"] == "side resistance near the ground surface"] == []
    # the casing, where deeper, takes its place
    cased = DrillShaftAnalysis(
        shaft=DrillShaft(diameter=1.2, length=15.0, casing_depth=3.0),
        soil=_ds2().soil, top_exclusion=0.0).compute()
    assert cased.layer_breakdown[0]["effective_top_m"] == 3.0


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
    assert "OCR = 241.8 / 156.8 = 1.543" in sub
    assert "= 0.379" in sub and "f_SN = 0.379 × 156.8 = 59.4 kPa" in sub
    assert layer.result_value == "3,022.6"
