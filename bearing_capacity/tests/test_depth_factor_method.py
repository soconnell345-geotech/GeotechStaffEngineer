"""BACKLOG GM-197: the depth factors are a stated choice, pinned to the
printed sources, and GM-203: Vesic's load inclination factors say what they
took.

"hansen" (default): Brinch Hansen's (1970) k = Df/B for Df/B < 1 and
arctan(Df/B) from 1, as UFC 3-220-20 Tables 5-7 and 5-8 note B (pp.
295-296) print it. "gec6": GEC-6 Eq. 5-14 (p. 63) with d_q of Table 5-4
(p. 57) on the surcharge term alone. "none": no depth factors (GEC-6 Sec.
5.2.3.3)."""

import math

import pytest

from bearing_capacity.capacity import BearingCapacityAnalysis
from bearing_capacity.factors import depth_factors, depth_parameter_k
from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer

#: GEC-6 Table 5-4 (p. 57), "Depth correction factor (Brinch Hansen,
#: 1970)": d_q by friction angle and Df/Bf, as printed.
GEC6_TABLE_5_4 = {
    32: {1: 1.20, 2: 1.30, 4: 1.35, 8: 1.40},
    37: {1: 1.20, 2: 1.25, 4: 1.30, 8: 1.35},
    42: {1: 1.15, 2: 1.20, 4: 1.25, 8: 1.30},
}


@pytest.mark.parametrize("phi, df_b, printed", [
    (phi, df_b, dq) for phi, row in GEC6_TABLE_5_4.items()
    for df_b, dq in row.items()])
def test_gec6_reproduces_every_printed_cell_of_table_5_4(phi, df_b, printed):
    dc, dq, dg = depth_factors(phi, Df=df_b * 2.0, B=2.0, method="vesic",
                               depth_method="gec6")
    # printed to the nearest 0.05
    assert round(dq / 0.05) * 0.05 == pytest.approx(printed, abs=1e-9)
    assert (dc, dg) == (1.0, 1.0)            # Eq. 5-14: d_q alone
    # the Hansen default gives the same d_q over the printed range
    assert depth_factors(phi, df_b * 2.0, 2.0, "vesic")[1] == \
        pytest.approx(dq, rel=1e-12)


def test_k_one_does_not_reproduce_the_df_b_one_cells():
    # the old k = Df/B = 1 at Df/B = 1 gives 1.276 at 32 deg, which rounds
    # to 1.30; Table 5-4 prints 1.20 (GM-197 before/after: 1.276 -> 1.217)
    t = 2 * math.tan(math.radians(32)) * (1 - math.sin(math.radians(32))) ** 2
    assert round((1 + t) / 0.05) * 0.05 == pytest.approx(1.30)
    assert depth_factors(32, 2.0, 2.0)[1] == pytest.approx(1 + t * math.pi / 4)
    assert depth_factors(32, 2.0, 2.0)[1] == pytest.approx(1.217, abs=5e-4)


def test_gec6_is_one_below_df_b_one_as_its_worked_example_takes_it():
    # GEC-6 p. B-67: "Since the ratio ... Df, to the footing width, Bf, will
    # be less than one, ... dq, should be set equal to 1.0"
    assert depth_factors(38, 1.5, 6.0, "vesic", "gec6") == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("df_b", [0.25, 0.5, 0.999, 1.0, 1.5, 3.0, 8.0])
@pytest.mark.parametrize("phi", [0.0, 25.0, 32.0, 40.0])
def test_hansen_is_ufc_tables_5_7_and_5_8_as_printed(phi, df_b):
    # note B: "For Df / B < 1: k = Df / B. For Df / B >= 1: k = tan-1(Df/B)
    # in radians"; Table 5-8: dq = 1 + 2 k tan phi (1 - sin phi)^2; Table
    # 5-7 (phi = 0): dc = 0.4 k on the additive form, 1 + 0.4 k here
    k = df_b if df_b < 1 else math.atan(df_b)
    assert depth_parameter_k(df_b * 2.0, 2.0) == pytest.approx(k)
    dc, dq, dg = depth_factors(phi, df_b * 2.0, 2.0, "vesic")
    p = math.radians(phi)
    assert dq == pytest.approx(1 + 2 * k * math.tan(p) * (1 - math.sin(p)) ** 2)
    if phi == 0:
        assert dc == pytest.approx(1 + 0.4 * k)
    assert dg == 1.0


def test_hansen_agrees_with_the_reference_library_where_installed():
    ch5 = pytest.importorskip("geotech_references.dm7_2.chapter5")
    for phi in (20.0, 32.0, 40.0):
        for df_b in (0.5, 1.0, 2.0, 6.0):
            dq = depth_factors(phi, df_b * 2.0, 2.0, "vesic")[1]
            assert dq == pytest.approx(
                ch5.hansen_depth_factors(df_b * 2.0, 2.0, phi)[1], rel=1e-12)
    assert depth_factors(0.0, 3.0, 2.0, "vesic")[0] == pytest.approx(
        1 + ch5.hansen_depth_factor_phi0(3.0, 2.0))


def test_meyerhof_keeps_table_5_6_k_up_to_df_b_one():
    # UFC 3-220-20 Table 5-6 prints k = Df/B for Meyerhof: unchanged at 1
    assert depth_parameter_k(2.0, 2.0, "meyerhof") == 1.0
    assert depth_parameter_k(3.0, 2.0, "meyerhof") == pytest.approx(
        math.atan(1.5))


def test_none_and_the_refusals():
    assert depth_factors(32, 8.0, 2.0, "vesic", "none") == (1.0, 1.0, 1.0)
    assert depth_factors(32, 8.0, 2.0, "meyerhof", "none") == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="'vesic'"):
        depth_factors(32, 4.0, 2.0, "meyerhof", "gec6")
    with pytest.raises(ValueError, match="depth_method"):
        depth_factors(32, 4.0, 2.0, "vesic", "brinch")


# ── the analysis and its choices ──────────────────────────────────────

def _analysis(depth=2.0, layer=None, footing=None, **kw):
    spec = dict(width=2.0, length=2.0, depth=depth, shape="square")
    spec.update(footing or {})
    return BearingCapacityAnalysis(
        footing=Footing(**spec),
        soil=BearingSoilProfile(layer1=SoilLayer(**(layer or dict(
            cohesion=0.0, friction_angle=32.0, unit_weight=19.0)))), **kw)


def _choice(result, what):
    found = [c for c in result.choices if c["what"] == what]
    assert len(found) == 1, [c["what"] for c in result.choices]
    return found[0]


def test_df_b_one_before_and_after():
    # GM-197: square 2 m, Df 2 m, phi 32: d_q 1.276 -> 1.217, q_ult
    # 2,170.70 -> 2,085.89 kPa (-3.9 %); clay c 50 kPa: d_c 1.400 -> 1.314,
    # q_ult 467.76 -> 441.29 kPa; GM-281 adds d_c as UFC 3-220-20 Eq. 5-21
    # does: 50 x 5.14 x (1.2 + 0.4 pi/4) + 36 = 425.14 kPa
    assert _analysis().compute().q_ultimate == pytest.approx(2085.89, abs=0.01)
    clay = _analysis(layer=dict(cohesion=50.0, friction_angle=0.0,
                                unit_weight=18.0)).compute()
    assert clay.dc == pytest.approx(1 + 0.4 * math.pi / 4)
    assert clay.q_ultimate == pytest.approx(
        50.0 * 5.14 * (1.2 + 0.4 * math.pi / 4) + 18.0 * 2.0)
    assert clay.q_ultimate == pytest.approx(425.14, abs=0.01)


def test_the_result_states_the_depth_factors_and_the_others():
    r = _analysis(depth=4.0).compute()
    d = _choice(r, "depth factors")
    assert d["how"] == "default"
    assert d["chosen"].startswith("k = arctan(Df/B) = 1.107 at Df/B = 2.00")
    assert "note B" in d["basis"]
    alts = d["alternatives"]
    assert any("'gec6'" in a and "Table 5-4" in a for a in alts)
    assert any("'none'" in a and "5.2.3.3" in a for a in alts)

    g = _analysis(depth=4.0, depth_factor_method="gec6").compute()
    d = _choice(g, "depth factors")
    assert d["how"] == "given"
    assert "Table 5-4" in d["chosen"] and "Eq. 5-14" in d["basis"]
    assert (g.dc, g.dgamma) == (1.0, 1.0)
    # c = 0: d_c is on a zero term, so the two agree here
    assert g.q_ultimate == pytest.approx(r.q_ultimate)
    n = _analysis(depth=4.0, depth_factor_method="none").compute()
    assert (n.dc, n.dq, n.dgamma) == (1.0, 1.0, 1.0)
    assert n.q_ultimate < r.q_ultimate


def test_a_surface_footing_has_no_depth_choice():
    r = _analysis(depth=0.0).compute()
    assert not [c for c in r.choices if c["what"] == "depth factors"]


def test_gec6_needs_vesic_and_the_name_is_checked():
    with pytest.raises(ValueError, match="factor_method='vesic'"):
        _analysis(factor_method="meyerhof", depth_factor_method="gec6")
    with pytest.raises(ValueError, match="depth_factor_method"):
        _analysis(depth_factor_method="vesic")


def test_two_layers_take_the_depth_method_too():
    a = BearingCapacityAnalysis(
        footing=Footing(width=2.0, length=2.0, depth=2.0, shape="square"),
        soil=BearingSoilProfile(
            layer1=SoilLayer(friction_angle=32.0, unit_weight=19.0,
                             thickness=1.0),
            layer2=SoilLayer(cohesion=20.0, unit_weight=17.0)),
        depth_factor_method="none")
    assert a.compute().dq == 1.0


def test_the_inclination_choice_says_what_vesic_took():
    sand = _analysis(load_inclination=10.0).compute()
    i = _choice(sand, "load inclination factors")
    assert "V cancels" in i["chosen"] and "Annex D.4" in i["basis"]
    assert any("Table 5-6" in a for a in i["alternatives"])
    assert any("5.2.3.7" in a for a in i["alternatives"])
    clay = _analysis(layer=dict(cohesion=10.0, friction_angle=30.0,
                                unit_weight=18.0),
                     footing=dict(shape="strip", length=None),
                     load_inclination=10.0, vertical_load=300.0).compute()
    i = _choice(clay, "load inclination factors")
    assert "V = 300 kN per metre" in i["chosen"]
    assert "A' = 2.00 m2 per metre" in i["chosen"]
    assert not [c for c in _analysis(
        load_inclination=10.0, factor_method="meyerhof").compute().choices
        if c["what"] == "load inclination factors"]


def test_the_calc_steps_cite_the_printed_tables():
    from bearing_capacity.calc_steps import get_calc_steps
    for kw, ref in (({}, "Tables 5-7, 5-8"),
                    ({"depth_factor_method": "gec6"}, "Table 5-4"),
                    ({"depth_factor_method": "none"}, "5.2.3.3")):
        a = _analysis(depth=4.0, load_inclination=5.0, **kw)
        r = a.compute()
        steps = {getattr(s, "title", ""): s for sec in get_calc_steps(r, a)
                 for s in getattr(sec, "items", [])}
        assert ref in steps["Depth Factors"].reference
        assert "Annex D.4" in steps["Inclination Factors"].reference
        assert "tan(β)" in steps["Inclination Factors"].equation
