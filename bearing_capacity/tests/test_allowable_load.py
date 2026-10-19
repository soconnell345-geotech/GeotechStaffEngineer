"""BACKLOG GM-170: the allowable structural load on a footing, with its basis.

Waves w12 and w13 (F18, F18j) headlined "an allowable column load of about
1,545 kN" on a 2 m square footing 1 m deep: q_all x A, the GROSS load on the
bearing surface, which carries the footing's and the soil's weight as well
as the column's (UFC 3-220-20 Sec. 5-2.2 Eq. 5-1). The result now gives the
structural load on the net allowable pressure (Eq. 5-3 and 5-14) beside the
gross load, each named for what it is, so the gross product is not taken
for the column load."""

import pytest

from bearing_capacity.capacity import BearingCapacityAnalysis
from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer


def _analysis(factor_method="vesic", **footing):
    spec = dict(width=2.0, length=2.0, depth=1.0, shape="square")
    spec.update(footing)
    return BearingCapacityAnalysis(
        footing=Footing(**spec),
        soil=BearingSoilProfile(layer1=SoilLayer(
            cohesion=0.0, friction_angle=32.0, unit_weight=19.0)),
        factor_of_safety=3.0, factor_method=factor_method)


def _f18(**footing):
    return _analysis(**footing).compute()


def test_the_f18_footing_gives_the_structural_load_on_the_net_pressure():
    r = _f18()
    assert r.q_allowable == pytest.approx(386.26, abs=0.01)
    assert r.q_overburden == pytest.approx(19.0)
    # UFC 3-220-20 Eq. 5-14: q_all,net = q_ult / FS - sigma_zD
    assert r.q_allowable_net == pytest.approx(386.26 - 19.0, abs=0.01)
    loads = r.allowable_loads()
    assert loads["area_m2"] == pytest.approx(4.0)
    # Eq. 5-3: Q_DL+LL = q_net x A, about 1,469 kN, not 1,545 kN
    assert loads["structural_load_kN"] == pytest.approx(1469.0, abs=0.1)
    assert loads["gross_load_kN"] == pytest.approx(1545.0, abs=0.1)
    assert "Eq. 5-14" in loads["basis"] and "Eq. 5-3" in loads["basis"]
    assert "Eq. 5-1" in loads["basis"] and "not the structural load" \
        in loads["basis"]
    assert "GEC-6 Sec. 5.2.2 (p. 53)" in loads["basis"]
    d = r.to_dict()
    assert d["q_allowable_net_kPa"] == pytest.approx(367.26, abs=0.01)
    assert d["allowable_load"] == loads
    # the numbers the result gave before are unchanged
    assert d["q_allowable_kPa"] == pytest.approx(386.26, abs=0.01)
    assert d["q_net_kPa"] == pytest.approx(1139.77, abs=0.01)


def test_a_strip_gives_its_loads_per_metre():
    r = _f18(shape="strip", length=None)
    loads = r.allowable_loads()
    assert set(loads) == {"structural_load_kN_per_m", "gross_load_kN_per_m",
                          "width_m", "basis"}
    assert loads["width_m"] == pytest.approx(2.0)
    assert loads["structural_load_kN_per_m"] == pytest.approx(
        r.q_allowable_net * 2.0, abs=0.1)


def test_an_eccentric_load_works_on_the_effective_area():
    r = _f18(eccentricity_B=0.2)
    loads = r.allowable_loads()
    assert loads["area_m2"] == pytest.approx(1.6 * 2.0)
    assert loads["structural_load_kN"] == pytest.approx(
        r.q_allowable_net * 3.2, abs=0.1)


@pytest.mark.parametrize("method", ["vesic", "terzaghi"])
def test_the_summary_and_the_calc_steps_say_it(method):
    from bearing_capacity.calc_steps import get_calc_steps
    from calc_package.data_model import CalcStep
    a = _analysis(method)
    r = a.compute()
    s = r.summary()
    assert f"qall,net = {r.q_allowable_net:,.1f} kPa" in s
    assert "UFC 3-220-20 Eq. 5-14" in s
    assert (f"Allowable structural load Q = "
            f"{r.allowable_loads()['structural_load_kN']:,.0f} kN") in s
    steps = [i for sec in get_calc_steps(r, a) for i in sec.items
             if isinstance(i, CalcStep)]
    by_name = {st.result_name: st for st in steps}
    assert by_name["q_all,net"].result_value == f"{r.q_allowable_net:,.1f}"
    assert by_name["q_all,net"].reference == "UFC 3-220-20 Eq. 5-14"
    assert by_name["Q_DL+LL"].reference == "UFC 3-220-20 Eq. 5-3"
    if method == "vesic":
        assert by_name["Q_DL+LL"].result_value == "1,469"


def test_the_tool_returns_them():
    from funhouse_agent.dispatch import call_agent
    out = call_agent("bearing_capacity", "bearing_capacity_analysis", {
        "width": 2.0, "length": 2.0, "depth": 1.0, "shape": "square",
        "friction_angle": 32, "unit_weight": 19, "factor_of_safety": 3})
    assert out["allowable_load"]["structural_load_kN"] == pytest.approx(
        1469.0, abs=0.1)
    assert out["allowable_load"]["gross_load_kN"] == pytest.approx(
        1545.0, abs=0.1)

