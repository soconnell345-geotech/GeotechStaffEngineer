"""BACKLOG GM-188: each bearing capacity method with its own N-gamma.

Wave w14 (F18) ran ``factor_method='meyerhof'`` and got q_ult = 1,466 kPa:
Meyerhof's shape and depth factors with Vesic's N-gamma, 2 (Nq + 1) tan
phi, because ``ngamma_method`` defaulted to 'vesic' for both factor sets.
Meyerhof's own N-gamma, (Nq - 1) tan(1.4 phi) (UFC 3-220-20 Table 5-6,
p. 294), gives 1,241 kPa. The result now also states its choices: the
method, the N-gamma, the shear mode and the allowable pressure's basis."""

import pytest

from bearing_capacity.capacity import BearingCapacityAnalysis, OWN_NGAMMA
from bearing_capacity.factors import (
    bearing_capacity_Ngamma, bearing_capacity_Nq)
from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer


def _analysis(factor_method="vesic", phi=32.0, footing=None, **kw):
    spec = dict(width=2.0, length=2.0, depth=1.0, shape="square")
    spec.update(footing or {})
    return BearingCapacityAnalysis(
        footing=Footing(**spec),
        soil=BearingSoilProfile(layer1=SoilLayer(
            cohesion=0.0, friction_angle=phi, unit_weight=19.0)),
        factor_of_safety=3.0, factor_method=factor_method, **kw)


def _choice(result, what):
    found = [c for c in result.choices if c["what"] == what]
    assert len(found) == 1, [c["what"] for c in result.choices]
    return found[0]


# ── the printed N-gamma ───────────────────────────────────────────────

def test_meyerhof_ngamma_is_the_printed_one():
    # UFC 3-220-20 Table 5-2 (p. 289) tabulates Meyerhof's N-gamma: 22 at
    # 32 deg
    assert round(bearing_capacity_Ngamma(32.0, "meyerhof")) == 22
    # Example B-1.6 (p. 603): "For soil with a friction angle of 37
    # degrees, the Meyerhof bearing capacity factors are Nq = 42.92 and
    # N-gamma = 53.27"
    assert bearing_capacity_Nq(37.0) == pytest.approx(42.92, abs=0.005)
    assert bearing_capacity_Ngamma(37.0, "meyerhof") == pytest.approx(
        53.27, abs=0.005)
    # Example B-7.4 (p. 656): mean phi' 35 deg, "Nq = 33, N-gamma = 37"
    assert round(bearing_capacity_Ngamma(35.0, "meyerhof")) == 37


def test_vesic_ngamma_is_the_printed_one():
    # GEC-6 Eq. 5-8 (p. 53), Table 5-1 (p. 54): 30.21 at 32 deg, 48.03 at
    # 35 deg (V-021 checks 48.0 against GEC-6 Example 1)
    assert bearing_capacity_Ngamma(32.0, "vesic") == pytest.approx(
        30.21, abs=0.005)
    assert bearing_capacity_Ngamma(35.0, "vesic") == pytest.approx(
        48.03, abs=0.005)


# ── the default N-gamma follows the method ────────────────────────────

@pytest.mark.parametrize("factor_method", ["vesic", "meyerhof", "terzaghi"])
def test_each_method_defaults_to_its_own_ngamma(factor_method):
    a = _analysis(factor_method)
    assert a.ngamma_method == OWN_NGAMMA[factor_method]


def test_the_f18_meyerhof_footing_uses_meyerhof_ngamma():
    r = _analysis("meyerhof").compute()
    assert r.ngamma_method == "meyerhof"
    assert r.Ngamma == pytest.approx(22.02, abs=0.01)
    assert r.q_ultimate == pytest.approx(1240.96, abs=0.01)


def test_the_old_mixed_method_is_still_there_when_named():
    r = _analysis("meyerhof", ngamma_method="vesic").compute()
    assert r.q_ultimate == pytest.approx(1465.88, abs=0.01)
    ng = _choice(r, "N-gamma")
    assert "a mixed method" in ng["basis"]
    assert ng["how"] == "given"


def test_vesic_and_terzaghi_results_do_not_change():
    assert _analysis("vesic").compute().q_ultimate == pytest.approx(
        1158.77, abs=0.01)
    r = _analysis("terzaghi").compute()
    assert r.ngamma_method == "kumbhojkar"


# ── Meyerhof: shape and inclined load not combined ────────────────────

def test_meyerhof_shape_factors_are_one_under_an_inclined_load():
    # UFC 3-220-20 Table 5-6 note A (p. 294): "Do not combine factors for
    # shape and inclined load"
    plain = _analysis("meyerhof").compute()
    assert plain.sc > 1.0 and plain.sq > 1.0
    r = _analysis("meyerhof", load_inclination=10.0).compute()
    assert (r.sc, r.sq, r.sgamma) == (1.0, 1.0, 1.0)
    assert r.q_ultimate == pytest.approx(594.93, abs=0.01)
    shape = _choice(r, "shape factors under inclined load")
    assert "note A" in shape["basis"]


def test_vesic_shape_factors_stay_under_an_inclined_load():
    r = _analysis("vesic", load_inclination=10.0).compute()
    assert r.sc > 1.0
    assert not [c for c in r.choices
                if c["what"] == "shape factors under inclined load"]


# ── the choices block ─────────────────────────────────────────────────

def test_the_result_states_method_ngamma_shear_mode_and_basis():
    r = _analysis("meyerhof").compute()
    assert [c["what"] for c in r.choices] == [
        "bearing capacity method", "N-gamma", "shear mode",
        "allowable bearing pressure", "depth factors"]
    method = _choice(r, "bearing capacity method")
    assert "Meyerhof (1963)" in method["chosen"]
    assert "Table 5-6 (p. 294)" in method["basis"]
    assert len(method["alternatives"]) == 2
    ng = _choice(r, "N-gamma")
    assert ng["how"] == "default"
    assert ng["chosen"].startswith("22.02 = (Nq - 1) tan(1.4 phi)")
    assert "the method's own" in ng["basis"]
    assert any(a.startswith("30.21 = 2 (Nq + 1) tan phi")
               for a in ng["alternatives"])
    assert _choice(r, "shear mode")["chosen"].startswith("general shear")
    basis = _choice(r, "allowable bearing pressure")
    assert basis["chosen"].startswith("gross")
    assert "Eq. 5-13 and 5-14" in basis["basis"]
    d = r.to_dict()
    assert [c["what"] for c in d["choices"]] == [c["what"]
                                                 for c in r.choices]
    assert "Choices:" in r.summary()


def test_terzaghi_states_its_ngamma_source_and_local_shear():
    r = _analysis("terzaghi", failure_mode="local").compute()
    ng = _choice(r, "N-gamma")
    assert "Kumbhojkar" in ng["chosen"]
    assert any("Coduto" in a for a in ng["alternatives"])
    shear = _choice(r, "shear mode")
    assert shear["chosen"].startswith("local shear")
    assert shear["how"] == "given"


def test_meyerhof_states_the_depth_parameter():
    k = _choice(_analysis("meyerhof").compute(), "depth factors")
    assert k["chosen"].startswith("k = Df/B = 0.500")
    r = _analysis("meyerhof", footing=dict(depth=3.0)).compute()
    k = _choice(r, "depth factors")
    assert "arctan(Df/B)" in k["chosen"] and k["how"] == "default"
    assert "Table 5-6" in k["basis"]
    assert "k = Df/B = 1.50" in k["alternatives"][-1]
    # GEC-6's d_q goes with GEC-6's own (Vesic) factors only
    assert not [a for a in k["alternatives"] if "'gec6'" in a]


def test_the_calc_steps_cite_meyerhof_from_ufc_table_5_6():
    from bearing_capacity.calc_steps import get_calc_steps
    a = _analysis("meyerhof", load_inclination=10.0)
    r = a.compute()
    steps = [item for section in get_calc_steps(r, a)
             for item in getattr(section, "items", [])]
    refs = {getattr(s, "title", ""): getattr(s, "reference", "")
            for s in steps}
    assert "UFC 3-220-20 Table 5-6" in refs["Bearing Capacity Factor N_γ"]
    assert refs["Shape Factors"] == "Meyerhof; UFC 3-220-20 Table 5-6"
    shape = [s for s in steps if getattr(s, "title", "") == "Shape Factors"]
    assert "note A" in shape[0].substitution


def test_the_tool_reports_meyerhof_ngamma_as_the_default_in_force():
    from funhouse_agent.dispatch import call_agent
    out = call_agent("bearing_capacity", "bearing_capacity_analysis", {
        "width": 2.0, "length": 2.0, "depth": 1.0, "shape": "square",
        "friction_angle": 32.0, "cohesion": 0.0, "unit_weight": 19.0,
        "factor_method": "meyerhof"})
    assert out["q_ultimate_kPa"] == pytest.approx(1240.96, abs=0.01)
    assert out["defaults_used"]["ngamma_method"] == "meyerhof"
    hows = {c["what"]: c["how"] for c in out["choices"]}
    assert hows["bearing capacity method"] == "given"
    assert hows["N-gamma"] == "default"
