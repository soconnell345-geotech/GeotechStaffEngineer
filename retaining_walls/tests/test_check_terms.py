"""GM-143: the cantilever wall's result carries the terms each check used,
and the calc steps print those rather than working Ka, Pa and the weights
out again."""

import dataclasses

import pytest

from retaining_walls.cantilever import (
    _sliding_dict, _overturning_dict, _bearing_dict, analyze_cantilever_wall,
    check_bearing, check_overturning, check_sliding,
)
from retaining_walls.earth_pressure import (
    active_force_components, horizontal_force_active,
)
from retaining_walls.geometry import CantileverWallGeometry

CASES = {
    "level rankine": (dict(wall_height=6.0, base_width=3.6, toe_length=0.6,
                           stem_thickness_top=0.3, stem_thickness_base=0.6),
                      dict(pressure_method="rankine")),
    "sloped rankine, surcharge": (
        dict(wall_height=5.0, base_width=3.2, toe_length=0.5,
             stem_thickness_base=0.5, backfill_slope=15.0, surcharge=10.0),
        dict(pressure_method="rankine")),
    "coulomb, cohesion, passive key": (
        dict(wall_height=5.0, base_width=3.0, toe_length=0.6,
             has_shear_key=True, key_depth=0.6),
        dict(pressure_method="coulomb", c_backfill=5.0, include_passive=True,
             c_foundation=10.0, q_allowable=250.0)),
    "outside the middle third, toe side": (
        dict(wall_height=5.0, base_width=2.4, toe_length=0.3),
        dict(pressure_method="rankine", q_allowable=200.0)),
    "given interface": (
        dict(wall_height=4.0, base_width=2.8, toe_length=0.4),
        dict(delta_base=24.0, base_adhesion=5.0,
             sliding_cohesion="friction_plus_adhesion")),
    # GM-212: phi_foundation 0, so the undrained rule, ca = cu on B
    "undrained foundation": (
        dict(wall_height=4.0, base_width=2.8, toe_length=0.4),
        dict(phi_foundation=0.0, c_foundation=40.0)),
    "friction and table adhesion": (
        dict(wall_height=4.0, base_width=2.8, toe_length=0.4),
        dict(phi_foundation=26.0, c_foundation=30.0,
             sliding_cohesion="friction_plus_adhesion")),
}


def _run(name):
    g, kw = CASES[name]
    geom = CantileverWallGeometry(**g)
    return geom, kw, analyze_cantilever_wall(geom, 18.0, 32.0, **kw)


@pytest.mark.parametrize("name", list(CASES))
def test_the_checks_dicts_come_from_the_terms_the_result_carries(name):
    geom, kw, r = _run(name)
    args = (geom, 18.0, 32.0, kw.get("c_backfill", 0.0))
    method = kw.get("pressure_method", "rankine")
    assert _sliding_dict(r.sliding_terms, r.FOS_sliding_required) \
        == check_sliding(*args, c_foundation=kw.get("c_foundation", 0.0),
                         phi_foundation=kw.get("phi_foundation"),
                         pressure_method=method,
                         include_passive=kw.get("include_passive", False),
                         delta_base=kw.get("delta_base"),
                         base_adhesion=kw.get("base_adhesion"),
                         sliding_cohesion=kw.get("sliding_cohesion"))
    assert _overturning_dict(r.overturning_terms,
                             r.FOS_overturning_required) \
        == check_overturning(*args, pressure_method=method)
    assert _bearing_dict(r.bearing_terms) == check_bearing(
        *args, q_allowable=kw.get("q_allowable"), pressure_method=method)


@pytest.mark.parametrize("name", list(CASES))
def test_the_terms_add_up_to_the_results_numbers(name):
    geom, kw, r = _run(name)
    ep, b, o = r.earth_pressure, r.bearing_terms, r.overturning_terms
    # the thrust is the earth-pressure module's, parts and total
    parts = active_force_components(18.0, ep["H_active"], ep["Ka"],
                                    kw.get("c_backfill", 0.0),
                                    geom.surcharge)
    assert ep["Pa"] == parts["Pa"] == horizontal_force_active(
        18.0, ep["H_active"], ep["Ka"], kw.get("c_backfill", 0.0),
        geom.surcharge)[0]
    assert ep["Pa"] == pytest.approx(max(
        ep["Pa_earth"] + ep["Pa_surcharge"] + ep["Pa_cohesion"], 0.0))
    # the vertical forces are what each check sums (GM-157: a surcharge
    # over the heel that is not a resisting force is summed for bearing
    # only)
    f = r.vertical_forces
    assert sum(i["W"] for i in f if i["bearing"]) == pytest.approx(b["V"])
    assert sum(i["M"] for i in f if i["bearing"]) \
        == pytest.approx(b["M_stab"])
    assert sum(i["W"] for i in f if i["resists"]) == pytest.approx(o["V"])
    assert sum(i["M"] for i in f if i["resists"]) \
        == pytest.approx(o["M_stabilizing"])
    assert r.FOS_overturning == round(o["M_stabilizing"]
                                      / o["M_overturning"], 3)
    # the eccentricity check is on the overturning check's loads; the
    # bearing pressures on the bearing check's
    assert r.eccentricity == round(o["eccentricity"], 3)
    assert r.q_toe == round(b["q_toe"], 1)


def test_the_stem_arm_is_the_modules():
    geom, _, r = _run("level rankine")
    stem = next(f for f in r.vertical_forces
                if f["component"] == "Stem (rect)")
    assert stem["x"] == pytest.approx(geom.toe_length
                                      + geom.stem_thickness_top / 2.0)


def test_toe_side_resultant_uses_the_toe_triangle():
    _, _, r = _run("outside the middle third, toe side")
    b = r.bearing_terms
    assert b["distribution"] == "triangular_toe"
    assert b["B_eff"] == pytest.approx(3.0 * b["x_resultant"])
    assert b["q_toe"] == pytest.approx(2.0 * b["V"] / b["B_eff"])


def test_heel_side_resultant_prints_the_heel_triangle(monkeypatch):
    # GM-143: the steps printed q_toe = 2V / (3 x_R) for a resultant
    # beyond the middle third on either side; on the heel side the module
    # works out q_heel = 2V / (3 (B - x_R)). (A real wall rarely gets
    # there, so the weight is placed near the heel, as in GM-103's test.)
    import retaining_walls.cantilever as cw
    monkeypatch.setattr(cw, "_wall_weight_items", lambda g, gb, gc: [
        {"component": "weight near the heel", "W": 500.0, "x": 3.9}])
    geom = CantileverWallGeometry(wall_height=3.0, base_width=4.0,
                                  toe_length=0.2, stem_thickness_top=0.3,
                                  stem_thickness_base=0.3,
                                  base_thickness=0.4)
    r = cw.analyze_cantilever_wall(geom, 18.0, 45.0, q_allowable=2000.0)
    b = r.bearing_terms
    assert b["distribution"] == "triangular_heel"
    assert b["q_heel"] == pytest.approx(
        2.0 * b["V"] / (3.0 * (r.base_width - b["x_resultant"])))
    text = _steps(r, {"geom": geom, "gamma_backfill": 18.0,
                      "phi_backfill": 45.0, "q_allowable": 2000.0})
    assert "q_heel = 2V / (3 \u00d7 (B - x_R))" in text
    assert "q_toe = 2V / (3 \u00d7 x_R)" not in text


def _steps(result, analysis):
    from retaining_walls.calc_steps import get_calc_steps
    return repr(get_calc_steps(result, analysis))


@pytest.mark.parametrize("name", list(CASES))
def test_calc_steps_print_the_results_terms(name):
    geom, kw, r = _run(name)
    analysis = {"geom": geom, "gamma_backfill": 18.0, "phi_backfill": 32.0,
                **kw}
    text = _steps(r, analysis)
    stem = next(f for f in r.vertical_forces
                if f["component"] == "Stem (rect)")
    assert f"'Stem (rect)', '{stem['W']:.1f}', '{stem['x']:.3f}'" in text
    o, b = r.overturning_terms, r.bearing_terms
    if abs(o["V"] - b["V"]) > 1e-9:
        # GM-157: each check's own total
        assert f"'TOTAL, bearing', '{b['V']:.1f}'" in text
        assert (f"'TOTAL, sliding / overturning / eccentricity', "
                f"'{o['V']:.1f}'") in text
    else:
        assert f"'TOTAL', '{b['V']:.1f}'" in text
    assert (f"FOS_ot = {r.overturning_terms['M_stabilizing']:.1f} / "
            f"{r.overturning_terms['M_overturning']:.1f}") in text
    # a result built by hand carries no terms: the steps work them out
    # with the module's own term functions and print the same
    bare = dataclasses.replace(r, earth_pressure=None, vertical_forces=None,
                               sliding_terms=None, overturning_terms=None,
                               bearing_terms=None)
    assert _steps(bare, analysis) == text
