"""GM-141: a module's calc steps render from the library path too.

The calc packages now hand the steps the inputs their tool's run used, and
the steps print those. A caller of ``generate_calc_package`` with the
module's own result and only some of the inputs it gave the module function
(the library path, no adapter) must still get a package: an input left out
is the one the module function took (its signature default) or is named
without a number, never a KeyError.
"""

import math

import pytest

from calc_package import generate_calc_package


def _gi(method, analyze, **kw):
    import ground_improvement as gi
    return ("ground_improvement", getattr(gi, analyze)(**kw),
            {"method": method})


def _mo():
    from seismic_geotech import mononobe_okabe_KAE, seismic_earth_pressure
    from seismic_geotech.results import SeismicEarthPressureResult
    phi, kh = 32.0, 0.2
    delta = 2.0 / 3.0 * phi
    KAE = mononobe_okabe_KAE(phi, delta, kh, 0.0, 0.0, 0.0)
    KA = math.tan(math.pi / 4 - math.radians(phi) / 2) ** 2
    f = seismic_earth_pressure(19.0, 5.0, KAE, KA)
    result = SeismicEarthPressureResult(
        KAE=KAE, KPE=0.0, PAE_total=f["PAE_total_kN_per_m"],
        PA_static=f["PA_static_kN_per_m"],
        delta_PAE=f["delta_PAE_kN_per_m"],
        height_of_application=f["height_of_application_m"],
        phi=phi, delta=delta, kh=kh, kv=0.0)
    # only the inputs a library caller passes: no gamma, H or KA
    return "seismic_geotech", result, {"phi": phi, "kh": kh}


def _site():
    from seismic_geotech import classify_site, site_coefficients
    return ("seismic_geotech",
            site_coefficients(classify_site(vs30=270.0), Ss=1.0, S1=0.4), {})


def _liq():
    from seismic_geotech import evaluate_liquefaction
    from seismic_geotech.results import LiquefactionResult
    rows = evaluate_liquefaction([2, 4, 6], [8, 12, 20], [5, 10, 15],
                                 [18, 19, 19], amax_g=0.3, gwt_depth=1.5)
    return ("seismic_geotech",
            LiquefactionResult(layer_results=rows, amax_g=0.3,
                               gwt_depth=1.5), {})


def _wave(analysis):
    from wave_equation import (discretize_pile, generate_bearing_graph,
                               get_hammer, make_cushion_from_properties)
    result = generate_bearing_graph(
        get_hammer("Delmag D30-32"),
        make_cushion_from_properties(0.1, 0.1, 3.5e6),
        discretize_pile(15.0, 0.01, 200e6))
    return "wave_equation", result, analysis


def _group(analysis):
    from pile_group import (GroupLoad, analyze_vertical_group_simple,
                            create_rectangular_layout)
    piles = create_rectangular_layout(3, 3, 1.5, 1.5)
    load = GroupLoad(Vz=1800.0)
    return ("pile_group", analyze_vertical_group_simple(piles, load),
            {"piles": piles, "load": load, **analysis})


def _wall():
    from retaining_walls import CantileverWallGeometry, analyze_cantilever_wall
    geom = CantileverWallGeometry(wall_height=4.0)
    # the inputs analyze_cantilever_wall requires, and nothing else
    return ("retaining_walls",
            analyze_cantilever_wall(geom, gamma_backfill=18.0,
                                    phi_backfill=30.0),
            {"geom": geom, "gamma_backfill": 18.0, "phi_backfill": 30.0})


CASES = {
    "aggregate_piers": lambda: _gi("aggregate_piers",
                                   "analyze_aggregate_piers",
                                   column_diameter=0.76, spacing=2.0),
    "wick_drains": lambda: _gi("wick_drains", "analyze_wick_drains",
                               spacing=1.5, ch=3.0, cv=1.0, Hdr=5.0,
                               time=1.0),
    "surcharge": lambda: _gi("surcharge", "analyze_surcharge_preloading",
                             S_ultimate=0.3, surcharge_kPa=50.0, cv=3.0,
                             Hdr=4.0, sigma_v0=40.0),
    "surcharge_with_drains": lambda: _gi(
        "surcharge", "analyze_surcharge_preloading", S_ultimate=0.3,
        surcharge_kPa=50.0, cv=3.0, Hdr=4.0, ch=3.0, drain_spacing=1.5),
    "vibro": lambda: _gi("vibro", "analyze_vibro_compaction",
                         fines_content=8.0, initial_N_spt=12.0),
    "seismic_earth_pressure": _mo,
    "site_classification": _site,
    "liquefaction": _liq,
    "wave_no_inputs": lambda: _wave({}),
    "wave_quakes_only": lambda: _wave({"quake_side": 0.0025}),
    "pile_group": lambda: _group({}),
    "pile_group_efficiency": lambda: _group({
        "n_rows": 3, "n_cols": 3, "spacing": 1.5, "pile_diameter": 0.4,
        "soil_type": "cohesive", "cu": 50.0}),
    "pile_group_efficiency_no_soil": lambda: _group({
        "n_rows": 3, "n_cols": 3, "spacing": 1.5, "pile_diameter": 0.4}),
    "cantilever_wall": _wall,
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_steps_render_from_the_inputs_given(case):
    module, result, analysis = CASES[case]()
    html = generate_calc_package(module=module, result=result,
                                 analysis=analysis, project_name="t")
    assert "<html" in html.lower()
    assert "Traceback" not in html


def test_mo_ka_left_out_is_rankine_from_phi():
    module, result, analysis = _mo()
    html = generate_calc_package(module=module, result=result,
                                 analysis=analysis)
    ka = math.tan(math.pi / 4 - math.radians(32.0) / 2) ** 2
    assert f"{ka:.4f}" in html
    # gamma and H not given: the force steps name them, never a number
    assert "0.5 × γ × H²" in html


def test_wick_drains_inputs_left_out_are_the_modules_defaults():
    from ground_improvement.calc_steps import get_input_summary
    module, result, analysis = CASES["wick_drains"]()
    items = {i.name: i.value for i in get_input_summary(result, analysis)}
    # analyze_wick_drains' defaults (GM-291): smear ignored, so no smear
    # ratio is listed, and a 2.5 in drain (GEC-13 p. 2-35)
    assert "s_r" not in items
    assert items["d_w"] == pytest.approx(0.0635)


def test_a_required_input_left_out_is_named():
    # no invented backfill: the steps ask for what the module required
    module, result, analysis = _wall()
    del analysis["phi_backfill"]
    with pytest.raises(ValueError, match="phi_backfill"):
        generate_calc_package(module=module, result=result,
                              analysis=analysis)
