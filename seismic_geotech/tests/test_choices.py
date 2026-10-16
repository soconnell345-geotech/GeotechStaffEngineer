"""What the seismic earth pressure and liquefaction results say they chose
(backlog GM-115), and rd and MSF reported per layer."""

from seismic_geotech import evaluate_liquefaction
from seismic_geotech.liquefaction import (
    liquefaction_choices, magnitude_scaling_factor, stress_reduction_rd,
)
from seismic_geotech.mononobe_okabe import (
    earth_pressure_choices, mo_no_solution_warning,
)
from seismic_geotech.results import LiquefactionResult


def _by_what(choices):
    return {c["what"]: c for c in choices}


def test_mo_choices_say_the_defaults_and_the_static_basis():
    c = _by_what(earth_pressure_choices(32, 0.15, ka="rankine", forces=True))
    assert c["KAE"]["chosen"] == "Mononobe-Okabe"
    assert c["kh"]["how"] == "given" and c["kv"]["how"] == "default"
    assert c["wall friction delta"]["chosen"] == "21.3 deg (2/3 phi)"
    assert c["static KA"]["chosen"] == "Rankine, 0.3073"
    assert c["height of delta-PAE"]["basis"].startswith(
        "Seed & Whitman (1970); UFC 3-220-20 Sec. 4-4.4.1")
    given = _by_what(earth_pressure_choices(32, 0.15, kv=0.05, delta_deg=16))
    assert given["wall friction delta"]["how"] == "given"
    assert given["kv"]["how"] == "given"
    assert "static KA" not in given and "height of delta-PAE" not in given
    # a kv the caller typed is given, even when it is the default value
    assert _by_what(earth_pressure_choices(32, 0.15, kv=0.0))["kv"]["how"] \
        == "given"


def test_assumed_ka_of_the_package_is_labelled():
    ka = _by_what(earth_pressure_choices(36, 0.15, ka=0.333,
                                         ka_how="default"))["static KA"]
    assert ka["chosen"] == "0.333 assumed (Rankine Ka for phi 30 deg)"
    assert ka["basis"] == "not computed from phi (36 deg here)"
    assert ka["how"] == "default"


def test_mo_fallback_is_named_and_warned():
    # phi - theta - i < 0: the module falls back to Ka + 0.75 kh
    c = _by_what(earth_pressure_choices(20, 0.5))
    assert c["KAE"]["chosen"].startswith("Ka + 0.75 kh (Seed & Whitman")
    assert "has no solution" in c["KAE"]["basis"]
    assert "leaves out the backfill slope i, the wall batter, kv and the wall friction delta" \
        in c["KAE"]["basis"]
    assert any("trial wedge" in a for a in c["KAE"]["alternatives"])
    note = mo_no_solution_warning(20, 0.5)
    assert "no solution" in note and "trial wedge" in note
    assert mo_no_solution_warning(32, 0.15) is None


def test_rd_and_msf_per_layer_are_the_factors_inside_csr():
    layers = evaluate_liquefaction(
        layer_depths=[2.0, 6.0, 12.0], layer_N160=[8, 12, 35],
        layer_FC=[5, 15, 10], layer_gamma=[18, 19, 19], amax_g=0.3,
        gwt_depth=1.5, M=7.0)
    for r in layers:
        assert r["rd"] == round(stress_reduction_rd(r["depth_m"]), 4)
        assert r["MSF"] == round(magnitude_scaling_factor(7.0), 3)
        expected = (0.65 * 0.3 * r["sigma_v_kPa"] / r["sigma_v_eff_kPa"]
                    * r["rd"] / r["MSF"])
        assert abs(expected - r["CSR"]) < 2e-3 * r["CSR"]
    at_75 = evaluate_liquefaction([3.0], [10], [5], [18], 0.3, 1.0, M=7.5)
    assert at_75[0]["MSF"] == 1.0


def test_liquefaction_choices_name_the_procedure_and_what_it_leaves_out():
    layers = evaluate_liquefaction([2.0, 6.0], [8, 35], [5, 10], [18, 19],
                                   0.3, 1.5, M=7.0)
    c = _by_what(liquefaction_choices(layers, 7.0))
    assert "Youd et al. (2001)" in c["triggering procedure"]["chosen"]
    assert c["overburden correction K_sigma"]["chosen"] == "not applied (1.0)"
    assert "6 m" in c["CRR where (N1)60cs >= 30"]["basis"]
    result = LiquefactionResult(layer_results=layers, amax_g=0.3,
                                magnitude=7.0, choices=list(c.values()))
    assert result.to_dict()["choices"] == list(c.values())
    (t,) = result.get_tables()
    assert t.column("rd") == [r["rd"] for r in layers]
    assert t.column("MSF") == [r["MSF"] for r in layers]
