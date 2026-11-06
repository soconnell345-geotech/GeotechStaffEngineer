"""DR-11 slice 2: cases split by their printed qualifier, more label
spellings, the shape from the case's own labels, and the OWN-130 setting for
a printout that gives only gamma'.

Offline, no model calls. Every label is made up in the shape of the generic
examples in ``module_work/dr11/LABELS_SUMMARY.md``; every number is
invented, and each expected value comes from the module itself.
"""

import pytest

from funhouse_agent.dispatch import call_agent
from report_ingest.recompute import (GAMMA_PRIME_ENV, recompute_calculation,
                                     recompute_record, table_markdown)


def nq(name, value, unit="", page=4):
    return {"name": name, "value": {"value": value, "unit": unit,
                                    "prov": {"page": page, "method": "text",
                                             "confidence": 1.0}}}


def words(name, text):
    return {"name": name, "text": text}


def bearing(**params):
    base = {"width": 2.0, "depth": 1.0, "unit_weight": 18.5,
            "factor_of_safety": 3.0, "shape": "square",
            "factor_method": "vesic"}
    return call_agent("bearing_capacity", "bearing_capacity_analysis",
                      {**base, **params})


def bearing_calc(results, inputs, method="Vesic",
                 subject="Square footing F-9"):
    return {"kind": "shallow_foundation_bearing", "method": method,
            "subject": subject, "pages": [3, 4], "inputs": list(inputs),
            "results": list(results)}


SHARED = [nq("Footing width B", 2.0, "m"), nq("Embedment depth Df", 1.0, "m"),
          nq("Unit weight", 18.5, "kN/m3"), nq("Factor of safety", 3.0)]


# -- 1. the case split ---------------------------------------------------------

def test_a_printout_with_several_cases_is_recomputed_case_by_case():
    # stratum x load case, the qualifier in several word orders
    l1 = bearing(friction_angle=30)["q_ultimate_kPa"]
    l2 = bearing(friction_angle=34)["q_ultimate_kPa"]
    s1 = bearing(cohesion=50, friction_angle=0)["q_ultimate_kPa"]
    s2 = bearing(cohesion=80, friction_angle=0)["q_ultimate_kPa"]
    inputs = SHARED + [
        nq("phi, stratum 1, long term", 30, "deg"),
        nq("Long-term friction angle (Stratum 2)", 34, "deg"),
        nq("su, stratum 1, short term", 50, "kPa"),
        nq("Stratum 2 undrained shear strength (short-term)", 80, "kPa")]
    results = [
        nq("Ultimate bearing capacity, stratum 1, long term", round(l1),
           "kPa"),
        nq("q_ult (stratum 2, long term)", round(l2), "kPa"),
        nq("Short-term ultimate bearing capacity, stratum 1", round(s1),
           "kPa"),
        nq("Stratum 2 ultimate bearing capacity - short term", round(s2),
           "kPa"),
        nq("Nq, stratum 3", 23.0)]           # a case with no claim: no row
    rows = recompute_calculation(bearing_calc(results, inputs))
    by_case = {r["case"]: r for r in rows}
    assert set(by_case) == {"stratum 1, long term", "stratum 2, long term",
                            "stratum 1, short term", "stratum 2, short term"}
    for case, want in (("stratum 1, long term", l1),
                       ("stratum 2, long term", l2),
                       ("stratum 1, short term", s1),
                       ("stratum 2, short term", s2)):
        row = by_case[case]
        assert row["status"] == "recomputed", row
        assert row["flag"] == "agrees", row
        assert row["recomputed_si"] == pytest.approx(want)
    # the long-term case never sees the short-term su, and the reverse
    assert "su" not in by_case["stratum 1, long term"]["inputs_used"]
    assert "friction_angle" not in by_case["stratum 2, short term"][
        "inputs_used"]
    md = table_markdown(recompute_record({"calculations": [
        bearing_calc(results, inputs)]}))
    assert "[stratum 2, short term]" in md


def test_a_case_letter_in_brackets_picks_its_own_width():
    a = bearing(width=1.5, friction_angle=32)["q_allowable_kPa"]
    b = bearing(width=2.5, friction_angle=32)["q_allowable_kPa"]
    inputs = [nq("B (Case A)", 1.5, "m"), nq("B (Case B)", 2.5, "m"),
              nq("Df", 1.0, "m"), nq("phi", 32, "deg"),
              nq("Unit weight", 18.5, "kN/m3"), nq("FS", 3.0)]
    rows = recompute_calculation(bearing_calc(
        [nq("q_all (Case A)", round(a), "kPa"),
         nq("Case B: allowable bearing pressure", round(b), "kPa")], inputs))
    by_case = {r["case"]: r for r in rows}
    assert by_case["case A"]["flag"] == "agrees", by_case["case A"]
    assert by_case["case B"]["flag"] == "agrees", by_case["case B"]
    assert by_case["case A"]["inputs_used"]["width"]["value"] == 1.5


def test_a_claim_naming_no_case_among_cased_inputs_is_not_guessed():
    inputs = SHARED[1:] + [nq("phi", 32, "deg"), nq("B (Case A)", 1.5, "m"),
                           nq("B (Case B)", 2.5, "m")]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", 900, "kPa")], inputs))
    assert row["status"] == "not_recomputed"
    assert "case" not in row
    assert any("width" in m and "more than once" in m
               for m in row["inputs_missing"])


# -- 2. labels taken wrongly ---------------------------------------------------

def test_ratios_of_width_length_and_depth_are_not_the_dimensions():
    q = bearing(friction_angle=32)["q_ultimate_kPa"]
    inputs = SHARED + [nq("phi", 32, "deg"), nq("L/B", 1.0),
                       nq("B/L ratio", 1.0), nq("B/Df", 2.0)]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", round(q), "kPa")], inputs))
    assert row["flag"] == "agrees", row
    assert "length" not in row["inputs_used"]
    assert row["inputs_used"]["depth"]["value"] == 1.0


ELASTIC = [nq("Applied pressure q", 150, "kPa"),
           nq("Footing width B", 2.0, "m"),
           nq("Elastic modulus Es", 20000, "kPa")]


def elastic_calc(results, inputs, method="elastic"):
    return {"kind": "settlement", "method": method,
            "subject": "Square footing", "pages": [7],
            "inputs": list(inputs), "results": list(results)}


def elastic_mm(**extra):
    return call_agent("settlement", "elastic_settlement",
                      {"q_net": 150, "B": 2.0, "Es": 20000, "shape": "square",
                       **extra})["immediate_settlement_mm"]


def test_a_correlation_ratio_and_water_pressure_are_not_applied_pressure():
    se = elastic_mm()
    (row,) = recompute_calculation(elastic_calc(
        [nq("Immediate settlement", round(se, 2), "mm")],
        ELASTIC + [nq("q/Nc", 6.0, "kPa"), nq("qc/N", 400, "kPa"),
                   nq("Water pressure at base", 9.8, "kPa")]))
    assert row["flag"] == "agrees", row
    assert row["inputs_used"]["q_net"]["value"] == 150


CONSOL = [nq("Layer thickness", 3.0, "m"),
          nq("Depth to center of layer", 4.5, "m"),
          nq("Initial void ratio e0", 0.95),
          nq("Compression index Cc", 0.32),
          nq("Recompression index Cr", 0.04),
          nq("Initial effective stress", 65, "kPa"),
          nq("Stress increase", 45, "kPa")]


def consol_mm(**layer):
    base = {"thickness": 3.0, "depth_to_center": 4.5, "e0": 0.95, "Cc": 0.32,
            "Cr": 0.04, "sigma_v0": 65}
    return call_agent("settlement", "consolidation_settlement",
                      {"layers": [{**base, **layer}], "delta_sigma": 45}
                      )["consolidation_settlement_mm"]


def test_drain_thickness_and_a_stress_ratio_are_not_layer_inputs():
    sc = consol_mm()
    (row,) = recompute_calculation(elastic_calc(
        [nq("Consolidation settlement", round(sc, 1), "mm")],
        CONSOL + [nq("Drain thickness", 0.004, "m"),
                  nq("Overburden stress ratio", 1.4)],
        method="consolidation"))
    assert row["flag"] == "agrees", row
    assert row["inputs_used"]["thickness"]["value"] == 3.0


def test_the_height_of_a_soil_layer_is_its_thickness():
    sc = consol_mm(thickness=2.0)
    inputs = [i for i in CONSOL if i["name"] != "Layer thickness"]
    (row,) = recompute_calculation(elastic_calc(
        [nq("Consolidation settlement", round(sc, 1), "mm")],
        inputs + [nq("Height of soil layer (H)", 2.0, "m")],
        method="consolidation"))
    assert row["flag"] == "agrees", row


def test_settings_and_sentences_are_skipped():
    se = elastic_mm()
    (row,) = recompute_calculation(elastic_calc(
        [nq("Immediate settlement", round(se, 2), "mm")],
        ELASTIC + [nq("Use elastic modulus from CPT (yes or no)", 1),
                   nq("a) modulus taken as 2.5 qc", 1),
                   nq("Footing width B reduced for the eccentricity of the "
                      "column load shown on structural drawings", 1.7, "m"),
                   words("Modulus", "from correlation")]))
    assert row["flag"] == "agrees", row
    assert row["inputs_used"]["Es"]["value"] == 20000
    assert row["inputs_used"]["B"]["value"] == 2.0


def test_an_unconfined_strength_is_never_su():
    q = bearing(cohesion=40, friction_angle=0)["q_ultimate_kPa"]
    inputs = SHARED + [nq("Undrained shear strength su", 40, "kPa"),
                       nq("Unconfined compressive strength qu (= 2 su)", 80,
                          "kPa")]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", round(q), "kPa")], inputs))
    assert row["flag"] == "agrees", row
    assert row["inputs_used"]["su"]["value"] == 40


# -- 2. labels no pattern matched ----------------------------------------------

@pytest.mark.parametrize("label", ["ρ", "rho_i", "Se", "δ", "Δh"])
def test_settlement_symbols_are_claims_with_a_length_unit(label):
    se = elastic_mm()
    (row,) = recompute_calculation(elastic_calc(
        [nq(label, round(se, 2), "mm"), nq("ρ (soil)", 1900, "kg/m3")],
        ELASTIC))
    assert row["status"] == "recomputed", row
    assert row["quantity"] == label and row["flag"] == "agrees"


@pytest.mark.parametrize("label,key", [("qall", "q_allowable_kPa"),
                                       ("q_a", "q_allowable_kPa"),
                                       ("qa,net", "q_allowable_net_kPa"),
                                       ("q_ult", "q_ultimate_kPa")])
def test_qall_qa_and_qult_are_claims(label, key):
    want = bearing(friction_angle=32)[key]
    (row,) = recompute_calculation(bearing_calc(
        [nq(label, round(want), "kPa")], SHARED + [nq("phi", 32, "deg")]))
    assert row["flag"] == "agrees", row
    assert row["recomputed_si"] == pytest.approx(want)


def test_depth_to_water_below_the_footing_adds_df_and_says_so():
    want = bearing(friction_angle=32, gwt_depth=1.5)["q_ultimate_kPa"]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", round(want), "kPa")],
        SHARED + [nq("phi", 32, "deg"),
                  nq("Depth to water below footing", 0.5, "m")]))
    assert row["flag"] == "agrees", row
    gwt = row["inputs_used"]["gwt_depth"]
    assert gwt["value"] == pytest.approx(1.5)
    assert "Df" in gwt["note"] and "added" in gwt["note"]


def test_water_depth_from_the_ground_is_taken_as_printed():
    want = bearing(friction_angle=32, gwt_depth=1.2)["q_ultimate_kPa"]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", round(want), "kPa")],
        SHARED + [nq("phi", 32, "deg"), nq("Water depth", 1.2, "m")]))
    assert row["flag"] == "agrees", row


def test_unit_wt_and_density_are_the_unit_weight_not_relative_density():
    want = bearing(friction_angle=32, unit_weight=19.0)["q_ultimate_kPa"]
    inputs = [i for i in SHARED if i["name"] != "Unit weight"]
    for label, value, unit in (("Unit Wt.", 19.0, "kN/m3"),
                               ("Soil density", 120.95, "pcf")):
        (row,) = recompute_calculation(bearing_calc(
            [nq("Ultimate bearing capacity", round(want), "kPa")],
            inputs + [nq("phi", 32, "deg"), nq(label, value, unit),
                      nq("Relative density Dr", 60, "%")]))
        assert row["flag"] == "agrees", (label, row)


def test_depth_factors_printed_as_d_q_or_dg_mean_depth_factors_were_used():
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", 900, "kPa"), nq("d q", 1.2),
         nq("dg", 1.0)], SHARED + [nq("phi", 32, "deg")]))
    assert row["alternatives"] == []


def test_the_do_not_alias_list_holds():
    # Esur is not Es, Is is not Iw, sigma'zD is not sigma'v0
    se = elastic_mm()
    (row,) = recompute_calculation(elastic_calc(
        [nq("Immediate settlement", round(se, 2), "mm")],
        ELASTIC + [nq("Esur", 60000, "kPa")]))
    assert row["flag"] == "agrees" and row["inputs_used"]["Es"]["value"] \
        == 20000
    (row,) = recompute_calculation(elastic_calc(
        [nq("Immediate settlement", 12, "mm")],
        ELASTIC + [nq("Influence factor Is", 0.6)]))
    assert row["status"] == "not_recomputed" and "Is" in row["reason"]
    sc = consol_mm()
    (row,) = recompute_calculation(elastic_calc(
        [nq("Consolidation settlement", round(sc, 1), "mm")],
        CONSOL + [nq("σ'zD", 30, "kPa")], method="consolidation"))
    assert row["flag"] == "agrees", row
    # an allowable "from settlement" among the inputs is not a claim
    (row,) = recompute_calculation(bearing_calc(
        [nq("Bearing check", 1.0)],
        SHARED + [nq("phi", 32, "deg"),
                  nq("Allowable bearing pressure from settlement", 150,
                     "kPa")]))
    assert row["status"] == "not_compared"


# -- 3. the shape from the case's own labels ----------------------------------

def test_the_shape_comes_from_the_case_s_own_labels():
    strip = bearing(width=1.2, shape="strip",
                    friction_angle=32)["q_ultimate_kPa"]
    square = bearing(width=2.0, shape="square",
                     friction_angle=32)["q_ultimate_kPa"]
    inputs = [nq("Strip footing width B", 1.2, "m"),
              nq("Square footing width B", 2.0, "m"),
              nq("Df", 1.0, "m"), nq("phi", 32, "deg"),
              nq("Unit weight", 18.5, "kN/m3"), nq("FS", 3.0)]
    rows = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity (strip footing)", round(strip), "kPa"),
         nq("Square footing: ultimate bearing capacity", round(square),
            "kPa")], inputs, subject="Strip and square footings"))
    by_case = {r["case"]: r for r in rows}
    assert by_case["strip"]["flag"] == "agrees", by_case["strip"]
    assert by_case["strip"]["shape"].startswith("strip (printed in this "
                                                "case's labels")
    assert by_case["square"]["flag"] == "agrees", by_case["square"]


def test_the_shape_from_the_labels_when_the_words_name_several():
    strip = bearing(width=1.2, shape="strip",
                    friction_angle=32)["q_ultimate_kPa"]
    inputs = [nq("B", 1.2, "m"), nq("Df", 1.0, "m"), nq("phi", 32, "deg"),
              nq("Unit weight", 18.5, "kN/m3"), nq("FS", 3.0),
              words("Footing type", "Strip")]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", round(strip), "kPa")], inputs,
        subject="Square and strip footings"))
    assert row["flag"] == "agrees", row
    assert "this case's labels" in row["shape"]


# -- 4. gamma' alone (OWN-130) -------------------------------------------------

GP_INPUTS = [i for i in SHARED if i["name"] != "Unit weight"] + [
    nq("phi", 32, "deg"), nq("γ'", 9.0, "kN/m3")]


def test_gamma_prime_alone_is_refused_by_default_as_water_level_not_printed(
        monkeypatch):
    monkeypatch.delenv(GAMMA_PRIME_ENV, raising=False)
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", 500, "kPa")], GP_INPUTS))
    assert row["status"] == "not_recomputed"
    assert row["water_level"] == "not printed"
    assert "water level not printed" in row["reason"]
    assert row["gamma_prime_rule"].startswith("refuse")
    assert recompute_record({"calculations": []})["gamma_prime_rule"] == \
        "refuse"


def test_gamma_prime_alone_runs_as_the_submerged_case_when_set(monkeypatch):
    want = bearing(friction_angle=32, unit_weight=9.0)["q_ultimate_kPa"]
    calc = bearing_calc([nq("Ultimate bearing capacity", round(want), "kPa")],
                        GP_INPUTS)
    (row,) = recompute_calculation(calc, gamma_prime="submerged")
    assert row["flag"] == "agrees", row
    assert row["gamma_prime_rule"].startswith("submerged")
    assert "submerged case" in row["inputs_used"]["unit_weight"]["note"]
    assert "water_level" not in row
    monkeypatch.setenv(GAMMA_PRIME_ENV, "submerged")
    (row,) = recompute_calculation(calc)
    assert row["flag"] == "agrees"
    assert recompute_record({"calculations": [calc]})["gamma_prime_rule"] \
        == "submerged"


def test_gamma_prime_with_a_water_depth_below_ground_is_refused_either_way():
    for rule in ("refuse", "submerged"):
        (row,) = recompute_calculation(bearing_calc(
            [nq("Ultimate bearing capacity", 500, "kPa")],
            GP_INPUTS + [nq("Depth to groundwater", 1.0, "m")]),
            gamma_prime=rule)
        assert row["status"] == "not_recomputed", rule
        assert "unit_weight: not printed" in row["inputs_missing"]
        assert any("above the water is not printed" in m
                   for m in row["inputs_missing"])


def test_gamma_prime_beside_a_total_unit_weight_is_not_used():
    want = bearing(friction_angle=32)["q_ultimate_kPa"]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", round(want), "kPa")],
        SHARED + [nq("phi", 32, "deg"), nq("γ'", 9.0, "kN/m3")]),
        gamma_prime="submerged")
    assert row["flag"] == "agrees"
    assert "not used" in row["inputs_used"]["gamma_prime"]["note"]
    assert "gamma_prime_rule" not in row


def test_an_unknown_gamma_prime_rule_is_refused():
    with pytest.raises(ValueError, match="gamma_prime rule"):
        recompute_calculation(bearing_calc([], SHARED), gamma_prime="guess")
