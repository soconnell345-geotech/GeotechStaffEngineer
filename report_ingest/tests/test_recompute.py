"""DR-11: a report's calculations recomputed through the app's modules.

Offline, no model calls. Every record here is made up: the labels are the
kind a printout carries, the numbers are invented, and each expected
recomputed value comes from the module itself, not from this file.
"""

import json

import pytest

from funhouse_agent.dispatch import call_agent
from report_ingest.model import CALC_KINDS, ReportRecord
from report_ingest.recompute import (NOT_RECOMPUTED, TOLERANCE, find_record,
                                     make_recompute_tool,
                                     recompute_calculation, recompute_record,
                                     table_markdown)


def nq(name, value, unit="", page=4):
    return {"name": name, "value": {"value": value, "unit": unit,
                                    "prov": {"page": page, "method": "text",
                                             "confidence": 1.0}}}


def words(name, text):
    return {"name": name, "text": text}


# -- bearing capacity --------------------------------------------------------

BEARING_INPUTS = [nq("Footing width B", 2.0, "m"),
                  nq("Embedment depth Df", 1.0, "m"),
                  nq("Friction angle phi", 32, "deg"),
                  nq("Unit weight", 18.5, "kN/m3"),
                  nq("Factor of safety", 3.0)]
BEARING_PARAMS = {"width": 2.0, "depth": 1.0, "friction_angle": 32,
                  "unit_weight": 18.5, "factor_of_safety": 3.0,
                  "shape": "square"}


def module_bearing(**extra):
    return call_agent("bearing_capacity", "bearing_capacity_analysis",
                      {**BEARING_PARAMS, **extra})


def bearing_calc(results, inputs=BEARING_INPUTS, method="",
                 subject="Square footing F-9"):
    return {"kind": "shallow_foundation_bearing", "method": method,
            "subject": subject, "pages": [3, 4], "inputs": list(inputs),
            "results": list(results)}


def test_a_matching_bearing_claim_agrees_and_names_the_module_default():
    q_ult = module_bearing()["q_ultimate_kPa"]
    rows = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", round(q_ult), "kPa", page=5)]))
    (row,) = rows
    assert row["status"] == "recomputed"
    assert row["flag"] == "agrees"
    assert row["recomputed_si"] == pytest.approx(q_ult)
    assert row["claimed"] == {"printed": f"{round(q_ult)} kPa", "page": 6}
    assert row["pages"] == [4, 5]
    assert row["module"] == "bearing_capacity"
    assert row["method_used"] == "vesic"
    assert "names no method" in row["method_basis"]
    assert row["method_reference"]          # the module's own basis text
    # the module's other cited options, run and listed with their values
    assert {a["method"] for a in row["alternatives"]} == {"meyerhof",
                                                          "terzaghi"}
    meyer = module_bearing(factor_method="meyerhof")["q_ultimate_kPa"]
    alt = next(a for a in row["alternatives"] if a["method"] == "meyerhof")
    assert alt["recomputed_si"] == pytest.approx(meyer)
    assert row["inputs_used"]["width"] == {
        "value": 2.0, "label": "Footing width B", "printed": "2 m",
        "page": 5}
    assert row["shape"].startswith("square (printed")


def test_a_bearing_claim_that_differs_is_flagged():
    q_all = module_bearing()["q_allowable_kPa"]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Allowable bearing pressure", round(q_all * 0.7), "kPa")]))
    assert row["status"] == "recomputed"
    assert row["flag"] == "differs"
    assert row["difference_pct"] > 100 * TOLERANCE
    assert "gross allowable" in row["note"]


def test_a_net_allowable_is_compared_with_the_net_value():
    net = module_bearing()["q_allowable_net_kPa"]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Net allowable bearing pressure", round(net), "kPa")]))
    assert row["flag"] == "agrees"
    assert row["recomputed_si"] == pytest.approx(net)


def test_us_customary_units_are_converted_by_the_record():
    inputs = [nq("Footing Width B (ft)", 6.0, "ft"),
              nq("Footing depth", 3.0, "ft"),
              nq("phi", 30, ""),
              nq("Unit weight of soil", 120, "pcf"),
              nq("FS", 3)]
    si = call_agent("bearing_capacity", "bearing_capacity_analysis",
                    {"width": 6 * 0.3048, "depth": 3 * 0.3048,
                     "friction_angle": 30,
                     "unit_weight": 120 * 0.157087463, "shape": "square",
                     "factor_of_safety": 3})
    psf = si["q_ultimate_kPa"] / 0.04788026
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity (psf)", round(psf), "psf")],
        inputs=inputs))
    assert row["flag"] == "agrees", row
    assert abs(row["difference_pct"]) < 0.5
    # a friction angle printed without a unit is read as degrees, and said
    assert "degrees" in row["inputs_used"]["friction_angle"]["note"]


def test_a_printed_method_is_used_and_no_alternatives_are_run():
    meyer = module_bearing(factor_method="meyerhof")["q_ultimate_kPa"]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", round(meyer), "kPa")],
        method="Meyerhof (1963)"))
    assert row["method_used"] == "meyerhof"
    assert row["method_basis"].startswith("printed")
    assert row["alternatives"] == []
    assert row["flag"] == "agrees"


def test_an_unstated_method_shows_which_cited_option_a_claim_matches():
    terz = module_bearing(factor_method="terzaghi")["q_ultimate_kPa"]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", round(terz), "kPa")]))
    assert row["method_used"] == "vesic"
    assert row["flag"] == "differs"
    alt = next(a for a in row["alternatives"] if a["method"] == "terzaghi")
    assert abs(alt["difference_pct"]) < 0.5


def test_a_missing_bearing_input_is_named_never_filled_in():
    inputs = [i for i in BEARING_INPUTS if i["name"] != "Unit weight"]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", 900, "kPa")], inputs=inputs))
    assert row["status"] == "not_recomputed"
    assert any(m.startswith("unit_weight") for m in row["inputs_missing"])
    assert "recomputed_si" not in row


def test_no_strength_and_no_shape_are_both_reported():
    inputs = [nq("Width", 2.0, "m"), nq("Embedment", 1.0, "m"),
              nq("Unit weight", 18.0, "kN/m3")]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", 900, "kPa")], inputs=inputs,
        subject="Footing F-2"))
    missing = " ".join(row["inputs_missing"])
    assert "friction_angle or cohesion" in missing
    assert "footing shape" in missing


def test_a_width_printed_twice_with_different_values_is_not_chosen():
    inputs = BEARING_INPUTS + [nq("B", 2.5, "m")]
    (row,) = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", 900, "kPa")], inputs=inputs))
    assert row["status"] == "not_recomputed"
    assert any("width" in m and "more than once" in m
               for m in row["inputs_missing"])


def test_an_allowable_with_no_factor_of_safety_is_not_compared():
    inputs = [i for i in BEARING_INPUTS if i["name"] != "Factor of safety"]
    q_ult = module_bearing()["q_ultimate_kPa"]
    rows = recompute_calculation(bearing_calc(
        [nq("Ultimate bearing capacity", round(q_ult), "kPa"),
         nq("Allowable bearing capacity", 300, "kPa")], inputs=inputs))
    by_q = {r["quantity"]: r for r in rows}
    assert by_q["Ultimate bearing capacity"]["flag"] == "agrees"
    allow = by_q["Allowable bearing capacity"]
    assert allow["status"] == "not_compared"
    assert "factor of safety" in allow["reason"]


def test_a_bearing_printout_with_no_claimed_pressure_says_so():
    (row,) = recompute_calculation(bearing_calc(
        [words("Bearing check", "OK"), nq("Bearing capacity factor Nq", 23)]))
    assert row["status"] == "not_compared"
    assert "no ultimate or allowable" in row["reason"]


# -- settlement --------------------------------------------------------------

ELASTIC_INPUTS = [nq("Applied pressure q", 150, "kPa"),
                  nq("Footing width B", 2.0, "m"),
                  nq("Elastic modulus Es", 20000, "kPa")]


def module_elastic(**extra):
    return call_agent("settlement", "elastic_settlement",
                      {"q_net": 150, "B": 2.0, "Es": 20000, **extra})


def settlement_calc(results, inputs, method="", subject="Square footing"):
    return {"kind": "settlement", "method": method, "subject": subject,
            "pages": [7], "inputs": list(inputs), "results": list(results)}


def test_an_elastic_claim_agrees_with_the_method_inferred_from_its_inputs():
    se_mm = module_elastic(shape="square")["immediate_settlement_mm"]
    (row,) = recompute_calculation(settlement_calc(
        [nq("Immediate settlement", round(se_mm, 1), "mm")], ELASTIC_INPUTS))
    assert row["status"] == "recomputed", row
    assert row["flag"] == "agrees"
    assert row["module_method"] == "elastic_settlement"
    assert row["method_used"] == "elastic"
    assert "names no method" in row["method_basis"]
    assert "a modulus and no compression index" in row["method_basis"]
    # what the module took for what the printout did not give
    assert row["assumed"]["nu"] == 0.3
    assert "Iw" in row["assumed"]


def test_an_elastic_claim_that_differs_is_flagged():
    se_in = module_elastic(Iw=0.9)["immediate_settlement_m"] / 0.0254
    (row,) = recompute_calculation(settlement_calc(
        [nq("Total settlement", round(se_in * 1.5, 2), "in")],
        ELASTIC_INPUTS + [nq("Influence factor Iw", 0.9)],
        method="Elastic theory"))
    assert row["method_basis"] == "printed: elastic"
    assert row["flag"] == "differs"
    assert row["difference_pct"] < -100 * TOLERANCE


def test_an_elastic_claim_missing_the_modulus_is_not_recomputed():
    inputs = ELASTIC_INPUTS[:2]
    (row,) = recompute_calculation(settlement_calc(
        [nq("Immediate settlement", 15, "mm")], inputs,
        method="elastic"))
    assert row["status"] == "not_recomputed"
    assert "Es: not printed" in row["inputs_missing"]


def test_a_strip_with_no_influence_factor_is_not_filled_in():
    (row,) = recompute_calculation(settlement_calc(
        [nq("Immediate settlement", 15, "mm")], ELASTIC_INPUTS,
        subject="Continuous wall footing"))
    assert row["status"] == "not_recomputed"
    assert any(m.startswith("Iw") for m in row["inputs_missing"])


CONSOLIDATION_INPUTS = [nq("Layer thickness", 3.0, "m"),
                        nq("Depth to center of layer", 4.5, "m"),
                        nq("Initial void ratio e0", 0.95),
                        nq("Compression index Cc", 0.32),
                        nq("Recompression index Cr", 0.04),
                        nq("Initial effective stress", 65, "kPa"),
                        nq("Stress increase", 45, "kPa")]
CONSOLIDATION_LAYER = {"thickness": 3.0, "depth_to_center": 4.5, "e0": 0.95,
                       "Cc": 0.32, "Cr": 0.04, "sigma_v0": 65}


def module_consolidation(**layer):
    return call_agent("settlement", "consolidation_settlement",
                      {"layers": [{**CONSOLIDATION_LAYER, **layer}],
                       "delta_sigma": 45})


def test_a_consolidation_claim_agrees_and_says_normally_consolidated():
    sc = module_consolidation()["consolidation_settlement_mm"]
    (row,) = recompute_calculation(settlement_calc(
        [nq("Consolidation settlement", round(sc), "mm")],
        CONSOLIDATION_INPUTS, method="1-D consolidation (e-log p)"))
    assert row["status"] == "recomputed", row
    assert row["flag"] == "agrees"
    assert row["module_method"] == "consolidation_settlement"
    assert "normally consolidated" in row["assumed"]["sigma_p"]
    assert row["inputs_used"]["Cr"]["value"] == 0.04
    assert row["inputs_used"]["Cc"]["value"] == 0.32


def test_a_printed_preconsolidation_pressure_is_used():
    sc = module_consolidation(sigma_p=90)["consolidation_settlement_mm"]
    (row,) = recompute_calculation(settlement_calc(
        [nq("Primary consolidation settlement", round(sc, 1), "mm")],
        CONSOLIDATION_INPUTS + [nq("Preconsolidation pressure", 90, "kPa")]))
    assert row["flag"] == "agrees", row
    assert "sigma_p" not in row["assumed"]
    assert "compression index and no modulus" in row["method_basis"]


def test_a_consolidation_claim_missing_cr_is_not_recomputed():
    inputs = [i for i in CONSOLIDATION_INPUTS
              if i["name"] != "Recompression index Cr"]
    (row,) = recompute_calculation(settlement_calc(
        [nq("Consolidation settlement", 120, "mm")], inputs,
        method="consolidation"))
    assert row["status"] == "not_recomputed"
    assert "Cr: not printed" in row["inputs_missing"]


def test_an_unstated_method_whose_inputs_fit_both_is_not_recomputed():
    (row,) = recompute_calculation(settlement_calc(
        [nq("Settlement", 40, "mm")],
        CONSOLIDATION_INPUTS + [nq("Elastic modulus Es", 20000, "kPa")]))
    assert row["status"] == "not_recomputed"
    assert "both elastic and consolidation" in row["reason"]


def test_schmertmann_is_left_for_a_later_slice():
    (row,) = recompute_calculation(settlement_calc(
        [nq("Settlement", 20, "mm")], ELASTIC_INPUTS,
        method="Schmertmann (1978)"))
    assert row["status"] == "not_recomputed"
    assert "Schmertmann" in row["reason"] and "later slice" in row["reason"]


def test_two_settlements_printed_are_not_guessed_between():
    (row,) = recompute_calculation(settlement_calc(
        [nq("Settlement at center", 20, "mm"),
         nq("Settlement at corner", 10, "mm")], ELASTIC_INPUTS))
    assert row["status"] == "not_compared"
    assert "more than one settlement" in row["reason"]


# -- the kinds not recomputed -------------------------------------------------

@pytest.mark.parametrize("kind", [k for k in CALC_KINDS if k not in (
    "shallow_foundation_bearing", "settlement")])
def test_every_other_kind_is_a_row_saying_why(kind):
    (row,) = recompute_calculation({
        "kind": kind, "pages": [10], "results": [nq("FS", 1.4)]})
    assert row["status"] == "not_recomputed"
    assert row["reason"] == NOT_RECOMPUTED[kind]
    assert row["pages"] == [11]


def test_slope_stability_is_held():
    (row,) = recompute_calculation({"kind": "slope_stability",
                                    "results": [nq("Factor of safety", 1.5)]})
    assert "held" in row["reason"] and "not recomputed yet" in row["reason"]


# -- the record, the table and the tool ---------------------------------------

def record_dict():
    q_ult = module_bearing()["q_ultimate_kPa"]
    return {"document": {"report_id": "R-TEST-1"}, "calculations": [
        bearing_calc([nq("Ultimate bearing capacity", round(q_ult), "kPa"),
                      nq("Allowable bearing pressure", 50, "kPa")]),
        {"kind": "slope_stability", "pages": [20],
         "results": [nq("FS", 1.3)]},
        settlement_calc([nq("Immediate settlement", 15, "mm")],
                        ELASTIC_INPUTS[:2], method="elastic"),
    ]}


def test_a_record_gives_one_table_with_counts():
    table = recompute_record(record_dict())
    assert table["report_id"] == "R-TEST-1"
    assert table["tolerance"] == TOLERANCE
    assert [r["calc_index"] for r in table["rows"]] == [0, 0, 1, 2]
    c = table["counts"]
    assert c["calculations"] == 3 and c["rows"] == 4
    assert c["recomputed"] == 2 and c["not_recomputed"] == 2
    assert c["agrees"] == 1 and c["differs"] == 1
    md = table_markdown(table)
    assert "R-TEST-1" in md and md.count("\n| ") == 5   # header + 4 rows
    assert "held" in md


def test_the_tolerance_is_the_caller_s():
    q_ult = module_bearing()["q_ultimate_kPa"]
    rec = {"calculations": [bearing_calc(
        [nq("Ultimate bearing capacity", round(q_ult * 1.05), "kPa")])]}
    assert recompute_record(rec)["rows"][0]["flag"] == "agrees"
    assert recompute_record(rec, tolerance=0.02)["rows"][0]["flag"] == \
        "differs"


def test_a_record_file_and_its_folder_both_load(tmp_path):
    folder = tmp_path / "R-TEST-1"
    folder.mkdir()
    rec = ReportRecord.model_validate(record_dict())
    (folder / "report.record.json").write_text(rec.model_dump_json(),
                                               encoding="utf-8")
    assert recompute_record(str(folder))["counts"]["rows"] == 4
    assert recompute_record(str(folder / "report.record.json"))[
        "counts"]["rows"] == 4


def test_the_tool_finds_the_one_record_in_the_working_folder(tmp_path):
    from funhouse_agent._fileio import working_dir_bound
    tool = make_recompute_tool()
    assert tool.name == "recompute_report_calcs"
    with working_dir_bound(str(tmp_path)):
        assert "report_ingest first" in json.loads(tool.invoke({}))["error"]
        sub = tmp_path / "ingest" / "R-TEST-1"
        sub.mkdir(parents=True)
        rec = ReportRecord.model_validate(record_dict())
        (sub / "report.record.json").write_text(rec.model_dump_json(),
                                                encoding="utf-8")
        out = json.loads(tool.invoke({}))
        assert out["counts"]["rows"] == 4
        named = json.loads(tool.invoke({"record": "ingest/R-TEST-1"}))
        assert named["report_id"] == "R-TEST-1"
        assert find_record("ingest/R-TEST-1").endswith("report.record.json")
        (tmp_path / "other").mkdir()
        (tmp_path / "other" / "report.record.json").write_text(
            rec.model_dump_json(), encoding="utf-8")
        assert "2 records" in json.loads(tool.invoke({}))["error"]


def test_a_small_cap_keeps_the_flagged_rows_first():
    from report_ingest.recompute import _fit
    text = _fit(recompute_record(record_dict()), 1500)
    assert len(text) <= 1500
    shown = json.loads(text)
    assert shown["rows"][0].get("flag") == "differs"
    assert "flagged first" in shown["cut"]


def test_the_tool_comes_and_goes_with_report_ingest(monkeypatch):
    pytest.importorskip("deepagents")
    pytest.importorskip("planlens.document.roles")
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage

    from funhouse_agent.deep.agent import build_deep_agent

    def names(agent):
        return set(agent.nodes["tools"].bound.tools_by_name)

    model = FakeMessagesListChatModel(responses=[AIMessage(content="ok")])
    monkeypatch.delenv("GEOTECH_REPORT_INGEST", raising=False)
    off = names(build_deep_agent(model))
    assert "recompute_report_calcs" not in off
    on = names(build_deep_agent(model, enable_report_ingest=True))
    assert {"report_ingest", "recompute_report_calcs"} <= on
