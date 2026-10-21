"""``subsurface.write_diggs``: DIGGS 2.6 from data the agent has read.

Field session 2026-10-06: asked for DIGGS from a report's logs, the agent had
no DIGGS writer, typed XML into ``save_file`` under the DIGGS 2.6 namespace
and called it "a best-effort partial DIGGS XML" -- a file that fails the
DIGGS schema at its root element. These tests drive the new method through
the dispatch layer the agent uses, with synthetic logs.
"""

import json
import os

import pytest

pytest.importorskip("pydantic")

from funhouse_agent.dispatch import call_agent, describe_method, list_methods


def _m(v, unit="m"):
    return {"value": v, "unit": unit}


BORING = {
    "investigation_id": "B-1", "kind": "boring", "depth_unit": "m",
    "elevation": _m(4.6), "total_depth": _m(10.05),
    "date_started": "2026-01-28", "date_finished": "2026-01-29",
    "drilling": {"method": "mud rotary", "hammer_type": "automatic"},
    "layers": [
        {"top": _m(0.0), "bottom": _m(1.5), "description": "Gray poorly "
         "graded GRAVEL, loose, dry (fill)", "uscs": "GP"},
        {"top": _m(1.5), "bottom": _m(10.05), "description": "Brown "
         "poorly graded SAND, loose to dense, wet", "uscs": "SP"},
    ],
    "samples": [{"sample_id": "S-1", "top": _m(1.0), "bottom": _m(1.45),
                 "kind": "spt"},
                {"sample_id": "S-2", "top": _m(3.0), "bottom": _m(3.45),
                 "kind": "spt", "water_content": 18.0}],
    "spt": [{"depth_top": _m(1.0), "depth_bottom": _m(1.45),
             "blows": [2, 1, 2], "n": 3, "sample_id": "S-1"},
            {"depth_top": _m(3.0), "depth_bottom": _m(3.45),
             "blows": [7, 10, 8], "n": 18, "sample_id": "S-2"}],
    "water": [{"depth": _m(3.0), "when": "while_drilling"}],
    "pages": [21, 22],
}

LAB = [
    {"kind": "atterberg", "investigation_id": "B-1", "sample_id": "S-2",
     "depth_top": _m(3.0), "result": {"kind": "atterberg", "ll": 32.0,
                                      "pl": 18.0, "pi": 14.0}},
    {"kind": "gradation", "investigation_id": "B-1", "sample_id": "S-2",
     "depth_top": _m(3.0), "result": {"kind": "gradation",
                                      "gravel_percent": 1.0,
                                      "sand_percent": 96.0,
                                      "fines_percent": 3.0}},
]


def _j(value):
    """The dispatch layer answers with a dict (the tool wrapper serialises
    it); accept either."""
    return json.loads(value) if isinstance(value, str) else value


def _call(params):
    return _j(call_agent("subsurface", "write_diggs", params))


def test_the_method_is_on_the_map():
    methods = _j(list_methods("subsurface"))
    assert "write_diggs" in json.dumps(methods)
    desc = json.dumps(_j(describe_method("subsurface", "write_diggs")))
    assert len(desc) < 8000                    # reaches the model whole
    assert "never hand-write DIGGS XML" in desc


def test_a_log_and_its_lab_become_checked_diggs(tmp_path):
    out = tmp_path / "site.diggs.xml"
    res = _call({"investigations": [BORING], "lab_tests": LAB,
                 "project": {"name": "Synthetic site", "number": "000"},
                 "output_path": str(out)})
    assert "error" not in res, res
    assert res["file_exists"] and out.stat().st_size > 1000
    assert res["read_back"]["equal"], res["read_back"]
    if res["schema_check"]["checked"]:
        assert res["schema_check"]["valid"], res["schema_check"]
        assert res["verdict"].startswith("valid DIGGS 2.6")
    assert res["written"]["investigations"] == 1
    assert res["written"]["lab_tests"] == 2
    # and the app's own reader takes it back
    back = _j(call_agent("subsurface", "parse_diggs",
                         {"file_path": str(out)}))
    assert back["n_investigations"] == 1
    assert back["investigations"][0]["investigation_id"] == "B-1"


def test_investigations_may_arrive_as_json_text(tmp_path):
    out = tmp_path / "a.xml"
    res = _call({"investigations": json.dumps([BORING]),
                 "output_path": str(out)})
    assert "error" not in res and res["read_back"]["equal"]


def test_data_that_does_not_fit_writes_nothing_and_names_the_fields(tmp_path):
    bad = dict(BORING)
    bad["layers"] = [{"top": 0.0, "description": "no unit"}]   # not {value}
    bad["colour"] = "brown"                                     # unknown key
    out = tmp_path / "bad.xml"
    res = _call({"investigations": [bad], "output_path": str(out)})
    assert "NO file was written" in res["error"]
    assert not out.exists()
    text = " ".join(res["problems"])
    assert "investigations[0].layers[0].top" in text
    assert "investigations[0].colour" in text


def test_text_in_n_that_is_not_a_refusal_still_names_the_field(tmp_path):
    # 2026-10-14 on Funhouse: an agent put a printed ">50" in n, was refused
    # 17 times and concluded refusal could not be written. A refusal notation
    # is now accepted (below); text that is NOT one is still refused, and the
    # problem line names the field and what n takes.
    bad = json.loads(json.dumps(BORING))
    bad["spt"][1]["n"] = "WOH"
    res = _call({"investigations": [bad],
                 "output_path": str(tmp_path / "r.xml")})
    assert "NO file was written" in res["error"]
    assert not (tmp_path / "r.xml").exists()
    text = " ".join(res["problems"])
    assert "investigations[0].spt[1].n" in text
    assert "'WOH' is not one" in text and "leave n out" in text


@pytest.mark.parametrize("n", [-3, 12.5, "-3", "N/A", "twelve"])
def test_an_n_that_is_neither_a_count_nor_a_refusal_is_refused(tmp_path, n):
    bad = json.loads(json.dumps(BORING))
    bad["spt"][1]["n"] = n
    res = _call({"investigations": [bad],
                 "output_path": str(tmp_path / "r.xml")})
    assert "NO file was written" in res["error"], res
    assert "investigations[0].spt[1].n" in " ".join(res["problems"])


def test_a_refusal_written_as_the_hint_says_is_accepted(tmp_path):
    good = json.loads(json.dumps(BORING))
    good["spt"][1].update({"blows": [20, 30, '50/5"'], "refusal": True})
    good["spt"][1].pop("n")
    out = tmp_path / "r.xml"
    res = _call({"investigations": [good], "output_path": str(out)})
    assert "error" not in res, res
    assert res["file_exists"] and res["read_back"]["equal"], res["read_back"]


def test_a_schema_check_that_could_not_run_is_neither_valid_nor_invalid(
        tmp_path, monkeypatch):
    """RI-20: with no validator on the host the schema check does not run.
    The answer says so -- unchecked, valid unknown -- and never "valid"."""
    from subsurface_characterization.formats import diggs_validation
    monkeypatch.setattr(diggs_validation, "has_schema_check",
                        lambda *_a, **_k: False)
    res = _call({"investigations": [BORING], "lab_tests": LAB,
                 "project": {"name": "Synthetic site", "number": "000"},
                 "output_path": str(tmp_path / "site.diggs.xml")})
    assert "error" not in res, res
    check = res["schema_check"]
    assert check["checked"] is False and check["valid"] is None
    assert any("not installed" in e for e in check["errors"])
    assert "did not run" in res["verdict"]
    assert not res["verdict"].startswith("valid DIGGS")
    assert "do not call it valid" in res["verdict"]


def test_a_water_analysis_in_mg_per_litre_is_written_without_a_note(
        tmp_path):
    """RI-20: mg/L had no conversion, so every water analysis came back with
    an "unconverted" cross-check though DIGGS takes it as printed. A note
    fires only where a conversion was needed and failed."""
    chem = {"kind": "chemical", "investigation_id": "B-1",
            "sample_id": "S-2", "depth_top": _m(3.0),
            "result": {"kind": "chemical", "pH": {"value": 7.6, "unit": ""},
                       "sulfate": {"value": 250.0, "unit": "mg/L"},
                       "chloride": {"value": 80.0, "unit": "mg/l"},
                       "sulfides": {"value": 3.0, "unit": "meq/L"}}}
    out = tmp_path / "site.diggs.xml"
    res = _call({"investigations": [BORING], "lab_tests": LAB + [chem],
                 "project": {"name": "Synthetic site", "number": "000"},
                 "output_path": str(out)})
    assert "error" not in res, res
    if res["schema_check"]["checked"]:
        assert res["schema_check"]["valid"], res["schema_check"]
    assert res["read_back"]["equal"], res["read_back"]
    unconverted = [e for e in res["cross_checks"].get("entries", [])
                   if e.get("kind") == "unconverted"]
    said = json.dumps(unconverted)
    assert "mg/L" not in said and "mg/l" not in said
    # what needed a conversion and had none is still named
    assert "meq/L" in said
    xml = out.read_text(encoding="utf-8")
    assert ">mg/L<" in xml


def test_hand_typed_xml_is_not_diggs():
    """What the agent wrote in the field session, in shape: invented
    elements under the DIGGS namespace. The schema says so."""
    from report_ingest.diggs_writer import diggs_schema_gate
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<DIGGS xmlns="http://diggsml.org/schemas/2.6.a">'
           '<investigations><investigation id="B-1"/></investigations>'
           '</DIGGS>')
    ok, errors = diggs_schema_gate(xml)
    if ok is None:
        pytest.skip("no DIGGS schema check on this host")
    assert ok is False


def test_unknown_parameter_is_refused():
    res = _call({"investigations": [BORING], "output_path": "/tmp/x.xml",
                 "format": "diggs"})
    assert "error" in res and "format" in res["error"]


# -- the reconciler's cross-checks (owner, 2026-10-08) ----------------------
# write_diggs joins the data up the way report ingest does before writing:
# lab tests linked to their boring and sample, a summary table compared with
# the sheets. It never changes a value; every disagreement is reported.

SUMMARY = {
    "kind": "summary_table", "pages": [61],
    "result": {"kind": "summary_table", "rows": [
        {"investigation_id": "B-1", "sample_id": "S-2", "depth_top": _m(3.0),
         "ll": 18.0, "pl": 32.0, "pi": 14.0}]},
}


def test_clean_data_raises_no_cross_check(tmp_path):
    res = _call({"investigations": [BORING], "lab_tests": LAB,
                 "output_path": str(tmp_path / "a.xml")})
    assert res["cross_checks"]["n"] == 0, res["cross_checks"]
    assert "cross-checks raised" not in res["verdict"]


def test_a_summary_table_that_disagrees_with_its_sheet_is_reported(tmp_path):
    res = _call({"investigations": [BORING], "lab_tests": LAB + [SUMMARY],
                 "output_path": str(tmp_path / "b.xml")})
    cc = res["cross_checks"]
    conflicts = [e for e in cc["entries"] if e["kind"] == "conflict"]
    fields = {e["where"].split(" ")[0] for e in conflicts}
    assert {"lab_tests.ll", "lab_tests.pl"} <= fields, cc
    # both values and both pages are kept; nothing was resolved
    ll = next(e for e in conflicts if e["where"].startswith("lab_tests.ll"))
    assert any("18" in v for v in ll["values"])
    assert any("32" in v for v in ll["values"])
    assert 61 in ll["pages"]
    assert "cross-checks raised" in res["verdict"]
    # the file itself is still written from the data as given
    assert res["file_exists"] is True


def test_a_lab_test_naming_an_unknown_boring_is_reported(tmp_path):
    orphan = dict(LAB[0], investigation_id="B-9")
    res = _call({"investigations": [BORING], "lab_tests": [orphan],
                 "output_path": str(tmp_path / "c.xml")})
    text = json.dumps(res["cross_checks"])
    assert res["cross_checks"]["n"] >= 1
    assert "B-9" in text


# -- N9 (Foundry brief 5): the kind list and the code agree -----------------
# The description listed the kinds without summary_table, then said a summary
# table goes in "as one object of kind summary_table". GPT-5.4 filed it as
# kind 'other', was refused without being told which kind to use, and wrote
# the file with no summary at all -- so the planted LL/PL swap was never
# compared.

def test_the_described_kinds_are_the_models_kinds():
    import re
    from typing import get_args
    from report_ingest.model import LabKind

    desc = _j(describe_method("subsurface", "write_diggs"))
    text = json.dumps(desc)
    listed = re.search(r"kind \(atterberg\|([a-z_|]+)\)", text)
    assert listed, "the lab_tests kind list is gone from the description"
    kinds = {"atterberg", *listed.group(1).split("|")}
    assert kinds == set(get_args(LabKind))
    assert "summary_table" in kinds


def test_a_summary_filed_under_other_is_refused_with_the_kind_to_use(tmp_path):
    """The brief-5 call shape: a summary table's rows under kind 'other'."""
    wrong = dict(SUMMARY, kind="other")
    res = _call({"investigations": [BORING], "lab_tests": LAB + [wrong],
                 "output_path": str(tmp_path / "s.xml")})
    assert "error" in res and not (tmp_path / "s.xml").exists()
    problem = " ".join(res["problems"])
    assert "give the test kind 'summary_table'" in problem, problem


def test_a_summary_under_its_own_kind_is_compared(tmp_path):
    res = _call({"investigations": [BORING], "lab_tests": LAB + [SUMMARY],
                 "output_path": str(tmp_path / "t.xml")})
    assert "error" not in res and res["file_exists"] is True
    assert any(e["kind"] == "conflict"
               for e in res["cross_checks"]["entries"])


# -- RI-27 (field session 2026-10-14): silent values and refusal rows -------
# Three defects, each read in full in the session's TRACE_REVIEW (2.4, 4,
# P2.2): a printed ">50" given as n refused a 17-row set whole; "not
# encountered" given with the hole's depth came back as water at the bottom
# of every dry hole; and a position nobody printed was written, and read
# back, as 0 -- 33 of 40 explorations at elevation 0. Every case goes
# through the dispatch layer the agent uses and back through parse_diggs.

def _parse(path):
    return _j(call_agent("subsurface", "parse_diggs",
                         {"file_path": str(path)}))


def _back(path):
    """``{investigation_id: summary}`` as parse_diggs hands them back."""
    return {inv["investigation_id"]: inv
            for inv in _parse(path)["investigations"]}


def _refusal_set(n_rows=17):
    """A set whose every row prints '50 - >50' for the drives, N '> 50'."""
    return {
        "investigation_id": "P-1", "kind": "boring", "depth_unit": "m",
        "x": 10.0, "y": 20.0, "elevation": _m(100.0),
        "total_depth": _m(1.5 * n_rows + 1.0),
        "layers": [{"top": _m(0.0), "bottom": _m(1.5 * n_rows + 1.0),
                    "description": "GRAVEL"}],
        "spt": [{"depth_top": _m(1.5 * (k + 1)), "blows": [50, "-", ">50"],
                 "n": "> 50", "refusal": True} for k in range(n_rows)],
    }


def test_a_set_of_printed_refusals_is_written_not_refused(tmp_path):
    out = tmp_path / "p.diggs.xml"
    located = dict(json.loads(json.dumps(BORING)), x=1.0, y=2.0)
    res = _call({"investigations": [located, _refusal_set()],
                 "output_path": str(out)})
    assert "error" not in res, res
    assert res["file_exists"] and res["read_back"]["equal"], res["read_back"]
    if res["schema_check"]["checked"]:
        assert res["schema_check"]["valid"], res["schema_check"]
    assert res["written"]["spt"] == 2 + 17
    # every normalisation is named, and the count survives the 10-row cut
    assert res["written"]["n_normalised"] == 17
    assert len(res["written"]["normalised"]) == 10
    assert res["written"]["normalised"][0].startswith(
        "P-1 spt[0] at 1.5 m: N printed as '> 50'")
    assert "17 value(s) were recorded differently" in res["verdict"]
    # the printed text is in the file, under the N value, as text; no N
    # number was invented for it
    from subsurface_characterization.diggs26 import parse_diggs26_result_sets
    sets = [rs for rs in parse_diggs26_result_sets(
        content=out.read_text(encoding="utf-8"))
        if rs.investigation_id == "P-1"
        and rs.procedure == "DrivenPenetrationTest"]
    assert len(sets) == 17
    assert all("> 50" in rs.values("n_value") for rs in sets)
    assert "N_spt" not in _back(out)["P-1"]["parameters"]


@pytest.mark.parametrize("printed", [
    ">50", "> 50", ">=50", "50+", "R", "Ref.", "Refusal", "refused", "50R",
    '50/5"', "50/10cm", "100/0.1m", "50 for 4in"])
def test_each_refusal_notation_is_accepted_and_kept(tmp_path, printed):
    hole = json.loads(json.dumps(BORING))
    hole["spt"][1] = {"depth_top": _m(3.0), "blows": [12], "n": printed}
    out = tmp_path / "r.xml"
    res = _call({"investigations": [hole], "output_path": str(out)})
    assert "error" not in res, res
    assert res["read_back"]["equal"], res["read_back"]
    assert len(res["written"]["normalised"]) == 1
    assert "which was not set" in res["written"]["normalised"][0]
    assert printed in out.read_text(encoding="utf-8").replace("&gt;", ">")


def test_a_refusal_with_no_countable_drive_is_still_in_the_file(tmp_path):
    hole = json.loads(json.dumps(BORING))
    hole["spt"][1] = {"depth_top": _m(3.0), "n": "Refusal"}
    out = tmp_path / "r.xml"
    res = _call({"investigations": [hole], "output_path": str(out)})
    assert "error" not in res and res["read_back"]["equal"], res
    assert res["written"]["spt"] == 2
    assert "Refusal" in out.read_text(encoding="utf-8")


def test_empty_text_in_n_is_no_printed_n_and_is_said(tmp_path):
    hole = json.loads(json.dumps(BORING))
    hole["spt"][1]["n"] = " "
    res = _call({"investigations": [hole],
                 "output_path": str(tmp_path / "e.xml")})
    assert "error" not in res and res["read_back"]["equal"], res
    assert "n was empty text" in res["written"]["normalised"][0]


def _dry(kind="boring", **water):
    return {"investigation_id": "B-7", "kind": kind, "depth_unit": "m",
            "x": 50.0, "y": 60.0, "elevation": _m(876.0),
            "total_depth": _m(12.0),
            "layers": [{"top": _m(0.0), "bottom": _m(12.0),
                        "description": "SILTY SAND", "uscs": "SM"}],
            "water": [dict({"when": "not_encountered"}, **water)]}


@pytest.mark.parametrize("kind", ["boring", "test_pit"])
@pytest.mark.parametrize("with_depth", [True, False])
def test_not_encountered_is_never_a_water_level(tmp_path, kind, with_depth):
    hole = _dry(kind, **({"depth": _m(12.0)} if with_depth else {}))
    out = tmp_path / "dry.xml"
    res = _call({"investigations": [hole], "output_path": str(out)})
    assert "error" not in res, res
    assert res["read_back"]["equal"], res["read_back"]
    if res["schema_check"]["checked"]:
        assert res["schema_check"]["valid"], res["schema_check"]
    xml = out.read_text(encoding="utf-8")
    assert "water_depth" not in xml and "WaterStrikeReading" not in xml
    if kind == "boring":
        assert "<notEncountered>true</notEncountered>" in xml
    else:
        assert "Groundwater not encountered" in xml
    back = _back(out)["B-7"]
    assert back["gwl_depth_m"] is None
    assert back["gwl_not_encountered"] is True
    said = " ".join(res["written"]["normalised"])
    assert ("is NOT a water level" in said) is with_depth


def test_a_reading_beside_not_encountered_is_the_reading(tmp_path):
    hole = _dry(depth=_m(12.0))
    hole["water"].append({"depth": _m(5.2), "when": "after_hours",
                          "hours": 24})
    out = tmp_path / "w.xml"
    res = _call({"investigations": [hole], "output_path": str(out)})
    assert res["read_back"]["equal"], res["read_back"]
    back = _back(out)["B-7"]
    assert back["gwl_depth_m"] == pytest.approx(5.2)
    assert back["gwl_not_encountered"] is False
    assert any("one or the other" in s for s in res["written"]["skipped"])


def test_the_gate_catches_not_encountered_read_back_as_a_level():
    """What the 2026-10-14 file held: water at the bottom of a dry hole. The
    gate compared depths only and passed it; it must not."""
    from report_ingest.diggs_writer import diggs_roundtrip_gate, write_diggs
    from report_ingest.model import Investigation
    dry = Investigation.model_validate(_dry(depth=_m(12.0)))
    wet = dry.model_copy(deep=True)
    wet.water[0].when = "at_completion"          # the old writer's reading
    ok, diffs = diggs_roundtrip_gate(write_diggs([wet]), [dry])
    assert not ok
    assert any("came back where 'groundwater not encountered' went in" in d
               for d in diffs), diffs


def test_a_position_nobody_gave_is_unknown_not_zero(tmp_path):
    hole = {k: v for k, v in json.loads(json.dumps(BORING)).items()
            if k != "elevation"}
    hole.update({"x": 120.5, "y": -45.25})
    pit = {"investigation_id": "TP-1", "kind": "test_pit",
           "depth_unit": "m", "total_depth": _m(1.5),
           "layers": [{"top": _m(0.0), "bottom": _m(1.5),
                       "description": "FILL"}]}
    out = tmp_path / "pos.xml"
    res = _call({"investigations": [hole, pit], "output_path": str(out)})
    assert "error" not in res, res
    assert res["read_back"]["equal"], res["read_back"]
    if res["schema_check"]["checked"]:
        assert res["schema_check"]["valid"], res["schema_check"]
    assert res["written"]["position_unknown"] == ["B-1", "TP-1"]
    assumed = " ".join(res["written"]["assumed"])
    assert "B-1: no elevation given" in assumed
    assert "TP-1: no x, y and elevation given" in assumed
    assert "2 exploration(s) had no x, y and/or elevation" in res["verdict"]
    xml = out.read_text(encoding="utf-8")
    assert "120.5 -45.25 NaN" in xml and "NaN NaN NaN" in xml
    assert ">0 0 0<" not in xml
    back = _back(out)
    assert (back["B-1"]["x"], back["B-1"]["y"]) == (120.5, -45.25)
    assert back["B-1"]["elevation_m"] is None
    assert back["TP-1"]["x"] is None and back["TP-1"]["y"] is None
    assert back["TP-1"]["elevation_m"] is None


def test_a_given_position_is_not_flagged(tmp_path):
    hole = dict(json.loads(json.dumps(BORING)), x=1.0, y=2.0)
    res = _call({"investigations": [hole],
                 "output_path": str(tmp_path / "ok.xml")})
    assert res["written"]["position_unknown"] == []
    assert res["written"]["assumed"] == []
    assert "had no x, y" not in res["verdict"]


def test_the_gate_catches_a_zero_where_no_position_was_given():
    """The writer of 2026-10-14 filled 0, 0, 0 and the gate passed it."""
    from report_ingest.diggs_writer import diggs_roundtrip_gate, write_diggs
    from report_ingest.model import Investigation, Quantity
    unknown = Investigation.model_validate(
        {k: v for k, v in BORING.items() if k != "elevation"})
    zeroed = unknown.model_copy(deep=True)
    zeroed.x, zeroed.y = 0.0, 0.0
    zeroed.elevation = Quantity(value=0.0, unit="m")
    ok, diffs = diggs_roundtrip_gate(write_diggs([zeroed]), [unknown])
    assert not ok
    joined = " ".join(diffs)
    assert "x came back as 0.0 although none was given" in joined
    assert "ground elevation came back as 0.0 m although none" in joined


def test_plots_leave_out_what_they_cannot_place(tmp_path):
    """An unknown position read back is drawn nowhere -- not at 0, 0 -- and
    each plot says what it left out."""
    placed = dict(json.loads(json.dumps(BORING)), x=0.0, y=0.0)
    other = dict(json.loads(json.dumps(BORING)), investigation_id="B-2",
                 x=30.0, y=40.0)
    lost = {k: v for k, v in json.loads(json.dumps(BORING)).items()
            if k != "elevation"}
    lost["investigation_id"] = "B-9"
    out = tmp_path / "plot.xml"
    res = _call({"investigations": [placed, other, lost],
                 "output_path": str(out)})
    assert res["read_back"]["equal"], res["read_back"]
    key = _parse(out)["site_key"]
    plan = _j(call_agent("subsurface", "plot_plan_view",
                         {"site_key": key, "output_format": "metadata"}))
    assert plan["n_investigations"] == 2
    assert any(s.startswith("B-9: no x, y") for s in plan["skipped"])
    section = _j(call_agent("subsurface", "plot_cross_section", {
        "site_key": key, "investigation_ids": ["B-1", "B-2", "B-9"],
        "output_format": "metadata"}))
    assert section["n_investigations"] == 2
    assert any(s.startswith("B-9:") for s in section["skipped"])
    depth = _j(call_agent("subsurface", "plot_parameter_vs_depth", {
        "site_key": key, "parameter": "N_spt", "use_elevation": True,
        "output_format": "metadata"}))
    assert depth["n_investigations"] == 2
    assert any(s.startswith("B-9: no ground elevation")
               for s in depth["skipped"])


# -- RI-28 (field session 2026-10-14): depths off the drawn scale, and the
# log checked against itself ---------------------------------------------
# 54 of 60 SPT records were held back as "not printed" because each sample's
# depths are a box drawn against the log's depth scale; a refusal printed
# `60/15"` in a 0.15 m interval became a 0.381 m drive; a drive record lost a
# drive; a boring's layers ran a metre under its total depth. The file
# carried all of it with no cross-check entry (TRACE_REVIEW P3.2-P3.3).

def _drives(*records, total=10.05):
    """BORING with these driven records, each (top, bottom, blows, extra)."""
    hole = json.loads(json.dumps(BORING))
    hole["total_depth"] = _m(total)
    hole["spt"] = []
    for top, bottom, blows, *extra in records:
        row = {"depth_top": _m(top), "depth_bottom": _m(bottom),
               "blows": blows}
        if extra:
            row.update(extra[0])
        hole["spt"].append(row)
    return hole


def _entries(res, kind):
    return [e for e in res["cross_checks"]["entries"] if e["kind"] == kind]


def test_the_description_says_a_depth_off_the_drawn_scale_is_read():
    desc = json.dumps(_j(describe_method("subsurface", "write_diggs")))
    assert "AS PRINTED on the log (never inferred)" not in desc
    assert "drawn depth scale" in desc and "read from the log" in desc
    assert "never hold back a record whose values you have" in desc
    assert "name each thing left out" in desc
    # the fields a log prints that had nowhere to go
    for key in ("increment {value, unit}", "recovery {value, unit}",
                "rqd_percent", "sample_id (may be empty)"):
        assert key in desc, key
    assert "below the total depth" in desc        # the cross-checks named


def test_a_depth_read_off_the_scale_goes_in_with_its_reading(tmp_path):
    """The reading is noted in a prov on the value (or the hole) and the
    file is written and reads back like any other."""
    hole = json.loads(json.dumps(BORING))
    note = {"page": 21, "method": "grid",
            "note": "sample depths off the depth scale +/-0.01 m"}
    for s in hole["samples"]:
        s["top"]["prov"] = dict(note)
        s["bottom"]["prov"] = dict(note)
    hole["prov"] = [dict(note)]
    res = _call({"investigations": [hole],
                 "output_path": str(tmp_path / "grid.xml")})
    assert "error" not in res, res
    assert res["read_back"]["equal"], res["read_back"]
    assert res["cross_checks"]["n"] == 0, res["cross_checks"]


def test_a_refusal_read_in_the_wrong_unit_is_reported_not_refused(tmp_path):
    # printed 60/15" beside "SS 3.5 to 3.65 m"; written 60/15in = 0.381 m
    hole = _drives((1.0, 1.45, [2, 1, 2]), (3.5, 3.65, ["60/15in"],
                                            {"refusal": True}))
    res = _call({"investigations": [hole],
                 "output_path": str(tmp_path / "r.xml")})
    assert res["file_exists"] is True and "error" not in res
    (e,) = _entries(res, "conflict")
    assert "3.50 m" in e["detail"] and "0.381 m" in e["detail"]
    assert "0.150 m" in e["detail"]
    assert "drives 0.381 m" in e["values"]
    assert "cross-checks raised" in res["verdict"]
    # given in the form's own unit it agrees
    ok = _drives((1.0, 1.45, [2, 1, 2]), (3.5, 3.65, ["60/15cm"],
                                          {"refusal": True}))
    res = _call({"investigations": [ok],
                 "output_path": str(tmp_path / "r2.xml")})
    assert res["cross_checks"]["n"] == 0, res["cross_checks"]


def test_a_dropped_drive_is_reported_by_sum_and_by_count(tmp_path):
    # the form drives three 15 cm increments; one record gives two
    hole = _drives((1.05, 1.50, [12, 14, 16]), (2.55, 3.0, [10, 17, 15]),
                   (4.05, 4.50, [3, 4]), (5.55, 6.0, [20, 33, 39]))
    for row in hole["spt"]:
        row["increment"] = _m(0.15)
    res = _call({"investigations": [hole],
                 "output_path": str(tmp_path / "d.xml")})
    (count,) = _entries(res, "count_mismatch")
    assert "4.05 m" in count["detail"] and "give 3" in count["detail"]
    (total,) = _entries(res, "conflict")
    assert "0.300 m" in total["detail"] and "0.450 m" in total["detail"]


def test_one_wrong_assumption_is_one_entry(tmp_path):
    """Four records of four 20 cm drives each, all given as plain counts (6 in
    assumed: 0.610 m) in 0.80 m intervals: one entry naming every depth."""
    hole = _drives(*[(t, t + 0.80, [5, 6, 7, 8])
                     for t in (1.0, 2.5, 4.0, 5.5)])
    res = _call({"investigations": [hole],
                 "output_path": str(tmp_path / "g.xml")})
    (e,) = _entries(res, "conflict")
    assert "4 driven records" in e["detail"]
    assert "1.00, 2.50, 4.00, 5.50" in e["detail"]
    assert "0.610 m" in e["detail"] and "0.800 m" in e["detail"]
    assert "0.1524 m (6 in)" in e["detail"]


def test_anything_below_the_total_depth_is_reported(tmp_path):
    hole = json.loads(json.dumps(BORING))
    hole["total_depth"] = _m(9.0)            # the layers run to 10.05 m
    hole["water"] = [{"depth": _m(9.5), "when": "at_completion"}]
    res = _call({"investigations": [hole],
                 "output_path": str(tmp_path / "td.xml")})
    assert res["file_exists"] is True
    (e,) = _entries(res, "out_of_range")
    assert "below its total depth of 9.00 m" in e["detail"]
    assert "layer 2 bottom 10.05 m" in e["detail"]
    assert "water (at_completion) 9.50 m" in e["detail"]
    assert e["pages"] == [21, 22]


# -- RI-31 (field session 2026-10-14, turn 3): records handed over as FILES -
# Helpers typed their records into their replies and the primary retyped
# 29,000 characters of them into write_diggs; a blow count, a bottom depth, a
# printed 60/15" and one exploration's coordinates went wrong on the way
# (TRACE_REVIEW P3.3, P3.6). A helper now saves its records as a JSON file and
# write_diggs reads the file, confined to the conversation's working folder.

from funhouse_agent._fileio import working_dir_bound  # noqa: E402


def _hole(iid, **kw):
    hole = json.loads(json.dumps(BORING))
    hole["investigation_id"] = iid
    hole.update(kw)
    return hole


def _save(folder, name, data):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data),
                    encoding="utf-8")
    return path


def test_records_from_files_are_written_as_inline_ones_are(tmp_path):
    _save(tmp_path, "logs.json", [BORING])
    _save(tmp_path, "lab.json", LAB)
    with working_dir_bound(str(tmp_path)):
        res = _call({"investigations_file": "logs.json",
                     "lab_tests_file": "lab.json",
                     "output_path": "f.diggs.xml"})
        inline = _call({"investigations": [BORING], "lab_tests": LAB,
                        "output_path": "i.diggs.xml"})
    assert "error" not in res, res
    assert res["read_back"]["equal"], res["read_back"]
    if res["schema_check"]["checked"]:
        assert res["schema_check"]["valid"], res["schema_check"]
    assert res["written"] == inline["written"]
    assert res["cross_checks"]["n"] == inline["cross_checks"]["n"] == 0
    assert res["sources"] == [
        {"from": "logs.json", "investigations": ["B-1"], "lab_tests": 0},
        {"from": "lab.json", "investigations": [], "lab_tests": 2}]
    assert "sources" not in inline            # inline-only: as before
    assert (tmp_path / "f.diggs.xml").is_file()


def test_files_and_inline_combine_and_each_source_is_named(tmp_path):
    _save(tmp_path, "helper_a.json", [_hole("B-2", x=10.0, y=0.0)])
    _save(tmp_path, "sub/helper_b.json", {"investigations": [
        _hole("B-3", x=20.0, y=0.0), _hole("B-4", x=30.0, y=0.0)]})
    with working_dir_bound(str(tmp_path)):
        res = _call({"investigations": [_hole("B-1", x=0.0, y=0.0)],
                     # a list of files, as models send it: JSON text
                     "investigations_file":
                         '["helper_a.json", "sub/helper_b.json"]',
                     "output_path": "all.xml"})
    assert "error" not in res, res
    assert res["written"]["investigations"] == 4
    assert res["read_back"]["equal"], res["read_back"]
    assert {s["from"]: s["investigations"] for s in res["sources"]} == {
        "inline": ["B-1"], "helper_a.json": ["B-2"],
        "sub/helper_b.json": ["B-3", "B-4"]}
    back = _back(tmp_path / "all.xml")
    assert sorted(back) == ["B-1", "B-2", "B-3", "B-4"]


@pytest.mark.parametrize("differ", [False, True])
def test_one_exploration_in_two_inputs_is_refused_naming_both(tmp_path,
                                                              differ):
    other = _hole("B-1", x=5.0) if differ else BORING
    _save(tmp_path, "dup.json", [_hole("B-2"), other])
    with working_dir_bound(str(tmp_path)):
        res = _call({"investigations": [BORING],
                     "investigations_file": "dup.json",
                     "output_path": "d.xml"})
    assert "NO file was written" in res["error"], res
    assert not (tmp_path / "d.xml").exists()
    (line,) = [p for p in res["problems"] if "'B-1'" in p]
    assert "investigations[0]" in line
    assert "'dup.json' investigations[1]" in line
    assert ("the records differ" if differ
            else "the same record each time") in line
    assert "give it once" in res["hint"]


def test_two_files_with_one_exploration_are_refused(tmp_path):
    _save(tmp_path, "a.json", [BORING])
    _save(tmp_path, "b.json", [BORING])
    with working_dir_bound(str(tmp_path)):
        res = _call({"investigations_file": ["a.json", "b.json"],
                     "output_path": "ab.xml"})
    assert "NO file was written" in res["error"]
    assert any("'a.json' investigations[0]" in p
               and "'b.json' investigations[0]" in p
               for p in res["problems"]), res["problems"]


def test_a_files_records_get_the_same_key_checks(tmp_path):
    bad = dict(BORING)
    bad["layers"] = [{"top": 0.0, "description": "no unit"}]
    bad["colour"] = "brown"
    _save(tmp_path, "bad.json", [bad])
    with working_dir_bound(str(tmp_path)):
        res = _call({"investigations_file": "bad.json",
                     "output_path": "bad.xml"})
    assert "NO file was written" in res["error"]
    assert not (tmp_path / "bad.xml").exists()
    text = " ".join(res["problems"])
    assert "'bad.json' investigations[0].colour" in text
    assert "'bad.json' investigations[0].layers[0].top" in text
    assert "save it again with save_file" in res["hint"]


def test_a_file_outside_the_conversation_is_never_read(tmp_path):
    conv, elsewhere = tmp_path / "conv", tmp_path / "other"
    _save(conv, "mine.json", [BORING])
    outside = _save(elsewhere, "theirs.json", [_hole("X-1")])
    for name in (str(outside), "../other/theirs.json"):
        with working_dir_bound(str(conv)):
            res = _call({"investigations_file": name,
                         "output_path": "o.xml"})
        assert "NO file was written" in res["error"], res
        assert "outside this conversation's files" in res["problems"][0]
        assert "mine.json" in res["problems"][0]      # what is here, by name
    assert not (conv / "o.xml").exists()


def test_a_missing_file_names_the_files_here(tmp_path):
    _save(tmp_path, "logs_2026.json", [BORING])
    with working_dir_bound(str(tmp_path)):
        res = _call({"investigations_file": "logs.json",
                     "output_path": "m.xml"})
    (line,) = res["problems"]
    assert "no file 'logs.json'" in line and "logs_2026.json" in line
    assert "save it with save_file" in line


def test_wrapped_and_fenced_files_are_read(tmp_path):
    _save(tmp_path, "fenced.json",
          "```json\n" + json.dumps([_hole("B-5")], indent=1) + "\n```\n")
    _save(tmp_path, "one.json", _hole("B-6"))           # a single object
    with working_dir_bound(str(tmp_path)):
        res = _call({"investigations_file": ["fenced.json", "one.json"],
                     "output_path": "w.xml"})
    assert "error" not in res, res
    assert [s["investigations"] for s in res["sources"]] == [["B-5"],
                                                             ["B-6"]]


def test_a_file_holding_lab_tests_too_is_read_for_both_or_refused(tmp_path):
    _save(tmp_path, "both.json", {"investigations": [BORING],
                                  "lab_tests": LAB})
    with working_dir_bound(str(tmp_path)):
        half = _call({"investigations_file": "both.json",
                      "output_path": "h.xml"})
        whole = _call({"investigations_file": "both.json",
                       "lab_tests_file": "both.json",
                       "output_path": "b.xml"})
    assert "NO file was written" in half["error"]
    assert "also holds 'lab_tests'" in half["problems"][0]
    assert "give the file as lab_tests_file too" in half["problems"][0]
    assert "error" not in whole, whole
    assert whole["written"]["lab_tests"] == 2
    assert whole["sources"] == [{"from": "both.json",
                                 "investigations": ["B-1"], "lab_tests": 2}]


def test_files_that_cannot_be_used_are_named(tmp_path):
    _save(tmp_path, "notes.md", "# What a helper read\nB-1 | 3.0 m | N=17")
    _save(tmp_path, "odd.json", {"investigations": [BORING], "meta": 1})
    _save(tmp_path, "logs.json", [BORING])
    with working_dir_bound(str(tmp_path)):
        res = _call({"investigations_file": ["notes.md", "odd.json",
                                             "logs.json", "logs.json"],
                     "output_path": "u.xml"})
    assert "NO file was written" in res["error"]
    text = " ".join(res["problems"])
    assert "'notes.md': not valid JSON" in text
    assert "'odd.json': unknown key(s) ['meta']" in text
    assert "'logs.json' is named twice" in text
    assert not (tmp_path / "u.xml").exists()


def test_no_records_at_all_names_both_ways_to_give_them():
    res = _call({"output_path": "/tmp/none.xml"})
    assert "investigations_file" in res["error"]
    assert "'investigations'" in res["error"]


def test_the_description_says_how_a_helper_hands_records_over():
    desc = _j(describe_method("subsurface", "write_diggs"))
    text = json.dumps(desc)
    assert len(text) < 8000
    assert desc["parameters"]["investigations"]["required"] is False
    hand = desc["parameters"]["investigations_file"]["description"]
    assert "save_file" in hand and "never retype" in hand
    assert "name the file in its reply" in hand
    assert "lab_tests_file" in desc["parameters"]
    assert "sources" in desc["returns"]


def test_a_helpers_saved_files_are_named_to_the_primary():
    """The helper side in code: whatever its reply says, the files it saved
    reach the primary by name (keep_work.saved_note) -- and a stopped
    helper's summary names them too."""
    from funhouse_agent.deep import keep_work
    rec = keep_work.HelperRecord("t1", "general-purpose", "B-1 to JSON")
    rec.add_result("save_file", "c1", json.dumps({"saved": "b1.json",
                                                  "file_exists": True}))
    rec.add_result("save_file", "c2", json.dumps({"error": "disk full",
                                                  "saved": "no.json"}))
    rec.add_result("read_document", "c3", "x" * 100)
    assert keep_work.saved_files(rec) == ["b1.json"]
    assert "Files it saved: b1.json." in keep_work.summary_paragraph(
        rec, "it stopped")
    note = keep_work.saved_note(["b1.json"])
    assert "'b1.json'" in note and "investigations_file" in note


def test_a_helpers_file_reaches_write_diggs_through_the_agent(tmp_path):
    """End to end through a real compiled deep agent: the helper saves its
    records with save_file and answers in one line that names no file; the
    task result names the file (in code); the primary passes that name to
    write_diggs, which writes checked DIGGS from it."""
    pytest.importorskip("deepagents")
    import re

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    from funhouse_agent.deep.agent import build_deep_agent

    class Relay(BaseChatModel):
        @property
        def _llm_type(self):
            return "relay"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            first = next(m.content for m in messages
                         if isinstance(m, HumanMessage))
            step = sum(1 for m in messages if isinstance(m, AIMessage))
            if first.startswith("PRIMARY"):
                if step == 0:
                    reply = AIMessage(content="", tool_calls=[{
                        "name": "task", "id": "task_0",
                        "args": {"description": "HELPER: B-1 as a JSON file",
                                 "subagent_type": "general-purpose"}}])
                elif step == 1:
                    told = [m for m in messages if isinstance(m, ToolMessage)
                            ][-1].content
                    name = re.search(r"saved, in the working folder: '([^']+)'",
                                     told).group(1)
                    reply = AIMessage(content="", tool_calls=[{
                        "name": "call_agent", "id": "w1",
                        "args": {"agent_name": "subsurface",
                                 "method": "write_diggs",
                                 "parameters": {
                                     "investigations_file": name,
                                     "output_path": "site.diggs.xml"}}}])
                else:
                    reply = AIMessage(content="Done.")
            elif step == 0:
                reply = AIMessage(content="", tool_calls=[{
                    "name": "save_file", "id": "s1",
                    "args": {"path": "b1_records.json",
                             "content": json.dumps([BORING])}}])
            else:
                reply = AIMessage(content="Saved the records.")
            return ChatResult(generations=[ChatGeneration(message=reply)])

    agent = build_deep_agent(model=Relay(), reference_mode="off")
    with working_dir_bound(str(tmp_path)):
        state = agent.invoke({"messages": [{"role": "user",
                                            "content": "PRIMARY go"}]},
                             config={"recursion_limit": 40})
    results = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    task = next(m for m in results if m.tool_call_id == "task_0")
    assert task.content.startswith("Saved the records.")
    assert "'b1_records.json'" in task.content
    wrote = json.loads(next(m for m in results
                            if m.tool_call_id == "w1").content)
    assert "error" not in wrote, wrote
    assert wrote["read_back"]["equal"], wrote["read_back"]
    assert wrote["sources"] == [{"from": "b1_records.json",
                                 "investigations": ["B-1"], "lab_tests": 0}]
    assert (tmp_path / "site.diggs.xml").is_file()
