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


def test_hand_typed_xml_is_not_diggs():
    """What the agent wrote in the field session, in shape: invented
    elements under the DIGGS namespace. The schema says so."""
    from report_ingest.diggs_writer import diggs_schema_gate
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<DIGGS xmlns="http://diggsml.org/schemas/2.6.a">'
           '<investigations><investigation id="B-1"/></investigations>'
           '</DIGGS>')
    ok, errors = diggs_schema_gate(xml)
    if any("not installed" in e for e in errors):
        pytest.skip("pydiggs not installed")
    assert not ok


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
