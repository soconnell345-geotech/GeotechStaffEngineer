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
