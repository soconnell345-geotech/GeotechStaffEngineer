"""Offline tests of ``ags_truth``: AGS4 data as logs-stage answer sheets.

Every value here is made up and written in the test -- an AGS4 file with two
holes, and blank one- and two-page "scans" -- so no private file is read and
no model is called. What is checked is what the scoring of the public blind
set relies on: the groups parse with their units, the answer sheet fills the
logs stage's schema (and scores itself perfectly through the real scorer),
what has no slot is listed rather than dropped, a depth not in metres stops
the build, a hole missing groups still gives an answer sheet, the assembled
PDF's pages are in hole order, and the logs stage runs on a ``P`` set through
``run_claude_local`` with ``--reports-dir``.

Run::

    .venv/Scripts/python -m pytest module_work/report_ingest_harness/tests/test_ags_truth.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("python_ags4")

from module_work.report_ingest_harness import ags_truth as at  # noqa: E402

MADE_UP = '''"GROUP","PROJ"
"HEADING","PROJ_ID","PROJ_NAME","PROJ_LOC","PROJ_CLNT","PROJ_CONT","PROJ_ENG","PROJ_MEMO"
"UNIT","","","","","","",""
"TYPE","ID","X","X","X","X","X","X"
"DATA","J100","Made-up Bridge","Nowhere","A Client","A Driller","An Engineer","a memo"

"GROUP","ABBR"
"HEADING","ABBR_HDNG","ABBR_CODE","ABBR_DESC","ABBR_LIST","ABBR_REM"
"UNIT","","","","",""
"TYPE","X","X","X","X","X"
"DATA","LOCA_TYPE","CP","Cable percussion","",""
"DATA","LOCA_TYPE","TP","Trial pit","",""
"DATA","SAMP_TYPE","D","Small disturbed sample","",""

"GROUP","LOCA"
"HEADING","LOCA_ID","LOCA_TYPE","LOCA_NATE","LOCA_NATN","LOCA_GL","LOCA_FDEP","LOCA_STAR","LOCA_ENDD","LOCA_LOCZ","LOCA_REM","BGS_LOCA_ID"
"UNIT","","","m","m","m","m","yyyy-mm-dd","yyyy-mm-dd","m","",""
"TYPE","ID","PA","2DP","2DP","2DP","2DP","DT","DT","2DP","X","X"
"DATA","BH1","CP","100.00","200.00","50.00","10.00","2001-02-03","2001-02-05","","","111"
"DATA","TP1","TP","110.00","210.00","","2.00","2001-02-04","2001-02-04","40.50","Stopped on a boulder.","222"

"GROUP","GEOL"
"HEADING","LOCA_ID","GEOL_TOP","GEOL_BASE","GEOL_DESC","GEOL_LEG"
"UNIT","","m","m","",""
"TYPE","ID","2DP","2DP","X","PA"
"DATA","BH1","1.20","10.00","Firm grey CLAY.","201"
"DATA","BH1","0.00","1.20","MADE GROUND.","102"
"DATA","TP1","0.00","2.00","Soft brown SAND.","401"

"GROUP","SAMP"
"HEADING","LOCA_ID","SAMP_TOP","SAMP_REF","SAMP_TYPE","SAMP_ID","SAMP_BASE","SAMP_UBLO","SAMP_RECV","SAMP_REM"
"UNIT","","m","","","","m","","%",""
"TYPE","ID","2DP","X","PA","ID","2DP","0DP","0DP","X"
"DATA","BH1","1.00","1","D","S1","1.45","","",""
"DATA","BH1","2.00","2","U","S2","2.45","40","90",""
"DATA","BH1","3.00","3","B","S3","3.50","","",""
"DATA","BH1","5.00","4","D","S4","5.20","","","taken on refusal"

"GROUP","ISPT"
"HEADING","LOCA_ID","ISPT_TOP","ISPT_REP","ISPT_HAM"
"UNIT","","m","",""
"TYPE","ID","2DP","X","X"
"DATA","BH1","1.00","N=9 (1,1/2,2,2,3)","H7"
"DATA","BH1","3.00","N=22 (3,4/5,5,6,6)","H7"
"DATA","BH1","5.00","50 (10,15 for 50mm/30,20 for 40mm)","H7"

"GROUP","WSTG"
"HEADING","LOCA_ID","WSTG_DPTH","WSTG_DTIM","WSTG_CAS","WSTG_SEAL"
"UNIT","","m","yyyy-mm-ddThh:mm","m","m"
"TYPE","ID","2DP","DT","2DP","2DP"
"DATA","BH1","4.00","2001-02-04T10:00","3.50","6.00"

"GROUP","WSTD"
"HEADING","LOCA_ID","WSTG_DPTH","WSTD_NMIN","WSTD_POST"
"UNIT","","m","min","m"
"TYPE","ID","2DP","0DP","2DP"
"DATA","BH1","4.00","5","3.80"
"DATA","BH1","4.00","20","3.20"

"GROUP","LNMC"
"HEADING","LOCA_ID","SAMP_TOP","SAMP_REF","SAMP_TYPE","SAMP_ID","SPEC_REF","SPEC_DPTH","LNMC_MC"
"UNIT","","m","","","","","m","%"
"TYPE","ID","2DP","X","PA","ID","X","2DP","0DP"
"DATA","BH1","2.00","2","U","S2","1","2.10","24"

"GROUP","LLPL"
"HEADING","LOCA_ID","SAMP_TOP","SAMP_REF","SAMP_TYPE","SAMP_ID","SPEC_REF","SPEC_DPTH","LLPL_LL","LLPL_PL","LLPL_PI"
"UNIT","","m","","","","","m","%","%","%"
"TYPE","ID","2DP","X","PA","ID","X","2DP","0DP","0DP","0DP"
"DATA","BH1","2.00","2","U","S2","1","2.10","45","20","25"

"GROUP","GRAG"
"HEADING","LOCA_ID","SAMP_TOP","SAMP_REF","SAMP_TYPE","SAMP_ID","SPEC_REF","SPEC_DPTH","GRAG_FINE","GRAG_SAND"
"UNIT","","m","","","","","m","%","%"
"TYPE","ID","2DP","X","PA","ID","X","2DP","0DP","0DP"
"DATA","BH1","7.00","9","B","S9","1","7.00","35","60"

"GROUP","LDEN"
"HEADING","LOCA_ID","SAMP_TOP","SAMP_REF","SAMP_TYPE","SAMP_ID","SPEC_REF","SPEC_DPTH","LDEN_BDEN","LDEN_DDEN"
"UNIT","","m","","","","","m","Mg/m3","Mg/m3"
"TYPE","ID","2DP","X","PA","ID","X","2DP","2DP","2DP"
"DATA","BH1","2.00","2","U","S2","1","2.10","2.05","1.70"

"GROUP","CDIA"
"HEADING","LOCA_ID","CDIA_DPTH","CDIA_DIAM"
"UNIT","","m","mm"
"TYPE","ID","2DP","0DP"
"DATA","BH1","6.00","150"
'''

#: Only what a file must have to give an answer sheet; the depth units left
#: blank, as the BGS exports leave them on most groups.
BARE = '''"GROUP","PROJ"
"HEADING","PROJ_ID","PROJ_NAME"
"UNIT","",""
"TYPE","ID","X"
"DATA","J200","Bare Site"

"GROUP","LOCA"
"HEADING","LOCA_ID","LOCA_TYPE","LOCA_FDEP"
"UNIT","","",""
"TYPE","ID","PA","X"
"DATA","BH9","ZZ","6"

"GROUP","GEOL"
"HEADING","LOCA_ID","GEOL_TOP","GEOL_BASE","GEOL_DESC"
"UNIT","","","",""
"TYPE","ID","X","X","X"
"DATA","BH9","0","6","Stiff CLAY."
'''


def _write(tmp_path: Path, text: str, name: str = "made_up.ags") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture()
def ags(tmp_path):
    return at.read_ags(_write(tmp_path, MADE_UP))


def _sample(truth, top, kind=None):
    hits = [s for s in truth["samples"] if abs(s["top"] - top) < 1e-6
            and (kind is None or s["type"] == kind)]
    assert len(hits) == 1, hits
    return hits[0]


# -- reading ------------------------------------------------------------------

def test_the_groups_parse_with_their_units(ags):
    assert {"PROJ", "ABBR", "LOCA", "GEOL", "SAMP", "ISPT", "WSTG", "WSTD",
            "LNMC", "LLPL", "GRAG", "LDEN", "CDIA"} <= set(ags)
    assert [r["LOCA_ID"] for r in ags["LOCA"]["rows"]] == ["BH1", "TP1"]
    assert ags["GEOL"]["units"]["GEOL_TOP"] == "m"
    assert ags["LDEN"]["units"]["LDEN_DDEN"] == "Mg/m3"
    assert at.abbreviations(ags)[("LOCA_TYPE", "CP")] == "Cable percussion"


@pytest.mark.parametrize("text, n, blows, refusal, parsed", [
    ("N=12 (1,2/2,3,3,4)", 12, [1, 2, 2, 3, 3, 4], False, True),
    ("50 (4,21 for 60mm/40,10)", None, [4, 21, 40, 10], True, True),
    ("N=7", 7, [], False, True),
    ("N=0 (1,1/-,-,-,-)", 0, [1, 1], False, True),
    ("", None, [], False, False),
    ("N=5 (a,b/c)", 5, [], False, False),
])
def test_an_spt_report_gives_n_and_the_increments(text, n, blows, refusal,
                                                  parsed):
    got = at.parse_spt_report(text)
    assert (got["n"], got["blows"], got["refusal"], got["parsed"]) == (
        n, blows, refusal, parsed)


def test_a_partial_increment_is_kept_in_words():
    got = at.parse_spt_report("50 (4,21 for 60mm/40,10)")
    assert got["partial"] == ["21 blows for 60 mm"]


# -- one hole -------------------------------------------------------------------

def test_a_borehole_fills_the_logs_schema(ags):
    truth = at.truth_for_hole(ags, "BH1", item_id="P99_p0", pages=[0, 1])

    assert truth["id"] == "P99_p0" and truth["pages"] == [0, 1]
    assert truth["investigation_id"] == "BH1"
    assert truth["depth_unit"] == "m" and truth["kind"] == "boring"
    fields = truth["fields"]
    assert fields["boring_id"] == "BH1"
    assert fields["project"] == "Made-up Bridge"
    assert fields["contract_number"] == "J100"
    assert fields["client"] == "A Client"
    assert fields["contractor"] == "A Driller"
    assert fields["method"] == "Cable percussion"
    assert fields["total_depth"] == 10.0
    assert fields["ground_surface_elevation"] == 50.0
    assert (fields["easting"], fields["northing"]) == (100.0, 200.0)
    assert fields["hammer"] == "H7"
    # Recorded, never scored by default: the scorer compares text and AGS
    # writes ISO dates.
    assert "date_started" not in fields and "date_finished" not in fields
    assert fields["date_started_ags"] == "2001-02-03"

    assert [(ly["top"], ly["bottom"]) for ly in truth["layers"]] == [
        (0.0, 1.2), (1.2, 10.0)]
    assert truth["layers"][0]["description"] == "MADE GROUND."

    # The SPT sits on the small disturbed sample from its spoon ...
    spoon = _sample(truth, 1.0)
    assert spoon["id"] == "1" and spoon["n"] == 9
    assert spoon["blows"] == [1, 1, 2, 2, 2, 3]
    assert spoon["spt_report"] == "N=9 (1,1/2,2,2,3)"
    # ... but never on a bulk sample at the same depth.
    assert _sample(truth, 3.0, "B")["blows"] is None
    spt = _sample(truth, 3.0, "spt")
    assert spt["n"] == 22 and spt["bottom"] is None
    # A refusal: no N, the blows in order, the short drives in words.
    refusal = _sample(truth, 5.0)
    assert refusal["n"] is None and refusal["refusal"] is True
    assert refusal["blows"] == [10, 15, 30, 20]
    assert "15 blows for 50 mm" in refusal["note"]
    assert "taken on refusal" in refusal["note"]
    # The tube's blow count and recovery, and the laboratory values on the
    # sample they were tested on.
    tube = _sample(truth, 2.0)
    assert tube["blows"] == [40] and tube["n"] is None
    assert tube["recovery"] == 90.0
    assert (tube["wc"], tube["ll"], tube["pl"], tube["pi"]) == (
        24.0, 45.0, 20.0, 25.0)
    # A grading on a sample SAMP does not list gets a row of its own.
    graded = _sample(truth, 7.0)
    assert graded["fines"] == 35.0 and graded["id"] == "9"

    # The strike, and the level it rose to at the LAST reading.
    assert [w["depth"] for w in truth["water"]] == [3.2, 4.0]
    strike = truth["water"][1]
    assert strike["casing"] == 3.5 and strike["date"] == "2001-02-04T10:00"
    assert "20 min" in truth["water"][0]["when"]

    assert truth["units"]["GEOL_TOP"] == "m"
    assert truth["units"]["SAMP_RECV"] == "%"
    assert truth["units"]["LNMC_MC"] == "%"
    assert truth["ags"]["bgs_loca_id"] == "111"


def test_what_has_no_slot_is_listed_not_dropped(ags):
    truth = at.truth_for_hole(ags, "BH1", item_id="P99_p0", pages=[0])
    listed = {}
    for row in truth["ags_unslotted"]:
        listed.setdefault(row["group"], []).append(row)

    # Whole groups with no slot.
    assert listed["CDIA"][0]["values"] == {"CDIA_DPTH": "6.00",
                                           "CDIA_DIAM": "150"}
    assert listed["LDEN"][0]["values"]["LDEN_DDEN"] == "1.70"
    # Headings with no slot inside slotted groups.
    assert [r["values"] for r in listed["GEOL"]] == [{"GEOL_LEG": "201"},
                                                     {"GEOL_LEG": "102"}]
    assert listed["GRAG"][0]["values"] == {"GRAG_SAND": "60"}
    assert listed["WSTG"][0]["values"] == {"WSTG_SEAL": "6.00"}
    assert listed["PROJ"][0]["values"] == {"PROJ_MEMO": "a memo"}
    assert {"SAMP_ID"} <= set(listed["SAMP"][0]["values"])
    # The earlier water reading.
    assert listed["WSTD"][0]["values"]["WSTD_POST"] == "3.80"
    # Nothing of the other hole leaks in.
    assert all(row["values"].get("LOCA_ID") is None
               for row in truth["ags_unslotted"])


def test_the_answer_sheet_scores_itself_perfectly(ags):
    from module_work.report_ingest_harness.truth_records import (
        investigations_from_truth,
    )
    from report_ingest.log_scoring import score_record

    truth = json.loads(json.dumps(
        at.truth_for_hole(ags, "BH1", item_id="P99_p0", pages=[0])))
    score = score_record(truth, investigations_from_truth(truth))

    assert score.total.total > 20
    assert score.total.found == score.total.total, score.total.misses
    assert {"sample_depth", "blows", "n_value", "layer_top", "water",
            "index", "recovery", "fields"} <= set(score.scores)


def test_a_pit_with_a_missing_ground_level_and_no_samples(ags):
    truth = at.truth_for_hole(ags, "TP1", item_id="P99_p3", pages=[3])

    assert truth["kind"] == "test_pit"
    assert truth["fields"]["test_pit_id"] == "TP1"
    assert "boring_id" not in truth["fields"]
    assert truth["fields"]["ground_surface_elevation"] == 40.5
    assert "LOCA_LOCZ" in truth["units"] and "blank" in truth["units"][
        "LOCA_LOCZ"]
    assert truth["remarks"] == "Stopped on a boulder."
    assert truth["samples"] == [] and truth["water"] == []
    assert len(truth["layers"]) == 1


def test_a_bare_file_with_blank_units_still_gives_an_answer_sheet(tmp_path):
    ags = at.read_ags(_write(tmp_path, BARE))
    truth = at.truth_for_hole(ags, "BH9", item_id="P98_p0", pages=[0])

    assert truth["layers"] == [{"top": 0.0, "bottom": 6.0,
                                "description": "Stiff CLAY.", "uscs": None}]
    assert truth["samples"] == [] and truth["water"] == []
    assert truth["units"]["GEOL_TOP"].startswith("m (blank")
    # A type code the file does not spell out is kept, and not scored.
    assert truth["fields"]["loca_type"] == "ZZ"
    assert "method" not in truth["fields"]
    assert at.counts(truth)["layers"] == 1


def test_a_depth_not_in_metres_stops_the_build(tmp_path):
    text = BARE.replace('"UNIT","","","",""', '"UNIT","","ft","ft",""')
    ags = at.read_ags(_write(tmp_path, text))
    with pytest.raises(ValueError, match="ft"):
        at.truth_for_hole(ags, "BH9", item_id="P98_p0", pages=[0])


def test_the_spt_goes_on_the_sample_whose_interval_is_the_drive():
    # Two small disturbed samples at one top, in either order: the SPT
    # takes the one 0.45 m long, whatever the file's row order.
    jar = {"id": "7", "top": 6.0, "bottom": None, "type": "D"}
    spoon = {"id": "8", "top": 6.0, "bottom": 6.45, "type": "D"}
    assert at._spoon_sample([dict(jar), dict(spoon)], 6.0)["id"] == "8"
    assert at._spoon_sample([dict(spoon), dict(jar)], 6.0)["id"] == "8"
    assert at._spoon_sample([dict(jar)], 6.0)["id"] == "7"
    long_d = {"id": "9", "top": 6.0, "bottom": 7.0, "type": "D"}
    assert at._spoon_sample([long_d], 6.0) is None


def test_dates_are_scored_only_when_asked(ags):
    truth = at.truth_for_hole(ags, "BH1", item_id="P99_p0", pages=[0],
                              score_dates=True)
    assert truth["fields"]["date_started"] == "2001-02-03"
    assert truth["fields"]["date_finished"] == "2001-02-05"
    assert truth["ags"]["dates_scored"] is True


# -- the items --------------------------------------------------------------------

def _scan(path: Path, pages: int, label: str) -> None:
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for k in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"{label} sheet {k + 1}", fontsize=11)
    doc.save(str(path))
    doc.close()


@pytest.fixture()
def made_up_set(tmp_path):
    """A set folder ``P99`` with three scans: BH2 (no AGS), BH10 (no AGS,
    two pages) and BH1 (AGS location 111). The AGS file's TP1 has no scan.
    The BGS IDs run against the names, so only the natural order of the
    names gives BH1, BH2, BH10."""
    raw = tmp_path / "raw"
    folder = raw / "P99"
    (folder / "ags").mkdir(parents=True)
    _write(folder / "ags", MADE_UP, "made_up.ags")
    holes = [(3, "BH2", 1, "444"), (4, "BH10", 2, "333"), (5, "BH1", 1, "111")]
    records = []
    for bgs, name, pages, loca in holes:
        _scan(folder / f"scan_{bgs}.pdf", pages, name)
        records.append({"name": f"MADE UP SITE {name}", "bgs_id": float(bgs),
                        "ags_log_url": f"https://example.invalid/x?loca_id="
                                       f"{loca}"})
    (folder / "onshore_index_records.json").write_text(
        json.dumps(records), encoding="utf-8")
    stale = raw / "truth" / "logs" / "P99_p9.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")
    return raw


def test_build_assembles_in_hole_order_and_writes_the_sheets(made_up_set):
    import pymupdf

    raw = made_up_set
    summary = at.build(raw, ["P99"])["P99"]

    with pymupdf.open(str(raw / "assembled" / "P99.pdf")) as doc:
        assert doc.page_count == 4
    pages = json.loads((raw / "assembled" / "P99_pages.json")
                       .read_text(encoding="utf-8"))
    assert [p["hole"] for p in pages["pages"]] == ["BH1", "BH2", "BH10",
                                                   "BH10"]
    assert [p["scan_page"] for p in pages["pages"]] == [0, 0, 0, 1]
    assert pages["ags_locations_with_no_scan"][0]["loca_id"] == "TP1"

    logs = raw / "truth" / "logs"
    assert sorted(p.name for p in logs.iterdir()) == ["OPEN.txt",
                                                      "P99_p0.json"]
    truth = json.loads((logs / "P99_p0.json").read_text(encoding="utf-8"))
    assert truth["investigation_id"] == "BH1" and truth["pages"] == [0]
    assert (logs / "OPEN.txt").read_text(encoding="utf-8").strip() == "(none)"
    assert summary["pages"] == 4 and summary["scans"] == 3
    assert summary["ags_locations"] == 2 and summary["answer_sheets"] == 1
    assert summary["ags_locations_with_no_scan"] == 1
    assert summary["spt_n"] == 2 and summary["spt_blows"] == 3
    assert summary["tube_blows"] == 1 and summary["water"] == 2
    assert summary["lab"] == 5


def _second_export(text: str) -> str:
    """BH1 exported again under location 112: no ABBR rows, no TP1, and
    one more value (a strike remark) than the first export holds."""
    abbr = text.index('"GROUP","ABBR"')
    text = text[:abbr] + text[text.index('"GROUP","LOCA"'):]
    lines = [ln for ln in text.splitlines() if '"TP1"' not in ln]
    text = "\n".join(lines) + "\n"
    text = text.replace('"111"', '"112"')
    head = '"HEADING","LOCA_ID","WSTG_DPTH","WSTG_DTIM","WSTG_CAS","WSTG_SEAL"'
    text = text.replace(head, head + ',"WSTG_REM"')
    text = text.replace('"UNIT","","m","yyyy-mm-ddThh:mm","m","m"',
                        '"UNIT","","m","yyyy-mm-ddThh:mm","m","m",""')
    text = text.replace('"TYPE","ID","2DP","DT","2DP","2DP"',
                        '"TYPE","ID","2DP","DT","2DP","2DP","X"')
    return text.replace('"3.50","6.00"', '"3.50","6.00","seepage"')


def test_a_hole_exported_twice_uses_the_fuller_dataset(made_up_set):
    raw = made_up_set
    folder = raw / "P99"
    _write(folder / "ags", _second_export(MADE_UP), "5_112.ags")
    records = json.loads((folder / "onshore_index_records.json")
                         .read_text(encoding="utf-8"))
    for record in records:
        if record["bgs_id"] == 5.0:
            record["ags_log_url"] += ",https://example.invalid/x?loca_id=112"
    (folder / "onshore_index_records.json").write_text(
        json.dumps(records), encoding="utf-8")

    summary = at.build(raw, ["P99"])["P99"]

    assert summary["answer_sheets"] == 1
    assert summary["holes_with_two_datasets"] == 1
    assert summary["datasets_disagreeing"] == 0
    truth = json.loads((raw / "truth" / "logs" / "P99_p0.json")
                       .read_text(encoding="utf-8"))
    # The fuller export is used, and its codes are spelled out from the
    # other export's ABBR rows.
    assert truth["ags"]["file"] == "P99/ags/5_112.ags"
    assert truth["ags"]["other_datasets"] == ["made_up.ags"]
    assert truth["ags"]["other_datasets_agree"] is True
    assert truth["fields"]["method"] == "Cable percussion"
    assert truth["water"][1]["note"] == "seepage"


def test_the_logs_stage_runs_on_a_public_set_item(made_up_set, monkeypatch):
    """The paid command's wiring at $0: ``--raw-dir`` and ``--reports-dir``
    at the set, a scripted model, one run file, scored as blind."""
    pytest.importorskip("langchain_core")
    from langchain_core.messages import AIMessage

    from module_work.live_smoke.models import scripted_model
    from module_work.report_ingest_harness import run_claude_local as rcl
    from report_ingest import engine

    raw = made_up_set
    at.build(raw, ["P99"])
    monkeypatch.setattr(engine, "MODEL_PRICES", dict(engine.MODEL_PRICES))
    for key in [k for k in __import__("os").environ
                if k.startswith(rcl.SWITCH_PREFIX)]:
        monkeypatch.delenv(key)

    def answer(messages):
        return AIMessage(content="", tool_calls=[{
            "name": "answer", "id": f"call_{len(messages)}",
            "args": {"investigation_id": "BH1", "depth_unit": "m"}}])

    made = []

    def build(model_id, callbacks):
        fake = scripted_model(answer, usage={"input_tokens": 1000,
                                             "output_tokens": 100,
                                             "total_tokens": 1100},
                              callbacks=callbacks, model=model_id)
        made.append(fake)
        return fake

    out = raw.parent / "out"
    args = rcl.build_parser().parse_args([
        "run", "--stage", "logs", "--model", "claude-sonnet-5-5",
        "--wave", "t-wave", "--cap", "1", "--raw-dir", str(raw),
        "--reports-dir", str(raw / "assembled"),
        "--ledger", str(raw.parent / "ledger.json"), "--out-dir", str(out)])
    assert rcl.run(args, chat_builder=build) == rcl.OK

    blob = json.loads((out / "logs" / "P99_p0.json").read_text(
        encoding="utf-8"))
    assert blob["report"] == "P99" and blob["set"] == "blind"
    assert blob["pages"] == [0]
    assert not (blob.get("after") or {}).get("error")
    assert sum(len(fake.calls) for fake in made) >= 1
