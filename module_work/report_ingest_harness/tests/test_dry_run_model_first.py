"""Offline test of ``dry_run_model_first``: both arms on a made-up report.

A made-up report, a made-up answer sheet and a made-up saved reading stand
in for the private corpus and the plain baseline's run files. The dry run
replays the reading in both arms through ``score_on_cluster``; what is
checked is that both arms read the item, each run file names its arm, and
the before and after columns agree, since both arms were handed the same
reading and are scored by the same scorer.

Run::

    .venv/Scripts/python -m pytest module_work/report_ingest_harness/tests/test_dry_run_model_first.py -q
"""

from __future__ import annotations

import json

import pytest

pymupdf = pytest.importorskip("pymupdf")

from module_work.report_ingest_harness import dry_run_model_first as dry  # noqa: E402
from module_work.report_ingest_harness import run_claude_local as rcl  # noqa: E402

ITEM = "settlement__R01_p2"

#: Three made-up pages; the calculation sits on page 2.
PAGES = (
    ("BEARING CAPACITY CHECK", "Footing width B = 1.7 m"),
    ("A PAGE OF NOTHING IN PARTICULAR",),
    ("SETTLEMENT CALCULATION", "Footing width B = 2.0 m",
     "Settlement S = 25 mm"),
)


@pytest.fixture()
def made_up(tmp_path):
    raw = tmp_path / "raw"
    (raw / "corpus").mkdir(parents=True)
    doc = pymupdf.open()
    for lines in PAGES:
        page = doc.new_page()
        for n, line in enumerate(lines):
            page.insert_text((72, 72 + 18 * n), line, fontsize=11)
    doc.save(str(raw / "corpus" / "R01.pdf"))
    doc.close()
    truth = raw / "truth" / "calc"
    truth.mkdir(parents=True)
    (truth / f"{ITEM}.json").write_text(json.dumps({
        "id": ITEM, "report": "R01", "kind": "settlement", "pages": [2],
        "program": None, "method": "", "subject": "",
        "inputs": [{"name": "Footing width B", "value": 2.0, "unit": "m",
                    "page": 2}],
        "results": [{"name": "Settlement S", "value": 25.0, "unit": "mm",
                     "page": 2}]}), encoding="utf-8")
    saved = tmp_path / "saved" / "plain_calc_opus" / "calc"
    saved.mkdir(parents=True)
    (saved / f"{ITEM}.json").write_text(json.dumps({"reading": {
        "kind": "settlement",
        "inputs": [{"name": "Footing width B", "value": 2.0, "unit": "m",
                    "prov": {"page": 2}}],
        "results": [{"name": "Settlement S", "value": 25.0, "unit": "mm",
                     "prov": {"page": 2}}]}}), encoding="utf-8")
    return tmp_path


def test_both_arms_read_the_item_and_agree_on_one_reading(made_up, capsys):
    pytest.importorskip("report_ingest.locate")
    code = dry.main(["--stages", "calc", "--items", "1",
                     "--raw-dir", str(made_up / "raw"),
                     "--saved-dir", str(made_up / "saved"),
                     "--out-dir", str(made_up / "out")])
    assert code == rcl.OK
    off = json.loads((made_up / "out" / "off" / "calc" / f"{ITEM}.json")
                     .read_text(encoding="utf-8"))
    on = json.loads((made_up / "out" / "on" / "calc" / f"{ITEM}.json")
                    .read_text(encoding="utf-8"))
    assert off["model_first"] == "off" and on["model_first"] == "on"
    assert on["route"] == "pdf" and on["placement"]["at_a_spot"] == 2
    assert off["before"]["overall"] == on["before"]["overall"]
    assert off["after"]["overall"] == on["after"]["overall"]
    assert on["after"]["overall"]["found"] == on["after"]["overall"]["total"]
    printed = capsys.readouterr().out
    assert "before column the same in both arms: True" in printed
    assert "(no model called)" in printed


def test_a_call_with_no_saved_reading_gets_no_answer():
    from report_ingest.calc_reader import CalcReading

    engine = dry.ReplayEngine("replay", {})
    engine.stage, engine.item = "calc", ITEM
    reply = engine.complete([{"role": "user", "content": "made up"}],
                            output_format=CalcReading)
    assert reply.parsed is None and len(engine.calls) == 1


def test_every_report_s_narrative_call_is_sized_without_a_call(made_up,
                                                               capsys):
    """RI-175: ``--size-narratives`` sizes the model-first narrative call
    of every report in the corpus folder, sent text first, and writes the
    counts; no engine is called (there is none)."""
    pytest.importorskip("planlens.document.roles")
    out = made_up / "out"
    code = dry.main(["--size-narratives", "--max-request-tokens-in",
                     "100000", "--raw-dir", str(made_up / "raw"),
                     "--out-dir", str(out)])
    assert code == rcl.OK
    sizes = json.loads((out / "narrative_sizes.json").read_text(
        encoding="utf-8"))
    assert (sizes["route"], sizes["limit"]) == ("pictures", 100000)
    (row,) = sizes["reports"]
    assert row["id"] == "R01" and row["calls"] == 1
    assert row["sent"] == row["chosen"] == 3 and row["left_out"] == 0
    assert row["text_pages"] + row["pictures"] == 3
    assert 0 < row["estimated_tokens_in"][0] <= 100000
    assert not row["over_limit"] and row["problem"] == []
    printed = capsys.readouterr().out
    assert "1 of 1 report(s) in one call within the limit" in printed
    assert not (out / "on").exists()
