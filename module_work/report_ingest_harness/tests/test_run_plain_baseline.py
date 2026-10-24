"""Offline tests of ``run_plain_baseline``: the wiring, at $0.

A made-up report and made-up truth files stand in for the private corpus,
and a scripted chat model (``live_smoke.models``) stands in for Claude: no
key, no network. What is checked is what the paid run relies on -- the
item's own pages are cut out in order, they reach the model as ONE PDF
document block beside one short instruction and the answer tool, the answer
is built into a record and scored by the stage's own scorer, every call
lands on the meter and the ledger at the model's rates, the cap refuses the
next call and leaves that item to be read again, and RESULTS.md sets the
plain arm beside a pipeline arm.

Run::

    .venv/Scripts/python -m pytest module_work/report_ingest_harness/tests/test_run_plain_baseline.py -q
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")
pymupdf = pytest.importorskip("pymupdf")

from module_work.live_smoke.models import scripted_model  # noqa: E402
from module_work.report_ingest_harness import run_claude_local as rcl  # noqa: E402
from module_work.report_ingest_harness import run_plain_baseline as rpb  # noqa: E402
from report_ingest import engine  # noqa: E402

MODEL = "claude-opus-5-5"
USAGE = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}
#: One call at the Opus 5.5 rates on file: 1,000 x $4 + 100 x $20 a million.
PER_CALL = 0.006

#: Three made-up pages; the calculation items sit on pages 2 and 0.
PAGES = (
    ("BEARING CAPACITY CHECK", "Footing width B = 1.7 m",
     "Allowable bearing pressure qa = 173 kPa"),
    ("A PAGE OF NOTHING IN PARTICULAR",),
    ("SETTLEMENT CALCULATION", "Footing width B = 2.0 m",
     "Settlement S = 25 mm"),
)
ITEMS = ("settlement__R01_p2", "shallow_foundation_bearing__R01_p0")


@pytest.fixture()
def made_up(tmp_path, monkeypatch):
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
    (truth / f"{ITEMS[0]}.json").write_text(json.dumps({
        "id": ITEMS[0], "report": "R01", "kind": "settlement", "pages": [2],
        "program": None, "method": "", "subject": "",
        "inputs": [{"name": "Footing width B", "value": 2.0, "unit": "m",
                    "page": 2}],
        "results": [{"name": "Settlement S", "value": 25.0, "unit": "mm",
                     "page": 2}]}), encoding="utf-8")
    (truth / f"{ITEMS[1]}.json").write_text(json.dumps({
        "id": ITEMS[1], "report": "R01",
        "kind": "shallow_foundation_bearing", "pages": [0],
        "program": None, "method": "", "subject": "",
        "inputs": [{"name": "Footing width B", "value": 1.7, "unit": "m",
                    "page": 0}],
        "results": [{"name": "Allowable bearing pressure qa", "value": 173.0,
                     "unit": "kPa", "page": 0}]}), encoding="utf-8")
    (truth / "OPEN.txt").write_text("R01\n", encoding="utf-8")

    # A pipeline arm's run file for one item, as run_claude_local writes it.
    arm = tmp_path / "pipeline" / "calc_opus" / "calc"
    arm.mkdir(parents=True)
    (arm / f"{ITEMS[0]}.json").write_text(json.dumps({
        "set": "open", "page_sources": {"text": 1},
        "after": {"overall": {"found": 1, "total": 3},
                  "model_alone": {"overall": {"found": 0, "total": 3}}}}),
        encoding="utf-8")
    monkeypatch.setattr(engine, "MODEL_PRICES", dict(engine.MODEL_PRICES))
    return tmp_path


def _args(tmp: Path, *extra: str, cap: str = "1.0", wave: str = "t-plain"):
    return rpb.build_parser().parse_args([
        "run", "--stage", "calc", "--model", MODEL, "--wave", wave,
        "--cap", cap, "--raw-dir", str(tmp / "raw"),
        "--ledger", str(tmp / "ledger.json"), "--out-dir", str(tmp / "out"),
        "--pipeline-dir", str(tmp / "pipeline"), *extra])


def _answer(messages):
    """The answer tool called with a made-up reading: the settlement item's
    two values when the PDF holds the settlement page, else the bearing
    item's."""
    from langchain_core.messages import AIMessage

    said = json.dumps([m.content for m in messages], default=str)
    settlement = "page 2" in said
    args = ({"kind": "settlement",
             "inputs": [{"name": "Footing width B", "value": 2.0, "unit": "m",
                         "prov": {"page": 2}}],
             "results": [{"name": "Settlement S", "value": 25.0,
                          "unit": "mm", "prov": {"page": 2}}]}
            if settlement else
            {"kind": "shallow_foundation_bearing",
             "inputs": [{"name": "Footing width B", "value": 1.7, "unit": "m",
                         "prov": {"page": 0}}],
             "results": []})
    return AIMessage(content="", tool_calls=[{
        "name": "answer", "id": f"call_{len(messages)}", "args": args}])


def _builder(made: list, seen: list):
    def build(model_id, callbacks):
        def policy(messages):
            seen.append(messages)
            return _answer(messages)
        fake = scripted_model(policy, usage=dict(USAGE), callbacks=callbacks,
                              model=model_id)
        made.append(fake)
        return fake
    return build


def _ledger(tmp: Path) -> dict:
    return json.loads((tmp / "ledger.json").read_text(encoding="utf-8"))


def _text_of(pdf: bytes) -> list:
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return [page.get_text() for page in doc]


def test_the_item_pages_are_cut_out_in_order(made_up):
    source = made_up / "raw" / "corpus" / "R01.pdf"
    data, n, sanitized = rpb.cut_pdf(source, [2, 0])
    assert n == 2 and not sanitized
    text = _text_of(data)
    assert "SETTLEMENT CALCULATION" in text[0]
    assert "BEARING CAPACITY CHECK" in text[1]
    whole, n_all, _ = rpb.cut_pdf(source, None)
    assert n_all == 3 and whole == source.read_bytes()
    with pytest.raises(ValueError):
        rpb.cut_pdf(source, [3])


def test_a_cut_too_big_to_send_is_sanitized(made_up, monkeypatch):
    source = made_up / "raw" / "corpus" / "R01.pdf"
    monkeypatch.setattr(rpb, "SANITIZE_ABOVE", 10)
    data, n, sanitized = rpb.cut_pdf(source, [0])
    assert sanitized and n == 1
    assert "BEARING CAPACITY CHECK" in _text_of(data)[0]


def test_the_pdf_reaches_the_api_as_one_document_block(made_up):
    from langchain_anthropic.chat_models import _format_messages
    from report_ingest.calc_reader import CalcReading
    from report_ingest.engine import CostMeter, text_block, user

    seen: list = []
    fake = scripted_model(lambda m: (seen.append(m), _answer(m))[1],
                          usage=dict(USAGE), model=MODEL)
    eng = rpb._engine_class()(fake, meter=CostMeter())
    data, _n, _s = rpb.cut_pdf(made_up / "raw" / "corpus" / "R01.pdf", [2])
    said = rpb.instruction("calc", "R01", [2])
    reply = eng.complete([user(rpb.pdf_block(data), text_block(said))],
                         output_format=CalcReading)

    assert reply.parsed is not None and reply.parsed.kind == "settlement"
    _system, formatted = _format_messages(seen[0])
    blocks = formatted[0]["content"]
    docs = [b for b in blocks if b.get("type") == "document"]
    assert len(docs) == 1 and blocks[0] is docs[0]
    assert docs[0]["source"]["media_type"] == "application/pdf"
    import base64
    assert base64.b64decode(docs[0]["source"]["data"]) == data
    assert not any(b.get("type") == "image" for b in blocks)
    text = [b["text"] for b in blocks if b.get("type") == "text"]
    assert len(text) == 1 and "PDF page 1 is page 2" in text[0]
    # The instruction names the kinds and nothing of the pipeline's brief.
    assert "settlement" in text[0] and "STARTING RECORD" not in text[0]
    # The system turn is the engine's own note to answer with the tool.
    assert "`answer` tool" in str(_system)


def test_a_log_answer_keeps_every_depth_and_is_scored_the_stage_s_way():
    from report_ingest.log_reader import LogReading

    reading = LogReading.model_validate({
        "investigation_id": "B-1", "depth_unit": "ft",
        "layers": [{"top": 0.0, "bottom": 4.0, "description": "made-up clay",
                    "uscs": "CL", "prov": {"page": 7}},
                   {"top": 4.0, "description": "made-up sand", "uscs": "SP",
                    "prov": {"page": 7}}],
        "spt": [{"depth_top": 5.0, "blows": ["3", "4", "5"], "n": 9,
                 "prov": {"page": 7}}],
        "water": [{"depth": 6.0, "when": "while_drilling",
                   "prov": {"page": 7}}]})
    truth = {"id": "R99_p7", "pages": [7], "depth_unit": "ft",
             "layers": [{"top": 0.0, "uscs": "CL"}, {"top": 4.0,
                                                     "uscs": "SP"}],
             "samples": [{"top": 5.0, "blows": [3, 4, 5], "n": 9}],
             "water": [{"depth": 6.0}]}
    after = rpb.build_and_score("logs", truth, reading, [7], "R99")
    assert after["error"] is None
    assert after["overall"]["found"] == after["overall"]["total"] > 0
    assert after["record"][0]["depth_unit"] == "ft"
    assert len(after["record"][0]["layers"]) == 2


def test_the_stage_runs_every_call_is_metered_and_results_are_written(made_up):
    made: list = []
    seen: list = []
    code = rpb.run(_args(made_up), chat_builder=_builder(made, seen))

    assert code == rcl.OK
    out = made_up / "out"
    assert sorted(p.stem for p in (out / "calc").glob("*.json")) == \
        sorted(ITEMS)
    sent = sum(len(f.calls) for f in made)
    assert sent == len(ITEMS)                    # ONE call an item
    wave = _ledger(made_up)["waves"]["t-plain"]
    assert wave["calls"] == sent
    assert wave["spent_usd"] == pytest.approx(sent * PER_CALL)
    assert wave["scenarios"]["plain_calc"]["calls"] == sent

    calls = [json.loads(line) for line in (out / "spend_calls.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    assert sorted(c["item"] for c in calls) == sorted(ITEMS)
    assert all(c["pdf_pages"] == 1 and c["model"] == MODEL for c in calls)

    blob = json.loads((out / "calc" / f"{ITEMS[0]}.json")
                      .read_text(encoding="utf-8"))
    assert blob["pages"] == [2] and blob["pdf_pages"] == 1
    assert blob["set"] == "open"
    # kind, program (none printed, none said), the input and the result.
    assert (blob["after"]["overall"]["found"],
            blob["after"]["overall"]["total"]) == (4, 4)
    assert blob["cost"]["calls"] == 1
    assert blob["cost"]["dollars"] == pytest.approx(PER_CALL)
    other = json.loads((out / "calc" / f"{ITEMS[1]}.json")
                       .read_text(encoding="utf-8"))
    assert other["after"]["scores"]["results"]["found"] == 0

    text = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "# The plain baseline: calc" in text
    # The pipeline arm's item sits beside the plain one, re-scored where it
    # can be and as saved where it cannot.
    row = next(line for line in text.splitlines()
               if line.startswith(f"| {ITEMS[0]} |"))
    assert "4/4" in row and "1/3" in row and "0/3" in row and "text 1" in row
    assert "## The instruction" in text

    # The same command again reads nothing: every item has its run file.
    made2: list = []
    assert rpb.run(_args(made_up), chat_builder=_builder(made2, [])) == rcl.OK
    assert sum(len(f.calls) for f in made2) == 0


def test_the_cap_refuses_the_next_item_and_a_raised_cap_reads_it(made_up):
    made: list = []
    code = rpb.run(_args(made_up, cap=str(PER_CALL * 0.5)),
                   chat_builder=_builder(made, []))
    assert code == rcl.STOPPED_BY_CAP
    out = made_up / "out"
    done = sorted(p.stem for p in (out / "calc").glob("*.json"))
    assert len(done) == 1                         # the first; none past it
    assert sum(len(f.calls) for f in made) == 1
    assert _ledger(made_up)["waves"]["t-plain"]["stopped_by_cap"]

    made2: list = []
    code = rpb.run(_args(made_up, cap="1.0"), chat_builder=_builder(made2, []))
    assert code == rcl.OK
    assert sorted(p.stem for p in (out / "calc").glob("*.json")) == \
        sorted(ITEMS)
    assert sum(len(f.calls) for f in made2) == 1


def test_no_item_starts_past_the_time_limit(made_up):
    made: list = []
    code = rpb.run(_args(made_up, "--time-limit", "-1"),
                   chat_builder=_builder(made, []))
    assert code == rcl.OK
    assert sum(len(f.calls) for f in made) == 0
    assert not list((made_up / "out" / "calc").glob("*.json"))


def test_an_out_dir_git_would_track_is_refused(made_up):
    tracked = rcl.REPO / "module_work" / "report_ingest_harness" / "_plain_t"
    args = _args(made_up)
    args.out_dir = str(tracked)
    try:
        assert rpb.run(args, chat_builder=_builder([], [])) == rcl.REFUSED
    finally:
        if tracked.exists():
            import shutil
            shutil.rmtree(tracked)
    assert not (made_up / "ledger.json").exists()


def test_an_arm_is_not_resumed_under_another_instruction(made_up, monkeypatch):
    assert rpb.run(_args(made_up, "--only", ITEMS[0]),
                   chat_builder=_builder([], [])) == rcl.OK
    monkeypatch.setitem(rpb.INSTRUCTIONS, "calc", "Something else.")
    assert rpb.run(_args(made_up), chat_builder=_builder([], [])) == \
        rcl.REFUSED
    assert os.path.isfile(made_up / "out" / "arm.json")
