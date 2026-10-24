"""Offline tests of ``run_claude_local``: the wiring, at $0.

A made-up two-page calculation printout and two made-up truth files stand in
for the private corpus, and a scripted chat model (``live_smoke.models``)
stands in for Claude: no key, no network. What is checked is what the paid
run relies on -- the calculation stage really runs through the chat model,
every call lands on the meter and the ledger at the model's rates, the run
files and the ledger agree, the cap refuses the next call once it is spent
(and nothing goes out past it), a capped item is read again by the next run,
two ledgers on one file keep one total, and an arm cannot be resumed under
other settings or written where git would track it.

Run::

    .venv/Scripts/python -m pytest module_work/report_ingest_harness/tests/test_run_claude_local.py -q
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
from report_ingest import engine  # noqa: E402

MODEL = "claude-sonnet-5-5"
USAGE = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}
#: One call at the Sonnet 5.5 rates on file: 1,000 x $2 + 100 x $10 a million.
PER_CALL = 0.003

PAGES = (
    ("SETTLEMENT CALCULATION", "Footing width B = 2.0 m",
     "Applied pressure q = 150 kPa", "Settlement S = 25 mm"),
    ("BEARING CAPACITY CHECK", "Footing width B = 1.5 m",
     "Allowable bearing pressure qa = 200 kPa"),
)
ITEMS = ("settlement__R01_p0", "shallow_foundation_bearing__R01_p1")


@pytest.fixture()
def made_up(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    (raw / "corpus").mkdir(parents=True)
    (raw / "di").mkdir()
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
        "id": ITEMS[0], "report": "R01", "kind": "settlement", "pages": [0],
        "program": None, "method": "", "subject": "",
        "inputs": [{"name": "Footing width B", "value": 2.0, "unit": "m",
                    "page": 0}],
        "results": [{"name": "Settlement S", "value": 25.0, "unit": "mm",
                     "page": 0}]}), encoding="utf-8")
    (truth / f"{ITEMS[1]}.json").write_text(json.dumps({
        "id": ITEMS[1], "report": "R01",
        "kind": "shallow_foundation_bearing", "pages": [1],
        "program": None, "method": "", "subject": "",
        "inputs": [{"name": "Footing width B", "value": 1.5, "unit": "m",
                    "page": 1}],
        "results": [{"name": "Allowable bearing pressure qa", "value": 200.0,
                     "unit": "kPa", "page": 1}]}), encoding="utf-8")
    (truth / "OPEN.txt").write_text("R01\n", encoding="utf-8")

    # The run prices the scorecard in-process and clears the ingest
    # switches; both are put back after the test.
    monkeypatch.setattr(engine, "MODEL_PRICES", dict(engine.MODEL_PRICES))
    for key, value in list(os.environ.items()):
        if key.startswith(rcl.SWITCH_PREFIX):
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("GEOTECH_INGEST_MADE_UP", "0")
    return tmp_path


def _args(tmp: Path, *extra: str, cap: str = "1.0", wave: str = "t-wave"):
    return rcl.build_parser().parse_args([
        "run", "--stage", "calc", "--model", MODEL, "--wave", wave,
        "--cap", cap, "--raw-dir", str(tmp / "raw"),
        "--ledger", str(tmp / "ledger.json"), "--out-dir", str(tmp / "out"),
        *extra])


def _answer(messages):
    """The calculation reader's structured answer, as a call to the answer
    tool the engine offers: one made-up result with its page and box."""
    from langchain_core.messages import AIMessage

    return AIMessage(content="", tool_calls=[{
        "name": "answer", "id": f"call_{len(messages)}", "args": {
            "kind": "settlement", "inputs": [],
            "results": [{"name": "Settlement S", "value": 25.0, "unit": "mm",
                         "prov": {"page": 0,
                                  "bbox": [72.0, 115.0, 260.0, 130.0]}}]}}])


def _builder(made: list):
    def build(model_id, callbacks):
        fake = scripted_model(_answer, usage=dict(USAGE),
                              callbacks=callbacks, model=model_id)
        made.append(fake)
        return fake
    return build


def _sent(made: list) -> int:
    return sum(len(fake.calls) for fake in made)


def _ledger(tmp: Path) -> dict:
    return json.loads((tmp / "ledger.json").read_text(encoding="utf-8"))


def _done(out: Path) -> list:
    return sorted(p.stem for p in (out / "calc").glob("*.json"))


def test_the_stage_runs_and_every_call_is_metered(made_up):
    made: list = []
    code = rcl.run(_args(made_up), chat_builder=_builder(made))

    assert code == rcl.OK
    out = made_up / "out"
    assert _done(out) == sorted(ITEMS)
    sent = _sent(made)
    assert sent >= len(ITEMS)
    wave = _ledger(made_up)["waves"]["t-wave"]
    assert wave["calls"] == sent
    assert wave["spent_usd"] == pytest.approx(sent * PER_CALL)
    assert _ledger(made_up)["cumulative_usd"] == pytest.approx(
        sent * PER_CALL)
    assert wave["scenarios"]["calc_sonnet"]["calls"] == sent

    calls = [json.loads(line) for line in (out / "spend_calls.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    assert len(calls) == sent
    assert {c["model"] for c in calls} == {MODEL}
    assert all("lc_images" in c for c in calls)

    # The run files are priced as the ledger prices the calls.
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["calc"]["cost"]["calls"] == sent
    assert results["calc"]["cost"]["dollars"] == pytest.approx(
        sent * PER_CALL)
    text = (out / "RESULTS.md").read_text(encoding="utf-8")
    assert "## Claude, locally (run_claude_local)" in text
    assert "agree: calls, tokens and dollars" in text
    assert "### Per item" in text and ITEMS[0] in text

    # The reader took the answer: no item failed, and the value it read
    # carries its page and box into the record.
    blobs = rcl.run_files(out, "calc")
    assert rcl.call_health(blobs)["items_with_error"] == 0
    assert rcl.call_health(blobs)["answers_not_taken"] == 0
    record = rcl.provenance(blobs)["record"]
    assert record["records"] == len(ITEMS)
    assert record["values"] >= 1 and record["none"] == 0
    assert record["own_page"] == record["own"] == record["values"]
    model = rcl.provenance(blobs)["model_record"]
    assert model["own_page"] == model["values"] >= 1
    assert model["own_box"] >= 1 and model["methods"] == {
        "model": model["values"]}
    assert json.loads((out / "arm.json").read_text(encoding="utf-8"))[
        "model"] == MODEL


def test_the_cap_refuses_the_next_call_and_a_raised_cap_goes_on(made_up):
    made: list = []
    code = rcl.run(_args(made_up, cap="0.001"), chat_builder=_builder(made))

    # The first call goes out (nothing was spent) and costs more than the
    # cap; the first item is read whole on it. Every call after it is
    # refused before it reaches the model, so the second item is set aside
    # to be read again rather than kept as done.
    assert code == rcl.STOPPED_BY_CAP
    assert _sent(made) == 1
    wave = _ledger(made_up)["waves"]["t-wave"]
    assert wave["calls"] == 1 and wave["stopped_by_cap"]
    assert wave["spent_usd"] == pytest.approx(PER_CALL)
    out = made_up / "out"
    assert _done(out) == [ITEMS[0]]
    assert (out / "calc" / "_capped" / f"{ITEMS[1]}.json").is_file()
    blob = (out / "calc" / f"{ITEMS[0]}.json").read_text(encoding="utf-8")
    assert not rcl.CAP_TEXT.search(blob)

    # The same wave and cap again: refused before any model is built.
    again: list = []
    assert rcl.run(_args(made_up, cap="0.001"),
                   chat_builder=_builder(again)) == rcl.REFUSED
    assert again == []

    # A raised cap on the same wave reads what the cap cut.
    more: list = []
    assert rcl.run(_args(made_up, cap="1.0"),
                   chat_builder=_builder(more)) == rcl.OK
    assert _done(out) == sorted(ITEMS)
    assert _sent(more) == 1                     # the first item is not redone
    assert _ledger(made_up)["waves"]["t-wave"]["calls"] == 2
    # A refused call cost nothing and left no trace in the spend: the two
    # runs' metered calls and the two run files agree.
    assert "agree: calls, tokens and dollars" in (
        out / "RESULTS.md").read_text(encoding="utf-8")


def test_an_arm_is_not_resumed_under_other_settings(made_up, capsys):
    made: list = []
    assert rcl.run(_args(made_up, "--env", "GEOTECH_INGEST_MADE_UP=1"),
                   chat_builder=_builder(made)) == rcl.OK
    assert os.environ["GEOTECH_INGEST_MADE_UP"] == "1"

    later: list = []
    assert rcl.run(_args(made_up), chat_builder=_builder(later)) \
        == rcl.REFUSED
    assert later == []
    assert "switches" in capsys.readouterr().out
    # Every ingest switch is cleared before an arm sets its own.
    assert "GEOTECH_INGEST_MADE_UP" not in os.environ


def test_sets_and_only_pick_items_and_copy_the_open_file(made_up):
    truth = made_up / "raw" / "truth"
    assert [p.stem for p in rcl.pick_items("calc", truth, "open", [])] \
        == sorted(ITEMS)
    assert rcl.pick_items("calc", truth, "blind", []) == []
    chosen = rcl.pick_items("calc", truth, "all", [ITEMS[1]])
    assert [p.stem for p in chosen] == [ITEMS[1]]
    assert [p.stem for p in rcl.pick_items("calc", truth, "all", ["R01"])] \
        == sorted(ITEMS)
    with pytest.raises(ValueError, match="R99"):
        rcl.pick_items("calc", truth, "all", ["R99"])

    out = made_up / "out"
    root = rcl.truth_for_run("calc", truth, chosen, out, picked=True)
    assert root == out / "_truth"
    assert sorted(p.name for p in (root / "calc").iterdir()) == [
        "OPEN.txt", f"{ITEMS[1]}.json"]
    assert rcl.truth_for_run("calc", truth, chosen, out,
                             picked=False) == truth


def test_two_ledgers_on_one_file_keep_one_total(tmp_path):
    path = tmp_path / "ledger.json"
    a, b = rcl.shared_ledger(path), rcl.shared_ledger(path)
    a.begin_wave("a", 5.0, model=MODEL, mode="t")
    b.begin_wave("b", 5.0, model=MODEL, mode="t")
    a.add("a", 1.0, {"output": 10})
    b.add("b", 2.0, {"output": 20})
    a.add("a", 0.5, {"output": 5})

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["cumulative_usd"] == pytest.approx(3.5)
    assert data["waves"]["a"]["spent_usd"] == pytest.approx(1.5)
    assert data["waves"]["b"]["spent_usd"] == pytest.approx(2.0)
    assert b.cumulative == pytest.approx(3.5)        # read off the file
    assert not Path(str(path) + ".lock").exists()


def test_an_out_dir_git_would_track_is_refused(made_up):
    probe = rcl.REPO / "module_work" / "report_ingest_harness" / "_t_probe"
    if rcl.git_ignored(probe, rcl.REPO) is not False:
        pytest.skip("needs git and a checkout")
    args = rcl.build_parser().parse_args([
        "run", "--stage", "calc", "--model", MODEL, "--wave", "t-wave",
        "--cap", "1", "--raw-dir", str(made_up / "raw"),
        "--ledger", str(made_up / "ledger.json"), "--out-dir", str(probe)])
    made: list = []

    assert rcl.run(args, chat_builder=_builder(made)) == rcl.REFUSED
    assert made == [] and not probe.exists()


def _rows_file(out: Path, stage: str, name: str, rows: list) -> None:
    folder = out / stage
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(
        {"id": name, "score": {"error": None},
         "cost": {"per_call": rows}}), encoding="utf-8")


def test_an_answer_emptied_by_its_dropped_keys_is_an_error(tmp_path):
    # Made-up rows, as the engine writes them since 2026-10-24: one answer
    # unwrapped, one left with nothing once its unknown key was dropped.
    out = tmp_path / "out"
    _rows_file(out, "narrative", "R01", [
        {"call": 1, "outcome": "reply", "stop": "tool_use",
         "notes": ["the answer arrived wrapped in 'arguments'; unwrapped"],
         "answer_args": ['{"arguments": {"projectName": "Made-up"}}']},
        {"call": 2, "outcome": "reply", "stop": "tool_use",
         "notes": ["unknown keys dropped from the answer: arguments",
                   "an answer was not taken: nothing was left once the "
                   "unknown keys arguments were dropped"],
         "answer_args": ['{"arguments": "see above"}']},
        {"call": 3, "outcome": "reply", "stop": "tool_use",
         "notes": ["unknown keys dropped from the answer: confidence"],
         "answer_args": ['{"projectName": "Made-up", "confidence": 1}']},
    ])
    health = rcl.call_health(rcl.run_files(out, "narrative"))
    assert (health["answers_unwrapped"], health["answers_emptied"],
            health["answers_not_taken"], health["unknown_keys_dropped"],
            health["emptied_unverified"]) == (1, 1, 1, 2, 0)
    assert health["emptied_keys"] == {"arguments": 1}

    lines = rcl.check_lines(out, "narrative")
    (error,) = [line for line in lines if line.startswith("ERROR")]
    assert error.startswith("ERROR: 1 answer(s) held nothing")
    assert "'arguments' 1" in error


def test_a_row_from_before_the_fix_is_flagged_unverified(tmp_path):
    # A taken answer that lost a top-level key, with no raw arguments kept:
    # the shape the wrapped-answer bug left in the run files.
    out = tmp_path / "out"
    _rows_file(out, "narrative", "R02", [
        {"call": 1, "outcome": "reply", "stop": "tool_use",
         "notes": ["unknown keys dropped from the answer: arguments"]},
        {"call": 2, "outcome": "reply", "stop": "tool_use",
         "notes": ["unknown keys dropped from the answer: results.0.colour"]},
        {"call": 3, "outcome": "reply", "stop": "tool_use",
         "notes": ["unknown keys dropped from the answer: arguments",
                   "an answer was not taken: it does not match the schema "
                   "(kind: Field required)"]},
    ])
    health = rcl.call_health(rcl.run_files(out, "narrative"))
    assert health["emptied_unverified"] == 1, (
        "a nested key, or an answer not taken, is not suspect")
    lines = rcl.check_lines(out, "narrative")
    (error,) = [line for line in lines if line.startswith("ERROR")]
    assert error.startswith("ERROR (unverified): 1 answer(s)")
    assert error.endswith("check the reading.")


def test_the_check_reads_the_engine_s_own_notes(tmp_path):
    # The rows come from the engine itself, so the check and the engine's
    # wording cannot drift apart. Made-up schema and values.
    from pydantic import BaseModel

    from report_ingest.tests.chat_fixtures import (ScriptedChatModel,
                                                   answer_call)

    class Reading(BaseModel):
        model_config = {"extra": "forbid"}
        site: str = ""
        depth_m: float = 0.0

    class Loose(BaseModel):
        site: str = ""

    model = ScriptedChatModel(replies=[
        answer_call({"arguments": {"answer": {"site": "Made-up"}}}),
        answer_call({"arguments": {"colour": "red"}}),
        answer_call({"site": "A", "arguments": {"site": "B"}}),
        answer_call({"site": "A", "<b>odd</b>": {"depth_m": 3.0}}),
        answer_call({"<b>odd key</b>": 1, "fine_key": 2}),
        answer_call({"stray": 1}),
    ])
    chat = engine.ChatModelEngine(model)
    for schema in (Reading, Reading, Reading, Reading, Loose, Reading):
        chat.complete([engine.user(engine.text_block("go"))],
                      output_format=schema)
    out = tmp_path / "out"
    _rows_file(out, "narrative", "R03", chat.meter.to_dict()["per_call"])

    health = rcl.call_health(rcl.run_files(out, "narrative"))
    assert (health["answers_unwrapped"], health["answers_emptied"],
            health["stray_objects"], health["answers_not_taken"],
            health["emptied_unverified"]) == (2, 3, 2, 3, 0)
    assert health["emptied_keys"] == {"colour": 1, "(other)": 1,
                                      "fine_key": 1, "stray": 1}
    assert health["stray_keys"] == {"arguments": 1, "(other)": 1}
    flagged = [line for line in rcl.check_lines(out, "narrative")
               if line.startswith(("ERROR", "WARNING"))]
    assert [e.split(" answer(s)")[0] for e in flagged] == [
        "ERROR: 3", "WARNING: 2"]
    assert "odd" not in "\n".join(flagged)
