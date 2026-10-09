"""The suite scores what the user receives (Foundry brief 5, N1 and N5).

N1: a valid DIGGS file was written to ``/tmp`` in 6 of 6 runs and failed
``file_produced`` 6 of 6: the runner collected only its ``files/`` folder,
while the app's turn job copies any file a tool REPORTS writing elsewhere.
The runner now does the same, with the app's own code. (N2 also keeps a
``call_agent`` writer's file in the run's folder in the first place.)

N5: the "values" and "borings" checks of the report-extraction tasks read
only the answer. Every value was in the delivered CSV / docx / md, and one
correct answer named its borings as ranges ("BH-1 to BH-3"). Those checks
now read the delivered files' text too and read ranges; and "shell" (a soil
description the question never asks for) is no longer checked.
"""

import json
import os
import uuid

import pytest

pytest.importorskip("planlens.tools")
pytest.importorskip("fitz")

from funhouse_agent import review_flags  # noqa: E402
from funhouse_agent.review_eval import checks as C  # noqa: E402
from funhouse_agent.review_eval.tasks import OPEN_TASKS, Task  # noqa: E402


@pytest.fixture(autouse=True)
def _no_switches(monkeypatch):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


def _task(tid):
    return next(t for t in OPEN_TASKS if t.id == tid)


# ---------------------------------------------------------------------------
# N5: ranges, delivered files, the task's terms
# ---------------------------------------------------------------------------

def test_id_ranges_name_their_members():
    text = C.expand_id_ranges(C.normalize(
        "Borings B-1 to B-3 (2026) and BH-1 – BH-3 (2011); samples S-1 "
        "through S-4; pages 7-10."))
    for want in ("b-2", "bh-2", "s-3"):
        assert f"{want}," in text or f"{want})" in text, (want, text)
    assert "pages 7-10." in text                      # not an id range
    assert C.expand_id_ranges("b-3 to b-1") == "b-3 to b-1"   # backwards


def test_set_match_reads_a_range_as_its_members():
    borings = ["B-1", "B-2", "B-3", "BH-1", "BH-2", "BH-3"]
    ok, detail = C.check_set_match(
        "The 2026 borings B-1 to B-3 and the 2011 borings BH-1 to BH-3.",
        vocabulary=borings, expected=borings)
    assert ok, detail
    ok, _ = C.check_set_match("B-1 and BH-1 to BH-3", vocabulary=borings,
                              expected=borings)
    assert not ok                                     # B-2, B-3 not named


def _write_docx(path, rows):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            t.cell(i, j).text = str(v)
    d.save(str(path))


def test_a_with_files_check_reads_the_delivered_tables(tmp_path):
    """The brief-5 shape: a short answer pointing at the files, the values
    in the files."""
    csv = tmp_path / "borings.csv"
    csv.write_text("boring,depth_m,N\nB-3,7.0,43\n", encoding="utf-8")
    md = tmp_path / "lab.md"
    md.write_text("| B-2 S-5 | fines | 41.7 % |\n", encoding="utf-8")
    _write_docx(tmp_path / "lab.docx", [["B-1 BULK-1", "MDD 1.94 Mg/m3"],
                                        ["B-2 W-1", "sulfate 1,850 mg/L"]])
    (tmp_path / "plot.png").write_bytes(b"\x89PNG not text")
    files = [str(p) for p in tmp_path.iterdir()]
    task = _task("report-extract-all")
    values = next(c for c in task.checks if c["type"] == "contains_all")
    answer = "The tables are in the attached Word file and CSVs."
    got = C.run_check(values, answer, files=files)
    assert got["passed"], got["detail"]
    assert "lab.docx" in got["detail"] and "borings.csv" in got["detail"]
    alone = C.run_check({k: v for k, v in values.items()
                         if k != "with_files"}, answer, files=files)
    assert not alone["passed"]                        # the old reading


def test_the_extraction_tasks_read_their_files_and_drop_shell():
    extract = _task("report-extract-all")
    values = next(c for c in extract.checks if c["type"] == "contains_all")
    assert "shell" not in json.dumps(values["terms"])
    assert values["with_files"] is True
    for tid in ("report-extract-all", "report-extract-diggs"):
        borings = next(c for c in _task(tid).checks
                       if c["type"] == "set_match")
        assert borings["with_files"] is True, tid


def test_a_with_files_check_with_no_files_reads_the_answer():
    ok = C.run_check({"type": "contains_all", "terms": ["43"],
                      "with_files": True}, "N = 43 at 7.0 m", files=[])
    assert ok["passed"] and "answer +" not in ok["detail"]


# ---------------------------------------------------------------------------
# N1 + N2: the runner delivers what a tool wrote
# ---------------------------------------------------------------------------

def _one_call_model(name, args, answer):
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel)
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            if not any(getattr(m, "type", "") == "tool" for m in messages):
                msg = AIMessage(content="", tool_calls=[
                    {"name": name, "args": args, "id": "c1"}])
            else:
                msg = AIMessage(content=answer)
            return ChatResult(generations=[ChatGeneration(message=msg)])

    return Model(responses=[AIMessage(content="x")])


def test_a_file_reported_outside_the_run_is_delivered(tmp_path):
    """N1: save_file honours an absolute path outside the run's folder; the
    runner copies the reported file in, as the app does."""
    from funhouse_agent.review_eval.runner import run_task
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    target = outside / "table.csv"
    model = _one_call_model("save_file", {"path": str(target),
                                          "content": "B-1,9\n"},
                            "Saved the table.")
    task = Task(id="n1-save", question="Save a table.",
                documents=list(_task("fixture-markups").documents),
                category="extract", doc_type="report", auto_checks=False,
                checks=[{"type": "file_produced", "ext": ".csv"}])
    res = run_task(task, model, arm="baseline", arm_env={}, docs_dir=None,
                   run_dir=str(tmp_path / "run"))
    assert res["error"] is None, res.get("traceback")
    assert target.is_file()                     # where the tool put it
    assert os.path.join("files", "table.csv") in res["files"]
    assert res["imported_outputs"] == {
        str(target): os.path.join("files", "table.csv")}
    assert res["score"]["passed"], res["checks"]


def test_write_diggs_to_tmp_is_delivered_on_the_geotech_page(tmp_path):
    """N2 through the agent: the brief-5 call, call_agent write_diggs with a
    /tmp path, lands in the run's files/ and file_produced passes."""
    from funhouse_agent.review_eval.runner import run_task
    name = f"harbour_road_{uuid.uuid4().hex[:8]}.diggs.xml"
    boring = {"investigation_id": "B-1", "kind": "boring", "depth_unit": "m",
              "total_depth": {"value": 6.0, "unit": "m"}}
    model = _one_call_model("call_agent", {
        "agent_name": "subsurface", "method": "write_diggs",
        "parameters": {"investigations": [boring],
                       "output_path": f"/tmp/{name}"}}, "Wrote the file.")
    task = Task(id="n2-diggs", question="Write DIGGS.",
                documents=list(_task("fixture-markups").documents),
                category="extract", doc_type="report", page="geotech",
                auto_checks=False,
                checks=[{"type": "file_produced", "ext": ".xml"}])
    res = run_task(task, model, arm="baseline", arm_env={}, docs_dir=None,
                   run_dir=str(tmp_path / "run"))
    assert res["error"] is None, res.get("traceback")
    assert os.path.join("files", name) in res["files"], res["files"]
    assert res["score"]["passed"], res["checks"]
    assert not os.path.exists(os.path.join(os.path.abspath("/tmp"), name))
    act = (tmp_path / "run" / "activity.jsonl").read_text(encoding="utf-8")
    assert "output_note" in act                 # the agent was told where
