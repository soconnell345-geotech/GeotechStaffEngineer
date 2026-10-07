"""The suite's coverage tasks (plan W4) - offline, no model, no network.

Pinned: the synthetic report is what its tasks say; ``pages_covered`` is
derived from the run's own activity (every tool call of the primary and its
helpers), never from the app's ledger; ``states_coverage`` wants a count of
what was read; a task can be asked on the geotech page; and on a scripted
model the ``coverage`` arm turns the extraction task that ``baseline`` fails.
"""

import json
import re

import pytest

pytest.importorskip("planlens.tools")
fitz = pytest.importorskip("fitz")

from funhouse_agent import review_flags  # noqa: E402
from funhouse_agent.review_eval import checks as C  # noqa: E402
from funhouse_agent.review_eval import report_fixture as RF  # noqa: E402
from funhouse_agent.review_eval.tasks import OPEN_TASKS, Task  # noqa: E402


@pytest.fixture(autouse=True)
def _no_switches(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


def test_the_report_fixture_is_what_its_tasks_say():
    gt = RF.build_synthetic_extraction_report()
    doc = fitz.open(stream=gt.pdf, filetype="pdf")
    try:
        assert doc.page_count == gt.n_pages == 30
        for p in RF.SCANNED_LOG_PAGES:            # scans: no text at all
            assert not doc[p].get_text().strip()
            assert doc[p].get_images()
        for p, b in zip(RF.VECTOR_LOG_PAGES, ("B-1", "B-1", "B-2", "B-3")):
            assert f"BORING NO. {b}" in doc[p].get_text()
        sheet = doc[18].get_text()
        assert "B-2" in sheet and "S-3" in sheet and "32" in sheet
        summary = doc[16].get_text().split("\n")
        i = summary.index("S-3", summary.index("B-2"))
        # the planted error: LL and PL swapped on the summary
        assert summary[i + 1:i + 7] == ["4.0", "27.5", "20", "32", "12", "-"]
        assert "1,850" in doc[29].get_text()
    finally:
        doc.close()
    assert set(RF.TARGET_PAGES) == set(gt.log_pages) | set(gt.lab_pages)


def test_the_report_resolves_as_a_suite_document():
    from funhouse_agent.review_eval import documents as D
    name, data = D.resolve("fixture_report")
    assert name.endswith(".pdf") and data[:4] == b"%PDF"


def _activity(calls):
    recs = []
    for i, (name, args, result) in enumerate(calls):
        recs.append({"event": "tool_start", "run_id": str(i), "name": name,
                     "args": args, "agent": "primary"})
        recs.append({"event": "tool_end", "run_id": str(i), "name": name,
                     "result": result, "agent": "general-purpose"})
    return recs


def test_pages_covered_is_derived_from_the_activity():
    act = _activity([
        ("read_document", {"handle": "h", "pages": "0-5"},
         json.dumps({"pages_returned": "0-3", "next": {"pages": "4-5"}})),
        ("analyze_pdf_page", {"attachment_key": "r.pdf", "page": 9},
         json.dumps({"page": 9})),
        ("read_pdf_text", {"source": "r.pdf"},
         json.dumps({"pages": [{"page": 8, "has_text_layer": False}]})),
        ("render_region", {"attachment_key": "r.pdf", "page": 7},
         json.dumps({"error": "no engine"})),
    ])
    ok, detail = C.check_pages_covered("", activity=act,
                                       pages=[0, 1, 2, 3, 9], look_pages=[9])
    assert ok, detail
    ok, detail = C.check_pages_covered("", activity=act,
                                       pages=[0, 4, 7, 8, 9], look_pages=[8])
    assert not ok and "2 of 5" in detail
    assert "not read: PDF pages 5, 8, 9" in detail
    assert "PDF pages 9 were read as text but have no text layer" in detail
    ok, detail = C.check_pages_covered("", activity=[], pages=[1])
    assert not ok and "no activity" in detail
    # through run_check, as the runner calls it
    r = C.run_check({"type": "pages_covered", "pages": [0, 1]}, "",
                    activity=act)
    assert r["passed"]


def test_states_coverage_wants_a_count_of_what_was_read():
    assert C.check_states_coverage("Coverage: 10 of 23 lab sheets read.")[0]
    assert C.check_states_coverage("I read all 14 laboratory sheets")[0]
    assert not C.check_states_coverage("I read the lab sheets.")[0]
    assert not C.check_states_coverage("The ratio is 3 of 4.")[0]


def _report_model(final_answer, *, read_all_first=False):
    """Opens the uploaded report, reads three lab sheets, answers; after a
    coverage note, reads the rest and looks at the scans, then answers
    ``final_answer``. ``read_all_first`` reads everything up front."""
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel)
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from funhouse_agent.coverage import GATE_PREFIX

    class Model(FakeMessagesListChatModel):
        calls: list = []

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            if len(messages) == 1 and isinstance(messages[0].content, list):
                return self._r(AIMessage(content="A scanned boring log."))
            self.calls.append(1)
            humans = [m for m in messages if m.type == "human"]
            key = re.search(r"'([^']+\.pdf)'", str(humans[0].content)).group(1)
            tools = [m for m in messages if m.type == "tool"]
            names = [m.name for m in tools]
            gate = any(str(m.content).startswith(GATE_PREFIX)
                       for m in humans)
            handle = next((json.loads(m.content)["handle"] for m in tools
                           if m.name == "open_document"), None)
            n = len(self.calls)
            if handle is None:
                return self._calls([("open_document", {"source": key})], n)
            if "read_document" not in names:
                # one page a call when reading everything: a long range
                # comes back in parts (a 'next' cursor) the fake would not
                # follow
                calls = ([("read_document", {"handle": handle,
                                             "pages": str(p)})
                          for p in range(30)] if read_all_first else
                         [("read_document", {"handle": handle,
                                             "pages": "16-18"})])
                if read_all_first:
                    calls += [("analyze_pdf_page", {"attachment_key": key,
                                                    "page": p})
                              for p in RF.SCANNED_LOG_PAGES]
                return self._calls(calls, n)
            if gate and names.count("read_document") < 2:
                return self._calls(
                    [("read_document", {"handle": handle,
                                        "pages": "7-10,19-29"})]
                    + [("analyze_pdf_page", {"attachment_key": key,
                                             "page": p})
                       for p in RF.SCANNED_LOG_PAGES], n)
            if gate or read_all_first:
                return self._r(AIMessage(content=final_answer))
            return self._r(AIMessage(content="Tables of the three Atterberg "
                                             "sheets I read."))

        def _r(self, msg):
            return ChatResult(generations=[ChatGeneration(message=msg)])

        def _calls(self, calls, n):
            return self._r(AIMessage(content="", tool_calls=[
                {"name": name, "args": args, "id": f"c{n}_{i}"}
                for i, (name, args) in enumerate(calls)]))

    return Model(responses=[AIMessage(content="x")], calls=[])


def test_the_coverage_arm_is_what_turns_the_extraction_task(tmp_path):
    """Offline, on a scripted model that stops after three sheets unless it
    is told: `baseline` fails the coverage checks, `coverage` passes them -
    the arm measures the change it adds, through the run's own activity and
    answer only."""
    from funhouse_agent.review_eval import score_review_suite
    task = next(t for t in OPEN_TASKS if t.id == "report-extract-all")
    model = _report_model(task.truth)
    res = score_review_suite(model, ids=["report-extract-all"],
                             arms=("baseline", "coverage"),
                             out_dir=tmp_path, verbose=False)
    base = res["results"]["baseline"]["report-extract-all"]
    cov = res["results"]["coverage"]["report-extract-all"]
    assert base["error"] is None and cov["error"] is None, cov.get("traceback")
    by = {c["label"]: c for c in base["checks"]}
    assert not by["every log page and laboratory page was read"]["passed"]
    assert not by["the answer states its coverage as counts"]["passed"]
    assert cov["score"]["passed"], cov["checks"]
    assert (tmp_path / "runs" / "coverage" / "report-extract-all" /
            "coverage.json").is_file()
    # the ledger lives beside the activity, never among the produced files
    assert not any("coverage.json" in f for f in cov["files"])
    assert "`coverage` FIXES report-extract-all" in res["results_md"]
    assert "By app page" in res["results_md"]


def test_a_geotech_page_task_is_run_on_the_geotech_page(tmp_path):
    from funhouse_agent.review_eval.runner import build_page_agent, run_task
    task = next(t for t in OPEN_TASKS if t.id == "report-extract-diggs")
    assert task.page == "geotech"
    model = _report_model(task.truth, read_all_first=True)
    res = run_task(task, model, arm="baseline", arm_env={}, docs_dir=None,
                   run_dir=str(tmp_path / "r"))
    assert res["error"] is None, res.get("traceback")
    assert res["page"] == "geotech"
    by = {c["label"]: c for c in res["checks"]}
    read = by["every log page and laboratory page was read"]
    assert read["passed"], (read["detail"], res["tool_counts"])
    # the geotech page's own agent: its analysis catalogue is on the surface
    agent = build_page_agent(model, {}, str(tmp_path / "g"), [],
                             page="geotech")
    assert "call_agent" in set(agent.nodes["tools"].bound.tools_by_name)
    review = build_page_agent(model, {}, str(tmp_path / "v"), [])
    assert "call_agent" not in set(review.nodes["tools"].bound.tools_by_name)


def test_a_task_round_trips_its_page():
    t = Task.from_dict({"id": "x", "question": "q", "page": "geotech"})
    assert t.page == "geotech" and Task.from_dict(t.to_dict()).page == "geotech"
    assert Task.from_dict({"id": "y", "question": "q"}).page == "review"


def test_the_coverage_arms_are_switches_on_the_default_agents():
    A = review_flags
    assert A.ARMS["coverage"] == {A.COVERAGE_ENV: "1"}
    assert A.ARMS["checklist"] == {A.COVERAGE_ENV: "1", A.CHECKLIST_ENV: "1"}
    with A.switches(A.ARMS["checklist"]):
        assert A.coverage() and A.checklist() and not A.lean_agent()
    with A.switches(A.ARMS["baseline"]):
        assert not A.coverage() and not A.checklist()
