"""Coverage in code (plan W4): the ledger, the gate, the tools, the checklist.

Offline: no model, no network. The document is the review suite's synthetic
report (``funhouse_agent/review_eval/report_fixture.py``): new logs as vector
pages (7-10), old logs as scans (12-14), a laboratory appendix of a summary
and 13 sheets (16-29). The agents are driven by scripted fake models.

What is pinned:

* which pages a tool call READ comes from the call (its arguments and its
  result), and a scanned page read only as text was not read;
* the inventory comes from the document's own structure (planlens' page
  roles), the same rules report ingest builds its work items from;
* coverage is counted from the ledger, never from the model's words;
* the gate tells the agent ONCE, with the list, and never loops - on both
  pages, with a helper's reads counting, and it stands down rather than cost
  a turn its answer;
* the checklist is DATA, its code items run, its judgement items are handed
  over, and a code item that cannot run is handed over too.
"""

import json
import os

import pytest

pytest.importorskip("planlens.document.roles")
pytest.importorskip("fitz")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import Field  # noqa: E402

from funhouse_agent import coverage as C  # noqa: E402
from funhouse_agent import report_checklist as RC  # noqa: E402
from funhouse_agent import review_flags  # noqa: E402
from funhouse_agent.deep import coverage_tools as CT  # noqa: E402
from funhouse_agent.review_eval import report_fixture as RF  # noqa: E402

KEY = "report.pdf"


@pytest.fixture(autouse=True)
def _switches(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


@pytest.fixture(scope="module")
def report():
    return RF.build_synthetic_extraction_report()


@pytest.fixture
def att(report):
    return {KEY: report.pdf}


def _look(ledger, page, key=KEY):
    return ledger.record_call("analyze_pdf_page",
                              {"attachment_key": key, "page": page},
                              json.dumps({"page": page, "analysis": "seen"}))


def _handle(att):
    from funhouse_agent import document_tools
    out = document_tools.dispatch_document_tool(
        "open_document", {"source": KEY}, attachments=att)
    return json.loads(out)["handle"]


def _read(ledger, att, pages):
    from funhouse_agent import document_tools
    h = _handle(att)
    out = document_tools.dispatch_document_tool(
        "read_document", {"handle": h, "pages": pages}, attachments=att,
        max_chars=14000)
    return ledger.record_call("read_document", {"handle": h, "pages": pages},
                              out)


# ---------------------------------------------------------------------------
# What a call read
# ---------------------------------------------------------------------------

def test_pages_come_from_the_result_and_a_failed_call_reads_nothing():
    ref, got = C.pages_from_call(
        "read_document", {"handle": "doc_x", "pages": "0-40"},
        json.dumps({"handle": "doc_x", "pages_returned": "3-5,9",
                    "next": {"pages": "10-40"}}))
    assert ref == ("handle", "doc_x")
    assert got == [(3, "text"), (4, "text"), (5, "text"), (9, "text")]
    ref, got = C.pages_from_call(
        "read_pdf_text", {"source": "r.pdf"},
        json.dumps({"pages": [{"page": 2, "has_text_layer": True},
                              {"page": 3, "has_text_layer": False}]}))
    assert got == [(2, "text"), (3, "text_empty")]
    assert C.pages_from_call("analyze_pdf_page",
                             {"attachment_key": "r.pdf", "page": 7},
                             '{"page": 7}')[1] == [(7, "look")]
    assert C.pages_from_call("render_region",
                             {"attachment_key": "r.pdf", "page": 7},
                             '{"error": "no vision engine"}') == (None, [])
    _ref, got = C.pages_from_call(
        "sweep_pages", {"source": "r.pdf"},
        json.dumps({"pages_checked": "0-3",
                    "unanswered": [{"page": 2, "why": "x"}]}))
    assert got == [(0, "sweep"), (1, "sweep"), (3, "sweep")]
    # A search is not a read; nor are thumbnails.
    assert C.pages_from_call("search_document", {"handle": "h"},
                             '{"hits": [{"page": 4}]}') == (None, [])


def test_parse_and_compact_pages():
    assert C.parse_pages("2-4, 7,9-9") == [2, 3, 4, 7, 9]
    assert C.parse_pages([1, "3-4"], n_pages=4) == [1, 3]
    assert C.parse_pages(None) == [] and C.parse_pages(True) == []
    assert C.compact([9, 2, 3, 4, 7]) == "2-4,7,9"
    assert C.compact([0, 1], offset=1) == "1-2"


def test_states_coverage():
    assert C.states_coverage("10 of 23 lab sheets read")
    assert C.states_coverage("all 14 laboratory sheets")
    assert not C.states_coverage("I read the lab sheets.")


# ---------------------------------------------------------------------------
# The inventory and the ledger
# ---------------------------------------------------------------------------

def test_inventory_is_the_documents_own_structure(att, report):
    ledger = C.CoverageLedger(attachments=att)
    key = ledger.resolve(("source", KEY))
    inv = ledger.docs[key]["inventory"]
    assert inv.n_pages == report.n_pages == 30
    logs = [p.page for p in inv.pages if p.group == "logs"]
    lab = [p.page for p in inv.pages if p.group == "lab"]
    assert logs == list(report.log_pages) == [7, 8, 9, 10, 12, 13, 14]
    assert lab == list(report.lab_pages)
    assert [p.page for p in inv.pages if not p.has_text] == list(
        report.look_pages)
    assert sorted(inv.targets(C.DATA_GROUPS)) == sorted(report.target_pages)
    assert {p.page for p in inv.pages if p.group == "front"} == {0, 1, 6, 11,
                                                                 15}


def test_one_document_whatever_it_is_called(att, report, tmp_path):
    path = tmp_path / "copy.pdf"
    path.write_bytes(report.pdf)
    ledger = C.CoverageLedger(attachments=att)
    _look(ledger, 7)
    _look(ledger, 8, key=str(path))
    _read(ledger, att, "9")
    assert len(ledger.docs) == 1
    key = next(iter(ledger.docs))
    assert {7, 8, 9} <= set(ledger.docs[key]["reads"])


def test_a_scan_read_as_text_is_not_read(att):
    ledger = C.CoverageLedger(attachments=att)
    ledger.record_call("read_pdf_text", {"source": KEY, "pages": "12"},
                       json.dumps({"pages": [{"page": 12,
                                              "has_text_layer": False}]}))
    key = next(iter(ledger.docs))
    assert not ledger.covered(key, 12)
    row = [r for r in ledger.group_rows(key) if r["group"] == "logs"][0]
    assert row["read_as_text_but_no_text_layer"] == "12"
    _look(ledger, 12)
    assert ledger.covered(key, 12)


def test_counts_marks_and_the_statement(att):
    ledger = C.CoverageLedger(attachments=att)
    _read(ledger, att, "16-25")
    key = next(iter(ledger.docs))
    ledger.mark(key, [16, 17], "extracted")
    ledger.mark(key, [26], "skipped", "duplicate of sheet 25")
    ledger.mark(key, [28], "extracted")            # claimed, never read
    lab = [r for r in ledger.group_rows(key) if r["group"] == "lab"][0]
    assert (lab["pages"], lab["read"], lab["extracted"], lab["skipped"]) == \
        (14, 10, 3, 1)
    assert lab["not_read"] == "27-29" and lab["not_read_pdf"] == "28-30"
    assert lab["marked_extracted_but_never_read"] == "28"
    st = ledger.statement(key)
    assert "laboratory sheets 10 of 14 read" in st
    assert "not read: PDF pages 28-30" in st and "1 skipped" in st
    front = [r for r in ledger.group_rows(key) if r["group"] == "front"][0]
    assert "not_read" not in front                  # never asked for


def test_the_ledger_survives_a_rebuild(att, tmp_path):
    path = C.ledger_path(str(tmp_path))
    ledger = C.CoverageLedger(path=path, attachments=att)
    _look(ledger, 12)
    key = next(iter(ledger.docs))
    ledger.mark(key, [13], "skipped", "illegible")
    again = C.CoverageLedger(path=path, attachments=att)
    assert again.covered(key, 12)
    assert again.docs[key]["marks"][13]["reason"] == "illegible"
    assert again.docs[key]["inventory"].n_pages == 30
    data = json.loads(open(path, encoding="utf-8").read())
    assert "about" in data and key in data["documents"]


def test_parallel_tool_calls_all_land_in_the_saved_ledger(att, tmp_path):
    """Tools run in parallel threads; every read must be in the file."""
    import threading
    path = C.ledger_path(str(tmp_path))
    ledger = C.CoverageLedger(path=path, attachments=att)
    _look(ledger, 0)                              # build the inventory once
    threads = [threading.Thread(target=_look, args=(ledger, p))
               for p in range(1, 30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    again = C.CoverageLedger(path=path, attachments=att)
    key = next(iter(again.docs))
    assert sorted(again.docs[key]["reads"]) == list(range(30))


def test_record_call_never_raises():
    ledger = C.CoverageLedger(attachments={})
    assert ledger.record_call("analyze_pdf_page",
                              {"attachment_key": "nothing.pdf", "page": 1},
                              "{}") == []
    assert ledger.record_call("read_document", {"handle": "doc_none"},
                              "not json") == []


# ---------------------------------------------------------------------------
# The gate's arming and its note
# ---------------------------------------------------------------------------

def test_reading_some_data_pages_holds_the_turn_to_every_data_page(att):
    ledger = C.CoverageLedger(attachments=att)
    ledger.begin_turn("1:a")
    _read(ledger, att, "16-18")                    # three lab sheets only
    armed = ledger.armed("1:a")
    assert list(armed.values())[0] == (C.DATA_GROUPS, "data")
    note = ledger.gate_note("1:a")
    assert note.startswith(C.GATE_PREFIX)
    assert "laboratory sheets: 11 of 14 not read - pages 19-29 (PDF pages " \
           "20-30)" in note
    # the logs are held too, and the scans must be LOOKED at
    assert "exploration logs: 7 of 7 not read" in note
    assert "pages 12-14 have no text layer, so they must be looked at" in note
    assert "This note comes once" in note


def test_reading_only_text_pages_holds_nothing(att):
    ledger = C.CoverageLedger(attachments=att)
    ledger.begin_turn("1:a")
    _read(ledger, att, "2-4")
    assert ledger.armed("1:a") == {} and ledger.gate_note("1:a") is None


def test_a_declared_task_covers_every_page_but_the_tabs(att):
    ledger = C.CoverageLedger(attachments=att)
    ledger.begin_turn("1:a")
    key = ledger.resolve(("source", KEY))
    ledger.declare(key)
    for p in RF.TARGET_PAGES:
        _look(ledger, p)
    note = ledger.gate_note("1:a")
    # data all read; the text pages and the plan are still owed
    assert "text pages: 3 of 3 not read" in note
    assert "plans, profiles and figures: 1 of 1 not read" in note
    assert "cover, contents and divider" not in note


def test_everything_read_then_only_a_missing_count_is_asked_for(att):
    ledger = C.CoverageLedger(attachments=att)
    ledger.begin_turn("1:a")
    key = ledger.resolve(("source", KEY))
    ledger.declare(key)
    for p in range(30):
        _look(ledger, p)
    note = ledger.gate_note("1:a", answer="Here are the tables.")
    assert note and "Every page this task covers has been read." in note
    assert ledger.gate_note("1:a", answer="All 14 lab sheets: 14 of 14 "
                                          "read.") is None
    # a data-only turn with everything read is not nagged for a count
    ledger2 = C.CoverageLedger(attachments=att)
    ledger2.begin_turn("1:b")
    for p in RF.TARGET_PAGES:
        _look(ledger2, p)
    assert ledger2.gate_note("1:b", answer="Here.") is None


def test_skipped_with_a_reason_is_accounted_for(att):
    ledger = C.CoverageLedger(attachments=att)
    ledger.begin_turn("1:a")
    _read(ledger, att, "7-10,16-29")
    key = next(iter(ledger.docs))
    ledger.mark(key, [12, 13, 14], "skipped", "the 2011 logs are superseded")
    assert ledger.gate_note("1:a", answer="7 of 7 logs") is None


def test_reads_of_an_earlier_turn_count_but_do_not_arm(att):
    ledger = C.CoverageLedger(attachments=att)
    ledger.begin_turn("1:a")
    _read(ledger, att, "16-29")
    ledger.begin_turn("2:b")                       # a later, unrelated turn
    assert ledger.armed("2:b") == {}
    _look(ledger, 7)                               # now a log is read
    note = ledger.gate_note("2:b")
    assert "laboratory sheets:" not in note         # read in turn 1...
    assert "laboratory sheets 14 of 14 read" in note    # ...and counted
    assert "exploration logs: 6 of 7 not read" in note


def test_writing_data_out_arms_the_documents_read_and_marks_its_pages(att):
    ledger = C.CoverageLedger(attachments=att)
    ledger.begin_turn("1:a")
    _read(ledger, att, "7-10")
    params = {"investigations": [{"investigation_id": "B-1", "pages": [7, 8]},
                                 {"investigation_id": "B-2"}],
              "lab_tests": [{"kind": "atterberg", "investigation_id": "B-2",
                             "sample_id": "S-3", "pages": [18]}],
              "output_path": "x.xml"}
    ledger.record_call("call_agent", {"agent_name": "subsurface",
                                      "method": "write_diggs",
                                      "parameters": params},
                       json.dumps({"output_path": "x.xml", "verdict": "ok"}))
    key = next(iter(ledger.docs))
    assert ledger.armed("1:a")[key] == (C.TARGET_GROUPS, "output")
    marks = ledger.docs[key]["marks"]
    assert {p for p, m in marks.items() if m["status"] == "extracted"} == \
        {7, 8, 18}
    assert ledger.outputs[-1]["rows_without_pages"] == ["B-2"]
    # a failed write is not an output
    ledger.record_call("call_agent", {"agent_name": "subsurface",
                                      "method": "write_diggs",
                                      "parameters": {}},
                       json.dumps({"error": "nothing to write"}))
    assert len(ledger.outputs) == 1


def test_write_diggs_data_given_at_the_top_level_is_read_too(att):
    """call_agent hoists stray top-level keys into parameters; so does the
    ledger."""
    ledger = C.CoverageLedger(attachments=att)
    _look(ledger, 9)
    ledger.record_call("call_agent", {
        "agent_name": "subsurface", "method": "write_diggs",
        "investigations": [{"investigation_id": "B-2", "pages": [9]}],
        "parameters": {"output_path": "x.xml"}},
        json.dumps({"output_path": "x.xml"}))
    assert ledger.outputs[-1]["rows"] == 1
    assert ledger.outputs[-1]["pages_cited"] == "9"


# ---------------------------------------------------------------------------
# The middleware, the tools and the builds
# ---------------------------------------------------------------------------

def test_turn_key_ignores_the_gate_note_and_the_auto_continue():
    from webapp.core import CONTINUE_NUDGE
    assert CT.CONTINUE_NUDGE == CONTINUE_NUDGE      # kept in step
    base = [HumanMessage("extract it"), AIMessage("draft")]
    k = CT.turn_key(base)
    assert CT.turn_key(base + [HumanMessage(C.GATE_PREFIX + " read more")]) == k
    assert CT.turn_key(base + [AIMessage("let me"),
                               HumanMessage(CONTINUE_NUDGE)]) == k
    assert CT.turn_key(base + [HumanMessage("now the next question")]) != k


class _Scripted(BaseChatModel):
    """Opens the report, reads three lab sheets, answers; after the gate's
    note (``obey``) reads the rest and looks at the scans, then answers."""

    log: list = Field(default_factory=list)
    obey: bool = True
    helper: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted-coverage"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if len(messages) == 1 and isinstance(messages[0].content, list):
            return self._say("A scanned boring log.")       # a vision call
        tools = [m for m in messages if m.type == "tool"]
        humans = [m for m in messages if m.type == "human"]
        if humans and str(humans[0].content).startswith("HELPER:"):
            # the general-purpose helper: read the scans, report
            if not tools:
                return self._calls([("analyze_pdf_page",
                                     {"attachment_key": KEY, "page": p})
                                    for p in (12, 13, 14)])
            return self._say("The three scanned logs are BH-1, BH-2, BH-3.")
        gate = any(str(m.content).startswith(C.GATE_PREFIX) for m in humans)
        self.log.append(gate)
        handle = None
        for m in tools:
            if m.name == "open_document":
                handle = json.loads(m.content)["handle"]
        names = [m.name for m in tools]
        if handle is None:
            return self._calls([("open_document", {"source": KEY})])
        if "read_document" not in names:
            return self._calls([("read_document",
                                 {"handle": handle, "pages": "16-18"})])
        if self.helper and "task" not in names:
            return self._calls([("task", {
                "description": "HELPER: look at pages 12-14 of report.pdf",
                "subagent_type": "general-purpose"})])
        if not gate:
            return self._say("Here are the tables for the sheets I read.")
        if self.obey and names.count("read_document") < 2:
            calls = [("read_document", {"handle": handle,
                                        "pages": "7-10,19-29"})]
            if not self.helper:
                calls += [("analyze_pdf_page", {"attachment_key": KEY,
                                                "page": p})
                          for p in (12, 13, 14)]
            return self._calls(calls)
        return self._say("Final tables. Coverage: exploration logs 7 of 7 "
                         "read; laboratory sheets 14 of 14 read.")

    def _say(self, text):
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=text))])

    def _calls(self, calls):
        n = len(self.log)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="", tool_calls=[{"name": name, "args": args,
                                     "id": f"c{n}_{i}"}
                                    for i, (name, args) in
                                    enumerate(calls)]))])


def _build(model, att, tmp_path, page="review"):
    from funhouse_agent.review_eval.runner import build_page_agent
    files = tmp_path / "files"
    files.mkdir(parents=True, exist_ok=True)
    (files / KEY).write_bytes(att[KEY])
    return build_page_agent(model, att, str(files), [], page=page)


def _run(agent, tmp_path, text="Put the data in tables."):
    from webapp import core
    from webapp.activity_log import ActivityLogger
    logger = ActivityLogger(str(tmp_path), turn=1)
    history = [{"role": "user", "content": text}]
    answer = None
    for item in core.stream_turn(agent, history, "t1", recursion_limit=50,
                                 callbacks=[logger]):
        if item["kind"] == "turn_done":
            answer = item["answer"]
    return answer


@pytest.mark.parametrize("page", ["review", "geotech"])
def test_the_gate_tells_once_and_the_agent_reads_the_rest(att, tmp_path,
                                                         monkeypatch, page):
    from webapp.activity_log import load
    monkeypatch.setenv(review_flags.COVERAGE_ENV, "1")
    model = _Scripted()
    agent = _build(model, att, tmp_path, page)
    assert agent.geotech_extraction_recursion_limit == CT.EXTRACTION_STEP_LIMIT
    answer = _run(agent, tmp_path)
    assert model.log.count(False) >= 1 and model.log[-1] is True
    assert "14 of 14" in answer
    ledger = agent.geotech_coverage_ledger
    key = next(iter(ledger.docs))
    assert all(ledger.covered(key, p) for p in RF.TARGET_PAGES)
    gates = [r for r in load(str(tmp_path)) if r["event"] == "coverage_gate"]
    assert len(gates) == 1 and "11 of 14 not read" in gates[0]["text"]
    assert os.path.isfile(tmp_path / "coverage.json")


def test_a_model_that_ignores_the_note_is_not_asked_twice(att, tmp_path,
                                                          monkeypatch):
    from webapp.activity_log import load
    monkeypatch.setenv(review_flags.COVERAGE_ENV, "1")
    model = _Scripted(obey=False)
    agent = _build(model, att, tmp_path)
    answer = _run(agent, tmp_path)
    assert answer and model.log == [False, False, False, True]
    assert len([r for r in load(str(tmp_path))
                if r["event"] == "coverage_gate"]) == 1


def test_a_helpers_reads_count(att, tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.COVERAGE_ENV, "1")
    model = _Scripted(helper=True)
    agent = _build(model, att, tmp_path)
    _run(agent, tmp_path)
    ledger = agent.geotech_coverage_ledger
    key = next(iter(ledger.docs))
    for p in (12, 13, 14):
        assert "general-purpose" in ledger.docs[key]["reads"][p]["by"]
    assert all(ledger.covered(key, p) for p in RF.TARGET_PAGES)


def test_the_gate_stands_down_when_the_turn_is_out_of_steps(att, monkeypatch):
    ledger = C.CoverageLedger(attachments=att)
    gate = CT.CoverageGate(ledger)
    msgs = [HumanMessage("extract"), AIMessage("done")]
    gate.before_agent({"messages": msgs}, None)
    _read(ledger, att, "16-18")
    monkeypatch.setattr(CT, "_steps_left", lambda: 5)
    assert gate.after_model({"messages": msgs}, None) is None
    assert "skipped" in ledger.fired[CT.turn_key(msgs)]
    # ...and near a model-call budget it stays silent
    ledger2 = C.CoverageLedger(attachments=att)
    gate2 = CT.CoverageGate(ledger2, budget=10)
    gate2.before_agent({"messages": msgs}, None)
    _read(ledger2, att, "16-18")
    monkeypatch.setattr(CT, "_steps_left", lambda: 100)
    assert gate2.after_model({"messages": msgs,
                              "run_model_call_count": 9}, None) is None
    out = gate2.after_model({"messages": msgs, "run_model_call_count": 3},
                            None)
    assert out["jump_to"] == "model"
    assert out["messages"][0].content.startswith(C.GATE_PREFIX)
    assert gate2.after_model({"messages": msgs}, None) is None   # once


def test_a_tool_call_turn_is_never_gated(att, monkeypatch):
    ledger = C.CoverageLedger(attachments=att)
    gate = CT.CoverageGate(ledger)
    msgs = [HumanMessage("extract")]
    gate.before_agent({"messages": msgs}, None)
    _read(ledger, att, "16-18")
    calling = AIMessage(content="", tool_calls=[{"name": "x", "args": {},
                                                  "id": "1"}])
    assert gate.after_model({"messages": msgs + [calling]}, None) is None


def test_switches_off_the_build_is_unchanged(att, tmp_path):
    agent = _build(_Scripted(), att, tmp_path)
    assert not hasattr(agent, "geotech_coverage_ledger")
    names = set(agent.nodes["tools"].bound.tools_by_name)
    assert not names & {"document_coverage", "report_checklist"}


def test_the_tools_are_on_both_pages_with_the_switches(att, tmp_path,
                                                       monkeypatch):
    monkeypatch.setenv(review_flags.COVERAGE_ENV, "1")
    monkeypatch.setenv(review_flags.CHECKLIST_ENV, "1")
    for page in ("review", "geotech"):
        agent = _build(_Scripted(), att, tmp_path / page, page)
        names = set(agent.nodes["tools"].bound.tools_by_name)
        assert {"document_coverage", "report_checklist"} <= names, page


def test_the_lean_agent_and_its_helper_carry_it(att, monkeypatch):
    from funhouse_agent.deep import review_agent
    monkeypatch.setenv(review_flags.COVERAGE_ENV, "1")
    seen = {}
    real = review_agent.make_reader_tool

    def spy(model, tools, budget=14, extra_middleware=None):
        seen["mw"] = extra_middleware
        return real(model, tools, budget, extra_middleware)

    monkeypatch.setattr(review_agent, "make_reader_tool", spy)
    agent = review_agent.build_review_agent(_Scripted(), attachments=att)
    assert isinstance(agent.geotech_coverage_ledger, C.CoverageLedger)
    assert any(isinstance(m, CT.CoverageRecorder) for m in seen["mw"])
    names = set(agent.nodes["tools"].bound.tools_by_name)
    assert "document_coverage" in names


def test_every_helper_of_the_deep_build_records(att, monkeypatch):
    from funhouse_agent.deep import agent as A
    monkeypatch.setenv(review_flags.COVERAGE_ENV, "1")
    captured = {}
    real = A.create_deep_agent

    def spy(**kw):
        captured.update(kw)
        return real(**kw)

    monkeypatch.setattr(A, "create_deep_agent", spy)
    A.build_deep_agent(_Scripted(), attachments=att, enable_calc_subagent=True)
    specs = captured["subagents"]
    assert {s["name"] for s in specs} >= {"references", "reviewer", "calc",
                                          "general-purpose"}
    for s in specs:
        assert any(isinstance(m, CT.CoverageRecorder)
                   and not isinstance(m, CT.CoverageGate)
                   for m in s["middleware"]), s["name"]
    assert any(isinstance(m, CT.CoverageGate)
               for m in captured["middleware"])


def test_document_coverage_tool(att):
    ledger = C.CoverageLedger(attachments=att)
    tool = CT.make_coverage_tool(ledger)
    out = json.loads(tool.invoke({"source": KEY}))
    assert out["n_pages"] == 30 and "statement" in out
    assert {g["group"] for g in out["groups"]} >= {"logs", "lab"}
    key = next(iter(ledger.docs))
    assert ledger.docs[key]["declared"] is not None
    err = json.loads(tool.invoke({"source": KEY, "skipped": "12-14"}))
    assert "reason" in err["error"]
    out = json.loads(tool.invoke({"source": KEY, "skipped": "12-14",
                                  "reason": "superseded",
                                  "extracted": [16]}))
    assert out["marked"] == {"extracted": "16", "skipped": "12-14"}
    assert json.loads(tool.invoke({"source": "nope.pdf"}))["error"]
    assert "every page any tool or helper" in \
        CT.DOCUMENT_COVERAGE_DESCRIPTION.replace("tool or helper reads",
                                                 "tool or helper")


# ---------------------------------------------------------------------------
# The checklist
# ---------------------------------------------------------------------------

def test_the_checklist_is_a_draft_of_data():
    data = RC.load_checklist()
    assert data["status"] == "DRAFT"
    items = data["items"]
    assert len({i["id"] for i in items}) == len(items)
    assert {i["section"] for i in items} == {
        "Completeness", "Consistency", "Currency", "Traceability",
        "Plausibility", "Recommendations"}
    for i in items:
        assert i["mode"] in ("code", "judge") and i["text"]
        if i["mode"] == "code":
            assert i["check"] in RC.CHECKS, i["id"]
    assert sum(i["mode"] == "code" for i in items) == 7
    assert sum(i["mode"] == "judge" for i in items) == 11


def test_the_checklist_ships_in_the_wheel():
    text = open(os.path.join(os.path.dirname(RC.__file__), "..",
                             "pyproject.toml"), encoding="utf-8").read()
    assert "report_review_checklist.json" in text


def test_exploration_ids():
    assert RC.exploration_ids("Borings B-1, B-2 and BH3; TP-12 and CPT 4") \
        == {"B-1", "B-2", "BH-3", "TP-12", "CPT-4"}
    assert RC.exploration_ids("Table B-1, Figure B-2, Appendix B-3") == set()
    assert RC.exploration_ids("Sample S-3 of B12 in Zone B-4") == set()


def test_checklist_on_the_report(att):
    ledger = C.CoverageLedger(attachments=att)
    for p in range(30):
        _look(ledger, p)
    key = next(iter(ledger.docs))
    res = RC.run_checklist(ledger, key)
    by = {r["id"]: r for r in res["code_checks"]}
    assert by["completeness.pages_read"]["status"] == "pass"
    # the 2011 logs are scans: code cannot see their ids, so it asks for a look
    ex = by["completeness.explorations_have_logs"]
    assert ex["status"] == "unsure" and "BH-1" in ex["detail"]
    assert by["consistency.summary_vs_sheets"]["status"] == "not_run"
    judge = {r["id"] for r in res["for_you_to_judge"]}
    assert "completeness.groundwater" in judge
    assert "completeness.explorations_have_logs" in judge   # unsure -> judge
    assert "consistency.summary_vs_sheets" in judge         # not run -> judge
    assert res["status"] == "DRAFT"


def test_checklist_reads_the_cross_checks_of_the_data_written(att, tmp_path):
    """The real write_diggs (W2: the reconciler's cross-checks), fed the
    fixture's planted error, through the ledger and into the checklist."""
    from funhouse_agent.dispatch import call_agent
    ledger = C.CoverageLedger(attachments=att)
    for p in RF.TARGET_PAGES:
        _look(ledger, p)
    params = {
        "investigations": [{
            "investigation_id": "B-2", "kind": "boring", "depth_unit": "m",
            "total_depth": {"value": 7.4, "unit": "m"},
            "samples": [{"sample_id": "S-3", "kind": "spt",
                         "top": {"value": 4.0, "unit": "m"}}],
            "pages": [9]}],
        "lab_tests": [
            {"kind": "atterberg", "investigation_id": "B-2",
             "sample_id": "S-3", "depth_top": {"value": 4.0, "unit": "m"},
             "pages": [18], "result": {"kind": "atterberg", "ll": 32,
                                       "pl": 20, "pi": 12}},
            {"kind": "atterberg", "investigation_id": "B-9",
             "sample_id": "S-1", "pages": [17],
             "result": {"kind": "atterberg", "ll": 30, "pl": 18, "pi": 12}},
            {"kind": "summary_table", "pages": [16], "result": {
                "kind": "summary_table", "rows": [
                    {"investigation_id": "B-2", "sample_id": "S-3",
                     "depth_top": {"value": 4.0, "unit": "m"},
                     "ll": 20, "pl": 32, "pi": 12}]}}],
        "output_path": str(tmp_path / "x.diggs.xml")}
    result = call_agent("subsurface", "write_diggs", params)
    ledger.record_call("call_agent", {"agent_name": "subsurface",
                                      "method": "write_diggs",
                                      "parameters": params}, result)
    key = next(iter(ledger.docs))
    res = RC.run_checklist(ledger, key)
    by = {r["id"]: r for r in res["code_checks"]}
    sv = by["consistency.summary_vs_sheets"]
    assert sv["status"] == "fail"
    assert any("summary table 20" in e and "atterberg sheet 32" in e
               for e in sv["entries"])
    assert by["consistency.lab_ids_on_logs"]["status"] == "fail"
    assert "B-9" in by["consistency.lab_ids_on_logs"]["detail"]
    sh = by["completeness.summary_and_sheets"]
    assert sh["status"] == "fail" and "B-9 S-1" in sh["detail"]
    assert by["traceability.values_cite_pages"]["status"] == "pass"
    assert by["consistency.units"]["status"] == "pass"
    lines = RC.failed_lines(ledger, key)
    assert any("Summary-table values" in ln for ln in lines)


def test_register_check_is_the_hook(att):
    ledger = C.CoverageLedger(attachments=att)
    _look(ledger, 7)
    key = next(iter(ledger.docs))
    data = {"title": "t", "status": "DRAFT", "items": [
        {"id": "x", "section": "S", "mode": "code", "check": "custom_hook",
         "text": "a check the lead wires later"}]}
    res = RC.run_checklist(ledger, key, data)
    assert res["code_checks"][0]["status"] == "not_run"
    assert res["for_you_to_judge"][0]["id"] == "x"
    RC.register_check("custom_hook", lambda ctx, item: {
        "status": "pass", "detail": "wired"})
    try:
        res = RC.run_checklist(ledger, key, data)
        assert res["code_checks"][0] == {
            "id": "x", "section": "S", "text": "a check the lead wires later",
            "status": "pass", "detail": "wired"}
        assert res["for_you_to_judge"] == []
    finally:
        RC.CHECKS.pop("custom_hook", None)


def test_the_checklist_tool_and_the_gate_carry_failed_code_checks(att,
                                                                   monkeypatch):
    ledger = C.CoverageLedger(attachments=att)
    tool = CT.make_checklist_tool(ledger)
    out = json.loads(tool.invoke({"source": KEY}))
    assert out["status"] == "DRAFT" and out["for_you_to_judge"]
    gate = CT.CoverageGate(ledger, checklist=True)
    msgs = [HumanMessage("extract"), AIMessage("done")]
    gate.before_agent({"messages": msgs}, None)
    ledger.outputs.append({"seq": ledger.next_seq(), "rows": 1,
                           "rows_without_pages": ["B-1"], "pages_cited": "",
                           "cross_checks": {"n": 0, "by_kind": {},
                                            "entries": []}})
    for p in RF.TARGET_PAGES:
        _look(ledger, p)
    monkeypatch.setattr(CT, "_steps_left", lambda: 100)
    out = gate.after_model({"messages": msgs}, None)
    note = out["messages"][0].content
    assert "Every extracted value cites its page" in note


# -- the step limits (owner, 2026-10-08) ------------------------------------
# An ordinary turn keeps the app's own cap; a turn that takes data out of a
# document may run to EXTRACTION_STEP_LIMIT (150).

class _Req:
    def __init__(self, messages, tool_call=None):
        self.state = {"messages": messages}
        self.messages = list(messages)
        self.tool_call = tool_call or {}

    def override(self, **kw):
        out = _Req(kw.get("messages", self.messages), self.tool_call)
        out.tools = kw.get("tools")
        return out


def _gate(att):
    ledger = C.CoverageLedger(attachments=att)
    return ledger, CT.CoverageGate(ledger)


def _at(monkeypatch, allowance, used):
    monkeypatch.setattr(CT, "_step_allowance", lambda: allowance)
    monkeypatch.setattr(CT, "_steps_used", lambda: used)


def test_an_ordinary_turn_gets_its_last_call_at_its_own_cap(att, monkeypatch):
    _ledger, gate = _gate(att)
    msgs = [HumanMessage(content="What is this report about?")]
    seen = []
    _at(monkeypatch, 50, 30)
    gate.wrap_model_call(_Req(msgs), lambda r: seen.append(r))
    assert getattr(seen[-1], "tools", "untouched") == "untouched"
    _at(monkeypatch, 50, 48)
    gate.wrap_model_call(_Req(msgs), lambda r: seen.append(r))
    assert seen[-1].tools == []
    assert CT.ALLOWANCE_NUDGE in seen[-1].messages[-1].content


def test_an_extraction_turn_runs_past_the_ordinary_cap(att, monkeypatch):
    ledger, gate = _gate(att)
    msgs = [HumanMessage(content="Put every boring log in a table.")]
    ledger.begin_turn(CT.turn_key(msgs))
    _read(ledger, att, [RF.TARGET_PAGES[0]])          # a data page read
    assert ledger.armed(CT.turn_key(msgs))
    seen = []
    _at(monkeypatch, 50, 120)
    gate.wrap_model_call(_Req(msgs), lambda r: seen.append(r))
    assert getattr(seen[-1], "tools", "untouched") == "untouched"


def test_calling_report_ingest_makes_it_an_extraction_turn(att, monkeypatch):
    _ledger, gate = _gate(att)
    msgs = [HumanMessage(content="Ingest this report.")]
    from langchain_core.messages import ToolMessage
    gate.wrap_tool_call(_Req(msgs, {"name": "report_ingest", "args": {}}),
                        lambda r: ToolMessage(content="{}", tool_call_id="x"))
    assert gate.is_extraction(msgs)
    seen = []
    _at(monkeypatch, 50, 100)
    gate.wrap_model_call(_Req(msgs), lambda r: seen.append(r))
    assert getattr(seen[-1], "tools", "untouched") == "untouched"


def test_no_allowance_means_no_change(att, monkeypatch):
    _ledger, gate = _gate(att)
    msgs = [HumanMessage(content="Hello")]
    seen = []
    _at(monkeypatch, None, 500)
    gate.wrap_model_call(_Req(msgs), lambda r: seen.append(r))
    assert getattr(seen[-1], "tools", "untouched") == "untouched"


def test_the_web_app_runs_the_turn_under_the_extraction_cap():
    from webapp import core

    class _Agent:
        geotech_extraction_recursion_limit = CT.EXTRACTION_STEP_LIMIT

        def __init__(self):
            self.configs = []

        def stream(self, payload, config=None, stream_mode=None):
            self.configs.append(config)
            return iter(())

    agent = _Agent()
    list(core.stream_turn(agent, [{"role": "user", "content": "hi"}], "t1",
                          recursion_limit=50))
    cfg = agent.configs[0]
    assert cfg["recursion_limit"] == CT.EXTRACTION_STEP_LIMIT
    assert cfg["configurable"][CT.ALLOWANCE_KEY] == 50

    plain = _Agent()
    plain.geotech_extraction_recursion_limit = None
    list(core.stream_turn(plain, [{"role": "user", "content": "hi"}], "t1",
                          recursion_limit=50))
    assert plain.configs[0]["recursion_limit"] == 50
    assert CT.ALLOWANCE_KEY not in plain.configs[0]["configurable"]


# ---------------------------------------------------------------------------
# Foundry brief 5 (module_work/review_eval_results/2026-10-08_foundry_
# 5.33.0rc1/TRACE_REVIEW.md, "The fix list" group 2)
# ---------------------------------------------------------------------------
# CV2 / N4: every gated answer reached the user as the reply before the
# gate's note glued to the reply after it. The gate now takes the first
# reply out of the conversation (the model writes its answer once, whole)
# and the web app delivers only the reply after the note.

from types import SimpleNamespace  # noqa: E402

from langchain_core.messages import (AIMessageChunk, RemoveMessage,  # noqa: E402
                                     ToolMessage)

DRAFT = "Sheet B-2 S-3: LL 32, PL 20 (PDF page 19)."
ADDENDUM = "Coverage: laboratory sheets 3 of 14 read; the rest not needed."
WHOLE = DRAFT + "\n\n" + ADDENDUM


class _Drafter(BaseChatModel):
    """Reads three lab sheets and answers. After the gate's note it does what
    brief 5's Sol did in 4 of 11 gated runs: if its first reply is still in
    the conversation it writes only an addendum; if not, the whole answer."""

    seen_draft_after_note: list = Field(default_factory=list)
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-drafter"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.calls += 1
        tools = [m for m in messages if m.type == "tool"]
        humans = [m for m in messages if m.type == "human"]
        gate = any(str(m.content).startswith(C.GATE_PREFIX) for m in humans)
        handle = None
        for m in tools:
            if m.name == "open_document":
                handle = json.loads(m.content)["handle"]
        if handle is None:
            return _reply(calls=[("open_document", {"source": KEY})])
        if not any(m.name == "read_document" for m in tools):
            return _reply(calls=[("read_document",
                                  {"handle": handle, "pages": "16-18"})])
        if not gate:
            return _reply(DRAFT)
        draft_kept = any(m.type == "ai" and DRAFT in str(m.content)
                         for m in messages)
        self.seen_draft_after_note.append(draft_kept)
        return _reply(ADDENDUM if draft_kept else WHOLE)


def _reply(text="", calls=()):
    return ChatResult(generations=[ChatGeneration(message=AIMessage(
        content=text, tool_calls=[{"name": n, "args": a, "id": f"c{i}_{n}"}
                                  for i, (n, a) in enumerate(calls)]))])


@pytest.mark.parametrize("page", ["review", "geotech"])
def test_a_gated_turn_delivers_one_whole_answer(att, tmp_path, monkeypatch,
                                                page):
    from webapp.activity_log import load
    monkeypatch.setenv(review_flags.COVERAGE_ENV, "1")
    model = _Drafter()
    agent = _build(model, att, tmp_path, page)
    answer = _run(agent, tmp_path)
    # the model never saw its first reply after the note...
    assert model.seen_draft_after_note == [False]
    # ...so it wrote the answer whole, and that is what the user gets, once
    assert answer == WHOLE
    assert answer.count("LL 32") == 1
    records = load(str(tmp_path))
    gates = [r for r in records if r["event"] == "coverage_gate"]
    assert len(gates) == 1
    assert "held back" in gates[0]["text"]
    assert "write your whole answer" in gates[0]["text"]
    # the held-back reply stays in the record of the run
    said = [r.get("text") for r in records if r["event"] == "model_end"]
    assert DRAFT in said and WHOLE in said


def _chunk(text, node="model"):
    return ("messages", (AIMessageChunk(content=text),
                         {"langgraph_node": node}))


class _Replay:
    """A compiled agent's stream, replayed: the shape of a gated pass as
    LangGraph emits it (the reply streamed, the model node's update, the
    gate's update taking the reply out and adding its note, a tool round,
    the reply after the note)."""

    def __init__(self, draft, after, tool_round=True):
        self.draft, self.after, self.tool_round = draft, after, tool_round

    def stream(self, payload, config=None, stream_mode=None):
        note = C.GATE_PREFIX + " Before your answer goes to the user ..."
        for word in self.draft.split(" "):
            yield _chunk(word + " ")
        yield ("updates", {"model": {"messages": [
            AIMessage(content=self.draft, id="draft-1")]}})
        yield ("messages", (HumanMessage(content=note),
                            {"langgraph_node": "CoverageGate.after_model"}))
        yield ("updates", {"CoverageGate.after_model": {
            "messages": [RemoveMessage(id="draft-1"),
                         HumanMessage(content=note)],
            "jump_to": "model"}})
        if self.tool_round:
            yield ("updates", {"model": {"messages": [AIMessage(
                content="", tool_calls=[{"name": "analyze_pdf_page",
                                         "args": {"page": 5}, "id": "t1"}])]}})
            yield ("updates", {"tools": {"messages": [ToolMessage(
                content='{"page": 5}', tool_call_id="t1",
                name="analyze_pdf_page")]}})
        for word in self.after.split(" ") if self.after else []:
            yield _chunk(word + " ")
        if self.after:
            yield ("updates", {"model": {"messages": [
                AIMessage(content=self.after)]}})


def _stream(agent):
    from webapp import core
    items = list(core.stream_turn(agent, [{"role": "user", "content": "q"}],
                                  "t1", recursion_limit=50))
    done = [i for i in items if i["kind"] == "turn_done"][0]
    return items, done["answer"]


@pytest.mark.parametrize("draft,after", [
    # brief 5's common shape: the reply, then the whole answer again with
    # the coverage counts in it
    ("Allowable pressure 398.4 kPa, matching page 4.",
     "Allowable pressure 398.4 kPa, matching page 4. Coverage: calculation "
     "pages 1 of 1 read."),
    # the run-on shape: a short reply, then a reply that starts mid-thought
    ("Page 8 duplicates page 3.",
     "Yes. Page 8 duplicates page 3; their text and footer match. Coverage: "
     "exploration logs 3 of 3 read."),
])
def test_stream_turn_replays_a_gated_pass(draft, after):
    from webapp import core
    items, answer = _stream(_Replay(draft, after))
    assert answer.strip() == after
    assert draft not in answer.replace(after, "")
    # the user is told what happened, and the live text does not run on
    assert {"kind": "tool_call", "text": core.GATE_STATUS} in items
    live = "".join(i["text"] for i in items if i["kind"] == "token")
    assert "\n\n" in live and live.index("\n\n") > live.index(draft[:10])


def test_nothing_after_the_note_keeps_the_held_back_reply():
    _items, answer = _stream(_Replay("The answer as first written.", ""))
    assert answer.strip() == "The answer as first written."


def test_an_ungated_turn_is_assembled_as_before():
    from webapp import core

    class _Plain:
        def stream(self, payload, config=None, stream_mode=None):
            yield _chunk("One ")
            yield _chunk("answer.")
            yield ("updates", {"model": {"messages": [
                AIMessage(content="One answer.")]}})

    items, answer = _stream(_Plain())
    assert answer == "One answer."
    assert not any(i.get("text") == core.GATE_STATUS for i in items)
    assert not core.coverage_gate_spoke({"model": {"messages": [
        HumanMessage(content="an ordinary user message")]}})
    assert not core.coverage_gate_spoke({"CoverageGate.after_model": None})


# CV1: only strong role evidence makes a page a log, a lab sheet or a
# field-test sheet; weak roles never arm the gate or become its targets.

def _summary(page, kind="text", chars=900):
    return SimpleNamespace(page=page, kind=kind, n_text_chars=chars,
                           text_reliable=True)


def _role(page, role, conf, tag="page-title", item=None, why=""):
    return SimpleNamespace(page=page, role=role, confidence=conf,
                           evidence={"tag": tag, "why": why}, item_id=item)


def test_weak_and_lone_roles_are_not_data_pages():
    """The shape of a criteria manual: many pages the rules call laboratory
    sheets on weak evidence, and one strong one alone among prose."""
    sums = [_summary(p) for p in range(12)]
    roles = [_role(p, "narrative", 0.85, "narrative-block")
             for p in range(12)]
    for p in (2, 3, 6, 7, 10):
        roles[p] = _role(p, "lab_test", 0.6,
                         why="laboratory test name on a page of working")
    roles[5] = _role(5, "lab_test", 0.9, why="laboratory test title")
    roles[8] = _role(8, "boring_log", 0.6, "tab-declares")   # text page
    pages = C.classify_pages(sums, roles)
    assert [p.page for p in pages if p.group in C.DATA_GROUPS] == []
    by = {p.page: p for p in pages}
    assert by[2].group == "other" and by[2].role == "lab_test"
    assert by[2].not_data.startswith("weak evidence (planlens 0.60)")
    assert "says nothing about itself" in by[8].not_data
    assert by[5].not_data.startswith("stands alone")


def test_strong_roles_in_a_set_are_data_pages():
    """The shape of a report: logs that name themselves, a log's
    continuation sheet, older logs as scans behind their tab, a laboratory
    appendix behind a divider."""
    sums = [_summary(p, "mixed", 400) for p in range(14)]
    roles = [_role(0, "cover", 0.8, "cover"),
             _role(1, "narrative", 0.85, "narrative-block"),
             _role(2, "divider", 0.85, "tab"),
             _role(3, "boring_log", 0.9, item="i1"),
             _role(4, "boring_log", 0.6, item="i1",
                   why="log form shape, 7 log form fields"),
             _role(5, "boring_log", 0.9, item="i2"),
             _role(6, "divider", 0.85, "tab"),
             _role(7, "boring_log", 0.6, "tab-declares", item="i3"),
             _role(8, "boring_log", 0.6, "tab-declares", item="i3"),
             _role(9, "divider", 0.85, "tab"),
             _role(10, "lab_test", 0.8, item="i4"),
             _role(11, "lab_test", 0.8, item="i5"),
             _role(12, "divider", 0.85, "tab"),
             _role(13, "lab_test", 0.9, item="i6")]
    for p in (7, 8):
        sums[p] = _summary(p, "scanned", 0)
    pages = C.classify_pages(sums, roles)
    data = {p.page: p.group for p in pages if p.group in C.DATA_GROUPS}
    assert data == {3: "logs", 4: "logs", 5: "logs", 7: "logs", 8: "logs",
                    10: "lab", 11: "lab", 13: "lab"}
    assert all(p.not_data is None for p in pages)


def test_the_report_fixture_keeps_every_data_page(att, report):
    ledger = C.CoverageLedger(attachments=att)
    key = ledger.resolve(("source", KEY))
    inv = ledger.docs[key]["inventory"]
    assert sorted(inv.targets(C.DATA_GROUPS)) == sorted(report.target_pages)
    assert [p.page for p in inv.pages if p.not_data] == []
    assert inv.rules == C.INVENTORY_RULES
    # the scans are held by their tab: a scan cannot name itself
    assert {inv.pages[p].evidence for p in (12, 13, 14)} == {"tab-declares"}


def test_a_page_that_inherits_its_role_but_says_nothing_is_not_held():
    """brief 5 fixture-duplicate-page: the two log forms were read, and the
    gate then asked for the duplicated TEXT page its tab called a log."""
    from funhouse_agent.review_eval import documents as D
    name, data = D.resolve("fixture_submittal")
    ledger = C.CoverageLedger(attachments={name: data})
    ledger.begin_turn("1:a")
    for p in (5, 6, 7):
        _look(ledger, p, key=name)
    key = next(iter(ledger.docs))
    inv = ledger.docs[key]["inventory"]
    assert inv.targets(C.DATA_GROUPS) == [5, 6]
    assert inv.pages[7].group == "other" and inv.pages[7].role == "boring_log"
    assert ledger.gate_note("1:a", answer="Page 8 duplicates page 3.") is None


def test_an_inventory_saved_under_older_rules_is_rebuilt(att, tmp_path):
    path = C.ledger_path(str(tmp_path))
    ledger = C.CoverageLedger(path=path, attachments=att)
    _look(ledger, 12)
    key = next(iter(ledger.docs))
    data = json.loads(open(path, encoding="utf-8").read())
    data["documents"][key]["inventory"].pop("rules")
    for p in data["documents"][key]["inventory"]["pages"]:
        p["group"] = "lab"                         # the old rules' mistake
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(data))
    again = C.CoverageLedger(path=path, attachments=att)
    assert again.docs[key]["inventory"] is None    # not trusted
    assert 12 in again.docs[key]["reads"]          # the reads are kept
    _look(again, 13)                               # touched: rebuilt
    inv = again.docs[key]["inventory"]
    assert inv.rules == C.INVENTORY_RULES and inv.pages[2].group != "lab"


# CV3: a declaration has a page scope, and it stays put across calls.

def test_a_declared_task_is_held_to_its_own_pages(att):
    ledger = C.CoverageLedger(attachments=att)
    ledger.begin_turn("1:a")
    tool = CT.make_coverage_tool(ledger)
    out = json.loads(tool.invoke({"source": KEY, "pages": "16-29"}))
    assert out["task"] == {"open": True, "pages": "16-29",
                           "pages_pdf": "17-30"}
    assert out["task_pages_pdf"] == "17-30"
    assert {g["group"] for g in out["groups"]} == {"lab"}
    for p in range(16, 28):
        _look(ledger, p)
    note = ledger.gate_note("1:a", answer="The summary disagrees.")
    assert "laboratory sheets: 2 of 14 not read - pages 28-29" in note
    # nothing outside the task's pages is asked for
    for word in ("exploration logs", "text pages", "plans, profiles"):
        assert word not in note
    assert "(the task's pages: PDF pages 17-30)" in note
    assert "Coverage of report.pdf (PDF pages 17-30): laboratory sheets " \
           "12 of 14 read" in note
    _look(ledger, 28)
    _look(ledger, 29)
    assert ledger.gate_note("1:a", answer="laboratory sheets 14 of 14 "
                                          "read") is None


def test_marking_pages_does_not_declare_or_move_the_task(att):
    ledger = C.CoverageLedger(attachments=att)
    ledger.begin_turn("1:a")
    tool = CT.make_coverage_tool(ledger)
    # a call that only marks pages opens no task
    out = json.loads(tool.invoke({"source": KEY, "skipped": "12-14",
                                  "reason": "superseded"}))
    key = next(iter(ledger.docs))
    assert ledger.docs[key]["declared"] is None
    assert out["task"] == {"open": False}
    assert ledger.armed("1:a") == {}
    # the task, scoped; then marks and a status call leave it where it is
    tool.invoke({"source": KEY, "pages": "7-14"})
    declared = ledger.docs[key]["declared"]
    out = json.loads(tool.invoke({"source": KEY, "extracted": "7-10"}))
    assert out["marked"] == {"extracted": "7-10"}
    out = json.loads(tool.invoke({"source": KEY}))
    assert ledger.docs[key]["declared"] == declared
    assert ledger.scope(key) == list(range(7, 15))
    assert out["task"]["pages"] == "7-14"
    # naming pages moves it
    tool.invoke({"source": KEY, "pages": [7, 8, 9, 10]})
    assert ledger.scope(key) == [7, 8, 9, 10]
    # a new user turn's task starts over: the whole document
    ledger.begin_turn("2:b")
    assert not ledger.declared_this_turn(key)
    out = json.loads(tool.invoke({"source": KEY}))
    assert ledger.scope(key) is None
    assert out["task"]["pages"] == "whole document"
    assert ledger.armed("2:b")[key] == (C.TARGET_GROUPS, "declared")


def test_marks_given_in_other_shapes_are_read_or_reported(att):
    ledger = C.CoverageLedger(attachments=att)
    tool = CT.make_coverage_tool(ledger)
    out = json.loads(tool.invoke({"source": KEY, "extracted": {
        "logs": "7-10,12-14", "lab": [16, 17]}}))
    assert out["marked"] == {"extracted": "7-10,12-14,16-17"}
    out = json.loads(tool.invoke({"source": KEY, "extracted": [
        {"pages": [18, 19], "group": "lab"}]}))
    assert out["marked"] == {"extracted": "18-19"}
    out = json.loads(tool.invoke({"source": KEY, "extracted": "the logs"}))
    assert "the logs" in out["marks_not_read"]
    err = json.loads(tool.invoke({"source": KEY, "pages": "chapter 12"}))
    assert "pages=" in err["error"]


# CV4: the tool describes itself - what it is for, and what it costs.

def test_the_coverage_tool_says_what_it_is_for_and_what_it_costs(att):
    text = CT.DOCUMENT_COVERAGE_DESCRIPTION
    assert "takes the data out of many of its pages" in text
    assert "What it costs" in text
    assert ("A question answered from a few pages, a search or a list of "
            "titles does not need it.") in text
    assert "whole-set review" not in text and "reviews all of it" not in text
    tool = CT.make_coverage_tool(C.CoverageLedger(attachments=att))
    assert "pages" in tool.args


# N6: a page whose rows log_grid returned was read; measure is not a read.

def _grid(pages, rows, nxt=None):
    out = {"handle": "doc_g", "pages": pages,
           "rows": [{"page": p, "text": "x"} for p in rows]}
    if nxt is not None:
        out["next_offset"] = nxt
    return json.dumps(out)


def test_log_grid_rows_returned_in_full_are_reads():
    def got(args, result):
        ref, pages = C.pages_from_call("log_grid", args, result)
        assert ref in (None, ("handle", "doc_g"))
        return [p for p, how in pages if how == "text"]

    span = [7, 8, 9, 10, 12, 13, 14]
    a = {"handle": "doc_g", "pages": "7-10,12-14"}
    # windows of rows, in page order; a page may go on into the next window
    assert got(a, _grid(span, [7] * 24, 24)) == []
    assert got(dict(a, offset=24), _grid(span, [7] * 6 + [8] * 17, 47)) == [7]
    # a page whose rows end exactly at a window's end completes in the next
    assert got(dict(a, offset=47), _grid(span, [9] * 24, 71)) == [7, 8]
    assert got(dict(a, offset=71), _grid(span, [9] * 6 + [10] * 20)) == \
        [7, 8, 9, 10]
    # one call that returns every row
    assert got({"handle": "doc_g", "pages": "9"}, _grid([9], [9] * 30)) == [9]
    # no rows (rows=false; a scan's grid with no text): no page's content
    assert got({"handle": "doc_g", "pages": "12-14"},
               _grid([12, 13, 14], [])) == []
    assert got(a, json.dumps({"handle": "doc_g", "pages": span,
                              "n_rows": 60})) == []
    assert got(a, json.dumps({"error": "no such handle"})) == []
    # measure answers one question about a page: not a read
    assert C.pages_from_call("measure", {"source": "doc_g", "page": 0},
                             json.dumps({"page": 0, "value": {}})) == \
        (None, [])


def test_log_grid_reads_count_in_the_ledger(att):
    """The real tool over the report fixture's logs, paged to the end: the
    ledger counts the text pages read, and the scans still want a look."""
    from funhouse_agent import document_tools
    if not document_tools.has_tool("log_grid"):
        pytest.skip("this planlens has no log_grid")
    ledger = C.CoverageLedger(attachments=att)
    ledger.begin_turn("1:a")
    h = _handle(att)
    args = {"handle": h, "pages": "7-10,12-14", "rows": True}
    for _ in range(20):
        out = document_tools.dispatch_document_tool("log_grid", args,
                                                    attachments=att,
                                                    max_chars=14000)
        ledger.record_call("log_grid", dict(args), out)
        nxt = json.loads(out).get("next_offset")
        if not nxt:
            break
        args = dict(args, offset=nxt)
    key = next(iter(ledger.docs))
    assert [p for p in (7, 8, 9, 10) if ledger.covered(key, p)] == \
        [7, 8, 9, 10]
    assert not any(ledger.covered(key, p) for p in (12, 13, 14))
    note = ledger.gate_note("1:a")
    assert "exploration logs: 3 of 7 not read - pages 12-14" in note
