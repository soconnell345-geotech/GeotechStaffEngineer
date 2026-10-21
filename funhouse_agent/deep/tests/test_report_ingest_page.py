"""Report ingest on the GeotechStaffEngineer page (BACKLOG RI-35), offline.

The page builds a report's record on demand, through the ``report_ingest``
sub-agent, behind ``GEOTECH_REPORT_INGEST`` (OFF by default); with
``GEOTECH_REVIEW_CHECKLIST`` on as well, the report review checklist's code
checks run on that record. Everything is written to the conversation's own
folder.

One turn, the way the app runs it: the page's agent built as
``webapp/app.py`` builds it, the conversation's files folder bound to the
turn as the turn worker binds it, a scripted primary model that delegates the
synthetic public report (``report_ingest.tests.narrative_fixtures``) and then
asks the checklist for its judgement items, and the ingest on the scripted
engine the ingest's own tests use.
"""

import json
import os

import pytest

pytest.importorskip("planlens.document.roles")
pytest.importorskip("pymupdf")
pytest.importorskip("langgraph")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import Field  # noqa: E402

from funhouse_agent import review_flags  # noqa: E402
from report_ingest.tests.fake_engine import FakeEngine  # noqa: E402
from report_ingest.tests.narrative_fixtures import (  # noqa: E402
    build_narrative_report,
)
from report_ingest.tests.test_graph import full_script  # noqa: E402

KEY = "SYN.pdf"


@pytest.fixture(autouse=True)
def _switches(monkeypatch):
    for env in review_flags.ALL_ENVS + review_flags.SETTINGS_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


@pytest.fixture(scope="module")
def report_pdf():
    return build_narrative_report().pdf


@pytest.fixture
def engine(monkeypatch):
    """The ingest's engine: the page's own model is no ingest engine here
    (BACKLOG RI-03), so the one the app would dig out is the scripted one."""
    from report_ingest import engine as engine_mod

    fake = FakeEngine(full_script())
    monkeypatch.setattr(engine_mod, "engine_for",
                        lambda obj, model=None, meter=None: fake)
    return fake


class _Primary(BaseChatModel):
    """Delegates the report, asks the checklist about the record, answers."""

    seen: list = Field(default_factory=list)
    #: each tool's result as this model was shown it
    results: dict = Field(default_factory=dict)
    turns: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-report-ingest"

    def bind_tools(self, tools, **kw):
        self.seen.append(sorted(getattr(t, "name", None)
                                or t.get("name", "") for t in tools))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        results = {m.name: m.content for m in messages if m.type == "tool"}
        self.results.update(results)
        if "task" not in results:
            return self._calls([("task", {
                "description": f"Read {KEY} into its record.",
                "subagent_type": "report_ingest"})])
        if "report_checklist" not in results:
            # deepagents hands back the helper's structured response
            record = json.loads(results["task"])["paths"]["record"]
            return self._calls([("report_checklist", {"record": record})])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="The report is read; the checklist is above."))])

    def _calls(self, calls):
        self.turns += 1
        n = self.turns
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="", tool_calls=[{"name": name, "args": args,
                                     "id": f"c{n}_{i}"}
                                    for i, (name, args) in
                                    enumerate(calls)]))])


def _build(model, report_pdf, tmp_path, page="geotech"):
    from funhouse_agent.review_eval.runner import build_page_agent

    files = tmp_path / "conversation" / "files"
    files.mkdir(parents=True, exist_ok=True)
    (files / KEY).write_bytes(report_pdf)
    agent = build_page_agent(model, {KEY: report_pdf}, str(files), [],
                             page=page)
    return agent, files


def _run(agent, files, tmp_path):
    from funhouse_agent._fileio import working_dir_bound
    from webapp import core
    from webapp.activity_log import ActivityLogger

    logger = ActivityLogger(str(tmp_path), turn=1)
    history = [{"role": "user", "content": f"Read {KEY} and check it."}]
    answer = None
    with working_dir_bound(str(files)):
        for item in core.stream_turn(agent, history, "t1",
                                     recursion_limit=50, callbacks=[logger]):
            if item["kind"] == "turn_done":
                answer = item["answer"]
    return answer


def _tool_names(agent):
    return set(agent.nodes["tools"].bound.tools_by_name)


def test_a_turn_reads_the_report_and_checks_its_record(report_pdf, engine,
                                                       tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.REPORT_INGEST_ENV, "1")
    monkeypatch.setenv(review_flags.CHECKLIST_ENV, "1")
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    model = _Primary()
    agent, files = _build(model, report_pdf, tmp_path)
    answer = _run(agent, files, tmp_path)

    assert answer == "The report is read; the checklist is above."
    # The ingest ran through the sub-agent, every scripted call spent.
    assert engine.n_calls == len(engine.turns)
    # The record and the checklist's result are in the conversation's own
    # folder, beside each other; nothing went to the working directory.
    folder = files / "report_ingest" / "SYN"
    record = folder / "report.record.json"
    assert record.is_file()
    with open(folder / "report.checklist.json", encoding="utf-8") as handle:
        checked = json.load(handle)
    assert checked["status"] == "DRAFT"
    assert checked["report"]
    assert len(checked["code_checks"]) >= 6
    assert {row["status"] for row in checked["code_checks"]} <= {
        "pass", "fail", "unsure", "not_run"}
    assert not any(elsewhere.iterdir())
    # The delegation's answer (its structured response) carries the counts
    # and the failures, and names the file.
    delegated = json.loads(model.results["task"])
    summary = delegated["checklist"]
    assert summary["status"] == "DRAFT"
    assert sum(summary["counts"].values()) == len(checked["code_checks"])
    assert len(summary["failed"]) == min(6, summary["counts"]["fail"])
    assert summary["to_judge"] == len(checked["for_you_to_judge"])
    assert delegated["paths"]["checklist"] == str(
        folder / "report.checklist.json")
    assert delegated["paths"]["record"] == str(record)
    # The checklist tool, asked about that record, ran the same checks and
    # handed back the items to judge.
    asked = json.loads(model.results["report_checklist"])
    assert asked["record"] == str(record)
    assert asked["for_you_to_judge"]
    assert [r["status"] for r in asked["code_checks"]] == [
        r["status"] for r in checked["code_checks"]]


def test_without_the_checklist_the_record_comes_alone(report_pdf, engine,
                                                      tmp_path, monkeypatch):
    monkeypatch.setenv(review_flags.REPORT_INGEST_ENV, "1")

    class _Once(_Primary):
        def _generate(self, messages, stop=None, run_manager=None, **kw):
            if not any(m.type == "tool" for m in messages):
                return super()._generate(messages, stop, run_manager, **kw)
            self.results.update({m.name: m.content for m in messages
                                 if m.type == "tool"})
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="Read."))])

    model = _Once()
    agent, files = _build(model, report_pdf, tmp_path)
    _run(agent, files, tmp_path)
    folder = files / "report_ingest" / "SYN"

    assert (folder / "report.record.json").is_file()
    assert not (folder / "report.checklist.json").exists()
    assert json.loads(model.results["task"])["checklist"] is None
    assert "report_checklist" not in _tool_names(agent)


def test_the_switch_is_off_by_default_and_never_on_the_review_page(
        report_pdf, tmp_path, monkeypatch):
    agent, _files = _build(_Primary(), report_pdf, tmp_path / "off")
    assert "report_ingest" not in _tool_names(agent)

    monkeypatch.setenv(review_flags.REPORT_INGEST_ENV, "1")
    agent, _files = _build(_Primary(), report_pdf, tmp_path / "on")
    assert "report_ingest" in _tool_names(agent)
    agent, _files = _build(_Primary(), report_pdf, tmp_path / "review",
                           page="review")
    assert "report_ingest" not in _tool_names(agent)


def test_the_build_s_own_word_wins_over_the_switch(monkeypatch):
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )

    from funhouse_agent.deep.agent import build_deep_agent

    model = FakeMessagesListChatModel(responses=[AIMessage(content="ok")])
    monkeypatch.setenv(review_flags.REPORT_INGEST_ENV, "1")
    assert "report_ingest" not in _tool_names(
        build_deep_agent(model, enable_report_ingest=False))
    monkeypatch.delenv(review_flags.REPORT_INGEST_ENV)
    assert "report_ingest" in _tool_names(
        build_deep_agent(model, enable_report_ingest=True))


def test_a_checklist_that_cannot_run_leaves_the_record(tmp_path,
                                                       monkeypatch):
    from report_ingest import checklist as checklist_mod
    from report_ingest.subagent import CHECKLIST_NAME, run_checklist_on

    def broken(record=None, data=None):
        raise OSError("the checklist file is missing")

    monkeypatch.setattr(checklist_mod, "run_checklist", broken)
    out = run_checklist_on(None, tmp_path)

    assert out == {"error": "the checklist did not run: OSError: the "
                            "checklist file is missing"}
    assert not os.path.exists(tmp_path / CHECKLIST_NAME)
