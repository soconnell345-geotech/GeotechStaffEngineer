"""The whole-report ingest on the app's surface — offline, no model, no key.

Three things are under test and none of them is the ingest itself (that has
its own suite): that the tool and the sub-agent appear only when they are
asked for AND the installed planlens can serve them, that the sub-agent spec
is built the way every other sub-agent spec in this build is, and that a run
through the tool comes back as the compact result rather than the record.
"""

import copy
import json

import pytest

pytest.importorskip("planlens.tools")
pytest.importorskip("planlens.document.roles")
fitz = pytest.importorskip("fitz")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from funhouse_agent import document_tools  # noqa: E402
from funhouse_agent.deep.agent import build_deep_agent  # noqa: E402
from funhouse_agent.deep.prompt import REPORT_INGEST_NUDGE  # noqa: E402
from funhouse_agent.deep.scratch_guard import (  # noqa: E402
    ScratchFilesystemGuard,
)
from funhouse_agent.deep.tools import make_report_ingest_tool  # noqa: E402
from report_ingest.tests.fake_engine import FakeEngine  # noqa: E402
from report_ingest.tests.narrative_fixtures import (  # noqa: E402
    build_narrative_report,
)
from report_ingest.tests.test_graph import full_script  # noqa: E402


@pytest.fixture(scope="module")
def pdf(tmp_path_factory):
    path = tmp_path_factory.mktemp("reports") / "SYN.pdf"
    path.write_bytes(build_narrative_report().pdf)
    return path


@pytest.fixture()
def model():
    return FakeMessagesListChatModel(responses=[AIMessage(content="ok")])


def _older_planlens(monkeypatch, without=()):
    """Make the installed planlens look like one without those tools."""
    from planlens.tools import specs as planlens_specs
    kept = [spec for spec in planlens_specs.TOOL_SPECS
            if spec["name"] not in without]
    monkeypatch.setattr(planlens_specs, "TOOL_SPECS", kept)


def _tool_names(agent):
    return set(agent.nodes["tools"].bound.tools_by_name)


def _capture(monkeypatch):
    """What ``build_deep_agent`` hands ``create_deep_agent``.

    The same spy the Phase-3 suite uses: the sub-agent specs and the system
    prompt are arguments to a call, and reading them there is the one place
    they are still the objects this build made rather than something
    deepagents has compiled.
    """
    from funhouse_agent.deep import agent as agent_mod

    captured = {}
    original = agent_mod.create_deep_agent

    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(agent_mod, "create_deep_agent", spy)
    return captured


def _specs(captured):
    return {spec["name"]: spec for spec in captured.get("subagents") or ()}


class TestWhenTheToolAppears:

    def test_it_is_off_by_default(self, model):
        assert "report_ingest" not in _tool_names(build_deep_agent(model))

    def test_it_appears_when_it_is_asked_for(self, model):
        agent = build_deep_agent(model, enable_report_ingest=True)

        assert "report_ingest" in _tool_names(agent)

    def test_it_hides_on_a_planlens_that_cannot_serve_it(self, monkeypatch,
                                                        model):
        _older_planlens(monkeypatch, without=("log_grid",))

        assert make_report_ingest_tool() == []
        agent = build_deep_agent(model, enable_report_ingest=True)
        assert "report_ingest" not in _tool_names(agent)

    def test_the_roles_tool_is_needed_too(self, monkeypatch):
        _older_planlens(monkeypatch, without=("document_roles",))

        assert document_tools.report_ingest_supported() is False
        assert make_report_ingest_tool() == []

    def test_the_prompt_line_follows_the_tool(self, model, monkeypatch):
        captured = _capture(monkeypatch)
        build_deep_agent(model, enable_report_ingest=True)
        assert REPORT_INGEST_NUDGE in captured["system_prompt"]

        build_deep_agent(model)
        assert REPORT_INGEST_NUDGE not in captured["system_prompt"]

        _older_planlens(monkeypatch, without=("log_grid",))
        build_deep_agent(model, enable_report_ingest=True)
        assert REPORT_INGEST_NUDGE not in captured["system_prompt"]


class TestTheSubAgentSpec:

    def test_it_is_attached_with_the_tool(self, model, monkeypatch):
        captured = _capture(monkeypatch)
        build_deep_agent(model, enable_report_ingest=True)
        specs = _specs(captured)

        assert "report_ingest" in specs
        assert "runnable" in specs["report_ingest"]
        assert "whole geotechnical report" in \
            specs["report_ingest"]["description"].lower()

    def test_the_spec_carries_the_guard_and_the_budget(self, model,
                                                       monkeypatch):
        from funhouse_agent.deep.limits import ModelCallBudgetMiddleware

        captured = _capture(monkeypatch)
        build_deep_agent(model, enable_report_ingest=True)
        spec = _specs(captured)["report_ingest"]
        kinds = [type(m) for m in spec["middleware"]]

        assert ScratchFilesystemGuard in kinds
        assert ModelCallBudgetMiddleware in kinds

    def test_it_is_absent_by_default(self, model, monkeypatch):
        captured = _capture(monkeypatch)
        build_deep_agent(model)

        assert "report_ingest" not in _specs(captured)

    def test_the_graph_answers_with_a_structured_response(self, pdf,
                                                          tmp_path):
        from report_ingest.subagent import build_report_ingest_graph

        engine = FakeEngine(full_script())
        graph = build_report_ingest_graph(
            lambda: engine, out_dir_factory=lambda _s: str(tmp_path))
        state = graph.invoke(
            {"messages": [HumanMessage(content=f"Ingest {pdf} please.")]})

        answer = state["structured_response"]
        assert answer.investigations == 2 and answer.lab_tests == 2
        assert answer.workflow == "standard"
        assert answer.diggs_ok is True
        assert set(answer.paths) >= {"record", "summary", "page", "diggs"}
        assert answer.summary.startswith("# ")
        assert "Rosewood" in state["messages"][-1].content

    def test_a_delegation_that_names_no_file_says_so(self, tmp_path):
        from report_ingest.subagent import build_report_ingest_graph

        graph = build_report_ingest_graph(
            lambda: FakeEngine([]), out_dir_factory=lambda _s: str(tmp_path))
        state = graph.invoke(
            {"messages": [HumanMessage(content="read the report please")]})

        assert "no PDF was named" in state["structured_response"].error

    def test_no_engine_means_it_says_so_rather_than_reading_anyway(
            self, pdf, tmp_path):
        from report_ingest.subagent import build_report_ingest_graph

        graph = build_report_ingest_graph(
            lambda: None, out_dir_factory=lambda _s: str(tmp_path))
        state = graph.invoke(
            {"messages": [HumanMessage(content=f"Ingest {pdf}")]})

        assert "no ingest engine" in state["structured_response"].error


class TestARunThroughTheTool:

    def test_the_compact_result_comes_back_not_the_record(self, pdf,
                                                          tmp_path):
        engine = FakeEngine(full_script())
        (tool,) = make_report_ingest_tool(engine=engine,
                                          out_dir=str(tmp_path))
        answer = json.loads(tool.invoke({"source": str(pdf)}))

        assert answer["investigations"] == 2
        assert answer["lab_tests"] == 2
        assert answer["n_pages"] == 22
        assert answer["diggs_ok"] is True
        assert "investigations" not in answer.get("paths", {})
        assert "record" in answer["paths"]
        assert len(json.dumps(answer)) < 4000      # a paragraph, not a record

    def test_the_caller_s_questions_ride_through(self, pdf, tmp_path):
        from report_ingest.narrative_reader import ReadExtra
        from report_ingest.tests.test_graph import (
            lab_turn, log_turn, narrative_turn, review_turns, triage_turn,
        )
        from report_ingest.narrative_reader import NarrativeReading

        narrative = narrative_turn()
        reading = narrative["final"].model_copy(update={
            "extra_answers": [ReadExtra(
                question="What embedment is required?",
                answer="A minimum of 0.6 m below finished grade.",
                page=4, quote="a minimum embedment of 0.6 m")]})
        assert isinstance(reading, NarrativeReading)
        script = ([triage_turn()] + review_turns() + [{"final": reading}]
                  + [log_turn("B-1"), log_turn("TP-1", "test_pit"),
                     lab_turn("atterberg", 12), lab_turn("gradation", 13)])
        (tool,) = make_report_ingest_tool(engine=FakeEngine(script),
                                          out_dir=str(tmp_path))
        answer = json.loads(tool.invoke(
            {"source": str(pdf),
             "questions": "What embedment is required?"}))

        assert answer["answers"] == [
            {"question": "What embedment is required?",
             "answer": "A minimum of 0.6 m below finished grade."}]

    def test_an_attachment_key_resolves(self, tmp_path):
        engine = FakeEngine(full_script())
        (tool,) = make_report_ingest_tool(
            engine=engine, out_dir=str(tmp_path),
            attachments={"report.pdf": build_narrative_report().pdf})
        answer = json.loads(tool.invoke({"source": "report.pdf"}))

        assert answer["n_pages"] == 22

    def test_a_source_that_is_neither_is_an_error_not_a_crash(self, tmp_path):
        (tool,) = make_report_ingest_tool(engine=FakeEngine([]),
                                          out_dir=str(tmp_path))
        answer = json.loads(tool.invoke({"source": "nowhere.pdf"}))

        assert "not an attachment key" in answer["error"]

    def test_no_engine_is_an_answer_rather_than_a_missing_tool(self,
                                                               pdf, tmp_path):
        (tool,) = make_report_ingest_tool(engine=None, out_dir=str(tmp_path))
        answer = json.loads(tool.invoke({"source": str(pdf)}))

        assert "no ingest engine is configured" in answer["error"]


def test_the_wiring_modules_import_neither_anthropic_nor_langgraph():
    """The ingest's own modules stay cheap and SDK-free.

    ``anthropic`` is optional and not installed on the cluster, and LangGraph
    is imported only when a sub-agent graph is actually built -- so the
    package's own modules must pull in neither, however the app is wired.
    """
    import subprocess
    import sys

    code = ("import sys; "
            "import report_ingest.subagent, report_ingest.graph, "
            "report_ingest.run_folder; "
            "print('anthropic' in sys.modules, 'langgraph' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == "False False"
