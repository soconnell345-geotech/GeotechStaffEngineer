"""The report library on the app's surface — offline, no model, no key.

The same three things the ingest's wiring suite checks, for the other half of
the pair: that the tool and the sub-agent appear only when they are asked for
AND there is a library to ask, that the spec is built the way every other
sub-agent spec in this build is, and that a run through the tool comes back as
the compact cited answer rather than as a pile of rows.
"""

import json

import pytest

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402

from funhouse_agent.deep.agent import build_deep_agent  # noqa: E402
from funhouse_agent.deep.prompt import (  # noqa: E402
    REPORT_INGEST_NUDGE, REPORT_LIBRARY_NUDGE,
)
from funhouse_agent.deep.scratch_guard import (  # noqa: E402
    ScratchFilesystemGuard,
)
from funhouse_agent.deep.tools import make_report_library_tool  # noqa: E402
from report_ingest.tests.library_fixtures import build_library  # noqa: E402
from report_ingest.tests.test_library_agent import (  # noqa: E402
    ToolCallingFake,
)


@pytest.fixture(scope="module")
def library_root(tmp_path_factory):
    folder = tmp_path_factory.mktemp("library")
    build_library(folder)
    return str(folder)


@pytest.fixture()
def model():
    return FakeMessagesListChatModel(responses=[AIMessage(content="ok")])


def _tool_names(agent):
    return set(agent.nodes["tools"].bound.tools_by_name)


def _capture(monkeypatch):
    """What ``build_deep_agent`` hands ``create_deep_agent``."""
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
        assert "report_library" not in _tool_names(build_deep_agent(model))

    def test_it_appears_when_it_is_asked_for(self, model, library_root):
        agent = build_deep_agent(model, enable_report_library=True,
                                 library_root=library_root)

        assert "report_library" in _tool_names(agent)

    def test_it_hides_when_there_is_no_library_to_ask(self, model, tmp_path):
        agent = build_deep_agent(model, enable_report_library=True,
                                 library_root=str(tmp_path / "nothing_here"))

        assert "report_library" not in _tool_names(agent)

    def test_it_hides_on_a_folder_that_holds_no_record(self, model,
                                                      tmp_path):
        (tmp_path / "notes.txt").write_text("nothing", encoding="utf-8")

        assert make_report_library_tool(library_root=str(tmp_path)) == []

    def test_asking_for_it_with_no_root_is_not_an_error(self, model):
        agent = build_deep_agent(model, enable_report_library=True)

        assert "report_library" not in _tool_names(agent)

    def test_the_prompt_line_follows_the_tool(self, model, library_root,
                                              monkeypatch, tmp_path):
        captured = _capture(monkeypatch)
        build_deep_agent(model, enable_report_library=True,
                         library_root=library_root)
        assert REPORT_LIBRARY_NUDGE in captured["system_prompt"]

        build_deep_agent(model)
        assert REPORT_LIBRARY_NUDGE not in captured["system_prompt"]

        build_deep_agent(model, enable_report_library=True,
                         library_root=str(tmp_path / "nothing"))
        assert REPORT_LIBRARY_NUDGE not in captured["system_prompt"]

    def test_the_two_halves_are_independent(self, model, library_root,
                                            monkeypatch):
        """Ingesting and asking are separate switches.

        The library can be on where no report can be read (a folder restored
        from SharePoint on a host with no engine), and the ingest can be on
        with nothing yet read.
        """
        captured = _capture(monkeypatch)
        build_deep_agent(model, enable_report_library=True,
                         library_root=library_root)

        assert REPORT_LIBRARY_NUDGE in captured["system_prompt"]
        assert REPORT_INGEST_NUDGE not in captured["system_prompt"]


class TestTheSubAgentSpec:

    def test_it_is_attached_with_the_tool(self, model, library_root,
                                          monkeypatch):
        captured = _capture(monkeypatch)
        build_deep_agent(model, enable_report_library=True,
                         library_root=library_root)
        specs = _specs(captured)

        assert "report_library" in specs
        assert "runnable" in specs["report_library"]
        assert "cites the report id" in specs["report_library"]["description"]

    def test_the_spec_carries_the_guard_and_the_budget(self, model,
                                                       library_root,
                                                       monkeypatch):
        from funhouse_agent.deep.limits import ModelCallBudgetMiddleware

        captured = _capture(monkeypatch)
        build_deep_agent(model, enable_report_library=True,
                         library_root=library_root)
        kinds = [type(m) for m in _specs(captured)["report_library"]
                 ["middleware"]]

        assert ScratchFilesystemGuard in kinds
        assert ModelCallBudgetMiddleware in kinds

    def test_it_is_absent_by_default(self, model, monkeypatch):
        captured = _capture(monkeypatch)
        build_deep_agent(model)

        assert "report_library" not in _specs(captured)

    def test_both_sub_agents_can_stand_side_by_side(self, model,
                                                    library_root,
                                                    monkeypatch):
        pytest.importorskip("planlens.document.roles")
        from funhouse_agent import document_tools

        if not document_tools.report_ingest_supported():
            pytest.skip("the installed planlens cannot serve the ingest")
        captured = _capture(monkeypatch)
        build_deep_agent(model, enable_report_ingest=True,
                         enable_report_library=True,
                         library_root=library_root)
        specs = _specs(captured)

        assert {"report_ingest", "report_library"} <= set(specs)


class TestARunThroughTheTool:

    def test_the_compact_cited_answer_comes_back(self, library_root):
        model = ToolCallingFake(responses=[
            AIMessage(content="", tool_calls=[
                {"name": "compare", "args": {"field": "siteClass"},
                 "id": "c1"}]),
            AIMessage(content="L01 is Site Class D (L01, p4) and L02 is Site "
                              "Class C (L02, p4)."),
        ])
        (tool,) = make_report_library_tool(model=model,
                                           library_root=library_root)
        answer = json.loads(tool.invoke(
            {"question": "What site class did each report give?"}))

        assert answer["answer"].startswith("L01 is Site Class D")
        assert {"report": "L01", "page": 4} in answer["citations"]
        assert "L05" in answer["reports_consulted"]
        assert len(json.dumps(answer)) < 4000   # a paragraph, not a table

    def test_no_model_means_it_says_so_rather_than_answering_anyway(
            self, library_root):
        (tool,) = make_report_library_tool(model=None,
                                           library_root=library_root)
        answer = json.loads(tool.invoke({"question": "anything?"}))

        assert "no model is configured" in answer["error"]

    def test_a_ceiling_passed_through_holds(self, library_root):
        model = ToolCallingFake(responses=[
            AIMessage(content="", tool_calls=[
                {"name": "library_stats", "args": {}, "id": "a"}]),
            AIMessage(content="", tool_calls=[
                {"name": "library_stats", "args": {}, "id": "b"}]),
        ])
        (tool,) = make_report_library_tool(model=model,
                                           library_root=library_root,
                                           max_tool_calls=1)
        answer = json.loads(tool.invoke({"question": "loop please"}))

        assert answer["queries"] == 1


def test_the_ingest_result_names_the_library_the_report_landed_in(tmp_path):
    """A report read this turn is queryable this turn.

    ``IngestSummary.library_root`` is the folder the database sits in, and
    that folder is a library -- of one report after a single ingest, of the
    whole run where the caller passed a shared database.
    """
    from report_ingest.library_agent import library_available
    from report_ingest.subagent import IngestSummary
    from report_ingest.tests.library_fixtures import record_l01
    from report_ingest.writers import write_outputs

    out = tmp_path / "one_report"
    write_outputs(record_l01(), out, write_diggs_file=False)
    summary = IngestSummary(library_root=str(out))

    assert "library_root" in IngestSummary.model_fields
    assert library_available(summary.library_root) is True
