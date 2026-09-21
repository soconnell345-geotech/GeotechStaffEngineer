"""The library as a sub-agent: cited text out, ceilings held, nothing invented.

Three things are under test and none of them is a model's judgement. That the
QUERY LAYER renders every result as compact text with `(report, page)` on
every fact; that the GRAPH stops at its ceilings and answers with the
structured shape the primary agent reads; and that a citation the queries did
not return is reported as a gap rather than passed through as though it had
been checked.
"""

from __future__ import annotations

import json

import pytest

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage

from report_ingest.library import Library
from report_ingest.library_agent import (
    LIBRARY_SYSTEM_PROMPT, LIBRARY_TOOL_SPECS, MAX_TOOL_CALLS, TOOL_NAMES,
    LibraryAnswer, answer_question, build_answer,
    build_report_library_graph, build_report_library_subagent,
    library_available, run_query,
)
from report_ingest.tests.library_fixtures import build_library


class ToolCallingFake(FakeMessagesListChatModel):
    """A fake chat model that accepts tools and replies from a script.

    Two things on top of ``FakeMessagesListChatModel``. It accepts
    ``bind_tools``, which a model with tools must. And it hands back a COPY
    of each scripted message with no id: the stock fake returns the same
    object every time it cycles, and LangGraph's message reducer replaces a
    message whose id it has already seen rather than appending it -- so a
    scripted loop would silently stop after one pass.
    """

    def bind_tools(self, tools, **kwargs):          # noqa: D102 - a fake
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult

        message = self.responses[self.i]
        self.i = (self.i + 1) % len(self.responses)
        return ChatResult(generations=[ChatGeneration(
            message=message.model_copy(update={"id": None}))])


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    folder = tmp_path_factory.mktemp("library")
    build_library(folder)
    return str(folder)


@pytest.fixture()
def library(root):
    lib = Library(root)
    yield lib
    lib.close()


def _call(name, arguments=None, call_id="c1"):
    return AIMessage(content="", tool_calls=[
        {"name": name, "args": dict(arguments or {}), "id": call_id}])


class TestTheToolSpecs:

    def test_every_spec_is_a_json_schema_with_a_description(self):
        for spec in LIBRARY_TOOL_SPECS:
            assert spec["name"] and len(spec["description"]) > 40
            assert spec["parameters"]["type"] == "object"
            for name, schema in spec["parameters"]["properties"].items():
                assert schema.get("type"), name
                assert schema.get("description"), name

    def test_every_spec_has_a_handler_and_every_handler_a_spec(self):
        from report_ingest.library_agent import _HANDLERS

        assert set(TOOL_NAMES) == set(_HANDLERS)

    def test_the_required_arguments_are_the_ones_without_a_default(self):
        required = {spec["name"]: set(spec["parameters"]["required"])
                    for spec in LIBRARY_TOOL_SPECS}

        assert required["facts"] == {"report_id"}
        assert required["compare"] == {"field"}
        assert required["find"] == {"text"}
        assert required["disagreements"] == set()


class TestEveryQueryRendersCitedText:

    def test_library_stats(self, library):
        result = run_query(library, "library_stats")

        assert "7 report(s)" in result.text
        assert "bound inside another" in result.text
        assert not result.empty

    def test_list_reports(self, library):
        result = run_query(library, "list_reports", {"post": "Vale Harbour"})

        assert "L02:" in result.text and "(L05)" in result.text
        assert set(result.reports) == {"L02", "L02-bound1", "L05"}

    def test_list_reports_with_nothing_matching_is_empty_not_wrong(
            self, library):
        result = run_query(library, "list_reports", {"post": "Nowhere"})

        assert result.empty is True
        assert "No report in the library matches" in result.text

    def test_find(self, library):
        result = run_query(library, "find",
                           {"text": "severely corrosive", "k": 3})

        assert "(L05, p4)" in result.text
        assert ("L05", 4) in result.rows

    def test_where_is(self, library):
        result = run_query(library, "where_is",
                           {"text": "allowable bearing pressure of 3,000 psf"})

        assert "(L01, p4)" in result.text

    def test_facts(self, library):
        result = run_query(library, "facts",
                           {"report_id": "L02", "fields": ["siteClass"]})

        assert "siteClass: Site Class C (L02, p4)" in result.text
        assert '"Site Class C"' in result.text
        assert result.rows == [("L02", 4)]

    def test_facts_names_what_the_report_did_not_answer(self, library):
        result = run_query(library, "facts",
                           {"report_id": "L06", "fields": ["siteClass"]})

        assert "Not answered by this report: siteClass" in result.text
        assert result.empty is True

    def test_compare(self, library):
        result = run_query(library, "compare", {"field": "siteClass"})

        assert "L01: Site Class D (L01, p4)" in result.text
        assert "Did not answer it: L03, L04, L06" in result.text

    def test_explorations(self, library):
        result = run_query(library, "explorations", {"report_id": "L01",
                                                     "kind": "boring"})

        assert "B-1 (boring)" in result.text
        assert "total depth 30 ft" in result.text
        assert "(L01, p7, p8)" in result.text
        assert "Brown sandy lean CLAY" in result.text

    def test_lab_summary(self, library):
        result = run_query(library, "lab_summary", {"report_id": "L01"})

        assert "atterberg" in result.text and "ASTM D4318" in result.text
        assert "(L01, p12)" in result.text

    def test_calculations(self, library):
        result = run_query(library, "calculations", {"report_id": "L04"})

        assert "slope stability" in result.text
        assert "SLIDE2 9.0" in result.text
        assert "(L04, p20, p21)" in result.text

    def test_disagreements(self, library):
        result = run_query(library, "disagreements")

        assert "conflict at" in result.text
        assert "38 vs 41" in result.text
        assert "(L01, p12, p14)" in result.text

    def test_a_report_that_is_not_there_is_an_answer_not_a_crash(self,
                                                                 library):
        result = run_query(library, "facts", {"report_id": "L99"})

        assert result.empty is True
        assert "L99" in result.text and "The library holds:" in result.text

    def test_a_query_that_does_not_exist_says_which_do(self, library):
        result = run_query(library, "read_the_pdf")

        assert "no library query called" in result.error
        assert "library_stats" in result.error


class TestTheCeilings:

    def test_a_result_stops_at_max_rows(self, library):
        result = run_query(library, "list_reports", {}, max_rows=2)

        assert len([line for line in result.text.splitlines()
                    if line.startswith("L")]) == 2

    def test_a_result_stops_at_max_chars_and_says_it_was_cut(self, library):
        result = run_query(library, "disagreements", {}, max_chars=80)

        assert len(result.text) <= 100
        assert result.text.endswith("(cut)")

    def test_the_graph_refuses_the_query_after_the_ceiling(self, root):
        """The limit is Python, because middleware on the spec is ignored."""
        model = ToolCallingFake(responses=[
            _call("library_stats", {}, "loop"),      # cycles for ever
            _call("library_stats", {}, "loop2"),
        ])
        graph = build_report_library_graph(model, library_root=root,
                                           max_tool_calls=3)
        state = graph.invoke({"messages": [HumanMessage(content="go")],
                              "question": "go"})

        assert state["calls"] == 3
        refusals = [m for m in state["messages"]
                    if "ceiling of 3" in str(getattr(m, "content", ""))]
        assert refusals

    def test_the_graph_stops_taking_turns(self, root):
        model = ToolCallingFake(responses=[_call("library_stats", {}, "l")])
        graph = build_report_library_graph(model, library_root=root,
                                           max_tool_calls=100)
        state = graph.invoke({"messages": [HumanMessage(content="go")]})

        assert isinstance(state["structured_response"], LibraryAnswer)
        assert state["turns"] <= MAX_TOOL_CALLS + 2


class TestTheStructuredAnswer:

    def test_the_shape_the_primary_reads(self, root):
        model = ToolCallingFake(responses=[
            _call("compare", {"field": "siteClass"}),
            AIMessage(content="L01 is Site Class D (L01, p4) and L02 is Site "
                              "Class C (L02, p4)."),
        ])
        answer = answer_question("What site class did each report give?",
                                 model=model, library_root=root)

        assert isinstance(answer, LibraryAnswer)
        assert answer.answer.startswith("L01 is Site Class D")
        assert [(c.report, c.page) for c in answer.citations] == [
            ("L01", 4), ("L02", 4)]
        assert set(answer.reports_consulted) == {"L01", "L02", "L02-bound1",
                                                 "L05"}
        assert answer.gaps == []
        assert answer.queries == 1
        assert answer.error == ""

    def test_a_page_no_query_returned_is_a_gap_not_a_citation(self, root):
        model = ToolCallingFake(responses=[
            _call("compare", {"field": "siteClass"}),
            AIMessage(content="L01 is Site Class D (L01, p99)."),
        ])
        answer = answer_question("site class?", model=model,
                                 library_root=root)

        assert answer.citations == []
        assert answer.gaps == ["the answer cites L01 page 99, which no query "
                               "returned"]

    def test_a_report_no_query_returned_is_a_gap(self, root):
        model = ToolCallingFake(responses=[
            _call("facts", {"report_id": "L01", "fields": ["siteClass"]}),
            AIMessage(content="It is Site Class D (L01, p4), like L99 (L99)."),
        ])
        answer = answer_question("site class?", model=model,
                                 library_root=root)

        assert [(c.report, c.page) for c in answer.citations] == [("L01", 4)]
        assert any("L99" in gap for gap in answer.gaps)

    def test_an_empty_query_becomes_a_gap(self, root):
        model = ToolCallingFake(responses=[
            _call("list_reports", {"post": "Nowhere"}),
            AIMessage(content="The library holds no report for that post."),
        ])
        answer = answer_question("anything from Nowhere?", model=model,
                                 library_root=root)

        assert "the query list_reports returned nothing" in answer.gaps

    def test_the_answers_own_gap_lines_are_lifted_out_of_the_prose(self,
                                                                   root):
        model = ToolCallingFake(responses=[
            _call("compare", {"field": "siteClass"}),
            AIMessage(content="L02 is Site Class C (L02, p4).\n"
                              "Gap: three reports gave no site class."),
        ])
        answer = answer_question("site class?", model=model,
                                 library_root=root)

        assert "Gap:" not in answer.answer
        assert "three reports gave no site class." in answer.gaps

    def test_a_model_that_cannot_call_tools_says_so_rather_than_guessing(
            self, root):
        model = FakeMessagesListChatModel(responses=[AIMessage(content="hi")])
        answer = answer_question("anything", model=model, library_root=root)

        assert "cannot call tools" in answer.error
        assert answer.answer == ""

    def test_a_failed_query_is_a_gap_and_the_answer_still_comes_back(self,
                                                                     root):
        model = ToolCallingFake(responses=[
            _call("compare", {"field": "soilColour"}),
            AIMessage(content="The library cannot answer that."),
        ])
        answer = answer_question("what colour?", model=model,
                                 library_root=root)

        assert answer.answer == "The library cannot answer that."
        assert any("the query compare failed" in gap for gap in answer.gaps)

    def test_several_queries_are_all_counted(self, root):
        model = ToolCallingFake(responses=[
            _call("library_stats", {}, "a"),
            _call("list_reports", {"firm": "Meridian"}, "b"),
            AIMessage(content="Meridian wrote L02 (L02) and L05 (L05)."),
        ])
        answer = answer_question("who wrote what?", model=model,
                                 library_root=root)

        assert answer.queries == 2
        assert [(c.report, c.page) for c in answer.citations] == [
            ("L02", None), ("L05", None)]


class TestBuildAnswer:
    """The citation check on its own, with no graph and no model."""

    def test_a_citation_the_rows_carry_is_kept(self):
        from report_ingest.library_agent import QueryResult

        result = QueryResult(text="", rows=[("L01", 4)], reports=["L01"])
        answer = build_answer("It is D (L01, p4).", [("facts", result)],
                              ["L01"])

        assert [(c.report, c.page) for c in answer.citations] == [("L01", 4)]

    def test_a_word_in_brackets_is_not_read_as_a_citation(self):
        from report_ingest.library_agent import QueryResult

        result = QueryResult(text="", rows=[("L01", 4)], reports=["L01"])
        answer = build_answer("It is D (probably) at L01 (L01, p4).",
                              [("facts", result)], ["L01"])

        assert [(c.report, c.page) for c in answer.citations] == [("L01", 4)]

    def test_the_page_spellings_a_model_uses_are_all_read(self):
        from report_ingest.library_agent import QueryResult

        result = QueryResult(text="", rows=[("L01", 4), ("L02", 7)],
                             reports=["L01", "L02"])
        answer = build_answer("one (L01, page 4), two (l02, p7).",
                              [("facts", result)], ["L01", "L02"])

        assert [(c.report, c.page) for c in answer.citations] == [
            ("L01", 4), ("L02", 7)]


class TestTheSubAgentSpec:

    def test_it_is_the_shape_deepagents_takes(self, root):
        model = ToolCallingFake(responses=[AIMessage(content="ok")])
        spec = build_report_library_subagent(model, library_root=root)

        assert spec["name"] == "report_library"
        assert "cites the report id" in spec["description"]
        assert "runnable" in spec
        assert hasattr(spec["runnable"], "invoke")

    def test_a_model_call_budget_is_declared_when_asked_for(self, root):
        from funhouse_agent.deep.limits import ModelCallBudgetMiddleware

        model = ToolCallingFake(responses=[AIMessage(content="ok")])
        spec = build_report_library_subagent(model, library_root=root,
                                             max_model_calls=6)

        assert [type(m) for m in spec["middleware"]] == [
            ModelCallBudgetMiddleware]

    def test_the_runnable_answers_with_a_structured_response(self, root):
        model = ToolCallingFake(responses=[
            _call("library_stats"),
            AIMessage(content="Seven reports, 2014 to 2026 (L01)."),
        ])
        spec = build_report_library_subagent(model, library_root=root)
        state = spec["runnable"].invoke(
            {"messages": [HumanMessage(content="what is in the library?")]})

        assert isinstance(state["structured_response"], LibraryAnswer)
        assert "Seven reports" in state["structured_response"].answer


class TestTheSystemPrompt:

    def test_it_states_the_rules_that_keep_an_answer_honest(self):
        text = LIBRARY_SYSTEM_PROMPT.lower()

        assert "every fact in your answer comes from a tool result" in text
        assert "(report id, page)" in text
        assert "does not hold the answer" in text
        assert "gap:" in text

    def test_it_names_every_query_the_model_can_make(self):
        for name in TOOL_NAMES:
            assert name in LIBRARY_SYSTEM_PROMPT


class TestFeatureDetection:

    def test_a_folder_of_records_is_a_library(self, root):
        assert library_available(root) is True

    def test_an_empty_folder_is_not(self, tmp_path):
        assert library_available(str(tmp_path)) is False

    def test_a_folder_that_is_not_there_is_not(self, tmp_path):
        assert library_available(str(tmp_path / "nowhere")) is False

    def test_no_root_at_all_is_not(self):
        assert library_available(None) is False
        assert library_available("") is False


def test_the_answer_json_is_small_enough_for_a_tool_result(root):
    from report_ingest.library_agent import answer_as_json

    model = ToolCallingFake(responses=[
        _call("compare", {"field": "siteClass"}),
        AIMessage(content="L01 is D (L01, p4), L02 is C (L02, p4)."),
    ])
    answer = answer_question("site class?", model=model, library_root=root)
    blob = answer_as_json(answer)

    assert json.loads(blob)["answer"].startswith("L01 is D")
    assert len(blob) < 2000
