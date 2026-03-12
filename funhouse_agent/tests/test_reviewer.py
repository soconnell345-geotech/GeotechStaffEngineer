"""Tests for funhouse_agent.reviewer — reference-checking reviewer agent.

All tests use mock engines — no API key needed.
"""

import json
import pytest

from funhouse_agent.agent import GeotechAgent
from funhouse_agent.reviewer import (
    run_review, needs_revision, _has_computations, _build_review_prompt,
    _filter_tool_call, REFERENCE_MODULES, REVIEWER_SYSTEM_PROMPT,
)
from funhouse_agent.react_support import ToolCall


# ---------------------------------------------------------------------------
# Unit tests for reviewer helpers
# ---------------------------------------------------------------------------

class TestHasComputations:
    def test_empty_log(self):
        assert _has_computations([]) is False

    def test_no_call_agent(self):
        log = [{"tool_name": "list_methods", "arguments": {}}]
        assert _has_computations(log) is False

    def test_with_call_agent(self):
        log = [
            {"tool_name": "describe_method", "arguments": {}},
            {"tool_name": "call_agent", "arguments": {
                "agent_name": "bearing_capacity",
                "method": "bearing_capacity_analysis",
            }},
        ]
        assert _has_computations(log) is True


class TestNeedsRevision:
    def test_none_review(self):
        assert needs_revision(None) is False

    def test_pass(self):
        assert needs_revision("REVIEW_STATUS: PASS\n\nLooks good.") is False

    def test_flag(self):
        assert needs_revision("REVIEW_STATUS: FLAG\n\nNote something.") is False

    def test_revise(self):
        assert needs_revision("REVIEW_STATUS: REVISE\n\nFix this.") is True


class TestFilterToolCall:
    def test_list_methods_allowed(self):
        tc = ToolCall(
            tool_name="list_methods",
            arguments={"agent_name": "dm7"},
            raw_json="", reasoning="",
        )
        assert _filter_tool_call(tc) is None

    def test_reference_module_allowed(self):
        tc = ToolCall(
            tool_name="call_agent",
            arguments={"agent_name": "dm7", "method": "some_func"},
            raw_json="", reasoning="",
        )
        assert _filter_tool_call(tc) is None

    def test_computation_module_blocked(self):
        tc = ToolCall(
            tool_name="call_agent",
            arguments={"agent_name": "bearing_capacity", "method": "vesic"},
            raw_json="", reasoning="",
        )
        result = _filter_tool_call(tc)
        assert result is not None
        assert "computation module" in result

    def test_all_reference_modules_allowed(self):
        for mod in REFERENCE_MODULES:
            tc = ToolCall(
                tool_name="call_agent",
                arguments={"agent_name": mod},
                raw_json="", reasoning="",
            )
            assert _filter_tool_call(tc) is None


class TestBuildReviewPrompt:
    def test_includes_question_and_answer(self):
        prompt = _build_review_prompt(
            question="What is the bearing capacity?",
            answer="The bearing capacity is 250 kPa.",
            tool_log=[],
        )
        assert "bearing capacity" in prompt
        assert "250 kPa" in prompt

    def test_lists_computations(self):
        log = [{
            "tool_name": "call_agent",
            "arguments": {
                "agent_name": "bearing_capacity",
                "method": "vesic_analysis",
                "parameters": {"width": 2.0, "phi": 30},
            },
        }]
        prompt = _build_review_prompt("Q", "A", log)
        assert "bearing_capacity.vesic_analysis" in prompt


# ---------------------------------------------------------------------------
# Mock engine for integration tests
# ---------------------------------------------------------------------------

class MockEngine:
    """Mock engine that returns canned responses in sequence."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._call_index = 0
        self.chat_calls = []

    def chat(self, user, system="", temperature=0):
        self.chat_calls.append({"user": user, "system": system})
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return "No more responses"


# ---------------------------------------------------------------------------
# Integration: run_review with mock engine
# ---------------------------------------------------------------------------

class TestRunReview:
    def test_skips_when_no_computations(self):
        engine = MockEngine(["should not be called"])
        result = run_review(
            engine=engine,
            question="What is phi?",
            answer="phi is 30 degrees",
            tool_log=[{"tool_name": "list_agents"}],
        )
        assert result is None
        assert len(engine.chat_calls) == 0

    def test_returns_review_text(self):
        """Reviewer that gives an immediate PASS without tool calls."""
        review_response = (
            "REVIEW_STATUS: PASS\n\n"
            "CONFIRMATIONS:\n"
            "- Vesic bearing capacity method appropriate per DM7.2 Ch4\n\n"
            "FLAGS:\n"
            "- None\n\n"
            "REVISIONS_NEEDED:\n"
            "- None"
        )
        engine = MockEngine([review_response])
        result = run_review(
            engine=engine,
            question="Calculate bearing capacity",
            answer="q_ult = 450 kPa using Vesic (1973)",
            tool_log=[{"tool_name": "call_agent", "arguments": {
                "agent_name": "bearing_capacity",
                "method": "vesic",
                "parameters": {"width": 2.0},
            }}],
        )
        assert result is not None
        assert "REVIEW_STATUS: PASS" in result
        assert "Vesic" in result

    def test_reviewer_with_tool_call(self):
        """Reviewer looks up a DM7 method then gives review."""
        responses = [
            # Round 1: reviewer wants to check DM7
            'Thought: Let me check bearing capacity factors in DM7.\n\n'
            '<tool_call>\n'
            '{"tool_name": "list_methods", "agent_name": "dm7"}\n'
            '</tool_call>',
            # Round 2: final review
            "REVIEW_STATUS: FLAG\n\n"
            "CONFIRMATIONS:\n"
            "- Method is appropriate\n\n"
            "FLAGS:\n"
            "- Consider checking water table effects per DM7.1 Ch5\n\n"
            "REVISIONS_NEEDED:\n"
            "- None",
        ]
        engine = MockEngine(responses)
        result = run_review(
            engine=engine,
            question="Bearing capacity?",
            answer="q_ult = 300 kPa",
            tool_log=[{"tool_name": "call_agent", "arguments": {
                "agent_name": "bearing_capacity",
                "method": "vesic",
                "parameters": {},
            }}],
        )
        assert "REVIEW_STATUS: FLAG" in result
        assert "water table" in result
        assert len(engine.chat_calls) == 2


# ---------------------------------------------------------------------------
# Integration: GeotechAgent with review=True
# ---------------------------------------------------------------------------

class TestAgentWithReview:
    def test_review_disabled_by_default(self):
        engine = MockEngine(["The answer is 42."])
        agent = GeotechAgent(genai_engine=engine, review=False)
        result = agent.ask("What is 6*7?")
        assert "Reviewer Notes" not in result.answer
        assert result.answer == "The answer is 42."

    def test_review_appended_on_pass(self):
        """Primary agent does a computation, reviewer PASSes."""
        responses = [
            # Primary round 1: call describe_method
            'Thought: Let me look up the method.\n\n'
            '<tool_call>\n'
            '{"tool_name": "describe_method", "agent_name": "bearing_capacity", '
            '"method": "bearing_capacity_analysis"}\n'
            '</tool_call>',
            # Primary round 2: call_agent
            'Thought: Now run the calculation.\n\n'
            '<tool_call>\n'
            '{"tool_name": "call_agent", "agent_name": "bearing_capacity", '
            '"method": "bearing_capacity_analysis", '
            '"parameters": {"width": 2.0, "depth": 1.5, "phi": 30, "gamma": 18.0}}\n'
            '</tool_call>',
            # Primary round 3: final answer
            'The ultimate bearing capacity is 450 kPa using Vesic (1973).',
            # Reviewer round 1: immediate PASS
            "REVIEW_STATUS: PASS\n\n"
            "CONFIRMATIONS:\n"
            "- Vesic method appropriate per DM7.2 Chapter 4\n\n"
            "FLAGS:\n- None\n\nREVISIONS_NEEDED:\n- None",
        ]
        engine = MockEngine(responses)
        agent = GeotechAgent(genai_engine=engine, review=True)
        result = agent.ask("Calculate bearing capacity for 2m footing, phi=30")
        assert "450 kPa" in result.answer
        assert "Reviewer Notes:" in result.answer
        assert "REVIEW_STATUS: PASS" in result.answer

    def test_review_triggers_revision(self):
        """Reviewer says REVISE, primary agent revises."""
        responses = [
            # Primary round 1: call_agent directly
            'Thought: Running bearing capacity.\n\n'
            '<tool_call>\n'
            '{"tool_name": "call_agent", "agent_name": "bearing_capacity", '
            '"method": "bearing_capacity_analysis", '
            '"parameters": {"width": 2.0, "depth": 1.5, "phi": 30, "gamma": 18.0}}\n'
            '</tool_call>',
            # Primary round 2: final answer (with an issue)
            'The bearing capacity is 450 kPa. FOS = 3.0 is adequate.',
            # Reviewer: REVISE
            "REVIEW_STATUS: REVISE\n\n"
            "CONFIRMATIONS:\n"
            "- Calculation method correct\n\n"
            "FLAGS:\n- None\n\n"
            "REVISIONS_NEEDED:\n"
            "- Per UFC 3-220-10, minimum FOS for bearing capacity is 3.0 "
            "for normal loads but the applied load was not stated. "
            "Clarify the demand vs capacity.",
            # Primary revision response
            "REVISED: The ultimate bearing capacity is 450 kPa (Vesic 1973). "
            "With an applied footing pressure of 150 kPa, FOS = 450/150 = 3.0, "
            "which meets the minimum per UFC 3-220-10.",
        ]
        engine = MockEngine(responses)
        agent = GeotechAgent(genai_engine=engine, review=True)
        result = agent.ask("Bearing capacity of 2m footing")
        assert "REVISED" in result.answer
        assert "Reviewer Notes:" in result.answer
        assert "REVIEW_STATUS: REVISE" in result.answer

    def test_no_review_when_no_computations(self):
        """If the agent only lists methods, no review is triggered."""
        responses = [
            # Primary: list_agents then answer
            'Thought: Let me list modules.\n\n'
            '<tool_call>\n'
            '{"tool_name": "list_agents"}\n'
            '</tool_call>',
            # Final answer (no call_agent was used)
            'There are 50 modules available.',
        ]
        engine = MockEngine(responses)
        agent = GeotechAgent(genai_engine=engine, review=True)
        result = agent.ask("What modules are available?")
        assert "Reviewer Notes" not in result.answer
