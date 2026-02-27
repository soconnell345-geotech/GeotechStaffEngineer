"""
Tests for chat_agent module.

All tests use mock chat functions — no API keys or network calls required.
"""

import json
import pytest

from chat_agent.parser import parse_response, ToolCall, ParseResult, VALID_TOOLS
from chat_agent.react_prompt import (
    build_system_prompt,
    build_system_prompt_compact,
    REACT_PREFIX,
)
from chat_agent.agent import (
    GeotechChatAgent,
    AgentResult,
    ConversationHistory,
    dispatch_tool,
    _truncate,
)
from trial_agent.agent_registry import AGENT_NAMES


# ===================================================================
# Helpers
# ===================================================================

def make_scripted_chat(responses: list[str]):
    """Create a mock chat function that returns pre-scripted responses in order."""
    call_count = [0]

    def chat_fn(prompt, system_prompt, temp):
        idx = min(call_count[0], len(responses) - 1)
        call_count[0] += 1
        return responses[idx]

    chat_fn._call_count = call_count
    return chat_fn


# ===================================================================
# Parser tests
# ===================================================================

class TestParser:
    """Tests for <tool_call> tag parsing."""

    def test_no_tool_call_returns_none(self):
        """Plain text with no tags → final answer."""
        result = parse_response("The bearing capacity is 500 kPa.")
        assert result.tool_call is None
        assert "500 kPa" in result.full_text

    def test_valid_call_agent(self):
        """Standard call_agent extraction."""
        text = (
            'Thought: I need to calculate bearing capacity.\n\n'
            '<tool_call>\n'
            '{"tool_name": "call_agent", "agent_name": "bearing_capacity", '
            '"method": "bearing_capacity_analysis", '
            '"parameters": {"width": 2.0, "phi": 30}}\n'
            '</tool_call>'
        )
        result = parse_response(text)
        assert result.tool_call is not None
        assert result.tool_call.tool_name == "call_agent"
        assert result.tool_call.arguments["agent_name"] == "bearing_capacity"
        assert result.tool_call.arguments["parameters"]["width"] == 2.0
        assert "bearing capacity" in result.tool_call.reasoning.lower()

    def test_valid_list_methods(self):
        text = (
            '<tool_call>\n'
            '{"tool_name": "list_methods", "agent_name": "settlement"}\n'
            '</tool_call>'
        )
        result = parse_response(text)
        assert result.tool_call.tool_name == "list_methods"
        assert result.tool_call.arguments["agent_name"] == "settlement"

    def test_valid_describe_method(self):
        text = (
            '<tool_call>\n'
            '{"tool_name": "describe_method", "agent_name": "axial_pile", '
            '"method": "axial_pile_capacity"}\n'
            '</tool_call>'
        )
        result = parse_response(text)
        assert result.tool_call.tool_name == "describe_method"
        assert result.tool_call.arguments["method"] == "axial_pile_capacity"

    def test_valid_list_agents(self):
        text = '<tool_call>\n{"tool_name": "list_agents"}\n</tool_call>'
        result = parse_response(text)
        assert result.tool_call.tool_name == "list_agents"
        assert result.tool_call.arguments == {}

    def test_malformed_json_raises(self):
        text = '<tool_call>\n{bad json}\n</tool_call>'
        with pytest.raises(ValueError, match="Malformed JSON"):
            parse_response(text)

    def test_missing_tool_name_raises(self):
        text = '<tool_call>\n{"agent_name": "bearing_capacity"}\n</tool_call>'
        with pytest.raises(ValueError, match="Missing 'tool_name'"):
            parse_response(text)

    def test_invalid_tool_name_raises(self):
        text = '<tool_call>\n{"tool_name": "delete_everything"}\n</tool_call>'
        with pytest.raises(ValueError, match="Invalid tool_name"):
            parse_response(text)

    def test_non_object_json_raises(self):
        text = '<tool_call>\n[1, 2, 3]\n</tool_call>'
        with pytest.raises(ValueError, match="must be an object"):
            parse_response(text)

    def test_multiple_tags_uses_first(self):
        """Only the first <tool_call> is parsed."""
        text = (
            '<tool_call>\n{"tool_name": "list_agents"}\n</tool_call>\n'
            '<tool_call>\n{"tool_name": "list_methods", "agent_name": "settlement"}\n</tool_call>'
        )
        result = parse_response(text)
        assert result.tool_call.tool_name == "list_agents"

    def test_whitespace_in_tags(self):
        """Whitespace inside tags is handled."""
        text = '<tool_call>  \n  {"tool_name": "list_agents"}  \n  </tool_call>'
        result = parse_response(text)
        assert result.tool_call.tool_name == "list_agents"

    def test_reasoning_extraction(self):
        """Text before the tag is captured as reasoning."""
        text = (
            "Thought: Let me list the agents first.\n\n"
            '<tool_call>\n{"tool_name": "list_agents"}\n</tool_call>'
        )
        result = parse_response(text)
        assert "list the agents" in result.tool_call.reasoning

    def test_valid_tools_set(self):
        """VALID_TOOLS contains the expected tools."""
        assert VALID_TOOLS == {"call_agent", "list_methods", "describe_method",
                                "list_agents"}


# ===================================================================
# System prompt tests
# ===================================================================

class TestSystemPrompt:
    """Tests for ReAct prompt construction."""

    def test_full_prompt_contains_react_prefix(self):
        prompt = build_system_prompt()
        assert "<tool_call>" in prompt
        assert "list_agents" in prompt
        assert "call_agent" in prompt

    def test_full_prompt_contains_agent_catalog(self):
        prompt = build_system_prompt()
        assert "bearing_capacity" in prompt
        assert "44" in prompt
        assert "SI" in prompt

    def test_compact_prompt_has_quick_reference(self):
        prompt = build_system_prompt_compact()
        assert "Quick Reference" in prompt
        assert "<tool_call>" in prompt

    def test_compact_prompt_shorter_than_full(self):
        full = build_system_prompt()
        compact = build_system_prompt_compact()
        assert len(compact) < len(full)


# ===================================================================
# Conversation history tests
# ===================================================================

class TestConversationHistory:
    """Tests for conversation formatting."""

    def test_empty_history(self):
        h = ConversationHistory()
        prompt = h.format_prompt()
        assert "[Assistant]" in prompt
        assert len(h) == 0

    def test_single_user_message(self):
        h = ConversationHistory()
        h.add_user("What is bearing capacity?")
        prompt = h.format_prompt()
        assert "[User]" in prompt
        assert "bearing capacity" in prompt
        assert prompt.endswith("[Assistant]\n")

    def test_multi_turn_formatting(self):
        h = ConversationHistory()
        h.add_user("Calculate BC")
        h.add_assistant("Thought: I need to call bearing_capacity agent.")
        h.add_tool_result('{"q_ultimate_kPa": 500}')
        prompt = h.format_prompt()
        assert "[User]" in prompt
        assert "[Assistant]" in prompt
        assert "<tool_result>" in prompt
        assert "500" in prompt

    def test_trailing_assistant_tag(self):
        """Prompt always ends with [Assistant] to cue generation."""
        h = ConversationHistory()
        h.add_user("Hi")
        prompt = h.format_prompt()
        assert prompt.strip().endswith("[Assistant]")

    def test_clear(self):
        h = ConversationHistory()
        h.add_user("Test")
        h.add_assistant("Response")
        assert len(h) == 2
        h.clear()
        assert len(h) == 0

    def test_token_estimate(self):
        h = ConversationHistory()
        h.add_user("A" * 400)  # ~100 tokens
        est = h.token_estimate()
        assert 80 <= est <= 120

    def test_tool_result_wrapped_in_tags(self):
        h = ConversationHistory()
        h.add_tool_result('{"result": 42}')
        prompt = h.format_prompt()
        assert "<tool_result>" in prompt
        assert "</tool_result>" in prompt


# ===================================================================
# Agent loop tests (mock chat function)
# ===================================================================

class TestAgentLoop:
    """Tests for the full ReAct loop with mock chat functions."""

    def test_direct_answer_no_tools(self):
        """Model answers directly without tool calls."""
        chat_fn = make_scripted_chat([
            "The answer is 42 kPa. No tools needed."
        ])
        agent = GeotechChatAgent(chat_fn=chat_fn)
        result = agent.ask("What is 42?")
        assert "42 kPa" in result.answer
        assert result.rounds == 1
        assert len(result.tool_calls) == 0

    def test_single_tool_call_round(self):
        """Model calls one tool, then gives final answer."""
        n = len(AGENT_NAMES)
        chat_fn = make_scripted_chat([
            # Round 1: list_agents tool call
            'Thought: Let me see what agents are available.\n\n'
            '<tool_call>\n{"tool_name": "list_agents"}\n</tool_call>',
            # Round 2: final answer (no tool call)
            f'There are {n} agents available covering geotechnical engineering.',
        ])
        agent = GeotechChatAgent(chat_fn=chat_fn)
        result = agent.ask("What agents are available?")
        assert str(n) in result.answer
        assert result.rounds == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool_name"] == "list_agents"

    def test_multi_round_tool_calls(self):
        """Model uses list_methods → describe_method → call_agent → answer."""
        chat_fn = make_scripted_chat([
            # Round 1: list_methods
            '<tool_call>\n{"tool_name": "list_methods", "agent_name": "bearing_capacity"}\n</tool_call>',
            # Round 2: describe_method
            '<tool_call>\n{"tool_name": "describe_method", "agent_name": "bearing_capacity", '
            '"method": "bearing_capacity_factors"}\n</tool_call>',
            # Round 3: call_agent
            '<tool_call>\n{"tool_name": "call_agent", "agent_name": "bearing_capacity", '
            '"method": "bearing_capacity_factors", "parameters": {"phi": 30}}\n</tool_call>',
            # Round 4: final answer
            'The bearing capacity factors for phi=30 are Nc=30.14, Nq=18.40.',
        ])
        agent = GeotechChatAgent(chat_fn=chat_fn)
        result = agent.ask("What are the bearing capacity factors for phi=30?")
        assert result.rounds == 4
        assert len(result.tool_calls) == 3
        assert result.tool_calls[0]["tool_name"] == "list_methods"
        assert result.tool_calls[1]["tool_name"] == "describe_method"
        assert result.tool_calls[2]["tool_name"] == "call_agent"

    def test_max_rounds_reached(self):
        """Agent stops after max_rounds and returns partial result."""
        # All responses are tool calls — never gives a final answer
        responses = [
            '<tool_call>\n{"tool_name": "list_agents"}\n</tool_call>'
        ] * 5
        chat_fn = make_scripted_chat(responses)
        agent = GeotechChatAgent(chat_fn=chat_fn, max_rounds=3)
        result = agent.ask("Keep calling tools forever")
        assert result.rounds == 3
        assert "maximum" in result.answer.lower()
        assert len(result.tool_calls) == 3

    def test_error_recovery_malformed_json(self):
        """Malformed JSON triggers error feedback, model self-corrects."""
        chat_fn = make_scripted_chat([
            # Round 1: malformed JSON
            '<tool_call>\n{bad json here}\n</tool_call>',
            # Round 2: corrected, gives final answer
            'Sorry about that. The answer is 100 kPa.',
        ])
        agent = GeotechChatAgent(chat_fn=chat_fn)
        result = agent.ask("Calculate something")
        assert "100 kPa" in result.answer
        assert result.rounds == 2

    def test_multi_turn_conversation(self):
        """Follow-up questions see prior context."""
        chat_fn = make_scripted_chat([
            "The bearing capacity is 500 kPa.",
            "Based on the previous result of 500 kPa, settlement is 25 mm.",
        ])
        agent = GeotechChatAgent(chat_fn=chat_fn)

        r1 = agent.ask("Calculate BC")
        assert "500" in r1.answer

        r2 = agent.ask("Now estimate settlement")
        assert "25 mm" in r2.answer
        # History should have both Q&A pairs
        assert len(agent.history) == 4  # user, asst, user, asst

    def test_reset_clears_history(self):
        """Reset starts a fresh conversation."""
        chat_fn = make_scripted_chat([
            "First answer.",
            "Second answer after reset.",
        ])
        agent = GeotechChatAgent(chat_fn=chat_fn)
        agent.ask("First question")
        assert len(agent.history) > 0
        agent.reset()
        assert len(agent.history) == 0

    def test_verbose_mode(self, capsys):
        """Verbose mode prints round info."""
        chat_fn = make_scripted_chat([
            '<tool_call>\n{"tool_name": "list_agents"}\n</tool_call>',
            "Final answer here.",
        ])
        agent = GeotechChatAgent(chat_fn=chat_fn, verbose=True)
        agent.ask("Test verbose")
        captured = capsys.readouterr()
        assert "Round 1" in captured.out
        assert "list_agents" in captured.out

    def test_on_tool_call_callback(self):
        """Callback fires for each tool call."""
        log = []

        def callback(name, args, result):
            log.append(name)

        chat_fn = make_scripted_chat([
            '<tool_call>\n{"tool_name": "list_agents"}\n</tool_call>',
            "Done.",
        ])
        agent = GeotechChatAgent(chat_fn=chat_fn, on_tool_call=callback)
        agent.ask("Test callback")
        assert log == ["list_agents"]


# ===================================================================
# Utility tests
# ===================================================================

class TestUtilities:
    """Tests for helper functions."""

    def test_truncate_short_text(self):
        assert _truncate("hello", 100) == "hello"

    def test_truncate_long_text(self):
        text = "x" * 200
        result = _truncate(text, 50)
        assert len(result) < 200
        assert "[truncated]" in result
        assert result.startswith("x" * 50)

    def test_truncate_exact_limit(self):
        text = "x" * 100
        assert _truncate(text, 100) == text


# ===================================================================
# Dispatch tests
# ===================================================================

class TestDispatch:
    """Tests for tool dispatch routing."""

    def test_dispatch_unknown_tool(self):
        tc = ToolCall(
            tool_name="nonexistent",
            arguments={},
            raw_json="{}",
            reasoning="",
        )
        result_str = dispatch_tool(tc)
        result = json.loads(result_str)
        assert "error" in result

    def test_dispatch_list_agents(self):
        tc = ToolCall(
            tool_name="list_agents",
            arguments={},
            raw_json='{"tool_name": "list_agents"}',
            reasoning="",
        )
        result_str = dispatch_tool(tc)
        result = json.loads(result_str)
        assert "bearing_capacity" in result
        assert "settlement" in result
        assert len(result) == len(AGENT_NAMES)

    def test_dispatch_list_methods_real(self):
        """Test dispatch with real agent_registry for list_methods."""
        tc = ToolCall(
            tool_name="list_methods",
            arguments={"agent_name": "bearing_capacity", "category": ""},
            raw_json="{}",
            reasoning="",
        )
        result_str = dispatch_tool(tc)
        result = json.loads(result_str)
        # Should have methods, not an error
        assert "error" not in result

    def test_dispatch_call_agent_unknown_agent(self):
        """Unknown agent name returns error dict."""
        tc = ToolCall(
            tool_name="call_agent",
            arguments={
                "agent_name": "nonexistent_agent",
                "method": "some_method",
                "parameters": {},
            },
            raw_json="{}",
            reasoning="",
        )
        result_str = dispatch_tool(tc)
        result = json.loads(result_str)
        assert "error" in result
        assert "Unknown agent" in result["error"]


# ===================================================================
# AgentResult tests
# ===================================================================

class TestAgentResult:
    """Tests for the AgentResult dataclass."""

    def test_default_values(self):
        r = AgentResult(answer="test")
        assert r.answer == "test"
        assert r.tool_calls == []
        assert r.rounds == 0
        assert r.total_time_s == 0.0
        assert r.conversation_turns == 0

    def test_timing(self):
        """ask() records elapsed time."""
        chat_fn = make_scripted_chat(["Quick answer."])
        agent = GeotechChatAgent(chat_fn=chat_fn)
        result = agent.ask("How fast?")
        assert result.total_time_s >= 0
        assert result.total_time_s < 5.0  # should be near-instant with mock
