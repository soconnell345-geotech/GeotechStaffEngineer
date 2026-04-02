"""Tests for native OpenAI tool calling — NativeToolEngine + native_tools.

All tests use mock objects — no API key or PrompterAPI needed.
"""

import json
import pytest

from funhouse_agent.agent import GeotechAgent
from funhouse_agent.engine import NativeToolEngine
from funhouse_agent.native_tools import (
    OPENAI_TOOLS,
    EXTENDED_TOOL_NAMES,
    dispatch_native_tool,
)


# ---------------------------------------------------------------------------
# Mock PrompterAPI + OpenAI client
# ---------------------------------------------------------------------------

class MockMessage:
    """A single message from the mock OpenAI API."""

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockChoice:
    def __init__(self, message):
        self.message = message


class MockResponse:
    def __init__(self, message):
        self.choices = [MockChoice(message)]


class MockToolCall:
    """Mimics openai.types.chat.ChatCompletionMessageToolCall."""

    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = MockFunction(name, arguments)
        self.type = "function"


class MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = json.dumps(arguments)


class MockChatCompletions:
    """Mock for client.chat.completions.create()."""

    def __init__(self, responses):
        self._responses = responses
        self._call_index = 0
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return MockResponse(MockMessage(content="No more responses"))


class MockChat:
    def __init__(self, completions):
        self.completions = completions


class MockClient:
    def __init__(self, responses):
        self.chat = MockChat(MockChatCompletions(responses))


class MockPrompter:
    """Minimal mock of PrompterAPI — just .client and .chat_model."""

    def __init__(self, responses, chat_model="gpt-4o"):
        self.client = MockClient(responses)
        self.chat_model = chat_model

    def get_embedding(self, text):
        return [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_response(name, arguments, call_id="call_001"):
    """Build a MockResponse with a single tool call."""
    tc = MockToolCall(call_id, name, arguments)
    return MockResponse(MockMessage(tool_calls=[tc]))


def _make_final_response(content):
    """Build a MockResponse with a plain text answer."""
    return MockResponse(MockMessage(content=content))


# ---------------------------------------------------------------------------
# Tests: Tool schema structure
# ---------------------------------------------------------------------------

class TestToolSchemas:
    def test_seven_tools_defined(self):
        assert len(OPENAI_TOOLS) == 7

    def test_all_have_function_type(self):
        for tool in OPENAI_TOOLS:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "parameters" in tool["function"]

    def test_tool_names(self):
        names = {t["function"]["name"] for t in OPENAI_TOOLS}
        expected = {
            "list_agents", "list_methods", "describe_method", "call_agent",
            "analyze_image", "analyze_pdf_page", "save_file",
        }
        assert names == expected

    def test_call_agent_has_nested_parameters(self):
        call_agent_tool = next(
            t for t in OPENAI_TOOLS
            if t["function"]["name"] == "call_agent"
        )
        props = call_agent_tool["function"]["parameters"]["properties"]
        assert "parameters" in props
        assert props["parameters"]["type"] == "object"
        assert props["parameters"].get("additionalProperties") is True

    def test_extended_tool_names(self):
        assert EXTENDED_TOOL_NAMES == {
            "analyze_image", "analyze_pdf_page", "save_file",
        }


# ---------------------------------------------------------------------------
# Tests: dispatch_native_tool
# ---------------------------------------------------------------------------

class TestDispatchNativeTool:
    def test_list_agents(self):
        result = json.loads(dispatch_native_tool("list_agents", {}))
        assert isinstance(result, dict)
        assert "bearing_capacity" in result

    def test_list_methods(self):
        result = json.loads(dispatch_native_tool(
            "list_methods",
            {"agent_name": "bearing_capacity"},
        ))
        assert isinstance(result, dict)
        assert not result.get("error")

    def test_describe_method(self):
        result = json.loads(dispatch_native_tool(
            "describe_method",
            {"agent_name": "bearing_capacity",
             "method": "bearing_capacity_analysis"},
        ))
        assert "parameters" in result

    def test_call_agent_valid(self):
        result = json.loads(dispatch_native_tool(
            "call_agent",
            {
                "agent_name": "bearing_capacity",
                "method": "bearing_capacity_analysis",
                "parameters": {
                    "width": 2.0,
                    "length": 2.0,
                    "depth": 1.5,
                    "shape": "square",
                    "friction_angle": 30,
                    "unit_weight": 18.0,
                    "cohesion": 0,
                },
            },
        ))
        assert "error" not in result

    def test_call_agent_flattened_params(self):
        """LLM puts method params at top level instead of inside 'parameters'."""
        result = json.loads(dispatch_native_tool(
            "call_agent",
            {
                "agent_name": "bearing_capacity",
                "method": "bearing_capacity_analysis",
                "width": 2.0,
                "length": 2.0,
                "depth": 1.5,
                "shape": "square",
                "friction_angle": 30,
                "unit_weight": 18.0,
                "cohesion": 0,
            },
        ))
        assert "error" not in result

    def test_call_agent_mixed_params(self):
        """LLM sends some params nested, some flattened — both get merged."""
        result = json.loads(dispatch_native_tool(
            "call_agent",
            {
                "agent_name": "bearing_capacity",
                "method": "bearing_capacity_analysis",
                "parameters": {
                    "width": 2.0,
                    "length": 2.0,
                    "depth": 1.5,
                    "shape": "square",
                },
                "friction_angle": 30,
                "unit_weight": 18.0,
                "cohesion": 0,
            },
        ))
        assert "error" not in result

    def test_call_agent_bad_module(self):
        result = json.loads(dispatch_native_tool(
            "call_agent",
            {"agent_name": "nonexistent", "method": "x", "parameters": {}},
        ))
        assert "error" in result

    def test_unknown_tool(self):
        result = json.loads(dispatch_native_tool("bogus_tool", {}))
        assert "error" in result
        assert "Unknown tool" in result["error"]

    def test_save_file(self, tmp_path):
        outpath = str(tmp_path / "test.txt")
        result = json.loads(dispatch_native_tool(
            "save_file",
            {"path": outpath, "content": "hello"},
            save_fn=None,
        ))
        assert "saved" in result


# ---------------------------------------------------------------------------
# Tests: NativeToolEngine
# ---------------------------------------------------------------------------

class TestNativeToolEngine:
    def test_model_from_prompter(self):
        prompter = MockPrompter([], chat_model="gpt-4o-2024-11-20")
        engine = NativeToolEngine(prompter)
        assert engine.model == "gpt-4o-2024-11-20"

    def test_model_override(self):
        prompter = MockPrompter([], chat_model="gpt-4o")
        engine = NativeToolEngine(prompter, model="gpt-4o-mini")
        assert engine.model == "gpt-4o-mini"

    def test_model_updates_live(self):
        prompter = MockPrompter([], chat_model="gpt-4o")
        engine = NativeToolEngine(prompter)
        assert engine.model == "gpt-4o"
        prompter.chat_model = "gpt-4o-2025-01-15"
        assert engine.model == "gpt-4o-2025-01-15"

    def test_native_tool_calling_flag(self):
        prompter = MockPrompter([])
        engine = NativeToolEngine(prompter)
        assert engine.native_tool_calling is True

    def test_client_access(self):
        prompter = MockPrompter([])
        engine = NativeToolEngine(prompter)
        assert engine.client is prompter.client

    def test_chat_plain_text(self):
        resp = _make_final_response("Hello world")
        prompter = MockPrompter([resp])
        engine = NativeToolEngine(prompter)
        result = engine.chat("Hi")
        assert result == "Hello world"

    def test_get_embedding_delegates(self):
        prompter = MockPrompter([])
        engine = NativeToolEngine(prompter)
        result = engine.get_embedding("test")
        assert result == [0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Tests: GeotechAgent with NativeToolEngine
# ---------------------------------------------------------------------------

class TestGeotechAgentNative:
    def test_detects_native_engine(self):
        prompter = MockPrompter([])
        engine = NativeToolEngine(prompter)
        agent = GeotechAgent(genai_engine=engine)
        assert agent._is_native is True

    def test_non_native_engine_not_detected(self):
        class PlainEngine:
            def chat(self, user, system="", temperature=0):
                return "answer"
        agent = GeotechAgent(genai_engine=PlainEngine())
        assert agent._is_native is False

    def test_direct_answer_no_tools(self):
        """Model gives a final answer without calling any tools."""
        prompter = MockPrompter([
            _make_final_response("The answer is 42."),
        ])
        engine = NativeToolEngine(prompter)
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("What is 6 times 7?")
        assert "42" in result.answer
        assert result.rounds == 1
        assert len(result.tool_calls) == 0

    def test_single_tool_call_then_answer(self):
        """Model calls list_agents, then gives a final answer."""
        prompter = MockPrompter([
            _make_tool_response("list_agents", {}),
            _make_final_response("There are 50 modules available."),
        ])
        engine = NativeToolEngine(prompter)
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("What modules do you have?")
        assert "50" in result.answer
        assert result.rounds == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool_name"] == "list_agents"

    def test_multi_round_tool_workflow(self):
        """Model follows list_methods → describe_method → call_agent → answer."""
        prompter = MockPrompter([
            _make_tool_response(
                "list_methods",
                {"agent_name": "bearing_capacity"},
                call_id="call_1",
            ),
            _make_tool_response(
                "describe_method",
                {"agent_name": "bearing_capacity",
                 "method": "bearing_capacity_analysis"},
                call_id="call_2",
            ),
            _make_tool_response(
                "call_agent",
                {"agent_name": "bearing_capacity",
                 "method": "bearing_capacity_analysis",
                 "parameters": {
                     "width": 2.0, "length": 2.0, "depth": 1.5,
                     "shape": "square", "phi": 30, "gamma": 18.0, "c": 0,
                 }},
                call_id="call_3",
            ),
            _make_final_response("The ultimate bearing capacity is 1234 kPa."),
        ])
        engine = NativeToolEngine(prompter)
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("Calculate bearing capacity")
        assert result.rounds == 4
        assert len(result.tool_calls) == 3
        tools_used = [tc["tool_name"] for tc in result.tool_calls]
        assert tools_used == [
            "list_methods", "describe_method", "call_agent",
        ]

    def test_dispatch_error_logged(self):
        """Dispatch error (bad module) is captured in error_log."""
        prompter = MockPrompter([
            _make_tool_response(
                "call_agent",
                {"agent_name": "nonexistent", "method": "x",
                 "parameters": {}},
            ),
            _make_final_response("I encountered an error."),
        ])
        engine = NativeToolEngine(prompter)
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("Do something")
        assert len(result.errors) == 1
        assert result.errors[0]["type"] == "dispatch"

    def test_on_tool_call_hook_fires(self):
        """The _on_tool_call callback fires for each tool call."""
        hook_calls = []

        def hook(name, args, result_str):
            hook_calls.append(name)

        prompter = MockPrompter([
            _make_tool_response("list_agents", {}),
            _make_final_response("Done."),
        ])
        engine = NativeToolEngine(prompter)
        agent = GeotechAgent(
            genai_engine=engine, on_tool_call=hook,
        )
        agent.ask("List modules")
        assert hook_calls == ["list_agents"]

    def test_max_rounds_exhausted(self):
        """Agent stops after max_rounds even if model keeps calling tools."""
        # 5 rounds of list_agents, no final answer
        responses = [
            _make_tool_response("list_agents", {}, call_id=f"call_{i}")
            for i in range(5)
        ]
        prompter = MockPrompter(responses)
        engine = NativeToolEngine(prompter)
        agent = GeotechAgent(genai_engine=engine, max_rounds=3)
        result = agent.ask("Keep going forever")
        assert "maximum" in result.answer.lower()
        assert result.rounds == 3

    def test_tools_parameter_passed_to_client(self):
        """Verify that the tools parameter is actually sent to the client."""
        prompter = MockPrompter([
            _make_final_response("Answer."),
        ])
        engine = NativeToolEngine(prompter)
        agent = GeotechAgent(genai_engine=engine)
        agent.ask("Hi")
        calls = prompter.client.chat.completions.calls
        assert len(calls) == 1
        assert "tools" in calls[0]
        assert len(calls[0]["tools"]) == 7
        assert calls[0]["tool_choice"] == "auto"

    def test_model_name_passed_to_client(self):
        """Verify the model name from prompter reaches the API call."""
        prompter = MockPrompter(
            [_make_final_response("OK")],
            chat_model="gpt-4o-2025-03-15",
        )
        engine = NativeToolEngine(prompter)
        agent = GeotechAgent(genai_engine=engine)
        agent.ask("Hi")
        calls = prompter.client.chat.completions.calls
        assert calls[0]["model"] == "gpt-4o-2025-03-15"

    def test_tool_result_sent_back_to_model(self):
        """After a tool call, the result is appended as a 'tool' message."""
        prompter = MockPrompter([
            _make_tool_response("list_agents", {}, call_id="tc_42"),
            _make_final_response("Here are your modules."),
        ])
        engine = NativeToolEngine(prompter)
        agent = GeotechAgent(genai_engine=engine)
        agent.ask("Show me modules")

        # Second API call should include the tool result message
        calls = prompter.client.chat.completions.calls
        assert len(calls) == 2
        second_messages = calls[1]["messages"]
        tool_msgs = [m for m in second_messages
                     if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "tc_42"
        # Result should be valid JSON containing module names
        result_data = json.loads(tool_msgs[0]["content"])
        assert "bearing_capacity" in result_data


# ---------------------------------------------------------------------------
# Tests: Multiple tool calls in a single response
# ---------------------------------------------------------------------------

class TestParallelToolCalls:
    def test_multiple_tool_calls_single_response(self):
        """Model returns multiple tool_calls in one response."""
        tc1 = MockToolCall("call_a", "list_agents", {})
        tc2 = MockToolCall(
            "call_b", "list_methods",
            {"agent_name": "bearing_capacity"},
        )
        multi_msg = MockMessage(tool_calls=[tc1, tc2])
        multi_resp = MockResponse(multi_msg)

        prompter = MockPrompter([
            multi_resp,
            _make_final_response("Got both results."),
        ])
        engine = NativeToolEngine(prompter)
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("Show me everything")
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["tool_name"] == "list_agents"
        assert result.tool_calls[1]["tool_name"] == "list_methods"


# ---------------------------------------------------------------------------
# Tests: Multi-turn conversation persistence
# ---------------------------------------------------------------------------

class TestConversationPersistence:
    def test_follow_up_retains_context(self):
        """Second ask() should include messages from the first ask()."""
        prompter = MockPrompter([
            # Q1: tool call then answer
            _make_tool_response("list_agents", {}, call_id="call_q1"),
            _make_final_response("There are 50 modules."),
            # Q2: direct answer (follow-up)
            _make_final_response("Yes, bearing_capacity was in that list."),
        ])
        engine = NativeToolEngine(prompter)
        agent = GeotechAgent(genai_engine=engine)

        r1 = agent.ask("What modules do you have?")
        assert "50" in r1.answer

        r2 = agent.ask("Was bearing capacity in there?")
        assert "bearing_capacity" in r2.answer

        # The third API call (Q2) should contain full history:
        # system + user Q1 + assistant tool_call + tool result + assistant answer + user Q2
        calls = prompter.client.chat.completions.calls
        assert len(calls) == 3  # Q1 round1, Q1 round2, Q2
        q2_messages = calls[2]["messages"]
        # Should have more than just system + user (i.e., prior turns present)
        user_msgs = [
            m for m in q2_messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        assert len(user_msgs) == 2  # Q1 and Q2

    def test_reset_clears_native_messages(self):
        """reset() should clear conversation so next ask() starts fresh."""
        prompter = MockPrompter([
            _make_final_response("Answer to Q1."),
            _make_final_response("Answer to Q2."),
        ])
        engine = NativeToolEngine(prompter)
        agent = GeotechAgent(genai_engine=engine)

        agent.ask("Q1")
        agent.reset()
        agent.ask("Q2")

        # After reset, Q2's API call should have only system + user (no Q1)
        calls = prompter.client.chat.completions.calls
        assert len(calls) == 2
        q2_messages = calls[1]["messages"]
        user_msgs = [
            m for m in q2_messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        assert len(user_msgs) == 1
        assert user_msgs[0]["content"] == "Q2"
