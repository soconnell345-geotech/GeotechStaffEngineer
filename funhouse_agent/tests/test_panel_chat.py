"""Tests for funhouse_agent.panel_chat — Panel chat interface.

All tests use mock engines — no API key or Panel server needed.
"""

import json
import pytest

pn = pytest.importorskip("panel")

from funhouse_agent.agent import GeotechAgent
from funhouse_agent.panel_chat import (
    ChatApp,
    _tool_summary,
    _format_tool_call,
    _format_file_output,
)


# ---------------------------------------------------------------------------
# Mock engine
# ---------------------------------------------------------------------------

class MockEngine:
    def __init__(self, responses=None):
        self._responses = responses or ["Final answer: 42"]
        self._idx = 0

    def chat(self, user, system="", temperature=0):
        if self._idx < len(self._responses):
            r = self._responses[self._idx]
            self._idx += 1
            return r
        return "No more responses"

    def analyze_image(self, image_input, user_prompt=""):
        return "I see a cross-section"

    def get_embedding(self, text):
        return [0.1]


def _make_app(responses=None, **kwargs):
    """Create a ChatApp with a mock engine."""
    engine = MockEngine(responses or ["Final answer: 42"])
    agent = GeotechAgent(genai_engine=engine, **kwargs)
    return ChatApp(agent), agent


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestToolSummary:
    def test_call_agent(self):
        s = _tool_summary("call_agent", {"agent_name": "bearing", "method": "vesic"})
        assert "bearing" in s
        assert "vesic" in s

    def test_list_methods(self):
        s = _tool_summary("list_methods", {"agent_name": "settlement"})
        assert "settlement" in s

    def test_describe_method(self):
        s = _tool_summary("describe_method", {"agent_name": "pile", "method": "nordlund"})
        assert "pile" in s and "nordlund" in s

    def test_analyze_image(self):
        s = _tool_summary("analyze_image", {"attachment_key": "plan.png"})
        assert "plan.png" in s

    def test_save_file(self):
        s = _tool_summary("save_file", {"path": "output/report.html"})
        assert "output/report.html" in s

    def test_unknown_tool(self):
        s = _tool_summary("list_agents", {})
        assert s == "list_agents"


class TestFormatToolCall:
    def test_basic_format(self):
        md = _format_tool_call(
            "call_agent",
            {"agent_name": "bearing", "method": "vesic"},
            '{"qult": 500}',
        )
        assert "<details>" in md
        assert "call_agent" in md
        assert "bearing" in md
        assert "qult" in md

    def test_empty_preview(self):
        md = _format_tool_call("list_agents", {}, "")
        assert "<details>" in md


class TestFormatFileOutput:
    def test_basic(self):
        md = _format_file_output("/tmp/report.html", "html")
        assert "/tmp/report.html" in md
        assert "html" in md


# ---------------------------------------------------------------------------
# ChatApp construction
# ---------------------------------------------------------------------------

class TestChatAppInit:
    def test_creates_layout(self):
        app, _ = _make_app()
        layout = app.panel()
        assert layout is not None

    def test_custom_title(self):
        app, _ = _make_app()
        app2 = ChatApp(
            GeotechAgent(genai_engine=MockEngine()),
            title="Custom Title",
        )
        layout = app2.panel()
        assert layout is not None

    def test_output_files_empty(self):
        app, _ = _make_app()
        assert app.output_files == []


# ---------------------------------------------------------------------------
# Tool call capture
# ---------------------------------------------------------------------------

class TestToolCallCapture:
    def test_capture_standard_tool(self):
        app, agent = _make_app()
        app._capture_tool_call(
            "call_agent",
            {"agent_name": "bearing", "method": "vesic"},
            '{"qult": 500}',
        )
        assert len(app._tool_calls) == 1
        assert app._tool_calls[0]["tool_name"] == "call_agent"

    def test_capture_save_file(self):
        app, agent = _make_app()
        app._capture_tool_call(
            "save_file",
            {"path": "output/report.html"},
            json.dumps({"saved": "/abs/output/report.html"}),
        )
        assert len(app._output_files) == 1
        assert app._output_files[0]["path"] == "/abs/output/report.html"
        assert app._output_files[0]["format"] == "html"

    def test_capture_calc_package(self):
        app, agent = _make_app()
        app._capture_tool_call(
            "call_agent",
            {"agent_name": "calc_package", "method": "render"},
            json.dumps({"output_path": "/tmp/calc.html", "status": "success",
                         "format": "html"}),
        )
        assert len(app._output_files) == 1
        assert app._output_files[0]["format"] == "html"

    def test_capture_non_json_result(self):
        app, agent = _make_app()
        app._capture_tool_call("list_agents", {}, "plain text result")
        assert len(app._tool_calls) == 1
        assert len(app._output_files) == 0

    def test_capture_error_result(self):
        app, agent = _make_app()
        app._capture_tool_call(
            "call_agent",
            {"agent_name": "bad", "method": "fail"},
            json.dumps({"error": "Module not found"}),
        )
        assert len(app._tool_calls) == 1
        assert len(app._output_files) == 0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_state(self):
        app, agent = _make_app()
        # Simulate some state
        app._tool_calls.append({"tool_name": "test"})
        app._output_files.append({"path": "/tmp/test", "format": "txt"})
        agent.add_attachment("test.png", b"fake")

        app._on_reset(None)

        assert app._tool_calls == []
        assert app._output_files == []
        assert len(agent.attachments) == 0


# ---------------------------------------------------------------------------
# Stats rendering
# ---------------------------------------------------------------------------

class TestStats:
    def test_initial_stats(self):
        app, _ = _make_app()
        html = app._render_stats()
        assert "Tokens: ~0" in html
        assert "Turns: 0" in html

    def test_stats_with_result(self):
        from funhouse_agent.react_support import AgentResult
        result = AgentResult(
            answer="test", rounds=3, total_time_s=5.2,
            conversation_turns=6,
        )
        app, _ = _make_app()
        html = app._render_stats(result)
        assert "3 rounds" in html
        assert "5.2s" in html

    def test_stats_with_errors(self):
        from funhouse_agent.react_support import AgentResult
        result = AgentResult(
            answer="test", rounds=1, total_time_s=1.0,
            errors=[{"type": "parse", "message": "bad"}],
        )
        app, _ = _make_app()
        html = app._render_stats(result)
        assert "1 error" in html


# ---------------------------------------------------------------------------
# Attachment badges
# ---------------------------------------------------------------------------

class TestAttachments:
    def test_badges_update(self):
        app, agent = _make_app()
        agent.add_attachment("plan.png", b"fake")
        agent.add_attachment("report.pdf", b"fake")
        app._update_badges()
        assert "plan.png" in app._attachment_badges.object
        assert "report.pdf" in app._attachment_badges.object

    def test_badges_empty(self):
        app, _ = _make_app()
        app._update_badges()
        assert app._attachment_badges.object == ""


# ---------------------------------------------------------------------------
# Message callback (integration)
# ---------------------------------------------------------------------------

class TestMessageCallback:
    def test_simple_answer(self):
        app, agent = _make_app(["The answer is 42."])
        results = list(app._on_message("What is 6*7?", "User", app._chat))
        # Last yield should be the answer
        assert any("42" in str(r) for r in results)

    def test_with_tool_call(self):
        responses = [
            '<tool_call>\n{"tool_name": "list_agents"}\n</tool_call>',
            "Here are the available agents.",
        ]
        app, agent = _make_app(responses)
        results = list(app._on_message("What tools?", "User", app._chat))
        assert len(results) >= 1

    def test_empty_message_ignored(self):
        app, agent = _make_app()
        results = list(app._on_message("", "User", app._chat))
        assert results == []

    def test_whitespace_message_ignored(self):
        app, agent = _make_app()
        results = list(app._on_message("   ", "User", app._chat))
        assert results == []

    def test_engine_exception(self):
        """Engine raising an exception should yield an error message."""
        class FailEngine:
            def chat(self, user, system="", temperature=0):
                raise RuntimeError("API down")
            def analyze_image(self, image_input, user_prompt=""):
                return ""

        agent = GeotechAgent(genai_engine=FailEngine())
        app = ChatApp(agent)
        results = list(app._on_message("test", "User", app._chat))
        assert any("Error" in str(r) for r in results)
