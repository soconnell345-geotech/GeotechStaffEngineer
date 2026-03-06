"""Tests for funhouse_agent.notebook — NotebookChat ipywidgets wrapper.

All tests use mock engines — no API key or Jupyter kernel needed.
"""

import json
import pytest

ipywidgets = pytest.importorskip("ipywidgets")

from funhouse_agent.agent import GeotechAgent
from funhouse_agent.notebook import (
    NotebookChat,
    _escape,
    _format_answer,
)


# ---------------------------------------------------------------------------
# Mock engine (same pattern as test_agent.py)
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


def _make_chat(responses=None, **kwargs):
    """Create a NotebookChat with a mock engine."""
    engine = MockEngine(responses or ["Final answer: 42"])
    agent = GeotechAgent(genai_engine=engine, **kwargs)
    return NotebookChat(agent)


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_escape_html(self):
        assert _escape("<script>") == "&lt;script&gt;"

    def test_escape_ampersand(self):
        assert _escape("a & b") == "a &amp; b"

    def test_format_answer_bold(self):
        result = _format_answer("**bold** text")
        assert "<b>bold</b>" in result

    def test_format_answer_newlines(self):
        result = _format_answer("line1\nline2")
        assert "<br>" in result

    def test_format_answer_escapes_html(self):
        result = _format_answer("<script>alert(1)</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


# ---------------------------------------------------------------------------
# Stats rendering
# ---------------------------------------------------------------------------

class TestStats:
    def test_initial_stats(self):
        chat = _make_chat()
        html = chat._render_stats()
        assert "Tokens:" in html
        assert "Turns: 0" in html
        assert "Last:" not in html  # no result yet

    def test_stats_with_result(self):
        from chat_agent.agent import AgentResult

        result = AgentResult(
            answer="x", tool_calls=[], rounds=3,
            total_time_s=2.5, conversation_turns=4,
        )
        chat = _make_chat()
        html = chat._render_stats(result)
        assert "3 rounds" in html
        assert "2.5s" in html


# ---------------------------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_empty_chat(self):
        chat = _make_chat()
        html = chat._render_chat_html()
        assert "<style>" in html  # CSS block present

    def test_user_message_rendered(self):
        chat = _make_chat()
        chat._messages.append({"role": "user", "content": "Hello"})
        html = chat._render_chat_html()
        assert "You:" in html
        assert "Hello" in html

    def test_assistant_message_rendered(self):
        chat = _make_chat()
        chat._messages.append({"role": "assistant", "content": "The answer is 42"})
        html = chat._render_chat_html()
        assert "Agent:" in html
        assert "The answer is 42" in html

    def test_tool_call_rendered_as_details(self):
        chat = _make_chat()
        chat._messages.append({
            "role": "tool",
            "tool_name": "call_agent",
            "arguments": {"agent_name": "bearing_capacity", "method": "compute"},
            "result_preview": '{"q_ult": 500}',
        })
        html = chat._render_chat_html()
        assert "<details" in html
        assert "call_agent(bearing_capacity, compute)" in html

    def test_file_link_rendered(self):
        chat = _make_chat()
        chat._messages.append({
            "role": "file",
            "path": "/tmp/output.html",
            "format": "html",
        })
        html = chat._render_chat_html()
        assert "Output:" in html
        assert "/tmp/output.html" in html
        assert "(html)" in html

    def test_status_rendered_italic(self):
        chat = _make_chat()
        chat._messages.append({"role": "status", "content": "Thinking..."})
        html = chat._render_chat_html()
        assert "<em>" in html
        assert "Thinking..." in html

    def test_tool_summary_list_methods(self):
        chat = _make_chat()
        chat._messages.append({
            "role": "tool",
            "tool_name": "list_methods",
            "arguments": {"agent_name": "settlement"},
            "result_preview": "...",
        })
        html = chat._render_chat_html()
        assert "list_methods(settlement)" in html

    def test_tool_summary_save_file(self):
        chat = _make_chat()
        chat._messages.append({
            "role": "tool",
            "tool_name": "save_file",
            "arguments": {"path": "out.html"},
            "result_preview": "...",
        })
        html = chat._render_chat_html()
        assert "save_file(out.html)" in html


# ---------------------------------------------------------------------------
# Tool call detection
# ---------------------------------------------------------------------------

class TestToolCallDetection:
    def test_save_file_detected(self):
        chat = _make_chat()
        chat._handle_tool_call(
            "save_file",
            {"path": "report.html", "content": "<html>"},
            json.dumps({"saved": "/abs/report.html"}),
        )
        assert len(chat._output_files) == 1
        assert chat._output_files[0]["path"] == "/abs/report.html"
        assert chat._output_files[0]["format"] == "html"

    def test_calc_package_output_detected(self):
        chat = _make_chat()
        chat._handle_tool_call(
            "call_agent",
            {"agent_name": "calc_package", "method": "bearing_capacity_package"},
            json.dumps({
                "status": "success",
                "output_path": "/tmp/bc_calc.html",
                "format": "html",
            }),
        )
        assert len(chat._output_files) == 1
        assert chat._output_files[0]["path"] == "/tmp/bc_calc.html"

    def test_non_file_tool_call_no_output(self):
        chat = _make_chat()
        chat._handle_tool_call(
            "call_agent",
            {"agent_name": "bearing_capacity", "method": "compute"},
            json.dumps({"q_ult": 500}),
        )
        assert len(chat._output_files) == 0

    def test_pending_tool_calls_tracked(self):
        chat = _make_chat()
        chat._handle_tool_call(
            "list_agents", {}, json.dumps({"agents": ["a", "b"]}),
        )
        assert len(chat._pending_tool_calls) == 1
        assert chat._pending_tool_calls[0]["tool_name"] == "list_agents"

    def test_invalid_json_result_no_crash(self):
        chat = _make_chat()
        chat._handle_tool_call("call_agent", {}, "not json")
        assert len(chat._pending_tool_calls) == 1
        assert len(chat._output_files) == 0

    def test_original_callback_chained(self):
        captured = []

        def orig(name, args, res):
            captured.append(name)

        engine = MockEngine()
        agent = GeotechAgent(genai_engine=engine, on_tool_call=orig)
        chat = NotebookChat(agent)
        chat._handle_tool_call("list_agents", {}, "{}")
        assert captured == ["list_agents"]

    def test_output_files_property(self):
        chat = _make_chat()
        chat._output_files.append({"path": "/a.html", "format": "html"})
        chat._output_files.append({"path": "/b.pdf", "format": "pdf"})
        assert chat.output_files == ["/a.html", "/b.pdf"]


# ---------------------------------------------------------------------------
# Chat flow (send / reset)
# ---------------------------------------------------------------------------

class TestChatFlow:
    def test_send_adds_messages(self):
        chat = _make_chat(["The bearing capacity is 500 kPa."])
        chat._input.value = "What is the bearing capacity?"
        chat._on_send()

        roles = [m["role"] for m in chat._messages]
        assert "user" in roles
        assert "assistant" in roles
        # Thinking status should be removed
        assert "status" not in roles

    def test_send_user_content_correct(self):
        chat = _make_chat(["Answer"])
        chat._input.value = "Hello"
        chat._on_send()

        user_msgs = [m for m in chat._messages if m["role"] == "user"]
        assert user_msgs[0]["content"] == "Hello"

    def test_send_clears_input(self):
        chat = _make_chat(["Answer"])
        chat._input.value = "Test"
        chat._on_send()
        assert chat._input.value == ""

    def test_empty_input_ignored(self):
        chat = _make_chat()
        chat._input.value = "   "
        chat._on_send()
        assert len(chat._messages) == 0

    def test_send_during_processing_ignored(self):
        chat = _make_chat()
        chat._is_processing = True
        chat._input.value = "Test"
        chat._on_send()
        assert len(chat._messages) == 0

    def test_error_shows_in_chat(self):
        engine = MockEngine()
        engine.chat = lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("API down")
        )
        agent = GeotechAgent(genai_engine=engine)
        chat = NotebookChat(agent)
        chat._input.value = "test"
        chat._on_send()

        error_msgs = [
            m for m in chat._messages
            if m["role"] == "assistant" and "Error" in m["content"]
        ]
        assert len(error_msgs) == 1
        assert "RuntimeError" in error_msgs[0]["content"]

    def test_reset_clears_everything(self):
        chat = _make_chat(["Answer"])
        chat._input.value = "Hello"
        chat._on_send()
        assert len(chat._messages) > 0

        chat._on_reset()
        assert len(chat._messages) == 0
        assert len(chat._output_files) == 0
        assert chat._attachment_badges.value == ""

    def test_send_button_reenabled_after_send(self):
        chat = _make_chat(["Answer"])
        chat._input.value = "Test"
        chat._on_send()
        assert chat._send_btn.disabled is False
        assert chat._is_processing is False


# ---------------------------------------------------------------------------
# Attachment upload
# ---------------------------------------------------------------------------

class TestUpload:
    def test_upload_v8_format(self):
        """ipywidgets 8.x: value is a tuple of dicts — call handler directly."""
        chat = _make_chat()
        fake_value = (
            {"name": "plan.png", "content": b"\x89PNG"},
        )
        # Simulate the observe callback without setting widget.value
        chat._on_upload_change({"new": fake_value})
        assert "plan.png" in chat._agent.attachments

    def test_upload_v7_format(self):
        """ipywidgets 7.x: value is a dict of {name: {content: bytes}}."""
        chat = _make_chat()
        fake_value = {"site.jpg": {"content": b"\xff\xd8"}}
        # Simulate the observe callback without setting widget.value
        chat._on_upload_change({"new": fake_value})
        assert "site.jpg" in chat._agent.attachments

    def test_badges_updated_after_upload(self):
        chat = _make_chat()
        chat._agent.add_attachment("test.png", b"\x89PNG")
        chat._update_attachment_badges()
        assert "test.png" in chat._attachment_badges.value

    def test_attach_from_path(self, tmp_path):
        """chat.attach(path) loads a file and adds it as an attachment."""
        img = tmp_path / "site_plan.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        chat = _make_chat()
        chat.attach(str(img))
        assert "site_plan.png" in chat._agent.attachments
        assert chat._agent.attachments["site_plan.png"][:4] == b"\x89PNG"

    def test_attach_custom_key(self, tmp_path):
        """chat.attach(path, key=...) uses the custom key."""
        f = tmp_path / "boring_name.jpg"
        f.write_bytes(b"\xff\xd8\xff")
        chat = _make_chat()
        chat.attach(str(f), key="cross_section")
        assert "cross_section" in chat._agent.attachments
        assert "boring_name.jpg" not in chat._agent.attachments

    def test_attach_updates_badges(self, tmp_path):
        """chat.attach() updates the badge display."""
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-1.4")
        chat = _make_chat()
        chat.attach(str(f))
        assert "report.pdf" in chat._attachment_badges.value


# ---------------------------------------------------------------------------
# Widget structure
# ---------------------------------------------------------------------------

class TestWidgetStructure:
    def test_display_returns_vbox(self):
        chat = _make_chat()
        container = chat.display()
        assert isinstance(container, ipywidgets.VBox)

    def test_container_has_four_children(self):
        chat = _make_chat()
        # stats, chat, upload row, input row
        assert len(chat._container.children) == 4
