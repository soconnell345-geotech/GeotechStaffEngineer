"""Tests for funhouse_agent.agent — GeotechAgent with engine DI.

All tests use mock engines — no API key needed.
"""

import json
import pytest

from funhouse_agent.agent import GeotechAgent
from funhouse_agent.engine import GenAIEngine
from funhouse_agent.vision_tools import EXTENDED_TOOLS
from chat_agent.agent import AgentResult


# ---------------------------------------------------------------------------
# Mock engines
# ---------------------------------------------------------------------------

class MockEngine:
    """Mock engine that returns canned responses."""

    def __init__(self, responses=None):
        self._responses = responses or ["Final answer: 42"]
        self._call_index = 0
        self.chat_calls = []
        self.vision_calls = []

    def chat(self, user, system="", temperature=0):
        self.chat_calls.append(user)
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return "No more responses"

    def analyze_image(self, image_input, user_prompt=""):
        self.vision_calls.append({"image": image_input, "prompt": user_prompt})
        return "I see a geotechnical cross-section"

    def get_embedding(self, text):
        return [0.1, 0.2, 0.3]


class MockNoVisionEngine:
    """Engine without analyze_image."""

    def __init__(self, response="Final answer"):
        self._response = response

    def chat(self, user, system="", temperature=0):
        return self._response


# ---------------------------------------------------------------------------
# Tests: Constructor
# ---------------------------------------------------------------------------

class TestGeotechAgentConstructor:
    def test_creates_with_engine(self):
        engine = MockEngine()
        agent = GeotechAgent(genai_engine=engine)
        assert agent is not None

    def test_has_vision_with_engine(self):
        engine = MockEngine()
        agent = GeotechAgent(genai_engine=engine)
        assert agent.has_vision is True

    def test_no_vision_without_method(self):
        engine = MockNoVisionEngine()
        agent = GeotechAgent(genai_engine=engine)
        assert agent.has_vision is False

    def test_empty_attachments(self):
        engine = MockEngine()
        agent = GeotechAgent(genai_engine=engine)
        assert agent.attachments == {}


# ---------------------------------------------------------------------------
# Tests: Text ReAct
# ---------------------------------------------------------------------------

class TestTextReact:
    def test_simple_answer(self):
        engine = MockEngine(responses=["The bearing capacity is 500 kPa."])
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("What is bearing capacity?")
        assert isinstance(result, AgentResult)
        assert "500 kPa" in result.answer

    def test_tool_use_then_answer(self):
        engine = MockEngine(responses=[
            # Round 1: tool call
            'Let me check.\n<tool_call>\n'
            '{"tool_name": "list_agents"}\n'
            '</tool_call>',
            # Round 2: final answer
            "Based on the agents, I can help with bearing capacity.",
        ])
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("What agents are available?")
        assert result.rounds == 2
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool_name"] == "list_agents"

    def test_max_rounds(self):
        # All responses are tool calls — will hit max rounds
        tool_response = (
            '<tool_call>\n'
            '{"tool_name": "list_agents"}\n'
            '</tool_call>'
        )
        engine = MockEngine(responses=[tool_response] * 20)
        agent = GeotechAgent(genai_engine=engine, max_rounds=3)
        result = agent.ask("Keep looping")
        assert result.rounds == 3
        assert "maximum" in result.answer.lower()

    def test_parse_error_recovery(self):
        engine = MockEngine(responses=[
            # Bad tool call
            '<tool_call>\n{invalid json}\n</tool_call>',
            # Good answer after error
            "Sorry, here is the answer: 42",
        ])
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("What?")
        assert "42" in result.answer

    def test_verbose_mode(self, capsys):
        engine = MockEngine(responses=["The answer is 7."])
        agent = GeotechAgent(genai_engine=engine, verbose=True)
        agent.ask("Quick question")
        captured = capsys.readouterr()
        assert "FINAL ANSWER" in captured.out

    def test_on_tool_call_callback(self):
        calls = []

        def callback(name, args, result):
            calls.append(name)

        engine = MockEngine(responses=[
            '<tool_call>\n{"tool_name": "list_agents"}\n</tool_call>',
            "Done!",
        ])
        agent = GeotechAgent(genai_engine=engine, on_tool_call=callback)
        agent.ask("List agents")
        assert "list_agents" in calls


# ---------------------------------------------------------------------------
# Tests: Attachments
# ---------------------------------------------------------------------------

class TestAttachments:
    def test_add_attachment(self):
        engine = MockEngine()
        agent = GeotechAgent(genai_engine=engine)
        agent.add_attachment("site_plan", b"fake png data")
        assert "site_plan" in agent.attachments
        assert agent.attachments["site_plan"] == b"fake png data"

    def test_multiple_attachments(self):
        engine = MockEngine()
        agent = GeotechAgent(genai_engine=engine)
        agent.add_attachment("plan", b"data1")
        agent.add_attachment("section", b"data2")
        assert len(agent.attachments) == 2

    def test_reset_clears_attachments(self):
        engine = MockEngine()
        agent = GeotechAgent(genai_engine=engine)
        agent.add_attachment("plan", b"data")
        agent.ask("test")  # adds to history
        agent.reset()
        assert agent.attachments == {}
        assert len(agent.history) == 0


# ---------------------------------------------------------------------------
# Tests: Vision tools from ReAct loop
# ---------------------------------------------------------------------------

class TestVisionToolDispatch:
    def test_analyze_image_in_react(self):
        engine = MockEngine(responses=[
            '<tool_call>\n'
            '{"tool_name": "analyze_image", "attachment_key": "photo", '
            '"prompt": "Extract slope geometry"}\n'
            '</tool_call>',
            "Based on the image analysis, the slope angle is 30 degrees.",
        ])
        agent = GeotechAgent(genai_engine=engine)
        agent.add_attachment("photo", b"fake image bytes")
        result = agent.ask("Analyze this photo")
        assert result.rounds == 2
        assert result.tool_calls[0]["tool_name"] == "analyze_image"
        assert len(engine.vision_calls) == 1

    def test_missing_attachment_error(self):
        engine = MockEngine(responses=[
            '<tool_call>\n'
            '{"tool_name": "analyze_image", "attachment_key": "missing", '
            '"prompt": "Extract"}\n'
            '</tool_call>',
            "The attachment was not found.",
        ])
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("Analyze image")
        # The error should be fed back to the agent
        assert result.rounds == 2


# ---------------------------------------------------------------------------
# Tests: Direct methods (bypass ReAct)
# ---------------------------------------------------------------------------

class TestDirectMethods:
    def test_extract_geometry_from_image(self):
        def mock_vision(image_bytes, prompt):
            return json.dumps({
                "surface_points": [[0, 10], [20, 5]],
                "boundary_profiles": {},
            })

        engine = MockEngine()
        engine.analyze_image = mock_vision
        agent = GeotechAgent(genai_engine=engine)
        result = agent.extract_geometry_from_image(b"fake png")
        from pdf_import.results import PdfParseResult
        assert isinstance(result, PdfParseResult)
        assert len(result.surface_points) == 2

    def test_extract_geometry_from_pdf(self):
        """Test PDF geometry extraction (requires PyMuPDF for rendering)."""
        import fitz
        # Create a minimal PDF
        doc = fitz.open()
        doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()

        def mock_vision(image_bytes, prompt):
            return json.dumps({
                "surface_points": [[0, 15], [30, 10]],
                "boundary_profiles": {"Clay": [[0, 5], [30, 3]]},
            })

        engine = MockEngine()
        engine.analyze_image = mock_vision
        agent = GeotechAgent(genai_engine=engine)
        result = agent.extract_geometry_from_pdf(pdf_bytes, page=0)
        from pdf_import.results import PdfParseResult
        assert isinstance(result, PdfParseResult)
        assert "Clay" in result.boundary_profiles

    def test_analyze_pdf_report(self):
        import fitz
        doc = fitz.open()
        doc.new_page()
        pdf_bytes = doc.tobytes()
        doc.close()

        engine = MockEngine()
        engine.analyze_image = lambda img, prompt: "This is a boring log report."
        agent = GeotechAgent(genai_engine=engine)
        result = agent.analyze_pdf_report(pdf_bytes)
        assert "boring log" in result


# ---------------------------------------------------------------------------
# Tests: Engine swapping
# ---------------------------------------------------------------------------

class TestEngineSwapping:
    def test_same_question_different_engines(self):
        engine1 = MockEngine(responses=["Answer from engine 1"])
        engine2 = MockEngine(responses=["Answer from engine 2"])

        agent1 = GeotechAgent(genai_engine=engine1)
        agent2 = GeotechAgent(genai_engine=engine2)

        r1 = agent1.ask("Same question")
        r2 = agent2.ask("Same question")

        assert "engine 1" in r1.answer
        assert "engine 2" in r2.answer

    def test_no_vision_engine_text_only(self):
        engine = MockNoVisionEngine(response="Text only answer")
        agent = GeotechAgent(genai_engine=engine)
        assert agent.has_vision is False
        result = agent.ask("Simple question")
        assert "Text only" in result.answer


# ---------------------------------------------------------------------------
# Tests: Extended tool set
# ---------------------------------------------------------------------------

class TestExtendedTools:
    def test_extended_tools_include_standard(self):
        from chat_agent.parser import VALID_TOOLS
        for tool in VALID_TOOLS:
            assert tool in EXTENDED_TOOLS

    def test_extended_tools_include_vision(self):
        assert "analyze_image" in EXTENDED_TOOLS
        assert "analyze_pdf_page" in EXTENDED_TOOLS

    def test_standard_tools_still_work(self):
        engine = MockEngine(responses=[
            '<tool_call>\n'
            '{"tool_name": "list_agents"}\n'
            '</tool_call>',
            "Here are the agents: bearing_capacity, settlement, ...",
        ])
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("List available agents")
        assert result.tool_calls[0]["tool_name"] == "list_agents"


# ---------------------------------------------------------------------------
# Tests: History management
# ---------------------------------------------------------------------------

class TestHistoryManagement:
    def test_multi_turn_conversation(self):
        engine = MockEngine(responses=[
            "Bearing capacity is 500 kPa.",
            "Settlement would be about 25 mm.",
        ])
        agent = GeotechAgent(genai_engine=engine)
        r1 = agent.ask("What is bearing capacity?")
        r2 = agent.ask("And settlement?")
        assert len(agent.history) >= 4  # 2 user + 2 assistant turns

    def test_reset_clears_history(self):
        engine = MockEngine(responses=["Answer", "Answer2"])
        agent = GeotechAgent(genai_engine=engine)
        agent.ask("Question 1")
        assert len(agent.history) > 0
        agent.reset()
        assert len(agent.history) == 0


# ---------------------------------------------------------------------------
# Tests: save_file tool
# ---------------------------------------------------------------------------

class TestSaveFileTool:
    """Tests for the save_file tool via ReAct dispatch."""

    def test_save_file_default_local(self, tmp_path):
        """save_file with default save_fn writes to local filesystem."""
        out = tmp_path / "report.html"
        engine = MockEngine(responses=[
            '<tool_call>\n'
            '{"tool_name": "save_file", '
            f'"path": "{out}", '
            '"content": "<html><body>Report</body></html>"}\n'
            '</tool_call>',
            "File saved successfully.",
        ])
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("Save a report")
        assert result.rounds == 2
        assert result.tool_calls[0]["tool_name"] == "save_file"
        assert out.exists()
        assert "<html>" in out.read_text()

    def test_save_file_custom_save_fn(self):
        """save_fn injection routes through the custom function."""
        saved = {}

        def mock_save(path, content):
            saved[path] = content
            return f"/dbfs/{path}"

        engine = MockEngine(responses=[
            '<tool_call>\n'
            '{"tool_name": "save_file", '
            '"path": "output/report.html", '
            '"content": "hello"}\n'
            '</tool_call>',
            "Done.",
        ])
        agent = GeotechAgent(genai_engine=engine, save_fn=mock_save)
        agent.ask("Save it")
        assert "output/report.html" in saved
        assert saved["output/report.html"] == "hello"

    def test_save_file_base64_binary(self, tmp_path):
        """Base64-encoded content is decoded before saving."""
        import base64
        raw = b"\x89PNG\r\n\x1a\nfake png"
        b64 = base64.b64encode(raw).decode()
        out = tmp_path / "plot.png"
        engine = MockEngine(responses=[
            '<tool_call>\n'
            '{"tool_name": "save_file", '
            f'"path": "{out}", '
            f'"content": "{b64}", '
            '"encoding": "base64"}\n'
            '</tool_call>',
            "Saved plot.",
        ])
        agent = GeotechAgent(genai_engine=engine)
        agent.ask("Save the plot")
        assert out.exists()
        assert out.read_bytes() == raw

    def test_save_file_missing_path(self):
        """Missing path returns an error in the tool result."""
        engine = MockEngine(responses=[
            '<tool_call>\n'
            '{"tool_name": "save_file", '
            '"content": "data"}\n'
            '</tool_call>',
            "Error noted.",
        ])
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("Save without path")
        assert result.rounds == 2

    def test_save_file_missing_content(self):
        """Missing content returns an error in the tool result."""
        engine = MockEngine(responses=[
            '<tool_call>\n'
            '{"tool_name": "save_file", '
            '"path": "out.txt"}\n'
            '</tool_call>',
            "Error noted.",
        ])
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("Save without content")
        assert result.rounds == 2

    def test_save_file_creates_directories(self, tmp_path):
        """Nested directories are created automatically."""
        out = tmp_path / "deep" / "nested" / "dir" / "report.html"
        engine = MockEngine(responses=[
            '<tool_call>\n'
            '{"tool_name": "save_file", '
            f'"path": "{out}", '
            '"content": "nested report"}\n'
            '</tool_call>',
            "Done.",
        ])
        agent = GeotechAgent(genai_engine=engine)
        agent.ask("Save nested")
        assert out.exists()
        assert out.read_text() == "nested report"

    def test_save_file_invalid_base64(self):
        """Invalid base64 returns an error, not a crash."""
        engine = MockEngine(responses=[
            '<tool_call>\n'
            '{"tool_name": "save_file", '
            '"path": "out.pdf", '
            '"content": "not-valid-base64!!!", '
            '"encoding": "base64"}\n'
            '</tool_call>',
            "Error noted.",
        ])
        agent = GeotechAgent(genai_engine=engine)
        result = agent.ask("Bad base64")
        assert result.rounds == 2

    def test_save_fn_error_returns_json_error(self):
        """If save_fn raises, the error is returned as JSON, not re-raised."""
        def bad_save(path, content):
            raise PermissionError("Access denied")

        engine = MockEngine(responses=[
            '<tool_call>\n'
            '{"tool_name": "save_file", '
            '"path": "out.txt", '
            '"content": "data"}\n'
            '</tool_call>',
            "Permission error noted.",
        ])
        agent = GeotechAgent(genai_engine=engine, save_fn=bad_save)
        result = agent.ask("Try to save")
        assert result.rounds == 2

    def test_extended_tools_include_save_file(self):
        """save_file is in the extended tool set."""
        assert "save_file" in EXTENDED_TOOLS

    def test_save_file_callback_fires(self):
        """on_tool_call callback fires for save_file."""
        calls = []

        def callback(name, args, result):
            calls.append(name)

        saved = {}

        def mock_save(path, content):
            saved[path] = content
            return path

        engine = MockEngine(responses=[
            '<tool_call>\n'
            '{"tool_name": "save_file", '
            '"path": "out.txt", '
            '"content": "data"}\n'
            '</tool_call>',
            "Done.",
        ])
        agent = GeotechAgent(
            genai_engine=engine, save_fn=mock_save, on_tool_call=callback,
        )
        agent.ask("Save it")
        assert "save_file" in calls
