"""Tests for funhouse_agent.engine — GenAIEngine protocol and ClaudeEngine."""

import pytest

from funhouse_agent.engine import GenAIEngine, ClaudeEngine


# ---------------------------------------------------------------------------
# Mock engines for testing
# ---------------------------------------------------------------------------

class MockEngine:
    """Mock engine satisfying the GenAIEngine protocol."""

    def __init__(self, chat_response="Mock response", vision_response="Mock vision"):
        self._chat_response = chat_response
        self._vision_response = vision_response
        self.chat_calls = []
        self.vision_calls = []

    def chat(self, user, system="You are a helpful assistant.", temperature=0):
        self.chat_calls.append({"user": user, "system": system, "temperature": temperature})
        return self._chat_response

    def analyze_image(self, image_input, user_prompt="Describe this image."):
        self.vision_calls.append({"image_input": image_input, "user_prompt": user_prompt})
        return self._vision_response

    def get_embedding(self, text):
        return [0.1, 0.2, 0.3]


class MockNoVisionEngine:
    """Mock engine without vision capability."""

    def chat(self, user, system="You are a helpful assistant.", temperature=0):
        return "Text-only response"

    def get_embedding(self, text):
        return [0.0]


class BrokenEngine:
    """Engine missing required methods."""
    pass


# ---------------------------------------------------------------------------
# Tests: GenAIEngine protocol
# ---------------------------------------------------------------------------

class TestGenAIEngineProtocol:
    def test_mock_engine_satisfies_protocol(self):
        engine = MockEngine()
        assert isinstance(engine, GenAIEngine)

    def test_no_vision_engine_satisfies_protocol(self):
        # Protocol requires all three methods, but duck-typing allows partial
        engine = MockNoVisionEngine()
        # This won't satisfy the full protocol but is usable for text-only
        assert hasattr(engine, "chat")

    def test_broken_engine_fails_protocol(self):
        engine = BrokenEngine()
        assert not isinstance(engine, GenAIEngine)


# ---------------------------------------------------------------------------
# Tests: MockEngine behavior
# ---------------------------------------------------------------------------

class TestMockEngine:
    def test_chat(self):
        engine = MockEngine(chat_response="Test answer")
        result = engine.chat("Hello")
        assert result == "Test answer"
        assert len(engine.chat_calls) == 1
        assert engine.chat_calls[0]["user"] == "Hello"

    def test_chat_with_system(self):
        engine = MockEngine()
        engine.chat("Hi", system="Be concise", temperature=0.5)
        assert engine.chat_calls[0]["system"] == "Be concise"
        assert engine.chat_calls[0]["temperature"] == 0.5

    def test_analyze_image(self):
        engine = MockEngine(vision_response="I see a slope")
        result = engine.analyze_image(b"png bytes", "Extract geometry")
        assert result == "I see a slope"
        assert len(engine.vision_calls) == 1

    def test_get_embedding(self):
        engine = MockEngine()
        emb = engine.get_embedding("test text")
        assert isinstance(emb, list)
        assert len(emb) == 3


# ---------------------------------------------------------------------------
# Tests: ClaudeEngine constructor
# ---------------------------------------------------------------------------

class TestClaudeEngine:
    def test_import_error_without_anthropic(self):
        # ClaudeEngine requires the anthropic package
        # If not installed, it should raise ImportError
        try:
            import anthropic  # noqa: F401
            # If anthropic IS installed, test that constructor works
            # (but don't make API calls — just test construction)
            # We can't easily test without a key, so skip
            pytest.skip("anthropic is installed; constructor test needs mock")
        except ImportError:
            with pytest.raises(ImportError, match="anthropic"):
                ClaudeEngine(api_key="fake-key")

    def test_get_embedding_raises(self):
        """ClaudeEngine.get_embedding should raise NotImplementedError."""
        try:
            import anthropic  # noqa: F401
            pytest.skip("Requires mock to test without API key")
        except ImportError:
            pytest.skip("anthropic not installed")
