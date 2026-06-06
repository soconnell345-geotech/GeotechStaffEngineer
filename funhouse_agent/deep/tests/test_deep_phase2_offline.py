"""Offline tests for the v5.0 deepagents Phase 2 pieces (NO API key, NO network).

Covers:
  (a) LangChainVisionEngine.analyze_image — builds the standard multimodal
      HumanMessage (image_url data-URI block + text prompt) and returns the
      model's text. Driven by a stub model that records the messages it saw.
  (b) build_deep_agent(model=<fake>) with engine=None default-wraps vision —
      the vision tools are present and analyze_image is reachable through the
      wrapped fake model.
  (c) PrompterChatModel — message translation, tool_calls parsing, proxy-safe
      assistant dicts (no null fields), and bind_tools schema capture. Driven by
      a fake prompter whose .client.chat.completions.create returns a canned
      OpenAI-shaped object.

Run from the worktree root with the venv python::

    cd <worktree>
    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_deep_phase2_offline.py -v
"""

import json
from types import SimpleNamespace

import pytest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from funhouse_agent.deep.vision_engine import LangChainVisionEngine, _content_to_text
from funhouse_agent.deep.databricks_bridge import (
    PrompterChatModel,
    _lc_message_to_openai,
)


# ===========================================================================
# (a) LangChainVisionEngine
# ===========================================================================

class _RecordingModel:
    """A stub LangChain-ish model: records the messages handed to .invoke and
    returns a canned AIMessage. Never touches a network."""

    def __init__(self, reply="canned vision text"):
        self.reply = reply
        self.seen = None

    def invoke(self, messages):
        self.seen = messages
        return AIMessage(content=self.reply)


def test_vision_engine_returns_model_text():
    model = _RecordingModel("Kp is about 9.5 from the chart.")
    engine = LangChainVisionEngine(model)
    out = engine.analyze_image(b"\x89PNG fake bytes", "Read Kp off this chart.")
    assert out == "Kp is about 9.5 from the chart."


def test_vision_engine_builds_multimodal_message():
    model = _RecordingModel()
    engine = LangChainVisionEngine(model)
    engine.analyze_image(b"rawbytes", "Describe the cross-section.")

    # One HumanMessage with the image_url data-URI block + the text prompt.
    assert len(model.seen) == 1
    msg = model.seen[0]
    assert isinstance(msg, HumanMessage)
    assert isinstance(msg.content, list)

    image_blocks = [b for b in msg.content if b.get("type") == "image_url"]
    text_blocks = [b for b in msg.content if b.get("type") == "text"]
    assert len(image_blocks) == 1
    assert len(text_blocks) == 1

    url = image_blocks[0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    # The base64 payload decodes back to the original bytes.
    import base64
    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload) == b"rawbytes"
    assert text_blocks[0]["text"] == "Describe the cross-section."


def test_vision_engine_accepts_file_path(tmp_path):
    img = tmp_path / "chart.png"
    img.write_bytes(b"file-png-bytes")
    model = _RecordingModel("ok")
    engine = LangChainVisionEngine(model)
    out = engine.analyze_image(str(img), "What is this?")
    assert out == "ok"

    import base64
    url = model.seen[0].content[0]["image_url"]["url"]
    payload = url.split(",", 1)[1]
    assert base64.b64decode(payload) == b"file-png-bytes"


def test_vision_engine_rejects_bad_input():
    engine = LangChainVisionEngine(_RecordingModel())
    with pytest.raises(TypeError):
        engine.analyze_image(12345, "nope")


def test_content_to_text_handles_block_lists():
    # str passthrough
    assert _content_to_text("plain") == "plain"
    # list of text blocks (Claude-style)
    blocks = [
        {"type": "text", "text": "Hello "},
        {"type": "image_url", "image_url": {"url": "x"}},
        {"type": "text", "text": "world"},
    ]
    assert _content_to_text(blocks) == "Hello world"


# ===========================================================================
# (b) build_deep_agent default-wraps vision
# ===========================================================================

def _fake_model():
    return GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))


def _compiled_tool_names(agent):
    return set(agent.nodes["tools"].bound.tools_by_name.keys())


def test_build_deep_agent_default_wraps_vision_tools_present():
    """engine=None + a model object → vision tools are bound on the primary."""
    from funhouse_agent.deep.agent import build_deep_agent

    agent = build_deep_agent(model=_fake_model())
    names = _compiled_tool_names(agent)
    for expected in ("analyze_image", "analyze_pdf_page",
                     "read_reference_figure", "save_file"):
        assert expected in names, f"{expected} missing from {sorted(names)}"


def test_build_deep_agent_vision_reachable_through_wrapped_model():
    """The default-wrapped engine actually routes analyze_image through the
    model: the bound analyze_image tool returns the model's text.

    We drive the primary's analyze_image tool directly with an attachment so no
    LLM planning is needed — only the vision engine call, which hits our
    recording model rather than a network."""
    from funhouse_agent.deep.agent import build_primary_tools
    from funhouse_agent.dispatch import ANALYSIS_MODULES

    model = _RecordingModel("chart shows N_q approx 18")
    engine = LangChainVisionEngine(model)
    tools = build_primary_tools(
        allowed_agents=ANALYSIS_MODULES,
        engine=engine,
        attachments={"site": b"png-bytes"},
    )
    analyze = next(t for t in tools if t.name == "analyze_image")
    out = json.loads(analyze.invoke({"attachment_key": "site",
                                     "prompt": "Read N_q"}))
    assert out["analysis"] == "chart shows N_q approx 18"
    # Confirm the multimodal message reached the model.
    assert model.seen is not None
    assert any(b.get("type") == "image_url" for b in model.seen[0].content)


# ===========================================================================
# (c) PrompterChatModel
# ===========================================================================

def _make_openai_response(content, tool_calls=None, finish_reason="stop"):
    """A canned OpenAI-shaped response (SimpleNamespace) for the fake client."""
    tc_objs = []
    for tc in tool_calls or []:
        tc_objs.append(SimpleNamespace(
            id=tc["id"],
            type="function",
            function=SimpleNamespace(
                name=tc["name"],
                arguments=tc["arguments"],
            ),
        ))
    message = SimpleNamespace(content=content, tool_calls=tc_objs or None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(
        choices=[choice],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                              total_tokens=15),
    )


class _FakeCompletions:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakePrompter:
    """Fake PrompterAPI: only .client.chat.completions.create + .chat_model."""

    def __init__(self, response, chat_model="fake-gpt"):
        self.chat_model = chat_model
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions(response))
        )

    @property
    def completions(self):
        return self.client.chat.completions


def test_prompter_chat_model_plain_text():
    resp = _make_openai_response("The factor of safety is 3.0.")
    prompter = _FakePrompter(resp)
    model = PrompterChatModel(prompter=prompter)

    result = model.invoke([HumanMessage(content="What is the FoS?")])
    assert isinstance(result, AIMessage)
    assert result.content == "The factor of safety is 3.0."
    assert result.tool_calls == []
    # Model id came from prompter.chat_model.
    assert prompter.completions.last_kwargs["model"] == "fake-gpt"
    assert prompter.completions.last_kwargs["temperature"] == 0.0


def test_prompter_chat_model_parses_tool_calls():
    resp = _make_openai_response(
        content="",
        tool_calls=[{
            "id": "call_abc",
            "name": "call_agent",
            "arguments": json.dumps({
                "agent_name": "bearing_capacity",
                "method": "bearing_capacity_analysis",
                "parameters": {"width": 2.0},
            }),
        }],
        finish_reason="tool_calls",
    )
    prompter = _FakePrompter(resp)
    model = PrompterChatModel(prompter=prompter)

    result = model.invoke([HumanMessage(content="Compute it.")])
    assert isinstance(result, AIMessage)
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    # LangChain canonical tool-call shape: {name, args(dict), id}.
    assert tc["name"] == "call_agent"
    assert isinstance(tc["args"], dict)
    assert tc["args"]["agent_name"] == "bearing_capacity"
    assert tc["args"]["parameters"]["width"] == 2.0
    assert tc["id"] == "call_abc"


def test_prompter_chat_model_finish_reason_mapped():
    resp = _make_openai_response("done", finish_reason="tool_calls")
    prompter = _FakePrompter(resp)
    model = PrompterChatModel(prompter=prompter)
    result = model._generate([HumanMessage(content="hi")])
    info = result.generations[0].generation_info
    assert info["finish_reason"] == "tool_calls"
    assert info["usage"]["total_tokens"] == 15


def test_assistant_tool_calls_serialize_as_plain_dicts_no_null_fields():
    """The proxy rejects null function_call/refusal/audio/annotations. An
    AIMessage with tool_calls must translate to a plain dict with exactly
    role/content/tool_calls and each tool_call = {id, type, function}."""
    ai = AIMessage(
        content="",
        tool_calls=[{
            "name": "call_agent",
            "args": {"agent_name": "settlement"},
            "id": "call_1",
        }],
    )
    out = _lc_message_to_openai(ai)
    assert set(out.keys()) == {"role", "content", "tool_calls"}
    assert out["role"] == "assistant"
    # No null pass-through fields anywhere.
    assert "function_call" not in out
    assert "refusal" not in out
    assert "audio" not in out
    assert "annotations" not in out

    tc = out["tool_calls"][0]
    assert set(tc.keys()) == {"id", "type", "function"}
    assert tc["id"] == "call_1"
    assert tc["type"] == "function"
    assert set(tc["function"].keys()) == {"name", "arguments"}
    assert tc["function"]["name"] == "call_agent"
    # arguments is a JSON STRING (OpenAI contract), not a dict.
    assert tc["function"]["arguments"] == json.dumps({"agent_name": "settlement"})

    # Whole message is JSON-serializable with no None values lurking.
    serialized = json.dumps(out)
    assert "null" not in serialized


def test_message_translation_roles():
    assert _lc_message_to_openai(SystemMessage(content="sys")) == {
        "role": "system", "content": "sys",
    }
    assert _lc_message_to_openai(HumanMessage(content="hi")) == {
        "role": "user", "content": "hi",
    }
    tool_msg = _lc_message_to_openai(
        ToolMessage(content="result", tool_call_id="call_9")
    )
    assert tool_msg == {
        "role": "tool", "tool_call_id": "call_9", "content": "result",
    }


def test_human_multimodal_content_passthrough():
    """Multimodal HumanMessage content (vision) passes through verbatim."""
    blocks = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "text", "text": "describe"},
    ]
    out = _lc_message_to_openai(HumanMessage(content=blocks))
    assert out["role"] == "user"
    assert out["content"] == blocks


def test_bind_tools_stores_openai_schema():
    from funhouse_agent.deep.tools import make_core_tools

    resp = _make_openai_response("ok")
    prompter = _FakePrompter(resp)
    model = PrompterChatModel(prompter=prompter)

    core = make_core_tools()
    bound = model.bind_tools(core)

    # bind_tools returns a copy carrying the converted OpenAI tool schemas.
    assert bound is not model
    assert bound.openai_tools is not None
    assert len(bound.openai_tools) == len(core)
    schema = bound.openai_tools[0]
    assert schema["type"] == "function"
    assert "name" in schema["function"]
    names = {s["function"]["name"] for s in bound.openai_tools}
    assert {"list_agents", "list_methods", "describe_method",
            "call_agent"} <= names

    # On generation the bound tools + tool_choice="auto" are sent.
    bound.invoke([HumanMessage(content="go")])
    sent = prompter.completions.last_kwargs
    assert "tools" in sent
    assert sent["tool_choice"] == "auto"
    assert len(sent["tools"]) == len(core)


def test_model_override_takes_precedence():
    resp = _make_openai_response("ok")
    prompter = _FakePrompter(resp, chat_model="prompter-default")
    model = PrompterChatModel(prompter=prompter, model="explicit-model")
    model.invoke([HumanMessage(content="hi")])
    assert prompter.completions.last_kwargs["model"] == "explicit-model"


def test_llm_type():
    model = PrompterChatModel(prompter=_FakePrompter(_make_openai_response("x")))
    assert model._llm_type == "funhouse-prompter-chat"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
