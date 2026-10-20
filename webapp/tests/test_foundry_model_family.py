"""The model family on Foundry: which proxy a Claude model reaches (INF-25).

The owner runs the app on Foundry with Claude Opus 5.5 / Sonnet 5.5 RIDs
typed into the sidebar or set at startup. The proxy route used to be chosen
by the word ``anthropic`` in the RID, which a Foundry Claude RID need not
carry. A model id may now state its family as a prefix (``anthropic:`` /
``openai:``), the sidebar sets it beside the RID box, and with no prefix the
family is read off the id (Claude, Opus, Sonnet, Haiku, Fable, Anthropic).

The last tests send real requests to a fake Anthropic-messages proxy on
localhost: no Foundry, no network beyond 127.0.0.1.
"""

import importlib.util
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from webapp import core, engine_config
from webapp.engine_config import (model_family, split_model_family,
                                  with_model_family)

# A Claude RID that does not name its maker: the case the old rule missed.
PLAIN_RID = "ri.language-model-service..language-model.lm-7f3a9c"
CLAUDE_RID = "ri.language-model-service..language-model.claude-opus-5-5"
GPT_RID = "ri.language-model-service..language-model.gpt-5-2"

_HAS_LC_ANTHROPIC = importlib.util.find_spec("langchain_anthropic") is not None
_HAS_LC_OPENAI = importlib.util.find_spec("langchain_openai") is not None
needs_anthropic = pytest.mark.skipif(not _HAS_LC_ANTHROPIC,
                                     reason="langchain_anthropic not installed")

_CLEAR = ("ANTHROPIC_API_KEY", "GEOTECH_FOUNDRY_MODELS", "GEOTECH_WEBAPP_MODEL",
          "GEOTECH_FOUNDRY_TOKEN", "FOUNDRY_TOKEN", "GEOTECH_FOUNDRY_HOST",
          "FOUNDRY_HOSTNAME", "FOUNDRY_URL", "GEOTECH_DEPLOYMENT",
          "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "FOUNDRY_PROXY_URL",
          "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "MAESTRO_CERT_PEM_PATH",
          "GEOTECH_FOUNDRY_PROMPT_CACHE")


@pytest.fixture
def foundry(monkeypatch, tmp_path):
    """A Foundry deployment with a token and a host set."""
    for env in _CLEAR:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.setenv("FOUNDRY_TOKEN", "tok-123")
    monkeypatch.setenv("FOUNDRY_HOSTNAME", "stack.example.palantirfoundry.com")
    return monkeypatch


class TestFamilyParsing:
    def test_split(self):
        assert split_model_family(f"anthropic:{PLAIN_RID}") == (
            "anthropic", PLAIN_RID)
        assert split_model_family(f" OpenAI : {GPT_RID} ") == (
            "openai", GPT_RID)
        assert split_model_family(PLAIN_RID) == ("auto", PLAIN_RID)
        assert split_model_family(None) == ("auto", "")
        # A colon that is not a family prefix is part of the id.
        bedrock = "us.anthropic.claude-opus-5-5-v1:0"
        assert split_model_family(bedrock) == ("auto", bedrock)
        assert split_model_family("anthropic:") == ("auto", "anthropic:")

    def test_with_family(self):
        assert with_model_family(PLAIN_RID, "anthropic") == (
            f"anthropic:{PLAIN_RID}")
        assert with_model_family(f"anthropic:{PLAIN_RID}", "openai") == (
            f"openai:{PLAIN_RID}")
        assert with_model_family(PLAIN_RID, "auto") == PLAIN_RID
        # Auto keeps a prefix typed into the id itself.
        assert with_model_family(f"anthropic:{PLAIN_RID}", "auto") == (
            f"anthropic:{PLAIN_RID}")
        assert with_model_family("", "anthropic") == ""

    @pytest.mark.parametrize("mid", [
        CLAUDE_RID, "ri.x..language-model.anthropic-claude-4-6-opus",
        "Claude_Opus_5_5", "SONNET_5_5", "claude-haiku-4-5",
        f"anthropic:{PLAIN_RID}", f"anthropic:{GPT_RID}"])
    def test_anthropic(self, mid):
        assert model_family(mid) == "anthropic"

    @pytest.mark.parametrize("mid", [
        GPT_RID, PLAIN_RID, "GPT_5_1", f"openai:{CLAUDE_RID}"])
    def test_openai(self, mid):
        assert model_family(mid) == "openai"

    @pytest.mark.parametrize("mid", [
        # A Claude word inside a random chunk of the model's name...
        "ri.language-model-service..language-model.x7opus3k",
        "ri.language-model-service..language-model.magnumopus-2",
        "ri.language-model-service..language-model.gpt-5-fabled",
        # ...or outside the model's name (a RID's other fields).
        "ri.language-model-service.sonnet-stack.language-model.gpt-5-2"])
    def test_auto_reads_whole_words_of_the_name(self, mid):
        assert model_family(mid) == "openai"

    @pytest.mark.parametrize("mid", [
        "ri.language-model-service..language-model.opus4",
        "claude3-haiku", "ri.x.anthropic..language-model.lm-1"])
    def test_auto_word_edges(self, mid):
        assert model_family(mid) == "anthropic"

    def test_rid_detected_with_a_family(self):
        assert engine_config.is_foundry_model_id(f"anthropic:{PLAIN_RID}")
        assert not engine_config.is_foundry_model_id("anthropic:Claude_5")


@needs_anthropic
class TestRouting:
    def test_stated_family_sends_plain_rid_to_anthropic_proxy(self, foundry):
        res = engine_config.resolve_engine(model_id=f"anthropic:{PLAIN_RID}")
        assert res.ok and res.source == "foundry", res.message
        assert type(res.model).__name__ == "FoundryChatAnthropic"
        assert res.model.model == PLAIN_RID          # the prefix is not sent
        assert str(res.model.anthropic_api_url) == (
            "https://stack.example.palantirfoundry.com"
            "/api/v2/llm/proxy/anthropic")
        assert "Anthropic proxy" in res.message
        assert "family chosen" in res.message

    def test_claude_rid_auto(self, foundry):
        res = engine_config.resolve_engine(model_id=CLAUDE_RID)
        assert type(res.model).__name__ == "FoundryChatAnthropic", res.message
        assert "family read off the id" in res.message

    def test_plain_rid_auto_stays_openai(self, foundry):
        res = engine_config.resolve_engine(model_id=PLAIN_RID)
        if _HAS_LC_OPENAI:
            assert type(res.model).__name__ == "ChatOpenAI", res.message
        else:
            assert res.source == "error"

    def test_stated_openai_overrides_the_id(self, foundry):
        res = engine_config.resolve_engine(model_id=f"openai:{CLAUDE_RID}")
        if _HAS_LC_OPENAI:
            assert type(res.model).__name__ == "ChatOpenAI", res.message
            assert res.model.model_name == CLAUDE_RID

    @pytest.mark.parametrize("mid", ["Claude_Opus_5_5",
                                     "anthropic:Opus55Name"])
    def test_claude_api_name_goes_to_proxy_not_sdk(self, foundry, mid):
        """The palantir_models SDK engine speaks to GPT models only."""
        res = engine_config.resolve_engine(model_id=mid)
        assert res.source == "foundry", res.message
        assert type(res.model).__name__ == "FoundryChatAnthropic"
        assert res.model.model == split_model_family(mid)[1]

    @pytest.mark.parametrize("mid", ["GPT_5_1", "openai:GPT_5_1"])
    def test_gpt_api_name_still_uses_sdk_route(self, foundry, mid):
        res = engine_config.resolve_engine(model_id=mid)
        assert res.source in ("foundry_sdk", "error"), res.message
        assert "GPT_5_1" in res.message and "openai:" not in res.message

    def test_startup_env_carries_the_family(self, foundry):
        foundry.setenv(core.FOUNDRY_MODELS_ENV,
                       f"Opus 5.5=anthropic:{PLAIN_RID},GPT 5.2={GPT_RID}")
        assert core.default_model_id() == f"anthropic:{PLAIN_RID}"
        assert core.model_label(f"anthropic:{PLAIN_RID}") == "Opus 5.5"
        res = engine_config.resolve_engine(model_id=core.default_model_id())
        assert type(res.model).__name__ == "FoundryChatAnthropic", res.message

    def test_claude_api_name_without_token_says_so(self, foundry):
        foundry.delenv("FOUNDRY_TOKEN")
        res = engine_config.resolve_engine(model_id="Claude_Opus_5_5")
        assert res.source == "error"
        assert "Claude model" in res.message
        assert "looks like a Foundry RID" not in res.message
        assert "ANTHROPIC_API_KEY" not in res.message

    def test_builder_gets_the_bare_id(self, monkeypatch):
        for env in _CLEAR:
            monkeypatch.delenv(env, raising=False)
        got = []

        def builder(model_id):
            got.append(model_id)
            return object()

        monkeypatch.setattr(engine_config, "_MODEL_BUILDER", builder)
        res = engine_config.resolve_engine(
            model_id="anthropic:funhouse-gpt-high")
        assert got == ["funhouse-gpt-high"]
        assert res.model_name == "funhouse-gpt-high"

    def test_key_path_strips_a_family(self, monkeypatch):
        for env in _CLEAR:
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        res = engine_config.resolve_engine(
            model_id="anthropic:claude-opus-4-8")
        assert res.source == "anthropic", res.message
        assert res.model.model == "claude-opus-4-8"


# ---------------------------------------------------------------------------
# A fake Anthropic-messages proxy on localhost
# ---------------------------------------------------------------------------

def _sse(events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n"
                   for e in events).encode()


#: The proxy's two doors: the stack's own path and, inside a Code Workspace
#: or a published app, the language-model service behind the container's
#: Foundry proxy (INF-36).
_MESSAGES_PATHS = ("/api/v2/llm/proxy/anthropic/v1/messages",
                   "/language-model-service/api/proxy/anthropic/v1/messages")


def _refusal(body):
    """What the Foundry Anthropic proxy answers 400 to, measured with both
    Claude 5.5 RIDs on 2026-10-20 (INF-36): the field it names, else None."""
    if "cache_control" in body or any(
            isinstance(t, dict) and "cache_control" in t
            for t in body.get("tools") or []):
        return "{unrecognizedProperty=cache_control}"
    if "service_tier" in body:
        return "{unrecognizedProperty=service_tier}"
    if (body.get("thinking") or {}).get("type") == "enabled":
        return ('"thinking.type.enabled" is not supported for this model. '
                'Use "thinking.type.adaptive"')
    if (body.get("tool_choice") or {}).get("type") in ("any", "tool"):
        return ('tool_choice: type "tool" and "any" are not supported for '
                "this model.")
    return None


class _FakeProxy(BaseHTTPRequestHandler):
    """A stand-in Foundry Anthropic proxy: answers like the real one,
    refusals included."""

    seen: list = []

    def log_message(self, *args):                       # quiet
        pass

    def _reply(self, status, payload, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):                                  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        body = json.loads(raw or b"{}")
        images = [b["source"] for m in body.get("messages") or []
                  if isinstance(m.get("content"), list)
                  for b in m["content"] if b.get("type") == "image"]
        system = body.get("system")
        refused = _refusal(body)
        type(self).seen.append({"path": self.path,
                                "auth": self.headers.get("Authorization"),
                                "model": body.get("model"),
                                "max_tokens": body.get("max_tokens"),
                                "images": images,
                                "top_cache": "cache_control" in body,
                                "cache_marks": raw.count(b'"cache_control"'),
                                "system_cache": isinstance(system, list) and any(
                                    "cache_control" in b for b in system),
                                "tool_cache": any(
                                    "cache_control" in t
                                    for t in body.get("tools") or []),
                                "thinking": body.get("thinking"),
                                "stream": bool(body.get("stream")),
                                "refused": refused,
                                "tools": [t.get("name")
                                          for t in body.get("tools") or []]})
        if not self.path.endswith(_MESSAGES_PATHS):
            self._reply(404, b'{"errorCode":"NOT_FOUND"}')
            return
        if refused:
            self._reply(400, json.dumps({
                "errorCode": "INVALID_ARGUMENT",
                "parameters": {"unsafeParams": refused}}).encode())
            return
        last = json.dumps(body.get("messages", [])[-1:])
        if body.get("tools") and "Call the tool" in last:
            block = {"type": "tool_use", "id": "toolu_1",
                     "name": body["tools"][0]["name"],
                     "input": {"a": 2, "b": 3}}
            stop = "tool_use"
        else:
            block = {"type": "text", "text": "OK from the fake proxy"}
            stop = "end_turn"
        msg = {"id": "msg_1", "type": "message", "role": "assistant",
               "model": body.get("model"), "content": [block],
               "stop_reason": stop, "stop_sequence": None,
               "usage": {"input_tokens": 5, "output_tokens": 3}}
        if not body.get("stream"):
            self._reply(200, json.dumps(msg).encode())
            return
        start = dict(msg, content=[], stop_reason=None,
                     usage={"input_tokens": 5, "output_tokens": 0})
        if block["type"] == "tool_use":
            opened = dict(block, input={})
            delta = {"type": "input_json_delta",
                     "partial_json": json.dumps(block["input"])}
        else:
            opened = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        events = [{"type": "message_start", "message": start},
                  {"type": "content_block_start", "index": 0,
                   "content_block": opened},
                  {"type": "content_block_delta", "index": 0, "delta": delta},
                  {"type": "content_block_stop", "index": 0},
                  {"type": "message_delta",
                   "delta": {"stop_reason": stop, "stop_sequence": None},
                   "usage": {"output_tokens": 3}},
                  {"type": "message_stop"}]
        self._reply(200, _sse(events), "text/event-stream")


@pytest.fixture
def fake_proxy(foundry):
    _FakeProxy.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeProxy)
    _FakeProxy.port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    foundry.setenv("FOUNDRY_HOSTNAME",
                   f"http://127.0.0.1:{server.server_address[1]}")
    foundry.setenv("NO_PROXY", "127.0.0.1,localhost")
    yield _FakeProxy.seen
    server.shutdown()
    server.server_close()


@needs_anthropic
def test_diagnostics_reach_the_anthropic_proxy(fake_proxy):
    """The sidebar's Connection diagnostics, against the fake proxy: every
    request goes to the Anthropic-messages path with the token as a bearer
    and the bare RID as the model."""
    from webapp import diagnostics
    checks = diagnostics.run_diagnostics(f"anthropic:{PLAIN_RID}")
    by_name = {c["name"]: c for c in checks}
    for name in ("engine resolution", "plain request (invoke)",
                 "streaming request", "tool-calling request"):
        assert by_name[name]["status"] == "pass", (name, by_name[name])
    assert fake_proxy, "no request reached the fake proxy"
    assert {r["path"] for r in fake_proxy} == {
        "/api/v2/llm/proxy/anthropic/v1/messages"}
    assert {r["model"] for r in fake_proxy} == {PLAIN_RID}
    assert {r["auth"] for r in fake_proxy} == {"Bearer tok-123"}
    assert any(r["stream"] for r in fake_proxy)
    assert any(r["tools"] == ["add_numbers"] for r in fake_proxy)


@needs_anthropic
def test_sidebar_family_choice_reaches_the_proxy(fake_proxy):
    """Type a plain RID, choose Anthropic beside it, Use this model: the
    session's model carries the family and the engine is the Anthropic
    proxy."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    app = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "app.py")
    at = AppTest.from_file(app, default_timeout=60).run()
    assert not at.exception
    rid = [t for t in at.text_input
           if str(t.key or "").startswith("custom_model_")][0]
    fam = [s for s in at.selectbox
           if str(s.key or "").startswith("custom_family_")][0]
    use = [b for b in at.button
           if str(b.key or "").startswith("use_custom_")][0]
    rid.set_value(PLAIN_RID)
    fam.set_value("anthropic")
    use.set_value(True)
    at.run()
    assert not at.exception
    assert at.session_state["model"] == f"anthropic:{PLAIN_RID}"
    assert at.session_state["engine"].source == "foundry"
    assert "Anthropic proxy" in at.session_state["engine"].message
    texts = " | ".join(str(e.value) for e in at.success)
    assert "Anthropic proxy" in texts, texts


@needs_anthropic
def test_vision_side_call_reaches_the_proxy_as_an_anthropic_image(
        fake_proxy, monkeypatch):
    """A page image the vision tools send (``analyze_image`` and friends,
    an OpenAI-shaped ``image_url`` data URI) reaches the Anthropic proxy as
    an Anthropic base64 image block, with the bare RID and a max_tokens."""
    import base64
    from funhouse_agent.deep.inline_images import _probe_png
    from funhouse_agent.deep.vision_engine import LangChainVisionEngine
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")   # the one call only
    res = engine_config.resolve_engine(model_id=f"anthropic:{PLAIN_RID}")
    png = _probe_png()
    text = LangChainVisionEngine(res.model).analyze_image(
        png, "What colour is the square?", max_output_tokens=2000)
    assert text == "OK from the fake proxy"
    req = fake_proxy[-1]
    assert req["model"] == PLAIN_RID and req["auth"] == "Bearer tok-123"
    assert req["max_tokens"] == 2000
    assert req["images"] == [{"type": "base64", "media_type": "image/png",
                              "data": base64.b64encode(png).decode()}]


@needs_anthropic
@pytest.mark.parametrize("setting, marks", [(None, True), ("0", False)])
def test_prompt_cache_mark_is_never_top_level(fake_proxy, monkeypatch,
                                              setting, marks):
    """The agent's prompt-caching middleware asks for a top-level
    ``cache_control``, which only Anthropic's own API accepts. On the
    Foundry proxy it goes on a message block instead, or nowhere with
    GEOTECH_FOUNDRY_PROMPT_CACHE=0."""
    if setting is not None:
        monkeypatch.setenv(engine_config.FOUNDRY_PROMPT_CACHE_ENV, setting)
    res = engine_config.resolve_engine(model_id=f"anthropic:{PLAIN_RID}")
    reply = res.model.invoke("Say OK",
                             cache_control={"type": "ephemeral", "ttl": "5m"})
    assert "OK" in reply.content
    req = fake_proxy[-1]
    assert req["top_cache"] is False
    assert (req["cache_marks"] > 0) is marks


@needs_anthropic
def test_agent_middleware_marks_reach_the_proxy_on_blocks(fake_proxy):
    """The deep agent's own caching middleware, on a real model request:
    system and message marks, none at the top level or on a tool (INF-36:
    the proxy refuses both)."""
    from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    res = engine_config.resolve_engine(model_id=f"anthropic:{PLAIN_RID}")
    agent = create_agent(res.model, tools=[add_numbers],
                         system_prompt="You are terse.",
                         middleware=[AnthropicPromptCachingMiddleware()])
    agent.invoke({"messages": [{"role": "user", "content": "Say OK"}]})
    req = fake_proxy[-1]
    assert req["refused"] is None
    assert req["top_cache"] is False and req["tool_cache"] is False
    assert req["system_cache"]
    assert req["cache_marks"] >= 2


# ---------------------------------------------------------------------------
# INF-36: the route the container gives, and the fields the proxy refuses
# ---------------------------------------------------------------------------

@pytest.fixture
def proxy_route(fake_proxy, foundry):
    """The stand-in reached the way a Code Workspace reaches the proxy:
    FOUNDRY_PROXY_URL (here with http://, so no scheme is added), with
    FOUNDRY_HOSTNAME pointing at a sidecar that answers nothing."""
    foundry.setenv("FOUNDRY_HOSTNAME", "http://127.0.0.1:9")
    foundry.setenv("FOUNDRY_PROXY_URL",
                   f"http://127.0.0.1:{_FakeProxy.port}/fp")
    return fake_proxy


@needs_anthropic
def test_the_proxy_url_route_passes_the_diagnostics(proxy_route):
    from webapp import diagnostics
    checks = diagnostics.run_diagnostics(f"anthropic:{PLAIN_RID}")
    by_name = {c["name"]: c for c in checks}
    for name in ("engine resolution", "plain request (invoke)",
                 "streaming request", "tool-calling request"):
        assert by_name[name]["status"] == "pass", (name, by_name[name])
    assert {r["path"] for r in proxy_route} == {
        "/fp/language-model-service/api/proxy/anthropic/v1/messages"}
    detail = by_name["engine resolution"]["detail"]
    assert "route: FOUNDRY_PROXY_URL" in detail
    assert "FOUNDRY_PROXY_URL: http://127.0.0.1:" in detail
    assert "tok-123" not in detail


@needs_anthropic
def test_a_provided_base_url_wins_over_the_proxy_url(proxy_route, foundry):
    foundry.setenv("FOUNDRY_PROXY_URL", "http://127.0.0.1:9/nothing")
    foundry.setenv("ANTHROPIC_BASE_URL",
                   f"http://127.0.0.1:{_FakeProxy.port}/fp/"
                   "language-model-service/api/proxy/anthropic/")
    res = engine_config.resolve_engine(model_id=f"anthropic:{PLAIN_RID}")
    assert "the base URL the container provides" in res.message
    assert "ANTHROPIC" not in res.message     # Foundry names no provider key
    assert res.model.invoke("Say OK").content == "OK from the fake proxy"
    assert proxy_route[-1]["path"].startswith("/fp/language-model-service/")


@needs_anthropic
def test_the_stand_in_refuses_what_the_proxy_refuses(proxy_route):
    """A plain ChatAnthropic sends service_tier as asked, and the stand-in
    answers 400 naming the field, as the proxy does."""
    import anthropic
    from langchain_anthropic import ChatAnthropic
    route = engine_config.foundry_proxy_route("anthropic")
    plain = ChatAnthropic(model=PLAIN_RID, anthropic_api_url=route.url,
                          anthropic_api_key="tok-123", max_tokens=50,
                          model_kwargs={"service_tier": "auto"})
    with pytest.raises(anthropic.BadRequestError, match="service_tier"):
        plain.invoke("Say OK")


@needs_anthropic
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("asked", [
    {"model_kwargs": {"service_tier": "auto"}},
    {"thinking": {"type": "enabled", "budget_tokens": 1024}},
    {"call": {"cache_control": {"type": "ephemeral"}}},
], ids=["service_tier", "thinking_enabled", "top_level_cache"])
def test_refused_fields_never_reach_the_proxy(proxy_route, asked, stream):
    route = engine_config.foundry_proxy_route("anthropic")
    cls = engine_config.foundry_chat_anthropic_class()
    kw = {k: v for k, v in asked.items() if k != "call"}
    model = cls(model=PLAIN_RID, anthropic_api_url=route.url,
                anthropic_api_key="tok-123", max_tokens=4096,
                default_headers={"Authorization": "Bearer tok-123"}, **kw)
    call = asked.get("call", {})
    if stream:
        text = "".join(str(c.content) for c in model.stream("Say OK", **call))
    else:
        text = model.invoke("Say OK", **call).content
    assert "OK from the fake proxy" in str(text)
    req = proxy_route[-1]
    assert req["refused"] is None and req["stream"] is stream
    if "thinking" in kw:
        assert req["thinking"] == {"type": "adaptive"}


@needs_anthropic
def test_a_streamed_tool_round_trip_with_the_caching_middleware(proxy_route):
    """The agent's shape on Foundry: caching middleware (system, last tool
    and top-level marks asked for), a tool call and its answer, streamed."""
    from langchain.agents import create_agent
    from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
    from langchain_core.tools import tool

    ran = []

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        ran.append((a, b))
        return a + b

    res = engine_config.resolve_engine(model_id=f"anthropic:{PLAIN_RID}")
    agent = create_agent(res.model, tools=[add_numbers],
                         system_prompt="You are terse.",
                         middleware=[AnthropicPromptCachingMiddleware()])
    chunks = list(agent.stream(
        {"messages": [{"role": "user", "content": "Call the tool, 2 and 3."}]},
        stream_mode="messages"))
    assert chunks and ran == [(2, 3)]
    assert len(proxy_route) == 2
    for req in proxy_route:
        assert req["refused"] is None and req["stream"]
        assert not req["tool_cache"] and not req["top_cache"]
        assert req["system_cache"] and req["tools"] == ["add_numbers"]


@needs_anthropic
@pytest.mark.parametrize("choice", ["any", "add_numbers"])
def test_a_forced_tool_is_stopped_before_the_proxy(proxy_route, choice):
    from langchain_core.tools import tool

    @tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    res = engine_config.resolve_engine(model_id=f"anthropic:{PLAIN_RID}")
    bound = res.model.bind_tools([add_numbers], tool_choice=choice)
    with pytest.raises(engine_config.FoundryProxyLimit, match="tool_choice"):
        bound.invoke("Call the tool, 2 and 3.")
    assert proxy_route == []
    # auto goes through as before
    res.model.bind_tools([add_numbers], tool_choice="auto").invoke("Say OK")
    assert proxy_route[-1]["refused"] is None


def test_the_sanitizer_on_a_payload():
    payload = {
        "model": "m", "cache_control": {"type": "ephemeral", "ttl": "5m"},
        "system": "You are terse.", "service_tier": "auto",
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "tools": [{"name": "a"}, {"name": "b",
                                  "cache_control": {"type": "ephemeral"}}],
        "tool_choice": {"type": "auto"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "hi",
             "cache_control": {"type": "ephemeral"}}]}]}
    changed = engine_config.sanitize_foundry_anthropic_payload(payload)
    assert "cache_control" not in payload and "service_tier" not in payload
    assert payload["system"] == [{"type": "text", "text": "You are terse.",
                                  "cache_control": {"type": "ephemeral",
                                                    "ttl": "5m"}}]
    assert payload["tools"] == [{"name": "a"}, {"name": "b"}]
    assert payload["thinking"] == {"type": "adaptive"}
    # a message block's mark is accepted by the proxy and stays
    assert payload["messages"][0]["content"][0]["cache_control"]
    assert len(changed) == 4
    # an existing system mark is not doubled
    payload = {"cache_control": {"type": "ephemeral"},
               "system": [{"type": "text", "text": "a",
                           "cache_control": {"type": "ephemeral"}},
                          {"type": "text", "text": "b"}]}
    engine_config.sanitize_foundry_anthropic_payload(payload)
    assert "cache_control" not in payload["system"][1]
    for choice in ({"type": "any"}, {"type": "tool", "name": "a"}):
        with pytest.raises(engine_config.FoundryProxyLimit):
            engine_config.sanitize_foundry_anthropic_payload(
                {"tool_choice": choice})
