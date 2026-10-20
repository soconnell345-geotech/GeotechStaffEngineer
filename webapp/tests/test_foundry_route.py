"""Which proxy URL a Foundry model request goes to, and the TLS it checks
(INF-36), plus two Connection-diagnostics lines from the first Foundry run
(INF-39).

In a Code Workspace and a published app FOUNDRY_HOSTNAME is a local data
sidecar that answers every LLM-proxy path with 401; the language-model
service is reached through the container's Foundry proxy. No network here:
routing, env parsing and model construction only. The requests themselves
go to the stand-in proxy in ``test_foundry_model_family.py``.
"""

import importlib.util

import pytest

from webapp import diagnostics, engine_config
from webapp.engine_config import foundry_proxy_route

_HAS_LC_ANTHROPIC = importlib.util.find_spec("langchain_anthropic") is not None
_HAS_LC_OPENAI = importlib.util.find_spec("langchain_openai") is not None

ROUTE_ENVS = ("GEOTECH_FOUNDRY_HOST", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL",
              "FOUNDRY_PROXY_URL", "FOUNDRY_HOSTNAME", "FOUNDRY_URL",
              "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "MAESTRO_CERT_PEM_PATH",
              "GEOTECH_DEPLOYMENT")
PROXY = "proxy.example.test/api/foundry-proxy"
LMS = "https://proxy.example.test/api/foundry-proxy/language-model-service"


@pytest.fixture
def env(monkeypatch):
    for name in ROUTE_ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FOUNDRY_TOKEN", "tok-123")
    return monkeypatch


def test_the_proxy_url_gives_both_doors(env):
    env.setenv("FOUNDRY_PROXY_URL", PROXY)
    env.setenv("FOUNDRY_HOSTNAME", "localhost:8080")       # the sidecar
    a = foundry_proxy_route("anthropic")
    o = foundry_proxy_route("openai")
    assert a.url == f"{LMS}/api/proxy/anthropic"
    assert o.url == f"{LMS}/api/proxy/openai/v1"
    assert a.source == o.source == "FOUNDRY_PROXY_URL"


def test_the_order_of_the_settings(env):
    env.setenv("FOUNDRY_HOSTNAME", "localhost:8080")
    assert foundry_proxy_route("anthropic").url == (
        "https://localhost:8080/api/v2/llm/proxy/anthropic")
    env.setenv("FOUNDRY_PROXY_URL", PROXY)
    assert foundry_proxy_route("anthropic").source == "FOUNDRY_PROXY_URL"
    env.setenv("ANTHROPIC_BASE_URL", "https://given.example.test/x/anthropic/")
    route = foundry_proxy_route("anthropic")
    assert (route.url, route.source) == (
        "https://given.example.test/x/anthropic", "ANTHROPIC_BASE_URL")
    assert route.label == "the base URL the container provides"
    # the OpenAI door has its own provided URL, and is not given Claude's
    assert foundry_proxy_route("openai").source == "FOUNDRY_PROXY_URL"
    # the app's own explicit setting wins over everything
    env.setenv("GEOTECH_FOUNDRY_HOST", "http://127.0.0.1:5000/")
    route = foundry_proxy_route("anthropic")
    assert (route.url, route.source) == (
        "http://127.0.0.1:5000/api/v2/llm/proxy/anthropic",
        "GEOTECH_FOUNDRY_HOST")


def test_no_route_says_where_to_set_one(env):
    assert foundry_proxy_route("anthropic") is None
    res = engine_config.resolve_engine(
        model_id="anthropic:ri.language-model-service..language-model.lm-1")
    assert res.source == "error"
    assert "FOUNDRY_PROXY_URL" in res.message
    assert "ANTHROPIC" not in res.message


def test_the_described_route_carries_no_credentials():
    route = engine_config.FoundryRoute(
        "https://user:secret@proxy.example.test:8443/p/anthropic?token=t",
        "FOUNDRY_PROXY_URL")
    assert route.describe() == (
        "FOUNDRY_PROXY_URL: https://proxy.example.test:8443/p/anthropic")


@pytest.mark.skipif(not _HAS_LC_ANTHROPIC, reason="langchain_anthropic")
def test_claude_on_the_proxy_url(env):
    env.setenv("FOUNDRY_PROXY_URL", PROXY)
    res = engine_config.resolve_engine(
        model_id="anthropic:ri.language-model-service..language-model.lm-1")
    assert res.ok, res.message
    assert str(res.model.anthropic_api_url) == f"{LMS}/api/proxy/anthropic"
    assert "route: FOUNDRY_PROXY_URL" in res.message
    assert res.route == f"FOUNDRY_PROXY_URL: {LMS}/api/proxy/anthropic"


def test_a_ca_bundle_is_the_first_named_file_that_exists(env, tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("x")
    env.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    env.setenv("MAESTRO_CERT_PEM_PATH", str(ca))
    assert engine_config.foundry_ca_bundle() == str(ca)
    env.delenv("MAESTRO_CERT_PEM_PATH")
    assert engine_config.foundry_ca_bundle() is None


@pytest.mark.skipif(not _HAS_LC_ANTHROPIC, reason="langchain_anthropic")
def test_the_clients_check_tls_against_the_named_bundle(env):
    certifi = pytest.importorskip("certifi")
    env.setenv("FOUNDRY_PROXY_URL", PROXY)
    env.setenv("REQUESTS_CA_BUNDLE", certifi.where())
    sync, asyn = engine_config.foundry_http_clients(certifi.where())
    res = engine_config.resolve_engine(
        model_id="anthropic:ri.language-model-service..language-model.lm-1")
    assert "CA bundle set" in res.message
    assert res.model._client._client is sync
    assert res.model._async_client._client is asyn
    if _HAS_LC_OPENAI:
        res = engine_config.resolve_engine(
            model_id="openai:ri.language-model-service..language-model.gpt-5-2")
        assert res.model.http_client is sync
        assert res.model.http_async_client is asyn


@pytest.mark.skipif(not _HAS_LC_ANTHROPIC, reason="langchain_anthropic")
def test_without_a_bundle_the_sdk_client_is_unchanged(env):
    env.setenv("FOUNDRY_PROXY_URL", PROXY)
    res = engine_config.resolve_engine(
        model_id="anthropic:ri.language-model-service..language-model.lm-1")
    assert "CA bundle" not in res.message
    assert str(res.model._client.base_url).startswith(
        f"{LMS}/api/proxy/anthropic")


class _Response:
    def __init__(self, status):
        self.status_code = status


@pytest.mark.parametrize("foundry", [True, False])
def test_the_upload_probe_on_foundry_is_not_an_alarm(monkeypatch, foundry):
    """PUT 403 / POST 405 on localhost: on Foundry that is expected and
    uploads work in the published app (owner, 2026-10-20); elsewhere it
    still warns."""
    import requests
    codes = {"PUT": 403, "POST": 405}
    monkeypatch.setattr(requests, "request",
                        lambda method, *a, **k: _Response(codes[method]))
    if foundry:
        monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    else:
        monkeypatch.delenv("GEOTECH_DEPLOYMENT", raising=False)
    check = diagnostics._upload_probe_check()
    assert "PUT -> 403; POST -> 405" in check["detail"]
    if foundry:
        assert check["status"] == diagnostics.PASS
        assert "works in the published app" in check["detail"]
    else:
        assert check["status"] == diagnostics.WARN
