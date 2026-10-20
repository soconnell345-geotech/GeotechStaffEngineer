"""Which model each helper runs on, as the app decides it (BACKLOG INF-41).
No network: settings, label lookup and model construction only. Requests
through the stand-in proxy are in ``test_foundry_model_family.py``."""

import importlib.util

import pytest

from webapp import core, helper_models

needs_anthropic = pytest.mark.skipif(
    importlib.util.find_spec("langchain_anthropic") is None,
    reason="langchain_anthropic")

OPUS = "anthropic:ri.language-model-service..language-model.lm-1a2b3c"
SONNET = "anthropic:ri.language-model-service..language-model.lm-4d5e6f"
ENVS = ("GEOTECH_DEPLOYMENT", "GEOTECH_HELPER_MODELS", "GEOTECH_FOUNDRY_MODELS",
        "GEOTECH_WEBAPP_MODEL", "GEOTECH_FOUNDRY_HOST", "ANTHROPIC_BASE_URL",
        "OPENAI_BASE_URL", "FOUNDRY_HOSTNAME", "FOUNDRY_URL", "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE", "MAESTRO_CERT_PEM_PATH")


@pytest.fixture
def foundry(monkeypatch):
    for name in ENVS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    monkeypatch.setenv("FOUNDRY_TOKEN", "tok-123")
    monkeypatch.setenv("FOUNDRY_PROXY_URL", "proxy.example.test/foundry-proxy")
    monkeypatch.setenv(core.FOUNDRY_MODELS_ENV,
                       f"Opus 5.5={OPUS},Sonnet 5.5={SONNET}")
    return monkeypatch


def _by_name(helpers):
    return {h.name: h for h in helpers}


def test_parse():
    assert helper_models.parse(" references=Sonnet 5.5 , calc=main,x") == {
        "references": "Sonnet 5.5", "calc": "main"}
    assert helper_models.parse(None) == {}


@pytest.mark.parametrize("deployment", [None, "tinyapps"])
def test_other_hosts_have_no_helper_models(monkeypatch, deployment):
    for name in ENVS:
        monkeypatch.delenv(name, raising=False)
    if deployment:
        monkeypatch.setenv("GEOTECH_DEPLOYMENT", deployment)
    assert helper_models.settings() == {}
    assert helper_models.resolve("claude-opus-4-8") == []
    assert helper_models.build_kwargs([]) == {}


@needs_anthropic
def test_on_foundry_the_references_helper_is_sonnet(foundry):
    assert core.default_model_id() == OPUS
    helpers = helper_models.resolve(OPUS)
    ref = _by_name(helpers)["references"]
    assert ref.own_model and (ref.model_id, ref.label) == (SONNET, "Sonnet 5.5")
    assert ref.source == helper_models.SOURCE_HOST
    assert type(ref.model).__name__ == "FoundryChatAnthropic"
    assert ref.model.model == SONNET.split(":", 1)[1]
    assert str(ref.model.anthropic_api_url).endswith(
        "/language-model-service/api/proxy/anthropic")
    assert ref.route.startswith("FOUNDRY_PROXY_URL: ")
    assert helper_models.build_kwargs(helpers) == {
        "subagent_models": {"references": ref.model}}
    line = helper_models.summary("Opus 5.5", helpers)
    assert line == (f"main agent: Opus 5.5; references: Sonnet 5.5 ({SONNET}; "
                    "this host's default); every other helper: the main "
                    "agent's model")


@needs_anthropic
def test_the_env_and_the_sidebar_override_the_default(foundry):
    foundry.setenv(helper_models.ENV, "references=main, calc=Sonnet 5.5")
    helpers = _by_name(helper_models.resolve(OPUS))
    assert not helpers["references"].own_model
    assert helpers["references"].source == helper_models.ENV
    assert helpers["calc"].own_model and helpers["calc"].model_id == SONNET
    helpers = _by_name(helper_models.resolve(
        OPUS, {"references": SONNET, "calc": "main"}))
    assert helpers["references"].own_model
    assert helpers["references"].source == helper_models.SOURCE_SIDEBAR
    assert not helpers["calc"].own_model


@needs_anthropic
def test_a_model_not_in_the_list_is_taken_as_an_id(foundry):
    other = "anthropic:ri.language-model-service..language-model.lm-777"
    foundry.setenv(helper_models.ENV, f"references={other}")
    ref = _by_name(helper_models.resolve(OPUS))["references"]
    assert ref.own_model and ref.model_id == other and ref.label == other


def test_the_same_model_as_the_main_agent_needs_no_client(foundry):
    ref = _by_name(helper_models.resolve(SONNET))["references"]
    assert not ref.own_model and ref.label == "Sonnet 5.5"
    assert "Sonnet 5.5, the main agent's model" in ref.describe()


def test_a_default_missing_from_the_list_says_so(foundry):
    foundry.setenv(core.FOUNDRY_MODELS_ENV, f"Opus 5.5={OPUS}")
    ref = _by_name(helper_models.resolve(OPUS))["references"]
    assert not ref.own_model
    assert ref.note == ("'Sonnet 5.5' is not in the model list "
                        "(GEOTECH_FOUNDRY_MODELS)")
    assert "ANTHROPIC" not in ref.describe()


def test_a_helper_model_that_does_not_resolve_falls_back(foundry):
    foundry.delenv("FOUNDRY_TOKEN")
    ref = _by_name(helper_models.resolve(OPUS))["references"]
    assert not ref.own_model and ref.note.startswith(
        "its model did not resolve")
