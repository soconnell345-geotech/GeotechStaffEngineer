"""Offline tests for the Foundry LLM-proxy engine path (docs/FOUNDRY.md).

No network: these only exercise routing, env parsing, and model construction.
"""

import importlib.util

import pytest

from webapp import core, engine_config

GPT_RID = "ri.language-model-service..language-model.gpt-5-2"
CLAUDE_RID = "ri.language-model-service..language-model.anthropic-claude-4-6-opus"

_HAS_LC_OPENAI = importlib.util.find_spec("langchain_openai") is not None


@pytest.fixture
def foundry_env(monkeypatch):
    monkeypatch.setenv("GEOTECH_FOUNDRY_TOKEN", "tok-123")
    monkeypatch.setenv("GEOTECH_FOUNDRY_HOST", "stack.example.palantirfoundry.com")
    monkeypatch.delenv("GEOTECH_WEBAPP_MODEL", raising=False)


class TestRouting:
    def test_rid_detected(self):
        assert engine_config.is_foundry_model_id(GPT_RID)
        assert not engine_config.is_foundry_model_id("claude-opus-4-8")
        assert not engine_config.is_foundry_model_id(None)

    def test_rid_without_creds_is_clear_error(self, monkeypatch):
        for env in (engine_config.FOUNDRY_TOKEN_ENVS
                    + engine_config.FOUNDRY_HOST_ENVS):
            monkeypatch.delenv(env, raising=False)
        res = engine_config.resolve_engine(model_id=GPT_RID)
        assert res.source == "error"
        assert "FOUNDRY" in res.message.upper()

    def test_claude_rid_builds_chat_anthropic_on_proxy(self, foundry_env):
        res = engine_config.resolve_engine(model_id=CLAUDE_RID)
        assert res.ok and res.source == "foundry", res.message
        assert type(res.model).__name__ == "ChatAnthropic"
        base = str(getattr(res.model, "anthropic_api_url", ""))
        assert base == ("https://stack.example.palantirfoundry.com"
                        "/api/v2/llm/proxy/anthropic")

    def test_gpt_rid_routes_to_openai_proxy(self, foundry_env):
        res = engine_config.resolve_engine(model_id=GPT_RID)
        if _HAS_LC_OPENAI:
            assert res.ok and res.source == "foundry", res.message
            assert type(res.model).__name__ == "ChatOpenAI"
        else:   # helpful install message instead of a crash
            assert res.source == "error"
            assert "langchain-openai" in res.message

    def test_non_rid_still_uses_anthropic_key_path(self, foundry_env,
                                                   monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        res = engine_config.resolve_engine(model_id="claude-opus-4-8")
        assert res.source == "anthropic", res.message

    def test_host_scheme_normalized(self, monkeypatch):
        monkeypatch.setenv("GEOTECH_FOUNDRY_HOST", "https://h.example.com/")
        assert engine_config.foundry_base_url() == "https://h.example.com"


class TestModelChoices:
    def test_env_list_parsed_labels_and_bare(self, monkeypatch):
        monkeypatch.setenv(core.FOUNDRY_MODELS_ENV,
                           f"GPT 5.2={GPT_RID}, {CLAUDE_RID}")
        fm = core.foundry_model_choices()
        assert [c["id"] for c in fm] == [GPT_RID, CLAUDE_RID]
        assert fm[0]["label"] == "GPT 5.2"
        assert fm[1]["label"] == CLAUDE_RID

    def test_foundry_models_prepended_and_default(self, monkeypatch):
        monkeypatch.setenv(core.FOUNDRY_MODELS_ENV, f"GPT 5.2={GPT_RID}")
        monkeypatch.delenv("GEOTECH_WEBAPP_MODEL", raising=False)
        choices = core.model_choices()
        assert choices[0]["id"] == GPT_RID
        assert core.default_model_id() == GPT_RID
        assert core.model_label(GPT_RID) == "GPT 5.2"

    def test_no_env_keeps_curated_defaults(self, monkeypatch):
        monkeypatch.delenv(core.FOUNDRY_MODELS_ENV, raising=False)
        monkeypatch.delenv("GEOTECH_WEBAPP_MODEL", raising=False)
        assert core.default_model_id() == core.MODEL_CHOICES[0]["id"]


class TestEntryStub:
    def test_app_path_points_at_packaged_script(self):
        from webapp.foundry_entry import app_path
        import os
        p = app_path()
        assert os.path.isfile(p) and p.endswith("app.py")
