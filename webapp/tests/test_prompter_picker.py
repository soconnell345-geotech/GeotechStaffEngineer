"""Prompter model picker (owner TODO 2026-07-31): a deployment builder that
accepts a model id + GEOTECH_PROMPTER_MODELS choices in the sidebar picker."""

import pytest

import webapp.core as core
import webapp.engine_config as engine_config
from webapp import databricks_launcher as dl


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    engine_config.register_model_builder(None)
    for e in ("GEOTECH_PROMPTER_MODELS", "GEOTECH_FOUNDRY_MODELS",
              "GEOTECH_WEBAPP_MODEL", "GEOTECH_DEPLOYMENT",
              "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(e, raising=False)
    yield
    engine_config.register_model_builder(None)


# ---------------------------------------------------------------------------
# engine_config: builder receives the picker selection
# ---------------------------------------------------------------------------

class _Model:
    def __init__(self, model_id=None):
        self.model_id = model_id


def test_arg_builder_receives_picked_model():
    engine_config.register_model_builder(lambda mid=None: _Model(mid))
    res = engine_config.resolve_engine("funhouse-gpt-medium")
    assert res.ok and res.source == "prompter"
    assert res.model.model_id == "funhouse-gpt-medium"
    assert res.model_name == "funhouse-gpt-medium"


def test_arg_builder_none_selection_uses_default():
    engine_config.register_model_builder(lambda mid=None: _Model(mid))
    res = engine_config.resolve_engine()
    assert res.ok and res.model.model_id is None
    assert res.model_name == "_Model"            # no selection -> type name


def test_zero_arg_builder_unchanged():
    sentinel = _Model()
    engine_config.register_model_builder(lambda: sentinel)
    res = engine_config.resolve_engine("funhouse-gpt-medium")
    assert res.ok and res.model is sentinel
    assert res.model_name == "_Model"            # deployment-fixed


def test_builder_failure_still_reported():
    def _boom(mid=None):
        raise RuntimeError("nope")
    engine_config.register_model_builder(_boom)
    res = engine_config.resolve_engine("x")
    assert res.source == "error" and "nope" in res.message


# ---------------------------------------------------------------------------
# core: choices env + picker gating
# ---------------------------------------------------------------------------

def test_prompter_model_choices_parse(monkeypatch):
    monkeypatch.setenv(core.PROMPTER_MODELS_ENV,
                       "GPT deep=funhouse-gpt-high, funhouse-gpt-medium,,")
    got = core.prompter_model_choices()
    assert [(c["label"], c["id"]) for c in got] == [
        ("GPT deep", "funhouse-gpt-high"),
        ("funhouse-gpt-medium", "funhouse-gpt-medium")]


def test_model_choices_prompter_only_when_builder_and_env(monkeypatch):
    monkeypatch.setenv(core.PROMPTER_MODELS_ENV,
                       "funhouse-gpt-high,funhouse-gpt-medium")
    # env alone (no builder) does NOT hijack the picker
    ids = [c["id"] for c in core.model_choices()]
    assert "claude-opus-4-8" in ids
    # builder + env -> Prompter models are the only choices
    engine_config.register_model_builder(lambda mid=None: _Model(mid))
    ids = [c["id"] for c in core.model_choices()]
    assert ids == ["funhouse-gpt-high", "funhouse-gpt-medium"]
    assert core.default_model_id() == "funhouse-gpt-high"
    assert core.model_label("funhouse-gpt-medium") == "funhouse-gpt-medium"


def test_builder_without_env_keeps_current_behavior():
    engine_config.register_model_builder(lambda: _Model())
    ids = [c["id"] for c in core.model_choices()]
    assert "claude-opus-4-8" in ids              # curated list still shown
    # (the app's "fixed by the deployment" caption covers this case)


# ---------------------------------------------------------------------------
# launcher: publishes the choices + model-aware bootstrap builder
# ---------------------------------------------------------------------------

class _FakeSpark:
    class conf:
        @staticmethod
        def get(key):
            return {dl._ORG_ID_KEY: "1", dl._CLUSTER_ID_KEY: "c",
                    "spark.databricks.workspaceUrl": "h.example"}[key]


def _launch(monkeypatch, **kw):
    captured = {}

    def popen(argv, env=None, **kwargs):
        captured["env"] = env

        class _P:
            def terminate(self):
                pass
        return _P()

    h = dl.run_on_databricks(spark=_FakeSpark(), quiet=True, _popen=popen, **kw)
    h.stop()
    return captured["env"]


def test_launcher_default_models_env(monkeypatch, tmp_path):
    env = _launch(monkeypatch)
    assert env["GEOTECH_PROMPTER_MODELS"] == \
        "funhouse-gpt-high,funhouse-gpt-medium"


def test_launcher_models_override_and_disable(monkeypatch):
    env = _launch(monkeypatch, models=["Fast=funhouse-gpt-medium"])
    assert env["GEOTECH_PROMPTER_MODELS"] == "Fast=funhouse-gpt-medium"
    env = _launch(monkeypatch, models=[])
    assert "GEOTECH_PROMPTER_MODELS" not in env


def test_bootstrap_builder_is_model_aware():
    src = dl.render_bootstrap_script(
        app_path="/x/app.py", repo_root="/r", base="/b", port=8501,
        model="funhouse-gpt-high")
    assert "def _build(model_id=None):" in src
    assert "model=model_id or MODEL" in src
