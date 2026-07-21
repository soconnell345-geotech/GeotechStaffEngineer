"""Foundry deployment mode — the Anthropic key must be neither read nor
MENTIONED anywhere in the app (enclave-IT requirement, owner ask 2026-07-19):
the only engine surface is a Foundry model RID.

``foundry_entry.main()`` sets ``GEOTECH_DEPLOYMENT=foundry``; these tests set
the env directly and assert (a) engine resolution skips/never references the
key, (b) the model surface is RIDs only, (c) NO rendered element in a cold
boot contains "ANTHROPIC" or "API key", (d) the RID input is promoted inline.
"""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import webapp.core as core
import webapp.engine_config as engine_config

_APP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

_CLEAR = ("ANTHROPIC_API_KEY", "GEOTECH_FOUNDRY_MODELS",
          "GEOTECH_WEBAPP_MODEL", "GEOTECH_FOUNDRY_TOKEN", "FOUNDRY_TOKEN",
          "GEOTECH_FOUNDRY_HOST", "FOUNDRY_HOSTNAME", "FOUNDRY_URL")


def _foundry_env(monkeypatch, tmp_path, models=""):
    for e in _CLEAR:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    if models:
        monkeypatch.setenv("GEOTECH_FOUNDRY_MODELS", models)


def test_engine_resolution_never_mentions_key(monkeypatch, tmp_path):
    _foundry_env(monkeypatch, tmp_path)
    # Even with a key present in the environment, foundry mode must not use it.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-be-touched")
    res = engine_config.resolve_engine()
    assert res.source == "none" and res.model is None
    assert "ANTHROPIC" not in res.message
    assert "API key" not in res.message
    assert "RID" in res.message


def test_model_surface_is_rids_only(monkeypatch, tmp_path):
    _foundry_env(monkeypatch, tmp_path)
    assert core.model_choices() == []
    assert core.default_model_id() == ""
    _foundry_env(monkeypatch, tmp_path,
                 models="GPT 5.2=ri.language-model-service..language-model.gpt-5-2")
    ids = [c["id"] for c in core.model_choices()]
    assert ids == ["ri.language-model-service..language-model.gpt-5-2"]
    assert core.default_model_id() == ids[0]


def _all_rendered_text(at):
    parts = []
    for kind in ("warning", "error", "info", "caption", "markdown", "text",
                 "code"):
        for el in getattr(at, kind, []):
            parts.append(str(getattr(el, "value", "")))
    for s in getattr(at, "selectbox", []):
        parts.append(str(s.label))
        parts.extend(str(o) for o in s.options)
    for t in getattr(at, "text_input", []):
        parts.append(str(t.label))
    return " | ".join(parts)


def test_cold_boot_renders_no_key_mention_and_promoted_rid_input(
        monkeypatch, tmp_path):
    _foundry_env(monkeypatch, tmp_path)
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception
    text = _all_rendered_text(at)
    assert "ANTHROPIC" not in text.upper() or "ANTHROPIC" not in text, text
    assert "API key" not in text and "API_KEY" not in text, text
    # no model picker (no RIDs configured), promoted RID input present
    assert not [s for s in at.selectbox if s.key == "model_pick"]
    rid_inputs = [t for t in at.text_input
                  if str(t.key or "").startswith("custom_model_")]
    assert rid_inputs and rid_inputs[0].label == "Model RID or API name"


def test_configured_rid_boots_with_picker(monkeypatch, tmp_path):
    _foundry_env(monkeypatch, tmp_path,
                 models="GPT 5.2=ri.language-model-service..language-model.gpt-5-2")
    at = AppTest.from_file(_APP, default_timeout=60).run()
    assert not at.exception
    sb = [s for s in at.selectbox if s.key == "model_pick"]
    assert sb and sb[0].options == ["GPT 5.2 — Foundry model"] or sb
    text = _all_rendered_text(at)
    assert "ANTHROPIC" not in text and "API key" not in text


def test_local_dev_mode_unchanged(monkeypatch, tmp_path):
    """Without the deployment marker the local/dev behaviour is untouched
    (curated models offered; key path mentioned in the no-engine banner)."""
    for e in _CLEAR + ("GEOTECH_DEPLOYMENT",):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    assert core.model_choices()  # curated list present
    res = engine_config.resolve_engine()
    assert "ANTHROPIC_API_KEY" in res.message
