"""Regression: the "Custom model id (advanced)" box must actually apply the id.

Found live on the first Foundry deployment (2026-07-17): pasting a Foundry RID
and clicking "Use this model" appeared to do nothing. The keyed Model selectbox
(``key="model_pick"``) keeps its own sticky state across reruns, so on the
rerun after the click it still reported the OLD model as "the user's pick";
``_picked != ss.model`` then reverted the RID within one render cycle. The same
clobber hit conversation resume (a saved conversation's model silently flipped
back to the widget's remembered value).

The fix: ``_resolve_and_build`` flags programmatic model changes
(``ss._model_dirty``) and the sidebar pushes ``ss.model`` INTO the widget's
session state before the widget is instantiated.

Runs headless via ``streamlit.testing.v1.AppTest`` — no API key, no network:
with no engine env vars at all, applying a RID must (a) survive renders and
(b) surface the SPECIFIC missing-token message, not the generic no-engine one.
"""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

_APP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

RID = "ri.language-model-service..language-model.gpt-5-2"

_ENGINE_ENVS = (
    "ANTHROPIC_API_KEY",
    "GEOTECH_FOUNDRY_TOKEN", "FOUNDRY_TOKEN",
    "GEOTECH_FOUNDRY_HOST", "FOUNDRY_HOSTNAME", "FOUNDRY_URL",
    "GEOTECH_FOUNDRY_MODELS",
)


def _mk_at(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    for env in _ENGINE_ENVS:
        monkeypatch.delenv(env, raising=False)
    return AppTest.from_file(_APP, default_timeout=60)


def _apply_rid(at):
    ti = [t for t in at.text_input if str(t.key or "").startswith("custom_model_")]
    bt = [b for b in at.button if str(b.key or "").startswith("use_custom_")]
    assert ti and bt, "custom-model input/button not found in the sidebar"
    ti[0].set_value(RID)
    bt[0].set_value(True)
    at.run()


def _model_pick(at):
    return [s for s in at.selectbox if s.key == "model_pick"][0]


def test_custom_rid_survives_renders(monkeypatch, tmp_path):
    at = _mk_at(monkeypatch, tmp_path).run()
    assert _model_pick(at).value != RID          # sanity: starts on a default
    _apply_rid(at)
    assert _model_pick(at).value == RID          # applied...
    at.run()
    assert _model_pick(at).value == RID          # ...and STAYS applied


def test_custom_rid_shows_specific_foundry_message(monkeypatch, tmp_path):
    at = _mk_at(monkeypatch, tmp_path).run()
    _apply_rid(at)
    texts = [str(el.value) for el in list(at.error) + list(at.warning)]
    assert any("Foundry RID" in t and "token" in t for t in texts), texts


def test_dropdown_switching_still_works(monkeypatch, tmp_path):
    """Model switches now ride the on_change callback (stale-echo hardening);
    a genuine dropdown pick must still switch the session model."""
    at = _mk_at(monkeypatch, tmp_path).run()
    before = at.session_state["model"]
    sb = _model_pick(at)
    assert len(sb.options) >= 2, "expected at least two model choices"
    idx = (sb.index + 1) % len(sb.options)
    sb.select_index(idx)
    at.run()
    after = at.session_state["model"]
    assert after != before, "dropdown pick did not switch the model"
    assert _model_pick(at).value == after
