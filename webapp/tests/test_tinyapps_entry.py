"""The Tiny Apps entry: two pages over one shell, and the profiles behind them."""

import os

import pytest

from webapp import engine_config, profiles, tinyapps_entry
from webapp.identity import ANONYMOUS, parse_principal


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for e in ("GEOTECH_DEPLOYMENT", "GEOTECH_APP_PROFILE", "PROMPTER_URL",
              "PROMPTER_MODEL", "PROMPTER_API_KEY"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setattr(tinyapps_entry, "_ENGINE_REGISTERED", False)
    engine_config.register_model_builder(None)
    yield
    engine_config.register_model_builder(None)


def test_app_path_points_at_packaged_app():
    p = tinyapps_entry.app_path()
    assert os.path.basename(p) == "app.py"
    assert os.path.exists(p)


def test_deployment_marker_and_engine_registration(monkeypatch):
    tinyapps_entry.mark_deployment()
    assert os.environ["GEOTECH_DEPLOYMENT"] == "tinyapps"
    assert engine_config.is_tinyapps_deployment()
    # no Prompter settings: nothing registered, the app will show its banner
    assert tinyapps_entry.register_engine() is False
    assert not engine_config.has_model_builder()


def test_main_runs_two_pages_document_review_first(monkeypatch):
    """``main`` sets the page config once, flags it, and hands ``st.navigation``
    the Document Review page as the default with the geotech page second."""
    import streamlit as st
    calls = {}
    session = {}

    class _Nav:
        def run(self):
            calls["ran"] = True

    def _page(fn, title=None, icon=None, url_path=None, default=False):
        return {"fn": fn, "title": title, "icon": icon, "url_path": url_path,
                "default": default}

    def _navigation(pages):
        calls["pages"] = pages
        return _Nav()

    monkeypatch.setattr(st, "Page", _page)
    monkeypatch.setattr(st, "navigation", _navigation)
    monkeypatch.setattr(st, "set_page_config",
                        lambda **kw: calls.setdefault("config", kw))
    monkeypatch.setattr(st, "session_state", session)

    tinyapps_entry.main()

    assert calls["ran"]
    pages = calls["pages"]
    # the default page is served at "/" — Streamlit rejects a url_path on it
    assert [p["url_path"] for p in pages] == [None, "geotech"]
    assert pages[0]["default"] is True and pages[1]["default"] is False
    assert pages[0]["title"] == "Document Review"
    assert pages[1]["title"] == "GeotechStaffEngineer"
    # each page function records its profile, then runs the shell
    # each page function sets THE page config (its own title and icon),
    # flags it for the shell, records its profile, then runs the shell
    ran = []
    monkeypatch.setattr("runpy.run_path", lambda *a, **k: ran.append(a[0]))
    pages[0]["fn"]()
    assert session[profiles.SESSION_KEY] == "document_review"
    assert session["_page_config_set"] is True
    assert calls["config"]["page_title"] == "Document Review"
    calls.pop("config")
    pages[1]["fn"]()
    assert session[profiles.SESSION_KEY] == "geotech"
    assert calls["config"]["page_title"] == "GeotechStaffEngineer"
    assert len(ran) == 2 and all(p.endswith("app.py") for p in ran)


# ------------------------------------------------------------------ profiles

def test_default_profile_is_geotech_and_env_can_pick(monkeypatch):
    assert profiles.current() is profiles.GEOTECH
    monkeypatch.setenv(profiles.PROFILE_ENV, "document_review")
    assert profiles.current() is profiles.DOCUMENT_REVIEW
    assert profiles.get("nonsense") is profiles.GEOTECH


def test_document_review_profile_builds_a_different_agent():
    kw = profiles.DOCUMENT_REVIEW.build_kwargs()
    assert kw["allowed_agents"] == () and kw["reference_mode"] == "off"
    assert "geotechnical" not in kw["system_prompt"][:200].lower()
    assert "annotate_document" in kw["system_prompt"]
    assert "write_docx" in kw["system_prompt"]
    assert "_document_review_prompt" not in kw
    assert profiles.GEOTECH.build_kwargs() == {}
    assert profiles.DOCUMENT_REVIEW.orientation and not profiles.GEOTECH.orientation
    assert "{names}" in profiles.DOCUMENT_REVIEW.orientation_request


def test_session_root_keeps_the_legacy_layout_for_the_single_user_app(tmp_path):
    base = str(tmp_path)
    # nobody identified, DEV_IDENTITY, the Databricks launcher's email: ONE
    # person per process — the geotech page keeps the data root itself
    dev = parse_principal("CORP\\jdoe", "dev")
    email = parse_principal("jane.doe@state.gov", "email")
    for who in (ANONYMOUS, dev, email):
        assert not who.multi_user
        assert profiles.session_root(profiles.GEOTECH, who, base) == base
        assert profiles.session_root(profiles.DOCUMENT_REVIEW, who, base) == \
            os.path.join(base, "pages", "document_review")
    # the proxy header = a multi-user host: every page under the person
    jdoe = parse_principal("CORP\\jdoe", "header")
    assert jdoe.multi_user
    assert profiles.session_root(profiles.GEOTECH, jdoe, base) == \
        os.path.join(base, "users", "corp__jdoe", "geotech")
    assert profiles.session_root(profiles.DOCUMENT_REVIEW, jdoe, base) == \
        os.path.join(base, "users", "corp__jdoe", "document_review")


def test_empty_scope_builds_an_agent_with_no_dispatch_tools():
    """The document-review profile: no list_agents/call_agent, no reference
    sub-agents, the document-review prompt — over a fake model."""
    pytest.importorskip("deepagents")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from funhouse_agent.deep.agent import build_deep_agent, build_primary_tools

    names = {t.name for t in build_primary_tools(allowed_agents=())}
    assert not names & {"list_agents", "list_methods", "describe_method",
                        "call_agent"}
    assert "analyze_image" in names and "save_file" in names
    agent = build_deep_agent(FakeListChatModel(responses=["ok"]),
                             **profiles.DOCUMENT_REVIEW.build_kwargs())
    assert agent is not None
