"""Headless AppTest coverage for the two-page shell (webapp.profiles in app.py).

The same ``webapp/app.py`` serves the geotech page and the document-review
page. These prove, with the engine and agent mocked:
  * the review page renders its own title, hides the specialist picker and
    hands ``build_agent`` the profile's overrides (no modules, no reference
    sub-agent, the document-review prompt, the signed-in person as markup
    author);
  * an identified user's conversations live under their own root, and the
    sidebar lists only those;
  * an upload on the review page triggers the automatic orientation turn,
    written into the transcript as the user's own message;
  * the geotech page with nobody signed in is byte-for-byte the old app
    (default root, specialist picker present, no orientation).
"""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import webapp.core as core
import webapp.engine_config as engine_config
from webapp import profiles
from webapp.engine_config import EngineResolution

_APP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _fake_engine(*_a, **_k):
    return EngineResolution(model=object(), source="prompter",
                            model_name="fake-model", message="")


def _stream_ok(_agent, _messages, _thread_id, **_kw):
    yield {"kind": "token", "text": "Oriented."}
    yield {"kind": "turn_done", "answer": "Oriented.", "turn_tokens": 3}


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """An AppTest factory that records every ``build_agent`` call's kwargs."""
    built = []
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    for e in ("GEOTECH_APP_PROFILE", "DEV_IDENTITY", "GEOTECH_USER_EMAIL",
              "GEOTECH_DEPLOYMENT", "GEOTECH_UPLOAD_MODE"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("GEOTECH_UPLOAD_MODE", "http")   # the stock uploader
    monkeypatch.setattr(engine_config, "resolve_engine", _fake_engine)

    def _build(model, attachments, temp_dir, artifacts, **kw):
        built.append(kw)
        return object()
    monkeypatch.setattr(core, "build_agent", _build)
    monkeypatch.setattr(core, "build_reviewer_agent",
                        lambda kind, *_a, **_k: f"reviewer:{kind}")
    monkeypatch.setattr(core, "stream_turn", _stream_ok)

    def make(profile=None, identity=None, header=None):
        """``identity`` = DEV_IDENTITY (a single-user process); ``header`` =
        the value the IIS proxy would send (a multi-user host)."""
        if profile:
            monkeypatch.setenv(profiles.PROFILE_ENV, profile)
        if identity:
            monkeypatch.setenv("DEV_IDENTITY", identity)
        if header:
            from webapp import identity as _identity
            monkeypatch.setattr(_identity, "_streamlit_header_values",
                                lambda: [header])
        return AppTest.from_file(_APP, default_timeout=30)
    make.built = built
    make.root = str(tmp_path)
    return make


def test_geotech_page_anonymous_is_the_old_app(harness):
    at = harness().run()
    assert not at.exception
    assert at.title[0].value.endswith("GeotechStaffEngineer")
    assert any(s.label == "Agent" for s in at.selectbox)       # specialists
    kw = harness.built[-1]
    assert "system_prompt" not in kw and kw.get("allowed_agents") is None
    tid = at.session_state["thread_id"]
    assert core.thread_root(tid) is None                        # default root
    assert core.conversation_dir(tid).startswith(
        os.path.join(harness.root, "conversations"))


def test_review_page_builds_the_document_review_agent(harness):
    at = harness(profile="document_review", header="CORP\\jdoe").run()
    assert not at.exception
    assert at.title[0].value.endswith("Document Review")
    assert not any(s.label == "Agent" for s in at.selectbox)   # no picker
    assert any("Signed in as" in c.value and "jdoe" in c.value
               for c in at.sidebar.caption)
    kw = harness.built[-1]
    assert kw["allowed_agents"] == () and kw["reference_mode"] == "off"
    assert kw["enable_calc_subagent"] is False
    assert "document-review assistant" in kw["system_prompt"]
    assert kw["markup_author"] == "jdoe via GeotechStaffEngineer (AI draft)"
    tid = at.session_state["thread_id"]
    expected_root = os.path.join(harness.root, "users", "corp__jdoe",
                                 "document_review")
    assert core.thread_root(tid) == os.path.abspath(expected_root)
    assert core.conversation_dir(tid).startswith(expected_root)


def test_upload_on_the_review_page_triggers_orientation(harness, tmp_path):
    at = harness(profile="document_review", header="CORP\\jdoe").run()
    assert not at.exception
    # Stage an attachment the way the sidebar handler does, then queue the
    # orientation exactly as the upload branch does, and rerun: the request
    # goes out as the user's message and the mocked stream answers it.
    tid = at.session_state["thread_id"]
    temp_dir = at.session_state["temp_dir"]
    atts = core.stage_uploads(at.session_state["attachments"], temp_dir,
                              [("plans.pdf", b"%PDF-1.4 fake")])
    at.session_state["pending_orientation"] = [a.key for a in atts]
    at.run()
    assert not at.exception
    roles = [e["role"] for e in at.session_state["transcript"]]
    assert roles == ["user", "assistant"]
    user_text = at.session_state["transcript"][0]["text"]
    assert "orientation" in user_text and "plans.pdf" in user_text
    assert "Do not review it yet" in user_text
    assert at.session_state["transcript"][1]["text"] == "Oriented."
    # ...and it is persisted under the user's own root, tagged owner + page
    meta = core.load_meta(tid)
    assert meta["owner"] == "jdoe" and meta["page"] == "document_review"
    # nothing is queued twice
    assert "pending_orientation" not in at.session_state


def test_single_user_identity_keeps_the_geotech_layout(harness):
    """DEV_IDENTITY (and the Databricks launcher's email) name one person in
    a process that serves only them: a display name and a markup author, no
    new folder — the geotech page is the old app; the review page sits in
    its own folder beside it."""
    at = harness(identity="CORP\\jdoe").run()
    assert not at.exception
    assert any("Signed in as" in c.value for c in at.sidebar.caption)
    tid = at.session_state["thread_id"]
    assert core.thread_root(tid) is None
    assert harness.built[-1]["markup_author"].startswith("jdoe via ")
    at2 = harness(profile="document_review", identity="CORP\\jdoe").run()
    assert not at2.exception
    tid2 = at2.session_state["thread_id"]
    assert core.thread_root(tid2) == os.path.abspath(
        os.path.join(harness.root, "pages", "document_review"))


def test_geotech_page_never_orients(harness):
    at = harness(identity="CORP\\jdoe").run()
    temp_dir = at.session_state["temp_dir"]
    core.stage_uploads(at.session_state["attachments"], temp_dir,
                       [("log.pdf", b"%PDF-1.4 fake")])
    # even a queued flag is ignored on a page whose profile does not orient
    at.session_state["pending_orientation"] = ["log.pdf"]
    at.run()
    assert not at.exception
    assert at.session_state["transcript"] == []
