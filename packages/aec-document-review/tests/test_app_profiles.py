"""Headless AppTest coverage for the two-page shell (aec_document_review.profiles in app.py).

The same ``aec_document_review/app.py`` serves the geotech page and the document-review
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

import aec_document_review.core as core
import aec_agent_core.engine_config as engine_config
from aec_document_review import profiles
from aec_agent_core.engine_config import EngineResolution

_APP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aec_document_review", "app.py")


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
    monkeypatch.setattr(core, "stream_turn", _stream_ok)

    def make(profile=None, identity=None, header=None):
        """``identity`` = DEV_IDENTITY (a single-user process); ``header`` =
        the value the IIS proxy would send (a multi-user host)."""
        if profile:
            monkeypatch.setenv(profiles.PROFILE_ENV, profile)
        if identity:
            monkeypatch.setenv("DEV_IDENTITY", identity)
        if header:
            from aec_document_review import identity as _identity
            monkeypatch.setattr(_identity, "_streamlit_header_values",
                                lambda: [header])
        return AppTest.from_file(_APP, default_timeout=30)
    make.built = built
    make.root = str(tmp_path)
    return make






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
    # owner: the person's unique key (domain__user), so two jdoes in two
    # domains never share a SharePoint folder; owner_name for display
    assert meta["owner"] == "corp__jdoe" and meta["owner_name"] == "jdoe"
    assert meta["page"] == "document_review"
    # titled after the attached file, not the orientation request (A10)
    assert meta["title"] == "plans.pdf"
    # nothing is queued twice
    assert "pending_orientation" not in at.session_state




