""""Find a past conversation" lists each PAGE's own mirrors (2026-10-06).

Since 5.32.0 each page lists its own folder of mirrored conversations
(``SharePointStore.list_remote_conversations(owner, page)``). The owner then
found the geotech page's past conversations missing while the Document Review
page listed its own. The cause was not SharePoint and not the folder layout:
the two pages share ONE ``st.session_state`` (Streamlit's multipage design),
and the sidebar cached the listing under one key, ``sp_remote_list``. The
Document Review page is the app's root, so it is the page a session opens on;
its listing was cached first, and the geotech page then showed that cached
list -- the review page's conversations -- instead of asking for its own.

These tests drive the real shell (AppTest) over a fake store that answers per
page, open the review page first, switch to the geotech page in the SAME
session, and check that each page asked for and shows its own list -- and
that a Restore on the geotech page restores from the geotech folder.
"""

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import webapp.core as core
import webapp.engine_config as engine_config
import webapp.sharepoint_store as sharepoint_store
from webapp import profiles
from webapp.engine_config import EngineResolution

_APP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

#: What the fake SharePoint holds, per page: the geotech page's mirrors sit
#: flat under conversations/ (the layout every deployment has had), the
#: review page's under conversations/document_review/.
LISTS = {
    "geotech": [{"name": "Slope_question_2026-10-02",
                 "path": "R/conversations/Slope_question_2026-10-02",
                 "date": "2026-10-02"}],
    "document_review": [{"name": "Plan_review_2026-10-03",
                         "path": "R/conversations/document_review/"
                                 "Plan_review_2026-10-03",
                         "date": "2026-10-03"}],
}


class FakeStore:
    """The parts of SharePointStore the sidebar uses, answering per page."""

    configured = True
    last_sync = None

    def __init__(self):
        self.list_calls = []
        self.restore_calls = []

    def list_remote_conversations(self, owner=None, page=None):
        self.list_calls.append((owner, page))
        return [dict(e) for e in LISTS.get(page or "geotech", [])]

    def restore_conversation(self, name, root=None, overwrite=False,
                             owner=None, page=None):
        self.restore_calls.append((name, owner, page))
        return {"status": "error", "errors": ["test: not restored"],
                "thread_id": None, "downloaded": 0, "duration_s": 0.0}

    def mirror_conversation(self, thread_id, root=None):
        return {"uploaded": 0, "skipped": 0, "errors": [], "duration_s": 0.0}


def _fake_engine(*_a, **_k):
    return EngineResolution(model=object(), source="prompter",
                            model_name="fake-model", message="")


@pytest.fixture
def two_pages(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    for e in ("GEOTECH_APP_PROFILE", "DEV_IDENTITY", "GEOTECH_USER_EMAIL",
              "GEOTECH_DEPLOYMENT"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("GEOTECH_UPLOAD_MODE", "http")
    monkeypatch.setattr(engine_config, "resolve_engine", _fake_engine)
    monkeypatch.setattr(core, "build_agent", lambda *a, **k: object())
    monkeypatch.setattr(core, "build_reviewer_agent", lambda *a, **k: object())
    store = FakeStore()
    monkeypatch.setattr(sharepoint_store, "get_store", lambda *a, **k: store)
    # Open on the review page, the app's root, as a real session does.
    monkeypatch.setenv(profiles.PROFILE_ENV, "document_review")
    at = AppTest.from_file(_APP, default_timeout=30)
    return at, store


def _listed(at):
    """Every conversation name the sidebar shows as a restore row."""
    names = {e["name"] for rows in LISTS.values() for e in rows}
    shown = set()
    for c in at.sidebar.caption:
        text = str(c.value).lstrip("✓ ").strip()
        if text in names:
            shown.add(text)
    return shown


def test_cache_key_is_per_page_and_per_person():
    key = sharepoint_store.listing_cache_key
    # the geotech page is the same page however it is named
    assert key() == key(page="geotech") == key(owner=None, page="")
    assert key(page="document_review") != key(page="geotech")
    assert key(owner="Jane Doe", page="geotech") != key(page="geotech")
    assert key(owner="Jane Doe", page="document_review") != \
        key(owner="John Roe", page="document_review")


def test_each_page_lists_its_own_conversations_in_one_session(two_pages):
    at, store = two_pages
    at.run()
    assert not at.exception
    assert _listed(at) == {"Plan_review_2026-10-03"}
    assert store.list_calls[-1] == (None, "document_review")

    # Same browser session, now the geotech page (what the page link does).
    at.session_state[profiles.SESSION_KEY] = "geotech"
    at.run()
    assert not at.exception
    assert at.title[0].value.endswith("GeotechStaffEngineer")
    assert (None, "geotech") in store.list_calls
    assert _listed(at) == {"Slope_question_2026-10-02"}

    # ...and back: the review page still shows its own, without re-listing.
    n_calls = len(store.list_calls)
    at.session_state[profiles.SESSION_KEY] = "document_review"
    at.run()
    assert not at.exception
    assert _listed(at) == {"Plan_review_2026-10-03"}
    assert len(store.list_calls) == n_calls          # cached per page


def test_restore_on_the_geotech_page_uses_the_geotech_folder(two_pages):
    at, store = two_pages
    at.run()
    at.session_state[profiles.SESSION_KEY] = "geotech"
    at.run()
    button = next(b for b in at.sidebar.button
                  if b.key == "sp_restore_Slope_question_2026-10-02")
    button.click().run()
    assert not at.exception
    assert store.restore_calls == [("Slope_question_2026-10-02", None,
                                    "geotech")]
    # The failed restore is reported on THIS page and not on the other one.
    assert any("Restore failed" in e.value for e in at.sidebar.error)
    at.session_state[profiles.SESSION_KEY] = "document_review"
    at.run()
    assert not any("Restore failed" in e.value for e in at.sidebar.error)


def test_refresh_relists_only_this_page(two_pages):
    at, store = two_pages
    at.run()
    at.session_state[profiles.SESSION_KEY] = "geotech"
    at.run()
    before = list(store.list_calls)
    next(b for b in at.sidebar.button
         if b.key == "sp_restore_refresh").click().run()
    assert not at.exception
    new = store.list_calls[len(before):]
    assert new == [(None, "geotech")]
