"""TA-40's two edits to the page (``app.py``), made after task C's branch,
which holds the page, was merged with TA-40's.

* "Sync now" goes through ``sharepoint_store.save_after_change``: where the
  store saves in the background (Foundry, and SharePoint by the Tiny Apps
  route since TA-40) the sync is queued like a turn's save, instead of a
  large first mirror holding the clicker's page under the spinner. Where it
  does not, it mirrors at once as before.
* "Find a past conversation" shows why a listing failed (the store's
  ``RemoteListing.error``) instead of "Nothing mirrored yet".

Offline: the real page (AppTest) over a stand-in store; made-up names.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import aec_agent_core.engine_config as engine_config
import aec_document_review.core as core
import aec_document_review.sharepoint_store as sharepoint_store
from aec_agent_core.engine_config import EngineResolution
from aec_document_review import profiles

_APP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "aec_document_review", "app.py")

FAILED = ("SharePoint could not be listed: GraphError: 403 the app has no "
          "grant on this site")


class Store:
    """The parts of the store the sidebar uses. ``background`` as the real
    store's property; ``list_error`` makes the listing fail."""

    configured = True
    last_sync = None

    def __init__(self, background=False, list_error=None):
        self.background = background
        self.list_error = list_error
        self.queued = []
        self.mirrored = []

    def last_sync_for(self, thread_id):
        return None

    def save_soon(self, thread_id):
        self.queued.append(thread_id)

    def save_state(self, thread_id):
        return None

    def list_remote_conversations(self, owner=None, page=None):
        out = sharepoint_store.RemoteListing()
        out.error = self.list_error
        return out

    def mirror_conversation(self, thread_id, root=None):
        self.mirrored.append(thread_id)
        return {"uploaded": 0, "skipped": 0, "errors": [],
                "duration_s": 0.0, "thread_id": thread_id}


def _page(monkeypatch, tmp_path, store):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    for e in ("GEOTECH_APP_PROFILE", "DEV_IDENTITY", "GEOTECH_USER_EMAIL",
              "GEOTECH_DEPLOYMENT"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("GEOTECH_UPLOAD_MODE", "http")
    monkeypatch.setattr(engine_config, "resolve_engine", lambda *a, **k:
                        EngineResolution(model=object(), source="prompter",
                                         model_name="fake", message=""))
    monkeypatch.setattr(core, "build_agent", lambda *a, **k: object())
    monkeypatch.setattr(sharepoint_store, "get_store", lambda *a, **k: store)
    monkeypatch.setenv(profiles.PROFILE_ENV, "document_review")
    at = AppTest.from_file(_APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def _click(at, key=None, prefix=None):
    next(b for b in at.sidebar.button
         if (key is not None and b.key == key)
         or (prefix is not None and str(b.key).startswith(prefix))
         ).click().run()
    assert not at.exception, at.exception


# ---------------------------------------------------------------------------
# "Sync now"
# ---------------------------------------------------------------------------

def test_sync_now_is_queued_where_the_store_saves_in_the_background(
        monkeypatch, tmp_path):
    """Before: the page ran the whole mirror under its spinner, whatever
    the store."""
    store = Store(background=True)
    at = _page(monkeypatch, tmp_path, store)
    _click(at, prefix="spsync_")
    assert store.queued == [at.session_state["thread_id"]]
    assert store.mirrored == []


def test_sync_now_mirrors_at_once_where_the_store_does_not(
        monkeypatch, tmp_path):
    store = Store(background=False)
    at = _page(monkeypatch, tmp_path, store)
    _click(at, prefix="spsync_")
    assert store.mirrored == [at.session_state["thread_id"]]
    assert store.queued == []


# ---------------------------------------------------------------------------
# A listing that failed
# ---------------------------------------------------------------------------

def test_a_failed_listing_says_why(monkeypatch, tmp_path):
    """Before: "Nothing mirrored yet (or SharePoint could not be listed
    ...)", the same words for a failure as for an empty library."""
    at = _page(monkeypatch, tmp_path, Store(list_error=FAILED))
    _click(at, key="sp_restore_load")
    warnings = [str(w.value) for w in at.sidebar.warning]
    assert f"{FAILED} Refresh to try again." in warnings
    captions = [str(c.value) for c in at.sidebar.caption]
    assert not any(c.startswith("Nothing mirrored yet") for c in captions)


def test_an_empty_library_says_nothing_is_mirrored_yet(monkeypatch, tmp_path):
    at = _page(monkeypatch, tmp_path, Store())
    _click(at, key="sp_restore_load")
    captions = [str(c.value) for c in at.sidebar.caption]
    assert "Nothing mirrored yet." in captions
    assert not any("could not be listed" in str(w.value)
                   for w in at.sidebar.warning)
