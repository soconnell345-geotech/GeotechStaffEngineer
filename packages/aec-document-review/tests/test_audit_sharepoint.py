"""Audit of the SharePoint store (2026-11-18): findings with a small fix.

The Graph service is the fake of ``test_ta40_sharepoint_off_turn`` (its
``urllib.request.urlopen`` stand-in). Every tenant, site, person, file and
secret here is made up. Each test fails before its fix.
"""

import time
import urllib.request

import pytest

import aec_document_review.core as core
import aec_document_review.graph_sharepoint as gs
import aec_document_review.sharepoint_store as sp
from tests.test_ta40_sharepoint_off_turn import (FakeGraph, _conversation,
                                                 _http, _store)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (sp.ENV_SITE, sp.ENV_CLIENT_ID, sp.ENV_CLIENT_SECRET,
                 sp.ENV_TOKEN, sp.ENV_TOKEN_FILE, sp.ENV_DRIVE, sp.ENV_ROOT,
                 gs.ENV_CLOUD, gs.ENV_LOGIN_URL, gs.ENV_API_URL,
                 "GEOTECH_FOUNDRY_CONVERSATION_DATASET"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(gs.time, "sleep", lambda s: None)
    yield


@pytest.fixture
def graph(monkeypatch):
    fake = FakeGraph()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def _wait_for(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_a_failed_background_save_is_tried_again_without_a_new_turn(
        tmp_path, graph, monkeypatch):
    """A save queued after a turn meets a busy service (503 on every
    send). Before the fix nothing tried it again until that tester's next
    turn: a tester who stopped there had their last turn kept only on the
    host's disk. Now the store tries it again by itself."""
    monkeypatch.setattr(sp, "SAVE_RETRY_S", 0.05, raising=False)
    monkeypatch.setattr(gs, "BUSY_HOLD_MIN_S", 0.0)  # the service's wait
    root = str(tmp_path)
    _conversation(root, "AAAA1111", {"files/a.pdf": b"x" * 10})
    core.register_thread_root("AAAA1111", root)
    busy = {"on": True}

    def before(method, path, req):
        if busy["on"] and method == "PUT":
            raise _http(req.full_url, 503)

    graph.before = before
    store = _store()
    sp.save_after_change(store, "AAAA1111")
    assert _wait_for(lambda: (store.last_sync_for("AAAA1111") or {})
                     .get("errors"))
    busy["on"] = False
    assert _wait_for(lambda: any(p.endswith("files/a.pdf")
                                 for p in list(graph.files)), timeout=5.0), \
        "the failed save was never tried again"
    assert _wait_for(lambda: store.last_sync_for("AAAA1111")["errors"] == [])


def test_scope_holds_with_the_library_alone_as_the_root(monkeypatch):
    """With the base folder set to a document library alone, every spelling
    of another person's conversation folder is closed to a conversation."""
    pytest.importorskip("langchain_core")
    import aec_document_review.sharepoint_tools as spt
    monkeypatch.setenv(sp.ENV_ROOT, "Shared Documents")
    store = sp.SharePointStore(file_manager=object())
    monkeypatch.setattr(sp, "_STORE", store)
    scope = spt._Scope([store.conversations_base("corp__alice")],
                       "corp__alice")
    for path in ("Shared Documents/conversations/corp_bob/a.pdf",
                 "/sites/Alpha/Documents/conversations/corp_bob/a.pdf",
                 "sites/Alpha/Documents/conversations/corp_bob"):
        assert scope.where(spt._resolve(path)) == "private", path
    assert scope.where(spt._resolve(
        "Shared Documents/conversations/corp_alice/a.pdf")) == "own"


def test_a_half_set_up_app_registration_says_conversations_are_not_kept(
        monkeypatch):
    """Three of the four app-registration settings set, the secret's name
    misspelled: nothing reaches SharePoint, and before the fix the page
    showed no line saying so (``not_kept_reason`` spoke only on Foundry)."""
    from aec_document_review import tinyapps_settings
    for name in ("GRAPH_CLIENT_SECRET", "GRAPH_CLIENT_CERT", "GRAPH_CLIENT_KEY",
                 "SHAREPOINT_SITE_URLS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GRAPH_TENANT_ID", "made-up-tenant")
    monkeypatch.setenv("GRAPH_CLIENT_ID", "made-up-client")
    monkeypatch.setenv("GRAPH_CLIENT_SECERT", "made-up-secret-value")
    monkeypatch.setenv("SHAREPOINT_SITE_URL", "https://contoso.sharepoint.com/sites/Alpha")
    monkeypatch.setenv(tinyapps_settings.DOTENV_PATH_VAR, "/nonexistent/.env")
    tinyapps_settings.reset()
    try:
        store = sp.SharePointStore()
        assert not store.configured
        reason = store.not_kept_reason()
        assert reason and "GRAPH_CLIENT_SECRET" in reason
        assert "made-up-secret-value" not in reason
    finally:
        tinyapps_settings.reset()


def test_a_site_that_is_not_found_is_a_failed_listing(graph):
    """The site address names no site (a mistyped path: the site lookup
    answers 404). Before the fix ``ls`` read that 404 as an empty folder,
    so the past-conversations list said nothing was kept, with no error."""
    def before(method, path, req):
        if path.startswith("sites/") and ":" in path:
            raise _http(req.full_url, 404)

    graph.before = before
    listing = _store().list_remote_conversations()
    assert list(listing) == []
    assert listing.error and "404" in listing.error
