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
