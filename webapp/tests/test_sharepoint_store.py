"""Offline tests for the SharePoint permanent-storage mirror.

A fake Funhouse file manager stands in for the SDK (which only exists on
Databricks); the real conversation-directory layout comes from webapp.core.
"""

import json
import os
import time

import pytest

import webapp.core as core
import webapp.sharepoint_store as sp

_ENVS = (sp.ENV_SITE, sp.ENV_CLIENT_ID, sp.ENV_CLIENT_SECRET, sp.ENV_TOKEN,
         sp.ENV_TOKEN_FILE, sp.ENV_DRIVE, sp.ENV_ROOT)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for e in _ENVS:
        monkeypatch.delenv(e, raising=False)
    yield


class FakeFM:
    """Records calls; can be told to fail specific uploads."""

    def __init__(self, fail_names=()):
        self.uploads = []          # (local, remote, overwrite)
        self.folders = []
        self.fail_names = set(fail_names)

    def create_folder(self, path):
        self.folders.append(path)
        return True

    def upload_file(self, local, remote, overwrite=False):
        if os.path.basename(remote) in self.fail_names:
            raise RuntimeError("boom")
        self.uploads.append((local, remote, overwrite))
        return True

    def get_web_url(self, path):
        return f"https://sp.example/{path}"


def _make_conversation(root, tid="SPT1"):
    core.ensure_conversation(tid, root=root)
    conv = core.conversation_dir(tid, root)
    files = os.path.join(conv, "files")
    os.makedirs(files, exist_ok=True)
    with open(os.path.join(files, "calc_package.pdf"), "wb") as fh:
        fh.write(b"%PDF fake")
    return tid, conv


# ---------------------------------------------------------------------------
# configured()
# ---------------------------------------------------------------------------

def test_configured_env_logic(monkeypatch):
    assert not sp.configured()
    monkeypatch.setenv(sp.ENV_SITE, "https://t.sharepoint.com/sites/x")
    assert not sp.configured()                       # site alone insufficient
    monkeypatch.setenv(sp.ENV_TOKEN, "tok")
    assert sp.configured()                           # site + token
    monkeypatch.delenv(sp.ENV_TOKEN)
    monkeypatch.setenv(sp.ENV_TOKEN_FILE, "/tmp/tok.txt")
    assert sp.configured()                           # site + token file
    monkeypatch.delenv(sp.ENV_TOKEN_FILE)
    monkeypatch.setenv(sp.ENV_CLIENT_ID, "cid")
    assert not sp.configured()                       # id without secret
    monkeypatch.setenv(sp.ENV_CLIENT_SECRET, "sec")
    assert sp.configured()                           # site + id + secret


def test_token_file_route_reads_fresh_token(tmp_path, monkeypatch):
    """The token-file route builds a provider that re-reads the file each call
    (so the notebook-side refresher's new tokens are picked up)."""
    import sys, types
    tok_file = tmp_path / "sp_token.txt"
    tok_file.write_text("tok-1")
    monkeypatch.setenv(sp.ENV_SITE, "https://t.sharepoint.com/sites/x")
    monkeypatch.setenv(sp.ENV_TOKEN_FILE, str(tok_file))
    monkeypatch.setenv(sp.ENV_DRIVE, "Documents")

    captured = {}

    def fake_create_from_provider(site, provider, **kwargs):
        captured["site"], captured["provider"] = site, provider
        captured["kwargs"] = kwargs
        return types.SimpleNamespace(file_manager="FAKE_FM")

    graph_mod = types.ModuleType(
        "funhouse.services.sharepoint.graph.graph_client")
    graph_mod.create_sharepoint_client_from_token_provider = \
        fake_create_from_provider
    for name, mod in (
        ("funhouse", types.ModuleType("funhouse")),
        ("funhouse.services", types.ModuleType("funhouse.services")),
        ("funhouse.services.sharepoint",
         types.ModuleType("funhouse.services.sharepoint")),
        ("funhouse.services.sharepoint.graph",
         types.ModuleType("funhouse.services.sharepoint.graph")),
        ("funhouse.services.sharepoint.graph.graph_client", graph_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    fm = sp._build_file_manager()
    assert fm == "FAKE_FM"
    assert captured["site"] == "https://t.sharepoint.com/sites/x"
    assert captured["kwargs"] == {"drive_name": "Documents"}
    assert captured["provider"]() == "tok-1"
    tok_file.write_text("tok-2")                    # refresher rotated it
    assert captured["provider"]() == "tok-2"


# ---------------------------------------------------------------------------
# mirror_conversation
# ---------------------------------------------------------------------------

def test_mirror_uploads_full_layout_then_skips(tmp_path):
    root = str(tmp_path)
    tid, conv = _make_conversation(root)
    fm = FakeFM()
    store = sp.SharePointStore(file_manager=fm)

    s1 = store.mirror_conversation(tid, root=root)
    assert not s1["errors"]
    assert s1["uploaded"] >= 2                       # meta.json + files/... pdf
    remotes = [r for (_l, r, _o) in fm.uploads]
    base = f"{sp.DEFAULT_ROOT}/conversations/{tid}"
    assert all(r.startswith(base + "/") for r in remotes)
    assert f"{base}/files/calc_package.pdf" in remotes
    assert all(o is True for (_l, _r, o) in fm.uploads)   # deterministic replace
    # manifest exists locally and is never uploaded
    assert os.path.exists(os.path.join(conv, sp.MANIFEST_NAME))
    assert not any(sp.MANIFEST_NAME in r for r in remotes)
    assert s1["web_url"] == f"https://sp.example/{base}"

    # second pass: nothing changed -> all skipped, no new uploads
    n = len(fm.uploads)
    s2 = store.mirror_conversation(tid, root=root)
    assert s2["uploaded"] == 0 and s2["skipped"] >= 2
    assert len(fm.uploads) == n


def test_mirror_reuploads_changed_file(tmp_path):
    root = str(tmp_path)
    tid, conv = _make_conversation(root)
    fm = FakeFM()
    store = sp.SharePointStore(file_manager=fm)
    store.mirror_conversation(tid, root=root)
    n = len(fm.uploads)

    target = os.path.join(conv, "files", "calc_package.pdf")
    time.sleep(0.01)
    with open(target, "wb") as fh:
        fh.write(b"%PDF fake v2 - longer")
    s = store.mirror_conversation(tid, root=root)
    assert s["uploaded"] == 1
    assert len(fm.uploads) == n + 1
    assert fm.uploads[-1][1].endswith("files/calc_package.pdf")


def test_mirror_upload_failure_is_captured_not_raised(tmp_path):
    root = str(tmp_path)
    tid, conv = _make_conversation(root)
    fm = FakeFM(fail_names={"calc_package.pdf"})
    store = sp.SharePointStore(file_manager=fm)
    s = store.mirror_conversation(tid, root=root)
    assert any("calc_package.pdf" in e for e in s["errors"])
    assert s["uploaded"] >= 1                        # the others still went up
    # failed file is NOT marked done -> retried next sync
    fm.fail_names.clear()
    s2 = store.mirror_conversation(tid, root=root)
    assert s2["uploaded"] == 1 and not s2["errors"]


def test_mirror_missing_conversation_is_noop(tmp_path):
    store = sp.SharePointStore(file_manager=FakeFM())
    s = store.mirror_conversation("NOPE", root=str(tmp_path))
    assert s["uploaded"] == 0 and s["skipped"] == 0 and not s["errors"]


def test_mirror_without_client_reports_error(tmp_path, monkeypatch):
    """Configured env but SDK missing/unbuildable -> readable error, no raise."""
    root = str(tmp_path)
    tid, _ = _make_conversation(root)
    monkeypatch.setenv(sp.ENV_SITE, "https://t.sharepoint.com/sites/x")
    monkeypatch.setenv(sp.ENV_TOKEN, "tok")
    store = sp.SharePointStore()                     # no injected fm
    monkeypatch.setattr(sp, "_build_file_manager",
                        lambda: (_ for _ in ()).throw(ImportError("no funhouse")))
    s = store.mirror_conversation(tid, root=root)
    assert s["uploaded"] == 0
    assert any("SharePoint client" in e for e in s["errors"])


def test_custom_root_env(tmp_path, monkeypatch):
    monkeypatch.setenv(sp.ENV_ROOT, "Shared Documents/Custom/App/")
    store = sp.SharePointStore(file_manager=FakeFM())
    assert store.session_folder("T1") == \
        "Shared Documents/Custom/App/conversations/T1"


def test_folder_chain_created_once(tmp_path):
    root = str(tmp_path)
    tid, _ = _make_conversation(root)
    fm = FakeFM()
    store = sp.SharePointStore(file_manager=fm)
    store.mirror_conversation(tid, root=root)
    store.mirror_conversation(tid, root=root)        # second sync: no re-mkdir
    assert len(fm.folders) == len(set(fm.folders))
    base = f"{sp.DEFAULT_ROOT}/conversations/{tid}"
    assert f"{base}/files" in fm.folders
