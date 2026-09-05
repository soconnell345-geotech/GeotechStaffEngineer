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


# ---------------------------------------------------------------------------
# Folder naming (owner request 2026-09-04: name the mirror after the
# conversation, not the thread-id hex)
# ---------------------------------------------------------------------------

def test_sanitize_folder_name_strips_what_sharepoint_rejects():
    assert sp.sanitize_folder_name("Praia Downdrag") == "Praia_Downdrag"
    # every forbidden character becomes a single underscore run
    assert sp.sanitize_folder_name('a/b\\c:d*e?f"g<h>i|j#k%l') == \
        "a_b_c_d_e_f_g_h_i_j_k_l"
    assert sp.sanitize_folder_name("tabs\tand\nnewlines") == \
        "tabs_and_newlines"
    # SharePoint rejects leading/trailing dots and spaces; inner spaces became
    # underscores first, so the trim has only dots/underscores left to remove
    assert sp.sanitize_folder_name("  ..Site Plan..  ") == "Site_Plan"
    # reserved tokens are broken up, not passed through
    assert "_vti_" not in sp.sanitize_folder_name("x_vti_y")
    assert sp.sanitize_folder_name("~$draft") == "draft"
    # length cap, and nothing usable -> empty (caller falls back to thread id)
    assert len(sp.sanitize_folder_name("z" * 300)) == sp.MAX_NAME_CHARS
    assert sp.sanitize_folder_name("///") == ""
    assert sp.sanitize_folder_name(None) == "" and sp.sanitize_folder_name("") == ""
    # non-ASCII letters are legal in SharePoint and are kept
    assert sp.sanitize_folder_name("Écrou naïf") == "Écrou_naïf"


def test_conversation_folder_uses_title_and_creation_date():
    created = time.mktime((2026, 9, 4, 13, 0, 0, 0, 0, -1))
    meta = {"thread_id": "abc123def456", "title": "Praia Downdrag",
            "created": created}
    assert sp.conversation_folder("abc123def456", meta) == \
        "Praia_Downdrag_2026-09-04"


def test_conversation_folder_falls_back_to_thread_id_when_unnamed():
    for title in (None, "", "   ", "New conversation", "Untitled"):
        meta = {"thread_id": "deadbeef", "title": title, "created": 0.0}
        assert sp.conversation_folder("deadbeef", meta) == "deadbeef"
    # no meta at all (never-synced / unreadable) also falls back
    assert sp.conversation_folder("deadbeef", None) == "deadbeef"
    # a title made only of forbidden characters leaves nothing to name it with
    assert sp.conversation_folder(
        "deadbeef", {"title": "??/??", "created": 0.0}) == "deadbeef"


def test_conversation_folder_disambiguates_same_name_same_day():
    """Two same-named conversations must never share ONE remote folder — they
    would overwrite each other's meta.json / transcript.jsonl."""
    day = time.mktime((2026, 9, 4, 9, 0, 0, 0, 0, -1))
    older = {"thread_id": "aaaaaaaaaaaa", "title": "Site A", "created": day}
    newer = {"thread_id": "bbbbbbbbbbbb", "title": "Site A", "created": day + 60}
    both = [older, newer]
    # the earliest-created keeps the clean name; the later one is sharded
    assert sp.conversation_folder("aaaaaaaaaaaa", older, both) == \
        "Site_A_2026-09-04"
    assert sp.conversation_folder("bbbbbbbbbbbb", newer, both) == \
        "Site_A_2026-09-04_bbbbbb"
    # and the two never collide
    assert sp.conversation_folder("aaaaaaaaaaaa", older, both) != \
        sp.conversation_folder("bbbbbbbbbbbb", newer, both)
    # a different day is already unique -> no shard
    other_day = {"thread_id": "cccccccccccc", "title": "Site A",
                 "created": day + 86400}
    assert sp.conversation_folder("cccccccccccc", other_day,
                                  both + [other_day]) == "Site_A_2026-09-05"


def test_session_folder_uses_the_conversation_name(tmp_path):
    root = str(tmp_path)
    tid = "1234567890ab"
    core.ensure_conversation(tid, root=root)
    core.rename_conversation(tid, "Praia Downdrag", root=root)
    store = sp.SharePointStore(file_manager=FakeFM())
    folder = store.session_folder(tid, root=root)
    assert folder.startswith(f"{sp.DEFAULT_ROOT}/conversations/Praia_Downdrag_")
    assert tid not in folder


def test_mirror_lands_in_the_named_folder(tmp_path):
    root = str(tmp_path)
    tid, _conv = _make_conversation(root, "namedconv001")
    core.rename_conversation(tid, "MCAC Micropile", root=root)
    fm = FakeFM()
    store = sp.SharePointStore(file_manager=fm)
    s = store.mirror_conversation(tid, root=root)
    assert not s["errors"] and s["uploaded"] >= 2
    assert "/conversations/MCAC_Micropile_" in s["folder"]
    assert all("/MCAC_Micropile_" in r for (_l, r, _o) in fm.uploads)


def test_rename_moves_the_mirror_and_leaves_a_pointer(tmp_path):
    """A later rename re-mirrors under the new name and drops MOVED.txt in the
    old folder (the file manager has no server-side move)."""
    root = str(tmp_path)
    tid, _conv = _make_conversation(root, "renameconv01")
    core.rename_conversation(tid, "Working Name", root=root)
    fm = FakeFM()
    store = sp.SharePointStore(file_manager=fm)
    first = store.mirror_conversation(tid, root=root)
    old_folder = first["folder"]
    assert "Working_Name_" in old_folder
    n_before = len(fm.uploads)

    core.rename_conversation(tid, "Praia Downdrag", root=root)
    second = store.mirror_conversation(tid, root=root)
    assert second["renamed_from"] == old_folder
    assert "Praia_Downdrag_" in second["folder"]
    # every file re-uploaded under the new name; nothing left "skipped"
    assert second["uploaded"] == n_before and second["skipped"] == 0
    new_remotes = [r for (_l, r, _o) in fm.uploads[n_before:]]
    assert any(r == f"{old_folder}/{sp.MOVED_NAME}" for r in new_remotes)
    assert any(r.startswith(second["folder"] + "/") for r in new_remotes)
    # the web_url is re-read for the new folder rather than kept stale
    assert second["web_url"] == f"https://sp.example/{second['folder']}"

    # a third sync is a no-op again (manifest now tracks the new folder)
    third = store.mirror_conversation(tid, root=root)
    assert third["uploaded"] == 0 and third["skipped"] >= 2
    assert "renamed_from" not in third


def test_old_flat_manifest_is_read_not_discarded(tmp_path):
    """Upgrading the app must not re-upload every conversation: a pre-folder
    manifest (the flat rel->stamp map) is still honored."""
    root = str(tmp_path)
    tid, conv = _make_conversation(root, "flatmanifest")
    fm = FakeFM()
    store = sp.SharePointStore(file_manager=fm)
    store.mirror_conversation(tid, root=root)
    # rewrite the manifest in the OLD flat shape, with no folder recorded
    files = sp._manifest_files(sp._load_manifest(conv))
    assert files
    with open(os.path.join(conv, sp.MANIFEST_NAME), "w", encoding="utf-8") as fh:
        json.dump(files, fh)

    store2 = sp.SharePointStore(file_manager=FakeFM())
    s = store2.mirror_conversation(tid, root=root)
    assert s["uploaded"] == 0 and s["skipped"] >= 2   # stamps still matched
    assert "renamed_from" not in s                    # no folder -> no "move"


def test_moved_pointer_failure_is_captured_not_raised(tmp_path):
    root = str(tmp_path)
    tid, _conv = _make_conversation(root, "movedfail001")
    core.rename_conversation(tid, "First", root=root)
    fm = FakeFM()
    store = sp.SharePointStore(file_manager=fm)
    store.mirror_conversation(tid, root=root)
    core.rename_conversation(tid, "Second", root=root)
    fm.fail_names = {sp.MOVED_NAME}
    s = store.mirror_conversation(tid, root=root)
    assert any(sp.MOVED_NAME in e for e in s["errors"])
    assert s["uploaded"] >= 2            # the real re-mirror still happened
    assert "Second_" in s["folder"]


def test_fix_web_url_repairs_single_slash():
    assert sp.fix_web_url("https:/usdos.sharepoint.com/sites/X?web=1") == \
        "https://usdos.sharepoint.com/sites/X?web=1"
    assert sp.fix_web_url("http:/host/p") == "http://host/p"
    # already-valid and non-URL inputs untouched
    assert sp.fix_web_url("https://ok.example/p") == "https://ok.example/p"
    assert sp.fix_web_url("") == "" and sp.fix_web_url(None) == ""


def test_mirror_web_url_is_normalized(tmp_path):
    root = str(tmp_path)
    tid, _ = _make_conversation(root, "SPURL1")

    class SlashFM(FakeFM):
        def get_web_url(self, path):
            return f"https:/sp.example/{path}?web=1"     # SDK single-slash form
    store = sp.SharePointStore(file_manager=SlashFM())
    s = store.mirror_conversation(tid, root=root)
    assert s["web_url"].startswith("https://sp.example/")
