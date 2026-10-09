"""Other people's conversations are private; a conversation's file is shared
by its mirrored copy (live smoke wave 1, 2026-10-08: A1, A5, A6).

F10: on a multi-user host Bob's agent searched SharePoint for "10.17a",
found the file in Alice's conversation folder (``conversations/alice/...``),
downloaded it into Bob's conversation and described it. F04/F19/F26: "save
it to SharePoint and send me the link" uploaded a timestamped second copy
beside the mirror's own copy and linked the copy nothing updates.

The fake file manager answers like the Tiny Apps Graph client: library-
relative paths, ``[]`` for a missing folder. Every name is synthetic.
"""

import json
import os
import time

import pytest

pytest.importorskip("langchain_core")

import webapp.core as core
import webapp.sharepoint_store as sp
import webapp.sharepoint_tools as spt

ROOT = "Shared Documents/General/GSE_app"
ALICE_FILE = (f"{ROOT}/conversations/alice/document_review/"
              "Alice_review_2026-10-08/files/10.17a.pdf")
BOB_OLD = (f"{ROOT}/conversations/bob/document_review/"
           "Old_bob_review_2026-10-01/files/bob_notes.pdf")
SHARED = f"{ROOT}/uploaded references/shared_spec.pdf"
LEGACY = f"{ROOT}/conversations/Legacy_chat_2026-09-01/files/legacy_10.17a.pdf"


class TreeFM:
    def __init__(self, files):
        self.files = dict(files)
        self.downloads, self.uploads, self.listed = [], [], []

    @staticmethod
    def _norm(path):
        p = str(path).replace("\\", "/").strip().strip("/")
        if p.lower().startswith(("sites/", "teams/")):
            p = p.split("/", 2)[2] if p.count("/") >= 2 else ""
        return p

    def _children(self, folder):
        folder = self._norm(folder).rstrip("/")
        out, seen = [], set()
        for path, data in self.files.items():
            if not path.lower().startswith(folder.lower() + "/"):
                continue
            rest = path[len(folder) + 1:]
            head = rest.split("/", 1)[0]
            if "/" in rest:
                if head not in seen:
                    seen.add(head)
                    out.append({"name": head, "type": "folder",
                                "path": f"{folder}/{head}"})
            else:
                out.append({"name": head, "type": "file", "size": len(data),
                            "path": f"{folder}/{head}"})
        return out

    def ls(self, path):
        self.listed.append(path)
        return self._children(path)

    def get_folder_details(self, path):
        folder = self._norm(path).rstrip("/")
        return ({"id": "x", "folder": {}}
                if any(p.startswith(folder + "/") for p in self.files) else {})

    def download_file(self, path, local_path=None, return_bytes=True,
                      overwrite=False):
        self.downloads.append(path)
        key = self._norm(path)
        if key not in self.files:
            raise FileNotFoundError(f"Download failed: 404 {path}")
        with open(local_path, "wb") as fh:
            fh.write(self.files[key])
        return True

    def search_filenames(self, query, path=None):
        scope = self._norm(path or "").lower()
        return [{"name": p.rsplit("/", 1)[-1], "type": "file", "path": p}
                for p in self.files
                if query.lower() in p.rsplit("/", 1)[-1].lower()
                and p.lower().startswith(scope)]

    def upload_file(self, local, remote, overwrite=False):
        key = self._norm(remote)
        if key in self.files and not overwrite:
            return False
        with open(local, "rb") as fh:
            self.files[key] = fh.read()
        self.uploads.append((key, overwrite))
        return True

    def create_folder(self, path):
        return True

    def get_web_url(self, path):
        return f"https://sp.example/{self._norm(path)}"


@pytest.fixture
def site(monkeypatch, tmp_path):
    monkeypatch.setenv(sp.ENV_SITE, "https://t.sharepoint.com/sites/X")
    monkeypatch.setenv(sp.ENV_TOKEN, "tok")
    monkeypatch.setenv(sp.ENV_ROOT, ROOT)
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    fm = TreeFM({ALICE_FILE: b"%PDF alice", BOB_OLD: b"%PDF bob",
                 SHARED: b"%PDF shared", LEGACY: b"%PDF legacy"})
    monkeypatch.setattr(sp, "_STORE", sp.SharePointStore(file_manager=fm))
    spt._DOWNLOADS.clear()
    return fm


def _conversation(monkeypatch, tmp_path, person, *, owner=True,
                  multi_user=True, bind=True):
    """A conversation record laid out the way the app lays it out:
    ``<data>/users/<key>/document_review/conversations/<tid>`` on a
    multi-user host, ``<data>/conversations/<tid>`` otherwise."""
    data = tmp_path / "data"
    root = (data / "users" / f"corp__{person}" / "document_review"
            if multi_user else data)
    tid = f"{person}0001"
    rec = root / "conversations" / tid
    (rec / "files").mkdir(parents=True, exist_ok=True)
    meta = {"thread_id": tid, "title": f"{person.title()} review",
            "created": time.time()}
    if owner:
        meta.update(owner=person, page="document_review")
    (rec / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    core.register_thread_root(tid, str(root))
    if bind:
        monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(rec / "files"))
    tools = {t.name: t for t in
             spt.tools_if_configured(thread_id=tid, record_dir=str(rec))[0]}
    return tid, rec, tools


# -- A1: Bob cannot reach Alice ---------------------------------------------

def test_bobs_search_never_finds_alices_file(site, monkeypatch, tmp_path):
    _tid, _rec, bob = _conversation(monkeypatch, tmp_path, "bob")
    out = bob["sharepoint_search_files"].invoke({"query": "10.17a"})
    assert "alice" not in out.lower() and ALICE_FILE not in out
    # the single-user layout's old folders are nobody's to browse here either
    assert "legacy_10.17a" not in out
    assert "No files matching" in out and "private" in out
    assert site.downloads == []
    # his own past conversations and the shared folders are found
    assert "bob_notes.pdf" in bob["sharepoint_search_files"].invoke(
        {"query": "notes"})
    assert "shared_spec.pdf" in bob["sharepoint_search_files"].invoke(
        {"query": "shared_spec"})


def test_bobs_listing_of_the_conversations_shows_only_his_own(
        site, monkeypatch, tmp_path):
    _tid, _rec, bob = _conversation(monkeypatch, tmp_path, "bob")
    out = bob["sharepoint_list_files"].invoke({"path": "conversations"})
    assert "[folder] bob" in out
    assert "alice" not in out.lower() and "Legacy_chat" not in out
    assert "bob_notes.pdf" in bob["sharepoint_list_files"].invoke(
        {"path": "conversations/bob/document_review/"
                 "Old_bob_review_2026-10-01/files"})


@pytest.mark.parametrize("spelling", [
    "conversations/alice",
    "conversations/ALICE/document_review",
    f"{ROOT}/conversations/alice/document_review",
    "/sites/X/Shared Documents/General/GSE_app/conversations/alice",
    "Documents/General/GSE_app/conversations/alice",
    "GSE_app/conversations/alice.",
    "conversations/bob/../alice",
    "conversations/%61lice",
    ("https://t.sharepoint.com/sites/X/Shared%20Documents/Forms/AllItems.aspx"
     "?id=%2Fsites%2FX%2FShared%20Documents%2FGeneral%2FGSE_app%2F"
     "conversations%2Falice"),
])
def test_every_spelling_of_alices_folder_is_refused(site, monkeypatch,
                                                    tmp_path, spelling):
    _tid, _rec, bob = _conversation(monkeypatch, tmp_path, "bob")
    out = bob["sharepoint_list_files"].invoke({"path": spelling})
    assert "refused" in out and "private" in out
    assert "10.17a" not in out
    assert not any("alice" in str(p).lower() for p in site.listed)


def test_bob_cannot_download_from_or_upload_into_alices_folder(
        site, monkeypatch, tmp_path):
    _tid, rec, bob = _conversation(monkeypatch, tmp_path, "bob")
    out = bob["sharepoint_download_file"].invoke({"path": ALICE_FILE})
    assert "refused" in out and site.downloads == []
    assert not (rec / "files" / "10.17a.pdf").exists()
    # by name, the fallback search does not find hers either
    out = bob["sharepoint_download_file"].invoke({"path": "10.17a.pdf"})
    assert "not found" in out and "alice" not in out.lower()
    assert all("alice" not in d.lower() for d in site.downloads)
    (rec / "files" / "mine.pdf").write_bytes(b"%PDF mine")
    out = bob["sharepoint_upload_file"].invoke(
        {"local_path": "mine.pdf",
         "dest_folder": "conversations/alice/document_review"})
    assert "refused" in out
    assert not any("alice" in k for k, _ in site.uploads)
    # the conversations folder itself is no destination either
    out = bob["sharepoint_upload_file"].invoke(
        {"local_path": "mine.pdf", "dest_folder": "conversations"})
    assert "refused" in out


def test_alice_still_reaches_her_own_file(site, monkeypatch, tmp_path):
    _tid, rec, alice = _conversation(monkeypatch, tmp_path, "alice")
    assert "10.17a.pdf" in alice["sharepoint_search_files"].invoke(
        {"query": "10.17a"})
    out = alice["sharepoint_download_file"].invoke({"path": ALICE_FILE})
    assert out.startswith("Downloaded") and \
        (rec / "files" / "10.17a.pdf").read_bytes() == b"%PDF alice"


def test_an_untagged_conversation_on_a_multi_user_host_fails_closed(
        site, monkeypatch, tmp_path):
    """The app records the owner at the start of the turn; if that write
    failed, the folder layout still says whose conversation this is."""
    _tid, _rec, bob = _conversation(monkeypatch, tmp_path, "bob", owner=False)
    out = bob["sharepoint_search_files"].invoke({"query": "10.17a"})
    assert "alice" not in out.lower()


def test_a_single_user_host_is_unchanged(site, monkeypatch, tmp_path):
    """Nobody identified, DEV_IDENTITY or the Databricks launcher's email:
    no owner is recorded and every folder is reachable as before."""
    _tid, _rec, me = _conversation(monkeypatch, tmp_path, "me", owner=False,
                                   multi_user=False)
    out = me["sharepoint_search_files"].invoke({"query": "10.17a"})
    assert "10.17a.pdf" in out and "alice" in out
    assert "[folder] alice" in me["sharepoint_list_files"].invoke(
        {"path": "conversations"})


def test_the_unbound_tools_keep_the_old_rules(site):
    assert "10.17a.pdf" in spt.sharepoint_search_files.invoke(
        {"query": "10.17a"})


# -- A5: the mirrored copy, linked -------------------------------------------

def test_a_conversation_file_is_linked_where_the_mirror_keeps_it(
        site, monkeypatch, tmp_path):
    tid, rec, bob = _conversation(monkeypatch, tmp_path, "bob")
    report = rec / "files" / "report.pdf"
    report.write_bytes(b"%PDF v1")
    session = sp.get_store().session_folder(tid)
    out = bob["sharepoint_upload_file"].invoke({"local_path": "report.pdf"})
    remote = f"{session}/files/report.pdf"
    assert remote in out and "ONE copy" in out and "Link: https://" in out
    assert str(tmp_path) not in out                     # A6: no server path
    assert site.files[remote] == b"%PDF v1"
    names = [k.rsplit("/", 1)[-1] for k, _ in site.uploads]
    assert not any(n.startswith("report_") for n in names)   # no timestamp
    # Edited and asked again: the SAME file on SharePoint follows the edit.
    time.sleep(0.01)
    report.write_bytes(b"%PDF v2 longer")
    out2 = bob["sharepoint_upload_file"].invoke(
        {"local_path": "report.pdf", "dest_folder": f"{session}/files"})
    assert remote in out2 and site.files[remote] == b"%PDF v2 longer"
    assert not any(k.rsplit("/", 1)[-1].startswith("report_")
                   for k, _ in site.uploads)


def test_an_explicit_other_folder_keeps_the_timestamp_rule(
        site, monkeypatch, tmp_path):
    _tid, rec, bob = _conversation(monkeypatch, tmp_path, "bob")
    (rec / "files" / "spec.pdf").write_bytes(b"%PDF x")
    site.files[f"{ROOT}/deliverables/spec.pdf"] = b"older"
    out = bob["sharepoint_upload_file"].invoke(
        {"local_path": "spec.pdf", "dest_folder": "deliverables"})
    assert out.startswith("Uploaded 'spec.pdf' -> ")
    assert f"{ROOT}/deliverables/spec_" in out          # a new, stamped name
    assert str(tmp_path) not in out


def test_only_this_conversations_files_can_be_uploaded(
        site, monkeypatch, tmp_path):
    """A2 on the upload side: another person's file on the server is not
    this conversation's to publish."""
    _tid, _arec, _alice = _conversation(monkeypatch, tmp_path, "alice",
                                        bind=False)
    theirs = _arec / "files" / "alice_private.pdf"
    theirs.write_bytes(b"%PDF hers")
    _tid, rec, bob = _conversation(monkeypatch, tmp_path, "bob")
    out = bob["sharepoint_upload_file"].invoke(
        {"local_path": str(theirs), "dest_folder": "deliverables"})
    assert "Local file not found" in out and "only those" in out
    assert str(rec) not in out          # Bob's own folder is never spelled out
    assert site.uploads == []
