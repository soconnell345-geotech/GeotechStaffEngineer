"""Offline tests for the Foundry dataset store (BACKLOG INF-38).

A fake ``foundry.transforms`` module stands in for the workspace dataset
library: no network, and every alias, path and name below is made up. The
fake keeps what the AI FDE measured from a published app (2026-11-03):

* ``Dataset.get`` of an alias the workspace does not define fails with
  ``404 FoundryContainerSidecar:AliasNotDefined``;
* ``upload_directory`` refuses a folder outside the outputs folder
  (``IllegalArgumentError``), uploads EVERYTHING under it in one
  transaction, and returns ``{local path: dataset path}``, the dataset path
  being the path relative to the outputs folder;
* ``upload_file`` is a transaction of its own;
* a write while another transaction is open fails with
  ``400 Catalog:SimultaneousOpenTransactionsNotAllowed``;
* ``files()`` lists the whole dataset (no prefix filter, hidden files only
  when asked), and ``File.download()`` returns a local path;
* a later write overwrites a path and keeps every other file.
"""

import json
import os
import sys
import threading
import time
import types
from datetime import datetime, timezone

import pytest

import webapp.core as core
import webapp.foundry_dataset_store as fds
import webapp.sharepoint_store as sp

ALIAS = "made_up_conversations"
TOP = fds.DEFAULT_ROOT

_ENVS = (fds.ENV_DATASET, "GEOTECH_DEPLOYMENT", "GEOTECH_FOUNDRY_TOKEN",
         "FOUNDRY_TOKEN", "GEOTECH_FOUNDRY_HOST", "FOUNDRY_HOSTNAME",
         "FOUNDRY_URL",
         sp.ENV_SITE, sp.ENV_CLIENT_ID, sp.ENV_CLIENT_SECRET, sp.ENV_TOKEN,
         sp.ENV_TOKEN_FILE, sp.ENV_DRIVE, sp.ENV_ROOT)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for e in _ENVS:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setattr(sp, "_graph_configured", lambda: False)
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(fds.DatasetFileManager, "_warned_foreign", False)
    yield


# ---------------------------------------------------------------------------
# the fake workspace dataset library
# ---------------------------------------------------------------------------

class IllegalArgumentError(Exception):
    pass


class SidecarError(Exception):
    """The library's refusals carry the status and the error name."""


class FakeFile:
    def __init__(self, ds, path, data):
        self.path = path
        self.size_bytes = len(data)
        self.updated_time = datetime(2026, 1, 2, tzinfo=timezone.utc)
        self._ds = ds

    def download(self):
        return self._ds.download(self.path)


class FakeDataset:
    """One dataset's files, with the one-open-transaction rule."""

    def __init__(self, lib):
        self.lib = lib
        self.files_ = {}             # dataset path -> bytes
        self.calls = []
        self.busy = 0                # writes refused before one goes through
        self.open = False
        self.download_delay = 0.0
        self.gate = None             # an Event a write waits on
        self.active = 0
        self.peak = 0
        self.n = 0
        self._lock = threading.Lock()

    def _begin(self, what):
        self.calls.append((what,))
        if self.busy:
            self.busy -= 1
            raise SidecarError("400 Catalog:SimultaneousOpenTransactionsNotAllowed"
                               " {'openTransactionRid': 'ri.made-up.1'}")
        if self.open:
            raise SidecarError("400 Catalog:SimultaneousOpenTransactionsNotAllowed")
        self.open = True
        if self.gate is not None:
            self.gate.wait(5)

    def upload_file(self, path, logical_path=None):
        self._begin("upload_file")
        try:
            logical = (logical_path or os.path.basename(path)).lstrip("/")
            with open(path, "rb") as fh:
                self.files_[logical] = fh.read()
            self.calls[-1] = ("upload_file", logical)
            return logical
        finally:
            self.open = False

    def upload_directory(self, local_dir_path):
        out = os.path.normpath(self.lib.outputs)
        given = os.path.normpath(os.path.abspath(local_dir_path))
        if given != out and not given.startswith(out + os.sep):
            raise IllegalArgumentError(
                f"Provided path {local_dir_path} should be in {out}")
        self._begin("upload_directory")
        try:
            result = {}
            for dirpath, _dirs, names in os.walk(given):
                for name in names:
                    local = os.path.join(dirpath, name)
                    logical = os.path.relpath(local, out).replace(os.sep, "/")
                    with open(local, "rb") as fh:
                        self.files_[logical] = fh.read()
                    result[local] = logical
            self.calls[-1] = ("upload_directory", sorted(result.values()))
            return result
        finally:
            self.open = False

    def files(self, mode="current", show_hidden_files=False):
        self.calls.append(("files",))
        return iter([FakeFile(self, p, b)
                     for p, b in sorted(self.files_.items())
                     if show_hidden_files or not any(
                         s.startswith(("_", ".")) for s in p.split("/"))])

    def download(self, path):
        with self._lock:
            self.calls.append(("download", path))
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.n += 1
            n = self.n
        try:
            if self.download_delay:
                time.sleep(self.download_delay)
            if path not in self.files_:
                raise SidecarError(f"404 FileNotFound {path}")
            dst = os.path.join(self.lib.cache, f"{n}_{path.rsplit('/', 1)[-1]}")
            with open(dst, "wb") as fh:
                fh.write(self.files_[path])
            return dst
        finally:
            with self._lock:
                self.active -= 1

    def count(self, kind):
        return sum(1 for c in self.calls if c[0] == kind)

    def uploads(self):
        return [c for c in self.calls
                if c[0] in ("upload_file", "upload_directory") and len(c) > 1]


class FakeLibrary:
    """``foundry.transforms`` with one dataset under :data:`ALIAS`."""

    def __init__(self, tmp_path):
        self.outputs = str(tmp_path / "foundry" / "outputs")
        self.cache = str(tmp_path / "library_cache")
        os.makedirs(self.cache, exist_ok=True)
        self.ds = FakeDataset(self)
        lib = self

        class Dataset:
            @staticmethod
            def get(alias):
                lib.ds.calls.append(("get", alias))
                if alias != ALIAS:
                    raise SidecarError("404 FoundryContainerSidecar:"
                                       f"AliasNotDefined {{'alias': '{alias}'}}")
                return lib.ds

        pkg = types.ModuleType("foundry")
        pkg.__path__ = []
        mod = types.ModuleType("foundry.transforms")
        mod.Dataset = Dataset
        pkg.transforms = mod
        self.modules = {"foundry": pkg, "foundry.transforms": mod}

    def install(self, monkeypatch):
        for name, module in self.modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        monkeypatch.setattr(fds, "OUTPUTS_DIR", self.outputs)
        return self

    def leftovers(self):
        """Every file left in the outputs folder."""
        return sorted(os.path.relpath(os.path.join(d, n), self.outputs)
                      .replace(os.sep, "/")
                      for d, _s, names in os.walk(self.outputs)
                      for n in names)


@pytest.fixture
def lib(monkeypatch, tmp_path):
    return FakeLibrary(tmp_path).install(monkeypatch)


def _fm(lib, pauses=None, **kw):
    sleep = pauses.append if pauses is not None else (lambda s: None)
    return fds.DatasetFileManager(ALIAS, outputs_dir=lib.outputs,
                                  sleep=sleep, **kw)


def _conversation(root, tid, title, created, extra=0):
    core.ensure_conversation(tid, title=title, root=root)
    meta = core.load_meta(tid, root)
    meta["created"] = created
    core.save_meta(tid, meta, root)
    conv = core.conversation_dir(tid, root)
    with open(os.path.join(conv, "transcript.jsonl"), "w",
              encoding="utf-8") as fh:
        fh.write(json.dumps({"role": "user", "text": "hello"}) + "\n")
    files = os.path.join(conv, "files")
    os.makedirs(files, exist_ok=True)
    with open(os.path.join(files, "calc.pdf"), "wb") as fh:
        fh.write(b"%PDF " + b"x" * 3000)
    with open(os.path.join(files, "figure.png"), "wb") as fh:
        fh.write(b"\x89PNG" + b"p" * 200)
    for i in range(extra):
        with open(os.path.join(files, f"extra_{i}.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write(f"made-up note {i}")
    return conv


_CREATED = time.mktime((2026, 1, 2, 12, 0, 0, 0, 0, -1))


def _write(path, data=b"x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


# ---------------------------------------------------------------------------
# the file manager alone
# ---------------------------------------------------------------------------

def test_ls_filters_the_whole_listing_and_makes_folders(lib):
    lib.ds.files_ = {f"{TOP}/conversations/Alpha_2026-01-02/meta.json": b"{}",
                     f"{TOP}/conversations/Alpha_2026-01-02/files/a.pdf": b"%",
                     f"{TOP}/conversations/Beta_2026-01-01/meta.json": b"{}",
                     f"{TOP}/conversations/stray.txt": b"x",
                     f"{TOP}/other/z.txt": b"z",
                     "someone_else/file.txt": b"y"}
    fm = _fm(lib)
    top = {(e["name"], e["type"]) for e in fm.ls(f"{TOP}/conversations")}
    assert top == {("Alpha_2026-01-02", "folder"),
                   ("Beta_2026-01-01", "folder"), ("stray.txt", "file")}
    alpha = {(e["name"], e["type"], e["path"]) for e in
             fm.ls(f"/{TOP}/conversations/Alpha_2026-01-02/")}
    assert alpha == {
        ("meta.json", "file", f"{TOP}/conversations/Alpha_2026-01-02/meta.json"),
        ("files", "folder", f"{TOP}/conversations/Alpha_2026-01-02/files")}
    assert fm.ls(f"{TOP}/conversations/Nope") == []
    # outside a batch each listing reads the whole dataset once
    assert lib.ds.count("files") == 3 and lib.ds.count("get") == 1
    deep = [e["path"] for e in fm.ls(f"{TOP}/conversations", recursive=True,
                                     type="files")]
    assert len(deep) == 4 and f"{TOP}/other/z.txt" not in deep
    assert fm.create_folder(f"{TOP}/x/y")["path"] == f"{TOP}/x/y"   # no-op
    assert fm.get_web_url(f"{TOP}/conversations") == ""
    assert not lib.ds.uploads()


def test_hidden_files_are_listed_too(lib):
    """A name starting ``_`` or ``.`` is hidden by default in a listing; the
    app asks for hidden files, so a restore misses none."""
    lib.ds.files_ = {f"{TOP}/c/T/files/_made_up.txt": b"h",
                     f"{TOP}/c/T/meta.json": b"{}"}
    names = {e["name"] for e in _fm(lib).ls(f"{TOP}/c/T", recursive=True,
                                            type="files")}
    assert names == {"_made_up.txt", "meta.json"}


def test_a_single_upload_is_one_transaction_and_keeps_other_files(lib,
                                                                  tmp_path):
    fm = _fm(lib)
    note = _write(str(tmp_path / "note.txt"), b"first")
    other = _write(str(tmp_path / "other.txt"), b"other")
    assert fm.upload_file(note, f"{TOP}/n/note.txt", overwrite=True)
    assert fm.upload_file(other, f"{TOP}/n/other.txt", overwrite=True)
    _write(note, b"second")
    assert fm.upload_file(note, f"{TOP}/n/note.txt", overwrite=True)
    assert lib.ds.files_ == {f"{TOP}/n/note.txt": b"second",
                             f"{TOP}/n/other.txt": b"other"}
    assert lib.ds.count("upload_file") == 3
    assert lib.ds.count("upload_directory") == 0
    assert fm.download_file(f"{TOP}/n/note.txt") == b"second"
    out = tmp_path / "back" / "note.txt"
    assert fm.download_file(f"{TOP}/n/note.txt", local_path=str(out),
                            return_bytes=False, overwrite=True) is True
    assert out.read_bytes() == b"second"
    with pytest.raises(ValueError):
        fm.upload_file(note, f"{TOP}/../escape.txt")


def test_a_batch_is_one_upload_directory_and_clears_what_it_staged(lib,
                                                                   tmp_path):
    fm = _fm(lib)
    fm.begin_batch()
    assert fm.end_batch() == 0
    assert lib.ds.calls == []                  # nothing changed, no save
    fm.begin_batch()
    paths = []
    for i in range(3):
        f = _write(str(tmp_path / f"f{i}.txt"), b"x" * (i + 1))
        paths.append(f"{TOP}/c/T/f{i}.txt")
        fm.upload_file(f, paths[-1], overwrite=True)
    # a temporary file gone before the save still goes (it was copied)
    gone = _write(str(tmp_path / "MOVED.txt"), b"moved")
    fm.upload_file(gone, f"{TOP}/c/Old/MOVED.txt", overwrite=True)
    os.remove(gone)
    assert lib.ds.files_ == {}                 # nothing written yet
    assert fm.end_batch() == 4
    assert lib.ds.uploads() == [("upload_directory",
                                 sorted(paths + [f"{TOP}/c/Old/MOVED.txt"]))]
    assert lib.ds.files_[f"{TOP}/c/Old/MOVED.txt"] == b"moved"
    assert lib.leftovers() == []
    assert os.listdir(lib.outputs) == []       # the folders it made, too


def test_upload_directory_refuses_a_folder_outside_the_outputs_folder(
        lib, tmp_path):
    """The library takes only the outputs folder; staged elsewhere, the
    save fails (and is cleared) rather than writing anything."""
    elsewhere = str(tmp_path / "not_outputs")
    fm = fds.DatasetFileManager(ALIAS, outputs_dir=elsewhere,
                                sleep=lambda s: None)
    fm.begin_batch()
    fm.upload_file(_write(str(tmp_path / "a.txt")), f"{TOP}/a.txt")
    with pytest.raises(IllegalArgumentError):
        fm.end_batch()
    assert lib.ds.files_ == {}
    assert [os.path.join(d, n) for d, _s, ns in os.walk(elsewhere)
            for n in ns] == []


def test_a_foreign_file_in_outputs_is_never_uploaded_or_deleted(lib,
                                                                tmp_path):
    foreign = _write(os.path.join(lib.outputs, "made_up_other_job",
                                  "result.csv"), b"theirs")
    fm = _fm(lib)
    for round_ in range(2):
        fm.begin_batch()
        fm.upload_file(_write(str(tmp_path / "a.txt"), b"a"), f"{TOP}/c/a.txt")
        fm.upload_file(_write(str(tmp_path / "b.txt"), b"b"), f"{TOP}/c/b.txt")
        assert fm.end_batch() == 2
        note = fm.take_note()
        if round_ == 0:
            assert note and "file by file" in note
        else:
            assert note is None                # said once per process
    assert lib.ds.count("upload_directory") == 0
    assert lib.ds.count("upload_file") == 4    # one transaction per file
    assert set(lib.ds.files_) == {f"{TOP}/c/a.txt", f"{TOP}/c/b.txt"}
    assert os.path.isfile(foreign)
    assert lib.leftovers() == ["made_up_other_job/result.csv"]


def test_the_apps_own_leftover_in_outputs_is_cleared_not_sent(lib, tmp_path):
    """Anything under the app's own folder in the outputs folder is left
    from a save that stopped: cleared, never uploaded."""
    _write(os.path.join(lib.outputs, TOP, "conversations", "Old_2026-01-01",
                        "stale.json"), b"old")
    fm = _fm(lib)
    fm.begin_batch()
    fm.upload_file(_write(str(tmp_path / "a.txt")), f"{TOP}/c/a.txt")
    fm.end_batch()
    assert lib.ds.uploads() == [("upload_directory", [f"{TOP}/c/a.txt"])]
    assert lib.leftovers() == []


def test_a_busy_dataset_is_tried_again_with_a_growing_pause(lib, tmp_path):
    lib.ds.busy = 2
    pauses = []
    fm = _fm(lib, pauses)
    fm.begin_batch()
    fm.upload_file(_write(str(tmp_path / "a.txt")), f"{TOP}/a.txt")
    assert fm.end_batch() == 1
    assert pauses == [1.0, 2.0] and lib.ds.count("upload_directory") == 3
    assert lib.ds.files_ == {f"{TOP}/a.txt": b"x"}
    assert lib.leftovers() == []


def test_a_dataset_busy_for_the_whole_wait_gives_up_plainly(lib, tmp_path):
    lib.ds.busy = 99
    pauses = []
    fm = _fm(lib, pauses)
    fm.begin_batch()
    fm.upload_file(_write(str(tmp_path / "a.txt")), f"{TOP}/a.txt")
    with pytest.raises(fds.StorageBusy) as err:
        fm.end_batch()
    assert "another save holds the conversation dataset" in str(err.value)
    assert sum(pauses) == pytest.approx(30.0)
    assert lib.ds.files_ == {} and lib.leftovers() == []


def test_an_alias_the_workspace_does_not_define_says_how_to_import(lib):
    fm = fds.DatasetFileManager("made_up_missing", outputs_dir=lib.outputs)
    with pytest.raises(fds.AliasNotDefined) as err:
        fm.ls(TOP)
    text = str(err.value)
    assert "made_up_missing" in text and "import the dataset into the " \
        "workspace (Data panel, read and write), push" in text


def test_a_restore_downloads_in_parallel_through_one_listing(lib, tmp_path):
    lib.ds.download_delay = 0.05
    for i in range(14):
        lib.ds.files_[f"{TOP}/c/T/files/f{i:02d}.txt"] = f"n{i}".encode()
    fm = _fm(lib)
    fm.begin_batch()
    pairs = [(f"{TOP}/c/T/files/f{i:02d}.txt",
              str(tmp_path / "back" / f"f{i:02d}.txt")) for i in range(14)]
    errors = fm.download_files(pairs + [(f"{TOP}/c/T/missing.txt",
                                         str(tmp_path / "back" / "m.txt"))])
    fm.end_batch(commit=False)
    assert errors[:14] == [None] * 14
    assert isinstance(errors[14], FileNotFoundError)
    assert 1 < lib.ds.peak <= fds.PARALLEL_DOWNLOADS
    assert lib.ds.count("files") == 1
    for i in range(14):
        assert (tmp_path / "back" / f"f{i:02d}.txt").read_bytes() == \
            f"n{i}".encode()


# ---------------------------------------------------------------------------
# through the store: mirror, list, restore
# ---------------------------------------------------------------------------

def _store(lib, pauses=None, **kw):
    return sp.SharePointStore(file_manager=_fm(lib, pauses, **kw))


def test_the_store_names_foundry_storage_and_its_own_root(lib):
    store = _store(lib)
    assert store.backend == "foundry" and store.label == "Foundry storage"
    assert store.background is True
    assert store.root() == TOP
    assert store.conversations_base() == f"{TOP}/conversations"
    assert store.conversations_base("Jane Doe") == \
        f"{TOP}/conversations/Jane_Doe"


def test_a_mirror_is_one_save_and_overwrites_only_what_changed(lib,
                                                               tmp_path):
    store = _store(lib)
    root = str(tmp_path / "container")
    conv = _conversation(root, "T-ONE", "Footing check", _CREATED)
    up = store.mirror_conversation("T-ONE", root=root)
    assert not up["errors"], up
    assert up["uploaded"] == 4 and up["web_url"] == ""
    assert up["folder"] == f"{TOP}/conversations/Footing_check_2026-01-02"
    assert lib.ds.count("upload_directory") == 1
    assert lib.ds.count("files") == 1          # the folder-name check, once
    assert sorted(p.rsplit("/", 1)[-1] for p in lib.ds.files_) == [
        "calc.pdf", "figure.png", "meta.json", "transcript.jsonl"]
    assert not any(sp.MANIFEST_NAME in p for p in lib.ds.files_)
    assert lib.leftovers() == []

    # nothing changed: no call at all
    calls = len(lib.ds.calls)
    again = store.mirror_conversation("T-ONE", root=root)
    assert again["uploaded"] == 0 and again["skipped"] == 4
    assert len(lib.ds.calls) == calls

    # one file changed: it alone, overwritten; the others kept
    pdf = os.path.join(conv, "files", "calc.pdf")
    _write(pdf, b"%PDF revised")
    os.utime(pdf, (time.time() + 5, time.time() + 5))
    third = store.mirror_conversation("T-ONE", root=root)
    assert third["uploaded"] == 1 and not third["errors"], third
    assert lib.ds.uploads()[-1] == ("upload_directory",
                                    [f"{up['folder']}/files/calc.pdf"])
    assert lib.ds.files_[f"{up['folder']}/files/calc.pdf"] == b"%PDF revised"
    assert len(lib.ds.files_) == 4


def test_the_restore_round_trip(lib, tmp_path):
    store = _store(lib)
    before = str(tmp_path / "before_rebuild")
    after = str(tmp_path / "after_rebuild")
    conv_a = _conversation(before, "T-RT", "Wall design", _CREATED, extra=9)
    assert not store.mirror_conversation("T-RT", root=before)["errors"]

    listing = store.list_remote_conversations()
    assert [r["name"] for r in listing] == ["Wall_design_2026-01-02"]
    lib.ds.download_delay = 0.03
    lists_before = lib.ds.count("files")
    res = store.restore_conversation("Wall_design_2026-01-02", root=after)
    assert res["status"] == "restored" and res["thread_id"] == "T-RT", res
    assert res["downloaded"] == 13 and not res["errors"]
    assert lib.ds.count("files") - lists_before == 1     # listed once
    assert 1 < lib.ds.peak <= fds.PARALLEL_DOWNLOADS      # in parallel
    conv_b = core.conversation_dir("T-RT", after)
    for rel in ["meta.json", "transcript.jsonl", "files/calc.pdf",
                "files/figure.png"] + [f"files/extra_{i}.txt"
                                       for i in range(9)]:
        with open(os.path.join(conv_a, *rel.split("/")), "rb") as a, \
                open(os.path.join(conv_b, *rel.split("/")), "rb") as b:
            assert a.read() == b.read(), rel
    assert [m["thread_id"] for m in core.list_conversations(after)] == ["T-RT"]
    # the restored manifest is complete: the next sync sends nothing
    again = store.mirror_conversation("T-RT", root=after)
    assert again["uploaded"] == 0 and not again["errors"], again


def test_a_restore_never_writes_outside_the_conversation_folder(lib,
                                                                tmp_path):
    """A dataset's file paths are free text: one with a ``..`` segment
    (either slash) is skipped with one warning, the rest restored."""
    store = _store(lib)
    before = str(tmp_path / "before_rebuild")
    after = str(tmp_path / "after_rebuild")
    _conversation(before, "T-PATH", "Wall design", _CREATED)
    assert not store.mirror_conversation("T-PATH", root=before)["errors"]
    base = f"{TOP}/conversations/Wall_design_2026-01-02"
    lib.ds.files_[f"{base}/../../made_up_escape.txt"] = b"x"
    lib.ds.files_[f"{base}/files/..\\..\\..\\made_up_escape2.txt"] = b"y"
    res = store.restore_conversation("Wall_design_2026-01-02", root=after)
    assert res["status"] == "restored" and res["downloaded"] == 4, res
    assert len(res["errors"]) == 1, res
    assert res["errors"][0].startswith("2 file(s) not restored")
    found = [n for _d, _s, names in os.walk(tmp_path) for n in names
             if "made_up_escape" in n]
    assert found == []
    assert sp._local_path_inside(after, "/etc/made_up") is None
    assert sp._local_path_inside(after, "C:/made_up") is None
    assert sp._local_path_inside(after, "files/a.pdf") == os.path.join(
        after, "files", "a.pdf")


def test_the_record_files_the_trace_reviews_read_are_carried(lib, tmp_path,
                                                            monkeypatch):
    """Foundry wiring audit row 8: activity.jsonl, FEEDBACK.md (with
    feedback.jsonl) and trace.jsonl, written by the app's own writers, go
    to the dataset with the turn and come back with a restore."""
    from webapp import activity_log, feedback
    monkeypatch.delenv("GEOTECH_TRACE", raising=False)
    store = _store(lib)
    root = str(tmp_path / "before")
    conv = _conversation(root, "T-REC", "Record check", _CREATED)
    assert activity_log.append_event(conv, "page_error", turn=1,
                                     detail="made-up event")
    feedback.record(conv, source="user", kind="user_feedback",
                    summary="made-up note")
    core.write_turn_trace("T-REC", {"turn_tokens": 1, "tools": []}, root)
    up = store.mirror_conversation("T-REC", root=root)
    assert not up["errors"], up
    names = {p[len(up["folder"]) + 1:] for p in lib.ds.files_}
    for rel in ("activity.jsonl", "FEEDBACK.md", "feedback.jsonl",
                "trace.jsonl"):
        assert rel in names, (rel, names)
        with open(os.path.join(conv, rel), "rb") as fh:
            assert lib.ds.files_[f"{up['folder']}/{rel}"] == fh.read(), rel
    res = store.restore_conversation(up["folder"].rsplit("/", 1)[-1],
                                     root=str(tmp_path / "after"))
    assert res["status"] == "restored", res
    back = core.conversation_dir("T-REC", str(tmp_path / "after"))
    for rel in ("activity.jsonl", "FEEDBACK.md", "trace.jsonl"):
        assert os.path.isfile(os.path.join(back, rel)), rel


def test_each_owner_lists_and_restores_only_their_folder(lib, tmp_path):
    store = _store(lib)
    root = str(tmp_path / "users")
    _conversation(root, "T-A", "Pile cap", _CREATED)
    core.tag_conversation("T-A", root=root, owner="corp__ann")
    _conversation(root, "T-B", "Slope", _CREATED)
    core.tag_conversation("T-B", root=root, owner="corp__bob")
    assert not store.mirror_conversation("T-A", root=root)["errors"]
    assert not store.mirror_conversation("T-B", root=root)["errors"]
    ann = store.list_remote_conversations(owner="corp__ann")
    assert [r["name"] for r in ann] == ["Pile_cap_2026-01-02"]
    assert [r["name"] for r in store.list_remote_conversations(
        owner="corp__bob")] == ["Slope_2026-01-02"]
    bad = store.restore_conversation("Slope_2026-01-02",
                                     root=str(tmp_path / "x"),
                                     owner="corp__ann")
    assert bad["status"] == "error"


def test_a_busy_save_never_breaks_the_turn_and_goes_next_time(lib, tmp_path):
    lib.ds.busy = 99
    pauses = []
    store = _store(lib, pauses)
    root = str(tmp_path)
    _conversation(root, "T-F", "Bearing", _CREATED)
    up = store.mirror_conversation("T-F", root=root)    # never raises
    assert up["uploaded"] == 0 and len(up["errors"]) == 1, up
    assert "4 file(s) not saved: StorageBusy: another save holds" in \
        up["errors"][0]
    assert lib.ds.files_ == {} and lib.leftovers() == []
    lib.ds.busy = 0                                      # the other save ends
    up2 = store.mirror_conversation("T-F", root=root)
    assert up2["uploaded"] == 4 and not up2["errors"], up2


def test_the_foreign_file_note_reaches_the_sync_summary_once(lib, tmp_path):
    _write(os.path.join(lib.outputs, "made_up_other_job.csv"), b"theirs")
    store = _store(lib)
    root = str(tmp_path)
    conv = _conversation(root, "T-N", "Footing", _CREATED)
    up = store.mirror_conversation("T-N", root=root)
    assert not up["errors"] and "file by file" in up["note"], up
    _write(os.path.join(conv, "files", "new.txt"), b"n")
    assert "note" not in store.mirror_conversation("T-N", root=root)


# ---------------------------------------------------------------------------
# saving after a turn, in the background
# ---------------------------------------------------------------------------

def test_saves_after_a_turn_run_in_the_background_in_order(lib):
    store = _store(lib)
    gate = threading.Event()
    order = []

    def mirror(thread_id, root=None):
        order.append(thread_id)
        if len(order) == 1:
            gate.wait(5)
        return {"uploaded": 0, "skipped": 0, "errors": [], "thread_id":
                thread_id, "duration_s": 0.0}
    store.mirror_conversation = mirror
    assert sp.save_after_change(store, "T-A") is None     # queued
    for _ in range(100):
        if store.save_state("T-A") == "saving":
            break
        time.sleep(0.01)
    assert store.save_state("T-A") == "saving"
    sp.save_after_change(store, "T-B")
    sp.save_after_change(store, "T-B")         # already waiting: once
    sp.save_after_change(store, "T-A")         # saving now: again after
    assert store.save_state("T-B") == "waiting"
    assert store.save_state("T-C") is None
    gate.set()
    assert store.flush(5)
    assert order == ["T-A", "T-B", "T-A"]
    assert store.save_state("T-A") is None and store.save_state("T-B") is None


def test_a_queued_save_outlives_the_turn_that_asked(lib, tmp_path):
    """The turn's worker queues the save and ends; the save still runs, and
    the sidebar reads its record from the store."""
    store = _store(lib)
    root = str(tmp_path)
    _conversation(root, "T-BG", "Pile cap", _CREATED)
    core.register_thread_root("T-BG", root)        # as the app does
    try:
        lib.ds.gate = threading.Event()
        results = []
        turn = threading.Thread(target=lambda: results.append(
            sp.save_after_change(store, "T-BG")))
        turn.start()
        turn.join()
        assert results == [None] and not turn.is_alive()
        assert lib.ds.files_ == {}
        lib.ds.gate.set()
        assert store.flush(10)
    finally:
        core.register_thread_root("T-BG", None)
    shown = sp.sync_for_conversation(store, "T-BG", None)
    assert shown and shown["uploaded"] == 4 and not shown["errors"], shown
    assert any(p.endswith("/transcript.jsonl") for p in lib.ds.files_)


def test_the_stores_newer_record_wins_over_an_older_session_record(lib):
    store = _store(lib)
    old = {"uploaded": 1, "skipped": 0, "errors": [], "thread_id": "T-S"}
    new = {"uploaded": 2, "skipped": 1, "errors": [], "thread_id": "T-S"}
    store._syncs["T-S"] = new
    assert sp.sync_for_conversation(store, "T-S", old) is new
    assert sp.sync_for_conversation(store, "T-OTHER", old) is None
    assert sp.sync_for_conversation(object(), "T-S", old) is old


# ---------------------------------------------------------------------------
# configuration: Foundry selects the dataset; everywhere else is unchanged
# ---------------------------------------------------------------------------

def _foundry(monkeypatch, alias=ALIAS):
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    monkeypatch.setenv(fds.ENV_DATASET, alias)


def test_on_foundry_the_dataset_is_the_permanent_storage(lib, monkeypatch,
                                                         tmp_path):
    _foundry(monkeypatch)
    monkeypatch.setenv(sp.ENV_SITE, "https://example.sharepoint.test/x")
    monkeypatch.setenv(sp.ENV_TOKEN, "made-up")
    store = sp.SharePointStore()
    assert store.configured and store.backend == "foundry"
    assert store.label == "Foundry storage" and store.background
    assert store.not_kept_reason() is None          # the alias works
    # the agent's SharePoint tools never run on the dataset
    assert sp.configured() is False
    fm = store.file_manager()
    assert isinstance(fm, fds.DatasetFileManager)
    assert fm.alias == ALIAS and fm.outputs_dir == lib.outputs
    root = str(tmp_path)
    _conversation(root, "T-E2E", "Lateral pile", _CREATED)
    assert not store.mirror_conversation("T-E2E", root=root)["errors"]
    assert any(p.startswith(f"{TOP}/conversations/") for p in lib.ds.files_)


def test_a_resource_id_is_not_an_alias(lib, monkeypatch, tmp_path):
    _foundry(monkeypatch, "ri.foundry.main.dataset.made-up-0000")
    reason = fds.unconfigured_reason()
    assert "resource ID" in reason and "import the dataset into the " \
        "workspace (Data panel, read and write), push" in reason
    store = sp.SharePointStore()
    assert not store.configured
    assert store.not_kept_reason() == f"{fds.ENV_DATASET} is set, but {reason}."
    root = str(tmp_path)
    _conversation(root, "T-RI", "Pavement", _CREATED)
    up = store.mirror_conversation("T-RI", root=root)    # never raises
    assert up["errors"][0].startswith("Foundry storage client:")
    assert lib.ds.calls == []


def test_the_library_missing_turns_storage_off_with_a_plain_reason(
        monkeypatch, tmp_path):
    _foundry(monkeypatch)
    monkeypatch.setitem(sys.modules, "foundry", None)       # import fails
    monkeypatch.setitem(sys.modules, "foundry.transforms", None)
    assert not fds.library_installed()
    store = sp.SharePointStore()
    assert not store.configured
    assert store.backend == "foundry" and store.label == "Foundry storage"
    reason = fds.unconfigured_reason()
    assert "foundry.transforms" in reason and "Code Workspace" in reason
    assert store.not_kept_reason() == f"{fds.ENV_DATASET} is set, but {reason}."
    assert store.list_remote_conversations() == []
    res = store.restore_conversation("anything_2026-01-02", root=str(tmp_path))
    assert res["status"] == "error" and "not configured" in res["errors"][0]


def test_funhouse_and_tiny_apps_are_unchanged(monkeypatch, tmp_path):
    # the dataset setting alone, off Foundry, changes nothing
    monkeypatch.setenv(fds.ENV_DATASET, ALIAS)
    for deployment in (None, "tinyapps"):
        if deployment:
            monkeypatch.setenv("GEOTECH_DEPLOYMENT", deployment)
        assert not fds.selected()
        store = sp.SharePointStore()
        assert not store.configured and not sp.configured()
        assert store.backend == "sharepoint" and store.label == "SharePoint"
        assert store.background is False and store.not_kept_reason() is None
        assert store.root() == sp.DEFAULT_ROOT
        monkeypatch.setenv(sp.ENV_SITE, "https://example.sharepoint.test/x")
        monkeypatch.setenv(sp.ENV_TOKEN, "made-up")
        assert sp.configured() and sp.SharePointStore().configured
        assert sp.SharePointStore().backend == "sharepoint"
        monkeypatch.delenv(sp.ENV_SITE)
        monkeypatch.delenv(sp.ENV_TOKEN)
    # a SharePoint file manager has no batch hook and is driven as before,
    # and a save after a turn is made at once, its summary returned
    assert sp._begin_batch(object()) is False

    class _PlainStore:
        background = False

        def mirror_conversation(self, thread_id):
            return {"thread_id": thread_id, "uploaded": 1}
    assert sp.save_after_change(_PlainStore(), "T-P") == {
        "thread_id": "T-P", "uploaded": 1}


class _SidebarStore:
    """The parts of the store the sidebar uses, as the dataset's."""

    backend = "foundry"
    label = "Foundry storage"
    last_sync = None

    def __init__(self, configured=True):
        self.configured = configured

    def list_remote_conversations(self, owner=None, page=None):
        return []

    def mirror_conversation(self, thread_id, root=None):
        return {"uploaded": 0, "skipped": 0, "errors": [], "duration_s": 0.0}


def _sidebar(monkeypatch, tmp_path, store, conversations=0):
    """The real shell (AppTest) over ``store``, the geotech page, with
    ``conversations`` saved ones already in its folder; run once."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import webapp.engine_config as engine_config
    from webapp import profiles
    from webapp.engine_config import EngineResolution

    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    for e in ("GEOTECH_APP_PROFILE", "DEV_IDENTITY", "GEOTECH_USER_EMAIL"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("GEOTECH_UPLOAD_MODE", "http")
    monkeypatch.setattr(engine_config, "resolve_engine",
                        lambda *a, **k: EngineResolution(
                            model=object(), source="prompter",
                            model_name="fake-model", message=""))
    monkeypatch.setattr(core, "build_agent", lambda *a, **k: object())
    monkeypatch.setattr(sp, "get_store", lambda *a, **k: store)
    monkeypatch.setenv(profiles.PROFILE_ENV, "geotech")
    for i in range(conversations):
        core.ensure_conversation(f"T-OLD{i}", title=f"Old question {i}",
                                 root=str(tmp_path))
    app = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "app.py")
    at = AppTest.from_file(app, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def test_the_sidebar_names_foundry_storage(monkeypatch, tmp_path):
    """The permanent-storage block over the dataset says Foundry storage,
    never SharePoint."""
    at = _sidebar(monkeypatch, tmp_path, _SidebarStore())
    captions = [str(c.value) for c in at.sidebar.caption]
    assert any(c.startswith("Foundry storage is on") for c in captions), \
        captions
    assert any("Foundry storage could not be listed" in c for c in captions)
    assert not any("SharePoint" in c for c in captions), captions


def test_the_sidebar_shows_a_save_in_the_background(monkeypatch, tmp_path):
    class _Saving(_SidebarStore):
        def save_state(self, thread_id):
            return "saving"

    at = _sidebar(monkeypatch, tmp_path, _Saving())
    captions = [str(c.value) for c in at.sidebar.caption]
    assert "Foundry storage: saving the latest turn…" in captions, captions
    assert not any(c.startswith("Foundry storage is on") for c in captions)


def _foundry_app(monkeypatch, tmp_path):
    """The real shell on a Foundry deployment, over the real store built
    from the settings (the geotech page); run once."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import webapp.engine_config as engine_config
    from webapp import profiles
    from webapp.engine_config import EngineResolution

    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "app"))
    for e in ("GEOTECH_APP_PROFILE", "DEV_IDENTITY", "GEOTECH_USER_EMAIL"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv(profiles.PROFILE_ENV, "geotech")
    monkeypatch.setattr(engine_config, "resolve_engine",
                        lambda *a, **k: EngineResolution(
                            model=object(), source="foundry",
                            model_name="fake-model", message=""))
    monkeypatch.setattr(core, "build_agent", lambda *a, **k: object())
    monkeypatch.setattr(sp, "_STORE", None)
    app = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "app.py")
    at = AppTest.from_file(app, default_timeout=30).run()
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("case", ["no alias", "resource ID",
                                  "library missing", "alias not defined"])
def test_on_foundry_the_sidebar_says_when_conversations_are_not_kept(
        lib, monkeypatch, tmp_path, case):
    """On master the sidebar drew no storage block when the dataset was not
    usable, and only Connection diagnostics said why: the owner would take
    conversations for kept. Now a short block says they are NOT kept, with
    the diagnostics line's own reason and fix."""
    from webapp import diagnostics as diag
    if case == "resource ID":
        monkeypatch.setenv(fds.ENV_DATASET, "ri.foundry.main.dataset.made-up-0")
    elif case == "library missing":
        monkeypatch.setenv(fds.ENV_DATASET, ALIAS)
        monkeypatch.setitem(sys.modules, "foundry.transforms", None)
    elif case == "alias not defined":
        monkeypatch.setenv(fds.ENV_DATASET, "made_up_missing")
    at = _foundry_app(monkeypatch, tmp_path)
    detail = diag._conversation_storage_check()["detail"]
    warnings = [str(w.value) for w in at.sidebar.warning]
    assert ("**Conversations are NOT kept across restarts.** " + detail) \
        in warnings, (warnings, detail)
    assert not any(str(c.value).startswith("Foundry storage is on")
                   for c in at.sidebar.caption)
    if case == "no alias":
        assert fds.ENV_DATASET in detail and "Download this conversation" \
            in detail
    else:
        assert "Data panel, read and write" in detail or \
            "Code Workspace" in detail, detail


def test_the_not_kept_block_is_foundry_only_and_absent_when_kept(
        lib, monkeypatch, tmp_path):
    monkeypatch.setenv(fds.ENV_DATASET, ALIAS)              # the alias works
    at = _foundry_app(monkeypatch, tmp_path / "kept")
    assert not any("NOT kept" in str(w.value) for w in at.sidebar.warning)
    assert any(str(c.value).startswith("Foundry storage is on")
               for c in at.sidebar.caption)
    # Funhouse with nothing configured: no storage block at all, as before
    monkeypatch.delenv("GEOTECH_DEPLOYMENT")
    monkeypatch.delenv(fds.ENV_DATASET)
    assert sp.SharePointStore().not_kept_reason() is None
    assert fds.not_kept_reason() is None


def test_an_alias_not_defined_is_learned_once_and_cleared_by_success(
        lib, monkeypatch):
    _foundry(monkeypatch, "made_up_missing")
    store = sp.SharePointStore()
    assert store.configured                     # nothing tells until a call
    first = store.not_kept_reason()
    assert "made_up_missing" in first and "(AliasNotDefined)" in first
    assert store.not_kept_reason() == first
    assert lib.ds.count("get") == 1             # checked once, not per draw
    # the dataset imported under that name: the next call works and clears it
    fm = store.file_manager()
    fm.alias = ALIAS
    fm.ls(TOP)
    assert fm.alias_error is None and store.not_kept_reason() is None


def test_a_refused_save_shows_the_foundry_storage_warning(monkeypatch,
                                                         tmp_path):
    """An alias the workspace does not define names no token or 401/403;
    the sidebar still gives the import-the-dataset warning (made-up error
    text in the library's shape)."""
    refused = ("4 file(s) not saved: AliasNotDefined: the workspace has no "
               "dataset alias 'made_up_missing': import the dataset ...")

    class _Refused(_SidebarStore):
        def last_sync_for(self, thread_id):
            return {"uploaded": 0, "skipped": 0, "errors": [refused],
                    "duration_s": 0.1, "thread_id": thread_id}

    at = _sidebar(monkeypatch, tmp_path, _Refused())
    warnings = [str(w.value) for w in at.sidebar.warning]
    assert any(w.startswith("Foundry storage could not reach the "
                            "conversation dataset") for w in warnings), warnings
    assert not any("stage_sharepoint" in w for w in warnings), warnings


@pytest.mark.parametrize("configured, expected", [
    (True, "**Find a past conversation** under Permanent storage below"),
    (False, "**Restore a downloaded conversation** below"),
])
def test_the_filters_no_match_line_points_at_a_block_drawn(
        monkeypatch, tmp_path, configured, expected):
    """Foundry wiring audit: the conversation filter's "no match" line
    pointed at Permanent storage, which Foundry did not draw. It now names
    the block that is there: the store's search, else the zip restore."""
    at = _sidebar(monkeypatch, tmp_path, _SidebarStore(configured),
                  conversations=9)
    at.sidebar.text_input(key="conv_filter").input("no such title").run()
    assert not at.exception, at.exception
    lines = [str(c.value) for c in at.sidebar.caption
             if str(c.value).startswith("No saved conversation matches")]
    assert len(lines) == 1 and expected in lines[0], lines
    if configured:
        assert at.sidebar.expander and any(
            "Find a past conversation" in str(e.label)
            for e in at.sidebar.expander)


# ---------------------------------------------------------------------------
# the app end to end: a turn saves; a fresh container lists and restores it
# ---------------------------------------------------------------------------

def test_a_turn_saves_and_a_fresh_container_restores_it(lib, monkeypatch,
                                                        tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import webapp.engine_config as engine_config
    from webapp import profiles
    from webapp.engine_config import EngineResolution
    from webapp.tests.test_app_smoke import _stream_ok

    _foundry(monkeypatch)
    for e in ("GEOTECH_APP_PROFILE", "DEV_IDENTITY", "GEOTECH_USER_EMAIL"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv(profiles.PROFILE_ENV, "geotech")
    monkeypatch.setattr(engine_config, "resolve_engine",
                        lambda *a, **k: EngineResolution(
                            model=object(), source="foundry",
                            model_name="fake-model", message=""))
    monkeypatch.setattr(core, "build_agent", lambda *a, **k: object())
    monkeypatch.setattr(core, "stream_turn", _stream_ok)
    app = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "app.py")

    # the first container: one turn, saved after it in the background
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "first"))
    monkeypatch.setattr(sp, "_STORE", None)
    at = AppTest.from_file(app, default_timeout=30).run()
    assert not at.exception, at.exception
    lib.ds.gate = threading.Event()            # the dataset write hangs
    at.chat_input[0].set_value("What is the made-up bearing capacity?").run()
    assert not at.exception, at.exception
    tid = at.session_state["thread_id"]
    # the turn ended, and the page drew, while the save was still open
    assert [e["role"] for e in at.session_state["transcript"]] == [
        "user", "assistant"]
    assert lib.ds.files_ == {}
    captions = [str(c.value) for c in at.sidebar.caption]
    assert any(c.startswith("Foundry storage: ") and "saved" in c
               or c == "Foundry storage: saving the latest turn…"
               for c in captions), captions
    lib.ds.gate.set()
    first_store = sp.get_store()
    assert first_store.background and first_store.flush(10)
    assert first_store.last_sync_for(tid)["errors"] == []
    saved = {p for p in lib.ds.files_ if p.startswith(f"{TOP}/conversations/")}
    folder = next(p for p in saved if p.endswith("/transcript.jsonl")) \
        .split("/")[2]
    assert lib.ds.count("upload_directory") == 1 and lib.leftovers() == []

    # a fresh container: an empty disk, a new store, the same dataset
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "rebuilt"))
    monkeypatch.setattr(sp, "_STORE", None)
    at2 = AppTest.from_file(app, default_timeout=30).run()
    assert not at2.exception, at2.exception
    assert not [b for b in at2.sidebar.button if str(b.key) == f"open_{tid}"]
    restore = next(b for b in at2.sidebar.button
                   if str(b.key) == f"sp_restore_{folder}")
    restore.click().run()
    assert not at2.exception, at2.exception
    assert at2.session_state["thread_id"] == tid
    back = core.load_transcript(tid)
    assert [e["role"] for e in back] == ["user", "assistant"]
    assert back[1]["text"] == "Hello world."


# ---------------------------------------------------------------------------
# Connection diagnostics
# ---------------------------------------------------------------------------

def test_the_diagnostics_line(lib, monkeypatch):
    from webapp import diagnostics as diag
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    c = diag._conversation_storage_check()
    assert c["status"] == diag.SKIP and fds.ENV_DATASET in c["detail"]
    assert "lost when it is rebuilt" in c["detail"]

    lib.ds.files_ = {f"{TOP}/conversations/A/meta.json": b"{}"}
    _foundry(monkeypatch)
    c = diag._conversation_storage_check()
    assert c["status"] == diag.PASS, c
    assert ALIAS in c["detail"] and "1 of them the app's" in c["detail"]
    assert not lib.ds.uploads()                     # reads, never writes

    _foundry(monkeypatch, "ri.foundry.main.dataset.made-up-0000")
    c = diag._conversation_storage_check()
    assert c["status"] == diag.FAIL and "resource ID" in c["detail"]
    assert "Data panel, read and write" in c["detail"]

    _foundry(monkeypatch, "made_up_missing")
    c = diag._conversation_storage_check()
    assert c["status"] == diag.FAIL and "AliasNotDefined" in c["detail"]
    assert "Data panel, read and write" in c["detail"]

    _foundry(monkeypatch)
    monkeypatch.setitem(sys.modules, "foundry.transforms", None)
    c = diag._conversation_storage_check()
    assert c["status"] == diag.FAIL and "foundry.transforms" in c["detail"]


# ---------------------------------------------------------------------------
# report ingest's durable mirror (RI-203) takes the same file manager
# ---------------------------------------------------------------------------

def test_report_ingests_mirror_fits_unchanged(lib, tmp_path):
    from funhouse_agent.run_mirror import Mirror, file_manager_of
    fm = _fm(lib)
    out = tmp_path / "run_out"
    (out / "runs").mkdir(parents=True)
    (out / "summary.json").write_text('{"n": 2}', encoding="utf-8")
    (out / "runs" / "R01.json").write_text('{"ok": true}', encoding="utf-8")
    mirror = Mirror(sharepoint=fm)
    up = mirror.mirror_dir(out, mirror.remote_for(out.name))
    assert up["uploaded"] == 2 and not up["errors"], up
    assert f"{TOP}/report_ingest/run_out/runs/R01.json" in lib.ds.files_
    assert lib.ds.count("upload_file") == 2          # no batch: one each
    back = tmp_path / "wiped"
    res = mirror.restore_dir(mirror.remote_for(out.name), back)
    assert res["downloaded"] == 2 and not res["errors"], res
    assert (back / "runs" / "R01.json").read_text(encoding="utf-8") == \
        '{"ok": true}'
    # the app's store is unwrapped to the same file manager
    assert file_manager_of(sp.SharePointStore(file_manager=fm)) is fm
