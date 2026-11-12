"""Offline tests for the Foundry dataset store under its core name,
:mod:`aec_agent_core.foundry_dataset_store` (BACKLOG INF-38; moved into the
shared core from GeotechStaffEngineer's ``webapp``, INF-74): the file
manager on its own, its settings, and the base folder named after the
application that handed its name in. They need only the core (CI's
core-only job runs them). The store, the sidebar and Connection diagnostics
on top of it are tested with each application:
``webapp/tests/test_foundry_dataset_store.py`` in GeotechStaffEngineer and
``tests/test_foundry_host.py`` in aec-document-review.

A fake ``foundry.transforms`` (``foundry_fake.py`` beside this file) stands
in for the workspace dataset library: no network, and every alias, path and
name below is made up.
"""

import os
import sys

import pytest

import aec_agent_core.foundry_dataset_store as fds
from aec_agent_core import app_details
from aec_agent_core.hooks import clear_hook, hooks, set_hook
from foundry_fake import (ALIAS, NO_COMMIT, NOT_RESOLVED, FakeLibrary,
                          IllegalArgumentError, SidecarError)
from foundry_fake import write as _write

#: The made-up application these tests hand in; the dataset's base folder
#: is named after it.
APP = "Made-up App"
TOP = APP

_ENVS = (fds.ENV_DATASET, fds.ENV_ROOT, "GEOTECH_DEPLOYMENT")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for e in _ENVS:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setattr(fds.DatasetFileManager, "_warned_foreign", False)
    monkeypatch.setattr(fds, "_READS", fds._Reads())   # what reads found
    before = hooks().get(app_details.HOOK)
    app_details.hand_in(name=APP)
    yield
    if before is None:
        clear_hook(app_details.HOOK)
    else:
        set_hook(app_details.HOOK, before)
    fds.wait_checks(5)


@pytest.fixture
def lib(monkeypatch, tmp_path):
    return FakeLibrary(tmp_path).install(monkeypatch)


def _fm(lib, pauses=None, **kw):
    sleep = pauses.append if pauses is not None else (lambda s: None)
    return fds.DatasetFileManager(ALIAS, outputs_dir=lib.outputs,
                                  sleep=sleep, **kw)


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


def test_the_no_commit_error_is_matched_by_name_and_wording():
    assert fds._is_no_commit(SidecarError(NO_COMMIT))
    assert not fds._is_no_commit(SidecarError(NOT_RESOLVED))   # another cause
    assert not fds._is_no_commit(SidecarError(              # another error name
        "400 Catalog:MadeUpError {'message': 'The dataset does not have any "
        "committed transaction'}"))
    try:
        try:
            raise SidecarError(NO_COMMIT)
        except SidecarError as inner:
            raise RuntimeError("made-up wrapper") from inner
    except RuntimeError as outer:
        assert fds._is_no_commit(outer)                        # wrapped


# ---------------------------------------------------------------------------
# the core names no application; the settings, without a store (INF-74)
# ---------------------------------------------------------------------------

def test_the_base_folder_is_the_name_the_application_handed_in(lib,
                                                               monkeypatch,
                                                               tmp_path):
    """No application is named in the core: the dataset's base folder is
    the handed-in name (GeotechStaffEngineer's stays "GeotechStaffEngineer",
    Document Review's is "Document Review"), ``GEOTECH_SHAREPOINT_ROOT``
    overrides it, and a save stages under the folder's first segment."""
    assert fds.DEFAULT_ROOT == APP and fds._app_top() == APP
    app_details.hand_in(name="Another Made-up Tool")
    assert fds.DEFAULT_ROOT == "Another Made-up Tool"
    clear_hook(app_details.HOOK)
    assert fds.DEFAULT_ROOT == app_details.DEFAULTS["name"]
    app_details.hand_in(name=APP)
    assert fds.ENV_ROOT == "GEOTECH_SHAREPOINT_ROOT"
    monkeypatch.setenv(fds.ENV_ROOT, "/Made_up_folder/sub/")
    assert fds._app_top() == "Made_up_folder"
    fm = _fm(lib)
    fm.begin_batch()
    fm.upload_file(_write(str(tmp_path / "a.txt")),
                   "Made_up_folder/sub/c/a.txt")
    assert fm.end_batch() == 1
    assert lib.ds.uploads() == [("upload_directory",
                                 ["Made_up_folder/sub/c/a.txt"])]
    assert fm.probe() == "readable, 1 file(s), 1 of them the app's"
    with pytest.raises(AttributeError):
        fds.NOT_A_SETTING


def test_on_foundry_with_an_alias_the_dataset_is_selected(lib, monkeypatch):
    assert not fds.selected() and not fds.configured()
    assert fds.unconfigured_reason() == "this is not a Foundry deployment"
    assert fds.not_kept_reason() is None             # off Foundry: nothing
    monkeypatch.setenv(fds.ENV_DATASET, ALIAS)
    assert not fds.selected()                        # the alias alone: no
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    assert fds.dataset_alias() == ALIAS
    assert fds.selected() and fds.configured()
    assert fds.unconfigured_reason() is None
    fm = fds.DatasetFileManager(outputs_dir=lib.outputs)
    assert fm.alias == ALIAS and fm.BACKEND == "foundry"
    assert fds.not_kept_reason(fm) is None           # no read failed yet
    assert fds.wait_checks(5)
    assert fds.not_kept_reason(fm) is None           # the read worked


def test_without_an_alias_or_with_a_resource_id_or_no_library(monkeypatch):
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    assert not fds.selected()
    reason = fds.not_kept_reason()
    assert fds.ENV_DATASET in reason and "lost when it is rebuilt" in reason
    with pytest.raises(RuntimeError):
        fds.DatasetFileManager()
    monkeypatch.setenv(fds.ENV_DATASET, "ri.foundry.main.dataset.made-up-0")
    assert fds.selected() and not fds.configured()
    assert "resource ID" in fds.unconfigured_reason()
    monkeypatch.setenv(fds.ENV_DATASET, ALIAS)
    monkeypatch.setitem(sys.modules, "foundry", None)      # import fails
    monkeypatch.setitem(sys.modules, "foundry.transforms", None)
    assert not fds.library_installed() and not fds.configured()
    assert "foundry.transforms" in fds.unconfigured_reason()
    assert fds.not_kept_reason().startswith(f"{fds.ENV_DATASET} is set, but ")
