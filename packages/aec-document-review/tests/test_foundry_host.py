"""Document Review on Palantir Foundry (BACKLOG INF-74).

- The Foundry entry (:mod:`aec_document_review.foundry_entry`) marks the
  process as a Foundry deployment before the page runs, and the two-line
  workspace file starts the page.
- On Foundry with the dataset's alias set, the page's store saves the
  conversations to the Foundry dataset through the shared core's file
  manager (``aec_agent_core.foundry_dataset_store``). Until INF-74 the page
  imported ``aec_document_review.foundry_dataset_store``, which the package
  did not have: the store fell back to no permanent storage without a word,
  and Connection diagnostics' storage line raised ImportError.
- Who is viewing: nobody is named on Foundry, so the page runs as one
  shared, single-user page, as GeotechStaffEngineer's does there.

A small fake ``foundry.transforms`` stands in for the workspace dataset
library: no network, and every alias, path and name below is made up.
"""

import os
import sys
import types
from datetime import datetime, timezone

import pytest

import aec_document_review.core as core
import aec_document_review.diagnostics as diag
import aec_document_review.sharepoint_store as sp
from aec_agent_core import engine_config
from aec_agent_core import foundry_dataset_store as fds
from aec_agent_core.engine_config import EngineResolution
from aec_document_review import foundry_entry, identity, pages_entry

ALIAS = "made_up_review_conversations"

#: Env the tests set or clear, and the entry WRITES straight into
#: ``os.environ`` (``GEOTECH_DEPLOYMENT``): restored by hand afterwards,
#: since monkeypatch records no undo for a variable it deleted while absent.
_ENV = ("GEOTECH_DEPLOYMENT", fds.ENV_DATASET, sp.ENV_ROOT, sp.ENV_SITE,
        sp.ENV_CLIENT_ID, sp.ENV_CLIENT_SECRET, sp.ENV_TOKEN,
        sp.ENV_TOKEN_FILE, sp.ENV_DRIVE, "GEOTECH_APP_PROFILE",
        "DEV_IDENTITY", "GEOTECH_USER_EMAIL", "WEBSITE_SITE_NAME", "APP_ENV",
        "IDENTITY_ENDPOINT", "GEOTECH_FOUNDRY_MODELS", "ANTHROPIC_API_KEY")


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    before = {e: os.environ.get(e) for e in _ENV}
    for e in _ENV:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    monkeypatch.setattr(sp, "_graph_configured", lambda: False)
    monkeypatch.setattr(sp, "_STORE", None)
    monkeypatch.setattr(fds, "_READS", fds._Reads())   # what reads found
    yield
    fds.wait_checks(5)
    for e, value in before.items():
        if value is None:
            os.environ.pop(e, None)
        else:
            os.environ[e] = value


# ---------------------------------------------------------------------------
# a small fake of the workspace dataset library
# ---------------------------------------------------------------------------

class _File:
    def __init__(self, ds, path):
        self.path = path
        self.size_bytes = len(ds.files_[path])
        self.updated_time = datetime(2026, 1, 2, tzinfo=timezone.utc)
        self._ds = ds

    def download(self):
        dst = os.path.join(self._ds.cache, self.path.replace("/", "__"))
        with open(dst, "wb") as fh:
            fh.write(self._ds.files_[self.path])
        return dst


class _Dataset:
    """One dataset of raw files: ``upload_directory`` takes only the
    outputs folder and stores each file at its path relative to it."""

    def __init__(self, outputs, cache):
        self.outputs, self.cache = outputs, cache
        self.files_ = {}
        self.calls = []

    def files(self, show_hidden_files=False):
        self.calls.append("files")
        return [_File(self, p) for p in sorted(self.files_)]

    def upload_file(self, local, logical_path):
        self.calls.append("upload_file")
        with open(local, "rb") as fh:
            self.files_[logical_path.lstrip("/")] = fh.read()

    def upload_directory(self, local_dir_path):
        self.calls.append("upload_directory")
        assert os.path.normpath(local_dir_path) == \
            os.path.normpath(self.outputs)
        for folder, _dirs, names in os.walk(local_dir_path):
            for name in names:
                local = os.path.join(folder, name)
                rel = os.path.relpath(local, local_dir_path)
                with open(local, "rb") as fh:
                    self.files_[rel.replace(os.sep, "/")] = fh.read()


@pytest.fixture
def dataset(monkeypatch, tmp_path):
    """``foundry.transforms`` with one dataset under :data:`ALIAS`."""
    outputs = str(tmp_path / "foundry" / "outputs")
    cache = str(tmp_path / "library_cache")
    os.makedirs(cache)
    ds = _Dataset(outputs, cache)

    class Dataset:
        @staticmethod
        def get(alias):
            if alias != ALIAS:
                raise RuntimeError("404 FoundryContainerSidecar:"
                                   f"AliasNotDefined {{'alias': '{alias}'}}")
            return ds

    pkg = types.ModuleType("foundry")
    pkg.__path__ = []
    mod = types.ModuleType("foundry.transforms")
    mod.Dataset = Dataset
    pkg.transforms = mod
    monkeypatch.setitem(sys.modules, "foundry", pkg)
    monkeypatch.setitem(sys.modules, "foundry.transforms", mod)
    monkeypatch.setattr(fds, "OUTPUTS_DIR", outputs)
    return ds


def _foundry(monkeypatch, alias=ALIAS):
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    monkeypatch.setenv(fds.ENV_DATASET, alias)


# ---------------------------------------------------------------------------
# the entry
# ---------------------------------------------------------------------------

def test_the_entry_marks_foundry_before_the_page_runs(monkeypatch):
    seen = []
    monkeypatch.setattr(pages_entry, "run", lambda: seen.append(
        (os.environ.get("GEOTECH_DEPLOYMENT"),
         engine_config.is_foundry_deployment())))
    foundry_entry.main()
    assert seen == [("foundry", True)]
    assert foundry_entry.DEPLOYMENT_NAME == "foundry"
    # a value already set is kept, as GeotechStaffEngineer's entry does
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "made-up-host")
    foundry_entry.main()
    assert seen[-1] == ("made-up-host", False)


def test_the_two_line_workspace_file_starts_the_page_on_foundry(
        dataset, monkeypatch, tmp_path):
    """The file Foundry publishes, run by Streamlit: the page starts with no
    error and no API-key surface, nobody is refused, and the sidebar's
    permanent storage is Foundry storage."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    _foundry(monkeypatch)
    monkeypatch.setenv("GEOTECH_UPLOAD_MODE", "http")
    monkeypatch.setattr(engine_config, "resolve_engine",
                        lambda *a, **k: EngineResolution(
                            model=object(), source="foundry",
                            model_name="fake-model", message=""))
    monkeypatch.setattr(core, "build_agent", lambda *a, **k: object())
    at = AppTest.from_string(
        "from aec_document_review.foundry_entry import main\nmain()\n",
        default_timeout=60).run()
    assert not at.exception, at.exception
    assert os.environ["GEOTECH_DEPLOYMENT"] == "foundry"
    assert not any("could not tell who you are" in str(e.value)
                   for e in at.error)
    assert at.chat_input
    assert "Permanent storage" in [str(h.value) for h in at.sidebar.subheader]
    captions = " | ".join(str(c.value) for c in at.sidebar.caption)
    assert "Foundry storage" in captions, captions
    assert "SharePoint" not in captions, captions
    text = " | ".join(str(getattr(el, "value", "")) for kind in (
        "warning", "error", "info", "caption", "markdown")
        for el in getattr(at, kind, []))
    assert "ANTHROPIC" not in text and "API key" not in text


# ---------------------------------------------------------------------------
# the store: Foundry storage through the core's file manager
# ---------------------------------------------------------------------------

def _conversation(root, tid, title):
    core.ensure_conversation(tid, title=title, root=root)
    meta = core.load_meta(tid, root)
    meta["created"] = datetime(2026, 1, 2, 12, tzinfo=timezone.utc).timestamp()
    core.save_meta(tid, meta, root)
    conv = core.conversation_dir(tid, root)
    with open(os.path.join(conv, "transcript.jsonl"), "w",
              encoding="utf-8") as fh:
        fh.write('{"role": "user", "text": "made-up question"}\n')
    return conv


def test_on_foundry_the_store_saves_to_the_dataset(dataset, monkeypatch,
                                                   tmp_path):
    _foundry(monkeypatch)
    monkeypatch.setenv(sp.ENV_SITE, "https://example.sharepoint.test/x")
    monkeypatch.setenv(sp.ENV_TOKEN, "made-up")
    store = sp.SharePointStore()
    assert store.configured and store.backend == "foundry"
    assert store.label == "Foundry storage" and store.background
    assert store.not_kept_reason() is None          # the alias works
    assert sp.configured() is False     # the SharePoint tools stay off
    fm = store.file_manager()
    assert isinstance(fm, fds.DatasetFileManager) and fm.alias == ALIAS
    # the base folder is this application's name, handed to the core
    assert store.root() == fds.DEFAULT_ROOT == "Document Review"
    root = str(tmp_path / "container")
    _conversation(root, "T-FDY", "Made-up spec check")
    up = store.mirror_conversation("T-FDY", root=root)
    assert not up["errors"] and up["uploaded"] >= 2, up
    assert dataset.calls.count("upload_directory") == 1      # one save
    assert any(p.startswith("Document Review/conversations/")
               and p.endswith("/transcript.jsonl") for p in dataset.files_)
    # a fresh container lists the conversation and brings it back
    sp._STORE = None
    names = [r["name"] for r in sp.SharePointStore()
             .list_remote_conversations()]
    assert names == [up["folder"].rsplit("/", 1)[-1]], (names, up)


def test_off_foundry_the_dataset_setting_changes_nothing(monkeypatch):
    monkeypatch.setenv(fds.ENV_DATASET, ALIAS)
    for deployment in (None, "tinyapps"):
        if deployment:
            monkeypatch.setenv("GEOTECH_DEPLOYMENT", deployment)
        store = sp.SharePointStore()
        assert store.backend == "sharepoint" and not store.configured
        assert store.not_kept_reason() is None
        assert store.root() == sp.DEFAULT_ROOT


def test_on_foundry_without_an_alias_the_page_says_they_are_not_kept(
        monkeypatch):
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    store = sp.SharePointStore()
    assert not store.configured
    reason = store.not_kept_reason()
    assert reason and fds.ENV_DATASET in reason
    assert "lost when it is rebuilt" in reason


def test_the_store_reads_the_cores_base_folder_setting():
    assert sp.ENV_ROOT == fds.ENV_ROOT


# ---------------------------------------------------------------------------
# Connection diagnostics
# ---------------------------------------------------------------------------

def test_the_diagnostics_storage_line(dataset, monkeypatch):
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    c = diag._conversation_storage_check()
    assert c["status"] == diag.SKIP and fds.ENV_DATASET in c["detail"]
    dataset.files_ = {"Document Review/conversations/A/meta.json": b"{}"}
    _foundry(monkeypatch)
    c = diag._conversation_storage_check()
    assert c["status"] == diag.PASS, c
    assert ALIAS in c["detail"] and "1 of them the app's" in c["detail"]
    _foundry(monkeypatch, "made_up_missing")
    c = diag._conversation_storage_check()
    assert c["status"] == diag.FAIL and "AliasNotDefined" in c["detail"]


# ---------------------------------------------------------------------------
# who is viewing
# ---------------------------------------------------------------------------

def test_on_foundry_nobody_is_named_and_nobody_is_refused(monkeypatch):
    """A Foundry app is not told who is viewing (no sign-in header), and no
    App Service setting is there: the page serves everyone as the one
    unnamed user, as GeotechStaffEngineer's Foundry page does. So one
    shared folder of conversations, not one per viewer."""
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    who = identity.current_identity()
    assert who is identity.ANONYMOUS
    assert not who.refused and not who.multi_user
    from aec_document_review import profiles
    root = profiles.session_root(profiles.DOCUMENT_REVIEW, who)
    assert root == os.path.join(core.data_root(), "pages",
                                profiles.DOCUMENT_REVIEW.name)
