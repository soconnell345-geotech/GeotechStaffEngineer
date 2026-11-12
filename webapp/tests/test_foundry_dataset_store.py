"""Offline tests for the Foundry dataset store (BACKLOG INF-38) as the
geotech page uses it: through the store, the sidebar, the app and
Connection diagnostics. The module is the shared core's since INF-74
(``aec_agent_core.foundry_dataset_store``); ``webapp.foundry_dataset_store``
is an alias of it, the name these tests import it by. The file manager on
its own is tested with the core, in
``packages/aec-agent-core/tests/test_foundry_dataset_store.py``.

A fake ``foundry.transforms`` module stands in for the workspace dataset
library (``packages/aec-agent-core/tests/foundry_fake.py``, loaded from
there): no network, and every alias, path and name below is made up. The
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
* a later write overwrites a path and keeps every other file;
* a new dataset with no committed transaction answers a read with
  ``400 FoundryContainerService:FailedToResolveDataset`` ("does not have
  any committed transaction", the FDE's build i), and its first write
  makes the first commit.
"""

import importlib.util
import json
import os
import sys
import threading
import time

import pytest

import webapp.core as core
import webapp.foundry_dataset_store as fds
import webapp.sharepoint_store as sp
from aec_agent_core import foundry_dataset_store as core_fds


def _core_fake():
    """The fake library, kept beside the core's tests with the module it
    fakes (``packages/aec-agent-core/tests/foundry_fake.py``)."""
    if "foundry_fake" in sys.modules:
        return sys.modules["foundry_fake"]
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "packages", "aec-agent-core", "tests",
        "foundry_fake.py")
    spec = importlib.util.spec_from_file_location("foundry_fake", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["foundry_fake"] = module
    spec.loader.exec_module(module)
    return module


_fake = _core_fake()
FakeLibrary = _fake.FakeLibrary
SidecarError = _fake.SidecarError
NO_COMMIT = _fake.NO_COMMIT
NOT_RESOLVED = _fake.NOT_RESOLVED

ALIAS = _fake.ALIAS
#: GeotechStaffEngineer's folder in the dataset, as before the move.
TOP = "GeotechStaffEngineer"

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
    monkeypatch.setattr(fds, "_READS", fds._Reads())   # what reads found
    yield
    fds.wait_checks(5)


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
# the module moved into the shared core (INF-74)
# ---------------------------------------------------------------------------

def test_the_old_name_is_the_cores_module():
    """``webapp.foundry_dataset_store`` IS the core's module (an alias, as
    for every earlier move), the geotech page keeps its folder in the
    dataset, and its store reads the same base-folder setting as the core."""
    import webapp
    assert fds is core_fds
    assert webapp.CORE_ALIASES["webapp.foundry_dataset_store"] == \
        "aec_agent_core.foundry_dataset_store"
    assert webapp.foundry_dataset_store is core_fds
    from webapp.foundry_dataset_store import DatasetFileManager
    assert DatasetFileManager is core_fds.DatasetFileManager
    assert core_fds.__file__.replace(os.sep, "/").endswith(
        "aec_agent_core/foundry_dataset_store.py")
    assert fds.DEFAULT_ROOT == TOP
    assert sp.ENV_ROOT == core_fds.ENV_ROOT


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
    next(b for b in at.sidebar.button
         if str(b.key) == "sp_restore_load").click().run()
    assert not at.exception, at.exception
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
                                  "library missing", "alias not defined",
                                  "dataset not resolved"])
def test_on_foundry_the_sidebar_says_when_conversations_are_not_kept(
        lib, monkeypatch, tmp_path, case):
    """On master the sidebar drew no storage block when the dataset was not
    usable, and only Connection diagnostics said why: the owner would take
    conversations for kept. Now a short block says they are NOT kept, with
    the diagnostics line's own reason and fix. "dataset not resolved" is
    ``FailedToResolveDataset`` for a cause other than a new, empty dataset
    (INF-65): it still fails, and the sidebar never says storage is on."""
    from webapp import diagnostics as diag
    if case == "resource ID":
        monkeypatch.setenv(fds.ENV_DATASET, "ri.foundry.main.dataset.made-up-0")
    elif case == "library missing":
        monkeypatch.setenv(fds.ENV_DATASET, ALIAS)
        monkeypatch.setitem(sys.modules, "foundry.transforms", None)
    elif case == "alias not defined":
        monkeypatch.setenv(fds.ENV_DATASET, "made_up_missing")
    elif case == "dataset not resolved":
        monkeypatch.setenv(fds.ENV_DATASET, ALIAS)
        lib.ds.files_error = NOT_RESOLVED
    at = _foundry_app(monkeypatch, tmp_path)
    fds.wait_checks(5)                         # the sidebar's one check
    at.run()
    assert not at.exception, at.exception
    detail = diag._conversation_storage_check()["detail"]
    warnings = [str(w.value) for w in at.sidebar.warning]
    assert ("**Conversations are NOT kept across restarts.** " + detail) \
        in warnings, (warnings, detail)
    assert not any(str(c.value).startswith("Foundry storage is on")
                   for c in at.sidebar.caption)
    if case == "no alias":
        assert fds.ENV_DATASET in detail and "Download this conversation" \
            in detail
    elif case == "dataset not resolved":
        assert "could not be read" in detail and \
            "FailedToResolveDataset" in detail, detail
    else:
        assert "Data panel, read and write" in detail or \
            "Code Workspace" in detail, detail


@pytest.mark.parametrize("new_dataset", [False, True])
def test_the_not_kept_block_is_foundry_only_and_absent_when_kept(
        lib, monkeypatch, tmp_path, new_dataset):
    """``new_dataset``: a dataset with nothing committed yet is kept too
    (INF-65): the first save makes its first commit."""
    from webapp import diagnostics as diag
    lib.ds.uncommitted = new_dataset
    monkeypatch.setenv(fds.ENV_DATASET, ALIAS)              # the alias works
    at = _foundry_app(monkeypatch, tmp_path / "kept")
    fds.wait_checks(5)
    at.run()
    assert not at.exception, at.exception
    assert not any("NOT kept" in str(w.value) for w in at.sidebar.warning)
    assert any(str(c.value).startswith("Foundry storage is on")
               for c in at.sidebar.caption)
    c = diag._conversation_storage_check()
    assert c["status"] == diag.PASS, c
    assert ("readable, 0 files: a new dataset" in c["detail"]) == new_dataset
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
    store.not_kept_reason()                     # starts the one check
    assert fds.wait_checks(5)
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

    # a fresh container: an empty disk, a new store, the same dataset (the
    # same process here, so the first container's check stands; a real
    # one makes one listing in the background, INF-65)
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "rebuilt"))
    monkeypatch.setattr(sp, "_STORE", None)
    assert fds.wait_checks(5)
    listed = lib.ds.count("files")
    at2 = AppTest.from_file(app, default_timeout=30).run()
    assert not at2.exception, at2.exception
    assert not [b for b in at2.sidebar.button if str(b.key) == f"open_{tid}"]
    # the first draw does not wait on a listing: it is read when asked for
    assert lib.ds.count("files") == listed
    assert not [b for b in at2.sidebar.button
                if str(b.key) == f"sp_restore_{folder}"]
    next(b for b in at2.sidebar.button
         if str(b.key) == "sp_restore_load").click().run()
    assert not at2.exception, at2.exception
    assert lib.ds.count("files") == listed + 1
    at2.run()                                  # cached for the session
    assert lib.ds.count("files") == listed + 1
    restore = next(b for b in at2.sidebar.button
                   if str(b.key) == f"sp_restore_{folder}")
    restore.click().run()
    assert not at2.exception, at2.exception
    assert at2.session_state["thread_id"] == tid
    back = core.load_transcript(tid)
    assert [e["role"] for e in back] == ["user", "assistant"]
    assert back[1]["text"] == "Hello world."


# ---------------------------------------------------------------------------
# a new dataset with nothing committed; the sidebar and diagnostics agree;
# the save line updates by itself (INF-65)
# ---------------------------------------------------------------------------

def test_a_new_dataset_reads_as_empty_and_its_first_save_commits(
        lib, monkeypatch, tmp_path):
    """Build i: a new raw-files dataset refused every read with
    ``FailedToResolveDataset`` until the owner uploaded a placeholder file.
    Now it is an empty store: nothing listed, diagnostics PASS, and the
    first save makes the first commit, under the plain folder name (an
    empty listing is not a failed one, so no shard is added)."""
    from webapp import diagnostics as diag
    lib.ds.uncommitted = True
    fm = _fm(lib)
    assert fm.ls(f"{TOP}/conversations") == []
    assert fm.probe().startswith("readable, 0 files: a new dataset")
    _foundry(monkeypatch)
    c = diag._conversation_storage_check()
    assert c["status"] == diag.PASS and "readable, 0 files" in c["detail"], c
    store = _store(lib)
    assert store.list_remote_conversations() == []
    root = str(tmp_path)
    _conversation(root, "T-NEW", "Footing check", _CREATED)
    up = store.mirror_conversation("T-NEW", root=root)
    assert not up["errors"] and up["uploaded"] == 4, up
    assert up["folder"] == f"{TOP}/conversations/Footing_check_2026-01-02"
    assert lib.ds.count("upload_directory") == 1 and not lib.ds.uncommitted
    assert fm.probe() == "readable, 4 file(s), 4 of them the app's"
    assert [r["name"] for r in store.list_remote_conversations()] == [
        "Footing_check_2026-01-02"]


def test_when_dataset_get_itself_refuses_a_new_dataset(lib, tmp_path):
    """Which call refuses a new dataset (``Dataset.get`` or ``files()``) is
    unconfirmed: either way the listing is empty. A save then fails
    plainly, as before; the app tries no other way to make the first
    commit (module_work/inf38/CHECK.md, a check for the FDE)."""
    lib.ds.get_error = NO_COMMIT
    store = _store(lib)
    fm = store.file_manager()
    assert fm.ls(TOP) == [] and fm.probe().startswith("readable, 0 files")
    root = str(tmp_path)
    _conversation(root, "T-GET", "Pile cap", _CREATED)
    up = store.mirror_conversation("T-GET", root=root)
    assert up["uploaded"] == 0 and len(up["errors"]) == 1, up
    assert "FailedToResolveDataset" in up["errors"][0]
    assert lib.leftovers() == []


def test_a_failed_read_reaches_the_sidebar_and_a_good_one_clears_it(
        lib, monkeypatch):
    """The sidebar and Connection diagnostics read one record: a probe
    that fails after the sidebar's check passed turns the sidebar to NOT
    kept, in the diagnostics line's own words; a read that works clears
    it."""
    from webapp import diagnostics as diag
    _foundry(monkeypatch)
    store = sp.SharePointStore()
    assert store.not_kept_reason() is None          # starts the one check
    assert fds.wait_checks(5)
    assert store.check_state() is None and store.not_kept_reason() is None
    lib.ds.files_error = NOT_RESOLVED
    c = diag._conversation_storage_check()
    assert c["status"] == diag.FAIL, c
    assert store.not_kept_reason() == c["detail"]
    lib.ds.files_error = None
    assert diag._conversation_storage_check()["status"] == diag.PASS
    assert store.not_kept_reason() is None
    assert lib.ds.count("files") == 3               # the check, two probes


def test_the_sidebar_says_checking_until_the_first_read_ends(
        lib, monkeypatch, tmp_path):
    """The sidebar's first look starts one listing in the background, and
    the page does not wait for it: it says it is checking, never that
    storage is on, until a read has worked."""
    lib.ds.files_gate = threading.Event()           # the listing hangs
    monkeypatch.setenv(fds.ENV_DATASET, ALIAS)
    at = _foundry_app(monkeypatch, tmp_path)
    captions = [str(c.value) for c in at.sidebar.caption]
    assert "Foundry storage: checking the conversation dataset…" in \
        captions, captions
    assert not any(c.startswith("Foundry storage is on") for c in captions)
    lib.ds.files_gate.set()
    assert fds.wait_checks(5)
    at.run()
    assert not at.exception, at.exception
    captions = [str(c.value) for c in at.sidebar.caption]
    assert any(c.startswith("Foundry storage is on") for c in captions), \
        captions
    assert not any("checking" in c for c in captions)
    assert lib.ds.count("files") == 1               # once per process


def test_the_save_line_updates_by_itself_when_a_background_save_ends(
        monkeypatch, tmp_path):
    """Build i: after the turn the FDE saw no save line; the result showed
    only on the next click. While a save runs the status lines are a
    fragment that looks again every few seconds; when the save ends it
    reruns the page once, which shows "saved N file(s)" and stops looking
    (no fragment on that run)."""
    st = pytest.importorskip("streamlit")
    fragments = []
    real = st.fragment

    def recording(func=None, *, run_every=None, **kw):
        fragments.append((getattr(func, "__name__", ""), run_every))
        return real(func, run_every=run_every, **kw)
    monkeypatch.setattr(st, "fragment", recording)

    class _Ending(_SidebarStore):
        """A save that ends while the fragment looks: saving on two looks,
        the second a timed run's (a poll or more after the page drew the
        fragment; AppTest runs no timer, so the look waits that long)."""

        looks = 0

        def save_state(self, thread_id):
            self.looks += 1
            if self.looks == 2:
                time.sleep(2.0 / 2 + 0.05)
            return "saving" if self.looks <= 2 else None

        def last_sync_for(self, thread_id):
            if self.looks <= 2:
                return None
            return {"uploaded": 5, "skipped": 0, "errors": [],
                    "duration_s": 6.8, "at": time.time(),
                    "thread_id": thread_id}

    store = _Ending()
    at = _sidebar(monkeypatch, tmp_path, store)
    captions = [str(c.value) for c in at.sidebar.caption]
    assert "Foundry storage: saved 5 file(s) just now, 0 up-to-date · 6.8s" \
        in captions, captions
    assert not any("saving the latest turn" in c for c in captions)
    polled = [f for f in fragments if f[0] == "_storage_status"]
    assert polled == [("_storage_status", 2.0)], fragments
    assert store.looks == 5                     # the rerun looked once more


def test_a_click_or_a_message_in_the_run_where_a_save_ends_is_kept(
        monkeypatch, tmp_path):
    """The fragment's first draw is part of the page's own run. A rerun
    there would end that run before the chat box and the buttons below,
    and Streamlit drops a message sent or a click made in it; so only a
    timed run reruns the page."""

    class _Switch(_SidebarStore):
        """No save until ``armed``; then saving on two looks (the page's
        busy check and the fragment's first draw), then done."""

        armed = False
        looks = 0
        synced = 0

        def save_state(self, thread_id):
            if not self.armed:
                return None
            self.looks += 1
            return "saving" if self.looks <= 2 else None

        def mirror_conversation(self, thread_id, root=None):
            self.synced += 1
            return super().mirror_conversation(thread_id, root)

    def answer(_agent, _messages, _thread_id, **_kw):
        yield {"kind": "turn_done", "answer": "Made-up answer.",
               "turn_tokens": 1}

    monkeypatch.setattr(core, "stream_turn", answer)
    store = _Switch()
    at = _sidebar(monkeypatch, tmp_path, store)
    store.armed = True
    next(b for b in at.sidebar.button
         if str(b.key).startswith("spsync_")).click().run()
    assert not at.exception, at.exception
    assert store.synced == 1                    # Sync now ran
    store.looks = 0
    at.chat_input[0].set_value("made-up question").run()
    assert not at.exception, at.exception
    assert [e["text"] for e in at.session_state["transcript"]
            if e["role"] == "user"] == ["made-up question"]


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
